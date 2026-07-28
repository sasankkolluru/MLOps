"""
================================================================================
prediction/predict.py
================================================================================
Scores brand-new, raw student records with the trained stacked ensemble
(training/train_model.py). "Raw" means the same columns and format as
data/raw/student_employability_dataset.csv (minus the Employable column,
which is what we're predicting).

PIPELINE (raw input -> prediction):
    0. version check            -> preprocessing artifacts and model artifacts
                                    are from a compatible training run
    1. unknown-column check     -> unexpected columns are reported and dropped,
                                    not silently ignored
    2. duplicate check          -> repeated Student_IDs / fully duplicate rows
                                    are flagged in the output, not silently scored
    3. data-type validation     -> every value is coerced to numeric; anything
                                    that fails is reported and treated as missing
    4. missing-value imputation -> filled with the TRAINING set's saved
                                    median (numeric) / mode (binary), never 0
    5. range validation         -> out-of-range values (CGPA outside 0-10, etc.)
                                    are clipped to valid bounds and reported
    6. binary cleaning          -> robust normalisation (handles 0/1, True/False,
                                    "yes"/"no", stray floats), not just `> 0`
    7. outlier capping          -> TRAINING set's saved IQR bounds
    8. skew correction          -> TRAINING set's saved log1p offsets
    9. feature engineering      -> Total_Tech_Skills, Avg_Soft_Skill_Score, Experience_Score
   10. scaling                  -> TRAINING set's fitted StandardScaler
   11. scoring                  -> EBM / NGBoost / DeepForest / LCE (calibrated)
                                    -> Logistic Regression meta-learner -> optimised threshold
   12. probability explanation  -> which base learner(s) drove the decision
   13. skill gap identification -> for students predicted Not Employable, which
                                    features they trail the Employable group on,
                                    ranked by the model's actual permutation
                                    importance (not just any raw gap), reported
                                    in REAL-WORLD units (CGPA points, raw skill
                                    scores, % of peers with a tool) -- not scaled
                                    z-score differences.

USAGE
-----
As a script, scoring a CSV of new students:
    python predict.py --input path/to/new_students.csv --output path/to/predictions.csv

As a library, scoring a single student:
    from predict import predict_single
    result = predict_single({
        "CGPA": 8.2, "Communication": 7, "Aptitude": 6, "Problem_Solving": 8,
        "Teamwork": 7, "Projects": 3, "Internships": 1, "Certifications": 2,
        "Java": 1, "Python": 1, "C++": 0, "SQL": 1, "DSA": 1, "OOP": 1,
        "Git": 1, "Spring_Boot": 0, "React": 0, "AWS": 0,
    })
================================================================================
"""

import os
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "train_preprocessed.csv")

KNOWN_ID_COLS = ["Student_ID"]

# Default valid ranges, used only if preprocessing_artifacts.pkl was built
# before valid_ranges existed / failed to save it (Issue #3).
DEFAULT_RANGES = {
    "CGPA": (0, 10),
    "Communication": (0, 10),
    "Aptitude": (0, 10),
    "Problem_Solving": (0, 10),
    "Teamwork": (0, 10),
    "Projects": (0, None),
    "Internships": (0, None),
    "Certifications": (0, None),
}


def warn(msg):
    print(f"  [warn] {msg}")


def info(msg):
    print(f"  [info] {msg}")


# --------------------------------------------------------------------------- #
# 0. ARTIFACT LOADING + VERSION CHECKING  ("No Version Checking")
# --------------------------------------------------------------------------- #
def load_artifacts():
    required = ["preprocessing_artifacts.pkl", "base_learners.pkl", "meta_learner.pkl",
                "feature_columns.pkl", "optimal_threshold.pkl"]
    missing = [f for f in required if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        raise FileNotFoundError(
            f"Missing required artifact(s) in {MODELS_DIR}: {missing}. "
            "Run preprocessing/preprocess.py and training/train_model.py first."
        )

    prep = joblib.load(os.path.join(MODELS_DIR, "preprocessing_artifacts.pkl"))
    base_learners = joblib.load(os.path.join(MODELS_DIR, "base_learners.pkl"))
    meta_model = joblib.load(os.path.join(MODELS_DIR, "meta_learner.pkl"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    threshold = joblib.load(os.path.join(MODELS_DIR, "optimal_threshold.pkl"))

    metadata_path = os.path.join(MODELS_DIR, "model_metadata.pkl")
    model_metadata = joblib.load(metadata_path) if os.path.exists(metadata_path) else None

    # --- Compatibility check: was the model trained on features this
    #     preprocessing pipeline actually produces? Catches stale artifacts
    #     from a mismatched preprocess.py / train_model.py run. --- #
    missing_features = set(feature_columns) - set(prep["raw_feature_columns"])
    if missing_features:
        raise ValueError(
            f"Version mismatch: the trained model expects feature(s) {sorted(missing_features)} "
            "that the current preprocessing pipeline does not produce. "
            "Re-run preprocessing/preprocess.py and training/train_model.py together, "
            "in order, so both artifacts come from the same pipeline version."
        )

    if "valid_ranges" not in prep:
        warn("preprocessing_artifacts.pkl has no saved 'valid_ranges' (older artifact) "
             "-> falling back to DEFAULT_RANGES for range validation.")
    prep["valid_ranges"] = prep.get("valid_ranges", DEFAULT_RANGES)

    # --- Informational version/environment reporting --- #
    prep_version = prep.get("pipeline_version", "unknown")
    prep_sklearn = prep.get("sklearn_version", "unknown")
    if model_metadata:
        model_version = model_metadata.get("model_version", "unknown")
        model_sklearn = model_metadata.get("sklearn_version", "unknown")
        trained_at = model_metadata.get("trained_at", "unknown")
        info(f"preprocessing pipeline v{prep_version} (created {prep.get('artifact_created_at', '?')}) "
             f"+ model v{model_version} (trained {trained_at}).")
        if prep_sklearn != model_sklearn:
            warn(f"scikit-learn version mismatch between preprocessing ({prep_sklearn}) "
                 f"and training ({model_sklearn}) -- pickled estimators can behave "
                 f"inconsistently across versions. Consider retraining in a matching environment.")
    else:
        warn("model_metadata.pkl not found (older training run) -> skipping model-side version check. "
             "Re-run training/train_model.py to enable it.")

    try:
        import sklearn
        current_sklearn = sklearn.__version__
        if current_sklearn != prep_sklearn:
            warn(f"Running scikit-learn {current_sklearn}, but artifacts were built with "
                 f"{prep_sklearn}. Pickled models may not load/predict identically.")
    except Exception:
        pass

    return prep, base_learners, meta_model, feature_columns, threshold, model_metadata


# --------------------------------------------------------------------------- #
# 1. UNKNOWN-COLUMN CHECK  ("Unknown Columns Ignored")
# --------------------------------------------------------------------------- #
def check_unknown_columns(df, prep):
    expected_inputs = set(prep["skill_cols"]) | set(prep["soft_skill_cols"]) | \
        {"CGPA", "Projects", "Internships", "Certifications"} | set(KNOWN_ID_COLS)
    unknown = [c for c in df.columns if c not in expected_inputs]
    if unknown:
        warn(f"Unexpected column(s) found and will be IGNORED (not silently dropped without "
             f"telling you): {unknown}. Check for typos or a mismatched schema.")
        df = df.drop(columns=unknown)
    return df


# --------------------------------------------------------------------------- #
# 2. DUPLICATE CHECK  ("Doesn't Check Duplicate Students")
# --------------------------------------------------------------------------- #
def flag_duplicates(df):
    dup_id_mask = pd.Series(False, index=df.index)
    if "Student_ID" in df.columns:
        dup_id_mask = df["Student_ID"].duplicated(keep=False) & df["Student_ID"].notna()
        n_dup_ids = int(dup_id_mask.sum())
        if n_dup_ids:
            warn(f"{n_dup_ids} row(s) share a duplicate Student_ID: "
                 f"{sorted(df.loc[dup_id_mask, 'Student_ID'].unique().tolist())}. "
                 "Each row is still scored independently -- check your source data.")

    feature_cols = [c for c in df.columns if c not in KNOWN_ID_COLS]
    dup_row_mask = df.duplicated(subset=feature_cols, keep=False)
    n_dup_rows = int(dup_row_mask.sum())
    if n_dup_rows:
        warn(f"{n_dup_rows} row(s) are exact feature-level duplicates of another row "
             "(same answers, different or missing Student_ID). Flagged in output, not dropped.")

    return dup_id_mask, dup_row_mask


# --------------------------------------------------------------------------- #
# 3 + 4. DATA-TYPE VALIDATION + MISSING-VALUE IMPUTATION
# --------------------------------------------------------------------------- #
def cast_and_impute(df, prep):
    numeric_raw_cols = ["CGPA", "Communication", "Aptitude", "Problem_Solving",
                         "Teamwork", "Projects", "Internships", "Certifications"]
    binary_cols = prep["binary_cols"]

    string_map = {"1": 1, "0": 0, "yes": 1, "no": 0, "y": 1, "n": 0, "true": 1, "false": 0}
    for c in binary_cols:
        if c not in df.columns:
            continue
        col = df[c]
        if not pd.api.types.is_numeric_dtype(col):
            mapped = col.astype(str).str.strip().str.lower().map(string_map)
            col = mapped.where(mapped.notna(), col)
        col = pd.to_numeric(col, errors="coerce")
        n_bad = int((~col.isin([0, 1]) & col.notna()).sum())
        if n_bad:
            info(f"'{c}': {n_bad} out-of-range value(s) normalised to 0/1 by sign "
                 "(e.g. 2 -> 1, -3 -> 0).")
        col = col.apply(lambda v: (1 if v > 0 else 0) if pd.notna(v) and v not in (0, 1) else v)
        df[c] = col

    total_coerced = 0
    for c in numeric_raw_cols:
        if c not in df.columns:
            continue
        before_na = df[c].isna().sum()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        after_na = df[c].isna().sum()
        newly_bad = after_na - before_na
        if newly_bad > 0:
            total_coerced += newly_bad
            info(f"'{c}': {newly_bad} non-numeric value(s) could not be parsed -> treated as missing.")
    if total_coerced:
        warn(f"{total_coerced} total value(s) across numeric columns failed type validation "
             "and were treated as missing (see imputation below).")

    n_missing_total = 0
    for c in numeric_raw_cols:
        if c in df.columns and df[c].isna().any():
            n = int(df[c].isna().sum())
            n_missing_total += n
            fill_val = prep["raw_numeric_medians"].get(c)
            if fill_val is None:
                warn(f"No saved training median for '{c}' -> cannot impute; rows will fail downstream.")
                continue
            df[c] = df[c].fillna(fill_val)
            info(f"'{c}': filled {n} missing value(s) with training median = {fill_val:.2f}.")
    for c in binary_cols:
        if c in df.columns and df[c].isna().any():
            n = int(df[c].isna().sum())
            n_missing_total += n
            fill_val = prep["raw_binary_modes"].get(c)
            if fill_val is None:
                warn(f"No saved training mode for '{c}' -> cannot impute; rows will fail downstream.")
                continue
            df[c] = df[c].fillna(fill_val)
            info(f"'{c}': filled {n} missing value(s) with training mode = {fill_val}.")
    if n_missing_total == 0:
        info("No missing values found.")

    return df


# --------------------------------------------------------------------------- #
# 5. RANGE VALIDATION  ("No Range Validation")
# --------------------------------------------------------------------------- #
def validate_ranges(df, prep):
    for c, (lo, hi) in prep["valid_ranges"].items():
        if c not in df.columns:
            continue
        mask_low = df[c] < lo if lo is not None else pd.Series(False, index=df.index)
        mask_high = df[c] > hi if hi is not None else pd.Series(False, index=df.index)
        n_out = int((mask_low | mask_high).sum())
        if n_out:
            warn(f"'{c}': {n_out} value(s) outside the valid range "
                 f"[{lo if lo is not None else '-inf'}, {hi if hi is not None else 'inf'}] "
                 "-> clipped to the nearest valid bound.")
            if lo is not None:
                df[c] = df[c].clip(lower=lo)
            if hi is not None:
                df[c] = df[c].clip(upper=hi)
    return df


# --------------------------------------------------------------------------- #
# 6-10. OUTLIER CAPPING / SKEW / FEATURE ENGINEERING / SCALING
# --------------------------------------------------------------------------- #
def apply_outlier_bounds(df, outlier_bounds):
    for c, (lower, upper) in outlier_bounds.items():
        if c in df.columns:
            df[c] = df[c].clip(lower=lower, upper=upper)
    return df


def apply_skew_transform(df, skew_transform_info):
    for c, train_min in skew_transform_info.items():
        if c in df.columns:
            df[c] = np.log1p((df[c] - train_min).clip(lower=0))
    return df


def engineer_features(df, skill_cols, soft_skill_cols):
    df["Total_Tech_Skills"] = df[skill_cols].sum(axis=1)
    df["Avg_Soft_Skill_Score"] = df[soft_skill_cols].mean(axis=1)
    df["Experience_Score"] = df["Projects"] + df["Internships"] + df["Certifications"]
    return df


def preprocess_new_data(raw_df, prep):
    df = raw_df.copy()

    required_cols = set(prep["skill_cols"]) | set(prep["soft_skill_cols"]) | \
        {"CGPA", "Projects", "Internships", "Certifications"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input data is missing required column(s): {sorted(missing_cols)}")

    df = check_unknown_columns(df, prep)
    dup_id_mask, dup_row_mask = flag_duplicates(df)
    df = cast_and_impute(df, prep)
    df = validate_ranges(df, prep)

    # Keep a copy of the fully-cleaned, pre-scaling row values (real units,
    # already imputed/clipped/range-validated). This is what skill-gap
    # reporting compares against -- NOT the scaled model input -- so gaps
    # come out in the same units a counselor already understands.
    df_real_units = df.copy()

    df = apply_outlier_bounds(df, prep["outlier_bounds"])
    df = apply_skew_transform(df, prep["skew_transform_info"])
    df = engineer_features(df, prep["skill_cols"], prep["soft_skill_cols"])
    df_real_units = engineer_features(df_real_units, prep["skill_cols"], prep["soft_skill_cols"])

    for c in prep["raw_feature_columns"]:
        if c not in df.columns:
            raise ValueError(f"Input data is missing engineered/raw column: '{c}'")
    df = df[prep["raw_feature_columns"]]
    df_real_units = df_real_units[[c for c in prep["raw_feature_columns"] if c in df_real_units.columns]]
    df[prep["numeric_cols"]] = prep["scaler"].transform(df[prep["numeric_cols"]])

    return df, df_real_units, dup_id_mask, dup_row_mask


# --------------------------------------------------------------------------- #
# 11. SCORING
# --------------------------------------------------------------------------- #
def score(df_model_ready, base_learners, meta_model, feature_columns, threshold):
    X = df_model_ready[feature_columns]
    meta_features = pd.DataFrame(index=X.index)
    for name, model in base_learners.items():
        meta_features[f"{name}_proba"] = model.predict_proba(X)[:, 1]

    proba = meta_model.predict_proba(meta_features)[:, 1]
    pred = (proba >= threshold).astype(int)
    label = np.where(pred == 1, "Employable", "Not Employable")
    return proba, pred, label, meta_features


# --------------------------------------------------------------------------- #
# 12. PROBABILITY EXPLANATION  ("No Probability Explanation")
# --------------------------------------------------------------------------- #
def explain_predictions(meta_features, meta_model):
    """For each row, ranks base learners by how much their OWN opinion
    (above/below 0.5, weighted by how much the meta-learner trusts that
    learner) shaped the final call."""
    coefs = dict(zip(meta_features.columns, meta_model.coef_[0]))
    explanations = []
    for _, row in meta_features.iterrows():
        scored = []
        for col in meta_features.columns:
            proba = row[col]
            deviation = proba - 0.5
            influence = abs(coefs[col] * deviation)
            lean = "Employable" if deviation > 0 else "Not Employable"
            scored.append((col, influence, lean, proba))
        scored.sort(key=lambda t: t[1], reverse=True)
        parts = [f"{col.replace('_proba', '')} (p={proba:.2f}, leans {lean})"
                 for col, influence, lean, proba in scored[:2]]
        explanations.append("; ".join(parts))
    return explanations


# --------------------------------------------------------------------------- #
# 13. SKILL GAP IDENTIFICATION  -- now reported in REAL-WORLD units
# --------------------------------------------------------------------------- #
def _unscale_row(scaled_row, prep):
    """Inverts the numeric preprocessing pipeline for one row (a pandas
    Series indexed by feature name) so values are back in real-world units:
      1. undo StandardScaler on prep['numeric_cols']
      2. undo the log1p skew correction on any column that had one
    NOTE: outlier-capping (IQR clip) is not inverted. A value that was
    clipped during training reports as the capped bound rather than the
    original extreme -- a deliberate, minor approximation that does not
    change which features show up as gaps or in what order.
    """
    numeric_cols = prep["numeric_cols"]
    unscaled_vals = prep["scaler"].inverse_transform(
        scaled_row[numeric_cols].values.reshape(1, -1)
    )[0]
    real = scaled_row.copy()
    real[numeric_cols] = unscaled_vals

    for c, train_min in prep.get("skew_transform_info", {}).items():
        if c in real.index:
            real[c] = np.expm1(real[c]) + train_min
    return real


def _load_employable_reference(feature_columns, prep):
    """Real-world-unit average feature values for students labelled
    Employable -- the benchmark a Not-Employable prediction is compared
    against. train_preprocessed.csv is in SCALED space, so each row's
    numeric columns are unscaled first, then averaged -- unscale-then-average
    rather than average-then-unscale, since StandardScaler is linear and this
    keeps the result exact either way but makes the per-row unscale reusable.
    """
    if not os.path.exists(TRAIN_DATA_PATH):
        return None
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    if "Employable" not in train_df.columns:
        return None
    cols = [c for c in feature_columns if c in train_df.columns]
    employable_rows = train_df[train_df["Employable"] == 1][cols]

    numeric_cols = [c for c in prep["numeric_cols"] if c in cols]
    unscaled = prep["scaler"].inverse_transform(employable_rows[numeric_cols])
    employable_real = employable_rows.copy()
    employable_real[numeric_cols] = unscaled
    for c, train_min in prep.get("skew_transform_info", {}).items():
        if c in employable_real.columns:
            employable_real[c] = np.expm1(employable_real[c]) + train_min

    return employable_real.mean()


def _load_importance_ranking(feature_columns):
    path = os.path.join(MODELS_DIR, "skill_gap_report_permutation.csv")
    if os.path.exists(path):
        imp = pd.read_csv(path).set_index("feature")["importance_mean"]
        return imp.reindex(feature_columns).fillna(0)
    return pd.Series(1.0, index=feature_columns)


def identify_skill_gaps(df_model_ready, df_real_units, feature_columns, prep, pred, top_n=3):
    """For every row predicted Not Employable, returns a short, counselor-
    readable string listing the features that most (a) fall below the
    Employable-group average in REAL units AND (b) matter to the model
    (per permutation importance).

    - Continuous features (CGPA, Communication, ...): "Communication: you
      scored 5, Employable average is 8 -- gap of 3 points."
    - Binary skill/tool columns (Java, Python, ...) were never scaled, so
      the Employable average is already an interpretable proportion:
      "Java: you don't have this -- 74% of Employable students do."
    """
    employable_avg = _load_employable_reference(feature_columns, prep)
    importance = _load_importance_ranking(feature_columns)
    binary_cols = set(prep.get("binary_cols", []))

    if employable_avg is None:
        return ["(skill gap reference unavailable -- run preprocessing/train_model.py first)"] * len(df_model_ready)

    results = []
    for i in range(len(df_model_ready)):
        if pred[i] == 1:
            results.append("")
            continue

        real_row = df_real_units.iloc[i]
        scored_gaps = []
        for c in feature_columns:
            if c not in employable_avg.index or c not in real_row.index:
                continue
            gap = employable_avg[c] - real_row[c]  # positive = below Employable average, in real units
            if gap > 0.05:
                scored_gaps.append((c, gap, gap * max(importance.get(c, 0), 0)))
        scored_gaps.sort(key=lambda t: t[2], reverse=True)
        top = scored_gaps[:top_n]

        if not top:
            results.append("No standout gap vs. Employable group on modeled features.")
            continue

        parts = []
        for c, gap, _ in top:
            if c in binary_cols:
                pct = employable_avg[c] * 100
                parts.append(f"{c}: missing -- {pct:.0f}% of Employable students have it")
            else:
                current = real_row[c]
                target = employable_avg[c]
                parts.append(f"{c}: you={current:.1f}, Employable avg={target:.1f} (gap={gap:.1f})")
        results.append("; ".join(parts))

    return results


# --------------------------------------------------------------------------- #
# PUBLIC API
# --------------------------------------------------------------------------- #
def predict_dataframe(raw_df, explain=True, skill_gaps=True):
    prep, base_learners, meta_model, feature_columns, threshold, _ = load_artifacts()

    if "Student_ID" not in raw_df.columns:
        info("No 'Student_ID' column found -- duplicate-ID checking will be skipped.")

    df_model_ready, df_real_units, dup_id_mask, dup_row_mask = preprocess_new_data(raw_df, prep)
    proba, pred, label, meta_features = score(df_model_ready, base_learners, meta_model,
                                               feature_columns, threshold)

    result = raw_df.copy().reset_index(drop=True)
    result["Predicted_Proba"] = proba
    result["Predicted_Class"] = pred
    result["Predicted_Label"] = label
    result["Duplicate_Student_ID"] = dup_id_mask.values
    result["Duplicate_Feature_Row"] = dup_row_mask.values

    if explain:
        result["Prediction_Explanation"] = explain_predictions(meta_features, meta_model)
    if skill_gaps:
        result["Skill_Gap_Areas"] = identify_skill_gaps(df_model_ready, df_real_units,
                                                          feature_columns, prep, pred)

    return result


def predict_single(student_dict):
    raw_df = pd.DataFrame([student_dict])
    result = predict_dataframe(raw_df)
    row = result.iloc[0]
    return {
        "Predicted_Label": row["Predicted_Label"],
        "Predicted_Proba": float(row["Predicted_Proba"]),
        "Predicted_Class": int(row["Predicted_Class"]),
        "Prediction_Explanation": row.get("Prediction_Explanation", ""),
        "Skill_Gap_Areas": row.get("Skill_Gap_Areas", ""),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Score new students with the trained stacked ensemble.")
    parser.add_argument("--input", required=True, help="Path to a CSV of new, raw student rows.")
    parser.add_argument("--output", default=None, help="Where to write predictions CSV "
                                                         "(default: <input>_predictions.csv).")
    parser.add_argument("--no-explanations", action="store_true", help="Skip probability explanations.")
    parser.add_argument("--no-skill-gaps", action="store_true", help="Skip skill gap identification.")
    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    print(f"Loaded {len(raw_df)} new student row(s) from '{args.input}'.")

    result = predict_dataframe(raw_df, explain=not args.no_explanations, skill_gaps=not args.no_skill_gaps)

    output_path = args.output
    if output_path is None:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_predictions.csv"
    result.to_csv(output_path, index=False)

    print(f"\nPredicted class distribution:\n{result['Predicted_Label'].value_counts()}")
    print(f"Saved predictions -> {output_path}")


if __name__ == "__main__":
    main()