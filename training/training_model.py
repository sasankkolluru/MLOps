"""
================================================================================
training/train_model.py
================================================================================
Trains the stacked ensemble you designed:

                Feature Engineering & Scaling (preprocessing/preprocess.py)
                                    |
                ┌────────┬─────────┬──────────┬─────────┐
                │        │         │          │
              EBM    NGBoost   DeepForest    LCE            <- level-0 (base learners)
                │        │         │          │
                └────────┴─────────┴──────────┘
                     Predicted Probabilities (out-of-fold)
                                    |
                        Logistic Regression                  <- level-1 (meta learner)
                                    |
                      Employable / Not Employable

WHY OUT-OF-FOLD (OOF) STACKING:
Feeding the meta-learner predictions that a base model made on data it was
*trained* on leaks information and gives falsely optimistic meta-features.
Instead, each base learner's contribution to the meta-training set is
generated with manual StratifiedKFold cross-validation (proba on held-out
folds only). Each base learner is then refit on the FULL training set so it
is as strong as possible when producing meta-features for the untouched
test set.

GRACEFUL DEGRADATION:
EBM / NGBoost / Deep Forest / LCE are specialised libraries that are not
part of core scikit-learn. If one isn't installed, this script swaps in a
strong scikit-learn substitute with a similar inductive bias, prints a
warning, and keeps running end-to-end. Install the exact libraries to get
your original architecture:

    pip install interpret ngboost deep-forest lce-sklearn joblib

(package names: interpret -> ExplainableBoostingClassifier,
 ngboost -> NGBClassifier, deep-forest -> CascadeForestClassifier,
 lce-sklearn -> LCEClassifier)

================================================================================
WHAT'S NEW IN THIS VERSION (added on top of the original pipeline, nothing
from the original architecture / fallback logic was removed):
  1. Hyperparameter tuning      -> RandomizedSearchCV per base learner
  2. Feature selection          -> SelectFromModel (model-based, before training)
  3. Probability calibration    -> CalibratedClassifierCV on the final refit model
  4. Early stopping             -> enabled on every learner that supports it
  5. Class imbalance handling   -> sample_weight='balanced' passed to every fit()
  6. Random seed everywhere     -> random / numpy / every estimator / every split
  7. Threshold optimization     -> best F1 threshold found via OOF predictions
  8. Cross-validation metrics   -> per-fold accuracy/F1/ROC-AUC/PR-AUC, mean +/- std,
                                    saved so evaluate_model.py can run significance tests
================================================================================
"""

import os
import random
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    precision_recall_curve,
)
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# GLOBAL REPRODUCIBILITY  ("Random Seed Everywhere")
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

N_SPLITS = 5                 # outer folds for OOF stacking + CV metrics
CALIBRATION_CV = 3           # inner folds used only for probability calibration
TUNING_CV = 3                # folds used inside RandomizedSearchCV
TUNING_N_ITER = 8            # candidate settings tried per base learner
USE_CLASS_WEIGHTING = True   # toggle for "Class Imbalance Handling"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train_preprocessed.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "test_preprocessed.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def log(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# --------------------------------------------------------------------------- #
# 1. BASE LEARNERS  (real library if available, sklearn substitute otherwise)
#    Early stopping is switched on here wherever the estimator supports it.
# --------------------------------------------------------------------------- #
def build_base_learners():
    learners = {}

    # ---- EBM (Explainable Boosting Machine) ---------------------------- #
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
        learners["EBM"] = ExplainableBoostingClassifier(
            random_state=RANDOM_STATE,
            early_stopping_rounds=50,   # native early stopping
        )
    except ImportError:
        print("  [warn] 'interpret' not installed -> substituting EBM with "
              "HistGradientBoostingClassifier (pip install interpret for the real EBM).")
        learners["EBM"] = HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            early_stopping=True,            # native early stopping
            validation_fraction=0.1,
            n_iter_no_change=15,
        )

    # ---- NGBoost (probabilistic gradient boosting) ---------------------- #
    try:
        from ngboost import NGBClassifier
        learners["NGBoost"] = NGBClassifier(
            random_state=RANDOM_STATE, verbose=False,
            early_stopping_rounds=50,   # passed through at fit() time if supported
        )
    except ImportError:
        print("  [warn] 'ngboost' not installed -> substituting NGBoost with "
              "GradientBoostingClassifier (pip install ngboost for the real NGBoost).")
        learners["NGBoost"] = GradientBoostingClassifier(
            random_state=RANDOM_STATE,
            validation_fraction=0.1,
            n_iter_no_change=15,        # native early stopping
            tol=1e-4,
        )

    # ---- Deep Forest (gcForest cascade) ---------------------------------- #
    try:
        from deepforest import CascadeForestClassifier
        learners["DeepForest"] = CascadeForestClassifier(
            random_state=RANDOM_STATE, verbose=0,
        )  # deep-forest has automatic cascade-level early stopping built in
    except ImportError:
        print("  [warn] 'deep-forest' not installed -> substituting Deep Forest with "
              "RandomForestClassifier (pip install deep-forest for the real cascade forest).")
        learners["DeepForest"] = RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        )  # no native early stopping concept for a plain random forest

    # ---- LCE (Local Cascade Ensemble) ------------------------------------ #
    try:
        from lce import LCEClassifier
        learners["LCE"] = LCEClassifier(random_state=RANDOM_STATE)
    except ImportError:
        print("  [warn] 'lce' not installed -> substituting LCE with "
              "ExtraTreesClassifier (pip install lce-sklearn for the real LCE).")
        learners["LCE"] = ExtraTreesClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        )

    return learners


# --------------------------------------------------------------------------- #
# 2. HYPERPARAMETER TUNING  (RandomizedSearchCV, per base learner)
# --------------------------------------------------------------------------- #
def get_param_distribution(model):
    """Small, fast search spaces keyed off the estimator's class name, covering
    both the real specialised libraries and the sklearn fallbacks. Unknown
    estimator types return an empty dict (tuning is skipped, defaults used)."""
    name = type(model).__name__
    grids = {
        # sklearn fallbacks
        "HistGradientBoostingClassifier": {
            "max_iter": [100, 200, 300],
            "max_depth": [None, 5, 10],
            "learning_rate": [0.03, 0.05, 0.1, 0.2],
            "l2_regularization": [0.0, 0.1, 1.0],
        },
        "GradientBoostingClassifier": {
            "n_estimators": [100, 200, 300],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.03, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.85, 1.0],
        },
        "RandomForestClassifier": {
            "n_estimators": [200, 300, 400],
            "max_depth": [None, 10, 20, 30],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        },
        "ExtraTreesClassifier": {
            "n_estimators": [200, 300, 400],
            "max_depth": [None, 10, 20, 30],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        },
        # real specialised libraries (used automatically if installed)
        "ExplainableBoostingClassifier": {
            "learning_rate": [0.01, 0.02, 0.05],
            "max_bins": [128, 256],
            "max_interaction_bins": [16, 32],
        },
        "NGBClassifier": {
            "n_estimators": [200, 400, 600],
            "learning_rate": [0.01, 0.02, 0.05],
        },
        "LCEClassifier": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
        },
    }
    return grids.get(name, {})


def tune_hyperparameters(base_learners, X_train, y_train, sample_weight):
    log("STEP: HYPERPARAMETER TUNING (RandomizedSearchCV per base learner)")
    tuned_learners = {}
    tuning_summary = []

    for name, model in base_learners.items():
        param_dist = get_param_distribution(model)
        if not param_dist:
            print(f"  {name}: no tuning grid registered for {type(model).__name__} -> using defaults.")
            tuned_learners[name] = model
            continue
        try:
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_dist,
                n_iter=TUNING_N_ITER,
                cv=StratifiedKFold(n_splits=TUNING_CV, shuffle=True, random_state=RANDOM_STATE),
                scoring="f1",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                error_score="raise",
            )
            fit_kwargs = {"sample_weight": sample_weight} if USE_CLASS_WEIGHTING else {}
            search.fit(X_train, y_train, **fit_kwargs)
            tuned_learners[name] = search.best_estimator_
            print(f"  {name}: best CV F1 = {search.best_score_:.4f} | best params = {search.best_params_}")
            tuning_summary.append({"model": name, "best_cv_f1": search.best_score_,
                                    "best_params": str(search.best_params_)})
        except Exception as e:
            print(f"  [warn] Tuning failed for {name} ({e}) -> falling back to default hyperparameters.")
            tuned_learners[name] = model

    if tuning_summary:
        pd.DataFrame(tuning_summary).to_csv(
            os.path.join(MODELS_DIR, "hyperparameter_tuning_summary.csv"), index=False
        )
    return tuned_learners


# --------------------------------------------------------------------------- #
# 3. FEATURE SELECTION  (model-based, applied before base learners are trained)
# --------------------------------------------------------------------------- #
def select_features(X_train, X_test, y_train, sample_weight):
    log("STEP: FEATURE SELECTION (SelectFromModel, median-importance threshold)")
    selector_model = RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1,
        class_weight="balanced" if USE_CLASS_WEIGHTING else None,
    )
    selector_model.fit(X_train, y_train, sample_weight=sample_weight if USE_CLASS_WEIGHTING else None)
    selector = SelectFromModel(selector_model, threshold="median", prefit=True)
    support = selector.get_support()
    selected_cols = X_train.columns[support].tolist()
    dropped_cols = X_train.columns[~support].tolist()

    print(f"Kept {len(selected_cols)} / {X_train.shape[1]} features.")
    if dropped_cols:
        print(f"Dropped (below-median importance): {dropped_cols}")

    return X_train[selected_cols], X_test[selected_cols], selected_cols


# --------------------------------------------------------------------------- #
# 4. OUT-OF-FOLD STACKING FEATURES + CROSS-VALIDATION METRICS
# --------------------------------------------------------------------------- #
def build_oof_meta_features(base_learners, X_train, y_train, sample_weight):
    """
    Manual StratifiedKFold loop (instead of cross_val_predict) so we can:
      - pass per-fold sample_weight for class-imbalance handling
      - record per-fold accuracy / F1 / ROC-AUC / PR-AUC for every learner
        ("Missing Cross Validation Metrics")
    Returns:
      meta_train      : DataFrame of OOF probabilities (one column per learner)
      fitted_learners : each base learner refit on the FULL training set
      cv_fold_metrics : per-fold metrics for every base learner
      fold_id         : which outer fold each training row fell into
    """
    log("STEP: BUILDING OUT-OF-FOLD META-FEATURES + CROSS-VALIDATION METRICS")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    meta_train = pd.DataFrame(index=X_train.index, columns=[f"{n}_proba" for n in base_learners], dtype=float)
    fold_id = pd.Series(index=X_train.index, dtype=int)

    cv_rows = []
    X_train_np = X_train.reset_index(drop=True)
    y_train_np = y_train.reset_index(drop=True)
    sw_np = pd.Series(sample_weight, index=y_train_np.index) if sample_weight is not None else None

    for name, model in base_learners.items():
        print(f"  Cross-validating {name} ...")
        oof_proba = np.zeros(len(X_train_np))
        for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_train_np, y_train_np)):
            X_tr, X_val = X_train_np.iloc[tr_idx], X_train_np.iloc[val_idx]
            y_tr, y_val = y_train_np.iloc[tr_idx], y_train_np.iloc[val_idx]
            fold_model = clone(model)

            if USE_CLASS_WEIGHTING and sw_np is not None:
                try:
                    fold_model.fit(X_tr, y_tr, sample_weight=sw_np.iloc[tr_idx].values)
                except TypeError:
                    fold_model.fit(X_tr, y_tr)  # estimator doesn't accept sample_weight
            else:
                fold_model.fit(X_tr, y_tr)

            val_proba = fold_model.predict_proba(X_val)[:, 1]
            oof_proba[val_idx] = val_proba
            fold_id.iloc[val_idx] = fold_i

            val_pred = (val_proba >= 0.5).astype(int)
            cv_rows.append({
                "fold": fold_i, "model": name,
                "accuracy": accuracy_score(y_val, val_pred),
                "f1": f1_score(y_val, val_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_val, val_proba),
                "pr_auc": average_precision_score(y_val, val_proba),
            })

        meta_train[f"{name}_proba"] = oof_proba

        # Refit on the FULL training set -> used later to score the test set
        if USE_CLASS_WEIGHTING and sw_np is not None:
            try:
                model.fit(X_train_np, y_train_np, sample_weight=sw_np.values)
            except TypeError:
                model.fit(X_train_np, y_train_np)
        else:
            model.fit(X_train_np, y_train_np)

    meta_train.index = X_train.index
    cv_fold_metrics = pd.DataFrame(cv_rows)

    log("CROSS-VALIDATION METRICS (mean +/- std across folds)")
    summary = cv_fold_metrics.groupby("model")[["accuracy", "f1", "roc_auc", "pr_auc"]].agg(["mean", "std"])
    print(summary.round(4))

    return meta_train, base_learners, cv_fold_metrics, fold_id


def build_test_meta_features(base_learners, X_test):
    log("STEP: BUILDING META-FEATURES FOR THE TEST SET (level-0 predictions)")
    meta_test = pd.DataFrame(index=X_test.index)
    for name, model in base_learners.items():
        meta_test[f"{name}_proba"] = model.predict_proba(X_test)[:, 1]
    return meta_test


# --------------------------------------------------------------------------- #
# 5. PROBABILITY CALIBRATION  (applied to the final, fully-refit base learners)
# --------------------------------------------------------------------------- #
def calibrate_base_learners(base_learners, X_train, y_train, sample_weight):
    log("STEP: PROBABILITY CALIBRATION (CalibratedClassifierCV, sigmoid)")
    calibrated = {}
    for name, model in base_learners.items():
        try:
            calibrator = CalibratedClassifierCV(
                estimator=clone(model), method="sigmoid", cv=CALIBRATION_CV,
            )
            try:
                if USE_CLASS_WEIGHTING:
                    calibrator.fit(X_train, y_train, sample_weight=sample_weight)
                else:
                    calibrator.fit(X_train, y_train)
            except TypeError:
                calibrator.fit(X_train, y_train)
            calibrated[name] = calibrator
            print(f"  {name}: calibrated with {CALIBRATION_CV}-fold sigmoid (Platt) scaling.")
        except Exception as e:
            print(f"  [warn] Calibration failed for {name} ({e}) -> using uncalibrated model.")
            calibrated[name] = model
    return calibrated


# --------------------------------------------------------------------------- #
# 6. META LEARNER + THRESHOLD OPTIMIZATION
# --------------------------------------------------------------------------- #
def train_meta_learner(meta_train, y_train):
    log("STEP: TRAINING META-LEARNER (Logistic Regression)")
    meta_model = LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE,
        class_weight="balanced" if USE_CLASS_WEIGHTING else None,
    )
    meta_model.fit(meta_train, y_train)
    print("Meta-learner coefficients (weight given to each base learner):")
    for col, coef in zip(meta_train.columns, meta_model.coef_[0]):
        print(f"  {col:>18s} : {coef:+.4f}")
    return meta_model


def optimize_threshold(meta_model, meta_train, y_train):
    """Finds the probability threshold that maximises F1 on the OOF
    meta-training predictions (never touches the test set)."""
    log("STEP: THRESHOLD OPTIMIZATION (maximise F1 on OOF predictions)")
    oof_meta_proba = meta_model.predict_proba(meta_train)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, oof_meta_proba)
    f1_scores = np.where((precision + recall) > 0,
                          2 * precision * recall / (precision + recall + 1e-12), 0)
    best_idx = np.argmax(f1_scores[:-1]) if len(thresholds) > 0 else 0
    best_threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5
    print(f"Default threshold 0.5 -> F1 = {f1_score(y_train, (oof_meta_proba >= 0.5).astype(int)):.4f}")
    print(f"Optimised threshold {best_threshold:.4f} -> F1 = {f1_scores[best_idx]:.4f}")
    return float(best_threshold)


def stacked_cv_metrics(cv_fold_metrics, meta_train, fold_id, y_train, meta_model, threshold):
    """Adds a 'STACKED' row per fold to cv_fold_metrics using the already
    trained meta-learner scored on each fold's OOF meta-features, so
    evaluate_model.py can run paired significance tests fold-by-fold."""
    rows = []
    for fold_i in sorted(fold_id.unique()):
        idx = fold_id[fold_id == fold_i].index
        proba = meta_model.predict_proba(meta_train.loc[idx])[:, 1]
        pred = (proba >= threshold).astype(int)
        y_fold = y_train.loc[idx]
        rows.append({
            "fold": fold_i, "model": "STACKED",
            "accuracy": accuracy_score(y_fold, pred),
            "f1": f1_score(y_fold, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_fold, proba),
            "pr_auc": average_precision_score(y_fold, proba),
        })
    return pd.concat([cv_fold_metrics, pd.DataFrame(rows)], ignore_index=True)


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    log("STEP 1: LOAD PREPROCESSED DATA")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train: {train_df.shape}  Test: {test_df.shape}")

    X_train = train_df.drop(columns=["Employable"])
    y_train = train_df["Employable"]
    X_test = test_df.drop(columns=["Employable"])
    y_test = test_df["Employable"]

    # ---- Class imbalance handling: sample weights used in every fit() ---- #
    sample_weight = compute_sample_weight("balanced", y_train) if USE_CLASS_WEIGHTING else None
    if USE_CLASS_WEIGHTING:
        counts = y_train.value_counts()
        ratio = counts.min() / counts.max()
        print(f"Class distribution: {dict(counts)}  (minority/majority ratio = {ratio:.2f})")
        print("Class-imbalance handling ENABLED: sample_weight='balanced' passed to every fit().")

    # ---- Feature selection (before any base learner sees the data) ------ #
    X_train, X_test, selected_features = select_features(X_train, X_test, y_train, sample_weight)

    log("STEP 2: INITIALISE BASE LEARNERS")
    base_learners = build_base_learners()

    # ---- Hyperparameter tuning ------------------------------------------ #
    base_learners = tune_hyperparameters(base_learners, X_train, y_train, sample_weight)

    # ---- OOF meta-features, per-fold CV metrics, full refit ------------- #
    meta_train, base_learners, cv_fold_metrics, fold_id = build_oof_meta_features(
        base_learners, X_train, y_train, sample_weight
    )

    # ---- Calibrate the fully-refit base learners ------------------------ #
    calibrated_learners = calibrate_base_learners(base_learners, X_train, y_train, sample_weight)

    # ---- Level-0 predictions on the untouched test set (calibrated) ----- #
    meta_test = build_test_meta_features(calibrated_learners, X_test)

    # ---- Level-1: meta learner + threshold optimisation ------------------ #
    meta_model = train_meta_learner(meta_train, y_train)
    best_threshold = optimize_threshold(meta_model, meta_train, y_train)
    cv_fold_metrics = stacked_cv_metrics(cv_fold_metrics, meta_train, fold_id, y_train, meta_model, best_threshold)

    log("STEP: SAVE MODELS & ARTIFACTS")
    joblib.dump(calibrated_learners, os.path.join(MODELS_DIR, "base_learners.pkl"))
    joblib.dump(meta_model, os.path.join(MODELS_DIR, "meta_learner.pkl"))
    joblib.dump(selected_features, os.path.join(MODELS_DIR, "feature_columns.pkl"))
    joblib.dump(best_threshold, os.path.join(MODELS_DIR, "optimal_threshold.pkl"))
    cv_fold_metrics.to_csv(os.path.join(MODELS_DIR, "cv_fold_metrics.csv"), index=False)
    print(f"Saved: base_learners.pkl, meta_learner.pkl, feature_columns.pkl, "
          f"optimal_threshold.pkl, cv_fold_metrics.csv -> {MODELS_DIR}")

    meta_test_with_target = meta_test.copy()
    meta_test_with_target["Employable"] = y_test.values
    meta_test_with_target.to_csv(os.path.join(MODELS_DIR, "meta_test_features.csv"), index=False)

    log("TRAINING COMPLETE")
    print(f"Optimised decision threshold saved: {best_threshold:.4f} (evaluate_model.py will load and use it).")
    print("Run evaluation/evaluate_model.py next to score the stacked model on the test set.")


if __name__ == "__main__":
    main()