"""
================================================================================
evaluation/evaluate_model.py
================================================================================
Loads the trained stacked ensemble (training/train_model.py) and evaluates it
on the held-out test set:

  1. Metrics for every base learner AND the stacked model, including PR-AUC,
     using the OPTIMISED decision threshold saved during training (not a
     blind 0.5 cutoff).
  2. A proper Skill-Gap-Identification engine: instead of a naive
     mean(employable) - mean(not-employable) difference, it uses
     permutation importance on the actual stacked model (and SHAP if it's
     installed) to find which features truly drive the prediction.
  3. Statistical significance testing between the stacked model and each
     base learner:
       - McNemar's test        (paired, instance-level, on the test set)
       - Paired t-test         (paired, fold-level, on training CV metrics)
       - Wilcoxon signed-rank  (paired, fold-level, on training CV metrics)
================================================================================
WHAT'S NEW IN THIS VERSION (added on top of the original evaluation script,
nothing from the original metrics / confusion-matrix / report logic was
removed):
  1. PR-AUC (average precision) added to every metrics row
  2. Uses the OPTIMISED threshold from train_model.py instead of 0.5
  3. Skill Gap Engine rebuilt on permutation importance (+ SHAP if available)
     of the actual stacked model, instead of a raw scaled mean-difference
  4. Statistical significance: McNemar, paired t-test, Wilcoxon signed-rank
================================================================================
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, roc_curve,
)
from scipy.stats import chi2, binomtest, ttest_rel, wilcoxon

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "test_preprocessed.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def log(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_metrics(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)   # <-- NEW: PR-AUC
    print(f"{name:>12s} | Acc {acc:.4f} | Prec {prec:.4f} | Rec {rec:.4f} "
          f"| F1 {f1:.4f} | ROC-AUC {auc:.4f} | PR-AUC {pr_auc:.4f}")
    return {"model": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "roc_auc": auc, "pr_auc": pr_auc}


# --------------------------------------------------------------------------- #
# Stacked ensemble wrapped as a single sklearn-compatible estimator so that
# permutation_importance can be computed on the WHOLE pipeline (raw features
# in -> final stacked prediction out), not just on one base learner.
# --------------------------------------------------------------------------- #
class StackedEnsembleWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, base_learners, meta_model, threshold=0.5):
        self.base_learners = base_learners
        self.meta_model = meta_model
        self.threshold = threshold
        self.classes_ = np.array([0, 1])

    def _meta_features(self, X):
        meta = pd.DataFrame(index=X.index if hasattr(X, "index") else range(len(X)))
        for name, model in self.base_learners.items():
            meta[f"{name}_proba"] = model.predict_proba(X)[:, 1]
        return meta

    def predict_proba(self, X):
        meta = self._meta_features(X)
        proba_pos = self.meta_model.predict_proba(meta)[:, 1]
        return np.column_stack([1 - proba_pos, proba_pos])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def fit(self, X, y):
        # Already fitted upstream in train_model.py; fit() is a no-op so
        # this estimator satisfies sklearn's API (needed by permutation_importance).
        return self


# --------------------------------------------------------------------------- #
# STATISTICAL SIGNIFICANCE TESTS
# --------------------------------------------------------------------------- #
def mcnemar_test(y_true, pred_a, pred_b):
    """Paired McNemar's test on instance-level correct/incorrect outcomes.
    b = A correct & B wrong, c = A wrong & B correct.
    Uses the exact binomial test for small discordant counts (n < 25),
    otherwise the standard chi-square approximation with continuity correction."""
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return 0.0, 1.0
    if n < 25:
        p_value = binomtest(min(b, c), n, 0.5).pvalue
        stat = min(b, c)
    else:
        stat = (abs(b - c) - 1) ** 2 / n
        p_value = 1 - chi2.cdf(stat, df=1)
    return stat, p_value


def run_significance_tests(y_test, stacked_pred, base_preds, cv_fold_metrics):
    log("STEP: STATISTICAL SIGNIFICANCE TESTING")

    print("McNemar's test (instance-level, test set) -- is STACKED significantly")
    print("different from each base learner on the SAME test instances?\n")
    mcnemar_rows = []
    for name, pred in base_preds.items():
        stat, p = mcnemar_test(y_test.values, stacked_pred, pred)
        sig = "YES (p<0.05)" if p < 0.05 else "no"
        print(f"  STACKED vs {name:<12s}: statistic={stat:.4f}  p-value={p:.4f}  significant={sig}")
        mcnemar_rows.append({"comparison": f"STACKED_vs_{name}", "statistic": stat,
                              "p_value": p, "significant_at_0.05": p < 0.05})

    print("\nPaired t-test & Wilcoxon signed-rank test (fold-level, on the "
          "per-fold F1 scores recorded during training's cross-validation) --")
    print("is STACKED consistently better across folds, not just on average?\n")
    ttest_rows = []
    if cv_fold_metrics is not None and "STACKED" in cv_fold_metrics["model"].unique():
        stacked_folds = cv_fold_metrics[cv_fold_metrics["model"] == "STACKED"].sort_values("fold")["f1"].values
        for name in [m for m in cv_fold_metrics["model"].unique() if m != "STACKED"]:
            base_folds = cv_fold_metrics[cv_fold_metrics["model"] == name].sort_values("fold")["f1"].values
            if len(base_folds) != len(stacked_folds) or len(base_folds) < 2:
                continue
            t_stat, t_p = ttest_rel(stacked_folds, base_folds)
            try:
                w_stat, w_p = wilcoxon(stacked_folds, base_folds)
            except ValueError:
                w_stat, w_p = np.nan, np.nan  # identical scores in every fold
            print(f"  STACKED vs {name:<12s}: paired t-test p={t_p:.4f} | "
                  f"Wilcoxon p={w_p if np.isnan(w_p) else round(w_p, 4)}")
            ttest_rows.append({"comparison": f"STACKED_vs_{name}",
                                "ttest_stat": t_stat, "ttest_p": t_p,
                                "wilcoxon_stat": w_stat, "wilcoxon_p": w_p})
    else:
        print("  [info] cv_fold_metrics.csv has no 'STACKED' rows -> re-run the "
              "latest training/train_model.py to enable this test.")

    pd.DataFrame(mcnemar_rows).to_csv(os.path.join(MODELS_DIR, "mcnemar_test_results.csv"), index=False)
    if ttest_rows:
        pd.DataFrame(ttest_rows).to_csv(os.path.join(MODELS_DIR, "paired_significance_tests.csv"), index=False)
    print(f"\nSaved: mcnemar_test_results.csv"
          + (", paired_significance_tests.csv" if ttest_rows else "") + f" -> {MODELS_DIR}")


# --------------------------------------------------------------------------- #
# SKILL GAP IDENTIFICATION ENGINE v2  (permutation importance, model-driven)
# --------------------------------------------------------------------------- #
def skill_gap_analysis_v2(stacked_estimator, X_test, y_test, predictions_out):
    log("STEP: SKILL GAP IDENTIFICATION ENGINE (v2 - permutation importance)")
    print("The old version only compared raw mean scores between groups, which "
          "conflates correlation with actual model influence (e.g. a skill split "
          "50/50 in both groups can still show a nonzero 'gap' that means nothing).")
    print("This version measures how much each feature actually moves the STACKED "
          "model's predictions when permuted -- a model-driven, not descriptive, gap.\n")

    # ---- Permutation importance on the real stacked pipeline ------------ #
    perm = permutation_importance(
        stacked_estimator, X_test, y_test,
        scoring="f1", n_repeats=15, random_state=RANDOM_STATE, n_jobs=-1,
    )
    perm_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    print("Permutation importance (drop in F1 when a feature is shuffled), highest first:")
    print(perm_df.round(4).to_string(index=False))

    # ---- SHAP, only if installed (optional deeper explanation) ---------- #
    try:
        import shap
        print("\n[shap] library detected -> computing SHAP values on a sample of the test set ...")
        sample = X_test.sample(min(200, len(X_test)), random_state=RANDOM_STATE)
        explainer = shap.KernelExplainer(lambda X: stacked_estimator.predict_proba(pd.DataFrame(X, columns=X_test.columns))[:, 1],
                                          shap.sample(X_test, 50, random_state=RANDOM_STATE))
        shap_values = explainer.shap_values(sample, nsamples=100)
        shap_importance = pd.DataFrame({
            "feature": X_test.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        print(shap_importance.round(4).to_string(index=False))
        shap_importance.to_csv(os.path.join(MODELS_DIR, "shap_importance.csv"), index=False)
    except ImportError:
        print("\n[info] 'shap' not installed -> skipping SHAP values "
              "(pip install shap for individual-prediction-level explanations). "
              "Permutation importance above is model-agnostic and already "
              "reflects true predictive influence, just without per-student detail.")

    print("\nNOTE ON SCALE: features are on the SCALED (StandardScaler) space, so "
          "compare relative importance ranking, not raw units.")

    print("\nCurriculum recommendation candidates (ranked by real model influence, not raw mean gap):")
    for _, row in perm_df.head(5).iterrows():
        print(f"  - Prioritise '{row['feature']}': permutation importance "
              f"{row['importance_mean']:.4f} (+/- {row['importance_std']:.4f}) -- "
              f"shuffling this feature drops the stacked model's F1 by that amount, "
              f"meaning the model genuinely relies on it, not just a coincidental average difference.")

    perm_df.to_csv(os.path.join(MODELS_DIR, "skill_gap_report_permutation.csv"), index=False)
    print(f"\nSaved: skill_gap_report_permutation.csv -> {MODELS_DIR}")

    # ---- Keep the original descriptive mean-gap table too, clearly labelled
    #      as supplementary/descriptive, NOT a measure of true influence. ---- #
    log("SUPPLEMENTARY: DESCRIPTIVE MEAN-GAP TABLE (NOT a causal/importance measure)")
    predicted_no = predictions_out[predictions_out["Predicted"] == 0]
    predicted_yes = predictions_out[predictions_out["Predicted"] == 1]
    if predicted_no.empty or predicted_yes.empty:
        print("One of the predicted classes is empty in this test set -> skipping.")
        return
    gap_rows = []
    for c in X_test.columns:
        gap_rows.append({
            "feature": c,
            "avg_employable": predicted_yes[c].mean(),
            "avg_not_employable": predicted_no[c].mean(),
            "mean_gap": predicted_yes[c].mean() - predicted_no[c].mean(),
        })
    gap_df = pd.DataFrame(gap_rows).sort_values("mean_gap", ascending=False)
    print(gap_df.round(3).to_string(index=False))
    print("\n[caveat] A large mean gap does not prove the feature drives predictions "
          "(see permutation importance above for that). Use this table only for "
          "descriptive context, e.g. reporting to non-technical stakeholders.")
    gap_df.to_csv(os.path.join(MODELS_DIR, "skill_gap_report_descriptive.csv"), index=False)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    log("STEP 1: LOAD TEST DATA AND TRAINED ARTIFACTS")
    test_df = pd.read_csv(TEST_PATH)
    X_test = test_df.drop(columns=["Employable"])
    y_test = test_df["Employable"]

    base_learners = joblib.load(os.path.join(MODELS_DIR, "base_learners.pkl"))
    meta_model = joblib.load(os.path.join(MODELS_DIR, "meta_learner.pkl"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))

    threshold_path = os.path.join(MODELS_DIR, "optimal_threshold.pkl")
    optimal_threshold = joblib.load(threshold_path) if os.path.exists(threshold_path) else 0.5
    if os.path.exists(threshold_path):
        print(f"Loaded optimised decision threshold from training: {optimal_threshold:.4f}")
    else:
        print("[info] optimal_threshold.pkl not found -> defaulting to 0.5 "
              "(re-run the latest training/train_model.py to enable threshold optimisation).")

    X_test = X_test[feature_columns]  # enforce identical column order / selected features
    print(f"Loaded {len(base_learners)} base learners + 1 meta-learner. Test shape: {X_test.shape}")

    cv_fold_metrics_path = os.path.join(MODELS_DIR, "cv_fold_metrics.csv")
    cv_fold_metrics = pd.read_csv(cv_fold_metrics_path) if os.path.exists(cv_fold_metrics_path) else None

    # --------------------------------------------------------------- #
    # STEP 2: SCORE EACH BASE LEARNER INDIVIDUALLY  (still 0.5 -- base
    # learners were never threshold-tuned, only the stacked model was)
    # --------------------------------------------------------------- #
    log("STEP 2: BASE LEARNER PERFORMANCE (individually, for comparison)")
    results = []
    base_preds = {}
    meta_test = pd.DataFrame(index=X_test.index)
    for name, model in base_learners.items():
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        meta_test[f"{name}_proba"] = proba
        base_preds[name] = pred
        results.append(print_metrics(name, y_test, pred, proba))

    # --------------------------------------------------------------- #
    # STEP 3: SCORE THE STACKED (META) MODEL -- using the OPTIMISED threshold
    # --------------------------------------------------------------- #
    log("STEP 3: STACKED MODEL PERFORMANCE (Logistic Regression meta-learner)")
    stacked_proba = meta_model.predict_proba(meta_test)[:, 1]
    stacked_pred_05 = (stacked_proba >= 0.5).astype(int)
    stacked_pred_opt = (stacked_proba >= optimal_threshold).astype(int)

    print("-- at default threshold 0.5 --")
    print_metrics("STACKED@0.5", y_test, stacked_pred_05, stacked_proba)
    print("-- at optimised threshold (from training OOF predictions) --")
    results.append(print_metrics(f"STACKED@{optimal_threshold:.2f}", y_test, stacked_pred_opt, stacked_proba))
    stacked_pred = stacked_pred_opt  # this is the one used for everything below

    results_df = pd.DataFrame(results).set_index("model")
    log("STEP 4: SUMMARY COMPARISON TABLE")
    print(results_df.round(4))
    best_model = results_df["f1"].idxmax()
    print(f"\nBest F1 score -> {best_model}")

    # --------------------------------------------------------------- #
    # STEP 5: DETAILED REPORT FOR THE STACKED MODEL
    # --------------------------------------------------------------- #
    log("STEP 5: STACKED MODEL - CONFUSION MATRIX & CLASSIFICATION REPORT")
    cm = confusion_matrix(y_test, stacked_pred)
    print("Confusion Matrix (rows=actual, cols=predicted) [0=Not Employable, 1=Employable]")
    print(pd.DataFrame(cm, index=["Actual_No", "Actual_Yes"], columns=["Pred_No", "Pred_Yes"]))
    print("\nClassification Report:")
    print(classification_report(y_test, stacked_pred, target_names=["Not Employable", "Employable"]))

    fpr, tpr, _ = roc_curve(y_test, stacked_proba)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        os.path.join(MODELS_DIR, "roc_curve_stacked.csv"), index=False
    )
    predictions_out = X_test.copy()
    predictions_out["Actual"] = y_test.values
    predictions_out["Predicted"] = stacked_pred
    predictions_out["Predicted_Proba"] = stacked_proba
    predictions_out.to_csv(os.path.join(MODELS_DIR, "test_predictions.csv"), index=False)
    results_df.to_csv(os.path.join(MODELS_DIR, "model_comparison.csv"))
    print(f"\nSaved: model_comparison.csv, test_predictions.csv, roc_curve_stacked.csv -> {MODELS_DIR}")

    # --------------------------------------------------------------- #
    # STEP 6: STATISTICAL SIGNIFICANCE TESTS
    # --------------------------------------------------------------- #
    run_significance_tests(y_test, stacked_pred, base_preds, cv_fold_metrics)

    # --------------------------------------------------------------- #
    # STEP 7: SKILL GAP IDENTIFICATION (v2, permutation-importance based)
    # --------------------------------------------------------------- #
    stacked_estimator = StackedEnsembleWrapper(base_learners, meta_model, threshold=optimal_threshold)
    skill_gap_analysis_v2(stacked_estimator, X_test, y_test, predictions_out)


if __name__ == "__main__":
    main()