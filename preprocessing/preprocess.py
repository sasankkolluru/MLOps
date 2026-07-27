"""
================================================================================
END-TO-END DATA PREPROCESSING PIPELINE
Dataset : student_employability_dataset.csv
Target  : Employable (Yes/No)
================================================================================
Raw Data -> Data Collection -> Data Inspection -> Data Cleaning ->
Remove Duplicates -> Handle Missing Values -> Handle Inconsistent Data ->
Remove Noisy Data -> Outlier Detection & Treatment -> Data Transformation ->
Categorical Encoding -> Feature Scaling -> Feature Selection ->
Feature Extraction/Engineering -> Data Balancing -> Train-Test Split ->
Final Preprocessed Data
================================================================================
Every step is defensive: it checks whether an issue exists before acting,
so the script runs cleanly whether the raw data is messy or already clean.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

RANDOM_STATE = 42
INPUT_PATH = "student_employability_dataset.csv"
OUTPUT_DIR = "."

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)


def log(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ------------------------------------------------------------------------- #
# 1. DATA COLLECTION
# ------------------------------------------------------------------------- #
def load_data(path):
    log("STEP 1: DATA COLLECTION")
    df = pd.read_csv(path)
    print(f"Loaded '{path}' -> shape = {df.shape}")
    return df


# ------------------------------------------------------------------------- #
# 2. DATA INSPECTION
# ------------------------------------------------------------------------- #
def inspect_data(df):
    log("STEP 2: DATA INSPECTION")
    print("Shape:", df.shape)
    print("\nDtypes:\n", df.dtypes)
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nDescribe (numeric):\n", df.describe())

    cat_like_cols = df.select_dtypes(include=["object", "string", "str"]).columns
    if len(cat_like_cols) > 0:
        print("\nUnique values in categorical/text columns:")
        for c in cat_like_cols:
            print(f"  {c}: {df[c].unique()[:10]}")
    return df


# ------------------------------------------------------------------------- #
# 3. DATA CLEANING
# ------------------------------------------------------------------------- #
def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed duplicates: {before - len(df)} rows dropped -> new shape {df.shape}")
    return df


def handle_missing_values(df):
    missing_before = df.isnull().sum().sum()
    if missing_before == 0:
        print("No missing values found. Nothing to impute.")
        return df

    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include=["object", "string", "str"]).columns

    for c in num_cols:
        if df[c].isnull().any():
            median_val = df[c].median()
            df[c] = df[c].fillna(median_val)
            print(f"  Filled {c} missing values with median = {median_val}")

    for c in cat_cols:
        if df[c].isnull().any():
            mode_val = df[c].mode()[0]
            df[c] = df[c].fillna(mode_val)
            print(f"  Filled {c} missing values with mode = '{mode_val}'")

    print(f"Total missing values handled: {missing_before}")
    return df


def handle_inconsistent_data(df):
    """Standardise text formatting, casing, whitespace and inconsistent labels."""
    changed = False

    if "Student_ID" in df.columns:
        cleaned = df["Student_ID"].astype(str).str.strip().str.upper()
        if not cleaned.equals(df["Student_ID"]):
            changed = True
        df["Student_ID"] = cleaned

    if "Employable" in df.columns:
        df["Employable"] = df["Employable"].astype(str).str.strip().str.capitalize()
        valid_map = {
            "Yes": "Yes", "Y": "Yes", "1": "Yes", "True": "Yes",
            "No": "No", "N": "No", "0": "No", "False": "No",
        }
        mapped = df["Employable"].map(valid_map)
        unmapped = mapped.isnull().sum()
        if unmapped > 0:
            print(f"  Warning: {unmapped} unrecognised 'Employable' labels dropped.")
            df = df[mapped.notnull()].reset_index(drop=True)
            mapped = mapped.dropna().reset_index(drop=True)
        df["Employable"] = mapped
        changed = True

    binary_cols = ["Java", "Python", "C++", "SQL", "DSA", "OOP", "Git",
                    "Spring_Boot", "React", "AWS"]
    for c in binary_cols:
        if c in df.columns:
            bad = ~df[c].isin([0, 1])
            if bad.any():
                print(f"  Fixing {bad.sum()} invalid entries in '{c}' (forcing to 0/1)")
                df[c] = df[c].apply(lambda v: 1 if v not in [0, 1] and v > 0 else (0 if v not in [0, 1] else v))
                changed = True

    print("Inconsistent data handled." if changed else "No inconsistent formatting detected.")
    return df


def remove_noisy_data(df):
    """Drop rows with logically impossible / nonsensical values."""
    before = len(df)

    if "CGPA" in df.columns:
        df = df[(df["CGPA"] >= 0) & (df["CGPA"] <= 10)]

    score_cols = ["Communication", "Aptitude", "Problem_Solving", "Teamwork"]
    for c in score_cols:
        if c in df.columns:
            df = df[(df[c] >= 1) & (df[c] <= 10)]

    count_cols = ["Projects", "Internships", "Certifications"]
    for c in count_cols:
        if c in df.columns:
            df = df[df[c] >= 0]

    df = df.reset_index(drop=True)
    removed = before - len(df)
    print(f"Noisy/invalid rows removed: {removed} -> new shape {df.shape}")
    return df


# ------------------------------------------------------------------------- #
# 4. OUTLIER DETECTION & TREATMENT (IQR capping)
# ------------------------------------------------------------------------- #
def treat_outliers(df, cols):
    log("STEP: OUTLIER DETECTION & TREATMENT (IQR method)")
    total_capped = 0
    for c in cols:
        Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n_outliers = ((df[c] < lower) | (df[c] > upper)).sum()
        if n_outliers > 0:
            df[c] = df[c].clip(lower=lower, upper=upper)
            total_capped += n_outliers
            print(f"  {c}: {n_outliers} outliers capped to [{lower:.2f}, {upper:.2f}]")
    if total_capped == 0:
        print("No outliers detected outside 1.5*IQR bounds.")
    return df


# ------------------------------------------------------------------------- #
# 5. DATA TRANSFORMATION
# ------------------------------------------------------------------------- #
def transform_data(df, cols):
    log("STEP: DATA TRANSFORMATION (skew correction)")
    for c in cols:
        skew = df[c].skew()
        if abs(skew) > 1:  # strongly skewed -> log1p transform (data is non-negative)
            df[c] = np.log1p(df[c] - df[c].min())
            print(f"  {c}: skew={skew:.2f} -> applied log1p transform")
    print("Transformation check complete.")
    return df


# ------------------------------------------------------------------------- #
# 6. CATEGORICAL ENCODING
# ------------------------------------------------------------------------- #
def encode_categorical(df):
    log("STEP: CATEGORICAL DATA ENCODING")
    le = LabelEncoder()
    df["Employable"] = le.fit_transform(df["Employable"])  # No=0, Yes=1
    print(f"Encoded target 'Employable' -> classes {list(le.classes_)} => {list(le.transform(le.classes_))}")

    if "Student_ID" in df.columns:
        df = df.drop(columns=["Student_ID"])
        print("Dropped 'Student_ID' (identifier column, no predictive value)")
    return df, le


# ------------------------------------------------------------------------- #
# 7. FEATURE ENGINEERING
# ------------------------------------------------------------------------- #
def engineer_features(df):
    log("STEP: FEATURE EXTRACTION / ENGINEERING")
    skill_cols = ["Java", "Python", "C++", "SQL", "DSA", "OOP", "Git", "Spring_Boot", "React", "AWS"]
    soft_skill_cols = ["Communication", "Aptitude", "Problem_Solving", "Teamwork"]

    df["Total_Tech_Skills"] = df[skill_cols].sum(axis=1)
    df["Avg_Soft_Skill_Score"] = df[soft_skill_cols].mean(axis=1)
    df["Experience_Score"] = df["Projects"] + df["Internships"] + df["Certifications"]

    print("Created engineered features: Total_Tech_Skills, Avg_Soft_Skill_Score, Experience_Score")
    return df


# ------------------------------------------------------------------------- #
# 8. FEATURE SCALING
# ------------------------------------------------------------------------- #
def scale_features(X_train, X_test, numeric_cols):
    log("STEP: FEATURE SCALING (StandardScaler)")
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    print(f"Scaled {len(numeric_cols)} numeric columns using training-set statistics only.")
    return X_train_scaled, X_test_scaled, scaler


# ------------------------------------------------------------------------- #
# 9. FEATURE SELECTION
# ------------------------------------------------------------------------- #
def select_features(X_train, y_train, X_test, k="all"):
    log("STEP: FEATURE SELECTION")

    vt = VarianceThreshold(threshold=0.0)
    vt.fit(X_train)
    kept_after_variance = X_train.columns[vt.get_support()]
    dropped_variance = set(X_train.columns) - set(kept_after_variance)
    if dropped_variance:
        print(f"Dropped zero-variance features: {dropped_variance}")
    X_train = X_train[kept_after_variance]
    X_test = X_test[kept_after_variance]

    selector = SelectKBest(score_func=f_classif, k=(k if k != "all" else "all"))
    selector.fit(X_train, y_train)
    scores = pd.Series(selector.scores_, index=X_train.columns).sort_values(ascending=False)
    print("Feature importance (ANOVA F-score), highest to lowest:")
    print(scores)

    return X_train, X_test, scores


# ------------------------------------------------------------------------- #
# 10. DATA BALANCING
# ------------------------------------------------------------------------- #
def balance_data(X_train, y_train):
    log("STEP: DATA BALANCING (train set only)")
    counts = y_train.value_counts()
    print("Class distribution before balancing:\n", counts)

    minority_ratio = counts.min() / counts.max()
    if minority_ratio >= 0.8:
        print(f"Classes are reasonably balanced (ratio={minority_ratio:.2f} >= 0.8). Skipping balancing.")
        return X_train, y_train

    print(f"Imbalance detected (ratio={minority_ratio:.2f}). Applying random oversampling of the minority class.")
    train = X_train.copy()
    train["__target__"] = y_train.values
    majority_class = counts.idxmax()
    minority_class = counts.idxmin()

    majority_df = train[train["__target__"] == majority_class]
    minority_df = train[train["__target__"] == minority_class]
    minority_upsampled = minority_df.sample(n=len(majority_df), replace=True, random_state=RANDOM_STATE)

    balanced = pd.concat([majority_df, minority_upsampled]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    y_bal = balanced.pop("__target__")
    print("Class distribution after balancing:\n", y_bal.value_counts())
    return balanced, y_bal


# ------------------------------------------------------------------------- #
# MAIN PIPELINE
# ------------------------------------------------------------------------- #
def main():
    df = load_data(INPUT_PATH)
    df = inspect_data(df)

    log("STEP 3: DATA CLEANING")
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = handle_inconsistent_data(df)
    df = remove_noisy_data(df)

    numeric_cols_for_outliers = ["CGPA", "Communication", "Aptitude", "Problem_Solving", "Teamwork"]
    df = treat_outliers(df, numeric_cols_for_outliers)
    df = transform_data(df, ["Projects", "Internships", "Certifications"])

    df, target_encoder = encode_categorical(df)
    df = engineer_features(df)

    log("STEP: TRAIN-TEST SPLIT")
    X = df.drop(columns=["Employable"])
    y = df["Employable"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    X_train, X_test, scaler = scale_features(X_train, X_test, numeric_cols)

    X_train, X_test, feature_scores = select_features(X_train, y_train, X_test)

    X_train_bal, y_train_bal = balance_data(X_train, y_train)

    log("FINAL PREPROCESSED DATA")
    train_final = X_train_bal.copy()
    train_final["Employable"] = y_train_bal.values
    test_final = X_test.copy()
    test_final["Employable"] = y_test.values

    train_final.to_csv(f"{OUTPUT_DIR}/train_preprocessed.csv", index=False)
    test_final.to_csv(f"{OUTPUT_DIR}/test_preprocessed.csv", index=False)

    print(f"Final training set: {train_final.shape}")
    print(f"Final test set:     {test_final.shape}")
    print("\nSaved: train_preprocessed.csv, test_preprocessed.csv")
    print("\nPipeline completed successfully with no errors.")

    return train_final, test_final


if __name__ == "__main__":
    main()