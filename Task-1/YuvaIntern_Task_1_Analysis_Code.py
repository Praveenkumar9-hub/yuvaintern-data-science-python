"""
YuvaIntern - Virtual Data Science with Python Trainee
Task 1: Data Acquisition, Cleaning, and Preprocessing

Dataset: UCI Adult (Census Income) Dataset
Source: UCI Machine Learning Repository
DOI: https://doi.org/10.24432/C5XW20
"""

# 1. Install once if required:
# pip install ucimlrepo pandas numpy matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# -------------------------
# A. DATA ACQUISITION
# -------------------------
adult = fetch_ucirepo(id=2)

X = adult.data.features.copy()
y = adult.data.targets.copy()

# Combine features and target for the cleaning workflow
df = pd.concat([X, y], axis=1)

# Standardize target column name if necessary
if len(df.columns) > 14:
    df = df.rename(columns={df.columns[-1]: "income"})

print("Dataset shape:", df.shape)
print(df.head())
print(df.info())

# -------------------------
# B. INITIAL DATA QUALITY CHECK
# -------------------------
print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False))

print("\nDuplicate rows:", df.duplicated().sum())

print("\nUnique values:")
print(df.nunique().sort_values())

print("\nNumeric summary:")
print(df.describe().T)

# -------------------------
# C. NORMALIZE TEXT FIELDS
# -------------------------
cat_cols = df.select_dtypes(include=["object", "category"]).columns

for col in cat_cols:
    df[col] = df[col].astype("string").str.strip()

# UCI Adult may expose missing values as '?' depending on the import method.
df = df.replace({"?": pd.NA, "": pd.NA, "NA": pd.NA, "N/A": pd.NA})

print("\nMissing values after standardization:")
print(df.isna().sum().sort_values(ascending=False))

# -------------------------
# D. HANDLE MISSING VALUES
# -------------------------
# Categorical columns are filled with an explicit "Unknown" category.
# This avoids silently deleting a large number of observations.
categorical_missing_cols = [
    c for c in ["workclass", "occupation", "native-country"]
    if c in df.columns
]

for col in categorical_missing_cols:
    df[col] = df[col].fillna("Unknown")

# Numeric columns: use median imputation if any numeric missing values exist.
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# -------------------------
# E. REMOVE EXACT DUPLICATES
# -------------------------
before_duplicates = len(df)
df = df.drop_duplicates().reset_index(drop=True)
removed_duplicates = before_duplicates - len(df)
print("\nExact duplicate rows removed:", removed_duplicates)

# -------------------------
# F. RANGE / CONSISTENCY CHECKS
# -------------------------
# Values outside these domain ranges are treated as invalid.
range_rules = {
    "age": (17, 90),
    "education-num": (1, 16),
    "hours-per-week": (1, 99),
}

for col, (low, high) in range_rules.items():
    if col in df.columns:
        invalid = ~df[col].between(low, high)
        print(f"{col}: invalid records =", int(invalid.sum()))
        # Do not delete automatically; inspect before deciding.

# -------------------------
# G. OUTLIER REVIEW USING IQR
# -------------------------
numeric_cols = df.select_dtypes(include=np.number).columns

outlier_report = []
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_report.append([col, q1, q3, lower, upper, int(count)])

outlier_df = pd.DataFrame(
    outlier_report,
    columns=["column", "Q1", "Q3", "lower_bound", "upper_bound", "outlier_count"]
)

print("\nIQR outlier report:")
print(outlier_df)

# Important: statistically detected outliers are not automatically deleted.
# Values such as high capital gains/losses can be legitimate observations.

# -------------------------
# H. FINAL VALIDATION
# -------------------------
print("\nFinal shape:", df.shape)
print("\nRemaining missing values:")
print(df.isna().sum().sort_values(ascending=False))

print("\nFinal data types:")
print(df.dtypes)

# -------------------------
# I. SAVE CLEANED DATA
# -------------------------
df.to_csv("adult_cleaned_task1.csv", index=False)
outlier_df.to_csv("adult_outlier_report_task1.csv", index=False)

print("\nSaved: adult_cleaned_task1.csv")
print("Saved: adult_outlier_report_task1.csv")
