"""
YuvaIntern – Virtual Data Science with Python Trainee
Task 2: Exploratory Data Analysis and Visualization

Dataset: Palmer Penguins
Official public source:
https://github.com/allisonhorst/palmerpenguins
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the complete public dataset
url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
df = pd.read_csv(url)

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())

print("\nDescriptive statistics:")
print(df.describe(include="all"))

# EDA working copy
eda = df.copy()

# Keep categorical missing values visible
eda["sex"] = eda["sex"].fillna("Unknown")

# For numerical plotting that requires complete body mass,
# use median imputation only in the EDA working copy.
eda["body_mass_g"] = eda["body_mass_g"].fillna(
    eda["body_mass_g"].median()
)

# -------------------------------------------------
# 1. Species distribution
# -------------------------------------------------
plt.figure(figsize=(8, 5))
sns.countplot(data=eda, x="species")
plt.title("Penguin Records by Species")
plt.xlabel("Species")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig("species_distribution.png", dpi=200)
plt.show()

# -------------------------------------------------
# 2. Body-mass distribution
# -------------------------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(
    data=eda,
    x="body_mass_g",
    hue="species",
    kde=True,
    element="step"
)
plt.title("Body Mass Distribution by Species")
plt.xlabel("Body Mass (g)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("body_mass_distribution.png", dpi=200)
plt.show()

# -------------------------------------------------
# 3. Flipper length vs body mass
# -------------------------------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=eda,
    x="flipper_length_mm",
    y="body_mass_g",
    hue="species",
    style="sex"
)
plt.title("Flipper Length vs Body Mass")
plt.xlabel("Flipper Length (mm)")
plt.ylabel("Body Mass (g)")
plt.legend(title="Species / Sex")
plt.tight_layout()
plt.savefig("flipper_vs_body_mass.png", dpi=200)
plt.show()

# -------------------------------------------------
# 4. Bill length vs bill depth
# -------------------------------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=eda,
    x="bill_length_mm",
    y="bill_depth_mm",
    hue="species"
)
plt.title("Bill Length vs Bill Depth")
plt.xlabel("Bill Length (mm)")
plt.ylabel("Bill Depth (mm)")
plt.legend(title="Species")
plt.tight_layout()
plt.savefig("bill_length_vs_depth.png", dpi=200)
plt.show()

# -------------------------------------------------
# 5. Correlation matrix
# -------------------------------------------------
numeric_cols = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]

corr = eda[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr,
    annot=True,
    cmap="viridis",
    vmin=-1,
    vmax=1
)
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=200)
plt.show()

# -------------------------------------------------
# 6. Species-level aggregation
# -------------------------------------------------
species_summary = (
    eda.groupby("species")[numeric_cols]
       .mean()
       .round(2)
)

print("\nMean measurements by species:")
print(species_summary)

# -------------------------------------------------
# 7. Island/species aggregation
# -------------------------------------------------
island_species = pd.crosstab(
    eda["island"],
    eda["species"]
)

print("\nSpecies counts by island:")
print(island_species)

# -------------------------------------------------
# 8. IQR outlier review
# -------------------------------------------------
print("\nPotential IQR outliers:")

for col in numeric_cols:
    q1 = eda[col].quantile(0.25)
    q3 = eda[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    count = (
        (eda[col] < lower) |
        (eda[col] > upper)
    ).sum()

    print(
        f"{col}: {int(count)} "
        f"(bounds: {lower:.2f} to {upper:.2f})"
    )

# Save outputs
eda.to_csv("penguins_eda_task2_cleaned.csv", index=False)
species_summary.to_csv("penguins_species_summary_task2.csv")
corr.to_csv("penguins_correlation_matrix_task2.csv")

print("\nEDA complete. Output files saved.")
