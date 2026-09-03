"""
YuvaIntern Week 6 Capstone
Student Performance Prediction and Segmentation

Dataset: UCI Student Performance (Portuguese course), supplied as student-por.csv.
Pipeline: acquisition -> cleaning -> EDA -> feature engineering ->
Random Forest classification -> cross-validation -> K-Means clustering -> evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("student-por.csv", sep=";")

# Target engineering: Pass if final grade G3 >= 10, otherwise Fail.
df["performance"] = np.where(df["G3"] >= 10, "Pass", "Fail")

# Supervised learning: exclude G3 to avoid target leakage.
X = df.drop(columns=["G3", "performance"])
y = df["performance"]

categorical = X.select_dtypes(include="object").columns
numeric = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("numeric", StandardScaler(), numeric)
])

model = RandomForestClassifier(
    n_estimators=300, random_state=42, class_weight="balanced"
)
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred, pos_label="Pass"))
print("Recall:", recall_score(y_test, pred, pos_label="Pass"))
print("F1:", f1_score(y_test, pred, pos_label="Pass"))

cv = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
print("5-fold CV:", cv)
print("Mean CV accuracy:", cv.mean())

# Unsupervised learning: exclude grade columns so clusters describe student characteristics.
cluster_cols = [c for c in df.columns if c not in ["G1", "G2", "G3", "performance"]]
Xc = df[cluster_cols]
catc = Xc.select_dtypes(include="object").columns
numc = Xc.select_dtypes(exclude="object").columns

cluster_preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), catc),
    ("numeric", StandardScaler(), numc)
])
Z = cluster_preprocessor.fit_transform(Xc)

scores = {}
for k in range(2, 6):
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(Z)
    scores[k] = silhouette_score(Z, labels)

best_k = max(scores, key=scores.get)
final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
df["cluster"] = final_kmeans.fit_predict(Z)

print("Best k:", best_k)
print("Best silhouette:", scores[best_k])
print(df.groupby("cluster")[["G3", "G1", "G2", "studytime", "failures", "absences"]].mean())
