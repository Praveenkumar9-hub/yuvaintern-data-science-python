"""
YuvaIntern Task 5 - Deep Learning Application in Data Science

Project: Handwritten Digit Recognition
Framework-style approach: Neural Network / MLP using scikit-learn.
Dataset: Public sklearn Digits dataset (8x8 grayscale handwritten digits).
"""

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# 1. Load public handwritten-digit dataset
digits = load_digits()
X = digits.data / 16.0       # normalize pixel values from 0-16 to 0-1
y = digits.target

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Build neural network
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=32,
    learning_rate_init=0.001,
    max_iter=100,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=8,
    random_state=42
)

# 4. Train
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Weighted Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Weighted Recall:", recall_score(y_test, y_pred, average="weighted"))
print("Weighted F1:", f1_score(y_test, y_pred, average="weighted"))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nTraining iterations:", model.n_iter_)
