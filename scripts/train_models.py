import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from catboost import CatBoostClassifier, CatBoostRegressor

BASE = Path(__file__).resolve().parent.parent

# ── 1. Load dataset ──────────────────────────────────────────────────────────
df = pd.read_csv(BASE / "data" / "Student Social Media And Mental Health Impact.csv")

# ── 3 & 4. Feature matrix & targets ─────────────────────────────────────────
target_class = "Stress_Level"
target_reg = "Mental_Health_Score"

X = df.drop(columns=[target_class, target_reg])
y_class = df[target_class]
y_reg = df[target_reg]

# ── 2. Identify categorical columns (from X only) ────────────────────────────
cat_features = [col for col in X.columns if X[col].dtype in ("object", "string")]
print(f"Categorical features detected ({len(cat_features)}): {cat_features}")

# ── 5. Train/test split (80/20) ─────────────────────────────────────────────
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = (
    train_test_split(X, y_class, y_reg, test_size=0.20, random_state=42)
)

print(f"\nTraining samples: {X_train.shape[0]}  |  Test samples: {X_test.shape[0]}")

# ── 6 & 7. CatBoost Classifier ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  CLASSIFICATION — Predicting Stress_Level")
print("=" * 60)

clf = CatBoostClassifier(cat_features=cat_features, verbose=0, random_seed=42)
clf.fit(X_train, y_class_train)

y_class_pred = clf.predict(X_test)

acc = accuracy_score(y_class_test, y_class_pred)
print(f"\nAccuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_class_test, y_class_pred))

# ── 8 & 9. CatBoost Regressor ───────────────────────────────────────────────
print("=" * 60)
print("  REGRESSION — Predicting Mental_Health_Score")
print("=" * 60)

reg = CatBoostRegressor(cat_features=cat_features, verbose=0, random_seed=42)
reg.fit(X_train, y_reg_train)

y_reg_pred = reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
mae = mean_absolute_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)

print(f"\nRMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")
print("=" * 60)
