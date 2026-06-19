import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# ── 1. Load dataset ──────────────────────────────────────────────────────────
df = pd.read_csv(BASE / "data" / "Student Social Media And Mental Health Impact.csv")

# ── 2. Shape & missing values ───────────────────────────────────────────────
print(f"Dataset shape: {df.shape}")
print("\nMissing values per column:")
print(df.isnull().sum())

# ── 3. Target variable distributions ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Mental_Health_Score histogram
axes[0].hist(df["Mental_Health_Score"], bins=20, color="steelblue", edgecolor="white")
axes[0].set_title("Distribution of Mental Health Score")
axes[0].set_xlabel("Mental Health Score")
axes[0].set_ylabel("Frequency")

# Stress_Level bar chart
stress_counts = df["Stress_Level"].value_counts().sort_index()
axes[1].bar(stress_counts.index.astype(str), stress_counts.values, color="coral", edgecolor="white")
axes[1].set_title("Distribution of Stress Level")
axes[1].set_xlabel("Stress Level")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig(BASE / "outputs" / "target_distributions.png", dpi=150)
plt.close()
print(f"\nSaved: {BASE / 'outputs' / 'target_distributions.png'}")

# ── 4. Correlation heatmap ──────────────────────────────────────────────────
numeric_cols = ["Age", "Avg_Daily_Usage_Hours", "Study_Hours",
                "Sleep_Hours_Per_Night", "Mental_Health_Score"]
available_cols = [c for c in numeric_cols if c in df.columns]

corr = df[available_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Variables")
plt.tight_layout()
plt.savefig(BASE / "outputs" / "correlation_heatmap.png", dpi=150)
plt.close()
print(f"Saved: {BASE / 'outputs' / 'correlation_heatmap.png'}")
