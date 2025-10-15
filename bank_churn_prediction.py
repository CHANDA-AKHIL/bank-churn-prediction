# ============================================================
# Kaggle Playground Series - Season 4, Episode 1
# Binary Classification - Bank Customer Churn Prediction
# Logistic Regression with Visualizations
# Author: [Chanda Akhil]
# Description: This script predicts bank customer churn using Logistic Regression on Kaggle's synthetic dataset.
# Includes data preprocessing, model training, ROC-AUC evaluation, and visualizations.
# Dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# STEP 1: Load Data
# ============================================================
# Load datasets from 'data/' folder; download from https://kaggle.com/competitions/playground-series-s4e1/data
try:
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Please download datasets and place in 'data/' folder.")
    exit(1)

print(" Data loaded successfully!")
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# ============================================================
# STEP 2: Basic Cleaning
# ============================================================
id_col = 'id'
target_col = 'Exited'

X = train.drop([id_col, target_col], axis=1)
y = train[target_col]
test_ids = test[id_col]
X_test = test.drop(id_col, axis=1)

# ============================================================
# STEP 3: Safe Label Encoding (handles unseen labels)
# ============================================================
for col in X.columns:
    if X[col].dtype == 'object':
        combined = pd.concat([X[col], X_test[col]], axis=0)
        le = LabelEncoder()
        le.fit(combined.astype(str))
        X[col] = le.transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

print(" All categorical columns encoded successfully!")

# ============================================================
# STEP 4: Scaling
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# STEP 5: Train / Valid Split
# ============================================================
X_train, X_valid, y_train, y_valid = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# STEP 6: Train Logistic Regression Model
# ============================================================
model = LogisticRegression(
    solver='lbfgs', max_iter=1000, random_state=42, n_jobs=-1
)
print("Training Logistic Regression model...")
model.fit(X_train, y_train)

# ============================================================
# STEP 7: Validation Performance
# ============================================================
val_pred = model.predict_proba(X_valid)[:, 1]
val_auc = roc_auc_score(y_valid, val_pred)
print(f"Validation AUC: {val_auc:.5f}")

# ============================================================
# STEP 8: Final Predictions & Submission
# ============================================================
preds = model.predict_proba(X_test_scaled)[:, 1]

submission = pd.DataFrame({
    'id': test_ids,
    'Exited': preds
})
submission.to_csv('submission.csv', index=False)
print("submission.csv generated successfully!")
print(submission.head())

# ============================================================
# STEP 9: Visualizations
# ============================================================
# (A) Churn Status Distribution
plt.figure(figsize=(6, 5))
sns.countplot(x=target_col, data=train, palette=["#2E86AB", "#A23B72"])
plt.title("Churn Status Distribution", fontsize=16, fontweight='bold')
plt.xlabel("Churn Status (0=Stayed, 1=Exited)")
plt.ylabel("Count")
for container in plt.gca().containers:
    plt.gca().bar_label(container)
plt.tight_layout()
plt.savefig('visualizations/churn_status_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# (B) Correlation Heatmap
plt.figure(figsize=(10, 8))
numeric_cols = train.select_dtypes(include=np.number)
correlation = numeric_cols.corr()
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, center=0, linewidths=0.5, square=True)
plt.title("Correlation Heatmap", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap_churn.png', dpi=300, bbox_inches='tight')
plt.show()

# (C) ROC Curve with AUC
fpr, tpr, _ = roc_curve(y_valid, val_pred)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], '--', color='gray', label='Random Classifier')
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve - Bank Churn Prediction", fontsize=16, fontweight='bold')
plt.legend(fontsize=11, loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/roc_curve_churn.png', dpi=300, bbox_inches='tight')
plt.show()

# (D) Feature Importance (using Logistic Regression coefficients)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), palette="viridis")
plt.title("Top 10 Feature Importances (Logistic Regression)", fontsize=16, fontweight='bold')
plt.xlabel("Absolute Coefficient Value")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig('visualizations/feature_importance_churn.png', dpi=300, bbox_inches='tight')
plt.show()


print("All visualizations saved!")
