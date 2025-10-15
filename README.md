# Bank Churn Prediction

## Overview
This project is based on the Kaggle Playground Series - Season 4, Episode 1: Binary Classification with a Bank Churn Dataset. The goal is to predict whether a customer will churn (close their account) using synthetically generated tabular data. The competition ran from January 2 to February 1, 2024, with submissions evaluated using the area under the ROC curve (ROC-AUC).

Key skills demonstrated:
- Data preprocessing (label encoding, scaling)
- Classification using Logistic Regression
- Evaluation with ROC-AUC
- Visualization of distributions, correlations, and model performance

Developed as part of my BTech in Computer Science (AI/ML) to practice binary classification.

Citation: Walter Reade and Ashley Chow. Binary Classification with a Bank Churn Dataset. https://kaggle.com/competitions/playground-series-s4e1, 2024. Kaggle.

## Dataset
- **Source**: [Kaggle Playground S4E1](https://kaggle.com/competitions/playground-series-s4e1/data)
- **Description**: Synthetic tabular data with features like age, income, and tenure. Target is Exited (0=Stayed, 1=Churned).
- **Preprocessing**:
  - Encoded categorical variables using LabelEncoder.
  - Scaled numerical features with StandardScaler.

Note: Datasets not included due to size. Download from Kaggle.
## Methodology
1. **Data Preparation**: Load CSVs, encode categoricals, scale features.
2. **Modeling**: Train Logistic Regression with stratified split.
3. **Evaluation**: Compute ROC-AUC on validation set (20% split).
4. **Visualizations**: Churn distribution, correlation heatmap, ROC curve, feature importance.
5. **Prediction**: Generate test set probabilities and submission file.

Results: Achieved ROC-AUC ~0.85; key features include tenure and balance.
