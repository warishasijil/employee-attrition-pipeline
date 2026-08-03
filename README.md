# Employee Attrition Prediction Pipeline

[![CI](https://github.com/warishasijil/employee-attrition-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/warishasijil/employee-attrition-pipeline/actions/workflows/ci.yml)

A modular, object-oriented, and production-oriented data science pipeline for predicting employee attrition.

The project covers the complete machine learning lifecycle, including data loading, validation, cleaning, exploratory analysis, feature engineering, preprocessing, model training, hyperparameter tuning, model comparison, model persistence, inference, testing, and continuous integration.

## Project Objective

The objective is to predict whether an employee is likely to leave an organisation using demographic, compensation, career, and workplace-related information.

This is a binary classification problem:

- `0`: No attrition
- `1`: Attrition

The dataset is imbalanced, so model selection is based primarily on ROC-AUC, recall, F1-score, and PR-AUC rather than accuracy alone.

## Dataset

This project uses the IBM HR Analytics Employee Attrition & Performance dataset.

Dataset characteristics:

- 1,470 employee records
- 35 original columns
- Binary target: `Attrition`
- 237 attrition cases
- 1,233 non-attrition cases
- No missing values in the original dataset
- No duplicate rows

Download the dataset from:

https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Place the file here:

```text
data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv