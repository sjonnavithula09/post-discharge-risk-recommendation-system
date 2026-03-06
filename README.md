# Post-Hospital Discharge Strategy Recommendation System

## Project Overview
This project develops a predictive framework to identify hospitals at high risk of excess 30-day readmissions using CMS Hospital Readmissions Reduction Program (HRRP) data.

The goal is to classify hospitals based on readmission performance and support strategy recommendations for improving hospital outcomes.

## Repository Structure
post-discharge-risk-recommendation-system
│
├── notebooks
│ ├── 01_prep_eda_features.ipynb
│ └── 02_modeling.ipynb
│
├── data
│ ├── raw
│ └── processed
│
├── reports
│
└── README.md

## Notebooks

### 01_prep_eda_features.ipynb
Performs data ingestion, cleaning, exploratory data analysis, and hospital-level feature engineering.

### 02_modeling.ipynb
Implements baseline machine learning models including Logistic Regression and Random Forest to classify hospitals into high-risk and low-risk groups.

## Modeling Approach

Target Variable:
High Risk = volume-weighted Excess Readmission Ratio > 1.0

Models Used:
- Logistic Regression
- Random Forest

Evaluation Metrics:
- ROC-AUC
- Balanced Accuracy
