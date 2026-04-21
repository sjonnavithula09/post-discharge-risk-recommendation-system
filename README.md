Post-Hospital Discharge Risk Recommendation System
Project Overview

This project develops an end-to-end risk assessment and recommendation system to identify hospitals at high risk of excess 30-day readmissions using CMS Hospital Readmissions Reduction Program (HRRP) data.

The system goes beyond prediction by combining:

Risk classification
Feature-based explanation
Actionable recommendations

The final output is an interactive Streamlit dashboard that allows users to assess hospital performance and receive targeted improvement strategies.

Objectives
Classify hospitals into high-risk and low-risk categories
Interpret key drivers of hospital readmission performance
Provide feature-driven recommendations for improvement
Build an interactive dashboard for real-time decision support
Key Concept

High Risk = Volume-weighted Excess Readmission Ratio (ERR) > 1.0
(ERR > 1 indicates excess readmissions compared to CMS benchmarks)

Repository Structure
post-discharge-risk-recommendation-system
│
├── README.md
│
├── notebooks
│   ├── 01_prep_eda_features.ipynb
│   ├── 02_baseline_modeling.ipynb
│   └── 03_modeling_final.ipynb
│
├── data
│   ├── raw
│   └── processed
│
├── models
│   ├── final_hospital_risk_model.pkl
│   └── model_features.pkl
│
├── app
│   └── app.py
│
└── reports
    ├── poster.pdf
    └── presentation.pdf
Notebooks
🔹 01_prep_eda_features.ipynb
Data ingestion and cleaning
Exploratory Data Analysis (EDA)
Handling missing values
Feature engineering at hospital level
🔹 02_baseline_modeling.ipynb
Baseline model implementation
Logistic Regression and initial Random Forest
Initial evaluation and comparison
🔹 03_modeling_final.ipynb
Advanced modeling and comparison
Hyperparameter tuning
Cross-validation
Final model selection
Feature Engineering

The model uses engineered hospital-level features:

avg_err → Average readmission ratio
err_std → Variability across conditions
total_discharges → Hospital volume
measure_count → Number of measures reported
stability_z → Performance stability

These features capture both performance and operational characteristics of hospitals.

Modeling Approach
Models Evaluated:
Logistic Regression
Random Forest
Gradient Boosting
XGBoost
Evaluation Metrics:
ROC-AUC
Accuracy
F1 Score
Cross-validation performance
Final Model:

Tuned Random Forest

Reason for selection:

Stable performance across folds
Handles non-linear feature interactions
Comparable accuracy with better generalization
System Architecture
Data → Feature Engineering → Model → Risk Assessment → Explanation → Recommendation Engine → Dashboard
Recommendation System

The recommendation engine is feature-driven and rule-based.

Instead of assigning fixed recommendations based on risk level:

Recommendations are generated based on feature values
Different hospitals with the same risk level receive different suggestions
Examples:
High avg_err → Improve discharge planning
High variability → Standardize workflows
High volume → Strengthen care coordination
Dashboard (Streamlit App)

The Streamlit dashboard provides:

Hospital selection or manual input
Risk prediction (probability + classification + tier)
Explanation of contributing factors
Tailored recommendations
Run the app:
streamlit run app/app.py
Key Results
All models achieved ROC-AUC ~0.94–0.95
Minimal performance gap across models
Feature engineering contributed significantly to model performance
Key Insights
Hospital readmission risk is strongly driven by:
Average ERR
Variability in performance
Discharge volume
Model complexity had limited impact compared to feature quality
Interpretability is critical for real-world healthcare applications
Limitations
Uses hospital-level aggregated data
Does not capture patient-level variability
Recommendations are rule-based (not learned from outcomes)
Future Work
Incorporate patient-level data
Develop data-driven recommendation learning
Deploy system with scalable infrastructure
Deliverables
Streamlit dashboard (app/app.py)

Author

Srinija Jonnavithula
M.S. Applied Data Science
University of Florida

Final Note

This project transforms predictive modeling into a decision-support system by combining risk assessment, interpretability, and actionable recommendations.
