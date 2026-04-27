---

## Recommendation System

The recommendation engine is **feature-driven and rule-based**. Instead of assigning fixed recommendations based on risk level, recommendations are generated based on feature values — so different hospitals with the same risk level receive different suggestions.

| Feature Signal | Recommendation |
|---|---|
| High `avg_err` | Improve discharge planning |
| High variability | Standardize workflows |
| High volume | Strengthen care coordination |

---

## Dashboard (Streamlit App)

The Streamlit dashboard provides:

- Hospital selection or manual input
- Risk prediction (probability + classification + tier)
- Explanation of contributing factors
- Tailored recommendations

**Run the app:**
```bash
streamlit run app/app.py
```

---

## Key Results

- All models achieved **ROC-AUC ~0.94–0.95**
- Minimal performance gap across models
- Feature engineering contributed significantly to model performance

---

## Key Insights

- Hospital readmission risk is strongly driven by **average ERR**, **performance variability**, and **discharge volume**
- Model complexity had limited impact compared to **feature quality**
- **Interpretability** is critical for real-world healthcare applications

---

## Limitations

- Uses hospital-level **aggregated** data
- Does not capture patient-level variability
- Recommendations are rule-based (not learned from outcomes)

---

## Future Work

- Incorporate patient-level data
- Develop data-driven recommendation learning
- Deploy system with scalable infrastructure

---

## Deliverables

- Streamlit dashboard (`app/app.py`)
- Trained model (`models/final_hospital_risk_model.pkl`)
- Project report and presentation (`reports/`)

---

## Author

**Srinija Jonnavithula**
M.S. Applied Data Science — University of Florida
