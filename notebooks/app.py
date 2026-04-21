import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title="Post-Hospital Discharge Risk Recommendation System",
    page_icon="🏥",
    layout="wide"
)

# =========================================================
# Paths
# =========================================================
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

MODEL_PATH = PROJECT_ROOT / "models" / "final_hospital_risk_model.pkl"
FEATURES_PATH = PROJECT_ROOT / "models" / "model_features.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cms_hrrp_hospital_modeling_dataset.parquet"

# =========================================================
# Load artifacts
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_features():
    return joblib.load(FEATURES_PATH)

@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)
    df = df.dropna(subset=["volume_weighted_err"]).copy()
    df["high_risk"] = (df["volume_weighted_err"] > 1.0).astype(int)
    return df

model = load_model()
model_features = load_features()
df = load_data()

# =========================================================
# Helper functions
# =========================================================
def generate_recommendations(row: pd.Series) -> dict:
    operational = []
    monitoring = []
    care_coordination = []

    if row["avg_err"] > 1.0:
        operational.append(
            "Review discharge planning quality and strengthen patient transition protocols."
        )

    if row["err_std"] > 0.08:
        operational.append(
            "Reduce performance variability by standardizing readmission-related workflows across units."
        )
        monitoring.append(
            "Track unit-level variation more closely to identify inconsistent departments."
        )

    if row["measure_count"] < 4:
        monitoring.append(
            "Expand monitoring across additional quality measures to improve visibility into hospital performance."
        )

    if row["total_discharges"] > 10000:
        care_coordination.append(
            "High discharge volume suggests stronger follow-up coordination and transition support may be needed."
        )

    if row["stability_z"] > 1.0:
        monitoring.append(
            "Investigate unstable performance trends and monitor readmission behavior more frequently."
        )

    if not operational and not monitoring and not care_coordination:
        monitoring.append(
            "Current performance appears relatively stable. Maintain existing interventions and continue periodic monitoring."
        )

    return {
        "Operational Focus": operational,
        "Monitoring Focus": monitoring,
        "Care Coordination Focus": care_coordination
    }

def prepare_input_row(
    avg_err: float,
    avg_err_z: float,
    err_std: float,
    total_discharges: float,
    measure_count: float,
    stability_z: float
) -> pd.DataFrame:
    row_dict = {
        "avg_err": avg_err,
        "avg_err_z": avg_err_z,
        "err_std": err_std,
        "total_discharges": total_discharges,
        "measure_count": measure_count,
        "stability_z": stability_z
    }

    input_df = pd.DataFrame([row_dict])
    input_df = input_df.reindex(columns=model_features)
    return input_df

def get_risk_tier(prob: float) -> str:
    if prob >= 0.70:
        return "High Risk"
    elif prob >= 0.40:
        return "Moderate Risk"
    return "Low Risk"

def get_risk_explanation(prob: float) -> str:
    if prob >= 0.70:
        return (
            "High Risk means the hospital shows strong model-based evidence of elevated readmission risk "
            "relative to historical hospital-level performance patterns. This level suggests the hospital may "
            "require near-term intervention in discharge planning, monitoring, or care coordination."
        )
    elif prob >= 0.40:
        return (
            "Moderate Risk means the hospital does not appear critically elevated, but it shows enough warning "
            "signal to justify closer monitoring. This level may indicate developing instability or mixed "
            "performance across readmission-related features."
        )
    return (
        "Low Risk means the hospital appears relatively stable based on the model and historical feature patterns. "
        "This does not mean zero risk, but it suggests that current performance is closer to expected or manageable levels."
    )

def get_reason_flags(row: pd.Series, prob: float) -> list[str]:
    reasons = []

    if row["avg_err"] > 1.0:
        reasons.append("Average ERR is above the CMS benchmark, indicating excess readmission pressure.")

    if row["err_std"] > 0.08:
        reasons.append("ERR variability is elevated, suggesting inconsistent performance across measures or time.")

    if row["measure_count"] < 4:
        reasons.append("A lower measure count may reduce coverage of performance monitoring across conditions.")

    if row["total_discharges"] > 10000:
        reasons.append("High discharge volume increases operational burden and may complicate care transitions.")

    if row["stability_z"] > 1.0:
        reasons.append("The stability score suggests unstable readmission behavior that may need closer review.")

    if not reasons:
        if prob >= 0.70:
            reasons.append("Risk is driven by the combined pattern of hospital-level performance features rather than a single extreme factor.")
        elif prob >= 0.40:
            reasons.append("The model detects mild to moderate warning patterns across the input features.")
        else:
            reasons.append("The current feature pattern appears relatively stable and does not show strong high-risk signals.")

    return reasons

def assess_hospital(input_df: pd.DataFrame) -> dict:
    prob = float(model.predict_proba(input_df)[0, 1])
    pred = int(model.predict(input_df)[0])
    tier = get_risk_tier(prob)
    explanation = get_risk_explanation(prob)
    reasons = get_reason_flags(input_df.iloc[0], prob)
    recs = generate_recommendations(input_df.iloc[0])

    return {
        "risk_probability": prob,
        "risk_class": pred,
        "risk_tier": tier,
        "risk_explanation": explanation,
        "reasons": reasons,
        "recommendations": recs
    }

# =========================================================
# Header
# =========================================================
st.title("🏥 Post-Hospital Discharge Risk Recommendation System")
st.markdown(
    """
This dashboard assesses hospital readmission-related risk using engineered hospital-level performance features
and provides targeted recommendations to support improvement planning.
"""
)

# =========================================================
# Input mode on main page
# =========================================================
st.subheader("Choose Assessment Mode")

mode = st.radio(
    "How would you like to assess a hospital?",
    ["Select Existing Hospital", "Manual Input"],
    horizontal=True
)

selected_row = None

# =========================================================
# Hospital selection
# =========================================================
if mode == "Select Existing Hospital":
    st.subheader("Select a Hospital")

    hospital_options = (
        df["facility_name"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    selected_hospital = st.selectbox(
        "Choose a hospital from the processed dataset",
        hospital_options
    )

    selected_matches = df[df["facility_name"].astype(str) == selected_hospital]

    if selected_matches.empty:
        st.error("No matching hospital record found.")
        st.stop()

    selected_row = selected_matches.iloc[0]

    st.subheader("Selected Hospital Overview")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.write(f"**Hospital:** {selected_row['facility_name']}")
        st.write(f"**Facility ID:** {selected_row['facility_id']}")

    with c2:
        st.write(f"**State:** {selected_row['state']}")
        st.write(f"**Observed Risk Label:** {'High Risk' if selected_row['high_risk'] == 1 else 'Low Risk'}")

    with c3:
        st.write(f"**Volume-Weighted ERR:** {selected_row['volume_weighted_err']:.3f}")
        st.write(f"**Avg ERR:** {selected_row['avg_err']:.3f}")

    default_avg_err = float(selected_row["avg_err"])
    default_avg_err_z = float(selected_row["avg_err_z"])
    default_err_std = float(selected_row["err_std"])
    default_total_discharges = float(selected_row["total_discharges"])
    default_measure_count = float(selected_row["measure_count"])
    default_stability_z = float(selected_row["stability_z"])

else:
    default_avg_err = 1.00
    default_avg_err_z = 0.00
    default_err_std = 0.05
    default_total_discharges = 5000.0
    default_measure_count = 4.0
    default_stability_z = 0.00

# =========================================================
# Inputs
# =========================================================
st.subheader("Hospital Profile Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    avg_err = st.number_input(
        "Average ERR (avg_err)",
        min_value=0.0,
        value=default_avg_err,
        step=0.01,
        format="%.4f"
    )
    avg_err_z = st.number_input(
        "Average ERR Z-score (avg_err_z)",
        value=default_avg_err_z,
        step=0.01,
        format="%.4f"
    )

with col2:
    err_std = st.number_input(
        "ERR Variability (err_std)",
        min_value=0.0,
        value=default_err_std,
        step=0.01,
        format="%.4f"
    )
    total_discharges = st.number_input(
        "Total Discharges",
        min_value=0.0,
        value=default_total_discharges,
        step=100.0,
        format="%.0f"
    )

with col3:
    measure_count = st.number_input(
        "Measure Count",
        min_value=0.0,
        value=default_measure_count,
        step=1.0,
        format="%.0f"
    )
    stability_z = st.number_input(
        "Stability Z-score (stability_z)",
        value=default_stability_z,
        step=0.01,
        format="%.4f"
    )

input_df = prepare_input_row(
    avg_err=avg_err,
    avg_err_z=avg_err_z,
    err_std=err_std,
    total_discharges=total_discharges,
    measure_count=measure_count,
    stability_z=stability_z
)

# =========================================================
# Assessment
# =========================================================
assess_clicked = st.button("Run Risk Assessment")

if assess_clicked:
    result = assess_hospital(input_df)

    prob = result["risk_probability"]
    pred_class = result["risk_class"]
    tier = result["risk_tier"]

    st.markdown("---")
    st.subheader("Risk Assessment Summary")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Risk Probability", f"{prob:.3f}")
    with r2:
        st.metric("Model Risk Classification", "High Risk" if pred_class == 1 else "Low Risk")
    with r3:
        st.metric("Risk Tier", tier)

    if tier == "High Risk":
        st.error("This hospital shows elevated readmission-related risk.")
    elif tier == "Moderate Risk":
        st.warning("This hospital shows moderate readmission-related risk.")
    else:
        st.success("This hospital shows relatively low readmission-related risk.")

    # -----------------------------------------------------
    # Explanatory panel
    # -----------------------------------------------------
    st.subheader("What This Risk Level Means")
    st.info(result["risk_explanation"])

    st.subheader("Why the Hospital May Be Receiving This Risk Level")
    for reason in result["reasons"]:
        st.write(f"- {reason}")

    # -----------------------------------------------------
    # Recommendations
    # -----------------------------------------------------
    st.subheader("Recommended Next Steps")

    recommendations = result["recommendations"]

    for section_name, items in recommendations.items():
        if items:
            st.markdown(f"**{section_name}**")
            for item in items:
                st.write(f"- {item}")

    # -----------------------------------------------------
    # Input data used
    # -----------------------------------------------------
    st.subheader("Hospital Profile Used for Assessment")
    st.dataframe(input_df, use_container_width=True)

    # -----------------------------------------------------
    # Compare with actual record if hospital selected
    # -----------------------------------------------------
    if mode == "Select Existing Hospital" and selected_row is not None:
        st.subheader("Reference Hospital Record")
        reference_cols = [
            "facility_name",
            "state",
            "avg_err",
            "avg_err_z",
            "err_std",
            "total_discharges",
            "measure_count",
            "stability_z",
            "volume_weighted_err",
            "high_risk"
        ]
        reference_df = pd.DataFrame([selected_row[reference_cols]])
        st.dataframe(reference_df, use_container_width=True)

# =========================================================
# About section
# =========================================================
with st.expander("About this dashboard"):
    st.markdown(
        """
- **Primary function:** assess hospital readmission-related risk and recommend improvement actions  
- **Final model:** Tuned Random Forest  
- **Features used:** avg_err, avg_err_z, err_std, total_discharges, measure_count, stability_z  
- **Training risk definition:** High Risk = volume-weighted ERR > 1.0  
- **Recommendation logic:** feature-driven and rule-based, so hospitals with the same risk level may still receive different recommendations
"""
    )