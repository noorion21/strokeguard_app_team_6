import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load pipeline ──────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return joblib.load("stroke_pipeline.pkl")

pipe = load_pipeline()

# ── Extract pipeline steps for SHAP ───────────────────────────────
preprocessor = pipe.named_steps["preprocessor"]
selector     = pipe.named_steps["selector"]
clf          = pipe.named_steps["clf"]

# Rebuild feature names after selection (same logic as notebook Cell 12)
NUMERIC_FEATURES     = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL_FEATURES = ["gender", "ever_married", "work_type",
                        "Residence_type", "smoking_status"]
BINARY_FEATURES      = ["hypertension", "heart_disease"]

try:
    raw_names = (
        NUMERIC_FEATURES
        + list(preprocessor.named_transformers_["cat"]["encoder"]
               .get_feature_names_out(CATEGORICAL_FEATURES))
        + BINARY_FEATURES
    )
    feature_names = [n for n, s in zip(raw_names, selector.support_) if s]
except Exception:
    feature_names = [f"feature_{i}" for i in range(selector.n_features_in_)]

explainer = shap.TreeExplainer(clf)

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="StrokeGuard ML",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 StrokeGuard ML")
st.caption("Stroke risk prediction — StrokeGuard ML (Team 6, MSc Machine Learning)")
st.markdown("---")

# ── Sidebar inputs ─────────────────────────────────────────────────
st.sidebar.header("Patient Information")

age              = st.sidebar.slider("Age", 1, 100, 55)
avg_glucose      = st.sidebar.number_input("Avg Glucose Level (mg/dL)", 50.0, 300.0, 100.0, step=0.5)
bmi              = st.sidebar.number_input("BMI", 10.0, 60.0, 28.0, step=0.1)
gender           = st.sidebar.selectbox("Gender", ["Male", "Female"])
ever_married     = st.sidebar.selectbox("Ever Married", ["Yes", "No"])
work_type        = st.sidebar.selectbox("Work Type",
                       ["Private", "Self-employed", "Govt_job",
                        "children", "Never_worked"])
residence        = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
smoking_status   = st.sidebar.selectbox("Smoking Status",
                       ["never smoked", "formerly smoked",
                        "smokes", "Unknown"])
hypertension     = st.sidebar.selectbox("Hypertension", ["No", "Yes"])
heart_disease    = st.sidebar.selectbox("Heart Disease", ["No", "Yes"])

predict_btn = st.sidebar.button("🔍 Predict Stroke Risk", use_container_width=True)

# ── Prediction ─────────────────────────────────────────────────────
if predict_btn:
    patient = pd.DataFrame([{
        "age":               age,
        "avg_glucose_level": avg_glucose,
        "bmi":               bmi,
        "gender":            gender,
        "ever_married":      ever_married,
        "work_type":         work_type,
        "Residence_type":    residence,
        "smoking_status":    smoking_status,
        "hypertension":      int(hypertension == "Yes"),
        "heart_disease":     int(heart_disease == "Yes"),
    }])

    proba = pipe.predict_proba(patient)[0, 1]

    # ── Risk display ───────────────────────────────────────────────
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Stroke Probability")
        if proba >= 0.5:
            st.error(f"## {proba:.1%}")
            st.error("⚠️ HIGH RISK — recommend clinical follow-up")
        elif proba >= 0.2:
            st.warning(f"## {proba:.1%}")
            st.warning("🟡 MODERATE RISK — monitor closely")
        else:
            st.success(f"## {proba:.1%}")
            st.success("✅ LOW RISK")

        st.metric("Random baseline", "4.87%")
        st.caption("Primary metric: PR-AUC. Model: XGBoost (scale_pos_weight=23, recall=96%)")

    # ── SHAP waterfall ─────────────────────────────────────────────
    with col2:
        st.subheader("SHAP Explanation — Why this prediction?")

        # Preprocess → select (no SMOTE at inference)
        X_pre = preprocessor.transform(patient)
        X_sel = selector.transform(X_pre)
        X_shap = pd.DataFrame(X_sel, columns=feature_names)

        shap_vals = explainer(X_shap)

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(shap_vals[0], max_display=10, show=False)
        plt.title(
            f"Patient risk factors  |  P(stroke) = {proba:.3f}",
            fontsize=12, pad=10
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.caption(
            "Red bars push probability UP (toward stroke). "
            "Blue bars push probability DOWN. "
            "Bar length = magnitude of contribution."
        )

# ── Default state ──────────────────────────────────────────────────
else:
    st.info("👈 Fill in patient details in the sidebar and click **Predict Stroke Risk**.")
    st.markdown("""
    **About this tool**
    - Model: XGBoost trained on the Stroke Prediction Dataset (fedesoriano, Kaggle)  
    - Pipeline: KNN Imputation → Quantile Transform → SMOTE → RFECV → XGBoost  
    - Test recall: **96%** (48/50 stroke cases correctly flagged)  
    - Test PR-AUC: **0.1895** vs random baseline 0.0489  
    - Interpretability: SHAP TreeExplainer waterfall per patient  

    ⚠️ *This tool is for educational purposes only and not a clinical diagnostic.*
    """)