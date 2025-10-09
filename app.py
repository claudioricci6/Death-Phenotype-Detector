# ============================================================
# ☠️ Death-Phenotype Detector (Streamlit deploy version)
# ============================================================
# Uses pre-trained Isolation Forest model (scp_death_detector.pkl)
# to classify patients as "Death Phenotype" vs "Not Death Phenotype"
# ============================================================

import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------
# Load trained model artefacts
# ------------------------------------------------------------
try:
    bundle = joblib.load("scp_death_detector.pkl")
    scaler = bundle["scaler"]
    iso = bundle["iso"]
    cols = bundle["columns"]
    threshold = bundle["threshold"]
except Exception as e:
    st.error("❌ Could not load model file 'scp_death_detector.pkl'. Please ensure it is present in the same directory.")
    st.stop()

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="Death-Phenotype Detector", layout="centered")

st.title("☠️ Death-Phenotype Detector")
st.markdown("""
Identify whether a patient's clinical profile corresponds to the **Death-Phenotype**,  
based on four SHAP-identified baseline variables.
""")

# ------------------------------------------------------------
# Input form
# ------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Male gender", ["No", "Yes"])
    tp = st.selectbox("Total pancreatectomy", ["No", "Yes"])
with col2:
    rural = st.selectbox("Rural or small metropolitan hospital", ["No", "Yes"])
    ln_ratio = st.number_input("Lymph-node ratio", min_value=0.0, max_value=1.0, step=0.01, format="%.2f")

# Create model input
patient = {
    "iSex": 1 if sex == "Yes" else 0,
    "LN ratio": ln_ratio,
    "iInt_TP": 1 if tp == "Yes" else 0,
    "iZona2": 1 if rural == "Yes" else 0
}

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
if st.button("Classify patient"):
    try:
        X_new = pd.DataFrame([patient])[cols]
        X_new["LN ratio"] = scaler.transform(X_new[["LN ratio"]])
        if_score = float(-iso.decision_function(X_new)[0])

        st.markdown("---")
        st.write(f"**Anomaly score (IF_score): {if_score:.3f}**")
        st.write(f"**Threshold for Death-Phenotype:** {threshold:.3f}")

        if if_score >= threshold:
            st.error("☠️ **This patient is classified as DEATH PHENOTYPE**")
            st.markdown("🔴 *Highly anomalous profile resembling cancer-specific deaths*")
        else:
            st.success("✅ **This patient is NOT a death phenotype**")
            st.markdown("🟢 *Typical clinical profile, low anomaly score*")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

else:
    st.info("Insert values and press *Classify patient*")

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("""
---
*Model based on 4 SHAP-identified baseline variables: male gender, total pancreatectomy, lymph-node ratio, and rural/small-metropolitan hospital.*  
*Patients exceeding the 95th percentile of Isolation Forest anomaly scores are labeled as “Death-Phenotype”.*  
*For research and educational purposes only.*
""")
