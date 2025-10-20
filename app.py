# ============================================================
# ☠️ Death-Phenotype / High-Risk Detector (7 variables) - Streamlit
# ============================================================
# Uses the 7-feature Isolation Forest model saved as 'scp_highrisk_detector.pkl'
# to classify patients as High-Risk (Death-Phenotype-like) vs Low-Risk
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------------------------------------------------
# Load trained model artefacts (7 variables bundle)
# ------------------------------------------------------------
st.set_page_config(page_title="High-Risk (Death-Phenotype) Detector", layout="centered")

try:
    bundle = joblib.load("scp_highrisk_detector.pkl")
    scaler       = bundle["scaler"]
    iso          = bundle["iso"]
    cols         = bundle["columns"]       # full training column order after encoding
    numeric_cols = bundle["numeric_cols"]  # ["LN ratio", "Size max (mm)", "LN pos"]
    threshold    = bundle["threshold"]     # 95th percentile IF_score used in training
except Exception as e:
    st.error("❌ Could not load 'scp_highrisk_detector.pkl'. Make sure it is in the same folder.")
    st.stop()

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("☠️ High-Risk (Death-Phenotype) Detector — 7 variables")
st.markdown("""
Estimate whether a patient’s profile matches a **High-Risk / Death-Phenotype-like** pattern  
using **7 features** derived from your training pipeline.
""")

col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Male gender (iSex)", ["No", "Yes"])                  # iSex → iSex_1
    rural = st.selectbox("Rural/small metropolitan hospital (iZona2)", ["No", "Yes"])  # iZona2 → iZona2_1
    typical = st.selectbox("Typical resection (iInt==1)", ["Yes", "No"])     # recoded to iInt2 (No→1)
    stage4 = st.selectbox("Stage IV (iStage)", ["No", "Yes"])                # iStage → iStage_4
with col2:
    ln_ratio = st.number_input("Lymph-node ratio", min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
    size_mm  = st.number_input("Tumor size (mm)", min_value=0.0, step=1.0, format="%.0f")
    ln_pos   = st.number_input("Lymph-nodes positive (count)", min_value=0.0, step=1.0, format="%.0f")

st.markdown("---")

# ------------------------------------------------------------
# Build model input (mirror the training preprocessing!)
# ------------------------------------------------------------
def build_encoded_row():
    # Raw (pre-encoding) fields exactly as in training
    patient_raw = {
        "iSex":            1 if sex == "Yes" else 0,
        "iInt2":           0 if typical == "Yes" else 1,  # iInt2=1 if NOT typical (iInt>1)
        "iZona2":          1 if rural == "Yes" else 0,
        "iStage":          4 if stage4 == "Yes" else 0,   # 4 to activate iStage_4 dummy; 0 otherwise
        "LN ratio":        float(ln_ratio),
        "Size max (mm)":   float(size_mm),
        "LN pos":          float(ln_pos),
    }
    X_new_base = pd.DataFrame([patient_raw])

    # One-hot encode like training
    X_new_enc = pd.get_dummies(
        X_new_base,
        columns=["iSex", "iInt2", "iZona2", "iStage"],
        drop_first=True
    )

    # Ensure all training columns exist (add missing dummy columns as 0)
    for c in cols:
        if c not in X_new_enc.columns:
            X_new_enc[c] = 0

    # Keep exact training column order
    X_new_enc = X_new_enc[cols]

    # Scale numeric columns using the fitted scaler
    X_new_enc[numeric_cols] = scaler.transform(X_new_enc[numeric_cols])

    return X_new_enc

# ------------------------------------------------------------
# Predict
# ------------------------------------------------------------
if st.button("Classify patient"):
    try:
        X_new_enc = build_encoded_row()
        if_score = float(-iso.decision_function(X_new_enc)[0])

        st.write(f"**Anomaly score (IF_score): {if_score:.3f}**")
        st.write(f"**High-Risk threshold (95th pct): {threshold:.3f}**")

        st.markdown("---")
        if if_score >= threshold:
            st.error("☠️ **HIGH-RISK / Death-Phenotype-like**")
            st.markdown("🔴 *Atypical clinical profile with high-risk of cancer death*")
        else:
            st.success("✅ **LOW-RISK**")
            st.markdown("🟢 *Typical clinical profile with low-risk of cancer death*")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
else:
    st.info("Insert values and press **Classify patient**")

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("""
---
**Model inputs (7 features):**  
- iSex → *Male gender*  
- iInt2 → *Typical resection (binary recode of iInt)*  
- iZona2 → *Rural/small metropolitan hospital*  
- iStage → *Stage IV dummy (iStage_4)*  
- LN ratio (scaled)  
- Size max (mm) (scaled)  
- LN pos (scaled)  

Patients above the **95th percentile** of Isolation Forest anomaly scores are labeled **High-Risk**.  
*For research and educational purposes only.*
""")

