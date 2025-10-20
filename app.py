# ============================================================
# ☠️ High-Risk Detector for Solid Pseudopapillary Tumor - Streamlit (TOP-10)
# ============================================================
# Uses the 10-feature Isolation Forest model saved as 'scp_highrisk_detector.pkl'
# to classify patients as High-Risk (Death-Phenotype-like) vs Low-Risk
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------------------------------------------------
# Load trained model artefacts (TOP-10 bundle)
# ------------------------------------------------------------
st.set_page_config(page_title="High-Risk Detector for Solid Pseudopapillary Tumor", layout="centered")

try:
    bundle = joblib.load("scp_highrisk_detector.pkl")
    scaler       = bundle["scaler"]
    iso          = bundle["iso"]
    cols         = bundle["columns"]       # full training column order after encoding (TOP-10)
    numeric_cols = bundle["numeric_cols"]  # ["LN ratio","Size max (mm)","LN pos","LN esaminati"]
    threshold    = bundle["threshold"]     # 95th percentile IF_score used in training
except Exception as e:
    st.error("❌ Could not load 'scp_highrisk_detector.pkl'. Make sure it is in the same folder.")
    st.stop()

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("☠️ High-Risk Detector for Solid Pseudopapillary Tumor — TOP-10 features")
st.markdown("""
Estimate whether a patient’s profile matches a **High-Risk / Death-Phenotype-like** pattern  
using **10 features** derived from the training pipeline.
""")

col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Male gender", ["No", "Yes"])  # iSex → iSex_1
    rural = st.selectbox("Rural/small metropolitan hospital", ["No", "Yes"])  # iZona2 → iZona2_1
    # Default = "No"; Yes => iInt2=1 (risk ↑), No => iInt2=0 (risk ↓)
    typical = st.selectbox("Typical resection", ["No", "Yes"], index=0)
    stage4 = st.selectbox("Stage IV", ["No", "Yes"])  # iStage → iStage_4
with col2:
    ln_ratio = st.number_input("Lymph-node ratio", min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
    size_mm  = st.number_input("Tumor size (mm)", min_value=0.0, step=1.0, format="%.0f")
    ln_pos   = st.number_input("Lymph-nodes positive (count)", min_value=0.0, step=1.0, format="%.0f")
    ln_exam  = st.number_input("Lymph-nodes examined (count)", min_value=0.0, step=1.0, format="%.0f")

# Age decade per dummies 7–8 (coerente con training: iEtaDecade ∈ {1..8})
age_decade = st.selectbox(
    "Age decade (training encoding 1–8)",
    ["1","2","3","4","5","6","7","8"],
    index=6  # default "7"
)

st.markdown("---")

# ------------------------------------------------------------
# Build model input (mirror the training preprocessing!)
# ------------------------------------------------------------
def build_encoded_row():
    # Raw (pre-encoding) fields exactly as in training (TOP-10)
    patient_raw = {
        "iSex":            1 if sex == "Yes" else 0,
        "iInt2":           1 if typical == "Yes" else 0,  # YES → higher risk (iInt2=1); NO → iInt2=0
        "iZona2":          1 if rural == "Yes" else 0,
        "iStage":          4 if stage4 == "Yes" else 0,   # 4 to activate iStage_4 dummy; 0 otherwise
        "iEtaDecade":      int(age_decade),               # genera dummies iEtaDecade_7 e _8 con drop_first=True
        "LN ratio":        float(ln_ratio),
        "Size max (mm)":   float(size_mm),
        "LN pos":          float(ln_pos),
        "LN esaminati":    float(ln_exam),
    }
    X_new_base = pd.DataFrame([patient_raw])

    # One-hot encode like training (TOP-10)
    X_new_enc = pd.get_dummies(
        X_new_base,
        columns=["iSex", "iInt2", "iZona2", "iStage", "iEtaDecade"],
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
            st.markdown("🔴 *Atypical clinical profile with high risk of cancer-specific death*")
        else:
            st.success("✅ **LOW-RISK**")
            st.markdown("🟢 *Profile not indicative of death-phenotype (low anomaly score)*")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
else:
    st.info("Insert values and press **Classify patient**")

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("""
---
### 🧠 Model information
This prototype was developed using data from the **SEER (Surveillance, Epidemiology, and End Results) database**,  
applying a **machine learning anomaly-detection approach** (Isolation Forest)  
to identify patients whose clinical profiles resemble a *death-phenotype* pattern.

Patients with anomaly scores above the **95th percentile** of the Isolation Forest distribution  
are labeled as **High-Risk / Death-Phenotype-like**.

**Notes.**
- *Typical resection* includes **pancreaticoduodenectomy, left pancreatectomy, and total pancreatectomy**.  
- The model here uses **10 features**: LN ratio, iZona2 (rural/small metro), iSex (male), iInt2 (typical resection recode),
  iStage (Stage IV), tumor size (mm), LN positive (count), **LN examined (count)**,
  and **age-decade dummies** (7 and 8) generated from `iEtaDecade`.

---

**Developed by [Claudio Ricci, MD, PhD](mailto:claudio.ricci@unibo.it)**  
*Associate Professor of Surgery — University of Bologna*  
*IRCCS Azienda Ospedaliero–Universitaria di Bologna (S. Orsola–Malpighi Hospital)*  

© 2025 — For research and educational purposes only.
""")

