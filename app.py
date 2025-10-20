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
    cols         = bundle["columns"]       # training column order after encoding (TOP-10)
    numeric_cols = bundle["numeric_cols"]  # ["LN ratio","Size max (mm)","LN pos","LN esaminati"]
    threshold    = bundle["threshold"]     # 95th percentile IF_score used in training
except Exception as e:
    st.error("❌ Could not load 'scp_highrisk_detector.pkl'. Make sure it is in the same folder.")
    st.stop()

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("☠️ High-Risk Detector for Solid Pseudopapillary Tumor")
st.markdown("""
Estimate whether a patient’s profile matches a **High-Risk / Death-Phenotype-like** pattern  
using **10 features** derived from the training pipeline.
""")

col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Male gender", ["No", "Yes"])                 # iSex → iSex_1
    rural = st.selectbox("Rural/small metropolitan hospital", ["No", "Yes"])  # iZona2 → iZona2_1
    typical = st.selectbox("Typical resection", ["No", "Yes"], index=0)       # Yes ⇒ iInt2=1 (risk ↑)
    stage4 = st.selectbox("Stage IV", ["No", "Yes"])                  # iStage → iStage_4
with col2:
    size_mm  = st.number_input("Tumor size (mm)", min_value=0.0, step=1.0, format="%.0f")
    ln_pos   = st.number_input("Lymph-nodes positive (count)", min_value=0.0, step=1.0, format="%.0f")
    ln_exam  = st.number_input("Lymph-nodes examined (count)", min_value=0.0, step=1.0, format="%.0f")

# Age flags coerenti col training: baseline <60 (nessun dummy), 60–69 ⇒ iEtaDecade_7=1, ≥70 ⇒ iEtaDecade_8=1
age_flag = st.selectbox("Age category", ["<60 yrs", "60–69 yrs", "≥70 yrs"], index=0)

# --- Compute LN ratio automatically ---
if ln_exam == 0:
    ln_ratio_calc = 0.0
    if ln_pos > 0:
        st.warning("LN examined is 0: lymph-node ratio set to 0.0.")
else:
    if ln_pos > ln_exam:
        st.warning("Lymph-nodes positive exceeds nodes examined. Capping ratio at 1.0 and using LN pos = LN examined for ratio.")
    ln_ratio_calc = min(max(ln_pos, 0.0), ln_exam) / ln_exam  # bound to [0,1]

st.info(f"**Computed lymph-node ratio:** {ln_ratio_calc:.2f}")

st.markdown("---")

# ------------------------------------------------------------
# Build model input (mirror the training preprocessing!)
# ------------------------------------------------------------
def build_encoded_row():
    # Raw (pre-encoding) fields esattamente come in training (TOP-10)
    patient_raw = {
        "iSex":            1 if sex == "Yes" else 0,
        "iInt2":           1 if typical == "Yes" else 0,  # YES → higher risk (iInt2=1)
        "iZona2":          1 if rural == "Yes" else 0,
        "iStage":          4 if stage4 == "Yes" else 0,   # 4 to activate iStage_4 dummy; 0 otherwise
        "LN ratio":        float(ln_ratio_calc),          # <-- auto-computed
        "Size max (mm)":   float(size_mm),
        "LN pos":          float(ln_pos),
        "LN esaminati":    float(ln_exam),
    }
    X_new_base = pd.DataFrame([patient_raw])

    # One-hot encode like training (senza iEtaDecade: creiamo direttamente i flag dopo)
    X_new_enc = pd.get_dummies(
        X_new_base,
        columns=["iSex", "iInt2", "iZona2", "iStage"],
        drop_first=True
    )

    # ---- Age flags (training used iEtaDecade_7 and iEtaDecade_8) ----
    X_new_enc["iEtaDecade_7"] = 1 if age_flag == "60–69 yrs" else 0
    X_new_enc["iEtaDecade_8"] = 1 if age_flag == "≥70 yrs"  else 0

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
This prototype was developed using data from the **SEER (Surveillance, Epidemiology, and End Results) database**, applying a **machine learning anomaly-detection approach** (Isolation Forest)  
to identify patients whose clinical profiles resemble a *death-phenotype* pattern.

Patients with anomaly scores above the **95th percentile** of the Isolation Forest distribution are labeled as **High-Risk / Death-Phenotype-like**.

**Notes.**
- *Typical resection* includes **pancreaticoduodenectomy, left pancreatectomy, and total pancreatectomy**.  
- The model was trained on a cohort of **387 patients** extracted from the SEER database.  
- The apparent (in-sample) discrimination of the model reached an **AUC = 0.891**,  
  while the cross-validated (out-of-sample, single-patient) performance yielded an **AUC = 0.727**. 

---

**Developed by [Claudio Ricci, MD, PhD](mailto:claudio.ricci@unibo.it)**  
*Associate Professor of Surgery — University of Bologna*  
*IRCCS Azienda Ospedaliero–Universitaria di Bologna (S. Orsola–Malpighi Hospital)*  

© 2025 — For research and educational purposes only.
""")

