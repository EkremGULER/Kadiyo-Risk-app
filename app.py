import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

# --------------------------
# 🎨 CUSTOM CSS (MODERN SAĞLIK TEMASI)
# --------------------------
st.markdown("""
<style>

body {
    background-color: #F7F9FC;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #0A84FF, #5BC0F8);
    -webkit-background-clip: text;
    color: transparent;
    margin-bottom: 10px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #E0E6ED;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.05);
    margin-bottom: 25px;
}

.result-box {
    padding: 22px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: 600;
}

.btn-custom {
    background-color: #0A84FF;
    color: white;
    padding: 14px 28px;
    border-radius: 10px;
    font-size: 18px;
    border: none;
    width: 100%;
}
.btn-custom:hover {
    background-color: #006FE6;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# 📌 MODEL YÜKLEME (LOCAL + DRIVE FALLBACK)
# -------------------------------------------------
@st.cache_resource
def load_model():
    file_id = "1WdRoUATILi2VUCuyOEFAnrpoVJ7t69y-"
    url = f"https://drive.google.com/uc?id={file_id}"

    model_path = "cardio_ensemble_model.pkl"

    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)

    model = joblib.load("cardio_ensemble_model.pkl")
    feature_cols = joblib.load("cardio_feature_cols.pkl")
    return model, feature_cols

model, feature_cols = load_model()

# ------------------------------------------
# 🎯 SAYFA BAŞLIĞI
# ------------------------------------------
st.markdown('<h1 class="main-title">🫀 Kardiyovasküler Hastalık Risk Tahmin Modeli</h1>', unsafe_allow_html=True)

st.write("""
Bu uygulama, **Modern Tıbbi Veri Bilimi Teknikleri** kullanarak kişilerin kardiyovasküler hastalık riskini tahmin eder.
Mavi-beyaz sağlık temasına göre tasarlanmıştır.
""")

# ------------------------------------------
# 📝 KULLANICI GİRİŞLERİ (KART TASARIMI)
# ------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📋 Kişisel ve Klinik Bilgiler")

col1, col2 = st.columns(2)

with col1:
    age_years = st.slider("Yaş (yıl)", 29, 65, 50)
    height = st.slider("Boy (cm)", 130, 210, 170)
    weight = st.slider("Kilo (kg)", 40, 150, 70)

    ap_hi = st.slider("Sistolik Tansiyon (mmHg)", 80, 240, 130)
    ap_lo = st.slider("Diyastolik Tansiyon (mmHg)", 40, 150, 80)

with col2:
    cholesterol = st.number_input("Total Kolesterol (mg/dL)", 100, 350, 180)
    gluc = st.number_input("Açlık Kan Şekeri (mg/dL)", 60, 300, 90)

    smoke = st.selectbox("Sigara Kullanımı", ["Hayır", "Evet"])
    alco = st.selectbox("Alkol Kullanımı", ["Hayır", "Evet"])
    active = st.selectbox("Fiziksel Aktivite", ["Aktif", "Pasif"])

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------
# 🧮 TÜREVSEL ÖZELLİKLER (BMI, Tansiyon, vb.)
# ----------------------------------------------------------
bmi = weight / ((height / 100) ** 2)
pulse_pressure = ap_hi - ap_lo
age_bp_index = age_years * ap_hi

# Sigara/alkol model etkisi ters çevrilmişti → düzeltiyoruz
smoke_corrected = 1 if smoke == "Evet" else 0
alco_corrected = 1 if alco == "Evet" else 0
active_corrected = 0 if active == "Pasif" else 1

# Kolesterol kategorisi — literatür
if cholesterol <= 200:
    chol_cat = 1
elif cholesterol <= 240:
    chol_cat = 2
else:
    chol_cat = 3

# Glikoz kategorisi — literatür
if gluc <= 100:
    gluc_cat = 1
elif gluc <= 126:
    gluc_cat = 2
else:
    gluc_cat = 3

# ---------------------------------
# MODEL GİRDİSİ
# ---------------------------------
input_dict = {
    "age_years": age_years,
    "height": height,
    "weight": weight,
    "ap_hi": ap_hi,
    "ap_lo": ap_lo,
    "cholesterol": chol_cat,
    "gluc": gluc_cat,
    "smoke": smoke_corrected,
    "alco": alco_corrected,
    "active": active_corrected,
    "bmi": bmi,
    "pulse_pressure": pulse_pressure,
    "age_bp_index": age_bp_index
}

input_df = pd.DataFrame([[input_dict[col] for col in feature_cols]], columns=feature_cols)

# ------------------------------------------
# 🔘 HESAPLA BUTONU
# ------------------------------------------
if st.button("🔍 Risk Tahminini Hesapla", key="predict", use_container_width=True):
    prob = model.predict_proba(input_df)[0][1]
    risk_pct = prob * 100
    pred = model.predict(input_df)[0]

    # Sonuç kutusu
    if pred == 1:
        st.markdown(
            f'<div class="result-box" style="background:#FFE5E5; color:#B00020;">⚠ YÜKSEK RİSK: Tahmini risk %{risk_pct:.1f}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-box" style="background:#E2F4FF; color:#004A7C;">✅ DÜŞÜK RİSK: Tahmini risk %{risk_pct:.1f}</div>',
            unsafe_allow_html=True
        )

