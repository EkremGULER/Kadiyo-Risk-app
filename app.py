import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

# -------------------------------------------------
# Modeli ve feature kolonlarını yükleyen fonksiyon
# -------------------------------------------------
@st.cache_resource
def load_model():
    """
    Model dosyası yoksa Google Drive'dan indir,
    sonra modeli ve feature kolonlarını yükle.
    """
    file_id = "1WdRoUATILi2VUCuyOEFAnrpoVJ7t69y-"
    url = f"https://drive.google.com/uc?id={file_id}"

    model_path = "cardio_ensemble_model.pkl"

    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)

    model = joblib.load(model_path)
    feature_cols = joblib.load("cardio_feature_cols.pkl")

    return model, feature_cols

model, feature_cols = load_model()

# ----------------------------------------------------------
# Sayfa ayarları
# ----------------------------------------------------------
st.set_page_config(
    page_title="Kardiyovasküler Hastalık Risk Tahmin Modeli",
    page_icon="❤️",
    layout="wide"
)

st.title("🫀 Kardiyovasküler Hastalık Risk Tahmin Modeli")

st.write(
    """
    Bu uygulama, **Lojistik Regresyon + Random Forest + XGBoost**
    modellerinden oluşan bir **Ensemble (Topluluk) Yapay Zekâ Modeli** ile
    kardiyovasküler hastalık riskini tahmin eder.
    
    Kullanılan veri seti, 70.000'den fazla bireyin demografik ve klinik
    özelliklerini içeren **Cardio Vascular Disease** veri setidir.
    """
)

st.markdown("---")

# ----------------------------------------------------------
# Kullanıcı girişleri
# ----------------------------------------------------------
st.header("📋 Kişisel ve Klinik Bilgiler")

col1, col2 = st.columns(2)

with col1:
    age_years = st.slider("Yaş (yıl)", 29, 65, 50)
    height = st.slider("Boy (cm)", 130, 210, 170)
    weight = st.slider("Kilo (kg)", 40, 150, 75)
    ap_hi = st.slider("Sistolik Tansiyon (ap_hi)", 80, 240, 130)
    ap_lo = st.slider("Diyastolik Tansiyon (ap_lo)", 40, 180, 80)

with col2:
    cholesterol = st.selectbox(
        "Kolesterol Düzeyi",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "1 - Normal",
            2: "2 - Yüksek",
            3: "3 - Çok Yüksek"
        }[x]
    )
    gluc = st.selectbox(
        "Glikoz Düzeyi",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "1 - Normal",
            2: "2 - Yüksek",
            3: "3 - Çok Yüksek"
        }[x]
    )
    smoke = st.selectbox(
        "Sigara Kullanımı",
        options=[0, 1],
        format_func=lambda x: "Evet" if x == 1 else "Hayır"
    )
    alco = st.selectbox(
        "Alkol Kullanımı",
        options=[0, 1],
        format_func=lambda x: "Evet" if x == 1 else "Hayır"
    )
    active = st.selectbox(
        "Fiziksel Aktivite",
        options=[0, 1],
        format_func=lambda x: "Aktif (Düzenli Hareketli)" if x == 1 else "Pasif (Hareketsiz)"
    )

st.markdown("---")

# ----------------------------------------------------------
# Türetilmiş (engineered) özellikler
# ----------------------------------------------------------
bmi = weight / ((height / 100) ** 2)
pulse_pressure = ap_hi - ap_lo
age_bp_index = age_years * ap_hi
lifestyle_score = smoke + alco + (1 - active)  # 0-3 arası skor (yüksekse daha riskli)

with st.expander("ℹ Hesaplanan Ek Özellikler"):
    st.write(f"**BMI (Vücut Kitle İndeksi):** {bmi:.1f}")
    st.write(f"**Nabız Basıncı (ap_hi - ap_lo):** {pulse_pressure}")
    st.write(f"**Yaş x Tansiyon İndeksi:** {age_bp_index}")
    st.write(
        f"**Yaşam Tarzı Skoru (0-3, yüksek skor = daha riskli):** {lifestyle_score}"
    )

# ----------------------------------------------------------
# Girdi vektörünü, modelin beklediği sıralamada hazırlama
# ----------------------------------------------------------
input_dict = {
    "age_years": age_years,
    "height": height,
    "weight": weight,
    "ap_hi": ap_hi,
    "ap_lo": ap_lo,
    "cholesterol": cholesterol,
    "gluc": gluc,
    "smoke": smoke,
    "alco": alco,
    "active": active,
    "bmi": bmi,
    "pulse_pressure": pulse_pressure,
    "age_bp_index": age_bp_index,
    "lifestyle_score": lifestyle_score
}

input_df = pd.DataFrame([[input_dict[col] for col in feature_cols]], columns=feature_cols)

st.markdown("---")

# ----------------------------------------------------------
# Tahmin butonu
# ----------------------------------------------------------
if st.button("🔍 Kardiyovasküler Risk Tahminini Hesapla"):
    prob = model.predict_proba(input_df)[0][1]  # cardio = 1 olasılığı
    pred = model.predict(input_df)[0]
    risk_yuzde = prob * 100

    if pred == 1:
        st.error(
            f"⚠ **YÜKSEK RİSK:** Model bu kişinin kardiyovasküler hastalık riskini "
            f"yaklaşık **%{risk_yuzde:.1f}** olarak tahmin ediyor."
        )
    else:
        st.success(
            f"✅ **DÜŞÜK RİSK:** Model bu kişinin kardiyovasküler hastalık riskini "
            f"yaklaşık **%{risk_yuzde:.1f}** olarak tahmin ediyor."
        )

    st.markdown(
        "> **Not:** Bu model, klinik kararı desteklemek için tasarlanmış bir "
        "karar destek sistemidir. Tek başına tıbbi tanı veya tedavi kararında "
        "kullanılmamalıdır."
    )
