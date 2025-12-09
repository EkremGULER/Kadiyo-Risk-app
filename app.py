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
    (Bu kısım Streamlit Cloud'da da çalışır.)
    """
    # 1) Google Drive dosya ID (paylaştığın linkten)
    file_id = "1WdRoUATILi2VUCuyOEFAnrpoVJ7t69y-"
    url = f"https://drive.google.com/uc?id={file_id}"

    # 2) Sunucuda kullanacağımız dosya adı
    model_path = "cardio_ensemble_model.pkl"

    # 3) Eğer model dosyası yoksa indir
    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)

    # 4) Model ve feature kolonlarını yükle
    model = joblib.load(model_path)
    feature_cols = joblib.load("cardio_feature_cols.pkl")

    return model, feature_cols


# Model ve feature listesini yükle
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

    > ⚠️ Bu uygulama *sadece eğitim / demo amaçlıdır*. 
    > Buradaki değer aralıkları, klinik rehber yerine geçmez.
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
    # --- Sayısal Kolesterol (mg/dL) ---
    total_chol = st.slider(
        "Toplam Kolesterol (mg/dL)  (Sadece demo amaçlı aralıklar)",
        100, 320, 190,
        help="Bu aralıklar klinik rehber değildir, sadece modeli beslemek için kullanılmaktadır."
    )
    # Kolesterolü 1-3 kategorisine çevir (sadece model için)
    if total_chol < 200:
        cholesterol = 1
    elif total_chol < 240:
        cholesterol = 2
    else:
        cholesterol = 3

    # --- Sayısal Glukoz (mg/dL) ---
    fasting_glucose = st.slider(
        "Açlık Glukoz (mg/dL)  (Sadece demo amaçlı aralıklar)",
        60, 220, 90,
        help="Bu aralıklar klinik rehber değildir, sadece modeli beslemek için kullanılmaktadır."
    )
    # Glukozu 1-3 kategorisine çevir (sadece model için)
    if fasting_glucose < 100:
        gluc = 1
    elif fasting_glucose < 126:
        gluc = 2
    else:
        gluc = 3

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
# Sigara / Alkol model düzeltmeleri
# ----------------------------------------------------------
# Kullanıcı 1 seçerse, modelde 1 riskli olacağı için aynen geçsin, 0 seçerse 0 olsun.
smoke_corrected = 1 if smoke == 1 else 0
alco_corrected = 1 if alco == 1 else 0

# ----------------------------------------------------------
# Girdi vektörünü, modelin beklediği sıralamada hazırlama
# ----------------------------------------------------------
input_dict = {
    "age_years": age_years,
    "height": height,
    "weight": weight,
    "ap_hi": ap_hi,
    "ap_lo": ap_lo,
    "cholesterol": cholesterol,   # 1-3 kategori
    "gluc": gluc,                 # 1-3 kategori
    "smoke": smoke_corrected,
    "alco": alco_corrected,
    "active": active,
    "bmi": bmi,
    "pulse_pressure": pulse_pressure,
    "age_bp_index": age_bp_index,
    "lifestyle_score": lifestyle_score,
}

# feature_cols sırasına göre DF oluştur
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
