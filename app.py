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
    # Google Drive dosya ID (senin paylaştığın linkten)
    file_id = "1WdRoUATILi2VUCuyOEFAnrpoVJ7t69y-"
    url = f"https://drive.google.com/uc?id={file_id}"

    # Sunucuda kullanacağımız dosya adı
    model_path = "cardio_ensemble_model.pkl"

    # Eğer model dosyası yoksa indir
    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)

    # Model ve feature kolonlarını yükle
    model = joblib.load(model_path)
    feature_cols = joblib.load("cardio_feature_cols.pkl")

    return model, feature_cols


# Model & feature listesini yükle
model, feature_cols = load_model()

# -------------------------------------------------
# Sayfa ayarları
# -------------------------------------------------
st.set_page_config(
    page_title="Kardiyovasküler Hastalık Risk Tahmin Modeli",
    page_icon="🫀",
    layout="wide",
)

st.title("🫀 Kardiyovasküler Hastalık Risk Tahmin Modeli")

st.write(
    """
    Bu uygulama, *Lojistik Regresyon + Random Forest + XGBoost*
    modellerinden oluşan bir *Ensemble (Topluluk) Yapay Zekâ Modeli* ile
    kardiyovasküler hastalık riskini tahmin eder.
    
    Kullanılan veri seti, 70.000'den fazla bireyin demografik ve klinik
    özelliklerini içeren *Cardio Vascular Disease* veri setidir.
    """
)

st.markdown("---")

# -------------------------------------------------
# Kullanıcı girişleri
# -------------------------------------------------
st.header("📋 Kişisel ve Klinik Bilgiler")

col1, col2 = st.columns(2)

with col1:
    age_years = st.slider("Yaş (yıl)", 29, 65, 50, help="Veri seti 29-65 yaş aralığını kapsıyor.")
    height = st.slider("Boy (cm)", 130, 210, 170)
    weight = st.slider("Kilo (kg)", 40, 150, 75)
    ap_hi = st.slider("Sistolik Tansiyon (mmHg)", 80, 240, 130)
    ap_lo = st.slider("Diyastolik Tansiyon (mmHg)", 40, 130, 80)

with col2:
    # Kolesterol mg/dL
    cholesterol_mg = st.slider(
        "Total Kolesterol (mg/dL)",
        100,
        320,
        190,
        help="200 ve altı: Sağlıklı, 200-240: Sınırda, 240 üzeri: Yüksek",
    )

    # Açlık kan şekeri mg/dL
    gluc_mg = st.slider(
        "Açlık Kan Şekeri / Glukoz (mg/dL)",
        60,
        250,
        95,
        help="70-100: Normal, 100-126: Prediyabet, 126 ve üzeri: Diyabet",
    )

    smoke = st.selectbox(
        "Sigara Kullanımı",
        options=[0, 1],
        format_func=lambda x: "Evet" if x == 1 else "Hayır",
    )
    alco = st.selectbox(
        "Alkol Kullanımı",
        options=[0, 1],
        format_func=lambda x: "Evet" if x == 1 else "Hayır",
    )
    active = st.selectbox(
        "Fiziksel Aktivite",
        options=[0, 1],
        format_func=lambda x: "Aktif (Düzenli hareketli)" if x == 1 else "Pasif (Hareketsiz)",
    )

st.markdown("---")

# -------------------------------------------------
# Literatüre göre kategorilere dönüştürmeler
# -------------------------------------------------

# Kolesterol: 1=Sağlıklı, 2=Sınırda, 3=Yüksek
if cholesterol_mg <= 200:
    cholesterol_cat = 1
    chol_txt = "Sağlıklı (≤200 mg/dL)"
elif cholesterol_mg <= 240:
    cholesterol_cat = 2
    chol_txt = "Sınırda (200-240 mg/dL)"
else:
    cholesterol_cat = 3
    chol_txt = "Yüksek (>240 mg/dL)"

# Açlık kan şekeri: 1=Normal, 2=Prediyabet, 3=Diyabet
if gluc_mg < 70:
    gluc_cat = 1
    gluc_txt = "Düşük (<70 mg/dL)"
elif gluc_mg <= 100:
    gluc_cat = 1
    gluc_txt = "Normal (70-100 mg/dL)"
elif gluc_mg <= 126:
    gluc_cat = 2
    gluc_txt = "Prediyabet (100-126 mg/dL)"
else:
    gluc_cat = 3
    gluc_txt = "Diyabet (≥126 mg/dL)"

# -------------------------------------------------
# Türetilmiş (engineered) özellikler
# -------------------------------------------------
bmi = weight / ((height / 100) ** 2)  # kg/m²

# BMI sınıflaması
if bmi < 18.5:
    bmi_cat = "Zayıf"
elif bmi < 25:
    bmi_cat = "Sağlıklı"
elif bmi < 30:
    bmi_cat = "Fazla kilolu"
elif bmi < 35:
    bmi_cat = "I. Derece obezite"
elif bmi < 40:
    bmi_cat = "II. Derece obezite"
else:
    bmi_cat = "III. Derece obezite"

pulse_pressure = ap_hi - ap_lo
age_bp_index = age_years * ap_hi

# Yaşam tarzı skoru (0-3; burada sadece bilgilendirme için)
lifestyle_score = smoke + alco + (1 - active)

# Tansiyon sınıflaması (basitleştirilmiş)
if ap_hi < 120 and ap_lo < 80:
    bp_cat = "Optimal"
elif 120 <= ap_hi <= 129 and ap_lo < 85:
    bp_cat = "Normal / Yüksek-Normal"
elif 130 <= ap_hi <= 139 or 85 <= ap_lo <= 89:
    bp_cat = "Yüksek-Normal"
elif 140 <= ap_hi <= 159 or 90 <= ap_lo <= 99:
    bp_cat = "1. derece hipertansiyon"
elif 160 <= ap_hi <= 179 or 100 <= ap_lo <= 109:
    bp_cat = "2. derece hipertansiyon"
elif ap_hi >= 180 or ap_lo >= 110:
    bp_cat = "3. derece hipertansiyon"
else:
    bp_cat = "Sınırda / belirsiz"

# Sigara / alkol semantiğini modele göre düzelt
# Modele giderken 0 = riskli (içiyor), 1 = içmiyor olacak şekilde ters çeviriyoruz
smoke_corrected = 0 if smoke == 1 else 1
alco_corrected = 0 if alco == 1 else 1

# -------------------------------------------------
# Hesaplanan ek özellikleri göster
# -------------------------------------------------
with st.expander("ℹ Hesaplanan Ek Özellikler ve Kategoriler"):
    st.write(f"*BMI (Vücut Kitle İndeksi):* {bmi:.1f} kg/m² — {bmi_cat}")
    st.write(
        f"*Kolesterol Kategorisi:* {chol_txt} "
        f"(modele giden değer: {cholesterol_cat})"
    )
    st.write(
        f"*Glukoz Kategorisi:* {gluc_txt} "
        f"(modele giden değer: {gluc_cat})"
    )
    st.write(f"*Nabız Basıncı (ap_hi - ap_lo):* {pulse_pressure} mmHg")
    st.write(f"*Tansiyon Kategorisi:* {bp_cat}")
    st.write(f"*Yaş x Tansiyon İndeksi:* {age_bp_index}")
    st.write(
        f"*Yaşam Tarzı Skoru (0-3):* {lifestyle_score} "
        f"(yüksek skor = daha riskli profil)"
    )
    st.write(
        f"*Sigara (modele giden):* {smoke_corrected} "
        f"— 0: içiyor, 1: içmiyor"
    )
    st.write(
        f"*Alkol (modele giden):* {alco_corrected} "
        f"— 0: kullanıyor, 1: kullanmıyor"
    )

st.markdown("---")

# -------------------------------------------------
# Girdi vektörünü, modelin beklediği sırada hazırlama
# -------------------------------------------------
# Temel değişken sözlüğü
base_input = {
    "age_years": age_years,
    "height": height,
    "weight": weight,
    "ap_hi": ap_hi,
    "ap_lo": ap_lo,
    "cholesterol": cholesterol_cat,  # kategorik
    "gluc": gluc_cat,                # kategorik
    "smoke": smoke_corrected,
    "alco": alco_corrected,
    "active": active,
    "bmi": bmi,
    "pulse_pressure": pulse_pressure,
    "age_bp_index": age_bp_index,
    "lifestyle_score": lifestyle_score,
}

# feature_cols içindeki sıraya göre tek satırlık dataframe oluştur
row = [base_input.get(col, 0) for col in feature_cols]
input_df = pd.DataFrame([row], columns=feature_cols)

# -------------------------------------------------
# Tahmin butonu
# -------------------------------------------------
if st.button("🔍 Kardiyovasküler Risk Tahminini Hesapla"):
    prob = float(model.predict_proba(input_df)[0][1])  # cardio = 1 olasılığı
    pred = int(model.predict(input_df)[0])
    risk_yuzde = prob * 100

    if pred == 1:
        st.error(
            f"⚠ *YÜKSEK RİSK:* Model bu kişinin kardiyovasküler hastalık riskini "
            f"yaklaşık *%{risk_yuzde:.1f}* olarak tahmin ediyor."
        )
    else:
        st.success(
            f"✅ *DÜŞÜK RİSK:* Model bu kişinin kardiyovasküler hastalık riskini "
            f"yaklaşık *%{risk_yuzde:.1f}* olarak tahmin ediyor."
        )

    st.markdown(
        "> *Not:* Bu model, klinik kararı desteklemek için tasarlanmış bir "
        "karar destek sistemidir. Tek başına tıbbi tanı veya tedavi kararında "
        "kullanılmamalıdır."
    )
