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
    margin-bottom: 6px;
}

.sub-title {
    text-align: center;
    color: #4F5B66;
    font-size: 15px;
    margin-bottom: 24px;
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

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# 📌 MODEL YÜKLEME (LOCAL + DRIVE FALLBACK)
# -------------------------------------------------
@st.cache_resource
def load_model():
    """
    Model ve feature listesi:
    - Model: Kardiyovasküler hastalık (cardio=0/1) için ensemble sınıflandırıcı
    - feature_cols: Eğitimde kullanılan kolon isimleri
    """
    file_id = "1WdRoUATILi2VUCuyOEFAnrpoVJ7t69y-"
    url = f"https://drive.google.com/uc?id={file_id}"

    model_path = "cardio_ensemble_model.pkl"

    # Model dosyası yoksa Google Drive'dan indir
    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)

    model = joblib.load("cardio_ensemble_model.pkl")
    feature_cols = joblib.load("cardio_feature_cols.pkl")
    return model, feature_cols

model, feature_cols = load_model()

# ------------------------------------------
# 🎯 SAYFA BAŞLIĞI ve TEKNİK AÇIKLAMA
# ------------------------------------------
st.markdown('<h1 class="main-title">🫀 Kardiyovasküler Hastalık Risk Tahmin Modeli</h1>',
            unsafe_allow_html=True)

st.markdown(
    """
<div class="sub-title">
Bu web arayüzü, <b>Lojistik Regresyon + Random Forest + XGBoost</b> tabanlı
bir <b>ensemble makine öğrenmesi modeli</b> ile kardiyovasküler hastalık riskini tahmin eder.
Model, 70.000 gözlem içeren <i>Cardio Vascular Disease</i> veri seti üzerinde eğitilmiş ve
demografik, antropometrik ve biyokimyasal değişkenleri kullanmaktadır.
</div>
""",
    unsafe_allow_html=True
)

# ------------------------------------------
# 📝 KULLANICI GİRİŞLERİ (KART TASARIMI)
# ------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📋 Kişisel ve Klinik Bilgiler")

col1, col2 = st.columns(2)

with col1:
    gender_str = st.selectbox("Cinsiyet", ["Kadın", "Erkek"])
    # Cardio veri setinde gender: 1 = kadın, 2 = erkek
    gender = 1 if gender_str == "Kadın" else 2

    age_years = st.slider("Yaş (yıl)", 29, 65, 50)
    height = st.slider("Boy (cm)", 130, 210, 170)
    weight = st.slider("Kilo (kg)", 40, 150, 70)

    ap_hi = st.slider("Sistolik Tansiyon (mmHg)", 80, 240, 130)
    ap_lo = st.slider("Diyastolik Tansiyon (mmHg)", 40, 150, 80)

with col2:
    # Total kolesterol (literatüre göre sınıflandırılacak)
    cholesterol = st.number_input("Total Kolesterol (mg/dL)", 100, 350, 180)

    # Açlık kan şekeri (AKŞ) - literatüre göre sınıflandırılacak
    gluc = st.number_input("Açlık Kan Şekeri (mg/dL)", 60, 300, 95)

    smoke_str = st.selectbox("Sigara Kullanımı", ["Hayır", "Evet"])
    alco_str = st.selectbox("Alkol Kullanımı", ["Hayır", "Evet"])
    active_str = st.selectbox("Fiziksel Aktivite", ["Aktif (Düzenli)", "Pasif (Hareketsiz)"])

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------
# 🧮 TÜREVSEL ÖZELLİKLER (BMI, Yaşam Tarzı Skoru vb.)
# ----------------------------------------------------------
bmi = weight / ((height / 100) ** 2)  # VKİ = kg / (m²)
pulse_pressure = ap_hi - ap_lo
age_bp_index = age_years * ap_hi

# Sigara / alkol / aktivite kodlamaları (model için 0/1)
smoke_corrected = 1 if smoke_str == "Evet" else 0
alco_corrected = 1 if alco_str == "Evet" else 0
active_corrected = 1 if active_str.startswith("Aktif") else 0  # aktif=1, pasif=0

# Literatüre göre kolesterol kategorisi:
#  - ≤200 : Sağlıklı
#  - 200–240 : Sınırda
#  - >240 : Yüksek
if cholesterol <= 200:
    chol_cat = 1
elif cholesterol <= 240:
    chol_cat = 2
else:
    chol_cat = 3

# Literatüre göre AKŞ (açlık kan şekeri) kategorisi:
#  - 70–100 : Normal
#  - 100–126 : Prediyabet
#  - ≥126 : Diyabet
if gluc <= 100:
    gluc_cat = 1
elif gluc <= 126:
    gluc_cat = 2
else:
    gluc_cat = 3

# Yaşam tarzı skoru (0–3, yüksek skor = daha riskli)
# sigara(1) + alkol(1) + hareketsizlik(1)
lifestyle_score = smoke_corrected + alco_corrected + (1 - active_corrected)

# ----------------------------------------------------------
# 🔑 MODELİN BEKLEDİĞİ TÜM KOLONLARI OLUŞTUR
# ----------------------------------------------------------
# id modeli etkilemeyen dummy bir alan, 0 veriyoruz
id_val = 0

# input_dict ANAHTARLARI -> feature_cols ile birebir uyumlu
input_dict = {
    "id": id_val,
    "gender": gender,
    "height": height,
    "weight": weight,
    "ap_hi": ap_hi,
    "ap_lo": ap_lo,
    "cholesterol": chol_cat,
    "gluc": gluc_cat,
    "smoke": smoke_corrected,
    "alco": alco_corrected,
    "active": active_corrected,
    "age_years": age_years,
    "bmi": bmi,
    "pulse_pressure": pulse_pressure,
    "age_bp_index": age_bp_index,
    "lifestyle_score": lifestyle_score
}

# feature_cols sırasına göre vektör oluştur
row = [input_dict[col] for col in feature_cols]
input_df = pd.DataFrame([row], columns=feature_cols)

# ----------------------------------------------------------
# 🔘 TAHMİN BUTONU
# ----------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
if st.button("🔍 Kardiyovasküler Risk Tahminini Hesapla", use_container_width=True):
    prob = model.predict_proba(input_df)[0][1]  # cardio=1 olasılığı
    risk_pct = prob * 100
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.markdown(
            f'<div class="result-box" style="background:#FFE5E5; color:#B00020;">'
            f'⚠ <b>YÜKSEK RİSK:</b> Model, bu bireyin kardiyovasküler hastalık riskini '
            f'takip eden 10 yıllık dönemde yaklaşık <b>%{risk_pct:.1f}</b> olarak tahmin etmektedir.'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-box" style="background:#E2F4FF; color:#004A7C;">'
            f'✅ <b>DÜŞÜK RİSK:</b> Model, bu bireyin kardiyovasküler hastalık riskini '
            f'takip eden 10 yıllık dönemde yaklaşık <b>%{risk_pct:.1f}</b> olarak tahmin etmektedir.'
            f'</div>',
            unsafe_allow_html=True
        )

st.markdown(
    """
> ℹ️ **Not (Teknik Açıklama):** Bu çıktı, denetimli makine öğrenmesi ile eğitilmiş bir sınıflandırıcının
> olasılık tahminidir. Model, klinik kararı destekleyen bir araçtır; tek başına tanı veya tedavi
> kararı vermek için kullanılmamalıdır.
""")

st.markdown("</div>", unsafe_allow_html=True)
