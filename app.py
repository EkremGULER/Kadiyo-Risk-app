import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

# -------------------------------------------------
# SAYFA AYARI
# -------------------------------------------------
st.set_page_config(
    page_title="Kardiyovasküler Hastalık Risk Tahmini",
    page_icon="🫀",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #F7F9FC;
}

/* Başlık */
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

/* Kartlar */
.card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #E0E6ED;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.05);
    margin-bottom: 25px;
}

/* Sonuç kutusu */
.result-box {
    padding: 22px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: 600;
}

/* Slider rengi biraz daha belirgin */
.stSlider > div[data-baseweb="slider"] > div {
    background: #FF8888;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# MODEL YÜKLEME
# -------------------------------------------------
@st.cache_resource
def load_model():
    # Google Drive yedeği (gdown ile)
    file_id = "1WdRoUATILi2VUCuyOEFAnrpoVJ7t69y-"
    url = f"https://drive.google.com/uc?id={file_id}"
    model_path = "cardio_ensemble_model.pkl"

    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)

    model = joblib.load("cardio_ensemble_model.pkl")
    feature_cols = joblib.load("cardio_feature_cols.pkl")
    return model, feature_cols

model, feature_cols = load_model()

# -------------------------------------------------
# BAŞLIK VE GENEL AÇIKLAMA
# -------------------------------------------------
st.markdown('<h1 class="main-title">🫀 Kardiyovasküler Hastalık Risk Tahmin Modeli</h1>',
            unsafe_allow_html=True)

st.markdown(
    """
<div class="sub-title">
Bu web arayüzü, <b>Lojistik Regresyon + Random Forest + XGBoost</b> tabanlı
bir <b>ensemble makine öğrenmesi modeli</b> kullanarak bireylerin kardiyovasküler
hastalık riskini tahmin eder. Model, 70.000 gözlem içeren <i>Cardio Vascular Disease</i>
veri seti üzerinde eğitilmiş olup demografik, antropometrik ve biyokimyasal göstergeleri kullanmaktadır.
</div>
""",
    unsafe_allow_html=True
)

# Ana düzen: sol (form+sonuç), sağ (bilgi kartları)
left_col, right_col = st.columns([2.2, 1.2])

# -------------------------------------------------
# SOL KOLON: FORM + SONUÇLAR
# -------------------------------------------------
with left_col:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 Kişisel ve Klinik Bilgiler")

    col1, col2 = st.columns(2)

    with col1:
        gender_str = st.selectbox("Cinsiyet", ["Kadın", "Erkek"])
        # Veri setinde gender: 1=Kadın, 2=Erkek
        gender = 1 if gender_str == "Kadın" else 2

        age_years = st.slider("Yaş (yıl)", 29, 65, 50)
        height = st.slider("Boy (cm)", 130, 210, 170)
        weight = st.slider("Kilo (kg)", 40, 150, 70)

        ap_hi = st.slider("Sistolik Tansiyon (mmHg)", 80, 240, 130)
        ap_lo = st.slider("Diyastolik Tansiyon (mmHg)", 40, 150, 80)

    with col2:
        cholesterol = st.number_input("Total Kolesterol (mg/dL)", 100, 350, 180)
        gluc = st.number_input("Açlık Kan Şekeri (mg/dL)", 60, 300, 95)

        smoke_str = st.selectbox("Sigara Kullanımı", ["Hayır", "Evet"])
        alco_str = st.selectbox("Alkol Kullanımı", ["Hayır", "Evet"])
        active_str = st.selectbox("Fiziksel Aktivite", ["Aktif (Düzenli)", "Pasif (Hareketsiz)"])

    # -------------------------------------------------
    # TÜREVSEL ÖZELLİKLER
    # -------------------------------------------------
    bmi = weight / ((height / 100) ** 2)  # kg/m²
    pulse_pressure = ap_hi - ap_lo
    age_bp_index = age_years * ap_hi

    # Sigara / alkol / aktivite 0-1 kodlama
    smoke_corrected = 1 if smoke_str == "Evet" else 0
    alco_corrected = 1 if alco_str == "Evet" else 0
    active_corrected = 1 if active_str.startswith("Aktif") else 0

    # Kolesterol kategorisi (literatür)
    # <=200: sağlıklı, 200-240: sınırda, >240: yüksek
    if cholesterol <= 200:
        chol_cat = 1
        chol_txt = "Sağlıklı (≤200 mg/dL)"
    elif cholesterol <= 240:
        chol_cat = 2
        chol_txt = "Sınırda (200–240 mg/dL)"
    else:
        chol_cat = 3
        chol_txt = "Yüksek (>240 mg/dL)"

    # Açlık kan şekeri kategorisi
    # 70–100: normal, 100–126: prediyabet, ≥126: diyabet
    if gluc <= 100:
        gluc_cat = 1
        gluc_txt = "Normal (70–100 mg/dL)"
    elif gluc <= 126:
        gluc_cat = 2
        gluc_txt = "Prediyabet (100–126 mg/dL)"
    else:
        gluc_cat = 3
        gluc_txt = "Diyabet (≥126 mg/dL)"

    # Yaşam tarzı skoru (0-3, yüksek skor = daha riskli)
    lifestyle_score = smoke_corrected + alco_corrected + (1 - active_corrected)

    # BMI kategorisi (senin verdiğin tabloya göre)
    if bmi < 18.5:
        bmi_cat = "Zayıf"
    elif bmi < 25:
        bmi_cat = "Sağlıklı"
    elif bmi < 30:
        bmi_cat = "Fazla kilolu"
    elif bmi < 35:
        bmi_cat = "I. derece obezite"
    elif bmi < 40:
        bmi_cat = "II. derece obezite"
    else:
        bmi_cat = "III. derece obezite"

    # Basit tansiyon kategorisi (sistolik odaklı, görseline yakın)
    if ap_hi < 120 and ap_lo < 80:
        bp_cat = "Optimal"
    elif 120 <= ap_hi <= 129 and ap_lo < 84:
        bp_cat = "Normal / Yüksek-normal"
    elif 130 <= ap_hi <= 139:
        bp_cat = "Yüksek-normal"
    elif 140 <= ap_hi <= 159:
        bp_cat = "1. derece hipertansiyon"
    elif 160 <= ap_hi <= 179:
        bp_cat = "2. derece hipertansiyon"
    else:
        bp_cat = "3. derece hipertansiyon veya izole sistolik HT"

    # -------------------------------------------------
    # MODELE GİRECEK VEKÖR (feature_cols ile birebir uyumlu)
    # -------------------------------------------------
    id_val = 0  # dummy
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

    row = [input_dict[col] for col in feature_cols]
    input_df = pd.DataFrame([row], columns=feature_cols)

    # -------------------------------------------------
    # HESAPLANAN EK ÖZELLİKLER (EXPANDER)
    # -------------------------------------------------
    with st.expander("ℹ Hesaplanan Ek Özellikler"):
        st.write(f"**Vücut Kitle İndeksi (BMI):** {bmi:.1f} kg/m²  –  _{bmi_cat}_")
        st.write(f"**Nabız Basıncı (ap_hi - ap_lo):** {pulse_pressure} mmHg")
        st.write(f"**Yaş × Sistolik Tansiyon İndeksi:** {age_bp_index:.0f}")
        st.write(f"**Yaşam Tarzı Skoru (0–3):** {lifestyle_score}  "
                 "(sigara + alkol + hareketsizlik)")
        st.write(f"**Kan Basıncı Kategorisi:** {bp_cat}")
        st.write(f"**Kolesterol Durumu:** {chol_txt}")
        st.write(f"**Açlık Kan Şekeri Durumu:** {gluc_txt}")

    # -------------------------------------------------
    # TAHMİN BUTONU ve SONUÇ
    # -------------------------------------------------
    if st.button("🔍 Kardiyovasküler Risk Tahminini Hesapla", use_container_width=True):
        prob = model.predict_proba(input_df)[0][1]
        risk_pct = prob * 100
        pred = model.predict(input_df)[0]

        if pred == 1:
            st.markdown(
                f'<div class="result-box" style="background:#FFE5E5; color:#B00020;">'
                f'⚠ <b>YÜKSEK RİSK:</b> Model, bu bireyin kardiyovasküler hastalık '
                f'taşıma olasılığını yaklaşık <b>%{risk_pct:.1f}</b> olarak tahmin etmektedir.'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box" style="background:#E2F4FF; color:#004A7C;">'
                f'✅ <b>DÜŞÜK RİSK:</b> Model, bu bireyin kardiyovasküler hastalık '
                f'taşıma olasılığını yaklaşık <b>%{risk_pct:.1f}</b> olarak tahmin etmektedir.'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown(
            """
> ℹ️ **Not (Teknik Açıklama):** Bu çıktı, denetimli makine öğrenmesi ile eğitilmiş
> bir sınıflandırıcının olasılık tahminidir. Model, klinik karar sürecini desteklemek
> amacıyla tasarlanmıştır; tek başına tanı veya tedavi kararı vermek için kullanılmamalıdır.
""")

    st.markdown("</div>", unsafe_allow_html=True)  # form kartı kapanışı

# -------------------------------------------------
# SAĞ KOLON: VERİ SETİ ve MODEL BİLGİLERİ
# -------------------------------------------------
with right_col:
    # Veri seti kartı
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Kullanılan Veri Seti")
    st.markdown(
        """
- **Kaynak:** Cardio Vascular Disease veri seti  
- **Gözlem sayısı:** 70.000+ birey  
- **Değişkenler:** yaş, cinsiyet, boy, kilo, kan basıncı, kolesterol, glikoz,
  sigara, alkol, fiziksel aktivite vb.  
- **Hedef değişken:** `cardio` (0 = hastalık yok, 1 = kardiyovasküler hastalık var)
""")
    st.markdown("</div>", unsafe_allow_html=True)

    # Model kartı
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🤖 Kullanılan Yapay Zekâ Modelleri")
    st.markdown(
        """
- **Lojistik Regresyon**  
- **Random Forest**  
- **XGBoost**  
- Bu üç modelin çıktıları, bir **Ensemble (Topluluk) Modeli**
  ile birleştirilmiştir (olasılıkların ortalaması / soft voting).
""")
    st.markdown("</div>", unsafe_allow_html=True)

    # Performans kartı
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Eğitim Performansı (Test Kümesi)")
    st.markdown(
        """
- **Doğruluk (Accuracy):** ≈ 0.736  
- **Duyarlılık (Recall):** ≈ 0.70  
- **F1 Skoru:** ≈ 0.72  
- **ROC–AUC:** ≈ 0.80  

Bu metrikler, modelin sınıflar arasındaki ayrımı istatistiksel olarak
anlamlı bir düzeyde öğrendiğini göstermektedir.
""")
    st.markdown("</div>", unsafe_allow_html=True)
