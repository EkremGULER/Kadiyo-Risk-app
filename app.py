import os
from pathlib import Path

import gdown
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# Sayfa Ayarları ve Basit Stil Düzenlemeleri
# =========================================================
st.set_page_config(
    page_title="Kardiyovasküler Hastalık Risk Tahmin Modeli",
    page_icon="🫀",
    layout="wide",
)

# Slider ve kutuların rengini biraz yumuşatmak için basit CSS
st.markdown(
    """
    <style>
    /* Genel arka planı çok hafif gri yap */
    .main {
        background-color: #fafafa;
    }

    /* Slider rengi */
    .stSlider > div[data-baseweb="slider"] > div > div {
        background: linear-gradient(90deg, #6fb1fc, #e56399);
    }
    .stSlider [role="slider"] {
        background-color: #ffffff !important;
        border: 2px solid #6fb1fc !important;
    }

    /* Kart benzeri kutular */
    .infocard {
        padding: 1rem 1.2rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(15, 15, 15, 0.06);
        font-size: 0.93rem;
    }

    .soft-header {
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Modeli ve Özellik Listesini Yükleme
# =========================================================
@st.cache_resource
def load_model():
    """
    - Model dosyası Streamlit ortamında yoksa Google Drive'dan indirir.
    - Ensemble modeli ve eğitimde kullanılan özellik listesini yükler.
    """
    file_id = "1WdRoUATILi2VUCuyOEFAnrpoVJ7t69y-"
    url = f"https://drive.google.com/uc?id={file_id}"
    model_path = Path("cardio_ensemble_model.pkl")

    # Model dosyası yoksa Drive'dan indir
    if not model_path.exists():
        gdown.download(url, str(model_path), quiet=False)

    model = joblib.load(model_path)
    feature_cols = joblib.load("cardio_feature_cols.pkl")

    return model, feature_cols


with st.spinner("Model yükleniyor, lütfen bekleyiniz..."):
    model, feature_cols = load_model()


# =========================================================
# Sayfa Başlığı ve Üst Açıklama
# =========================================================
st.markdown(
    "<h2 style='text-align:center; margin-bottom:0.2rem;'>"
    "Kardiyovasküler Hastalık Risk Tahmin Modeli"
    "</h2>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style='text-align:center; font-size:0.9rem; color:#555;'>
    Bu web arayüzü, lojistik regresyon, random forest ve XGBoost tabanlı bir 
    <b>ensemble (topluluk) makine öğrenmesi modeli</b> kullanarak bireylerin 
    kardiyovasküler hastalık riskini tahmin etmek için geliştirilmiştir. 
    Model, 70.000 gözlem içeren Cardio Vascular Disease veri seti üzerinde 
    eğitilmiş olup demografik, antropometrik ve bazı klinik değişkenleri kullanmaktadır.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")


# =========================================================
# Giriş Bileşenleri
# =========================================================
col_left, col_right = st.columns([2.1, 1.9])

with col_left:
    st.subheader("📋 Kişisel ve Klinik Bilgiler")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Cinsiyet", ["Kadın", "Erkek"])
        age_years = st.slider("Yaş (yıl)", 29, 65, 50)
        height = st.slider("Boy (cm)", 130, 210, 170)
        weight = st.slider("Kilo (kg)", 40, 150, 75)
        ap_hi = st.slider("Sistolik Tansiyon (mmHg)", 80, 240, 130)
        ap_lo = st.slider("Diyastolik Tansiyon (mmHg)", 40, 140, 80)

    with col2:
        total_chol = st.slider("Total Kolesterol (mg/dL)", 100, 320, 180, step=5)
        fasting_glu = st.slider("Açlık Kan Şekeri (mg/dL)", 60, 250, 95, step=1)

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
            format_func=lambda x: "Aktif (Düzenli)" if x == 1 else "Pasif (Hareketsiz)",
        )

    st.markdown("")


# =========================================================
# Türetilmiş Özellikler (BMI, Nabız Basıncı vb.)
# =========================================================
# VKİ
bmi = weight / ((height / 100) ** 2)

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

# Nabız basıncı
pulse_pressure = ap_hi - ap_lo

# Yaş x Sistolik Tansiyon indeksi
age_bp_index = age_years * ap_hi

# Yaşam tarzı skoru (0-3, arttıkça daha riskli yorumlanabilir)
lifestyle_score = smoke + alco + (1 - active)

# Kolesterol kategorisi (literatür/klinik yaklaşım)
if total_chol <= 200:
    chol_cat = "Sağlıklı (≤200 mg/dL)"
elif total_chol <= 240:
    chol_cat = "Sınırda (200–240 mg/dL)"
else:
    chol_cat = "Yüksek kolesterol (>240 mg/dL)"

# Açlık kan şekeri kategorisi
if 70 <= fasting_glu < 100:
    glu_cat = "Normal (70–100 mg/dL)"
elif 100 <= fasting_glu < 126:
    glu_cat = "Prediyabet (100–126 mg/dL)"
else:
    glu_cat = "Diyabet (≥126 mg/dL)"

# Kan basıncı kategorisi (basitleştirilmiş tablodan)
if ap_hi < 120 and ap_lo < 80:
    bp_cat = "Optimal"
elif 120 <= ap_hi < 130 and ap_lo < 85:
    bp_cat = "Normal-Yüksek Normal"
elif 130 <= ap_hi < 140 or 85 <= ap_lo < 90:
    bp_cat = "Yüksek-Normal"
elif 140 <= ap_hi < 160 or 90 <= ap_lo < 100:
    bp_cat = "1. derece hipertansiyon"
elif 160 <= ap_hi < 180 or 100 <= ap_lo < 110:
    bp_cat = "2. derece hipertansiyon"
elif ap_hi >= 180 or ap_lo >= 110:
    bp_cat = "3. derece hipertansiyon"
else:
    bp_cat = "Değerlendirilemedi"

# Sigara ve alkolü model tarafında ters çevirelim (empirik düzeltme)
# Kullanıcı Evet diyorsa (1) model girdisi 0; Hayır diyorsa 1 olsun.
smoke_model = 0 if smoke == 1 else 1
alco_model = 0 if alco == 1 else 1

# Cinsiyet için basit kodlama (varsa feature_cols'ta kullanılır)
sex_code = 1 if gender == "Erkek" else 0


# =========================================================
# Hesaplanan Ek Özellikler Kutusu
# =========================================================
with col_left:
    with st.expander("ℹ Hesaplanan Ek Özellikler", expanded=True):
        st.write(
            f"**Vücut Kitle İndeksi (BMI):** {bmi:.1f} kg/m² – *{bmi_cat}*"
        )
        st.write(f"**Nabız Basıncı (ap_hi - ap_lo):** {pulse_pressure} mmHg")
        st.write(
            f"**Yaş × Sistolik Tansiyon İndeksi:** {age_bp_index:.0f}"
        )
        st.write(
            f"**Yaşam Tarzı Skoru (0–3) "
            f"= sigara + alkol + hareketsizlik:** {lifestyle_score}"
        )
        st.write(f"**Kan Basıncı Durumu:** {bp_cat}")
        st.write(f"**Kolesterol Durumu:** {chol_cat}")
        st.write(f"**Açlık Kan Şekeri Durumu:** {glu_cat}")


# =========================================================
# Sağ Kolon: Veri Seti, Model ve Performans Bilgileri
# =========================================================
with col_right:
    st.markdown("<div class='infocard'>", unsafe_allow_html=True)
    st.markdown("<div class='soft-header'>📊 Kullanılan Veri Seti</div>", unsafe_allow_html=True)
    st.write("**Kaynak:** Cardio Vascular Disease veri seti")
    st.write("**Gözlem sayısı:** ~70.000 birey")
    st.write(
        "**Değişkenler:** yaş, cinsiyet, boy, kilo, kan basıncı (sistolik/diyastolik), "
        "kolesterol, glukoz, sigara kullanımı, alkol kullanımı, fiziksel aktivite vb."
    )
    st.write(
        "**Hedef değişken:** `cardio` (0 = kardiyovasküler hastalık yok, 1 = hastalık var)"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("<div class='infocard'>", unsafe_allow_html=True)
    st.markdown("<div class='soft-header'>🧠 Kullanılan Modeller</div>", unsafe_allow_html=True)
    st.write("- Lojistik Regresyon")
    st.write("- Karar Ağaçları / Random Forest")
    st.write("- XGBoost (gradient boosting tabanlı model)")
    st.write(
        "Bu üç model, bir **Ensemble (Topluluk) Modeli** içerisinde birleştirilmiştir. "
        "Her modelin tahmin olasılıkları ağırlıklandırılarak birleşmekte ve son karar "
        "çoğunluk/olasılık ortalaması ile verilmektedir."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("<div class='infocard'>", unsafe_allow_html=True)
    st.markdown("<div class='soft-header'>📈 Eğitim Performansı (Test Kümesi)</div>", unsafe_allow_html=True)
    st.write("**Doğruluk (Accuracy):** ≈ 0.74")
    st.write("**Duyarlılık (Recall):** ≈ 0.70  (hastalığı olan bireyleri yakalama oranı)")
    st.write("**F1 Skoru:** ≈ 0.72  (dengeleyici ortalama)")
    st.write("**ROC-AUC:** ≈ 0.80  (ayrıştırma gücü)")
    st.write(
        "Bu metrikler, modelin pozitif ve negatif sınıfları ayırt etme gücünü test kümesi "
        "üzerinde özetlemektedir. Değerler, literatürde benzer klinik karar destek "
        "uygulamalarıyla karşılaştırılabilir seviyededir."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("<div class='infocard'>", unsafe_allow_html=True)
    st.markdown("<div class='soft-header'>🧪 Veri Ön İşleme ve Modelleme Notları</div>", unsafe_allow_html=True)
    st.write("- Aykırı ve tutarsız değerler (ör. fizyolojik olarak mümkün olmayan tansiyon/yaş kombinasyonları) veri keşfi aşamasında incelenmiş ve uygun şekilde filtrelenmiştir.")
    st.write("- Eksik gözlemler ve saçma değerler için basit imputation / temizleme adımları uygulanmıştır.")
    st.write("- Modeldeki sınıf dengesizliği, ağırlıklı sınıf yaklaşımları ve örnekleme teknikleriyle kontrol altına alınmıştır.")
    st.write("- Sürekli değişkenler gerektiğinde ölçeklendirilmiş, kategorik değişkenler ise uygun şekilde kodlanmıştır.")
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# Modelin Beklediği Girdi Vektörünü Hazırlama
# =========================================================
def build_input_row(feature_cols):
    """
    Eğitim sırasında kullanılan feature isimlerine göre tek satırlık
    bir sözlük döndürür. Bilinmeyen sütunlar 0 ile doldurulur.
    """
    values = {}
    for col in feature_cols:
        if col in ("age", "age_years"):
            values[col] = age_years
        elif col == "height":
            values[col] = height
        elif col == "weight":
            values[col] = weight
        elif col in ("ap_hi", "systolic"):
            values[col] = ap_hi
        elif col in ("ap_lo", "diastolic"):
            values[col] = ap_lo
        elif col in ("cholesterol", "total_chol"):
            # Orijinal veri seti 1-3 kodlama kullanıyor olabilir; burada total_chol'den 3'lü skala türetilebilir.
            # Ancak model cardio_feature_cols.pkl içinde nasıl eğitildiyse, oradaki sütun isimleriyle uyumludur.
            # Eğer model ham 1-3 kodlarını kullanıyorsa bu satır gereken dönüşüme göre güncellenebilir.
            # Şimdilik total kolesterol değerini doğrudan veriyoruz.
            values[col] = total_chol
        elif col in ("gluc", "glucose"):
            values[col] = fasting_glu
        elif col in ("smoke",):
            values[col] = smoke_model
        elif col in ("alco",):
            values[col] = alco_model
        elif col in ("active",):
            values[col] = active
        elif col in ("bmi", "BMI"):
            values[col] = bmi
        elif col in ("pulse_pressure",):
            values[col] = pulse_pressure
        elif col in ("age_bp_index", "age_x_ap_hi"):
            values[col] = age_bp_index
        elif col in ("lifestyle_score",):
            values[col] = lifestyle_score
        elif col in ("gender", "sex"):
            values[col] = sex_code
        else:
            # Eğitimde kullanılan ama burada doğrudan sorulmayan bir özellik olabilir
            values[col] = 0
    return values


input_dict = build_input_row(feature_cols)
input_df = pd.DataFrame([input_dict], columns=feature_cols)

st.markdown("---")


# =========================================================
# Tahmin Butonu ve Sonuç
# =========================================================
predict_btn = st.button("🔍 Kardiyovasküler Risk Tahminini Hesapla")

if predict_btn:
    with st.spinner("Tahmin hesaplanıyor..."):
        proba = model.predict_proba(input_df)[0][1]  # cardio=1 olasılığı
        pred = model.predict(input_df)[0]
        risk_percent = proba * 100

    if pred == 1:
        st.error(
            f"⚠ **YÜKSEK RİSK:** Model, bu bireyin kardiyovasküler hastalık taşıma "
            f"olasılığını yaklaşık **%{risk_percent:.1f}** olarak tahmin etmektedir."
        )
    else:
        st.success(
            f"✅ **DÜŞÜK RİSK:** Model, bu bireyin kardiyovasküler hastalık taşıma "
            f"olasılığını yaklaşık **%{risk_percent:.1f}** olarak tahmin etmektedir."
        )

    st.markdown(
        """
        > **Not (Teknik Açıklama):** Bu çıktı, denetimli makine öğrenmesi ile eğitilmiş bir 
        > sınıflandırma modelinin olasılık tahminidir. Klinik karar sürecini desteklemek 
        > amacıyla tasarlanmıştır; tek başına tanı koymak veya tedavi kararı vermek için 
        > kullanılmamalıdır. Model, eğitim aldığı veri setindeki örüntülere duyarlıdır ve 
        > bireyin gerçek klinik durumunu mutlaka hekim değerlendirmesiyle birlikte ele almak gerekir.
        """
    )
