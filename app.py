# -*- coding: utf-8 -*-
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# İsteğe bağlı: Colab/Cloud ilk kurulumlarda lazımsa aktif et
try:
    import gdown  # noqa: F401
except Exception:
    # Runtime'da gdown yoksa kur
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "gdown", "-q"], check=False)
import gdown


# =========================
# Sayfa Ayarları + Stil
# =========================
st.set_page_config(
    page_title="Kardiyovasküler Risk Tahmin Modeli",
    page_icon="❤️",
    layout="wide",
)

# yumuşak arkaplan, slider rengi, card görünümleri
st.markdown(
    """
<style>
/* Genel font ve arkaplan */
html, body, [class*="css"]  {
    font-family: "Inter", "Segoe UI", "Helvetica", Arial, sans-serif;
}

/* Başlığın üstündeki beyaz boşluğu daralt */
.block-container { padding-top: 1.2rem; }

/* Sliderları daha yumuşak renk yap */
.stSlider > div[data-baseweb="slider"] > div > div {
    background: linear-gradient(90deg, #4fb0ff 0%, #7fd7c5 100%) !important;
}
.stSlider > div[data-baseweb="slider"] > div > div > div {
    background-color: #0ea5e9 !important;
}

/* Kart (kutu) stili */
.card {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    border-radius: 10px;
    padding: 16px 18px;
}
.card h4 {
    margin: 0 0 8px 0;
    padding: 0;
}

/* Küçük bilgi badge'leri */
.badge {
    display: inline-block;
    background: #eef6ff;
    color: #2563eb;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 12px;
    border: 1px solid #dbeafe;
}

/* Sonuç kutularının metinleri biraz daha okunur */
.result-note {
    margin-top: 10px;
    font-size: 14px;
    color: #374151;
}

/* Bölüm başlıkları */
.section-title {
    font-weight: 700;
    font-size: 16.5px;
    margin-bottom: 8px;
}
.subtext {
    font-size: 13.5px; 
    color: #6b7280;
}
</style>
    """,
    unsafe_allow_html=True
)

# =========================
# Modeli ve kolonları yükle
# =========================
DRIVE_FILE_ID = "1WdRoUATILi2VUCuyOEFAnrpoVJ7t69y-"  # cardio_ensemble_model.pkl Google Drive ID
MODEL_PATH = "cardio_ensemble_model.pkl"
FEATURE_PATH = "cardio_feature_cols.pkl"  # repo içinde olmalı


@st.cache_resource(show_spinner=True)
def load_artifacts():
    """Model ve kolon listesini (feature_cols) yükler. Model yoksa Drive'dan indirir."""
    # Model yoksa Drive'dan indir
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_PATH)
    return model, feature_cols


model, feature_cols = load_artifacts()

# =========================
# Yardımcı fonksiyonlar
# =========================
def cholesterol_category(total_chol):
    """
    Literatür eşiği:
      ≤200 mg/dL -> 1 (Normal)
      200-239   -> 2 (Sınırda)
      ≥240      -> 3 (Yüksek)
    """
    if total_chol <= 200:
        return 1
    elif total_chol < 240:
        return 2
    else:
        return 3


def glucose_category(fbg):
    """
    Açlık Kan Şekeri:
      70-100   -> 1 (Normal)
      100-126  -> 2 (Prediyabet)
      ≥126     -> 3 (Diyabet)
      (70 altını da klinikte hipoglisemi olarak kabul ederiz, pratikte 1 tutuyoruz.)
    """
    if fbg < 100:
        return 1
    elif fbg < 126:
        return 2
    else:
        return 3


def bp_category(ap_hi, ap_lo):
    """
    Basitleştirilmiş tablo:
    - Optimal     : ap_hi <120 ve ap_lo <80
    - Normal      : 120–129 / 80–84
    - Yüksek Norm : 130–139 / 85–89
    - HT evre-1   : 140–159 / 90–99
    - HT evre-2   : 160–179 /100–109
    - HT evre-3   : ≥180 / ≥110
    Not: Klinik tablolar "ve/veya" geçer; burada daha yalın bir karar ağacı kullanıldı.
    """
    if ap_hi < 120 and ap_lo < 80:
        return "Optimal"
    if (120 <= ap_hi <= 129) or (80 <= ap_lo <= 84):
        return "Normal"
    if (130 <= ap_hi <= 139) or (85 <= ap_lo <= 89):
        return "Yüksek Normal"
    if (140 <= ap_hi <= 159) or (90 <= ap_lo <= 99):
        return "1. derece Hipertansiyon"
    if (160 <= ap_hi <= 179) or (100 <= ap_lo <= 109):
        return "2. derece Hipertansiyon"
    if ap_hi >= 180 or ap_lo >= 110:
        return "3. derece Hipertansiyon"
    return "—"


def build_input_row(feature_cols, mapping):
    """
    Modelin beklediği kolon sırasına göre tek satırlık DataFrame üretir.
    mapping: {'age_years': val, 'ap_hi': val, ...}
    """
    row = []
    for col in feature_cols:
        row.append(mapping.get(col, 0))
    return pd.DataFrame([row], columns=feature_cols)


# =========================
# Üst Başlık
# =========================
left, mid, right = st.columns([1, 6, 1])
with mid:
    st.markdown(
        "<h2 style='text-align:center; margin-bottom:2px;'>❤️ Kardiyovasküler Hastalık Risk Tahmin Modeli</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='subtext' style='text-align:center;'>"
        "Bu web arayüzü; <b>Lojistik Regresyon</b>, <b>Rastgele Orman</b> ve <b>XGBoost</b> tabanlı "
        "bir <b>ensemble (topluluk)</b> makine öğrenmesi modeli kullanarak bireylerin kardiyovasküler hastalık "
        "riskini tahmin eder. Model, 70.000 gözlem içeren Cardio Vascular Disease veri seti üzerinde eğitilmiş olup demografik, "
        "antropometrik ve hemodinamik göstergeleri kullanmaktadır."
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# Girdi Alanı
# =========================
c_left, c_right = st.columns([1.4, 1])

# ---- Sol: girdiler
with c_left:
    st.markdown("<div class='section-title'>📋 Kişisel ve Klinik Bilgiler</div>", unsafe_allow_html=True)
    sex = st.selectbox("Cinsiyet", ["Kadın", "Erkek"])
    gender = 1 if sex == "Erkek" else 0  # modelde 'gender' varsa kullanacağız

    age_years = st.slider("Yaş (yıl)", 29, 65, 46)
    height = st.slider("Boy (cm)", 130, 210, 170)
    weight = st.slider("Kilo (kg)", 40, 160, 96)
    ap_hi = st.slider("Sistolik Tansiyon (mmHg)", 80, 240, 130)
    ap_lo = st.slider("Diyastolik Tansiyon (mmHg)", 40, 180, 80)

    # Sağ tarafta kalan biyokimya ve yaşam tarzı girdileri:
    st.markdown("<br>", unsafe_allow_html=True)
    total_chol = st.slider("Total Kolesterol (mg/dL)", 100, 320, 215)
    fbg = st.slider("Açlık Kan Şekeri (mg/dL)", 70, 250, 113)

    smoke = st.selectbox("Sigara Kullanımı", ["Hayır", "Evet"])
    alco = st.selectbox("Alkol Kullanımı", ["Hayır", "Evet"])
    active = st.selectbox("Fiziksel Aktivite", ["Pasif (Hareketsiz)", "Aktif (Düzenli)"])

    # ---- Türetilen göstergeler
    bmi = weight / ((height / 100) ** 2)
    pulse_pressure = ap_hi - ap_lo
    age_bp_index = age_years * ap_hi

    # Yaşam tarzı skoru: sigara/alkol/aktif değil -> daha riskli => skor artmalı
    smoke_r = 1 if smoke == "Evet" else 0
    alco_r = 1 if alco == "Evet" else 0
    active_r = 1 if active.startswith("Aktif") else 0
    lifestyle_score = smoke_r + alco_r + (1 - active_r)  # 0-3 arasında, yüksek = daha riskli

    # Kategorik eşleştirmeler (model 1/2/3 bekliyorsa)
    chol_cat = cholesterol_category(total_chol)
    gluc_cat = glucose_category(fbg)
    bp_cat_text = bp_category(ap_hi, ap_lo)

    # ---- Tahmin butonu (ek özelliklerden önce)
    st.markdown(
        """
        <div style='margin-top: 8px; padding:10px; background:#eef6ff; border-left:4px solid #5b9bff; border-radius:6px;'>
        ℹ️ <b>Not:</b> Tüm değerleri girdikten sonra aşağıdaki butona basarak kardiyovasküler risk tahmininizi hesaplayabilirsiniz.
        </div>
        """,
        unsafe_allow_html=True
    )
    predict_btn = st.button("🔍 Kardiyovasküler Risk Tahminini Hesapla")

    # ---- Ek Özellikler (expandable)
    with st.expander("ℹ️ Hesaplanan Ek Özellikler"):
        st.markdown(
            f"- <b>Vücut Kitle İndeksi (BMI):</b> {bmi:.1f} kg/m²  "
            f"{'(18.5–24.9: normal, 25–29.9: fazla kilolu, 30+: obezite)'}  \n"
            f"- <b>Nabız Basıncı (ap_hi - ap_lo):</b> {pulse_pressure} mmHg  \n"
            f"- <b>Yaş × Sistolik Tansiyon İndeksi:</b> {age_bp_index}  \n"
            f"- <b>Kan Basıncı Kategorisi:</b> {bp_cat_text}  \n"
            f"- <b>Yaşam Tarzı Skoru (0–3; yüksek skor = daha riskli):</b> {lifestyle_score}",
            unsafe_allow_html=True
        )

# ---- Sağ: açıklama kutuları (kutu içinde ve sıralı)
with c_right:
    st.markdown("<div class='card'><h4>📚 Kullanılan Veri Seti</h4>", unsafe_allow_html=True)
    st.markdown(
        "- **Kaynak:** Cardio Vascular Disease veri seti  \n"
        "- **Gözlem Sayısı:** ~70.000 birey  \n"
        "- **Değişkenler:** Yaş, cinsiyet, boy, kilo, kan basıncı (sistolik/diyastolik), kolesterol, glikoz, sigara, alkol, fiziksel aktivite vb.  \n"
        "- **Hedef Değişken:** `cardio` (0 = hastalık yok, 1 = kardiyovasküler hastalık var)",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h4>🧪 Veri Ön İşleme</h4>", unsafe_allow_html=True)
    st.markdown(
        "- Aykırı ve tutarsız değerler (ör. klinik olarak uyumsuz tansiyon/kilo/boy kombinasyonları) veri keşif aşamasında incelenerek uygun şekilde filtrelendi.  \n"
        "- Eksik değerler için uygun yöntemler ve/veya istatistiksel yaklaşımlar kullanıldı.  \n"
        "- Sürekli değişkenler gerektiğinde ölçeklendirildi; kategorik değişkenler için uygun kodlama yapıldı.",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h4>🧠 Kullanılan Modeller</h4>", unsafe_allow_html=True)
    st.markdown(
        "- Lojistik Regresyon  \n"
        "- Karar Ağaçları / Rastgele Orman  \n"
        "- XGBoost (gradient boosting)  \n"
        "  \n"
        "Bu üç model, bir <b>Ensemble (Topluluk) Modeli</b> içerisinde birleştirilmiştir (olasılıkların ağırlıklı/çoğunluk oylaması).",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h4>📈 Eğitim Performansı (Test Kümesi)</h4>", unsafe_allow_html=True)
    st.markdown(
        "- **Doğruluk (Accuracy):** ~0.74  \n"
        "- **Duyarlılık (Recall):** ~0.70  \n"
        "- **F1-Skoru:** ~0.72  \n"
        "- **ROC-AUC:** ~0.80  \n"
        "<span class='subtext'>*Not*: Metrikler, modelin sınıflar arasındaki ayırımı test kümesi üzerinde anlamlı bir düzeyde öğrendiğini göstermektedir.</span>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Tahmin – mapping & sonuç
# =========================

# Modelin beklediği isimleri -> girdiler/hesaplananlar
name_to_value = {
    "gender": gender,
    "age_years": age_years,
    "height": height,
    "weight": weight,
    "ap_hi": ap_hi,
    "ap_lo": ap_lo,
    "cholesterol": chol_cat,   # 1/2/3 (Normal / Sınırda / Yüksek)
    "gluc": gluc_cat,          # 1/2/3 (Normal / Prediyabet / Diyabet)
    "smoke": smoke_r,          # 0/1
    "alco": alco_r,            # 0/1
    "active": active_r,        # 0/1 (Aktif=1)
    "bmi": float(bmi),
    "pulse_pressure": int(pulse_pressure),
    "age_bp_index": int(age_bp_index),
    "lifestyle_score": int(lifestyle_score),  # yüksek = daha riskli (düzeltildi)
    # Eğer modelde farklı mühendislikli alanlar varsa buraya ekleyebilirsin.
}

# Butona basılmadıysa bilgilendirme kutusu
if not predict_btn:
    st.info("Henüz tahmin yapılmadı. Lütfen bilgileri girip ‘Kardiyovasküler Risk Tahminini Hesapla’ butonuna tıklayınız.")
else:
    # Modelin beklediği sırada tek satırlık dataframe
    input_df = build_input_row(feature_cols, name_to_value)

    # Olasılık ve sınıf
    prob = model.predict_proba(input_df)[0][1]  # cardio=1 (hastalık) olasılığı
    pred = int(model.predict(input_df)[0])
    risk_pct = prob * 100

    if pred == 1:
        st.error(
            f"⚠️ <b>YÜKSEK RİSK:</b> Model, bu bireyin kardiyovasküler hastalık geliştirme olasılığını "
            f"yaklaşık <b>%{risk_pct:.1f}</b> olarak tahmin etmektedir.",
            unsafe_allow_html=True,
        )
    else:
        st.success(
            f"✅ <b>DÜŞÜK RİSK:</b> Model, bu bireyin kardiyovasküler hastalık geliştirme olasılığını "
            f"yaklaşık <b>%{risk_pct:.1f}</b> olarak tahmin etmektedir.",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class='result-note'>
        <b>Teknik Açıklama:</b> Olasılık, eğitim veri setinde oluşturulan topluluk modelinin,
        gözleme benzer bireylerin sınıf dağılımına dayalı tahminidir. Bu çıktı; klinik kararı desteklemek için
        tasarlanmış bir karar destek sistemidir; <u>tek başına tanı koymak veya tedavi planlamak için kullanılmamalıdır</u>.
        </div>
        """,
        unsafe_allow_html=True
    )
