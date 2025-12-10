import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

# =========================================================
# SAYFA AYARLARI
# =========================================================
st.set_page_config(
    page_title="Kardiyovasküler Hastalık Risk Tahmin Modeli",
    page_icon="🫀",
    layout="wide"
)

# ---------------------------------------------------------
# CSS DÜZENİ
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    body { font-family: "Segoe UI"; background-color: #f7f9fc; }
    .main { padding-top: 10px; }

    .app-title { text-align:center; font-size:32px; font-weight:700; margin-bottom:4px; }
    .app-subtitle { text-align:center; font-size:15px; color:#555; max-width:960px; margin:0 auto 18px auto; }

    .info-card {
        background:white; border-radius:10px; padding:14px 18px;
        box-shadow:0 2px 6px rgba(0,0,0,0.06);
        border:1px solid #e5e7eb; margin-bottom:12px; font-size:13px;
    }

    .info-card h4 { margin-top:0; font-size:15px; font-weight:600; }

    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #ec4899, #6366f1);
    }
    .stSlider > div > div > div:nth-child(2) > div {
        background-color: #e5e7eb;
    }

    .stButton>button {
        background: linear-gradient(90deg, #ec4899, #6366f1);
        color:white; border-radius:999px; border:none;
        padding:0.45rem 1.4rem; font-size:0.9rem; font-weight:600;
    }
    .stButton>button:hover { background: linear-gradient(90deg, #db2777, #4f46e5); }

    .tech-note {
        font-size:11px; color:#6b7280; margin-top:4px; text-align:justify;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# MODEL YÜKLEME
# =========================================================
@st.cache_resource
def load_model():
    file_id = "1WdRoUATILi2VUCuyOEFAnrpoVJ7t69y-"
    url = f"https://drive.google.com/uc?id={file_id}"
    model_path = "cardio_ensemble_model.pkl"

    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)

    model = joblib.load(model_path)
    feature_cols = joblib.load("cardio_feature_cols.pkl")
    return model, feature_cols


model, feature_cols = load_model()

# =========================================================
# 1) PRIOR-ADJUSTMENT İLE OLABİLİRLİK KALİBRASYONU
# =========================================================
def calibrate_probability(p_ml, train_prevalence=0.50, population_prevalence=0.10):
    eps = 1e-6
    p = min(max(p_ml, eps), 1 - eps)

    old_odds = p / (1 - p)
    train_odds = train_prevalence / (1 - train_prevalence)
    pop_odds = population_prevalence / (1 - population_prevalence)

    prior_ratio = pop_odds / train_odds

    new_odds = old_odds * prior_ratio
    new_p = new_odds / (1 + new_odds)
    return new_p


# =========================================================
# BAŞLIK
# =========================================================
st.markdown("<div class='app-title'>🫀 Kardiyovasküler Hastalık Risk Tahmin Modeli</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class='app-subtitle'>
    Bu web arayüzü, lojistik regresyon, karar ağaçları ve XGBoost tabanlı bir 
    <b>ensemble (topluluk) makine öğrenmesi modeli</b> kullanarak bireylerin kardiyovasküler
    hastalık riskini tahmin etmek için geliştirilmiştir. Model, yaklaşık 70.000 gözlem içeren 
    Cardio Vascular Disease veri seti üzerinde eğitilmiş olup demografik, antropometrik 
    ve bazı klinik değişkenleri kullanmaktadır.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# =========================================================
# SAYFA LAYOUT
# =========================================================
left_col, right_col = st.columns([1.3, 1.0])

# =========================================================
# SOL SÜTUN – KİŞİSEL BİLGİLER
# =========================================================
with left_col:
    st.subheader("📋 Kişisel ve Klinik Bilgiler")

    c1, c2 = st.columns(2)

    with c1:
        gender = st.selectbox("Cinsiyet", ["Kadın", "Erkek"])
        age_years = st.slider("Yaş (yıl)", 29, 65, 50)
        height = st.slider("Boy (cm)", 130, 210, 170)
        weight = st.slider("Kilo (kg)", 40, 150, 75)
        ap_hi = st.slider("Sistolik Tansiyon (mmHg)", 80, 240, 130)
        ap_lo = st.slider("Diyastolik Tansiyon (mmHg)", 40, 180, 80)

    with c2:
        total_chol = st.slider("Total Kolesterol (mg/dL)", 120, 320, 200)
        fasting_glucose = st.slider("Açlık Kan Şekeri (mg/dL)", 60, 250, 95)

        smoke = st.selectbox("Sigara Kullanımı", [0, 1], format_func=lambda x: "Evet" if x else "Hayır")
        alco = st.selectbox("Alkol Kullanımı", [0, 1], format_func=lambda x: "Evet" if x else "Hayır")
        active = st.selectbox("Fiziksel Aktivite", [0, 1], format_func=lambda x: "Aktif" if x else "Pasif")

    # ----------------------------------------------
    # TÜRETİLMİŞ ÖZELLİKLER
    # ----------------------------------------------
    bmi = weight / ((height / 100) ** 2)
    pulse_pressure = ap_hi - ap_lo
    age_bp_index = age_years * ap_hi
    lifestyle_score = (1 - smoke) + (1 - alco) + active

    predict_btn = st.button("🔍 Kardiyovasküler Risk Tahminini Hesapla")
    st.caption("Lütfen tüm bilgileri girdikten sonra butona tıklayın.")

    input_dict = {
        "age_years": age_years,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": total_chol,
        "gluc": fasting_glucose,
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "bmi": bmi,
        "pulse_pressure": pulse_pressure,
        "age_bp_index": age_bp_index,
        "lifestyle_score": lifestyle_score,
    }

    input_df = pd.DataFrame([[input_dict[col] for col in feature_cols]], columns=feature_cols)

    # ----------------------------------------------
    # EK ÖZELLİKLER ALANI
    # ----------------------------------------------
    with st.expander("ℹ Hesaplanan Ek Özellikler", expanded=True):
        st.markdown(
            f"""
            <div>
            <b>BMI:</b> {bmi:.1f} kg/m²<br>
            <b>Nabız Basıncı:</b> {pulse_pressure} mmHg<br>
            <b>Yaş × Sistolik Tansiyon:</b> {age_bp_index}<br>
            <b>Yaşam Tarzı Skoru:</b> {lifestyle_score}<br>
            <b>Kolesterol Durumu:</b> {total_chol}<br>
            <b>Glukoz Durumu:</b> {fasting_glucose}<br>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ----------------------------------------------
    # TAHMİN
    # ----------------------------------------------
    if predict_btn:
        prob_raw = model.predict_proba(input_df)[0][1]
        prob = calibrate_probability(prob_raw, 0.50, 0.10)  # ← EN KRİTİK ADIM
        pred = 1 if prob > 0.20 else 0          # ← Yeni karar eşiği (literatüre uygun)

        risk_yuzde = prob * 100

        if pred == 1:
            st.error(
                f"⚠️ <b>YÜKSEK RİSK:</b> Bu bireyin kardiyovasküler hastalık geliştirme olasılığı yaklaşık <b>%{risk_yuzde:.1f}</b> olarak tahmin edilmektedir.",
                unsafe_allow_html=True
            )
        else:
            st.success(
                f"✅ <b>DÜŞÜK RİSK:</b> Bu bireyin kardiyovasküler hastalık geliştirme olasılığı yaklaşık <b>%{risk_yuzde:.1f}</b> olarak tahmin edilmektedir.",
                unsafe_allow_html=True
            )

        st.markdown(
            """
            <div class='tech-note'>
            <b>Teknik Açıklama:</b> Gösterilen olasılık, eğitim veri seti üzerinde oluşturulan topluluk modelinin ham tahmini,
            kardiyovasküler hastalık prevalansına ilişkin literatürden alınmış oranlarla yeniden kalibre edilerek hesaplanmıştır.
            Bu çıktı, bireylerin göreli risk düzeyini anlamaya yardımcı olmayı amaçlayan bir karar destek göstergesidir;
            klinik tanı veya tedavi kararı yerine geçmez.
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# SAĞ SÜTUN – BİLGİ KARTLARI
# =========================================================
with right_col:

    st.markdown(
        """
        <div class="info-card">
            <h4>📊 Kullanılan Veri Seti</h4>
            Cardio Vascular Disease veri seti (~70.000 gözlem).
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <h4>🧪 Veri Ön İşleme</h4>
            Aykırı tansiyon değerleri filtrelendi, kayıp değerler imputasyonla tamamlandı,
            sürekli değişkenler gerektiğinde ölçeklendirildi.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <h4>🧠 Kullanılan Modeller</h4>
            Lojistik Regresyon, Random Forest ve XGBoost modelleri
            bir ensemble yapısı içinde birleştirilmiştir.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <h4>📈 Eğitim Performansı</h4>
            Accuracy ≈ 0.74 — Recall ≈ 0.70 — F1 ≈ 0.72 — ROC-AUC ≈ 0.80
        </div>
        """,
        unsafe_allow_html=True,
    )
