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
# Basit tema / CSS düzeni
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    body { font-family: "Segoe UI", sans-serif; background-color: #f7f9fc; }
    .main { padding-top: 10px; }
    .app-title { text-align:center; font-size:32px; font-weight:700; margin-bottom:4px; }
    .app-subtitle { text-align:center; font-size:14px; color:#444; max-width:1000px; margin:0 auto 20px auto; }

    .info-card {
        background-color:#ffffff;
        border-radius:10px;
        padding:14px 18px;
        box-shadow:0 2px 6px rgba(15,23,42,0.08);
        border:1px solid #e5e7eb;
        margin-bottom:12px;
        font-size:12px;
    }
    .info-card h4 { margin-top:0; margin-bottom:6px; font-size:15px; font-weight:600; color:#111827; }

    .feature-box { font-size:11.5px; line-height:1.5; }

    .stSlider > div > div > div > div { background: linear-gradient(90deg,#ec4899,#6366f1); }
    .stSlider > div > div > div:nth-child(2) > div { background-color:#e5e7eb; }

    .stButton>button {
        background: linear-gradient(90deg,#ec4899,#6366f1);
        color:white; border-radius:999px; border:none;
        padding:0.45rem 1.4rem; font-size:0.9rem; font-weight:600;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg,#db2777,#4f46e5);
    }

    .tech-note { font-size:11px; color:#6b7280; margin-top:6px; text-align:justify; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# MODELİ YÜKLE
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
# BAŞLIK VE GENEL AÇIKLAMA
# =========================================================
st.markdown(
    "<div class='app-title'>🫀 Kardiyovasküler Hastalık Risk Tahmin Modeli</div>",
    unsafe_allow_html=True,
)

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
# LAYOUT
# =========================================================
left_col, right_col = st.columns([1.3, 1.0])

# =========================================================
# SOL SÜTUN
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
        total_chol = st.slider("Total Kolesterol (mg/dL)", 120, 320, 200, step=5)
        fasting_glucose = st.slider("Açlık Kan Şekeri (mg/dL)", 60, 250, 95, step=1)
        smoke = st.selectbox("Sigara Kullanımı", [0,1], format_func=lambda x: "Evet" if x==1 else "Hayır")
        alco = st.selectbox("Alkol Kullanımı", [0,1], format_func=lambda x: "Evet" if x==1 else "Hayır")
        active = st.selectbox("Fiziksel Aktivite", [0,1], format_func=lambda x: "Aktif" if x==1 else "Pasif")

    # Ek özellikler
    bmi = weight / ((height / 100) ** 2)
    pulse_pressure = ap_hi - ap_lo
    age_bp_index = age_years * ap_hi
    lifestyle_score = (1 - smoke) + (1 - alco) + active  # 0–3

    st.markdown("")

    predict_btn = st.button("🔍 Kardiyovasküler Risk Tahminini Hesapla")
    st.caption("Lütfen tüm bilgileri girdikten sonra butona tıklayın. Model, tahmin sonucunu bu alanın hemen altında gösterecektir.")

    # Girdi sırası
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

    input_df = pd.DataFrame([[input_dict[c] for c in feature_cols]], columns=feature_cols)

    # =========================================================
    # EK ÖZELLİKLER
    # =========================================================
    with st.expander("ℹ Hesaplanan Ek Özellikler", expanded=True):
        st.markdown(
            f"""
            <div class="feature-box">
            <b>Vücut Kitle İndeksi (BMI):</b> {bmi:.1f} kg/m²<br>
            <b>Nabız Basıncı:</b> {pulse_pressure} mmHg<br>
            <b>Yaş × Sistolik Tansiyon:</b> {age_bp_index}<br>
            <b>Yaşam Tarzı Skoru:</b> {lifestyle_score}<br>
            <b>Kolesterol Durumu:</b> {"Sağlıklı (<200)" if total_chol<=200 else "Sınırda (200–240)" if total_chol<=240 else "Yüksek (>240)"}<br>
            <b>Açlık Kan Şekeri:</b> {"Normal (70–100)" if 70<=fasting_glucose<100 else "Prediyabet (100–126)" if fasting_glucose<126 else "Diyabet (≥126)"}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================
    # RİSK TAHMİNİ (HTML KUTULARI)
    # =========================================================
    if predict_btn:

        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]
        risk_yuzde = prob * 100

        if pred == 1:
            risk_html = f"""
            <div style="
                background:#fdecec; border:1px solid #f8c7c7; color:#b91c1c;
                padding:14px; border-radius:8px; font-size:15px; margin-top:12px;">
                <span style="margin-right:6px;">⚠️⚠️</span>
                <b>YÜKSEK RİSK:</b> Model, bu bireyin kardiyovasküler hastalık 
                geliştirme olasılığını yaklaşık <b>%{risk_yuzde:.1f}</b> olarak tahmin etmektedir.
            </div>
            """
            st.markdown(risk_html, unsafe_allow_html=True)

        else:
            ok_html = f"""
            <div style="
                background:#ecfdf5; border:1px solid #a7f3d0; color:#065f46;
                padding:14px; border-radius:8px; font-size:15px; margin-top:12px;">
                <span style="margin-right:6px;">✅</span>
                <b>DÜŞÜK RİSK:</b> Model, bu bireyin kardiyovasküler hastalık 
                geliştirme olasılığını yaklaşık <b>%{risk_yuzde:.1f}</b> olarak tahmin etmektedir.
            </div>
            """
            st.markdown(ok_html, unsafe_allow_html=True)

        st.markdown(
            """
            <div class='tech-note'>
            <b>Teknik Açıklama:</b> Olasılık değeri, modelin eğitim setindeki benzer gözlemler 
            üzerinden sınıf dağılımını öğrenmesi ile hesaplanmıştır. Çıktı yalnızca klinik karar 
            destek amaçlıdır; doğrudan tanı veya tedavi amacıyla kullanılmamalıdır.
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# SAĞ SÜTUN (Bilgi Kartları)
# =========================================================
with right_col:

    st.markdown(
        """
        <div class="info-card">
            <h4>📊 Kullanılan Veri Seti</h4>
            <ul>
                <li><b>Kaynak:</b> Cardio Vascular Disease veri seti</li>
                <li><b>Gözlem sayısı:</b> ~70.000</li>
                <li><b>Değişkenler:</b> yaş, boy, kilo, tansiyon, kolesterol, glikoz, yaşam tarzı</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <h4>🧪 Veri Ön İşleme</h4>
            <ul>
                <li>Aykırı tansiyon değerleri filtrelendi.</li>
                <li>Kayıp değerler akıllı imputasyon ile tamamlandı.</li>
                <li>Kategorik değişkenler uygun şekilde kodlandı.</li>
                <li>Sürekli değişkenler gerektiğinde ölçeklendirildi.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <h4>🧠 Kullanılan Yapay Zekâ Modelleri</h4>
            <ul>
                <li><b>Lojistik Regresyon:</b> Temel doğrusal risk ilişkilerini yakalar.</li>
                <li><b>Random Forest:</b> Karmaşık ilişkileri ve etkileşimleri öğrenir.</li>
                <li><b>XGBoost:</b> Gradyan artırmalı güçlü bir sınıflandırma algoritmasıdır.</li>
                <li>Bu modeller birlikte <b>ensemble (oylama)</b> yapısında birleştirilmiştir.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <h4>📈 Eğitim Performansı</h4>
            <ul>
                <li><b>Accuracy:</b> ~0.74</li>
                <li><b>Recall:</b> ~0.70</li>
                <li><b>F1 Score:</b> ~0.72</li>
                <li><b>ROC-AUC:</b> ~0.80</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
