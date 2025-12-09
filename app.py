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
    /* Genel arka plan ve font */
    body {
        font-family: "Segoe UI", sans-serif;
        background-color: #f7f9fc;
    }

    .main {
        padding-top: 10px;
    }

    /* Başlık */
    .app-title {
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .app-subtitle {
        text-align: center;
        font-size: 15px;
        color: #555;
        max-width: 950px;
        margin: 0 auto 20px auto;
        line-height: 1.5;
    }

    /* Kart tasarımı */
    .info-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
        border: 1px solid #e5e7eb;
        margin-bottom: 12px;
        font-size: 12px;
    }
    .info-card h4 {
        margin-top: 0;
        margin-bottom: 6px;
        font-size: 14px;
        font-weight: 600;
        color: #111827;
    }
    .info-card ul {
        padding-left: 18px;
        margin-bottom: 0;
    }

    /* Ek özellikler alanı */
    .feature-box {
        font-size: 11.5px;
        line-height: 1.5;
    }

    /* Slider rengi (daha yumuşak) */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #ec4899, #6366f1);
    }
    .stSlider > div > div > div:nth-child(2) > div {
        background-color: #e5e7eb;
    }

    /* Tahmin butonu */
    .stButton>button {
        background: linear-gradient(90deg, #ec4899, #6366f1);
        color: white;
        border-radius: 999px;
        border: none;
        padding: 0.45rem 1.4rem;
        font-size: 0.9rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #db2777, #4f46e5);
    }

    /* Teknik not */
    .tech-note {
        font-size: 11px;
        color: #6b7280;
        margin-top: 4px;
        text-align: justify;
    }
    .hint-text {
        font-size: 11px;
        color: #6b7280;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# MODELİ YÜKLE
# =========================================================
@st.cache_resource
def load_model():
    """
    Eğer model sunucu dizininde yoksa Google Drive'dan indirir,
    ardından eğitilmiş topluluk modelini ve feature isimlerini yükler.
    """
    file_id = "1WdRoUATILi2VUCuyOEFAnrpoVJ7t69y-"
    url = f"https://drive.google.com/uc?id={file_id}"
    model_path = "cardio_ensemble_model.pkl"

    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)

    # Ana topluluk model
    model = joblib.load(model_path)
    # Eğitim sırasında kullanılan feature sırası
    feature_cols = joblib.load("cardio_feature_cols.pkl")

    return model, feature_cols


model, feature_cols = load_model()

# Yardımcı fonksiyonlar (kolesterol / glukoz kategorileri)
def chol_category(total_chol):
    """mg/dL -> 1/2/3 kategorisi"""
    if total_chol <= 200:
        return 1  # Normal
    elif total_chol <= 240:
        return 2  # Sınırda
    else:
        return 3  # Yüksek


def gluc_category(fasting_glucose):
    """mg/dL -> 1/2/3 kategorisi"""
    if fasting_glucose < 100:
        return 1  # Normal
    elif fasting_glucose < 126:
        return 2  # Prediyabet
    else:
        return 3  # Diyabet


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
# SAYFA YERLEŞİMİ
# =========================================================
left_col, right_col = st.columns([1.3, 1.0])

# =========================================================
# SOL SÜTUN: KİŞİSEL VE KLİNİK BİLGİLER
# =========================================================
with left_col:
    st.subheader("📋 Kişisel ve Klinik Bilgiler")

    c1, c2 = st.columns(2)

    with c1:
        gender = st.selectbox("Cinsiyet", options=["Kadın", "Erkek"])
        age_years = st.slider("Yaş (yıl)", 29, 65, 50)
        height = st.slider("Boy (cm)", 130, 210, 170)
        weight = st.slider("Kilo (kg)", 40, 150, 75)
        ap_hi = st.slider("Sistolik Tansiyon (mmHg)", 80, 240, 130)
        ap_lo = st.slider("Diyastolik Tansiyon (mmHg)", 40, 180, 80)

    with c2:
        total_chol = st.slider("Total Kolesterol (mg/dL)", 120, 320, 200, step=5)
        fasting_glucose = st.slider("Açlık Kan Şekeri (mg/dL)", 60, 250, 95, step=1)

        # 0 = Hayır, 1 = Evet (UI için)
        smoke_ui = st.selectbox(
            "Sigara Kullanımı",
            options=[0, 1],
            format_func=lambda x: "Evet" if x == 1 else "Hayır",
        )
        alco_ui = st.selectbox(
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

    # -----------------------------------------------------
    # TÜRETİLMİŞ ÖZELLİKLER
    # -----------------------------------------------------
    bmi = weight / ((height / 100) ** 2)
    pulse_pressure = ap_hi - ap_lo
    age_bp_index = age_years * ap_hi

    # Literatüre dayalı yaşam tarzı skoru:
    # Sigara (Evet)  -> +1
    # Alkol (Evet)   -> +1
    # Pasif (0)      -> +1
    lifestyle_score = smoke_ui + alco_ui + (1 - active)  # 0–3, yüksek = daha riskli

    # Kolesterol / Glukoz kategorileri (modele böyle gidiyor)
    chol_cat = chol_category(total_chol)
    gluc_cat = gluc_category(fasting_glucose)

    # -----------------------------------------------------
    # TAHMİN BUTONU (ek özelliklerden önce)
    # -----------------------------------------------------
    st.markdown("")
    predict_btn = st.button("🔍 Kardiyovasküler Risk Tahminini Hesapla")

    st.markdown(
        "<div class='hint-text'>Lütfen tüm bilgileri girdikten sonra butona tıklayın. "
        "Model, tahmin sonucunu bu alanın hemen altında gösterecektir.</div>",
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # MODELE GİDECEK GİRDİLER (feature_cols sırasına göre)
    # -----------------------------------------------------
    # Model eğitiminde kullanılan anlamlarla tutarlı:
    # smoke/alco: 0=Hayır, 1=Evet (risk arttırıcı)
    input_dict = {
        "age_years": age_years,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": chol_cat,        # 1/2/3 kategori
        "gluc": gluc_cat,               # 1/2/3 kategori
        "smoke": smoke_ui,
        "alco": alco_ui,
        "active": active,
        "bmi": bmi,
        "pulse_pressure": pulse_pressure,
        "age_bp_index": age_bp_index,
        "lifestyle_score": lifestyle_score,
    }

    # DataFrame'i feature_cols sırasına göre oluştur
    input_df = pd.DataFrame([[input_dict[col] for col in feature_cols]],
                            columns=feature_cols)

    # ----------------------------------------------
    # HESAPLANAN EK ÖZELLİKLER
    # ----------------------------------------------
    with st.expander("ℹ Hesaplanan Ek Özellikler", expanded=True):
        st.markdown(
            f"""
            <div class="feature-box">
            <b>Vücut Kitle İndeksi (BMI):</b> {bmi:.1f} kg/m² – 
            {"Zayıf" if bmi < 18.5 else "Sağlıklı" if bmi < 25 else "Fazla kilolu" if bmi < 30 else "1. derece obezite" if bmi < 35 else "2. derece obezite" if bmi < 40 else "3. derece obezite"}<br>
            <b>Nabız Basıncı (ap_hi − ap_lo):</b> {pulse_pressure:.0f} mmHg<br>
            <b>Yaş × Sistolik Tansiyon İndeksi:</b> {age_bp_index:.0f}<br>
            <b>Yaşam Tarzı Skoru (0–3, yüksek skor = daha riskli):</b> {lifestyle_score} 
            (sigara: {'var' if smoke_ui else 'yok'}, alkol: {'var' if alco_ui else 'yok'}, aktivite: {'aktif' if active else 'pasif'})<br>
            <b>Kan Basıncı Kategorisi (sistolik/diyastolik):</b> {ap_hi}/{ap_lo} mmHg<br>
            <b>Kolesterol Durumu (mg/dL):</b> { "Sağlıklı (≤200)" if total_chol <= 200 else "Sınırda (200–240)" if total_chol <= 240 else "Yüksek (>240)" }<br>
            <b>Açlık Kan Şekeri Durumu (mg/dL):</b> { "Normal (<100)" if fasting_glucose < 100 else "Prediyabet (100–126)" if fasting_glucose < 126 else "Diyabet (≥126)" }
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ----------------------------------------------
    # TAHMİN ÇIKTISI
    # ----------------------------------------------
    if predict_btn:
        prob = model.predict_proba(input_df)[0][1]  # 1 sınıfı (hastalık) olasılığı
        pred = model.predict(input_df)[0]
        risk_yuzde = prob * 100

        if pred == 1:
            st.error(
                f"⚠️ YÜKSEK RİSK: Model, bu bireyin kardiyovasküler hastalık "
                f"geliştirme olasılığını yaklaşık %{risk_yuzde:.1f} olarak tahmin etmektedir."
            )
        else:
            st.success(
                f"✅ DÜŞÜK RİSK: Model, bu bireyin kardiyovasküler hastalık "
                f"geliştirme olasılığını yaklaşık %{risk_yuzde:.1f} olarak tahmin etmektedir."
            )

        st.markdown(
            """
            <div class='tech-note'>
            <b>Teknik Açıklama:</b> Olasılık, eğitim veri setinde oluşturulan topluluk
            modelinin, gözleme benzer bireylerin sınıf dağılımına dayalı tahminidir.
            Bu model, klinik kararı desteklemek için tasarlanmış bir karar destek sistemidir;
            tek başına tanı veya tedavi kararında kullanılmamalıdır.
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# SAĞ SÜTUN: BİLGİ KARTLARI
# =========================================================
with right_col:
    # ----------------- Kullanılan Veri Seti ----------------
    st.markdown(
        """
        <div class="info-card">
            <h4>📊 Kullanılan Veri Seti</h4>
            <ul>
                <li><b>Kaynak:</b> Cardio Vascular Disease veri seti</li>
                <li><b>Gözlem sayısı:</b> ~70.000 birey</li>
                <li><b>Değişkenler:</b> yaş, cinsiyet, boy, kilo, kan basıncı (sistolik/diyastolik),
                    kolesterol, glikoz, sigara ve alkol kullanımı, fiziksel aktivite vb.</li>
                <li><b>Hedef değişken:</b> <code>cardio</code> (0 = hastalık yok, 1 = kardiyovasküler hastalık var)</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----------------- Veri Ön İşleme ----------------------
    st.markdown(
        """
        <div class="info-card">
            <h4>🧪 Veri Ön İşleme ve Modellemenin Notları</h4>
            <ul>
                <li>Olası aykırı ve tutarsız değerler (özellikle kan basıncı kombinasyonları) 
                    veri keşfi aşamasında klinik eşikler dikkate alınarak filtrelenmiştir.</li>
                <li>Kayıp değerler, değişkenin dağılımına göre uygun <i>imputasyon</i> yöntemleriyle giderilmiştir.</li>
                <li>Sürekli değişkenler gerektiğinde ölçeklendirilmiş, kategorik değişkenler uygun biçimde kodlanmıştır.</li>
                <li>Model performansı, eğitim/test ayrımı ve sınıf dengesini gözeten metriklerle değerlendirilmiştir.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----------------- Kullanılan Modeller -----------------
    st.markdown(
        """
        <div class="info-card">
            <h4>🧠 Kullanılan Modeller</h4>
            <ul>
                <li><b>Lojistik Regresyon:</b> Doğrusal karar sınırı ile temel risk faktörlerinin katsayılarını
                    tahmin eden klasik bir denetimli öğrenme modelidir. Değişkenlerin etkisi istatistiksel olarak yorumlanabilir.</li>
                <li><b>Karar Ağaçları / Random Forest:</b> Özellikle doğrusal olmayan ilişkileri, eşik değerlerini ve
                    değişkenler arası etkileşimleri yakalayan ağaç tabanlı yapay zekâ modelleridir. Random Forest, birçok ağacın
                    birlikte oy kullanması ile daha kararlı tahminler üretir.</li>
                <li><b>XGBoost:</b> Art arda kurulan gradyan artırmalı karar ağaçlarından oluşan, hataları kademeli olarak azaltan
                    güçlü bir boosting algoritmasıdır. Özellikle tablosal klinik verilerde yüksek performansıyla bilinir.</li>
                <li>Bu üç modelin olasılık çıktıları, <b>ensemble (topluluk)</b> yapısı içinde birleştirilmiş ve bireysel model
                    hatalarını dengeleyen, daha genellenebilir bir risk tahmin sistemi elde edilmiştir.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----------------- Eğitim Performansı ------------------
    st.markdown(
        """
        <div class="info-card">
            <h4>📈 Eğitim Performansı (Test Kümesi)</h4>
            <ul>
                <li><b>Doğruluk (Accuracy):</b> ≈ 0.74</li>
                <li><b>Duyarlılık (Recall):</b> ≈ 0.70 (hastalığı olan bireyi yakalama oranı)</li>
                <li><b>F1 Skoru:</b> ≈ 0.72 (dengeli ortalama)</li>
                <li><b>ROC-AUC:</b> ≈ 0.80 (ayrıştırma gücü)</li>
                <li>Bu değerler, modelin kardiyovasküler hastalık var/yok ayrımını istatistiksel olarak anlamlı bir düzeyde
                    öğrendiğini göstermektedir.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
