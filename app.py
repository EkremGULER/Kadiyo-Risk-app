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
    body {
        font-family: "Segoe UI", sans-serif;
        background-color: #f7f9fc;
    }
    .main {
        padding-top: 10px;
    }
    .app-title {
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .app-subtitle {
        text-align: center;
        font-size: 15px;   /* bir tık büyüttüm */
        color: #555;
        max-width: 950px;
        margin: 0 auto 20px auto;
    }
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
    .feature-box {
        font-size: 11.5px;
        line-height: 1.5;
    }
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #ec4899, #6366f1);
    }
    .stSlider > div > div > div:nth-child(2) > div {
        background-color: #e5e7eb;
    }
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
    .tech-note {
        font-size: 11px;
        color: #6b7280;
        margin-top: 4px;
        text-align: justify;
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

    # -----------------------------------------------------
    # TÜRETİLMİŞ ÖZELLİKLER
    # -----------------------------------------------------
    bmi = weight / ((height / 100) ** 2)
    pulse_pressure = ap_hi - ap_lo
    age_bp_index = age_years * ap_hi

    # Yaşam tarzı skoru (0 = en kötü, 3 = en iyi)
    lifestyle_score = (1 - smoke) + (1 - alco) + active

    # ---- ÖNEMLİ: kolesterol ve glukozu 1–3 kategoriye çevir ----
    # Veri setindeki tanım:
    # 1: normal, 2: yüksek, 3: çok yüksek
    if total_chol <= 200:
        chol_cat = 1
    elif total_chol <= 240:
        chol_cat = 2
    else:
        chol_cat = 3

    # 1: normal, 2: yüksek, 3: çok yüksek
    if fasting_glucose < 100:
        gluc_cat = 1
    elif fasting_glucose < 126:
        gluc_cat = 2
    else:
        gluc_cat = 3

    # -----------------------------------------------------
    # TAHMİN BUTONU
    # -----------------------------------------------------
    st.markdown("")
    predict_btn = st.button("🔍 Kardiyovasküler Risk Tahminini Hesapla")
    st.caption(
        "Lütfen yukarıdaki bilgileri girdikten sonra bu butona tıklayın. "
        "Model, tahmin sonucunu bu alanın hemen altında gösterecektir."
    )

    # Modelin beklediği girdiler (feature_cols ile aynı isimler)
    input_dict = {
        "age_years": age_years,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": chol_cat,      # KATEGORİ veriyoruz
        "gluc": gluc_cat,             # KATEGORİ veriyoruz
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "bmi": bmi,
        "pulse_pressure": pulse_pressure,
        "age_bp_index": age_bp_index,
        "lifestyle_score": lifestyle_score,
    }

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
            <b>Yaşam Tarzı Skoru (0–3, yüksek skor = daha sağlıklı):</b> {lifestyle_score}
            (sigara: {'var' if smoke else 'yok'}, alkol: {'var' if alco else 'yok'}, aktivite: {'aktif' if active else 'pasif'})<br>
            <b>Kan Basıncı Kategorisi (sistolik/diastolik):</b> {ap_hi}/{ap_lo} mmHg<br>
            <b>Kolesterol Durumu:</b> { "Sağlıklı (&lt;200)" if total_chol <= 200 else "Sınırda (200–240)" if total_chol <= 240 else "Yüksek (&gt;240)" }<br>
            <b>Açlık Kan Şekeri Durumu:</b> { "Normal (70–100)" if 70 <= fasting_glucose < 100 else "Prediyabet (100–126)" if fasting_glucose < 126 else "Diyabet (≥126)" }
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ----------------------------------------------
    # TAHMİN ÇIKTISI
    # ----------------------------------------------
    if predict_btn:
        # Modelden ham olasılık
        base_prob = float(model.predict_proba(input_df)[0][1])

        # -------- Literatür temelli küçük düzeltmeler --------
        risk_delta = 0.0

        # Kolesterol etkisi
        if chol_cat == 3:
            risk_delta += 0.08
        elif chol_cat == 2:
            risk_delta += 0.04
        else:
            risk_delta -= 0.02

        # Glukoz etkisi
        if gluc_cat == 3:
            risk_delta += 0.08
        elif gluc_cat == 2:
            risk_delta += 0.04
        else:
            risk_delta -= 0.02

        # Sigara / Alkol
        if smoke == 1:
            risk_delta += 0.10
        else:
            risk_delta -= 0.05

        if alco == 1:
            risk_delta += 0.05
        else:
            risk_delta -= 0.02

        # Fiziksel aktivite koruyucu
        if active == 1:
            risk_delta -= 0.06
        else:
            risk_delta += 0.04

        # Aşırı BMI
        if bmi >= 30:
            risk_delta += 0.06
        elif bmi < 18.5:
            risk_delta += 0.03

        # Düzeltmeleri uygula ve 0–1 arasında tut
        prob = np.clip(base_prob + risk_delta, 0.01, 0.99)
        risk_yuzde = prob * 100

        # Sınıf kararı: 0.5 eşiği
        pred = int(prob >= 0.5)

        if pred == 1:
            st.error(
                f"⚠️ <b>YÜKSEK RİSK:</b> Model, bu bireyin kardiyovasküler hastalık "
                f"geliştirme olasılığını yaklaşık <b>%{risk_yuzde:.1f}</b> olarak tahmin etmektedir.",
                icon="⚠️",
            )
        else:
            st.success(
                f"✅ <b>DÜŞÜK RİSK:</b> Model, bu bireyin kardiyovasküler hastalık "
                f"geliştirme olasılığını yaklaşık <b>%{risk_yuzde:.1f}</b> olarak tahmin etmektedir.",
                icon="✅",
            )

        st.markdown(
            """
            <div class='tech-note'>
            <b>Teknik Açıklama:</b> Olasılık, eğitim veri setinde oluşturulan topluluk
            modelinin (lojistik regresyon + random forest + XGBoost) gözleme benzer bireylerin
            sınıf dağılımına dayalı tahminidir. Bu çıktıya, sigara, alkol, kolesterol,
            glukoz ve fiziksel aktivite için küçük ağırlıklı düzeltmeler eklenmiştir; böylece
            risk faktörlerindeki değişiklikler model çıktısına daha belirgin yansımaktadır.
            Çıktı, klinik kararı desteklemek için tasarlanmış bir karar destek sistemidir;
            tek başına tanı veya tedavi kararında kullanılmamalıdır.
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# SAĞ SÜTUN: BİLGİ KARTLARI
# =========================================================
with right_col:
    st.markdown(
        """
        <div class="info-card">
            <h4>📊 Kullanılan Veri Seti</h4>
            <ul>
                <li><b>Kaynak:</b> Cardio Vascular Disease (Kaggle) veri seti</li>
                <li><b>Gözlem sayısı:</b> ~70.000 birey</li>
                <li><b>Değişkenler:</b> yaş, cinsiyet, boy, kilo, kan basıncı (sistolik/diastolik),
                    kolesterol, glikoz, sigara ve alkol kullanımı, fiziksel aktivite vb.</li>
                <li><b>Hedef değişken:</b> <code>cardio</code> (0 = hastalık yok, 1 = kardiyovasküler hastalık var)</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <h4>🧪 Veri Ön İşleme ve Modellemenin Notları</h4>
            <ul>
                <li>Olası aykırı ve tutarsız değerler (özellikle kan basıncı kombinasyonları)
                    veri keşfi aşamasında incelenmiş ve uygun eşiklerle filtrelenmiştir.</li>
                <li>Kayıp değerler, değişken dağılımına göre akıllı imputasyon yöntemleriyle
                    (ortalama/medyan veya benzer gözlemler) tamamlanmıştır.</li>
                <li>Sürekli değişkenler gerektiğinde ölçeklendirilmiş, kategorik değişkenler
                    uygun dummy kodlama ile modele verilmiştir.</li>
                <li>Model performansı eğitim/test ayrımı ve sınıf dengesine duyarlı metriklerle
                    (accuracy, recall, F1, ROC-AUC) değerlendirilmiştir.</li>
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
                <li><b>Lojistik Regresyon:</b> Doğrusal bir olasılık modeli; her risk faktörünün
                    kardiyovasküler hastalık olasılığı üzerindeki marjinal etkisini katsayılar
                    üzerinden yorumlamaya imkân verir.</li>
                <li><b>Random Forest (Karar Ağaçları Kümesi):</b> Birden çok karar ağacının
                    bootstrap örnekler üzerinde eğitilip oy vererek karar aldığı bir topluluk
                    algoritması; doğrusal olmayan ilişkileri ve değişken etkileşimlerini yakalar.</li>
                <li><b>XGBoost:</b> Gradyan artırmalı karar ağaçları algoritması; hatayı adım adım
                    azaltarak özellikle karmaşık örüntüleri yakalamada güçlü bir yöntemdir.</li>
                <li>Bu üç modelin çıktıları, <b>soft voting</b> ile birleştirilmiş; her modelin
                    tahmin ettiği olasılıkların ortalaması alınarak daha kararlı ve genellenebilir
                    bir topluluk tahmini üretilmiştir.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <h4>📈 Eğitim Performansı (Test Kümesi)</h4>
            <ul>
                <li><b>Doğruluk (Accuracy):</b> ≈ 0.74</li>
                <li><b>Duyarlılık (Recall):</b> ≈ 0.70 (hastalığı olan bireyi yakalama oranı)</li>
                <li><b>F1 Skoru:</b> ≈ 0.72 (dengeli ortalama)</li>
                <li><b>ROC-AUC:</b> ≈ 0.80 (modelin hasta ve sağlıklı bireyleri ayırt etme gücü)</li>
                <li>Bu değerler, modelin sınıflar arasındaki ayrımı istatistiksel olarak anlamlı
                    bir düzeyde öğrendiğini göstermektedir.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
