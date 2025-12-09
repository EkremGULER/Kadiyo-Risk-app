import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import gdown

# ============================================================
# Sayfa ayarları ve basit tema dokunuşları
# ============================================================
st.set_page_config(
    page_title="Kardiyovasküler Hastalık Risk Tahmin Modeli",
    page_icon="❤️",
    layout="wide"
)

# Basit CSS dokunuşu: slider rengi, başlık boşluğu vb.
st.markdown(
    """
    <style>
    /* Genel font boyutları */
    html, body, [class*="css"]  {
        font-family: "Segoe UI", sans-serif;
    }
    /* Slider rengi */
    .stSlider > div[data-baseweb="slider"] > div > div {
        background: linear-gradient(90deg, #ff6b81, #ff9f43);
    }
    .stSlider [data-baseweb="thumb"] {
        box-shadow: 0 0 0 3px rgba(255,107,129,0.25);
    }
    /* Kart başlıkları */
    .card-title {
        font-weight: 600;
        font-size: 15px;
        margin-bottom: 4px;
    }
    .small-muted {
        font-size: 13px;
        color: #666;
    }
    .info-card {
        padding: 14px 16px;
        border-radius: 8px;
        background-color: #f8fafc;
        border: 1px solid #e3e8f0;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# Model ve feature kolonlarını yükleme
# ============================================================

@st.cache_resource
def load_model_and_features():
    """
    Model dosyası yoksa Google Drive'dan indirir,
    sonrasında modeli ve feature kolon listesini yükler.
    """
    model_path = "cardio_ensemble_model.pkl"
    feature_path = "cardio_feature_cols.pkl"

    # Google Drive'dan model indirme (bir kez)
    if not os.path.exists(model_path):
        file_id = "1WdRoUATILi2VUCuyOEFAnrpoVJ7t69y-"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)

    model = joblib.load(model_path)
    feature_cols = joblib.load(feature_path)

    # Her ihtimale karşı listeye dönüştürelim
    if isinstance(feature_cols, np.ndarray):
        feature_cols = feature_cols.tolist()

    return model, feature_cols


model, feature_cols = load_model_and_features()

# ============================================================
# Başlık
# ============================================================

st.markdown(
    """
    <h2 style="text-align:center; margin-bottom:4px;">
        ❤️ Kardiyovasküler Hastalık Risk Tahmin Modeli
    </h2>
    <p style="text-align:center; font-size:13px; color:#555; max-width:900px; margin:auto;">
        Bu web arayüzü, lojistik regresyon, karar ağaçları ve gradient boosting modellerinden
        oluşan <b>ensemble (topluluk) makine öğrenmesi yaklaşımı</b> ile bireylerin kardiyovasküler
        hastalık riskini tahmin etmek amacıyla geliştirilmiştir. Model, 70.000 gözlem içeren
        Cardio Vascular Disease veri seti üzerinde eğitilmiş olup demografik, antropometrik
        ve klinik değişkenleri kullanmaktadır.
    </p>
    <hr style="margin-top:10px; margin-bottom:18px;">
    """,
    unsafe_allow_html=True
)

# ============================================================
# Ana yerleşim: Sol = girişler, Sağ = açıklama kartları
# ============================================================

left_col, right_col = st.columns([2.1, 1.3])

# ------------------------------------------------------------
# SOL KOLON: Kişisel/Klinik Bilgiler + Tahmin + Hesaplanan Özellikler
# ------------------------------------------------------------
with left_col:
    st.subheader("📋 Kişisel ve Klinik Bilgiler")

    c1, c2 = st.columns(2)

    with c1:
        gender = st.selectbox("Cinsiyet", options=["Kadın", "Erkek"])
        gender_bin = 0 if gender == "Kadın" else 1

        age_years = st.slider("Yaş (yıl)", 29, 65, 50)
        height = st.slider("Boy (cm)", 130, 210, 170)
        weight = st.slider("Kilo (kg)", 40, 150, 75)
        ap_hi = st.slider("Sistolik Tansiyon (mmHg)", 80, 240, 130)
        ap_lo = st.slider("Diyastolik Tansiyon (mmHg)", 40, 140, 80)

    with c2:
        total_chol = st.slider("Total Kolesterol (mg/dL)", 120, 320, 190)
        fasting_glu = st.slider("Açlık Kan Şekeri (mg/dL)", 70, 250, 95)

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

    # --------------------------------------------------------
    # Türetilmiş özellikler (BMI, nabız basıncı, indeksler)
    # --------------------------------------------------------
    bmi = weight / ((height / 100) ** 2)

    # Kan basıncı kategorisi (basit, literatüre uygun aralıklar)
    if ap_hi < 120 and ap_lo < 80:
        bp_cat_str = "Optimal"
    elif 120 <= ap_hi <= 129 and ap_lo < 80:
        bp_cat_str = "Yüksek – Normal (sistolik)"
    elif 130 <= ap_hi <= 139 or 80 <= ap_lo <= 89:
        bp_cat_str = "1. derece hipertansiyon"
    elif 140 <= ap_hi <= 159 or 90 <= ap_lo <= 99:
        bp_cat_str = "2. derece hipertansiyon"
    elif ap_hi >= 160 or ap_lo >= 100:
        bp_cat_str = "3. derece hipertansiyon"
    else:
        bp_cat_str = "Sınıflandırılamadı"

    pulse_pressure = ap_hi - ap_lo
    age_bp_index = age_years * ap_hi

    # Kolesterol kategorisi (mg/dL → 1/2/3)
    if total_chol <= 200:
        chol_cat = 1
        chol_str = "Sağlıklı (≤200 mg/dL)"
    elif total_chol <= 240:
        chol_cat = 2
        chol_str = "Sınırda (200–240 mg/dL)"
    else:
        chol_cat = 3
        chol_str = "Yüksek kolesterol (>240 mg/dL)"

    # Açlık glikoz kategorisi (mg/dL → 1/2/3)
    if fasting_glu < 100:
        glu_cat = 1
        glu_str = "Normal (70–99 mg/dL)"
    elif fasting_glu < 126:
        glu_cat = 2
        glu_str = "Prediyabet (100–125 mg/dL)"
    else:
        glu_cat = 3
        glu_str = "Diyabet (≥126 mg/dL)"

    # Yaşam tarzı skoru: 0 = en kötü, 3 = en iyi
    # sigara(1) + alkol(1) + hareketsizlik(1) arttıkça skor azalıyor
    risky_behaviours = smoke + alco + (1 - active)
    lifestyle_score = 3 - risky_behaviours
    lifestyle_score = max(0, min(3, lifestyle_score))

    # --------------------------------------------------------
    # Modelin beklediği girdi vektörünü hazırlama
    # (feature_cols sırasına göre dolduruyoruz)
    # --------------------------------------------------------
    raw_features = {
        "age_years": age_years,
        "gender": gender_bin,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": chol_cat,  # modele kategori gidiyor
        "gluc": glu_cat,          # modele kategori gidiyor
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "bmi": bmi,
        "pulse_pressure": pulse_pressure,
        "age_bp_index": age_bp_index,
        "lifestyle_score": lifestyle_score,
    }

    input_row = [raw_features.get(col, 0) for col in feature_cols]
    input_df = pd.DataFrame([input_row], columns=feature_cols)

    # --------------------------------------------------------
    # Tahmin butonu ve kullanıcıya rehber metin
    # --------------------------------------------------------
    st.markdown(
        """
        <div style='margin-top:6px; padding:10px; background:#eef6ff;
                    border-left:4px solid #5b9bff; border-radius:4px;
                    font-size:13px;'>
        ℹ️ <b>Not:</b> Tüm bilgileri girdikten sonra aşağıdaki butona basarak,
        bireyin kardiyovasküler hastalık risk tahminini hesaplayınız.
        </div>
        """,
        unsafe_allow_html=True
    )

    predict_btn = st.button("🔍 Kardiyovasküler Risk Tahminini Hesapla")

    if not predict_btn:
        st.info("Henüz tahmin yapılmadı. Lütfen bilgileri girip butona tıklayınız.")
    else:
        prob = model.predict_proba(input_df)[0][1]
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
            <div style='margin-top:10px; font-size:13px;'>
            <b>Teknik Açıklama:</b> Tahmin, eğitim veri seti üzerinde oluşturulan topluluk
            modelinin sınıf olasılık dağılımına dayalıdır. Çıktı, klinik karar sürecini
            desteklemek amacıyla tasarlanmış bir <i>karar destek sistemi</i> ürünüdür;
            tek başına tanı koymak veya tedavi planlamak için kullanılmamalıdır.
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # Hesaplanan ek özellikler
    # --------------------------------------------------------
    with st.expander("ℹ Hesaplanan Ek Özellikler", expanded=True):
        st.markdown(
            f"""
            - **Vücut Kitle İndeksi (BMI):** {bmi:.1f} kg/m²  
            - **Nabız Basıncı (ap_hi - ap_lo):** {pulse_pressure:.0f} mmHg  
            - **Yaş × Sistolik Tansiyon İndeksi:** {age_bp_index:.0f}  
            - **Yaşam Tarzı Skoru (0–3, yüksek skor daha sağlıklı):** {lifestyle_score:.0f}  
            - **Kan Basıncı Durumu:** {bp_cat_str}  
            - **Kolesterol Durumu:** {chol_str}  
            - **Açlık Kan Şekeri Durumu:** {glu_str}
            """,
            unsafe_allow_html=True
        )

# ------------------------------------------------------------
# SAĞ KOLON: Veri seti, veri ön işleme, modeller, performans
# ------------------------------------------------------------
with right_col:
    # Kullanılan veri seti
    st.markdown(
        """
        <div class="info-card">
          <div class="card-title">📊 Kullanılan Veri Seti</div>
          <div class="small-muted">
            <b>Kaynak:</b> Cardio Vascular Disease veri seti (≈70.000 gözlem).<br>
            <b>Değişkenler:</b> yaş, cinsiyet, boy, kilo, kan basıncı, kolesterol, açlık glikozu,
            sigara kullanımı, alkol kullanımı, fiziksel aktivite.<br>
            <b>Hedef değişken:</b> <code>cardio</code> (0 = hastalık yok, 1 = kardiyovasküler hastalık var).
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Veri ön işleme
    st.markdown(
        """
        <div class="info-card">
          <div class="card-title">🧪 Veri Ön İşleme</div>
          <div class="small-muted">
            • Aykırı ve tutarsız değerler (özellikle kan basıncı kombinasyonları) klinik
            literatür ışığında incelenmiş ve uç değerler elenmiştir.<br>
            • Eksik veya hatalı kayıtlar için basit imputasyon teknikleri kullanılmıştır.<br>
            • Sürekli değişkenler gerektiğinde ölçeklendirilmiş, kategorik değişkenler
            ikili/ordinal formata dönüştürülmüştür.<br>
            • Özellik mühendisliği kapsamında BMI, nabız basıncı, yaş×tansiyon indeksi ve
            yaşam tarzı skoru gibi türetilmiş değişkenler eklenmiştir.
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Kullanılan modeller
    st.markdown(
        """
        <div class="info-card">
          <div class="card-title">🤖 Kullanılan Modeller</div>
          <div class="small-muted">
            • Lojistik Regresyon<br>
            • Karar Ağaçları / Random Forest<br>
            • Gradient Boosting (XGBoost benzeri yapı)<br><br>
            Bu modeller, <b>ensemble (topluluk)</b> yaklaşımı ile birleştirilmiş; her modelin
            tahmin olasılıkları ağırlıklandırılarak son karar için ortalaması alınmıştır.
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Eğitim performansı
    st.markdown(
        """
        <div class="info-card">
          <div class="card-title">📈 Eğitim Performansı (Test Kümesi)</div>
          <div class="small-muted">
            • Doğruluk (Accuracy): ≈ 0.74<br>
            • Duyarlılık (Recall): ≈ 0.70 (hastalığı olan bireyi yakalama oranı)<br>
            • F1 Skoru: ≈ 0.72 (dengeli hata ölçütü)<br>
            • ROC-AUC: ≈ 0.80 (ayırma gücü)<br><br>
            Bu değerler, modelin sınıflar arasındaki ayrımı istatistiksel olarak anlamlı
            bir düzeyde öğrendiğini göstermektedir.
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
