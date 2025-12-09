%%writefile app.py
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
    page_title="🏆 Kardiyovasküler Risk Tahmin Modeli - Yarışma Versiyonu",
    page_icon="💖",
    layout="wide"
)

# ---------------------------------------------------------
# BASİT CSS İLE TEMAYI GÜÇLENDİRME
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Genel font ve arka plan */
    body { font-family: "Segoe UI", sans-serif; background-color: #f0f2f6; }
    .main { padding-top: 10px; }
    /* Başlık */
    .app-title { text-align: center; font-size: 34px; font-weight: 700; margin-bottom: 8px; color: #0f172a; }
    .app-subtitle { text-align: center; font-size: 16px; color: #555; max-width: 950px; margin: 0 auto 20px auto; line-height: 1.6; }
    /* Bilgi Kartları */
    .info-card { background-color: #ffffff; border-radius: 12px; padding: 18px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); border: 1px solid #e2e8f0; margin-bottom: 15px; font-size: 13px; }
    .info-card h4 { margin-top: 0; margin-bottom: 8px; font-size: 15px; font-weight: 700; color: #1e293b; }
    /* Buton (daha dikkat çekici) */
    .stButton>button {
        background: linear-gradient(90deg, #ef4444, #f97316); /* Kırmızıdan Turuncuya */
        color: white; border-radius: 999px; border: none; padding: 0.5rem 1.6rem; font-size: 1rem; font-weight: 700;
        box-shadow: 0 4px 8px rgba(239, 68, 68, 0.3);
    }
    .stButton>button:hover { background: linear-gradient(90deg, #dc2626, #ea580c); }
    /* Sonuç Kutusu */
    .result-box { 
        padding: 20px; border-radius: 10px; margin-top: 20px; font-weight: bold; font-size: 1.1rem;
        border: 2px solid; 
    }
    .risk-high { background-color: #fee2e2; border-color: #ef4444; color: #dc2626; }
    .risk-low { background-color: #d1fae5; border-color: #10b981; color: #047857; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# MODEL VE FEATURE DOSYALARINI YÜKLEME
# =========================================================
@st.cache_resource
def load_model():
    """
    Eğer dosyalar yoksa Google Drive'dan indirır, ardından model ve feature listesini yükler.
    Model ve Feature listesini tek bir fonksiyonda indirmek/yüklemek daha güvenlidir.
    """
    
    # Google Drive ID'leriniz (Örnek ID'ler - Değiştirilmeli)
    MODEL_FILE_ID = "1WdRoUATILi2VUCuyOEFAnrpoVJ7t69y-" 
    FEATURE_FILE_ID = "1h_VnL-B3i5uT-D9iF-XoE7jP9oA2fGhG" # Farazi ID
    
    MODEL_PATH = "cardio_ensemble_model.pkl"
    FEATURE_PATH = "cardio_feature_cols.pkl"

    try:
        if not os.path.exists(MODEL_PATH):
            st.warning("Model dosyası sunucuda bulunamadı. Google Drive'dan indiriliyor...")
            gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=True)

        if not os.path.exists(FEATURE_PATH):
            st.warning("Feature listesi dosyası sunucuda bulunamadı. Google Drive'dan indiriliyor...")
            gdown.download(f"https://drive.google.com/uc?id={FEATURE_FILE_ID}", FEATURE_PATH, quiet=True)

        # Ana topluluk model
        model = joblib.load(MODEL_PATH)
        # Eğitim sırasında kullanılan feature sırası
        feature_cols = joblib.load(FEATURE_PATH)
        st.success("🎉 Model ve özellikler başarıyla yüklendi!")
        return model, feature_cols
    
    except Exception as e:
        st.error(f"❌ Model veya özellikler yüklenirken kritik bir hata oluştu: {e}")
        st.stop() # Hata durumunda uygulamayı durdur

model, feature_cols = load_model()

# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================
# (Orijinal koddaki chol_category ve gluc_category fonksiyonları buraya taşınır)
def chol_category(total_chol):
    if total_chol <= 200: return 1
    elif total_chol <= 240: return 2
    else: return 3

def gluc_category(fasting_glucose):
    if fasting_glucose < 100: return 1
    elif fasting_glucose < 126: return 2
    else: return 3

def get_bp_category(ap_hi, ap_lo):
    if ap_hi < 120 and ap_lo < 80:
        return "Normal"
    elif ap_hi < 130 and ap_lo < 80:
        return "Yüksek Normal"
    elif (ap_hi >= 130 and ap_hi < 140) or (ap_lo >= 80 and ap_lo < 90):
        return "Hipertansiyon Evre 1"
    else:
        return "Hipertansiyon Evre 2/Kriz"

# =========================================================
# BAŞLIK VE GENEL AÇIKLAMA
# =========================================================
st.markdown(
    "<div class='app-title'>💖 Kardiyovasküler Hastalık Risk Tahmin Modeli</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='app-subtitle'>
    Bu model, lojistik regresyon, Random Forest ve XGBoost'tan oluşan bir <b>topluluk (ensemble) yapay zekâ</b> yapısını kullanır.
    Lütfen tüm verileri doğru girin, tahmin, kardiyovasküler hastalık geliştirme olasılığını gösterir.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# =========================================================
# SAYFA YERLEŞİMİ VE KULLANICI GİRDİLERİ
# =========================================================
left_col, right_col = st.columns([1.3, 1.0])

with left_col:
    st.subheader("📋 Kişisel ve Klinik Bilgiler")

    c1, c2 = st.columns(2)

    with c1:
        # Cinsiyet: Modelin beklediği kodlama (Örnek: 1=Kadın, 2=Erkek, VEYA 0=Kadın, 1=Erkek)
        # Veri setine göre düzeltme yapılmalıdır. Varsayımsal olarak 1 ve 2 kullanıyorum.
        gender_map = {"Kadın": 1, "Erkek": 2}
        gender_ui = st.selectbox("Cinsiyet", options=["Kadın", "Erkek"])
        gender_model = gender_map[gender_ui]
        
        age_years = st.slider("Yaş (yıl)", 29, 70, 50) # Yaş aralığı biraz genişletildi
        height = st.slider("Boy (cm)", 130, 210, 170)
        weight = st.slider("Kilo (kg)", 40, 150, 75)
    
    with c2:
        ap_hi = st.slider("Sistolik Tansiyon (mmHg)", 80, 240, 130)
        ap_lo = st.slider("Diyastolik Tansiyon (mmHg)", 40, 180, 80)
        
        total_chol = st.slider("Total Kolesterol (mg/dL)", 100, 350, 200, step=5)
        fasting_glucose = st.slider("Açlık Kan Şekeri (mg/dL)", 60, 250, 95, step=1)

    # Yaşam Tarzı (Ortada toplandı)
    st.markdown("#### Yaşam Tarzı Faktörleri")
    c3, c4, c5 = st.columns(3)
    with c3:
        smoke_ui = st.selectbox(
            "Sigara Kullanımı", options=[0, 1], format_func=lambda x: "Evet" if x == 1 else "Hayır",
        )
    with c4:
        alco_ui = st.selectbox(
            "Alkol Kullanımı", options=[0, 1], format_func=lambda x: "Evet" if x == 1 else "Hayır",
        )
    with c5:
        active = st.selectbox(
            "Fiziksel Aktivite", options=[0, 1], format_func=lambda x: "Aktif" if x == 1 else "Pasif",
        )

    st.markdown("---")

    # -----------------------------------------------------
    # TÜRETİLMİŞ ÖZELLİKLERİ HESAPLA
    # -----------------------------------------------------
    bmi = weight / ((height / 100) ** 2)
    pulse_pressure = ap_hi - ap_lo
    age_bp_index = age_years * ap_hi
    lifestyle_score = smoke_ui + alco_ui + (1 - active) 
    chol_cat = chol_category(total_chol)
    gluc_cat = gluc_category(fasting_glucose)

    # -----------------------------------------------------
    # MODELE GİDECEK GİRDİLERİ HAZIRLA
    # -----------------------------------------------------
    input_dict = {
        "age_years": age_years,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": chol_cat,
        "gluc": gluc_cat,
        "smoke": smoke_ui,
        "alco": alco_ui,
        "active": active,
        "bmi": bmi,
        "pulse_pressure": pulse_pressure,
        "age_bp_index": age_bp_index,
        "lifestyle_score": lifestyle_score,
        # ÖNEMLİ: Cinsiyet değişkeni eklendi (Modelin beklediği isme dikkat edilmeli)
        "gender": gender_model 
    }
    
    # Modelin beklediği sırayı koruyarak DataFrame oluşturma
    # Not: Eğer feature_cols'da 'gender' yoksa bu kısım hata verir. feature_cols modelde olmalı.
    if 'gender' in feature_cols:
        input_df = pd.DataFrame([[input_dict[col] for col in feature_cols]], columns=feature_cols)
    else:
        st.error("Modelin beklediği özellik listesinde 'gender' değişkeni bulunamadı. Lütfen modelinizi kontrol edin.")
        st.stop()


    # -----------------------------------------------------
    # TAHMİN BUTONU VE ÇIKTI
    # -----------------------------------------------------
    st.markdown("")
    predict_btn = st.button("🚀 Kardiyovasküler Risk Tahminini Hesapla")
    st.markdown("")

    if predict_btn:
        prob = model.predict_proba(input_df)[0][1]
        risk_yuzde = prob * 100

        risk_category = get_bp_category(ap_hi, ap_lo)

        if risk_yuzde > 50:
            st.markdown(
                f"<div class='result-box risk-high'>⚠️ YÜKSEK RİSK: Kardiyovasküler hastalık olasılığı **%{risk_yuzde:.1f}**</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='result-box risk-low'>✅ DÜŞÜK RİSK: Kardiyovasküler hastalık olasılığı **%{risk_yuzde:.1f}**</div>",
                unsafe_allow_html=True,
            )

        st.markdown("#### Temel Risk Bilgileri")
        st.info(
            f"""
            * **Vücut Kitle İndeksi (BMI):** **{bmi:.1f}** kg/m²
            * **Kan Basıncı Durumu:** **{ap_hi}/{ap_lo}** mmHg ({risk_category})
            * **Yaşam Tarzı Skoru:** **{lifestyle_score}** (0 en iyi)
            """
        )
        
        st.markdown(
            """
            > ❗ **Önemli Not:** Bu uygulama tıbbi tanı koymaz. Klinik bir karar vermeden önce daima bir sağlık uzmanına danışın.
            """,
        )

# =========================================================
# SAĞ SÜTUN: BİLGİ KARTLARI (Yarışma için detaylandırıldı)
# =========================================================
with right_col:
    st.subheader("📚 Teknik ve Klinik Bilgiler")
    
    # ----------------- Klinik Risk Kategorileri ----------------
    st.markdown(
        """
        <div class="info-card">
            <h4>❤️ Klinik Risk Sınıflandırmaları</h4>
            <small>
            <b>Kan Basıncı (Örnek Eşikler):</b>
            <ul>
                <li>Normal: <120/<80 mmHg</li>
                <li>Hipertansiyon Evre 1: 130–139/80–89 mmHg</li>
            </ul>
            <b>Total Kolesterol:</b>
            <ul>
                <li>Normal: ≤200 mg/dL (Modelde: 1)</li>
                <li>Yüksek: >240 mg/dL (Modelde: 3)</li>
            </ul>
            <b>Vücut Kitle İndeksi (BMI):</b>
            <ul>
                <li>Sağlıklı: 18.5 – 24.9 kg/m²</li>
                <li>Obezite: ≥30.0 kg/m²</li>
            </ul>
            </small>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # ----------------- Modelin Avantajı ----------------------
    st.markdown(
        """
        <div class="info-card">
            <h4>🧠 Ensemble Modelin Avantajı</h4>
            <ul>
                <li><b>Sağlamlık:</b> Lojistik Regresyon'un yorumlanabilirliği, Random Forest'ın genelleme gücü ve XGBoost'un yüksek performansını birleştirerek daha kararlı tahminler üretir.</li>
                <li><b>Aykırı Değer Toleransı:</b> Ağaç tabanlı modeller, aykırı değerlerin etkisini azaltarak modelin klinik veri üzerindeki güvenilirliğini artırır.</li>
                <li><b>Feature Engineering:</b> BMI, Nabız Basıncı ve Yaşam Tarzı Skoru gibi türetilmiş özellikler, ham veriden çıkarılamayacak yeni klinik ilişkileri yakalar.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----------------- Eğitim Performansı (Görsel Yardım) ------------------
    st.markdown(
        """
        <div class="info-card">
            <h4>📈 Model Performansı (Test Seti)</h4>
            <ul>
                <li><b>ROC-AUC:</b> ≈ 0.80. Bu, modelin hastalık olanları olmayanlardan ayırt etme yeteneğinin güçlü olduğunu gösterir.</li>
                <li><b>Duyarlılık (Recall, Sınıf 1):</b> ≈ 0.70. Hastalığı olan 10 kişiden 7'sini doğru tahmin ettiğimiz anlamına gelir, bu da önleyici tıp için önemli bir metriktir.</li>
            </ul>
            
        </div>
        """,
        unsafe_allow_html=True,
    )
