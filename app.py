import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
# Google Drive'dan indirme için gdown kütüphanesinin kurulu olduğundan emin olun.
# Eğer sunucuda gdown kullanılamıyorsa bu kısmı yerel dosya yüklemeye çevirmeniz gerekir.
try:
    import gdown
except ImportError:
    st.error("Gdown kütüphanesi kurulu değil. Model yükleme mekanizması çalışmayabilir.")
    st.stop()

# =========================================================
# SAYFA AYARLARI
# =========================================================
st.set_page_config(
    page_title="🏆 Kardiyovasküler Risk Tahmin Modeli - Final Versiyonu",
    page_icon="💖",
    layout="wide"
)

# ---------------------------------------------------------
# CSS İLE PROFESYONEL TEMALANDIRMA
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global Styling */
    body { font-family: "Segoe UI", sans-serif; background-color: #f0f2f6; }
    .app-title { text-align: center; font-size: 36px; font-weight: 800; margin-bottom: 8px; color: #0f172a; }
    .app-subtitle { text-align: center; font-size: 16px; color: #555; max-width: 950px; margin: 0 auto 20px auto; line-height: 1.6; }
    
    /* Info Cards */
    .info-card { background-color: #ffffff; border-radius: 12px; padding: 18px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); border: 1px solid #e2e8f0; margin-bottom: 15px; font-size: 13px; }
    .info-card h4 { margin-top: 0; margin-bottom: 8px; font-size: 16px; font-weight: 700; color: #1e293b; }
    
    /* Prediction Button */
    .stButton>button {
        background: linear-gradient(90deg, #10b981, #059669); /* Yeşilin Güven Veren Tonları */
        color: white; border-radius: 999px; border: none; padding: 0.7rem 2rem; font-size: 1.1rem; font-weight: 700;
        box-shadow: 0 4px 8px rgba(16, 185, 129, 0.4);
        width: 100%;
    }
    .stButton>button:hover { background: linear-gradient(90deg, #059669, #04785e); }
    
    /* Result Box */
    .result-box { 
        padding: 25px; border-radius: 12px; margin-top: 25px; font-weight: bold; font-size: 1.2rem;
        border: 2px solid; text-align: center;
    }
    .risk-high { background-color: #fef2f2; border-color: #ef4444; color: #b91c1c; }
    .risk-low { background-color: #f0fdf4; border-color: #10b981; color: #059669; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 1. MODEL VE FEATURE DOSYALARINI GÜVENLİ YÜKLEME
# =========================================================
@st.cache_resource
def load_model():
    """
    Model ve feature listesi dosyalarının güvenli bir şekilde indirilip yüklendiği fonksiyon.
    Hata durumunda kullanıcıya bilgilendirme yapar.
    """
    
    # Lütfen BU ID'leri kendi Google Drive dosya ID'lerinizle DEĞİŞTİRİNİZ!
    MODEL_FILE_ID = "YOUR_MODEL_DRIVE_ID_HERE" 
    FEATURE_FILE_ID = "YOUR_FEATURE_LIST_DRIVE_ID_HERE"
    
    MODEL_PATH = "cardio_ensemble_model.pkl"
    FEATURE_PATH = "cardio_feature_cols.pkl"

    try:
        # Model İndirme
        if not os.path.exists(MODEL_PATH) and MODEL_FILE_ID != "YOUR_MODEL_DRIVE_ID_HERE":
            st.warning("Model dosyası indiriliyor...")
            gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=True)

        # Feature Listesi İndirme
        if not os.path.exists(FEATURE_PATH) and FEATURE_FILE_ID != "YOUR_FEATURE_LIST_DRIVE_ID_HERE":
            st.warning("Feature listesi dosyası indiriliyor...")
            gdown.download(f"https://drive.google.com/uc?id={FEATURE_FILE_ID}", FEATURE_PATH, quiet=True)

        # Yükleme
        if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_PATH):
            model = joblib.load(MODEL_PATH)
            feature_cols = joblib.load(FEATURE_PATH)
            st.info("✅ Model ve özellikler başarıyla yüklendi.")
            return model, feature_cols
        else:
            st.error("❌ Model veya özellik dosyaları bulunamadı. Lütfen dosya adlarını ve Drive ID'lerini kontrol edin.")
            return None, None
    
    except Exception as e:
        st.error(f"❌ Kritik Yükleme Hatası: Model yüklenirken bir sorun oluştu. Detay: {e}")
        return None, None

model, feature_cols = load_model()

# =========================================================
# 2. YARDIMCI KLİNİK FONKSİYONLAR
# =========================================================
def chol_category(total_chol):
    if total_chol <= 200: return 1
    elif total_chol <= 240: return 2
    else: return 3

def gluc_category(fasting_glucose):
    if fasting_glucose < 100: return 1
    elif fasting_glucose < 126: return 2
    else: return 3

def get_bp_category(ap_hi, ap_lo):
    if ap_hi < 120 and ap_lo < 80: return "Normal"
    elif ap_hi < 130 and ap_lo < 80: return "Yüksek Normal"
    elif (ap_hi >= 130 and ap_hi < 140) or (ap_lo >= 80 and ap_lo < 90): return "Hipertansiyon Evre 1"
    else: return "Hipertansiyon Evre 2/Kriz"

# =========================================================
# 3. BAŞLIK VE GENEL AÇIKLAMA
# =========================================================
st.markdown(
    "<div class='app-title'>💖 Kardiyovasküler Hastalık Risk Tahmin Modeli</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='app-subtitle'>
    Bu uygulama, lojistik regresyon, Random Forest ve XGBoost'tan oluşan bir <b>topluluk (ensemble) makine öğrenmesi modeli</b> kullanarak
    bireylerin kardiyovasküler hastalık geliştirme olasılığını tahmin eder.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# =========================================================
# 4. KULLANICI GİRDİLERİ
# =========================================================
if model is not None:
    left_col, right_col = st.columns([1.5, 1.0])

    with left_col:
        st.header("📋 Kişisel ve Klinik Bilgiler")

        c1, c2 = st.columns(2)

        with c1:
            # Cinsiyet: Modelin beklediği kodlama (Veri setine göre 1=Kadın, 2=Erkek varsayılmıştır)
            gender_map = {"Kadın": 1, "Erkek": 2}
            gender_ui = st.selectbox("Cinsiyet", options=["Kadın", "Erkek"])
            gender_model = gender_map[gender_ui]
            
            age_years = st.slider("Yaş (yıl)", 29, 70, 50) 
            height = st.slider("Boy (cm)", 130, 210, 170)
            weight = st.slider("Kilo (kg)", 40, 150, 75)
        
        with c2:
            ap_hi = st.slider("Sistolik Tansiyon (mmHg)", 80, 240, 130)
            ap_lo = st.slider("Diyastolik Tansiyon (mmHg)", 40, 180, 80)
            
            total_chol = st.slider("Total Kolesterol (mg/dL)", 100, 350, 200, step=5)
            fasting_glucose = st.slider("Açlık Kan Şekeri (mg/dL)", 60, 250, 95, step=1)

        st.subheader("🚶 Yaşam Tarzı Faktörleri")
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
        # TÜRETİLMİŞ ÖZELLİKLERİ HESAPLA (Feature Engineering)
        # -----------------------------------------------------
        bmi = weight / ((height / 100) ** 2)
        pulse_pressure = ap_hi - ap_lo
        age_bp_index = age_years * ap_hi
        lifestyle_score = smoke_ui + alco_ui + (1 - active) 
        chol_cat = chol_category(total_chol)
        gluc_cat = gluc_category(fasting_glucose)

        # Tüm olası girdileri içeren sözlük
        all_input_dict = {
            "age_years": age_years, "height": height, "weight": weight, "ap_hi": ap_hi, "ap_lo": ap_lo, 
            "cholesterol": chol_cat, "gluc": gluc_cat, "smoke": smoke_ui, "alco": alco_ui, "active": active, 
            "bmi": bmi, "pulse_pressure": pulse_pressure, "age_bp_index": age_bp_index, 
            "lifestyle_score": lifestyle_score, "gender": gender_model # Cinsiyet eklendi
        }
        
        # SADECE feature_cols listesindeki özellikler modele gönderilir (Hata çözümü)
        input_data = {col: all_input_dict[col] for col in feature_cols if col in all_input_dict}

        # Eğer feature_cols'da olması gereken bir şey eksikse uyarı ver.
        if len(input_data) != len(feature_cols):
            missing_features = set(feature_cols) - set(input_data.keys())
            st.warning(f"⚠️ Modelin beklediği bazı özellikler eksik: {missing_features}. Model tahmin yapamayabilir.")
        
        # DataFrame oluşturma (Modelin beklediği sırayı korur)
        input_df = pd.DataFrame([[input_data[col] for col in feature_cols]], columns=feature_cols)

        # -----------------------------------------------------
        # TAHMİN BUTONU VE ÇIKTI
        # -----------------------------------------------------
        predict_btn = st.button("🚀 Kardiyovasküler Risk Tahminini Hesapla", key="main_button")
        st.markdown("")

        if predict_btn:
            try:
                prob = model.predict_proba(input_df)[0][1]
                risk_yuzde = prob * 100

                risk_category = get_bp_category(ap_hi, ap_lo)

                if risk_yuzde >= 50:
                    st.markdown(
                        f"<div class='result-box risk-high'>🚨 YÜKSEK RİSK: Hastalık Geliştirme Olasılığı **%{risk_yuzde:.1f}**</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div class='result-box risk-low'>👍 DÜŞÜK RİSK: Hastalık Geliştirme Olasılığı **%{risk_yuzde:.1f}**</div>",
                        unsafe_allow_html=True,
                    )

                st.markdown("#### 📊 Hesaplanan Özet Bulgular")
                st.info(
                    f"""
                    * **Vücut Kitle İndeksi (BMI):** **{bmi:.1f}** kg/m² ({ "Obez" if bmi >= 30 else "Fazla Kilolu" if bmi >= 25 else "Normal" })
                    * **Kan Basıncı:** **{ap_hi}/{ap_lo}** mmHg ({risk_category})
                    * **Yaşam Tarzı Skoru:** **{lifestyle_score}** (0 en düşük risk)
                    """
                )
                
                st.markdown(
                    """
                    > ❗ **Önemli Not:** Bu uygulama, yapay zekâ tabanlı bir destek sistemidir. Tek başına tıbbi teşhis koymaz veya tedavi kararı vermez.
                    """,
                )
            except Exception as e:
                st.error(f"Tahmin sırasında bir hata oluştu. Veri boyutlarını kontrol edin: {e}")


    # =========================================================
    # 5. SAĞ SÜTUN: BİLGİ KARTLARI (AÇIKLAMA KISMI)
    # =========================================================
    with right_col:
        st.header("🧠 Teknik ve Klinik Bilgiler")
        
        # ----------------- Klinik Risk Kategorileri ----------------
        st.markdown(
            """
            <div class="info-card">
                <h4>🔬 Klinik Parametreler</h4>
                <small>
                Modelin kullandığı bazı klinik eşikler:
                <ul>
                    <li><b>Kan Basıncı:</b> Hipertansiyon Evre 1 (130-139/80-89 mmHg) ve üstü risk faktörüdür.</li>
                    <li><b>Kolesterol:</b> >240 mg/dL (Çok Yüksek) risk kategorisine (3) eşittir.</li>
                    <li><b>Glikoz:</b> ≥126 mg/dL (Diyabet) risk kategorisine (3) eşittir.</li>
                    <li><b>BMI:</b> ≥30.0 kg/m² obezite, ciddi risk artışına neden olur.</li>
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
                <h4>🚀 Ensemble Model Yapısı</h4>
                <ul>
                    <li><b>Topluluk (Ensemble):</b> Lojistik Regresyon, Random Forest ve XGBoost modellerinin kararlarını birleştirerek tek bir modelden daha **sağlam ve genellenebilir** bir tahmin sunar.</li>
                    <li><b>Özellik Mühendisliği (Feature Engineering):</b> **BMI** ve **Nabız Basıncı** gibi türetilmiş değerler, ham veriden daha yüksek tahminsel güce sahiptir.</li>
                    <li><b>Güvenilirlik:</b> Farklı algoritmaların zayıf yönlerini dengeleyerek, tek bir modelin aşırı öğrenme (overfitting) riskini azaltır.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------- Eğitim Performansı ------------------
        st.markdown(
            """
            <div class="info-card">
                <h4>📈 Eğitim Performansı Özeti (Test Seti)</h4>
                <ul>
                    <li><b>ROC-AUC:</b> ≈ 0.80. Modelin, hastalık olanları olmayanlardan ayırt etme yeteneği yüksektir.</li>
                    <li><b>Duyarlılık (Recall):</b> ≈ 0.70. Hastalığı olan kişilerin büyük bir kısmını doğru tahmin edebilmektedir, bu da erken uyarı sistemleri için kritik öneme sahiptir.</li>
                    <li><b>Doğruluk (Accuracy):</b> ≈ 0.74.</li>
                </ul>
                <small>Bu metrikler, modelin klinik veriler üzerinde istatistiksel olarak anlamlı bir performans sergilediğini göstermektedir.</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
