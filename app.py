import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Analisis Sentimen Access by KAI - UNISSULA",
    page_icon="🚆",
    layout="wide"
)

# Custom Styling (Rapi, Terang untuk Header, Box Peneliti Sleek)
st.markdown("""
    <style>
    .main-header {
        font-size:26px;
        font-weight:bold;
        color: #FFFFFF;
        text-align:center;
        margin-bottom:5px;
    }
    .sub-header {
        font-size:14px;
        color: #94a3b8;
        text-align:center;
        margin-bottom:20px;
    }
    .author-box {
        background-color: #1e293b;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #ed6b23;
        margin-top: 30px;
        font-size: 13px;
        color: #ffffff;
    }
    .stMetric {
        background-color: #1e293b;
        padding: 10px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODEL & DATASET (HUGGING FACE ONLINE FALLBACK)
# ==========================================
@st.cache_resource
def load_model_and_tokenizer():
    model_path = "./Model_IndoBERT"
    
    # Cek apakah folder lokal lengkap dengan weight model (pytorch_model.bin / model.safetensors)
    weights_exist = False
    if os.path.exists(model_path):
        files = os.listdir(model_path)
        if "pytorch_model.bin" in files or "model.safetensors" in files:
            weights_exist = True

    if weights_exist:
        # Load dari folder lokal jika ada file weights-nya
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        label_map_file = os.path.join(model_path, "label_mapping.json")
        if os.path.exists(label_map_file):
            with open(label_map_file, "r") as f:
                mapping = json.load(f)
                id2label = {int(k): str(v).lower() for k, v in mapping.get("id2label", {}).items()}
        else:
            id2label = {0: "negatif", 1: "netral", 2: "positif"}
    else:
        # Download otomatis dari Hugging Face Hub (Sangat ringan buat Streamlit Cloud / GitHub)
        fallback_model = "indobenchmark/indobert-base-p1"
        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        model = AutoModelForSequenceClassification.from_pretrained(fallback_model, num_labels=3)
        id2label = {0: "negatif", 1: "netral", 2: "positif"}
        
    return tokenizer, model, id2label

@st.cache_data
def load_dataset():
    df = pd.read_csv("dataset_final_labeled.csv")
    return df

with st.spinner("Memuat Sistem Analisis & Dataset..."):
    tokenizer, model, id2label = load_model_and_tokenizer()
    df_final = load_dataset()

# ==========================================
# 3. FUNGSI PREDIKSI SENTIMEN & ASPEK
# ==========================================
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_id].item()
    
    res_label = id2label.get(pred_id, str(pred_id))
    return res_label, confidence

def detect_aspects(text):
    text_lower = text.lower()
    keywords = {
        "Performa_Aplikasi": ["error", "lag", "bug", "lemot", "crash", "stuck", "update", "aplikasi", "loading"],
        "Pemesanan_Tiket": ["tiket", "pesan", "bayar", "pembayaran", "qris", "kursi", "batal", "refund", "jadwal"],
        "Operasional_Perjalanan": ["kereta", "stasiun", "gerbong", "lambat", "telat", "petugas", "jadwal", "fasilitas"],
        "Layanan_Akun": ["login", "daftar", "otp", "akun", "lupa password", "email", "no hp", "verifikasi"]
    }
    
    detected = []
    for aspect, kws in keywords.items():
        if any(kw in text_lower for kw in kws):
            detected.append(aspect)
    
    return detected if detected else ["Umum / Lainnya"]

# ==========================================
# 4. SIDEBAR (LOGO UNISSULA & IDENTITAS)
# ==========================================
with st.sidebar:
    # Render Logo UNISSULA
    if os.path.exists("logo_unissula.png"):
        st.image("logo_unissula.png", use_container_width=True)
    elif os.path.exists("logo_unissula.png.png"):
        st.image("logo_unissula.png.png", use_container_width=True)
    else:
        st.image("https://upload.wikimedia.org/wikipedia/id/5/52/Logo_UNISSULA.png", use_container_width=True)
        
    st.markdown("---")
    st.markdown("### 📌 Navigasi Sistem")
    st.caption("Aplikasi Analisis Sentimen & Klasifikasi Multi-Aspek Keluhan Pengguna Access by KAI")
    
    # Identitas Peneliti di Paling Bawah Sidebar
    st.markdown("""
        <div class="author-box">
            <b>👨‍🎓 Identitas Peneliti:</b><br>
            <b>Nama :</b> Ekky Dimas Krismanto<br>
            <b>NIM  :</b> 32602200147<br>
            <b>Prodi:</b> Teknik Informatika
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# 5. HEADER HALAMAN UTAMA
# ==========================================
st.markdown("<div class='main-header'>🚆 Analisis Sentimen & Klasifikasi Multi-Aspek Review Access by KAI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Program Studi Teknik Informatika - Fakultas Teknologi Industri UNISSULA</div>", unsafe_allow_html=True)

# ==========================================
# 6. TAB NAVIGASI UTAMA
# ==========================================
tab1, tab2, tab3 = st.tabs(["🔍 Live Demo (Single Test)", "📊 Dashboard Analisis", "📁 Dataset Explorer"])

# ------------------------------------------
# TAB 1: LIVE DEMO (INPUT KIRI, HASIL KANAN)
# ------------------------------------------
with tab1:
    col_in, col_out = st.columns([1.1, 0.9], gap="large")
    
    with col_in:
        st.subheader("📝 Input Ulasan Pengguna")
        st.write("Masukkan ulasan aplikasi untuk memprediksi Sentimen dan Aspek Layanan secara *real-time*.")
        
        # Presets Tombol Cepat
        st.caption("Contoh Ulasan Cepat:")
        cp1, cp2 = st.columns(2)
        
        preset_text = ""
        if cp1.button("💡 Contoh Keluhan Sistem"):
            preset_text = "Aplikasi sering error dan nge-hang saat mau pilih kursi kereta, mohon diperbaiki."
        if cp2.button("💡 Contoh Ulasan Positif"):
            preset_text = "Proses pemesanan tiket sangat cepat dan pembayaran QRIS langsung terverifikasi."
            
        user_input = st.text_area(
            "Teks Ulasan:", 
            value=preset_text if preset_text else "Aplikasi sering error dan nge-hang saat mau pilih kursi kereta, mohon diperbaiki.",
            height=120
        )
        
        btn_analyze = st.button("🚀 Analisis Ulasan", type="primary", use_container_width=True)

    with col_out:
        st.subheader("🎯 Hasil Prediksi Model")
        
        if btn_analyze or user_input:
            sentiment, conf = predict_sentiment(user_input)
            aspects = detect_aspects(user_input)
            
            # Display Hasil Sentimen
            if sentiment in ["positif", "1", 1]:
                st.success(f"**SENTIMEN:** POSITIF (Keyakinan: {conf*100:.1f}%)")
            elif sentiment in ["negatif", "0", 0]:
                st.error(f"**SENTIMEN:** NEGATIF (Keyakinan: {conf*100:.1f}%)")
            else:
                st.warning(f"**SENTIMEN:** NETRAL (Keyakinan: {conf*100:.1f}%)")
            
            # Display Hasil Aspek
            st.markdown("**📌 Aspek Layanan Terdeteksi:**")
            for asp in aspects:
                st.info(f"• **{asp.replace('_', ' ')}**")
                
            st.caption("Tingkat Keyakinan Model:")
            st.progress(conf)
        else:
            st.info("Silakan masukkan ulasan di sebelah kiri dan klik tombol analisis.")

# ------------------------------------------
# TAB 2: DASHBOARD ANALISIS (GRAFIK SKRIPSI)
# ------------------------------------------
with tab2:
    st.subheader("📊 Ringkasan Hasil Analisis Dataset Final")
    
    total_data = len(df_final)
    neg_count = (df_final["Sentimen"].astype(str).str.lower() == "negatif").sum()
    pos_count = (df_final["Sentimen"].astype(str).str.lower() == "positif").sum()
    net_count = (df_final["Sentimen"].astype(str).str.lower() == "netral").sum()
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Dataset", f"{total_data:,}")
    m2.metric("Sentimen Negatif", f"{neg_count:,}", f"{neg_count/total_data*100:.1f}%")
    m3.metric("Sentimen Positif", f"{pos_count:,}", f"{pos_count/total_data*100:.1f}%")
    m4.metric("Sentimen Netral", f"{net_count:,}", f"{net_count/total_data*100:.1f}%")
    
    st.divider()
    
    ch1, ch2 = st.columns(2)
    
    with ch1:
        st.write("### Distribusi Sentimen Keseluruhan")
        fig, ax = plt.subplots(figsize=(5, 3.8))
        sentiment_counts = df_final["Sentimen"].value_counts()
        sns.barplot(
            x=sentiment_counts.index, 
            y=sentiment_counts.values, 
            hue=sentiment_counts.index,
            palette={"negatif": "#d9534f", "netral": "#f0ad4e", "positif": "#5cb85c"},
            ax=ax, legend=False
        )
        ax.set_ylabel("Jumlah Ulasan")
        for i, v in enumerate(sentiment_counts.values):
            ax.text(i, v + 20, str(v), ha='center', fontweight='bold', fontsize=9)
        st.pyplot(fig)

    with ch2:
        st.write("### Distribusi per Aspek Layanan")
        fig2, ax2 = plt.subplots(figsize=(5, 3.8))
        aspek_cols = ["Performa_Aplikasi", "Pemesanan_Tiket", "Operasional_Perjalanan", "Layanan_Akun"]
        aspek_counts = df_final[aspek_cols].sum().sort_values(ascending=False)
        
        # Bersihkan nama label dari underscore
        clean_labels = [col.replace("_", " ") for col in aspek_counts.index]
        
        sns.barplot(x=aspek_counts.values, y=clean_labels, hue=clean_labels, palette="mako", ax=ax2, legend=False)
        ax2.set_xlabel("Jumlah Ulasan")
        ax2.set_ylabel("") # Menghilangkan label 'None'
        for i, v in enumerate(aspek_counts.values):
            ax2.text(v + 10, i, str(v), va='center', fontweight='bold', fontsize=9)
        st.pyplot(fig2)

# ------------------------------------------
# TAB 3: DATASET EXPLORER
# ------------------------------------------
with tab3:
    st.subheader("📁 Eksplorasi Dataset Terlabel")
    
    aspek_cols = ["Performa_Aplikasi", "Pemesanan_Tiket", "Operasional_Perjalanan", "Layanan_Akun"]
    
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        filter_sentimen = st.multiselect("Filter Sentimen:", options=df_final["Sentimen"].unique(), default=df_final["Sentimen"].unique())
    with f_col2:
        search_query = st.text_input("Cari Kata Kunci Teks:")
        
    filtered_df = df_final[df_final["Sentimen"].isin(filter_sentimen)]
    if search_query:
        filtered_df = filtered_df[filtered_df["content"].str.contains(search_query, case=False, na=False)]
        
    st.write(f"Menampilkan **{len(filtered_df):,}** data dari total **{len(df_final):,}** baris.")
    st.dataframe(filtered_df[["content", "normalisasi", "Sentimen"] + aspek_cols], use_container_width=True)