import streamlit as st

# Konfigurasi halaman agar tampilannya lebar (wide mode)
st.set_page_config(
    page_title="ReturnGuard - AI Return-Risk Checker",
    page_icon="🔄",
    layout="wide"
)

# Custom CSS untuk Sidebar pill style, animasi, dan komponen
st.markdown("""
    <style>
    /* Latar belakang utama aplikasi */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Animasi transisi fade-in saat halaman dimuat/berubah */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in-container {
        animation: fadeIn 0.4s ease-in-out;
    }

    /* Styling Sidebar Gelap */
    [data-testid="stSidebar"] {
        background-color: #0d1b2a !important;
        padding-top: 1rem;
    }
    
    /* Sembunyikan radio button bawaan streamlit dan ubah jadi card pill */
    [data-testid="stSidebar"] .stRadio > label {
        display: none;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
        gap: 6px;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        background-color: transparent;
        border-radius: 10px;
        padding: 10px 14px;
        color: #94a3b8;
        font-weight: 500;
        width: 100%;
        transition: all 0.2s ease-in-out;
        border: 1px solid transparent;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
        background-color: #1b2a4a;
        color: #ffffff;
    }
    /* Styling khusus untuk menu aktif (pill background seperti gambar kanan) */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"] {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #334155 !important;
        font-weight: 600;
    }

    /* Tombol Utama */
    .stButton>button {
        background-color: #6366f1;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        width: 100%;
        padding: 0.6rem;
    }
    .stButton>button:hover {
        background-color: #4f46e5;
        color: white;
    }
    
    /* Kotak Metrik & Risiko */
    .metric-box-normal {
        background-color: #e6f4ea;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        color: #137333;
    }
    .metric-box-danger {
        background-color: #fde8e8;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        color: #9b1c1c;
        border: 1px solid #f8b4b4;
    }
    .high-risk-box {
        background-color: #fde8e8;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #dc2626;
        color: #9b1c1c;
        margin-top: 15px;
    }
    .low-risk-box {
        background-color: #def7ec;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #03543f;
        color: #03543f;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    # Bagian Header Logo & Subtitle
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 2px;">
            <div style="background-color: #6366f1; padding: 6px 10px; border-radius: 10px; color: white; font-size: 16px;">🔄</div>
            <span style="color: white; font-size: 20px; font-weight: 700;">ReturnGuard</span>
        </div>
        <p style="color: #94a3b8; font-size: 12px; margin-left: 2px; margin-bottom: 25px;">AI return-risk checker</p>
    """, unsafe_allow_html=True)
    
    # Navigasi Menu (Gaya Pill seperti gambar kanan)
    selected_menu = st.radio(
        "Navigation",
        ["📈 Dashboard", "✨ Prediction", "🧠 Model Insight"],
        index=1  # Default ke Prediction
    )

# Membersihkan string menu untuk logika halaman
clean_menu = selected_menu.split(" ")[1]

# ----------------- KONTEN UTAMA DENGAN ANIMASI FADE-IN -----------------
st.markdown('<div class="fade-in-container">', unsafe_allow_html=True)

if clean_menu == "Prediction":
    st.markdown("## Check a return case")
    st.caption("Enter values manually or generate a sample.")

    # Layout Utama: Dibagi 2 Kolom (Kiri: Form Input, Kanan: Validation Box)
    col1, col2 = st.columns([2.1, 1], gap="large")

    with col1:
        st.markdown("### Customer & return features")
        st.caption("All fields are numeric and non-negative.")

        # Baris 1: Account Age & Total Orders (Default = 0)
        c1, c2 = st.columns(2)
        with c1:
            account_age = st.number_input("ACCOUNT AGE", min_value=0, value=0, step=1, help="days")
        with c2:
            total_orders = st.number_input("TOTAL ORDERS", min_value=0, value=0, step=1, help="orders")

        # Baris 2: Total Returns & Refund Amount (Default = 0)
        c3, c4 = st.columns(2)
        with c3:
            total_returns = st.number_input("TOTAL RETURNS", min_value=0, value=0, step=1, help="returns")
        with c4:
            refund_amount = st.number_input("REFUND AMOUNT (USD)", min_value=0.0, value=0.0, step=1.0, help="USD")

        # Baris 3: Days Until Return & Derived Return Rate (Default = 0)
        c5, c6 = st.columns(2)
        with c5:
            days_until_return = st.number_input("DAYS UNTIL RETURN", min_value=0, value=0, step=1, help="days")
        with c6:
            st.markdown("<p style='margin-bottom: 5px; font-size:12px; font-weight:bold; color:#555;'>Derived return rate</p>", unsafe_allow_html=True)
            
            # Menghitung otomatis persentase return rate
            if total_orders > 0:
                return_rate = (total_returns / total_orders) * 100
            else:
                return_rate = 0.0
            
            # Cek status anomali secara real-time untuk kotak persentase
            is_high_return_rate = return_rate >= 60.0
            is_fast_return = days_until_return <= 2
            
            if is_high_return_rate and is_fast_return:
                st.markdown(f'<div class="metric-box-danger">{return_rate:.1f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-box-normal">{return_rate:.1f}%</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tombol Prediksi dengan Logika Anomali
        if st.button("🪄 Predict return behavior"):
            if total_orders == 0:
                st.warning("Total Orders tidak boleh 0 agar perhitungan rasio valid.")
            elif total_returns > total_orders:
                st.error("Validasi gagal: Total Returns tidak boleh lebih besar dari Total Orders!")
            else:
                if is_high_return_rate and is_fast_return:
                    st.markdown(f"""
                        <div class="high-risk-box">
                            ⚠️ <b>ANOMALY DETECTED / HIGH RISK!</b><br>
                            - Return rate mencapai <b>{return_rate:.1f}%</b> (di atas ambang batas 60%).<br>
                            - Kecepatan retur sangat cepat yaitu <b>{days_until_return} hari</b> (≤ 2 hari).<br>
                            - Total refund sebesar <b>USD {refund_amount}</b> mengindikasikan pola pengembalian masif yang mencurigakan.
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="low-risk-box">
                            ✅ <b>LOW RISK</b><br>
                            Aktivitas akun berada di bawah batas anomali (Return rate: <b>{return_rate:.1f}%</b>, Days until return: <b>{days_until_return} hari</b>).
                        </div>
                    """, unsafe_allow_html=True)

    with col2:
        # Kotak Hitam Samping (Input Health & Validation)
        st.markdown("""
            <div style="background-color: #111827; padding: 25px; border-radius: 12px; color: white;">
                <div style="background-color: #1e293b; color: #38bdf8; padding: 4px 10px; border-radius: 20px; display: inline-block; font-size: 11px; font-weight: bold; margin-bottom: 15px;">
                    ✅ INPUT HEALTH
                </div>
                <h3 style="color: white; font-size: 18px; margin-top: 0; margin-bottom: 20px;">Validation before prediction</h3>
                
                <p style="font-size: 14px; margin-bottom: 10px;">✅ Orders > 0</p>
                <p style="font-size: 14px; margin-bottom: 10px;">✅ Returns ≤ orders</p>
                <p style="font-size: 14px; margin-bottom: 10px;">✅ Refund ≥ 0</p>
                <p style="font-size: 14px; margin-bottom: 20px;">✅ No missing values</p>
                
                <hr style="border-color: #374151; margin-bottom: 15px;">
                
                <p style="font-size: 12px; color: #9ca3af; line-height: 1.4;">
                    💡 Kotak persentase akan otomatis berubah menjadi merah jika Return rate ≥ 60% & Days until return ≤ 2 hari.
                </p>
            </div>
        """, unsafe_allow_html=True)

elif clean_menu == "Dashboard":
    st.title("📊 Dashboard")
    st.write("Halaman ringkasan data transaksi dan statistik keseluruhan.")

elif clean_menu == "Model Insight":
    st.title("🧠 Model Insight")
    st.write("Informasi mengenai performa model machine learning yang digunakan.")

st.markdown('</div>', unsafe_allow_html=True)
