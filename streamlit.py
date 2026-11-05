import streamlit as st

# Konfigurasi halaman
st.set_page_config(
    page_title="ReturnGuard - AI Return-Risk Checker",
    page_icon="🔄",
    layout="wide"
)


def h(s: str) -> str:
    """Rapatkan HTML jadi satu baris.
    Penyebab HTML tampil sebagai kode di Streamlit: baris yang diawali 4+ spasi
    (atau baris kosong) dianggap code block oleh Markdown."""
    return " ".join(line.strip() for line in s.strip().splitlines())


# ----------------- CSS -----------------
st.markdown("""
<style>
.main { background-color: #f8f9fa; }

[data-testid="stSidebar"] { background-color: #0d1b2a !important; padding-top: 1rem; }
[data-testid="stSidebar"] .stRadio > label { display: none; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap: 6px; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background-color: transparent; border-radius: 10px; padding: 10px 14px;
    color: #94a3b8; font-weight: 500; width: 100%;
    transition: all 0.2s ease-in-out; border: 1px solid transparent;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover { background-color: #1b2a4a; color: #ffffff; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"] {
    background-color: #1e293b !important; color: #ffffff !important;
    border: 1px solid #334155 !important; font-weight: 600;
}

.stButton>button {
    background-color: #6366f1; color: white; font-weight: bold;
    border-radius: 8px; width: 100%; padding: 0.6rem;
}
.stButton>button:hover { background-color: #4f46e5; color: white; }

/* ---------- Cards ---------- */
.dashboard-card {
    background-color: #ffffff; padding: 22px; border-radius: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06); border: 1px solid #f1f5f9;
    color: #0f172a; height: 100%; box-sizing: border-box;
}
.card-title { font-size: 16px; font-weight: 700; color: #0f172a; }
.card-sub { font-size: 12px; color: #64748b; }
.card-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; }

.stat-label { font-size: 12px; font-weight: 700; color: #64748b; letter-spacing: .5px; }
.stat-value { font-size: 34px; font-weight: 800; color: #0f172a; margin-top: 14px; line-height: 1; }
.stat-icon { padding: 8px; border-radius: 10px; font-size: 16px; }

/* ---------- Prediction mix ---------- */
.mix-body { display: flex; align-items: center; gap: 22px; margin: 6px 0 20px 0; }
.donut { width: 118px; height: 118px; border-radius: 50%; position: relative; flex-shrink: 0; }
.donut-hole {
    position: absolute; inset: 16px; background: #ffffff; border-radius: 50%;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.donut-hole b { font-size: 22px; color: #0f172a; line-height: 1; }
.donut-hole span { font-size: 10px; color: #94a3b8; margin-top: 3px; }
.mix-legend { flex: 1; }
.mix-row { margin-bottom: 14px; }
.mix-row:last-child { margin-bottom: 0; }
.mix-row-top { display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; margin-bottom: 6px; }
.mix-row-top small { font-weight: 500; color: #94a3b8; margin-left: 4px; }
.bar-bg { width: 100%; border-radius: 8px; height: 8px; }
.bar-fill { height: 8px; border-radius: 8px; }
.mix-foot { font-size: 11px; color: #94a3b8; text-align: center; padding-top: 14px; border-top: 1px dashed #e2e8f0; }

/* ---------- History table ---------- */
.hist-scroll { overflow-x: auto; }
.hist-grid {
    display: grid; align-items: center; column-gap: 12px; min-width: 760px;
    grid-template-columns: 1.1fr 1fr 0.8fr 0.8fr 1fr 1fr 1fr 1.3fr;
}
.hist-head {
    padding: 10px 14px; background: #f8fafc; border-radius: 10px;
    font-size: 11px; font-weight: 700; color: #64748b; letter-spacing: .5px; text-transform: uppercase;
}
.hist-row { padding: 14px; border-bottom: 1px solid #f1f5f9; font-size: 13px; color: #334155; }
.hist-row:last-child { border-bottom: none; }
.hist-row .cell-case { font-weight: 700; color: #0f172a; }
.hist-empty { text-align: center; color: #94a3b8; font-size: 13px; padding: 28px 0 10px 0; }

.badge-table-fraud {
    background-color: #fee2e2; color: #dc2626; padding: 4px 10px; border-radius: 6px;
    font-size: 12px; font-weight: 600; display: inline-block; white-space: nowrap;
}
.badge-table-safe {
    background-color: #dcfce7; color: #166534; padding: 4px 10px; border-radius: 6px;
    font-size: 12px; font-weight: 600; display: inline-block; white-space: nowrap;
}

/* ---------- Prediction page ---------- */
.metric-box-normal {
    background-color: #e6f4ea; padding: 10px; border-radius: 8px;
    text-align: center; font-weight: bold; color: #137333;
}
.metric-box-danger {
    background-color: #fde8e8; padding: 10px; border-radius: 8px; text-align: center;
    font-weight: bold; color: #9b1c1c; border: 1px solid #f8b4b4;
}
.high-risk-box {
    background-color: #fde8e8; padding: 15px; border-radius: 10px;
    border-left: 5px solid #dc2626; color: #9b1c1c; margin-top: 15px;
}
.low-risk-box {
    background-color: #def7ec; padding: 15px; border-radius: 10px;
    border-left: 5px solid #03543f; color: #03543f; margin-top: 15px;
}

/* ---------- Validation panel ---------- */
.val-card {
    background: linear-gradient(160deg, #111827 0%, #0f172a 100%);
    padding: 24px; border-radius: 16px; color: white; border: 1px solid #1f2937;
}
.val-pill {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 11px; font-weight: 700; margin-bottom: 14px; letter-spacing: .5px;
}
.pill-ok { background: #064e3b; color: #6ee7b7; }
.pill-bad { background: #7f1d1d; color: #fca5a5; }
.val-title { color: white; font-size: 18px; font-weight: 700; margin: 0 0 18px 0; }
.val-item {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 12px; border-radius: 10px; margin-bottom: 8px; font-size: 14px;
}
.val-ok { background: rgba(16,185,129,.10); color: #d1fae5; }
.val-bad { background: rgba(239,68,68,.12); color: #fecaca; }
.val-note {
    font-size: 12px; color: #9ca3af; line-height: 1.5;
    margin-top: 16px; padding-top: 14px; border-top: 1px solid #374151;
}
</style>
""", unsafe_allow_html=True)

# ----------------- Session State -----------------
if "history_predictions" not in st.session_state:
    st.session_state.history_predictions = []

history = st.session_state.history_predictions

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown(h("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 2px;">
            <div style="background-color: #6366f1; padding: 6px 10px; border-radius: 10px; color: white; font-size: 16px;">🔄</div>
            <span style="color: white; font-size: 20px; font-weight: 700;">ReturnGuard</span>
        </div>
        <p style="color: #94a3b8; font-size: 12px; margin-left: 2px; margin-bottom: 25px;">AI return-risk checker</p>
    """), unsafe_allow_html=True)

    selected_menu = st.radio(
        "Navigation",
        ["📈 Dashboard", "✨ Prediction", "🧠 Model Insight"],
        index=0
    )

clean_menu = selected_menu.split(" ", 1)[1]

# ----------------- Statistik -----------------
total_analyses = len(history)
fraud_count = sum(1 for item in history if item["result"] == "Fraudulent")
non_fraud_count = total_analyses - fraud_count
fraud_rate = (fraud_count / total_analyses * 100) if total_analyses > 0 else 0.0
non_fraud_rate = (non_fraud_count / total_analyses * 100) if total_analyses > 0 else 0.0

# ======================= DASHBOARD =======================
if clean_menu == "Dashboard":
    st.markdown("<h2 style='margin-bottom: 0;'>Dashboard</h2>", unsafe_allow_html=True)
    st.caption("Overview of prediction activity")
    st.markdown("<br>", unsafe_allow_html=True)

    # ---- 3 kotak statistik ----
    stats = [
        ("📊", "#eff6ff", "#3b82f6", "TOTAL ANALYSES", total_analyses),
        ("⚠️", "#fee2e2", "#dc2626", "FRAUDULENT", fraud_count),
        ("✅", "#dcfce7", "#166534", "NON-FRAUDULENT", non_fraud_count),
    ]
    for col, (icon, bg, fg, label, value) in zip(st.columns(3, gap="large"), stats):
        with col:
            st.markdown(h(f"""
                <div class="dashboard-card">
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span class="stat-icon" style="background-color:{bg}; color:{fg};">{icon}</span>
                        <span class="stat-label">{label}</span>
                    </div>
                    <div class="stat-value">{value}</div>
                </div>
            """), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    g_col1, g_col2 = st.columns([2.1, 1], gap="large")

    # ---- Trend (SVG, satu card utuh) ----
    with g_col1:
        W, H, PAD = 600, 230, 34
        plot_w, plot_h = W - PAD - 14, H - 2 * 24

        def y_of(rate):
            return 24 + plot_h - (min(rate, 100) / 100) * plot_h

        grid = ""
        for v in (0, 25, 50, 75, 100):
            y = y_of(v)
            grid += (f'<line x1="{PAD}" y1="{y}" x2="{W-14}" y2="{y}" stroke="#e2e8f0" stroke-width="1"/>'
                     f'<text x="{PAD-8}" y="{y+4}" font-size="10" fill="#94a3b8" text-anchor="end">{v}%</text>')

        thr_y = y_of(60)
        grid += (f'<line x1="{PAD}" y1="{thr_y}" x2="{W-14}" y2="{thr_y}" stroke="#f59e0b" stroke-width="1.2" stroke-dasharray="5 4"/>'
                 f'<text x="{W-16}" y="{thr_y-5}" font-size="10" fill="#d97706" text-anchor="end">Threshold 60%</text>')

        if total_analyses == 0:
            body = (f'<text x="{W/2}" y="{H/2}" font-size="13" fill="#94a3b8" text-anchor="middle">'
                    f'Belum ada data. Lakukan prediksi untuk melihat tren.</text>')
        else:
            n = total_analyses
            pts = []
            for i, item in enumerate(history):
                x = PAD + plot_w / 2 if n == 1 else PAD + i * plot_w / (n - 1)
                pts.append((x, y_of(float(item["return_rate"])), item))
            body = ""
            if n > 1:
                path = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts)
                area = f"{pts[0][0]:.1f},{y_of(0)} {path} {pts[-1][0]:.1f},{y_of(0)}"
                body += f'<polygon points="{area}" fill="#6366f1" opacity="0.08"/>'
                body += f'<polyline points="{path}" fill="none" stroke="#6366f1" stroke-width="2.5" stroke-linejoin="round"/>'
            for x, y, item in pts:
                c = "#dc2626" if item["result"] == "Fraudulent" else "#16a34a"
                body += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{c}" stroke="#fff" stroke-width="2"/>'

        st.markdown(h(f"""
            <div class="dashboard-card">
                <div class="card-head">
                    <div>
                        <div class="card-title">Prediction trend</div>
                        <div class="card-sub">Return rate per analisis (🔴 fraud · 🟢 low risk)</div>
                    </div>
                    <span class="card-sub">Real-time update</span>
                </div>
                <svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg">{grid}{body}</svg>
            </div>
        """), unsafe_allow_html=True)

    # ---- Prediction Mix Ratio ----
    with g_col2:
        if total_analyses == 0:
            donut_bg = "#e2e8f0"
        else:
            donut_bg = f"conic-gradient(#dc2626 0 {fraud_rate:.2f}%, #16a34a {fraud_rate:.2f}% 100%)"

        st.markdown(h(f"""
            <div class="dashboard-card">
                <div class="card-title">Prediction Mix Ratio</div>
                <div class="card-sub" style="margin-bottom: 18px;">Perbandingan fraud vs non-fraud</div>
                <div class="mix-body">
                    <div class="donut" style="background: {donut_bg};">
                        <div class="donut-hole"><b>{total_analyses}</b><span>cases</span></div>
                    </div>
                    <div class="mix-legend">
                        <div class="mix-row">
                            <div class="mix-row-top">
                                <span style="color:#dc2626;">🔴 Fraudulent<small>({fraud_count})</small></span>
                                <span>{fraud_rate:.1f}%</span>
                            </div>
                            <div class="bar-bg" style="background:#fee2e2;"><div class="bar-fill" style="width:{fraud_rate:.1f}%; background:#dc2626;"></div></div>
                        </div>
                        <div class="mix-row">
                            <div class="mix-row-top">
                                <span style="color:#166534;">🟢 Non-Fraud<small>({non_fraud_count})</small></span>
                                <span>{non_fraud_rate:.1f}%</span>
                            </div>
                            <div class="bar-bg" style="background:#dcfce7;"><div class="bar-fill" style="width:{non_fraud_rate:.1f}%; background:#16a34a;"></div></div>
                        </div>
                    </div>
                </div>
                <div class="mix-foot">Terakumulasi dari {total_analyses} total kasus uji coba</div>
            </div>
        """), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Recent predictions history (semua dalam SATU blok HTML) ----
    if total_analyses == 0:
        rows_html = ('<div class="hist-empty">Belum ada data riwayat prediksi. '
                     'Silakan lakukan prediksi di menu Prediction.</div>')
        table_html = ""
    else:
        rows = ""
        for item in reversed(history):
            if item["result"] == "Fraudulent":
                badge = '<span class="badge-table-fraud">⚠️ Fraudulent</span>'
            else:
                badge = '<span class="badge-table-safe">✅ Low Risk</span>'
            rows += (
                '<div class="hist-grid hist-row">'
                f'<div class="cell-case">{item["case"]}</div>'
                f'<div>{item["account_age"]} days</div>'
                f'<div>{item["total_orders"]}</div>'
                f'<div>{item["total_returns"]}</div>'
                f'<div>{float(item["return_rate"]):.1f}%</div>'
                f'<div>${item["refund_amount"]:,.0f}</div>'
                f'<div>{item["days_until_return"]} days</div>'
                f'<div>{badge}</div>'
                '</div>'
            )
        header = (
            '<div class="hist-grid hist-head">'
            '<div>Case</div><div>Account Age</div><div>Orders</div><div>Returns</div>'
            '<div>Return Rate</div><div>Refund</div><div>Days to Return</div><div>Result</div>'
            '</div>'
        )
        table_html = f'<div class="hist-scroll">{header}{rows}</div>'
        rows_html = ""

    st.markdown(h(f"""
        <div class="dashboard-card">
            <div class="card-head" style="margin-bottom: 14px;">
                <div class="card-title">Recent predictions history</div>
                <span class="card-sub">{total_analyses} record(s)</span>
            </div>
            {table_html}{rows_html}
        </div>
    """), unsafe_allow_html=True)

# ======================= PREDICTION =======================
elif clean_menu == "Prediction":
    st.markdown("## Check a return case")
    st.caption("Enter values manually or generate a sample.")

    col1, col2 = st.columns([2.1, 1], gap="large")

    with col1:
        st.markdown("### Customer & return features")
        st.caption("All fields are numeric and non-negative.")

        c1, c2 = st.columns(2)
        with c1:
            account_age = st.number_input("ACCOUNT AGE", min_value=0, value=0, step=1, help="days")
        with c2:
            total_orders = st.number_input("TOTAL ORDERS", min_value=0, value=0, step=1, help="orders")

        c3, c4 = st.columns(2)
        with c3:
            total_returns = st.number_input("TOTAL RETURNS", min_value=0, value=0, step=1, help="returns")
        with c4:
            refund_amount = st.number_input("REFUND AMOUNT (USD)", min_value=0.0, value=0.0, step=1.0, help="USD")

        c5, c6 = st.columns(2)
        with c5:
            days_until_return = st.number_input("DAYS UNTIL RETURN", min_value=0, value=0, step=1, help="days")
        with c6:
            st.markdown("<p style='margin-bottom: 5px; font-size:12px; font-weight:bold; color:#555;'>Derived return rate</p>",
                        unsafe_allow_html=True)

            return_rate = (total_returns / total_orders) * 100 if total_orders > 0 else 0.0
            is_high_return_rate = return_rate >= 60.0
            is_fast_return = days_until_return <= 2

            box = "metric-box-danger" if (is_high_return_rate and is_fast_return) else "metric-box-normal"
            st.markdown(f'<div class="{box}">{return_rate:.1f}%</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🪄 Predict return behavior"):
            if total_orders == 0:
                st.warning("Total Orders tidak boleh 0 agar perhitungan rasio valid.")
            elif total_returns > total_orders:
                st.error("Validasi gagal: Total Returns tidak boleh lebih besar dari Total Orders!")
            else:
                if is_high_return_rate and is_fast_return:
                    result_status = "Fraudulent"
                    st.markdown(h(f"""
                        <div class="high-risk-box">
                            ⚠️ <b>ANOMALY DETECTED / HIGH RISK!</b><br>
                            - Return rate mencapai <b>{return_rate:.1f}%</b> (di atas ambang batas 60%).<br>
                            - Kecepatan retur sangat cepat yaitu <b>{days_until_return} hari</b> (≤ 2 hari).<br>
                            - Total refund sebesar <b>USD {refund_amount:,.2f}</b> mengindikasikan pola pengembalian masif yang mencurigakan.
                        </div>
                    """), unsafe_allow_html=True)
                else:
                    result_status = "Low Risk"
                    st.markdown(h(f"""
                        <div class="low-risk-box">
                            ✅ <b>LOW RISK</b><br>
                            Aktivitas akun berada di bawah batas anomali (Return rate: <b>{return_rate:.1f}%</b>, Days until return: <b>{days_until_return} hari</b>).
                        </div>
                    """), unsafe_allow_html=True)

                # Simpan SEMUA field dari form Prediction
                st.session_state.history_predictions.append({
                    "case": f"#RF-{len(history) + 2049}",
                    "account_age": account_age,
                    "total_orders": total_orders,
                    "total_returns": total_returns,
                    "refund_amount": float(refund_amount),
                    "days_until_return": days_until_return,
                    "return_rate": round(return_rate, 1),
                    "result": result_status,
                })
                st.success("✨ Hasil analisis berhasil disimpan ke riwayat Dashboard!")

    with col2:
        # Validasi DINAMIS sesuai input saat ini
        checks = [
            ("Orders > 0", total_orders > 0),
            ("Returns ≤ orders", total_returns <= total_orders),
            ("Refund ≥ 0", refund_amount >= 0),
            ("No missing values", True),
        ]
        all_ok = all(ok for _, ok in checks)
        pill = ('<span class="val-pill pill-ok">✅ INPUT HEALTHY</span>' if all_ok
                else '<span class="val-pill pill-bad">⚠️ NEEDS ATTENTION</span>')

        items = ""
        for label, ok in checks:
            cls = "val-ok" if ok else "val-bad"
            icon = "✅" if ok else "❌"
            items += f'<div class="val-item {cls}"><span>{label}</span><span>{icon}</span></div>'

        st.markdown(h(f"""
            <div class="val-card">
                {pill}
                <div class="val-title">Validation before prediction</div>
                {items}
                <div class="val-note">
                    💡 Setiap kali kamu menekan tombol Predict, hasilnya akan otomatis terekam
                    dan mempengaruhi grafik tren serta persentase di Dashboard.
                </div>
            </div>
        """), unsafe_allow_html=True)

# ======================= MODEL INSIGHT =======================
elif clean_menu == "Model Insight":
    st.title("🧠 Model Insight")
    st.write("Informasi mengenai performa model machine learning yang digunakan.")
