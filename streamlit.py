from pathlib import Path

import joblib
import pandas as pd
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


# ----------------- LOAD MODEL (sekali saja, di-cache) -----------------
MODEL_PATH = Path(__file__).parent / "return_abuse_detector_model.joblib"

# Warna & penjelasan tiap kelas hasil model
CLASS_INFO = {
    "Legitimate":        {"color": "#16a34a", "bg": "#dcfce7", "desc": "Pola retur wajar dan tidak menunjukkan indikasi penyalahgunaan."},
    "Fraudulent Return": {"color": "#dc2626", "bg": "#fee2e2", "desc": "Pola mengarah ke pengembalian curang (mis. tracking tidak valid, dispute berulang)."},
    "Policy Abuser":     {"color": "#d97706", "bg": "#fef3c7", "desc": "Memanfaatkan celah kebijakan retur secara berulang."},
    "Wardrobing":        {"color": "#7c3aed", "bg": "#ede9fe", "desc": "Indikasi membeli lalu mengembalikan barang setelah dipakai."},
}

FEATURE_LABELS = {
    "return_rate_pct": "Return rate (%)",
    "total_returns_lifetime": "Total returns",
    "customer_support_contacts": "Support contacts",
    "days_to_return": "Days to return",
    "previous_dispute_count": "Previous disputes",
    "wishlist_to_cart_time_hrs": "Wishlist → cart (hrs)",
    "tracking_number_valid": "Tracking valid",
    "refund_amount_requested_usd": "Refund amount (USD)",
    "avg_order_value_usd": "Avg order value (USD)",
    "total_orders_lifetime": "Total orders",
}


@st.cache_resource(show_spinner="Memuat model AI...")
def load_model():
    if not MODEL_PATH.exists():
        return None, f"File model tidak ditemukan: {MODEL_PATH.name} (taruh sefolder dengan main.py)"
    try:
        return joblib.load(MODEL_PATH), None
    except Exception as e:  # versi scikit-learn tidak cocok, file rusak, dll
        return None, f"Gagal memuat model: {e}"


model, model_error = load_model()


def predict_case(features: dict):
    """Jalankan model -> (label, confidence, {kelas: probabilitas})"""
    X = pd.DataFrame([features])[list(model.feature_names_in_)]  # urutan kolom harus sama dengan saat training
    proba = model.predict_proba(X)[0]
    probs = dict(zip(model.classes_, proba))
    label = max(probs, key=probs.get)
    return label, float(probs[label]), probs


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
    display: grid; align-items: center; column-gap: 12px; min-width: 860px;
    grid-template-columns: 1.1fr 1fr 0.8fr 0.8fr 1fr 1fr 1fr 1.9fr;
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

/* ---------- Hasil model ---------- */
.prob-row { margin-bottom: 10px; }
.prob-top { display: flex; justify-content: space-between; font-size: 12px; font-weight: 600; margin-bottom: 4px; }
.prob-bg { width: 100%; height: 8px; border-radius: 8px; background: rgba(0,0,0,.08); }
.prob-fill { height: 8px; border-radius: 8px; }
.result-label { font-size: 20px; font-weight: 800; margin: 2px 0 4px 0; }
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
            lbl = item.get("label", item["result"])
            conf = item.get("confidence")
            conf_txt = f" · {conf:.0f}%" if conf is not None else ""
            if item["result"] == "Fraudulent":
                badge = f'<span class="badge-table-fraud">⚠️ {lbl}{conf_txt}</span>'
            else:
                badge = f'<span class="badge-table-safe">✅ {lbl}{conf_txt}</span>'
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
    st.caption("Isi data kasus retur, lalu model AI akan menganalisis pola penyalahgunaannya.")

    if model is None:
        st.error(model_error)
        st.stop()

    col1, col2 = st.columns([2.1, 1], gap="large")

    with col1:
        st.markdown("### Customer & return features")
        st.caption("All fields are numeric and non-negative.")

        c1, c2 = st.columns(2)
        with c1:
            account_age = st.number_input("ACCOUNT AGE", min_value=0, value=0, step=1, help="days (hanya untuk riwayat, bukan fitur model)")
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
            avg_order_value = st.number_input("AVG ORDER VALUE (USD)", min_value=0.0, value=0.0, step=1.0, help="USD")

        c7, c8 = st.columns(2)
        with c7:
            support_contacts = st.number_input("CUSTOMER SUPPORT CONTACTS", min_value=0, value=0, step=1, help="contacts")
        with c8:
            dispute_count = st.number_input("PREVIOUS DISPUTE COUNT", min_value=0, value=0, step=1, help="disputes")

        c9, c10 = st.columns(2)
        with c9:
            wishlist_hrs = st.number_input("WISHLIST TO CART TIME (HRS)", min_value=0.0, value=0.0, step=0.5, help="hours")
        with c10:
            tracking_valid_label = st.selectbox("TRACKING NUMBER VALID", ["Yes", "No"], help="Apakah nomor resi valid?")
            tracking_valid = 1 if tracking_valid_label == "Yes" else 0

        st.markdown("<p style='margin-bottom: 5px; font-size:12px; font-weight:bold; color:#555;'>Derived return rate</p>",
                    unsafe_allow_html=True)
        return_rate = (total_returns / total_orders) * 100 if total_orders > 0 else 0.0
        box = "metric-box-danger" if return_rate >= 60.0 else "metric-box-normal"
        st.markdown(f'<div class="{box}">{return_rate:.1f}%</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🪄 Predict return behavior"):
            if total_orders == 0:
                st.warning("Total Orders tidak boleh 0 agar perhitungan rasio valid.")
            elif total_returns > total_orders:
                st.error("Validasi gagal: Total Returns tidak boleh lebih besar dari Total Orders!")
            else:
                # Nama kunci HARUS sama persis dengan fitur saat model dilatih
                features = {
                    "return_rate_pct": return_rate,
                    "total_returns_lifetime": total_returns,
                    "customer_support_contacts": support_contacts,
                    "days_to_return": days_until_return,
                    "previous_dispute_count": dispute_count,
                    "wishlist_to_cart_time_hrs": wishlist_hrs,
                    "tracking_number_valid": tracking_valid,
                    "refund_amount_requested_usd": refund_amount,
                    "avg_order_value_usd": avg_order_value,
                    "total_orders_lifetime": total_orders,
                }
                label, confidence, probs = predict_case(features)
                is_flagged = label != "Legitimate"
                info = CLASS_INFO.get(label, CLASS_INFO["Legitimate"])

                prob_rows = ""
                for cls, p in sorted(probs.items(), key=lambda kv: kv[1], reverse=True):
                    ci = CLASS_INFO.get(cls, {"color": "#64748b"})
                    prob_rows += (
                        f'<div class="prob-row"><div class="prob-top"><span>{cls}</span><span>{p*100:.1f}%</span></div>'
                        f'<div class="prob-bg"><div class="prob-fill" style="width:{p*100:.1f}%; background:{ci["color"]};"></div></div></div>'
                    )

                box_cls = "high-risk-box" if is_flagged else "low-risk-box"
                icon = "⚠️" if is_flagged else "✅"
                st.markdown(h(f"""
                    <div class="{box_cls}">
                        <div style="font-size:12px; font-weight:700; letter-spacing:.5px;">{icon} HASIL ANALISIS MODEL AI</div>
                        <div class="result-label">{label}</div>
                        <div style="font-size:13px; margin-bottom:14px;">Confidence <b>{confidence*100:.1f}%</b> — {info["desc"]}</div>
                        {prob_rows}
                    </div>
                """), unsafe_allow_html=True)

                st.session_state.history_predictions.append({
                    "case": f"#RF-{len(history) + 2049}",
                    "account_age": account_age,
                    "total_orders": total_orders,
                    "total_returns": total_returns,
                    "refund_amount": float(refund_amount),
                    "days_until_return": days_until_return,
                    "return_rate": round(return_rate, 1),
                    "label": label,
                    "confidence": round(confidence * 100, 1),
                    "result": "Fraudulent" if is_flagged else "Low Risk",
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
        for lbl, ok in checks:
            cls = "val-ok" if ok else "val-bad"
            ic = "✅" if ok else "❌"
            items += f'<div class="val-item {cls}"><span>{lbl}</span><span>{ic}</span></div>'

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
    st.markdown("<h2 style='margin-bottom: 0;'>🧠 Model Insight</h2>", unsafe_allow_html=True)
    st.caption("Informasi mengenai model machine learning yang digunakan.")
    st.markdown("<br>", unsafe_allow_html=True)

    if model is None:
        st.error(model_error)
        st.stop()

    m1, m2 = st.columns([1, 1.6], gap="large")

    with m1:
        class_rows = "".join(
            f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:10px; font-size:13px;">'
            f'<span style="width:10px; height:10px; border-radius:50%; background:{CLASS_INFO.get(c, {"color": "#64748b"})["color"]};"></span>'
            f'<b>{c}</b></div>'
            for c in model.classes_
        )
        st.markdown(h(f"""
            <div class="dashboard-card">
                <div class="card-title">Model summary</div>
                <div class="card-sub" style="margin-bottom:16px;">{type(model).__name__}</div>
                <div style="font-size:13px; margin-bottom:6px;">Jumlah fitur: <b>{model.n_features_in_}</b></div>
                <div style="font-size:13px; margin-bottom:16px;">Jumlah pohon: <b>{getattr(model, "n_estimators", "-")}</b></div>
                <div class="stat-label" style="margin-bottom:10px;">KELAS OUTPUT</div>
                {class_rows}
            </div>
        """), unsafe_allow_html=True)

    with m2:
        imp = sorted(zip(model.feature_names_in_, model.feature_importances_), key=lambda kv: kv[1], reverse=True)
        top = imp[0][1] if imp else 1
        imp_rows = "".join(
            f'<div class="prob-row"><div class="prob-top"><span>{FEATURE_LABELS.get(n, n)}</span><span>{v*100:.1f}%</span></div>'
            f'<div class="prob-bg"><div class="prob-fill" style="width:{v/top*100:.1f}%; background:#6366f1;"></div></div></div>'
            for n, v in imp
        )
        st.markdown(h(f"""
            <div class="dashboard-card">
                <div class="card-title">Feature importance</div>
                <div class="card-sub" style="margin-bottom:16px;">Seberapa besar tiap fitur memengaruhi keputusan model</div>
                {imp_rows}
            </div>
        """), unsafe_allow_html=True)
