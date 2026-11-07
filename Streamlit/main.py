import json
import math
import secrets
import time
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

COOLDOWN_SECONDS = 5  # jeda antar prediksi


def h(s: str) -> str:
    """Rapatkan HTML jadi satu baris.
    Penyebab HTML tampil sebagai kode di Streamlit: baris yang diawali 4+ spasi
    (atau baris kosong) dianggap code block oleh Markdown."""
    return " ".join(line.strip() for line in s.strip().splitlines())


# ----------------- LOAD MODEL (sekali saja, di-cache) -----------------
MODEL_PATH = Path(__file__).parent / "return_abuse_detector_model.joblib"
METRICS_PATH = Path(__file__).parent / "model_metrics.json"  # dibuat oleh evaluate_model.py

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


@st.cache_data(show_spinner=False)
def _read_metrics(path_str: str, mtime: float):
    # mtime ikut jadi kunci cache, jadi file JSON baru otomatis terbaca
    try:
        return json.loads(Path(path_str).read_text(encoding="utf-8"))
    except Exception:
        return None


def load_metrics():
    if not METRICS_PATH.exists():
        return None
    return _read_metrics(str(METRICS_PATH), METRICS_PATH.stat().st_mtime)


model, model_error = load_model()


def detect_theme() -> str:
    """'light' atau 'dark' sesuai tema Streamlit yang aktif (termasuk mode System)."""
    try:
        t = st.context.theme.type  # Streamlit >= 1.46
        if t in ("light", "dark"):
            return t
    except Exception:
        pass
    return "dark" if st.get_option("theme.base") == "dark" else "light"


IS_DARK = detect_theme() == "dark"


def predict_case(features: dict):
    """Jalankan model -> (label, confidence, {kelas: probabilitas})"""
    X = pd.DataFrame([features])[list(model.feature_names_in_)]  # urutan kolom harus sama dengan saat training
    proba = model.predict_proba(X)[0]
    probs = dict(zip(model.classes_, proba))
    label = max(probs, key=probs.get)
    return label, float(probs[label]), probs


CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # tanpa O/0/I/1 supaya tidak membingungkan


def generate_case_code(existing_codes) -> str:
    """Buat kode unik acak, mis. RG-7K3QX9. Dijamin tidak sama dengan kode yang sudah ada."""
    existing = set(existing_codes)
    while True:
        code = "RG-" + "".join(secrets.choice(CODE_ALPHABET) for _ in range(6))
        if code not in existing:
            return code


# ----------------- CSS -----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* =====================================================
   DESIGN TOKENS
   ===================================================== */
:root {
    --ink: #0f172a;
    --ink-2: #334155;
    --muted: #64748b;
    --faint: #94a3b8;
    --line: #e6eaf2;
    --surface: #ffffff;
    --surface-2: #f8fafc;
    --page: #f4f6fb;

    --brand: #6366f1;
    --brand-2: #7c3aed;
    --brand-soft: #eef2ff;
    --brand-ink: #4338ca;

    --ok: #16a34a;     --ok-soft: #dcfce7;   --ok-ink: #166534;
    --bad: #dc2626;    --bad-soft: #fee2e2;  --bad-ink: #991b1b;
    --warn: #d97706;   --warn-soft: #fef3c7; --warn-ink: #92400e;
    --info: #2563eb;   --info-soft: #dbeafe; --info-ink: #1e40af;

    --navy-1: #0b1626;
    --navy-2: #12213a;
    --navy-3: #1b2d4d;
    --navy-line: #25395c;

    --r-sm: 8px;
    --r-md: 12px;
    --r-lg: 16px;
    --r-xl: 22px;

    --shadow-1: 0 1px 2px rgba(15,23,42,.05), 0 1px 3px rgba(15,23,42,.04);
    --shadow-2: 0 4px 12px rgba(15,23,42,.06), 0 2px 4px rgba(15,23,42,.04);
    --shadow-brand: 0 8px 22px rgba(99,102,241,.35);

    --ease: cubic-bezier(.22,.8,.3,1);
}

/* =====================================================
   BASE
   ===================================================== */
html, body, [class*="css"], .stApp, .stMarkdown, button, input, select, textarea {
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
}
[data-testid="stAppViewContainer"] { background: var(--page); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1280px; }
footer { visibility: hidden; }
h1, h2, h3 { letter-spacing: -.02em; color: var(--ink); }
a { text-underline-offset: 3px; }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 10px; border: 2px solid transparent; background-clip: padding-box; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; background-clip: padding-box; border: 2px solid transparent; }

/* =====================================================
   MOTION (dipakai hemat: hero, hasil prediksi, panel validasi)
   ===================================================== */
@keyframes fadeUp   { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
@keyframes popIn    { 0% { transform: scale(.97); opacity: 0; } 100% { transform: scale(1); opacity: 1; } }
@keyframes navPulse { 0% { box-shadow: 0 0 0 0 rgba(99,102,241,.5); } 100% { box-shadow: 0 0 0 14px rgba(99,102,241,0); } }
@keyframes growX    { from { transform: scaleX(0); } to { transform: scaleX(1); } }

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
}

/* =====================================================
   SIDEBAR
   ===================================================== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--navy-1) 0%, var(--navy-2) 100%) !important;
    border-right: 1px solid var(--navy-line);
    padding-top: 1rem;
}
[data-testid="stSidebar"] div[role="radiogroup"] { gap: 8px; }
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    position: relative; width: 100%; box-sizing: border-box; margin: 0;
    background: rgba(255,255,255,.03);
    border: 1px solid transparent; border-radius: var(--r-md);
    padding: 12px 14px; color: #a8b5cb; font-weight: 600; cursor: pointer;
    transition: background .2s var(--ease), color .2s var(--ease), border-color .2s var(--ease), transform .2s var(--ease);
}
/* sembunyikan bulatan radio bawaan */
[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child { display: none !important; }
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
[data-testid="stSidebar"] div[role="radiogroup"] > label p { font-size: 14.5px; color: inherit; margin: 0; }

[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: rgba(99,102,241,.14); border-color: rgba(99,102,241,.45);
    color: #fff; transform: translateX(3px);
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:active { transform: scale(.98); }
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked),
[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
    background: linear-gradient(135deg, var(--brand) 0%, var(--brand-2) 100%) !important;
    border-color: rgba(255,255,255,.25) !important; color: #fff !important;
    box-shadow: var(--shadow-brand);
    animation: navPulse .6s ease-out;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:focus-within { outline: 2px solid #a5b4fc; outline-offset: 2px; }

/* =====================================================
   BUTTON
   ===================================================== */
.stButton > button {
    position: relative; overflow: hidden;
    background: linear-gradient(135deg, var(--brand) 0%, var(--brand-2) 100%);
    color: #fff; font-weight: 700; border: none; letter-spacing: .01em;
    border-radius: var(--r-md); width: 100%; padding: .7rem 1rem;
    box-shadow: 0 2px 8px rgba(99,102,241,.3);
    transition: transform .18s var(--ease), box-shadow .25s var(--ease), filter .25s var(--ease);
}
.stButton > button:hover { color: #fff; transform: translateY(-1px); box-shadow: var(--shadow-brand); filter: brightness(1.06); }
.stButton > button:active { transform: translateY(0) scale(.98); box-shadow: 0 2px 6px rgba(99,102,241,.4); }
.stButton > button:focus:not(:active) { color: #fff; }
.stButton > button:focus-visible { outline: 3px solid #c7d2fe; outline-offset: 2px; }
.stButton > button::after {
    content: ""; position: absolute; inset: 0; pointer-events: none;
    background-image: radial-gradient(circle, #fff 10%, transparent 10.01%);
    background-repeat: no-repeat; background-position: 50%;
    transform: scale(10); opacity: 0; transition: transform .5s, opacity 1s;
}
.stButton > button:active::after { transform: scale(0); opacity: .3; transition: 0s; }
.stButton > button:disabled {
    background: #cbd5e1 !important; color: #64748b !important;
    box-shadow: none !important; transform: none !important; filter: none !important; cursor: not-allowed;
}

/* =====================================================
   FORM INPUT
   ===================================================== */
[data-testid="stNumberInput"] label p, [data-testid="stSelectbox"] label p {
    font-size: 11.5px !important; font-weight: 700 !important; color: var(--muted) !important; letter-spacing: .04em;
}
[data-baseweb="input"], [data-baseweb="select"] > div, [data-testid="stNumberInput"] input {
    border-radius: var(--r-md) !important;
}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {
    box-shadow: 0 0 0 3px rgba(99,102,241,.2) !important; border-color: var(--brand) !important;
}

/* =====================================================
   TABS
   ===================================================== */
[data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
button[data-baseweb="tab"] {
    background: transparent; border-radius: var(--r-md) var(--r-md) 0 0; padding: 10px 16px;
    color: var(--muted); font-weight: 600; transition: color .2s, background .2s;
}
button[data-baseweb="tab"]:hover { color: var(--brand-ink); background: var(--brand-soft); }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--brand-ink); }
[data-baseweb="tab-highlight"] { background-color: var(--brand) !important; height: 3px !important; border-radius: 3px; }

/* =====================================================
   CARD
   ===================================================== */
.dashboard-card {
    background: var(--surface); padding: 22px; border-radius: var(--r-lg);
    border: 1px solid var(--line); box-shadow: var(--shadow-1);
    color: var(--ink); height: 100%; box-sizing: border-box;
    transition: box-shadow .2s var(--ease), border-color .2s var(--ease);
}
.dashboard-card:hover { box-shadow: var(--shadow-2); border-color: #dbe1ee; }
.card-title { font-size: 16px; font-weight: 700; color: var(--ink); letter-spacing: -.01em; }
.card-sub { font-size: 12.5px; color: var(--muted); line-height: 1.5; }
.card-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; gap: 12px; }

.stat-label { font-size: 12px; font-weight: 600; color: var(--muted); letter-spacing: .02em; }
.stat-value { font-size: 34px; font-weight: 800; color: var(--ink); margin-top: 14px; line-height: 1; letter-spacing: -.03em; font-variant-numeric: tabular-nums; }
.stat-icon { padding: 8px; border-radius: var(--r-md); font-size: 16px; line-height: 1; }

/* =====================================================
   DASHBOARD: PREDICTION MIX
   ===================================================== */
.mix-body { display: flex; align-items: center; gap: 22px; margin: 6px 0 20px 0; }
.donut { width: 120px; height: 120px; border-radius: 50%; position: relative; flex-shrink: 0; }
.donut-hole {
    position: absolute; inset: 16px; background: var(--surface); border-radius: 50%;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    box-shadow: inset 0 0 0 1px var(--line);
}
.donut-hole b { font-size: 22px; color: var(--ink); line-height: 1; font-variant-numeric: tabular-nums; }
.donut-hole span { font-size: 10.5px; color: var(--faint); margin-top: 3px; }
.mix-legend { flex: 1; min-width: 0; }
.mix-row { margin-bottom: 14px; }
.mix-row:last-child { margin-bottom: 0; }
.mix-row-top { display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; margin-bottom: 6px; }
.mix-row-top small { font-weight: 500; color: var(--faint); margin-left: 4px; }
.bar-bg { width: 100%; border-radius: 8px; height: 8px; overflow: hidden; }
.bar-fill { height: 8px; border-radius: 8px; transform-origin: left; animation: growX .7s var(--ease) both; }
.mix-foot { font-size: 11.5px; color: var(--faint); text-align: center; padding-top: 14px; border-top: 1px dashed var(--line); }

/* =====================================================
   DASHBOARD: HISTORY TABLE
   ===================================================== */
.hist-scroll { overflow-x: auto; }
.hist-grid {
    display: grid; align-items: center; column-gap: 12px; min-width: 860px;
    grid-template-columns: 1.1fr 1fr 0.8fr 0.8fr 1fr 1fr 1fr 1.9fr;
}
.hist-head {
    padding: 10px 14px; background: var(--surface-2); border-radius: var(--r-md);
    font-size: 11.5px; font-weight: 700; color: var(--muted); letter-spacing: .03em;
}
.hist-row { padding: 14px; border-bottom: 1px solid #f1f5f9; font-size: 13px; color: var(--ink-2); transition: background .15s; font-variant-numeric: tabular-nums; }
.hist-row:hover { background: var(--surface-2); }
.hist-row:last-child { border-bottom: none; }
.hist-row .cell-case { font-weight: 700; color: var(--ink); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; }
.hist-empty { text-align: center; color: var(--faint); font-size: 13px; padding: 28px 0 10px 0; }

.badge-table-fraud, .badge-table-safe {
    padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; display: inline-block; white-space: nowrap;
}
.badge-table-fraud { background: var(--bad-soft); color: var(--bad); }
.badge-table-safe  { background: var(--ok-soft);  color: var(--ok-ink); }

/* =====================================================
   PREDICTION PAGE
   ===================================================== */
.high-risk-box, .low-risk-box {
    padding: 18px 20px; border-radius: var(--r-md); margin-top: 15px; animation: popIn .35s var(--ease) both;
}
.high-risk-box { background: #fef2f2; border: 1px solid #fecaca; border-left: 5px solid var(--bad); color: var(--bad-ink); }
.low-risk-box  { background: #f0fdf4; border: 1px solid #bbf7d0; border-left: 5px solid var(--ok);  color: #14532d; }

.rate-label { font-size: 12px; font-weight: 600; letter-spacing: .02em; color: var(--muted); margin-bottom: 8px; }
.rate-card {
    background: linear-gradient(135deg, var(--navy-2) 0%, var(--navy-3) 100%);
    border: 1px solid var(--navy-line); border-left: 5px solid var(--c);
    border-radius: var(--r-md); padding: 16px 20px;
}
.rate-top { display: flex; justify-content: space-between; align-items: center; }
.rate-value { font-size: 34px; font-weight: 800; color: var(--c); line-height: 1; letter-spacing: -.03em; font-variant-numeric: tabular-nums; }
.rate-chip { font-size: 11px; font-weight: 700; letter-spacing: .04em; padding: 5px 12px; border-radius: 999px; background: var(--bg); color: var(--c); }
.rate-bar-bg { position: relative; width: 100%; height: 6px; border-radius: 6px; background: rgba(148,163,184,.2); margin-top: 14px; }
.rate-bar-fill { height: 6px; border-radius: 6px; background: var(--c); transition: width .4s var(--ease); }
.rate-bar-mark { position: absolute; left: 60%; top: -3px; width: 2px; height: 12px; background: #f59e0b; }
.rate-foot { font-size: 11.5px; color: #9fb0c8; margin-top: 8px; }

/* ---------- Validation panel ---------- */
.val-card {
    background: linear-gradient(160deg, #111827 0%, #0f172a 100%);
    padding: 24px; border-radius: var(--r-lg); color: #fff; border: 1px solid #1f2937; box-shadow: var(--shadow-2);
}
.val-pill { display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 11px; font-weight: 700; margin-bottom: 14px; letter-spacing: .04em; }
.pill-ok  { background: #064e3b; color: #6ee7b7; }
.pill-bad { background: #7f1d1d; color: #fca5a5; }
.val-title { color: #fff; font-size: 18px; font-weight: 700; margin: 0 0 18px 0; }
.val-item { display: flex; align-items: center; justify-content: space-between; padding: 10px 12px; border-radius: var(--r-md); margin-bottom: 8px; font-size: 14px; }
.val-ok  { background: rgba(16,185,129,.10); color: #d1fae5; }
.val-bad { background: rgba(239,68,68,.14);  color: #fecaca; }
.val-note { font-size: 12px; color: #9ca3af; line-height: 1.6; margin-top: 16px; padding-top: 14px; border-top: 1px solid #374151; }

/* ---------- Hasil model (probabilitas) ---------- */
.prob-row { margin-bottom: 10px; }
.prob-top { display: flex; justify-content: space-between; font-size: 12.5px; font-weight: 600; margin-bottom: 4px; gap: 8px; }
.prob-bg { width: 100%; height: 8px; border-radius: 8px; background: rgba(15,23,42,.08); overflow: hidden; }
.prob-fill { height: 8px; border-radius: 8px; transform-origin: left; animation: growX .7s var(--ease) both; }
.result-label { font-size: 22px; font-weight: 800; margin: 2px 0 4px 0; letter-spacing: -.02em; }

/* ---------- Detail customer (dialog) ---------- */
.detail-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; gap: 10px; flex-wrap: wrap; }
.detail-case { font-size: 22px; font-weight: 800; letter-spacing: -.02em; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 18px; }
.detail-item { background: rgba(148,163,184,.12); border: 1px solid rgba(148,163,184,.18); border-radius: var(--r-md); padding: 10px 14px; }
.detail-item small { display: block; font-size: 11.5px; font-weight: 600; letter-spacing: .02em; opacity: .65; margin-bottom: 2px; }
.detail-item b { font-size: 16px; font-variant-numeric: tabular-nums; }
.detail-section { font-size: 12.5px; font-weight: 700; letter-spacing: .02em; opacity: .7; margin: 6px 0 10px 0; }

/* =====================================================
   HOME
   ===================================================== */
.home-hero {
    position: relative; overflow: hidden;
    background: linear-gradient(135deg, #2e2a7a 0%, #5558e8 55%, #8b5cf6 100%);
    border-radius: var(--r-xl); padding: 44px 40px; color: #fff;
    box-shadow: 0 16px 40px rgba(79,70,229,.35);
    animation: fadeUp .5s var(--ease) both;
}
.home-hero::before, .home-hero::after {
    content: ""; position: absolute; border-radius: 50%; pointer-events: none;
    background: radial-gradient(circle, rgba(255,255,255,.18) 0%, transparent 70%);
}
.home-hero::before { width: 380px; height: 380px; right: -90px; top: -140px; }
.home-hero::after  { width: 260px; height: 260px; right: 160px; bottom: -150px; opacity: .6; }
.home-hero > * { position: relative; z-index: 1; }
.home-badge { display: inline-block; font-size: 12px; font-weight: 700; letter-spacing: .04em; padding: 6px 14px; border-radius: 999px; background: rgba(255,255,255,.16); color: #fff; border: 1px solid rgba(255,255,255,.2); }
.home-title { font-size: 46px; font-weight: 800; margin: 14px 0 8px 0; color: #fff; line-height: 1.08; letter-spacing: -.035em; }
.home-sub { font-size: 16px; line-height: 1.7; max-width: 720px; color: #e0e7ff; }
.home-cta { display: inline-block; margin-top: 22px; font-size: 13.5px; color: #fff; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.2); padding: 10px 16px; border-radius: var(--r-md); }

.home-h { font-size: 18px; font-weight: 800; color: var(--ink); margin-bottom: 12px; letter-spacing: -.02em; }
.home-p { font-size: 14px; line-height: 1.75; color: var(--ink-2); margin: 0 0 10px 0; }
.home-list { font-size: 14px; line-height: 1.7; color: var(--ink-2); padding-left: 20px; margin: 0; }
.home-list li { margin-bottom: 6px; }
.home-section { font-size: 21px; font-weight: 800; margin: 4px 0 14px 0; letter-spacing: -.025em; color: var(--ink); }

.chip { display: inline-block; padding: 5px 12px; border-radius: 999px; margin: 0 6px 8px 0; font-size: 12px; font-weight: 600; background: var(--brand-soft); color: var(--brand-ink); }
.chip-ok  { background: var(--ok-soft);  color: var(--ok-ink); }
.chip-bad { background: var(--bad-soft); color: var(--bad); }

.step { display: flex; gap: 14px; align-items: flex-start; padding: 14px 0; border-bottom: 1px dashed var(--line); }
.step:last-child { border-bottom: none; }
.step-num {
    width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0;
    background: linear-gradient(135deg, var(--brand) 0%, var(--brand-2) 100%); color: #fff;
    font-weight: 800; font-size: 14px; display: flex; align-items: center; justify-content: center;
}
.step-title { font-size: 14px; font-weight: 700; color: var(--ink); }
.step-desc { font-size: 13px; color: var(--muted); line-height: 1.6; margin-top: 2px; }

/* ---------- Tech stack ---------- */
.tech-card {
    background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-lg);
    box-shadow: var(--shadow-1); text-align: center; padding: 22px 12px 18px 12px; height: 100%; box-sizing: border-box;
    transition: transform .2s var(--ease), box-shadow .2s var(--ease), border-color .2s var(--ease);
}
.tech-card:hover { transform: translateY(-3px); box-shadow: var(--shadow-2); border-color: #c7d2fe; }
.tech-logo { height: 44px; width: 44px; object-fit: contain; display: block; margin: 0 auto 12px auto; }
.tech-logo-fallback {
    height: 44px; width: 44px; margin: 0 auto 12px auto; border-radius: 12px;
    background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%); color: #fff;
    display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 17px; letter-spacing: -.03em;
}
.tech-name { font-size: 14.5px; font-weight: 700; color: var(--ink); }
.tech-desc { font-size: 12px; color: var(--muted); margin-top: 4px; line-height: 1.45; }

/* =====================================================
   MODEL INSIGHT
   ===================================================== */
.kpi {
    background: var(--surface); border: 1px solid var(--line); border-top: 4px solid var(--c, var(--brand));
    border-radius: var(--r-lg); padding: 18px 20px; box-shadow: var(--shadow-1); height: 100%; box-sizing: border-box;
}
.kpi-label { font-size: 12px; font-weight: 600; color: var(--muted); letter-spacing: .02em; }
.kpi-value { font-size: 30px; font-weight: 800; color: var(--ink); letter-spacing: -.03em; margin-top: 8px; line-height: 1; font-variant-numeric: tabular-nums; }
.kpi-sub { font-size: 12px; color: var(--faint); margin-top: 8px; line-height: 1.45; }

.callout { border-radius: var(--r-md); padding: 14px 16px; font-size: 13.5px; line-height: 1.65; border: 1px solid; margin: 0 0 4px 0; }
.callout b { display: block; margin-bottom: 2px; font-size: 13.5px; }
.callout-info { background: var(--info-soft); border-color: #bfdbfe; color: var(--info-ink); }
.callout-warn { background: var(--warn-soft); border-color: #fde68a; color: var(--warn-ink); }
.callout-ok   { background: var(--ok-soft);   border-color: #bbf7d0; color: var(--ok-ink); }
.callout-bad  { background: var(--bad-soft);  border-color: #fecaca; color: var(--bad-ink); }

.split-bar { display: flex; height: 44px; border-radius: var(--r-md); overflow: hidden; border: 1px solid var(--line); margin: 6px 0 10px 0; }
.split-train { background: linear-gradient(135deg, var(--brand) 0%, var(--brand-2) 100%); color: #fff; }
.split-test  { background: #f59e0b; color: #fff; }
.split-bar > div { display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 700; white-space: nowrap; }

.tbl-scroll { overflow-x: auto; }
.tbl { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
.tbl th { text-align: left; font-size: 11.5px; font-weight: 700; color: var(--muted); letter-spacing: .03em; padding: 10px 12px; background: var(--surface-2); }
.tbl th:first-child { border-radius: var(--r-md) 0 0 var(--r-md); }
.tbl th:last-child  { border-radius: 0 var(--r-md) var(--r-md) 0; }
.tbl td { padding: 12px; border-bottom: 1px solid #f1f5f9; color: var(--ink-2); }
.tbl tr:last-child td { border-bottom: none; }
.tbl td:first-child { font-weight: 700; color: var(--ink); }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 8px; }

.cm { border-collapse: separate; border-spacing: 4px; margin: 0 auto; font-variant-numeric: tabular-nums; }
.cm th { font-size: 11.5px; font-weight: 600; color: var(--muted); padding: 6px 8px; text-align: center; }
.cm th.rowh { text-align: right; max-width: 130px; }
.cm td { text-align: center; padding: 14px 16px; border-radius: var(--r-sm); min-width: 92px; font-size: 14px; font-weight: 700; line-height: 1.25; }
.cm td small { display: block; font-size: 10.5px; font-weight: 500; opacity: .85; }
.cm-axis { text-align: center; font-size: 11.5px; color: var(--faint); margin: 4px 0; }

.param-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; }
.param { background: var(--surface-2); border: 1px solid var(--line); border-radius: var(--r-md); padding: 8px 12px; }
.param small { display: block; font-size: 11px; color: var(--muted); font-weight: 600; }
.param b { font-size: 13px; color: var(--ink); word-break: break-word; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-weight: 600; }

.concept { background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-lg); padding: 18px 20px; box-shadow: var(--shadow-1); height: 100%; box-sizing: border-box; }
.concept-title { font-size: 15px; font-weight: 700; color: var(--ink); margin-bottom: 6px; letter-spacing: -.01em; }
.concept-body { font-size: 13.2px; color: var(--ink-2); line-height: 1.7; }
.concept-body code, .callout code, .home-p code { background: var(--brand-soft); color: var(--brand-ink); padding: 1px 6px; border-radius: 6px; font-size: 12px; }
.concept-tag { display: inline-block; font-size: 11px; font-weight: 700; padding: 2px 9px; border-radius: 999px; margin-bottom: 8px; background: var(--brand-soft); color: var(--brand-ink); }
.concept-tag.warn { background: var(--warn-soft); color: var(--warn-ink); }
.concept-tag.ok   { background: var(--ok-soft);   color: var(--ok-ink); }

/* =====================================================
   RESPONSIVE
   ===================================================== */
@media (max-width: 900px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .home-hero { padding: 30px 22px; }
    .home-title { font-size: 32px; }
    .mix-body { flex-direction: column; align-items: flex-start; }
    .detail-grid { grid-template-columns: 1fr; }
    .stat-value { font-size: 28px; }
    .cm td { min-width: 64px; padding: 10px 8px; }
}
</style>
""", unsafe_allow_html=True)

# ----------------- THEME LAYER (light / dark / system) -----------------
# Harus setelah blok CSS utama supaya menimpa aturan lama.
LIGHT_TOKENS = """
:root {
    color-scheme: light;
    --line-soft:#f1f5f9; --line-hover:#dbe1ee; --track:rgba(15,23,42,.08);
    --ok-line:#bbf7d0; --bad-line:#fecaca; --warn-line:#fde68a; --info-line:#bfdbfe;
    --btn-off-bg:#cbd5e1; --btn-off-fg:#64748b;

    --side-bg:linear-gradient(180deg,#ffffff 0%,#f3f5fb 100%); --side-line:#e6eaf2;
    --side-ink:#0f172a; --side-muted:#64748b;
    --side-item:#475569; --side-item-bg:rgba(15,23,42,.03);
    --side-hover-bg:#eef2ff; --side-hover-line:rgba(99,102,241,.4); --side-hover-ink:#4338ca;

    --rate-bg:linear-gradient(135deg,#ffffff 0%,#f8fafc 100%); --rate-line:#e6eaf2;
    --rate-foot:#64748b; --rate-track:rgba(15,23,42,.10);

    --val-bg:#ffffff; --val-line:#e6eaf2; --val-ink:#0f172a;
    --val-note:#64748b; --val-note-line:#e6eaf2;
    --val-ok-bg:#dcfce7;  --val-ok-fg:#166534;
    --val-bad-bg:#fee2e2; --val-bad-fg:#991b1b;
    --pill-ok-bg:#dcfce7; --pill-ok-fg:#166534;
    --pill-bad-bg:#fee2e2; --pill-bad-fg:#991b1b;
}
"""

DARK_TOKENS = """
:root {
    color-scheme: dark;
    --ink:#e8ecf6; --ink-2:#c5cde0; --muted:#9aa7be; --faint:#7d8aa3;
    --line:#24314a; --surface:#111a2e; --surface-2:#18233b; --page:#0a1020;
    --brand-soft:rgba(99,102,241,.22); --brand-ink:#a5b4fc;
    --ok-soft:rgba(22,163,74,.18);   --ok-ink:#86efac;
    --bad-soft:rgba(220,38,38,.18);  --bad-ink:#fca5a5;
    --warn-soft:rgba(217,119,6,.20); --warn-ink:#fcd34d;
    --info-soft:rgba(37,99,235,.20); --info-ink:#93c5fd;
    --line-soft:#1b2740; --line-hover:#344563; --track:rgba(148,163,184,.22);
    --ok-line:rgba(22,163,74,.4); --bad-line:rgba(220,38,38,.4);
    --warn-line:rgba(217,119,6,.45); --info-line:rgba(37,99,235,.45);
    --btn-off-bg:#273349; --btn-off-fg:#7d8aa3;

    --side-bg:linear-gradient(180deg,#0b1626 0%,#12213a 100%); --side-line:#25395c;
    --side-ink:#ffffff; --side-muted:#94a3b8;
    --side-item:#a8b5cb; --side-item-bg:rgba(255,255,255,.03);
    --side-hover-bg:rgba(99,102,241,.14); --side-hover-line:rgba(99,102,241,.45); --side-hover-ink:#ffffff;

    --rate-bg:linear-gradient(135deg,#12213a 0%,#1b2d4d 100%); --rate-line:#25395c;
    --rate-foot:#9fb0c8; --rate-track:rgba(148,163,184,.2);

    --val-bg:linear-gradient(160deg,#111827 0%,#0f172a 100%); --val-line:#1f2937; --val-ink:#ffffff;
    --val-note:#9ca3af; --val-note-line:#374151;
    --val-ok-bg:rgba(16,185,129,.10);  --val-ok-fg:#d1fae5;
    --val-bad-bg:rgba(239,68,68,.14);  --val-bad-fg:#fecaca;
    --pill-ok-bg:#064e3b; --pill-ok-fg:#6ee7b7;
    --pill-bad-bg:#7f1d1d; --pill-bad-fg:#fca5a5;
}
"""

THEME_RULES = """
/* ---- page & teks dasar ---- */
.stApp, [data-testid="stAppViewContainer"] { background: var(--page) !important; }
.stApp, .stApp [data-testid="stMarkdownContainer"] { color: var(--ink); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { color: var(--muted) !important; }
[data-testid="stTooltipIcon"] svg { color: var(--faint) !important; }

/* ---- SIDEBAR (ikut tema) ---- */
[data-testid="stSidebar"] { background: var(--side-bg) !important; border-right: 1px solid var(--side-line) !important; }
[data-testid="stSidebar"] svg { color: var(--side-muted) !important; fill: var(--side-muted); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: var(--side-ink); }
[data-testid="stSidebar"] div[role="radiogroup"] > label { background: var(--side-item-bg); color: var(--side-item); }
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: var(--side-hover-bg); border-color: var(--side-hover-line); color: var(--side-hover-ink); }
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) svg { fill: #fff; color: #fff !important; }

/* ---- input angka & selectbox ---- */
[data-baseweb="input"] { background: var(--surface) !important; border: 1px solid var(--line) !important; }
[data-baseweb="base-input"] { background: transparent !important; }
[data-baseweb="input"] input, [data-baseweb="base-input"] input, [data-baseweb="select"] input {
    color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important;
    caret-color: var(--brand); background: transparent !important;
}
input::placeholder { color: var(--faint) !important; -webkit-text-fill-color: var(--faint) !important; opacity: 1; }
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] { background: var(--surface-2) !important; color: var(--ink) !important; }
[data-testid="stNumberInputStepDown"]:hover, [data-testid="stNumberInputStepUp"]:hover { background: var(--brand-soft) !important; }
[data-baseweb="select"] > div { background: var(--surface) !important; border: 1px solid var(--line) !important; }
[data-baseweb="select"] * { color: var(--ink) !important; }
[data-baseweb="select"] svg { fill: var(--muted) !important; }
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {
    border-color: var(--brand) !important; box-shadow: 0 0 0 3px rgba(99,102,241,.25) !important;
}

/* ---- dropdown / popover ---- */
[data-baseweb="popover"] > div, [data-baseweb="popover"] ul { background: var(--surface) !important; color: var(--ink) !important; }
[data-baseweb="popover"] ul { border: 1px solid var(--line); }
[data-baseweb="popover"] li, [data-baseweb="popover"] li * { background: transparent !important; color: var(--ink) !important; }
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"] { background: var(--brand-soft) !important; }

/* ---- tombol nonaktif ---- */
.stButton > button:disabled { background: var(--btn-off-bg) !important; color: var(--btn-off-fg) !important; }

/* ---- DERIVED RETURN RATE CARD (ikut tema) ---- */
.rate-card { background: var(--rate-bg) !important; border-color: var(--rate-line) !important; border-left-color: var(--c) !important; }
.rate-bar-bg { background: var(--rate-track) !important; }
.rate-foot { color: var(--rate-foot) !important; }

/* ---- VALIDATION PANEL (ikut tema) ---- */
.val-card { background: var(--val-bg) !important; border: 1px solid var(--val-line) !important; color: var(--val-ink) !important; }
.val-title { color: var(--val-ink) !important; }
.val-note { color: var(--val-note) !important; border-top-color: var(--val-note-line) !important; }
.val-ok  { background: var(--val-ok-bg) !important;  color: var(--val-ok-fg) !important; }
.val-bad { background: var(--val-bad-bg) !important; color: var(--val-bad-fg) !important; }
.pill-ok  { background: var(--pill-ok-bg) !important;  color: var(--pill-ok-fg) !important; }
.pill-bad { background: var(--pill-bad-bg) !important; color: var(--pill-bad-fg) !important; }

/* ---- komponen buatan sendiri ---- */
.dashboard-card:hover { border-color: var(--line-hover); }
.hist-row, .tbl td { border-bottom-color: var(--line-soft); }
.prob-bg, .bar-bg { background: var(--track) !important; }
.badge-table-fraud { color: var(--bad-ink); }
.high-risk-box { background: var(--bad-soft); border: 1px solid var(--bad-line); border-left: 5px solid var(--bad); color: var(--bad-ink); }
.low-risk-box  { background: var(--ok-soft);  border: 1px solid var(--ok-line);  border-left: 5px solid var(--ok);  color: var(--ok-ink); }
.callout-info { border-color: var(--info-line); }
.callout-warn { border-color: var(--warn-line); }
.callout-ok   { border-color: var(--ok-line); }
.callout-bad  { border-color: var(--bad-line); }

/* ---- SVG trend chart ---- */
.rg-grid { stroke: var(--line); }
.rg-axis { fill: var(--faint); }
.rg-dot  { stroke: var(--surface); }
"""

st.markdown(
    "<style>" + h(DARK_TOKENS if IS_DARK else LIGHT_TOKENS) + " " + h(THEME_RULES) + "</style>",
    unsafe_allow_html=True,
)

# ----------------- NAVIGATION (sidebar) STYLE -----------------
NAV_TOKENS = (
    "--nav-card:rgba(255,255,255,.05); --nav-hover:rgba(99,102,241,.16); --nav-shadow:0 10px 24px rgba(0,0,0,.35);"
    if IS_DARK else
    "--nav-card:#ffffff; --nav-hover:#eef2ff; --nav-shadow:0 8px 20px rgba(99,102,241,.12);"
)

NAV_CSS = """
:root { %s }

[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding: 1.1rem 1rem 1.5rem 1rem; }

/* ---- brand card ---- */
.nav-brand {
    display: flex; align-items: center; gap: 12px; padding: 14px;
    background: var(--nav-card); border: 1px solid var(--side-line);
    border-radius: 16px; box-shadow: var(--nav-shadow);
}
.nav-logo {
    width: 42px; height: 42px; border-radius: 13px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 20px; color: #fff;
    background: linear-gradient(135deg, #6366f1 0%%, #7c3aed 100%%);
    box-shadow: 0 6px 16px rgba(99,102,241,.45);
}
.nav-name { font-size: 18px; font-weight: 800; letter-spacing: -.02em; color: var(--side-ink); line-height: 1.1; }
.nav-tag  { font-size: 11.5px; color: var(--side-muted); margin-top: 3px; }
.nav-label {
    font-size: 10.5px; font-weight: 800; letter-spacing: .14em; color: var(--side-muted);
    margin: 26px 0 10px 8px;
}

/* ---- nav items ---- */
[data-testid="stSidebar"] div[role="radiogroup"] { gap: 6px !important; }
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    position: relative; overflow: hidden;
    padding: 13px 16px !important; border-radius: 14px !important;
    background: transparent !important; border: 1px solid transparent !important;
    color: var(--side-item) !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label p {
    font-size: 14.5px !important; font-weight: 600 !important; letter-spacing: .005em;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: var(--nav-hover) !important; border-color: var(--side-hover-line) !important;
    color: var(--side-hover-ink) !important; transform: translateX(4px);
}
/* active item */
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(135deg, #6366f1 0%%, #7c3aed 100%%) !important;
    border-color: rgba(255,255,255,.22) !important; color: #fff !important;
    box-shadow: 0 8px 20px rgba(99,102,241,.40); transform: none;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked):hover { color: #fff !important; transform: none; }
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked)::after {
    content: ""; position: absolute; right: 16px; top: 50%%; width: 7px; height: 7px;
    border-radius: 50%%; background: #fff; transform: translateY(-50%%);
    box-shadow: 0 0 0 4px rgba(255,255,255,.25);
}

/* ---- status card ---- */
.nav-foot {
    margin-top: 28px; padding: 14px; border-radius: 16px;
    background: var(--nav-card); border: 1px solid var(--side-line);
}
.nav-foot-title { font-size: 10.5px; font-weight: 800; letter-spacing: .14em; color: var(--side-muted); margin-bottom: 10px; }
.nav-stat { display: flex; align-items: center; justify-content: space-between; font-size: 12.5px; color: var(--side-item); padding: 4px 0; }
.nav-stat b { color: var(--side-ink); font-variant-numeric: tabular-nums; }
.nav-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%%; margin-right: 8px; }
.nav-dot.on  { background: #22c55e; box-shadow: 0 0 0 3px rgba(34,197,94,.25); }
.nav-dot.off { background: #ef4444; box-shadow: 0 0 0 3px rgba(239,68,68,.25); }
""" % NAV_TOKENS

st.markdown("<style>" + h(NAV_CSS) + "</style>", unsafe_allow_html=True)

# ----------------- Session State -----------------
if "history_predictions" not in st.session_state:
    st.session_state.history_predictions = []

if "last_predict_ts" not in st.session_state:
    st.session_state.last_predict_ts = 0.0

if "last_result" not in st.session_state:
    st.session_state.last_result = None

history = st.session_state.history_predictions


# ----------------- Helper tampilan -----------------
def pct(v, d=1):
    return f"{float(v) * 100:.{d}f}%"


def kpi_html(label, value, sub="", color="#6366f1"):
    return (f'<div class="kpi" style="--c:{color};"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>')


def callout_html(kind, title, body):
    return f'<div class="callout callout-{kind}"><b>{title}</b>{body}</div>'


def bar_row(label, frac, text, color):
    frac = max(0.0, min(1.0, float(frac)))
    return (f'<div class="prob-row"><div class="prob-top"><span>{label}</span><span>{text}</span></div>'
            f'<div class="prob-bg"><div class="prob-fill" style="width:{frac*100:.1f}%; background:{color};"></div></div></div>')


def concept_html(title, body, tag=None, tag_kind=""):
    t = f'<span class="concept-tag {tag_kind}">{tag}</span><br>' if tag else ""
    return f'<div class="concept">{t}<div class="concept-title">{title}</div><div class="concept-body">{body}</div></div>'


def confusion_html(labels, matrix):
    """Heatmap confusion matrix. Warna diagonal = benar (indigo), di luar diagonal = salah (merah)."""
    head = "".join(f"<th>{c}</th>" for c in labels)
    rows = ""
    for i, row in enumerate(matrix):
        total = sum(row) or 1
        cells = ""
        for j, v in enumerate(row):
            frac = v / total
            alpha = 0.06 + 0.85 * frac
            base = "79,70,229" if i == j else "220,38,38"
            fg = "#fff" if alpha > 0.5 else "var(--ink)"
            cells += (f'<td style="background:rgba({base},{alpha:.2f}); color:{fg};">{v:,}'
                      f'<small>{frac*100:.1f}% baris</small></td>')
        rows += f'<tr><th class="rowh">{labels[i]}</th>{cells}</tr>'
    return h(f"""
        <div class="cm-axis">Prediksi model →</div>
        <div class="tbl-scroll"><table class="cm"><tr><th></th>{head}</tr>{rows}</table></div>
        <div class="cm-axis">↑ Label sebenarnya (tiap baris dijumlahkan = 100%)</div>
    """)


def _render_detail(item: dict):
    """Isi popup detail customer: semua kriteria yang diinput di halaman Prediction."""
    feats = item.get("features", {})
    is_flagged = item["result"] == "Fraudulent"
    badge_cls = "badge-table-fraud" if is_flagged else "badge-table-safe"
    icon = "⚠️" if is_flagged else "✅"
    info = CLASS_INFO.get(item.get("label", ""), CLASS_INFO["Legitimate"])

    def fmt(key, val):
        if key == "tracking_number_valid":
            return "Yes" if val else "No"
        if key == "return_rate_pct":
            return f"{float(val):.1f}%"
        if key in ("refund_amount_requested_usd", "avg_order_value_usd"):
            return f"${float(val):,.2f}"
        return f"{val:g}" if isinstance(val, float) else str(val)

    items = f'<div class="detail-item"><small>Account age</small><b>{item["account_age"]} days</b></div>'
    for key, label in FEATURE_LABELS.items():
        if key in feats:
            items += f'<div class="detail-item"><small>{label}</small><b>{fmt(key, feats[key])}</b></div>'

    probs_html = ""
    for cls, p in sorted(item.get("probs", {}).items(), key=lambda kv: kv[1], reverse=True):
        ci = CLASS_INFO.get(cls, {"color": "#64748b"})
        probs_html += bar_row(cls, p, f"{p*100:.1f}%", ci["color"])

    st.markdown(h(f"""
        <div class="detail-head">
            <div class="detail-case">{item["case"]}</div>
            <span class="{badge_cls}">{icon} {item.get("label", item["result"])} · {item.get("confidence", 0):.0f}%</span>
        </div>
        <div style="font-size:13px; margin-bottom:16px; opacity:.8;">{info["desc"]}</div>
        <div class="detail-section">Customer & return features</div>
        <div class="detail-grid">{items}</div>
        <div class="detail-section">Probabilitas tiap kelas</div>
        {probs_html}
    """), unsafe_allow_html=True)


# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown(h("""
        <div class="nav-brand">
            <div class="nav-logo">🔄</div>
            <div>
                <div class="nav-name">ReturnGuard</div>
                <div class="nav-tag">AI return-risk checker</div>
            </div>
        </div>
        <div class="nav-label">MENU</div>
    """), unsafe_allow_html=True)

    selected_menu = st.radio(
        "Navigation",
        ["🏠 Home", "📈 Dashboard", "✨ Prediction", "🔍 Customer Detail", "🧠 Model Insight"],
        index=0,
        label_visibility="collapsed",
    )

    _model_ok = model is not None
    st.markdown(h(f"""
        <div class="nav-foot">
            <div class="nav-foot-title">STATUS</div>
            <div class="nav-stat">
                <span><span class="nav-dot {'on' if _model_ok else 'off'}"></span>Model</span>
                <b>{'Loaded' if _model_ok else 'Missing'}</b>
            </div>
            <div class="nav-stat"><span>Analyses this session</span><b>{len(history)}</b></div>
        </div>
    """), unsafe_allow_html=True)

clean_menu = selected_menu.split(" ", 1)[1]

# ----------------- Statistik -----------------
total_analyses = len(history)
fraud_count = sum(1 for item in history if item["result"] == "Fraudulent")
non_fraud_count = total_analyses - fraud_count
fraud_rate = (fraud_count / total_analyses * 100) if total_analyses > 0 else 0.0
non_fraud_rate = (non_fraud_count / total_analyses * 100) if total_analyses > 0 else 0.0

# ======================= HOME =======================
if clean_menu == "Home":
    n_feat_model = model.n_features_in_ if model is not None else len(FEATURE_LABELS)
    model_name = type(model).__name__ if model is not None else "-"

    # ---- Hero ----
    st.markdown(h("""
        <div class="home-hero">
            <div class="home-badge">🔄 AI RETURN-RISK CHECKER</div>
            <div class="home-title">ReturnGuard</div>
            <div class="home-sub">
                Sistem berbasis machine learning untuk mendeteksi penyalahgunaan pengembalian barang
                (return abuse) pada e-commerce, sehingga transaksi mencurigakan bisa dikenali lebih cepat
                tanpa mengganggu pelanggan yang jujur.
            </div>
            <div class="home-cta">👉 Buka menu <b>Prediction</b> di sidebar untuk mencoba langsung.</div>
        </div>
    """), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- 3 kotak ringkasan ----
    tiles = [
        ("🗂️", "#eff6ff", "#3b82f6", "JUMLAH DATA LATIH", "60.000"),
        ("🧩", "#ede9fe", "#7c3aed", "FITUR INPUT MODEL", str(n_feat_model)),
        ("🏷️", "#fef3c7", "#d97706", "KELAS OUTPUT", str(len(CLASS_INFO))),
    ]
    for col, (icon, bg, fg, label, value) in zip(st.columns(3, gap="large"), tiles):
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

    # ---- Latar belakang & Tujuan ----
    bg_col, goal_col = st.columns(2, gap="large")

    with bg_col:
        st.markdown(h("""
            <div class="dashboard-card">
                <div class="home-h">📌 Latar Belakang & Masalah</div>
                <p class="home-p">
                    Platform e-commerce perlu menyediakan kebijakan pengembalian barang dan pengembalian dana
                    yang mudah untuk menjaga kepuasan pelanggan.
                </p>
                <p class="home-p">
                    Namun, ada pelanggan yang memanfaatkan kebijakan tersebut secara tidak semestinya,
                    misalnya dengan melakukan pengembalian berulang yang mencurigakan, mengajukan klaim
                    pengembalian dana palsu, atau bentuk penyalahgunaan lainnya.
                </p>
                <p class="home-p">
                    Memeriksa setiap transaksi secara manual itu lambat dan mahal, sedangkan kebijakan yang
                    terlalu ketat justru merugikan pelanggan yang jujur. ReturnGuard hadir sebagai alat bantu
                    untuk menyaring transaksi yang perlu diperiksa lebih lanjut.
                </p>
            </div>
        """), unsafe_allow_html=True)

    with goal_col:
        class_rows = "".join(
            f'<div style="display:flex; gap:10px; margin-bottom:10px;">'
            f'<span style="width:10px; height:10px; border-radius:50%; background:{v["color"]}; margin-top:5px; flex-shrink:0;"></span>'
            f'<div style="font-size:13px; line-height:1.5;"><b>{k}</b><br><span style="color:var(--muted);">{v["desc"]}</span></div></div>'
            for k, v in CLASS_INFO.items()
        )
        st.markdown(h(f"""
            <div class="dashboard-card">
                <div class="home-h">🎯 Tujuan</div>
                <p class="home-p">
                    Mengembangkan sistem yang dapat menganalisis transaksi pengembalian barang pada e-commerce
                    dan mengklasifikasikannya menjadi 2 kategori, yaitu
                    <span class="chip chip-ok">✅ WAJAR</span> dan
                    <span class="chip chip-bad">⚠️ TIDAK WAJAR</span>,
                    sehingga dapat membantu e-commerce mengidentifikasi transaksi mencurigakan.
                </p>
                <p class="home-p" style="margin-bottom:12px;">
                    Di dalam model, kategori <b>tidak wajar</b> dirinci lagi menjadi tiga jenis penyalahgunaan,
                    sehingga hasilnya lebih informatif:
                </p>
                {class_rows}
            </div>
        """), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Fitur aplikasi ----
    st.markdown("<div class='home-section'>Apa yang bisa dilakukan ReturnGuard?</div>", unsafe_allow_html=True)
    features_app = [
        ("📈", "Dashboard", "Ringkasan seluruh aktivitas prediksi: total analisis, rasio fraud vs non-fraud, tren return rate, dan riwayat terbaru."),
        ("✨", "Prediction", "Isi data kasus retur, lalu model memberi label, tingkat keyakinan (confidence), dan probabilitas tiap kelas. Input divalidasi otomatis agar tidak ada data yang mustahil."),
        ("🔍", "Customer Detail", "Cari kasus lewat kode unik (mis. RG-7K3QX9) untuk melihat ulang semua data input dan hasil analisisnya."),
        ("🧠", "Model Insight", "Transparansi model: cara training, pembagian data, performa, matriks kebingungan, feature importance, dan penjelasan konsep ML."),
    ]
    for col, (icon, title, desc) in zip(st.columns(4, gap="medium"), features_app):
        with col:
            st.markdown(h(f"""
                <div class="dashboard-card">
                    <div style="font-size:26px; margin-bottom:8px;">{icon}</div>
                    <div class="card-title" style="margin-bottom:6px;">{title}</div>
                    <div class="home-p" style="font-size:13px; margin:0;">{desc}</div>
                </div>
            """), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Dataset & Machine learning ----
    ds_col, ml_col = st.columns(2, gap="large")

    with ds_col:
        dist = [("Legitimate", 70.1), ("Policy Abuser", 11.9), ("Fraudulent Return", 10.2), ("Wardrobing", 7.7)]
        dist_rows = "".join(bar_row(n, p / 100, f"{p:.1f}%", CLASS_INFO[n]["color"]) for n, p in dist)
        st.markdown(h(f"""
            <div class="dashboard-card">
                <div class="home-h">🗂️ Dataset</div>
                <p class="home-p">
                    Model dilatih menggunakan
                    <a href="https://huggingface.co/datasets/sarveshchhetri/E-commerce_return_abuse_detection_dataset"
                       target="_blank" style="color:#6366f1; font-weight:600;">E-commerce Return Abuse Detection Dataset</a>
                    dari Hugging Face (lisensi Apache-2.0).
                </p>
                <ul class="home-list">
                    <li><b>60.000</b> transaksi retur e-commerce, tanpa missing value</li>
                    <li><b>35</b> kolom: demografi pelanggan, perilaku, pembayaran, pengiriman, dan alasan retur</li>
                    <li>Target: <b>abuse_type</b> (multi-class, tidak seimbang)</li>
                    <li>Data bersifat <b>sintetis</b> (dibuat untuk keperluan riset dan pembelajaran)</li>
                </ul>
                <div class="stat-label" style="margin:14px 0 10px 0;">Distribusi kelas pada dataset</div>
                {dist_rows}
            </div>
        """), unsafe_allow_html=True)

    with ml_col:
        feat_chips = "".join(f'<span class="chip">{lbl}</span>' for lbl in FEATURE_LABELS.values())
        st.markdown(h(f"""
            <div class="dashboard-card">
                <div class="home-h">🧠 Machine Learning</div>
                <ul class="home-list">
                    <li><b>Jenis:</b> Supervised Learning, <b>Classification</b> (multi-class)</li>
                    <li><b>Model:</b> {model_name}</li>
                    <li><b>Input:</b> {n_feat_model} fitur numerik per kasus retur</li>
                    <li><b>Output:</b> label kelas, confidence, dan probabilitas tiap kelas</li>
                    <li><b>Aturan keputusan:</b> kelas dengan probabilitas tertinggi menjadi hasil.
                        Jika bukan <i>Legitimate</i>, kasus ditandai <b>TIDAK WAJAR</b></li>
                </ul>
                <div class="stat-label" style="margin:14px 0 10px 0;">Fitur yang digunakan model</div>
                <div>{feat_chips}</div>
            </div>
        """), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Cara pembuatan ----
    st.markdown("<div class='home-section'>Bagaimana ReturnGuard dibuat?</div>", unsafe_allow_html=True)
    steps = [
        ("Pengumpulan data", "Mengambil dataset return abuse dari Hugging Face sebagai bahan latih."),
        ("Eksplorasi & pembersihan data", "Memeriksa distribusi kelas, nilai kosong, dan pola antar fitur (EDA)."),
        ("Pemilihan fitur", "Memilih fitur yang relevan dan bisa diisi operator, seperti return rate, jumlah dispute, dan validitas tracking."),
        ("Pelatihan model", "Melatih model klasifikasi multi-class untuk mengenali pola Legitimate, Fraudulent Return, Policy Abuser, dan Wardrobing."),
        ("Evaluasi", "Menguji model pada data yang tidak dipakai saat pelatihan untuk memastikan hasilnya layak."),
        ("Ekspor & integrasi", "Model disimpan sebagai file joblib lalu dipasang ke aplikasi web Streamlit ini."),
        ("Version control", "Seluruh kode dikelola dengan Git agar perubahan terlacak dan mudah dikerjakan bersama."),
    ]
    steps_html = "".join(
        f'<div class="step"><div class="step-num">{i}</div>'
        f'<div><div class="step-title">{t}</div><div class="step-desc">{d}</div></div></div>'
        for i, (t, d) in enumerate(steps, start=1)
    )
    st.markdown(h(f"""<div class="dashboard-card">{steps_html}</div>"""), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Tech stack (logo asli) ----
    st.markdown("<div class='home-section'>Tech stack</div>", unsafe_allow_html=True)
    DEV = "https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons"
    stack = [
        ("Python", "Bahasa pemrograman utama", f"{DEV}/python/python-original.svg"),
        ("Streamlit", "Framework antarmuka web", "https://cdn.simpleicons.org/streamlit/FF4B4B"),
        ("Git", "Version control", f"{DEV}/git/git-original.svg"),
        ("scikit-learn", "Pelatihan model ML", f"{DEV}/scikitlearn/scikitlearn-original.svg"),
        ("pandas", "Pengolahan data tabular", f"{DEV}/pandas/pandas-original.svg"),
        ("joblib", "Penyimpanan & pemuatan model", None),  # joblib tidak punya logo resmi di CDN ikon
    ]
    for col, (name, desc, logo) in zip(st.columns(6, gap="small"), stack):
        with col:
            if logo:
                logo_html = f'<img class="tech-logo" src="{logo}" alt="Logo {name}" loading="lazy">'
            else:
                logo_html = '<div class="tech-logo-fallback">jl</div>'
            st.markdown(h(f"""
                <div class="tech-card">
                    {logo_html}
                    <div class="tech-name">{name}</div>
                    <div class="tech-desc">{desc}</div>
                </div>
            """), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Catatan penggunaan ----
    st.markdown(h("""
        <div class="dashboard-card">
            <div class="home-h">⚠️ Catatan Penggunaan</div>
            <ul class="home-list">
                <li>Hasil model adalah <b>alat bantu keputusan</b>, bukan vonis. Kasus yang ditandai tetap perlu ditinjau manusia.</li>
                <li>Dataset bersifat sintetis, jadi performa pada data transaksi nyata perlu divalidasi ulang.</li>
                <li>Sekitar 70% data adalah kelas Legitimate (tidak seimbang), sehingga model bisa lebih sering salah pada kelas yang jarang muncul.</li>
                <li>Kasus yang salah ditandai (false positive) bisa merugikan pelanggan jujur, jadi gunakan hasilnya dengan bijak.</li>
            </ul>
        </div>
    """), unsafe_allow_html=True)


# ======================= DASHBOARD =======================
elif clean_menu == "Dashboard":
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
            grid += (f'<line class="rg-grid" x1="{PAD}" y1="{y}" x2="{W-14}" y2="{y}" stroke="#e2e8f0" stroke-width="1"/>'
                     f'<text class="rg-axis" x="{PAD-8}" y="{y+4}" font-size="10" fill="#94a3b8" text-anchor="end">{v}%</text>')

        thr_y = y_of(60)
        grid += (f'<line x1="{PAD}" y1="{thr_y}" x2="{W-14}" y2="{thr_y}" stroke="#f59e0b" stroke-width="1.2" stroke-dasharray="5 4"/>'
                 f'<text x="{W-16}" y="{thr_y-5}" font-size="10" fill="#d97706" text-anchor="end">Threshold 60%</text>')

        if total_analyses == 0:
            body = (f'<text class="rg-axis" x="{W/2}" y="{H/2}" font-size="13" fill="#94a3b8" text-anchor="middle">'
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
                body += f'<circle class="rg-dot" cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{c}" stroke="#fff" stroke-width="2"/>'

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
                                <span style="color:var(--ok-ink);">🟢 Non-Fraud<small>({non_fraud_count})</small></span>
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

        st.caption("Field di bawah dibatasi otomatis oleh field di atasnya "
                   "(mis. returns tidak bisa melebihi orders). Isi dari atas ke bawah.")

        def clamp_state(key, max_val):
            # Turunkan nilai tersimpan kalau melebihi batas baru.
            # Harus dipanggil SEBELUM widget-nya dibuat, kalau tidak Streamlit error.
            if key in st.session_state and st.session_state[key] > max_val:
                st.session_state[key] = max_val

        # --- Baris 1: tidak punya batas turunan ---
        c1, c2 = st.columns(2)
        with c1:
            account_age = st.number_input(
                "ACCOUNT AGE", min_value=0, step=1, key="in_account_age",
                help="days (hanya untuk riwayat, bukan fitur model)")
        with c2:
            total_orders = st.number_input(
                "TOTAL ORDERS", min_value=0, step=1, key="in_total_orders", help="orders")

        # --- Baris 2: returns <= orders ---
        max_returns = int(total_orders)
        clamp_state("in_total_returns", max_returns)
        c3, c4 = st.columns(2)
        with c3:
            total_returns = st.number_input(
                "TOTAL RETURNS", min_value=0, max_value=max_returns, step=1, key="in_total_returns",
                help=f"Maksimal {max_returns} (tidak boleh melebihi total orders)")
        with c4:
            avg_order_value = st.number_input(
                "AVG ORDER VALUE (USD)", min_value=0.0, step=1.0, key="in_avg_order_value", help="USD")

        # --- Baris 3: refund <= avg order value x total orders ; days <= account age ---
        max_refund = float(avg_order_value) * float(total_orders)
        clamp_state("in_refund_amount", max_refund)
        max_days = int(account_age)
        clamp_state("in_days_until_return", max_days)
        c5, c6 = st.columns(2)
        with c5:
            refund_amount = st.number_input(
                "REFUND AMOUNT (USD)", min_value=0.0, max_value=max_refund, step=1.0, key="in_refund_amount",
                help=f"Maksimal ${max_refund:,.2f} (avg order value × total orders)")
        with c6:
            days_until_return = st.number_input(
                "DAYS UNTIL RETURN", min_value=0, max_value=max_days, step=1, key="in_days_until_return",
                help=f"Maksimal {max_days} hari (tidak boleh melebihi umur akun)")

        # --- Baris 4: disputes <= orders ---
        max_disputes = int(total_orders)
        clamp_state("in_dispute_count", max_disputes)
        c7, c8 = st.columns(2)
        with c7:
            support_contacts = st.number_input(
                "CUSTOMER SUPPORT CONTACTS", min_value=0, step=1, key="in_support_contacts", help="contacts")
        with c8:
            dispute_count = st.number_input(
                "PREVIOUS DISPUTE COUNT", min_value=0, max_value=max_disputes, step=1, key="in_dispute_count",
                help=f"Maksimal {max_disputes} (tidak boleh melebihi total orders)")

        # --- Baris 5: wishlist time <= account age (dalam jam) ---
        max_wishlist = float(account_age) * 24.0
        clamp_state("in_wishlist_hrs", max_wishlist)
        c9, c10 = st.columns(2)
        with c9:
            wishlist_hrs = st.number_input(
                "WISHLIST TO CART TIME (HRS)", min_value=0.0, max_value=max_wishlist, step=0.5, key="in_wishlist_hrs",
                help=f"Maksimal {max_wishlist:,.0f} jam (umur akun × 24)")
        with c10:
            tracking_valid_label = st.selectbox("TRACKING NUMBER VALID", ["Yes", "No"], help="Apakah nomor resi valid?")
            tracking_valid = 1 if tracking_valid_label == "Yes" else 0

        # ---- Derived return rate ----
        return_rate = (total_returns / total_orders) * 100 if total_orders > 0 else 0.0
        is_high = return_rate >= 60.0
        if IS_DARK:
            r_color = "#f87171" if is_high else "#34d399"
            r_bg = "rgba(248,113,113,.15)" if is_high else "rgba(52,211,153,.15)"
        else:
            r_color = "#dc2626" if is_high else "#16a34a"
            r_bg = "rgba(220,38,38,.10)" if is_high else "rgba(22,163,74,.10)"
        r_status = "⚠️ HIGH" if is_high else "✅ NORMAL"

        st.markdown(h(f"""
            <div class="rate-label">Derived return rate</div>
            <div class="rate-card" style="--c:{r_color}; --bg:{r_bg};">
                <div class="rate-top">
                    <span class="rate-value">{return_rate:.1f}%</span>
                    <span class="rate-chip">{r_status}</span>
                </div>
                <div class="rate-bar-bg">
                    <div class="rate-bar-fill" style="width:{min(return_rate, 100):.1f}%;"></div>
                    <div class="rate-bar-mark"></div>
                </div>
                <div class="rate-foot">Total returns ÷ total orders · threshold 60%</div>
            </div>
        """), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- Tombol + cooldown ----
        remaining = max(0.0, COOLDOWN_SECONDS - (time.time() - st.session_state.last_predict_ts))
        btn_label = f"⏳ Tunggu {math.ceil(remaining)} detik..." if remaining > 0 else "🪄 Predict return behavior"

        clicked = st.button(btn_label, disabled=remaining > 0)

        if clicked and remaining <= 0:
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
                    prob_rows += bar_row(cls, p, f"{p*100:.1f}%", ci["color"])

                box_cls = "high-risk-box" if is_flagged else "low-risk-box"
                icon = "⚠️" if is_flagged else "✅"
                result_html = h(f"""
                    <div class="{box_cls}">
                        <div style="font-size:12px; font-weight:700; letter-spacing:.04em;">{icon} HASIL ANALISIS MODEL AI</div>
                        <div class="result-label">{label}</div>
                        <div style="font-size:13px; margin-bottom:14px;">Confidence <b>{confidence*100:.1f}%</b> — {info["desc"]}</div>
                        {prob_rows}
                    </div>
                """)

                new_code = generate_case_code(it["case"] for it in history)
                st.session_state.history_predictions.append({
                    "case": new_code,
                    "account_age": account_age,
                    "total_orders": total_orders,
                    "total_returns": total_returns,
                    "refund_amount": float(refund_amount),
                    "days_until_return": days_until_return,
                    "return_rate": round(return_rate, 1),
                    "label": label,
                    "confidence": round(confidence * 100, 1),
                    "result": "Fraudulent" if is_flagged else "Low Risk",
                    # data lengkap untuk popup detail customer
                    "features": features,
                    "probs": {str(k): float(v) for k, v in probs.items()},
                })

                # simpan hasil supaya tetap tampil saat countdown me-refresh halaman
                st.session_state.last_result = {"html": result_html, "code": new_code}
                st.session_state.last_predict_ts = time.time()
                st.rerun()

        # tampilkan hasil terakhir (tetap muncul selama & setelah cooldown)
        if st.session_state.last_result:
            st.markdown(st.session_state.last_result["html"], unsafe_allow_html=True)
            st.success(f"✨ Tersimpan dengan kode unik **{st.session_state.last_result['code']}**. "
                       "Cari kode ini di menu Customer Detail untuk melihat detailnya.")

    with col2:
        # Validasi DINAMIS sesuai input saat ini
        checks = [
            ("Orders > 0", total_orders > 0),
            ("Returns ≤ orders", total_returns <= total_orders),
            ("Refund ≤ total spend", refund_amount <= avg_order_value * total_orders),
            ("Disputes ≤ orders", dispute_count <= total_orders),
            ("Return days ≤ account age", days_until_return <= account_age),
            ("Wishlist time ≤ account age", wishlist_hrs <= account_age * 24),
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

    # countdown live: refresh tiap 1 detik sampai cooldown selesai
    if max(0.0, COOLDOWN_SECONDS - (time.time() - st.session_state.last_predict_ts)) > 0:
        time.sleep(1)
        st.rerun()

# ======================= CUSTOMER DETAIL =======================
elif clean_menu == "Customer Detail":
    st.markdown("<h2 style='margin-bottom: 0;'>🔍 Customer Detail</h2>", unsafe_allow_html=True)
    st.caption("Cari customer lewat kode uniknya untuk melihat semua kriteria yang diinput.")
    st.markdown("<br>", unsafe_allow_html=True)

    if total_analyses == 0:
        st.info("Belum ada data. Lakukan prediksi di menu Prediction terlebih dahulu.")
        st.stop()

    f_col, d_col = st.columns([1, 1.6], gap="large")

    with f_col:
        all_cases = [item["case"] for item in reversed(history)]
        # selectbox = kotak pencarian live: daftar kode menyaring otomatis tiap huruf diketik
        picked_case = st.selectbox(
            "🔎 Cari berdasarkan kode unik",
            all_cases,
            index=None,
            placeholder="Ketik kode, Contoh:RG-718",
            key="detail_case_select",
        )
        st.caption(f"{len(all_cases)} kasus tersimpan. Ketik sebagian kode, lalu klik kode yang muncul.")

    with d_col:
        if picked_case:
            _render_detail(next(it for it in history if it["case"] == picked_case))
        else:
            st.info("Ketik atau pilih kode unik di kolom pencarian untuk melihat detail customer.")

# ======================= MODEL INSIGHT =======================
elif clean_menu == "Model Insight":
    st.markdown("<h2 style='margin-bottom: 0;'>🧠 Model Insight</h2>", unsafe_allow_html=True)
    st.caption("Cara model dilatih, diuji, dan seberapa jauh hasilnya bisa dipercaya.")
    st.markdown("<br>", unsafe_allow_html=True)

    if model is None:
        st.error(model_error)
        st.stop()

    M = load_metrics()  # None kalau model_metrics.json belum dibuat
    class_names = [str(c) for c in model.classes_]

    if M is None:
        st.markdown(callout_html(
            "warn", "Metrik evaluasi belum tersedia",
            "Angka performa (accuracy, F1, confusion matrix, dll) tidak ditulis manual di kode supaya selalu "
            "sesuai dengan model aslinya. Jalankan <code>python evaluate_model.py</code> sekali di folder ini, "
            "lalu refresh halaman. Bagian yang tidak butuh metrik (hyperparameter, feature importance, "
            "penjelasan konsep) tetap tampil di bawah."), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    tab_sum, tab_train, tab_perf, tab_feat, tab_concept = st.tabs(
        ["Ringkasan", "Training & Split Data", "Performa", "Fitur", "Konsep ML"]
    )

    # ------------------------------------------------------------------ RINGKASAN
    with tab_sum:
        if M:
            o = M["overall"]
            k_cols = st.columns(4, gap="medium")
            kpis = [
                ("Accuracy (data uji)", pct(o["accuracy"]), f"Tebakan acak mayoritas: {pct(o['majority_baseline_accuracy'])}", "#6366f1"),
                ("Macro F1", pct(o["macro_f1"]), "Rata-rata F1 semua kelas, adil untuk kelas langka", "#7c3aed"),
                ("Balanced accuracy", pct(o["balanced_accuracy"]), "Rata-rata recall per kelas", "#0ea5e9"),
                ("ROC-AUC (OvR)", f"{o['roc_auc_ovr_macro']:.3f}", "1.0 = sempurna, 0.5 = sama dengan menebak", "#16a34a"),
            ]
            for col, (lbl, val, sub, colr) in zip(k_cols, kpis):
                with col:
                    st.markdown(kpi_html(lbl, val, sub, colr), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        s1, s2 = st.columns([1, 1.5], gap="large")
        with s1:
            class_rows = "".join(
                f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:10px; font-size:13px;">'
                f'<span class="dot" style="background:{CLASS_INFO.get(c, {"color": "#64748b"})["color"]}; margin:0;"></span>'
                f'<b>{c}</b></div>'
                for c in class_names
            )
            st.markdown(h(f"""
                <div class="dashboard-card">
                    <div class="card-title">Model summary</div>
                    <div class="card-sub" style="margin-bottom:16px;">{type(model).__name__}</div>
                    <div style="font-size:13px; margin-bottom:6px;">Jumlah fitur: <b>{model.n_features_in_}</b></div>
                    <div style="font-size:13px; margin-bottom:16px;">Jumlah pohon: <b>{getattr(model, "n_estimators", "-")}</b></div>
                    <div class="stat-label" style="margin-bottom:10px;">Kelas output</div>
                    {class_rows}
                </div>
            """), unsafe_allow_html=True)
        with s2:
            try:
                params = model.get_params()
            except Exception:
                params = {}
            param_html = "".join(
                f'<div class="param"><small>{k}</small><b>{v}</b></div>' for k, v in params.items()
            ) or '<div class="card-sub">Hyperparameter tidak tersedia.</div>'
            st.markdown(h(f"""
                <div class="dashboard-card">
                    <div class="card-title">Hyperparameter</div>
                    <div class="card-sub" style="margin-bottom:16px;">
                        Pengaturan yang dipilih <b>sebelum</b> training dan tidak dipelajari dari data
                        (contoh: jumlah pohon, kedalaman maksimum).
                    </div>
                    <div class="param-grid">{param_html}</div>
                </div>
            """), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        pipeline = [
            ("Data mentah", "60.000 transaksi retur, 35 kolom, target <code>abuse_type</code>."),
            ("Seleksi fitur", f"Dipilih {model.n_features_in_} fitur numerik yang bisa diisi operator. Kolom lain tidak dipakai."),
            ("Split data", "Data dibagi menjadi data latih dan data uji. Data uji disembunyikan dari model sampai evaluasi akhir."),
            ("Training", "Model mempelajari pola dari data latih: kombinasi fitur apa yang biasanya berujung pada tiap kelas."),
            ("Evaluasi", "Model menebak data uji. Tebakan dibandingkan dengan label asli untuk menghitung metrik."),
            ("Inference", "Di halaman Prediction, input operator diubah jadi satu baris fitur lalu diberikan ke <code>predict_proba</code>."),
        ]
        st.markdown(h("""<div class="dashboard-card"><div class="card-title" style="margin-bottom:6px;">Alur dari data sampai prediksi</div>""" +
                      "".join(f'<div class="step"><div class="step-num">{i}</div><div><div class="step-title">{t}</div><div class="step-desc">{d}</div></div></div>'
                               for i, (t, d) in enumerate(pipeline, 1)) + "</div>"), unsafe_allow_html=True)

    # ------------------------------------------------------------------ TRAINING & SPLIT
    with tab_train:
        if M:
            d = M["dataset"]
            n_train, n_test = d["n_train"], d["n_test"]
            tr_pct = n_train / (n_train + n_test) * 100
            te_pct = 100 - tr_pct
            st.markdown(h(f"""
                <div class="dashboard-card">
                    <div class="card-title">Pembagian data (train / test split)</div>
                    <div class="card-sub" style="margin-bottom:10px;">
                        {d["n_rows"]:,} baris dibagi dengan <code>test_size={d["test_size"]}</code>,
                        <code>random_state={d["random_state"]}</code>, stratified: <b>{"ya" if d.get("stratified") else "tidak"}</b>.
                    </div>
                    <div class="split-bar">
                        <div class="split-train" style="width:{tr_pct:.1f}%;">Data latih · {n_train:,} ({tr_pct:.0f}%)</div>
                        <div class="split-test" style="width:{te_pct:.1f}%;">Data uji · {n_test:,} ({te_pct:.0f}%)</div>
                    </div>
                </div>
            """), unsafe_allow_html=True)

            src = M.get("source", {}).get("mode")
            if src == "retrain_clone":
                st.markdown(callout_html(
                    "info", "Cara angka ini dihitung",
                    "Skor diambil dari model dengan hyperparameter yang sama, dilatih ulang pada porsi data latih, "
                    "lalu diuji pada data uji yang tidak pernah dilihatnya. Kalau skripnya dijalankan dengan "
                    "<code>--use-saved</code>, yang diuji adalah file joblib asli."), unsafe_allow_html=True)
            elif src == "saved_model":
                st.markdown(callout_html(
                    "warn", "Pastikan split-nya sama dengan saat training",
                    "Metrik dihitung dari file joblib asli. Angkanya baru jujur kalau <code>random_state</code> dan "
                    "<code>test_size</code> sama persis dengan yang dipakai di notebook training. Kalau beda, sebagian "
                    "data uji sebenarnya sudah pernah dilihat model."), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            ctr, cte = d["class_counts_train"], d["class_counts_test"]
            rows = ""
            for c in class_names:
                a, b = ctr.get(c, 0), cte.get(c, 0)
                col = CLASS_INFO.get(c, {"color": "#64748b"})["color"]
                rows += (f'<tr><td><span class="dot" style="background:{col};"></span>{c}</td>'
                         f'<td>{a:,}</td><td>{a/n_train*100:.1f}%</td><td>{b:,}</td><td>{b/n_test*100:.1f}%</td></tr>')
            st.markdown(h(f"""
                <div class="dashboard-card">
                    <div class="card-title">Distribusi kelas di data latih vs data uji</div>
                    <div class="card-sub" style="margin-bottom:14px;">
                        Persentase tiap kelas harus hampir sama di kedua sisi. Itulah gunanya stratified split.
                    </div>
                    <div class="tbl-scroll"><table class="tbl">
                        <tr><th>Kelas</th><th>Latih (jumlah)</th><th>Latih (%)</th><th>Uji (jumlah)</th><th>Uji (%)</th></tr>
                        {rows}
                    </table></div>
                </div>
            """), unsafe_allow_html=True)
        else:
            st.markdown(h("""
                <div class="dashboard-card">
                    <div class="card-title">Pembagian data (train / test split)</div>
                    <div class="card-sub">Jumlah baris dan distribusi kelas pasti akan muncul di sini setelah <code>evaluate_model.py</code> dijalankan.</div>
                </div>
            """), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        t1, t2 = st.columns(2, gap="large")
        with t1:
            st.markdown(concept_html(
                "Kenapa data dibagi?",
                "Kalau model diuji pakai data yang sama dengan data latihnya, nilainya hampir pasti bagus karena "
                "ia bisa saja sekadar menghafal. Data uji adalah \"ujian dengan soal baru\": model hanya boleh "
                "melihatnya sekali, di akhir. Skor pada data ini adalah perkiraan terbaik kita tentang performa "
                "pada kasus nyata yang belum pernah dilihat.", "Prinsip dasar"), unsafe_allow_html=True)
        with t2:
            st.markdown(concept_html(
                "Kenapa stratified?",
                "Dataset ini tidak seimbang: sekitar 70% Legitimate dan hanya sekitar 8% Wardrobing. Split acak biasa "
                "bisa saja menghasilkan data uji dengan proporsi kelas yang berbeda jauh dari data latih. "
                "<code>stratify=y</code> menjaga proporsi tiap kelas tetap sama di kedua sisi.", "Detail penting"), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        t3, t4 = st.columns(2, gap="large")
        with t3:
            st.markdown(concept_html(
                "Cross-validation",
                "Satu kali split bisa kebetulan mudah atau sulit. Cross-validation membagi data latih menjadi "
                "K bagian, lalu bergantian memakai satu bagian sebagai ujian dan sisanya untuk belajar. "
                "Rata-rata dan simpangan bakunya menunjukkan seberapa stabil modelnya. "
                "Hasilnya ada di tab Performa.", "Stabilitas"), unsafe_allow_html=True)
        with t4:
            st.markdown(concept_html(
                "Tuning hyperparameter jangan pakai data uji",
                "Kalau kamu mencoba banyak pengaturan lalu memilih yang skornya terbaik di data uji, data uji itu "
                "ikut \"terpakai\" dan skornya jadi terlalu optimis. Idealnya ada tiga bagian: "
                "<b>latih</b> untuk belajar, <b>validasi</b> (atau cross-validation) untuk memilih pengaturan, "
                "dan <b>uji</b> yang disentuh sekali saja di akhir.", "Sering terlewat", "warn"), unsafe_allow_html=True)

    # ------------------------------------------------------------------ PERFORMA
    with tab_perf:
        if not M:
            st.info("Jalankan evaluate_model.py untuk menampilkan metrik performa.")
        else:
            o = M["overall"]
            b = M["binary"]

            # ---- baseline & overfitting ----
            p1, p2 = st.columns(2, gap="large")
            with p1:
                gap = o["train_accuracy"] - o["accuracy"]
                st.markdown(h(f"""
                    <div class="dashboard-card">
                        <div class="card-title">Dibandingkan dengan tebakan paling malas</div>
                        <div class="card-sub" style="margin-bottom:14px;">
                            Kalau model selalu menjawab "Legitimate", accuracy-nya sudah setinggi persentase kelas itu.
                            Model hanya berguna kalau jauh di atas garis ini.
                        </div>
                        {bar_row("Selalu jawab kelas mayoritas", o["majority_baseline_accuracy"], pct(o["majority_baseline_accuracy"]), "#94a3b8")}
                        {bar_row("Model (accuracy data uji)", o["accuracy"], pct(o["accuracy"]), "#6366f1")}
                        {bar_row("Model (macro F1 data uji)", o["macro_f1"], pct(o["macro_f1"]), "#7c3aed")}
                    </div>
                """), unsafe_allow_html=True)
            with p2:
                tg = o["train_accuracy"] - o["accuracy"]
                if tg > 0.10:
                    verdict = callout_html("bad", "Indikasi overfitting", f"Selisih accuracy latih vs uji {tg*100:.1f} poin. Model terlalu menghafal data latih.")
                elif tg > 0.03:
                    verdict = callout_html("warn", "Sedikit gap", f"Selisih accuracy latih vs uji {tg*100:.1f} poin. Masih wajar, tapi perlu dipantau.")
                else:
                    verdict = callout_html("ok", "Tidak ada tanda overfitting", f"Selisih accuracy latih vs uji hanya {tg*100:.1f} poin.")
                cvr = M["cv"]
                st.markdown(h(f"""
                    <div class="dashboard-card">
                        <div class="card-title">Latih vs uji, dan stabilitas</div>
                        <div class="card-sub" style="margin-bottom:14px;">Model yang bagus di data latih tapi jelek di data uji berarti menghafal, bukan belajar.</div>
                        {bar_row("Accuracy data latih", o["train_accuracy"], pct(o["train_accuracy"]), "#0ea5e9")}
                        {bar_row("Accuracy data uji", o["accuracy"], pct(o["accuracy"]), "#6366f1")}
                        <div style="margin:12px 0;">{verdict}</div>
                        <div class="card-sub">
                            Cross-validation {cvr["folds"]}-fold (hanya di data latih):
                            accuracy <b>{pct(cvr["accuracy_mean"])}</b> ± {cvr["accuracy_std"]*100:.2f},
                            macro F1 <b>{pct(cvr["macro_f1_mean"])}</b> ± {cvr["macro_f1_std"]*100:.2f}.
                            Simpangan baku kecil berarti hasil stabil di berbagai potongan data.
                        </div>
                    </div>
                """), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ---- per kelas ----
            rows = ""
            for c in class_names:
                m = M["per_class"].get(c)
                if not m:
                    continue
                col = CLASS_INFO.get(c, {"color": "#64748b"})["color"]
                rows += (f'<tr><td><span class="dot" style="background:{col};"></span>{c}</td>'
                         f'<td>{pct(m["precision"])}</td><td>{pct(m["recall"])}</td><td>{pct(m["f1"])}</td>'
                         f'<td>{m["roc_auc"]:.3f}</td><td>{m["avg_precision"]:.3f}</td><td>{m["support"]:,}</td></tr>')
            st.markdown(h(f"""
                <div class="dashboard-card">
                    <div class="card-title">Performa per kelas</div>
                    <div class="card-sub" style="margin-bottom:14px;">
                        <b>Precision</b>: dari yang ditebak kelas ini, berapa yang benar.
                        <b>Recall</b>: dari semua kasus kelas ini yang asli, berapa yang berhasil ditangkap.
                        <b>F1</b>: gabungan keduanya. <b>Support</b>: jumlah kasus kelas itu di data uji.
                    </div>
                    <div class="tbl-scroll"><table class="tbl">
                        <tr><th>Kelas</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>Avg precision</th><th>Support</th></tr>
                        {rows}
                    </table></div>
                </div>
            """), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ---- confusion matrix ----
            cm_col, bin_col = st.columns([1.5, 1], gap="large")
            with cm_col:
                st.markdown(h(f"""
                    <div class="dashboard-card">
                        <div class="card-title">Confusion matrix</div>
                        <div class="card-sub" style="margin-bottom:14px;">
                            Diagonal (biru) = tebakan benar. Sel merah = salah tebak. Baca per baris untuk melihat
                            model salah mengira kelas apa jadi apa.
                        </div>
                        {confusion_html(M["confusion_matrix"]["labels"], M["confusion_matrix"]["matrix"])}
                    </div>
                """), unsafe_allow_html=True)
            with bin_col:
                st.markdown(h(f"""
                    <div class="dashboard-card">
                        <div class="card-title">Tampilan biner: wajar vs tidak wajar</div>
                        <div class="card-sub" style="margin-bottom:14px;">
                            Ini yang dilihat operator. Tiga kelas penyalahgunaan digabung jadi satu: "tidak wajar".
                        </div>
                        {bar_row("Precision (flag yang benar)", b["precision"], pct(b["precision"]), "#6366f1")}
                        {bar_row("Recall (abuse yang tertangkap)", b["recall"], pct(b["recall"]), "#16a34a")}
                        {bar_row("False positive rate", b["false_positive_rate"], pct(b["false_positive_rate"]), "#dc2626")}
                        {bar_row("False negative rate", b["false_negative_rate"], pct(b["false_negative_rate"]), "#d97706")}
                        <div class="card-sub" style="margin-top:12px;">
                            Dari <b>{b["tn"] + b["fp"]:,}</b> pelanggan jujur di data uji, <b>{b["fp"]:,}</b> salah ditandai.
                            Dari <b>{b["tp"] + b["fn"]:,}</b> kasus abuse, <b>{b["fn"]:,}</b> lolos.
                        </div>
                    </div>
                """), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            weakest = min(M["per_class"].items(), key=lambda kv: kv[1]["recall"])
            st.markdown(callout_html(
                "info", "Cara membaca hasil ini",
                f"Kelas dengan recall terendah adalah <b>{weakest[0]}</b> ({pct(weakest[1]['recall'])}). "
                "Pada data tidak seimbang, accuracy mudah terlihat tinggi walau kelas langka sering terlewat, "
                "jadi lihat recall dan F1 per kelas, bukan accuracy saja. Karena datanya sintetis, "
                "skor tinggi belum menjamin performa yang sama di data transaksi nyata."), unsafe_allow_html=True)

    # ------------------------------------------------------------------ FITUR
    with tab_feat:
        f1c, f2c = st.columns(2, gap="large")
        with f1c:
            if hasattr(model, "feature_importances_"):
                imp = sorted(zip(model.feature_names_in_, model.feature_importances_), key=lambda kv: kv[1], reverse=True)
                top = imp[0][1] if imp else 1
                imp_rows = "".join(
                    bar_row(FEATURE_LABELS.get(n, n), v / top, f"{v*100:.1f}%", "#6366f1") for n, v in imp
                )
            else:
                imp_rows = '<div class="card-sub">Model ini tidak punya feature_importances_.</div>'
            st.markdown(h(f"""
                <div class="dashboard-card">
                    <div class="card-title">Feature importance (bawaan model)</div>
                    <div class="card-sub" style="margin-bottom:16px;">
                        Dihitung dari seberapa banyak tiap fitur mengurangi ketidakmurnian (impurity) saat pohon
                        memecah data. Dihitung pada data latih.
                    </div>
                    {imp_rows}
                </div>
            """), unsafe_allow_html=True)
        with f2c:
            if M and M.get("permutation_importance"):
                pimp = sorted(M["permutation_importance"].items(), key=lambda kv: kv[1]["mean"], reverse=True)
                ptop = max(pimp[0][1]["mean"], 1e-9)
                prow = "".join(
                    bar_row(FEATURE_LABELS.get(n, n), max(v["mean"], 0) / ptop,
                            f'{v["mean"]*100:.2f} ± {v["std"]*100:.2f}', "#0ea5e9")
                    for n, v in pimp
                )
                st.markdown(h(f"""
                    <div class="dashboard-card">
                        <div class="card-title">Permutation importance (data uji)</div>
                        <div class="card-sub" style="margin-bottom:16px;">
                            Satu fitur diacak, lalu dilihat seberapa turun macro F1 model (dalam poin persen).
                            Makin besar penurunannya, makin model bergantung pada fitur itu.
                        </div>
                        {prow}
                    </div>
                """), unsafe_allow_html=True)
            else:
                st.markdown(h("""
                    <div class="dashboard-card">
                        <div class="card-title">Permutation importance (data uji)</div>
                        <div class="card-sub">Muncul setelah evaluate_model.py dijalankan. Metode ini lebih tepercaya daripada importance bawaan model.</div>
                    </div>
                """), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        g1, g2, g3 = st.columns(3, gap="medium")
        with g1:
            st.markdown(concept_html(
                "Importance bukan sebab-akibat",
                "Fitur yang penting artinya model memakainya untuk menebak, bukan berarti fitur itu "
                "<i>menyebabkan</i> penyalahgunaan. Jangan jadikan ini alasan menolak retur pelanggan.", "Hati-hati"), unsafe_allow_html=True)
        with g2:
            st.markdown(concept_html(
                "Impurity importance bisa bias",
                "Metode bawaan pohon cenderung menguntungkan fitur dengan banyak nilai unik (misalnya angka kontinu) "
                "dan dihitung pada data latih. Permutation importance di data uji lebih jujur.", "Sering terlewat", "warn"), unsafe_allow_html=True)
        with g3:
            st.markdown(concept_html(
                "Fitur yang saling berkaitan",
                "<code>return_rate_pct</code> dihitung dari <code>total_returns_lifetime</code> dibagi "
                "<code>total_orders_lifetime</code>. Fitur yang berkorelasi akan membagi \"jatah\" importance-nya, "
                "jadi satu fitur bisa tampak kurang penting padahal informasinya sama kuatnya.", "Korelasi"), unsafe_allow_html=True)

    # ------------------------------------------------------------------ KONSEP ML
    with tab_concept:
        st.markdown(callout_html(
            "info", "Kenapa tab ini ada",
            "Skor sebuah model hanya berarti kalau cara mengukurnya benar. Berikut konsep klasifikasi yang paling "
            "sering bikin hasil terlihat bagus padahal menyesatkan."), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        concepts = [
            ("Accuracy bisa menipu", "Pada data 70% Legitimate, model yang selalu menjawab Legitimate sudah 70% akurat dan nol berguna. Untuk data tidak seimbang, pakai precision, recall, F1, balanced accuracy, atau macro F1.", "Imbalance", "warn"),
            ("Precision vs recall", "Precision tinggi: sedikit pelanggan jujur yang salah ditandai. Recall tinggi: sedikit abuse yang lolos. Menaikkan yang satu biasanya menurunkan yang lain. Pilih keseimbangan sesuai biaya kesalahan di bisnismu.", "Trade-off", ""),
            ("Macro vs weighted average", "Macro F1 memperlakukan semua kelas setara, jadi kelas langka sama pentingnya. Weighted F1 dibobot jumlah data sehingga didominasi kelas besar. Untuk deteksi abuse, macro lebih jujur.", "Metrik", ""),
            ("Biaya kesalahan tidak sama", "Menolak retur pelanggan jujur (false positive) merusak kepercayaan. Meloloskan penipu (false negative) merugikan uang. Dua-duanya punya harga berbeda, dan itu menentukan seberapa agresif model boleh menandai.", "Bisnis", ""),
            ("Data leakage", "Terjadi saat informasi dari data uji, atau dari jawaban itu sendiri, bocor ke training. Contoh: split berbeda dari saat training, normalisasi dihitung sebelum split, atau ada fitur yang baru diketahui setelah keputusan dibuat. Skor jadi terlalu bagus dan runtuh di dunia nyata.", "Bahaya", "warn"),
            ("Overfitting & underfitting", "Overfitting: bagus di data latih, jelek di data uji (menghafal). Underfitting: jelek di keduanya (terlalu sederhana). Bandingkan skor latih vs uji untuk mendiagnosisnya.", "Diagnosis", ""),
            ("Probabilitas belum tentu terkalibrasi", "Confidence 90% tidak otomatis berarti benar 90 dari 100 kali. Random forest cenderung kurang percaya diri pada ekstrem. Kalau angka confidence akan dipakai untuk keputusan, periksa kalibrasi (calibration curve) dan pertimbangkan <code>CalibratedClassifierCV</code>.", "Confidence", "warn"),
            ("ROC-AUC dan PR-AUC", "ROC-AUC mengukur kemampuan model mengurutkan kasus positif di atas negatif tanpa bergantung pada ambang batas. Pada kelas sangat langka, average precision (PR-AUC) biasanya lebih sensitif dan lebih jujur.", "Metrik", ""),
            ("Ambang keputusan (threshold)", "Model memilih kelas dengan probabilitas tertinggi. Kamu bisa menggeser aturan itu, misalnya menandai kasus kalau total probabilitas non-Legitimate di atas 40%, untuk menukar precision dengan recall. Garis 60% di halaman Prediction hanya indikator return rate, bukan bagian dari model.", "Operasional", ""),
            ("Menangani kelas tidak seimbang", "Pilihannya: <code>class_weight='balanced'</code>, oversampling (SMOTE), undersampling, atau menggeser threshold. Evaluasi tetap harus pada data uji asli yang tidak diresampling.", "Teknik", ""),
            ("Data sintetis", "Dataset ini dibuat oleh generator, jadi pola antar fitur mungkin lebih bersih dan lebih mudah daripada data asli. Skor tinggi di sini adalah bukti bahwa pipeline-nya benar, bukan bukti bahwa model siap dipakai di toko sungguhan.", "Realita", "warn"),
            ("Setelah dipakai: monitoring", "Perilaku pelanggan dan penipu berubah (data drift). Model perlu dipantau, dievaluasi ulang dengan kasus nyata yang sudah ditinjau manusia, dan dilatih ulang secara berkala. Simpan juga <code>random_state</code> dan versi library supaya hasil bisa direproduksi.", "Produksi", ""),
        ]
        for i in range(0, len(concepts), 2):
            cc = st.columns(2, gap="large")
            for col, (t, body, tag, kind) in zip(cc, concepts[i:i + 2]):
                with col:
                    st.markdown(concept_html(t, body, tag, kind), unsafe_allow_html=True)
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)