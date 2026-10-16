# -*- coding: utf-8 -*-
"""Theme + header chung. CSS giữ nguyên bản cũ cho khỏi lệch giao diện."""
import base64
import io

import streamlit as st
from PIL import Image

from src.ui.artwork import one_line

CSS_HOME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;800&display=swap');
:root { --navy:#0B1F3A; --navy2:#12305A; --blue:#2B7BBF; --sky:#E8F3FB; --amber:#FFB547; --ink:#10243E; --mute:#4A5D75; --line:#D3E2F0; }
.block-container { max-width: 1120px; padding-top: 3.5rem; }
@media (max-width: 640px) { .block-container { padding-top: 3rem; } }
html { scroll-behavior:smooth; }
.stApp button:focus-visible, .stApp a:focus-visible, .stApp input:focus-visible { outline:3px solid var(--blue); outline-offset:2px; }
.stMarkdown, .stMarkdown p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, [data-baseweb="tab"] p,
.ss-top, .ss-hero, .ss-sec, .ss-ky, .ss-modes { font-family: 'Be Vietnam Pro', 'Segoe UI', system-ui, sans-serif; }
[data-baseweb="tab"] p { font-weight: 600; }
[data-baseweb="tab-highlight"] { background-color: var(--navy) !important; }

.ss-top { display:flex; align-items:center; gap:16px; margin-bottom:20px; padding:14px 18px; background:linear-gradient(180deg,#F8FBFE 0%,#FFFFFF 100%); border:1px solid var(--line); border-radius:18px; }
.ss-logo { width:60px; height:60px; border-radius:50%; flex:none; box-shadow:0 0 0 3px #fff, 0 0 0 5px var(--amber); transition:transform .25s ease, box-shadow .25s ease; }
.ss-top:hover .ss-logo { transform:scale(1.06) rotate(-3deg); box-shadow:0 0 0 3px #fff, 0 0 0 5px var(--blue), 0 8px 20px rgba(11,31,58,.18); }
.ss-name { font-size:clamp(2rem, 1.4rem + 2.2vw, 2.7rem); font-weight:800; color:var(--navy); line-height:1.05; letter-spacing:-.02em; text-shadow:0 1px 0 #fff; transition:color .2s ease, letter-spacing .25s ease; }
.ss-top:hover .ss-name { color:var(--navy2); letter-spacing:-.01em; }
.ss-brand-sub { font-size:1rem; color:var(--navy2); font-weight:600; margin-top:4px; opacity:1; }
.ss-brand-sub::after { content:""; display:block; width:64px; height:3px; border-radius:999px; background:var(--amber); margin-top:6px; transition:width .25s ease; }
.ss-top:hover .ss-brand-sub::after { width:110px; }

.ss-hero { display:grid; grid-template-columns: 1.15fr .85fr; gap:8px; align-items:center; background:var(--navy);
  border-radius:24px; padding:44px 44px 36px; margin:6px 0 40px; overflow:hidden; }
.ss-hero .ss-h { color:#fff; font-size:clamp(1.8rem, 3.6vw, 2.9rem); font-weight:800; line-height:1.14; letter-spacing:-.01em; max-width:15em; }
.ss-hero p { color:#B9CCE3; font-size:1.05rem; line-height:1.65; max-width:34em; margin:18px 0 26px; }
.ss-hint { display:inline-block; border:1.5px solid var(--amber); color:var(--amber); border-radius:999px; padding:8px 18px; font-weight:600; font-size:.92rem; }
.ss-hero .art { width:100%; max-height:440px; }
.ss-hero .v { fill:#6FB8EA; opacity:0; animation: ssIn .5s ease forwards; }
.ss-hero .lm { opacity:0; animation: ssIn .6s ease 1.3s forwards; }
.ss-hero .tx { fill:#fff; font:600 14px 'Be Vietnam Pro', sans-serif; }
.ss-hero .tx.s { fill:var(--amber); font-size:12px; }
@keyframes ssIn { to { opacity:1; } }
@media (prefers-reduced-motion: reduce) { .ss-hero .v, .ss-hero .lm { animation:none; opacity:1; } }

.ss-sec { margin: 0 0 40px; }
.ss-sec .ss-st { font-size:1.45rem; font-weight:800; color:var(--navy); margin-bottom:18px; }
.ss-steps { list-style:none; margin:0; padding:0; display:grid; grid-template-columns:repeat(4, 1fr); gap:16px; }
.ss-steps li { position:relative; background:#fff; border:1px solid var(--line); border-radius:14px; padding:20px 18px 18px; color:var(--ink); font-size:.93rem; line-height:1.6; }
.ss-steps .n { display:flex; align-items:center; justify-content:center; width:34px; height:34px; border-radius:50%; background:var(--navy); color:#fff; font-weight:800; margin-bottom:12px; }
.ss-steps b { display:block; font-size:1.05rem; color:var(--navy); margin-bottom:6px; }
.ss-warn { margin-top:16px; color:var(--mute); font-size:.92rem; }

.ss-ky { display:grid; grid-template-columns:.8fr 1.2fr; gap:36px; align-items:center; background:var(--sky); border-radius:24px; padding:34px 40px; margin-bottom:40px; }
.ss-ky .art { width:100%; max-height:400px; }
.ss-ky .v { fill:#8FBDE0; }
.ss-ky .tk { fill:var(--navy); font:700 15px 'Be Vietnam Pro', sans-serif; }
.ss-ky .ss-st { font-size:1.45rem; font-weight:800; color:var(--navy); margin-bottom:12px; }
.ss-ky p { color:var(--ink); line-height:1.7; margin:0 0 12px; }
.ss-note { border-left:4px solid var(--amber); background:#fff; border-radius:0 10px 10px 0; padding:10px 14px; color:var(--ink); font-size:.92rem; }

.ss-modes { list-style:none; margin:0; padding:0; display:grid; grid-template-columns:repeat(auto-fit, minmax(190px, 1fr)); gap:12px; }
.ss-modes li { border:1px solid var(--line); border-radius:12px; padding:14px 16px; color:var(--mute); font-size:.9rem; line-height:1.5; }
.ss-modes b { display:block; color:var(--navy); font-size:.98rem; margin-bottom:4px; }

.disclaimer-box { background:#FFF8E6; border:1px solid #F3DDA0; padding:12px 16px; border-radius:12px; color:#7A4B07; font-size:.9rem; margin-top:30px; text-align:center; }
[data-testid="stFileUploadDropzone"] { background-color:#F8FAFC !important; border:2px dashed #38BDF8 !important; border-radius:12px !important; }
[data-testid="stMetricValue"] { color:#0F172A !important; font-weight:700 !important; }
.exercise-card { background-color:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:15px; margin-bottom:15px; }
.exercise-title { color:#0369A1; font-weight:700; font-size:1.1rem; margin-bottom:8px; }
.stApp { background-color:#FFFFFF; color:#1A202C; }


/* ---- tiêu đề trang con, nhãn nhỏ ---- */
.ss-eyebrow { font-family:'Be Vietnam Pro','Segoe UI',sans-serif; font-size:.78rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:var(--blue); margin:26px 0 8px; }
.ss-head { margin:6px 0 4px; font-family:'Be Vietnam Pro','Segoe UI',sans-serif; }
.ss-head .t { font-size:1.55rem; font-weight:800; color:var(--navy); line-height:1.2; }
.ss-head .d { color:var(--mute); margin-top:4px; font-size:.97rem; }

/* ---- nút chọn dạng viên thuốc cho mọi st.radio (chỉ áp dụng khi trình duyệt hỗ trợ :has) ---- */
@supports selector(:has(*)) {
  div[role="radiogroup"] { gap:8px; flex-wrap:wrap; }
  div[role="radiogroup"] > label { background:#fff; border:1px solid var(--line); border-radius:999px; padding:10px 18px; margin:0; cursor:pointer; transition:background .15s, border-color .15s; min-height:44px; display:inline-flex; align-items:center; }
  div[role="radiogroup"] > label:focus-within { outline:3px solid var(--blue); outline-offset:2px; }
  div[role="radiogroup"] > label > div:first-child { display:none; }
  div[role="radiogroup"] > label:hover { border-color:var(--blue); }
  div[role="radiogroup"] > label:has(input:checked) { background:var(--navy); border-color:var(--navy); }
  div[role="radiogroup"] > label:has(input:checked) p { color:#fff !important; }
  div[role="radiogroup"] > label p { font-weight:600; font-size:.93rem; }
}
/* Fallback khi trình duyệt không hỗ trợ :has: giữ radio gốc dễ bấm */
@supports not selector(:has(*)) {
  div[role="radiogroup"] label { min-height:44px; }
}

/* ---- thẻ chỉ số, thông báo, nút ---- */
[data-testid="stMetric"] { background:#F4F8FC; border:1px solid var(--line); border-radius:14px; padding:14px 16px; min-height:96px; }
[data-testid="stAlert"] { border-radius:12px; }
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:12px; }
.stButton > button, .stDownloadButton > button { border-radius:999px; font-weight:600; min-height:44px; }
.stButton > button:hover, .stDownloadButton > button:hover { filter:brightness(.97); }

/* ---- P0: badge độ tin cậy + thẻ kết quả gọn ---- */
.ss-badge { display:inline-block; border-radius:999px; padding:4px 12px; font-size:.8rem; font-weight:700; }
.ss-badge-ok { background:#E6F4EA; color:#137333; border:1px solid #B7DFC0; }
.ss-badge-warn { background:#FEF3C7; color:#92400E; border:1px solid #FCD34D; }
.ss-badge-bad { background:#FEE2E2; color:#991B1B; border:1px solid #FCA5A5; }
.ss-result { border:1px solid var(--line); border-radius:16px; padding:16px 18px; background:#fff; margin:12px 0 16px; }
.ss-quality { color:var(--mute); font-size:.9rem; line-height:1.6; }
.ss-next { background:#F4F8FC; border:1px dashed var(--line); border-radius:12px; padding:10px 14px; color:var(--ink); font-size:.9rem; }

/* ---- tab Phương pháp & giới hạn (v2) ---- */
.ss-method-hero { background:linear-gradient(135deg, var(--navy) 0%, var(--navy2) 100%); border-radius:20px; padding:28px 30px; color:#fff; margin:6px 0 22px; }
.ss-method-hero .t { font-size:1.35rem; font-weight:800; line-height:1.3; }
.ss-method-hero .d { color:#B9CCE3; margin-top:8px; font-size:.95rem; line-height:1.65; max-width:62em; }
.ss-method-hero .row { display:flex; gap:10px; flex-wrap:wrap; margin-top:16px; }
.ss-pipeline { display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; margin-top:16px; }
.ss-pipeline .step { background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.16); border-radius:14px; padding:14px 16px; }
.ss-pipeline .n { display:inline-flex; align-items:center; justify-content:center; width:28px; height:28px; border-radius:50%; background:var(--amber); color:var(--navy); font-weight:800; font-size:.85rem; margin-bottom:8px; }
.ss-pipeline b { display:block; font-size:.95rem; margin-bottom:4px; }
.ss-pipeline span { color:#B9CCE3; font-size:.85rem; line-height:1.55; }
.ss-cards { display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:14px; }
.ss-card { border:1px solid var(--line); border-radius:14px; padding:18px; background:#fff; color:var(--ink); font-size:.92rem; line-height:1.6; }
.ss-card:hover { border-color:#9CC3E5; box-shadow:0 2px 12px rgba(43,123,191,.10); }
.ss-card b { display:block; color:var(--navy); font-size:1.02rem; margin-bottom:4px; }
.ss-card .ss-ico { display:inline-flex; align-items:center; justify-content:center; width:34px; height:34px; border-radius:10px; background:var(--sky); color:var(--navy); font-weight:800; margin-bottom:10px; }
.ss-card p { margin:0; }
.ss-tag { display:inline-block; background:var(--sky); color:var(--blue); border-radius:999px; padding:2px 10px; font-size:.75rem; font-weight:600; margin-bottom:8px; }
.ss-card.warn { background:#FFF8E6; border-color:#F3DDA0; grid-column:1 / -1; }
.ss-legend { display:flex; gap:8px; flex-wrap:wrap; margin:0 0 12px; }
.ss-pill { display:inline-flex; align-items:center; gap:6px; border-radius:999px; padding:5px 12px; font-size:.8rem; font-weight:700; border:1px solid; }
.ss-pill-green { background:#E6F4EA; color:#137333; border-color:#B7DFC0; }
.ss-pill-yellow { background:#FEF3C7; color:#92400E; border-color:#FCD34D; }
.ss-pill-red { background:#FEE2E2; color:#991B1B; border-color:#FCA5A5; }
.ss-table { width:100%; border-collapse:separate; border-spacing:0; font-size:.92rem; color:var(--ink); background:#fff; border:1px solid var(--line); border-radius:14px; overflow:hidden; }
.ss-table th { text-align:left; color:#fff; font-weight:700; font-size:.8rem; padding:10px 12px; background:var(--navy); }
.ss-table td { padding:10px 12px; border-bottom:1px solid var(--line); vertical-align:top; }
.ss-table tbody tr:nth-child(even) td { background:#F8FBFE; }
.ss-table tbody tr:hover td { background:#EFF6FC; }
.ss-table tbody tr:last-child td { border-bottom:none; }
.ss-table td.num { white-space:nowrap; font-variant-numeric:tabular-nums; }
.ss-dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }
.ss-limits { list-style:none; margin:0; padding:0; display:grid; gap:10px; }
.ss-limits li { border-left:4px solid var(--amber); background:#FFFCF3; border-radius:0 10px 10px 0; padding:12px 16px; color:var(--ink); font-size:.93rem; line-height:1.6; }
.ss-limits li b { color:var(--navy); }
.ss-box { background:var(--sky); border-radius:14px; padding:14px 18px; color:var(--ink); font-size:.93rem; line-height:1.6; margin-bottom:14px; }
.ss-table-wrap { overflow-x:auto; border-radius:14px; }
.ss-method-cta { background:#FFF8E6; border:1px solid #F3DDA0; border-radius:14px; padding:14px 18px; color:#7A4B07; font-size:.92rem; line-height:1.6; margin-top:16px; transition:box-shadow .2s ease, transform .2s ease; }
.ss-method-cta:hover { box-shadow:0 4px 16px rgba(224,138,0,.18); transform:translateY(-1px); }

/* ---- hover chung ---- */
a { transition:color .18s ease; }
a:hover { color:var(--navy2); }
.stButton > button, .stDownloadButton > button { transition:transform .18s ease, box-shadow .18s ease, filter .18s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { transform:translateY(-1px); box-shadow:0 6px 18px rgba(11,31,58,.16); }
.stButton > button:active, .stDownloadButton > button:active { transform:translateY(0); }
[data-testid="stMetric"] { transition:transform .2s ease, box-shadow .2s ease, border-color .2s ease; }
[data-testid="stMetric"]:hover { transform:translateY(-2px); border-color:#9CC3E5; box-shadow:0 6px 18px rgba(43,123,191,.14); }
[data-testid="stExpander"] { transition:border-color .2s ease, box-shadow .2s ease; }
[data-testid="stExpander"]:hover { border-color:#9CC3E5; box-shadow:0 4px 14px rgba(43,123,191,.10); }
[data-testid="stFileUploadDropzone"] { transition:border-color .2s ease, background-color .2s ease, box-shadow .2s ease; }
[data-testid="stFileUploadDropzone"]:hover { border-color:var(--blue) !important; background-color:#EFF6FC !important; box-shadow:0 4px 14px rgba(43,123,191,.12); }
[data-baseweb="tab"] { transition:background-color .18s ease; border-radius:10px 10px 0 0; }
[data-baseweb="tab"]:hover { background-color:#EFF6FC; }
.ss-steps li, .ss-modes li, .exercise-card { transition:transform .2s ease, box-shadow .2s ease, border-color .2s ease; }
.ss-steps li:hover, .ss-modes li:hover, .exercise-card:hover { transform:translateY(-3px); border-color:#9CC3E5; box-shadow:0 8px 22px rgba(11,31,58,.12); }
.ss-steps li:hover .n { background:var(--blue); }
.ss-ky { transition:box-shadow .22s ease; }
.ss-ky:hover { box-shadow:0 8px 26px rgba(43,123,191,.14); }
.ss-hero .ss-hint { transition:background-color .2s ease, color .2s ease, transform .2s ease; }
.ss-hero .ss-hint:hover { background:var(--amber); color:var(--navy); transform:translateY(-1px); }
.ss-pipeline .step { transition:transform .2s ease, background-color .2s ease; }
.ss-pipeline .step:hover { transform:translateY(-2px); background:rgba(255,255,255,.14); }
.ss-badge, .ss-pill { transition:transform .18s ease, box-shadow .18s ease; }
.ss-badge:hover, .ss-pill:hover { transform:scale(1.04); box-shadow:0 3px 10px rgba(11,31,58,.14); }
.ss-fig { margin:0 0 12px; }
.ss-fig img { display:block; width:100%; height:auto; border-radius:12px; }
.ss-figcap { text-align:center; color:var(--mute); font-size:.85rem; margin-top:6px; }
.stImage img { transition:transform .25s ease, box-shadow .25s ease; }
.stImage img:hover { transform:scale(1.01); box-shadow:0 8px 22px rgba(11,31,58,.16); }

/* ---- scroll-linked animations (dùng chung toàn dự án) ---- */
.ss-scroll-progress { position:sticky; top:0; z-index:5; height:4px; border-radius:999px;
  background:linear-gradient(90deg, var(--amber), var(--blue)); transform-origin:0 50%; transform:scaleX(0); }
@keyframes ssRise { from { opacity:0; transform:translateY(72px) scale(.96); } to { opacity:1; transform:none; } }
@keyframes ssLeft { from { opacity:0; transform:translateX(-84px) rotate(-.6deg); } to { opacity:1; transform:none; } }
@keyframes ssRight { from { opacity:0; transform:translateX(84px) rotate(.6deg); } to { opacity:1; transform:none; } }
@keyframes ssZoom { from { opacity:0; transform:translateY(36px) scale(.82); } to { opacity:1; transform:scale(1); } }
@keyframes ssParallax { from { transform:translateY(-42px) scale(1.03); } to { transform:translateY(52px) scale(1.03); } }
@keyframes ssGrow { from { transform:scaleX(0); } to { transform:scaleX(1); } }
/* Mặc định: hiện bình thường để không trắng trang khi trình duyệt cũ */
.ss-reveal, .ss-reveal-left, .ss-reveal-right, .ss-reveal-zoom { opacity:1; }
@supports (animation-timeline: view()) {
  .ss-scroll-progress { animation:ssGrow linear both; animation-timeline:scroll(); }
  .ss-reveal { opacity:0; animation:ssRise .9s cubic-bezier(.22,.8,.28,1) both; animation-timeline:view(); animation-range:entry 0% cover 45%; }
  .ss-reveal-left { opacity:0; animation:ssLeft .9s cubic-bezier(.22,.8,.28,1) both; animation-timeline:view(); animation-range:entry 0% cover 45%; }
  .ss-reveal-right { opacity:0; animation:ssRight .9s cubic-bezier(.22,.8,.28,1) both; animation-timeline:view(); animation-range:entry 0% cover 45%; }
  .ss-reveal-zoom { opacity:0; animation:ssZoom .9s cubic-bezier(.22,.8,.28,1) both; animation-timeline:view(); animation-range:entry 0% cover 40%; }
  .ss-hero .art { animation:ssParallax linear both; animation-timeline:view(); animation-range:exit 0% exit 100%; }
  .ss-steps li:nth-child(1), .ss-modes li:nth-child(1) { animation-range:entry 0% cover 35%; }
  .ss-steps li:nth-child(2), .ss-modes li:nth-child(2) { animation-range:entry 5% cover 42%; }
  .ss-steps li:nth-child(3), .ss-modes li:nth-child(3) { animation-range:entry 10% cover 48%; }
  .ss-steps li:nth-child(4) { animation-range:entry 15% cover 52%; }
  /* Áp dụng chung: hero phương pháp, bảng, giới hạn, bài tập, thẻ kết quả */
  .ss-method-hero { opacity:0; animation:ssZoom .9s cubic-bezier(.22,.8,.28,1) both; animation-timeline:view(); animation-range:entry 0% cover 40%; }
  .ss-method-hero .step:nth-child(1) { animation-range:entry 0% cover 35%; }
  .ss-method-hero .step:nth-child(2) { animation-range:entry 5% cover 42%; }
  .ss-method-hero .step:nth-child(3) { animation-range:entry 10% cover 48%; }
  .ss-cards .ss-card { opacity:0; animation:ssRise .9s cubic-bezier(.22,.8,.28,1) both; animation-timeline:view(); animation-range:entry 0% cover 42%; }
  .ss-table-wrap { opacity:0; animation:ssRise .9s cubic-bezier(.22,.8,.28,1) both; animation-timeline:view(); animation-range:entry 0% cover 40%; }
  .ss-limits li { opacity:0; animation:ssLeft .9s cubic-bezier(.22,.8,.28,1) both; animation-timeline:view(); animation-range:entry 0% cover 38%; }
  .ss-limits li:nth-child(even) { animation-name:ssRight; }
  .ss-method-cta, .ss-box, .exercise-card, .ss-next { opacity:0; animation:ssRise .9s cubic-bezier(.22,.8,.28,1) both; animation-timeline:view(); animation-range:entry 0% cover 38%; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation:none !important; transition:none !important; }
  html { scroll-behavior:auto; }
}

@media (max-width: 860px) {
  .ss-hero { grid-template-columns:1fr; padding:30px 24px 20px; }
  .ss-hero .art { max-height:300px; }
  .ss-steps { grid-template-columns:1fr 1fr; }
  .ss-ky { grid-template-columns:1fr; padding:26px 22px; }
  .ss-ky .art { max-height:300px; }
  .ss-pipeline { grid-template-columns:1fr; }
  .ss-method-hero { padding:22px 20px; }
}
@media (max-width: 520px) { .ss-steps { grid-template-columns:1fr; } }
</style>
"""


def load_logo():
    for name in [
        "gen-n-z8308486911094_3cf6e9f66d814eabd93c0c5ae610e055-modified.png",
        "gen-n-z8308486911094_3cf6e9f66d814eabd93c0c5ae610e055-modified.jpg",
    ]:
        try:
            return Image.open(name)
        except Exception:
            continue
    return None


def logo_uri(logo):
    if logo is None:
        return None
    im = logo.convert("RGBA")
    im.thumbnail((120, 120))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def eyebrow(text):
    st.markdown(f'<div class="ss-eyebrow">{text}</div>', unsafe_allow_html=True)


def page_head(title, desc):
    st.markdown(
        one_line(f'<div class="ss-head"><div class="t">{title}</div><div class="d">{desc}</div></div>'),
        unsafe_allow_html=True,
    )


def header_html(logo):
    uri = logo_uri(logo)
    img = f'<img class="ss-logo" src="{uri}" alt="Logo S-Spine"/>' if uri else ""
    return one_line(
        f'<div class="ss-top">{img}<div><div class="ss-name">S-Spine</div>'
        f'<div class="ss-brand-sub">AI đánh giá và phân tích tư thế học đường</div></div></div>'
    )
