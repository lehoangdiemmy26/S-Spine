# -*- coding: utf-8 -*-
"""Pipeline chạy phân tích: đọc ảnh -> bù máy nghiêng -> đo -> gộp nhiều ảnh."""
import numpy as np
import streamlit as st

from src.analysis.front import analyze_front, classify_front
from src.analysis.side import analyze_side, classify_side
from src.config.settings import (
    BLOCKING,
    MAX_PHOTOS,
    ROLL_MAX_FIX,
    ROLL_MIN_FIX,
    SPREAD_WARN,
)
from src.domain.levels import lower_side
from src.domain.models import Result
from src.vision.image_io import load_image
from src.vision.pose import detect
from src.vision.roll import estimate_roll, rotate_image


@st.cache_data(show_spinner="Đang phân tích ảnh...", ttl=300, max_entries=30)
def analyze_cached(kind, data, view, posture, auto_roll):
    img = load_image(data)
    lms, mask = detect(img)
    if lms is None:
        return None, ["Không nhận diện được người trong ảnh"]

    manual = isinstance(auto_roll, float) and not isinstance(auto_roll, bool)
    roll = float(auto_roll) if manual else estimate_roll(img, lms)
    fixed = False
    limit = 20.0 if manual else ROLL_MAX_FIX
    if roll is not None and auto_roll is not False and ROLL_MIN_FIX <= abs(roll) <= limit:
        rotated = rotate_image(img, roll)
        lms2, mask2 = detect(rotated)
        if lms2 is not None:
            img, lms, mask, fixed = rotated, lms2, mask2, True

    if kind == "front":
        res, bad = analyze_front(lms, img, view, mask)
    else:
        res, bad = analyze_side(lms, img, posture)
    if res is None:
        return None, bad

    res.metrics["may_nghieng_uoc_luong_do"] = None if roll is None else round(roll, 2)
    res.metrics["da_bu_may_nghieng"] = fixed
    if not fixed and not manual and roll is not None and abs(roll) > 1.5 and auto_roll is not False:
        res.notes.append(
            f"Máy có vẻ nghiêng ≈ {roll:.1f}° nhưng không bù được. "
            "Hãy đặt máy ngang bằng hoặc nhập tay góc nghiêng máy"
        )
    blocked = [n for n in res.notes if n.startswith(BLOCKING)]
    if blocked:
        return None, blocked + ["Ảnh chưa đạt chuẩn, vui lòng CHỤP LẠI."]
    if fixed:
        src = "nhập tay" if manual else "ước lượng từ các đường thẳng đứng trong nền"
        res.info.append(f"Đã bù máy nghiêng ≈ {roll:+.1f}° ({src}).")
    elif roll is None:
        res.info.append(
            "Không ước lượng được độ nghiêng của máy (nền ít đường thẳng đứng). "
            "Hãy đặt máy cố định, ngang bằng."
        )
    return res, []


def analyze_file(f, kind, view="back", posture="stand", auto_roll=True):
    return analyze_cached(kind, f.getvalue(), view, posture, auto_roll)


def merge_results(rs, kind):
    """Gộp nhiều ảnh cùng một người: lấy trung vị từng chỉ số."""
    notes = sorted({n for r in rs for n in r.notes})
    info = []
    m0 = rs[0].metrics
    if kind == "front":
        signed = [r.metrics["vai_phai_thap_hon_co_dau_do"] for r in rs]
        med = float(np.median(signed))
        trunks = [
            r.metrics["lech_than_so_voi_hong_do"]
            for r in rs
            if r.metrics["lech_than_so_voi_hong_do"] is not None
        ]
        trunk = float(np.median(trunks)) if trunks else None
        spread = max(signed) - min(signed)
        level, summary = classify_front(abs(med), lower_side(med), trunk)
        metrics = {
            "goc_lech_vai_do": round(abs(med), 2),
            "vai_thap_hon": lower_side(med),
            "vai_phai_thap_hon_co_dau_do": round(med, 2),
            "lech_than_so_voi_hong_do": None if trunk is None else round(trunk, 2),
            "huong_chup": m0["huong_chup"],
        }
    else:
        angs = [r.metrics["goc_tai_vai_do"] for r in rs]
        leans = [r.metrics["nghieng_than_do"] for r in rs]
        ang, lean = float(np.median(angs)), float(np.median(leans))
        spread = max(angs) - min(angs)
        level, summary = classify_side(ang, lean, "sit" if m0["tu_the"] == "ngồi" else "stand")
        metrics = {
            "goc_tai_vai_do": round(ang, 1),
            "nghieng_than_do": round(lean, 1),
            "huong_mat": m0["huong_mat"],
            "tu_the": m0["tu_the"],
        }
    metrics["so_anh"] = len(rs)
    metrics["dao_dong_do"] = round(spread, 2)
    info.append(f"Kết quả là trung vị của {len(rs)} ảnh (dao động giữa các ảnh ≈ {spread:.1f}°).")
    if spread > SPREAD_WARN[kind]:
        notes.append("Các ảnh cho kết quả chênh nhau nhiều: tư thế chưa ổn định hoặc ảnh kém tin cậy")
    return Result(rs[0].image, metrics, level, summary, notes, info)


def run_many(files, kind, view, posture, auto_roll):
    singles, bad = [], []
    for i, f in enumerate(files[:MAX_PHOTOS], 1):
        r, p = analyze_file(f, kind, view, posture, auto_roll)
        if r is None:
            bad += [f"Ảnh {i}: {x}" for x in p]
        else:
            singles.append(r)
    if not singles:
        return None, bad, []
    if len(singles) == 1:
        return singles[0], bad, singles
    return merge_results(singles, kind), bad, singles
