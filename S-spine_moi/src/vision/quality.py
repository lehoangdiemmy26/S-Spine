# -*- coding: utf-8 -*-
"""Kiểm tra mốc + chất lượng ảnh (tối, mờ, phân giải, người quá nhỏ...)."""
import cv2
import numpy as np

from src.config.landmarks import LM_NAMES
from src.config.settings import (
    BLUR_P99_MIN,
    DARK_MIN,
    PERSON_MIN_FRAC,
    SYM_BLOCK,
    SYM_WARN,
    VIS_MIN,
)


def check_points(lms, idxs):
    out = []
    for i in idxs:
        p = lms[i]
        if not (0.0 <= p.x <= 1.0 and 0.0 <= p.y <= 1.0):
            out.append(f"Mốc {LM_NAMES[i]} nằm ngoài khung ảnh (ảnh bị cắt)")
        elif p.visibility < VIS_MIN:
            out.append(f"Mốc {LM_NAMES[i]} bị che hoặc không rõ")
    return out


def person_roi(img, lms, margin=0.06):
    """Hộp bao quanh người theo các mốc thấy rõ; không đủ mốc thì lấy cả ảnh."""
    h, w = img.shape[:2]
    pts = [
        (p.x, p.y) for p in lms
        if p.visibility >= 0.5 and 0.0 <= p.x <= 1.0 and 0.0 <= p.y <= 1.0
    ]
    if len(pts) < 4:
        return img
    xs, ys = zip(*pts)
    x0 = int(max(0, min(xs) - margin) * w)
    x1 = int(min(1, max(xs) + margin) * w)
    y0 = int(max(0, min(ys) - margin) * h)
    y1 = int(min(1, max(ys) + margin) * h)
    if x1 - x0 < 40 or y1 - y0 < 40:
        return img
    return img[y0:y1, x0:x1]


def sharpness_p99(gray):
    """Độ nét: p99 của |Laplacian| sau khi đưa vùng ảnh về cao 640px."""
    h, w = gray.shape
    k = 640.0 / h
    g = cv2.resize(gray, (max(1, int(w * k)), 640), interpolation=cv2.INTER_AREA)
    return float(np.percentile(np.abs(cv2.Laplacian(g, cv2.CV_64F)), 99))


def image_notes(img, lms):
    notes = []
    h, w = img.shape[:2]
    roi = cv2.cvtColor(person_roi(img, lms), cv2.COLOR_RGB2GRAY)
    if roi.mean() < DARK_MIN:
        notes.append("Ảnh khá tối (vùng có người), hãy chụp nơi đủ sáng")
    if sharpness_p99(roi) < BLUR_P99_MIN:
        notes.append("Ảnh bị mờ hoặc nhiễu, hãy giữ máy cố định và lấy nét vào người")
    if min(h, w) < 480:
        notes.append("Độ phân giải thấp (nên từ 480 px)")
    ys = [p.y for p in lms if p.visibility >= 0.5 and 0.0 <= p.y <= 1.0]
    if ys and max(ys) - min(ys) < PERSON_MIN_FRAC:
        notes.append("Người chiếm quá ít diện tích ảnh, hãy chụp gần hơn")
    return notes


def silhouette_asymmetry(mask, a, b):
    """So khoảng cách từ mốc vai tới viền cơ thể hai bên (theo bề rộng vai)."""
    if mask is None:
        return None
    h, w = mask.shape
    width = b[0] - a[0]
    if width <= 0:
        return None
    y = min(h - 1, max(0, int(round((a[1] + b[1]) / 2.0))))
    row = mask[y] > 0.5
    xa0 = max(0, int(a[0] - 0.4 * width))
    xb0 = int(b[0])
    left = np.where(row[xa0:int(a[0]) + 1])[0]
    right = np.where(row[xb0:min(w, int(b[0] + 0.4 * width) + 1)])[0]
    if len(left) == 0 or len(right) == 0:
        return None
    gap_a = int(a[0]) - (xa0 + left.min())
    gap_b = right.max()
    return abs(gap_a - gap_b) / width
