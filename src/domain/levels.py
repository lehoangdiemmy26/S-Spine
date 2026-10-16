# -*- coding: utf-8 -*-
"""Xếp mức xanh/vàng/đỏ. Hàm nhỏ, không phụ thuộc streamlit để dễ test."""
from src.config.settings import EAR_OK, EAR_WATCH, LEVEL_EPS


def level_of(v, th):
    if v < th[0]:
        return 0
    return 1 if v <= th[1] else 2


def ear_level(a):
    if a >= EAR_OK:
        return 0
    return 1 if a >= EAR_WATCH else 2


def lower_side(signed):
    # signed > 0 nghĩa là vai phải người thấp hơn (đã quy đổi theo hướng chụp)
    if abs(signed) < LEVEL_EPS:
        return "cân bằng"
    return "phải" if signed > 0 else "trái"


def cap(s):
    if not s:
        return s
    return s[:1].upper() + s[1:]


def reliability(notes):
    if not notes:
        return "Tốt"
    if len(notes) == 1:
        return "Trung bình"
    return "Thấp"
