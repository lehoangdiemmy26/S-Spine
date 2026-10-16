# -*- coding: utf-8 -*-
"""Mấy công thức hình học nhỏ dùng chung cho ảnh chính diện + nghiêng."""
import math

from src.config.landmarks import L_EAR, L_FOOT, L_HEEL, NOSE, R_EAR, R_FOOT, R_HEEL


def px(p, w, h):
    return p.x * w, p.y * h


def facing_of(lms, ear_i, w):
    """+1 mặt quay sang phải ảnh, -1 sang trái, None là không biết."""
    nose, ear = lms[NOSE], lms[ear_i]
    if nose.visibility >= 0.5 and 0 <= nose.x <= 1 and abs(nose.x - ear.x) * w > 0.01 * w:
        return 1 if nose.x > ear.x else -1
    if ear_i == L_EAR:
        heel, foot = lms[L_HEEL], lms[L_FOOT]
    else:
        heel, foot = lms[R_HEEL], lms[R_FOOT]
    if heel.visibility >= 0.5 and foot.visibility >= 0.5 and abs(foot.x - heel.x) * w > 0.01 * w:
        return 1 if foot.x > heel.x else -1
    return None


def ear_shoulder_angle(ear, sh, facing):
    # góc giữa phương ngang và đường vai->tai (90 độ = tai ngay trên vai)
    return math.degrees(math.atan2(sh[1] - ear[1], (ear[0] - sh[0]) * facing))


def trunk_lean(sh, hip, facing):
    # góc đường hông->vai so với phương thẳng đứng, dương = nghiêng ra trước
    return math.degrees(math.atan2((sh[0] - hip[0]) * facing, hip[1] - sh[1]))
