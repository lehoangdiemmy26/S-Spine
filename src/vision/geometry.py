# -*- coding: utf-8 -*-
"""Mấy công thức hình học nhỏ dùng chung cho ảnh chính diện + nghiêng."""
import math

from src.config.landmarks import (
    L_EAR, L_FOOT, L_HEEL, L_HIP, L_SH, NOSE, R_EAR, R_FOOT, R_HEEL, R_HIP, R_SH,
)


def px(p, w, h):
    return p.x * w, p.y * h


def facing_of(lms, ear_i, w, hip_i=None, knee_i=None):
    """+1 mặt quay sang phải ảnh, -1 sang trái, None là không biết.

    Thứ tự: mũi so với tai -> gót/mũi chân -> (khi ngồi) đầu gối so với hông."""
    nose, ear = lms[NOSE], lms[ear_i]
    if nose.visibility >= 0.5 and 0 <= nose.x <= 1 and abs(nose.x - ear.x) * w > 0.01 * w:
        return 1 if nose.x > ear.x else -1
    if ear_i == L_EAR:
        heel, foot = lms[L_HEEL], lms[L_FOOT]
    else:
        heel, foot = lms[R_HEEL], lms[R_FOOT]
    if heel.visibility >= 0.5 and foot.visibility >= 0.5 and abs(foot.x - heel.x) * w > 0.01 * w:
        return 1 if foot.x > heel.x else -1
    if hip_i is not None and knee_i is not None:
        hip, knee = lms[hip_i], lms[knee_i]
        if hip.visibility >= 0.5 and knee.visibility >= 0.5 and abs(knee.x - hip.x) * w > 0.05 * w:
            return 1 if knee.x > hip.x else -1
    return None


def pick_side(lms):
    """Chọn bên người quay về phía máy: tổng độ tin cậy tai + vai + hông (không chỉ tai)."""
    def s(ids):
        return sum(lms[i].visibility for i in ids)

    return s((L_EAR, L_SH, L_HIP)) >= s((R_EAR, R_SH, R_HIP))


def ear_shoulder_angle(ear, sh, facing):
    # góc giữa phương ngang và đường vai->tai (90 độ = tai ngay trên vai)
    return math.degrees(math.atan2(sh[1] - ear[1], (ear[0] - sh[0]) * facing))


def trunk_lean(sh, hip, facing):
    # góc đường hông->vai so với phương thẳng đứng, dương = nghiêng ra trước
    return math.degrees(math.atan2((sh[0] - hip[0]) * facing, hip[1] - sh[1]))
