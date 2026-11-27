# -*- coding: utf-8 -*-
"""Kiểm tra các hàm lấy mốc cho tư thế ngồi (dùng mốc giả, không cần MediaPipe)."""
from types import SimpleNamespace as N

from src.analysis.side import sit_side_sanity
from src.vision.geometry import facing_of, pick_side

W = H = 1000
L_EAR, R_EAR, L_SH, R_SH, L_HIP, R_HIP, L_KNEE = 7, 8, 11, 12, 23, 24, 25


def _lms():
    return [N(x=0.5, y=0.5, visibility=0.9) for _ in range(33)]


def _sitting_facing_right():
    l = _lms()
    l[0] = N(x=0.5, y=0.2, visibility=0.2)        # mũi không tin cậy
    l[L_EAR] = N(x=0.45, y=0.20, visibility=0.9)
    l[L_SH] = N(x=0.44, y=0.35, visibility=0.9)
    l[L_HIP] = N(x=0.42, y=0.62, visibility=0.9)
    l[L_KNEE] = N(x=0.65, y=0.65, visibility=0.9)  # đầu gối phía trước (bên phải ảnh)
    return l


def test_facing_knee_fallback():
    l = _sitting_facing_right()
    assert facing_of(l, L_EAR, W) is None                        # không có mũi/chân thì chịu
    assert facing_of(l, L_EAR, W, L_HIP, L_KNEE) == 1            # có đầu gối thì biết


def test_sanity_ok_and_bad_hip():
    l = _sitting_facing_right()
    assert sit_side_sanity(l, L_EAR, L_SH, L_HIP, L_KNEE, 1, W, H) == []
    l[L_HIP] = N(x=0.44, y=0.43, visibility=0.9)                 # hông bị đoán sát vai
    assert len(sit_side_sanity(l, L_EAR, L_SH, L_HIP, L_KNEE, 1, W, H)) >= 1


def test_sanity_knee_behind_hip():
    l = _sitting_facing_right()
    l[L_KNEE] = N(x=0.30, y=0.65, visibility=0.9)                # đầu gối nằm sau hông
    assert len(sit_side_sanity(l, L_EAR, L_SH, L_HIP, L_KNEE, 1, W, H)) == 1


def test_pick_side_uses_ear_shoulder_hip():
    l = _sitting_facing_right()
    for i in (R_EAR, R_SH, R_HIP):
        l[i] = N(x=0.5, y=0.5, visibility=0.3)
    assert pick_side(l) is True
    for i in (L_EAR, L_SH, L_HIP):
        l[i].visibility = 0.2
    assert pick_side(l) is False
