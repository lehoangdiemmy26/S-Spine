# -*- coding: utf-8 -*-
"""Đo đầu nhô ra trước + nghiêng thân trên ảnh nghiêng 90 độ."""
import math

import cv2

from src.config.landmarks import L_EAR, L_HIP, L_SH, R_EAR, R_HIP, R_SH
from src.config.settings import EAR_OK, EAR_WATCH, ROT_WARN_SIDE, TH, VIS_MIN
from src.domain.levels import cap, ear_level, level_of
from src.domain.models import Result
from src.vision.geometry import ear_shoulder_angle, facing_of, px, trunk_lean
from src.vision.overlay import ip, put_text, sizes
from src.vision.quality import check_points, image_notes


def classify_side(ear_ang, trunk, posture):
    th = TH["trunk_lean_sit" if posture == "sit" else "trunk_lean_stand"]
    l1 = ear_level(ear_ang)
    l2 = level_of(max(0.0, trunk), th)
    why = []
    if l1 > 0:
        why.append(
            f"đầu nhô ra trước (góc tai-vai ≈ {ear_ang:.0f}°, ngưỡng tham khảo ≥ {EAR_OK:.0f}°)"
        )
    if l2 > 0:
        why.append(f"thân nghiêng ra trước ≈ {trunk:.0f}°")
    if why:
        return max(l1, l2), cap("; ".join(why)) + "."
    return max(l1, l2), "Các chỉ số nằm trong ngưỡng tham khảo."


def analyze_side(lms, img, posture):
    h, w = img.shape[:2]
    # bên nào thấy rõ tai hơn thì đo bên đó
    left = lms[L_EAR].visibility >= lms[R_EAR].visibility
    if left:
        ear_i, sh_i, hip_i = L_EAR, L_SH, L_HIP
    else:
        ear_i, sh_i, hip_i = R_EAR, R_SH, R_HIP

    bad = check_points(lms, [ear_i, sh_i, hip_i])
    if bad:
        return None, bad + ["Hãy chụp thấy đủ từ đầu đến hông, người nằm hoàn toàn trong khung"]

    facing = facing_of(lms, ear_i, w)
    if facing is None:
        return None, ["Không xác định được hướng quay mặt (hãy chụp nghiêng 90°, thấy rõ mũi và tai)"]

    ear, sh, hip = px(lms[ear_i], w, h), px(lms[sh_i], w, h), px(lms[hip_i], w, h)
    torso = math.hypot(sh[0] - hip[0], sh[1] - hip[1])
    if torso < 1:
        return None, ["Mốc vai và hông trùng nhau, không đo được"]

    ang = ear_shoulder_angle(ear, sh, facing)
    lean = trunk_lean(sh, hip, facing)

    notes = image_notes(img, lms)
    other = lms[R_SH if left else L_SH]
    other_hip = lms[R_HIP if left else L_HIP]
    off_sh = (
        other.visibility >= VIS_MIN
        and 0 <= other.x <= 1
        and abs(other.x * w - sh[0]) / torso > ROT_WARN_SIDE
    )
    off_hip = (
        other_hip.visibility >= VIS_MIN
        and 0 <= other_hip.x <= 1
        and abs(other_hip.x * w - hip[0]) / torso > ROT_WARN_SIDE
    )
    if off_sh or off_hip:
        notes.append(
            "Thân người có vẻ chưa nghiêng 90° so với camera (hai vai/hai hông chưa trùng nhau). "
            "Nếu đây là ảnh sau lưng/chính diện, hãy đổi sang góc chụp sau lưng/chính diện"
        )

    level, summary = classify_side(ang, lean, posture)

    out = img.copy()
    t, r, fs = sizes(w)
    top = int(max(0, min(ear[1], sh[1]) - 0.06 * h))
    cv2.line(out, ip(hip), ip((hip[0], top)), (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(out, ip(sh), ip((sh[0] + facing * 0.15 * w, sh[1])), (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(out, ip(ear), ip(sh), (255, 255, 0), t, cv2.LINE_AA)
    cv2.line(out, ip(sh), ip(hip), (255, 0, 255), t, cv2.LINE_AA)
    cv2.circle(out, ip(ear), r, (255, 0, 0), -1)
    cv2.circle(out, ip(sh), r, (0, 255, 0), -1)
    cv2.circle(out, ip(hip), r, (0, 0, 255), -1)
    put_text(out, f"Ear-shoulder {ang:.0f} deg", (10, int(30 * fs / 0.5)), fs, (255, 255, 0))
    put_text(out, f"Trunk {lean:+.0f} deg", (10, int(60 * fs / 0.5)), fs, (255, 0, 255))

    metrics = {
        "goc_tai_vai_do": round(ang, 1),
        "nghieng_than_do": round(lean, 1),
        "huong_mat": "phải ảnh" if facing > 0 else "trái ảnh",
        "tu_the": "ngồi" if posture == "sit" else "đứng",
    }
    return Result(out, metrics, level, summary, notes), []
