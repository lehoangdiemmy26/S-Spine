# -*- coding: utf-8 -*-
"""Đo lệch vai + lệch thân trên ảnh sau lưng / chính diện."""
import math

import cv2

from src.config.landmarks import L_HIP, L_SH, R_HIP, R_SH
from src.config.settings import ROT_WARN_FRONT, SYM_BLOCK, SYM_WARN, TH
from src.domain.levels import cap, lower_side, level_of
from src.domain.models import Result
from src.vision.geometry import px
from src.vision.overlay import ip, put_text, sizes
from src.vision.quality import check_points, image_notes, silhouette_asymmetry


def classify_front(tilt, lower, trunk):
    level = level_of(tilt, TH["shoulder_tilt"])
    why = []
    if level > 0 and lower != "cân bằng":
        why.append(f"vai {lower} thấp hơn ≈ {tilt:.1f}°")
    if trunk is not None:
        lt = level_of(abs(trunk), TH["trunk_shift"])
        if lt > 0:
            side = "phải" if trunk > 0 else "trái"
            why.append(f"thân lệch sang {side} so với hông ≈ {abs(trunk):.1f}°")
        level = max(level, lt)
    if why:
        return level, cap("; ".join(why)) + "."
    return level, f"Lệch vai ≈ {tilt:.1f}°, trong ngưỡng tham khảo."


def analyze_front(lms, img, view, mask=None):
    h, w = img.shape[:2]
    bad = check_points(lms, [L_SH, R_SH])
    if bad:
        return None, bad

    p1, p2 = px(lms[L_SH], w, h), px(lms[R_SH], w, h)
    # a = vai bên trái ảnh
    a, b = (p1, p2) if p1[0] <= p2[0] else (p2, p1)
    if b[0] - a[0] < 0.05 * w:
        return None, [
            "Hai vai gần như trùng nhau: ảnh này có vẻ là ảnh NGHIÊNG. "
            'Hãy chọn góc chụp "Nghiêng bên hông" hoặc chụp lại thẳng từ phía sau/trước'
        ]

    tilt_img = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
    # signed > 0 nghĩa là vai PHẢI người thấp hơn.
    # chụp sau lưng: phải ảnh = phải người; chụp trước mặt thì ngược lại
    signed = tilt_img if view == "back" else -tilt_img
    lower = lower_side(signed)

    notes = image_notes(img, lms)
    ms = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    mh, trunk = None, None
    if check_points(lms, [L_HIP, R_HIP]):
        notes.append("Không đo được độ lệch thân (hông bị che hoặc ngoài khung)")
    else:
        mh = (
            (lms[L_HIP].x + lms[R_HIP].x) / 2.0 * w,
            (lms[L_HIP].y + lms[R_HIP].y) / 2.0 * h,
        )
        torso = math.hypot(ms[0] - mh[0], ms[1] - mh[1])
        if torso > 1:
            t = math.degrees(math.atan2(ms[0] - mh[0], mh[1] - ms[1]))
            trunk = t if view == "back" else -t
            if (b[0] - a[0]) / torso < ROT_WARN_FRONT:
                notes.append("Người có vẻ bị xoay so với camera (vai nhìn hẹp)")

    asym = silhouette_asymmetry(mask, a, b)
    if asym is not None and asym > SYM_BLOCK:
        notes.append(
            "Mốc vai không đối xứng với viền cơ thể (áo rộng, tóc che vai hoặc người xoay), hãy chụp lại"
        )
    elif asym is not None and asym > SYM_WARN:
        notes.append(
            "Mốc vai hơi lệch so với viền cơ thể (áo rộng hoặc tóc che vai), kết quả kém chắc chắn"
        )

    level, summary = classify_front(abs(signed), lower, trunk)

    out = img.copy()
    t, r, fs = sizes(w)
    names = ("Vai Trai", "Vai Phai") if view == "back" else ("Vai Phai", "Vai Trai")
    cv2.line(out, ip(a), ip((b[0], a[1])), (255, 255, 0), t, cv2.LINE_AA)
    cv2.line(out, ip(a), ip(b), (0, 255, 0), t, cv2.LINE_AA)
    cv2.circle(out, ip(a), r, (255, 0, 0), -1)
    cv2.circle(out, ip(b), r, (0, 0, 255), -1)
    put_text(out, names[0], (ip(a)[0] - 20, max(20, ip(a)[1] - 2 * r)), fs, (255, 80, 80))
    put_text(out, names[1], (ip(b)[0] - 40, max(20, ip(b)[1] - 2 * r)), fs, (80, 120, 255))
    put_text(out, f"{abs(signed):.1f} deg", (ip(a)[0], ip(a)[1] + 4 * r), fs, (0, 255, 0))
    if mh is not None:
        cv2.line(out, ip(ms), ip(mh), (255, 0, 255), t, cv2.LINE_AA)
        cv2.line(out, ip(mh), ip((mh[0], ms[1])), (200, 200, 200), 1, cv2.LINE_AA)

    metrics = {
        "goc_lech_vai_do": round(abs(signed), 2),
        "vai_thap_hon": lower,
        "vai_phai_thap_hon_co_dau_do": round(signed, 2),
        "lech_than_so_voi_hong_do": None if trunk is None else round(trunk, 2),
        "huong_chup": "sau lưng" if view == "back" else "phía trước",
    }
    return Result(out, metrics, level, summary, notes), []
