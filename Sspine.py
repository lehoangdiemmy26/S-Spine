# -*- coding: utf-8 -*-
# S-Spine - tầm soát tư thế học đường từ ảnh chụp (dùng MediaPipe Pose)
#
# Lưu ý: app chỉ SÀNG LỌC, không chẩn đoán. Các ngưỡng trong bảng TH là ngưỡng tạm,
# cần được hiệu chỉnh bằng dữ liệu đo thật (xem tab "Kiểm chứng").

import csv
import io
import math
import threading
from dataclasses import dataclass, field
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
import mediapipe as mp

# ---------------------------------------------------------
# Thông số
# ---------------------------------------------------------
MAX_SIDE = 1280        # thu nhỏ ảnh về tối đa 1280 px cho mô hình
VIS_MIN = 0.6          # độ tin cậy tối thiểu của mốc
MAX_PHOTOS = 5         # số ảnh tối đa mỗi lần
LEVEL_EPS = 0.3        # dưới góc này coi như cân bằng

ROLL_MIN_FIX = 0.5     # máy nghiêng từ 0.5° mới bù
ROLL_MAX_FIX = 8.0     # nghiêng quá 8° thì không tự bù, yêu cầu chụp lại
SYM_WARN = 0.15        # mốc vai lệch đối xứng so với viền cơ thể quá 15% bề rộng vai
PERSON_MIN_FRAC = 0.30
ROT_WARN_FRONT = 0.60  # rộng vai / dài thân nhỏ hơn mức này -> người bị xoay
ROT_WARN_SIDE = 0.35   # ảnh nghiêng: hai vai cách nhau quá mức này -> chưa nghiêng 90°
SPREAD_WARN = {"front": 2.0, "side": 8.0}   # độ dao động giữa các ảnh (độ)

# Ngưỡng (t1, t2): giá trị < t1 xanh, <= t2 vàng, lớn hơn là đỏ.  TẤT CẢ ĐỀU LÀ TẠM THỜI.
TH = {
    "shoulder_tilt": (2.0, 4.0),
    "trunk_shift": (3.0, 6.0),
    "trunk_lean_stand": (5.0, 10.0),
    "trunk_lean_sit": (10.0, 20.0),
    "delta_tilt": (2.0, 4.0),
    "delta_trunk": (3.0, 6.0),
    "delta_ear_drop": (5.0, 10.0),
    "bag_ratio_pct": (10.0, 15.0),
}
EAR_OK, EAR_WATCH = 50.0, 40.0   # góc tai-vai (gần giống CVA): >=50 xanh, 40-50 vàng, <40 đỏ

# chỉ số mốc của MediaPipe Pose
NOSE, L_EAR, R_EAR = 0, 7, 8
L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24
L_HEEL, R_HEEL, L_FOOT, R_FOOT = 29, 30, 31, 32
LM_NAMES = {0: "mũi", 7: "tai trái", 8: "tai phải", 11: "vai trái", 12: "vai phải",
            23: "hông trái", 24: "hông phải"}

LEVEL_TXT = {0: "🟢 Bình thường", 1: "🟡 Cần theo dõi", 2: "🔴 Nên kiểm tra thêm"}
_lock = threading.Lock()


@dataclass
class Result:
    image: np.ndarray
    metrics: dict
    level: int
    summary: str
    notes: list = field(default_factory=list)   # cảnh báo làm giảm độ tin cậy
    info: list = field(default_factory=list)    # thông tin thêm, không trừ điểm


def level_of(v, th):
    if v < th[0]:
        return 0
    return 1 if v <= th[1] else 2


def ear_level(a):
    if a >= EAR_OK:
        return 0
    return 1 if a >= EAR_WATCH else 2


def cap(s):
    return s[:1].upper() + s[1:]


def reliability(notes):
    return "Tốt" if not notes else ("Trung bình" if len(notes) == 1 else "Thấp")


# ---------------------------------------------------------
# Mô hình & đọc ảnh
# ---------------------------------------------------------
@st.cache_resource
def get_pose():
    # các bản mediapipe mới không còn mp.solutions -> phải ghim phiên bản cũ
    if not hasattr(mp, "solutions"):
        return None
    for cx in (2, 1):
        try:
            return mp.solutions.pose.Pose(static_image_mode=True, model_complexity=cx,
                                          enable_segmentation=True, min_detection_confidence=0.5)
        except Exception:
            continue
    return None


def load_image(data):
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")   # xoay theo EXIF
    w, h = img.size
    k = MAX_SIDE / float(max(w, h))
    if k < 1:
        img = img.resize((round(w * k), round(h * k)), Image.LANCZOS)
    return np.ascontiguousarray(np.array(img))


def detect(img):
    with _lock:
        r = get_pose().process(img)
    if not r.pose_landmarks:
        return None, None
    lms = [SimpleNamespace(x=p.x, y=p.y, visibility=p.visibility) for p in r.pose_landmarks.landmark]
    return lms, r.segmentation_mask


# ---------------------------------------------------------
# Máy nghiêng: ước lượng từ các đường thẳng đứng của nền (khung cửa, cạnh tường...)
# ---------------------------------------------------------
def estimate_roll(img, lms=None):
    """Trả về góc nghiêng của máy (độ, dương = ảnh bị xoay thuận chiều kim đồng hồ) hoặc None."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    k = 800.0 / max(g.shape)
    if k < 1:
        g = cv2.resize(g, None, fx=k, fy=k)
    h, w = g.shape
    edges = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 30, 100)
    if lms:   # bỏ vùng có người, vì viền người không phải đường thẳng đứng chuẩn
        xs = [p.x for p in lms if p.visibility > 0.3]
        if xs:
            edges[:, int(max(0, min(xs) - 0.05) * w):int(min(1, max(xs) + 0.05) * w)] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 35, minLineLength=int(0.10 * h), maxLineGap=int(0.02 * h))
    leans = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            a = math.degrees(math.atan2(x1 - x2, y2 - y1))   # đỉnh nghiêng sang phải -> dương
            if abs(a) <= 10:
                leans.append(a)
    if len(leans) < 3:
        return None
    med = float(np.median(leans))
    if float(np.median(np.abs(np.array(leans) - med))) > 1.2:   # các đường không thống nhất
        return None
    return med


def rotate_image(img, angle):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------
# Kiểm tra mốc & chất lượng ảnh
# ---------------------------------------------------------
def check_points(lms, idxs):
    out = []
    for i in idxs:
        p = lms[i]
        if not (0.0 <= p.x <= 1.0 and 0.0 <= p.y <= 1.0):
            out.append(f"Mốc {LM_NAMES[i]} nằm ngoài khung ảnh (ảnh bị cắt)")
        elif p.visibility < VIS_MIN:
            out.append(f"Mốc {LM_NAMES[i]} bị che hoặc không rõ")
    return out


def image_notes(img, lms):
    notes = []
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if gray.mean() < 75:
        notes.append("Ảnh khá tối")
    small = cv2.resize(gray, (640, max(1, int(640 * h / w)))) if w != 640 else gray
    if cv2.Laplacian(small, cv2.CV_64F).var() < 40:
        notes.append("Ảnh bị mờ hoặc nhiễu")
    if min(h, w) < 480:
        notes.append("Độ phân giải thấp (nên từ 480 px)")
    ys = [p.y for p in lms if p.visibility >= 0.5 and 0.0 <= p.y <= 1.0]
    if ys and max(ys) - min(ys) < PERSON_MIN_FRAC:
        notes.append("Người chiếm quá ít diện tích ảnh, hãy chụp gần hơn")
    return notes


def silhouette_asymmetry(mask, a, b):
    """So khoảng cách từ mốc vai tới viền cơ thể ở hai bên (tính theo bề rộng vai).
    Áo rộng, tóc che vai hoặc mốc đặt sai thường làm hai bên lệch nhau."""
    if mask is None:
        return None
    h, w = mask.shape
    width = b[0] - a[0]
    y = min(h - 1, max(0, int(round((a[1] + b[1]) / 2.0))))
    row = mask[y] > 0.5
    xa0 = max(0, int(a[0] - 0.4 * width))
    xb0 = int(b[0])
    left = np.where(row[xa0:int(a[0]) + 1])[0]                      # phần cơ thể bên trái mốc vai
    right = np.where(row[xb0:min(w, int(b[0] + 0.4 * width) + 1)])[0]   # phần cơ thể bên phải mốc vai
    if len(left) == 0 or len(right) == 0:
        return None
    gap_a = int(a[0]) - (xa0 + left.min())     # từ mốc ra tới mép ngoài cơ thể
    gap_b = right.max()
    return abs(gap_a - gap_b) / width


# ---------------------------------------------------------
# Hình học
# ---------------------------------------------------------
def px(p, w, h):
    return p.x * w, p.y * h


def facing_of(lms, ear_i, w):
    """+1: mặt quay sang phải ảnh, -1: sang trái, None: không biết"""
    nose, ear = lms[NOSE], lms[ear_i]
    if nose.visibility >= 0.5 and 0 <= nose.x <= 1 and abs(nose.x - ear.x) * w > 0.01 * w:
        return 1 if nose.x > ear.x else -1
    heel, foot = (lms[L_HEEL], lms[L_FOOT]) if ear_i == L_EAR else (lms[R_HEEL], lms[R_FOOT])
    if heel.visibility >= 0.5 and foot.visibility >= 0.5 and abs(foot.x - heel.x) * w > 0.01 * w:
        return 1 if foot.x > heel.x else -1
    return None


def ear_shoulder_angle(ear, sh, facing):
    # góc giữa phương ngang và đường vai->tai (90° = tai ngay trên vai)
    return math.degrees(math.atan2(sh[1] - ear[1], (ear[0] - sh[0]) * facing))


def trunk_lean(sh, hip, facing):
    # góc đường hông->vai so với phương thẳng đứng, dương = nghiêng ra trước
    return math.degrees(math.atan2((sh[0] - hip[0]) * facing, hip[1] - sh[1]))


def classify_front(tilt, lower, trunk):
    level = level_of(tilt, TH["shoulder_tilt"])
    why = []
    if level > 0 and lower != "cân bằng":
        why.append(f"vai {lower} thấp hơn ≈ {tilt:.1f}°")
    if trunk is not None:
        lt = level_of(abs(trunk), TH["trunk_shift"])
        if lt > 0:
            why.append(f"thân lệch sang {'phải' if trunk > 0 else 'trái'} so với hông ≈ {abs(trunk):.1f}°")
        level = max(level, lt)
    return level, (cap("; ".join(why)) + "." if why else f"Lệch vai ≈ {tilt:.1f}°, trong ngưỡng tham khảo.")


def classify_side(ear_ang, trunk, posture):
    th = TH["trunk_lean_sit" if posture == "sit" else "trunk_lean_stand"]
    l1, l2 = ear_level(ear_ang), level_of(max(0.0, trunk), th)
    why = []
    if l1 > 0:
        why.append(f"đầu nhô ra trước (góc tai-vai ≈ {ear_ang:.0f}°, bình thường ≥ {EAR_OK:.0f}°)")
    if l2 > 0:
        why.append(f"thân nghiêng ra trước ≈ {trunk:.0f}°")
    return max(l1, l2), (cap("; ".join(why)) + "." if why else "Các chỉ số nằm trong ngưỡng tham khảo.")


def lower_side(signed):
    return "cân bằng" if abs(signed) < LEVEL_EPS else ("phải" if signed > 0 else "trái")


# ---------------------------------------------------------
# Vẽ
# ---------------------------------------------------------
def ip(p):
    return int(round(p[0])), int(round(p[1]))


def put_text(img, s, org, fs, color):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, fs, color, 2, cv2.LINE_AA)


def sizes(w):
    return max(2, int(round(w / 300))), max(4, int(round(w / 110))), max(0.5, w / 1100)


# ---------------------------------------------------------
# Phân tích ảnh sau lưng / chính diện
# ---------------------------------------------------------
def analyze_front(lms, img, view, mask=None):
    h, w = img.shape[:2]
    bad = check_points(lms, [L_SH, R_SH])
    if bad:
        return None, bad
    p1, p2 = px(lms[L_SH], w, h), px(lms[R_SH], w, h)
    a, b = (p1, p2) if p1[0] <= p2[0] else (p2, p1)      # a = vai bên trái ảnh
    if b[0] - a[0] < 0.05 * w:
        return None, ["Hai vai gần như trùng nhau: hãy chụp thẳng từ phía sau/trước, không chụp nghiêng"]

    tilt_img = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))    # dương: vai bên phải ảnh thấp hơn
    # signed > 0 nghĩa là vai PHẢI của người thấp hơn
    # chụp sau lưng: phải ảnh = phải người. Chụp phía trước: phải ảnh = trái người
    signed = tilt_img if view == "back" else -tilt_img
    lower = lower_side(signed)

    notes = image_notes(img, lms)
    ms = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    mh, trunk = None, None
    if check_points(lms, [L_HIP, R_HIP]):
        notes.append("Không đo được độ lệch thân (hông bị che hoặc ngoài khung)")
    else:
        mh = ((lms[L_HIP].x + lms[R_HIP].x) / 2.0 * w, (lms[L_HIP].y + lms[R_HIP].y) / 2.0 * h)
        torso = math.hypot(ms[0] - mh[0], ms[1] - mh[1])
        if torso > 1:
            t = math.degrees(math.atan2(ms[0] - mh[0], mh[1] - ms[1]))
            trunk = t if view == "back" else -t
            if (b[0] - a[0]) / torso < ROT_WARN_FRONT:
                notes.append("Người có vẻ bị xoay so với camera (vai nhìn hẹp)")

    asym = silhouette_asymmetry(mask, a, b)
    if asym is not None and asym > SYM_WARN:
        notes.append("Mốc vai không đối xứng với viền cơ thể (áo rộng, tóc che vai hoặc người xoay)")

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


# ---------------------------------------------------------
# Phân tích ảnh nghiêng
# ---------------------------------------------------------
def analyze_side(lms, img, posture):
    h, w = img.shape[:2]
    left = lms[L_EAR].visibility >= lms[R_EAR].visibility
    ear_i, sh_i, hip_i = (L_EAR, L_SH, L_HIP) if left else (R_EAR, R_SH, R_HIP)
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
    if other.visibility >= VIS_MIN and 0 <= other.x <= 1 and abs(other.x * w - sh[0]) / torso > ROT_WARN_SIDE:
        notes.append("Thân người có vẻ chưa nghiêng 90° so với camera")

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


# ---------------------------------------------------------
# Chạy phân tích (có cache) + bù máy nghiêng
# ---------------------------------------------------------
@st.cache_data(show_spinner="Đang phân tích ảnh...", ttl=300, max_entries=30)
def analyze_cached(kind, data, view, posture, auto_roll):
    img = load_image(data)
    lms, mask = detect(img)
    if lms is None:
        return None, ["Không nhận diện được người trong ảnh"]

    roll = estimate_roll(img, lms)
    fixed = False
    if roll is not None and auto_roll and ROLL_MIN_FIX <= abs(roll) <= ROLL_MAX_FIX:
        rotated = rotate_image(img, roll)        # xoay ngược lại đúng góc máy bị nghiêng
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
    if fixed:
        res.info.append(f"Đã bù máy nghiêng ≈ {roll:+.1f}° (ước lượng từ các đường thẳng đứng trong nền).")
    elif roll is None:
        res.info.append("Không ước lượng được độ nghiêng của máy (nền ít đường thẳng đứng). "
                        "Hãy đặt máy cố định, ngang bằng.")
    elif abs(roll) > 1.5:
        res.notes.append(f"Máy có vẻ nghiêng ≈ {roll:.1f}° nhưng chưa được bù")
    return res, []


def analyze_file(f, kind, view="back", posture="stand", auto_roll=True):
    return analyze_cached(kind, f.getvalue(), view, posture, auto_roll)


def merge_results(rs, kind):
    """Gộp nhiều ảnh của cùng một người: lấy trung vị từng chỉ số."""
    notes = sorted({n for r in rs for n in r.notes})
    info = []
    m0 = rs[0].metrics
    if kind == "front":
        signed = [r.metrics["vai_phai_thap_hon_co_dau_do"] for r in rs]
        med = float(np.median(signed))
        trunks = [r.metrics["lech_than_so_voi_hong_do"] for r in rs if r.metrics["lech_than_so_voi_hong_do"] is not None]
        trunk = float(np.median(trunks)) if trunks else None
        spread = max(signed) - min(signed)
        level, summary = classify_front(abs(med), lower_side(med), trunk)
        metrics = {
            "goc_lech_vai_do": round(abs(med), 2), "vai_thap_hon": lower_side(med),
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
        metrics = {"goc_tai_vai_do": round(ang, 1), "nghieng_than_do": round(lean, 1),
                   "huong_mat": m0["huong_mat"], "tu_the": m0["tu_the"]}
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
    return (singles[0] if len(singles) == 1 else merge_results(singles, kind)), bad, singles


# ---------------------------------------------------------
# Giao diện: tiện ích
# ---------------------------------------------------------
def show_image(img, caption=None):
    # tên tham số của st.image thay đổi theo phiên bản, thử lần lượt
    for kw in ({"width": "stretch"}, {"use_container_width": True}, {"use_column_width": True}, {}):
        try:
            st.image(img, caption=caption, **kw)
            return
        except Exception:
            continue


def image_inputs(label, key, source, multiple=True):
    if source == "Tải ảnh lên":
        got = st.file_uploader(label, type=["jpg", "jpeg", "png"], key=f"u_{key}", accept_multiple_files=multiple)
        if not multiple:
            return [got] if got else []
        return list(got or [])
    f = st.camera_input(label, key=f"c_{key}")
    return [f] if f else []


def source_selector(key):
    s = st.radio("Chọn phương thức đầu vào:", ["Tải ảnh lên", "Chụp bằng Camera"], key=f"src_{key}", horizontal=True)
    if s == "Chụp bằng Camera":
        st.caption("Camera trước của một số điện thoại lật gương ảnh, làm đảo trái/phải. Nên dùng camera sau.")
    return s


def view_selector(key):
    v = st.radio("Ảnh được chụp từ phía:", ["Sau lưng", "Phía trước mặt"], key=f"view_{key}", horizontal=True)
    return "back" if v == "Sau lưng" else "front"


def clothing_check(key):
    ok = st.checkbox("Xác nhận: em mặc áo ôm sát, tóc không che vai, máy đặt cố định", key=f"cl_{key}")
    return [] if ok else ["Chưa xác nhận áo ôm sát / tóc không che vai / máy cố định"]


def show_level(level, text):
    {0: st.success, 1: st.warning, 2: st.error}[level](f"**{LEVEL_TXT[level]}** - {text}")


def show_problems(bad):
    st.error("**Không thể phân tích ảnh này:**\n\n" + "\n".join(f"- {p}" for p in bad) +
             "\n\nHãy chụp lại theo tab **Hướng dẫn chụp ảnh chuẩn**.")


def show_quality(res, extra=None):
    notes = res.notes + (extra or [])
    rel = reliability(notes)
    st.caption(f"Độ tin cậy của ảnh: **{rel}**")
    for n in notes:
        st.caption("• " + n)
    for n in res.info:
        st.caption("ℹ️ " + n)
    if rel == "Thấp":
        st.warning("Độ tin cậy thấp: kết quả chỉ để tham khảo, nên chụp lại ảnh rõ và đúng chuẩn hơn.")


def csv_bytes(label, res, extra=None):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["muc", "chi_so", "gia_tri"])
    for k, v in res.metrics.items():
        w.writerow([label, k, "" if v is None else v])
    w.writerow([label, "muc_do", LEVEL_TXT[res.level]])
    w.writerow([label, "do_tin_cay", reliability(res.notes + (extra or []))])
    return buf.getvalue().encode("utf-8-sig")


def csv_button(label, res, key, extra=None):
    st.download_button("⬇️ Tải chỉ số (CSV) để đối chiếu đo thủ công", data=csv_bytes(label, res, extra),
                       file_name=f"{key}.csv", mime="text/csv", key=f"dl_{key}")


def show_gallery(res, caption, gallery):
    if gallery:
        cols = st.columns(min(3, len(gallery)))
        for i, img in enumerate(gallery):
            with cols[i % len(cols)]:
                show_image(img, f"Ảnh {i + 1}")
    else:
        show_image(res.image, caption)


def adams_hint():
    st.info("🔎 Nếu nghi ngờ lệch vai hoặc vẹo cột sống, bước sàng lọc tiếp theo thường dùng là **nghiệm pháp cúi người "
            "(Adams)** do nhân viên y tế thực hiện, đo góc xoay thân bằng scoliometer (thường chuyển khám khi khoảng "
            "5-7° trở lên). Ảnh 2D của ứng dụng không nhìn thấy độ xoay của đốt sống nên không thay thế được bước này.")


def render_front(res, caption, key, gallery=None, extra=None):
    m = res.metrics
    show_gallery(res, caption, gallery)
    c1, c2, c3 = st.columns(3)
    c1.metric("Góc lệch vai", f"{m['goc_lech_vai_do']:.1f}°", help="Góc giữa đường nối hai vai và phương ngang.")
    c2.metric("Vai thấp hơn", str(m["vai_thap_hon"]))
    t = m["lech_than_so_voi_hong_do"]
    c3.metric("Lệch thân so với hông", "—" if t is None else f"{abs(t):.1f}°",
              help="Góc đường hông→vai so với phương thẳng đứng (nhìn từ sau/trước).")
    show_level(res.level, res.summary)
    show_quality(res, extra)
    if res.level > 0:
        adams_hint()
    csv_button(caption, res, key, extra)


def render_side(res, caption, key, gallery=None, extra=None):
    m = res.metrics
    show_gallery(res, caption, gallery)
    c1, c2, c3 = st.columns(3)
    c1.metric("Góc tai–vai", f"{m['goc_tai_vai_do']:.0f}°",
              help="Góc giữa phương ngang và đường vai→tai. Càng nhỏ, đầu càng nhô ra trước. Gần với CVA nhưng dùng mốc vai thay cho C7.")
    c2.metric("Góc nghiêng thân", f"{m['nghieng_than_do']:+.0f}°", help="Dương: thân nghiêng ra trước.")
    c3.metric("Hướng quay mặt", str(m["huong_mat"]))
    show_level(res.level, res.summary)
    show_quality(res, extra)
    csv_button(caption, res, key, extra)


def run_and_show(files, kind, view, posture, auto_roll, caption, key, extra=None):
    res, bad, singles = run_many(files, kind, view, posture, auto_roll)
    if res is None:
        show_problems(bad)
        return None
    if bad:
        st.warning("Một số ảnh bị bỏ qua:\n\n" + "\n".join(f"- {p}" for p in bad))
    gallery = [r.image for r in singles] if len(singles) > 1 else None
    if kind == "front":
        render_front(res, caption, key, gallery, extra)
    else:
        render_side(res, caption, key, gallery, extra)
    return res


# ---------------------------------------------------------
# Gợi ý vận động
# ---------------------------------------------------------
def card(title, lines):
    body = "".join(f"<p>{x}</p>" for x in lines)
    st.markdown(f'<div class="exercise-card"><div class="exercise-title">{title}</div>{body}</div>',
                unsafe_allow_html=True)


def show_recommendations(level):
    st.markdown("---")
    st.subheader("Gợi ý vận động hỗ trợ tư thế")
    if level == 0:
        st.success("**Tư thế trong ngưỡng tham khảo.** Hãy đứng dậy vận động nhẹ sau mỗi 30-45 phút học.")
        c1, c2 = st.columns(2)
        with c1:
            card("1. Xoay vai & mở ngực", ["Đứng thẳng, xoay tròn hai vai từ trước ra sau.", "⏱ Gợi ý: 10-15 lần mỗi hướng."])
        with c2:
            card("2. Nghiêng cổ giãn cơ", ["Nghiêng đầu sang một bên, tay hỗ trợ nhẹ.", "⏱ Gợi ý: giữ 15 giây mỗi bên."])
    elif level == 1:
        st.warning("**Cần theo dõi:** nên chụp lại sau 1-2 tuần để so sánh và chú ý tư thế khi học.")
        c1, c2 = st.columns(2)
        with c1:
            card("1. Tư thế Mèo - Bò", ["Quỳ bốn điểm; hít vào võng lưng, thở ra cong lưng.", "⏱ Gợi ý: 10-12 lần/ngày."])
        with c2:
            card("2. Chống tường mở vai", ["Đứng cách tường khoảng 0.5 m, chống tay, nhẹ nhàng đưa ngực về phía tường.",
                                          "⏱ Gợi ý: giữ 20 giây x 3 lần."])
    else:
        st.error("**Nên kiểm tra thêm:** kết quả vượt ngưỡng tham khảo. Hãy báo phụ huynh/giáo viên và đến cơ sở y tế "
                 "chuyên khoa cột sống để được khám. Ảnh chụp **không** thể chẩn đoán bệnh.")
        st.caption("Các động tác dưới đây chỉ là vận động nhẹ, nên hỏi ý kiến bác sĩ trước khi tập thường xuyên.")
        c1, c2, c3 = st.columns(3)
        with c1:
            card("1. Nằm sấp nâng ngực nhẹ", ["Nằm sấp, nâng nhẹ ngực và tay khỏi sàn, siết cơ lưng.", "⏱ Gợi ý: giữ 10 giây x vài lần."])
        with c2:
            card("2. Giãn & mở ngực", ["Nằm ngửa trên khăn cuộn đặt dưới lưng ngực, mở rộng hai tay.", "⏱ Gợi ý: 2-3 phút mỗi ngày."])
        with c3:
            card("3. Plank khuỷu tay (nhẹ)", ["Chống khuỷu tay, giữ thân thẳng; có thể tựa đầu gối nếu mỏi.", "⏱ Gợi ý: giữ 20-30 giây x vài lần."])


# ---------------------------------------------------------
# Các mục chức năng
# ---------------------------------------------------------
def mode_standing(auto_roll):
    st.subheader("1. Tầm soát tư thế đứng tĩnh")
    sub = st.selectbox("Chọn góc chụp:", ["Sau lưng / Chính diện (kiểm tra lệch vai)",
                                          "Nghiêng bên hông (kiểm tra khom lưng & đầu nhô)"])
    st.caption(f"Có thể tải tối đa {MAX_PHOTOS} ảnh của cùng một người (chụp rời nhau vài giây); app lấy trung vị để giảm sai số.")
    src = source_selector("m1")
    if sub.startswith("Sau lưng"):
        view = view_selector("m1")
        extra = clothing_check("m1")
        files = image_inputs("Ảnh sau lưng/chính diện:", "m1_front", src)
        if files:
            res = run_and_show(files, "front", view, "stand", auto_roll, "Đứng - lệch vai", "dung_lech_vai", extra)
            if res:
                show_recommendations(res.level)
    else:
        files = image_inputs("Ảnh nghiêng bên hông (90°):", "m1_side", src)
        if files:
            res = run_and_show(files, "side", "back", "stand", auto_roll, "Đứng - nghiêng", "dung_nghieng")
            if res:
                show_recommendations(res.level)


def mode_bag(auto_roll):
    st.subheader("2. Tầm soát khi đeo cặp sách")
    st.info("Mục này chỉ cho biết tư thế KHI đeo cặp. Muốn biết cặp làm thay đổi tư thế bao nhiêu, hãy dùng mục 4.")
    c1, c2 = st.columns(2)
    bag = c1.number_input("Trọng lượng cặp (kg) - do bạn nhập", 0.0, 30.0, 0.0, 0.5)
    body = c2.number_input("Cân nặng của em (kg) - không bắt buộc", 0.0, 150.0, 0.0, 1.0)
    worst = 0
    if bag > 0 and body > 0:
        pct = bag / body * 100.0
        lv = level_of(pct, TH["bag_ratio_pct"])
        worst = lv
        show_level(lv, f"Cặp nặng {bag:g} kg ≈ {pct:.0f}% cân nặng (khuyến nghị thường gặp: khoảng 10%, không nên quá 15%).")
    src = source_selector("m2")
    view = view_selector("m2")
    extra = clothing_check("m2")
    ca, cb = st.columns(2)
    with ca:
        f_front = image_inputs("Ảnh đeo cặp (sau lưng/chính diện)", "m2_front", src, multiple=False)
    with cb:
        f_side = image_inputs("Ảnh đeo cặp (nghiêng bên hông)", "m2_side", src, multiple=False)
    shown = False
    if f_front:
        st.markdown("#### Ảnh sau lưng / chính diện")
        r = run_and_show(f_front, "front", view, "stand", auto_roll, "Đeo cặp - lệch vai", "cap_lech_vai", extra)
        if r:
            shown, worst = True, max(worst, r.level)
    if f_side:
        st.markdown("#### Ảnh nghiêng bên hông")
        r = run_and_show(f_side, "side", "back", "stand", auto_roll, "Đeo cặp - nghiêng", "cap_nghieng")
        if r:
            shown, worst = True, max(worst, r.level)
    if shown:
        show_recommendations(worst)
    elif not f_front and not f_side:
        st.info("Hãy tải lên hoặc chụp ít nhất một ảnh để bắt đầu.")


def mode_sitting(auto_roll):
    st.subheader("3. Tư thế ngồi học")
    st.info("Có thể dùng một hoặc cả hai góc chụp: sau lưng (lệch vai) và nghiêng bên hông (cúi đầu, khom lưng). "
            "Tư thế ngồi thay đổi liên tục, nên hãy chụp vài ảnh ở các thời điểm khác nhau.")
    src = source_selector("m3")
    extra = clothing_check("m3")
    ca, cb = st.columns(2)
    with ca:
        f_back = image_inputs("Ảnh ngồi từ phía sau lưng", "m3_back", src)
    with cb:
        f_side = image_inputs("Ảnh ngồi nghiêng bên hông (90°)", "m3_side", src)
    worst, shown = 0, False
    if f_back:
        st.markdown("#### Góc sau lưng")
        r = run_and_show(f_back, "front", "back", "sit", auto_roll, "Ngồi - sau lưng", "ngoi_sau_lung", extra)
        if r:
            shown, worst = True, max(worst, r.level)
    if f_side:
        st.markdown("#### Góc nghiêng")
        r = run_and_show(f_side, "side", "back", "sit", auto_roll, "Ngồi - nghiêng", "ngoi_nghieng")
        if r:
            shown, worst = True, max(worst, r.level)
    if shown:
        show_recommendations(worst)
    elif not f_back and not f_side:
        st.info("Hãy tải lên hoặc chụp ít nhất một ảnh để bắt đầu.")


def mode_compare(auto_roll):
    st.subheader("4. So sánh trước và sau khi đeo cặp")
    st.info("Chụp **cùng vị trí, cùng khoảng cách, cùng chiều cao máy** cho cả hai ảnh để so sánh đáng tin.")
    sub = st.selectbox("Chọn góc chụp so sánh:", ["Sau lưng / Chính diện (lệch vai)", "Nghiêng bên hông (khom lưng, đầu nhô)"])
    src = source_selector("m4")
    is_front = sub.startswith("Sau lưng")
    view = view_selector("m4") if is_front else "back"
    ca, cb = st.columns(2)
    with ca:
        f1 = image_inputs("1. Ảnh KHÔNG đeo cặp", "m4_norm", src, multiple=False)
    with cb:
        f2 = image_inputs("2. Ảnh KHI đeo cặp", "m4_pack", src, multiple=False)
    bag = st.number_input("Trọng lượng cặp (kg) - do bạn nhập", 0.0, 30.0, 0.0, 0.5, key="m4_bag")
    if not (f1 and f2):
        return
    kind = "front" if is_front else "side"
    r1, p1 = analyze_file(f1[0], kind, view, "stand", auto_roll)
    r2, p2 = analyze_file(f2[0], kind, view, "stand", auto_roll)
    if r1 is None or r2 is None:
        if r1 is None:
            st.markdown("**Ảnh 1 (không đeo cặp):**")
            show_problems(p1)
        if r2 is None:
            st.markdown("**Ảnh 2 (đeo cặp):**")
            show_problems(p2)
        return
    ca, cb = st.columns(2)
    with ca:
        show_image(r1.image, "Ảnh 1: không đeo cặp")
    with cb:
        show_image(r2.image, "Ảnh 2: đeo cặp")

    if is_front:
        k = "vai_phai_thap_hon_co_dau_do"
        d = r2.metrics[k] - r1.metrics[k]
        lv = level_of(abs(d), TH["delta_tilt"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Lệch vai ban đầu", f"{r1.metrics['goc_lech_vai_do']:.1f}°")
        c2.metric("Lệch vai khi đeo", f"{r2.metrics['goc_lech_vai_do']:.1f}°",
                  delta=f"{r2.metrics['goc_lech_vai_do'] - r1.metrics['goc_lech_vai_do']:+.1f}°", delta_color="inverse")
        c3.metric("Cặp (bạn nhập)", f"{bag:g} kg")
        if lv == 0:
            show_level(0, f"Cặp không làm thay đổi đáng kể độ cân bằng vai (thay đổi ≈ {abs(d):.1f}°).")
        else:
            show_level(lv, f"Khi đeo cặp, {'vai phải' if d > 0 else 'vai trái'} hạ thấp thêm ≈ {abs(d):.1f}°. "
                           "Hãy chỉnh hai quai cho lực đều hai bên, cặp áp sát giữa lưng, đồ nặng đặt sát lưng rồi chụp lại.")
    else:
        d_lean = r2.metrics["nghieng_than_do"] - r1.metrics["nghieng_than_do"]
        d_ear = r2.metrics["goc_tai_vai_do"] - r1.metrics["goc_tai_vai_do"]
        lv = max(level_of(d_lean, TH["delta_trunk"]), level_of(-d_ear, TH["delta_ear_drop"]))
        c1, c2, c3 = st.columns(3)
        c1.metric("Nghiêng thân khi đeo", f"{r2.metrics['nghieng_than_do']:+.0f}°", delta=f"{d_lean:+.1f}°", delta_color="inverse")
        c2.metric("Góc tai–vai khi đeo", f"{r2.metrics['goc_tai_vai_do']:.0f}°", delta=f"{d_ear:+.1f}°")
        c3.metric("Cặp (bạn nhập)", f"{bag:g} kg")
        if lv == 0:
            show_level(0, "Cặp không làm thân ngả ra trước hay đầu nhô ra trước đáng kể.")
        else:
            show_level(lv, "Cặp kéo thân trên ngả ra trước và/hoặc đầu nhô ra trước. Hãy siết quai cho cặp áp sát lưng, "
                           "dùng đai ngực/đai bụng nếu có và giảm bớt đồ nặng.")
    for r, name in ((r1, "Ảnh 1"), (r2, "Ảnh 2")):
        if reliability(r.notes) != "Tốt":
            st.caption(f"{name} - độ tin cậy {reliability(r.notes)}: {'; '.join(r.notes)}")


# ---------------------------------------------------------
# Tab kiểm chứng với đo thủ công
# ---------------------------------------------------------
def agreement_stats(app_vals, manual_vals):
    a, m = np.asarray(app_vals, float), np.asarray(manual_vals, float)
    d = a - m
    bias, sd = float(d.mean()), float(d.std(ddof=1)) if len(d) > 1 else 0.0
    r = float(np.corrcoef(a, m)[0, 1]) if len(a) > 2 and a.std() > 0 and m.std() > 0 else float("nan")
    return {"n": len(a), "MAE": float(np.abs(d).mean()), "bias": bias,
            "LoA_low": bias - 1.96 * sd, "LoA_high": bias + 1.96 * sd, "r": r}


def confusion(app_vals, manual_vals, cut):
    a, m = np.asarray(app_vals, float) >= cut, np.asarray(manual_vals, float) >= cut
    tp, fp, fn, tn = int((a & m).sum()), int((a & ~m).sum()), int((~a & m).sum()), int((~a & ~m).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return tp, fp, fn, tn, sens, spec


def tab_validate(auto_roll):
    st.subheader("Kiểm chứng với đo thủ công")
    st.caption("Tải nhiều ảnh, nhập góc bạn đo thủ công (ImageJ/Kinovea/thước đo góc) vào cột cuối, "
               "app tính sai số và độ nhạy/độ đặc hiệu. Chỉ dùng ảnh đã được người chụp đồng ý.")
    what = st.selectbox("Chỉ số cần kiểm chứng", ["Góc lệch vai (ảnh sau lưng)", "Góc tai–vai (ảnh nghiêng)",
                                                   "Góc nghiêng thân (ảnh nghiêng)"])
    front = what.startswith("Góc lệch vai")
    view = view_selector("val") if front else "back"
    files = st.file_uploader("Ảnh cần kiểm chứng", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="val_files")
    if not files:
        return
    rows = []
    for f in files:
        res, bad = analyze_file(f, "front" if front else "side", view, "stand", auto_roll)
        if res is None:
            rows.append({"Ảnh": f.name, "App (độ)": None, "Đo thủ công (độ)": None, "Ghi chú": "; ".join(bad)})
            continue
        if front:
            v = res.metrics["goc_lech_vai_do"]
        elif what.startswith("Góc tai"):
            v = res.metrics["goc_tai_vai_do"]
        else:
            v = res.metrics["nghieng_than_do"]
        rows.append({"Ảnh": f.name, "App (độ)": round(v, 2), "Đo thủ công (độ)": None, "Ghi chú": reliability(res.notes)})
    edited = st.data_editor(pd.DataFrame(rows), key="val_table", hide_index=True,
                            disabled=["Ảnh", "App (độ)", "Ghi chú"])
    ok = edited.dropna(subset=["App (độ)"]).copy()
    ok["Đo thủ công (độ)"] = pd.to_numeric(ok["Đo thủ công (độ)"], errors="coerce")
    ok = ok.dropna(subset=["Đo thủ công (độ)"])
    if len(ok) < 3:
        st.info("Nhập ít nhất 3 giá trị đo thủ công để xem thống kê (nên từ 30 ca trở lên).")
        return
    s = agreement_stats(ok["App (độ)"], ok["Đo thủ công (độ)"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số ca", s["n"])
    c2.metric("Sai số tuyệt đối TB (MAE)", f"{s['MAE']:.2f}°")
    c3.metric("Sai lệch trung bình (bias)", f"{s['bias']:+.2f}°")
    c4.metric("Hệ số tương quan r", "—" if math.isnan(s["r"]) else f"{s['r']:.2f}")
    st.caption(f"Giới hạn đồng thuận 95% (Bland-Altman): {s['LoA_low']:+.2f}° đến {s['LoA_high']:+.2f}°")

    st.markdown("**Độ nhạy / độ đặc hiệu theo một ngưỡng bất thường**")
    default_cut = TH["shoulder_tilt"][0] if front else 10.0
    cut = st.number_input("Ngưỡng (độ): từ giá trị này trở lên coi là bất thường", 0.0, 90.0, float(default_cut), 0.5, key="val_cut")
    tp, fp, fn, tn, sens, spec = confusion(ok["App (độ)"], ok["Đo thủ công (độ)"], cut)
    st.dataframe(pd.DataFrame([[tp, fn], [fp, tn]], index=["Đo thủ công: bất thường", "Đo thủ công: bình thường"],
                              columns=["App: bất thường", "App: bình thường"]))
    e1, e2 = st.columns(2)
    e1.metric("Độ nhạy", "—" if math.isnan(sens) else f"{sens * 100:.0f}%")
    e2.metric("Độ đặc hiệu", "—" if math.isnan(spec) else f"{spec * 100:.0f}%")
    st.caption("Muốn chọn ngưỡng tốt, hãy thử nhiều giá trị và chọn mức cân bằng giữa độ nhạy và độ đặc hiệu "
               "(hoặc dùng đường cong ROC trên bộ dữ liệu lớn hơn). Chuẩn đối chiếu tốt nhất là góc Cobb trên X-quang.")
    st.download_button("⬇️ Tải bảng kiểm chứng (CSV)", edited.to_csv(index=False).encode("utf-8-sig"),
                       file_name="kiem_chung.csv", mime="text/csv", key="dl_val")


# ---------------------------------------------------------
# Trang
# ---------------------------------------------------------
CSS = """
<style>
.stApp { background-color: #FFFFFF; color: #1A202C; }
.main-title { color: #0284C7; font-weight: 800; font-size: 2.5rem; margin-bottom: 0px; }
.sub-title { color: #475569; font-size: 1.05rem; font-weight: 500; margin-bottom: 20px; }
[data-testid="stFileUploadDropzone"] { background-color: #F8FAFC !important; border: 2px dashed #38BDF8 !important; border-radius: 12px !important; }
[data-testid="stMetricValue"] { color: #0F172A !important; font-weight: 700 !important; }
.guide-card { background-color: #F0F9FF; border-left: 5px solid #0284C7; padding: 15px; border-radius: 8px; margin-bottom: 15px; }
.exercise-card { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 15px; margin-bottom: 15px; }
.exercise-title { color: #0369A1; font-weight: 700; font-size: 1.1rem; margin-bottom: 8px; }
.disclaimer-box { background-color: #FFFBEB; border: 1px solid #FDE68A; padding: 12px; border-radius: 8px; color: #92400E; font-size: 0.9rem; margin-top: 30px; text-align: center; }
</style>
"""


def load_logo():
    for name in ["gen-n-z8308486911094_3cf6e9f66d814eabd93c0c5ae610e055-modified.png",
                 "gen-n-z8308486911094_3cf6e9f66d814eabd93c0c5ae610e055-modified.jpg"]:
        try:
            return Image.open(name)
        except Exception:
            continue
    return None


def tab_guide():
    st.subheader("Quy trình chụp ảnh tầm soát chuẩn")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown('<div class="guide-card"><h4>1. Trang phục</h4><ul>'
                    '<li>Áo thun ôm sát, không áo khoác dày hay áo rộng.</li>'
                    '<li>Buộc gọn tóc dài để lộ cổ và vai.</li></ul></div>', unsafe_allow_html=True)
    with g2:
        st.markdown('<div class="guide-card"><h4>2. Góc chụp</h4><ul>'
                    '<li><b>Sau lưng/chính diện:</b> máy thẳng góc, người đứng thẳng đối diện.</li>'
                    '<li><b>Nghiêng:</b> quay đúng 90°, thấy đủ từ đầu đến hông.</li>'
                    '<li>Chọn đúng "chụp từ phía sau lưng / phía trước".</li></ul></div>', unsafe_allow_html=True)
    with g3:
        st.markdown('<div class="guide-card"><h4>3. Máy ảnh</h4><ul>'
                    '<li>Cách 1.5 - 2 m, ngang tầm ngực/lưng.</li>'
                    '<li><b>Đặt máy cố định, ngang bằng</b> (giá đỡ/bàn); máy nghiêng làm sai góc lệch vai.</li>'
                    '<li>Nền có đường thẳng đứng (khung cửa, góc tường) giúp app tự bù máy nghiêng.</li>'
                    '<li>Chụp 3-5 ảnh rời nhau vài giây để lấy trung vị.</li></ul></div>', unsafe_allow_html=True)


def tab_method():
    st.subheader("Phương pháp & giới hạn")
    st.markdown(
        "**Cách đo.** Mô hình MediaPipe Pose tìm các mốc vai, hông, tai. Mọi chỉ số là góc thật (độ) tính từ mốc.\n\n"
        "- **Góc lệch vai:** góc giữa đường nối hai vai và phương ngang.\n"
        "- **Lệch thân so với hông:** góc đường (giữa hông → giữa vai) so với phương thẳng đứng, ảnh sau lưng/chính diện.\n"
        "- **Góc tai–vai:** góc giữa phương ngang và đường vai → tai ở ảnh nghiêng. Gần với CVA nhưng **không phải CVA chuẩn**, "
        "vì không có mốc đốt sống C7 nên dùng mốc vai thay thế.\n"
        "- **Góc nghiêng thân:** góc đường hông → vai so với phương thẳng đứng, dương là nghiêng ra trước.\n\n"
        "**Giảm sai số.** Tự ước lượng máy nghiêng từ các đường thẳng đứng trong nền rồi xoay ảnh bù lại; "
        "kiểm tra mốc nằm trong khung, đủ rõ; so độ đối xứng của mốc vai với viền cơ thể; lấy trung vị của nhiều ảnh."
    )
    rows = [("Lệch vai (độ)", "shoulder_tilt"), ("Lệch thân, ảnh sau lưng (độ)", "trunk_shift"),
            ("Nghiêng thân khi đứng (độ)", "trunk_lean_stand"), ("Nghiêng thân khi ngồi (độ)", "trunk_lean_sit"),
            ("Tăng lệch vai khi đeo cặp (độ)", "delta_tilt"), ("Tăng nghiêng thân khi đeo cặp (độ)", "delta_trunk"),
            ("Giảm góc tai–vai khi đeo cặp (độ)", "delta_ear_drop"), ("Cặp / cân nặng (%)", "bag_ratio_pct")]
    md = "| Chỉ số | Xanh khi nhỏ hơn | Vàng khi ≤ (lớn hơn là đỏ) |\n|---|---|---|\n"
    md += "\n".join(f"| {n} | {TH[k][0]:g} | {TH[k][1]:g} |" for n, k in rows)
    md += f"\n| Góc tai–vai (độ) | ≥ {EAR_OK:g} xanh | ≥ {EAR_WATCH:g} vàng, nhỏ hơn là đỏ |"
    st.markdown("**Ngưỡng phân loại: tạm thời, chưa được kiểm chứng lâm sàng.** Dùng tab *Kiểm chứng* để đối chiếu và hiệu chỉnh.")
    st.markdown(md)
    st.markdown(
        "**Giới hạn.**\n"
        "- Ảnh 2D không thay thế X-quang hay khám lâm sàng, không chẩn đoán vẹo/gù cột sống. Vẹo cột sống là biến dạng 3D "
        "có xoay đốt sống; nghiệm pháp cúi người (Adams) với scoliometer mới đánh giá được góc xoay thân.\n"
        "- Mốc do mô hình ước lượng; áo rộng, tóc che vai, ảnh tối/mờ vẫn có thể làm lệch mốc. App chỉ cảnh báo được một phần.\n"
        "- Việc bù máy nghiêng chỉ làm được khi nền có đủ đường thẳng đứng; nếu không, hãy đặt máy cố định, ngang bằng.\n"
        "- Mỗi ảnh chỉ ghi lại một khoảnh khắc, chưa cho biết đó có phải thói quen lâu dài hay không.\n"
        "- App không ghi ảnh ra đĩa hay cơ sở dữ liệu; ảnh chỉ được xử lý tạm trong bộ nhớ."
    )


def main():
    logo = load_logo()
    st.set_page_config(page_title="S-Spine | Tầm Soát Lệch Vai & Tư Thế Học Đường",
                       page_icon=logo if logo is not None else "🛡️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    if get_pose() is None:
        st.error("Không khởi tạo được MediaPipe Pose. Hãy ghim phiên bản có `mp.solutions` trong requirements.txt "
                 "(ví dụ `mediapipe==0.10.14`) rồi triển khai lại.")
        st.stop()

    c1, c2 = st.columns([0.08, 0.92])
    with c1:
        if logo is not None:
            st.image(logo, width=65)
        else:
            st.markdown("<h1>🛡️</h1>", unsafe_allow_html=True)
    with c2:
        st.markdown('<h1 class="main-title" style="margin-top: 5px;">S-Spine</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Ứng dụng AI hỗ trợ tầm soát tư thế & lệch vai học đường</p>', unsafe_allow_html=True)

    t_guide, t_app, t_val, t_method = st.tabs(["Hướng dẫn chụp ảnh chuẩn", "Tầm soát & phân tích AI",
                                                "Kiểm chứng (đo thủ công)", "Phương pháp & giới hạn"])
    with t_guide:
        tab_guide()
    with t_app:
        auto_roll = st.checkbox("Tự động bù máy nghiêng (khuyến nghị)", value=True, key="auto_roll")
        mode = st.radio("Chọn mục tầm soát tư thế:", ["1. Đứng tĩnh (lệch vai / khom lưng)", "2. Đeo cặp sách",
                                                       "3. Tư thế ngồi học", "4. So sánh trước & sau khi đeo cặp"])
        st.markdown("---")
        if mode.startswith("1"):
            mode_standing(auto_roll)
        elif mode.startswith("2"):
            mode_bag(auto_roll)
        elif mode.startswith("3"):
            mode_sitting(auto_roll)
        else:
            mode_compare(auto_roll)
    with t_val:
        tab_validate(st.session_state.get("auto_roll", True))
    with t_method:
        tab_method()

    st.markdown('<div class="disclaimer-box"><b>Lưu ý quan trọng:</b> Ứng dụng chỉ hỗ trợ tầm soát và mang tính chất tham khảo '
                'học đường, không thay thế chẩn đoán của bác sĩ chuyên khoa.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
