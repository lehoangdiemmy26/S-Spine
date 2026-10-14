# -*- coding: utf-8 -*-
# S-Spine - tầm soát tư thế học đường từ ảnh chụp (dùng MediaPipe Pose)
#
# Lưu ý: app chỉ SÀNG LỌC, không chẩn đoán. Các ngưỡng trong bảng TH là ngưỡng THAM KHẢO CỦA NHÓM
# (chưa có nguồn chuẩn cho từng ngưỡng), cần được hiệu chỉnh bằng dữ liệu đo thật.

import base64
import csv
import io
import math
import threading
from dataclasses import dataclass, field
from types import SimpleNamespace

import cv2
import numpy as np
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
ROLL_MAX_FIX = 8.0     # nghiêng quá 8° thì không tự bù, yêu cầu chụp lại hoặc nhập tay góc nghiêng
SYM_WARN = 0.15        # mốc vai lệch đối xứng so với viền cơ thể quá 15% bề rộng vai
PERSON_MIN_FRAC = 0.30
ROT_WARN_FRONT = 0.60  # rộng vai / dài thân nhỏ hơn mức này -> người bị xoay
ROT_WARN_SIDE = 0.35   # ảnh nghiêng: hai vai cách nhau quá mức này -> chưa nghiêng 90°
SPREAD_WARN = {"front": 2.0, "side": 8.0}   # độ dao động giữa các ảnh (độ)

# Các cảnh báo dưới đây nghĩa là ảnh KHÔNG đạt chuẩn -> không phân tích, yêu cầu chụp lại
BLOCKING = ("Ảnh khá tối", "Ảnh bị mờ", "Độ phân giải thấp", "Người chiếm quá ít", "Người có vẻ bị xoay",
            "Mốc vai không đối xứng", "Thân người có vẻ chưa nghiêng 90°", "Máy có vẻ nghiêng")

# Ngưỡng (t1, t2): giá trị < t1 xanh, <= t2 vàng, lớn hơn là đỏ.  TẤT CẢ LÀ NGƯỠNG THAM KHẢO CỦA NHÓM.
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
        return None, ["Hai vai gần như trùng nhau: ảnh này có vẻ là ảnh NGHIÊNG. Hãy chọn góc chụp \"Nghiêng bên hông\" hoặc chụp lại thẳng từ phía sau/trước"]

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
    other_hip = lms[R_HIP if left else L_HIP]
    off_sh = other.visibility >= VIS_MIN and 0 <= other.x <= 1 and abs(other.x * w - sh[0]) / torso > ROT_WARN_SIDE
    off_hip = other_hip.visibility >= VIS_MIN and 0 <= other_hip.x <= 1 and abs(other_hip.x * w - hip[0]) / torso > ROT_WARN_SIDE
    if off_sh or off_hip:
        notes.append("Thân người có vẻ chưa nghiêng 90° so với camera (hai vai/hai hông chưa trùng nhau). "
                     "Nếu đây là ảnh sau lưng/chính diện, hãy đổi sang góc chụp sau lưng/chính diện")

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

    manual = isinstance(auto_roll, float) and not isinstance(auto_roll, bool)
    roll = float(auto_roll) if manual else estimate_roll(img, lms)
    fixed = False
    limit = 20.0 if manual else ROLL_MAX_FIX
    if roll is not None and auto_roll is not False and ROLL_MIN_FIX <= abs(roll) <= limit:
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
    # ảnh không đạt chuẩn -> yêu cầu chụp lại
    if not fixed and not manual and roll is not None and abs(roll) > 1.5 and auto_roll is not False:
        res.notes.append(f"Máy có vẻ nghiêng ≈ {roll:.1f}° nhưng không bù được. Hãy đặt máy ngang bằng hoặc nhập tay góc nghiêng máy")
    blocked = [n for n in res.notes if n.startswith(BLOCKING)]
    if blocked:
        return None, blocked + ["👉 Ảnh chưa đạt chuẩn, vui lòng CHỤP LẠI."]
    if fixed:
        res.info.append(f"Đã bù máy nghiêng ≈ {roll:+.1f}° ({'nhập tay' if manual else 'ước lượng từ các đường thẳng đứng trong nền'}).")
    elif roll is None:
        res.info.append("Không ước lượng được độ nghiêng của máy (nền ít đường thẳng đứng). "
                        "Hãy đặt máy cố định, ngang bằng.")
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
    c1.metric("Đầu nhô ra trước (góc tai–vai)", f"{m['goc_tai_vai_do']:.0f}°",
              help="Góc giữa phương ngang và đường vai→tai. Càng nhỏ, đầu càng nhô ra trước. Gần với CVA nhưng dùng mốc vai thay cho C7.")
    c2.metric("Nghiêng thân", f"{m['nghieng_than_do']:+.0f}°", help="Dương: thân nghiêng ra trước.")
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
    st.subheader("1.1 Đứng tĩnh (lệch vai / khom, đầu nhô ra trước)")
    sub = st.selectbox("Chọn góc chụp:", ["Sau lưng / Chính diện (kiểm tra lệch vai)",
                                          "Nghiêng bên hông (kiểm tra tư thế khom & đầu nhô ra trước)"])
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
    st.subheader("1.2 Đeo cặp sách")
    st.info("Mục này chỉ cho biết tư thế KHI đeo cặp. Muốn biết cặp làm thay đổi tư thế bao nhiêu, hãy dùng mục 3 (So sánh trước & sau khi đeo cặp).")
    c1, c2 = st.columns(2)
    bag = c1.number_input("Trọng lượng cặp (kg) - do bạn nhập", 0.0, 30.0, 0.0, 0.5)
    body = c2.number_input("Cân nặng của em (kg) - cần nhập để tính tỷ lệ cặp/cân nặng", 0.0, 150.0, 0.0, 1.0)
    if bag > 0 and body <= 0:
        st.warning("Chưa nhập cân nặng nên chưa tính được tỷ lệ cặp/cân nặng (chỉ số khuyến nghị ≤ 10-15%).")
    st.caption("Cặp có thể che mốc vai/hông. Nếu mốc trên ảnh kết quả đặt sai chỗ, hãy chụp lại.")
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
    st.subheader("2. Tư thế ngồi học")
    st.info("Có thể dùng một hoặc cả hai góc chụp: sau lưng (lệch vai) và nghiêng bên hông (cúi đầu, khom, đầu nhô ra trước). "
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


# ---------------------------------------------------------
# Gù lưng (ước tính từ ảnh có nhãn đánh dấu, người dùng chạm chọn điểm)
# ---------------------------------------------------------
# Bộ mốc lấy theo giao thức DIPA (C7, T2, T4, T6, T8, T10, T12, L2, L4, S2): ở đây dùng 6 mốc T2 -> T12
KY_LABELS = ["T2 (đếm xuống 2 mỏm gai dưới C7; C7 là mỏm nhô rõ nhất ở gốc cổ)", "T4", "T6", "T8", "T10",
             "T12 (cuối lồng ngực, ngang đáy xương bả vai là mốc tham khảo)"]
KY_N = len(KY_LABELS)
KY_DISPLAY_W = 560


def kyphosis_angle(pts):
    """pts: KY_N điểm (x, y) theo thứ tự từ T2 xuống T12 (ảnh nghiêng).
    Dựng đường cong x = f(y) bậc 3 (bình phương tối thiểu) qua các điểm, lấy tiếp tuyến ở T2 và T12,
    trả về góc giữa hai tiếp tuyến (độ). Đây là CÁCH LÀM CỦA NHÓM theo tinh thần DIPA (tiếp tuyến tại T2, T12);
    công thức lượng giác chính xác của DIPA chưa được đối chiếu với toàn văn bài báo."""
    ys = np.array([p[1] for p in pts], dtype=float)
    xs = np.array([p[0] for p in pts], dtype=float)
    if not np.all(np.diff(ys) > 0):
        return None
    c = np.polyfit(ys, xs, 3)
    d = np.polyder(c)
    a_top = math.degrees(math.atan(np.polyval(d, ys[0])))
    a_bot = math.degrees(math.atan(np.polyval(d, ys[-1])))
    return abs(a_top - a_bot)


def mode_kyphosis():
    st.subheader("1.3 Gù lưng (ước tính từ ảnh)")
    with st.expander("📌 Hướng dẫn dán 6 nhãn lên lưng (T2, T4, T6, T8, T10, T12)", expanded=False):
        st.markdown(KY_GUIDE)
    _ky_measure()


KY_GUIDE = """
**Chuẩn bị**
- 6 nhãn tròn nhỏ (đường kính khoảng 1–1,5 cm), màu tương phản với da/áo (vàng chanh, cam). Nên dùng màu khác cho **T2 và T12** (hai điểm cuối dùng tính tiếp tuyến).
- Bút viết da hoặc bút kẻ mắt, thước dây.
- Trang phục: **áo ba lỗ/áo thể thao hở lưng ôm sát** (dán lên da, chính xác nhất) hoặc **áo thun mỏng ôm sát, không nhăn** (kín đáo hơn nhưng kém chính xác hơn). Không bắt học sinh cởi áo.
- Người dán cùng giới tính với học sinh, đo nơi kín đáo, có người lớn có mặt, đã được phụ huynh/nhà trường đồng ý. Học sinh có quyền từ chối.
- Nên để **một người cố định** dán nhãn cho cả đợt đo, và không trộn nhóm dán trên da với nhóm dán trên áo khi tính sai số.

**Các bước**
1. **Tìm C7:** cho học sinh cúi đầu hết mức, sờ giữa gáy từ trên xuống. Mỏm gai nhô rõ nhất ở gốc cổ thường là C7 (nếu thấy hai mỏm nhô, thường mỏm trên là C7).
2. **Tìm T2:** từ C7 đếm xuống hai mỏm gai (mỏm 1 = T1, mỏm 2 = T2). Đánh dấu bằng bút.
3. **Tìm T12:** đếm xuống từ T2 thêm 10 mỏm gai. Đối chiếu: đường nối hai mào chậu thường ngang L4, đếm lên 4 mỏm (L3, L2, L1, T12). Nếu hai cách lệch quá 1 mỏm, sờ lại. Đánh dấu bằng bút.
4. **Chia đều T4, T6, T8, T10:** đo khoảng cách dọc cột sống từ T2 đến T12 = L. Đánh dấu cách T2 lần lượt **20% L (T4), 40% L (T6), 60% L (T8), 80% L (T10)**. Đối chiếu: góc dưới xương bả vai (tay buông) thường ngang T7, tức nằm giữa T6 và T8.
5. **Dán nhãn:** dán tâm nhãn đúng lên điểm đánh dấu, ép phẳng. Cả 6 nhãn cùng kích thước, nằm trên đường giữa lưng. Cho học sinh đứng thẳng lại, kiểm tra các nhãn xếp thành một đường tương đối dọc; nhãn nào lệch hẳn thì dán lại.
6. **Chụp ảnh:** học sinh đứng **nghiêng 90°**, bên có nhãn quay vào máy, mắt nhìn thẳng, tay buông hoặc khoanh nhẹ trước ngực. Máy cố định, ngang bằng, ngang tầm giữa lưng. Không nhún vai hay ưỡn ngực quá mức.
7. **Trong app:** chạm lần lượt từ trên xuống **T2 → T4 → T6 → T8 → T10 → T12**. Chạm sai thì dùng nút "Bỏ điểm vừa chọn".

**Lưu ý sai số:** xác định mỏm gai bằng cách sờ thường lệch khoảng một đốt. Nên nhờ người thứ hai dán lại ở một vài học sinh để ước lượng độ lặp lại giữa hai người đo.
"""


def _ky_measure():
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
    except Exception:
        st.error("Thiếu thư viện chạm chọn điểm. Hãy thêm `streamlit-image-coordinates` vào requirements.txt rồi triển khai lại.")
        return
    st.info("**Cách làm:** dán 6 nhãn màu nhỏ lên mỏm gai (T2, T4, T6, T8, T10, T12; xác định bằng cách sờ và đếm từ C7), chụp ảnh nghiêng 90° rồi chạm lần lượt vào từng nhãn "
            "theo thứ tự từ trên xuống (đường cong bậc 3 qua 6 điểm, tiếp tuyến tại T2 và T12). Kết quả là **góc gù ước tính từ ảnh**, không phải góc Cobb và không thay thế X-quang.")
    src = source_selector("m5")
    extra = clothing_check("m5")
    files = image_inputs("Ảnh nghiêng có nhãn đánh dấu trên lưng:", "m5_side", src, multiple=False)
    if not files:
        return
    data = files[0].getvalue()
    sig = f"{len(data)}-{hash(data[:2048])}"
    if st.session_state.get("ky_sig") != sig:
        st.session_state.update(ky_sig=sig, ky_pts=[], ky_last=None)
    img = load_image(data)
    h, w = img.shape[:2]
    scale = w / float(KY_DISPLAY_W)
    pts = st.session_state["ky_pts"]
    disp = cv2.resize(img, (KY_DISPLAY_W, int(round(h / scale))))
    for i, (x, y) in enumerate(pts):
        c = (int(x / scale), int(y / scale))
        cv2.circle(disp, c, 6, (255, 0, 0), -1)
        put_text(disp, str(i + 1), (c[0] + 8, c[1] - 6), 0.6, (255, 255, 0))
    if len(pts) < KY_N:
        st.markdown(f"**Chạm vào: {len(pts) + 1}/{KY_N} - {KY_LABELS[len(pts)]}**")
    val = streamlit_image_coordinates(Image.fromarray(disp), key=f"ky_img_{sig}_{len(pts)}")
    if val and len(pts) < KY_N:
        key = (val["x"], val["y"], len(pts))
        if st.session_state.get("ky_last") != key:
            st.session_state["ky_last"] = key
            k = w / float(val.get("width") or KY_DISPLAY_W)
            pts.append((val["x"] * k, val["y"] * k))
            st.rerun()
    c1, c2 = st.columns(2)
    if c1.button("↩️ Bỏ điểm vừa chọn", disabled=not pts):
        pts.pop()
        st.session_state["ky_last"] = None
        st.rerun()
    if c2.button("🗑️ Chọn lại từ đầu", disabled=not pts):
        st.session_state.update(ky_pts=[], ky_last=None)
        st.rerun()
    if len(pts) < KY_N:
        return
    ang = kyphosis_angle(pts)
    if ang is None:
        st.error("Thứ tự điểm chưa đúng (phải đi từ T2 xuống T12, từ trên xuống dưới). Hãy chọn lại.")
        return
    st.metric("Góc gù ước tính từ ảnh", f"{ang:.0f}°",
              help="Góc giữa hai tiếp tuyến của đường cong qua 6 nhãn, tại T2 và T12. Không phải góc Cobb.")
    st.warning("Chưa có ngưỡng chuẩn cho lứa tuổi học sinh nên app **không phân mức** bình thường/bất thường. "
               "Phương pháp ảnh có sai số so với X-quang (nghiên cứu DIPA ở người lớn: ±4°, nhưng bản này của nhóm chưa kiểm chứng). "
               "Nếu con số lớn hoặc có lo ngại, hãy đến cơ sở y tế chuyên khoa để được khám.")
    if extra:
        st.caption("• " + extra[0])
    st.caption("Kết quả phụ thuộc vào việc bạn chạm đúng vị trí nhãn; nên đối chiếu với người có chuyên môn đo bằng thước dẻo/kyphometer.")



def mode_compare(auto_roll):
    st.subheader("3. So sánh trước và sau khi đeo cặp")
    st.info("Chụp **cùng vị trí, cùng khoảng cách, cùng chiều cao máy** cho cả hai ảnh để so sánh đáng tin.")
    sub = st.selectbox("Chọn góc chụp so sánh:", ["Sau lưng / Chính diện (lệch vai)", "Nghiêng bên hông (tư thế khom, đầu nhô ra trước)"])
    src = source_selector("m4")
    is_front = sub.startswith("Sau lưng")
    view = view_selector("m4") if is_front else "back"
    ca, cb = st.columns(2)
    with ca:
        f1 = image_inputs("1. Ảnh KHÔNG đeo cặp", "m4_norm", src, multiple=False)
    with cb:
        f2 = image_inputs("2. Ảnh KHI đeo cặp", "m4_pack", src, multiple=False)
    bag = st.number_input("Trọng lượng cặp (kg) - do bạn nhập", 0.0, 30.0, 0.0, 0.5, key="m4_bag")
    extra = clothing_check("m4")
    st.caption("Cặp có thể che mốc vai/hông. Nếu mốc trên ảnh kết quả đặt sai chỗ, hãy chụp lại.")
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
        c2.metric("Đầu nhô ra trước (tai–vai) khi đeo", f"{r2.metrics['goc_tai_vai_do']:.0f}°", delta=f"{d_ear:+.1f}°")
        c3.metric("Cặp (bạn nhập)", f"{bag:g} kg")
        if lv == 0:
            show_level(0, "Cặp không làm thân ngả ra trước hay đầu nhô ra trước đáng kể.")
        else:
            show_level(lv, "Cặp kéo thân trên ngả ra trước và/hoặc đầu nhô ra trước. Hãy siết quai cho cặp áp sát lưng, "
                           "dùng đai ngực/đai bụng nếu có và giảm bớt đồ nặng.")
    for r, name in ((r1, "Ảnh 1"), (r2, "Ảnh 2")):
        nn = r.notes + extra
        if reliability(nn) != "Tốt":
            st.caption(f"{name} - độ tin cậy {reliability(nn)}: {'; '.join(nn)}")


# ---------------------------------------------------------
# Trang
# ---------------------------------------------------------
def load_logo():
    for name in ["gen-n-z8308486911094_3cf6e9f66d814eabd93c0c5ae610e055-modified.png",
                 "gen-n-z8308486911094_3cf6e9f66d814eabd93c0c5ae610e055-modified.jpg"]:
        try:
            return Image.open(name)
        except Exception:
            continue
    return None


# ---------------------------------------------------------
# Trang đầu (tab Hướng dẫn): giao diện
# ---------------------------------------------------------


def _one_line(html):
    return "".join(line.strip() for line in html.strip().splitlines())


def _curve(t):
    """Đường cong cột sống nhìn nghiêng (mặt hướng sang phải): gù ngực rồi ưỡn thắt lưng. Trả về (x, y)."""
    return (175 - 34 * math.sin(2 * math.pi * t) + 14 * (1 - t) ** 3, 70 + 330 * t)


def _tangent(t, e=1e-3):
    (x0, y0), (x1, y1) = _curve(t - e), _curve(t + e)
    return x1 - x0, y1 - y0


def spine_svg(mode):
    """mode='hero': cột sống + mốc tai-vai-hông. mode='ky': cột sống + 6 nhãn T2..T12 ở mặt lưng."""
    n = 19
    parts = []
    for i in range(n):
        t = i / (n - 1)
        x, y = _curve(t)
        dx, dy = _tangent(t)
        ang = -math.degrees(math.atan2(dx, dy))
        w, h = 24 + 12 * t, 14 + 5 * t
        parts.append(f'<rect class="v" style="animation-delay:{i * 55}ms" x="{-w / 2:.1f}" y="{-h / 2:.1f}" '
                     f'width="{w:.1f}" height="{h:.1f}" rx="5" transform="translate({x:.1f} {y:.1f}) rotate({ang:.1f})"/>')
    body = "".join(parts)
    if mode == "hero":
        ex, ey = 192, 40          # tai
        sx, sy = 183, 110         # vai
        hx, hy = 194, 354         # hông
        body = (f'<circle cx="200" cy="36" r="27" fill="none" stroke="#8FB4D9" stroke-width="2"/>' + body +
                f'<g class="lm">'
                f'<line x1="{ex}" y1="{ey}" x2="{ex}" y2="{hy + 14}" stroke="#E7EEF7" stroke-width="1.5" stroke-dasharray="4 5"/>'
                f'<line x1="{ex}" y1="{ey}" x2="{sx}" y2="{sy}" stroke="#FFB547" stroke-width="2.5"/>'
                f'<line x1="{sx}" y1="{sy}" x2="{sx + 70}" y2="{sy}" stroke="#FFB547" stroke-width="1.5" stroke-dasharray="4 4"/>'
                f'<path d="M {sx + 34} {sy} A 34 34 0 0 0 {sx + 34 * math.cos(math.radians(-83)):.1f} {sy + 34 * math.sin(math.radians(-83)):.1f}" '
                f'fill="none" stroke="#FFB547" stroke-width="2"/>'
                f'<circle cx="{ex}" cy="{ey}" r="6" fill="#FFB547"/><circle cx="{sx}" cy="{sy}" r="6" fill="#FFB547"/>'
                f'<circle cx="{hx}" cy="{hy}" r="6" fill="#FFB547"/>'
                f'<text x="236" y="44" class="tx">Tai</text><text x="216" y="132" class="tx">Vai</text>'
                f'<text x="216" y="360" class="tx">Hông</text>'
                f'<text x="{sx + 44}" y="{sy - 24}" class="tx s">góc tai–vai</text></g>')
        return (f'<svg class="art" viewBox="0 0 360 430" role="img" aria-label="Sơ đồ cột sống nhìn nghiêng với các mốc tai, vai, hông">'
                f'{body}</svg>')
    labels = ["T2", "T4", "T6", "T8", "T10", "T12"]
    marks = []
    ts = [0.10 + i * (0.45 / 5) for i in range(6)]
    for lab, t in zip(labels, ts):
        x, y = _curve(t)
        dx, dy = _tangent(t)
        ln = math.hypot(dx, dy)
        nx, ny = -dy / ln, dx / ln          # pháp tuyến hướng ra phía lưng (bên trái)
        mx, my = x + nx * 30, y + ny * 30
        marks.append((lab, t, mx, my, dx / ln, dy / ln))
    g = ""
    for lab, t, mx, my, ux, uy in marks:
        if lab in ("T2", "T12"):
            g += (f'<line x1="{mx - ux * 60:.1f}" y1="{my - uy * 60:.1f}" x2="{mx + ux * 60:.1f}" y2="{my + uy * 60:.1f}" '
                  f'stroke="#E08A00" stroke-width="2" stroke-dasharray="5 4"/>')
    for lab, t, mx, my, ux, uy in marks:
        g += (f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="8" fill="#FFB547" stroke="#fff" stroke-width="2"/>'
              f'<text x="{mx - 16:.1f}" y="{my + 5:.1f}" text-anchor="end" class="tk">{lab}</text>')
    return (f'<svg class="art" viewBox="0 0 360 430" role="img" aria-label="Sơ đồ 6 nhãn dán trên lưng từ T2 đến T12">'
            f'{body}{g}</svg>')


def logo_uri(logo):
    if logo is None:
        return None
    im = logo.convert("RGBA")
    im.thumbnail((120, 120))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


CSS_HOME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;800&display=swap');
:root { --navy:#0B1F3A; --navy2:#12305A; --blue:#2B7BBF; --sky:#E8F3FB; --amber:#FFB547; --ink:#10243E; --mute:#4A5D75; --line:#D3E2F0; }
.block-container { max-width: 1120px; padding-top: 5rem; }
@media (max-width: 640px) { .block-container { padding-top: 4.5rem; } }
.stMarkdown, .stMarkdown p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, [data-baseweb="tab"] p,
.ss-top, .ss-hero, .ss-sec, .ss-ky, .ss-modes { font-family: 'Be Vietnam Pro', 'Segoe UI', system-ui, sans-serif; }
[data-baseweb="tab"] p { font-weight: 600; }
[data-baseweb="tab-highlight"] { background-color: var(--navy) !important; }

.ss-top { display:flex; align-items:center; gap:14px; margin-bottom:18px; }
.ss-logo { width:48px; height:48px; border-radius:50%; flex:none; }
.ss-name { font-size:1.55rem; font-weight:800; color:var(--navy); line-height:1.1; }
.ss-tag { font-size:.95rem; color:var(--mute); margin-top:2px; }

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

@media (max-width: 860px) {
  .ss-hero { grid-template-columns:1fr; padding:30px 24px 20px; }
  .ss-hero .art { max-height:300px; }
  .ss-steps { grid-template-columns:1fr 1fr; }
  .ss-ky { grid-template-columns:1fr; padding:26px 22px; }
  .ss-ky .art { max-height:300px; }
}
@media (max-width: 520px) { .ss-steps { grid-template-columns:1fr; } }
</style>
"""


def header_html(logo):
    uri = logo_uri(logo)
    img = f'<img class="ss-logo" src="{uri}" alt="Logo S-Spine"/>' if uri else ""
    return _one_line(f'<div class="ss-top">{img}<div><div class="ss-name">S-Spine</div>'
                     f'<div class="ss-tag">AI đánh giá và phân tích tư thế học đường</div></div></div>')


def tab_guide():
    st.markdown(_one_line(f"""
    <section class="ss-hero">
      <div>
        <div class="ss-h" role="heading" aria-level="2">Chụp một tấm ảnh, thấy tư thế đang lệch ở đâu</div>
        <p>S-Spine tìm vị trí tai, vai, hông trên ảnh rồi tính các góc tư thế. Kết quả chỉ để sàng lọc, không thay thế chẩn đoán của bác sĩ.</p>
        <span class="ss-hint">Bắt đầu ở tab “Đánh giá &amp; phân tích tư thế”</span>
      </div>
      <div>{spine_svg("hero")}</div>
    </section>"""), unsafe_allow_html=True)

    st.markdown(_one_line("""
    <section class="ss-sec">
      <div class="ss-st" role="heading" aria-level="2">Chụp đúng trong 4 bước</div>
      <ol class="ss-steps">
        <li><div class="n">1</div><b>Mặc gọn</b>Áo thun ôm sát, không áo khoác dày hay áo rộng. Buộc gọn tóc dài để lộ cổ và vai.</li>
        <li><div class="n">2</div><b>Đứng đúng hướng</b>Ảnh sau lưng hoặc chính diện: đứng thẳng đối diện máy. Ảnh nghiêng: quay đúng 90°, thấy đủ từ đầu đến hông. Nhớ chọn đúng “chụp từ phía sau lưng / phía trước” trong app.</li>
        <li><div class="n">3</div><b>Đặt máy</b>Cách 1,5–2 m, ngang tầm ngực/lưng, cố định và ngang bằng (giá đỡ hoặc bàn). Nền có khung cửa hay góc tường giúp app bù khi máy hơi nghiêng.</li>
        <li><div class="n">4</div><b>Chụp nhiều ảnh</b>Chụp 3–5 ảnh cách nhau vài giây. App lấy giá trị trung vị để giảm sai số.</li>
      </ol>
      <div class="ss-warn">Ảnh chưa đạt chuẩn (mờ, tối, người bị xoay, máy nghiêng…) sẽ bị từ chối và app yêu cầu chụp lại.</div>
    </section>"""), unsafe_allow_html=True)

    st.markdown(_one_line(f"""
    <section class="ss-ky">
      <div>{spine_svg("ky")}</div>
      <div>
        <div class="ss-st" role="heading" aria-level="2">Đo gù lưng: dán 6 nhãn dọc cột sống</div>
        <p>Dán nhãn màu nhỏ lên mỏm gai T2, T4, T6, T8, T10, T12, chụp nghiêng 90°, rồi chạm lần lượt vào từng nhãn trên ảnh. App dựng đường cong qua 6 điểm và tính góc giữa hai tiếp tuyến ở T2 và T12.</p>
        <p>Mặc áo ba lỗ hở lưng ôm sát để dán trực tiếp lên da, không cần cởi áo. Cách tìm từng vị trí có trong mục 1 → Gù lưng.</p>
        <div class="ss-note">Đây là góc gù ước tính từ ảnh, không phải góc Cobb và chưa thay thế X-quang.</div>
      </div>
    </section>"""), unsafe_allow_html=True)

    st.markdown(_one_line("""
    <section class="ss-sec">
      <div class="ss-st" role="heading" aria-level="2">3 mục kiểm tra</div>
      <ul class="ss-modes">
        <li><b>1. Tư thế đứng</b>Đứng tĩnh (lệch vai, khom, đầu nhô ra trước), đeo cặp sách và gù lưng ước tính từ 6 nhãn</li>
        <li><b>2. Ngồi học</b>Tư thế khi ngồi</li>
        <li><b>3. Trước &amp; sau đeo cặp</b>So sánh hai ảnh của cùng một em</li>
      </ul>
    </section>"""), unsafe_allow_html=True)


def tab_method():
    st.subheader("Phương pháp & giới hạn")
    st.markdown(
        "**Cách đo.** Mô hình MediaPipe Pose tìm các mốc vai, hông, tai. Mốc do mô hình ước lượng; mọi chỉ số là góc tính trên ảnh 2D (độ) từ các mốc đó.\n\n"
        "- **Góc lệch vai:** góc giữa đường nối hai vai và phương ngang.\n"
        "- **Lệch thân so với hông:** góc đường (giữa hông → giữa vai) so với phương thẳng đứng, ảnh sau lưng/chính diện.\n"
        "- **Đầu nhô ra trước (góc tai–vai):** góc giữa phương ngang và đường vai → tai ở ảnh nghiêng. Gần với CVA nhưng **không phải CVA chuẩn**, "
        "vì không có mốc đốt sống C7 nên dùng mốc vai thay thế.\n"
        "- **Nghiêng thân:** góc đường hông → vai so với phương thẳng đứng, dương là nghiêng ra trước.\n\n"
        "- **Gù lưng (ước tính từ ảnh):** người dùng dán 6 nhãn lên mỏm gai (T2, T4, T6, T8, T10, T12; xác định bằng cách sờ và đếm từ C7) rồi chạm chọn từng nhãn trên ảnh nghiêng; app dựng đường cong bậc 3 qua 6 điểm rồi tính góc giữa hai tiếp tuyến ở T2 và T12. Đây là cách làm theo tinh thần phương pháp DIPA, **không phải góc Cobb** và chưa có ngưỡng chuẩn cho học sinh nên app chỉ hiển thị giá trị, không phân mức.\n\n"
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
    st.markdown("**Ngưỡng phân loại: ngưỡng tham khảo của nhóm, chưa có nguồn chuẩn và chưa được kiểm chứng lâm sàng.** Cần đối chiếu với đo thủ công để hiệu chỉnh.")
    st.markdown(md)
    st.markdown(
        "**Giới hạn.**\n"
        "- Ảnh 2D không thay thế X-quang hay khám lâm sàng, không chẩn đoán vẹo/gù cột sống. Hai chỉ số ở ảnh nghiêng (đầu nhô ra trước, nghiêng thân) **không phải là đo độ gù**. Vẹo cột sống là biến dạng 3D "
        "có xoay đốt sống; nghiệm pháp cúi người (Adams) với scoliometer mới đánh giá được góc xoay thân.\n"
        "- Mốc do mô hình ước lượng; áo rộng, tóc che vai, ảnh tối/mờ vẫn có thể làm lệch mốc. App chỉ cảnh báo được một phần.\n"
        "- Việc bù máy nghiêng chỉ làm được khi nền có đủ đường thẳng đứng; nếu không, hãy đặt máy cố định, ngang bằng.\n"
        "- Mỗi ảnh chỉ ghi lại một khoảnh khắc, chưa cho biết đó có phải thói quen lâu dài hay không.\n"
        "- App không ghi ảnh ra đĩa hay cơ sở dữ liệu; ảnh chỉ được xử lý tạm trong bộ nhớ."
    )


def main():
    logo = load_logo()
    st.set_page_config(page_title="S-Spine | AI đánh giá và phân tích tư thế học đường",
                       page_icon=logo if logo is not None else "🛡️", layout="wide")
    st.markdown(CSS_HOME, unsafe_allow_html=True)

    if get_pose() is None:
        st.error("Không khởi tạo được MediaPipe Pose. Hãy ghim phiên bản có `mp.solutions` trong requirements.txt "
                 "(ví dụ `mediapipe==0.10.14`) rồi triển khai lại.")
        st.stop()

    st.markdown(header_html(logo), unsafe_allow_html=True)

    t_guide, t_app, t_method = st.tabs(["Hướng dẫn chụp ảnh chuẩn", "Đánh giá & phân tích tư thế",
                                         "Phương pháp & giới hạn"])
    with t_guide:
        tab_guide()
    with t_app:
        tilt_mode = st.radio("Bù máy nghiêng:", ["Tự động (khuyến nghị)", "Nhập tay góc nghiêng máy", "Không bù"],
                             horizontal=True, key="tilt_mode")
        if tilt_mode.startswith("Tự"):
            auto_roll = True
        elif tilt_mode.startswith("Nhập"):
            auto_roll = float(st.number_input("Góc nghiêng máy (độ, dương = ảnh xoay thuận chiều kim đồng hồ)",
                                              -20.0, 20.0, 0.0, 0.5, key="manual_roll"))
        else:
            auto_roll = False
        mode = st.radio("Chọn mục tầm soát tư thế:", ["1. Tư thế đứng (đứng tĩnh, đeo cặp, gù lưng)",
                                                       "2. Tư thế ngồi học", "3. So sánh trước & sau khi đeo cặp"], key="main_mode")
        st.markdown("---")
        if mode.startswith("1"):
            sub = st.radio("Chức năng:", ["Đứng tĩnh", "Đeo cặp sách", "Gù lưng (ước tính từ ảnh)"],
                           horizontal=True, key="sub_mode_1")
            st.markdown("---")
            if sub.startswith("Đứng"):
                mode_standing(auto_roll)
            elif sub.startswith("Đeo"):
                mode_bag(auto_roll)
            else:
                mode_kyphosis()
        elif mode.startswith("2"):
            mode_sitting(auto_roll)
        else:
            mode_compare(auto_roll)
    with t_method:
        tab_method()

    st.markdown('<div class="disclaimer-box"><b>Lưu ý quan trọng:</b> Ứng dụng chỉ hỗ trợ tầm soát và mang tính chất tham khảo '
                'học đường, không thay thế chẩn đoán của bác sĩ chuyên khoa.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
