# -*- coding: utf-8 -*-
"""Ước tính góc gù từ 6 nhãn dán trên lưng (T2 -> T12).

Cách làm của nhóm theo tinh thần DIPA: dựng đường cong x = f(y) bậc 3
qua 6 điểm rồi lấy góc giữa 2 tiếp tuyến ở T2 và T12.
Đây KHÔNG phải góc Cobb, không thay thế X-quang.
"""
import math

import numpy as np

from src.config.landmarks import L_HIP, L_SH, R_HIP, R_SH
from src.config.settings import ROT_WARN_SIDE, VIS_MIN

KY_LABELS = [
    "T2 (đếm xuống 2 mỏm gai dưới C7; C7 là mỏm nhô rõ nhất ở gốc cổ)",
    "T4",
    "T6",
    "T8",
    "T10",
    "T12 (cuối lồng ngực, ngang đáy xương bả vai là mốc tham khảo)",
]
KY_N = len(KY_LABELS)
KY_DISPLAY_W = 560

# dao động góc khi bỏ từng điểm giữa mà vượt mức này là chạm chưa ổn
KY_STAB_MAX = 12.0

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


def kyphosis_angle(pts):
    ys = np.array([p[1] for p in pts], dtype=float)
    xs = np.array([p[0] for p in pts], dtype=float)
    if not np.all(np.diff(ys) > 0):
        return None
    c = np.polyfit(ys, xs, 3)
    d = np.polyder(c)
    a_top = math.degrees(math.atan(np.polyval(d, ys[0])))
    a_bot = math.degrees(math.atan(np.polyval(d, ys[-1])))
    return abs(a_top - a_bot)


def kyphosis_stability(pts):
    """Bỏ lần lượt từng điểm giữa rồi tính lại, xem góc nhảy bao nhiêu."""
    angs = []
    for k in range(1, len(pts) - 1):
        a = kyphosis_angle([p for i, p in enumerate(pts) if i != k])
        if a is not None:
            angs.append(a)
    return (max(angs) - min(angs)) if angs else 0.0


def side_photo_note(img):
    """Ảnh gù phải là nghiêng 90 độ: hai vai cách xa nhau là chưa đạt."""
    try:
        from src.vision.pose import detect

        lms, _ = detect(img)
    except Exception:
        return None
    if lms is None:
        return None
    h, w = img.shape[:2]
    a, b, c, d = lms[L_SH], lms[R_SH], lms[L_HIP], lms[R_HIP]
    if min(a.visibility, b.visibility, c.visibility, d.visibility) < VIS_MIN:
        return None
    ms = ((a.x + b.x) / 2.0 * w, (a.y + b.y) / 2.0 * h)
    mh = ((c.x + d.x) / 2.0 * w, (c.y + d.y) / 2.0 * h)
    torso = math.hypot(ms[0] - mh[0], ms[1] - mh[1])
    if torso > 1 and abs(a.x - b.x) * w / torso > ROT_WARN_SIDE:
        return "Ảnh có vẻ chưa chụp nghiêng 90° (hai vai còn cách xa nhau). Góc gù có thể bị lệch, nên chụp lại."
    return None


def ky_quality_notes(pts, img, ang):
    notes = []
    p = np.array(pts, dtype=float)
    seg = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
    if seg.min() < 1 or seg.max() / seg.min() > 2.0:
        notes.append(
            "Khoảng cách giữa các nhãn không đều (các nhãn dán cách đều nhau). "
            "Hãy kiểm tra đã chạm đúng tâm từng nhãn chưa."
        )
    dx, dy = p[-1, 0] - p[0, 0], p[-1, 1] - p[0, 1]
    if abs(math.degrees(math.atan2(dx, dy))) > 30:
        notes.append(
            "Đường nối T2 → T12 nghiêng nhiều so với phương thẳng đứng: người đứng nghiêng hoặc ảnh bị xoay."
        )
    spread = kyphosis_stability(pts)
    if spread > KY_STAB_MAX:
        notes.append(
            f"Góc thay đổi tới {spread:.0f}° khi bỏ lần lượt từng nhãn giữa, nghĩa là có điểm chạm lệch. "
            "Hãy chạm lại các nhãn cho đúng tâm."
        )
    n = side_photo_note(img)
    if n:
        notes.append(n)
    return notes, spread
