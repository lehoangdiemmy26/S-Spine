# -*- coding: utf-8 -*-
"""Các thông số chung của S-Spine.

Gom hết magic number về một chỗ để sau này hiệu chỉnh cho dễ.
Tất cả ngưỡng TH đều là ngưỡng THAM KHẢO của nhóm, chưa phải chuẩn y khoa.
"""

# ảnh đưa về tối đa 1280px trước khi cho qua model
MAX_SIDE = 1280
# visibility tối thiểu của mốc thì mới tính là "thấy rõ"
VIS_MIN = 0.6
# cho phép tải tối đa mấy ảnh / lần (lấy trung vị)
MAX_PHOTOS = 5
# dưới góc này thì coi như cân bằng, khỏi báo lệch
LEVEL_EPS = 0.3

# bù máy nghiêng
ROLL_MIN_FIX = 0.5   # nghiêng từ 0.5 độ mới cần bù
ROLL_MAX_FIX = 8.0   # quá 8 độ thì không tự bù nữa, báo chụp lại
SYM_WARN = 0.15      # mốc vai lệch so với viền cơ thể quá 15% -> cảnh báo
SYM_BLOCK = 0.30     # quá 30% -> ảnh không đạt, bắt chụp lại

# chất lượng ảnh (đo trên vùng có người)
DARK_MIN = 75        # mean gray thấp hơn là tối
BLUR_P99_MIN = 20.0  # p99 |Laplacian| thấp hơn là mờ. Cần hiệu chỉnh bằng ảnh thật
PERSON_MIN_FRAC = 0.30

# phát hiện người bị xoay / chưa đủ 90 độ
ROT_WARN_FRONT = 0.60
ROT_WARN_SIDE = 0.35
SPREAD_WARN = {"front": 2.0, "side": 8.0}

# Mấy câu này xuất hiện là ảnh KHÔNG đạt -> chặn, không phân tích
BLOCKING = (
    "Ảnh khá tối",
    "Ảnh bị mờ",
    "Độ phân giải thấp",
    "Người chiếm quá ít",
    "Người có vẻ bị xoay",
    "Mốc vai không đối xứng",
    "Thân người có vẻ chưa nghiêng 90°",
    "Máy có vẻ nghiêng",
)

# Ngưỡng (t1, t2): < t1 xanh, <= t2 vàng, còn lại đỏ
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

# góc tai-vai. Mượn ý tưởng CVA nhưng dùng mốc vai thay C7 nên KHÔNG tương đương CVA chuẩn
EAR_OK, EAR_WATCH = 50.0, 40.0
