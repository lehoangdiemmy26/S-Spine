# -*- coding: utf-8 -*-
"""Id mốc MediaPipe Pose + nhãn tiếng Việt để báo lỗi."""

NOSE, L_EAR, R_EAR = 0, 7, 8
L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24
L_HEEL, R_HEEL, L_FOOT, R_FOOT = 29, 30, 31, 32

LM_NAMES = {
    0: "mũi",
    7: "tai trái",
    8: "tai phải",
    11: "vai trái",
    12: "vai phải",
    23: "hông trái",
    24: "hông phải",
}

LEVEL_TXT = {
    0: "🟢 Bình thường",
    1: "🟡 Cần theo dõi",
    2: "🔴 Nên kiểm tra thêm",
}
