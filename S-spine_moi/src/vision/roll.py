# -*- coding: utf-8 -*-
"""Ước lượng máy nghiêng từ các đường thẳng đứng của nền (khung cửa, cạnh tường...)."""
import math

import cv2
import numpy as np


def estimate_roll(img, lms=None):
    """Trả về góc nghiêng máy (độ, dương = ảnh xoay thuận chiều kim đồng hồ) hoặc None."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    k = 800.0 / max(g.shape)
    if k < 1:
        g = cv2.resize(g, None, fx=k, fy=k)
    h, w = g.shape
    edges = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 30, 100)
    if lms:
        # bỏ vùng có người, viền người không phải đường chuẩn
        xs = [p.x for p in lms if p.visibility > 0.3]
        if xs:
            x0 = int(max(0, min(xs) - 0.05) * w)
            x1 = int(min(1, max(xs) + 0.05) * w)
            edges[:, x0:x1] = 0
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 360, 35,
        minLineLength=int(0.10 * h), maxLineGap=int(0.02 * h),
    )
    leans = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            a = math.degrees(math.atan2(x1 - x2, y2 - y1))
            if abs(a) <= 10:
                leans.append(a)
    if len(leans) < 3:
        return None
    med = float(np.median(leans))
    spread = float(np.median(np.abs(np.array(leans) - med)))
    if spread > 1.2:  # các đường không thống nhất
        return None
    return med


def rotate_image(img, angle):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
