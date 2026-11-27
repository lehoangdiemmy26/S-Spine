# -*- coding: utf-8 -*-
"""Vẽ mốc + góc lên ảnh để người dùng đối chiếu."""
import cv2


def ip(p):
    return int(round(p[0])), int(round(p[1]))


def put_text(img, s, org, fs, color):
    # viền đen cho dễ đọc trên nền sáng/tối
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, fs, color, 2, cv2.LINE_AA)


def sizes(w):
    t = max(2, int(round(w / 300)))
    r = max(4, int(round(w / 110)))
    fs = max(0.5, w / 1100)
    return t, r, fs
