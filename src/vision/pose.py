# -*- coding: utf-8 -*-
"""Khởi tạo MediaPipe Pose + detect mốc.

Giữ nguyên cách làm cũ: thử model_complexity 2 rồi rớt xuống 1,
có lock vì MediaPipe không thread-safe.
"""
import threading
from types import SimpleNamespace

import numpy as np
import streamlit as st
import mediapipe as mp

_lock = threading.Lock()

# giữ lại lỗi cuối để màn hình chính hiện chẩn đoán chi tiết
_LAST_ERROR = {"msg": None}


def last_pose_error():
    return _LAST_ERROR["msg"]


@st.cache_resource
def get_pose():
    # bản mediapipe mới bỏ mp.solutions -> phải ghim bản cũ (0.10.14)
    if not hasattr(mp, "solutions"):
        _LAST_ERROR["msg"] = (
            f"mediapipe {getattr(mp, '__version__', '?')} không có mp.solutions. "
            "Cần ghim mediapipe==0.10.14."
        )
        return None
    errs = []
    for cx in (2, 1):
        try:
            pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=cx,
                enable_segmentation=True,
                min_detection_confidence=0.5,
            )
            _LAST_ERROR["msg"] = None
            return pose
        except Exception as e:
            errs.append(f"complexity={cx}: {type(e).__name__}: {e}")
            continue
    _LAST_ERROR["msg"] = " | ".join(errs) if errs else "không rõ nguyên nhân"
    return None


def detect(img):
    with _lock:
        r = get_pose().process(img)
    if not r.pose_landmarks:
        return None, None
    lms = [
        SimpleNamespace(x=p.x, y=p.y, visibility=p.visibility)
        for p in r.pose_landmarks.landmark
    ]
    return lms, r.segmentation_mask


def detect_refined(img):
    """Dò lần 1 trên cả ảnh, rồi dò lại trên vùng cắt quanh người.

    Người nhỏ trong khung (hay gặp ở ảnh ngồi chụp từ xa) sẽ được mô hình đo chính xác hơn.
    Mask phân vùng vẫn lấy từ lần dò đầu (đúng toạ độ ảnh gốc).
    """
    lms, mask = detect(img)
    if lms is None:
        return None, None
    h, w = img.shape[:2]
    pts = [(p.x, p.y) for p in lms if p.visibility >= 0.3 and 0.0 <= p.x <= 1.0 and 0.0 <= p.y <= 1.0]
    if len(pts) < 4:
        return lms, mask
    xs, ys = zip(*pts)
    m = 0.12
    x0, x1 = int(max(0, min(xs) - m) * w), int(min(1, max(xs) + m) * w)
    y0, y1 = int(max(0, min(ys) - m) * h), int(min(1, max(ys) + m) * h)
    if (x1 - x0) * (y1 - y0) > 0.7 * w * h or x1 - x0 < 100 or y1 - y0 < 200:
        return lms, mask   # người đã chiếm gần hết khung hoặc vùng cắt quá nhỏ: không cần dò lại
    l2, _ = detect(np.ascontiguousarray(img[y0:y1, x0:x1]))
    if l2 is None:
        return lms, mask
    cw, ch = x1 - x0, y1 - y0
    for p in l2:
        p.x, p.y = (x0 + p.x * cw) / w, (y0 + p.y * ch) / h
    return l2, mask
