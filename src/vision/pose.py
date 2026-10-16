# -*- coding: utf-8 -*-
"""Khởi tạo MediaPipe Pose + detect mốc.

Giữ nguyên cách làm cũ: thử model_complexity 2 rồi rớt xuống 1,
có lock vì MediaPipe không thread-safe.
"""
import threading
from types import SimpleNamespace

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
