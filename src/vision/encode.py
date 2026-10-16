# -*- coding: utf-8 -*-
"""Chuan hoa anh thanh PNG bytes truoc khi dua cho st.image.

Ly do: duong truyen ndarray cua st.image thay doi theo tung phien ban
Streamlit (tham so width/use_container_width/use_column_width, cach
marshalling), tung gay anh phan tich bi vo tren moi truong cloud.
PNG bytes la dinh dang on dinh nhat cho st.image tren moi phien ban.
"""
import io

import numpy as np
from PIL import Image


def image_to_png_bytes(img):
    """Tra ve bytes PNG, hoac None neu du lieu anh khong hop le."""
    try:
        if isinstance(img, (bytes, bytearray)):
            data = bytes(img)
            if not data:
                return None
            # kiem tra nhanh co dung la anh doc duoc khong
            Image.open(io.BytesIO(data)).verify()
            return data
        if isinstance(img, Image.Image):
            arr = np.array(img.convert("RGB"))
        else:
            arr = np.ascontiguousarray(img)
        if arr.ndim == 2:  # anh xam -> RGB
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            return None
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        arr = np.ascontiguousarray(arr[:, :, :3])
        if arr.shape[0] < 1 or arr.shape[1] < 1:
            return None
        buf = io.BytesIO()
        Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
        return buf.getvalue() or None
    except Exception:
        return None
