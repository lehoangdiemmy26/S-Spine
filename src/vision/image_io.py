# -*- coding: utf-8 -*-
"""Đọc / resize ảnh. Tách riêng để sau này đổi sang đọc từ disk cũng dễ."""
import io

import numpy as np
from PIL import Image, ImageOps

from src.config.settings import MAX_SIDE


def load_image(data: bytes) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    w, h = img.size
    k = MAX_SIDE / float(max(w, h))
    if k < 1:
        img = img.resize((round(w * k), round(h * k)), Image.LANCZOS)
    arr = np.ascontiguousarray(np.array(img))
    return arr
