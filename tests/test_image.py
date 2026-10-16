# -*- coding: utf-8 -*-
"""Kiem tra bo chuan hoa anh sang PNG bytes (thuan, khong can Streamlit runtime)."""
import io

import numpy as np
from PIL import Image

from src.vision.encode import image_to_png_bytes


def _open(data):
    im = Image.open(io.BytesIO(data))
    im.load()
    return im


def test_rgb_ndarray_roundtrip():
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, (64, 48, 3)).astype(np.uint8)
    data = image_to_png_bytes(arr)
    assert isinstance(data, bytes) and len(data) > 0
    im = _open(data)
    assert im.mode == "RGB" and im.size == (48, 64)
    assert np.array_equal(np.array(im), arr)


def test_grayscale_and_float_normalized():
    gray = np.full((32, 24), 128, dtype=np.uint8)
    im = _open(image_to_png_bytes(gray))
    assert im.mode == "RGB" and im.size == (24, 32)
    fl = np.full((16, 16, 3), 300.0)  # vuot 255 -> clip
    assert np.array(_open(image_to_png_bytes(fl))).max() == 255


def test_rgba_drops_alpha_and_bytes_passthrough():
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    assert _open(image_to_png_bytes(rgba)).mode == "RGB"
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="PNG")
    raw = buf.getvalue()
    assert image_to_png_bytes(raw) == raw
    assert image_to_png_bytes(bytearray(raw)) == raw


def test_invalid_returns_none():
    assert image_to_png_bytes(np.zeros((10,), dtype=np.uint8)) is None
    assert image_to_png_bytes(np.zeros((0, 0, 3), dtype=np.uint8)) is None
    assert image_to_png_bytes(b"") is None
    assert image_to_png_bytes(b"khong-phai-anh") is None
    assert image_to_png_bytes(0) is None
