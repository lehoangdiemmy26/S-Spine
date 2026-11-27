# -*- coding: utf-8 -*-
"""Struct chứa kết quả một lần đo."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Result:
    image: np.ndarray
    metrics: dict
    level: int          # 0 xanh, 1 vàng, 2 đỏ
    summary: str
    notes: list = field(default_factory=list)  # cảnh báo làm giảm độ tin cậy
    info: list = field(default_factory=list)   # thông tin thêm, không trừ điểm
