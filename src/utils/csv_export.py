# -*- coding: utf-8 -*-
"""Xuất chỉ số ra CSV để đối chiếu đo thủ công."""
import io
import csv

from src.config.landmarks import LEVEL_TXT
from src.domain.levels import reliability


def csv_bytes(label, res, extra=None):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["muc", "chi_so", "gia_tri"])
    for k, v in res.metrics.items():
        w.writerow([label, k, "" if v is None else v])
    w.writerow([label, "muc_do", LEVEL_TXT[res.level]])
    w.writerow([label, "do_tin_cay", reliability(res.notes + (extra or []))])
    return buf.getvalue().encode("utf-8-sig")
