# -*- coding: utf-8 -*-
"""Mục đứng tĩnh: lệch vai + khom/đầu nhô."""
import streamlit as st

from src.config.settings import MAX_PHOTOS
from src.ui.recommendations import show_recommendations
from src.ui.theme import page_head
from src.ui.widgets import (
    clothing_check,
    image_inputs,
    run_and_show,
    source_selector,
    view_selector,
)


def mode_standing(auto_roll):
    page_head("Đứng tĩnh", "Kiểm tra lệch vai, tư thế khom và đầu nhô ra trước khi đứng.")
    sub = st.selectbox(
        "Chọn góc chụp:",
        ["Sau lưng / Chính diện (kiểm tra lệch vai)",
         "Nghiêng bên hông (kiểm tra tư thế khom & đầu nhô ra trước)"],
    )
    st.caption(
        f"Có thể tải tối đa {MAX_PHOTOS} ảnh của cùng một người (chụp rời nhau vài giây); "
        "app lấy trung vị để giảm sai số."
    )
    src = source_selector("m1")
    if sub.startswith("Sau lưng"):
        view = view_selector("m1")
        extra = clothing_check("m1")
        files = image_inputs("Ảnh sau lưng/chính diện:", "m1_front", src)
        if files:
            res = run_and_show(files, "front", view, "stand", auto_roll,
                               "Đứng - lệch vai", "dung_lech_vai", extra)
            if res:
                show_recommendations(res.level)
    else:
        files = image_inputs("Ảnh nghiêng bên hông (90°):", "m1_side", src)
        if files:
            res = run_and_show(files, "side", "back", "stand", auto_roll,
                               "Đứng - nghiêng", "dung_nghieng")
            if res:
                show_recommendations(res.level)
