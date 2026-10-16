# -*- coding: utf-8 -*-
"""Mục ngồi học: nhận cả góc sau lưng + nghiêng."""
import streamlit as st

from src.ui.recommendations import show_recommendations
from src.ui.theme import page_head
from src.ui.widgets import clothing_check, image_inputs, run_and_show, source_selector


def mode_sitting(auto_roll):
    page_head("Tư thế ngồi học", "Kiểm tra tư thế khi ngồi học từ ảnh chụp.")
    st.info(
        "Có thể dùng một hoặc cả hai góc chụp: sau lưng (lệch vai) và nghiêng bên hông "
        "(cúi đầu, khom, đầu nhô ra trước). Tư thế ngồi thay đổi liên tục, nên hãy chụp vài ảnh "
        "ở các thời điểm khác nhau."
    )
    src = source_selector("m3")
    extra = clothing_check("m3")
    ca, cb = st.columns(2)
    with ca:
        f_back = image_inputs("Ảnh ngồi từ phía sau lưng", "m3_back", src)
    with cb:
        f_side = image_inputs("Ảnh ngồi nghiêng bên hông (90°)", "m3_side", src)

    worst, shown = 0, False
    if f_back:
        st.markdown("#### Góc sau lưng")
        r = run_and_show(f_back, "front", "back", "sit", auto_roll,
                         "Ngồi - sau lưng", "ngoi_sau_lung", extra)
        if r:
            shown, worst = True, max(worst, r.level)
    if f_side:
        st.markdown("#### Góc nghiêng")
        r = run_and_show(f_side, "side", "back", "sit", auto_roll,
                         "Ngồi - nghiêng", "ngoi_nghieng")
        if r:
            shown, worst = True, max(worst, r.level)
    if shown:
        show_recommendations(worst)
    elif not f_back and not f_side:
        st.info("Hãy tải lên hoặc chụp ít nhất một ảnh để bắt đầu.")
