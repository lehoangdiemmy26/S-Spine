# -*- coding: utf-8 -*-
"""Mục đeo cặp: tư thế khi đeo + tỉ lệ cặp/cân nặng."""
import streamlit as st

from src.config.settings import TH
from src.domain.levels import level_of
from src.ui.recommendations import show_recommendations
from src.ui.theme import page_head
from src.ui.widgets import (
    clothing_check,
    image_inputs,
    run_and_show,
    show_level,
    source_selector,
    view_selector,
)


def mode_bag(auto_roll):
    page_head("Đeo cặp sách", "Xem tư thế khi đang đeo cặp và tỷ lệ cặp/cân nặng.")
    st.info(
        "Mục này chỉ cho biết tư thế KHI đeo cặp. Muốn biết cặp làm thay đổi tư thế bao nhiêu, "
        "hãy dùng mục “So sánh trước & sau khi đeo cặp”."
    )
    c1, c2 = st.columns(2)
    bag = c1.number_input("Trọng lượng cặp (kg) - do bạn nhập", 0.0, 30.0, 0.0, 0.5)
    body = c2.number_input("Cân nặng của em (kg) - cần nhập để tính tỷ lệ cặp/cân nặng", 0.0, 150.0, 0.0, 1.0)
    if bag > 0 and body <= 0:
        st.warning("Chưa nhập cân nặng nên chưa tính được tỷ lệ cặp/cân nặng (chỉ số khuyến nghị ≤ 10-15%).")
    st.caption("Cặp có thể che mốc vai/hông. Nếu mốc trên ảnh kết quả đặt sai chỗ, hãy chụp lại.")

    worst = 0
    if bag > 0 and body > 0:
        pct = bag / body * 100.0
        lv = level_of(pct, TH["bag_ratio_pct"])
        worst = lv
        show_level(
            lv,
            f"Cặp nặng {bag:g} kg ≈ {pct:.0f}% cân nặng "
            "(khuyến nghị thường gặp: khoảng 10%, không nên quá 15%).",
        )

    src = source_selector("m2")
    view = view_selector("m2")
    extra = clothing_check("m2")
    ca, cb = st.columns(2)
    with ca:
        f_front = image_inputs("Ảnh đeo cặp (sau lưng/chính diện)", "m2_front", src, multiple=False)
    with cb:
        f_side = image_inputs("Ảnh đeo cặp (nghiêng bên hông)", "m2_side", src, multiple=False)

    shown = False
    if f_front:
        st.markdown("#### Ảnh sau lưng / chính diện")
        r = run_and_show(f_front, "front", view, "stand", auto_roll,
                         "Đeo cặp - lệch vai", "cap_lech_vai", extra)
        if r:
            shown, worst = True, max(worst, r.level)
    if f_side:
        st.markdown("#### Ảnh nghiêng bên hông")
        r = run_and_show(f_side, "side", "back", "stand", auto_roll,
                         "Đeo cặp - nghiêng", "cap_nghieng")
        if r:
            shown, worst = True, max(worst, r.level)
    if shown:
        show_recommendations(worst)
    elif not f_front and not f_side:
        st.info("Hãy tải lên hoặc chụp ít nhất một ảnh để bắt đầu.")
