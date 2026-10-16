# -*- coding: utf-8 -*-
"""Mục so sánh trước/sau khi đeo cặp (2 ảnh cùng một em)."""
import streamlit as st

from src.analysis.pipeline import analyze_file
from src.config.settings import TH
from src.domain.levels import level_of, reliability
from src.ui.theme import page_head
from src.ui.widgets import (
    clothing_check,
    image_inputs,
    show_image,
    show_level,
    show_problems,
    source_selector,
    view_selector,
)


def mode_compare(auto_roll):
    page_head(
        "So sánh trước & sau khi đeo cặp",
        "Hai ảnh của cùng một em để xem cặp làm tư thế thay đổi thế nào.",
    )
    st.info("Chụp **cùng vị trí, cùng khoảng cách, cùng chiều cao máy** cho cả hai ảnh để so sánh đáng tin.")
    sub = st.selectbox(
        "Chọn góc chụp so sánh:",
        ["Sau lưng / Chính diện (lệch vai)", "Nghiêng bên hông (tư thế khom, đầu nhô ra trước)"],
    )
    src = source_selector("m4")
    is_front = sub.startswith("Sau lưng")
    view = view_selector("m4") if is_front else "back"
    ca, cb = st.columns(2)
    with ca:
        f1 = image_inputs("1. Ảnh KHÔNG đeo cặp", "m4_norm", src, multiple=False)
    with cb:
        f2 = image_inputs("2. Ảnh KHI đeo cặp", "m4_pack", src, multiple=False)
    bag = st.number_input("Trọng lượng cặp (kg) - do bạn nhập", 0.0, 30.0, 0.0, 0.5, key="m4_bag")
    extra = clothing_check("m4")
    st.caption("Cặp có thể che mốc vai/hông. Nếu mốc trên ảnh kết quả đặt sai chỗ, hãy chụp lại.")
    if not (f1 and f2):
        return

    kind = "front" if is_front else "side"
    r1, p1 = analyze_file(f1[0], kind, view, "stand", auto_roll)
    r2, p2 = analyze_file(f2[0], kind, view, "stand", auto_roll)
    if r1 is None or r2 is None:
        if r1 is None:
            st.markdown("**Ảnh 1 (không đeo cặp):**")
            show_problems(p1)
        if r2 is None:
            st.markdown("**Ảnh 2 (đeo cặp):**")
            show_problems(p2)
        return

    ca, cb = st.columns(2)
    with ca:
        show_image(r1.image, "Ảnh 1: không đeo cặp")
    with cb:
        show_image(r2.image, "Ảnh 2: đeo cặp")

    if is_front:
        k = "vai_phai_thap_hon_co_dau_do"
        d = r2.metrics[k] - r1.metrics[k]
        lv = level_of(abs(d), TH["delta_tilt"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Lệch vai ban đầu", f"{r1.metrics['goc_lech_vai_do']:.1f}°")
        c2.metric(
            "Lệch vai khi đeo", f"{r2.metrics['goc_lech_vai_do']:.1f}°",
            delta=f"{r2.metrics['goc_lech_vai_do'] - r1.metrics['goc_lech_vai_do']:+.1f}°",
            delta_color="inverse",
        )
        c3.metric("Cặp (bạn nhập)", f"{bag:g} kg")
        if lv == 0:
            show_level(0, f"Cặp không làm thay đổi đáng kể độ cân bằng vai (thay đổi ≈ {abs(d):.1f}°).")
        else:
            side = "vai phải" if d > 0 else "vai trái"
            show_level(
                lv,
                f"Khi đeo cặp, {side} hạ thấp thêm ≈ {abs(d):.1f}°. "
                "Hãy chỉnh hai quai cho lực đều hai bên, cặp áp sát giữa lưng, đồ nặng đặt sát lưng rồi chụp lại.",
            )
    else:
        d_lean = r2.metrics["nghieng_than_do"] - r1.metrics["nghieng_than_do"]
        d_ear = r2.metrics["goc_tai_vai_do"] - r1.metrics["goc_tai_vai_do"]
        lv = max(level_of(d_lean, TH["delta_trunk"]), level_of(-d_ear, TH["delta_ear_drop"]))
        c1, c2, c3 = st.columns(3)
        c1.metric("Nghiêng thân khi đeo", f"{r2.metrics['nghieng_than_do']:+.0f}°",
                  delta=f"{d_lean:+.1f}°", delta_color="inverse")
        c2.metric("Đầu nhô ra trước (tai–vai) khi đeo", f"{r2.metrics['goc_tai_vai_do']:.0f}°",
                  delta=f"{d_ear:+.1f}°")
        c3.metric("Cặp (bạn nhập)", f"{bag:g} kg")
        if lv == 0:
            show_level(0, "Cặp không làm thân ngả ra trước hay đầu nhô ra trước đáng kể.")
        else:
            show_level(
                lv,
                "Cặp kéo thân trên ngả ra trước và/hoặc đầu nhô ra trước. Hãy siết quai cho cặp áp sát lưng, "
                "dùng đai ngực/đai bụng nếu có và giảm bớt đồ nặng.",
            )
    for r, name in ((r1, "Ảnh 1"), (r2, "Ảnh 2")):
        nn = r.notes + extra
        if reliability(nn) != "Tốt":
            st.caption(f"{name} - độ tin cậy {reliability(nn)}: {'; '.join(nn)}")
