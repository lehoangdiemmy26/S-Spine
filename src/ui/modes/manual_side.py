# -*- coding: utf-8 -*-
"""Tự chạm 3 mốc (tai, vai, hông) trên ảnh nghiêng khi tự động lấy mốc sai.

Chế độ này không bù máy nghiêng, nên cần đặt máy ngang bằng khi chụp.
"""
import cv2
import streamlit as st
from PIL import Image

from src.analysis.side import classify_side
from src.ui.widgets import show_level
from src.vision.geometry import ear_shoulder_angle, trunk_lean
from src.vision.image_io import load_image
from src.vision.overlay import put_text

SIDE_PTS = [
    "Tai (giữa lỗ tai)",
    "Vai (mỏm cùng vai, chỗ nhô tròn trên đầu xương cánh tay)",
    "Hông (mấu chuyển lớn, điểm nhô ngoài xương đùi)",
]
DISPLAY_W = 560


def manual_side_measure(f, posture):
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
    except Exception:
        st.error(
            "Thiếu thư viện chạm chọn điểm. Hãy thêm `streamlit-image-coordinates` "
            "vào requirements.txt rồi triển khai lại."
        )
        return
    data = f.getvalue()
    sig = f"{len(data)}-{hash(data[:2048])}"
    if st.session_state.get("ms_sig") != sig:
        st.session_state.update(ms_sig=sig, ms_pts=[], ms_last=None)
    pts = st.session_state["ms_pts"]
    img = load_image(data)
    h, w = img.shape[:2]
    scale = w / float(DISPLAY_W)
    disp = cv2.resize(img, (DISPLAY_W, int(round(h / scale))))
    for i, (x, y) in enumerate(pts):
        c = (int(x / scale), int(y / scale))
        cv2.circle(disp, c, 6, (255, 0, 0), -1)
        put_text(disp, str(i + 1), (c[0] + 8, c[1] - 6), 0.6, (255, 255, 0))

    face = st.radio("Mặt người quay về phía", ["Phải ảnh", "Trái ảnh"], horizontal=True, key="ms_face")
    facing = 1 if face == "Phải ảnh" else -1
    if len(pts) < 3:
        st.markdown(f"**Chạm vào: {len(pts) + 1}/3 - {SIDE_PTS[len(pts)]}**")
    val = streamlit_image_coordinates(Image.fromarray(disp), key=f"ms_img_{sig}_{len(pts)}")
    if val and len(pts) < 3:
        key = (val["x"], val["y"], len(pts))
        if st.session_state.get("ms_last") != key:
            st.session_state["ms_last"] = key
            k = w / float(val.get("width") or DISPLAY_W)
            pts.append((val["x"] * k, val["y"] * k))
            st.rerun()

    c1, c2 = st.columns(2)
    if c1.button("↩️ Bỏ điểm vừa chọn", disabled=not pts, key="ms_undo"):
        pts.pop()
        st.session_state["ms_last"] = None
        st.rerun()
    if c2.button("🗑️ Chọn lại từ đầu", disabled=not pts, key="ms_reset"):
        st.session_state.update(ms_pts=[], ms_last=None)
        st.rerun()
    if len(pts) < 3:
        return

    ear, sh, hip = pts
    if hip[1] <= sh[1]:
        st.error("Hông phải nằm thấp hơn vai trong ảnh. Hãy chọn lại.")
        return
    ang, lean = ear_shoulder_angle(ear, sh, facing), trunk_lean(sh, hip, facing)
    level, summary = classify_side(ang, lean, posture)
    m1, m2 = st.columns(2)
    m1.metric("Góc tai-vai", f"{ang:.0f}°")
    m2.metric("Thân nghiêng ra trước", f"{lean:+.0f}°")
    show_level(level, summary)
    st.caption("Chế độ chọn mốc bằng tay không bù máy nghiêng: hãy đặt máy ngang bằng khi chụp.")
