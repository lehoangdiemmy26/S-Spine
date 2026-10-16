# -*- coding: utf-8 -*-
"""Mục gù lưng: dán 6 nhãn rồi chạm chọn trên ảnh."""
import cv2
import streamlit as st
from PIL import Image

from src.analysis.kyphosis import (
    KY_DISPLAY_W,
    KY_GUIDE,
    KY_LABELS,
    KY_N,
    ky_quality_notes,
    kyphosis_angle,
)
from src.ui.theme import page_head
from src.ui.widgets import clothing_check, image_inputs, source_selector
from src.vision.image_io import load_image
from src.vision.overlay import put_text


def mode_kyphosis():
    page_head("Gù lưng (ước tính từ ảnh)", "Dán 6 nhãn lên lưng, chụp nghiêng rồi chạm chọn từng nhãn trên ảnh.")
    with st.expander("📌 Hướng dẫn dán 6 nhãn lên lưng (T2, T4, T6, T8, T10, T12)", expanded=False):
        st.markdown(KY_GUIDE)
    _ky_measure()


def _ky_measure():
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
    except Exception:
        st.error(
            "Thiếu thư viện chạm chọn điểm. Hãy thêm `streamlit-image-coordinates` "
            "vào requirements.txt rồi triển khai lại."
        )
        return

    st.info(
        "**Cách làm:** dán 6 nhãn màu nhỏ lên mỏm gai (T2, T4, T6, T8, T10, T12; xác định bằng cách sờ và đếm từ C7), "
        "chụp ảnh nghiêng 90° rồi chạm lần lượt vào từng nhãn theo thứ tự từ trên xuống "
        "(đường cong bậc 3 qua 6 điểm, tiếp tuyến tại T2 và T12). Kết quả là **góc gù ước tính từ ảnh**, "
        "không phải góc Cobb và không thay thế X-quang."
    )
    src = source_selector("m5")
    extra = clothing_check("m5")
    files = image_inputs("Ảnh nghiêng có nhãn đánh dấu trên lưng:", "m5_side", src, multiple=False)
    if not files:
        return

    data = files[0].getvalue()
    sig = f"{len(data)}-{hash(data[:2048])}"
    if st.session_state.get("ky_sig") != sig:
        st.session_state.update(ky_sig=sig, ky_pts=[], ky_last=None)

    img = load_image(data)
    h, w = img.shape[:2]
    scale = w / float(KY_DISPLAY_W)
    pts = st.session_state["ky_pts"]
    disp = cv2.resize(img, (KY_DISPLAY_W, int(round(h / scale))))
    for i, (x, y) in enumerate(pts):
        c = (int(x / scale), int(y / scale))
        cv2.circle(disp, c, 6, (255, 0, 0), -1)
        put_text(disp, str(i + 1), (c[0] + 8, c[1] - 6), 0.6, (255, 255, 0))

    if len(pts) < KY_N:
        st.markdown(f"**Chạm vào: {len(pts) + 1}/{KY_N} - {KY_LABELS[len(pts)]}**")
    val = streamlit_image_coordinates(Image.fromarray(disp), key=f"ky_img_{sig}_{len(pts)}")
    if val and len(pts) < KY_N:
        key = (val["x"], val["y"], len(pts))
        if st.session_state.get("ky_last") != key:
            st.session_state["ky_last"] = key
            k = w / float(val.get("width") or KY_DISPLAY_W)
            pts.append((val["x"] * k, val["y"] * k))
            st.rerun()

    c1, c2 = st.columns(2)
    if c1.button("↩️ Bỏ điểm vừa chọn", disabled=not pts):
        pts.pop()
        st.session_state["ky_last"] = None
        st.rerun()
    if c2.button("🗑️ Chọn lại từ đầu", disabled=not pts):
        st.session_state.update(ky_pts=[], ky_last=None)
        st.rerun()
    if len(pts) < KY_N:
        return

    ang = kyphosis_angle(pts)
    if ang is None:
        st.error("Thứ tự điểm chưa đúng (phải đi từ T2 xuống T12, từ trên xuống dưới). Hãy chọn lại.")
        return

    qnotes, spread = ky_quality_notes(pts, img, ang)
    st.metric(
        "Góc gù ước tính từ ảnh", f"{ang:.0f}°",
        help="Góc giữa hai tiếp tuyến của đường cong qua 6 nhãn, tại T2 và T12. Không phải góc Cobb.",
    )
    st.caption(f"Độ ổn định: góc dao động {spread:.0f}° khi bỏ lần lượt từng nhãn giữa (càng nhỏ càng đáng tin).")
    if qnotes:
        st.error("**Độ tin cậy thấp, nên chụp hoặc chạm lại:**\n\n" + "\n".join(f"- {n}" for n in qnotes))
    else:
        st.success("Các kiểm tra chất lượng đều đạt (nhãn cách đều, người đứng thẳng, góc ổn định khi bỏ từng nhãn).")
    st.warning(
        "Chưa có ngưỡng chuẩn cho lứa tuổi học sinh nên app **không phân mức** bình thường/bất thường. "
        "Phương pháp ảnh có sai số so với X-quang (nghiên cứu DIPA ở người lớn: ±4°, "
        "nhưng bản này của nhóm chưa kiểm chứng). Nếu con số lớn hoặc có lo ngại, hãy đến cơ sở y tế chuyên khoa để được khám."
    )
    if extra:
        st.caption("• " + extra[0])
    st.caption(
        "Kết quả phụ thuộc vào việc bạn chạm đúng vị trí nhãn; "
        "nên đối chiếu với người có chuyên môn đo bằng thước dẻo/kyphometer."
    )
