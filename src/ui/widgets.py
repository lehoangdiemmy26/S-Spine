# -*- coding: utf-8 -*-
"""Mấy widget nhập liệu + hiển thị kết quả dùng chung cho các mục."""
import streamlit as st

from src.analysis.pipeline import run_many
from src.config.landmarks import LEVEL_TXT
from src.domain.levels import reliability
from src.ui.theme import eyebrow
from src.utils.csv_export import csv_bytes
from src.vision.encode import image_to_png_bytes


def show_image(img, caption=None):
    # encode sang PNG bytes truoc: dinh dang on dinh nhat cho st.image
    # tren moi phien ban Streamlit / moi truong (local, cloud).
    data = image_to_png_bytes(img)
    if data is None:
        st.error(
            "Không hiển thị được ảnh phân tích (dữ liệu ảnh không hợp lệ). "
            "Hãy chụp/tải lại ảnh khác."
        )
        return
    # chi rẽ nhánh theo TypeError (tham số width chưa có ở bản cũ);
    # lỗi khác phải hiện ra chứ không nuốt, để còn chẩn đoán được.
    try:
        st.image(data, caption=caption, width="stretch")
    except TypeError:
        st.image(data, caption=caption)


def image_inputs(label, key, source, multiple=True):
    if source == "Tải ảnh lên":
        got = st.file_uploader(label, type=["jpg", "jpeg", "png"], key=f"u_{key}",
                                accept_multiple_files=multiple)
        if not multiple:
            return [got] if got else []
        return list(got or [])
    f = st.camera_input(label, key=f"c_{key}")
    return [f] if f else []


def source_selector(key):
    s = st.radio("Chọn phương thức đầu vào:", ["Tải ảnh lên", "Chụp bằng Camera"],
                 key=f"src_{key}", horizontal=True,
                 help="Tải ảnh có sẵn hoặc chụp trực tiếp bằng camera.")
    if s == "Chụp bằng Camera":
        st.caption("Nên dùng camera sau. Camera trước của một số điện thoại lật gương ảnh, làm đảo trái/phải.")
    return s


def view_selector(key):
    v = st.radio("Ảnh được chụp từ phía:", ["Sau lưng", "Phía trước mặt"],
                 key=f"view_{key}", horizontal=True,
                 help="Chọn đúng hướng chụp, app dùng để xác định vai trái/phải.")
    return "back" if v == "Sau lưng" else "front"


def clothing_check(key):
    ok = st.checkbox(
        "Đã mặc áo ôm sát, tóc gọn không che vai, máy đặt cố định",
        key=f"cl_{key}",
        help="Nếu chưa chắc chắn, cứ để trống. App chỉ giảm độ tin cậy, không chặn phân tích.",
    )
    if ok:
        return []
    return ["Chưa tick xác nhận: nếu áo rộng/tóc che vai thì mốc vai có thể lệch"]


def show_level(level, text):
    {0: st.success, 1: st.warning, 2: st.error}[level](f"**{LEVEL_TXT[level]}** - {text}")


def show_problems(bad):
    st.error(
        "**Không thể phân tích ảnh này:**\n\n" + "\n".join(f"{i+1}. {p}" for i, p in enumerate(bad))
    )
    st.markdown(
        '<div class="ss-next">Cách khắc phục nhanh: 1) chụp nơi đủ sáng, 2) giữ máy cố định ngang bằng, '
        '3) đứng xa 1,5–2 m cho người chiếm đầy khung. Xem chi tiết ở tab '
        '<b>Hướng dẫn chụp ảnh chuẩn</b>.</div>',
        unsafe_allow_html=True,
    )


def show_quality(res, extra=None):
    notes = res.notes + (extra or [])
    rel = reliability(notes)
    badge = {"Tốt": "ss-badge-ok", "Trung bình": "ss-badge-warn", "Thấp": "ss-badge-bad"}[rel]
    st.markdown(
        f'Dộ tin cậy của ảnh: <span class="ss-badge {badge}">{rel}</span>',
        unsafe_allow_html=True,
    )
    if not notes and not res.info:
        return
    if len(notes) <= 1 and rel != "Thấp":
        for n in notes:
            st.caption("• " + n)
        for n in res.info:
            st.caption("ℹ️ " + n)
        return
    with st.expander(f"Xem chi tiết ({len(notes)} lưu ý)", expanded=(rel == "Thấp")):
        for n in notes:
            st.caption("• " + n)
        for n in res.info:
            st.caption("ℹ️ " + n)
    if rel == "Thấp":
        st.warning("Độ tin cậy thấp: kết quả chỉ để tham khảo, nên chụp lại ảnh rõ và đúng chuẩn hơn.")


def csv_button(label, res, key, extra=None):
    st.download_button(
        "⬇️ Tải chỉ số (CSV) để đối chiếu đo thủ công",
        data=csv_bytes(label, res, extra),
        file_name=f"{key}.csv", mime="text/csv", key=f"dl_{key}",
        help="Tải file CSV chứa các góc đo để đối chiếu thủ công.",
        use_container_width=True,
    )


def show_gallery(res, caption, gallery):
    if gallery:
        cols = st.columns(min(3, len(gallery)))
        for i, img in enumerate(gallery):
            with cols[i % len(cols)]:
                show_image(img, f"Ảnh {i + 1}")
    else:
        show_image(res.image, caption)


def adams_hint():
    st.info(
        " Nếu nghi ngờ lệch vai hoặc vẹo cột sống, bước sàng lọc tiếp theo thường dùng là **nghiệm pháp cúi người "
        "(Adams)** do nhân viên y tế thực hiện, đo góc xoay thân bằng scoliometer (thường chuyển khám khi khoảng "
        "5-7° trở lên). Ảnh 2D của ứng dụng không nhìn thấy độ xoay của đốt sống nên không thay thế được bước này."
    )


def render_front(res, caption, key, gallery=None, extra=None):
    m = res.metrics
    show_gallery(res, caption, gallery)
    c1, c2, c3 = st.columns(3)
    c1.metric("Góc lệch vai", f"{m['goc_lech_vai_do']:.1f}°",
              help="Góc giữa đường nối hai vai và phương ngang.")
    c2.metric("Vai thấp hơn", str(m["vai_thap_hon"]))
    t = m["lech_than_so_voi_hong_do"]
    c3.metric("Lệch thân so với hông", "—" if t is None else f"{abs(t):.1f}°",
              help="Góc đường hông→vai so với phương thẳng đứng (nhìn từ sau/trước).")
    show_level(res.level, res.summary)
    show_quality(res, extra)
    if res.level > 0:
        adams_hint()
    csv_button(caption, res, key, extra)


def render_side(res, caption, key, gallery=None, extra=None):
    m = res.metrics
    show_gallery(res, caption, gallery)
    c1, c2, c3 = st.columns(3)
    c1.metric("Đầu nhô ra trước (góc tai–vai)", f"{m['goc_tai_vai_do']:.0f}°",
              help="Góc giữa phương ngang và đường vai→tai. Càng nhỏ, đầu càng nhô ra trước. "
                   "Gần với CVA nhưng dùng mốc vai thay cho C7.")
    c2.metric("Nghiêng thân", f"{m['nghieng_than_do']:+.0f}°", help="Dương: thân nghiêng ra trước.")
    c3.metric("Hướng quay mặt", str(m["huong_mat"]))
    show_level(res.level, res.summary)
    show_quality(res, extra)
    csv_button(caption, res, key, extra)


def run_and_show(files, kind, view, posture, auto_roll, caption, key, extra=None):
    res, bad, singles = run_many(files, kind, view, posture, auto_roll)
    if res is None:
        show_problems(bad)
        return None
    if bad:
        st.warning("Một số ảnh bị bỏ qua:\n\n" + "\n".join(f"- {p}" for p in bad))
    gallery = [r.image for r in singles] if len(singles) > 1 else None
    eyebrow("Kết quả phân tích")
    if kind == "front":
        render_front(res, caption, key, gallery, extra)
    else:
        render_side(res, caption, key, gallery, extra)
    return res
