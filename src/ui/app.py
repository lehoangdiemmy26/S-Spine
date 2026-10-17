# -*- coding: utf-8 -*-
"""Hàm main() dựng layout Streamlit: header + 3 tab + chọn mục."""
import streamlit as st

from src.ui.modes.bag import mode_bag
from src.ui.modes.compare import mode_compare
from src.ui.modes.kyphosis_page import mode_kyphosis
from src.ui.modes.sitting import mode_sitting
from src.ui.modes.standing import mode_standing
from src.ui.pages.guide import tab_guide
from src.ui.pages.method import tab_method
from src.ui.theme import CSS_HOME, eyebrow, header_html, load_logo
from src.vision.pose import get_pose, last_pose_error


def main():
    logo = load_logo()
    st.set_page_config(
        page_title="S-Spine | AI đánh giá và phân tích tư thế học đường",
        page_icon=logo if logo is not None else "image-0 (2).png",
        layout="wide",
    )
    st.markdown(CSS_HOME, unsafe_allow_html=True)

    if get_pose() is None:
        import mediapipe as mp
        import sys

        st.error(
            "Không khởi tạo được MediaPipe Pose. Hãy ghim phiên bản có `mp.solutions` trong requirements.txt "
            "(ví dụ `mediapipe==0.10.14`) rồi triển khai lại."
        )
        with st.expander("Chi tiết kỹ thuật (để chẩn đoán)"):
            st.code(
                f"python: {sys.version.split()[0]}\n"
                f"mediapipe: {getattr(mp, '__version__', '?')}\n"
                f"mp.solutions: {hasattr(mp, 'solutions')}\n"
                f"mediapipe tại: {getattr(mp, '__file__', '?')}\n"
                f"lỗi: {last_pose_error()}"
            )
            st.caption(
                "Mẹo hay gặp: nếu lỗi là FileNotFoundError ...pose_landmark_cpu.binarypb thì venv đang nằm trong "
                "thư mục có dấu/unicode (ví dụ “Diễm My”). Hãy tạo venv ở đường dẫn không dấu, ví dụ "
                "`C:\\Users\\admin\\.venvs\\s-spine-py310`, rồi chạy lại. Trên Streamlit Cloud nhớ có packages.txt "
                "(libgl1, libglib2.0-0)."
            )
        st.stop()

    st.markdown(header_html(logo), unsafe_allow_html=True)

    t_guide, t_app, t_method = st.tabs(
        ["Hướng dẫn chụp ảnh chuẩn", "Đánh giá & phân tích tư thế", "Phương pháp & giới hạn"]
    )
    with t_guide:
        tab_guide()
    with t_app:
        eyebrow("Bước 1 · Chọn mục")
        mode = st.radio(
            "Chọn mục tầm soát tư thế:",
            ["Tư thế đứng", "Tư thế ngồi học", "So sánh trước & sau đeo cặp"],
            key="main_mode_v2", horizontal=True, label_visibility="collapsed",
        )
        sub = None
        if mode == "Tư thế đứng":
            sub = st.radio(
                "Chức năng:", ["Đứng tĩnh", "Đeo cặp sách", "Gù lưng"],
                key="sub_mode_v2", horizontal=True, label_visibility="collapsed",
            )
        with st.expander("⚙️ Cài đặt nâng cao: bù máy nghiêng"):
            tilt_mode = st.radio(
                "Bù máy nghiêng:",
                ["Tự động (khuyến nghị)", "Nhập tay góc nghiêng máy", "Không bù"],
                key="tilt_mode", horizontal=True,
            )
            if tilt_mode.startswith("Tự"):
                auto_roll = True
            elif tilt_mode.startswith("Nhập"):
                auto_roll = float(
                    st.number_input(
                        "Góc nghiêng máy (độ, dương = ảnh xoay thuận chiều kim đồng hồ)",
                        -20.0, 20.0, 0.0, 0.5, key="manual_roll",
                    )
                )
            else:
                auto_roll = False

        eyebrow("Bước 2 · Tải ảnh và xem kết quả")
        if mode == "Tư thế đứng":
            if sub == "Đứng tĩnh":
                mode_standing(auto_roll)
            elif sub == "Đeo cặp sách":
                mode_bag(auto_roll)
            else:
                mode_kyphosis()
        elif mode == "Tư thế ngồi học":
            mode_sitting(auto_roll)
        else:
            mode_compare(auto_roll)
    with t_method:
        tab_method()

    st.markdown(
        '<div class="disclaimer-box"><b>Lưu ý quan trọng:</b> Ứng dụng chỉ hỗ trợ tầm soát và mang tính chất tham khảo '
        'học đường, không thay thế chẩn đoán của bác sĩ chuyên khoa.</div>',
        unsafe_allow_html=True,
    )
