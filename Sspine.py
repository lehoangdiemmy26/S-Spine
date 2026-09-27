import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image
import mediapipe as mp

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="S-Spine | Tầm soát lệch vai & cột sống",
    page_icon="🩺",
    layout="wide"
)
# Logo trường hoặc biểu tượng ứng dụng
try:
    st.image("gen-n-z8308486911094_3cf6e9f66d814eabd93c0c5ae610e055-modified.png", width=140)
except Exception:
    pass
# Giao diện Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; color: #1A202C; }
    [data-testid="stSidebar"] { background-color: #F8FAFC; border-right: 1px solid #E2E8F0; }
    .main-title { color: #0284C7; font-weight: 800; font-size: 2.6rem; margin-bottom: 0px; }
    .sub-title { color: #475569; font-size: 1.05rem; font-weight: 500; margin-bottom: 25px; }
    [data-testid="stFileUploadDropzone"] { background-color: #F8FAFC !important; border: 2px dashed #38BDF8 !important; border-radius: 12px !important; }
    [data-testid="stMetricValue"] { color: #0F172A !important; font-weight: 700 !important; }
    .guide-card {
        background-color: #F0F9FF;
        border-left: 5px solid #0284C7;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .exercise-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .exercise-title {
        color: #0369A1;
        font-weight: 700;
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🩺 S-Spine</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Ứng dụng AI Tầm Soát Biến Dạng Cột Sống Học Đường | Lượng giác & Vector (Toán 10-11)</p>', unsafe_allow_html=True)

# Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose

# =========================================================
# HÀM PHÂN TÍCH VÀ TÍNH TOÁN GÓC NGHIÊNG CÂN BẰNG VAI
# =========================================================
def process_and_analyze(image_pil):
    # Resize ảnh để xử lý mượt mà
    max_size = 900
    w_orig, h_orig = image_pil.size
    if max(w_orig, h_orig) > max_size:
        scale = max_size / float(max(w_orig, h_orig))
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        image_pil = image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

    img_np = np.array(image_pil.convert('RGB'))
    h, w, _ = img_np.shape
    
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5) as pose:
        results = pose.process(img_np)
        
        if not results.pose_landmarks:
            return None, 0.0, "Không nhận diện được người", False, ""
            
        landmarks = results.pose_landmarks.landmark
        
        # 1. Lấy tọa độ 2 mỏm vai từ MediaPipe
        # Index 11: LEFT_SHOULDER (Vai Trái cơ thể), Index 12: RIGHT_SHOULDER (Vai Phải cơ thể)
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Lấy thêm các điểm mặt để phát hiện góc chụp (Trước hay Sau lưng)
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        
        # Kiểm tra xem người dùng chụp từ SAU LƯNG hay MẶT ĐỐI MẶT
        # Nếu mũi nằm phía sau mỏm vai (z_nose > z_shoulders) -> Nhìn từ Sau lưng
        avg_shoulder_z = (left_shoulder.z + right_shoulder.z) / 2.0
        is_back_view = nose.z > avg_shoulder_z or nose.visibility < 0.5
        view_text = "Góc nhìn: Sau lưng" if is_back_view else "Góc nhìn: Đối diện"
        
        # Chuyển sang tọa độ Pixel
        p_left = (int(left_shoulder.x * w), int(left_shoulder.y * h))   # Vai Trái thực tế
        p_right = (int(right_shoulder.x * w), int(right_shoulder.y * h)) # Vai Phải thực tế

        # Kiểm tra góc nghiêng camera
        shoulder_depth_diff = abs(left_shoulder.z - right_shoulder.z)
        is_angle_valid = shoulder_depth_diff < 0.35  

        annotated_img = img_np.copy()
        
        # 2. Vẽ điểm & đường kết nối mỏm vai
        # Vai Trái: Màu Đỏ (Red), Vai Phải: Màu Xanh Dương (Blue)
        cv2.circle(annotated_img, p_left, 8, (255, 0, 0), -1)   # Vai Trái
        cv2.circle(annotated_img, p_right, 8, (0, 0, 255), -1)  # Vai Phải
        cv2.line(annotated_img, p_left, p_right, (0, 255, 0), 3) # Đường nối 2 vai
        
        # 3. Vẽ đường tham chiếu cân bằng 0 độ (Đường nét đứt/vàng ngang)
        y_avg = int((p_left[1] + p_right[1]) / 2)
        x_min = min(p_left[0], p_right[0]) - 30
        x_max = max(p_left[0], p_right[0]) + 30
        cv2.line(annotated_img, (max(0, x_min), y_avg), (min(w, x_max), y_avg), (255, 255, 0), 2, cv2.LINE_AA)

        # chú thích trực quan trên ảnh
        cv2.putText(annotated_img, "Vai Trai", (p_left[0] - 30, p_left[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(annotated_img, "Vai Phai", (p_right[0] - 30, p_right[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 4. TÍNH TOÁN GÓC NGHIÊNG ALPHA LƯỢNG GIÁC
        # Trong hệ tọa độ ảnh OpenCV: Y tăng từ TRÊN xuống DƯỚI (Y lớn hơn = vị trí THẤP hơn = XỆ)
        dy = p_left[1] - p_right[1] # y_left - y_right
        dx = abs(p_left[0] - p_right[0])
        
        if dx == 0:
            angle_deg = 90.0
        else:
            angle_deg = math.degrees(math.atan(abs(dy) / float(dx)))
            
        # 5. XÁC ĐỊNH BÊN BỊ XỆ CHÍNH XÁC THEO CƠ THỂ
        if abs(dy) < 3: # Chênh lệch nhỏ hơn 3 pixel coi như cân bằng
            direction = "Cân bằng"
        elif dy > 0:
            # dy > 0 -> y_left > y_right -> Vai Trái nằm thấp hơn Vai Phải -> Xệ VAI TRÁI
            direction = "Xệ VAI TRÁI"
        else:
            # dy < 0 -> y_left < y_right -> Vai Phải nằm thấp hơn Vai Trái -> Xệ VAI PHẢI
            direction = "Xệ VAI PHẢI"
            
        return annotated_img, angle_deg, direction, is_angle_valid, view_text

# =========================================================
# BÀI TẬP PHỤC HỒI Y KHOA (S-SPINE CARE)
# =========================================================
def show_exercise_recommendations(status_type, angle_val):
    st.markdown("---")
    st.subheader("🏋️ Lộ Trình Luyện Tập & Phục Hồi Cá Nhân Hóa (S-Spine Care)")
    
    if status_type == "normal":
        st.success("🎉 **Tư thế của bạn rất chuẩn!** Hãy duy trì thói quen sinh hoạt tốt và thực hiện 2 bài tập giãn cơ nhẹ nhàng sau mỗi 45 phút ngồi học:")
        
        ex1, ex2 = st.columns(2)
        with ex1:
            st.markdown("""
            <div class="exercise-card">
                <div class="exercise-title">1. Xoay Vai & Mở Tầng Ngực (Shoulder Rolls)</div>
                <p><b>Cách thực hiện:</b> Đứng thẳng, thả lỏng tay. Xoay tròn hai vai từ trước ra sau nhẹ nhàng.</p>
                <p>⏱️ <b>Thời lượng:</b> 10-15 lần mỗi hướng.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with ex2:
            st.markdown("""
            <div class="exercise-card">
                <div class="exercise-title">2. Nghiêng Cổ Mở Rộng Cơ Cầu Vai (Neck Stretch)</div>
                <p><b>Cách thực hiện:</b> Nghiêng đầu sang trái/phải, dùng tay kéo nhẹ đầu để giãn cơ cổ vai.</p>
                <p>⏱️ <b>Thời lượng:</b> Giữ 15 giây mỗi bên.</p>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.warning(f"⚠️ **Góc lệch {angle_val:.2f}°:** Phát hiện xu hướng lệch vai do phân bổ trọng lực không đều. Dưới đây là chuỗi **3 Bài tập Vật lý trị liệu phục hồi tại nhà** dành riêng cho bạn:")
        
        ex1, ex2, ex3 = st.columns(3)
        
        with ex1:
            st.markdown("""
            <div class="exercise-card">
                <div class="exercise-title">1. Tư thế Con Mèo - Con Bò (Cat-Cow Pose)</div>
                <p><b>Tác dụng:</b> Cải thiện độ linh hoạt cột sống ngực và thắt lưng.</p>
                <p><b>Cách tập:</b> Quỳ 4 điểm (tay & gối). Hít vào võng lưng ngẩng đầu, thở ra cong lưng hóp bụng.</p>
                <p>⏱️ <b>Liều lượng:</b> 10 - 12 lần/ngày.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with ex2:
            st.markdown("""
            <div class="exercise-card">
                <div class="exercise-title">2. Chống Tường Mở Vai (Wall Push/Stretch)</div>
                <p><b>Tác dụng:</b> Tăng cường sức mạnh cơ lưng trên, kéo giãn cơ ngực bị co thắt.</p>
                <p><b>Cách tập:</b> Đứng đối diện tường cách 0.5m, chống 2 tay lên tường. Nhấn nhẹ ngực về phía tường.</p>
                <p>⏱️ <b>Liều lượng:</b> Giữ 20 giây x 3 lần.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with ex3:
            st.markdown("""
            <div class="exercise-card">
                <div class="exercise-title">3. Tư thế Tấm Ván Nhẹ (Modified Plank)</div>
                <p><b>Tác dụng:</b> Tăng cường cơ lõi (Core) giúp giữ cột sống thẳng khi đeo cặp nặng.</p>
                <p><b>Cách tập:</b> Chống khuỷu tay và cẳng tay xuống sàn, giữ thân người thành một đường thẳng từ vai đến gối.</p>
                <p>⏱️ <b>Liều lượng:</b> Giữ 30 giây x 3 hiệp.</p>
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# GIAO DIỆN TABS HƯỚNG DẪN & TẦM SOÁT
# =========================================================
tab_guide, tab_app = st.tabs(["📐 Hướng Dẫn Chụp Ảnh Chuẩn", "📊 Tầm Soát & Phân Tích AI"])

with tab_guide:
    st.subheader("📋 Quy Trình Chụp Ảnh Tầm Soát Chuẩn Y Khoa")
    st.info("💡 Tuân thủ 3 quy tắc sau để tránh sai số do góc chụp hoặc trang phục:")
    
    g1, g2, g3 = st.columns(3)
    
    with g1:
        st.markdown("""
        <div class="guide-card">
            <h4>👕 1. Trang Phục</h4>
            <ul>
                <li>Mặc <b>áo thun ôm sát body</b> (hoặc áo dệt kim).</li>
                <li>Tránh áo phông quá rộng che mất đường viền vai & cột sống.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with g2:
        st.markdown("""
        <div class="guide-card">
            <h4>🧍 2. Tư Thế Đứng</h4>
            <ul>
                <li>Đứng thẳng tự nhiên, thả lỏng 2 tay dọc theo thân người.</li>
                <li>Có thể chụp từ <b>mặt trước đối diện</b> hoặc <b>từ sau lưng</b>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with g3:
        st.markdown("""
        <div class="guide-card">
            <h4>📸 3. Góc Máy Camera</h4>
            <ul>
                <li>Đặt điện thoại <b>ngang tầm ngực</b> (khoảng cách 1.5m - 2m).</li>
                <li>Camera đặt song song cơ thể, tránh nghiêng góc máy lên/xuống.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab_app:
    st.sidebar.header("⚙️ Thông số Tải trọng")
    bag_weight = st.sidebar.number_input("Trọng lượng cặp sách (kg):", min_value=0.0, value=4.5, step=0.5)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Ảnh Đứng Tĩnh (Thả lỏng)")
        file1 = st.file_uploader("Tải ảnh đứng tự nhiên:", type=['jpg', 'png', 'jpeg'], key="1")
        
    with col2:
        st.subheader("2. Ảnh Đeo Cặp Sách")
        file2 = st.file_uploader("Tải ảnh đeo cặp:", type=['jpg', 'png', 'jpeg'], key="2")

    st.markdown("---")

    # CHẾ ĐỘ 1: TẢI ẢNH TĨNH
    if file1 and not file2:
        img1 = Image.open(file1)
        res_img, angle1, dir1, valid1, view_text1 = process_and_analyze(img1)
        
        if res_img is not None:
            if not valid1:
                st.warning("⚠️ **CẢNH BÁO GÓC CHỤP:** Camera bị nghiêng/xéo góc! Vui lòng xem tab 'Hướng Dẫn Chụp Ảnh' để đạt độ chính xác cao nhất.")
            
            st.image(res_img, caption=f"AI quét mỏm vai ({view_text1})", use_container_width=True)
            st.header("📊 PHÂN TÍCH TƯ THẾ TỰ NHIÊN")
            
            m1, m2 = st.columns(2)
            m1.metric("Góc nghiêng vai (α1)", f"{angle1:.2f}°")
            m2.metric("Trạng thái vai", dir1)
            
            if angle1 < 1.5:
                st.success("✅ **Tư thế chuẩn:** Khung xương vai cân bằng tự nhiên.")
                show_exercise_recommendations("normal", angle1)
            elif 1.5 <= angle1 < 3.5:
                st.warning(f"⚠️ **CẢNH BÁO MỨC NHẸ:** Dáng đứng bị **{dir1}** lệch **{angle1:.2f}°**!")
                show_exercise_recommendations("warning", angle1)
            else:
                st.error(f"🚨 **BÁO ĐỘNG ĐỎ:** Lệch vai nghiêm trọng **{angle1:.2f}°** ({dir1})! Khuyên nên đi khám X-quang kiểm tra Góc Cobb.")
                show_exercise_recommendations("warning", angle1)
        else:
            st.error("⚠️ AI không tìm thấy cơ thể người trong ảnh. Vui lòng thử lại với ảnh rõ hơn!")

    # CHẾ ĐỘ 2: TẢI ẢNH ĐEO CẶP
    elif file2 and not file1:
        img2 = Image.open(file2)
        res_img, angle2, dir2, valid2, view_text2 = process_and_analyze(img2)
        
        if res_img is not None:
            if not valid2:
                st.warning("⚠️ **CẢNH BÁO GÓC CHỤP:** Camera bị nghiêng/xéo góc!")
            
            st.image(res_img, caption=f"AI quét mỏm vai mang tải ({view_text2})", use_container_width=True)
            st.header("📊 PHÂN TÍCH TƯ THẾ MANG TẢI")
            
            adjust_cm = angle2 * 0.85
            m1, m2 = st.columns(2)
            m1.metric("Góc nghiêng vai (α2)", f"{angle2:.2f}°")
            m2.metric("Trạng thái vai", dir2)
            
            if angle2 < 1.5:
                st.success("✅ **Trạng thái tốt:** Cột sống chịu tải cân bằng.")
                show_exercise_recommendations("normal", angle2)
            elif 1.5 <= angle2 < 3.5:
                st.warning(f"⚠️ **CẢNH BÁO:** Đeo cặp làm **{dir2}** lệch **{angle2:.2f}°**! Nên thu ngắn dây đeo bên bị xệ khoảng **{adjust_cm:.1f} cm**.")
                show_exercise_recommendations("warning", angle2)
            else:
                st.error(f"🚨 **BÁO ĐỘNG ĐỎ:** Tải trọng cặp {bag_weight}kg làm lệch vai nghiêm trọng **{angle2:.2f}°** ({dir2})!")
                show_exercise_recommendations("warning", angle2)
        else:
            st.error("⚠️ AI không tìm thấy cơ thể người trong ảnh.")

    # CHẾ ĐỘ 3: TẢI ĐỦ 2 ẢNH (SO SÁNH Δα)
    elif file1 and file2:
        img1 = Image.open(file1)
        img2 = Image.open(file2)
        
        res1, angle1, dir1, valid1, view_text1 = process_and_analyze(img1)
        res2, angle2, dir2, valid2, view_text2 = process_and_analyze(img2)
        
        if res1 is not None and res2 is not None:
            if not valid1 or not valid2:
                st.warning("⚠️ **CẢNH BÁO GÓC CHỤP:** Có ảnh chụp bị nghiêng góc!")
            
            st.image([res1, res2], caption=[f"Ảnh 1: Đứng tĩnh ({view_text1})", f"Ảnh 2: Đeo cặp ({view_text2})"], use_container_width=True)
            st.header("📊 KẾT QUẢ SO SÁNH BIẾN DẠNG ĐỘNG (Δα)")
            
            delta_alpha = abs(angle2 - angle1)
            adjust_cm = delta_alpha * 0.85
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Góc vai tĩnh (α1)", f"{angle1:.2f}°")
            m2.metric("Góc vai mang tải (α2)", f"{angle2:.2f}°")
            m3.metric("Độ biến dạng động (Δα)", f"{delta_alpha:.2f}°", delta_color="inverse")
            
            if delta_alpha < 1.5:
                st.success("✅ **Trạng thái an toàn:** Tải trọng cặp phân bổ đều.")
                show_exercise_recommendations("normal", delta_alpha)
            elif 1.5 <= delta_alpha < 3.5:
                st.warning(f"⚠️ **CẢNH BÁO MỨC TRUNG BÌNH:** Cặp làm lệch vai thêm **{delta_alpha:.2f}°** ({dir2})! Đề xuất thu ngắn dây đeo bên xệ **{adjust_cm:.1f} cm**.")
                show_exercise_recommendations("warning", delta_alpha)
            else:
                st.error(f"🚨 **BÁO ĐỘNG ĐỎ:** Chỉ số biến dạng Δα = **{delta_alpha:.2f}°**! Giảm ngay trọng lượng cặp sách.")
                show_exercise_recommendations("warning", delta_alpha)
        else:
            st.error("⚠️ AI không phân tích được 1 trong 2 ảnh. Vui lòng kiểm tra lại ảnh!")

    else:
        st.info("👆 Tải ảnh lên để S-Spine nhận diện và phân tích ngay nhé!")
