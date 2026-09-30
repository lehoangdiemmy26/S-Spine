import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image
import mediapipe as mp

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="S-Spine | Tầm soát lệch vai & cột sống",
    page_icon="gen-n-z8308486911094_3cf6e9f66d814eabd93c0c5ae610e055-modified.png",
    layout="wide"
)

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
    .disclaimer-box {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        padding: 12px;
        border-radius: 8px;
        color: #92400E;
        font-size: 0.9rem;
        margin-top: 30px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">  S-Spine</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Ứng dụng AI Tầm Soát Biến Dạng Cột Sống Học Đường</p>', unsafe_allow_html=True)

mp_pose = mp.solutions.pose

# =========================================================
# 1. PHÂN TÍCH ĐỨNG CHÍNH DIỆN (ĐO LỆCH VAI / VẸO) - CÓ CHUẨN HÓA
# =========================================================
def process_standing_front(image_pil):
    max_size = 900
    w_orig, h_orig = image_pil.size
    if max(w_orig, h_orig) > max_size:
        scale = max_size / float(max(w_orig, h_orig))
        image_pil = image_pil.resize((int(w_orig * scale), int(h_orig * scale)), Image.LANCZOS)

    img_np = np.array(image_pil.convert('RGB'))
    h, w, _ = img_np.shape
    
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5) as pose:
        results = pose.process(img_np)
        if not results.pose_landmarks:
            return None, 0.0, "Không nhận diện được người", False, ""
            
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        
        # Kiểm tra Visibility của các điểm cốt lõi
        if (left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5 or 
            left_hip.visibility < 0.5 or right_hip.visibility < 0.5):
            return None, 0.0, "Các mốc cơ thể bị khuất hoặc không rõ", False, ""

        is_back_view = nose.z > ((left_shoulder.z + right_shoulder.z) / 2.0) or nose.visibility < 0.5
        view_text = "Góc nhìn: Sau lưng" if is_back_view else "Góc nhìn: Đối diện"
        
        p_left = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        p_right = (int(right_shoulder.x * w), int(right_shoulder.y * h))
        
        # Tính chiều cao thân người (torso_height) để chuẩn hóa tỉ lệ (tránh phụ thuộc khoảng cách camera)
        mid_hip_y = (left_hip.y + right_hip.y) / 2.0
        mid_sh_y = (left_shoulder.y + right_shoulder.y) / 2.0
        torso_height = max(10.0, abs(mid_hip_y - mid_sh_y) * h)

        annotated_img = img_np.copy()
        cv2.circle(annotated_img, p_left, 8, (255, 0, 0), -1)
        cv2.circle(annotated_img, p_right, 8, (0, 0, 255), -1)
        cv2.line(annotated_img, p_left, p_right, (0, 255, 0), 3)
        
        y_avg = int((p_left[1] + p_right[1]) / 2)
        cv2.line(annotated_img, (max(0, min(p_left[0], p_right[0]) - 30), y_avg), (min(w, max(p_left[0], p_right[0]) + 30), y_avg), (255, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(annotated_img, "Vai Trai", (p_left[0] - 30, p_left[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(annotated_img, "Vai Phai", (p_right[0] - 30, p_right[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        dy = p_left[1] - p_right[1] 
        dx = abs(p_left[0] - p_right[0])
        raw_angle = 90.0 if dx == 0 else math.degrees(math.atan(abs(dy) / float(dx)))
        
        # Chuẩn hóa góc theo tỷ lệ thân người
        normalized_angle = raw_angle * (150.0 / torso_height)
            
        if abs(dy) < 3: 
            direction = "Cân bằng"
        elif dy > 0:
            direction = "Xệ VAI TRÁI"
        else:
            direction = "Xệ VAI PHẢI"
            
        return annotated_img, normalized_angle, direction, True, view_text

# =========================================================
# 2. PHÂN TÍCH GÓC NGHIÊNG BÊN HÔNG (ĐO ĐỘ GÙ & CỔ) - CÓ CHUẨN HÓA
# =========================================================
def process_standing_side(image_pil):
    max_size = 900
    w_orig, h_orig = image_pil.size
    if max(w_orig, h_orig) > max_size:
        scale = max_size / float(max(w_orig, h_orig))
        image_pil = image_pil.resize((int(w_orig * scale), int(h_orig * scale)), Image.LANCZOS)

    img_np = np.array(image_pil.convert('RGB'))
    h, w, _ = img_np.shape
    
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5) as pose:
        results = pose.process(img_np)
        if not results.pose_landmarks:
            return None, 0.0, 0.0, "Không nhận diện được người", False
            
        landmarks = results.pose_landmarks.landmark
        
        use_left = landmarks[mp_pose.PoseLandmark.LEFT_EAR].visibility > landmarks[mp_pose.PoseLandmark.RIGHT_EAR].visibility
        
        idx_ear = mp_pose.PoseLandmark.LEFT_EAR if use_left else mp_pose.PoseLandmark.RIGHT_EAR
        idx_shoulder = mp_pose.PoseLandmark.LEFT_SHOULDER if use_left else mp_pose.PoseLandmark.RIGHT_SHOULDER
        idx_hip = mp_pose.PoseLandmark.LEFT_HIP if use_left else mp_pose.PoseLandmark.RIGHT_HIP
        
        ear = landmarks[idx_ear]
        shoulder = landmarks[idx_shoulder]
        hip = landmarks[idx_hip]
        
        if ear.visibility < 0.5 or shoulder.visibility < 0.5 or hip.visibility < 0.5:
            return None, 0.0, 0.0, "Góc chụp nghiêng bị khuất điểm mốc", False
        
        p_ear = (int(ear.x * w), int(ear.y * h))
        p_sh = (int(shoulder.x * w), int(shoulder.y * h))
        p_hip = (int(hip.x * w), int(hip.y * h))
        
        torso_height = max(10.0, abs(p_hip[1] - p_sh[1]))

        annotated_img = img_np.copy()
        cv2.circle(annotated_img, p_ear, 7, (255, 0, 0), -1)
        cv2.circle(annotated_img, p_sh, 7, (0, 255, 0), -1)
        cv2.circle(annotated_img, p_hip, 7, (0, 0, 255), -1)
        
        cv2.line(annotated_img, p_ear, p_sh, (255, 255, 0), 2)
        cv2.line(annotated_img, p_sh, p_hip, (255, 0, 255), 2)
        
        vertical_top = (p_hip[0], p_ear[1] - 30)
        cv2.line(annotated_img, p_hip, vertical_top, (200, 200, 200), 1, cv2.LINE_AA)

        # Tính toán chuẩn hóa theo tỉ lệ thân người
        kyphosis_offset = p_sh[0] - p_hip[0] 
        kyphosis_score = max(0.0, (float(kyphosis_offset) / torso_height) * 40.0)
        
        head_offset = ((p_ear[0] - p_sh[0]) / torso_height) * 50.0
        
        status_side = "Tư thế nghiêng chuẩn"
        if kyphosis_score > 8 or head_offset > 20:
            status_side = "Nguy cơ Gù lưng & Chu đầu (Nặng)"
        elif kyphosis_score > 4 or head_offset > 10:
            status_side = "Hơi khom lưng nhẹ"

        return annotated_img, kyphosis_score, head_offset, status_side, True

# =========================================================
# 3. PHÂN TÍCH TƯ THẾ NGỒI (TỪ SAU LƯNG) - CÓ CHUẨN HÓA
# =========================================================
def process_sitting_back(image_pil):
    max_size = 900
    w_orig, h_orig = image_pil.size
    if max(w_orig, h_orig) > max_size:
        scale = max_size / float(max(w_orig, h_orig))
        image_pil = image_pil.resize((int(w_orig * scale), int(h_orig * scale)), Image.LANCZOS)

    img_np = np.array(image_pil.convert('RGB'))
    h, w, _ = img_np.shape
    
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5) as pose:
        results = pose.process(img_np)
        if not results.pose_landmarks:
            return None, 0.0, 0.0, "Không nhận diện được người", False
            
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        
        if (left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5 or 
            left_hip.visibility < 0.5 or right_hip.visibility < 0.5):
            return None, 0.0, 0.0, "Không nhìn rõ vai hoặc hông khi ngồi", False
        
        p_left_s = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        p_right_s = (int(right_shoulder.x * w), int(right_shoulder.y * h))
        p_left_h = (int(left_hip.x * w), int(left_hip.y * h))
        p_right_h = (int(right_hip.x * w), int(right_hip.y * h))
        
        torso_height = max(10.0, abs(((p_left_h[1] + p_right_h[1])/2) - ((p_left_s[1] + p_right_s[1])/2)))
        torso_width = max(10.0, abs(p_left_s[0] - p_right_s[0]))
        
        annotated_img = img_np.copy()
        
        dy_s = p_left_s[1] - p_right_s[1]
        dx_s = abs(p_left_s[0] - p_right_s[0])
        raw_s_angle = 90.0 if dx_s == 0 else math.degrees(math.atan(abs(dy_s) / float(dx_s)))
        shoulder_angle = raw_s_angle * (150.0 / torso_height)
        
        mid_shoulder_x = (p_left_s[0] + p_right_s[0]) / 2.0
        mid_hip_x = (p_left_h[0] + p_right_h[0]) / 2.0
        offset_x = abs(mid_shoulder_x - mid_hip_x)
        hump_score = (offset_x / torso_width) * 30.0
        
        cv2.circle(annotated_img, p_left_s, 7, (255, 0, 0), -1)
        cv2.circle(annotated_img, p_right_s, 7, (0, 0, 255), -1)
        cv2.line(annotated_img, p_left_s, p_right_s, (0, 255, 0), 2)
        
        status_sit = "Ngồi cân đối"
        if shoulder_angle > 3.0 or hump_score > 10:
            status_sit = "Lệch vai & Gù khi ngồi (Nặng)"
        elif shoulder_angle > 1.2 or hump_score > 5:
            status_sit = "Lệch vai/Gù nhẹ khi ngồi"

        return annotated_img, shoulder_angle, hump_score, status_sit, True

# =========================================================
# GỢI Ý BÀI TẬP VẬT LÝ TRỊ LIỆU
# =========================================================
def show_exercise_recommendations(status_type):
    st.markdown("---")
    st.subheader("🏋️ Lộ Trình Luyện Tập & Phục Hồi Cá Nhân Hóa (S-Spine Care)")
    
    if status_type == "normal":
        st.success("🎉 **Tư thế của bạn rất chuẩn!** Hãy duy trì thói quen vận động nhẹ nhàng sau mỗi 45 phút học tập.")
        ex1, ex2 = st.columns(2)
        with ex1:
            st.markdown("""
            <div class="exercise-card">
                <div class="exercise-title">1. Xoay Vai & Mở Tầng Ngực</div>
                <p><b>Cách thực hiện:</b> Đứng thẳng, thả lỏng tay. Xoay tròn hai vai từ trước ra sau.</p>
                <p>⏱️ <b>Thời lượng:</b> 10-15 lần mỗi hướng.</p>
            </div>
            """, unsafe_allow_html=True)
        with ex2:
            st.markdown("""
            <div class="exercise-card">
                <div class="exercise-title">2. Nghiêng Cổ Giãn Cơ Cầu Vai</div>
                <p><b>Cách thực hiện:</b> Nghiêng đầu sang trái/phải, dùng tay kéo nhẹ đầu giãn cơ.</p>
                <p>⏱️ <b>Thời lượng:</b> Giữ 15 giây mỗi bên.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Phát hiện chỉ số lệch hoặc gù lưng. Dưới đây là 3 bài tập vật lý trị liệu giúp cải thiện cột sống:")
        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            st.markdown("""
            <div class="exercise-card">
                <div class="exercise-title">1. Tư thế Con Mèo - Con Bò</div>
                <p><b>Tác dụng:</b> Tăng độ linh hoạt cột sống ngực, giảm gù.</p>
                <p><b>Cách tập:</b> Quỳ 4 điểm. Hít võng lưng ngẩng đầu, thở cong lưng hóp bụng.</p>
                <p>⏱️ <b>Liều lượng:</b> 10 - 12 lần/ngày.</p>
            </div>
            """, unsafe_allow_html=True)
        with ex2:
            st.markdown("""
            <div class="exercise-card">
                <div class="exercise-title">2. Chống Tường Mở Vai</div>
                <p><b>Tác dụng:</b> Tăng sức mạnh cơ lưng trên, kéo giãn cơ ngực.</p>
                <p><b>Cách tập:</b> Đứng cách tường 0.5m, chống tay lên tường, nhấn nhẹ ngực.</p>
                <p>⏱️ <b>Liều lượng:</b> Giữ 20 giây x 3 lần.</p>
            </div>
            """, unsafe_allow_html=True)
        with ex3:
            st.markdown("""
            <div class="exercise-card">
                <div class="exercise-title">3. Tấm Ván Nhẹ (Modified Plank)</div>
                <p><b>Tác dụng:</b> Tăng cường cơ lõi giữ cột sống thẳng.</p>
                <p><b>Cách tập:</b> Chống khuỷu tay và cẳng tay xuống sàn, giữ thân người thẳng.</p>
                <p>⏱️ <b>Liều lượng:</b> Giữ 30 giây x 3 hiệp.</p>
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# GIAO DIỆN CHÍNH
# =========================================================
tab_guide, tab_app = st.tabs(["📐 Hướng Dẫn Chụp Ảnh Chuẩn", "📊 Tầm Soát & Phân Tích AI"])

with tab_guide:
    st.subheader("📋 Quy Trình Chụp Ảnh Tầm Soát Chuẩn Y Khoa")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("""
        <div class="guide-card">
            <h4>👕 1. Trang Phục</h4>
            <ul><li>Mặc áo thun ôm sát body để thấy rõ đường viền vai và sống lưng.</li></ul>
        </div>
        """, unsafe_allow_html=True)
    with g2:
        st.markdown("""
        <div class="guide-card">
            <h4>🧍 2. Góc Chụp Đa Dạng</h4>
            <ul>
                <li><b>Chính diện/Sau lưng:</b> Đo lệch vai ngang.</li>
                <li><b>Nghiêng bên hông (90 độ):</b> Đo độ gù & chu đầu.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with g3:
        st.markdown("""
        <div class="guide-card">
            <h4>📸 3. Khoảng Cách</h4>
            <ul><li>Đặt máy ngang tầm ngực/lưng ở khoảng cách 1.5m - 2m.</li></ul>
        </div>
        """, unsafe_allow_html=True)

with tab_app:
    st.sidebar.header("⚙️ Cấu hình Tầm Soát")
    bag_weight = st.sidebar.number_input("Trọng lượng cặp sách (kg):", min_value=0.0, value=4.5, step=0.5)

    analysis_mode = st.radio(
        "Chọn chế độ phân tích tư thế:",
        [
            "1. Đứng Tĩnh - Chính Diện (Kiểm tra lệch vai)", 
            "2. Đứng Nghiêng - Bên Hông (Đo độ gù & cổ)", 
            "3. Đeo Cặp Sách (Chính diện & Nghiêng)",
            "4. 🪑 Tư Thế Ngồi Học & Đo Độ Gù (Từ sau lưng)"
        ],
        horizontal=False
    )

    st.markdown("---")

    if analysis_mode == "1. Đứng Tĩnh - Chính Diện (Kiểm tra lệch vai)":
        st.subheader("1. Tầm Soát Tư Thế Đứng Tĩnh (Chính Diện/Sau Lưng)")
        input_type = st.radio("Chọn phương thức đầu vào:", ["Tải ảnh lên", "Chụp bằng Camera"], key="t1")
        file = st.file_uploader("Tải ảnh:", type=['jpg', 'png', 'jpeg'], key="u1") if input_type == "Tải ảnh lên" else st.camera_input("Chụp ảnh", key="c1")
        
        if file:
            img = Image.open(file)
            res_img, angle, direction, valid, vt = process_standing_front(img)
            if valid and res_img is not None:
                st.image(res_img, caption=f"AI phân tích ({vt})", use_container_width=True)
                
                if angle < 1.2:
                    level_label = "🟢 Bình thường (Chuẩn)"
                elif angle <= 3.0:
                    level_label = "🟡 Mức độ Nhẹ"
                else:
                    level_label = "🔴 Mức độ Nặng (Cần chú ý)"

                m1, m2, m3 = st.columns(3)
                m1.metric("Gốc nghiêng vai chuẩn hóa", f"{angle:.2f}°")
                m2.metric("Trạng thái vai", direction)
                m3.metric("Phân loại", level_label)
                
                if angle < 1.2:
                    st.success("✅ Tư thế chuẩn xác!")
                    show_exercise_recommendations("normal")
                elif angle <= 3.0:
                    st.warning(f"⚠ Dáng đứng bị **{direction}** ở mức **Nhẹ** (lệch {angle:.2f}°). Nên thực hiện các bài tập giãn cơ và chú ý dáng đi đứng.")
                    show_exercise_recommendations("warning")
                else:
                    st.error(f"🚨 Phát hiện lệch vai rõ rệt ở mức **Nặng** ({direction} lệch {angle:.2f}°)! Cần điều chỉnh tư thế ngay và tham khảo tư vấn chuyên khoa.")
                    show_exercise_recommendations("warning")
            else:
                st.error(f"⚠️ {direction if res_img is None else 'Không tìm thấy người trong ảnh'}. Hãy chắc chắn khung hình thấy rõ toàn thân trên.")

    elif analysis_mode == "2. Đứng Nghiêng - Bên Hông (Đo độ gù & cổ)":
        st.subheader("2. Đo Độ Gù & Chu Đầu (Góc Chụp Nghiêng 90 Độ)")
        st.info("💡 **Lưu ý:** Người chụp đứng ngang bên cạnh học sinh để lấy trọn góc nhìn từ Tai - Vai - Hông.")
        input_type = st.radio("Chọn phương thức đầu vào:", ["Tải ảnh lên", "Chụp bằng Camera"], key="t2")
        file = st.file_uploader("Tải ảnh nghiêng:", type=['jpg', 'png', 'jpeg'], key="u2") if input_type == "Tải ảnh lên" else st.camera_input("Chụp ảnh nghiêng", key="c2")
        
        if file:
            img = Image.open(file)
            res_img, kyphosis, head_off, status_side, valid = process_standing_side(img)
            if valid and res_img is not None:
                st.image(res_img, caption="AI phân tích đường cong cột sống bên hông", use_container_width=True)
                
                if kyphosis < 4 and head_off < 10:
                    level_label = "🟢 Bình thường"
                elif kyphosis <= 8 and head_off <= 20:
                    level_label = "🟡 Mức độ Nhẹ"
                else:
                    level_label = "🔴 Mức độ Nặng"

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Chỉ số gù lưng", f"{kyphosis:.1f}")
                s2.metric("Độ chu đầu (FHP)", f"{head_off:.1f}")
                s3.metric("Đánh giá", status_side)
                s4.metric("Phân loại", level_label)
                
                if kyphosis < 4 and head_off < 10:
                    st.success("✅ Đường cong cột sống sinh lý hoàn toàn bình thường!")
                    show_exercise_recommendations("normal")
                elif kyphosis <= 8 or head_off <= 20:
                    st.warning("⚠️ Phát hiện xu hướng khom vai/cổ rướn trước ở mức **Nhẹ**. Hãy tích cực tập luyện mở ngực!")
                    show_exercise_recommendations("warning")
                else:
                    st.error("🚨 Phát hiện tình trạng gù lưng và chu đầu ở mức **Nặng**! Cần nghiêm túc thực hiện các bài tập vật lý trị liệu.")
                    show_exercise_recommendations("warning")
            else:
                st.error(f"⚠️ {status_side if res_img is None else 'Không nhận diện rõ các mốc cơ thể ở góc nghiêng này.'}")

    elif analysis_mode == "3. Đeo Cặp Sách (Chính diện & Nghiêng)":
        st.subheader("3. Tầm Soát Khi Đeo Cặp Sách (Đánh giá áp lực tải trọng)")
        c_front, c_side = st.columns(2)
        with c_front:
            f_front = st.file_uploader("Ảnh đeo cặp (Chính diện/Sau lưng)", type=['jpg', 'png', 'jpeg'], key="uf")
        with c_side:
            f_side = st.file_uploader("Ảnh đeo cặp (Nghiêng bên hông)", type=['jpg', 'png', 'jpeg'], key="us")
            
        if f_front and f_side:
            img_f = Image.open(f_front)
            img_s = Image.open(f_side)
            res_f, angle_f, dir_f, valid_f, _ = process_standing_front(img_f)
            res_s, ky_s, head_s, _, valid_s = process_standing_side(img_s)
            
            if valid_f and valid_s and res_f is not None and res_s is not None:
                st.image([res_f, res_s], caption=["Ảnh 1: Cân bằng vai mang tải", "Ảnh 2: Độ gù khi mang tải"], use_container_width=True)
                
                if angle_f < 1.2 and ky_s < 5:
                    bag_level = "🟢 An toàn"
                elif angle_f <= 2.5 and ky_s <= 8:
                    bag_level = "🟡 Ảnh hưởng Nhẹ"
                else:
                    bag_level = "🔴 Ảnh hưởng Nặng"

                m1, m2, m3 = st.columns(3)
                m1.metric("Góc lệch vai mang tải", f"{angle_f:.2f}°")
                m2.metric("Chỉ số gù khi đeo", f"{ky_s:.1f}")
                m3.metric("Đánh giá tải trọng", bag_level)
                
                if bag_level == "🟢 An toàn":
                    st.success(f"✅ Tải trọng {bag_weight}kg an toàn, không gây biến dạng tư thế.")
                    show_exercise_recommendations("normal")
                elif bag_level == "🟡 Ảnh hưởng Nhẹ":
                    st.warning(f"⚠️ Cặp sách {bag_weight}kg gây lệch vai/gù ở mức **Nhẹ** ({dir_f}). Hãy đeo đều hai quai và thu ngắn dây.")
                    show_exercise_recommendations("warning")
                else:
                    st.error(f"🚨 Cặp sách {bag_weight}kg quá nặng gây ảnh hưởng **Nặng** đến cột sống! Cần giảm bớt trọng lượng balo ngay.")
                    show_exercise_recommendations("warning")
            else:
                st.error("⚠️ Một trong hai bức ảnh không nhận diện rõ khung người. Vui lòng kiểm tra lại góc chụp.")

    elif analysis_mode == "4. 🪑 Tư Thế Ngồi Học & Đo Độ Gù (Từ sau lưng)":
        st.subheader("4. Tầm Soát Tư Thế Ngồi Học & Lệch Trục Lưng")
        input_type = st.radio("Chọn phương thức đầu vào:", ["Tải ảnh lên", "Chụp bằng Camera"], key="t4")
        file = st.file_uploader("Tải ảnh ngồi:", type=['jpg', 'png', 'jpeg'], key="u4") if input_type == "Tải ảnh lên" else st.camera_input("Chụp ảnh ngồi", key="c4")
        
        if file:
            img = Image.open(file)
            res_img, s_angle, hump, status_sit, valid = process_sitting_back(img)
            if valid and res_img is not None:
                st.image(res_img, caption="AI phân tích tư thế ngồi học", use_container_width=True)
                
                if hump < 5 and s_angle < 1.5:
                    sit_level = "🟢 Chuẩn"
                elif hump <= 10 and s_angle <= 3.0:
                    sit_level = "🟡 Lệch/Gù Nhẹ"
                else:
                    sit_level = "🔴 Lệch/Gù Nặng"

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Góc lệch vai ngồi", f"{s_angle:.2f}°")
                s2.metric("Trục lệch lưng", f"{hump:.1f}")
                s3.metric("Đánh giá", status_sit)
                s4.metric("Phân loại", sit_level)
                
                if sit_level == "🟢 Chuẩn":
                    st.success("✅ Tư thế ngồi học chuẩn và cân đối.")
                    show_exercise_recommendations("normal")
                elif sit_level == "🟡 Lệch/Gù Nhẹ":
                    st.warning("⚠️️ Phát hiện tư thế khom lưng hoặc lệch vai **Nhẹ** khi ngồi học. Hãy điều chỉnh khoảng cách bàn ghế và góc nhìn.")
                    show_exercise_recommendations("warning")
                else:
                    st.error("🚨 Phát hiện tư thế ngồi lệch trục sống và gù ở mức **Nặng**! Cần sửa ngay thói quen chống cằm hoặc nghiêng người khi viết bài.")
                    show_exercise_recommendations("warning")
            else:
                st.error(f"⚠️ {status_sit if res_img is None else 'Không nhận diện được khung người trong ảnh ngồi.'}")

# Miễn trừ trách nhiệm y tế (Disclaimer) footer
st.markdown("""
<div class="disclaimer-box">
    <b>⚠️ MIỄN TRỪ TRÁCH NHIỆM Y TẾ:</b> Kết quả phân tích từ hệ thống S-Spine chỉ mang tính chất tầm soát, tham khảo và hỗ trợ giáo dục sức khỏe học đường, <b>không có giá trị thay thế chẩn đoán y khoa chính thức</b> từ bác sĩ chuyên khoa hoặc chuyên gia vật lý trị liệu.
</div>
""", unsafe_allow_html=True)
