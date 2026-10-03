import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image
import mediapipe as mp

# =========================================================
# CẤU HÌNH LOGO TỰ ĐỘNG NHẬN DIỆN .PNG / .JPG
# =========================================================
logo_img = "🛡️" # Fallback mặc định nếu không tìm thấy file
for filename in [
    "gen-n-z8308486911094_3cf6e9f66d814eabd93c0c5ae610e055-modified.png",
    "gen-n-z8308486911094_3cf6e9f66d814eabd93c0c5ae610e055-modified.jpg"
]:
    try:
        logo_img = Image.open(filename)
        break
    except Exception:
        continue

# =========================================================
# CẤU HÌNH TRANG & GIAO DIỆN CHÍNH (CSS)
# =========================================================
st.set_page_config(
    page_title="S-Spine | Tầm Soát Lệch Vai & Cột Sống Học Đường",
    page_icon=logo_img if isinstance(logo_img, Image.Image) else "🛡️",
    layout="wide"
)

st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; color: #1A202C; }
    [data-testid="stSidebar"] { background-color: #F8FAFC; border-right: 1px solid #E2E8F0; }
    .main-title { color: #0284C7; font-weight: 800; font-size: 2.5rem; margin-bottom: 0px; }
    .sub-title { color: #475569; font-size: 1.05rem; font-weight: 500; margin-bottom: 20px; }
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
        margin-bottom: 8px;
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

# Hiển thị tiêu đề kèm logo ở góc trái nếu load thành công
col_logo, col_title = st.columns([0.08, 0.92])
with col_logo:
    if isinstance(logo_img, Image.Image):
        st.image(logo_img, width=65)
    else:
        st.markdown("<h1>🛡️</h1>", unsafe_allow_html=True)
with col_title:
    st.markdown('<h1 class="main-title" style="margin-top: 5px;">S-Spine</h1>', unsafe_allow_html=True)

st.markdown('<p class="sub-title">Ứng dụng AI Hỗ Trợ Tầm Soát Biến Dạng Cột Sống & Tư Thế Học Đường</p>', unsafe_allow_html=True)

mp_pose = mp.solutions.pose

# =========================================================
# HÀM 1: PHÂN TÍCH ĐỨNG CHÍNH DIỆN / SAU LƯNG (ĐO LỆCH VAI)
# =========================================================
def process_standing_front(image_pil):
    max_size = 900
    w_orig, h_orig = image_pil.size
    if max(w_orig, h_orig) > max_size:
        scale = max_size / float(max(w_orig, h_orig))
        image_pil = image_pil.resize((int(w_orig * scale), int(h_orig * scale)), Image.Resampling.LANCZOS)

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
        
        if (left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5 or 
            left_hip.visibility < 0.5 or right_hip.visibility < 0.5):
            return None, 0.0, "Các mốc cơ thể bị khuất", False, ""

        is_back_view = nose.z > ((left_shoulder.z + right_shoulder.z) / 2.0) or nose.visibility < 0.5
        view_text = "Góc nhìn: Sau lưng" if is_back_view else "Góc nhìn: Chính diện"
        
        p_left = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        p_right = (int(right_shoulder.x * w), int(right_shoulder.y * h))
        
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
        normalized_angle = raw_angle * (150.0 / torso_height)
            
        if abs(dy) < 3: 
            direction = "Cân bằng"
        elif dy > 0:
            direction = "Xệ VAI TRÁI"
        else:
            direction = "Xệ VAI PHẢI"
            
        return annotated_img, normalized_angle, direction, True, view_text

# =========================================================
# HÀM 2: PHÂN TÍCH GÓC NGHIÊNG BÊN HÔNG (ĐO ĐỘ GÙ & CỔ)
# =========================================================
def process_standing_side(image_pil):
    max_size = 900
    w_orig, h_orig = image_pil.size
    if max(w_orig, h_orig) > max_size:
        scale = max_size / float(max(w_orig, h_orig))
        image_pil = image_pil.resize((int(w_orig * scale), int(h_orig * scale)), Image.Resampling.LANCZOS)

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

        kyphosis_offset = p_sh[0] - p_hip[0] 
        kyphosis_score = max(0.0, (float(kyphosis_offset) / torso_height) * 40.0)
        head_offset = ((p_ear[0] - p_sh[0]) / torso_height) * 50.0
        
        status_side = "Tư thế chuẩn"
        if kyphosis_score > 8 or head_offset > 20:
            status_side = "Nguy cơ Gù & Chu đầu (Nặng)"
        elif kyphosis_score > 4 or head_offset > 10:
            status_side = "Hơi khom lưng nhẹ"

        return annotated_img, kyphosis_score, head_offset, status_side, True

# =========================================================
# HÀM 3: PHÂN TÍCH TƯ THẾ NGỒI (TỪ SAU LƯNG)
# =========================================================
def process_sitting_back(image_pil):
    max_size = 900
    w_orig, h_orig = image_pil.size
    if max(w_orig, h_orig) > max_size:
        scale = max_size / float(max(w_orig, h_orig))
        image_pil = image_pil.resize((int(w_orig * scale), int(h_orig * scale)), Image.Resampling.LANCZOS)

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
# GỢI Ý BÀI TẬP VẬT LÝ TRỊ LIỆU CÁ NHÂN HÓA
# =========================================================
def show_exercise_recommendations(status_type):
    st.markdown("---")
    st.subheader("Lộ Trình Luyện Tập & Phục Hồi Cột Sống (S-Spine Care)")
    
    if status_type == "normal":
        st.success(" **Tư thế của bạn rất chuẩn!** Hãy duy trì thói quen vận động nhẹ nhàng sau mỗi 45 phút học tập.")
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
        st.warning("⚠️ Phát hiện chỉ số lệch hoặc gù lưng. Dưới đây là các bài tập vật lý trị liệu giúp cải thiện cột sống:")
        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            st.markdown("""
            <div class="exercise-card">
                <div class="exercise-title">1. Tư thế Con Mèo - Con Bò</div>
                <p><b>Tác dụng:</b> Tăng độ linh hoạt cột sống ngực, giảm gù.</p>
                <p><b>Cách tập:</b> Quỳ 4 điểm. Hít võng lưng ngẩng đầu, thở cong lưng hóp bụng.</p>
                <p>⏱ <b>Liều lượng:</b> 10 - 12 lần/ngày.</p>
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
# GIAO DIỆN CHÍNH (TABS & MODES)
# =========================================================
tab_guide, tab_app = st.tabs([" Hướng Dẫn Chụp Ảnh Chuẩn", " Tầm Soát & Phân Tích AI"])

with tab_guide:
    st.subheader("Quy Trình Chụp Ảnh Tầm Soát Chuẩn Y Khoa")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("""
        <div class="guide-card">
            <h4> 1. Trang Phục</h4>
            <ul><li>Mặc áo thun ôm sát body để thấy rõ đường viền vai và sống lưng.</li></ul>
        </div>
        """, unsafe_allow_html=True)
    with g2:
        st.markdown("""
        <div class="guide-card">
            <h4> 2. Góc Chụp Đa Dạng</h4>
            <ul>
                <li><b>Chính diện/Sau lưng:</b> Đo lệch vai ngang.</li>
                <li><b>Nghiêng bên hông (90 độ):</b> Đo độ gù & chu đầu.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with g3:
        st.markdown("""
        <div class="guide-card">
            <h4> 3. Khoảng Cách</h4>
            <ul><li>Đặt máy ngang tầm ngực/lưng ở khoảng cách 1.5m - 2m.</li></ul>
        </div>
        """, unsafe_allow_html=True)

with tab_app:
    st.sidebar.header("⚙ Cấu hình Tầm Soát")
    bag_weight = st.sidebar.number_input("Trọng lượng cặp sách (kg):", min_value=0.0, value=4.5, step=0.5)

    analysis_mode = st.radio(
        "Chọn mục tầm soát tư thế:",
        [
            "1. Đứng Tĩnh (Đo lệch vai / độ gù)", 
            "2. Đeo Cặp Sách (Đánh giá tải trọng)",
            "3. Tư Thế Ngồi Học & Đo Độ Gù (Sau lưng & Nghiêng)",
            "4. So Sánh Trước & Sau Khi Đeo Cặp (Chênh lệch & Chỉnh dây)"
        ],
        horizontal=False
    )

    st.markdown("---")

    # =========================================================
    # MỤC 1: ĐỨNG TĨNH (TÙY CHỌN GÓC CHỤP)
    # =========================================================
    if analysis_mode == "1. Đứng Tĩnh (Đo lệch vai / độ gù)":
        st.subheader("1. Tầm Soát Tư Thế Đứng Tĩnh")
        
        standing_sub_mode = st.selectbox(
            "Chọn góc chụp:",
            ["Chính diện / Sau lưng (Kiểm tra lệch vai)", "Nghiêng bên hông (Kiểm tra độ gù & chu đầu)"]
        )
        
        input_type = st.radio("Chọn phương thức đầu vào:", ["Tải ảnh lên", "Chụp bằng Camera"], key="t1_stand")
        
        if standing_sub_mode == "Chính diện / Sau lưng (Kiểm tra lệch vai)":
            file = st.file_uploader("Tải ảnh chính diện/sau lưng:", type=['jpg', 'png', 'jpeg'], key="u1_front") if input_type == "Tải ảnh lên" else st.camera_input("Chụp ảnh", key="c1_front")
            
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
                    m1.metric("Góc nghiêng vai chuẩn hóa", f"{angle:.2f}°")
                    m2.metric("Trạng thái vai", direction)
                    m3.metric("Phân loại", level_label)
                    
                    if angle < 1.2:
                        st.success("✅ Tư thế chuẩn xác!")
                        show_exercise_recommendations("normal")
                    elif angle <= 3.0:
                        st.warning(f"⚠ Dáng đứng bị **{direction}** ở mức **Nhẹ** (lệch {angle:.2f}°). Nên thực hiện các bài tập giãn cơ và chú ý dáng đi đứng.")
                        show_exercise_recommendations("warning")
                    else:
                        st.error(f"🚨 Phát hiện lệch vai rõ rệt ở mức **Nặng** ({direction} lệch {angle:.2f}°)! Cần điều chỉnh tư thế ngay.")
                        show_exercise_recommendations("warning")
                else:
                    st.error("⚠️ Không tìm thấy người trong ảnh hoặc mốc cơ thể bị khuất.")
        
        else: # Góc nghiêng
            file = st.file_uploader("Tải ảnh nghiêng bên hông:", type=['jpg', 'png', 'jpeg'], key="u1_side") if input_type == "Tải ảnh lên" else st.camera_input("Chụp ảnh nghiêng", key="c1_side")
            
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
                    st.error("⚠️ Không nhận diện rõ các mốc cơ thể ở góc nghiêng này.")

    # =========================================================
    # MỤC 2: ĐEO CẶP SÁCH
    # =========================================================
    elif analysis_mode == "2. Đeo Cặp Sách (Đánh giá tải trọng)":
        st.subheader("2. Tầm Soát Khi Đeo Cặp Sách (Đánh giá áp lực tải trọng)")
        st.info(" Tải lên **cả 2 ảnh** (Ảnh mặt trước/sau để kiểm tra lệch vai và Ảnh nghiêng bên hông để kiểm tra độ gù khi mang balo) để hệ thống tổng hợp đánh giá nhé!")
        
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
                    st.warning(f"⚠️ Cặp sách {bag_weight}kg gây lệch vai/gù ở mức **Nhẹ** ({dir_f}). Hãy đeo đều hai quai.")
                    show_exercise_recommendations("warning")
                else:
                    st.error(f"🚨 Cặp sách {bag_weight}kg quá nặng gây ảnh hưởng **Nặng** đến cột sống! Cần giảm bớt trọng lượng balo ngay.")
                    show_exercise_recommendations("warning")
            else:
                st.error("⚠ Một trong hai bức ảnh không nhận diện rõ khung người.")

    # =========================================================
    # MỤC 3: TƯ THẾ NGỒI HỌC & ĐO ĐỘ GÙ (KẾT HỢP LINH HOẠT 2 GÓC CHỤP)
    # =========================================================
    elif analysis_mode == "3. Tư Thế Ngồi Học & Đo Độ Gù (Sau lưng & Nghiêng)":
        st.subheader("3. Tầm Soát Tư Thế Ngồi Học & Độ Gù (Kết hợp Đa Góc)")
        st.info("💡 Ở chế độ này, cậu có thể linh hoạt tải lên **1 trong 2 góc chụp** hoặc **cả 2 góc chụp** (Sau lưng để kiểm tra lệch vai khi ngồi, Nghiêng bên hông để kiểm tra độ gù lưng và cúi đầu sát bàn). Hệ thống sẽ tự động phân tích ảnh nào được cung cấp!")

        col_up1, col_up2 = st.columns(2)
        with col_up1:
            file_sit_back = st.file_uploader(" Ảnh ngồi từ Sau lưng (Kiểm tra lệch vai)", type=['jpg', 'png', 'jpeg'], key="u_sit_back")
        with col_up2:
            file_sit_side = st.file_uploader(" Ảnh ngồi Nghiêng bên hông (Kiểm tra độ gù)", type=['jpg', 'png', 'jpeg'], key="u_sit_side")

        has_analysis = False

        if file_sit_back:
            has_analysis = True
            st.markdown("---")
            st.markdown("#### 🔍 Kết quả phân tích Góc Ngồi - Sau Lưng")
            img_b = Image.open(file_sit_back)
            res_img_b, s_angle, hump, status_sit, valid_b = process_sitting_back(img_b)
            if valid_b and res_img_b is not None:
                st.image(res_img_b, caption="AI phân tích độ cân đối vai và trục lưng khi ngồi", use_container_width=True)
                
                if hump < 5 and s_angle < 1.5:
                    sit_level_b = "🟢 Chuẩn"
                elif hump <= 10 and s_angle <= 3.0:
                    sit_level_b = "🟡 Lệch/Gù Nhẹ"
                else:
                    sit_level_b = "🔴 Lệch/Gù Nặng"

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Góc lệch vai ngồi", f"{s_angle:.2f}°")
                s2.metric("Trục lệch lưng", f"{hump:.1f}")
                s3.metric("Đánh giá", status_sit)
                s4.metric("Phân loại", sit_level_b)
                
                if sit_level_b == "🟢 Chuẩn":
                    st.success("✅ Tư thế ngồi học từ sau lưng chuẩn và cân đối.")
                elif sit_level_b == "🟡 Lệch/Gù Nhẹ":
                    st.warning("⚠ Phát hiện tư thế khom lưng hoặc lệch vai **Nhẹ** khi ngồi học.")
                else:
                    st.error("🚨 Phát hiện tư thế ngồi lệch trục sống và gù ở mức **Nặng**! Cần điều chỉnh ngay.")
            else:
                st.error("⚠️ Không nhận diện được khung người trong ảnh ngồi từ sau lưng.")

        if file_sit_side:
            has_analysis = True
            st.markdown("---")
            st.markdown("#### 🔍 Kết quả phân tích Góc Ngồi - Nghiêng Bên Hông")
            img_s = Image.open(file_sit_side)
            res_img_s, ky_sit, head_sit, status_sit_side, valid_s = process_standing_side(img_s)
            if valid_s and res_img_s is not None:
                st.image(res_img_s, caption="AI phân tích độ gù cột sống ngực và cúi đầu khi ngồi học", use_container_width=True)
                
                if ky_sit < 4 and head_sit < 10:
                    sit_level_s = "🟢 Chuẩn"
                elif ky_sit <= 8 and head_sit <= 20:
                    sit_level_s = "🟡 Khom lưng nhẹ"
                else:
                    sit_level_s = "🔴 Gù nặng / Cúi sát bàn"

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Chỉ số gù lưng ngồi", f"{ky_sit:.1f}")
                k2.metric("Độ rướn đầu khi cúi", f"{head_sit:.1f}")
                k3.metric("Đánh giá góc nghiêng", status_sit_side)
                k4.metric("Phân loại", sit_level_s)
                
                if sit_level_s == "🟢 Chuẩn":
                    st.success("✅ Góc độ cúi và lưng khi ngồi học rất chuẩn khoa học.")
                elif sit_level_s == "🟡 Khom lưng nhẹ":
                    st.warning("⚠️ Có xu hướng cúi sát bàn hoặc khom lưng nhẹ khi làm bài.")
                else:
                    st.error("🚨 Cảnh báo gù lưng và cúi đầu quá sát mặt bàn! Nguy cơ cận thị và cong vẹo cột sống cao.")
            else:
                st.error("⚠️ Không nhận diện rõ các mốc cơ thể ở ảnh nghiêng này.")

        if has_analysis:
            show_exercise_recommendations("warning")
        else:
            if not file_sit_back and not file_sit_side:
                st.info("👆 Vui lòng tải lên ít nhất một trong hai ảnh (Sau lưng hoặc Nghiêng bên hông) để AI bắt đầu tầm soát nhé!")

    # =========================================================
    # MỤC 4: SO SÁNH TRƯỚC & SAU KHI ĐEO CẶP (TÍNH NĂNG MỚI)
    # =========================================================
    elif analysis_mode == "4. So Sánh Trước & Sau Khi Đeo Cặp (Chênh lệch & Chỉnh dây)":
        st.subheader("4. Đối Chiếu Tư Thế: Trước vs. Sau Khi Đeo Cặp Sách")
        st.info("💡 Tính năng này giúp so sánh trực tiếp sự thay đổi của cơ thể khi mang balo so với lúc đứng thẳng bình thường. Hãy chọn góc chụp và tải lên 2 bức ảnh tương ứng.")

        comp_sub_mode = st.selectbox(
            "Chọn góc chụp so sánh:",
            ["Chính diện / Sau lưng (So sánh độ lệch vai & mất cân bằng)", "Nghiêng bên hông (So sánh độ gù lưng & ngả người ra trước)"]
        )

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            file_normal = st.file_uploader("1️⃣ Ảnh KHÔNG đeo cặp (Trạng thái gốc)", type=['jpg', 'png', 'jpeg'], key="u_comp_norm")
        with col_b2:
            file_backpack = st.file_uploader("2️⃣ Ảnh KHI ĐEO CẶP SÁCH", type=['jpg', 'png', 'jpeg'], key="u_comp_pack")

        if file_normal and file_backpack:
            img_norm = Image.open(file_normal)
            img_pack = Image.open(file_backpack)
            
            st.markdown("---")
            
            if "Chính diện / Sau lưng" in comp_sub_mode:
                res_n, angle_n, dir_n, valid_n, _ = process_standing_front(img_norm)
                res_p, angle_p, dir_p, valid_p, _ = process_standing_front(img_pack)
                
                if valid_n and valid_p and res_n is not None and res_p is not None:
                    st.image([res_n, res_p], caption=["Ảnh 1: Trước khi đeo cặp", "Ảnh 2: Khi đeo cặp sách"], use_container_width=True)
                    
                    delta_angle = angle_p - angle_n
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Lệch vai ban đầu", f"{angle_n:.2f}°")
                    c2.metric("Lệch vai khi mang balo", f"{angle_p:.2f}°", delta=f"{delta_angle:+.2f}°")
                    c3.metric("Trọng lượng balo", f"{bag_weight} kg")
                    
                    st.markdown("### 💡 Phân Tích & Lời Khuyên Chỉnh Dây Cặp Sách:")
                    
                    if abs(delta_angle) < 0.8 and angle_p < 1.5:
                        st.success("🎉 **Tuyệt vời!** Cặp sách không làm thay đổi đáng kể độ cân bằng vai của bạn. Dây đeo hiện tại đang rất phù hợp.")
                    else:
                        st.warning(f"⚠️ Trọng lượng {bag_weight}kg làm góc lệch vai thay đổi **{delta_angle:+.2f}°** (Trạng thái: {dir_p}).")
                        
                        # Đưa ra lời khuyên cụ thể về chiều dài dây đeo
                        if "VAI" in dir_p:
                            sh_side = "trái" if "TRÁI" in dir_p else "phải"
                            opp_side = "phải" if sh_side == "trái" else "trái"
                            st.markdown(f"""
                            * **Nguyên nhân:** Lực kéo của balo đang bị dồn lệch sang phía bên kia khiến vai bị kéo xệ hoặc lệch.
                            * **Hành động điều chỉnh dây:** 
                              * **Nới lỏng** dây đeo bên vai **{sh_side}** khoảng **1.0 - 1.5 cm**.
                              * **Rút ngắn** dây đeo bên vai **{opp_side}** khoảng **1.0 cm** để cân bằng lại lực kéo đều hai bên vai.
                              * Đảm bảo đáy balo nằm ngang thắt lưng, không bị trễ xuống quá mông.
                            """)
                        else:
                            st.markdown(f"""
                            * **Nguyên nhân:** Balo đang bị nặng hoặc quai đeo hai bên không đều lực.
                            * **Hành động điều chỉnh dây:** 
                              * Cả hai quai đeo đang chênh lệch lực kéo. Hãy kiểm tra và **rút ngắn đồng thời cả 2 quai đeo khoảng 2 cm** để balo áp sát hoàn toàn vào cột sống ngực, hạn chế tình trạng giật lùi về sau.
                            """)
                else:
                    st.error("⚠️ AI không nhận diện rõ khung người ở một trong hai bức ảnh. Vui lòng thử lại với ảnh rõ hơn.")
            
            else: # Nghiêng bên hông
                res_n, ky_n, head_n, _, valid_n = process_standing_side(img_norm)
                res_p, ky_p, head_p, _, valid_p = process_standing_side(img_pack)
                
                if valid_n and valid_p and res_n is not None and res_p is not None:
                    st.image([res_n, res_p], caption=["Ảnh 1: Trước khi đeo (Nghiêng)", "Ảnh 2: Khi đeo cặp (Nghiêng)"], use_container_width=True)
                    
                    delta_ky = ky_p - ky_n
                    delta_head = head_p - head_n
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Chỉ số gù ban đầu", f"{ky_n:.1f}")
                    c2.metric("Chỉ số gù khi mang balo", f"{ky_p:.1f}", delta=f"{delta_ky:+.1f}")
                    c3.metric("Độ rướn đầu tăng thêm", f"{delta_head:+.1f}")
                    
                    st.markdown("### 💡 Phân Tích & Lời Khuyên Chỉnh Dây Cặp Sách:")
                    
                    if delta_ky < 2.0 and delta_head < 3.0:
                        st.success("✅ Trọng lượng balo phân bổ tốt, không làm lưng bị khom thêm nhiều khi đứng.")
                    else:
                        st.warning(f"⚠️ Khi đeo cặp {bag_weight}kg, chỉ số gù lưng tăng lên **{delta_ky:+.1f} đơn vị** và cổ bị rướn ra trước nhiều hơn.")
                        st.markdown(f"""
                        * **Nguyên nhân:** Dây đeo cặp đang bị **quá dài**, khiến trọng tâm balo bị kéo sà xuống thấp (dưới thắt lưng), làm cơ thể phải ngả người về trước hoặc ngửa cổ để bù trừ trọng lực.
                        * **Hành động điều chỉnh dây:**
                          * Tiến hành **rút ngắn quai đeo balo lên khoảng 3 - 5 cm** sao cho đỉnh trên của balo ngang tầm vai và đáy balo nằm sát thắt lưng (cách eo khoảng 5cm).
                          * Sử dụng thêm **dây đai ngực (nếu có)** để cố định hai quai không bị bè ra ngoài, giúp phân bổ đều lực lên lồng ngực thay vì đè nặng cột sống thắt lưng.
                        """)
                else:
                    st.error("⚠️ Không nhận diện rõ các mốc cơ thể ở góc nghiêng này.")
        else:
            st.info("👆 Vui lòng tải lên đầy đủ **cả 2 bức ảnh** (Không đeo và Khi đeo cặp) để hệ thống tiến hành đối chiếu thông số và đưa ra lời khuyên chỉnh dây cụ thể!")

# Miễn trừ trách nhiệm y tế footer
st.markdown("""
<div class="disclaimer-box">
    <b>⚠️ MIỄN TRỪ TRÁCH NHIỆM Y TẾ:</b> Kết quả phân tích từ hệ thống S-Spine chỉ mang tính chất tầm soát, tham khảo và hỗ trợ giáo dục sức khỏe học đường, <b>không có giá trị thay thế chẩn đoán y khoa chính thức</b> từ bác sĩ chuyên khoa hoặc chuyên gia vật lý trị liệu.
</div>
""", unsafe_allow_html=True)
