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
# GỢI Ý BÀI TẬP VẬT LÝ TRỊ LIỆU CÁ NHÂN HÓA (ĐÚNG BỆNH)
# =========================================================
def show_exercise_recommendations(status_type):
    st.markdown("---")
    st.subheader("📋 Lộ Trình Luyện Tập & Phục Hồi Cột Sống (Cá Nhân Hóa)")
    
    if status_type == "normal":
        st.success(" **Tư thế của bạn rất chuẩn!** Hãy duy trì thói quen vận động nhẹ nhàng sau mỗi 45 phút học tập.")
        ex1, ex2 = st.columns(2)
        with ex1:
            st.markdown("""
            <div class="exercise-card">
                <div class="exercise-title">1. Xoay Vai & Mở Tầng Ngực</div>
                <p><b>Cách thực hiện:</b> Đứng thẳng, thả lỏng tay. Xoay tròn hai vai từ trước ra sau.</p>
                <p>⏱ <b>Thời lượng:</b> 10-15 lần mỗi hướng.</p>
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
            
    elif status_type == "light":
        st.warning(" **Tình trạng ở mức Nhẹ:** Cần tập trung các bài tập giãn cơ và cân bằng cơ thể sớm để tránh tiến triển nặng.")
        ex1, ex2 = st.columns(2)
        with ex1:
            st.markdown("""
            <div class="exercise-card">
                <div class="exercise-title">1. Tư thế Con Mèo - Con Bò</div>
                <p><b>Tác dụng:</b> Tăng độ linh hoạt cột sống ngực, giảm gù nhẹ.</p>
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
            
    else: # heavy
        st.error("**Tình trạng ở mức Nặng:** Phát hiện biến dạng/lệch rõ rệt. Cần nghiêm túc thực hiện các bài tập chuyên sâu phục hồi.")
        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            st.markdown("""
            <div class="exercise-card">
                <div class="exercise-title">1. Tập Cơ Lưng Trên (Prine Cobra)</div>
                <p><b>Tác dụng:</b> Tăng cường cơ dựng sống, kéo vai về vị trí cân đối.</p>
                <p><b>Cách tập:</b> Nằm sấp, nâng nhẹ ngực và tay lên khỏi sàn, siết cơ lưng.</p>
                <p>⏱️ <b>Liều lượng:</b> Giữ 10 giây x 10 hiệp.</p>
            </div>
            """, unsafe_allow_html=True)
        with ex2:
            st.markdown("""
            <div class="exercise-card">
                <div class="exercise-title">2. Chữa Gù & Mở Ngực Chuyên Sâu</div>
                <p><b>Tác dụng:</b> Giải phóng áp lực đè nén cột sống ngực.</p>
                <p><b>Cách tập:</b> Dùng trụ lăn foam hoặc bóng yoga đặt dưới lưng ngực để ngả người mở rộng.</p>
                <p>⏱️ <b>Liều lượng:</b> 3 phút mỗi ngày.</p>
            </div>
            """, unsafe_allow_html=True)
        with ex3:
            st.markdown("""
            <div class="exercise-card">
                <div class="exercise-title">3. Tấm Ván Core (Modified Plank)</div>
                <p><b>Tác dụng:</b> Củng cố toàn diện nhóm cơ lõi giữ cột sống thẳng trục.</p>
                <p><b>Cách tập:</b> Chống khuỷu tay vuông góc, giữ thân người thẳng cứng.</p>
                <p>⏱️ <b>Liều lượng:</b> Giữ 30 giây x 3 hiệp.</p>
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# GIAO DIỆN CHÍNH (TABS & MODES)
# =========================================================
tab_guide, tab_app = st.tabs(["Hướng Dẫn Chụp Ảnh Chuẩn", "Tầm Soát & Phân Tích AI"])

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
    st.sidebar.header("⚙️ Cấu hình Tầm Soát")
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
    # MỤC 1: ĐỨNG TĨNH
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
                        st.success("Tư thế chuẩn xác!")
                        show_exercise_recommendations("normal")
                    elif angle <= 3.0:
                        st.warning(f"⚠️ Dáng đứng bị **{direction}** ở mức **Nhẹ** (lệch {angle:.2f}°).")
                        show_exercise_recommendations("light")
                    else:
                        st.error(f"🚨 Phát hiện lệch vai rõ rệt ở mức **Nặng** ({direction} lệch {angle:.2f}°)! Cần điều chỉnh tư thế ngay.")
                        show_exercise_recommendations("heavy")
                else:
                    st.error(" Không tìm thấy người trong ảnh hoặc mốc cơ thể bị khuất. Vui lòng kiểm tra lại ánh sáng, góc chụp và **tải lên một bức ảnh khác** rõ ràng hơn nhé!")
        
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
                        st.success("Đường cong cột sống sinh lý hoàn toàn bình thường!")
                        show_exercise_recommendations("normal")
                    elif kyphosis <= 8 or head_off <= 20:
                        st.warning("⚠️ Phát hiện xu hướng khom vai/cổ rướn trước ở mức **Nhẹ**.")
                        show_exercise_recommendations("light")
                    else:
                        st.error("🚨 Phát hiện tình trạng gù lưng và chu đầu ở mức **Nặng**!")
                        show_exercise_recommendations("heavy")
                else:
                    st.error("⚠️ Không nhận diện rõ các mốc cơ thể ở góc nghiêng này. Vui lòng kiểm tra lại tư thế và **tải lên một bức ảnh khác**.")

    # =========================================================
    # MỤC 2: ĐEO CẶP SÁCH (HỖ TRỢ LINH HOẠT 1 HOẶC 2 ẢNH)
    # =========================================================
    elif analysis_mode == "2. Đeo Cặp Sách (Đánh giá tải trọng)":
        st.subheader("2. Tầm Soát Khi Đeo Cặp Sách (Đánh giá áp lực tải trọng)")
        st.info("💡 **Mẹo:** Bạn có thể cung cấp **1 trong 2 ảnh** (hoặc cả 2 ảnh nếu có) để hệ thống tiến hành tầm soát linh hoạt.")
        
        input_type_bag = st.radio("Chọn phương thức đầu vào:", ["Tải ảnh lên", "Chụp bằng Camera"], key="t2_bag")

        c_front, c_side = st.columns(2)
        with c_front:
            if input_type_bag == "Tải ảnh lên":
                f_front = st.file_uploader("Ảnh đeo cặp (Chính diện/Sau lưng) - Không bắt buộc", type=['jpg', 'png', 'jpeg'], key="uf")
            else:
                f_front = st.camera_input("Chụp ảnh đeo cặp (Chính diện/Sau lưng)", key="cf")
        with c_side:
            if input_type_bag == "Tải ảnh lên":
                f_side = st.file_uploader("Ảnh đeo cặp (Nghiêng bên hông) - Không bắt buộc", type=['jpg', 'png', 'jpeg'], key="us")
            else:
                f_side = st.camera_input("Chụp ảnh đeo cặp (Nghiêng bên hông)", key="cs")
            
        has_bag_analysis = False
        worst_bag_level = "normal"

        if f_front:
            st.markdown("---")
            st.markdown("#### 📊 Phân tích từ Ảnh Đeo Cặp (Chính diện / Sau lưng)")
            img_f = Image.open(f_front)
            res_f, angle_f, dir_f, valid_f, _ = process_standing_front(img_f)
            
            if valid_f and res_f is not None:
                has_bag_analysis = True
                st.image(res_f, caption="Ảnh phân tích độ cân bằng vai khi mang balo", use_container_width=True)
                
                if angle_f < 1.2:
                    level_f = "🟢 Cân bằng"
                elif angle_f <= 2.5:
                    level_f = "🟡 Lệch vai nhẹ"
                    if worst_bag_level == "normal": worst_bag_level = "light"
                else:
                    level_f = "🔴 Lệch vai nặng"
                    worst_bag_level = "heavy"

                m1, m2, m3 = st.columns(3)
                m1.metric("Góc lệch vai mang tải", f"{angle_f:.2f}°")
                m2.metric("Trạng thái vai", dir_f)
                m3.metric("Đánh giá vai", level_f)
                
                if level_f == "🟢 Cân bằng":
                    st.success(f" Trọng lượng {bag_weight}kg không làm lệch vai khi đứng.")
                elif level_f == "🟡 Lệch vai nhẹ":
                    st.warning(f"⚠️ Trọng lượng {bag_weight}kg gây {dir_f} ở mức **Nhẹ**.")
                else:
                    st.error(f"🚨 Trọng lượng {bag_weight}kg gây lệch vai rõ rệt ở mức **Nặng** ({dir_f})!")
            else:
                st.error(" Không nhận diện rõ khung người trong ảnh chính diện/sau lưng. Vui lòng thử lại với ảnh rõ hơn.")

        if f_side:
            st.markdown("---")
            st.markdown("#### 🔍 Phân tích từ Ảnh Đeo Cặp (Nghiêng Bên Hông)")
            img_s = Image.open(f_side)
            res_s, ky_s, head_s, _, valid_s = process_standing_side(img_s)
            
            if valid_s and res_s is not None:
                has_bag_analysis = True
                st.image(res_s, caption="Ảnh phân tích độ gù khi mang balo", use_container_width=True)
                
                if ky_s < 5:
                    level_s = "🟢 An toàn"
                elif ky_s <= 8:
                    level_s = "🟡 Gù/Khom lưng nhẹ"
                    if worst_bag_level == "normal": worst_bag_level = "light"
                else:
                    level_s = "🔴 Gù nặng do tải trọng"
                    worst_bag_level = "heavy"

                k1, k2, k3 = st.columns(3)
                k1.metric("Chỉ số gù khi đeo", f"{ky_s:.1f}")
                k2.metric("Độ rướn đầu (FHP)", f"{head_s:.1f}")
                k3.metric("Đánh giá tải trọng", level_s)
                
                if level_s == "🟢 An toàn":
                    st.success(f" Cột sống ngực giữ độ cong tốt dưới trọng lượng {bag_weight}kg.")
                elif level_s == "🟡 Gù/Khom lưng nhẹ":
                    st.warning(f"⚠️ Trọng lượng {bag_weight}kg làm tăng độ khom lưng/gù nhẹ.")
                else:
                    st.error(f"🚨 Trọng lượng {bag_weight}kg quá nặng, gây gù lưng nặng khi mang!")
            else:
                st.error(" Không nhận diện rõ khung người ở góc chụp nghiêng. Vui lòng thử lại với ảnh rõ hơn.")

        # Xử lý hiển thị lộ trình luyện tập thông minh dựa trên kết quả thực tế
        if has_bag_analysis:
            show_exercise_recommendations(worst_bag_level)
        else:
            if not f_front and not f_side:
                st.info("📌 Vui lòng tải lên hoặc chụp ít nhất **một trong hai ảnh** (Chính diện/Sau lưng hoặc Nghiêng bên hông) để AI bắt đầu tầm soát tải trọng cặp sách nhé!")

    # =========================================================
    # MỤC 3: TƯ THẾ NGỒI HỌC & ĐO ĐỘ GÙ
    # =========================================================
    elif analysis_mode == "3. Tư Thế Ngồi Học & Đo Độ Gù (Sau lưng & Nghiêng)":
        st.subheader("3. Tầm Soát Tư Thế Ngồi Học & Độ Gù (Kết hợp Đa Góc)")
        st.info("Bạn có thể chọn tải ảnh lên hoặc chụp trực tiếp bằng camera cho 1 hoặc cả 2 góc chụp (Sau lưng và Nghiêng bên hông).")

        input_type_sit = st.radio("Chọn phương thức đầu vào:", ["Tải ảnh lên", "Chụp bằng Camera"], key="t3_sit")

        col_up1, col_up2 = st.columns(2)
        with col_up1:
            if input_type_sit == "Tải ảnh lên":
                file_sit_back = st.file_uploader("Ảnh ngồi từ Sau lưng (Kiểm tra lệch vai)", type=['jpg', 'png', 'jpeg'], key="u_sit_back")
            else:
                file_sit_back = st.camera_input("Chụp ảnh ngồi từ Sau lưng", key="c_sit_back")
        with col_up2:
            if input_type_sit == "Tải ảnh lên":
                file_sit_side = st.file_uploader("Ảnh ngồi Nghiêng bên hông (Kiểm tra độ gù)", type=['jpg', 'png', 'jpeg'], key="u_sit_side")
            else:
                file_sit_side = st.camera_input("Chụp ảnh ngồi Nghiêng bên hông", key="c_sit_side")

        has_valid_analysis = False
        worst_level = "normal"

        if file_sit_back:
            st.markdown("---")
            st.markdown("####  Kết quả phân tích Góc Ngồi - Sau Lưng")
            img_b = Image.open(file_sit_back)
            res_img_b, s_angle, hump, status_sit, valid_b = process_sitting_back(img_b)
            
            if valid_b and res_img_b is not None:
                has_valid_analysis = True
                st.image(res_img_b, caption="AI phân tích độ cân đối vai và trục lưng khi ngồi", use_container_width=True)
                
                if hump < 5 and s_angle < 1.5:
                    sit_level_b = "🟢 Chuẩn"
                elif hump <= 10 and s_angle <= 3.0:
                    sit_level_b = "🟡 Lệch/Gù Nhẹ"
                    if worst_level == "normal": worst_level = "light"
                else:
                    sit_level_b = "🔴 Lệch/Gù Nặng"
                    worst_level = "heavy"

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Góc lệch vai ngồi", f"{s_angle:.2f}°")
                s2.metric("Trục lệch lưng", f"{hump:.1f}")
                s3.metric("Đánh giá", status_sit)
                s4.metric("Phân loại", sit_level_b)
                
                if sit_level_b == "🟢 Chuẩn":
                    st.success(" Tư thế ngồi học từ sau lưng chuẩn và cân đối.")
                elif sit_level_b == "🟡 Lệch/Gù Nhẹ":
                    st.warning("⚠️ Phát hiện tư thế khom lưng hoặc lệch vai **Nhẹ** khi ngồi học.")
                else:
                    st.error("🚨 Phát hiện tư thế ngồi lệch trục sống và gù ở mức **Nặng**!")
            else:
                st.error(" Không nhận diện được khung người trong ảnh ngồi từ sau lưng. Vui lòng **tải lên ảnh khác**.")

        if file_sit_side:
            st.markdown("---")
            st.markdown("#### 🔍 Kết quả phân tích Góc Ngồi - Nghiêng Bên Hông")
            img_s = Image.open(file_sit_side)
            res_img_s, ky_sit, head_sit, status_sit_side, valid_s = process_standing_side(img_s)
            
            if valid_s and res_img_s is not None:
                has_valid_analysis = True
                st.image(res_img_s, caption="AI phân tích độ gù cột sống ngực và cúi đầu khi ngồi học", use_container_width=True)
                
                if ky_sit < 4 and head_sit < 10:
                    sit_level_s = "🟢 Chuẩn"
                elif ky_sit <= 8 and head_sit <= 20:
                    sit_level_s = "🟡 Khom lưng nhẹ"
                    if worst_level == "normal": worst_level = "light"
                else:
                    sit_level_s = "🔴 Gù nặng / Cúi sát bàn"
                    worst_level = "heavy"

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Chỉ số gù lưng ngồi", f"{ky_sit:.1f}")
                k2.metric("Độ rướn đầu khi cúi", f"{head_sit:.1f}")
                k3.metric("Đánh giá góc nghiêng", status_sit_side)
                k4.metric("Phân loại", sit_level_s)
                
                if sit_level_s == "🟢 Chuẩn":
                    st.success(" Góc độ cúi và lưng khi ngồi học rất chuẩn khoa học.")
                elif sit_level_s == "🟡 Khom lưng nhẹ":
                    st.warning("⚠️ Có xu hướng cúi sát bàn hoặc khom lưng nhẹ khi làm bài.")
                else:
                    st.error("🚨 Cảnh báo gù lưng và cúi đầu quá sát mặt bàn!")
            else:
                st.error(" Không nhận diện rõ các mốc cơ thể ở ảnh nghiêng này. Vui lòng **tải lên ảnh khác**.")

        if has_valid_analysis:
            show_exercise_recommendations(worst_level)
        else:
            if not file_sit_back and not file_sit_side:
                st.info(" Vui lòng tải lên hoặc chụp ít nhất một trong hai ảnh (Sau lưng hoặc Nghiêng bên hông) để AI bắt đầu tầm soát nhé!")

    # =========================================================
    # MỤC 4: SO SÁNH TRƯỚC & SAU KHI ĐEO CẶP
    # =========================================================
    elif analysis_mode == "4. So Sánh Trước & Sau Khi Đeo Cặp (Chênh lệch & Chỉnh dây)":
        st.subheader("4. Đối Chiếu Tư Thế: Trước và sau khi đeo cặp")
        st.info("Tính năng này giúp so sánh trực tiếp sự thay đổi của cơ thể khi mang cặp sách so với lúc đứng thẳng bình thường.")

        comp_sub_mode = st.selectbox(
            "Chọn góc chụp so sánh:",
            ["Chính diện / Sau lưng (So sánh độ lệch vai & mất cân bằng)", "Nghiêng bên hông (So sánh độ gù lưng & ngả người ra trước)"]
        )

        input_type_comp = st.radio("Chọn phương thức đầu vào:", ["Tải ảnh lên", "Chụp bằng Camera"], key="t4_comp")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if input_type_comp == "Tải ảnh lên":
                file_normal = st.file_uploader("1. Ảnh không đeo cặp (trạng thái gốc)", type=['jpg', 'png', 'jpeg'], key="u_comp_norm")
            else:
                file_normal = st.camera_input("1. Chụp ảnh không đeo cặp", key="c_comp_norm")
        with col_b2:
            if input_type_comp == "Tải ảnh lên":
                file_backpack = st.file_uploader("2. Ảnh khi đeo cặp", type=['jpg', 'png', 'jpeg'], key="u_comp_pack")
            else:
                file_backpack = st.camera_input("2. Chụp ảnh khi đeo cặp", key="c_comp_pack")

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
                    
                    st.markdown("### 🛠️ Phân Tích & Lời Khuyên Chỉnh Dây Cặp Sách:")
                    
                    if abs(delta_angle) < 0.8 and angle_p < 1.5:
                        st.success("🌟 **Tuyệt vời!** Cặp sách không làm thay đổi đáng kể độ cân bằng vai của bạn.")
                    else:
                        st.warning(f"⚠️️ Trọng lượng {bag_weight}kg làm góc lệch vai thay đổi **{delta_angle:+.2f}°**.")
                        
                        if "VAI" in dir_p:
                            sh_side = "trái" if "TRÁI" in dir_p else "phải"
                            opp_side = "phải" if sh_side == "trái" else "trái"
                            st.markdown(f"""
                            * **Nguyên nhân:** Lực kéo của balo đang bị dồn lệch sang phía bên kia.
                            * **Hành động điều chỉnh dây:** 
                              * **Nới lỏng** dây đeo bên vai **{sh_side}** khoảng **1.0 - 1.5 cm**.
                              * **Rút ngắn** dây đeo bên vai **{opp_side}** khoảng **1.0 cm** để cân bằng lực.
                            """)
                        else:
                            st.markdown("""
                            * **Nguyên nhân:** Balo đang nặng hoặc quai đeo chưa cân lực.
                            * **Hành động điều chỉnh:** Cân chỉnh lại hai dây đeo dài bằng nhau, áp sát vào lưng.
                            """)
                else:
                    st.error("⚠ Không nhận diện rõ khung người ở một trong hai ảnh. Vui lòng kiểm tra lại và **tải lên ảnh khác** rõ hơn.")
            else:
                res_n, ky_n, head_n, _, valid_n = process_standing_side(img_norm)
                res_p, ky_p, head_p, _, valid_p = process_standing_side(img_pack)
                
                if valid_n and valid_p and res_n is not None and res_p is not None:
                    st.image([res_n, res_p], caption=["Ảnh 1: Trước khi đeo cặp", "Ảnh 2: Khi đeo cặp sách"], use_container_width=True)
                    
                    delta_ky = ky_p - ky_n
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Chỉ số gù ban đầu", f"{ky_n:.1f}")
                    c2.metric("Chỉ số gù khi đeo", f"{ky_p:.1f}", delta=f"{delta_ky:+.1f}")
                    c3.metric("Trọng lượng balo", f"{bag_weight} kg")
                    
                    if delta_ky < 2.0:
                        st.success(" Tải trọng balo không gây gù lưng hay ngả người ra trước quá mức.")
                    else:
                        st.warning(f" Trọng lượng {bag_weight}kg làm tăng độ gù thêm **{delta_ky:+.1f}**.")
                        st.markdown("""
                        * **Lời khuyên:** Balo đang kéo thân trên ngả về trước. Hãy điều chỉnh quai đeo ôm sát lưng, thắt dây đai ngực/bụng để cố định trọng tâm.
                        """)
                else:
                    st.error(" Không nhận diện rõ khung người ở góc chụp nghiêng. Vui lòng **tải lên ảnh khác**.")

# Footer thông báo y khoa
st.markdown("""
    <div class="disclaimer-box">
        <b>Lưu ý quan trọng:</b> Ứng dụng này chỉ hỗ trợ tầm soát và mang tính chất tham khảo học đường, không thay thế cho chẩn đoán từ bác sĩ chuyên khoa y tế.
    </div>
""", unsafe_allow_html=True)
