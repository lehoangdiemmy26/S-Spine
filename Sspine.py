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
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">  S-Spine</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Ứng dụng AI Tầm Soát Biến Dạng Cột Sống Học Đường</p>', unsafe_allow_html=True)

mp_pose = mp.solutions.pose

# =========================================================
# 1. PHÂN TÍCH ĐỨNG CHÍNH DIỆN (ĐO LỆCH VAI / VẸO)
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
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        
        is_back_view = nose.z > ((left_shoulder.z + right_shoulder.z) / 2.0) or nose.visibility < 0.5
        view_text = "Góc nhìn: Sau lưng" if is_back_view else "Góc nhìn: Đối diện"
        
        p_left = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        p_right = (int(right_shoulder.x * w), int(right_shoulder.y * h))

        shoulder_depth_diff = abs(left_shoulder.z - right_shoulder.z)
        is_angle_valid = shoulder_depth_diff < 0.35 

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
        angle_deg = 90.0 if dx == 0 else math.degrees(math.atan(abs(dy) / float(dx)))
            
        if abs(dy) < 3: 
            direction = "Cân bằng"
        elif dy > 0:
            direction = "Xệ VAI TRÁI"
        else:
            direction = "Xệ VAI PHẢI"
            
        return annotated_img, angle_deg, direction, is_angle_valid, view_text

# =========================================================
# 2. PHÂN TÍCH GÓC NGHIÊNG BÊN HÔNG (ĐO ĐỘ GÙ & CỔ)
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
        
        # Chọn bên cơ thể rõ hơn (dựa vào visibility của tai hoặc vai trái/phải)
        use_left = landmarks[mp_pose.PoseLandmark.LEFT_EAR].visibility > landmarks[mp_pose.PoseLandmark.RIGHT_EAR].visibility
        
        idx_ear = mp_pose.PoseLandmark.LEFT_EAR if use_left else mp_pose.PoseLandmark.RIGHT_EAR
        idx_shoulder = mp_pose.PoseLandmark.LEFT_SHOULDER if use_left else mp_pose.PoseLandmark.RIGHT_SHOULDER
        idx_hip = mp_pose.PoseLandmark.LEFT_HIP if use_left else mp_pose.PoseLandmark.RIGHT_HIP
        
        ear = landmarks[idx_ear]
        shoulder = landmarks[idx_shoulder]
        hip = landmarks[idx_hip]
        
        p_ear = (int(ear.x * w), int(ear.y * h))
        p_sh = (int(shoulder.x * w), int(shoulder.y * h))
        p_hip = (int(hip.x * w), int(hip.y * h))
        
        annotated_img = img_np.copy()
        
        # Vẽ các điểm mốc và đường nối Tai - Vai - Hông
        cv2.circle(annotated_img, p_ear, 7, (255, 0, 0), -1)
        cv2.circle(annotated_img, p_sh, 7, (0, 255, 0), -1)
        cv2.circle(annotated_img, p_hip, 7, (0, 0, 255), -1)
        
        cv2.line(annotated_img, p_ear, p_sh, (255, 255, 0), 2)
        cv2.line(annotated_img, p_sh, p_hip, (255, 0, 255), 2)
        
        # Đường thẳng đứng tham chiếu từ hông lên
        vertical_top = (p_hip[0], p_ear[1] - 30)
        cv2.line(annotated_img, p_hip, vertical_top, (200, 200, 200), 1, cv2.LINE_AA)

        # Tính toán độ gù (Kyphosis offset dựa vào độ ngả vai so với đường thẳng hông)
        kyphosis_offset = p_sh[0] - p_hip[0]  # Khoảng cách ngang dịch chuyển
        kyphosis_score = max(0.0, float(kyphosis_offset) / max(1.0, (p_hip[1] - p_sh[1])) * 50.0)
        
        # Tính độ chu đầu về trước (Forward Head Posture - FHP)
        head_offset = p_ear[0] - p_sh[0]
        
        status_side = "Tư thế nghiêng chuẩn"
        if kyphosis_score > 10 or head_offset > 25:
            status_side = "Nguy cơ Gù lưng & Chu đầu"
        elif kyphosis_score > 6:
            status_side = "Hơi khom lưng nhẹ"

        return annotated_img, kyphosis_score, head_offset, status_side, True

# =========================================================
# 3. PHÂN TÍCH TƯ THẾ NGỒI (TỪ SAU LƯNG)
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
        
        p_left_s = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        p_right_s = (int(right_shoulder.x * w), int(right_shoulder.y * h))
        p_left_h = (int(left_hip.x * w), int(left_hip.y * h))
        p_right_h = (int(right_hip.x * w), int(right_hip.y * h))
        
        annotated_img = img_np.copy()
        
        dy_s = p_left_s[1] - p_right_s[1]
        dx_s = abs(p_left_s[0] - p_right_s[0])
        shoulder_angle = 90.0 if dx_s == 0 else math.degrees(math.atan(abs(dy_s) / float(dx_s)))
        
        mid_shoulder_x = (p_left_s[0] + p_right_s[0]) / 2.0
        mid_hip_x = (p_left_h[0] + p_right_h[0]) / 2.0
        torso_width = abs(p_left_s[0] - p_right_s[0])
        offset_x = abs(mid_shoulder_x - mid_hip_x)
        hump_score = (offset_x / max(torso_width, 1)) * 35.0
        
        cv2.circle(annotated_img, p_left_s, 7, (255, 0, 0), -1)
        cv2.circle(annotated_img, p_right_s, 7, (0, 0, 255), -1)
        cv2.line(annotated_img, p_left_s, p_right_s, (0, 255, 0), 2)
        
        status_sit = "Ngồi cân đối"
        if shoulder_angle > 2.5:
            status_sit = "Lệch vai khi ngồi"
        if hump_score > 12:
            status_sit += " & Nguy cơ gù lưng"

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
            if res_img is not None:
                st.image(res_img, caption=f"AI phân tích ({vt})", use_container_width=True)
                m1, m2 = st.columns(2)
                m1.metric("Góc nghiêng vai (α)", f"{angle:.2f}°")
                m2.metric("Trạng thái vai", direction)
                
                if angle < 1.5:
                    st.success("✅ Tư thế chuẩn xác!")
                    show_exercise_recommendations("normal")
                else:
                    st.warning(f"⚠️ Dáng đứng bị **{direction}** lệch **{angle:.2f}°**!")
                    show_exercise_recommendations("warning")
            else:
                st.error("⚠️ Không tìm thấy người trong ảnh.")

    elif analysis_mode == "2. Đứng Nghiêng - Bên Hông (Đo độ gù & cổ)":
        st.subheader("2. Đo Độ Gù & Chu Đầu (Góc Chụp Nghiêng 90 Độ)")
        st.info("💡 **Lưu ý:** Người chụp đứng ngang bên cạnh học sinh để lấy trọn góc nhìn từ Tai - Vai - Hông.")
        input_type = st.radio("Chọn phương thức đầu vào:", ["Tải ảnh lên", "Chụp bằng Camera"], key="t2")
        file = st.file_uploader("Tải ảnh nghiêng:", type=['jpg', 'png', 'jpeg'], key="u2") if input_type == "Tải ảnh lên" else st.camera_input("Chụp ảnh nghiêng", key="c2")
        
        if file:
            img = Image.open(file)
            res_img, kyphosis, head_off, status_side, valid = process_standing_side(img)
            if res_img is not None:
                st.image(res_img, caption="AI phân tích đường cong cột sống bên hông", use_container_width=True)
                s1, s2, s3 = st.columns(3)
                s1.metric("Chỉ số gù lưng", f"{kyphosis:.1f}")
                s2.metric("Độ chu đầu (FHP)", f"{head_off:.1f}px")
                s3.metric("Đánh giá", status_side)
                
                if kyphosis < 6 and head_off < 20:
                    st.success("✅ Đường cong cột sống sinh lý hoàn toàn bình thường!")
                    show_exercise_recommendations("normal")
                else:
                    st.warning(f"⚠️ Phát hiện xu hướng khom vai/cổ rướn trước! Hãy tập trung kéo giãn ngực và lưng trên.")
                    show_exercise_recommendations("warning")
            else:
                st.error("⚠️ Không nhận diện rõ các mốc cơ thể ở góc nghiêng này.")

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
            res_f, angle_f, dir_f, _, _ = process_standing_front(img_f)
            res_s, ky_s, head_s, _, _ = process_standing_side(img_s)
            
            if res_f is not None and res_s is not None:
                st.image([res_f, res_s], caption=["Ảnh 1: Cân bằng vai mang tải", "Ảnh 2: Độ gù khi mang tải"], use_container_width=True)
                m1, m2 = st.columns(2)
                m1.metric("Góc lệch vai mang tải", f"{angle_f:.2f}°")
                m2.metric("Chỉ số gù khi đeo cặp", f"{ky_s:.1f}")
                
                if angle_f < 1.5 and ky_s < 7:
                    st.success(f"✅ Tải trọng {bag_weight}kg an toàn, không gây biến dạng tư thế.")
                    show_exercise_recommendations("normal")
                else:
                    st.warning(f"⚠️ Cặp sách {bag_weight}kg gây áp lực lớn làm lệch vai ({dir_f}) hoặc khom lưng. Nên thu ngắn dây đeo và giảm bớt đồ trong cặp!")
                    show_exercise_recommendations("warning")

    elif analysis_mode == "4. 🪑 Tư Thế Ngồi Học & Đo Độ Gù (Từ sau lưng)":
        st.subheader("4. Tầm Soát Tư Thế Ngồi Học & Lệch Trục Lưng")
        input_type = st.radio("Chọn phương thức đầu vào:", ["Tải ảnh lên", "Chụp bằng Camera"], key="t4")
        file = st.file_uploader("Tải ảnh ngồi:", type=['jpg', 'png', 'jpeg'], key="u4") if input_type == "Tải ảnh lên" else st.camera_input("Chụp ảnh ngồi", key="c4")
        
        if file:
            img = Image.open(file)
            res_img, s_angle, hump, status_sit, valid = process_sitting_back(img)
            if res_img is not None:
                st.image(res_img, caption="AI phân tích tư thế ngồi học", use_container_width=True)
                s1, s2, s3 = st.columns(3)
                s1.metric("Góc lệch vai khi ngồi", f"{s_angle:.2f}°")
                s2.metric("Chỉ số lệch trục lưng", f"{hump:.1f}")
                s3.metric("Đánh giá", status_sit)
                
                if hump < 8 and s_angle < 2.0:
                    st.success("✅ Tư thế ngồi học chuẩn và cân đối.")
                    show_exercise_recommendations("normal")
                else:
                    st.warning("⚠️ Phát hiện tư thế khom lưng hoặc lệch vai khi ngồi học bàn. Hãy điều chỉnh khoảng cách ghế và ánh sáng!")
                    show_exercise_recommendations("warning")
            else:
                st.error("⚠️ Không nhận diện được khung người trong ảnh ngồi.")
