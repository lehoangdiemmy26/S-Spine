import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

# =========================================================
# CẤU HÌNH GIAO DIỆN & CSS
# =========================================================
st.set_page_config(page_title="S-Spine - Tầm soát tư thế học đường", page_icon="🦴", layout="wide")

st.markdown("""
    <style>
    .exercise-card {
        background-color: #f8f9fa;
        border-left: 5px solid #4CAF50;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .exercise-title {
        font-weight: bold;
        color: #2e7d32;
        margin-bottom: 5px;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 15px;
        border-radius: 5px;
        margin-top: 30px;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_standing_front(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False) as pose:
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            return None, 0, "", False
        
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        ls_y, rs_y = left_shoulder.y * h, right_shoulder.y * h
        diff_y = abs(ls_y - rs_y)
        
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        direction = "Vai phải cao hơn" if rs_y < ls_y else "Vai trái cao hơn"
        if diff_y < 10:
            direction = "Cân đối"
            
        return annotated_image, diff_y, direction, True

def process_standing_side(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False) as pose:
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            return None, 0, 0, False
        
        landmarks = results.pose_landmarks.landmark
        ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        
        kyphosis_val = abs(ear.x - shoulder.x) * 100
        
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        return annotated_image, kyphosis_val, 0, True

# =========================================================
# GIAO DIỆN CHÍNH (STREAMLIT)
# =========================================================
st.title("🦴 S-Spine: Hệ thống Tầm soát & Đánh giá Tư thế Cột sống")
st.markdown("Ứng dụng hỗ trợ phân tích tư thế học đường, phát hiện lệch vai, đánh giá góc gù và tác động của balo.")

st.sidebar.header("🧭 Điều hướng tính năng")
choice = st.sidebar.radio(
    "Chọn mục tầm soát tư thế:",
    [
        "1. Đứng Tĩnh (Đo lệch vai / độ gù)",
        "2. Đeo Cặp Sách (Đánh giá tải trọng - Hỗ trợ 1 hoặc 2 ảnh)",
        "3. Tư Thế Ngồi Học & Đo Độ Gù (Hỗ trợ 1 hoặc 2 ảnh)",
        "4. So Sánh Trước & Sau Khi Đeo Cặp (Chênh lệch & Chỉnh dây)"
    ]
)

# ---------------------------------------------------------
# MỤC 1: ĐỨNG TĨNH
# ---------------------------------------------------------
if "1. Đứng Tĩnh" in choice:
    st.header("1. Tầm soát tư thế Đứng Tĩnh")
    st.write("Tải lên 1 ảnh chụp tư thế đứng thẳng (chính diện hoặc nghiêng) để phân tích nhanh độ cân đối vai hoặc mức độ khom lưng.")
    
    uploaded_file = st.file_uploader("Tải lên ảnh đứng:", type=["jpg", "jpeg", "png"], key="m1")
    pose_type = st.radio("Góc chụp:", ["Chính diện (Kiểm tra lệch vai)", "Nghiêng (Kiểm tra độ gù)"], key="r1")
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        if "Chính diện" in pose_type:
            res_img, diff, dir_s, valid = process_standing_front(img)
            if valid:
                st.image(res_img, caption="Kết quả phân tích chính diện", use_container_width=True)
                st.metric("Chênh lệch độ cao vai", f"{diff:.1f} px")
                st.info(f"Đánh giá: {dir_s}")
            else:
                st.error("Không nhận diện được khung xương, vui lòng thử lại với ảnh rõ hơn.")
        else:
            res_img, ky, _, valid = process_standing_side(img)
            if valid:
                st.image(res_img, caption="Kết quả phân tích góc nghiêng", use_container_width=True)
                st.metric("Chỉ số độ khom/gù", f"{ky:.1f}")
            else:
                st.error("Không nhận diện được khung xương, vui lòng thử lại với ảnh rõ hơn.")

# ---------------------------------------------------------
# MỤC 2: ĐEO CẶP SÁCH (HỖ TRỢ 1 HOẶC 2 ẢNH)
# ---------------------------------------------------------
elif "2. Đeo Cặp Sách" in choice:
    st.header("2. Đánh giá khi Đeo Cặp Sách (Linh hoạt 1 hoặc 2 ảnh)")
    
    mode = st.radio("Chọn chế độ kiểm tra:", ["Kiểm tra với 1 ảnh đơn", "So sánh với 2 ảnh (Trước & Sau khi đeo)"], key="mode_m2")
    bag_weight = st.number_input("Trọng lượng balo ước tính (kg):", min_value=0.5, max_value=20.0, value=3.5, step=0.5, key="w2")

    if mode == "Kiểm tra với 1 ảnh đơn":
        uploaded_file = st.file_uploader("Tải lên 1 ảnh đeo balo:", type=["jpg", "jpeg", "png"], key="m2_single")
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            res_img, ky_val, _, valid = process_standing_side(img)
            if valid:
                st.image(res_img, caption="Phân tích tư thế đeo balo", use_container_width=True)
                st.metric("Trọng lượng balo hiện tại", f"{bag_weight} kg")
                if bag_weight > 5.0:
                    st.warning(f"⚠️ Trọng lượng balo ({bag_weight} kg) khá nặng, có nguy cơ kéo lệch trọng tâm cột sống.")
                else:
                    st.success("✅ Trọng lượng balo ở mức an toàn.")
                st.markdown("""
                    <div class="exercise-card">
                        <div class="exercise-title">💡 Lời khuyên khi sử dụng balo:</div>
                        <ul>
                            <li>Đeo sát balo vào lưng, dùng đầy đủ cả hai quai để phân bổ đều lực.</li>
                            <li>Trọng lượng balo học sinh tốt nhất không quá 10-15% trọng lượng cơ thể.</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error("⚠️ Không nhận diện được khung xương từ ảnh tải lên.")
    else: # So sánh 2 ảnh
        col_a, col_b = st.columns(2)
        with col_a:
            img_norm_file = st.file_uploader("Ảnh 1: Không đeo balo", type=["jpg", "jpeg", "png"], key="m2_norm")
        with col_b:
            img_pack_file = st.file_uploader("Ảnh 2: Khi đeo balo", type=["jpg", "jpeg", "png"], key="m2_pack")
            
        if img_norm_file and img_pack_file:
            bytes_n = np.asarray(bytearray(img_norm_file.read()), dtype=np.uint8)
            bytes_p = np.asarray(bytearray(img_pack_file.read()), dtype=np.uint8)
            img_n = cv2.imdecode(bytes_n, 1)
            img_p = cv2.imdecode(bytes_p, 1)
            
            res_n, ky_n, _, valid_n = process_standing_side(img_n)
            res_p, ky_p, _, valid_p = process_standing_side(img_p)
            
            if valid_n and valid_p:
                st.image([res_n, res_p], caption=["Ảnh 1: Không đeo balo", "Ảnh 2: Khi đeo balo"], use_container_width=True)
                delta_ky = ky_p - ky_n
                c1, c2, c3 = st.columns(3)
                c1.metric("Độ gù ban đầu", f"{ky_n:.1f}")
                c2.metric("Độ gù khi đeo balo", f"{ky_p:.1f}", delta=f"{delta_ky:+.1f}")
                c3.metric("Trọng lượng balo", f"{bag_weight} kg")
                
                if delta_ky > 2.0:
                    st.warning(f"⚠️ Balo nặng {bag_weight}kg làm tăng độ khom lưng rõ rệt khi mang.")
                else:
                    st.success("✅ Ảnh hưởng của balo không đáng kể đến độ gù lưng.")
            else:
                st.error("⚠️ Không nhận diện được mốc cơ thể ở một trong hai ảnh.")

# ---------------------------------------------------------
# MỤC 3: TƯ THẾ NGỒI HỌC & ĐO ĐỘ GÙ (HỖ TRỢ 1 HOẶC 2 ẢNH)
# ---------------------------------------------------------
elif "3. Tư Thế Ngồi Học" in choice:
    st.header("3. Tầm soát Tư Thế Ngồi Học (Linh hoạt 1 hoặc 2 ảnh)")
    
    mode_m3 = st.radio("Chọn chế độ kiểm tra:", ["Kiểm tra với 1 ảnh ngồi", "So sánh với 2 ảnh (Sai tư thế vs Đúng tư thế / Hoặc 2 góc độ)"], key="mode_m3")

    if mode_m3 == "Kiểm tra với 1 ảnh ngồi":
        uploaded_file = st.file_uploader("Tải lên 1 ảnh ngồi học:", type=["jpg", "jpeg", "png"], key="m3_single")
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            res_img, ky_val, _, valid = process_standing_side(img)
            if valid:
                st.image(res_img, caption="Phân tích tư thế ngồi", use_container_width=True)
                st.metric("Chỉ số khom lưng / cúi đầu", f"{ky_val:.1f}")
                if ky_val > 15:
                    st.warning("⚠️ Phát hiện tư thế cúi đầu hoặc khom lưng sâu khi ngồi học.")
                else:
                    st.success("✅ Tư thế ngồi học tương đối thẳng.")
                st.markdown("""
                    <div class="exercise-card">
                        <div class="exercise-title">💡 Gợi ý cải thiện tư thế ngồi:</div>
                        <ul>
                            <li>Giữ khoảng cách từ mắt đến bàn khoảng 25–30 cm.</li>
                            <li>Đảm bảo lưng thẳng, vuông góc với mặt ghế.</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error("⚠️ Không nhận diện rõ mốc cơ thể khi ngồi.")
    else: # So sánh 2 ảnh
        col_a, col_b = st.columns(2)
        with col_a:
            img_file1 = st.file_uploader("Ảnh 1 (Ví dụ: Ngồi cúi)", type=["jpg", "jpeg", "png"], key="m3_img1")
        with col_b:
            img_file2 = st.file_uploader("Ảnh 2 (Ví dụ: Ngồi thẳng)", type=["jpg", "jpeg", "png"], key="m3_img2")
            
        if img_file1 and img_file2:
            bytes_1 = np.asarray(bytearray(img_file1.read()), dtype=np.uint8)
            bytes_2 = np.asarray(bytearray(img_file2.read()), dtype=np.uint8)
            img_1 = cv2.imdecode(bytes_1, 1)
            img_2 = cv2.imdecode(bytes_2, 1)
            
            res_1, ky_1, _, valid_1 = process_standing_side(img_1)
            res_2, ky_2, _, valid_2 = process_standing_side(img_2)
            
            if valid_1 and valid_2:
                st.image([res_1, res_2], caption=["Ảnh 1", "Ảnh 2"], use_container_width=True)
                diff_val = ky_2 - ky_1
                c1, c2 = st.columns(2)
                c1.metric("Chỉ số gù Ảnh 1", f"{ky_1:.1f}")
                c2.metric("Chỉ số gù Ảnh 2", f"{ky_2:.1f}", delta=f"{diff_val:+.1f}")
                
                st.info("💡 So sánh giúp người dùng dễ dàng nhận thấy sự khác biệt giữa tư thế ngồi gục và tư thế ngồi chuẩn thẳng lưng.")
            else:
                st.error("⚠️ Không nhận diện đủ mốc cơ thể ở các ảnh so sánh.")

# ---------------------------------------------------------
# MỤC 4: SO SÁNH TRƯỚC & SAU KHI ĐEO CẶP (2 ẢNH SO SÁNH)
# ---------------------------------------------------------
elif "4. So Sánh Trước & Sau Khi Đeo Cặp" in choice:
    st.header("4. So Sánh Chuyên Sâu (Trước & Sau Khi Đeo Cặp)")
    st.write("Tải lên 2 ảnh (Ảnh 1: Đứng thẳng tự nhiên | Ảnh 2: Khi mang balo) để đo độ chênh lệch.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        img_norm_file = st.file_uploader("Ảnh 1: Đứng thẳng tự nhiên", type=["jpg", "jpeg", "png"], key="norm")
    with col_b:
        img_pack_file = st.file_uploader("Ảnh 2: Khi mang balo", type=["jpg", "jpeg", "png"], key="pack")
        
    bag_weight = st.number_input("Trọng lượng balo (kg):", min_value=0.5, max_value=20.0, value=3.5, step=0.5, key="w4")
    comparison_type = st.radio("Góc so sánh:", ["Chính diện (Lệch vai)", "Nghiêng (Độ gù)"], key="r4")
    
    if img_norm_file and img_pack_file:
        bytes_norm = np.asarray(bytearray(img_norm_file.read()), dtype=np.uint8)
        bytes_pack = np.asarray(bytearray(img_pack_file.read()), dtype=np.uint8)
        img_norm = cv2.imdecode(bytes_norm, 1)
        img_pack = cv2.imdecode(bytes_pack, 1)
        
        if "Chính diện" in comparison_type:
            res_n, diff_n, dir_n, valid_n = process_standing_front(img_norm)
            res_p, diff_p, dir_p, valid_p = process_standing_front(img_pack)
            
            if valid_n and valid_p:
                st.image([res_n, res_p], caption=["Ảnh 1: Đứng thẳng tự nhiên", "Ảnh 2: Đeo cặp khi đứng"], use_container_width=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Lệch vai ban đầu", f"{diff_n:.1f} px")
                c2.metric("Lệch vai khi đeo balo", f"{diff_p:.1f} px", delta=f"{diff_p - diff_n:+.1f} px")
                c3.metric("Trọng lượng balo", f"{bag_weight} kg")
                
                st.markdown("### 📊 Đánh Giá Tác Động Lên Cột Sống Cổ & Vai:")
                if "cao hơn" in dir_p:
                    st.warning(f"⚠️ Phát hiện hiện tượng mất cân đối lực khi mang ({dir_p}).")
                    st.markdown(f"""
                        <div class="exercise-card">
                            <div class="exercise-title">💡 Gợi ý điều chỉnh dây balo:</div>
                            <ul>
                                <li>Phát hiện hiện tượng mất cân đối lực khi mang ({dir_p}).</li>
                                <li><b>Cách khắc phục:</b> Hãy kiểm tra và cân chỉnh lại độ dài hai quai balo cho đều nhau, ôm sát lưng, không để lệch sang một bên vai.</li>
                                <li><b>Tải trọng khuyến nghị:</b> Balo học sinh không nên vượt quá 10-15% trọng lượng cơ thể (Hiện tại: {bag_weight} kg).</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success("✅ Hai vai cân đối, tư thế mang balo tốt.")
            else:
                st.error("⚠️ Không nhận diện được đầy đủ mốc cơ thể ở một trong hai ảnh so sánh.")

        else: # So sánh góc nghiêng bên hông
            res_n_s, ky_n, _, valid_n_s = process_standing_side(img_norm)
            res_p_s, ky_p, _, valid_p_s = process_standing_side(img_pack)
            
            if valid_n_s and valid_p_s:
                st.image([res_n_s, res_p_s], caption=["Ảnh 1: Đứng thẳng tự nhiên (Nghiêng)", "Ảnh 2: Đeo cặp khi đứng (Nghiêng)"], use_container_width=True)
                delta_ky = ky_p - ky_n
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Chỉ số gù ban đầu", f"{ky_n:.1f}")
                c2.metric("Chỉ số gù khi mang balo", f"{ky_p:.1f}", delta=f"{delta_ky:+.1f}")
                c3.metric("Trọng lượng balo", f"{bag_weight} kg")
                
                st.markdown("### 📊 Đánh Giá Tác Động Lên Cột Sống Ngực:")
                if delta_ky < 2.0:
                    st.success("✅ Trọng lượng balo phù hợp, không làm tăng đáng kể độ khom/gù lưng khi đứng.")
                else:
                    st.warning(f"⚠️ Trọng lượng {bag_weight}kg làm tăng độ gù ngực lên **{delta_ky:+.1f} đơn vị**. Balo đang kéo thân người bạn ngả về trước.")
                    st.markdown("""
                        <div class="exercise-card">
                            <div class="exercise-title">💡 Lời khuyên tư thế & balo:</div>
                            <ul>
                                <li>Sử dụng balo có đệm lưng và dây đeo ngực/hông trợ lực để phân bổ đều trọng lượng.</li>
                                <li>Đeo sát balo vào lưng, không để quai bị chùng xuống thấp qua mông.</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("⚠️ Không nhận diện rõ mốc cơ thể ở ảnh nghiêng so sánh.")

# =========================================================
# KHUYẾN CÁO Y TẾ (DISCLAIMER) CUỐI TRANG
# =========================================================
st.markdown("""
    <div class="disclaimer-box">
    <b>⚠️ KHUYẾN CÁO Y TẾ QUAN TRỌNG:</b> Ứng dụng <b>S-Spine</b> sử dụng trí tuệ nhân tạo (AI) và công nghệ thị giác máy tính nhằm hỗ trợ tầm soát, cảnh báo sớm nguy cơ lệch vai, gù lưng và tư thế học đường. Kết quả trên mang tính chất tham khảo, không thay thế cho chẩn đoán hoặc chỉ định từ bác sĩ chuyên khoa cột sống / vật lý trị liệu.
    </div>
""", unsafe_allow_html=True)
