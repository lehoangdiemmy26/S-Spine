import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

# Đảm bảo khai báo mp_pose chuẩn như thế này:
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
# Cấu hình trang
st.set_page_config(
    page_title="S-Spine | Tầm soát góc nghiêng",
    page_icon="gen-n-z8308486911094_3cf6e9f66d814eabd93c0c5ae610e055-modified.png",
    layout="wide"
)

# Logo
st.image("gen-n-z8308486911094_3cf6e9f66d814eabd93c0c5ae610e055-modified.png", width=160)

# Giao diện Theme Y tế & Bài tập
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; color: #1A202C; }
    [data-testid="stSidebar"] { background-color: #F8FAFC; border-right: 1px solid #E2E8F0; }
    .main-title { color: #0284C7; font-weight: 800; font-size: 2.8rem; margin-bottom: 0px; }
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

# Khởi tạo & Cache MediaPipe Pose để tiết kiệm RAM/CPU trên Streamlit
@st.cache_resource
def get_mp_pose():
    return mp.solutions.pose.Pose(
        static_image_mode=True, 
        model_complexity=1,
        min_detection_confidence=0.5
    )

mp_pose = mp.solutions.pose

# =========================================================
# HÀM XỬ LÝ VÀ NHẬN DIỆN MỎM VAI
# =========================================================
def process_and_analyze(image_pil):
    max_size = 800
    w_orig, h_orig = image_pil.size
    if max(w_orig, h_orig) > max_size:
        scale = max_size / float(max(w_orig, h_orig))
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        image_pil = image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

    img_np = np.array(image_pil.convert('RGB'))
    h, w, _ = img_np.shape
    
    pose = get_mp_pose()
    results = pose.process(img_np)
    
    if not results.pose_landmarks:
        return None, 0.0, "Không nhận diện được", False
        
    landmarks = results.pose_landmarks.landmark
    
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    
    # Kiểm tra góc chụp chuẩn
    shoulder_depth_diff = abs(left_shoulder.z - right_shoulder.z)
    is_angle_valid = shoulder_depth_diff < 0.35  
    
    p1 = (int(left_shoulder.x * w), int(left_shoulder.y * h))   # Vai Trái
    p2 = (int(right_shoulder.x * w), int(right_shoulder.y * h)) # Vai Phải

    annotated_img = img_np.copy()
    cv2.line(annotated_img, p1, p2, (0, 255, 0), 3)
    cv2.circle(annotated_img, p1, 6, (255, 0, 0), -1)
    cv2.circle(annotated_img, p2, 6, (255, 0, 0), -1)
    
    x1, y1 = p1
    x2, y2 = p2
    v = (x2 - x1, y2 - y1)
    u = (1, 0)
    
    dot_product = u[0] * v[0] + u[1] * v[1]
    magnitude_v = math.sqrt(v[0]**2 + v[1]**2)
    
    if magnitude_v == 0:
        return annotated_img, 0.0, "Cân bằng", is_angle_valid
        
    cos_angle = max(0.0, min(1.0, abs(dot_product) / magnitude_v))
    angle_deg = math.degrees(math.acos(cos_angle))
    
    if abs(y1 - y2) < 3:
        direction = "Cân bằng"
    elif y1 > y2:
        direction = "Xệ VAI TRÁI"
    else:
        direction = "Xệ VAI PHẢI"
        
    return annotated_img, angle_deg, direction, is_angle_valid

# =========================================================
# HÀM MÔ TẢ VÀ HIỂN THỊ BÀI TẬP VẬT LÝ TRỊ LIỆU KÈM MINH HỌA
# =========================================================
def show_exercise_recommendations(status_type, angle_val):
    st.markdown("---")
    st.subheader("🏋️ Lộ Trình Luyện Tập & Phục Hồi Cá Nhân Hóa (S-Spine Care)")
    
    if status_type == "normal":
        st.success("🎉 **Tư thế của bạn rất chuẩn!** Hãy duy trì thói quen sinh hoạt tốt và thực hiện 2 bài tập giãn cơ nhẹ nhàng này sau mỗi 45 phút ngồi học:")
        
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
        st.warning(f"⚠️ **Góc lệch {angle_val:.2f}°:** Phát hiện xu hướng lệch vai/võng lưng do phân bổ trọng lực không đều. Dưới đây là chuỗi **3 Bài tập Vật lý trị liệu phục hồi tại nhà** dành riêng cho bạn:")
        
        ex1, ex2, ex3 = st.columns(3)
        
        with ex1:
            st.markdown("""
            <div class="exercise-card">
                <div class="exercise-title">1. Tư thế Con Mèo - Con Bò (Cat-Cow Pose)</div>
                <p><b>Tác dụng:</b> Cải thiện độ linh hoạt cột sống ngực và thắt lưng.</p>
                <p><b>Cách tập:</b> Quỳ 4 điểm (tay & gối). Hít vào võng lưng ngẩng đầu (Con bò), thở ra cong lưng hóp bụng (Con mèo).</p>
                <p>⏱️ <b>Liều lượng:</b> 10 - 12 lần/ngày.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with ex2:
            st.markdown("""
            <div class="exercise-card">
                <div class="exercise-title">2. Chống Tường Mở Vai (Wall Push/Stretch)</div>
                <p><b>Tác dụng:</b> Tăng cường sức mạnh cơ lưng trên, kéo giãn cơ ngực bị co thắt.</p>
                <p><b>Cách tập:</b> Đứng đối diện tường cách 0.5m, chống 2 tay lên tường. Nhấn nhẹ ngực về phía tường để căng vai.</p>
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
# GIAO DIỆN TABS
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
                <li>Tránh áo phông rộng thùng xình che mất đường cong lưng & hông.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with g2:
        st.markdown("""
        <div class="guide-card">
            <h4>🧍 2. Tư Thế Đứng</h4>
            <ul>
                <li>Đứng nghiêng <b>đúng 90°</b> so với camera.</li>
                <li>Khoanh hai tay trước ngực để không che cột sống ngực.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with g3:
        st.markdown("""
        <div class="guide-card">
            <h4>📸 3. Góc Máy Camera</h4>
            <ul>
                <li>Đặt điện thoại <b>ngang tầm ngực</b> (khoảng cách 1.5m - 2m).</li>
                <li>Camera đặt song song cơ thể, không chúc máy lên hoặc nghiêng xuống.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab_app:
    st.sidebar.header("⚙️ Thông số Tải trọng")
    bag_weight = st.sidebar.number_input("Trọng lượng cặp sách (kg):", min_value=0.0, value=4.5, step=0.5)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Ảnh Đứng Tĩnh")
        file1 = st.file_uploader("Tải ảnh đứng thả lỏng:", type=['jpg', 'png', 'jpeg'], key="1")
        
    with col2:
        st.subheader("2. Ảnh Đeo Cặp Sách")
        file2 = st.file_uploader("Tải ảnh đeo cặp:", type=['jpg', 'png', 'jpeg'], key="2")

    st.markdown("---")

    # CHẾ ĐỘ 1: TẢI ẢNH TĨNH
    if file1 and not file2:
        img1 = Image.open(file1)
        res_img, angle1, dir1, valid1 = process_and_analyze(img1)
        
        if res_img is not None:
            if not valid1:
                st.warning("⚠️ **CẢNH BÁO GÓC CHỤP:** Ảnh chụp bị lệch/chéo góc! Vui lòng xem tab 'Hướng Dẫn Chụp Ảnh Chuẩn' để đạt độ chính xác cao nhất.")
            
            st.image(res_img, caption="AI quét vị trí mỏm vai", use_container_width=True)
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
                st.error(f"🚨 **BÁO ĐỘNG ĐỎ:** Lệch vai nghiêm trọng **{angle1:.2f}°** ({dir1})! Gia đình nên đưa học sinh đi khám chuyên khoa Cơ xương khớp để chụp X-quang đo **Góc Cobb**.")
                show_exercise_recommendations("warning", angle1)
        else:
            st.error("⚠️ AI không tìm thấy cơ thể người trong ảnh. Vui lòng thử lại với ảnh rõ hơn!")

    # CHẾ ĐỘ 2: TẢI ẢNH ĐEO CẶP
    elif file2 and not file1:
        img2 = Image.open(file2)
        res_img, angle2, dir2, valid2 = process_and_analyze(img2)
        
        if res_img is not None:
            if not valid2:
                st.warning("⚠️ **CẢNH BÁO GÓC CHỤP:** Camera bị chéo góc! Vui lòng chụp lại đúng góc nghiêng 90°.")
            
            st.image(res_img, caption="AI quét vị trí mỏm vai khi mang tải", use_container_width=True)
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
        
        res1, angle1, dir1, valid1 = process_and_analyze(img1)
        res2, angle2, dir2, valid2 = process_and_analyze(img2)
        
        if res1 is not None and res2 is not None:
            if not valid1 or not valid2:
                st.warning("⚠️ **CẢNH BÁO GÓC CHỤP:** Có ảnh chụp bị xéo góc! Vui lòng xem tab 'Hướng Dẫn Chụp Ảnh Chuẩn' để đạt độ chính xác cao nhất.")
            
            st.image([res1, res2], caption=["Ảnh 1: Đứng tĩnh", "Ảnh 2: Đeo cặp"], use_container_width=True)
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
                st.error(f"🚨 **BÁO ĐỘNG ĐỎ:** Chỉ số biến dạng Δα = **{delta_alpha:.2f}°**! Giảm ngay trọng lượng cặp và nên đi chụp X-quang kiểm tra **Góc Cobb**.")
                show_exercise_recommendations("warning", delta_alpha)
        else:
            st.error("⚠️ AI không phân tích được 1 trong 2 ảnh. Vui lòng kiểm tra lại ảnh đầu vào!")

    else:
        st.info("👆 Tải ảnh lên để S-Spine nhận diện và phân tích ngay nhé!")
