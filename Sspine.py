import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image
import mediapipe as mp

# Thiết lập cấu hình trang Streamlit
st.set_page_config(
    page_title="S-Spine: Tầm soát tư thế học đường",
    page_icon="🦴",
    layout="wide"
)

# Khởi tạo an toàn MediaPipe Pose để tránh lỗi AttributeError ở các phiên bản mới
mp_pose = getattr(mp, "solutions", None)
if mp_pose is not None:
    mp_pose = mp_pose.pose
else:
    import mediapipe.python.solutions.pose as mp_pose_alt
    mp_pose = mp_pose_alt

mp_drawing = mp.solutions.drawing_utils

# --- Giao diện tiêu đề ---
st.title("🦴 S-Spine: Hệ thống tầm soát tư thế học đường")
st.markdown("""
Ứng dụng hỗ trợ phân tích tư thế học sinh (gù lưng, lệch vai, vẹo cột sống) thông qua hình ảnh hoặc webcam bằng công nghệ AI.
""")

# --- Sidebar chọn chế độ ---
st.sidebar.header("Cài đặt phân tích")
app_mode = st.sidebar.selectbox(
    "Chọn chế độ hoạt động",
    ["Giới thiệu & Hướng dẫn", "Phân tích qua Hình ảnh", "Phân tích qua Webcam (Trực tiếp)"]
)

# Hàm tính góc giữa 3 điểm (A, B, C) với B là đỉnh góc
def calculate_angle(a, b, c):
    a = np.array(a)  # Điểm thứ nhất
    b = np.array(b)  # Điểm đỉnh
    c = np.array(c)  # Điểm thứ ba
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

# Hàm phân tích tư thế từ các tọa độ landmark
def analyze_pose(landmarks, image_shape):
    h, w, _ = image_shape
    
    # Lấy tọa độ các điểm mốc quan trọng
    # Vai trái/phải, Hông trái/phải, Tai, Mũi
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
    
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]
    
    left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * w,
                landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * h]
    
    # 1. Đánh giá độ lệch vai (Chênh lệch chiều cao trục Y giữa vai trái và phải)
    shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
    
    # 2. Đánh giá độ nghiêng cột sống / lệch hông
    hip_diff = abs(left_hip[1] - right_hip[1])
    
    # 3. Đánh giá độ cúi đầu / gù lưng (Forward Head / Slouching) dựa trên góc tai - vai - hông
    spine_angle = calculate_angle(left_ear, left_shoulder, left_hip)
    
    # Tổng hợp kết quả và đưa ra cảnh báo
    warnings = []
    status = "Bình thường"
    
    if shoulder_diff > 15:
        warnings.append(f"⚠️ Phát hiện lệch vai rõ rệt (Chênh lệch: {shoulder_diff:.1f}px)")
        status = "Cần chú ý"
    elif shoulder_diff > 8:
        warnings.append(f"⚡ Vai hơi mất cân đối (Chênh lệch: {shoulder_diff:.1f}px)")
        
    if hip_diff > 15:
        warnings.append(f"⚠️ Phát hiện lệch hông / khung chậu (Chênh lệch: {hip_diff:.1f}px)")
        status = "Cần chú ý"
        
    # Góc cột sống nghiêng quá mức tiêu chuẩn gù (thường < 160 độ đối với tư thế đứng thẳng tự nhiên)
    if spine_angle < 155:
        warnings.append(f"⚠️️ Có dấu hiệu gù lưng / chúi đầu ra trước (Góc đo được: {spine_angle:.1f}°)")
        status = "Cần chú ý"

    if not warnings:
        warnings.append("✅ Tư thế ở mức ổn định, cân đối.")

    return {
        "status": status,
        "shoulder_diff": shoulder_diff,
        "hip_diff": hip_diff,
        "spine_angle": spine_angle,
        "warnings": warnings
    }


# --- Chế độ 1: Giới thiệu ---
if app_mode == "Giới thiệu & Hướng dẫn":
    st.subheader("📖 Về ứng dụng S-Spine")
    st.write("""
    Học sinh thường đối mặt với nguy cơ vẹo cột sống, gù lưng và lệch vai do ngồi học sai tư thế hoặc đeo balo lệch một bên. 
    **S-Spine** ứng dụng Thị giác máy tính (Computer Vision) để tự động nhận diện các điểm mốc cơ thể và đưa ra đánh giá nhanh chóng.
    """)
    
    st.markdown("### 🛠 Hướng dẫn sử dụng:")
    st.markdown("""
    1. **Phân tích qua Hình ảnh:** Tải lên một bức ảnh chụp toàn thân (từ phía trước hoặc phía sau) ở góc độ rõ ràng.
    2. **Phân tích qua Webcam:** Đứng trước camera để hệ thống quét và phân tích trực tiếp theo thời gian thực.
    3. **Đọc kết quả:** Xem các chỉ số lệch vai, lệch hông, góc độ cột sống và nhận khuyến nghị điều chỉnh tư thế.
    """)


# --- Chế độ 2: Phân tích qua Hình ảnh ---
elif app_mode == "Phân tích qua Hình ảnh":
    st.subheader("📷 Tải lên hình ảnh tư thế học sinh")
    uploaded_file = st.file_uploader("Chọn tệp ảnh (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Ảnh gốc", use_column_width=True)
            
        with st.spinner("Đang phân tích khung xương và tư thế..."):
            with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
                results = pose.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
                
                if results.pose_landmarks:
                    annotated_image = image_np.copy()
                    mp_drawing.draw_landmarks(
                        annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                    )
                    
                    analysis = analyze_pose(results.pose_landmarks.landmark, image_np.shape)
                    
                    with col2:
                        st.image(annotated_image, caption="Kết quả phân tích khung xương", use_column_width=True)
                        
                    st.markdown("---")
                    st.subheader("📊 Kết quả đánh giá chi tiết:")
                    
                    if analysis["status"] == "Bình thường":
                        st.success(f"Trạng thái: **{analysis['status']}**")
                    else:
                        st.warning(f"Trạng thái: **{analysis['status']}**")
                        
                    for warn in analysis["warnings"]:
                        st.write(f"- {warn}")
                else:
                    st.error("Không thể nhận diện được tư thế người trong ảnh. Vui lòng chọn ảnh chụp toàn thân rõ nét hơn.")


# --- Chế độ 3: Phân tích qua Webcam ---
elif app_mode == "Phân tích qua Webcam (Trực tiếp)":
    st.subheader("📹 Phân tích tư thế trực tiếp qua Webcam")
    st.info("Nhấn nút dưới đây để khởi chạy luồng xử lý video trực tiếp trên giao diện.")
    
    run_webcam = st.checkbox("Bật Camera trực tiếp")
    
    if run_webcam:
        frame_window = st.image([])
        warning_container = st.empty()
        
        cap = cv2.VideoCapture(0)
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened() and run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Không thể truy cập vào camera.")
                    break
                
                # Chuyển đổi màu từ BGR sang RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                    )
                    analysis = analyze_pose(results.pose_landmarks.landmark, image.shape)
                    
                    # Hiển thị trạng thái lên khung hình
                    color = (0, 255, 0) if analysis["status"] == "Bình thường" else (0, 0, 255)
                    cv2.putText(image, f"Trạng thái: {analysis['status']}", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                    
                    with warning_container.container():
                        if analysis["status"] == "Bình thường":
                            st.success(f"Trạng thái hiện tại: {analysis['status']}")
                        else:
                            st.warning(f"Trạng thái hiện tại: {analysis['status']}")
                            for w in analysis["warnings"]:
                                st.write(w)
                
                # Hiển thị video lên Streamlit
                frame_window.image(image, channels="BGR")
                
        cap.release()
