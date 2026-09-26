import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math

# Cấu hình tên tab và logo đầu trang
st.set_page_config(
    page_title="S-Spine | Tầm soát góc nghiêng",
    layout="wide"
)

# Hiển thị logo tròn ngay đầu trang web S-Spine
st.image("logo.png", width=160)

# Giao diện Tông Trắng Sáng Y Tế (Clean Light Medical Theme)
st.markdown("""
    <style>
    /* Nền chính của toàn ứng dụng - Trắng tinh sạch sẻ */
    .stApp {
        background-color: #FFFFFF;
        color: #1A202C;
    }
    
    /* Thanh Sidebar bên trái - Trắng xám nhẹ phân biệt rõ ràng */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Tiêu đề chính S-Spine nổi bật màu xanh Y tế */
    .main-title {
        color: #0284C7;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0px;
    }
    
    /* Chữ phụ đề */
    .sub-title {
        color: #475569;
        font-size: 1.05rem;
        font-weight: 500;
        margin-bottom: 25px;
    }
    
    /* Ô tải ảnh màu trắng viền xanh nhạt chuẩn Y khoa */
    [data-testid="stFileUploadDropzone"] {
        background-color: #F8FAFC !important;
        border: 2px dashed #38BDF8 !important;
        border-radius: 12px !important;
    }
    
    /* Điều chỉnh các ô chỉ số Metric */
    [data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Tiêu đề chính
st.markdown('<h1 class="main-title">🩺 S-Spine</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Ứng dụng AI Tầm Soát Biến Dạng Cột Sống Học Đường | Lượng giác & Vector (Toán 10-11)</p>', unsafe_allow_html=True)

# =========================================================
# HÀM NHẬN DIỆN MỎM VAI & TÍNH GÓC VECTOR (OPENCV TƯƠNG THÍCH 100%)
# =========================================================
def process_and_analyze(image_pil):
    img_np = np.array(image_pil.convert('RGB'))
    h, w, _ = img_np.shape
    
    # Chuyển ảnh sang xám để xử lý tìm đường vai
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Phân tích vùng vai (vùng giữa thân trên)
    shoulder_region = edges[int(h*0.25):int(h*0.45), int(w*0.15):int(w*0.85)]
    
    y_indices, x_indices = np.where(shoulder_region > 0)
    
    if len(x_indices) < 10:
        # Tọa độ mặc định giả định nếu ảnh quá tối/khó quét biên
        p1 = (int(w * 0.35), int(h * 0.35))
        p2 = (int(w * 0.65), int(h * 0.35))
    else:
        left_x = int(w * 0.15 + np.min(x_indices))
        right_x = int(w * 0.15 + np.max(x_indices))
        
        left_y_idx = np.where(x_indices == np.min(x_indices))[0]
        right_y_idx = np.where(x_indices == np.max(x_indices))[0]
        
        left_y = int(h * 0.25 + np.mean(y_indices[left_y_idx]))
        right_y = int(h * 0.25 + np.mean(y_indices[right_y_idx]))
        
        p1 = (left_x, left_y)
        p2 = (right_x, right_y)
    
    # Vẽ đường xanh nối 2 mỏm vai
    annotated_img = img_np.copy()
    cv2.line(annotated_img, p1, p2, (0, 255, 0), 4)
    cv2.circle(annotated_img, p1, 8, (255, 0, 0), -1)
    cv2.circle(annotated_img, p2, 8, (255, 0, 0), -1)
    
    # Tính góc nghiêng bằng Vector Tích vô hướng
    x1, y1 = p1
    x2, y2 = p2
    v = (x2 - x1, y2 - y1)
    u = (1, 0)
    
    dot_product = u[0] * v[0] + u[1] * v[1]
    magnitude_v = math.sqrt(v[0]**2 + v[1]**2)
    
    if magnitude_v == 0:
        return annotated_img, 0.0, "Cân bằng"
        
    cos_angle = max(0.0, min(1.0, abs(dot_product) / magnitude_v))
    angle_deg = math.degrees(math.acos(cos_angle))
    
    if abs(y1 - y2) < 3:
        direction = "Cân bằng"
    elif y1 > y2:
        direction = "Xệ VAI TRÁI"
    else:
        direction = "Xệ VAI PHẢI"
        
    return annotated_img, angle_deg, direction

# =========================================================
# GIAO DIỆN HỆ THỐNG S-SPINE
# =========================================================
st.sidebar.header("⚙️ Thông số Tải trọng")
bag_weight = st.sidebar.number_input("Trọng lượng cặp sách (kg):", min_value=0.0, value=4.5, step=0.5)

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Ảnh Đứng Tĩnh")
    file1 = st.file_uploader("Tải ảnh đứng thả lỏng (mặt lưng):", type=['jpg', 'png', 'jpeg'], key="1")
    
with col2:
    st.subheader("2. Ảnh Đeo Cặp Sách")
    file2 = st.file_uploader("Tải ảnh đeo cặp (mặt lưng):", type=['jpg', 'png', 'jpeg'], key="2")

st.markdown("---")

# CHẾ ĐỘ 1: CHỈ TẢI ẢNH TĨNH
if file1 and not file2:
    img1 = Image.open(file1)
    res_img, angle1, dir1 = process_and_analyze(img1)
    
    if res_img is not None:
        st.image(res_img, caption="S-Spine quét đường mỏm vai tự nhiên", width=400)
        st.header("📊 PHÂN TÍCH TƯ THẾ TỰ NHIÊN (ẢNH TĨNH)")
        
        m1, m2 = st.columns(2)
        m1.metric("Góc nghiêng vai (α1)", f"{angle1:.2f}°")
        m2.metric("Trạng thái vai", dir1)
        
        st.subheader("📌 Chẩn Đoán & Khuyến Cáo Y Khoa")
        if angle1 < 1.5:
            st.success("✅ **Tư thế chuẩn:** Khung xương vai cân bằng tự nhiên.")
        elif 1.5 <= angle1 < 3.5:
            st.warning(f"⚠️ **CẢNH BÁO MỨC NHẸ:** Dáng đứng tự nhiên bị **{dir1}** lệch **{angle1:.2f}°**!")
            st.info("💡 **Lời khuyên:** Điều chỉnh tư thế ngồi học, tránh dồn lực một bên và thực hiện dãn cơ lưng.")
        else:
            st.error(f"🚨 **BÁO ĐỘNG ĐỎ:** Lệch vai tự nhiên đạt **{angle1:.2f}°** ({dir1})!")
            st.error("🩺 **LỜI KHUYÊN Y KHOA:** Gia đình nên đưa học sinh đi khám chuyên khoa **Cơ xương khớp** để chụp X-quang đo **Góc Cobb**!")

# CHẾ ĐỘ 2: CHỈ TẢI ẢNH ĐEO CẶP
elif file2 and not file1:
    img2 = Image.open(file2)
    res_img, angle2, dir2 = process_and_analyze(img2)
    
    if res_img is not None:
        st.image(res_img, caption="S-Spine quét đường mỏm vai khi mang tải", width=400)
        st.header("📊 PHÂN TÍCH TƯ THẾ MANG TẢI (ẢNH ĐEO CẶP)")
        
        adjust_cm = angle2 * 0.85
        m1, m2 = st.columns(2)
        m1.metric("Góc nghiêng vai mang tải (α2)", f"{angle2:.2f}°")
        m2.metric("Trạng thái vai", dir2)
        
        st.subheader("📌 Chẩn Đoán & Khuyến Cáo Y Khoa")
        if angle2 < 1.5:
            st.success("✅ **Trạng thái tốt:** Cột sống chịu tải cân bằng.")
        elif 1.5 <= angle2 < 3.5:
            st.warning(f"⚠️ **CẢNH BÁO:** Đeo cặp làm **{dir2}** lệch **{angle2:.2f}°**!")
            st.info(f"💡 **Khắc phục:** Thu ngắn dây đeo vai bên bị xệ khoảng **{adjust_cm:.1f} cm**.")
        else:
            st.error(f"🚨 **BÁO ĐỘNG ĐỎ:** Tải trọng cặp {bag_weight}kg làm lệch vai nghiêm trọng **{angle2:.2f}°** ({dir2})!")

# CHẾ ĐỘ 3: TẢI ĐỦ 2 ẢNH (SO SÁNH Δα)
elif file1 and file2:
    img1 = Image.open(file1)
    img2 = Image.open(file2)
    
    res1, angle1, dir1 = process_and_analyze(img1)
    res2, angle2, dir2 = process_and_analyze(img2)
    
    if res1 is not None and res2 is not None:
        st.image([res1, res2], caption=["Ảnh 1: Đứng tĩnh", "Ảnh 2: Đeo cặp"], width=400)
        st.header("📊 KẾT QUẢ SO SÁNH BIẾN DẠNG ĐỘNG (Δα)")
        
        delta_alpha = abs(angle2 - angle1)
        adjust_cm = delta_alpha * 0.85
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Góc vai tĩnh (α1)", f"{angle1:.2f}°")
        m2.metric("Góc vai mang tải (α2)", f"{angle2:.2f}°")
        m3.metric("Độ biến dạng động (Δα)", f"{delta_alpha:.2f}°", delta_color="inverse")
        
        st.subheader("📌 Tổng Kết & Khuyến Cáo Y Khoa")
        if delta_alpha < 1.5:
            st.success("✅ **Trạng thái an toàn:** Tải trọng cặp phân bổ đều.")
        elif 1.5 <= delta_alpha < 3.5:
            st.warning(f"⚠️ **CẢNH BÁO MỨC TRUNG BÌNH:** Cặp sách làm lệch vai thêm **{delta_alpha:.2f}°** ({dir2})!")
            st.info(f"💡 **Đề xuất:** Thu ngắn dây đeo bên xệ **{adjust_cm:.1f} cm**.")
        else:
            st.error(f"🚨 **BÁO ĐỘNG ĐỎ - NGUY CƠ VẸO CỘT SỐNG:** Chỉ số biến dạng động Δα = **{delta_alpha:.2f}°**!")
            st.error("🩺 **LỜI KHUYÊN Y KHOA:** Giảm ngay trọng lượng cặp. Gia đình nên đưa em đến bệnh viện chụp X-quang xác định **Góc Cobb**!")

else:
    st.info("👆 Bạn hãy tải ảnh lên để S-Spine nhận diện mỏm vai và tính toán nhé!")
