# -*- coding: utf-8 -*-
"""Tab phương pháp & giới hạn: đo gì, bằng cách nào, hiểu tới đâu (bản v2)."""
import streamlit as st

from src.config.settings import EAR_OK, EAR_WATCH, TH
from src.ui.artwork import one_line
from src.ui.theme import eyebrow, page_head


def _threshold_cell(key):
    t1, t2 = TH[key]
    return t1, t2


def tab_method():
    page_head("Phương pháp & giới hạn", "App đo gì, đo bằng cách nào và nên hiểu kết quả đến đâu.")

    st.markdown(one_line("""
    <div class="ss-method-hero">
      <div class="t">Ảnh 2D thành góc tư thế trong 3 bước</div>
      <div class="d">Mô hình MediaPipe Pose tìm các mốc tai, vai, hông trên ảnh. Mọi chỉ số là
      <b>góc tính trên ảnh 2D</b> (đơn vị độ) từ các mốc do mô hình ước lượng, sau đó xếp mức
      xanh / vàng / đỏ theo ngưỡng tham khảo của nhóm.</div>
      <div class="ss-pipeline">
        <div class="step"><div class="n">1</div><b>Tìm mốc</b><span>Tai, vai, hông bằng MediaPipe Pose. Mốc mờ hoặc ngoài khung thì báo chụp lại.</span></div>
        <div class="step"><div class="n">2</div><b>Tính góc</b><span>Lệch vai so với phương ngang, lệch thân so với phương đứng, góc tai–vai, góc gù từ 6 nhãn.</span></div>
        <div class="step"><div class="n">3</div><b>Xếp mức</b><span>Dưới t1 là xanh, đến t2 là vàng, trên t2 là đỏ. Ngưỡng là tham khảo, chưa phải chuẩn y khoa.</span></div>
      </div>
    </div>"""), unsafe_allow_html=True)

    eyebrow("1 · Các chỉ số đang đo")
    st.markdown(one_line(f"""
    <div class="ss-cards">
      <div class="ss-card"><span class="ss-tag">Ảnh sau lưng / chính diện</span><div class="ss-ico">1</div><b>Góc lệch vai</b><p>Góc giữa đường nối hai vai và phương ngang. Ngưỡng xanh &lt; {TH['shoulder_tilt'][0]:g}°, vàng đến {TH['shoulder_tilt'][1]:g}°.</p></div>
      <div class="ss-card"><span class="ss-tag">Ảnh sau lưng / chính diện</span><div class="ss-ico">2</div><b>Lệch thân so với hông</b><p>Góc đường từ giữa hông đến giữa vai so với phương thẳng đứng. Ngưỡng xanh &lt; {TH['trunk_shift'][0]:g}°.</p></div>
      <div class="ss-card"><span class="ss-tag">Ảnh nghiêng</span><div class="ss-ico">3</div><b>Đầu nhô ra trước (góc tai–vai)</b><p>Góc giữa phương ngang và đường vai → tai. Xanh từ {EAR_OK:g}° trở lên. Gần với CVA nhưng <b style="display:inline">không phải CVA chuẩn</b> vì dùng mốc vai thay C7.</p></div>
      <div class="ss-card"><span class="ss-tag">Ảnh nghiêng</span><div class="ss-ico">4</div><b>Nghiêng thân</b><p>Góc đường hông → vai so với phương thẳng đứng, dương là nghiêng ra trước. Ngưỡng đứng / ngồi khác nhau.</p></div>
      <div class="ss-card warn"><span class="ss-tag">Ảnh nghiêng có nhãn</span><div class="ss-ico">5</div><b>Gù lưng (ước tính từ ảnh)</b><p>Chạm 6 nhãn T2 → T12. App dựng đường cong bậc 3 qua 6 điểm, tính góc giữa hai tiếp tuyến ở T2 và T12 theo tinh thần DIPA. <b style="display:inline">Không phải góc Cobb</b>, chưa có ngưỡng cho học sinh nên không phân mức.</p></div>
    </div>"""), unsafe_allow_html=True)

    eyebrow("2 · App giảm sai số bằng gì")
    st.markdown(one_line("""
    <div class="ss-cards">
      <div class="ss-card"><div class="ss-ico">A</div><b>Bù máy nghiêng</b><p>Ước lượng từ các đường thẳng đứng ở nền rồi xoay ảnh bù lại trong 0,5–8°. Quá nghiêng thì yêu cầu đặt lại máy hoặc nhập tay.</p></div>
      <div class="ss-card"><div class="ss-ico">B</div><b>Kiểm tra ảnh đầu vào</b><p>Mốc phải nằm trong khung và đủ rõ. Ảnh tối, mờ, phân giải thấp, người quá nhỏ hoặc xoay người đều bị từ chối.</p></div>
      <div class="ss-card"><div class="ss-ico">C</div><b>Đối chiếu viền cơ thể</b><p>So độ đối xứng của mốc vai với viền segmentation để phát hiện áo rộng, tóc che vai hoặc mốc đặt sai.</p></div>
      <div class="ss-card"><div class="ss-ico">D</div><b>Nhiều ảnh, lấy trung vị</b><p>Chụp 3–5 ảnh cách nhau vài giây, lấy trung vị để giảm nhiễu tư thế nhất thời. Dao động lớn thì cảnh báo.</p></div>
    </div>"""), unsafe_allow_html=True)

    eyebrow("3 · Ngưỡng phân loại")
    st.markdown(one_line("""
    <div class="ss-box"><b>Đây là ngưỡng tham khảo của nhóm</b>, chưa có nguồn chuẩn và chưa được kiểm chứng lâm sàng.
    Dùng để sàng lọc học đường, không dùng để chẩn đoán. Riêng ngưỡng góc tai–vai không phải ngưỡng CVA chuẩn.</div>
    <div class="ss-legend">
      <span class="ss-pill ss-pill-green">Xanh · Bình thường</span>
      <span class="ss-pill ss-pill-yellow">Vàng · Cần theo dõi</span>
      <span class="ss-pill ss-pill-red">Đỏ · Nên kiểm tra thêm</span>
    </div>"""), unsafe_allow_html=True)

    rows = [
        ("Lệch vai", "độ", "shoulder_tilt"),
        ("Lệch thân (sau lưng)", "độ", "trunk_shift"),
        ("Nghiêng thân khi đứng", "độ", "trunk_lean_stand"),
        ("Nghiêng thân khi ngồi", "độ", "trunk_lean_sit"),
        ("Tăng lệch vai khi đeo cặp", "độ", "delta_tilt"),
        ("Tăng nghiêng thân khi đeo cặp", "độ", "delta_trunk"),
        ("Giảm góc tai–vai khi đeo cặp", "độ", "delta_ear_drop"),
        ("Cặp / cân nặng", "%", "bag_ratio_pct"),
    ]
    body = "".join(
        f'<tr><td><b>{n}</b><br><span style="color:#4A5D75;font-size:.8rem">({u})</span></td>'
        f'<td class="num"><span class="ss-dot" style="background:#2E9E5B"></span>&lt; {TH[k][0]:g}</td>'
        f'<td class="num"><span class="ss-dot" style="background:#E3A008"></span>{TH[k][0]:g} – {TH[k][1]:g}</td>'
        f'<td class="num"><span class="ss-dot" style="background:#D64545"></span>&gt; {TH[k][1]:g}</td></tr>'
        for n, u, k in rows
    )
    body += (
        f'<tr><td><b>Góc tai–vai</b><br><span style="color:#4A5D75;font-size:.8rem">(độ, càng lớn càng thẳng)</span></td>'
        f'<td class="num"><span class="ss-dot" style="background:#2E9E5B"></span>≥ {EAR_OK:g}</td>'
        f'<td class="num"><span class="ss-dot" style="background:#E3A008"></span>{EAR_WATCH:g} – {EAR_OK:g}</td>'
        f'<td class="num"><span class="ss-dot" style="background:#D64545"></span>&lt; {EAR_WATCH:g}</td></tr>'
    )
    st.markdown(one_line(f"""
    <div class="ss-table-wrap"><table class="ss-table">
      <thead><tr><th scope="col">Chỉ số</th><th scope="col">Xanh</th><th scope="col">Vàng</th><th scope="col">Đỏ</th></tr></thead>
      <tbody>{body}</tbody></table></div>"""), unsafe_allow_html=True)

    with st.expander("Ví dụ cách đọc bảng", expanded=False):
        t1, t2 = _threshold_cell("shoulder_tilt")
        st.markdown(
            f"Lệch vai **1,5°** (< {t1:g}°) là xanh. **{t1:g}–{t2:g}°** là vàng, cần theo dõi và chụp lại sau 1–2 tuần. "
            f"Trên **{t2:g}°** là đỏ, nên báo phụ huynh/giáo viên và đi khám chuyên khoa. "
            f"Góc tai–vai ngược chiều: càng nhỏ càng nhô ra trước."
        )

    eyebrow("4 · Giới hạn cần biết")
    st.markdown(one_line("""
    <ul class="ss-limits">
      <li><b>Không thay thế X-quang hay khám lâm sàng.</b> Không chẩn đoán vẹo hay gù. Hai chỉ số ảnh nghiêng không phải là đo độ gù.</li>
      <li><b>Vẹo là biến dạng 3D có xoay đốt sống.</b> Nghiệm pháp cúi người (Adams) với scoliometer mới đánh giá được góc xoay thân (thường chuyển khám từ 5–7°).</li>
      <li><b>Mốc do mô hình ước lượng.</b> Áo rộng, tóc che vai, cặp che mốc, ảnh tối/mờ vẫn có thể lệch mốc. App chỉ cảnh báo được một phần.</li>
      <li><b>Bù nghiêng có điều kiện.</b> Chỉ làm được khi nền có đủ đường thẳng đứng. Nếu không, hãy đặt máy cố định, ngang bằng.</li>
      <li><b>Một ảnh là một khoảnh khắc.</b> Chưa cho biết đó có phải thói quen lâu dài hay không. Nên chụp 3–5 ảnh và theo dõi theo thời gian.</li>
      <li><b>Quyền riêng tư.</b> App không ghi ảnh ra đĩa hay cơ sở dữ liệu; ảnh chỉ xử lý tạm trong bộ nhớ máy chủ.</li>
    </ul>
    <div class="ss-method-cta"><b>Khi nào nên đi khám?</b> Mức đỏ, hoặc mức vàng kéo dài 1–2 tuần chụp lại vẫn vàng/đỏ, hoặc gia đình lo lắng — hãy đến cơ sở y tế chuyên khoa cột sống. Mang theo ảnh và file CSV từ app để bác sĩ đối chiếu.</div>"""), unsafe_allow_html=True)
