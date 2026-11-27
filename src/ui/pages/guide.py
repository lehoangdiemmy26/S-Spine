# -*- coding: utf-8 -*-
"""Tab hướng dẫn chụp ảnh chuẩn (bản thử nghiệm scroll-linked animation)."""
import streamlit as st

from src.ui.artwork import one_line, spine_svg


def tab_guide():
    st.markdown(one_line("""
    <div class="ss-scroll-progress" aria-hidden="true"></div>
    <section class="ss-hero ss-reveal-zoom">
      <div>
        <div class="ss-h" role="heading" aria-level="2">Chụp một tấm ảnh, thấy tư thế đang lệch ở đâu</div>
        <p>S-Spine tìm vị trí tai, vai, hông trên ảnh rồi tính các góc tư thế. Kết quả chỉ để sàng lọc, không thay thế chẩn đoán của bác sĩ.</p>
        <span class="ss-hint">Bắt đầu ở tab “Đánh giá &amp; phân tích tư thế”</span>
      </div>
      <div>{hero}</div>
    </section>""".replace("{hero}", spine_svg("hero"))), unsafe_allow_html=True)

    st.markdown(one_line("""
    <section class="ss-sec ss-reveal">
      <div class="ss-st" role="heading" aria-level="2">Chụp đúng trong 4 bước</div>
      <ol class="ss-steps">
        <li class="ss-reveal"><div class="n">1</div><b>Mặc gọn</b>Áo thun ôm sát, không áo khoác dày hay áo rộng. Buộc gọn tóc dài để lộ cổ và vai.</li>
        <li class="ss-reveal"><div class="n">2</div><b>Đứng đúng hướng</b>Ảnh sau lưng hoặc chính diện: đứng thẳng đối diện máy. Ảnh nghiêng: quay đúng 90°, thấy đủ từ đầu đến hông. Nhớ chọn đúng “chụp từ phía sau lưng / phía trước” trong app.</li>
        <li class="ss-reveal"><div class="n">3</div><b>Đặt máy</b>Cách 1,5–2 m, ngang tầm ngực/lưng, cố định và ngang bằng (giá đỡ hoặc bàn). Nền có khung cửa hay góc tường giúp app bù khi máy hơi nghiêng.</li>
        <li class="ss-reveal"><div class="n">4</div><b>Chụp nhiều ảnh</b>Chụp 3–5 ảnh cách nhau vài giây. App lấy giá trị trung vị để giảm sai số.</li>
      </ol>
      <div class="ss-warn">Ảnh chưa đạt chuẩn (mờ, tối, người bị xoay, máy nghiêng…) sẽ bị từ chối và app yêu cầu chụp lại.</div>
    </section>"""), unsafe_allow_html=True)

    st.markdown(one_line(f"""
    <section class="ss-ky ss-reveal">
      <div class="ss-reveal-left">{spine_svg("ky")}</div>
      <div class="ss-reveal-right">
        <div class="ss-st" role="heading" aria-level="2">Đo gù lưng: dán 6 nhãn dọc cột sống</div>
        <p>Dán nhãn màu nhỏ lên mỏm gai T2, T4, T6, T8, T10, T12, chụp nghiêng 90°, rồi chạm lần lượt vào từng nhãn trên ảnh. App dựng đường cong qua 6 điểm và tính góc giữa hai tiếp tuyến ở T2 và T12.</p>
        <p>Mặc áo ba lỗ hở lưng ôm sát để dán trực tiếp lên da, không cần cởi áo. Cách tìm từng vị trí có trong mục Tư thế đứng → Gù lưng.</p>
        <div class="ss-note">Đây là góc gù ước tính từ ảnh, không phải góc Cobb và chưa thay thế X-quang.</div>
      </div>
    </section>"""), unsafe_allow_html=True)

    st.markdown(one_line("""
    <section class="ss-sec ss-reveal">
      <div class="ss-st" role="heading" aria-level="2">3 mục kiểm tra</div>
      <ul class="ss-modes">
        <li class="ss-reveal"><b>1. Tư thế đứng</b>Đứng tĩnh (lệch vai, khom, đầu nhô ra trước), đeo cặp sách và gù lưng ước tính từ 6 nhãn</li>
        <li class="ss-reveal"><b>2. Ngồi học</b>Tư thế khi ngồi</li>
        <li class="ss-reveal"><b>3. Trước &amp; sau đeo cặp</b>So sánh hai ảnh của cùng một em</li>
      </ul>
    </section>"""), unsafe_allow_html=True)
