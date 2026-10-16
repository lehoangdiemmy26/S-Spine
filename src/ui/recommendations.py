# -*- coding: utf-8 -*-
"""Gợi ý vận động theo mức. Chỉ mang tính tham khảo, không phải chỉ định y khoa."""
import streamlit as st


def card(title, lines):
    body = "".join(f"<p>{x}</p>" for x in lines)
    st.markdown(
        f'<div class="exercise-card"><div class="exercise-title">{title}</div>{body}</div>',
        unsafe_allow_html=True,
    )


def show_recommendations(level):
    st.markdown("---")
    st.subheader("Gợi ý vận động hỗ trợ tư thế")
    if level == 0:
        st.success("**Tư thế trong ngưỡng tham khảo.** Hãy đứng dậy vận động nhẹ sau mỗi 30-45 phút học.")
        c1, c2 = st.columns(2)
        with c1:
            card("1. Xoay vai & mở ngực",
                 ["Đứng thẳng, xoay tròn hai vai từ trước ra sau.", "⏱ Gợi ý: 10-15 lần mỗi hướng."])
        with c2:
            card("2. Nghiêng cổ giãn cơ",
                 ["Nghiêng đầu sang một bên, tay hỗ trợ nhẹ.", "⏱ Gợi ý: giữ 15 giây mỗi bên."])
    elif level == 1:
        st.warning("**Cần theo dõi:** nên chụp lại sau 1-2 tuần để so sánh và chú ý tư thế khi học.")
        c1, c2 = st.columns(2)
        with c1:
            card("1. Tư thế Mèo - Bò",
                 ["Quỳ bốn điểm; hít vào võng lưng, thở ra cong lưng.", "⏱ Gợi ý: 10-12 lần/ngày."])
        with c2:
            card("2. Chống tường mở vai",
                 ["Đứng cách tường khoảng 0.5 m, chống tay, nhẹ nhàng đưa ngực về phía tường.",
                  "⏱ Gợi ý: giữ 20 giây x 3 lần."])
    else:
        st.error(
            "**Nên kiểm tra thêm:** kết quả vượt ngưỡng tham khảo. Hãy báo phụ huynh/giáo viên và đến cơ sở y tế "
            "chuyên khoa cột sống để được khám. Ảnh chụp **không** thể chẩn đoán bệnh."
        )
        st.caption("Các động tác dưới đây chỉ là vận động nhẹ, nên hỏi ý kiến bác sĩ trước khi tập thường xuyên.")
        c1, c2, c3 = st.columns(3)
        with c1:
            card("1. Nằm sấp nâng ngực nhẹ",
                 ["Nằm sấp, nâng nhẹ ngực và tay khỏi sàn, siết cơ lưng.", "⏱ Gợi ý: giữ 10 giây x vài lần."])
        with c2:
            card("2. Giãn & mở ngực",
                 ["Nằm ngửa trên khăn cuộn đặt dưới lưng ngực, mở rộng hai tay.", "⏱ Gợi ý: 2-3 phút mỗi ngày."])
        with c3:
            card("3. Plank khuỷu tay (nhẹ)",
                 ["Chống khuỷu tay, giữ thân thẳng; có thể tựa đầu gối nếu mỏi.", "⏱ Gợi ý: giữ 20-30 giây x vài lần."])
