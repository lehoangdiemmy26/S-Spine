# -*- coding: utf-8 -*-
from src.domain.levels import level_of, ear_level, lower_side, reliability
from src.analysis.front import classify_front
from src.analysis.side import classify_side
from src.analysis.kyphosis import kyphosis_angle, kyphosis_stability


def test_levels():
    assert level_of(1.9, (2.0, 4.0)) == 0
    assert level_of(2.0, (2.0, 4.0)) == 1
    assert level_of(5.0, (2.0, 4.0)) == 2
    assert ear_level(55) == 0
    assert ear_level(45) == 1
    assert ear_level(30) == 2
    assert lower_side(0.1) == "cân bằng"
    assert lower_side(1.2) == "phải"
    assert lower_side(-2.0) == "trái"
    assert reliability([]) == "Tốt"
    assert reliability(["a"]) == "Trung bình"
    assert reliability(["a", "b"]) == "Thấp"


def test_classify():
    lv, txt = classify_front(1.0, "cân bằng", 1.0)
    assert lv == 0
    lv2, _ = classify_front(5.0, "phải", 1.0)
    assert lv2 == 2
    lv3, _ = classify_side(60.0, 2.0, "stand")
    assert lv3 == 0
    lv4, _ = classify_side(30.0, 12.0, "stand")
    assert lv4 == 2


def test_kyphosis_straight():
    # 6 điểm gần thẳng đứng, hơi cong -> góc nhỏ nhưng tính được
    pts = [(100.0, 10.0), (102.0, 30.0), (104.0, 50.0), (104.0, 70.0), (102.0, 90.0), (100.0, 110.0)]
    a = kyphosis_angle(pts)
    assert a is not None and a >= 0
    s = kyphosis_stability(pts)
    assert s >= 0
