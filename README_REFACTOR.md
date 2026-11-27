# S-Spine (bản refactor)

Code gốc giữ nguyên ở `s_spine_v15.py` để đối chiếu (không sửa).

Bản tách file nằm ở `app.py` + `src/`.

## Chạy đúng cách (quan trọng)

Dùng Python 3.10 + `uv`, venv đặt ở đường dẫn **không dấu**:

```powershell
chcp 65001
$env:PYTHONUTF8='1'
uv python install 3.10
uv venv --python 3.10 C:\Users\admin\.venvs\s-spine-py310
uv pip install -r requirements.txt --python C:\Users\admin\.venvs\s-spine-py310\Scripts\python.exe
C:\Users\admin\.venvs\s-spine-py310\Scripts\python.exe -m streamlit run app.py
```

Hoặc chạy nhanh: `.\run-app.ps1`

> Vì sao lỗi "Không khởi tạo được MediaPipe Pose" dù đã ghim `mediapipe==0.10.14`?
> Thư mục project có dấu (`Diễm My`) nên venv `.venv` nằm trong đó cũng dính dấu.
> MediaPipe (C++) không mở được file model `pose_landmark_cpu.binarypb` qua đường dẫn
> unicode -> `Pose()` ném `FileNotFoundError`, `get_pose()` trả về `None`.
> Venv mới ở `C:\Users\admin\.venvs\s-spine-py310` (toàn ASCII) khắc phục điều này cho
> cả bản cũ (`streamlit run s_spine_v15.py`) lẫn bản mới (`streamlit run app.py`).
> Khi `get_pose()` thất bại, app mới hiện thêm expander "Chi tiết kỹ thuật" để xem lỗi thật.

## Bố cục

- `src/config/` : hằng số, ngưỡng, id mốc MediaPipe
- `src/domain/` : struct kết quả + xếp mức xanh/vàng/đỏ
- `src/vision/` : đọc ảnh, MediaPipe, bù máy nghiêng, kiểm tra chất lượng ảnh
- `src/analysis/` : đo góc sau lưng/chính diện, nghiêng, gù lưng, gộp nhiều ảnh
- `src/ui/` : theme/CSS, widget, trang hướng dẫn, các mục đo, hàm main
- `src/utils/` : xuất CSV
- `tests/` : kiểm tra các hàm thuần (không cần camera/model)

## Triển khai Streamlit Cloud

- `requirements.txt` : pin `mediapipe==0.10.14`, `opencv-python-headless==4.10.0.84`, `numpy<2`
- `packages.txt` : `libgl1`, `libglib2.0-0` (thư viện hệ thống cho OpenCV/MediaPipe)

> App chỉ sàng lọc tư thế học đường, không chẩn đoán. Các ngưỡng trong
> `src/config/settings.py` là ngưỡng tham khảo của nhóm, cần hiệu chỉnh
> bằng dữ liệu đo thật.

## Bản sửa lấy mốc sai ở tư thế ngồi (v16)

| File | Thay đổi |
|---|---|
| `src/config/landmarks.py` | thêm `L_KNEE`, `R_KNEE` |
| `src/config/settings.py` | thêm `SIT_TORSO_MIN_RATIO`, `SIT_ROLL_MIN_LINES`; thêm "Mốc hông có vẻ sai" vào `BLOCKING` |
| `src/vision/geometry.py` | `facing_of` có dự phòng đầu gối; thêm `pick_side` (tai + vai + hông) |
| `src/vision/pose.py` | thêm `detect_refined` (dò lại trên vùng cắt quanh người) |
| `src/vision/roll.py` | `estimate_roll(..., min_lines=3)` |
| `src/analysis/side.py` | dùng `pick_side`, đầu gối; thêm `sit_side_sanity` (chỉ khi ngồi) |
| `src/analysis/pipeline.py` | ảnh ngồi dùng `detect_refined` và đòi nhiều đường thẳng hơn mới tự bù máy nghiêng |
| `src/ui/modes/manual_side.py` (mới) | tự chạm 3 mốc tai, vai, hông |
| `src/ui/modes/sitting.py` | ô tích "Tự chọn mốc bằng tay" |
| `tests/test_sitting.py` (mới) | test cho các hàm mới |

Hai ngưỡng `SIT_TORSO_MIN_RATIO = 1.5` và `SIT_ROLL_MIN_LINES = 8` là ước lượng, cần hiệu chỉnh bằng ảnh ngồi thật.
