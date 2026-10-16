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
