# -*- coding: utf-8 -*-
# Chạy app bằng venv Python 3.10 đặt ở đường dẫn KHÔNG DẤU.
# Lý do: MediaPipe không đọc được model khi venv nằm trong thư mục có dấu/unicode
# (ví dụ "...\Diễm My\.venv" -> FileNotFoundError pose_landmark_cpu.binarypb).
$ErrorActionPreference = "Stop"
chcp 65001 | Out-Null
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$VenvPy = "C:\Users\admin\.venvs\s-spine-py310\Scripts\python.exe"
if (!(Test-Path $VenvPy)) {
  uv venv --python 3.10 C:\Users\admin\.venvs\s-spine-py310
  uv pip install -r requirements.txt --python C:\Users\admin\.venvs\s-spine-py310\Scripts\python.exe
}
& $VenvPy -m streamlit run app.py
