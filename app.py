# -*- coding: utf-8 -*-
"""Điểm vào của app S-Spine (bản refactor).

Giữ mỏng nhất có thể: mọi logic nằm trong src/, ở đây chỉ gọi main().
Chạy bằng:  .venv\\Scripts\\streamlit run app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui.app import main

if __name__ == "__main__":
    main()
