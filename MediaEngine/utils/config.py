# -*- coding: utf-8 -*-
"""
MediaEngine 包内配置入口。

与项目根目录 `config.py` 使用同一套 Settings，避免重复定义；
供 `MediaEngine.utils` 与 `from .utils.config import Settings` 使用。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保可导入项目根目录下的 config 模块（例如从子目录直接运行脚本时）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_root = str(_PROJECT_ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)

from config import Settings, reload_settings, settings  # noqa: E402

__all__ = ["Settings", "settings", "reload_settings"]
