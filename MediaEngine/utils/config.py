# -*- coding: utf-8 -*-
"""
MediaEngine package configuration entry.

Uses the same Settings as the project root directory `config.py` to avoid duplicate definitions;
Used for `MediaEngine.utils` and `from .utils.config import Settings`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the config module in the project root directory can be imported (e.g., when running scripts directly from subdirectories)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_root = str(_PROJECT_ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)

from config import Settings, reload_settings, settings  # noqa: E402

__all__ = ["Settings", "settings", "reload_settings"]
