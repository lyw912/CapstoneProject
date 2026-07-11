"""QueryEngine configuration entry point.

QueryEngine uses the same root Settings instance as Flask, MediaEngine, and the
fusion supervisor so provider switches and MindSpider safety flags cannot drift
inside one Coordinator run.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
project_root = str(_PROJECT_ROOT)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Settings, reload_settings, settings  # noqa: E402

__all__ = ["Settings", "reload_settings", "settings"]
