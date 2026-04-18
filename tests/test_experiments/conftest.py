"""Ensure the project root is on sys.path (mirrors tests/conftest.py).

pytest discovers test modules bottom-up; this file guarantees the
``experiments`` and ``code`` packages import cleanly when tests are run
from this subdirectory directly.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_CODE_DIR = _ROOT / "code"
if _CODE_DIR.is_dir() and "code" not in sys.modules:
    pkg = types.ModuleType("code")
    pkg.__path__ = [str(_CODE_DIR)]
    sys.modules["code"] = pkg
