"""Make ``code.*`` importable from the repo root.

``code`` is used as a package name in this project (``code/A2Q``,
``code/I2QA``, ``code/iterative``). Adding the repo root to ``sys.path``
lets ``import code.iterative.metrics`` work in tests run via ``pytest``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
