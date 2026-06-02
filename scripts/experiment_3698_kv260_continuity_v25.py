#!/usr/bin/env python3
"""Entrypoint for Exp 3698 KV260 continuity v25."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../python")))

from carnot.experiment_3698_kv260_continuity_v25 import main


if __name__ == "__main__":  # pragma: no cover
    main()
