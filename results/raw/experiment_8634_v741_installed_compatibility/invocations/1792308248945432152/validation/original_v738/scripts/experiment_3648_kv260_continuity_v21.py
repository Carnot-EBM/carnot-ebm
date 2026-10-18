#!/usr/bin/env python3
"""Entrypoint for Exp 3648 KV260 continuity v21."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../python")))

from carnot.experiment_3648_kv260_continuity_v21 import main


if __name__ == "__main__":  # pragma: no cover
    main()
