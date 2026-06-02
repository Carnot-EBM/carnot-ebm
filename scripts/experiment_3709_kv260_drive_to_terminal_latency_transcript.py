#!/usr/bin/env python3
"""Entrypoint for Exp 3709 KV260 terminal-candidate latency transcript."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../python")))

from carnot.experiment_3709_kv260_drive_to_terminal_latency_transcript import main


if __name__ == "__main__":  # pragma: no cover
    main()
