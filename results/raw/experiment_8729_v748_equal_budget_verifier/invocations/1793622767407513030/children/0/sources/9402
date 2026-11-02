#!/usr/bin/env python3
"""Thin CLI for REQ-REPORT-7653 live ARC generalization research."""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from carnot.experiment_7653_v667_arc_live_generalization import main


if __name__ == "__main__":
    raise SystemExit(main())
