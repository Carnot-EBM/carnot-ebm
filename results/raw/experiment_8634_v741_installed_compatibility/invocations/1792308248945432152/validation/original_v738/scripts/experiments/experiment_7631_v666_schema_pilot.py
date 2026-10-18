#!/usr/bin/env python3
"""Thin entrypoint for the Exp7631 V666 paired schema pilot."""

from __future__ import annotations

import os

# Keep the orchestrator CPU-only. The owned model server receives its device in
# the worker path after the Exp7630 lease and inventory rechecks pass.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from carnot.experiment_7631_v666_schema_pilot import main


if __name__ == "__main__":
    raise SystemExit(main())
