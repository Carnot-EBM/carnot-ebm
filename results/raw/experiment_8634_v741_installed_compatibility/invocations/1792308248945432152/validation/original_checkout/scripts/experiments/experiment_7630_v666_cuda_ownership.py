#!/usr/bin/env python3
"""Thin entrypoint for the Exp7630 no-model CUDA ownership qualification."""

from __future__ import annotations

import os

# Fix CPU preflight before importing Carnot code that could import a numerical runtime.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from carnot.experiment_7630_v666_cuda_ownership import main


if __name__ == "__main__":
    raise SystemExit(main())
