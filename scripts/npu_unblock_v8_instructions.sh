#!/bin/bash
# AMD XDNA NPU Unblock v8 — run as human operator (not conductor)
#
# RETRO-067: The AMD XDNA NPU has been blocked for 7 consecutive milestones
# (Exps 292, 303, 314, 335, 435, 511, 589) by missing ninja and openblas.
# This script documents the exact commands needed to unblock the NPU path.
# Run these once as the human operator; then re-run the NPU experiment.

# Install missing prerequisites:
sudo pacman -S ninja openblas

# Verify:
ninja --version && echo 'ninja OK'
python3 -c 'import openblas' 2>/dev/null || echo 'openblas: check pkg'

# Then re-run NPU experiment:
JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_293_npu_constraint_model.py
