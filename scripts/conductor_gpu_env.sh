#!/usr/bin/env bash
# Source this before launching GPU-tagged experiments.
# RETRO-012 fix: propagates CARNOT_FORCE_LIVE=1 into conductor subprocesses.
#
# Usage (in a wrapper script or Makefile):
#   source scripts/conductor_gpu_env.sh
#   python scripts/research_conductor.py
#
# Why: research_conductor.py never sets CARNOT_FORCE_LIVE in the subprocess
# environment it spawns for GPU experiments.  Sourcing this script injects
# the variable into the calling shell so child processes inherit it.
# This closed three consecutive milestones (2026.04.29, 2026.05.06, 2026.05.20)
# of idle GPUs despite hardware being ready (Exp 352: is_live_capable=True).
export CARNOT_FORCE_LIVE=1
