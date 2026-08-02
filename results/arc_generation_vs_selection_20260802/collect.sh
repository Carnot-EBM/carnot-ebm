#!/usr/bin/env bash
set -uo pipefail
cd /home/ianblenke/github.com/ianblenke/carnot
export CARNOT_REPO=$PWD
export SCRATCH_E3=/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/e3_scratch
export GVS_GAME_TIMEOUT_S=2700
export GVS_ENGINE_TIMEOUT_S=90
L=results/arc_generation_vs_selection_20260802/out/collect.log
echo "=== MAIN CORPORA $(date -u +%FT%TZ) ===" >> "$L"
.venv/bin/python results/arc_generation_vs_selection_20260802/run.py 2>&1 | grep -vE "INFO|WARNING|^2026-" >> "$L"
echo "=== BESTOFN $(date -u +%FT%TZ) ===" >> "$L"
.venv/bin/python results/arc_generation_vs_selection_20260802/run_bestofn.py 2>&1 | grep -vE "INFO|WARNING|^2026-" >> "$L"
echo "=== ALL DONE $(date -u +%FT%TZ) ===" >> "$L"
