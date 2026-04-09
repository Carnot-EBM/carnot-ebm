#!/bin/bash
# Autoresearch v6: Constraint-Based Reasoning via Ising/thrml
# Runs experiments 42, 46, 45 unattended overnight
set -euo pipefail
cd /home/ianblenke/github.com/ianblenke/carnot

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== AUTORESEARCH V6 STARTED ==="

# Exp 42: Arithmetic constraint verification
log "--- Experiment 42: Arithmetic Constraints ---"
sg render -c 'TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 PYTHONUNBUFFERED=1 .venv/bin/python scripts/experiment_42_arithmetic_ising.py' 2>&1 | tee /tmp/exp42.log || true
log "Exp 42 done"

git add scripts/experiment_42_arithmetic_ising.py data/ 2>/dev/null || true
git commit -m "Exp 42: arithmetic constraint verification via Ising/thrml

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" 2>/dev/null || true
git push 2>/dev/null || true

# Exp 46: Scale SAT to 500+ vars
log "--- Experiment 46: Scale SAT ---"
sg render -c 'TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 PYTHONUNBUFFERED=1 .venv/bin/python scripts/experiment_46_scale_sat.py' 2>&1 | tee /tmp/exp46.log || true
log "Exp 46 done"

git add scripts/experiment_46_scale_sat.py 2>/dev/null || true
git commit -m "Exp 46: scale SAT to 500+ variables

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" 2>/dev/null || true
git push 2>/dev/null || true

# Exp 45: Logical consistency verification
log "--- Experiment 45: Logical Consistency ---"
sg render -c 'TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 PYTHONUNBUFFERED=1 .venv/bin/python scripts/experiment_45_logical_consistency.py' 2>&1 | tee /tmp/exp45.log || true
log "Exp 45 done"

git add scripts/experiment_45_logical_consistency.py 2>/dev/null || true
git commit -m "Exp 45: logical consistency verification via Ising/thrml

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" 2>/dev/null || true
git push 2>/dev/null || true

log "=== AUTORESEARCH V6 COMPLETE ==="
