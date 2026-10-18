#!/usr/bin/env bash
# session_startup.sh — Pre-session GPU health check + CARNOT_FORCE_LIVE propagation.
#
# RETRO-007: zombie GPU processes from prior sessions consume VRAM at session start.
# RETRO-008: no standardised pre-flight check before the research conductor launches.
# RETRO-015: CARNOT_FORCE_LIVE=1 was never exported into conductor subprocess environments,
#            causing four consecutive milestones of idle GPUs. This script now exports
#            CARNOT_FORCE_LIVE=1 so all child processes inherit it when this script is
#            sourced before the conductor runs.
#
# What this script does:
#   1. Exports CARNOT_FORCE_LIVE=1 (RETRO-015 fix — belt-and-suspenders: also sources
#      conductor_gpu_env.sh which sets the same variable).
#   2. Checks nvidia-smi availability; exits cleanly if absent (CI environments).
#   3. Lists all CUDA GPUs; counts how many are detected.
#   4. Detects zombie GPU processes via DualGPUMonitor (0% utilisation, >100 MiB VRAM).
#      Falls back to nvidia-smi --query-compute-apps parsing if Python import fails.
#   5. If --kill-zombies is set: sends SIGKILL to each zombie PID.
#      If --dry-run is set: prints PIDs but does NOT kill.
#   6. Prints a single canonical summary line:
#        SESSION STARTUP: n_gpus=X zombies=Y killed=Z all_healthy=True/False
#   7. Exits 0 always (health check, not a blocking gate).
#
# Usage:
#   source scripts/session_startup.sh              # env propagation (RETRO-015 fix)
#   ./scripts/session_startup.sh [--dry-run] [--kill-zombies]  # health check mode
#
# Spec: REQ-INFRA-008, REQ-INFRA-017,
#       SCENARIO-INFRA-012, SCENARIO-INFRA-013, SCENARIO-INFRA-021

set -uo pipefail
# Note: -e is intentionally omitted — this script must exit 0 even on errors.

# ---------------------------------------------------------------------------
# RETRO-015 fix: export CARNOT_FORCE_LIVE=1 into all child processes.
# This MUST come first, before any nvidia-smi calls or zombie detection,
# so that even if the health-check section fails the env var is already set.
# ---------------------------------------------------------------------------

_SESSION_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$_SESSION_SCRIPT_DIR/conductor_gpu_env.sh" ]; then
    # shellcheck source=scripts/conductor_gpu_env.sh
    source "$_SESSION_SCRIPT_DIR/conductor_gpu_env.sh"
fi
export CARNOT_FORCE_LIVE=1
echo "[session_startup] CARNOT_FORCE_LIVE=1 exported at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

DRY_RUN=0
KILL_ZOMBIES=0

for arg in "$@"; do
    case "${arg}" in
        --dry-run)
            DRY_RUN=1
            ;;
        --kill-zombies)
            KILL_ZOMBIES=1
            ;;
        *)
            echo "Unknown argument: ${arg}" >&2
            ;;
    esac
done

# ---------------------------------------------------------------------------
# nvidia-smi availability check (SCENARIO-INFRA-013: CI-safe)
# ---------------------------------------------------------------------------

if ! command -v nvidia-smi &>/dev/null; then
    echo "nvidia-smi not found — skipping GPU health check"
    echo "SESSION STARTUP: n_gpus=0 zombies=0 killed=0 all_healthy=False"
    exit 0
fi

# ---------------------------------------------------------------------------
# Count visible GPUs
# ---------------------------------------------------------------------------

GPU_COUNT=0
if nvidia-smi -L &>/dev/null; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | grep -c "GPU " || true)
fi

# ---------------------------------------------------------------------------
# Detect zombie processes via DualGPUMonitor, with CSV fallback
# ---------------------------------------------------------------------------

ZOMBIE_PIDS=""

# Try Python DualGPUMonitor first (most accurate zombie detection)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ZOMBIE_PIDS=$(
    PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}" \
    python3 -c "
import sys
try:
    from carnot.pipeline.dual_gpu_monitor import DualGPUMonitor
    zombies = DualGPUMonitor().detect_zombies()
    pids = [str(z.pid) for z in zombies]
    print(' '.join(pids))
except Exception:
    pass
" 2>/dev/null || true
)

# Fallback: parse nvidia-smi compute-apps CSV if Python import failed or returned nothing.
# A process is a zombie candidate if it holds memory; we cannot determine utilisation
# from this query alone, so we only flag them when the GPU shows 0% overall utilisation.
if [[ -z "${ZOMBIE_PIDS}" ]]; then
    # Get per-GPU utilisation
    declare -A GPU_UTIL
    while IFS=, read -r util_raw; do
        idx=${#GPU_UTIL[@]}
        util=$(echo "${util_raw}" | tr -d ' %')
        GPU_UTIL["${idx}"]="${util:-0}"
    done < <(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null || true)

    # For each compute process, flag if device utilisation is 0 and VRAM > 100 MiB
    while IFS=, read -r pid gpu_idx mem_raw; do
        pid=$(echo "${pid}" | tr -d ' ')
        gpu_idx=$(echo "${gpu_idx}" | tr -d ' ')
        mem=$(echo "${mem_raw}" | tr -d ' MiB')
        [[ -z "${pid}" || "${pid}" == "pid" ]] && continue
        util="${GPU_UTIL[${gpu_idx}]:-0}"
        if [[ "${util}" == "0" ]] && [[ "${mem:-0}" -gt 100 ]] 2>/dev/null; then
            ZOMBIE_PIDS="${ZOMBIE_PIDS} ${pid}"
        fi
    done < <(nvidia-smi --query-compute-apps=pid,gpu_index,used_memory \
             --format=csv,noheader 2>/dev/null || true)

    ZOMBIE_PIDS=$(echo "${ZOMBIE_PIDS}" | tr ' ' '\n' | grep -v '^$' | sort -u | tr '\n' ' ' | sed 's/ $//')
fi

ZOMBIE_COUNT=$(echo "${ZOMBIE_PIDS}" | tr ' ' '\n' | grep -v '^$' | wc -l | tr -d ' ')
if [[ -z "${ZOMBIE_PIDS}" ]]; then
    ZOMBIE_COUNT=0
fi

# ---------------------------------------------------------------------------
# Kill zombie processes (only when --kill-zombies and NOT --dry-run)
# ---------------------------------------------------------------------------

KILLED_COUNT=0

if [[ "${KILL_ZOMBIES}" -eq 1 && "${DRY_RUN}" -eq 0 && -n "${ZOMBIE_PIDS}" ]]; then
    for pid in ${ZOMBIE_PIDS}; do
        [[ -z "${pid}" ]] && continue
        echo "Killing zombie GPU process PID ${pid} (SIGKILL)" >&2
        # Note: killing another user's process may require sudo.
        if kill -9 "${pid}" 2>/dev/null; then
            KILLED_COUNT=$((KILLED_COUNT + 1))
        else
            echo "WARNING: could not kill PID ${pid} — may need sudo" >&2
        fi
    done
elif [[ "${DRY_RUN}" -eq 1 && -n "${ZOMBIE_PIDS}" ]]; then
    echo "DRY RUN: would kill zombie PIDs: ${ZOMBIE_PIDS}" >&2
fi

# ---------------------------------------------------------------------------
# Compute all_healthy and print summary
# ---------------------------------------------------------------------------

ALL_HEALTHY="False"
if [[ "${GPU_COUNT}" -ge 2 && "${ZOMBIE_COUNT}" -eq 0 ]]; then
    ALL_HEALTHY="True"
fi

echo "SESSION STARTUP: n_gpus=${GPU_COUNT} zombies=${ZOMBIE_COUNT} killed=${KILLED_COUNT} all_healthy=${ALL_HEALTHY}"

exit 0
