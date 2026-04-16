"""Pre-session startup health check for Carnot research sessions.

**Why this module exists:**
    The 2026.04.24 retrospective (RETRO-007/RETRO-008) found two recurring issues
    at session start:

    1. **Zombie GPU processes** (RETRO-007): orphaned processes holding VRAM from
       a prior session were consuming memory before the new experiment even started,
       causing GPU-OOM errors that were misdiagnosed as model bugs.

    2. **No pre-flight check** (RETRO-008): the research conductor launched
       experiments without first verifying that both RTX 3090s were visible and
       healthy.  A five-second pre-flight would have saved multiple wasted runs.

    This module provides:
    - ``parse_session_startup_output(output)`` — parses the stdout of
      ``scripts/session_startup.sh`` into a structured dict.
    - ``run_session_startup(dry_run)`` — invokes the shell script and returns
      the parsed dict.  When ``dry_run=True`` no processes are killed.

Spec: REQ-INFRA-008,
      SCENARIO-INFRA-012, SCENARIO-INFRA-013
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

_log = logging.getLogger(__name__)

# Absolute path to the shell script, resolved relative to this file's location.
# Layout: python/carnot/pipeline/session_startup.py  →  scripts/session_startup.sh
_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "session_startup.sh"

# Regex for the summary line emitted by session_startup.sh:
#   SESSION STARTUP: n_gpus=X zombies=Y killed=Z all_healthy=T/F
_SUMMARY_RE = re.compile(
    r"SESSION STARTUP:\s+"
    r"n_gpus=(?P<n_gpus>\d+)\s+"
    r"zombies=(?P<zombies>\d+)\s+"
    r"killed=(?P<killed>\d+)\s+"
    r"all_healthy=(?P<all_healthy>True|False)",
    re.IGNORECASE,
)


def parse_session_startup_output(output: str) -> dict:
    """Parse the stdout produced by ``scripts/session_startup.sh``.

    Searches for the canonical summary line::

        SESSION STARTUP: n_gpus=X zombies=Y killed=Z all_healthy=True/False

    If the line is absent (e.g. nvidia-smi not found and script printed nothing
    matching), returns safe zero-values with ``all_healthy=False``.

    Parameters
    ----------
    output : str
        Full stdout text of the session_startup.sh invocation.

    Returns
    -------
    dict
        Keys:
        - ``n_gpus_detected`` (int): number of CUDA GPUs found.
        - ``n_zombies_found`` (int): zombie processes counted.
        - ``n_zombies_killed`` (int): zombie processes actually killed (0 in dry-run).
        - ``all_healthy`` (bool): True iff n_gpus_detected >= 2 and n_zombies_found == 0.
    """
    match = _SUMMARY_RE.search(output)
    if match is None:
        # Script ran but produced no parseable summary — CI / no nvidia-smi case.
        _log.debug("No SESSION STARTUP summary line found in output; defaulting to unhealthy")
        return {
            "n_gpus_detected": 0,
            "n_zombies_found": 0,
            "n_zombies_killed": 0,
            "all_healthy": False,
        }

    n_gpus = int(match.group("n_gpus"))
    zombies = int(match.group("zombies"))
    killed = int(match.group("killed"))
    # Recompute all_healthy from parsed values rather than trusting the string
    # literal, so the Python rule is the single source of truth.
    all_healthy = n_gpus >= 2 and zombies == 0

    return {
        "n_gpus_detected": n_gpus,
        "n_zombies_found": zombies,
        "n_zombies_killed": killed,
        "all_healthy": all_healthy,
    }


def run_session_startup(dry_run: bool = True) -> dict:
    """Run ``scripts/session_startup.sh`` and return the parsed health summary.

    Calls the shell script with ``--dry-run`` when ``dry_run=True`` (the default),
    which means zombie PIDs are printed but NOT killed.  To actually kill zombies,
    pass ``dry_run=False`` — but be aware that this requires the caller to have
    appropriate OS permissions (or sudo access) to send SIGKILL to other processes.

    The function never raises.  If the script is missing, not executable, or
    nvidia-smi is absent, a safe degraded dict is returned:
    ``{n_gpus_detected: 0, n_zombies_found: 0, n_zombies_killed: 0, all_healthy: False}``.

    Parameters
    ----------
    dry_run : bool
        When True (default), pass ``--dry-run`` to the script; never kill processes.
        When False, pass ``--kill-zombies`` instead.

    Returns
    -------
    dict
        Same keys as ``parse_session_startup_output``.
    """
    cmd = [str(_SCRIPT_PATH)]
    if dry_run:
        cmd.append("--dry-run")
    else:
        cmd.append("--kill-zombies")

    _log.info("Running session startup check: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        _log.warning("session_startup.sh not found at %s — returning unhealthy", _SCRIPT_PATH)
        return {"n_gpus_detected": 0, "n_zombies_found": 0, "n_zombies_killed": 0, "all_healthy": False}
    except subprocess.TimeoutExpired:
        _log.warning("session_startup.sh timed out — returning unhealthy")
        return {"n_gpus_detected": 0, "n_zombies_found": 0, "n_zombies_killed": 0, "all_healthy": False}
    except Exception as exc:  # pragma: no cover — unexpected OS-level failure
        _log.warning("session_startup.sh error: %s — returning unhealthy", exc)
        return {"n_gpus_detected": 0, "n_zombies_found": 0, "n_zombies_killed": 0, "all_healthy": False}

    combined = result.stdout + "\n" + result.stderr
    return parse_session_startup_output(combined)
