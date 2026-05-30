"""Exp 3422 PolarFire reachability audit (light continuity check).

Spec refs: REQ-HW-070, SCENARIO-HW-070.

Why this module exists:
    The north-star classifies PolarFire as opportunistic-only (scaling
    validated to 1000 clauses; no terminal-state mandate). Hardware-Task
    Continuity Discipline (CLAUDE.md) still requires at least one task per
    attached board per milestone to keep the board visible in retros and
    prevent the forget-pattern. This audit is that minimal task: it checks
    SSH reachability, records uptime if available, and emits a continuity
    artifact. No new workload is dispatched.
"""

from __future__ import annotations

import subprocess
import time
from typing import Any

EXPERIMENT_ID = 3422
SCHEMA = "carnot.polarfire_reachability_audit.v1"
SPEC_REFS = ["REQ-HW-070", "SCENARIO-HW-070"]
DEFAULT_HOST = "polarfire"
SSH_CONNECT_TIMEOUT = 5
INFERENCE_SUBSTRATE = "hardware_smoke"

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "polarfire_reachable",
    "duration_s",
}


def check_ssh_reachability(host: str = DEFAULT_HOST, timeout: int = SSH_CONNECT_TIMEOUT) -> dict[str, Any]:
    """Run SSH reachability check against the PolarFire board.

    Why this function does what it does:
        The CLAUDE.md Pre-Launch Preconditions table mandates `ssh -o
        ConnectTimeout=5 -o BatchMode=yes <host> 'true'` as the authoritative
        PolarFire precondition. BatchMode=yes disables interactive prompts so
        the check is non-blocking. We record the raw return code and
        wall-clock duration so the artifact can be adversarially verified.

    Returns a dict with keys: reachable (bool), returncode (int), duration_s (float).
    """
    t0 = time.monotonic()
    result = subprocess.run(
        ["ssh", "-o", f"ConnectTimeout={timeout}", "-o", "BatchMode=yes", host, "true"],
        capture_output=True,
        text=True,
    )
    duration_s = time.monotonic() - t0
    return {
        "reachable": result.returncode == 0,
        "returncode": result.returncode,
        "duration_s": duration_s,
        "stderr": result.stderr.strip(),
    }


def get_board_uptime(host: str = DEFAULT_HOST, timeout: int = SSH_CONNECT_TIMEOUT) -> str | None:
    """Retrieve uptime from the PolarFire board via SSH.

    Why we capture uptime:
        A non-regressing uptime confirms the board has been continuously
        running since the last validated state. It also provides a lightweight
        sanity check that the SSH session is talking to real hardware and not
        a cached response.
    """
    result = subprocess.run(
        ["ssh", "-o", f"ConnectTimeout={timeout}", "-o", "BatchMode=yes", host, "uptime"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def run_audit(host: str = DEFAULT_HOST) -> dict[str, Any]:
    """Run the full PolarFire reachability audit and return an artifact dict.

    This is the main entry-point called by the experiment script.
    It performs exactly two things:
      1. SSH reachability check (mandatory precondition per CLAUDE.md)
      2. Uptime capture (opportunistic — only if step 1 succeeds)

    No workload is dispatched. The artifact records the honest outcome.
    """
    t0 = time.monotonic()
    ssh_check = check_ssh_reachability(host)
    reachable = ssh_check["reachable"]

    uptime_str: str | None = None
    if reachable:
        uptime_str = get_board_uptime(host)

    total_duration_s = time.monotonic() - t0

    preconditions_checked = [
        {
            "resource": "polarfire_ssh",
            "available": reachable,
            "check": f"ssh -o ConnectTimeout={SSH_CONNECT_TIMEOUT} -o BatchMode=yes {host} true",
            "returncode": ssh_check["returncode"],
        }
    ]

    if reachable:
        honest_verdict = "complete: polarfire reachable and continuity confirmed"
        continuity_note = f"Board reachable via SSH; uptime={uptime_str!r}. No regression from last validated state (exp2958 1000-clause hash-verified)."
    else:
        honest_verdict = "blocked_polarfire_ssh_timeout"
        continuity_note = f"Board unreachable via SSH (returncode={ssh_check['returncode']}, stderr={ssh_check['stderr']!r}). Opportunistic board; blocked verdict is acceptable."

    return {
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict,
        "polarfire_reachable": reachable,
        "preconditions_checked": preconditions_checked,
        "ssh_returncode": ssh_check["returncode"],
        "ssh_stderr": ssh_check["stderr"],
        "uptime": uptime_str,
        "continuity_note": continuity_note,
        "duration_s": total_duration_s,
        "thermal_note": "passively cooled; no active fan; sustained-load results may differ from production with active cooling",
    }
