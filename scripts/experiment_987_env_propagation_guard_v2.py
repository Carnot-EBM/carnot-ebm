#!/usr/bin/env python3
"""Experiment 987: Session-Boundary-Persistent EnvPropagationGuard (v2).

Root cause addressed (RETRO-015 recurrence):
    CARNOT_* env vars (CARNOT_FORCE_LIVE, CARNOT_N_SPINS, etc.) are NOT propagated
    to subprocesses when the conductor launches experiments.  The Exp 855 fix patched
    the in-session state but did not survive conductor session boundaries — each new
    session starts without the env vars.

Prior failure:
    Exp 975 (same experiment) never produced a result artifact because the script
    lacked a try/finally guard.  When an error occurred during implementation, the
    artifact write was skipped, blocking 6 downstream experiments.

What this experiment does:
    1. Creates ~/.carnot/ directory.
    2. Adds EnvPropagationGuard.propagate() and write_state_file() methods to
       scripts/experiment_template.py.
    3. Verifies that propagate() sets CARNOT_FORCE_LIVE=1 in the current process.
    4. Verifies that a subprocess inherits the var via the state file mechanism.
    5. Writes the result artifact UNCONDITIONALLY in a finally block.

Spec: REQ-INFRA-080, REQ-INFRA-081, SCENARIO-INFRA-090, SCENARIO-INFRA-091
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap — must happen before any other import so CARNOT_FORCE_LIVE=1 is
# visible to the ExperimentTemplate constructor.
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root))

from scripts.experiment_template import EnvPropagationGuard  # noqa: E402

# Propagate env vars FIRST — this is the fix being verified.
_propagated = EnvPropagationGuard.propagate()
EnvPropagationGuard.write_state_file()

# NOW import ExperimentTemplate (its __init__ calls load_session_env, which is now seeded)
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
_log = logging.getLogger(__name__)

EXP_ID = 987
TITLE = "Session-Boundary-Persistent EnvPropagationGuard"
DELIVERABLE = "results/experiment_987_env_propagation_guard_v2.json"


def _utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_date() -> str:
    import datetime

    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%d")


def main() -> None:
    started_at = _utc_now()
    t0 = time.perf_counter()

    output_path = _repo_root / DELIVERABLE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result: dict = {}
    status = "unknown"
    honest_verdict = "unknown"

    try:
        # ------------------------------------------------------------------
        # Phase 1: verify in-process propagation
        # ------------------------------------------------------------------
        _log.info("[Phase 1] Verifying in-process CARNOT_FORCE_LIVE propagation")
        force_live = os.environ.get("CARNOT_FORCE_LIVE")
        in_process_ok = force_live == "1"
        _log.info(
            "CARNOT_FORCE_LIVE in-process: %s (expected '1', ok=%s)", force_live, in_process_ok
        )

        # ------------------------------------------------------------------
        # Phase 2: verify state file was written
        # ------------------------------------------------------------------
        _log.info("[Phase 2] Verifying ~/.carnot/conductor_state.sh")
        state_file = EnvPropagationGuard.STATE_FILE
        state_file_exists = state_file.exists()
        state_file_content = state_file.read_text() if state_file_exists else ""
        state_file_has_force_live = "export CARNOT_FORCE_LIVE=1" in state_file_content
        state_file_has_shebang = state_file_content.startswith("#!/bin/sh")
        _log.info(
            "STATE_FILE exists=%s has_force_live=%s has_shebang=%s path=%s",
            state_file_exists,
            state_file_has_force_live,
            state_file_has_shebang,
            state_file,
        )

        # ------------------------------------------------------------------
        # Phase 3: verify subprocess propagation via env parameter
        # ------------------------------------------------------------------
        _log.info("[Phase 3] Verifying subprocess inherits CARNOT_FORCE_LIVE via env=os.environ")
        proc_result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os; print(os.environ.get('CARNOT_FORCE_LIVE', 'NOT_SET'))",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ},  # explicit env copy — same as conductor pattern
        )
        subprocess_output = proc_result.stdout.strip()
        subprocess_propagation_ok = subprocess_output == "1"
        _log.info(
            "Subprocess output: '%s' (expected '1', ok=%s)",
            subprocess_output,
            subprocess_propagation_ok,
        )

        # ------------------------------------------------------------------
        # Phase 4: verify a fresh Python process can source the state file
        # ------------------------------------------------------------------
        _log.info("[Phase 4] Verifying a bare subprocess can source the state file")
        bare_env = {k: v for k, v in os.environ.items() if not k.startswith("CARNOT_")}
        bare_proc = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys, os; "
                    f"sys.path.insert(0, {str(_repo_root)!r}); "
                    "from scripts.experiment_template import EnvPropagationGuard; "
                    "EnvPropagationGuard.propagate(); "
                    "print(os.environ.get('CARNOT_FORCE_LIVE', 'NOT_SET'))"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env=bare_env,
        )
        bare_output = bare_proc.stdout.strip()
        bare_propagation_ok = bare_output == "1"
        bare_stderr = bare_proc.stderr.strip()
        _log.info(
            "Bare subprocess output: '%s' (expected '1', ok=%s) stderr=%s",
            bare_output,
            bare_propagation_ok,
            bare_stderr[:200] if bare_stderr else "",
        )

        # ------------------------------------------------------------------
        # Determine overall result
        # ------------------------------------------------------------------
        all_ok = (
            in_process_ok
            and state_file_exists
            and state_file_has_force_live
            and state_file_has_shebang
            and subprocess_propagation_ok
            and bare_propagation_ok
        )

        if all_ok:
            honest_verdict = "env_propagation_guard_verified"
            status = "success"
        elif in_process_ok and state_file_has_force_live and subprocess_propagation_ok:
            honest_verdict = "env_propagation_partial_bare_subprocess_failed"
            status = "partial"
        else:
            honest_verdict = "env_propagation_guard_failed"
            status = "failed"

        result = {
            "in_process_ok": in_process_ok,
            "force_live_value": force_live,
            "state_file_exists": state_file_exists,
            "state_file_path": str(state_file),
            "state_file_has_force_live": state_file_has_force_live,
            "state_file_has_shebang": state_file_has_shebang,
            "subprocess_propagation_ok": subprocess_propagation_ok,
            "subprocess_output": subprocess_output,
            "bare_subprocess_propagation_ok": bare_propagation_ok,
            "bare_subprocess_output": bare_output,
            "propagated_vars": sorted(_propagated.keys()),
            "retro_015_resolved": all_ok,
        }

    except Exception as exc:
        _log.exception("Unhandled exception in experiment 987")
        honest_verdict = "implementation_error"
        status = "error"
        result = {"error": str(exc), "error_type": type(exc).__name__}

    finally:
        # UNCONDITIONAL artifact write — the lesson from Exp 975's missing artifact.
        duration_s = round(time.perf_counter() - t0, 3)
        artifact = {
            "experiment": EXP_ID,
            "schema": "carnot.experiment.v1",
            "title": TITLE,
            "run_date": _run_date(),
            "started_at": started_at,
            "finished_at": _utc_now(),
            "duration_s": duration_s,
            "status": status,
            "honest_verdict": honest_verdict,
            **result,
        }
        output_path.write_text(json.dumps(artifact, indent=2))
        _log.info(
            "Artifact written to %s (status=%s, verdict=%s, duration=%.1fs)",
            output_path,
            status,
            honest_verdict,
            duration_s,
        )


if __name__ == "__main__":
    main()
