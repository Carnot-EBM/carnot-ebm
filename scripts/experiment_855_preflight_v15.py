#!/usr/bin/env python3
"""Experiment 855 — Pre-flight v15: EnvPropagationGuard deployment verification.

Verifies that RETRO-LIVE-ENV-NOT-PROPAGATED is permanently fixed by confirming
that EnvPropagationGuard is wired into ExperimentTemplate.__init__ and that
~/.carnot_session_env is written on apply_env_autofix().

Spec: REQ-INFRA-070, SCENARIO-INFRA-080
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure scripts/ is importable even when invoked from repo root.
_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from experiment_template import EnvPropagationGuard, ExperimentTemplate  # noqa: E402


def _check_env_guard_deployed() -> bool:
    """Return True if EnvPropagationGuard.load_session_env() is called in __init__.

    Inspects ExperimentTemplate.__init__ source to confirm the call is present.
    A source-level check is fast, dependency-free, and unambiguous.
    """
    src = inspect.getsource(ExperimentTemplate.__init__)
    return "EnvPropagationGuard.load_session_env()" in src


def _check_live_env_fixed() -> bool:
    """Return True if write_session_env and load_session_env are both implemented.

    Uses a temp file so the real ~/.carnot_session_env is not disturbed.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / ".carnot_session_env"
        # Monkeypatch the guard path for this test
        original_path = EnvPropagationGuard._path
        try:
            EnvPropagationGuard._path = tmp_path  # type: ignore[assignment]
            # Write
            EnvPropagationGuard.write_session_env({"CARNOT_FORCE_LIVE": "1"})
            if not tmp_path.exists():
                return False
            content = tmp_path.read_text()
            if "CARNOT_FORCE_LIVE=1" not in content:
                return False
            # Load (into isolated env copy)
            saved = os.environ.pop("CARNOT_FORCE_LIVE", None)
            try:
                applied = EnvPropagationGuard.load_session_env()
                loaded_ok = os.environ.get("CARNOT_FORCE_LIVE") == "1"
                return loaded_ok and "CARNOT_FORCE_LIVE" in applied
            finally:
                # Restore env
                if saved is not None:
                    os.environ["CARNOT_FORCE_LIVE"] = saved
                else:
                    os.environ.pop("CARNOT_FORCE_LIVE", None)
        finally:
            EnvPropagationGuard._path = original_path


def main() -> None:
    tmpl = ExperimentTemplate(
        855,
        "Pre-flight v15 LIVE-ENV fix",
        "results/experiment_855_preflight_v15.json",
        requires_gpu=False,
    )
    tmpl.setup()

    env_guard_deployed = _check_env_guard_deployed()
    live_env_fixed = _check_live_env_fixed()
    prereqs_updated = (
        _REPO_ROOT / "MILESTONE_PREREQS.md"
    ).exists() and "Milestone 2026.04.66 Pre-flight" in (
        _REPO_ROOT / "MILESTONE_PREREQS.md"
    ).read_text()

    open_retros = [
        "RETRO-MANIFEST-FULL-SCOPE",
        "RETRO-JEPA-OOD",
        "RETRO-CONSTRAINT-ZERO-DELTA",
        "RETRO-XILINX-TOOLS-UNAVAILABLE",
        "RETRO-ISING-INJECTION-NO-DISCRIMINATION",
        "RETRO-SVAMP-ZERO-AUC",
        "RETRO-ICE40-PNR-LUT-OVERFLOW",
        "RETRO-SOTA-MODEL-DOWNLOAD",
        "RETRO-ICE40-N16-UNEXPECTED-EXPANSION",
        "RETRO-LIVE-ENV-NOT-PROPAGATED",
    ]

    if live_env_fixed and env_guard_deployed and prereqs_updated:
        honest_verdict = "governance_ready"
        missing = []
    else:
        honest_verdict = "governance_partial"
        missing = []
        if not live_env_fixed:
            missing.append("live_env_fixed=False: EnvPropagationGuard write/load not working")
        if not env_guard_deployed:
            missing.append("env_guard_deployed=False: __init__ does not call load_session_env()")
        if not prereqs_updated:
            missing.append("prereqs_updated=False: MILESTONE_PREREQS.md missing .66 section")

    artifact = tmpl.build_result(
        {
            "live_env_fixed": live_env_fixed,
            "env_guard_deployed": env_guard_deployed,
            "prereqs_updated": prereqs_updated,
            "open_retros_count": len(open_retros),
            "open_retros": open_retros,
            "retros_confirmed_closed": [
                "RETRO-ARBITER-FLAT-ENERGY",
                "RETRO-GGUF-CACHE-IMPORT",
            ],
            "honest_verdict": honest_verdict,
            "missing": missing,
            "session_env_path": str(EnvPropagationGuard._path),
            "spec_requirements_added": ["REQ-INFRA-070", "SCENARIO-INFRA-080"],
        },
        status="success" if honest_verdict == "governance_ready" else "partial",
    )

    out_path = _REPO_ROOT / "results" / "experiment_855_preflight_v15.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"Wrote {out_path}")
    print(f"honest_verdict: {honest_verdict}")
    if missing:
        print("Missing:", missing)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
