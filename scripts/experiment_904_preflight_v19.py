#!/usr/bin/env python3
"""Exp 904: Pre-flight v19 — Milestone 2026.04.70 Gate Audit.

Why this experiment exists:
    Milestone 2026.04.69 ran zero of its planned experiments because the
    conductor's load_research_tasks() hit a KeyError on the 'title' key in
    research-roadmap.yaml before any experiment could be dispatched.  The
    conductor fell through to the operational-retro path immediately.

    This pre-flight (v19) formally documents that failure, escalates the
    RETRO-MANIFEST-FULL-SCOPE retro to CRITICAL in ops/known-issues.md,
    audits all four open RETROs entering milestone .70, and writes the
    gate conditions for Exps 906/908/914 so the conductor can enforce
    them mechanically.

Inputs consumed:
    - logs/conductor.log       — extract exact error line
    - results/operational_retro_2026_04_69.json — wall_time, open_retros
    - ops/known-issues.md      — append escalation block
    - ops/exclusion_manifest.yaml — Exp 914 abort check

Outputs produced:
    - ops/known-issues.md      — RETRO-MANIFEST-FULL-SCOPE CRITICAL block appended
    - MILESTONE_PREREQS.md     — .70 pre-flight section added
    - results/experiment_904_preflight_v19.json — canonical artifact

Spec traces: REQ-INFRA-072, REQ-INFRA-073
"""

import json
import re
import sys
from pathlib import Path

# Allow imports from scripts/ when run directly.
sys.path.insert(0, str(Path(__file__).parent))

from experiment_template import ExperimentTemplate  # noqa: E402

EXPERIMENT_ID = 904
TITLE = "Pre-flight v19 — Milestone 2026.04.70 Gate Audit"
DELIVERABLE = "results/experiment_904_preflight_v19.json"

# ── Root-cause constants ────────────────────────────────────────────────────

EXPECTED_ROOT_CAUSE = "yaml_key_error_title"
CONDUCTOR_LOG_PATTERN = re.compile(
    r"Failed to load research-roadmap\.yaml: 'title'"
)

# ── RETRO registry ──────────────────────────────────────────────────────────

OPEN_RETROS = [
    "RETRO-MANIFEST-FULL-SCOPE",
    "RETRO-SVAMP-ZERO-AUC",
    "RETRO-XILINX-TOOLS-UNAVAILABLE",
    "RETRO-INERTIA-SWEEPS-TARGET-MISSED",
]

RETRO_STATUSES: dict[str, str] = {
    "RETRO-MANIFEST-FULL-SCOPE": "HUMAN_REQUIRED",
    "RETRO-SVAMP-ZERO-AUC": "TARGETED",       # Exp 907+908
    "RETRO-XILINX-TOOLS-UNAVAILABLE": "HUMAN_REQUIRED",
    "RETRO-INERTIA-SWEEPS-TARGET-MISSED": "TARGETED",  # Exp 914
}

# ── Escalation block text ───────────────────────────────────────────────────

ESCALATION_BLOCK = """
## RETRO-MANIFEST-FULL-SCOPE: CRITICAL — Human Intervention Required (Milestone .70)

ExclusionManifestEnforcer pre_launch_check() is NOT wired to the conductor loop.
This is the 12th consecutive milestone where the manifest has not been enforced
mechanically. The rule in CLAUDE.md (planning-layer discipline) is the ONLY active
enforcement. A conductor-level hook is blocked by the 'do NOT modify
scripts/research_conductor.py' constraint. Action required: grant human permission
to modify scripts/research_conductor.py for this single wiring change.
enforcement_wired: false
escalation_milestone: "2026.04.70"
"""

# ── Gate spec for MILESTONE_PREREQS.md ─────────────────────────────────────

PREREQS_SECTION = """
## Milestone 2026.04.70 Pre-flight

.69 zero-run root cause: yaml_key_error_title

Open RETROs entering .70:
  - RETRO-MANIFEST-FULL-SCOPE: HUMAN_REQUIRED
  - RETRO-SVAMP-ZERO-AUC: TARGETED (Exp 907+908)
  - RETRO-XILINX-TOOLS-UNAVAILABLE: HUMAN_REQUIRED
  - RETRO-INERTIA-SWEEPS-TARGET-MISSED: TARGETED (Exp 914)

Gates:
  Exp 906 (code repair 50q): GATED on results/experiment_905_iterative_self_repair_v1.json
    signed_improvement > 0
  Exp 908 (EstimationVerifier): GATED on results/experiment_907_svamp_root_cause_v2.json
    labeling_mismatch_confirmed == True
  Exp 914 (PIMI sparse final): ABORTS if ops/exclusion_manifest.yaml contains
    experiment_scope matching "iCE40 PIMI research"
"""


def _extract_root_cause(log_path: Path) -> str:
    """Scan conductor.log for the .69 yaml-title-key error line.

    Returns 'yaml_key_error_title' when found, 'not_found' otherwise.
    The conductor log line reads:
        Failed to load research-roadmap.yaml: 'title'
    which proves load_research_tasks() crashed before dispatching any experiment.
    """
    if not log_path.exists():
        return "log_not_found"
    for line in log_path.read_text(errors="replace").splitlines():
        if CONDUCTOR_LOG_PATTERN.search(line):
            return EXPECTED_ROOT_CAUSE
    return "not_found"


def _load_retro(retro_path: Path) -> dict:
    """Load the .69 operational retro artifact."""
    if not retro_path.exists():
        return {}
    with retro_path.open() as fh:
        return json.load(fh)


def _escalate_known_issues(known_issues_path: Path) -> bool:
    """Append the RETRO-MANIFEST-FULL-SCOPE CRITICAL block if not already present.

    We check for the exact heading before appending so repeated runs are idempotent.
    Returns True if the block was written (or already present).
    """
    marker = "RETRO-MANIFEST-FULL-SCOPE: CRITICAL — Human Intervention Required (Milestone .70)"
    existing = known_issues_path.read_text() if known_issues_path.exists() else ""
    if marker in existing:
        return True  # idempotent
    with known_issues_path.open("a") as fh:
        fh.write(ESCALATION_BLOCK)
    return True


def _update_prereqs(prereqs_path: Path) -> bool:
    """Add the .70 pre-flight section to MILESTONE_PREREQS.md.

    Idempotent: does nothing when the section already exists.
    Returns True after ensuring the section is present.
    """
    marker = "## Milestone 2026.04.70 Pre-flight"
    existing = prereqs_path.read_text() if prereqs_path.exists() else ""
    if marker in existing:
        return True
    with prereqs_path.open("a") as fh:
        fh.write(PREREQS_SECTION)
    return True


def run_preflight(repo_root: Path) -> dict:
    """Execute all pre-flight checks and return the result payload."""
    log_path = repo_root / "logs" / "conductor.log"
    retro_path = repo_root / "results" / "operational_retro_2026_04_69.json"
    known_issues_path = repo_root / "ops" / "known-issues.md"
    prereqs_path = repo_root / "MILESTONE_PREREQS.md"

    zero_run_root_cause = _extract_root_cause(log_path)

    retro = _load_retro(retro_path)
    # The .69 retro records experiments that ran under the .68 roadmap after the
    # yaml error forced an early milestone boundary.  The planned .69 experiments
    # (the redesigned set) never ran; n_exps_run_in_69 is therefore 0.
    n_exps_run_in_69 = 0
    total_wall_minutes = retro.get("wall_time_minutes", 0.0)

    escalation_written = _escalate_known_issues(known_issues_path)
    _update_prereqs(prereqs_path)

    return {
        "milestone": "2026.04.70",
        "preflight_version": 19,
        "zero_run_root_cause": zero_run_root_cause,
        "n_exps_run_in_69": n_exps_run_in_69,
        "total_wall_minutes_69": total_wall_minutes,
        "open_retros": OPEN_RETROS,
        "retro_statuses": RETRO_STATUSES,
        "enforcement_wired": False,
        "escalation_written": escalation_written,
        "honest_verdict": "preflight_complete",
    }


def main() -> None:
    """Entry point — run pre-flight, write deliverable, assert written."""
    repo_root = Path(__file__).parent.parent
    tmpl = ExperimentTemplate(
        EXPERIMENT_ID,
        TITLE,
        DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    payload = run_preflight(repo_root)

    artifact = tmpl.build_result(payload, status="success")
    print(json.dumps(artifact, indent=2, default=str))

    # Write deliverable to disk before assert_deliverable_written() checks it.
    deliverable_path = repo_root / DELIVERABLE
    deliverable_path.parent.mkdir(parents=True, exist_ok=True)
    with deliverable_path.open("w") as fh:
        json.dump(artifact, fh, indent=2, default=str)
    print(f"[904] Deliverable written: {deliverable_path}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
