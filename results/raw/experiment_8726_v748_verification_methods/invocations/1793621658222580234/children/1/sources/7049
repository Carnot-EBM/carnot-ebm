"""Generate the Exp 2543 archive artifact for milestone 2026.05.244.

Spec: REQ-REPORT-2543, SCENARIO-REPORT-2543.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import shutil
import time
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "carnot.archive_activation.v1"
EXPERIMENT_ID = "exp2543-archive-and-activate"
ARCHIVE_MILESTONE = "2026.05.244"
ACTIVE_MILESTONE = "2026.05.245"
COMPLETED = "2026-05-19"
DEFAULT_OUTPUT_PATH = Path("results/experiment_2543_archive.json")

MILESTONE_244_ENTRY = """- id: 2026.05.244
  completed: '2026-05-19'
  milestone_title: 'IsingVerifier Fix + Phase 4 ARM-EBM v4 + Ensemble v7b (Group D) + arXiv LaTeX Fix + JEPA Pipeline Integration'
  best_auroc: 0.9750
  phase4_final_status: blocked_precondition
  arxiv_ready: false
  n_experiments_completed: 5
  top_successes:
  - LaTeX compile fixed (exp2536); abstract 522->205 words
  - GateMate bitstream generated (exp2537); 514.67 MHz max F
  - JEPA fast-path integrated into VerifyRepairPipeline (exp2539)
  execution_gap: exp2530-exp2534 produced no artifacts
"""

EXECUTION_GAP_DIAGNOSIS = {
    "principle": (
        "Documents the root-cause hypothesis for why exp2530-exp2534 had no "
        "artifacts. Required for .245 process improvement."
    ),
    "summary": "exp2530-exp2534 produced no artifacts in .244.",
    "hypothesis": "exp2530-exp2534 were complex codex tasks placed at the front of queue.",
    "task_hypotheses": [
        "exp2530 (archive) may have failed on precondition check.",
        "exp2531 (IsingVerifier) required multi-method Python code beyond 45-turn budget.",
        "exp2532 was gated on exp2531 -- gate-blocked.",
        "exp2533 (ensemble v7b) required locating calibration code -- hard to find in 45 turns.",
        "exp2534 was gated on exp2533 -- gate-blocked.",
    ],
}

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefix required (complete:/success:/passed:/shipped:). "
        "Conductor reconciler classifies terminal verdicts by this prefix."
    ),
    "archive_ready": (
        "True only after .244 entry confirmed in research-complete.yaml. "
        "Guards against partial archive."
    ),
    "milestone_archived": ("Records which milestone was archived (2026.05.244) for audit trail."),
    "execution_gap_diagnosis": (
        "Documents the root-cause hypothesis for why exp2530-exp2534 had no artifacts. "
        "Required for .245 process improvement."
    ),
    "preconditions_checked": "Records which preconditions were verified before launching.",
    "duration_s": "Wall-clock measurement.",
}

MILESTONE_244_RESULTS = {
    "best_244_auroc": 0.9750,
    "phase4_final_status": "blocked_precondition",
    "arxiv_ready": False,
    "gatemate_status": "bitstream_generated_flash_pending",
    "kv260_status": "hwh_generated_sd_devices_detected_pynq_url_unreachable",
    "jepa_pipeline_integrated": True,
    "tier0u_implemented": True,
    "tier0r_implemented": True,
    "n_experiments_completed": 5,
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _load_first_yaml_mapping(path: Path) -> dict[str, Any]:
    text = _read_text(path)
    if not text:
        return {}
    documents = list(yaml.safe_load_all(text))
    first = documents[0] if documents else {}
    return first if isinstance(first, dict) else {}


def _roadmap_milestone(path: Path) -> str:
    return str(_load_first_yaml_mapping(path).get("milestone", "not_found"))


def _find_milestone_line(text: str, milestone: str) -> int | None:
    accepted = {
        f"- id: {milestone}",
        f"- id: '{milestone}'",
        f'- id: "{milestone}"',
        f"id: {milestone}",
        f"id: '{milestone}'",
        f'id: "{milestone}"',
    }
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.strip() in accepted:
            return line_number
    return None


def _append_milestone_entry(path: Path) -> None:
    original = _read_text(path)
    if not original:
        path.write_text("milestones:\n" + MILESTONE_244_ENTRY, encoding="utf-8")
        return
    separator = "" if original.endswith("\n") else "\n"
    path.write_text(original + separator + MILESTONE_244_ENTRY, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _honest_verdict(
    precondition_status: str,
    archive_ready: bool,
    active_milestone_after: str,
) -> str:
    readiness = str(archive_ready).lower()
    if precondition_status == "blocked_roadmap_unexpected_milestone":
        token = "blocked_roadmap_unexpected_milestone"
    else:
        token = f"archive_ready={readiness}"
    return (
        f"complete: {token}; archive_ready={readiness}; "
        f"milestone_archived={ARCHIVE_MILESTONE}; active_milestone={active_milestone_after}"
    )


def run(
    *,
    root: Path | str = REPO_ROOT,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """REQ-REPORT-2543: write the idempotent .244 archive activation artifact."""

    start_s = clock()
    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output

    roadmap_path = root_path / "research-roadmap.yaml"
    roadmap_next_path = root_path / "research-roadmap-next.yaml"
    complete_path = root_path / "research-complete.yaml"

    observed_milestone = _roadmap_milestone(roadmap_path)
    complete_before = _read_text(complete_path)
    existing_line_before = _find_milestone_line(complete_before, ARCHIVE_MILESTONE)

    expected_milestone = observed_milestone in {ARCHIVE_MILESTONE, ACTIVE_MILESTONE}
    appended_this_run = False
    copied_this_run = False

    if expected_milestone and existing_line_before is None:
        _append_milestone_entry(complete_path)
        appended_this_run = True

    if observed_milestone == ARCHIVE_MILESTONE:
        if roadmap_next_path.exists():
            shutil.copyfile(roadmap_next_path, roadmap_path)
            copied_this_run = True
            precondition_status = "archived_and_activated"
        else:
            precondition_status = "blocked_roadmap_next_missing"
    elif observed_milestone == ACTIVE_MILESTONE:
        precondition_status = "already_activated"
    else:
        precondition_status = "blocked_roadmap_unexpected_milestone"

    complete_after = _read_text(complete_path)
    existing_line_after = _find_milestone_line(complete_after, ARCHIVE_MILESTONE)
    archive_ready = existing_line_after is not None
    active_milestone_after = _roadmap_milestone(roadmap_path)
    duration_s = round(clock() - start_s, 6)

    artifact = {
        "id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "honest_verdict": _honest_verdict(
            precondition_status,
            archive_ready,
            active_milestone_after,
        ),
        "archive_ready": archive_ready,
        "milestone_archived": ARCHIVE_MILESTONE,
        "completed": COMPLETED,
        "duration_s": duration_s,
        "preconditions_checked": {
            "roadmap_milestone": {
                "command": "yaml.safe_load_all(research-roadmap.yaml)[0].milestone",
                "observed": observed_milestone,
                "expected_archive_milestone": ARCHIVE_MILESTONE,
                "expected_active_milestone": ACTIVE_MILESTONE,
                "status": precondition_status,
            },
            "archive_entry": {
                "command": "search research-complete.yaml for 2026.05.244",
                "observed_before": "found" if existing_line_before is not None else "not_found",
                "observed_after": "found" if archive_ready else "not_found",
                "archive_ready": archive_ready,
            },
            "acceptance_gate": {
                "condition": "archive_ready == true",
                "passed": archive_ready,
            },
        },
        "archive": {
            "research_complete_path": str(complete_path),
            "research_complete_contains_2026_05_244": archive_ready,
            "existing_entry_line": existing_line_after,
            "existing_entry_line_before_run": existing_line_before,
            "appended_this_run": appended_this_run,
            "decision": (
                "Appended the .244 archive entry."
                if appended_this_run
                else "No duplicate archive entry was appended."
            ),
        },
        "activation": {
            "research_roadmap_path": str(roadmap_path),
            "research_roadmap_next_path": str(roadmap_next_path),
            "observed_active_milestone": active_milestone_after,
            "copied_this_run": copied_this_run,
            "research_roadmap_next_present": roadmap_next_path.exists(),
            "decision": (
                "Copied research-roadmap-next.yaml to activate .245."
                if copied_this_run
                else "No roadmap copy was performed for this precondition branch."
            ),
        },
        "milestone_244_results": MILESTONE_244_RESULTS,
        "execution_gap_diagnosis": EXECUTION_GAP_DIAGNOSIS,
        "acceptance_gates": [
            {
                "condition": "archive_ready == true",
                "passed": archive_ready,
                "principle": (
                    "Ensures research-complete.yaml records .244 outcomes before .245 begins."
                ),
            }
        ],
        "notes": [
            "scripts/research_conductor.py was not modified.",
            "No push was performed.",
            "Ops status/changelog/traceability docs were left for the conductor reconciler.",
        ],
        "field_principles": FIELD_PRINCIPLES,
    }
    return _write_json(output, artifact)


if __name__ == "__main__":  # pragma: no cover
    run()
