"""Exp 5363: archive .488 truth and emit the .489 transition context.

Spec refs: REQ-REPORT-5363, SCENARIO-REPORT-5363,
SCENARIO-REPORT-5363-BLOCKED-INPUT.

This module is deliberately record-only. It reads the completed .488 capstone
and the .489 roadmap context, then writes the first .489 execution artifact.
It records the missing literal next-roadmap file honestly because conductor
activation can consume that file before this task runs.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5363_transition_v489.json")
CAPSTONE_RELATIVE_PATH = Path("results/experiment_5362_capstone_v488.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPERIMENT = "experiment_5363_transition_v489"
EXPERIMENT_ID = "exp5363-v489-transition-and-archive"
MILESTONE = "2026.07.489"
PRIOR_MILESTONE = "2026.07.488"
SCHEMA = "carnot.experiment_5363_transition_v489.v1"
RANDOM_SEED = 5363
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked_")

PRIOR_GATE_FIELDS = (
    "structured_protocol_clean",
    "constraint_tax_panel_ready",
    "tokenprob_feature_rows_ready",
    "carry_token_energy_signal_ready",
    "dependency_provenance_ready",
    "memory_tool_drift_ready",
    "self_learning_scaleup_ready",
    "solver_projection_ready",
    "pbit_schedule_signal_ready",
    "arc_new_level_banked",
    "hardware_speedup_claim",
)

EXPECTED_TASK_IDS = [
    "exp5363-v489-transition-and-archive",
    "exp5364-v489-sota-source-delta",
    "exp5365-v489-grammar-budget-protocol-preflight",
    "exp5366-v489-live-grammar-budgeted-sota-protocol",
    "exp5367-v489-constraint-tax-tool-action-panel-v2",
    "exp5368-v489-budget-curated-memory-governance",
    "exp5369-v489-budgeted-continuous-self-learning-scaleup",
    "exp5370-v489-overwrite-solver-guidance-matrix",
    "exp5371-v489-pbit-boundary-exchange-schedule",
    "exp5372-v489-token-feature-precondition-gate",
    "exp5373-v489-arc-salience-re86-levelup",
    "exp5374-v489-hardware-continuity-receipts",
    "exp5375-v489-capstone",
]

EXPECTED_PHASE_NAMES = [
    "Phase 0 - Transition and Fresh Source Delta",
    "Phase 1 - Grammar-Budgeted Structured SOTA",
    "Phase 2 - Budget-Curated Continuous Self-Learning",
    "Phase 3 - Solver Guidance, Internal-Feature Preconditions, ARC, and Hardware",
    "Phase 4 - Capstone",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Honest terminal state derived from source availability and protected-file checks.",
    "milestone": (
        "Must equal 2026.07.489 so this transition cannot be confused with the "
        "archived .488 capstone."
    ),
    "prior_milestone": (
        "Must equal 2026.07.488 because all carried gates come from the completed "
        "prior milestone."
    ),
    "prior_capstone_path": "Names the exact .488 capstone used as the source of truth.",
    "prior_gate_summary": (
        "Carries only the requested .488 gate booleans so downstream tasks inherit "
        "the actual truth table."
    ),
    "roadmap_next_present": (
        "Bare boolean records whether the literal pre-staged next-roadmap file "
        "existed at execution time."
    ),
    "roadmap_doc_present": (
        "Bare boolean proves the vNEXT roadmap document was available for phase and "
        "expectation extraction."
    ),
    "planned_task_count": "Counts the .489 tasks in the roadmap source used for execution.",
    "planned_task_ids": (
        "Ordered task ids preserve the conductor execution range for downstream gates."
    ),
    "downstream_gate_expectations": (
        "Summarizes the structured, self-learning, solver, token, ARC, and hardware "
        "expectations downstream tasks must honor."
    ),
    "active_roadmap_modified": "Bare boolean must remain false because this task is record-only.",
    "conductor_modified": "Bare boolean must remain false by operator instruction.",
    "honest_verdict": (
        "One-line terminal summary distinguishes clean transition context from "
        "missing-input blockage."
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "milestone",
    "prior_milestone",
    "prior_capstone_path",
    "prior_gate_summary",
    "roadmap_next_present",
    "roadmap_doc_present",
    "planned_task_count",
    "planned_task_ids",
    "downstream_gate_expectations",
    "active_roadmap_modified",
    "conductor_modified",
    "honest_verdict",
)

REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "run_date",
    "spec_refs",
    "result_path",
    "field_principles",
    "duration_s",
    "random_seed",
    "inference_substrate",
    "planned_phase_names",
    "roadmap_doc_task_range",
    "planned_task_source",
    "failed_preconditions",
    "preconditions_checked",
    "cited_upstream_artifacts",
    "reproducibility_checksum",
    *REQUIRED_ARTIFACT_FIELDS,
)

SPEC_REFS = [
    "REQ-REPORT-5363",
    "SCENARIO-REPORT-5363",
    "SCENARIO-REPORT-5363-BLOCKED-INPUT",
]

SOURCE_CONTEXT_PATHS = (
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    Path("ops/status.md"),
    Path("ops/conductor-log.md"),
    CAPSTONE_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
)

DOWNSTREAM_GATE_EXPECTATIONS: JsonDict = {
    "structured": {
        "expectation": (
            "Run a deterministic grammar-budget and completion-slack preflight before "
            "any live GGUF structured-output rerun; keep constraint-tax work gated."
        ),
        "clean_gate": (
            "parse_success_rate>=0.95, schema_success_rate>=0.90, "
            "final_json_extraction_rate>=0.95, unsafe_false_accepts=0, "
            "methodology_duration_s>=60"
        ),
        "inherits_prior_gate": "structured_protocol_clean=false",
    },
    "self_learning": {
        "expectation": (
            "Extend clean dependency and drift work into budget-curated memory with "
            "value-minus-harm per byte, rollback recovery, stale/poison deflection, "
            "and no model-weight mutation."
        ),
        "inherits_prior_gates": [
            "dependency_provenance_ready=true",
            "memory_tool_drift_ready=true",
            "self_learning_scaleup_ready=true",
        ],
    },
    "solver": {
        "expectation": (
            "Treat neural guidance as overwriteable hints under solver authority; "
            "preserve fallback completeness and add p-bit boundary-exchange timing."
        ),
        "inherits_prior_gates": [
            "solver_projection_ready=true",
            "pbit_schedule_signal_ready=true",
        ],
    },
    "token": {
        "expectation": (
            "Use a token/internal-feature precondition gate; do not promote text-only "
            "completion artifacts into internal-energy or hallucination claims."
        ),
        "inherits_prior_gates": [
            "tokenprob_feature_rows_ready=true",
            "carry_token_energy_signal_ready=false",
        ],
    },
    "ARC": {
        "expectation": (
            "Run the live-path salience repair and re86 +1 attempt without claiming "
            "offline ground-truth BFS, hand-built adapters, or a banked level unless "
            "the live agent actually earns it."
        ),
        "inherits_prior_gate": "arc_new_level_banked=false",
    },
    "hardware": {
        "expectation": (
            "Collect continuity receipts only unless authenticated board timing, "
            "baseline timing, workload hash, and repeatability evidence support a "
            "speedup claim."
        ),
        "speedup_claim_allowed": False,
        "inherits_prior_gate": "hardware_speedup_claim=false",
    },
}


def value_of(value: Any) -> Any:
    """Return the machine value from older principle-wrapped artifacts."""

    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def path_sha256(path: Path) -> str | None:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def payload_checksum(payload: JsonMap) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    if not path.exists():
        return {}, {"exists": False, "loadable": False, "error": "missing"}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, {"exists": True, "loadable": False, "error": "malformed_json"}
    if not isinstance(parsed, dict):
        return {}, {"exists": True, "loadable": False, "error": "not_json_object"}
    return parsed, {
        "exists": True,
        "loadable": True,
        "error": None,
        "sha256": path_sha256(path),
    }


def read_yaml_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    if not path.exists():
        return {}, {"exists": False, "loadable": False, "error": "missing"}
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}, {"exists": True, "loadable": False, "error": "malformed_yaml"}
    if not isinstance(parsed, dict):
        return {}, {"exists": True, "loadable": False, "error": "not_yaml_object"}
    return parsed, {
        "exists": True,
        "loadable": True,
        "error": None,
        "sha256": path_sha256(path),
    }


def document_contains_milestone(path: Path, milestone: str) -> bool:
    return path.exists() and milestone in path.read_text(encoding="utf-8", errors="replace")


def extract_phase_names(document_text: str) -> list[str]:
    return [
        line.removeprefix("### ").strip()
        for line in document_text.splitlines()
        if line.startswith("### Phase ")
    ]


def extract_task_range(document_text: str) -> str | None:
    match = re.search(r"^\*\*Task range:\*\*\s*(.+)$", document_text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def extract_roadmap_tasks(roadmap: JsonMap) -> list[str]:
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list):
        return []
    return [str(row["id"]) for row in tasks if isinstance(row, Mapping) and "id" in row]


def empty_prior_gate_summary() -> JsonDict:
    return {field: None for field in PRIOR_GATE_FIELDS}


def extract_prior_gate_summary(capstone: JsonMap) -> JsonDict:
    return {field: value_of(capstone.get(field)) for field in PRIOR_GATE_FIELDS}


def git_path_modified(root: Path, relative_path: Path) -> bool:
    if not (root / ".git").exists():
        return False
    result = subprocess.run(
        ("git", "status", "--short", "--", str(relative_path)),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _modification_status(
    root: Path,
    relative_path: Path,
    overrides: Mapping[Path | str, bool] | None,
) -> bool:
    if overrides is None:
        return git_path_modified(root, relative_path)
    if relative_path in overrides:
        return bool(overrides[relative_path])
    return bool(overrides.get(str(relative_path), git_path_modified(root, relative_path)))


def _yaml_milestone(roadmap: JsonMap) -> str | None:
    milestone = roadmap.get("milestone")
    return str(milestone) if milestone is not None else None


def capstone_failures(capstone: JsonMap) -> list[str]:
    if not capstone:
        return ["capstone_missing_or_unloadable"]
    failures: list[str] = []
    capstone_milestone = value_of(capstone.get("milestone"))
    capstone_status = value_of(capstone.get("status"))
    capstone_verdict = value_of(capstone.get("honest_verdict"))
    if capstone_milestone != PRIOR_MILESTONE:
        failures.append(
            f"capstone_milestone_expected_{PRIOR_MILESTONE}_observed_{capstone_milestone}"
        )
    if capstone_status != "complete":
        failures.append(f"capstone_status_expected_complete_observed_{capstone_status}")
    if not isinstance(capstone_verdict, str) or not capstone_verdict.startswith(
        TERMINAL_PREFIXES
    ):
        failures.append("capstone_honest_verdict_missing_terminal_prefix")
    return failures


def cited_upstream_artifacts(root: Path) -> list[JsonDict]:
    return [
        {
            "relative_path": str(relative_path),
            "exists": (root / relative_path).exists(),
            "sha256": path_sha256(root / relative_path),
            "read_only": True,
        }
        for relative_path in SOURCE_CONTEXT_PATHS
    ]


def precondition_summary(
    *,
    root: Path,
    capstone_meta: JsonMap,
    active_roadmap: JsonMap,
    active_meta: JsonMap,
    next_roadmap: JsonMap,
    next_meta: JsonMap,
    active_roadmap_modified: bool,
    conductor_modified: bool,
) -> JsonDict:
    doc_path = root / VNEXT_RELATIVE_PATH
    doc_text = doc_path.read_text(encoding="utf-8", errors="replace") if doc_path.exists() else ""
    active_milestone = _yaml_milestone(active_roadmap)
    next_milestone = _yaml_milestone(next_roadmap)
    return {
        "capstone_present": capstone_meta.get("exists") is True,
        "capstone_loadable": capstone_meta.get("loadable") is True,
        "roadmap_doc_present": doc_path.exists(),
        "roadmap_doc_names_milestone": MILESTONE in doc_text,
        "active_roadmap_present": active_meta.get("exists") is True,
        "active_roadmap_loadable": active_meta.get("loadable") is True,
        "active_roadmap_milestone": active_milestone,
        "roadmap_next_present": next_meta.get("exists") is True,
        "roadmap_next_loadable": next_meta.get("loadable") is True,
        "roadmap_next_milestone": next_milestone,
        "roadmap_next_names_milestone": next_milestone == MILESTONE,
        "active_roadmap_modified": active_roadmap_modified,
        "conductor_modified": conductor_modified,
        "phase_names": extract_phase_names(doc_text),
        "task_range": extract_task_range(doc_text),
        "checked_paths": [str(path) for path in SOURCE_CONTEXT_PATHS],
    }


def select_task_source(
    *,
    active_roadmap: JsonMap,
    next_roadmap: JsonMap,
    preconditions: JsonMap,
) -> tuple[Path, list[str]]:
    if (
        preconditions.get("roadmap_next_present") is True
        and preconditions.get("roadmap_next_names_milestone") is True
    ):
        return ROADMAP_NEXT_RELATIVE_PATH, extract_roadmap_tasks(next_roadmap)
    return ROADMAP_RELATIVE_PATH, extract_roadmap_tasks(active_roadmap)


def failed_preconditions(
    *,
    capstone_failure_rows: Sequence[str],
    preconditions: JsonMap,
    planned_task_ids: Sequence[str],
) -> list[str]:
    failures = list(capstone_failure_rows)
    if preconditions.get("roadmap_doc_names_milestone") is not True:
        failures.append(f"roadmap_doc_missing_or_mismatch_{MILESTONE}")
    if preconditions.get("active_roadmap_milestone") != MILESTONE:
        failures.append(
            f"active_roadmap_milestone_expected_{MILESTONE}_observed_"
            f"{preconditions.get('active_roadmap_milestone')}"
        )
    if (
        preconditions.get("roadmap_next_present") is True
        and preconditions.get("roadmap_next_names_milestone") is not True
    ):
        failures.append(
            f"roadmap_next_milestone_expected_{MILESTONE}_observed_"
            f"{preconditions.get('roadmap_next_milestone')}"
        )
    if not planned_task_ids:
        failures.append("planned_task_ids_missing")
    elif list(planned_task_ids) != EXPECTED_TASK_IDS:
        failures.append("planned_task_ids_mismatch")
    if preconditions.get("phase_names") != EXPECTED_PHASE_NAMES:
        failures.append("roadmap_phase_names_missing_or_mismatch")
    if preconditions.get("active_roadmap_modified") is True:
        failures.append("active_roadmap_modified")
    if preconditions.get("conductor_modified") is True:
        failures.append("conductor_modified")
    return failures


def build_honest_verdict(
    *,
    status: str,
    roadmap_next_present: bool,
    planned_task_source: str,
    failures: Sequence[str],
) -> str:
    if status == "complete":
        return (
            "complete: .488 gates archived and .489 task range recorded from "
            f"{planned_task_source}; roadmap_next_present="
            f"{str(roadmap_next_present).lower()}; no active roadmap or conductor edit."
        )
    first_failure = failures[0] if failures else "unknown"
    return (
        "blocked_transition_v489: required transition input failed "
        f"({first_failure}); roadmap_next_present={str(roadmap_next_present).lower()}; "
        "no active roadmap or conductor edit."
    )


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = "20260707",
    duration_s: float | None = None,
    modification_status: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    start = time.perf_counter()
    capstone, capstone_meta = read_json_mapping(root / CAPSTONE_RELATIVE_PATH)
    active_roadmap, active_meta = read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    next_roadmap, next_meta = read_yaml_mapping(root / ROADMAP_NEXT_RELATIVE_PATH)
    active_modified = _modification_status(root, ROADMAP_RELATIVE_PATH, modification_status)
    conductor_dirty = _modification_status(root, CONDUCTOR_RELATIVE_PATH, modification_status)
    preconditions = precondition_summary(
        root=root,
        capstone_meta=capstone_meta,
        active_roadmap=active_roadmap,
        active_meta=active_meta,
        next_roadmap=next_roadmap,
        next_meta=next_meta,
        active_roadmap_modified=active_modified,
        conductor_modified=conductor_dirty,
    )
    task_source, task_ids = select_task_source(
        active_roadmap=active_roadmap,
        next_roadmap=next_roadmap,
        preconditions=preconditions,
    )
    failures = failed_preconditions(
        capstone_failure_rows=capstone_failures(capstone),
        preconditions=preconditions,
        planned_task_ids=task_ids,
    )
    status = "complete" if not failures else "blocked"
    roadmap_next_present = bool(preconditions["roadmap_next_present"])
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": round(
            float(duration_s if duration_s is not None else time.perf_counter() - start),
            6,
        ),
        "random_seed": RANDOM_SEED,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "status": status,
        "milestone": MILESTONE,
        "prior_milestone": PRIOR_MILESTONE,
        "prior_capstone_path": str(CAPSTONE_RELATIVE_PATH),
        "prior_gate_summary": (
            extract_prior_gate_summary(capstone) if capstone else empty_prior_gate_summary()
        ),
        "roadmap_next_present": roadmap_next_present,
        "roadmap_doc_present": bool(preconditions["roadmap_doc_present"]),
        "planned_task_count": len(task_ids),
        "planned_task_ids": list(task_ids),
        "planned_phase_names": list(preconditions["phase_names"]),
        "roadmap_doc_task_range": preconditions["task_range"],
        "planned_task_source": str(task_source),
        "downstream_gate_expectations": json.loads(json.dumps(DOWNSTREAM_GATE_EXPECTATIONS)),
        "active_roadmap_modified": active_modified,
        "conductor_modified": conductor_dirty,
        "failed_preconditions": failures,
        "preconditions_checked": preconditions,
        "cited_upstream_artifacts": cited_upstream_artifacts(root),
        "honest_verdict": build_honest_verdict(
            status=status,
            roadmap_next_present=roadmap_next_present,
            planned_task_source=str(task_source),
            failures=failures,
        ),
        "reproducibility_checksum": "",
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    validate_artifact(payload)
    return payload


def validate_artifact(payload: JsonMap) -> None:
    missing = [field for field in REQUIRED_SCHEMA_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if payload.get("schema") != SCHEMA:
        raise ValueError("schema mismatch")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    status = payload["status"]
    if status not in {"complete", "blocked"}:
        raise ValueError("status must be complete or blocked")
    if payload["milestone"] != MILESTONE:
        raise ValueError("milestone mismatch")
    if payload["prior_milestone"] != PRIOR_MILESTONE:
        raise ValueError("prior_milestone mismatch")
    if payload["prior_capstone_path"] != str(CAPSTONE_RELATIVE_PATH):
        raise ValueError("prior_capstone_path mismatch")
    prior_gate_summary = payload["prior_gate_summary"]
    if not isinstance(prior_gate_summary, Mapping) or set(prior_gate_summary) != set(
        PRIOR_GATE_FIELDS
    ):
        raise ValueError("prior_gate_summary must contain the required .488 gate fields")
    for field in ("roadmap_next_present", "roadmap_doc_present"):
        if not isinstance(payload[field], bool):
            raise ValueError(f"{field} must be a bare bool")
    planned_task_ids = payload["planned_task_ids"]
    if not isinstance(planned_task_ids, list) or not all(
        isinstance(task_id, str) for task_id in planned_task_ids
    ):
        raise ValueError("planned_task_ids must be a list of strings")
    if status == "complete" and planned_task_ids != EXPECTED_TASK_IDS:
        raise ValueError("planned_task_ids mismatch")
    if payload["planned_task_count"] != len(planned_task_ids):
        raise ValueError("planned_task_count must match planned_task_ids")
    if status == "complete" and payload["planned_task_count"] != len(EXPECTED_TASK_IDS):
        raise ValueError("planned_task_count mismatch")
    if payload["downstream_gate_expectations"] != DOWNSTREAM_GATE_EXPECTATIONS:
        raise ValueError("downstream_gate_expectations mismatch")
    if payload["active_roadmap_modified"] is not False:
        raise ValueError("active_roadmap_modified must be false")
    if payload["conductor_modified"] is not False:
        raise ValueError("conductor_modified must be false")
    honest_verdict = payload["honest_verdict"]
    if not isinstance(honest_verdict, str) or not honest_verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must start with complete: or blocked_")
    failures = payload["failed_preconditions"]
    if status == "complete" and failures:
        raise ValueError("complete status cannot carry failed preconditions")
    if status == "blocked" and not failures:
        raise ValueError("blocked status must carry failed preconditions")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")


def run(
    *,
    root: Path = REPO_ROOT,
    run_date: str = "20260707",
    duration_s: float | None = None,
) -> Path:
    artifact = build_artifact(root=root, run_date=run_date, duration_s=duration_s)
    out_path = root / RESULT_RELATIVE_PATH
    write_json(out_path, artifact)
    return out_path


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--run-date", default="20260707")
    args = parser.parse_args(argv)
    print(run(root=args.root, run_date=args.run_date))
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
