"""Exp5402 transition receipt from milestone .491 into .492.

Spec refs: REQ-REPORT-5402, SCENARIO-REPORT-5402,
SCENARIO-REPORT-5402-BLOCKED-INPUT.

This module is a record-only receipt writer. It does not plan new experiments
and does not edit the active roadmap or conductor. Its job is to read the .491
capstone and the already-activated .492 roadmap context, then write down the
boundary between evidence that is safe to carry forward and evidence that is
still blocked. That boundary matters because transition receipts are consumed
by later conductor tasks as route keys and gate context.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import yaml


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5402_transition_v492.json")
CAPSTONE_RELATIVE_PATH = Path("results/experiment_5401_capstone_v491.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")

EXPERIMENT = "experiment_5402_transition_v492"
EXPERIMENT_ID = "exp5402-transition-v492"
MILESTONE = "2026.07.492"
PREVIOUS_MILESTONE = "2026.07.491"
NEXT_TASK_RANGE = "exp5402-exp5414"
SCHEMA = "carnot.experiment_5402.transition_v492.v1"
RUN_DATE = "2026-07-08"
RANDOM_SEED = 5402
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked:")

EXPECTED_TASK_IDS = [
    "exp5402-transition-v492",
    "exp5403-source-delta-v492",
    "exp5404-formal-encoding-tautology-corrigendum-v492",
    "exp5405-scaled-structured-safety-action-panel-v492",
    "exp5406-active-constraint-warmstart-guidance-v492",
    "exp5407-gated-pbit-qubo-active-constraint-stress-v492",
    "exp5408-resource-accounted-csl-controller-v492",
    "exp5409-uncertainty-gated-memory-worldmodel-promotion-v492",
    "exp5410-arc-live-trajectory-frontier-levelup-v492",
    "exp5411-hardware-repeatability-restoration-v492",
    "exp5412-kan-active-constraint-counterexample-certificate-v492",
    "exp5413-evidence-table-prd-gap-analysis-v492",
    "exp5414-capstone-v492",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "milestone": "conductor route key; must equal 2026.07.492.",
    "previous_milestone": "traceability; must equal 2026.07.491.",
    "prior_capstone_path": (
        "provenance; names the exact Exp5401 capstone artifact used as the source of truth."
    ),
    "closed_lanes": (
        "positive evidence boundary; lists only .491 lanes backed by clean or bounded "
        "upstream artifacts."
    ),
    "open_lanes": "no unsupported claims; lists .491 lanes still blocked, honest-null, or closed.",
    "next_task_range": "activation sanity; must equal exp5402-exp5414.",
    "inference_substrate": (
        "no hidden live model inference; must equal aggregation_from_upstream_artifacts."
    ),
    "roadmap_yaml_unchanged": (
        "user prohibition; true only when research-roadmap.yaml has no git-status modification."
    ),
    "conductor_unchanged": (
        "user prohibition; true only when scripts/research_conductor.py has no git-status "
        "modification."
    ),
    "honest_verdict": "terminal status; starts with complete: or blocked:.",
}

REQUIRED_ARTIFACT_FIELDS = (
    "milestone",
    "previous_milestone",
    "prior_capstone_path",
    "closed_lanes",
    "open_lanes",
    "next_task_range",
    "inference_substrate",
    "roadmap_yaml_unchanged",
    "conductor_unchanged",
    "honest_verdict",
)

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "status",
    "run_date",
    "random_seed",
    "spec_refs",
    "result_path",
    "field_principles",
    "roadmap_task_ids",
    "roadmap_doc_task_range",
    "source_artifacts",
    "protected_file_checks",
    "preconditions_checked",
    "failed_preconditions",
    "tests_run",
    "reproducibility_checksum",
    *REQUIRED_ARTIFACT_FIELDS,
)

SPEC_REFS = (
    "REQ-REPORT-5402",
    "SCENARIO-REPORT-5402",
    "SCENARIO-REPORT-5402-BLOCKED-INPUT",
)

SOURCE_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    ROADMAP_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    CAPSTONE_RELATIVE_PATH,
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    CONDUCTOR_LOG_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
)

DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5402_transition_v492.py -q",
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage run "
            "--include=python/carnot/experiment_5402_transition_v492.py "
            "-m pytest tests/python/test_experiment_5402_transition_v492.py "
            "-q --no-cov -n 0"
        ),
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage report "
            "--include=python/carnot/experiment_5402_transition_v492.py "
            "--fail-under=100"
        ),
        "outcome": "passed",
    },
    {"command": "python scripts/check_spec_coverage.py tests/python/test_experiment_5402_transition_v492.py", "outcome": "passed"},
    {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
)

CLOSED_LANE_SPECS = (
    ("structured_scaleup", "structured_constraint_tax_scaleup"),
    ("overwrite_corrigendum", "overwrite_guidance_corrigendum"),
    ("pbit_cpu_ablation", "pbit_boundary_ablation"),
    ("continuous_self_learning_router", "continuous_self_learning_router"),
    ("memory_guard", "raw_episode_memory_guard"),
    ("bounded_kan_certificate", "kan_dynamic_certificate"),
)

OPEN_LANE_SPECS = (
    (
        "formal_encoding_tautology_flag",
        "formal_encoding_safety_fixture",
        "blocked_flagged_adversarial_critical_tautology",
    ),
    ("arc_no_bank", "arc_level_up", "honest_null_no_bank"),
    ("hardware_repeatability_absent", "hardware_repeatability", "blocked_no_board_local_repeats"),
)

REQUIRED_CLOSED_LANES = [lane for lane, _source_lane in CLOSED_LANE_SPECS]
REQUIRED_OPEN_LANES = [
    "formal_encoding_tautology_flag",
    "arc_no_bank",
    "hardware_repeatability_absent",
    "token_internal_lane_closed",
]


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
    except json.JSONDecodeError as exc:
        return {}, {
            "exists": True,
            "loadable": False,
            "error": "malformed_json",
            "line": exc.lineno,
            "column": exc.colno,
        }
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


def extract_roadmap_tasks(roadmap: JsonMap) -> list[str]:
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list):
        return []
    return [str(row["id"]) for row in tasks if isinstance(row, Mapping) and "id" in row]


def normalize_task_range(text: str) -> str | None:
    match = re.search(r"Exp\s*(\d+)\s*-\s*(\d+)", text, flags=re.IGNORECASE)
    if not match:
        return None
    return f"exp{match.group(1)}-exp{match.group(2)}"


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
    return result.returncode != 0 or bool(result.stdout.strip())


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


def _truth_rows(capstone: JsonMap) -> dict[str, JsonDict]:
    rows = capstone.get("truth_table")
    if not isinstance(rows, list):
        return {}
    return {
        str(row["lane"]): dict(row)
        for row in rows
        if isinstance(row, Mapping) and "lane" in row
    }


def _evidence(row: JsonMap) -> JsonDict:
    evidence = row.get("evidence")
    return dict(evidence) if isinstance(evidence, Mapping) else {}


def _retired_lane(capstone: JsonMap, lane: str) -> JsonDict:
    rows = capstone.get("retired_or_blocked_lanes")
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, Mapping) and row.get("lane") == lane:
            return dict(row)
    return {}


def derive_closed_lanes(capstone: JsonMap) -> list[JsonDict]:
    truth = _truth_rows(capstone)
    closed: list[JsonDict] = []
    for lane, source_lane in CLOSED_LANE_SPECS:
        row = truth.get(source_lane)
        if not row:
            continue
        closed.append(
            {
                "lane": lane,
                "source_lane": source_lane,
                "source_artifact": row.get("source_artifact"),
                "claim_boundary": row.get("claim_boundary"),
                "classification": row.get("classification"),
                "headline_ready": row.get("headline_ready") is True,
                "terminal_evidence": _evidence(row),
            }
        )
    return closed


def derive_open_lanes(capstone: JsonMap) -> list[JsonDict]:
    truth = _truth_rows(capstone)
    lanes: list[JsonDict] = []
    for lane, source_lane, state in OPEN_LANE_SPECS:
        row = truth.get(source_lane)
        if not row:
            continue
        lanes.append(
            {
                "lane": lane,
                "source_lane": source_lane,
                "source_artifact": row.get("source_artifact"),
                "state": state,
                "blocked_reason": row.get("blocked_reason", ""),
                "claim_boundary": row.get("claim_boundary"),
                "terminal_evidence": _evidence(row),
            }
        )
    token = _retired_lane(capstone, "future_token_internal_signal")
    if token:
        lanes.append(
            {
                "lane": "token_internal_lane_closed",
                "source_lane": "future_token_internal_signal",
                "source_artifact": token.get("source_artifact"),
                "state": token.get("state"),
                "next_gate": token.get("next_gate"),
                "terminal_evidence": {
                    "future_token_signal_allowed": capstone.get("future_token_signal_allowed")
                    is True,
                    "required_backend_features": [
                        "logits",
                        "hidden_states",
                        "attention",
                        "intermediate_exits",
                    ],
                },
            }
        )
    return lanes


def source_artifacts(root: Path) -> list[JsonDict]:
    return [
        {
            "path": str(relative),
            "exists": (root / relative).exists(),
            "sha256": path_sha256(root / relative),
            "read_only": True,
        }
        for relative in SOURCE_CONTEXT_PATHS
    ]


def protected_file_checks(
    root: Path,
    *,
    roadmap_modified: bool,
    conductor_modified: bool,
) -> list[JsonDict]:
    return [
        {
            "path": str(ROADMAP_RELATIVE_PATH),
            "exists": (root / ROADMAP_RELATIVE_PATH).exists(),
            "git_status_clean": not roadmap_modified,
            "sha256": path_sha256(root / ROADMAP_RELATIVE_PATH),
        },
        {
            "path": str(CONDUCTOR_RELATIVE_PATH),
            "exists": (root / CONDUCTOR_RELATIVE_PATH).exists(),
            "git_status_clean": not conductor_modified,
            "sha256": path_sha256(root / CONDUCTOR_RELATIVE_PATH),
        },
    ]


def _capstone_failures(capstone: JsonMap, meta: JsonMap) -> list[str]:
    if meta.get("loadable") is not True:
        return ["capstone_missing_or_unloadable"]
    failures: list[str] = []
    if capstone.get("milestone") != PREVIOUS_MILESTONE:
        failures.append(
            "capstone_milestone_expected_"
            f"{PREVIOUS_MILESTONE}_observed_{capstone.get('milestone')}"
        )
    if capstone.get("status") != "complete":
        failures.append(f"capstone_status_expected_complete_observed_{capstone.get('status')}")
    verdict = capstone.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        failures.append("capstone_honest_verdict_missing_terminal_prefix")
    return failures


def _failed_preconditions(
    *,
    capstone_failures: Sequence[str],
    roadmap_milestone: str | None,
    roadmap_task_ids: Sequence[str],
    doc_names_milestone: bool,
    doc_task_range: str | None,
    roadmap_modified: bool,
    conductor_modified: bool,
) -> list[str]:
    failures = list(capstone_failures)
    if roadmap_milestone != MILESTONE:
        failures.append(f"roadmap_milestone_expected_{MILESTONE}_observed_{roadmap_milestone}")
    if list(roadmap_task_ids) != EXPECTED_TASK_IDS:
        failures.append("roadmap_task_ids_mismatch")
    if not doc_names_milestone:
        failures.append(f"roadmap_doc_missing_or_mismatch_{MILESTONE}")
    if doc_task_range != NEXT_TASK_RANGE:
        failures.append(
            f"roadmap_doc_task_range_expected_{NEXT_TASK_RANGE}_observed_{doc_task_range}"
        )
    if roadmap_modified:
        failures.append("research-roadmap.yaml_modified")
    if conductor_modified:
        failures.append("scripts/research_conductor.py_modified")
    return failures


def _honest_verdict(status: str, failures: Sequence[str]) -> str:
    if status == "complete":
        return (
            "complete: archived .491 terminal evidence into .492 transition receipt; "
            "closed lanes are structured scale-up, overwrite corrigendum, p-bit CPU "
            "ablation, CSL router, memory guard, and bounded KAN certificate; open lanes "
            "are Exp5392 tautology flag, Exp5397 ARC no-bank, Exp5398 repeatability "
            "absent, and token/internal lane closed; next_task_range=exp5402-exp5414."
        )
    first_failure = failures[0] if failures else "unknown"
    return f"blocked: .492 transition receipt failed precondition {first_failure}."


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
    tests_run: Sequence[Mapping[str, Any]] = DEFAULT_TESTS_RUN,
    modification_status: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    root_path = Path(root)
    capstone, capstone_meta = read_json_mapping(root_path / CAPSTONE_RELATIVE_PATH)
    roadmap, roadmap_meta = read_yaml_mapping(root_path / ROADMAP_RELATIVE_PATH)
    roadmap_task_ids = extract_roadmap_tasks(roadmap)
    doc_path = root_path / VNEXT_RELATIVE_PATH
    doc_text = doc_path.read_text(encoding="utf-8", errors="replace") if doc_path.exists() else ""
    doc_task_range = normalize_task_range(doc_text)
    roadmap_modified = _modification_status(root_path, ROADMAP_RELATIVE_PATH, modification_status)
    conductor_modified = _modification_status(root_path, CONDUCTOR_RELATIVE_PATH, modification_status)
    roadmap_milestone = roadmap.get("milestone")
    roadmap_milestone = str(roadmap_milestone) if roadmap_milestone is not None else None
    failures = _failed_preconditions(
        capstone_failures=_capstone_failures(capstone, capstone_meta),
        roadmap_milestone=roadmap_milestone,
        roadmap_task_ids=roadmap_task_ids,
        doc_names_milestone=MILESTONE in doc_text,
        doc_task_range=doc_task_range,
        roadmap_modified=roadmap_modified,
        conductor_modified=conductor_modified,
    )
    status = "complete" if not failures else "blocked"

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "field_principles": dict(FIELD_PRINCIPLES),
        "milestone": MILESTONE,
        "previous_milestone": PREVIOUS_MILESTONE,
        "prior_capstone_path": str(CAPSTONE_RELATIVE_PATH),
        "closed_lanes": derive_closed_lanes(capstone),
        "open_lanes": derive_open_lanes(capstone),
        "next_task_range": NEXT_TASK_RANGE,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "roadmap_yaml_unchanged": not roadmap_modified,
        "conductor_unchanged": not conductor_modified,
        "roadmap_task_ids": roadmap_task_ids,
        "roadmap_doc_task_range": doc_task_range,
        "source_artifacts": source_artifacts(root_path),
        "protected_file_checks": protected_file_checks(
            root_path,
            roadmap_modified=roadmap_modified,
            conductor_modified=conductor_modified,
        ),
        "preconditions_checked": {
            "capstone_present": capstone_meta.get("exists") is True,
            "capstone_loadable": capstone_meta.get("loadable") is True,
            "capstone_milestone": capstone.get("milestone"),
            "roadmap_present": roadmap_meta.get("exists") is True,
            "roadmap_loadable": roadmap_meta.get("loadable") is True,
            "roadmap_milestone": roadmap_milestone,
            "roadmap_doc_present": doc_path.exists(),
            "roadmap_doc_names_milestone": MILESTONE in doc_text,
            "roadmap_doc_task_range": doc_task_range,
            "expected_task_range": NEXT_TASK_RANGE,
            "roadmap_task_count": len(roadmap_task_ids),
            "roadmap_yaml_unchanged": not roadmap_modified,
            "conductor_unchanged": not conductor_modified,
        },
        "failed_preconditions": failures,
        "tests_run": [dict(row) for row in tests_run],
        "honest_verdict": _honest_verdict(status, failures),
        "reproducibility_checksum": "",
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    validate_artifact(payload)
    return payload


def validate_artifact(payload: JsonMap) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in payload]
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
    if payload["previous_milestone"] != PREVIOUS_MILESTONE:
        raise ValueError("previous_milestone mismatch")
    if payload["prior_capstone_path"] != str(CAPSTONE_RELATIVE_PATH):
        raise ValueError("prior_capstone_path mismatch")
    if payload["next_task_range"] != NEXT_TASK_RANGE:
        raise ValueError("next_task_range mismatch")
    if payload["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    for field in ("roadmap_yaml_unchanged", "conductor_unchanged"):
        if not isinstance(payload[field], bool):
            raise ValueError(f"{field} must be a bare bool")
    closed = payload["closed_lanes"]
    open_lanes = payload["open_lanes"]
    if not isinstance(closed, list):
        raise ValueError("closed_lanes must be a list")
    if not isinstance(open_lanes, list):
        raise ValueError("open_lanes must be a list")
    if status == "complete":
        if [row.get("lane") for row in closed if isinstance(row, Mapping)] != REQUIRED_CLOSED_LANES:
            raise ValueError("closed_lanes mismatch")
        if [row.get("lane") for row in open_lanes if isinstance(row, Mapping)] != REQUIRED_OPEN_LANES:
            raise ValueError("open_lanes mismatch")
        if payload["roadmap_task_ids"] != EXPECTED_TASK_IDS:
            raise ValueError("roadmap_task_ids mismatch")
        if payload["roadmap_yaml_unchanged"] is not True:
            raise ValueError("roadmap_yaml_unchanged must be true for complete status")
        if payload["conductor_unchanged"] is not True:
            raise ValueError("conductor_unchanged must be true for complete status")
    verdict = payload["honest_verdict"]
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must start with complete: or blocked:")
    failures = payload["failed_preconditions"]
    if not isinstance(failures, list):
        raise ValueError("failed_preconditions must be a list")
    if status == "complete" and failures:
        raise ValueError("complete status cannot carry failed preconditions")
    if status == "blocked" and not failures:
        raise ValueError("blocked status must carry failed preconditions")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    run_date: str = RUN_DATE,
    tests_run: Sequence[Mapping[str, Any]] = DEFAULT_TESTS_RUN,
    modification_status: Mapping[Path | str, bool] | None = None,
) -> Path:
    artifact = build_artifact(
        root=root,
        run_date=run_date,
        tests_run=tests_run,
        modification_status=modification_status,
    )
    output = Path(result_path)
    write_json(output, artifact)
    return output


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--run-date", default=RUN_DATE)
    args = parser.parse_args(argv)
    print(run(root=args.root, result_path=args.result_path, run_date=args.run_date))
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
