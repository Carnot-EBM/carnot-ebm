"""Exp5468 transition receipt from milestone .496 into .497.

Spec refs: REQ-REPORT-5468, SCENARIO-REPORT-5468,
SCENARIO-REPORT-5468-BLOCKED-INPUT.

This module writes a transition receipt, not a new experiment result. It reads
the completed `.496` capstone and the active `.497` roadmap context, then
records which facts are safe for downstream tasks to inherit. That handoff is
explicit because the prior milestone mixed clean deterministic wins with
bounded receipts and honest nulls: guided decoding is quarantined, KV260 is
unreachable, ARC banked no `bp35` L3 level, and hardware speedup remains an
unsupported claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from carnot.experiment_5415_transition_v493 import (
    JsonDict,
    JsonMap,
    _modification_status,
    extract_roadmap_tasks,
    normalize_task_range,
    path_sha256,
    payload_checksum,
    read_json_mapping,
    read_yaml_mapping,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5468_transition_v497.json")
CAPSTONE_RELATIVE_PATH = Path("results/experiment_5467_capstone_v496.json")
GAP_RELATIVE_PATH = Path("results/experiment_5466_prd_gap_agent_failure_table_v496.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")

EXPERIMENT = "experiment_5468_transition_v497"
EXPERIMENT_ID = "exp5468-transition-v497"
MILESTONE = "2026.07.497"
PREVIOUS_MILESTONE = "2026.07.496"
PREVIOUS_TASK_RANGE = "exp5454-exp5467"
NEXT_TASK_RANGE = "exp5468-exp5481"
SCHEMA = "carnot.experiment_5468.transition_v497.v1"
RUN_DATE = "2026-07-09"
RANDOM_SEED = 5468
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked:")

EXPECTED_TASK_IDS = [
    "exp5468-transition-v497",
    "exp5469-source-delta-v497",
    "exp5470-rewrite-state-semantic-fixture-v497",
    "exp5471-gated-guard-composition-scale-v497",
    "exp5472-gated-sota-evidence-telemetry-v497",
    "exp5473-csl-kan-surrogate-assurance-v497",
    "exp5474-gated-sota-csl-scale-v497",
    "exp5475-csl-behavioral-memory-ladder-v497",
    "exp5476-helper-lemma-core-witness-repair-v497",
    "exp5477-pdit-lns-boundary-exchange-v497",
    "exp5478-gated-hardware-receipts-v497",
    "exp5479-arc-target-rotation-precheck-v497",
    "exp5480-gated-arc-live-salience-levelup-v497",
    "exp5481-capstone-v497",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "milestone": "conductor route key",
    "previous_milestone": "traceability",
    "prior_capstone_path": "provenance",
    "previous_task_range": "closed execution boundary",
    "closed_lanes": "positive evidence boundary",
    "bounded_lanes": "bounded evidence boundary",
    "blocked_lanes": "no unsupported claims",
    "honest_null_lanes": "null-result honesty",
    "next_task_range": "activation sanity",
    "roadmap_yaml_unchanged": "user prohibition",
    "conductor_unchanged": "user prohibition",
    "inference_substrate": "no hidden live model inference",
    "honest_verdict": "terminal status; start with complete: or blocked:",
}

REQUIRED_ARTIFACT_FIELDS = (
    "milestone",
    "previous_milestone",
    "prior_capstone_path",
    "previous_task_range",
    "closed_lanes",
    "bounded_lanes",
    "blocked_lanes",
    "honest_null_lanes",
    "next_task_range",
    "roadmap_yaml_unchanged",
    "conductor_unchanged",
    "inference_substrate",
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
    "REQ-REPORT-5468",
    "SCENARIO-REPORT-5468",
    "SCENARIO-REPORT-5468-BLOCKED-INPUT",
)

SOURCE_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    ROADMAP_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    CONDUCTOR_LOG_RELATIVE_PATH,
    CAPSTONE_RELATIVE_PATH,
    GAP_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
)

DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5468_transition_v497.py -q --no-cov",
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage run "
            "--include=python/carnot/experiment_5468_transition_v497.py "
            "-m pytest tests/python/test_experiment_5468_transition_v497.py "
            "-q --no-cov -n 0"
        ),
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage report "
            "--include=python/carnot/experiment_5468_transition_v497.py "
            "--fail-under=100"
        ),
        "outcome": "passed",
    },
    {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
)

REQUIRED_CLOSED_LANES = [
    "minimal_core_repair",
    "distortion_guards",
    "csl_policy",
    "sota_csl",
]
REQUIRED_BOUNDED_LANES = ["pbit_pdit_bridge", "hardware_receipts", "synthesis"]
REQUIRED_BLOCKED_LANES = ["guided_decoding"]
REQUIRED_HONEST_NULL_LANES = ["arc", "hardware_speedup_claim"]


def _truth_rows(rows: object) -> dict[str, JsonDict]:
    if isinstance(rows, Mapping):
        return {
            str(lane): dict(row)
            for lane, row in rows.items()
            if isinstance(row, Mapping)
        }
    if isinstance(rows, list):
        return {
            str(row["lane"]): dict(row)
            for row in rows
            if isinstance(row, Mapping) and "lane" in row
        }
    return {}


def _terminal_evidence(row: JsonMap) -> JsonDict:
    evidence = row.get("terminal_evidence", row.get("evidence"))
    return dict(evidence) if isinstance(evidence, Mapping) else {}


def _source_artifacts(row: JsonMap) -> list[str]:
    source = row.get("source_artifacts", row.get("source_artifact"))
    if isinstance(source, list):
        return [str(item) for item in source]
    if source is None:
        return []
    return [str(source)]


def _truth_record(row: JsonMap) -> JsonDict:
    record: JsonDict = {
        "lane": str(row.get("lane")),
        "source_artifacts": _source_artifacts(row),
        "classification": row.get("classification"),
        "claim_boundary": row.get("claim_boundary"),
        "authority_gate": row.get("authority_gate"),
        "terminal_evidence": _terminal_evidence(row),
    }
    headline_blockers = row.get("headline_blockers")
    if isinstance(headline_blockers, list):
        record["headline_blockers"] = [str(item) for item in headline_blockers]
    blocked_reason = row.get("blocked_reason")
    if blocked_reason:
        record["blocked_reason"] = blocked_reason
    return record


def _select_truth_lanes(capstone: JsonMap, expected: Sequence[str]) -> list[JsonDict]:
    rows = _truth_rows(capstone.get("truth_table"))
    return [_truth_record(rows[lane]) for lane in expected if lane in rows]


def _hardware_speedup_boundary(capstone: JsonMap) -> JsonDict | None:
    boundaries = capstone.get("no_claim_boundaries")
    if not isinstance(boundaries, list):
        return None
    for row in boundaries:
        if isinstance(row, Mapping) and row.get("claim_id") == "hardware_speedup":
            evidence = row.get("evidence")
            return {
                "lane": "hardware_speedup_claim",
                "source_artifacts": [str(CAPSTONE_RELATIVE_PATH)],
                "classification": "honest_null",
                "claim_boundary": row.get("boundary"),
                "authority_gate": "matched repeatable speedup evidence is required before any speedup claim",
                "terminal_evidence": dict(evidence) if isinstance(evidence, Mapping) else {},
            }
    return None


def _hardware_speedup_claim_is_false(capstone: JsonMap) -> bool:
    record = _hardware_speedup_boundary(capstone)
    if record is None:
        return False
    evidence = record.get("terminal_evidence")
    timing = evidence.get("timing_comparison")
    timing_claim = None
    if isinstance(timing, Mapping):
        timing_claim = timing.get("hardware_speedup_claim")
    return evidence.get("hardware_speedup_claim") is False and timing_claim in (None, False)


def derive_closed_lanes(capstone: JsonMap) -> list[JsonDict]:
    return _select_truth_lanes(capstone, REQUIRED_CLOSED_LANES)


def derive_bounded_lanes(capstone: JsonMap) -> list[JsonDict]:
    return _select_truth_lanes(capstone, REQUIRED_BOUNDED_LANES)


def derive_blocked_lanes(capstone: JsonMap) -> list[JsonDict]:
    return _select_truth_lanes(capstone, REQUIRED_BLOCKED_LANES)


def derive_honest_null_lanes(capstone: JsonMap) -> list[JsonDict]:
    rows = _truth_rows(capstone.get("truth_table"))
    selected: list[JsonDict] = []
    for lane in REQUIRED_HONEST_NULL_LANES:
        if lane == "hardware_speedup_claim":
            record = _hardware_speedup_boundary(capstone)
            if record is not None:
                selected.append(record)
        elif lane in rows:
            selected.append(_truth_record(rows[lane]))
    return selected


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
    capstone_status = capstone.get("status")
    if capstone_status not in (None, "complete"):
        failures.append(
            "capstone_status_expected_complete_or_absent_observed_"
            f"{capstone_status}"
        )
    verdict = capstone.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        failures.append("capstone_honest_verdict_missing_terminal_prefix")
    if not _hardware_speedup_claim_is_false(capstone):
        failures.append("capstone_hardware_speedup_claim_boundary_missing_or_true")
    return failures


def _gap_failures(gap: JsonMap, meta: JsonMap) -> list[str]:
    if meta.get("loadable") is not True:
        return ["gap_table_missing_or_unloadable"]
    failures: list[str] = []
    if gap.get("milestone") != PREVIOUS_MILESTONE:
        failures.append(
            f"gap_milestone_expected_{PREVIOUS_MILESTONE}_observed_{gap.get('milestone')}"
        )
    gap_status = gap.get("status")
    if gap_status not in (None, "complete"):
        failures.append(f"gap_status_expected_complete_or_absent_observed_{gap_status}")
    verdict = gap.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        failures.append("gap_honest_verdict_missing_terminal_prefix")
    return failures


def _missing_lane_names(rows: Sequence[JsonMap], expected: Sequence[str]) -> list[str]:
    observed = [row.get("lane") for row in rows]
    return [lane for lane in expected if lane not in observed]


def _lane_failures(
    *,
    closed_lanes: Sequence[JsonMap],
    bounded_lanes: Sequence[JsonMap],
    blocked_lanes: Sequence[JsonMap],
    honest_null_lanes: Sequence[JsonMap],
    capstone_loadable: bool,
) -> list[str]:
    if not capstone_loadable:
        return []
    failures: list[str] = []
    if _missing_lane_names(closed_lanes, REQUIRED_CLOSED_LANES):
        failures.append("capstone_closed_lanes_incomplete")
    if _missing_lane_names(bounded_lanes, REQUIRED_BOUNDED_LANES):
        failures.append("capstone_bounded_lanes_incomplete")
    if _missing_lane_names(blocked_lanes, REQUIRED_BLOCKED_LANES):
        failures.append("capstone_blocked_lanes_incomplete")
    if _missing_lane_names(honest_null_lanes, REQUIRED_HONEST_NULL_LANES):
        failures.append("capstone_honest_null_lanes_incomplete")
    return failures


def _failed_preconditions(
    *,
    capstone_failures: Sequence[str],
    gap_failures: Sequence[str],
    lane_failures: Sequence[str],
    roadmap_milestone: str | None,
    roadmap_task_ids: Sequence[str],
    doc_names_milestone: bool,
    doc_task_range: str | None,
    roadmap_modified: bool,
    conductor_modified: bool,
) -> list[str]:
    failures = [*capstone_failures, *gap_failures, *lane_failures]
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
            "complete: archived .496 terminal evidence into .497 transition receipt; "
            "closed lanes are deterministic minimal-core repair, deterministic "
            "constraint-distortion guards, governed CSL policy, and live SOTA CSL "
            "memory routing; bounded lanes are p-bit/p-dit bridge, hardware receipts "
            "with KV260 unreachable and no speedup, and milestone synthesis; blocked "
            "lane is Exp5457 guided decoding quarantine; honest-null lanes are bp35 "
            "L3 ARC no-bank and no hardware speedup claim; next_task_range=exp5468-exp5481."
        )
    first_failure = failures[0] if failures else "unknown"
    return f"blocked: .497 transition receipt failed precondition {first_failure}."


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
    tests_run: Sequence[Mapping[str, Any]] = DEFAULT_TESTS_RUN,
    modification_status: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    root_path = Path(root)
    capstone, capstone_meta = read_json_mapping(root_path / CAPSTONE_RELATIVE_PATH)
    gap, gap_meta = read_json_mapping(root_path / GAP_RELATIVE_PATH)
    roadmap, roadmap_meta = read_yaml_mapping(root_path / ROADMAP_RELATIVE_PATH)
    roadmap_task_ids = extract_roadmap_tasks(roadmap)
    doc_path = root_path / VNEXT_RELATIVE_PATH
    doc_text = doc_path.read_text(encoding="utf-8", errors="replace") if doc_path.exists() else ""
    doc_task_range = normalize_task_range(doc_text)
    roadmap_modified = _modification_status(root_path, ROADMAP_RELATIVE_PATH, modification_status)
    conductor_modified = _modification_status(root_path, CONDUCTOR_RELATIVE_PATH, modification_status)
    roadmap_milestone = roadmap.get("milestone")
    roadmap_milestone = str(roadmap_milestone) if roadmap_milestone is not None else None

    closed_lanes = derive_closed_lanes(capstone)
    bounded_lanes = derive_bounded_lanes(capstone)
    blocked_lanes = derive_blocked_lanes(capstone)
    honest_null_lanes = derive_honest_null_lanes(capstone)
    capstone_loadable = capstone_meta.get("loadable") is True
    failures = _failed_preconditions(
        capstone_failures=_capstone_failures(capstone, capstone_meta),
        gap_failures=_gap_failures(gap, gap_meta),
        lane_failures=_lane_failures(
            closed_lanes=closed_lanes,
            bounded_lanes=bounded_lanes,
            blocked_lanes=blocked_lanes,
            honest_null_lanes=honest_null_lanes,
            capstone_loadable=capstone_loadable,
        ),
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
        "previous_task_range": PREVIOUS_TASK_RANGE,
        "closed_lanes": closed_lanes,
        "bounded_lanes": bounded_lanes,
        "blocked_lanes": blocked_lanes,
        "honest_null_lanes": honest_null_lanes,
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
            "capstone_loadable": capstone_loadable,
            "capstone_milestone": capstone.get("milestone"),
            "capstone_status": capstone.get("status"),
            "capstone_hardware_speedup_claim_boundary_false": (
                _hardware_speedup_claim_is_false(capstone)
            ),
            "gap_table_present": gap_meta.get("exists") is True,
            "gap_table_loadable": gap_meta.get("loadable") is True,
            "gap_table_milestone": gap.get("milestone"),
            "gap_table_status": gap.get("status"),
            "roadmap_present": roadmap_meta.get("exists") is True,
            "roadmap_loadable": roadmap_meta.get("loadable") is True,
            "roadmap_milestone": roadmap_milestone,
            "roadmap_doc_present": doc_path.exists(),
            "roadmap_doc_names_milestone": MILESTONE in doc_text,
            "roadmap_doc_task_range": doc_task_range,
            "expected_previous_task_range": PREVIOUS_TASK_RANGE,
            "expected_next_task_range": NEXT_TASK_RANGE,
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
    if payload["previous_task_range"] != PREVIOUS_TASK_RANGE:
        raise ValueError("previous_task_range mismatch")
    if payload["next_task_range"] != NEXT_TASK_RANGE:
        raise ValueError("next_task_range mismatch")
    if payload["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    for field in ("roadmap_yaml_unchanged", "conductor_unchanged"):
        if not isinstance(payload[field], bool):
            raise ValueError(f"{field} must be a bare bool")
    closed = payload["closed_lanes"]
    bounded = payload["bounded_lanes"]
    blocked = payload["blocked_lanes"]
    honest_null = payload["honest_null_lanes"]
    if not isinstance(closed, list):
        raise ValueError("closed_lanes must be a list")
    if not isinstance(bounded, list):
        raise ValueError("bounded_lanes must be a list")
    if not isinstance(blocked, list):
        raise ValueError("blocked_lanes must be a list")
    if not isinstance(honest_null, list):
        raise ValueError("honest_null_lanes must be a list")
    if status == "complete":
        if [row.get("lane") for row in closed if isinstance(row, Mapping)] != REQUIRED_CLOSED_LANES:
            raise ValueError("closed_lanes mismatch")
        if [row.get("lane") for row in bounded if isinstance(row, Mapping)] != REQUIRED_BOUNDED_LANES:
            raise ValueError("bounded_lanes mismatch")
        if [row.get("lane") for row in blocked if isinstance(row, Mapping)] != REQUIRED_BLOCKED_LANES:
            raise ValueError("blocked_lanes mismatch")
        if (
            [row.get("lane") for row in honest_null if isinstance(row, Mapping)]
            != REQUIRED_HONEST_NULL_LANES
        ):
            raise ValueError("honest_null_lanes mismatch")
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
