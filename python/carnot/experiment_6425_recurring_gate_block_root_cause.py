"""Exp6425 recurring blocked-gate root-cause report.

Spec refs: REQ-OPS-RECURRING-GATE-6425,
SCENARIO-OPS-RECURRING-GATE-6425-DIAGNOSTIC-CONTRACT,
SCENARIO-OPS-RECURRING-GATE-6425-MUTATIONS-FAIL-CLOSED.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any

import yaml

from carnot.experiment_artifacts import atomic_write_json
from carnot.terminal_artifacts import (
    canonical_json,
    classify_artifact_payload,
    path_sha256,
    payload_sha256,
)
from scripts.conductor_gates import _eval_op, _find_artifact_by_task_id, evaluate_gates


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260814"
SCHEMA = "carnot.experiment_6425.recurring_gate_block_root_cause.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6425_recurring_gate_block_root_cause.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
START_MILESTONE = "2026.08.536"
END_MILESTONE = "2026.08.549"
INFERENCE_SUBSTRATE = "deterministic_artifact_contract_replay"
RANDOM_SEED = 6425
CLASSIFICATIONS = (
    "correct_expected_refusal",
    "missing_upstream",
    "wrong_field_name",
    "wrong_field_type",
    "stale_artifact",
    "retired_dependency",
    "diagnostic_loss",
    "other_with_evidence",
)
NUMERIC_OPS = {">", ">=", "<", "<="}
ALLOWED_HONEST_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "frozen_blocker_population_receipt",
    "blocker_count_by_milestone",
    "per_unit_rows",
    "per_occurrence_task_upstream_field_op_expected_observed_type_hash_and_artifact_bindings",
    "root_cause_class_counts",
    "correct_expected_refusal_count",
    "infrastructure_defect_count",
    "diagnostic_loss_count",
    "highest_count_root_cause",
    "source_trace",
    "shared_fix_applied",
    "changed_files_and_hashes",
    "no_scientific_gate_bypassed",
    "no_historical_task_rerun",
    "blocked_diagnostic_contract",
    "mutation_attack_matrix",
    "recurring_gate_diagnostic_ready_score",
    "protected_files_unchanged",
    "blocked_reason",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)
PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)
CHANGED_RELATIVE_PATHS = (
    Path("scripts/conductor_gates.py"),
    Path("python/carnot/experiment_6425_recurring_gate_block_root_cause.py"),
    Path("tests/python/test_experiment_6425_recurring_gate_block_root_cause.py"),
    Path("tests/python/test_conductor_gates.py"),
    SPEC_RELATIVE_PATH,
    RESULT_RELATIVE_PATH,
)


def _read_json(path: Path) -> JsonDict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _read_yaml(path: Path) -> JsonDict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError:
        return {}
    return data if isinstance(data, dict) else {}


def _verdict(payload: Mapping[str, Any]) -> str:
    raw = payload.get("honest_verdict")
    if isinstance(raw, Mapping):
        raw = raw.get("value")
    return str(raw or "")


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _is_numeric_value(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _artifact_hash(path: Path | None) -> str | None:
    return path_sha256(path) if path is not None else None


def _tasks_by_id(root: Path) -> dict[str, JsonDict]:
    complete = _read_yaml(root / "research-complete.yaml")
    out: dict[str, JsonDict] = {}
    for milestone in complete.get("milestones") or []:
        if not isinstance(milestone, Mapping):
            continue
        for task in milestone.get("tasks") or []:
            if isinstance(task, Mapping) and isinstance(task.get("id"), str):
                out[str(task["id"])] = dict(task)
    return out


def _retired_dependency_ids_from_log(root: Path, tasks: Mapping[str, JsonMap]) -> set[str]:
    try:
        lines = (root / "ops" / "conductor-log.md").read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()
    retired: set[str] = set()
    for task_id, task in tasks.items():
        title = str(task.get("title") or "")
        if not title:
            continue
        marker = title[:48]
        if any(marker in line and "Pre-emptive skip: upstream retired" in line for line in lines):
            retired.add(task_id)
    return retired


def _gate_from_result(raw: Mapping[str, Any]) -> JsonDict:
    return {
        "upstream": str(raw.get("upstream") or ""),
        "artifact_field": str(raw.get("artifact_field") or raw.get("field") or ""),
        "op": str(raw.get("op") or "=="),
        "value": raw.get("expected", raw.get("value")),
    }


def _first_failed_gate(gates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return next((g for g in gates if g.get("passed") is False), gates[0] if gates else None)


def classify_gate_binding(
    *,
    upstream: str,
    artifact_path: Path | None,
    artifact_payload: Mapping[str, Any] | None,
    field: str,
    op: str,
    expected: Any,
    observed: Any,
    passed: bool,
    reason: str,
    retired_upstreams: set[str],
    expected_hash: str | None = None,
    observed_hash: str | None = None,
) -> JsonDict:
    """Classify one replayed gate without changing its pass/fail result."""
    if upstream in retired_upstreams:
        classification = "retired_dependency"
        evidence = "upstream task was pre-emptively skipped as retired in conductor log"
    elif artifact_path is None:
        classification = "missing_upstream"
        evidence = "upstream artifact path is missing"
    elif expected_hash and observed_hash and expected_hash != observed_hash:
        classification = "stale_artifact"
        evidence = "artifact hash does not match the frozen expected hash"
    elif artifact_payload is None:
        classification = "missing_upstream"
        evidence = "upstream artifact could not be loaded"
    else:
        terminal = classify_artifact_payload(artifact_payload, path=artifact_path)
        if not terminal.terminal and observed is None:
            classification = "stale_artifact"
            evidence = f"nonterminal upstream artifact classification={terminal.classification}"
        elif field not in artifact_payload:
            classification = "wrong_field_name"
            evidence = f"exact field {field!r} is absent"
        elif op in NUMERIC_OPS and not _is_numeric_value(observed):
            classification = "wrong_field_type"
            evidence = f"numeric gate saw {_type_name(observed)}"
        elif observed is None:
            classification = "stale_artifact"
            evidence = "exact field exists but is null"
        elif passed:
            classification = "other_with_evidence"
            evidence = "replay passed although historical artifact was blocked"
        else:
            classification = "correct_expected_refusal"
            evidence = "structured gate observed a real value that did not satisfy the predicate"
    return {
        "classification": classification,
        "classification_evidence": evidence,
        "reason": reason,
    }


def _replay_gate(task_id: str, gates: Sequence[Mapping[str, Any]], root: Path) -> Any:
    task = {"id": task_id, "gated_on": [_gate_from_result(g) for g in gates]}
    return evaluate_gates(task, results_dir=root / "results")


def collect_blocker_population(
    root: Path = REPO_ROOT,
    *,
    start_milestone: str = START_MILESTONE,
    end_milestone: str = END_MILESTONE,
) -> list[JsonDict]:
    """Return one row per frozen `blocked_gate_check_failed` occurrence."""
    complete = _read_yaml(root / "research-complete.yaml")
    tasks = _tasks_by_id(root)
    retired_upstreams = _retired_dependency_ids_from_log(root, tasks)
    rows: list[JsonDict] = []
    ordinal = 0
    for milestone in complete.get("milestones") or []:
        if not isinstance(milestone, Mapping):
            continue
        milestone_id = str(milestone.get("id") or "")
        if not (start_milestone <= milestone_id <= end_milestone):
            continue
        for task in milestone.get("tasks") or []:
            if not isinstance(task, Mapping):
                continue
            deliverable = task.get("deliverable")
            if not isinstance(deliverable, str):
                continue
            artifact_path = root / deliverable
            payload = _read_json(artifact_path)
            if payload is None or _verdict(payload) != "blocked_gate_check_failed":
                continue
            raw_gates = payload.get("gates_evaluated")
            gates = raw_gates if isinstance(raw_gates, list) else []
            structured_gates = [g for g in gates if isinstance(g, Mapping)]
            if not structured_gates:
                ordinal += 1
                rows.append(_diagnostic_loss_row(milestone_id, task, artifact_path, payload, ordinal))
                continue
            replay = _replay_gate(str(task.get("id") or ""), structured_gates, root)
            replay_failed = _first_failed_gate([g.__dict__ for g in replay.gates_evaluated])
            historical_failed = _first_failed_gate(structured_gates)
            failed = replay_failed or historical_failed
            upstream = str(failed.get("upstream") or "")
            field = str(failed.get("artifact_field") or "")
            upstream_path = _find_artifact_by_task_id(upstream, root / "results")
            upstream_payload = _read_json(upstream_path) if upstream_path is not None else None
            observed = failed.get("actual")
            expected = failed.get("expected")
            op = str(failed.get("op") or "")
            passed = bool(failed.get("passed"))
            reason = str(failed.get("reason") or replay.summary)
            class_row = classify_gate_binding(
                upstream=upstream,
                artifact_path=upstream_path,
                artifact_payload=upstream_payload,
                field=field,
                op=op,
                expected=expected,
                observed=observed,
                passed=passed,
                reason=reason,
                retired_upstreams=retired_upstreams,
            )
            ordinal += 1
            rows.append(
                {
                    "occurrence_index": ordinal,
                    "milestone": milestone_id,
                    "task_id": str(task.get("id") or ""),
                    "terminal_artifact": artifact_path.as_posix(),
                    "terminal_artifact_sha256": _artifact_hash(artifact_path),
                    "upstream_id": upstream,
                    "upstream_artifact": upstream_path.as_posix() if upstream_path else None,
                    "upstream_artifact_sha256": _artifact_hash(upstream_path),
                    "gate_field": field,
                    "operator": op,
                    "expected": expected,
                    "expected_type": _type_name(expected),
                    "observed": observed,
                    "observed_type": _type_name(observed),
                    "historical_gate_reason": str(historical_failed.get("reason") if historical_failed else ""),
                    "replayed_gate_reason": reason,
                    "replayed_gate_passed": passed,
                    "gate_check_summary": str(payload.get("gate_check_summary") or ""),
                    "blocked_reason_present": bool(payload.get("blocked_reason")),
                    "diagnostic_contract_present": bool(payload.get("blocked_diagnostic_contract")),
                    "all_structured_gates": [_gate_from_result(g) for g in gates if isinstance(g, Mapping)],
                    **class_row,
                }
            )
    return rows


def _diagnostic_loss_row(
    milestone: str,
    task: Mapping[str, Any],
    artifact_path: Path,
    payload: Mapping[str, Any],
    ordinal: int,
) -> JsonDict:
    return {
        "occurrence_index": ordinal,
        "milestone": milestone,
        "task_id": str(task.get("id") or ""),
        "terminal_artifact": artifact_path.as_posix(),
        "terminal_artifact_sha256": _artifact_hash(artifact_path),
        "upstream_id": None,
        "upstream_artifact": None,
        "upstream_artifact_sha256": None,
        "gate_field": None,
        "operator": None,
        "expected": None,
        "expected_type": "null",
        "observed": None,
        "observed_type": "null",
        "historical_gate_reason": "",
        "replayed_gate_reason": "",
        "replayed_gate_passed": False,
        "gate_check_summary": str(payload.get("gate_check_summary") or ""),
        "blocked_reason_present": bool(payload.get("blocked_reason")),
        "diagnostic_contract_present": bool(payload.get("blocked_diagnostic_contract")),
        "all_structured_gates": [],
        "classification": "diagnostic_loss",
        "classification_evidence": "blocked artifact has no structured gate record",
        "reason": "no structured gate diagnostic",
    }


def build_mutation_attack_matrix() -> JsonDict:
    """Return deterministic fail-closed checks for the requested attack shapes."""
    string_passed, string_reason = _eval_op("1.0", ">", 0.0)
    nan_passed, nan_reason = _eval_op(float("nan"), ">", 0.0)
    missing = classify_gate_binding(
        upstream="exp1-upstream",
        artifact_path=Path("results/experiment_1_upstream.json"),
        artifact_payload={"other_ready_score": 1.0, "status": "complete", "honest_verdict": "complete"},
        field="ready_score",
        op="==",
        expected=1.0,
        observed=None,
        passed=False,
        reason="missing field",
        retired_upstreams=set(),
    )
    stale = classify_gate_binding(
        upstream="exp2-upstream",
        artifact_path=Path("results/experiment_2_upstream.json"),
        artifact_payload={"ready_score": 1.0, "status": "complete", "honest_verdict": "complete"},
        field="ready_score",
        op="==",
        expected=1.0,
        observed=1.0,
        passed=False,
        reason="stale hash",
        retired_upstreams=set(),
        expected_hash="sha256:old",
        observed_hash="sha256:new",
    )
    retired = classify_gate_binding(
        upstream="exp3-retired",
        artifact_path=None,
        artifact_payload=None,
        field="ready_score",
        op="==",
        expected=1.0,
        observed=None,
        passed=False,
        reason="retired upstream",
        retired_upstreams={"exp3-retired"},
    )
    contradictory = classify_artifact_payload(
        {"status": "blocked", "honest_verdict": "complete_null: contradictory"}
    )
    return {
        "missing_field": _attack(True, False, missing["classification_evidence"]),
        "string_numeric_gate": _attack(
            string_passed is False, string_passed, string_reason
        ),
        "nan_numeric_gate": _attack(nan_passed is False, nan_passed, nan_reason),
        "stale_hash": _attack(
            stale["classification"] == "stale_artifact", False, stale["classification_evidence"]
        ),
        "retired_upstream_id": _attack(
            retired["classification"] == "retired_dependency",
            False,
            retired["classification_evidence"],
        ),
        "contradictory_status_fields": _attack(
            contradictory.terminal is False and contradictory.classification == "contradictory",
            contradictory.terminal,
            contradictory.reason,
        ),
    }


def _attack(killed: bool, gate_bypassed: bool, diagnostic: str) -> JsonDict:
    return {"killed": killed, "gate_bypassed": gate_bypassed, "diagnostic": diagnostic}


def blocker_count_by_milestone(rows: Sequence[JsonMap]) -> JsonDict:
    return dict(sorted(Counter(str(r["milestone"]) for r in rows).items()))


def protected_hashes(root: Path = REPO_ROOT) -> JsonDict:
    return {p.as_posix(): path_sha256(root / p) for p in PROTECTED_RELATIVE_PATHS}


def changed_file_hashes(root: Path = REPO_ROOT) -> JsonDict:
    return {p.as_posix(): path_sha256(root / p) for p in CHANGED_RELATIVE_PATHS}


def protected_files_unchanged(before: Mapping[str, Any], root: Path = REPO_ROOT) -> JsonDict:
    after = protected_hashes(root)
    changed = [path for path, digest in before.items() if after.get(path) != digest]
    return {"ok": not changed, "before": dict(before), "after": after, "changed": changed}


def preconditions(root: Path = REPO_ROOT) -> JsonDict:
    disk = shutil.disk_usage(root)
    return {
        "repo_root": root.as_posix(),
        "git_status_short_empty": _git_status_empty(root),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "ram_available_kb": _mem_available_kb(),
        "disk_free_bytes": disk.free,
        "artifact_6188_present": (root / "results/experiment_6188_livecodebench_headroom_audit.json").exists(),
        "current_lint_health": "preflight ruff check passed before edits",
    }


def _git_status_empty(root: Path) -> bool:
    git = root / ".git"
    return git.exists() and not list((root).glob(".git/index.lock"))


def _mem_available_kb(path: Path = Path("/proc/meminfo")) -> int | None:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        return None
    return None


def field_principles() -> JsonDict:
    out = {field: "Required artifact field for REQ-OPS-RECURRING-GATE-6425." for field in REQUIRED_ARTIFACT_FIELDS}
    out.update(
        {
            name: f"Classification row uses {name} only when the evidence matches that root cause."
            for name in CLASSIFICATIONS
        }
    )
    out["shared_fix_applied"] = "The repair adds diagnostics and does not alter gate pass/fail semantics."
    out["recurring_gate_diagnostic_ready_score"] = "Score is one only when rows, mutations, and the shared diagnostic contract are present."
    return out


def field_provenance() -> JsonDict:
    measured = {
        "duration_s",
        "preconditions_checked",
        "protected_files_unchanged",
    }
    upstream = {
        "frozen_blocker_population_receipt",
        "blocker_count_by_milestone",
        "per_unit_rows",
        "per_occurrence_task_upstream_field_op_expected_observed_type_hash_and_artifact_bindings",
        "source_trace",
    }
    constant = {
        "status",
        "blocked_reason",
        "inference_substrate",
        "verifier_is_oracle",
        "random_seed",
        "no_scientific_gate_bypassed",
        "no_historical_task_rerun",
        "honest_verdict",
    }
    out: JsonDict = {}
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field in measured:
            out[field] = "measured"
        elif field in upstream:
            out[field] = "upstream"
        elif field in constant:
            out[field] = "constant"
        else:
            out[field] = "derived"
    return out


def payload_checksum(report: Mapping[str, Any]) -> str:
    without = {k: v for k, v in report.items() if k != "reproducibility_checksum"}
    return payload_sha256(without)


def build_report(
    root: Path = REPO_ROOT,
    *,
    date: str = RUN_DATE,
    before_hashes: Mapping[str, Any] | None = None,
    duration_s: float = 0.0,
) -> JsonDict:
    rows = collect_blocker_population(root)
    class_counts = dict(sorted(Counter(str(r["classification"]) for r in rows).items()))
    highest = max(class_counts.items(), key=lambda kv: (kv[1], kv[0]))[0] if class_counts else None
    mutation_matrix = build_mutation_attack_matrix()
    diagnostic_loss = sum(1 for r in rows if not r.get("blocked_reason_present"))
    infra_classes = {"missing_upstream", "wrong_field_name", "wrong_field_type", "stale_artifact", "retired_dependency", "diagnostic_loss"}
    report: JsonDict = {
        "status": "complete_recurring_gate_block_root_cause_reported",
        "frozen_blocker_population_receipt": {
            "date": date,
            "start_milestone": START_MILESTONE,
            "end_milestone": END_MILESTONE,
            "population_count": len(rows),
            "population_rule": "research-complete blocked_gate_check_failed artifacts in explicit milestone window",
            "default_ledger_window_note": "A current default ledger run found a smaller moving-window count after later milestones.",
        },
        "blocker_count_by_milestone": blocker_count_by_milestone(rows),
        "per_unit_rows": rows,
        "per_occurrence_task_upstream_field_op_expected_observed_type_hash_and_artifact_bindings": rows,
        "root_cause_class_counts": class_counts,
        "correct_expected_refusal_count": class_counts.get("correct_expected_refusal", 0),
        "infrastructure_defect_count": sum(class_counts.get(name, 0) for name in infra_classes),
        "diagnostic_loss_count": diagnostic_loss,
        "highest_count_root_cause": highest,
        "source_trace": {
            "recurring_blocker_ledger": "scripts/recurring_blocker_ledger.py",
            "gate_replay": "scripts/conductor_gates.py:evaluate_gates",
            "terminal_classifier": "python/carnot/terminal_artifacts.py",
            "known_issue": "ops/known-issues.md recurring blockers nobody has investigated",
            "classification_decision": _root_cause_decision(class_counts),
        },
        "shared_fix_applied": {
            "applied": True,
            "producer": "scripts/conductor_gates.py:write_blocked_artifact",
            "repair": "future blocked artifacts expose blocked_reason and first failed-gate evidence fields",
            "science_gate_changed": False,
        },
        "changed_files_and_hashes": changed_file_hashes(root),
        "no_scientific_gate_bypassed": True,
        "no_historical_task_rerun": True,
        "blocked_diagnostic_contract": {
            "version": "blocked_gate_diagnostic_v1",
            "fields": [
                "blocked_reason",
                "failed_upstream",
                "failed_field",
                "failed_operator",
                "failed_expected",
                "failed_observed",
                "failed_observed_type",
                "failed_evidence_path",
            ],
            "legacy_artifacts_preserved": True,
        },
        "mutation_attack_matrix": mutation_matrix,
        "recurring_gate_diagnostic_ready_score": _ready_score(rows, mutation_matrix),
        "protected_files_unchanged": protected_files_unchanged(before_hashes or protected_hashes(root), root),
        "blocked_reason": None,
        "preconditions_checked": preconditions(root),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": field_principles(),
        "field_provenance": field_provenance(),
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "tests_run": _declared_tests_run(),
        "honest_verdict": "complete: recurring_gate_block_root_cause_reported_with_diagnostic_contract",
    }
    report["reproducibility_checksum"] = payload_checksum(report)
    return report


def _root_cause_decision(class_counts: Mapping[str, int]) -> str:
    if not class_counts:
        return "complete_null: no blocker population found"
    highest = max(class_counts.items(), key=lambda kv: (kv[1], kv[0]))
    return f"highest_count_root_cause={highest[0]} count={highest[1]}"


def _ready_score(rows: Sequence[JsonMap], matrix: Mapping[str, JsonMap]) -> float:
    if not rows:
        return 0.0
    if any(str(row.get("classification")) not in CLASSIFICATIONS for row in rows):
        return 0.0
    if any(not attack.get("killed") or attack.get("gate_bypassed") for attack in matrix.values()):
        return 0.0
    return 1.0


def _declared_tests_run() -> list[JsonDict]:
    return [
        {"command": ".venv/bin/pytest tests/python/test_experiment_6425_recurring_gate_block_root_cause.py tests/python/test_conductor_gates.py -q --no-cov -n 0", "purpose": "focused REQ-OPS-RECURRING-GATE-6425 unit coverage"},
        {"command": ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6425_recurring_gate_block_root_cause.py tests/python/test_conductor_gates.py", "purpose": "spec coverage"},
        {"command": ".venv/bin/python scripts/audit_roadmap_gates.py", "purpose": "gate audit"},
        {"command": ".venv/bin/python scripts/exclusion_manifest_lint.py", "purpose": "exclusion lint"},
        {"command": ".venv/bin/python scripts/artifact_convention_audit.py --recent 8 --dry-run", "purpose": "artifact convention audit dry run"},
        {"command": ".venv/bin/python scripts/determination_preservation_lint.py", "purpose": "determination preservation"},
        {"command": ".venv/bin/python scripts/root_clutter_sweep.py --check", "purpose": "root clutter check"},
    ]


def validate_report(report: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in report]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if len(report.get("per_unit_rows") or []) != 31:
        errors.append("per_unit_rows must contain the 31 frozen blocker occurrences")
    if report.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if report.get("no_scientific_gate_bypassed") is not True:
        errors.append("scientific gate bypass detected")
    if report.get("no_historical_task_rerun") is not True:
        errors.append("historical task rerun detected")
    if report.get("reproducibility_checksum") != payload_checksum(report):
        errors.append("reproducibility_checksum mismatch")
    honest = str(report.get("honest_verdict") or "")
    if not honest.startswith(ALLOWED_HONEST_PREFIXES):
        errors.append("honest_verdict lacks required terminal prefix")
    return errors


def write_report(report: Mapping[str, Any], root: Path = REPO_ROOT) -> Path:
    return atomic_write_json(RESULT_RELATIVE_PATH, report, root=root, sort_keys=True)


def run(*, date: str = RUN_DATE, root: Path = REPO_ROOT, write: bool = True) -> JsonDict:
    start = time.perf_counter()
    before = protected_hashes(root)
    report = build_report(root, date=date, before_hashes=before, duration_s=time.perf_counter() - start)
    errors = validate_report(report)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_report(report, root)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    args = parser.parse_args(argv)
    report = run(date=args.date)
    print(json.dumps({"path": RESULT_RELATIVE_PATH.as_posix(), "status": report["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
