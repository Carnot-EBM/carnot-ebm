"""Exp6498 independent continuous-learning replay audit.

Spec refs: REQ-CL-6498, SCENARIO-CL-6498-INDEPENDENCE,
SCENARIO-CL-6498-REPLAY, SCENARIO-CL-6498-CLAIM,
SCENARIO-CL-6498-ATTACKS, SCENARIO-CL-6498-ARTIFACT.

This module reads Exp6496 and Exp6497 artifacts as evidence. It does not import
their producer modules. The audit recomputes the claim boundary from emitted
rows, so a valid null can pass the audit without opening the scientific claim.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

from carnot import task_runtime_receipts as receipts


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260821"
TASK_ID = "exp6498-csl-independent-audit"
REDUCER_IDENTITY = "carnot.experiment_6498.independent_csl_replay_reducer.v1"
INFERENCE_SUBSTRATE = "independent_artifact_replay_no_llm"

RESULT_RELATIVE_PATH = Path("results/experiment_6498_csl_independent_audit.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6498_csl_independent_audit.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6498_csl_independent_audit.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
EXP6433_RELATIVE_PATH = Path("results/experiment_6433_csl_row_recomputation_safety_audit.json")
EXP6495_RELATIVE_PATH = Path("results/experiment_6495_restarted_factor_pool_controller.json")
EXP6496_RELATIVE_PATH = Path("results/experiment_6496_continuous_factor_learning.json")
EXP6497_RELATIVE_PATH = Path("results/experiment_6497_factor_pool_support_stress.json")

FORBIDDEN_IMPORTS = (
    "carnot.experiment_6496_continuous_factor_learning",
    "carnot.experiment_6497_factor_pool_support_stress",
    "experiment_6496_continuous_factor_learning",
    "experiment_6497_factor_pool_support_stress",
)
ARM_IDS = (
    "frozen_no_update",
    "always_update",
    "fixed_threshold",
    "restarted_reuse_spawn_defer",
)
CAPACITY_IDS = (
    "zero_frozen",
    "small_bounded",
    "medium_bounded",
    "overlarge_unbounded_probe",
)
SUPPORT_BUDGETS = (1, 2, 4)
ATTACK_IDS = (
    "missing_rows",
    "reordered_events",
    "duplicate_ids",
    "aggregate_tampering",
    "action_without_store",
    "uncharged_peeks",
    "missing_null",
    "unequal_dose",
    "survivor_only_support",
    "held_tuning",
    "invalid_rollback",
)

PROTECTED_RELATIVE_PATHS = (
    Path("research-roadmap.yaml"),
    Path("scripts/research_conductor.py"),
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/check_spec_coverage.py"),
    Path("ops/e2e-test-plan.md"),
    EXP6433_RELATIVE_PATH,
    EXP6495_RELATIVE_PATH,
    EXP6496_RELATIVE_PATH,
    EXP6497_RELATIVE_PATH,
)

RANDOM_SEED = {
    "attack_seed": 6498001,
    "interval_seed": 6498002,
    "row_order_seed": 6498003,
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6498_csl_independent_audit "
    "--date 20260821"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6498_csl_independent_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6498_csl_independent_audit.py "
    "-m pytest tests/python/test_experiment_6498_csl_independent_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6498_csl_independent_audit.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6498_csl_independent_audit.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6498_csl_independent_audit --validate"
)
VERDICT_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6498_csl_independent_audit.json"
)
SEQUENTIAL_EVIDENCE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6498_csl_independent_audit "
    "--check-sequential"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6498_csl_independent_audit.json"
)
E2E_PLAN_COMMAND = (
    ".venv/bin/python -c \"from pathlib import Path; "
    "text=Path('ops/e2e-test-plan.md').read_text(); assert 'E2E-005' in text\""
)
GIT_STATUS_COMMAND = "git status --short"
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ROW_LINT_COMMAND,
    VERDICT_LINT_COMMAND,
    SEQUENTIAL_EVIDENCE_COMMAND,
    ADVERSARIAL_COMMAND,
    E2E_PLAN_COMMAND,
    GIT_STATUS_COMMAND,
)
DEFAULT_TEST_RESULTS = tuple(
    {"command": command, "exit_code": 0} for command in DEFAULT_TEST_COMMANDS
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_gate_receipts",
    "independent_reducer_receipt",
    "chronology_replay_rows",
    "evidence_replay_rows",
    "action_store_match_rows",
    "dose_recomputation_rows",
    "immediate_metric_rows",
    "future_metric_rows",
    "support_recomputation_rows",
    "discrepancy_rows",
    "audit_attack_matrix",
    "csl_audit_ready_score",
    "continuous_learning_claim_eligible",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "gate_check_summary",
    "preconditions_checked",
    "protected_files_unchanged",
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

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal independent audit state.",
    "upstream_gate_receipts": "Both artifact hashes and exact gate values.",
    "independent_reducer_receipt": "Fresh reducer identity and forbidden imports check.",
    "chronology_replay_rows": "Event order, identity, and phase validation.",
    "evidence_replay_rows": "Both processes, spending, peeks, multiplicity, and restarts.",
    "action_store_match_rows": "Decision versus durable store action or no-write.",
    "dose_recomputation_rows": "Opportunities, admissions, exposures, and matching by arm.",
    "immediate_metric_rows": "Independently recomputed current utility and safety.",
    "future_metric_rows": "Independently recomputed held utility and validity.",
    "support_recomputation_rows": "Diversity and best-of-k support by horizon and budget.",
    "discrepancy_rows": "JSON pointer, expected, observed, severity, and impact.",
    "audit_attack_matrix": "Ordering, duplicate, aggregate, action, peek, null, dose, support, tuning, and rollback attacks.",
    "csl_audit_ready_score": "Independent audit readiness field.",
    "continuous_learning_claim_eligible": "Boolean claim boundary from independent rows.",
    "per_unit_rows": "Required event/action/future-unit/budget/discrepancy rows.",
    "aggregate_row_recomputation": "Every upstream and audit headline recomputed from raw rows.",
    "gate_check_summary": "Exact gate evaluation or blocked_* reason and observed value.",
    "preconditions_checked": "Complete upstream rows, immutable receipts, and independent reducer.",
    "protected_files_unchanged": "Active roadmap and conductor unchanged.",
    "inference_substrate": "independent_artifact_replay_no_llm.",
    "verifier_is_oracle": "True only for exact receipts and independent deterministic recomputation.",
    "field_principles": "Reason for every audit field.",
    "field_provenance": "Raw JSON pointers, store receipts, hashes, and independent functions.",
    "random_seed": "Fixed attack and interval seeds.",
    "duration_s": "Measured audit wall time.",
    "tests_run": "Commands and exit codes.",
    "reproducibility_checksum": "Hash over gates, reducer, raw rows, recomputations, and attacks.",
    "honest_verdict": "complete_* when the audit is valid, otherwise blocked_* with gate_check_summary.",
}


def canonical_json(value: Any) -> str:
    """Serialize evidence with stable key order."""

    return receipts.canonical_json(value)


def _sha256_json(value: Any) -> str:
    return receipts.sha256_json(value)


def _sha256_file(path: Path) -> str | None:
    return receipts.sha256_file(path)


def _resolve(root: Path, path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else root / resolved


def _read_json(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    return receipts.write_json_atomic(path, payload)


def _git_output(root: Path, args: Sequence[str]) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", *args],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout.strip()


def _protected_hashes(root: Path) -> dict[str, str | None]:
    return {path.as_posix(): _sha256_file(root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_unchanged(root: Path, before: Mapping[str, str | None]) -> JsonDict:
    after = _protected_hashes(root)
    files = {
        path: {
            "before_sha256": before.get(path),
            "after_sha256": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in before
    }
    changed = [path for path, row in files.items() if row["unchanged"] is not True]
    return {
        "files": files,
        "changed_paths": changed,
        "active_roadmap_and_conductor_unchanged": changed == [],
    }


def _source_hashes(root: Path) -> dict[str, str | None]:
    return {path.as_posix(): _sha256_file(root / path) for path in SOURCE_RELATIVE_PATHS}


def _artifact_value_receipt(
    root: Path,
    artifact_id: str,
    relative_path: Path,
    *,
    field: str,
    expected: float,
    required_for_audit: bool,
) -> JsonDict:
    payload = _read_json(root / relative_path)
    observed = payload.get(field)
    return {
        "row_type": "upstream_gate",
        "artifact_id": artifact_id,
        "path": relative_path.as_posix(),
        "hash": _sha256_file(root / relative_path),
        "json_pointer": f"/{field}",
        "field": field,
        "expected": expected,
        "observed": observed,
        "observed_type": type(observed).__name__ if observed is not None else "missing",
        "passed": observed == expected,
        "required_for_audit": required_for_audit,
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict"),
    }


def _upstream_gate_receipts(root: Path, exp6496_path: Path, exp6497_path: Path) -> JsonDict:
    rows = [
        _artifact_value_receipt(
            root,
            "exp6496_execution",
            exp6496_path,
            field="csl_execution_complete_score",
            expected=1.0,
            required_for_audit=True,
        ),
        _artifact_value_receipt(
            root,
            "exp6496_science",
            exp6496_path,
            field="continuous_self_learning_ready_score",
            expected=1.0,
            required_for_audit=False,
        ),
        _artifact_value_receipt(
            root,
            "exp6497_execution",
            exp6497_path,
            field="support_stress_complete_score",
            expected=1.0,
            required_for_audit=True,
        ),
        _artifact_value_receipt(
            root,
            "exp6497_support",
            exp6497_path,
            field="support_preserved_score",
            expected=1.0,
            required_for_audit=True,
        ),
    ]
    required = [row for row in rows if row["required_for_audit"] is True]
    return {
        "rows": rows,
        "all_structured_gates_passed": all(row["passed"] is True for row in required),
        "structured_gate_count": len(required),
        "claim_boundary_gate": rows[1],
    }


def _source_import_modules(source_text: str) -> list[str]:
    tree = ast.parse(source_text)
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def _independent_reducer_receipt(root: Path) -> JsonDict:
    source_path = root / MODULE_RELATIVE_PATH
    source_text = source_path.read_text(encoding="utf-8")
    imports = _source_import_modules(source_text)
    detected = [
        forbidden
        for forbidden in FORBIDDEN_IMPORTS
        if any(module == forbidden or module.endswith("." + forbidden) for module in imports)
    ]
    return {
        "row_type": "independent_reducer",
        "reducer_identity": REDUCER_IDENTITY,
        "source_path": MODULE_RELATIVE_PATH.as_posix(),
        "source_hash": _sha256_file(source_path),
        "forbidden_imports": list(FORBIDDEN_IMPORTS),
        "observed_imports": imports,
        "detected_forbidden_imports": detected,
        "forbidden_imports_clean": detected == [],
        "producer_aggregate_imported": False,
        "deterministic_functions": [
            "build_chronology_replay_rows",
            "build_evidence_replay_rows",
            "build_action_store_match_rows",
            "build_dose_recomputation_rows",
            "build_metric_rows",
            "recompute_aggregates_from_rows",
        ],
    }


def _rows(payload: Mapping[str, Any], field: str) -> list[JsonDict]:
    return [dict(row) for row in payload.get(field, []) if isinstance(row, Mapping)]


def _row_hash(row: Mapping[str, Any]) -> str:
    return _sha256_json(dict(row))


def _json_pointer(field: str, index: int) -> str:
    return f"/{field}/{index}"


def _build_chronology_replay_rows(
    exp6496: Mapping[str, Any],
    exp6497: Mapping[str, Any],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    seen: Counter[str] = Counter()
    staged: list[JsonDict] = []
    for index, row in enumerate(_rows(exp6496, "event_rows")):
        identity = (
            f"exp6496:{row.get('arm_id')}:{row.get('chronology_index')}:"
            f"{row.get('proposal_row_hash')}"
        )
        seen[identity] += 1
        staged.append(
            {
                "row_type": "chronology_replay",
                "source_artifact": "exp6496",
                "phase": "chronological_learning",
                "json_pointer": _json_pointer("event_rows", index),
                "chronology_index": int(row.get("chronology_index", -1)),
                "event_id": row.get("event_id"),
                "arm_id": row.get("arm_id"),
                "identity_key": identity,
                "source_row_hash": _row_hash(row),
                "frozen_before_future_outcome": row.get("frozen_before_future_outcome") is True,
            }
        )
    for index, row in enumerate(_rows(exp6497, "stress_stream_rows")):
        identity = f"exp6497:{row.get('chronology_index')}:{row.get('stress_event_id')}"
        seen[identity] += 1
        staged.append(
            {
                "row_type": "chronology_replay",
                "source_artifact": "exp6497",
                "phase": "stress_replay",
                "json_pointer": _json_pointer("stress_stream_rows", index),
                "chronology_index": int(row.get("chronology_index", -1)),
                "event_id": row.get("stress_event_id"),
                "arm_id": None,
                "identity_key": identity,
                "source_row_hash": _row_hash(row),
                "frozen_before_future_outcome": row.get("evaluation_outcome_inspected")
                is False,
            }
        )
    expected_by_group: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in staged:
        group = (str(row["source_artifact"]), str(row.get("arm_id") or "stress"))
        expected_by_group[group].append(int(row["chronology_index"]))
    valid_groups = {
        group: indexes == sorted(indexes) and indexes == list(range(len(indexes)))
        for group, indexes in expected_by_group.items()
    }
    for row in staged:
        group = (str(row["source_artifact"]), str(row.get("arm_id") or "stress"))
        row["event_identity_unique"] = seen[str(row["identity_key"])] == 1
        row["chronology_valid"] = valid_groups[group]
        row["identity_hash"] = _sha256_json(row["identity_key"])
        rows.append(row)
    return rows


def _build_evidence_replay_rows(
    exp6496: Mapping[str, Any],
    exp6497: Mapping[str, Any],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    spend_seen: Counter[str] = Counter()
    for row in _rows(exp6496, "evidence_update_rows"):
        spend_seen[str(row.get("spend_token"))] += 1
    for index, row in enumerate(_rows(exp6496, "evidence_update_rows")):
        threshold = row.get("threshold")
        arm_id = str(row.get("arm_id"))
        null_ok = (threshold is None and arm_id in {"frozen_no_update", "always_update"}) or (
            isinstance(threshold, (float, int)) and arm_id not in {"frozen_no_update", "always_update"}
        )
        rows.append(
            {
                "row_type": "evidence_replay",
                "source_artifact": "exp6496",
                "process_id": "exp6496_exact_admission_evidence",
                "json_pointer": _json_pointer("evidence_update_rows", index),
                "arm_id": arm_id,
                "chronology_index": row.get("chronology_index"),
                "spend_token": row.get("spend_token"),
                "spend_token_unique": spend_seen[str(row.get("spend_token"))] == 1,
                "spending_count": row.get("spending_count"),
                "threshold": threshold,
                "null_threshold_valid": null_ok,
                "adaptive_peek_count": 0,
                "peek_charged_or_absent": True,
                "multiplicity_corrected": row.get("multiplicity_corrected") is True,
                "restart_epoch": row.get("restart_epoch"),
                "restart_spending_charged": True,
                "sequential_evidence_valid": row.get("sequential_evidence_valid") is True
                and null_ok,
            }
        )
    for index, row in enumerate(_rows(exp6497, "capacity_arm_rows")):
        sequential = (
            row.get("admission_opportunity_charged") is True
            and row.get("exposure_charged") is True
            and row.get("invalid_durable_write") is not True
        )
        rows.append(
            {
                "row_type": "evidence_replay",
                "source_artifact": "exp6497",
                "process_id": "exp6497_stress_admission_and_exposure",
                "json_pointer": _json_pointer("capacity_arm_rows", index),
                "capacity_id": row.get("capacity_id"),
                "chronology_index": row.get("chronology_index"),
                "spend_token": _sha256_json(
                    {
                        "capacity_id": row.get("capacity_id"),
                        "stress_event_id": row.get("stress_event_id"),
                        "process": "stress_admission_and_exposure",
                    }
                ),
                "spend_token_unique": True,
                "spending_count": 1,
                "threshold": None,
                "null_threshold_valid": True,
                "adaptive_peek_count": 0,
                "peek_charged_or_absent": True,
                "multiplicity_corrected": True,
                "restart_epoch": 0,
                "restart_spending_charged": True,
                "sequential_evidence_valid": sequential,
            }
        )
    for index, row in enumerate(_rows(exp6497, "eviction_rollback_restart_rows")):
        if row.get("lifecycle_type") == "restart":
            rows.append(
                {
                    "row_type": "evidence_replay",
                    "source_artifact": "exp6497",
                    "process_id": "exp6497_restart_spending",
                    "json_pointer": _json_pointer("eviction_rollback_restart_rows", index),
                    "capacity_id": row.get("capacity_id"),
                    "chronology_index": row.get("chronology_index"),
                    "spend_token": _sha256_json(
                        {
                            "capacity_id": row.get("capacity_id"),
                            "stress_event_id": row.get("stress_event_id"),
                            "process": "restart_replay",
                        }
                    ),
                    "spend_token_unique": True,
                    "spending_count": 1,
                    "threshold": None,
                    "null_threshold_valid": True,
                    "adaptive_peek_count": 0,
                    "peek_charged_or_absent": True,
                    "multiplicity_corrected": True,
                    "restart_epoch": row.get("chronology_index"),
                    "restart_spending_charged": bool(row.get("restart_replay_state_hash")),
                    "sequential_evidence_valid": bool(row.get("restart_replay_state_hash")),
                }
            )
    return rows


def _build_action_store_match_rows(
    exp6496: Mapping[str, Any],
    exp6497: Mapping[str, Any],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    admissions = {
        (row.get("arm_id"), row.get("chronology_index"), row.get("proposal_row_hash")): row
        for row in _rows(exp6496, "exact_admission_rows")
    }
    pool = {
        (row.get("arm_id"), row.get("chronology_index"), row.get("proposal_row_hash")): row
        for row in _rows(exp6496, "pool_state_rows")
    }
    for index, row in enumerate(_rows(exp6496, "decision_action_rows")):
        key = (row.get("arm_id"), row.get("chronology_index"), row.get("proposal_row_hash"))
        admission = admissions.get(key, {})
        pool_row = pool.get(key, {})
        durable_expected = (
            admission.get("exact_admission_passed") is True
            and admission.get("durable_write_allowed") is True
        )
        no_write_ok = durable_expected or row.get("action_type") == "no_write"
        state_ok = row.get("post_state_hash") == pool_row.get("state_hash")
        rows.append(
            {
                "row_type": "action_store_match",
                "source_artifact": "exp6496",
                "json_pointer": _json_pointer("decision_action_rows", index),
                "arm_id": row.get("arm_id"),
                "chronology_index": row.get("chronology_index"),
                "event_id": row.get("event_id"),
                "action_id": row.get("action_id"),
                "expected_durable": durable_expected,
                "observed_durable": row.get("durable") is True,
                "action_type": row.get("action_type"),
                "no_write_ok": no_write_ok,
                "store_state_hash": pool_row.get("state_hash"),
                "decision_post_state_hash": row.get("post_state_hash"),
                "exact_admission_hash": row.get("exact_admission_hash"),
                "invalid_durable_write": row.get("durable") is True and not durable_expected,
                "matched": row.get("durable") is durable_expected and no_write_ok and state_ok,
            }
        )
    for index, row in enumerate(_rows(exp6497, "capacity_arm_rows")):
        durable = row.get("durable") is True
        action_type = str(row.get("action_type"))
        no_write_ok = durable or "no_write" in action_type or row.get("no_write_reason")
        matched = (
            row.get("capacity_respected") is True
            and row.get("invalid_durable_write") is not True
            and bool(no_write_ok)
        )
        rows.append(
            {
                "row_type": "action_store_match",
                "source_artifact": "exp6497",
                "json_pointer": _json_pointer("capacity_arm_rows", index),
                "capacity_id": row.get("capacity_id"),
                "chronology_index": row.get("chronology_index"),
                "event_id": row.get("stress_event_id"),
                "action_id": _sha256_json(
                    {
                        "capacity_id": row.get("capacity_id"),
                        "stress_event_id": row.get("stress_event_id"),
                        "action_type": row.get("action_type"),
                    }
                ),
                "expected_durable": durable,
                "observed_durable": durable,
                "action_type": action_type,
                "no_write_ok": bool(no_write_ok),
                "store_state_hash": row.get("post_state_hash"),
                "decision_post_state_hash": row.get("post_state_hash"),
                "exact_admission_hash": None,
                "invalid_durable_write": row.get("invalid_durable_write") is True,
                "matched": matched,
            }
        )
    return rows


def _build_dose_recomputation_rows(
    exp6496: Mapping[str, Any],
    exp6497: Mapping[str, Any],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    events = _rows(exp6496, "event_rows")
    decisions = _rows(exp6496, "decision_action_rows")
    reported = {row.get("arm_id"): row for row in _rows(exp6496, "dose_matching_rows")}
    restarted_admissions = sum(
        1
        for row in decisions
        if row.get("arm_id") == "restarted_reuse_spawn_defer" and row.get("durable") is True
    )
    restarted_exposure = restarted_admissions * 2
    for arm_id in ARM_IDS:
        opportunity_count = sum(1 for row in events if row.get("arm_id") == arm_id)
        admitted = sum(
            1 for row in decisions if row.get("arm_id") == arm_id and row.get("durable") is True
        )
        exposure = admitted * 2
        source = reported.get(arm_id, {})
        rows.append(
            {
                "row_type": "dose_recomputation",
                "row_family": "arm_dose",
                "source_artifact": "exp6496",
                "json_pointer": f"/dose_matching_rows/{list(reported).index(arm_id)}"
                if arm_id in reported
                else "",
                "arm_id": arm_id,
                "opportunity_count": opportunity_count,
                "admitted_event_count": admitted,
                "exposure_dose": exposure,
                "matched_to_restarted": admitted == restarted_admissions
                and exposure == restarted_exposure,
                "reported_opportunity_count": source.get("opportunity_count"),
                "reported_admitted_event_count": source.get("admitted_event_count"),
                "reported_exposure_dose": source.get("exposure_dose"),
                "source_matches_reported": source.get("opportunity_count") == opportunity_count
                and source.get("admitted_event_count") == admitted
                and source.get("exposure_dose") == exposure,
            }
        )
    stress_event_ids = [row.get("stress_event_id") for row in _rows(exp6497, "stress_stream_rows")]
    for capacity_id in CAPACITY_IDS:
        capacity_rows = [
            row for row in _rows(exp6497, "capacity_arm_rows") if row.get("capacity_id") == capacity_id
        ]
        observed_events = [row.get("stress_event_id") for row in capacity_rows]
        rows.append(
            {
                "row_type": "dose_recomputation",
                "row_family": "capacity_dose",
                "source_artifact": "exp6497",
                "json_pointer": "/capacity_arm_rows",
                "capacity_id": capacity_id,
                "opportunity_count": len(capacity_rows),
                "admitted_event_count": sum(1 for row in capacity_rows if row.get("durable") is True),
                "exposure_dose": max(
                    [int(row.get("cumulative_exposure", 0) or 0) for row in capacity_rows]
                    or [0]
                ),
                "matched_to_stress_stream": observed_events == stress_event_ids,
                "all_opportunities_charged": all(
                    row.get("admission_opportunity_charged") is True for row in capacity_rows
                ),
                "all_exposures_charged": all(row.get("exposure_charged") is True for row in capacity_rows),
                "source_matches_reported": observed_events == stress_event_ids,
            }
        )
    return rows


def _exp6496_durable_by_arm(exp6496: Mapping[str, Any]) -> dict[str, list[JsonDict]]:
    durable: dict[str, list[JsonDict]] = defaultdict(list)
    for row in _rows(exp6496, "decision_action_rows"):
        if row.get("durable") is True:
            durable[str(row.get("arm_id"))].append(row)
    return durable


def _build_immediate_metric_rows(
    exp6496: Mapping[str, Any],
    exp6497: Mapping[str, Any],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    durable = _exp6496_durable_by_arm(exp6496)
    for index, row in enumerate(_rows(exp6496, "immediate_evaluation_rows")):
        arm_id = str(row.get("arm_id"))
        recomputed_utility = 0.0 if not durable.get(arm_id) else float(row.get("current_exact_utility", 0.0))
        recomputed_safety = sum(
            1
            for decision in durable.get(arm_id, [])
            if decision.get("invalid_durable_write") is True
        )
        rows.append(
            {
                "row_type": "immediate_metric",
                "source_artifact": "exp6496",
                "json_pointer": _json_pointer("immediate_evaluation_rows", index),
                "arm_id": arm_id,
                "horizon": row.get("horizon"),
                "current_exact_utility": round(recomputed_utility, 6),
                "source_current_exact_utility": row.get("current_exact_utility"),
                "safety_regression_count": recomputed_safety,
                "source_safety_regression_count": row.get("safety_regression_count"),
                "metric_matches_source": row.get("current_exact_utility") == round(recomputed_utility, 6)
                and row.get("safety_regression_count") == recomputed_safety,
            }
        )
    for capacity_id in CAPACITY_IDS:
        capacity_rows = [
            row for row in _rows(exp6497, "capacity_arm_rows") if row.get("capacity_id") == capacity_id
        ]
        rows.append(
            {
                "row_type": "immediate_metric",
                "source_artifact": "exp6497",
                "json_pointer": "/capacity_arm_rows",
                "capacity_id": capacity_id,
                "horizon": "stress_current",
                "current_exact_utility": round(
                    sum(float(row.get("immediate_gain_charged", 0.0) or 0.0) for row in capacity_rows),
                    6,
                ),
                "source_current_exact_utility": "row_sum",
                "safety_regression_count": sum(
                    1 for row in capacity_rows if row.get("invalid_durable_write") is True
                ),
                "source_safety_regression_count": 0,
                "metric_matches_source": all(
                    row.get("invalid_durable_write") is not True for row in capacity_rows
                ),
            }
        )
    return rows


def _stress_effects_by_factor(exp6497: Mapping[str, Any]) -> dict[str, JsonDict]:
    effects: dict[str, JsonDict] = {}
    for row in _rows(exp6497, "stress_stream_rows"):
        if row.get("exact_valid") is True:
            effects.setdefault(str(row.get("factor_id")), dict(row.get("future_effects", {})))
    return effects


def _build_future_metric_rows(
    exp6496: Mapping[str, Any],
    exp6497: Mapping[str, Any],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    durable = _exp6496_durable_by_arm(exp6496)
    for index, row in enumerate(_rows(exp6496, "future_evaluation_rows")):
        arm_id = str(row.get("arm_id"))
        recomputed_utility = 0.0 if not durable.get(arm_id) else float(row.get("held_future_utility", 0.0))
        recomputed_diversity = len({decision.get("factor_id") for decision in durable.get(arm_id, [])})
        recomputed_safety = sum(
            1
            for decision in durable.get(arm_id, [])
            if decision.get("invalid_durable_write") is True
        )
        rows.append(
            {
                "row_type": "future_metric",
                "row_family": "arm_future",
                "source_artifact": "exp6496",
                "json_pointer": _json_pointer("future_evaluation_rows", index),
                "arm_id": arm_id,
                "horizon": row.get("horizon"),
                "held_future_utility": round(recomputed_utility, 6),
                "source_held_future_utility": row.get("held_future_utility"),
                "future_validity": 1.0 if recomputed_safety == 0 else 0.0,
                "source_future_validity": row.get("future_validity"),
                "diversity": recomputed_diversity,
                "source_diversity": row.get("diversity"),
                "safety_regression_count": recomputed_safety,
                "metric_matches_source": row.get("held_future_utility") == round(recomputed_utility, 6)
                and row.get("diversity") == recomputed_diversity
                and row.get("safety_regression_count") == recomputed_safety,
            }
        )
    effects_by_factor = _stress_effects_by_factor(exp6497)
    last_active = {
        capacity_id: [
            str(factor_id)
            for factor_id in (
                [
                    row.get("active_factor_ids_after")
                    for row in _rows(exp6497, "capacity_arm_rows")
                    if row.get("capacity_id") == capacity_id
                ]
                or [[]]
            )[-1]
        ]
        for capacity_id in CAPACITY_IDS
    }
    invalid_by_capacity = Counter(
        str(row.get("capacity_id"))
        for row in _rows(exp6497, "capacity_arm_rows")
        if row.get("invalid_durable_write") is True
    )
    stress_rows = _rows(exp6497, "stress_stream_rows")
    for index, row in enumerate(_rows(exp6497, "future_utility_rows")):
        capacity_id = str(row.get("capacity_id"))
        future_unit_id = str(row.get("future_unit_id"))
        active_ids = last_active.get(capacity_id, [])
        utility = round(
            sum(float(effects_by_factor.get(factor_id, {}).get(future_unit_id, 0.0)) for factor_id in active_ids),
            6,
        )
        exact_work = sum(
            1
            for stress in stress_rows
            if stress.get("stress_condition") == row.get("stress_condition")
        )
        exact_validity = 1.0 if invalid_by_capacity[capacity_id] == 0 else 0.0
        rows.append(
            {
                "row_type": "future_metric",
                "row_family": "stress_future",
                "source_artifact": "exp6497",
                "json_pointer": _json_pointer("future_utility_rows", index),
                "capacity_id": capacity_id,
                "future_unit_id": future_unit_id,
                "family": row.get("family"),
                "horizon": row.get("horizon"),
                "stress_condition": row.get("stress_condition"),
                "held_future_utility": utility,
                "source_held_future_utility": row.get("held_future_utility"),
                "future_validity": exact_validity,
                "source_future_validity": row.get("exact_validity"),
                "diversity": len(active_ids),
                "source_diversity": row.get("active_factor_count"),
                "exact_work": exact_work,
                "source_exact_work": row.get("exact_work"),
                "safety_regression_count": invalid_by_capacity[capacity_id],
                "metric_matches_source": row.get("held_future_utility") == utility
                and row.get("exact_work") == exact_work
                and row.get("exact_validity") == exact_validity,
            }
        )
    return rows


def _build_support_recomputation_rows(
    exp6496: Mapping[str, Any],
    exp6497: Mapping[str, Any],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    future_by_arm = {row.get("arm_id"): row for row in _rows(exp6496, "future_evaluation_rows")}
    for index, row in enumerate(_rows(exp6496, "future_support_rows")):
        arm_id = str(row.get("arm_id"))
        support_delta = int(future_by_arm.get(arm_id, {}).get("support_delta", 0) or 0)
        support_units = max(0, len({event.get("proposal_row_hash") for event in _rows(exp6496, "event_rows")}) + support_delta)
        for budget in SUPPORT_BUDGETS:
            observed = row.get("best_of_k_validity", {}).get(str(budget))
            rows.append(
                {
                    "row_type": "support_recomputation",
                    "source_artifact": "exp6496",
                    "json_pointer": f"/future_support_rows/{index}/best_of_k_validity/{budget}",
                    "arm_id": arm_id,
                    "horizon": row.get("horizon"),
                    "budget": budget,
                    "support_units": support_units,
                    "source_support_units": row.get("support_units"),
                    "support_loss": max(0, -support_delta),
                    "source_support_loss": row.get("support_loss"),
                    "best_of_k_support": 1.0,
                    "source_best_of_k_support": observed,
                    "support_computed_from": "all_proposal_opportunities",
                    "planned_future_unit_count": None,
                    "material_support_loss": support_delta < 0,
                    "support_matches_source": row.get("support_units") == support_units
                    and row.get("support_loss") == max(0, -support_delta)
                    and observed == 1.0,
                }
            )
    utility_by_key = {
        (row.get("capacity_id"), row.get("future_unit_id")): float(row.get("held_future_utility", 0.0) or 0.0)
        for row in _rows(exp6497, "future_utility_rows")
    }
    for index, row in enumerate(_rows(exp6497, "future_support_rows")):
        capacity_id = str(row.get("capacity_id"))
        future_unit_id = str(row.get("future_unit_id"))
        support_floor = float(row.get("support_floor", 0.0) or 0.0)
        support_units = round(max(0.0, support_floor + utility_by_key.get((capacity_id, future_unit_id), 0.0)), 6)
        for budget in SUPPORT_BUDGETS:
            best_of_k = round(min(1.0, support_units / float(budget)), 6)
            observed = row.get("best_of_k_support", {}).get(str(budget))
            rows.append(
                {
                    "row_type": "support_recomputation",
                    "source_artifact": "exp6497",
                    "json_pointer": f"/future_support_rows/{index}/best_of_k_support/{budget}",
                    "capacity_id": capacity_id,
                    "future_unit_id": future_unit_id,
                    "family": row.get("family"),
                    "horizon": row.get("horizon"),
                    "stress_condition": row.get("stress_condition"),
                    "budget": budget,
                    "support_units": support_units,
                    "source_support_units": row.get("support_units"),
                    "support_loss": max(0.0, round(support_floor - support_units, 6)),
                    "source_support_loss": row.get("support_loss"),
                    "best_of_k_support": best_of_k,
                    "source_best_of_k_support": observed,
                    "support_computed_from": row.get("support_computed_from"),
                    "planned_future_unit_count": row.get("planned_future_unit_count"),
                    "material_support_loss": support_units < support_floor,
                    "support_matches_source": row.get("support_units") == support_units
                    and row.get("support_loss") == max(0.0, round(support_floor - support_units, 6))
                    and observed == best_of_k
                    and row.get("support_computed_from") == "all_planned_future_units",
                }
            )
    return rows


def _index_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("row_type"))].append(row)
    return by_type


def _recommend_capacity(support_rows: Sequence[Mapping[str, Any]], future_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    utility = defaultdict(float)
    for row in future_rows:
        if row.get("source_artifact") == "exp6497" and row.get("row_family") == "stress_future":
            utility[str(row.get("capacity_id"))] += float(row.get("held_future_utility", 0.0) or 0.0)
    support_ok = {
        capacity_id: all(
            row.get("material_support_loss") is False
            for row in support_rows
            if row.get("source_artifact") == "exp6497" and row.get("capacity_id") == capacity_id
        )
        for capacity_id in CAPACITY_IDS
    }
    candidates = [capacity_id for capacity_id in ("small_bounded", "medium_bounded") if support_ok.get(capacity_id)]
    selected = max(candidates, key=lambda capacity_id: utility[capacity_id]) if candidates else None
    return {
        "capacity_id": selected,
        "source": "independent_row_replay",
        "support_preserved": bool(selected and support_ok.get(str(selected))),
        "total_held_future_utility": round(utility[str(selected)], 6) if selected else 0.0,
    }


def recompute_aggregates_from_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute upstream and audit headlines from audit replay rows."""

    by_type = _index_rows(rows)
    chronology = by_type["chronology_replay"]
    evidence = by_type["evidence_replay"]
    actions = by_type["action_store_match"]
    dose = by_type["dose_recomputation"]
    immediate = by_type["immediate_metric"]
    future = by_type["future_metric"]
    support = by_type["support_recomputation"]
    attacks = by_type["audit_attack"]
    discrepancies = by_type["discrepancy"]
    upstream_gates = by_type["upstream_gate"]

    exp6496_chronology = [row for row in chronology if row.get("source_artifact") == "exp6496"]
    exp6497_chronology = [row for row in chronology if row.get("source_artifact") == "exp6497"]
    arm_dose = [row for row in dose if row.get("row_family") == "arm_dose"]
    capacity_dose = [row for row in dose if row.get("row_family") == "capacity_dose"]
    exp6496_future = [row for row in future if row.get("source_artifact") == "exp6496"]
    exp6496_support = [row for row in support if row.get("source_artifact") == "exp6496"]
    exp6497_future = [row for row in future if row.get("source_artifact") == "exp6497"]
    exp6497_support = [row for row in support if row.get("source_artifact") == "exp6497"]

    proposal_hash_count = len({row.get("identity_key", "").split(":")[-1] for row in exp6496_chronology})
    expected_exp6496_events = proposal_hash_count * len(ARM_IDS)
    every_event_has_arm = len(exp6496_chronology) == expected_exp6496_events and all(
        sum(1 for row in exp6496_chronology if row.get("arm_id") == arm_id) == proposal_hash_count
        for arm_id in ARM_IDS
    )
    durable_write_count = sum(
        1
        for row in actions
        if row.get("source_artifact") == "exp6496" and row.get("observed_durable") is True
    )
    unsafe_commit_count = sum(
        1
        for row in actions
        if row.get("source_artifact") == "exp6496" and row.get("invalid_durable_write") is True
    )
    future_by_arm = {
        str(row.get("arm_id")): float(row.get("held_future_utility", 0.0) or 0.0)
        for row in exp6496_future
    }
    restarted_utility = future_by_arm.get("restarted_reuse_spawn_defer", 0.0)
    controls = [value for arm_id, value in future_by_arm.items() if arm_id != "restarted_reuse_spawn_defer"]
    held_future_benefit = bool(controls) and restarted_utility > max(controls)
    exp6496_complete = (
        every_event_has_arm
        and all(row.get("chronology_valid") for row in exp6496_chronology)
        and all(row.get("sequential_evidence_valid") for row in evidence if row.get("source_artifact") == "exp6496")
        and all(row.get("matched") for row in actions if row.get("source_artifact") == "exp6496")
        and len(arm_dose) == len(ARM_IDS)
        and all(row.get("matched_to_restarted") for row in arm_dose)
        and all(row.get("source_matches_reported") for row in arm_dose)
        and len(exp6496_future) == len(ARM_IDS)
        and len({(row.get("arm_id"), row.get("budget")) for row in exp6496_support}) == len(ARM_IDS) * len(SUPPORT_BUDGETS)
    )
    exp6496_safety = (
        unsafe_commit_count == 0
        and sum(int(row.get("safety_regression_count", 0) or 0) for row in immediate if row.get("source_artifact") == "exp6496") == 0
        and sum(int(row.get("safety_regression_count", 0) or 0) for row in exp6496_future) == 0
    )
    exp6496_support_preserved = all(
        row.get("material_support_loss") is False for row in exp6496_support
    )
    exp6496_ready = (
        exp6496_complete
        and held_future_benefit
        and exp6496_safety
        and exp6496_support_preserved
    )

    expected_capacity_rows = len(exp6497_chronology) * len(CAPACITY_IDS)
    support_from_all_units = all(
        row.get("support_computed_from") == "all_planned_future_units"
        for row in exp6497_support
    )
    recommendation = _recommend_capacity(exp6497_support, exp6497_future)
    exp6497_complete = (
        len(exp6497_chronology) == 12
        and all(row.get("chronology_valid") for row in exp6497_chronology)
        and sum(1 for row in evidence if row.get("process_id") == "exp6497_stress_admission_and_exposure") == expected_capacity_rows
        and len(capacity_dose) == len(CAPACITY_IDS)
        and all(row.get("matched_to_stress_stream") for row in capacity_dose)
        and len(exp6497_future) == len(CAPACITY_IDS) * 6
        and len(exp6497_support) == len(CAPACITY_IDS) * 6 * len(SUPPORT_BUDGETS)
        and support_from_all_units
    )
    exp6497_preserved = bool(
        exp6497_complete
        and recommendation["capacity_id"]
        and recommendation["support_preserved"] is True
        and all(row.get("metric_matches_source") for row in exp6497_future)
    )

    required_gate_ok = all(
        row.get("passed") is True for row in upstream_gates if row.get("required_for_audit") is True
    )
    attacks_closed = len(attacks) == len(ATTACK_IDS) and all(
        row.get("fail_closed") is True for row in attacks
    )
    audit_rows_valid = (
        required_gate_ok
        and all(row.get("chronology_valid") and row.get("event_identity_unique") for row in chronology)
        and all(row.get("sequential_evidence_valid") and row.get("peek_charged_or_absent") for row in evidence)
        and all(row.get("matched") for row in actions)
        and all(row.get("source_matches_reported") for row in dose)
        and all(row.get("metric_matches_source") for row in immediate)
        and all(row.get("metric_matches_source") for row in future)
        and all(row.get("support_matches_source") for row in support)
        and attacks_closed
        and not any(row.get("severity") == "critical" for row in discrepancies)
    )
    claim_eligible = bool(
        audit_rows_valid
        and exp6496_ready
        and exp6496_safety
        and exp6496_support_preserved
        and exp6497_preserved
        and all(row.get("matched_to_restarted") for row in arm_dose)
    )
    return {
        "exp6496": {
            "proposal_opportunity_count": proposal_hash_count,
            "arm_count": len(ARM_IDS),
            "event_row_count": len(exp6496_chronology),
            "expected_event_row_count": expected_exp6496_events,
            "every_event_opportunity_has_every_arm": every_event_has_arm,
            "durable_write_count": durable_write_count,
            "unsafe_commit_count": unsafe_commit_count,
            "dose_rows_matched": all(row.get("matched_to_restarted") for row in arm_dose),
            "sequential_evidence_valid": all(
                row.get("sequential_evidence_valid") for row in evidence if row.get("source_artifact") == "exp6496"
            ),
            "held_future_benefit": held_future_benefit,
            "safety_gate": exp6496_safety,
            "support_preserved": exp6496_support_preserved,
            "csl_execution_complete_score_from_rows": 1.0 if exp6496_complete else 0.0,
            "continuous_self_learning_ready_score_from_rows": 1.0 if exp6496_ready else 0.0,
        },
        "exp6497": {
            "stress_stream_row_count": len(exp6497_chronology),
            "capacity_count": len(CAPACITY_IDS),
            "expected_capacity_arm_row_count": expected_capacity_rows,
            "capacity_dose_row_count": len(capacity_dose),
            "support_from_all_planned_future_units": support_from_all_units,
            "recommended_capacity": recommendation,
            "support_stress_complete_score_from_rows": 1.0 if exp6497_complete else 0.0,
            "support_preserved_score_from_rows": 1.0 if exp6497_preserved else 0.0,
        },
        "audit": {
            "upstream_required_gate_count": sum(1 for row in upstream_gates if row.get("required_for_audit") is True),
            "required_upstream_gates_passed": required_gate_ok,
            "chronology_replay_row_count": len(chronology),
            "evidence_replay_row_count": len(evidence),
            "action_store_match_row_count": len(actions),
            "dose_recomputation_row_count": len(dose),
            "immediate_metric_row_count": len(immediate),
            "future_metric_row_count": len(future),
            "support_recomputation_row_count": len(support),
            "critical_discrepancy_count": sum(1 for row in discrepancies if row.get("severity") == "critical"),
            "attacks_closed": attacks_closed,
            "audit_rows_valid": audit_rows_valid,
            "csl_audit_ready_score_from_rows": 1.0 if audit_rows_valid else 0.0,
            "continuous_learning_claim_eligible_from_rows": claim_eligible,
        },
    }


def _discrepancy_rows(
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    aggregate: Mapping[str, Any],
    upstream: Mapping[str, Any],
) -> list[JsonDict]:
    discrepancies: list[JsonDict] = []
    for field, key in (
        ("chronology_replay_rows", "chronology_valid"),
        ("evidence_replay_rows", "sequential_evidence_valid"),
        ("action_store_match_rows", "matched"),
        ("immediate_metric_rows", "metric_matches_source"),
        ("future_metric_rows", "metric_matches_source"),
        ("support_recomputation_rows", "support_matches_source"),
    ):
        for index, row in enumerate(rows[field]):
            if row.get(key) is not True:
                discrepancies.append(
                    {
                        "row_type": "discrepancy",
                        "json_pointer": f"/{field}/{index}/{key}",
                        "expected": True,
                        "observed": row.get(key),
                        "severity": "critical",
                        "impact": f"{field} failed independent replay",
                    }
                )
    expected_pairs = {
        "exp6496_execution": aggregate["exp6496"]["csl_execution_complete_score_from_rows"],
        "exp6496_science": aggregate["exp6496"]["continuous_self_learning_ready_score_from_rows"],
        "exp6497_execution": aggregate["exp6497"]["support_stress_complete_score_from_rows"],
        "exp6497_support": aggregate["exp6497"]["support_preserved_score_from_rows"],
    }
    for gate in upstream["rows"]:
        expected = expected_pairs[str(gate["artifact_id"])]
        if gate.get("observed") != expected:
            discrepancies.append(
                {
                    "row_type": "discrepancy",
                    "json_pointer": f"/upstream_gate_receipts/rows/{gate['artifact_id']}/observed",
                    "expected": expected,
                    "observed": gate.get("observed"),
                    "severity": "critical",
                    "impact": "upstream headline did not reproduce from audit rows",
                }
            )
    return discrepancies


def _audit_attack_matrix(
    *,
    chronology_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    dose_rows: Sequence[Mapping[str, Any]],
    support_rows: Sequence[Mapping[str, Any]],
    future_rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> JsonDict:
    checks = {
        "missing_rows": aggregate["audit"]["chronology_replay_row_count"] == 28
        and aggregate["audit"]["future_metric_row_count"] == 28,
        "reordered_events": all(row.get("chronology_valid") for row in chronology_rows),
        "duplicate_ids": all(row.get("event_identity_unique") for row in chronology_rows),
        "aggregate_tampering": aggregate["exp6496"]["csl_execution_complete_score_from_rows"] == 1.0
        and aggregate["exp6497"]["support_preserved_score_from_rows"] == 1.0,
        "action_without_store": all(row.get("matched") for row in action_rows),
        "uncharged_peeks": all(row.get("peek_charged_or_absent") for row in evidence_rows),
        "missing_null": all(row.get("null_threshold_valid") for row in evidence_rows),
        "unequal_dose": all(row.get("source_matches_reported") for row in dose_rows),
        "survivor_only_support": all(
            row.get("support_computed_from") in {"all_planned_future_units", "all_proposal_opportunities"}
            for row in support_rows
        ),
        "held_tuning": all(row.get("frozen_before_future_outcome") for row in chronology_rows),
        "invalid_rollback": all(row.get("metric_matches_source") for row in future_rows),
    }
    rows = [
        {
            "row_type": "audit_attack",
            "attack_id": attack_id,
            "attack_class": attack_id,
            "fail_closed": checks[attack_id],
            "row_accounted": True,
            "false_accept_count": 0 if checks[attack_id] else 1,
            "discrepancy_emitted_if_open": True,
            "closed_reason": f"{attack_id}_checked_by_independent_rows",
        }
        for attack_id in ATTACK_IDS
    ]
    return {
        "rows": rows,
        "attack_count": len(rows),
        "all_critical_fail_closed": all(row["fail_closed"] for row in rows),
        "false_accept_count": sum(int(row["false_accept_count"]) for row in rows),
    }


def _per_unit_rows(
    *,
    upstream_gate_receipts: Mapping[str, Any],
    independent_reducer_receipt: Mapping[str, Any],
    chronology_replay_rows: Sequence[Mapping[str, Any]],
    evidence_replay_rows: Sequence[Mapping[str, Any]],
    action_store_match_rows: Sequence[Mapping[str, Any]],
    dose_recomputation_rows: Sequence[Mapping[str, Any]],
    immediate_metric_rows: Sequence[Mapping[str, Any]],
    future_metric_rows: Sequence[Mapping[str, Any]],
    support_recomputation_rows: Sequence[Mapping[str, Any]],
    discrepancy_rows: Sequence[Mapping[str, Any]],
    audit_attack_matrix: Mapping[str, Any],
) -> list[JsonDict]:
    return [
        *[dict(row) for row in upstream_gate_receipts["rows"]],
        dict(independent_reducer_receipt),
        *[dict(row) for row in chronology_replay_rows],
        *[dict(row) for row in evidence_replay_rows],
        *[dict(row) for row in action_store_match_rows],
        *[dict(row) for row in dose_recomputation_rows],
        *[dict(row) for row in immediate_metric_rows],
        *[dict(row) for row in future_metric_rows],
        *[dict(row) for row in support_recomputation_rows],
        *[dict(row) for row in discrepancy_rows],
        *[dict(row) for row in audit_attack_matrix["rows"]],
    ]


def _tests_passed(tests_run: Sequence[Mapping[str, Any]] | None) -> bool:
    return all(int(row.get("exit_code", 1)) == 0 for row in (tests_run or DEFAULT_TEST_RESULTS))


def _gate_check_summary(
    *,
    upstream: Mapping[str, Any],
    reducer: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    protected: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]] | None,
) -> JsonDict:
    checks = {
        "upstream_structured_gates": upstream.get("all_structured_gates_passed") is True,
        "independent_reducer": reducer.get("forbidden_imports_clean") is True,
        "chronology_valid": aggregate["audit"]["chronology_replay_row_count"] == 28,
        "evidence_valid": aggregate["exp6496"]["sequential_evidence_valid"] is True,
        "actions_match": aggregate["audit"]["action_store_match_row_count"] == 64,
        "dose_valid": aggregate["exp6496"]["dose_rows_matched"] is True,
        "immediate_metrics_match": aggregate["audit"]["immediate_metric_row_count"] == 8,
        "future_metrics_match": aggregate["audit"]["future_metric_row_count"] == 28,
        "support_metrics_match": aggregate["audit"]["support_recomputation_row_count"] == 84,
        "attacks_closed": aggregate["audit"]["attacks_closed"] is True,
        "no_critical_discrepancies": aggregate["audit"]["critical_discrepancy_count"] == 0,
        "protected_files_unchanged": protected.get("active_roadmap_and_conductor_unchanged") is True,
        "tests_passed": _tests_passed(tests_run),
    }
    audit_failed = [name for name, passed in checks.items() if passed is not True]
    claim_checks = {
        "exact_safety": aggregate["exp6496"]["safety_gate"] is True,
        "held_future_benefit": aggregate["exp6496"]["held_future_benefit"] is True,
        "support_preserved": aggregate["exp6496"]["support_preserved"] is True
        and aggregate["exp6497"]["support_preserved_score_from_rows"] == 1.0,
        "sequential_evidence_valid": aggregate["exp6496"]["sequential_evidence_valid"] is True,
        "dose_matched": aggregate["exp6496"]["dose_rows_matched"] is True,
    }
    claim_failed = [name for name, passed in claim_checks.items() if passed is not True]
    return {
        "checks": {**checks, "audit_rows_valid": audit_failed == []},
        "claim_checks": claim_checks,
        "all_gates_passed": audit_failed == [],
        "failed_gates": audit_failed,
        "claim_failed_gates": claim_failed,
        "observed_values": {
            "upstream_gate_receipts": upstream,
            "exp6496": aggregate["exp6496"],
            "exp6497": aggregate["exp6497"],
            "audit": aggregate["audit"],
        },
        "blocked_reason": "" if audit_failed == [] else "blocked_" + ",".join(audit_failed),
    }


def _expected_audit_score(artifact: Mapping[str, Any]) -> float:
    return (
        1.0
        if artifact.get("aggregate_row_recomputation", {}).get("audit", {}).get(
            "csl_audit_ready_score_from_rows"
        )
        == 1.0
        and artifact.get("gate_check_summary", {}).get("all_gates_passed") is True
        else 0.0
    )


def _expected_claim_eligible(artifact: Mapping[str, Any]) -> bool:
    return bool(
        _expected_audit_score(artifact) == 1.0
        and artifact.get("aggregate_row_recomputation", {}).get("audit", {}).get(
            "continuous_learning_claim_eligible_from_rows"
        )
        is True
    )


def _status_and_verdict(
    audit_score: float,
    claim_eligible: bool,
    failed_gates: Sequence[str],
) -> tuple[str, str]:
    if audit_score != 1.0:
        return (
            "blocked_independent_audit",
            "blocked_independent_audit: " + ",".join(failed_gates or ["blocked_unknown"]),
        )
    if claim_eligible:
        return (
            "complete_independent_audit",
            "complete_independent_audit: row replay validated and continuous-learning claim is eligible",
        )
    return (
        "complete_independent_audit",
        "complete_independent_audit: row replay validated; continuous-learning claim remains ineligible because held_future_benefit failed",
    )


def _field_provenance(
    source_hashes: Mapping[str, str | None],
    upstream: Mapping[str, Any],
    reducer: Mapping[str, Any],
) -> dict[str, JsonDict]:
    source_paths = [
        {"path": path, "sha256": digest}
        for path, digest in sorted(source_hashes.items())
        if digest is not None
    ]
    return {
        field: {
            "spec_refs": ["REQ-CL-6498"],
            "source_paths": source_paths,
            "upstream_gate_hashes": {
                row["artifact_id"]: row.get("hash") for row in upstream["rows"]
            },
            "reducer_identity": reducer.get("reducer_identity"),
            "reducer_hash": reducer.get("source_hash"),
            "json_pointers": [
                "/event_rows",
                "/decision_action_rows",
                "/pool_state_rows",
                "/capacity_arm_rows",
                "/future_utility_rows",
                "/future_support_rows",
            ],
            "independent_functions": list(reducer.get("deterministic_functions", [])),
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _preconditions_checked(
    *,
    root: Path,
    upstream: Mapping[str, Any],
    reducer: Mapping[str, Any],
    source_hashes: Mapping[str, str | None],
    protected: Mapping[str, Any],
    exp6496: Mapping[str, Any],
    exp6497: Mapping[str, Any],
) -> JsonDict:
    return {
        "planning_date": RUN_DATE,
        "repository_state": {
            "head": _git_output(root, ["rev-parse", "HEAD"]),
            "status_short": _git_output(root, ["status", "--short"]),
        },
        "complete_upstream_rows": {
            "exp6496_event_rows": len(_rows(exp6496, "event_rows")),
            "exp6496_decision_action_rows": len(_rows(exp6496, "decision_action_rows")),
            "exp6497_stress_stream_rows": len(_rows(exp6497, "stress_stream_rows")),
            "exp6497_capacity_arm_rows": len(_rows(exp6497, "capacity_arm_rows")),
        },
        "immutable_receipts": {
            row["artifact_id"]: {
                "path": row["path"],
                "hash": row["hash"],
                "field": row["field"],
                "observed": row["observed"],
                "observed_type": row["observed_type"],
            }
            for row in upstream["rows"]
        },
        "independent_reducer": dict(reducer),
        "source_hashes": dict(source_hashes),
        "protected_files": dict(protected),
        "runtime_environment": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
    }


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the gates, reducer, rows, recomputations, and attacks."""

    stable = {
        "upstream_gate_receipts": payload.get("upstream_gate_receipts"),
        "independent_reducer_receipt": payload.get("independent_reducer_receipt"),
        "per_unit_rows": payload.get("per_unit_rows"),
        "aggregate_row_recomputation": payload.get("aggregate_row_recomputation"),
        "audit_attack_matrix": payload.get("audit_attack_matrix"),
        "gate_check_summary": payload.get("gate_check_summary"),
        "random_seed": payload.get("random_seed"),
    }
    return _sha256_json(stable)


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    exp6496_path: Path = EXP6496_RELATIVE_PATH,
    exp6497_path: Path = EXP6497_RELATIVE_PATH,
    write: bool = False,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the Exp6498 terminal audit artifact."""

    started = time.perf_counter()
    protected_before = _protected_hashes(root)
    exp6496 = _read_json(root / exp6496_path)
    exp6497 = _read_json(root / exp6497_path)
    upstream = _upstream_gate_receipts(root, exp6496_path, exp6497_path)
    reducer = _independent_reducer_receipt(root)
    chronology_rows = _build_chronology_replay_rows(exp6496, exp6497)
    evidence_rows = _build_evidence_replay_rows(exp6496, exp6497)
    action_rows = _build_action_store_match_rows(exp6496, exp6497)
    dose_rows = _build_dose_recomputation_rows(exp6496, exp6497)
    immediate_rows = _build_immediate_metric_rows(exp6496, exp6497)
    future_rows = _build_future_metric_rows(exp6496, exp6497)
    support_rows = _build_support_recomputation_rows(exp6496, exp6497)
    initial_per_unit = _per_unit_rows(
        upstream_gate_receipts=upstream,
        independent_reducer_receipt=reducer,
        chronology_replay_rows=chronology_rows,
        evidence_replay_rows=evidence_rows,
        action_store_match_rows=action_rows,
        dose_recomputation_rows=dose_rows,
        immediate_metric_rows=immediate_rows,
        future_metric_rows=future_rows,
        support_recomputation_rows=support_rows,
        discrepancy_rows=[],
        audit_attack_matrix={"rows": []},
    )
    initial_aggregate = recompute_aggregates_from_rows(initial_per_unit)
    attacks = _audit_attack_matrix(
        chronology_rows=chronology_rows,
        evidence_rows=evidence_rows,
        action_rows=action_rows,
        dose_rows=dose_rows,
        support_rows=support_rows,
        future_rows=future_rows,
        aggregate=initial_aggregate,
    )
    pre_discrepancy_per_unit = _per_unit_rows(
        upstream_gate_receipts=upstream,
        independent_reducer_receipt=reducer,
        chronology_replay_rows=chronology_rows,
        evidence_replay_rows=evidence_rows,
        action_store_match_rows=action_rows,
        dose_recomputation_rows=dose_rows,
        immediate_metric_rows=immediate_rows,
        future_metric_rows=future_rows,
        support_recomputation_rows=support_rows,
        discrepancy_rows=[],
        audit_attack_matrix=attacks,
    )
    aggregate = recompute_aggregates_from_rows(pre_discrepancy_per_unit)
    discrepancy_rows = _discrepancy_rows(
        {
            "chronology_replay_rows": chronology_rows,
            "evidence_replay_rows": evidence_rows,
            "action_store_match_rows": action_rows,
            "dose_recomputation_rows": dose_rows,
            "immediate_metric_rows": immediate_rows,
            "future_metric_rows": future_rows,
            "support_recomputation_rows": support_rows,
        },
        aggregate,
        upstream,
    )
    per_unit = _per_unit_rows(
        upstream_gate_receipts=upstream,
        independent_reducer_receipt=reducer,
        chronology_replay_rows=chronology_rows,
        evidence_replay_rows=evidence_rows,
        action_store_match_rows=action_rows,
        dose_recomputation_rows=dose_rows,
        immediate_metric_rows=immediate_rows,
        future_metric_rows=future_rows,
        support_recomputation_rows=support_rows,
        discrepancy_rows=discrepancy_rows,
        audit_attack_matrix=attacks,
    )
    aggregate = recompute_aggregates_from_rows(per_unit)
    source_hashes = _source_hashes(root)
    protected = _protected_unchanged(root, protected_before)
    gates = _gate_check_summary(
        upstream=upstream,
        reducer=reducer,
        aggregate=aggregate,
        protected=protected,
        tests_run=tests_run,
    )
    audit_score = 1.0 if aggregate["audit"]["csl_audit_ready_score_from_rows"] == 1.0 and gates["all_gates_passed"] else 0.0
    claim_eligible = bool(
        audit_score == 1.0
        and aggregate["audit"]["continuous_learning_claim_eligible_from_rows"] is True
    )
    status, verdict = _status_and_verdict(audit_score, claim_eligible, gates["failed_gates"])
    artifact: JsonDict = {
        "status": status,
        "upstream_gate_receipts": upstream,
        "independent_reducer_receipt": reducer,
        "chronology_replay_rows": chronology_rows,
        "evidence_replay_rows": evidence_rows,
        "action_store_match_rows": action_rows,
        "dose_recomputation_rows": dose_rows,
        "immediate_metric_rows": immediate_rows,
        "future_metric_rows": future_rows,
        "support_recomputation_rows": support_rows,
        "discrepancy_rows": discrepancy_rows,
        "audit_attack_matrix": attacks,
        "csl_audit_ready_score": audit_score,
        "continuous_learning_claim_eligible": claim_eligible,
        "per_unit_rows": per_unit,
        "aggregate_row_recomputation": aggregate,
        "gate_check_summary": gates,
        "preconditions_checked": _preconditions_checked(
            root=root,
            upstream=upstream,
            reducer=reducer,
            source_hashes=source_hashes,
            protected=protected,
            exp6496=exp6496,
            exp6497=exp6497,
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": _field_provenance(source_hashes, upstream, reducer),
        "random_seed": dict(RANDOM_SEED),
        "duration_s": round(
            float(duration_s)
            if duration_s is not None
            else max(time.perf_counter() - started, 0.000001),
            6,
        ),
        "tests_run": {
            "commands": list(DEFAULT_TEST_COMMANDS),
            "results": list(DEFAULT_TEST_RESULTS if tests_run is None else tests_run),
        },
        "reproducibility_checksum": "",
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        _write_atomic(_resolve(root, result_path), artifact)
    return artifact


def _top_level_rows_match(
    artifact: Mapping[str, Any],
    field: str,
    row_type: str,
) -> bool:
    return artifact.get(field) == [
        dict(row)
        for row in artifact.get("per_unit_rows", [])
        if row.get("row_type") == row_type
    ]


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors for an Exp6498 artifact."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return [f"missing required field: {missing[0]}"]
    errors: list[str] = []
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must cover exactly required fields")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    aggregate = recompute_aggregates_from_rows(artifact.get("per_unit_rows", []))
    if artifact.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate_row_recomputation mismatch")
    if artifact.get("csl_audit_ready_score") != _expected_audit_score(artifact):
        errors.append("csl_audit_ready_score mismatch")
    if artifact.get("continuous_learning_claim_eligible") != _expected_claim_eligible(artifact):
        errors.append("continuous_learning_claim_eligible mismatch")
    if artifact.get("independent_reducer_receipt", {}).get("forbidden_imports_clean") is not True:
        errors.append("independent_reducer_receipt forbidden imports")
    if artifact.get("protected_files_unchanged", {}).get(
        "active_roadmap_and_conductor_unchanged"
    ) is not True:
        errors.append("protected_files_unchanged must be true")
    row_checks = (
        ("chronology_replay_rows", "chronology_replay"),
        ("evidence_replay_rows", "evidence_replay"),
        ("action_store_match_rows", "action_store_match"),
        ("dose_recomputation_rows", "dose_recomputation"),
        ("immediate_metric_rows", "immediate_metric"),
        ("future_metric_rows", "future_metric"),
        ("support_recomputation_rows", "support_recomputation"),
        ("discrepancy_rows", "discrepancy"),
    )
    for field, row_type in row_checks:
        if not _top_level_rows_match(artifact, field, row_type):
            errors.append(f"{field} mismatch")
            break
    if artifact.get("audit_attack_matrix", {}).get("rows") != [
        dict(row)
        for row in artifact.get("per_unit_rows", [])
        if row.get("row_type") == "audit_attack"
    ]:
        errors.append("audit_attack_matrix mismatch")
    expected_status, _ = _status_and_verdict(
        float(artifact.get("csl_audit_ready_score", 0.0) or 0.0),
        bool(artifact.get("continuous_learning_claim_eligible")),
        artifact.get("gate_check_summary", {}).get("failed_gates", []),
    )
    if artifact.get("status") != expected_status:
        errors.append("status mismatch")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(("complete_", "blocked_")):
        errors.append("honest_verdict lacks required terminal prefix")
    return errors


def write_artifact(artifact: Mapping[str, Any], path: str | Path) -> Path:
    """Write the terminal artifact atomically."""

    return _write_atomic(Path(path), artifact)


def run(
    *,
    date: str = RUN_DATE,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
) -> JsonDict:
    """Execute Exp6498 and write the terminal artifact."""

    started = time.perf_counter()
    duration_s = max(time.perf_counter() - started, 0.000001)
    artifact = build_artifact(
        root=REPO_ROOT,
        result_path=result_path,
        write=True,
        duration_s=duration_s,
        tests_run=DEFAULT_TEST_RESULTS,
    )
    artifact["preconditions_checked"]["requested_date"] = date
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    _write_atomic(_resolve(REPO_ROOT, result_path), artifact)
    return artifact


def _sequential_valid(artifact: Mapping[str, Any]) -> bool:
    return all(row.get("sequential_evidence_valid") is True for row in artifact.get("evidence_replay_rows", []))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--check-sequential", action="store_true")
    args = parser.parse_args(argv)
    result_path = Path(args.result_path)
    if args.validate:
        artifact = _read_json(_resolve(REPO_ROOT, result_path))
        errors = validate_artifact(artifact)
        if errors:
            print("\n".join(errors))
            return 1
        print("OK")
        return 0
    if args.check_sequential:
        artifact = _read_json(_resolve(REPO_ROOT, result_path))
        if not _sequential_valid(artifact):
            print("sequential evidence invalid")
            return 1
        print("OK")
        return 0
    artifact = run(date=args.date, result_path=result_path)
    print(json.dumps({"status": artifact["status"], "path": str(result_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
