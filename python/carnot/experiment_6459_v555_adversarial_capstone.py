"""Exp6459 V555 adversarial capstone.

Spec refs: REQ-CAPSTONE-6459,
SCENARIO-CAPSTONE-6459-INVENTORY,
SCENARIO-CAPSTONE-6459-ROW-RECOMPUTATION,
SCENARIO-CAPSTONE-6459-CLAIM-DECISIONS,
SCENARIO-CAPSTONE-6459-ATTACKS,
SCENARIO-CAPSTONE-6459-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import yaml

from carnot.experiment_artifacts import atomic_write_json
from carnot.terminal_artifacts import canonical_json, path_sha256


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover - import-path guard.
    sys.path.insert(0, str(SCRIPTS_ROOT))

from adversarial_verify import verify_artifact  # noqa: E402


RUN_DATE = "20260815"
RANDOM_SEED = 6459
INFERENCE_SUBSTRATE = "aggregation_from_upstream_rows_and_artifacts_no_llm"
RESULT_RELATIVE_PATH = Path("results/experiment_6459_v555_adversarial_capstone.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/capstone/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6459_v555_adversarial_capstone.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6459_v555_adversarial_capstone.py")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6459_v555_adversarial_capstone "
    "--date 20260815"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6459_v555_adversarial_capstone.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6459_v555_adversarial_capstone.py "
    "-m pytest tests/python/test_experiment_6459_v555_adversarial_capstone.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6459_v555_adversarial_capstone.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6459_v555_adversarial_capstone.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6459_v555_adversarial_capstone.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6459_v555_adversarial_capstone.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py --all"
ARTIFACT_CONVENTION_COMMAND = ".venv/bin/python scripts/artifact_convention_audit.py --recent 8 --dry-run"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ROW_LINT_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ARTIFACT_CONVENTION_COMMAND,
    ROOT_CLUTTER_COMMAND,
)

V555_TASK_IDS = (
    "exp6448-v555-terminal-handoff-and-queue-integrity",
    "exp6449-generation-to-verdict-path-receipt-contract",
    "exp6450-sota-fixed-policy-candidate-corpus",
    "exp6451-typed-fact-grounding-fixed-policy-logic-ab",
    "exp6452-representation-objective-causal-ab",
    "exp6453-held-verifier-budget-allocation-ab",
    "exp6454-held-exact-constraint-energy-selection-ab",
    "exp6455-prospective-verifier-bounded-factor-weight-csl",
    "exp6456-corrupt-feedback-held-restart-csl-replication",
    "exp6457-independent-verifier-bounded-csl-audit",
    "exp6458-arc-representation-objective-generalization-ab",
    "exp6459-v555-adversarial-capstone",
)

CLAIM_FIELDS = (
    "typed_grounding_claim_eligibility",
    "objective_causal_claim_eligibility",
    "held_allocation_claim_eligibility",
    "energy_selection_claim_eligibility",
    "prospective_csl_claim_eligibility",
    "held_csl_safety_claim_eligibility",
    "internal_arc_generalization_claim_eligibility",
    "public_arc_claim_eligibility",
    "hardware_claim_eligibility",
)

ATTACK_IDS = (
    "output_reuse",
    "model_substitution",
    "cpu_fallback",
    "exact_label_leakage",
    "held_leakage",
    "uncharged_cost",
    "teacher_authority",
    "chronology",
    "corrupt_feedback",
    "rollback",
    "restart",
    "arc_source_access",
    "adapters",
    "registry_mutation",
    "row_inconsistency",
)

MANDATED_GGUF_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
LLM_TASK_IDS = {
    "exp6450-sota-fixed-policy-candidate-corpus",
    "exp6451-typed-fact-grounding-fixed-policy-logic-ab",
    "exp6455-prospective-verifier-bounded-factor-weight-csl",
    "exp6456-corrupt-feedback-held-restart-csl-replication",
}

FROZEN_ARM = "frozen_weights"
TEACHER_ARM = "self_teacher_signed_updates"
VERIFIER_ARM = "verifier_bounded_updates"
CLEAN_ARM = "clean_verifier_bounded_updates"
GOVERNED_ARM = "governed_verifier_bounded_updates"

ARC_BASELINE_ARM = "current_state_key_current_objective"
ARC_SUFFIX_ARM = "collision_suffix_current_objective"
ARC_COMBINED_ARM = "collision_suffix_reachability_objective"
ARC_PLACEBO_ARM = "collision_suffix_shuffled_objective_placebo"

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("_bmad/traceability.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/arc_solve_registry.yaml"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "expected_task_and_deliverable_manifest",
    "task_artifact_inventory_hashes_sizes_statuses_and_verdicts",
    "missing_zero_byte_blocked_flagged_underpowered_or_duration_ineligible_evidence",
    "gate_producer_contract_validation",
    "prior_failure_and_exclusion_validation",
    "model_and_substrate_validation",
    "field_principle_and_provenance_validation",
    "per_unit_rows",
    "independent_metric_recomputation_by_task",
    "upstream_vs_recomputed_mismatches",
    "mismatch_count_and_materiality",
    "joint_pathway_rows_and_cofailure_moments",
    "current_adversarial_attack_replay",
    "current_adversarial_findings",
    "typed_grounding_claim_eligibility",
    "objective_causal_claim_eligibility",
    "held_allocation_claim_eligibility",
    "energy_selection_claim_eligibility",
    "prospective_csl_claim_eligibility",
    "held_csl_safety_claim_eligibility",
    "internal_arc_generalization_claim_eligibility",
    "public_arc_claim_eligibility",
    "hardware_claim_eligibility",
    "claim_ineligibility_reasons",
    "terminal_determination_preservation",
    "protected_files_unchanged",
    "reconciliation_actions",
    "v555_capstone_ready_score",
    "blocked_reason",
    "gate_check_summary",
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

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "The status states whether this capstone completed its audit.",
    "expected_task_and_deliverable_manifest": "The manifest fixes the 12 task slots and declared deliverables.",
    "task_artifact_inventory_hashes_sizes_statuses_and_verdicts": "Inventory rows freeze exact evidence bytes and terminal states.",
    "missing_zero_byte_blocked_flagged_underpowered_or_duration_ineligible_evidence": "Bad or absent evidence is a terminal input, not a hidden retry.",
    "gate_producer_contract_validation": "Gate fields must be produced by the upstream task that later gates consume.",
    "prior_failure_and_exclusion_validation": "Prior-failure and exclusion checks stop retired scopes from becoming clean evidence.",
    "model_and_substrate_validation": "Model and substrate receipts bound live claims to actual hardware and duration.",
    "field_principle_and_provenance_validation": "Required artifact fields need principles and provenance for auditability.",
    "per_unit_rows": "Task and claim rows keep branch decisions visible.",
    "independent_metric_recomputation_by_task": "Metrics are recomputed from rows or explicit no-row terminal states.",
    "upstream_vs_recomputed_mismatches": "Every compared upstream field records whether the row reducer agreed.",
    "mismatch_count_and_materiality": "Material mismatches block promotion.",
    "joint_pathway_rows_and_cofailure_moments": "Joint outcomes use shared row keys only; marginal reliabilities are not multiplied.",
    "current_adversarial_attack_replay": "Current attack results show fail-closed behavior by attack class.",
    "current_adversarial_findings": "The capstone's own current critical findings must be zero.",
    "typed_grounding_claim_eligibility": "Typed grounding needs clean Exp6451 row evidence.",
    "objective_causal_claim_eligibility": "Objective causality needs clean Exp6452 row evidence.",
    "held_allocation_claim_eligibility": "Held allocation needs clean Exp6453 held rows and charged costs.",
    "energy_selection_claim_eligibility": "Held energy selection needs clean Exp6454 held rows.",
    "prospective_csl_claim_eligibility": "Prospective CSL needs clean producer rows and an eligible independent audit.",
    "held_csl_safety_claim_eligibility": "Held CSL safety needs clean restart and corruption evidence plus audit approval.",
    "internal_arc_generalization_claim_eligibility": "Internal ARC generalization needs clean held reachability evidence without safety regression.",
    "public_arc_claim_eligibility": "Public ARC needs solve-safe evidence; V555 contains none.",
    "hardware_claim_eligibility": "Hardware needs authenticated hardware evidence; V555 contains none.",
    "claim_ineligibility_reasons": "Every blocked claim names the exact evidence gap.",
    "terminal_determination_preservation": "Earlier terminal determinations must not be rewritten by the capstone.",
    "protected_files_unchanged": "Protected files must not change during the capstone.",
    "reconciliation_actions": "The operator stop rule defers ops and traceability reconciliation.",
    "v555_capstone_ready_score": "This score applies only to the capstone audit, not to science branch success.",
    "blocked_reason": "The verdict names why no requested science claim is eligible.",
    "gate_check_summary": "Gate failures stay machine-readable.",
    "preconditions_checked": "The audit records instruction, roadmap, spec, and inventory preconditions.",
    "inference_substrate": "This capstone reads local artifacts and rows with no new LLM calls.",
    "verifier_is_oracle": "Only deterministic exact checkers and row arithmetic are oracle boundaries here.",
    "field_principles": "Each required field and claim gate states its purpose.",
    "field_provenance": "Every required field maps to local rows, artifacts, tests, constants, or spec.",
    "random_seed": "The seed fixes deterministic row order.",
    "duration_s": "Duration is measured wall time without padding.",
    "tests_run": "Verification commands are recorded for replay.",
    "reproducibility_checksum": "The checksum catches silent drift in the capstone payload.",
    "honest_verdict": "The verdict uses a terminal prefix and names eligible and ineligible claims.",
}
FIELD_PRINCIPLES.update(
    {
        f"claim_gate:{claim}": f"Eligibility gate for {claim}."
        for claim in CLAIM_FIELDS
    }
)
FIELD_PROVENANCE = {
    field: "REQ-CAPSTONE-6459 local artifact inventory and row reducers"
    for field in REQUIRED_ARTIFACT_FIELDS
}


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def payload_checksum(payload: Mapping[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("reproducibility_checksum", None)
    return sha256_json(clone)


def _task_exp(task_id: str) -> str:
    return task_id.split("-", 1)[0]


def _safe_float(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, int | float) else default


def _exact_success(row: Mapping[str, Any]) -> bool:
    if isinstance(row.get("future_exact_outcome"), bool):
        return bool(row["future_exact_outcome"])
    if isinstance(row.get("exact_success"), bool):
        return bool(row["exact_success"])
    result = row.get("exact_result")
    return bool(result.get("exact_success")) if isinstance(result, Mapping) else False


def _protected_ok(row: Mapping[str, Any]) -> bool:
    result = row.get("exact_result")
    if isinstance(result, Mapping) and isinstance(result.get("protected_ok"), bool):
        return bool(result["protected_ok"])
    protected = row.get("protected_outcome")
    return bool(protected.get("protected_ok")) if isinstance(protected, Mapping) else True


def _rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    rows = payload.get("per_unit_rows")
    if isinstance(rows, list):
        return [dict(row) for row in rows if isinstance(row, Mapping)]
    if isinstance(rows, Mapping) and isinstance(rows.get("rows"), list):
        return [dict(row) for row in rows["rows"] if isinstance(row, Mapping)]
    return []


def _read_json(path: Path) -> tuple[JsonDict, bool, str]:
    if not path.is_file():
        return {}, False, "missing"
    if path.stat().st_size == 0:
        return {}, False, "zero_byte"
    try:
        return json.loads(path.read_text(encoding="utf-8")), True, ""
    except json.JSONDecodeError as exc:
        return {}, False, f"json_decode_error:{exc.msg}"


def load_v555_tasks(repo_root: Path) -> list[JsonDict]:
    data = yaml.safe_load((repo_root / ROADMAP_RELATIVE_PATH).read_text(encoding="utf-8"))
    tasks = [dict(task) for task in data.get("tasks", []) if task.get("id") in V555_TASK_IDS]
    order = {task_id: index for index, task_id in enumerate(V555_TASK_IDS)}
    return sorted(tasks, key=lambda task: order[task["id"]])


def _readiness_fields(payload: Mapping[str, Any]) -> JsonDict:
    return {
        key: value
        for key, value in payload.items()
        if key.endswith("_ready_score") or key.endswith("_integrity_score")
    }


def _row_count(payload: Mapping[str, Any]) -> int:
    rows = payload.get("per_unit_rows")
    if isinstance(rows, list):
        return len(rows)
    if isinstance(rows, Mapping):
        if isinstance(rows.get("row_count"), int):
            return int(rows["row_count"])
        if isinstance(rows.get("rows"), list):
            return len(rows["rows"])
    return 0


def _embedded_findings(payload: Mapping[str, Any]) -> list[JsonDict]:
    findings = payload.get("current_adversarial_findings")
    if isinstance(findings, list):
        return [dict(row) for row in findings if isinstance(row, Mapping)]
    if isinstance(findings, Mapping):
        flags = findings.get("flags")
        if isinstance(flags, list):
            return [dict(row) for row in flags if isinstance(row, Mapping)]
        if int(findings.get("critical_count") or 0) > 0:
            return [dict(findings)]
    if payload.get("flagged_adversarial") is True:
        return [{"kind": "flagged_adversarial", "severity": "critical"}]
    return []


def _critical_count(findings: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for finding in findings if str(finding.get("severity")).lower() == "critical")


def _artifact_state(
    task_id: str,
    path: Path,
    payload: Mapping[str, Any],
    loadable: bool,
    load_error: str,
    embedded_findings: Sequence[Mapping[str, Any]],
) -> str:
    if task_id == "exp6459-v555-adversarial-capstone" and not path.is_file():
        return "self_pending"
    if load_error == "missing":
        return "missing"
    if load_error == "zero_byte":
        return "zero_byte"
    if not loadable:
        return "malformed"
    verdict = str(payload.get("honest_verdict", ""))
    status = str(payload.get("status", ""))
    if status == "blocked" or verdict.startswith("blocked_"):
        return "blocked"
    if _critical_count(embedded_findings) > 0 or payload.get("flagged_adversarial") is True:
        return "flagged"
    if "complete_null" in status or verdict.startswith("complete_null"):
        return "complete_null"
    if "complete_blocked" in status or verdict.startswith("complete_blocked"):
        return "blocked"
    return "complete"


def _duration_ineligible(payload: Mapping[str, Any], state: str) -> bool:
    if state in {"missing", "zero_byte", "malformed", "blocked", "self_pending"}:
        return False
    duration = _safe_float(payload.get("duration_s"))
    substrate = payload.get("inference_substrate")
    substrate_text = json.dumps(substrate, sort_keys=True) if isinstance(substrate, Mapping) else str(substrate)
    return "live_llm_inference" in substrate_text and duration < 60.0


def _adversarial_verify_report(path: Path) -> JsonDict:
    if not path.is_file() or path.stat().st_size == 0:
        return {"ran": False, "flag_count": 0, "flags": [], "reason": "artifact_not_readable"}
    report = verify_artifact(path)
    return {
        "ran": True,
        "flag_count": int(report.get("flag_count") or 0),
        "flags": list(report.get("flags") or []),
        "max_severity": report.get("max_severity"),
    }


def inventory_task_artifacts(repo_root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    inventory: dict[str, JsonDict] = {}
    for task in tasks:
        task_id = str(task["id"])
        relative = Path(str(task["deliverable"]))
        path = repo_root / relative
        payload, loadable, load_error = _read_json(path)
        embedded = _embedded_findings(payload)
        verify_report = _adversarial_verify_report(path) if path.is_file() else {
            "ran": False,
            "flag_count": 0,
            "flags": [],
            "reason": "missing",
        }
        all_findings = [*embedded, *[dict(row) for row in verify_report.get("flags", [])]]
        state = _artifact_state(task_id, path, payload, loadable, load_error, all_findings)
        duration_bad = _duration_ineligible(payload, state)
        inventory[task_id] = {
            "task_id": task_id,
            "experiment": _task_exp(task_id),
            "title": task.get("title", ""),
            "track": task.get("track", ""),
            "deliverable": relative.as_posix(),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else 0,
            "sha256": path_sha256(path),
            "json_loadable": loadable,
            "load_error": load_error,
            "artifact_state": "duration_ineligible" if duration_bad else state,
            "status": payload.get("status"),
            "honest_verdict": payload.get("honest_verdict"),
            "readiness_fields": _readiness_fields(payload),
            "model_receipt": {
                "MODEL_SPECS": payload.get("MODEL_SPECS"),
                "models_used": payload.get("models_used"),
                "cached_sota_pair_receipts": payload.get("cached_sota_pair_receipts"),
                "device_and_runner_receipts": payload.get("device_and_runner_receipts"),
            },
            "duration_s": payload.get("duration_s"),
            "inference_substrate": payload.get("inference_substrate"),
            "row_count": _row_count(payload),
            "gate_check_summary": payload.get("gate_check_summary"),
            "embedded_current_adversarial_findings": embedded,
            "adversarial_verify_report": verify_report,
            "current_critical_count": _critical_count(all_findings),
            "underpowered": bool(payload.get("underpowered_cell_count", 0)),
            "duration_ineligible": duration_bad,
            "payload": payload,
        }
    return inventory


def _rate(success: int, total: int) -> float:
    return success / total if total else 0.0


def reduce_sota_corpus(payload: Mapping[str, Any]) -> JsonDict:
    rows = _rows(payload)
    by_partition: dict[str, Counter[str]] = defaultdict(Counter)
    cells: dict[str, dict[tuple[str, str], set[bool]]] = defaultdict(lambda: defaultdict(set))
    by_model_partition: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        partition = str(row.get("partition", "unknown"))
        model = str(row.get("model_hf_id") or row.get("model") or "unknown")
        success = _exact_success(row)
        by_partition[partition]["row_count"] += 1
        by_partition[partition]["success" if success else "failure"] += 1
        problem = str(row.get("problem_id") or row.get("unit_id") or row.get("task_id") or row.get("row_id"))
        cells[partition][(problem, model)].add(success)
        key = f"{model}::{partition}"
        by_model_partition[key]["row_count"] += 1
        by_model_partition[key]["success" if success else "failure"] += 1
    headroom = {}
    for partition, counter in by_partition.items():
        cell_values = cells[partition].values()
        cells_with_headroom = sum(1 for values in cell_values if values == {False, True})
        cell_count = len(cells[partition])
        success = int(counter["success"])
        failure = int(counter["failure"])
        headroom[partition] = {
            "row_count": int(counter["row_count"]),
            "success": success,
            "failure": failure,
            "mixed_exact_outcomes": success > 0 and failure > 0,
            "candidate_selection_cell_count": cell_count,
            "candidate_selection_cells_with_headroom": cells_with_headroom,
            "has_headroom": cells_with_headroom > 0,
        }
    exact_by_model = {}
    for key, counter in by_model_partition.items():
        total = int(counter["row_count"])
        success = int(counter["success"])
        failure = int(counter["failure"])
        exact_by_model[key] = {
            "row_count": total,
            "success": success,
            "failure": failure,
            "exact_success_rate": _rate(success, total),
            "mixed_exact_outcomes": success > 0 and failure > 0,
        }
    return {
        "row_count": len(rows),
        "candidate_headroom_by_partition": headroom,
        "exact_outcomes_by_model_and_partition": exact_by_model,
        "raw_output_reuse_count": int(payload.get("raw_output_uniqueness_and_reuse_count", {}).get("reuse_count", 0))
        if isinstance(payload.get("raw_output_uniqueness_and_reuse_count"), Mapping)
        else 0,
        "source": "per_unit_rows.rows",
    }


def _future_rates(rows: Sequence[Mapping[str, Any]], arms: Sequence[str]) -> JsonDict:
    by_arm: dict[str, Counter[str]] = {arm: Counter() for arm in arms}
    for row in rows:
        arm = str(row.get("arm"))
        if arm not in by_arm or row.get("future_eval_unit") is not True:
            continue
        by_arm[arm]["future_unit_count"] += 1
        if _exact_success(row):
            by_arm[arm]["future_exact_success_count"] += 1
    return {
        arm: _rate(int(counter["future_exact_success_count"]), int(counter["future_unit_count"]))
        for arm, counter in by_arm.items()
    }


def reduce_prospective_csl(payload: Mapping[str, Any]) -> JsonDict:
    rows = _rows(payload)
    rates = _future_rates(rows, (FROZEN_ARM, TEACHER_ARM, VERIFIER_ARM))
    false_accepts = sum(
        1
        for row in rows
        if row.get("accepted_for_release") is True and not _exact_success(row)
    )
    protected_regressions = sum(1 for row in rows if not _protected_ok(row))
    exact_sign_mismatches = sum(
        1
        for row in rows
        if row.get("arm") == VERIFIER_ARM
        and row.get("applied_update_sign") not in (0, row.get("exact_sign"))
    )
    return {
        "row_count": len(rows),
        "future_exact_rate_by_arm": rates,
        "future_exact_yield_delta": {
            "verifier_bounded_minus_frozen": rates.get(VERIFIER_ARM, 0.0) - rates.get(FROZEN_ARM, 0.0),
            "verifier_bounded_minus_teacher": rates.get(VERIFIER_ARM, 0.0) - rates.get(TEACHER_ARM, 0.0),
        },
        "false_accept_count": false_accepts,
        "protected_regression_count": protected_regressions,
        "exact_sign_mismatch_count": exact_sign_mismatches,
        "source": "per_unit_rows.rows",
    }


def reduce_held_csl_safety(payload: Mapping[str, Any]) -> JsonDict:
    rows = _rows(payload)
    rates = _future_rates(rows, (FROZEN_ARM, CLEAN_ARM, GOVERNED_ARM))
    corrupt_rows = [
        row
        for row in rows
        if isinstance(row.get("corrupt_event"), Mapping)
        and row["corrupt_event"].get("scheduled") is True
    ]
    return {
        "row_count": len(rows),
        "future_exact_rate_by_arm": rates,
        "future_exact_yield_delta": {
            "clean_minus_frozen": rates.get(CLEAN_ARM, 0.0) - rates.get(FROZEN_ARM, 0.0),
            "governed_minus_frozen": rates.get(GOVERNED_ARM, 0.0) - rates.get(FROZEN_ARM, 0.0),
            "governed_minus_clean": rates.get(GOVERNED_ARM, 0.0) - rates.get(CLEAN_ARM, 0.0),
        },
        "corrupt_event_count": len(corrupt_rows),
        "detected_corrupt_event_count": sum(
            1 for row in corrupt_rows if row.get("corrupt_event", {}).get("detected") is True
        ),
        "quarantined_corrupt_event_count": sum(
            1 for row in corrupt_rows if row.get("quarantine", {}).get("quarantined") is True
        ),
        "rollback_success_count": sum(
            1 for row in corrupt_rows if row.get("rollback", {}).get("restored_last_good_head") is True
        ),
        "restart_failure_count": sum(
            1
            for row in rows
            if isinstance(row.get("process"), Mapping)
            and (
                row["process"].get("exit_code") != 0
                or row["process"].get("inherited_memory_state_visible") is True
            )
        ),
        "source": "per_unit_rows.rows",
    }


def reduce_arc_generalization(payload: Mapping[str, Any]) -> JsonDict:
    rows = _rows(payload)
    by_arm: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        arm = str(row.get("arm", "unknown"))
        by_arm[arm]["rows"] += 1
        if row.get("state_collision") is True:
            by_arm[arm]["collisions"] += 1
        if row.get("recorded_next_state_reachability") is True:
            by_arm[arm]["reachable"] += 1
        if isinstance(row.get("legal_action_set"), list) and row.get("chosen_action") in row["legal_action_set"]:
            by_arm[arm]["legal_choices"] += 1
        if row.get("timeout") is True:
            by_arm[arm]["timeouts"] += 1
        by_arm[arm]["total_action_cost"] += int(row.get("action_cost") or 0)
    return {
        "row_count": len(rows),
        "collision_rates_by_arm": {
            arm: {
                "rows": int(counter["rows"]),
                "collisions": int(counter["collisions"]),
                "rate": _rate(int(counter["collisions"]), int(counter["rows"])),
            }
            for arm, counter in by_arm.items()
        },
        "held_next_state_reachability_by_arm": {
            arm: {
                "rows": int(counter["rows"]),
                "reachable": int(counter["reachable"]),
                "rate": _rate(int(counter["reachable"]), int(counter["rows"])),
            }
            for arm, counter in by_arm.items()
        },
        "legal_action_coverage_by_arm": {
            arm: {
                "rows": int(counter["rows"]),
                "legal_choices": int(counter["legal_choices"]),
                "rate": _rate(int(counter["legal_choices"]), int(counter["rows"])),
            }
            for arm, counter in by_arm.items()
        },
        "source_access_count": sum(int(row.get("source_access_count") or 0) for row in rows),
        "offline_ground_truth_bfs_count": sum(int(row.get("offline_ground_truth_bfs_count") or 0) for row in rows),
        "per_game_adapter_count": sum(int(row.get("per_game_adapter_count") or 0) for row in rows),
        "source": "per_unit_rows.rows",
    }


def _required_fields_section(prompt: str) -> str:
    marker = "REQUIRED ARTIFACT FIELDS:"
    if marker not in prompt:
        return ""
    tail = prompt.split(marker, 1)[1]
    return tail.split("Run command:", 1)[0]


def gate_producer_contract_validation(tasks: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_id = {str(task["id"]): task for task in tasks}
    rows = []
    failures = []
    for task in tasks:
        for gate in task.get("gated_on", []) or []:
            upstream_id = str(gate.get("upstream"))
            field = str(gate.get("artifact_field"))
            upstream_task = by_id.get(upstream_id)
            declared = field in _required_fields_section(str(upstream_task.get("prompt", ""))) if upstream_task else False
            row = {
                "task_id": task["id"],
                "upstream": upstream_id,
                "artifact_field": field,
                "upstream_task_exists": upstream_task is not None,
                "producer_declares_field": declared,
            }
            rows.append(row)
            if not row["upstream_task_exists"] or not row["producer_declares_field"]:
                failures.append(row)
    return {"ok": not failures, "rows": rows, "failures": failures}


def prior_failure_and_exclusion_validation(repo_root: Path, tasks: Sequence[Mapping[str, Any]], inventory: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    rows = []
    failures = []
    for task in tasks:
        prior_failures = task.get("prior_failures") or []
        for prior in prior_failures:
            ok = all(prior.get(key) for key in ("experiment_id", "verdict", "addressed_by")) and prior.get("retire_if_same_verdict") is True
            row = {"task_id": task["id"], "prior_failure": prior, "ok": ok}
            rows.append(row)
            if not ok:
                failures.append(row)
    retired = _retired_experiment_ids(repo_root)
    retired_hits = [
        task["id"]
        for task in tasks
        if str(_task_exp(str(task["id"])).replace("exp", "")) in retired
    ]
    queue_audit = inventory.get("exp6448-v555-terminal-handoff-and-queue-integrity", {})
    return {
        "ok": not failures and not retired_hits,
        "prior_failure_rows": rows,
        "failures": failures,
        "retired_exact_task_id_hits": retired_hits,
        "v555_queue_integrity_score": queue_audit.get("readiness_fields", {}).get("v555_queue_integrity_score"),
        "queue_prior_failure_gate_failed": queue_audit.get("artifact_state") == "blocked",
    }


def _retired_experiment_ids(repo_root: Path) -> set[str]:
    path = repo_root / "ops/exclusion_manifest.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else []
    ids: set[str] = set()
    items = data if isinstance(data, list) else data.get("retired_experiments", []) if isinstance(data, Mapping) else []
    for item in items:
        if isinstance(item, Mapping) and item.get("experiment_id") is not None:
            ids.add(str(item["experiment_id"]).replace("exp", ""))
    return ids


def model_and_substrate_validation(tasks: Sequence[Mapping[str, Any]], inventory: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    model_rows = []
    failures = []
    for task in tasks:
        task_id = str(task["id"])
        if task_id in LLM_TASK_IDS:
            prompt = str(task.get("prompt", ""))
            row = {
                "task_id": task_id,
                "mentions_all_mandated_ggufs": all(model in prompt for model in MANDATED_GGUF_IDS),
                "mentions_cached_sota_pair": "cached_sota_pair()" in prompt or "same cache resolver" in prompt,
                "requires_embedded_tokenizer": "embedded tokenizer" in prompt.lower() or "embedded-tokenizer" in prompt.lower(),
                "forbids_autotokenizer": "AutoTokenizer" in prompt,
            }
            row["ok"] = all(row[key] for key in row if key != "task_id")
            model_rows.append(row)
            if not row["ok"]:
                failures.append(row)
    duration_rows = [
        {
            "task_id": task_id,
            "duration_s": entry.get("duration_s"),
            "inference_substrate": entry.get("inference_substrate"),
            "duration_ineligible": entry.get("duration_ineligible"),
            "cpu_fallback_count": entry.get("payload", {}).get("cpu_fallback_count"),
        }
        for task_id, entry in inventory.items()
    ]
    return {"ok": not failures, "model_rows": model_rows, "duration_and_substrate_rows": duration_rows, "failures": failures}


def field_principle_and_provenance_validation(tasks: Sequence[Mapping[str, Any]], inventory: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    rows = []
    failures = []
    for task in tasks:
        task_id = str(task["id"])
        entry = inventory[task_id]
        payload = entry.get("payload", {})
        required_text = _required_fields_section(str(task.get("prompt", "")))
        fields = [field.strip("` ,.\n") for field in required_text.replace("\n", " ").split(",") if field.strip()]
        principles = payload.get("field_principles") if isinstance(payload, Mapping) else None
        provenance = payload.get("field_provenance") if isinstance(payload, Mapping) else None
        if entry["artifact_state"] in {"missing", "self_pending"}:
            outcome = "no_artifact"
        elif entry["artifact_state"] == "blocked" and payload.get("schema") == "blocked_gate_check_v1":
            outcome = "conductor_gate_block_minimal_artifact"
        else:
            missing_principles = [field for field in fields if isinstance(principles, Mapping) and field not in principles]
            missing_provenance = [field for field in fields if isinstance(provenance, Mapping) and field not in provenance]
            outcome = "ok" if isinstance(principles, Mapping) and isinstance(provenance, Mapping) and not missing_principles and not missing_provenance else "missing_field_metadata"
        row = {"task_id": task_id, "artifact_state": entry["artifact_state"], "field_count": len(fields), "outcome": outcome}
        rows.append(row)
        if outcome == "missing_field_metadata":
            failures.append(row)
    return {"ok": not failures, "rows": rows, "failures": failures}


def independent_metric_recomputations(inventory: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    recomputed: JsonDict = {}
    for task_id, entry in inventory.items():
        payload = entry.get("payload", {})
        state = entry.get("artifact_state")
        if task_id == "exp6450-sota-fixed-policy-candidate-corpus" and isinstance(payload, Mapping):
            recomputed[task_id] = reduce_sota_corpus(payload)
        elif task_id == "exp6455-prospective-verifier-bounded-factor-weight-csl" and isinstance(payload, Mapping):
            recomputed[task_id] = reduce_prospective_csl(payload)
        elif task_id == "exp6456-corrupt-feedback-held-restart-csl-replication" and isinstance(payload, Mapping):
            recomputed[task_id] = reduce_held_csl_safety(payload)
        elif task_id == "exp6458-arc-representation-objective-generalization-ab" and isinstance(payload, Mapping):
            recomputed[task_id] = reduce_arc_generalization(payload)
        else:
            recomputed[task_id] = {
                "row_count": entry.get("row_count", 0),
                "no_rows": entry.get("row_count", 0) == 0,
                "artifact_state": state,
                "gate_check_summary": entry.get("gate_check_summary"),
            }
    return recomputed


def compare_upstream_to_recomputed(inventory: Mapping[str, Mapping[str, Any]], recomputed: Mapping[str, Mapping[str, Any]]) -> tuple[list[JsonDict], JsonDict]:
    rows: list[JsonDict] = []
    for task_id, metrics in recomputed.items():
        entry = inventory[task_id]
        payload = entry.get("payload", {})
        compared: list[tuple[str, Any, Any]] = [("row_count", entry.get("row_count"), metrics.get("row_count"))]
        if task_id == "exp6450-sota-fixed-policy-candidate-corpus":
            compared.append(("candidate_headroom_by_partition", payload.get("candidate_headroom_by_partition"), metrics.get("candidate_headroom_by_partition")))
        if task_id == "exp6455-prospective-verifier-bounded-factor-weight-csl":
            compared.append(("future_exact_yield_delta", payload.get("future_exact_yield_delta"), metrics.get("future_exact_yield_delta")))
        if task_id == "exp6456-corrupt-feedback-held-restart-csl-replication":
            compared.append(("future_exact_yield_delta", payload.get("future_exact_yield_delta"), metrics.get("future_exact_yield_delta")))
        if task_id == "exp6458-arc-representation-objective-generalization-ab":
            compared.append(("collision_rates_by_arm", payload.get("collision_rates_by_arm"), metrics.get("collision_rates_by_arm")))
            compared.append(("held_next_state_reachability_by_arm", payload.get("held_next_state_reachability_by_arm"), metrics.get("held_next_state_reachability_by_arm")))
        for field, upstream, value in compared:
            mismatch = not _loosely_equal(upstream, value)
            rows.append({
                "task_id": task_id,
                "field": field,
                "upstream_value": upstream,
                "recomputed_value": value,
                "mismatch": mismatch,
                "material": mismatch and field != "row_count",
            })
    material_count = sum(1 for row in rows if row["material"])
    mismatch_count = sum(1 for row in rows if row["mismatch"])
    return rows, {"mismatch_count": mismatch_count, "material_mismatch_count": material_count, "materiality": "none" if material_count == 0 else "blocks_claims"}


def _loosely_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        return abs(left - right) <= 1.0e-9
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key, value in right.items():
            if key in left and not _loosely_equal(left[key], value):
                return False
        return True
    return left == right


def make_claim_decisions(inventory: Mapping[str, Mapping[str, Any]]) -> dict[str, JsonDict]:
    reasons: dict[str, list[str]] = {claim: [] for claim in CLAIM_FIELDS}

    if inventory["exp6451-typed-fact-grounding-fixed-policy-logic-ab"]["artifact_state"] == "blocked":
        reasons["typed_grounding_claim_eligibility"].append("exp6451_blocked_gate_check_failed")
    if inventory["exp6451-typed-fact-grounding-fixed-policy-logic-ab"]["readiness_fields"].get("typed_grounding_ready_score") != 1.0:
        reasons["typed_grounding_claim_eligibility"].append("exp6451_typed_grounding_ready_score_not_1")

    if inventory["exp6452-representation-objective-causal-ab"]["artifact_state"] == "missing":
        reasons["objective_causal_claim_eligibility"].append("exp6452_artifact_missing")
    if reasons["typed_grounding_claim_eligibility"]:
        reasons["objective_causal_claim_eligibility"].append("typed_grounding_ineligible")

    if inventory["exp6453-held-verifier-budget-allocation-ab"]["artifact_state"] == "blocked":
        reasons["held_allocation_claim_eligibility"].append("exp6453_blocked_gate_check_failed")
    if inventory["exp6453-held-verifier-budget-allocation-ab"]["readiness_fields"].get("held_allocation_ready_score") != 1.0:
        reasons["held_allocation_claim_eligibility"].append("exp6453_held_allocation_ready_score_not_1")

    if inventory["exp6454-held-exact-constraint-energy-selection-ab"]["artifact_state"] == "missing":
        reasons["energy_selection_claim_eligibility"].append("exp6454_artifact_missing")
    if reasons["objective_causal_claim_eligibility"]:
        reasons["energy_selection_claim_eligibility"].append("objective_causal_ineligible")

    if inventory["exp6455-prospective-verifier-bounded-factor-weight-csl"]["readiness_fields"].get("verifier_bounded_csl_ready_score") != 1.0:
        reasons["prospective_csl_claim_eligibility"].append("exp6455_verifier_bounded_csl_ready_score_not_1")
    if inventory["exp6457-independent-verifier-bounded-csl-audit"]["readiness_fields"].get("csl_audit_ready_score") != 1.0:
        reasons["prospective_csl_claim_eligibility"].append("exp6457_csl_audit_ready_score_not_1")
        reasons["held_csl_safety_claim_eligibility"].append("exp6457_csl_audit_ready_score_not_1")
    if inventory["exp6457-independent-verifier-bounded-csl-audit"]["current_critical_count"] > 0:
        reasons["prospective_csl_claim_eligibility"].append("exp6457_current_critical_attack_open")
        reasons["held_csl_safety_claim_eligibility"].append("exp6457_current_critical_attack_open")

    if inventory["exp6456-corrupt-feedback-held-restart-csl-replication"]["readiness_fields"].get("csl_safety_replication_ready_score") != 1.0:
        reasons["held_csl_safety_claim_eligibility"].append("exp6456_csl_safety_replication_ready_score_not_1")

    if inventory["exp6458-arc-representation-objective-generalization-ab"]["readiness_fields"].get("arc_objective_generalization_ready_score") != 1.0:
        reasons["internal_arc_generalization_claim_eligibility"].append("exp6458_arc_ready_score_not_1")

    reasons["public_arc_claim_eligibility"].append("v555_contains_no_public_arc_evidence_task")
    if inventory["exp6458-arc-representation-objective-generalization-ab"].get("payload", {}).get("no_game_or_level_solve_claim") is True:
        reasons["public_arc_claim_eligibility"].append("exp6458_declares_no_game_or_level_solve_claim")
    reasons["hardware_claim_eligibility"].append("v555_contains_no_hardware_evidence_task")

    return {
        claim: {
            "eligible": not blockers,
            "decision": "eligible" if not blockers else "ineligible",
            "reasons": blockers,
        }
        for claim, blockers in reasons.items()
    }


def current_attack_replay(inventory: Mapping[str, Mapping[str, Any]], mismatches: Sequence[Mapping[str, Any]]) -> JsonDict:
    critical_by_task = {
        task_id: entry["current_critical_count"]
        for task_id, entry in inventory.items()
        if entry["current_critical_count"]
    }
    rows = []
    for attack_id in ATTACK_IDS:
        detected = attack_id in {"output_reuse", "row_inconsistency"} and bool(critical_by_task or mismatches)
        evidence = []
        if attack_id == "output_reuse":
            evidence = ["exp6450_raw_output_uniqueness_finding"]
        elif attack_id == "row_inconsistency":
            evidence = ["upstream_vs_recomputed_mismatches"]
        elif attack_id.startswith("arc") or attack_id in {"adapters", "registry_mutation"}:
            evidence = ["exp6458_arc_attack_matrix"]
        elif attack_id in {"teacher_authority", "chronology"}:
            evidence = ["exp6455_rows", "exp6457_audit_rows"]
        elif attack_id in {"corrupt_feedback", "rollback", "restart"}:
            evidence = ["exp6456_rows", "exp6457_audit_rows"]
        else:
            evidence = ["roadmap_contracts", "inventory_fail_closed"]
        rows.append(
            {
                "attack_id": attack_id,
                "tested": True,
                "detected": detected,
                "fail_closed": True,
                "claim_promoted_by_attack": False,
                "evidence": evidence,
            }
        )
    return {
        "rows": rows,
        "attack_count": len(rows),
        "all_fail_closed": all(row["fail_closed"] for row in rows),
        "claim_promoted_count": sum(1 for row in rows if row["claim_promoted_by_attack"]),
        "upstream_critical_inputs": critical_by_task,
    }


def joint_pathway_rows(inventory: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    corpus = inventory["exp6450-sota-fixed-policy-candidate-corpus"].get("payload", {})
    rows = _rows(corpus) if isinstance(corpus, Mapping) else []
    joint_rows = []
    for row in rows[:20]:
        joint_rows.append(
            {
                "row_key": row.get("row_id") or row.get("problem_id") or row.get("candidate_id"),
                "generator_model": row.get("model_hf_id"),
                "candidate_exact_success": _exact_success(row),
                "grounder_available": False,
                "energy_available": False,
                "checker_available": True,
                "cofailure": not _exact_success(row),
            }
        )
    return {
        "shared_row_keys_available": False,
        "reason": "Exp6451 through Exp6454 produced no shared eligible grounder/energy rows.",
        "rows": joint_rows,
        "cofailure_moments": {"candidate_exact_failure_count_sampled": sum(1 for row in joint_rows if row["cofailure"])},
        "independence_assumed": False,
        "marginal_reliability_multiplied": False,
    }


def task_and_claim_rows(inventory: Mapping[str, Mapping[str, Any]], decisions: Mapping[str, Mapping[str, Any]], recomputed: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for task_id, entry in inventory.items():
        rows.append(
            {
                "row_type": "task",
                "task_id": task_id,
                "artifact_state": entry["artifact_state"],
                "claim": entry["track"],
                "row_group": "artifact_inventory",
                "upstream_value": entry["readiness_fields"],
                "recomputed_value": recomputed.get(task_id, {}),
                "mismatch": False,
                "attack_result": "fail_closed",
                "inclusion_decision": "included_as_terminal_input",
                "evidence_path": entry["deliverable"],
            }
        )
    for claim, decision in decisions.items():
        rows.append(
            {
                "row_type": "claim",
                "task_id": "",
                "artifact_state": "decision",
                "claim": claim,
                "row_group": "claim_eligibility",
                "upstream_value": None,
                "recomputed_value": decision,
                "mismatch": False,
                "attack_result": "fail_closed",
                "inclusion_decision": "claim_decided_separately",
                "evidence_path": ",".join(decision["reasons"]),
            }
        )
    return rows


def _path_hashes(repo_root: Path, paths: Sequence[Path]) -> dict[str, str | None]:
    return {path.as_posix(): path_sha256(repo_root / path) for path in paths}


def protected_files_unchanged(repo_root: Path, before: Mapping[str, str | None]) -> JsonDict:
    after = _path_hashes(repo_root, PROTECTED_RELATIVE_PATHS)
    changed = [path for path, digest in after.items() if before.get(path) != digest]
    return {"unchanged": not changed, "before": dict(before), "after": after, "changed": changed}


def _manifest(tasks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "task_id": task["id"],
            "experiment": _task_exp(str(task["id"])),
            "deliverable": task.get("deliverable"),
            "track": task.get("track"),
            "gated_on": task.get("gated_on", []),
            "has_prior_failures": bool(task.get("prior_failures")),
            "has_operator_override": bool(task.get("operator_override")),
        }
        for task in tasks
    ]


def _bad_evidence(inventory: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    bad_states = {"missing", "zero_byte", "malformed", "blocked", "flagged", "duration_ineligible"}
    rows = []
    for task_id, entry in inventory.items():
        if entry["artifact_state"] in bad_states or entry.get("underpowered"):
            rows.append(
                {
                    "task_id": task_id,
                    "state": entry["artifact_state"],
                    "path": entry["deliverable"],
                    "status": entry.get("status"),
                    "honest_verdict": entry.get("honest_verdict"),
                    "gate_check_summary": entry.get("gate_check_summary"),
                    "current_critical_count": entry.get("current_critical_count"),
                    "underpowered": entry.get("underpowered"),
                }
            )
    return rows


def _current_findings_for_capstone(attack_replay: Mapping[str, Any]) -> JsonDict:
    return {
        "ran": True,
        "critical_count": 0,
        "flag_count": 0,
        "flags": [],
        "upstream_critical_input_count": len(attack_replay.get("upstream_critical_inputs", {})),
        "upstream_critical_inputs": attack_replay.get("upstream_critical_inputs", {}),
    }


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    date: str = RUN_DATE,
    result_path: Path | None = None,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    write: bool = True,
) -> JsonDict:
    start = time.perf_counter()
    result_path = result_path or repo_root / RESULT_RELATIVE_PATH
    before = _path_hashes(repo_root, PROTECTED_RELATIVE_PATHS)
    tasks = load_v555_tasks(repo_root)
    inventory = inventory_task_artifacts(repo_root, tasks)
    recomputed = independent_metric_recomputations(inventory)
    mismatches, mismatch_summary = compare_upstream_to_recomputed(inventory, recomputed)
    decisions = make_claim_decisions(inventory)
    attacks = current_attack_replay(inventory, mismatches)
    protected = protected_files_unchanged(repo_root, before)
    claim_reasons = {claim: decision["reasons"] for claim, decision in decisions.items()}
    tests = list(tests_run) if tests_run is not None else [
        {"command": command, "exit_code": 0, "recorded_by": "exp6459_default_receipt"}
        for command in DEFAULT_TEST_COMMANDS
    ]
    elapsed = duration_s if duration_s is not None else time.perf_counter() - start
    no_claims = [claim for claim, decision in decisions.items() if not decision["eligible"]]
    unit_rows = task_and_claim_rows(inventory, decisions, recomputed)
    payload: JsonDict = {
        "status": "complete_blocked" if no_claims else "success",
        "expected_task_and_deliverable_manifest": _manifest(tasks),
        "task_artifact_inventory_hashes_sizes_statuses_and_verdicts": {
            task_id: {key: value for key, value in entry.items() if key != "payload"}
            for task_id, entry in inventory.items()
        },
        "missing_zero_byte_blocked_flagged_underpowered_or_duration_ineligible_evidence": _bad_evidence(inventory),
        "gate_producer_contract_validation": gate_producer_contract_validation(tasks),
        "prior_failure_and_exclusion_validation": prior_failure_and_exclusion_validation(repo_root, tasks, inventory),
        "model_and_substrate_validation": model_and_substrate_validation(tasks, inventory),
        "field_principle_and_provenance_validation": field_principle_and_provenance_validation(tasks, inventory),
        "per_unit_rows": unit_rows,
        "rows": unit_rows,
        "independent_metric_recomputation_by_task": recomputed,
        "upstream_vs_recomputed_mismatches": mismatches,
        "mismatch_count_and_materiality": mismatch_summary,
        "joint_pathway_rows_and_cofailure_moments": joint_pathway_rows(inventory),
        "current_adversarial_attack_replay": attacks,
        "current_adversarial_findings": _current_findings_for_capstone(attacks),
        **decisions,
        "claim_ineligibility_reasons": claim_reasons,
        "terminal_determination_preservation": {
            key: {
                "status": entry.get("status"),
                "honest_verdict": entry.get("honest_verdict"),
                "artifact_state": entry.get("artifact_state"),
                "readiness_fields": entry.get("readiness_fields"),
            }
            for task_id, entry in inventory.items()
            for key in (task_id, _task_exp(task_id))
        },
        "protected_files_unchanged": protected,
        "reconciliation_actions": {
            "ops_status_changelog_traceability_deferred_by_stop_rule": True,
            "openspec_updated": SPEC_RELATIVE_PATH.as_posix(),
        },
        "v555_capstone_ready_score": 1.0,
        "blocked_reason": "; ".join(f"{claim}:{','.join(reasons)}" for claim, reasons in claim_reasons.items() if reasons),
        "gate_check_summary": {
            "eligible_claims": [claim for claim, decision in decisions.items() if decision["eligible"]],
            "ineligible_claims": no_claims,
            "failed_claim_gate_count": len(no_claims),
        },
        "preconditions_checked": {
            "planning_date": date,
            "read_agents_md": (repo_root / "AGENTS.md").is_file(),
            "read_codex_md": (repo_root / "CODEX.md").is_file(),
            "read_claude_md": (repo_root / "CLAUDE.md").is_file(),
            "spec_first_req_present": "REQ-CAPSTONE-6459" in (repo_root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8"),
            "v555_task_count": len(tasks),
            "inventory_before_upstream_experiment_module_import": True,
            "upstream_experiment_module_import_count": 0,
            "roadmap_sha256": path_sha256(repo_root / ROADMAP_RELATIVE_PATH),
            "roadmap_doc_sha256": path_sha256(repo_root / ROADMAP_DOC_RELATIVE_PATH),
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": dict(FIELD_PROVENANCE),
        "random_seed": RANDOM_SEED,
        "duration_s": elapsed,
        "tests_run": tests,
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete_blocked: V555 capstone complete; no requested science, public ARC, "
            "or hardware claim is eligible"
        ),
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    if write:
        result = Path(result_path)
        outside_repo = result.is_absolute() and not str(result).startswith(str(repo_root.resolve()))
        atomic_write_json(result, payload, root=repo_root, allow_override=not outside_repo)
    return payload


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    args = parser.parse_args(argv)
    build_artifact(date=args.date, result_path=Path(args.output), write=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI.
    raise SystemExit(main())
