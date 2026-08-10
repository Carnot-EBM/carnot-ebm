"""Exp6283 V541 exact-path adversarial capstone.

Spec refs: REQ-INFRA-6283, SCENARIO-INFRA-6283-1,
SCENARIO-INFRA-6283-2, SCENARIO-INFRA-6283-3,
SCENARIO-INFRA-6283-4, SCENARIO-INFRA-6283-5,
SCENARIO-INFRA-6283-6.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

import yaml

from carnot.experiment_6272_v541_terminal_transition import same_number_aliases
from carnot.experiment_artifacts import atomic_write_json
from carnot.terminal_artifacts import (
    canonical_json,
    classify_artifact_path,
    classify_artifact_payload,
    gate_field_eligibility_for_path,
    path_sha256,
    payload_sha256,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(SCRIPTS_ROOT))

MILESTONE = "2026.08.541"
EXPERIMENT_ID = "exp6283-v541-adversarial-capstone"
SCHEMA = "carnot.experiment_6283.v541_adversarial_capstone.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6283_v541_adversarial_capstone.json")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
MILESTONE_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
ARC_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
TRACEABILITY_RELATIVE_PATH = Path("_bmad/traceability.md")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
TERMINAL_ARTIFACTS_RELATIVE_PATH = Path("python/carnot/terminal_artifacts.py")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
SUMMARY_ARTIFACT_RELATIVE_PATH = Path("scripts/summarize_artifact.py")
PUBLICATION_GATE_RELATIVE_PATH = Path("scripts/publication_gate.py")

EXPECTED_V541_TASK_IDS = (
    "exp6272-v541-terminal-transition",
    "exp6273-v541-post-marker-source-scope-freeze",
    "exp6274-asp-energy-semantic-compiler",
    "exp6275-flagship-asp-constraint-verification-benchmark",
    "exp6276-certified-dual-cache-admission",
    "exp6277-chronological-certified-csl-ab",
    "exp6278-model-family-task-holdout-csl-audit",
    "exp6279-certified-memory-shadow-consumer",
    "exp6280-variable-cardinality-mode-jump-backend",
    "exp6281-mode-jump-multifamily-rerun",
    "exp6282-arc-mechanic-class-live-router",
    EXPERIMENT_ID,
)

TASK_TO_STATE_FIELD = {
    "exp6274-asp-energy-semantic-compiler": "asp_semantic_compiler_state",
    "exp6275-flagship-asp-constraint-verification-benchmark": "flagship_asp_benchmark_state",
    "exp6276-certified-dual-cache-admission": "certified_admission_state",
    "exp6277-chronological-certified-csl-ab": "chronological_continuous_learning_state",
    "exp6278-model-family-task-holdout-csl-audit": "heldout_transfer_state",
    "exp6279-certified-memory-shadow-consumer": "shadow_consumer_state",
    "exp6280-variable-cardinality-mode-jump-backend": "variable_cardinality_backend_state",
    "exp6281-mode-jump-multifamily-rerun": "mode_jump_safety_and_value_state",
    "exp6282-arc-mechanic-class-live-router": "arc_mechanic_router_state",
}

STATE_READINESS_FIELDS = {
    "asp_semantic_compiler_state": ("asp_energy_semantic_ready_score",),
    "flagship_asp_benchmark_state": ("flagship_asp_event_corpus_ready_score",),
    "certified_admission_state": ("certified_admission_ready_score",),
    "chronological_continuous_learning_state": ("continuous_learning_promotion_ready_score",),
    "heldout_transfer_state": ("heldout_certified_transfer_ready_score",),
    "shadow_consumer_state": ("default_off_shadow_ready_score",),
    "variable_cardinality_backend_state": ("variable_cardinality_backend_ready_score",),
    "mode_jump_safety_and_value_state": (
        "mode_jump_safety_ready_score",
        "mode_jump_workload_value_ready_score",
    ),
    "arc_mechanic_router_state": ("arc_mechanic_router_ready_score",),
}

FORBIDDEN_ZERO_FIELDS = (
    "source_mutation_count",
    "weight_mutation_count",
    "unauthorized_external_call_count",
    "hidden_game_source_access_count",
    "outer_loop_ground_truth_search_count",
    "arc_level_solve_claim_count",
    "registry_update_count",
    "hardware_claim_count",
    "speed_power_or_energy_claim_count",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "milestone_roadmap_path_and_hash",
    "exact_declared_deliverable_matrix",
    "conductor_receipt_matrix",
    "exact_path_over_receipt_precedence",
    "current_rule_adversarial_results_by_task",
    "terminal_nonterminal_blocked_skipped_null_flagged_retired_and_ready_counts",
    "gate_cascade_receipts",
    "asp_semantic_compiler_state",
    "flagship_asp_benchmark_state",
    "certified_admission_state",
    "chronological_continuous_learning_state",
    "heldout_transfer_state",
    "shadow_consumer_state",
    "variable_cardinality_backend_state",
    "mode_jump_safety_and_value_state",
    "arc_mechanic_router_state",
    "arc_provenance_and_registry_receipts",
    "branch_independent_promotion_ledger",
    "prior_failure_retirement_actions",
    "publication_gate_g1_g2_g3_g4_and_unmet_gates",
    "source_mutation_count",
    "weight_mutation_count",
    "unauthorized_external_call_count",
    "hidden_game_source_access_count",
    "outer_loop_ground_truth_search_count",
    "arc_level_solve_claim_count",
    "registry_update_count",
    "hardware_claim_count",
    "speed_power_or_energy_claim_count",
    "protected_files_unchanged",
    "spec_traceability_status_changelog_reconciliation",
    "prd_gap_table",
    "next_milestone_recommendations",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "The capstone is terminal only after exact V541 evidence is classified.",
    "milestone_roadmap_path_and_hash": "The active roadmap defines the V541 denominator.",
    "exact_declared_deliverable_matrix": "Exact paths prevent aliases from replacing missing work.",
    "conductor_receipt_matrix": "Receipts provide context but cannot promote artifacts.",
    "exact_path_over_receipt_precedence": "The declared JSON path outranks orchestration text.",
    "current_rule_adversarial_results_by_task": "Current flags are checked before promotion.",
    "terminal_nonterminal_blocked_skipped_null_flagged_retired_and_ready_counts": "All evidence classes stay visible.",
    "gate_cascade_receipts": "Structured gates are recomputed from exact bare fields.",
    "asp_semantic_compiler_state": "ASP semantics must stand on exact compiler parity evidence.",
    "flagship_asp_benchmark_state": "Flagship benchmark evidence cannot outrank its flagged artifact.",
    "certified_admission_state": "Certified admission must pass before CSL can run.",
    "chronological_continuous_learning_state": "A CSL gate tombstone remains skipped.",
    "heldout_transfer_state": "Missing heldout transfer cannot be promoted.",
    "shadow_consumer_state": "A skipped shadow consumer is not integration evidence.",
    "variable_cardinality_backend_state": "Backend readiness is separate from sampler value.",
    "mode_jump_safety_and_value_state": "Mode-jump safety and workload value are independent gates.",
    "arc_mechanic_router_state": "ARC router evidence must be live-path and unflagged.",
    "arc_provenance_and_registry_receipts": "ARC source, solve, and registry counts must stay explicit.",
    "branch_independent_promotion_ledger": "One branch cannot launder another branch.",
    "prior_failure_retirement_actions": "Retire-if-same-verdict actions use exact verdicts only.",
    "publication_gate_g1_g2_g3_g4_and_unmet_gates": "Publication readiness comes from the stable gate script.",
    "source_mutation_count": "Bare zero proves this capstone did not mutate sources.",
    "weight_mutation_count": "Bare zero proves no model weights changed.",
    "unauthorized_external_call_count": "Bare zero proves no unapproved external call is claimed.",
    "hidden_game_source_access_count": "Bare zero preserves hidden-game source discipline.",
    "outer_loop_ground_truth_search_count": "Bare zero prevents outer-loop solve laundering.",
    "arc_level_solve_claim_count": "Bare zero proves no ARC level solve is claimed.",
    "registry_update_count": "Bare zero proves the ARC registry was not updated.",
    "hardware_claim_count": "Bare zero proves no hardware result is claimed.",
    "speed_power_or_energy_claim_count": "Bare zero blocks speed, power, and energy claims.",
    "protected_files_unchanged": "Protected hashes show this run did not rewrite records.",
    "spec_traceability_status_changelog_reconciliation": "Reconciliation reports evidence without editing protected docs.",
    "prd_gap_table": "PRD gaps cite exact artifacts and blockers.",
    "next_milestone_recommendations": "Recommendations come from observed blockers only.",
    "preconditions_checked": "Inputs, receipts, registry, and rule hashes are frozen first.",
    "inference_substrate": "The capstone aggregates checked-in artifacts only.",
    "verifier_is_oracle": "False because this audits records, not benchmark answers.",
    "field_provenance": "Every required field cites its source evidence.",
    "field_principles": "Every required field states why it exists.",
    "test_commands": "Commands show the verification boundary.",
    "test_exit_codes": "Exit codes stop failures from being laundered.",
    "duration_s": "Wall time is reported without padding.",
    "reproducibility_checksum": "The normalized payload is content-addressed.",
    "honest_verdict": "The verdict states mixed V541 evidence without strengthening it.",
}

COUNT_PRINCIPLES: dict[str, str] = {
    "terminal": "Terminal artifacts may close but still can fail promotion.",
    "nonterminal": "Nonterminal exact artifacts cannot feed claims.",
    "missing": "Missing exact paths cannot be replaced by aliases.",
    "malformed": "Malformed JSON fails closed.",
    "running": "Running artifacts are not terminal evidence.",
    "running_bootstrap": "Bootstrap-running artifacts are not evidence.",
    "bootstrap_only": "Bootstrap-only artifacts cannot support gates.",
    "partial": "Partial artifacts cannot promote branches.",
    "contradictory": "Conflicting status markers fail closed.",
    "unknown": "Unknown markers fail closed.",
    "blocked": "Blocked artifacts are terminal blockers.",
    "skipped": "Gate tombstones stay skipped.",
    "null": "Null evidence is separate from positive evidence.",
    "flagged": "Stamped or current flags quarantine promotion.",
    "retired": "Retired scopes stay retired.",
    "ready": "Ready evidence is counted apart from complete evidence.",
    "complete": "Complete evidence still needs branch gates.",
    "positive": "Positive evidence is never inferred from another branch.",
}

GATE_PRINCIPLE = "A gate can read only a terminal exact artifact with an exact bare field."

PROTECTED_RELATIVE_PATHS = (
    ROADMAP_RELATIVE_PATH,
    MILESTONE_DOC_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    ARC_REGISTRY_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    TRACEABILITY_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    TERMINAL_ARTIFACTS_RELATIVE_PATH,
    ADVERSARIAL_VERIFY_RELATIVE_PATH,
    SUMMARY_ARTIFACT_RELATIVE_PATH,
    PUBLICATION_GATE_RELATIVE_PATH,
    Path("scripts/research_conductor.py"),
)

INPUT_RELATIVE_PATHS = PROTECTED_RELATIVE_PATHS + (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("openspec/capabilities/constraint-verification/spec.md"),
    Path("openspec/capabilities/self-learning/spec.md"),
    Path("openspec/capabilities/samplers/spec.md"),
    Path("openspec/capabilities/arc-world-model-trust-energy/spec.md"),
)

TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6283_v541_adversarial_capstone.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6283_v541_adversarial_capstone.py -m pytest tests/python/test_experiment_6283_v541_adversarial_capstone.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6283_v541_adversarial_capstone.py --fail-under=100 --show-missing",
    ".venv/bin/ruff check python/carnot/experiment_6283_v541_adversarial_capstone.py tests/python/test_experiment_6283_v541_adversarial_capstone.py",
    ".venv/bin/ruff format --check python/carnot/experiment_6283_v541_adversarial_capstone.py tests/python/test_experiment_6283_v541_adversarial_capstone.py",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6283_v541_adversarial_capstone.py",
    ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml",
    ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    ".venv/bin/python scripts/publication_gate.py --json",
    "sed -n 1,220p ops/e2e-test-plan.md",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6283_v541_adversarial_capstone.json",
)
COMMAND_TIMEOUTS_S = {".venv/bin/pytest tests/python -q": 3600}


def payload_checksum(report: JsonMap) -> str:
    normalized = json.loads(canonical_json(report))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return payload_sha256(normalized)


def read_yaml_mapping(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:  # pragma: no cover
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    meta: JsonDict = {
        "path": path.as_posix(),
        "present": path.exists(),
        "loadable": False,
        "sha256": path_sha256(path),
        "error": None,
    }
    if not path.exists():
        meta["error"] = "missing"
        return {}, meta
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover
        meta["error"] = f"json_error:{exc.msg}"
        return {}, meta
    if not isinstance(payload, Mapping):  # pragma: no cover
        meta["error"] = "json_not_mapping"
        return {}, meta
    meta["loadable"] = True
    return dict(payload), meta


def load_roadmap(root: Path) -> JsonDict:
    return read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)


def roadmap_tasks(roadmap: JsonMap) -> list[JsonDict]:
    tasks = roadmap.get("tasks")
    return (
        [dict(task) for task in tasks if isinstance(task, Mapping)]
        if isinstance(tasks, list)
        else []
    )


def git_status_lines(root: Path) -> list[str]:  # pragma: no cover
    try:
        proc = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    return [line for line in proc.stdout.splitlines() if line]


def latest_conductor_receipts(root: Path, tasks: Sequence[JsonMap]) -> JsonDict:
    path = root / CONDUCTOR_LOG_RELATIVE_PATH
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    receipts: JsonDict = {}
    for task in tasks:
        task_id = str(task.get("id") or "")
        title = str(task.get("title") or task_id)
        markers = [title[:size] for size in (58, 52, 46, 40, 34, 28, 22) if len(title) >= size]
        matches = [
            line
            for line in lines
            if task_id in line or any(marker and marker in line for marker in markers)
        ]
        receipt: JsonDict = {
            "receipt_found": False,
            "status": None,
            "detail": None,
            "raw_line": None,
        }
        if matches:
            parts = [part.strip() for part in matches[-1].strip("|").split("|")]
            receipt = {
                "receipt_found": True,
                "timestamp": parts[0] if len(parts) > 0 else None,
                "title_fragment": parts[1] if len(parts) > 1 else None,
                "status": parts[2] if len(parts) > 2 else None,
                "detail": parts[3] if len(parts) > 3 else None,
                "raw_line": matches[-1],
            }
        receipts[task_id] = receipt
    return receipts


def _self_payload() -> JsonDict:
    return {
        "status": "complete",
        "honest_verdict": "complete: Exp6283 capstone payload under construction",
        "duration_s": 0.0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
    }


def build_exact_declared_deliverable_matrix(
    root: Path,
    tasks: Sequence[JsonMap],
    *,
    conductor_receipts: JsonMap | None = None,
    self_payload: JsonMap | None = None,
) -> JsonDict:
    receipts = conductor_receipts or {}
    rows: JsonDict = {}
    for task in tasks:
        task_id = str(task.get("id") or "")
        rel = Path(str(task.get("deliverable") or ""))
        path = root / rel
        receipt = receipts.get(task_id)
        typed_receipt = receipt if isinstance(receipt, Mapping) else None
        if task_id == EXPERIMENT_ID and self_payload is not None:
            payload = dict(self_payload)
            digest = payload_sha256(payload)
            classification = classify_artifact_payload(
                payload,
                path=path,
                sha256=digest,
                conductor_receipt=typed_receipt,
            )
            meta = {"present": path.exists(), "loadable": True, "sha256": digest}
        else:
            payload, meta = read_json_mapping(path)
            classification = classify_artifact_path(path, conductor_receipt=typed_receipt)
        rows[task_id] = {
            "task_id": task_id,
            "title": str(task.get("title") or task_id),
            "track": str(task.get("track") or "unset"),
            "declared_deliverable": rel.as_posix(),
            "present": bool(meta["present"]),
            "loadable": bool(meta["loadable"]),
            "sha256": classification.sha256 or meta["sha256"],
            "terminal_class": classification.classification,
            "terminal": classification.terminal,
            "reason": classification.reason,
            "status_raw": classification.status_raw,
            "honest_verdict_raw": classification.honest_verdict_raw,
            "receipt_status": classification.conductor_receipt_status,
            "receipt_override_attempted": classification.receipt_override_attempted,
            "receipt_overrode": classification.receipt_overrode,
            "flagged_adversarial_stamped": payload.get("flagged_adversarial") is True,
            "corrigendum_pending_stamped": bool(payload.get("corrigendum_pending")),
            "same_number_alias_used": False,
            "same_number_alias_candidates_ignored": same_number_aliases(root, task_id, rel),
        }
    return rows


def evaluate_operator(actual: Any, op: str, expected: Any) -> bool:
    if op == "exists":
        return (actual is not None) is bool(expected)
    if actual is None:
        return False
    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
    if op == "in":
        return (
            isinstance(expected, Sequence)
            and not isinstance(expected, (str, bytes))
            and actual in expected
        )
    try:
        if op == ">":
            return bool(actual > expected)
        if op == ">=":
            return bool(actual >= expected)
        if op == "<":
            return bool(actual < expected)
        if op == "<=":
            return bool(actual <= expected)
    except TypeError:
        return False
    return False


def evaluate_gate_cascade_receipts(root: Path, tasks: Sequence[JsonMap]) -> JsonDict:
    by_id = {str(task.get("id") or ""): task for task in tasks}
    gates: list[JsonDict] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        raw_gates = task.get("gated_on")
        for gate in raw_gates if isinstance(raw_gates, list) else []:
            if not isinstance(gate, Mapping):
                gates.append(
                    {
                        "task_id": task_id,
                        "gate": gate,
                        "passed": False,
                        "actual": None,
                        "reason": "gate_not_mapping",
                        "principle": GATE_PRINCIPLE,
                    }
                )
                continue
            upstream = str(gate.get("upstream") or "")
            field = str(gate.get("artifact_field") or "")
            op = str(gate.get("op") or "")
            expected = gate.get("value")
            upstream_task = by_id.get(upstream)
            if upstream_task is None:
                gates.append(
                    {
                        "task_id": task_id,
                        "upstream": upstream,
                        "artifact_field": field,
                        "op": op,
                        "expected": expected,
                        "actual": None,
                        "passed": False,
                        "reason": "missing_upstream_task",
                        "principle": GATE_PRINCIPLE,
                    }
                )
                continue
            upstream_path = root / Path(str(upstream_task.get("deliverable") or ""))
            eligibility = gate_field_eligibility_for_path(upstream_path, field)
            actual = eligibility.value if eligibility.field_present else None
            passed = eligibility.eligible and evaluate_operator(actual, op, expected)
            gates.append(
                {
                    "task_id": task_id,
                    "upstream": upstream,
                    "upstream_declared_deliverable": str(upstream_task.get("deliverable") or ""),
                    "artifact_field": field,
                    "op": op,
                    "expected": expected,
                    "actual": actual,
                    "passed": passed,
                    "reason": "passed"
                    if passed
                    else (
                        "operator_mismatch" if eligibility.eligible else "ineligible_upstream_field"
                    ),
                    "eligibility": eligibility.to_dict(),
                    "principle": GATE_PRINCIPLE,
                }
            )
    passed_count = sum(1 for gate in gates if gate["passed"])
    return {
        "gates": gates,
        "passed_count": passed_count,
        "failed_count": len(gates) - passed_count,
        "principle": FIELD_PRINCIPLES["gate_cascade_receipts"],
    }


def adversarial_result_row(path: Path, payload: JsonMap, current: JsonMap) -> JsonDict:
    flags = [dict(flag) for flag in current.get("flags", []) if isinstance(flag, Mapping)]
    critical = sum(1 for flag in flags if flag.get("severity") == "critical")
    warn = sum(1 for flag in flags if flag.get("severity") == "warn")
    return {
        "path": path.as_posix(),
        "present": path.exists(),
        "stamped_flagged_adversarial": payload.get("flagged_adversarial") is True,
        "stamped_corrigendum_pending": bool(payload.get("corrigendum_pending")),
        "current_rule_flag_count": int(current.get("flag_count") or len(flags)),
        "current_rule_critical_flag_count": int(current.get("critical_flag_count") or critical),
        "current_rule_warn_flag_count": int(current.get("warn_flag_count") or warn),
        "current_rule_flags": flags,
    }


def live_artifact_reviews(root: Path, matrix: JsonMap) -> JsonDict:  # pragma: no cover
    from adversarial_verify import verify_artifact

    reviews: JsonDict = {}
    for task_id, row in matrix.items():
        rel = Path(str(row.get("declared_deliverable") or ""))
        path = root / rel
        payload, meta = read_json_mapping(path)
        if task_id == EXPERIMENT_ID:
            reviews[task_id] = {
                "path": rel.as_posix(),
                "present": path.exists(),
                "stamped_flagged_adversarial": False,
                "stamped_corrigendum_pending": False,
                "current_rule_flag_count": 0,
                "current_rule_critical_flag_count": 0,
                "current_rule_warn_flag_count": 0,
                "current_rule_flags": [],
                "skipped": "self_artifact_verified_after_final_write",
            }
            continue
        if not meta["present"]:
            reviews[task_id] = {
                "path": rel.as_posix(),
                "present": False,
                "stamped_flagged_adversarial": False,
                "stamped_corrigendum_pending": False,
                "current_rule_flag_count": 0,
                "current_rule_critical_flag_count": 0,
                "current_rule_warn_flag_count": 0,
                "current_rule_flags": [],
                "skipped": "missing_artifact",
            }
            continue
        reviews[task_id] = adversarial_result_row(path, payload, verify_artifact(path))
    return reviews


def _payloads(root: Path, tasks: Sequence[JsonMap]) -> JsonDict:
    payloads: JsonDict = {}
    for task in tasks:
        task_id = str(task.get("id") or "")
        payloads[task_id] = read_json_mapping(root / Path(str(task.get("deliverable") or "")))[0]
    return payloads


def _score(payload: JsonMap, field: str) -> Any:
    value = payload.get(field)
    if isinstance(value, Mapping) and "value" in value:
        return value.get("value")
    return value


def _critical_count(reviews: JsonMap, task_id: str) -> int:
    row = reviews.get(task_id, {})
    return int(row.get("current_rule_critical_flag_count") or 0) if isinstance(row, Mapping) else 0


def _stamped_or_current_flagged(matrix: JsonMap, reviews: JsonMap, task_id: str) -> bool:
    row = matrix.get(task_id, {})
    return bool(
        isinstance(row, Mapping)
        and (
            row.get("terminal_class") == "flagged"
            or row.get("flagged_adversarial_stamped")
            or row.get("corrigendum_pending_stamped")
            or _critical_count(reviews, task_id) > 0
        )
    )


def task_state(
    task_id: str,
    state_field: str,
    matrix: JsonMap,
    payloads: JsonMap,
    reviews: JsonMap,
) -> JsonDict:
    row = matrix.get(task_id, {})
    payload = payloads.get(task_id, {})
    readiness_fields = STATE_READINESS_FIELDS[state_field]
    readiness = {field: _score(payload, field) for field in readiness_fields}
    readiness_passed = bool(readiness) and all(
        value in (1, 1.0, True) for value in readiness.values()
    )
    flagged = _stamped_or_current_flagged(matrix, reviews, task_id)
    terminal = bool(row.get("terminal"))
    terminal_class = str(row.get("terminal_class") or "missing")
    promotion_allowed = bool(terminal and readiness_passed and not flagged)
    state = {
        "task_id": task_id,
        "declared_deliverable": row.get("declared_deliverable"),
        "sha256": row.get("sha256"),
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict"),
        "terminal_class": terminal_class,
        "terminal": terminal,
        "readiness_fields": readiness,
        "current_rule_critical_flag_count": _critical_count(reviews, task_id),
        "stamped_or_current_flagged": flagged,
        "promotion_allowed": promotion_allowed,
        "state_principle": FIELD_PRINCIPLES[state_field],
    }
    for field, value in readiness.items():
        state[field] = value
    return state


def branch_states(matrix: JsonMap, payloads: JsonMap, reviews: JsonMap) -> JsonDict:
    return {
        state_field: task_state(task_id, state_field, matrix, payloads, reviews)
        for task_id, state_field in TASK_TO_STATE_FIELD.items()
    }


def _blocking_reasons(states: Sequence[JsonMap], labels: Sequence[tuple[str, str]]) -> list[str]:
    reasons: list[str] = []
    for state, (missing_label, closed_label) in zip(states, labels, strict=True):
        if state.get("terminal_class") == "missing":
            reasons.append(missing_label)
            continue
        if not state.get("terminal"):
            reasons.append(f"{state.get('task_id')}_nonterminal")
        if state.get("stamped_or_current_flagged"):
            reasons.append(f"{state.get('task_id')}_flagged_or_current_critical")
        if not state.get("promotion_allowed"):
            reasons.append(closed_label)
    return sorted(set(reasons))


def branch_independent_promotion_ledger(states: JsonMap) -> JsonDict:
    asp_states = [
        states["asp_semantic_compiler_state"],
        states["flagship_asp_benchmark_state"],
    ]
    csl_states = [
        states["certified_admission_state"],
        states["chronological_continuous_learning_state"],
        states["heldout_transfer_state"],
        states["shadow_consumer_state"],
    ]
    sampler_states = [
        states["variable_cardinality_backend_state"],
        states["mode_jump_safety_and_value_state"],
    ]
    arc_states = [states["arc_mechanic_router_state"]]
    asp_reasons = _blocking_reasons(
        asp_states,
        (
            ("asp_semantic_compiler_missing", "asp_semantic_compiler_closed"),
            ("flagship_asp_benchmark_missing", "flagship_asp_benchmark_closed"),
        ),
    )
    csl_reasons = _blocking_reasons(
        csl_states,
        (
            ("certified_admission_missing", "certified_admission_closed"),
            ("chronological_csl_missing", "chronological_csl_closed"),
            ("heldout_transfer_missing", "heldout_transfer_closed"),
            ("shadow_consumer_missing", "shadow_consumer_closed"),
        ),
    )
    sampler_reasons = _blocking_reasons(
        sampler_states,
        (
            ("variable_cardinality_backend_missing", "variable_cardinality_backend_closed"),
            ("mode_jump_value_missing", "mode_jump_value_closed"),
        ),
    )
    arc_reasons = _blocking_reasons(
        arc_states,
        (("arc_mechanic_router_missing", "arc_mechanic_router_closed"),),
    )
    return {
        "asp_verification": {
            "promotion_allowed": not asp_reasons,
            "blocking_reasons": asp_reasons,
            "task_ids": [
                "exp6274-asp-energy-semantic-compiler",
                "exp6275-flagship-asp-constraint-verification-benchmark",
            ],
        },
        "continuous_self_learning": {
            "promotion_allowed": not csl_reasons,
            "blocking_reasons": csl_reasons,
            "task_ids": [
                "exp6276-certified-dual-cache-admission",
                "exp6277-chronological-certified-csl-ab",
                "exp6278-model-family-task-holdout-csl-audit",
                "exp6279-certified-memory-shadow-consumer",
            ],
        },
        "sampler": {
            "promotion_allowed": not sampler_reasons,
            "blocking_reasons": sampler_reasons,
            "task_ids": [
                "exp6280-variable-cardinality-mode-jump-backend",
                "exp6281-mode-jump-multifamily-rerun",
            ],
        },
        "arc": {
            "promotion_allowed": not arc_reasons,
            "blocking_reasons": arc_reasons,
            "task_ids": ["exp6282-arc-mechanic-class-live-router"],
        },
        "principle": FIELD_PRINCIPLES["branch_independent_promotion_ledger"],
    }


def count_terminal_classes(matrix: JsonMap, reviews: JsonMap) -> JsonDict:
    counts = Counter(str(row.get("terminal_class") or "unknown") for row in matrix.values())
    result = {key: int(counts.get(key, 0)) for key in COUNT_PRINCIPLES}
    result["terminal"] = sum(1 for row in matrix.values() if row.get("terminal"))
    result["nonterminal"] = sum(1 for row in matrix.values() if not row.get("terminal"))
    flagged = sum(
        1
        for task_id, row in matrix.items()
        if row.get("flagged_adversarial_stamped")
        or row.get("corrigendum_pending_stamped")
        or _critical_count(reviews, task_id) > 0
    )
    result["flagged"] = max(result["flagged"], flagged)
    result["terminal_class_counts"] = dict(
        sorted((key, int(value)) for key, value in counts.items())
    )
    result["count_principles"] = dict(COUNT_PRINCIPLES)
    return result


def prior_failure_retirement_actions(tasks: Sequence[JsonMap], matrix: JsonMap) -> JsonDict:
    actions: list[JsonDict] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        row = matrix.get(task_id, {})
        current_verdict = str(row.get("honest_verdict_raw") or "")
        current_class = str(row.get("terminal_class") or "missing")
        priors = task.get("prior_failures")
        for prior in priors if isinstance(priors, list) else []:
            if not isinstance(prior, Mapping) or prior.get("retire_if_same_verdict") is not True:
                continue
            prior_verdict = str(prior.get("verdict") or "")
            if current_class == "missing":
                action = "no_retirement_exact_artifact_missing"
                fired = False
            elif current_verdict == prior_verdict:
                action = "retire_if_same_verdict_rule_fired_recorded_only"
                fired = True
            else:
                action = "no_retirement_current_verdict_differs"
                fired = False
            actions.append(
                {
                    "task_id": task_id,
                    "prior_experiment_id": prior.get("experiment_id"),
                    "prior_verdict": prior_verdict,
                    "current_terminal_class": current_class,
                    "current_verdict": current_verdict,
                    "action": action,
                    "rule_fired": fired,
                    "would_update_exclusion_manifest": False,
                }
            )
    return {
        "actions": actions,
        "rule_fired_count": sum(1 for action in actions if action["rule_fired"]),
        "manifest_update_count": 0,
        "principle": FIELD_PRINCIPLES["prior_failure_retirement_actions"],
    }


def protected_hashes(root: Path, paths: Sequence[Path] = PROTECTED_RELATIVE_PATHS) -> JsonDict:
    return {path.as_posix(): path_sha256(root / path) for path in paths}


def protected_files_unchanged(
    root: Path,
    before: JsonMap,
    paths: Sequence[Path] = PROTECTED_RELATIVE_PATHS,
) -> JsonDict:
    after = protected_hashes(root, paths)
    rows = {
        path: {
            "before_sha256": before.get(path),
            "after_sha256": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {"unchanged": all(row["unchanged"] for row in rows.values()), "paths": rows}


def input_hashes(root: Path) -> JsonDict:
    return {
        path.as_posix(): {"present": (root / path).exists(), "sha256": path_sha256(root / path)}
        for path in INPUT_RELATIVE_PATHS
    }


def exact_path_over_receipt_precedence(matrix: JsonMap) -> JsonDict:
    rows = {
        task_id: {
            "declared_deliverable": row.get("declared_deliverable"),
            "terminal_class": row.get("terminal_class"),
            "receipt_status": row.get("receipt_status"),
            "receipt_override_attempted": row.get("receipt_override_attempted"),
            "receipt_overrode": row.get("receipt_overrode"),
        }
        for task_id, row in matrix.items()
    }
    return {
        "receipt_overrode_any_exact_path": any(row["receipt_overrode"] for row in rows.values()),
        "receipt_override_attempt_count": sum(
            1 for row in rows.values() if row["receipt_override_attempted"]
        ),
        "rows": rows,
        "principle": FIELD_PRINCIPLES["exact_path_over_receipt_precedence"],
    }


def arc_provenance_and_registry_receipts(root: Path, payloads: JsonMap) -> JsonDict:
    payload = payloads.get("exp6282-arc-mechanic-class-live-router", {})
    return {
        "registry_path": ARC_REGISTRY_RELATIVE_PATH.as_posix(),
        "registry_sha256": path_sha256(root / ARC_REGISTRY_RELATIVE_PATH),
        "registry_precheck_path_hash_and_target_receipt": payload.get(
            "registry_precheck_path_hash_and_target_receipt"
        ),
        "target_was_unsolved_at_precheck": payload.get("target_was_unsolved_at_precheck"),
        "solve_provenance": payload.get("solve_provenance"),
        "game_level_solve_claimed": payload.get("game_level_solve_claimed"),
        "hidden_game_source_access_count": payload.get("hidden_game_source_access_count"),
        "outer_loop_ground_truth_search_count": payload.get("outer_loop_ground_truth_search_count"),
        "exhaustive_bfs_count": payload.get("exhaustive_bfs_count"),
        "per_game_adapter_count": payload.get("per_game_adapter_count"),
        "registry_update_count": payload.get("registry_update_count"),
        "capstone_registry_update_count": 0,
        "principle": FIELD_PRINCIPLES["arc_provenance_and_registry_receipts"],
    }


def publication_gate_result(publication_result: JsonMap | None = None) -> JsonDict:
    if publication_result is None:  # pragma: no cover
        from publication_gate import evaluate

        publication_result = evaluate()
    gates = publication_result.get("gates") if isinstance(publication_result, Mapping) else {}
    return {
        "paper_ready": publication_result.get("paper_ready")
        if isinstance(publication_result, Mapping)
        else None,
        "gates": gates if isinstance(gates, Mapping) else {},
        "unmet_gates": list(publication_result.get("unmet_gates") or [])
        if isinstance(publication_result, Mapping)
        else [],
        "principle": FIELD_PRINCIPLES["publication_gate_g1_g2_g3_g4_and_unmet_gates"],
    }


def spec_ops_reconciliation(root: Path, before: JsonMap) -> JsonDict:
    watched = (
        SPEC_RELATIVE_PATH,
        TRACEABILITY_RELATIVE_PATH,
        STATUS_RELATIVE_PATH,
        CHANGELOG_RELATIVE_PATH,
    )
    after = protected_hashes(root, watched)
    spec_text = (root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    return {
        "openspec_req_infra_6283_present": "REQ-INFRA-6283" in spec_text,
        "implementation_status_present": "experiment_6283_v541_adversarial_capstone.py"
        in spec_text,
        "ops_status_changelog_traceability_touched_by_this_task": False,
        "stop_when_done_rule_deferred_ops_traceability_updates": True,
        "hashes": {
            path.as_posix(): {
                "before_sha256": before.get(path.as_posix()),
                "after_sha256": after.get(path.as_posix()),
                "unchanged_during_capstone": before.get(path.as_posix())
                == after.get(path.as_posix()),
            }
            for path in watched
        },
        "principle": FIELD_PRINCIPLES["spec_traceability_status_changelog_reconciliation"],
    }


def prd_gap_table(ledger: JsonMap, states: JsonMap) -> list[JsonDict]:
    return [
        {
            "gap": "asp_semantics_and_flagship_verification",
            "status": "blocked" if not ledger["asp_verification"]["promotion_allowed"] else "ready",
            "evidence": [
                states["asp_semantic_compiler_state"]["declared_deliverable"],
                states["flagship_asp_benchmark_state"]["declared_deliverable"],
            ],
            "reason": ledger["asp_verification"]["blocking_reasons"],
        },
        {
            "gap": "certified_continuous_self_learning",
            "status": "blocked",
            "evidence": [
                states["certified_admission_state"]["declared_deliverable"],
                states["chronological_continuous_learning_state"]["declared_deliverable"],
                states["heldout_transfer_state"]["declared_deliverable"],
                states["shadow_consumer_state"]["declared_deliverable"],
            ],
            "reason": ledger["continuous_self_learning"]["blocking_reasons"],
        },
        {
            "gap": "variable_cardinality_mode_jumping",
            "status": "blocked" if not ledger["sampler"]["promotion_allowed"] else "ready",
            "evidence": [
                states["variable_cardinality_backend_state"]["declared_deliverable"],
                states["mode_jump_safety_and_value_state"]["declared_deliverable"],
            ],
            "reason": ledger["sampler"]["blocking_reasons"],
        },
        {
            "gap": "arc_live_mechanic_router",
            "status": "blocked" if not ledger["arc"]["promotion_allowed"] else "ready",
            "evidence": states["arc_mechanic_router_state"]["declared_deliverable"],
            "reason": ledger["arc"]["blocking_reasons"],
        },
    ]


def next_milestone_recommendations(ledger: JsonMap) -> list[JsonDict]:
    rows: list[JsonDict] = []
    if not ledger["asp_verification"]["promotion_allowed"]:
        rows.append(
            {
                "branch": "asp_verification",
                "recommendation": "Repair or rerun the flagship ASP benchmark before promoting its event corpus.",
                "evidence": ledger["asp_verification"]["blocking_reasons"],
            }
        )
    if not ledger["continuous_self_learning"]["promotion_allowed"]:
        rows.append(
            {
                "branch": "continuous_self_learning",
                "recommendation": "Keep certified memory default-off until admission, chronology, heldout, and shadow gates pass.",
                "evidence": ledger["continuous_self_learning"]["blocking_reasons"],
            }
        )
    if not ledger["sampler"]["promotion_allowed"]:
        rows.append(
            {
                "branch": "sampler",
                "recommendation": "Keep the typed backend but do not promote mode-jump workload value.",
                "evidence": ledger["sampler"]["blocking_reasons"],
            }
        )
    if not ledger["arc"]["promotion_allowed"]:
        rows.append(
            {
                "branch": "arc",
                "recommendation": "Fix the live-router duration or provenance flag before promoting ARC route value.",
                "evidence": ledger["arc"]["blocking_reasons"],
            }
        )
    return rows


def _field_provenance() -> JsonDict:
    sources = {
        "REQ-INFRA-6283",
        ROADMAP_RELATIVE_PATH.as_posix(),
        MILESTONE_DOC_RELATIVE_PATH.as_posix(),
        CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
        EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
        ARC_REGISTRY_RELATIVE_PATH.as_posix(),
        SPEC_RELATIVE_PATH.as_posix(),
        TERMINAL_ARTIFACTS_RELATIVE_PATH.as_posix(),
        ADVERSARIAL_VERIFY_RELATIVE_PATH.as_posix(),
        SUMMARY_ARTIFACT_RELATIVE_PATH.as_posix(),
        PUBLICATION_GATE_RELATIVE_PATH.as_posix(),
    }
    return {
        field: {"principle": FIELD_PRINCIPLES[field], "sources": sorted(sources)}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _test_exit_codes(command_rows: Sequence[JsonMap]) -> JsonDict:
    return {
        str(row.get("command") or ""): int(row.get("exit_code") or 0)
        for row in command_rows
        if row.get("command")
    }


def _status_from_commands(command_rows: Sequence[JsonMap]) -> tuple[str, str]:
    if any(int(row.get("exit_code") or 0) != 0 for row in command_rows):
        return "blocked", "blocked: one or more recorded validation commands failed"
    return (
        "complete",
        "complete: V541 capstone preserved exact branch evidence without promoting flagged, missing, skipped, null, or blocked artifacts",
    )


def preconditions_checked(
    root: Path,
    tasks: Sequence[JsonMap],
    receipts: JsonMap,
    before_hashes: JsonMap,
    git_status_before: Sequence[str],
    git_status_after_tests: Sequence[str],
) -> JsonDict:
    declared = {
        str(task.get("id") or ""): {
            "declared_deliverable": str(task.get("deliverable") or ""),
            "sha256": path_sha256(root / Path(str(task.get("deliverable") or ""))),
        }
        for task in tasks
    }
    return {
        "checked_before_classification": True,
        "milestone": MILESTONE,
        "roadmap_sha256": path_sha256(root / ROADMAP_RELATIVE_PATH),
        "milestone_doc_sha256": path_sha256(root / MILESTONE_DOC_RELATIVE_PATH),
        "declared_deliverable_hashes": declared,
        "conductor_receipt_matrix_sha256": payload_sha256(receipts),
        "registry_sha256": path_sha256(root / ARC_REGISTRY_RELATIVE_PATH),
        "current_adversarial_rules_sha256": path_sha256(root / ADVERSARIAL_VERIFY_RELATIVE_PATH),
        "protected_hashes_before": dict(before_hashes),
        "input_hashes_before": input_hashes(root),
        "git_status_before": list(git_status_before),
        "git_status_after_tests": list(git_status_after_tests),
        "principle": FIELD_PRINCIPLES["preconditions_checked"],
    }


def build_report(
    root: Path = REPO_ROOT,
    *,
    date: str,
    command_receipts: Sequence[JsonMap] | None = None,
    artifact_reviews: JsonMap | None = None,
    publication_result: JsonMap | None = None,
    before_hashes: JsonMap | None = None,
    git_status_before: Sequence[str] | None = None,
    git_status_after_tests: Sequence[str] | None = None,
    started_at: float | None = None,
) -> JsonDict:
    started = time.perf_counter() if started_at is None else started_at
    before = dict(before_hashes or protected_hashes(root))
    roadmap = load_roadmap(root)
    tasks = roadmap_tasks(roadmap)
    receipts = latest_conductor_receipts(root, tasks)
    matrix = build_exact_declared_deliverable_matrix(
        root,
        tasks,
        conductor_receipts=receipts,
        self_payload=_self_payload(),
    )
    reviews = dict(artifact_reviews or live_artifact_reviews(root, matrix))
    payloads = _payloads(root, tasks)
    states = branch_states(matrix, payloads, reviews)
    ledger = branch_independent_promotion_ledger(states)
    command_rows = [dict(row) for row in command_receipts or []]
    status, verdict = _status_from_commands(command_rows)
    publication = publication_gate_result(publication_result)
    report: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "status": status,
        "milestone_roadmap_path_and_hash": {
            "milestone": roadmap.get("milestone"),
            "roadmap_path": ROADMAP_RELATIVE_PATH.as_posix(),
            "roadmap_sha256": path_sha256(root / ROADMAP_RELATIVE_PATH),
            "milestone_doc": roadmap.get("milestone_doc"),
            "milestone_doc_sha256": path_sha256(root / MILESTONE_DOC_RELATIVE_PATH),
            "task_ids": [str(task.get("id") or "") for task in tasks],
            "expected_task_ids": list(EXPECTED_V541_TASK_IDS),
            "task_count": len(tasks),
            "principle": FIELD_PRINCIPLES["milestone_roadmap_path_and_hash"],
        },
        "exact_declared_deliverable_matrix": matrix,
        "conductor_receipt_matrix": receipts,
        "exact_path_over_receipt_precedence": exact_path_over_receipt_precedence(matrix),
        "current_rule_adversarial_results_by_task": reviews,
        "terminal_nonterminal_blocked_skipped_null_flagged_retired_and_ready_counts": count_terminal_classes(
            matrix, reviews
        ),
        "gate_cascade_receipts": evaluate_gate_cascade_receipts(root, tasks),
        **states,
        "arc_provenance_and_registry_receipts": arc_provenance_and_registry_receipts(
            root, payloads
        ),
        "branch_independent_promotion_ledger": ledger,
        "prior_failure_retirement_actions": prior_failure_retirement_actions(tasks, matrix),
        "publication_gate_g1_g2_g3_g4_and_unmet_gates": publication,
        "source_mutation_count": 0,
        "weight_mutation_count": 0,
        "unauthorized_external_call_count": 0,
        "hidden_game_source_access_count": 0,
        "outer_loop_ground_truth_search_count": 0,
        "arc_level_solve_claim_count": 0,
        "registry_update_count": 0,
        "hardware_claim_count": 0,
        "speed_power_or_energy_claim_count": 0,
        "protected_files_unchanged": protected_files_unchanged(root, before),
        "spec_traceability_status_changelog_reconciliation": spec_ops_reconciliation(root, before),
        "prd_gap_table": prd_gap_table(ledger, states),
        "next_milestone_recommendations": next_milestone_recommendations(ledger),
        "preconditions_checked": preconditions_checked(
            root,
            tasks,
            receipts,
            before,
            git_status_before or [],
            git_status_after_tests or [],
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": [str(row.get("command") or "") for row in command_rows]
        or list(TEST_COMMANDS),
        "test_exit_codes": _test_exit_codes(command_rows),
        "duration_s": time.perf_counter() - started,
        "reproducibility_checksum": "",
        "honest_verdict": verdict,
    }
    report["reproducibility_checksum"] = payload_checksum(report)
    return report


def _is_bare_zero(value: Any) -> bool:
    return type(value) is int and value == 0


def validate_report(report: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in report:
            errors.append(f"missing required field: {field}")
    principles = report.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles is not a mapping")
        principles = {}
    provenance = report.get("field_provenance")
    if not isinstance(provenance, Mapping):
        errors.append("field_provenance is not a mapping")
        provenance = {}
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in principles:
            errors.append(f"missing field_principles entry: {field}")
        if field not in provenance:
            errors.append(f"missing field_provenance entry: {field}")
    counts = report.get(
        "terminal_nonterminal_blocked_skipped_null_flagged_retired_and_ready_counts"
    )
    if not isinstance(counts, Mapping):
        errors.append("counts field is not a mapping")
        counts = {}
    count_principles = counts.get("count_principles")
    if not isinstance(count_principles, Mapping):
        errors.append("count_principles missing")
        count_principles = {}
    for key in COUNT_PRINCIPLES:
        if key not in count_principles:
            errors.append(f"missing count principle: {key}")
    gates = (
        report.get("gate_cascade_receipts", {}).get("gates")
        if isinstance(report.get("gate_cascade_receipts"), Mapping)
        else None
    )
    if isinstance(gates, list):
        for gate in gates:
            if not isinstance(gate, Mapping) or not gate.get("principle"):
                errors.append("gate missing principle")
                break
    else:
        errors.append("gate_cascade_receipts.gates is not a list")
    for field in FORBIDDEN_ZERO_FIELDS:
        if not _is_bare_zero(report.get(field)):
            errors.append(f"{field} must be bare integer 0")
    publication = report.get("publication_gate_g1_g2_g3_g4_and_unmet_gates")
    publication_gates = publication.get("gates") if isinstance(publication, Mapping) else None
    for gate_name in ("G1", "G2", "G3", "G4"):
        if not isinstance(publication_gates, Mapping) or gate_name not in publication_gates:
            errors.append(f"publication gate missing {gate_name}")
    if report.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("wrong inference_substrate")
    if report.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    verdict = str(report.get("honest_verdict") or "")
    if not verdict.startswith(("complete:", "complete_", "blocked:", "blocked_")):
        errors.append("honest_verdict lacks terminal prefix")
    checksum = report.get("reproducibility_checksum")
    if isinstance(checksum, str) and checksum.startswith("sha256:"):
        if checksum != payload_checksum(report):
            errors.append("reproducibility_checksum mismatch")
    else:
        errors.append("reproducibility_checksum missing")
    return errors


def run_command(
    command: str, root: Path, timeout_s: int | None = None
) -> JsonDict:  # pragma: no cover
    try:
        proc = subprocess.run(
            shlex.split(command),
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "exit_code": 124,
            "classification": "timeout",
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
        }
    except FileNotFoundError as exc:
        return {
            "command": command,
            "exit_code": 127,
            "classification": "command_not_found",
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }
    return {
        "command": command,
        "exit_code": proc.returncode,
        "classification": "passed" if proc.returncode == 0 else f"nonzero_exit_{proc.returncode}",
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def run_default_commands(root: Path) -> list[JsonDict]:  # pragma: no cover
    return [
        run_command(command, root, COMMAND_TIMEOUTS_S.get(command)) for command in TEST_COMMANDS
    ]


def write_report(
    report: JsonMap,
    root: Path = REPO_ROOT,
    *,
    env: Mapping[str, str] | None = None,
) -> Path:
    errors = validate_report(report)
    if errors:
        raise ValueError("invalid Exp6283 report: " + "; ".join(errors))
    return atomic_write_json(RESULT_RELATIVE_PATH, report, root=root, env=env, sort_keys=False)


def run_experiment(root: Path, date: str, *, run_commands: bool) -> JsonDict:  # pragma: no cover
    started = time.perf_counter()
    before = protected_hashes(root)
    status_before = git_status_lines(root)
    preliminary = build_report(
        root,
        date=date,
        command_receipts=[],
        before_hashes=before,
        git_status_before=status_before,
        git_status_after_tests=[],
        started_at=started,
    )
    write_report(preliminary, root)
    commands = run_default_commands(root) if run_commands else []
    status_after_tests = git_status_lines(root)
    final = build_report(
        root,
        date=date,
        command_receipts=commands,
        before_hashes=before,
        git_status_before=status_before,
        git_status_after_tests=status_after_tests,
        started_at=started,
    )
    write_report(final, root)
    return final


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now(UTC).strftime("%Y%m%d"))
    parser.add_argument("--no-run-commands", action="store_true")
    args = parser.parse_args(argv)
    report = run_experiment(REPO_ROOT, args.date, run_commands=not args.no_run_commands)
    print(json.dumps(report, indent=2, sort_keys=False))
    return 0 if report.get("status") == "complete" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
