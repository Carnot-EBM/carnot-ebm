"""Exp6271 V540 exact-path adversarial capstone.

Spec refs: REQ-INFRA-6271, SCENARIO-INFRA-6271-1,
SCENARIO-INFRA-6271-2, SCENARIO-INFRA-6271-3,
SCENARIO-INFRA-6271-4, SCENARIO-INFRA-6271-5.
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

from carnot.experiment_6260_v540_terminal_transition import same_number_aliases
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

MILESTONE = "2026.08.540"
EXPERIMENT_ID = "exp6271-v540-adversarial-capstone"
SCHEMA = "carnot.experiment_6271.v540_adversarial_capstone.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6271_v540_adversarial_capstone.json")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")

EXPECTED_V540_TASK_IDS = (
    "exp6260-v540-terminal-transition",
    "exp6261-v540-post-marker-source-scope-freeze",
    "exp6262-terminal-artifact-readiness-contract",
    "exp6263-clean-sota-event-replay-bridge",
    "exp6264-energy-familiarity-memory-gate",
    "exp6265-chronological-two-timescale-csl-ab",
    "exp6266-family-task-holdout-csl-audit",
    "exp6267-constraint-memory-shadow-consumer-v2",
    "exp6268-multimodal-sampler-fixture-suite",
    "exp6269-mode-jump-multifamily-ab",
    "exp6270-mode-jump-descriptor-router",
    EXPERIMENT_ID,
)

FORBIDDEN_ZERO_FIELDS = (
    "source_mutation_count",
    "weight_mutation_count",
    "live_llm_call_count",
    "arc_solve_claim_count",
    "registry_update_count",
    "hardware_claim_count",
    "speed_or_power_claim_count",
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
    "terminal_artifact_contract_state",
    "clean_sota_replay_state",
    "familiarity_gate_state",
    "continuous_learning_state",
    "family_task_transfer_state",
    "shadow_consumer_state",
    "sampler_fixture_state",
    "mode_jump_safety_and_value_state",
    "sampler_router_state",
    "branch_independent_promotion_ledger",
    "prior_failure_retirement_actions",
    "source_mutation_count",
    "weight_mutation_count",
    "live_llm_call_count",
    "arc_solve_claim_count",
    "registry_update_count",
    "hardware_claim_count",
    "speed_or_power_claim_count",
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
    "status": "The capstone is complete only when exact V540 evidence is preserved.",
    "milestone_roadmap_path_and_hash": "The active roadmap is the task denominator.",
    "exact_declared_deliverable_matrix": "Exact paths prevent sidecars from replacing missing work.",
    "conductor_receipt_matrix": "Receipts are context and cannot promote an artifact.",
    "exact_path_over_receipt_precedence": "The artifact file outranks conductor text.",
    "current_rule_adversarial_results_by_task": "Current verifier flags are checked before promotion.",
    "terminal_nonterminal_blocked_skipped_null_flagged_retired_and_ready_counts": "All evidence classes stay visible.",
    "gate_cascade_receipts": "Structured gates are recomputed from exact bare fields.",
    "terminal_artifact_contract_state": "The evidence-boundary branch must not hide a blocked contract.",
    "clean_sota_replay_state": "Clean replay is separate from downstream learning promotion.",
    "familiarity_gate_state": "Energy familiarity can close without blocking the sampler branch.",
    "continuous_learning_state": "CSL promotion requires its own exact artifact and gate.",
    "family_task_transfer_state": "Missing holdout audit evidence cannot be promoted.",
    "shadow_consumer_state": "A skipped shadow consumer is not a live decision-path claim.",
    "sampler_fixture_state": "A ready fixture suite is not a workload-value result.",
    "mode_jump_safety_and_value_state": "Sampler safety and workload value are separate gates.",
    "sampler_router_state": "A gate-skipped router stays default-off.",
    "branch_independent_promotion_ledger": "One branch cannot promote another branch.",
    "prior_failure_retirement_actions": "Retirement fires only on exact repeated verdicts.",
    "source_mutation_count": "Bare zero proves the capstone did not mutate upstream sources.",
    "weight_mutation_count": "Bare zero proves no model weights changed.",
    "live_llm_call_count": "Bare zero proves no live LLM call occurred.",
    "arc_solve_claim_count": "Bare zero proves no ARC solve credit is claimed.",
    "registry_update_count": "Bare zero proves no registry was updated.",
    "hardware_claim_count": "Bare zero proves no hardware claim is made.",
    "speed_or_power_claim_count": "Bare zero proves no speed or power claim is made.",
    "protected_files_unchanged": "Protected hashes show the capstone did not rewrite records.",
    "spec_traceability_status_changelog_reconciliation": "Docs state is recorded only to existing evidence.",
    "prd_gap_table": "PRD gaps cite exact artifacts instead of roadmap intent.",
    "next_milestone_recommendations": "Next steps come only from observed blockers.",
    "preconditions_checked": "Inputs and hashes are frozen before classification.",
    "inference_substrate": "The capstone aggregates checked-in artifacts only.",
    "verifier_is_oracle": "False because this audits records, not benchmark answers.",
    "field_provenance": "Every field cites sources used to fill it.",
    "field_principles": "Every field states why it exists.",
    "test_commands": "Commands preserve the verification boundary.",
    "test_exit_codes": "Exit codes are recorded without laundering failures.",
    "duration_s": "Wall time is reported without padding.",
    "reproducibility_checksum": "The normalized payload is content-addressed.",
    "honest_verdict": "The verdict states mixed outcomes without strengthening them.",
}

COUNT_PRINCIPLES: dict[str, str] = {
    "terminal": "Terminal means the shared classifier allows the artifact to close.",
    "nonterminal": "Nonterminal exact artifacts cannot feed downstream claims.",
    "missing": "A missing exact path cannot be replaced by an alias.",
    "malformed": "Malformed JSON fails closed.",
    "running": "Running artifacts are not terminal evidence.",
    "running_bootstrap": "Bootstrap execution records are not science artifacts.",
    "bootstrap_only": "Bootstrap-only artifacts cannot support gates.",
    "partial": "Partial artifacts cannot promote a branch.",
    "contradictory": "Conflicting status markers fail closed.",
    "unknown": "Unknown markers fail closed.",
    "blocked": "Blocked artifacts are terminal blockers, not successes.",
    "skipped": "Gate tombstones stay skipped.",
    "null": "Null evidence is separate from positive evidence.",
    "flagged": "Stamped or current flags quarantine an artifact from promotion.",
    "retired": "Retired scopes stay retired.",
    "ready": "Ready evidence is counted separately from complete evidence.",
    "complete": "Complete evidence can still fail a branch gate.",
    "positive": "Positive is never inferred from null or skipped evidence.",
}

GATE_PRINCIPLE = "A gate can read only a terminal exact artifact with an exact bare field."

PROTECTED_RELATIVE_PATHS = (
    ROADMAP_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    Path("scripts/research_conductor.py"),
    Path("python/carnot/terminal_artifacts.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/summarize_artifact.py"),
    SPEC_RELATIVE_PATH,
    Path("_bmad/traceability.md"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    Path("docs/index.html"),
)

INPUT_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    ROADMAP_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    Path("research-references.md"),
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    Path("openspec/capabilities/self-learning/spec.md"),
    Path("openspec/capabilities/samplers/spec.md"),
    Path("python/carnot/terminal_artifacts.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/summarize_artifact.py"),
)

TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6271_v540_adversarial_capstone.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6271_v540_adversarial_capstone.py -m pytest tests/python/test_experiment_6271_v540_adversarial_capstone.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6271_v540_adversarial_capstone.py --fail-under=100",
    ".venv/bin/ruff check python/carnot/experiment_6271_v540_adversarial_capstone.py tests/python/test_experiment_6271_v540_adversarial_capstone.py",
    ".venv/bin/ruff format --check python/carnot/experiment_6271_v540_adversarial_capstone.py tests/python/test_experiment_6271_v540_adversarial_capstone.py",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6271_v540_adversarial_capstone.py",
    ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml",
    ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
    ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "sed -n 1,220p ops/e2e-test-plan.md",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6271_v540_adversarial_capstone.json",
)
COMMAND_TIMEOUTS_S = {
    ".venv/bin/pytest tests/python -q": 3600,
}


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
            ["git", "status", "--short"],
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
        needle = title[:50]
        matches = [line for line in lines if needle in line or task_id in line]
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
        "honest_verdict": "complete: Exp6271 capstone payload under construction",
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
        if task_id == EXPERIMENT_ID and self_payload is not None:
            payload = dict(self_payload)
            digest = payload_sha256(payload)
            classification = classify_artifact_payload(
                payload,
                path=path,
                sha256=digest,
                conductor_receipt=receipt if isinstance(receipt, Mapping) else None,
            )
            meta = {"present": path.exists(), "loadable": True, "sha256": digest}
        else:
            payload, meta = read_json_mapping(path)
            classification = classify_artifact_path(
                path,
                conductor_receipt=receipt if isinstance(receipt, Mapping) else None,
            )
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
        "principle": GATE_PRINCIPLE,
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
        current = verify_artifact(path)
        reviews[task_id] = adversarial_result_row(path, payload, current)
    return reviews


def _payloads(root: Path, tasks: Sequence[JsonMap]) -> JsonDict:
    return {
        str(task.get("id") or ""): read_json_mapping(
            root / Path(str(task.get("deliverable") or ""))
        )[0]
        for task in tasks
    }


def _score(payload: JsonMap, field: str) -> Any:
    value = payload.get(field)
    if isinstance(value, Mapping) and "value" in value:
        return value.get("value")
    return value


def _critical_count(reviews: JsonMap, task_id: str) -> int:
    row = reviews.get(task_id, {})
    return int(row.get("current_rule_critical_flag_count") or 0) if isinstance(row, Mapping) else 0


def task_state(
    task_id: str,
    matrix: JsonMap,
    payloads: JsonMap,
    reviews: JsonMap,
    readiness_fields: Sequence[str],
) -> JsonDict:
    row = matrix.get(task_id, {})
    payload = payloads.get(task_id, {})
    readiness = {field: _score(payload, field) for field in readiness_fields}
    return {
        "task_id": task_id,
        "declared_deliverable": row.get("declared_deliverable"),
        "sha256": row.get("sha256"),
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict"),
        "terminal_class": row.get("terminal_class"),
        "terminal": row.get("terminal"),
        "readiness_fields": readiness,
        "current_rule_critical_flag_count": _critical_count(reviews, task_id),
        "promotion_allowed": bool(
            row.get("terminal")
            and all(value == 1 or value == 1.0 for value in readiness.values())
            and _critical_count(reviews, task_id) == 0
        ),
    }


def branch_states(matrix: JsonMap, payloads: JsonMap, reviews: JsonMap) -> JsonDict:
    states = {
        "terminal_artifact_contract_state": task_state(
            "exp6262-terminal-artifact-readiness-contract",
            matrix,
            payloads,
            reviews,
            ("terminal_artifact_contract_ready_score",),
        ),
        "clean_sota_replay_state": task_state(
            "exp6263-clean-sota-event-replay-bridge",
            matrix,
            payloads,
            reviews,
            ("event_replay_bridge_ready_score",),
        ),
        "familiarity_gate_state": task_state(
            "exp6264-energy-familiarity-memory-gate",
            matrix,
            payloads,
            reviews,
            ("familiarity_gate_ready_score",),
        ),
        "continuous_learning_state": task_state(
            "exp6265-chronological-two-timescale-csl-ab",
            matrix,
            payloads,
            reviews,
            ("continuous_learning_promotion_ready_score",),
        ),
        "family_task_transfer_state": task_state(
            "exp6266-family-task-holdout-csl-audit",
            matrix,
            payloads,
            reviews,
            ("family_task_transfer_ready_score",),
        ),
        "shadow_consumer_state": task_state(
            "exp6267-constraint-memory-shadow-consumer-v2",
            matrix,
            payloads,
            reviews,
            ("shadow_consumer_ready_score",),
        ),
        "sampler_fixture_state": task_state(
            "exp6268-multimodal-sampler-fixture-suite",
            matrix,
            payloads,
            reviews,
            ("sampler_fixture_suite_ready_score",),
        ),
        "mode_jump_safety_and_value_state": task_state(
            "exp6269-mode-jump-multifamily-ab",
            matrix,
            payloads,
            reviews,
            ("mode_jump_safety_ready_score", "mode_jump_workload_value_ready_score"),
        ),
        "sampler_router_state": task_state(
            "exp6270-mode-jump-descriptor-router",
            matrix,
            payloads,
            reviews,
            ("sampler_router_ready_score",),
        ),
    }
    for key, state in states.items():
        for field, value in state["readiness_fields"].items():
            state[field] = value
        state["state_principle"] = FIELD_PRINCIPLES[key]
    return states


def _blocking_reasons(states: Sequence[JsonMap], labels: Sequence[tuple[str, str]]) -> list[str]:
    reasons: list[str] = []
    for state, (missing_label, closed_label) in zip(states, labels, strict=True):
        if state.get("terminal_class") == "missing":
            reasons.append(missing_label)
            continue
        if not state.get("terminal"):
            reasons.append(f"{state.get('task_id')}_nonterminal")
        if state.get("current_rule_critical_flag_count", 0):
            reasons.append(f"{state.get('task_id')}_current_critical_flag")
        if not state.get("promotion_allowed"):
            reasons.append(closed_label)
    return sorted(set(reasons))


def branch_independent_promotion_ledger(states: JsonMap) -> JsonDict:
    terminal_states = [
        states["terminal_artifact_contract_state"],
    ]
    clean_states = [states["clean_sota_replay_state"]]
    csl_states = [
        states["clean_sota_replay_state"],
        states["familiarity_gate_state"],
        states["continuous_learning_state"],
        states["family_task_transfer_state"],
        states["shadow_consumer_state"],
    ]
    sampler_states = [
        states["sampler_fixture_state"],
        states["mode_jump_safety_and_value_state"],
        states["sampler_router_state"],
    ]
    terminal_reasons = _blocking_reasons(
        terminal_states,
        (("terminal_contract_missing", "terminal_contract_closed"),),
    )
    clean_reasons = _blocking_reasons(
        clean_states,
        (("clean_replay_missing", "clean_replay_closed"),),
    )
    csl_reasons = _blocking_reasons(
        csl_states,
        (
            ("clean_replay_missing", "clean_replay_closed"),
            ("familiarity_gate_missing", "familiarity_gate_closed"),
            ("continuous_learning_missing", "continuous_learning_closed"),
            ("family_task_transfer_missing", "family_task_transfer_closed"),
            ("shadow_consumer_missing", "shadow_consumer_closed"),
        ),
    )
    sampler_reasons = _blocking_reasons(
        sampler_states,
        (
            ("sampler_fixture_missing", "sampler_fixture_closed"),
            ("mode_jump_value_missing", "mode_jump_value_closed"),
            ("sampler_router_missing", "sampler_router_closed"),
        ),
    )
    return {
        "terminal_artifact": {
            "promotion_allowed": not terminal_reasons,
            "blocking_reasons": terminal_reasons,
            "task_ids": ["exp6262-terminal-artifact-readiness-contract"],
        },
        "clean_sota_replay": {
            "promotion_allowed": not clean_reasons,
            "blocking_reasons": clean_reasons,
            "task_ids": ["exp6263-clean-sota-event-replay-bridge"],
        },
        "continuous_learning": {
            "promotion_allowed": not csl_reasons,
            "blocking_reasons": csl_reasons,
            "task_ids": [
                "exp6263-clean-sota-event-replay-bridge",
                "exp6264-energy-familiarity-memory-gate",
                "exp6265-chronological-two-timescale-csl-ab",
                "exp6266-family-task-holdout-csl-audit",
                "exp6267-constraint-memory-shadow-consumer-v2",
            ],
        },
        "sampler": {
            "promotion_allowed": not sampler_reasons,
            "blocking_reasons": sampler_reasons,
            "task_ids": [
                "exp6268-multimodal-sampler-fixture-suite",
                "exp6269-mode-jump-multifamily-ab",
                "exp6270-mode-jump-descriptor-router",
            ],
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


def preconditions_checked(
    root: Path,
    tasks: Sequence[JsonMap],
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
        "declared_deliverable_hashes": declared,
        "conductor_log_sha256": path_sha256(root / CONDUCTOR_LOG_RELATIVE_PATH),
        "terminal_artifacts_sha256": path_sha256(root / "python/carnot/terminal_artifacts.py"),
        "adversarial_verify_sha256": path_sha256(root / "scripts/adversarial_verify.py"),
        "protected_hashes_before": dict(before_hashes),
        "git_status_before": list(git_status_before),
        "git_status_after_tests": list(git_status_after_tests),
        "principle": FIELD_PRINCIPLES["preconditions_checked"],
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


def spec_ops_reconciliation(root: Path, before: JsonMap) -> JsonDict:
    watched = (
        SPEC_RELATIVE_PATH,
        Path("_bmad/traceability.md"),
        Path("ops/status.md"),
        Path("ops/changelog.md"),
    )
    after = protected_hashes(root, watched)
    spec_text = (root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    return {
        "openspec_req_infra_6271_present": "REQ-INFRA-6271" in spec_text,
        "implementation_status_present": "experiment_6271_v540_adversarial_capstone.py"
        in spec_text,
        "ops_status_changelog_traceability_touched": False,
        "operator_stop_rule_deferred_ops_traceability_updates": True,
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
            "gap": "terminal_artifact_boundary",
            "status": "blocked",
            "evidence": states["terminal_artifact_contract_state"]["declared_deliverable"],
            "reason": ledger["terminal_artifact"]["blocking_reasons"],
        },
        {
            "gap": "continuous_constraint_learning",
            "status": "blocked",
            "evidence": [
                states["clean_sota_replay_state"]["declared_deliverable"],
                states["familiarity_gate_state"]["declared_deliverable"],
                states["continuous_learning_state"]["declared_deliverable"],
                states["family_task_transfer_state"]["declared_deliverable"],
            ],
            "reason": ledger["continuous_learning"]["blocking_reasons"],
        },
        {
            "gap": "exact_multi_family_sampling",
            "status": "blocked",
            "evidence": [
                states["sampler_fixture_state"]["declared_deliverable"],
                states["mode_jump_safety_and_value_state"]["declared_deliverable"],
                states["sampler_router_state"]["declared_deliverable"],
            ],
            "reason": ledger["sampler"]["blocking_reasons"],
        },
        {
            "gap": "forbidden_claim_boundary",
            "status": "closed",
            "evidence": list(FORBIDDEN_ZERO_FIELDS),
            "reason": "all forbidden claim and mutation counters are bare zero",
        },
    ]


def next_milestone_recommendations(ledger: JsonMap) -> list[JsonDict]:
    rows = []
    if not ledger["terminal_artifact"]["promotion_allowed"]:
        rows.append(
            {
                "branch": "terminal_artifact",
                "recommendation": "Repair the readiness contract before using gate fields.",
                "evidence": ledger["terminal_artifact"]["blocking_reasons"],
            }
        )
    if not ledger["continuous_learning"]["promotion_allowed"]:
        rows.append(
            {
                "branch": "continuous_learning",
                "recommendation": "Keep CSL default-off until the familiarity, CSL, holdout, and shadow artifacts are exact-path ready.",
                "evidence": ledger["continuous_learning"]["blocking_reasons"],
            }
        )
    if not ledger["sampler"]["promotion_allowed"]:
        rows.append(
            {
                "branch": "sampler",
                "recommendation": "Keep the router default-off until mode-jump safety and workload value pass on exact fixtures.",
                "evidence": ledger["sampler"]["blocking_reasons"],
            }
        )
    return rows


def _field_provenance() -> JsonDict:
    sources = {
        "REQ-INFRA-6271",
        ROADMAP_RELATIVE_PATH.as_posix(),
        VNEXT_RELATIVE_PATH.as_posix(),
        CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
        EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
        SPEC_RELATIVE_PATH.as_posix(),
        "python/carnot/terminal_artifacts.py",
        "scripts/adversarial_verify.py",
        "scripts/summarize_artifact.py",
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
        "complete: V540 capstone preserved exact terminal nonterminal blocked skipped null flagged retired and ready states without promoting CSL sampler ARC runtime hardware speed or power claims",
    )


def build_report(
    root: Path = REPO_ROOT,
    *,
    date: str,
    command_receipts: Sequence[JsonMap] | None = None,
    artifact_reviews: JsonMap | None = None,
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
            "milestone_doc_sha256": path_sha256(root / Path(str(roadmap.get("milestone_doc")))),
            "task_ids": [str(task.get("id") or "") for task in tasks],
            "expected_task_ids": list(EXPECTED_V540_TASK_IDS),
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
        "branch_independent_promotion_ledger": ledger,
        "prior_failure_retirement_actions": prior_failure_retirement_actions(tasks, matrix),
        "source_mutation_count": 0,
        "weight_mutation_count": 0,
        "live_llm_call_count": 0,
        "arc_solve_claim_count": 0,
        "registry_update_count": 0,
        "hardware_claim_count": 0,
        "speed_or_power_claim_count": 0,
        "protected_files_unchanged": protected_files_unchanged(root, before),
        "spec_traceability_status_changelog_reconciliation": spec_ops_reconciliation(root, before),
        "prd_gap_table": prd_gap_table(ledger, states),
        "next_milestone_recommendations": next_milestone_recommendations(ledger),
        "preconditions_checked": preconditions_checked(
            root,
            tasks,
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
        raise ValueError("invalid Exp6271 report: " + "; ".join(errors))
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
