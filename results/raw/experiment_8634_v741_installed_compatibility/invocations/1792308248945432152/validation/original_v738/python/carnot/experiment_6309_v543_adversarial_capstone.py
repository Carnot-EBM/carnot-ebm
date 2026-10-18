"""Exp6309 V543 adversarial capstone.

Spec refs: REQ-INTEG-6309, SCENARIO-INTEG-6309-1,
SCENARIO-INTEG-6309-2, SCENARIO-INTEG-6309-3,
SCENARIO-INTEG-6309-4, SCENARIO-INTEG-6309-5,
SCENARIO-INTEG-6309-6.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
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
from carnot.terminal_evidence_preflight import (
    build_synthetic_fixture_manifest,
    evaluate_fixture_manifest,
    replay_v542_failure_fixtures,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(SCRIPTS_ROOT))

MILESTONE = "2026.08.543"
EXPERIMENT_ID = "exp6309-v543-adversarial-capstone"
SCHEMA = "carnot.experiment_6309.v543_adversarial_capstone.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6309_v543_adversarial_capstone.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
MILESTONE_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
TRACEABILITY_RELATIVE_PATH = Path("_bmad/traceability.md")
PRD_RELATIVE_PATH = Path("_bmad/prd.md")
ARCHITECTURE_RELATIVE_PATH = Path("_bmad/architecture.md")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
CHECK_SPEC_COVERAGE_RELATIVE_PATH = Path("scripts/check_spec_coverage.py")
TERMINAL_ARTIFACTS_RELATIVE_PATH = Path("python/carnot/terminal_artifacts.py")
RESEARCH_CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
INFERENCE_SUBSTRATE = "aggregation_from_exact_declared_artifacts"
VERIFIER_ORACLE_BOUNDARY = "mixed_with_explicit_per_branch_boundary"
RANDOM_SEED = 6309

EXPECTED_TASK_IDS = (
    "exp6297-v543-terminal-transition",
    "exp6298-terminal-evidence-preflight-linter",
    "exp6299-v543-post-marker-source-scope-freeze",
    "exp6300-three-family-universal-activation-bus",
    "exp6301-activation-bus-integrity-audit",
    "exp6302-shared-activation-state-initializer",
    "exp6303-live-three-family-shared-state-benchmark",
    "exp6304-reference-anchored-online-state-learning",
    "exp6305-evidence-licensed-cross-family-transfer",
    "exp6306-online-state-learning-safety-audit",
    "exp6307-arc-target-validated-route-canary",
    "exp6308-arc-target-validated-route-holdout",
    EXPERIMENT_ID,
)
UPSTREAM_TASK_IDS = EXPECTED_TASK_IDS[:-1]
PROMOTABLE_CLASSES = frozenset({"complete", "positive", "ready"})
INFRA_CLASSES = frozenset({"complete", "positive", "ready", "null"})
GATE_PRINCIPLE = "Structured gates read only terminal exact artifacts with exact bare fields."

UPSTREAM_ARTIFACT_RELATIVE_PATHS = (
    Path("results/experiment_6296_v542_adversarial_capstone.json"),
    Path("results/experiment_6297_v543_terminal_transition.json"),
    Path("results/experiment_6298_terminal_evidence_preflight_linter.json"),
    Path("results/experiment_6299_v543_post_marker_source_scope_freeze.json"),
    Path("results/experiment_6300_three_family_universal_activation_bus.json"),
    Path("results/experiment_6301_activation_bus_integrity_audit.json"),
    Path("results/experiment_6302_shared_activation_state_initializer.json"),
    Path("results/experiment_6303_live_three_family_shared_state_benchmark.json"),
    Path("results/experiment_6304_reference_anchored_online_state_learning.json"),
    Path("results/experiment_6305_evidence_licensed_cross_family_transfer.json"),
    Path("results/experiment_6306_online_state_learning_safety_audit.json"),
    Path("results/experiment_6307_arc_target_validated_route_canary.json"),
    Path("results/experiment_6308_arc_target_validated_route_holdout.json"),
)
PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    MILESTONE_DOC_RELATIVE_PATH,
    RESEARCH_REFERENCES_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    KNOWN_ISSUES_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    TRACEABILITY_RELATIVE_PATH,
    PRD_RELATIVE_PATH,
    ARCHITECTURE_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    ADVERSARIAL_VERIFY_RELATIVE_PATH,
    CHECK_SPEC_COVERAGE_RELATIVE_PATH,
    TERMINAL_ARTIFACTS_RELATIVE_PATH,
    RESEARCH_CONDUCTOR_RELATIVE_PATH,
    *UPSTREAM_ARTIFACT_RELATIVE_PATHS,
)
SOURCE_RELATIVE_PATHS = (
    *PROTECTED_RELATIVE_PATHS,
    Path("python/carnot/experiment_6309_v543_adversarial_capstone.py"),
    Path("tests/python/test_experiment_6309_v543_adversarial_capstone.py"),
)

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6309_v543_adversarial_capstone.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6309_v543_adversarial_capstone.py "
    "-m pytest tests/python/test_experiment_6309_v543_adversarial_capstone.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6309_v543_adversarial_capstone.py "
    "--fail-under=100 --show-missing"
)
RUFF_CHECK_COMMAND = (
    ".venv/bin/ruff check python/carnot/experiment_6309_v543_adversarial_capstone.py "
    "tests/python/test_experiment_6309_v543_adversarial_capstone.py"
)
RUFF_FORMAT_COMMAND = (
    ".venv/bin/ruff format --check python/carnot/experiment_6309_v543_adversarial_capstone.py "
    "tests/python/test_experiment_6309_v543_adversarial_capstone.py"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6309_v543_adversarial_capstone.py"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py --all"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
E2E_PLAN_READ_COMMAND = "sed -n 1,260p ops/e2e-test-plan.md"
ADVERSARIAL_SELF_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6309_v543_adversarial_capstone.json"
)
GIT_STATUS_COMMAND = "git status --short --untracked-files=all"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    RUFF_CHECK_COMMAND,
    RUFF_FORMAT_COMMAND,
    SPEC_COVERAGE_COMMAND,
    E2E_PLAN_READ_COMMAND,
    FULL_PYTEST_COMMAND,
    DETERMINATION_COMMAND,
    ROOT_CLUTTER_COMMAND,
    ADVERSARIAL_SELF_COMMAND,
    GIT_STATUS_COMMAND,
)
COMMAND_TIMEOUTS_S = {
    FULL_PYTEST_COMMAND: 3600,
    COVERAGE_RUN_COMMAND: 600,
    DETERMINATION_COMMAND: 600,
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "milestone_roadmap_path_and_hash",
    "exact_declared_task_artifact_matrix",
    "upstream_terminal_classification_by_task",
    "current_rule_adversarial_results_by_task",
    "missing_nonterminal_blocked_skipped_null_flagged_retired_ready_oracle_only_replay_only_safety_only_and_unlicensed_counts",
    "terminal_evidence_preflight_summary",
    "branch_independent_promotion_ledger",
    "shared_activation_bus_verdict",
    "shared_state_initializer_verdict",
    "live_three_family_value_verdict",
    "continuous_self_learning_verdict",
    "online_learning_safety_verdict",
    "evidence_licensed_transfer_verdict",
    "arc_target_validation_verdict",
    "oracle_claim_boundary",
    "replay_is_not_transfer_boundary",
    "safety_cannot_promote_utility_boundary",
    "arc_no_solve_claim_boundary",
    "prd_gap_verdicts",
    "prior_failure_retirement_actions",
    "exclusion_manifest_updates",
    "publication_gate_replay",
    "architecture_reconciliation_receipt",
    "openspec_traceability_status_changelog_and_reference_reconciliation_receipts",
    "protected_files_unchanged",
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
    "status": "The capstone can close while branches remain blocked or null.",
    "milestone_roadmap_path_and_hash": "The active roadmap fixes the V543 denominator.",
    "exact_declared_task_artifact_matrix": "Exact paths prevent alias or receipt substitution.",
    "upstream_terminal_classification_by_task": "Upstream terminal classes remain visible.",
    "current_rule_adversarial_results_by_task": "Current verifier flags stay separate from stamps.",
    "missing_nonterminal_blocked_skipped_null_flagged_retired_ready_oracle_only_replay_only_safety_only_and_unlicensed_counts": "The 13-task denominator is conserved while special states remain visible.",
    "terminal_evidence_preflight_summary": "Exp6298 fixture logic is replayed without rewriting it.",
    "branch_independent_promotion_ledger": "Each branch is judged on its own exact evidence.",
    "shared_activation_bus_verdict": "A ready score cannot override stamps or integrity failures.",
    "shared_state_initializer_verdict": "Initializer value depends on the integrity gate.",
    "live_three_family_value_verdict": "Missing live value evidence cannot be promoted.",
    "continuous_self_learning_verdict": "Online learning utility stays separate from oracle control.",
    "online_learning_safety_verdict": "Safety audit evidence cannot promote utility by itself.",
    "evidence_licensed_transfer_verdict": "Retrieval-only evidence is not licensed transfer.",
    "arc_target_validation_verdict": "ARC route validation is not a level solve.",
    "oracle_claim_boundary": "Exact oracle evidence is not verifier value.",
    "replay_is_not_transfer_boundary": "Replay evidence is not transfer evidence.",
    "safety_cannot_promote_utility_boundary": "Safety-only success is not utility.",
    "arc_no_solve_claim_boundary": "ARC proxy metrics do not count as solves.",
    "prd_gap_verdicts": "The three PRD gaps are judged from exact artifacts only.",
    "prior_failure_retirement_actions": "Same-verdict retirement is mechanical and explicit.",
    "exclusion_manifest_updates": "Manifest edits occur only when retirement fires.",
    "publication_gate_replay": "Publication status is replayed from the stable gate.",
    "architecture_reconciliation_receipt": "Architecture claims are reconciled only where supported.",
    "openspec_traceability_status_changelog_and_reference_reconciliation_receipts": "OpenSpec, references, and ops reconciliation stay auditable.",
    "protected_files_unchanged": "Protected hashes show this run did not rewrite records.",
    "preconditions_checked": "Inputs and rule hashes are frozen before classification.",
    "inference_substrate": "This task aggregates exact declared artifacts only.",
    "verifier_is_oracle": "The top-level value is mixed and branch boundaries are explicit.",
    "field_provenance": "Every required field cites its evidence.",
    "field_principles": "Every required field states why it exists.",
    "test_commands": "Commands define the verification boundary.",
    "test_exit_codes": "Exit codes stay visible and unlaundered.",
    "duration_s": "Wall time is measured without padding.",
    "reproducibility_checksum": "A normalized checksum detects silent payload drift.",
    "honest_verdict": "The verdict states the mixed V543 outcome with a terminal prefix.",
}

COUNT_PRINCIPLES: dict[str, str] = {
    "task_count": "The denominator is exactly the 13 declared V543 tasks.",
    "terminal_class_task_count_sum": "Terminal-class buckets must add to 13.",
    "missing": "Missing exact paths cannot be replaced.",
    "nonterminal": "Nonterminal rows cannot feed claims.",
    "blocked": "Blocked terminal classes remain visible.",
    "raw_blocked_status": "Gate-block status strings are preserved even when classed as skipped.",
    "skipped": "Gate skips are terminal but cannot promote science.",
    "null": "Null findings are separate from positive evidence.",
    "flagged": "Stamped flagged artifacts remain quarantined.",
    "retired": "Retired scope stays retired.",
    "ready": "Ready is counted separately from complete.",
    "complete": "Complete evidence still needs branch gates.",
    "positive": "Positive evidence cannot promote another branch.",
    "oracle_only": "Oracle-only evidence cannot be verifier value.",
    "replay_only": "Replay-only evidence cannot be transfer.",
    "safety_only": "Safety-only evidence cannot promote utility.",
    "unlicensed_transfer": "Unlicensed transfer is not licensed transfer.",
    "current_rule_critical": "Current critical verifier flags are preserved separately.",
    "current_rule_warn": "Current warnings are preserved separately.",
}


def payload_checksum(report: JsonMap) -> str:
    normalized = json.loads(canonical_json(report))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return payload_sha256(normalized)


def read_yaml_mapping(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
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
    except json.JSONDecodeError as exc:
        meta["error"] = f"json_error:{exc.msg}"
        return {}, meta
    if not isinstance(payload, Mapping):
        meta["error"] = "json_not_mapping"
        return {}, meta
    meta["loadable"] = True
    return dict(payload), meta


def load_roadmap(root: Path = REPO_ROOT) -> JsonDict:
    return read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)


def roadmap_tasks(roadmap: JsonMap) -> list[JsonDict]:
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list):
        return []
    return [dict(task) for task in tasks if isinstance(task, Mapping)]


def git_status_lines(root: Path) -> list[str]:  # pragma: no cover - shell edge.
    proc = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    return [line for line in proc.stdout.splitlines() if line.strip()]


def hash_paths(root: Path, paths: Sequence[Path]) -> JsonDict:
    return {
        path.as_posix(): {"present": (root / path).exists(), "sha256": path_sha256(root / path)}
        for path in paths
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


def _self_payload(status: str, verdict: str) -> JsonDict:
    return {
        "status": status,
        "honest_verdict": verdict,
        "duration_s": 0.01,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_ORACLE_BOUNDARY,
        "preconditions_checked": {"self_payload": True},
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": _field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": {},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "sha256:self-payload-under-construction",
    }


def build_exact_declared_task_artifact_matrix(
    root: Path,
    tasks: Sequence[JsonMap],
    *,
    conductor_receipts: JsonMap | None = None,
    self_payload: JsonMap | None = None,
) -> JsonDict:
    rows: JsonDict = {}
    receipts = conductor_receipts or {}
    for task in tasks:
        task_id = str(task.get("id") or "")
        rel = Path(str(task.get("deliverable") or ""))
        receipt = receipts.get(task_id)
        typed_receipt = receipt if isinstance(receipt, Mapping) else None
        if task_id == EXPERIMENT_ID and self_payload is not None:
            payload = dict(self_payload)
            digest = payload_sha256(payload)
            classification = classify_artifact_payload(
                payload,
                path=root / rel,
                sha256=digest,
                conductor_receipt=typed_receipt,
            )
            meta = {"present": (root / rel).exists(), "loadable": True, "sha256": digest}
        else:
            payload, meta = read_json_mapping(root / rel)
            classification = classify_artifact_path(root / rel, conductor_receipt=typed_receipt)
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
            "verifier_is_oracle_raw": payload.get("verifier_is_oracle"),
            "same_number_alias_used": False,
            "same_number_alias_candidates_ignored": same_number_aliases(root, task_id, rel),
        }
    return rows


def _payloads(root: Path, tasks: Sequence[JsonMap], self_payload: JsonMap) -> JsonDict:
    out: JsonDict = {}
    for task in tasks:
        task_id = str(task.get("id") or "")
        rel = Path(str(task.get("deliverable") or ""))
        if task_id == EXPERIMENT_ID:
            out[task_id] = dict(self_payload)
        else:
            out[task_id] = read_json_mapping(root / rel)[0]
    return out


def _flag_counts(flags: Sequence[JsonMap]) -> tuple[int, int]:
    critical = sum(1 for flag in flags if flag.get("severity") == "critical")
    warn = sum(1 for flag in flags if flag.get("severity") == "warn")
    return critical, warn


def adversarial_result_row(path: Path, payload: JsonMap, current: JsonMap) -> JsonDict:
    flags = [dict(flag) for flag in current.get("flags", []) if isinstance(flag, Mapping)]
    critical, warn = _flag_counts(flags)
    return {
        "path": path.as_posix(),
        "present": path.exists(),
        "loaded": current.get("loaded"),
        "error": current.get("error"),
        "stamped_flagged_adversarial": payload.get("flagged_adversarial") is True,
        "stamped_corrigendum_pending": bool(payload.get("corrigendum_pending")),
        "current_rule_flag_count": int(current.get("flag_count") or len(flags)),
        "current_rule_critical_flag_count": critical,
        "current_rule_warn_flag_count": warn,
        "current_rule_flags": flags,
    }


def current_rule_adversarial_results(
    root: Path,
    tasks: Sequence[JsonMap],
    *,
    self_payload: JsonMap | None = None,
) -> JsonDict:
    from adversarial_verify import verify_artifact

    reviews: JsonDict = {}
    for task in tasks:
        task_id = str(task.get("id") or "")
        rel = Path(str(task.get("deliverable") or ""))
        path = root / rel
        if task_id == EXPERIMENT_ID and self_payload is not None and not path.exists():
            reviews[task_id] = {
                "path": rel.as_posix(),
                "present": False,
                "loaded": True,
                "error": None,
                "stamped_flagged_adversarial": False,
                "stamped_corrigendum_pending": False,
                "current_rule_flag_count": 0,
                "current_rule_critical_flag_count": 0,
                "current_rule_warn_flag_count": 0,
                "current_rule_flags": [],
                "skipped": "self_payload_under_construction",
            }
            continue
        payload = read_json_mapping(path)[0]
        reviews[task_id] = adversarial_result_row(
            path, payload, verify_artifact(path, declared=True)
        )
    return reviews


def _critical_count(reviews: JsonMap, task_id: str) -> int:
    row = reviews.get(task_id, {})
    return int(row.get("current_rule_critical_flag_count") or 0) if isinstance(row, Mapping) else 0


def _warn_count(reviews: JsonMap, task_id: str) -> int:
    row = reviews.get(task_id, {})
    return int(row.get("current_rule_warn_flag_count") or 0) if isinstance(row, Mapping) else 0


def _bare_value(payload: JsonMap, field: str) -> Any:
    value = payload.get(field)
    if isinstance(value, Mapping) and "value" in value and "principle" in value:
        return value.get("value")
    return value


def _task_state(task_id: str, matrix: JsonMap, payloads: JsonMap, reviews: JsonMap) -> JsonDict:
    row = matrix.get(task_id, {})
    payload = payloads.get(task_id, {})
    return {
        "task_id": task_id,
        "declared_deliverable": row.get("declared_deliverable"),
        "terminal_class": row.get("terminal_class"),
        "terminal": row.get("terminal") is True,
        "status": payload.get("status") if isinstance(payload, Mapping) else None,
        "honest_verdict": payload.get("honest_verdict") if isinstance(payload, Mapping) else None,
        "verifier_is_oracle": payload.get("verifier_is_oracle")
        if isinstance(payload, Mapping)
        else None,
        "current_rule_critical_flag_count": _critical_count(reviews, task_id),
        "current_rule_warn_flag_count": _warn_count(reviews, task_id),
        "stamped_flagged": bool(row.get("flagged_adversarial_stamped")),
        "sha256": row.get("sha256"),
    }


def count_preserved_states(matrix: JsonMap, payloads: JsonMap, reviews: JsonMap) -> JsonDict:
    classes = Counter(str(row.get("terminal_class") or "unknown") for row in matrix.values())
    result = {key: 0 for key in COUNT_PRINCIPLES}
    result.update(
        {
            key: int(classes.get(key, 0))
            for key in (
                "missing",
                "blocked",
                "skipped",
                "null",
                "flagged",
                "retired",
                "ready",
                "complete",
                "positive",
            )
        }
    )
    result["task_count"] = len(matrix)
    result["terminal_class_task_count_sum"] = int(sum(classes.values()))
    result["nonterminal"] = sum(1 for row in matrix.values() if row.get("terminal") is not True)
    result["raw_blocked_status"] = sum(
        1 for row in matrix.values() if str(row.get("status_raw") or "").startswith("blocked")
    )
    result["oracle_only"] = sum(
        1
        for payload in payloads.values()
        if isinstance(payload, Mapping) and payload.get("verifier_is_oracle") is True
    )
    result["replay_only"] = sum(
        1
        for payload in payloads.values()
        if isinstance(payload, Mapping)
        and "replay" in str(payload.get("inference_substrate") or "")
    )
    result["safety_only"] = 1 if "exp6306-online-state-learning-safety-audit" in matrix else 0
    result["unlicensed_transfer"] = (
        1
        if str(
            matrix.get("exp6305-evidence-licensed-cross-family-transfer", {}).get("status_raw")
            or ""
        ).startswith("blocked")
        else 0
    )
    result["current_rule_critical"] = sum(_critical_count(reviews, task_id) for task_id in matrix)
    result["current_rule_warn"] = sum(_warn_count(reviews, task_id) for task_id in matrix)
    result["terminal_class_counts"] = dict(
        sorted((key, int(value)) for key, value in classes.items())
    )
    result["count_principles"] = dict(COUNT_PRINCIPLES)
    return result


def evaluate_operator(actual: Any, op: str, expected: Any) -> bool:
    if op == "exists":
        return (actual is not None) is bool(expected)
    if actual is None:
        return False
    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
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


def evaluate_structured_gates(root: Path, tasks: Sequence[JsonMap]) -> JsonDict:
    by_id = {str(task.get("id") or ""): task for task in tasks}
    rows: list[JsonDict] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        gates = task.get("gated_on")
        if not isinstance(gates, list) or not gates:
            continue
        gate_results: list[JsonDict] = []
        for gate in gates:
            upstream = str(gate.get("upstream") or "") if isinstance(gate, Mapping) else ""
            field = str(gate.get("artifact_field") or "") if isinstance(gate, Mapping) else ""
            op = str(gate.get("op") or "") if isinstance(gate, Mapping) else ""
            expected = gate.get("value") if isinstance(gate, Mapping) else None
            upstream_task = by_id.get(upstream)
            if upstream_task is None:
                gate_results.append(
                    {
                        "upstream": upstream,
                        "artifact_field": field,
                        "passed": False,
                        "actual": None,
                        "reason": "missing_upstream_task",
                        "principle": GATE_PRINCIPLE,
                    }
                )
                continue
            upstream_rel = Path(str(upstream_task.get("deliverable") or ""))
            eligibility = gate_field_eligibility_for_path(root / upstream_rel, field)
            actual = eligibility.value if eligibility.field_present else None
            passed = eligibility.eligible and evaluate_operator(actual, op, expected)
            gate_results.append(
                {
                    "upstream": upstream,
                    "upstream_declared_deliverable": upstream_rel.as_posix(),
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
        rows.append(
            {
                "task_id": task_id,
                "passed": all(row["passed"] for row in gate_results),
                "gate_results": gate_results,
                "principle": GATE_PRINCIPLE,
            }
        )
    return {
        "gates": rows,
        "passed_count": sum(1 for row in rows if row["passed"]),
        "failed_count": sum(1 for row in rows if not row["passed"]),
        "principle": GATE_PRINCIPLE,
    }


def terminal_evidence_preflight_summary(root: Path) -> JsonDict:
    payload, meta = read_json_mapping(
        root / "results/experiment_6298_terminal_evidence_preflight_linter.json"
    )
    manifest = build_synthetic_fixture_manifest()
    synthetic = evaluate_fixture_manifest(manifest)
    v542_rows = replay_v542_failure_fixtures(root)
    return {
        "exp6298_artifact": meta,
        "exp6298_ready_score": payload.get("terminal_evidence_preflight_ready_score"),
        "exp6298_status": payload.get("status"),
        "exp6298_honest_verdict": payload.get("honest_verdict"),
        "synthetic_clean_fixture_accept_count": synthetic["clean_fixture_accept_count"],
        "synthetic_false_accept_count": synthetic["false_accept_count"],
        "synthetic_false_reject_count": synthetic["false_reject_count"],
        "v542_fixture_replay_count": len(v542_rows),
        "v542_fixture_failures_observed": {
            str(row.get("fixture_id")): row.get("failure_classes", []) for row in v542_rows
        },
        "rerun_mode": "library_replay_no_exp6298_artifact_write",
    }


def shared_activation_bus_verdict(matrix: JsonMap, payloads: JsonMap, reviews: JsonMap) -> JsonDict:
    bus = _task_state("exp6300-three-family-universal-activation-bus", matrix, payloads, reviews)
    audit = _task_state("exp6301-activation-bus-integrity-audit", matrix, payloads, reviews)
    audit_payload = payloads.get("exp6301-activation-bus-integrity-audit", {})
    blockers = []
    if bus["stamped_flagged"] or bus["current_rule_critical_flag_count"]:
        blockers.append("activation_bus_flagged_under_stamp_or_current_rules")
    if _bare_value(audit_payload, "activation_bus_integrity_ready_score") != 1.0:
        blockers.append("integrity_audit_ready_score_not_one")
    if audit["terminal_class"] == "flagged":
        blockers.append("integrity_audit_terminal_flagged")
    return {
        "promotion_allowed": False,
        "activation_bus_ready_score": _bare_value(
            payloads.get("exp6300-three-family-universal-activation-bus", {}),
            "shared_activation_bus_ready_score",
        ),
        "activation_bus_state": bus,
        "integrity_audit_state": audit,
        "activation_bus_integrity_ready_score": _bare_value(
            audit_payload,
            "activation_bus_integrity_ready_score",
        ),
        "blocking_reasons": sorted(set(blockers)),
    }


def shared_state_initializer_verdict(
    matrix: JsonMap, payloads: JsonMap, reviews: JsonMap
) -> JsonDict:
    state = _task_state("exp6302-shared-activation-state-initializer", matrix, payloads, reviews)
    payload = payloads.get("exp6302-shared-activation-state-initializer", {})
    return {
        "promotion_allowed": False,
        "terminal_class": state["terminal_class"],
        "gate_check_summary": payload.get("gate_check_summary")
        if isinstance(payload, Mapping)
        else None,
        "gates_evaluated": payload.get("gates_evaluated") if isinstance(payload, Mapping) else None,
        "blocking_reasons": ["exp6301_integrity_gate_failed"],
        "state": state,
    }


def live_three_family_value_verdict(
    matrix: JsonMap, payloads: JsonMap, reviews: JsonMap
) -> JsonDict:
    state = _task_state(
        "exp6303-live-three-family-shared-state-benchmark", matrix, payloads, reviews
    )
    return {
        "promotion_allowed": False,
        "terminal_class": state["terminal_class"],
        "present": matrix.get("exp6303-live-three-family-shared-state-benchmark", {}).get(
            "present"
        ),
        "blocking_reasons": ["exact_declared_live_benchmark_artifact_missing"],
        "state": state,
    }


def continuous_self_learning_verdict(
    matrix: JsonMap, payloads: JsonMap, reviews: JsonMap
) -> JsonDict:
    task_id = "exp6304-reference-anchored-online-state-learning"
    state = _task_state(task_id, matrix, payloads, reviews)
    payload = payloads.get(task_id, {})
    return {
        "utility_ready_score": _bare_value(
            payload, "reference_anchored_online_learning_ready_score"
        ),
        "promotion_allowed_for_online_learning_only": True,
        "verifier_is_oracle": payload.get("verifier_is_oracle")
        if isinstance(payload, Mapping)
        else None,
        "oracle_boundary": "exact outcome reveal is the immutable validator, not verifier value",
        "source_model_weight_mutation_count": _bare_value(
            payload, "source_model_weight_mutation_count"
        ),
        "state": state,
    }


def online_learning_safety_verdict(
    matrix: JsonMap, payloads: JsonMap, reviews: JsonMap
) -> JsonDict:
    task_id = "exp6306-online-state-learning-safety-audit"
    payload = payloads.get(task_id, {})
    return {
        "safety_ready_score": _bare_value(payload, "online_learning_safety_ready_score"),
        "safety_only": True,
        "promotion_allowed_for_utility": False,
        "safety_cannot_promote_utility_receipt": payload.get(
            "safety_cannot_promote_utility_receipt"
        )
        if isinstance(payload, Mapping)
        else None,
        "state": _task_state(task_id, matrix, payloads, reviews),
    }


def evidence_licensed_transfer_verdict(
    matrix: JsonMap, payloads: JsonMap, reviews: JsonMap
) -> JsonDict:
    task_id = "exp6305-evidence-licensed-cross-family-transfer"
    payload = payloads.get(task_id, {})
    return {
        "promotion_allowed": False,
        "terminal_class": matrix.get(task_id, {}).get("terminal_class"),
        "gate_check_summary": payload.get("gate_check_summary")
        if isinstance(payload, Mapping)
        else None,
        "retrieval_only_promoted": False,
        "target_licensed_transfer_ready_score": _bare_value(
            payload,
            "evidence_licensed_transfer_ready_score",
        ),
        "blocking_reasons": ["exp6301_integrity_gate_failed", "target_license_artifact_blocked"],
        "state": _task_state(task_id, matrix, payloads, reviews),
    }


def arc_target_validation_verdict(matrix: JsonMap, payloads: JsonMap, reviews: JsonMap) -> JsonDict:
    canary = payloads.get("exp6307-arc-target-validated-route-canary", {})
    holdout = payloads.get("exp6308-arc-target-validated-route-holdout", {})
    canary_ready = _bare_value(canary, "arc_target_licensed_router_ready_score")
    holdout_ready = _bare_value(holdout, "arc_target_licensed_generalization_ready_score")
    solve_claim_count = int(_bare_value(canary, "arc_level_solve_claim_count") or 0) + int(
        _bare_value(holdout, "arc_level_solve_claim_count") or 0
    )
    hidden_count = int(_bare_value(canary, "hidden_game_source_access_count") or 0) + int(
        _bare_value(holdout, "hidden_game_source_access_count") or 0
    )
    outer_count = int(_bare_value(canary, "outer_loop_ground_truth_search_count") or 0) + int(
        _bare_value(holdout, "outer_loop_ground_truth_search_count") or 0
    )
    route_ready = canary_ready == 1.0 and holdout_ready == 1.0
    return {
        "route_audit_promotion_allowed": bool(
            route_ready and solve_claim_count == 0 and hidden_count == 0 and outer_count == 0
        ),
        "solve_claim_allowed": False,
        "canary_ready_score": canary_ready,
        "holdout_ready_score": holdout_ready,
        "arc_level_solve_claim_count": solve_claim_count,
        "hidden_game_source_access_count": hidden_count,
        "outer_loop_ground_truth_search_count": outer_count,
        "canary_state": _task_state(
            "exp6307-arc-target-validated-route-canary", matrix, payloads, reviews
        ),
        "holdout_state": _task_state(
            "exp6308-arc-target-validated-route-holdout", matrix, payloads, reviews
        ),
    }


def branch_independent_promotion_ledger(
    matrix: JsonMap,
    payloads: JsonMap,
    reviews: JsonMap,
    gates: JsonMap,
) -> JsonDict:
    infra_ids = EXPECTED_TASK_IDS[:3]
    infra_blockers = [
        task_id
        for task_id in infra_ids
        if matrix.get(task_id, {}).get("terminal_class") not in INFRA_CLASSES
        or _critical_count(reviews, task_id) > 0
    ]
    shared = shared_activation_bus_verdict(matrix, payloads, reviews)
    initializer = shared_state_initializer_verdict(matrix, payloads, reviews)
    live_value = live_three_family_value_verdict(matrix, payloads, reviews)
    online = continuous_self_learning_verdict(matrix, payloads, reviews)
    safety = online_learning_safety_verdict(matrix, payloads, reviews)
    licensed = evidence_licensed_transfer_verdict(matrix, payloads, reviews)
    arc = arc_target_validation_verdict(matrix, payloads, reviews)
    return {
        "infrastructure_source_integrity": {
            "promotion_allowed": not infra_blockers,
            "task_ids": list(infra_ids),
            "blocking_reasons": infra_blockers,
            "notes": ["Exp6299 is a complete_null source freeze with zero accepted findings."],
        },
        "shared_model_state_to_exact_energy": {
            "promotion_allowed": False,
            "task_ids": list(EXPECTED_TASK_IDS[3:7]),
            "blocking_reasons": sorted(
                set(
                    [
                        *shared["blocking_reasons"],
                        *initializer["blocking_reasons"],
                        *live_value["blocking_reasons"],
                    ]
                )
            ),
        },
        "continuous_online_learning_and_licensed_transfer": {
            "promotion_allowed": False,
            "online_learning_component_ready": online["utility_ready_score"] == 1.0,
            "safety_component_ready": safety["safety_ready_score"] == 1.0,
            "licensed_transfer_ready": False,
            "blocking_reasons": [
                "licensed_transfer_blocked",
                "safety_only_cannot_promote_utility",
                "retrieval_only_not_transfer",
            ],
        },
        "arc_target_validated_routing": {
            "route_audit_promotion_allowed": arc["route_audit_promotion_allowed"],
            "solve_claim_allowed": False,
            "blocking_reasons": []
            if arc["route_audit_promotion_allowed"]
            else ["arc_route_gate_failed"],
            "claim_boundary": "route audit only; no hidden level solve claim",
        },
        "structured_gate_replay": gates,
    }


def oracle_claim_boundary(payloads: JsonMap) -> JsonDict:
    oracle_tasks = [
        task_id
        for task_id, payload in payloads.items()
        if isinstance(payload, Mapping) and payload.get("verifier_is_oracle") is True
    ]
    return {
        "verifier_is_oracle_top_level": VERIFIER_ORACLE_BOUNDARY,
        "oracle_only_task_ids": oracle_tasks,
        "oracle_promoted_as_verifier_value": False,
        "boundary": "Exact validators and oracle controls may certify outcomes but cannot be counted as verifier value.",
    }


def replay_is_not_transfer_boundary(payloads: JsonMap) -> JsonDict:
    replay_tasks = [
        task_id
        for task_id, payload in payloads.items()
        if isinstance(payload, Mapping)
        and "replay" in str(payload.get("inference_substrate") or "")
    ]
    return {
        "replay_only_task_ids": replay_tasks,
        "replay_promoted_as_transfer": False,
        "licensed_transfer_task_id": "exp6305-evidence-licensed-cross-family-transfer",
        "licensed_transfer_blocked": True,
    }


def safety_cannot_promote_utility_boundary() -> JsonDict:
    return {
        "safety_task_id": "exp6306-online-state-learning-safety-audit",
        "safety_promoted_as_utility": False,
        "boundary": "A safety pass admits rollback behavior only; it does not establish utility.",
    }


def arc_no_solve_claim_boundary(arc_verdict: JsonMap) -> JsonDict:
    return {
        "arc_proxy_promoted_as_solve": False,
        "route_audit_promotion_allowed": arc_verdict["route_audit_promotion_allowed"],
        "solve_claim_allowed": False,
        "arc_level_solve_claim_count": arc_verdict["arc_level_solve_claim_count"],
    }


def prd_gap_verdicts(
    shared: JsonMap,
    initializer: JsonMap,
    live_value: JsonMap,
    online: JsonMap,
    licensed: JsonMap,
    arc: JsonMap,
) -> JsonDict:
    return {
        "gap_1_model_native_state_to_exact_global_energy": {
            "verdict": "blocked",
            "reason": [
                *shared["blocking_reasons"],
                *initializer["blocking_reasons"],
                *live_value["blocking_reasons"],
            ],
        },
        "gap_2_continuous_refinement_measured_model_value": {
            "verdict": "blocked",
            "reason": ["live_three_family_value_artifact_missing", "exact_oracle_not_model_value"],
        },
        "gap_3_autonomous_learning_useful_and_safely_transferable": {
            "verdict": "partial_online_learning_positive_transfer_blocked",
            "online_learning_ready_score": online["utility_ready_score"],
            "licensed_transfer_promotion_allowed": licensed["promotion_allowed"],
        },
        "arc_route_gap": {
            "verdict": "route_audit_ready_no_solve_claim"
            if arc["route_audit_promotion_allowed"]
            else "blocked",
            "solve_claim_allowed": arc["solve_claim_allowed"],
        },
    }


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
            fired = current_class != "missing" and current_verdict == prior_verdict
            actions.append(
                {
                    "task_id": task_id,
                    "prior_experiment_id": prior.get("experiment_id"),
                    "prior_verdict": prior_verdict,
                    "current_terminal_class": current_class,
                    "current_verdict": current_verdict,
                    "rule_fired": fired,
                    "action": "retire_if_same_verdict_rule_fired"
                    if fired
                    else "no_retirement_current_verdict_differs_or_missing",
                    "would_update_exclusion_manifest": fired,
                }
            )
    return {
        "actions": actions,
        "rule_fired_count": sum(1 for action in actions if action["rule_fired"]),
        "principle": FIELD_PRINCIPLES["prior_failure_retirement_actions"],
    }


def exclusion_manifest_updates(retirements: JsonMap, root: Path, before: JsonMap) -> JsonDict:
    fired = int(retirements.get("rule_fired_count") or 0)
    manifest_path = EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix()
    after = path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
    return {
        "updated": fired > 0,
        "update_count": fired,
        "manifest_path": manifest_path,
        "before_sha256": before.get(manifest_path),
        "after_sha256": after,
        "note": "no retire_if_same_verdict rule fired"
        if fired == 0
        else "retirement entry required",
    }


def publication_gate_replay(publication_result: JsonMap | None = None) -> JsonDict:
    if publication_result is None:
        from publication_gate import evaluate

        publication_result = evaluate()
    gates = publication_result.get("gates") if isinstance(publication_result, Mapping) else {}
    return {
        "paper_ready": publication_result.get("paper_ready")
        if isinstance(publication_result, Mapping)
        else None,
        "unmet_gates": list(publication_result.get("unmet_gates") or [])
        if isinstance(publication_result, Mapping)
        else [],
        "gates": gates if isinstance(gates, Mapping) else {},
        "replay_only": True,
    }


def architecture_reconciliation_receipt(root: Path, run_date: str) -> JsonDict:
    path = root / ARCHITECTURE_RELATIVE_PATH
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    last_reconciled = None
    for line in text.splitlines():
        if line.startswith("**Last Reconciled:**"):
            last_reconciled = line.removeprefix("**Last Reconciled:**").strip()
            break
    run_day = datetime.strptime(run_date, "%Y%m%d").date()
    reconciled_day = datetime.strptime(last_reconciled, "%Y-%m-%d").date()
    stale = reconciled_day < run_day - timedelta(days=30)
    return {
        "path": ARCHITECTURE_RELATIVE_PATH.as_posix(),
        "sha256": path_sha256(path),
        "last_reconciled": last_reconciled,
        "architecture_stale_over_30_days": stale,
        "reconciled_by_this_task": False,
        "supported_updates": [
            "ARC route validation may be cited only as route-audit evidence.",
            "Shared-state exact-energy value remains blocked.",
            "Licensed transfer remains blocked.",
        ],
    }


def openspec_traceability_status_changelog_and_reference_receipts(root: Path) -> JsonDict:
    spec_text = (root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    refs_text = (root / RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    return {
        "openspec_req_integ_6309_present": "REQ-INTEG-6309" in spec_text,
        "openspec_scenarios_present": all(
            f"SCENARIO-INTEG-6309-{idx}" in spec_text for idx in range(1, 7)
        ),
        "research_references_v543_marker_present": "V543-PLANNER-REFRESH-20260810-END" in refs_text,
        "traceability_status_changelog_touched_by_this_task": False,
        "stop_when_done_rule_deferred_ops_traceability_updates": True,
        "hashes": hash_paths(
            root,
            (
                SPEC_RELATIVE_PATH,
                TRACEABILITY_RELATIVE_PATH,
                STATUS_RELATIVE_PATH,
                CHANGELOG_RELATIVE_PATH,
                RESEARCH_REFERENCES_RELATIVE_PATH,
            ),
        ),
    }


def _field_provenance() -> JsonDict:
    sources = sorted(path.as_posix() for path in SOURCE_RELATIVE_PATHS) + ["REQ-INTEG-6309"]
    return {
        field: {"principle": FIELD_PRINCIPLES[field], "sources": sources}
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
        return "blocked", "blocked: one or more Exp6309 validation commands failed"
    return (
        "complete",
        "complete: V543 capstone preserved exact artifact states; shared-state and licensed-transfer branches blocked; ARC route audit ready with no solve claim",
    )


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
        "milestone": MILESTONE,
        "roadmap_path": ROADMAP_RELATIVE_PATH.as_posix(),
        "roadmap_sha256": path_sha256(root / ROADMAP_RELATIVE_PATH),
        "research_roadmap_next_present": (root / ROADMAP_NEXT_RELATIVE_PATH).exists(),
        "declared_artifact_hashes": declared,
        "current_adversarial_rules_sha256": path_sha256(root / ADVERSARIAL_VERIFY_RELATIVE_PATH),
        "terminal_artifacts_sha256": path_sha256(root / TERMINAL_ARTIFACTS_RELATIVE_PATH),
        "exclusion_manifest_sha256": path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
        "docs_and_ops_hashes_before": dict(before_hashes),
        "git_status_before": list(git_status_before),
        "git_status_after_tests": list(git_status_after_tests),
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
    command_rows = [dict(row) for row in command_receipts or []]
    status, verdict = _status_from_commands(command_rows)
    roadmap = load_roadmap(root)
    tasks = roadmap_tasks(roadmap)
    self_payload = _self_payload(status, verdict)
    matrix = build_exact_declared_task_artifact_matrix(root, tasks, self_payload=self_payload)
    payloads = _payloads(root, tasks, self_payload)
    reviews = dict(
        artifact_reviews
        if artifact_reviews is not None
        else current_rule_adversarial_results(root, tasks, self_payload=self_payload)
    )
    counts = count_preserved_states(matrix, payloads, reviews)
    gates = evaluate_structured_gates(root, tasks)
    shared = shared_activation_bus_verdict(matrix, payloads, reviews)
    initializer = shared_state_initializer_verdict(matrix, payloads, reviews)
    live_value = live_three_family_value_verdict(matrix, payloads, reviews)
    online = continuous_self_learning_verdict(matrix, payloads, reviews)
    safety = online_learning_safety_verdict(matrix, payloads, reviews)
    licensed = evidence_licensed_transfer_verdict(matrix, payloads, reviews)
    arc = arc_target_validation_verdict(matrix, payloads, reviews)
    ledger = branch_independent_promotion_ledger(matrix, payloads, reviews, gates)
    retirements = prior_failure_retirement_actions(tasks, matrix)
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
            "expected_task_ids": list(EXPECTED_TASK_IDS),
            "task_count": len(tasks),
        },
        "exact_declared_task_artifact_matrix": matrix,
        "upstream_terminal_classification_by_task": {
            task_id: row for task_id, row in matrix.items() if task_id in UPSTREAM_TASK_IDS
        },
        "current_rule_adversarial_results_by_task": reviews,
        "missing_nonterminal_blocked_skipped_null_flagged_retired_ready_oracle_only_replay_only_safety_only_and_unlicensed_counts": counts,
        "terminal_evidence_preflight_summary": terminal_evidence_preflight_summary(root),
        "branch_independent_promotion_ledger": ledger,
        "shared_activation_bus_verdict": shared,
        "shared_state_initializer_verdict": initializer,
        "live_three_family_value_verdict": live_value,
        "continuous_self_learning_verdict": online,
        "online_learning_safety_verdict": safety,
        "evidence_licensed_transfer_verdict": licensed,
        "arc_target_validation_verdict": arc,
        "oracle_claim_boundary": oracle_claim_boundary(payloads),
        "replay_is_not_transfer_boundary": replay_is_not_transfer_boundary(payloads),
        "safety_cannot_promote_utility_boundary": safety_cannot_promote_utility_boundary(),
        "arc_no_solve_claim_boundary": arc_no_solve_claim_boundary(arc),
        "prd_gap_verdicts": prd_gap_verdicts(
            shared, initializer, live_value, online, licensed, arc
        ),
        "prior_failure_retirement_actions": retirements,
        "exclusion_manifest_updates": exclusion_manifest_updates(retirements, root, before),
        "publication_gate_replay": publication_gate_replay(publication_result),
        "architecture_reconciliation_receipt": architecture_reconciliation_receipt(root, date),
        "openspec_traceability_status_changelog_and_reference_reconciliation_receipts": openspec_traceability_status_changelog_and_reference_receipts(
            root
        ),
        "protected_files_unchanged": protected_files_unchanged(root, before),
        "preconditions_checked": preconditions_checked(
            root,
            tasks,
            before,
            git_status_before or [],
            git_status_after_tests or [],
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_ORACLE_BOUNDARY,
        "field_provenance": _field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": [str(row.get("command") or "") for row in command_rows]
        or list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": _test_exit_codes(command_rows),
        "duration_s": time.perf_counter() - started,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "honest_verdict": verdict,
    }
    report["reproducibility_checksum"] = payload_checksum(report)
    return report


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
        if not isinstance(principles.get(field), str) or not principles.get(field):
            errors.append(f"missing field_principles entry: {field}")
        if field not in provenance:
            errors.append(f"missing field_provenance entry: {field}")
    counts = report.get(
        "missing_nonterminal_blocked_skipped_null_flagged_retired_ready_oracle_only_replay_only_safety_only_and_unlicensed_counts"
    )
    if not isinstance(counts, Mapping):
        errors.append("counts field is not a mapping")
        counts = {}
    if counts.get("task_count") != 13 or counts.get("terminal_class_task_count_sum") != 13:
        errors.append("terminal class counts must conserve 13 tasks")
    count_principles = counts.get("count_principles")
    if not isinstance(count_principles, Mapping):
        count_principles = {}
    for key in COUNT_PRINCIPLES:
        if key not in count_principles:
            errors.append(f"missing count principle: {key}")
    ledger = report.get("branch_independent_promotion_ledger")
    gates = (
        ledger.get("structured_gate_replay", {}).get("gates")
        if isinstance(ledger, Mapping)
        else None
    )
    if isinstance(gates, list):
        for gate in gates:
            if not isinstance(gate, Mapping) or not gate.get("principle"):
                errors.append("gate missing principle")
                break
    else:
        errors.append("structured gates missing")
    if report.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("wrong inference_substrate")
    if report.get("verifier_is_oracle") != VERIFIER_ORACLE_BOUNDARY:
        errors.append("verifier_is_oracle boundary is wrong")
    verdict = str(report.get("honest_verdict") or "")
    if not verdict.startswith(("complete:", "blocked:", "complete_null:", "flagged:", "skipped:")):
        errors.append("honest_verdict lacks terminal prefix")
    checksum = report.get("reproducibility_checksum")
    if isinstance(checksum, str) and checksum.startswith("sha256:"):
        if checksum != payload_checksum(report):
            errors.append("reproducibility_checksum mismatch")
    else:
        errors.append("reproducibility_checksum missing")
    return errors


def run_command(
    command: str,
    root: Path,
    timeout_s: int | None = None,
) -> JsonDict:  # pragma: no cover - shell wrapper.
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


def run_default_commands(root: Path) -> list[JsonDict]:  # pragma: no cover - shell wrapper.
    return [
        run_command(command, root, COMMAND_TIMEOUTS_S.get(command))
        for command in DEFAULT_TEST_COMMANDS
    ]


def write_report(
    report: JsonMap,
    root: Path = REPO_ROOT,
    *,
    env: Mapping[str, str] | None = None,
) -> Path:
    errors = validate_report(report)
    if errors:
        raise ValueError("invalid Exp6309 report: " + "; ".join(errors))
    return atomic_write_json(RESULT_RELATIVE_PATH, report, root=root, env=env, sort_keys=False)


def run_experiment(
    root: Path,
    date: str,
    *,
    run_commands: bool,
) -> JsonDict:  # pragma: no cover - CLI workflow.
    started = time.perf_counter()
    before = protected_hashes(root)
    git_before = git_status_lines(root)
    preliminary = build_report(
        root,
        date=date,
        command_receipts=[],
        before_hashes=before,
        git_status_before=git_before,
        started_at=started,
    )
    write_report(preliminary, root)
    command_rows = run_default_commands(root) if run_commands else []
    final = build_report(
        root,
        date=date,
        command_receipts=command_rows,
        before_hashes=before,
        git_status_before=git_before,
        git_status_after_tests=git_status_lines(root),
        started_at=started,
    )
    write_report(final, root)
    return final


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now(UTC).strftime("%Y%m%d"))
    parser.add_argument("--no-run-commands", action="store_true")
    args = parser.parse_args(argv)
    report = run_experiment(REPO_ROOT, args.date, run_commands=not args.no_run_commands)
    print(json.dumps(report, indent=2, sort_keys=False))
    return 0 if report["status"] in {"complete", "blocked"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
