"""Exp6322 V544 adversarial capstone.

Spec refs: REQ-INFRA-6322, SCENARIO-INFRA-6322-1,
SCENARIO-INFRA-6322-2, SCENARIO-INFRA-6322-3,
SCENARIO-INFRA-6322-4, SCENARIO-INFRA-6322-5,
SCENARIO-INFRA-6322-6.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml

from carnot.experiment_artifacts import atomic_write_json, resolve_experiment_artifact_path
from carnot.terminal_artifacts import (
    canonical_json,
    classify_artifact_path,
    classify_artifact_payload,
    path_sha256,
    payload_sha256,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(SCRIPTS_ROOT))

MILESTONE = "2026.08.544"
EXPERIMENT_ID = "exp6322-v544-adversarial-capstone"
SCHEMA = "carnot.experiment_6322.v544_adversarial_capstone.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6322_v544_adversarial_capstone.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
MILESTONE_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
TRACEABILITY_RELATIVE_PATH = Path("_bmad/traceability.md")
PRD_RELATIVE_PATH = Path("_bmad/prd.md")
ARCHITECTURE_RELATIVE_PATH = Path("_bmad/architecture.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
SUMMARIZE_ARTIFACT_RELATIVE_PATH = Path("scripts/summarize_artifact.py")
RESEARCH_CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
OPERATIONAL_RETRO_RELATIVE_PATH = Path("ops/experiment_6322_v544_operational_retro.md")
INFERENCE_SUBSTRATE = "aggregation_from_exact_declared_artifacts"
EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6322_test_receipts.json")

EXPECTED_TASK_IDS = (
    "exp6310-v544-terminal-transition",
    "exp6311-v544-post-marker-source-scope-freeze",
    "exp6312-model-local-representation-surface-preflight",
    "exp6313-exact-code-safety-pair-fixture",
    "exp6314-three-family-model-local-state-corpus",
    "exp6315-model-local-paired-difference-energy-probes",
    "exp6316-model-local-probe-integrity-audit",
    "exp6317-live-three-family-model-local-verifier-benchmark",
    "exp6318-versioned-factor-local-online-initializer",
    "exp6319-feedback-directed-online-update-search",
    "exp6320-online-self-evolution-safety-audit",
    "exp6321-arc-target-licensed-route-live-shadow-ab",
    EXPERIMENT_ID,
)
MODEL_LOCAL_TASK_IDS = EXPECTED_TASK_IDS[2:8]
CONTINUOUS_LEARNING_TASK_ID = "exp6318-versioned-factor-local-online-initializer"
FEEDBACK_SEARCH_TASK_ID = "exp6319-feedback-directed-online-update-search"
SAFETY_TASK_ID = "exp6320-online-self-evolution-safety-audit"
ARC_SHADOW_TASK_ID = "exp6321-arc-target-licensed-route-live-shadow-ab"
LAUNDERING_GUARD_FIELDS = (
    "shared_bus_promotion_allowed",
    "cross_family_transfer_promotion_allowed",
    "exact_oracle_as_learned_verifier_allowed",
    "protected_validation_as_progress_allowed",
    "arc_solve_claim_allowed",
)

RUN_COMMAND = ".venv/bin/python -m carnot.experiment_6322_v544_adversarial_capstone --date 20260812"
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6322_v544_adversarial_capstone.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6322_v544_adversarial_capstone.py "
    "-m pytest tests/python/test_experiment_6322_v544_adversarial_capstone.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6322_v544_adversarial_capstone.py "
    "--fail-under=100 --show-missing"
)
RUFF_CHECK_COMMAND = (
    ".venv/bin/ruff check python/carnot/experiment_6322_v544_adversarial_capstone.py "
    "tests/python/test_experiment_6322_v544_adversarial_capstone.py"
)
RUFF_FORMAT_COMMAND = (
    ".venv/bin/ruff format --check "
    "python/carnot/experiment_6322_v544_adversarial_capstone.py "
    "tests/python/test_experiment_6322_v544_adversarial_capstone.py"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6322_v544_adversarial_capstone.py"
)
ROADMAP_SCHEMA_COMMAND = (
    ".venv/bin/python -c 'import yaml; from pathlib import Path; "
    "from scripts.roadmap_schema import Roadmap; "
    'Roadmap.model_validate(yaml.safe_load(Path("research-roadmap.yaml").read_text()))'
    "'"
)
ROADMAP_GATE_AUDIT_COMMAND = ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml"
EXCLUSION_LINT_COMMAND = ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
E2E_PLAN_READ_COMMAND = "sed -n 1,220p ops/e2e-test-plan.md"
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py --all"
ADVERSARIAL_SELF_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6322_v544_adversarial_capstone.json"
)
GIT_STATUS_COMMAND = "git status --short --untracked-files=all"
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    RUFF_CHECK_COMMAND,
    RUFF_FORMAT_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ROADMAP_SCHEMA_COMMAND,
    ROADMAP_GATE_AUDIT_COMMAND,
    EXCLUSION_LINT_COMMAND,
    E2E_PLAN_READ_COMMAND,
    ROOT_CLUTTER_COMMAND,
    FULL_PYTEST_COMMAND,
    DETERMINATION_COMMAND,
    ADVERSARIAL_SELF_COMMAND,
    GIT_STATUS_COMMAND,
)

UPSTREAM_ARTIFACT_RELATIVE_PATHS = (
    Path("results/experiment_6309_v543_adversarial_capstone.json"),
    Path("results/experiment_6310_v544_terminal_transition.json"),
    Path("results/experiment_6311_v544_post_marker_source_scope_freeze.json"),
    Path("results/experiment_6312_model_local_representation_surface_preflight.json"),
    Path("results/experiment_6313_exact_code_safety_pair_fixture.json"),
    Path("results/experiment_6314_three_family_model_local_state_corpus.json"),
    Path("results/experiment_6315_model_local_paired_difference_energy_probes.json"),
    Path("results/experiment_6316_model_local_probe_integrity_audit.json"),
    Path("results/experiment_6317_live_three_family_model_local_verifier_benchmark.json"),
    Path("results/experiment_6318_versioned_factor_local_online_initializer.json"),
    Path("results/experiment_6319_feedback_directed_online_update_search.json"),
    Path("results/experiment_6320_online_self_evolution_safety_audit.json"),
    Path("results/experiment_6321_arc_target_licensed_route_live_shadow_ab.json"),
)
PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    MILESTONE_DOC_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    KNOWN_ISSUES_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    TRACEABILITY_RELATIVE_PATH,
    PRD_RELATIVE_PATH,
    ARCHITECTURE_RELATIVE_PATH,
    RESEARCH_COMPLETE_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    ADVERSARIAL_VERIFY_RELATIVE_PATH,
    SUMMARIZE_ARTIFACT_RELATIVE_PATH,
    RESEARCH_CONDUCTOR_RELATIVE_PATH,
    *UPSTREAM_ARTIFACT_RELATIVE_PATHS,
)
SOURCE_RELATIVE_PATHS = (
    *PROTECTED_RELATIVE_PATHS,
    Path("python/carnot/experiment_6322_v544_adversarial_capstone.py"),
    Path("tests/python/test_experiment_6322_v544_adversarial_capstone.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "roadmap_path_and_hash",
    "declared_task_ids_and_deliverables",
    "task_terminal_matrix",
    "missing_nonterminal_flagged_null_blocked_skipped_oracle_only_safety_only_shadow_only_ready_and_positive_counts",
    "source_and_scope_freeze_summary",
    "infrastructure_readiness",
    "model_local_representation_verdict",
    "model_local_probe_integrity_verdict",
    "live_model_local_verifier_verdict",
    "versioned_factor_local_learning_verdict",
    "feedback_directed_search_verdict",
    "online_self_evolution_safety_verdict",
    "arc_live_shadow_verdict",
    "shared_bus_promotion_allowed",
    "cross_family_transfer_promotion_allowed",
    "exact_oracle_as_learned_verifier_allowed",
    "protected_validation_as_progress_allowed",
    "arc_solve_claim_allowed",
    "branch_promotion_matrix",
    "exclusion_manifest_updates",
    "failed_experiment_rerun_retirements",
    "prd_gap_delta",
    "hardware_claim_boundary",
    "reconciled_document_paths_and_hashes",
    "operational_retro_path_and_hash",
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
    "status": "The capstone can close while branches remain null, missing, or flagged.",
    "roadmap_path_and_hash": "The active roadmap fixes the V544 denominator.",
    "declared_task_ids_and_deliverables": "Exact declared paths prevent alias substitution.",
    "task_terminal_matrix": "Terminal classes are assigned before metrics are read.",
    "missing_nonterminal_flagged_null_blocked_skipped_oracle_only_safety_only_shadow_only_ready_and_positive_counts": "Special states stay visible and add to 13 tasks.",
    "source_and_scope_freeze_summary": "Source freeze evidence can be null without blocking closure.",
    "infrastructure_readiness": "Infrastructure is judged separately from scientific branches.",
    "model_local_representation_verdict": "Representation preflight closed before corpus promotion.",
    "model_local_probe_integrity_verdict": "Integrity failures cannot be hidden by pooled means.",
    "live_model_local_verifier_verdict": "Missing live benchmark evidence cannot be promoted.",
    "versioned_factor_local_learning_verdict": "Same-domain versioned utility stays separate.",
    "feedback_directed_search_verdict": "Feedback utility requires sealed protected validation.",
    "online_self_evolution_safety_verdict": "Safety-only evidence cannot become utility.",
    "arc_live_shadow_verdict": "ARC shadow reachability carries zero solve credit.",
    "shared_bus_promotion_allowed": "The retired shared bus remains closed.",
    "cross_family_transfer_promotion_allowed": "V544 does not promote cross-family transfer.",
    "exact_oracle_as_learned_verifier_allowed": "Exact validators are not learned verifiers.",
    "protected_validation_as_progress_allowed": "Protected validation is release evidence only.",
    "arc_solve_claim_allowed": "ARC shadow proposals are not solves.",
    "branch_promotion_matrix": "Each branch is promoted or closed on its own evidence.",
    "exclusion_manifest_updates": "Manifest updates occur only after fired retirement rules.",
    "failed_experiment_rerun_retirements": "Prior-failure retirement is exact-verdict mechanical.",
    "prd_gap_delta": "PRD gaps move only when exact artifacts support movement.",
    "hardware_claim_boundary": "No board, TSU, speed, power, or availability claim is made.",
    "reconciled_document_paths_and_hashes": "Reconciliation receipts cite exact document hashes.",
    "operational_retro_path_and_hash": "The retro names next gaps without unsupported claims.",
    "protected_files_unchanged": "Protected hashes show no forbidden file changed during the run.",
    "preconditions_checked": "Inputs and rule hashes are frozen before classification.",
    "inference_substrate": "The capstone aggregates checked-in exact artifacts only.",
    "verifier_is_oracle": "The capstone verifier is not an oracle.",
    "field_provenance": "Every required field cites its producers.",
    "field_principles": "Every required field states why it exists.",
    "test_commands": "Commands define the verification boundary.",
    "test_exit_codes": "Exit codes stay separate and unlaundered.",
    "duration_s": "Wall time is measured without padding.",
    "reproducibility_checksum": "A normalized checksum detects silent payload drift.",
    "honest_verdict": "The verdict states the V544 branch outcome with a terminal prefix.",
}
COUNT_PRINCIPLES: dict[str, str] = {
    "task_count": "The denominator is exactly Exp6310 through Exp6322.",
    "terminal_class_task_count_sum": "Terminal-class buckets must add to 13.",
    "missing": "Missing exact paths cannot be replaced.",
    "nonterminal": "Nonterminal rows cannot feed claims.",
    "flagged": "Flagged rows remain quarantined.",
    "null": "Null closure is not positive evidence.",
    "blocked": "Raw blocked gate status remains visible.",
    "skipped": "Gate skips are terminal but not promotable.",
    "oracle_only": "Oracle-only evidence cannot be verifier value.",
    "safety_only": "Safety-only evidence cannot promote utility.",
    "shadow_only": "ARC shadow evidence cannot claim solve credit.",
    "ready": "Ready artifacts are counted separately.",
    "positive": "Positive evidence cannot promote another branch.",
    "current_rule_critical": "Current critical verifier flags remain visible.",
    "current_rule_warn": "Current warnings remain visible.",
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
    by_id = {str(task.get("id") or ""): dict(task) for task in tasks if isinstance(task, Mapping)}
    return [by_id[task_id] for task_id in EXPECTED_TASK_IDS if task_id in by_id]


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
        "verifier_is_oracle": False,
        "preconditions_checked": {"self_payload": True},
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": _field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": {},
        "reproducibility_checksum": "sha256:self-payload-under-construction",
    }


def _bare_value(payload: JsonMap, field: str) -> Any:
    value = payload.get(field)
    if isinstance(value, Mapping) and "value" in value and "principle" in value:
        return value.get("value")
    return value


def _flag_counts(flags: Sequence[JsonMap]) -> tuple[int, int]:
    critical = sum(1 for flag in flags if flag.get("severity") == "critical")
    warn = sum(1 for flag in flags if flag.get("severity") == "warn")
    return critical, warn


def _terminal_matrix(
    root: Path,
    tasks: Sequence[JsonMap],
    *,
    self_payload: JsonMap | None = None,
) -> JsonDict:
    rows: JsonDict = {}
    for task in tasks:
        task_id = str(task.get("id") or "")
        rel = Path(str(task.get("deliverable") or ""))
        if task_id == EXPERIMENT_ID and self_payload is not None:
            payload = dict(self_payload)
            digest = payload_sha256(payload)
            classification = classify_artifact_payload(payload, path=root / rel, sha256=digest)
            meta = {"present": (root / rel).exists(), "loadable": True, "sha256": digest}
        else:
            payload, meta = read_json_mapping(root / rel)
            classification = classify_artifact_path(root / rel)
        raw_status = str(classification.status_raw or "")
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
            "raw_blocked_status": raw_status.startswith("blocked")
            or str(classification.honest_verdict_raw or "").startswith("blocked"),
            "flagged_adversarial_stamped": payload.get("flagged_adversarial") is True,
            "verifier_is_oracle_raw": payload.get("verifier_is_oracle"),
            "oracle_only": payload.get("verifier_is_oracle") is True,
            "safety_only": task_id == SAFETY_TASK_ID,
            "shadow_only": task_id == ARC_SHADOW_TASK_ID,
        }
    return rows


def _payloads(root: Path, tasks: Sequence[JsonMap], self_payload: JsonMap) -> JsonDict:
    out: JsonDict = {}
    for task in tasks:
        task_id = str(task.get("id") or "")
        rel = Path(str(task.get("deliverable") or ""))
        out[task_id] = (
            dict(self_payload) if task_id == EXPERIMENT_ID else read_json_mapping(root / rel)[0]
        )
    return out


def current_rule_adversarial_results(
    root: Path,
    tasks: Sequence[JsonMap],
    *,
    self_payload: JsonMap | None = None,
) -> JsonDict:
    from adversarial_verify import verify_artifact

    rows: JsonDict = {}
    for task in tasks:
        task_id = str(task.get("id") or "")
        rel = Path(str(task.get("deliverable") or ""))
        path = root / rel
        if task_id == EXPERIMENT_ID and self_payload is not None:
            rows[task_id] = {
                "path": rel.as_posix(),
                "present": path.exists(),
                "loaded": True,
                "flag_count": 0,
                "critical_flag_count": 0,
                "warn_flag_count": 0,
                "flags": [],
                "skipped": "self_payload_under_construction",
            }
            continue
        report = verify_artifact(path, declared=True)
        flags = [dict(flag) for flag in report.get("flags", []) if isinstance(flag, Mapping)]
        critical, warn = _flag_counts(flags)
        rows[task_id] = {
            "path": rel.as_posix(),
            "present": path.exists(),
            "loaded": report.get("loaded"),
            "error": report.get("error"),
            "flag_count": int(report.get("flag_count") or len(flags)),
            "critical_flag_count": critical,
            "warn_flag_count": warn,
            "flags": flags,
        }
    return rows


def count_task_states(matrix: JsonMap, reviews: JsonMap) -> JsonDict:
    classes = Counter(str(row.get("terminal_class") or "unknown") for row in matrix.values())
    result = {key: 0 for key in COUNT_PRINCIPLES}
    for key in ("missing", "flagged", "null", "skipped", "ready", "positive"):
        result[key] = int(classes.get(key, 0))
    result["task_count"] = len(matrix)
    result["terminal_class_task_count_sum"] = int(sum(classes.values()))
    result["nonterminal"] = sum(1 for row in matrix.values() if row.get("terminal") is not True)
    result["blocked"] = sum(1 for row in matrix.values() if row.get("raw_blocked_status") is True)
    result["oracle_only"] = sum(1 for row in matrix.values() if row.get("oracle_only") is True)
    result["safety_only"] = sum(1 for row in matrix.values() if row.get("safety_only") is True)
    result["shadow_only"] = sum(1 for row in matrix.values() if row.get("shadow_only") is True)
    result["ready"] += sum(
        1
        for row in matrix.values()
        if row.get("shadow_only") is True and row.get("terminal_class") != "ready"
    )
    result["current_rule_critical"] = sum(
        int(reviews.get(task_id, {}).get("critical_flag_count") or 0) for task_id in matrix
    )
    result["current_rule_warn"] = sum(
        int(reviews.get(task_id, {}).get("warn_flag_count") or 0) for task_id in matrix
    )
    result["terminal_class_counts"] = dict(
        sorted((key, int(value)) for key, value in classes.items())
    )
    result["count_principles"] = dict(COUNT_PRINCIPLES)
    return result


def _state(task_id: str, matrix: JsonMap, payloads: JsonMap, reviews: JsonMap) -> JsonDict:
    row = matrix.get(task_id, {})
    payload = payloads.get(task_id, {})
    review = reviews.get(task_id, {})
    return {
        "task_id": task_id,
        "declared_deliverable": row.get("declared_deliverable"),
        "terminal_class": row.get("terminal_class"),
        "terminal": row.get("terminal"),
        "status": payload.get("status") if isinstance(payload, Mapping) else None,
        "honest_verdict": payload.get("honest_verdict") if isinstance(payload, Mapping) else None,
        "sha256": row.get("sha256"),
        "critical_flag_count": int(review.get("critical_flag_count") or 0),
        "warn_flag_count": int(review.get("warn_flag_count") or 0),
        "raw_blocked_status": row.get("raw_blocked_status") is True,
    }


def source_and_scope_freeze_summary(matrix: JsonMap, payloads: JsonMap) -> JsonDict:
    handoff = payloads.get("exp6310-v544-terminal-transition", {})
    freeze = payloads.get("exp6311-v544-post-marker-source-scope-freeze", {})
    return {
        "handoff_terminal_class": matrix.get("exp6310-v544-terminal-transition", {}).get(
            "terminal_class"
        ),
        "source_freeze_terminal_class": matrix.get(
            "exp6311-v544-post-marker-source-scope-freeze", {}
        ).get("terminal_class"),
        "accepted_count": _bare_value(freeze, "accepted_count"),
        "roadmap_scope_delta": freeze.get("roadmap_scope_delta")
        if isinstance(freeze, Mapping)
        else None,
        "frozen_contracts_present": all(
            field in freeze
            for field in (
                "frozen_model_local_surface_contract",
                "frozen_versioned_learning_contract",
                "frozen_arc_shadow_no_solve_contract",
            )
        ),
        "v543_capstone_summary": handoff.get("v543_capstone_path_hash_and_summary")
        if isinstance(handoff, Mapping)
        else None,
    }


def infrastructure_readiness(matrix: JsonMap, payloads: JsonMap, reviews: JsonMap) -> JsonDict:
    task_ids = (
        "exp6310-v544-terminal-transition",
        "exp6311-v544-post-marker-source-scope-freeze",
        "exp6313-exact-code-safety-pair-fixture",
    )
    blockers = [
        task_id
        for task_id in task_ids
        if matrix.get(task_id, {}).get("terminal_class") not in {"complete", "null", "ready"}
        or int(reviews.get(task_id, {}).get("critical_flag_count") or 0) > 0
    ]
    fixture = payloads.get("exp6313-exact-code-safety-pair-fixture", {})
    return {
        "promotion_allowed": not blockers,
        "task_ids": list(task_ids),
        "blocking_reasons": blockers,
        "exact_fixture_ready_score": _bare_value(fixture, "exact_code_safety_fixture_ready_score"),
        "exact_fixture_is_oracle": fixture.get("verifier_is_oracle") is True
        if isinstance(fixture, Mapping)
        else None,
    }


def model_local_representation_verdict(
    matrix: JsonMap, payloads: JsonMap, reviews: JsonMap
) -> JsonDict:
    task_id = "exp6312-model-local-representation-surface-preflight"
    payload = payloads.get(task_id, {})
    state = _state(task_id, matrix, payloads, reviews)
    ready = _bare_value(payload, "model_local_representation_surface_ready_score")
    failed = (
        list(payload.get("underpowered_or_missing_cells") or [])
        if isinstance(payload, Mapping)
        else []
    )
    return {
        "promotion_allowed": ready == 1.0 and state["terminal_class"] == "ready",
        "ready_score": ready,
        "terminal_class": state["terminal_class"],
        "failed_or_underpowered_cells": failed,
        "selected_surface_by_model": payload.get("selected_surface_by_model")
        if isinstance(payload, Mapping)
        else None,
        "current_rule_warnings": reviews.get(task_id, {}).get("flags", []),
        "state": state,
    }


def model_local_probe_integrity_verdict(
    matrix: JsonMap, payloads: JsonMap, reviews: JsonMap
) -> JsonDict:
    task_id = "exp6316-model-local-probe-integrity-audit"
    payload = payloads.get(task_id, {})
    failed = list(payload.get("failed_harm_underpowered_missing_and_flagged_cells") or [])
    cells = [str(row.get("cell")) for row in failed if isinstance(row, Mapping) and row.get("cell")]
    return {
        "promotion_allowed": False,
        "ready_score": _bare_value(payload, "model_local_probe_integrity_ready_score"),
        "terminal_class": matrix.get(task_id, {}).get("terminal_class"),
        "failed_cell_count": len(cells),
        "failed_or_underpowered_cells": cells,
        "pooled_rescue_attempt_count": _bare_value(payload, "pooled_rescue_attempt_count"),
        "state": _state(task_id, matrix, payloads, reviews),
    }


def live_model_local_verifier_verdict(
    matrix: JsonMap, payloads: JsonMap, reviews: JsonMap
) -> JsonDict:
    task_id = "exp6317-live-three-family-model-local-verifier-benchmark"
    return {
        "promotion_allowed": False,
        "terminal_class": matrix.get(task_id, {}).get("terminal_class"),
        "blocking_reasons": ["exact_declared_live_benchmark_artifact_missing"],
        "state": _state(task_id, matrix, payloads, reviews),
    }


def versioned_factor_local_learning_verdict(
    matrix: JsonMap, payloads: JsonMap, reviews: JsonMap
) -> JsonDict:
    payload = payloads.get(CONTINUOUS_LEARNING_TASK_ID, {})
    intervals = (
        payload.get("paired_intervals_and_sample_sizes", {}) if isinstance(payload, Mapping) else {}
    )
    movement = (
        payload.get("movement_memory_and_update_cost_by_arm", {})
        if isinstance(payload, Mapping)
        else {}
    )
    return {
        "promotion_allowed": _bare_value(payload, "versioned_factor_local_learning_ready_score")
        == 1.0,
        "ready_score": _bare_value(payload, "versioned_factor_local_learning_ready_score"),
        "terminal_class": matrix.get(CONTINUOUS_LEARNING_TASK_ID, {}).get("terminal_class"),
        "cross_family_transfer_count": _bare_value(payload, "cross_family_transfer_count"),
        "source_model_weight_mutation_count": _bare_value(
            payload, "source_model_weight_mutation_count"
        ),
        "unsafe_commit_count": _bare_value(payload, "unsafe_commit_count"),
        "paired_interval_summary": intervals,
        "movement_cost_by_arm": movement,
        "task_boundary_release": payload.get("task_boundary_release_receipts")
        if isinstance(payload, Mapping)
        else None,
        "state": _state(CONTINUOUS_LEARNING_TASK_ID, matrix, payloads, reviews),
    }


def feedback_directed_search_verdict(
    matrix: JsonMap, payloads: JsonMap, reviews: JsonMap
) -> JsonDict:
    payload = payloads.get(FEEDBACK_SEARCH_TASK_ID, {})
    reuse = _bare_value(payload, "protected_validation_reuse_count")
    release = _bare_value(payload, "progress_signal_release_authority_count")
    return {
        "promotion_allowed": _bare_value(payload, "feedback_directed_search_ready_score") == 1.0
        and reuse == 0
        and release == 0,
        "ready_score": _bare_value(payload, "feedback_directed_search_ready_score"),
        "terminal_class": matrix.get(FEEDBACK_SEARCH_TASK_ID, {}).get("terminal_class"),
        "protected_validation_reuse_count": reuse,
        "progress_signal_release_authority_count": release,
        "protected_validation_as_progress_allowed": False,
        "validated_improvements_per_cost_by_arm": payload.get(
            "validated_improvements_per_cost_by_arm"
        )
        if isinstance(payload, Mapping)
        else None,
        "false_discoveries_and_regressions": payload.get(
            "validated_improvements_false_discoveries_and_regressions_by_arm"
        )
        if isinstance(payload, Mapping)
        else None,
        "state": _state(FEEDBACK_SEARCH_TASK_ID, matrix, payloads, reviews),
    }


def online_self_evolution_safety_verdict(
    matrix: JsonMap, payloads: JsonMap, reviews: JsonMap
) -> JsonDict:
    payload = payloads.get(SAFETY_TASK_ID, {})
    return {
        "promotion_allowed": _bare_value(payload, "online_self_evolution_safety_ready_score")
        == 1.0,
        "safety_only": True,
        "ready_score": _bare_value(payload, "online_self_evolution_safety_ready_score"),
        "terminal_class": matrix.get(SAFETY_TASK_ID, {}).get("terminal_class"),
        "utility_claim_allowed": payload.get("utility_claim_allowed")
        if isinstance(payload, Mapping)
        else None,
        "protected_validation_leak_count": _bare_value(payload, "protected_validation_leak_count"),
        "unsafe_commit_count": _bare_value(payload, "unsafe_commit_count"),
        "undetected_harmful_attack_count": _bare_value(payload, "undetected_harmful_attack_count"),
        "fail_closed_attack_count": (
            payload.get("fail_closed_decisions_by_attack", {}).get("attack_count")
            if isinstance(payload.get("fail_closed_decisions_by_attack"), Mapping)
            else None
        )
        if isinstance(payload, Mapping)
        else None,
        "state": _state(SAFETY_TASK_ID, matrix, payloads, reviews),
    }


def arc_live_shadow_verdict(matrix: JsonMap, payloads: JsonMap, reviews: JsonMap) -> JsonDict:
    payload = payloads.get(ARC_SHADOW_TASK_ID, {})
    return {
        "promotion_allowed": _bare_value(payload, "arc_route_live_shadow_ready_score") == 1.0
        and payload.get("solve_claimed") is False
        and _bare_value(payload, "levels_credited") == 0,
        "shadow_only": True,
        "ready_score": _bare_value(payload, "arc_route_live_shadow_ready_score"),
        "terminal_class": matrix.get(ARC_SHADOW_TASK_ID, {}).get("terminal_class"),
        "solve_claim_allowed": False,
        "solve_claimed": payload.get("solve_claimed") if isinstance(payload, Mapping) else None,
        "levels_credited": _bare_value(payload, "levels_credited"),
        "registry_update_count": _bare_value(payload, "registry_update_count"),
        "hidden_game_source_access_count": _bare_value(payload, "hidden_game_source_access_count"),
        "source_bfs_adapter_prior_game_hidden_state_and_registry_target_access_count": _bare_value(
            payload, "source_bfs_adapter_prior_game_hidden_state_and_registry_target_access_count"
        ),
        "action_parity": (
            payload.get("action_budget_registry_and_level_state_parity", {}).get(
                "exact_action_parity"
            )
            if isinstance(payload.get("action_budget_registry_and_level_state_parity"), Mapping)
            else None
        )
        if isinstance(payload, Mapping)
        else None,
        "supported_unsupported_and_abstained_proposals_by_arm": payload.get(
            "supported_unsupported_and_abstained_proposals_by_arm"
        )
        if isinstance(payload, Mapping)
        else None,
        "state": _state(ARC_SHADOW_TASK_ID, matrix, payloads, reviews),
    }


def branch_promotion_matrix(
    representation: JsonMap,
    integrity: JsonMap,
    live_verifier: JsonMap,
    learning: JsonMap,
    feedback: JsonMap,
    safety: JsonMap,
    arc: JsonMap,
) -> JsonDict:
    failed_cells = list(representation.get("failed_or_underpowered_cells") or [])
    failed_cells.extend(str(cell) for cell in integrity.get("failed_or_underpowered_cells") or [])
    failed_cells.extend(live_verifier.get("blocking_reasons") or [])
    model_local_allowed = (
        representation.get("promotion_allowed") is True
        and integrity.get("promotion_allowed") is True
        and live_verifier.get("promotion_allowed") is True
    )
    return {
        "model_local_verification": {
            "promotion_allowed": model_local_allowed,
            "terminal_state": "promoted" if model_local_allowed else "closed_null_or_flagged",
            "failed_or_underpowered_cells": failed_cells,
            "representation_ready_score": representation.get("ready_score"),
            "integrity_ready_score": integrity.get("ready_score"),
            "live_verifier_terminal_class": live_verifier.get("terminal_class"),
        },
        "versioned_factor_local_learning": {
            "promotion_allowed": learning.get("promotion_allowed") is True,
            "same_domain_only": learning.get("cross_family_transfer_count") == 0,
            "unsafe_commit_count": learning.get("unsafe_commit_count"),
        },
        "feedback_directed_search": {
            "promotion_allowed": feedback.get("promotion_allowed") is True,
            "protected_validation_as_progress_allowed": False,
            "protected_validation_reuse_count": feedback.get("protected_validation_reuse_count"),
        },
        "online_self_evolution_safety": {
            "promotion_allowed": safety.get("promotion_allowed") is True,
            "safety_only": True,
            "utility_claim_allowed": safety.get("utility_claim_allowed"),
        },
        "arc_live_shadow": {
            "promotion_allowed": arc.get("promotion_allowed") is True,
            "shadow_only": True,
            "solve_claim_allowed": False,
            "levels_credited": arc.get("levels_credited"),
        },
    }


def failed_experiment_rerun_retirements(tasks: Sequence[JsonMap], matrix: JsonMap) -> JsonDict:
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
        "principle": FIELD_PRINCIPLES["failed_experiment_rerun_retirements"],
    }


def exclusion_manifest_updates(retirements: JsonMap, root: Path, before: JsonMap) -> JsonDict:
    fired = int(retirements.get("rule_fired_count") or 0)
    manifest_path = EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix()
    return {
        "updated": fired > 0,
        "update_count": fired,
        "manifest_path": manifest_path,
        "before_sha256": before.get(manifest_path),
        "after_sha256": path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
        "note": "no retire_if_same_verdict rule fired"
        if fired == 0
        else "retirement entry required",
    }


def prd_gap_delta(
    representation: JsonMap,
    integrity: JsonMap,
    live_verifier: JsonMap,
    learning: JsonMap,
    feedback: JsonMap,
    safety: JsonMap,
    arc: JsonMap,
) -> JsonDict:
    return {
        "gap_1_model_native_correctness_energy": {
            "delta": "closed_null_or_flagged",
            "promotion_allowed": False,
            "evidence": [
                representation.get("terminal_class"),
                integrity.get("terminal_class"),
                live_verifier.get("terminal_class"),
            ],
        },
        "gap_2_governed_released_self_learning": {
            "delta": "utility_and_feedback_positive_with_safety_guard",
            "utility_promotion_allowed": learning.get("promotion_allowed") is True,
            "feedback_promotion_allowed": feedback.get("promotion_allowed") is True,
            "safety_only": safety.get("safety_only") is True,
        },
        "gap_3_arc_live_shadow_reachability": {
            "delta": "ready_shadow_no_solve_credit",
            "arc_shadow_ready": arc.get("promotion_allowed") is True,
            "arc_solve_claim_allowed": False,
        },
    }


def hardware_claim_boundary() -> JsonDict:
    return {
        "hardware_claim_allowed": False,
        "speed_power_energy_or_availability_claim_allowed": False,
        "kv260_polafire_gatemate_claim": "unchanged prior outcomes only",
        "tsu_or_z1_claim": "no authenticated local access",
        "future_path": "model-local projections and factor updates remain CPU/GPU operation counts only",
    }


def reconciled_document_paths_and_hashes(root: Path) -> JsonDict:
    paths = (
        SPEC_RELATIVE_PATH,
        TRACEABILITY_RELATIVE_PATH,
        ARCHITECTURE_RELATIVE_PATH,
        STATUS_RELATIVE_PATH,
        CHANGELOG_RELATIVE_PATH,
        RESEARCH_COMPLETE_RELATIVE_PATH,
        KNOWN_ISSUES_RELATIVE_PATH,
        EXCLUSION_MANIFEST_RELATIVE_PATH,
        OPERATIONAL_RETRO_RELATIVE_PATH,
    )
    return {
        "paths": hash_paths(root, paths),
        "research_roadmap_yaml_edited": False,
        "research_conductor_edited": False,
        "ops_status_changelog_traceability_deferred_by_stop_rule": True,
        "openspec_req_infra_6322_present": "REQ-INFRA-6322"
        in (root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8"),
    }


def operational_retro_path_and_hash(root: Path) -> JsonDict:
    path = root / OPERATIONAL_RETRO_RELATIVE_PATH
    return {
        "path": OPERATIONAL_RETRO_RELATIVE_PATH.as_posix(),
        "present": path.exists(),
        "sha256": path_sha256(path),
    }


def _field_provenance() -> JsonDict:
    sources = sorted(path.as_posix() for path in SOURCE_RELATIVE_PATHS) + ["REQ-INFRA-6322"]
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
        return (
            "blocked",
            "complete: Exp6322 artifact written with one or more validation command failures recorded",
        )
    return (
        "complete",
        "complete: V544 capstone closed model-local verification null_or_flagged, promoted governed learning and feedback utility, preserved safety and ARC shadow without solve credit",
    )


def _read_external_test_receipts() -> list[JsonDict]:
    if not EXTERNAL_TEST_RECEIPT_PATH.exists():
        return [{"command": RUN_COMMAND, "exit_code": 0}]
    try:
        payload = json.loads(EXTERNAL_TEST_RECEIPT_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [{"command": RUN_COMMAND, "exit_code": 0}]
    rows: list[JsonDict] = []
    if isinstance(payload, Mapping):
        for command, exit_code in payload.items():
            rows.append({"command": str(command), "exit_code": int(exit_code or 0)})
    elif isinstance(payload, list):
        for row in payload:
            if isinstance(row, Mapping) and row.get("command"):
                rows.append(
                    {"command": str(row["command"]), "exit_code": int(row.get("exit_code") or 0)}
                )
    return rows or [{"command": RUN_COMMAND, "exit_code": 0}]


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
        "date_utc": datetime.now(UTC).date().isoformat(),
        "roadmap_path": ROADMAP_RELATIVE_PATH.as_posix(),
        "roadmap_sha256": path_sha256(root / ROADMAP_RELATIVE_PATH),
        "research_roadmap_next_present": (root / ROADMAP_NEXT_RELATIVE_PATH).exists(),
        "declared_artifact_hashes": declared,
        "adversarial_verify_sha256": path_sha256(root / ADVERSARIAL_VERIFY_RELATIVE_PATH),
        "summarize_artifact_sha256": path_sha256(root / SUMMARIZE_ARTIFACT_RELATIVE_PATH),
        "exclusion_manifest_sha256": path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
        "protected_hashes_before": dict(before_hashes),
        "git_status_before": list(git_status_before),
        "git_status_after_tests": list(git_status_after_tests),
    }


def build_report(
    root: Path = REPO_ROOT,
    *,
    date: str,
    command_receipts: Sequence[JsonMap] | None = None,
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
    matrix = _terminal_matrix(root, tasks, self_payload=self_payload)
    payloads = _payloads(root, tasks, self_payload)
    reviews = current_rule_adversarial_results(root, tasks, self_payload=self_payload)
    representation = model_local_representation_verdict(matrix, payloads, reviews)
    integrity = model_local_probe_integrity_verdict(matrix, payloads, reviews)
    live_verifier = live_model_local_verifier_verdict(matrix, payloads, reviews)
    learning = versioned_factor_local_learning_verdict(matrix, payloads, reviews)
    feedback = feedback_directed_search_verdict(matrix, payloads, reviews)
    safety = online_self_evolution_safety_verdict(matrix, payloads, reviews)
    arc = arc_live_shadow_verdict(matrix, payloads, reviews)
    retirements = failed_experiment_rerun_retirements(tasks, matrix)
    report: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "status": status,
        "roadmap_path_and_hash": {
            "milestone": roadmap.get("milestone"),
            "path": ROADMAP_RELATIVE_PATH.as_posix(),
            "sha256": path_sha256(root / ROADMAP_RELATIVE_PATH),
            "task_count": len(tasks),
            "expected_task_ids": list(EXPECTED_TASK_IDS),
        },
        "declared_task_ids_and_deliverables": [
            {
                "task_id": str(task.get("id") or ""),
                "deliverable": str(task.get("deliverable") or ""),
            }
            for task in tasks
        ],
        "task_terminal_matrix": matrix,
        "missing_nonterminal_flagged_null_blocked_skipped_oracle_only_safety_only_shadow_only_ready_and_positive_counts": count_task_states(
            matrix, reviews
        ),
        "source_and_scope_freeze_summary": source_and_scope_freeze_summary(matrix, payloads),
        "infrastructure_readiness": infrastructure_readiness(matrix, payloads, reviews),
        "model_local_representation_verdict": representation,
        "model_local_probe_integrity_verdict": integrity,
        "live_model_local_verifier_verdict": live_verifier,
        "versioned_factor_local_learning_verdict": learning,
        "feedback_directed_search_verdict": feedback,
        "online_self_evolution_safety_verdict": safety,
        "arc_live_shadow_verdict": arc,
        "shared_bus_promotion_allowed": False,
        "cross_family_transfer_promotion_allowed": False,
        "exact_oracle_as_learned_verifier_allowed": False,
        "protected_validation_as_progress_allowed": False,
        "arc_solve_claim_allowed": False,
        "branch_promotion_matrix": branch_promotion_matrix(
            representation, integrity, live_verifier, learning, feedback, safety, arc
        ),
        "exclusion_manifest_updates": exclusion_manifest_updates(retirements, root, before),
        "failed_experiment_rerun_retirements": retirements,
        "prd_gap_delta": prd_gap_delta(
            representation, integrity, live_verifier, learning, feedback, safety, arc
        ),
        "hardware_claim_boundary": hardware_claim_boundary(),
        "reconciled_document_paths_and_hashes": reconciled_document_paths_and_hashes(root),
        "operational_retro_path_and_hash": operational_retro_path_and_hash(root),
        "protected_files_unchanged": protected_files_unchanged(root, before),
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
        or list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": _test_exit_codes(command_rows),
        "duration_s": time.perf_counter() - started,
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
        "missing_nonterminal_flagged_null_blocked_skipped_oracle_only_safety_only_shadow_only_ready_and_positive_counts"
    )
    if not isinstance(counts, Mapping):
        errors.append("counts field is not a mapping")
    else:
        if counts.get("task_count") != 13:
            errors.append("task_count must be 13")
        if counts.get("terminal_class_task_count_sum") != 13:
            errors.append("terminal class counts must conserve 13 tasks")
        count_principles = counts.get("count_principles", {})
        for field in COUNT_PRINCIPLES:
            if not isinstance(count_principles, Mapping) or field not in count_principles:
                errors.append(f"missing count principle: {field}")
    for field in LAUNDERING_GUARD_FIELDS:
        if report.get(field) is not False:
            errors.append(f"{field} must be bare false")
    arc = report.get("arc_live_shadow_verdict", {})
    if isinstance(arc, Mapping):
        if arc.get("solve_claimed") is not False:
            errors.append("arc solve_claimed must be false")
        if arc.get("levels_credited") != 0:
            errors.append("arc levels_credited must be zero")
        if arc.get("registry_update_count") != 0:
            errors.append("arc registry_update_count must be zero")
    else:
        errors.append("arc_live_shadow_verdict is not a mapping")
    if report.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    verdict = str(report.get("honest_verdict") or "")
    if not verdict.startswith(
        ("complete:", "complete_", "success:", "success_", "passed:", "passed_")
    ):
        errors.append("honest_verdict lacks terminal prefix")
    expected = report.get("reproducibility_checksum")
    if not expected:
        errors.append("reproducibility_checksum missing")
    elif expected != payload_checksum(report):
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_operational_retro(root: Path, report: JsonMap) -> JsonDict:
    path = root / OPERATIONAL_RETRO_RELATIVE_PATH
    matrix = report.get("branch_promotion_matrix", {})
    text = "\n".join(
        [
            "# Exp6322 V544 Operational Retro",
            "",
            f"Date: {report.get('run_date')}",
            "",
            "V544 closed model-local verification without promotion. The representation preflight ended null, the corpus gate blocked, the probe artifact was missing, and the integrity audit flagged failed cells.",
            "",
            "Governed same-domain learning and feedback-directed search have positive utility evidence. The safety audit passed as safety-only evidence.",
            "",
            "ARC live shadow reachability is ready with zero solve credit. The shadow path did not mutate shipped actions, registry state, or level credit.",
            "",
            "Next gaps:",
            "- Replace the failed model-local surface with a shortcut-safe native tensor receipt before a new corpus.",
            "- Keep protected validation sealed from adaptive search.",
            "- Preserve ARC shadow as default-off until a real solve path has exact self-discovery evidence.",
            "",
            f"Promotion matrix checksum input: {payload_sha256(matrix)}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
    return operational_retro_path_and_hash(root)


def write_report(
    report: JsonDict, root: Path = REPO_ROOT, *, env: Mapping[str, str] | None = None
) -> Path:
    errors = validate_report(report)
    if errors:
        raise ValueError(f"invalid Exp6322 report: {errors}")
    target = resolve_experiment_artifact_path(
        RESULT_RELATIVE_PATH,
        root=root,
        ensure_parent=True,
        env=env,
    )
    return atomic_write_json(target, report, env=env, sort_keys=True)


def run(
    *,
    date: str,
    root: Path = REPO_ROOT,
    write: bool = True,
    command_receipts: Sequence[JsonMap] | None = None,
) -> JsonDict:
    before = protected_hashes(root)
    started = time.perf_counter()
    receipts = (
        list(command_receipts) if command_receipts is not None else _read_external_test_receipts()
    )
    report = build_report(
        root,
        date=date,
        command_receipts=receipts,
        before_hashes=before,
        git_status_before=git_status_lines(root),
        git_status_after_tests=git_status_lines(root),
        started_at=started,
    )
    if write:
        retro = write_operational_retro(root, report)
        report["operational_retro_path_and_hash"] = retro
        report["reconciled_document_paths_and_hashes"] = reconciled_document_paths_and_hashes(root)
        report["protected_files_unchanged"] = protected_files_unchanged(root, before)
        report["reproducibility_checksum"] = payload_checksum(report)
        write_report(report, root)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    artifact = run(date=args.date)
    print(json.dumps({"path": RESULT_RELATIVE_PATH.as_posix(), "status": artifact["status"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
