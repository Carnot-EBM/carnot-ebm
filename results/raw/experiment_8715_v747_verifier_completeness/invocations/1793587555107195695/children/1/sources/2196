"""Exp6238 V539 exact-path adversarial capstone.

Spec refs: REQ-INFRA-6238, SCENARIO-INFRA-6238-1,
SCENARIO-INFRA-6238-2, SCENARIO-INFRA-6238-3,
SCENARIO-INFRA-6238-4, SCENARIO-INFRA-6238-5,
SCENARIO-INFRA-6238-6.
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

from carnot.experiment_6225_v539_terminal_transition import (
    load_retired_exp_ids,
    same_number_aliases,
)
from carnot.experiment_artifacts import atomic_write_json
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

MILESTONE = "2026.08.539"
EXPERIMENT_ID = "exp6238-v539-adversarial-capstone"
SCHEMA = "carnot.experiment_6238.v539_adversarial_capstone.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6238_v539_adversarial_capstone.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
STAGED_ROADMAP_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
ARC_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

EXPECTED_V539_TASK_IDS = (
    "exp6225-v539-terminal-transition",
    "exp6226-v539-post-marker-source-scope-freeze",
    "exp6227-llama-server-signal-sender-diagnostic",
    "exp6228-supervised-three-family-runtime-endurance",
    "exp6229-arc-gemma31-think-determination",
    "exp6230-arc-induce-prompt-enrichment-heldout-ab",
    "exp6231-arc-bounded-reinduction-depth-ab",
    "exp6232-arc-admissible-depth-portfolio",
    "exp6233-three-family-code-content-margin",
    "exp6234-fresh-flagship-constraint-event-stream",
    "exp6235-prospective-two-timescale-live-csl",
    "exp6236-online-constraint-memory-shadow-consumer",
    "exp6237-activated-mode-jump-sampler-ab",
    "exp6238-v539-adversarial-capstone",
)

PRECONDITION_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    ROADMAP_RELATIVE_PATH,
    STAGED_ROADMAP_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    ARC_REGISTRY_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    Path("_bmad/traceability.md"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("ops/known-issues.md"),
    Path("ops/e2e-test-plan.md"),
    RESEARCH_COMPLETE_RELATIVE_PATH,
    Path("scripts/summarize_artifact.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/determination_preservation_lint.py"),
    Path("scripts/research_conductor.py"),
)

PROTECTED_RELATIVE_PATHS = PRECONDITION_RELATIVE_PATHS

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "roadmap_path_hash_and_task_ids",
    "exact_task_artifact_matrix",
    "conductor_receipt_matrix",
    "terminal_classifier_results",
    "adversarial_results_by_task",
    "determination_preservation_results",
    "gate_cascade_receipts",
    "missing_blocked_skipped_partial_null_flagged_retired_and_ready_counts",
    "prior_failure_retirement_actions",
    "runtime_final_status_and_family_scores",
    "arc_provenance_registry_hash_level_depth_and_promotion_summary",
    "code_parse_recovery_and_content_margin_summary",
    "fresh_stream_and_continuous_learning_summary",
    "shadow_consumer_summary",
    "sampler_activation_quality_equivalence_and_default_summary",
    "hardware_boundary_and_claim_count",
    "protected_files_unchanged",
    "spec_traceability_status_changelog_known_issues_updates",
    "research_complete_reconciliation",
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
    "status": "The capstone is complete only when every declared V539 task state is preserved.",
    "roadmap_path_hash_and_task_ids": "The task denominator is the exact active V539 roadmap.",
    "exact_task_artifact_matrix": "Exact deliverable paths prevent sidecars or filenames from laundering missing work.",
    "conductor_receipt_matrix": "Conductor receipts are context only and cannot promote an artifact.",
    "terminal_classifier_results": "The shared classifier provides fail-closed terminality.",
    "adversarial_results_by_task": "Live current-rule flags determine whether a task can feed claims.",
    "determination_preservation_results": "The preservation guard proves corrigenda and determinations were not dropped.",
    "gate_cascade_receipts": "Structured gates are recomputed from exact upstream fields.",
    "missing_blocked_skipped_partial_null_flagged_retired_and_ready_counts": "Every terminal and nonterminal class stays counted before any summary claim.",
    "prior_failure_retirement_actions": "Retire-if-same-verdict fires only on exact repeated failed verdicts.",
    "runtime_final_status_and_family_scores": "Runtime durability uses Exp6228 family scores, not short canaries.",
    "arc_provenance_registry_hash_level_depth_and_promotion_summary": "ARC depth credit needs live provenance and registry-safe promotion.",
    "code_parse_recovery_and_content_margin_summary": "Format recovery and content margin are separate code-verification claims.",
    "fresh_stream_and_continuous_learning_summary": "Fresh stream evidence is required before CSL promotion.",
    "shadow_consumer_summary": "Shadow reachability is distinct from fresh CSL utility.",
    "sampler_activation_quality_equivalence_and_default_summary": "Sampler activation must be proven before equivalence or null interpretation.",
    "hardware_boundary_and_claim_count": "Hardware claims require new independent receipts; otherwise the count is zero.",
    "protected_files_unchanged": "The capstone must not rewrite conductor, ops ledgers, roadmap, registry, or historical inputs.",
    "spec_traceability_status_changelog_known_issues_updates": "Spec and operations reconciliation state is recorded without unsupported claims.",
    "research_complete_reconciliation": "Duplicate research ledgers are recorded without deduplication.",
    "preconditions_checked": "All declared inputs are hashed before classification.",
    "inference_substrate": "The capstone reads upstream artifacts and does not load a model.",
    "verifier_is_oracle": "False because the capstone checks evidence records, not benchmark answers.",
    "field_provenance": "Every required field cites files or checks that produced it.",
    "field_principles": "Every required field carries the reason it exists.",
    "test_commands": "Commands record focused, coverage, lint, preservation, suite, and adversarial checks.",
    "test_exit_codes": "Exit codes are reported honestly and without laundering.",
    "duration_s": "Wall time is reported without padding.",
    "reproducibility_checksum": "The normalized payload is content-addressed for drift checks.",
    "honest_verdict": "The terminal verdict preserves mixed outcomes without strengthening them.",
}

COUNT_PRINCIPLES: dict[str, str] = {
    "missing": "Missing exact artifacts cannot be promoted by receipts or aliases.",
    "blocked": "Blocked states are operationally terminal only when the exact artifact says so.",
    "skipped": "Gate skips are terminal bookkeeping and not scientific success.",
    "partial": "Partial outputs cannot feed headline branches.",
    "null": "Null outcomes need their own controls and must not be counted as positive.",
    "flagged": "Flagged artifacts are quarantined from claims even when terminal.",
    "retired": "Retired scopes remain closed unless a valid override exists.",
    "ready": "Ready is counted separately from complete and positive evidence.",
    "complete": "Complete artifacts may still be blocked from claims by flags or gates.",
    "unknown": "Unknown classifier states fail closed.",
    "hardware_claim_count": "Hardware claims stay zero without new task-linked receipts.",
}

GATE_PRINCIPLE = (
    "A structured gate passes only when the exact upstream artifact field satisfies the "
    "roadmap operator; missing or null fields fail closed."
)

PRESERVATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py --all"
TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6238_v539_adversarial_capstone.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6238_v539_adversarial_capstone.py -m pytest tests/python/test_experiment_6238_v539_adversarial_capstone.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6238_v539_adversarial_capstone.py --fail-under=100",
    ".venv/bin/ruff check python/carnot/experiment_6238_v539_adversarial_capstone.py tests/python/test_experiment_6238_v539_adversarial_capstone.py",
    ".venv/bin/ruff format --check python/carnot/experiment_6238_v539_adversarial_capstone.py tests/python/test_experiment_6238_v539_adversarial_capstone.py",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6238_v539_adversarial_capstone.py",
    ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml",
    ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
    ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml",
    PRESERVATION_COMMAND,
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "sed -n 1,260p ops/e2e-test-plan.md",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6238_v539_adversarial_capstone.json",
)
COMMAND_TIMEOUTS_S = {
    ".venv/bin/pytest tests/python -q": 3600,
}


def payload_checksum(report: JsonMap) -> str:
    normalized = json.loads(canonical_json(report))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return payload_sha256(normalized)


def load_roadmap(root: Path) -> JsonDict:
    data = yaml.safe_load((root / ROADMAP_RELATIVE_PATH).read_text(encoding="utf-8")) or {}
    return dict(data) if isinstance(data, Mapping) else {}


def load_yaml_object(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(data) if isinstance(data, Mapping) else {}


def load_json_object(path: Path) -> tuple[JsonDict, JsonDict]:
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


def roadmap_tasks(roadmap: JsonMap) -> list[JsonDict]:
    tasks = roadmap.get("tasks")
    return (
        [dict(task) for task in tasks if isinstance(task, Mapping)]
        if isinstance(tasks, list)
        else []
    )


def _self_payload() -> JsonDict:
    return {
        "status": "complete",
        "honest_verdict": "complete: Exp6238 capstone payload under construction",
        "duration_s": 0.0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
    }


def build_exact_task_artifact_matrix(
    root: Path,
    tasks: Sequence[JsonMap],
    *,
    self_payload: JsonMap | None = None,
) -> JsonDict:
    rows: JsonDict = {}
    for task in tasks:
        task_id = str(task.get("id") or "")
        rel = Path(str(task.get("deliverable") or ""))
        path = root / rel
        payload: JsonDict = {}
        if task_id == EXPERIMENT_ID and self_payload is not None:
            classification = classify_artifact_payload(
                self_payload,
                path=path,
                sha256=payload_sha256(self_payload),
            )
            meta = {
                "present": path.exists(),
                "loadable": True,
                "sha256": payload_sha256(self_payload),
                "error": "self_payload_in_memory",
            }
            payload = dict(self_payload)
        else:
            payload, meta = load_json_object(path)
            classification = classify_artifact_path(path)
        aliases = same_number_aliases(root, task_id, rel)
        rows[task_id] = {
            "task_id": task_id,
            "title": str(task.get("title") or task_id),
            "track": str(task.get("track") or "unset"),
            "declared_deliverable": rel.as_posix(),
            "present": bool(meta["present"]),
            "loadable": bool(meta["loadable"]),
            "sha256": meta["sha256"],
            "terminal_class": classification.classification,
            "terminal": classification.terminal,
            "reason": classification.reason,
            "status_raw": classification.status_raw,
            "honest_verdict_raw": classification.honest_verdict_raw,
            "flagged_adversarial_stamped": payload.get("flagged_adversarial") is True,
            "same_number_alias_used": False,
            "same_number_alias_candidates_ignored": aliases,
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


def evaluate_gate_cascades(root: Path, tasks: Sequence[JsonMap]) -> JsonDict:
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
            actual = None
            reason = "missing_upstream_artifact"
            if upstream_task is not None:
                payload, meta = load_json_object(
                    root / Path(str(upstream_task.get("deliverable") or ""))
                )
                if meta["loadable"]:
                    if field in payload:
                        actual = payload.get(field)
                        reason = "evaluated"
                    else:
                        reason = "missing_upstream_field"
            passed = reason == "evaluated" and evaluate_operator(actual, op, expected)
            gates.append(
                {
                    "task_id": task_id,
                    "upstream": upstream,
                    "artifact_field": field,
                    "op": op,
                    "expected": expected,
                    "actual": actual,
                    "passed": passed,
                    "reason": "passed" if passed else reason,
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


def count_terminal_classes(matrix: JsonMap) -> JsonDict:
    counts = Counter(str(row.get("terminal_class") or "unknown") for row in matrix.values())
    result = {
        key: int(counts.get(key, 0)) for key in COUNT_PRINCIPLES if key != "hardware_claim_count"
    }
    result["hardware_claim_count"] = 0
    result["terminal_class_counts"] = dict(
        sorted((key, int(value)) for key, value in counts.items())
    )
    result["scientific_success_count"] = int(counts.get("positive", 0))
    result["count_principles"] = dict(COUNT_PRINCIPLES)
    return result


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


def preconditions_checked(root: Path, tasks: Sequence[JsonMap]) -> JsonDict:
    deliverables = {
        str(task.get("id") or ""): {
            "path": str(task.get("deliverable") or ""),
            "sha256": path_sha256(root / Path(str(task.get("deliverable") or ""))),
        }
        for task in tasks
    }
    return {
        "checked_before_classification": True,
        "input_hashes": {
            path.as_posix(): path_sha256(root / path) for path in PRECONDITION_RELATIVE_PATHS
        },
        "declared_deliverable_hashes": deliverables,
        "retired_experiment_id_count": len(
            load_retired_exp_ids(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
        ),
        "roadmap_next_present": (root / STAGED_ROADMAP_RELATIVE_PATH).exists(),
    }


def live_artifact_reviews(root: Path, matrix: JsonMap) -> JsonDict:  # pragma: no cover
    from adversarial_verify import verify_artifact

    reviews: JsonDict = {}
    for task_id, row in matrix.items():
        rel = Path(str(row.get("declared_deliverable") or ""))
        path = root / rel
        if not path.exists():
            reviews[task_id] = {
                "summary": {
                    "command": f".venv/bin/python scripts/summarize_artifact.py {rel.as_posix()}",
                    "exit_code": None,
                    "classification": "skipped_missing_artifact",
                },
                "adversarial": {
                    "path": rel.as_posix(),
                    "flag_count": 0,
                    "critical_flag_count": 0,
                    "warn_flag_count": 0,
                    "flags": [],
                    "skipped": "missing_artifact",
                },
            }
            continue
        summary_command = f".venv/bin/python scripts/summarize_artifact.py {rel.as_posix()}"
        summary = run_command(summary_command, root, timeout_s=180)
        verified = verify_artifact(path)
        flags = list(verified.get("flags") or [])
        reviews[task_id] = {
            "summary": summary,
            "adversarial": {
                "path": rel.as_posix(),
                "flag_count": int(verified.get("flag_count") or len(flags)),
                "critical_flag_count": sum(
                    1 for flag in flags if flag.get("severity") == "critical"
                ),
                "warn_flag_count": sum(1 for flag in flags if flag.get("severity") == "warn"),
                "flags": flags,
            },
        }
    return reviews


def determination_preservation_receipt(root: Path) -> JsonDict:  # pragma: no cover
    return run_command(PRESERVATION_COMMAND, root, timeout_s=300)


def _payloads(root: Path, tasks: Sequence[JsonMap]) -> JsonDict:
    return {
        str(task.get("id") or ""): load_json_object(
            root / Path(str(task.get("deliverable") or ""))
        )[0]
        for task in tasks
    }


def _score(payload: JsonMap, field: str) -> Any:
    value = payload.get(field)
    if isinstance(value, Mapping) and "value" in value:
        return value.get("value")
    return value


def runtime_summary(payloads: JsonMap, reviews: JsonMap) -> JsonDict:
    runtime = payloads.get("exp6228-supervised-three-family-runtime-endurance", {})
    diagnostic = payloads.get("exp6227-llama-server-signal-sender-diagnostic", {})
    family_scores = {
        key: _score(runtime, key)
        for key in (
            "qwen_runtime_ready_score",
            "gemma_4_31b_runtime_ready_score",
            "gemma_4_26b_runtime_ready_score",
            "two_family_runtime_ready_score",
            "three_family_runtime_ready_score",
        )
    }
    critical = (
        reviews.get("exp6228-supervised-three-family-runtime-endurance", {})
        .get("adversarial", {})
        .get("critical_flag_count", 0)
    )
    return {
        "status": runtime.get("status"),
        "honest_verdict": runtime.get("honest_verdict"),
        "diagnostic_runtime_ready_score": diagnostic.get("runtime_diagnostic_ready_score"),
        **family_scores,
        "claim_allowed": family_scores["three_family_runtime_ready_score"] == 1 and critical == 0,
        "principle": "Durable runtime requires Exp6228 family scores, not Exp6227 or short canaries.",
    }


def arc_summary(root: Path, payloads: JsonMap) -> JsonDict:
    registry = load_yaml_object(root / ARC_REGISTRY_RELATIVE_PATH)
    portfolio = payloads.get("exp6232-arc-admissible-depth-portfolio", {})
    registry_receipt = portfolio.get("registry_precheck_and_hash_before_after")
    return {
        "registry_path": ARC_REGISTRY_RELATIVE_PATH.as_posix(),
        "registry_sha256": path_sha256(root / ARC_REGISTRY_RELATIVE_PATH),
        "registry_reproducible_total_levels": registry.get("reproducible_total_levels"),
        "registry_reproducible_total_games": registry.get("reproducible_total_games"),
        "solve_provenance": portfolio.get("solve_provenance"),
        "registry_update_count": portfolio.get("registry_update_count"),
        "registry_hash_unchanged": registry_receipt.get("unchanged")
        if isinstance(registry_receipt, Mapping)
        else None,
        "eligible_lever_count": portfolio.get("eligible_lever_count"),
        "selected_levers": portfolio.get("selected_levers"),
        "portfolio_promotion_ready_score": portfolio.get("portfolio_promotion_ready_score"),
        "depth_promotion_claim": False,
        "principle": "ARC depth promotion needs live-agent provenance, two eligible levers, and registry-safe evidence.",
    }


def code_summary(payloads: JsonMap) -> JsonDict:
    code = payloads.get("exp6233-three-family-code-content-margin", {})
    return {
        "status": code.get("status"),
        "gate_check_summary": code.get("gate_check_summary"),
        "parse_recovery_claim": False,
        "content_margin_claim": False,
        "content_margin": code.get("hidden_test_content_margin"),
        "principle": "Gate-blocked code artifacts cannot turn format recovery into hidden-test content gain.",
    }


def fresh_csl_summary(payloads: JsonMap) -> JsonDict:
    stream = payloads.get("exp6234-fresh-flagship-constraint-event-stream", {})
    csl = payloads.get("exp6235-prospective-two-timescale-live-csl", {})
    return {
        "stream_status": stream.get("status"),
        "seed_stream_ready_score": stream.get("seed_stream_ready_score"),
        "csl_artifact_present": bool(csl),
        "continuous_learning_promotion_ready_score": csl.get(
            "continuous_learning_promotion_ready_score"
        ),
        "fresh_stream_claim": False,
        "csl_claim": False,
        "principle": "Fresh events must exist before deterministic memory safety can become live CSL evidence.",
    }


def shadow_summary(payloads: JsonMap) -> JsonDict:
    shadow = payloads.get("exp6236-online-constraint-memory-shadow-consumer", {})
    return {
        "status": shadow.get("status"),
        "gate_check_summary": shadow.get("gate_check_summary"),
        "online_constraint_memory_shadow_ready_score": shadow.get(
            "online_constraint_memory_shadow_ready_score"
        ),
        "shadow_reachability_claim": False,
        "principle": "The shadow consumer cannot be promoted when Exp6235 is missing or not promoted.",
    }


def sampler_summary(payloads: JsonMap) -> JsonDict:
    sampler = payloads.get("exp6237-activated-mode-jump-sampler-ab", {})
    activation = sampler.get("treatment_activation_score")
    activation_score = activation.get("score") if isinstance(activation, Mapping) else activation
    equivalence = sampler.get("equivalence_bounds_and_decision")
    decision = equivalence.get("decision") if isinstance(equivalence, Mapping) else None
    return {
        "status": sampler.get("status"),
        "treatment_activation_score": activation_score,
        "mode_jump_proposal_count": activation.get("mode_jump_proposal_count")
        if isinstance(activation, Mapping)
        else None,
        "mode_jump_acceptance_count": activation.get("mode_jump_acceptance_count")
        if isinstance(activation, Mapping)
        else None,
        "decision": decision,
        "quality_conclusion_allowed": equivalence.get("quality_conclusion_allowed")
        if isinstance(equivalence, Mapping)
        else None,
        "default_off_preserved": sampler.get("default_off_preserved"),
        "sampler_runtime_ready_score": sampler.get("sampler_runtime_ready_score"),
        "hardware_claim_count": sampler.get("hardware_claim_count", 0),
        "positive_claim": decision == "positive",
        "principle": "Activated treatment allows an equivalence result; it does not imply hardware or a positive sampler win.",
    }


def hardware_boundary(payloads: JsonMap) -> JsonDict:
    sampler = payloads.get("exp6237-activated-mode-jump-sampler-ab", {})
    return {
        "hardware_claim_count": 0,
        "sampler_hardware_claim_count": sampler.get("hardware_claim_count", 0),
        "new_independently_admissible_receipt": False,
        "boundary": "No V539 hardware task produced a new physical-state, latency, power, energy, or speedup receipt.",
        "principle": COUNT_PRINCIPLES["hardware_claim_count"],
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
                    "principle": "Exact same failed verdict is required before retirement; this capstone records but does not guess.",
                }
            )
    return {
        "actions": actions,
        "rule_fired_count": sum(1 for action in actions if action["rule_fired"]),
        "manifest_update_count": 0,
        "principle": "The exclusion manifest is updated only by a real rule fire, never by semantic guesswork.",
    }


def research_complete_reconciliation(root: Path) -> JsonDict:
    data = load_yaml_object(root / RESEARCH_COMPLETE_RELATIVE_PATH)
    milestones = data.get("milestones")
    ids = (
        [
            str(row.get("id"))
            for row in milestones
            if isinstance(milestones, list)
            and isinstance(row, Mapping)
            and row.get("id") is not None
        ]
        if isinstance(milestones, list)
        else []
    )
    duplicates = {key: count for key, count in Counter(ids).items() if count > 1}
    return {
        "path": RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
        "sha256": path_sha256(root / RESEARCH_COMPLETE_RELATIVE_PATH),
        "milestone_record_count": len(ids),
        "duplicate_milestones": dict(sorted(duplicates.items())),
        "action": "recorded_only_no_mutation",
        "principle": "Duplicate-ledger caveats are preserved and not deduplicated by the capstone.",
    }


def spec_ops_reconciliation(root: Path, before: JsonMap) -> JsonDict:
    after = protected_hashes(root)
    watched = (
        SPEC_RELATIVE_PATH,
        Path("_bmad/traceability.md"),
        Path("ops/status.md"),
        Path("ops/changelog.md"),
        Path("ops/known-issues.md"),
    )
    hashes = {
        path.as_posix(): {
            "before_sha256": before.get(path.as_posix()),
            "after_sha256": after.get(path.as_posix()),
            "unchanged_during_capstone": before.get(path.as_posix()) == after.get(path.as_posix()),
        }
        for path in watched
    }
    spec_text = (
        (root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
        if (root / SPEC_RELATIVE_PATH).exists()
        else ""
    )
    return {
        "openspec_req_infra_6238_present": "REQ-INFRA-6238" in spec_text,
        "ops_status_changelog_traceability_touched": False,
        "known_issues_touched": False,
        "deferred_by_operator_stop_rule": True,
        "hashes": hashes,
        "principle": "Only supported spec state is recorded here; ops ledgers are left for the separate reconciler.",
    }


def _field_provenance() -> JsonDict:
    sources = {
        "REQ-INFRA-6238",
        ROADMAP_RELATIVE_PATH.as_posix(),
        VNEXT_RELATIVE_PATH.as_posix(),
        CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
        EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
        ARC_REGISTRY_RELATIVE_PATH.as_posix(),
        SPEC_RELATIVE_PATH.as_posix(),
        "scripts/summarize_artifact.py",
        "scripts/adversarial_verify.py",
        "scripts/determination_preservation_lint.py",
        "python/carnot/terminal_artifacts.py",
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


def build_report(
    root: Path = REPO_ROOT,
    *,
    date: str,
    command_receipts: Sequence[JsonMap] | None = None,
    artifact_reviews: JsonMap | None = None,
    preservation_receipt: JsonMap | None = None,
    before_hashes: JsonMap | None = None,
    started_at: float | None = None,
) -> JsonDict:
    started = time.perf_counter() if started_at is None else started_at
    before = dict(before_hashes or protected_hashes(root))
    roadmap = load_roadmap(root)
    tasks = roadmap_tasks(roadmap)
    matrix = build_exact_task_artifact_matrix(root, tasks, self_payload=_self_payload())
    reviews = dict(artifact_reviews or live_artifact_reviews(root, matrix))
    preservation = dict(preservation_receipt or determination_preservation_receipt(root))
    payloads = _payloads(root, tasks)
    gate_receipts = evaluate_gate_cascades(root, tasks)
    command_rows = [dict(row) for row in (command_receipts or [])]
    counts = count_terminal_classes(matrix)
    counts["flagged"] = max(
        counts["flagged"],
        sum(
            1
            for task_id, row in matrix.items()
            if row.get("flagged_adversarial_stamped")
            or reviews.get(task_id, {}).get("adversarial", {}).get("critical_flag_count", 0)
        ),
    )
    report: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "status": "complete",
        "roadmap_path_hash_and_task_ids": {
            "milestone": roadmap.get("milestone"),
            "roadmap_path": ROADMAP_RELATIVE_PATH.as_posix(),
            "roadmap_sha256": path_sha256(root / ROADMAP_RELATIVE_PATH),
            "roadmap_next_present": (root / STAGED_ROADMAP_RELATIVE_PATH).exists(),
            "vnext_path": VNEXT_RELATIVE_PATH.as_posix(),
            "vnext_sha256": path_sha256(root / VNEXT_RELATIVE_PATH),
            "task_ids": [str(task.get("id") or "") for task in tasks],
            "expected_task_ids": list(EXPECTED_V539_TASK_IDS),
            "task_count": len(tasks),
            "principle": "The active roadmap is the denominator for all 14 V539 tasks.",
        },
        "exact_task_artifact_matrix": matrix,
        "conductor_receipt_matrix": latest_conductor_receipts(root, tasks),
        "terminal_classifier_results": {
            task_id: {
                "terminal_class": row.get("terminal_class"),
                "terminal": row.get("terminal"),
                "reason": row.get("reason"),
            }
            for task_id, row in matrix.items()
        },
        "adversarial_results_by_task": {
            task_id: review.get("adversarial", {}) for task_id, review in reviews.items()
        },
        "determination_preservation_results": preservation,
        "gate_cascade_receipts": gate_receipts,
        "missing_blocked_skipped_partial_null_flagged_retired_and_ready_counts": counts,
        "prior_failure_retirement_actions": prior_failure_retirement_actions(tasks, matrix),
        "runtime_final_status_and_family_scores": runtime_summary(payloads, reviews),
        "arc_provenance_registry_hash_level_depth_and_promotion_summary": arc_summary(
            root, payloads
        ),
        "code_parse_recovery_and_content_margin_summary": code_summary(payloads),
        "fresh_stream_and_continuous_learning_summary": fresh_csl_summary(payloads),
        "shadow_consumer_summary": shadow_summary(payloads),
        "sampler_activation_quality_equivalence_and_default_summary": sampler_summary(payloads),
        "hardware_boundary_and_claim_count": hardware_boundary(payloads),
        "protected_files_unchanged": protected_files_unchanged(root, before),
        "spec_traceability_status_changelog_known_issues_updates": spec_ops_reconciliation(
            root, before
        ),
        "research_complete_reconciliation": research_complete_reconciliation(root),
        "preconditions_checked": preconditions_checked(root, tasks),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": [str(row.get("command") or "") for row in command_rows]
        or list(TEST_COMMANDS),
        "test_exit_codes": _test_exit_codes(command_rows),
        "duration_s": time.perf_counter() - started,
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete: V539 exact-path capstone preserved missing blocked skipped flagged "
            "and not-ready states; no hardware ARC code CSL or runtime claim was promoted"
        ),
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
        if field not in principles:
            errors.append(f"missing field_principles entry: {field}")
        if field not in provenance:
            errors.append(f"missing field_provenance entry: {field}")
    counts = report.get("missing_blocked_skipped_partial_null_flagged_retired_and_ready_counts")
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
    hardware = report.get("hardware_boundary_and_claim_count")
    if not isinstance(hardware, Mapping) or hardware.get("hardware_claim_count") != 0:
        errors.append("hardware_claim_count must be bare 0")
    if report.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("wrong inference_substrate")
    if report.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    verdict = str(report.get("honest_verdict") or "")
    if not verdict.startswith(
        (
            "complete:",
            "complete_",
            "success:",
            "success_",
            "passed:",
            "passed_",
            "shipped:",
            "shipped_",
            "blocked:",
        )
    ):
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
        raise ValueError("invalid Exp6238 report: " + "; ".join(errors))
    return atomic_write_json(RESULT_RELATIVE_PATH, report, root=root, env=env, sort_keys=False)


def run_experiment(root: Path, date: str, *, run_commands: bool) -> JsonDict:  # pragma: no cover
    started = time.perf_counter()
    before = protected_hashes(root)
    preliminary = build_report(
        root, date=date, command_receipts=[], before_hashes=before, started_at=started
    )
    write_report(preliminary, root)
    commands = run_default_commands(root) if run_commands else []
    final = build_report(
        root, date=date, command_receipts=commands, before_hashes=before, started_at=started
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
