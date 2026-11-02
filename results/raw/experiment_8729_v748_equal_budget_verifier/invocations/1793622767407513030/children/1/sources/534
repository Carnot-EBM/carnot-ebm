"""Exp 2976 intent-preserving DCCD repair protocol.

Spec: REQ-VERIFY-2976, SCENARIO-VERIFY-2976.

This module is intentionally an aggregation artifact, not a live model
experiment. The .279 DCCD rerun proved that stricter structure can look safer
while destroying the candidate stream: Exp 2964 drove false accepts to zero
only because schema-only DCCD produced no passing repairs. The protocol below
turns that failure into explicit downstream gates for Exp 2977.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


ARTIFACT_FILENAME = "experiment_2976_dccd_adaptrack_tracecoder_protocol_v1.json"
RUN_DATE = "20260524"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
DOWNSTREAM_MIN_TASKS = 20
TRACE_COVERAGE_FLOOR = 0.8
LEGACY_MODEL_POLICY = (
    "Legacy small models are allowed only for CPU smoke tests and must never "
    "be reported as headline repair results."
)
MANDATORY_HEADLINE_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "intent_preserving_repair_protocol_ready",
    "trace_execution_plan_ready",
    "downstream_min_tasks",
    "required_model_specs",
    "mandatory_headline_model_ids",
    "legacy_model_policy",
    "baseline_conditions",
    "repair_manifest_schema",
    "acceptance_gates",
    "schema_regression_guard",
    "syntax_regression_guard",
    "false_accept_guard",
    "prior_failure_addressed",
    "inference_substrate",
    "duration_s",
)
SOURCE_ARTIFACTS = (
    {
        "experiment_id": "exp2963",
        "filename": "experiment_2963_dccd_repair_protocol_manifest_v1.json",
        "role": "schema_only_dccd_protocol_manifest",
    },
    {
        "experiment_id": "exp2964",
        "filename": "experiment_2964_sota_dccd_repair_replication_v1.json",
        "role": "failed_live_dccd_rerun_metrics",
    },
    {
        "experiment_id": "exp2952",
        "filename": "experiment_2952_sota_taxonomy_guided_code_repair_eval_v1.json",
        "role": "taxonomy_guided_repair_reference",
    },
    {
        "experiment_id": "exp2953",
        "filename": "experiment_2953_code_verifier_threshold_policy_v1.json",
        "role": "deterministic_verifier_threshold_policy",
    },
)
RESEARCH_SWEEP_TOKENS = (
    "2026-05-24 Post-.279 Planning Sweep",
    "AdapTrack",
    "TraceCoder",
    "Thinking Before Constraining",
    "backtracking",
)


ClockFunc = Callable[[], float]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duration(monotonic: ClockFunc, start: float) -> float:
    return round(monotonic() - start, 6)


def _source_preconditions(repo_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for source in SOURCE_ARTIFACTS:
        path = repo_root / "results" / str(source["filename"])
        present = path.exists()
        if not present:
            missing.append(str(path))
        records.append(
            {
                "resource": str(source["experiment_id"]),
                "path": str(path),
                "role": str(source["role"]),
                "available": present,
                "sha256": _sha256(path) if present else "",
            }
        )
    return records, missing


def _research_sweep_precondition(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "research-references.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    missing_tokens = [token for token in RESEARCH_SWEEP_TOKENS if token not in text]
    return {
        "resource": "post_279_research_sweep",
        "path": str(path),
        "available": not missing_tokens,
        "required_tokens": list(RESEARCH_SWEEP_TOKENS),
        "missing_tokens": missing_tokens,
    }


def _load_sources(repo_root: Path) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for source in SOURCE_ARTIFACTS:
        path = repo_root / "results" / str(source["filename"])
        payloads[str(source["experiment_id"])] = json.loads(path.read_text(encoding="utf-8"))
    return payloads


def _mode_summary(mode_metrics: dict[str, Any], mode: str) -> dict[str, Any]:
    metrics = mode_metrics[mode]
    return {
        "mode": mode,
        "candidate_count": metrics["candidate_count"],
        "pass_at_1": metrics["pass_at_1"],
        "pass_at_k": metrics["pass_at_k"],
        "schema_failure_rate": metrics["schema_failure_rate"],
        "syntax_failure_rate": metrics["syntax_failure_rate"],
        "false_accept_rate": metrics["false_accept_rate"],
        "verifier_acceptance_rate": metrics["verifier_acceptance_rate"],
    }


def _failure_comparison(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    exp2964 = sources["exp2964"]
    mode_metrics = exp2964["mode_metrics"]
    return {
        "exp2963_protocol_ready": sources["exp2963"]["dccd_repair_protocol_ready"],
        "exp2952_taxonomy_guided_pass_at_1_delta": sources["exp2952"][
            "pass_at_1_delta"
        ],
        "exp2953_default_threshold": sources["exp2953"]["selected_default_threshold"],
        "exp2964_baseline": _mode_summary(mode_metrics, "baseline_no_taxonomy"),
        "exp2964_dccd_structured": _mode_summary(mode_metrics, "dccd_structured"),
        "exp2964_taxonomy_guided": _mode_summary(mode_metrics, "taxonomy_guided"),
        "schema_failure_rate_delta": exp2964["schema_failure_rate_delta"],
        "syntax_failure_rate_delta": exp2964["syntax_failure_rate_delta"],
        "false_accept_delta": exp2964["false_accept_delta"],
        "pass_at_1_delta": exp2964["pass_at_1_delta"],
        "pass_at_k_delta": exp2964["pass_at_k_delta"],
    }


def _failure_correlations(comparison: dict[str, Any]) -> dict[str, Any]:
    dccd = comparison["exp2964_dccd_structured"]
    baseline = comparison["exp2964_baseline"]
    return {
        "schema_failures": {
            "correlated_dccd_fields": [
                "mode=dccd_structured",
                "schema_valid=false",
                "parser_status=not_run",
                "schema_errors_present",
            ],
            "baseline_rate": baseline["schema_failure_rate"],
            "dccd_rate": dccd["schema_failure_rate"],
            "delta": comparison["schema_failure_rate_delta"],
            "interpretation": "schema-only DCCD overconstrained manifest emission before runnable code existed",
        },
        "syntax_failures": {
            "correlated_dccd_fields": [
                "mode=dccd_structured",
                "syntax_success=false",
                "parser_status=not_run",
                "test_status=not_run",
            ],
            "baseline_rate": baseline["syntax_failure_rate"],
            "dccd_rate": dccd["syntax_failure_rate"],
            "delta": comparison["syntax_failure_rate_delta"],
            "interpretation": "schema collapse prevented AST parsing, so syntax failures rose instead of being repaired",
        },
        "false_accepts": {
            "correlated_dccd_fields": [
                "false_accept=false",
                "verifier_accepted=false",
                "pass_at_1=0.0",
                "pass_at_k=0.0",
            ],
            "baseline_rate": baseline["false_accept_rate"],
            "dccd_rate": dccd["false_accept_rate"],
            "delta": comparison["false_accept_delta"],
            "interpretation": (
                "zero false accepts in schema-only DCCD is not progress because pass@1 and "
                "pass@k are both zero"
            ),
        },
        "pass_rate_regression": {
            "correlated_dccd_fields": [
                "mode=dccd_structured",
                "candidate_count=40",
                "schema_failure_rate=1.0",
                "verifier_acceptance_rate=0.0",
            ],
            "baseline_pass_at_1": baseline["pass_at_1"],
            "dccd_pass_at_1": dccd["pass_at_1"],
            "delta": comparison["pass_at_1_delta"],
            "interpretation": "hard structure erased useful draft intent instead of repairing it",
        },
    }


def _repair_manifest_schema() -> dict[str, Any]:
    return {
        "schema_version": "carnot.intent_preserving_repair_manifest.v1",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "draft_intent",
            "constrained_patch",
            "backtracking_steps",
            "execution_trace",
            "verifier_result",
            "schema_result",
            "syntax_result",
            "false_accept_audit",
            "acceptance_reason",
        ],
        "properties": {
            "draft_intent": {
                "type": "object",
                "required": ["raw_draft_ref", "intent_summary", "prompt_sha256"],
                "description": "The unconstrained model draft and its intended algorithm before masks are applied.",
            },
            "constrained_patch": {
                "type": "object",
                "required": ["patch_ref", "schema_backend", "diff_from_draft"],
                "description": "The constrained candidate plus the backend that shaped it.",
            },
            "backtracking_steps": {
                "type": "array",
                "min_items": 1,
                "items": {
                    "type": "object",
                    "required": [
                        "step_index",
                        "constraint_trigger",
                        "rollback_target",
                        "intent_preservation_check",
                    ],
                },
            },
            "execution_trace": {
                "type": "array",
                "min_items": 1,
                "items": {
                    "type": "object",
                    "required": ["command", "exit_code", "stdout_ref", "stderr_ref", "failing_assertions"],
                },
            },
            "verifier_result": {
                "type": "object",
                "required": ["threshold", "score", "accepted_by_verifier"],
            },
            "schema_result": {
                "type": "object",
                "required": ["schema_valid", "schema_errors"],
            },
            "syntax_result": {
                "type": "object",
                "required": ["ast_parse_ok", "parser_status", "static_checks"],
            },
            "false_accept_audit": {
                "type": "object",
                "required": ["passed_tests", "verifier_only_accept", "known_false_accept_pattern"],
            },
            "acceptance_reason": {
                "type": "string",
                "enum": [
                    "accepted_all_deterministic_gates",
                    "rejected_schema_regression",
                    "rejected_syntax_regression",
                    "rejected_false_accept_risk",
                    "rejected_intent_drift",
                    "rejected_missing_trace",
                ],
            },
        },
    }


def _baseline_conditions() -> list[dict[str, Any]]:
    return [
        {
            "condition_id": "baseline_no_taxonomy",
            "source": "exp2964",
            "role": "raw repair generation without taxonomy or DCCD structure",
        },
        {
            "condition_id": "schema_only_dccd",
            "source": "exp2964",
            "role": "diagnostic reproduction of the schema-collapse failure mode",
            "promotion_allowed": False,
        },
        {
            "condition_id": "intent_preserving_dccd",
            "source": "exp2977",
            "role": "draft-first repair with constrained patching and explicit intent checks",
        },
        {
            "condition_id": "trace_aware_repair",
            "source": "exp2977",
            "role": "intent-preserving repair with runtime traces before acceptance",
        },
    ]


def _required_model_specs() -> list[dict[str, Any]]:
    return [
        {
            "selection_rule": "call_cached_sota_pair_first",
            "minimum_headline_models": 1,
            "mandatory_hf_ids": list(MANDATORY_HEADLINE_MODEL_IDS),
            "required_for": "headline_results",
        },
        {
            "selection_rule": "legacy_small_models_are_cpu_smoke_only",
            "legacy_examples": ["Qwen/Qwen3.5-0.8B", "google/gemma-4-E4B-it"],
            "headline_allowed": False,
        },
    ]


def _acceptance_gates(max_false_accept_rate: float) -> dict[str, Any]:
    condition_ids = [condition["condition_id"] for condition in _baseline_conditions()]
    return {
        "minimum_tasks": {"metric": "n_tasks", "op": ">=", "value": DOWNSTREAM_MIN_TASKS},
        "condition_coverage": {
            "required_conditions": condition_ids,
            "op": "contains_all",
        },
        "headline_model_gate": {
            "metric": "models_used",
            "op": "contains_at_least_one",
            "value": list(MANDATORY_HEADLINE_MODEL_IDS),
            "requires_cached_sota_pair_attempt": True,
        },
        "pass_at_1_delta": {"metric": "pass_at_1_delta", "op": ">", "value": 0.0},
        "pass_at_k_delta": {"metric": "pass_at_k_delta", "op": ">=", "value": 0.0},
        "schema_failure_rate_delta": {
            "metric": "schema_failure_rate_delta",
            "op": "<=",
            "value": 0.0,
        },
        "syntax_failure_rate_delta": {
            "metric": "syntax_failure_rate_delta",
            "op": "<=",
            "value": 0.0,
        },
        "false_accept_delta": {"metric": "false_accept_delta", "op": "<=", "value": 0.0},
        "false_accept_rate": {
            "metric": "intent_preserving_false_accept_rate",
            "op": "<=",
            "value": max_false_accept_rate,
        },
        "runtime_trace_coverage": {
            "metric": "runtime_trace_coverage",
            "op": ">=",
            "value": TRACE_COVERAGE_FLOOR,
        },
        "per_candidate_acceptance": {
            "requires": [
                "draft_intent_present",
                "schema_result.schema_valid",
                "syntax_result.ast_parse_ok",
                "execution_trace_present",
                "available_tests_pass",
                "verifier_result.score>=threshold",
                "false_accept_audit.verifier_only_accept=false",
                "acceptance_reason=accepted_all_deterministic_gates",
            ]
        },
    }


def _schema_guard(comparison: dict[str, Any]) -> dict[str, Any]:
    dccd = comparison["exp2964_dccd_structured"]
    return {
        "guard_id": "schema_failure_rate_must_not_regress",
        "baseline_rate": comparison["exp2964_baseline"]["schema_failure_rate"],
        "blocked_prior_pattern": {
            "experiment_id": "exp2964",
            "mode": "dccd_structured",
            "observed_rate": dccd["schema_failure_rate"],
            "observed_delta": comparison["schema_failure_rate_delta"],
        },
        "pass_condition": "schema_failure_rate_delta <= 0 and every accepted repair has schema_valid=true",
    }


def _syntax_guard(comparison: dict[str, Any]) -> dict[str, Any]:
    dccd = comparison["exp2964_dccd_structured"]
    return {
        "guard_id": "syntax_failure_rate_must_not_regress",
        "baseline_rate": comparison["exp2964_baseline"]["syntax_failure_rate"],
        "blocked_prior_pattern": {
            "experiment_id": "exp2964",
            "mode": "dccd_structured",
            "observed_rate": dccd["syntax_failure_rate"],
            "observed_delta": comparison["syntax_failure_rate_delta"],
        },
        "pass_condition": "syntax_failure_rate_delta <= 0 and every accepted repair parses with ast.parse",
    }


def _false_accept_guard(sources: dict[str, dict[str, Any]], comparison: dict[str, Any]) -> dict[str, Any]:
    return {
        "guard_id": "false_accept_reduction_cannot_mask_zero_pass_rate",
        "max_false_accept_rate": sources["exp2953"]["expected_false_accept_rate_at_default"],
        "selected_default_threshold": sources["exp2953"]["selected_default_threshold"],
        "baseline_false_accept_rate": comparison["exp2964_baseline"]["false_accept_rate"],
        "blocked_prior_pattern": {
            "experiment_id": "exp2964",
            "mode": "dccd_structured",
            "observed_false_accept_rate": comparison["exp2964_dccd_structured"][
                "false_accept_rate"
            ],
            "observed_pass_at_1": comparison["exp2964_dccd_structured"]["pass_at_1"],
        },
        "verifier_only_accepts_count_as_pass": False,
        "pass_condition": (
            "false_accept_delta <= 0, intent_preserving_false_accept_rate <= threshold policy, "
            "and pass_at_1_delta > 0"
        ),
    }


def _blocked_artifact(
    *,
    honest_verdict: str,
    preconditions: list[dict[str, Any]],
    start: float,
    monotonic: ClockFunc,
) -> dict[str, Any]:
    return {
        "honest_verdict": honest_verdict,
        "intent_preserving_repair_protocol_ready": False,
        "trace_execution_plan_ready": False,
        "downstream_min_tasks": DOWNSTREAM_MIN_TASKS,
        "required_model_specs": _required_model_specs(),
        "mandatory_headline_model_ids": list(MANDATORY_HEADLINE_MODEL_IDS),
        "legacy_model_policy": LEGACY_MODEL_POLICY,
        "baseline_conditions": [],
        "repair_manifest_schema": {"schema_version": "blocked", "required": [], "properties": {}},
        "acceptance_gates": {},
        "schema_regression_guard": {},
        "syntax_regression_guard": {},
        "false_accept_guard": {},
        "prior_failure_addressed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _duration(monotonic, start),
        "preconditions_checked": preconditions,
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "run_date": RUN_DATE,
        "artifact": ARTIFACT_FILENAME,
    }


def build_artifact(
    *,
    repo_root: Path,
    monotonic: ClockFunc = time.monotonic,
) -> dict[str, Any]:
    start = monotonic()
    source_records, missing_sources = _source_preconditions(repo_root)
    research_record = _research_sweep_precondition(repo_root)
    preconditions = [*source_records, research_record]
    if missing_sources:
        return _blocked_artifact(
            honest_verdict="blocked_missing_upstream_repair_artifacts",
            preconditions=preconditions,
            start=start,
            monotonic=monotonic,
        )
    if not research_record["available"]:
        return _blocked_artifact(
            honest_verdict="blocked_missing_post_279_research_sweep",
            preconditions=preconditions,
            start=start,
            monotonic=monotonic,
        )

    sources = _load_sources(repo_root)
    comparison = _failure_comparison(sources)
    false_accept_guard = _false_accept_guard(sources, comparison)
    artifact = {
        "honest_verdict": (
            "complete: intent-preserving trace-aware DCCD protocol ready for exp2977; "
            "no pass-rate improvement claimed"
        ),
        "intent_preserving_repair_protocol_ready": True,
        "trace_execution_plan_ready": True,
        "downstream_min_tasks": DOWNSTREAM_MIN_TASKS,
        "required_model_specs": _required_model_specs(),
        "mandatory_headline_model_ids": list(MANDATORY_HEADLINE_MODEL_IDS),
        "legacy_model_policy": LEGACY_MODEL_POLICY,
        "baseline_conditions": _baseline_conditions(),
        "repair_manifest_schema": _repair_manifest_schema(),
        "acceptance_gates": _acceptance_gates(false_accept_guard["max_false_accept_rate"]),
        "schema_regression_guard": _schema_guard(comparison),
        "syntax_regression_guard": _syntax_guard(comparison),
        "false_accept_guard": false_accept_guard,
        "prior_failure_addressed": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _duration(monotonic, start),
        "upstream_failure_comparison": comparison,
        "dccd_failure_correlations": _failure_correlations(comparison),
        "downstream_evaluation_plan": {
            "n_tasks_min": DOWNSTREAM_MIN_TASKS,
            "conditions": _baseline_conditions(),
            "metrics": [
                "pass_at_1_delta",
                "pass_at_k_delta",
                "schema_failure_rate_delta",
                "syntax_failure_rate_delta",
                "false_accept_delta",
                "runtime_trace_coverage",
            ],
            "raw_evidence_required": [
                "raw_draft_ref",
                "constrained_patch_ref",
                "execution_trace_ref",
                "schema_errors",
                "syntax_errors",
                "failing_assertions",
            ],
        },
        "source_artifacts": source_records,
        "preconditions_checked": preconditions,
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "run_date": RUN_DATE,
        "artifact": ARTIFACT_FILENAME,
    }
    return artifact


def run_experiment(
    *,
    repo_root: Path,
    artifact_path: Path | None = None,
    monotonic: ClockFunc = time.monotonic,
) -> dict[str, Any]:
    destination = artifact_path or repo_root / "results" / ARTIFACT_FILENAME
    artifact = build_artifact(repo_root=repo_root, monotonic=monotonic)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
