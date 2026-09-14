"""Run the V642 live batch canary or fail closed before model work.

The current same-milestone fixture is a hard dependency. A readiness number
cannot override a blocked, disqualified, or quarantined producer. This module
therefore keeps the live schedule reusable but publishes no generated data when
the dependency terminal class is invalid.

Spec refs: REQ-VERIFY-7307 and SCENARIO-VERIFY-7307-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import shlex
import subprocess
import threading
import time
from typing import Any

import yaml

from carnot import experiment_7291_v641_reuse_fixture as reuse
from carnot import experiment_7306_v642_batch_fixture as fixture
from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.sota_models import current_model
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]
RUN_DATE = "20260914"
MILESTONE = "2026.09.642"
EXPERIMENT_ID = "exp7307-batch-canary"
SCHEMA = "carnot.exp7307.v642_batch_canary.v1"
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QUANTIZATION = "Q4_K_M"
MODEL_SPECS: list[JsonDict] = [{"hf_id": MODEL_ID, "quantization": QUANTIZATION}]
DEVELOPMENT_SEED = fixture.DEVELOPMENT_SEED
EVALUATION_SEED = fixture.EVALUATION_SEED
PROMPT_TEMPLATE = "exp7306_public_request_builder_v1"
ARMS = fixture.ARMS
PLANNED_SOURCE_VERSIONS = 2
PLANNED_UNITS = 8
PLANNED_CALLS = 16
PLANNED_OUTPUT_TOKENS = 7_680

UPSTREAM_PATH = Path("results/experiment_7306_v642_batch_fixture.json")
UPSTREAM_PUBLIC_PATH = Path("results/raw/experiment_7306_v642_batch_fixture/public_panel.json")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
RESEARCH_PROGRAM_PATH = Path("research-program.md")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
SOTA_MODEL_PATH = Path("python/carnot/inference/sota_models.py")
MODULE_PATH = Path("python/carnot/experiment_7307_v642_batch_canary.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7307_v642_batch_canary.py")
TEST_PATH = Path("tests/python/test_experiment_7307_v642_batch_canary.py")
RESULT_PATH = Path("results/experiment_7307_v642_batch_canary.json")
RAW_DIR = Path("results/raw/experiment_7307_v642_batch_canary")
CANDIDATE_PATH = RAW_DIR / "terminal_candidate.json"
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7307_v642_batch_canary.json")

KNOWN_UPSTREAM_FULL_SUITE_SHA256 = (
    "sha256:823b3369a44a241aa4f0c41e6d25cefa1397a3a5169f5eb1d5653097feba7a60"
)

ZERO_INVOCATION_COUNTS: JsonDict = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
}

FIELD_PRINCIPLES: JsonDict = {
    "schema": "Version the record while keeping ordinary top-level experiment_id and milestone.",
    "status": "Write the terminal result only after the work and checks; checkpoints remain separate.",
    "run_date": "Use 20260914 with actual UTC start/end and monotonic phase timing.",
    "preconditions_checked": "Hash real inputs and record actual availability and failed checks.",
    "MODEL_SPECS": "Actual current executable model identities; historical identities remain in sidecars.",
    "model_invoked": "True for any attempted model load or generation, even when output is unusable.",
    "invocation_counts": "Separate attempted, completed, failed, cancelled, and in-flight loads and generations.",
    "inference_substrate": "Describe actual computation with a recognized literal, not intended work.",
    "inference_substrate_class": "Full generation has a 60s floor; bounded generation 10s; load-only 2s. Never pad time.",
    "execution_venue": "Record actual host/device work; a CPU replay is not GPU or FPGA execution.",
    "duration_s": "Measure total elapsed and disjoint phase spans including failed work.",
    "random_seed": "Seal development and independent evaluation seeds before seeing outcomes.",
    "reproducibility_checksum": "Bind code, inputs, config, model when used, and raw evidence.",
    "source_artifact_hashes": "Authenticate producer identity, terminal class, and quarantine state.",
    "rows": "Record every comparative unit/arm with metrics, costs, errors, abstentions, and censoring.",
    "sample_size_budget": "Keep planned, attempted, complete, and censored counts with the frozen stopping rule.",
    "acceptance_gate_results": "Every check has expected, observed, passed, and a one-line principle explaining its purpose.",
    "gate_check_summary": "Every blocked_* names upstream/check, exact field, observed value, and expected value.",
    "verifier_is_oracle": "Shared evaluator authority permits circular_positive only, not positive scientific value.",
    "honest_verdict": "Completed findings start complete_ or complete:; external failure starts blocked_. State the finding.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. Only unfinished own work is partial; unchanged external failure is blocked.",
    "validation_receipts": "Record exact commands, scope, exit codes, elapsed time, and log hashes; retain failures.",
    "batch_canary_ready_score": "One for real bounded transport and replay, not source correctness.",
    "per_call_rows": "Native request/output, IDs, budgets, and exact attempts make the canary auditable.",
    "parser_replay_receipt": "Independent bytes-to-decisions reconstruction must agree.",
    "runtime_model_identity": "The cached model and actual backend must match the directive.",
}

REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        *FIELD_PRINCIPLES,
        "experiment_id",
        "milestone",
        "field_principles",
        "timestamps",
        "phase_spans",
        "inference_mode",
        "execution_host",
        "repository_health",
        "call_budget_contract",
        "sealed_development_manifest",
        "gpu_receipts",
        "methodology_note",
    }
)

REQUIRED_VALIDATION_NAMES = (
    "focused_pytest",
    "affected_suites",
    "full_python_suite",
    "scoped_coverage",
    "scoped_coverage_report",
    "ruff_check",
    "ruff_format",
    "changed_module_mypy",
    "scoped_spec_coverage",
    "e2e_terminal_candidate",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


def _utc_now() -> str:
    """Record a real UTC boundary without treating wall time as evidence."""

    return datetime.now(UTC).isoformat()


def _canonical_json(value: Any) -> str:
    """Serialize evidence once so hashes and comparisons use identical bytes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    """Label content hashes so readers cannot confuse them with raw identifiers."""

    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _sha256_file(path: Path) -> str:
    """Hash the bytes that the experiment actually authenticated."""

    return _sha256_bytes(path.read_bytes())


def _progress(phase: int, event: str, **details: Any) -> None:
    """Flush a machine-readable line at each phase or slow-call boundary."""

    print(
        f"[exp7307] {_canonical_json({'phase': phase, 'event': event, **details})}",
        flush=True,
    )


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind durable evidence while excluding local timing observations."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "timestamps", "phase_spans", "reproducibility_checksum"}
    }
    return _sha256_bytes(_canonical_json(stable).encode("utf-8"))


def gate_row(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Keep each exact comparison and explain why it controls execution."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
        "principle": principle,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Project the first failed comparison without changing either value."""

    failure = next((row for row in checks if row.get("passed") is not True), None)
    if failure is None:
        return {
            "failed_check": None,
            "upstream": None,
            "field": None,
            "expected_value": None,
            "observed_value": None,
        }
    return {
        "failed_check": failure.get("check"),
        "upstream": failure.get("upstream"),
        "field": failure.get("field"),
        "expected_value": failure.get("expected_value"),
        "observed_value": failure.get("observed_value"),
    }


def call_budget_contract() -> JsonDict:
    """Freeze the equal arm-version budget inherited from the CPU fixture."""

    return {
        "source_versions": PLANNED_SOURCE_VERSIONS,
        "claims_per_source_version": 4,
        "serial_versioned_verifier": {"calls": 10, "allocated_output_tokens": 2_560},
        "batched_versioned_verifier": {"calls": 4, "allocated_output_tokens": 2_560},
        "batched_warm_prefix_direct": {"calls": 2, "allocated_output_tokens": 2_560},
        "per_arm_source_version_output_tokens": 1_280,
        "total_generation_calls_maximum": PLANNED_CALLS,
        "total_allocated_output_tokens": PLANNED_OUTPUT_TOKENS,
        "retry_calls": 0,
        "repair_calls": 0,
    }


def build_schedule(public_panel: Mapping[str, Any]) -> list[JsonDict]:
    """Select one development group and retain its two four-claim revisions."""

    groups = list(public_panel.get("development_groups") or [])
    if not groups:
        raise ValueError("development_group_denominator")
    schedule = fixture.build_call_schedule([groups[0]], require_full_denominator=False)
    for row in schedule:
        row["prompt_template"] = PROMPT_TEMPLATE
        row["decoding_parameters"] = {
            "temperature": 0.0,
            "seed": DEVELOPMENT_SEED,
            "max_output_tokens": row["allocated_output_tokens"],
        }
    return schedule


def schedule_errors(schedule: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject changed call counts, budgets, identifiers, or decode settings."""

    errors: list[str] = []
    if len(schedule) != PLANNED_CALLS:
        errors.append("call_count")
    call_ids = [row.get("call_id") for row in schedule]
    if len(set(call_ids)) != len(call_ids):
        errors.append("duplicate_call_id")
    expected_counts = {
        "serial_versioned_verifier": 10,
        "batched_versioned_verifier": 4,
        "batched_warm_prefix_direct": 2,
    }
    if Counter(row.get("arm") for row in schedule) != expected_counts:
        errors.append("arm_call_counts")
    if sum(int(row.get("allocated_output_tokens", 0) or 0) for row in schedule) != 7_680:
        errors.append("allocated_output_tokens")
    expected_decode = {
        "temperature": 0.0,
        "seed": DEVELOPMENT_SEED,
    }
    if any(
        dict(row.get("decoding_parameters") or {}).get(key) != value
        for row in schedule
        for key, value in expected_decode.items()
    ):
        errors.append("decoding_parameters")
    if any(row.get("prompt_template") != PROMPT_TEMPLATE for row in schedule):
        errors.append("prompt_template")
    for arm in ARMS:
        ids = {
            str(claim_id)
            for row in schedule
            if row.get("arm") == arm
            for claim_id in row.get("claim_ids") or []
        }
        if len(ids) != PLANNED_UNITS:
            errors.append(f"identifier_accounting:{arm}")
    return list(dict.fromkeys(errors))


def dependency_gate_rows(
    upstream: Mapping[str, Any],
    public_panel: Mapping[str, Any],
    exclusions: Any,
) -> list[JsonDict]:
    """Reject a failed producer even when its numeric readiness field is one."""

    quarantined = bool(upstream.get("quarantined") or upstream.get("flagged_adversarial"))
    terminal_class = upstream.get("verdict_class")
    model_directive = current_model().get("hf_id")
    excluded = reuse._manifest_lists_experiment(exclusions, EXPERIMENT_ID) or (
        reuse._manifest_lists_experiment(exclusions, "exp7306-batch-fixture")
    )
    public_text = _canonical_json(public_panel)
    return [
        gate_row(
            "upstream_status",
            "exp7306-batch-fixture",
            "status",
            "complete",
            upstream.get("status"),
            upstream.get("status") == "complete",
            "A same-milestone input must be terminal before the canary can consume it.",
        ),
        gate_row(
            "batch_fixture_ready",
            "exp7306-batch-fixture",
            "batch_fixture_ready_score",
            1,
            upstream.get("batch_fixture_ready_score"),
            upstream.get("batch_fixture_ready_score") == 1,
            "The CPU fixture must seal the request and parser contract first.",
        ),
        gate_row(
            "upstream_quarantine",
            "exp7306-batch-fixture",
            "quarantined_or_flagged_adversarial",
            False,
            quarantined,
            not quarantined,
            "Quarantined evidence cannot authorize new model work.",
        ),
        gate_row(
            "upstream_terminal_class",
            "exp7306-batch-fixture",
            "verdict_class",
            "not blocked or disqualified",
            terminal_class,
            terminal_class not in {"blocked", "disqualified"},
            "A numeric readiness value cannot override a failed terminal class.",
        ),
        gate_row(
            "upstream_producer_identity",
            "exp7306-batch-fixture",
            "experiment_id",
            "exp7306-batch-fixture",
            upstream.get("experiment_id"),
            upstream.get("experiment_id") == "exp7306-batch-fixture",
            "The dependency must come from the named producer.",
        ),
        gate_row(
            "public_panel_schema",
            UPSTREAM_PUBLIC_PATH.as_posix(),
            "schema",
            "carnot.batch_fixture.public_panel.v1",
            public_panel.get("schema"),
            public_panel.get("schema") == "carnot.batch_fixture.public_panel.v1",
            "The live requests must use the sealed public panel format.",
        ),
        gate_row(
            "development_split_available",
            UPSTREAM_PUBLIC_PATH.as_posix(),
            "development_group_count",
            8,
            len(public_panel.get("development_groups") or []),
            len(public_panel.get("development_groups") or []) == 8,
            "The canary may use only the sealed development split.",
        ),
        gate_row(
            "public_authority_isolation",
            UPSTREAM_PUBLIC_PATH.as_posix(),
            "private_label_fields_present",
            False,
            any(
                field in public_text
                for field in ("expected_decision", "construction_label", "gold_span")
            ),
            not any(
                field in public_text
                for field in ("expected_decision", "construction_label", "gold_span")
            ),
            "Evaluator labels cannot enter model prompts or transport checks.",
        ),
        gate_row(
            "retirement_and_exclusion",
            EXCLUSION_PATH.as_posix(),
            "experiment_or_dependency_listed",
            False,
            excluded,
            not excluded,
            "Retired or excluded work cannot silently resume.",
        ),
        gate_row(
            "current_model_directive",
            SOTA_MODEL_PATH.as_posix(),
            "hf_id",
            MODEL_ID,
            model_directive,
            model_directive == MODEL_ID,
            "The canary cannot silently select another cached model.",
        ),
    ]


def authenticate_inputs(root: Path) -> tuple[list[JsonDict], JsonDict]:
    """Hash and parse every external input before any cache or CUDA check."""

    paths = {
        "upstream": root / UPSTREAM_PATH,
        "public_panel": root / UPSTREAM_PUBLIC_PATH,
        "exclusion_manifest": root / EXCLUSION_PATH,
        "research_program": root / RESEARCH_PROGRAM_PATH,
        "verification_spec": root / SPEC_PATH,
        "sota_model_registry": root / SOTA_MODEL_PATH,
        "module": root / MODULE_PATH,
        "entrypoint": root / WRAPPER_PATH,
        "focused_tests": root / TEST_PATH,
    }
    checks: list[JsonDict] = []
    for name, path in paths.items():
        available = path.is_file()
        checks.append(
            {
                **gate_row(
                    "required_path_available",
                    name,
                    "path",
                    "file",
                    "file" if available else "missing",
                    available,
                    "Every named input must exist before model work starts.",
                ),
                "path": str(path),
                "sha256": _sha256_file(path) if available else None,
            }
        )
    results_root = (root / "results").resolve()
    for name, relative in (
        ("terminal_result", RESULT_PATH),
        ("raw_evidence", RAW_DIR),
        ("checkpoint", CHECKPOINT_PATH),
    ):
        target = (root / relative).resolve()
        inside = target.is_relative_to(results_root)
        checks.append(
            gate_row(
                "declared_output_path",
                name,
                "path_scope",
                "inside_results",
                "inside_results" if inside else str(target),
                inside,
                "Task outputs must stay inside the declared results tree.",
            )
        )
    if any(row["passed"] is not True for row in checks):
        return checks, {}
    try:
        upstream = json.loads(paths["upstream"].read_text(encoding="utf-8"))
        public_panel = json.loads(paths["public_panel"].read_text(encoding="utf-8"))
        exclusions = yaml.safe_load(paths["exclusion_manifest"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        checks.append(
            gate_row(
                "input_serialization",
                "declared_inputs",
                "json_or_yaml",
                "valid",
                f"{type(exc).__name__}:{exc}",
                False,
                "Malformed evidence cannot be repaired during authentication.",
            )
        )
        return checks, {}
    upstream_errors = fixture.validate_artifact(upstream)
    checks.append(
        gate_row(
            "upstream_artifact_validation",
            "exp7306-batch-fixture",
            "schema_and_checksum",
            [],
            upstream_errors,
            not upstream_errors,
            "The exact producer validator must accept the stored artifact bytes.",
        )
    )
    checks.extend(dependency_gate_rows(upstream, public_panel, exclusions))
    return checks, {
        "upstream": upstream,
        "public_panel": public_panel,
        "exclusions": exclusions,
        "private_labels_opened": False,
    }


def classify_inference(counts: Mapping[str, Any]) -> JsonDict:
    """Derive model use only from recorded load and generation attempts."""

    loads = int(counts.get("model_loads_attempted", 0) or 0)
    generations = int(counts.get("generation_calls_attempted", 0) or 0)
    if generations:
        return {
            "model_invoked": True,
            "inference_substrate": "live_llm_inference",
            "inference_substrate_class": "model_bounded_generation",
            "inference_mode": "live_gpu",
        }
    if loads:
        return {
            "model_invoked": True,
            "inference_substrate": "model_load_no_generation",
            "inference_substrate_class": "model_load_no_generation",
            "inference_mode": "not_invoked",
        }
    return {
        "model_invoked": False,
        "inference_substrate": "blocked_before_qualifying_computation",
        "inference_substrate_class": "blocked_no_run",
        "inference_mode": "not_invoked",
    }


def source_artifact_hashes(root: Path, checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Record producer class and quarantine state beside every authenticated hash."""

    upstream = json.loads((root / UPSTREAM_PATH).read_text(encoding="utf-8"))
    values: JsonDict = {}
    for row in checks:
        path_text = row.get("path")
        if not path_text or not row.get("sha256"):
            continue
        path = Path(str(path_text))
        key = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
        is_upstream = path == root / UPSTREAM_PATH
        values[key] = {
            "sha256": row["sha256"],
            "producer_identity": upstream.get("experiment_id") if is_upstream else row["upstream"],
            "terminal_class": upstream.get("verdict_class") if is_upstream else "source_file",
            "quarantined": bool(upstream.get("quarantined") or upstream.get("flagged_adversarial"))
            if is_upstream
            else False,
        }
    return values


def base_artifact(run_date: str, *, started_at_utc: str | None = None) -> JsonDict:
    """Create a schema-complete checkpoint before fallible authentication."""

    classification = classify_inference(ZERO_INVOCATION_COUNTS)
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "running",
        "run_date": run_date,
        "timestamps": {"started_at_utc": started_at_utc or _utc_now(), "completed_at_utc": None},
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [],
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        **classification,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "execution_venue": "host",
        "execution_host": platform.node() or "unknown",
        "duration_s": 0.0,
        "random_seed": {
            "development": DEVELOPMENT_SEED,
            "independent_evaluation": EVALUATION_SEED,
            "evaluation_observed": False,
            "sealed_before_outcomes": True,
        },
        "reproducibility_checksum": "pending",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_source_versions": PLANNED_SOURCE_VERSIONS,
            "planned_units_per_arm": PLANNED_UNITS,
            "planned_comparative_rows": PLANNED_UNITS * len(ARMS),
            "planned_generation_calls": PLANNED_CALLS,
            "planned_output_tokens": PLANNED_OUTPUT_TOKENS,
            "attempted_generation_calls": 0,
            "completed_generation_calls": 0,
            "failed_generation_calls": 0,
            "cancelled_generation_calls": 0,
            "censored_generation_calls": PLANNED_CALLS,
            "complete_comparative_rows": 0,
            "censored_comparative_rows": PLANNED_UNITS * len(ARMS),
            "stopping_rule": "stop on any failed external dependency; otherwise attempt the fixed sixteen calls once without retry, repair, or tuning",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": gate_summary([]),
        "verifier_is_oracle": True,
        "honest_verdict": "partial_exp7307_unfinished_checkpoint_only",
        "verdict_class": "partial",
        "validation_receipts": [],
        "batch_canary_ready_score": 0,
        "per_call_rows": [],
        "parser_replay_receipt": {
            "status": "not_attempted_external_block",
            "native_rows_attempted": 0,
            "native_rows_replayed": 0,
            "decision_discrepancies": [],
            "all_identifiers_accounted": False,
        },
        "runtime_model_identity": {
            "directive_hf_id": MODEL_ID,
            "quantization": QUANTIZATION,
            "resolution_status": "not_resolved_external_dependency_block",
            "actual_gguf_path": None,
            "gguf_sha256": None,
            "embedded_tokenizer": "not_checked_external_dependency_block",
            "backend": "not_loaded",
            "cuda_support": "not_checked_external_dependency_block",
            "cache_capability": "not_checked_external_dependency_block",
        },
        "phase_spans": [],
        "repository_health": {
            "all_required_validations_passed": False,
            "failed_validation_names": [],
            "preserved_upstream_failure": "full_python_suite",
            "preserved_upstream_full_suite_log_sha256": KNOWN_UPSTREAM_FULL_SUITE_SHA256,
        },
        "call_budget_contract": call_budget_contract(),
        "sealed_development_manifest": {
            "source_versions": PLANNED_SOURCE_VERSIONS,
            "claim_identifiers_per_arm": PLANNED_UNITS,
            "schedule_status": "not_built_external_dependency_block",
            "evaluation_rows_used": 0,
        },
        "gpu_receipts": {
            "reservation_status": "not_attempted_external_dependency_block",
            "cuda_work_observed": False,
            "other_users_interrupted": False,
        },
        "methodology_note": (
            "This is a bounded transport canary with no source-verification value claim. "
            "The disqualified fixture stopped work before model resolution, loading, generation, or replay."
        ),
    }


def _acceptance_rows(checks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Expose prerequisite comparisons in the ordinary acceptance-gate shape."""

    return [
        {
            "criterion": row.get("check"),
            "expected": deepcopy(row.get("expected_value")),
            "observed": deepcopy(row.get("observed_value")),
            "passed": row.get("passed") is True,
            "principle": row.get("principle"),
        }
        for row in checks
    ]


def finalize_blocked_artifact(
    artifact: JsonDict,
    checks: Sequence[Mapping[str, Any]],
    *,
    validation_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    completed_at_utc: str | None = None,
) -> JsonDict:
    """Finish an external failure without fabricating model or parser work."""

    summary = gate_summary(checks)
    artifact["status"] = "blocked"
    artifact["verdict_class"] = "blocked"
    artifact["preconditions_checked"] = [deepcopy(dict(row)) for row in checks]
    artifact["acceptance_gate_results"] = _acceptance_rows(checks)
    artifact["gate_check_summary"] = summary
    artifact["batch_canary_ready_score"] = 0
    artifact.update(classify_inference(artifact["invocation_counts"]))
    owner = "exp7306" if str(summary.get("upstream", "")).startswith("exp7306") else "exp7307"
    artifact["honest_verdict"] = f"blocked_{owner}_{summary.get('failed_check')}"
    artifact["validation_receipts"] = [deepcopy(dict(row)) for row in validation_receipts]
    failed_validations = [
        str(row.get("name")) for row in validation_receipts if row.get("passed") is not True
    ]
    artifact["repository_health"].update(
        {
            "all_required_validations_passed": bool(validation_receipts) and not failed_validations,
            "failed_validation_names": failed_validations,
        }
    )
    artifact["duration_s"] = duration_s
    artifact["timestamps"]["completed_at_utc"] = completed_at_utc or _utc_now()
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: object) -> list[str]:
    """Cold-check the blocked record and its exact external failure boundary."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping"]
    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in value]
    expected = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "field_principles": FIELD_PRINCIPLES,
        "MODEL_SPECS": MODEL_SPECS,
        "execution_venue": "host",
        "verifier_is_oracle": True,
        "status": "blocked",
        "verdict_class": "blocked",
        "batch_canary_ready_score": 0,
    }
    errors.extend(
        field for field, expected_value in expected.items() if value.get(field) != expected_value
    )
    counts = value.get("invocation_counts")
    if (
        not isinstance(counts, Mapping)
        or set(counts) != set(ZERO_INVOCATION_COUNTS)
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in counts.values()
        )
    ):
        errors.append("invocation_counts")
    elif any(
        value.get(field) != observed for field, observed in classify_inference(counts).items()
    ):
        errors.append("inference_classification")
    checks = value.get("preconditions_checked")
    summary = value.get("gate_check_summary")
    if (
        not isinstance(checks, list)
        or not isinstance(summary, Mapping)
        or summary != gate_summary(checks)
    ):
        errors.append("gate_check_summary")
    elif not summary.get("failed_check"):
        errors.append("blocked_without_failure")
    if not str(value.get("honest_verdict", "")).startswith("blocked_"):
        errors.append("honest_verdict")
    if value.get("rows") != [] or value.get("per_call_rows") != []:
        errors.append("blocked_rows")
    replay = value.get("parser_replay_receipt")
    if not isinstance(replay, Mapping) or replay.get("status") != "not_attempted_external_block":
        errors.append("parser_replay_receipt")
    duration = value.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        errors.append("duration_s")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum")
    return list(dict.fromkeys(errors))


def validation_commands(root: Path, candidate: Path) -> list[tuple[str, list[str]]]:
    """Return the exact scoped, full-suite, coverage, E2E, and artifact checks."""

    python = str(root / ".venv/bin/python")
    pytest = str(root / ".venv/bin/pytest")
    test = TEST_PATH.as_posix()
    changed = [MODULE_PATH.as_posix(), WRAPPER_PATH.as_posix(), test]
    common = ["-o", "addopts=", "-n", "0", "--no-cov"]
    coverage_file = "/tmp/.coverage-exp7307-v642"
    return [
        (
            "focused_pytest",
            [pytest, *common, "--basetemp=/tmp/exp7307-focused", test, "-q"],
        ),
        (
            "affected_suites",
            [
                pytest,
                *common,
                "--basetemp=/tmp/exp7307-affected",
                test,
                "tests/python/test_experiment_7306_v642_batch_fixture.py",
                "tests/python/test_experiment_7292_v641_reuse_canary.py",
                "-q",
            ],
        ),
        ("full_python_suite", [pytest, "tests/python", "-q"]),
        (
            "scoped_coverage",
            [
                python,
                "-u",
                "-m",
                "coverage",
                "run",
                f"--data-file={coverage_file}",
                f"--include=*/{MODULE_PATH.name}",
                "-m",
                "pytest",
                *common,
                "--basetemp=/tmp/exp7307-coverage",
                test,
                "-q",
            ],
        ),
        (
            "scoped_coverage_report",
            [
                python,
                "-u",
                "-m",
                "coverage",
                "report",
                f"--data-file={coverage_file}",
                f"--include=*/{MODULE_PATH.name}",
                "--show-missing",
                "--fail-under=100",
            ],
        ),
        ("ruff_check", [python, "-u", "-m", "ruff", "check", *changed]),
        ("ruff_format", [python, "-u", "-m", "ruff", "format", "--check", *changed]),
        ("changed_module_mypy", [python, "-u", "-m", "mypy", MODULE_PATH.as_posix()]),
        ("scoped_spec_coverage", [python, "-u", "scripts/check_spec_coverage.py", test]),
        (
            "e2e_terminal_candidate",
            [
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--check-artifact",
                str(candidate),
            ],
        ),
        (
            "adversarial_verify",
            [python, "-u", "scripts/adversarial_verify.py", str(candidate)],
        ),
        (
            "verdict_row_consistency_strict",
            [
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ],
        ),
    ]


def _run_command(
    root: Path, name: str, command: Sequence[str], log_path: Path
) -> JsonDict:  # pragma: no cover - subprocess integration is exercised by the entrypoint.
    """Stream child output and report truthful heartbeats during quiet calls."""

    _progress(5, "subprocess_before", operation=name, command=shlex.join(command))
    started = time.monotonic()
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = f"{root / 'python'}:{root}"
    process = subprocess.Popen(  # noqa: S603 - commands are fixed repository-local argv.
        list(command),
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    messages: queue.Queue[str | None] = queue.Queue()

    def _reader() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            messages.put(line)
        messages.put(None)

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()
    lines: list[str] = []
    while True:
        try:
            line = messages.get(timeout=60)
        except queue.Empty:
            _progress(
                5,
                "subprocess_heartbeat",
                operation=name,
                elapsed_s=round(time.monotonic() - started, 3),
                outstanding_call=True,
            )
            continue
        if line is None:
            break
        lines.append(line)
        print(f"[exp7307:{name}] {line.rstrip()}", flush=True)
    exit_code = process.wait()
    reader.join(timeout=1)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("".join(lines), encoding="utf-8")
    duration = time.monotonic() - started
    _progress(5, "subprocess_after", operation=name, exit_code=exit_code, elapsed_s=duration)
    return {
        "name": name,
        "command": shlex.join(command),
        "scope": {
            "focused_pytest": "new behavior",
            "affected_suites": "new and explicitly affected suites",
            "full_python_suite": "repository Python suite",
            "scoped_coverage": "new module lines",
            "scoped_coverage_report": "100 percent new module line coverage",
            "ruff_check": "changed Python files",
            "ruff_format": "changed Python files",
            "changed_module_mypy": "new module",
            "scoped_spec_coverage": "exact new tests",
            "e2e_terminal_candidate": "entrypoint cold validation",
            "adversarial_verify": "terminal candidate",
            "verdict_row_consistency_strict": "terminal candidate strict row consistency",
        }[name],
        "exit_code": exit_code,
        "passed": exit_code == 0,
        "duration_s": duration,
        "log_path": log_path.relative_to(root).as_posix(),
        "log_sha256": _sha256_file(log_path),
    }


def _run_validations(
    root: Path, candidate: Path
) -> list[JsonDict]:  # pragma: no cover - required local command orchestration.
    """Run each fixed command once and preserve failures without repair."""

    validation_dir = root / RAW_DIR / "validation"
    receipts = []
    commands = validation_commands(root, candidate)
    for index, (name, command) in enumerate(commands, start=1):
        _progress(5, "validation_unit_start", completed=index - 1, total=len(commands), name=name)
        receipts.append(_run_command(root, name, command, validation_dir / f"{name}.log"))
        _progress(5, "validation_unit_end", completed=index, total=len(commands), name=name)
    return receipts


def _pending_receipts(root: Path, candidate: Path) -> list[JsonDict]:
    """Mark validation as pending without making a success-shaped checkpoint."""

    return [
        {
            "name": name,
            "command": shlex.join(command),
            "scope": "pending",
            "exit_code": None,
            "passed": False,
            "duration_s": 0.0,
            "log_path": f"{RAW_DIR.as_posix()}/validation/{name}.log",
            "log_sha256": "pending",
        }
        for name, command in validation_commands(root, candidate)
    ]


def run_experiment(
    root: Path | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:  # pragma: no cover - the unbuffered entrypoint owns file publication.
    """Authenticate, validate the honest block, then atomically publish it."""

    repository = root or find_repo_root(start=__file__)
    result_path = repository / RESULT_PATH
    candidate_path = repository / CANDIDATE_PATH
    checkpoint_path = repository / CHECKPOINT_PATH
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        errors = validate_artifact(existing)
        if errors:
            raise ValueError(f"existing Exp7307 artifact is invalid: {errors}")
        _progress(0, "stable_terminal_exists", path=str(result_path))
        return dict(existing)
    os.environ["CARNOT_FORCE_LIVE"] = "1"
    started = time.monotonic()
    artifact = base_artifact(run_date)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(checkpoint_path, artifact, allow_override=True, sort_keys=True)
    _progress(0, "phase_start", operation="authenticate_inputs_and_output_paths")
    phase = time.monotonic()
    checks, _inputs = authenticate_inputs(repository)
    artifact["phase_spans"] = [
        {"phase": "authenticate_inputs", "duration_s": time.monotonic() - phase}
    ]
    artifact["source_artifact_hashes"] = source_artifact_hashes(repository, checks)
    failures = [row for row in checks if row.get("passed") is not True]
    _progress(0, "phase_end", failed_checks=len(failures))
    if not failures:
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        atomic_write_json(checkpoint_path, artifact, allow_override=True, sort_keys=True)
        raise RuntimeError(
            "authenticated fixture unexpectedly permits live work; no placeholder emitted"
        )

    _progress(1, "phase_start", operation="assemble_external_block_candidate")
    pending = _pending_receipts(repository, candidate_path)
    candidate = finalize_blocked_artifact(
        artifact,
        checks,
        validation_receipts=pending,
        duration_s=time.monotonic() - started,
    )
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(candidate_path, candidate, allow_override=True, sort_keys=True)
    atomic_write_json(checkpoint_path, candidate, allow_override=True, sort_keys=True)
    _progress(1, "phase_end", gate_check_summary=candidate["gate_check_summary"])

    _progress(5, "phase_start", operation="required_validation_commands")
    validation_phase = time.monotonic()
    receipts = _run_validations(repository, candidate_path)
    artifact["phase_spans"].append(
        {"phase": "required_validations", "duration_s": time.monotonic() - validation_phase}
    )
    _progress(
        5,
        "phase_end",
        passed=sum(row["passed"] is True for row in receipts),
        total=len(receipts),
    )
    terminal = finalize_blocked_artifact(
        artifact,
        checks,
        validation_receipts=receipts,
        duration_s=time.monotonic() - started,
    )
    errors = validate_artifact(terminal)
    if errors:
        atomic_write_json(checkpoint_path, terminal, allow_override=True, sort_keys=True)
        raise ValueError(f"invalid Exp7307 terminal artifact: {errors}")
    _progress(6, "write_before", path=str(result_path))
    atomic_write_json(result_path, terminal, allow_override=False, sort_keys=True)
    _progress(6, "write_after", path=str(result_path))
    return terminal


def _date_argument(value: str) -> str:
    """Accept only the date fixed by the V642 execution contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Run the canary or cold-check one task-owned terminal candidate."""

    print("[exp7307] startup", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_date_argument)
    parser.add_argument("--check-artifact", type=Path)
    args = parser.parse_args(argv)
    if args.check_artifact is not None:
        _progress(0, "phase_start", operation="cold_validate_terminal_candidate")
        value = json.loads(args.check_artifact.read_text(encoding="utf-8"))
        errors = validate_artifact(value)
        _progress(0, "phase_end", errors=errors)
        return int(bool(errors))
    run_experiment(run_date=args.date)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
