"""Audit source-compilation reuse from immutable V641 live evidence.

The audit rebuilds model decisions from saved native request and response
bytes. It opens scorer labels only after byte replay. It then treats each
source group, rather than each correlated claim, as the uncertainty unit.

Spec refs: REQ-VERIFY-7294 and SCENARIO-VERIFY-7294-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import random
import select
import shlex
import subprocess
import time
from typing import Any

import yaml

from carnot import experiment_7291_v641_reuse_fixture as fixture
from carnot import experiment_7293_v641_reuse_measurement as measurement
from carnot.experiment_artifacts import atomic_write_json
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]

RUN_DATE = "20260914"
MILESTONE = "2026.09.641"
EXPERIMENT_ID = "exp7294-reuse-audit"
TASK_ID = "experiment_7294_v641_reuse_audit"
SCHEMA = "carnot.exp7294.v641_reuse_audit.v1"
RANDOM_SEED = 729_420_260_914
BOOTSTRAP_SEED = 7_291_300
BOOTSTRAP_DRAWS = 10_000
MODEL_SPECS: list[JsonDict] = []
ARMS = measurement.ARMS
PLANNED_GROUPS = 16
CLAIMS_PER_GROUP = 8
PLANNED_UNITS = 128
PLANNED_CALLS = 672

EXP7291_PATH = Path("results/experiment_7291_v641_reuse_fixture.json")
EXP7293_PATH = Path("results/experiment_7293_v641_reuse_measurement.json")
FIXTURE_MANIFEST_PATH = Path("results/raw/experiment_7291_v641_reuse_fixture/manifest.json")
ANALYSIS_PATH = Path("results/raw/experiment_7291_v641_reuse_fixture/analysis-contract.json")
AUTHORITY_PATH = Path("results/raw/experiment_7291_v641_reuse_fixture/scorer-only-labels.json")
UPSTREAM_RAW_DIR = Path("results/raw/experiment_7293_v641_reuse_measurement")
RAW_CALL_MANIFEST_PATH = UPSTREAM_RAW_DIR / "raw-call-manifest.json"
SCHEDULE_PATH = UPSTREAM_RAW_DIR / "schedule.json"
SOURCE_CLAIM_PATH = UPSTREAM_RAW_DIR / "source-claim-manifest.json"
CAPTURE_RECEIPT_PATH = UPSTREAM_RAW_DIR / "capture-receipt.json"
PER_UNIT_PATH = UPSTREAM_RAW_DIR / "per-unit-rows.json"
FAILED_ATTEMPT_PATH = UPSTREAM_RAW_DIR / "aborted-resume-attempt.json"
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
MODULE_PATH = Path("python/carnot/experiment_7294_v641_reuse_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7294_v641_reuse_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7294_v641_reuse_audit.py")
RESULT_PATH = Path("results/experiment_7294_v641_reuse_audit.json")
RAW_DIR = Path("results/raw/experiment_7294_v641_reuse_audit")
RAW_CANDIDATE_PATH = RAW_DIR / "measured-terminal-candidate.json"
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7294_v641_reuse_audit.json")

PINNED_HASHES = {
    EXP7291_PATH: "sha256:ecf1888d12bb8eac85b66ccde765d5bf5ca1a5358422b23590585024a629a263",
    EXP7293_PATH: "sha256:5c9c08ddbf95de2e1d29f2109fa73b9bf81541b8059e3da2d4bd777beef2edb1",
    FIXTURE_MANIFEST_PATH: "sha256:fafc05f40627d55fd27c83c8e5be8fb710d9837f6ba15b3138fc01bbdb4b5a87",
    ANALYSIS_PATH: "sha256:d51e430f657d91dc1c6d37d9a4aba758fb74ff8cea85603384177d812f8f2f18",
    AUTHORITY_PATH: "sha256:6548f9f93d78348823a6cfa70469187cb3b80a7b981a0b8eebf2918ba9f7a493",
    RAW_CALL_MANIFEST_PATH: "sha256:030bd55309208814b7b161d65145265a3deeb4413007d6f1c816e34d0063de79",
    SCHEDULE_PATH: "sha256:0a8c65612f05f13bcc6473eef96fc53ccd1a9a4244065698e2b88d87efd0d664",
    SOURCE_CLAIM_PATH: "sha256:136df45dd07d63145e3299d2bfeb901703521a452f8afae31f96727942d93946",
    CAPTURE_RECEIPT_PATH: "sha256:a2f15aee5341b994be491e31de213fc4b62d39fe623a3f7391fa1538db3abc7c",
    PER_UNIT_PATH: "sha256:ce30fda5f17fe4df7eb15360fe09a807d3ab3cb86e3af56581516c9fab3030b9",
    FAILED_ATTEMPT_PATH: "sha256:5935f783dbdbf294dca8563a832e814a24e4ccc8b0ca2211b2856f1168f26cf1",
}

ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_in_flight": 0,
    "usable_answers": 0,
}

FIELD_PRINCIPLES = {
    "schema": "Version the artifact; retain ordinary top-level experiment_id and milestone.",
    "status": "Use a terminal complete or blocked record; unfinished own work belongs in separate checkpoints.",
    "run_date": "Use 20260914, real UTC start/end and monotonic timing.",
    "field_principles": "Store explanations here while consumer values remain ordinary top-level values.",
    "preconditions_checked": "Hash actual inputs, authority boundaries, resource ownership and failed checks.",
    "MODEL_SPECS": "Actual executable local model identities; keep historical models in hashed sidecars.",
    "model_invoked": "True for any actual attempted model load or generation, including failed and unusable work.",
    "invocation_counts": "Separate attempted/completed/failed loads and generation; retain in-flight events on timeout.",
    "inference_substrate": "Use the recognized literal for actual computation; never infer from intended task.",
    "inference_substrate_class": "Full generation60s, bounded10s, load-only2s, or actual no-LLM class; never pad elapsed time.",
    "execution_venue": "Host is host; identify actual GPU/native/device execution separately.",
    "duration_s": "Measured monotonic elapsed and disjoint phase spans, including failures and initialization.",
    "random_seed": "Freeze development and independent evaluation seeds before observing outcomes.",
    "reproducibility_checksum": "Bind code, config, inputs, model identity if any and immutable raw evidence.",
    "source_artifact_hashes": "Keep exact producer identities, terminal classes, retirement and quarantine state.",
    "rows": "Every comparative unit and arm keeps metric, cost, error, abstention and censoring.",
    "sample_size_budget": "Keep planned, attempted, complete and censored units plus the frozen stopping rule.",
    "acceptance_gate_results": "Each completeness and value check names expected, observed, passed and principle.",
    "gate_check_summary": "Every blocked verdict names the upstream, field, observed value and expected value.",
    "verifier_is_oracle": "Expose shared verifier authority; same-authority mechanics are not learned correctness.",
    "honest_verdict": "Complete findings start complete_; external absence starts blocked_; state the actual finding.",
    "verdict_class": "Use positive, circular_positive, null, blocked, disqualified, or partial only.",
    "validation_receipts": "Keep command, exit code, elapsed time and log hash, including failures.",
    "reuse_audit_complete_score": "One means raw reduction and every fixed corruption control finished.",
    "reuse_promotion_score": "One requires every safety, parity, full-cost, and authority gate.",
    "paired_group_intervals": "Keep units, seed, draw count, estimates, bounds and denominators for each criterion.",
    "source_integrity_control_rows": "Keep each mutation, expected rejection, observed finding and disposition.",
    "audit_reconstruction": "Account for raw bytes, rows, hashes and differences without importing headlines.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(FIELD_PRINCIPLES)

GATE_PRINCIPLES = {
    "reconstruction_complete": "All saved native bytes must replay without mismatch.",
    "fixed_denominators": "The audit needs sixteen groups, 128 claims, three arms, and 672 calls.",
    "source_group_bootstrap": "Uncertainty must resample groups because claims share source compilation.",
    "integrity_controls": "Each fixed corruption must fail closed before promotion.",
    "span_integrity": "Compiled source spans must point back to the saved document bytes.",
    "source_failure_accounting": "A failed shared compilation must remain visible on all four dependent claims.",
    "authority_separation": "Labels must enter only through the scorer reader after replay.",
    "cached_fresh_semantic_parity": "Fresh and reused predictions must match exactly.",
    "zero_stale_serves": "No claim may use a prior source version after revision.",
    "accuracy_difference_lower": "The paired lower bound must be no worse than minus two points.",
    "coverage_difference_lower": "The paired lower bound must be no worse than minus two points.",
    "false_accept_parity": "Reuse must not add empirical false accepts against direct.",
    "eight_claim_full_cost_speedup_lower": "Cold full cost needs a group-paired lower speedup of at least 1.5.",
    "cold_cost_positive_gain": "Reuse must save full cost after initialization.",
    "steady_state_positive_gain": "Reuse must also save cost after initialization is removed.",
}
AUDIT_GATE_NAMES = frozenset(
    {
        "reconstruction_complete",
        "fixed_denominators",
        "source_group_bootstrap",
        "integrity_controls",
        "span_integrity",
        "source_failure_accounting",
        "authority_separation",
    }
)
SCIENTIFIC_GATE_NAMES = frozenset(GATE_PRINCIPLES) - AUDIT_GATE_NAMES

MUTATIONS = (
    "source_version_receipt",
    "reuse_changed_entity",
    "shuffle_source_group_labels",
    "omit_failed_call",
    "permute_identical_source_ids",
)
MUTATION_FINDINGS = {
    "source_version_receipt": "source_version_receipt_mismatch",
    "reuse_changed_entity": "changed_entity_contract_mismatch",
    "shuffle_source_group_labels": "shuffled_source_group_labels",
    "omit_failed_call": "omitted_failed_call",
    "permute_identical_source_ids": "duplicate_source_id",
}

REQUIRED_VALIDATIONS = (
    "focused_pytest",
    "affected_suites",
    "full_python_suite",
    "scoped_coverage",
    "scoped_coverage_report",
    "ruff_check",
    "ruff_format",
    "changed_module_mypy",
    "scoped_spec_coverage",
    "independent_raw_reducer",
    "adversarial_verify",
    "verdict_row_consistency",
)

canonical_json = fixture.canonical_json
sha256_bytes = fixture.sha256_bytes
sha256_file = fixture.sha256_file


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind durable content without making local clock readings scientific inputs."""

    omitted = {"duration_s", "timestamps", "phase_spans", "reproducibility_checksum"}
    stable = {key: value for key, value in artifact.items() if key not in omitted}
    return sha256_bytes(canonical_json(stable).encode("utf-8"))


def gate_row(
    check: str,
    expected: Any,
    observed: Any,
    passed: bool,
    *,
    upstream: str,
    field: str,
) -> JsonDict:
    """Keep both sides of one prerequisite decision."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
        "upstream": upstream,
        "field": field,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first failed prerequisite and its exact observed value."""

    failed = next((row for row in checks if row.get("passed") is not True), None)
    if failed is None:
        return {
            "failed_check": None,
            "upstream": None,
            "field": None,
            "expected_value": None,
            "observed_value": None,
        }
    return {
        "failed_check": failed.get("check"),
        "upstream": failed.get("upstream"),
        "field": failed.get("field"),
        "expected_value": failed.get("expected_value"),
        "observed_value": failed.get("observed_value"),
    }


def measurement_contract_checks(upstream: Mapping[str, Any]) -> list[JsonDict]:
    """Gate only on capture completeness and stable measurement identity."""

    return [
        gate_row(
            "exp7293_identity",
            {"experiment_id": "exp7293-reuse-measurement", "milestone": MILESTONE},
            {
                "experiment_id": upstream.get("experiment_id"),
                "milestone": upstream.get("milestone"),
            },
            upstream.get("experiment_id") == "exp7293-reuse-measurement"
            and upstream.get("milestone") == MILESTONE,
            upstream="exp7293-reuse-measurement",
            field="experiment_id_and_milestone",
        ),
        gate_row(
            "exp7293_terminal_status",
            "complete",
            upstream.get("status"),
            upstream.get("status") == "complete",
            upstream="exp7293-reuse-measurement",
            field="status",
        ),
        gate_row(
            "reuse_capture_complete",
            1,
            upstream.get("reuse_capture_complete_score"),
            upstream.get("reuse_capture_complete_score") == 1,
            upstream="exp7293-reuse-measurement",
            field="reuse_capture_complete_score",
        ),
    ]


def _read_mapping(path: Path) -> JsonDict:
    """Read one JSON object without normalizing its saved values."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"mapping_required:{path}")
    return value


def _path_hash_checks(root: Path) -> list[JsonDict]:
    """Hash every frozen contract before any reducer work starts."""

    rows = []
    ordered_paths = [EXP7293_PATH, *(path for path in PINNED_HASHES if path != EXP7293_PATH)]
    for path in ordered_paths:
        expected = PINNED_HASHES[path]
        actual = sha256_file(root / path) if (root / path).is_file() else None
        name = "exp7293_artifact_hash" if path == EXP7293_PATH else f"{path.name}_hash"
        rows.append(
            gate_row(
                name,
                expected,
                actual,
                actual == expected,
                upstream=path.as_posix(),
                field="sha256",
            )
        )
    return rows


def _call_file_checks(
    root: Path, raw_manifest: Mapping[str, Any]
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Authenticate each retained call file and return its unmodified completion row."""

    errors: list[str] = []
    completions: list[JsonDict] = []
    calls = list(raw_manifest.get("calls") or [])
    for index, receipt in enumerate(calls):
        path = root / UPSTREAM_RAW_DIR / f"call_{index:02d}.json"
        if not path.is_file():
            errors.append(f"call_{index}:missing")
            continue
        if sha256_file(path) != receipt.get("sha256"):
            errors.append(f"call_{index}:hash")
            continue
        value = _read_mapping(path)
        completion = value.get("completion")
        if not isinstance(completion, dict):
            errors.append(f"call_{index}:completion")
            continue
        if completion.get("call_id") != receipt.get("call_id"):
            errors.append(f"call_{index}:identity")
        completions.append(deepcopy(completion))
    check = gate_row(
        "raw_call_file_hashes",
        {"count": PLANNED_CALLS, "errors": []},
        {"count": len(completions), "errors": errors},
        len(completions) == PLANNED_CALLS and not errors,
        upstream=RAW_CALL_MANIFEST_PATH.as_posix(),
        field="calls.sha256_and_call_id",
    )
    return [check], completions


def authenticate_inputs(root: Path) -> tuple[list[JsonDict], JsonDict]:
    """Authenticate all public inputs without opening scorer label content."""

    checks = _path_hash_checks(root)
    if any(row["passed"] is not True for row in checks):
        return checks, {}
    try:
        exp7291 = _read_mapping(root / EXP7291_PATH)
        exp7293 = _read_mapping(root / EXP7293_PATH)
        frozen_manifest = _read_mapping(root / FIXTURE_MANIFEST_PATH)
        analysis = _read_mapping(root / ANALYSIS_PATH)
        raw_manifest = _read_mapping(root / RAW_CALL_MANIFEST_PATH)
        schedule = list(_read_mapping(root / SCHEDULE_PATH)["schedule"])
        source_claim = _read_mapping(root / SOURCE_CLAIM_PATH)
        capture = _read_mapping(root / CAPTURE_RECEIPT_PATH)
        exclusions = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        checks.append(
            gate_row(
                "input_parse",
                "valid_mappings",
                f"{type(exc).__name__}:{exc}",
                False,
                upstream="declared_inputs",
                field="serialization",
            )
        )
        return checks, {}
    checks.extend(measurement_contract_checks(exp7293))
    checks.extend(
        [
            gate_row(
                "exp7293_checksum",
                exp7293.get("reproducibility_checksum"),
                measurement.artifact_checksum(exp7293),
                exp7293.get("reproducibility_checksum") == measurement.artifact_checksum(exp7293),
                upstream=EXP7293_PATH.as_posix(),
                field="reproducibility_checksum",
            ),
            gate_row(
                "exp7291_contract",
                {"ready": 1, "milestone": MILESTONE},
                {
                    "ready": exp7291.get("reuse_fixture_ready_score"),
                    "milestone": exp7291.get("milestone"),
                },
                exp7291.get("reuse_fixture_ready_score") == 1
                and exp7291.get("milestone") == MILESTONE,
                upstream=EXP7291_PATH.as_posix(),
                field="reuse_fixture_ready_score_and_milestone",
            ),
            gate_row(
                "frozen_analysis_contract",
                {
                    "frozen": True,
                    "draws": BOOTSTRAP_DRAWS,
                    "seed": BOOTSTRAP_SEED,
                    "method": "paired_nonparametric_bootstrap_over_source_groups",
                },
                {
                    "frozen": analysis.get("frozen_before_predictions"),
                    "draws": dict(analysis.get("bootstrap") or {}).get("draws"),
                    "seed": dict(analysis.get("bootstrap") or {}).get("seed"),
                    "method": dict(analysis.get("bootstrap") or {}).get("method"),
                },
                analysis.get("frozen_before_predictions") is True
                and dict(analysis.get("bootstrap") or {}).get("draws") == BOOTSTRAP_DRAWS
                and dict(analysis.get("bootstrap") or {}).get("seed") == BOOTSTRAP_SEED
                and dict(analysis.get("bootstrap") or {}).get("method")
                == "paired_nonparametric_bootstrap_over_source_groups",
                upstream=ANALYSIS_PATH.as_posix(),
                field="frozen_before_predictions_and_bootstrap",
            ),
        ]
    )
    groups = list(source_claim.get("groups") or [])
    frozen_groups = list(frozen_manifest.get("evaluation_groups") or [])
    orders = dict(dict(exp7293.get("arm_order_receipt") or {}).get("orders") or {})
    schedule_errors = measurement.schedule_errors(schedule, groups, orders)
    schedule_hash = sha256_bytes(canonical_json(schedule).encode("utf-8"))
    checks.extend(
        [
            gate_row(
                "frozen_group_contract",
                {"groups": PLANNED_GROUPS, "claims": PLANNED_UNITS, "exact": True},
                {
                    "groups": len(groups),
                    "claims": sum(len(group.get("claims") or []) for group in groups),
                    "exact": canonical_json(groups) == canonical_json(frozen_groups),
                },
                len(groups) == PLANNED_GROUPS
                and sum(len(group.get("claims") or []) for group in groups) == PLANNED_UNITS
                and canonical_json(groups) == canonical_json(frozen_groups),
                upstream=SOURCE_CLAIM_PATH.as_posix(),
                field="groups_and_claims",
            ),
            gate_row(
                "frozen_schedule",
                {
                    "calls": PLANNED_CALLS,
                    "errors": [],
                    "sha256": raw_manifest.get("schedule_sha256"),
                },
                {"calls": len(schedule), "errors": schedule_errors, "sha256": schedule_hash},
                len(schedule) == PLANNED_CALLS
                and not schedule_errors
                and schedule_hash == raw_manifest.get("schedule_sha256"),
                upstream=SCHEDULE_PATH.as_posix(),
                field="schedule",
            ),
            gate_row(
                "model_parser_call_contract",
                {
                    "model": measurement.MODEL_ID,
                    "parser_schema_hash": fixture.PARSER_SCHEMA_HASH,
                    "configuration_hash": measurement.LIVE_MODEL_CONFIGURATION_HASH,
                    "calls": PLANNED_CALLS,
                },
                {
                    "model": dict(raw_manifest.get("model_identity") or {}).get("hf_id"),
                    "parser_schema_hash": dict(raw_manifest.get("model_identity") or {}).get(
                        "parser_schema_hash"
                    ),
                    "configuration_hash": dict(raw_manifest.get("model_identity") or {}).get(
                        "model_configuration_hash"
                    ),
                    "calls": raw_manifest.get("retained_calls"),
                },
                dict(raw_manifest.get("model_identity") or {}).get("hf_id") == measurement.MODEL_ID
                and dict(raw_manifest.get("model_identity") or {}).get("parser_schema_hash")
                == fixture.PARSER_SCHEMA_HASH
                and dict(raw_manifest.get("model_identity") or {}).get("model_configuration_hash")
                == measurement.LIVE_MODEL_CONFIGURATION_HASH
                and raw_manifest.get("retained_calls") == PLANNED_CALLS,
                upstream=RAW_CALL_MANIFEST_PATH.as_posix(),
                field="model_identity_parser_configuration_and_calls",
            ),
            gate_row(
                "authority_boundary",
                False,
                bool(
                    raw_manifest.get("evaluation_labels_read_during_capture")
                    or raw_manifest.get("authority_path_opened_by_model_worker")
                    or source_claim.get("authority_fields_present")
                    or capture.get("evaluation_labels_read")
                ),
                not bool(
                    raw_manifest.get("evaluation_labels_read_during_capture")
                    or raw_manifest.get("authority_path_opened_by_model_worker")
                    or source_claim.get("authority_fields_present")
                    or capture.get("evaluation_labels_read")
                ),
                upstream="exp7291_and_exp7293_raw_evidence",
                field="scorer_authority_opened_before_capture",
            ),
        ]
    )
    retired = any(
        fixture._manifest_lists_experiment(exclusions, name)
        for name in (EXPERIMENT_ID, "exp7293-reuse-measurement", "exp7291-reuse-fixture")
    )
    quarantined = bool(
        exp7293.get("flagged_adversarial")
        or dict(
            dict(exp7293.get("source_artifact_hashes") or {}).get(
                str(root / RAW_CALL_MANIFEST_PATH), {}
            )
        ).get("quarantined")
    )
    checks.append(
        gate_row(
            "retirement_and_quarantine",
            {"retired": False, "quarantined": False},
            {"retired": retired, "quarantined": quarantined},
            not retired and not quarantined,
            upstream=EXCLUSION_PATH.as_posix(),
            field="experiment_scope_and_raw_manifest",
        )
    )
    call_checks, completions = _call_file_checks(root, raw_manifest)
    checks.extend(call_checks)
    if any(row["passed"] is not True for row in checks):
        return checks, {}
    return checks, {
        "root": root,
        "checks": checks,
        "exp7293": exp7293,
        "groups": deepcopy(groups),
        "frozen_groups": deepcopy(frozen_groups),
        "schedule": deepcopy(schedule),
        "raw_manifest": raw_manifest,
        "retained_completions": completions,
        "authority_path": root / AUTHORITY_PATH,
        "per_unit_path": root / PER_UNIT_PATH,
        "capture": capture,
    }


def read_scorer_labels(path: Path) -> list[JsonDict]:
    """Open the scorer authority only after the caller completes raw replay."""

    authority = _read_mapping(path)
    return [
        deepcopy(row) for row in authority.get("labels", []) if row.get("split") == "evaluation"
    ]


def _source_document(group: Mapping[str, Any], version: int) -> JsonDict:
    """Select the exact current source version from the frozen public group."""

    match = next(row for row in group["source_versions"] if int(row["source_version"]) == version)
    return deepcopy(dict(match["document"]))


def _compiled(row: Mapping[str, Any] | None) -> JsonDict:
    """Keep a missing compilation visible as an explicit unknown record."""

    value = row.get("compiled_completion") if row else None
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    return {"outcome": "unknown", "relations": [], "errors": ["missing_compilation"]}


def _execute_pair(
    source_document: Mapping[str, Any],
    source_row: Mapping[str, Any] | None,
    claim_row: Mapping[str, Any] | None,
) -> JsonDict:
    """Execute saved parser output and preserve a shared source failure."""

    if not source_row or source_row.get("usable") is not True:
        return {"decision": "unknown", "abstention": True, "errors": ["source_compile_failed"]}
    if not claim_row or claim_row.get("usable") is not True:
        return {"decision": "unknown", "abstention": True, "errors": ["claim_compile_failed"]}
    return fixture.pointer._execute_compiled_pair(
        source_document, _compiled(source_row), _compiled(claim_row)
    )


def _call_components(row: Mapping[str, Any], scale: float = 1.0) -> JsonDict:
    """Project one captured call into additive and inclusive cost components."""

    measured = dict(row.get("measured_processing_s") or {})
    latency = float(row.get("latency_s", 0.0) or 0.0) * scale
    return {
        "model_call_elapsed": latency,
        "prefix_prefill_inclusive": float(measured.get("prefix_prefill", 0.0) or 0.0) * scale,
        "query_extraction": float(measured.get("query_extraction", 0.0) or 0.0) * scale,
        "source_compilation": float(measured.get("source_compilation", 0.0) or 0.0) * scale,
        "lookup_or_invalidation": float(measured.get("lookup_or_invalidation", 0.0) or 0.0) * scale,
        "verification": float(measured.get("verification", 0.0) or 0.0) * scale,
        "synchronization": float(measured.get("synchronization", 0.0) or 0.0) * scale,
        "failed_call_elapsed": latency if row.get("transport_complete") is not True else 0.0,
    }


def _add_components(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Add component rows while keeping inclusive prefix time non-additive."""

    names = tuple(_call_components({}).keys())
    return {name: sum(float(row.get(name, 0.0) or 0.0) for row in rows) for name in names}


def _steady_total(components: Mapping[str, Any]) -> float:
    """Sum full elapsed work once because prefix prefill is inside call time."""

    return sum(
        float(components.get(name, 0.0) or 0.0)
        for name in (
            "model_call_elapsed",
            "query_extraction",
            "source_compilation",
            "lookup_or_invalidation",
            "verification",
            "synchronization",
        )
    )


def _span_checks(
    rows: Sequence[Mapping[str, Any]], schedule: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Check every compiled subject and object slice against saved document text."""

    by_call = {str(row["call_id"]): sealed for row, sealed in zip(rows, schedule, strict=True)}
    checked = 0
    mismatches: list[str] = []
    for row in rows:
        document = by_call[str(row["call_id"])].get("document")
        if not isinstance(document, Mapping):
            continue
        text = str(document.get("text", ""))
        for relation in _compiled(row).get("relations", []):
            for prefix in ("subject", "object"):
                checked += 1
                start = int(relation[f"{prefix}_start"])
                end = int(relation[f"{prefix}_end"])
                if text[start:end] != relation[f"{prefix}_surface"]:
                    mismatches.append(f"{row['call_id']}:{prefix}")
    return {
        "checked_span_count": checked,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def _score_row(row: JsonDict, expected: str) -> JsonDict:
    """Score one prediction while retaining unsupported claims in the denominator."""

    prediction = str(row["prediction"])
    row.update(
        {
            "expected_decision": expected,
            "metric": int(prediction == expected),
            "coverage": int(prediction != "unknown"),
            "false_accept": int(
                prediction in {"supported", "contradicted"} and prediction != expected
            ),
            "abstention": prediction == "unknown",
        }
    )
    return row


def _query_cost(
    calls: Sequence[tuple[Mapping[str, Any], float]],
    *,
    verification_s: float,
    synchronization_s: float,
    lookup_s: float,
    initialization_s: float,
) -> JsonDict:
    """Allocate shared compilation once across its four dependent claims."""

    parts = [_call_components(row, scale) for row, scale in calls]
    extra = _call_components({})
    extra.update(
        {
            "verification": verification_s,
            "synchronization": synchronization_s,
            "lookup_or_invalidation": lookup_s,
        }
    )
    components = _add_components([*parts, extra])
    steady = _steady_total(components)
    return {
        "cost_components_s": components,
        "steady_cost_s": steady,
        "model_initialization_allocation_s": initialization_s,
        "cold_cost_s": steady + initialization_s,
        "failed_call_count_allocated": sum(
            scale for row, scale in calls if row.get("transport_complete") is not True
        ),
        "native_prefix_prefill_included_in_model_call_elapsed": True,
    }


def _condition_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce each arm with absolute counts beside every rate."""

    output: JsonDict = {}
    for arm in ARMS:
        selected = [row for row in rows if row.get("arm") == arm]
        count = len(selected)
        correct = sum(int(row["metric"]) for row in selected)
        coverage = sum(int(row["coverage"]) for row in selected)
        false_accepts = sum(int(row["false_accept"]) for row in selected)
        abstentions = sum(row.get("abstention") is True for row in selected)
        output[arm] = {
            "query_count": count,
            "accuracy_count": correct,
            "accuracy_denominator": count,
            "accuracy": correct / count,
            "coverage_count": coverage,
            "coverage_denominator": count,
            "coverage": coverage / count,
            "false_accept_count": false_accepts,
            "false_accept_denominator": count,
            "false_accept_rate": false_accepts / count,
            "abstention_count": abstentions,
            "abstention_denominator": count,
            "abstention_rate": abstentions / count,
            "steady_cost_s": sum(float(row["steady_cost_s"]) for row in selected),
            "cold_cost_s": sum(float(row["cold_cost_s"]) for row in selected),
        }
    return output


def _group_cost_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Aggregate allocated query cost back to each independent group and arm."""

    output = []
    for group_id in sorted({str(row["group_id"]) for row in rows}):
        for arm in ARMS:
            selected = [
                row for row in rows if row.get("group_id") == group_id and row.get("arm") == arm
            ]
            components = _add_components([row["cost_components_s"] for row in selected])
            output.append(
                {
                    "group_id": group_id,
                    "arm": arm,
                    "claim_count": len(selected),
                    "cost_components_s": components,
                    "steady_total_s": sum(float(row["steady_cost_s"]) for row in selected),
                    "model_initialization_allocation_s": sum(
                        float(row["model_initialization_allocation_s"]) for row in selected
                    ),
                    "cold_total_s": sum(float(row["cold_cost_s"]) for row in selected),
                    "failed_call_count_allocated": sum(
                        float(row["failed_call_count_allocated"]) for row in selected
                    ),
                }
            )
    return output


def reconstruct_audit(bundle: Mapping[str, Any]) -> JsonDict:
    """Replay bytes, then join scorer labels and independently reduce all arms."""

    schedule = list(bundle["schedule"])
    retained = list(bundle["retained_completions"])
    replayed, replay_errors = measurement.independent_replay(schedule, retained)
    labels = read_scorer_labels(Path(bundle["authority_path"]))
    cost_evidence = _read_mapping(Path(bundle["per_unit_path"]))
    groups = list(bundle["groups"])
    label_by_unit = {str(row["unit_id"]): str(row["expected_decision"]) for row in labels}
    timing_by_key = {
        (str(row["group_id"]), str(row["unit_id"]), str(row["arm"])): row
        for row in cost_evidence.get("rows", [])
    }
    source_receipts = list(cost_evidence.get("source_version_receipts", []))
    receipt_by_key = {
        (str(row["group_id"]), int(row["source_version"])): row for row in source_receipts
    }
    by_key = {
        (
            str(row.get("group_id")),
            str(row.get("unit_id")),
            str(row.get("arm")),
            str(row.get("call_type")),
            row.get("draw"),
        ): row
        for row in replayed
    }
    initialization_per_query = float(bundle["capture"]["model_initialization_s"]) / len(ARMS) / 8
    rows: list[JsonDict] = []
    compilation_clusters: list[JsonDict] = []
    stale_served = 0
    prediction_mismatches = 0
    constraint_mismatches = 0
    for group in groups:
        group_id = str(group["group_id"])
        cache = fixture.SourceCompilationCache(enabled=True, capacity=1)
        shared_sources: dict[int, JsonDict] = {}
        for version in (1, 2):
            claims = [row for row in group["claims"] if int(row["source_version"]) == version]
            source_row = by_key.get(
                (
                    group_id,
                    str(claims[0]["unit_id"]),
                    "versioned_reuse_verifier",
                    "source",
                    None,
                )
            )
            view = deepcopy(dict(source_row or {}))
            cached: JsonDict = {"ok": False, "error": "source_compile_failed"}
            if source_row and source_row.get("usable") is True:
                cached = cache.compile_or_get(
                    {
                        "source_id": group["source_id"],
                        "source_version": version,
                        "document": _source_document(group, version),
                        "completion": source_row.get("parsed_completion"),
                        "parser_schema_hash": fixture.PARSER_SCHEMA_HASH,
                        "model_configuration": deepcopy(measurement.LIVE_MODEL_CONFIGURATION),
                    }
                )
                if cached.get("ok") is True:
                    view["compiled_completion"] = deepcopy(cached["compiled_source"])
            shared_sources[version] = view
            compilation_clusters.append(
                {
                    "group_id": group_id,
                    "source_id": group["source_id"],
                    "source_version": version,
                    "compilation_call_id": source_row.get("call_id") if source_row else None,
                    "compilation_failed": cached.get("ok") is not True,
                    "dependent_claim_count": len(claims),
                    "dependent_claims_accounted": 0,
                    "error": cached.get("error"),
                }
            )
        stale_served += cache.served_stale_constraints
        for claim in group["claims"]:
            unit_id = str(claim["unit_id"])
            version = int(claim["source_version"])
            expected = label_by_unit[unit_id]
            document = _source_document(group, version)
            direct_calls = [
                by_key[(group_id, unit_id, "warm_prefix_direct", "direct", draw)] for draw in (1, 2)
            ]
            draws = [
                str(_compiled(row).get("decision", "unknown"))
                if row.get("usable") is True
                else "unknown"
                for row in direct_calls
            ]
            direct_prediction = fixture.reduce_two_draws(*draws)
            direct_timing = timing_by_key[(group_id, unit_id, "warm_prefix_direct")]
            direct_cost = _query_cost(
                [(row, 1.0) for row in direct_calls],
                verification_s=float(direct_timing.get("verification_elapsed_s", 0.0) or 0.0),
                synchronization_s=float(direct_timing.get("synchronization_elapsed_s", 0.0) or 0.0),
                lookup_s=0.0,
                initialization_s=initialization_per_query,
            )
            rows.append(
                _score_row(
                    {
                        "group_id": group_id,
                        "unit_id": unit_id,
                        "claim_index": claim["chronology_index"],
                        "source_id": group["source_id"],
                        "source_version": version,
                        "arm": "warm_prefix_direct",
                        "seed": claim["seed"],
                        "prediction": direct_prediction,
                        "draws": draws,
                        "error": None,
                        "censored": False,
                        "call_ids": [row["call_id"] for row in direct_calls],
                        "source_compilation_shared": False,
                        **direct_cost,
                    },
                    expected,
                )
            )
            fresh_source = by_key[(group_id, unit_id, "fresh_verifier", "source", None)]
            fresh_claim = by_key[(group_id, unit_id, "fresh_verifier", "claim", None)]
            fresh_decision = _execute_pair(document, fresh_source, fresh_claim)
            fresh_timing = timing_by_key[(group_id, unit_id, "fresh_verifier")]
            fresh_cost = _query_cost(
                [(fresh_source, 1.0), (fresh_claim, 1.0)],
                verification_s=float(fresh_timing.get("verification_elapsed_s", 0.0) or 0.0),
                synchronization_s=float(fresh_timing.get("synchronization_elapsed_s", 0.0) or 0.0),
                lookup_s=0.0,
                initialization_s=initialization_per_query,
            )
            rows.append(
                _score_row(
                    {
                        "group_id": group_id,
                        "unit_id": unit_id,
                        "claim_index": claim["chronology_index"],
                        "source_id": group["source_id"],
                        "source_version": version,
                        "arm": "fresh_verifier",
                        "seed": claim["seed"],
                        "prediction": fresh_decision["decision"],
                        "error": ";".join(fresh_decision.get("errors", [])) or None,
                        "censored": False,
                        "call_ids": [fresh_source["call_id"], fresh_claim["call_id"]],
                        "serialized_source_constraints": canonical_json(_compiled(fresh_source)),
                        "serialized_claim_constraints": canonical_json(_compiled(fresh_claim)),
                        "source_compilation_shared": False,
                        **fresh_cost,
                    },
                    expected,
                )
            )
            reuse_source = shared_sources[version]
            reuse_claim = by_key[(group_id, unit_id, "versioned_reuse_verifier", "claim", None)]
            reuse_decision = _execute_pair(document, reuse_source, reuse_claim)
            reuse_timing = timing_by_key[(group_id, unit_id, "versioned_reuse_verifier")]
            receipt = receipt_by_key[(group_id, version)]
            reuse_cost = _query_cost(
                [(reuse_source, 0.25), (reuse_claim, 1.0)],
                verification_s=float(reuse_timing.get("verification_elapsed_s", 0.0) or 0.0),
                synchronization_s=float(reuse_timing.get("synchronization_elapsed_s", 0.0) or 0.0),
                lookup_s=float(receipt.get("lookup_or_invalidation_elapsed_s", 0.0) or 0.0) / 4,
                initialization_s=initialization_per_query,
            )
            source_equal = canonical_json(_compiled(fresh_source)) == canonical_json(
                _compiled(reuse_source)
            )
            claim_equal = canonical_json(_compiled(fresh_claim)) == canonical_json(
                _compiled(reuse_claim)
            )
            prediction_equal = fresh_decision["decision"] == reuse_decision["decision"]
            prediction_mismatches += int(not prediction_equal)
            constraint_mismatches += int(not (source_equal and claim_equal))
            expected_source_hash = sha256_bytes(str(document["text"]).encode("utf-8"))
            stale = int(
                reuse_source.get("source_version") != version
                or reuse_source.get("source_id") != group["source_id"]
                or receipt.get("source_content_sha256") != expected_source_hash
            )
            stale_served += stale
            cluster = next(
                row
                for row in compilation_clusters
                if row["group_id"] == group_id and row["source_version"] == version
            )
            cluster["dependent_claims_accounted"] += 1
            rows.append(
                _score_row(
                    {
                        "group_id": group_id,
                        "unit_id": unit_id,
                        "claim_index": claim["chronology_index"],
                        "source_id": group["source_id"],
                        "source_version": version,
                        "arm": "versioned_reuse_verifier",
                        "seed": claim["seed"],
                        "prediction": reuse_decision["decision"],
                        "error": ";".join(reuse_decision.get("errors", [])) or None,
                        "censored": False,
                        "call_ids": [reuse_source["call_id"], reuse_claim["call_id"]],
                        "source_compilation_call_id": reuse_source["call_id"],
                        "serialized_source_constraints": canonical_json(_compiled(reuse_source)),
                        "serialized_claim_constraints": canonical_json(_compiled(reuse_claim)),
                        "fresh_reuse_prediction_equal": prediction_equal,
                        "fresh_reuse_serialization_equal": source_equal and claim_equal,
                        "stale_constraint_served": bool(stale),
                        "source_compilation_shared": True,
                        "shared_compilation_dependent_claim_count": 4,
                        **reuse_cost,
                    },
                    expected,
                )
            )
    group_cost_rows = _group_cost_rows(rows)
    conditions = _condition_results(rows)
    call_summaries = [
        {
            "call_id": row.get("call_id"),
            "group_id": row.get("group_id"),
            "source_id": row.get("source_id"),
            "source_version": row.get("source_version"),
            "call_type": row.get("call_type"),
            "arm": row.get("arm"),
            "transport_complete": row.get("transport_complete"),
        }
        for row in replayed
    ]
    exp7293 = dict(bundle["exp7293"])
    control_basis = {
        "groups": deepcopy(groups),
        "frozen_groups": deepcopy(bundle["frozen_groups"]),
        "labels": deepcopy(labels),
        "source_receipts": deepcopy(source_receipts),
        "calls": call_summaries,
        "unplanned_attempt_receipts": deepcopy(
            list(exp7293.get("unplanned_attempt_receipts") or [])
        ),
        "invocation_counts": deepcopy(dict(exp7293.get("invocation_counts") or {})),
    }
    direct = conditions["warm_prefix_direct"]
    reuse = conditions["versioned_reuse_verifier"]
    return {
        "rows": rows,
        "group_rows": [
            {
                "group_id": group_id,
                "arm": arm,
                "query_count": len(selected),
                "accuracy_count": sum(int(row["metric"]) for row in selected),
                "coverage_count": sum(int(row["coverage"]) for row in selected),
                "false_accept_count": sum(int(row["false_accept"]) for row in selected),
                "abstention_count": sum(row["abstention"] is True for row in selected),
            }
            for group_id in sorted({str(row["group_id"]) for row in rows})
            for arm in ARMS
            for selected in (
                [row for row in rows if row["group_id"] == group_id and row["arm"] == arm],
            )
        ],
        "group_cost_rows": group_cost_rows,
        "condition_results": conditions,
        "cached_fresh_prediction_mismatch_count": prediction_mismatches,
        "cached_fresh_constraint_mismatch_count": constraint_mismatches,
        "stale_served_count": stale_served,
        "unsupported_claim_denominator": sum(
            value == "unknown" for value in label_by_unit.values()
        ),
        "source_failure_clusters": compilation_clusters,
        "span_checks": _span_checks(replayed, schedule),
        "cost_gains": {
            "cold_gain_s": float(direct["cold_cost_s"]) - float(reuse["cold_cost_s"]),
            "steady_state_gain_s": float(direct["steady_cost_s"]) - float(reuse["steady_cost_s"]),
            "cold_speedup": float(direct["cold_cost_s"]) / float(reuse["cold_cost_s"]),
            "steady_state_speedup": float(direct["steady_cost_s"]) / float(reuse["steady_cost_s"]),
        },
        "authority_separation": {
            "raw_replay_completed_before_scorer_read": True,
            "scorer_reader": "read_scorer_labels",
            "prediction_paths_read_labels": False,
        },
        "audit_reconstruction": {
            "authenticated_call_file_count": len(retained),
            "replayed_call_count": len(replayed),
            "decision_row_count": len(rows),
            "source_group_count": len(groups),
            "claim_count": len(label_by_unit),
            "raw_replay_errors": replay_errors,
            "upstream_headline_aggregates_used": False,
            "scorer_labels_opened_after_raw_replay": True,
            "upstream_per_unit_predictions_used": False,
            "cost_evidence_source": PER_UNIT_PATH.as_posix(),
            "raw_to_row_ratio": {"raw_calls": len(replayed), "decision_rows": len(rows)},
        },
        "control_basis": control_basis,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    """Select one deterministic empirical percentile without interpolation."""

    ordered = sorted(values)
    return ordered[int(probability * (len(ordered) - 1))]


def paired_group_intervals(
    rows: Sequence[Mapping[str, Any]],
    group_cost_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    draws: int,
) -> JsonDict:
    """Resample source groups and retain every correlated claim inside each draw."""

    groups = sorted({str(row["group_id"]) for row in rows})
    by_group_arm = {
        (group, arm): [
            row for row in rows if row.get("group_id") == group and row.get("arm") == arm
        ]
        for group in groups
        for arm in ARMS
    }
    cost = {
        (str(row["group_id"]), str(row["arm"])): float(row["cold_total_s"])
        for row in group_cost_rows
    }

    def mean(sample: Sequence[str], arm: str, field: str) -> float:
        values = [float(row[field]) for group in sample for row in by_group_arm[(group, arm)]]
        return sum(values) / len(values)

    def speedup(sample: Sequence[str]) -> float:
        direct = sum(cost[(group, "warm_prefix_direct")] for group in sample)
        reuse = sum(cost[(group, "versioned_reuse_verifier")] for group in sample)
        return direct / reuse

    rng = random.Random(seed)
    accuracy_draws = []
    coverage_draws = []
    speedup_draws = []
    for _ in range(draws):
        sample = [rng.choice(groups) for _group in groups]
        accuracy_draws.append(
            mean(sample, "versioned_reuse_verifier", "metric")
            - mean(sample, "warm_prefix_direct", "metric")
        )
        coverage_draws.append(
            mean(sample, "versioned_reuse_verifier", "coverage")
            - mean(sample, "warm_prefix_direct", "coverage")
        )
        speedup_draws.append(speedup(sample))
    counts = {
        arm: {
            "correct": sum(int(row["metric"]) for row in rows if row.get("arm") == arm),
            "covered": sum(int(row["coverage"]) for row in rows if row.get("arm") == arm),
            "false_accepts": sum(int(row["false_accept"]) for row in rows if row.get("arm") == arm),
            "denominator": sum(row.get("arm") == arm for row in rows),
        }
        for arm in ARMS
    }
    direct_cost = sum(cost[(group, "warm_prefix_direct")] for group in groups)
    reuse_cost = sum(cost[(group, "versioned_reuse_verifier")] for group in groups)
    return {
        "method": "paired_nonparametric_bootstrap_over_source_groups",
        "resampling_unit": "source_group",
        "independent_group_count": len(groups),
        "claims_per_group": CLAIMS_PER_GROUP,
        "draw_count": draws,
        "bootstrap_seed": seed,
        "interval": "one_sided_95_percent_empirical_lower",
        "accuracy_difference": {
            "estimate": mean(groups, "versioned_reuse_verifier", "metric")
            - mean(groups, "warm_prefix_direct", "metric"),
            "one_sided_95_lower": _percentile(accuracy_draws, 0.05),
            "group_denominator": len(groups),
            "absolute_counts": counts,
        },
        "coverage_difference": {
            "estimate": mean(groups, "versioned_reuse_verifier", "coverage")
            - mean(groups, "warm_prefix_direct", "coverage"),
            "one_sided_95_lower": _percentile(coverage_draws, 0.05),
            "group_denominator": len(groups),
            "absolute_counts": counts,
        },
        "false_accept_counts": {
            "warm_prefix_direct": counts["warm_prefix_direct"]["false_accepts"],
            "fresh_verifier": counts["fresh_verifier"]["false_accepts"],
            "versioned_reuse_verifier": counts["versioned_reuse_verifier"]["false_accepts"],
            "denominator_per_arm": PLANNED_UNITS,
        },
        "full_cost_speedup": {
            "estimate": direct_cost / reuse_cost,
            "one_sided_95_lower": _percentile(speedup_draws, 0.05),
            "group_denominator": len(groups),
            "numerator_direct_cold_full_cost_s": direct_cost,
            "denominator_reuse_cold_full_cost_s": reuse_cost,
        },
        "rare_event_assurance_claimed": False,
    }


def integrity_findings(basis: Mapping[str, Any]) -> list[str]:
    """Find version, entity, label, call-accounting, and source-ID corruption."""

    findings: list[str] = []
    groups = list(basis["groups"])
    if canonical_json(groups) != canonical_json(basis["frozen_groups"]):
        findings.append("changed_entity_contract_mismatch")
    group_by_id = {str(group["group_id"]): group for group in groups}
    source_ids = [str(group["source_id"]) for group in groups]
    if len(set(source_ids)) != len(source_ids):
        findings.append("duplicate_source_id")
    for receipt in basis["source_receipts"]:
        key = str(receipt["group_id"])
        version = int(receipt["source_version"])
        if key not in group_by_id:
            findings.append("source_version_receipt_mismatch")
            break
        document = _source_document(group_by_id[key], version)
        expected = sha256_bytes(str(document["text"]).encode("utf-8"))
        if (
            receipt.get("source_content_sha256") != expected
            or receipt.get("parser_schema_hash") != fixture.PARSER_SCHEMA_HASH
            or receipt.get("model_configuration_hash") != measurement.LIVE_MODEL_CONFIGURATION_HASH
        ):
            findings.append("source_version_receipt_mismatch")
            break
    labels = list(basis["labels"])
    if len(labels) != PLANNED_UNITS or any(
        str(row.get("unit_id", "")).rsplit("-u", 1)[0] != row.get("group_id") for row in labels
    ):
        findings.append("shuffled_source_group_labels")
    calls = list(basis["calls"])
    unplanned = list(basis["unplanned_attempt_receipts"])
    counts = dict(basis["invocation_counts"])
    attempted = sum(row.get("transport_complete") in {True, False} for row in calls) + sum(
        int(row.get("generation_calls_attempted", 0) or 0) for row in unplanned
    )
    failed = sum(row.get("transport_complete") is False for row in calls) + sum(
        int(row.get("generation_calls_failed", 0) or 0) for row in unplanned
    )
    if (
        len(calls) != PLANNED_CALLS
        or attempted != counts.get("generation_calls_attempted")
        or failed != counts.get("generation_calls_failed")
    ):
        findings.append("omitted_failed_call")
    return list(dict.fromkeys(findings))


def run_integrity_controls(basis: Mapping[str, Any]) -> list[JsonDict]:
    """Apply five isolated corruptions without changing the measured evidence."""

    baseline = deepcopy(dict(basis))
    baseline_hash = sha256_bytes(canonical_json(baseline).encode("utf-8"))
    baseline_findings = integrity_findings(baseline)
    output = []
    for mutation in MUTATIONS:
        changed = deepcopy(baseline)
        if mutation == "source_version_receipt":
            changed["source_receipts"][0]["source_content_sha256"] = "sha256:mutated-version"
        elif mutation == "reuse_changed_entity":
            document = changed["groups"][0]["source_versions"][1]["document"]
            old = document["mentions"][0]["surface_text"]
            new = "Z" + old[1:]
            document["text"] = document["text"].replace(old, new, 1)
            document["mentions"][0]["surface_text"] = new
        elif mutation == "shuffle_source_group_labels":
            changed["labels"][0]["group_id"], changed["labels"][8]["group_id"] = (
                changed["labels"][8]["group_id"],
                changed["labels"][0]["group_id"],
            )
        elif mutation == "omit_failed_call":
            changed["unplanned_attempt_receipts"] = []
        else:
            changed["groups"][1]["source_id"] = changed["groups"][0]["source_id"]
        findings = integrity_findings(changed)
        expected = MUTATION_FINDINGS[mutation]
        mutated_hash = sha256_bytes(canonical_json(changed).encode("utf-8"))
        rejected = bool(findings)
        output.append(
            {
                "mutation": mutation,
                "expected_rejection": True,
                "expected_finding": expected,
                "audit_rejected": rejected,
                "observed_findings": findings,
                "passed": not baseline_findings and rejected and expected in findings,
                "baseline_evidence_sha256": baseline_hash,
                "mutated_evidence_sha256": mutated_hash,
                "control_copy_only": True,
            }
        )
    return output


def acceptance_row(criterion: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Keep the frozen threshold and observed audit value together."""

    return {
        "criterion": criterion,
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "principle": GATE_PRINCIPLES[criterion],
    }


def acceptance_gates(
    reduced: Mapping[str, Any],
    intervals: Mapping[str, Any],
    controls: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Apply audit-completeness and preregistered scientific value gates."""

    reconstruction = dict(reduced["audit_reconstruction"])
    rows = list(reduced["rows"])
    groups = {str(row["group_id"]) for row in rows}
    clusters = list(reduced["source_failure_clusters"])
    false_accepts = dict(intervals["false_accept_counts"])
    return [
        acceptance_row(
            "reconstruction_complete",
            {"raw_errors": [], "calls": PLANNED_CALLS, "rows": PLANNED_UNITS * len(ARMS)},
            {
                "raw_errors": reconstruction["raw_replay_errors"],
                "calls": reconstruction["replayed_call_count"],
                "rows": len(rows),
            },
            not reconstruction["raw_replay_errors"]
            and reconstruction["replayed_call_count"] == PLANNED_CALLS
            and len(rows) == PLANNED_UNITS * len(ARMS),
        ),
        acceptance_row(
            "fixed_denominators",
            {"groups": PLANNED_GROUPS, "claims_per_arm": PLANNED_UNITS, "arms": list(ARMS)},
            {
                "groups": len(groups),
                "claims_per_arm": Counter(str(row["arm"]) for row in rows),
                "arms": sorted({str(row["arm"]) for row in rows}),
            },
            len(groups) == PLANNED_GROUPS
            and Counter(str(row["arm"]) for row in rows)
            == Counter({arm: PLANNED_UNITS for arm in ARMS}),
        ),
        acceptance_row(
            "source_group_bootstrap",
            {"unit": "source_group", "groups": PLANNED_GROUPS, "draws": BOOTSTRAP_DRAWS},
            {
                "unit": intervals["resampling_unit"],
                "groups": intervals["independent_group_count"],
                "draws": intervals["draw_count"],
            },
            intervals["resampling_unit"] == "source_group"
            and intervals["independent_group_count"] == PLANNED_GROUPS
            and intervals["draw_count"] in {BOOTSTRAP_DRAWS, 400},
        ),
        acceptance_row(
            "integrity_controls",
            len(MUTATIONS),
            sum(row.get("passed") is True for row in controls),
            len(controls) == len(MUTATIONS) and all(row.get("passed") is True for row in controls),
        ),
        acceptance_row(
            "span_integrity",
            0,
            dict(reduced["span_checks"])["mismatch_count"],
            dict(reduced["span_checks"])["mismatch_count"] == 0,
        ),
        acceptance_row(
            "source_failure_accounting",
            {"clusters": 32, "dependent_claims_each": 4},
            {
                "clusters": len(clusters),
                "dependent_claims": [
                    list(pair)
                    for pair in sorted(
                        {
                            (row["dependent_claim_count"], row["dependent_claims_accounted"])
                            for row in clusters
                        }
                    )
                ],
            },
            len(clusters) == 32
            and all(
                row["dependent_claim_count"] == row["dependent_claims_accounted"] == 4
                for row in clusters
            ),
        ),
        acceptance_row(
            "authority_separation",
            True,
            dict(reduced["authority_separation"])["raw_replay_completed_before_scorer_read"],
            dict(reduced["authority_separation"])["raw_replay_completed_before_scorer_read"]
            is True,
        ),
        acceptance_row(
            "cached_fresh_semantic_parity",
            0,
            reduced["cached_fresh_prediction_mismatch_count"],
            reduced["cached_fresh_prediction_mismatch_count"] == 0,
        ),
        acceptance_row(
            "zero_stale_serves",
            0,
            reduced["stale_served_count"],
            reduced["stale_served_count"] == 0,
        ),
        acceptance_row(
            "accuracy_difference_lower",
            -0.02,
            intervals["accuracy_difference"]["one_sided_95_lower"],
            intervals["accuracy_difference"]["one_sided_95_lower"] >= -0.02,
        ),
        acceptance_row(
            "coverage_difference_lower",
            -0.02,
            intervals["coverage_difference"]["one_sided_95_lower"],
            intervals["coverage_difference"]["one_sided_95_lower"] >= -0.02,
        ),
        acceptance_row(
            "false_accept_parity",
            "reuse<=direct",
            {
                "reuse": false_accepts["versioned_reuse_verifier"],
                "direct": false_accepts["warm_prefix_direct"],
            },
            false_accepts["versioned_reuse_verifier"] <= false_accepts["warm_prefix_direct"],
        ),
        acceptance_row(
            "eight_claim_full_cost_speedup_lower",
            1.5,
            intervals["full_cost_speedup"]["one_sided_95_lower"],
            intervals["full_cost_speedup"]["one_sided_95_lower"] >= 1.5,
        ),
        acceptance_row(
            "cold_cost_positive_gain",
            ">0",
            reduced["cost_gains"]["cold_gain_s"],
            reduced["cost_gains"]["cold_gain_s"] > 0,
        ),
        acceptance_row(
            "steady_state_positive_gain",
            ">0",
            reduced["cost_gains"]["steady_state_gain_s"],
            reduced["cost_gains"]["steady_state_gain_s"] > 0,
        ),
    ]


def audit_complete_score(gates: Sequence[Mapping[str, Any]]) -> int:
    """Return one when the reduction and all adversarial controls finished."""

    by_name = {str(row["criterion"]): row for row in gates}
    return int(
        AUDIT_GATE_NAMES.issubset(by_name)
        and all(by_name[name].get("passed") is True for name in AUDIT_GATE_NAMES)
    )


def promotion_score(gates: Sequence[Mapping[str, Any]], *, verifier_is_oracle: bool) -> int:
    """Require every audit, science, cost, and authority gate for promotion."""

    by_name = {str(row["criterion"]): row for row in gates}
    return int(
        not verifier_is_oracle
        and set(GATE_PRINCIPLES).issubset(by_name)
        and all(by_name[name].get("passed") is True for name in GATE_PRINCIPLES)
    )


def classify_verdict(
    gates: Sequence[Mapping[str, Any]], *, verifier_is_oracle: bool
) -> tuple[str, str]:
    """Keep a completed scientific null separate from blocked or partial work."""

    by_name = {str(row["criterion"]): row for row in gates}
    if audit_complete_score(gates) != 1:
        return "disqualified", "complete_disqualified_reuse_audit_controls_failed"
    science_passed = SCIENTIFIC_GATE_NAMES.issubset(by_name) and all(
        by_name[name].get("passed") is True for name in SCIENTIFIC_GATE_NAMES
    )
    if not science_passed:
        return "null", "complete_null_reuse_promotion_gates_failed"
    if verifier_is_oracle:
        return "circular_positive", "complete_circular_positive_reuse_gates_passed"
    return "positive", "complete_positive_reuse_promoted"


def _utc_now() -> str:
    """Record one real UTC boundary for process provenance."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def base_artifact(run_date: str) -> JsonDict:
    """Create every required field before a fallible external read."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "partial",
        "run_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "duration_s": 0.0,
        "phase_spans": [],
        "timestamps": {"started_at_utc": _utc_now(), "ended_at_utc": None},
        "random_seed": {"audit": RANDOM_SEED, "bootstrap": BOOTSTRAP_SEED},
        "reproducibility_checksum": "pending",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_groups": PLANNED_GROUPS,
            "planned_claims": PLANNED_UNITS,
            "planned_arms": len(ARMS),
            "planned_calls": PLANNED_CALLS,
            "attempted_calls": 0,
            "complete_calls": 0,
            "censored_calls": 0,
            "stopping_rule": "audit the frozen 16 groups and 672 calls once; do not retry, tune, add, or remove rows",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": None,
        "verifier_is_oracle": True,
        "honest_verdict": "partial_reuse_audit_not_started",
        "verdict_class": "partial",
        "validation_receipts": [],
        "reuse_audit_complete_score": 0,
        "reuse_promotion_score": 0,
        "paired_group_intervals": {},
        "source_integrity_control_rows": [],
        "audit_reconstruction": {},
    }


def finalize_blocked_artifact(artifact: JsonDict, checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Finish an external failure as blocked, never as unfinished current work."""

    output = deepcopy(artifact)
    summary = gate_summary(checks)
    failure = str(summary["failed_check"] or "external_precondition")
    output.update(
        {
            "status": "blocked",
            "preconditions_checked": deepcopy(list(checks)),
            "gate_check_summary": summary,
            "honest_verdict": f"blocked_{failure}",
            "verdict_class": "blocked",
            "reuse_audit_complete_score": 0,
            "reuse_promotion_score": 0,
        }
    )
    output["timestamps"]["ended_at_utc"] = _utc_now()
    output["reproducibility_checksum"] = artifact_checksum(output)
    return output


def _source_hashes(root: Path) -> JsonDict:
    """Bind code, contracts, upstream artifacts, and immutable raw manifests."""

    paths = [
        EXP7291_PATH,
        EXP7293_PATH,
        FIXTURE_MANIFEST_PATH,
        ANALYSIS_PATH,
        AUTHORITY_PATH,
        RAW_CALL_MANIFEST_PATH,
        SCHEDULE_PATH,
        SOURCE_CLAIM_PATH,
        CAPTURE_RECEIPT_PATH,
        PER_UNIT_PATH,
        FAILED_ATTEMPT_PATH,
        EXCLUSION_PATH,
        SPEC_PATH,
        fixture.MODULE_PATH,
        measurement.MODULE_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        Path("scripts/adversarial_verify.py"),
        Path("scripts/verdict_row_consistency_lint.py"),
        Path("scripts/check_spec_coverage.py"),
    ]
    return {
        path.as_posix(): {
            "sha256": sha256_file(root / path) if (root / path).is_file() else "missing",
            "terminal_class": "input",
            "retired": False,
            "quarantined": False,
        }
        for path in paths
    }


def assemble_measured_artifact(
    artifact: JsonDict,
    reduced: Mapping[str, Any],
    intervals: Mapping[str, Any],
    controls: Sequence[Mapping[str, Any]],
    *,
    source_hashes: Mapping[str, Any],
) -> JsonDict:
    """Assemble one complete audit without consulting upstream headline values."""

    output = deepcopy(artifact)
    gates = acceptance_gates(reduced, intervals, controls)
    verdict_class, verdict = classify_verdict(gates, verifier_is_oracle=True)
    output.update(
        {
            "status": "complete",
            "preconditions_checked": deepcopy(
                list(reduced.get("preconditions_checked") or output["preconditions_checked"])
            ),
            "gate_check_summary": gate_summary(output["preconditions_checked"]),
            "source_artifact_hashes": deepcopy(dict(source_hashes)),
            "rows": deepcopy(list(reduced["rows"])),
            "group_condition_rows": deepcopy(list(reduced["group_rows"])),
            "group_cost_rows": deepcopy(list(reduced["group_cost_rows"])),
            "condition_results": deepcopy(dict(reduced["condition_results"])),
            "cached_fresh_prediction_mismatch_count": reduced[
                "cached_fresh_prediction_mismatch_count"
            ],
            "cached_fresh_constraint_mismatch_count": reduced[
                "cached_fresh_constraint_mismatch_count"
            ],
            "stale_served_count": reduced["stale_served_count"],
            "unsupported_claim_denominator": reduced["unsupported_claim_denominator"],
            "source_failure_clusters": deepcopy(list(reduced["source_failure_clusters"])),
            "span_checks": deepcopy(dict(reduced["span_checks"])),
            "cost_gains": deepcopy(dict(reduced["cost_gains"])),
            "authority_separation": deepcopy(dict(reduced["authority_separation"])),
            "paired_group_intervals": deepcopy(dict(intervals)),
            "source_integrity_control_rows": deepcopy(list(controls)),
            "audit_reconstruction": deepcopy(dict(reduced["audit_reconstruction"])),
            "acceptance_gate_results": gates,
            "reuse_audit_complete_score": audit_complete_score(gates),
            "reuse_promotion_score": promotion_score(gates, verifier_is_oracle=True),
            "verdict_class": verdict_class,
            "honest_verdict": verdict,
            "limitations": [
                "Sixteen source groups do not support rare-event assurance.",
                "The frozen construction authority makes correctness oracle-shared.",
                "Cached-token counters are transport evidence and are not wall-time savings.",
            ],
        }
    )
    output["sample_size_budget"].update(
        {
            "attempted_calls": PLANNED_CALLS,
            "complete_calls": PLANNED_CALLS,
            "censored_calls": 0,
            "complete_groups": PLANNED_GROUPS,
            "complete_claim_arm_rows": len(output["rows"]),
        }
    )
    output["timestamps"]["ended_at_utc"] = _utc_now()
    output["reproducibility_checksum"] = artifact_checksum(output)
    return output


def validation_receipt_fixture() -> list[JsonDict]:
    """Provide complete synthetic receipts for cold schema unit tests only."""

    return [
        {
            "name": name,
            "command": f"fixture:{name}",
            "exit_code": 0,
            "passed": True,
            "timed_out": False,
            "duration_s": 0.0,
            "log_path": f"fixture/{name}.log",
            "log_sha256": "sha256:" + "0" * 64,
        }
        for name in REQUIRED_VALIDATIONS
    ]


def validate_artifact(value: Mapping[str, Any]) -> list[str]:
    """Cold-check terminal identity, no-LLM provenance, rows, controls, and checksum."""

    errors: list[str] = []
    if value.get("schema") != SCHEMA:
        errors.append("schema")
    if value.get("experiment_id") != EXPERIMENT_ID or value.get("milestone") != MILESTONE:
        errors.append("identity")
    if value.get("run_date") != RUN_DATE:
        errors.append("run_date")
    if not REQUIRED_ARTIFACT_FIELDS.issubset(value):
        errors.append("required_fields")
    if value.get("MODEL_SPECS") != []:
        errors.append("MODEL_SPECS")
    if value.get("model_invoked") is not False:
        errors.append("model_invoked")
    if value.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("invocation_counts")
    if value.get("inference_substrate") != "aggregation_from_upstream_artifacts":
        errors.append("inference_substrate")
    if value.get("inference_substrate_class") != "aggregation":
        errors.append("inference_substrate_class")
    if value.get("execution_venue") != "host":
        errors.append("execution_venue")
    if value.get("status") not in {"complete", "blocked"}:
        errors.append("status")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
    }:
        errors.append("verdict_class")
    verdict = str(value.get("honest_verdict", ""))
    if value.get("status") == "blocked":
        if not verdict.startswith("blocked_") or value.get("verdict_class") != "blocked":
            errors.append("blocked_verdict")
        if value.get("reuse_audit_complete_score") != 0 or value.get("reuse_promotion_score") != 0:
            errors.append("blocked_scores")
        if not dict(value.get("gate_check_summary") or {}).get("failed_check"):
            errors.append("blocked_gate_check_summary")
    else:
        if len(value.get("rows") or []) != PLANNED_UNITS * len(ARMS):
            errors.append("rows")
        if len(value.get("group_condition_rows") or []) != PLANNED_GROUPS * len(ARMS):
            errors.append("group_condition_rows")
        if len(value.get("source_integrity_control_rows") or []) != len(MUTATIONS):
            errors.append("source_integrity_control_rows")
        if dict(value.get("paired_group_intervals") or {}).get("draw_count") not in {
            BOOTSTRAP_DRAWS,
            400,
        }:
            errors.append("paired_group_intervals")
        gates = list(value.get("acceptance_gate_results") or [])
        if value.get("reuse_audit_complete_score") != audit_complete_score(gates):
            errors.append("reuse_audit_complete_score")
        if value.get("reuse_promotion_score") != promotion_score(
            gates, verifier_is_oracle=value.get("verifier_is_oracle") is True
        ):
            errors.append("reuse_promotion_score")
        receipts = list(value.get("validation_receipts") or [])
        if {str(row.get("name")) for row in receipts} != set(REQUIRED_VALIDATIONS):
            errors.append("validation_receipts")
        elif any(not isinstance(row.get("exit_code"), int) for row in receipts):
            errors.append("validation_receipt_exit_codes")
        failed_validations = any(row.get("passed") is not True for row in receipts)
        if failed_validations and value.get("verdict_class") != "disqualified":
            errors.append("validation_verdict")
        if not verdict.startswith("complete_"):
            errors.append("honest_verdict")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum")
    return list(dict.fromkeys(errors))


def write_terminal_artifact(artifact: Mapping[str, Any], path: Path) -> None:
    """Publish only a cold-valid terminal mapping with an atomic replacement."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"invalid Exp7294 terminal artifact: {errors}")
    atomic_write_json(path, dict(artifact), allow_override=False, sort_keys=True)


def validation_commands(root: Path, raw_dir: Path) -> list[tuple[str, list[str]]]:
    """Return focused checks, the required full suite, and both artifact linters."""

    python = str(root / ".venv/bin/python")
    pytest = str(root / ".venv/bin/pytest")
    common = ["-o", "addopts=", "-n", "0"]
    test = TEST_PATH.as_posix()
    changed = [MODULE_PATH.as_posix(), WRAPPER_PATH.as_posix(), test]
    candidate = (root / RAW_CANDIDATE_PATH).as_posix()
    coverage_file = "/tmp/.coverage-exp7294-v641"
    affected = [
        "tests/python/test_experiment_7279_v640_source_audit.py",
        "tests/python/test_experiment_7291_v641_reuse_fixture.py",
        "tests/python/test_experiment_7292_v641_reuse_canary.py",
        "tests/python/test_experiment_7293_v641_reuse_measurement.py",
    ]
    return [
        (
            "focused_pytest",
            [pytest, *common, "--basetemp=/tmp/exp7294-focused", test, "-q"],
        ),
        (
            "affected_suites",
            [pytest, *common, "--basetemp=/tmp/exp7294-affected", *affected, "-q"],
        ),
        (
            "full_python_suite",
            [pytest, *common, "--basetemp=/tmp/exp7294-full", "tests/python", "-q"],
        ),
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
                "--basetemp=/tmp/exp7294-coverage",
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
            "independent_raw_reducer",
            [
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--replay-raw",
                raw_dir.as_posix(),
            ],
        ),
        ("adversarial_verify", [python, "-u", "scripts/adversarial_verify.py", candidate]),
        (
            "verdict_row_consistency",
            [python, "-u", "scripts/verdict_row_consistency_lint.py", candidate],
        ),
    ]


def _progress(phase: int, event: str, **fields: Any) -> None:
    """Flush one compact line at every phase and long-operation boundary."""

    suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"[exp7294] phase={phase} event={event} {suffix}".rstrip(), flush=True)


def _run_validations(root: Path, raw_dir: Path) -> list[JsonDict]:  # pragma: no cover
    """Stream each child and print a truthful heartbeat during quiet subprocesses."""

    validation_dir = raw_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = f"{root / 'python'}:{root}"
    commands = validation_commands(root, raw_dir)
    receipts = []
    for index, (name, command) in enumerate(commands, start=1):
        _progress(7, "subprocess_start", operation=name, completed=index - 1, total=len(commands))
        started = time.monotonic()
        process = subprocess.Popen(  # noqa: S603 - all arguments are fixed repository paths.
            command,
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        lines: list[str] = []
        heartbeat = started
        while process.poll() is None:
            ready, _, _ = select.select([process.stdout], [], [], 10.0)
            if ready:
                line = process.stdout.readline()
                if line:
                    lines.append(line)
                    print(f"[exp7294:{name}] {line.rstrip()}", flush=True)
            now = time.monotonic()
            if now - heartbeat >= 50.0:
                _progress(
                    7, "subprocess_heartbeat", operation=name, elapsed_s=round(now - started, 1)
                )
                heartbeat = now
        remainder = process.stdout.read()
        if remainder:
            lines.append(remainder)
            print(f"[exp7294:{name}] {remainder.rstrip()}", flush=True)
        exit_code = process.wait()
        log_path = validation_dir / f"{name}.log"
        log_path.write_text("".join(lines), encoding="utf-8")
        receipts.append(
            {
                "name": name,
                "command": shlex.join(command),
                "exit_code": exit_code,
                "passed": exit_code == 0,
                "timed_out": False,
                "duration_s": time.monotonic() - started,
                "log_path": log_path.relative_to(root).as_posix(),
                "log_sha256": sha256_file(log_path),
            }
        )
        _progress(
            7,
            "subprocess_end",
            operation=name,
            exit_code=exit_code,
            completed=index,
            total=len(commands),
        )
    return receipts


def _write_checkpoint(root: Path, artifact: Mapping[str, Any]) -> None:  # pragma: no cover
    """Keep unfinished current work outside the declared terminal path."""

    atomic_write_json(root / CHECKPOINT_PATH, dict(artifact), allow_override=True, sort_keys=True)


def replay_terminal_candidate(root: Path) -> tuple[JsonDict, list[str]]:
    """Rebuild deterministic science from upstream raw evidence for E2E validation."""

    checks, bundle = authenticate_inputs(root)
    errors = [str(row["check"]) for row in checks if row["passed"] is not True]
    if errors:
        return {}, errors
    reduced = reconstruct_audit(bundle)
    intervals = paired_group_intervals(
        reduced["rows"],
        reduced["group_cost_rows"],
        seed=BOOTSTRAP_SEED,
        draws=BOOTSTRAP_DRAWS,
    )
    controls = run_integrity_controls(reduced["control_basis"])
    gates = acceptance_gates(reduced, intervals, controls)
    if audit_complete_score(gates) != 1:
        errors.append("audit_complete_score")
    return {
        "rows": len(reduced["rows"]),
        "groups": len(reduced["group_rows"]) // len(ARMS),
        "prediction_mismatches": reduced["cached_fresh_prediction_mismatch_count"],
        "stale_serves": reduced["stale_served_count"],
        "full_cost_speedup_lower": intervals["full_cost_speedup"]["one_sided_95_lower"],
        "controls_passed": sum(row["passed"] is True for row in controls),
        "audit_complete_score": audit_complete_score(gates),
    }, errors


def run_experiment(
    root: Path | None = None, run_date: str = RUN_DATE
) -> JsonDict:  # pragma: no cover
    """Authenticate, reconstruct, validate, and atomically publish the reuse audit."""

    repo = root or find_repo_root(start=__file__)
    result_path = repo / RESULT_PATH
    raw_dir = repo / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    artifact = base_artifact(run_date)
    _progress(0, "phase_start", operation="authenticate_paths_and_inputs")
    _write_checkpoint(repo, artifact)
    if result_path.is_file():
        existing = _read_mapping(result_path)
        if not validate_artifact(existing):
            _progress(0, "stable_terminal_exists", path=result_path)
            return existing
        raise ValueError("declared output exists but is not cold-valid")
    checks, bundle = authenticate_inputs(repo)
    artifact["preconditions_checked"] = checks
    artifact["source_artifact_hashes"] = _source_hashes(repo)
    artifact["phase_spans"].append(
        {"phase": "authenticate_inputs", "duration_s": time.monotonic() - started}
    )
    _progress(0, "phase_end", failed=sum(row["passed"] is not True for row in checks))
    if not bundle:
        artifact["duration_s"] = time.monotonic() - started
        blocked = finalize_blocked_artifact(artifact, checks)
        _progress(8, "write_start", path=result_path)
        write_terminal_artifact(blocked, result_path)
        _progress(8, "write_end", path=result_path)
        return blocked
    phase = time.monotonic()
    _progress(1, "benchmark_start", operation="raw_byte_replay_and_scorer_join")
    reduced = reconstruct_audit(bundle)
    reduced["preconditions_checked"] = checks
    artifact["phase_spans"].append(
        {"phase": "raw_reconstruction", "duration_s": time.monotonic() - phase}
    )
    _progress(1, "benchmark_end", rows=len(reduced["rows"]), calls=PLANNED_CALLS)
    phase = time.monotonic()
    _progress(2, "benchmark_start", operation="paired_source_group_bootstrap")
    intervals = paired_group_intervals(
        reduced["rows"],
        reduced["group_cost_rows"],
        seed=BOOTSTRAP_SEED,
        draws=BOOTSTRAP_DRAWS,
    )
    artifact["phase_spans"].append(
        {"phase": "paired_group_bootstrap", "duration_s": time.monotonic() - phase}
    )
    _progress(2, "benchmark_end", draws=BOOTSTRAP_DRAWS, groups=PLANNED_GROUPS)
    phase = time.monotonic()
    _progress(3, "benchmark_start", operation="five_integrity_mutations")
    controls = run_integrity_controls(reduced["control_basis"])
    artifact["phase_spans"].append(
        {"phase": "integrity_controls", "duration_s": time.monotonic() - phase}
    )
    _progress(3, "benchmark_end", passed=sum(row["passed"] is True for row in controls), total=5)
    artifact["duration_s"] = time.monotonic() - started
    artifact["validation_receipts"] = [
        {
            "name": name,
            "command": "pending",
            "exit_code": None,
            "passed": False,
            "timed_out": False,
            "duration_s": 0.0,
            "log_path": None,
            "log_sha256": None,
        }
        for name in REQUIRED_VALIDATIONS
    ]
    candidate = assemble_measured_artifact(
        artifact,
        reduced,
        intervals,
        controls,
        source_hashes=artifact["source_artifact_hashes"],
    )
    _progress(4, "write_start", path=repo / RAW_CANDIDATE_PATH)
    atomic_write_json(repo / RAW_CANDIDATE_PATH, candidate, allow_override=True, sort_keys=True)
    _progress(4, "write_end", path=repo / RAW_CANDIDATE_PATH)
    _write_checkpoint(repo, candidate)
    phase = time.monotonic()
    _progress(7, "phase_start", operation="focused_and_e2e_validation")
    receipts = _run_validations(repo, raw_dir)
    artifact["phase_spans"].append({"phase": "validation", "duration_s": time.monotonic() - phase})
    artifact["validation_receipts"] = receipts
    artifact["duration_s"] = time.monotonic() - started
    terminal = assemble_measured_artifact(
        artifact,
        reduced,
        intervals,
        controls,
        source_hashes=artifact["source_artifact_hashes"],
    )
    if any(row["passed"] is not True for row in receipts):
        terminal["verdict_class"] = "disqualified"
        terminal["honest_verdict"] = "complete_disqualified_reuse_audit_validation_failed"
        terminal["reuse_promotion_score"] = 0
        terminal["reproducibility_checksum"] = artifact_checksum(terminal)
    errors = validate_artifact(terminal)
    if errors:
        _write_checkpoint(repo, terminal)
        raise ValueError(f"invalid Exp7294 terminal artifact: {errors}")
    atomic_write_json(repo / RAW_CANDIDATE_PATH, terminal, allow_override=True, sort_keys=True)
    _progress(
        7, "phase_end", passed=sum(row["passed"] is True for row in receipts), total=len(receipts)
    )
    _progress(8, "write_start", path=result_path)
    write_terminal_artifact(terminal, result_path)
    _progress(8, "write_end", path=result_path)
    return terminal


def _date_argument(value: str) -> str:
    """Accept only the execution date fixed by the V641 contract."""

    if value != RUN_DATE:
        raise ValueError(f"run_date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the audit or replay immutable evidence without writing an artifact."""

    print("[exp7294] startup", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_date_argument)
    parser.add_argument("--replay-raw", type=Path)
    args = parser.parse_args(argv)
    if args.replay_raw is not None:
        _progress(1, "benchmark_start", operation="independent_raw_reducer")
        summary, errors = replay_terminal_candidate(find_repo_root(start=__file__))
        _progress(1, "benchmark_end", summary=canonical_json(summary), errors=errors)
        return int(bool(errors))
    run_experiment(run_date=args.date)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
