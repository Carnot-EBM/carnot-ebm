"""Qualify the sealed V654 option interface with current scoped checks.

This task replays a scripted interface and authenticates existing raw shards.
It does not load a model, fit a selector, or measure held-out efficacy.

Spec refs: REQ-VERIFY-7476 and SCENARIO-VERIFY-7476-*.
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
import platform
import tempfile
import time
from typing import Any

from carnot import experiment_7462_v654_option_protocol as v654
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]

RUN_DATE = "20260921"
MILESTONE = "2026.09.655"
EXPERIMENT_ID = "exp7476-v655-option-qualification"
SCHEMA = "carnot.exp7476.v655.option_qualification.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7476_v655_option_qualification.json")
RAW_DIR = Path("results/raw/experiment_7476_v655_option_qualification")
MODULE_PATH = Path("python/carnot/experiment_7476_v655_option_qualification.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7476_v655_option_qualification.py")
TEST_PATH = Path("tests/python/test_experiment_7476_v655_option_qualification.py")
SPEC_PATH = REPO_ROOT / "openspec/capabilities/verification/spec.md"
V654_RESULT_PATH = Path("results/experiment_7462_v654_option_protocol.json")
V654_RAW_DIR = Path("results/raw/experiment_7462_v654_option_protocol")

MODEL_SPECS: list[JsonDict] = []
MODEL_SPECS_LOWER: list[JsonDict] = []
INVOCATION_COUNTS = {
    f"{operation}_{state}": 0
    for operation in ("model_loads", "forward_calls", "generation_calls")
    for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
}
INFERENCE_SUBSTRATE = "artifact_aggregation_and_scripted_option_interface_qualification"
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"
EXPECTED_COUNTS = {
    "training": 180,
    "calibration_tuning": 60,
    "internal_test": 60,
    "online": 160,
    "external": 74,
}
FIT_SEEDS = (65_401, 65_402, 65_403, 65_404, 65_405)
ORDERING_SEED = 65_400
AUDIT_SEED = 65_409
BOOTSTRAP_SEED = 6_540_010
TOKEN_CEILING = 2_048
FULL_SUITE_OBSERVATION_NAME = "all_python_tests_required_once"
SMALL_EBM_TRAINING = {
    "attempted": False,
    "fit_attempts": 0,
    "fit_completions": 0,
    "fit_failures": 0,
    "duration_s": 0.0,
    "receipt_class": "small_ebm_training",
    "deferred_to": "post_capture_numeric_fit",
}

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7462_v654_option_protocol.py"),
    Path("tests/python/test_experiment_7462_v654_option_protocol.py"),
    V654_RESULT_PATH,
    Path("python/carnot/experiment_7449_v653_source_protocol.py"),
    Path("openspec/capabilities/verification/spec.md"),
    V654_RAW_DIR / "manifest.json",
    V654_RAW_DIR / "cohort_predictors.jsonl",
    V654_RAW_DIR / "cohort_evaluators.jsonl",
    V654_RAW_DIR / "cohort_groups.jsonl",
    V654_RAW_DIR / "planned_rows.jsonl",
    V654_RAW_DIR / "option_readout_contract.json",
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "started_at_utc",
    "ended_at_utc",
    "started_monotonic_ns",
    "ended_monotonic_ns",
    "clock_identity",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_specs",
    "model_invoked",
    "invocation_counts",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "device_identity",
    "duration_s",
    "duration_components_s",
    "phase_spans",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "rows",
    "sample_size_budget",
    "acceptance_gate_results",
    "gate_check_summary",
    "honest_verdict",
    "verdict_class",
    "verifier_is_oracle",
    "flagged_adversarial",
    "validation_receipts",
    "field_principles",
    "option_protocol_ready_score",
    "cohort_manifest",
    "comparison_plan",
    "repository_health",
    "option_readout_contract",
    "small_ebm_training",
    "current_invocation_events",
    "historical_inference_sidecars",
    "independent_reduction",
    "promotion_score",
    "numbered_e2e_applicable",
    "capability_e2e",
    "production_defaults_changed",
    "external_publication_authorized",
)

AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def classify_repository_health(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep the historical broad-suite failure outside current readiness."""

    observations = [
        deepcopy(dict(row)) for row in receipts if row.get("name") == FULL_SUITE_OBSERVATION_NAME
    ]
    for row in observations:
        row["required_for_exp7476"] = False
        row["classification"] = "unrelated_repository_health"
        row["interrupted_after_failures"] = row.get("exit_code") == 2
    return {
        "status": "degraded_unrelated_baseline" if observations else "not_observed",
        "observations": observations,
        "controls_option_protocol_readiness": False,
        "principle": (
            "A historical broad-suite failure remains visible but cannot replace current "
            "task-scoped checks."
        ),
    }


def replay_option_interface() -> JsonDict:
    """Replay the reusable V654 option helper with scripted score buffers."""

    return v654.run_interface_controls()


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Read one authenticated row shard after the V654 reducer checks its hash."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def reduce_sealed_cohort(raw_dir: Path) -> JsonDict:
    """Rehash and independently inspect the V654 cohort identity fields."""

    replay = v654.reload_raw_protocol(raw_dir)
    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    predictors = _read_jsonl(raw_dir / "cohort_predictors.jsonl")
    evaluators = _read_jsonl(raw_dir / "cohort_evaluators.jsonl")
    groups = _read_jsonl(raw_dir / "cohort_groups.jsonl")
    planned = _read_jsonl(raw_dir / "planned_rows.jsonl")

    predictor_keys = [str(row.get("row_key")) for row in predictors]
    evaluator_keys = [str(row.get("row_key")) for row in evaluators]
    group_ids = [str(row.get("group_id")) for row in groups]
    source_roles: dict[str, set[str]] = {}
    group_roles: dict[str, set[str]] = {}
    for row in groups:
        source_roles.setdefault(str(row.get("source_hash")), set()).add(str(row.get("role")))
        group_roles.setdefault(str(row.get("group_id")), set()).add(str(row.get("role")))

    expected_licenses = {"MIT", "CC BY-NC-SA 4.0"}
    response_ids = sorted(
        f"{row.get('corpus')}:{row.get('group_id')}:{row.get('response_id')}" for row in evaluators
    )
    shard_hashes_valid = all(
        sha256_file(raw_dir / str(row["path"])) == row["sha256"]
        for row in manifest.get("shards", [])
    )
    identity_and_text = (
        len(predictors) == len(set(predictor_keys)) == 534
        and set(predictor_keys) == set(evaluator_keys)
        and all(
            row.get("source_text")
            and row.get("response_text")
            and str(row.get("source_hash", "")).startswith("sha256:")
            and str(row.get("response_hash", "")).startswith("sha256:")
            for row in predictors
        )
    )
    response_ids_complete = len(response_ids) == 534 and all(
        row.get("response_id") not in (None, "") for row in evaluators
    )
    provenance_complete = all(
        row.get("annotation_provenance") and row.get("annotation_disposition") for row in evaluators
    )
    licenses_preserved = (
        {str(row.get("license")) for row in groups} == expected_licenses
        and {str(row.get("license")) for row in predictors} == expected_licenses
        and {
            str(value.get("license"))
            for value in (manifest.get("licenses") or {}).values()
            if isinstance(value, Mapping)
        }
        == expected_licenses
    )
    disjoint = all(len(roles) == 1 for roles in source_roles.values()) and all(
        len(roles) == 1 for roles in group_roles.values()
    )
    planned_ids = {str(row.get("group_id")) for row in planned}
    development_groups_disjoint = planned_ids == set(group_ids)
    passed = all(
        (
            replay["counts"] == EXPECTED_COUNTS,
            replay["total_groups"] == 534,
            replay["planned_cells"] == 1228,
            replay["all_cells_unstarted"] is True,
            replay["interface_passed"] is True,
            replay["predictor_isolated"] is True,
            identity_and_text,
            response_ids_complete,
            provenance_complete,
            licenses_preserved,
            disjoint,
            shard_hashes_valid,
            development_groups_disjoint,
        )
    )
    return {
        **replay,
        "manifest": manifest,
        "shard_hashes_valid": shard_hashes_valid,
        "predictor_identity_and_text_complete": identity_and_text,
        "selected_response_ids_complete": response_ids_complete,
        "selected_response_ids_hash": canonical_hash(response_ids),
        "label_provenance_complete": provenance_complete,
        "licenses_preserved": licenses_preserved,
        "group_and_source_role_disjoint": disjoint,
        "development_transport_groups_disjoint": development_groups_disjoint,
        "passed": passed,
    }


def qualify_comparison_plan(plan: Mapping[str, Any]) -> JsonDict:
    """Check that V655 preserves every prespecified statistical choice."""

    costs = plan.get("decision_costs") or {}
    binary_brier = plan.get("primary_outcome") == "paired_multiclass_brier"
    group_uncertainty = plan.get("bootstrap") == {"draws": 10_000, "unit": "source_group"}
    five_seeds = plan.get("fit_seeds") == list(FIT_SEEDS)
    calibration_only = plan.get("threshold_fit_role") == "calibration_tuning_only"
    external_untouched = (
        plan.get("minimum_usable_groups", {}).get("external") == 60
        and plan.get("stopping_rule")
        == "freeze denominators before capture; do not replace failures or exclusions"
    )
    cost_grid = costs == {
        "false_accept": 10.0,
        "false_reject": 1.0,
        "escalate": 0.2,
        "false_accept_sensitivity": [5.0, 20.0],
        "operator_approved_deployment_policy": False,
    }
    checks = (
        binary_brier,
        group_uncertainty,
        five_seeds,
        calibration_only,
        external_untouched,
        cost_grid,
        plan.get("complete_prompt_token_ceiling") == TOKEN_CEILING,
        plan.get("truncate_to_fit") is False,
    )
    return {
        "binary_brier": binary_brier,
        "group_level_uncertainty": group_uncertainty,
        "five_fitting_seeds": five_seeds,
        "calibration_only_model_choice": calibration_only,
        "external_assessment_untouched": external_untouched,
        "cost_grid_preserved": cost_grid,
        "token_ceiling_and_no_truncation_preserved": checks[-2] and checks[-1],
        "overlength_and_unavailable_counts": "deferred_until_capture",
        "scientific_benefit_measured": False,
        "passed": all(checks),
    }


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    principle: str,
) -> JsonDict:
    """Build one explicit gate whose principle names the prevented failure."""

    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": "==",
        "passed": observed == expected,
        "principle": principle,
    }


def build_acceptance_gates(
    *,
    interface_passed: bool,
    cohort_passed: bool,
    plan_passed: bool,
    current_checks_passed: bool,
    source_hashes_passed: bool,
) -> list[JsonDict]:
    """Keep validity and readiness independent from unmeasured benefit."""

    validity = interface_passed and cohort_passed and plan_passed and source_hashes_passed
    return [
        _gate(
            "option_interface_replay",
            "required_validity",
            True,
            interface_passed,
            "Wrong token positions or option mappings would make the readout invalid.",
        ),
        _gate(
            "sealed_cohort_identity",
            "required_validity",
            True,
            cohort_passed,
            "Changed cohort identities could hide overlap or outcome-driven replacement.",
        ),
        _gate(
            "statistical_protocol_frozen",
            "required_validity",
            True,
            plan_passed,
            "Post-capture metric or cost changes could select a favorable analysis.",
        ),
        _gate(
            "source_artifact_hashes",
            "required_validity",
            True,
            source_hashes_passed,
            "Unbound source bytes could change after the qualification was reported.",
        ),
        _gate(
            "required_validity",
            "required_validity",
            True,
            validity,
            "A positive scientific metric cannot excuse invalid evidence.",
        ),
        _gate(
            "current_required_checks",
            "readiness",
            True,
            current_checks_passed,
            "A failed affected check must close readiness even when old failures are unrelated.",
        ),
        _gate(
            "readiness",
            "readiness",
            True,
            validity and current_checks_passed,
            "A valid null must not suppress an independent measurement.",
        ),
        _gate(
            "scientific_benefit_deferred",
            "scientific_benefit",
            False,
            False,
            "A small sample, a favorable seed or an analytic fixture cannot substitute for held-out value.",
        ),
    ]


def reduce_qualification(
    gates: Sequence[Mapping[str, Any]],
    repository_health: Mapping[str, Any],
    *,
    flagged_adversarial: bool,
) -> JsonDict:
    """Reduce current gates without rehabilitating the historical failure."""

    required = [row for row in gates if row.get("category") != "scientific_benefit"]
    ready = bool(required) and all(row.get("passed") is True for row in required)
    if not ready or flagged_adversarial:
        return {
            "option_protocol_ready_score": 0,
            "status": "disqualified",
            "honest_verdict": "complete_disqualified_option_qualification_validation",
            "verdict_class": "disqualified",
            "promotion_score": 0,
            "repository_health_status": repository_health.get("status"),
        }
    return {
        "option_protocol_ready_score": 1,
        "status": "complete",
        "honest_verdict": "complete_null_option_qualification_ready_no_efficacy_measurement",
        "verdict_class": "null",
        "promotion_score": 0,
        "repository_health_status": repository_health.get("status"),
    }


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind all terminal evidence except the checksum field itself."""

    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key != "reproducibility_checksum"}
    )


def validate_artifact_shape(value: Mapping[str, Any]) -> list[str]:
    """Check terminal identity and the no-model contract without reading shards."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in value]
    if missing:
        return [f"missing_field:{field}" for field in missing]
    errors: list[str] = []
    if (
        value.get("schema") != SCHEMA
        or value.get("experiment_id") != EXPERIMENT_ID
        or value.get("milestone") != MILESTONE
        or value.get("run_date") != RUN_DATE
    ):
        errors.append("identity_invalid")  # pragma: no cover - terminal mutation.
    if (
        value.get("MODEL_SPECS") != []
        or value.get("model_specs") != []
        or value.get("model_invoked") is not False
        or value.get("invocation_counts") != INVOCATION_COUNTS
        or value.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS
        or value.get("execution_venue") != EXECUTION_VENUE
    ):
        errors.append("current_model_contract_invalid")
    if value.get("option_protocol_ready_score") not in {0, 1}:
        errors.append("option_protocol_ready_score_invalid")  # pragma: no cover
    if value.get("promotion_score") != 0:
        errors.append("promotion_nonzero")  # pragma: no cover
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_invalid")  # pragma: no cover
    return errors


def utc_now() -> str:  # pragma: no cover - real process boundary.
    """Return an aware UTC time for one measured boundary."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush a phase boundary so a bounded run never appears stalled."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7476] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _load_object(path: Path) -> JsonDict:  # pragma: no cover - current artifact I/O.
    """Read one JSON object or return an empty object for invalid bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _precondition(
    check: str, path: str, field: str, expected: Any, observed: Any
) -> JsonDict:  # pragma: no cover - artifact construction.
    """Record one prerequisite value before dependent validation starts."""

    return {
        "check": check,
        "upstream": path,
        "path": path,
        "field": field,
        "expected": expected,
        "observed": observed,
        "op": "==",
        "passed": observed == expected,
    }


def collect_preconditions(
    root: Path,
) -> tuple[list[JsonDict], dict[str, JsonDict], JsonDict]:  # pragma: no cover
    """Authenticate exact inputs, V654 flags, and the failed broad receipt."""

    checks: list[JsonDict] = []
    hashes: dict[str, JsonDict] = {}
    for relative in INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            _precondition(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else None,
            )
        )
        if available:
            original_flag = None
            if relative == V654_RESULT_PATH:
                original_flag = _load_object(path).get("flagged_adversarial")
            stat = path.stat()
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "owner_uid": stat.st_uid,
                "owner_gid": stat.st_gid,
                "device_id": stat.st_dev,
                "original_flagged_adversarial": original_flag,
            }

    spec_text = SPEC_PATH.read_text(encoding="utf-8") if SPEC_PATH.is_file() else ""
    checks.append(
        _precondition(
            "driving_requirement",
            "openspec/capabilities/verification/spec.md",
            "REQ-*",
            "REQ-VERIFY-7476",
            "REQ-VERIFY-7476" if "REQ-VERIFY-7476" in spec_text else None,
        )
    )
    previous = _load_object(root / V654_RESULT_PATH)
    for field, expected in (
        ("schema", "carnot.exp7462.v654.option_protocol.v1"),
        ("verdict_class", "disqualified"),
        ("honest_verdict", "complete_disqualified_option_protocol_validation"),
        ("flagged_adversarial", True),
        ("option_protocol_ready_score", 0),
    ):
        checks.append(
            _precondition(
                f"v654_{field}", V654_RESULT_PATH.as_posix(), field, expected, previous.get(field)
            )
        )
    health = classify_repository_health(previous.get("validation_receipts") or [])
    observation = health["observations"][0] if health["observations"] else {}
    for field, expected in (
        ("exit_code", 2),
        ("passed", False),
        (
            "log_sha256",
            "sha256:46e530d225d2a13970316c369662682c551da67781d2572917a73c3b6420d4da",
        ),
    ):
        checks.append(
            _precondition(
                f"historical_full_suite_{field}",
                V654_RESULT_PATH.as_posix(),
                f"validation_receipts.{FULL_SUITE_OBSERVATION_NAME}.{field}",
                expected,
                observation.get(field),
            )
        )
    exclusion = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    checks.append(
        _precondition(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            EXPERIMENT_ID in exclusion,
        )
    )
    return checks, hashes, health


def _source_hashes_valid(root: Path, rows: object) -> bool:  # pragma: no cover - cold I/O.
    """Rehash every declared source so changed bytes close qualification."""

    if not isinstance(rows, Mapping):
        return False
    return all(
        isinstance(row, Mapping)
        and (root / str(row.get("path"))).is_file()
        and sha256_file(root / str(row.get("path"))) == row.get("sha256")
        for row in rows.values()
    )


def _validation_passed(receipts: object, names: Sequence[str]) -> bool:  # pragma: no cover
    """Require exactly one passing current receipt for each named command."""

    return (
        isinstance(receipts, list)
        and all(
            sum(row.get("name") == name and row.get("passed") is True for row in receipts) == 1
            for name in names
        )
        and not any(
            row.get("name") in {FULL_SUITE_OBSERVATION_NAME, "full_python_suite"}
            for row in receipts
        )
    )


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:  # pragma: no cover
    """Name every current failure and the first exact failed comparison."""

    failed = [
        row
        for row in gates
        if row.get("passed") is not True and row.get("category") != "scientific_benefit"
    ]
    first = failed[0] if failed else None
    return {
        "passed": not failed,
        "failed_check": first.get("check") if first else None,
        "upstream": "current_exp7476_qualification" if first else None,
        "path": RAW_DIR.as_posix() if first else None,
        "field": "passed" if first else None,
        "expected": first.get("expected") if first else None,
        "observed": first.get("observed") if first else None,
        "op": first.get("op") if first else None,
        "failed_required_checks": [row["check"] for row in failed],
        "historical_repository_health_controls_readiness": False,
    }


def _field_principles() -> dict[str, str]:  # pragma: no cover - terminal schema.
    """Explain why every required field exists in the terminal record."""

    specific = {
        "schema": "Versioned schema with exact roadmap experiment_id, milestone and terminal status prevents silent reader drift.",
        "run_date": "Use 20260921; measured clocks and process identity prevent a replay from posing as current work.",
        "preconditions_checked": "Named resources, paths, ownership and observed values prevent dependent work on unauthenticated inputs.",
        "MODEL_SPECS": "An empty uppercase model list prevents numeric reducer work from posing as a model task.",
        "model_specs": "The empty lowercase alias prevents schema readers from inventing a current model.",
        "model_invoked": "Attempted current model work stays distinct from archived or scripted events.",
        "invocation_counts": "Balanced load, forward and generation counts expose failed, cancelled or unfinished calls.",
        "inference_substrate": "The substrate distinguishes artifact aggregation from native readout, generation and numeric learning.",
        "inference_substrate_class": "The no-model class prevents short qualification work from claiming model compute.",
        "execution_venue": "The host venue and actual CPU/CUDA identities keep historical board evidence separate.",
        "duration_s": "Measured duration without padding keeps model, numeric and validation work auditable.",
        "phase_spans": "Timestamped progress and checkpoints expose silent or unfinished operations.",
        "random_seed": "Frozen ordering, fitting, audit and bootstrap seeds prevent favorable reselection.",
        "reproducibility_checksum": "One checksum binds code, protocol, roles, model identity, shards and validation scope.",
        "source_artifact_hashes": "Exact upstream bytes preserve original flags and prevent silent evidence replacement.",
        "rows": "Event rows and hash-bound group shards keep independent units, failures and censoring visible.",
        "sample_size_budget": "Separate planned, attempted, complete, failed, censored, excluded and unstarted counts prevent denominator drift.",
        "acceptance_gate_results": "Typed checks preserve validity, readiness and scientific benefit as separate decisions.",
        "gate_check_summary": "Every blocked verdict names the failed check, upstream, path, expected and observed value.",
        "honest_verdict": "A complete terminal null reports qualification without claiming held-out benefit.",
        "verdict_class": "A closed class prevents retryable work or external absence from being mislabeled.",
        "verifier_is_oracle": "False prevents an interface verifier from authorizing a positive scientific claim.",
        "flagged_adversarial": "Real reader flags stay visible and cannot be cleared to open a gate.",
        "validation_receipts": "Exact commands, exits and log hashes establish current scope apart from baseline health.",
        "field_principles": "Field-level reasons make the evidence understandable without hidden context.",
        "option_protocol_ready_score": "A bare 0 or 1 follows current checks and protocol validity, not scientific benefit.",
        "cohort_manifest": "Source-group role hashes prevent hidden overlap or outcome-driven replacement.",
        "comparison_plan": "Frozen budgets, controls, metrics and multiplicity prevent analysis after results are known.",
        "repository_health": "The historical broad-suite failure remains failed and separate from current required checks.",
    }
    return {
        field: specific.get(field, "This field preserves one required part of the terminal audit.")
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _event_rows(
    cohort: Mapping[str, Any], interface: Mapping[str, Any]
) -> list[JsonDict]:  # pragma: no cover
    """Reference large shards and retain one row for each current control event."""

    rows = [
        {
            "unit_id": f"sealed-role:{role}",
            "unit_type": "hash_bound_group_shard_partition",
            "role": role,
            "units": count,
            "disposition": "complete",
            "failed": False,
            "censored": False,
            "shard": "cohort_groups.jsonl",
        }
        for role, count in EXPECTED_COUNTS.items()
    ]
    rows.extend(
        {
            "unit_id": f"interface-mutation:{name}",
            "unit_type": "scripted_interface_control",
            "disposition": "complete",
            "passed": passed,
            "failed": not passed,
            "censored": False,
            "current_model_calls": 0,
        }
        for name, passed in interface.get("mutation_controls", {}).items()
    )
    rows.append(
        {
            "unit_id": "sealed-cohort-reduction",
            "unit_type": "independent_raw_reduction",
            "units": cohort.get("total_groups"),
            "disposition": "complete",
            "passed": cohort.get("passed"),
            "failed": cohort.get("passed") is not True,
            "censored": False,
        }
    )
    return rows


def independent_reduce(
    artifact: Mapping[str, Any], *, root: Path, require_terminal: bool
) -> JsonDict:  # pragma: no cover - fresh-process raw reader.
    """Recompute readiness from raw shards, source hashes and exact receipts."""

    cohort = reduce_sealed_cohort(root / V654_RAW_DIR)
    interface = replay_option_interface()
    plan = qualify_comparison_plan(v654.comparison_plan())
    required_names = [*AFFECTED_CHECK_NAMES]
    if require_terminal:
        required_names.extend(TERMINAL_CHECK_NAMES)
    current_checks = _validation_passed(artifact.get("validation_receipts"), required_names)
    hashes_passed = _source_hashes_valid(root, artifact.get("source_artifact_hashes"))
    gates = build_acceptance_gates(
        interface_passed=interface.get("passed") is True,
        cohort_passed=cohort.get("passed") is True,
        plan_passed=plan.get("passed") is True,
        current_checks_passed=current_checks,
        source_hashes_passed=hashes_passed,
    )
    reduced = reduce_qualification(
        gates,
        artifact.get("repository_health") or {},
        flagged_adversarial=bool(artifact.get("flagged_adversarial")),
    )
    return {
        **reduced,
        "raw_manifest_hash": cohort["manifest_hash"],
        "group_counts": cohort["counts"],
        "total_groups": cohort["total_groups"],
        "planned_cells": cohort["planned_cells"],
        "all_cells_unstarted": cohort["all_cells_unstarted"],
        "selected_response_ids_hash": cohort["selected_response_ids_hash"],
        "required_validation_passed": current_checks,
        "source_artifact_hashes_passed": hashes_passed,
        "acceptance_gate_results": gates,
    }


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    """Build the cold replay and exact terminal readers for one candidate."""

    python = ".venv/bin/python"
    common = ("--date", RUN_DATE, "--root", ".")
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                "declared_entrypoint_cold_replay",
                (python, "-u", WRAPPER_PATH.as_posix(), *common, "--cold-replay", str(candidate)),
                "candidate_artifact",
            ),
            "required_validation",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "independent_raw_reduction",
                (
                    python,
                    "-u",
                    WRAPPER_PATH.as_posix(),
                    *common,
                    "--independent-reduce",
                    str(candidate),
                ),
                "candidate_raw_rows",
            ),
            "required_validation",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                "candidate_artifact",
            ),
            "required_validation",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "verdict_row_consistency_strict",
                (
                    python,
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "candidate_artifact",
            ),
            "required_validation",
            True,
        ),
    ]


def _phase_span(
    phase: str, phase_started: float, run_started: float, units: int, checkpoint: str
) -> JsonDict:  # pragma: no cover
    """Record one monotonic phase interval and its completed-unit checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "checkpoint": checkpoint,
    }


def _build_artifact(  # pragma: no cover - terminal assembly.
    *,
    root: Path,
    preconditions: list[JsonDict],
    source_hashes: dict[str, JsonDict],
    health: Mapping[str, Any],
    cohort: Mapping[str, Any],
    interface: Mapping[str, Any],
    plan: Mapping[str, Any],
    validation_receipts: list[JsonDict],
    phase_spans: list[JsonDict],
    started_at: str,
    started_ns: int,
    ended_ns: int,
    qualification_s: float,
    flagged_adversarial: bool,
    require_terminal: bool,
) -> JsonDict:
    """Assemble a candidate from current controls and immutable V654 bytes."""

    names = [*AFFECTED_CHECK_NAMES, *(TERMINAL_CHECK_NAMES if require_terminal else ())]
    current_checks = _validation_passed(validation_receipts, names)
    hashes_passed = _source_hashes_valid(root, source_hashes)
    gates = build_acceptance_gates(
        interface_passed=interface.get("passed") is True,
        cohort_passed=cohort.get("passed") is True,
        plan_passed=plan.get("passed") is True,
        current_checks_passed=current_checks,
        source_hashes_passed=hashes_passed,
    )
    reduction = reduce_qualification(gates, health, flagged_adversarial=flagged_adversarial)
    duration = (ended_ns - started_ns) / 1_000_000_000
    rows = _event_rows(cohort, interface)
    manifest = deepcopy(cohort["manifest"])
    manifest.update(
        {
            "selected_response_ids_hash": cohort["selected_response_ids_hash"],
            "source_texts_hash_bound": cohort["predictor_identity_and_text_complete"],
            "selected_response_ids_hash_bound": cohort["selected_response_ids_complete"],
            "label_provenance_hash_bound": cohort["label_provenance_complete"],
            "development_transport_groups": [],
            "development_transport_groups_disjoint": cohort[
                "development_transport_groups_disjoint"
            ],
        }
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": reduction["status"],
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "started_monotonic_ns": started_ns,
        "ended_monotonic_ns": ended_ns,
        "clock_identity": {
            "wall": "datetime.now(datetime.UTC)",
            "monotonic": "time.monotonic_ns",
            "pid": os.getpid(),
        },
        "preconditions_checked": preconditions,
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(INVOCATION_COUNTS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "device_identity": {
            "host": platform.node(),
            "platform": platform.platform(),
            "cpu": platform.processor() or platform.machine(),
            "cuda_used": False,
            "cuda_devices_observed": [],
            "historical_board_evidence_used": False,
        },
        "duration_s": duration,
        "duration_components_s": {
            "model_load_s": 0.0,
            "forward_s": 0.0,
            "generation_s": 0.0,
            "numeric_fitting_s": 0.0,
            "qualification_s": qualification_s,
            "validation_s": sum(float(row.get("duration_s") or 0.0) for row in validation_receipts),
        },
        "phase_spans": phase_spans,
        "random_seed": {
            "fit_seeds": list(FIT_SEEDS),
            "ordering_seed": ORDERING_SEED,
            "audit_seed": AUDIT_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "deterministic_reducers": "no_rng",
        },
        "reproducibility_checksum": None,
        "source_artifact_hashes": deepcopy(source_hashes),
        "rows": rows,
        "sample_size_budget": {
            "registered_independent_groups": deepcopy(EXPECTED_COUNTS),
            "planned": 534,
            "attempted": 534,
            "complete": 534,
            "failed": 0,
            "censored": 0,
            "excluded": 0,
            "unstarted": 0,
            "planned_capture_cells": 1228,
            "unstarted_capture_cells": 1228,
            "overlength_groups": None,
            "unavailable_groups": None,
            "usable_group_counts": None,
            "capture_eligibility_status": "deferred_until_tokenizer_and_model_capture",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "honest_verdict": reduction["honest_verdict"],
        "verdict_class": reduction["verdict_class"],
        "verifier_is_oracle": False,
        "flagged_adversarial": flagged_adversarial,
        "validation_receipts": validation_receipts,
        "field_principles": _field_principles(),
        "option_protocol_ready_score": reduction["option_protocol_ready_score"],
        "cohort_manifest": manifest,
        "comparison_plan": deepcopy(dict(v654.comparison_plan())),
        "repository_health": deepcopy(dict(health)),
        "option_readout_contract": deepcopy(dict(interface)),
        "small_ebm_training": deepcopy(SMALL_EBM_TRAINING),
        "current_invocation_events": [],
        "historical_inference_sidecars": [
            {
                "path": V654_RESULT_PATH.as_posix(),
                "sha256": source_hashes[V654_RESULT_PATH.as_posix()]["sha256"],
                "scope": "archived_disqualified_protocol_and_model_shaped_scripted_events",
                "original_flagged_adversarial": True,
                "counted_as_current_inference": False,
            }
        ],
        "independent_reduction": None,
        "promotion_score": 0,
        "numbered_e2e_applicable": [],
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "cold_replay": "fresh_process",
            "independent_raw_reduction": "fresh_process",
            "numbered_runtime_e2e": "not_applicable_pure_reporting_and_protocol_qualification",
        },
        "production_defaults_changed": False,
        "external_publication_authorized": False,
    }
    artifact["independent_reduction"] = independent_reduce(
        artifact, root=root, require_terminal=require_terminal
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(
    value: object, *, root: Path = REPO_ROOT, require_terminal: bool = True
) -> list[str]:  # pragma: no cover - cold terminal reader.
    """Cold-check raw evidence, current scope, hashes, principles, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping_required"]
    artifact = dict(value)
    errors = validate_artifact_shape(artifact)
    try:
        reduction = independent_reduce(artifact, root=root, require_terminal=require_terminal)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        reduction = {}
        errors.append("independent_reduction_failed")
    if artifact.get("independent_reduction") != reduction:
        errors.append("independent_reduction_mismatch")
    for key in (
        "option_protocol_ready_score",
        "status",
        "honest_verdict",
        "verdict_class",
        "promotion_score",
    ):
        if artifact.get(key) != reduction.get(key):
            errors.append(f"{key}_mismatch")
    if not _source_hashes_valid(root, artifact.get("source_artifact_hashes")):
        errors.append("source_artifact_hashes_invalid")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def run_experiment(
    root: Path, run_date: str, *, output: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - declared capability E2E.
    """Authenticate, qualify, validate, replay, and atomically publish V655."""

    if run_date != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    run_started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    spans: list[JsonDict] = []

    progress(run_started, "preconditions", "start")
    phase_started = time.monotonic()
    preconditions, source_hashes, health = collect_preconditions(root)
    if not all(row["passed"] for row in preconditions):
        failed = next(row for row in preconditions if not row["passed"])
        raise RuntimeError(
            f"precondition_failed:{failed['path']}:{failed['field']}:{failed['observed']}"
        )
    spans.append(
        _phase_span("preconditions", phase_started, run_started, len(preconditions), "inputs")
    )
    progress(run_started, "preconditions", "complete", completed=len(preconditions))

    progress(run_started, "model_load", "before", planned=0)
    phase_started = time.monotonic()
    spans.append(_phase_span("model_load", phase_started, run_started, 0, "no_model_load"))
    progress(run_started, "model_load", "after", completed=0)
    progress(run_started, "generation", "before", planned=0)
    phase_started = time.monotonic()
    spans.append(_phase_span("generation", phase_started, run_started, 0, "no_generation"))
    progress(run_started, "generation", "after", completed=0)

    progress(run_started, "qualification", "before_benchmark", planned=534)
    qualification_started = time.monotonic()
    phase_started = time.monotonic()
    interface = replay_option_interface()
    cohort = reduce_sealed_cohort(root / V654_RAW_DIR)
    plan = qualify_comparison_plan(v654.comparison_plan())
    if not (interface.get("passed") and cohort.get("passed") and plan.get("passed")):
        raise RuntimeError("protocol_qualification_failed")
    spans.append(_phase_span("qualification", phase_started, run_started, 534, "sealed_cohort"))
    progress(run_started, "qualification", "after_benchmark", completed=534)
    qualification_s = time.monotonic() - qualification_started

    private_root = Path(tempfile.mkdtemp(prefix="exp7476-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    progress(run_started, "affected_validation", "before_subprocesses", planned=len(commands))
    phase_started = time.monotonic()
    affected = (
        []
        if plan_errors
        else run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=root / RAW_DIR / "validation/affected",
        )
    )
    affected_reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(
        _phase_span(
            "affected_validation", phase_started, run_started, len(affected), "affected_checks"
        )
    )
    progress(
        run_started,
        "affected_validation",
        "after_subprocesses",
        completed=len(affected),
        passed=affected_reduction["passed"],
    )

    flagged = bool(plan_errors) or not bool(affected_reduction["passed"])
    candidate = _build_artifact(
        root=root,
        preconditions=preconditions,
        source_hashes=source_hashes,
        health=health,
        cohort=cohort,
        interface=interface,
        plan=plan,
        validation_receipts=affected,
        phase_spans=spans,
        started_at=started_at,
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        qualification_s=qualification_s,
        flagged_adversarial=flagged,
        require_terminal=False,
    )
    candidate_path = root / RAW_DIR / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    terminal_plan = _terminal_commands(candidate_path)
    progress(run_started, "terminal_validation", "before_subprocesses", planned=len(terminal_plan))
    phase_started = time.monotonic()
    terminal = run_categorized_commands(
        root, terminal_plan, log_dir=root / RAW_DIR / "validation/terminal"
    )
    spans.append(
        _phase_span(
            "terminal_validation", phase_started, run_started, len(terminal), "terminal_readers"
        )
    )
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        run_started,
        "terminal_validation",
        "after_subprocesses",
        completed=len(terminal),
        passed=terminal_passed,
        critical=critical,
    )

    final = _build_artifact(
        root=root,
        preconditions=preconditions,
        source_hashes=source_hashes,
        health=health,
        cohort=cohort,
        interface=interface,
        plan=plan,
        validation_receipts=[*affected, *terminal],
        phase_spans=spans,
        started_at=started_at,
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        qualification_s=qualification_s,
        flagged_adversarial=flagged or not terminal_passed or critical,
        require_terminal=True,
    )
    errors = validate_artifact(final, root=root, require_terminal=True)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(run_started, "publish", "before_atomic_terminal", path=output)
    atomic_json(root / output, final)
    progress(run_started, "publish", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the fixed run date and fresh-process reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Run V655 or one fresh-process terminal reader."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = args.root.resolve()
    if args.cold_replay is not None:
        value = _load_object(args.cold_replay)
        errors = (
            validate_artifact(value, root=root, require_terminal=False)
            if value
            else ["artifact_unreadable"]
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        value = _load_object(args.independent_reduce)
        reduced = independent_reduce(value, root=root, require_terminal=False) if value else {}
        passed = bool(value) and reduced == value.get("independent_reduction")
        print(json.dumps({"passed": passed, "reduced": reduced}, sort_keys=True), flush=True)
        return int(not passed)
    run_experiment(root, args.date, output=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
