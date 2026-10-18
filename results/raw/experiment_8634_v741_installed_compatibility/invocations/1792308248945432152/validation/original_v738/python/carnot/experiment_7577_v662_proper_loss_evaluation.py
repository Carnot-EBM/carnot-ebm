"""Evaluate frozen proper-loss heads on exposed cached evaluator groups.

This read-only experiment separates label-free forecast construction from the
label evaluator. It measures descriptive reuse and cannot refresh the exposed
rows into confirmatory evidence.

Spec refs: REQ-CL-7577 and SCENARIO-CL-7577-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.experiment_7561_v661_recalibration_prototype import map_probability
from carnot.experiment_7567_v661_source_evaluation import loss_fields, primary_decision
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260923"
MILESTONE = "2026.09.662"
EXPERIMENT_ID = "exp7577-v662-proper-loss-evaluation"
SCHEMA = "carnot.exp7577.v662.proper_loss_evaluation.v1"
RESULT_PATH = Path("results/experiment_7577_v662_proper_loss_evaluation.json")
RAW_DIR = Path("results/raw/experiment_7577_v662_proper_loss_evaluation")
MODULE_PATH = Path("python/carnot/experiment_7577_v662_proper_loss_evaluation.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7577_v662_proper_loss_evaluation.py")
TEST_PATH = Path("tests/python/test_experiment_7577_v662_proper_loss_evaluation.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
PROTOCOL_ARTIFACT_PATH = Path("results/experiment_7575_v662_cached_learning_protocol.json")
FIT_ARTIFACT_PATH = Path("results/experiment_7576_v662_proper_loss_energy.json")
TERMINAL_CANDIDATE_PATH = RAW_DIR / "terminal_candidate.json"
AFFECTED_MANIFEST_PATH = RAW_DIR / "affected_validation_manifest.json"
EVALUATION_ROWS_PATH = RAW_DIR / "evaluation_rows.jsonl"
HISTORICAL_MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
EVALUATION_SEED = 7_575_101
EXPECTED_GROUPS = 80
FROZEN_HEAD_ARMS = (
    "proper_loss_monotone",
    "raw_original",
    "temperature_original",
    "unconstrained_nine_knot",
)
ARMS = (*FROZEN_HEAD_ARMS, "escalate_all")
ZERO_INVOCATION_COUNTS = {
    name: {state: 0 for state in ("attempted", "completed", "failed", "cancelled")}
    for name in ("model_loads", "forward_calls", "generation_calls", "tokens")
}
REQUIRED_VALIDATION_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_row_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
REQUIRED_PRINCIPLE_FIELDS = (
    "honest_verdict",
    "verdict_class",
    "flagged_adversarial",
    "gate_check_summary",
    "acceptance_gate_results",
    "rows",
    "inference_substrate_class",
    "MODEL_SPECS",
    "invocation_counts",
    "duration_s",
    "source_artifact_hashes",
    "validation_receipts",
    "field_principles",
    "verifier_is_oracle",
    "static_measurement_complete_score",
    "exploratory_probability_benefit_score",
    "exploratory_decision_benefit_score",
    "fresh_confirmatory_claim_allowed",
    "paired_intervals",
)
AFFECTED_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def _load_object(path: Path) -> JsonDict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _resolved(root: Path, label: str) -> Path:
    path = Path(label)
    return path if path.is_absolute() else root / path


def _source_hash(path: Path, root: Path) -> JsonDict:
    resolved = path if path.is_absolute() else root / path
    try:
        label = resolved.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        label = str(resolved.resolve())
    return {"path": label, "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}


def _precondition(
    check: str,
    upstream: str,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
    *,
    op: str = "eq",
    passed: bool,
) -> JsonDict:
    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "op": op,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
    }


def validate_head_manifest(manifest: Mapping[str, Any]) -> JsonDict:
    """Authenticate each head and the manifest before any outcome is opened."""

    copied = deepcopy(dict(manifest))
    observed_manifest = copied.pop("manifest_sha256", None)
    if observed_manifest != canonical_hash(copied):
        raise ValueError("head_manifest_digest_drift")
    heads = copied.get("heads")
    if not isinstance(heads, Mapping) or set(heads) != set(FROZEN_HEAD_ARMS):
        raise ValueError("head_roster_invalid")
    digests: dict[str, str] = {}
    for name in FROZEN_HEAD_ARMS:
        head = deepcopy(dict(heads[name]))
        observed = head.pop("head_sha256", None)
        if head.get("name") != name or observed != canonical_hash(head):
            raise ValueError(f"head_digest_drift:{name}")
        digests[name] = str(observed)
    strongest = copied.get("strongest_comparator")
    if (
        copied.get("frozen_before_policy_access") is not True
        or not isinstance(strongest, Mapping)
        or strongest.get("name") not in FROZEN_HEAD_ARMS[1:]
        or strongest.get("selected_on_role") != "tune"
        or strongest.get("frozen_before_policy_access") is not True
    ):
        raise ValueError("head_freeze_contract_invalid")
    return {
        "manifest_sha256": str(observed_manifest),
        "head_count": len(digests),
        "head_digests": digests,
        "strongest_comparator": str(strongest["name"]),
        "frozen_before_evaluation_access": True,
    }


def _sidecar_check(
    root: Path,
    receipt: Mapping[str, Any] | None,
    *,
    name: str,
    upstream: str,
    source_hashes: list[JsonDict],
) -> tuple[JsonDict, Path | None]:
    """Verify one upstream sidecar without repairing a missing receipt."""

    row = receipt if isinstance(receipt, Mapping) else {}
    label = str(row.get("path") or "")
    path = _resolved(root, label) if label else None
    observed = sha256_file(path) if path is not None and path.is_file() else "missing"
    expected = row.get("sha256") or "declared_sha256"
    passed = bool(path is not None and path.is_file() and observed == expected)
    if passed and path is not None:
        source_hashes.append(_source_hash(path, root))
    return (
        _precondition(
            name,
            upstream,
            label or "missing_sidecar_path",
            "sha256",
            expected,
            observed,
            passed=passed,
        ),
        path if passed else None,
    )


def collect_preconditions(root: Path) -> tuple[list[JsonDict], list[JsonDict]]:
    """Authenticate instructions, requirements, upstream fields, and sidecars."""

    root = root.resolve()
    checks: list[JsonDict] = []
    source_hashes: list[JsonDict] = []
    owned = (
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("scripts/experiment_template.py"),
        Path("python/carnot/reporting/current_work_receipt.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("python/carnot/experiment_7358_v646_validation_contract.py"),
        Path("python/carnot/experiment_7567_v661_source_evaluation.py"),
        Path("python/carnot/experiment_7569_v661_decision_learning_audit.py"),
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        SPEC_PATH,
    )
    for relative in owned:
        path = root / relative
        exists = path.is_file()
        checks.append(
            _precondition(
                f"owned_resource_{relative.name}",
                "worktree",
                relative.as_posix(),
                "exists",
                True,
                exists,
                passed=exists,
            )
        )
        if exists:
            source_hashes.append(_source_hash(path, root))
    spec_path = root / SPEC_PATH
    spec_present = spec_path.is_file() and "REQ-CL-7577" in spec_path.read_text(encoding="utf-8")
    checks.append(
        _precondition(
            "matching_requirement",
            "continuous-learning-spec",
            SPEC_PATH.as_posix(),
            "REQ-CL-7577",
            "present",
            "present" if spec_present else "missing",
            passed=spec_present,
        )
    )
    artifacts: dict[str, JsonDict] = {}
    for name, relative in (
        ("exp7575", PROTOCOL_ARTIFACT_PATH),
        ("exp7576", FIT_ARTIFACT_PATH),
    ):
        path = root / relative
        exists = path.is_file()
        checks.append(
            _precondition(
                f"{name}_artifact_exists",
                name,
                relative.as_posix(),
                "exists",
                True,
                exists,
                passed=exists,
            )
        )
        if exists:
            try:
                artifacts[name] = _load_object(path)
            except (OSError, ValueError, json.JSONDecodeError):
                artifacts[name] = {}
            source_hashes.append(_source_hash(path, root))
    if set(artifacts) != {"exp7575", "exp7576"}:
        return checks, source_hashes

    protocol_artifact = artifacts["exp7575"]
    fit_artifact = artifacts["exp7576"]
    field_checks = (
        ("exp7575_cached_roles_ready", "exp7575", protocol_artifact, "cached_roles_ready_score", 1),
        ("exp7575_verdict", "exp7575", protocol_artifact, "verdict_class", ("null", "positive")),
        ("exp7575_not_flagged", "exp7575", protocol_artifact, "flagged_adversarial", False),
        ("exp7576_fit_ready", "exp7576", fit_artifact, "proper_loss_fit_ready_score", 1),
        ("exp7576_baseline_ready", "exp7576", fit_artifact, "baseline_ready_score", 1),
        ("exp7576_verdict", "exp7576", fit_artifact, "verdict_class", ("null", "positive")),
        ("exp7576_not_flagged", "exp7576", fit_artifact, "flagged_adversarial", False),
    )
    for check, upstream, artifact, field, expected in field_checks:
        observed = artifact.get(field)
        passed = observed in expected if isinstance(expected, tuple) else observed == expected
        checks.append(
            _precondition(
                check,
                upstream,
                (PROTOCOL_ARTIFACT_PATH if upstream == "exp7575" else FIT_ARTIFACT_PATH).as_posix(),
                field,
                list(expected) if isinstance(expected, tuple) else expected,
                observed,
                op="in" if isinstance(expected, tuple) else "eq",
                passed=passed,
            )
        )
    cached_check, cached_path = _sidecar_check(
        root,
        protocol_artifact.get("raw_sidecars", {}).get("cached_roles"),
        name="cached_roles_hash",
        upstream="exp7575",
        source_hashes=source_hashes,
    )
    checks.append(cached_check)
    protocol_check, protocol_path = _sidecar_check(
        root,
        protocol_artifact.get("raw_sidecars", {}).get("frozen_protocol"),
        name="frozen_protocol_hash",
        upstream="exp7575",
        source_hashes=source_hashes,
    )
    checks.append(protocol_check)
    manifest_receipt = fit_artifact.get("raw_sidecars", {}).get("head_manifest")
    head_check, head_path = _sidecar_check(
        root,
        manifest_receipt,
        name="head_manifest_hash",
        upstream="exp7576",
        source_hashes=source_hashes,
    )
    checks.append(head_check)
    role_rows: list[JsonDict] = []
    if cached_path is not None:
        try:
            with cached_path.open(encoding="utf-8") as stream:
                for line in stream:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("cached_role_row_invalid")
                    if value.get("role") == "test":
                        role_rows.append(value)
        except (OSError, ValueError, json.JSONDecodeError):
            role_rows = []
    checks.append(
        _precondition(
            "evaluation_group_count",
            "exp7575",
            cached_check["path"],
            "role=test count",
            EXPECTED_GROUPS,
            len(role_rows),
            passed=len(role_rows) == EXPECTED_GROUPS,
        )
    )
    protocol: JsonDict = {}
    if protocol_path is not None:
        try:
            protocol = _load_object(protocol_path)
        except (OSError, ValueError, json.JSONDecodeError):
            protocol = {}
    payload = protocol.get("hash_payload")
    digest_valid = bool(
        isinstance(payload, Mapping)
        and protocol.get("protocol_sha256") == canonical_hash(payload)
        and protocol.get("role_counts", {}).get("test") == EXPECTED_GROUPS
        and protocol.get("retention_ids") == protocol.get("role_ids", {}).get("test")
    )
    checks.append(
        _precondition(
            "protocol_digest_and_roster",
            "exp7575",
            protocol_check["path"],
            "protocol_sha256_and_test_roster",
            True,
            digest_valid,
            passed=digest_valid,
        )
    )
    head_valid = False
    observed_head_digest: Any = "missing"
    if head_path is not None:
        try:
            head_manifest = _load_object(head_path)
            validate_head_manifest(head_manifest)
            observed_head_digest = head_manifest.get("manifest_sha256")
            head_valid = True
        except (OSError, ValueError, json.JSONDecodeError):
            observed_head_digest = "invalid"
    checks.append(
        _precondition(
            "head_manifest_digest",
            "exp7576",
            head_check["path"],
            "manifest_sha256",
            "self_consistent",
            observed_head_digest,
            passed=head_valid,
        )
    )
    deduplicated = {(row["path"], row["sha256"]): row for row in source_hashes}
    return checks, list(deduplicated.values())


def split_evaluation_rows(
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    expected_groups: int = EXPECTED_GROUPS,
) -> tuple[list[JsonDict], JsonDict]:
    """Freeze label-free features and a separate hash-bound label evaluator."""

    copied = [deepcopy(dict(row)) for row in rows]
    if len(copied) != expected_groups:
        raise ValueError("evaluation_group_count_invalid")
    source_ids = [str(row.get("source_id") or "") for row in copied]
    if any(not source_id for source_id in source_ids) or len(set(source_ids)) != len(source_ids):
        raise ValueError("duplicate_source_id")
    retention_ids = [str(value) for value in protocol.get("retention_ids") or []]
    if source_ids != retention_ids:
        raise ValueError("evaluation_order_map_drift")
    role_ids = [str(value) for value in protocol.get("role_ids", {}).get("test") or []]
    if source_ids != role_ids:
        raise ValueError("evaluation_roster_drift")
    hash_payload = protocol.get("hash_payload")
    if isinstance(hash_payload, Mapping) and protocol.get("protocol_sha256") != canonical_hash(
        hash_payload
    ):
        raise ValueError("protocol_digest_drift")
    features: list[JsonDict] = []
    labels: list[JsonDict] = []
    mapping_rows: list[JsonDict] = []
    for row in copied:
        if row.get("role") != "test" or row.get("label") not in (0, 1):
            raise ValueError("evaluation_label_invalid")
        requests = row.get("request_hashes")
        if not isinstance(requests, list) or len(requests) != 6 or len(set(requests)) != 6:
            raise ValueError("option_mapping_invalid")
        probability = float(row.get("probability"))
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("evaluation_probability_invalid")
        source_id = str(row["source_id"])
        mapping_hash = canonical_hash(requests)
        mapping_rows.append({"source_id": source_id, "option_mapping_sha256": mapping_hash})
        features.append(
            {
                "source_id": source_id,
                "group_id": str(row.get("group_id") or source_id),
                "p_original": probability,
                "option_mapping_sha256": mapping_hash,
                "context_sha256": str(row.get("context_sha256") or ""),
                "response_sha256": str(row.get("response_sha256") or ""),
                "feature_provenance": "exp7575_cached_original_probability",
            }
        )
        labels.append({"source_id": source_id, "observed_error": int(row["label"])})
    evaluator = {
        "labels": labels,
        "label_count": len(labels),
        "label_binding_sha256": canonical_hash(labels),
        "expected_source_ids": source_ids,
        "option_mapping_bindings": mapping_rows,
        "option_mapping_binding_sha256": canonical_hash(mapping_rows),
        "protocol_sha256": str(protocol.get("protocol_sha256") or "fixture_protocol"),
        "minimum_non_escalation_fraction": float(
            protocol.get("minimum_non_escalation_fraction", 0.10)
        ),
        "bootstrap_draws": int(protocol.get("bootstrap_replays_per_order", 1000)),
        "holm_adjusted_static_contrasts": bool(
            protocol.get("holm_adjusted_static_contrasts", True)
        ),
    }
    return features, evaluator


def _temperature_probability(probability: float, temperature: float) -> float:
    """Apply the immutable scalar temperature without using evaluator labels."""

    if not 0.0 <= probability <= 1.0 or not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_probability_invalid")
    clipped = min(1.0 - 1e-9, max(1e-9, probability))
    logit = math.log(clipped / (1.0 - clipped)) / temperature
    return 1.0 / (1.0 + math.exp(-logit))


def score_frozen_heads(
    features: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> list[JsonDict]:
    """Construct all forecasts before the separate evaluator opens labels."""

    identity = validate_head_manifest(manifest)
    heads = manifest["heads"]
    forecasts: list[JsonDict] = []
    seen: set[str] = set()
    for feature in features:
        if "label" in feature or "observed_error" in feature:
            raise ValueError("feature_label_leakage")
        source_id = str(feature.get("source_id") or "")
        if not source_id or source_id in seen:
            raise ValueError("duplicate_feature")
        seen.add(source_id)
        original = float(feature.get("p_original"))
        if not math.isfinite(original) or not 0.0 <= original <= 1.0:
            raise ValueError("feature_probability_invalid")
        values = {
            "proper_loss_monotone": map_probability(
                original, heads["proper_loss_monotone"]["theta"]
            ),
            "raw_original": original,
            "temperature_original": _temperature_probability(
                original, float(heads["temperature_original"]["temperature"])
            ),
            "unconstrained_nine_knot": map_probability(
                original, heads["unconstrained_nine_knot"]["theta"]
            ),
        }
        for arm in FROZEN_HEAD_ARMS:
            forecasts.append(
                {
                    "source_id": source_id,
                    "group_id": str(feature.get("group_id") or source_id),
                    "arm": arm,
                    "q": float(values[arm]),
                    "option_mapping_sha256": str(feature.get("option_mapping_sha256") or ""),
                    "head_sha256": identity["head_digests"][arm],
                    "feature_provenance": str(feature.get("feature_provenance") or ""),
                }
            )
    return forecasts


def evaluate_forecasts(
    forecasts: Sequence[Mapping[str, Any]],
    evaluator: Mapping[str, Any],
    *,
    expected_arms: Sequence[str] = FROZEN_HEAD_ARMS,
    include_escalate_all: bool = True,
) -> list[JsonDict]:
    """Join sealed labels and score forecasts only after custody freezes."""

    labels = [deepcopy(dict(row)) for row in evaluator.get("labels") or []]
    if canonical_hash(labels) != evaluator.get("label_binding_sha256"):
        raise ValueError("label_binding_drift")
    expected_sources = [str(value) for value in evaluator.get("expected_source_ids") or []]
    if [str(row.get("source_id") or "") for row in labels] != expected_sources:
        raise ValueError("label_roster_drift")
    mapping_rows = [deepcopy(dict(row)) for row in evaluator.get("option_mapping_bindings") or []]
    if canonical_hash(mapping_rows) != evaluator.get("option_mapping_binding_sha256"):
        raise ValueError("option_mapping_binding_drift")
    expected_mapping = {
        str(row["source_id"]): str(row["option_mapping_sha256"]) for row in mapping_rows
    }
    label_by_source = {
        str(row["source_id"]): int(row["observed_error"])
        for row in labels
        if row.get("observed_error") in (0, 1)
    }
    if set(label_by_source) != set(expected_sources):
        raise ValueError("evaluation_label_invalid")
    expected_arm_set = {str(value) for value in expected_arms}
    by_pair: dict[tuple[str, str], JsonDict] = {}
    for raw in forecasts:
        forecast = deepcopy(dict(raw))
        source_id = str(forecast.get("source_id") or "")
        arm = str(forecast.get("arm") or "")
        key = (source_id, arm)
        if key in by_pair:
            raise ValueError("duplicate_forecast")
        if source_id not in label_by_source or arm not in expected_arm_set:
            raise ValueError("forecast_roster_invalid")
        if forecast.get("option_mapping_sha256") != expected_mapping.get(source_id):
            raise ValueError("option_mapping_drift")
        probability = float(forecast.get("q"))
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("forecast_probability_invalid")
        by_pair[key] = forecast
    required_pairs = {
        (source_id, arm) for source_id in expected_sources for arm in expected_arm_set
    }
    if set(by_pair) != required_pairs:
        raise ValueError("forecast_roster_invalid")
    rows: list[JsonDict] = []
    output_arms = (
        (*tuple(expected_arms), "escalate_all") if include_escalate_all else tuple(expected_arms)
    )
    for source_id in expected_sources:
        label = label_by_source[source_id]
        for arm in output_arms:
            constant = arm == "escalate_all"
            forecast = (
                {
                    "group_id": source_id,
                    "q": 0.5,
                    "head_sha256": "constant_policy:escalate_all:0.2",
                    "feature_provenance": "registered_constant_policy",
                }
                if constant
                else by_pair[(source_id, arm)]
            )
            probability = float(forecast["q"])
            losses = loss_fields(probability, label)
            decision = (
                {"action": "escalate", "realized_cost": 0.2}
                if constant
                else primary_decision(probability, label)
            )
            rows.append(
                {
                    "unit_id": source_id,
                    "source_component_id": source_id,
                    "group_id": str(forecast.get("group_id") or source_id),
                    "arm": arm,
                    "q": probability,
                    "probability": probability,
                    "observed_error": label,
                    "label": label,
                    "brier": float(losses["brier"]),
                    "log_loss": float(losses["log_loss"]),
                    "action": decision["action"],
                    "realized_cost": float(decision["realized_cost"]),
                    "non_escalated": decision["action"] != "escalate",
                    "raw_brier_numerator": float(losses["brier"]),
                    "raw_brier_denominator": 1,
                    "raw_log_loss_numerator": float(losses["log_loss"]),
                    "raw_log_loss_denominator": 1,
                    "raw_cost_numerator": float(decision["realized_cost"]),
                    "raw_cost_denominator": 1,
                    "metric_direction": "lower_loss_and_cost_are_better",
                    "seed": EVALUATION_SEED,
                    "censored": False,
                    "status": "complete",
                    "complete": True,
                    "failed": False,
                    "excluded": False,
                    "provenance": {
                        "feature_source": forecast["feature_provenance"],
                        "head_sha256": forecast["head_sha256"],
                        "label_source": "separate_exp7575_evaluator",
                        "option_mapping_sha256": expected_mapping[source_id],
                    },
                }
            )
    return rows


def registered_settings(
    evaluator: Mapping[str, Any],
    *,
    draws: int | None = None,
    strongest_comparator: str = "temperature_original",
) -> JsonDict:
    """Return fixed gates without consulting any evaluation outcome."""

    selected_draws = int(draws or evaluator.get("bootstrap_draws") or 1000)
    if selected_draws < 1 or strongest_comparator not in FROZEN_HEAD_ARMS[1:]:
        raise ValueError("registered_settings_invalid")
    return {
        "expected_groups": len(evaluator.get("expected_source_ids") or []),
        "bootstrap_draws": selected_draws,
        "bootstrap_seed": EVALUATION_SEED,
        "candidate_arm": "proper_loss_monotone",
        "brier_comparators": ["raw_original", strongest_comparator],
        "strongest_comparator": strongest_comparator,
        "holm_family_alpha": 0.05,
        "brier_improvement_lower95": 0.0,
        "decision_cost_improvement_lower95": 0.0,
        "minimum_non_escalation_fraction": float(
            evaluator.get("minimum_non_escalation_fraction", 0.10)
        ),
        "resampling_unit": "source_component",
        "fresh_confirmatory_claim_allowed": False,
        "no_post_evaluation_fit_or_selection": True,
    }


def _validated_rows(
    rows: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]
) -> dict[str, dict[str, Mapping[str, Any]]]:
    """Reject incomplete or arithmetically changed source-arm rows."""

    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        source = str(row.get("source_component_id") or "")
        arm = str(row.get("arm") or "")
        if not source or arm not in ARMS or arm in grouped[source]:
            raise ValueError("row_identity_invalid")
        if any(
            (
                row.get("status") != "complete",
                row.get("complete") is not True,
                row.get("failed") is not False,
                row.get("excluded") is not False,
                row.get("censored") is not False,
            )
        ):
            raise ValueError("row_disposition_invalid")
        probability = float(row.get("q"))
        label = int(row.get("observed_error"))
        losses = loss_fields(probability, label)
        if not math.isclose(float(row.get("brier")), losses["brier"], abs_tol=1e-12):
            raise ValueError("row_brier_invalid")
        if not math.isclose(float(row.get("log_loss")), losses["log_loss"], abs_tol=1e-12):
            raise ValueError("row_log_loss_invalid")
        if row.get("raw_brier_denominator") != 1 or not math.isclose(
            float(row.get("raw_brier_numerator")), losses["brier"], abs_tol=1e-12
        ):
            raise ValueError("row_brier_arithmetic_invalid")
        if row.get("raw_log_loss_denominator") != 1 or not math.isclose(
            float(row.get("raw_log_loss_numerator")), losses["log_loss"], abs_tol=1e-12
        ):
            raise ValueError("row_log_loss_arithmetic_invalid")
        expected_decision = (
            {"action": "escalate", "realized_cost": 0.2}
            if arm == "escalate_all"
            else primary_decision(probability, label)
        )
        if row.get("action") != expected_decision["action"] or not math.isclose(
            float(row.get("realized_cost")),
            float(expected_decision["realized_cost"]),
            abs_tol=1e-12,
        ):
            raise ValueError("row_decision_invalid")
        if row.get("raw_cost_denominator") != 1 or not math.isclose(
            float(row.get("raw_cost_numerator")),
            float(expected_decision["realized_cost"]),
            abs_tol=1e-12,
        ):
            raise ValueError("row_cost_arithmetic_invalid")
        if row.get("metric_direction") != "lower_loss_and_cost_are_better":
            raise ValueError("row_direction_invalid")
        grouped[source][arm] = row
    if len(grouped) != int(settings["expected_groups"]):
        raise ValueError("expected_group_count_mismatch")
    if any(set(arm_rows) != set(ARMS) for arm_rows in grouped.values()):
        raise ValueError("arm_roster_invalid")
    if any(
        len({int(row["observed_error"]) for row in arm_rows.values()}) != 1
        for arm_rows in grouped.values()
    ):
        raise ValueError("group_label_disagreement")
    return grouped


def _paired_interval(improvements: Sequence[float], indices: np.ndarray) -> JsonDict:
    """Return both contrast signs from shared component bootstrap draws."""

    values = np.asarray(improvements, dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) < 1
        or indices.ndim != 2
        or indices.shape[1] != len(values)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("paired_interval_shape_invalid")
    means = values[indices].mean(axis=1)
    improvement = float(values.mean())
    lower = float(np.quantile(means, 0.025))
    upper = float(np.quantile(means, 0.975))
    return {
        "direction": "control_minus_candidate_positive_is_better",
        "positive_is_better_improvement": improvement,
        "candidate_minus_control_loss": -improvement,
        "improvement_ci95": [lower, upper],
        "candidate_minus_control_ci95": [-upper, -lower],
        "lower95": lower,
        "upper95": upper,
        "one_sided_p": float((1 + np.sum(means <= 0.0)) / (len(means) + 1)),
        "group_count": len(values),
    }


def _holm_intervals(
    comparisons: Mapping[str, Mapping[str, Any]], *, alpha: float
) -> dict[str, JsonDict]:
    """Apply the registered Holm step-down family to paired Brier rows."""

    ordered = sorted(comparisons, key=lambda name: (comparisons[name]["one_sided_p"], name))
    if not ordered:
        raise ValueError("holm_family_empty")
    output = {name: deepcopy(dict(row)) for name, row in comparisons.items()}
    running = 0.0
    for rank, name in enumerate(ordered, start=1):
        multiplier = len(ordered) - rank + 1
        adjusted = min(1.0, multiplier * float(comparisons[name]["one_sided_p"]))
        running = max(running, adjusted)
        output[name].update(
            {
                "holm_rank": rank,
                "holm_adjusted_p": running,
                "holm_alpha": alpha,
                "holm_passed": running < alpha and float(comparisons[name]["lower95"]) > 0.0,
            }
        )
    return output


def reduce_rows(rows: Sequence[Mapping[str, Any]], *, settings: Mapping[str, Any]) -> JsonDict:
    """Reduce complete source components into registered exploratory gates."""

    grouped = _validated_rows(rows, settings)
    sources = sorted(grouped)
    candidate = str(settings["candidate_arm"])
    comparators = [str(value) for value in settings["brier_comparators"]]
    rng = np.random.default_rng(int(settings["bootstrap_seed"]))
    indices = rng.integers(0, len(sources), size=(int(settings["bootstrap_draws"]), len(sources)))
    brier_raw = {
        comparator: _paired_interval(
            [
                float(grouped[source][comparator]["brier"])
                - float(grouped[source][candidate]["brier"])
                for source in sources
            ],
            indices,
        )
        for comparator in comparators
    }
    brier = _holm_intervals(brier_raw, alpha=float(settings["holm_family_alpha"]))
    strongest = str(settings["strongest_comparator"])
    decision = _paired_interval(
        [
            float(grouped[source][strongest]["realized_cost"])
            - float(grouped[source][candidate]["realized_cost"])
            for source in sources
        ],
        indices,
    )
    decision["comparator"] = strongest
    coverage_values = np.asarray(
        [float(bool(grouped[source][candidate]["non_escalated"])) for source in sources],
        dtype=np.float64,
    )
    coverage_means = coverage_values[indices].mean(axis=1)
    coverage = {
        "arm": candidate,
        "fraction": float(coverage_values.mean()),
        "lower95": float(np.quantile(coverage_means, 0.025)),
        "upper95": float(np.quantile(coverage_means, 0.975)),
        "floor": float(settings["minimum_non_escalation_fraction"]),
        "passed": float(np.quantile(coverage_means, 0.025))
        >= float(settings["minimum_non_escalation_fraction"]),
    }
    metrics: dict[str, JsonDict] = {}
    for arm in ARMS:
        arm_rows = [grouped[source][arm] for source in sources]
        brier_numerator = float(sum(float(row["raw_brier_numerator"]) for row in arm_rows))
        log_numerator = float(sum(float(row["raw_log_loss_numerator"]) for row in arm_rows))
        cost_numerator = float(sum(float(row["raw_cost_numerator"]) for row in arm_rows))
        metrics[arm] = {
            "raw_brier_numerator": brier_numerator,
            "raw_brier_denominator": len(arm_rows),
            "mean_brier": brier_numerator / len(arm_rows),
            "raw_log_loss_numerator": log_numerator,
            "raw_log_loss_denominator": len(arm_rows),
            "mean_log_loss": log_numerator / len(arm_rows),
            "raw_cost_numerator": cost_numerator,
            "raw_cost_denominator": len(arm_rows),
            "mean_primary_cost": cost_numerator / len(arm_rows),
            "non_escalation_fraction": float(
                np.mean([bool(row["non_escalated"]) for row in arm_rows])
            ),
            "n_source_components": len(arm_rows),
        }
    probability_passed = all(row["holm_passed"] is True for row in brier.values())
    decision_cost_passed = float(decision["lower95"]) > float(
        settings["decision_cost_improvement_lower95"]
    )
    decision_passed = decision_cost_passed and coverage["passed"] is True
    return {
        "static_measurement_complete_score": 1,
        "exploratory_probability_benefit_score": int(probability_passed),
        "exploratory_decision_benefit_score": int(decision_passed),
        "probability_metrics": metrics,
        "paired_intervals": {
            "unit_definition": str(settings["resampling_unit"]),
            "paired_rows": len(sources),
            "draws": int(settings["bootstrap_draws"]),
            "seed": int(settings["bootstrap_seed"]),
            "correction_method": "Holm one-sided step-down across registered Brier contrasts",
            "brier": brier,
            "decision_cost": decision,
        },
        "coverage_interval": coverage,
        "probability_gate_passed": probability_passed,
        "decision_cost_gate_passed": decision_cost_passed,
        "decision_coverage_gate_passed": coverage["passed"],
        "sample_size_budget": {
            "independent_unit": str(settings["resampling_unit"]),
            "planned": int(settings["expected_groups"]),
            "attempted": len(sources),
            "completed": len(sources),
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": max(0, int(settings["expected_groups"]) - len(sources)),
            "arms_are_independent_units": False,
        },
    }


def run_evaluator_controls(
    features: Sequence[Mapping[str, Any]], evaluator: Mapping[str, Any]
) -> JsonDict:
    """Run informative and corrupt fixtures through the production evaluator."""

    label_by_source = {
        str(row["source_id"]): int(row["observed_error"]) for row in evaluator.get("labels") or []
    }
    forecasts: list[JsonDict] = []
    for feature in features:
        source_id = str(feature["source_id"])
        label = label_by_source[source_id]
        for arm, probability in (
            ("informative_forecast", 0.99 if label else 0.01),
            ("uninformative_control", 0.5),
        ):
            forecasts.append(
                {
                    "source_id": source_id,
                    "group_id": str(feature["group_id"]),
                    "arm": arm,
                    "q": probability,
                    "option_mapping_sha256": feature["option_mapping_sha256"],
                    "head_sha256": f"oracle_fixture:{arm}",
                    "feature_provenance": "oracle_defined_positive_control",
                }
            )
    evaluated = evaluate_forecasts(
        forecasts,
        evaluator,
        expected_arms=("informative_forecast", "uninformative_control"),
        include_escalate_all=False,
    )
    by_arm: dict[str, list[JsonDict]] = defaultdict(list)
    for row in evaluated:
        by_arm[str(row["arm"])].append(row)
    informative = float(np.mean([row["brier"] for row in by_arm["informative_forecast"]]))
    control = float(np.mean([row["brier"] for row in by_arm["uninformative_control"]]))

    def rejected(mutator: Any) -> JsonDict:
        try:
            mutator()
        except ValueError as error:
            return {"rejected": True, "error": str(error)}
        return {"rejected": False, "error": None}

    swapped = deepcopy(dict(evaluator))
    swapped["labels"] = list(reversed(swapped["labels"]))
    remapped = deepcopy(forecasts)
    remapped[0]["option_mapping_sha256"] = "sha256:changed"
    duplicated = deepcopy(forecasts)
    duplicated[-1] = deepcopy(duplicated[0])
    controls = {
        "positive_control": {
            "verdict_class": "circular_positive" if control - informative > 0.0 else "null",
            "row_contrast_supported": control - informative > 0.0,
            "positive_is_better_improvement": control - informative,
            "candidate_minus_control_loss": informative - control,
            "verifier_is_oracle": True,
            "rows": evaluated,
        },
        "swapped_labels": rejected(
            lambda: evaluate_forecasts(
                forecasts,
                swapped,
                expected_arms=("informative_forecast", "uninformative_control"),
                include_escalate_all=False,
            )
        ),
        "changed_option_mapping": rejected(
            lambda: evaluate_forecasts(
                remapped,
                evaluator,
                expected_arms=("informative_forecast", "uninformative_control"),
                include_escalate_all=False,
            )
        ),
        "duplicate_component": rejected(
            lambda: evaluate_forecasts(
                duplicated,
                evaluator,
                expected_arms=("informative_forecast", "uninformative_control"),
                include_escalate_all=False,
            )
        ),
    }
    controls["passed"] = bool(
        controls["positive_control"]["row_contrast_supported"]
        and controls["swapped_labels"]["rejected"]
        and controls["changed_option_mapping"]["rejected"]
        and controls["duplicate_component"]["rejected"]
    )
    controls["controls_sha256"] = canonical_hash(
        {key: value for key, value in controls.items() if key != "controls_sha256"}
    )
    return controls


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    failures = [deepcopy(dict(row)) for row in checks if row.get("passed") is not True]
    return {
        "passed": not failures,
        "failed_count": len(failures),
        "failed_checks": [str(row.get("check")) for row in failures],
        "first_failure": failures[0] if failures else None,
    }


def _field_principles() -> JsonDict:
    """Carry one-line claim guards beside every required terminal field."""

    return {
        "honest_verdict": "Use a complete_ terminal prefix; completion does not establish benefit.",
        "verdict_class": "Use one closed class; partial means only unfinished task-owned work.",
        "flagged_adversarial": "Persist the exact terminal outcome; flagged evidence cannot open readiness.",
        "gate_check_summary": "Blocked work names check, upstream, path, field, operator, expected, and observed values.",
        "acceptance_gate_results": "Validity, readiness, probability benefit, and action benefit remain separate.",
        "rows": "Each source-arm row keeps raw arithmetic, direction, seed, censoring, and provenance.",
        "inference_substrate_class": "Actual and planned substrates stay separate; zero model calls forbid live claims.",
        "MODEL_SPECS": "Cached-only work has no current model; historical identity remains source custody.",
        "invocation_counts": "Current loads, forwards, generations, and tokens are counted independently.",
        "duration_s": "Monotonic current work excludes inherited timing and artificial sleeps.",
        "source_artifact_hashes": "Every conclusion binds exact source bytes and missing producers fail pre-gate.",
        "validation_receipts": "Each bounded check binds its command, worktree, exit, and log hash.",
        "field_principles": "Interpretation guards travel with the emitted fields.",
        "verifier_is_oracle": "Oracle-defined or label-accessing controls cannot support an oracle-distinct claim.",
        "static_measurement_complete_score": "One marks a valid comparison even when benefit is absent.",
        "exploratory_probability_benefit_score": "Only registered paired Brier gates set this descriptive score.",
        "exploratory_decision_benefit_score": "Cost improvement and non-escalation support must pass independently.",
        "fresh_confirmatory_claim_allowed": "False because prior source labels and outcomes were accessed.",
        "paired_intervals": "Intervals bind source units, paired rows, draws, seed, and correction method.",
    }


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Hash stable terminal content while excluding only the self-reference."""

    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key != "reproducibility_checksum"}
    )


def _terminal_base(duration_s: float) -> JsonDict:
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "no_model_load": True,
        "MODEL_SPECS": [],
        "model_specs": [],
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_id": HISTORICAL_MODEL_ID,
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "inference_substrate": "deterministic_cached_proper_loss_evaluator_no_llm",
        "planned_inference_substrate": "cached_qwen_probability_evaluation_without_current_inference",
        "duration_s": float(duration_s),
        "random_seed": EVALUATION_SEED,
        "verifier_is_oracle": True,
        "fresh_confirmatory_claim_allowed": False,
        "no_deployment_promotion": True,
        "submitted_externally": False,
        "field_principles": _field_principles(),
    }


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
) -> JsonDict:
    """Represent an external prerequisite failure without substitute rows."""

    summary = _gate_summary(checks)
    first = summary["first_failure"] or {"check": "unknown_precondition"}
    artifact: JsonDict = {
        **_terminal_base(duration_s),
        "honest_verdict": f"complete_blocked_{first['check']}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": summary,
        "precondition_checks": [deepcopy(dict(row)) for row in checks],
        "acceptance_gate_results": {
            "validity": {"passed": False, "reason": "external_precondition_failed"},
            "readiness": {"passed": False, "reason": "measurement_not_started"},
            "probability_benefit": {"passed": False, "reason": "measurement_not_started"},
            "decision_benefit": {"passed": False, "reason": "measurement_not_started"},
        },
        "rows": [],
        "probability_metrics": {},
        "paired_intervals": {},
        "coverage_interval": {},
        "evaluator_controls": {},
        "head_identity": {},
        "registered_settings": {},
        "static_measurement_complete_score": 0,
        "exploratory_probability_benefit_score": 0,
        "exploratory_decision_benefit_score": 0,
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "validation_receipts": [],
        "raw_sidecars": {},
        "capability_e2e": {
            "learning_lifecycle": [],
            "numbered_runtime_e2e": [],
            "reason": "measurement blocked before capability exercise",
        },
    }
    artifact["inference_substrate"] = "blocked_no_run"
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute every comparative result from retained per-source rows."""

    return reduce_rows(
        artifact.get("rows") or [],
        settings=artifact.get("registered_settings") or {},
    )


def build_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
    controls: Mapping[str, Any],
    head_identity: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    raw_sidecars: Mapping[str, Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    """Assemble a completed descriptive comparison from independently reducible rows."""

    reduction = reduce_rows(rows, settings=settings)
    controls_passed = controls.get("passed") is True
    probability_score = int(reduction["exploratory_probability_benefit_score"])
    decision_score = int(reduction["exploratory_decision_benefit_score"])
    ready = controls_passed and head_identity.get("head_count") == len(FROZEN_HEAD_ARMS)
    if not ready:
        verdict = "disqualified"
        honest = "complete_disqualified_evaluator_control_or_head_custody_failed"
    elif probability_score or decision_score:
        verdict = "positive"
        honest = "complete_positive_exploratory_exposed_rows_no_fresh_claim"
    else:
        verdict = "null"
        honest = "complete_null_frozen_proper_loss_evaluation_no_supported_benefit"
    artifact: JsonDict = {
        **_terminal_base(duration_s),
        "honest_verdict": honest,
        "verdict_class": verdict,
        "flagged_adversarial": False,
        "gate_check_summary": _gate_summary(preconditions),
        "precondition_checks": [deepcopy(dict(row)) for row in preconditions],
        "rows": [deepcopy(dict(row)) for row in rows],
        "registered_settings": deepcopy(dict(settings)),
        "head_identity": deepcopy(dict(head_identity)),
        "evaluator_controls": deepcopy(dict(controls)),
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "raw_sidecars": deepcopy(dict(raw_sidecars)),
        "static_measurement_complete_score": reduction["static_measurement_complete_score"],
        "exploratory_probability_benefit_score": probability_score,
        "exploratory_decision_benefit_score": decision_score,
        "probability_metrics": reduction["probability_metrics"],
        "paired_intervals": reduction["paired_intervals"],
        "coverage_interval": reduction["coverage_interval"],
        "sample_size_budget": reduction["sample_size_budget"],
        "signed_effects": {
            "brier": deepcopy(reduction["paired_intervals"]["brier"]),
            "decision_cost": deepcopy(reduction["paired_intervals"]["decision_cost"]),
        },
        "acceptance_gate_results": {
            "validity": {
                "passed": _gate_summary(preconditions)["passed"] and controls_passed,
                "checks": "authenticated sources plus positive and corruption controls",
            },
            "readiness": {
                "passed": ready,
                "checks": "all frozen heads share one complete 80-group roster",
            },
            "probability_benefit": {
                "passed": bool(probability_score),
                "checks": "registered paired Brier gates with Holm correction",
            },
            "decision_benefit": {
                "passed": bool(decision_score),
                "cost_improvement_passed": reduction["decision_cost_gate_passed"],
                "coverage_support_passed": reduction["decision_coverage_gate_passed"],
            },
        },
        "prior_verdict_disposition": {
            "prior_experiment": "exp7567-source-evaluation",
            "prior_literal_verdict": "complete_null_source_evaluation_no_supported_benefit",
            "retired_for_repetition": verdict == "null",
            "retirement_scope": "only the repeated literal verdict for this frozen exposed-row construction",
            "scientific_hypothesis_retired": False,
        },
        "capability_e2e": {
            "learning_lifecycle": ["predict", "release", "update", "persist", "reload"],
            "lifecycle_scope": "authenticated upstream protocol exercise; this task performs no update",
            "numbered_runtime_e2e": [],
            "numbered_runtime_e2e_reason": "read-only reporting changes no ARC, binding, or shared sampler runtime",
            "fresh_process_reader_names": [
                str(row.get("name"))
                for row in validation_receipts
                if row.get("name") in TERMINAL_CHECK_NAMES
            ],
        },
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _validate_receipts(receipts: Sequence[Mapping[str, Any]]) -> None:
    """Require one successful receipt for every affected and terminal check."""

    by_name = {str(row.get("name")): row for row in receipts}
    required = (*REQUIRED_VALIDATION_NAMES, *TERMINAL_CHECK_NAMES)
    missing = [name for name in required if name not in by_name]
    if missing:
        raise ValueError(f"validation_receipt_missing:{missing}")
    for name in required:
        row = by_name[name]
        if row.get("passed") is not True or row.get("exit_code") != 0 or row.get("timed_out"):
            raise ValueError(f"validation_receipt_failed:{name}")
        if not row.get("command") or not row.get("worktree") or not row.get("log_sha256"):
            raise ValueError(f"validation_receipt_incomplete:{name}")


def validate_artifact(
    value: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_validation: bool = True,
) -> JsonDict:
    """Cold-check identity, source rows, reductions, claim boundaries, and receipts."""

    artifact = deepcopy(dict(value))
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("artifact_identity_invalid")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        raise ValueError("artifact_checksum_invalid")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_specs") != []:
        raise ValueError("model_specs_not_empty")
    if (
        artifact.get("no_model_load") is not True
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        raise ValueError("current_model_activity_invalid")
    if (
        artifact.get("fresh_confirmatory_claim_allowed") is not False
        or artifact.get("no_deployment_promotion") is not True
    ):
        raise ValueError("claim_boundary_invalid")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        raise ValueError("verdict_class_invalid")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        raise ValueError("terminal_prefix_invalid")
    if set(REQUIRED_PRINCIPLE_FIELDS) - set(artifact.get("field_principles") or {}):
        raise ValueError("field_principles_incomplete")
    if artifact.get("verdict_class") == "blocked":
        first = artifact.get("gate_check_summary", {}).get("first_failure") or {}
        required = {"check", "upstream", "path", "field", "op", "expected", "observed"}
        if artifact.get("rows") != [] or artifact.get("static_measurement_complete_score") != 0:
            raise ValueError("blocked_measurement_invalid")
        if not required <= set(first):
            raise ValueError("blocked_gate_summary_invalid")
        return {"valid": True, "blocked": True}
    reduced = independent_reduce(artifact)
    if reduced["probability_metrics"] != artifact.get("probability_metrics"):
        raise ValueError("probability_metrics_mismatch")
    if reduced["paired_intervals"] != artifact.get("paired_intervals"):
        raise ValueError("paired_intervals_mismatch")
    if reduced["coverage_interval"] != artifact.get("coverage_interval"):
        raise ValueError("coverage_interval_mismatch")
    if reduced["exploratory_probability_benefit_score"] != artifact.get(
        "exploratory_probability_benefit_score"
    ):
        raise ValueError("probability_benefit_mismatch")
    if reduced["exploratory_decision_benefit_score"] != artifact.get(
        "exploratory_decision_benefit_score"
    ):
        raise ValueError("decision_benefit_mismatch")
    controls = artifact.get("evaluator_controls") or {}
    control_hash = canonical_hash(
        {key: value for key, value in controls.items() if key != "controls_sha256"}
    )
    if controls.get("controls_sha256") != control_hash or controls.get("passed") is not True:
        raise ValueError("evaluator_controls_invalid")
    ready = artifact.get("head_identity", {}).get("head_count") == len(FROZEN_HEAD_ARMS)
    if artifact.get("static_measurement_complete_score") != 1 or not ready:
        raise ValueError("static_measurement_incomplete")
    expected_class = (
        "positive"
        if reduced["exploratory_probability_benefit_score"]
        or reduced["exploratory_decision_benefit_score"]
        else "null"
    )
    if artifact.get("verdict_class") != expected_class:
        raise ValueError("verdict_reduction_mismatch")
    gates = artifact.get("acceptance_gate_results") or {}
    if (
        gates.get("validity", {}).get("passed") is not True
        or gates.get("readiness", {}).get("passed") is not True
        or gates.get("probability_benefit", {}).get("passed")
        is not bool(reduced["exploratory_probability_benefit_score"])
        or gates.get("decision_benefit", {}).get("passed")
        is not bool(reduced["exploratory_decision_benefit_score"])
    ):
        raise ValueError("acceptance_gate_mismatch")
    if artifact.get("flagged_adversarial") is not False:
        raise ValueError("flagged_artifact_not_usable")
    if require_validation:
        for receipt in artifact.get("source_artifact_hashes") or []:
            path = _resolved(root.resolve(), str(receipt.get("path") or ""))
            if not path.is_file() or sha256_file(path) != receipt.get("sha256"):
                raise ValueError(f"source_hash_invalid:{receipt.get('path')}")
        for name, receipt in (artifact.get("raw_sidecars") or {}).items():
            path = _resolved(root.resolve(), str(receipt.get("path") or ""))
            if not path.is_file() or sha256_file(path) != receipt.get("sha256"):
                raise ValueError(f"sidecar_hash_invalid:{name}")
        _validate_receipts(artifact.get("validation_receipts") or [])
    return {"valid": True, "blocked": False, **reduced}


def cold_replay(path: Path, *, root: Path = REPO_ROOT) -> JsonDict:
    """Validate exact serialized bytes through the public reader path."""

    return validate_artifact(_load_object(path), root=root)


def independent_reduce_artifact(path: Path, *, root: Path = REPO_ROOT) -> JsonDict:
    """Recompute all comparative rows after full serialized-evidence validation."""

    artifact = _load_object(path)
    validate_artifact(artifact, root=root)
    return independent_reduce(artifact)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], root: Path) -> JsonDict:
    """Persist raw rows atomically so the terminal artifact can bind exact bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
    temporary.replace(path)
    return {**_source_hash(path, root), "rows": len(rows)}


def _load_evaluation_inputs(root: Path) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Load authenticated sidecars after collect_preconditions closes every gate."""

    protocol_artifact = _load_object(root / PROTOCOL_ARTIFACT_PATH)
    fit_artifact = _load_object(root / FIT_ARTIFACT_PATH)
    cached_path = _resolved(root, str(protocol_artifact["raw_sidecars"]["cached_roles"]["path"]))
    protocol_path = _resolved(
        root, str(protocol_artifact["raw_sidecars"]["frozen_protocol"]["path"])
    )
    head_path = _resolved(root, str(fit_artifact["raw_sidecars"]["head_manifest"]["path"]))
    rows: list[JsonDict] = []
    with cached_path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if isinstance(value, dict) and value.get("role") == "test":
                rows.append(value)
    return rows, _load_object(protocol_path), _load_object(head_path)


def progress(  # pragma: no cover - production liveness output.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Emit one flushed boundary with truthful monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7577] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _write_affected_manifest(root: Path) -> JsonDict:  # pragma: no cover
    """Freeze the exact affected-file scope before command expansion."""

    files = (
        *AFFECTED_MANIFEST.test_paths,
        *AFFECTED_MANIFEST.changed_modules,
        *AFFECTED_MANIFEST.static_paths,
    )
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "worktree": str(root.resolve()),
        "test_paths": list(AFFECTED_MANIFEST.test_paths),
        "changed_modules": list(AFFECTED_MANIFEST.changed_modules),
        "static_paths": list(AFFECTED_MANIFEST.static_paths),
        "file_hashes": {path: sha256_file(root / path) for path in files},
    }
    atomic_json(root / AFFECTED_MANIFEST_PATH, manifest)
    return manifest


def _prepare_private_parents(command: validation_scope.CommandSpec) -> None:  # pragma: no cover
    """Create private pytest and coverage parents immediately before a child."""

    for argument in command.argv:
        if argument.startswith("--basetemp=") or argument.startswith("--data-file="):
            Path(argument.split("=", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
    environment = dict(getattr(command, "command_environment", ()))
    if coverage_file := environment.get("COVERAGE_FILE"):
        Path(coverage_file).parent.mkdir(parents=True, exist_ok=True)


def _run_planned(  # pragma: no cover - bounded capability subprocesses.
    root: Path,
    commands: Sequence[PlannedCommand],
    *,
    log_dir: Path,
) -> list[JsonDict]:
    """Run one owned process at a time with fresh private parents and heartbeats."""

    receipts: list[JsonDict] = []
    for index, planned in enumerate(commands):
        _prepare_private_parents(planned.spec)
        rows = run_categorized_commands(
            root,
            [planned],
            log_dir=log_dir / f"{index:02d}_{planned.spec.name}",
            heartbeat_s=60.0,
        )
        rows[0]["worktree"] = str(root.resolve())
        receipts.extend(rows)
    return receipts


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    """Build exact cold replay, reduction, adversarial, and strict readers."""

    python = ".venv/bin/python"
    commands = (
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--root",
                ".",
                "--date",
                RUN_DATE,
                "--cold-replay",
                candidate.as_posix(),
            ),
            "candidate_capability_e2e",
        ),
        validation_scope.CommandSpec(
            "independent_row_reduction",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--root",
                ".",
                "--date",
                RUN_DATE,
                "--independent-reduce",
                candidate.as_posix(),
            ),
            "candidate_raw_reduction",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", candidate.as_posix()),
            "candidate_safety",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                candidate.as_posix(),
            ),
            "candidate_row_consistency",
        ),
    )
    return [PlannedCommand(command, "required_terminal_validation", True) for command in commands]


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover - capability E2E.
    """Authenticate, evaluate, validate, and publish the exact frozen candidate."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    started = time.monotonic()
    progress(started, "startup", "flushed_progress", root=root)
    progress(started, "preconditions", "before_authentication")
    checks, source_hashes = collect_preconditions(root)
    failed = next((row for row in checks if row.get("passed") is not True), None)
    if failed is not None:
        blocked = build_blocked_artifact(
            checks, source_hashes, duration_s=time.monotonic() - started
        )
        progress(started, "publish", "before_atomic_blocked", check=failed["check"])
        atomic_json(root / RESULT_PATH, blocked)
        progress(started, "publish", "after_atomic_blocked", check=failed["check"])
        return blocked
    progress(started, "preconditions", "after_authentication", completed_units=len(checks))

    progress(started, "validation_manifest", "before_freeze")
    _write_affected_manifest(root)
    progress(started, "validation_manifest", "after_freeze")
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7577-validation-", dir="/tmp"))
    commands = build_command_plan(root, AFFECTED_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, AFFECTED_MANIFEST, commands)
    if plan_errors:
        raise RuntimeError(f"validation_plan_invalid:{plan_errors}")
    progress(started, "affected_validation", "before_subprocesses", commands=len(commands))
    affected = _run_planned(
        root,
        [PlannedCommand(command, "required_affected_validation", True) for command in commands],
        log_dir=root / RAW_DIR / "validation" / "affected",
    )
    affected_reduction = reduce_affected_receipts(root, AFFECTED_MANIFEST, affected)
    progress(
        started,
        "affected_validation",
        "after_subprocesses",
        completed_units=len(affected),
        passed=affected_reduction["passed"],
    )
    if affected_reduction["passed"] is not True:
        raise RuntimeError(f"affected_validation_failed:{affected_reduction}")

    progress(started, "model_load", "before_no_model_load_declaration")
    progress(started, "model_load", "after_no_model_load_declaration", model_calls=0)
    progress(started, "evaluation", "before_benchmark", expected_groups=EXPECTED_GROUPS)
    role_rows, protocol, manifest = _load_evaluation_inputs(root)
    features, evaluator = split_evaluation_rows(role_rows, protocol)
    head_identity = validate_head_manifest(manifest)
    settings = registered_settings(
        evaluator, strongest_comparator=str(head_identity["strongest_comparator"])
    )
    forecasts = score_frozen_heads(features, manifest)
    rows = evaluate_forecasts(forecasts, evaluator)
    controls = run_evaluator_controls(features, evaluator)
    reduce_rows(rows, settings=settings)
    progress(started, "evaluation", "after_benchmark", completed_units=len(features))

    progress(started, "persist", "before_checkpoint")
    row_receipt = _write_jsonl(root / EVALUATION_ROWS_PATH, rows, root)
    manifest_receipt = _source_hash(root / AFFECTED_MANIFEST_PATH, root)
    raw_sidecars = {
        "evaluation_rows": row_receipt,
        "affected_validation_manifest": manifest_receipt,
    }
    progress(started, "persist", "after_checkpoint", completed_units=len(rows))
    provisional = build_artifact(
        rows=rows,
        settings=settings,
        controls=controls,
        head_identity=head_identity,
        preconditions=checks,
        source_hashes=source_hashes,
        validation_receipts=affected,
        raw_sidecars=raw_sidecars,
        duration_s=time.monotonic() - started,
    )
    validate_artifact(provisional, root=root, require_validation=False)
    atomic_json(root / TERMINAL_CANDIDATE_PATH, provisional)
    progress(started, "terminal_validation", "before_subprocesses", commands=4)
    terminal = _run_planned(
        root,
        _terminal_commands(TERMINAL_CANDIDATE_PATH),
        log_dir=root / RAW_DIR / "validation" / "terminal",
    )
    if not all(row.get("passed") is True for row in terminal):
        raise RuntimeError("terminal_validation_failed")
    final = build_artifact(
        rows=rows,
        settings=settings,
        controls=controls,
        head_identity=head_identity,
        preconditions=checks,
        source_hashes=source_hashes,
        validation_receipts=[*affected, *terminal],
        raw_sidecars=raw_sidecars,
        duration_s=time.monotonic() - started,
    )
    validate_artifact(final, root=root)
    atomic_json(root / TERMINAL_CANDIDATE_PATH, final)
    progress(started, "exact_terminal_validation", "before_subprocesses", commands=4)
    exact = _run_planned(
        root,
        _terminal_commands(TERMINAL_CANDIDATE_PATH),
        log_dir=root / RAW_DIR / "validation" / "exact_terminal",
    )
    if not all(row.get("passed") is True for row in exact):
        raise RuntimeError("exact_terminal_validation_failed")
    progress(started, "publish", "before_atomic_terminal", verdict=final["verdict_class"])
    atomic_json(root / RESULT_PATH, final)
    validate_artifact(_load_object(root / RESULT_PATH), root=root)
    progress(started, "publish", "after_atomic_terminal", path=RESULT_PATH)
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed producer and two read-only fresh-process modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default=RUN_DATE)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--cold-replay", type=Path)
    modes.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    """Run the producer or an exact bounded candidate reader."""

    args = parse_args(argv)
    root = args.root.resolve()
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    if args.cold_replay is not None:
        path = _resolved(root, str(args.cold_replay))
        result = validate_artifact(_load_object(path), root=root, require_validation=False)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    if args.independent_reduce is not None:
        path = _resolved(root, str(args.independent_reduce))
        artifact = _load_object(path)
        validate_artifact(artifact, root=root, require_validation=False)
        print(json.dumps(independent_reduce(artifact), sort_keys=True), flush=True)
        return 0
    run_experiment(root, args.date)
    return 0
