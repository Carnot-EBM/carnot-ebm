"""Capture sealed official-test and online option-logit features.

The capture loop stays label blind. Evaluator labels open only after prediction
bytes freeze, and collection never counts as predictive benefit.

Spec refs: REQ-VERIFY-7565 and SCENARIO-VERIFY-7565-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import experiment_7535_v659_native_pilot as native
from carnot import experiment_7533_v659_tool_protocol as protocol
from carnot import experiment_7548_v660_capture_runner as capture
from carnot import experiment_7563_v661_native_pilot as pilot
from carnot import gpu_lease_phase_journal as lease_api
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.inference.sota_models import cached_current_model
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json


JsonDict = dict[str, Any]
RUN_DATE = "20260923"
MILESTONE = "2026.09.661"
VERSION = "v661"
EXPERIMENT_ID = "exp7565-v661-test-online-capture"
SCHEMA = "carnot.exp7565.v661.test_online_capture.v1"
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_SPECS = [MODEL_ID]
ROLE_COUNTS = {"test": 80, "online": 160}
ROLES = tuple(ROLE_COUNTS)
CAPTURE_GROUPS = 240
CAPTURE_FORWARDS = 1440
COLLECTION_LIMIT_S = 3000.0
VALIDATION_RESERVE_S = 900.0
HARD_LIMIT_S = COLLECTION_LIMIT_S + VALIDATION_RESERVE_S
CONDITIONS = capture.CONDITIONS
OPTION_ORDERS = capture.OPTION_ORDERS

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7565_v661_test_online_capture.json")
RAW_DIR = Path("results/raw/experiment_7565_v661_test_online_capture")
CANDIDATE_PATH = RAW_DIR / "measured_terminal_candidate.json"
CHECKPOINT_PATH = RAW_DIR / "checkpoint.json"
MODULE_PATH = Path("python/carnot/experiment_7565_v661_test_online_capture.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7565_v661_test_online_capture.py")
TEST_PATH = Path("tests/python/test_experiment_7565_v661_test_online_capture.py")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
PILOT_PATH = Path("results/experiment_7563_v661_native_pilot.json")
PROTOCOL_PATH = Path("results/experiment_7533_v659_tool_protocol.json")
RUNNER_PATH = Path("results/experiment_7548_v660_capture_runner.json")
PILOT_MODULE_PATH = Path("python/carnot/experiment_7563_v661_native_pilot.py")
PILOT_WRAPPER_PATH = Path("scripts/experiments/experiment_7563_v661_native_pilot.py")
EXPOSURE_PATH = Path("results/raw/experiment_7517_v658_source_protocol/exposure_inventory.json")

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

CaptureError = capture.CaptureRunnerError
EvaluationCaptureError = CaptureError
canonical_hash = capture.canonical_hash
sha256_file = capture.sha256_file
gate = capture.gate
gate_check_summary = capture.gate_check_summary


def reduce_invocation_events(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count current calls without converting zero generations into no-model work."""

    return pilot.reduce_invocation_events(events)


def _contract_gate(
    check: str,
    expected: object,
    observed: object,
    *,
    upstream: str,
    field: str,
    path: str | None = None,
    op: str = "==",
) -> JsonDict:
    passed = observed == expected if op == "==" else observed in expected
    return gate(
        check,
        "external_precondition",
        expected,
        observed,
        op,
        passed,
        "Changed or missing upstream evidence must stop dependent model work.",
        upstream=upstream,
        field=field,
        path=path,
    )


def authenticate_pilot(
    artifact: Mapping[str, Any], current_hashes: Mapping[str, str]
) -> tuple[list[JsonDict], JsonDict]:
    """Authenticate the current pilot and every source byte that it delegated."""

    checks = [
        _contract_gate(
            "pilot_identity",
            "exp7563-v661-native-pilot",
            artifact.get("experiment_id"),
            upstream="exp7563-v661-native-pilot",
            field="experiment_id",
            path=PILOT_PATH.as_posix(),
        ),
        _contract_gate(
            "pilot_native_tool_ready",
            1,
            artifact.get("native_tool_ready_score"),
            upstream="exp7563-v661-native-pilot",
            field="native_tool_ready_score",
            path=PILOT_PATH.as_posix(),
        ),
        _contract_gate(
            "pilot_eval_capture_feasible",
            1,
            artifact.get("eval_capture_feasible_score"),
            upstream="exp7563-v661-native-pilot",
            field="eval_capture_feasible_score",
            path=PILOT_PATH.as_posix(),
        ),
        _contract_gate(
            "pilot_verdict_class",
            ("null", "positive"),
            artifact.get("verdict_class"),
            upstream="exp7563-v661-native-pilot",
            field="verdict_class",
            path=PILOT_PATH.as_posix(),
            op="in",
        ),
        _contract_gate(
            "pilot_adversarial_state",
            False,
            artifact.get("flagged_adversarial"),
            upstream="exp7563-v661-native-pilot",
            field="flagged_adversarial",
            path=PILOT_PATH.as_posix(),
        ),
    ]
    upstream_hashes = artifact.get("source_artifact_hashes")
    bound = dict(upstream_hashes) if isinstance(upstream_hashes, Mapping) else {}
    expected_bound = {
        path.as_posix(): bound.get(path.as_posix())
        for path in (PROTOCOL_PATH, RUNNER_PATH, PILOT_MODULE_PATH, PILOT_WRAPPER_PATH)
    }
    observed_bound = {path: current_hashes.get(path) for path in expected_bound}
    checks.append(
        _contract_gate(
            "pilot_bound_sources",
            expected_bound,
            observed_bound,
            upstream="exp7563-v661-native-pilot",
            field="source_artifact_hashes.bound_sources",
            path=PILOT_PATH.as_posix(),
        )
    )
    model_specs = artifact.get("model_specs")
    resolved = list(model_specs) if isinstance(model_specs, list) else []
    model_sha256 = (
        resolved[0].get("model_sha256") if resolved and isinstance(resolved[0], Mapping) else None
    )
    checks.extend(
        (
            _contract_gate(
                "pilot_model_plan",
                MODEL_SPECS,
                artifact.get("MODEL_SPECS"),
                upstream="exp7563-v661-native-pilot",
                field="MODEL_SPECS",
                path=PILOT_PATH.as_posix(),
            ),
            _contract_gate(
                "pilot_model_hash",
                True,
                isinstance(model_sha256, str) and model_sha256.startswith("sha256:"),
                upstream="exp7563-v661-native-pilot",
                field="model_specs[0].model_sha256",
                path=PILOT_PATH.as_posix(),
            ),
        )
    )
    role_manifest = artifact.get("role_manifest")
    role_map = dict(role_manifest) if isinstance(role_manifest, Mapping) else {}
    sealed = role_map.get("sealed_capture")
    sealed_map = dict(sealed) if isinstance(sealed, Mapping) else {}
    evaluation = sealed_map.get("evaluation")
    evaluation_map = dict(evaluation) if isinstance(evaluation, Mapping) else {}
    role_contract = {
        "roles": list(ROLES),
        "group_count": CAPTURE_GROUPS,
        "forward_count": CAPTURE_FORWARDS,
    }
    observed_role_contract = {
        "roles": evaluation_map.get("roles"),
        "group_count": evaluation_map.get("group_count"),
        "forward_count": evaluation_map.get("forward_count"),
    }
    checks.append(
        _contract_gate(
            "pilot_evaluation_role_contract",
            role_contract,
            observed_role_contract,
            upstream="exp7563-v661-native-pilot",
            field="role_manifest.sealed_capture.evaluation",
            path=PILOT_PATH.as_posix(),
        )
    )
    return checks, {
        "model_sha256": model_sha256,
        "role_counts": deepcopy(ROLE_COUNTS),
        "role_schedule_hashes": deepcopy(evaluation_map.get("role_schedule_hashes")),
        "source_label_access": [],
        "bound_source_hashes": expected_bound,
    }


def compile_evaluation_schedule(
    groups: Sequence[Mapping[str, Any]],
    *,
    expected_role_counts: Mapping[str, int] = ROLE_COUNTS,
) -> list[JsonDict]:
    """Select only the two sealed evaluation roles without rebuilding requests."""

    observed = dict(Counter(str(group.get("role")) for group in groups))
    if observed != dict(expected_role_counts):
        raise CaptureError(f"evaluation_role_counts:{dict(expected_role_counts)}:{observed}")
    components_by_role = {
        role: {str(row.get("component_hash")) for row in groups if row.get("role") == role}
        for role in ROLES
    }
    if components_by_role["test"] & components_by_role["online"]:
        raise CaptureError("component_overlap")
    normalized: list[JsonDict] = []
    for source in groups:
        group = deepcopy(dict(source))
        role = str(group.get("role"))
        donor_role = str(group.get("donor_role", role))
        if donor_role != role:
            raise CaptureError(f"cross_role_donor:{role}:{donor_role}")
        group["donor_role"] = donor_role
        normalized.append(group)
    schedule = capture.compile_capture_schedule(
        normalized, expected_role_counts=expected_role_counts
    )
    by_component = {str(group["component_hash"]): str(group["donor_role"]) for group in normalized}
    for row in schedule:
        row["donor_role"] = by_component[str(row["component_hash"])]
    return schedule


def reduce_evaluation_rows(
    rows: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    expected_role_counts: Mapping[str, int] = ROLE_COUNTS,
) -> JsonDict:
    """Reduce native rows and add byte, model, donor, and role custody."""

    reduction = capture.reduce_capture_rows(
        rows, schedule, expected_role_counts=expected_role_counts
    )
    failures = set(reduction["failed_checks"])
    expected_by_id = {str(row.get("request_id")): row for row in schedule}
    for row in rows:
        expected = expected_by_id.get(str(row.get("request_id")))
        if expected is None:
            continue
        if row.get("donor_role") != row.get("role"):
            failures.add("cross_role_donor")
        prompt = row.get("prompt")
        reply = row.get("reply")
        if (
            not isinstance(prompt, str)
            or not isinstance(reply, str)
            or row.get("prompt_utf8_bytes") != len(prompt.encode("utf-8"))
            or row.get("reply_utf8_bytes") != len(reply.encode("utf-8"))
            or row.get("complete_source_window") != prompt
            or row.get("complete_response_window") != reply
        ):
            failures.add("transport_bytes")
        identity = row.get("model_identity_sha256")
        if not isinstance(identity, str) or not identity.startswith("sha256:"):
            failures.add("model_identity")
    role_by_group = {str(row.get("group_hash")): str(row.get("role")) for row in schedule}
    comparative = [deepcopy(dict(row)) for row in reduction["comparative_rows"]]
    for row in comparative:
        row["role"] = role_by_group.get(str(row.get("group_hash")))
        row["disposition"] = "complete_feature_only_no_truth_label"
    return {
        **reduction,
        "passed": not failures,
        "failed_checks": sorted(failures),
        "comparative_rows": comparative,
        "role_counts": dict(expected_role_counts),
        "role_group_counts": dict(
            Counter(str(group[0].get("role")) for group in _grouped(rows) if group)
        ),
        "role_forward_counts": dict(Counter(str(row.get("role")) for row in rows)),
        "cross_role_donor_count": sum(row.get("donor_role") != row.get("role") for row in rows),
    }


def _grouped(rows: Sequence[Mapping[str, Any]]) -> list[list[JsonDict]]:
    groups: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("group_hash"))].append(deepcopy(dict(row)))
    return list(groups.values())


def _checkpoint_document(
    schedule: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    model_identity_sha256: str,
    receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    reset_rows = [
        {
            "request_id": row.get("request_id"),
            "state_reset": row.get("state_reset"),
            "kv_state_id": row.get("kv_state_id"),
        }
        for row in rows
    ]
    return {
        "schema": "carnot.exp7565.v661.evaluation_checkpoint.v1",
        "schedule_sha256": canonical_hash(schedule),
        "model_identity_sha256": model_identity_sha256,
        "completed_group_hashes": sorted({str(row.get("group_hash")) for row in rows}),
        "request_ids_sha256": canonical_hash([row.get("request_id") for row in rows]),
        "reset_evidence_sha256": canonical_hash(reset_rows),
        "raw_rows_sha256": canonical_hash(rows),
        "group_receipts": [deepcopy(dict(receipt)) for receipt in receipts],
    }


def write_checkpoint(
    path: Path,
    schedule: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    model_identity_sha256: str,
) -> JsonDict:
    """Persist complete groups separately so each checkpoint is bounded."""

    schedule_by_hash: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for request in schedule:
        schedule_by_hash[str(request.get("group_hash"))].append(request)
    receipts: list[JsonDict] = []
    group_dir = path.parent / "checkpoint_groups"
    for group_rows in _grouped(rows):
        group_hash = str(group_rows[0].get("group_hash"))
        expected = schedule_by_hash.get(group_hash, [])
        counts = Counter(str(row.get("role")) for row in expected[::6])
        reduced = reduce_evaluation_rows(group_rows, expected, dict(counts))
        if len(group_rows) != 6 or reduced["passed"] is not True:
            raise EvaluationCaptureError("checkpoint_partial_or_changed")
        target = group_dir / f"{group_hash.removeprefix('sha256:')}.jsonl"
        payload = capture._jsonl_bytes(group_rows)
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        if (
            target.is_file()
            and target.stat().st_size == len(payload)
            and sha256_file(target) == digest
        ):
            receipt = {
                "path": capture._path_label(target, REPO_ROOT),
                "sha256": digest,
                "bytes": len(payload),
                "rows": len(group_rows),
            }
        else:
            receipt = capture.write_jsonl_sidecar(target, group_rows, root=REPO_ROOT)
        receipts.append(receipt)
    document = _checkpoint_document(schedule, rows, model_identity_sha256, receipts)
    atomic_json(path, document)
    return document


def load_checkpoint(
    path: Path,
    schedule: Sequence[Mapping[str, Any]],
    model_identity_sha256: str,
) -> list[JsonDict]:
    """Resume only exact model, schedule, complete-group, and reset evidence."""

    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationCaptureError("checkpoint_identity") from exc
    if not isinstance(value, Mapping):
        raise EvaluationCaptureError("checkpoint_identity")
    if (
        value.get("schema") != "carnot.exp7565.v661.evaluation_checkpoint.v1"
        or value.get("schedule_sha256") != canonical_hash(schedule)
        or value.get("model_identity_sha256") != model_identity_sha256
    ):
        raise EvaluationCaptureError("checkpoint_identity")
    receipts = value.get("group_receipts")
    if not isinstance(receipts, list):
        raise EvaluationCaptureError("checkpoint_identity")
    rows: list[JsonDict] = []
    try:
        for receipt in receipts:
            if not isinstance(receipt, Mapping):
                raise EvaluationCaptureError("checkpoint_identity")
            rows.extend(capture._read_jsonl_receipt(receipt, REPO_ROOT))
    except (OSError, json.JSONDecodeError, EvaluationCaptureError) as exc:
        raise EvaluationCaptureError("checkpoint_identity") from exc
    expected = _checkpoint_document(schedule, rows, model_identity_sha256, receipts)
    if dict(value) != expected:
        raise EvaluationCaptureError("checkpoint_identity")
    schedule_by_hash: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for request in schedule:
        schedule_by_hash[str(request.get("group_hash"))].append(request)
    for group_rows in _grouped(rows):
        expected_rows = schedule_by_hash.get(str(group_rows[0].get("group_hash")), [])
        counts = Counter(str(row.get("role")) for row in expected_rows[::6])
        if reduce_evaluation_rows(group_rows, expected_rows, dict(counts))["passed"] is not True:
            raise EvaluationCaptureError("checkpoint_partial_or_changed")
    return rows


def _fixture_rows(schedule: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    rows = [capture._scripted_native_row(row, index) for index, row in enumerate(schedule)]
    for row in rows:
        order = list(row["option_order"])
        semantic = dict(row["full_logits_by_option_id"])
        row.update(
            {
                "display_logits": {
                    " A": semantic[order[0]],
                    " B": semantic[order[1]],
                },
                "reply": "",
                "reply_utf8_bytes": 0,
                "prompt_encoding": "utf-8",
                "reply_encoding": "utf-8",
                "complete_source_window": row["prompt"],
                "complete_response_window": "",
                "model_identity_sha256": "sha256:model-identity",
            }
        )
    return rows


def run_qualification_controls(output_dir: Path) -> JsonDict:
    """Prove role, option, donor, and checkpoint corruption fail closed."""

    output_dir.mkdir(parents=True, exist_ok=True)
    role_counts = {role: 1 for role in ROLES}
    groups = [capture._fixture_group(role, 0) for role in ROLES]
    for group in groups:
        group["donor_role"] = group["role"]
    schedule = compile_evaluation_schedule(groups, expected_role_counts=role_counts)
    rows = _fixture_rows(schedule)
    baseline = reduce_evaluation_rows(rows, schedule, role_counts)

    changed_order = deepcopy(rows)
    changed_order[0]["option_order"] = list(reversed(changed_order[0]["option_order"]))
    order_reduction = reduce_evaluation_rows(changed_order, schedule, role_counts)

    changed_role = deepcopy(rows)
    changed_role[0]["role"] = "online"
    role_reduction = reduce_evaluation_rows(changed_role, schedule, role_counts)

    changed_donor = deepcopy(rows)
    changed_donor[0]["donor_role"] = "policy"
    donor_reduction = reduce_evaluation_rows(changed_donor, schedule, role_counts)

    checkpoint = output_dir / "checkpoint.json"
    write_checkpoint(checkpoint, schedule, rows, "sha256:model-identity")
    document = json.loads(checkpoint.read_text(encoding="utf-8"))
    document["model_identity_sha256"] = "sha256:changed"
    atomic_json(checkpoint, document)
    checkpoint_failures: list[str] = []
    try:
        load_checkpoint(checkpoint, schedule, "sha256:model-identity")
    except EvaluationCaptureError:
        checkpoint_failures.append("checkpoint_identity")
    mutation_failures = {
        "role": role_reduction,
        "option_order": order_reduction,
        "donor_role": donor_reduction,
        "checkpoint_identity": {
            "passed": not checkpoint_failures,
            "failed_checks": checkpoint_failures,
        },
    }
    passed = baseline["passed"] is True and all(
        row["passed"] is False for row in mutation_failures.values()
    )
    return {
        "passed": passed,
        "baseline_reduction": baseline,
        "mutation_failures": mutation_failures,
        "counted_as_current_model_invocations": False,
    }


def _protocol_row(row: Mapping[str, Any]) -> JsonDict:
    """Supply display aliases when scripted controls retain semantic logits only."""

    normalized = deepcopy(dict(row))
    if not isinstance(normalized.get("display_logits"), Mapping):
        order = list(normalized.get("option_order") or [])
        semantic = normalized.get("full_logits_by_option_id")
        if len(order) != 2 or not isinstance(semantic, Mapping):
            raise CaptureError("readout_mapping_invalid")
        normalized["display_logits"] = {
            " A": semantic.get(order[0]),
            " B": semantic.get(order[1]),
        }
    return normalized


def _prediction_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Reduce each six-cell group without carrying any evaluator label."""

    predictions: list[JsonDict] = []
    for group_rows in _grouped(rows):
        if len(group_rows) != 6:
            raise CaptureError("prediction_group_incomplete")
        first = group_rows[0]
        protocol_rows = [_protocol_row(row) for row in group_rows]
        original = [row for row in protocol_rows if row.get("condition") == "original"]
        if len(original) != 2:
            raise CaptureError("prediction_original_cells_invalid")
        probabilities = [protocol.semantic_unsupported_probability(row) for row in original]
        predictions.append(
            {
                "component_hash": first.get("component_hash"),
                "group_hash": first.get("group_hash"),
                "role": first.get("role"),
                "tool_type": first.get("tool_type"),
                "feature_views": protocol.build_feature_vector(protocol_rows),
                "original_unsupported_probability": sum(probabilities) / len(probabilities),
                "prediction_source": "mapped_native_option_logits",
                "label_scope": "original_source_only",
            }
        )
    return predictions


def online_baseline(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Freeze the original-source mean before any online outcome opens."""

    original = [
        row for row in rows if row.get("role") == "online" and row.get("condition") == "original"
    ]
    values = [protocol.semantic_unsupported_probability(_protocol_row(row)) for row in original]
    if not values:
        raise CaptureError("online_original_predictions_missing")
    return {
        "method": "mean_mapped_original_source_probability",
        "mean_probability": sum(values) / len(values),
        "original_forward_count": len(values),
        "unsupported_label": 1,
        "label_semantics": "one_means_unsupported_original_content",
        "intervention_absence_used_as_label": False,
        "frozen_before_online_label_release": True,
    }


def freeze_prediction_sidecar(
    raw_dir: Path, rows: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], JsonDict]:
    """Persist label-free features before any evaluator row opens."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    predictions = _prediction_rows(rows)
    prediction_receipt = capture.write_jsonl_sidecar(
        raw_dir / "predictions.jsonl", predictions, root=REPO_ROOT
    )
    return predictions, prediction_receipt


def write_evaluator_sidecars(
    raw_dir: Path,
    predictions: Sequence[Mapping[str, Any]],
    prediction_receipt: Mapping[str, Any],
    evaluators: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Write labels only after the exact prediction receipt exists."""

    predictor_by_key = {
        (str(row.get("component_hash")), str(row.get("role"))): row for row in predictions
    }
    evaluator_by_key: dict[tuple[str, str], JsonDict] = {}
    for source in evaluators:
        key = (str(source.get("component_hash")), str(source.get("role")))
        label = source.get("label")
        if key not in predictor_by_key or label not in {0, 1} or key in evaluator_by_key:
            raise CaptureError("evaluator_identity_or_label_invalid")
        evaluator_by_key[key] = {
            "component_hash": key[0],
            "role": key[1],
            "label": int(label),
            "label_semantics": "one_means_unsupported_original_content",
            "label_provenance": source.get("label_provenance", "injected_tool_errors"),
            "arrival_rank": source.get("arrival_rank"),
            "prediction_freeze_sha256": canonical_hash(predictions),
        }
    if set(evaluator_by_key) != set(predictor_by_key):
        raise CaptureError("evaluator_identity_or_label_invalid")
    labels_by_role = {
        role: [evaluator_by_key[key] for key in predictor_by_key if key[1] == role]
        for role in ROLES
    }
    test_receipt = capture.write_jsonl_sidecar(
        raw_dir / "test_labels.jsonl", labels_by_role["test"], root=REPO_ROOT
    )
    online_receipt = capture.write_jsonl_sidecar(
        raw_dir / "online_labels.jsonl", labels_by_role["online"], root=REPO_ROOT
    )
    return {
        "predictions": deepcopy(dict(prediction_receipt)),
        "test_labels": test_receipt,
        "online_labels": online_receipt,
        "prediction_freeze_sha256": canonical_hash(predictions),
        "label_release_after_prediction_freeze": True,
        "online_release_delay": 8,
        "roles": list(ROLES),
    }


def write_role_sidecars(
    raw_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    evaluators: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Use the same ordered freeze and release boundary in fixture work."""

    predictions, receipt = freeze_prediction_sidecar(raw_dir, rows)
    return write_evaluator_sidecars(raw_dir, predictions, receipt, evaluators)


def build_exposure_audit(
    inventory: Mapping[str, Any], access_events: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Preserve prior outcome access so capture cannot claim fresh evidence."""

    labels_consumed = sum(int(row.get("labels_consumed") or 0) for row in access_events)
    prior = labels_consumed > 0 or int(inventory.get("exposed_candidate_groups") or 0) > 0
    return {
        "prior_outcome_inspection": prior,
        "fresh_evidence_claim": False,
        "confirmatory_claim_allowed": False,
        "preserved_inventory_sha256": inventory.get("inventory_sha256"),
        "candidate_official_training_groups": inventory.get("candidate_official_training_groups"),
        "exposed_candidate_groups": inventory.get("exposed_candidate_groups"),
        "fresh_eligible_groups": inventory.get("fresh_eligible_groups"),
        "prior_labels_consumed": labels_consumed,
        "disposition": "previous_outcome_access_preserved_not_fresh_confirmatory_evidence",
    }


def write_raw_manifest(
    raw_dir: Path,
    schedule: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Shard complete groups so every transport sidecar remains bounded."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    schedule_groups = _grouped(schedule)
    row_groups = _grouped(rows)

    def write_shards(name: str, groups: Sequence[Sequence[Mapping[str, Any]]]) -> list[JsonDict]:
        receipts: list[JsonDict] = []
        if not groups:
            return [capture.write_jsonl_sidecar(raw_dir / f"{name}_000.jsonl", [], root=REPO_ROOT)]
        for start in range(0, len(groups), 30):
            shard = [dict(row) for group in groups[start : start + 30] for row in group]
            receipt = capture.write_jsonl_sidecar(
                raw_dir / f"{name}_{start // 30:03d}.jsonl", shard, root=REPO_ROOT
            )
            if int(receipt["bytes"]) >= 20 * 1024 * 1024:
                raise EvaluationCaptureError("raw_sidecar_size_limit")
            receipts.append(receipt)
        return receipts

    schedule_receipts = write_shards("schedule", schedule_groups)
    row_receipts = write_shards("native_rows", row_groups)
    manifest: JsonDict = {
        "schedule_shards": schedule_receipts,
        "native_row_shards": row_receipts,
        "schedule_sha256": canonical_hash(schedule),
        "native_rows_sha256": canonical_hash(rows),
        "group_count": len(row_groups),
        "forward_count": len(rows),
        "all_below_20_mib": True,
    }
    if len(schedule_receipts) == 1:
        manifest["forward_schedule"] = schedule_receipts[0]
    if len(row_receipts) == 1:
        manifest["native_rows"] = row_receipts[0]
    return manifest


def _reload_shards(receipts: object, root: Path) -> list[JsonDict]:
    if not isinstance(receipts, list) or not receipts:
        raise EvaluationCaptureError("sidecar_receipt_invalid")
    rows: list[JsonDict] = []
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            raise EvaluationCaptureError("sidecar_receipt_invalid")
        rows.extend(capture._read_jsonl_receipt(receipt, root))
    return rows


def independent_reduce(value: Mapping[str, Any], *, root: Path = REPO_ROOT) -> JsonDict:
    """Reload exact sidecars and recompute capture custody with zero model calls."""

    if value.get("verdict_class") == "blocked":
        budget = value.get("sample_size_budget")
        passed = bool(
            value.get("rows") == []
            and value.get("model_invoked") is False
            and isinstance(budget, Mapping)
            and budget.get("unstarted") == CAPTURE_FORWARDS
        )
        return {"passed": passed, "mode": "blocked_no_measurement", "forward_count": 0}
    manifest = value.get("raw_manifest")
    raw = dict(manifest) if isinstance(manifest, Mapping) else {}
    try:
        schedule = _reload_shards(raw.get("schedule_shards"), root)
        rows = _reload_shards(raw.get("native_row_shards"), root)
        role_sidecars = raw.get("role_sidecars")
        if not isinstance(role_sidecars, Mapping):
            raise CaptureError("sidecar_receipt_invalid")
        predictions = capture._read_jsonl_receipt(role_sidecars["predictions"], root)
        test_labels = capture._read_jsonl_receipt(role_sidecars["test_labels"], root)
        online_labels = capture._read_jsonl_receipt(role_sidecars["online_labels"], root)
    except (EvaluationCaptureError, OSError, json.JSONDecodeError):
        return {"passed": False, "error": "sidecar_receipt_invalid"}
    role_counts = value.get("role_counts")
    if not isinstance(role_counts, Mapping):
        return {"passed": False, "error": "role_counts_invalid"}
    reduction = reduce_evaluation_rows(rows, schedule, dict(role_counts))
    declared = value.get("capture_reduction")
    reduction_matches = reduction == declared
    ready = value.get("capture_bundle_complete_score") == 1
    expected_predictions = _prediction_rows(rows)
    sidecars_pass = bool(
        predictions == expected_predictions
        and all("label" not in row for row in predictions)
        and len(test_labels) == ROLE_COUNTS["test"]
        and len(online_labels) == ROLE_COUNTS["online"]
        and {row.get("role") for row in test_labels} == {"test"}
        and {row.get("role") for row in online_labels} == {"online"}
        and all("display_logits" not in row for row in [*test_labels, *online_labels])
    )
    passed = reduction_matches and sidecars_pass and (reduction.get("passed") is ready)
    return {
        "passed": passed,
        "capture_ready": reduction.get("passed") is True,
        "forward_count": reduction.get("complete_forward_count"),
        "group_count": reduction.get("complete_group_count"),
        "schedule_sha256": canonical_hash(schedule),
        "native_rows_sha256": canonical_hash(rows),
        "reduction_sha256": canonical_hash(reduction),
        "declared_reduction_sha256": canonical_hash(declared),
        "prediction_sha256": canonical_hash(predictions),
        "role_sidecars_passed": sidecars_pass,
    }


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind every terminal field except the checksum that contains the digest."""

    stable = deepcopy(dict(value))
    stable.pop("reproducibility_checksum", None)
    return canonical_hash(stable)


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    specific = {
        "experiment_id": "The exact task, milestone, and run date prevent evidence reuse.",
        "preconditions_checked": "Recorded observations prevent fabricated fallback inputs.",
        "MODEL_SPECS": "The planned model name prevents silent model substitution.",
        "model_specs": "Resolved file, hash, quantization, and runtime bind actual work.",
        "model_invoked": "The invocation flag prevents aggregation from looking like inference.",
        "invocation_counts": "Typed call states expose failed, cancelled, and unstarted work.",
        "inference_substrate_class": "The actual class sets the correct duration floor.",
        "inference_substrate": "The option-logit substrate prevents a generation claim.",
        "execution_venue": "A legal venue value stays separate from device identity.",
        "duration_s": "Monotonic timing prevents invented or padded model duration.",
        "random_seed": "Frozen seeds make model, ordering, and bootstrap work reproducible.",
        "reproducibility_checksum": "The checksum binds code, inputs, settings, and evidence.",
        "rows": "Absolute group metrics prevent missing observations from becoming zero.",
        "sample_size_budget": "Full accounting keeps failed and unstarted units visible.",
        "acceptance_gate_results": "Typed gates separate validity, readiness, and benefit.",
        "gate_check_summary": "Exact failed operands route a block without guesswork.",
        "honest_verdict": "A complete prefix prevents false retry classification.",
        "verdict_class": "The closed class separates absence, invalidity, and partial work.",
        "verifier_is_oracle": "The false value prevents probabilistic energy from certifying truth.",
        "flagged_adversarial": "The field preserves verifier findings instead of erasing them.",
        "validation_receipts": "Exact command receipts prove the checked scope and exits.",
        "test_capture_ready_score": "One requires 80 groups and 480 authenticated forwards.",
        "online_capture_ready_score": "One requires 160 groups and 960 authenticated forwards.",
        "capture_bundle_complete_score": "One requires both role captures to be ready.",
        "raw_manifest": "Hash-bound sidecars retain exact transport bytes outside the headline.",
        "role_counts": "Fixed 80/160 roles prevent silent sample reduction.",
        "source_label_access": "An empty list proves the capture runner stayed label blind.",
        "exposure_audit": "Prior outcome access prevents a false fresh-evidence claim.",
        "online_baseline": "The original-source mean prevents absence becoming a truth label.",
        "predictive_benefit_measured": "False prevents feature collection from claiming efficacy.",
    }
    return {
        field: specific.get(field, f"The {field} field preserves one auditable terminal operand.")
        for field in fields
    }


def _validation_passed(receipts: object) -> bool:
    if not isinstance(receipts, list):
        return False
    passed = {
        str(row.get("name"))
        for row in receipts
        if isinstance(row, Mapping) and row.get("passed") is True and row.get("exit_code") == 0
    }
    return set((*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)) <= passed


def _base_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "schema_binding": {
            "experiment_id": EXPERIMENT_ID,
            "milestone": MILESTONE,
            "version": VERSION,
            "run_date": RUN_DATE,
        },
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "version": VERSION,
        "run_date": RUN_DATE,
        "title": "V661 sealed test and online source-intervention capture",
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": reduce_invocation_events([]),
        "historical_invocation_counts": {},
        "inference_substrate_class": "model_load_no_generation",
        "inference_substrate": "live_llm_embedding_extraction",
        "readout_kind": "option_logits",
        "execution_venue": "host",
        "execution_device": "real_cuda_required",
        "duration_s": max(0.000001, float(duration_s)),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "process_identity": {
            "pid": os.getpid(),
            "process_start_ticks": lease_api.proc_start_ticks(os.getpid()),
            "python": os.path.realpath(os.sys.executable),
            "clock": "time.monotonic",
        },
        "random_seed": {
            "model": 661565,
            "fitting": 661566,
            "ordering": 661567,
            "bootstrap": 661568,
        },
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "adversarial_corrections": [],
        "positive_claim": False,
        "benefit_measured": False,
        "predictive_benefit_measured": False,
        "generator_weights_changed": False,
        "production_defaults_changed": False,
        "external_publication_authorized": False,
        "applicable_numbered_e2e": [],
        "private_llm_off_real_environment_smoke": "not_applicable_isolated_feature_capture",
        "role_counts": deepcopy(ROLE_COUNTS),
        "source_label_access": [],
        "evaluator_label_access": {
            "after_prediction_freeze": False,
            "roles_opened": [],
            "labels_consumed": 0,
        },
    }


def _ready_gates(
    reduction: Mapping[str, Any], controls: Mapping[str, Any], validation_ok: bool
) -> list[JsonDict]:
    return [
        gate(
            "raw_capture_custody",
            "validity",
            True,
            reduction.get("passed"),
            "is",
            reduction.get("passed") is True,
            "Invalid transport rows cannot support evaluation readiness.",
            upstream="raw_manifest",
            field="capture_reduction.passed",
        ),
        gate(
            "complete_group_budget",
            "readiness",
            CAPTURE_GROUPS,
            reduction.get("complete_group_count"),
            "==",
            reduction.get("complete_group_count") == CAPTURE_GROUPS,
            "All fixed groups are required before an evaluation consumer can run.",
            upstream="raw_manifest",
            field="capture_reduction.complete_group_count",
        ),
        gate(
            "complete_forward_budget",
            "readiness",
            CAPTURE_FORWARDS,
            reduction.get("complete_forward_count"),
            "==",
            reduction.get("complete_forward_count") == CAPTURE_FORWARDS,
            "All six cells per group are required for option-order averaging.",
            upstream="raw_manifest",
            field="capture_reduction.complete_forward_count",
        ),
        gate(
            "test_role_budget",
            "readiness",
            {"groups": 80, "forwards": 480},
            {
                "groups": reduction.get("role_group_counts", {}).get("test"),
                "forwards": reduction.get("role_forward_counts", {}).get("test"),
            },
            "==",
            reduction.get("role_group_counts", {}).get("test") == 80
            and reduction.get("role_forward_counts", {}).get("test") == 480,
            "The test score requires every sealed official-test cell.",
            upstream="raw_manifest",
            field="capture_reduction.test_role_counts",
        ),
        gate(
            "online_role_budget",
            "readiness",
            {"groups": 160, "forwards": 960},
            {
                "groups": reduction.get("role_group_counts", {}).get("online"),
                "forwards": reduction.get("role_forward_counts", {}).get("online"),
            },
            "==",
            reduction.get("role_group_counts", {}).get("online") == 160
            and reduction.get("role_forward_counts", {}).get("online") == 960,
            "The online score requires every sealed arrival and option cell.",
            upstream="raw_manifest",
            field="capture_reduction.online_role_counts",
        ),
        gate(
            "zero_cross_role_donors",
            "validity",
            0,
            reduction.get("cross_role_donor_count"),
            "==",
            reduction.get("cross_role_donor_count") == 0,
            "Cross-role donors would leak the sealed split boundary.",
            upstream="raw_manifest",
            field="capture_reduction.cross_role_donor_count",
        ),
        gate(
            "qualification_mutations",
            "validity",
            True,
            controls.get("passed"),
            "is",
            controls.get("passed") is True,
            "Required corruptions must fail before captured evidence can publish.",
            upstream="qualification_controls",
            field="passed",
        ),
        gate(
            "required_validation",
            "validity",
            True,
            validation_ok,
            "is",
            validation_ok,
            "Scoped and terminal checks must pass before publication.",
            upstream="validation_receipts",
            field="all_required_checks_passed",
        ),
        gate(
            "predictive_benefit_measured",
            "benefit",
            False,
            False,
            "is",
            True,
            "Feature acquisition cannot substitute for empirical predictive benefit.",
            upstream="capture_only",
            field="predictive_benefit_measured",
        ),
    ]


def build_complete_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    model_specs: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    historical_invocation_counts: Mapping[str, Any],
    raw_manifest: Mapping[str, Any],
    reduction: Mapping[str, Any],
    controls: Mapping[str, Any],
    ownership: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    require_validation: bool,
    exposure_audit: Mapping[str, Any] | None = None,
    online_baseline_state: Mapping[str, Any] | None = None,
    live_error: str | None = None,
) -> JsonDict:
    """Build one complete, partial, or invalid capture without an efficacy claim."""

    validation_ok = _validation_passed(list(validation_receipts)) if require_validation else True
    gates = _ready_gates(reduction, controls, validation_ok)
    ready = all(row["passed"] is True for row in gates if row["category"] != "benefit")
    capture_valid = reduction.get("passed") is True and controls.get("passed") is True
    test_ready = capture_valid and reduction.get("role_group_counts", {}).get("test") == 80
    test_ready = test_ready and reduction.get("role_forward_counts", {}).get("test") == 480
    online_ready = capture_valid and reduction.get("role_group_counts", {}).get("online") == 160
    online_ready = online_ready and reduction.get("role_forward_counts", {}).get("online") == 960
    completed = int(reduction.get("complete_forward_count") or 0)
    if ready:
        verdict_class = "null"
        verdict = "complete_null_test_online_capture_ready_benefit_unmeasured"
    elif live_error is not None and completed < CAPTURE_FORWARDS:
        verdict_class = "partial"
        verdict = "complete_partial_owned_test_online_capture_retryable"
    else:
        verdict_class = "disqualified"
        verdict = "complete_disqualified_test_online_capture_invalid_evidence"
    artifact = _base_artifact(
        preconditions=preconditions,
        source_hashes=source_hashes,
        duration_s=duration_s,
        phase_spans=phase_spans,
    )
    counts = reduce_invocation_events(events)
    resumed = int(historical_invocation_counts.get("resumed_forward_count") or 0)
    attempted = min(CAPTURE_FORWARDS, int(counts["forward_calls_attempted"]) + resumed)
    failed = int(counts["forwards"]["failed"])
    artifact.update(
        {
            "model_specs": [deepcopy(dict(row)) for row in model_specs],
            "model_invoked": counts["model_loads_attempted"] > 0,
            "invocation_counts": counts,
            "current_invocation_events": [deepcopy(dict(row)) for row in events],
            "historical_invocation_counts": deepcopy(dict(historical_invocation_counts)),
            "rows": [deepcopy(dict(row)) for row in reduction.get("comparative_rows") or []],
            "raw_manifest": deepcopy(dict(raw_manifest)),
            "capture_reduction": deepcopy(dict(reduction)),
            "qualification_controls": deepcopy(dict(controls)),
            "sample_size_budget": {
                "planned": CAPTURE_FORWARDS,
                "attempted": attempted,
                "completed": completed,
                "excluded": 0,
                "failed": failed,
                "censored": 0,
                "unstarted": max(0, CAPTURE_FORWARDS - attempted),
            },
            "ownership_receipt": deepcopy(dict(ownership)),
            "acceptance_gate_results": gates,
            "gate_check_summary": gate_check_summary(gates),
            "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
            "test_capture_ready_score": int(test_ready),
            "online_capture_ready_score": int(online_ready),
            "capture_bundle_complete_score": int(test_ready and online_ready and ready),
            "evaluator_label_access": {
                "after_prediction_freeze": bool(
                    isinstance(raw_manifest.get("role_sidecars"), Mapping)
                    and raw_manifest["role_sidecars"].get("label_release_after_prediction_freeze")
                    is True
                ),
                "roles_opened": list(ROLES) if test_ready and online_ready else [],
                "labels_consumed": CAPTURE_GROUPS if test_ready and online_ready else 0,
            },
            "exposure_audit": deepcopy(dict(exposure_audit or {})),
            "online_baseline": deepcopy(dict(online_baseline_state or {})),
            "honest_verdict": verdict,
            "verdict_class": verdict_class,
            "complete": 1,
            "ready": int(ready),
        }
    )
    if live_error is not None:
        artifact["live_error"] = live_error
    artifact["field_principles"] = _field_principles(
        (*artifact.keys(), "field_principles", "reproducibility_checksum")
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


class BoundNativeTransport:  # pragma: no cover - real model boundary.
    """Add byte and model custody without changing the qualified scorer."""

    def __init__(self, runner: native.NativeOptionRunner, model_identity_sha256: str) -> None:
        self.runner = runner
        self.model_identity_sha256 = model_identity_sha256

    def score(self, request: JsonDict, owner: JsonDict) -> JsonDict:
        row = self.runner.score(request, owner)
        row.update(
            {
                "reply": "",
                "reply_utf8_bytes": 0,
                "prompt_encoding": "utf-8",
                "reply_encoding": "utf-8",
                "complete_source_window": row["prompt"],
                "complete_response_window": "",
                "model_identity_sha256": self.model_identity_sha256,
            }
        )
        return row


class EvaluationCaptureRunner:  # pragma: no cover - real model boundary.
    """Run and checkpoint only complete groups from the qualified transport."""

    def __init__(
        self,
        *,
        schedule: Sequence[Mapping[str, Any]],
        transport: BoundNativeTransport,
        checkpoint_path: Path,
        owner: Mapping[str, Any],
        model_identity_sha256: str,
    ) -> None:
        self.schedule = [deepcopy(dict(row)) for row in schedule]
        self.transport = transport
        self.checkpoint_path = checkpoint_path
        self.owner = deepcopy(dict(owner))
        self.model_identity_sha256 = model_identity_sha256

    def run(self, *, started: float) -> JsonDict:
        retained = load_checkpoint(self.checkpoint_path, self.schedule, self.model_identity_sha256)
        resumed_groups = len({str(row["group_hash"]) for row in retained})
        complete_hashes = {str(row["group_hash"]) for row in retained}
        groups: dict[str, list[JsonDict]] = defaultdict(list)
        for row in self.schedule:
            groups[str(row["group_hash"])].append(row)
        events: list[JsonDict] = []
        failed_units: list[JsonDict] = []
        error: str | None = None
        collection_started = time.monotonic()
        for group_index, (group_hash, requests) in enumerate(groups.items()):
            if group_hash in complete_hashes:
                continue
            if time.monotonic() - collection_started >= COLLECTION_LIMIT_S:
                error = "collection_limit_reached"
                break
            group_rows: list[JsonDict] = []
            for request in requests:
                call_id = f"fit-{int(request['schedule_index']):04d}"
                pilot.progress(
                    started,
                    "test_online_capture",
                    "before_forward",
                    completed_units=len(retained) + len(group_rows),
                    planned=CAPTURE_FORWARDS,
                    call_id=call_id,
                )
                events.append(_event("forward", "attempted", call_id))
                try:
                    row = native._call_with_heartbeats(
                        lambda request=request: self.transport.score(request, self.owner),
                        started=started,
                        phase="test_online_capture",
                        pending=call_id,
                    )
                except BaseException as exc:  # noqa: BLE001 - retain exact owned failure.
                    error = f"{type(exc).__name__}:{exc}"
                    events.append(_event("forward", "failed", call_id))
                    failed_units.append(
                        {
                            "request_id": request.get("request_id"),
                            "group_hash": group_hash,
                            "role": request.get("role"),
                            "disposition": "failed",
                            "error": error,
                        }
                    )
                    group_rows = []
                    break
                events.append(_event("forward", "completed", call_id))
                group_rows.append(deepcopy(dict(row)))
                pilot.progress(
                    started,
                    "test_online_capture",
                    "after_forward",
                    completed_units=len(retained) + len(group_rows),
                    planned=CAPTURE_FORWARDS,
                )
            if error is not None:
                break
            role_counts = {str(requests[0]["role"]): 1}
            if reduce_evaluation_rows(group_rows, requests, role_counts)["passed"] is not True:
                error = "group_custody_failed"
                failed_units.extend(group_rows)
                break
            retained.extend(group_rows)
            complete_hashes.add(group_hash)
            write_checkpoint(
                self.checkpoint_path,
                self.schedule,
                retained,
                self.model_identity_sha256,
            )
            pilot.progress(
                started,
                "test_online_capture",
                "group_checkpointed",
                completed_groups=len(complete_hashes),
                planned_groups=CAPTURE_GROUPS,
                role=requests[0]["role"],
                group_index=group_index,
            )
        return {
            "status": "complete" if len(complete_hashes) == len(groups) else "partial",
            "error": error,
            "rows": retained,
            "events": events,
            "failed_units": failed_units,
            "resumed_group_count": resumed_groups,
            "completed_group_count": len(complete_hashes),
            "planned_group_count": len(groups),
            "duration_s": time.monotonic() - collection_started,
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
    Path("python/carnot/experiment_7548_v660_capture_runner.py"),
    Path("python/carnot/experiment_7533_v659_tool_protocol.py"),
    Path("python/carnot/experiment_7535_v659_native_pilot.py"),
    Path("python/carnot/inference/sota_models.py"),
    PILOT_PATH,
    PROTOCOL_PATH,
    RUNNER_PATH,
    PILOT_MODULE_PATH,
    PILOT_WRAPPER_PATH,
    EXPOSURE_PATH,
    SPEC_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)


def _load_object(path: Path) -> JsonDict:  # pragma: no cover - live file boundary.
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def progress(started: float, phase: str, event: str, **details: object) -> None:  # pragma: no cover
    """Flush each slow boundary with monotonic time and completed units."""

    print(
        json.dumps(
            {"phase": phase, "event": event, "elapsed_s": time.monotonic() - started, **details},
            sort_keys=True,
            default=str,
        ),
        flush=True,
    )


def _phase_span(
    phase: str, phase_started: float, run_started: float, units: int, checkpoint: str
) -> JsonDict:  # pragma: no cover
    return {
        "phase": phase,
        "start_monotonic_offset_s": phase_started - run_started,
        "end_monotonic_offset_s": time.monotonic() - run_started,
        "duration_s": time.monotonic() - phase_started,
        "completed_units": units,
        "checkpoint": checkpoint,
    }


def collect_preconditions(
    root: Path, started: float, run_date: str
) -> tuple[list[JsonDict], dict[str, str], JsonDict]:  # pragma: no cover
    """Authenticate current sources, pilot authority, cached model, and runtime."""

    checks = [
        _contract_gate("run_date", RUN_DATE, run_date, upstream="command_line", field="--date")
    ]
    hashes: dict[str, str] = {}
    for relative in INPUT_PATHS:
        path = root / relative
        observed = "readable_nonempty_bytes" if path.is_file() and path.stat().st_size else None
        checks.append(
            _contract_gate(
                f"required_input:{relative.as_posix()}",
                "readable_nonempty_bytes",
                observed,
                upstream=relative.as_posix(),
                field="path.bytes",
                path=relative.as_posix(),
            )
        )
        if observed is not None:
            hashes[relative.as_posix()] = sha256_file(path)
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        _contract_gate(
            "driving_requirement",
            "REQ-VERIFY-7565",
            "REQ-VERIFY-7565" if "REQ-VERIFY-7565" in spec_text else None,
            upstream=SPEC_PATH.as_posix(),
            field="REQ-*",
            path=SPEC_PATH.as_posix(),
        )
    )
    pilot_artifact = _load_object(root / PILOT_PATH)
    pilot_checks, pilot_context = authenticate_pilot(pilot_artifact, hashes)
    checks.extend(pilot_checks)
    protocol_checks, sealed_context = capture.authenticate_protocol(root)
    checks.extend(protocol_checks)
    exposure_inventory = _load_object(root / EXPOSURE_PATH)
    checks.append(
        _contract_gate(
            "historical_exposure_inventory",
            True,
            bool(
                exposure_inventory.get("inventory_sha256")
                and exposure_inventory.get("fresh_eligible_groups") == 0
            ),
            upstream="historical_source_exposure_union",
            field="inventory_sha256_and_fresh_eligible_groups",
            path=EXPOSURE_PATH.as_posix(),
        )
    )

    model = cached_current_model(preferred_quant="Q4_K_M")
    model_path = Path(str(model.get("model_path"))) if model else Path("/nonexistent")
    model_ok = bool(
        model
        and model.get("hf_id") == MODEL_ID
        and "Q4_K_M" in model_path.name
        and model_path.is_file()
    )
    model_hash: str | None = None
    if model_ok:
        progress(started, "preconditions", "before_model_hash", bytes=model_path.stat().st_size)
        model_hash = native._call_with_heartbeats(
            lambda: sha256_file(model_path),
            started=started,
            phase="preconditions",
            pending="model_sha256",
        )
        progress(started, "preconditions", "after_model_hash", sha256=model_hash)
        hashes[str(model_path)] = model_hash
    checks.append(
        _contract_gate(
            "cached_pilot_model",
            {
                "hf_id": MODEL_ID,
                "quantization": "Q4_K_M",
                "model_sha256": pilot_context.get("model_sha256"),
            },
            {
                "hf_id": model.get("hf_id") if model else None,
                "quantization": "Q4_K_M" if "Q4_K_M" in model_path.name else None,
                "model_sha256": model_hash,
            },
            upstream="carnot.inference.sota_models.cached_current_model",
            field="hf_id_quantization_hash",
            path=str(model_path),
        )
    )
    try:
        import llama_cpp  # noqa: PLC0415

        runtime = {
            "available": True,
            "version": getattr(llama_cpp, "__version__", "unknown"),
            "module_path": str(Path(llama_cpp.__file__).resolve()),
        }
    except Exception as exc:  # noqa: BLE001
        runtime = {"available": False, "error": f"{type(exc).__name__}:{exc}"}
    checks.append(
        _contract_gate(
            "llama_cpp_runtime",
            True,
            runtime["available"],
            upstream="python_environment",
            field="llama_cpp.available",
            path=str(runtime.get("module_path")),
        )
    )
    return (
        checks,
        hashes,
        {
            "pilot": pilot_artifact,
            "pilot_context": pilot_context,
            "sealed": sealed_context,
            "model": model or {},
            "model_path": model_path,
            "model_hash": model_hash,
            "runtime": runtime,
            "exposure_inventory": exposure_inventory,
        },
    )


def freeze_evaluation_schedule(
    context: Mapping[str, Any], *, root: Path, started: float
) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover - sealed sidecar boundary.
    """Read predictor-only interventions and select both evaluation roles."""

    sealed = dict(context["sealed"])
    protocol_artifact = sealed.get("artifact")
    protocol_map = dict(protocol_artifact) if isinstance(protocol_artifact, Mapping) else {}
    role_manifest = protocol_map.get("role_manifest")
    role_rows = (
        list(role_manifest.get("groups") or []) if isinstance(role_manifest, Mapping) else []
    )
    component_roles = {
        str(row.get("component_hash")): str(row.get("role"))
        for row in role_rows
        if isinstance(row, Mapping)
    }
    groups: list[JsonDict] = []
    for name, receipt in sorted(dict(sealed["intervention_receipts"]).items()):
        progress(started, "capture_schedule", "before_intervention_read", shard=name)
        rows = capture._read_jsonl_receipt(dict(receipt), root)
        progress(
            started,
            "capture_schedule",
            "after_intervention_read",
            shard=name,
            completed_units=len(rows),
        )
        for source in rows:
            role = str(source.get("role"))
            if role not in ROLE_COUNTS:
                continue
            donor_hash = str(source.get("donor_component_hash"))
            groups.append(
                {
                    "component_hash": source.get("component_hash"),
                    "donor_component_hash": donor_hash,
                    "donor_role": component_roles.get(donor_hash),
                    "role": role,
                    "tool_type": source.get("tool_type"),
                    "label_scope": "original_source_only",
                    "requests": deepcopy(list(source.get("requests") or [])),
                }
            )
    schedule = compile_evaluation_schedule(groups)
    schedule_reduction = capture.validate_capture_schedule(
        schedule, expected_role_counts=ROLE_COUNTS
    )
    pilot_hashes = context["pilot_context"].get("role_schedule_hashes")
    if schedule_reduction.get("role_schedule_hashes") != pilot_hashes:
        raise EvaluationCaptureError("pilot_schedule_hashes_changed")
    return schedule, {
        "role_counts": deepcopy(ROLE_COUNTS),
        "roles": list(ROLES),
        "group_count": CAPTURE_GROUPS,
        "forward_count": CAPTURE_FORWARDS,
        "role_schedule_hashes": deepcopy(schedule_reduction["role_schedule_hashes"]),
        "schedule_sha256": canonical_hash(schedule),
        "source_label_access": [],
        "native_transport_rebuilt": False,
    }


def read_evaluator_labels(
    context: Mapping[str, Any], *, root: Path, started: float
) -> list[JsonDict]:  # pragma: no cover - evaluator boundary.
    """Open sealed labels only after prediction bytes have frozen."""

    sealed = dict(context["sealed"])
    protocol_artifact = sealed.get("artifact")
    protocol_map = dict(protocol_artifact) if isinstance(protocol_artifact, Mapping) else {}
    shards = protocol_map.get("sealed_shards")
    shard_map = dict(shards) if isinstance(shards, Mapping) else {}
    labels: list[JsonDict] = []
    for role, name in (("test", "test_labels"), ("online", "online_release")):
        receipt = shard_map.get(name)
        if not isinstance(receipt, Mapping):
            raise CaptureError(f"sealed_label_receipt_missing:{name}")
        progress(started, "label_release", "before_read", role=role)
        role_rows = capture._read_jsonl_receipt(dict(receipt), root)
        progress(
            started,
            "label_release",
            "after_read",
            role=role,
            completed_units=len(role_rows),
        )
        if len(role_rows) != ROLE_COUNTS[role] or {row.get("role") for row in role_rows} != {role}:
            raise CaptureError(f"sealed_label_role_count:{role}")
        labels.extend(role_rows)
    return labels


def _run_affected_validation(
    root: Path, started: float
) -> tuple[list[JsonDict], bool]:  # pragma: no cover - subprocess boundary.
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7565-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private)
    errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    if errors:
        raise EvaluationCaptureError("validation_plan_invalid:" + ",".join(errors))
    progress(started, "affected_validation", "before_subprocesses", planned=len(commands))
    receipts = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=root / RAW_DIR / "validation/affected",
        heartbeat_s=60.0,
    )
    progress(
        started,
        "affected_validation",
        "after_subprocesses",
        completed_units=len(receipts),
    )
    reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, receipts)
    return [dict(row) for row in receipts], reduction.get("passed") is True


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    relative = candidate.relative_to(REPO_ROOT).as_posix()
    python = ".venv/bin/python"
    specs = (
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (python, "-u", WRAPPER_PATH.as_posix(), "--date", RUN_DATE, "--cold-replay", relative),
            "completion",
            180,
        ),
        validation_scope.CommandSpec(
            "independent_raw_reduction",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--independent-reduce",
                relative,
            ),
            "completion",
            180,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", relative),
            "safety",
            180,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", relative),
            "completion",
            180,
        ),
    )
    return [PlannedCommand(spec, spec.scope, True) for spec in specs]


def _refresh_artifact(
    artifact: JsonDict,
    *,
    receipts: Sequence[Mapping[str, Any]],
    spans: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> None:  # pragma: no cover
    artifact["validation_receipts"] = [deepcopy(dict(row)) for row in receipts]
    artifact["phase_spans"] = [deepcopy(dict(row)) for row in spans]
    artifact["duration_s"] = max(0.000001, float(duration_s))
    validation_ok = _validation_passed(artifact["validation_receipts"])
    for row in artifact.get("acceptance_gate_results") or []:
        if isinstance(row, dict) and row.get("check") == "required_validation":
            row["observed"] = validation_ok
            row["passed"] = validation_ok
    if artifact.get("verdict_class") not in {"blocked", "partial"} and not validation_ok:
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
        artifact["test_capture_ready_score"] = 0
        artifact["online_capture_ready_score"] = 0
        artifact["capture_bundle_complete_score"] = 0
        artifact["ready"] = 0
    artifact["gate_check_summary"] = gate_check_summary(
        artifact.get("acceptance_gate_results") or []
    )
    artifact["field_principles"] = _field_principles(
        (*artifact.keys(), "field_principles", "reproducibility_checksum")
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)


def _run_terminal_validation(
    root: Path,
    artifact: JsonDict,
    affected: Sequence[Mapping[str, Any]],
    spans: list[JsonDict],
    started: float,
) -> tuple[list[JsonDict], bool]:  # pragma: no cover - subprocess boundary.
    candidate = root / CANDIDATE_PATH
    artifact["validation_receipts"] = [deepcopy(dict(row)) for row in affected]
    artifact["field_principles"] = _field_principles(
        (*artifact.keys(), "field_principles", "reproducibility_checksum")
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    atomic_json(candidate, artifact)
    commands = _terminal_commands(candidate)
    phase_started = time.monotonic()
    progress(started, "terminal_validation", "before_subprocesses", planned=len(commands))
    terminal = run_categorized_commands(
        root,
        commands,
        log_dir=root / RAW_DIR / "validation/terminal",
        heartbeat_s=60.0,
    )
    spans.append(
        _phase_span(
            "terminal_validation",
            phase_started,
            started,
            len(terminal),
            CANDIDATE_PATH.as_posix(),
        )
    )
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal),
    )
    receipts = [*affected, *[dict(row) for row in terminal]]
    return receipts, all(row.get("passed") is True for row in terminal)


def _finish_candidate(
    root: Path,
    artifact: JsonDict,
    affected: Sequence[Mapping[str, Any]],
    spans: list[JsonDict],
    started: float,
) -> int:  # pragma: no cover
    receipts, terminal_ok = _run_terminal_validation(root, artifact, affected, spans, started)
    if not terminal_ok and artifact.get("verdict_class") != "blocked":
        artifact["honest_verdict"] = "complete_disqualified_required_terminal_validation"
        artifact["verdict_class"] = "disqualified"
        artifact["test_capture_ready_score"] = 0
        artifact["online_capture_ready_score"] = 0
        artifact["capture_bundle_complete_score"] = 0
        artifact["ready"] = 0
    _refresh_artifact(
        artifact,
        receipts=receipts,
        spans=spans,
        duration_s=time.monotonic() - started,
    )
    errors = validate_artifact(artifact, require_validation=True)
    if errors:
        raise EvaluationCaptureError("terminal_artifact_invalid:" + ",".join(errors))
    progress(started, "publish", "before_atomic_write", path=RESULT_PATH.as_posix())
    atomic_json(root / RESULT_PATH, artifact)
    progress(started, "publish", "after_atomic_write", bytes=(root / RESULT_PATH).stat().st_size)
    return 0


def _blocked_finish(
    root: Path,
    failed: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, str],
    affected: Sequence[Mapping[str, Any]],
    spans: list[JsonDict],
    started: float,
    **extra: object,
) -> int:  # pragma: no cover
    artifact = build_blocked_artifact(
        failed_gate=failed,
        preconditions=checks,
        source_hashes=hashes,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    artifact.update(deepcopy(extra))
    artifact["field_principles"] = _field_principles(
        (*artifact.keys(), "field_principles", "reproducibility_checksum")
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return _finish_candidate(root, artifact, affected, spans, started)


def _stable_model_identity(model: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    """Exclude lease-specific fields so an identical model can resume safely."""

    return {
        key: deepcopy(model.get(key))
        for key in (
            "hf_id",
            "model_path",
            "model_sha256",
            "quantization",
            "runtime",
            "embedded_tokenizer",
            "n_ctx",
            "requested_offload",
            "model_block_count",
            "native_chat_template",
        )
    }


def run_experiment(root: Path, run_date: str) -> int:  # pragma: no cover - live orchestration.
    """Validate, acquire one GPU, capture 1,440 forwards, and publish evidence."""

    started = time.monotonic()
    spans: list[JsonDict] = []
    progress(started, "preconditions", "start")
    phase_started = time.monotonic()
    progress(started, "preconditions", "before_named_inputs")
    checks, hashes, context = collect_preconditions(root, started, run_date)
    progress(started, "preconditions", "after_named_inputs", completed_units=len(checks))
    spans.append(_phase_span("preconditions", phase_started, started, len(checks), "authenticated"))
    failed = next((row for row in checks if row.get("passed") is not True), None)

    phase_started = time.monotonic()
    affected, affected_ok = _run_affected_validation(root, started)
    spans.append(
        _phase_span(
            "affected_validation",
            phase_started,
            started,
            len(affected),
            "scoped_commands_complete",
        )
    )
    if not affected_ok:
        validation_gate = gate(
            "required_affected_validation",
            "validity",
            True,
            False,
            "is",
            False,
            "Required scoped checks must pass before model work.",
            upstream="affected_validation_receipts",
            field="all_required_checks_passed",
        )
        artifact = build_blocked_artifact(
            failed_gate=validation_gate,
            preconditions=checks,
            source_hashes=hashes,
            duration_s=time.monotonic() - started,
            phase_spans=spans,
        )
        artifact["honest_verdict"] = "complete_disqualified_required_affected_validation"
        artifact["verdict_class"] = "disqualified"
        artifact["field_principles"] = _field_principles(
            (*artifact.keys(), "field_principles", "reproducibility_checksum")
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        return _finish_candidate(root, artifact, affected, spans, started)
    if failed is not None:
        return _blocked_finish(root, failed, checks, hashes, affected, spans, started)

    phase_started = time.monotonic()
    progress(started, "capture_schedule", "before_evaluation_role_wrapper")
    try:
        schedule, capture_manifest = freeze_evaluation_schedule(context, root=root, started=started)
    except EvaluationCaptureError as exc:
        schedule_gate = _contract_gate(
            "sealed_evaluation_schedule",
            True,
            False,
            upstream="exp7563-v661-native-pilot",
            field="role_manifest.sealed_capture.evaluation",
            path=PILOT_PATH.as_posix(),
        )
        checks.append(schedule_gate)
        return _blocked_finish(
            root,
            schedule_gate,
            checks,
            hashes,
            affected,
            spans,
            started,
            schedule_error=f"{type(exc).__name__}:{exc}",
        )
    schedule_receipts = write_raw_manifest(root / RAW_DIR / "planned", schedule, [])
    progress(
        started,
        "capture_schedule",
        "after_evaluation_role_wrapper",
        completed_units=CAPTURE_GROUPS,
        forwards=len(schedule),
    )
    spans.append(
        _phase_span(
            "capture_schedule",
            phase_started,
            started,
            CAPTURE_GROUPS,
            "evaluation_schedule_frozen",
        )
    )

    phase_started = time.monotonic()
    progress(started, "qualification_controls", "before_private_mutations")
    controls_dir = Path(tempfile.mkdtemp(prefix="carnot-exp7565-controls-", dir="/tmp"))
    controls = run_qualification_controls(controls_dir)
    progress(
        started,
        "qualification_controls",
        "after_private_mutations",
        completed_units=4,
        passed=controls["passed"],
    )
    spans.append(
        _phase_span(
            "qualification_controls",
            phase_started,
            started,
            4,
            "private_controls_complete",
        )
    )
    if controls["passed"] is not True:
        control_gate = _contract_gate(
            "qualification_mutations",
            True,
            controls.get("passed"),
            upstream="private_qualification_controls",
            field="passed",
        )
        checks.append(control_gate)
        artifact = build_blocked_artifact(
            failed_gate=control_gate,
            preconditions=checks,
            source_hashes=hashes,
            duration_s=time.monotonic() - started,
            phase_spans=spans,
        )
        artifact["honest_verdict"] = "complete_disqualified_qualification_controls"
        artifact["verdict_class"] = "disqualified"
        artifact["qualification_controls"] = controls
        artifact["field_principles"] = _field_principles(
            (*artifact.keys(), "field_principles", "reproducibility_checksum")
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        return _finish_candidate(root, artifact, affected, spans, started)

    progress(started, "gpu_admission", "before_wait", wait_limit_s=native.ADMISSION_WAIT_S)
    phase_started = time.monotonic()
    selected_gpu, admission = native.wait_for_owned_gpu(started)
    progress(
        started,
        "gpu_admission",
        "after_wait",
        selected_uuid=selected_gpu.get("uuid") if selected_gpu else None,
        waited_s=admission["waited_s"],
    )
    spans.append(
        _phase_span(
            "gpu_admission",
            phase_started,
            started,
            int(selected_gpu is not None),
            "one_gpu_selected" if selected_gpu else "wait_closed",
        )
    )
    admission_gate = _contract_gate(
        "owned_gpu_available",
        True,
        selected_gpu is not None,
        upstream="nvidia-smi",
        field="admissible_single_gpu",
        path="nvidia-smi_compute_process_inventory",
    )
    checks.append(admission_gate)
    if selected_gpu is None:
        return _blocked_finish(
            root,
            admission_gate,
            checks,
            hashes,
            affected,
            spans,
            started,
            current_gpu_admission=admission,
            planned_schedule_manifest=schedule_receipts,
        )

    model_path = Path(context["model_path"])
    lease_dir = Path(os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"))
    progress(started, "gpu_lease", "before_acquire", device_uuid=selected_gpu["uuid"])
    try:
        lease = lease_api.GpuLease.acquire(
            runtime_dir=lease_dir,
            task_id=EXPERIMENT_ID,
            device_uuid=str(selected_gpu["uuid"]),
            expected_model=str(model_path),
            vram_before_mb=int(selected_gpu["memory_used_mb"]),
            ttl_s=4800.0,
        )
    except lease_api.LeaseError as exc:
        progress(started, "gpu_lease", "after_acquire", acquired=False, error=str(exc))
        lease_gate = _contract_gate(
            "fresh_exclusive_gpu_lease",
            True,
            False,
            upstream=str(lease_dir),
            field="exclusive_device_lease_acquired",
            path=str(lease_dir),
        )
        checks.append(lease_gate)
        return _blocked_finish(
            root,
            lease_gate,
            checks,
            hashes,
            affected,
            spans,
            started,
            lease_error=f"{type(exc).__name__}:{exc}",
        )
    progress(started, "gpu_lease", "after_acquire", acquired=True, lease_id=lease.lease_id)
    owner = lease.owner_receipt()

    inventory = native._gpu_inventory()
    rechecked = native._admissible_gpu(inventory)
    capacity_gate = _contract_gate(
        "capacity_recheck_before_load",
        str(selected_gpu["uuid"]),
        rechecked.get("uuid") if rechecked else None,
        upstream="nvidia-smi",
        field="admissible_gpu_uuid",
        path="nvidia-smi_compute_process_inventory",
    )
    checks.append(capacity_gate)
    if capacity_gate["passed"] is not True:
        lease.transition("terminal_blocked")
        release = lease.release()
        return _blocked_finish(
            root,
            capacity_gate,
            checks,
            hashes,
            affected,
            spans,
            started,
            current_gpu_admission={
                **admission,
                "capacity_recheck_inventory": inventory,
                "lease_release": release,
            },
        )

    runner = native.NativeOptionRunner(
        model_path, int(selected_gpu["index"]), str(selected_gpu["uuid"])
    )
    events: list[JsonDict] = []
    model_identity: JsonDict = {}
    ownership: JsonDict = {}
    collection: JsonDict = {
        "status": "partial",
        "error": "model_load_not_completed",
        "rows": [],
        "events": [],
        "failed_units": [],
        "resumed_group_count": 0,
        "completed_group_count": 0,
        "planned_group_count": CAPTURE_GROUPS,
        "duration_s": 0.0,
    }
    live_error: str | None = None
    inference_ok = False
    release: JsonDict = {}
    try:
        lease.transition("admitted")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu["index"])
        lease.transition("loading")
        phase_started = time.monotonic()
        progress(started, "model_load", "before", gpu_uuid=selected_gpu["uuid"])
        events.append(_event("model_load", "attempted", "qwen-load"))
        try:
            model_identity = native._call_with_heartbeats(
                lambda: runner.load(model_hash=str(context["model_hash"]), owner=owner),
                started=started,
                phase="model_load",
                pending="qwen_gguf_weight_load",
            )
        except BaseException:
            events.append(_event("model_load", "failed", "qwen-load"))
            raise
        events.append(_event("model_load", "completed", "qwen-load"))
        metadata = dict(getattr(runner.llm, "metadata", {}) or {})
        chat_template = metadata.get("tokenizer.chat_template")
        cuda_build = bool(runner.backend.llama_supports_gpu_offload())
        owned_vram = native._owned_vram_mb(os.getpid(), str(selected_gpu["uuid"]))
        block_count = model_identity.get("model_block_count")
        offload_real = isinstance(owned_vram, int) and owned_vram >= 15_000
        observed_layers = int(block_count) if offload_real and isinstance(block_count, int) else 0
        if not chat_template:
            raise EvaluationCaptureError("native_chat_template_missing")
        if not cuda_build or not offload_real or observed_layers <= 0:
            raise EvaluationCaptureError("owned_cuda_offload_not_observed")
        model_identity["native_chat_template"] = {
            "present": True,
            "sha256": native.sha256_text(str(chat_template)),
        }
        model_identity["cuda_build"] = cuda_build
        model_identity["observed_offload"] = {
            "gpu_uuid": selected_gpu["uuid"],
            "owned_pid": os.getpid(),
            "owned_pid_vram_mb": owned_vram,
            "offloaded_layers": observed_layers,
            "passed": True,
        }
        stable_identity = _stable_model_identity(model_identity)
        model_identity_sha256 = canonical_hash(stable_identity)
        model_identity["stable_model_identity_sha256"] = model_identity_sha256
        ownership = {
            **deepcopy(owner),
            "selected_gpu_index": selected_gpu["index"],
            "selected_gpu_uuid": selected_gpu["uuid"],
            "selected_gpu_name": selected_gpu["name"],
            "physical_gpu_uuid": selected_gpu["uuid"],
            "baseline_memory_used_mb": selected_gpu["memory_used_mb"],
            "owned_pid_vram_mb": owned_vram,
            "measured_owned_offloaded_layers": observed_layers,
            "offload_real": True,
            "cuda_build": cuda_build,
            "admission_receipt": admission,
            "capacity_recheck_inventory": inventory,
            "canonical_lease_directory": str(lease_dir),
            "signals_sent": [],
        }
        lease.transition("resident", vram_mb=int(owned_vram or 0))
        lease.transition("inferencing")
        progress(
            started,
            "model_load",
            "after",
            owned_vram_mb=owned_vram,
            offloaded_layers=observed_layers,
        )
        spans.append(_phase_span("model_load", phase_started, started, 1, "qwen_resident"))

        phase_started = time.monotonic()
        progress(started, "test_online_capture", "before_loop", planned=CAPTURE_FORWARDS)
        transport = BoundNativeTransport(runner, model_identity_sha256)
        collection = EvaluationCaptureRunner(
            schedule=schedule,
            transport=transport,
            checkpoint_path=root / CHECKPOINT_PATH,
            owner=owner,
            model_identity_sha256=model_identity_sha256,
        ).run(started=started)
        events.extend(collection["events"])
        inference_ok = collection["status"] == "complete"
        progress(
            started,
            "test_online_capture",
            "after_loop",
            completed_units=len(collection["rows"]),
            completed_groups=collection["completed_group_count"],
            status=collection["status"],
        )
        spans.append(
            _phase_span(
                "test_online_capture",
                phase_started,
                started,
                len(collection["rows"]),
                CHECKPOINT_PATH.as_posix(),
            )
        )
        live_error = collection.get("error")
    except BaseException as exc:  # noqa: BLE001 - owned failure becomes terminal evidence.
        live_error = f"{type(exc).__name__}:{exc}"
        progress(started, "live_measurement", "error", error=live_error)
    finally:
        progress(started, "model_unload", "before")
        runner.close()
        gc.collect()
        progress(started, "model_unload", "after")
        phase = lease.document.get("phase")
        if phase in {"resident", "inferencing"}:
            lease.transition("unloading")
            after_vram = pilot._wait_for_owned_unload(
                os.getpid(), str(selected_gpu["uuid"]), started=started
            )
            unload_observed = after_vram is None or after_vram < native.GPU_IDLE_MAX_USED_MB
            ownership["post_close_owned_vram_mb"] = after_vram
            ownership["model_unload_threshold_mb"] = native.GPU_IDLE_MAX_USED_MB
            ownership["model_unload_observed"] = unload_observed
            lease.transition(
                "validating",
                vram_mb=int(after_vram or 0),
                exit_code=0 if inference_ok else 1,
                unload_observed=unload_observed,
            )
            lease.transition("terminal_complete" if inference_ok else "terminal_blocked")
        elif phase in {"preflight", "admitted", "loading"}:
            lease.transition("terminal_blocked")
        release = lease.release()
        ownership["release"] = release

    rows = [deepcopy(dict(row)) for row in collection.get("rows") or []]
    phase_started = time.monotonic()
    progress(started, "raw_custody", "before_sidecars", completed_units=len(rows))
    raw_manifest = write_raw_manifest(root / RAW_DIR / "capture", schedule, rows)
    reduction = reduce_evaluation_rows(rows, schedule)
    role_sidecars: JsonDict = {}
    baseline: JsonDict = {}
    exposure = build_exposure_audit(
        context["exposure_inventory"],
        dict(context["sealed"].get("artifact") or {})
        .get("exposure_inventory", {})
        .get("events", []),
    )
    if reduction.get("passed") is True and len(rows) == CAPTURE_FORWARDS:
        progress(started, "prediction_freeze", "before_sidecar", completed_units=len(rows))
        predictions, prediction_receipt = freeze_prediction_sidecar(
            root / RAW_DIR / "capture", rows
        )
        baseline = online_baseline(rows)
        progress(
            started,
            "prediction_freeze",
            "after_sidecar",
            completed_units=len(predictions),
            sha256=prediction_receipt["sha256"],
        )
        evaluators = read_evaluator_labels(context, root=root, started=started)
        role_sidecars = write_evaluator_sidecars(
            root / RAW_DIR / "capture", predictions, prediction_receipt, evaluators
        )
        raw_manifest["role_sidecars"] = role_sidecars
    progress(
        started,
        "raw_custody",
        "after_sidecars",
        completed_units=len(rows),
        custody_passed=reduction["passed"],
    )
    spans.append(
        _phase_span(
            "raw_custody",
            phase_started,
            started,
            len(rows),
            "capture_sidecars_complete",
        )
    )
    resumed_forwards = int(collection.get("resumed_group_count") or 0) * 6
    historical_counts = {
        "resumed_forward_count": resumed_forwards,
        "resumed_group_count": int(collection.get("resumed_group_count") or 0),
        "classification": "same_experiment_checkpoint_not_current_process_calls",
    }
    if not ownership:
        ownership = {
            **deepcopy(owner),
            "selected_gpu_uuid": selected_gpu["uuid"],
            "offload_real": False,
            "canonical_lease_directory": str(lease_dir),
            "signals_sent": [],
            "release": release,
        }
    if not model_identity:
        model_identity = {
            "hf_id": MODEL_ID,
            "model_path": str(model_path),
            "model_sha256": context["model_hash"],
            "quantization": "Q4_K_M",
            "runtime": context["runtime"],
            "load_error": live_error,
        }
    completed_ids = {str(row.get("request_id")) for row in rows}
    failed_units = [deepcopy(dict(row)) for row in collection.get("failed_units") or []]
    failed_ids = {str(row.get("request_id")) for row in failed_units}
    unstarted_units = [
        {
            "request_id": request.get("request_id"),
            "group_hash": request.get("group_hash"),
            "role": request.get("role"),
            "disposition": "unstarted",
        }
        for request in schedule
        if str(request.get("request_id")) not in completed_ids | failed_ids
    ]
    artifact = build_complete_artifact(
        preconditions=checks,
        source_hashes=hashes,
        model_specs=[model_identity],
        events=events,
        historical_invocation_counts=historical_counts,
        raw_manifest=raw_manifest,
        reduction=reduction,
        controls=controls,
        ownership=ownership,
        validation_receipts=affected,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        require_validation=False,
        exposure_audit=exposure,
        online_baseline_state=baseline,
        live_error=live_error,
    )
    artifact["capture_schedule_manifest"] = capture_manifest
    artifact["current_gpu_admission"] = admission
    artifact["unit_dispositions"] = {
        "failed": failed_units,
        "unstarted": unstarted_units,
    }
    artifact["field_principles"] = _field_principles(
        (*artifact.keys(), "field_principles", "reproducibility_checksum")
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return _finish_candidate(root, artifact, affected, spans, started)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the live command and two read-only evidence modes."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def _argument_path(path: Path) -> Path:  # pragma: no cover
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run owned capture or inspect one candidate without new model work."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        print(
            json.dumps(
                {
                    "mode": "argument_check",
                    "passed": False,
                    "expected": RUN_DATE,
                    "observed": args.date,
                }
            ),
            flush=True,
        )
        return 1
    if args.cold_replay is not None:
        artifact = _load_object(_argument_path(args.cold_replay))
        errors = validate_artifact(artifact, require_validation=False)
        print(
            json.dumps({"mode": "cold_replay", "passed": not errors, "errors": errors}),
            flush=True,
        )
        return int(bool(errors))
    if args.independent_reduce is not None:
        artifact = _load_object(_argument_path(args.independent_reduce))
        reduction = independent_reduce(artifact)
        print(json.dumps({"mode": "independent_reduce", **reduction}, sort_keys=True), flush=True)
        return int(reduction.get("passed") is not True)
    return run_experiment(REPO_ROOT, args.date)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


def build_blocked_artifact(
    *,
    failed_gate: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Publish external absence with no invented load or forward evidence."""

    artifact = _base_artifact(
        preconditions=preconditions,
        source_hashes=source_hashes,
        duration_s=duration_s,
        phase_spans=phase_spans,
    )
    failed = deepcopy(dict(failed_gate))
    reason = str(failed.get("check") or "external_precondition")
    artifact.update(
        {
            "inference_substrate_class": "blocked_no_run",
            "inference_substrate": "blocked_no_run",
            "planned_inference_substrate_class": "model_load_no_generation",
            "planned_inference_substrate": "live_llm_embedding_extraction",
            "execution_device": "none_precondition_blocked",
            "rows": [],
            "raw_manifest": None,
            "capture_reduction": None,
            "qualification_controls": None,
            "current_invocation_events": [],
            "sample_size_budget": {
                "planned": CAPTURE_FORWARDS,
                "attempted": 0,
                "completed": 0,
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": CAPTURE_FORWARDS,
            },
            "acceptance_gate_results": [failed],
            "gate_check_summary": gate_check_summary([failed]),
            "validation_receipts": [],
            "test_capture_ready_score": 0,
            "online_capture_ready_score": 0,
            "capture_bundle_complete_score": 0,
            "honest_verdict": f"complete_blocked_{reason}",
            "verdict_class": "blocked",
            "complete": 1,
            "ready": 0,
        }
    )
    artifact["field_principles"] = _field_principles(
        (*artifact.keys(), "field_principles", "reproducibility_checksum")
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _event(operation: str, state: str, call_id: str) -> JsonDict:
    return {
        "operation": operation,
        "state": state,
        "call_id": call_id,
        "monotonic_ns": time.monotonic_ns(),
        "scope": "current",
    }


def build_artifact_for_test(output_dir: Path) -> JsonDict:
    """Build a full-size production-shaped fixture outside tracked results."""

    groups: list[JsonDict] = []
    for role, count in ROLE_COUNTS.items():
        for index in range(count):
            group = capture._fixture_group(role, index)
            group["donor_role"] = role
            groups.append(group)
    schedule = compile_evaluation_schedule(groups)
    rows = _fixture_rows(schedule)
    reduction = reduce_evaluation_rows(rows, schedule)
    raw_manifest = write_raw_manifest(output_dir / "raw", schedule, rows)
    evaluators = [
        {
            "component_hash": group["component_hash"],
            "role": group["role"],
            "label": index % 2,
            "arrival_rank": index,
            "label_provenance": "injected_tool_errors",
        }
        for index, group in enumerate(groups)
    ]
    raw_manifest["role_sidecars"] = write_role_sidecars(output_dir / "raw", rows, evaluators)
    events = [_event("model_load", "attempted", "fixture-load")]
    events.append(_event("model_load", "completed", "fixture-load"))
    for index in range(CAPTURE_FORWARDS):
        events.append(_event("forward", "attempted", f"fixture-{index:04d}"))
        events.append(_event("forward", "completed", f"fixture-{index:04d}"))
    controls = run_qualification_controls(output_dir / "controls")
    return build_complete_artifact(
        preconditions=[],
        source_hashes={},
        model_specs=[
            {
                "hf_id": MODEL_ID,
                "model_path": "/cache/Qwen3.8-27B-Q4_K_M.gguf",
                "model_sha256": "sha256:model",
                "quantization": "Q4_K_M",
                "runtime": {"name": "llama_cpp", "version": "fixture"},
            }
        ],
        events=events,
        historical_invocation_counts={},
        raw_manifest=raw_manifest,
        reduction=reduction,
        controls=controls,
        ownership={"device_uuid": "GPU-fixture", "signals_sent": []},
        validation_receipts=[],
        duration_s=3.0,
        phase_spans=[],
        require_validation=False,
        exposure_audit=build_exposure_audit(
            {
                "inventory_sha256": "sha256:fixture",
                "candidate_official_training_groups": 1,
                "exposed_candidate_groups": 1,
                "fresh_eligible_groups": 0,
            },
            [{"labels_consumed": 1}],
        ),
        online_baseline_state=online_baseline(rows),
    )


def validate_artifact(value: object, *, require_validation: bool = True) -> list[str]:
    """Cold-check identity, custody, call accounting, claims, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors: list[str] = []
    if artifact.get("schema") != SCHEMA:
        errors.append("schema_mismatch")
    if artifact.get("experiment_id") != EXPERIMENT_ID or artifact.get("run_date") != RUN_DATE:
        errors.append("identity_mismatch")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict_not_terminal")
    for score in (
        "test_capture_ready_score",
        "online_capture_ready_score",
        "capture_bundle_complete_score",
    ):
        if type(artifact.get(score)) is not int or artifact.get(score) not in {0, 1}:
            errors.append(f"{score}_not_bare_binary")
    if artifact.get("MODEL_SPECS") != MODEL_SPECS:
        errors.append("planned_model_specs_invalid")
    if artifact.get("source_label_access") != []:
        errors.append("capture_label_access_invalid")
    if artifact.get("predictive_benefit_measured") is not False:
        errors.append("predictive_benefit_claimed")
    if artifact.get("readout_kind") != "option_logits":
        errors.append("readout_kind_invalid")
    if artifact.get("role_counts") != ROLE_COUNTS:
        errors.append("role_counts_invalid")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or not principles:
        errors.append("field_principles_missing")
    events = artifact.get("current_invocation_events")
    if not isinstance(events, list):
        errors.append("invocation_evidence_invalid")
    elif artifact.get("invocation_counts") != reduce_invocation_events(events):
        errors.append("invocation_counts_mismatch")
    counts = artifact.get("invocation_counts")
    if isinstance(counts, Mapping):
        generations = counts.get("generations")
        if not isinstance(generations, Mapping) or generations.get("attempted") != 0:
            errors.append("generation_call_detected")
    budget = artifact.get("sample_size_budget")
    if not isinstance(budget, Mapping) or budget.get("planned") != CAPTURE_FORWARDS:
        errors.append("sample_size_budget_invalid")
    gates = artifact.get("acceptance_gate_results")
    if not isinstance(gates, list) or artifact.get("gate_check_summary") != gate_check_summary(
        gates
    ):
        errors.append("gate_check_summary_mismatch")
    if artifact.get("verdict_class") == "blocked":
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate_invalid")
        if artifact.get("planned_inference_substrate_class") != "model_load_no_generation":
            errors.append("blocked_planned_substrate_invalid")
        if artifact.get("rows") != [] or artifact.get("model_specs") != []:
            errors.append("blocked_artifact_fabricated_work")
        if any(
            artifact.get(score) != 0
            for score in (
                "test_capture_ready_score",
                "online_capture_ready_score",
                "capture_bundle_complete_score",
            )
        ):
            errors.append("blocked_artifact_ready")
    else:
        if artifact.get("inference_substrate_class") != "model_load_no_generation":
            errors.append("inference_substrate_class_invalid")
        reduction = artifact.get("capture_reduction")
        if not isinstance(reduction, Mapping):
            errors.append("capture_reduction_missing")
        elif artifact.get("capture_bundle_complete_score") == 1 and not (
            reduction.get("passed") is True
            and reduction.get("complete_group_count") == CAPTURE_GROUPS
            and reduction.get("complete_forward_count") == CAPTURE_FORWARDS
            and reduction.get("role_group_counts") == ROLE_COUNTS
            and reduction.get("role_forward_counts") == {"test": 480, "online": 960}
            and reduction.get("cross_role_donor_count") == 0
        ):
            errors.append("ready_capture_counts_invalid")
        if artifact.get("capture_bundle_complete_score") == 1:
            raw = artifact.get("raw_manifest")
            sidecars = raw.get("role_sidecars") if isinstance(raw, Mapping) else None
            access = artifact.get("evaluator_label_access")
            exposure = artifact.get("exposure_audit")
            baseline = artifact.get("online_baseline")
            if (
                not isinstance(sidecars, Mapping)
                or sidecars.get("label_release_after_prediction_freeze") is not True
            ):
                errors.append("role_sidecar_custody_invalid")
            if not isinstance(access, Mapping) or access.get("after_prediction_freeze") is not True:
                errors.append("evaluator_release_order_invalid")
            if (
                not isinstance(exposure, Mapping)
                or exposure.get("fresh_evidence_claim") is not False
            ):
                errors.append("exposure_audit_invalid")
            if (
                not isinstance(baseline, Mapping)
                or baseline.get("intervention_absence_used_as_label") is not False
            ):
                errors.append("online_baseline_invalid")
    if require_validation and not _validation_passed(artifact.get("validation_receipts")):
        errors.append("required_validation_failed")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors
