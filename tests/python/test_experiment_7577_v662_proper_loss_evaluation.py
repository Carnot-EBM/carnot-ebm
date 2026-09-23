"""Tests for REQ-CL-7577 and SCENARIO-CL-7577-*.

The fixtures are small, but they exercise the same evaluator that scores the
80 exposed groups. This keeps custody failures separate from empirical nulls.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_7577_v662_proper_loss_evaluation as exp
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file


def _role_rows(count: int = 8) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(count):
        source_id = f"source-{index:03d}"
        label = index % 2
        rows.append(
            {
                "source_id": source_id,
                "group_id": f"group-{index:03d}",
                "role": "test",
                "probability": 0.25 + 0.5 * label,
                "label": label,
                "request_hashes": [f"{source_id}-request-{cell}" for cell in range(6)],
                "context_sha256": f"context-{index:03d}",
                "response_sha256": f"response-{index:03d}",
                "diagnostics_are_labels": False,
            }
        )
    return rows


def _protocol(rows: list[dict[str, object]]) -> dict[str, object]:
    source_ids = [str(row["source_id"]) for row in rows]
    payload = {
        "role_counts": {"fit": 160, "tune": 40, "policy": 40, "online": 160, "test": len(rows)},
        "retention_ids": source_ids,
        "role_ids": {"test": source_ids},
        "bootstrap_replays_per_order": 1000,
        "holm_adjusted_static_contrasts": True,
        "minimum_non_escalation_fraction": 0.10,
        "costs": {
            "accept": "5q",
            "reject": "1-q",
            "escalate": 0.2,
            "tie_breaker": "escalate",
        },
        "resampling_unit": "source_component",
    }
    payload["protocol_sha256"] = canonical_hash(
        {
            "role_counts": payload["role_counts"],
            "retention_ids": source_ids,
            "minimum_non_escalation_fraction": 0.10,
        }
    )
    return payload


def _head(name: str, **values: object) -> dict[str, object]:
    head = {"name": name, "converged": True, **values}
    head["head_sha256"] = canonical_hash(head)
    return head


def _head_manifest() -> dict[str, object]:
    knots = [index / 8 for index in range(9)]
    heads = {
        "proper_loss_monotone": _head("proper_loss_monotone", theta=knots, knot_locations=knots),
        "raw_original": _head("raw_original", parameter_count=0),
        "temperature_original": _head("temperature_original", parameter_count=1, temperature=0.5),
        "unconstrained_nine_knot": _head(
            "unconstrained_nine_knot", theta=knots, knot_locations=knots
        ),
    }
    manifest: dict[str, object] = {
        "schema": "carnot.exp7576.v662.head_manifest.v1",
        "frozen_before_policy_access": True,
        "heads": heads,
        "strongest_comparator": {
            "name": "temperature_original",
            "selected_on_role": "tune",
            "frozen_before_policy_access": True,
        },
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    return manifest


def _evaluated(count: int = 8) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = _role_rows(count)
    features, evaluator = exp.split_evaluation_rows(rows, _protocol(rows), expected_groups=count)
    forecasts = exp.score_frozen_heads(features, _head_manifest())
    return exp.evaluate_forecasts(forecasts, evaluator), evaluator


def test_req_cl_7577_custody_separates_features_and_labels() -> None:
    """REQ-CL-7577; SCENARIO-CL-7577-CUSTODY."""

    rows = _role_rows()
    features, evaluator = exp.split_evaluation_rows(rows, _protocol(rows), expected_groups=8)
    assert len(features) == 8
    assert all("label" not in row and "observed_error" not in row for row in features)
    assert evaluator["label_count"] == 8
    assert evaluator["expected_source_ids"] == [row["source_id"] for row in rows]
    assert evaluator["minimum_non_escalation_fraction"] == 0.10


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("duplicate", "duplicate_source_id"),
        ("order", "evaluation_order_map_drift"),
        ("roster", "evaluation_roster_drift"),
        ("mapping", "option_mapping_invalid"),
    ],
)
def test_req_cl_7577_custody_rejects_invalid_rows(mutation: str, match: str) -> None:
    """REQ-CL-7577; SCENARIO-CL-7577-CUSTODY rejects corrupt custody."""

    rows = _role_rows()
    protocol = _protocol(rows)
    if mutation == "duplicate":
        rows[-1]["source_id"] = rows[0]["source_id"]
    elif mutation == "order":
        protocol["retention_ids"] = list(reversed(protocol["retention_ids"]))
    elif mutation == "roster":
        protocol["role_ids"] = {"test": [*protocol["role_ids"]["test"][:-1], "other"]}
    else:
        rows[0]["request_hashes"] = ["only-one"]
    with pytest.raises(ValueError, match=match):
        exp.split_evaluation_rows(rows, protocol, expected_groups=8)


def test_req_cl_7577_heads_are_hash_bound_before_scoring() -> None:
    """REQ-CL-7577; SCENARIO-CL-7577-CUSTODY binds every frozen head."""

    manifest = _head_manifest()
    identity = exp.validate_head_manifest(manifest)
    assert identity["head_count"] == 4
    assert identity["strongest_comparator"] == "temperature_original"
    assert set(identity["head_digests"]) == set(exp.FROZEN_HEAD_ARMS)

    changed = deepcopy(manifest)
    changed["heads"]["temperature_original"]["temperature"] = 2.0
    changed["manifest_sha256"] = canonical_hash(
        {key: value for key, value in changed.items() if key != "manifest_sha256"}
    )
    with pytest.raises(ValueError, match="head_digest_drift"):
        exp.validate_head_manifest(changed)

    changed = deepcopy(manifest)
    changed["heads"].pop("raw_original")
    changed["manifest_sha256"] = canonical_hash(
        {key: value for key, value in changed.items() if key != "manifest_sha256"}
    )
    with pytest.raises(ValueError, match="head_roster_invalid"):
        exp.validate_head_manifest(changed)


def test_req_cl_7577_rows_report_probability_loss_action_and_cost() -> None:
    """REQ-CL-7577; SCENARIO-CL-7577-ROWS keeps all raw operands."""

    rows, _evaluator = _evaluated()
    assert len(rows) == 8 * len(exp.ARMS)
    assert {row["arm"] for row in rows} == set(exp.ARMS)
    raw = next(row for row in rows if row["arm"] == "raw_original")
    assert raw["q"] == raw["probability"]
    assert raw["observed_error"] in (0, 1)
    assert raw["brier"] == pytest.approx((raw["q"] - raw["observed_error"]) ** 2)
    assert raw["log_loss"] == pytest.approx(
        -math.log(raw["q"] if raw["observed_error"] else 1.0 - raw["q"])
    )
    assert raw["raw_brier_denominator"] == 1
    assert raw["raw_log_loss_denominator"] == 1
    assert raw["raw_cost_denominator"] == 1
    assert raw["metric_direction"] == "lower_loss_and_cost_are_better"
    assert raw["censored"] is False
    assert raw["provenance"]["feature_source"] == "exp7575_cached_original_probability"

    constant = next(row for row in rows if row["arm"] == "escalate_all")
    assert constant["q"] == 0.5
    assert constant["action"] == "escalate"
    assert constant["realized_cost"] == 0.2
    assert constant["non_escalated"] is False
    assert constant["brier"] == 0.25
    assert constant["log_loss"] == pytest.approx(math.log(2.0))


def test_req_cl_7577_separate_evaluator_rejects_label_and_mapping_corruption() -> None:
    """REQ-CL-7577; SCENARIO-CL-7577-CONTROLS uses production checks."""

    source = _role_rows()
    features, evaluator = exp.split_evaluation_rows(source, _protocol(source), expected_groups=8)
    forecasts = exp.score_frozen_heads(features, _head_manifest())

    swapped = deepcopy(evaluator)
    swapped["labels"] = list(reversed(swapped["labels"]))
    with pytest.raises(ValueError, match="label_binding_drift"):
        exp.evaluate_forecasts(forecasts, swapped)

    remapped = deepcopy(forecasts)
    remapped[0]["option_mapping_sha256"] = "sha256:changed"
    with pytest.raises(ValueError, match="option_mapping_drift"):
        exp.evaluate_forecasts(remapped, evaluator)

    duplicate = deepcopy(forecasts)
    duplicate[-1]["source_id"] = duplicate[0]["source_id"]
    with pytest.raises(ValueError, match="duplicate_forecast"):
        exp.evaluate_forecasts(duplicate, evaluator)


def test_req_cl_7577_reducer_preserves_both_contrast_signs() -> None:
    """REQ-CL-7577; SCENARIO-CL-7577-GATES reports both directions."""

    rows, evaluator = _evaluated(20)
    settings = exp.registered_settings(evaluator, draws=200)
    reduced = exp.reduce_rows(rows, settings=settings)
    assert reduced["static_measurement_complete_score"] == 1
    assert reduced["paired_intervals"]["unit_definition"] == "source_component"
    assert reduced["paired_intervals"]["draws"] == 200
    for contrast in reduced["paired_intervals"]["brier"].values():
        assert contrast["candidate_minus_control_loss"] == pytest.approx(
            -contrast["positive_is_better_improvement"]
        )
        assert contrast["direction"] == "control_minus_candidate_positive_is_better"
    assert reduced["probability_metrics"]["escalate_all"]["mean_primary_cost"] == 0.2
    assert reduced["probability_metrics"]["escalate_all"]["non_escalation_fraction"] == 0.0
    assert reduced["coverage_interval"]["floor"] == 0.10


def test_req_cl_7577_positive_and_corruption_controls_use_actual_evaluator() -> None:
    """REQ-CL-7577; SCENARIO-CL-7577-CONTROLS detects sensitivity."""

    source = _role_rows(20)
    features, evaluator = exp.split_evaluation_rows(source, _protocol(source), expected_groups=20)
    controls = exp.run_evaluator_controls(features, evaluator)
    assert controls["passed"] is True
    assert controls["positive_control"]["verdict_class"] == "circular_positive"
    assert controls["positive_control"]["row_contrast_supported"] is True
    assert controls["positive_control"]["positive_is_better_improvement"] > 0.0
    assert controls["positive_control"]["verifier_is_oracle"] is True
    assert controls["swapped_labels"]["rejected"] is True
    assert controls["changed_option_mapping"]["rejected"] is True
    assert controls["duplicate_component"]["rejected"] is True


def _preconditions(passed: bool = True) -> list[dict[str, object]]:
    return [
        {
            "check": "upstream_ready",
            "upstream": "exp7576-proper-loss-energy",
            "path": "results/experiment_7576_v662_proper_loss_energy.json",
            "field": "proper_loss_fit_ready_score",
            "op": "eq",
            "expected": 1,
            "observed": 1 if passed else None,
            "passed": passed,
        }
    ]


def _receipts() -> list[dict[str, object]]:
    names = [*exp.REQUIRED_VALIDATION_NAMES, *exp.TERMINAL_CHECK_NAMES]
    return [
        {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "command": ["python", name],
            "worktree": str(exp.REPO_ROOT),
            "log_sha256": f"sha256:{name}",
        }
        for name in names
    ]


def _artifact(*, receipts: bool = False) -> dict[str, object]:
    source = _role_rows(20)
    features, evaluator = exp.split_evaluation_rows(source, _protocol(source), expected_groups=20)
    manifest = _head_manifest()
    rows = exp.evaluate_forecasts(exp.score_frozen_heads(features, manifest), evaluator)
    controls = exp.run_evaluator_controls(features, evaluator)
    spec_path = exp.REPO_ROOT / exp.SPEC_PATH
    return exp.build_artifact(
        rows=rows,
        settings=exp.registered_settings(evaluator, draws=200),
        controls=controls,
        head_identity=exp.validate_head_manifest(manifest),
        preconditions=_preconditions(),
        source_hashes=[
            {
                "path": exp.SPEC_PATH.as_posix(),
                "sha256": sha256_file(spec_path),
                "bytes": spec_path.stat().st_size,
            }
        ],
        validation_receipts=_receipts() if receipts else [],
        raw_sidecars={},
        duration_s=1.25,
    )


def test_req_cl_7577_artifact_keeps_completion_separate_from_benefit() -> None:
    """REQ-CL-7577; SCENARIO-CL-7577-TERMINAL permits a valid null."""

    artifact = _artifact()
    checked = exp.validate_artifact(artifact, require_validation=False)
    assert checked["valid"] is True
    assert artifact["static_measurement_complete_score"] == 1
    assert artifact["fresh_confirmatory_claim_allowed"] is False
    assert artifact["no_deployment_promotion"] is True
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_specs"] == []
    assert artifact["no_model_load"] is True
    assert artifact["inference_substrate"].endswith("_no_llm")
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["invocation_counts"] == exp.ZERO_INVOCATION_COUNTS
    assert artifact["verdict_class"] in {"null", "circular_positive"}
    assert str(artifact["honest_verdict"]).startswith("complete_")
    assert artifact["acceptance_gate_results"]["validity"]["passed"] is True
    assert artifact["acceptance_gate_results"]["readiness"]["passed"] is True
    assert artifact["capability_e2e"]["numbered_runtime_e2e"] == []
    assert artifact["capability_e2e"]["learning_lifecycle"] == [
        "predict",
        "release",
        "update",
        "persist",
        "reload",
    ]
    assert set(exp.REQUIRED_PRINCIPLE_FIELDS) <= set(artifact["field_principles"])


def test_req_cl_7577_artifact_detects_row_and_claim_drift() -> None:
    """REQ-CL-7577; SCENARIO-CL-7577-TERMINAL independently reduces rows."""

    artifact = _artifact()
    changed = deepcopy(artifact)
    changed["rows"][0]["brier"] += 0.1
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    with pytest.raises(ValueError, match="row_brier_invalid"):
        exp.validate_artifact(changed, require_validation=False)

    changed = deepcopy(artifact)
    changed["exploratory_probability_benefit_score"] = 1
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    with pytest.raises(ValueError, match="probability_benefit_mismatch"):
        exp.validate_artifact(changed, require_validation=False)


def test_req_cl_7577_blocked_artifact_names_exact_failed_operand() -> None:
    """REQ-CL-7577 blocked work does not invent substitute evidence."""

    artifact = exp.build_blocked_artifact(_preconditions(False), [], duration_s=0.5)
    checked = exp.validate_artifact(artifact, require_validation=False)
    assert checked == {"valid": True, "blocked": True}
    assert artifact["honest_verdict"] == "complete_blocked_upstream_ready"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["rows"] == []
    assert artifact["static_measurement_complete_score"] == 0
    assert artifact["gate_check_summary"]["first_failure"] == _preconditions(False)[0]
    assert {
        "check",
        "upstream",
        "path",
        "field",
        "op",
        "expected",
        "observed",
    } <= set(artifact["gate_check_summary"]["first_failure"])


def test_req_cl_7577_validation_receipts_are_required_at_publication() -> None:
    """REQ-CL-7577 terminal publication binds every scoped reader."""

    artifact = _artifact()
    with pytest.raises(ValueError, match="validation_receipt_missing"):
        exp.validate_artifact(artifact)

    artifact = _artifact(receipts=True)
    assert exp.validate_artifact(artifact)["valid"] is True
    artifact["validation_receipts"][0]["exit_code"] = 1
    artifact["validation_receipts"][0]["passed"] = False
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
    with pytest.raises(ValueError, match="validation_receipt_failed"):
        exp.validate_artifact(artifact)


def test_req_cl_7577_cold_replay_and_independent_reduction(tmp_path: Path) -> None:
    """REQ-CL-7577; SCENARIO-CL-7577-TERMINAL reads exact bytes."""

    artifact = _artifact(receipts=True)
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.cold_replay(path)["valid"] is True
    reduced = exp.independent_reduce_artifact(path)
    assert reduced["static_measurement_complete_score"] == 1
    assert reduced["paired_intervals"] == artifact["paired_intervals"]


def test_req_cl_7577_actual_preconditions_and_source_hashes() -> None:
    """REQ-CL-7577 authenticates owned and upstream resources before use."""

    checks, hashes = exp.collect_preconditions(exp.REPO_ROOT)
    assert checks
    assert all(row["passed"] is True for row in checks)
    by_check = {row["check"]: row for row in checks}
    assert by_check["exp7576_fit_ready"]["observed"] == 1
    assert by_check["exp7575_cached_roles_ready"]["observed"] == 1
    assert by_check["evaluation_group_count"]["observed"] == 80
    assert by_check["head_manifest_digest"]["passed"] is True
    by_path = {row["path"]: row for row in hashes}
    for relative in (
        "results/experiment_7575_v662_cached_learning_protocol.json",
        "results/experiment_7576_v662_proper_loss_energy.json",
        "results/raw/experiment_7575_v662_cached_learning_protocol/cached_roles.jsonl",
        "results/raw/experiment_7575_v662_cached_learning_protocol/frozen_protocol.json",
        "results/raw/experiment_7576_v662_proper_loss_energy/head_manifest.json",
    ):
        assert by_path[relative]["sha256"] == sha256_file(exp.REPO_ROOT / relative)


def test_req_cl_7577_preconditions_fail_closed_for_missing_root(tmp_path: Path) -> None:
    """REQ-CL-7577 missing external producers become blocked operands."""

    checks, hashes = exp.collect_preconditions(tmp_path)
    assert hashes == []
    assert any(row["passed"] is False for row in checks)
    producer = next(row for row in checks if row["check"] == "exp7575_artifact_exists")
    assert producer["observed"] is False
    assert producer["path"]


def test_req_cl_7577_cli_parser_keeps_root_and_date_explicit(tmp_path: Path) -> None:
    """REQ-CL-7577 thin CLI supports producer and fresh-reader modes."""

    args = exp.parse_args(["--root", str(tmp_path), "--date", exp.RUN_DATE])
    assert args.root == tmp_path
    assert args.date == exp.RUN_DATE
    assert args.cold_replay is None
    assert args.independent_reduce is None

    with pytest.raises(SystemExit):
        exp.parse_args(["--cold-replay", "one", "--independent-reduce", "two"])


def test_req_cl_7577_low_level_custody_guards(tmp_path: Path) -> None:
    """REQ-CL-7577 exercises malformed bytes, heads, roles, and forecasts."""

    not_object = tmp_path / "not-object.json"
    not_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object_required"):
        exp._load_object(not_object)
    external = tmp_path / "source.json"
    external.write_text("{}", encoding="utf-8")
    assert exp._source_hash(external, exp.REPO_ROOT)["path"] == str(external)

    manifest = _head_manifest()
    outer_changed = deepcopy(manifest)
    outer_changed["schema"] = "changed"
    with pytest.raises(ValueError, match="head_manifest_digest_drift"):
        exp.validate_head_manifest(outer_changed)
    freeze_changed = deepcopy(manifest)
    freeze_changed["frozen_before_policy_access"] = False
    freeze_changed["manifest_sha256"] = canonical_hash(
        {key: value for key, value in freeze_changed.items() if key != "manifest_sha256"}
    )
    with pytest.raises(ValueError, match="head_freeze_contract_invalid"):
        exp.validate_head_manifest(freeze_changed)

    source = _role_rows()
    protocol = _protocol(source)
    with pytest.raises(ValueError, match="evaluation_group_count_invalid"):
        exp.split_evaluation_rows(source[:-1], _protocol(source[:-1]), expected_groups=8)
    digest_protocol = deepcopy(protocol)
    digest_protocol["hash_payload"] = {"registered": True}
    digest_protocol["protocol_sha256"] = "sha256:changed"
    with pytest.raises(ValueError, match="protocol_digest_drift"):
        exp.split_evaluation_rows(source, digest_protocol, expected_groups=8)
    invalid_label = deepcopy(source)
    invalid_label[0]["label"] = 2
    with pytest.raises(ValueError, match="evaluation_label_invalid"):
        exp.split_evaluation_rows(invalid_label, protocol, expected_groups=8)
    invalid_probability = deepcopy(source)
    invalid_probability[0]["probability"] = 2.0
    with pytest.raises(ValueError, match="evaluation_probability_invalid"):
        exp.split_evaluation_rows(invalid_probability, protocol, expected_groups=8)
    with pytest.raises(ValueError, match="temperature_probability_invalid"):
        exp._temperature_probability(0.5, 0.0)

    features, evaluator = exp.split_evaluation_rows(source, protocol, expected_groups=8)
    leaked = deepcopy(features)
    leaked[0]["label"] = 0
    with pytest.raises(ValueError, match="feature_label_leakage"):
        exp.score_frozen_heads(leaked, manifest)
    duplicated = [*deepcopy(features), deepcopy(features[0])]
    with pytest.raises(ValueError, match="duplicate_feature"):
        exp.score_frozen_heads(duplicated, manifest)
    invalid_feature = deepcopy(features)
    invalid_feature[0]["p_original"] = math.nan
    with pytest.raises(ValueError, match="feature_probability_invalid"):
        exp.score_frozen_heads(invalid_feature, manifest)

    forecasts = exp.score_frozen_heads(features, manifest)
    bad = deepcopy(evaluator)
    bad["expected_source_ids"] = list(reversed(bad["expected_source_ids"]))
    with pytest.raises(ValueError, match="label_roster_drift"):
        exp.evaluate_forecasts(forecasts, bad)
    bad = deepcopy(evaluator)
    bad["option_mapping_bindings"][0]["option_mapping_sha256"] = "changed"
    with pytest.raises(ValueError, match="option_mapping_binding_drift"):
        exp.evaluate_forecasts(forecasts, bad)
    bad = deepcopy(evaluator)
    bad["labels"][0]["observed_error"] = 2
    bad["label_binding_sha256"] = canonical_hash(bad["labels"])
    with pytest.raises(ValueError, match="evaluation_label_invalid"):
        exp.evaluate_forecasts(forecasts, bad)
    bad_forecasts = deepcopy(forecasts)
    bad_forecasts[0]["arm"] = "unknown"
    with pytest.raises(ValueError, match="forecast_roster_invalid"):
        exp.evaluate_forecasts(bad_forecasts, evaluator)
    bad_forecasts = deepcopy(forecasts)
    bad_forecasts[0]["q"] = 2.0
    with pytest.raises(ValueError, match="forecast_probability_invalid"):
        exp.evaluate_forecasts(bad_forecasts, evaluator)
    with pytest.raises(ValueError, match="forecast_roster_invalid"):
        exp.evaluate_forecasts(forecasts[:-1], evaluator)
    with pytest.raises(ValueError, match="registered_settings_invalid"):
        exp.registered_settings(evaluator, draws=-1)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("status", "failed", "row_disposition_invalid"),
        ("log_loss", 99.0, "row_log_loss_invalid"),
        ("raw_brier_denominator", 2, "row_brier_arithmetic_invalid"),
        ("raw_log_loss_denominator", 2, "row_log_loss_arithmetic_invalid"),
        ("action", "accept", "row_decision_invalid"),
        ("raw_cost_denominator", 2, "row_cost_arithmetic_invalid"),
        ("metric_direction", "higher_is_better", "row_direction_invalid"),
    ],
)
def test_req_cl_7577_row_mutation_guards(field: str, value: object, match: str) -> None:
    """REQ-CL-7577 raw row guards each reject their named mutation."""

    rows, evaluator = _evaluated()
    rows[0][field] = value
    with pytest.raises(ValueError, match=match):
        exp.reduce_rows(rows, settings=exp.registered_settings(evaluator, draws=20))


def test_req_cl_7577_reducer_rejects_roster_and_interval_drift() -> None:
    """REQ-CL-7577 keeps cluster units complete before interval construction."""

    rows, evaluator = _evaluated()
    settings = exp.registered_settings(evaluator, draws=20)
    duplicated = deepcopy(rows)
    duplicated.append(deepcopy(duplicated[0]))
    with pytest.raises(ValueError, match="row_identity_invalid"):
        exp.reduce_rows(duplicated, settings=settings)
    with pytest.raises(ValueError, match="expected_group_count_mismatch"):
        exp.reduce_rows(rows, settings={**settings, "expected_groups": 9})
    incomplete = [row for row in rows if row is not rows[0]]
    with pytest.raises(ValueError, match="arm_roster_invalid"):
        exp.reduce_rows(incomplete, settings=settings)
    disagreement = deepcopy(rows)
    constant = next(row for row in disagreement if row["arm"] == "escalate_all")
    constant["observed_error"] = 1 - constant["observed_error"]
    with pytest.raises(ValueError, match="group_label_disagreement"):
        exp.reduce_rows(disagreement, settings=settings)
    with pytest.raises(ValueError, match="paired_interval_shape_invalid"):
        exp._paired_interval([], np.zeros((1, 0), dtype=np.int64))
    with pytest.raises(ValueError, match="holm_family_empty"):
        exp._holm_intervals({}, alpha=0.05)


def test_req_cl_7577_private_row_io_loads_authenticated_inputs(tmp_path: Path) -> None:
    """REQ-CL-7577 raw receipts retain exact rows and real upstream operands."""

    path = tmp_path / "rows.jsonl"
    receipt = exp._write_jsonl(path, [{"source": "a"}, {"source": "b"}], tmp_path)
    assert receipt["rows"] == 2
    assert receipt["sha256"] == sha256_file(path)
    role_rows, protocol, manifest = exp._load_evaluation_inputs(exp.REPO_ROOT)
    assert len(role_rows) == exp.EXPECTED_GROUPS
    assert protocol["role_counts"]["test"] == exp.EXPECTED_GROUPS
    assert exp.validate_head_manifest(manifest)["head_count"] == len(exp.FROZEN_HEAD_ARMS)


def test_req_cl_7577_build_classifies_disqualified_and_positive() -> None:
    """REQ-CL-7577 classification is driven by controls and retained row effects."""

    source = _role_rows(20)
    features, evaluator = exp.split_evaluation_rows(source, _protocol(source), expected_groups=20)
    manifest = _head_manifest()
    controls = exp.run_evaluator_controls(features, evaluator)
    forecasts = exp.score_frozen_heads(features, manifest)
    labels = {row["source_id"]: row["observed_error"] for row in evaluator["labels"]}
    for forecast in forecasts:
        if forecast["arm"] == "proper_loss_monotone":
            forecast["q"] = 0.01 if labels[forecast["source_id"]] == 0 else 0.99
    rows = exp.evaluate_forecasts(forecasts, evaluator)
    arguments = {
        "rows": rows,
        "settings": exp.registered_settings(evaluator, draws=500),
        "controls": controls,
        "head_identity": exp.validate_head_manifest(manifest),
        "preconditions": _preconditions(),
        "source_hashes": [],
        "validation_receipts": [],
        "raw_sidecars": {},
        "duration_s": 0.1,
    }
    positive = exp.build_artifact(**arguments)
    assert positive["verdict_class"] == "positive"
    assert exp.validate_artifact(positive, require_validation=False)["valid"] is True
    invalid_controls = deepcopy(controls)
    invalid_controls["passed"] = False
    disqualified = exp.build_artifact(**{**arguments, "controls": invalid_controls})
    assert disqualified["verdict_class"] == "disqualified"


def test_req_cl_7577_reader_rejects_terminal_envelope_mutations() -> None:
    """REQ-CL-7577 cold validation rejects every terminal claim-boundary drift."""

    def rejected(mutator: Callable[[dict[str, object]], None], match: str) -> None:
        artifact = _artifact(receipts=True)
        mutator(artifact)
        artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
        with pytest.raises(ValueError, match=match):
            exp.validate_artifact(artifact)

    rejected(lambda row: row.update(schema="changed"), "artifact_identity_invalid")
    artifact = _artifact(receipts=True)
    artifact["reproducibility_checksum"] = "changed"
    with pytest.raises(ValueError, match="artifact_checksum_invalid"):
        exp.validate_artifact(artifact)
    rejected(lambda row: row.update(MODEL_SPECS=["forbidden"]), "model_specs_not_empty")
    rejected(lambda row: row.update(no_model_load=False), "current_model_activity_invalid")
    rejected(
        lambda row: row.update(fresh_confirmatory_claim_allowed=True),
        "claim_boundary_invalid",
    )
    rejected(lambda row: row.update(verdict_class="unknown"), "verdict_class_invalid")
    rejected(lambda row: row.update(honest_verdict="null"), "terminal_prefix_invalid")
    rejected(
        lambda row: row["field_principles"].pop("paired_intervals"),
        "field_principles_incomplete",
    )

    blocked = exp.build_blocked_artifact(_preconditions(False), [], duration_s=0.1)
    blocked["rows"] = [{"invented": True}]
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    with pytest.raises(ValueError, match="blocked_measurement_invalid"):
        exp.validate_artifact(blocked, require_validation=False)
    blocked = exp.build_blocked_artifact(_preconditions(False), [], duration_s=0.1)
    blocked["gate_check_summary"]["first_failure"].pop("upstream")
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    with pytest.raises(ValueError, match="blocked_gate_summary_invalid"):
        exp.validate_artifact(blocked, require_validation=False)

    rejected(
        lambda row: row["validation_receipts"][0].update(command=[]),
        "validation_receipt_incomplete",
    )
    rejected(
        lambda row: row["source_artifact_hashes"][0].update(sha256="changed"),
        "source_hash_invalid",
    )
    rejected(
        lambda row: row["raw_sidecars"].update(
            missing={"path": "missing.json", "sha256": "changed"}
        ),
        "sidecar_hash_invalid",
    )


def test_req_cl_7577_reader_rejects_reduction_and_gate_mutations() -> None:
    """REQ-CL-7577 independent reduction owns metrics, effects, and gate states."""

    def rejected(mutator: Callable[[dict[str, object]], None], match: str) -> None:
        artifact = _artifact()
        mutator(artifact)
        artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
        with pytest.raises(ValueError, match=match):
            exp.validate_artifact(artifact, require_validation=False)

    rejected(lambda row: row["probability_metrics"].clear(), "probability_metrics_mismatch")
    rejected(lambda row: row["paired_intervals"].clear(), "paired_intervals_mismatch")
    rejected(lambda row: row["coverage_interval"].clear(), "coverage_interval_mismatch")
    rejected(
        lambda row: row.update(exploratory_decision_benefit_score=9),
        "decision_benefit_mismatch",
    )
    rejected(
        lambda row: row["evaluator_controls"].update(controls_sha256="changed"),
        "evaluator_controls_invalid",
    )
    rejected(
        lambda row: row.update(static_measurement_complete_score=0),
        "static_measurement_incomplete",
    )
    rejected(lambda row: row.update(verdict_class="positive"), "verdict_reduction_mismatch")
    rejected(
        lambda row: row["acceptance_gate_results"]["validity"].update(passed=False),
        "acceptance_gate_mismatch",
    )
    rejected(lambda row: row.update(flagged_adversarial=True), "flagged_artifact_not_usable")


def test_req_cl_7577_preconditions_reduce_malformed_external_bytes(tmp_path: Path) -> None:
    """REQ-CL-7577 malformed producers and sidecars become explicit failed gates."""

    results = tmp_path / "results"
    results.mkdir()
    protocol_artifact = tmp_path / exp.PROTOCOL_ARTIFACT_PATH
    fit_artifact = tmp_path / exp.FIT_ARTIFACT_PATH
    protocol_artifact.write_text("[]", encoding="utf-8")
    fit_artifact.write_text("[]", encoding="utf-8")
    checks, _hashes = exp.collect_preconditions(tmp_path)
    assert (
        next(row for row in checks if row["check"] == "exp7575_cached_roles_ready")["passed"]
        is False
    )

    raw = results / "raw"
    raw.mkdir()
    cached = raw / "cached.jsonl"
    protocol = raw / "protocol.json"
    heads = raw / "heads.json"
    cached.write_text("[]\n", encoding="utf-8")
    protocol.write_text("[]", encoding="utf-8")
    heads.write_text("{}", encoding="utf-8")
    protocol_artifact.write_text(
        json.dumps(
            {
                "cached_roles_ready_score": 1,
                "verdict_class": "null",
                "flagged_adversarial": False,
                "raw_sidecars": {
                    "cached_roles": {
                        "path": "results/raw/cached.jsonl",
                        "sha256": sha256_file(cached),
                    },
                    "frozen_protocol": {
                        "path": "results/raw/protocol.json",
                        "sha256": sha256_file(protocol),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    fit_artifact.write_text(
        json.dumps(
            {
                "proper_loss_fit_ready_score": 1,
                "baseline_ready_score": 1,
                "verdict_class": "null",
                "flagged_adversarial": False,
                "raw_sidecars": {
                    "head_manifest": {
                        "path": "results/raw/heads.json",
                        "sha256": sha256_file(heads),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    checks, _hashes = exp.collect_preconditions(tmp_path)
    by_check = {row["check"]: row for row in checks}
    assert by_check["evaluation_group_count"]["observed"] == 0
    assert by_check["protocol_digest_and_roster"]["observed"] is False
    assert by_check["head_manifest_digest"]["observed"] == "invalid"


def test_req_cl_7577_control_guard_records_unrejected_corruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-7577 controls fail readiness if the evaluator stops rejecting drift."""

    source = _role_rows()
    features, evaluator = exp.split_evaluation_rows(source, _protocol(source), expected_groups=8)
    actual_evaluator = exp.evaluate_forecasts
    calls = 0

    def permissive(*args: object, **kwargs: object) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return actual_evaluator(*args, **kwargs)  # type: ignore[arg-type]
        return []

    monkeypatch.setattr(exp, "evaluate_forecasts", permissive)
    controls = exp.run_evaluator_controls(features, evaluator)
    assert controls["passed"] is False
    assert controls["swapped_labels"]["rejected"] is False
