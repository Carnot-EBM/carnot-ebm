"""Tests for the sealed V648 calibrated-decision training run.

Spec refs: REQ-AUTO-7385 and SCENARIO-AUTO-7385-01 through
SCENARIO-AUTO-7385-05.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_7385_v648_decision_training as exp


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_rows(partition: str = "training") -> list[dict[str, object]]:
    return [
        {
            "source_row_index": 0,
            "group_id": "g0",
            "partition": partition,
            "entity_uptake": 0.0,
            "falsifiability_score": 0.0,
            "label": 0,
        },
        {
            "source_row_index": 1,
            "group_id": "g1",
            "partition": partition,
            "entity_uptake": 0.2,
            "falsifiability_score": 0.1,
            "label": 0,
        },
        {
            "source_row_index": 2,
            "group_id": "g2",
            "partition": partition,
            "entity_uptake": 0.8,
            "falsifiability_score": 0.7,
            "label": 1,
        },
        {
            "source_row_index": 3,
            "group_id": "g3",
            "partition": partition,
            "entity_uptake": 1.0,
            "falsifiability_score": 1.0,
            "label": 1,
        },
    ]


def _numeric_unit(arm: str = "training_prevalence", seed: int = exp.TRAINING_SEEDS[0]):
    state = exp.train_arm(arm, seed, _tiny_rows(), steps=3)
    state["affine"] = {"slope": 1.0, "intercept": 0.0, "update_count": 0}
    state["selected_policy"] = {
        "accept_threshold": 0.05,
        "reject_threshold": 0.99,
        "accept_enabled": True,
        "reject_enabled": False,
        "threshold_index": 4,
    }
    return exp.scoring_unit(state)


def test_protocol_authentication_recomputes_every_sealed_boundary() -> None:
    """SCENARIO-AUTO-7385-01 rejects changed terminal and partition evidence."""

    root = Path(__file__).resolve().parents[2]
    protocol_path = root / exp.PROTOCOL_PATH
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    observed_hashes = {
        exp.PROTOCOL_PATH.as_posix(): _sha256(protocol_path),
        exp.CORPUS_PATH.as_posix(): _sha256(root / exp.CORPUS_PATH),
        exp.TRUSTED_LABEL_PATH.as_posix(): _sha256(root / exp.TRUSTED_LABEL_PATH),
    }
    assert exp.authenticate_protocol(protocol, observed_hashes) == []

    mutations = (
        ("status", "partial"),
        ("verdict_class", "disqualified"),
        ("flagged_adversarial", True),
        ("decision_protocol_ready_score", 0),
    )
    for field, value in mutations:
        changed = deepcopy(protocol)
        changed[field] = value
        errors = exp.authenticate_protocol(changed, observed_hashes)
        assert any(field in error for error in errors)

    changed = deepcopy(protocol)
    changed["gate_check_summary"]["all_required_passed"] = False
    assert any(
        "all_required_passed" in error
        for error in exp.authenticate_protocol(changed, observed_hashes)
    )
    changed = deepcopy(protocol)
    changed["partition_membership"][0]["partition"] = "training"
    assert any(
        "partition_membership_sha256" in error
        for error in exp.authenticate_protocol(changed, observed_hashes)
    )
    changed_hashes = dict(observed_hashes)
    changed_hashes[exp.PROTOCOL_PATH.as_posix()] = "sha256:" + "0" * 64
    assert any(
        "artifact_sha256" in error for error in exp.authenticate_protocol(protocol, changed_hashes)
    )


def test_all_frozen_arms_train_and_preserve_numeric_optimizer_state() -> None:
    """SCENARIO-AUTO-7385-02 fits every arm from training rows and records updates."""

    for arm in exp.ARMS:
        state = exp.train_arm(arm, exp.TRAINING_SEEDS[0], _tiny_rows(), steps=4)
        assert state["arm"] == arm
        assert state["seed"] == exp.TRAINING_SEEDS[0]
        assert state["training_partition"] == "training"
        assert state["update_count"] == (0 if arm == "training_prevalence" else 4)
        assert state["pre_weight_sha256"].startswith("sha256:")
        assert state["post_weight_sha256"].startswith("sha256:")
        assert state["loss_curve"]
        assert all(math.isfinite(point["loss"]) for point in state["loss_curve"])
        exp.validate_numeric_tree(state["weights"])
        exp.validate_numeric_tree(state["optimizer_state"])

    with pytest.raises(ValueError, match="training rows only"):
        exp.train_arm("l2_logistic_calibration", 1, _tiny_rows("final_test"), steps=2)
    with pytest.raises(ValueError, match="both labels"):
        exp.train_arm("l2_logistic_calibration", 1, _tiny_rows()[:2], steps=2)
    with pytest.raises(ValueError, match="unknown arm"):
        exp.train_arm("invented", 1, _tiny_rows(), steps=2)


def test_gibbs_energy_matches_independent_numpy_and_checkpoint_is_plain_numeric(
    tmp_path: Path,
) -> None:
    """SCENARIO-AUTO-7385-02 keeps a replayable numeric checkpoint."""

    state = exp.train_arm("natural_prevalence_bernoulli_gibbs", 7382001, _tiny_rows(), steps=5)
    parity = exp.direct_numpy_energy_check(state, _tiny_rows()[:3])
    assert parity["passed"] is True
    assert parity["max_abs_delta"] <= 1e-6
    checkpoint = exp.checkpoint_payload(state, partition_hash="sha256:partition")
    path = tmp_path / "checkpoint.json"
    exp.atomic_json(path, checkpoint)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["architecture"] == {"input_dim": 2, "hidden_dims": [4], "output_dim": 1}
    assert loaded["partition_membership_sha256"] == "sha256:partition"
    assert exp.checkpoint_hash(loaded).startswith("sha256:")


def test_affine_calibration_and_policy_use_their_own_roles() -> None:
    """SCENARIO-AUTO-7385-02 and -03 keep calibration and policy roles separate."""

    calibration = _tiny_rows("probability_calibration")
    logits = [-2.0, -1.0, 1.0, 2.0]
    affine = exp.fit_affine_transform(logits, [0, 0, 1, 1], steps=20)
    assert affine["update_count"] == 20
    assert all(math.isfinite(affine[key]) for key in ("slope", "intercept"))
    with pytest.raises(ValueError, match="both labels"):
        exp.fit_affine_transform(logits[:2], [0, 0], steps=2)

    policy_rows = _tiny_rows("policy_calibration")
    probabilities = [
        {**row, "probability": probability}
        for row, probability in zip(policy_rows, (0.001, 0.02, 0.8, 0.999), strict=True)
    ]
    rows, selected = exp.select_policy(probabilities)
    assert len(rows) == len(exp.THRESHOLD_PAIRS)
    assert selected["selection_partition"] == "policy_calibration"
    assert selected["selection_rule"] == "coverage_then_utility_then_registered_order"
    assert selected["threshold_index"] in range(len(exp.THRESHOLD_PAIRS))
    assert all(row["simultaneous_test_count"] == 250 for row in rows)

    duplicate = [*probabilities, {**probabilities[0], "source_row_index": 99, "label": 1}]
    representatives = exp.label_blind_group_representatives(duplicate)
    assert len(representatives) == 4
    assert next(row for row in representatives if row["group_id"] == "g0")["source_row_index"] == 0
    with pytest.raises(ValueError, match="policy-calibration"):
        exp.select_policy(_tiny_rows("training"))


def test_trusted_score_request_rejects_non_numeric_or_unsealed_state() -> None:
    """SCENARIO-AUTO-7385-04 rejects unsafe state before label access."""

    unit = _numeric_unit()
    request = {
        "schema": exp.SCORING_REQUEST_SCHEMA,
        "protocol_sha256": exp.EXPECTED_PROTOCOL_SHA256,
        "partition_membership_sha256": "sha256:partition",
        "trusted_label_sha256": "sha256:labels",
        "units": [unit],
    }
    expected_units = {f"{unit['arm']}:{unit['seed']}"}
    assert (
        exp.validate_scoring_request(
            request,
            protocol_sha256=exp.EXPECTED_PROTOCOL_SHA256,
            partition_hash="sha256:partition",
            trusted_label_hash="sha256:labels",
            expected_units=expected_units,
        )
        == []
    )

    bad_cases = []
    extra = deepcopy(request)
    extra["code"] = "print('unsafe')"
    bad_cases.append(extra)
    missing = deepcopy(request)
    missing.pop("protocol_sha256")
    bad_cases.append(missing)
    wrong_hash = deepcopy(request)
    wrong_hash["partition_membership_sha256"] = "sha256:wrong"
    bad_cases.append(wrong_hash)
    nonfinite = deepcopy(request)
    nonfinite["units"][0]["weights"]["probability"] = float("nan")
    bad_cases.append(nonfinite)
    duplicate = deepcopy(request)
    duplicate["units"].append(deepcopy(duplicate["units"][0]))
    bad_cases.append(duplicate)
    for bad in bad_cases:
        assert exp.validate_scoring_request(
            bad,
            protocol_sha256=exp.EXPECTED_PROTOCOL_SHA256,
            partition_hash="sha256:partition",
            trusted_label_hash="sha256:labels",
            expected_units=expected_units,
        )


def test_final_scoring_emits_complete_rows_and_metrics() -> None:
    """REQ-AUTO-7385 emits every registered loss, action, risk, and cost field."""

    unit = _numeric_unit()
    features = [
        {key: value for key, value in row.items() if key != "label"} | {"partition": "final_test"}
        for row in _tiny_rows()
    ]
    labels = [
        {
            "source_row_index": row["source_row_index"],
            "group_id": row["group_id"],
            "label": row["label"],
        }
        for row in _tiny_rows()
    ]
    rows, metrics = exp.score_final_rows([unit], features, labels)
    assert len(rows) == 4
    required = {
        "arm",
        "seed",
        "group_id",
        "label",
        "raw_energy",
        "probability",
        "decision",
        "brier_contribution",
        "log_loss_contribution",
        "correctness",
        "risk",
        "measured_cost",
    }
    assert all(required <= set(row) for row in rows)
    summary = metrics[f"{unit['arm']}:{unit['seed']}"]
    assert summary["effective_groups"] == 4
    assert 0.0 <= summary["brier"] <= 1.0
    assert 0.0 <= summary["auroc"] <= 1.0
    assert 0.0 <= summary["pr_auc"] <= 1.0
    with pytest.raises(ValueError, match="label identity"):
        exp.score_final_rows([unit], features, labels[:-1])


def test_pr_auc_counts_tied_constant_scores_at_prevalence() -> None:
    """REQ-AUTO-7385 reports a tie-stable PR-AUC for the prevalence control."""

    assert exp._binary_pr_auc([0, 0, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.25)


def test_paired_group_intervals_and_registered_value_conjunction_are_deterministic() -> None:
    """SCENARIO-AUTO-7385-05 keeps the complete registered conjunction explicit."""

    rows = []
    for index in range(8):
        group = f"g{index}"
        for arm, brier, log_loss, decision in (
            ("natural_prevalence_bernoulli_gibbs", 0.05, 0.20, "accept"),
            ("training_prevalence", 0.20, 0.40, "accept"),
            ("l2_logistic_calibration", 0.15, 0.35, "accept"),
        ):
            rows.append(
                {
                    "arm": arm,
                    "seed": 7382001,
                    "group_id": group,
                    "brier_contribution": brier,
                    "log_loss_contribution": log_loss,
                    "decision": decision,
                    "risk": {"harmful_action": False},
                }
            )
    first = exp.paired_group_intervals(rows, draws=100, seed=7382307)
    second = exp.paired_group_intervals(rows, draws=100, seed=7382307)
    assert first == second
    conjunction = exp.reduce_calibration_value(
        paired_intervals=first,
        metrics={
            "natural_prevalence_bernoulli_gibbs": {
                "log_loss": 0.20,
                "coverage": 1.0,
                "incorrect_accept_risk": 0.0,
                "correct_reject_risk": None,
            },
            "training_prevalence": {"log_loss": 0.40, "coverage": 1.0},
            "l2_logistic_calibration": {"log_loss": 0.35, "coverage": 1.0},
        },
        policy_certified=True,
    )
    assert conjunction["calibration_value_score"] == 1
    failed = exp.reduce_calibration_value(
        paired_intervals=first,
        metrics={
            "natural_prevalence_bernoulli_gibbs": {
                "log_loss": 0.50,
                "coverage": 0.10,
                "incorrect_accept_risk": 0.20,
                "correct_reject_risk": None,
            },
            "training_prevalence": {"log_loss": 0.40, "coverage": 1.0},
            "l2_logistic_calibration": {"log_loss": 0.35, "coverage": 1.0},
        },
        policy_certified=False,
    )
    assert failed["calibration_value_score"] == 0
    assert failed["passed"] is False


def test_artifact_reducer_keeps_capture_independent_of_benefit() -> None:
    """SCENARIO-AUTO-7385-05 treats complete null science as terminal evidence."""

    artifact = exp.build_fixture_artifact()
    assert exp.validate_artifact(artifact) == []
    assert artifact["decision_capture_complete_score"] == 1
    assert artifact["calibration_value_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["promotion_score"] == 0
    assert artifact["inference_substrate"]["value"]
    assert set(artifact["field_principles"]) == set(artifact)

    changed = deepcopy(artifact)
    changed["model_invoked"] = True
    assert "current_model_declaration_mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["rows"].pop()
    assert "row_completeness_mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["calibration_value_score"] = 1
    assert "calibration_value_reduction_mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:changed"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed)


def test_blocked_artifact_names_exact_failed_gate_and_has_no_dependent_work() -> None:
    """SCENARIO-AUTO-7385-01 emits a terminal blocked record, never a partial."""

    checks = [
        {
            "check": "upstream_flag",
            "upstream": exp.PROTOCOL_PATH.as_posix(),
            "artifact_field": "flagged_adversarial",
            "expected": False,
            "observed": True,
            "passed": False,
        }
    ]
    artifact = exp.build_blocked_artifact(checks, {})
    assert artifact["status"].startswith("blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["first_required_failure"] == checks[0]
    assert artifact["rows"] == []
    assert artifact["validation_receipts"] == []
    assert artifact["decision_capture_complete_score"] == 0
    assert artifact["calibration_value_score"] == 0
    assert exp.validate_artifact(artifact) == []


def test_rejection_edges_and_pure_reducers_are_covered(tmp_path: Path) -> None:
    """REQ-AUTO-7385 fail-closes malformed numeric, metric, and request inputs."""

    payload_path = tmp_path / "payload.bin"
    payload_path.write_bytes(b"payload")
    assert exp.sha256_file(payload_path) == "sha256:" + hashlib.sha256(b"payload").hexdigest()
    for value, message in (
        (True, "booleans"),
        ({1: 2.0}, "keys"),
        ("code", "non-numeric"),
    ):
        with pytest.raises(ValueError, match=message):
            exp.validate_numeric_tree(value)

    bad_features = _tiny_rows()
    bad_features[0]["entity_uptake"] = float("nan")
    with pytest.raises(ValueError, match="finite 2-vectors"):
        exp.train_arm("l2_logistic_calibration", 1, bad_features, steps=2)
    with pytest.raises(ValueError, match="between zero"):
        exp.train_arm("l2_logistic_calibration", 1, _tiny_rows(), steps=501)
    assert exp._json_tree(np.int64(3)) == 3
    assert exp._json_tree("unchanged") == "unchanged"

    logistic = exp.train_arm("l2_logistic_calibration", 7382001, _tiny_rows(), steps=2)
    assert math.isfinite(exp._raw_energy(logistic, [0.1, 0.2]))
    corrected = exp.train_arm("prior_corrected_nce_gibbs", 7382001, _tiny_rows(), steps=2)
    corrected["training_prevalence"] = 0.25
    assert not math.isclose(
        exp._base_logit(corrected, [0.1, 0.2]),
        exp._raw_energy(corrected, [0.1, 0.2]),
    )
    assert exp.direct_numpy_energy_check(logistic, _tiny_rows())["performed"] is False
    with pytest.raises(ValueError, match="common length"):
        exp.fit_affine_transform([], [])
    with pytest.raises(ValueError, match="finite"):
        exp.fit_affine_transform([float("nan"), 0.0], [0, 1], steps=1)

    assert exp.validate_scoring_request(
        [],
        protocol_sha256="p",
        partition_hash="m",
        trusted_label_hash="l",
        expected_units=set(),
    ) == ["request_not_object"]
    assert "units_not_list" in exp.validate_scoring_request(
        {
            "schema": exp.SCORING_REQUEST_SCHEMA,
            "protocol_sha256": "p",
            "partition_membership_sha256": "m",
            "trusted_label_sha256": "l",
            "units": {},
        },
        protocol_sha256="p",
        partition_hash="m",
        trusted_label_hash="l",
        expected_units=set(),
    )
    assert "unit_set_mismatch" in exp.validate_scoring_request(
        {
            "schema": exp.SCORING_REQUEST_SCHEMA,
            "protocol_sha256": "p",
            "partition_membership_sha256": "m",
            "trusted_label_sha256": "l",
            "units": [],
        },
        protocol_sha256="p",
        partition_hash="m",
        trusted_label_hash="l",
        expected_units={"missing:1"},
    )
    unit = _numeric_unit()
    request = {
        "schema": exp.SCORING_REQUEST_SCHEMA,
        "protocol_sha256": "p",
        "partition_membership_sha256": "m",
        "trusted_label_sha256": "l",
        "units": [{"bad": 1}, unit],
    }
    request["units"][1]["seed"] = 1
    request["units"][1]["weights"] = {"probability": [0.5, 0.6]}
    errors = exp.validate_scoring_request(
        request,
        protocol_sha256="p",
        partition_hash="m",
        trusted_label_hash="l",
        expected_units={"training_prevalence:1"},
    )
    assert "unit_keys_mismatch" in errors
    assert any(error.startswith("unit_identity_invalid") for error in errors)
    assert any(error.startswith("weight_shape_invalid") for error in errors)

    with pytest.raises(ValueError, match="both labels"):
        exp._binary_auroc([0, 0], [0.1, 0.2])
    with pytest.raises(ValueError, match="positive"):
        exp._binary_pr_auc([0, 0], [0.1, 0.2])
    disabled_accept = exp._decision_for_policy(
        0.001,
        {
            "accept_threshold": 0.05,
            "reject_threshold": 0.9,
            "accept_enabled": False,
            "reject_enabled": False,
        },
        "fixture",
    )
    disabled_reject = exp._decision_for_policy(
        0.99,
        {
            "accept_threshold": 0.05,
            "reject_threshold": 0.9,
            "accept_enabled": True,
            "reject_enabled": False,
        },
        "fixture",
    )
    assert disabled_accept["reason"] == "accept_action_uncertified_escalation"
    assert disabled_reject["reason"] == "reject_action_uncertified_escalation"
    with pytest.raises(ValueError, match="require rows"):
        exp.paired_group_intervals([], draws=2)
    incomplete = [
        {
            "arm": exp.PRIMARY_VALUE_ARM,
            "group_id": "g",
            "brier_contribution": 0.1,
            "log_loss_contribution": 0.2,
            "decision": "accept",
        }
    ]
    assert exp.paired_group_intervals(incomplete, draws=2)[exp.PRIMARY_VALUE_ARM] == {}


def test_aggregate_diagnostics_and_artifact_mutations() -> None:
    """REQ-AUTO-7385 recomputes aggregate evidence and rejects changed declarations."""

    metrics = {
        "unit": {
            "arm": exp.PRIMARY_VALUE_ARM,
            "effective_groups": 2,
            "prevalence": 0.5,
            "brier": 0.1,
            "log_loss": 0.2,
            "auroc": 0.9,
            "pr_auc": 0.8,
            "coverage": 0.5,
            "utility": 0.5,
            "incorrect_accept_risk": 0.0,
            "correct_reject_risk": 0.5,
            "accept_count": 1,
            "reject_count": 1,
            "escalate_count": 0,
        }
    }
    aggregate = exp._aggregate_arm_metrics(metrics)
    assert aggregate[exp.PRIMARY_VALUE_ARM]["incorrect_accept_risk"] == 0.0
    assert aggregate[exp.PRIMARY_VALUE_ARM]["correct_reject_risk"] == 0.5
    assert exp._gate("minimum", "completion", 2, 3, ">=")["passed"] is True
    summary = exp._gate_summary(
        [
            exp._gate("required", "completion", True, False),
            exp._gate("science", "scientific_efficacy", True, False),
        ]
    )
    assert summary["failed_required_count"] == 1
    assert summary["failed_scientific_gate_count"] == 1

    state = exp.train_arm(exp.PRIMARY_VALUE_ARM, 7382001, _tiny_rows(), steps=2)
    state["affine"] = exp.fit_affine_transform([-2, -1, 1, 2], [0, 0, 1, 1], steps=2)
    state["selected_policy"] = {
        "accept_threshold": 0.05,
        "reject_threshold": 0.99,
        "accept_enabled": True,
        "reject_enabled": False,
        "threshold_index": 4,
    }
    diagnostic = exp._calibration_diagnostic(
        state, _tiny_rows("probability_calibration"), calibrated=True
    )
    assert diagnostic["rows"] == 4
    state.update(
        {
            "calibration_metrics_pre_affine": diagnostic,
            "calibration_metrics_post_affine": diagnostic,
            "energy_recomputation_check": {"passed": True},
            "fit_duration_s": 0.1,
            "checkpoint_path": "fixture.json",
            "checkpoint_sha256": "sha256:fixture",
        }
    )
    record = exp._training_run_record(state)
    reports = exp._objective_reports(
        {exp.PRIMARY_VALUE_ARM: aggregate[exp.PRIMARY_VALUE_ARM]}, [record]
    )
    assert reports["log_loss_trained_head"]["arm"] == exp.PRIMARY_VALUE_ARM
    request = exp._scoring_request([state])
    assert request["units"][0]["arm"] == exp.PRIMARY_VALUE_ARM

    artifact = exp.build_fixture_artifact()
    assert exp.validate_artifact(None) == ["artifact_not_object"]
    mutations = (
        ("schema", "wrong", "identity_mismatch"),
        ("invocation_counts", {}, "current_invocation_counts_nonzero"),
        ("inference_substrate_class", "aggregation", "substrate_class_mismatch"),
        ("execution_venue", "host_cpu", "execution_venue_mismatch"),
        ("promotion_score", 1, "promotion_nonzero"),
        ("field_principles", {}, "field_principles_incomplete"),
        ("verdict_class", "unknown", "verdict_class_invalid"),
    )
    for field, value, expected in mutations:
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["flagged_adversarial"] = True
    assert "adversarial_scores_nonzero" in exp.validate_artifact(changed)
    blocked = exp.build_blocked_artifact(
        [
            {
                "check": "x",
                "upstream": "x",
                "artifact_field": "x",
                "expected": 1,
                "observed": 0,
                "passed": False,
            }
        ],
        {},
    )
    blocked["rows"] = [{}]
    blocked["decision_capture_complete_score"] = 1
    blocked["gate_check_summary"]["first_required_failure"] = None
    blocked_errors = exp.validate_artifact(blocked)
    assert "blocked_artifact_has_dependent_work" in blocked_errors
    assert "blocked_gate_summary_missing" in blocked_errors
    assert "blocked_capture_nonzero" in blocked_errors
