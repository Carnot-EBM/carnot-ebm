"""Tests for authenticated V655 typed calibration.

Spec: REQ-AUTO-7481 and SCENARIO-AUTO-7481-01 through -07.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_7481_v655_typed_calibration as exp


def _native_pair(
    group: str,
    role: str,
    label: int | None,
    first_odds: float,
    second_odds: float,
    *,
    arm: str = "full_source_response",
) -> list[dict[str, object]]:
    """Build two stable-ID option rows without relying on display order."""

    rows: list[dict[str, object]] = []
    for order, odds in (
        (["supported", "contains_unsupported"], first_odds),
        (["contains_unsupported", "supported"], second_odds),
    ):
        rows.append(
            {
                "group_id": group,
                "source_group_id": group,
                "group_hash": f"sha256:{group}",
                "role": role,
                "arm": arm,
                "row_kind": arm,
                "option_order": order,
                "raw_logits_by_option_id": {
                    "supported": 0.0,
                    "contains_unsupported": odds,
                },
                "gold_label": label,
                "eligible": True,
                "disposition": "complete",
                "error": None,
            }
        )
    return rows


def _predictor(group: str) -> dict[str, object]:
    return {
        "row_key": f"row-{group}",
        "group_id": group,
        "source_text": "The source states that the total is 10 units.",
        "response_text": "The total is 12 units and this can be checked.",
        "source_id": "forbidden-source-id",
        "response_generator": "forbidden-generator",
        "outcome_note": "forbidden-note",
    }


def _fit_rows(count: int, *, role: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(count):
        label = index % 2
        odds = 1.25 if label else -1.25
        base = np.asarray(
            [odds, float(label), 1.0 - float(label), 0.25, 0.5, 0.0],
            dtype=np.float64,
        )
        rows.append(
            {
                "group_id": f"{role}-{index:03d}",
                "role": role,
                "label": label,
                "features": {
                    "full": base.tolist(),
                    "verifier_only": np.asarray([0.0, *base[1:]]).tolist(),
                    "source_removal": [0.0, float(label), 0.5, 0.0, 0.0, 1.0],
                },
                "native_log_odds": odds,
            }
        )
    return rows


def test_spec_declares_v655_typed_calibration() -> None:
    """REQ-AUTO-7481: implementation starts from a named capability contract."""

    text = exp.SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-AUTO-7481" in text
    for suffix in range(1, 8):
        assert f"SCENARIO-AUTO-7481-0{suffix}" in text


def test_native_orders_are_remapped_before_average() -> None:
    """SCENARIO-AUTO-7481-01: display reversal cannot change stable log odds."""

    rows = _native_pair("g1", "training", 0, 2.0, 4.0)
    reduced, controls = exp.aggregate_native_rows(rows)

    assert controls == []
    assert len(reduced) == 1
    assert reduced[0]["native_log_odds"] == pytest.approx(3.0)
    assert reduced[0]["label"] == 1
    assert reduced[0]["order_count"] == 2

    shuffled = _native_pair("g2", "calibration_tuning", None, 1.0, 1.0, arm="shuffled_source")
    reduced, controls = exp.aggregate_native_rows([*rows, *shuffled])
    assert len(reduced) == 1
    assert controls[0]["label"] is None
    assert controls[0]["benefit_eligible"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.pop(), "option_order_pair_invalid"),
        (
            lambda rows: rows[0].__setitem__("raw_logits_by_option_id", None),
            "native_logits_missing",
        ),
        (
            lambda rows: rows[0].__setitem__(
                "raw_logits_by_option_id",
                {"supported": 0.0, "contains_unsupported": float("inf")},
            ),
            "native_logits_nonfinite",
        ),
        (lambda rows: rows[0].__setitem__("gold_label", 1), "gold_label_disagreement"),
    ],
)
def test_native_row_mutations_fail_closed(mutation: object, message: str) -> None:
    """SCENARIO-AUTO-7481-01: malformed order evidence cannot enter fitting."""

    rows = _native_pair("g1", "training", 0, 2.0, 4.0)
    mutation(rows)  # type: ignore[operator]
    with pytest.raises(ValueError, match=message):
        exp.aggregate_native_rows(rows)


def test_feature_projection_excludes_identity_and_supports_ablations() -> None:
    """SCENARIO-AUTO-7481-01/06: only frozen numeric inputs reach a head."""

    features, receipt = exp.project_predictor_features(_predictor("g1"), native_log_odds=2.0)

    assert set(features) == {"full", "verifier_only", "source_removal"}
    assert all(len(vector) == 6 for vector in features.values())
    assert features["full"][0] == 2.0
    assert features["verifier_only"][0] == 0.0
    assert features["source_removal"][-1] == 1.0
    assert receipt["labels_consumed"] is False
    assert receipt["identity_consumed"] is False
    assert receipt["denied_fields_present"] == [
        "outcome_note",
        "response_generator",
        "source_id",
    ]
    with pytest.raises(ValueError, match="native_log_odds_nonfinite"):
        exp.project_predictor_features(_predictor("g1"), native_log_odds=float("nan"))


def test_fit_bundle_freezes_five_seeds_without_heldout_input() -> None:
    """SCENARIO-AUTO-7481-02: five heads fit only train and tune only calibration."""

    training = _fit_rows(12, role="training")
    calibration = _fit_rows(8, role="calibration_tuning")
    first = exp.fit_numeric_bundle(training, calibration, steps=2)
    second = exp.fit_numeric_bundle(training, calibration, steps=2)

    assert first == second
    assert first["training_seeds"] == list(exp.TRAINING_SEEDS)
    assert first["roles_consumed"] == ["training", "calibration_tuning"]
    assert first["heldout_labels_consumed"] is False
    assert set(first["heads"]) == {
        "gibbs",
        "logistic",
        "verifier_only_gibbs",
        "shuffled_label_gibbs",
        "source_removal_gibbs",
    }
    assert all(len(states) == 5 for states in first["heads"].values())
    assert first["bundle_sha256"] == exp.numeric_bundle_hash(first)
    assert len({row["checkpoint_sha256"] for row in first["checkpoint_manifest"]}) == 25


def test_numeric_bundle_rejects_role_and_feature_leakage() -> None:
    """SCENARIO-AUTO-7481-02: role and feature leaks fail before fitting."""

    training = _fit_rows(4, role="training")
    calibration = _fit_rows(4, role="calibration_tuning")
    bad_role = deepcopy(training)
    bad_role[0]["role"] = "external"
    with pytest.raises(ValueError, match="training_role_invalid"):
        exp.fit_numeric_bundle(bad_role, calibration, steps=1)

    bad_features = deepcopy(training)
    bad_features[0]["features"]["full"].append(0.0)  # type: ignore[index, union-attr]
    with pytest.raises(ValueError, match="feature_shape_invalid"):
        exp.fit_numeric_bundle(bad_features, calibration, steps=1)

    bad_calibration = deepcopy(calibration)
    bad_calibration[0]["role"] = "internal_test"
    with pytest.raises(ValueError, match="calibration_role_invalid"):
        exp.fit_numeric_bundle(training, bad_calibration, steps=1)

    missing_view = deepcopy(training)
    del missing_view[0]["features"]["full"]  # type: ignore[index, union-attr]
    with pytest.raises(ValueError, match="feature_view_missing"):
        exp.fit_numeric_bundle(missing_view, calibration, steps=1)

    bad_label = deepcopy(training)
    bad_label[0]["label"] = 2
    with pytest.raises(ValueError, match="binary_label_invalid"):
        exp.fit_numeric_bundle(bad_label, calibration, steps=1)

    one_class = deepcopy(training)
    for row in one_class:
        row["label"] = 0
    with pytest.raises(ValueError, match="training_support_invalid"):
        exp.fit_numeric_bundle(one_class, calibration, steps=1)


def test_frozen_bundle_scores_every_real_arm_and_seed() -> None:
    """SCENARIO-AUTO-7481-02/03: frozen heads emit explicit group-arm-seed rows."""

    training = _fit_rows(8, role="training")
    calibration = _fit_rows(6, role="calibration_tuning")
    bundle = exp.fit_numeric_bundle(training, calibration, steps=1)
    scored = exp.score_numeric_bundle(bundle, calibration[:2])

    assert len(scored) == 54
    assert {row["arm"] for row in scored} == set(exp.ALL_ARMS)
    assert len([row for row in scored if row["arm"] == "gibbs"]) == 10

    missing = deepcopy(bundle)
    missing["heads"] = None
    with pytest.raises(ValueError, match="frozen_heads_missing"):
        exp.score_numeric_bundle(missing, calibration[:1])
    short = deepcopy(bundle)
    short["heads"]["gibbs"] = []
    with pytest.raises(ValueError, match="frozen_head_count_invalid:gibbs"):
        exp.score_numeric_bundle(short, calibration[:1])


def test_temperature_and_probability_metrics_are_exact() -> None:
    """SCENARIO-AUTO-7481-04: proper scores and support remain explicit."""

    logits = np.asarray([-2.0, -1.0, 1.0, 2.0])
    labels = np.asarray([0, 0, 1, 1])
    temperature = exp.select_temperature(logits, labels)
    probabilities = exp.sigmoid(logits / temperature)
    summary = exp.probability_metrics(labels, probabilities)

    assert temperature in exp.TEMPERATURE_GRID
    assert summary["n_groups"] == 4
    assert summary["class_support"] == {"supported": 2, "contains_unsupported": 2}
    assert 0.0 <= summary["brier"] < 0.25
    assert summary["auroc"] == 1.0
    assert 0.0 <= summary["ece"] <= 1.0
    assert exp.probability_metrics([0, 0], [0.1, 0.2])["auroc"] is None
    with pytest.raises(ValueError, match="temperature_input_invalid"):
        exp.select_temperature([1.0], [0, 1])
    with pytest.raises(ValueError, match="probability_metric_input_invalid"):
        exp.probability_metrics([], [])


def test_group_bootstrap_and_holm_do_not_count_seed_rows() -> None:
    """SCENARIO-AUTO-7481-03/04: inference resamples groups after seed averaging."""

    prediction_rows = [
        {
            "group_id": group,
            "arm": arm,
            "seed": seed,
            "label": index % 2,
            "probability": probability,
        }
        for index, group in enumerate(["a", "b", "c", "d"])
        for arm, probability in (("gibbs", 0.1 if index % 2 == 0 else 0.9), ("temperature", 0.5))
        for seed in ((1, 2) if arm == "gibbs" else (None,))
    ]
    deltas = exp.group_loss_deltas(
        prediction_rows,
        candidate="gibbs",
        control="temperature",
        loss="brier",
    )
    comparison = exp.paired_bootstrap(deltas, draws=200, seed=71)
    family = exp.holm_upper_bounds({"temperature": comparison}, alpha=0.05)

    assert len(deltas) == 4
    assert comparison["group_count"] == 4
    assert comparison["delta"] < 0.0
    assert family["temperature"]["holm_upper"] < 0.0
    with pytest.raises(ValueError, match="loss_name_invalid"):
        exp.group_loss_deltas(prediction_rows, candidate="gibbs", control="temperature", loss="x")
    with pytest.raises(ValueError, match="bootstrap_input_invalid"):
        exp.paired_bootstrap([], draws=0, seed=71)

    conflicting = deepcopy(prediction_rows)
    conflicting[1]["label"] = 1
    with pytest.raises(ValueError, match="prediction_label_disagreement"):
        exp.group_loss_deltas(
            conflicting,
            candidate="gibbs",
            control="temperature",
            loss="brier",
        )


def test_cost_policy_is_selected_without_evaluation_rows() -> None:
    """SCENARIO-AUTO-7481-05: thresholds use calibration labels only."""

    probabilities = [0.01, 0.1, 0.8, 0.99]
    labels = [0, 0, 1, 1]
    policy = exp.select_cost_policy(
        probabilities,
        labels,
        false_accept_cost=5.0,
        false_reject_cost=1.0,
        escalation_cost=0.5,
    )
    actions = [exp.typed_action(value, policy) for value in probabilities]
    costs = exp.policy_cost_rows(
        probabilities,
        labels,
        policy,
        false_accept_cost=5.0,
        false_reject_cost=1.0,
        escalation_cost=0.5,
    )

    assert policy["selection_role"] == "calibration_tuning"
    assert set(actions) <= {"accept", "reject", "escalate"}
    assert len(costs) == 4
    assert all(row["cost"] >= 0.0 for row in costs)
    with pytest.raises(ValueError, match="policy_input_invalid"):
        exp.select_cost_policy(
            [],
            [],
            false_accept_cost=5.0,
            false_reject_cost=1.0,
            escalation_cost=0.5,
        )


def test_decision_grid_reports_all_cells_and_holm_family() -> None:
    """SCENARIO-AUTO-7481-05: all nine costs remain visible even on a null."""

    calibration = []
    evaluation = []
    for index in range(20):
        label = index % 2
        for arm, probability in (
            ("gibbs", 0.05 if label == 0 else 0.95),
            ("temperature", 0.45 if label == 0 else 0.55),
            ("logistic", 0.4 if label == 0 else 0.6),
        ):
            row = {
                "group_id": f"g-{index}",
                "arm": arm,
                "seed": 1 if arm != "temperature" else None,
                "label": label,
                "probability": probability,
            }
            calibration.append({**row, "role": "calibration_tuning"})
            evaluation.append({**row, "role": "external"})
    report = exp.evaluate_cost_grid(calibration, evaluation, draws=200, seed=72)

    assert len(report["cells"]) == 9
    assert report["holm_family_size"] == 9
    assert report["decision_benefit_score"] in {0, 1}
    assert all(cell["best_simple_arm"] in exp.SIMPLE_ARMS for cell in report["cells"])
    assert report == json.loads(json.dumps(report))


def test_independent_reducer_recomputes_scores_from_rows() -> None:
    """SCENARIO-AUTO-7481-07: completion and benefit derive from raw rows."""

    rows = []
    for index in range(8):
        label = index % 2
        for role in ("internal_test", "external"):
            for arm, probability in (
                ("gibbs", 0.1 if label == 0 else 0.9),
                ("temperature", 0.5),
                ("logistic", 0.45 if label == 0 else 0.55),
            ):
                rows.append(
                    {
                        "group_id": f"{role}-{index}",
                        "role": role,
                        "arm": arm,
                        "seed": 655101 if arm != "temperature" else None,
                        "label": label,
                        "probability": probability,
                        "failed": False,
                    }
                )
    reduced = exp.reduce_prediction_rows(rows, bootstrap_draws=200)

    assert reduced["evaluation_complete"] is True
    assert reduced["role_group_counts"] == {"external": 8, "internal_test": 8}
    assert reduced["probability_report"]["external"]["gibbs"]["n_groups"] == 8
    assert reduced["probability_benefit_score"] in {0, 1}


def test_artifact_validation_detects_checksum_and_deployment_claim() -> None:
    """SCENARIO-AUTO-7481-07: cold validation rejects drift and certification."""

    artifact = exp.fixture_artifact()
    assert exp.validate_artifact(artifact, check_files=False) == []

    changed = deepcopy(artifact)
    changed["deployment_certificate_valid"] = True
    errors = exp.validate_artifact(changed, check_files=False)
    assert "deployment_certificate_forbidden" in errors
    assert "reproducibility_checksum_mismatch" in errors


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("schema", "wrong", "identity_invalid"),
        ("run_date", "20260101", "run_identity_invalid"),
        ("MODEL_SPECS", ["model"], "model_specs_not_empty"),
        ("model_invoked", True, "current_model_accounting_invalid"),
        ("execution_venue", "board", "substrate_invalid"),
        ("verifier_is_oracle", True, "oracle_declaration_invalid"),
        ("verdict_class", "unknown", "verdict_class_invalid"),
        ("frozen_policy_manifest", {}, "frozen_policy_hash_mismatch"),
    ],
)
def test_artifact_identity_mutations_fail_closed(field: str, value: object, error: str) -> None:
    """SCENARIO-AUTO-7481-07: terminal identity mutations have named failures."""

    artifact = exp.fixture_artifact()
    artifact[field] = value
    assert error in exp.validate_artifact(artifact, check_files=False)


def test_gate_rows_carry_registered_principles() -> None:
    """REQ-AUTO-7481: each gate explains the failure mode it prevents."""

    for category in ("required_validity", "readiness", "scientific_benefit"):
        gate = exp._gate("check", category, True, True, "==", True)
        assert gate["passed"] is True
        assert gate["principle"].endswith(".")


def test_preconditions_preserve_upstream_flags(tmp_path: Path) -> None:
    """REQ-AUTO-7481: exact ready fields and flags gate dependent fitting."""

    fit = {
        "fit_capture_ready_score": 1,
        "verdict_class": "null",
        "flagged_adversarial": False,
    }
    evaluation = {
        "evaluation_capture_ready_score": 1,
        "verdict_class": "null",
        "flagged_adversarial": False,
    }
    (tmp_path / "fit.json").write_text(json.dumps(fit), encoding="utf-8")
    (tmp_path / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")
    checks = exp.upstream_preconditions(tmp_path / "fit.json", tmp_path / "evaluation.json")

    assert all(row["passed"] for row in checks)
    bad = deepcopy(evaluation)
    bad["flagged_adversarial"] = True
    (tmp_path / "evaluation.json").write_text(json.dumps(bad), encoding="utf-8")
    checks = exp.upstream_preconditions(tmp_path / "fit.json", tmp_path / "evaluation.json")
    assert [row["check"] for row in checks if not row["passed"]] == ["eval_unflagged"]
