"""Tests for the frozen V661 source evaluation.

Spec refs: REQ-VERIFY-7567 and SCENARIO-VERIFY-7567-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_7567_v661_source_evaluation as exp


def _feature_rows(count: int = 4) -> list[dict[str, object]]:
    """Build label-free groups with both mapped presentation orders."""

    rows = []
    for index in range(count):
        sign = -1.0 if index % 2 == 0 else 1.0
        rows.append(
            {
                "component_hash": f"sha256:component-{index}",
                "group_hash": f"sha256:group-{index}",
                "role": "test",
                "tool_type": "fixture",
                "feature_views": {
                    "supported_first": [sign * 1.2, sign * 0.8, sign * 0.5],
                    "unsupported_first": [sign * 1.0, sign * 0.7, sign * 0.4],
                },
                "original_unsupported_probability": 0.5,
            }
        )
    return rows


def _label_rows(count: int = 4) -> list[dict[str, object]]:
    """Return labels whose identity matches ``_feature_rows``."""

    return [
        {
            "component_hash": f"sha256:component-{index}",
            "role": "test",
            "label": index % 2,
            "prediction_freeze_sha256": "sha256:upstream-freeze",
        }
        for index in range(count)
    ]


def test_label_blind_scoring_freezes_every_arm_and_order() -> None:
    """REQ-VERIFY-7567; SCENARIO-VERIFY-7567-CUSTODY."""

    heads = exp.fixture_frozen_heads()
    rows = exp.score_label_free_predictions(_feature_rows(), heads)

    assert len(rows) == 4 * len(exp.ARMS)
    assert {row["arm"] for row in rows} == set(exp.ARMS)
    assert all(set(row["option_probabilities"]) == set(exp.ORDER_NAMES) for row in rows)
    assert all(
        row["probability"]
        == pytest.approx(sum(row["option_probabilities"].values()) / len(exp.ORDER_NAMES))
        for row in rows
    )
    assert all("label" not in row for row in rows)
    assert all(row["head_identity"] for row in rows)
    assert all(
        row["source_intervention_diagnostics"]["source_neutral_absolute_delta"] is None
        for row in rows
        if row["arm"]
        in {"raw_original", "temperature_original", "original_only_local_basis_energy"}
    )

    visible = _feature_rows()
    visible[0]["label"] = 0
    with pytest.raises(ValueError, match="label_visible_during_prediction"):
        exp.score_label_free_predictions(visible, heads)

    missing_order = _feature_rows()
    missing_order[0]["feature_views"] = {"supported_first": [-1.0, -0.5, -0.2]}
    with pytest.raises(ValueError, match="feature_views_invalid"):
        exp.score_label_free_predictions(missing_order, heads)


def test_labels_attach_only_after_freeze_with_absolute_losses() -> None:
    """REQ-VERIFY-7567; SCENARIO-VERIFY-7567-ROWS."""

    predictions = exp.score_label_free_predictions(_feature_rows(), exp.fixture_frozen_heads())
    rows = exp.attach_test_labels(
        predictions,
        _label_rows(),
        prediction_sha256="sha256:local-predictions",
        expected_upstream_freeze="sha256:upstream-freeze",
    )

    assert len(rows) == len(predictions)
    assert all(0.0 <= row["brier"] <= 1.0 for row in rows)
    assert all(row["log_loss"] >= 0.0 for row in rows)
    assert all(row["action"] in {"accept", "reject", "escalate"} for row in rows)
    assert all(row["realized_cost"] >= 0.0 for row in rows)
    assert all(len(row["descriptive_cost_grid"]) == 9 for row in rows)
    assert all(row["prediction_freeze_sha256"] == "sha256:local-predictions" for row in rows)

    missing = _label_rows(3)
    with pytest.raises(ValueError, match="evaluation_group_mismatch"):
        exp.attach_test_labels(
            predictions,
            missing,
            prediction_sha256="sha256:local-predictions",
            expected_upstream_freeze="sha256:upstream-freeze",
        )

    duplicate = _label_rows()
    duplicate.append(deepcopy(duplicate[0]))
    with pytest.raises(ValueError, match="evaluation_label_duplicate"):
        exp.attach_test_labels(
            predictions,
            duplicate,
            prediction_sha256="sha256:local-predictions",
            expected_upstream_freeze="sha256:upstream-freeze",
        )


def test_loss_and_decision_boundaries_fail_closed() -> None:
    """REQ-VERIFY-7567 keeps probability and action rows finite."""

    assert exp.loss_fields(0.25, 0)["brier"] == pytest.approx(0.0625)
    assert exp.primary_decision(0.01, 0) == {"action": "accept", "realized_cost": 0.0}
    assert exp.primary_decision(0.99, 1) == {"action": "reject", "realized_cost": 0.0}
    assert exp.primary_decision(0.5, 1) == {"action": "escalate", "realized_cost": 0.2}
    with pytest.raises(ValueError, match="probability_or_label_invalid"):
        exp.loss_fields(float("nan"), 0)
    with pytest.raises(ValueError, match="probability_or_label_invalid"):
        exp.primary_decision(0.5, 2)


def test_positive_control_exercises_registered_paired_inference() -> None:
    """REQ-VERIFY-7567; SCENARIO-VERIFY-7567-INFERENCE/CONTROLS."""

    result = exp.run_analytical_positive_control(draws=256)

    assert result["passed"] is True
    assert result["probability_benefit_score"] == 1
    assert result["decision_benefit_score"] == 1
    assert result["reduction"]["support"]["valid_groups"] == 80
    assert result["reduction"]["support"]["label_counts"] == {"0": 40, "1": 40}
    assert len(result["reduction"]["paired_intervals"]["brier"]) == 3
    assert all(
        row["simultaneous_upper95"] < 0.0
        for row in result["reduction"]["paired_intervals"]["brier"].values()
    )


def test_supported_empirical_reduction_separates_probability_and_decision() -> None:
    """REQ-VERIFY-7567 reports probability and decision benefit separately."""

    rows = exp.analytical_panel_rows()
    settings = exp.evaluator_settings(draws=256, confirmatory_allowed=True)
    reduced = exp.reduce_rows(rows, settings=settings)

    assert reduced["probability_benefit_score"] == 1
    assert reduced["decision_benefit_score"] == 1
    assert reduced["sample_size_budget"]["completed"] == 80
    assert reduced["sample_size_budget"]["orders_are_independent_units"] is False
    assert reduced["sample_size_budget"]["arms_are_independent_units"] is False
    assert reduced["primary_cost_contrast"]["upper95"] < 0.0
    assert reduced["candidate_non_escalation_fraction"] >= 0.2

    exploratory = exp.reduce_rows(
        rows,
        settings=exp.evaluator_settings(draws=256, confirmatory_allowed=False),
    )
    assert exploratory["static_measurement_complete_score"] == 1
    assert exploratory["probability_benefit_score"] == 0
    assert exploratory["decision_benefit_score"] == 0
    assert "fresh_confirmatory_claim_forbidden" in exploratory["failed_benefit_gates"]


def test_support_and_arm_roster_cannot_be_hidden_by_bootstrap() -> None:
    """REQ-VERIFY-7567 keeps exclusions and duplicated views out of N."""

    rows = exp.analytical_panel_rows(group_count=60)
    reduced = exp.reduce_rows(
        rows,
        settings=exp.evaluator_settings(
            draws=64,
            confirmatory_allowed=True,
            expected_groups=60,
        ),
    )
    assert reduced["support"]["passed"] is False
    assert reduced["probability_benefit_score"] == 0

    incomplete = exp.analytical_panel_rows()
    incomplete.pop()
    with pytest.raises(ValueError, match="arm_roster_invalid"):
        exp.reduce_rows(incomplete, settings=exp.evaluator_settings(draws=32))

    duplicated = exp.analytical_panel_rows()
    duplicated.append(deepcopy(duplicated[0]))
    with pytest.raises(ValueError, match="duplicate_group_arm"):
        exp.reduce_rows(duplicated, settings=exp.evaluator_settings(draws=32))


def test_holm_and_paired_bootstrap_retain_direction_and_counts() -> None:
    """SCENARIO-VERIFY-7567-INFERENCE freezes source-component uncertainty."""

    indices = exp.bootstrap_indices(4, draws=64, seed=exp.BOOTSTRAP_SEED)
    first = exp.paired_interval([-0.2, -0.1, -0.3, -0.2], indices)
    second = exp.paired_interval([-0.1, -0.1, -0.1, -0.1], indices)
    adjusted = exp.holm_one_sided({"a": first, "b": second}, alpha=0.05)

    assert first["group_count"] == 4
    assert first["draws"] == 64
    assert first["upper95"] < 0.0
    assert {row["holm_rank"] for row in adjusted.values()} == {1, 2}
    assert all(0.0 <= row["holm_adjusted_p"] <= 1.0 for row in adjusted.values())

    with pytest.raises(ValueError, match="bootstrap_shape_invalid"):
        exp.bootstrap_indices(0, draws=1, seed=1)
    with pytest.raises(ValueError, match="paired_bootstrap_shape_invalid"):
        exp.paired_interval([], indices)


def test_shortcut_and_option_order_controls_are_diagnostic() -> None:
    """SCENARIO-VERIFY-7567-CONTROLS does not turn missing source into truth."""

    predictions = exp.score_label_free_predictions(_feature_rows(), exp.fixture_frozen_heads())
    controls = exp.run_challenge_controls(predictions)

    assert controls["passed"] is True
    assert controls["missing_source_confidence_is_oracle"] is False
    assert controls["source_independent_shortcut"]["candidate_mean_absolute_delta"] > 0.0
    assert set(controls["option_permutation_sensitivity"]) == set(exp.ARMS)

    broken = deepcopy(predictions)
    broken[0]["option_probabilities"] = {"supported_first": 0.2}
    with pytest.raises(ValueError, match="option_prediction_roster_invalid"):
        exp.run_challenge_controls(broken)


def test_preconditions_authenticate_current_upstreams() -> None:
    """REQ-VERIFY-7567; SCENARIO-VERIFY-7567-PRECONDITIONS."""

    checked = exp.collect_preconditions(exp.REPO_ROOT)

    assert checked["passed"] is True
    assert checked["rows"]
    assert all(row["passed"] is True for row in checked["rows"])
    assert any(row["field"] == "test_capture_ready_score" for row in checked["rows"])
    assert any(row["field"] == "energy_fit_ready_score" for row in checked["rows"])
    assert any(row["check"] == "sidecar_hash" for row in checked["rows"])

    inputs = exp.load_label_free_inputs(exp.REPO_ROOT, checked)
    assert len(inputs["features"]) == 80
    assert inputs["heads"]["bundle_sha256"]
    labels = exp.open_test_labels(exp.REPO_ROOT, inputs["label_reference"])
    assert len(labels) == 80

    with pytest.raises(ValueError, match="authenticated_sidecars_missing"):
        exp.load_label_free_inputs(exp.REPO_ROOT, {"sidecars": []})
    with pytest.raises(ValueError, match="authenticated_sidecars_missing"):
        exp.load_label_free_inputs(exp.REPO_ROOT, {"sidecars": {}})


def test_preconditions_reject_missing_upstream_sidecar_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-7567-PRECONDITIONS records a missing manifest field."""

    original = exp.load_json

    def altered(path: Path) -> dict[str, object]:
        value = original(path)
        if path.name == exp.CAPTURE_PATH.name:
            value = deepcopy(value)
            value["raw_manifest"]["role_sidecars"].pop("test_labels")
        return value

    monkeypatch.setattr(exp, "load_json", altered)
    checked = exp.collect_preconditions(exp.REPO_ROOT)
    assert checked["passed"] is False
    assert any(
        row["check"] == "sidecar_hash" and row["field"] == "capture_test_labels"
        for row in checked["rows"]
    )


def test_input_loaders_reject_roster_and_label_leakage(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7567-CUSTODY rejects malformed sealed sidecars."""

    heads_path = tmp_path / "heads.json"
    exp.atomic_json(heads_path, exp.fixture_frozen_heads())
    labels_path = tmp_path / "labels.jsonl"
    exp.write_jsonl_sidecar(labels_path, _label_rows(), root=tmp_path)

    def preconditions_for(path: Path) -> dict[str, object]:
        return {
            "sidecars": {
                "capture_predictions": {"path": str(path)},
                "capture_test_labels": {"path": str(labels_path)},
                "frozen_heads": {"path": str(heads_path)},
            },
            "upstream_context": {"capture_prediction_freeze_sha256": "sha256:freeze"},
        }

    prediction_path = tmp_path / "predictions.jsonl"
    exp.write_jsonl_sidecar(prediction_path, _feature_rows(4), root=tmp_path)
    with pytest.raises(ValueError, match="test_feature_roster_invalid"):
        exp.load_label_free_inputs(tmp_path, preconditions_for(prediction_path))

    leaked = _feature_rows(80)
    leaked[0]["label"] = 0
    exp.write_jsonl_sidecar(prediction_path, leaked, root=tmp_path)
    with pytest.raises(ValueError, match="label_visible_during_prediction"):
        exp.load_label_free_inputs(tmp_path, preconditions_for(prediction_path))

    with pytest.raises(ValueError, match="test_label_roster_invalid"):
        exp.open_test_labels(tmp_path, {"path": str(labels_path)})


def test_missing_external_input_builds_complete_blocked_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-7567; SCENARIO-VERIFY-7567-BLOCKED."""

    checked = exp.collect_preconditions(tmp_path)
    failed = next(row for row in checked["rows"] if row["passed"] is False)
    artifact = exp.blocked_artifact(failed, root=tmp_path)

    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["planned_inference_substrate_class"] == "no_model_load"
    assert artifact["sample_size_budget"]["unstarted"] == 80
    assert artifact["static_measurement_complete_score"] == 0
    assert artifact["probability_benefit_score"] == 0
    assert artifact["decision_benefit_score"] == 0
    assert artifact["gate_check_summary"]["path"] == failed["path"]
    assert exp.validate_artifact(artifact, root=tmp_path, verify_sources=False) == []


def test_fixture_artifact_is_cold_valid_and_mutations_fail(tmp_path: Path) -> None:
    """REQ-VERIFY-7567; SCENARIO-VERIFY-7567-E2E."""

    artifact = exp.fixture_artifact()
    assert exp.validate_artifact(artifact, verify_sources=False) == []
    assert artifact["static_measurement_complete_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["invocation_counts"] == exp.ZERO_INVOCATION_COUNTS
    assert set(artifact) <= set(artifact["field_principles"])
    assert all(
        {"expected", "observed", "op", "passed", "category", "principle"} <= set(gate)
        for gate in artifact["acceptance_gate_results"]
    )

    path = tmp_path / "artifact.json"
    exp.atomic_json(path, artifact)
    assert exp.cold_replay(path, root=tmp_path, verify_sources=False) == []
    assert exp.independent_replay(path) == []

    changed = deepcopy(artifact)
    changed["rows"][0]["brier"] += 0.1
    assert "independent_reduction_mismatch" in exp.validate_artifact(changed, verify_sources=False)
    changed = deepcopy(artifact)
    changed["MODEL_SPECS"] = [{"name": "forbidden"}]
    assert "model_specs_must_be_empty" in exp.validate_artifact(changed, verify_sources=False)
    changed = deepcopy(artifact)
    changed["field_principles"].pop("rows")
    assert "field_principles_incomplete" in exp.validate_artifact(changed, verify_sources=False)


def test_sidecar_round_trip_and_hash_guard(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7567-CUSTODY binds persisted predictions and reports."""

    path = tmp_path / "rows.jsonl"
    rows = [{"unit": 1}, {"unit": 2}]
    receipt = exp.write_jsonl_sidecar(path, rows, root=tmp_path)

    assert receipt["rows"] == 2
    assert exp.verify_sidecar(receipt, tmp_path) == []
    wrong_count = {**receipt, "rows": 3}
    assert exp.verify_sidecar(wrong_count, tmp_path) == ["sidecar_row_count_mismatch:rows.jsonl"]
    assert exp._path_label(Path("/etc/hosts"), tmp_path) == "/etc/hosts"
    path.write_text('{"unit":3}\n', encoding="utf-8")
    assert exp.verify_sidecar(receipt, tmp_path) == ["sidecar_sha256_mismatch:rows.jsonl"]

    missing = {"path": "missing.jsonl", "sha256": "sha256:none", "rows": 0}
    assert exp.verify_sidecar(missing, tmp_path) == ["sidecar_missing:missing.jsonl"]


def test_settings_identity_and_cli_modes() -> None:
    """REQ-VERIFY-7567 freezes production settings before outcomes."""

    settings = exp.evaluator_settings()
    assert settings["bootstrap_draws"] == 2_000
    assert settings["bootstrap_seed"] == 7_567_001
    assert settings["minimum_groups"] == 64
    assert settings["minimum_per_label"] == 12
    assert settings["brier_improvement"] == 0.01
    assert settings["no_post_test_refit"] is True

    parsed = exp.parse_args(["--date", exp.RUN_DATE, "--cold-replay", "candidate.json"])
    assert parsed.cold_replay == Path("candidate.json")
    parsed = exp.parse_args(["--date", exp.RUN_DATE, "--independent-reduce", "candidate.json"])
    assert parsed.independent_reduce == Path("candidate.json")


def test_json_readers_reject_non_objects(tmp_path: Path) -> None:
    """REQ-VERIFY-7567 keeps malformed evidence from becoming empty data."""

    object_path = tmp_path / "object.json"
    object_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    assert exp.load_json(object_path) == {"ok": True}
    object_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object_required"):
        exp.load_json(object_path)

    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("{}\n[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jsonl_object_required"):
        exp.load_jsonl(rows_path)


def test_frozen_head_and_prediction_mutations_fail_closed() -> None:
    """SCENARIO-VERIFY-7567-CUSTODY authenticates each scoring operand."""

    base = exp.fixture_frozen_heads()
    vector = np.asarray([1.0, 0.5, 0.25])
    with pytest.raises(ValueError, match="frozen_head_manifest_invalid"):
        exp._trained_probability(exp.CANDIDATE_ARM, vector, {})
    bad = deepcopy(base)
    bad["normalization"]["safe_scale"] = [1.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="normalization_invalid"):
        exp._trained_probability(exp.CANDIDATE_ARM, vector, bad)
    with pytest.raises(ValueError, match="trained_arm_invalid"):
        exp._trained_probability("unknown", vector, base)
    bad = deepcopy(base)
    bad["selected_heads"].pop(exp.CANDIDATE_ARM)
    with pytest.raises(ValueError, match="frozen_head_missing"):
        exp._trained_probability(exp.CANDIDATE_ARM, vector, bad)
    bad = deepcopy(base)
    bad["selected_heads"][exp.CANDIDATE_ARM]["checkpoint_sha256"] = "sha256:changed"
    with pytest.raises(ValueError, match="checkpoint_sha256_mismatch"):
        exp._trained_probability(exp.CANDIDATE_ARM, vector, bad)
    bad = deepcopy(base)
    checkpoint = bad["selected_heads"][exp.CANDIDATE_ARM]["checkpoint"]
    checkpoint["weights"] = [0.0]
    bad["selected_heads"][exp.CANDIDATE_ARM]["checkpoint_sha256"] = exp.canonical_hash(checkpoint)
    with pytest.raises(ValueError, match="checkpoint_shape_invalid"):
        exp._trained_probability(exp.CANDIDATE_ARM, vector, bad)

    bad = deepcopy(base)
    bad["strongest_comparator"]["family"] = "wrong"
    with pytest.raises(ValueError, match="strongest_comparator_invalid"):
        exp.score_label_free_predictions(_feature_rows(1), bad)
    invalid = _feature_rows(1)
    invalid[0]["role"] = "online"
    with pytest.raises(ValueError, match="prediction_identity_invalid"):
        exp.score_label_free_predictions(invalid, base)
    duplicate = _feature_rows(1) * 2
    with pytest.raises(ValueError, match="prediction_component_duplicate"):
        exp.score_label_free_predictions(duplicate, base)
    nonfinite = _feature_rows(1)
    nonfinite[0]["feature_views"]["supported_first"][0] = float("nan")
    with pytest.raises(ValueError, match="feature_vector_invalid"):
        exp.score_label_free_predictions(nonfinite, base)

    predictions = exp.score_label_free_predictions(_feature_rows(), base)
    invalid_labels = _label_rows()
    invalid_labels[0]["prediction_freeze_sha256"] = "sha256:wrong"
    with pytest.raises(ValueError, match="evaluation_label_invalid"):
        exp.attach_test_labels(
            predictions,
            invalid_labels,
            prediction_sha256="sha256:local",
            expected_upstream_freeze="sha256:upstream-freeze",
        )


def test_reducer_rejects_identity_disposition_and_decision_drift() -> None:
    """SCENARIO-VERIFY-7567-INFERENCE rejects drift before resampling."""

    settings = exp.evaluator_settings(draws=16)
    base = exp.analytical_panel_rows()

    changed = deepcopy(base)
    changed[0]["source_component_hash"] = ""
    with pytest.raises(ValueError, match="row_identity_invalid"):
        exp.reduce_rows(changed, settings=settings)
    changed = deepcopy(base)
    changed[len(exp.ARMS)]["source_component_hash"] = changed[0]["source_component_hash"]
    with pytest.raises(ValueError, match="source_component_mapping_invalid"):
        exp.reduce_rows(changed, settings=settings)
    changed = deepcopy(base)
    changed[0]["failed"] = True
    with pytest.raises(ValueError, match="row_disposition_invalid"):
        exp.reduce_rows(changed, settings=settings)
    changed = deepcopy(base)
    changed[0]["action"] = "reject"
    with pytest.raises(ValueError, match="primary_decision_mismatch"):
        exp.reduce_rows(changed, settings=settings)
    with pytest.raises(ValueError, match="expected_group_count_mismatch"):
        exp.reduce_rows(base, settings={**settings, "expected_groups": 79})
    changed = deepcopy(base)
    changed[1]["label"] = 1 - changed[1]["label"]
    changed[1]["y"] = changed[1]["label"]
    changed[1].update(exp.loss_fields(changed[1]["probability"], changed[1]["label"]))
    changed[1].update(exp.primary_decision(changed[1]["probability"], changed[1]["label"]))
    with pytest.raises(ValueError, match="group_label_disagreement"):
        exp.reduce_rows(changed, settings=settings)
    with pytest.raises(ValueError, match="holm_family_empty"):
        exp.holm_one_sided({}, alpha=0.05)


def test_challenge_control_rejects_nonfinite_or_missing_arms() -> None:
    """SCENARIO-VERIFY-7567-CONTROLS retains all diagnostic arms."""

    predictions = exp.score_label_free_predictions(_feature_rows(), exp.fixture_frozen_heads())
    changed = deepcopy(predictions)
    changed[0]["option_probabilities"][exp.ORDER_NAMES[0]] = float("nan")
    with pytest.raises(ValueError, match="option_prediction_nonfinite"):
        exp.run_challenge_controls(changed)
    changed = [row for row in predictions if row["arm"] != "raw_original"]
    with pytest.raises(ValueError, match="arm_roster_invalid"):
        exp.run_challenge_controls(changed)


def test_artifact_validator_names_schema_and_provenance_mutations(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7567-E2E cold validation rejects typed drift."""

    assert exp.validate_artifact([], verify_sources=False) == ["artifact_not_object"]
    base = exp.fixture_artifact()
    mutations = [
        (lambda row: row.pop("schema"), "required_fields_missing:schema"),
        (lambda row: row.update({"experiment_id": "wrong"}), "identity_mismatch"),
        (lambda row: row.update({"model_invoked": True}), "current_model_calls_nonzero"),
        (
            lambda row: row.update({"inference_substrate_class": "model_load_no_generation"}),
            "inference_substrate_class_mismatch",
        ),
        (lambda row: row.update({"inference_substrate": "wrong"}), "inference_substrate_mismatch"),
        (lambda row: row.update({"execution_venue": "host_cpu"}), "execution_venue_invalid"),
        (lambda row: row.update({"verdict_class": "unknown"}), "verdict_class_invalid"),
        (lambda row: row.update({"honest_verdict": "null"}), "honest_verdict_prefix_invalid"),
        (lambda row: row.update({"probability_benefit_score": True}), "bare_score_invalid"),
        (lambda row: row.update({"acceptance_gate_results": [{}]}), "acceptance_gates_invalid"),
    ]
    for mutate, expected in mutations:
        changed = deepcopy(base)
        mutate(changed)
        assert any(
            expected in error for error in exp.validate_artifact(changed, verify_sources=False)
        )

    changed = deepcopy(base)
    changed["probability_metrics"] = {}
    assert "independent_reduction_mismatch" in exp.validate_artifact(changed, verify_sources=False)
    changed = deepcopy(base)
    changed["static_measurement_complete_score"] = 0
    assert "independent_score_mismatch" in exp.validate_artifact(changed, verify_sources=False)

    evidence = tmp_path / "evidence.jsonl"
    reference = exp.write_jsonl_sidecar(evidence, [{"ok": True}], root=tmp_path)
    checked = deepcopy(base)
    checked["source_artifact_hashes"] = [exp.source_hash_row(evidence, tmp_path)]
    checked["raw_sidecars"] = {"evidence": reference}
    checked["field_principles"] = exp._field_principles((*checked.keys(), "field_principles"))
    checked["reproducibility_checksum"] = exp.artifact_checksum(checked)
    assert exp.validate_artifact(checked, root=tmp_path) == []
    evidence.write_text("changed\n", encoding="utf-8")
    errors = exp.validate_artifact(checked, root=tmp_path)
    assert any(error.startswith("source_hash_mismatch") for error in errors)
    assert any(error.startswith("sidecar_sha256_mismatch") for error in errors)


def test_build_artifact_classifies_positive_null_and_disqualified() -> None:
    """REQ-VERIFY-7567 keeps completion separate from empirical value."""

    rows = exp.analytical_panel_rows()
    controls = exp.run_challenge_controls(rows)
    positive = exp.run_analytical_positive_control(draws=64)

    def build(*, confirmatory: bool, validation: bool) -> dict[str, object]:
        return exp.build_artifact(
            preconditions=[exp.precondition_row("fixture", "fixture", "ready", 1, 1, "x")],
            source_hashes=[],
            rows=rows,
            settings=exp.evaluator_settings(draws=64, confirmatory_allowed=confirmatory),
            challenges=controls,
            positive_control=positive,
            exposure_audit={"confirmatory_claim_allowed": confirmatory},
            raw_sidecars={},
            validation_receipts=[],
            validation_passed=validation,
            phase_spans=[],
            duration_s=0.1,
        )

    assert build(confirmatory=True, validation=True)["verdict_class"] == "positive"
    assert build(confirmatory=False, validation=True)["verdict_class"] == "null"
    assert build(confirmatory=True, validation=False)["verdict_class"] == "disqualified"


def test_independent_helpers_reject_missing_inputs_and_resolve_absolute_path(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-7567 fresh readers fail closed on malformed candidates."""

    with pytest.raises(ValueError, match="independent_reduction_inputs_missing"):
        exp.independent_reduce({})
    absolute = tmp_path / "candidate.json"
    assert exp._argument_path(absolute, exp.REPO_ROOT) == absolute

    broken = exp.fixture_artifact()
    broken["rows"][0]["brier"] += 0.1
    path = tmp_path / "broken.json"
    exp.atomic_json(path, broken)
    errors = exp.independent_replay(path)
    assert any(error.startswith("independent_replay_failed") for error in errors)
