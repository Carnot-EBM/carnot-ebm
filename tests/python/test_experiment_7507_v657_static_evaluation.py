"""Tests for REQ-VERIFY-7507 and SCENARIO-VERIFY-7507-*.

Fixtures use synthetic frozen checkpoints and temporary sidecars. They never
open or rewrite the repository's held-out evaluator.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_7507_v657_static_evaluation as exp


def _features(role: str, count: int, offset: int = 0) -> list[exp.JsonDict]:
    """Build label-free rows whose source identities remain independent."""

    rows = []
    for index in range(count):
        value = index + offset
        center = (value - count / 2) / max(count, 1)
        rows.append(
            {
                "group_id": f"{role}-{index:03d}",
                "role": role,
                "source_hash": f"sha256:{role}-{index:03d}",
                "features": [
                    center,
                    0.1 + 0.01 * (value % 3),
                    0.2 + 0.02 * (value % 4),
                    0.3 + 0.03 * (value % 5),
                    0.04 * (value % 3),
                    0.05 * (value % 4),
                    0.06 * (value % 5),
                    1.0 + 0.1 * (value % 3),
                    5.0 + 0.01 * value,
                    6.0 + 0.02 * value,
                ],
                "raw_whole_expectation": 0.2 + 0.6 * (value % 2),
                "raw_max_window_probability": 0.3 + 0.5 * (value % 2),
            }
        )
    return rows


def _bundle(training: list[exp.JsonDict]) -> exp.JsonDict:
    """Freeze complete candidate grids around one authenticated transform."""

    transform = exp.reconstruct_transform(training)
    dimensions = {
        "window_gibbs": 80,
        "whole_only_gibbs": 1,
        "identical_ten_feature_logistic": 10,
    }
    candidates = []
    selected = {}
    for arm, width in dimensions.items():
        for seed in exp.FIT_SEEDS:
            for regularization in exp.REGULARIZATION_GRID:
                checkpoint = {
                    "coefficient": ([0.01] * width),
                    "bias": -0.05 + (seed - exp.FIT_SEEDS[0]) * 0.001,
                }
                candidates.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "regularization": regularization,
                        "status": "complete",
                        "checkpoint": checkpoint,
                        "checkpoint_sha256": exp.canonical_hash(checkpoint),
                    }
                )
        chosen = next(
            row
            for row in candidates
            if row["arm"] == arm
            and row["seed"] == exp.FIT_SEEDS[0]
            and row["regularization"] == 0.01
        )
        selected[arm] = {**deepcopy(chosen), "parameter_count": width + 1}
    frozen = {
        "schema": "fixture",
        "candidate_rows": candidates,
        "selected_heads": selected,
        "temperature_baseline": {"selected_temperature": 1.5},
        "simple_baseline": {"arm": "raw_whole_expectation"},
        "transform_sha256": exp.canonical_hash(transform),
        "frozen_policies": exp.fixture_policy_manifest(),
    }
    frozen["bundle_sha256"] = exp.canonical_hash(
        {key: value for key, value in frozen.items() if key != "bundle_sha256"}
    )
    return frozen


def _labeled_predictions(groups: int = 40) -> list[exp.JsonDict]:
    """Build complete rows with five genuine seeds for paired reduction."""

    rows = []
    arms = ("window_gibbs", "identical_ten_feature_logistic", "whole_only_gibbs")
    for index in range(groups):
        label = index % 2
        base = 0.82 if label else 0.18
        for arm in arms:
            adjustment = 0.0 if arm == "window_gibbs" else (-0.15 if label else 0.15)
            for seed in exp.FIT_SEEDS:
                probability = min(max(base + adjustment + (seed % 3) * 0.001, 0.01), 0.99)
                rows.append(
                    exp.attach_losses_and_decisions(
                        {
                            "group_id": f"test-{index:03d}",
                            "source_hash": f"sha256:test-{index:03d}",
                            "role": "test",
                            "arm": arm,
                            "fit_seed": seed,
                            "probability": probability,
                        },
                        label,
                    )
                )
        for arm, probability in (
            ("temperature_whole", base),
            ("raw_whole_expectation", base),
            ("raw_max_window_probability", base),
        ):
            rows.append(
                exp.attach_losses_and_decisions(
                    {
                        "group_id": f"test-{index:03d}",
                        "source_hash": f"sha256:test-{index:03d}",
                        "role": "test",
                        "arm": arm,
                        "fit_seed": None,
                        "probability": probability,
                    },
                    label,
                )
            )
    return rows


def test_transform_and_five_seed_predictions_are_frozen() -> None:
    """SCENARIO-VERIFY-7507-PROBABILITY authenticates every frozen seed."""

    training = _features("training", 20)
    test = _features("test", 6, 20)
    bundle = _bundle(training)
    transform = exp.authenticate_transform(training, bundle)
    selected = exp.select_frozen_candidates(bundle)
    assert set(selected) == set(exp.LEARNED_ARMS)
    assert all(len(rows) == 5 for rows in selected.values())
    assert all({row["regularization"] for row in rows} == {0.01} for rows in selected.values())
    predictions = exp.score_label_free_predictions(test, bundle, transform)
    assert len(predictions) == 6 * (5 * 3 + 3)
    assert {row["group_id"] for row in predictions} == {row["group_id"] for row in test}
    assert all("label" not in row for row in predictions)
    assert all(0.0 <= row["probability"] <= 1.0 for row in predictions)
    assert {row["fit_seed"] for row in predictions if row["arm"] == "window_gibbs"} == set(
        exp.FIT_SEEDS
    )


def test_transform_hash_and_candidate_grid_fail_closed() -> None:
    """REQ-VERIFY-7507 rejects unfrozen transforms and favorable seed selection."""

    training = _features("training", 12)
    bundle = _bundle(training)
    changed = deepcopy(bundle)
    changed["transform_sha256"] = "sha256:changed"
    with pytest.raises(ValueError, match="transform_sha256_mismatch"):
        exp.authenticate_transform(training, changed)
    changed = deepcopy(bundle)
    changed["candidate_rows"] = [
        row
        for row in changed["candidate_rows"]
        if not (
            row["arm"] == "window_gibbs"
            and row["seed"] == exp.FIT_SEEDS[0]
            and row["regularization"] == 0.01
        )
    ]
    with pytest.raises(ValueError, match="frozen_seed_grid_invalid"):
        exp.select_frozen_candidates(changed)


def test_prediction_hash_precedes_evaluation_labels(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7507-ACCESS makes label ordering observable in bytes."""

    predictions = [
        {
            "group_id": "test-000",
            "source_hash": "sha256:test-000",
            "role": "test",
            "arm": "raw_whole_expectation",
            "fit_seed": None,
            "probability": 0.25,
        }
    ]
    prediction_path = tmp_path / "prediction.jsonl"
    receipt = exp.write_jsonl_sidecar(prediction_path, predictions, root=tmp_path)
    assert receipt["rows"] == 1
    assert receipt["sha256"] == exp.sha256_file(prediction_path)
    labels = [{"group_id": "test-000", "role": "test", "label": 0}]
    attached = exp.attach_evaluation_labels(predictions, labels)
    assert attached[0]["label"] == 1
    assert attached[0]["brier"] == pytest.approx(0.75**2)
    assert len(attached[0]["decision_costs"]) == 9
    with pytest.raises(ValueError, match="evaluation_group_mismatch"):
        exp.attach_evaluation_labels(predictions, [])


def test_paired_probability_reduction_does_not_count_seeds() -> None:
    """SCENARIO-VERIFY-7507-PROBABILITY counts sources, not fit rows."""

    rows = _labeled_predictions(groups=100)
    reduced = exp.reduce_probability_rows(rows, draws=200, seed=exp.BOOTSTRAP_SEED)
    assert reduced["support"]["n_groups"] == 100
    assert reduced["support"]["class_support"] == {
        "supported": 50,
        "contains_unsupported": 50,
    }
    assert reduced["seed_rows_per_learned_arm"] == 5
    assert set(reduced["brier_contrasts"]) == {
        "identical_ten_feature_logistic",
        "whole_only_gibbs",
    }
    assert all(row["group_count"] == 100 for row in reduced["brier_contrasts"].values())
    assert reduced["log_loss_contrast"]["control"] == "raw_whole_expectation"
    assert reduced["static_probability_value_score"] == 1


def test_probability_gate_requires_support_and_both_controls() -> None:
    """REQ-VERIFY-7507 applies support separately from favorable effect."""

    rows = _labeled_predictions(groups=18)
    reduced = exp.reduce_probability_rows(rows, draws=100, seed=exp.BOOTSTRAP_SEED)
    assert reduced["support"]["passed"] is False
    assert reduced["static_probability_value_score"] == 0
    changed = deepcopy(_labeled_predictions())
    for row in changed:
        if row["arm"] == "identical_ten_feature_logistic":
            row.update(exp.loss_fields(row["probability"], row["label"]))
    reduced = exp.reduce_probability_rows(changed, draws=100, seed=exp.BOOTSTRAP_SEED)
    assert set(reduced["brier_contrasts"]) == set(exp.PROBABILITY_CONTROLS)


def test_nine_policy_cells_retain_actions_costs_and_coverage() -> None:
    """SCENARIO-VERIFY-7507-POLICY keeps escalation distinct from accuracy."""

    rows = _labeled_predictions()
    reduced = exp.reduce_policy_rows(rows, draws=200, seed=exp.BOOTSTRAP_SEED)
    assert len(reduced["cells"]) == 9
    assert reduced["holm_family_size"] == 9
    assert len(reduced["policy_rows"]) == 40 * (5 + 1) * 9
    assert all(row["action"] in exp.TIE_ORDER for row in reduced["policy_rows"])
    assert all(0.0 <= cell["candidate_coverage"] <= 1.0 for cell in reduced["cells"])
    assert all("coverage_matched_sensitivity" in cell for cell in reduced["cells"])
    assert all(cell["comparison"]["group_count"] == 40 for cell in reduced["cells"])
    assert reduced["selective_decision_value_score"] in {0, 1}


def test_independent_reduction_keeps_completion_separate_from_benefit() -> None:
    """SCENARIO-VERIFY-7507-NULL preserves readiness for a valid null."""

    rows = _labeled_predictions()
    settings = exp.evaluator_settings(draws=100)
    reduced = exp.reduce_rows(rows, settings=settings)
    assert reduced["static_evaluation_complete_score"] == 1
    assert reduced["static_probability_value_score"] == 0
    assert reduced["selective_decision_value_score"] in {0, 1}
    assert reduced["sample_size_budget"]["completed"] == 40
    assert reduced["sample_size_budget"]["failed"] == 0
    assert reduced["sample_size_budget"]["censored"] == 0


def test_real_preconditions_authenticate_ready_upstreams() -> None:
    """REQ-VERIFY-7507 checks exact Exp7504 and Exp7505 terminal bytes."""

    checked = exp.collect_preconditions(exp.REPO_ROOT)
    assert checked["passed"] is True
    assert all(row["passed"] for row in checked["rows"])
    assert checked["upstream"]["evidence_ready_score"] == 1
    assert checked["upstream"]["energy_fit_ready_score"] == 1
    assert checked["upstream"]["frozen_before_heldout_label_access"] is True
    assert checked["upstream"]["fresh_confirmatory_claim_allowed"] is False


def test_blocked_artifact_names_exact_external_failure(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7507-BLOCKED reports absence instead of inventing metrics."""

    failed = exp.precondition_row(
        "upstream_ready",
        "Exp7505",
        "energy_fit_ready_score",
        1,
        None,
        path="results/experiment_7505_v657_energy_fit.json",
    )
    artifact = exp.blocked_artifact(failed, root=tmp_path)
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["static_evaluation_complete_score"] == 0
    assert artifact["static_probability_value_score"] == 0
    assert artifact["selective_decision_value_score"] == 0
    assert artifact["gate_check_summary"]["observed"] is None
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert exp.validate_artifact(artifact, root=tmp_path, verify_sources=False) == []
    changed = deepcopy(artifact)
    changed["static_evaluation_complete_score"] = 1
    assert "blocked_score_mismatch" in exp.validate_artifact(
        changed, root=tmp_path, verify_sources=False
    )
    changed = deepcopy(artifact)
    changed["honest_verdict"] = "complete_null_wrong"
    assert "blocked_verdict_invalid" in exp.validate_artifact(
        changed, root=tmp_path, verify_sources=False
    )


def test_jsonl_sidecar_and_checksum_detect_mutation(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7507-E2E binds raw evidence and terminal content."""

    path = tmp_path / "rows.jsonl"
    rows = _labeled_predictions(groups=4)
    reference = exp.write_jsonl_sidecar(path, rows, root=tmp_path)
    assert exp.load_jsonl(path) == rows
    assert exp.verify_sidecar(reference, tmp_path) == []
    path.write_text("{}\n", encoding="utf-8")
    assert exp.verify_sidecar(reference, tmp_path) == [f"sidecar_sha256_mismatch:{path.name}"]
    value = {"schema": "fixture", "reproducibility_checksum": "pending"}
    value["reproducibility_checksum"] = exp.artifact_checksum(value)
    assert value["reproducibility_checksum"] == exp.artifact_checksum(value)


def test_cli_parser_requires_frozen_date() -> None:
    """REQ-VERIFY-7507 fixes the run date for replay identity."""

    assert exp.parse_args(["--date", exp.RUN_DATE]).date == exp.RUN_DATE
    with pytest.raises(SystemExit):
        exp.parse_args([])


def test_artifact_round_trip_reduces_raw_sidecars(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7507-E2E cold-reduces exact serialized per-unit rows."""

    rows = _labeled_predictions(groups=12)
    settings = exp.evaluator_settings(draws=20, expected_groups=12)
    reduced = exp.reduce_rows(rows, settings=settings)
    predictions = [
        {
            key: row[key]
            for key in ("group_id", "source_hash", "role", "arm", "fit_seed", "probability")
        }
        for row in rows
    ]
    prediction_path = tmp_path / "predictions.jsonl"
    row_path = tmp_path / "rows.jsonl"
    policy_path = tmp_path / "policy.jsonl"
    prediction_receipt = exp.write_jsonl_sidecar(prediction_path, predictions, root=tmp_path)
    row_receipt = exp.write_jsonl_sidecar(row_path, rows, root=tmp_path)
    policy_receipt = exp.write_jsonl_sidecar(policy_path, reduced["policy_rows"], root=tmp_path)
    artifact = exp.build_artifact(
        preconditions=[exp.precondition_row("fixture_ready", "fixture", "ready", True, True)],
        source_hashes=[],
        rows=rows,
        reduced=reduced,
        settings=settings,
        raw_sidecars={
            "label_free_predictions": prediction_receipt,
            "evaluation_rows": row_receipt,
            "policy_rows": policy_receipt,
        },
        prediction_receipt=prediction_receipt,
        label_access_receipt={"prediction_written_before_label_access": True},
        validation_receipts=[{"name": "fixture", "passed": True, "exit_code": 0}],
        validation_passed=True,
        phase_spans=[],
        started_at="2026-09-22T00:00:00+00:00",
        completed_at="2026-09-22T00:00:01+00:00",
        duration_s=1.0,
        historical_model_provenance={"scope": "historical"},
    )
    assert artifact["static_evaluation_complete_score"] == 1
    assert artifact["static_probability_value_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["execution_venue"] == "host"
    assert exp.validate_artifact(artifact, root=tmp_path, verify_sources=False) == []
    reduced_again = exp.independent_reduce(artifact)
    assert reduced_again["sample_size_budget"] == artifact["sample_size_budget"]
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.cold_replay(candidate, root=tmp_path, verify_sources=False) == []
    changed = deepcopy(artifact)
    changed["static_evaluation_complete_score"] = 0
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(
        changed, root=tmp_path, verify_sources=False
    )
    assert "independent_reduction_mismatch:static_evaluation_complete_score" in (
        exp.validate_artifact(changed, root=tmp_path, verify_sources=False)
    )
    mutations = (
        ("schema", "wrong", "artifact_identity_invalid"),
        ("run_date", "wrong", "artifact_date_invalid"),
        ("terminal_status", "partial", "terminal_status_invalid"),
        ("MODEL_SPECS", ["model"], "model_specs_nonempty"),
        ("model_invoked", True, "model_invoked_invalid"),
        ("invocation_counts", {}, "invocation_counts_invalid"),
        ("inference_substrate", "wrong", "inference_substrate_invalid"),
        ("inference_substrate_class", "wrong", "inference_substrate_class_invalid"),
        ("honest_verdict", "unfinished", "honest_verdict_not_terminal"),
        ("verdict_class", "wrong", "verdict_class_invalid"),
        ("static_probability_value_score", 2, "score_not_bare_binary"),
        ("field_principles", {}, "field_principles_incomplete"),
    )
    for field, replacement, expected_error in mutations:
        changed = deepcopy(artifact)
        changed[field] = replacement
        assert expected_error in exp.validate_artifact(changed, root=tmp_path, verify_sources=False)
    incomplete = deepcopy(artifact)
    incomplete.pop("schema")
    assert exp.validate_artifact(incomplete, root=tmp_path, verify_sources=False)[0].startswith(
        "required_fields_missing"
    )
    no_rows = deepcopy(artifact)
    no_rows["rows"] = []
    assert "evaluation_rows_missing" in exp.validate_artifact(
        no_rows, root=tmp_path, verify_sources=False
    )
    missing_operands = deepcopy(artifact)
    missing_operands["rows"] = "wrong"
    with pytest.raises(ValueError, match="reduction_operands_missing"):
        exp.independent_reduce(missing_operands)
    changed_policy = deepcopy(artifact)
    changed_policy["policy_rows"] = []
    assert "independent_reduction_mismatch:policy_rows" in exp.validate_artifact(
        changed_policy, root=tmp_path, verify_sources=False
    )
    bad_reduce = deepcopy(artifact)
    bad_reduce["rows"] = [row for row in rows if row["arm"] != "window_gibbs"]
    assert any(
        error.startswith("independent_reduction_failed")
        for error in exp.validate_artifact(bad_reduce, root=tmp_path, verify_sources=False)
    )
    source = tmp_path / "source.txt"
    source.write_text("bound", encoding="utf-8")
    with_source = deepcopy(artifact)
    with_source["source_artifact_hashes"] = [exp._source_row(source, tmp_path)]
    with_source["reproducibility_checksum"] = exp.artifact_checksum(with_source)
    source.write_text("changed", encoding="utf-8")
    assert "source_hash_mismatch:source.txt" in exp.validate_artifact(with_source, root=tmp_path)
    source.unlink()
    assert "source_missing:source.txt" in exp.validate_artifact(with_source, root=tmp_path)
    row_path.write_text("{}\n", encoding="utf-8")
    assert "evaluation_sidecar_content_mismatch" in exp.validate_artifact(
        artifact, root=tmp_path, verify_sources=False
    )
    exp.write_jsonl_sidecar(row_path, rows, root=tmp_path)
    prediction_path.write_text('{"label":0}\n', encoding="utf-8")
    assert "prediction_sidecar_contains_label" in exp.validate_artifact(
        artifact, root=tmp_path, verify_sources=False
    )


def test_fail_closed_numeric_and_access_guards(tmp_path: Path) -> None:
    """REQ-VERIFY-7507 rejects malformed frozen, label, and bootstrap operands."""

    object_path = tmp_path / "object.json"
    object_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object_required"):
        exp.load_json(object_path)
    jsonl_path = tmp_path / "rows.jsonl"
    jsonl_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jsonl_object_required"):
        exp.load_jsonl(jsonl_path)
    assert exp.verify_sidecar({"path": "missing.jsonl", "sha256": "x"}, tmp_path) == [
        "sidecar_missing:missing.jsonl"
    ]
    valid = tmp_path / "valid.jsonl"
    valid.write_text("{}\n", encoding="utf-8")
    assert exp.verify_sidecar(
        {"path": "valid.jsonl", "sha256": exp.sha256_file(valid), "rows": 2}, tmp_path
    ) == ["sidecar_row_count_mismatch:valid.jsonl"]
    with pytest.raises(ValueError, match="training_feature_shape_invalid"):
        exp.reconstruct_transform([])
    bad = _features("training", 4)
    bad[0]["features"][0] = float("nan")
    with pytest.raises(ValueError, match="training_feature_nonfinite"):
        exp.reconstruct_transform(bad)
    with pytest.raises(ValueError, match="bootstrap_shape_invalid"):
        exp.bootstrap_indices(0, draws=1, seed=1)
    with pytest.raises(ValueError, match="probability_or_label_invalid"):
        exp.loss_fields(float("nan"), 0)
    with pytest.raises(ValueError, match="label_visible_during_prediction"):
        exp.score_label_free_predictions(
            [{**_features("test", 1)[0], "label": 0}], _bundle(_features("training", 8)), {}
        )


def test_fail_closed_checkpoint_label_and_pairing_guards(tmp_path: Path) -> None:
    """REQ-VERIFY-7507 fails before malformed rows can enter inference."""

    training = _features("training", 8)
    bundle = _bundle(training)
    with pytest.raises(ValueError, match="frozen_candidate_manifest_missing"):
        exp.select_frozen_candidates({})
    changed = deepcopy(bundle)
    chosen = next(
        row
        for row in changed["candidate_rows"]
        if row["arm"] == "window_gibbs" and row["regularization"] == 0.01
    )
    chosen["checkpoint_sha256"] = "sha256:changed"
    with pytest.raises(ValueError, match="checkpoint_sha256_mismatch"):
        exp.select_frozen_candidates(changed)
    transform = exp.reconstruct_transform(training)
    with pytest.raises(ValueError, match="evaluation_feature_shape_invalid"):
        exp._designs([{"features": [1.0]}], transform)
    with pytest.raises(ValueError, match="checkpoint_coefficient_shape_invalid"):
        exp._checkpoint_probabilities({"coefficient": [1.0], "bias": 0.0}, np.ones((2, 3)))
    with pytest.raises(ValueError, match="checkpoint_bias_nonfinite"):
        exp._checkpoint_probabilities(
            {"coefficient": [1.0, 1.0, 1.0], "bias": float("nan")}, np.ones((2, 3))
        )
    with pytest.raises(ValueError, match="decision_action_invalid"):
        exp._cost("unknown", 0, false_accept=1.0, escalation=0.1)
    predictions = [
        {
            "group_id": "g",
            "source_hash": "s",
            "role": "test",
            "arm": "raw_whole_expectation",
            "fit_seed": None,
            "probability": 0.2,
        }
    ]
    labels = [{"group_id": "g", "role": "test", "label": 0}]
    assert len(exp.attach_evaluation_labels(predictions, [{"role": "training"}, *labels])) == 1
    with pytest.raises(ValueError, match="evaluation_label_duplicate"):
        exp.attach_evaluation_labels(predictions, [*labels, *labels])
    with pytest.raises(ValueError, match="evaluation_label_invalid"):
        exp.attach_evaluation_labels(predictions, [{**labels[0], "label": 2}])
    rows = _labeled_predictions(groups=4)
    changed_rows = deepcopy(rows)
    changed_rows[1]["source_hash"] = "different"
    with pytest.raises(ValueError, match="source_row_disagreement"):
        exp._averaged_arm_rows(changed_rows)
    with pytest.raises(ValueError, match="paired_inference_shape_invalid"):
        exp._paired_inference([1.0], np.zeros((2, 2), dtype=int), seed=1)
    with pytest.raises(ValueError, match="window_gibbs_rows_missing"):
        exp.reduce_probability_rows([], draws=2, seed=1)
    changed_rows = [
        row
        for row in rows
        if not (
            row["arm"] == "window_gibbs"
            and row["group_id"] == "test-000"
            and row["fit_seed"] == exp.FIT_SEEDS[0]
        )
    ]
    with pytest.raises(ValueError, match="seed_rows_invalid"):
        exp.reduce_probability_rows(changed_rows, draws=2, seed=1)
    with pytest.raises(ValueError, match="decision_cell_invalid"):
        exp._cell({}, "missing")
    with pytest.raises(ValueError, match="policy_group_mismatch"):
        exp.reduce_policy_rows([], draws=2, seed=1)
    without_seed = [
        row
        for row in rows
        if not (
            row["arm"] == "window_gibbs"
            and row["group_id"] == "test-000"
            and row["fit_seed"] == exp.FIT_SEEDS[0]
        )
    ]
    with pytest.raises(ValueError, match="policy_seed_rows_invalid"):
        exp.reduce_policy_rows(without_seed, draws=2, seed=1)
    assert exp._path_label(Path("/tmp/outside"), tmp_path).startswith("/tmp/")
    assert exp._gate_summary([{"check": "ok", "passed": True}])["all_passed"] is True
    assert exp.classify_terminal(0, 0, 0, True)[1] == "disqualified"
    assert exp.classify_terminal(1, 1, 0, True)[1] == "positive"
    assert exp.classify_terminal(1, 0, 0, False)[0].endswith("prior_exposure")
    assert exp.classify_terminal(1, 0, 0, True)[1] == "null"
