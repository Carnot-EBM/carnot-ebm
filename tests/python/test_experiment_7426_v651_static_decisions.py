"""Tests for REQ-AUTO-7426 static human source-support decisions."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import time

import numpy as np
import pytest

from carnot import experiment_7426_v651_static_decisions as exp


ROOT = Path(__file__).resolve().parents[2]


def _predictor(
    key: str,
    group: str,
    partition: str,
    *,
    task_type: str = "QA",
    feature_offset: float = 0.0,
) -> dict[str, object]:
    """Build one label-free row with the exact six-feature contract."""

    value = min(max(0.2 + feature_offset, 0.0), 1.0)
    return {
        "row_key": key,
        "group_id": group,
        "partition": partition,
        "task_type": task_type,
        "source_text": f"source {group} says value 10 and alpha",
        "response_text": f"answer {key} says value {10 if value < 0.5 else 11} and alpha",
        "source_token_count": 7,
        "response_token_count": 7,
        "source_truncated": False,
        "response_truncated": False,
        "certificate_selected": key.endswith("-0"),
        "features": {
            "numeric_novelty_with_context": value,
            "falsifiability_score": 0.1,
            "normalized_number_token_overlap": 1.0 - value,
            "normalized_content_token_overlap": 0.4 + value / 10,
            "max_answer_source_sentence_overlap": 0.3 + value / 10,
            "missing_or_empty_source": 0.0,
        },
    }


def _evaluator(
    predictor: dict[str, object], label: int, *, sensitivity: int | None = None
) -> dict[str, object]:
    """Build the separate human-label view for one predictor row."""

    return {
        "row_key": predictor["row_key"],
        "group_id": predictor["group_id"],
        "partition": predictor["partition"],
        "task_type": predictor["task_type"],
        "primary_label": label,
        "implicit_true_excluded_label": label if sensitivity is None else sensitivity,
        "certificate_selected": predictor["certificate_selected"],
    }


def _rows(groups_per_partition: int = 8) -> list[dict[str, object]]:
    """Return joined rows with both labels in each sealed role."""

    predictors: list[dict[str, object]] = []
    evaluators: list[dict[str, object]] = []
    for partition_index, partition in enumerate(exp.PARTITIONS):
        for index in range(groups_per_partition):
            predictor = _predictor(
                f"{partition}-{index}-0",
                f"{partition}-group-{index}",
                partition,
                task_type=("QA", "Summary", "Data2txt")[index % 3],
                feature_offset=(index + partition_index) / (groups_per_partition * 2),
            )
            predictors.append(predictor)
            evaluators.append(_evaluator(predictor, index % 2))
    return exp.join_predictor_labels(predictors, evaluators)


def _passing_receipts() -> list[dict[str, object]]:
    """Return the required affected and terminal receipt names."""

    return [
        {
            "name": name,
            "required": True,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
        }
        for name in (*exp.AFFECTED_CHECK_NAMES, *exp.TERMINAL_CHECK_NAMES)
    ]


def test_req_auto_7426_spec_precedes_implementation() -> None:
    """REQ-AUTO-7426: the implementation has a complete driving contract."""

    text = (ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "### REQ-AUTO-7426:" in text
    for number in range(1, 11):
        assert f"SCENARIO-AUTO-7426-{number:02d}" in text


def test_scenario_auto_7426_01_preconditions_authenticate_both_upstreams() -> None:
    """SCENARIO-AUTO-7426-01: exact ready, allowed, unflagged inputs pass."""

    checks, hashes, loaded = exp.collect_preconditions(ROOT)
    assert checks and all(row["passed"] is True for row in checks)
    assert set(loaded) == {"annotated_protocol", "spline_prototype", "corpus_manifest"}
    assert hashes[exp.ANNOTATED_PATH.as_posix()]["sha256"] == exp.EXPECTED_ANNOTATED_SHA256
    assert hashes[exp.SPLINE_PATH.as_posix()]["sha256"] == exp.EXPECTED_SPLINE_SHA256

    changed = deepcopy(loaded["annotated_protocol"])
    changed["annotated_protocol_ready_score"] = 0
    failed = exp.upstream_field_checks(changed, loaded["spline_prototype"])
    row = next(item for item in failed if item["check"] == "annotated_protocol_ready")
    assert row["passed"] is False
    blocked = exp.build_blocked_artifact(row)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["blocked_observed"] == 0


def test_scenario_auto_7426_02_support_probability_controls_action_direction() -> None:
    """SCENARIO-AUTO-7426-02: high support accepts and low support rejects."""

    policy = {
        "accept_threshold": 0.95,
        "reject_threshold": 0.05,
        "accept_enabled": True,
        "reject_enabled": True,
    }
    assert exp.typed_support_decision(0.98, policy)["action"] == "accept"
    assert exp.typed_support_decision(0.02, policy)["action"] == "reject"
    assert exp.typed_support_decision(0.50, policy)["action"] == "escalate"
    disabled = {**policy, "accept_enabled": False, "reject_enabled": False}
    assert exp.typed_support_decision(0.98, disabled)["action"] == "escalate"
    assert exp.typed_support_decision(None, policy)["action"] == "escalate"


def test_scenario_auto_7426_03_seed_average_precedes_affine_calibration() -> None:
    """SCENARIO-AUTO-7426-03: calibration consumes the mean seed probability."""

    raw = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5],
            [0.4, 0.5, 0.6],
            [0.5, 0.6, 0.7],
        ]
    )
    calibrated = exp.fit_ensemble_calibration(raw, [0, 1, 1], steps=2)
    assert calibrated["seed_count"] == 5
    assert calibrated["mean_raw_probabilities"] == pytest.approx([0.3, 0.4, 0.5])
    expected_logits = [exp.logit(value) for value in (0.3, 0.4, 0.5)]
    assert calibrated["calibration_logits"] == pytest.approx(expected_logits)


def test_scenario_auto_7426_04_sparse_dense_training_parity() -> None:
    """SCENARIO-AUTO-7426-04: both fixed-basis parameterizations remain equal."""

    rows = _rows()
    training = [row for row in rows if row["partition"] == "fit"]
    x = exp.feature_matrix(training)
    y = np.asarray([row["label"] for row in training], dtype=np.float64)
    pair = exp.fit_spline_pair(x, y, seed=65101, steps=3)
    assert pair["parameter_count"] == 49
    assert pair["updates"] == 3
    assert pair["max_parameter_gap"] <= exp.PARITY_TOLERANCE
    assert pair["max_probability_gap"] <= exp.PARITY_TOLERANCE
    assert pair["sparse_checkpoint"]["initial_state_sha256"]
    assert len(pair["loss_trace"]) == 4

    changed = deepcopy(pair)
    changed["dense_checkpoint"]["coef"][0] += 1e-4
    assert exp.spline_parity_errors(changed)


def test_scenario_auto_7426_05_policy_uses_one_group_representative() -> None:
    """SCENARIO-AUTO-7426-05: duplicates cannot inflate exact certificates."""

    rows: list[dict[str, object]] = []
    probabilities: dict[str, float] = {}
    for index in range(180):
        for sibling in range(2):
            key = f"row-{index:03d}-{sibling}"
            rows.append(
                {
                    "row_key": key,
                    "group_id": f"group-{index:03d}",
                    "label": 1,
                    "partition": "policy_calibration",
                }
            )
            probabilities[key] = 0.99
    candidates, selected = exp.select_policy(rows, probabilities)
    assert len(candidates) == 9
    assert all(row["representative_groups"] == 180 for row in candidates)
    assert selected["accept_enabled"] is True
    assert selected["reject_enabled"] is False
    assert selected["accept_certificate"]["simultaneous_test_count"] == 90
    assert exp.exact_risk_certificate([], risk_budget=0.05)["action_enabled"] is False


def test_scenario_auto_7426_07_source_controls_are_predictor_only() -> None:
    """SCENARIO-AUTO-7426-07: controls transform text before any label join."""

    predictors = [
        _predictor("a-0", "a", "fit", task_type="QA"),
        _predictor("b-0", "b", "fit", task_type="QA", feature_offset=0.2),
    ]
    masked = exp.condition_predictors(predictors, "source_masked")
    swapped = exp.condition_predictors(predictors, "source_swapped")
    assert all(row["features"]["missing_or_empty_source"] == 1.0 for row in masked)
    assert swapped[0]["source_text"] == predictors[1]["source_text"]
    assert swapped[1]["source_text"] == predictors[0]["source_text"]
    assert all("label" not in row for row in [*masked, *swapped])
    with pytest.raises(ValueError, match="two groups"):
        exp.condition_predictors(predictors[:1], "source_swapped")
    with pytest.raises(ValueError, match="condition"):
        exp.condition_predictors(predictors, "unknown")


def test_scenario_auto_7426_06_scoring_and_shards_preserve_all_rows(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7426-06: detailed rows and exclusions survive sharding."""

    rows = _rows()
    fit = exp.fit_condition(rows, condition="full_source", steps=2)
    official = [row for row in rows if row["partition"] == "final_test"]
    probability_rows, ensemble_rows = exp.score_official_rows(fit, official)
    assert len(probability_rows) == len(official) * len(exp.ARMS) * len(exp.TRAINING_SEEDS)
    assert len(ensemble_rows) == len(official) * len(exp.ARMS)
    assert all(row["authority"] == exp.LABEL_AUTHORITY for row in probability_rows)
    assert all(math.isfinite(row["brier_contribution"]) for row in probability_rows)

    exclusions = exp.excluded_probability_rows(
        [{"response_id": "bad-1", "reason": "excluded_malformed_annotation"}]
    )
    assert len(exclusions) == len(exp.CONDITIONS) * len(exp.ARMS) * len(exp.TRAINING_SEEDS)
    assert all(row["probability"] is None and row["status"] == "excluded" for row in exclusions)
    manifests = exp.write_detail_shards(tmp_path, [*probability_rows, *exclusions])
    loaded = exp.load_detail_shards(tmp_path, manifests)
    assert loaded == [*probability_rows, *exclusions]
    changed = deepcopy(manifests)
    changed[0]["sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="hash"):
        exp.load_detail_shards(tmp_path, changed)


def test_scenario_auto_7426_08_grouped_bootstrap_and_value_gate() -> None:
    """SCENARIO-AUTO-7426-08: two corrected primary contrasts reproduce."""

    rows: list[dict[str, object]] = []
    arm_probability = {
        "training_prevalence": 0.55,
        "raw_l2_logistic": 0.65,
        "gibbs_6_4_1": 0.62,
        "sparse_spline_49": 0.90,
        "dense_spline_logistic": 0.90,
    }
    for group_index in range(20):
        label = group_index % 2
        for arm, positive_probability in arm_probability.items():
            probability = positive_probability if label else 1.0 - positive_probability
            rows.append(
                exp.ensemble_metric_row(
                    arm=arm,
                    row_key=f"row-{group_index}",
                    group_id=f"group-{group_index}",
                    task_type="QA",
                    label=label,
                    probability=probability,
                    action="accept" if label else "reject",
                )
            )
    intervals = exp.grouped_bootstrap(rows, draws=200, seed=7426)
    assert intervals["primary_contrasts"] == ["raw_l2_logistic", "gibbs_6_4_1"]
    assert intervals["simultaneous_correction_count"] == 2
    metrics = exp.reduce_metrics(rows)
    value = exp.reduce_decision_value(intervals, metrics, parity_passed=True)
    assert value["checks"]["spline_basis_parity"] is True
    assert value["checks"]["simultaneous_brier_improvement"] is True
    assert value["decision_value_score"] == 1
    assert (
        exp.reduce_decision_value(intervals, metrics, parity_passed=False)["decision_value_score"]
        == 0
    )


def test_scenario_auto_7426_09_and_10_artifact_reduction_fails_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-AUTO-7426-09/10: capture and hashes reduce independently."""

    artifact = exp.build_fixture_artifact(tmp_path, validation_receipts=_passing_receipts())
    assert exp.validate_artifact(artifact, root=tmp_path) == []
    reduced = exp.independent_reduce(artifact, root=tmp_path)
    assert reduced["decision_capture_complete_score"] == 1
    assert artifact["verdict_class"] in {"positive", "null"}
    assert artifact["promotion_score"] == 0

    changed = deepcopy(artifact)
    changed["MODEL_SPECS"] = ["not-allowed"]
    assert "declaration_mismatch:MODEL_SPECS" in exp.validate_artifact(changed, root=tmp_path)
    changed = deepcopy(artifact)
    changed["rows"][0]["status"] = "failed"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed, root=tmp_path)


def test_req_auto_7426_defensive_shapes_and_cli_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-AUTO-7426: malformed inputs and reader modes fail closed."""

    rows = _rows()
    with pytest.raises(ValueError, match="six"):
        exp.feature_matrix([{**rows[0], "features": {"bad": 1.0}}])
    with pytest.raises(ValueError, match="both labels"):
        exp.fit_spline_pair(np.zeros((2, 6)), np.ones(2), seed=1, steps=1)
    with pytest.raises(ValueError, match="draws"):
        exp.grouped_bootstrap([], draws=0)
    with pytest.raises(ValueError, match="probability"):
        exp.logit(float("nan"))

    artifact = exp.build_fixture_artifact(tmp_path, validation_receipts=_passing_receipts())
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert (
        exp.main(["--date", exp.RUN_DATE, "--cold-replay", str(path), "--root", str(tmp_path)]) == 0
    )
    assert (
        exp.main(
            ["--date", exp.RUN_DATE, "--independent-reduce", str(path), "--root", str(tmp_path)]
        )
        == 0
    )

    called: dict[str, object] = {}

    def fake_run(root: Path, run_date: str, *, output_path: Path) -> dict[str, object]:
        called.update(root=root, run_date=run_date, output_path=output_path)
        return {}

    monkeypatch.setattr(exp, "run_experiment", fake_run)
    assert exp.main(["--date", exp.RUN_DATE, "--root", str(tmp_path)]) == 0
    assert called["root"] == tmp_path


def test_req_auto_7426_input_mutations_cover_fail_closed_branches(tmp_path: Path) -> None:
    """REQ-AUTO-7426: malformed joins, fits, and certificates are rejected."""

    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    assert exp._load_object(missing) == {}
    assert exp._load_object(malformed) == {}
    assert exp._load_object(scalar) == {}

    predictor = _predictor("row-0", "group-0", "fit")
    evaluator = _evaluator(predictor, 0)
    assert exp.condition_predictors([predictor], "full_source") == [predictor]
    with pytest.raises(ValueError, match="unique"):
        exp.join_predictor_labels([predictor], [evaluator, evaluator])
    with pytest.raises(ValueError, match="missing"):
        exp.join_predictor_labels([predictor], [])
    with pytest.raises(ValueError, match="label invalid"):
        exp.join_predictor_labels([predictor], [{**evaluator, "primary_label": None}])
    with pytest.raises(ValueError, match="identity mismatch"):
        exp.join_predictor_labels([predictor], [{**evaluator, "group_id": "wrong"}])
    extra = _predictor("row-1", "group-1", "fit")
    with pytest.raises(ValueError, match="outside"):
        exp.join_predictor_labels([predictor], [evaluator, _evaluator(extra, 1)])

    bad_features = deepcopy(predictor)
    bad_features["features"]["numeric_novelty_with_context"] = math.inf
    with pytest.raises(ValueError, match="bounded"):
        exp.feature_matrix([bad_features])
    with pytest.raises(ValueError, match="six columns"):
        exp.feature_matrix([])

    with pytest.raises(ValueError, match="six-column"):
        exp.fit_spline_pair(np.zeros((2, 5)), np.asarray([0, 1]), seed=1, steps=1)
    nonfinite = np.zeros((2, 6))
    nonfinite[0, 0] = math.nan
    with pytest.raises(ValueError, match="finite"):
        exp.fit_spline_pair(nonfinite, np.asarray([0, 1]), seed=1, steps=1)
    with pytest.raises(ValueError, match="between zero"):
        exp.fit_spline_pair(np.zeros((2, 6)), np.asarray([0, 1]), seed=1, steps=501)
    beats: list[int] = []
    exp.fit_spline_pair(
        np.vstack((np.zeros((1, 6)), np.ones((1, 6)))),
        np.asarray([0, 1]),
        seed=1,
        steps=100,
        heartbeat=beats.append,
    )
    assert beats == [100]

    valid_pair = exp.fit_spline_pair(
        np.vstack((np.zeros((1, 6)), np.ones((1, 6)))),
        np.asarray([0, 1]),
        seed=1,
        steps=1,
    )
    changed = deepcopy(valid_pair)
    changed["max_probability_gap"] = 1.0
    changed["max_gradient_gap"] = 1.0
    errors = exp.spline_parity_errors(changed)
    assert "spline_probability_parity_mismatch" in errors
    assert "spline_gradient_parity_mismatch" in errors
    assert exp.spline_parity_errors({}) == ["spline_parity_state_invalid"]

    with pytest.raises(ValueError, match="five seed"):
        exp.fit_ensemble_calibration(np.zeros((4, 3)), [0, 1, 0], steps=1)
    bad_raw = np.zeros((5, 3))
    bad_raw[0, 0] = math.inf
    with pytest.raises(ValueError, match="bounded"):
        exp.fit_ensemble_calibration(bad_raw, [0, 1, 0], steps=1)
    with pytest.raises(ValueError, match="both labels"):
        exp.fit_ensemble_calibration(np.full((5, 3), 0.5), [1, 1, 1], steps=1)
    with pytest.raises(ValueError, match="binary"):
        exp.exact_risk_certificate([2], risk_budget=0.05)

    policy = {
        "accept_threshold": 0.95,
        "reject_threshold": 0.05,
        "accept_enabled": False,
        "reject_enabled": False,
    }
    assert exp.typed_support_decision(0.01, policy)["reason"] == "reject_uncertified"
    with pytest.raises(ValueError, match="identity"):
        exp.select_policy([{"row_key": "", "group_id": "", "label": 0}], {"": 0.5})

    low_rows = [
        {
            "row_key": f"low-{index}",
            "group_id": f"low-group-{index}",
            "label": 0,
            "partition": "policy_calibration",
        }
        for index in range(180)
    ]
    _candidates, low_policy = exp.select_policy(
        low_rows, {str(row["row_key"]): 0.01 for row in low_rows}
    )
    assert low_policy["reject_enabled"] is True
    one_class = _rows()
    for row in one_class:
        if row["partition"] == "fit":
            row["label"] = 1
    with pytest.raises(ValueError, match="both labels"):
        exp.fit_condition(one_class, condition="full_source", steps=0)
    with pytest.raises(ValueError, match="condition"):
        exp.fit_condition(_rows(), condition="not-registered", steps=0)


def test_req_auto_7426_progress_callbacks_and_scoring_guards() -> None:
    """REQ-AUTO-7426: unit progress and final-role boundaries stay explicit."""

    events: list[tuple[str, str, int, int]] = []
    development_rows = [row for row in _rows() if row["partition"] != "final_test"]
    fit = exp.fit_condition(
        development_rows, condition="full_source", steps=0, emit=lambda *x: events.append(x)
    )
    assert len(events) == 40
    assert {event[0] for event in events} == set(exp.ARMS)
    with pytest.raises(ValueError, match="final_test"):
        exp.score_official_rows(fit, [row for row in _rows() if row["partition"] == "fit"])


def test_req_auto_7426_shard_and_bootstrap_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-AUTO-7426: shard bytes and paired group coverage fail closed."""

    assert exp.write_detail_shards(tmp_path / "empty", []) == []
    monkeypatch.setattr(exp, "MAX_SHARD_BYTES", 100)
    with pytest.raises(ValueError, match="single detail row"):
        exp.write_detail_shards(tmp_path / "large", [{"value": "x" * 200}])
    rows = [{"value": f"row-{index}"} for index in range(20)]
    manifests = exp.write_detail_shards(tmp_path / "split", rows)
    assert len(manifests) > 1

    size_changed = deepcopy(manifests)
    size_changed[0]["size_bytes"] += 1
    with pytest.raises(ValueError, match="size"):
        exp.load_detail_shards(tmp_path / "split", size_changed)

    bad_path = tmp_path / "split" / manifests[0]["path"]
    bad_path.write_text("{\n", encoding="utf-8")
    bad_json = deepcopy(manifests[:1])
    bad_json[0]["sha256"] = exp.sha256_file(bad_path)
    bad_json[0]["size_bytes"] = bad_path.stat().st_size
    with pytest.raises(ValueError, match="JSON"):
        exp.load_detail_shards(tmp_path / "split", bad_json)

    valid = exp.write_detail_shards(tmp_path / "row-count", [{"value": 1}])
    valid[0]["rows"] = 2
    with pytest.raises(ValueError, match="row"):
        exp.load_detail_shards(tmp_path / "row-count", valid)

    assert exp.reduce_metrics([]) == {"by_arm": {}}
    with pytest.raises(ValueError, match="source groups"):
        exp.grouped_bootstrap([], draws=1)
    one = [
        exp.ensemble_metric_row(
            arm="training_prevalence",
            row_key="r",
            group_id="g",
            task_type="QA",
            label=1,
            probability=0.5,
            action="escalate",
        )
    ]
    with pytest.raises(ValueError, match="arm missing"):
        exp.grouped_bootstrap(one, draws=1)
    paired = []
    for arm in exp.ARMS:
        paired.append(
            exp.ensemble_metric_row(
                arm=arm,
                row_key="r1",
                group_id="g1",
                task_type="QA",
                label=1,
                probability=0.5,
                action="escalate",
            )
        )
        if arm != exp.ARMS[-1]:
            paired.append(
                exp.ensemble_metric_row(
                    arm=arm,
                    row_key="r2",
                    group_id="g2",
                    task_type="QA",
                    label=0,
                    probability=0.5,
                    action="escalate",
                )
            )
    with pytest.raises(ValueError, match="paired group"):
        exp.grouped_bootstrap(paired, draws=1)


def test_req_auto_7426_artifact_validation_mutations(tmp_path: Path) -> None:
    """REQ-AUTO-7426: each material artifact mutation is named."""

    artifact = exp.build_fixture_artifact(tmp_path, validation_receipts=_passing_receipts())
    mutations: list[tuple[str, object, str]] = [
        ("verdict_class", "unknown", "verdict_class_invalid"),
        ("field_principles", {}, "field_principles_mismatch"),
        ("calibration_metrics", {}, "calibration_metrics_mismatch"),
        ("grouped_intervals", {}, "grouped_intervals_mismatch"),
        ("decision_value_reduction", {}, "decision_value_reduction_mismatch"),
    ]
    for field, value, expected in mutations:
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected in exp.validate_artifact(changed, root=tmp_path)

    blocked = deepcopy(artifact)
    blocked["verdict_class"] = "blocked"
    blocked["honest_verdict"] = "wrong"
    assert "blocked_verdict_prefix_invalid" in exp.validate_artifact(blocked, root=tmp_path)

    missing_shards = deepcopy(artifact)
    missing_shards["detail_row_directory"] = "missing"
    errors = exp.validate_artifact(missing_shards, root=tmp_path)
    assert any(error.startswith("independent_reduction_failed:") for error in errors)

    bad_checkpoint = deepcopy(artifact)
    bad_checkpoint["checkpoint_manifest"][0]["sha256"] = "sha256:" + "0" * 64
    assert any(
        error.startswith("checkpoint_bytes_mismatch:")
        for error in exp.validate_artifact(bad_checkpoint, root=tmp_path)
    )

    source_mutation = deepcopy(artifact)
    source_mutation["fixture_artifact"] = False
    source_mutation["source_artifact_hashes"] = {
        "invalid": 1,
        "missing": {"path": "missing", "sha256": "sha256:" + "0" * 64},
    }
    errors = exp.validate_artifact(source_mutation, root=tmp_path)
    assert "source_artifact_hash_row_invalid" in errors
    assert "source_artifact_hash_mismatch:missing" in errors

    external = tmp_path.parent / f"{tmp_path.name}-external-checkpoints"
    fit = exp.fit_condition(_rows(), condition="full_source", steps=0)
    manifests = exp.write_checkpoints(tmp_path, external, [fit])
    assert Path(manifests[0]["path"]).is_absolute()
    checkpoint = json.loads(Path(manifests[0]["path"]).read_text(encoding="utf-8"))
    assert (
        exp.canonical_hash(checkpoint["initial_checkpoint"])
        == checkpoint["initial_checkpoint_sha256"]
    )

    official = [row for row in _rows() if row["partition"] == "final_test"]
    probabilities, metrics = exp.score_official_rows(fit, official)
    external_detail = tmp_path / "external-detail"
    external_metric = tmp_path / "external-metric"
    detail_shards = exp.write_detail_shards(external_detail, probabilities)
    metric_shards = exp.write_detail_shards(external_metric, metrics, prefix="metrics")
    nested_root = tmp_path / "nested-root"
    nested_root.mkdir()
    external_artifact = exp.build_artifact(
        root=nested_root,
        fits=[fit],
        checkpoint_manifest=manifests,
        detail_shards=detail_shards,
        metric_shards=metric_shards,
        detail_directory=external_detail,
        metric_directory=external_metric,
        preconditions=[
            exp._precondition("fixture", "fixture", "fixture", "ready", "==", True, True)
        ],
        source_hashes={},
        validation_receipts=_passing_receipts(),
        phase_spans=[],
        started_at_utc="2026-09-19T00:00:00+00:00",
        completed_at_utc="2026-09-19T00:00:01+00:00",
        started_ns=1,
        ended_ns=2,
        excluded_count=0,
        fixture=True,
    )
    assert Path(external_artifact["detail_row_directory"]).is_absolute()
    assert Path(external_artifact["metric_row_directory"]).is_absolute()

    receipt = exp._receipt(
        started_ns=1,
        ended_ns=2,
        spans=[],
        training={"performed": False},
        root=ROOT,
    )
    assert len(receipt["receipt_sidecars"]) == 2
    span = exp._span("test", time.monotonic(), time.monotonic(), 1)
    assert span["completed_units"] == 1

    with pytest.raises(SystemExit, match="--date"):
        exp.main(["--date", "wrong", "--root", str(tmp_path)])
    unreadable = tmp_path / "unreadable.json"
    assert exp.cold_replay(unreadable, root=tmp_path) == ["artifact_unreadable_or_not_object"]
