"""Tests for the V650 calibrated precision and placement experiment.

Spec refs: REQ-REPORT-7419 and SCENARIO-REPORT-7419-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot import experiment_7419_v650_precision_placement as experiment
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES


ROOT = Path(__file__).resolve().parents[2]


def _checkpoint() -> dict[str, Any]:
    """Return a small frozen source-aware head for numeric unit tests."""

    return {
        "schema": "carnot.exp7413.numeric_checkpoint.v1",
        "arm": "source_aware_6_4_1_gibbs",
        "condition": "full_source",
        "seed": 65001,
        "architecture": [6, 4, 1],
        "weights": {
            "w1": [
                [0.9, -0.2, 0.4, -0.8, -0.9, 0.1],
                [-0.3, -0.1, -0.2, 1.1, 1.0, -0.1],
                [0.5, -0.1, 0.3, -0.7, -0.8, 0.1],
                [0.8, -0.1, 0.4, -0.9, -0.8, -0.1],
            ],
            "b1": [1.4, -0.2, 1.1, 1.4],
            "w_out": [1.3, -0.9, 1.0, 1.4],
            "b_out": 0.56,
        },
        "affine": {"slope": 1.2, "intercept": -0.13},
        "selected_policy": {
            "accept_threshold": 0.01,
            "reject_threshold": 0.9,
            "accept_enabled": False,
            "reject_enabled": False,
        },
    }


def _feature_row(index: int, partition: str, label: int | None) -> dict[str, Any]:
    """Build one bounded source-feature row without identity leakage."""

    values = np.asarray(
        [
            (index % 7) / 7,
            (index % 5) / 5,
            (index % 3) / 3,
            ((index + 1) % 7) / 7,
            ((index + 2) % 5) / 5,
            float(index % 2),
        ],
        dtype=np.float64,
    )
    return {
        "row_key": f"row-{index}",
        "group_id": f"group-{index // 2}",
        "partition": partition,
        "source_features": dict(zip(experiment.SOURCE_FEATURE_NAMES, values.tolist(), strict=True)),
        "label": label,
        "label_authority": "machine_annotation",
    }


def _rows() -> list[dict[str, Any]]:
    """Provide train-only scale rows and labeled plus unlabeled held-out rows."""

    return [
        *[_feature_row(index, "train", index % 2) for index in range(16)],
        *[_feature_row(100 + index, "final_test", index % 2) for index in range(8)],
        _feature_row(200, "final_test", None),
    ]


# REQ-REPORT-7419 / SCENARIO-REPORT-7419-BRANCH
def test_preconditions_authenticate_numeric_branch_and_preserve_board_block(tmp_path: Path) -> None:
    paths = experiment.ExperimentPaths.under(tmp_path)
    checks, hashes, context = experiment.collect_preconditions(
        ROOT,
        paths,
        candidate_paths=[tmp_path / "missing-receipt.md"],
    )

    numeric = [row for row in checks if row["category"] == "numeric_precondition"]
    assert numeric and all(row["passed"] for row in numeric)
    assert context["numeric_branch"]["eligible"] is True
    assert context["numeric_branch"]["checkpoint_seed"] == 65001
    assert context["numeric_branch"]["heldout_rows"] == 1995
    assert context["gatemate_changed_state"]["exists"] is False
    assert hashes[experiment.EXP7413_PATH.as_posix()] == experiment.EXPECTED_EXP7413_SHA256
    assert hashes[experiment.CHECKPOINT_PATH.as_posix()] == experiment.EXPECTED_CHECKPOINT_SHA256
    fixture = experiment.load_precision_fixture(ROOT)
    assert len([row for row in fixture["rows"] if row["partition"] == "final_test"]) == 1995


# REQ-REPORT-7419 / SCENARIO-REPORT-7419-QUANTIZATION
def test_quantization_scales_use_only_train_features_and_frozen_weights() -> None:
    rows = _rows()
    checkpoint = _checkpoint()
    first = experiment.derive_quantization(checkpoint, rows)
    changed = deepcopy(rows)
    for row in changed:
        if row["partition"] == "final_test":
            row["source_features"] = {name: 1.0 for name in experiment.SOURCE_FEATURE_NAMES}
            row["label"] = 1
    second = experiment.derive_quantization(checkpoint, changed)

    assert first == second
    assert first["scale_source"] == "checkpoint_weights_and_train_features_only"
    assert first["feature_row_count"] == 16
    assert all(value > 0 for value in first["weight_scales"].values())
    assert first["feature_scale"] > 0

    with pytest.raises(ValueError, match="training features"):
        experiment.derive_quantization(
            checkpoint, [row for row in rows if row["partition"] != "train"]
        )
    bad = deepcopy(checkpoint)
    bad["weights"]["w1"] = [[1.0]]
    with pytest.raises(ValueError, match="checkpoint"):
        experiment.derive_quantization(bad, rows)


# REQ-REPORT-7419 / SCENARIO-REPORT-7419-PARITY
def test_three_numeric_paths_retain_probabilities_actions_and_adverse_fixture() -> None:
    rows = _rows()
    checkpoint = _checkpoint()
    quantization = experiment.derive_quantization(checkpoint, rows)
    heldout = [row for row in rows if row["partition"] == "final_test"]
    matrix = experiment.feature_matrix(heldout)

    scalar = experiment.score_probabilities(matrix, checkpoint, "float64_scalar", quantization)
    vector = experiment.score_probabilities(matrix, checkpoint, "float32_vectorized", quantization)
    int8 = experiment.score_probabilities(matrix, checkpoint, "int8_float32_accum", quantization)
    precision_rows = experiment.build_precision_rows(heldout, checkpoint, quantization)
    reduction = experiment.reduce_precision(precision_rows)
    boundary = experiment.near_threshold_fixture(checkpoint, quantization)

    assert scalar.shape == vector.shape == int8.shape == (9,)
    assert np.isfinite(scalar).all() and np.isfinite(vector).all() and np.isfinite(int8).all()
    assert len(precision_rows) == 27
    assert reduction["invalid_numeric_count"] == 0
    assert reduction["measured_action_flip_count"] == 0
    assert reduction["unscored_rows_preserved"] == 3
    assert all(row["representation_bytes"] > 0 for row in precision_rows)
    assert any(row["adverse_rounding"] for row in boundary)
    assert {row["threshold_name"] for row in boundary} == {"accept", "reject"}

    with pytest.raises(ValueError, match="arm"):
        experiment.score_probabilities(matrix, checkpoint, "device", quantization)
    with pytest.raises(ValueError, match="feature matrix"):
        experiment.score_probabilities(
            np.asarray([1.0]), checkpoint, "float64_scalar", quantization
        )


# REQ-REPORT-7419 / SCENARIO-REPORT-7419-TIMING
def test_rotated_timing_covers_full_boundary_and_cold_initialization() -> None:
    rows = _rows()
    checkpoint = _checkpoint()
    quantization = experiment.derive_quantization(checkpoint, rows)
    heldout = [row for row in rows if row["partition"] == "final_test"]
    warm, cold = experiment.measure_service_cost(
        heldout * 16,
        checkpoint,
        quantization,
        batch_sizes=(1, 32, 128),
        blocks=2,
    )

    assert len(warm) == 18
    assert len(cold) == 3
    assert {row["arm"] for row in cold} == set(experiment.ARMS)
    for batch_size in (1, 32, 128):
        selected = [row for row in warm if row["batch_size"] == batch_size]
        assert {row["block"] for row in selected} == {0, 1}
        assert {row["arm"] for row in selected} == set(experiment.ARMS)
        assert len({tuple(row["rotated_arm_order"]) for row in selected}) == 2
    assert all(set(row["stage_durations_s"]) == set(experiment.STAGE_NAMES) for row in warm)
    assert all(row["total_service_s"] > 0 and row["serialized_bytes"] > 0 for row in warm)
    assert all(row["row_sha256"] == experiment.row_hash(row) for row in warm)

    with pytest.raises(ValueError, match="batch"):
        experiment.measure_service_cost(
            heldout, checkpoint, quantization, batch_sizes=(2,), blocks=1
        )


# REQ-REPORT-7419 / SCENARIO-REPORT-7419-TIMING
def test_paired_speed_gate_uses_upper_bound_and_keeps_numeric_null_valid() -> None:
    fast: list[dict[str, Any]] = []
    slow: list[dict[str, Any]] = []
    for block in range(30):
        for arm, total in (
            ("float64_scalar", 3.0),
            ("float32_vectorized", 2.0),
            ("int8_float32_accum", 1.0),
        ):
            fast.append(experiment.synthetic_timing_row(128, block, arm, total))
            slow.append(
                experiment.synthetic_timing_row(
                    128,
                    block,
                    arm,
                    2.5 if arm == "int8_float32_accum" else total,
                )
            )

    fast_summary = experiment.summarize_timing(fast, draws=500, seed=experiment.RANDOM_SEED)
    slow_summary = experiment.summarize_timing(slow, draws=500, seed=experiment.RANDOM_SEED)
    assert fast_summary["primary_ratio"]["ci95"][1] < 1
    assert fast_summary["speed_gate_passed"] is True
    assert slow_summary["primary_ratio"]["ci95"][1] >= 1
    assert slow_summary["speed_gate_passed"] is False

    with pytest.raises(ValueError, match="paired timing"):
        experiment.summarize_timing(fast[:-1], draws=10, seed=1)


# REQ-REPORT-7419 / SCENARIO-REPORT-7419-PLACEMENT
def test_amdahl_bound_is_feasibility_not_achieved_speed() -> None:
    rows = [
        experiment.synthetic_timing_row(128, block, "float32_vectorized", 100.0, scoring_s=80.0)
        for block in range(30)
    ]
    result = experiment.compute_amdahl(rows, batch_size=128, arm="float32_vectorized")

    assert result["unaccelerated_fraction"] == pytest.approx(0.2)
    assert result["infinite_accelerated_stage_upper_bound"] == pytest.approx(5.0)
    assert result["hundred_x_feasibility"] == {
        "operator": "<=",
        "expected": 0.01,
        "observed": pytest.approx(0.2),
        "passed": False,
    }
    assert result["achieved_device_acceleration"] is False


# REQ-REPORT-7419 / SCENARIO-REPORT-7419-BOARDS
def test_board_rows_preserve_three_independent_dispositions(tmp_path: Path) -> None:
    paths = experiment.ExperimentPaths.under(tmp_path)
    _, _, context = experiment.collect_preconditions(
        ROOT,
        paths,
        candidate_paths=[tmp_path / "missing.md"],
    )
    rows = experiment.build_board_rows(context)
    boards = {row["board"]: row for row in rows}

    assert set(boards) == {"KV260", "GateMate", "PolarFire"}
    assert boards["KV260"]["terminal_state"] == "graduated_preserved"
    assert boards["KV260"]["future_access"] == "ssh_only"
    assert boards["KV260"]["architecture_limit"] == "k_max<=5"
    assert boards["PolarFire"]["terminal_state"] == "graduated_cpu_dispatch_preserved"
    assert boards["PolarFire"]["fpga_sampling_claimed"] is False
    assert boards["GateMate"]["terminal_state"] == "blocked_changed_physical_state"
    assert all(row["hardware_ready_score"] == row["hardware_value_score"] == 0 for row in rows)
    assert all(row["new_hardware_execution_claimed"] is False for row in rows)


# REQ-REPORT-7419 / SCENARIO-REPORT-7419-ARTIFACT
def test_terminal_fixture_validates_and_material_mutations_fail_closed(tmp_path: Path) -> None:
    artifact = experiment.build_fixture_artifact()

    assert experiment.validate_artifact(artifact) == []
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == experiment.ZERO_INVOCATION_COUNTS
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["execution_venue"] == "host"
    assert artifact["precision_capture_complete_score"] == 1
    assert artifact["precision_value_score"] == 0
    assert artifact["hardware_ready_score"] == 0
    assert artifact["hardware_value_score"] == 0
    assert artifact["promotion_score"] == 0
    assert artifact["honest_verdict"].startswith("complete_")

    changed = deepcopy(artifact)
    changed["precision_rows"][0]["probability"] = 0.5
    changed["reproducibility_checksum"] = experiment.reproducibility_checksum(changed)
    assert "precision_reduction_mismatch" in experiment.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["timing_rows"][0]["row_sha256"] = "sha256:wrong"
    changed["reproducibility_checksum"] = experiment.reproducibility_checksum(changed)
    assert "timing_row_hash_mismatch" in experiment.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["hardware_ready_score"] = 1
    changed["reproducibility_checksum"] = experiment.reproducibility_checksum(changed)
    assert "fixed_declaration_mismatch" in experiment.validate_artifact(changed)

    path = tmp_path / "artifact.json"
    receipt = experiment.write_artifact(path, artifact)
    assert receipt["sha256"] == experiment.sha256_file(path)
    assert json.loads(path.read_text(encoding="utf-8")) == artifact


# REQ-REPORT-7419 / SCENARIO-REPORT-7419-BRANCH
def test_blocked_numeric_branch_keeps_board_accounting() -> None:
    artifact = experiment.build_fixture_artifact(numeric_available=False)

    assert experiment.validate_artifact(artifact) == []
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_numeric_calibration_branch_unavailable"
    assert artifact["precision_capture_complete_score"] == 0
    assert len(artifact["board_rows"]) == 3
    assert artifact["gate_check_summary"]["numeric_branch_blocker"] is not None
    assert artifact["gate_check_summary"]["gatemate_changed_state_prerequisite"] is not None


# REQ-REPORT-7419 / SCENARIO-REPORT-7419-ARTIFACT
def test_independent_reduction_scoped_plan_and_cli_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = experiment.build_fixture_artifact()
    raw = {
        "precision_rows": deepcopy(artifact["precision_rows"]),
        "timing_rows": deepcopy(artifact["timing_rows"]),
        "cold_initialization_rows": deepcopy(artifact["cold_initialization_rows"]),
        "board_rows": deepcopy(artifact["board_rows"]),
        "quantization": deepcopy(artifact["quantization"]),
    }
    assert all(experiment.independent_reduce(artifact, raw).values())

    commands = experiment.build_validation_plan(ROOT, tmp_path / "private")
    assert [command.name for command in commands] == list(REQUIRED_CHECK_NAMES)
    assert experiment.validate_validation_plan(ROOT, commands) == []
    focused = next(command for command in commands if command.name == "focused_pytest")
    assert {"-n", "0", "-o", "addopts=", "--no-cov"}.issubset(focused.argv)
    coverage = next(command for command in commands if command.name == "changed_module_coverage")
    assert any(argument.startswith("--basetemp=") for argument in coverage.argv)
    assert all(command.name != "full_python_suite" for command in commands)

    candidate = tmp_path / "candidate.json"
    experiment.atomic_json(candidate, artifact)
    assert experiment.main(["--date", experiment.RUN_DATE, "--cold-replay", str(candidate)]) == 0
    assert (
        experiment.main(["--date", experiment.RUN_DATE, "--independent-reduce", str(candidate)])
        == 0
    )
    monkeypatch.setattr(experiment, "run_experiment", lambda *_args, **_kwargs: {})
    assert experiment.main(["--date", experiment.RUN_DATE, "--output", str(tmp_path / "out")]) == 0


# REQ-REPORT-7419 / SCENARIO-REPORT-7419-ARTIFACT
def test_numeric_and_plan_guards_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable_json"):
        experiment.load_json(bad_json)
    with pytest.raises(ValueError, match="board_row_not_unique"):
        experiment._board_by_name([], "KV260")

    checkpoint = _checkpoint()
    rows = _rows()
    quantization = experiment.derive_quantization(checkpoint, rows)
    matrix = experiment.feature_matrix(rows[:1])
    with pytest.raises(ValueError, match="checkpoint fields"):
        experiment._weight_arrays({})
    with pytest.raises(ValueError, match="source feature keys"):
        experiment.feature_matrix([{"source_features": {}}])
    bad_feature = deepcopy(rows[0])
    bad_feature["source_features"][experiment.SOURCE_FEATURE_NAMES[0]] = 2.0
    with pytest.raises(ValueError, match="finite values"):
        experiment.feature_matrix([bad_feature])
    with pytest.raises(ValueError, match="shape"):
        experiment.feature_matrix([])
    with pytest.raises(ValueError, match="quantization receipt"):
        experiment.score_probabilities(matrix, checkpoint, "int8_float32_accum", {})
    with pytest.raises(ValueError, match="precision rows"):
        experiment.reduce_precision([])
    precision = experiment.build_precision_rows(
        [row for row in rows if row["partition"] == "final_test"], checkpoint, quantization
    )
    with pytest.raises(ValueError, match="arm coverage"):
        experiment.reduce_precision(precision[:-1])

    monkeypatch.setattr(np, "nextafter", lambda value, _toward: value)
    boundary_checkpoint = deepcopy(checkpoint)
    boundary_checkpoint["selected_policy"]["accept_threshold"] = 0.5
    boundary_checkpoint["selected_policy"]["reject_threshold"] = 0.5
    with pytest.raises(ValueError, match="adverse rounding"):
        experiment.near_threshold_fixture(boundary_checkpoint, quantization)
    monkeypatch.undo()

    with pytest.raises(ValueError, match="registered arm"):
        experiment._run_service(rows[:1], checkpoint, quantization, "device")
    experiment.measure_service_cost(
        rows,
        checkpoint,
        quantization,
        batch_sizes=(1,),
        blocks=1,
        emit_progress=True,
    )
    with pytest.raises(ValueError, match="arm and duration"):
        experiment.synthetic_timing_row(1, 0, "device", 1.0)
    with pytest.raises(ValueError, match="fit the total"):
        experiment.synthetic_timing_row(1, 0, "float64_scalar", 1.0, scoring_s=2.0)
    with pytest.raises(ValueError, match="common nonempty shape"):
        experiment._bootstrap_ratio(np.asarray([]), np.asarray([]), 1, 1)
    with pytest.raises(ValueError, match="positive"):
        experiment._bootstrap_ratio(np.asarray([1.0]), np.asarray([0.0]), 0, 1)
    with pytest.raises(ValueError, match="paired timing rows"):
        experiment.summarize_timing([])
    timing = [experiment.synthetic_timing_row(128, 0, arm, 1.0) for arm in experiment.ARMS]
    with pytest.raises(ValueError, match="duplicate"):
        experiment.summarize_timing([*timing, timing[0]])
    with pytest.raises(ValueError, match="matching timing"):
        experiment.compute_amdahl([], batch_size=128, arm="float32_vectorized")

    with pytest.raises(ValueError, match="three authenticated"):
        experiment.build_board_rows({})
    boards = experiment._fixture_board_rows()
    changed_context = {
        "historical_board_rows": boards,
        "gatemate_changed_state": {"exists": True, "date": experiment.RUN_DATE},
    }
    changed_boards = experiment.build_board_rows(changed_context)
    gate = next(row for row in changed_boards if row["board"] == "GateMate")
    assert gate["terminal_state"] == "changed_physical_state_future_prerequisite"

    assert experiment._required_validation_passed([]) is False
    partial_terminal = experiment._fixture_receipts()[:-1]
    assert experiment._required_validation_passed(partial_terminal) is False
    assert experiment._source_hash_rows({"x": None})["x"]["sha256"] is None

    monkeypatch.setattr(
        experiment,
        "_board_by_name",
        lambda *_args: (_ for _ in ()).throw(ValueError("missing")),
    )
    checks, _, _ = experiment.collect_preconditions(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path / "bad-boards"),
        candidate_paths=[tmp_path / "absent.md"],
    )
    assert all(row["passed"] is False for row in checks if row["category"] == "board_accounting")


# REQ-REPORT-7419 / SCENARIO-REPORT-7419-ARTIFACT
def test_artifact_classifies_disqualified_and_circular_positive() -> None:
    base = experiment.build_fixture_artifact()
    common = {
        "precision_rows": base["precision_rows"],
        "cold_rows": base["cold_initialization_rows"],
        "board_rows": base["board_rows"],
        "quantization": base["quantization"],
        "boundary_rows": base["near_threshold_fixture"],
        "preconditions": base["preconditions_checked"],
        "source_hashes": base["source_artifact_hashes"],
        "phase_spans": base["phase_spans"],
        "started_at_utc": base["started_at_utc"],
        "completed_at_utc": base["completed_at_utc"],
        "duration_ns": 1_000_000_000,
    }
    disqualified = experiment.assemble_artifact(
        **common,
        timing_rows=base["timing_rows"],
        validation_receipts=[],
    )
    assert disqualified["verdict_class"] == "disqualified"

    fast = [
        experiment.synthetic_timing_row(
            batch,
            block,
            arm,
            {"float64_scalar": 3.0, "float32_vectorized": 2.0, "int8_float32_accum": 1.0}[arm],
        )
        for batch in experiment.BATCH_SIZES
        for block in range(experiment.PAIRED_BLOCKS)
        for arm in experiment.ARMS
    ]
    positive = experiment.assemble_artifact(
        **common,
        timing_rows=fast,
        validation_receipts=experiment._fixture_receipts(),
        fixture=True,
    )
    assert positive["verdict_class"] == "circular_positive"
    assert positive["precision_value_score"] == 1


# REQ-REPORT-7419 / SCENARIO-REPORT-7419-ARTIFACT
def test_artifact_validator_exercises_each_terminal_guard(tmp_path: Path) -> None:
    artifact = experiment.build_fixture_artifact()

    def errors_after(change: Any, *, refresh_checksum: bool = True) -> list[str]:
        changed = deepcopy(artifact)
        change(changed)
        if refresh_checksum:
            changed["reproducibility_checksum"] = experiment.reproducibility_checksum(changed)
        return experiment.validate_artifact(changed)

    assert experiment.validate_artifact([]) == ["artifact_not_object"]
    assert any(
        value.startswith("required_fields_missing")
        for value in errors_after(lambda x: x.pop("status"))
    )
    assert "identity_mismatch" in errors_after(lambda x: x.__setitem__("schema", "wrong"))
    assert "verdict_class_invalid" in errors_after(
        lambda x: x.__setitem__("verdict_class", "unknown")
    )
    assert "board_rows_invalid" in errors_after(lambda x: x.__setitem__("board_rows", []))
    assert "board_row_hash_or_score_mismatch" in errors_after(
        lambda x: x["board_rows"][0].__setitem__("hardware_ready_score", 1)
    )
    assert "precision_rows_invalid" in errors_after(lambda x: x.__setitem__("precision_rows", [{}]))
    assert "precision_reduction_mismatch" in errors_after(
        lambda x: x.__setitem__("precision_rows", [])
    )
    assert "timing_summary_mismatch" in errors_after(
        lambda x: x["timing_summary"].__setitem__("speed_gate_passed", True)
    )
    assert "amdahl_analysis_mismatch" in errors_after(
        lambda x: x["amdahl_analysis"].__setitem__("achieved_device_acceleration", True)
    )

    def invalid_timing(x: dict[str, Any]) -> None:
        x["timing_rows"] = [experiment.synthetic_timing_row(128, 0, "float64_scalar", 1.0)]

    assert "timing_rows_invalid" in errors_after(invalid_timing)

    def missing_timing(x: dict[str, Any]) -> None:
        x["timing_rows"] = []

    assert "timing_summary_mismatch" in errors_after(missing_timing)

    blocked = experiment.build_fixture_artifact(numeric_available=False)
    blocked["honest_verdict"] = "wrong"
    blocked["reproducibility_checksum"] = experiment.reproducibility_checksum(blocked)
    assert "blocked_disposition_invalid" in experiment.validate_artifact(blocked)
    assert "precision_scores_invalid" in errors_after(
        lambda x: x.__setitem__("precision_capture_complete_score", 2)
    )

    def bad_capture(x: dict[str, Any]) -> None:
        x["timing_rows"] = []
        x["timing_summary"] = None
        x["amdahl_analysis"] = None

    assert "capture_evidence_invalid" in errors_after(bad_capture)
    assert "value_score_mismatch" in errors_after(
        lambda x: x.__setitem__("precision_value_score", 1)
    )
    assert "field_principles_mismatch" in errors_after(
        lambda x: x["field_principles"].pop("status")
    )
    assert "reproducibility_checksum_mismatch" in errors_after(
        lambda x: x.__setitem__("status", "changed"), refresh_checksum=False
    )
    with pytest.raises(ValueError, match="artifact_validation_failed"):
        experiment.write_artifact(tmp_path / "invalid.json", {"bad": True})

    source_checked = deepcopy(artifact)
    source_checked["fixture_artifact"] = False
    source_checked["source_artifact_hashes"] = {"bad": "not-a-row"}
    source_checked["field_principles"] = experiment._field_principles(
        [*source_checked, "field_principles", "reproducibility_checksum"]
    )
    source_checked["reproducibility_checksum"] = experiment.reproducibility_checksum(source_checked)
    assert "source_artifact_hash_row_invalid" in experiment.validate_artifact(source_checked)
    source_checked["source_artifact_hashes"] = {
        "missing": {
            "path": str(tmp_path / "missing"),
            "sha256": "sha256:" + "0" * 64,
            "original_flagged_adversarial": None,
        }
    }
    source_checked["reproducibility_checksum"] = experiment.reproducibility_checksum(source_checked)
    assert any(
        value.startswith("source_artifact_hash_mismatch")
        for value in experiment.validate_artifact(source_checked)
    )


# REQ-REPORT-7419 / SCENARIO-REPORT-7419-BRANCH
def test_loader_reader_and_receipt_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert experiment.cold_replay(tmp_path / "missing.json") == [
        "artifact_unreadable_or_not_object"
    ]

    original_load = experiment.load_json
    monkeypatch.setattr(experiment, "load_json", lambda _path: {})
    with pytest.raises(ValueError, match="feature records"):
        experiment.load_precision_fixture(ROOT)

    payloads = {
        experiment.EXP7413_PATH.name: {"paired_metric_rows": []},
        experiment.CHECKPOINT_PATH.name: _checkpoint(),
        experiment.FEATURE_PATH.name: {
            "records": [{"partition": "final_test", "row_key": "missing"}]
        },
    }
    monkeypatch.setattr(experiment, "load_json", lambda path: payloads[path.name])
    with pytest.raises(ValueError, match="metric row"):
        experiment.load_precision_fixture(ROOT)
    payloads[experiment.FEATURE_PATH.name] = {"records": [{"partition": "train"}]}
    with pytest.raises(ValueError, match="held-out cohort"):
        experiment.load_precision_fixture(ROOT)
    monkeypatch.setattr(experiment, "load_json", original_load)

    paths = experiment.ExperimentPaths.under(tmp_path)
    sidecar = experiment._historical_sidecar(tmp_path, paths)
    assert sidecar["scope"] == "historical_model_receipts"
    raw_inside = tmp_path / "inside.json"
    experiment.atomic_json(raw_inside, {"value": 1})
    assert experiment._raw_receipt(tmp_path, raw_inside)["path"] == "inside.json"
    raw_outside = tmp_path.parent / "outside-exp7419.json"
    experiment.atomic_json(raw_outside, {"value": 1})
    assert Path(experiment._raw_receipt(tmp_path, raw_outside)["path"]).is_absolute()

    with pytest.raises(SystemExit, match="--date"):
        experiment.main(["--date", "wrong"])
    nonobject = tmp_path / "nonobject.json"
    experiment.atomic_json(nonobject, {"placeholder": True})
    nonobject.write_text("[]\n", encoding="utf-8")
    assert (
        experiment.main(["--date", experiment.RUN_DATE, "--independent-reduce", str(nonobject)])
        == 1
    )
    candidate = tmp_path / "candidate.json"
    artifact = experiment.build_fixture_artifact()
    experiment.atomic_json(candidate, artifact)
    wrong_raw = tmp_path / "wrong-raw.json"
    experiment.atomic_json(wrong_raw, {"bad": True})
    assert (
        experiment.main(
            [
                "--date",
                experiment.RUN_DATE,
                "--independent-reduce",
                str(candidate),
                "--raw",
                str(wrong_raw),
            ]
        )
        == 1
    )
