"""Spec-linked tests for REQ-KAN-7432 sparse update placement."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_7432_v651_update_placement as exp


ROOT = Path(__file__).resolve().parents[2]


def _training() -> np.ndarray:
    return np.asarray(
        [[((row + 2 * column) % 11) / 10.0 for column in range(6)] for row in range(16)],
        dtype=np.float64,
    )


def _checkpoint() -> dict[str, object]:
    return exp.fixture_checkpoint(_training())


def _events(count: int = 5) -> list[dict[str, object]]:
    return [
        {
            "observation_id": f"fixture-{index}",
            "features": _training()[index % 16].tolist(),
            "label": index % 2,
            "revealed": index != 2,
            "prediction_index": index,
            "available_at": index + (2 if index == 1 else 0),
            "source_event_hash": f"sha256:{index + 1:064x}",
        }
        for index in range(count)
    ]


def test_req_kan_7432_spec_and_real_preconditions_authenticate() -> None:
    """REQ-KAN-7432: inputs are exact and board accounting is independent."""

    spec = (ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "REQ-KAN-7432" in spec
    checks, hashes, context = exp.collect_preconditions(ROOT)
    numeric = [row for row in checks if row["category"] == "numeric_precondition"]
    assert numeric and all(row["passed"] for row in numeric)
    assert exp.EXP7425_PATH.as_posix() in hashes
    assert exp.EXP7426_PATH.as_posix() in hashes
    assert exp.EXP7427_PATH.as_posix() in hashes
    assert context["numeric_branch"]["eligible"] is True
    boards = {row["board"]: row for row in exp.build_board_rows(context)}
    assert set(boards) == {"KV260", "GateMate", "PolarFire"}
    assert boards["KV260"]["future_access"] == "ssh_only"
    assert boards["KV260"]["architecture_limit"] == "k_max<=5"
    assert boards["GateMate"]["terminal_state"] == "blocked_changed_physical_state"
    assert boards["PolarFire"]["terminal_state"] == "graduated_cpu_dispatch_preserved"


def test_scenario_7432_02_scale_is_training_only_and_int32_is_checked() -> None:
    """SCENARIO-KAN-7432-02: scale provenance and overflow fail closed."""

    checkpoint = _checkpoint()
    scale = exp.derive_fixed_point_scale(checkpoint, _training())
    assert scale["source_partition"] == "fit_training_only"
    assert scale["label_count"] == 0
    assert scale["coefficient_step"] > 0
    vector = np.full(24, 120, dtype=np.int16)
    basis = np.full(24, 100, dtype=np.int16)
    value, overflow = exp.checked_int32_dot(vector, basis)
    assert value == 288_000 and overflow is False
    huge = np.full(24, 32_767, dtype=np.int16)
    _, overflow = exp.checked_int32_dot(huge, huge)
    assert overflow is True
    with pytest.raises(ValueError, match="matching one-dimensional"):
        exp.checked_int32_dot(np.ones(2), np.ones(3))


def test_scenarios_7432_03_04_replay_boundaries_order_overflow_and_restart(
    tmp_path: Path,
) -> None:
    """SCENARIO-KAN-7432-03/04: replay keeps masks, order, and durable state."""

    checkpoint = _checkpoint()
    events = _events()
    scale = exp.derive_fixed_point_scale(checkpoint, _training())
    rows, states = exp.replay_update_arms(checkpoint, events, scale)
    assert len(rows) == len(events) * len(exp.ARMS)
    assert {row["arm"] for row in rows} == set(exp.ARMS)
    assert [row["observation_id"] for row in rows[:: len(exp.ARMS)]] == [
        row["observation_id"] for row in events
    ]
    masked = [row for row in rows if row["observation_id"] == "fixture-2"]
    assert all(row["update_applied"] is False for row in masked)
    assert all(row["unsafe_wraparound"] is False for row in rows)
    assert all("coefficient_error_linf" in row and "probability_error" in row for row in rows)

    controls = exp.run_development_controls(checkpoint, scale, tmp_path)
    assert all(row["passed"] for row in controls)
    assert {row["control"] for row in controls} == {
        "near_threshold_actions",
        "knot_boundary_values",
        "overflow_detection",
        "delayed_event_order",
        "checkpoint_restart",
    }
    assert states["float64_dense"]["update_count"] == 4


def test_scenario_7432_05_rotated_complete_service_timing_and_summary(
    tmp_path: Path,
) -> None:
    """SCENARIO-KAN-7432-05: paired rows include persistence stages."""

    checkpoint = _checkpoint()
    scale = exp.derive_fixed_point_scale(checkpoint, _training())
    rows, cold = exp.measure_service_cost(
        _events(3), checkpoint, scale, tmp_path, batch_sizes=(1, 2), blocks=3
    )
    assert len(rows) == 2 * 3 * len(exp.ARMS)
    assert len(cold) == len(exp.ARMS)
    for row in rows:
        assert row["complete_service_s"] >= sum(row["stage_duration_s"].values())
        assert set(row["stage_duration_s"]) == {
            "feature_evaluation",
            "update",
            "journal_write_fsync",
            "checkpoint_write_fsync",
        }
    assert [row["arm"] for row in rows if row["batch_size"] == 1 and row["block"] == 0] == list(
        exp.ARMS
    )
    assert [row["arm"] for row in rows if row["batch_size"] == 1 and row["block"] == 1] == [
        *exp.ARMS[1:],
        exp.ARMS[0],
    ]
    summary = exp.summarize_timing(rows, draws=200, seed=7432)
    assert len(summary) == 2 * (len(exp.ARMS) - 1)
    assert all(row["paired_blocks"] == 3 for row in summary)


def test_scenario_7432_06_completion_is_independent_from_value() -> None:
    """SCENARIO-KAN-7432-06: parity and CI gates can produce an honest null."""

    artifact = exp.build_fixture_artifact()
    assert artifact["update_placement_complete_score"] == 1
    assert artifact["update_placement_value_score"] in (0, 1)
    changed = deepcopy(artifact)
    fixed = next(row for row in changed["update_rows"] if row["arm"] == "int16_fixed_sparse")
    fixed["action_flip"] = True
    changed = exp.rebuild_fixture_reductions(changed)
    assert changed["update_placement_complete_score"] == 1
    assert changed["update_placement_value_score"] == 0
    assert changed["verdict_class"] == "null"
    assert changed["honest_verdict"].startswith("complete_null")


def test_scenarios_7432_07_08_artifact_reduction_and_mutations_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-KAN-7432-07/08: readers bind rows, scores, and host limits."""

    artifact = exp.build_fixture_artifact()
    assert exp.validate_artifact(artifact) == []
    raw = exp.raw_view(artifact)
    assert all(exp.independent_reduce(artifact, raw).values())
    assert artifact["hardware_ready_score"] == 0
    assert artifact["hardware_value_score"] == 0
    assert artifact["promotion_score"] == 0
    assert artifact["execution_venue"] == "host"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert all(value == 0 for value in artifact["invocation_counts"].values())
    assert artifact["amdahl_analysis"]["required_unaccelerated_fraction"] == 0.01

    candidate = tmp_path / "candidate.json"
    exp.write_artifact(candidate, artifact)
    assert exp.cold_replay(candidate) == []
    mutated = deepcopy(artifact)
    mutated["hardware_value_score"] = 1
    assert "hardware_score_nonzero" in exp.validate_artifact(mutated)
    mutated = deepcopy(artifact)
    mutated["update_rows"][0]["probability"] += 0.01
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(mutated)


def test_validation_manifest_and_cli_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-KAN-7432: validation stays scoped and the wrapper modes are strict."""

    commands = exp.build_validation_plan(ROOT, tmp_path / "private")
    assert exp.validate_validation_plan(ROOT, commands) == []
    assert all("tests/python" not in command.argv for command in commands)
    assert any(exp.TEST_PATH.as_posix() in command.argv for command in commands)
    artifact = exp.build_fixture_artifact()
    candidate = tmp_path / "candidate.json"
    raw = tmp_path / "raw.json"
    exp.write_artifact(candidate, artifact)
    raw.write_text(json.dumps(exp.raw_view(artifact)), encoding="utf-8")
    assert exp.main(["--date", exp.RUN_DATE, "--cold-replay", str(candidate)]) == 0
    assert (
        exp.main(
            [
                "--date",
                exp.RUN_DATE,
                "--independent-reduce",
                str(candidate),
                "--raw",
                str(raw),
            ]
        )
        == 0
    )
    assert "reduction" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="--date"):
        exp.main(["--date", "20260101"])

    called: list[tuple[Path, str, Path]] = []
    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda root, date, *, output_path: called.append((root, date, output_path)) or {},
    )
    assert exp.main(["--date", exp.RUN_DATE, "--output", "out.json"]) == 0
    assert called == [(exp.REPO_ROOT, exp.RUN_DATE, Path("out.json"))]


def test_defensive_numeric_and_blocked_branches(tmp_path: Path) -> None:
    """REQ-KAN-7432: malformed numeric evidence blocks without erasing boards."""

    checkpoint = _checkpoint()
    scale = exp.derive_fixed_point_scale(checkpoint, _training())
    with pytest.raises(ValueError, match="registered arms"):
        exp.replay_update_arms(checkpoint, _events(), scale, arms=("unknown",))
    bad = _events()
    bad[0]["label"] = 3
    with pytest.raises(ValueError, match="binary"):
        exp.replay_update_arms(checkpoint, bad, scale)
    with pytest.raises(ValueError, match="nonempty"):
        exp.derive_fixed_point_scale(checkpoint, np.empty((0, 6)))

    blocked = exp.build_blocked_artifact(
        exp.gate_row(
            "missing_numeric_fixture",
            "numeric_precondition",
            "==",
            "present",
            None,
            False,
            "Unavailable numeric evidence cannot promote.",
        )
    )
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked_")
    assert len(blocked["board_rows"]) == 3
    assert blocked["update_placement_complete_score"] == 0
    assert exp.validate_artifact(blocked) == []


def test_all_defensive_reader_and_validator_branches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-KAN-7432: every terminal mutation guard rejects its named defect."""

    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="unreadable_json"):
        exp.load_json(missing)
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object_required"):
        exp.load_json(scalar)
    assert exp.cold_replay(missing) == ["artifact_unreadable_or_not_object"]
    assert len(exp.build_board_rows({})) == 3

    checkpoint = _checkpoint()
    bad_training = _training()
    bad_training[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        exp.derive_fixed_point_scale(checkpoint, bad_training)
    bad_checkpoint = deepcopy(checkpoint)
    bad_checkpoint["coef"] = [[1.0]]
    with pytest.raises(ValueError, match="6x8"):
        exp.derive_fixed_point_scale(bad_checkpoint, _training())
    scale = exp.derive_fixed_point_scale(checkpoint, _training())
    malformed = _events(1)
    malformed[0]["features"] = [1.0]
    with pytest.raises(ValueError, match="six finite"):
        exp.replay_update_arms(checkpoint, malformed, scale)
    wrong_order = _events(1)
    wrong_order[0]["prediction_index"] = 2
    with pytest.raises(ValueError, match="event order"):
        exp.replay_update_arms(checkpoint, wrong_order, scale)
    with pytest.raises(ValueError, match="positive"):
        exp.measure_service_cost([], checkpoint, scale, tmp_path, blocks=0)
    with pytest.raises(ValueError, match="matching"):
        exp._bootstrap_ratio([1.0], [1.0, 2.0], draws=2, seed=1)
    exp.measure_service_cost(
        _events(1),
        checkpoint,
        scale,
        tmp_path / "progress",
        batch_sizes=(1,),
        blocks=1,
        emit_progress=True,
    )
    assert "unit_complete" in capsys.readouterr().out

    artifact = exp.build_fixture_artifact()

    def errors_after(change: object, *, checksum: bool = True) -> list[str]:
        changed = deepcopy(artifact)
        assert callable(change)
        change(changed)
        if checksum:
            changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        return exp.validate_artifact(changed)

    assert exp.validate_artifact({})[0].startswith("missing_fields")
    assert "schema_or_identity_invalid" in errors_after(
        lambda row: row.__setitem__("schema", "wrong")
    )
    assert "run_identity_invalid" in errors_after(
        lambda row: row.__setitem__("run_date", "20260101")
    )
    assert "current_model_claim_invalid" in errors_after(
        lambda row: row.__setitem__("model_invoked", True)
    )
    assert "invocation_counts_nonzero" in errors_after(
        lambda row: row["invocation_counts"].__setitem__("model_loads_attempted", 1)
    )
    assert "substrate_class_invalid" in errors_after(
        lambda row: row.__setitem__("inference_substrate_class", "cpu")
    )
    assert "execution_venue_invalid" in errors_after(
        lambda row: row.__setitem__("execution_venue", "device")
    )
    assert "board_rows_invalid" in errors_after(lambda row: row.__setitem__("board_rows", []))
    assert "board_disposition_invalid" in errors_after(
        lambda row: row["board_rows"][0].__setitem__("future_access", "usb")
    )
    assert "field_principles_incomplete" in errors_after(
        lambda row: row.__setitem__("field_principles", {})
    )
    assert "completed_verdict_prefix_invalid" in errors_after(
        lambda row: row.__setitem__("honest_verdict", "invalid")
    )
    assert "numeric_rows_invalid" in errors_after(lambda row: row.__setitem__("update_rows", None))
    assert "update_reduction_mismatch" in errors_after(
        lambda row: row.__setitem__("update_reduction", {})
    )
    assert "timing_reduction_mismatch" in errors_after(
        lambda row: row.__setitem__("timing_summary", [])
    )
    assert "acceptance_gates_mismatch" in errors_after(
        lambda row: row.__setitem__("acceptance_gate_results", [])
    )
    assert "completion_score_mismatch" in errors_after(
        lambda row: row.__setitem__("update_placement_complete_score", 0)
    )
    assert "value_score_mismatch" in errors_after(
        lambda row: row.__setitem__("update_placement_value_score", 0)
    )
    assert "combined_rows_mismatch" in errors_after(lambda row: row.__setitem__("rows", []))
    assert "external_hardware_qualification_invalid" in errors_after(
        lambda row: row["hardware_mapping"].__setitem__("external_options_qualified", True)
    )
    with pytest.raises(ValueError, match="artifact_validation_failed"):
        exp.write_artifact(tmp_path / "bad.json", {**artifact, "hardware_value_score": 1})

    blocked = exp.build_blocked_artifact(
        exp.gate_row("absent", "numeric_precondition", "==", 1, None, False, "Absent.")
    )
    blocked["honest_verdict"] = "invalid"
    blocked["update_placement_complete_score"] = 1
    blocked["reproducibility_checksum"] = exp.reproducibility_checksum(blocked)
    blocked_errors = exp.validate_artifact(blocked)
    assert "blocked_verdict_prefix_invalid" in blocked_errors
    assert "blocked_score_nonzero" in blocked_errors

    slow = deepcopy(artifact["timing_rows"])
    for row in slow:
        if row["arm"] in {"float32_sparse", "int16_fixed_sparse"}:
            row["complete_service_s"] *= 2.0
    null_artifact = exp.assemble_artifact(
        update_rows=artifact["update_rows"],
        timing_rows=slow,
        cold_rows=artifact["cold_setup_rows"],
        board_rows=artifact["board_rows"],
        scale=artifact["fixed_point_scale"],
        controls=artifact["development_controls"],
        preconditions=artifact["preconditions_checked"],
        source_hashes=artifact["source_artifact_hashes"],
        validation_receipts=artifact["validation_receipts"],
        phase_spans=artifact["phase_spans"],
        started_at_utc=artifact["started_at_utc"],
        completed_at_utc=artifact["completed_at_utc"],
        started_monotonic_ns=0,
        ended_monotonic_ns=1_000_000,
        flagged_adversarial=False,
    )
    assert null_artifact["verdict_class"] == "null"

    candidate = tmp_path / "candidate.json"
    exp.write_artifact(candidate, artifact)
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(exp.raw_view(artifact)), encoding="utf-8")
    artifact_with_receipt = deepcopy(artifact)
    artifact_with_receipt["raw_evidence_receipt"] = {"sha256": "sha256:" + "0" * 64}
    artifact_with_receipt["field_principles"]["raw_evidence_receipt"] = "Bind raw bytes."
    artifact_with_receipt["reproducibility_checksum"] = exp.reproducibility_checksum(
        artifact_with_receipt
    )
    exp.write_artifact(candidate, artifact_with_receipt)
    assert (
        exp.main(
            [
                "--date",
                exp.RUN_DATE,
                "--independent-reduce",
                str(candidate),
                "--raw",
                str(raw),
            ]
        )
        == 1
    )
