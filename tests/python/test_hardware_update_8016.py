"""REQ-REPORT-8016: numerical controls cannot replace natural update custody."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import hardware_update_8016 as h
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


def test_cumulative_restart_and_overflow() -> None:
    """SCENARIO-REPORT-8016-NUMERIC: each update consumes the previous state."""
    plan = h.fixture()
    result = h.reduce(plan)
    assert result["verdict_class"] == "circular_positive"
    assert len(result["cumulative_error_rows"]) == len(plan["trajectory"]["updates"])
    assert result["quantized_update_ready_score"] == 1
    assert result["acceptance_gate_results"]["numeric"]["restart_agrees"]
    assert (
        result["cumulative_error_rows"][1]["state_before"]
        == result["cumulative_error_rows"][0]["state_after"]
    )
    assert result["sample_size_budget"]["independent"] == 0
    large = deepcopy(plan)
    large["trajectory"]["head"]["parameters"] = [10000.0] * 109
    failed = h.reduce(large)
    assert failed["quantized_update_ready_score"] == 0
    assert sum(r["saturation_count"] for r in failed["overflow_rows"]) > 0


def test_blocked_branch_and_null_costs() -> None:
    """REQ-REPORT-8016: no trajectory means no numerical measurement."""
    plan = h.fixture()
    plan.update(trajectory=None, fixture=False)
    result = h.reduce(plan)
    assert result["verdict_class"] == "blocked"
    assert result["hardware_evidence_ready_score"] == 0
    assert result["board_custody_ready_score"] == 1
    assert result["cumulative_error_rows"] == []
    assert result["acceleration_bounds"]["whole_service_ideal"] is None
    assert result["current_device_execution_count"] == 0


def authority(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Private source copies let mutation checks preserve real historical bytes."""
    value = h.fixture()
    value = dict(experiment_id=8003, board_rows=value["boards"], flagged_adversarial=False)
    for board in value["board_rows"]:
        source = root / (board["board"] + ".json")
        atomic_json(source, dict(board=board["board"], run_date="20260101"))
        board.update(source_path=str(source), source_hash=sha256_file(source))
    path = root / "results" / h.UPSTREAM[8003]
    atomic_json(path, value)
    monkeypatch.setattr(h, "PRIOR_HASH", sha256_file(path))


@pytest.mark.parametrize("mutation", ["none", "missing", "changed", "missing_primary"])
def test_independent_board_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """SCENARIO-REPORT-8016-SEAL: a lost board cannot erase the other rows."""
    root = tmp_path / "authority"
    authority(root, monkeypatch)
    if mutation == "missing":
        (root / "KV260.json").unlink()
    elif mutation == "changed":
        (root / "KV260.json").write_text("{}")
    elif mutation == "missing_primary":
        (root / "results" / h.UPSTREAM[8003]).unlink()
    plan = h.authenticate(root, tmp_path / "sealed")
    assert len(plan["boards"]) == 3
    assert plan["boards"][0]["custody_valid"] == (mutation == "none")
    assert plan["boards"][1]["custody_valid"] == (mutation != "missing_primary")
    assert plan["trajectory"] is None
    assert plan["checks"]


def test_replay_seals_and_claim_mutation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8016-SEAL: cold reduction verifies checkpoints and rows."""
    plan = h.fixture()
    path = tmp_path / "plan.json"
    atomic_json(path, plan)
    value = dict(
        h.reduce(plan),
        replay_inputs=plan,
        raw_shard_hashes=[dict(path=str(path), sha256=sha256_file(path))],
    )
    assert h.replay(value)["passed"]
    value["quantized_update_ready_score"] = 0
    with pytest.raises(ValueError, match="reduction_drift"):
        h.replay(value)
    value = dict(
        h.reduce(plan),
        replay_inputs=plan,
        raw_shard_hashes=[dict(path=str(path), sha256=sha256_file(path))],
    )
    path.write_text("{}")
    with pytest.raises(ValueError, match="receipt_drift"):
        h.replay(value)


def test_qualified_contract_and_zero_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8016: missing fields differ from failed zero gates."""
    authority(tmp_path, monkeypatch)
    path = tmp_path / "results" / h.UPSTREAM[8012]
    atomic_json(
        path,
        dict(experiment_id=8012, learning_measurement_ready_score=0, flagged_adversarial=False),
    )
    plan = h.authenticate(tmp_path, tmp_path / "sealed")
    assert any(
        r["artifact_field"] == "learning_measurement_ready_score" and r["observed"] == 0
        for r in plan["checks"]
    )
    atomic_json(path, dict(experiment_id=8012))
    with pytest.raises(ValueError, match="upstream_contract"):
        h.authenticate(tmp_path, tmp_path / "sealed2")


def test_qualified_updates_and_costs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8016-NUMERIC: qualified bytes drive an exact ordered replay."""
    authority(tmp_path, monkeypatch)
    trajectory = tmp_path / "trajectory.json"
    atomic_json(trajectory, h.fixture()["trajectory"])
    atomic_json(
        tmp_path / "results" / h.UPSTREAM[8012],
        dict(
            experiment_id=8012,
            learning_measurement_ready_score=1,
            flagged_adversarial=False,
            checkpoints=dict(trajectory=dict(path=str(trajectory), sha256=sha256_file(trajectory))),
        ),
    )
    atomic_json(
        tmp_path / "results" / h.UPSTREAM[8015],
        dict(
            experiment_id=8015,
            parity_ready_score=1,
            flagged_adversarial=False,
            native_timing=dict(kernel_s=1.0, host_s=2.0, ffi_s=3.0, storage_s=4.0),
        ),
    )
    plan = h.authenticate(tmp_path, tmp_path / "sealed")
    assert plan["trajectory"] == json.loads(trajectory.read_bytes())
    result = h.reduce(plan)
    assert result["verdict_class"] == "null"
    assert result["acceleration_bounds"]["whole_service_ideal"] == 1.0
    assert result["acceleration_bounds"]["hypothetical_kernel_ideal"] == pytest.approx(10 / 9)
    assert "historical_model_load_s" in result["missing_cost_components"]
    plan["trajectory"]["updates"] = []
    with pytest.raises(ValueError, match="trajectory_budget"):
        h.reduce(plan)


def test_validation_receipt_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8016-SEAL: failure logs and disqualification stay traceable."""
    value = dict(h.reduce(h.fixture()), replay_inputs=h.fixture())
    log = tmp_path / "log.txt"
    log.write_text("original")
    value["validation_receipts"] = [dict(log_path=str(log), log_sha256=sha256_file(log))]
    assert h.replay(value)["passed"]
    log.write_text("changed")
    with pytest.raises(ValueError, match="receipt_drift:validation"):
        h.replay(value)
    value["validation_receipts"] = [dict(required=True, passed=False)]
    value.update(
        verdict_class="disqualified",
        honest_verdict="complete_disqualified_owned_checks",
        hardware_evidence_ready_score=0,
    )
    assert h.replay(value)["passed"]


def test_retained_actual_board_sources(tmp_path: Path) -> None:
    """REQ-REPORT-8016: real history is read without issuing device operations."""
    plan = h.authenticate(h.ROOT, tmp_path / "sealed")
    assert all(b["custody_valid"] for b in plan["boards"])
    assert plan["boards"][1]["last_actual_execution_date"] == "20260912"
    assert plan["boards"][0]["last_actual_execution_hash"]
    assert plan["boards"][2]["blocker"] == "0xffffffff"
    assert any(
        r["artifact_field"] == "conditioned_fit_ready_score" and r["observed"] == 0
        for r in plan["checks"]
    )


def test_bound_raw_input_mutation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8016-SEAL: summaries cannot substitute another raw input."""
    path = tmp_path / "replay_inputs.json"
    atomic_json(path, dict(h.fixture(), fixture=False))
    value = dict(
        h.reduce(h.fixture()),
        replay_inputs=h.fixture(),
        raw_shard_hashes=[dict(path=str(path), sha256=sha256_file(path))],
    )
    with pytest.raises(ValueError, match="raw_reduction_drift"):
        h.replay(value)


def test_near_threshold_stress_is_circular() -> None:
    """SCENARIO-REPORT-8016-NUMERIC: action changes near gates are never omitted."""
    import math

    plan = h.fixture()
    plan["trajectory"]["head"]["parameters"] = [0.0] * 108 + [2 * math.log(3)]
    plan["trajectory"]["updates"] = [
        dict(
            id="threshold-stress",
            source_cluster_id="synthetic",
            x=[0.5] * 9,
            y=1,
            learning_rate=0.01,
        )
    ]
    value = h.reduce(plan)
    assert abs(value["cumulative_error_rows"][0]["float_probability_before"] - 0.75) < 1e-12
    assert value["cumulative_error_rows"][0]["fixture"]
    assert value["sample_size_budget"]["independent"] == 0
    assert "action_disagreements" in value["cumulative_error_rows"][0]
