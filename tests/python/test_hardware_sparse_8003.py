"""REQ-REPORT-8003, REQ-VERIFY-8003, REQ-SELF-8003: bounded CPU evidence."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot.reporting import hardware_sparse_8003 as h
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.verify import fixedpoint_sparse_8003 as f


def authority(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Private producer bytes permit custody mutations without changing history."""
    plan = h.fixture()
    pins = {}
    for eid, field in (
        (7977, "hardware_evidence_ready_score"),
        (7996, "sparse_fit_ready_score"),
        (8002, "service_measurement_ready_score"),
        (7990, "hardware_evidence_ready_score"),
    ):
        value = dict(
            experiment_id=eid,
            execution_date="20261001",
            run_date="20261001",
            verdict_class="null",
            flagged_adversarial=False,
            validation_receipts=[dict(required=True, passed=True)],
            **{field: 1},
        )
        terminal = root / f"terminal-{eid}.json"
        value["terminal_validation_sidecar_path"] = str(terminal)
        if eid == 7977:
            value["board_rows"] = deepcopy(plan["boards"])
            for board in value["board_rows"]:
                path = root / (board["board"] + ".json")
                atomic_json(path, dict(board=board["board"]))
                board.update(source_path=str(path), source_hash=sha256_file(path))
            value["preconditions_checked"] = []
            value["historical_required_failures"] = [dict(passed=False, name="historical")]
        if eid == 7996:
            value["checkpoints"] = {}
            for key, content in (
                ("heads", dict(heads=dict(spline=[plan["head"]]))),
                ("inputs", dict(data=plan["data"])),
            ):
                path = root / (key + ".json")
                atomic_json(path, content)
                value["checkpoints"][key] = dict(path=str(path), sha256=sha256_file(path))
        if eid == 8002:
            value.update(
                rows=plan["service_rows"],
                measurement_code_snapshot=[],
                code_config_hashes=[],
                raw_shard_hashes=[],
            )
        if eid == 7990:
            value.update(gate_check_summary=[dict(passed=False, observed="stale producer")])
        path = root / "results" / h.NAMES[eid]
        atomic_json(path, value)
        atomic_json(terminal, dict(passed=True, candidate_sha256=sha256_file(path)))
        pins[eid] = dict(primary=sha256_file(path), terminal=sha256_file(terminal))
    monkeypatch.setattr(h, "PINS", pins)
    return root


def test_formats_and_rounding() -> None:
    """SCENARIO-VERIFY-8003-BOUNDARY: integer arithmetic has explicit limits."""
    for bits in (16, 24):
        q = f.Fixed(bits)
        assert q.encode(0.5 / 4096) == 0
        assert q.encode(1.5 / 4096) == 2
        assert q.encode(-1.5 / 4096) == -2
        assert q.encode(1e8) == (1 << (bits - 1)) - 1
        assert q.encode(-1e8) == -(1 << (bits - 1))
        assert q.saturations == 2
        assert q.mul(4096, 2048) == 2048
        assert q.add(4096, -2048) == 2048


def test_actual_sparse_numerics() -> None:
    """REQ-VERIFY-8003: compare all values and count source groups once."""
    plan = h.fixture()
    result = f.evaluate(plan["head"], plan["data"])
    assert result["fixed_point_rows"]
    assert {r["bits"] for r in result["fixed_point_rows"]} == {16, 24}
    assert result["quantization_ready_score"] == 1
    assert all(
        len(r["float_gradient"]) == len(r["fixed_gradient"]) == 109
        for r in result["fixed_point_rows"]
    )
    assert all(r["coefficient_touches"] <= 37 for r in result["fixed_point_rows"])
    assert result["positive_control_results"]["working"]
    large = deepcopy(plan["head"])
    large["parameters"] = [10000.0] * 109
    failed = f.evaluate(large, plan["data"])
    assert failed["quantization_ready_score"] == 0
    assert failed["quantization_gates"]["16"]["unexpected_saturation_count"] > 0
    assert len(result["lookup_grids"]["basis"]) > 4000


def test_missing_probability_exclusion() -> None:
    """REQ-VERIFY-8003: excluded acquisitions cannot become lookup operands."""
    plan = h.fixture()
    plan["data"]["fit"].append(
        dict(
            q=None,
            features=None,
            y=0,
            source_cluster_id="missing",
            family_id="missing",
            status="failed",
        )
    )
    result = f.evaluate(plan["head"], plan["data"])
    assert result["independent_source_groups"] == 12
    assert all(r["family_id"] != "missing" for r in result["fixed_point_rows"])


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "missing_service",
        "stale_service",
        "missing_board",
        "missing_sparse",
        "missing_prior",
        "missing_terminal",
    ],
)
def test_independent_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """SCENARIO-REPORT-8003-CUSTODY: optional service cannot erase custody."""
    root = authority(tmp_path / "sources", monkeypatch)
    if mutation == "missing_service":
        (root / "results" / h.NAMES[8002]).unlink()
    elif mutation == "stale_service":
        (root / "results" / h.NAMES[8002]).write_text("{}")
    elif mutation == "missing_board":
        (root / "KV260.json").unlink()
    elif mutation == "missing_sparse":
        (root / "inputs.json").unlink()
    elif mutation == "missing_prior":
        (root / "results" / h.NAMES[7977]).unlink()
    elif mutation == "missing_terminal":
        (root / "terminal-7977.json").unlink()
    plan = h.authenticate(root, tmp_path / "sealed")
    assert len(plan["boards"]) == 3
    assert plan["board_custody_ready_score"] == int(
        mutation not in {"missing_board", "missing_prior", "missing_terminal"}
    )
    assert plan["service_available"] == (mutation not in {"missing_service", "stale_service"})
    value = h.reduce(plan)
    if "service" in mutation:
        assert value["ideal_amdahl_bound"] is None
        assert plan["optional_service_blockers"]
    if mutation in {"missing_board", "missing_sparse", "missing_prior", "missing_terminal"}:
        assert value["verdict_class"] == "blocked"
        assert value["gate_check_summary"]
    assert plan["historical_failed_operands"]
    assert all(
        {
            "upstream_id",
            "artifact_path",
            "artifact_hash",
            "artifact_field",
            "op",
            "expected",
            "observed",
        }
        <= set(r)
        for r in plan["checks"]
    )


def test_reconstruction_rejects_changes(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8003-REPLAY: raw reduction catches edited claims."""
    plan = h.fixture()
    value = h.reduce(plan)
    value["replay_inputs"] = plan
    assert h.replay(value)["passed"]
    value["modeled_100x_bound"] = 999
    with pytest.raises(ValueError, match="reduction_drift"):
        h.replay(value)
    value = h.reduce(plan)
    value["replay_inputs"] = plan
    path = tmp_path / "receipt.log"
    path.write_text("original")
    value["validation_receipts"] = [dict(log_path=str(path), log_sha256=sha256_file(path))]
    path.write_text("mutated")
    with pytest.raises(ValueError, match="receipt_drift"):
        h.replay(value)


def test_missing_contract_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8003: absent fields are contract errors, not zeros."""
    root = authority(tmp_path / "source", monkeypatch)
    path = root / "results" / h.NAMES[7977]
    value = json.loads(path.read_bytes())
    del value["hardware_evidence_ready_score"]
    atomic_json(path, value)
    h.PINS[7977]["primary"] = sha256_file(path)
    with pytest.raises(ValueError, match="upstream_contract"):
        h.authenticate(root, tmp_path / "sealed")


def test_replay_receipts_and_disqualification(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8003-REPLAY: missing bytes and unproven failures close gates."""
    plan = h.fixture()
    value = h.reduce(plan)
    value["replay_inputs"] = plan
    value["code_config_hashes"] = [dict(path=str(tmp_path / "absent"), sha256="sha256:absent")]
    with pytest.raises(ValueError, match="receipt_drift"):
        h.replay(value)
    value["code_config_hashes"] = []
    value.update(
        verdict_class="disqualified", hardware_evidence_ready_score=0, validation_receipts=[]
    )
    with pytest.raises(ValueError, match="owned_failure"):
        h.replay(value)
    value["validation_receipts"] = [dict(required=True, passed=False)]
    assert h.replay(value)["passed"]


def test_role_budget_and_fit_only_grid() -> None:
    """REQ-VERIFY-8003: the frozen panel preserves tune headroom within 128 groups."""
    plan = h.fixture()
    prototype = plan["data"]["fit"][0]
    for role in ("fit", "tune"):
        plan["data"][role] = [
            dict(prototype, family_id=f"{role}-{i}", source_cluster_id=f"{role}-{i}")
            for i in range(140)
        ]
    expected = f.grids(plan["head"], plan["data"]["fit"])
    result = f.evaluate(plan["head"], plan["data"])
    assert result["independent_source_groups"] == 128
    assert sum(r["role"] == "tune" and r["bits"] == 24 for r in result["fixed_point_rows"]) == 32
    assert result["lookup_grids"] == expected
    assert all(
        r["numerator"] >= abs(r["fixed_updated_probability"] - r["float_updated_probability"])
        for r in result["fixed_point_rows"]
    )


def test_sealed_replay_input(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8003-REPLAY: embedded inputs must match their sealed bytes."""
    plan = h.fixture()
    path = tmp_path / "inputs.json"
    atomic_json(path, plan)
    value = dict(
        h.reduce(plan),
        replay_inputs=plan,
        replay_input_reference=dict(path=str(path), sha256=sha256_file(path)),
    )
    assert h.replay(value)["passed"]
    value["replay_inputs"] = dict(plan, fixture=False)
    with pytest.raises(ValueError, match="input_drift"):
        h.replay(value)
