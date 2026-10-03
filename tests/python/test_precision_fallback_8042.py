"""REQ-REPORT-8042: test bounds, custody, costs and real private CLI paths."""

from copy import deepcopy
import ast
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from carnot import experiment_8042_v696_precision_fallback_boundary as e
from carnot.reporting import precision_fallback_8042 as h
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


def plan(updates: int = 2) -> dict:
    """Synthetic controls use the production recurrence without natural sample credit."""
    head = dict(parameters=[0.0, 0.25], decay_scale=1.0, calibration=[0.0, 1.0])
    events = [
        dict(
            kind="gradient",
            arm="recent64",
            seed=101,
            family_id="fixture",
            x=[1.0, 0.0],
            y=i % 2,
            update_index=i,
            source_cluster_id="fixture",
            expected_head_hash=None,
        )
        for i in range(updates)
    ]
    return dict(
        boards=[dict(board=b, custody_valid=True) for b in ("KV260", "PolarFire", "GateMate")],
        checks=[],
        references=[],
        historical_failed_operands=[],
        historical_outcomes=[],
        fixture=True,
        trajectory=dict(head=head, events=events),
        transactions=None,
    )


def test_intervals_and_long_recurrence() -> None:
    """SCENARIO-REPORT-8042-BOUND: bounds survive repeated sparse updates and restart."""
    v = h.reduce(plan(256))
    assert v["fallback_parity_ready_score"] == 1
    assert v["acceptance_gate_results"]["numeric"]["post_fallback_action_disagreements"] == 0
    assert all(r["contained"] and r["restart_agrees"] for r in v["interval_containment_rows"])
    assert v["fallback_rows"] and v["sample_size_budget"]["independent"] == 0
    assert v["positive_control_results"]["working"]
    for left, right in (([-2.0, 3.0], [-4.0, 1.0]), ([1.0, 2.0], [3.0, 4.0])):
        got = h.multiply(left, right)
        assert got[0] <= min(a * b for a in left for b in right)
        assert got[1] >= max(a * b for a in left for b in right)
    assert h.sigmoid([-100.0, 100.0]) == [0.0, 1.0]
    with pytest.raises(ValueError, match="decay_interval"):
        h.divide([1.0, 1.0], [-1.0, 1.0])


@pytest.mark.parametrize("threshold", [0.1, 0.5])
def test_threshold_and_saturation(threshold: float) -> None:
    """SCENARIO-REPORT-8042-BOUND: exact ties and overflow always use float64."""
    p = plan(1)
    p["trajectory"]["head"]["parameters"][0] = float(np.log(threshold / (1 - threshold)))
    v = h.reduce(p)
    assert v["fallback_rows"][0]["used_float64"]
    p["trajectory"]["head"]["parameters"][0] = 1e8
    v = h.reduce(p)
    assert v["acceptance_gate_results"]["numeric"]["handled_overflow"] > 0
    assert v["acceptance_gate_results"]["numeric"]["unhandled_overflow"] == 0
    assert v["fallback_parity_ready_score"] == 1


def test_independent_block_and_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8042-BOUND: external absence and budget cannot erase custody."""
    p = plan()
    p["trajectory"] = None
    v = h.reduce(p)
    assert v["hardware_custody_ready_score"] == 1 and v["numeric_branch_status"] == "blocked"
    assert v["gate_check_summary"][-1]["observed"] == "MISSING_QUALIFIED_TRACE"
    p["boards"][0]["custody_valid"] = False
    assert h.reduce(p)["verdict_class"] == "blocked"
    monkeypatch.setitem(h.CONFIG, "numerical_budget_s", -1)
    assert h.reduce(plan())["numeric_branch_status"] == "blocked"
    monkeypatch.setitem(h.CONFIG, "numerical_budget_s", 600)
    p = plan(0)
    with pytest.raises(ValueError, match="trajectory_budget"):
        h.numeric(p["trajectory"], True)


def test_costs_and_source_contracts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8042-COST: costs require matching byte-bound current receipts."""
    p = plan()
    p["transactions"] = [
        dict(
            arm="native",
            condition="recent64",
            transaction_ns=1000,
            arithmetic_ns=100,
            denominator=4,
            excluded=False,
            repetition=0,
        )
    ]
    v = h.reduce(p)
    bound = v["acceleration_bounds"]["hypothetical_compatible_arithmetic_100x"][0]
    assert 1 <= bound["speedup_bound"] <= 1000 / 901
    assert bound["fallback_transfer_serial_ns"] >= 0
    p["transactions"].append(dict(excluded=True))
    assert len(h.reduce(p)["acceleration_bounds"]["hypothetical_compatible_arithmetic_100x"]) == 1
    p["transactions"][0]["arithmetic_ns"] = 2000
    with pytest.raises(ValueError, match="cost_partition"):
        h.reduce(p)
    monkeypatch.setattr(
        h.history,
        "authenticate",
        lambda *a: dict(
            boards=plan()["boards"],
            checks=[],
            cited_upstream_artifacts=[],
            historical_failed_operands=[],
        ),
    )
    p = h.load(tmp_path, tmp_path / "raw")
    assert p["trajectory"] is None and p["transactions"] is None
    assert any(r["observed"] == "MISSING_CONTRACT_FIELD" for r in p["checks"])


def test_qualified_source_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8042-SEAL: final bindings, logs and journal rows authenticate."""
    monkeypatch.setattr(
        h.history,
        "authenticate",
        lambda *a: dict(
            boards=plan()["boards"],
            checks=[],
            cited_upstream_artifacts=[],
            historical_failed_operands=[],
        ),
    )
    from carnot.verify import windowed_online_8038 as window
    from test_causal_online_8025 import fixture

    data = fixture(64)
    data["sources"][5]["public_eligible"] = False
    trajectory = tmp_path / "trajectory"
    measured = window.measure(data, trajectory)
    for eid, (stem, ready) in h.SOURCES.items():
        path = tmp_path / "results" / (stem + ".json")
        terminal = tmp_path / f"terminal{eid}.json"
        sidecar = tmp_path / f"sidecar{eid}.json"
        log = tmp_path / f"check{eid}.log"
        log.write_text("passed original check\n")
        v = dict(
            experiment_id=eid,
            task_id=f"exp{eid}-fixture",
            run_date="20261002",
            verdict_class="null",
            flagged_adversarial=False,
            terminal_validation_sidecar_path=str(terminal),
            validation_receipts=[
                dict(
                    required=True,
                    passed=True,
                    exit_code=0,
                    log_path=str(log),
                    log_sha256=sha256_file(log),
                )
            ],
            **{ready: 1},
            trajectory_directory=str(trajectory),
            transaction_rows=[dict(checkpoint=h.reference(trajectory / "inputs.json"))],
            **measured,
        )
        atomic_json(path, v)
        binding = dict(
            primary_path=str(path), primary_sha256=sha256_file(path), sidecar_path=str(sidecar)
        )
        atomic_json(sidecar, dict(binding, report=dict(passed=True)))
        atomic_json(terminal, dict(publication=binding))
    old = tmp_path / "results/experiment_8029_v695_hardware_workload_boundary.json"
    atomic_json(
        old,
        dict(
            honest_verdict="complete_null_original",
            acceptance_gate_results=dict(numeric=dict(action_disagreements=1)),
        ),
    )
    loaded = h.load(tmp_path, tmp_path / "raw")
    assert loaded["trajectory"]["events"] and loaded["historical_outcomes"]
    assert all(Path(r["path"]).is_relative_to(tmp_path / "raw") for r in loaded["references"])
    assert loaded["trajectory"]["excluded_rows"]
    (tmp_path / "sidecar8038.json").unlink()
    assert h.load(tmp_path, tmp_path / "broken-sidecar")["trajectory"] is None
    path = tmp_path / "results" / (h.SOURCES[8038][0] + ".json")
    atomic_json(path, dict(experiment_id=8038))
    assert h.load(tmp_path, tmp_path / "other")["trajectory"] is None


def test_real_cli_and_reduction_tamper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8042-SEAL: runnable script publishes and cold rejects altered claims."""
    fixture = tmp_path / "fixture.json"
    atomic_json(fixture, plan())
    output = tmp_path / "results" / (e.NAME + ".json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            e.OWNED[-1],
            "--fixture-input",
            str(fixture),
            "--validation-worker",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(e.ROOT / e.OWNED[-1]), run_name="__main__")
    assert result.value.code == 0
    assert e.main(["--cold-replay", str(output)]) == 0
    original = json.loads(output.read_bytes())
    for field in ("fallback_parity_ready_score", "interval_containment_rows"):
        v = deepcopy(original)
        v[field] = 0 if field.endswith("score") else []
        atomic_json(output, v)
        assert e.main(["--cold-replay", str(output)]) == 1
    assert e.main(["--date", "20261001"]) == 1
    checkpoint = Path(original["checkpoint_references"][0]["path"])
    atomic_json(checkpoint, dict(changed=True))
    for ref in original["checkpoint_references"] + original["raw_shard_hashes"]:
        if ref["path"] == str(checkpoint):
            ref["sha256"] = sha256_file(checkpoint)
    atomic_json(output, original)
    assert e.main(["--cold-replay", str(output)]) == 1


@pytest.mark.parametrize("passed", [False, True])
def test_validation_and_reader_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passed: bool
) -> None:
    """SCENARIO-REPORT-8042-SEAL: owned failures close readiness; reader failures exit one."""
    fixture = tmp_path / "fixture.json"
    atomic_json(fixture, plan())
    log = tmp_path / "check.log"
    log.write_text("private owned check\n")
    monkeypatch.setattr(
        e,
        "validate",
        lambda *a: (
            [dict(required=True, passed=passed, log_path=str(log), log_sha256=sha256_file(log))],
            {p: dict(num_statements=1, missing_lines=0) for p in [*e.OWNED, e.GUARD]},
        ),
    )
    output = tmp_path / "results" / (e.NAME + ".json")
    assert e.main(["--fixture-input", str(fixture), "--output", str(output)]) == 0
    v = json.loads(output.read_bytes())
    assert v["hardware_custody_ready_score"] == int(passed)
    assert e.main(["--cold-replay", str(output)]) == 0
    monkeypatch.setattr(e, "reader_receipt", lambda *a, **k: dict(passed=False))
    assert (
        e.main(["--fixture-input", str(fixture), "--validation-worker", "--output", str(output)])
        == 1
    )


def test_numeric_drift_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8042-BOUND: mismatched current state and missed containment are failures."""
    p = plan()
    p["trajectory"]["events"][0]["expected_head_hash"] = "wrong"
    with pytest.raises(ValueError, match="trajectory_state_drift"):
        h.numeric(p["trajectory"], True)
    monkeypatch.setattr(h, "sigmoid", lambda *a: [0.0, 0.0])
    assert h.reduce(plan())["fallback_parity_ready_score"] == 0


@pytest.mark.parametrize("passed", [True, False])
def test_changed_guard_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passed: bool
) -> None:
    """SCENARIO-REPORT-8042-COUNTS: measure the modified statement without old-code coverage demands."""
    tree = ast.parse((e.ROOT / e.GUARD).read_text())
    line = next(
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.If) and "type(d[k1]) is int" in ast.unparse(n.test)
    )
    atomic_json(
        tmp_path / "guard-coverage.json",
        dict(files={e.GUARD: dict(executed_lines=[line] if passed else [])}),
    )
    monkeypatch.setattr(e, "prior_validate", lambda *a: ([], {}))
    _, counts = e.validate([], tmp_path, tmp_path)
    assert (tmp_path / "guard-coverage.json").is_file()
    assert counts[e.GUARD]["num_statements"] == 1
    assert counts[e.GUARD]["missing_lines"] == int(not passed)
