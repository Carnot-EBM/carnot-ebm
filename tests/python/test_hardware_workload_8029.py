"""REQ-REPORT-8029: custody, numerical replay and cost gates remain independent."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot import experiment_8029_v695_hardware_workload_boundary as e
from carnot.reporting import hardware_workload_8029 as h
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


def plan() -> dict:
    """SCENARIO-REPORT-8029-NUMERIC: a small control exercises the real geometry."""
    original = json.loads(
        (h.ROOT / "results/raw/experiment_8025_v695_causal_online_updates/inputs.json").read_text()
    )
    source = original["sources"][6]
    return dict(
        boards=[
            dict(board=b, custody_valid=True, historical_verdict="retained")
            for b in ("KV260", "PolarFire", "GateMate")
        ],
        checks=[],
        references=[],
        historical_failed_operands=[],
        fixture=True,
        trajectory=dict(
            head=original["head"],
            sources=[source],
            updates=[
                dict(arm="uniform", seed=101, family_id=source["family_id"], y=0, update_index=0)
            ],
        ),
        timings=None,
        scoring=None,
    )


def test_independent_branches() -> None:
    """SCENARIO-REPORT-8029-INDEPENDENT: science absence cannot erase custody."""
    p = plan()
    p["trajectory"] = None
    v = h.reduce(p)
    assert v["hardware_custody_ready_score"] == 1
    assert v["verdict_class"] == "null"
    assert v["numeric_branch_status"] == "blocked"
    assert v["quantized_update_ready_score"] == 0
    p["boards"][0]["custody_valid"] = False
    assert h.reduce(p)["hardware_custody_ready_score"] == 0
    assert h.reduce(p)["verdict_class"] == "blocked"


def test_numeric_restart_and_overflow() -> None:
    """SCENARIO-REPORT-8029-NUMERIC: all losses and restart decisions survive."""
    p = plan()
    rows = h.numeric(p["trajectory"], True)
    assert len(rows) == 1 and rows[0]["restart_agrees"]
    assert rows[0]["scale"] == 4096
    assert rows[0]["independent"] == 0
    p["trajectory"]["head"]["parameters"][0] = 1e8
    assert h.numeric(p["trajectory"], True)[0]["saturation_count"] > 0
    p["trajectory"]["updates"] = []
    with pytest.raises(ValueError, match="trajectory_budget"):
        h.numeric(p["trajectory"], True)
    v = h.reduce(plan())
    assert v["numeric_branch_status"] in {"qualified", "disqualified"}
    assert v["generalized_learning_benefit_score"] == 0


def test_workload_bounds() -> None:
    """SCENARIO-REPORT-8029-WORKLOAD: only matching operations earn a bound."""
    p = plan()
    p["timings"] = [
        dict(
            arm="cpu",
            batch=1,
            repetition=0,
            transaction_ns=1000,
            arithmetic_ns=100,
            binding_ns=50,
            serialization_ns=100,
            persistence_fsync_ns=200,
            feature_construction_ns=300,
        )
    ]
    p["scoring"] = [dict(duration_s=2, operation="teacher_forced_prefill_scoring")]
    v = h.reduce(p)
    b = v["acceleration_bounds"]["sparse_arithmetic_100x"][0]
    assert b["serial_share"] == 0.9
    assert b["speedup_bound"] == pytest.approx(1000 / 901)
    assert v["acceleration_bounds"]["hypothetical_sampling"] is None
    p["timings"][0].update(operation="ising_sampling", sampling_ns=100)
    assert h.reduce(p)["acceleration_bounds"]["hypothetical_sampling"]
    p["timings"][0]["arithmetic_ns"] = 2000
    with pytest.raises(ValueError, match="cost_partition"):
        h.reduce(p)


def test_load_contracts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8029-INDEPENDENT: zero and missing contracts fail closed."""
    history = dict(
        boards=plan()["boards"],
        checks=[],
        cited_upstream_artifacts=[],
        historical_failed_operands=[],
    )
    monkeypatch.setattr(h.history, "authenticate", lambda *a: deepcopy(history))
    p = h.load(tmp_path, tmp_path / "raw")
    assert p["trajectory"] is None and p["checks"]
    for eid, (stem, field) in h.SOURCES.items():
        path = tmp_path / "results" / (stem + ".json")
        atomic_json(
            path,
            dict(
                experiment_id=eid,
                run_date="20261002",
                verdict_class="null",
                flagged_adversarial=False,
                **{field: 0},
            ),
        )
    p = h.load(tmp_path, tmp_path / "raw")
    assert any(r["observed"] == 0 and not r["passed"] for r in p["checks"])
    path = tmp_path / "results" / (h.SOURCES[8025][0] + ".json")
    atomic_json(path, dict(experiment_id=8025))
    assert any(
        r["observed"] == "MISSING_CONTRACT_FIELD"
        for r in h.load(tmp_path, tmp_path / "raw")["checks"]
    )


def test_direct_cli_and_cold_tamper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8029-SEAL: private script publication and byte drift run."""
    source = tmp_path / "fixture.json"
    atomic_json(source, plan())
    output = tmp_path / "results" / (e.NAME + ".json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            e.OWNED[-1],
            "--fixture-input",
            str(source),
            "--validation-worker",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(e.ROOT / e.OWNED[-1]), run_name="__main__")
    assert result.value.code == 0
    assert e.main(["--cold-replay", str(output)]) == 0
    value = json.loads(output.read_text())
    assert all(
        Path(r["path"]).is_relative_to(output.parent / "raw")
        for r in value["checkpoint_references"]
    )
    value["hardware_custody_ready_score"] = 0
    atomic_json(output, value)
    assert e.main(["--cold-replay", str(output)]) == 1
    assert e.main(["--date", "20261001"]) == 1


def test_threshold_controls() -> None:
    """SCENARIO-REPORT-8029-NUMERIC: synthetic thresholds cannot earn natural credit."""
    controls = h.controls()
    assert controls["working"]
    assert all(r["independent"] == 0 and r["fixture"] for r in controls["rows"])
    assert any(r["action_disagreement"] for r in controls["rows"])


def test_qualified_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8029-INDEPENDENT: eligible current sources bind their bytes."""
    history = dict(
        boards=plan()["boards"],
        checks=[],
        cited_upstream_artifacts=[],
        historical_failed_operands=[],
    )
    monkeypatch.setattr(h.history, "authenticate", lambda *a: deepcopy(history))
    inputs = tmp_path / "inputs.json"
    atomic_json(inputs, {k: v for k, v in plan()["trajectory"].items() if k != "updates"})
    ref = dict(path=str(inputs), sha256=sha256_file(inputs))
    for eid, (stem, field) in h.SOURCES.items():
        path = tmp_path / "results" / (stem + ".json")
        atomic_json(
            path,
            dict(
                experiment_id=eid,
                run_date="20261002",
                verdict_class="null",
                flagged_adversarial=False,
                **{field: 1},
                raw_shard_hashes=[ref],
                update_rows=plan()["trajectory"]["updates"],
                checkpoint_references=[ref],
                timing_rows=[],
                validation_receipts=[dict(required=True, passed=True)],
                rows=[dict(started_monotonic_ns=1, ended_monotonic_ns=1000000001, id="score")],
            ),
        )
    atomic_json(
        tmp_path / "results/experiment_8024_likelihood_decision_test.json", dict(blocked=True)
    )
    atomic_json(
        tmp_path / "results/experiment_8016_v694_hardware_update_boundary.json",
        dict(honest_verdict="complete_blocked_original", verdict_class="blocked"),
    )
    loaded = h.load(tmp_path, tmp_path / "raw")
    assert loaded["trajectory"]["updates"]
    assert loaded["scoring"][0]["duration_s"] == 1
    assert all(Path(r["path"]).is_relative_to(tmp_path / "raw") for r in loaded["references"])
    assert loaded["preserved_historical_outcomes"][0]["verdict_class"] == "blocked"
    path = tmp_path / "results" / (h.SOURCES[8027][0] + ".json")
    value = json.loads(path.read_text())
    log = tmp_path / "failure.log"
    log.write_text("original failed assertion\n")
    value["validation_receipts"] = [
        dict(required=True, passed=False, log_path=str(log), log_sha256=sha256_file(log))
    ]
    atomic_json(path, value)
    loaded = h.load(tmp_path, tmp_path / "other-raw")
    assert loaded["timings"] is None
    assert any(
        r["artifact_field"] == "validation_receipts.required_checks_passed" and not r["passed"]
        for r in loaded["checks"]
    )


@pytest.mark.parametrize("passed", [True, False])
def test_owned_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passed: bool) -> None:
    """SCENARIO-REPORT-8029-SEAL: owned failures close readiness independently."""
    fixture = tmp_path / "fixture.json"
    atomic_json(fixture, plan())
    log = tmp_path / "owned.log"
    log.write_text("real private validation stub\n")
    monkeypatch.setattr(
        e,
        "validate",
        lambda *a: (
            [dict(required=True, passed=passed, log_path=str(log), log_sha256=sha256_file(log))],
            {p: dict(num_statements=1, missing_lines=0) for p in e.OWNED},
        ),
    )
    output = tmp_path / (e.NAME + ".json")
    assert e.main(["--fixture-input", str(fixture), "--output", str(output)]) == 0
    v = json.loads(output.read_text())
    assert v["hardware_custody_ready_score"] == int(passed)
    assert e.main(["--cold-replay", str(output)]) == 0
    checkpoint = Path(v["checkpoint_references"][0]["path"])
    atomic_json(checkpoint, dict(changed=True))
    for r in v["raw_shard_hashes"] + v["checkpoint_references"]:
        if r["path"] == str(checkpoint):
            r["sha256"] = sha256_file(checkpoint)
    atomic_json(output, v)
    assert e.main(["--cold-replay", str(output)]) == 1


def test_reader_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8029-SEAL: successful publication requires actual readers."""
    fixture = tmp_path / "fixture.json"
    atomic_json(fixture, plan())
    monkeypatch.setattr(e, "reader_receipt", lambda *a, **k: dict(passed=False))
    assert (
        e.main(
            [
                "--fixture-input",
                str(fixture),
                "--validation-worker",
                "--output",
                str(tmp_path / (e.NAME + ".json")),
            ]
        )
        == 1
    )


def test_private_validation_parent(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8029-SEAL: frozen pytest scratch has an existing parent."""
    e.commands(tmp_path)
    assert (tmp_path / "pytest").is_dir()
