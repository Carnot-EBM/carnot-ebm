"""REQ-REPORT-8693, REQ-VERIFY-8693: independent mapping and immutable replay."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import hardware_operation_boundary_8693 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary


def cli(tmp_path, *args):
    """Use the actual standalone entrypoint so import and rejection paths count."""
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )


def cost_producer(root, index=2):
    """Construct complete disjoint primitives through the existing atomic publisher."""
    number, version, suffix, field, boundary = e.PRODUCERS[index]
    output = root / f"results/experiment_{number}_{version}_{suffix}.json"
    primitive = output.parent / "raw" / output.stem / "cost.json"
    atomic_json(
        primitive,
        dict(
            boundary=boundary,
            cohort="fixture",
            arm="cpu",
            exit_code=0,
            start_ns=0,
            end_ns=100,
            spans=[dict(operation="binding_residual", start_ns=0, end_ns=100)],
        ),
    )
    value = dict(
        experiment_id=number,
        task_id=f"exp{number}-{suffix.replace('_', '-')}",
        honest_verdict="complete_null_fixture",
        verdict_class="null",
        required_checks_passed=True,
        flagged_adversarial=False,
        adversarial_findings=[],
        acceptance_gates={"owned": True},
        operation_cost_closure=[e.reference(primitive)],
        **{field: 1},
    )
    publish_primary(output, value, lambda p: dict(passed=True))
    return output, primitive, value


def test_independent_cost_admission_and_rejections(tmp_path):
    """REQ-REPORT-8693: each branch has its own gate and original operands."""
    path, primitive, value = cost_producer(tmp_path)
    with patch.object(e, "producer_authentication", return_value=[]):
        branch = e.admit(tmp_path, e.PRODUCERS[2])
        assert branch["qualified"] and branch["cells"][0]["conditional_ceiling"] is None
        assert not e.admit(tmp_path, e.PRODUCERS[0])["qualified"]
        for changed, state in [(0, "observed_zero"), (None, "observed")]:
            publish_primary(
                path, {**value, "batch_cost_ready_score": changed}, lambda p: dict(passed=True)
            )
            assert e.admit(tmp_path, e.PRODUCERS[2])["checks"][0]["observed_state"] == state
        del value["batch_cost_ready_score"]
        publish_primary(path, value, lambda p: dict(passed=True))
        assert e.admit(tmp_path, e.PRODUCERS[2])["checks"][0]["observed_state"] == "missing_field"
    value["batch_cost_ready_score"] = 1
    publish_primary(path, value, lambda p: dict(passed=True))
    assert not e.admit(tmp_path, e.PRODUCERS[2])["qualified"]
    primitive.write_text("{}")
    assert not e.admit(tmp_path, e.PRODUCERS[2])["qualified"]


def test_real_precision_and_board_map(tmp_path):
    """SCENARIO-REPORT-8693-MAPPING: precision survives missing cost siblings."""
    work = e.measure(e.ROOT, tmp_path / "raw", tmp_path, "20261101")
    before = deepcopy(work)
    value = e.build(work, [dict(passed=True)], tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    assert work == before
    assert value["source_admission_rows"][1]["qualified"]
    assert len(value["integer_precision_rows"]) == 15
    assert value["hardware_mapping_ready_score"] == 1
    assert value["physical_hardware_ready_score"] == 0
    assert {r["board"] for r in value["per_board_results"]} == {"KV260", "PolarFire", "GateMate"}
    assert value["verdict_class"] == "blocked"
    assert value["device_commands_executed"] == []
    assert all(r["conditional_ceiling"] is None for r in value["conditional_acceleration_bounds"])
    assert value["preserved_quarantines"]["Exp8667"]["flagged_adversarial"]
    assert not value["acquisition_priority_delta"]["changed"]


def test_private_cli_and_cold_replay(tmp_path):
    """SCENARIO-VERIFY-8693-REPLAY: real child exits and local tampering are visible."""
    for failed in [False, True]:
        output = tmp_path / str(failed) / (e.NAME + ".json")
        result = cli(
            tmp_path,
            "--private-control",
            "--root",
            tmp_path / "absent",
            "--output",
            output,
            *(["--owned-failure"] if failed else []),
        )
        assert result.returncode == 0, result.stdout + result.stderr
        value = json.loads(output.read_bytes())
        assert value["verdict_class"] == ("disqualified" if failed else "blocked")
        assert cli(tmp_path, "--cold-replay", output).returncode == 0
        work_path = Path(value["work_reference"]["path"])
        work = json.loads(work_path.read_bytes())
        work["evidence"]["branches"][0]["qualified"] = True
        atomic_json(work_path, work)
        atomic_json(output, e.build(work, value["validation_receipts"], work_path.parent, output))
        assert cli(tmp_path, "--cold-replay", output).returncode == 1
    for date in ["bad", "20260230"]:
        assert cli(tmp_path, "--date", date).returncode == 2
    assert cli(tmp_path, "--private-control").returncode == 2
    assert cli(tmp_path, "--cold-replay", tmp_path / "missing").returncode == 1


def test_runner_bindings_and_rejected_terminal(tmp_path, monkeypatch):
    """REQ-VERIFY-8693: consumer coverage cannot certify newly owned statements."""
    from carnot.reporting import hardware_boundary_runner_8693 as runner
    from carnot.reporting import hardware_evidence_validation_8665 as validation

    def inspect(raw):
        assert validation.e is e and validation.SUITES[0] == e.TEST
        return [dict(name="owned", passed=True)]

    monkeypatch.setattr(validation, "validate", inspect)
    assert runner.validate(tmp_path)[0]["passed"]
    assert (
        runner.run(
            tmp_path, tmp_path / "ok" / (e.NAME + ".json"), tmp_path, "20261101", False, False
        )
        == 0
    )
    monkeypatch.setattr(runner, "terminal", lambda *a: dict(passed=False))
    with pytest.raises(ValueError, match="candidate_rejected"):
        runner.run(
            tmp_path, tmp_path / "bad" / (e.NAME + ".json"), tmp_path, "20261101", False, False
        )


def test_precision_gate_states_and_terminal_rejections(tmp_path, monkeypatch):
    """REQ-REPORT-8693: flagged producers cannot grant numerical readiness."""
    path = tmp_path / "results/experiment_8692_v745_spline_precision_feasibility.json"
    source = json.loads((e.ROOT / "results" / path.name).read_bytes())
    original = {
        k: source[k]
        for k in [
            "experiment_id",
            "task_id",
            "honest_verdict",
            "verdict_class",
            "required_checks_passed",
            "flagged_adversarial",
            "adversarial_findings",
            "acceptance_gates",
            "precision_mapping_ready_score",
        ]
    }
    for mutation, state in [
        ({"precision_mapping_ready_score": 0}, "observed_zero"),
        ({"precision_mapping_ready_score": None}, "observed"),
        ({"flagged_adversarial": True}, "observed"),
        ({"acceptance_gates": {"owned": False}}, "observed"),
    ]:
        publish_primary(path, {**original, **mutation}, lambda p: dict(passed=True))
        branch = e.admit(tmp_path, e.PRODUCERS[1])
        assert not branch["qualified"] and branch["checks"][0]["observed_state"] == state
    missing = dict(original)
    del missing["precision_mapping_ready_score"]
    publish_primary(path, missing, lambda p: dict(passed=True))
    assert e.admit(tmp_path, e.PRODUCERS[1])["checks"][0]["observed_state"] == "missing_field"
    publish_primary(path, original, lambda p: dict(passed=True))
    monkeypatch.setattr(e, "reader_receipt", lambda *a, **kw: dict(passed=False))
    assert e.admit(tmp_path, e.PRODUCERS[1])["missing_reason"] == "terminal_reader_identity"


def test_original_producer_and_checkpoint_rejections(tmp_path, monkeypatch):
    """REQ-VERIFY-8693: rehashed reports must match the producer's own construction."""
    path = e.ROOT / "results/experiment_8692_v745_spline_precision_feasibility.json"
    value = json.loads(path.read_bytes())
    changed = {**value, "rows": []}
    with pytest.raises(ValueError, match="producer_construction"):
        e.producer_authentication(changed, path)
    changed = {
        **value,
        "primitive_reference": {**value["primitive_reference"], "substituted": True},
    }
    with pytest.raises(ValueError, match="independent_seals"):
        e.producer_authentication(changed, path)
    monkeypatch.setattr(e.precision, "authenticate", lambda: dict(failures=["blocked"]))
    with pytest.raises(ValueError, match="checkpoint_authentication"):
        e.producer_authentication(value, path)
    path, _, fixture = cost_producer(tmp_path)
    fixture.update(code_config_hashes=[], source_artifact_hashes=[])
    with pytest.raises(ValueError, match="declared_producer_source"):
        e.producer_authentication(fixture, path)
    producer_cli = e.ROOT / "scripts/experiments" / (path.stem + ".py")
    fixture["code_config_hashes"] = [dict(path=str(producer_cli), sha256="fixture")]
    monkeypatch.setattr(e, "checked", lambda ref: Path(ref["path"]))
    monkeypatch.setattr(e, "reference", lambda path: dict(path=str(path), sha256="fixture"))
    for code in [0, 7]:
        monkeypatch.setattr(
            e.subprocess, "run", lambda *a, **kw: type("R", (), {"returncode": code})()
        )
        if code:
            with pytest.raises(ValueError, match="cost_producer_replay"):
                e.producer_authentication(fixture, path)
        else:
            assert e.producer_authentication(fixture, path) == fixture["code_config_hashes"]


def test_seals_missing_operands_and_rehashed_original_costs(tmp_path):
    """SCENARIO-VERIFY-8693-REPLAY: a substituted local seal cannot invent costs."""
    cost_producer(tmp_path)
    raw = tmp_path / "raw"
    with patch.object(e, "producer_authentication", return_value=[]):
        work = e.measure(tmp_path, raw, tmp_path, "20261101")
        output = tmp_path / (e.NAME + ".json")
        receipts = [dict(passed=True)]
        atomic_json(output, e.build(work, receipts, raw, output))
        assert e.replay(output)
        wall = work["started_at_utc"]
        work["started_at_utc"] = "2000-01-01T00:00:00+00:00"
        atomic_json(raw / "final_work.json", work)
        work["work_path"] = str(raw / "final_work.json")
        atomic_json(output, e.build(work, receipts, raw, output))
        assert not e.replay(output)
        work["started_at_utc"] = wall
        del work["work_path"]
        work["evidence"]["branches"][2]["cells"][0]["total_s"] *= 2
        atomic_json(raw / "measurement.json", work)
        atomic_json(raw / "measurement.seal.json", e.reference(raw / "measurement.json"))
        atomic_json(output, e.build(work, receipts, raw, output))
        assert not e.replay(output)
        (raw / "measurement.seal.json").unlink()
        assert not e.replay(output)
