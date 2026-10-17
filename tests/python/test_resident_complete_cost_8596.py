"""REQ-REPORT-8596 / REQ-VERIFY-8596: fresh costs require original head authority."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import resident_complete_cost_8596 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


def test_original_gates_and_descriptor(tmp_path):
    """SCENARIO-REPORT-8596-FRESH: exact original seals precede any clocks."""
    bundle = e.admit(e.ROOT)
    assert len(bundle["requests"]) == 64
    assert bundle["target_model"]["parameter_count"] == 109
    assert bundle["target_model"]["original_producer"] == 8431
    assert bundle["dependencies_changed"] == []
    assert bundle["historical_cost_null"]
    with pytest.raises(e.base.GateFailure):
        e.admit(tmp_path)
    for index in range(2):
        pins = list(e.PINS)
        with patch.dict(e.PINS, {pins[index]: "sha256:wrong"}):
            with pytest.raises(e.base.GateFailure):
                e.admit(e.ROOT)


def test_real_children_fixed_design_and_replay(tmp_path):
    """REQ-VERIFY-8596: genuine numerical children include startup and shutdown."""
    with e.bindings("20261018"):
        bundle = e.admit(e.ROOT)
        assert len(e.base.roster()) == 240
        work = e.base.empty_work(tmp_path / "raw")
        work.update(bundle=bundle, run_date="20261018")
        for slot in e.base.roster()[:4]:
            row = e.base.session(bundle, slot, tmp_path / slot["arm"])
            assert len(row["requests"]) == 64
            assert row["requests"][0]["latency_s"] >= row["startup_s"]
            assert row["exit_code"] == 0
            assert sum(x["duration_s"] for x in row["operations"]) == pytest.approx(
                row["duration_s"]
            )
        work["ended_ns"] = work["started_ns"] + 1
        atomic_json(tmp_path / "raw" / "measurement.json", work)
        output = tmp_path / (e.NAME + ".json")
        value = e.artifact(work, [], output)
        assert value["experiment_id"] == 8596
        assert value["run_date"] == "20261018"
        assert value["target_model"] == bundle["target_model"]
        assert value["inference_substrate_class"] == "no_model_load"
        assert value["MODEL_SPECS"] == []
        assert value["current_llm_invocations"] == 0
        atomic_json(output, value)
        assert e.base.replay(output)
        changed = deepcopy(value)
        changed["target_model"]["parameter_count"] = 108
        changed["reproducibility_checksum"] = e.base.checksum(changed)
        atomic_json(output, changed)
        assert not e.base.replay(output)


def test_bound_context_restores_consumers():
    """REQ-VERIFY-8596: an adapter cannot rewrite historical producer authority."""
    original = e.base.admit
    with e.bindings("20261017"):
        assert e.base.TASK == e.TASK
        assert e.base.CLI == e.CLI
        assert e.base.admit is e.admit
    assert e.base.admit is original


def test_private_cli_missing_inputs_and_cold_replay(tmp_path):
    """SCENARIO-VERIFY-8596-REJECT: original CLI preserves external blocking."""
    output = tmp_path / (e.NAME + ".json")
    result = child(
        "private",
        [
            sys.executable,
            "-u",
            str(e.ROOT / e.CLI),
            "--date",
            "20261018",
            "--private-output",
            str(output),
            "--input-root",
            str(tmp_path / "missing"),
        ],
        tmp_path / "logs",
        deadline=120,
    )
    assert result["passed"], Path(result["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["run_date"] == "20261018"
    assert value["resident_cost_ready_score"] == 0
    assert len(value["rows"]) == value["intended_count"] == 240
    assert value["censored_count"] == 240
    for name, args, expected in [
        ("replay", ["--cold-replay", str(output)], 0),
        ("missing", ["--cold-replay", str(tmp_path / "absent")], 1),
        ("consume", ["--consume", str(output)], 1),
        ("date", ["--date", "20260230"], 2),
        ("short_date", ["--date", "2026"], 2),
        ("private_path", ["--private-output", str(e.ROOT / "invalid.json")], 2),
        ("input_path", ["--input-root", str(tmp_path)], 2),
    ]:
        assert child(
            name,
            [sys.executable, "-u", str(e.ROOT / e.CLI), *args],
            tmp_path / "logs",
            expected=expected,
            deadline=60,
        )["passed"]


def test_consumer_success_dispatch(monkeypatch, capsys):
    """REQ-VERIFY-8596: the CLI forwards the existing consumer result unchanged."""
    from carnot.reporting import resident_complete_runner_8596 as runner

    monkeypatch.setattr(e.base, "consume", lambda path: dict(passed=True, path=str(path)))
    assert runner.main(["--date", "20261018", "--consume", "/var/tmp/qualified.json"]) == 0
    assert '"passed": true' in capsys.readouterr().out
