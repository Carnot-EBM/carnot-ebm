"""REQ-REPORT-8876: retain installed costs even when learning evidence is absent."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import hardware_operation_budget_8876 as m


def validation():
    """Fixture checks cannot qualify a production artifact."""
    return dict(passed=False, receipts=[], coverage=dict(complete=False))


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """SCENARIO-REPORT-8876-CONTRACT: collect real immutable installed evidence."""
    return m.collect(tmp_path_factory.mktemp("hardware8876") / "raw")


def test_admission_and_counts():
    """REQ-REPORT-8876: a valid null is usable, while absent counters stay unknown."""
    value = json.loads((m.ROOT / m.engine.PRODUCERS["witness"][0]).read_bytes())
    assert m.admission(value, "witness") is None
    reduced = m.reduce({"witness": value})
    assert len(reduced["complete_cost_refs"]) == 15840
    assert reduced["sampling_call_count"] == 0
    assert reduced["end_to_end_gain_bound"] is None
    for arm in ("scan", "indexed", "size_aware"):
        expected = sum(
            r["operation_counts"].get("index_build", 0) for r in value["rows"] if r["arm"] == arm
        )
        assert (
            sum(
                r["operation_counts"].get("index_build", 0)
                for r in reduced["complete_cost_refs"]
                if r["arm"] == arm
            )
            == expected
        )
    assert any(r["status"] == "censored" for r in reduced["complete_cost_refs"])
    assert all(r["measured_board_time_s"] is None for r in reduced["board_operation_rows"])
    assert reduced["label_call_savings"]["saved_calls"] is None
    for savings in reduced["lookup_operation_savings"]:
        sample = [r for r in value["rows"] if r["boundary"] == savings["boundary"]]
        assert savings["saved_index_build_units"] == sum(
            r["operation_counts"].get("index_build", 0)
            * (1 if r["arm"] == "scan" else -1 if r["arm"] == "size_aware" else 0)
            for r in sample
        )
    field = "size_cost_ready_score"
    for changed, reason in [
        (None, "missing_field"),
        (0, "observed_zero"),
        (2, "rejected_producer"),
    ]:
        mutated = dict(value, **{field: changed})
        assert m.admission(mutated, "witness") == reason
    value["required_checks_passed"] = False
    assert m.admission(value, "witness") == "rejected_producer"
    assert m.admission({}, "learning") == "rejected_producer"
    assert m.reduce({})["complete_cost_refs"] == []


def test_build(work, monkeypatch):
    """SCENARIO-REPORT-8876-CONTRACT: blocked learning never suppresses installed work."""
    assert set(work["evidence"]) == {"witness"}, work["failures"]
    value = m.build(work, validation())
    assert value["verdict_class"] == "disqualified"
    assert value["complete_cost_refs"]
    assert value["hardware_budget_ready_score"] == 0
    assert value["physical_evidence_state"]["GateMate"]["retry_allowed"] is False
    assert value["numerical_training_calls"] == 0
    monkeypatch.setattr(m.engine, "verify_validation", lambda _: None)
    valid = dict(passed=True, receipts=[], coverage=dict(complete=True))
    qualified = m.build(work, valid)
    assert qualified["verdict_class"] == "blocked"
    assert qualified["required_checks_passed"] is True
    assert qualified["independent_count"] == 220
    assert len(qualified["learning_operation_rows"]) == 2
    assert {r["missing_reason"] for r in qualified["learning_operation_rows"]} == {"missing_file"}
    assert (
        qualified["completed_count"] + qualified["censored_count"] + qualified["blocked_count"]
        == qualified["intended_count"]
    )
    empty = deepcopy(work)
    empty["evidence"] = {}
    assert m.build(empty, valid)["inference_substrate_class"] == "blocked_no_run"
    empty["internal_failure"] = True
    assert m.build(empty, valid)["verdict_class"] == "disqualified"
    clean = deepcopy(work)
    clean["failures"] = []
    assert m.build(clean, valid)["verdict_class"] == "null"


def test_cli(tmp_path):
    """SCENARIO-REPORT-8876-CONTRACT: actual foreign-directory CLI rejects tampering."""
    output = tmp_path / (m.NAME + ".json")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for args, expected in [
        (["--private-output", str(output), "--date", "20261106"], 0),
        (["--cold-replay", str(output)], 0),
        (["--replay", str(tmp_path / "missing")], 1),
        (["--date", "20261105"], 2),
    ]:
        result = subprocess.run(
            [sys.executable, "-u", str(m.ROOT / m.CLI), *args],
            cwd="/tmp",
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == expected, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert "--private-output" in value["invocation"]["argv"]
    original = deepcopy(value)
    value["label_call_savings"]["saved_calls"] = 100
    value["reproducibility_checksum"] = m.canonical_hash(value["rows"])
    m.atomic_json(output, value)
    assert m.main(["--replay", str(output)]) == 1
    m.atomic_json(output, original)
    operand = Path(original["source_artifact_hashes"][0]["path"])
    operand.write_bytes(operand.read_bytes() + b"\nchanged")
    assert m.main(["--replay", str(output)]) == 1


def test_production_route(work, tmp_path, monkeypatch):
    """REQ-REPORT-8876: production calls the real qualification and terminal route."""
    from carnot.reporting import hardware_budget_validation_8876 as checks

    monkeypatch.setattr(checks, "qualify", lambda _: validation())
    monkeypatch.setattr(checks, "terminal", lambda p, _: dict(passed=True))
    monkeypatch.setattr(m, "collect", lambda _: work)
    monkeypatch.setattr(m, "DEFAULT_OUTPUT", tmp_path / (m.NAME + ".json"))
    assert m.main([]) == 0
