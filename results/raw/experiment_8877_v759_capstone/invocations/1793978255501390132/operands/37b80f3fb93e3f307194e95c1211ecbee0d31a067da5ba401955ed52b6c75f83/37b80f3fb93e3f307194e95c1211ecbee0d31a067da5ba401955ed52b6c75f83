"""REQ-REPORT-8851: retained CPU costs never establish physical acceleration."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import hardware_operation_budget_8851 as m


def test_independent_reductions():
    """SCENARIO-REPORT-8851-CONTRACT: either valid branch survives alone."""
    for branch, (name, _, _, _) in m.engine.PRODUCERS.items():
        evidence = json.loads((m.ROOT / name).read_bytes())
        reduced = m.reduce({branch: evidence})
        assert reduced["complete_cost_refs"]
        assert {r["cluster_id"] for r in reduced["board_operation_rows"]} == {
            "KV260",
            "PolarFire",
            "GateMate",
        }
        assert {r["arm"] for r in reduced["board_operation_rows"]} >= {
            "index_build",
            "predicate_lookup",
            "binding_joins",
            "energy_evaluation",
            "gradients",
            "serialization",
            "feedback",
            "fsync",
        }
        assert all(r["measured_board_time_s"] is None for r in reduced["board_operation_rows"])
        assert all("operation_counts_by_branch" in r for r in reduced["board_operation_rows"])
        assert reduced["sampling_call_count"] == 0
        if branch == "whole":
            assert len(reduced["memory_size_requirements"]) == 60
            assert reduced["preparation_cost_refs"]
            assert (
                reduced["complete_cost_refs"][0]["operation_count_units"]["serialization"]
                == "durable store saves; other serialization calls uncounted"
            )
        else:
            assert len(reduced["complete_cost_refs"]) == len(evidence["rows"])


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """REQ-REPORT-8851: private retained bytes also exercise original authorities."""
    return m.collect(tmp_path_factory.mktemp("retained") / "raw")


def validation():
    """SCENARIO-REPORT-8851-CONTRACT: fixture validation grants no readiness."""
    return dict(passed=False, coverage=dict(complete=False), receipts=[])


def test_build_and_replay(work, tmp_path):
    """SCENARIO-REPORT-8851-CONTRACT: independent pins reject repaired checksums."""
    assert set(work["evidence"]) == {"whole", "witness"}, work["failures"]
    value = m.build(work, validation())
    assert value["experiment_id"] == 8851
    assert value["execution_venue"] == "host"
    assert value["verdict_class"] == "disqualified"
    assert value["hardware_budget_ready_score"] == 0
    assert value["current_work_receipt"]["inference_substrate_class"] == "no_model_load"
    assert value["physical_evidence_state"]["GateMate"]["retry_allowed"] is False
    assert value["acquisition_priority"]["recommend_acquisition"] is False
    path = tmp_path / (m.NAME + ".json")
    m.atomic_json(path, value)
    assert m.replay(path)["passed"]
    value["board_operation_rows"][0]["measured_board_time_s"] = 0.001
    value["reproducibility_checksum"] = m.canonical_hash(value["board_operation_rows"])
    m.atomic_json(path, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        m.replay(path)


def test_terminal_classes(work, monkeypatch):
    """REQ-REPORT-8851: owned failure disqualifies; external absence blocks."""
    monkeypatch.setattr(m.engine, "verify_validation", lambda _: None)
    valid = dict(passed=True, coverage=dict(complete=True), receipts=[])
    assert m.build(work, valid)["verdict_class"] == "null"
    one = deepcopy(work)
    one["evidence"].pop("whole")
    one["failures"].append(dict(check="producer_admission", upstream="whole"))
    assert m.build(one, valid)["verdict_class"] == "null"
    one["evidence"] = {}
    assert m.build(one, valid)["verdict_class"] == "blocked"
    one["internal_failure"] = True
    assert m.build(one, valid)["verdict_class"] == "disqualified"
    one["internal_failure"] = False
    one["checks"]["free_disk_bytes"] = 0
    assert m.build(one, valid)["verdict_class"] == "disqualified"


def test_cli(tmp_path):
    """SCENARIO-REPORT-8851-CONTRACT: real cold children use no PYTHONPATH."""
    output = tmp_path / (m.NAME + ".json")
    argv = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for args, expected in [
        (["--private-output", str(output), "--date", "20261105"], 0),
        (["--cold-replay", str(output)], 0),
        (["--replay", str(tmp_path / "missing")], 1),
        (["--date", "20261104"], 2),
    ]:
        result = subprocess.run(
            argv + args, env=env, cwd="/tmp", capture_output=True, text=True, timeout=90
        )
        assert result.returncode == expected, result.stdout + result.stderr


def test_original_bytes_and_admission(tmp_path):
    """REQ-REPORT-8851: mutable specifications resolve only preserved exact bytes."""
    for branch, (name, _, field, _) in m.engine.PRODUCERS.items():
        value = json.loads((m.ROOT / name).read_bytes())
        assert m.engine.admission(value, branch) is None
        value[field] = None
        assert m.engine.admission(value, branch) == "missing_field"
        value[field] = 0
        assert m.engine.admission(value, branch) == "observed_zero"
        value[field] = 1
        value["required_checks_passed"] = False
        assert m.engine.admission(value, branch) == "rejected_producer"
    with pytest.raises(ValueError, match="operand_hash_drift"):
        m.retain(Path(__file__), tmp_path, "sha256:wrong")


def test_retention_guards(tmp_path, monkeypatch):
    """REQ-REPORT-8851: preserved authority and copied bytes must both match."""
    path = tmp_path / "openspec/capabilities/private/spec.md"
    path.parent.mkdir(parents=True)
    path.write_text("original")
    pin = m.sha256_file(path)
    saved = tmp_path / "authorities"
    saved.mkdir()
    (saved / pin.split(":")[-1]).write_bytes(path.read_bytes())
    monkeypatch.setattr(m, "AUTHORITIES", saved)
    path.write_text("new requirements")
    assert m.retain(path, tmp_path / "copy", pin)["sha256"] == pin
    (saved / pin.split(":")[-1]).write_text("corrupt")
    with pytest.raises(ValueError, match="operand_hash_drift"):
        m.retain(path, tmp_path / "bad", pin)
    original_hash = m.sha256_file
    monkeypatch.setattr(
        m, "sha256_file", lambda p: "sha256:corrupt" if "operands" in p.parts else original_hash(p)
    )
    with pytest.raises(ValueError, match="operand_hash_drift"):
        m.retain(path, tmp_path / "copy")


def test_missing_authority(work, monkeypatch):
    """REQ-REPORT-8851: absent board authority remains a named external block."""
    monkeypatch.setattr(m.engine, "verify_validation", lambda _: None)
    absent = deepcopy(work)
    absent["refs"] = [r for r in absent["refs"] if "8302" not in r["original_path"]]
    absent["failures"] = [dict(check="required_operand", upstream="board_authority_or_source")]
    result = m.build(absent, dict(passed=True, coverage=dict(complete=True), receipts=[]))
    assert result["verdict_class"] == "blocked"
    assert result["hardware_budget_ready_score"] == 0
    assert result["physical_evidence_state"]["GateMate"]["receipt"] is None


def test_main_with_bound_validation(tmp_path, monkeypatch):
    """REQ-REPORT-8851: production calls owned qualification and terminal scanners."""
    from carnot.reporting import hardware_budget_validation_8851 as checks

    monkeypatch.setattr(checks, "qualify", lambda _: validation())
    monkeypatch.setattr(checks, "terminal", lambda p, _: m.replay(p))
    monkeypatch.setattr(m, "DEFAULT_OUTPUT", tmp_path / (m.NAME + ".json"))
    assert m.main([]) == 0
