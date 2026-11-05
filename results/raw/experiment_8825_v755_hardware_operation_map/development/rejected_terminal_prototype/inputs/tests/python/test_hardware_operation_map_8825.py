"""REQ-REPORT-8825 and REQ-VERIFY-8825: unknown costs never become fractions."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import hardware_operation_map_8825 as m


def test_independent_census(tmp_path):
    """SCENARIO-REPORT-8825-OPERATIONS: absent branches retain every board operation."""
    result = m.reduce({})
    assert len(result["board_operation_rows"]) == 3 * len(m.OPERATIONS)
    assert set(result["supported_operation_counts"]) == {"KV260", "PolarFire", "GateMate"}
    assert result["sampling_call_count"] == 0
    assert result["target_latency_requirements"] == []
    assert all(r["missing_reason"] == "missing_file" for r in result["gate_check_summary"])
    assert result["operation_inventory"]["dense_forward"]["scalar_multiplies"] == 120
    assert result["operation_inventory"]["three_label_normalization"]["candidates"] == 3
    assert result["physical_evidence_state"]["GateMate"]["retry_allowed"] is False


@pytest.mark.parametrize(
    "producer,reason",
    [
        ({}, "missing_field"),
        ({"whole_cost_ready_score": 0}, "observed_zero"),
        ({"whole_cost_ready_score": 1, "flagged_adversarial": True}, "rejected_producer"),
    ],
)
def test_gate_missingness(producer, reason):
    """REQ-REPORT-8825: missing, zero and rejected operands remain distinct."""
    evidence = {"cost": {"value": producer, "path": "cost.json", "sha256": "sha256:abc"}}
    assert m.reduce(evidence)["gate_check_summary"][-1]["missing_reason"] == reason


def test_complete_cost_and_learning_independent():
    """REQ-REPORT-8825: complete whole budgets grant requirements only, never curves."""
    producer = {
        "whole_cost_ready_score": 1,
        "required_checks_passed": True,
        "flagged_adversarial": False,
        "honest_verdict": "complete_null_cost",
        "verdict_class": "null",
        "complete_operation_rows": [{"unit_id": "window", "total_s": 2.5, "denominator": 1}],
    }
    evidence = {"cost": {"value": producer, "path": "cost.json", "sha256": "sha256:abc"}}
    result = m.reduce(evidence)
    assert result["target_latency_requirements"][0]["target_latency_s"] == 0.025
    assert result["gate_check_summary"][0]["upstream"] == "learning"
    producer["complete_operation_rows"][0]["total_s"] = -1
    assert m.reduce(evidence)["target_latency_requirements"] == []
    evidence["learning"] = {
        "value": {
            "stream_rows_ready_score": 1,
            "required_checks_passed": True,
            "flagged_adversarial": False,
            "honest_verdict": "complete_null_stream",
            "verdict_class": "null",
        },
        "path": "learning.json",
        "sha256": "sha256:abc",
    }
    assert all(r["upstream"] != "learning" for r in m.reduce(evidence)["gate_check_summary"])


def test_private_cli_replay(tmp_path):
    """SCENARIO-VERIFY-8825-SEALED: real child replay rejects rehashed semantic drift."""
    output = tmp_path / (m.NAME + ".json")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    built = subprocess.run(
        command + ["--date", "20261105", "--private-output", str(output)],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    replay = subprocess.run(
        command + ["--replay", str(output)],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    artifact = json.loads(output.read_text())
    artifact["sampling_call_count"] = 1
    m.atomic_json(output, artifact)
    replay = subprocess.run(
        command + ["--replay", str(output)],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert replay.returncode == 1
    assert "reduction_drift" in replay.stdout
    with pytest.raises(ValueError, match="reduction_drift"):
        m.replay(output)
    missing = tmp_path / "absent.json"
    with pytest.raises(FileNotFoundError):
        m.replay(missing)


def test_build_and_seal(tmp_path):
    """REQ-VERIFY-8825: failed owned validation disqualifies; source changes fail replay."""
    evidence, refs = m.snapshot(tmp_path)
    validation = {"passed": False, "receipts": [], "coverage": {"percent": 0}}
    artifact = m.build(evidence, refs, validation, 1, 2, "20261105")
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["hardware_map_ready_score"] == 0
    output = tmp_path / (m.NAME + ".json")
    m.atomic_json(output, artifact)
    assert m.replay(output)["passed"]
    ref = next(iter(refs.values()))
    Path(ref["path"]).write_text("changed")
    with pytest.raises(ValueError, match="source_hash_drift"):
        m.replay(output)


def test_main_date_and_missing_replay(tmp_path):
    """REQ-VERIFY-8825: invalid dates fail before writing or executing science."""
    with pytest.raises(SystemExit):
        m.main(["--date", "20261104"])
    assert m.main(["--replay", str(tmp_path / "absent")]) == 1


def test_snapshot_admits_available_branch(tmp_path, monkeypatch):
    """REQ-REPORT-8825: retain optional producers without making them prerequisites."""
    from carnot.reporting.current_work_receipt import atomic_json

    monkeypatch.setattr(m, "BRANCHES", {"cost": ("producer.json", "whole_cost_ready_score")})
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "SOURCES", [])
    monkeypatch.setattr(m, "HISTORY", {})
    atomic_json(tmp_path / "producer.json", {"whole_cost_ready_score": 0})
    evidence, _ = m.snapshot(tmp_path / "raw")
    assert evidence["cost"]["value"]["whole_cost_ready_score"] == 0
    monkeypatch.setattr(m, "HISTORY", {"producer.json": "sha256:wrong"})
    with pytest.raises(ValueError, match="historical_authority_drift"):
        m.snapshot(tmp_path / "rejected")


def test_validation_and_main_paths(tmp_path, monkeypatch):
    """REQ-VERIFY-8825: validation measures all owned files and retains failed checks."""
    from carnot.reporting import hardware_operation_runner_8825 as r

    class Tracer:
        def __init__(self, **kwargs):
            pass

        def combine(self, **kwargs):
            pass

        def save(self):
            pass

        def json_report(self, outfile):
            m.atomic_json(
                Path(outfile),
                {
                    "files": {
                        str(i): {
                            "summary": {"num_statements": 1},
                            "missing_lines": [],
                            "excluded_lines": [],
                        }
                        for i in range(len(m.OWNED))
                    },
                    "totals": {"percent_covered": 100},
                },
            )

    monkeypatch.setattr(r, "Coverage", Tracer)
    monkeypatch.setattr(
        r,
        "run_commands",
        lambda root, plan, **kw: [{"passed": True, "scope": s.scope} for s in plan],
    )
    assert r.validate(tmp_path)["passed"]
    assert r.terminal(tmp_path / "candidate.json", tmp_path)["passed"]
    monkeypatch.setattr(r, "publish_primary", lambda output, value, validator: validator(output))
    monkeypatch.setattr(m, "replay", lambda path: {"passed": True})
    r.publish(tmp_path / "candidate.json", {}, tmp_path, private=True)
    r.publish(tmp_path / "candidate.json", {}, tmp_path, private=False)
    monkeypatch.setattr(
        r, "validate", lambda raw: {"passed": False, "receipts": [], "coverage": {"percent": 0}}
    )
    monkeypatch.setattr(r, "publish", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        m,
        "snapshot",
        lambda raw: (
            {},
            {p: {"path": str(m.ROOT / p), "sha256": m.sha256_file(m.ROOT / p)} for p in m.OWNED},
        ),
    )
    assert m.main(["--date", "20261105"]) == 0
    monkeypatch.setattr(
        r, "run_commands", lambda *args, **kw: [{"passed": False, "scope": "owned"}]
    )
    assert not r.validate(tmp_path)["passed"]


def test_replay_producer_and_receipt_tamper(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8825-REPLAY: byte seals and current clocks are independent checks."""
    evidence, refs = m.snapshot(tmp_path)
    source = tmp_path / "cost.json"
    m.atomic_json(source, {})
    evidence["cost"] = {"value": {}, "path": str(source), "sha256": m.sha256_file(source)}
    artifact = m.build(
        evidence,
        refs,
        {"passed": True, "receipts": [], "coverage": {"percent": 100}},
        1,
        2,
        "20261105",
    )
    path = tmp_path / (m.NAME + ".json")
    m.atomic_json(path, artifact)
    assert m.replay(path)["passed"]
    artifact["duration_s"] = 50
    m.atomic_json(path, artifact)
    with pytest.raises(ValueError, match="current_receipt_drift"):
        m.replay(path)
    m.atomic_json(source, {"changed": True})
    with pytest.raises(ValueError, match="producer_hash_drift"):
        m.replay(path)
