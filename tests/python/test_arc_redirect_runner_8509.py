"""REQ-VERIFY-8509: the thin CLI owns deadlines, rejection and exact-byte publication."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import arc_redirect_runner_8509 as runner
from carnot.reporting import arc_redirect_effect_8509 as e
from carnot.reporting.current_work_receipt import atomic_json


def test_cli_date_failure_and_missing_replay(tmp_path):
    """SCENARIO-VERIFY-8509: real children reject bad dates and missing primitive evidence."""
    for args, code in [
        (["--date", "20261013"], 2),
        (["--deliberate-error"], 7),
        (["--cold-replay", str(tmp_path / "absent.json")], 1),
    ]:
        proc = subprocess.run(
            [sys.executable, str(e.ROOT / e.CLI), *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == code, proc.stdout + proc.stderr


def test_private_cli_block_and_replay(tmp_path, monkeypatch):
    """REQ-VERIFY-8509: external absence is terminal blocked through unchanged consumers."""
    monkeypatch.setattr(runner, "validate_checks", lambda *_: [dict(passed=True)])
    output = tmp_path / "results" / (e.NAME + ".json")
    assert runner.main(["--root", str(tmp_path / "missing"), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["generalization_diagnostic_ready_score"] == 0
    assert value["gate_check_summary"]
    assert runner.main(["--cold-replay", str(output)]) == 0
    side = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    assert side["consumer"]["passed"]


def test_owned_validation_failure_disqualifies(tmp_path, monkeypatch):
    """REQ-VERIFY-8509: failed owned checks cannot receive null qualification."""
    monkeypatch.setattr(runner, "validate_checks", lambda *_: [dict(passed=False)])
    output = tmp_path / "results" / (e.NAME + ".json")
    assert runner.main(["--root", str(tmp_path / "missing"), "--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"


def test_gate_schema_and_changed_activation(tmp_path):
    """REQ-VERIFY-8509: unavailable is distinct from a recorded zero."""
    gate = runner.gate("input", tmp_path / "missing", "sha256", "abc", None)
    assert gate["observed"] is None and gate["operator"] == "=="
    import yaml

    (tmp_path / "research-roadmap.yaml").write_text(yaml.safe_dump({"tasks": [{"id": e.TASK}]}))
    checks, resources, hashes = runner.preconditions(tmp_path, tmp_path / "scratch")
    assert any(g["check"] == "activated_task" for g in checks)
    assert resources["optional_cuda_required"] is False
    assert hashes


def test_validation_collection_union_and_failure(tmp_path, monkeypatch):
    """REQ-VERIFY-8509: bounded disjoint shards preserve every original node and assertion."""
    calls = []

    def child(name, argv, logs, **kwargs):
        calls.append((name, argv))
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / (name + ".stdout")
        log.write_text(
            "tests/python/private.py::test_one\ntests/python/private.py::test_two\n"
            if name.startswith("collect")
            else ""
        )
        from coverage import Coverage

        shard = Coverage(data_file=str(tmp_path / "private" / ".coverage.fixture"))
        shard.start()
        e.progress("private_genuine_shard_copy_control")
        shard.stop()
        shard.save()
        return dict(passed=True, stdout_path=str(log))

    monkeypatch.setattr(runner, "child", child)
    receipts = runner.validate_checks(tmp_path / "raw", tmp_path / "private")
    assert receipts and all(r["passed"] for r in receipts)
    union = json.loads(
        (tmp_path / "raw" / "validation" / "historical_node_union.json").read_bytes()
    )
    assert all(v["union_equals_collection"] for v in union.values())
    assert any(
        "-n" in argv and "--no-cov" in argv for name, argv in calls if name.startswith("historical")
    )
    monkeypatch.setattr(runner, "child", lambda *_a, **_k: dict(passed=False))
    assert not all(r["passed"] for r in runner.validate_checks(tmp_path / "other", tmp_path / "p2"))


def test_private_live_receipt_publication(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8509: original primitives cold-replay through actual private consumers."""
    monkeypatch.setattr(runner, "validate_checks", lambda *_: [dict(passed=True)])
    output = tmp_path / "results" / (e.NAME + ".json")
    assert runner.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "null" and value["completed_count"] == 32
    assert value["generalization_diagnostic_ready_score"] == 1
    assert runner.main(["--cold-replay", str(output)]) == 0


def test_resource_and_deadline_rejection(tmp_path, monkeypatch):
    """REQ-VERIFY-8509: unavailable tools and an exhausted measurement window fail closed."""
    import shutil

    with monkeypatch.context() as patcher:
        patcher.setattr(
            runner.shutil, "disk_usage", lambda *_: shutil._ntuple_diskusage(100, 99, 1)
        )
        checks, _, _ = runner.preconditions(e.ROOT, tmp_path / "scratch")
        assert any(g["check"] == "free_disk_bytes" for g in checks)
    ticks = iter([0, 3601])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="measurement_deadline"):
        runner.main(
            [
                "--root",
                str(tmp_path / "missing"),
                "--output",
                str(tmp_path / "results" / (e.NAME + ".json")),
            ]
        )


def test_publication_failure_preserves_findings(tmp_path, monkeypatch):
    """REQ-VERIFY-8509: rejected candidates remain evidence; findings are never cleared."""
    work_path = tmp_path / "work.json"
    work = dict(
        source=None,
        fixture_rows=[],
        failures=[dict(check="missing")],
        preconditions_checked={},
        duration_s=1,
        phase_spans=[],
    )
    atomic_json(work_path, work)
    value = e.build(work_path, [dict(passed=True)], tmp_path / "results" / (e.NAME + ".json"))
    monkeypatch.setattr(runner, "child", lambda *_a, **_k: dict(passed=False))
    with pytest.raises(ValueError, match="candidate_rejected"):
        runner.publish(
            value,
            Path(value["terminal_validation_sidecar_path"]).parents[2] / (e.NAME + ".json"),
            tmp_path / "raw",
        )
    assert (tmp_path / "raw" / "publication_attempts.json").exists()
