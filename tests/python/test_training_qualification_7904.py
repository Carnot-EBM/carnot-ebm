"""Private runtime authorities for REQ-VERIFY-7904-V686 and REQ-REPORT-7904-V686."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.verify import training_qualification_7904 as q

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts/experiments/experiment_7904_v686_training_qualification.py"


def invoke(tmp_path: Path, *args: str, expected: int = 0) -> subprocess.CompletedProcess[str]:
    """Measure real child entrypoints so imports cannot stand in for CLI coverage."""
    command = [sys.executable, "-u"]
    if os.environ.get("CARNOT7904_COVERAGE"):
        command += ["-m", "coverage", "run", "--parallel-mode", "--include=" + q.INCLUDES]
    command += [str(CLI), "--date", "20260930", *args]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert result.returncode == expected, result.stdout + result.stderr
    (tmp_path / f"child-{len(list(tmp_path.glob('child-*')))}.log").write_text(
        result.stdout + result.stderr
    )
    return result


def test_checkpoint_closure_and_path_rejection(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7904-CHECKPOINT: upstream paths and learning bytes bind heads."""
    upstream = q.make_fixture(tmp_path / "input")
    identity = q.checkpoint_identity(upstream, "local_set", 67801, 1)
    assert all(name in identity["dependencies"] for name in q.NUMERICAL_MODULES)
    altered = json.loads(json.dumps(identity))
    altered["dependencies"]["python/carnot/verify/natural_training.py"] = "sha256:changed"
    with pytest.raises(ValueError, match="checkpoint dependency mismatch"):
        q.check_compatible(identity, altered)
    other = tmp_path / "another-upstream.json"
    other.write_bytes(upstream.read_bytes())
    with pytest.raises(ValueError, match="checkpoint dependency mismatch"):
        q.check_compatible(identity, q.checkpoint_identity(other, "local_set", 67801, 1))


def test_real_cli_fit_resume_and_replay(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7904-CLI: fitting seals fresh heads and replay checks primitive bytes."""
    upstream = q.make_fixture(tmp_path / "input")
    output, raw = tmp_path / "fixture.json", tmp_path / "raw"
    args = [
        "--fixture",
        "--upstream",
        str(upstream),
        "--output",
        str(output),
        "--raw-root",
        str(raw),
    ]
    invoke(tmp_path, *args)
    first = json.loads(output.read_text())
    assert first["verdict_class"] == "circular_positive"
    assert first["sample_size_budget"]["independent"] == 0
    assert first["acceptance_gate_results"]["decision_benefit"] is None
    invoke(tmp_path, *args, "--resume")
    assert (
        first["fixture_prediction_rows"]
        == json.loads(output.read_text())["fixture_prediction_rows"]
    )
    invoke(tmp_path, "--cold-replay", str(output))
    rows = Path(first["prediction_rows_path"])
    rows.write_text(rows.read_text() + "\n")
    invoke(tmp_path, "--cold-replay", str(output), expected=2)


def test_real_cli_external_block_and_owned_failures(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7904-CLI: external blocks and owned failures retain different verdicts."""
    output = tmp_path / "missing.json"
    invoke(
        tmp_path,
        "--fixture",
        "--upstream",
        str(tmp_path / "absent.json"),
        "--output",
        str(output),
        "--raw-root",
        str(tmp_path / "missing-raw"),
        expected=2,
    )
    blocked = json.loads(output.read_text())
    assert blocked["honest_verdict"].startswith("complete_blocked_")
    assert blocked["gate_check_summary"][0]["observed"] is None
    upstream = q.make_fixture(tmp_path / "input")
    for flag, value in (
        ("--fit-deadline-s", "0"),
        ("--terminal-recheck", str(tmp_path / "bad-reader.py")),
    ):
        (tmp_path / "bad-reader.py").write_text("raise SystemExit(1)\n")
        invoke(
            tmp_path,
            "--fixture",
            "--upstream",
            str(upstream),
            "--output",
            str(output),
            "--raw-root",
            str(tmp_path / flag[2:]),
            flag,
            value,
            expected=2,
        )
        failed = json.loads(output.read_text())
        assert failed["verdict_class"] == "disqualified"
        assert failed["training_runtime_ready_score"] == 0


def test_group_overlap_is_rejected(tmp_path: Path) -> None:
    """REQ-VERIFY-7904-V686: equal source groups cannot serve both fitting and tuning."""
    upstream = q.make_fixture(tmp_path / "input", overlap=True)
    with pytest.raises(ValueError, match="source group overlap"):
        q.fit_score(upstream, tmp_path / "output.json", tmp_path / "raw", 7904, "20260930")


def test_frozen_scope_history_and_producer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7904-SCOPE: actual orchestration retains old failures and requires full new coverage."""
    private, raw = tmp_path / "private", tmp_path / "durable"
    private.mkdir()
    upstream = q.make_fixture(tmp_path / "authority")
    fixture = q.make_fixture(private / "fixture-input")
    q.fit_score(fixture, private / "fixture.json", private / "fixture-raw", 7904, "20260930")
    (private / "coverage.json").write_text(
        json.dumps(
            {
                "files": {
                    name: {"summary": {"num_statements": 1, "covered_lines": 1, "missing_lines": 0}}
                    for name in q.OWNED
                }
            }
        )
    )

    def fake_execute(commands, raw, heartbeat_s=30, extra_env=None):
        return [
            {"name": item.name, "argv": list(item.argv), "passed": True, "actual_exit": 0}
            for item in commands
        ]

    monkeypatch.setattr(q, "execute", fake_execute)
    monkeypatch.setattr(q, "publish", lambda *args: True)
    output = tmp_path / "qualified.json"
    (raw / "repository_health").mkdir(parents=True)
    (raw / "repository_health/receipt.json").write_text(
        json.dumps(
            {"receipts": [{"passed": False, "actual_exit": -15, "name": "full_repository_suite"}]}
        )
    )
    assert q.qualify(upstream, output, raw, "20260930", private=private) == 0
    result = json.loads((raw / "qualified_candidate.json").read_text())
    assert result["training_runtime_ready_score"] == 1
    assert result["repository_health"]["full_suite"]["receipts"][0]["passed"] is False
    assert result["historical_required_failures"][-1]["actual_exit"] == 1
    manifest = json.loads(Path(result["validation_command_manifest_path"]).read_text())
    spec = next(row for row in manifest["commands"] if row["name"] == "scoped_spec_coverage")
    assert spec["argv"][-1].endswith("test_source_boundary_7852.py")
    assert CLI.name in q.INCLUDES
    (private / "coverage.json").write_text(json.dumps({"files": {}}))
    q.qualify(upstream, output, raw, "20260930", private=private)
    assert (
        json.loads((raw / "qualified_candidate.json").read_text())["training_runtime_ready_score"]
        == 0
    )


def test_actual_module_bytes_mutate_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7904-CHECKPOINT: unchanged wrapper cannot hide a changed learning file."""
    upstream = q.make_fixture(tmp_path / "input")
    clone = tmp_path / "code"
    for name in q.DEPENDENCIES:
        destination = clone / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    monkeypatch.setattr(q, "ROOT", clone)
    saved = q.checkpoint_identity(upstream, "local_set", 67801, 1)
    learning = clone / "python/carnot/verify/natural_training.py"
    learning.write_text(learning.read_text() + "\n# Changed numerical authority.\n")
    current = q.checkpoint_identity(upstream, "local_set", 67801, 1)
    assert current["dependencies"][q.MODULE] == saved["dependencies"][q.MODULE]
    with pytest.raises(ValueError, match="checkpoint dependency mismatch"):
        q.check_compatible(saved, current)


def test_checkpoint_and_primitive_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7904-CHECKPOINT: cold readers reject manifests, heads, predictions and counts."""
    upstream = q.make_fixture(tmp_path / "input")
    raw, output = tmp_path / "raw", tmp_path / "fixture.json"
    result = q.fit_score(upstream, output, raw, 7904, "20260930")
    manifest_path = Path(result["checkpoint_manifest_path"])
    manifest_bytes = manifest_path.read_bytes()
    manifest_path.write_bytes(manifest_bytes + b"\n")
    with pytest.raises(ValueError, match="checkpoint manifest mismatch"):
        q.replay(output)
    manifest_path.write_bytes(manifest_bytes)
    checkpoint = Path(json.loads(manifest_bytes)["checkpoint_path"])
    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint.write_bytes(checkpoint_bytes + b"\n")
    with pytest.raises(ValueError, match="checkpoint bytes mismatch"):
        q.replay(output)
    with pytest.raises(ValueError, match="checkpoint bytes mismatch"):
        q.fit_score(upstream, output, raw, 7904, "20260930", resume=True)
    checkpoint.write_bytes(checkpoint_bytes)
    with pytest.raises(TimeoutError, match="owned fitting deadline"):
        q.fit_score(upstream, output, raw, 7904, "20260930", resume=True, deadline_s=1e-9)
    real_predict = q.natural_training.predict
    monkeypatch.setattr(q.natural_training, "predict", lambda *args: [{"action": "changed"}] * 3)
    with pytest.raises(ValueError, match="cold prediction mismatch"):
        q.replay(output)
    monkeypatch.setattr(q.natural_training, "predict", real_predict)
    result["sample_size_budget"]["completed"] = 500
    output.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="primitive denominator mismatch"):
        q.replay(output)


def test_history_hash_tamper_and_real_child_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7904-SCOPE: altered historical log bytes and actual child deadlines fail."""
    original = q.sha256_file
    artifact = json.loads((ROOT / "results/experiment_7894_v685_energy_fit.json").read_text())
    log = next(
        Path(row["log_path"]) for row in artifact["validation_receipts"] if not row["passed"]
    )
    monkeypatch.setattr(
        q, "sha256_file", lambda path: "sha256:changed" if path == log else original(path)
    )
    with pytest.raises(q.custody.InputBlocked, match="historical_required_log"):
        q.history()
    monkeypatch.setattr(q, "sha256_file", original)
    receipts = q.execute(
        [
            q.CommandSpec(
                "private_timeout",
                (sys.executable, "-c", "import time; time.sleep(5)"),
                "private_failure",
                0.02,
            )
        ],
        tmp_path / "raw",
        heartbeat_s=0.01,
    )
    assert receipts[0]["timed_out"] is True and receipts[0]["passed"] is False
    assert original(Path(receipts[0]["log_path"])) == receipts[0]["log_sha256"]


def test_real_producer_missing_source(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7904-SCOPE: the actual producer route terminates with named external operands."""
    invoke(
        tmp_path,
        "--upstream",
        str(tmp_path / "absent.json"),
        "--output",
        str(tmp_path / "blocked.json"),
        "--raw-root",
        str(tmp_path / "raw"),
        expected=2,
    )
    assert json.loads((tmp_path / "blocked.json").read_text())["verdict_class"] == "blocked"


def test_producer_owned_missing_child_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7904-SCOPE: a failed owned CLI cannot leave a partial producer artifact."""
    upstream = q.make_fixture(tmp_path / "input")

    def failed_execute(commands, raw, heartbeat_s=30, extra_env=None):
        if commands[0].name == "adversarial":
            return real_execute(commands, raw, heartbeat_s, extra_env)
        return [
            {"name": item.name, "argv": list(item.argv), "passed": False, "actual_exit": 2}
            for item in commands
        ]

    real_execute = q.execute
    monkeypatch.setattr(q, "execute", failed_execute)
    output = tmp_path / "output.json"
    assert q.qualify(upstream, output, tmp_path / "raw", "20260930") == 2
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"


def test_actual_cli_rejects_equal_bytes_at_another_path(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7904-CHECKPOINT: a fresh path cannot promote an existing fitted head."""
    upstream = q.make_fixture(tmp_path / "input")
    raw, output = tmp_path / "raw", tmp_path / "fixture.json"
    q.fit_score(upstream, output, raw, 7904, "20260930")
    other = tmp_path / "other-authority.json"
    other.write_bytes(upstream.read_bytes())
    invoke(
        tmp_path,
        "--fixture",
        "--resume",
        "--upstream",
        str(other),
        "--output",
        str(output),
        "--raw-root",
        str(raw),
        expected=2,
    )
    assert "checkpoint dependency mismatch" in json.dumps(
        json.loads(output.read_text())["gate_check_summary"]
    )


def test_identity_mutation_rows_are_observed(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7904-CHECKPOINT: each recorded rejection comes from the compatibility reader."""
    upstream = q.make_fixture(tmp_path / "input")
    rows = q.mutation_rows(q.checkpoint_identity(upstream, "local_set", 67801, 1))
    assert {"upstream_path", *q.NUMERICAL_MODULES} <= {row["mutation"] for row in rows}
    assert all(
        row["rejected"] and row["observed"] == "checkpoint dependency mismatch" for row in rows
    )
