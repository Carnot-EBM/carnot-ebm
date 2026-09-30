"""Current orchestration checks for REQ-REPORT-7916-V687 and REQ-VERIFY-7916-V687."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.verify import training_qualification_7904 as prior
from carnot.verify import training_qualification_7916 as current

ROOT = prior.ROOT


def invoke(tmp_path: Path, *args: str, expected: int = 0) -> None:
    """Collect real script coverage so importing a CLI cannot qualify its branches."""
    command = [sys.executable, "-u"]
    if os.environ.get("CARNOT7916_COVERAGE"):
        command += ["-m", "coverage", "run", "--parallel-mode", "--include=" + current.INCLUDES]
    command += [str(ROOT / current.CLI), "--date", "20260930", *args]
    child = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert child.returncode == expected, child.stdout + child.stderr
    (tmp_path / f"child-{len(list(tmp_path.glob('child-*.log')))}.log").write_text(
        child.stdout + child.stderr
    )


def test_isolated_scope_and_frozen_dates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7916-SCOPE: current configuration cannot rewrite prior authority."""
    runtime = current.runtime()
    assert prior.MODULE != runtime.MODULE
    assert prior.CLI != runtime.CLI
    private = tmp_path / "private"
    private.mkdir()
    manifest, commands = runtime.freeze(private, tmp_path / "raw", "20260930")
    frozen = json.loads(manifest.read_text())
    assert frozen["execution_date"] == "20260930"
    assert frozen["historical_fixture_date"] == "20260929"
    for row in commands:
        if row.name.startswith("e2e_016"):
            assert row.argv[row.argv.index("--date") + 1] == "20260929"
    assert set(current.OWNED) <= set(frozen["changed_modules"])
    assert all(item["expected_exit"] == 0 for item in frozen["commands"])
    assert all(name in frozen["affected_dependency_closure"] for name in prior.NUMERICAL_MODULES)
    history = runtime.history()
    failures = [
        r for r in history["historical_required_failures"] if r.get("experiment_id") == 7904
    ]
    assert {r["name"] for r in failures} == {"e2e_016_fixture", "e2e_016_replay"}
    assert all("run_date_mismatch" in r["output_tail"] for r in failures)
    assert history["repository_health"]["historical_exp7904_covered_statements"] == 290


def test_wrong_historical_date_is_rejected(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7916-SCOPE: execution date cannot replace the historical fixture date."""
    command = [
        sys.executable,
        "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
        "--date",
        "20260930",
        "--fixture-e2e",
        str(tmp_path / "wrong-date.json"),
    ]
    child = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert child.returncode == 1 and "run_date_mismatch" in child.stderr
    assert not (tmp_path / "wrong-date.json").exists()


def test_current_cli_resume_replay_and_checkpoint_rejection(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7916-CHECKPOINT: only identical current authority permits reuse."""
    upstream = prior.make_fixture(tmp_path / "input")
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
    assert (first["experiment_id"], first["task_id"], first["milestone"]) == (
        7916,
        "exp7916-training-qualification",
        "2026.09.687",
    )
    assert first["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert set(current.OWNED) <= set(first["checkpoint_identity"]["dependencies"])
    invoke(tmp_path, *args, "--resume")
    assert (
        json.loads(output.read_text())["fixture_prediction_rows"]
        == first["fixture_prediction_rows"]
    )
    invoke(tmp_path, "--cold-replay", str(output))
    manifest = Path(first["checkpoint_manifest_path"])
    saved = json.loads(manifest.read_text())
    saved["identity"]["dependencies"]["python/carnot/verify/natural_training.py"] = "sha256:changed"
    manifest.write_text(json.dumps(saved))
    invoke(tmp_path, *args, "--resume", expected=2)
    assert "checkpoint dependency mismatch" in output.read_text()
    invoke(tmp_path, "--cold-replay", str(output), expected=2)


@pytest.mark.parametrize("fixture", [True, False])
def test_current_cli_missing_source(tmp_path: Path, fixture: bool) -> None:
    """SCENARIO-VERIFY-7916-CHECKPOINT: absent external evidence is a terminal block."""
    output = tmp_path / "blocked.json"
    args = ["--fixture"] if fixture else []
    invoke(
        tmp_path,
        *args,
        "--upstream",
        str(tmp_path / "absent.json"),
        "--output",
        str(output),
        "--raw-root",
        str(tmp_path / "raw"),
        expected=2,
    )
    result = json.loads(output.read_text())
    assert result["verdict_class"] == "blocked" and result["experiment_id"] == 7916
    assert result["gate_check_summary"][0]["observed"] is None


def test_current_cli_owned_timeout_and_second_terminal_failure(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7916-CHECKPOINT: owned failures cannot claim runtime readiness."""
    upstream = prior.make_fixture(tmp_path / "input")
    bad_reader = tmp_path / "bad-reader.py"
    bad_reader.write_text("raise SystemExit(1)\n")
    for flag, value in (("--fit-deadline-s", "0"), ("--terminal-recheck", str(bad_reader))):
        output = tmp_path / f"{flag[2:]}.json"
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
        result = json.loads(output.read_text())
        assert (
            result["verdict_class"] == "disqualified"
            and result["training_runtime_ready_score"] == 0
        )
        sidecar = json.loads(Path(str(output) + ".validators.json").read_text())
        assert sidecar["candidate_sha256"] == prior.sha256_file(output)


def test_historical_missing_and_hash_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7916-SCOPE: old receipts must remain authenticated prerequisites."""
    historical = current.HISTORY
    monkeypatch.setattr(current, "HISTORY", tmp_path / "missing.json")
    with pytest.raises(prior.custody.InputBlocked, match="exists"):
        current.historical()
    monkeypatch.setattr(current, "HISTORY", historical)
    artifact = json.loads(historical.read_text())
    log = next(
        Path(row["log_path"]) for row in artifact["validation_receipts"] if not row["passed"]
    )
    real_hash = prior.sha256_file
    monkeypatch.setattr(
        prior, "sha256_file", lambda path: "sha256:changed" if path == log else real_hash(path)
    )
    with pytest.raises(prior.custody.InputBlocked, match="historical_required_log"):
        current.historical()


def test_current_producer_reuses_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7916-SCOPE: new producer dispatch stays a parameterized library call."""
    from scripts.experiments import experiment_7916_v687_training_qualification as cli

    q = current.runtime()
    private = tmp_path / "private"
    private.mkdir()
    fixture = q.make_fixture(private / "fixture-input")
    q.fit_score(fixture, private / "fixture.json", private / "fixture-raw", 7916, "20260930")
    prior.atomic_json(
        private / "coverage.json",
        {
            "files": {
                name: {"summary": {"num_statements": 1, "covered_lines": 1, "missing_lines": 0}}
                for name in current.OWNED
            }
        },
    )

    def successful(commands, raw, heartbeat_s=30, extra_env=None):
        return [
            {"name": row.name, "argv": list(row.argv), "passed": True, "actual_exit": 0}
            for row in commands
        ]

    real_qualify = q.qualify
    monkeypatch.setattr(q, "execute", successful)
    monkeypatch.setattr(q, "publish", lambda *args: True)
    monkeypatch.setattr(
        q,
        "qualify",
        lambda upstream, output, raw, date: real_qualify(upstream, output, raw, date, private),
    )
    monkeypatch.setattr(current, "runtime", lambda: q)
    output, raw = tmp_path / "output.json", tmp_path / "raw"
    assert (
        cli.main(
            [
                "--date",
                "20260930",
                "--upstream",
                str(fixture),
                "--output",
                str(output),
                "--raw-root",
                str(raw),
            ]
        )
        == 0
    )
    result = json.loads((raw / "qualified_candidate.json").read_text())
    assert result["experiment_id"] == 7916 and result["training_runtime_ready_score"] == 1
    assert result["historical_required_failures"][-1]["experiment_id"] == 7904


def test_changed_current_dependency_and_other_upstream(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7916-CHECKPOINT: new producer bytes and upstream path bind authority."""
    q = current.runtime()
    upstream = q.make_fixture(tmp_path / "input")
    identity = q.checkpoint_identity(upstream, "local_set", 67801, 1)
    clone = tmp_path / "code"
    for name in q.DEPENDENCIES:
        path = clone / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    q.ROOT = clone
    learning = clone / "python/carnot/verify/natural_training.py"
    learning.write_bytes(learning.read_bytes() + b"\n# Private changed dependency.\n")
    with pytest.raises(ValueError, match="checkpoint dependency mismatch"):
        q.check_compatible(identity, q.checkpoint_identity(upstream, "local_set", 67801, 1))
    q.ROOT = ROOT
    other = tmp_path / "other.json"
    other.write_bytes(upstream.read_bytes())
    with pytest.raises(ValueError, match="checkpoint dependency mismatch"):
        q.check_compatible(identity, q.checkpoint_identity(other, "local_set", 67801, 1))
    assert all(row["rejected"] for row in q.mutation_rows(identity))
