"""Current bounded qualification checks for REQ-VERIFY-7784."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_7784_v677_training_runtime as exp
from carnot import experiment_7760_v675_online_runner as online


def test_query_modes_hold_template_version(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7784-ONLINE: each query keeps one admitted bank version."""
    checks, _, names = online.preflight(exp.ROOT)
    assert all(row["passed"] for row in checks)
    rows = exp.exercise_query_modes(tmp_path, names)
    assert {row["mode"] for row in rows} == {"queued_commit", "between_query_commit"}
    assert len(rows) == 16
    assert all(row["template_hash_before"] == row["template_hash_after"] for row in rows)
    assert all(row["predictions_before_feedback"] for row in rows)
    assert all(row["bank_path"] and Path(row["bank_path"]).is_file() for row in rows)
    assert any(row["mode"] == "queued_commit" and row["pending_feedback"] for row in rows)


def test_terminal_record_preserves_prior_and_gates(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7784-TERMINAL: failed validation closes both scores."""
    prior = {
        "rows": [{"family": "fixture-0", "arm": "response_set", "excluded": False}],
        "fixture_training_rows": [{"arm": "response_set", "parameter_count": 300}],
        "training_protocol_path": str(tmp_path / "training.json"),
        "online_protocol_path": str(tmp_path / "online.json"),
        "online_fixture": {"valid": True},
        "acceptance_gate_results": {"validity": True},
        "source_artifact_hashes": [],
        "preconditions_checked": {},
        "sample_size_budget": {"effective_independent_n": 4},
        "raw_paths": {},
    }
    (tmp_path / "training.json").write_text("{}")
    (tmp_path / "online.json").write_text("{}")
    failure = {
        "upstream_id": "exp7784",
        "artifact_path": "log",
        "artifact_hash": "sha256:x",
        "field": "exit_code",
        "operator": "==",
        "expected": 0,
        "observed": 1,
    }
    value = exp.current_record(prior, "20260927", [failure], [], [], 12.5)
    assert value["experiment_id"] == 7784
    assert value["milestone"] == "2026.09.677"
    assert value["honest_verdict"] == "complete_disqualified_required_validation"
    assert value["training_runtime_ready_score"] == 0
    assert value["online_runtime_ready_score"] == 0
    assert value["gate_check_summary"] == [failure]
    assert value["rows"] == prior["rows"]
    assert value["MODEL_SPECS"] == value["model_specs"] == []
    assert value["acceptance_gate_results"]["decision_benefit"] is None
    assert value["historical_exp7769_verdict"] == "complete_disqualified_required_validation"
    clean = exp.current_record(prior, "20260927", [], [], [], 12.5)
    assert clean["honest_verdict"].startswith("complete_circular_positive")
    assert clean["training_runtime_ready_score"] == 1
    assert clean["online_runtime_ready_score"] == 1


def test_cold_reducer_rejects_changed_rows(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7784-TERMINAL: fresh raw bytes bind candidate rows."""
    rows = [{"family": "fixture-0", "arm": "response_set"}]
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(rows))
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"rows": rows, "raw_paths": {"rows": str(path)}}))
    assert exp.cold_reduce(candidate)["row_count"] == 1
    path.write_text("[]")
    with pytest.raises(ValueError, match="raw_rows_invalid"):
        exp.cold_reduce(candidate)


@pytest.mark.parametrize("bad", [False, True])
def test_run_gates_all_receipts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: bool) -> None:
    """SCENARIO-VERIFY-7784-TERMINAL: every child exit affects the current gate."""
    private = tmp_path / "private"
    private.mkdir()
    scope = private / "frozen_scope.json"
    scope.write_text(
        json.dumps({"direct_tests": list(exp.TESTS[:1]), "transitive_tests": list(exp.TESTS[1:])})
    )
    root = tmp_path / "root"
    (root / "results").mkdir(parents=True)
    (root / "results/experiment_7769_v676_training_qualification.json").write_text(
        json.dumps(
            {"run_date": "20260927", "honest_verdict": "complete_disqualified_required_validation"}
        )
    )
    row_path = private / "rows.json"
    row_path.write_text("[]")
    train_path = private / "train.json"
    online_path = private / "online.json"
    train_path.write_text("{}")
    online_path.write_text("{}")
    measured = {
        "rows": [],
        "raw_paths": {"rows": str(row_path)},
        "fixture_training_rows": [{"arm": str(i)} for i in range(9)],
        "training_protocol_path": str(train_path),
        "online_protocol_path": str(online_path),
        "online_fixture": {"valid": True},
        "random_seed": 67501,
        "phase_spans": [],
        "preconditions_checked": {},
        "source_artifact_hashes": [],
        "validation_receipts": {},
    }
    monkeypatch.setattr(exp, "ROOT", root)
    monkeypatch.setattr(exp, "PRIVATE", private)
    monkeypatch.setattr(exp, "OUTPUT", root / "results/current.json")
    monkeypatch.setattr(exp, "SCOPE", scope)
    monkeypatch.setattr(
        exp.online, "preflight", lambda path: ([{"passed": True}], {}, [str(i) for i in range(16)])
    )
    monkeypatch.setattr(exp.prior, "run_fixture", lambda folder, date: measured)
    monkeypatch.setattr(
        exp,
        "exercise_query_modes",
        lambda folder, names: [{"template_hash_before": "a", "template_hash_after": "a"}],
    )

    def fake_supervise(
        name: str, argv: list[str], log: Path, timeout_s: float, start: float
    ) -> dict:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text('{"flagged_count": 0}' if name == "adversarial_verify" else "ok")
        if name.startswith("coverage_shard_"):
            Path(
                next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--data-file="))
            ).write_text("coverage")
        return {
            "name": name,
            "command_argv": argv,
            "exit_code": int(bad and name == "coverage_report"),
            "passed": not (bad and name == "coverage_report"),
            "log_path": str(log),
            "log_sha256": exp.sha256_file(log),
        }

    monkeypatch.setattr(exp, "supervise", fake_supervise)
    value = exp.run("20260927")
    assert value["training_runtime_ready_score"] == int(not bad)
    assert value["online_runtime_ready_score"] == int(not bad)
    assert value["flagged_adversarial"] is False
    assert len(value["coverage_shard_rows"]) == 5
    assert value["validation_receipts"]["terminal_readers"]
    assert bool(value["gate_check_summary"]) == bad
    assert exp.OUTPUT.is_file()


def test_cli_dispatches_both_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7784-TERMINAL: the thin CLI reaches both handlers."""
    import importlib.util

    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/experiments/experiment_7784_v677_training_runtime.py"
    )
    spec = importlib.util.spec_from_file_location("exp7784_cli_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.experiment, "run", lambda date: {"date": date})
    monkeypatch.setattr(module.experiment, "cold_reduce", lambda path: {"candidate": str(path)})
    assert module.main(["--date", "20260927"]) == 0
    assert module.main(["--cold-reduce", str(tmp_path / "candidate.json")]) == 0


def test_supervised_child_records_exit_and_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7784-TERMINAL: only the owned timed-out child is killed."""
    import subprocess
    import sys

    good = exp.supervise(
        "child", [sys.executable, "-c", "print('ok')"], tmp_path / "child.log", 20, 0
    )
    assert good["passed"] and good["exit_code"] == 0
    assert "ok" in Path(good["log_path"]).read_text()

    class SlowChild:
        def __init__(self) -> None:
            self.calls = 0
            self.terminated = False
            self.killed = False

        def wait(self, timeout: float | None = None) -> int:
            self.calls += 1
            if self.calls < 3:
                raise subprocess.TimeoutExpired("owned", timeout)
            return -9

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    child = SlowChild()
    monkeypatch.setattr(exp.subprocess, "Popen", lambda *args, **kwargs: child)
    bad = exp.supervise("slow", ["owned"], tmp_path / "slow.log", 0, 0)
    assert child.terminated and child.killed
    assert bad["timed_out"] and not bad["passed"]


def test_external_block_is_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7784-TERMINAL: failed external custody stops before fitting."""
    private = tmp_path / "private"
    private.mkdir()
    (private / "frozen_scope.json").write_text("{}")
    monkeypatch.setattr(exp, "PRIVATE", private)
    monkeypatch.setattr(exp, "SCOPE", private / "frozen_scope.json")
    monkeypatch.setattr(exp, "OUTPUT", tmp_path / "blocked.json")
    bad = {
        "upstream_id": "7742",
        "artifact_path": "missing.json",
        "artifact_sha256": None,
        "field": "exists",
        "operator": "==",
        "expected": True,
        "observed": False,
        "passed": False,
    }
    monkeypatch.setattr(exp.online, "preflight", lambda root: ([bad], {}, []))

    def fake_supervise(
        name: str, argv: list[str], log: Path, timeout_s: float, start: float
    ) -> dict:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text('{"flagged_count": 0}')
        return {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "log_path": str(log),
            "log_sha256": exp.sha256_file(log),
        }

    monkeypatch.setattr(exp, "supervise", fake_supervise)
    value = exp.run("20260927")
    assert value["verdict_class"] == "blocked"
    assert value["honest_verdict"] == "complete_blocked_external_precondition"
    assert len(value["rows"]) == 54
    assert value["training_runtime_ready_score"] == 0
    assert value["gate_check_summary"][0]["observed"] is False


def test_full_reducer_and_script_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7784-TERMINAL: full reducer and __main__ dispatch execute."""
    import runpy
    import sys

    rows = [{"family": "fixture-0"}]
    raw = tmp_path / "rows.json"
    raw.write_text(json.dumps(rows))
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps({"rows": rows, "raw_paths": {"rows": str(raw)}, "fixture_training_rows": []})
    )
    calls: list[Path] = []
    monkeypatch.setattr(exp.prior, "cold_reduce", lambda path: calls.append(path))
    assert exp.cold_reduce(candidate)["row_count"] == 1
    assert calls == [candidate]
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/experiments/experiment_7784_v677_training_runtime.py"
    )
    monkeypatch.setattr(sys, "argv", [str(path), "--cold-reduce", str(candidate)])
    with pytest.raises(SystemExit) as finished:
        runpy.run_path(str(path), run_name="__main__")
    assert finished.value.code == 0
