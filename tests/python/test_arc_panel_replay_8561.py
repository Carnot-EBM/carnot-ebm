"""REQ-REPORT-8561, REQ-VERIFY-8561: genuine child receipts own readiness."""

import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import coverage
import pytest

from carnot.reporting import arc_panel_replay_8548 as old
from carnot.reporting import arc_panel_replay_runner_8548 as runner
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


@pytest.mark.parametrize(
    "module,predicates",
    [
        (old, {"original_authority_invalid", "history_mismatch", "receipt_mismatch"}),
        (runner, {"resource_gate_failed"}),
    ],
)
def test_owned_branches_are_named_executable_predicates(module, predicates):
    """SCENARIO-VERIFY-8561-CHILDREN: branch anchors must have real line events."""
    tree = ast.parse(Path(module.__file__).read_text())
    branches = {
        n.test.id for n in ast.walk(tree) if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
    }
    assert predicates <= branches


def test_historical_actual_sets_and_source_pins(tmp_path, monkeypatch):
    """REQ-REPORT-8561: recompute all four differences from immutable actual data."""
    from carnot.reporting import arc_panel_replay_8561 as e

    receipt = e.historical_receipt()
    assert receipt["passed"]
    assert sorted(receipt["statement_differences"].values()) == [[], [78, 370, 377], [384]]
    assert receipt["original_terminal_replay"]["duration_s"] == 134.137591152
    assert receipt["original_terminal_replay"]["deadline_s"] == 240
    assert receipt["original_union"]["passed"]
    monkeypatch.setattr(e, "UPSTREAM_PIN", "sha256:bad")
    with pytest.raises(ValueError):
        e.historical_receipt()


def test_sysmon_real_child_discrepancy_and_repair(tmp_path):
    """REQ-VERIFY-8561: same collector and genuine child distinguish both forms."""
    from carnot.reporting import arc_panel_replay_8561 as e

    receipt = e.trace_reproduction(tmp_path)
    assert receipt["passed"]
    assert receipt["before"]["statement_difference"]
    assert receipt["after"]["statement_difference"] == []
    assert all(receipt[k]["child"]["exit_code"] == 0 for k in ("before", "after"))


def test_build_identity_and_owned_failure(tmp_path):
    """SCENARIO-REPORT-8561-REPLAY: missing and failed evidence cannot be ready."""
    from carnot.reporting import arc_panel_replay_8561 as e

    with e.adapters():
        work = runner.empty_work(tmp_path)
        path = tmp_path / "work.json"
        atomic_json(path, work)
        output = tmp_path / (e.NAME + ".json")
        value = old.build(path, output)
        assert (value["experiment_id"], value["run_date"]) == (8561, "20261016")
        assert value["verdict_class"] == "blocked" and old.replay(value)
        assert value["arc_panel_execution_ready_score"] == 0
        assert value["prior_invocation_diagnostics"]["verdict_class"] == "disqualified"
        assert value["prior_invocation_diagnostics"]["failed_checks"] == ["exact_owned_coverage"]
        work["checks"].append(dict(name="owned_failure", passed=False))
        atomic_json(path, work)
        assert old.build(path, output)["verdict_class"] == "disqualified"
        changed = deepcopy(value)
        changed["reproducibility_checksum"] = old.canonical_hash(changed)
        assert not old.replay(changed)


def cli(tmp_path, *args):
    """Run outside the checkout so the thin entry point must find its own imports."""
    from carnot.reporting import arc_panel_replay_8561 as e

    argv = [sys.executable, "-u", str(old.ROOT / e.CLI), *map(str, args)]
    measured = coverage.Coverage.current()
    if measured is not None:
        argv[1:2] = ["-m", "coverage", "run", "--rcfile=" + str(measured.config.config_file)]
    return subprocess.run(
        argv,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_actual_private_cli_and_consumer(tmp_path):
    """REQ-REPORT-8561: both terminal classes cross unchanged validators/readers."""
    from carnot.reporting import arc_panel_replay_8561 as e

    for failed in (False, True):
        output = tmp_path / str(failed) / (e.NAME + ".json")
        args = ["--date", "20261016", "--private-control", "--output", output]
        result = cli(tmp_path, *args, *(["--owned-failure"] if failed else []))
        assert result.returncode == 0, result.stdout + result.stderr
        value = json.loads(output.read_bytes())
        assert value["verdict_class"] == ("disqualified" if failed else "blocked")
        assert cli(tmp_path, "--cold-replay", output).returncode == 0
        side = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
        assert side["passed"] and side["consumer"]["passed"]
        assert side["consumer"]["gate_sha256"] == sha256_file(output)
        value["completed_count"] = 999
        atomic_json(output, value)
        assert cli(tmp_path, "--cold-replay", output).returncode == 1
    assert cli(tmp_path, "--date", "20261015").returncode == 2
    assert cli(tmp_path, "--cold-replay", tmp_path / "absent").returncode == 1


@pytest.mark.parametrize("ready", [0, 1])
def test_actual_future_panel_gate_reads_checked_bytes(tmp_path, ready):
    """SCENARIO-REPORT-8561-REPLAY: the future panel's exact gate controls readiness."""
    from carnot.reporting import arc_panel_replay_8561 as e

    candidate = tmp_path / (e.NAME + ".json")
    atomic_json(candidate, dict(task_id=e.TASK, arc_panel_execution_ready_score=ready))
    receipt = e.panel_consumer(candidate)
    assert receipt["accepted"] is bool(ready)
    assert receipt["matched_expectation"]
    assert receipt["primary_sha256"] == sha256_file(candidate)


@pytest.mark.parametrize("accepted", [False, True, None])
def test_measurement_failure_and_replay_adapters(tmp_path, monkeypatch, accepted):
    """REQ-VERIFY-8561: unavailable inputs and failed owned children stay explicit."""
    from carnot.reporting import arc_panel_replay_8561 as e

    def history():
        if accepted is None:
            raise ValueError("absent_original")
        return dict(passed=True)

    monkeypatch.setattr(e, "historical_receipt", history)
    monkeypatch.setattr(e, "trace_reproduction", lambda _: dict(passed=True))
    monkeypatch.setattr(old, "authenticate", lambda _: ({"historical_model_provenance": {}}, {}))
    monkeypatch.setattr(old, "history_receipts", lambda _: {})
    monkeypatch.setattr(old, "episode_hashes", lambda _: {})
    monkeypatch.setattr(old, "private_history", lambda *_: tmp_path / "candidate")

    def execute(name, argv, *_a, **_k):
        if name in ("primitive_replay", "original_fixtures") and accepted:
            atomic_json(Path(argv[-1]), [] if name == "primitive_replay" else {})
        return dict(name=name, passed=accepted)

    monkeypatch.setattr(runner, "invoke", execute)
    with e.adapters():
        work = runner.empty_work(tmp_path)
        e.measurement(work, old.ROOT, tmp_path, tmp_path)
    if accepted is None:
        assert work["failures"][0]["observed"] == "absent_original"
    else:
        assert work["checks"][-1]["passed"] is accepted
        assert work["terminal_replays"][0]["name"] == "original_terminal_replay"


def test_history_and_consumer_reject_changed_independent_operands(tmp_path, monkeypatch):
    """REQ-VERIFY-8561: repaired envelopes cannot replace actual SQL or panel gates."""
    from carnot.reporting import arc_panel_replay_8561 as e

    with monkeypatch.context() as patcher:
        patcher.setattr(e.coverage.CoverageData, "lines", lambda *_: [])
        with pytest.raises(ValueError, match="historical_statement_operands"):
            e.historical_receipt()
    with monkeypatch.context() as patcher:
        patcher.setattr(
            e.yaml,
            "safe_load",
            lambda _: dict(tasks=[dict(id="exp8562-arc-capability-panel", gated_on=[])]),
        )
        with pytest.raises(ValueError, match="future_panel_gate_drift"):
            e.panel_consumer(tmp_path / "unused")


def test_new_adapter_has_no_multiline_branch_anchor():
    """REQ-VERIFY-8561: new adapters must also emit their declared branch anchors."""
    from carnot.reporting import arc_panel_replay_8561 as e

    tree = ast.parse(Path(e.__file__).read_text())
    assert not [
        n.lineno for n in ast.walk(tree) if isinstance(n, ast.If) and n.lineno != n.test.lineno
    ]


def test_new_adapter_context_entries_have_executable_anchors():
    """REQ-VERIFY-8561: context entry lines also need actual sysmon receipts."""
    from carnot.reporting import arc_panel_replay_8561 as e

    tree = ast.parse(Path(e.__file__).read_text())
    assert not [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.With) and n.lineno != n.items[0].context_expr.lineno
    ]
