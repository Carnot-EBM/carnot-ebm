"""REQ-REPORT-7796: qualify current source custody without changing history."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot import experiment_7796_v678_source_view_qualification as exp
from carnot.experiment_7727_v673_development_corpus import COUNTS
from carnot.verify import evidence_views
from test_experiment_7740_v674_sentence_label_protocol import _fixture


def test_scenario_report_7796_validation_scope_rejects_drift() -> None:
    """Every planned child has one frozen argv and only explicit test files."""
    scope = json.loads(exp.SCOPE.read_text())
    assert exp.validate_scope(scope) == scope["commands"]
    names = [row["name"] for row in scope["commands"]]
    assert len(names) == len(set(names))
    assert "full_python_suite" in names
    for name, target in (
        ("focused_pytest", "tests/python"),
        ("cold_replay", "experiment_7768_v676_source_view_qualification.py"),
        ("focused_pytest", "-k"),
    ):
        changed = copy.deepcopy(scope)
        row = next(row for row in changed["commands"] if row["name"] == name)
        row["argv"].append(target)
        with pytest.raises(ValueError, match="validation_scope_drift"):
            exp.validate_scope(changed)
    changed = copy.deepcopy(scope)
    changed["tests"].pop()
    with pytest.raises(ValueError, match="validation_scope_drift"):
        exp.validate_scope(changed)


def test_scenario_report_7796_custody_real_and_missing(tmp_path: Path) -> None:
    """The science producer and conductor receipt keep separate operands."""
    checks = exp.preconditions(tmp_path)
    failed = [row for row in checks if not row["passed"]]
    assert {row["upstream_id"] for row in failed} >= {
        "exp7727",
        "exp7753_conductor_pre_gate",
    }
    assert all(
        set(row) >= {"artifact_path", "artifact_hash", "field", "operator", "expected", "observed"}
        for row in failed
    )
    assert all(row["passed"] for row in exp.preconditions(exp.ROOT))


def test_scenario_report_7796_views_and_label_isolation(tmp_path: Path) -> None:
    """Public views keep both premises and evaluator labels stay private."""
    manifest = _fixture(tmp_path)
    counts = {role: 1 for role in COUNTS}
    raw = tmp_path / "raw"
    raw.mkdir()
    prepared = exp.prepare(manifest, raw, counts)
    assert len(prepared["rows"]) == 7
    assert exp.replay(manifest, raw, counts)["families"] == 7
    row = prepared["rows"][0]
    assert set(row) >= {"view_a", "view_b", "source_sha256", "response_sha256"}
    for key in ("view_a", "view_b"):
        view = row[key]
        assert view["source_bytes"]
        assert view["answer_bytes"]
        assert len(view["windows"]) <= 128
        assert len(view["answer_units"]) <= 16
        assert len(view["pair_features"][0]) == 132
    assert (
        len(
            evidence_views.location_prior(
                evidence_views.deserialize_pair({"a": row["view_a"]})["a"]
            )
        )
        == len(row["view_a"]["windows"]) + 1
    )
    evaluator = manifest.parent / "fit_evaluator.jsonl"
    label = json.loads(evaluator.read_text())
    label["annotations"][0]["implicit_true"] = True
    label["label"] = 0
    evaluator.write_text(json.dumps(label) + "\n")
    assert exp.prepare_public_only(manifest, counts)[0] == row
    with pytest.raises(ValueError, match="evaluator_sha256"):
        exp.replay(manifest, raw, counts)


def test_scenario_report_7796_over_budget_and_unknown() -> None:
    """No family disappears when windows exceed the common request budget."""
    pair = evidence_views.prepare_views(b"A. " * 65, b"Caf\xc3\xa9.")
    assert pair["a"]["abstention"] == "source_windows_over_budget"
    decisions = exp.over_budget_decisions(pair)
    assert set(decisions) == set(evidence_views.ARMS)
    assert all(
        row == {"risk": 0.5, "brier": 0.25, "cost": 0.25, "action": "escalate"}
        for row in decisions.values()
    )
    assert exp.map_targets(b"Caf\xc3\xa9.", None)["targets"] == [None]


def test_scenario_report_7796_orchestration_records_child_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each required child receives exactly the sealed command vector."""
    scope = json.loads(exp.SCOPE.read_text())
    seen: list[list[str]] = []

    def fake_child(name: str, argv: list[str], log_path: Path) -> dict:
        seen.append(argv)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(name)
        return {
            "name": name,
            "command_argv": argv,
            "exit_code": 0,
            "passed": True,
            "log_path": str(log_path),
            "log_sha256": exp.sha256_file(log_path),
        }

    monkeypatch.setattr(exp, "run_child", fake_child)
    receipts = exp.run_validation(scope, tmp_path)
    assert seen == [row["argv"] for row in scope["commands"]]
    assert all(row["passed"] for row in receipts)


def test_scenario_report_7796_child_has_durable_log(tmp_path: Path) -> None:
    """The actual subprocess helper stores argv, exit, and exact log bytes."""
    log = tmp_path / "child.log"
    receipt = exp.run_child("private_probe", [sys.executable, "-c", "print('owned')"], log)
    assert receipt["passed"] is True
    assert receipt["exit_code"] == 0
    assert receipt["command_argv"] == [sys.executable, "-c", "print('owned')"]
    assert log.read_text() == "owned\n"
    assert receipt["log_sha256"] == exp.sha256_file(log)


def test_scenario_report_7796_terminal_reduction_and_cold_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows keep independent families and failed checks close readiness."""
    manifest = _fixture(tmp_path)
    counts = {role: 1 for role in COUNTS}
    raw = tmp_path / "raw"
    raw.mkdir()
    prepared = exp.prepare(manifest, raw, counts)
    scope = json.loads(exp.SCOPE.read_text())
    receipts = [
        {
            "name": command["name"],
            "passed": True,
            "exit_code": 0,
            "log_path": str(tmp_path / command["name"]),
            "log_sha256": "sha256:mock",
        }
        for command in scope["commands"]
    ]
    ready = exp.build_candidate(exp.ROOT, [], prepared, receipts, [], raw=raw)
    assert ready["verdict_class"] == "circular_positive"
    assert ready["sentence_protocol_ready_score"] == 1
    assert ready["evidence_view_ready_score"] == 1
    assert ready["sample_size_budget"]["independent_n"] == 7
    assert len(ready["rows"]) == 7
    assert all(row["annotation_status"] for row in ready["rows"])
    failed = copy.deepcopy(receipts)
    failed[0]["passed"] = False
    failed[0]["exit_code"] = 1
    (tmp_path / failed[0]["name"]).write_text("failure")
    candidate = exp.build_candidate(exp.ROOT, [], prepared, failed, [], raw=raw, flagged=True)
    assert candidate["verdict_class"] == "disqualified"
    assert candidate["source_view_manifest_path"] is None
    assert candidate["sentence_protocol_ready_score"] == 0
    assert {row["field"] for row in candidate["gate_check_summary"]} == {
        "worktree_imports.exit_code",
        "flagged_adversarial",
    }
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(ready))
    monkeypatch.setattr(exp, "MANIFEST", manifest)
    monkeypatch.setattr(exp.corpus, "COUNTS", counts)
    assert exp.cold_reduce(path, raw)["families"] == 7
    ready["rows"].pop()
    path.write_text(json.dumps(ready))
    with pytest.raises(ValueError, match="candidate_roster_mismatch"):
        exp.cold_reduce(path, raw)


def test_scenario_report_7796_blocked_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent external producer yields a terminal blocked file before work."""
    raw = tmp_path / "raw"
    output = tmp_path / "blocked.json"
    monkeypatch.setattr(exp, "RAW", raw)
    monkeypatch.setattr(exp, "OUTPUT", output)
    monkeypatch.setattr(
        exp,
        "preconditions",
        lambda root: [exp._check("exp7727", tmp_path / "missing", "is_file", True, False)],
    )
    with pytest.raises(ValueError, match="run_date_mismatch"):
        exp.run_experiment("20260927")
    blocked = exp.run_experiment("20260928")
    assert output.is_file()
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"].startswith("complete_blocked_")
    assert blocked["gate_check_summary"][0]["artifact_path"] == str(tmp_path / "missing")
    assert blocked["source_view_manifest_path"] is None
    assert blocked["sentence_protocol_ready_score"] == 0


def test_scenario_report_7796_rejection_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Private changes to authority and roster cannot pass by renaming a check."""
    scope = json.loads(exp.SCOPE.read_text())
    monkeypatch.setattr(exp, "SCOPE_SHA256", "sha256:changed")
    with pytest.raises(ValueError, match="validation_scope_drift"):
        exp.validate_scope(scope)
    monkeypatch.undo()
    original = exp.build_scoped_commands
    monkeypatch.setattr(exp, "build_scoped_commands", lambda *a, **k: [])
    with pytest.raises(ValueError, match="validation_scope_drift"):
        exp.validate_scope(scope)
    monkeypatch.setattr(
        exp,
        "build_scoped_commands",
        lambda *a, **k: [exp.CommandSpec("changed", ("changed",), "private")],
    )
    with pytest.raises(ValueError, match="validation_scope_drift"):
        exp.validate_scope(scope)
    monkeypatch.setattr(exp, "build_scoped_commands", lambda *a, **k: original(*a, **k)[:-1])
    with pytest.raises(ValueError, match="validation_scope_drift"):
        exp.validate_scope(scope)
    monkeypatch.undo()
    manifest = _fixture(tmp_path)
    counts = {role: 1 for role in COUNTS}
    counts["fit"] = 2
    with pytest.raises(ValueError, match="public_role_hash_or_count"):
        exp.prepare_public_only(manifest, counts)
    with pytest.raises(ValueError, match="view_is_within_budget"):
        exp.over_budget_decisions(evidence_views.prepare_views(b"A.", b"B."))
    original_reduce = exp.corpus.cold_reduce
    monkeypatch.setattr(
        exp.corpus, "cold_reduce", lambda *a: (_ for _ in ()).throw(ValueError("bad"))
    )
    checks = exp.preconditions(exp.ROOT)
    assert any(
        row["field"] == "cold_reduce" and row["observed"] == "ValueError:bad" for row in checks
    )
    monkeypatch.setattr(exp.corpus, "cold_reduce", original_reduce)


def test_scenario_report_7796_success_entrypoint_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The task entrypoint publishes only after the recorded children return."""
    monkeypatch.setattr(exp, "RAW", tmp_path / "raw")
    monkeypatch.setattr(exp, "OUTPUT", tmp_path / "result.json")
    monkeypatch.setattr(exp, "CANDIDATE", tmp_path / "candidate.json")
    monkeypatch.setattr(exp, "preconditions", lambda root: [{"passed": True}])
    monkeypatch.setattr(exp, "prepare", lambda *a: {"rows": [1]})
    monkeypatch.setattr(
        exp,
        "build_candidate",
        lambda *a, **k: {
            "rows": [1],
            "verdict_class": "circular_positive",
            "flagged_adversarial": k.get("flagged", False),
        },
    )
    log = tmp_path / "adversarial.json"
    log.write_text('{"flagged_count": 0}')
    monkeypatch.setattr(
        exp,
        "run_validation",
        lambda *a, **k: [{"name": "adversarial_verify", "log_path": str(log), "duration_s": 0.1}],
    )
    result = exp.run_experiment("20260928")
    assert result["flagged_adversarial"] is False
    assert json.loads((tmp_path / "result.json").read_text()) == result
    log.write_text("malformed")
    assert exp.run_experiment("20260928")["flagged_adversarial"] is True
    monkeypatch.setattr(exp, "run_experiment", lambda date: {"verdict_class": "disqualified"})
    assert exp.main(["--date", "20260928"]) == 1
    monkeypatch.setattr(exp, "cold_reduce", lambda path: {"families": 640})
    assert exp.main(["--date", "20260928", "--cold-replay", str(tmp_path)]) == 0
    monkeypatch.setattr(exp, "main", lambda: 0)
    with pytest.raises(SystemExit) as wrapped:
        runpy.run_path(
            str(exp.ROOT / "scripts/experiments/experiment_7796_v678_source_view_qualification.py"),
            run_name="__main__",
        )
    assert wrapped.value.code == 0


def test_scenario_report_7796_reuses_only_failed_hash_bound_broad_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An old failed broad run stays a diagnostic while readers see fresh bytes."""
    scope = json.loads(exp.SCOPE.read_text())
    full = next(row for row in scope["commands"] if row["name"] == "full_python_suite")
    log = tmp_path / "prior.log"
    log.write_text("timed out")
    prior = {
        "name": "full_python_suite",
        "command_argv": full["argv"],
        "passed": False,
        "exit_code": -15,
        "log_path": str(log),
        "log_sha256": exp.sha256_file(log),
    }
    seen = []

    def fake_child(name: str, argv: list[str], log_path: Path) -> dict:
        seen.append(name)
        return {"name": name, "command_argv": argv, "passed": True, "exit_code": 0}

    monkeypatch.setattr(exp, "run_child", fake_child)
    receipts = exp.run_validation(
        scope,
        tmp_path,
        before_readers=lambda: seen.append("candidate_refreshed"),
        prior_broad=prior,
    )
    assert "full_python_suite" not in seen
    assert seen.index("candidate_refreshed") < seen.index("cold_replay")
    assert (
        next(row for row in receipts if row["name"] == "full_python_suite")[
            "reused_prior_diagnostic"
        ]
        is True
    )
    log.write_text("changed")
    seen.clear()
    exp.run_validation(scope, tmp_path, prior_broad=prior)
    assert "full_python_suite" in seen
