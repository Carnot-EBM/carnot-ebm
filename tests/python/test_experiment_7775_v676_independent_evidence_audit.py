"""REQ-REPORT-7775: independent V676 audit checks."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot import experiment_7775_v676_independent_evidence_audit as audit
from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting import experiment_7303_validation_scope as checks


def static_rows() -> tuple[list[dict], dict]:
    """Two families keep views and seeds paired inside each independent unit."""
    rows = []
    for family, label in (("f0", 0), ("f1", 1)):
        for arm, probability in (
            ("constrained_set", 0.2),
            ("source_erased", 0.5),
            ("constant", 0.5),
        ):
            for seed in (0, 1):
                rows.append(
                    {
                        "family_id": family,
                        "label_join": family,
                        "label": label,
                        "role": "evaluation64",
                        "arm": arm,
                        "seed": seed,
                        "probability": probability if label == 0 else 1 - probability,
                        "action": "accept" if label == 0 else "reject",
                        "prediction_tick": 1,
                        "label_tick": 2,
                        "source_sha256": family,
                        "input_hash": family + arm,
                        "censored": False,
                    }
                )
    return rows, {
        "families": 2,
        "arms": ["constrained_set", "source_erased", "constant"],
        "seeds": [0, 1],
        "role": "evaluation64",
    }


def online_rows() -> tuple[list[dict], list[dict], dict]:
    """One feedback item changes a later adaptive decision after admission."""
    rows = []
    events = []
    for arm in ("adaptive", "frozen", "complete_static", "shuffled"):
        rows.append(
            {
                "family_id": "g0",
                "label_join": "g0",
                "label": 0,
                "role": "evaluation64",
                "arm": arm,
                "seed": 0,
                "probability": 0.2,
                "action": "accept",
                "prediction_tick": 1,
                "feedback_tick": 2,
                "source_sha256": "g0",
                "censored": False,
            }
        )
        events.extend(
            [
                {"kind": "prediction", "arm": arm, "family_id": "g0", "tick": 1},
                {"kind": "feedback", "arm": arm, "family_id": "g0", "tick": 2},
            ]
        )
    events += [
        {"kind": "proposal", "block": 0, "predicate": "p0", "tick": 3},
        {"kind": "admission", "block": 0, "feedback_id": "g0", "predicate": "p0", "tick": 4},
        {
            "kind": "later_prediction",
            "feedback_id": "g0",
            "predicate": "p0",
            "tick": 5,
            "before_action": "reject",
            "after_action": "accept",
        },
        {"kind": "restart", "tick": 6, "exact_parity": True},
    ]
    return (
        rows,
        events,
        {
            "families": 1,
            "arms": ["adaptive", "frozen", "complete_static", "shuffled"],
            "seeds": [0],
            "static_predicates": ["p0"],
            "retention_rows": [],
        },
    )


def write_producers(root: Path) -> None:
    """Create current producer and raw bytes only inside a private fixture."""
    for number, relative in audit.PLAN.items():
        rows, summary = static_rows() if number == 7772 else online_rows()[::2]
        raw = root / f"raw-{number}.json"
        raw.write_text(json.dumps({"rows": rows, "summary": summary}))
        producer = {
            "experiment_id": number,
            "milestone": "2026.09.676",
            "run_date": "20260927",
            "flagged_adversarial": False,
            "honest_verdict": "complete_null_valid",
            "verdict_class": "null",
            "raw_rows_path": raw.name,
            "raw_rows_sha256": sha256_file(raw),
        }
        if number == 7774:
            event_path = root / "events-7774.json"
            event_path.write_text(json.dumps(online_rows()[1]))
            producer["event_rows_path"] = event_path.name
            producer["event_rows_sha256"] = sha256_file(event_path)
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(producer))


def test_missing_producers_and_distinct_receipt(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7775-CUSTODY: no receipt can supply science."""
    sources, failures = audit.inspect_sources(tmp_path)
    assert {f["upstream_id"] for f in failures} == {"Exp7772", "Exp7774"}
    assert all(f["observed"] == "missing" for f in failures)
    assert all(s["sha256"] is None for s in sources)
    receipt = tmp_path / audit.PRE_GATE[7772]
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}")
    sources, failures = audit.inspect_sources(tmp_path)
    assert sources[0]["pre_gate_receipt"]["role"] == "explanation_only"
    assert len(failures) == 2
    result = audit.build_artifact(tmp_path, "20260927", sources, failures, None, None)
    assert result["honest_verdict"] == "complete_blocked_required_v676_evidence"
    assert result["acceptance_gate_results"]["readiness"] == 0
    assert result["sample_size_budget"]["effective_independent_n"] == 0


def test_static_recomputes_and_rejects_corruption() -> None:
    """SCENARIO-REPORT-7775-REDUCTION: labels and paired units are checked."""
    rows, summary = static_rows()
    result = audit.reduce_static(rows, summary)
    assert result["failed_checks"] == []
    assert result["effective_independent_n"] == 2
    assert result["by_arm"]["constrained_set"]["brier"] == pytest.approx(0.04)
    assert result["source_dependence"]["brier_advantage"] == pytest.approx(0.21)
    assert result["paired_tests"]["source_erased_brier"]["n"] == 2
    for change, expected in (
        (lambda r, s: r[0].update(label_join="wrong"), "label_join"),
        (lambda r, s: r.pop(), "roster"),
        (lambda r, s: r[0].update(probability=2), "probability"),
        (lambda r, s: s.update(families=3), "family_count"),
    ):
        rr, ss = deepcopy(rows), deepcopy(summary)
        change(rr, ss)
        assert expected in audit.reduce_static(rr, ss)["failed_checks"]


def test_online_replays_admissions_and_restart() -> None:
    """SCENARIO-REPORT-7775-REDUCTION: delayed labels precede changed decisions."""
    rows, events, summary = online_rows()
    reduced = audit.reduce_online(rows, events, summary)
    assert reduced["failed_checks"] == []
    assert reduced["paired_tests"]["frozen_brier"]["n"] == 1
    for change, expected in (
        (lambda r, e, s: e.append(deepcopy(e[-3])), "one_use_admission"),
        (lambda r, e, s: e[-2].update(tick=2), "causal_change"),
        (lambda r, e, s: e[-1].update(exact_parity=False), "cold_restart"),
        (lambda r, e, s: s.update(complete_static_predicates=["p0", "p1"]), "static_closure"),
    ):
        rr, ee, ss = deepcopy(rows), deepcopy(events), deepcopy(summary)
        change(rr, ee, ss)
        assert expected in audit.reduce_online(rr, ee, ss)["failed_checks"]
    summary["retention_rows"] = [deepcopy(rows[0])]
    assert audit.reduce_online(rows, events, summary)["retention"]["brier"] == pytest.approx(0.04)
    summary["pending_high_water"] = 21
    assert "queue_capacity" in audit.reduce_online(rows, events, summary)["failed_checks"]
    assert "cold_restart" in audit.reduce_online(rows, events[:-1], summary)["failed_checks"]


def test_private_cold_replay_and_child_basetemp(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7775-TERMINAL: exact bytes and a real child are required."""
    base = tmp_path / "nested" / "pytest"
    base.parent.mkdir(parents=True)
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; p=Path(sys.argv[1]); p.mkdir(); print(p.is_dir())",
            str(base),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode == 0 and child.stdout.strip() == "True"
    sources, failures = audit.inspect_sources(tmp_path)
    value = audit.build_artifact(tmp_path, "20260927", sources, failures, None, None)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(value))
    assert audit.cold_replay(candidate) == []
    candidate.write_text(json.dumps({**value, "rows": []}))
    assert "rows_changed" in audit.cold_replay(candidate)


def test_present_producers_are_reduced_without_headlines(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7775-CUSTODY: exact raw bytes open both branches."""
    write_producers(tmp_path)
    sources, failures = audit.inspect_sources(tmp_path)
    assert failures == [] and all(s["eligible"] for s in sources)
    branches, raw_failures = audit.read_branches(tmp_path, sources)
    assert raw_failures == []
    assert branches[7772]["effective_independent_n"] == 2
    assert branches[7774]["causal_changes"] == 1
    value = audit.build_artifact(tmp_path, "20260927", sources, [], branches[7772], branches[7774])
    assert value["independent_static_eligible"] is True
    assert value["independent_online_eligible"] is False  # no retention rows
    assert len(value["rows"]) == 2 + len(static_rows()[0]) + len(online_rows()[0])
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(value))
    assert audit.cold_replay(candidate) == []


def test_custody_rejects_wrong_fields_and_changed_raw_bytes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7775-CUSTODY: each failed field keeps its operand."""
    write_producers(tmp_path)
    path = tmp_path / audit.PLAN[7772]
    value = json.loads(path.read_text())
    value.update(
        experiment_id=1,
        milestone="old",
        run_date="old",
        flagged_adversarial=True,
        honest_verdict="open",
        verdict_class="blocked",
    )
    path.write_text(json.dumps(value))
    (tmp_path / "raw-7774.json").write_text("{}")
    sources, failures = audit.inspect_sources(tmp_path)
    assert sources[0]["state"] == "disqualified"
    assert sources[1]["state"] == "disqualified"
    assert {f["field"] for f in failures} >= {
        "experiment_id",
        "milestone",
        "run_date",
        "flagged_adversarial",
        "honest_verdict",
        "verdict_class",
        "raw_rows_sha256",
    }
    assert all("artifact_hash" in f and "operator" in f for f in failures)
    value["raw_rows_path"] = "../outside.json"
    path.write_text(json.dumps(value))
    _, failures = audit.inspect_sources(tmp_path)
    assert any(f["field"] == "raw_rows_path" for f in failures)
    path.write_text("[]")
    _, failures = audit.inspect_sources(tmp_path)
    assert any(f["field"] == "schema" for f in failures)


def test_reducer_rejects_raw_schema_and_saved_metric(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7775-REDUCTION: malformed rows lose eligibility."""
    rows, summary = static_rows()
    rows[0]["brier"] = 0.9
    rows[0]["role"] = "fit"
    rows[0]["prediction_tick"] = 3
    rows[0]["input_hash"] = "f0source_erased"
    rows[0]["source_sha256"] = "other"
    assert {
        "saved_metric",
        "role",
        "label_chronology",
        "erased_boundary",
        "source_identity",
    } <= set(audit.reduce_static(rows, summary)["failed_checks"])
    write_producers(tmp_path)
    raw = tmp_path / "raw-7772.json"
    raw.write_text(json.dumps({"rows": []}))
    producer = tmp_path / audit.PLAN[7772]
    value = json.loads(producer.read_text())
    value["raw_rows_sha256"] = sha256_file(raw)
    producer.write_text(json.dumps(value))
    sources, failures = audit.inspect_sources(tmp_path)
    assert failures == []
    _, failures = audit.read_branches(tmp_path, sources)
    assert failures[0]["field"] == "raw_schema"


def test_validation_runner_records_all_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7775-TERMINAL: required failures close validity."""
    scope = tmp_path / audit.SCOPE_PATH
    scope.parent.mkdir(parents=True)
    scope.write_text(
        json.dumps(
            {
                "test_paths": [
                    "tests/python/test_experiment_7775_v676_independent_evidence_audit.py"
                ],
                "changed_modules": [
                    "python/carnot/experiment_7775_v676_independent_evidence_audit.py"
                ],
                "static_paths": [
                    "scripts/experiments/experiment_7775_v676_independent_evidence_audit.py"
                ],
            }
        )
    )
    mode = {"fail_required": False, "bad_report": False}

    def fake_commands(
        root: Path, commands: list[checks.CommandSpec], *, log_dir: Path, **_: object
    ) -> list[dict]:
        receipts = []
        log_dir.mkdir(parents=True, exist_ok=True)
        for command in commands:
            log = log_dir / f"{command.name}.log"
            if command.name == "adversarial_verify" and not mode["bad_report"]:
                log.write_text('{"flagged_count": 0}')
            else:
                log.write_text("ok")
            receipts.append(
                {
                    "name": command.name,
                    "command_argv": list(command.argv),
                    "passed": not (mode["fail_required"] and command.name == "focused_pytest"),
                    "exit_code": 0,
                    "log_path": str(log.relative_to(root)),
                    "log_sha256": sha256_file(log),
                }
            )
        return receipts

    monkeypatch.setattr(checks, "run_commands", fake_commands)
    output = tmp_path / "output.json"
    result = audit.run_experiment(tmp_path, "20260927", output)
    assert result["verdict_class"] == "blocked"
    assert len(result["validation_receipts"]["required_commands"]) == 8
    assert len(result["validation_receipts"]["terminal_readers"]) == 3
    assert output.is_file()
    mode["fail_required"] = True
    mode["bad_report"] = True
    result = audit.run_experiment(tmp_path, "20260927", output)
    assert result["verdict_class"] == "disqualified"
    assert result["flagged_adversarial"] is True
    assert result["acceptance_gate_results"]["validity"] is False


def test_main_fixture_and_cold_reader(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """SCENARIO-REPORT-7775-TERMINAL: thin CLI writes and reads private bytes."""
    output = tmp_path / "result.json"
    assert (
        audit.main(["--fixture-root", str(tmp_path), "--output", str(output), "--date", "20260927"])
        == 0
    )
    assert audit.main(["--cold", str(output)]) == 0
    assert "cold_replay_errors" in capsys.readouterr().out
    value = json.loads(output.read_text())
    value["rows"] = []
    output.write_text(json.dumps(value))
    assert audit.main(["--cold", str(output)]) == 1


def test_remaining_invalid_primitive_paths(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7775-REDUCTION: every named corruption has a check."""
    assert audit.paired([])["n"] == 0
    with pytest.raises(ValueError, match="action"):
        audit.score({"probability": 0.2, "label": 0, "action": "guess"})
    rows, summary = static_rows()
    rows[1]["label"] = 1
    rows[4]["probability"] = 0.6
    failed = audit.reduce_static(rows, summary)["failed_checks"]
    assert "label_join" in failed and "constant_risk" in failed

    rows, events, summary = online_rows()
    rows[0].update(label_join="wrong", role="fit", feedback_tick=0, probability=2)
    rows.pop()
    events.append(deepcopy(events[-4]))
    events[-4]["tick"] = 1  # admission occurs before feedback
    summary["families"] = 2
    failed = audit.reduce_online(rows, events, summary)["failed_checks"]
    assert {
        "label_join",
        "role",
        "feedback_chronology",
        "event_join",
        "probability",
        "family_count",
        "roster",
        "proposal_credits",
        "queue_order",
    } <= set(failed)

    write_producers(tmp_path)
    producer = tmp_path / audit.PLAN[7772]
    value = json.loads(producer.read_text())
    value["raw_rows_path"] = "missing.json"
    value["raw_rows_sha256"] = None
    producer.write_text(json.dumps(value))
    _, failures = audit.inspect_sources(tmp_path)
    assert any(f["observed"] == "missing" and f["field"] == "raw_rows_path" for f in failures)

    write_producers(tmp_path)
    raw = tmp_path / "raw-7772.json"
    value = json.loads(raw.read_text())
    value["rows"][0]["label_join"] = "wrong"
    raw.write_text(json.dumps(value))
    producer = tmp_path / audit.PLAN[7772]
    value = json.loads(producer.read_text())
    value["raw_rows_sha256"] = sha256_file(raw)
    producer.write_text(json.dumps(value))
    sources, failures = audit.inspect_sources(tmp_path)
    assert failures == []
    branches, failures = audit.read_branches(tmp_path, sources)
    assert branches[7772]["failed_checks"] == ["label_join"]
    assert failures[0]["observed"] == "label_join"
    bad = deepcopy(branches[7772])
    bad["rows"][0]["action"] = "guess"
    artifact = audit.build_artifact(tmp_path, "20260927", sources, failures, bad, branches[7774])
    assert artifact["rows"][2]["metrics"] is None
    raw.write_text("[]")
    value = json.loads(producer.read_text())
    value["raw_rows_sha256"] = sha256_file(raw)
    producer.write_text(json.dumps(value))
    sources, failures = audit.inspect_sources(tmp_path)
    assert failures == []
    _, failures = audit.read_branches(tmp_path, sources)
    assert failures[0]["observed"] == "ValueError"
