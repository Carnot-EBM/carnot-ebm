"""REQ-REPORT-7762: independent V675 evidence tests."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot import experiment_7762_v675_independent_evidence_audit as audit
from carnot.reporting import experiment_7303_validation_scope as checks


def static_fixture() -> tuple[list[dict], dict]:
    """Use two independent families with paired seeds and distinct erased input."""
    rows = []
    for family in ("f0", "f1"):
        for arm in ("constrained_set", "augmented_set", "source_erased"):
            for seed in (0, 1):
                for view in ("A", "B"):
                    p = 0.1 if arm == "constrained_set" else 0.2
                    rows.append(
                        {
                            "family_id": family,
                            "arm": arm,
                            "seed": seed,
                            "view": view,
                            "role": "evaluation64",
                            "label": 0,
                            "label_join": family,
                            "probability": p,
                            "action": "accept",
                            "source_sha256": f"sha256:{family}",
                            "input_hash": f"sha256:{family}-{arm}",
                            "prediction_tick": 1,
                            "label_tick": 2,
                            "censored": False,
                        }
                    )
    summary = {
        "expected_families": 2,
        "seeds": [0, 1],
        "views": ["A", "B"],
        "arms": ["constrained_set", "augmented_set", "source_erased"],
        "role": "evaluation64",
    }
    return rows, summary


def online_fixture() -> tuple[list[dict], list[dict], dict]:
    """Show past feedback, one admission and all static predicates."""
    rows = []
    events = []
    for arm in ("adaptive", "frozen", "complete_static", "shuffled"):
        rows.append(
            {
                "family_id": "g0",
                "arm": arm,
                "seed": 0,
                "role": "evaluation64",
                "label": 0,
                "label_join": "g0",
                "probability": 0.1,
                "action": "accept",
                "prediction_tick": 1,
                "feedback_tick": 2,
                "source_sha256": "sha256:g0",
                "censored": False,
            }
        )
        events.extend(
            [
                {"kind": "prediction", "arm": arm, "family_id": "g0", "tick": 1},
                {"kind": "feedback", "arm": arm, "family_id": "g0", "tick": 2},
            ]
        )
    events.extend(
        [
            {"kind": "admission", "feedback_id": "g0", "predicate": "p0", "tick": 3},
            {
                "kind": "later_prediction",
                "feedback_id": "g0",
                "tick": 4,
                "probability": 0.2,
                "erased_probability": 0.1,
            },
        ]
    )
    summary = {
        "expected_families": 1,
        "arms": ["adaptive", "frozen", "complete_static", "shuffled"],
        "static_dictionary": {f"p{i}": 0.1 for i in range(16)},
        "complete_static_predicates": [f"p{i}" for i in range(16)],
        "proposal_count": 1,
        "pending_high_water": 1,
        "restart_exact_parity": True,
        "retention_rows": [],
    }
    return rows, events, summary


def write_producers(root: Path) -> None:
    """Write only private, byte-hashed producers and their raw row files."""
    static_rows, static_summary = static_fixture()
    online_rows, events, online_summary = online_fixture()
    for number, relative in audit.PLAN.items():
        raw = root / f"raw-{number}.json"
        raw.write_text(
            json.dumps(
                {
                    "rows": static_rows if number == 7758 else online_rows,
                    "summary": static_summary if number == 7758 else online_summary,
                }
            )
        )
        producer = {
            "experiment_id": number,
            "milestone": "2026.09.675",
            "run_date": "20260927",
            "flagged_adversarial": False,
            "honest_verdict": "complete_null_valid",
            "verdict_class": "null",
            "raw_rows_path": raw.name,
            "raw_rows_sha256": audit.sha256_file(raw),
        }
        if number == 7761:
            event_path = root / "events-7761.json"
            event_path.write_text(json.dumps(events))
            producer["event_rows_path"] = event_path.name
            producer["event_rows_sha256"] = audit.sha256_file(event_path)
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(producer))


def test_missing_producers_block_independently(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7762-CUSTODY: separate paths cannot borrow a receipt."""
    sources, failures = audit.inspect_sources(tmp_path)
    assert {row["upstream_id"] for row in failures} == {"Exp7758", "Exp7761"}
    assert all(row["observed"] == "missing" for row in failures)
    assert all(row["sha256"] is None for row in sources)
    value = audit.build_artifact(tmp_path, "20260927", sources, failures, None, None)
    assert value["honest_verdict"] == "complete_blocked_required_v675_evidence"
    assert value["verdict_class"] == "blocked"
    assert value["independent_static_eligible"] is False
    assert value["independent_online_eligible"] is False
    assert value["acceptance_gate_results"]["decision_benefit"] is None
    assert value["sample_size_budget"]["effective_independent_n"] == 0
    receipt = tmp_path / "results/experiment_7758_pre_gate.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text("{}")
    sources, failures = audit.inspect_sources(tmp_path)
    assert sources[0]["pre_gate_receipt"]["role"] == "explanation_only"
    assert sources[0]["eligible"] is False
    assert len(failures) == 2


def test_static_reduction_and_corruptions() -> None:
    """SCENARIO-REPORT-7762-REDUCTION: private copies lose eligibility."""
    rows, summary = static_fixture()
    clean = audit.reduce_static(rows, summary)
    assert clean["failed_checks"] == []
    assert clean["effective_independent_n"] == 2
    assert clean["by_arm"]["constrained_set"]["brier"] == pytest.approx(0.01)
    for change, expected in (
        (lambda r, s: r[0].update(label_join="wrong"), "label_join"),
        (lambda r, s: r[0].update(role="train"), "role"),
        (lambda r, s: r[0].update(probability=1.2), "probability"),
        (lambda r, s: s.update(expected_families=3), "family_count"),
        (lambda r, s: r.pop(), "roster"),
        (lambda r, s: r[8].update(input_hash="sha256:f0-augmented_set"), "erased_boundary"),
    ):
        changed_rows, changed_summary = deepcopy(rows), deepcopy(summary)
        change(changed_rows, changed_summary)
        assert expected in audit.reduce_static(changed_rows, changed_summary)["failed_checks"]


def test_online_chronology_and_predicate_mutations() -> None:
    """SCENARIO-REPORT-7762-REDUCTION: causal order and closure are checked."""
    rows, events, summary = online_fixture()
    assert audit.reduce_online(rows, events, summary)["failed_checks"] == []
    for change, expected in (
        (lambda r, e, s: e[-1].update(tick=2), "update_chronology"),
        (lambda r, e, s: s.update(complete_static_predicates=["p0"]), "static_closure"),
        (lambda r, e, s: e.append(deepcopy(e[-2])), "one_use_admission"),
        (lambda r, e, s: r[0].update(feedback_tick=0), "feedback_chronology"),
    ):
        changed = deepcopy(rows), deepcopy(events), deepcopy(summary)
        change(*changed)
        assert expected in audit.reduce_online(*changed)["failed_checks"]


def test_private_cold_reader_and_child_basetemp(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7762-TERMINAL: exact bytes survive a fresh child."""
    base = tmp_path / "nested" / "pytest"
    base.parent.mkdir(parents=True)
    probe = subprocess.run(
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
    assert probe.returncode == 0 and probe.stdout.strip() == "True"
    sources, failures = audit.inspect_sources(tmp_path)
    value = audit.build_artifact(tmp_path, "20260927", sources, failures, None, None)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(value))
    assert audit.cold_replay(candidate) == []
    candidate.write_text(json.dumps({**value, "rows": []}))
    assert "rows_changed" in audit.cold_replay(candidate)


def test_eligible_branch_and_raw_hash_mutations(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7762-REDUCTION: raw bytes and branches stay separate."""
    write_producers(tmp_path)
    sources, failures = audit.inspect_sources(tmp_path)
    assert failures == [] and all(source["eligible"] for source in sources)
    reductions, raw_failures, hashes = audit.read_branches(tmp_path, sources)
    assert raw_failures == [] and len(hashes) == 3
    assert reductions[7758]["effective_independent_n"] == 2
    assert reductions[7761]["effective_independent_n"] == 1
    value = audit.build_artifact(
        tmp_path, "20260927", sources, failures, reductions[7758], reductions[7761], hashes
    )
    assert value["independent_static_eligible"] is True
    assert value["independent_online_eligible"] is False  # No retention measurement.
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(value))
    assert audit.cold_replay(candidate) == []
    raw = tmp_path / "raw-7758.json"
    raw.write_text(raw.read_text() + " ")
    changed_sources, _ = audit.inspect_sources(tmp_path)
    _, changed_failures, _ = audit.read_branches(tmp_path, changed_sources)
    assert any(f["field"] == "raw_rows_path_sha256" for f in changed_failures)
    assert audit.cold_replay(candidate)


def test_private_entrypoint_and_source_disqualification(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7762-CUSTODY: one bad producer leaves the other readable."""
    write_producers(tmp_path)
    producer = tmp_path / audit.PLAN[7758]
    value = json.loads(producer.read_text())
    value["verdict_class"] = "blocked"
    producer.write_text(json.dumps(value))
    sources, failures = audit.inspect_sources(tmp_path)
    assert sources[0]["state"] == "disqualified"
    assert sources[1]["eligible"] is True
    assert any(f["field"] == "verdict_class" for f in failures)
    reductions, _, _ = audit.read_branches(tmp_path, sources)
    assert reductions[7758] is None and reductions[7761] is not None
    output = tmp_path / "audit.json"
    assert audit.main(["--fixture-root", str(tmp_path), "--output", str(output)]) == 0
    assert output.is_file()
    assert audit.main(["--cold", str(output)]) == 0
    assert (
        audit.run_experiment(tmp_path, "20260927", output, validate=False)["verdict_class"]
        == "blocked"
    )


@pytest.mark.parametrize(
    "bad_stage", [None, "focused_pytest", "full_python_suite", "adversarial_verify"]
)
def test_validation_receipts_control_terminal_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_stage: str | None
) -> None:
    """SCENARIO-REPORT-7762-TERMINAL: failed owned checks close readiness."""

    def fake_commands(*args: object, **kwargs: object) -> list[checks.CommandSpec]:
        return [
            checks.CommandSpec(name, ("true",), "private") for name in checks.REQUIRED_CHECK_NAMES
        ]

    def fake_run(_root: Path, commands: list[checks.CommandSpec], **kwargs: object) -> list[dict]:
        return [
            {
                "name": command.name,
                "passed": command.name != bad_stage,
                "exit_code": int(command.name == bad_stage),
                "timed_out": False,
                "log_sha256": "sha256:private",
            }
            for command in commands
        ]

    monkeypatch.setattr(checks, "build_scoped_commands", fake_commands)
    monkeypatch.setattr(checks, "run_commands", fake_run)
    output = tmp_path / "result.json"
    result = audit.run_experiment(tmp_path, "20260927", output, validate=True)
    assert output.is_file()
    assert len(result["validation_receipts"]["terminal_readers"]) == 3
    if bad_stage in {None, "full_python_suite"}:
        assert result["honest_verdict"] == "complete_blocked_required_v675_evidence"
    else:
        assert result["verdict_class"] == "disqualified"
        assert result["acceptance_gate_results"]["readiness"] == 0
    assert result["flagged_adversarial"] is (bad_stage == "adversarial_verify")
    assert bool(result["validation_receipts"]["global_suite_debt"]) is (
        bad_stage == "full_python_suite"
    )


def test_malformed_producer_fields_and_raw_files(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7762-CUSTODY: malformed custody fails closed."""
    write_producers(tmp_path)
    producer = tmp_path / audit.PLAN[7758]
    producer.write_text("[]")
    sources, failures = audit.inspect_sources(tmp_path)
    assert any(f["field"] == "schema" for f in failures)
    producer.write_text("{")
    _, failures = audit.inspect_sources(tmp_path)
    assert any(f["field"] == "schema" for f in failures)
    value = json.loads((tmp_path / audit.PLAN[7761]).read_text())
    value.update(
        experiment_id=0,
        honest_verdict="running",
        raw_rows_path=None,
    )
    (tmp_path / audit.PLAN[7761]).write_text(json.dumps(value))
    _, failures = audit.inspect_sources(tmp_path)
    assert {"experiment_id", "honest_verdict", "raw_rows_path"} <= {f["field"] for f in failures}
    write_producers(tmp_path)
    source, _ = audit.inspect_sources(tmp_path)
    raw = tmp_path / "raw-7758.json"
    raw.unlink()
    _, failures, _ = audit.read_branches(tmp_path, source)
    assert any(f["field"] == "raw_rows_path" for f in failures)
    write_producers(tmp_path)
    source, _ = audit.inspect_sources(tmp_path)
    raw.write_text("{")
    _, failures, _ = audit.read_branches(tmp_path, source)
    assert any(f["field"] == "raw_schema" for f in failures)
    write_producers(tmp_path)
    payload = json.loads(raw.read_text())
    payload["rows"][0]["role"] = "train"
    raw.write_text(json.dumps(payload))
    value = json.loads(producer.read_text())
    value["raw_rows_sha256"] = audit.sha256_file(raw)
    producer.write_text(json.dumps(value))
    source, _ = audit.inspect_sources(tmp_path)
    _, failures, _ = audit.read_branches(tmp_path, source)
    assert any(f["field"] == "raw_rows" and f["observed"] == "role" for f in failures)
    write_producers(tmp_path)
    source, _ = audit.inspect_sources(tmp_path)
    raw.write_text(json.dumps({"rows": []}))
    value = json.loads(producer.read_text())
    value["raw_rows_sha256"] = audit.sha256_file(raw)
    producer.write_text(json.dumps(value))
    _, failures, _ = audit.read_branches(tmp_path, source)
    assert any(f["field"] == "raw_schema" for f in failures)


def test_additional_row_rejection_paths() -> None:
    """SCENARIO-REPORT-7762-REDUCTION: controls reject altered rows."""
    rows, summary = static_fixture()
    assert audit._paired([], 1)["n"] == 0
    assert audit._score({"probability": 0.8, "label": 1, "action": "reject"})[2] == 0
    assert audit._score({"probability": 0.8, "label": 1, "action": "accept"})[2] == 5
    assert audit._score({"probability": 0.8, "label": 1, "action": "escalate"})[2] == 0.25
    assert audit._score({"probability": 0.2, "label": 0, "action": "reject"})[2] == 1
    with pytest.raises(ValueError, match="action"):
        audit._score({"probability": 0.2, "label": 0, "action": "bad"})
    for change, expected in (
        (lambda r: r[0].update(label_tick=0), "label_chronology"),
        (lambda r: r[0].update(source_sha256="other"), "source_identity"),
        (lambda r: r[0].update(label=1), "label_join"),
    ):
        changed = deepcopy(rows)
        change(changed)
        assert expected in audit.reduce_static(changed, summary)["failed_checks"]
    online, events, online_summary = online_fixture()
    for mutate, expected in (
        (lambda r, e, s: r[0].update(label_join="wrong"), "label_join"),
        (lambda r, e, s: r[0].update(role="train"), "role"),
        (lambda r, e, s: r[0].update(probability=2), "probability"),
        (lambda r, e, s: s.update(expected_families=2), "family_count"),
        (lambda r, e, s: r.pop(), "roster"),
        (lambda r, e, s: r[0].update(source_sha256="other"), "source_identity"),
        (lambda r, e, s: s.update(proposal_count=9), "proposal_credits"),
        (lambda r, e, s: s.update(pending_high_water=13), "pending_capacity"),
        (lambda r, e, s: s.update(restart_exact_parity=False), "restart_parity"),
        (lambda r, e, s: s.update(retention_rows=[{"role": "train"}]), "retention_role"),
    ):
        changed = deepcopy(online), deepcopy(events), deepcopy(online_summary)
        mutate(*changed)
        assert expected in audit.reduce_online(*changed)["failed_checks"]


def test_corrupt_probability_still_publishes_disqualified_rows(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7762-REDUCTION: malformed rows remain in the record."""
    write_producers(tmp_path)
    raw = tmp_path / "raw-7758.json"
    payload = json.loads(raw.read_text())
    payload["rows"][0]["probability"] = 2.0
    raw.write_text(json.dumps(payload))
    producer = tmp_path / audit.PLAN[7758]
    value = json.loads(producer.read_text())
    value["raw_rows_sha256"] = audit.sha256_file(raw)
    producer.write_text(json.dumps(value))
    result = audit.run_experiment(tmp_path, "20260927", tmp_path / "audit.json", validate=False)
    assert result["independent_static_eligible"] is False
    assert any(row["metrics"] is None for row in result["rows"] if row["arm"] == "constrained_set")
