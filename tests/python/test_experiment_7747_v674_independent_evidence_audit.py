"""REQ-REPORT-7747 and REQ-CL-7747-AUDIT private audit evidence."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from carnot import experiment_7747_v674_independent_evidence_audit as audit


def static_fixture() -> tuple[list[dict], dict]:
    """Give every arm the same independent family and five fixed seeds."""
    rows = []
    for arm in audit.STATIC_ARMS:
        for seed in range(5):
            rows.append(
                {
                    "family_id": "family-a",
                    "arm": arm,
                    "seed": seed,
                    "label": 1,
                    "probability": 0.8,
                    "action": "accept",
                    "brier": 0.04,
                    "nll": -__import__("math").log(0.8),
                    "cost": 0.0,
                    "source_sha256": "sha256:source",
                    "input_hash": "sha256:input",
                    "features_sha256": "sha256:features",
                    "annotation_sha256": "sha256:annotation",
                    "optimizer_steps": 20,
                    "temperature": 1.0,
                    "sentence_offsets_valid": True,
                    "prediction_before_label": True,
                    "censored": False,
                    "exclusions": [],
                }
            )
    return rows, {"pooled_brier": 0.04, "temperature": 1.0, "optimizer_steps": 20}


def online_fixture() -> tuple[list[dict], list[dict], dict]:
    """Use a past-only release and a fitted complete static control."""
    rows = []
    for arm in audit.ONLINE_ARMS:
        rows.append(
            {
                "family_id": "family-b",
                "arm": arm,
                "seed": 0,
                "label": 1,
                "probability": 0.8,
                "action": "accept",
                "brier": 0.04,
                "cost": 0.0,
                "source_sha256": "sha256:source-b",
                "input_hash": "sha256:input-b",
                "prediction_tick": 1,
                "feedback_tick": 2,
                "bank_version": 0,
                "censored": False,
                "exclusions": [],
            }
        )
    events = [
        {"kind": "prediction", "tick": 1, "family_id": "family-b", "arm": arm}
        for arm in audit.ONLINE_ARMS
    ]
    events += [
        {"kind": "feedback", "tick": 2, "family_id": "family-b", "arm": arm}
        for arm in audit.ONLINE_ARMS
    ]
    events.append(
        {
            "kind": "admission",
            "tick": 3,
            "feedback_id": "family-b",
            "predicate": "p0",
            "later_probability": 0.7,
            "erased_probability": 0.8,
        }
    )
    return (
        rows,
        events,
        {
            "static_dictionary": {f"p{x}": 0.1 for x in range(16)},
            "pending_high_water": 1,
            "proposal_count": 1,
            "restart_exact_parity": True,
            "consumed_exp7744_evaluation": False,
        },
    )


def test_static_family_reduction_and_mutations() -> None:
    """SCENARIO-REPORT-7747-REDUCTION: arithmetic and custody are checked."""
    rows, summary = static_fixture()
    reduced = audit.reduce_static(rows, summary)
    assert reduced["failed_checks"] == []
    assert reduced["effective_independent_n"] == 1
    assert reduced["by_arm"]["local_set"]["brier"] == pytest.approx(0.04)
    for mutate, check in [
        (lambda r, s: r.pop(), "arm_roster"),
        (lambda r, s: r[0].update(brier=0.9), "metric_arithmetic"),
        (lambda r, s: r[0].update(source_sha256="changed"), "source_identity"),
        (lambda r, s: r[0].update(sentence_offsets_valid=False), "sentence_offsets"),
        (lambda r, s: r[0].update(prediction_before_label=False), "label_leakage"),
        (
            lambda r, s: r[0].update(features_sha256="sha256:annotation"),
            "feature_annotation_separation",
        ),
        (lambda r, s: r[0].update(optimizer_steps=19), "optimizer_budget"),
        (lambda r, s: r[0].update(temperature=2), "temperature_selection"),
        (lambda r, s: s.update(pooled_brier=0), "pooled_metric"),
    ]:
        changed_rows, changed_summary = deepcopy(rows), deepcopy(summary)
        mutate(changed_rows, changed_summary)
        assert check in audit.reduce_static(changed_rows, changed_summary)["failed_checks"]


def test_online_causal_mutations() -> None:
    """SCENARIO-CL-7747-MUTATIONS: a structure change needs a forecast witness."""
    rows, events, summary = online_fixture()
    assert audit.reduce_online(rows, events, summary)["failed_checks"] == []
    for mutate, check in [
        (lambda r, e, s: r[0].update(feedback_tick=0), "future_feedback"),
        (lambda r, e, s: e.append(deepcopy(e[-1])), "one_use_admission"),
        (lambda r, e, s: s.update(pending_high_water=13), "pending_capacity"),
        (lambda r, e, s: s.update(proposal_count=9), "proposal_credits"),
        (lambda r, e, s: s.update(static_dictionary={}), "static_closure"),
        (lambda r, e, s: s.update(restart_exact_parity=False), "restart_parity"),
        (lambda r, e, s: s.update(consumed_exp7744_evaluation=True), "evaluation_isolation"),
        (lambda r, e, s: e[-1].update(erased_probability=0.7), "causal_erasure"),
        (
            lambda r, e, s: e.append(
                {"kind": "shuffled_label", "tick": 1, "origin_feedback_tick": 2}
            ),
            "future_origin_shuffled_label",
        ),
    ]:
        changed_rows, changed_events, changed_summary = (
            deepcopy(rows),
            deepcopy(events),
            deepcopy(summary),
        )
        mutate(changed_rows, changed_events, changed_summary)
        assert (
            check
            in audit.reduce_online(changed_rows, changed_events, changed_summary)["failed_checks"]
        )


def test_missing_required_sources_are_blocked(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7747-CUSTODY: absence is never measured as zero."""
    custody, hashes, failures = audit.inspect_sources(tmp_path)
    assert {x["upstream_id"] for x in failures} == {"Exp7744", "Exp7746"}
    assert all(x["op"] == "==" and x["observed"] == "missing" for x in failures)
    artifact = audit.make_artifact(tmp_path, "20260927", custody, hashes, failures, None, None, {})
    assert artifact["honest_verdict"] == "complete_blocked_required_v674_evidence"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["independent_audit_complete_score"] == 1
    assert artifact["independent_static_eligible"] is False
    assert artifact["independent_online_eligible"] is False
    assert artifact["acceptance_gate_results"]["decision_benefit"] is None
    assert artifact["sample_size_budget"]["effective_independent_n"] == 0


def test_private_sources_and_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7747-TERMINAL: bytes and row summary survive a cold read."""
    for number, relative in audit.PLAN.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "honest_verdict": "complete_null",
            "verdict_class": "null",
            "flagged_adversarial": False,
            "raw_rows_path": f"rows-{number}.json",
        }
        if number == 7746:
            payload["event_rows_path"] = "events-7746.json"
        path.write_text(json.dumps(payload))
    static_rows, static_summary = static_fixture()
    online_rows, online_events, online_summary = online_fixture()
    (tmp_path / "rows-7744.json").write_text(
        json.dumps({"rows": static_rows, "summary": static_summary})
    )
    (tmp_path / "rows-7746.json").write_text(
        json.dumps({"rows": online_rows, "summary": online_summary})
    )
    (tmp_path / "events-7746.json").write_text(json.dumps(online_events))
    private = audit.run_experiment(
        tmp_path, "20260927", tmp_path / "private-result.json", private_fixture=True
    )
    assert private["verdict_class"] == "circular_positive"
    assert private["verifier_is_oracle"] is True
    cli = (
        Path(audit.__file__).parents[2]
        / "scripts/experiments/experiment_7747_v674_independent_evidence_audit.py"
    )
    cli_result = tmp_path / "cli-result.json"
    assert (
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(cli),
                "--fixture-root",
                str(tmp_path),
                "--output",
                str(cli_result),
            ],
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )
    assert (
        subprocess.run(
            [sys.executable, "-u", str(cli), "--cold", str(cli_result)],
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )
    custody, hashes, failures = audit.inspect_sources(tmp_path)
    static, online, raw_hashes, raw_failures = audit.read_raw_sources(tmp_path, custody)
    assert not failures and not raw_failures
    assert static["failed_checks"] == online["failed_checks"] == []
    artifact = audit.make_artifact(
        tmp_path, "20260927", custody, hashes, failures, static, online, raw_hashes
    )
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact))
    assert audit.cold_replay(candidate) == []
    artifact["recomputed_static"]["pooled_brier"] = 0
    candidate.write_text(json.dumps(artifact))
    assert "static_reduction_changed" in audit.cold_replay(candidate)
    candidate.write_text(
        json.dumps(
            audit.make_artifact(
                tmp_path, "20260927", custody, hashes, failures, static, online, raw_hashes
            )
        )
    )
    (tmp_path / "rows-7744.json").write_text("{}")
    assert "raw_input_changed" in audit.cold_replay(candidate)


def test_additional_static_and_online_rejections() -> None:
    """SCENARIO-REPORT-7747-REDUCTION: invalid probabilities cannot enter means."""
    rows, summary = static_fixture()
    rows[0]["probability"] = 2
    assert "metric_arithmetic" in audit.reduce_static(rows, summary)["failed_checks"]
    rows, summary = static_fixture()
    rows[0]["action"] = "unknown"
    assert "metric_arithmetic" in audit.reduce_static(rows, summary)["failed_checks"]
    online, events, state = online_fixture()
    online[0]["probability"] = 2
    assert "metric_arithmetic" in audit.reduce_online(online, events, state)["failed_checks"]
    online, events, state = online_fixture()
    online[0]["cost"] = 5
    assert "metric_arithmetic" in audit.reduce_online(online, events, state)["failed_checks"]
    online, events, state = online_fixture()
    online.pop()
    assert "arm_roster" in audit.reduce_online(online, events, state)["failed_checks"]
    online, events, state = online_fixture()
    online[0]["source_sha256"] = "different"
    assert "source_identity" in audit.reduce_online(online, events, state)["failed_checks"]
    online, events, state = online_fixture()
    events.pop(0)
    assert "event_chronology" in audit.reduce_online(online, events, state)["failed_checks"]


def test_ineligible_and_broken_raw_contracts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7747-CUSTODY: a pre-gate receipt cannot become a source."""
    required = tmp_path / audit.PLAN[7744]
    required.parent.mkdir(parents=True)
    required.write_text("{")
    historical = tmp_path / audit.HISTORY[7734]
    historical.write_text(
        json.dumps(
            {
                "honest_verdict": "complete_null",
                "verdict_class": "null",
                "flagged_adversarial": False,
            }
        )
    )
    custody, hashes, failures = audit.inspect_sources(tmp_path)
    assert custody[0]["state"] == "ineligible"
    assert failures[0]["upstream_id"] == "Exp7744"
    assert hashes["pre_gate_receipts"][audit.PLAN[7744]]
    assert hashes["historical_disqualified_sources"][audit.HISTORY[7734]]
    required.write_text(
        json.dumps(
            {
                "honest_verdict": "complete_null",
                "verdict_class": "null",
                "flagged_adversarial": False,
            }
        )
    )
    custody, _, _ = audit.inspect_sources(tmp_path)
    _, _, _, failures = audit.read_raw_sources(tmp_path, custody)
    assert failures[0]["check"] == "raw_input_exists"
    required.write_text(
        json.dumps(
            {
                "honest_verdict": "complete_null",
                "verdict_class": "null",
                "flagged_adversarial": False,
                "raw_rows_path": "bad.json",
            }
        )
    )
    (tmp_path / "bad.json").write_text("{}")
    custody, _, _ = audit.inspect_sources(tmp_path)
    _, _, _, failures = audit.read_raw_sources(tmp_path, custody)
    assert failures[0]["check"] == "raw_schema"
    static_rows, static_summary = static_fixture()
    static_rows[0]["brier"] = 0
    (tmp_path / "bad.json").write_text(json.dumps({"rows": static_rows, "summary": static_summary}))
    _, _, _, failures = audit.read_raw_sources(tmp_path, custody)
    assert failures[0]["check"] == "metric_arithmetic"


def test_main_private_fixture_and_cold(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """SCENARIO-REPORT-7747-TERMINAL: the CLI writes and reopens its own bytes."""
    output = tmp_path / "result.json"
    assert audit.main(["--fixture-root", str(tmp_path), "--output", str(output)]) == 0
    assert output.is_file()
    assert audit.main(["--cold", str(output)]) == 0
    assert "cold_replay_errors" in capsys.readouterr().out
    value = json.loads(output.read_text())
    value["reproducibility_checksum"] = "changed"
    output.write_text(json.dumps(value))
    assert audit.main(["--cold", str(output)]) == 1
    audit.progress(time.monotonic(), "unit", "done", 1)


@pytest.mark.parametrize("failed_phase", [None, "affected", "terminal"])
def test_orchestration_records_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_phase: str | None
) -> None:
    """SCENARIO-REPORT-7747-TERMINAL: failed owned checks close validity."""
    from carnot.reporting import experiment_7303_validation_scope as checks

    def fake_run_commands(
        root: Path, commands: list, *, log_dir: Path, **_kwargs: object
    ) -> list[dict]:
        phase = log_dir.name
        return [
            {
                "name": command.name,
                "passed": phase != failed_phase,
                "exit_code": 0 if phase != failed_phase else 1,
                "timed_out": False,
                "log_sha256": "sha256:fixture",
            }
            for command in commands
        ]

    monkeypatch.setattr(checks, "run_commands", fake_run_commands)
    output = tmp_path / "out.json"
    value = audit.run_experiment(tmp_path, "20260927", output)
    assert output.is_file()
    assert len(value["validation_receipts"]["required_commands"]) == 8
    assert len(value["validation_receipts"]["terminal_readers"]) == 3
    if failed_phase == "terminal":
        assert value["honest_verdict"] == "complete_disqualified_terminal_reader"
        assert value["flagged_adversarial"] is True
    elif failed_phase == "affected":
        assert value["honest_verdict"] == "complete_disqualified_required_validation"
    else:
        assert value["honest_verdict"] == "complete_blocked_required_v674_evidence"
    assert all(span["duration_s"] >= 0 for span in value["phase_spans"])
