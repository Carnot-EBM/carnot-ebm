"""REQ-REPORT-7734 and REQ-CL-7734-AUDIT independent evidence checks."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot import experiment_7734_v673_independent_evidence_audit as audit


def fixture() -> tuple[list[dict], list[dict], dict]:
    """Build three paired arms with label release after every prediction."""
    primitives = [f"feature_{index}" for index in range(8)]
    dictionary = primitives + [
        f"{left}&{right}"
        for index, left in enumerate(primitives)
        for right in primitives[index + 1 :]
    ]
    rows = []
    events = []
    for role, unit in (("development", "a"), ("admission", "b"), ("retained", "c")):
        for arm in ("growth", "frozen", "complete_static"):
            rows.append(
                {
                    "unit_id": unit,
                    "role": role,
                    "arm": arm,
                    "source_sha256": f"source-{unit}",
                    "input_hash": f"input-{unit}",
                    "prediction_tick": 1,
                    "feedback_tick": 2,
                    "label": 1,
                    "probability": 0.8,
                    "base_probability": 0.5,
                    "brier": 0.04,
                    "censored": False,
                    "exclusions": [],
                }
            )
            events.extend(
                [
                    {"unit_id": unit, "arm": arm, "kind": "prediction", "tick": 1},
                    {"unit_id": unit, "arm": arm, "kind": "feedback_arrival", "tick": 2},
                ]
            )
    events.append({"unit_id": "b", "arm": "growth", "kind": "one_use_admission", "tick": 3})
    summary = {
        "exactly_once": True,
        "restart_exact_parity": True,
        "static_closure_complete": {
            "dictionary": dictionary,
            "weights": dict.fromkeys(dictionary, 0.1),
            "equality_check": True,
        },
        "decisions": [],
    }
    return rows, events, summary


def fixture_bank(summary: dict) -> dict:
    """Mirror the saved static-bank grammar without dynamic templates."""
    dictionary = summary["static_closure_complete"]["dictionary"]
    return {
        "config": {
            "grammar": {
                "primitives": dictionary[:8],
                "pairs": [name.split("&") for name in dictionary[8:]],
            }
        },
        "templates": [],
        "used_admissions": [],
        "proposal": None,
    }


def test_valid_raw_reduction_and_blocked_science(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7734-REDUCTION: count families once across arms."""
    rows, events, summary = fixture()
    reduced = audit.reduce_raw(rows, events, summary)
    assert reduced["failed_checks"] == []
    assert reduced["sample_size"]["effective_independent_families"] == 3
    assert reduced["by_arm"]["growth"]["brier"] == pytest.approx(0.04)
    plan = {7731: "missing.json", 7733: "missing2.json"}
    custody, hashes, failures = audit.inspect_sources(tmp_path, plan, {7731, 7733})
    artifact = audit.make_artifact(tmp_path, "20260927", custody, hashes, failures, reduced)
    assert artifact["honest_verdict"] == "complete_blocked_required_v673_evidence"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["independent_audit_complete_score"] == 1
    assert artifact["independent_static_eligible"] is False
    assert artifact["independent_online_eligible"] is False
    assert all(
        value is None
        for key, value in artifact["acceptance_gate_results"].items()
        if key != "validity"
    )


@pytest.mark.parametrize(
    ("change", "check"),
    [
        (lambda r, e, s: r[0].update(feedback_tick=0), "label_leakage"),
        (lambda r, e, s: r[0].update(source_sha256="swapped"), "source_identity"),
        (lambda r, e, s: r.pop(), "paired_input_support"),
        (lambda r, e, s: r[0].update(brier=0.9), "aggregate_contradiction"),
        (lambda r, e, s: e.append(deepcopy(e[-1])), "admission_one_use"),
        (lambda r, e, s: s.update(restart_exact_parity=False), "restart_replay"),
        (
            lambda r, e, s: s["static_closure_complete"].update(equality_check=False),
            "static_closure",
        ),
        (lambda r, e, s: s.update(exactly_once=False), "admission_one_use"),
        (lambda r, e, s: e.pop(1), "event_chronology"),
        (lambda r, e, s: r[0].update(probability=2.0), "corrected_confidence"),
        (lambda r, e, s: r[0].update(role="retention"), "retained_disjoint"),
        (lambda r, e, s: e[1].update(tick=0), "event_chronology"),
    ],
)
def test_private_mutations_are_rejected(change, check: str) -> None:
    """SCENARIO-CL-7734-MUTATIONS: each fault has a concrete failed gate."""
    rows, events, summary = fixture()
    change(rows, events, summary)
    assert check in {
        item["check"] for item in audit.reduce_raw(rows, events, summary)["failed_checks"]
    }


def test_flagged_and_null_source_custody(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7734-CUSTODY: null qualifies; flagged source does not."""
    path = tmp_path / "source.json"
    payload = {
        "honest_verdict": "complete_null",
        "verdict_class": "null",
        "flagged_adversarial": False,
    }
    path.write_text(json.dumps(payload))
    custody, hashes, failures = audit.inspect_sources(tmp_path, {7731: "source.json"}, {7731})
    assert not failures and custody[0]["state"] == "eligible"
    assert hashes["eligible_producers"]["source.json"].startswith("sha256:")
    payload["flagged_adversarial"] = True
    path.write_text(json.dumps(payload))
    custody, hashes, failures = audit.inspect_sources(tmp_path, {7731: "source.json"}, {7731})
    assert custody[0]["state"] == "flagged"
    assert hashes["flagged_historical_inputs"]["source.json"].startswith("sha256:")
    assert failures[0]["field"] == "flagged_adversarial"


def test_cold_replay_rejects_changed_source(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7734-CUSTODY: source bytes bind the result."""
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "honest_verdict": "complete_null",
                "verdict_class": "null",
                "flagged_adversarial": False,
            }
        )
    )
    plan = {7731: "source.json"}
    custody, hashes, failures = audit.inspect_sources(tmp_path, plan, {7731})
    artifact = audit.make_artifact(tmp_path, "20260927", custody, hashes, failures, None)
    artifact["source_plan"] = {"7731": "source.json"}
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact))
    assert audit.cold_replay(candidate) == []
    source.write_text(source.read_text() + " ")
    assert "source_artifact_hashes_changed" in audit.cold_replay(candidate)


def test_decision_cost_and_aggregate_checks() -> None:
    """SCENARIO-CL-7734-MUTATIONS: frozen forecasts govern admission."""
    rows, events, summary = fixture()
    summary["decisions"] = [
        {
            "frozen": [{"unit_id": "b", "base": 0.1, "candidate": 0.8}],
            "labels": [1],
            "mean_brier_reduction": 0.77,
            "base_false_accepts": 0,
            "candidate_false_accepts": 0,
            "accepted": True,
        }
    ]
    reduced = audit.reduce_raw(rows, events, summary)
    assert reduced["failed_checks"] == []
    summary["decisions"][0]["candidate_false_accepts"] = 1
    assert "false_accept_cost" in {
        x["check"] for x in audit.reduce_raw(rows, events, summary)["failed_checks"]
    }
    summary["decisions"][0]["candidate_false_accepts"] = 0
    summary["decisions"][0]["mean_brier_reduction"] = 0
    assert "decision_aggregate_contradiction" in {
        x["check"] for x in audit.reduce_raw(rows, events, summary)["failed_checks"]
    }
    summary["decisions"][0]["frozen"].append(deepcopy(summary["decisions"][0]["frozen"][0]))
    assert "admission_family_support" in {
        x["check"] for x in audit.reduce_raw(rows, events, summary)["failed_checks"]
    }
    summary["decisions"][0]["frozen"].pop()
    summary["decisions"][0]["base_false_accepts"] = 2
    assert "false_accept_cost" in {
        x["check"] for x in audit.reduce_raw(rows, events, summary)["failed_checks"]
    }


def test_static_bank_bytes_close_dictionary() -> None:
    """SCENARIO-CL-7734-MUTATIONS: saved bank is checked independently."""
    rows, events, summary = fixture()
    bank = fixture_bank(summary)
    assert audit.reduce_raw(rows, events, summary, bank)["failed_checks"] == []
    bank["templates"].append({"unexpected": True})
    assert "static_closure" in {
        x["check"] for x in audit.reduce_raw(rows, events, summary, bank)["failed_checks"]
    }


def test_corrupt_and_nonterminal_sources(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7734-CUSTODY: malformed bytes and pre-gates stay visible."""
    broken = tmp_path / "broken.json"
    broken.write_text("{")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"honest_verdict": "blocked_gate_check_failed"}))
    custody, hashes, failures = audit.inspect_sources(
        tmp_path, {7731: "broken.json", 7733: "receipt.json"}, {7731, 7733}
    )
    assert [row["state"] for row in custody] == ["pre_gate", "pre_gate"]
    assert len(hashes["pre_gate_receipts"]) == 2
    assert [row["field"] for row in failures] == ["json_object", "honest_verdict"]


def test_eligible_null_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7734-CUSTODY: complete null remains science eligible."""
    source = tmp_path / "null.json"
    source.write_text(
        json.dumps(
            {
                "honest_verdict": "complete_null",
                "verdict_class": "null",
                "flagged_adversarial": False,
            }
        )
    )
    custody, hashes, failures = audit.inspect_sources(
        tmp_path, {7731: "null.json", 7733: "null.json"}, {7731, 7733}
    )
    rows, events, summary = fixture()
    result = audit.make_artifact(
        tmp_path, "20260927", custody, hashes, failures, audit.reduce_raw(rows, events, summary)
    )
    assert result["verdict_class"] == "null"
    assert result["acceptance_gate_results"]["readiness"] is True
    assert result["independent_static_eligible"] and result["independent_online_eligible"]


@pytest.mark.parametrize(
    ("validation_ok", "reader_ok", "with_raw"),
    [(True, True, False), (False, True, False), (True, False, False), (True, True, True)],
)
def test_terminal_orchestration_with_private_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validation_ok: bool,
    reader_ok: bool,
    with_raw: bool,
) -> None:
    """SCENARIO-REPORT-7734-TERMINAL: publish real phase and receipt structure."""

    def fake_commands(*args, **kwargs):
        return [audit.checks.CommandSpec("focused_pytest", ("true",), "explicit_tests")]

    def fake_run(root, commands, **kwargs):
        names = [command.name for command in commands]
        return [
            {
                "name": name,
                "passed": reader_ok
                if name == "adversarial_verify"
                else validation_ok
                if name == "focused_pytest"
                else True,
                "log_sha256": f"sha256:{name}",
                "exit_code": 0,
                "command": name,
            }
            for name in names
        ]

    monkeypatch.setattr(audit.checks, "build_scoped_commands", fake_commands)
    monkeypatch.setattr(audit.checks, "run_commands", fake_run)
    monkeypatch.setattr(
        audit.checks,
        "reduce_required_checks",
        lambda rows: {"required_checks_passed": validation_ok},
    )
    if with_raw:
        rows, events, summary = fixture()
        rows[0]["brier"] = 0.9
        folder = tmp_path / "results/raw/experiment_7732_v673_causal_admission"
        folder.mkdir(parents=True)
        (folder / "rows.json").write_text(json.dumps(rows))
        (folder / "event_rows.json").write_text(json.dumps(events))
        (folder / "bank_complete_static.json").write_text(json.dumps(fixture_bank(summary)))
        producer = tmp_path / audit.PLAN[7732]
        producer.parent.mkdir(parents=True, exist_ok=True)
        producer.write_text(json.dumps(summary))
    output = tmp_path / "output.json"
    result = audit.run_experiment(tmp_path, "20260927", output)
    assert output.is_file()
    assert len(result["phase_spans"]) == 6
    assert result["validation_receipts"]["e2e_checks"][0]["passed"]
    assert result["flagged_adversarial"] is not reader_ok
    if not validation_ok or not reader_ok:
        assert result["verdict_class"] == "disqualified"
    else:
        assert result["verdict_class"] == "blocked"


def test_private_raw_replay_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7734-TERMINAL: cold CLI catches raw-row drift."""
    rows, events, summary = fixture()
    folder = tmp_path / "results/raw/experiment_7732_v673_causal_admission"
    folder.mkdir(parents=True)
    (folder / "rows.json").write_text(json.dumps(rows))
    (folder / "event_rows.json").write_text(json.dumps(events))
    producer = tmp_path / audit.PLAN[7732]
    producer.parent.mkdir(parents=True, exist_ok=True)
    producer.write_text(json.dumps(summary))
    plan = {7732: audit.PLAN[7732]}
    custody, hashes, failures = audit.inspect_sources(tmp_path, plan, set())
    artifact = audit.make_artifact(
        tmp_path, "20260927", custody, hashes, failures, audit.reduce_raw(rows, events, summary)
    )
    artifact["source_plan"] = {str(key): value for key, value in plan.items()}
    artifact["raw_input_hashes"] = {
        str(path.relative_to(tmp_path)): audit.sha256_file(path)
        for path in (folder / "rows.json", folder / "event_rows.json")
    }
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact))
    assert audit.cold_replay(candidate) == []
    assert audit.main(["--cold", str(candidate)]) == 0
    (folder / "rows.json").write_text((folder / "rows.json").read_text() + " ")
    assert any(error.startswith("raw_input_changed:") for error in audit.cold_replay(candidate))
    (folder / "rows.json").write_text(json.dumps(rows))
    artifact["recomputed_metrics"]["event_count"] = 0
    candidate.write_text(json.dumps(artifact))
    assert "raw_reduction_changed" in audit.cold_replay(candidate)
    assert audit.main(["--cold", str(candidate)]) == 1
    monkeypatch.setattr(audit, "run_experiment", lambda root, date, output: {})
    assert audit.main(["--date", "20260927", "--output", str(tmp_path / "unused")]) == 0
    monkeypatch.setattr(sys, "argv", ["audit", "--cold", str(candidate)])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module(
            "carnot.experiment_7734_v673_independent_evidence_audit", run_name="__main__"
        )
    assert stopped.value.code == 1
