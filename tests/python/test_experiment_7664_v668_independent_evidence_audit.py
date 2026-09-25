"""REQ-REPORT-7664: independent evidence and terminal custody."""

from copy import deepcopy
import json
import runpy
import sys

import pytest

from carnot import experiment_7664_v668_independent_evidence_audit as audit
from carnot.reporting import independent_evidence_audit as reducer


def fixture_inputs():
    """SCENARIO-REPORT-7664-REDUCTION: tiny paired source corpus."""
    source = {
        "unit_id": "g1",
        "arm": "original_source",
        "source_sha256": "sha256:one",
        "original_source_sha256": "sha256:one",
        "denominator": 2,
        "checked_structural_propositions": 1,
        "unknown_claims": 1,
        "whole_answer_certified": False,
        "historically_exposed": True,
        "censored": True,
        "excluded": False,
        "role": "online",
    }
    online = []
    for arm, p, cost in (("source", 0.2, 0.2), ("scalar", 0.3, 0.2), ("frozen", 0.4, 0.2)):
        online.append(
            {
                "unit_id": "g1",
                "arm": arm,
                "probability": p,
                "label": 0,
                "brier": p * p,
                "decision_cost": cost,
                "typed_action": "escalate",
                "origin_ordinal": 0,
                "label_release_ordinal": 2,
                "released_label": 0,
                "label_from": "g1",
                "censored": True,
                "excluded": False,
                "raw_metrics": {"unknown_claims": 1},
            }
        )
    return {
        "features": [source],
        "evaluation": [],
        "delayed": online,
        "delayed_events": [
            {
                "event_id": "g1",
                "origin_ordinal": 0,
                "release_ordinal": 2,
                "next_state_hash": "sha256:state",
                "acknowledgment": {"state_hash": "sha256:state", "durable": True},
            }
        ],
        "continuous": deepcopy(online),
        "feedback": [
            {
                "unit_id": "g1",
                "arm": arm,
                "origin_ordinal": 0,
                "label_release_ordinal": 2,
                "released_label": 0,
                "label_from": "g1",
                "acknowledgment": {"acknowledged": True, "durable": True},
            }
            for arm in ("source", "scalar", "frozen")
        ],
        "admissions": [{"arm": "source", "admission_ids": ["g1"], "accepted": True}],
        "retention": [
            {
                "unit_id": "r1",
                "role": "fit",
                "label": 1,
                "final_probability": 0.6,
                "frozen_probability": 0.5,
                "final_brier": 0.16,
                "frozen_brier": 0.25,
                "censored": True,
                "excluded": False,
            }
        ],
    }


def test_independent_reduction_and_unknown_denominator():
    """SCENARIO-REPORT-7664-REDUCTION: paired arms never increase N."""
    result = audit.reduce_evidence(fixture_inputs())
    assert result["coverage"]["independent_groups"] == 1
    assert result["coverage"]["unknown_claims"] == 1
    assert result["continuous"]["effective_groups"] == 1
    assert result["continuous"]["arms"]["source"]["brier"] == pytest.approx(0.04)
    assert result["retention"]["fit"]["brier_improvement"] == pytest.approx(0.09)


@pytest.mark.parametrize(
    "mutation",
    [
        "source_hash",
        "future_label",
        "admission_reuse",
        "arm_metric",
        "unknown_removed",
        "fixture_truth",
    ],
)
def test_private_mutations_fail_closed(mutation):
    """SCENARIO-REPORT-7664-CUSTODY: six private corruptions fail validation."""
    inputs = fixture_inputs()
    audit.mutate(inputs, mutation)
    with pytest.raises(ValueError):
        audit.reduce_evidence(inputs)


def test_missing_producer_has_exact_block_operand(tmp_path):
    """SCENARIO-REPORT-7664-CUSTODY: absence is blocked with exact field."""
    found, checks, _ = audit.authenticate_inputs(tmp_path)
    assert found == {"features": []}
    assert checks[0]["field"] == "exists"
    assert checks[0]["operator"] == "=="
    assert checks[0]["observed"] is False


def test_real_raw_rows_and_cold_publication(monkeypatch, tmp_path):
    """SCENARIO-REPORT-7664-TERMINAL: replay raw V668 rows to exact candidate."""
    found, checks, hashes = audit.authenticate_inputs(audit.ROOT)
    assert not checks
    summary = audit.reduce_evidence(audit.raw_inputs(found))
    assert summary["coverage"]["independent_groups"] == 248
    assert summary["coverage"]["unknown_claims"] == 272
    assert summary["continuous"]["arms"]["source"]["brier"] == pytest.approx(0.1463138993388428)
    assert all(audit.private_mutations(audit.raw_inputs(found)).values())
    monkeypatch.setattr(audit, "RAW", tmp_path / "raw")
    monkeypatch.setattr(audit, "NOTE", tmp_path / "note.md")
    monkeypatch.setattr(
        audit.checks,
        "run_scoped_validation",
        lambda *args, **kwargs: {
            "validation_receipts": [{"name": "focused_pytest", "passed": True}],
            "required_checks_passed": True,
            "failed_required_commands": [],
        },
    )
    monkeypatch.setattr(
        audit.checks,
        "run_commands",
        lambda root, commands, **kwargs: [
            {"name": command.name, "passed": True, "exit_code": 0, "log_sha256": "sha256:test"}
            for command in commands
        ],
    )
    output = tmp_path / "audit.json"
    value = audit.run_experiment(audit.ROOT, "20260925", output)
    assert output.is_file()
    assert value["verdict_class"] == "null"
    assert value["independent_audit_complete_score"] == 1
    assert len(value["rows"]) > 1000
    candidate = tmp_path / "raw" / "exact_terminal_candidate.json"
    assert audit.cold_replay(candidate) == []
    assert json.loads(output.read_text())["claim_findings"][0]["claim"] == "coverage"
    assert "Exposed-data limits" in (tmp_path / "note.md").read_text()
    assert hashes["producers"]


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_arm",
        "bad_probability",
        "bad_action",
        "bad_log_loss",
        "missing_arm",
        "bad_feedback",
        "bad_delayed_event",
        "bad_retention",
        "duplicate_retention",
        "empty_interval",
        "unknown_mutation",
    ],
)
def test_raw_operand_guards(change):
    """SCENARIO-REPORT-7664-REDUCTION: damaged raw operands never become metrics."""
    inputs = fixture_inputs()
    if change == "duplicate_arm":
        inputs["continuous"].append(dict(inputs["continuous"][0]))
    elif change == "bad_probability":
        inputs["continuous"][0]["probability"] = 2
    elif change == "bad_action":
        inputs["continuous"][0]["typed_action"] = "unknown"
    elif change == "bad_log_loss":
        inputs["evaluation"] = [dict(inputs["continuous"][0], clipped_log_loss=99)]
    elif change == "missing_arm":
        inputs["continuous"].append(dict(inputs["continuous"][0], unit_id="g2", label_from="g2"))
    elif change == "bad_feedback":
        inputs["feedback"][0]["acknowledgment"]["durable"] = False
    elif change == "bad_delayed_event":
        inputs["delayed_events"][0]["release_ordinal"] = 0
    elif change == "bad_retention":
        inputs["retention"][0]["final_brier"] = 0.8
    elif change == "duplicate_retention":
        inputs["retention"].append(dict(inputs["retention"][0]))
    elif change == "empty_interval":
        with pytest.raises(ValueError):
            reducer.paired_interval([], 7664)
        return
    else:
        with pytest.raises(ValueError):
            audit.mutate(inputs, "unknown")
        return
    with pytest.raises(ValueError):
        audit.reduce_evidence(inputs)


def test_invalid_receipt_and_manifest_hash(monkeypatch, tmp_path):
    """SCENARIO-REPORT-7664-CUSTODY: pre-gate receipts and drift stay distinct."""
    label = "results/producer.json"
    feature = "results/feature.jsonl"
    monkeypatch.setattr(audit, "PRODUCERS", {"corpus": label})
    monkeypatch.setattr(audit, "FEATURES", {"fit": feature})
    path = tmp_path / label
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"honest_verdict": "partial_bad", "verdict_class": "partial"}))
    found, checks, hashes = audit.authenticate_inputs(tmp_path)
    assert found["features"] == []
    assert checks[0]["check"] == "producer_valid"
    assert label in hashes["pre_gate_receipts"]
    path.write_text(
        json.dumps(
            {
                "honest_verdict": "complete_null",
                "verdict_class": "null",
                "validation_receipts": {
                    "required_checks_passed": True,
                    "terminal_readers": [{"passed": True}],
                },
                "feature_manifest_path": "results/manifest.json",
            }
        )
    )
    (tmp_path / feature).write_text(json.dumps({"unit_id": "g"}) + "\n")
    _, checks, _ = audit.authenticate_inputs(tmp_path)
    assert checks[0]["check"] == "manifest_exists"
    (tmp_path / "results/manifest.json").write_text(
        json.dumps({"roles": {"fit": {"feature_sha256": "sha256:wrong"}}})
    )
    _, checks, _ = audit.authenticate_inputs(tmp_path)
    assert checks[0]["check"] == "raw_custody"


def test_candidate_tampering_is_caught(monkeypatch, tmp_path):
    """SCENARIO-REPORT-7664-TERMINAL: cold reader checks four independent bonds."""
    found, checks, hashes = audit.authenticate_inputs(audit.ROOT)
    assert not checks
    summary = audit.reduce_evidence(audit.raw_inputs(found))
    value = audit.build_artifact(
        audit.ROOT,
        "20260925",
        found,
        checks,
        hashes,
        summary,
        audit.private_mutations(audit.raw_inputs(found)),
    )
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(value))
    assert audit.cold_replay(path) == []
    monkeypatch.setattr(sys, "argv", ["audit", "--cold", str(path)])
    with pytest.warns(RuntimeWarning), pytest.raises(SystemExit) as direct:
        runpy.run_module(
            "carnot.experiment_7664_v668_independent_evidence_audit", run_name="__main__"
        )
    assert direct.value.code == 0
    for key, replacement, expected in (
        ("gate_check_summary", [{"false": True}], "blocked gate operands changed"),
        ("source_artifact_hashes", {}, "source custody changed"),
        ("reproducibility_checksum", "bad", "reproducibility checksum changed"),
        ("independent_reduction", {}, "independent reduction changed"),
    ):
        changed = deepcopy(value)
        changed[key] = replacement
        path.write_text(json.dumps(changed))
        assert expected in audit.cold_replay(path)
    path.write_text(json.dumps(value))
    monkeypatch.setattr(audit, "private_mutations", lambda inputs: {"source_hash": False})
    assert "private mutation escaped" in audit.cold_replay(path)


@pytest.mark.parametrize("failure", ["affected", "terminal"])
def test_failed_validation_disqualifies_and_zeros_readiness(monkeypatch, tmp_path, failure):
    """SCENARIO-REPORT-7664-TERMINAL: invalid receipts are not scientific nulls."""
    monkeypatch.setattr(audit, "RAW", tmp_path / "raw")
    monkeypatch.setattr(audit, "NOTE", tmp_path / "note.md")
    monkeypatch.setattr(
        audit.checks,
        "run_scoped_validation",
        lambda *args, **kwargs: {
            "validation_receipts": [{"name": "focused_pytest", "passed": failure != "affected"}],
            "required_checks_passed": failure != "affected",
            "failed_required_commands": ["focused_pytest"] if failure == "affected" else [],
        },
    )
    monkeypatch.setattr(
        audit.checks,
        "run_commands",
        lambda root, commands, **kwargs: [
            {
                "name": command.name,
                "passed": command.name != "adversarial_verify" or failure != "terminal",
                "exit_code": 0,
                "log_sha256": "sha256:test",
            }
            for command in commands
        ],
    )
    value = audit.run_experiment(audit.ROOT, "20260925", tmp_path / "audit.json")
    assert value["verdict_class"] == "disqualified"
    assert value["independent_audit_complete_score"] == 0
    assert not next(g for g in value["acceptance_gate_results"] if g["gate"] == "readiness")[
        "passed"
    ]
    assert value["flagged_adversarial"] is (failure == "terminal")


def test_main_dispatch_and_unescaped_mutation(monkeypatch, tmp_path):
    """SCENARIO-REPORT-7664-TERMINAL: CLI readers return their real status."""
    monkeypatch.setattr(audit, "cold_replay", lambda path: [])
    assert audit.main(["--cold", str(tmp_path / "candidate")]) == 0
    monkeypatch.setattr(audit, "cold_replay", lambda path: ["drift"])
    assert audit.main(["--independent", str(tmp_path / "candidate")]) == 1
    calls = []
    monkeypatch.setattr(audit, "run_experiment", lambda *args: calls.append(args))
    assert audit.main(["--output", str(tmp_path / "out")]) == 0
    assert calls
    monkeypatch.setattr(audit, "mutate", lambda inputs, name: None)
    assert not all(audit.private_mutations(fixture_inputs()).values())
