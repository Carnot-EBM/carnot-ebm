"""REQ-REPORT-7721 and REQ-CL-7721-AUDIT evidence audit checks."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot.reporting.evidence_audit_v672 import inspect_sources, reduce_bundle
from carnot.experiment_7721_v672_independent_evidence_audit import cold_replay, make_artifact
from carnot import experiment_7721_v672_independent_evidence_audit as audit
from carnot.reporting import experiment_7303_validation_scope as validation


def fixture_bundle() -> dict:
    """Return two original families, including an explicit zero-coverage family."""
    return {
        "family_roster": ["source-a", "source-b"],
        "source_registry": {"source-a": "sha-a", "source-b": "sha-b"},
        "frozen_best_arm": "latent",
        "rows": [
            {
                "family_id": "source-a",
                "source_sha256": "sha-a",
                "arm": "latent",
                "label": 1,
                "probability_error": 0.8,
                "typed_action": "reject",
                "coverage": True,
                "annotation_origin": "human",
                "prediction_tick": 1,
                "label_release_tick": 2,
                "candidate_hash_at_prediction": "candidate-1",
                "candidate_hash_at_admission": "candidate-1",
                "static_bank_features": 8,
                "future_template_firings": 1,
                "retention_label": 1,
                "retention_probability_error": 0.7,
            },
            {
                "family_id": "source-b",
                "source_sha256": "sha-b",
                "arm": "pooled",
                "label": 0,
                "probability_error": None,
                "typed_action": "escalate",
                "coverage": False,
                "annotation_origin": "human",
                "prediction_tick": 1,
                "label_release_tick": 2,
                "candidate_hash_at_prediction": "candidate-1",
                "candidate_hash_at_admission": "candidate-1",
                "static_bank_features": 8,
                "future_template_firings": 0,
                "retention_label": None,
                "retention_probability_error": None,
            },
        ],
        "events": [
            {
                "kind": "proposal",
                "released_tick": 2,
                "use_tick": 3,
                "charged": True,
                "candidate_hash": "candidate-1",
            }
        ],
    }


def test_missing_science_is_exact_terminal_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7721-CUSTODY: missing science is blocked, not partial."""
    plan = {
        7718: "results/experiment_7718_v672_natural_decisions.json",
        7720: "results/experiment_7720_v672_continuous_acquisition.json",
    }
    custody, hashes, failures = inspect_sources(tmp_path, plan, {7718, 7720})
    artifact = make_artifact(tmp_path, "20260926", custody, hashes, failures, None)
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["independent_audit_complete_score"] == 0
    assert artifact["independent_static_eligible"] is False
    assert artifact["independent_online_eligible"] is False
    assert {check["upstream_id"] for check in artifact["gate_check_summary"]} == {
        "Exp7718",
        "Exp7720",
    }
    assert all(check["field"] == "exists" and check["observed"] is False for check in failures)
    assert artifact["acceptance_gate_results"]["probability"]["passed"] is None


def test_changed_upstream_bytes_fail_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7721-CUSTODY: exact bytes bind the cold replay."""
    path = tmp_path / "results/experiment_7718_v672_natural_decisions.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "honest_verdict": "complete_null_no_gain",
                "verdict_class": "null",
                "flagged_adversarial": False,
            }
        )
    )
    plan = {7718: str(path.relative_to(tmp_path))}
    custody, hashes, failures = inspect_sources(tmp_path, plan, {7718})
    assert not failures
    artifact = make_artifact(tmp_path, "20260926", custody, hashes, failures, None)
    artifact["source_plan"] = plan
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact))
    assert cold_replay(candidate) == []
    path.write_text(path.read_text() + " ")
    assert "source_artifact_hashes_changed" in cold_replay(candidate)


@pytest.mark.parametrize(
    ("change", "failure"),
    [
        (lambda b: b["rows"][0].update(family_id="source-b"), "source_identity"),
        (lambda b: b["rows"][0].update(label_release_tick=0), "future_label"),
        (lambda b: b["rows"].pop(), "family_coverage"),
        (lambda b: b["rows"][0].update(annotation_origin="injected"), "annotation_origin"),
        (lambda b: b["rows"][0].update(static_bank_features=0), "static_closure"),
        (lambda b: b["rows"][0].update(candidate_hash_at_admission="changed"), "frozen_candidate"),
    ],
)
def test_private_mutations_fail_distinct_checks(change, failure: str) -> None:
    """SCENARIO-REPORT-7721-REDUCTION and SCENARIO-CL-7721-MUTATIONS."""
    bundle = deepcopy(fixture_bundle())
    change(bundle)
    assert failure in {item["check"] for item in reduce_bundle(bundle)["failed_checks"]}


def test_valid_bundle_reduces_family_counts_without_reselecting() -> None:
    """SCENARIO-REPORT-7721-REDUCTION: zero coverage remains in denominator."""
    reduced = reduce_bundle(fixture_bundle())
    assert reduced["failed_checks"] == []
    assert reduced["sample_size"]["intended_families"] == 2
    assert reduced["sample_size"]["covered_families"] == 1
    assert reduced["by_arm"]["latent"]["brier_mean"] == pytest.approx(0.04)
    assert reduced["frozen_best_arm"] == "latent"


def test_inventory_rejects_broken_json_and_flagged_historical(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7721-CUSTODY: pre-gate and history stay separate."""
    broken = tmp_path / "broken.json"
    broken.write_text("{")
    historical = tmp_path / "historical.json"
    historical.write_text(
        json.dumps(
            {
                "honest_verdict": "complete_null",
                "verdict_class": "null",
                "flagged_adversarial": True,
            }
        )
    )
    custody, hashes, failures = inspect_sources(
        tmp_path, {7718: "broken.json", 7707: "historical.json"}, {7718}
    )
    assert len(failures) == 1 and failures[0]["field"] == "json_object"
    assert "broken.json" in hashes["pre_gate_receipts"]
    assert "historical.json" in hashes["flagged_historical_evidence"]
    assert [row["state"] for row in custody] == ["pre_gate", "pre_gate"]


def test_inventory_rejects_flagged_and_nonterminal_science(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7721-CUSTODY: terminal class and reader flag are checked."""
    path = tmp_path / "science.json"
    path.write_text(
        json.dumps(
            {"honest_verdict": "pending", "verdict_class": "null", "flagged_adversarial": False}
        )
    )
    _, _, failures = inspect_sources(tmp_path, {7718: "science.json"}, {7718})
    assert failures[0]["field"] == "honest_verdict"
    path.write_text(
        json.dumps(
            {
                "honest_verdict": "complete_null",
                "verdict_class": "null",
                "flagged_adversarial": True,
            }
        )
    )
    _, _, failures = inspect_sources(tmp_path, {7718: "science.json"}, {7718})
    assert failures[0]["field"] == "flagged_adversarial"


def test_reducer_rejects_roster_chronology_charge_and_action() -> None:
    """SCENARIO-CL-7721-MUTATIONS: later use needs a released charged event."""
    bundle = fixture_bundle()
    bundle["family_roster"].append("source-a")
    bundle["events"][0].update(released_tick=4, charged=False)
    bundle["rows"][0]["typed_action"] = "unknown"
    bundle["rows"][0]["probability_error"] = 1.5
    bundle["rows"].append(deepcopy(bundle["rows"][0]))
    checks = {item["check"] for item in reduce_bundle(bundle)["failed_checks"]}
    assert {
        "family_roster",
        "event_chronology",
        "proposal_charge",
        "typed_action",
        "probability",
        "duplicate_family_arm",
    } <= checks


def test_reducer_paired_contrast_is_frozen() -> None:
    """SCENARIO-REPORT-7721-REDUCTION: arms share the original family denominator."""
    bundle = fixture_bundle()
    pooled = deepcopy(bundle["rows"][0])
    pooled.update(arm="pooled", probability_error=0.6)
    bundle["rows"].append(pooled)
    reduced = reduce_bundle(bundle)
    assert reduced["failed_checks"] == []
    assert reduced["sample_size"]["observed_families"] == 2
    assert reduced["latent_vs_pooled_brier"] == pytest.approx(-0.12)


def _install_fake_validation(monkeypatch, *, fail_required=False, fail_terminal=False):
    monkeypatch.setattr(validation, "build_scoped_commands", lambda *args, **kwargs: [])

    def run_commands(_root, _commands, *, log_dir, extra_env, heartbeat_s=60.0):
        if log_dir.name == "affected":
            return [
                {
                    "name": name,
                    "passed": not (fail_required and name == "focused_pytest"),
                    "exit_code": int(fail_required and name == "focused_pytest"),
                    "timed_out": False,
                }
                for name in validation.REQUIRED_CHECK_NAMES
            ]
        if log_dir.name == "full":
            return [
                {"name": "full_python_suite", "passed": False, "exit_code": 2, "timed_out": False}
            ]
        candidate = log_dir.parent.parent / "terminal_candidate.json"
        assert candidate.is_file() and cold_replay(candidate) == []
        return [
            {
                "name": name,
                "passed": not (fail_terminal and name == "adversarial_verify"),
                "exit_code": int(fail_terminal and name == "adversarial_verify"),
                "timed_out": False,
            }
            for name in (
                "fresh_process_cold_replay",
                "adversarial_verify",
                "verdict_row_consistency_strict",
            )
        ]

    monkeypatch.setattr(validation, "run_commands", run_commands)


def test_run_publishes_exact_blocked_candidate(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-REPORT-7721-TERMINAL: external absence remains terminal."""
    monkeypatch.setattr(audit, "PLAN", {7718: "static.json", 7720: "online.json"})
    _install_fake_validation(monkeypatch)
    output = tmp_path / "result.json"
    result = audit.run_experiment(tmp_path, "20260926", output)
    assert json.loads(output.read_bytes()) == result
    assert result["verdict_class"] == "blocked"
    assert len(result["phase_spans"]) == 6
    assert result["phase_spans"][0]["end_s"] <= result["phase_spans"][1]["start_s"]
    assert result["validation_receipts"]["required_checks_passed"] is True


@pytest.mark.parametrize(("fail_required", "fail_terminal"), [(True, False), (False, True)])
def test_run_disqualifies_failed_readers(
    tmp_path: Path, monkeypatch, fail_required: bool, fail_terminal: bool
) -> None:
    """SCENARIO-REPORT-7721-TERMINAL: failed required commands zero readiness."""
    monkeypatch.setattr(audit, "PLAN", {7718: "static.json", 7720: "online.json"})
    _install_fake_validation(monkeypatch, fail_required=fail_required, fail_terminal=fail_terminal)
    result = audit.run_experiment(tmp_path, "20260926", tmp_path / "result.json")
    assert result["verdict_class"] == "disqualified"
    assert result["independent_audit_complete_score"] == 0
    assert result["flagged_adversarial"] is fail_terminal


def test_run_reduces_eligible_null_and_missing_raw(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-REPORT-7721-TERMINAL: valid null needs independently readable raw."""
    plan = {7718: "static.json", 7720: "online.json"}
    monkeypatch.setattr(audit, "PLAN", plan)
    for filename in plan.values():
        (tmp_path / filename).write_text(
            json.dumps(
                {
                    "honest_verdict": "complete_null_no_gain",
                    "verdict_class": "null",
                    "flagged_adversarial": False,
                }
            )
        )
    _install_fake_validation(monkeypatch)
    missing = audit.run_experiment(tmp_path, "20260926", tmp_path / "missing.json")
    assert missing["verdict_class"] == "blocked"
    assert missing["gate_check_summary"][0]["check"] == "raw_bundle_exists"
    raw_bundle = tmp_path / audit.RAW / "frozen_raw_bundle.json"
    raw_bundle.write_text(json.dumps(fixture_bundle()))
    complete = audit.run_experiment(tmp_path, "20260926", tmp_path / "complete.json")
    assert complete["verdict_class"] == "null"
    assert complete["independent_audit_complete_score"] == 1
    assert complete["recomputed_metrics"]["sample_size"]["intended_families"] == 2


def test_main_cold_and_run_dispatch(tmp_path: Path, monkeypatch, capsys) -> None:
    """SCENARIO-REPORT-7721-TERMINAL: CLI dispatches the cold reader."""
    monkeypatch.setattr(audit, "PLAN", {7718: "static.json"})
    custody, hashes, failures = inspect_sources(tmp_path, audit.PLAN, {7718})
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(make_artifact(tmp_path, "20260926", custody, hashes, failures, None))
    )
    assert audit.main(["--cold", str(candidate)]) == 0
    assert "cold_replay_errors" in capsys.readouterr().out
    monkeypatch.setattr(audit, "run_experiment", lambda *args: {})
    assert audit.main(["--date", "20260926", "--output", str(tmp_path / "x.json")]) == 0


def test_cold_replay_detects_changed_and_removed_raw(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-REPORT-7721-TERMINAL: raw metrics and presence bind the candidate."""
    monkeypatch.setattr(audit, "PLAN", {7718: "static.json", 7720: "online.json"})
    for path in (tmp_path / "static.json", tmp_path / "online.json"):
        path.write_text(
            json.dumps(
                {
                    "honest_verdict": "complete_null",
                    "verdict_class": "null",
                    "flagged_adversarial": False,
                }
            )
        )
    raw = tmp_path / audit.RAW
    raw.mkdir(parents=True)
    bundle = fixture_bundle()
    raw_path = raw / "frozen_raw_bundle.json"
    raw_path.write_text(json.dumps(bundle))
    _install_fake_validation(monkeypatch)
    audit.run_experiment(tmp_path, "20260926", tmp_path / "result.json")
    candidate = raw / "terminal_candidate.json"
    bundle["rows"][0]["probability_error"] = 0.7
    raw_path.write_text(json.dumps(bundle))
    assert "raw_reduction_changed" in cold_replay(candidate)
    bundle["rows"][0]["annotation_origin"] = "injected"
    raw_path.write_text(json.dumps(bundle))
    assert "gate_check_summary_changed" in cold_replay(candidate)
    raw_path.unlink()
    assert "raw_bundle_missing" in cold_replay(candidate)


def test_invalid_raw_disqualifies_eligibility(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-REPORT-7721-REDUCTION: a private mutation blocks claims."""
    monkeypatch.setattr(audit, "PLAN", {7718: "static.json", 7720: "online.json"})
    for path in (tmp_path / "static.json", tmp_path / "online.json"):
        path.write_text(
            json.dumps(
                {
                    "honest_verdict": "complete_null",
                    "verdict_class": "null",
                    "flagged_adversarial": False,
                }
            )
        )
    raw = tmp_path / audit.RAW
    raw.mkdir(parents=True)
    bundle = fixture_bundle()
    bundle["rows"][0]["annotation_origin"] = "injected"
    (raw / "frozen_raw_bundle.json").write_text(json.dumps(bundle))
    _install_fake_validation(monkeypatch)
    result = audit.run_experiment(tmp_path, "20260926", tmp_path / "result.json")
    assert result["verdict_class"] == "blocked"
    assert result["gate_check_summary"][0]["check"] == "raw_bundle_consistency"


def test_module_entrypoint_calls_main(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-REPORT-7721-TERMINAL: module CLI exits with cold result."""
    monkeypatch.setattr(audit, "PLAN", {7718: "static.json"})
    custody, hashes, failures = inspect_sources(tmp_path, audit.PLAN, {7718})
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(make_artifact(tmp_path, "20260926", custody, hashes, failures, None))
    )
    monkeypatch.setattr(sys, "argv", ["audit", "--cold", str(candidate)])
    with pytest.raises(SystemExit) as stop:
        runpy.run_path(audit.__file__, run_name="__main__")
    assert stop.value.code == 0
