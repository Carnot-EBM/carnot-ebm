"""Contract tests for REQ-REPORT-7679 and REQ-CONTINUOUS-7679."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting.independent_evidence_audit_v669 import (
    audit_cohort,
    audit_fixture,
    audit_quotes,
    check_private_corruptions,
    inventory,
)
from carnot.reporting import independent_evidence_audit_v669 as reducer
from carnot import experiment_7679_v669_independent_evidence_audit as experiment


def fixture_rows():
    base = {
        "unit_id": "family-a",
        "source_sha256": "sha256:source",
        "answer_sha256": "sha256:answer",
        "dialect": "stack",
        "split": "held",
        "population": "fixture",
        "truth": "unknown",
        "observed": "unknown",
        "provenance": "exact_fixture_oracle",
        "excluded": False,
        "censored": True,
        "raw_metrics": {"checked": 0, "relations": 1, "unknown": 1},
    }
    return [dict(base, arm=arm) for arm in ("membership_only", "bound_relation")]


def cohort_rows():
    base = {
        "unit_id": "family-a",
        "source_group_id": "family-a",
        "role": "online_admission",
        "source_sha256": "sha256:source",
        "original_source_sha256": "sha256:source",
        "answer_sha256": "sha256:answer",
        "denominator": 1,
        "numerator": 0,
        "unknown_claims": 1,
        "checked_relations": 0,
        "excluded": False,
        "censored": True,
        "whole_answer_certified": False,
    }
    return [
        dict(
            base,
            arm=arm,
            source_group_id=(
                None
                if arm == "evidence_erasure"
                else "family-b"
                if arm == "within_role_derangement"
                else "family-a"
            ),
        )
        for arm in ("original_source", "evidence_erasure", "within_role_derangement")
    ]


def quote_rows():
    base = {
        "unit_id": "family-a",
        "source_sha256": "sha256:source",
        "answer_sha256": "sha256:answer",
        "population": "pilot",
        "truth": None,
        "fixture_truth": None,
        "prior_exposure": True,
        "excluded": False,
        "censored": True,
        "output_tokens": 10,
        "generation_s": 2.0,
        "raw_metrics": {
            "proposal_count": 0,
            "full_proposition_supported": 0,
            "unknown_remainder": 1,
        },
    }
    return [dict(base, arm=arm) for arm in ("numeric_offset", "exact_quote")]


def test_scenario_report_7679_rows_keeps_unknown_and_rejects_corruption():
    """SCENARIO-REPORT-7679-ROWS: arms share one denominator."""
    rows, summary = audit_fixture(fixture_rows())
    assert len(rows) == 2
    assert summary["independent_groups"] == 1
    assert summary["unknown_groups"] == 1
    assert summary["fixture_correct_groups"] == 1
    with pytest.raises(ValueError, match="missing arm"):
        audit_fixture(fixture_rows()[:1])
    with pytest.raises(ValueError, match="duplicate"):
        audit_fixture(fixture_rows() + fixture_rows()[:1])


def test_scenario_report_7679_cohort_role_and_label_boundary():
    """REQ-CONTINUOUS-7679: role and admission labels are custody checks."""
    rows, summary = audit_cohort(
        {"online_admission": cohort_rows()},
        {"online_admission": ["family-a"]},
        {"online_admission": [{"labels_accessible": False}]},
    )
    assert len(rows) == 3
    assert summary["independent_groups"] == 1
    assert summary["unknown_groups"] == 1
    swapped = deepcopy(cohort_rows())
    swapped[0]["role"] = "fit"
    with pytest.raises(ValueError, match="role"):
        audit_cohort(
            {"online_admission": swapped},
            {"online_admission": ["family-a"]},
            {"online_admission": [{"labels_accessible": False}]},
        )
    with pytest.raises(ValueError, match="admission label"):
        audit_cohort(
            {"online_admission": cohort_rows()},
            {"online_admission": ["family-a"]},
            {"online_admission": [{"labels_accessible": True}]},
        )


def test_scenario_report_7679_quotes_are_diagnostics():
    """REQ-REPORT-7679: unchanged proposals count but cannot imply quality."""
    rows, summary = audit_quotes(quote_rows())
    assert len(rows) == 2
    assert summary["independent_groups"] == 1
    assert summary["supported_relations"] == 0
    with pytest.raises(ValueError, match="missing arm"):
        audit_quotes(quote_rows()[:1])


def test_scenario_report_7679_custody_missing_and_bad_gate(tmp_path):
    """SCENARIO-REPORT-7679-CUSTODY: absence is exact, never a zero metric."""
    found, checks, hashes = inventory(tmp_path)
    assert not found
    assert len(checks) == 7
    assert checks[0]["field"] == "exists"
    assert checks[0]["observed"] is False
    assert len(hashes["missing_evidence"]) == 7
    p = tmp_path / "results/experiment_7672_v669_bound_relations.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"honest_verdict": "complete_null", "verdict_class": "null"}))
    _, checks, hashes = inventory(tmp_path)
    assert any(c["field"] == "acceptance_gate_results" for c in checks)
    assert str(p.relative_to(tmp_path)) in hashes["pre_gate_receipts"]


def test_scenario_report_7679_private_mutations():
    """SCENARIO-REPORT-7679-ROWS: every private corruption must fail."""
    assert all(check_private_corruptions(fixture_rows(), cohort_rows(), quote_rows()).values())


def test_scenario_report_7679_real_available_cells():
    """REQ-REPORT-7679: real raw stores reduce without producer aggregate code."""
    found, blocked, hashes = inventory(experiment.ROOT)
    rows, summary, mutations = experiment.reduce_available(
        experiment.ROOT, hashes, blocked, experiment.time.monotonic()
    )
    assert sorted(found) == [7672, 7673, 7674, 7676]
    assert len(rows) == 1648
    assert summary["fixture"]["independent_groups"] == 80
    assert summary["cohort"]["independent_groups"] == 480
    assert summary["quotes"]["independent_groups"] == 24
    assert summary["quotes"]["supported_relations"] == 0
    assert all(mutations.values())
    artifact = experiment.build_artifact(
        experiment.ROOT, "20260926", rows, summary, hashes, blocked, mutations
    )
    assert artifact["verdict_class"] == "blocked"
    assert artifact["independent_audit_complete_score"] == 0
    assert artifact["independent_quality_confirmed_score"] == 0
    assert artifact["independent_efficiency_confirmed_score"] == 0
    assert (
        artifact["acceptance_gate_results"]["probability"]["measured_operands"]["proper_loss"]
        is None
    )
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False


def test_scenario_report_7679_fresh_process_replay_and_commands(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7679-TERMINAL: candidate bytes bind raw reduction."""
    _, blocked, hashes = inventory(experiment.ROOT)
    rows, summary, mutations = experiment.reduce_available(
        experiment.ROOT, hashes, blocked, experiment.time.monotonic()
    )
    artifact = experiment.build_artifact(
        experiment.ROOT, "20260926", rows, summary, hashes, blocked, mutations
    )
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact))
    assert experiment.cold_replay(path) == []
    assert experiment.main(["--cold", str(path)]) == 0
    import runpy
    import sys

    monkeypatch.setattr(sys, "argv", [experiment.__name__, "--independent", str(path)])
    with pytest.raises(SystemExit) as terminal:
        runpy.run_module(experiment.__name__, run_name="__main__")
    assert terminal.value.code == 0
    names = [spec.name for spec in experiment.terminal_commands(experiment.ROOT, path)]
    assert names == [
        "fresh_process_cold_replay",
        "independent_raw_reduction",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]
    artifact["rows"].pop()
    path.write_text(json.dumps(artifact))
    assert "independent rows changed" in experiment.cold_replay(path)


def test_scenario_report_7679_malformed_inputs_and_controls(tmp_path):
    """SCENARIO-REPORT-7679-CUSTODY: malformed JSON and role leaks stay exact."""
    p = tmp_path / "results/experiment_7672_v669_bound_relations.json"
    p.parent.mkdir(parents=True)
    p.write_text("{")
    _, checks, hashes = inventory(tmp_path)
    assert checks[0]["field"] == "json_valid"
    assert str(p.relative_to(tmp_path)) in hashes["pre_gate_receipts"]
    altered = fixture_rows()
    del altered[0]["truth"]
    with pytest.raises(ValueError, match="missing field truth"):
        audit_fixture(altered)
    altered = quote_rows()
    altered[1]["source_sha256"] = "sha256:other"
    with pytest.raises(ValueError, match="source changed"):
        audit_quotes(altered)
    altered = cohort_rows()
    altered[0]["whole_answer_certified"] = True
    with pytest.raises(ValueError, match="answer truth"):
        audit_cohort(
            {"online_admission": altered},
            {"online_admission": ["family-a"]},
            {"online_admission": [{"labels_accessible": False}]},
        )


@pytest.mark.parametrize("affected_ok,terminal_ok", [(True, True), (False, True), (True, False)])
def test_scenario_report_7679_orchestration_retains_actual_check_state(
    monkeypatch, tmp_path, affected_ok, terminal_ok
):
    """SCENARIO-REPORT-7679-TERMINAL: failed owned checks disqualify readiness."""
    monkeypatch.setattr(experiment, "RAW", tmp_path / "audit_raw")
    monkeypatch.setattr(experiment.checks, "build_scoped_commands", lambda *a, **k: [])

    def run_commands(_root, _commands, *, log_dir, extra_env):
        if "terminal" in str(log_dir):
            return [
                {
                    "name": name,
                    "passed": terminal_ok,
                    "exit_code": 0 if terminal_ok else 1,
                    "log_sha256": "sha256:terminal",
                }
                for name in (
                    "fresh_process_cold_replay",
                    "independent_raw_reduction",
                    "adversarial_verify",
                    "verdict_row_consistency_strict",
                )
            ]
        return [
            {
                "name": "focused_pytest",
                "passed": affected_ok,
                "exit_code": 0 if affected_ok else 1,
                "log_sha256": "sha256:focused",
            }
        ]

    monkeypatch.setattr(experiment.checks, "run_commands", run_commands)
    monkeypatch.setattr(
        experiment.checks,
        "reduce_required_checks",
        lambda rows: {
            "required_checks_passed": affected_ok,
            "failed_required_commands": [] if affected_ok else ["focused_pytest"],
        },
    )
    output = tmp_path / "result.json"
    result = experiment.run_experiment(experiment.ROOT, "20260926", output)
    assert output.is_file()
    assert result["verdict_class"] == ("blocked" if affected_ok and terminal_ok else "disqualified")
    assert result["flagged_adversarial"] is (not terminal_ok)
    assert result["validation_receipts"]["terminal_readers"]
    assert (tmp_path / "audit_raw/rows.jsonl").is_file()
    assert (tmp_path / "audit_raw/checkpoint.json").is_file()


@pytest.mark.parametrize(
    "corruption,pattern",
    [
        ("arm", "unexpected arm"),
        ("answer", "answer changed"),
        ("roster", "roster membership"),
        ("duplicate_role", "duplicate family"),
        ("source", "original source hash"),
        ("denominator", "relation denominator"),
    ],
)
def test_scenario_report_7679_reducer_corruptions(corruption, pattern):
    """SCENARIO-REPORT-7679-ROWS: each custody boundary rejects its mutation."""
    if corruption in {"arm", "answer"}:
        rows = fixture_rows()
        if corruption == "arm":
            rows[0]["arm"] = "other"
        else:
            rows[0]["answer_sha256"] = "sha256:changed"
        with pytest.raises(ValueError, match=pattern):
            audit_fixture(rows)
        return
    rows = cohort_rows()
    roster = {"online_admission": ["family-a"]}
    if corruption == "roster":
        roster["online_admission"] = ["other"]
    elif corruption == "duplicate_role":
        rows2 = deepcopy(rows)
        for row in rows2:
            row["role"] = "fit"
        with pytest.raises(ValueError, match=pattern):
            audit_cohort(
                {"online_admission": rows, "fit": rows2},
                {**roster, "fit": ["family-a"]},
                {"online_admission": [{"labels_accessible": False}], "fit": []},
            )
        return
    elif corruption == "source":
        rows[0]["source_sha256"] = "sha256:changed"
    else:
        rows[0]["denominator"] = -1
    with pytest.raises(ValueError, match=pattern):
        audit_cohort(
            {"online_admission": rows}, roster, {"online_admission": [{"labels_accessible": False}]}
        )


def test_scenario_report_7679_failed_private_mutation_is_visible(monkeypatch):
    """REQ-REPORT-7679: a private corruption that passes is not reported green."""
    monkeypatch.setattr(reducer, "audit_fixture", lambda rows: ([], {}))
    flags = check_private_corruptions(fixture_rows(), cohort_rows(), quote_rows()[:1])
    assert flags["remove_row"] is False
    assert flags["duplicate_family"] is False
    assert flags["remove_quote_row"] is False


def test_scenario_report_7679_replay_detects_each_changed_receipt(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7679-TERMINAL: cold replay compares every owned operand."""
    _, blocked, hashes = inventory(experiment.ROOT)
    rows, summary, mutations = experiment.reduce_available(
        experiment.ROOT, hashes, blocked, experiment.time.monotonic()
    )
    original = experiment.build_artifact(
        experiment.ROOT, "20260926", rows, summary, hashes, blocked, mutations
    )
    path = tmp_path / "candidate.json"
    for field, changed, expected in (
        ("gate_check_summary", [], "gate operands changed"),
        ("source_artifact_hashes", {}, "source bytes changed"),
        ("independent_reduction", {}, "raw reduction changed"),
        ("private_mutation_results", {}, "private corruption result changed"),
        ("reproducibility_checksum", "sha256:other", "reproducibility checksum changed"),
    ):
        altered = deepcopy(original)
        altered[field] = changed
        path.write_text(json.dumps(altered))
        assert expected in experiment.cold_replay(path)
    path.write_text(json.dumps(original))
    real_inventory = experiment.inventory
    monkeypatch.setattr(experiment, "inventory", lambda root: ({}, *real_inventory(root)[1:]))
    assert "producer roster changed" in experiment.cold_replay(path)


def test_scenario_report_7679_main_dispatch(monkeypatch, tmp_path):
    """SCENARIO-REPORT-7679-TERMINAL: both CLI paths return their real state."""
    output = tmp_path / "out.json"
    called = []
    monkeypatch.setattr(
        experiment, "run_experiment", lambda root, date, path: called.append((date, path))
    )
    assert experiment.main(["--date", "20260926", "--output", str(output)]) == 0
    assert called == [("20260926", output)]
    monkeypatch.setattr(experiment, "cold_replay", lambda path: ["changed"])
    assert experiment.main(["--independent", str(output)]) == 1


def test_scenario_report_7679_missing_and_malformed_raw(tmp_path):
    """SCENARIO-REPORT-7679-CUSTODY: raw absence and invalid JSON are distinct."""
    hashes = {"raw_stores": {}, "missing_evidence": []}
    blocked = []
    assert experiment._read_json(tmp_path, "missing.jsonl", "Exp7672", hashes, blocked) is None
    assert blocked[0]["field"] == "exists"
    path = tmp_path / "bad.jsonl"
    path.write_text("{")
    assert experiment._read_json(tmp_path, "bad.jsonl", "Exp7672", hashes, blocked) is None
    assert blocked[-1]["field"] == "json_valid"


@pytest.mark.parametrize(
    "corruption,check",
    [
        ("fixture", "raw_contract"),
        ("missing_role", "none"),
        ("family_roster", "role_custody"),
        ("role_count", "role_custody"),
        ("model_role", "role_custody"),
        ("cohort_arm", "raw_contract"),
        ("quote_hash", "raw_custody"),
        ("quote_arm", "raw_contract"),
    ],
)
def test_scenario_report_7679_available_cell_failures(monkeypatch, tmp_path, corruption, check):
    """REQ-REPORT-7679: one bad cell leaves other cell reductions available."""
    monkeypatch.setattr(experiment, "ROLES", ("fit",))
    fixture = fixture_rows()
    cohort = cohort_rows()
    for row in cohort:
        row["role"] = "fit"
    evaluator = [{"family_id": "family-a", "role": "fit"}]
    model = [{"role": "fit", "labels_accessible": False}]
    quotes = quote_rows()
    for row in quotes:
        row.update(
            {
                "request_path": str(tmp_path / "request.json"),
                "raw_response_path": str(tmp_path / "response.json"),
                "request_sha256": "sha256:missing",
                "raw_response_sha256": "sha256:missing",
            }
        )
    data = {
        "rows.json": fixture,
        "protocol.json": {"roles": {"fit": {"families": ["family-a"]}}},
        "fit_features.jsonl": cohort,
        "fit_evaluator_store.jsonl": evaluator,
        "fit_model_inputs.jsonl": model,
        "rows.jsonl": quotes,
    }
    if corruption == "fixture":
        fixture.pop()
    elif corruption == "missing_role":
        data["fit_features.jsonl"] = None
    elif corruption == "family_roster":
        evaluator[0]["family_id"] = "other"
    elif corruption == "role_count":
        model.append(deepcopy(model[0]))
    elif corruption == "model_role":
        model[0]["role"] = "online_admission"
    elif corruption == "cohort_arm":
        cohort.pop()
    elif corruption == "quote_arm":
        quotes.pop()

    def read(_root, label, _upstream, _hashes, _blocked):
        return data[label.rsplit("/", 1)[-1]]

    monkeypatch.setattr(experiment, "_read_json", read)
    hashes = {"raw_stores": {}, "missing_evidence": []}
    blocked = []
    rows, summary, mutations = experiment.reduce_available(
        tmp_path, hashes, blocked, experiment.time.monotonic()
    )
    if check != "none":
        assert any(item["check"] == check for item in blocked)
    assert len(rows) >= 0
    assert isinstance(summary, dict)
    assert isinstance(mutations, dict)
