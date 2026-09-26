"""Held-out decision contracts: REQ-REPORT-7704 and REQ-ENERGY-7704."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time

import pytest

from carnot.reporting.heldout_decisions import paired_reduce, score_evaluation
from carnot import experiment_7704_v671_heldout_decisions as experiment
from carnot.experiment_7704_v671_heldout_decisions import authenticate_inputs
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES


def _fixture() -> tuple[list[dict], list[dict], dict, dict, list[str]]:
    roster = ["family-a", "family-b"]
    features = [
        {
            "unit_id": unit,
            "role": "evaluation",
            "arm": view,
            "source_group_id": unit if view != "within_role_derangement" else roster[1 - i],
            "vector": [float(i)] * 8,
            "checked_count": i,
            "proposition_count": i,
            "censored": i == 0,
            "excluded": False,
            "exclusions": [],
            "provenance": {"source_sha256": f"source-{unit}"},
        }
        for i, unit in enumerate(roster)
        for view in ("original_source", "evidence_erasure", "within_role_derangement")
    ]
    labels = [
        {"family_id": unit, "role": "evaluation", "label": i, "provenance": "annotation"}
        for i, unit in enumerate(roster)
    ]
    heads = {
        "best_by_family": {
            family: {"key": family}
            for family in (
                "fit_prior",
                "atom_only",
                "matched_logistic",
                "matched_mlp",
                "typed_gibbs",
            )
        },
        "strongest_comparator": "matched_mlp",
        "heads": {
            family: {
                "schema": "carnot.exp7703.binary_energy.v1",
                "kind": "prior",
                "bias": 0.0,
                "input_width": 8,
                "feature_indices": None,
            }
            for family in (
                "fit_prior",
                "atom_only",
                "matched_logistic",
                "matched_mlp",
                "typed_gibbs",
            )
        },
    }
    heads["heads"]["atom_only"]["input_width"] = 5
    heads["heads"]["atom_only"]["feature_indices"] = [2, 3, 4, 6, 7]
    policy = {"thresholds": [0.0, 0.55], "costs": {"correct": 0, "wrong": 1, "escalate": 0.2}}
    return features, labels, heads, policy, roster


def test_req_report_7704_all_arms_and_zero_coverage() -> None:
    """SCENARIO-REPORT-7704-DECISION; SCENARIO-ENERGY-7704-ZERO-COVERAGE."""
    features, labels, heads, policy, roster = _fixture()
    rows = score_evaluation(features, labels, heads, policy, roster)
    assert len(rows) == 30
    assert {r["unit_id"] for r in rows} == set(roster)
    assert {r["arm"] for r in rows if r["unit_id"] == "family-a"} == {
        f"{family}:{view}"
        for family in heads["best_by_family"]
        for view in ("original_source", "evidence_erasure", "within_role_derangement")
    }
    assert all(r["label"] == 0 and r["checked_count"] == 0 for r in rows[:15])
    assert all(r["typed_action"] == "escalate" and r["decision_cost"] == 0.2 for r in rows)
    assert all(r["brier"] == 0.25 for r in rows)


def test_req_energy_7704_intervention_keeps_label_and_rejects_bad_roster() -> None:
    """Source interventions are paired views, not new labeled families."""
    features, labels, heads, policy, roster = _fixture()
    rows = score_evaluation(features, labels, heads, policy, roster)
    assert {r["label"] for r in rows if r["unit_id"] == "family-b"} == {1}
    altered = deepcopy(labels)
    altered[0]["family_id"] = "unknown"
    with pytest.raises(ValueError, match="label_roster"):
        score_evaluation(features, altered, heads, policy, roster)
    with pytest.raises(ValueError, match="feature_roster"):
        score_evaluation(features[:-1], labels, heads, policy, roster)


def test_req_report_7704_registered_gate_is_paired_and_conjunctive() -> None:
    """Forty blocks, fixed seed and both lower bounds govern benefit."""
    features, labels, heads, policy, roster = _fixture()
    rows = score_evaluation(features, labels, heads, policy, roster)
    result = paired_reduce(rows, roster, "matched_mlp", seed=7704, draws=10_000)
    assert result == paired_reduce(rows, roster, "matched_mlp", seed=7704, draws=10_000)
    assert result["paired_brier_reduction_ci"]["lower"] == 0
    assert result["paired_cost_reduction_ci"]["lower"] == 0
    assert result["registered_decision_benefit_score"] == 0
    assert result["decision_measurement_complete_score"] == 0  # only two fixture families
    assert result["effective_blocks"] == 2


def test_req_report_7704_custody_stops_before_labels(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7704-CUSTODY names missing exact bytes."""
    checks, hashes = authenticate_inputs(tmp_path)
    assert any(c["field"] == "exists" and c["observed"] is False for c in checks)
    assert all("upstream_id" in c and "artifact_path" in c for c in checks)
    assert hashes["missing_evidence"]
    assert "evaluation_evaluator_store.jsonl" not in json.dumps(hashes["producers"])


def test_req_energy_7704_invalid_registered_inputs() -> None:
    """Frozen settings and row cardinality fail closed."""
    features, labels, heads, policy, roster = _fixture()
    cases = [
        (
            "label_roster",
            lambda: score_evaluation(features, labels, heads, policy, roster + roster[:1]),
        ),
        (
            "head_families",
            lambda: score_evaluation(
                features, labels, {**heads, "best_by_family": {}}, policy, roster
            ),
        ),
        (
            "frozen_thresholds",
            lambda: score_evaluation(
                features, labels, heads, {**policy, "thresholds": [0.8, 0.2]}, roster
            ),
        ),
        (
            "frozen_costs",
            lambda: score_evaluation(features, labels, heads, {**policy, "costs": {}}, roster),
        ),
    ]
    for error, invoke in cases:
        with pytest.raises(ValueError, match=error):
            invoke()
    bad = deepcopy(features)
    bad[0]["excluded"] = True
    with pytest.raises(ValueError, match="feature_role_or_exclusion"):
        score_evaluation(bad, labels, heads, policy, roster)
    rows = score_evaluation(features, labels, heads, policy, roster)
    for error, args in (
        ("control_invalid", (rows, roster, "typed_gibbs")),
        ("bootstrap_registration", (rows, roster + roster[:1], "matched_mlp")),
        ("row_roster", (rows[:-1], roster, "matched_mlp")),
    ):
        with pytest.raises(ValueError, match=error):
            paired_reduce(*args, seed=7704)
    bad_rows = deepcopy(rows)
    bad_rows[0]["brier"] = float("nan")
    with pytest.raises(ValueError, match="paired_values"):
        paired_reduce(bad_rows, roster, "matched_mlp", seed=7704)


def _real_measurement() -> tuple[list[dict], dict, list[str], str, list[dict], dict]:
    root = experiment.ROOT
    checks, hashes = authenticate_inputs(root)
    assert all(check["passed"] for check in checks)
    features, labels, heads, policy, roster, control = experiment._inputs(root)
    rows = score_evaluation(features, labels, heads, policy, roster)
    reduction = paired_reduce(rows, roster, control, seed=7704)
    return rows, reduction, roster, control, checks, hashes


def test_req_report_7704_real_cold_reduction_rejects_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7704-TERMINAL replays bytes, rows, intervals and gate."""
    rows, reduced, roster, control, checks, hashes = _real_measurement()
    assert len(roster) == 40
    receipts = [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False}
        for name in REQUIRED_CHECK_NAMES
    ]
    candidate = experiment._artifact(
        "20260926", time.monotonic(), checks, hashes, rows, reduced, receipts, [], control
    )
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate))
    assert experiment.cold_reduce(path)["all_gates_replayed"] is True
    for field, expected in (
        ("source_artifact_hashes", "source_hash_mismatch"),
        ("rows", "raw_reduction_mismatch"),
        ("paired_brier_reduction_ci", "brier_interval_mismatch"),
        ("paired_cost_reduction_ci", "cost_interval_mismatch"),
        ("registered_decision_benefit_score", "benefit_gate_mismatch"),
        ("acceptance_gate_results", "acceptance_gate_mismatch"),
    ):
        changed = deepcopy(candidate)
        if field == "source_artifact_hashes":
            first = next(iter(changed[field]["producers"]))
            changed[field]["producers"][first] = "sha256:wrong"
        elif field == "rows":
            changed[field][0]["brier"] = 99
        elif field == "registered_decision_benefit_score":
            changed[field] = 1
        elif field == "acceptance_gate_results":
            changed[field][0]["passed"] = False
        else:
            changed[field] = {**changed[field], "lower": 99}
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=expected):
            experiment.cold_reduce(path)
    path.write_text(json.dumps(candidate))
    monkeypatch.setattr(experiment, "authenticate_inputs", lambda root: ([{"passed": False}], {}))
    with pytest.raises(ValueError, match="preflight_mismatch"):
        experiment.cold_reduce(path)


def test_req_report_7704_terminal_dispositions() -> None:
    """A measured null, absent input and failed check have distinct terminal classes."""
    rows, reduced, _roster, control, checks, hashes = _real_measurement()
    receipts = [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False}
        for name in REQUIRED_CHECK_NAMES
    ]
    null = experiment._artifact(
        "20260926", time.monotonic(), checks, hashes, rows, reduced, receipts, [], control
    )
    assert null["verdict_class"] == "null"
    assert null["decision_measurement_complete_score"] == 1
    assert null["registered_decision_benefit_score"] == 0
    assert len(null["acceptance_gate_results"]) == 8
    assert set(null["field_principles"]) >= {
        "paired_brier_reduction_ci",
        "paired_cost_reduction_ci",
        "gate:readiness",
    }
    positive_reduction = {**reduced, "registered_decision_benefit_score": 1}
    positive = experiment._artifact(
        "20260926",
        time.monotonic(),
        checks,
        hashes,
        rows,
        positive_reduction,
        receipts,
        [],
        control,
    )
    assert positive["verdict_class"] == "positive"
    failed = deepcopy(receipts)
    failed[0]["passed"] = False
    disqualified = experiment._artifact(
        "20260926", time.monotonic(), checks, hashes, rows, reduced, failed, [], control
    )
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["decision_measurement_complete_score"] == 0
    blocked = experiment._artifact(
        "20260926",
        time.monotonic(),
        [{"passed": False, "field": "exists"}],
        hashes,
        [],
        {},
        [],
        [],
        None,
    )
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"][0]["field"] == "exists"


def test_req_report_7704_candidate_phase_snapshot() -> None:
    """SCENARIO-REPORT-7704-TERMINAL keeps candidate bytes stable after exact readers."""
    spans = [{"phase": "measurement", "duration_s": 1.0}]
    artifact = experiment._artifact(
        "20260926",
        time.monotonic(),
        [{"passed": False}],
        {"producers": {}, "pre_gate_receipts": {}, "missing_evidence": []},
        [],
        {},
        [],
        spans,
        None,
    )
    spans.append({"phase": "terminal", "duration_s": 2.0})
    assert artifact["phase_spans"] == [{"phase": "measurement", "duration_s": 1.0}]
