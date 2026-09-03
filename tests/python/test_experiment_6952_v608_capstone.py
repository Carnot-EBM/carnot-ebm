"""Tests for the V608 independent capstone and V609 handoff.

Spec refs: REQ-REPORT-6952 and SCENARIO-REPORT-6952-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6952_v608_capstone as exp


REPO = Path(__file__).resolve().parents[2]


def _task(number: int = 6945) -> dict[str, object]:
    return {
        "number": number,
        "task_id": f"exp{number}-synthetic",
        "title": f"Synthetic {number}",
        "deliverable": f"results/experiment_{number}_synthetic.json",
        "gates": [],
        "prior_failures": [],
    }


def _payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "complete",
        "honest_verdict": "complete_positive_synthetic",
        "verdict_class": "positive",
        "verifier_is_oracle": False,
        "rows": [{"unit": "a", "score": 1.0}],
    }
    payload.update(updates)
    return payload


def _clean_adversarial() -> dict[str, object]:
    return {"flag_count": 0, "max_severity": 0, "flags": []}


def _positive_comparison_payload(score_field: str) -> dict[str, object]:
    return _payload(
        **{
            score_field: 1,
            "paired_metric_rows": [
                {"unit": "a", "delta_vs_strongest_baseline": 0.20},
                {"unit": "b", "delta_vs_strongest_baseline": 0.10},
            ],
            "confidence_interval_rows": [
                {"metric": score_field, "ci95_low": 0.04, "ci95_high": 0.26}
            ],
            "random_direction_rows": [{"passed": True}],
            "shuffled_label_rows": [{"passed": True}],
            "shuffled_control_rows": [{"passed": True}],
            "model_coverage_rows": [{"model": "m1", "passed": True}],
            "label_isolation_rows": [{"passed": True}],
            "leakage_rows": [{"passed": True}],
            "calibration_rows": [{"passed": True}],
            "shortcut_rows": [{"passed": True}],
        }
    )


def test_req_report_6952_spec_precedes_implementation() -> None:
    """REQ-REPORT-6952 owns every required capstone field and scenario."""

    text = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("REQ-REPORT-6952", 1)[1]

    assert {
        "SCENARIO-REPORT-6952-CONTRACT",
        "SCENARIO-REPORT-6952-STATES",
        "SCENARIO-REPORT-6952-CIRCULARITY",
        "SCENARIO-REPORT-6952-ROWS",
        "SCENARIO-REPORT-6952-VERDICT",
        "SCENARIO-REPORT-6952-SAFETY",
        "SCENARIO-REPORT-6952-RETIREMENT",
        "SCENARIO-REPORT-6952-COMPLETION",
    } <= set(exp.spec_anchors(section))
    assert exp.INFERENCE_SUBSTRATE in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in section


def test_scenario_report_6952_contract_is_exact_and_independent() -> None:
    """SCENARIO-REPORT-6952-CONTRACT compares the document with the YAML."""

    design = (REPO / exp.DESIGN_PATH).read_text(encoding="utf-8")
    roadmap = exp.load_yaml(REPO / exp.ROADMAP_PATH)
    rows = exp.build_contract_rows(design, roadmap)

    assert len(rows) == 12
    assert all(row["passed"] for row in rows)
    assert [row["number"] for row in rows] == list(range(6941, 6953))

    changed = deepcopy(roadmap)
    changed["tasks"][4]["title"] = "Changed title"
    changed["tasks"][8]["gated_on"][0]["artifact_field"] = "wrong_field"
    failed = exp.build_contract_rows(design, changed)
    assert failed[4]["title_match"] is False
    assert failed[8]["gates_match"] is False

    changed["tasks"].pop()
    assert exp.build_contract_rows(design, changed)[-1]["yaml_present"] is False


@pytest.mark.parametrize(
    ("conductor", "payload", "adversarial", "row_check", "state", "klass"),
    [
        ({"state": "preemptive_skip"}, None, None, None, "missing", "blocked"),
        (
            {"state": "gate_blocked"},
            _payload(
                status="blocked",
                honest_verdict="blocked_gate_check_failed",
                verdict_class="blocked",
            ),
            _clean_adversarial(),
            ("skipped", ["no recognised row container"]),
            "blocked",
            "blocked",
        ),
        (
            {"state": "ok"},
            _payload(verdict_class="circular_positive", verifier_is_oracle=True),
            _clean_adversarial(),
            ("ok", []),
            "circular_positive",
            "circular_positive",
        ),
        (
            {"state": "ok"},
            _payload(),
            {"flags": [{"kind": "TAUTOLOGY", "severity": "critical"}]},
            ("ok", []),
            "flagged",
            "disqualified",
        ),
        (
            {"state": "ok"},
            _payload(),
            _clean_adversarial(),
            ("findings", ["WINS_NOT_EXCEEDING_LOSSES"]),
            "row_headline_conflict",
            "disqualified",
        ),
        (
            {"state": "ok"},
            _payload(honest_verdict="partial_missing_units"),
            _clean_adversarial(),
            ("ok", []),
            "verdict_prefix_class_conflict",
            "disqualified",
        ),
        (
            {"state": "ok"},
            _payload(flagged_adversarial=True),
            _clean_adversarial(),
            ("ok", []),
            "flagged",
            "disqualified",
        ),
    ],
)
def test_scenario_report_6952_states_fail_closed(
    conductor: dict[str, object],
    payload: dict[str, object] | None,
    adversarial: dict[str, object] | None,
    row_check: tuple[str, list[str]] | None,
    state: str,
    klass: str,
) -> None:
    """SCENARIO-REPORT-6952-STATES preserves each terminal evidence state."""

    row = exp.classify_evidence(
        _task(), conductor, payload, adversarial, row_check, comparison_rows=[]
    )

    assert row["evidence_state"] == state
    assert row["verdict_class"] == klass


@pytest.mark.parametrize(
    ("honest_verdict", "verdict_class"),
    [
        ("complete_positive_synthetic", "null"),
        ("complete_null_synthetic", "positive"),
        ("blocked_synthetic", "partial"),
        ("partial_synthetic", "positive"),
    ],
)
def test_scenario_report_6952_prefix_class_conflicts_work_both_ways(
    honest_verdict: str, verdict_class: str
) -> None:
    """SCENARIO-REPORT-6952-VERDICT derives the class before comparison."""

    row = exp.classify_evidence(
        _task(),
        {"state": "ok"},
        _payload(honest_verdict=honest_verdict, verdict_class=verdict_class),
        _clean_adversarial(),
        ("ok", []),
        comparison_rows=[],
    )

    assert row["evidence_state"] == "verdict_prefix_class_conflict"
    assert row["verdict_class"] == "disqualified"


def test_scenario_report_6952_rows_recompute_every_control() -> None:
    """SCENARIO-REPORT-6952-ROWS recomputes positive gates from source rows."""

    payload = _positive_comparison_payload("causal_hidden_state_positive_score")
    rows = exp.recompute_headlines(6947, payload)

    assert rows[0]["recomputed"] == 1
    assert rows[0]["agrees"] is True
    assert all(rows[0]["checks"].values())

    payload["causal_hidden_state_positive_score"] = 0
    payload["random_direction_rows"] = [{"passed": False}]
    rows = exp.recompute_headlines(6947, payload)
    assert rows[0]["recomputed"] == 0
    assert rows[0]["checks"]["random_direction_control"] is False

    conflict = exp.classify_evidence(
        _task(6947),
        {"state": "ok"},
        _payload(),
        _clean_adversarial(),
        ("ok", []),
        comparison_rows=[{"agrees": False}],
    )
    assert conflict["evidence_state"] == "row_headline_conflict"


def test_scenario_report_6952_uses_each_branch_control_contract() -> None:
    """SCENARIO-REPORT-6952-ROWS uses declared controls, not invented fields."""

    prefix = _positive_comparison_payload("prefix_energy_positive_score")
    prefix.pop("shuffled_label_rows")
    prefix.pop("model_coverage_rows")
    prefix["shuffled_control_rows"] = [{"passed": True}]
    assert exp.recompute_headlines(6945, prefix)[0]["recomputed"] == 1

    hidden = _positive_comparison_payload("causal_hidden_state_positive_score")
    hidden.pop("label_isolation_rows")
    hidden.pop("model_coverage_rows")
    hidden["leakage_rows"] = [{"passed": True}]
    hidden["calibration_rows"] = [{"passed": True}]
    assert exp.recompute_headlines(6947, hidden)[0]["recomputed"] == 1

    arc = _positive_comparison_payload("branch_energy_positive_score")
    arc.pop("random_direction_rows")
    arc.pop("label_isolation_rows")
    arc.pop("model_coverage_rows")
    arc["leakage_rows"] = [{"passed": True}]
    arc["shortcut_rows"] = [{"passed": True}]
    assert exp.recompute_headlines(6949, arc)[0]["recomputed"] == 1


def test_scenario_report_6952_authority_never_synthesizes_scores() -> None:
    """REQ-REPORT-6952 copies a score only from its admissible producer."""

    payloads = {6945: _positive_comparison_payload("prefix_energy_positive_score")}
    states = {
        6945: {"verdict_class": "positive", "evidence_state": "positive"},
        6947: {"verdict_class": "blocked", "evidence_state": "missing"},
    }
    values, branch_rows = exp.authoritative_scores(payloads, states)

    assert values["prefix_energy_positive_score"] == 1
    assert values["causal_hidden_state_positive_score"] is None
    assert (
        next(row for row in branch_rows if row["field"] == "causal_hidden_state_positive_score")[
            "reason"
        ]
        == "authoritative_artifact_unavailable_or_inadmissible"
    )

    states[6945] = {"verdict_class": "disqualified", "evidence_state": "flagged"}
    assert exp.authoritative_scores(payloads, states)[0]["prefix_energy_positive_score"] is None

    states[6945] = {"verdict_class": "null", "evidence_state": "null"}
    assert exp.authoritative_scores(payloads, states)[0]["prefix_energy_positive_score"] is None


def test_scenario_report_6952_gate_safety_and_arc_checks() -> None:
    """SCENARIO-REPORT-6952-SAFETY checks gates, exact writes, and ARC limits."""

    tasks = [
        {
            **_task(6942),
            "task_id": "exp6942-producer",
            "gates": [],
        },
        {
            **_task(6943),
            "task_id": "exp6943-consumer",
            "gates": [
                {
                    "upstream": "exp6942-producer",
                    "artifact_field": "ready_score",
                    "op": "==",
                    "value": 1,
                }
            ],
        },
    ]
    gate_rows = exp.replay_gates(tasks, {6942: {"ready_score": 0}})
    assert gate_rows == [
        {
            "row_type": "gate_replay",
            "task_id": "exp6943-consumer",
            "upstream": "exp6942-producer",
            "artifact_field": "ready_score",
            "op": "==",
            "expected": 1,
            "observed": 0,
            "passed": False,
            "available": True,
        }
    ]

    trace = {
        "continuous_self_learning_task": True,
        "learning_tier": 2,
        "write_rows": [{"outcome_observed_before_write": True}],
        "model_hashes_before": {"m": "abc"},
        "model_hashes_after": {"m": "abc"},
        "no_model_weight_mutation": True,
    }
    safety = exp.build_safety_rows({6950: trace})
    assert safety[0]["passed"] is True

    trace["write_rows"] = [{"outcome_observed_before_write": False}]
    assert exp.build_safety_rows({6950: trace})[0]["delayed_exact_writes"] is False

    arc_rows = exp.build_arc_rows(
        {6948: {"solve_claimed": False}, 6949: {"solve_claimed": False, "default_off": True}},
        registry_unchanged=True,
    )
    assert all(row["passed"] for row in arc_rows)

    missing_declaration = exp.build_arc_rows({6948: {}}, registry_unchanged=True)[0]
    assert missing_declaration["available"] is False
    assert missing_declaration["passed"] is None


def test_scenario_report_6952_failed_boundaries_disqualify_claims() -> None:
    """SCENARIO-REPORT-6952-SAFETY prevents unsafe headline admission."""

    states = {
        6949: {
            "number": 6949,
            "verdict_class": "positive",
            "evidence_state": "positive",
            "admissible": True,
        },
        6950: {
            "number": 6950,
            "verdict_class": "positive",
            "evidence_state": "positive",
            "admissible": True,
        },
    }
    checked = exp.apply_boundary_checks(
        states,
        arc_rows=[{"number": 6949, "available": True, "passed": False}],
        safety_rows=[{"number": 6950, "available": True, "passed": False}],
    )

    assert checked[6949]["evidence_state"] == "arc_claim_boundary_conflict"
    assert checked[6950]["evidence_state"] == "continuous_learning_safety_conflict"
    assert checked[6949]["verdict_class"] == "disqualified"
    assert checked[6950]["admissible"] is False


def test_scenario_report_6952_retirement_requires_exact_repeat() -> None:
    """SCENARIO-REPORT-6952-RETIREMENT emits only exact repeat candidates."""

    task = _task(6948)
    task["prior_failures"] = [
        {
            "experiment_id": "old-a",
            "verdict": "blocked_gate_check_failed",
            "retire_if_same_verdict": True,
        },
        {
            "experiment_id": "old-b",
            "verdict": "complete_null_other",
            "retire_if_same_verdict": True,
        },
        {
            "experiment_id": "old-c",
            "verdict": "blocked_gate_check_failed",
            "retire_if_same_verdict": False,
        },
    ]
    states = {
        6948: {
            "task_id": task["task_id"],
            "honest_verdict": "blocked_gate_check_failed",
        }
    }

    rows = exp.exclusion_candidates([task], states)
    assert [row["prior_experiment_id"] for row in rows] == ["old-a"]


def test_req_report_6952_builds_current_partial_capstone() -> None:
    """SCENARIO-REPORT-6952-COMPLETION completes the audit without science promotion."""

    artifact = exp.build_artifact(REPO, "20260903")

    assert artifact["v608_capstone_complete_score"] == 1
    assert artifact["status"] == "complete_partial"
    assert artifact["verdict_class"] == "partial"
    assert artifact["honest_verdict"].startswith("partial_")
    assert len(artifact["task_contract_rows"]) == 12
    assert len(artifact["task_state_rows"]) == 12
    assert len(artifact["v609_gap_rows"]) == 3
    assert {row["number"] for row in artifact["task_state_rows"]} == set(range(6941, 6953))
    states = {row["number"]: row for row in artifact["task_state_rows"]}
    assert states[6941]["evidence_state"] == "null"
    assert states[6942]["evidence_state"] == "blocked"
    assert states[6943]["evidence_state"] == "blocked"
    assert states[6948]["evidence_state"] == "blocked"
    assert all(row["flag_count"] == 0 for row in artifact["adversarial_verify_rows"])
    assert artifact["prefix_energy_positive_score"] is None
    assert artifact["causal_hidden_state_positive_score"] is None
    assert artifact["branch_energy_positive_score"] is None
    assert artifact["trace_state_positive_score"] is None
    assert artifact["audited_trace_state_positive_score"] is None
    assert {row["number"] for row in artifact["exclusion_candidate_rows"]} == {6948}
    assert exp.validate_artifact(artifact) == []


def test_req_report_6952_missing_core_contract_blocks(tmp_path: Path) -> None:
    """REQ-REPORT-6952 writes a complete blocked shape for a missing contract."""

    artifact = exp.build_artifact(tmp_path, "20260903")

    assert artifact["v608_capstone_complete_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == exp.BLOCKED_VERDICT
    assert artifact["gate_check_summary"]["failed_check"] == "core_contract_preconditions"
    assert exp.validate_artifact(artifact) == []


def test_req_report_6952_validation_and_defensive_inputs(tmp_path: Path) -> None:
    """REQ-REPORT-6952 validation rejects malformed inputs and altered checksums."""

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("- not-a-map\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML root is not a mapping"):
        exp.load_yaml(bad_yaml)

    assert exp.roadmap_tasks({"tasks": {}}) == []
    assert exp.roadmap_tasks({"tasks": [None, {"id": "exp1"}]})[0]["number"] == 1
    assert exp._gate_pass(1, "!=", 1) is False
    assert exp._gate_pass(None, ">=", 1) is False
    assert exp._scalar({"value": 3, "principle": "direct"}) == 3
    assert exp._parse_gate("not a gate") == [{"unparsed": "not a gate"}]
    assert exp._rows_pass({}, "optional", required=False) is True
    assert exp._paired_gain({"paired_metric_rows": [None]}) is False
    assert exp._ci_above_zero({"confidence_interval_rows": [None]}) is False
    assert exp.recompute_headlines(6944, {}) == []
    assert exp._critical(None) is False

    log = "\n".join(
        [
            "ignored before activation",
            "| 2026-09-03 16:36 UTC | Milestone 2026.09.608 activated | OK | ready |",
            "| too short |",
            "| now | Unknown title | FAILED | no match |",
            "| now | Synthetic 6945 | FLAGGED | review |",
        ]
    )
    assert exp.parse_conductor_states(log, [_task()])[6945]["state"] == "flagged"
    assert exp._verdict_shape(_payload(status="flagged")) == "disqualified"
    assert exp._prefix_verdict_class("flagged_synthetic") == "disqualified"
    assert exp._verdict_shape(_payload(honest_verdict="flagged_synthetic")) == "disqualified"
    assert (
        exp._verdict_shape(_payload(honest_verdict="complete_circular_positive_fixture"))
        == "circular_positive"
    )
    assert exp._verdict_shape(_payload(verifier_is_oracle=True)) == "circular_positive"
    assert (
        exp._verdict_shape(
            _payload(
                honest_verdict="complete: class declared separately",
                verdict_class="circular_positive",
            )
        )
        == "circular_positive"
    )
    assert (
        exp._verdict_shape(
            _payload(
                honest_verdict="complete: class declared separately",
                verifier_is_oracle=True,
            )
        )
        == "circular_positive"
    )
    assert (
        exp._verdict_shape(
            _payload(
                honest_verdict="complete: class declared separately",
                verdict_class="null",
            )
        )
        == "null"
    )
    assert exp._verdict_shape({}) == "partial"

    malformed_gate_task = _task()
    malformed_gate_task["gates"] = ["bad"]
    assert exp.replay_gates([malformed_gate_task], {}) == []

    malformed_prior_task = _task()
    malformed_prior_task["prior_failures"] = ["bad"]
    assert exp.exclusion_candidates([malformed_prior_task], {}) == []

    assert exp._load_json(tmp_path / "missing.json") is None
    malformed_json = tmp_path / "malformed.json"
    malformed_json.write_text("{", encoding="utf-8")
    assert exp._load_json(malformed_json) is None
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert exp._load_json(list_json) is None

    artifact = exp.build_artifact(REPO, "20260903")
    broken = deepcopy(artifact)
    broken["honest_verdict"] = "complete_positive_wrong"
    broken["reproducibility_checksum"] = exp.reproducibility_checksum(broken)
    assert "honest_verdict prefix conflicts with verdict_class" in exp.validate_artifact(broken)

    broken = deepcopy(artifact)
    broken["field_principles"].pop("rows")
    broken["reproducibility_checksum"] = exp.reproducibility_checksum(broken)
    assert "field_principles missing rows" in exp.validate_artifact(broken)

    broken = deepcopy(artifact)
    broken["reproducibility_checksum"] = "wrong"
    assert "reproducibility_checksum mismatch" in exp.validate_artifact(broken)

    broken = deepcopy(artifact)
    broken["verdict_class"] = "blocked"
    broken["honest_verdict"] = "blocked_missing_summary"
    broken["gate_check_summary"] = {}
    broken["reproducibility_checksum"] = exp.reproducibility_checksum(broken)
    assert "blocked verdict requires an exact failed gate check" in exp.validate_artifact(broken)


def test_req_report_6952_model_and_cold_audit_safety_rows() -> None:
    """REQ-REPORT-6952 checks declared model records and cold-audit controls."""

    models = list(exp.EXPECTED_MODELS[6950])
    assert exp._expected_model_coverage(6945, {}) is True
    assert exp._expected_model_coverage(6950, {}) is False
    assert exp._expected_model_coverage(6950, {"model_specs": models[:-1]}) is False
    assert exp._expected_model_coverage(6950, {"model_specs": models}) is True

    cold = {
        key: [{"passed": True}]
        for key in (
            "event_order_rows",
            "future_label_isolation_rows",
            "receipt_recheck_rows",
            "trace_content_rows",
            "token_budget_rows",
            "model_immutability_rows",
        )
    }
    safety = exp.build_safety_rows({6951: cold})[1]
    assert safety["available"] is True
    assert safety["passed"] is True

    cold["event_order_rows"] = [{"passed": False}]
    assert exp.build_safety_rows({6951: cold})[1]["passed"] is False
    cold.pop("model_immutability_rows")
    assert exp.build_safety_rows({6951: cold})[1]["available"] is False


def test_req_report_6952_checker_and_terminal_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6952 keeps checker errors and all capstone terminals testable."""

    checker = tmp_path / "checker.py"
    checker.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(exp.importlib.util, "spec_from_file_location", lambda *_: None)
    with pytest.raises(RuntimeError, match="cannot load verifier"):
        exp._load_checker(checker, "missing")

    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        exp, "_load_checker", lambda *_: (_ for _ in ()).throw(RuntimeError("both failed"))
    )
    adversarial, row_check = exp._audit_artifact(tmp_path, artifact_path)
    assert adversarial["flags"][0]["kind"] == "VERIFIER_ERROR"
    assert row_check[0] == "unreadable"

    class Adversarial:
        @staticmethod
        def verify_artifact(path: Path) -> dict[str, object]:
            return {"flags": [], "path": str(path)}

    calls = iter([Adversarial(), RuntimeError("row checker failed")])

    def load_or_raise(*_: object) -> object:
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(exp, "_load_checker", load_or_raise)
    adversarial, row_check = exp._audit_artifact(tmp_path, artifact_path)
    assert adversarial["flags"] == []
    assert row_check == ("unreadable", ["VERIFIER_ERROR: row checker failed"])

    monkeypatch.setattr(
        exp,
        "build_contract_rows",
        lambda *_: [{"number": number, "passed": number != 6941} for number in range(6941, 6953)],
    )
    monkeypatch.setattr(exp, "_audit_artifact", lambda *_: (_clean_adversarial(), ("ok", [])))
    disqualified = exp.build_artifact(REPO, "20260903")
    assert disqualified["verdict_class"] == "disqualified"

    monkeypatch.setattr(
        exp,
        "build_contract_rows",
        lambda *_: [{"number": number, "passed": True} for number in range(6941, 6953)],
    )

    def positive_state(task: dict[str, object], *_: object, **__: object) -> dict[str, object]:
        return {
            "row_type": "task_state",
            "task_id": task["task_id"],
            "number": task["number"],
            "title": task["title"],
            "conductor_state": "ok",
            "artifact_state": "present",
            "evidence_state": "positive",
            "declared_verdict_class": "positive",
            "structural_verdict_class": "positive",
            "verdict_class": "positive",
            "honest_verdict": "complete_positive_synthetic",
            "admissible": True,
            "terminal": True,
            "row_check_status": "ok",
            "row_findings": [],
            "adversarial_critical": False,
        }

    monkeypatch.setattr(exp, "classify_evidence", positive_state)
    monkeypatch.setattr(
        exp,
        "build_arc_rows",
        lambda *_: [
            {"number": 6948, "available": True, "passed": True},
            {"number": 6949, "available": True, "passed": True},
        ],
    )
    monkeypatch.setattr(
        exp,
        "build_safety_rows",
        lambda *_: [
            {"number": 6950, "available": True, "passed": True},
            {"number": 6951, "available": True, "passed": True},
        ],
    )
    positive = exp.build_artifact(REPO, "20260903")
    assert positive["verdict_class"] == "positive"

    monkeypatch.setattr(exp, "build_artifact", lambda *_: {})
    assert exp.main(["--repo-root", str(REPO), "--output", str(tmp_path / "bad.json")]) == 1


def test_req_report_6952_cli_writes_only_requested_target(tmp_path: Path) -> None:
    """REQ-REPORT-6952 exposes the required writer without changing tracked data."""

    target = tmp_path / "exp6952.json"
    assert exp.main(["--repo-root", str(REPO), "--date", "20260903", "--output", str(target)]) == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert exp.validate_artifact(payload) == []
    assert exp.main(["--validate", "--output", str(target)]) == 0
    target.write_text("{}", encoding="utf-8")
    assert exp.main(["--validate", "--output", str(target)]) == 1
