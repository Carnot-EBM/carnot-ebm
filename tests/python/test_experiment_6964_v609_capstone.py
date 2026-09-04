"""Tests for the V609 independent capstone and V610 handoff.

Spec refs: REQ-REPORT-6964 and SCENARIO-REPORT-6964-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6964_v609_capstone as exp


REPO = Path(__file__).resolve().parents[2]


def _task(number: int = 6957) -> dict[str, object]:
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


def _positive_selection_payload(score_field: str) -> dict[str, object]:
    return _payload(
        **{
            score_field: 1,
            "certified_selection_run_complete_score": 1,
            "certified_selection_audit_complete_score": 1,
            "candidate_rows": [{"attempt_key": f"a{i}"} for i in range(6)],
            "candidate_group_rows": [
                {"group_id": "g1", "candidate_count": 3},
                {"group_id": "g2", "candidate_count": 3},
            ],
            "selection_rows": [{"terminal": True}],
            "paired_metric_rows": [
                {
                    "paired_top1_delta": 1,
                    "strongest_non_oracle_baseline": "syntax_validity",
                }
            ],
            "confidence_interval_rows": [{"ci95_lower": 0.1, "ci95_upper": 0.3}],
            "headroom_rows": [{"available_headroom": 5, "captured_headroom": 1}],
            "leakage_rows": [{"passed": True}],
            "tie_rows": [{"terminal": True}],
            "fresh_process_replay_rows": [{"replay_matches": True}],
            "shuffled_energy_rows": [{"terminal": True}],
            "fixed_order_rows": [{"terminal": True}],
            "label_isolation_rows": [{"passed": True}],
            "tie_policy_rows": [{"passed": True}],
            "candidate_order_rows": [{"passed": True}],
            "checkpoint_reload_rows": [{"passed": True}],
            "certificate_replay_rows": [{"passed": True}],
            "proposal_budget_rows": [{"passed": True}],
            "aggregate_consistency_rows": [{"passed": True}],
            "audit_rows": [{"passed": True, "terminal": True}],
            "gate_check_summary": {
                "strongest_non_oracle_baseline": "syntax_validity",
                "headroom_capture_rate": 0.2,
                "raw_positive_gate_passed": True,
            },
        }
    )


def test_req_report_6964_spec_precedes_implementation() -> None:
    """REQ-REPORT-6964 owns every required field and scenario."""

    text = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("REQ-REPORT-6964", 1)[1]

    assert {
        "SCENARIO-REPORT-6964-PREFLIGHT",
        "SCENARIO-REPORT-6964-CONTRACT",
        "SCENARIO-REPORT-6964-STATES",
        "SCENARIO-REPORT-6964-ROWS",
        "SCENARIO-REPORT-6964-VERDICT",
        "SCENARIO-REPORT-6964-SAFETY",
        "SCENARIO-REPORT-6964-RETIREMENT",
        "SCENARIO-REPORT-6964-COMPLETION",
    } <= set(exp.spec_anchors(section))
    assert exp.INFERENCE_SUBSTRATE in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in section


def test_scenario_report_6964_contract_is_exact_and_independent() -> None:
    """SCENARIO-REPORT-6964-CONTRACT compares both contract sources."""

    design = (REPO / exp.DESIGN_PATH).read_text(encoding="utf-8")
    roadmap = exp.load_yaml(REPO / exp.ROADMAP_PATH)
    rows = exp.build_contract_rows(design, roadmap)

    assert len(rows) == 12
    assert all(row["passed"] for row in rows)
    assert [row["number"] for row in rows] == list(range(6953, 6965))

    changed = deepcopy(roadmap)
    changed["tasks"][4]["title"] = "Changed title"
    changed["tasks"][6]["gated_on"][1]["artifact_field"] = "wrong_field"
    failed = exp.build_contract_rows(design, changed)
    assert failed[4]["title_match"] is False
    assert failed[6]["gates_match"] is False

    changed["tasks"].pop()
    assert exp.build_contract_rows(design, changed)[-1]["yaml_present"] is False


@pytest.mark.parametrize(
    ("conductor", "payload", "adversarial", "row_check", "state", "klass"),
    [
        ({"state": "not_recorded"}, None, None, None, "missing", "partial"),
        ({"state": "preemptive_skip"}, None, None, None, "missing", "blocked"),
        (
            {"state": "ok"},
            {"status": "blocked", "honest_verdict": "blocked_gate_check_failed"},
            _clean_adversarial(),
            ("skipped", []),
            "bootstrap_only",
            "blocked",
        ),
        (
            {"state": "ok"},
            _payload(
                honest_verdict="complete_circular_positive_contract",
                verdict_class="circular_positive",
                verifier_is_oracle=True,
            ),
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
            _payload(honest_verdict="complete_null_synthetic", verdict_class="positive"),
            _clean_adversarial(),
            ("ok", []),
            "verdict_prefix_class_conflict",
            "disqualified",
        ),
        (
            {"state": "flagged"},
            _payload(flagged_adversarial=True),
            _clean_adversarial(),
            ("ok", []),
            "flagged",
            "disqualified",
        ),
    ],
)
def test_scenario_report_6964_states_fail_closed(
    conductor: dict[str, object],
    payload: dict[str, object] | None,
    adversarial: dict[str, object] | None,
    row_check: tuple[str, list[str]] | None,
    state: str,
    klass: str,
) -> None:
    """SCENARIO-REPORT-6964-STATES preserves weaker evidence states."""

    row = exp.classify_evidence(
        _task(), conductor, payload, adversarial, row_check, comparison_rows=[]
    )

    assert row["evidence_state"] == state
    assert row["verdict_class"] == klass


def test_scenario_report_6964_rows_recompute_mapping_and_energy() -> None:
    """SCENARIO-REPORT-6964-ROWS derives exact mapping and convex scores."""

    mapping = _payload(
        sota_mapping_positive_score=1,
        proposal_rows=[
            {"terminal": True, "false_acceptance": False, "model_family": "m1"},
            {"terminal": True, "false_acceptance": False, "model_family": "m2"},
        ],
        z3_rows=[{"terminal": True}, {"terminal": True}],
        enumeration_rows=[{"terminal": True}, {"terminal": True}],
        authority_agreement_rows=[
            {"authorities_agree": True, "terminal": True},
            {"authorities_agree": True, "terminal": True},
        ],
        paired_metric_rows=[{"paired_accuracy_delta": 1}],
        confidence_interval_rows=[
            {"model_family": "m1", "ci95_lower": 0.1},
            {"model_family": "m2", "ci95_lower": 0.2},
        ],
        smt_certification_run_complete_score=1,
    )
    mapping_row = exp.recompute_headlines(6957, mapping)[0]
    assert mapping_row["recomputed"] == 1
    assert mapping_row["agrees"] is True

    convex = _payload(
        convex_factor_positive_score=1,
        convex_factor_run_complete_score=1,
        seed_rows=[{"terminal": True}],
        jensen_rows=[{"passed": True}],
        finite_difference_rows=[{"passed": True}],
        projection_rows=[{"passed": True}],
        ordering_rows=[{"terminal": True}],
        label_isolation_rows=[{"passed": True}],
        shuffled_label_rows=[{"frozen_before_optimizer": True}],
        fresh_process_replay_rows=[{"replay_matches": True}],
        confidence_interval_rows=[{"ci95_lower": 0.1}, {"ci95_lower": 0.2}],
    )
    convex_row = exp.recompute_headlines(6958, convex)[0]
    assert convex_row["recomputed"] == 1
    convex["jensen_rows"] = [{"passed": False}]
    assert exp.recompute_headlines(6958, convex)[0]["recomputed"] == 0


@pytest.mark.parametrize(
    ("number", "field"),
    [
        (6959, "certified_energy_positive_score"),
        (6960, "audited_certified_energy_positive_score"),
    ],
)
def test_scenario_report_6964_rows_recompute_selection_controls(number: int, field: str) -> None:
    """SCENARIO-REPORT-6964-ROWS checks baseline, headroom, and audit controls."""

    payload = _positive_selection_payload(field)
    row = exp.recompute_headlines(number, payload)[0]
    assert row["recomputed"] == 1
    assert row["agrees"] is True
    assert row["checks"]["strongest_baseline_named"] is True

    payload["confidence_interval_rows"] = [{"ci95_lower": 0.0}]
    assert exp.recompute_headlines(number, payload)[0]["recomputed"] == 0


def test_scenario_report_6964_authorities_do_not_synthesize_scores() -> None:
    """REQ-REPORT-6964 copies only an admissible authoritative score."""

    payload = _positive_selection_payload("certified_energy_positive_score")
    payloads = {6959: payload}
    states = {
        6959: {"verdict_class": "positive", "evidence_state": "positive"},
        6958: {"verdict_class": "disqualified", "evidence_state": "flagged"},
    }
    values, rows = exp.authoritative_scores(payloads, states)

    assert values["certified_energy_positive_score"] == 1
    assert values["convex_factor_positive_score"] is None
    assert (
        next(row for row in rows if row["field"] == "convex_factor_positive_score")["eligible"]
        is False
    )

    states[6959] = {"verdict_class": "blocked", "evidence_state": "blocked"}
    assert exp.authoritative_scores(payloads, states)[0]["certified_energy_positive_score"] is None


def test_scenario_report_6964_queue_safety_requires_exact_rows() -> None:
    """SCENARIO-REPORT-6964-SAFETY checks every external-memory boundary."""

    queue = {
        "continuous_self_learning_task": True,
        "learning_tier": 2,
        "queue_learning_run_complete_score": 1,
        "write_rows": [{"certificate_authority": "z3_and_exact_enumeration", "passed": True}],
        "chronology_rows": [{"passed": True}],
        "debt_arrival_rows": [{"passed": True}],
        "debt_service_rows": [{"passed": True}],
        "debt_balance_rows": [{"passed": True}],
        "restart_rows": [{"passed": True, "hard_reset": True}],
        "retention_rows": [{"passed": True}],
        "rollback_rows": [{"passed": True}],
        "future_label_isolation_rows": [{"passed": True}],
        "model_hashes_before": {"m": "abc"},
        "model_hashes_after": {"m": "abc"},
        "no_model_weight_mutation": True,
    }
    rows = exp.build_safety_rows({6962: queue})
    queue_row = next(row for row in rows if row["number"] == 6962)
    assert queue_row["passed"] is True

    queue["debt_balance_rows"] = [{"passed": False}]
    assert (
        next(row for row in exp.build_safety_rows({6962: queue}) if row["number"] == 6962)[
            "debt_math"
        ]
        is False
    )
    assert (
        next(row for row in exp.build_safety_rows({}) if row["number"] == 6962)["available"]
        is False
    )


def test_req_report_6964_claim_boundaries_and_retirement() -> None:
    """SCENARIO-REPORT-6964-RETIREMENT preserves claim and manifest limits."""

    clean = exp.build_claim_boundary_rows([_task()], {6957: _payload()})[0]
    assert clean["passed"] is True
    claimed = exp.build_claim_boundary_rows(
        [_task()], {6957: _payload(hardware_speed_claimed=True)}
    )[0]
    assert claimed["passed"] is False

    task = _task()
    task["prior_failures"] = [
        {
            "experiment_id": "old-a",
            "verdict": "complete_null_same",
            "retire_if_same_verdict": True,
        },
        {
            "experiment_id": "old-b",
            "verdict": "complete_null_same",
            "retire_if_same_verdict": False,
        },
    ]
    states = {6957: {"task_id": task["task_id"], "honest_verdict": "complete_null_same"}}
    candidates = exp.exclusion_candidates([task], states)
    assert [row["prior_experiment_id"] for row in candidates] == ["old-a"]
    assert candidates[0]["manifest_edited"] is False


def test_req_report_6964_builds_current_complete_disqualified_capstone() -> None:
    """SCENARIO-REPORT-6964-COMPLETION completes classification without promotion."""

    artifact = exp.build_artifact(REPO, "20260904")

    assert artifact["v609_capstone_complete_score"] == 1
    assert artifact["status"] == "complete_disqualified"
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["honest_verdict"].startswith("complete_disqualified_")
    assert len(artifact["task_contract_rows"]) == 12
    assert len(artifact["task_state_rows"]) == 12
    assert len(artifact["v610_gap_rows"]) == 3
    states = {row["number"]: row for row in artifact["task_state_rows"]}
    assert states[6954]["verdict_class"] == "circular_positive"
    assert states[6958]["evidence_state"] == "flagged"
    assert states[6961]["verdict_class"] == "circular_positive"
    assert states[6962]["verdict_class"] == "blocked"
    assert states[6963]["artifact_state"] == "bootstrap_only"
    assert artifact["sota_mapping_positive_score"] == 0
    assert artifact["convex_factor_positive_score"] is None
    assert artifact["certified_energy_positive_score"] == 0
    assert artifact["audited_certified_energy_positive_score"] == 0
    assert artifact["queue_learning_positive_score"] is None
    assert artifact["audited_queue_learning_positive_score"] is None
    assert {
        "python/carnot/experiment_6964_v609_capstone.py",
        "scripts/experiments/experiment_6964_v609_capstone.py",
    } <= set(artifact["source_artifact_hashes"])
    assert exp.validate_artifact(artifact) == []


def test_req_report_6964_missing_core_contract_blocks(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6964-PREFLIGHT keeps a missing contract inspectable."""

    artifact = exp.build_artifact(tmp_path, "20260904")

    assert artifact["v609_capstone_complete_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == exp.BLOCKED_VERDICT
    assert artifact["gate_check_summary"]["failed_check"] == "core_contract_preconditions"
    assert exp.validate_artifact(artifact) == []


def test_req_report_6964_validation_and_defensive_inputs(tmp_path: Path) -> None:
    """REQ-REPORT-6964 rejects malformed inputs and changed summaries."""

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("- not-a-map\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML root is not a mapping"):
        exp.load_yaml(bad_yaml)

    assert exp.roadmap_tasks({"tasks": {}}) == []
    assert exp.roadmap_tasks({"tasks": [None, {"id": "exp1"}]})[0]["number"] == 1
    assert exp._gate_pass(1, "!=", 1) is False
    assert exp._gate_pass(None, ">=", 1) is False
    assert exp._scalar({"value": 3, "principle": "direct"}) == 3
    assert exp._parse_gates("none") == []
    assert exp._parse_gates("not a gate") == [{"unparsed": "not a gate"}]
    assert exp._all_rows({}, "optional", required=False) is True
    assert exp.recompute_headlines(6954, {})
    assert exp.recompute_headlines(7000, {}) == []

    log = "\n".join(
        [
            "ignored before activation",
            "| now | Milestone 2026.09.609 activated | OK | ready |",
            "| too short |",
            "| now | Unknown title | FAILED | no match |",
            "| now | Synthetic 6957 | FLAGGED | review |",
        ]
    )
    assert exp.parse_conductor_states(log, [_task()])[6957]["state"] == "flagged"
    assert exp._verdict_shape(_payload(status="flagged")) == "disqualified"
    assert exp._verdict_shape({}) == "partial"
    assert exp._prefix_verdict_class("flagged_adversarial") == "disqualified"
    assert exp._prefix_verdict_class("null_no_gain") == "null"
    assert (
        exp._verdict_shape(_payload(verifier_is_oracle=True, honest_verdict="positive_claim"))
        == "circular_positive"
    )
    assert (
        exp._verdict_shape(_payload(verifier_is_oracle=True, honest_verdict="complete_result"))
        == "circular_positive"
    )
    assert exp._critical(None) is False

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

    artifact = exp.build_artifact(REPO, "20260904")
    broken = deepcopy(artifact)
    broken["honest_verdict"] = "complete_null_wrong"
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
    broken["verdict_class"] = "partial"
    broken["honest_verdict"] = "complete_partial_wrong"
    broken["reproducibility_checksum"] = exp.reproducibility_checksum(broken)
    assert "complete prefix cannot use partial verdict_class" in exp.validate_artifact(broken)

    broken = deepcopy(artifact)
    broken["sota_mapping_positive_score"] = 1
    broken["reproducibility_checksum"] = exp.reproducibility_checksum(broken)
    assert "sota_mapping_positive_score does not match authoritative row" in exp.validate_artifact(
        broken
    )

    broken = deepcopy(artifact)
    broken["v610_gap_rows"] = []
    broken["reproducibility_checksum"] = exp.reproducibility_checksum(broken)
    assert "completed capstone requires exactly three V610 gaps" in exp.validate_artifact(broken)

    blocked = exp.build_artifact(tmp_path, "20260904")
    blocked["gate_check_summary"] = {}
    blocked["reproducibility_checksum"] = exp.reproducibility_checksum(blocked)
    assert "blocked verdict requires an exact failed gate check" in exp.validate_artifact(blocked)


def test_scenario_report_6964_safety_defensive_replay() -> None:
    """SCENARIO-REPORT-6964-SAFETY rejects malformed debt and write order."""

    assert exp._debt_math({}) is False
    numeric = {
        "debt_arrival_rows": [
            {"model_id": "m", "event_id": 1, "arrival": 1.0},
            {"model_id": "m", "event_id": 2, "arrival": 0.0},
        ],
        "debt_service_rows": [
            {"model_id": "m", "event_id": 1, "service": 0.0},
            {"model_id": "m", "event_id": 2, "service": 0.5},
        ],
        "debt_balance_rows": [
            {"model_id": "m", "event_id": 1, "balance": 1.0},
            {"model_id": "m", "event_id": 2, "balance": 0.5},
        ],
    }
    assert exp._debt_math(numeric) is True
    numeric["debt_balance_rows"][0]["balance"] = "bad"
    assert exp._debt_math(numeric) is False
    numeric["debt_balance_rows"][0]["balance"] = 2.0
    assert exp._debt_math(numeric) is False

    assert exp._exact_writes({}) is False
    assert (
        exp._exact_writes({"write_rows": [{"passed": False, "certificate_authority": "self"}]})
        is False
    )
    assert exp._exact_writes({"write_rows": [{"write_admitted": False}]}) is True
    assert (
        exp._exact_writes(
            {
                "write_rows": [
                    {
                        "write_admitted": True,
                        "raw_output_durable_step": "1",
                        "exact_outcome_visible_step": 2,
                        "write_step": 3,
                    }
                ]
            }
        )
        is False
    )
    assert (
        exp._exact_writes(
            {
                "write_rows": [
                    {
                        "write_admitted": True,
                        "raw_output_durable_step": 2,
                        "exact_outcome_visible_step": 1,
                        "write_step": 3,
                    }
                ]
            }
        )
        is False
    )
    assert (
        exp._exact_writes(
            {
                "write_rows": [
                    {
                        "write_admitted": True,
                        "raw_output_durable_step": 1,
                        "exact_outcome_visible_step": 2,
                        "write_step": 3,
                    }
                ]
            }
        )
        is True
    )

    audit = {
        field: [{"passed": True}]
        for field in (
            "chronology_rows",
            "certificate_authority_rows",
            "future_label_isolation_rows",
            "write_order_rows",
            "debt_recompute_rows",
            "retention_rows",
            "restart_rows",
            "rollback_rows",
            "model_immutability_rows",
            "aggregate_consistency_rows",
        )
    }
    audit["field_principles"] = {"rows": "measured units"}
    audit_row = next(row for row in exp.build_safety_rows({6963: audit}) if row["number"] == 6963)
    assert audit_row["passed"] is True


def test_scenario_report_6964_boundary_and_outcome_reducers() -> None:
    """SCENARIO-REPORT-6964-VERDICT closes failed safety and outcome states."""

    states = {
        6957: {
            "verdict_class": "positive",
            "evidence_state": "positive",
            "admissible": True,
        }
    }
    checked = exp.apply_boundary_checks(
        states,
        [{"number": 6957, "passed": False}],
        [],
    )
    assert checked[6957]["verdict_class"] == "disqualified"
    assert checked[6957]["admissible"] is False

    assert exp.capstone_outcome(False, False, False)[1] == "disqualified"
    assert exp.capstone_outcome(True, True, False)[1] == "disqualified"
    assert exp.capstone_outcome(True, False, True)[1] == "null"
    assert exp.capstone_outcome(True, False, False)[1] == "positive"


def test_scenario_report_6964_missing_science_remains_runnable(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6964-STATES classifies absent science without rerunning it."""

    design = tmp_path / exp.DESIGN_PATH
    design.parent.mkdir(parents=True)
    design.write_text((REPO / exp.DESIGN_PATH).read_text(encoding="utf-8"), encoding="utf-8")
    roadmap = tmp_path / exp.ROADMAP_PATH
    roadmap.write_text((REPO / exp.ROADMAP_PATH).read_text(encoding="utf-8"), encoding="utf-8")

    artifact = exp.build_artifact(tmp_path, "20260904")

    states = {row["number"]: row for row in artifact["task_state_rows"]}
    assert states[6953]["artifact_state"] == "missing"
    assert artifact["v609_capstone_complete_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert exp.validate_artifact(artifact) == []


def test_req_report_6964_checker_errors_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6964 preserves checker failures and writes explicit targets."""

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

    monkeypatch.undo()
    target = tmp_path / "exp6964.json"
    assert exp.main(["--repo-root", str(REPO), "--date", "20260904", "--output", str(target)]) == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert exp.validate_artifact(payload) == []
    assert exp.main(["--validate", "--output", str(target)]) == 0
    target.write_text("{}", encoding="utf-8")
    assert exp.main(["--validate", "--output", str(target)]) == 1

    monkeypatch.setattr(exp, "build_artifact", lambda *_: {})
    assert exp.main(["--repo-root", str(REPO), "--date", "20260904", "--output", str(target)]) == 1
