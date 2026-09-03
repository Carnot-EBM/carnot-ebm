"""Tests for the cold V605 evidence capstone.

Spec refs: REQ-REPORT-6922 and SCENARIO-REPORT-6922-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6922_v605_independent_capstone as exp


REPO = Path(__file__).resolve().parents[2]


def _task(number: int = 6912) -> dict[str, object]:
    return {
        "number": number,
        "task_id": f"exp{number}-synthetic",
        "title": f"Synthetic {number}",
        "deliverable": f"results/experiment_{number}_synthetic.json",
        "gates": [],
        "prior_failures": [],
    }


def _payload(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "status": "complete",
        "honest_verdict": "complete_positive_synthetic",
        "verdict_class": "positive",
        "verifier_is_oracle": False,
        "rows": [{"score": 1.0}],
    }
    value.update(updates)
    return value


def _clean_adversarial() -> dict[str, object]:
    return {"flag_count": 0, "max_severity": 0, "flags": []}


def test_req_report_6922_spec_precedes_implementation() -> None:
    """REQ-REPORT-6922 owns the complete artifact and state contract."""

    text = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("REQ-REPORT-6922", 1)[1]

    assert {
        "SCENARIO-REPORT-6922-STATE",
        "SCENARIO-REPORT-6922-ADMISSIBILITY",
        "SCENARIO-REPORT-6922-CIRCULARITY",
        "SCENARIO-REPORT-6922-ROWS",
        "SCENARIO-REPORT-6922-TAINT",
        "SCENARIO-REPORT-6922-RETIREMENT",
        "SCENARIO-REPORT-6922-PROMOTION",
    } <= set(exp.spec_anchors(section))
    assert exp.INFERENCE_SUBSTRATE in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in section


@pytest.mark.parametrize(
    ("conductor", "artifact_state", "payload", "adversarial", "row_check", "state", "klass"),
    [
        ({"state": "not_recorded"}, "absent", None, None, None, "absent", "blocked"),
        (
            {
                "state": "preemptive_skip",
                "honest_verdict": "blocked_preemptively_upstream_retired",
            },
            "absent",
            None,
            None,
            None,
            "skipped",
            "blocked",
        ),
        ({"state": "ok"}, "stale", _payload(), _clean_adversarial(), ("ok", []), "stale", "disqualified"),
        (
            {"state": "flagged"},
            "present",
            _payload(),
            _clean_adversarial(),
            ("ok", []),
            "flagged",
            "disqualified",
        ),
        (
            {"state": "gate_blocked", "honest_verdict": "blocked_gate_check_failed"},
            "present",
            _payload(honest_verdict="blocked_gate_check_failed", verdict_class="blocked"),
            _clean_adversarial(),
            ("skipped", ["no recognised row container"]),
            "blocked",
            "blocked",
        ),
        (
            {"state": "ok"},
            "present",
            _payload(honest_verdict="complete_null_no_effect", verdict_class="null"),
            _clean_adversarial(),
            ("ok", []),
            "null",
            "null",
        ),
        (
            {"state": "ok"},
            "present",
            _payload(verifier_is_oracle=True, verdict_class="circular_positive"),
            _clean_adversarial(),
            ("ok", []),
            "circular_positive",
            "circular_positive",
        ),
        (
            {"state": "ok"},
            "present",
            _payload(flagged_adversarial=True),
            _clean_adversarial(),
            ("ok", []),
            "flagged",
            "disqualified",
        ),
        (
            {"state": "ok"},
            "present",
            _payload(),
            {"flag_count": 1, "max_severity": 2, "flags": [{"severity": "critical"}]},
            ("ok", []),
            "flagged",
            "disqualified",
        ),
        (
            {"state": "ok"},
            "present",
            _payload(),
            _clean_adversarial(),
            ("findings", ["WINS_NOT_EXCEEDING_LOSSES"]),
            "row_disagreement",
            "disqualified",
        ),
        (
            {"state": "ok"},
            "present",
            _payload(verifier_is_oracle=True, verdict_class="positive"),
            _clean_adversarial(),
            ("ok", []),
            "wrong_verdict_class",
            "disqualified",
        ),
    ],
)
def test_scenario_report_6922_state_and_admissibility_classes(
    conductor: dict[str, object],
    artifact_state: str,
    payload: dict[str, object] | None,
    adversarial: dict[str, object] | None,
    row_check: tuple[str, list[str]] | None,
    state: str,
    klass: str,
) -> None:
    """SCENARIO-REPORT-6922-STATE keeps every evidence state distinct."""

    row = exp.classify_evidence(
        _task(), conductor, artifact_state, payload, adversarial, row_check
    )

    assert row["evidence_state"] == state
    assert row["verdict_class"] == klass
    assert row["admissible"] is (klass in {"positive", "circular_positive", "null"})


def test_scenario_report_6922_contract_is_independent_and_exact() -> None:
    """REQ-REPORT-6922 compares the primary document and active YAML."""

    design = (REPO / exp.DESIGN_PATH).read_text(encoding="utf-8")
    roadmap = exp.load_yaml(REPO / exp.ROADMAP_PATH)
    rows = exp.build_contract_rows(design, roadmap)

    assert len(rows) == 12
    assert all(row["passed"] for row in rows)
    assert [row["task_id"] for row in rows] == [row["task_id"] for row in exp.EXPECTED_TASKS]
    assert rows[-1]["gates_match"] is True
    assert rows[-1]["prompt_ending_match"] is True
    assert rows[9]["required_model_ids_match"] is True

    changed = deepcopy(roadmap)
    changed["tasks"][9]["prompt"] = changed["tasks"][9]["prompt"].replace(
        exp.REQUIRED_MODEL_IDS[0], "wrong/model", 1
    )
    assert exp.build_contract_rows(design, changed)[9]["passed"] is False


def test_scenario_report_6922_rows_recompute_headlines() -> None:
    """SCENARIO-REPORT-6922-ROWS catches a reported event-count mismatch."""

    bank = {
        "qualified_model_relation_event_count": 9,
        "qualified_relation_event_bank_ready_score": 1,
        "admitted_event_rows": [
            {"model_produced": True, "family": "graph_coloring"},
            {"model_produced": True, "family": "scheduling"},
            {"model_produced": False, "family": "graph_coloring"},
        ],
    }
    generation = {
        "guided_generation_run_complete_score": 1,
        "exact_guidance_utility_score": 1,
        "per_model_arm_rows": [
            {"model_spec": model, "arm": arm, "cell_count": 30}
            for model in exp.REQUIRED_MODEL_IDS
            for arm in exp.GENERATION_ARMS
        ],
        "validity_delta_rows": [
            {
                "model_spec": model,
                "validity_delta": 0.0,
                "parse_failure_did_not_rise": True,
                "no_regression_over_0_02": True,
            }
            for model in exp.REQUIRED_MODEL_IDS
        ],
        "per_family_arm_rows": [{"family": family} for family in exp.REQUIRED_FAMILIES],
    }

    bank_rows = exp.recompute_headlines(6915, bank)
    generation_rows = exp.recompute_headlines(6920, generation)

    assert {row["field"]: row["recomputed"] for row in bank_rows} == {
        "qualified_model_relation_event_count": 2,
        "qualified_relation_event_bank_ready_score": 0,
    }
    assert all(row["agrees"] is False for row in bank_rows)
    assert {row["field"]: row["recomputed"] for row in generation_rows}[
        "exact_guidance_utility_score"
    ] == 0


def test_scenario_report_6922_taint_is_transitive_and_branch_local() -> None:
    """SCENARIO-REPORT-6922-TAINT follows dependencies but not siblings."""

    states = [
        {"task_id": "a", "evidence_state": "flagged", "verdict_class": "disqualified"},
        {"task_id": "b", "evidence_state": "positive", "verdict_class": "positive"},
        {"task_id": "c", "evidence_state": "positive", "verdict_class": "positive"},
        {"task_id": "d", "evidence_state": "positive", "verdict_class": "positive"},
    ]
    rows = exp.propagate_dependency_taint(states, {"b": ["a"], "c": ["b"], "d": []})

    by_target = {row["target_task_id"]: row for row in rows}
    assert by_target["b"]["root_task_ids"] == ["a"]
    assert by_target["c"]["root_task_ids"] == ["a"]
    assert "d" not in by_target


def test_scenario_report_6922_repeated_prior_verdict_retires() -> None:
    """SCENARIO-REPORT-6922-RETIREMENT emits actions only for exact repeats."""

    task = _task(6916)
    task["prior_failures"] = [
        {
            "experiment_id": "old-a",
            "verdict": "blocked_gate_check_failed",
            "addressed_by": "A changed prerequisite.",
            "retire_if_same_verdict": True,
        },
        {
            "experiment_id": "old-b",
            "verdict": "complete_null_old",
            "addressed_by": "A changed method.",
            "retire_if_same_verdict": True,
        },
    ]
    states = [
        {
            "task_id": task["task_id"],
            "honest_verdict": "blocked_gate_check_failed",
            "verdict_class": "blocked",
        }
    ]

    comparisons, actions = exp.compare_prior_verdicts([task], states)

    assert [row["verdict_repeated"] for row in comparisons] == [True, False]
    assert len(actions) == 1
    assert actions[0]["prior_experiment_id"] == "old-a"


def test_scenario_report_6922_false_promotion_guard() -> None:
    """SCENARIO-REPORT-6922-PROMOTION rejects circular and null promotion."""

    claims = [
        exp.claim_row("exact fixture passed", "circular_positive", promoted=False),
        exp.claim_row("exact guidance improved", "null", promoted=True),
        exp.claim_row("audit complete", "positive", promoted=True, scientific=False),
    ]

    assert exp.false_promotion_count(claims) == 1


def test_req_report_6922_current_artifact_is_complete_and_honest() -> None:
    """REQ-REPORT-6922 produces a replayable terminal synthesis."""

    artifact = exp.build_artifact(REPO, "20260903")

    assert artifact["v605_capstone_complete_score"] == 1
    assert artifact["false_promotion_count"] == 0
    assert artifact["honest_verdict"].startswith("complete_")
    assert len(artifact["task_state_rows"]) == 12
    assert {row["task_id"] for row in artifact["missing_artifact_rows"]} == {
        "exp6917-bounded-relation-memory-continuous-learning",
        "exp6918-relation-learning-cold-support-audit",
    }
    assert {row["task_id"] for row in artifact["skipped_task_rows"]} == {
        "exp6917-bounded-relation-memory-continuous-learning",
        "exp6918-relation-learning-cold-support-audit",
    }
    assert {row["disposition"] for row in artifact["branch_disposition_rows"]} <= {
        "adopt",
        "continue",
        "retire",
        "block",
    }
    assert len(artifact["branch_disposition_rows"]) == 4
    assert exp.validate_artifact(artifact) == []


def test_req_report_6922_blocked_preconditions_and_validation(tmp_path: Path) -> None:
    """REQ-REPORT-6922 writes a complete block when a global source is absent."""

    artifact = exp.build_artifact(tmp_path, "20260903")

    assert artifact["v605_capstone_complete_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == exp.BLOCKED_VERDICT
    assert artifact["gate_check_summary"]["failed_check"]
    assert exp.validate_artifact(artifact) == []

    broken = deepcopy(artifact)
    broken["false_promotion_count"] = 1
    broken["reproducibility_checksum"] = exp.reproducibility_checksum(broken)
    assert "false_promotion_count must be zero" in exp.validate_artifact(broken)


def test_req_report_6922_cli_writes_and_validates(tmp_path: Path) -> None:
    """REQ-REPORT-6922 exposes the required command-line writer."""

    target = tmp_path / "exp6922.json"
    assert exp.main(["--repo-root", str(REPO), "--date", "20260903", "--output", str(target)]) == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert exp.validate_artifact(payload) == []
    assert exp.main(["--validate", "--output", str(target)]) == 0
    target.write_text("{}", encoding="utf-8")
    assert exp.main(["--validate", "--output", str(target)]) == 1
