"""Tests for the V590 terminal branch disposition.

Spec refs: REQ-REPORT-6780 and SCENARIO-REPORT-6780-*.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

from carnot import experiment_6780_v590_branch_disposition as exp


REPO = Path(__file__).resolve().parents[2]


def _record(payload: dict | None = None, state: str = "present") -> dict:
    return {
        "artifact_state": state,
        "valid_json": state == "present",
        "payload": payload,
        "sha256": "sha256:" + "a" * 64 if payload is not None else None,
        "path": "results/source.json",
        "error": None,
    }


def _empty_sources() -> dict[str, dict]:
    return {task_id: _record(None, "missing") for task_id in exp.SHORT_TASK_IDS}


def _audit_bundle() -> dict:
    return {
        "summaries": [],
        "adversarial_findings": [],
        "verdict_row_consistency_findings": [],
        "recurring_blockers": {
            "window": 14,
            "blocked_task_count": 0,
            "diagnostic_coverage": {},
            "recurring": [],
        },
    }


def test_spec_anchor_and_required_fields_exist() -> None:
    """REQ-REPORT-6780: the reporting spec owns the full artifact contract."""

    text = (REPO / exp.REPORT_SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("REQ-REPORT-6780", 1)[1]
    for token in (
        "SCENARIO-REPORT-6780-PRECONDITIONS",
        "SCENARIO-REPORT-6780-ROW-RECOMPUTATION",
        "SCENARIO-REPORT-6780-BRANCH-ISOLATION",
        "SCENARIO-REPORT-6780-VALIDATION",
    ):
        assert token in section
    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(set(section.split("`")))


def test_planned_tasks_come_from_the_activated_v590_roadmap() -> None:
    """SCENARIO-REPORT-6780-PRECONDITIONS: all thirteen IDs stay visible."""

    planned, receipt = exp.load_planned_tasks(REPO)
    assert [row["task_id"] for row in planned] == list(exp.SHORT_TASK_IDS)
    assert [row["manifest_task_id"] for row in planned] == list(exp.EXPECTED_TASK_IDS)
    assert len({row["path"] for row in planned}) == 13
    assert receipt["next_roadmap_present"] is False
    assert receipt["selected_path"] == "research-roadmap.yaml"


def test_design_parser_fails_closed_on_wrong_or_incomplete_design() -> None:
    """REQ-REPORT-6780: an unrelated or incomplete design cannot define the run."""

    with pytest.raises(ValueError, match="V590 design milestone"):
        exp.parse_design_tasks("**Milestone:** `2026.08.589`")
    with pytest.raises(ValueError, match="deliverable missing"):
        exp.parse_design_tasks("**Milestone:** `2026.08.590`\n### Exp 6768: x")
    with pytest.raises(ValueError, match="invalid V590 task id"):
        exp.short_task_id("task-6768")


def test_source_loader_preserves_present_missing_invalid_and_current(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6780-PRECONDITIONS: bad inputs stay explicit rows."""

    planned = [
        {"task_id": "exp6768", "path": "results/good.json"},
        {"task_id": "exp6769", "path": "results/bad.json"},
        {"task_id": "exp6770", "path": "results/missing.json"},
        {"task_id": "exp6771", "path": "results/running.json"},
        {"task_id": "exp6780", "path": exp.RESULT_PATH.as_posix()},
    ]
    (tmp_path / "results").mkdir()
    (tmp_path / "results/good.json").write_text('{"status":"complete"}', encoding="utf-8")
    (tmp_path / "results/bad.json").write_text("[]", encoding="utf-8")
    (tmp_path / "results/running.json").write_text(
        '{"status":"running","rows":[{"metric":1}]}', encoding="utf-8"
    )
    sources = exp.load_source_artifacts(tmp_path, planned)
    assert sources["exp6768"]["artifact_state"] == "present"
    assert sources["exp6769"]["artifact_state"] == "invalid"
    assert sources["exp6770"]["artifact_state"] == "missing"
    assert sources["exp6771"]["artifact_state"] == "nonterminal"
    assert sources["exp6780"]["artifact_state"] == "current_synthesis"


def test_proof_and_repair_headlines_recompute_from_paired_rows() -> None:
    """SCENARIO-REPORT-6780-ROW-RECOMPUTATION: proof claims replay from rows."""

    sources = _empty_sources()
    sources["exp6768"] = _record(
        {"rows": [{"row_id": str(i)} for i in range(4)], "targetable_panel_ready": True}
    )
    sources["exp6769"] = _record(
        {
            "rows": [{"runtime_invoked": True}, {"runtime_invoked": True}],
            "dynamic_proof_grammar_ready": True,
            "valid_sat_reachable": True,
            "valid_unsat_reachable": True,
            "no_ghost_violations": 0,
        }
    )
    sources["exp6770"] = _record(
        {
            "rows": [
                {"pair_id": "a", "arm": "repaired_direct", "exact_valid": False},
                {"pair_id": "a", "arm": "dccd_environment", "exact_valid": True},
                {"pair_id": "b", "arm": "repaired_direct", "exact_valid": True},
                {"pair_id": "b", "arm": "dccd_environment", "exact_valid": True},
            ],
            "proof_transport_ab_completed": True,
        }
    )
    sources["exp6772"] = _record(
        {
            "rows": [
                {
                    "pair_id": "a",
                    "arm": "full_regeneration",
                    "exact_valid": False,
                    "harmful_flip": False,
                },
                {
                    "pair_id": "a",
                    "arm": "prefix_backtracking",
                    "exact_valid": True,
                    "harmful_flip": False,
                },
                {
                    "pair_id": "b",
                    "arm": "full_regeneration",
                    "exact_valid": True,
                    "harmful_flip": False,
                },
                {
                    "pair_id": "b",
                    "arm": "prefix_backtracking",
                    "exact_valid": False,
                    "harmful_flip": True,
                },
            ]
        }
    )
    proof = exp.recompute_proof(sources)
    repair = exp.recompute_repair(sources)
    assert proof["exact_valid_rate_by_arm"]["dccd_environment"] == {
        "numerator": 2,
        "denominator": 2,
        "rate": 1.0,
    }
    assert (
        proof["paired_exact_valid_effects"]["dccd_environment-minus-repaired_direct"]["mean_delta"]
        == 0.5
    )
    assert repair["paired_exact_valid_effect"]["pair_count"] == 2
    assert repair["paired_exact_valid_effect"]["mean_delta"] == 0.0
    assert repair["harmful_flips_by_arm"]["prefix_backtracking"]["numerator"] == 1


def test_missing_comparative_rows_are_null_not_zero() -> None:
    """SCENARIO-REPORT-6780-ROW-RECOMPUTATION: prose cannot fill missing rows."""

    sources = _empty_sources()
    sources["exp6770"] = _record(
        {"rows": [], "paired_exact_valid_deltas": {"dccd_environment-minus-repaired_direct": 0.0}}
    )
    sources["exp6772"] = _record({"rows": []})
    proof = exp.recompute_proof(sources)
    repair = exp.recompute_repair(sources)
    assert proof["paired_exact_valid_effects"]["cause"] == "no_eligible_comparative_rows"
    assert proof["paired_exact_valid_effects"]["value"] is None
    assert repair["paired_exact_valid_effect"]["value"] is None


def test_missing_memory_and_arc_rows_are_null_not_measured_zero() -> None:
    """SCENARIO-REPORT-6780-ROW-RECOMPUTATION: missing counts stay unknown."""

    sources = _empty_sources()
    memory = exp.recompute_continuous_memory(sources)
    arc = exp.recompute_arc(sources)
    assert memory["prequential_yield_by_arm"] == {
        "value": None,
        "denominator": 0,
        "cause": "no_eligible_prospective_rows",
    }
    assert memory["transaction_activity"]["commits"] is None
    assert memory["transaction_activity"]["cause"] == "no_eligible_prospective_rows"
    assert arc["row_recomputed_firings"] is None
    assert arc["supervisor_evidence_cause"] == "no_eligible_live_supervisor_rows"
    assert arc["tool_gap_event_count"] is None
    assert arc["tool_gap_event_count_cause"] == "tool_gap_events_missing"


def test_continuous_memory_recomputes_order_effects_activity_and_loss() -> None:
    """REQ-REPORT-6780: FR11 uses order rows and every cold audit gate."""

    sources = _empty_sources()
    sources["exp6774"] = _record(
        {
            "rows": [
                {
                    "order_id": "o1",
                    "arm": "no_memory",
                    "prequential_yield": 0.4,
                    "historical_loss": 0.6,
                    "commits": 0,
                    "rejects": 0,
                    "retrieval_count": 0,
                    "action_influence_count": 0,
                },
                {
                    "order_id": "o1",
                    "arm": "procedural_memory",
                    "prequential_yield": 0.7,
                    "historical_loss": 0.3,
                    "commits": 2,
                    "rejects": 1,
                    "retrieval_count": 2,
                    "action_influence_count": 1,
                },
                {
                    "order_id": "o2",
                    "arm": "no_memory",
                    "prequential_yield": 0.5,
                    "historical_loss": 0.5,
                    "commits": 0,
                    "rejects": 0,
                    "retrieval_count": 0,
                    "action_influence_count": 0,
                },
                {
                    "order_id": "o2",
                    "arm": "procedural_memory",
                    "prequential_yield": 0.6,
                    "historical_loss": 0.4,
                    "commits": 1,
                    "rejects": 2,
                    "retrieval_count": 1,
                    "action_influence_count": 1,
                },
            ],
            "prospective_csl_completed": True,
        }
    )
    sources["exp6775"] = _record(
        {
            "cold_audit_completed": True,
            "audit_gates": {name: True for name in exp.FR11_GATES},
        }
    )
    memory = exp.recompute_continuous_memory(sources)
    assert memory["procedural_minus_no_memory_order_effect"]["mean_delta"] == pytest.approx(0.2)
    assert memory["historical_loss_by_arm"]["procedural_memory"]["value"] == 0.35
    assert memory["transaction_activity"] == {
        "commits": 3,
        "rejects": 3,
        "rollbacks": 0,
        "retrievals": 3,
        "action_influences": 2,
    }
    assert memory["required_positive_gates"] == {name: True for name in exp.FR11_GATES}


def test_arc_adoption_requires_cold_actions_to_progress_not_transport() -> None:
    """REQ-REPORT-6780: ARC transport cannot stand in for action efficiency."""

    sources = _empty_sources()
    sources["exp6776"] = _record(
        {
            "rows": [{"arm_fired": 1, "arm_helped_counterfactual": 0}],
            "shadow_supervisor_transport_ready": True,
            "firings_after_by_arm": {"drop_goal_bias": 2},
            "evidence_floor_met_by_arm": {"drop_goal_bias": False},
        }
    )
    sources["exp6777"] = _record(
        {"tool_gap_transport_ready": True, "tool_gap_events": [], "analyzer_ingest_passed": True}
    )
    transport_only = exp.recompute_arc(sources)
    assert transport_only["tool_gap_transport_ready"] is True
    assert transport_only["adoption_positive"] is False

    sources["exp6778"] = _record(
        {
            "rows": [
                {"pair_id": "a", "arm": "control_unset", "actions_to_progress": 12},
                {"pair_id": "a", "arm": "selfparse", "actions_to_progress": 8},
                {"pair_id": "b", "arm": "control_unset", "actions_to_progress": 10},
                {"pair_id": "b", "arm": "selfparse", "actions_to_progress": 9},
            ],
            "actions_to_progress_ab_completed": True,
        }
    )
    sources["exp6779"] = _record(
        {"cold_actions_to_progress_audit_passed": True, "adoption_decision": "promote"}
    )
    arc = exp.recompute_arc(sources)
    assert arc["selfparse_minus_control_actions_effect"]["mean_delta"] == -2.5
    assert arc["adoption_positive"] is True

    worse = copy.deepcopy(sources)
    worse["exp6778"] = _record(
        {
            "rows": [
                {"pair_id": "a", "arm": "control_unset", "actions_to_progress": 8},
                {"pair_id": "a", "arm": "selfparse", "actions_to_progress": 12},
            ],
            "actions_to_progress_ab_completed": True,
        }
    )
    assert exp.recompute_arc(worse)["adoption_positive"] is False


def test_fr11_and_fr12_positive_credit_requires_cold_and_safety_gates() -> None:
    """REQ-REPORT-6780: positive branch credit requires every stated gate."""

    sources = _empty_sources()
    sources["exp6774"] = _record(
        {
            "prospective_csl_completed": True,
            "rows": [
                {
                    "order_id": "o1",
                    "arm": "no_memory",
                    "prequential_yield": 0.4,
                    "commits": 0,
                    "rejects": 0,
                    "retrieval_count": 0,
                    "action_influence_count": 0,
                },
                {
                    "order_id": "o1",
                    "arm": "procedural_memory",
                    "prequential_yield": 0.7,
                    "commits": 1,
                    "rejects": 1,
                    "retrieval_count": 1,
                    "action_influence_count": 1,
                },
            ],
        }
    )
    sources["exp6775"] = _record(
        {
            "cold_audit_completed": False,
            "audit_gates": {name: True for name in exp.FR11_GATES},
        }
    )
    memory = exp.recompute_continuous_memory(sources)
    fr11 = exp.build_fr11_disposition({"continuous_memory": memory}, "null")
    assert all(memory["required_positive_gates"].values())
    assert fr11["positive"] is False

    sources["exp6770"] = _record(
        {
            "proof_transport_ab_completed": True,
            "rows": [
                {"pair_id": "p1", "arm": "repaired_direct", "exact_valid": False},
                {"pair_id": "p1", "arm": "dccd_environment", "exact_valid": True},
            ],
        }
    )
    sources["exp6772"] = _record(
        {
            "rows": [
                {
                    "pair_id": "p1",
                    "arm": "full_regeneration",
                    "exact_valid": False,
                    "harmful_flip": False,
                    "support_loss": False,
                },
                {
                    "pair_id": "p1",
                    "arm": "prefix_backtracking",
                    "exact_valid": True,
                    "harmful_flip": True,
                    "support_loss": False,
                },
            ]
        }
    )
    headlines = exp.recompute_headlines(sources)
    fr12 = exp.build_fr12_disposition(headlines, "null")
    assert headlines["repair"]["paired_exact_valid_effect"]["mean_delta"] == 1.0
    assert headlines["repair"]["paired_harmful_flip_effect"]["mean_delta"] == 1.0
    assert fr12["positive"] is False


def test_actual_source_rows_preserve_terminal_and_missing_states() -> None:
    """SCENARIO-REPORT-6780-PRECONDITIONS: the live V590 roster is complete."""

    planned, _ = exp.load_planned_tasks(REPO)
    sources = exp.load_source_artifacts(REPO, planned)
    rows = exp.build_experiment_rows(planned, sources)
    assert len(rows) == 13
    assert {row["task_id"] for row in rows} == set(exp.SHORT_TASK_IDS)
    assert {row["task_id"] for row in rows if row["artifact_state"] == "missing"} == {
        "exp6772",
        "exp6774",
        "exp6775",
        "exp6778",
        "exp6779",
    }
    assert next(row for row in rows if row["task_id"] == "exp6770")["verdict_class"] == "blocked"
    assert next(row for row in rows if row["task_id"] == "exp6780")["verdict_class"] == "partial"


def test_branch_and_gap_classification_keeps_science_independent() -> None:
    """SCENARIO-REPORT-6780-BRANCH-ISOLATION: no branch vote is possible."""

    planned, _ = exp.load_planned_tasks(REPO)
    sources = exp.load_source_artifacts(REPO, planned)
    rows = exp.build_experiment_rows(planned, sources)
    headlines = exp.recompute_headlines(sources)
    branches = exp.build_branch_rows(rows, headlines, [], [])
    assert [row["branch"] for row in branches] == list(exp.BRANCH_ORDER)
    assert {row["branch"]: row["verdict_class"] for row in branches} == {
        "proof": "partial",
        "continuous_memory": "blocked",
        "arc": "blocked",
        "execution_contract": "partial",
    }
    assert all(row["next_action"] for row in branches)
    assert "pooled_milestone_success_score" not in headlines


def test_prior_recurrences_name_mechanical_retirements() -> None:
    """REQ-REPORT-6780: repeated retire-if-same routes become exclusions."""

    planned, _ = exp.load_planned_tasks(REPO)
    sources = exp.load_source_artifacts(REPO, planned)
    rows = exp.build_experiment_rows(planned, sources)
    recurrences = exp.build_prior_verdict_recurrences(planned, rows, exp.HONEST_VERDICT)
    retirements = exp.build_retirement_recommendations(recurrences)
    assert any(
        row["prior_experiment_id"] == "exp6767-v589-branch-disposition" for row in recurrences
    )
    assert any(row["current_task_id"] == "exp6780" for row in retirements)
    assert all(row["recommendation"] == "add_to_exclusion_manifest" for row in retirements)


def test_audit_helpers_preserve_outputs_and_row_findings(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-6780-VALIDATION: audit output is retained as evidence."""

    sources = {"exp6768": _record({"rows": [{"metric": 1}]})}
    sources["exp6768"]["path"] = exp.TASK_PATHS["exp6768"]

    def fake_run(args: list[str], root: Path) -> tuple[int, str]:
        assert root == REPO
        assert "summarize_artifact.py" in " ".join(args)
        return 1, "warning: stale-stamp"

    monkeypatch.setattr(exp, "_run_command", fake_run)
    summaries = exp.run_summaries(REPO, sources, ("exp6768",))
    assert summaries[0]["exit_code"] == 1
    assert summaries[0]["summary_excerpt"] == "warning: stale-stamp"

    findings = exp.run_row_consistency(REPO, sources, ("exp6768",))
    assert findings[0]["status"] in {"ok", "findings"}
    ledger = exp.run_recurring_blockers(window=1)
    assert ledger["window"] == 1


def test_artifact_build_is_valid_deterministic_and_honest() -> None:
    """REQ-REPORT-6780: the terminal artifact satisfies the closed schema."""

    artifact = exp.build_artifact(REPO, "20260830", audit_bundle=_audit_bundle(), duration_s=1.25)
    assert exp.validate_artifact(artifact, REPO) == []
    assert artifact["milestone"] == "2026.08.590"
    assert artifact["honest_verdict"].startswith("complete_partial")
    assert artifact["verdict_class"] == "partial"
    assert len(artifact["rows"]) == 13
    assert len(artifact["branch_rows"]) == 4
    assert len(artifact["prd_gap_disposition"]) == 3
    assert artifact["fr11_disposition"]["positive"] is False
    assert artifact["arc_disposition"]["adoption_positive"] is False
    assert artifact["verifier_is_oracle"] is False
    assert all(not row["updated"] for row in artifact["docs_reconciled"])
    assert {row["path"] for row in artifact["protected_files_unchanged"]} >= {
        "scripts/research_conductor.py",
        "research-roadmap.yaml",
    }

    changed_duration = exp.build_artifact(
        REPO, "20260830", audit_bundle=_audit_bundle(), duration_s=9.5
    )
    assert changed_duration["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_self_audit_reconciliation_preserves_critical_and_row_findings() -> None:
    """SCENARIO-REPORT-6780-VALIDATION: the capstone keeps its own audit flags."""

    artifact = exp.build_artifact(REPO, "20260830", audit_bundle=_audit_bundle(), duration_s=1.0)
    self_adversarial = {
        "artifact": str(REPO / exp.RESULT_PATH),
        "loaded": True,
        "exp_id": 6780,
        "flag_count": 2,
        "flags": [
            {
                "kind": "NONTERMINAL_DECLARED_ARTIFACT",
                "severity": "critical",
                "detail": "partial marker present",
            },
            {
                "kind": "VERDICT_PREFIX_CLASS_CONTRADICTION",
                "severity": "warn",
                "detail": "complete_partial conflicts with partial",
            },
        ],
    }
    self_row = {
        "task_id": "exp6780",
        "artifact": str(REPO / exp.RESULT_PATH),
        "status": "findings",
        "findings": ["ALL_ROWS_NULL: disposition rows are not outcome rows"],
        "blocking_count": 1,
        "warning_count": 0,
    }
    reconciled = exp.reconcile_self_audits(
        artifact,
        self_adversarial,
        self_row,
        REPO,
        duration_s=2.0,
    )
    assert reconciled["flagged_adversarial"] is True
    assert reconciled["corrigendum_pending"][0]["kind"] == "NONTERMINAL_DECLARED_ARTIFACT"
    assert reconciled["adversarial_findings"][-1] == self_adversarial
    assert reconciled["verdict_row_consistency_findings"][-1] == self_row
    execution = next(
        row for row in reconciled["branch_rows"] if row["branch"] == "execution_contract"
    )
    assert execution["adversarial_finding_count"] == 2
    assert execution["row_consistency_finding_count"] == 1
    assert reconciled["duration_s"] == 2.0
    assert exp.validate_artifact(reconciled, REPO) == []


def test_validator_rejects_schema_enum_rows_prefix_and_checksum() -> None:
    """REQ-REPORT-6780: invalid synthesis claims fail closed."""

    artifact = exp.build_artifact(REPO, "20260830", audit_bundle=_audit_bundle(), duration_s=1.0)
    cases = []
    missing = copy.deepcopy(artifact)
    missing.pop("rows")
    cases.append((missing, "missing_fields"))
    bad_class = copy.deepcopy(artifact)
    bad_class["branch_rows"][0]["verdict_class"] = "mystery"
    cases.append((bad_class, "branch_rows_closed_class"))
    bad_rows = copy.deepcopy(artifact)
    bad_rows["rows"] = bad_rows["rows"][:-1]
    cases.append((bad_rows, "expected_task_rows"))
    bad_prefix = copy.deepcopy(artifact)
    bad_prefix["honest_verdict"] = "partial result"
    cases.append((bad_prefix, "terminal_prefix"))
    bad_checksum = copy.deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    cases.append((bad_checksum, "reproducibility_checksum"))
    for payload, expected in cases:
        assert expected in exp.validate_artifact(payload, REPO)


def test_atomic_writer_and_cli_keep_repository_outputs_untouched(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6780-VALIDATION: publication uses an atomic replacement."""

    artifact = exp.build_artifact(REPO, "20260830", audit_bundle=_audit_bundle(), duration_s=1.0)
    target = tmp_path / "nested" / "artifact.json"
    exp.atomic_write_json(target, artifact)
    assert json.loads(target.read_text(encoding="utf-8")) == artifact
    assert list(target.parent.glob(".*.tmp")) == []

    output = tmp_path / "cli.json"
    assert (
        exp.main(
            [
                "--date",
                "20260830",
                "--root",
                str(REPO),
                "--output",
                str(output),
                "--no-external-audits",
            ]
        )
        == 0
    )
    assert exp.validate_artifact(json.loads(output.read_text(encoding="utf-8")), REPO) == []


def test_wrong_planning_date_is_rejected() -> None:
    """REQ-REPORT-6780: the fixed planning date is part of the evidence contract."""

    with pytest.raises(ValueError, match="planning date"):
        exp.build_artifact(REPO, "20260829", audit_bundle=_audit_bundle(), duration_s=1.0)


def test_low_level_helpers_cover_empty_and_fallback_shapes(tmp_path: Path) -> None:
    """REQ-REPORT-6780: sparse and legacy source shapes stay typed."""

    assert exp.sha256_file(tmp_path / "missing") is None
    assert exp._rate([])["rate"] is None
    assert exp._mean([])["value"] is None
    assert exp._pair_id({"metric": 1}) is None
    assert exp._model_ids({"model_specs": "one/model"}) == ["one/model"]
    assert exp._model_ids({"models_used": ["two/model"]}) == ["two/model"]
    assert exp._declared_class(_record(None, "invalid")) == "disqualified"
    for payload, expected in (
        ({"status": "complete", "flagged_adversarial": True}, "disqualified"),
        ({"status": "complete", "verifier_is_oracle": True}, "circular_positive"),
        ({"honest_verdict": "complete_null"}, "null"),
        ({"status": "complete"}, "positive"),
        ({"status": "unknown"}, "partial"),
    ):
        assert exp._declared_class(_record(payload)) == expected

    failures = exp._gate_failures(
        {
            "gate_check_summary": {
                "checks": [{"check": "x", "expected": True, "observed": False, "passed": False}]
            }
        },
        "exp6768",
    )
    assert failures[0]["check"] == "x"
    assert (
        exp._missing_gate_failures({"task_id": "exp6768", "gated_on": ["not-a-mapping"]}, {}) == []
    )


def test_roadmap_loader_guardrails_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6780-PRECONDITIONS: malformed plans cannot define evidence."""

    original_parse = exp.parse_design_tasks
    (tmp_path / exp.DESIGN_PATH).parent.mkdir(parents=True)
    (tmp_path / exp.DESIGN_PATH).write_text("design", encoding="utf-8")
    roadmap = tmp_path / exp.ACTIVE_ROADMAP_PATH

    roadmap.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping with a task list"):
        exp.load_planned_tasks(tmp_path)

    roadmap.write_text("milestone: wrong\ntasks: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not milestone"):
        exp.load_planned_tasks(tmp_path)

    roadmap.write_text(f"milestone: {exp.MILESTONE}\ntasks: []\n", encoding="utf-8")
    monkeypatch.setattr(exp, "parse_design_tasks", lambda text: [])
    with pytest.raises(ValueError, match="design must contain"):
        exp.load_planned_tasks(tmp_path)

    design = [
        {"task_id": short, "title": short, "deliverable": exp.TASK_PATHS[short]}
        for short in exp.SHORT_TASK_IDS
    ]
    monkeypatch.setattr(exp, "parse_design_tasks", lambda text: design)
    roadmap.write_text(
        yaml_text := json.dumps({"milestone": exp.MILESTONE, "tasks": []}), encoding="utf-8"
    )
    assert yaml_text
    with pytest.raises(ValueError, match="exact V590 task list"):
        exp.load_planned_tasks(tmp_path)

    tasks = [
        {"id": full, "title": short, "deliverable": exp.TASK_PATHS[short]}
        for full, short in zip(exp.EXPECTED_TASK_IDS, exp.SHORT_TASK_IDS, strict=True)
    ]
    tasks[0]["deliverable"] = "results/wrong.json"
    roadmap.write_text(json.dumps({"milestone": exp.MILESTONE, "tasks": tasks}), encoding="utf-8")
    with pytest.raises(ValueError, match="deliverable mismatch"):
        exp.load_planned_tasks(tmp_path)

    with pytest.raises(ValueError, match="deliverable missing"):
        original_parse("**Milestone:** `2026.08.590`\n### Exp 6768: one\n### Exp 6769: two")


def test_invalid_source_row_and_all_branch_classifier_outcomes() -> None:
    """SCENARIO-REPORT-6780-BRANCH-ISOLATION: every closed class has a local path."""

    plan = {
        "order": 1,
        "task_id": "exp6768",
        "manifest_task_id": exp.EXPECTED_TASK_IDS[0],
        "title": "x",
        "branch": "proof",
        "path": exp.TASK_PATHS["exp6768"],
        "gated_on": [],
    }
    invalid_rows = exp.build_experiment_rows([plan], {"exp6768": _record(None, "invalid")})
    assert invalid_rows[0]["honest_verdict"].startswith("complete_disqualified")

    positive_headlines = {
        "proof": {
            "proof_transport_ab_completed": True,
            "paired_exact_valid_effects": {
                "dccd_environment-minus-repaired_direct": {"mean_delta": 0.2}
            },
        },
        "repair": {
            "repair_rows": 2,
            "paired_exact_valid_effect": {"mean_delta": 0.1},
            "paired_harmful_flip_effect": {"pair_count": 2, "mean_delta": 0.0},
            "paired_support_loss_effect": {"pair_count": 2, "mean_delta": 0.0},
        },
        "continuous_memory": {
            "cold_audit_completed": True,
            "required_positive_gates": {"activity": True},
        },
        "arc": {"adoption_positive": True},
    }
    assert (
        exp._branch_class("proof", [{"verdict_class": "positive"}], positive_headlines)
        == "positive"
    )
    negative = copy.deepcopy(positive_headlines)
    negative["repair"]["paired_exact_valid_effect"]["mean_delta"] = 0.0
    assert exp._branch_class("proof", [{"verdict_class": "positive"}], negative) == "null"
    assert (
        exp._branch_class("continuous_memory", [{"verdict_class": "positive"}], positive_headlines)
        == "positive"
    )
    assert (
        exp._branch_class("arc", [{"verdict_class": "positive"}], positive_headlines) == "positive"
    )
    assert (
        exp._branch_class("arc", [{"verdict_class": "disqualified"}], positive_headlines)
        == "disqualified"
    )


def test_branch_rows_count_absolute_audit_paths() -> None:
    """SCENARIO-REPORT-6780-VALIDATION: source findings stay branch-visible."""

    planned, _ = exp.load_planned_tasks(REPO)
    sources = exp.load_source_artifacts(REPO, planned)
    rows = exp.build_experiment_rows(planned, sources)
    branches = exp.build_branch_rows(
        rows,
        exp.recompute_headlines(sources),
        [
            {
                "artifact": str(REPO / exp.TASK_PATHS["exp6768"]),
                "flag_count": 1,
                "flags": [{"severity": "warn"}],
            }
        ],
        [],
    )
    proof = next(row for row in branches if row["branch"] == "proof")
    assert proof["adversarial_finding_count"] == 1


def test_external_audit_bundle_and_skip_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-6780-VALIDATION: each audit runner is invoked and retained."""

    sources = {
        "exp6768": {
            **_record({"rows": []}),
            "path": exp.TASK_PATHS["exp6768"],
        },
        "exp6769": _record(None, "missing"),
    }
    assert exp.run_summaries(REPO, sources, ("exp6769",)) == []
    assert exp.run_row_consistency(REPO, sources, ("exp6769",)) == []
    reports = exp.run_adversarial(REPO, sources, ("exp6768", "exp6769"))
    assert reports[0]["artifact"] == str(REPO / exp.TASK_PATHS["exp6768"])

    monkeypatch.setattr(exp, "run_summaries", lambda root, rows: ["summary"])
    monkeypatch.setattr(exp, "run_adversarial", lambda root, rows: ["adversarial"])
    monkeypatch.setattr(exp, "run_row_consistency", lambda root, rows: ["row"])
    monkeypatch.setattr(exp, "run_recurring_blockers", lambda: {"recurring": []})
    assert exp.collect_audits(REPO, sources) == {
        "summaries": ["summary"],
        "adversarial_findings": ["adversarial"],
        "verdict_row_consistency_findings": ["row"],
        "recurring_blockers": {"recurring": []},
    }

    monkeypatch.setattr(exp.importlib.util, "spec_from_file_location", lambda *args: None)
    with pytest.raises(ImportError, match="cannot load required script"):
        exp._load_script_module(REPO, "missing")


def test_run_command_and_recurrence_ignore_path() -> None:
    """REQ-REPORT-6780: command and retirement helpers keep exact outcomes."""

    code, output = exp._run_command([sys.executable, "-c", "print('receipt')"], REPO)
    assert code == 0
    assert output == "receipt"
    planned = [
        {
            "task_id": "exp6780",
            "branch": "execution_contract",
            "prior_failures": ["bad", {"verdict": "x", "retire_if_same_verdict": False}],
        }
    ]
    rows = [{"task_id": "exp6780", "honest_verdict": exp.HONEST_VERDICT}]
    assert exp.build_prior_verdict_recurrences(planned, rows, exp.HONEST_VERDICT) == []


def test_validator_and_atomic_cleanup_cover_fail_closed_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6780: every top-level validation guard can refuse."""

    artifact = exp.build_artifact(REPO, "20260830", audit_bundle=_audit_bundle(), duration_s=1.0)
    mutations = []
    for field, value, finding in (
        ("milestone", "wrong", "milestone"),
        ("expected_task_ids", [], "expected_task_ids"),
        ("branch_rows", [], "branch_rows"),
        ("verdict_class", "wrong", "closed_verdict_class"),
        ("field_principles", {}, "field_principles"),
        ("verifier_is_oracle", True, "verifier_is_oracle"),
        ("pooled_milestone_success_score", 1.0, "pooled_score"),
    ):
        changed = copy.deepcopy(artifact)
        changed[field] = value
        mutations.append((changed, finding))
    for changed, finding in mutations:
        assert finding in exp.validate_artifact(changed, REPO)

    target = tmp_path / "not-created.json"
    monkeypatch.setattr(exp.os, "replace", lambda source, destination: None)
    exp.atomic_write_json(target, artifact)
    assert not target.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_main_refuses_prewrite_and_cold_reload_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6780-VALIDATION: CLI validation fails before success."""

    monkeypatch.setattr(exp, "build_artifact", lambda *args, **kwargs: {})
    monkeypatch.setattr(exp, "validate_artifact", lambda artifact, root: ["bad"])
    with pytest.raises(ValueError, match="invalid V590 artifact"):
        exp.main(["--root", str(REPO), "--output", str(tmp_path / "one.json")])

    calls = iter(([], ["bad reload"]))
    monkeypatch.setattr(exp, "validate_artifact", lambda artifact, root: next(calls))
    with pytest.raises(ValueError, match="cold reload"):
        exp.main(["--root", str(REPO), "--output", str(tmp_path / "two.json")])


def test_nonterminal_source_stays_partial_in_its_experiment_row() -> None:
    """SCENARIO-REPORT-6780-PRECONDITIONS: an unfinished file stays visible."""

    plan = {
        "order": 1,
        "task_id": "exp6768",
        "manifest_task_id": exp.EXPECTED_TASK_IDS[0],
        "title": "unfinished source",
        "branch": "proof",
        "path": exp.TASK_PATHS["exp6768"],
        "gated_on": [],
    }
    rows = exp.build_experiment_rows([plan], {"exp6768": _record({}, "nonterminal")})
    assert rows[0]["verdict_class"] == "partial"
    assert rows[0]["honest_verdict"].startswith("complete_partial_nonterminal")


def test_collect_self_audits_preserves_hard_and_advisory_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6780-VALIDATION: the capstone audits its own rows."""

    class Verifier:
        @staticmethod
        def verify_artifact(path: Path) -> dict[str, object]:
            return {"artifact": str(path), "flags": []}

    class Lint:
        HARD_CLASSES = ("ALL_ROWS_NULL",)

        @staticmethod
        def check_artifact(path: Path) -> tuple[str, list[str]]:
            assert path == tmp_path / "artifact.json"
            return "findings", ["ALL_ROWS_NULL: empty", "NO_HEADROOM: pinned"]

    modules = iter((Verifier, Lint))
    monkeypatch.setattr(exp, "_load_script_module", lambda root, name: next(modules))
    adversarial, row_report = exp.collect_self_audits(REPO, tmp_path / "artifact.json")
    assert adversarial["flags"] == []
    assert row_report["blocking_count"] == 1
    assert row_report["warning_count"] == 1


def test_main_runs_canonical_self_audits_without_writing_the_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6780-VALIDATION: canonical publication is audited twice."""

    monkeypatch.setattr(exp, "build_artifact", lambda *args, **kwargs: {"stage": "built"})
    monkeypatch.setattr(exp, "validate_artifact", lambda artifact, root: [])
    monkeypatch.setattr(
        exp,
        "reconcile_self_audits",
        lambda artifact, adversarial, rows, root, duration_s: {"stage": "reconciled"},
    )
    stable = ({"flags": []}, {"findings": []})
    monkeypatch.setattr(exp, "collect_self_audits", lambda root, path: stable)

    assert exp.main(["--root", str(tmp_path), "--date", "20260830"]) == 0
    written = json.loads((tmp_path / exp.RESULT_PATH).read_text(encoding="utf-8"))
    assert written == {"stage": "reconciled"}


def test_main_rejects_invalid_reconciliation_before_final_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6780-VALIDATION: post-audit validation fails closed."""

    monkeypatch.setattr(exp, "build_artifact", lambda *args, **kwargs: {})
    validation_results = iter(([], ["invalid after self-audit"]))
    monkeypatch.setattr(exp, "validate_artifact", lambda artifact, root: next(validation_results))
    monkeypatch.setattr(exp, "collect_self_audits", lambda root, path: ({"flags": []}, {}))
    monkeypatch.setattr(
        exp,
        "reconcile_self_audits",
        lambda artifact, adversarial, rows, root, duration_s: artifact,
    )

    with pytest.raises(ValueError, match="invalid V590 artifact after self-audit"):
        exp.main(["--root", str(tmp_path), "--date", "20260830"])


@pytest.mark.parametrize(
    ("reports", "message"),
    [
        (
            (({"flags": []}, {"findings": []}), ({"flags": ["changed"]}, {"findings": []})),
            "adversarial self-audit changed",
        ),
        (
            (({"flags": []}, {"findings": []}), ({"flags": []}, {"findings": ["changed"]})),
            "row self-audit changed",
        ),
    ],
)
def test_main_rejects_an_unstable_final_self_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reports: tuple[tuple[dict[str, object], dict[str, object]], ...],
    message: str,
) -> None:
    """SCENARIO-REPORT-6780-VALIDATION: a changing final audit fails closed."""

    monkeypatch.setattr(exp, "build_artifact", lambda *args, **kwargs: {})
    monkeypatch.setattr(exp, "validate_artifact", lambda artifact, root: [])
    monkeypatch.setattr(
        exp,
        "reconcile_self_audits",
        lambda artifact, adversarial, rows, root, duration_s: artifact,
    )
    audit_sequence = iter(reports)
    monkeypatch.setattr(exp, "collect_self_audits", lambda root, path: next(audit_sequence))

    with pytest.raises(ValueError, match=message):
        exp.main(["--root", str(tmp_path), "--date", "20260830"])
