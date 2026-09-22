"""Tests for REQ-ARC-WMTE-7512 and its opportunity-audit scenarios."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time

import pytest

from carnot import experiment_7512_v657_arc_opportunity as audit


ROOT = Path(__file__).resolve().parents[2]


def _fixture() -> dict[str, object]:
    return audit.build_artifact_for_test(ROOT)


def test_preconditions_authenticate_both_panels_and_registry() -> None:
    """REQ-ARC-WMTE-7512 and SCENARIO-ARC-WMTE-7512-PANEL-AUTHENTICATION."""

    checks, sources = audit.collect_preconditions(ROOT)

    assert checks
    assert all(row["passed"] for row in checks)
    assert sources[audit.PANEL_A_PATH.as_posix()]["sha256"].startswith("sha256:")
    assert sources[audit.PANEL_B_PATH.as_posix()]["sha256"].startswith("sha256:")
    assert sources[audit.REGISTRY_PATH.as_posix()]["sha256"].startswith("sha256:")


def test_raw_reduction_reconciles_support_tokens_and_censoring() -> None:
    """REQ-ARC-WMTE-7512 and SCENARIO-ARC-WMTE-7512-RAW-REDUCTION."""

    reduced = audit.reduce_upstreams(ROOT)

    assert len(reduced["rows"]) == 36
    assert len(reduced["per_game_results"]) == 12
    assert reduced["sample_size_budget"]["completed_independent_units"] == 36
    assert reduced["sample_size_budget"]["independent_game_clusters"] == 12
    assert reduced["token_counts"] == {
        "input_tokens": 516_690,
        "output_tokens": 18_432,
        "generated_tokens": 18_432,
        "completed_requests": 72,
    }
    assert all(row["progress_censored"] for row in reduced["rows"])
    assert all(row["censored_cost_ns"] == row["observed_episode_ns"] for row in reduced["rows"])
    assert reduced["pooling"]["source_compatible"] is True
    assert reduced["pooling"]["episode_support_passed"] is True
    assert reduced["pooling"]["game_support_passed"] is True


def test_missing_stage_timers_stay_unknown_and_suppress_cost_claim() -> None:
    """REQ-ARC-WMTE-7512 and SCENARIO-ARC-WMTE-7512-POOLING-AND-DENOMINATOR."""

    reduced = audit.reduce_upstreams(ROOT)
    coverage = reduced["cost_denominator_coverage"]

    assert coverage["complete_mutually_exclusive_denominator"] is False
    assert coverage["stages"]["request"]["status"] == "measured"
    assert coverage["stages"]["model_load"]["status"] == "measured_outside_episode"
    for name in ("native_forward", "sampling", "parsing", "planner", "environment", "update"):
        assert coverage["stages"][name]["duration_ns"] is None
        assert coverage["stages"][name]["status"] == "unknown"
    assert reduced["cost_share"] is None
    assert reduced["amdahl_upper_bound"] is None


def test_supervisor_evaluations_separate_unknown_eligibility_and_shadow_firing() -> None:
    """REQ-ARC-WMTE-7512 and SCENARIO-ARC-WMTE-7512-SUPERVISOR-CHOICES."""

    reduced = audit.reduce_upstreams(ROOT)
    supervisor = reduced["supervisor_disposition"]

    assert len(reduced["supervisor_opportunity_rows"]) == 6_480
    assert supervisor["gate_evaluation_count"] == 6_480
    assert supervisor["unknown_selected_eligibility_count"] == 6_480
    assert supervisor["explicit_true_selected_count"] == 0
    assert supervisor["recorded_firing_count"] == 36
    assert supervisor["applied_redirection_count"] == 0
    assert supervisor["eligible_firing_opportunity_count"] == 0
    assert supervisor["intervention_ledger"] == []
    assert supervisor["efficacy_estimate"] is None
    assert supervisor["efficacy_tuning_disposition"] == "retired_until_live_reachable_choices"
    fired = [row for row in reduced["supervisor_opportunity_rows"] if row["fired"]]
    assert len(fired) == 36
    assert all(row["selected_arm"] == "force_exploration_diversity" for row in fired)
    assert all(row["applied_redirection"] is False for row in fired)


def test_game_is_the_transfer_unit_and_zero_progress_cannot_generalize() -> None:
    """REQ-ARC-WMTE-7512 and SCENARIO-ARC-WMTE-7512-NO-TRANSFER."""

    reduced = audit.reduce_upstreams(ROOT)

    assert all(row["episode_count"] == 3 for row in reduced["per_game_results"])
    assert all(row["progressed_episode_count"] == 0 for row in reduced["per_game_results"])
    assert all(row["recorded_firing_count"] == 3 for row in reduced["per_game_results"])
    assert reduced["generalization_assessment"] == {
        "transfer_unit": "game",
        "unique_game_count": 12,
        "progressed_game_count": 0,
        "broad_generalization_supported": False,
        "reason": "all_zero_progress_panel_cannot_establish_transfer",
    }


def test_identity_mutation_forces_separate_descriptive_panels() -> None:
    """REQ-ARC-WMTE-7512 and SCENARIO-ARC-WMTE-7512-PANEL-AUTHENTICATION."""

    identities = audit.panel_identity_rows(ROOT)
    changed = deepcopy(identities)
    changed["B"]["quantization"] = "different"
    result = audit.evaluate_pooling([], changed, panel_states={"A": "qualified", "B": "qualified"})

    assert result["source_compatible"] is False
    assert "quantization" in result["mismatched_identities"]
    assert result["stratification"] == "separate_descriptive_panels"


def test_terminal_fixture_keeps_completion_separate_from_cost_readiness() -> None:
    """REQ-ARC-WMTE-7512 and SCENARIO-ARC-WMTE-7512-TERMINAL."""

    artifact = _fixture()

    assert artifact["status"] == "complete_null_zero_eligible_supervisor_opportunity"
    assert artifact["verdict_class"] == "null"
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert not any(artifact["invocation_counts"].values())
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["arc_opportunity_audit_complete_score"] == 1
    assert artifact["arc_cost_claim_ready_score"] == 0
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["new_solve_attempted"] is False
    assert artifact["new_level_credit"] == 0
    assert artifact["generalization_assessment"]["broad_generalization_supported"] is False
    assert audit.validate_artifact(artifact, ROOT, require_terminal=True) == []


def test_artifact_reader_rejects_current_calls_and_reduction_mutation() -> None:
    """REQ-ARC-WMTE-7512 and SCENARIO-ARC-WMTE-7512-TERMINAL."""

    artifact = _fixture()
    current = deepcopy(artifact)
    current["invocation_counts"]["generation_calls_attempted"] = 1
    changed = deepcopy(artifact)
    changed["rows"][0]["input_tokens"] += 1

    assert "current_invocation_counts_nonzero" in audit.validate_artifact(
        current, ROOT, require_terminal=True
    )
    assert "independent_reduction_mismatch" in audit.validate_artifact(
        changed, ROOT, require_terminal=True
    )


def test_blocked_classification_names_external_failure() -> None:
    """REQ-ARC-WMTE-7512 and SCENARIO-ARC-WMTE-7512-PANEL-AUTHENTICATION."""

    verdict, honest, complete, ready = audit.classify_terminal(
        external_available=False,
        evidence_valid=False,
        validation_passed=False,
        denominator_ready=False,
        opportunity_present=False,
    )

    assert (verdict, honest, complete, ready) == (
        "blocked",
        "complete_blocked_external_prerequisite_absent",
        0,
        0,
    )


def test_validation_plan_is_exact_and_reporting_only(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7512 and SCENARIO-ARC-WMTE-7512-TERMINAL."""

    plan = audit.build_validation_plan(ROOT, tmp_path)

    assert audit.validate_validation_plan(ROOT, plan) == []
    assert [row.name for row in plan] == list(audit.CURRENT_VALIDATION_NAMES)
    assert not any(row.name.startswith("e2e_") for row in plan)
    assert all(
        argument.rstrip("/") not in {"tests", "tests/python"}
        for row in plan
        for argument in row.argv
    )
    report = next(row for row in plan if row.name == "changed_module_coverage_report")
    assert "--fail-under=100" in report.argv


def test_validation_plan_and_terminal_command_mutations_fail_closed(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7512 and SCENARIO-ARC-WMTE-7512-TERMINAL."""

    plan = audit.build_validation_plan(ROOT, tmp_path)
    broad = replace(plan[1], argv=(*plan[1].argv, "tests/python"))
    errors = audit.validate_validation_plan(ROOT, [*plan, plan[0], broad])
    assert any(error.startswith("command_count:") for error in errors)
    assert any(error.startswith("broad_test_target:") for error in errors)

    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}\n", encoding="utf-8")
    terminal = audit.terminal_command_specs(ROOT, candidate)
    assert [row.name for row in terminal] == list(audit.TERMINAL_RECEIPT_NAMES)
    assert all(str(candidate) in row.argv for row in terminal)
    assert "--strict" in terminal[-1].argv


def test_replay_cli_reduces_exact_candidate(tmp_path: Path, capsys: object) -> None:
    """REQ-ARC-WMTE-7512 and SCENARIO-ARC-WMTE-7512-TERMINAL."""

    artifact = _fixture()
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")

    assert audit.main(["--replay", str(candidate), "--reduce-only"]) == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert json.loads(captured.out)["matches_declared"] is True


def test_small_defensive_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7512 and SCENARIO-ARC-WMTE-7512-RAW-REDUCTION."""

    assert audit.utc_now().endswith("Z")
    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert audit.load_object(missing) == {}
    assert audit.load_object(malformed) == {}
    malformed.write_text("[]", encoding="utf-8")
    assert audit.load_object(malformed) == {}
    with pytest.raises(ValueError, match="unsupported_gate_operator"):
        audit.gate(
            "bad",
            "validity",
            1,
            1,
            upstream="fixture",
            field="value",
            principle="Closed operators prevent ambiguous gates.",
            op="!=",
        )

    mismatched = [
        {
            "event": "stage_start",
            "run_id": "a",
            "process_id": 1,
            "clock_identity": "clock",
            "interval_start_monotonic_ns": 1,
        },
        {
            "event": "stage_end",
            "run_id": "b",
            "process_id": 1,
            "clock_identity": "clock",
            "interval_end_monotonic_ns": 2,
        },
    ]
    assert audit._selection_interval(mismatched) == (None, None)
    no_selection = [
        {
            "episode_id": "episode",
            "seam": "supervisor_arm_selection",
            "decision_id": "decision",
            "event": "stage_start",
        }
    ]
    assert (
        audit.reduce_supervisor_evaluations(
            "A", "game", 1, "episode", no_selection, progressed=False
        )
        == []
    )
    assert audit._stage_value({}, "absent") is None


def test_panel_reducer_disqualifies_unqualified_and_missing_shards(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7512 and SCENARIO-ARC-WMTE-7512-PANEL-AUTHENTICATION."""

    blocked = audit._reduce_panel(
        tmp_path,
        panel="A",
        producer={},
        episodes_path=Path("episodes.json"),
        qualified=False,
    )
    assert blocked["state"] == "blocked_unqualified"
    assert blocked["errors"] == ["upstream_qualification_missing"]

    (tmp_path / "episodes.json").write_text('{"rows": []}\n', encoding="utf-8")
    invalid = audit._reduce_panel(
        tmp_path,
        panel="A",
        producer={"seam_event_shards": [None]},
        episodes_path=Path("episodes.json"),
        qualified=True,
    )
    assert invalid["state"] == "disqualified_invalid"
    assert "interval_shard_authentication_failed" in invalid["errors"]
    assert "frozen_episode_schedule_mismatch" in invalid["errors"]


def test_classification_and_blocked_artifact_are_terminal() -> None:
    """REQ-ARC-WMTE-7512 and SCENARIO-ARC-WMTE-7512-PANEL-AUTHENTICATION."""

    assert audit.classify_terminal(
        external_available=True,
        evidence_valid=False,
        validation_passed=True,
        denominator_ready=True,
        opportunity_present=True,
    )[:2] == ("disqualified", "complete_disqualified_arc_opportunity_evidence")
    assert audit.classify_terminal(
        external_available=True,
        evidence_valid=True,
        validation_passed=True,
        denominator_ready=True,
        opportunity_present=True,
    ) == (
        "null",
        "complete_null_observational_opportunity_no_causal_efficacy",
        1,
        1,
    )
    failed = audit.gate(
        "source_bytes:missing",
        "validity",
        True,
        False,
        upstream="missing",
        field="readable_nonempty_bytes",
        principle="Missing external bytes must remain blocked.",
    )
    blocked = audit.build_blocked_artifact(
        [failed],
        {"missing": {"exists": False}},
        started_at_utc="2026-09-22T00:00:00Z",
        duration_s=0.1,
    )
    assert blocked["status"] == "complete_blocked_external_prerequisite_absent"
    assert blocked["gate_check_summary"]["first_failed_check"] == "source_bytes:missing"
    assert blocked["reproducibility_checksum"] == audit.artifact_checksum(blocked)
    span = audit._phase("test", time.monotonic(), time.monotonic(), 1)
    assert span["phase"] == "test"
    assert span["completed_units"] == 1


def test_artifact_defensive_failure_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-7512 and SCENARIO-ARC-WMTE-7512-TERMINAL."""

    artifact = _fixture()
    monkeypatch.setattr(
        audit,
        "independent_reduce",
        lambda _artifact, _root: {"matches_declared": True},
    )
    mutations = {
        "identity_mismatch:schema": ("schema", "wrong"),
        "new_solve_claimed": ("new_solve_attempted", True),
        "invalid_verdict_class": ("verdict_class", "unknown"),
        "honest_verdict_not_terminal": ("honest_verdict", "retry"),
        "cost_ready_without_denominator": ("arc_cost_claim_ready_score", 1),
        "numeric_cost_claim_without_denominator": ("cost_share", 0.1),
    }
    for expected, (field, value) in mutations.items():
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected in audit.validate_artifact(changed, ROOT, require_terminal=True)

    no_principle = deepcopy(artifact)
    no_principle["field_principles"]["schema"] = ""
    assert "field_principle_missing" in audit.validate_artifact(
        no_principle, ROOT, require_terminal=True
    )
    no_gate_principle = deepcopy(artifact)
    no_gate_principle["acceptance_gate_results"][0]["principle"] = ""
    assert "gate_principle_missing" in audit.validate_artifact(
        no_gate_principle, ROOT, require_terminal=True
    )
    scoped = deepcopy(artifact)
    scoped["validation_receipts"][0]["passed"] = False
    assert "current_scoped_validation_missing_or_failed" in audit.validate_artifact(
        scoped, ROOT, require_terminal=True
    )
    terminal = deepcopy(artifact)
    terminal["validation_receipts"][-1]["passed"] = False
    assert "terminal_validation_missing_or_failed" in audit.validate_artifact(
        terminal, ROOT, require_terminal=True
    )

    original_dumps = audit.json.dumps
    monkeypatch.setattr(audit.json, "dumps", lambda *_args, **_kwargs: "x" * (21 * 1024 * 1024))
    assert "artifact_exceeds_20_mib" in audit.validate_artifact(
        artifact, ROOT, require_terminal=True
    )
    monkeypatch.setattr(audit.json, "dumps", original_dumps)


def test_runtime_e2e_command_is_rejected(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7512 and SCENARIO-ARC-WMTE-7512-TERMINAL."""

    plan = audit.build_validation_plan(ROOT, tmp_path)
    fake = replace(plan[0], name="e2e_fake")
    errors = audit.validate_validation_plan(ROOT, [*plan, fake])
    assert "runtime_e2e_forbidden:e2e_fake" in errors
