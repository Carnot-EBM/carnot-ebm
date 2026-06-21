"""Tests for Exp 4528 .417 B-track infra carryforward.

Spec refs: REQ-ARC-FCP-4528, SCENARIO-ARC-FCP-4528.
"""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
MODULE_PATH = REPO / "python" / "carnot" / "experiment_4528_infra_carryforward.py"

_spec = importlib.util.spec_from_file_location("experiment_4528_infra_carryforward", MODULE_PATH)
exp4528 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(exp4528)


def _preconditions(ok: bool = True) -> dict[str, object]:
    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "gate_help_precondition": True,
        "upstream_artifact_present": True,
        "fixture_test_file_present": True,
        "spec_has_req_4528": True,
        "ok": ok,
    }


def _fixture_test_source() -> str:
    return "\n".join(
        [
            "def test_a1_frame_change_prune_fails_lost_core_m0r0(): pass",
            "def test_a2_imitation_prior_fails_core_traded_for_fringe(): pass",
            "def test_positive_core_faster_passes_improved(): pass",
            "def test_neutral_core_same_passes_non_inferior(): pass",
            "def test_bonus_solve_reported_but_core_required(): pass",
        ]
    )


def _upstream(*, stable_budget: int | None = None, tests_passed: bool = True) -> dict[str, object]:
    rows = [
        {
            "budget": 8000,
            "comparison_budget": 12000,
            "solved_games": ["lp85", "m0r0", "sp80", "vc33"],
            "comparison_solved_games": ["lp85", "ls20", "m0r0", "sp80", "su15", "vc33"],
            "stable_vs_1_5x": False,
        },
        {
            "budget": 12000,
            "comparison_budget": 18000,
            "solved_games": ["lp85", "ls20", "m0r0", "sp80", "su15", "vc33"],
            "comparison_solved_games": ["cd82", "lp85", "ls20", "m0r0", "sp80", "su15", "vc33"],
            "stable_vs_1_5x": False,
        },
    ]
    selected_default = 8000
    if stable_budget is not None:
        rows.append(
            {
                "budget": stable_budget,
                "comparison_budget": int(stable_budget * 1.5),
                "solved_games": ["lp85", "m0r0", "sp80", "vc33"],
                "comparison_solved_games": ["lp85", "m0r0", "sp80", "vc33"],
                "stable_vs_1_5x": True,
            }
        )
        selected_default = stable_budget
    return {
        "experiment": "experiment_4518_metric_harness_canonical",
        "honest_verdict": "shipped: metric_harness_canonical_ci_guarded",
        "canonical_game_set": ["lp85", "m0r0", "sp80", "vc33", "cd82", "ft09", "su15", "ls20"],
        "canonical_default_budget": selected_default,
        "canonical_baseline": {
            "action_metric_field": "actions",
            "core_games": ["lp85", "m0r0", "sp80", "vc33"],
            "median_actions_on_core": 7760.0,
            "guard": {"ok": True, "errors": []},
        },
        "positive_control_passed": True,
        "tests_added_pass": {
            "command": ".venv/bin/pytest tests/python/test_arc_submission_gate_verdict.py "
            "tests/python/test_experiment_4518_metric_harness_canonical.py -q --no-cov",
            "passed": tests_passed,
            "stdout_tail": "15 passed",
        },
        "headroom_budget_measurement": {
            "measured": True,
            "selected_default_budget": selected_default,
            "rows": rows,
            "measurements_by_budget": {
                "8000": {"solved_games": ["lp85", "m0r0", "sp80", "vc33"]},
                "12000": {"solved_games": ["lp85", "ls20", "m0r0", "sp80", "su15", "vc33"]},
            },
        },
    }


def test_req_arc_fcp_4528_spec_declares_carryforward_contract() -> None:
    """REQ-ARC-FCP-4528: OpenSpec anchors the carryforward artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-4528" in spec
    assert "SCENARIO-ARC-FCP-4528" in spec
    assert exp4528.RESULT_RELATIVE_PATH in spec
    assert exp4528.UPSTREAM_RELATIVE_PATH in spec
    for field, principle in exp4528.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_fcp_4528_records_audit_without_blind_budget_raise() -> None:
    """SCENARIO-ARC-FCP-4528: no stable headroom plateau is recorded, not raised."""

    artifact = exp4528.build_artifact(
        preconditions_checked=_preconditions(),
        upstream_artifact=_upstream(),
        fixture_test_source=_fixture_test_source(),
        tests_added_pass={"command": "fixture", "passed": True},
        duration_s=0.25,
    )

    assert exp4528.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == "complete: infra_audit_already_done"
    assert artifact["inference_substrate"] == exp4528.INFERENCE_SUBSTRATE
    assert artifact["b_track_status"]["upstream_417_b2_landed"]["canonical_gate"]["core_containment_gate_canonical"]
    assert artifact["b_track_status"]["upstream_417_b2_landed"]["fixture_ci_guard"]["ci_guarded"]
    assert artifact["b_track_status"]["upstream_417_b2_landed"]["fixture_ci_guard"]["fixture_count"] == 5
    headroom = artifact["b_track_status"]["upstream_417_b2_landed"]["headroom_budget"]
    assert headroom["measured"] is True
    assert headroom["b_star_measured"] is False
    assert headroom["selected_b_star"] is None
    assert artifact["b_track_status"]["this_task_completed"]["no_blind_budget_raise"] is True
    assert artifact["cited_upstream_artifacts"][0]["path"] == exp4528.UPSTREAM_RELATIVE_PATH


def test_req_arc_fcp_4528_selects_first_stable_b_star_when_present() -> None:
    """REQ-ARC-FCP-4528: B* is the first stable budget in the upstream table."""

    status = exp4528.audit_upstream(
        _upstream(stable_budget=16000),
        fixture_test_source=_fixture_test_source(),
    )

    headroom = status["upstream_417_b2_landed"]["headroom_budget"]
    assert headroom["b_star_measured"] is True
    assert headroom["selected_b_star"] == 16000
    assert headroom["upstream_default_budget"] == 16000
    assert status["this_task_completed"]["audit_only"] is True
    assert status["gaps"] == []

    fallback_default = exp4528.audit_upstream(
        {
            **_upstream(),
            "canonical_default_budget": None,
            "headroom_budget_measurement": {
                "measured": True,
                "selected_default_budget": "12000",
                "rows": "not-a-list",
            },
        },
        fixture_test_source=_fixture_test_source(),
    )
    assert fallback_default["upstream_417_b2_landed"]["headroom_budget"]["candidate_rows"] == []
    assert fallback_default["upstream_417_b2_landed"]["headroom_budget"]["upstream_default_budget"] == 12000

    hard_default = exp4528.audit_upstream(
        {
            **_upstream(),
            "canonical_default_budget": None,
            "headroom_budget_measurement": {"measured": True, "selected_default_budget": None, "rows": []},
        },
        fixture_test_source=_fixture_test_source(),
    )
    assert hard_default["upstream_417_b2_landed"]["headroom_budget"]["upstream_default_budget"] == 8000
    assert exp4528._int_or_none(True) is None
    assert exp4528._int_or_none("bad") is None


def test_scenario_arc_fcp_4528_run_writes_json_from_upstream_artifact(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-4528: run() writes the stable carryforward JSON."""

    upstream_path = tmp_path / exp4528.UPSTREAM_RELATIVE_PATH
    upstream_path.parent.mkdir(parents=True)
    upstream_path.write_text(json.dumps(_upstream()), encoding="utf-8")

    artifact = exp4528.run(
        root=tmp_path,
        write=True,
        preconditions_checked=_preconditions(),
        fixture_test_source=_fixture_test_source(),
        tests_added_pass={"command": "fixture", "passed": True},
        now=lambda: 1.0,
    )

    out = tmp_path / exp4528.RESULT_RELATIVE_PATH
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8")) == artifact
    assert artifact["b_track_status"]["this_task_completed"]["completed_missing_piece"] == "none"

    fixture_path = tmp_path / exp4528.FIXTURE_TEST_RELATIVE_PATH
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text(_fixture_test_source(), encoding="utf-8")
    assert exp4528.read_fixture_test_source(tmp_path) == _fixture_test_source()

    bad_upstream = _upstream()
    bad_upstream["canonical_default_budget"] = 12000
    upstream_path.write_text(json.dumps(bad_upstream), encoding="utf-8")
    try:
        exp4528.run(
            root=tmp_path,
            write=False,
            preconditions_checked=_preconditions(),
            fixture_test_source=_fixture_test_source(),
            tests_added_pass={"command": "fixture", "passed": True},
            now=lambda: 1.0,
        )
    except ValueError as exc:
        assert "no_blind_budget_raise" in str(exc)
    else:  # pragma: no cover - assertion guard.
        raise AssertionError("expected no_blind_budget_raise schema failure")


def test_req_arc_fcp_4528_honest_verdict_and_schema_guardrails() -> None:
    """REQ-ARC-FCP-4528: partial evidence produces terminal-prefixed audit verdicts."""

    status = exp4528.audit_upstream(_upstream(), fixture_test_source=_fixture_test_source())
    tests = {"passed": True}

    assert exp4528._honest_verdict(preconditions_checked={"ok": False}, b_track_status=status, tests_added_pass=tests) == (
        "blocked_infra_carryforward_preconditions"
    )

    missing_gate = exp4528.audit_upstream(
        {**_upstream(), "canonical_baseline": {"guard": {"ok": False}}},
        fixture_test_source=_fixture_test_source(),
    )
    assert exp4528._honest_verdict(
        preconditions_checked=_preconditions(),
        b_track_status=missing_gate,
        tests_added_pass=tests,
    ) == "complete: infra_carryforward_missing_canonical_gate_evidence"

    missing_fixtures = exp4528.audit_upstream(
        _upstream(tests_passed=False),
        fixture_test_source="def test_only_one_fixture(): pass",
    )
    assert exp4528._honest_verdict(
        preconditions_checked=_preconditions(),
        b_track_status=missing_fixtures,
        tests_added_pass=tests,
    ) == "complete: infra_carryforward_missing_fixture_ci_guard"

    no_headroom = exp4528.audit_upstream(
        {**_upstream(), "headroom_budget_measurement": {"measured": False, "rows": []}},
        fixture_test_source=_fixture_test_source(),
    )
    assert exp4528._honest_verdict(
        preconditions_checked=_preconditions(),
        b_track_status=no_headroom,
        tests_added_pass=tests,
    ) == "complete: infra_carryforward_missing_headroom_measurement"

    assert exp4528._honest_verdict(
        preconditions_checked=_preconditions(),
        b_track_status=status,
        tests_added_pass={"passed": False},
    ) == "complete: infra_carryforward_tests_not_green"

    valid = exp4528.build_artifact(
        preconditions_checked=_preconditions(),
        upstream_artifact=_upstream(),
        fixture_test_source=_fixture_test_source(),
        tests_added_pass={"command": "fixture", "passed": True},
        duration_s=0.25,
    )
    missing = dict(valid)
    missing.pop("schema")
    assert "missing required field schema" in exp4528.artifact_schema_errors(missing)

    malformed = {
        **valid,
        "honest_verdict": "working",
        "inference_substrate": "live",
        "field_principles": {},
        "preconditions_checked": [],
        "b_track_status": [],
        "cited_upstream_artifacts": {},
        "tests_added_pass": [],
        "reproducibility_checksum": 123,
        "leaderboard_submission": True,
        "result_path": "wrong.json",
    }
    malformed_errors = exp4528.artifact_schema_errors(malformed)
    assert "honest_verdict must start with a terminal prefix" in malformed_errors
    assert "inference_substrate must match the required substrate" in malformed_errors
    assert "field_principles must match required principles" in malformed_errors
    assert "preconditions_checked must be a mapping" in malformed_errors
    assert "b_track_status must be a mapping" in malformed_errors
    assert "cited_upstream_artifacts must be a list" in malformed_errors
    assert "tests_added_pass must be a mapping" in malformed_errors
    assert "reproducibility_checksum must be sha256-prefixed" in malformed_errors
    assert "leaderboard_submission must be false" in malformed_errors
    assert "result_path must point to the 4528 artifact" in malformed_errors

    blind_raise = {
        **valid,
        "b_track_status": {
            **valid["b_track_status"],
            "this_task_completed": {
                **valid["b_track_status"]["this_task_completed"],
                "no_blind_budget_raise": False,
            },
        },
    }
    blind_raise["reproducibility_checksum"] = exp4528.payload_checksum(blind_raise)
    assert "no_blind_budget_raise must be true for terminal audits" in exp4528.artifact_schema_errors(
        blind_raise
    )

    bad = {**valid, "reproducibility_checksum": "sha256:bad"}
    assert "reproducibility_checksum must match payload" in exp4528.artifact_schema_errors(bad)
