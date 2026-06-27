"""Tests for Exp 4865 hostile A1 induce-plan fork-probe audit.

Spec refs: REQ-ARC-WMTE-4865,
SCENARIO-ARC-WMTE-4865-A1-FORK-AUDIT,
SCENARIO-ARC-WMTE-4865-NON-TEST-CLASSIFICATION.
"""

from __future__ import annotations

import json
import ast
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4865_fork_probe_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _row(
    game: str, *, accuracy: float, bucket: str, migrated: bool | None = None
) -> dict[str, Any]:
    return {
        "game": game,
        "engine_heldout_accuracy": accuracy,
        "planned_bucket": bucket,
        "migrated": bucket == "COVERED" if migrated is None else migrated,
        "winning_prefix_len": 2,
        "planned_pool_size": 1 if bucket != "NEVER_ENUMERATED" else 0,
        "heldout_transition_count": 5,
        "live_path_methods_called": [
            "E3AgentPolicy._induce_and_plan",
            "arc_executable_world_model.load_engine",
            "arc_executable_world_model.plan_in_model",
        ],
    }


def _control(bucket: str = "COVERED", accuracy: float = 0.9) -> dict[str, Any]:
    return _row("tu93", accuracy=accuracy, bucket=bucket)


def _good_a1_artifact() -> dict[str, Any]:
    rows = {
        "cd82": _row("cd82", accuracy=0.8, bucket="COVERED"),
        "cn04": _row("cn04", accuracy=0.7, bucket="NEVER_ENUMERATED"),
        "ls20": _row("ls20", accuracy=0.9, bucket="ENUMERATED_BUT_LOST"),
    }
    return {
        "experiment_id": 4861,
        "honest_verdict": "complete_generation_wall_guidance_wall_high_accuracy_migration",
        "fork_verdict": "GUIDANCE_WALL",
        "per_game_fork": rows,
        "coverage_migration_count": 1,
        "median_engine_heldout_accuracy": 0.8,
        "positive_control_game": "tu93",
        "positive_control_migrated": True,
        "positive_control_fork": _control(),
        "planner_blind_to_banked_answer": True,
        "n_games_measured": 3,
        "live_path_reachable": True,
        "solve_provenance": "development_proxy",
        "inference_substrate": "live_llm_inference",
    }


def _good_source() -> str:
    return """
def measure_game_with_live_induce_plan(game, winning_prefix, proposer):
    from carnot.agentic import arc_executable_world_model as e3
    cold = _collect_cold_policy_transitions(game=game, proposer=proposer)
    policy = cold["policy"]
    policy._induce_and_plan()
    engine, is_done = e3.load_engine(game)
    plan = e3.plan_in_model(engine, is_done, policy.root_grid)
    reached = _execute_plan_reaches_l1(game, plan)
    return classify_planned_pool(
        game,
        winning_prefix,
        plan,
        planner_reached_l1_win=reached,
    )
"""


def _clean_auxiliary_results() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        {"returncode": 0, "stdout": "LIVE re-check: clean", "stderr": ""},
        {"loaded": True, "flag_count": 0, "flags": []},
        {"passed": True, "returncode": 0, "stdout_tail": "OK", "stderr_tail": ""},
    )


def test_req_arc_wmte_4865_spec_declares_fork_audit_contract() -> None:
    """REQ-ARC-WMTE-4865: OpenSpec anchors fields, scenarios, and result path."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4865",
        "SCENARIO-ARC-WMTE-4865-A1-FORK-AUDIT",
        "SCENARIO-ARC-WMTE-4865-NON-TEST-CLASSIFICATION",
        mod.RESULT_RELATIVE_PATH,
        mod.AUDIT_REPORT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4865_good_a1_is_genuinely_diagnostic() -> None:
    """SCENARIO-ARC-WMTE-4865-A1-FORK-AUDIT: all four hostile gates pass."""

    summary, adversarial, lint = _clean_auxiliary_results()
    audit = mod.audit_a1_artifact(
        _good_a1_artifact(),
        source_text=_good_source(),
        summarizer_result=summary,
        adversarial_result=adversarial,
        live_lint_result=lint,
    )

    assert audit["honest_verdict"] == "complete_a1_fork_probe_audited"
    assert audit["a1_genuinely_diagnostic"] is True
    assert audit["non_diagnostic_reasons"] == []
    assert audit["planner_blind_confirmed"] is True
    assert audit["positive_control_confirmed"] is True
    assert audit["numbers_match_fork"] is True
    assert audit["live_path_reachable_confirmed"] is True
    assert audit["solve_provenance_confirmed"] is True
    assert audit["checks"]["numbers_match_fork"]["computed_fork_verdict"] == "GUIDANCE_WALL"
    assert audit["checks"]["summarizer_and_adversarial_verify"]["passed"] is True


def test_scenario_arc_wmte_4865_non_test_variants_fail_closed() -> None:
    """SCENARIO-ARC-WMTE-4865-NON-TEST-CLASSIFICATION: every hostile gate can fail."""

    summary, adversarial, lint = _clean_auxiliary_results()
    injected_source = """
def measure_game_with_live_induce_plan(game, winning_prefix, proposer):
    from carnot.agentic import arc_executable_world_model as e3
    policy = seeded_frontier(winning_prefix)
    policy._induce_and_plan()
    engine, is_done = e3.load_engine(game)
    plan = e3.plan_in_model(engine, is_done, policy.root_grid, hint=winning_prefix)
    return classify_planned_pool(game, winning_prefix, plan, planner_reached_l1_win=False)
"""

    flagged_blind = _good_a1_artifact()
    flagged_blind["planner_blind_to_banked_answer"] = False

    control_failed = _good_a1_artifact()
    control_failed["positive_control_migrated"] = False
    control_failed["positive_control_fork"] = _control(bucket="NEVER_ENUMERATED", accuracy=0.95)

    wrong_fork = _good_a1_artifact()
    wrong_fork["fork_verdict"] = "PLANNER_GAP"

    too_few_games = _good_a1_artifact()
    too_few_games["per_game_fork"] = {"cd82": _row("cd82", accuracy=0.8, bucket="COVERED")}
    too_few_games["n_games_measured"] = 1

    dishonest = _good_a1_artifact()
    dishonest["live_path_reachable"] = False
    dishonest["solve_provenance"] = "live_agent_self_discovery"
    bad_lint = {"passed": False, "returncode": 1, "stdout_tail": "", "stderr_tail": "boom"}

    cases = [
        (_good_a1_artifact(), injected_source, lint, "banked_answer_used_before_classification"),
        (flagged_blind, _good_source(), lint, "artifact_planner_blind_flag_false"),
        (control_failed, _good_source(), lint, "positive_control_not_migrated"),
        (wrong_fork, _good_source(), lint, "fork_verdict_mismatch"),
        (too_few_games, _good_source(), lint, "n_games_measured_below_3"),
        (dishonest, _good_source(), bad_lint, "live_path_unreachable"),
        (dishonest, _good_source(), lint, "solve_provenance_not_development_proxy"),
    ]

    for artifact, source_text, lint_result, reason in cases:
        audit = mod.audit_a1_artifact(
            artifact,
            source_text=source_text,
            summarizer_result=summary,
            adversarial_result=adversarial,
            live_lint_result=lint_result,
        )
        assert audit["a1_genuinely_diagnostic"] is False
        assert reason in audit["non_diagnostic_reasons"]
        assert audit["honest_verdict"].startswith("complete_a1_fork_probe_non_test_")


def test_req_arc_wmte_4865_build_schema_and_report_write(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4865: artifact and markdown writes are checksum-stable."""

    source = tmp_path / mod.SOURCE_ARTIFACT_RELATIVE_PATH
    script = tmp_path / mod.SOURCE_SCRIPT_RELATIVE_PATH
    source.parent.mkdir(parents=True)
    script.parent.mkdir(parents=True)
    source.write_text(json.dumps(_good_a1_artifact()), encoding="utf-8")
    script.write_text(_good_source(), encoding="utf-8")
    summary, adversarial, lint = _clean_auxiliary_results()
    audit = mod.audit_a1_artifact(
        _good_a1_artifact(),
        source_text=_good_source(),
        summarizer_result=summary,
        adversarial_result=adversarial,
        live_lint_result=lint,
    )

    artifact = mod.build_artifact(
        source_path=source,
        source_script_path=script,
        source_artifact=_good_a1_artifact(),
        audit=audit,
        summarizer_result=summary,
        adversarial_result=adversarial,
        live_lint_result=lint,
        preconditions_checked={"ok": True, "source_artifact_present": True},
        duration_s=0.0,
    )
    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["duration_s"] == mod.DURATION_FLOOR_S
    assert artifact["source_artifact_checksum"] == mod.file_checksum(source)
    assert artifact["source_script_checksum"] == mod.file_checksum(script)

    result_path = mod.write_artifact(artifact, root=tmp_path)
    report_path = mod.append_markdown_report(artifact, root=tmp_path)
    mod.append_markdown_report(artifact, root=tmp_path)

    loaded = json.loads(result_path.read_text(encoding="utf-8"))
    report_text = report_path.read_text(encoding="utf-8")
    assert loaded == artifact
    assert report_text.count("## Experiment 4865 .448 A1 Fork Probe Audit") == 1
    assert "a1_genuinely_diagnostic" in report_text

    broken = dict(artifact)
    broken.update(
        {
            "a1_genuinely_diagnostic": "yes",
            "planner_blind_confirmed": "yes",
            "positive_control_confirmed": "yes",
            "numbers_match_fork": "yes",
            "field_principles": {},
            "inference_substrate": "live_llm_inference",
            "checks": [],
            "non_diagnostic_reasons": "none",
            "duration_s": 0.0,
            "reproducibility_checksum": "sha256:bad",
        }
    )
    errors = mod.artifact_schema_errors(broken)
    for expected in (
        "a1_genuinely_diagnostic_must_be_bool",
        "planner_blind_confirmed_must_be_bool",
        "positive_control_confirmed_must_be_bool",
        "numbers_match_fork_must_be_bool",
        "field_principles_mismatch",
        "inference_substrate_mismatch",
        "checks_must_be_dict",
        "non_diagnostic_reasons_must_be_list",
        "duration_below_aggregation_floor",
        "reproducibility_checksum_mismatch",
    ):
        assert expected in errors
    with pytest.raises(ValueError, match="a1_genuinely_diagnostic_must_be_bool"):
        mod.write_artifact(broken, root=tmp_path)


def test_req_arc_wmte_4865_run_checked_in_blocked_a1_artifact() -> None:
    """SCENARIO-ARC-WMTE-4865-NON-TEST-CLASSIFICATION: checked-in blocked A1 is void."""

    artifact = mod.run(root=REPO, write=False)

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"].startswith("complete_a1_fork_probe_non_test_")
    assert artifact["a1_genuinely_diagnostic"] is False
    assert artifact["planner_blind_confirmed"] is True
    assert artifact["positive_control_confirmed"] is False
    assert artifact["numbers_match_fork"] is False
    assert artifact["live_path_reachable_confirmed"] is False
    assert artifact["solve_provenance_confirmed"] is True
    assert "positive_control_not_migrated" in artifact["non_diagnostic_reasons"]
    assert "n_games_measured_below_3" in artifact["non_diagnostic_reasons"]
    assert "live_path_unreachable" in artifact["non_diagnostic_reasons"]
    assert artifact["summarizer_result"]["returncode"] == 2
    assert artifact["adversarial_result"]["flag_count"] >= 1


def test_scenario_arc_wmte_4865_blocked_preconditions_do_not_fabricate(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-4865: missing A1 inputs produce blocked audit output."""

    artifact = mod.run(root=tmp_path, write=True)

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == "blocked_a1_artifact_missing"
    assert artifact["a1_genuinely_diagnostic"] is False
    assert artifact["checks"] == {}
    assert "source_artifact_present" in artifact["preconditions_checked"]
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
    assert (tmp_path / mod.AUDIT_REPORT_RELATIVE_PATH).exists()


def test_req_arc_wmte_4865_defensive_branch_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-4865: malformed inputs fail closed without fabricated trust."""

    assert mod._safe_suffix([]) == "audited"
    assert mod._mapping([]) == {}
    assert mod._finite_float(True) is None
    assert mod._finite_float("0.5") is None
    assert mod._finite_float(float("nan")) is None
    assert (
        mod._computed_fork_verdict({"a": _row("a", accuracy=0.2, bucket="NEVER_ENUMERATED")})
        is None
    )
    assert (
        mod._computed_fork_verdict(
            {
                "a": _row("a", accuracy=0.8, bucket="NEVER_ENUMERATED"),
                "b": _row("b", accuracy=0.7, bucket="NEVER_ENUMERATED"),
                "c": _row("c", accuracy=0.9, bucket="ENUMERATED_BUT_LOST"),
            }
        )
        == "PLANNER_GAP"
    )
    assert (
        mod._computed_fork_verdict(
            {
                "a": _row("a", accuracy=0.2, bucket="NEVER_ENUMERATED"),
                "b": _row("b", accuracy=0.3, bucket="NEVER_ENUMERATED"),
                "c": _row("c", accuracy=0.4, bucket="NEVER_ENUMERATED"),
            }
        )
        == "INDUCER_CEILING"
    )
    assert mod._call_name(ast.parse("(lambda: None)()").body[0].value.func) == ""
    assert mod._first_parent_call(ast.Name(id="x")) is None

    not_object = tmp_path / "list.json"
    not_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        mod._read_json(not_object)

    summary, adversarial, lint = _clean_auxiliary_results()
    malformed_source_audit = mod.audit_a1_artifact(
        _good_a1_artifact(),
        source_text="def nope(:",
        summarizer_result=summary,
        adversarial_result=adversarial,
        live_lint_result=lint,
    )
    assert "a1_source_not_parseable" in malformed_source_audit["non_diagnostic_reasons"]

    missing_function_audit = mod.audit_a1_artifact(
        _good_a1_artifact(),
        source_text="def unrelated(): pass",
        summarizer_result=summary,
        adversarial_result=adversarial,
        live_lint_result=lint,
    )
    assert (
        "measure_game_with_live_induce_plan_missing"
        in (missing_function_audit["non_diagnostic_reasons"])
    )

    no_classification_audit = mod.audit_a1_artifact(
        _good_a1_artifact(),
        source_text="""
def measure_game_with_live_induce_plan(game, winning_prefix, proposer):
    policy = object()
    policy._induce_and_plan()
    return {}
""",
        summarizer_result=summary,
        adversarial_result=adversarial,
        live_lint_result=lint,
    )
    assert (
        "winning_prefix_not_used_for_classification"
        in (no_classification_audit["non_diagnostic_reasons"])
    )
    assert "load_engine_not_called" in no_classification_audit["non_diagnostic_reasons"]

    missing_induce_audit = mod.audit_a1_artifact(
        _good_a1_artifact(),
        source_text="""
def measure_game_with_live_induce_plan(game, winning_prefix, proposer):
    from carnot.agentic import arc_executable_world_model as e3
    engine, is_done = e3.load_engine(game)
    plan = e3.plan_in_model(engine, is_done, None)
    return classify_planned_pool(game, winning_prefix, plan, planner_reached_l1_win=False)
""",
        summarizer_result=summary,
        adversarial_result=adversarial,
        live_lint_result=lint,
    )
    assert "induce_and_plan_not_called" in missing_induce_audit["non_diagnostic_reasons"]

    bad_control = _good_a1_artifact()
    bad_control["positive_control_game"] = "not_tu93"
    bad_control["positive_control_fork"] = _control()
    assert (
        "positive_control_not_tu93"
        in mod.audit_a1_artifact(
            bad_control,
            source_text=_good_source(),
            summarizer_result=summary,
            adversarial_result=adversarial,
            live_lint_result=lint,
        )["non_diagnostic_reasons"]
    )

    bad_numbers = _good_a1_artifact()
    bad_numbers["n_games_measured"] = "many"
    bad_numbers["fork_verdict"] = "MAYBE"
    bad_numbers["coverage_migration_count"] = 9
    bad_numbers["median_engine_heldout_accuracy"] = 0.1
    bad_numbers["per_game_fork"] = {
        "aa00": _row("aa00", accuracy=1.5, bucket="MAYBE", migrated="yes"),
        "bb00": [],
        "cc00": _row("cc00", accuracy=0.8, bucket="COVERED", migrated=False),
    }
    bad_number_audit = mod.audit_a1_artifact(
        bad_numbers,
        source_text=_good_source(),
        summarizer_result=summary,
        adversarial_result=adversarial,
        live_lint_result=lint,
    )
    for reason in (
        "invalid_per_game_row",
        "invalid_planned_bucket",
        "invalid_engine_heldout_accuracy",
        "invalid_migrated_flag",
        "n_games_measured_not_integer",
        "row_migrated_mismatch",
        "invalid_fork_verdict",
        "coverage_migration_count_mismatch",
        "median_engine_heldout_accuracy_mismatch",
    ):
        assert reason in bad_number_audit["non_diagnostic_reasons"]

    no_valid_median = _good_a1_artifact()
    no_valid_median["per_game_fork"] = {
        "aa00": _row("aa00", accuracy=1.5, bucket="MAYBE", migrated=True)
    }
    no_valid_median["n_games_measured"] = 1
    no_valid_median["median_engine_heldout_accuracy"] = 0.3
    assert (
        "median_engine_heldout_accuracy_mismatch"
        in mod.audit_a1_artifact(
            no_valid_median,
            source_text=_good_source(),
            summarizer_result=summary,
            adversarial_result=adversarial,
            live_lint_result=lint,
        )["non_diagnostic_reasons"]
    )

    tool_failed = mod.audit_a1_artifact(
        _good_a1_artifact(),
        source_text=_good_source(),
        summarizer_result={"returncode": 2},
        adversarial_result={"loaded": True, "flag_count": 1},
        live_lint_result=lint,
    )
    assert tool_failed["a1_genuinely_diagnostic"] is True
    assert tool_failed["checks"]["summarizer_and_adversarial_verify"]["passed"] is False

    source = tmp_path / mod.SOURCE_ARTIFACT_RELATIVE_PATH
    script = tmp_path / mod.SOURCE_SCRIPT_RELATIVE_PATH
    source.parent.mkdir(parents=True, exist_ok=True)
    script.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(json.dumps(_good_a1_artifact()), encoding="utf-8")
    script.write_text(_good_source(), encoding="utf-8")
    audit = mod.audit_a1_artifact(
        _good_a1_artifact(),
        source_text=_good_source(),
        summarizer_result=summary,
        adversarial_result=adversarial,
        live_lint_result=lint,
    )
    artifact = mod.build_artifact(
        source_path=source,
        source_script_path=script,
        source_artifact=_good_a1_artifact(),
        audit=audit,
        summarizer_result=summary,
        adversarial_result=adversarial,
        live_lint_result=lint,
        preconditions_checked={"ok": True},
        duration_s=1.0,
    )
    broken = dict(artifact)
    broken.update(
        {
            "honest_verdict": "bad",
            "random_seed": 1,
            "reproducibility_checksum": "sha256:bad",
        }
    )
    assert "honest_verdict_missing_terminal_prefix" in mod.artifact_schema_errors(broken)
    assert "random_seed_mismatch" in mod.artifact_schema_errors(broken)
    impossible = dict(artifact, non_diagnostic_reasons=["boom"])
    impossible["reproducibility_checksum"] = "sha256:" + mod.payload_checksum(impossible)
    assert "diagnostic_artifact_has_failure_reasons" in mod.artifact_schema_errors(impossible)

    report_artifact = dict(artifact)
    report_artifact["checks"] = dict(artifact["checks"], malformed=[])
    assert "malformed" not in mod.render_markdown_section(report_artifact)

    existing_root = tmp_path / "existing_report"
    existing_report = existing_root / mod.AUDIT_REPORT_RELATIVE_PATH
    existing_report.parent.mkdir(parents=True)
    existing_report.write_text("# Prior Audit\n", encoding="utf-8")
    mod.append_markdown_report(artifact, root=existing_root)
    assert "Experiment 4865 .448 A1 Fork Probe Audit" in existing_report.read_text(encoding="utf-8")

    monkeypatch.setattr(mod, "check_preconditions", lambda _root: {"ok": True})
    monkeypatch.setattr(mod, "run_summarizer", lambda _path: summary)
    monkeypatch.setattr(mod, "run_adversarial_verify", lambda _path: adversarial)
    monkeypatch.setattr(mod, "run_arc_orphan_solver_lint", lambda _root: lint)
    written = mod.run(root=tmp_path, write=True, now=iter([1.0, 1.25]).__next__)
    assert written["duration_s"] == 0.25
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()

    blocked_no_write = mod.run(root=tmp_path / "missing_no_write", write=False)
    assert blocked_no_write["honest_verdict"] == "blocked_a1_artifact_missing"
    assert not (tmp_path / "missing_no_write" / mod.RESULT_RELATIVE_PATH).exists()

    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["boom"])
    with pytest.raises(ValueError, match="boom"):
        mod.run(root=tmp_path, write=False, now=iter([2.0, 2.25]).__next__)
