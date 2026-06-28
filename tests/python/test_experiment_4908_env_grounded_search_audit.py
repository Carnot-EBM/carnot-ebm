"""Tests for Exp 4908 env-grounded search A1/A1b adversarial audit.

Spec refs: REQ-ARC-WMTE-4908,
SCENARIO-ARC-WMTE-4908-A1-AUDIT,
SCENARIO-ARC-WMTE-4908-A1B-LIVE-OR-GATE-SKIPPED,
SCENARIO-ARC-WMTE-4908-BLOCKED-UPSTREAM.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4908_env_grounded_search_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _row(
    game: str,
    *,
    baseline: float = 0.04,
    env: float = 0.0,
    bucket: str = "NEVER_ENUMERATED",
    actions: int | None = None,
    states: int = 2,
    real_reads: int = 3,
    value_predictions: int = 0,
) -> dict[str, Any]:
    return {
        "game": game,
        "first_win_baseline": baseline,
        "first_win_env_grounded": env,
        "delta": round(env - baseline, 6),
        "actions_to_first_win": actions,
        "states_expanded": states,
        "bucket": bucket,
        "baseline_bucket": "NEVER_ENUMERATED",
        "migrated": bool(env > 0.0 and bucket == "COVERED"),
        "change_value_predictions_used": value_predictions,
        "real_env_value_reads": real_reads,
        "live_path_methods_called": [
            "StepwiseExplorer.action_prior",
            "arc_executable_world_model.load_engine",
            "arc_executable_world_model.plan_in_model",
        ],
    }


def _a1_artifact(*, lifted: bool = False) -> dict[str, Any]:
    if lifted:
        rows = {
            "cd82": _row("cd82", baseline=0.0, env=1.0, bucket="COVERED", actions=4),
            "cn04": _row("cn04", baseline=0.0, env=1.0, bucket="COVERED", actions=6),
            "ls20": _row("ls20", baseline=0.0, env=1.0, bucket="COVERED", actions=8),
        }
        median = 1.0
        ci95 = [1.0, 1.0]
        fork = "ENV_GROUNDED_SEARCH_UNLOCKS_FIRST_WIN"
        verdict = "success_env_grounded_search_first_win_unlocked_1.000000"
        migrations = 3
    else:
        rows = {
            "cd82": _row("cd82"),
            "cn04": _row("cn04"),
            "ls20": _row("ls20"),
        }
        median = -0.04
        ci95 = [-0.04, -0.04]
        fork = "WALL_DEEPER_THAN_VALUE_PREDICTION"
        verdict = "complete_env_grounded_search_no_first_win_lift_WALL_DEEPER_THAN_VALUE_PREDICTION"
        migrations = 0
    return {
        "experiment_id": 4903,
        "honest_verdict": verdict,
        "fork_verdict": fork,
        "value_grounded_first_win_delta_median": median,
        "value_grounded_first_win_delta_ci95": ci95,
        "median_actions_to_first_win": 6.0 if lifted else None,
        "coverage_migration_count": migrations,
        "change_location_prior_used_not_value": True,
        "positive_control_game": "tu93",
        "positive_control_non_degenerate": True,
        "positive_control_result": {
            "game": "tu93",
            "location_ranker_non_degenerate": True,
            "true_changing_action_rank": 2,
            "non_degenerate_rank_threshold": 5,
            "actual_changing_actions_seen": 4,
            "change_value_predictions_used": 0,
            "real_env_value_reads": 4,
        },
        "planner_blind_to_banked_answer": True,
        "verifier_is_oracle": False,
        "live_path_reachable": True,
        "solve_provenance": "development_proxy",
        "inference_substrate": "live_llm_inference",
        "duration_s": 60.0,
        "n_games_measured": len(rows),
        "per_game_first_win": rows,
        "env_grounded_search_config": {
            "bootstrap_iterations": 1000,
            "bounded_action_cost": 80,
            "change_location_prior_only": True,
            "env_supplies_change_value": True,
            "heldout_games": list(rows),
            "planner_blind_to_banked_answer": True,
        },
        "preconditions_checked": {"planner_blind_to_banked_answer": True},
        "random_seed": 20260628,
        "model_specs": {"name": "Qwen3.5-9B-MTP", "backend": "gpu0_cuda"},
    }


def _a1_source(*, leaked_prefix: bool = False, no_real_transition: bool = False) -> str:
    if leaked_prefix:
        return """
def measure_game_with_env_grounded_search(winning_prefix):
    ranked = list(winning_prefix)
    return ranked
"""
    if no_real_transition:
        return """
class ChangeLocationActionPrior:
    def score(self, grid, candidate):
        return 0.0

def interleaved_env_grounded_search(start_grid, engine, legal_actions, real_transition):
    next_grid = engine(start_grid, 1, None)
    return {"change_value_predictions_used": 1, "real_env_value_reads": 0}

def measure_game_with_env_grounded_search(winning_prefix):
    return _classify_after_search(winning_prefix=winning_prefix)
"""
    return """
class ChangeLocationActionPrior:
    def score(self, grid, candidate):
        predicted = self.engine(grid, 1, None)
        return count_nonzero(predicted != grid)

def interleaved_env_grounded_search(start_grid, engine, legal_actions, real_transition):
    prior = ChangeLocationActionPrior(engine)
    ranked = prior.rank(start_grid, legal_actions(start_grid))
    next_grid = real_transition(start_grid, ranked[0])
    real_env_value_reads += 1
    return {"change_value_predictions_used": 0, "real_env_value_reads": real_env_value_reads}

def measure_game_with_env_grounded_search(winning_prefix):
    result = interleaved_env_grounded_search(None, None, None, None)
    return _classify_after_search(winning_prefix=winning_prefix, path=result)
"""


def _a1b_artifact(source: dict[str, Any] | None = None) -> dict[str, Any]:
    a1 = source or _a1_artifact()
    rows = {
        game: {
            "game": game,
            "value_acc_code_baseline": 0.2,
            "value_acc_latent_action": 0.2,
            "delta": 0.0,
            "fit_transition_ids": ["fit:0", "fit:1"],
            "heldout_transition_ids": ["heldout:0", "heldout:1"],
            "baseline_transition_ids": ["heldout:0", "heldout:1"],
            "live_path_methods_called": [
                "LatentActionInterface",
                "arc_executable_world_model.load_engine",
            ],
        }
        for game in a1["per_game_first_win"]
    }
    return {
        "experiment_id": 4904,
        "honest_verdict": "complete_latent_action_no_value_lift_representation_invariant_4_classes",
        "fork_verdict": "VALUE_GAP_REPRESENTATION_INVARIANT_4_CLASSES",
        "duration_s": 178.75,
        "ran_genuinely_live": True,
        "delta_on_truly_heldout_split": True,
        "verifier_is_oracle": False,
        "live_path_reachable": True,
        "solve_provenance": "development_proxy",
        "inference_substrate": "live_llm_inference",
        "per_game_value_gap": rows,
        "latent_action_config": {
            "heldout_games": list(a1["per_game_first_win"]),
            "planner_blind_to_banked_answer": True,
        },
        "model_specs": {"name": "Qwen3.5-9B-MTP", "backend": "gpu0_cuda"},
    }


def _summary() -> dict[str, Any]:
    return {"returncode": 0, "stdout": "LIVE re-check: clean", "stderr": ""}


def _adv(flags: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"loaded": True, "flag_count": len(flags or []), "flags": list(flags or [])}


def _lint(passed: bool = True) -> dict[str, Any]:
    return {
        "passed": passed,
        "returncode": 0 if passed else 1,
        "stdout_tail": "OK",
        "stderr_tail": "",
    }


def test_req_arc_wmte_4908_spec_declares_audit_contract() -> None:
    """REQ-ARC-WMTE-4908: OpenSpec anchors fields, scenarios, and result path."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4908",
        "SCENARIO-ARC-WMTE-4908-A1-AUDIT",
        "SCENARIO-ARC-WMTE-4908-A1B-LIVE-OR-GATE-SKIPPED",
        "SCENARIO-ARC-WMTE-4908-BLOCKED-UPSTREAM",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4908_clean_a1_and_a1b_pass() -> None:
    """SCENARIO-ARC-WMTE-4908-A1-AUDIT: all load-bearing checks pass."""

    a1 = _a1_artifact()
    audit = mod.audit_sources(
        a1_artifact=a1,
        a1_source_text=_a1_source(),
        a1_summarizer_result=_summary(),
        a1_adversarial_result=_adv(),
        a1b_artifact=_a1b_artifact(a1),
        a1b_summarizer_result=_summary(),
        a1b_adversarial_result=_adv(),
        live_lint_result=_lint(),
    )

    assert audit["honest_verdict"] == "complete_a1_a1b_audited"
    assert audit["a1_value_from_real_env"] is True
    assert audit["a1_planner_blind"] is True
    assert audit["a1_positive_control_non_degenerate"] is True
    assert audit["a1_numbers_match_fork"] is True
    assert audit["a1_live_path_reachable"] is True
    assert audit["a1_trustworthy"] is True
    assert audit["a1b_ran_genuinely_live"] is True
    assert audit["a1b_gate_skipped"] is False
    assert audit["adversarial_flags_found"] is False
    assert all(row["passed"] for row in audit["claim_audit_table"])
    assert audit["checks"]["a1_numbers_match_fork"]["computed_fork_verdict"] == (
        "WALL_DEEPER_THAN_VALUE_PREDICTION"
    )


def test_scenario_arc_wmte_4908_hostile_a1_variants_fail_closed() -> None:
    """SCENARIO-ARC-WMTE-4908-A1-AUDIT: hostile A1 inputs do not pass the audit."""

    cases: list[tuple[dict[str, Any], str, str]] = []
    predicted = _a1_artifact()
    predicted["per_game_first_win"]["cd82"]["change_value_predictions_used"] = 1
    cases.append((predicted, _a1_source(), "a1_model_change_value_predictions_used"))

    no_reads = _a1_artifact()
    no_reads["per_game_first_win"]["cd82"]["real_env_value_reads"] = 0
    cases.append((no_reads, _a1_source(no_real_transition=True), "a1_real_env_value_reads_missing"))

    leaked = _a1_artifact()
    cases.append(
        (leaked, _a1_source(leaked_prefix=True), "banked_prefix_used_before_classification")
    )

    degenerate = _a1_artifact()
    degenerate["positive_control_non_degenerate"] = False
    degenerate["positive_control_result"]["location_ranker_non_degenerate"] = False
    degenerate["positive_control_result"]["true_changing_action_rank"] = 99
    cases.append((degenerate, _a1_source(), "tu93_change_location_prior_degenerate"))

    wrong_fork = _a1_artifact()
    wrong_fork["fork_verdict"] = "ENV_GROUNDED_SEARCH_UNLOCKS_FIRST_WIN"
    cases.append((wrong_fork, _a1_source(), "fork_verdict_mismatch"))

    for a1, source, reason in cases:
        audit = mod.audit_sources(
            a1_artifact=a1,
            a1_source_text=source,
            a1_summarizer_result=_summary(),
            a1_adversarial_result=_adv(),
            a1b_artifact=_a1b_artifact(a1),
            a1b_summarizer_result=_summary(),
            a1b_adversarial_result=_adv(),
            live_lint_result=_lint(),
        )
        assert reason in audit["a1_failure_reasons"]
        assert audit["a1_trustworthy"] is False


def test_scenario_arc_wmte_4908_a1b_live_gate_and_adversarial_flags() -> None:
    """SCENARIO-ARC-WMTE-4908-A1B-LIVE-OR-GATE-SKIPPED: live or skipped only."""

    lifted = _a1_artifact(lifted=True)
    skipped = mod.audit_sources(
        a1_artifact=lifted,
        a1_source_text=_a1_source(),
        a1_summarizer_result=_summary(),
        a1_adversarial_result=_adv(),
        a1b_artifact=None,
        a1b_summarizer_result=None,
        a1b_adversarial_result=None,
        live_lint_result=_lint(),
    )
    assert skipped["a1b_gate_skipped"] is True
    assert skipped["a1b_ran_genuinely_live"] is True

    missing = mod.audit_sources(
        a1_artifact=_a1_artifact(),
        a1_source_text=_a1_source(),
        a1_summarizer_result=_summary(),
        a1_adversarial_result=_adv(),
        a1b_artifact=None,
        a1b_summarizer_result=None,
        a1b_adversarial_result=None,
        live_lint_result=_lint(),
    )
    assert missing["a1b_ran_genuinely_live"] is False
    assert "a1b_missing_after_low_first_win_a1" in missing["a1b_failure_reasons"]

    short = _a1b_artifact()
    short["duration_s"] = 13.7
    short["ran_genuinely_live"] = False
    flagged = mod.audit_sources(
        a1_artifact=_a1_artifact(),
        a1_source_text=_a1_source(),
        a1_summarizer_result=_summary(),
        a1_adversarial_result=_adv([{"kind": "TAUTOLOGY", "severity": "warn"}]),
        a1b_artifact=short,
        a1b_summarizer_result=_summary(),
        a1b_adversarial_result=_adv([{"kind": "DURATION_TOO_SHORT", "severity": "critical"}]),
        live_lint_result=_lint(False),
    )
    assert flagged["adversarial_flags_found"] is True
    assert flagged["a1_live_path_reachable"] is False
    assert flagged["a1b_ran_genuinely_live"] is False
    assert "a1b_duration_too_short_flagged" in flagged["a1b_failure_reasons"]

    wrong_split = _a1b_artifact()
    wrong_split["latent_action_config"]["heldout_games"] = ["other"]
    audit = mod.audit_sources(
        a1_artifact=_a1_artifact(),
        a1_source_text=_a1_source(),
        a1_summarizer_result=_summary(),
        a1_adversarial_result=_adv(),
        a1b_artifact=wrong_split,
        a1b_summarizer_result=_summary(),
        a1b_adversarial_result=_adv(),
        live_lint_result=_lint(),
    )
    assert "a1b_not_same_heldout_split_as_a1" in audit["a1b_failure_reasons"]


def test_req_arc_wmte_4908_build_schema_write_and_run_paths(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """REQ-ARC-WMTE-4908: artifacts are checksum-stable and blocked preconditions exit."""

    blocked = mod.run(root=tmp_path, write=True, now=iter([1.0, 1.1]).__next__)
    assert blocked["honest_verdict"] == "blocked_upstream_artifact_missing"
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
    assert mod.artifact_schema_errors(blocked) == []

    a1_path = tmp_path / mod.A1_ARTIFACT_RELATIVE_PATH
    a1_source_path = tmp_path / mod.A1_SCRIPT_RELATIVE_PATH
    a1b_path = tmp_path / mod.A1B_ARTIFACT_RELATIVE_PATH
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    for path, content in (
        (a1_path, json.dumps(_a1_artifact())),
        (a1_source_path, _a1_source()),
        (a1b_path, json.dumps(_a1b_artifact())),
        (spec_path, "REQ-ARC-WMTE-4908"),
        (tmp_path / "scripts/summarize_artifact.py", ""),
        (tmp_path / "scripts/adversarial_verify.py", ""),
        (tmp_path / "scripts/arc_orphan_solver_lint.py", ""),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    monkeypatch.setattr(mod, "run_summarizer", lambda _path: _summary())
    monkeypatch.setattr(mod, "run_adversarial_verify", lambda _path: _adv())
    monkeypatch.setattr(mod, "run_arc_orphan_solver_lint", lambda _root: _lint())

    complete = mod.run(root=tmp_path, write=True, now=iter([2.0, 2.1]).__next__)
    assert complete["honest_verdict"] == "complete_a1_a1b_audited"
    assert complete["a1_trustworthy"] is True
    assert complete["a1_artifact_checksum"] == mod.file_checksum(a1_path)
    assert complete["a1_script_checksum"] == mod.file_checksum(a1_source_path)
    assert complete["a1b_artifact_checksum"] == mod.file_checksum(a1b_path)
    assert mod.artifact_schema_errors(complete) == []
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == complete

    broken = dict(complete)
    broken.update(
        {
            "field_principles": {},
            "honest_verdict": "bad",
            "a1_value_from_real_env": "yes",
            "a1_planner_blind": "yes",
            "a1_positive_control_non_degenerate": "yes",
            "a1_numbers_match_fork": "yes",
            "a1_trustworthy": "yes",
            "a1b_ran_genuinely_live": "yes",
            "a1b_gate_skipped": "yes",
            "adversarial_flags_found": "no",
            "inference_substrate": "live_llm_inference",
            "claim_audit_table": {},
            "checks": [],
            "a1_failure_reasons": "none",
            "a1b_failure_reasons": "none",
            "duration_s": 0.0,
            "random_seed": 0,
            "reproducibility_checksum": "sha256:bad",
        }
    )
    errors = mod.artifact_schema_errors(broken)
    for expected in (
        "field_principles_mismatch",
        "honest_verdict_missing_terminal_prefix",
        "a1_value_from_real_env_must_be_bool",
        "a1_trustworthy_must_be_bool",
        "a1b_gate_skipped_must_be_bool",
        "adversarial_flags_found_must_be_bool",
        "claim_audit_table_must_be_list",
        "checks_must_be_dict",
        "inference_substrate_mismatch",
        "random_seed_mismatch",
        "duration_below_aggregation_floor",
        "reproducibility_checksum_mismatch",
    ):
        assert expected in errors
    with pytest.raises(ValueError):
        mod.write_artifact(broken, root=tmp_path)


def test_req_arc_wmte_4908_checked_in_artifacts_match_audit() -> None:
    """REQ-ARC-WMTE-4908: checked-in A1/A1b artifacts produce the requested audit."""

    artifact = mod.run(root=REPO, write=False)

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == "complete_a1_a1b_audited"
    assert artifact["a1_value_from_real_env"] is True
    assert artifact["a1_planner_blind"] is True
    assert artifact["a1_positive_control_non_degenerate"] is True
    assert artifact["a1_numbers_match_fork"] is True
    assert artifact["a1_live_path_reachable"] is True
    assert artifact["a1_trustworthy"] is True
    assert artifact["a1b_ran_genuinely_live"] is True
    assert artifact["a1b_gate_skipped"] is False
    assert artifact["adversarial_flags_found"] is False
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_req_arc_wmte_4908_defensive_helpers(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4908: malformed inputs fail closed without fabricated passes."""

    assert mod._finite_float(True) is None
    assert mod._finite_float("nan") is None
    assert mod._finite_float("x") is None
    non_object = tmp_path / "array.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        mod._read_json(non_object)

    assert mod._full_call_name(ast.Constant(value=1)) == ""
    assert mod._call_name(None) == ""
    assert mod._call_name(ast.Name(id="f")) == "f"
    assert mod._call_name(ast.Attribute(value=ast.Name(id="obj"), attr="method")) == "method"
    assert mod._call_name(ast.Constant(value=1)) == ""
    expr = ast.parse("winning_prefix + 1")
    mod._attach_ast_parents(expr)
    assert (
        mod._first_parent_call(next(n for n in ast.walk(expr) if isinstance(n, ast.Name))) is None
    )

    value_parse, value_reasons = mod._source_value_checks("def bad(")
    blind_parse, blind_reasons = mod._source_planner_blind_checks("def bad(")
    missing_func, missing_func_reasons = mod._source_planner_blind_checks("x = 1")
    no_refs, no_ref_reasons = mod._source_planner_blind_checks(
        "def measure_game_with_env_grounded_search(winning_prefix):\n    return 1\n"
    )
    assert value_parse["passed"] is False
    assert "a1_source_not_parseable" in value_reasons
    assert blind_parse["passed"] is False
    assert "a1_source_not_parseable" in blind_reasons
    assert missing_func["passed"] is False
    assert "measure_game_with_env_grounded_search_missing" in missing_func_reasons
    assert no_refs["passed"] is False
    assert "winning_prefix_not_used_for_classification" in no_ref_reasons

    bad_value = _a1_artifact()
    bad_value["change_location_prior_used_not_value"] = False
    bad_value["env_grounded_search_config"]["change_location_prior_only"] = False
    bad_value["env_grounded_search_config"]["env_supplies_change_value"] = False
    bad_value["per_game_first_win"] = {}
    bad_value["positive_control_result"]["change_value_predictions_used"] = 1
    bad_value["positive_control_result"]["real_env_value_reads"] = 0
    _check, reasons = mod._a1_value_from_real_env_check(bad_value, _a1_source())
    for expected in (
        "a1_artifact_location_prior_flag_false",
        "a1_config_not_location_prior_only",
        "a1_config_env_supplies_value_missing",
        "a1_per_game_first_win_missing",
        "positive_control_model_change_value_predictions_used",
        "positive_control_real_env_value_reads_missing",
    ):
        assert expected in reasons

    bad_blind = _a1_artifact()
    bad_blind["planner_blind_to_banked_answer"] = False
    bad_blind["env_grounded_search_config"]["planner_blind_to_banked_answer"] = False
    bad_blind["preconditions_checked"]["planner_blind_to_banked_answer"] = False
    _check, reasons = mod._a1_planner_blind_check(bad_blind, _a1_source())
    for expected in (
        "a1_planner_blind_artifact_flag_false",
        "a1_planner_blind_config_flag_false",
        "a1_planner_blind_precondition_flag_false",
    ):
        assert expected in reasons

    bad_control = _a1_artifact()
    bad_control["positive_control_game"] = "not-tu93"
    bad_control["positive_control_result"]["game"] = "not-tu93"
    bad_control["positive_control_result"]["actual_changing_actions_seen"] = 0
    bad_control["positive_control_result"]["change_value_predictions_used"] = 1
    bad_control["positive_control_result"]["real_env_value_reads"] = 0
    _check, reasons = mod._a1_positive_control_check(bad_control)
    for expected in (
        "tu93_positive_control_missing",
        "tu93_no_actual_changing_actions_seen",
        "tu93_model_change_value_predictions_used",
        "tu93_real_env_reads_missing",
    ):
        assert expected in reasons

    bad_numbers = _a1_artifact()
    bad_numbers["per_game_first_win"]["cd82"].update(
        {
            "first_win_env_grounded": "bad",
            "delta": 0.0,
            "bucket": "BAD",
            "migrated": "yes",
            "states_expanded": "bad",
            "actions_to_first_win": "bad",
        }
    )
    bad_numbers["per_game_first_win"]["cn04"].update(
        {
            "first_win_env_grounded": 0.5,
            "delta": 0.0,
            "states_expanded": -1,
            "actions_to_first_win": -1,
        }
    )
    bad_numbers["per_game_first_win"].pop("ls20")
    bad_numbers["n_games_measured"] = "bad"
    bad_numbers["env_grounded_search_config"]["bootstrap_iterations"] = "bad"
    bad_numbers["env_grounded_search_config"]["bounded_action_cost"] = "bad"
    bad_numbers["value_grounded_first_win_delta_median"] = 999
    bad_numbers["value_grounded_first_win_delta_ci95"] = [999, 999]
    bad_numbers["median_actions_to_first_win"] = 999
    bad_numbers["coverage_migration_count"] = 999
    _check, reasons = mod._a1_numbers_match_fork_check(bad_numbers)
    for expected in (
        "invalid_first_win_metric",
        "invalid_bucket",
        "invalid_migrated_flag",
        "invalid_states_expanded",
        "invalid_actions_to_first_win",
        "first_win_delta_mismatch",
        "n_games_measured_below_3",
        "n_games_measured_not_integer",
        "n_games_measured_mismatch",
        "first_win_delta_median_mismatch",
        "first_win_delta_ci95_mismatch",
        "median_actions_to_first_win_mismatch",
        "coverage_migration_count_mismatch",
    ):
        assert expected in reasons
    assert mod._bootstrap_iterations(bad_numbers) == mod.exp4903.DEFAULT_BOOTSTRAP_ITERATIONS
    assert mod._bounded_action_cost(bad_numbers) == mod.exp4903.DEFAULT_BOUNDED_ACTION_COST

    bad_live = _a1_artifact()
    bad_live["live_path_reachable"] = False
    bad_live["per_game_first_win"]["cd82"]["live_path_methods_called"] = []
    bad_live["verifier_is_oracle"] = True
    bad_live["solve_provenance"] = "registry_bank"
    _check, reasons = mod._a1_live_path_check(bad_live, _lint())
    for expected in (
        "a1_artifact_live_path_false",
        "a1_live_path_methods_missing",
        "a1_verifier_is_oracle",
        "a1_solve_provenance_not_development_proxy",
    ):
        assert expected in reasons

    no_config_a1 = _a1_artifact()
    no_config_a1.pop("env_grounded_search_config")
    no_config_a1b = _a1b_artifact()
    no_config_a1b.pop("latent_action_config")
    assert mod._heldout_games_from_a1(no_config_a1) == sorted(no_config_a1["per_game_first_win"])
    assert mod._heldout_games_from_a1b(no_config_a1b) == sorted(no_config_a1b["per_game_value_gap"])

    bad_a1b = _a1b_artifact()
    bad_a1b["inference_substrate"] = "aggregation_from_upstream_artifacts"
    bad_a1b["verifier_is_oracle"] = True
    bad_a1b["live_path_reachable"] = False
    bad_a1b["solve_provenance"] = "registry_bank"
    bad_a1b["per_game_value_gap"]["cd82"]["live_path_methods_called"] = []
    _check, reasons, passed, skipped = mod._a1b_live_check(
        a1_artifact=_a1_artifact(),
        a1b_artifact=bad_a1b,
        a1b_summarizer_result={"returncode": 1},
        a1b_adversarial_result=_adv([{"kind": "CIRCULAR_MOAT_OVERCLAIM"}]),
    )
    assert passed is False
    assert skipped is False
    for expected in (
        "a1b_inference_substrate_not_live_llm",
        "a1b_verifier_is_oracle",
        "a1b_circular_moat_overclaim",
        "a1b_live_path_unreachable",
        "a1b_solve_provenance_not_development_proxy",
        "a1b_summarizer_failed",
    ):
        assert expected in reasons

    summary_failed = mod.audit_sources(
        a1_artifact=_a1_artifact(),
        a1_source_text=_a1_source(),
        a1_summarizer_result={"returncode": 1},
        a1_adversarial_result=_adv(),
        a1b_artifact=_a1b_artifact(),
        a1b_summarizer_result=_summary(),
        a1b_adversarial_result=_adv(),
        live_lint_result=_lint(),
    )
    assert "a1_summarizer_failed" in summary_failed["a1_failure_reasons"]

    trust_mismatch = {
        "honest_verdict": "complete_a1_a1b_audited",
        "field_principles": mod.FIELD_PRINCIPLES,
        "a1_value_from_real_env": True,
        "a1_planner_blind": True,
        "a1_positive_control_non_degenerate": True,
        "a1_numbers_match_fork": True,
        "a1_live_path_reachable": True,
        "a1_trustworthy": False,
        "a1b_ran_genuinely_live": True,
        "a1b_gate_skipped": False,
        "adversarial_flags_found": False,
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "claim_audit_table": [{"claim": "x", "passed": "yes"}],
        "checks": {},
        "a1_failure_reasons": [],
        "a1b_failure_reasons": [],
        "random_seed": mod.RANDOM_SEED,
        "duration_s": mod.DURATION_FLOOR_S,
        "reproducibility_checksum": "",
    }
    trust_mismatch.update(
        {field: None for field in mod.REQUIRED_ARTIFACT_FIELDS if field not in trust_mismatch}
    )
    trust_mismatch["reproducibility_checksum"] = mod.payload_checksum(trust_mismatch)
    schema_errors = mod.artifact_schema_errors(trust_mismatch)
    assert "a1_trustworthy_formula_mismatch" in schema_errors
    assert "claim_audit_table.0" in schema_errors
