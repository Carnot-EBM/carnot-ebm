"""Tests for Exp 4358 `.402` archive / `.403` activation.

Spec refs: REQ-REPORT-4358, SCENARIO-REPORT-4358,
SCENARIO-REPORT-4358-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import archive_v402_activate_v403_4358 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="81 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.401\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.402\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-18'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4357-capstone-v402\n"
        "    result: OK (conductor)\n"
    )
    return head + block * duplicates


def _manifest_text() -> str:
    return (
        "retired_extras:\n"
        "- id: cross_domain_selection_retired_exp4314_v399\n"
        "  experiment_ids: [exp4314]\n"
        "  operator_reopen_required: true\n"
        "  retire_if_same_verdict: true\n"
        "- id: cross_game_value_transfer_retired_exp4342_v401\n"
        "  experiment_ids: [exp4342]\n"
        "  operator_reopen_required: true\n"
        "  retire_if_same_verdict: true\n"
    )


def _registry_text() -> str:
    return (
        "schema_version: 1\n"
        "updated: '2026-06-17'\n"
        "reproducible_total_levels: 26\n"
        "reproducible_total_games: 15\n"
        "provisional_total_levels: 5\n"
    )


def _capstone(**overrides: object) -> dict:
    payload = {
        "honest_verdict": (
            "complete: v402_s3_open_arc_levels_23_action_efficiency_improves_publication_ready"
        ),
        "verifier_thesis_state": "moat_proven_leak_robust_but_s3_utility_open",
        "s3_moat_utility": "open",
        "reproducible_total_levels": 23,
        "paper_ready": True,
        "verifier_is_oracle": False,
        "verifier_is_oracle_honored": True,
        "s3_utility": {"status": "open_flagged_or_missing_s3", "s3_moat_utility": "open"},
        "arc_reproducible_progress": {
            "new_games_since_prior": 1,
            "prior_reproducible_total_levels": 21,
            "prior_reproducible_total_games": 13,
            "reproducible_total_levels": 23,
            "reproducible_total_games": 14,
        },
        "arc_e3_outcomes": {
            "games_with_new_reproducible_levels": ["ka59", "tn36"],
            "ka59": {
                "offline_reproduced": True,
                "reproduced_levels": 1,
                "verifier_is_oracle": True,
            },
            "deeper": {
                "games_with_new_reproducible_levels": ["tn36"],
                "per_target_scorecard": [
                    {
                        "game": "sc25",
                        "offline_reproduced": False,
                        "residual_win_mechanic_gap_class": (
                            "sc25_l2_live_recorded_not_offline_reproduced_spell_delta_gap"
                        ),
                    },
                    {
                        "game": "ar25",
                        "offline_reproduced": False,
                        "residual_win_mechanic_gap_class": (
                            "ar25_l2_hidden_rule_delta_not_reproduced_action7_undo_stack_gap"
                        ),
                    },
                    {"game": "tn36", "offline_reproduced": True, "new_reproduced_level": 7},
                ],
                "verifier_is_oracle": True,
            },
            "tr87_ft09": {
                "status": "partial",
                "per_game_scorecard": [
                    {
                        "game": "tr87",
                        "offline_reproduced": False,
                        "verifier_accuracy": 0.0,
                        "residual_mismatch_class": "missing_world_model_rule_gap_actions_1_2_3_4",
                    },
                    {
                        "game": "ft09",
                        "offline_reproduced": False,
                        "verifier_accuracy": 0.05,
                        "residual_mismatch_class": "missing_world_model_rule_gap_actions_6",
                    },
                ],
                "verifier_is_oracle": True,
            },
        },
        "action_efficiency": {
            "action_efficiency_improves": True,
            "held_out_actions_baseline": 25,
            "held_out_actions_learned": 16,
            "positive_control_passed": True,
            "reproduction_gated": True,
            "verifier_is_oracle": False,
        },
        "publication_gate": {"paper_ready": True, "unmet_gates": []},
        "flagged_artifacts_excluded": [
            {"experiment_id": 4348, "reason": "flagged_adversarial", "live_critical": True}
        ],
    }
    payload.update(overrides)
    return payload


def _s3(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "controls_not_differentiable",
        "flagged_adversarial": True,
        "controls_differentiated": False,
        "s3_guided_beats_control": False,
        "verifier_is_oracle": False,
        "scorer_leak_recheck_passed": True,
        "benchmark_n": 240,
        "s3_minus_best_of_k_delta": 0.266667,
        "s3_minus_self_reward_smc_delta": 0.266667,
        "s3_minus_unguided_delta": 0.266667,
        "control_noop_guard": {
            "bit_identical_selection_pairs": [
                ["unguided", "best_of_k"],
                ["unguided", "self_reward_smc"],
                ["best_of_k", "self_reward_smc"],
            ],
            "reason": "condition arms tied or did not change",
        },
        "corrigendum_pending": [
            {"kind": "TAUTOLOGY", "severity": "critical"},
            {"kind": "TAUTOLOGY", "severity": "critical"},
            {"kind": "TAUTOLOGY", "severity": "critical"},
        ],
        "benchmark_records_preview": [
            {
                "task_id": "choice_000",
                "unguided_option": "D",
                "best_of_k_option": "D",
                "self_reward_smc_option": "D",
                "s3_carnot_option": "D",
            }
        ],
    }
    payload.update(overrides)
    return payload


def _action_cost(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "success: learned_action_cost_reduces_actions_25_to_16",
        "action_efficiency_improves": True,
        "held_out_actions_baseline": 25,
        "held_out_actions_learned": 16,
        "positive_control_passed": True,
        "reproduction_gated": True,
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _stamp_fix(**overrides: object) -> dict:
    payload = {
        "honest_verdict": (
            "complete: registry_gaps_arc_reconciled_to_v402_truth_gap4_guard_passed_"
            "True_capstone_stamp_fix_verified_True"
        ),
        "capstone_stamp_fix_verified": True,
        "gap4_regression_guard_passed": True,
        "registries_reconciled": True,
        "capstone_stamp_fix": {
            "capstone_stamp_fix_verified": True,
            "circular_moat_overclaim_fired": False,
            "flags": [],
            "returncode": 0,
        },
    }
    payload.update(overrides)
    return payload


def make_repo(tmp_path: Path, *, duplicates: int = 1) -> Path:
    root = tmp_path
    root.mkdir(parents=True, exist_ok=True)
    (root / "research-complete.yaml").write_text(
        _research_complete_text(duplicates=duplicates), encoding="utf-8"
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "exclusion_manifest.yaml").write_text(_manifest_text(), encoding="utf-8")
    (root / "ops" / "arc_solve_registry.yaml").write_text(_registry_text(), encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(
        "milestone: 2026.06.403\n"
        "milestone_doc: openspec/change-proposals/research-roadmap-v403.md\n"
        "milestone_overview: RE-ATTEMPT the conversion with a fixed Prism harness\n",
        encoding="utf-8",
    )
    (root / "openspec" / "change-proposals").mkdir(parents=True, exist_ok=True)
    (root / "openspec" / "change-proposals" / "research-roadmap-v403.md").write_text(
        "# Research Roadmap v403\n\nFIXED, Prism-hardened harness + ARC deeper + compounds.\n",
        encoding="utf-8",
    )
    _write_json(root / "results" / "experiment_4357_capstone_v402.json", _capstone())
    _write_json(
        root / "results" / "experiment_4348_s3_stratified_verifier_guided_search.json",
        _s3(),
    )
    _write_json(
        root / "results" / "experiment_4353_learned_action_cost_heuristic_efficiency.json",
        _action_cost(),
    )
    _write_json(
        root / "results" / "experiment_4355_registry_gaps_hygiene_capstone_stamp_fix.json",
        _stamp_fix(),
    )
    return root


def run_happy(root: Path) -> dict:
    out = mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5)
    return json.loads(out.read_text(encoding="utf-8"))


def test_req_report_4358_spec_declares_contract() -> None:
    """REQ-REPORT-4358: OpenSpec declares the true `.402` scorecard contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    assert "REQ-REPORT-4358" in spec
    assert "SCENARIO-REPORT-4358" in spec
    assert "SCENARIO-REPORT-4358-BLOCKED-PRECONDITION" in spec
    assert "S3 conversion failed as a HARNESS" in spec
    assert "`s3_moat_utility=open`" in spec
    assert "`26` reproducible levels across `15` games" in spec
    assert "Exp 4342" in spec and "Exp 4314" in spec
    assert mod.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert mod.FIELD_PRINCIPLES["v402_close_state"] in spec
    assert mod.FIELD_PRINCIPLES["preconditions_checked"] in spec


def test_helpers_and_archive_record_editing(tmp_path: Path) -> None:
    """REQ-REPORT-4358: helper behavior is deterministic and YAML-safe."""

    assert mod.archive_record_count(_research_complete_text(duplicates=3)) == 3
    assert mod._record_id("- id: '2026.06.402'") == "2026.06.402"
    assert mod._record_id("  - id: nested") is None
    assert mod._yaml_quote("don't reopen") == "'don''t reopen'"
    assert mod._insert_before_tasks(["  title: no tasks"], "  finding: x") == [
        "  title: no tasks",
        "  finding: x",
    ]

    root = make_repo(tmp_path)
    close_state = mod.build_v402_close_state(mod.read_v402_sources(root))
    deduped, removed, action = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=3), close_state
    )
    assert (removed, action) == (2, "deduped")
    assert mod.archive_record_count(deduped) == 1
    assert "S3 conversion HARNESS-FAILED" in deduped
    assert "ARC 26 reproducible levels / 15 games" in deduped
    assert "action-cost heuristic WON 25->16" in deduped
    assert mod.yaml_parses(deduped)

    updated, removed2, action2 = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=1), close_state
    )
    assert (removed2, action2) == (0, "updated")
    unchanged, removed3, action3 = mod.dedupe_or_update_record(updated, close_state)
    assert (unchanged, removed3, action3) == (updated, 0, "unchanged")
    appended, removed4, action4 = mod.dedupe_or_update_record(
        "# history\nmilestones:\n- id: 2026.06.401\n  finding: prior\n", close_state
    )
    assert (removed4, action4) == (0, "appended")
    assert "activation_recorded: exp4358-archive-v402-activate-v403" in appended
    no_finding, _removed5, action5 = mod.dedupe_or_update_record(
        "milestones:\n- id: 2026.06.402\n  title: no finding\n  tasks:\n  - id: exp4357\n",
        close_state,
    )
    assert action5 == "updated"
    assert "  finding: '.402 close-state" in no_finding


def test_read_sources_and_build_v402_close_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4358: close-state records the true `.402` scorecard."""

    root = make_repo(tmp_path)
    sources = mod.read_v402_sources(root)
    cited = mod.build_cited_upstream(root)
    assert sources["4357"]["verifier_thesis_state"] == (
        "moat_proven_leak_robust_but_s3_utility_open"
    )
    assert {item["experiment_id"] for item in cited if item["kind"] == "artifact"} == {
        "4357",
        "4348",
        "4353",
        "4355",
    }
    assert all(item["sha256"] is None or mod.is_sha256(item["sha256"]) for item in cited)

    state = mod.build_v402_close_state(sources)
    assert state["summary"] == ("s3_harness_failed_moat_utility_open_arc26_games15_action_cost_won")
    assert state["verifier_thesis_state"] == "moat_proven_leak_robust_but_s3_utility_open"
    assert state["s3_conversion_axis_state"] == "HARNESS_FAILED_CONTROLS_NOT_DIFFERENTIABLE"
    assert state["s3_harness_failed"] is True
    assert state["s3_moat_utility"] == "open"
    assert state["moat_still_proven_leak_robust"] is True
    assert state["in_generation_moat_utility_untested"] is True
    assert state["s3_framing_bug"] == "multiple_choice_selection_argmax_logit_control_collapse"
    assert state["s3_controls_differentiated"] is False
    assert state["critical_tautology_flags"] == 3
    assert state["controls_not_differentiable"] is True
    assert state["arc_capstone_snapshot_reproducible_total_levels"] == 23
    assert state["arc_reproducible_total_levels"] == 26
    assert state["arc_reproducible_total_games"] == 15
    assert state["arc_registry_observed_total_levels"] == 26
    assert state["arc_registry_observed_total_games"] == 15
    assert state["ka59_new_game"] is True
    assert state["tn36_l7_reproduced"] is True
    assert state["open_arc_gaps"]["sc25_l2"] == "spell_delta_gap"
    assert state["open_arc_gaps"]["ar25_l2"] == "action7_undo_stack_gap"
    assert state["action_cost_heuristic_axis_state"] == "WON_ACTION_COST_HEURISTIC"
    assert state["held_out_actions_baseline"] == 25
    assert state["held_out_actions_learned"] == 16
    assert state["action_reduction"] == 9
    assert state["action_cost_positive_control_passed"] is True
    assert state["action_cost_verifier_is_oracle"] is False
    assert state["cross_game_value_transfer_axis_state"] == "RETIRED_EXP4342_THIRD_NULL"
    assert state["cross_game_value_transfer_manifest_reflected"] is True
    assert state["cross_domain_axis_state"] == "RETIRED_EXP4314_DOMAIN_BOUND"
    assert state["cross_domain_manifest_reflected"] is True
    assert state["capstone_stamp_fix_durable"] is True
    assert state["capstone_stamp_fix_flagged_count"] == 0
    assert state["paper_ready"] is True
    assert state["v403_frame"] == mod.V403_FRAME


def test_run_happy_path_writes_valid_artifact_and_updates_archive(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4358: complete path writes the terminal archive artifact."""

    root = make_repo(tmp_path, duplicates=2)
    artifact = run_happy(root)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["archived_milestone"] == "2026.06.402"
    assert artifact["activated_milestone"] == "2026.06.403"
    assert artifact["active_milestone_confirmed"] == "2026.06.403"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["research_complete_record_action"] == "deduped"
    assert artifact["research_complete_duplicates_removed"] == 1
    assert artifact["v402_close_state"]["s3_harness_failed"] is True
    assert artifact["v402_close_state"]["s3_moat_utility"] == "open"
    assert artifact["v402_close_state"]["arc_reproducible_total_levels"] == 26
    assert artifact["v402_close_state"]["arc_reproducible_total_games"] == 15
    assert artifact["v402_close_state"]["action_cost_heuristic_won"] is True
    assert (
        artifact["field_principles"]["v402_close_state"] == mod.FIELD_PRINCIPLES["v402_close_state"]
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.is_sha256(artifact["reproducibility_checksum"])
    complete_text = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(complete_text) == 1
    assert "S3 conversion HARNESS-FAILED" in complete_text
    assert "fixed Prism-hardened harness" in complete_text
    mod.validate_artifact(artifact)


def test_run_blocked_preconditions_before_research_complete_edit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4358-BLOCKED-PRECONDITION: blocked paths do not fabricate."""

    missing = mod.run(tmp_path, pretest_result=GREEN)
    assert json.loads(missing.read_text(encoding="utf-8"))["honest_verdict"] == (
        "blocked_research_complete_yaml_missing"
    )

    root = make_repo(tmp_path / "poison")
    (root / "research-complete.yaml").write_text("a: : :\n- [\n", encoding="utf-8")
    artifact = json.loads(mod.run(root, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"] == "blocked_research_complete_yaml_poison"

    root2 = make_repo(tmp_path / "manifest_missing")
    (root2 / "ops" / "exclusion_manifest.yaml").unlink()
    artifact2 = json.loads(mod.run(root2, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact2["honest_verdict"] == "blocked_exclusion_manifest_missing"

    root_manifest = make_repo(tmp_path / "manifest_poison")
    (root_manifest / "ops" / "exclusion_manifest.yaml").write_text(
        "a: : :\n- [\n", encoding="utf-8"
    )
    artifact_manifest = json.loads(
        mod.run(root_manifest, pretest_result=GREEN).read_text(encoding="utf-8")
    )
    assert artifact_manifest["honest_verdict"] == "blocked_exclusion_manifest_yaml_poison"

    root3 = make_repo(tmp_path / "red")
    before = (root3 / "research-complete.yaml").read_text(encoding="utf-8")
    artifact3 = json.loads(mod.run(root3, pretest_result=RED).read_text(encoding="utf-8"))
    assert artifact3["honest_verdict"] == "blocked_smart_subset_pretest_not_green"
    assert artifact3["preconditions_checked"]["smart_subset_pretest"]["green"] is False
    assert (root3 / "research-complete.yaml").read_text(encoding="utf-8") == before

    root4 = make_repo(tmp_path / "wrong_milestone")
    (root4 / "research-roadmap.yaml").write_text("milestone: 2026.06.402\n", encoding="utf-8")
    artifact4 = json.loads(mod.run(root4, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact4["honest_verdict"] == "blocked_v403_not_active"

    root5 = make_repo(tmp_path / "source_missing")
    (root5 / "results" / "experiment_4357_capstone_v402.json").unlink()
    artifact5 = json.loads(mod.run(root5, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact5["honest_verdict"] == "blocked_v402_capstone_missing"


def test_run_blocked_research_complete_edit_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4358: invalid archive edits are blocked before completion."""

    root = make_repo(tmp_path / "invalid")
    monkeypatch.setattr(
        mod, "dedupe_or_update_record", lambda text, state: ("a: : :\n- [", 0, "appended")
    )
    artifact = json.loads(mod.run(root, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"] == "blocked_research_complete_edit_invalid"

    root2 = make_repo(tmp_path / "after")
    calls = {"n": 0}

    def fake_parses(text: str) -> bool:
        calls["n"] += 1
        return calls["n"] != 4

    monkeypatch.setattr(mod, "yaml_parses", fake_parses)
    artifact2 = json.loads(mod.run(root2, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact2["honest_verdict"] == "blocked_research_complete_yaml_poison_after_edit"


def test_build_artifact_validation_and_entrypoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4358: schema validation and entrypoints preserve the contract."""

    root = make_repo(tmp_path)
    state = mod.build_v402_close_state(mod.read_v402_sources(root))
    complete = mod.build_complete_artifact(
        v402_close_state=state,
        preconditions_checked={"ok": True},
        duration_s=0.5,
        active_roadmap_path="research-roadmap.yaml",
        research_complete_record_action="updated",
        research_complete_duplicates_removed=0,
        cited_upstream_artifacts=mod.build_cited_upstream(root),
    )
    assert complete["honest_verdict"].startswith("success:")
    blocked = mod.build_blocked_artifact(
        "blocked_x",
        preconditions_checked={"ok": False},
        duration_s=0.1,
        active_milestone_confirmed="",
        active_roadmap_path="research-roadmap.yaml",
    )
    assert blocked["honest_verdict"] == "blocked_x"
    assert mod.is_sha256(blocked["reproducibility_checksum"])
    assert mod.terminal_verdict(state).startswith("success:")

    called_mod: dict[str, Path] = {}

    def fake_mod_run(root_path: Path = mod.REPO_ROOT) -> Path:
        called_mod["root"] = Path(root_path)
        return root / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(mod, "run", fake_mod_run)
    assert mod.main() == 0
    assert called_mod["root"] == mod.REPO_ROOT

    import carnot.experiment_4358_archive_v402_activate_v403 as entrypoint

    called: dict[str, Path] = {}

    def fake_run(root_path: Path = mod.REPO_ROOT) -> Path:
        called["root"] = Path(root_path)
        return root / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(entrypoint, "run", fake_run)
    assert entrypoint.main() == 0
    assert called["root"] == entrypoint.REPO_ROOT

    script_path = Path("results/experiment_4358_archive_v402_activate_v403.py")
    script_repo_root = str(script_path.resolve().parents[1])
    script_python_root = str(Path(script_repo_root) / "python")
    original_sys_path = list(sys.path)
    try:
        sys.path[:] = [
            item for item in sys.path if item not in {script_repo_root, script_python_root}
        ]
        spec = importlib.util.spec_from_file_location("exp4358_archive_script", script_path)
        assert spec and spec.loader
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
    finally:
        sys.path[:] = original_sys_path
    monkeypatch.setattr(script, "run", fake_run)
    assert script.main() == 0


def test_validate_artifact_rejection_guards(tmp_path: Path) -> None:
    """REQ-REPORT-4358: validation rejects artifacts that launder the `.402` truth."""

    good = run_happy(make_repo(tmp_path))

    def set_path(obj: dict, path: list[str], value: object) -> None:
        cur = obj
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = value

    cases = [
        ("missing required fields", lambda a: a.pop("v402_close_state")),
        ("terminal-prefixed", lambda a: a.__setitem__("honest_verdict", "done")),
        (
            "principle must match REQ-REPORT-4358",
            lambda a: a["field_principles"].__setitem__("v402_close_state", "wrong"),
        ),
        ("archived milestone", lambda a: a.__setitem__("archived_milestone", "2026.06.999")),
        ("activated milestone", lambda a: a.__setitem__("activated_milestone", "2026.06.999")),
        ("research-complete YAML", lambda a: a.__setitem__("research_complete_yaml_parses", False)),
        ("exclusion manifest", lambda a: a.__setitem__("exclusion_manifest_parses", False)),
        ("pretest suite", lambda a: a.__setitem__("pretest_suite_green", False)),
        ("active milestone", lambda a: a.__setitem__("active_milestone_confirmed", "2026.06.402")),
        ("v402_close_state must be a mapping", lambda a: a.__setitem__("v402_close_state", "x")),
        ("S3 harness", lambda a: set_path(a, ["v402_close_state", "s3_harness_failed"], False)),
        ("S3 utility", lambda a: set_path(a, ["v402_close_state", "s3_moat_utility"], "useful")),
        (
            "moat proven",
            lambda a: set_path(a, ["v402_close_state", "moat_still_proven_leak_robust"], False),
        ),
        (
            "ARC 26",
            lambda a: set_path(a, ["v402_close_state", "arc_reproducible_total_levels"], 23),
        ),
        (
            "ARC games",
            lambda a: set_path(a, ["v402_close_state", "arc_reproducible_total_games"], 14),
        ),
        (
            "action-cost win",
            lambda a: set_path(a, ["v402_close_state", "action_cost_heuristic_won"], False),
        ),
        (
            "cross-game retired",
            lambda a: set_path(
                a, ["v402_close_state", "cross_game_value_transfer_axis_state"], "OPEN"
            ),
        ),
        (
            "cross-domain retired",
            lambda a: set_path(a, ["v402_close_state", "cross_domain_axis_state"], "OPEN"),
        ),
        (
            "stamp fix",
            lambda a: set_path(a, ["v402_close_state", "capstone_stamp_fix_durable"], False),
        ),
        ("paper", lambda a: set_path(a, ["v402_close_state", "paper_ready"], False)),
        ("v403 frame", lambda a: set_path(a, ["v402_close_state", "v403_frame"], "redo")),
    ]
    for label, mutate in cases:
        artifact = copy.deepcopy(good)
        mutate(artifact)
        with pytest.raises(ValueError, match=label):
            mod.validate_artifact(artifact)
