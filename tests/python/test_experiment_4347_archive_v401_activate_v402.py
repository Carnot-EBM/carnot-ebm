"""Tests for Exp 4347 `.401` archive / `.402` activation.

Spec refs: REQ-REPORT-4347, SCENARIO-REPORT-4347,
SCENARIO-REPORT-4347-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v401_activate_v402_4347 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="81 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.400\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.401\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-17'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4346-capstone-v401\n"
        "    result: OK (conductor)\n"
    )
    return head + block * duplicates


def _capstone(**overrides: object) -> dict:
    payload = {
        "honest_verdict": (
            "complete: v401_in_generation_replicated_leak_robust_gate_MET_"
            "oracle_distinct_leak_robust_replicated_arc_levels_17_e3_reproduced_2"
        ),
        "headline_outcome": (
            "gate_MET_oracle_distinct_leak_robust_replicated__arc_levels_17_"
            "e3_2__self_learning_open__paper_ready"
        ),
        "verifier_thesis_state": "in_generation_moat_replicated_leak_robust",
        "diffusiongemma_gate_status": "MET_oracle_distinct_leak_robust_replicated",
        "in_generation_moat_replicates_headline": True,
        "arc_reproducible_total_levels": 17,
        "paper_ready": True,
        "verifier_is_oracle_honored": True,
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "CIRCULAR_MOAT_OVERCLAIM"}],
        "in_generation_moat": {
            "status": "replicated",
            "honest_verdict": "complete: in_generation_moat_replicates",
            "in_generation_moat_replicates_headline": True,
            "scorer_leak_recheck_passed": True,
            "controls_differentiated": True,
            "replication_ci95": [0.283333, 0.4375],
            "replication_ci95_excludes_zero": True,
            "carnot_minus_best_control_delta": 0.358333,
            "carnot_minus_self_reward_smc_delta": 0.320833,
            "benchmark_n": 240,
            "verifier_is_oracle": False,
        },
        "scorer_leak_audit": {
            "status": "passed",
            "honest_verdict": "complete: leak_robust_partial_state_scorer_built",
            "scorer_leak_audit_passed": True,
            "masked_answer_recovery_auroc": 0.559682,
            "process_ranking_auroc": 0.704633,
            "verifier_is_oracle": False,
        },
        "e3_arc_progress": {
            "status": "reproduced",
            "execution_grounded": True,
            "reproduced_levels_total": 2,
            "games": {
                "ar25": {
                    "game": "ar25",
                    "status": "reproduced",
                    "honest_verdict": "success_e3_ar25_L1_reproduced",
                    "offline_reproduced": True,
                    "plan_executed": True,
                    "reproduced_levels": 1,
                    "verifier_best_accuracy": 0.8875,
                    "verifier_is_oracle": True,
                },
                "ka59": {
                    "game": "ka59",
                    "status": "partial",
                    "honest_verdict": "complete_e3_ka59_partial_model_0.56",
                    "offline_reproduced": False,
                    "plan_executed": False,
                    "reproduced_levels": 0,
                    "verifier_best_accuracy": 0.5625,
                    "verifier_is_oracle": True,
                },
                "sc25": {
                    "game": "sc25",
                    "status": "reproduced",
                    "honest_verdict": "success_e3_sc25_L1_reproduced",
                    "offline_reproduced": True,
                    "plan_executed": True,
                    "reproduced_levels": 1,
                    "verifier_best_accuracy": 1.0,
                    "verifier_is_oracle": True,
                },
            },
        },
        "self_learning": {
            "status": "open",
            "honest_verdict": "complete: action_role_encoder_transfer_no_improvement",
            "learned_encoder_transfer_helps": False,
            "cross_game_state_reduction": 1.00635593220339,
            "cross_game_state_reduction_ci95": [1.0, 1.0168354897287482],
            "positive_control_passed": True,
            "verifier_is_oracle": False,
        },
    }
    payload.update(overrides)
    return payload


def _in_generation(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: in_generation_moat_replicates",
        "in_generation_moat_replicates": True,
        "scorer_leak_recheck_passed": True,
        "controls_differentiated": True,
        "replication_ci95": [0.283333, 0.4375],
        "carnot_minus_best_control_delta": 0.358333,
        "carnot_minus_self_reward_smc_delta": 0.320833,
        "benchmark_n": 240,
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _scorer(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: leak_robust_partial_state_scorer_built",
        "scorer_leak_audit_passed": True,
        "masked_answer_recovery_auroc": 0.559682,
        "process_ranking_auroc": 0.704633,
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _e3(game: str, *, reproduced: bool, levels: int, accuracy: float) -> dict:
    status = "reproduced" if reproduced else "partial"
    return {
        "game": game,
        "honest_verdict": f"success_e3_{game}_L1_reproduced" if reproduced else f"complete_e3_{game}_partial",
        "offline_reproduced": reproduced,
        "plan_executed": reproduced,
        "reproduced_levels": levels,
        "status": status,
        "verifier_best_accuracy": accuracy,
        "verifier_is_oracle": True,
    }


def _tr87_ft09() -> dict:
    return {
        "honest_verdict": "complete_e3_tr87_ft09_partial",
        "reproduced_levels_total": 0,
        "verifier_is_oracle": True,
        "per_game_scorecard": {
            "tr87": {
                "game": "tr87",
                "status": "complete_e3_tr87_partial_model_0.00",
                "offline_reproduced": False,
                "reproduced_levels": 0,
                "best_verifier_accuracy": 0.0,
            },
            "ft09": {
                "game": "ft09",
                "status": "complete_e3_ft09_partial_model_0.10",
                "offline_reproduced": False,
                "reproduced_levels": 0,
                "best_verifier_accuracy": 0.1,
            },
        },
    }


def _self_learning(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: action_role_encoder_transfer_no_improvement_positive_control_passed",
        "learned_encoder_transfer_helps": False,
        "cross_game_state_reduction": 1.00635593220339,
        "cross_game_state_reduction_ci95": [1.0, 1.0168354897287482],
        "positive_control_passed": True,
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _arc_registry_text() -> str:
    return (
        "schema_version: 1\n"
        "updated: '2026-06-17'\n"
        "reproducible_total_levels: 21\n"
        "reproducible_total_games: 13\n"
        "provisional_total_levels: 5\n"
    )


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


def make_repo(tmp_path: Path, *, duplicates: int = 1) -> Path:
    root = tmp_path
    root.mkdir(parents=True, exist_ok=True)
    (root / "research-complete.yaml").write_text(
        _research_complete_text(duplicates=duplicates), encoding="utf-8"
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "exclusion_manifest.yaml").write_text(_manifest_text(), encoding="utf-8")
    (root / "ops" / "arc_solve_registry.yaml").write_text(_arc_registry_text(), encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(
        "milestone: 2026.06.402\n"
        "milestone_doc: openspec/change-proposals/research-roadmap-v402.md\n"
        "milestone_overview: CONVERT the proven moat into a generation gain\n",
        encoding="utf-8",
    )
    (root / "openspec" / "change-proposals").mkdir(parents=True, exist_ok=True)
    (root / "openspec" / "change-proposals" / "research-roadmap-v402.md").write_text(
        "# Research Roadmap v402\n\nS3 + E3 deeper + action-cost self-learning.\n",
        encoding="utf-8",
    )
    _write_json(root / "results" / "experiment_4346_capstone_v401.json", _capstone())
    _write_json(
        root / "results" / "experiment_4338_in_generation_moat_replicate_leak_robust.json",
        _in_generation(),
    )
    _write_json(
        root / "results" / "experiment_4337_leak_robust_partial_state_scorer_build.json",
        _scorer(),
    )
    _write_json(
        root / "results" / "experiment_4339_e3_explore_verify_plan_ar25.json",
        _e3("ar25", reproduced=True, levels=1, accuracy=0.8875),
    )
    _write_json(
        root / "results" / "experiment_4340_e3_explore_verify_plan_ka59.json",
        _e3("ka59", reproduced=False, levels=0, accuracy=0.5625),
    )
    _write_json(
        root / "results" / "experiment_4341_e3_sc25_reproduction.json",
        _e3("sc25", reproduced=True, levels=1, accuracy=1.0),
    )
    _write_json(
        root / "results" / "experiment_4329_e3_executable_world_model_tr87_ft09.json",
        _tr87_ft09(),
    )
    _write_json(
        root / "results" / "experiment_4342_self_learning_action_role_cross_game_encoder.json",
        _self_learning(),
    )
    _write_json(
        root / "results" / "experiment_4314_cross_domain_selector_ir3de_cascal.json",
        {"honest_verdict": "complete: powered_collapse_cross_domain_domain_bound", "verifier_is_oracle": False},
    )
    return root


def run_happy(root: Path) -> dict:
    out = mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5)
    return json.loads(out.read_text(encoding="utf-8"))


def test_req_report_4347_spec_declares_contract() -> None:
    """REQ-REPORT-4347: OpenSpec declares the true `.401` scorecard contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    assert "REQ-REPORT-4347" in spec
    assert "SCENARIO-REPORT-4347" in spec
    assert "SCENARIO-REPORT-4347-BLOCKED-PRECONDITION" in spec
    assert "REPLICATED LEAK-ROBUST" in spec
    assert "21 reproducible levels across 13 games" in spec
    assert "Exp 4342's third powered null" in spec
    assert mod.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert mod.FIELD_PRINCIPLES["v401_close_state"] in spec
    assert mod.FIELD_PRINCIPLES["preconditions_checked"] in spec


def test_helpers_and_archive_record_editing(tmp_path: Path) -> None:
    """REQ-REPORT-4347: helper behavior is deterministic and YAML-safe."""

    assert mod.archive_record_count(_research_complete_text(duplicates=3)) == 3
    assert mod._record_id("- id: '2026.06.401'") == "2026.06.401"
    assert mod._record_id("  - id: nested") is None
    assert mod._yaml_quote("don't re-open") == "'don''t re-open'"
    assert mod._rounded_pair("bad", [1.0, 2.0]) == [1.0, 2.0]
    assert mod._ci_excludes_zero([0.1, 0.2])
    assert mod._insert_before_tasks(["  title: no tasks"], "  finding: x") == [
        "  title: no tasks",
        "  finding: x",
    ]

    root = make_repo(tmp_path)
    close_state = mod.build_v401_close_state(mod.read_v401_sources(root))
    deduped, removed, action = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=3), close_state
    )
    assert (removed, action) == (2, "deduped")
    assert mod.archive_record_count(deduped) == 1
    assert "moat REPLICATED LEAK-ROBUST" in deduped
    assert "ARC 21 reproducible levels / 13 games" in deduped
    assert "cross-game value transfer RETIRED" in deduped
    assert mod.yaml_parses(deduped)

    updated, removed2, action2 = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=1), close_state
    )
    assert (removed2, action2) == (0, "updated")
    unchanged, removed3, action3 = mod.dedupe_or_update_record(updated, close_state)
    assert (unchanged, removed3, action3) == (updated, 0, "unchanged")
    appended, removed4, action4 = mod.dedupe_or_update_record(
        "# history\nmilestones:\n- id: 2026.06.400\n  finding: prior\n", close_state
    )
    assert (removed4, action4) == (0, "appended")
    assert "activation_recorded: exp4347-archive-v401-activate-v402" in appended
    no_finding, _removed5, action5 = mod.dedupe_or_update_record(
        "milestones:\n- id: 2026.06.401\n  title: no finding\n  tasks:\n  - id: exp4346\n",
        close_state,
    )
    assert action5 == "updated"
    assert "  finding: '.401 close-state" in no_finding


def test_read_sources_and_build_v401_close_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4347: close-state records the true .401 scorecard."""

    root = make_repo(tmp_path)
    sources = mod.read_v401_sources(root)
    cited = mod.build_cited_upstream(root)
    assert sources["4346"]["verifier_thesis_state"] == "in_generation_moat_replicated_leak_robust"
    assert {item["experiment_id"] for item in cited if item["kind"] == "artifact"} == {
        "4346",
        "4338",
        "4337",
        "4339",
        "4340",
        "4341",
        "4329",
        "4342",
        "4314",
    }
    assert any(item["experiment_id"] == "v402_design_doc" for item in cited)
    assert all(item["sha256"] is None or mod.is_sha256(item["sha256"]) for item in cited)

    state = mod.build_v401_close_state(sources)
    assert state["summary"] == (
        "moat_replicated_leak_robust_gate_met_e3_ar25_sc25_l1_"
        "arc21_cross_game_transfer_retired"
    )
    assert state["verifier_thesis_state"] == "in_generation_moat_replicated_leak_robust"
    assert state["in_generation_axis_state"] == "REPLICATED_LEAK_ROBUST_ORACLE_DISTINCT"
    assert state["diffusiongemma_gate_status"] == "MET_oracle_distinct_leak_robust_replicated"
    assert state["in_generation_moat_replicates"] is True
    assert state["in_generation_verifier_is_oracle"] is False
    assert state["in_generation_delta_vs_best_control"] == 0.358
    assert state["in_generation_delta_vs_self_reward_smc"] == 0.321
    assert state["in_generation_replication_ci95"] == [0.283, 0.438]
    assert state["in_generation_replication_ci95_excludes_zero"] is True
    assert state["in_generation_benchmark_n"] == 240
    assert state["scorer_leak_audit_passed"] is True
    assert state["masked_answer_recovery_auroc"] == 0.56
    assert state["process_ranking_auroc"] == 0.705
    assert state["first_e3_solved_games"] == ["ar25", "sc25"]
    assert state["e3_reproduced_levels_total"] == 2
    assert state["e3_solve_levels"] == {"ar25": 1, "sc25": 1}
    assert state["e3_partial_games"] == ["ka59", "tr87", "ft09"]
    assert state["e3_partial_best_accuracy"] == {"ka59": 0.562, "tr87": 0.0, "ft09": 0.1}
    assert state["arc_capstone_stale_reproducible_total_levels"] == 17
    assert state["arc_reproducible_total_levels"] == 21
    assert state["arc_reproducible_total_games"] == 13
    assert state["cross_game_value_transfer_axis_state"] == "RETIRED_THIRD_POWERED_NULL"
    assert state["cross_game_value_transfer_manifest_reflected"] is True
    assert state["cross_domain_axis_state"] == "RETIRED_DOMAIN_BOUND"
    assert state["cross_domain_manifest_reflected"] is True
    assert state["paper_ready"] is True
    assert state["capstone_circular_moat_overclaim_is_stamping_bug"] is True
    assert state["v402_frame"] == mod.V402_FRAME


def test_run_happy_path_writes_valid_artifact_and_updates_archive(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4347: complete path writes the terminal archive artifact."""

    root = make_repo(tmp_path, duplicates=2)
    artifact = run_happy(root)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["archived_milestone"] == "2026.06.401"
    assert artifact["activated_milestone"] == "2026.06.402"
    assert artifact["active_milestone_confirmed"] == "2026.06.402"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["research_complete_record_action"] == "deduped"
    assert artifact["research_complete_duplicates_removed"] == 1
    assert artifact["v401_close_state"]["in_generation_axis_state"] == (
        "REPLICATED_LEAK_ROBUST_ORACLE_DISTINCT"
    )
    assert artifact["v401_close_state"]["arc_reproducible_total_levels"] == 21
    assert artifact["v401_close_state"]["arc_reproducible_total_games"] == 13
    assert artifact["field_principles"]["v401_close_state"] == mod.FIELD_PRINCIPLES[
        "v401_close_state"
    ]
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.is_sha256(artifact["reproducibility_checksum"])
    complete_text = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(complete_text) == 1
    assert "moat REPLICATED LEAK-ROBUST" in complete_text
    assert "S3 generation gain + E3 deeper + learned action-cost self-learning" in complete_text
    mod.validate_artifact(artifact)


def test_run_blocked_preconditions_before_research_complete_edit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4347-BLOCKED-PRECONDITION: blocked paths do not fabricate."""

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
    (root4 / "research-roadmap.yaml").write_text("milestone: 2026.06.401\n", encoding="utf-8")
    artifact4 = json.loads(mod.run(root4, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact4["honest_verdict"] == "blocked_v402_not_active"

    root5 = make_repo(tmp_path / "source_missing")
    (root5 / "results" / "experiment_4338_in_generation_moat_replicate_leak_robust.json").unlink()
    artifact5 = json.loads(mod.run(root5, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact5["honest_verdict"] == "blocked_in_generation_moat_missing"


def test_run_blocked_research_complete_edit_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4347: invalid archive edits are blocked before completion."""

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
    """REQ-REPORT-4347: schema validation and entrypoints preserve the contract."""

    root = make_repo(tmp_path)
    state = mod.build_v401_close_state(mod.read_v401_sources(root))
    complete = mod.build_complete_artifact(
        v401_close_state=state,
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

    import carnot.experiment_4347_archive_v401_activate_v402 as entrypoint

    called: dict[str, Path] = {}

    def fake_run(root_path: Path = mod.REPO_ROOT) -> Path:
        called["root"] = Path(root_path)
        return root / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(entrypoint, "run", fake_run)
    assert entrypoint.main() == 0
    assert called["root"] == entrypoint.REPO_ROOT

    script_path = Path("results/experiment_4347_archive_v401_activate_v402.py")
    spec = importlib.util.spec_from_file_location("exp4347_archive_script", script_path)
    assert spec and spec.loader
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "run", fake_run)
    assert script.main() == 0


def test_validate_artifact_rejection_guards(tmp_path: Path) -> None:
    """REQ-REPORT-4347: validation rejects artifacts that launder the `.401` truth."""

    good = run_happy(make_repo(tmp_path))

    def set_path(obj: dict, path: list[str], value: object) -> None:
        cur = obj
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = value

    cases = [
        ("missing required fields", lambda a: a.pop("v401_close_state")),
        ("terminal-prefixed", lambda a: a.__setitem__("honest_verdict", "done")),
        (
            "principle must match REQ-REPORT-4347",
            lambda a: a["field_principles"].__setitem__("v401_close_state", "wrong"),
        ),
        ("archived milestone", lambda a: a.__setitem__("archived_milestone", "2026.06.999")),
        ("activated milestone", lambda a: a.__setitem__("activated_milestone", "2026.06.999")),
        ("research-complete YAML", lambda a: a.__setitem__("research_complete_yaml_parses", False)),
        ("exclusion manifest", lambda a: a.__setitem__("exclusion_manifest_parses", False)),
        ("pretest suite", lambda a: a.__setitem__("pretest_suite_green", False)),
        ("active milestone", lambda a: a.__setitem__("active_milestone_confirmed", "2026.06.401")),
        ("v401_close_state must be a mapping", lambda a: a.__setitem__("v401_close_state", "x")),
        (
            "moat replicated",
            lambda a: set_path(a, ["v401_close_state", "in_generation_moat_replicates"], False),
        ),
        (
            "gate met",
            lambda a: set_path(a, ["v401_close_state", "diffusiongemma_gate_status"], "PENDING"),
        ),
        (
            "oracle distinct",
            lambda a: set_path(a, ["v401_close_state", "in_generation_verifier_is_oracle"], True),
        ),
        (
            "E3 solved games",
            lambda a: set_path(a, ["v401_close_state", "first_e3_solved_games"], ["ar25"]),
        ),
        (
            "ARC 21",
            lambda a: set_path(a, ["v401_close_state", "arc_reproducible_total_levels"], 17),
        ),
        (
            "cross-game retired",
            lambda a: set_path(a, ["v401_close_state", "cross_game_value_transfer_axis_state"], "OPEN"),
        ),
        (
            "cross-domain retired",
            lambda a: set_path(a, ["v401_close_state", "cross_domain_axis_state"], "OPEN"),
        ),
        ("paper", lambda a: set_path(a, ["v401_close_state", "paper_ready"], False)),
        ("v402 frame", lambda a: set_path(a, ["v401_close_state", "v402_frame"], "redo")),
    ]
    for label, mutate in cases:
        artifact = copy.deepcopy(good)
        mutate(artifact)
        with pytest.raises(ValueError, match=label):
            mod.validate_artifact(artifact)
