"""Tests for Exp 4324 `.399` archive / `.400` activation.

Spec refs: REQ-REPORT-4324, SCENARIO-REPORT-4324,
SCENARIO-REPORT-4324-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v399_activate_v400_4324 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="81 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.398\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.399\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-17'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4323-capstone-v399\n"
        "    result: OK (conductor)\n"
    )
    return head + block * duplicates


def _capstone(**overrides: object) -> dict:
    payload = {
        "honest_verdict": (
            "complete: v399_cross_domain_open_in_generation_moat_efficiency_open_"
            "arc_levels_23_self_learning_open_off_arc_execution_grounded_win"
        ),
        "headline_outcome": (
            "cross_domain_open__in_generation_moat__efficiency_open__arc_levels_23__"
            "self_learning_open__off_arc_execution_grounded_win__paper_ready"
        ),
        "verifier_thesis_state": "in_generation_moat_holds",
        "cross_domain_moat_holds": False,
        "in_generation_moat_holds": True,
        "efficiency_cascade_dominates": False,
        "paper_ready": True,
        "verifier_is_oracle_honored": True,
        "cross_domain": {
            "status": "open",
            "cross_domain_moat_holds": False,
            "reported_cross_domain_selection_holds": False,
            "cross_domain_delta": 0.2307692308,
            "cross_domain_delta_ci95": [-0.1153846154, 0.5384615385],
            "label_ablation_robust": True,
            "verifier_is_oracle": False,
        },
        "in_generation": {
            "status": "moat_holds",
            "in_generation_moat_holds": True,
            "reported_diffusiongemma_guidance_moat": True,
            "carnot_minus_best_control_delta": 0.225,
            "carnot_minus_self_reward_smc_delta": 0.35,
            "guidance_moat_ci95": [0.075, 0.375],
            "controls_differentiated": True,
            "verifier_is_oracle": False,
        },
        "efficiency": {
            "status": "open",
            "honest_verdict": (
                "complete: always_energy_already_dominates_acc_cascade_0.5500_"
                "cost_ratio_0.3019632358"
            ),
            "efficiency_cascade_dominates": False,
            "accuracy_always_energy": 0.6,
            "accuracy_always_judge": 0.25,
            "accuracy_cascade": 0.55,
            "cost_ratio_cascade": 0.3019632358,
            "verifier_is_oracle": False,
        },
        "arc": {
            "status": "included",
            "honest_verdict": "success: adapter_free_incremental_progress_cd82-fb555c5d_advanced_to_L1_total23",
            "total_levels_solved": 23,
            "levels_completed": 1,
            "offline_reproduced": True,
        },
        "self_learning": {
            "status": "open",
            "honest_verdict": "complete: arc_cross_game_value_head_no_improvement_positive_control_passed",
            "cross_game_transfer_helps": False,
            "cross_game_state_reduction": 1.0,
            "cross_game_state_reduction_ci95": [1.0, 1.0],
            "verifier_is_oracle": False,
        },
        "off_arc": {
            "status": "execution_grounded_win",
            "honest_verdict": "success: off_arc_demofit_beats_vote_accumulated_ci_excludes_zero",
            "off_arc_demofit_beats_vote": True,
            "off_arc_demofit_minus_vote_delta": 0.02,
            "off_arc_delta_ci95": [0.005, 0.04],
            "accumulated_n": 200,
            "verifier_is_oracle": True,
        },
    }
    payload.update(overrides)
    return payload


def _in_generation(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: diffusiongemma_step_stitching_moat_won",
        "diffusiongemma_guidance_moat": True,
        "carnot_minus_best_control_delta": 0.225,
        "carnot_minus_self_reward_smc_delta": 0.35,
        "guidance_moat_ci95": [0.075, 0.375],
        "controls_differentiated": True,
        "scorer_leak_recheck_passed": True,
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _cross_domain(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: powered_collapse_cross_domain_domain_bound",
        "cross_domain_selection_holds": False,
        "cross_domain_delta": 0.2307692308,
        "cross_domain_delta_ci95": [-0.1153846154, 0.5384615385],
        "label_ablation_robust": True,
        "held_out_task_n": 26,
        "verifier_is_oracle": False,
        "missing_verifier_gaps": [
            {
                "gap_id": "GAP-CROSS-DOMAIN-FAMILY-INVARIANT-SELECTION-4305",
                "failure_mode": "powered_collapse_cross_domain_domain_bound",
            }
        ],
    }
    payload.update(overrides)
    return payload


def _arc_registry_text() -> str:
    return (
        "reproducible_total_levels: 13\n"
        "reproducible_total_games: 11\n"
        "live_submissions:\n"
        "  - date: '2026-06-17'\n"
        "    scorecard_id: 0f6273ce-cf0d-426c-83e5-d745e4d45ea2\n"
        "    levels: 13\n"
        "    games: 11\n"
        "    games_env_matched: 11/11\n"
        "    mode: mode1_offline_reproduced_replay\n"
        "    note: FIRST live submission.\n"
        "prior_submitted_baseline_levels: 13\n"
    )


def make_repo(tmp_path: Path, *, duplicates: int = 1) -> Path:
    root = tmp_path
    root.mkdir(parents=True, exist_ok=True)
    (root / "research-complete.yaml").write_text(
        _research_complete_text(duplicates=duplicates), encoding="utf-8"
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "exclusion_manifest.yaml").write_text(
        "retired:\n- id: older_axis\n  retire_if_same_verdict: true\n", encoding="utf-8"
    )
    (root / "ops" / "arc_solve_registry.yaml").write_text(
        _arc_registry_text(), encoding="utf-8"
    )
    (root / "research-roadmap.yaml").write_text(
        "milestone: 2026.06.400\n"
        "milestone_doc: openspec/change-proposals/research-roadmap-v400.md\n"
        "milestone_overview: scale-the-in-generation-moat + E3-deep-tail-ARC\n",
        encoding="utf-8",
    )
    (root / "openspec" / "change-proposals").mkdir(parents=True, exist_ok=True)
    (root / "openspec" / "change-proposals" / "research-roadmap-v400.md").write_text(
        "# Research Roadmap v400\n\nCross-domain selection scope is RETIRED.\n",
        encoding="utf-8",
    )
    _write_json(root / "results" / "experiment_4323_capstone_v399.json", _capstone())
    _write_json(
        root / "results" / "experiment_4315_diffusiongemma_reward_guided_stitching.json",
        _in_generation(),
    )
    _write_json(
        root / "results" / "experiment_4314_cross_domain_selector_ir3de_cascal.json",
        _cross_domain(),
    )
    return root


def run_happy(root: Path) -> dict:
    out = mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5)
    return json.loads(out.read_text(encoding="utf-8"))


def test_req_report_4324_spec_declares_contract() -> None:
    """REQ-REPORT-4324: OpenSpec declares the true `.399` scorecard contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    assert "REQ-REPORT-4324" in spec
    assert "SCENARIO-REPORT-4324" in spec
    assert "SCENARIO-REPORT-4324-BLOCKED-PRECONDITION" in spec
    assert "in-generation moat CLOSED oracle-distinctly" in spec
    assert "cross-domain selection moat RETIRED as" in spec
    assert "always-energy-dominates" in spec
    assert "first live submission scorecard `0f6273ce-cf0d-426c-83e5-d745e4d45ea2`" in spec
    assert mod.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert mod.FIELD_PRINCIPLES["v399_close_state"] in spec
    assert mod.FIELD_PRINCIPLES["preconditions_checked"] in spec


def test_helpers_and_archive_record_editing(tmp_path: Path) -> None:
    """REQ-REPORT-4324: helper behavior is deterministic and YAML-safe."""

    assert mod.archive_record_count(_research_complete_text(duplicates=3)) == 3
    assert mod._record_id("- id: '2026.06.399'") == "2026.06.399"
    assert mod._record_id("  - id: nested") is None
    assert mod._yaml_quote("don't re-open") == "'don''t re-open'"
    assert mod._rounded_pair("bad", [1.0, 2.0]) == [1.0, 2.0]
    assert mod._ci_includes_zero([-0.1, 0.2])
    assert mod._ci_excludes_zero([0.1, 0.2])
    assert mod._insert_before_tasks(["  title: no tasks"], "  finding: x") == [
        "  title: no tasks",
        "  finding: x",
    ]

    root = make_repo(tmp_path)
    close_state = mod.build_v399_close_state(mod.read_v399_sources(root))
    deduped, removed, action = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=3), close_state
    )
    assert (removed, action) == (2, "deduped")
    assert mod.archive_record_count(deduped) == 1
    assert "in-generation moat CLOSED oracle-distinct" in deduped
    assert "cross-domain RETIRED domain-bound" in deduped
    assert "do NOT re-propose" in deduped
    assert mod.yaml_parses(deduped)

    updated, removed2, action2 = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=1), close_state
    )
    assert (removed2, action2) == (0, "updated")
    unchanged, removed3, action3 = mod.dedupe_or_update_record(updated, close_state)
    assert (unchanged, removed3, action3) == (updated, 0, "unchanged")
    appended, removed4, action4 = mod.dedupe_or_update_record(
        "# history\nmilestones:\n- id: 2026.06.398\n  finding: prior\n", close_state
    )
    assert (removed4, action4) == (0, "appended")
    assert "activation_recorded: exp4324-archive-v399-activate-v400" in appended
    no_finding, _removed5, action5 = mod.dedupe_or_update_record(
        "milestones:\n- id: 2026.06.399\n  title: no finding\n  tasks:\n  - id: exp4323\n",
        close_state,
    )
    assert action5 == "updated"
    assert "  finding: '.399 close-state" in no_finding


def test_read_sources_and_build_v399_close_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4324: close-state records the true .399 scorecard."""

    root = make_repo(tmp_path)
    sources = mod.read_v399_sources(root)
    cited = mod.build_cited_upstream(root)
    assert sources["4323"]["verifier_thesis_state"] == "in_generation_moat_holds"
    assert {item["experiment_id"] for item in cited if item["kind"] == "artifact"} == {
        "4323",
        "4315",
        "4314",
    }
    assert any(item["experiment_id"] == "arc_solve_registry" for item in cited)
    assert all(item["sha256"] is None or mod.is_sha256(item["sha256"]) for item in cited)

    state = mod.build_v399_close_state(sources)
    assert state["summary"] == (
        "in_generation_moat_closed_cross_domain_retired_efficiency_energy_dominates_"
        "arc_live13_self_learning_null_off_arc_marginal"
    )
    assert state["verifier_thesis_state"] == "in_generation_moat_holds"
    assert state["in_generation_axis_state"] == "CLOSED_ORACLE_DISTINCT"
    assert state["in_generation_moat_holds"] is True
    assert state["in_generation_delta_vs_best_control"] == 0.225
    assert state["in_generation_delta_vs_self_reward_smc"] == 0.35
    assert state["in_generation_ci95"] == [0.075, 0.375]
    assert state["in_generation_ci95_excludes_zero"] is True
    assert state["in_generation_controls_differentiated"] is True
    assert state["in_generation_scorer_leak_recheck_passed"] is True
    assert state["in_generation_verifier_is_oracle"] is False
    assert state["cross_domain_axis_state"] == "RETIRED_DOMAIN_BOUND"
    assert state["cross_domain_moat_holds"] is False
    assert state["cross_domain_delta"] == 0.231
    assert state["cross_domain_ci95_includes_zero"] is True
    assert state["cross_domain_label_ablation_robust"] is True
    assert state["cross_domain_retire_if_same_verdict"] is True
    assert state["cross_domain_do_not_repropose"] is True
    assert state["efficiency_axis_state"] == "ALWAYS_ENERGY_DOMINATES"
    assert state["efficiency_cascade_dominates"] is False
    assert state["efficiency_accuracy_always_energy"] == 0.6
    assert state["efficiency_accuracy_cascade"] == 0.55
    assert state["arc_reproducible_total_levels"] == 13
    assert state["arc_reproducible_total_games"] == 11
    assert state["arc_first_live_submission"] is True
    assert state["arc_live_submission_scorecard_id"] == "0f6273ce-cf0d-426c-83e5-d745e4d45ea2"
    assert state["arc_live_submission_games_env_matched"] == "11/11"
    assert state["self_learning_axis_state"] == "CROSS_GAME_TRANSFER_NULL"
    assert state["cross_game_transfer_helps"] is False
    assert state["cross_game_state_reduction"] == 1.0
    assert state["off_arc_axis_state"] == "MARGINAL_EXECUTION_GROUNDED_WIN"
    assert state["off_arc_demofit_beats_vote"] is True
    assert state["off_arc_delta"] == 0.02
    assert state["off_arc_verifier_is_oracle"] is True
    assert state["paper_ready"] is True
    assert state["v400_frame"] == mod.V400_FRAME


def test_run_happy_path_writes_valid_artifact_and_updates_archive(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4324: complete path writes the terminal archive artifact."""

    root = make_repo(tmp_path, duplicates=2)
    artifact = run_happy(root)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["archived_milestone"] == "2026.06.399"
    assert artifact["activated_milestone"] == "2026.06.400"
    assert artifact["active_milestone_confirmed"] == "2026.06.400"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["research_complete_record_action"] == "deduped"
    assert artifact["research_complete_duplicates_removed"] == 1
    assert artifact["v399_close_state"]["in_generation_axis_state"] == "CLOSED_ORACLE_DISTINCT"
    assert artifact["v399_close_state"]["cross_domain_axis_state"] == "RETIRED_DOMAIN_BOUND"
    assert artifact["v399_close_state"]["efficiency_axis_state"] == "ALWAYS_ENERGY_DOMINATES"
    assert artifact["v399_close_state"]["arc_reproducible_total_levels"] == 13
    assert artifact["v399_close_state"]["arc_first_live_submission"] is True
    assert artifact["field_principles"]["v399_close_state"] == mod.FIELD_PRINCIPLES[
        "v399_close_state"
    ]
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.is_sha256(artifact["reproducibility_checksum"])
    complete_text = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(complete_text) == 1
    assert "in-generation moat CLOSED oracle-distinct" in complete_text
    assert "cross-domain RETIRED domain-bound" in complete_text
    assert "learned-frame-encoder" in complete_text
    mod.validate_artifact(artifact)


def test_run_blocked_preconditions_before_research_complete_edit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4324-BLOCKED-PRECONDITION: blocked paths do not fabricate."""

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
    (root4 / "research-roadmap.yaml").write_text("milestone: 2026.06.399\n", encoding="utf-8")
    artifact4 = json.loads(mod.run(root4, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact4["honest_verdict"] == "blocked_v400_not_active"

    root5 = make_repo(tmp_path / "source_missing")
    (
        root5 / "results" / "experiment_4315_diffusiongemma_reward_guided_stitching.json"
    ).unlink()
    artifact5 = json.loads(mod.run(root5, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact5["honest_verdict"] == "blocked_in_generation_artifact_missing"


def test_run_blocked_research_complete_edit_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4324: invalid archive edits are blocked before completion."""

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
    """REQ-REPORT-4324: schema validation and entrypoints preserve the contract."""

    root = make_repo(tmp_path)
    state = mod.build_v399_close_state(mod.read_v399_sources(root))
    complete = mod.build_complete_artifact(
        v399_close_state=state,
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

    import carnot.experiment_4324_archive_v399_activate_v400 as entrypoint

    called: dict[str, Path] = {}

    def fake_run(root_path: Path = mod.REPO_ROOT) -> Path:
        called["root"] = Path(root_path)
        return root / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(entrypoint, "run", fake_run)
    assert entrypoint.main() == 0
    assert called["root"] == entrypoint.REPO_ROOT

    script_path = Path("results/experiment_4324_archive_v399_activate_v400.py")
    spec = importlib.util.spec_from_file_location("exp4324_archive_script", script_path)
    assert spec and spec.loader
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "run", fake_run)
    assert script.main() == 0


def test_validate_artifact_rejection_guards(tmp_path: Path) -> None:
    """REQ-REPORT-4324: validation rejects artifacts that launder the `.399` truth."""

    good = run_happy(make_repo(tmp_path))

    def set_path(obj: dict, path: list[str], value: object) -> None:
        cur = obj
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = value

    cases = [
        ("missing required fields", lambda a: a.pop("v399_close_state")),
        ("terminal-prefixed", lambda a: a.__setitem__("honest_verdict", "done")),
        (
            "principle must match REQ-REPORT-4324",
            lambda a: a["field_principles"].__setitem__("v399_close_state", "wrong"),
        ),
        ("archived milestone", lambda a: a.__setitem__("archived_milestone", "2026.06.999")),
        ("activated milestone", lambda a: a.__setitem__("activated_milestone", "2026.06.999")),
        ("research-complete YAML", lambda a: a.__setitem__("research_complete_yaml_parses", False)),
        ("exclusion manifest", lambda a: a.__setitem__("exclusion_manifest_parses", False)),
        ("pretest suite", lambda a: a.__setitem__("pretest_suite_green", False)),
        ("active milestone", lambda a: a.__setitem__("active_milestone_confirmed", "2026.06.399")),
        ("v399_close_state must be a mapping", lambda a: a.__setitem__("v399_close_state", "x")),
        (
            "in-generation closed",
            lambda a: set_path(a, ["v399_close_state", "in_generation_moat_holds"], False),
        ),
        (
            "in-generation oracle-distinct",
            lambda a: set_path(a, ["v399_close_state", "in_generation_verifier_is_oracle"], True),
        ),
        (
            "cross-domain retired",
            lambda a: set_path(a, ["v399_close_state", "cross_domain_axis_state"], "OPEN"),
        ),
        (
            "cross-domain no repropose",
            lambda a: set_path(a, ["v399_close_state", "cross_domain_do_not_repropose"], False),
        ),
        (
            "efficiency energy dominates",
            lambda a: set_path(a, ["v399_close_state", "efficiency_axis_state"], "CASCADE"),
        ),
        (
            "ARC 13",
            lambda a: set_path(a, ["v399_close_state", "arc_reproducible_total_levels"], 23),
        ),
        (
            "first live submission",
            lambda a: set_path(a, ["v399_close_state", "arc_first_live_submission"], False),
        ),
        (
            "cross-game transfer null",
            lambda a: set_path(a, ["v399_close_state", "cross_game_transfer_helps"], True),
        ),
        (
            "off-ARC execution-grounded",
            lambda a: set_path(a, ["v399_close_state", "off_arc_verifier_is_oracle"], False),
        ),
        ("paper", lambda a: set_path(a, ["v399_close_state", "paper_ready"], False)),
        ("v400 frame", lambda a: set_path(a, ["v399_close_state", "v400_frame"], "redo")),
    ]
    for label, mutate in cases:
        artifact = copy.deepcopy(good)
        mutate(artifact)
        with pytest.raises(ValueError, match=label):
            mod.validate_artifact(artifact)
