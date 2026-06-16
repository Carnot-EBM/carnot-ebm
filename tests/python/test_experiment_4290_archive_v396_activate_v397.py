"""Tests for Exp 4290 `.396` archive / `.397` activation.

Spec refs: REQ-REPORT-4290, SCENARIO-REPORT-4290,
SCENARIO-REPORT-4290-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v396_activate_v397_4290 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="81 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.395\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.396\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-16'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4289-capstone-v396\n"
        "    result: OK\n"
    )
    return head + block * duplicates


def _capstone(**overrides: object) -> dict:
    payload = {
        "experiment": 4289,
        "honest_verdict": (
            "complete: diffusiongemma_partial_state_blocked_arcgen_excluded_flagged_"
            "efficiency_parity_arc21"
        ),
        "headline_outcome": (
            "partial_state_blocked_arcgen_excluded_flagged_efficiency_parity_"
            "self_learning_excluded_flagged_arc21_game_ls20-9607627b_paper_ready"
        ),
        "guidance_moat_holds": False,
        "diffusiongemma_thesis_state": "partial_state_blocked",
        "cross_family_hardens_on_arcgen": False,
        "verifier_efficiency_parity": True,
        "paper_ready": True,
        "flagged_artifacts_excluded": [
            {"artifact_key": "4282_arcgen", "experiment_id": 4282},
            {"artifact_key": "4283_self_learning", "experiment_id": 4283},
        ],
        "diffusiongemma_guidance": _diffusiongemma(),
        "arcgen_cross_family": {"status": "excluded_flagged_adversarial"},
        "self_learning": {"status": "excluded_flagged_adversarial"},
        "efficiency": _efficiency(),
        "arc_progress": _arc_progress(),
    }
    payload.update(overrides)
    return payload


def _efficiency(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: efficiency_parity_at_lower_cost_true_delta_0.4423",
        "efficiency_parity_at_lower_cost": True,
        "accuracy_energy_verifier": 0.6538461538,
        "accuracy_llm_judge": 0.2115384615,
        "accuracy_delta": 0.4423076923,
        "accuracy_delta_ci95": [0.3076923077, 0.5769230769],
        "cost_ratio": 1.95e-08,
        "selection_task_n": 52,
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _diffusiongemma(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete_diffusiongemma_learned_verifier_cannot_score_partial_states",
        "diffusiongemma_guidance_moat": False,
        "carnot_minus_rfg_delta": 0.0,
        "carnot_minus_unguided_delta": 0.0,
        "guidance_moat_ci95": [0.0, 0.0],
        "guidance_changes_selection": True,
        "verifier_is_oracle": False,
        "headline_arm": {
            "learned_verifier_partial_state_support": {"can_score": False},
            "conditions": {
                "Carnot-verifier-guided": {"status": "blocked_partial_state_verifier"}
            },
        },
        "execution_grounded_arm": {
            "status": "not_run_after_headline_partial_state_block",
            "verifier_is_oracle": True,
        },
    }
    payload.update(overrides)
    return payload


def _arcgen(**overrides: object) -> dict:
    payload = {
        "experiment": "experiment_4282_arcgen_cross_family_stress",
        "honest_verdict": "complete: arcgen_cross_family_generalizes",
        "arcgen_cross_family_holds": True,
        "cross_family_delta": 1.0,
        "cross_family_ci95": [1.0, 1.0],
        "held_out_family_n": 10,
        "held_out_task_n": 50,
        "candidate_count": 200,
        "oracle_at_k": 1.0,
        "verifier_is_oracle": False,
        "pass_rates": {
            "vote_at_1": 0.0,
            "set_encoder_at_1": 1.0,
            "matched_control_at_1": 0.0,
        },
        "model_specs": {"arcgen_provenance": {"candidates_per_task": 4}},
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "DEGENERATE_SEPARATION"}],
        "per_substrate_delta": {"original_arc": _within_pool()},
    }
    payload.update(overrides)
    return payload


def _self_learning(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: powered_static_is_the_ceiling_for_self_learning",
        "online_adaptation_helps": False,
        "static_cross_family_delta": 0.5,
        "online_cross_family_delta": 0.5806451613,
        "tier2_cross_family_delta": 0.5,
        "adaptive_minus_static_ci95": [0.0, 0.16],
        "held_out_task_n": 48,
        "verifier_is_oracle": False,
        "pass_rates": {
            "static_family_mean_at_1": 0.7096774194,
            "online_family_mean_at_1": 0.7903225806,
            "tier2_family_mean_at_1": 0.7096774194,
        },
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "TAUTOLOGY"}],
    }
    payload.update(overrides)
    return payload


def _arc_progress(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "success: incremental_progress_ls20-9607627b_advanced_to_L1_total21",
        "total_levels_solved": 21,
        "total_levels": 21,
        "levels_completed": 1,
        "new_levels_solved_this_task": 1,
        "game_advanced": "ls20-9607627b",
    }
    payload.update(overrides)
    return payload


def _within_pool(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: cross_family_generalizes",
        "cross_family_win_holds": True,
        "cross_family_delta": 0.4038461538,
        "cross_family_ci95": [0.25, 0.5576923077],
        "ci95_excludes_zero": True,
        "held_out_family_n": 52,
        "held_out_task_n": 52,
        "within_minus_cross_gap": 0.0384615385,
        "oracle_at_k": 0.8269230769,
        "pass_rates": {"vote_at_1": 0.25, "set_encoder_at_1": 0.6538461538},
        "verifier_is_oracle": False,
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
    (root / "ops" / "exclusion_manifest.yaml").write_text(
        "retired:\n- experiment_id: 4282\n  reason: degenerate\n", encoding="utf-8"
    )
    (root / "research-roadmap.yaml").write_text("milestone: 2026.06.397\n", encoding="utf-8")
    (root / "docs" / "research-notes").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "research-notes" / "exp4282-arcgen-degenerate-audit-2026-06-16.md").write_text(
        "exp4282 is DEGENERATE; the cross-generator question remains OPEN.\n",
        encoding="utf-8",
    )
    (root / "openspec" / "change-proposals").mkdir(parents=True, exist_ok=True)
    (root / "openspec" / "change-proposals" / "research-roadmap-v397.md").write_text(
        "Frame .397 as close cross-generator, unblock in-generation, harden efficiency.\n",
        encoding="utf-8",
    )
    _write_json(root / "results" / "experiment_4289_capstone_v396.json", _capstone())
    _write_json(root / "results" / "experiment_4284_verifier_efficiency_vs_llm_judge.json", _efficiency())
    _write_json(root / "results" / "experiment_4281_diffusiongemma_energy_guided_full_run.json", _diffusiongemma())
    _write_json(root / "results" / "experiment_4282_arcgen_cross_family_stress.json", _arcgen())
    _write_json(root / "results" / "experiment_4283_self_learning_repowered_arcgen.json", _self_learning())
    _write_json(root / "results" / "experiment_4285_arc_incremental_progress_new_game.json", _arc_progress())
    _write_json(root / "results" / "experiment_4271_arc_cross_family_transfer_existing_pool.json", _within_pool())
    return root


def run_happy(root: Path) -> dict:
    out = mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5)
    return json.loads(out.read_text(encoding="utf-8"))


def test_req_report_4290_spec_declares_contract() -> None:
    """REQ-REPORT-4290: OpenSpec declares the `.396` scorecard contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    assert "REQ-REPORT-4290" in spec
    assert "SCENARIO-REPORT-4290" in spec
    assert "SCENARIO-REPORT-4290-BLOCKED-PRECONDITION" in spec
    assert "efficiency Pareto win" in spec
    assert "partial-state block" in spec
    assert "cross-generator transfer remains open" in spec
    assert "self-learning tier-2 no-op bug" in spec
    assert "ARC `21`" in spec
    assert mod.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert mod.FIELD_PRINCIPLES["v396_close_state"] in spec
    assert mod.FIELD_PRINCIPLES["preconditions_checked"] in spec


def test_helpers_and_archive_record_editing(tmp_path: Path) -> None:
    """REQ-REPORT-4290: helper behavior is deterministic and YAML-safe."""

    assert mod.archive_record_count(_research_complete_text(duplicates=3)) == 3
    assert mod._record_id("- id: '2026.06.396'") == "2026.06.396"
    assert mod._record_id("  - id: nested") is None
    assert mod._yaml_quote("it's open") == "'it''s open'"
    assert mod._rounded_pair("bad", [1.0, 2.0]) == [1.0, 2.0]
    assert mod._insert_before_tasks(["  title: no tasks"], "  finding: x") == [
        "  title: no tasks",
        "  finding: x",
    ]

    close_state = mod.build_v396_close_state(
        {
            "4289": _capstone(),
            "4284": _efficiency(),
            "4281": _diffusiongemma(),
            "4282": _arcgen(),
            "4283": _self_learning(),
            "4285": _arc_progress(),
            "4271": _within_pool(),
        }
    )
    deduped, removed, action = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=3), close_state
    )
    assert (removed, action) == (2, "deduped")
    assert mod.archive_record_count(deduped) == 1
    assert "EFFICIENCY Pareto win" in deduped
    assert "cross-generator still OPEN" in deduped
    assert mod.yaml_parses(deduped)

    updated, removed2, action2 = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=1), close_state
    )
    assert (removed2, action2) == (0, "updated")
    unchanged, removed3, action3 = mod.dedupe_or_update_record(updated, close_state)
    assert (unchanged, removed3, action3) == (updated, 0, "unchanged")
    appended, removed4, action4 = mod.dedupe_or_update_record(
        "# history\nmilestones:\n- id: 2026.06.395\n  finding: prior\n", close_state
    )
    assert (removed4, action4) == (0, "appended")
    assert "activation_recorded: exp4290-archive-v396-activate-v397" in appended


def test_read_sources_and_build_v396_close_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4290: close-state records the honest .396 scorecard."""

    root = make_repo(tmp_path)
    sources = mod.read_v396_sources(root)
    cited = mod.build_cited_upstream(root)
    assert sources["4289"]["diffusiongemma_thesis_state"] == "partial_state_blocked"
    assert {item["experiment_id"] for item in cited if item["kind"] == "artifact"} == {
        "4289",
        "4284",
        "4281",
        "4282",
        "4283",
        "4285",
        "4271",
    }
    assert all(item["sha256"] is None or mod.is_sha256(item["sha256"]) for item in cited)

    state = mod.build_v396_close_state(sources)
    assert state["summary"] == "efficiency_pareto_partial_state_blocked_cross_generator_open_arc21"
    assert state["efficiency_pareto_win"] is True
    assert state["efficiency_needs_hardening"] is True
    assert state["accuracy_energy_verifier"] == 0.654
    assert state["accuracy_llm_judge"] == 0.212
    assert state["judge_below_random"] is True
    assert state["accuracy_delta_ci95"] == [0.308, 0.577]
    assert state["cost_ratio"] == 1.95e-08
    assert state["efficiency_verifier_is_oracle"] is False
    assert state["diffusiongemma_guidance_blocked"] is True
    assert state["diffusiongemma_thesis_state"] == "partial_state_blocked"
    assert state["can_score_partial_states"] is False
    assert state["guidance_moat_holds"] is False
    assert state["cross_generator_open"] is True
    assert state["arcgen_degenerate"] is True
    assert state["arcgen_cross_family_delta"] == 1.0
    assert state["arcgen_cross_family_ci95"] == [1.0, 1.0]
    assert state["arcgen_vote_at_1"] == 0.0
    assert state["arcgen_oracle_at_k"] == 1.0
    assert state["arcgen_candidates_per_task"] == 4
    assert state["within_pool_win_stands"] is True
    assert state["within_pool_cross_family_delta"] == 0.404
    assert state["self_learning_tier2_noop_bug"] is True
    assert state["tier2_equals_static"] is True
    assert state["arc_total_levels_solved"] == 21
    assert state["arc_game_advanced"] == "ls20-9607627b"
    assert state["paper_ready"] is True
    assert state["flagged_artifact_ids"] == [4282, 4283]
    assert state["v397_frame"] == mod.V397_FRAME


def test_run_happy_path_writes_valid_artifact_and_updates_archive(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4290: complete path writes the terminal archive artifact."""

    root = make_repo(tmp_path, duplicates=2)
    artifact = run_happy(root)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["archived_milestone"] == "2026.06.396"
    assert artifact["activated_milestone"] == "2026.06.397"
    assert artifact["active_milestone_confirmed"] == "2026.06.397"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["research_complete_record_action"] == "deduped"
    assert artifact["research_complete_duplicates_removed"] == 1
    assert artifact["v396_close_state"]["efficiency_pareto_win"] is True
    assert artifact["v396_close_state"]["cross_generator_open"] is True
    assert artifact["v396_close_state"]["diffusiongemma_guidance_blocked"] is True
    assert artifact["field_principles"]["v396_close_state"] == mod.FIELD_PRINCIPLES[
        "v396_close_state"
    ]
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.is_sha256(artifact["reproducibility_checksum"])
    complete_text = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(complete_text) == 1
    assert "EFFICIENCY Pareto win" in complete_text
    assert "partial-state BLOCKED" in complete_text
    mod.validate_artifact(artifact)


def test_run_blocked_preconditions_before_research_complete_edit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4290-BLOCKED-PRECONDITION: blocked paths do not fabricate."""

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
    (root_manifest / "ops" / "exclusion_manifest.yaml").write_text("a: : :\n- [\n", encoding="utf-8")
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
    (root4 / "research-roadmap.yaml").write_text("milestone: 2026.06.396\n", encoding="utf-8")
    artifact4 = json.loads(mod.run(root4, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact4["honest_verdict"] == "blocked_v397_not_active"

    root5 = make_repo(tmp_path / "source_missing")
    (root5 / "results" / "experiment_4284_verifier_efficiency_vs_llm_judge.json").unlink()
    artifact5 = json.loads(mod.run(root5, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact5["honest_verdict"] == "blocked_efficiency_artifact_missing"


def test_run_blocked_research_complete_edit_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4290: invalid archive edits are blocked before completion."""

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
    """REQ-REPORT-4290: schema validation and entrypoints preserve the contract."""

    root = make_repo(tmp_path)
    state = mod.build_v396_close_state(mod.read_v396_sources(root))
    complete = mod.build_complete_artifact(
        v396_close_state=state,
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

    import carnot.experiment_4290_archive_v396_activate_v397 as entrypoint

    called: dict[str, Path] = {}

    def fake_run(root_path: Path = mod.REPO_ROOT) -> Path:
        called["root"] = Path(root_path)
        return root / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(entrypoint, "run", fake_run)
    assert entrypoint.main() == 0
    assert called["root"] == entrypoint.REPO_ROOT

    script_path = Path("results/experiment_4290_archive_v396_activate_v397.py")
    spec = importlib.util.spec_from_file_location("exp4290_archive_script", script_path)
    assert spec and spec.loader
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "run", fake_run)
    assert script.main() == 0


def test_validate_artifact_rejection_guards(tmp_path: Path) -> None:
    """REQ-REPORT-4290: validation rejects artifacts that launder the `.396` truth."""

    good = run_happy(make_repo(tmp_path))

    def set_path(obj: dict, path: list[str], value: object) -> None:
        cur = obj
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = value

    cases = [
        ("missing required fields", lambda a: a.pop("v396_close_state")),
        ("terminal-prefixed", lambda a: a.__setitem__("honest_verdict", "done")),
        ("principle must match REQ-REPORT-4290", lambda a: a["field_principles"].__setitem__("v396_close_state", "wrong")),
        ("archived milestone", lambda a: a.__setitem__("archived_milestone", "2026.06.999")),
        ("activated milestone", lambda a: a.__setitem__("activated_milestone", "2026.06.999")),
        ("research-complete YAML", lambda a: a.__setitem__("research_complete_yaml_parses", False)),
        ("exclusion manifest", lambda a: a.__setitem__("exclusion_manifest_parses", False)),
        ("pretest suite", lambda a: a.__setitem__("pretest_suite_green", False)),
        ("active milestone", lambda a: a.__setitem__("active_milestone_confirmed", "2026.06.396")),
        ("v396_close_state must be a mapping", lambda a: a.__setitem__("v396_close_state", "x")),
        ("efficiency win", lambda a: set_path(a, ["v396_close_state", "efficiency_pareto_win"], False)),
        ("efficiency hardening", lambda a: set_path(a, ["v396_close_state", "efficiency_needs_hardening"], False)),
        ("judge below random", lambda a: set_path(a, ["v396_close_state", "judge_below_random"], False)),
        ("oracle distinct efficiency", lambda a: set_path(a, ["v396_close_state", "efficiency_verifier_is_oracle"], True)),
        ("partial state blocked", lambda a: set_path(a, ["v396_close_state", "diffusiongemma_guidance_blocked"], False)),
        ("thesis state", lambda a: set_path(a, ["v396_close_state", "diffusiongemma_thesis_state"], "won")),
        ("cross-generator open", lambda a: set_path(a, ["v396_close_state", "cross_generator_open"], False)),
        ("arcgen degenerate", lambda a: set_path(a, ["v396_close_state", "arcgen_degenerate"], False)),
        ("arcgen vote", lambda a: set_path(a, ["v396_close_state", "arcgen_vote_at_1"], 0.1)),
        ("arcgen oracle", lambda a: set_path(a, ["v396_close_state", "arcgen_oracle_at_k"], 0.9)),
        ("within-pool stands", lambda a: set_path(a, ["v396_close_state", "within_pool_win_stands"], False)),
        ("tier2 no-op", lambda a: set_path(a, ["v396_close_state", "self_learning_tier2_noop_bug"], False)),
        ("ARC levels", lambda a: set_path(a, ["v396_close_state", "arc_total_levels_solved"], 20)),
        ("paper", lambda a: set_path(a, ["v396_close_state", "paper_ready"], False)),
        ("v397 frame", lambda a: set_path(a, ["v396_close_state", "v397_frame"], "redo")),
    ]
    for label, mutate in cases:
        artifact = copy.deepcopy(good)
        mutate(artifact)
        with pytest.raises(ValueError, match=label):
            mod.validate_artifact(artifact)
