"""Tests for Exp 4280 `.395` archive / `.396` activation.

Spec refs: REQ-REPORT-4280, SCENARIO-REPORT-4280,
SCENARIO-REPORT-4280-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v395_activate_v396_4280 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="81 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.394\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.395\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-16'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4279-capstone-v395\n"
        "    result: OK\n"
    )
    return head + block * duplicates


def _capstone(**overrides: object) -> dict:
    payload = {
        "experiment": 4279,
        "honest_verdict": (
            "complete: capstone_v395_cross_family_generalizes_self_learning_static_ceiling_"
            "diffusiongemma_full_run_ready_arc20_game_wa30-ee6fef47_paper_ready_"
            "cross_family_generalizes_True_hardened_win_True_"
            "diffusiongemma_full_run_gate_True_excluded_0"
        ),
        "headline_outcome": (
            "cross_family_generalizes_self_learning_static_ceiling_"
            "diffusiongemma_full_run_ready_arc20_game_wa30-ee6fef47_paper_ready"
        ),
        "cross_family_generalizes": True,
        "hardened_win": True,
        "diffusiongemma_full_run_gate": True,
        "paper_ready": True,
        "flagged_artifacts_excluded": [],
        "cross_family": _cross_family(),
        "hardening": {
            "hardened_win": True,
            "provenance_blind": {
                "win_survives_provenance_blind": True,
                "provenance_blind_delta": 0.3846153846,
                "provenance_blind_ci95": [0.25, 0.5192307692],
                "verifier_is_oracle": False,
            },
            "multiseed": {
                "oracle_distinct_win_replicates": True,
                "mean_delta": 0.4576923077,
                "cross_seed_ci95_excludes_zero": True,
                "n_seeds": 5,
                "verifier_is_oracle": False,
            },
        },
        "scale_up_readiness": _preflight(),
        "arc_progress": _arc_progress(),
        "self_learning": _self_learning(),
        "registry_read": {"registry_reconciled": True, "regression_guard_passed": True},
    }
    payload.update(overrides)
    return payload


def _cross_family(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: cross_family_generalizes",
        "cross_family_delta": 0.4038461538,
        "cross_family_ci95": [0.25, 0.5576923077],
        "ci95_excludes_zero": True,
        "cross_family_win_holds": True,
        "held_out_family_n": 52,
        "held_out_task_n": 52,
        "within_minus_cross_gap": 0.0384615385,
        "matched_control_delta": 0.4423076923,
        "oracle_at_k": 0.8269230769,
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _self_learning(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: static_is_the_ceiling_for_online_adaptation",
        "online_adaptation_helps": False,
        "static_cross_family_delta": 0.4038461538,
        "online_cross_family_delta": 0.5,
        "online_minus_static_ci95": [0.0, 0.1923076923],
        "held_out_task_n": 52,
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _preflight(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: diffusiongemma_loader_fix_preflight_go",
        "loader_repaired": True,
        "preflight_go": True,
        "guidance_changes_selection": True,
        "full_run_cost_estimate_s": 0.071224,
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _arc_progress(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "success: incremental_progress_wa30-ee6fef47_advanced_to_L1_total20",
        "total_levels_solved": 20,
        "total_levels": 20,
        "levels_completed": 1,
        "game_advanced": "wa30-ee6fef47",
    }
    payload.update(overrides)
    return payload


def _registry(**overrides: object) -> dict:
    payload = {
        "honest_verdict": (
            "complete: registry_gaps_manifest_reconciled_to_v395_truth_"
            "regression_guard_passed_True_retirements_2_gaps_logged_1"
        ),
        "registry_reconciled": True,
        "regression_guard_passed": True,
        "retirements_recorded": [
            {"id": "code_oracle_distinct_replication_corpus_specific_retired_exp4264"},
            {"id": "verifier_as_reward_in_loop_axis_out_of_band_exp4263"},
        ],
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
        "retired:\n- experiment_id: 4264\n  reason: corpus-specific\n", encoding="utf-8"
    )
    (root / "research-roadmap.yaml").write_text("milestone: 2026.06.396\n", encoding="utf-8")
    _write_json(root / "results" / "experiment_4279_capstone_v395.json", _capstone())
    _write_json(
        root / "results" / "experiment_4271_arc_cross_family_transfer_existing_pool.json",
        _cross_family(),
    )
    _write_json(
        root / "results" / "experiment_4273_arc_cross_family_online_adaptation.json",
        _self_learning(),
    )
    _write_json(
        root / "results" / "experiment_4274_diffusiongemma_loader_fix_preflight.json",
        _preflight(),
    )
    _write_json(
        root / "results" / "experiment_4275_arc_incremental_progress_new_game.json",
        _arc_progress(),
    )
    _write_json(
        root / "results" / "experiment_4277_verifier_registry_gaps_hygiene.json",
        _registry(),
    )
    return root


def run_happy(root: Path) -> dict:
    out = mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5)
    return json.loads(out.read_text(encoding="utf-8"))


def test_req_report_4280_spec_declares_contract() -> None:
    """REQ-REPORT-4280: OpenSpec declares the `.395` landmark contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    assert "REQ-REPORT-4280" in spec
    assert "SCENARIO-REPORT-4280" in spec
    assert "SCENARIO-REPORT-4280-BLOCKED-PRECONDITION" in spec
    assert "cross-family GENERALIZED" in spec
    assert "gate flipped open" in spec
    assert "`+0.404`" in spec
    assert "`hardened_win=true`" in spec
    assert "`diffusiongemma_full_run_gate=true`" in spec
    assert "ARC `20` levels" in spec
    assert mod.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert mod.FIELD_PRINCIPLES["v395_close_state"] in spec
    assert mod.FIELD_PRINCIPLES["preconditions_checked"] in spec


def test_helpers_and_archive_record_editing(tmp_path: Path) -> None:
    """REQ-REPORT-4280: helper behavior is deterministic and YAML-safe."""

    assert mod.archive_record_count(_research_complete_text(duplicates=3)) == 3
    assert mod._record_id("- id: '2026.06.395'") == "2026.06.395"
    assert mod._record_id("  - id: nested") is None
    assert mod._yaml_quote("it's open") == "'it''s open'"
    assert mod._rounded_pair("bad", [1.0, 2.0]) == [1.0, 2.0]
    assert mod._insert_before_tasks(["  title: no tasks"], "  finding: x") == [
        "  title: no tasks",
        "  finding: x",
    ]

    close_state = mod.build_v395_close_state(
        {
            "4279": _capstone(),
            "4271": _cross_family(),
            "4273": _self_learning(),
            "4274": _preflight(),
            "4275": _arc_progress(),
            "4277": _registry(),
        }
    )
    deduped, removed, action = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=3), close_state
    )
    assert (removed, action) == (2, "deduped")
    assert mod.archive_record_count(deduped) == 1
    assert "LANDMARK" in deduped
    assert "RUN the deferred DiffusionGemma full run" in deduped
    assert mod.yaml_parses(deduped)

    updated, removed2, action2 = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=1), close_state
    )
    assert (removed2, action2) == (0, "updated")
    unchanged, removed3, action3 = mod.dedupe_or_update_record(updated, close_state)
    assert (unchanged, removed3, action3) == (updated, 0, "unchanged")
    appended, removed4, action4 = mod.dedupe_or_update_record(
        "# history\nmilestones:\n- id: 2026.06.394\n  finding: prior\n", close_state
    )
    assert (removed4, action4) == (0, "appended")
    assert "activation_recorded: exp4280-archive-v395-activate-v396" in appended
    added_finding, removed5, action5 = mod.dedupe_or_update_record(
        "milestones:\n- id: 2026.06.395\n  title: missing finding\n  tasks:\n  - id: exp4279\n",
        close_state,
    )
    assert (removed5, action5) == (0, "updated")
    assert "cross-family GENERALIZED" in added_finding


def test_read_sources_and_build_v395_close_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4280: close-state records the landmark cross-family win."""

    root = make_repo(tmp_path)
    sources = mod.read_v395_sources(root)
    cited = mod.build_cited_upstream(root)
    assert sources["4279"]["hardened_win"] is True
    assert {item["experiment_id"] for item in cited} == {
        "4279",
        "4271",
        "4273",
        "4274",
        "4275",
        "4277",
    }
    assert all(item["sha256"] is None or mod.is_sha256(item["sha256"]) for item in cited)

    state = mod.build_v395_close_state(sources)
    assert state["summary"] == "cross_family_generalized_gate_open_loader_repaired_arc20"
    assert state["cross_family_generalizes"] is True
    assert state["cross_family_delta"] == 0.404
    assert state["cross_family_ci95"] == [0.25, 0.558]
    assert state["cross_family_ci95_excludes_zero"] is True
    assert state["held_out_task_n"] == 52
    assert state["within_minus_cross_gap"] == 0.0385
    assert state["verifier_is_oracle"] is False
    assert state["provenance_blind_delta"] == 0.385
    assert state["multiseed_mean_delta"] == 0.458
    assert state["hardened_win"] is True
    assert state["diffusiongemma_full_run_gate"] is True
    assert state["loader_repaired"] is True
    assert state["preflight_go"] is True
    assert state["arc_total_levels_solved"] == 20
    assert state["arc_game_advanced"] == "wa30-ee6fef47"
    assert state["self_learning_status"] == "static_is_the_ceiling"
    assert state["online_minus_static_ci95"] == [0.0, 0.192]
    assert state["code_oracle_distinct_retired"] is True
    assert state["verifier_as_reward_in_loop_retired"] is True
    assert state["paper_ready"] is True
    assert state["v396_frame"] == mod.V396_FRAME


def test_run_happy_path_writes_valid_artifact_and_updates_archive(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4280: complete path writes the terminal archive artifact."""

    root = make_repo(tmp_path, duplicates=2)
    artifact = run_happy(root)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["archived_milestone"] == "2026.06.395"
    assert artifact["activated_milestone"] == "2026.06.396"
    assert artifact["active_milestone_confirmed"] == "2026.06.396"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["research_complete_record_action"] == "deduped"
    assert artifact["research_complete_duplicates_removed"] == 1
    assert artifact["v395_close_state"]["cross_family_generalizes"] is True
    assert artifact["v395_close_state"]["hardened_win"] is True
    assert artifact["v395_close_state"]["diffusiongemma_full_run_gate"] is True
    assert artifact["field_principles"]["v395_close_state"] == mod.FIELD_PRINCIPLES[
        "v395_close_state"
    ]
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.is_sha256(artifact["reproducibility_checksum"])
    complete_text = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(complete_text) == 1
    assert "cross-family GENERALIZED" in complete_text
    assert "DiffusionGemma full-run gate OPEN" in complete_text
    mod.validate_artifact(artifact)


def test_run_blocked_preconditions_before_research_complete_edit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4280-BLOCKED-PRECONDITION: blocked paths do not fabricate."""

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
    (root4 / "research-roadmap.yaml").write_text("milestone: 2026.06.395\n", encoding="utf-8")
    artifact4 = json.loads(mod.run(root4, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact4["honest_verdict"] == "blocked_v396_not_active"

    root5 = make_repo(tmp_path / "source_missing")
    (root5 / "results" / "experiment_4274_diffusiongemma_loader_fix_preflight.json").unlink()
    artifact5 = json.loads(mod.run(root5, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact5["honest_verdict"] == "blocked_diffusiongemma_preflight_missing"


def test_run_blocked_research_complete_edit_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4280: invalid archive edits are blocked before completion."""

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
    """REQ-REPORT-4280: schema validation and entrypoints preserve the contract."""

    root = make_repo(tmp_path)
    state = mod.build_v395_close_state(mod.read_v395_sources(root))
    complete = mod.build_complete_artifact(
        v395_close_state=state,
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

    import carnot.experiment_4280_archive_v395_activate_v396 as entrypoint

    called: dict[str, Path] = {}

    def fake_run(root_path: Path = mod.REPO_ROOT) -> Path:
        called["root"] = Path(root_path)
        return root / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(entrypoint, "run", fake_run)
    assert entrypoint.main() == 0
    assert called["root"] == entrypoint.REPO_ROOT

    script_path = Path("results/experiment_4280_archive_v395_activate_v396.py")
    spec = importlib.util.spec_from_file_location("exp4280_archive_script", script_path)
    assert spec and spec.loader
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "run", fake_run)
    assert script.main() == 0


def test_validate_artifact_rejection_guards(tmp_path: Path) -> None:
    """REQ-REPORT-4280: validation rejects artifacts that launder the `.395` truth."""

    good = run_happy(make_repo(tmp_path))

    def set_path(obj: dict, path: list[str], value: object) -> None:
        cur = obj
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = value

    cases = [
        ("missing required fields", lambda a: a.pop("v395_close_state")),
        ("terminal-prefixed", lambda a: a.__setitem__("honest_verdict", "done")),
        ("principle must match REQ-REPORT-4280", lambda a: a["field_principles"].__setitem__("v395_close_state", "wrong")),
        ("archived milestone", lambda a: a.__setitem__("archived_milestone", "2026.06.999")),
        ("activated milestone", lambda a: a.__setitem__("activated_milestone", "2026.06.999")),
        ("research-complete YAML", lambda a: a.__setitem__("research_complete_yaml_parses", False)),
        ("exclusion manifest", lambda a: a.__setitem__("exclusion_manifest_parses", False)),
        ("pretest suite", lambda a: a.__setitem__("pretest_suite_green", False)),
        ("active milestone", lambda a: a.__setitem__("active_milestone_confirmed", "2026.06.395")),
        ("v395_close_state must be a mapping", lambda a: a.__setitem__("v395_close_state", "x")),
        ("cross-family", lambda a: set_path(a, ["v395_close_state", "cross_family_generalizes"], False)),
        ("cross delta", lambda a: set_path(a, ["v395_close_state", "cross_family_delta"], 0.0)),
        ("cross CI", lambda a: set_path(a, ["v395_close_state", "cross_family_ci95_excludes_zero"], False)),
        ("held-out task", lambda a: set_path(a, ["v395_close_state", "held_out_task_n"], 0)),
        ("oracle distinct", lambda a: set_path(a, ["v395_close_state", "verifier_is_oracle"], True)),
        ("provenance", lambda a: set_path(a, ["v395_close_state", "provenance_blind_delta"], 0.0)),
        ("multi-seed", lambda a: set_path(a, ["v395_close_state", "multiseed_mean_delta"], 0.0)),
        ("hardened win", lambda a: set_path(a, ["v395_close_state", "hardened_win"], False)),
        ("DiffusionGemma gate", lambda a: set_path(a, ["v395_close_state", "diffusiongemma_full_run_gate"], False)),
        ("loader", lambda a: set_path(a, ["v395_close_state", "loader_repaired"], False)),
        ("preflight", lambda a: set_path(a, ["v395_close_state", "preflight_go"], False)),
        ("ARC levels", lambda a: set_path(a, ["v395_close_state", "arc_total_levels_solved"], 19)),
        ("self-learning", lambda a: set_path(a, ["v395_close_state", "self_learning_status"], "online_win")),
        ("code retired", lambda a: set_path(a, ["v395_close_state", "code_oracle_distinct_retired"], False)),
        ("reward retired", lambda a: set_path(a, ["v395_close_state", "verifier_as_reward_in_loop_retired"], False)),
        ("paper", lambda a: set_path(a, ["v395_close_state", "paper_ready"], False)),
        ("v396 frame", lambda a: set_path(a, ["v395_close_state", "v396_frame"], "redo")),
    ]
    for label, mutate in cases:
        artifact = copy.deepcopy(good)
        mutate(artifact)
        with pytest.raises(ValueError, match=label):
            mod.validate_artifact(artifact)
