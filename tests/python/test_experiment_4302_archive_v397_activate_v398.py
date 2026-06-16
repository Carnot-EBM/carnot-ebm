"""Tests for Exp 4302 `.397` archive / `.398` activation.

Spec refs: REQ-REPORT-4302, SCENARIO-REPORT-4302,
SCENARIO-REPORT-4302-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v397_activate_v398_4302 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="81 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.396\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.397\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-16'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4301-capstone-v397\n"
        "    result: blocked_v397_artifacts_missing\n"
    )
    return head + block * duplicates


def _capstone(**overrides: object) -> dict:
    payload = {
        "experiment": 4301,
        "honest_verdict": "blocked_v397_artifacts_missing",
        "headline_outcome": "blocked_v397_artifacts_missing",
        "cross_generator_moat_closes": False,
        "in_generation_moat_holds": False,
        "efficiency_pareto_hardened": False,
        "paper_ready": None,
        "missing_upstream_artifacts": [
            {"artifact_key": "4294_efficiency", "experiment_id": 4294, "reason": "missing"}
        ],
        "flagged_artifacts_excluded": [{"artifact_key": "4293_generation", "experiment_id": 4293}],
    }
    payload.update(overrides)
    return payload


def _cross_generator(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: arcgen_cross_generator_generalizes",
        "cross_generator_holds": True,
        "cross_generator_delta": 0.5,
        "cross_generator_ci95": [0.2916666667, 0.7083333333],
        "held_out_generator_n": 8,
        "held_out_task_n": 24,
        "oracle_at_k": 0.75,
        "pass_rates": {"vote_at_1": 0.25, "set_encoder_at_1": 0.75},
        "non_degenerate_guards_pass": True,
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _partial_state(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: partial_state_diffusion_scorer_built_leak_free",
        "partial_state_scorer_built": True,
        "partial_state_leak_free": True,
        "partial_state_auroc": 0.966143,
        "leak_ablation_auroc": 0.937365,
        "scorer_loadable": True,
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _generation(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: diffusiongemma_guidance_moat_won",
        "diffusiongemma_guidance_moat": True,
        "carnot_minus_rfg_delta": 0.566667,
        "carnot_minus_unguided_delta": 0.566667,
        "guidance_moat_ci95": [0.366667, 0.766667],
        "condition_accuracy": {
            "carnot": 0.866667,
            "entrgi": 0.3,
            "rfg": 0.3,
            "unguided": 0.3,
        },
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _self_learning(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: adaptive_self_learning_improves_generalization",
        "online_adaptation_helps": True,
        "static_cross_family_delta": 0.4166666667,
        "online_cross_family_delta": 0.4833333333,
        "tier2_memory_cross_family_delta": 0.4277777778,
        "tier2_retrieval_cross_family_delta": 0.4555555556,
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _arc_progress(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "success: incremental_progress_r11l-495a7899_advanced_to_L1_total22",
        "total_levels_solved": 22,
        "total_levels": 22,
        "levels_completed": 1,
        "new_levels_solved_this_task": 1,
        "game_advanced": "r11l-495a7899",
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
        "retired:\n- experiment_id: 4293\n  reason: degenerate_controls\n", encoding="utf-8"
    )
    (root / "research-roadmap.yaml").write_text("milestone: 2026.06.398\n", encoding="utf-8")
    notes = root / "docs" / "research-notes"
    notes.mkdir(parents=True, exist_ok=True)
    (notes / "exp4301-capstone-blocked-spurious-false-2026-06-16.md").write_text(
        "exp4301 all-False was spurious; cross-generator CLOSED; efficiency unresolved.\n",
        encoding="utf-8",
    )
    (notes / "exp4293-in-generation-moat-degenerate-controls-audit-2026-06-16.md").write_text(
        "exp4293 controls were degenerate; in-generation moat NOT held.\n",
        encoding="utf-8",
    )
    roadmap_dir = root / "openspec" / "change-proposals"
    roadmap_dir.mkdir(parents=True, exist_ok=True)
    (roadmap_dir / "research-roadmap-v398.md").write_text(
        "Frame .398 as prove-efficiency-parity + establish-in-generation + broaden-to-cross-domain.\n",
        encoding="utf-8",
    )
    _write_json(root / "results" / "experiment_4301_capstone_v397.json", _capstone())
    _write_json(
        root / "results" / "experiment_4291_arcgen_cross_generator_nondegenerate.json",
        _cross_generator(),
    )
    _write_json(
        root / "results" / "experiment_4292_partial_state_diffusion_scorer_build.json",
        _partial_state(),
    )
    _write_json(
        root / "results" / "experiment_4293_diffusiongemma_energy_guided_run_partial_state.json",
        _generation(),
    )
    _write_json(
        root / "results" / "experiment_4295_self_learning_tier2_fixed_retrieval.json",
        _self_learning(),
    )
    _write_json(
        root / "results" / "experiment_4296_arc_incremental_progress_new_game.json", _arc_progress()
    )
    _write_json(
        root / "results" / "experiment_4299_verifier_registry_gaps_hygiene.json",
        {"honest_verdict": "blocked_v397_artifacts_missing"},
    )
    _write_json(
        root / "results" / "experiment_4300_hardware_continuity.json",
        {"honest_verdict": "complete: hardware continuity"},
    )
    return root


def run_happy(root: Path) -> dict:
    out = mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5)
    return json.loads(out.read_text(encoding="utf-8"))


def test_req_report_4302_spec_declares_contract() -> None:
    """REQ-REPORT-4302: OpenSpec declares the true `.397` scorecard contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    assert "REQ-REPORT-4302" in spec
    assert "SCENARIO-REPORT-4302" in spec
    assert "SCENARIO-REPORT-4302-BLOCKED-PRECONDITION" in spec
    assert "cross-generator axis CLOSED" in spec
    assert "in-generation moat is NOT held" in spec
    assert "efficiency is UNRESOLVED" in spec
    assert "ARC reached `22`" in spec
    assert mod.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert mod.FIELD_PRINCIPLES["v397_close_state"] in spec
    assert mod.FIELD_PRINCIPLES["preconditions_checked"] in spec


def test_helpers_and_archive_record_editing(tmp_path: Path) -> None:
    """REQ-REPORT-4302: helper behavior is deterministic and YAML-safe."""

    assert mod.archive_record_count(_research_complete_text(duplicates=3)) == 3
    assert mod._record_id("- id: '2026.06.397'") == "2026.06.397"
    assert mod._record_id("  - id: nested") is None
    assert mod._yaml_quote("efficiency isn't null") == "'efficiency isn''t null'"
    assert mod._rounded_pair("bad", [1.0, 2.0]) == [1.0, 2.0]
    assert mod._controls_degenerate(
        {
            "condition_accuracy": {"rfg": 0.1, "unguided": 0.2},
            "flagged_adversarial": True,
        }
    )
    assert mod._insert_before_tasks(["  title: no tasks"], "  finding: x") == [
        "  title: no tasks",
        "  finding: x",
    ]

    close_state = mod.build_v397_close_state(
        {
            "4301": _capstone(),
            "4291": _cross_generator(),
            "4292": _partial_state(),
            "4293": _generation(),
            "4295": _self_learning(),
            "4296": _arc_progress(),
            "4299": {"honest_verdict": "blocked_v397_artifacts_missing"},
            "4300": {"honest_verdict": "complete: hardware continuity"},
        },
        efficiency_present=False,
    )
    deduped, removed, action = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=3), close_state
    )
    assert (removed, action) == (2, "deduped")
    assert mod.archive_record_count(deduped) == 1
    assert "cross-generator CLOSED" in deduped
    assert "efficiency UNRESOLVED" in deduped
    assert mod.yaml_parses(deduped)

    updated, removed2, action2 = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=1), close_state
    )
    assert (removed2, action2) == (0, "updated")
    unchanged, removed3, action3 = mod.dedupe_or_update_record(updated, close_state)
    assert (unchanged, removed3, action3) == (updated, 0, "unchanged")
    appended, removed4, action4 = mod.dedupe_or_update_record(
        "# history\nmilestones:\n- id: 2026.06.396\n  finding: prior\n", close_state
    )
    assert (removed4, action4) == (0, "appended")
    assert "activation_recorded: exp4302-archive-v397-activate-v398" in appended
    no_finding, _removed5, action5 = mod.dedupe_or_update_record(
        "milestones:\n- id: 2026.06.397\n  title: no finding\n  tasks:\n  - id: exp4301\n",
        close_state,
    )
    assert action5 == "updated"
    assert "  finding: '.397 close-state" in no_finding


def test_read_sources_and_build_v397_close_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4302: close-state records the true .397 scorecard."""

    root = make_repo(tmp_path)
    sources = mod.read_v397_sources(root)
    cited = mod.build_cited_upstream(root)
    assert sources["4301"]["honest_verdict"] == "blocked_v397_artifacts_missing"
    assert {item["experiment_id"] for item in cited if item["kind"] == "artifact"} == {
        "4301",
        "4291",
        "4292",
        "4293",
        "4294",
        "4295",
        "4296",
        "4299",
        "4300",
    }
    assert all(item["sha256"] is None or mod.is_sha256(item["sha256"]) for item in cited)

    state = mod.build_v397_close_state(sources, efficiency_present=False)
    assert (
        state["summary"]
        == "cross_generator_closed_in_generation_degenerate_efficiency_unresolved_arc22"
    )
    assert state["capstone_blocked_spuriously"] is True
    assert state["cross_generator_axis_state"] == "CLOSED"
    assert state["cross_generator_closed"] is True
    assert state["cross_generator_delta"] == 0.5
    assert state["cross_generator_ci95"] == [0.292, 0.708]
    assert state["cross_generator_vote_at_1"] == 0.25
    assert state["cross_generator_oracle_at_k"] == 0.75
    assert state["cross_generator_non_degenerate"] is True
    assert state["cross_generator_verifier_is_oracle"] is False
    assert state["partial_state_scorer_built"] is True
    assert state["partial_state_leak_free"] is True
    assert state["partial_state_auroc"] == 0.966
    assert state["partial_state_yellow_flag"] is True
    assert state["in_generation_axis_state"] == "NOT_HELD_DEGENERATE_CONTROLS"
    assert state["in_generation_moat_holds"] is False
    assert state["in_generation_quarantined"] is True
    assert state["condition_accuracy"]["rfg"] == 0.3
    assert state["efficiency_axis_state"] == "UNRESOLVED_TASK_FAILED_NOT_NULL"
    assert state["efficiency_unresolved"] is True
    assert state["efficiency_artifact_missing"] is True
    assert state["efficiency_task_failed_not_null"] is True
    assert state["self_learning_online_helps"] is True
    assert state["online_cross_family_delta"] == 0.483
    assert state["static_cross_family_delta"] == 0.417
    assert state["online_minus_static_delta"] == 0.067
    assert state["arc_total_levels_solved"] == 22
    assert state["paper_ready"] is True
    assert state["v398_frame"] == mod.V398_FRAME


def test_run_happy_path_writes_valid_artifact_and_updates_archive(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4302: complete path writes the terminal archive artifact."""

    root = make_repo(tmp_path, duplicates=2)
    artifact = run_happy(root)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["archived_milestone"] == "2026.06.397"
    assert artifact["activated_milestone"] == "2026.06.398"
    assert artifact["active_milestone_confirmed"] == "2026.06.398"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["research_complete_record_action"] == "deduped"
    assert artifact["research_complete_duplicates_removed"] == 1
    assert artifact["v397_close_state"]["cross_generator_closed"] is True
    assert artifact["v397_close_state"]["in_generation_moat_holds"] is False
    assert artifact["v397_close_state"]["efficiency_unresolved"] is True
    assert (
        artifact["field_principles"]["v397_close_state"] == mod.FIELD_PRINCIPLES["v397_close_state"]
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.is_sha256(artifact["reproducibility_checksum"])
    complete_text = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(complete_text) == 1
    assert "cross-generator CLOSED" in complete_text
    assert "in-generation NOT held" in complete_text
    mod.validate_artifact(artifact)


def test_run_blocked_preconditions_before_research_complete_edit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4302-BLOCKED-PRECONDITION: blocked paths do not fabricate."""

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
    (root4 / "research-roadmap.yaml").write_text("milestone: 2026.06.397\n", encoding="utf-8")
    artifact4 = json.loads(mod.run(root4, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact4["honest_verdict"] == "blocked_v398_not_active"

    root5 = make_repo(tmp_path / "source_missing")
    (root5 / "results" / "experiment_4291_arcgen_cross_generator_nondegenerate.json").unlink()
    artifact5 = json.loads(mod.run(root5, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact5["honest_verdict"] == "blocked_cross_generator_artifact_missing"


def test_run_blocked_research_complete_edit_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4302: invalid archive edits are blocked before completion."""

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
    """REQ-REPORT-4302: schema validation and entrypoints preserve the contract."""

    root = make_repo(tmp_path)
    state = mod.build_v397_close_state(mod.read_v397_sources(root), efficiency_present=False)
    complete = mod.build_complete_artifact(
        v397_close_state=state,
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

    import carnot.experiment_4302_archive_v397_activate_v398 as entrypoint

    called: dict[str, Path] = {}

    def fake_run(root_path: Path = mod.REPO_ROOT) -> Path:
        called["root"] = Path(root_path)
        return root / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(entrypoint, "run", fake_run)
    assert entrypoint.main() == 0
    assert called["root"] == entrypoint.REPO_ROOT

    script_path = Path("results/experiment_4302_archive_v397_activate_v398.py")
    spec = importlib.util.spec_from_file_location("exp4302_archive_script", script_path)
    assert spec and spec.loader
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "run", fake_run)
    assert script.main() == 0


def test_validate_artifact_rejection_guards(tmp_path: Path) -> None:
    """REQ-REPORT-4302: validation rejects artifacts that launder the `.397` truth."""

    good = run_happy(make_repo(tmp_path))

    def set_path(obj: dict, path: list[str], value: object) -> None:
        cur = obj
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = value

    cases = [
        ("missing required fields", lambda a: a.pop("v397_close_state")),
        ("terminal-prefixed", lambda a: a.__setitem__("honest_verdict", "done")),
        (
            "principle must match REQ-REPORT-4302",
            lambda a: a["field_principles"].__setitem__("v397_close_state", "wrong"),
        ),
        ("archived milestone", lambda a: a.__setitem__("archived_milestone", "2026.06.999")),
        ("activated milestone", lambda a: a.__setitem__("activated_milestone", "2026.06.999")),
        ("research-complete YAML", lambda a: a.__setitem__("research_complete_yaml_parses", False)),
        ("exclusion manifest", lambda a: a.__setitem__("exclusion_manifest_parses", False)),
        ("pretest suite", lambda a: a.__setitem__("pretest_suite_green", False)),
        ("active milestone", lambda a: a.__setitem__("active_milestone_confirmed", "2026.06.397")),
        ("v397_close_state must be a mapping", lambda a: a.__setitem__("v397_close_state", "x")),
        (
            "cross-generator closed",
            lambda a: set_path(a, ["v397_close_state", "cross_generator_closed"], False),
        ),
        (
            "cross-generator axis state",
            lambda a: set_path(a, ["v397_close_state", "cross_generator_axis_state"], "OPEN"),
        ),
        (
            "cross-generator oracle distinct",
            lambda a: set_path(a, ["v397_close_state", "cross_generator_verifier_is_oracle"], True),
        ),
        (
            "partial-state scorer",
            lambda a: set_path(a, ["v397_close_state", "partial_state_scorer_built"], False),
        ),
        (
            "partial-state leak-free",
            lambda a: set_path(a, ["v397_close_state", "partial_state_leak_free"], False),
        ),
        (
            "in-generation not held",
            lambda a: set_path(a, ["v397_close_state", "in_generation_moat_holds"], True),
        ),
        (
            "degenerate controls",
            lambda a: set_path(a, ["v397_close_state", "in_generation_quarantined"], False),
        ),
        (
            "efficiency unresolved",
            lambda a: set_path(a, ["v397_close_state", "efficiency_unresolved"], False),
        ),
        (
            "task-failed-not-null",
            lambda a: set_path(a, ["v397_close_state", "efficiency_task_failed_not_null"], False),
        ),
        (
            "self-learning online",
            lambda a: set_path(a, ["v397_close_state", "self_learning_online_helps"], False),
        ),
        ("ARC 22", lambda a: set_path(a, ["v397_close_state", "arc_total_levels_solved"], 21)),
        ("paper", lambda a: set_path(a, ["v397_close_state", "paper_ready"], False)),
        ("v398 frame", lambda a: set_path(a, ["v397_close_state", "v398_frame"], "redo")),
    ]
    for label, mutate in cases:
        artifact = copy.deepcopy(good)
        mutate(artifact)
        with pytest.raises(ValueError, match=label):
            mod.validate_artifact(artifact)
