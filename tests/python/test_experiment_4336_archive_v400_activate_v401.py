"""Tests for Exp 4336 `.400` archive / `.401` activation.

Spec refs: REQ-REPORT-4336, SCENARIO-REPORT-4336,
SCENARIO-REPORT-4336-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v400_activate_v401_4336 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="81 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.399\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.400\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-17'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4335-capstone-v400\n"
        "    result: OK (conductor)\n"
    )
    return head + block * duplicates


def _capstone(**overrides: object) -> dict:
    payload = {
        "honest_verdict": (
            "complete: v400_in_generation_corpus_specific_gate_"
            "STILL_PENDING_second_corpus_scorer_leaky_arc_levels_13_e3_reproduced_0"
        ),
        "headline_outcome": (
            "in_generation_corpus_specific__adaptive_open_reasoning_corpus_fallback__"
            "arc_levels_13_e3_0__self_learning_open__paper_ready"
        ),
        "verifier_thesis_state": "in_generation_moat_corpus_specific",
        "diffusiongemma_gate_status": "STILL_PENDING_second_corpus_scorer_leaky",
        "in_generation_moat_replicates_headline": False,
        "arc_reproducible_total_levels": 13,
        "paper_ready": True,
        "verifier_is_oracle_honored": True,
        "in_generation_replication": {
            "status": "corpus_specific",
            "honest_verdict": "scorer_leaky_on_second_corpus",
            "in_generation_moat_replicates_headline": False,
            "scorer_leak_recheck_passed": False,
            "controls_differentiated": False,
            "replication_ci95": [0.0, 0.0],
            "replication_ci95_excludes_zero": False,
            "carnot_minus_best_control_delta": 0.0,
            "carnot_minus_self_reward_smc_delta": 0.0,
            "verifier_is_oracle": False,
        },
        "adaptive_scaleup": {
            "status": "open",
            "honest_verdict": "complete: adaptive_guidance_bounded_to_stitching_null",
            "adaptive_guidance_beats_control": False,
            "adaptive_ci95": [-0.075, 0.35],
            "adaptive_ci95_excludes_zero": False,
            "domain_used": "reasoning_corpus_fallback",
            "verifier_is_oracle": False,
        },
        "e3_deep_tail": {
            "status": "partial",
            "reproduced_levels_total": 0,
            "execution_grounded": True,
            "games": {
                "ar25": {
                    "game": "ar25",
                    "status": "partial",
                    "honest_verdict": "complete_e3_ar25_partial_model_0.89",
                    "offline_reproduced": False,
                    "reproduced_levels": 0,
                    "verifier_best_accuracy": 0.8875,
                    "verifier_accuracy_per_round": [0.8875],
                    "verifier_is_oracle": True,
                },
                "ka59": {
                    "game": "ka59",
                    "offline_reproduced": False,
                    "reproduced_levels": 0,
                    "verifier_best_accuracy": 0.5625,
                    "verifier_is_oracle": True,
                },
                "ft09": {
                    "game": "ft09",
                    "offline_reproduced": False,
                    "reproduced_levels": 0,
                    "verifier_best_accuracy": 0.1,
                    "verifier_is_oracle": True,
                },
                "tr87": {
                    "game": "tr87",
                    "offline_reproduced": False,
                    "reproduced_levels": 0,
                    "verifier_best_accuracy": 0.0,
                    "verifier_is_oracle": True,
                },
            },
        },
        "self_learning": {
            "status": "open",
            "honest_verdict": "complete: learned_frame_encoder_transfer_no_improvement",
            "learned_encoder_transfer_helps": False,
            "cross_game_state_reduction": 1.0084925690021231,
            "cross_game_state_reduction_ci95": [1.0, 1.0303068758652514],
            "verifier_is_oracle": False,
        },
        "arc_shallow": {
            "status": "included",
            "honest_verdict": "complete: adapter_free_shallow_tail_no_advance",
            "reproducible_total_levels": 13,
            "prior_reproducible_total_levels": 13,
            "offline_reproduced": False,
            "games_advanced": [],
            "verifier_is_oracle": True,
        },
    }
    payload.update(overrides)
    return payload


def _in_generation(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "scorer_leaky_on_second_corpus",
        "in_generation_moat_replicates": False,
        "scorer_leak_recheck_passed": False,
        "controls_differentiated": False,
        "replication_ci95": [0.0, 0.0],
        "carnot_minus_best_control_delta": 0.0,
        "carnot_minus_self_reward_smc_delta": 0.0,
        "verifier_is_oracle": False,
        "independent_leak_recheck": {
            "status": "measured",
            "answer_masked_auroc": 0.549719,
            "auroc_floor": 0.6,
            "scorer_leak_recheck_passed": False,
        },
    }
    payload.update(overrides)
    return payload


def _ar25(**overrides: object) -> dict:
    payload = {
        "game": "ar25",
        "honest_verdict": "complete_e3_ar25_partial_model_0.89",
        "offline_reproduced": False,
        "plan_executed": False,
        "reproduced_levels": 0,
        "verifier_best_accuracy": 0.8875,
        "verifier_accuracy_per_round": [0.8875],
        "verifier_is_oracle": True,
    }
    payload.update(overrides)
    return payload


def _arc_registry_text() -> str:
    return (
        "reproducible_total_levels: 13\n"
        "reproducible_total_games: 11\n"
        "provisional_total_levels: 5\n"
        "games:\n"
        "  - game: sc25\n"
        "    reproducibility: provisional\n"
        "    levels_reproduced: 0\n"
        "    levels_live_recorded: 5\n"
    )


def make_repo(tmp_path: Path, *, duplicates: int = 1) -> Path:
    root = tmp_path
    root.mkdir(parents=True, exist_ok=True)
    (root / "research-complete.yaml").write_text(
        _research_complete_text(duplicates=duplicates), encoding="utf-8"
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "exclusion_manifest.yaml").write_text(
        "retired:\n"
        "- id: cross_domain_selection_retired_exp4314_v399\n"
        "  operator_reopen_required: true\n"
        "  retire_if_same_verdict: true\n",
        encoding="utf-8",
    )
    (root / "ops" / "arc_solve_registry.yaml").write_text(
        _arc_registry_text(), encoding="utf-8"
    )
    (root / "research-roadmap.yaml").write_text(
        "milestone: 2026.06.401\n"
        "milestone_doc: openspec/change-proposals/research-roadmap-v401.md\n"
        "milestone_overview: SETTLE the in-generation moat with leak-robust scorer\n",
        encoding="utf-8",
    )
    (root / "openspec" / "change-proposals").mkdir(parents=True, exist_ok=True)
    (root / "openspec" / "change-proposals" / "research-roadmap-v401.md").write_text(
        "# Research Roadmap v401\n\nCross-domain selection stays RETIRED.\n",
        encoding="utf-8",
    )
    _write_json(root / "results" / "experiment_4335_capstone_v400.json", _capstone())
    _write_json(
        root / "results" / "experiment_4325_in_generation_moat_replicate_second_corpus.json",
        _in_generation(),
    )
    _write_json(
        root / "results" / "experiment_4327_e3_executable_world_model_ar25.json",
        _ar25(),
    )
    return root


def run_happy(root: Path) -> dict:
    out = mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5)
    return json.loads(out.read_text(encoding="utf-8"))


def test_req_report_4336_spec_declares_contract() -> None:
    """REQ-REPORT-4336: OpenSpec declares the true `.400` scorecard contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    assert "REQ-REPORT-4336" in spec
    assert "SCENARIO-REPORT-4336" in spec
    assert "SCENARIO-REPORT-4336-BLOCKED-PRECONDITION" in spec
    assert "in-generation moat as CORPUS-SPECIFIC" in spec
    assert "STILL_PENDING_second_corpus_scorer_leaky" in spec
    assert "ar25 closest at about 0.89" in spec
    assert mod.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert mod.FIELD_PRINCIPLES["v400_close_state"] in spec
    assert mod.FIELD_PRINCIPLES["preconditions_checked"] in spec


def test_helpers_and_archive_record_editing(tmp_path: Path) -> None:
    """REQ-REPORT-4336: helper behavior is deterministic and YAML-safe."""

    assert mod.archive_record_count(_research_complete_text(duplicates=3)) == 3
    assert mod._record_id("- id: '2026.06.400'") == "2026.06.400"
    assert mod._record_id("  - id: nested") is None
    assert mod._yaml_quote("don't re-open") == "'don''t re-open'"
    assert mod._rounded_pair("bad", [1.0, 2.0]) == [1.0, 2.0]
    assert mod._ci_includes_zero([-0.1, 0.2])
    assert mod._insert_before_tasks(["  title: no tasks"], "  finding: x") == [
        "  title: no tasks",
        "  finding: x",
    ]
    assert mod._best_e3_game({}) == ("ar25", {})
    assert mod._sc25_provisional_levels({"games": [{"game": "sc25", "levels_live_recorded": 4}]}) == 4
    assert mod._sc25_provisional_levels({"games": []}) == 5

    root = make_repo(tmp_path)
    close_state = mod.build_v400_close_state(mod.read_v400_sources(root))
    deduped, removed, action = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=3), close_state
    )
    assert (removed, action) == (2, "deduped")
    assert mod.archive_record_count(deduped) == 1
    assert "in-generation moat CORPUS-SPECIFIC" in deduped
    assert "DiffusionGemma gate STILL_PENDING_second_corpus_scorer_leaky" in deduped
    assert "cross-domain selection remains RETIRED" in deduped
    assert mod.yaml_parses(deduped)

    updated, removed2, action2 = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=1), close_state
    )
    assert (removed2, action2) == (0, "updated")
    unchanged, removed3, action3 = mod.dedupe_or_update_record(updated, close_state)
    assert (unchanged, removed3, action3) == (updated, 0, "unchanged")
    appended, removed4, action4 = mod.dedupe_or_update_record(
        "# history\nmilestones:\n- id: 2026.06.399\n  finding: prior\n", close_state
    )
    assert (removed4, action4) == (0, "appended")
    assert "activation_recorded: exp4336-archive-v400-activate-v401" in appended
    no_finding, _removed5, action5 = mod.dedupe_or_update_record(
        "milestones:\n- id: 2026.06.400\n  title: no finding\n  tasks:\n  - id: exp4335\n",
        close_state,
    )
    assert action5 == "updated"
    assert "  finding: '.400 close-state" in no_finding


def test_read_sources_and_build_v400_close_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4336: close-state records the true .400 scorecard."""

    root = make_repo(tmp_path)
    sources = mod.read_v400_sources(root)
    cited = mod.build_cited_upstream(root)
    assert sources["4335"]["verifier_thesis_state"] == "in_generation_moat_corpus_specific"
    assert {item["experiment_id"] for item in cited if item["kind"] == "artifact"} == {
        "4335",
        "4325",
        "4327",
    }
    assert any(item["experiment_id"] == "v401_design_doc" for item in cited)
    assert all(item["sha256"] is None or mod.is_sha256(item["sha256"]) for item in cited)

    state = mod.build_v400_close_state(sources)
    assert state["summary"] == (
        "in_generation_corpus_specific_gate_pending_e3_0_ar25_close_"
        "self_learning_null_arc13_cross_domain_retired"
    )
    assert state["verifier_thesis_state"] == "in_generation_moat_corpus_specific"
    assert state["in_generation_axis_state"] == "CORPUS_SPECIFIC_SCORER_LEAKED"
    assert state["in_generation_moat_replicates"] is False
    assert state["in_generation_scorer_leak_recheck_passed"] is False
    assert state["diffusiongemma_gate_status"] == "STILL_PENDING_second_corpus_scorer_leaky"
    assert state["adaptive_axis_state"] == "BOUNDED_TO_POST_HOC_STITCHING"
    assert state["adaptive_guidance_beats_control"] is False
    assert state["adaptive_ci95"] == [-0.075, 0.35]
    assert state["adaptive_ci95_includes_zero"] is True
    assert state["adaptive_domain_used"] == "reasoning_corpus_fallback"
    assert state["e3_axis_state"] == "DEEP_TAIL_PARTIAL_0_SOLVES_AR25_CLOSE"
    assert state["e3_reproduced_levels_total"] == 0
    assert state["e3_closest_game"] == "ar25"
    assert state["e3_ar25_verifier_best_accuracy"] == 0.89
    assert state["e3_ar25_plan_executed"] is False
    assert state["self_learning_axis_state"] == "LEARNED_FRAME_ENCODER_TRANSFER_NULL"
    assert state["learned_encoder_transfer_helps"] is False
    assert state["cross_game_state_reduction"] == 1.008
    assert state["arc_reproducible_total_levels"] == 13
    assert state["arc_reproducible_total_games"] == 11
    assert state["sc25_provisional_live_recorded_levels"] == 5
    assert state["cross_domain_axis_state"] == "RETIRED_DOMAIN_BOUND"
    assert state["cross_domain_do_not_repropose"] is True
    assert state["paper_ready"] is True
    assert state["v401_frame"] == mod.V401_FRAME


def test_run_happy_path_writes_valid_artifact_and_updates_archive(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4336: complete path writes the terminal archive artifact."""

    root = make_repo(tmp_path, duplicates=2)
    artifact = run_happy(root)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["archived_milestone"] == "2026.06.400"
    assert artifact["activated_milestone"] == "2026.06.401"
    assert artifact["active_milestone_confirmed"] == "2026.06.401"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["research_complete_record_action"] == "deduped"
    assert artifact["research_complete_duplicates_removed"] == 1
    assert artifact["v400_close_state"]["in_generation_axis_state"] == (
        "CORPUS_SPECIFIC_SCORER_LEAKED"
    )
    assert artifact["v400_close_state"]["e3_reproduced_levels_total"] == 0
    assert artifact["v400_close_state"]["arc_reproducible_total_levels"] == 13
    assert artifact["field_principles"]["v400_close_state"] == mod.FIELD_PRINCIPLES[
        "v400_close_state"
    ]
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.is_sha256(artifact["reproducibility_checksum"])
    complete_text = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(complete_text) == 1
    assert "in-generation moat CORPUS-SPECIFIC" in complete_text
    assert "sc25 reproduction" in complete_text
    mod.validate_artifact(artifact)


def test_run_blocked_preconditions_before_research_complete_edit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4336-BLOCKED-PRECONDITION: blocked paths do not fabricate."""

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
    (root4 / "research-roadmap.yaml").write_text("milestone: 2026.06.400\n", encoding="utf-8")
    artifact4 = json.loads(mod.run(root4, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact4["honest_verdict"] == "blocked_v401_not_active"

    root5 = make_repo(tmp_path / "source_missing")
    (root5 / "results" / "experiment_4325_in_generation_moat_replicate_second_corpus.json").unlink()
    artifact5 = json.loads(mod.run(root5, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact5["honest_verdict"] == "blocked_in_generation_replication_artifact_missing"


def test_run_blocked_research_complete_edit_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4336: invalid archive edits are blocked before completion."""

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
    """REQ-REPORT-4336: schema validation and entrypoints preserve the contract."""

    root = make_repo(tmp_path)
    state = mod.build_v400_close_state(mod.read_v400_sources(root))
    complete = mod.build_complete_artifact(
        v400_close_state=state,
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

    import carnot.experiment_4336_archive_v400_activate_v401 as entrypoint

    called: dict[str, Path] = {}

    def fake_run(root_path: Path = mod.REPO_ROOT) -> Path:
        called["root"] = Path(root_path)
        return root / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(entrypoint, "run", fake_run)
    assert entrypoint.main() == 0
    assert called["root"] == entrypoint.REPO_ROOT

    script_path = Path("results/experiment_4336_archive_v400_activate_v401.py")
    spec = importlib.util.spec_from_file_location("exp4336_archive_script", script_path)
    assert spec and spec.loader
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "run", fake_run)
    assert script.main() == 0


def test_validate_artifact_rejection_guards(tmp_path: Path) -> None:
    """REQ-REPORT-4336: validation rejects artifacts that launder the `.400` truth."""

    good = run_happy(make_repo(tmp_path))

    def set_path(obj: dict, path: list[str], value: object) -> None:
        cur = obj
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = value

    cases = [
        ("missing required fields", lambda a: a.pop("v400_close_state")),
        ("terminal-prefixed", lambda a: a.__setitem__("honest_verdict", "done")),
        (
            "principle must match REQ-REPORT-4336",
            lambda a: a["field_principles"].__setitem__("v400_close_state", "wrong"),
        ),
        ("archived milestone", lambda a: a.__setitem__("archived_milestone", "2026.06.999")),
        ("activated milestone", lambda a: a.__setitem__("activated_milestone", "2026.06.999")),
        ("research-complete YAML", lambda a: a.__setitem__("research_complete_yaml_parses", False)),
        ("exclusion manifest", lambda a: a.__setitem__("exclusion_manifest_parses", False)),
        ("pretest suite", lambda a: a.__setitem__("pretest_suite_green", False)),
        ("active milestone", lambda a: a.__setitem__("active_milestone_confirmed", "2026.06.400")),
        ("v400_close_state must be a mapping", lambda a: a.__setitem__("v400_close_state", "x")),
        (
            "in-generation corpus-specific",
            lambda a: set_path(a, ["v400_close_state", "in_generation_axis_state"], "CLOSED"),
        ),
        (
            "scorer leaked",
            lambda a: set_path(a, ["v400_close_state", "in_generation_scorer_leak_recheck_passed"], True),
        ),
        (
            "gate still pending",
            lambda a: set_path(a, ["v400_close_state", "diffusiongemma_gate_status"], "MET"),
        ),
        (
            "adaptive bounded",
            lambda a: set_path(a, ["v400_close_state", "adaptive_guidance_beats_control"], True),
        ),
        (
            "E3 zero solves",
            lambda a: set_path(a, ["v400_close_state", "e3_reproduced_levels_total"], 1),
        ),
        (
            "ar25 closest",
            lambda a: set_path(a, ["v400_close_state", "e3_closest_game"], "ka59"),
        ),
        (
            "self-learning null",
            lambda a: set_path(a, ["v400_close_state", "learned_encoder_transfer_helps"], True),
        ),
        (
            "ARC 13",
            lambda a: set_path(a, ["v400_close_state", "arc_reproducible_total_levels"], 14),
        ),
        (
            "ARC 11 games",
            lambda a: set_path(a, ["v400_close_state", "arc_reproducible_total_games"], 10),
        ),
        (
            "sc25 provisional",
            lambda a: set_path(a, ["v400_close_state", "sc25_provisional_live_recorded_levels"], 0),
        ),
        (
            "cross-domain retired",
            lambda a: set_path(a, ["v400_close_state", "cross_domain_axis_state"], "OPEN"),
        ),
        ("paper", lambda a: set_path(a, ["v400_close_state", "paper_ready"], False)),
        ("v401 frame", lambda a: set_path(a, ["v400_close_state", "v401_frame"], "redo")),
    ]
    for label, mutate in cases:
        artifact = copy.deepcopy(good)
        mutate(artifact)
        with pytest.raises(ValueError, match=label):
            mod.validate_artifact(artifact)
