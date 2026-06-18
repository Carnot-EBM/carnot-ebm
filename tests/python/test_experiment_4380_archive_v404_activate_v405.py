"""Tests for Exp 4380 `.404` archive / `.405` activation.

Spec refs: REQ-REPORT-4380, SCENARIO-REPORT-4380,
SCENARIO-REPORT-4380-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import archive_v404_activate_v405_4380 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="81 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.403\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.404\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-18'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4379-capstone-v404\n"
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
        "updated: '2026-06-18'\n"
        "reproducible_total_levels: 34\n"
        "reproducible_total_games: 17\n"
    )


def _capstone(**overrides: object) -> dict:
    payload = {
        "honest_verdict": (
            "complete: v404_efficiency_linear_is_settled_arc_levels_34_"
            "s3_retired_detector_positive_publication_ready"
        ),
        "efficiency_moat_state": "linear_is_settled",
        "s3_moat_utility": "retired",
        "detector_beats_chance": True,
        "reproducible_total_levels": 34,
        "verifier_thesis_state": "linear_settled_in_generation_retired_detector_positive",
        "paper_ready": True,
        "verifier_is_oracle": False,
        "verifier_is_oracle_honored": True,
        "arc_reproducible_progress": {
            "prior_reproducible_total_levels": 33,
            "prior_reproducible_total_games": 17,
            "reproducible_total_levels": 34,
            "reproducible_total_games": 17,
            "new_levels_since_prior": 1,
            "new_games_since_prior": 0,
            "status": "loaded",
            "path": "ops/arc_solve_registry.yaml",
        },
        "efficiency_moat": {
            "efficiency_moat_state": "linear_is_settled",
            "llm_heuristic_beats_linear": False,
            "held_out_actions_by_heuristic": {
                "bfs_baseline": 646,
                "linear": 646,
                "llm_generated": 646,
            },
            "status": "clean_powered_null",
            "static_leakage_clean": True,
            "reproduction_gated": True,
            "verifier_is_oracle": False,
        },
        "detector": {
            "detector_auroc": 0.918304,
            "detector_auroc_ci95": [0.909296, 0.926923],
            "detector_beats_chance": True,
            "n_candidates": 8829,
            "selection_headroom": {
                "headroom": 0.0,
                "oracle_at_k": 0.812097,
                "vote_at_1": 0.812097,
            },
            "status": "detector_positive",
            "verifier_is_oracle": False,
        },
        "publication_gate": {"paper_ready": True, "unmet_gates": []},
    }
    payload.update(overrides)
    return payload


def _efficiency_null(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: clean_powered_null_linear_not_beaten",
        "llm_heuristic_beats_linear": False,
        "held_out_actions_by_heuristic": {
            "bfs_baseline": 646,
            "linear": 646,
            "llm_generated": 646,
        },
        "static_leakage_clean": True,
        "reproduction_gated": True,
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _deployed_efficiency(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "success: action_efficiency_compounds_25_to_16",
        "action_efficiency_compounds": True,
        "deployed_into_solver_kit": True,
        "positive_control_passed": True,
        "reproduction_gated": True,
        "verifier_is_oracle": False,
        "compounding_curve": [
            {"corpus_size_k": 4, "held_out_actions_to_solve": 25},
            {"corpus_size_k": 19, "held_out_actions_to_solve": 16},
        ],
    }
    payload.update(overrides)
    return payload


def _diffusiongemma(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "retired_in_generation_conversion_unmeasurable",
        "s3_guided_beats_control": False,
        "scorer_requalified_leak_clean": False,
        "codila_control_differentiates": False,
        "controls_differentiated": False,
        "benchmark_n": 0,
        "retirement_gate": {
            "retired": True,
            "reason": "scorer_leaky_and_codila_not_differentiating",
        },
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _detector(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: detector_beats_chance_zero_selection_headroom_fover",
        "detector_auroc": 0.918304,
        "detector_auroc_ci95": [0.909296, 0.926923],
        "detector_beats_chance": True,
        "n_candidates": 8829,
        "selection_headroom": {
            "headroom": 0.0,
            "oracle_at_k": 0.812097,
            "vote_at_1": 0.812097,
        },
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _sota(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: sota_ingestion_v405_mapped",
        "flagged_for_v405": "biprm_processbench_detector_localization_v405",
        "methods_mapped": [
            {
                "arxiv_id_or_url": "2508.01682",
                "id": "biprm_processbench_detector_localization_v405",
                "name": "Bidirectional Process Reward Model for step-error detection",
            },
            {
                "arxiv_id_or_url": "2606.16070",
                "id": "mind_studio_e3_lookahead_fidelity_v405",
                "name": "Mind-Studio lookahead fidelity",
            },
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
    (root / "ops" / "exclusion_manifest.yaml").write_text(_manifest_text(), encoding="utf-8")
    (root / "ops" / "arc_solve_registry.yaml").write_text(_registry_text(), encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(
        "milestone: 2026.06.405\n"
        "milestone_doc: openspec/change-proposals/research-roadmap-v405.md\n"
        "milestone_overview: detector localization + abstention + ARC deeper\n",
        encoding="utf-8",
    )
    (root / "openspec" / "change-proposals").mkdir(parents=True, exist_ok=True)
    (root / "openspec" / "change-proposals" / "research-roadmap-v405.md").write_text(
        "# Research Roadmap v405\n\n"
        "DEEPEN the detector into actionable localization and abstention; ARC-deeper.\n",
        encoding="utf-8",
    )
    _write_json(root / "results" / "experiment_4379_capstone_v404.json", _capstone())
    _write_json(
        root / "results" / "experiment_4370_llm_generated_action_cost_heuristics.json",
        _efficiency_null(),
    )
    _write_json(
        root / "results" / "experiment_4364_self_learning_action_cost_compounds.json",
        _deployed_efficiency(),
    )
    _write_json(
        root / "results" / "experiment_4374_diffusiongemma_scorer_repair_or_retire.json",
        _diffusiongemma(),
    )
    _write_json(
        root / "results" / "experiment_4375_verifier_as_detector_measurement.json",
        _detector(),
    )
    _write_json(root / "results" / "experiment_4376_sota_ingestion_v405.json", _sota())
    return root


def run_happy(root: Path) -> dict:
    out = mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5)
    return json.loads(out.read_text(encoding="utf-8"))


def test_req_report_4380_spec_declares_contract() -> None:
    """REQ-REPORT-4380: OpenSpec declares the true `.404` scorecard contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    assert "REQ-REPORT-4380" in spec
    assert "SCENARIO-REPORT-4380" in spec
    assert "SCENARIO-REPORT-4380-BLOCKED-PRECONDITION" in spec
    assert "linear_is_settled" in spec
    assert "retired_in_generation_conversion_unmeasurable" in spec
    assert "biprm_processbench_detector_localization_v405" in spec
    assert mod.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert mod.FIELD_PRINCIPLES["v404_close_state"] in spec
    assert mod.FIELD_PRINCIPLES["preconditions_checked"] in spec


def test_helpers_and_archive_record_editing(tmp_path: Path) -> None:
    """REQ-REPORT-4380: helper behavior is deterministic and YAML-safe."""

    assert mod.archive_record_count(_research_complete_text(duplicates=3)) == 3
    assert mod._record_id("- id: '2026.06.404'") == "2026.06.404"
    assert mod._record_id("  - id: nested") is None
    assert mod._yaml_quote("don't reopen") == "'don''t reopen'"
    assert mod._insert_before_tasks(["  title: no tasks"], "  finding: x") == [
        "  title: no tasks",
        "  finding: x",
    ]
    assert mod._curve_actions(
        {"held_out_actions_first": 30, "held_out_actions_last": 18},
        {"held_out_actions_first": 25, "held_out_actions_last": 16},
    ) == (30, 18)
    assert mod._ci95(None, [0.1, 0.2]) == [0.1, 0.2]

    root = make_repo(tmp_path)
    close_state = mod.build_v404_close_state(mod.read_v404_sources(root))
    deduped, removed, action = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=3), close_state
    )
    assert (removed, action) == (2, "deduped")
    assert mod.archive_record_count(deduped) == 1
    assert "EFFICIENCY moat SETTLED" in deduped
    assert "DETECTION is the one ALIVE" in deduped
    assert "ARC 34 reproducible levels / 17 games" in deduped
    assert "biprm_processbench_detector_localization_v405" in deduped
    assert mod.yaml_parses(deduped)

    updated, removed2, action2 = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=1), close_state
    )
    assert (removed2, action2) == (0, "updated")
    unchanged, removed3, action3 = mod.dedupe_or_update_record(updated, close_state)
    assert (unchanged, removed3, action3) == (updated, 0, "unchanged")
    appended, removed4, action4 = mod.dedupe_or_update_record(
        "# history\nmilestones:\n- id: 2026.06.403\n  finding: prior\n", close_state
    )
    assert (removed4, action4) == (0, "appended")
    assert "activation_recorded: exp4380-archive-v404-activate-v405" in appended
    no_finding, _removed5, action5 = mod.dedupe_or_update_record(
        "milestones:\n- id: 2026.06.404\n  title: no finding\n  tasks:\n  - id: exp4379\n",
        close_state,
    )
    assert action5 == "updated"
    assert "  finding: '.404 close-state" in no_finding


def test_read_sources_and_build_v404_close_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4380: close-state records the true `.404` scorecard."""

    root = make_repo(tmp_path)
    sources = mod.read_v404_sources(root)
    cited = mod.build_cited_upstream(root)
    assert sources["4379"]["verifier_thesis_state"] == (
        "linear_settled_in_generation_retired_detector_positive"
    )
    assert {item["experiment_id"] for item in cited if item["kind"] == "artifact"} == {
        "4379",
        "4370",
        "4364",
        "4374",
        "4375",
        "4376",
    }
    assert all(item["sha256"] is None or mod.is_sha256(item["sha256"]) for item in cited)

    state = mod.build_v404_close_state(sources)
    assert state["summary"] == "efficiency_settled_s3_retired_detector_alive_arc34_v405_actionable"
    assert state["verifier_thesis_state"] == (
        "linear_settled_in_generation_retired_detector_positive"
    )
    assert state["efficiency_moat_axis_state"] == "SETTLED_LINEAR_IS_NEAR_OPTIMAL"
    assert state["llm_heuristic_beats_linear"] is False
    assert state["deployed_efficiency_moat_real"] is True
    assert state["held_out_actions_first"] == 25
    assert state["held_out_actions_last"] == 16
    assert state["efficiency_verifier_is_oracle"] is False
    assert state["s3_conversion_axis_state"] == "RETIRED_FOURTH_BLOCK"
    assert state["s3_moat_utility"] == "retired"
    assert state["scorer_requalified_leak_clean"] is False
    assert state["codila_control_differentiates"] is False
    assert state["benchmark_n"] == 0
    assert state["detector_axis_state"] == "ALIVE_ORACLE_DISTINCT_VEHICLE"
    assert state["detector_auroc"] == 0.918304
    assert state["detector_auroc_ci95_lower"] == 0.909296
    assert state["detector_beats_chance"] is True
    assert state["selection_headroom"] == 0.0
    assert state["detector_n_candidates"] == 8829
    assert state["detector_verifier_is_oracle"] is False
    assert state["arc_prior_reproducible_total_levels"] == 33
    assert state["arc_reproducible_total_levels"] == 34
    assert state["arc_reproducible_total_games"] == 17
    assert state["arc_new_levels_since_prior"] == 1
    assert state["flagged_for_v405"] == "biprm_processbench_detector_localization_v405"
    assert state["cross_game_value_transfer_axis_state"] == "RETIRED_EXP4342_THIRD_NULL"
    assert state["cross_domain_axis_state"] == "RETIRED_EXP4314_DOMAIN_BOUND"
    assert state["paper_ready"] is True
    assert state["v405_frame"] == mod.V405_FRAME


def test_run_happy_path_writes_valid_artifact_and_updates_archive(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4380: complete path writes the terminal archive artifact."""

    root = make_repo(tmp_path, duplicates=2)
    artifact = run_happy(root)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["archived_milestone"] == "2026.06.404"
    assert artifact["activated_milestone"] == "2026.06.405"
    assert artifact["active_milestone_confirmed"] == "2026.06.405"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["research_complete_record_action"] == "deduped"
    assert artifact["research_complete_duplicates_removed"] == 1
    assert artifact["v404_close_state"]["efficiency_moat_axis_state"] == (
        "SETTLED_LINEAR_IS_NEAR_OPTIMAL"
    )
    assert artifact["v404_close_state"]["s3_conversion_axis_state"] == "RETIRED_FOURTH_BLOCK"
    assert artifact["v404_close_state"]["detector_axis_state"] == (
        "ALIVE_ORACLE_DISTINCT_VEHICLE"
    )
    assert artifact["v404_close_state"]["arc_reproducible_total_levels"] == 34
    assert artifact["v404_close_state"]["arc_reproducible_total_games"] == 17
    assert artifact["v404_close_state"]["flagged_for_v405"] == (
        "biprm_processbench_detector_localization_v405"
    )
    assert (
        artifact["field_principles"]["v404_close_state"] == mod.FIELD_PRINCIPLES["v404_close_state"]
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.is_sha256(artifact["reproducibility_checksum"])
    complete_text = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(complete_text) == 1
    assert "DETECTION is the one ALIVE" in complete_text
    assert "actionable localization+abstention" in complete_text
    mod.validate_artifact(artifact)


def test_run_blocked_preconditions_before_research_complete_edit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4380-BLOCKED-PRECONDITION: blocked paths do not fabricate."""

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
    (root4 / "research-roadmap.yaml").write_text("milestone: 2026.06.404\n", encoding="utf-8")
    artifact4 = json.loads(mod.run(root4, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact4["honest_verdict"] == "blocked_v405_not_active"

    root5 = make_repo(tmp_path / "source_missing")
    (root5 / "results" / "experiment_4379_capstone_v404.json").unlink()
    artifact5 = json.loads(mod.run(root5, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact5["honest_verdict"] == "blocked_v404_capstone_missing"

    root6 = make_repo(tmp_path / "sota_missing")
    (root6 / "results" / "experiment_4376_sota_ingestion_v405.json").unlink()
    artifact6 = json.loads(mod.run(root6, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact6["honest_verdict"] == "blocked_sota_ingestion_v405_missing"


def test_run_blocked_research_complete_edit_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4380: invalid archive edits are blocked before completion."""

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
    """REQ-REPORT-4380: schema validation and entrypoints preserve the contract."""

    root = make_repo(tmp_path)
    state = mod.build_v404_close_state(mod.read_v404_sources(root))
    complete = mod.build_complete_artifact(
        v404_close_state=state,
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

    import carnot.experiment_4380_archive_v404_activate_v405 as entrypoint

    called: dict[str, Path] = {}

    def fake_run(root_path: Path = mod.REPO_ROOT) -> Path:
        called["root"] = Path(root_path)
        return root / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(entrypoint, "run", fake_run)
    assert entrypoint.main() == 0
    assert called["root"] == entrypoint.REPO_ROOT

    script_path = Path("results/experiment_4380_archive_v404_activate_v405.py")
    script_repo_root = str(script_path.resolve().parents[1])
    script_python_root = str(Path(script_repo_root) / "python")
    original_sys_path = list(sys.path)
    try:
        sys.path[:] = [
            item for item in sys.path if item not in {script_repo_root, script_python_root}
        ]
        spec = importlib.util.spec_from_file_location("exp4380_archive_script", script_path)
        assert spec and spec.loader
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
    finally:
        sys.path[:] = original_sys_path
    monkeypatch.setattr(script, "run", fake_run)
    assert script.main() == 0


def test_validate_artifact_rejection_guards(tmp_path: Path) -> None:
    """REQ-REPORT-4380: validation rejects artifacts that launder the `.404` truth."""

    good = run_happy(make_repo(tmp_path))

    def set_path(obj: dict, path: list[str], value: object) -> None:
        cur = obj
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = value

    cases = [
        ("missing required fields", lambda a: a.pop("v404_close_state")),
        ("terminal-prefixed", lambda a: a.__setitem__("honest_verdict", "done")),
        (
            "principle must match REQ-REPORT-4380",
            lambda a: a["field_principles"].__setitem__("v404_close_state", "wrong"),
        ),
        ("archived milestone", lambda a: a.__setitem__("archived_milestone", "2026.06.999")),
        ("activated milestone", lambda a: a.__setitem__("activated_milestone", "2026.06.999")),
        ("research-complete YAML", lambda a: a.__setitem__("research_complete_yaml_parses", False)),
        ("exclusion manifest", lambda a: a.__setitem__("exclusion_manifest_parses", False)),
        ("pretest suite", lambda a: a.__setitem__("pretest_suite_green", False)),
        ("active milestone", lambda a: a.__setitem__("active_milestone_confirmed", "2026.06.404")),
        ("v404_close_state must be a mapping", lambda a: a.__setitem__("v404_close_state", "x")),
        (
            "efficiency settled",
            lambda a: set_path(a, ["v404_close_state", "efficiency_moat_axis_state"], "OPEN"),
        ),
        (
            "llm heuristic",
            lambda a: set_path(a, ["v404_close_state", "llm_heuristic_beats_linear"], True),
        ),
        (
            "efficiency real",
            lambda a: set_path(a, ["v404_close_state", "deployed_efficiency_moat_real"], False),
        ),
        (
            "S3 retired",
            lambda a: set_path(a, ["v404_close_state", "s3_conversion_axis_state"], "OPEN"),
        ),
        ("S3 utility", lambda a: set_path(a, ["v404_close_state", "s3_moat_utility"], "open")),
        (
            "fourth block",
            lambda a: set_path(a, ["v404_close_state", "fourth_consecutive_block"], False),
        ),
        (
            "detector alive",
            lambda a: set_path(a, ["v404_close_state", "detector_axis_state"], "OPEN"),
        ),
        (
            "detector AUROC",
            lambda a: set_path(a, ["v404_close_state", "detector_auroc"], 0.5),
        ),
        (
            "selection headroom",
            lambda a: set_path(a, ["v404_close_state", "selection_headroom"], 0.1),
        ),
        (
            "ARC 34",
            lambda a: set_path(a, ["v404_close_state", "arc_reproducible_total_levels"], 33),
        ),
        (
            "ARC games",
            lambda a: set_path(a, ["v404_close_state", "arc_reproducible_total_games"], 16),
        ),
        (
            "flagged_for_v405",
            lambda a: set_path(a, ["v404_close_state", "flagged_for_v405"], "other"),
        ),
        (
            "cross-game retired",
            lambda a: set_path(
                a, ["v404_close_state", "cross_game_value_transfer_axis_state"], "OPEN"
            ),
        ),
        (
            "cross-domain retired",
            lambda a: set_path(a, ["v404_close_state", "cross_domain_axis_state"], "OPEN"),
        ),
        ("paper", lambda a: set_path(a, ["v404_close_state", "paper_ready"], False)),
        ("v405 frame", lambda a: set_path(a, ["v404_close_state", "v405_frame"], "redo")),
    ]
    for label, mutate in cases:
        artifact = copy.deepcopy(good)
        mutate(artifact)
        with pytest.raises(ValueError, match=label):
            mod.validate_artifact(artifact)
