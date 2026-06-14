"""Tests for Exp 4207 `.389` archive / `.390` activation.

Spec refs: REQ-REPORT-4207, SCENARIO-REPORT-4207,
SCENARIO-REPORT-4207-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v389_activate_v390_4207 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="81 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.388\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.389\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-14'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4206-capstone-v389\n"
        "    result: OK\n"
    )
    return head + block * duplicates


def _capstone(**overrides: object) -> dict:
    payload = {
        "experiment_id": 4206,
        "honest_verdict": (
            "complete: capstone_v389_verifier_reward_no_code_operating_point_"
            "status_NO-OPERATING-POINT_arc_levels15_flagged_skipped4"
        ),
        "headline_outcome": "verifier_reward_no_code_operating_point",
        "verifier_as_reward_status": "NO-OPERATING-POINT",
        "a_vs_b_training_signal": {
            "status": "blocked_a_vs_b_not_collected",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: "
                "exp4198-verifier-reward-3arm-rft-launch.training_launched "
                "(actual=False == expected=True)"
            ),
            "verifier_label_carries_signal": False,
            "positive_control_confirmed": False,
            "controls_confirmed": False,
        },
        "arc_progress": {
            "status": "included",
            "acceptance_gate_passed": True,
            "total_arc_levels_solved": 15,
            "total_arc_games_solved": 13,
            "new_levels_solved_this_task": 0,
        },
        "live_solver_vs_floor": {
            "status": "included",
            "solver_beats_floor_overall": True,
            "solver_beats_floor_accuracy": False,
            "solver_beats_floor_efficiency": True,
            "live_env_reachable": True,
            "live_env_metrics": {
                "levels_completed": 0,
                "actions_taken": 5,
                "environment": {"game_id": "lp85-305b61c3"},
            },
            "random_greedy_floor": {"actions_taken": 6},
        },
        "strongest_sota_flagged_for_v390": ("non_qwen_same_generator_random_label_ablation_v390"),
        "flagged_artifacts_skipped": [
            {"experiment_id": 4197},
            {"experiment_id": 4198},
            {"experiment_id": 4200},
            {"experiment_id": 4204},
        ],
    }
    payload.update(overrides)
    return payload


def _phase0(**overrides: object) -> dict:
    payload = {
        "experiment": "experiment_4197_verifier_reward_phase0_headroom_harness_build",
        "honest_verdict": (
            "complete: code_verifier_reward_operating_point_ready_phase0_"
            "0.956_j0.414_headroom0.600_harness_ready"
        ),
        "phase0_precision": 0.9561855670103093,
        "youden_j": 0.4137931034482759,
        "training_headroom_present": True,
        "harness_ready": True,
        "generation_suitability": {
            "base_passrate": 0.6,
            "own_visible_perfect_rate": 0.6,
            "training_headroom_present": True,
            "truncation_rate": 0.0,
        },
        "phase0_detail": {
            "phase0_clears": True,
            "phase0_precision": 0.9561855670103093,
            "youden_j": 0.4137931034482759,
        },
        "three_arm_smoke": {"harness_ready": True},
        "flagged_adversarial": True,
    }
    payload.update(overrides)
    return payload


def _launch(**overrides: object) -> dict:
    stable = (
        "/repo/results/verifier_reward_3arm_lora_rft/code_verifier_reward_lora_rft_a83b52882c198954"
    )
    payload = {
        "experiment": "experiment_4198_verifier_reward_3arm_rft_launch",
        "honest_verdict": "blocked_training_process_exited_before_checkpoint",
        "training_launched": False,
        "arm_corpus_sizes": {"A": 776, "B": 776, "C": 742, "D": 0},
        "accumulated_N": {"A": 776, "B": 776, "C": 742, "D": 0},
        "preconditions": {
            "a1_gate_clears": True,
            "arms_n_matched": True,
            "phase0_precision": 0.9561855670103093,
            "youden_j": 0.4137931034482759,
        },
        "operating_point": {"base_passrate": 0.6},
        "gold_control_early_read": {
            "status": "pending_training_checkpoint",
            "available": False,
            "base_passrate": 0.6,
        },
        "launch_status": {
            "status": "process_exited_early",
            "returncode": 1,
            "detached": True,
        },
        "stable_checkpoint_path": stable,
        "flagged_adversarial": True,
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
        "retired:\n- experiment_id: 2091\n  reason: archived\n", encoding="utf-8"
    )
    (root / "research-roadmap.yaml").write_text("milestone: 2026.06.390\n", encoding="utf-8")
    (root / "tests" / "python").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "python" / "test_pipeline_extract.py").write_text(
        "def test_pipeline_extract():\n    assert True\n", encoding="utf-8"
    )
    (root / "tests" / "python" / "test_docs.py").write_text(
        "def test_docs():\n    assert True\n", encoding="utf-8"
    )
    _write_json(root / "results" / "experiment_4206_capstone_v389.json", _capstone())
    _write_json(
        root / "results" / "experiment_4197_verifier_reward_phase0_headroom_harness_build.json",
        _phase0(),
    )
    _write_json(
        root / "results" / "experiment_4198_verifier_reward_3arm_rft_launch.json",
        _launch(
            stable_checkpoint_path=str(
                root
                / "results"
                / "verifier_reward_3arm_lora_rft"
                / "code_verifier_reward_lora_rft_a83b52882c198954"
            )
        ),
    )
    (
        root
        / "results"
        / "verifier_reward_3arm_lora_rft"
        / "code_verifier_reward_lora_rft_a83b52882c198954"
    ).mkdir(parents=True, exist_ok=True)
    return root


def run_happy(root: Path) -> dict:
    out = mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5)
    return json.loads(out.read_text(encoding="utf-8"))


def test_req_report_4207_spec_declares_contract() -> None:
    """REQ-REPORT-4207: OpenSpec declares the corrected archive contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    assert "REQ-REPORT-4207" in spec
    assert "SCENARIO-REPORT-4207" in spec
    assert "SCENARIO-REPORT-4207-BLOCKED-PRECONDITION" in spec
    assert "operating point CLEARED" in spec
    assert "phase0_precision=0.956" in spec
    assert "A=776" in spec and "B=776" in spec and "C=742" in spec
    assert "background LoRA process exited before its first checkpoint" in spec
    assert "not as a `NO-OPERATING-POINT` redo" in spec
    assert "non_qwen_random_label_ablation" in spec
    assert mod.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert mod.FIELD_PRINCIPLES["v389_close_state"] in spec
    assert mod.FIELD_PRINCIPLES["preconditions_checked"] in spec


def test_shared_helpers_and_archive_record_editing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4207: helper behavior is deterministic and YAML-safe."""

    assert mod.yaml_parses("a: 1\n") is True
    assert mod.yaml_parses("a: : :\n- [\n") is False
    assert mod.duration_from(None, None) == 0.0001
    assert mod.duration_from(100.0, 100.25) == 0.25
    assert mod.duration_from(100.0, 99.0) == 0.0001
    monkeypatch.setattr(mod.time, "perf_counter", lambda: 101.0)
    assert mod.duration_from(100.0, None) == 1.0
    assert mod.payload_checksum({"a": 1}) == mod.payload_checksum(
        {"a": 1, "reproducibility_checksum": "old"}
    )
    assert mod.is_sha256("a" * 64) is True
    assert mod.is_sha256("z" * 64) is False
    out = tmp_path / "artifact.json"
    mod.write_payload(out, {"b": 2, "a": 1})
    assert out.read_text(encoding="utf-8").startswith('{\n  "a"')
    assert mod._insert_before_tasks(["  title: no tasks"], "  finding: x") == [
        "  title: no tasks",
        "  finding: x",
    ]

    assert mod.archive_record_count(_research_complete_text(duplicates=3)) == 3
    close_state = mod.build_v389_close_state(
        {"4206": _capstone(), "4197": _phase0(), "4198": _launch()}, root=tmp_path
    )
    deduped, removed, action = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=3), close_state
    )
    assert action == "deduped"
    assert removed == 2
    assert mod.archive_record_count(deduped) == 1
    assert "operating point CLEARED" in deduped
    assert "INFRA artifact" in deduped
    assert "SYNCHRONOUSLY" in deduped
    assert mod.yaml_parses(deduped)
    updated, removed2, action2 = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=1), close_state
    )
    assert (removed2, action2) == (0, "updated")
    unchanged, removed4, action4 = mod.dedupe_or_update_record(updated, close_state)
    assert (unchanged, removed4, action4) == (updated, 0, "unchanged")
    old_activation, removed5, action5 = mod.dedupe_or_update_record(
        "milestones:\n- id: 2026.06.389\n  activation_recorded: old\n  tasks:\n  - id: exp4206\n",
        close_state,
    )
    assert (removed5, action5) == (0, "updated")
    assert "activation_recorded: exp4207-archive-v389-activate-v390" in old_activation
    no_tasks, removed6, action6 = mod.dedupe_or_update_record(
        "milestones:\n- id: 2026.06.389\n  title: no tasks\n",
        close_state,
    )
    assert (removed6, action6) == (0, "updated")
    assert "  finding: " in no_tasks
    appended, removed3, action3 = mod.dedupe_or_update_record(
        "# history\nmilestones:\n- id: 2026.06.388\n  finding: prior\n", close_state
    )
    assert (removed3, action3) == (0, "appended")
    assert "activation_recorded: exp4207-archive-v389-activate-v390" in appended


def test_precondition_helpers_and_source_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-4207-BLOCKED-PRECONDITION: resource probes are explicit."""

    assert mod._milestone_from_text("name: no milestone\n") == "unknown"
    assert mod.read_active_milestone(tmp_path) == ("unknown", "research-roadmap.yaml")
    (tmp_path / "research-roadmap-next.yaml").write_text(
        "milestone: 2026.06.390\n", encoding="utf-8"
    )
    assert mod.read_active_milestone(tmp_path) == ("2026.06.390", "research-roadmap-next.yaml")

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(bad) == {}
    list_path = tmp_path / "list.json"
    list_path.write_text("[1]", encoding="utf-8")
    assert mod.read_json_object(list_path) == {}
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"x": 1}), encoding="utf-8")
    assert mod.read_json_object(good) == {"x": 1}
    assert mod.is_sha256(mod.file_sha256(good))
    assert mod.file_sha256(tmp_path / "nope") is None

    root = make_repo(tmp_path / "repo")
    targets = mod.smart_subset_targets(root)
    assert "tests/python/test_pipeline_extract.py" in targets
    assert mod.smart_subset_command(targets)[0] == str(mod.PYTEST_BIN)
    assert mod.smart_subset_targets(tmp_path / "empty") == [mod.CORE_SMART_SUBSET[0]]
    assert mod._run_command(["true"], tmp_path).exit_code == 0
    assert mod._run_command(["definitely-not-a-real-binary-xyz"], tmp_path).exit_code == 127
    monkeypatch.setattr(mod, "_run_command", lambda command, root_path: GREEN)
    assert mod.run_smart_subset(root).exit_code == 0


def test_read_sources_and_build_v389_close_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4207: close-state records the .389 infra truth."""

    root = make_repo(tmp_path)
    sources = mod.read_v389_sources(root)
    assert sources["4206"]["verifier_as_reward_status"] == "NO-OPERATING-POINT"
    assert sources["4197"]["harness_ready"] is True
    assert sources["4198"]["arm_corpus_sizes"]["A"] == 776
    cited = mod.build_cited_upstream(root)
    assert {item["experiment_id"] for item in cited} == {"4206", "4197", "4198"}
    assert all(item["sha256"] is None or mod.is_sha256(item["sha256"]) for item in cited)

    state = mod.build_v389_close_state(sources, root=root)
    assert state["summary"] == (
        "operating_point_cleared_infra_killed_background_train_arc15_live_efficiency_only"
    )
    assert state["capstone_headline_is_infra_artifact"] is True
    assert state["capstone_headline_outcome"] == "verifier_reward_no_code_operating_point"
    assert state["capstone_verifier_as_reward_status"] == "NO-OPERATING-POINT"
    assert state["phase0_operating_point_status"] == "CLEARED"
    assert state["phase0_precision"] == 0.956
    assert state["youden_j"] == 0.414
    assert state["training_headroom"] == 0.6
    assert state["harness_ready"] is True
    assert state["three_arm_corpora_n_matched"] is True
    assert state["arm_a_certified_n"] == 776
    assert state["arm_b_random_label_n"] == 776
    assert state["arm_c_gold_n"] == 742
    assert state["base_passrate"] == 0.6
    assert state["training_launched"] is False
    assert state["background_process_exited_before_checkpoint"] is True
    assert state["background_training_infra_failed"] is True
    assert state["decisive_a_vs_b_collected"] is False
    assert state["a_vs_b_gate_blocked"] is True
    assert state["stable_checkpoint_slug"] == "code_verifier_reward_lora_rft_a83b52882c198954"
    assert state["stable_checkpoint_present"] is True
    assert mod._stable_checkpoint_present(root, "") is False
    assert state["infra_fix_for_v390"] == (
        "resume the stable checkpoint synchronously in-process with progress prints"
    )
    assert state["not_no_operating_point_redo"] is True
    assert state["total_levels_solved"] == 15
    assert state["total_games_solved"] == 13
    assert state["live_solver_efficiency_only_no_level"] is True
    assert state["live_solver_levels_completed"] == 0
    assert state["sota_flag_family"] == "non_qwen_random_label_ablation"
    assert state["strongest_sota_flagged_for_v390"] == (
        "non_qwen_same_generator_random_label_ablation_v390"
    )
    assert state["v390_planner_frame"] == (
        "oracle-distinct frontier headline + finish owed verifier-as-reward A-vs-B"
    )

    fallback = mod.build_v389_close_state(
        {"4206": _capstone(), "4197": _phase0(), "4198": _launch(arm_corpus_sizes={})},
        root=tmp_path,
    )
    assert fallback["arm_a_certified_n"] == 776
    assert fallback["three_arm_corpora_n_matched"] is True


def test_run_happy_path_writes_valid_artifact_and_updates_archive(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4207: complete path writes the terminal archive artifact."""

    root = make_repo(tmp_path, duplicates=2)
    artifact = run_happy(root)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["archived_milestone"] == "2026.06.389"
    assert artifact["activated_milestone"] == "2026.06.390"
    assert artifact["active_milestone_confirmed"] == "2026.06.390"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["research_complete_record_action"] == "deduped"
    assert artifact["research_complete_duplicates_removed"] == 1
    assert artifact["v389_close_state"]["phase0_operating_point_status"] == "CLEARED"
    assert artifact["v389_close_state"]["background_training_infra_failed"] is True
    assert artifact["v389_close_state"]["decisive_a_vs_b_collected"] is False
    assert artifact["v389_close_state"]["live_solver_efficiency_only_no_level"] is True
    assert (
        artifact["field_principles"]["v389_close_state"] == mod.FIELD_PRINCIPLES["v389_close_state"]
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.is_sha256(artifact["reproducibility_checksum"])
    complete_text = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(complete_text) == 1
    assert "operating point CLEARED" in complete_text
    assert "NOT NO-OPERATING-POINT" in complete_text
    mod.validate_artifact(artifact)


def test_run_real_pretest_branch_and_entrypoints_are_injectable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-4207: real pretest and CLI entrypoints can be substituted."""

    root = make_repo(tmp_path)
    monkeypatch.setattr(mod, "run_smart_subset", lambda root_path: GREEN)
    artifact = json.loads(mod.run(root, started_s=1.0, now_s=1.1).read_text(encoding="utf-8"))
    assert artifact["preconditions_checked"]["pretest_suite_green"] is True

    called_mod: dict[str, Path] = {}

    def fake_mod_run(root_path: Path = mod.REPO_ROOT) -> Path:
        called_mod["root"] = Path(root_path)
        return root / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(mod, "run", fake_mod_run)
    assert mod.main() == 0
    assert called_mod["root"] == mod.REPO_ROOT

    import carnot.experiment_4207_archive_v389_activate_v390 as entrypoint

    called: dict[str, Path] = {}

    def fake_run(root_path: Path = mod.REPO_ROOT) -> Path:
        called["root"] = Path(root_path)
        return root / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(entrypoint, "run", fake_run)
    assert entrypoint.main() == 0
    assert called["root"] == entrypoint.REPO_ROOT

    script_path = Path("results/experiment_4207_archive_v389_activate_v390.py")
    spec = importlib.util.spec_from_file_location("exp4207_archive_script", script_path)
    assert spec and spec.loader
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "run", fake_run)
    assert script.main() == 0


def test_run_blocked_preconditions_before_research_complete_edit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4207-BLOCKED-PRECONDITION: blocked paths do not fabricate success."""

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

    root3 = make_repo(tmp_path / "manifest_poison")
    (root3 / "ops" / "exclusion_manifest.yaml").write_text("a: : :\n- [\n", encoding="utf-8")
    artifact3 = json.loads(mod.run(root3, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact3["honest_verdict"] == "blocked_exclusion_manifest_yaml_poison"

    root4 = make_repo(tmp_path / "wrong_milestone")
    (root4 / "research-roadmap.yaml").write_text("milestone: 2026.06.389\n", encoding="utf-8")
    artifact4 = json.loads(mod.run(root4, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact4["honest_verdict"] == "blocked_v390_not_active"

    root5 = make_repo(tmp_path / "red")
    before = (root5 / "research-complete.yaml").read_text(encoding="utf-8")
    artifact5 = json.loads(mod.run(root5, pretest_result=RED).read_text(encoding="utf-8"))
    assert artifact5["honest_verdict"] == "blocked_smart_subset_pretest_not_green"
    assert artifact5["preconditions_checked"]["pretest_suite_green"] is False
    assert (root5 / "research-complete.yaml").read_text(encoding="utf-8") == before

    root6 = make_repo(tmp_path / "capstone_missing")
    (root6 / "results" / "experiment_4206_capstone_v389.json").unlink()
    artifact6 = json.loads(mod.run(root6, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact6["honest_verdict"] == "blocked_v389_capstone_missing"

    root7 = make_repo(tmp_path / "phase0_missing")
    (
        root7 / "results" / "experiment_4197_verifier_reward_phase0_headroom_harness_build.json"
    ).unlink()
    artifact7 = json.loads(mod.run(root7, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact7["honest_verdict"] == "blocked_phase0_artifact_missing"

    root8 = make_repo(tmp_path / "launch_missing")
    (root8 / "results" / "experiment_4198_verifier_reward_3arm_rft_launch.json").unlink()
    artifact8 = json.loads(mod.run(root8, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact8["honest_verdict"] == "blocked_launch_artifact_missing"


def test_run_blocked_research_complete_edit_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4207: invalid archive edits are blocked before completion."""

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


def test_build_artifact_shapes(tmp_path: Path) -> None:
    """REQ-REPORT-4207: complete and blocked builders keep schema shape."""

    root = make_repo(tmp_path)
    state = mod.build_v389_close_state(mod.read_v389_sources(root), root=root)
    complete = mod.build_complete_artifact(
        v389_close_state=state,
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


def test_validate_artifact_rejection_guards(tmp_path: Path) -> None:
    """REQ-REPORT-4207: validation rejects artifacts that launder the .389 truth."""

    good = run_happy(make_repo(tmp_path))

    def set_path(obj: dict, path: list[str], value: object) -> None:
        cur = obj
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = value

    cases = [
        ("missing required fields", lambda a: a.pop("v389_close_state")),
        ("terminal-prefixed", lambda a: a.__setitem__("honest_verdict", "done")),
        ("field_principles must be a mapping", lambda a: a.__setitem__("field_principles", "x")),
        ("missing field principles", lambda a: a.__setitem__("field_principles", {})),
        (
            "principle must match REQ-REPORT-4207",
            lambda a: a["field_principles"].__setitem__("v389_close_state", "wrong"),
        ),
        ("archived milestone", lambda a: a.__setitem__("archived_milestone", "2026.06.999")),
        ("activated milestone", lambda a: a.__setitem__("activated_milestone", "2026.06.999")),
        ("research-complete YAML", lambda a: a.__setitem__("research_complete_yaml_parses", False)),
        ("exclusion manifest", lambda a: a.__setitem__("exclusion_manifest_parses", False)),
        ("pretest suite", lambda a: a.__setitem__("pretest_suite_green", False)),
        ("active milestone", lambda a: a.__setitem__("active_milestone_confirmed", "2026.06.389")),
        ("v389_close_state must be a mapping", lambda a: a.__setitem__("v389_close_state", "x")),
        (
            "infra artifact",
            lambda a: set_path(
                a, ["v389_close_state", "capstone_headline_is_infra_artifact"], False
            ),
        ),
        (
            "phase0 status",
            lambda a: set_path(a, ["v389_close_state", "phase0_operating_point_status"], "MISSING"),
        ),
        ("phase0_precision", lambda a: set_path(a, ["v389_close_state", "phase0_precision"], 0.8)),
        ("youden_j", lambda a: set_path(a, ["v389_close_state", "youden_j"], 0.1)),
        (
            "training_headroom",
            lambda a: set_path(a, ["v389_close_state", "training_headroom"], 0.1),
        ),
        ("harness_ready", lambda a: set_path(a, ["v389_close_state", "harness_ready"], False)),
        (
            "N-matched",
            lambda a: set_path(a, ["v389_close_state", "three_arm_corpora_n_matched"], False),
        ),
        ("arm A", lambda a: set_path(a, ["v389_close_state", "arm_a_certified_n"], 775)),
        ("arm B", lambda a: set_path(a, ["v389_close_state", "arm_b_random_label_n"], 775)),
        ("arm C", lambda a: set_path(a, ["v389_close_state", "arm_c_gold_n"], 741)),
        ("base_passrate", lambda a: set_path(a, ["v389_close_state", "base_passrate"], 0.5)),
        (
            "training_launched",
            lambda a: set_path(a, ["v389_close_state", "training_launched"], True),
        ),
        (
            "background process",
            lambda a: set_path(
                a, ["v389_close_state", "background_process_exited_before_checkpoint"], False
            ),
        ),
        (
            "infra failure",
            lambda a: set_path(a, ["v389_close_state", "background_training_infra_failed"], False),
        ),
        ("A-vs-B", lambda a: set_path(a, ["v389_close_state", "decisive_a_vs_b_collected"], True)),
        ("gate-blocked", lambda a: set_path(a, ["v389_close_state", "a_vs_b_gate_blocked"], False)),
        (
            "stable checkpoint",
            lambda a: set_path(a, ["v389_close_state", "stable_checkpoint_slug"], "wrong"),
        ),
        (
            "synchronously",
            lambda a: set_path(a, ["v389_close_state", "infra_fix_for_v390"], "background"),
        ),
        ("redo", lambda a: set_path(a, ["v389_close_state", "not_no_operating_point_redo"], False)),
        (
            "total levels solved",
            lambda a: set_path(a, ["v389_close_state", "total_levels_solved"], 14),
        ),
        (
            "total games solved",
            lambda a: set_path(a, ["v389_close_state", "total_games_solved"], 12),
        ),
        (
            "efficiency-only",
            lambda a: set_path(
                a, ["v389_close_state", "live_solver_efficiency_only_no_level"], False
            ),
        ),
        (
            "live levels",
            lambda a: set_path(a, ["v389_close_state", "live_solver_levels_completed"], 1),
        ),
        ("SOTA flag", lambda a: set_path(a, ["v389_close_state", "sota_flag_family"], "qwen")),
        (
            "planner frame",
            lambda a: set_path(a, ["v389_close_state", "v390_planner_frame"], "redo"),
        ),
        ("duration_s", lambda a: a.__setitem__("duration_s", 0)),
        ("inference_substrate", lambda a: a.__setitem__("inference_substrate", "live_training")),
        ("cited_upstream_artifacts", lambda a: a.__setitem__("cited_upstream_artifacts", "x")),
        ("reproducibility_checksum", lambda a: a.__setitem__("reproducibility_checksum", "short")),
    ]
    for match, mutate in cases:
        artifact = copy.deepcopy(good)
        mutate(artifact)
        with pytest.raises(ValueError, match=match):
            mod.validate_artifact(artifact)

    mismatch = copy.deepcopy(good)
    mismatch["honest_verdict"] = "success: changed"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(mismatch)
