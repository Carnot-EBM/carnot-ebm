"""Tests for Exp 4219 `.390` archive / `.391` activation.

Spec refs: REQ-REPORT-4219, SCENARIO-REPORT-4219,
SCENARIO-REPORT-4219-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v390_activate_v391_4219 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="81 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.389\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.390\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-15'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4218-capstone-v390\n"
        "    result: OK\n"
    )
    return head + block * duplicates


def _capstone(**overrides: object) -> dict:
    payload = {
        "experiment_id": 4218,
        "honest_verdict": (
            "complete: capstone_v390_oracle_distinct_no_headroom_or_no_learnable_signal_"
            "oracle_NO-HEADROOM-OR-NO-SIGNAL_reward_ACCUMULATING_arc_levels16"
        ),
        "headline_outcome": "oracle_distinct_no_headroom_or_no_learnable_signal",
        "oracle_distinct_status": "NO-HEADROOM-OR-NO-SIGNAL",
        "learned_arc_verifier": {
            "honest_verdict": "blocked_arc_pool_no_candidate_labels",
            "accepted_rejected_n": {"accepted": 0, "rejected": 0, "total": 0},
            "selector_trained": False,
            "verifier_is_oracle": False,
        },
        "oracle_distinct_frontier": {
            "honest_verdict": "blocked_gate_check_failed",
            "comparison_ran": False,
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: "
                "exp4209-oracle-distinct-arc-verifier-build.selector_trained "
                "(actual=False == expected=True)"
            ),
            "headroom_present": True,
            "matched_control": False,
            "oracle_distinct_beats_vote": False,
            "oracle_distinct_status": "NO-HEADROOM-OR-NO-SIGNAL",
            "verifier_is_oracle": False,
        },
        "detector_selection_divergence": {
            "detection_auroc_by_domain": {"arc": 0.9016},
            "detection_auroc_ci95_by_domain": {"arc": [0.7828, 0.9984]},
            "n_by_domain": {"arc": 8041},
            "selector_headroom_by_domain": {"arc": 0.129},
            "verifier_is_oracle_by_domain": {"arc": False},
        },
        "verifier_as_reward": {
            "honest_verdict": "progress: accumulating_verifier_reward_training_no_eval_yet",
            "training": {
                "status": "failed",
                "error": "ValueError: Target module Gemma4ClippableLinear(...) is not supported.",
                "used_detached_process": False,
            },
            "verifier_as_reward_status": "ACCUMULATING",
            "verifier_is_oracle": True,
            "youden_j": 0.4137931034482759,
        },
        "arc_progress": {
            "honest_verdict": "success: incremental_progress_sc25-635fd71a_advanced_to_L2_total16",
            "levels_completed": 2,
            "new_levels_solved_this_task": 1,
            "total_arc_levels_solved": 16,
            "total_arc_games_solved": 13,
        },
        "live_solver_accuracy": {
            "honest_verdict": "complete: solver_completes_0_levels_live_lp85-305b61c3_efficiency_only",
            "levels_completed": 0,
            "solver_beats_floor_accuracy": False,
            "solver_beats_floor_efficiency": True,
            "solver_completes_level": False,
        },
        "flagged_artifacts_skipped": [
            {"experiment_id": 4212, "reason": "flagged_adversarial:true"},
            {"experiment_id": 4216, "reason": "flagged_adversarial:true"},
        ],
        "total_arc_levels_solved": 16,
    }
    payload.update(overrides)
    return payload


def _build(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "blocked_arc_pool_no_candidate_labels",
        "candidate_pool_source": "/repo/results/arc3_trm_verifier_rerank.json",
        "accepted_rejected_n": {"accepted": 0, "rejected": 0, "total": 0},
        "selector_trained": False,
        "oracle_distinct_auroc": 0.0,
        "learned_verifier_path": "",
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _gate(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "blocked_gate_check_failed",
        "status": "blocked",
        "blocked_at_layer": "conductor_pre_gate",
        "gate_check_summary": (
            "1 of 1 gate(s) failed; first failure: "
            "exp4209-oracle-distinct-arc-verifier-build.selector_trained "
            "(actual=False == expected=True)"
        ),
        "gates_evaluated": [
            {
                "artifact_field": "selector_trained",
                "actual": False,
                "expected": True,
                "passed": False,
            }
        ],
    }
    payload.update(overrides)
    return payload


def _detector(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: detector_selection_divergence_sudoku_math",
        "detection_auroc_by_domain": {"arc": 0.9016},
        "detection_auroc_ci95_by_domain": {"arc": [0.7828, 0.9984]},
        "selector_headroom_by_domain": {"arc": 0.129},
        "n_by_domain": {"arc": 8041},
        "verifier_is_oracle_by_domain": {"arc": False},
    }
    payload.update(overrides)
    return payload


def _reward(**overrides: object) -> dict:
    checkpoint = (
        "/repo/results/verifier_reward_3arm_lora_rft/code_verifier_reward_lora_rft_a83b52882c198954"
    )
    payload = {
        "honest_verdict": "progress: accumulating_verifier_reward_training_no_eval_yet",
        "arm_corpus_sizes": {"A": 776, "B": 776, "C": 742, "D": 0},
        "model_specs": {
            "a1_operating_point": {"base_passrate": 0.6},
            "on_policy_generator": "google/gemma-4-E4B-it",
            "qwen_train_base_forbidden": True,
        },
        "preconditions": {
            "arms_n_matched": True,
            "stable_checkpoint_path": checkpoint,
            "stable_checkpoint_readable": True,
        },
        "stable_checkpoint_path": checkpoint,
        "training": {
            "status": "failed",
            "error": "ValueError: Target module Gemma4ClippableLinear(...) is not supported.",
            "used_detached_process": False,
        },
        "verifier_is_oracle": True,
        "youden_j": 0.4137931034482759,
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
        "retired:\n- experiment_id: 4216\n  reason: flagged\n", encoding="utf-8"
    )
    (root / "research-roadmap.yaml").write_text("milestone: 2026.06.391\n", encoding="utf-8")
    (root / "tests" / "python").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "python" / "test_pipeline_extract.py").write_text(
        "def test_pipeline_extract():\n    assert True\n", encoding="utf-8"
    )
    (root / "tests" / "python" / "test_docs.py").write_text(
        "def test_docs():\n    assert True\n", encoding="utf-8"
    )
    _write_json(root / "results" / "experiment_4218_capstone_v390.json", _capstone())
    _write_json(
        root / "results" / "experiment_4209_oracle_distinct_arc_verifier_build.json", _build()
    )
    _write_json(
        root / "results" / "experiment_4210_oracle_distinct_arc_verifier_beats_vote.json",
        _gate(),
    )
    _write_json(root / "results" / "experiment_4208_verifier_as_detector_auroc.json", _detector())
    _write_json(
        root / "results" / "experiment_4211_verifier_as_reward_finish_synchronous.json",
        _reward(
            stable_checkpoint_path=str(
                root
                / "results"
                / "verifier_reward_3arm_lora_rft"
                / "code_verifier_reward_lora_rft_a83b52882c198954"
            ),
            preconditions={
                "arms_n_matched": True,
                "stable_checkpoint_path": str(
                    root
                    / "results"
                    / "verifier_reward_3arm_lora_rft"
                    / "code_verifier_reward_lora_rft_a83b52882c198954"
                ),
                "stable_checkpoint_readable": True,
            },
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


def test_req_report_4219_spec_declares_contract() -> None:
    """REQ-REPORT-4219: OpenSpec declares the archive correction contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    assert "REQ-REPORT-4219" in spec
    assert "SCENARIO-REPORT-4219" in spec
    assert "SCENARIO-REPORT-4219-BLOCKED-PRECONDITION" in spec
    assert "wrong-file DATA bug" in spec
    assert "AUROC `0.9016`" in spec
    assert "Gemma4ClippableLinear is not supported" in spec
    assert "Phase-0 precision `0.956`" in spec
    assert "total_levels_solved=16" in spec
    assert (
        "the de-risked oracle-distinct retry + the harness-first verifier-as-reward FINISH" in spec
    )
    assert mod.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert mod.FIELD_PRINCIPLES["v390_close_state"] in spec
    assert mod.FIELD_PRINCIPLES["preconditions_checked"] in spec


def test_shared_helpers_and_archive_record_editing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4219: helper behavior is deterministic and YAML-safe."""

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

    close_state = mod.build_v390_close_state(
        {
            "4218": _capstone(),
            "4209": _build(),
            "4210": _gate(),
            "4208": _detector(),
            "4211": _reward(),
        },
        root=tmp_path,
    )
    assert mod.archive_record_count(_research_complete_text(duplicates=3)) == 3
    deduped, removed, action = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=3), close_state
    )
    assert (removed, action) == (2, "deduped")
    assert mod.archive_record_count(deduped) == 1
    assert "wrong-file DATA bug" in deduped
    assert "AUROC=0.9016" in deduped
    assert "Gemma4ClippableLinear" in deduped
    assert mod.yaml_parses(deduped)
    updated, removed2, action2 = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=1), close_state
    )
    assert (removed2, action2) == (0, "updated")
    unchanged, removed3, action3 = mod.dedupe_or_update_record(updated, close_state)
    assert (unchanged, removed3, action3) == (updated, 0, "unchanged")
    old_activation, removed4, action4 = mod.dedupe_or_update_record(
        "milestones:\n- id: 2026.06.390\n  activation_recorded: old\n  tasks:\n  - id: exp4218\n",
        close_state,
    )
    assert (removed4, action4) == (0, "updated")
    assert "activation_recorded: exp4219-archive-v390-activate-v391" in old_activation
    appended, removed5, action5 = mod.dedupe_or_update_record(
        "# history\nmilestones:\n- id: 2026.06.389\n  finding: prior\n", close_state
    )
    assert (removed5, action5) == (0, "appended")
    assert "activation_recorded: exp4219-archive-v390-activate-v391" in appended


def test_precondition_helpers_and_source_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-4219-BLOCKED-PRECONDITION: resource probes are explicit."""

    assert mod._milestone_from_text("name: no milestone\n") == "unknown"
    assert mod.read_active_milestone(tmp_path) == ("unknown", "research-roadmap.yaml")
    (tmp_path / "research-roadmap-next.yaml").write_text(
        "milestone: 2026.06.391\n", encoding="utf-8"
    )
    assert mod.read_active_milestone(tmp_path) == ("2026.06.391", "research-roadmap-next.yaml")
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

    def raise_timeout(command: list[str], **_: object) -> None:
        raise mod.subprocess.TimeoutExpired(command, timeout=1)

    monkeypatch.setattr(mod.subprocess, "run", raise_timeout)
    assert mod._run_command(["slow"], tmp_path).exit_code == -1

    extra = root / "tests" / "python" / "test_extra_4219.py"
    extra.write_text("def test_extra():\n    assert True\n", encoding="utf-8")
    original_git_lines = mod._git_lines
    monkeypatch.setattr(
        mod,
        "_git_lines",
        lambda root_path, args: ["tests/python/test_extra_4219.py"] if args[:1] == ["diff"] else [],
    )
    assert "tests/python/test_extra_4219.py" in mod.smart_subset_targets(root)
    monkeypatch.setattr(mod, "_run_command", lambda command, root_path: GREEN)
    assert mod.run_smart_subset(root).exit_code == 0
    monkeypatch.setattr(
        mod,
        "_run_command",
        lambda command, root_path: mod.CommandResult(command, 0, "a.py\n\nb.py\n", ""),
    )
    monkeypatch.setattr(mod, "_git_lines", original_git_lines)
    assert mod._git_lines(root, ["diff"]) == ["a.py", "b.py"]


def test_read_sources_and_build_v390_close_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4219: close-state records the .390 infra truth."""

    root = make_repo(tmp_path)
    sources = mod.read_v390_sources(root)
    assert sources["4218"]["oracle_distinct_status"] == "NO-HEADROOM-OR-NO-SIGNAL"
    assert sources["4209"]["selector_trained"] is False
    assert sources["4211"]["training"]["status"] == "failed"
    cited = mod.build_cited_upstream(root)
    assert {item["experiment_id"] for item in cited} == {"4218", "4209", "4210", "4208", "4211"}
    assert all(item["sha256"] is None or mod.is_sha256(item["sha256"]) for item in cited)

    state = mod.build_v390_close_state(sources, root=root)
    assert state["summary"] == "oracle_distinct_wrong_file_data_bug_signal_exists_reward_peft_arc16"
    assert state["capstone_headline_is_infra_artifact"] is True
    assert state["capstone_oracle_distinct_status"] == "NO-HEADROOM-OR-NO-SIGNAL"
    assert state["oracle_distinct_comparison_ran"] is False
    assert state["oracle_distinct_gate_blocked_on_data"] is True
    assert state["wrong_file_candidate_pool_source"].endswith("arc3_trm_verifier_rerank.json")
    assert state["accepted_rejected_n"] == {"accepted": 0, "rejected": 0, "total": 0}
    assert state["selector_trained"] is False
    assert "selector_trained" in state["gate_check_summary"]
    assert state["working_label_loader"] == "scripts/exp_verifier_detector_auroc.py:load_arc_rows"
    assert state["working_label_pool_path"] == "results/arc3_gap3_stage2_eval_pool.json.gz"
    assert state["working_programs_path"] == "results/arc3_gap4_induced_programs.json"
    assert state["arc_labeled_candidate_n"] == 8041
    assert state["oracle_distinct_arc_detection_auroc"] == 0.9016
    assert state["oracle_distinct_arc_detection_auroc_ci95"] == [0.7828, 0.9984]
    assert state["arc_selector_headroom"] == 0.129
    assert state["arc_detector_verifier_is_oracle"] is False
    assert state["reward_training_status"] == "failed"
    assert state["reward_peft_attach_failed"] is True
    assert "Gemma4ClippableLinear" in state["reward_training_error"]
    assert state["reward_phase0_precision"] == 0.956
    assert state["reward_youden_j"] == 0.4138
    assert state["reward_corpora"] == {"A": 776, "B": 776, "C": 742}
    assert state["reward_checkpoint_slug"] == "code_verifier_reward_lora_rft_a83b52882c198954"
    assert state["reward_checkpoint_intact"] is True
    assert state["total_levels_solved"] == 16
    assert state["total_games_solved"] == 13
    assert state["live_solver_efficiency_only_no_level"] is True
    assert state["flagged_artifacts_skipped"] == [4212, 4216]
    assert state["v391_frame"] == (
        "the de-risked oracle-distinct retry + the harness-first verifier-as-reward FINISH"
    )

    fallback = mod.build_v390_close_state(
        {
            "4218": _capstone(detector_selection_divergence={}, flagged_artifacts_skipped="bad"),
            "4209": _build(accepted_rejected_n={}),
            "4210": _gate(gates_evaluated=[]),
            "4208": _detector(n_by_domain={}, detection_auroc_ci95_by_domain={"arc": "bad"}),
            "4211": _reward(arm_corpus_sizes={}),
        },
        root=tmp_path,
    )
    assert fallback["arc_labeled_candidate_n"] == 8041
    assert fallback["oracle_distinct_arc_detection_auroc_ci95"] == [0.7828, 0.9984]
    assert fallback["reward_corpora"] == {"A": 776, "B": 776, "C": 742}
    assert fallback["flagged_artifacts_skipped"] == []
    assert mod._checkpoint_present(tmp_path, "", False) is False
    relative_checkpoint = tmp_path / "relative_checkpoint"
    relative_checkpoint.mkdir()
    assert mod._checkpoint_present(tmp_path, "relative_checkpoint", False) is True


def test_run_happy_path_writes_valid_artifact_and_updates_archive(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4219: complete path writes the terminal archive artifact."""

    root = make_repo(tmp_path, duplicates=2)
    artifact = run_happy(root)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["archived_milestone"] == "2026.06.390"
    assert artifact["activated_milestone"] == "2026.06.391"
    assert artifact["active_milestone_confirmed"] == "2026.06.391"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["research_complete_record_action"] == "deduped"
    assert artifact["research_complete_duplicates_removed"] == 1
    assert artifact["v390_close_state"]["oracle_distinct_gate_blocked_on_data"] is True
    assert artifact["v390_close_state"]["oracle_distinct_arc_detection_auroc"] == 0.9016
    assert artifact["v390_close_state"]["reward_peft_attach_failed"] is True
    assert artifact["v390_close_state"]["total_levels_solved"] == 16
    assert (
        artifact["field_principles"]["v390_close_state"] == mod.FIELD_PRINCIPLES["v390_close_state"]
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.is_sha256(artifact["reproducibility_checksum"])
    complete_text = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(complete_text) == 1
    assert "wrong-file DATA bug" in complete_text
    assert "de-risked oracle-distinct retry" in complete_text
    mod.validate_artifact(artifact)


def test_run_real_pretest_branch_and_entrypoints_are_injectable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-4219: real pretest and CLI entrypoints can be substituted."""

    root = make_repo(tmp_path)
    monkeypatch.setattr(mod, "run_smart_subset", lambda root_path: GREEN)
    artifact = json.loads(mod.run(root, started_s=1.0, now_s=1.1).read_text(encoding="utf-8"))
    assert artifact["preconditions_checked"]["smart_subset_pretest"]["green"] is True

    called_mod: dict[str, Path] = {}

    def fake_mod_run(root_path: Path = mod.REPO_ROOT) -> Path:
        called_mod["root"] = Path(root_path)
        return root / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(mod, "run", fake_mod_run)
    assert mod.main() == 0
    assert called_mod["root"] == mod.REPO_ROOT

    import carnot.experiment_4219_archive_v390_activate_v391 as entrypoint

    called: dict[str, Path] = {}

    def fake_run(root_path: Path = mod.REPO_ROOT) -> Path:
        called["root"] = Path(root_path)
        return root / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(entrypoint, "run", fake_run)
    assert entrypoint.main() == 0
    assert called["root"] == entrypoint.REPO_ROOT

    script_path = Path("results/experiment_4219_archive_v390_activate_v391.py")
    spec = importlib.util.spec_from_file_location("exp4219_archive_script", script_path)
    assert spec and spec.loader
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "run", fake_run)
    assert script.main() == 0


def test_run_blocked_preconditions_before_research_complete_edit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4219-BLOCKED-PRECONDITION: blocked paths do not fabricate success."""

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

    root4 = make_repo(tmp_path / "red")
    before = (root4 / "research-complete.yaml").read_text(encoding="utf-8")
    artifact4 = json.loads(mod.run(root4, pretest_result=RED).read_text(encoding="utf-8"))
    assert artifact4["honest_verdict"] == "blocked_smart_subset_pretest_not_green"
    assert artifact4["preconditions_checked"]["smart_subset_pretest"]["green"] is False
    assert (root4 / "research-complete.yaml").read_text(encoding="utf-8") == before

    root5 = make_repo(tmp_path / "wrong_milestone")
    (root5 / "research-roadmap.yaml").write_text("milestone: 2026.06.390\n", encoding="utf-8")
    artifact5 = json.loads(mod.run(root5, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact5["honest_verdict"] == "blocked_v391_not_active"

    root6 = make_repo(tmp_path / "capstone_missing")
    (root6 / "results" / "experiment_4218_capstone_v390.json").unlink()
    artifact6 = json.loads(mod.run(root6, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact6["honest_verdict"] == "blocked_v390_capstone_missing"

    root7 = make_repo(tmp_path / "build_missing")
    (root7 / "results" / "experiment_4209_oracle_distinct_arc_verifier_build.json").unlink()
    artifact7 = json.loads(mod.run(root7, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact7["honest_verdict"] == "blocked_oracle_distinct_build_missing"

    root8 = make_repo(tmp_path / "gate_missing")
    (root8 / "results" / "experiment_4210_oracle_distinct_arc_verifier_beats_vote.json").unlink()
    artifact8 = json.loads(mod.run(root8, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact8["honest_verdict"] == "blocked_oracle_distinct_gate_missing"

    root9 = make_repo(tmp_path / "detector_missing")
    (root9 / "results" / "experiment_4208_verifier_as_detector_auroc.json").unlink()
    artifact9 = json.loads(mod.run(root9, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact9["honest_verdict"] == "blocked_detector_artifact_missing"

    root10 = make_repo(tmp_path / "reward_missing")
    (root10 / "results" / "experiment_4211_verifier_as_reward_finish_synchronous.json").unlink()
    artifact10 = json.loads(mod.run(root10, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact10["honest_verdict"] == "blocked_reward_artifact_missing"


def test_run_blocked_research_complete_edit_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4219: invalid archive edits are blocked before completion."""

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
    """REQ-REPORT-4219: complete and blocked builders keep schema shape."""

    root = make_repo(tmp_path)
    state = mod.build_v390_close_state(mod.read_v390_sources(root), root=root)
    complete = mod.build_complete_artifact(
        v390_close_state=state,
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
    """REQ-REPORT-4219: validation rejects artifacts that launder the .390 truth."""

    good = run_happy(make_repo(tmp_path))

    def set_path(obj: dict, path: list[str], value: object) -> None:
        cur = obj
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = value

    cases = [
        ("missing required fields", lambda a: a.pop("v390_close_state")),
        ("terminal-prefixed", lambda a: a.__setitem__("honest_verdict", "done")),
        ("field_principles must be a mapping", lambda a: a.__setitem__("field_principles", "x")),
        ("missing field principles", lambda a: a.__setitem__("field_principles", {})),
        (
            "principle must match REQ-REPORT-4219",
            lambda a: a["field_principles"].__setitem__("v390_close_state", "wrong"),
        ),
        ("archived milestone", lambda a: a.__setitem__("archived_milestone", "2026.06.999")),
        ("activated milestone", lambda a: a.__setitem__("activated_milestone", "2026.06.999")),
        ("research-complete YAML", lambda a: a.__setitem__("research_complete_yaml_parses", False)),
        ("exclusion manifest", lambda a: a.__setitem__("exclusion_manifest_parses", False)),
        ("pretest suite", lambda a: a.__setitem__("pretest_suite_green", False)),
        ("active milestone", lambda a: a.__setitem__("active_milestone_confirmed", "2026.06.390")),
        ("v390_close_state must be a mapping", lambda a: a.__setitem__("v390_close_state", "x")),
        (
            "infra artifact",
            lambda a: set_path(
                a, ["v390_close_state", "capstone_headline_is_infra_artifact"], False
            ),
        ),
        (
            "data block",
            lambda a: set_path(
                a, ["v390_close_state", "oracle_distinct_gate_blocked_on_data"], False
            ),
        ),
        (
            "comparison ran",
            lambda a: set_path(a, ["v390_close_state", "oracle_distinct_comparison_ran"], True),
        ),
        (
            "selector",
            lambda a: set_path(a, ["v390_close_state", "selector_trained"], True),
        ),
        (
            "accepted labels",
            lambda a: set_path(a, ["v390_close_state", "accepted_rejected_n"], {"total": 1}),
        ),
        (
            "AUROC",
            lambda a: set_path(a, ["v390_close_state", "oracle_distinct_arc_detection_auroc"], 0.5),
        ),
        (
            "candidate N",
            lambda a: set_path(a, ["v390_close_state", "arc_labeled_candidate_n"], 0),
        ),
        (
            "oracle flag",
            lambda a: set_path(a, ["v390_close_state", "arc_detector_verifier_is_oracle"], True),
        ),
        (
            "PEFT failure",
            lambda a: set_path(a, ["v390_close_state", "reward_peft_attach_failed"], False),
        ),
        (
            "phase0 precision",
            lambda a: set_path(a, ["v390_close_state", "reward_phase0_precision"], 0.1),
        ),
        (
            "Youden",
            lambda a: set_path(a, ["v390_close_state", "reward_youden_j"], 0.1),
        ),
        (
            "corpora",
            lambda a: set_path(a, ["v390_close_state", "reward_corpora"], {"A": 1, "B": 1, "C": 1}),
        ),
        (
            "checkpoint",
            lambda a: set_path(a, ["v390_close_state", "reward_checkpoint_intact"], False),
        ),
        ("ARC levels", lambda a: set_path(a, ["v390_close_state", "total_levels_solved"], 15)),
        ("ARC games", lambda a: set_path(a, ["v390_close_state", "total_games_solved"], 12)),
        (
            "live",
            lambda a: set_path(
                a, ["v390_close_state", "live_solver_efficiency_only_no_level"], False
            ),
        ),
        (
            "flagged",
            lambda a: set_path(a, ["v390_close_state", "flagged_artifacts_skipped"], [4212]),
        ),
        (
            "v391 frame",
            lambda a: set_path(a, ["v390_close_state", "v391_frame"], "redo"),
        ),
    ]
    for label, mutate in cases:
        artifact = copy.deepcopy(good)
        mutate(artifact)
        with pytest.raises(ValueError, match=label):
            mod.validate_artifact(artifact)
