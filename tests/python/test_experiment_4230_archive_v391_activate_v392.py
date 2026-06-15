"""Tests for Exp 4230 `.391` archive / `.392` activation.

Spec refs: REQ-REPORT-4230, SCENARIO-REPORT-4230,
SCENARIO-REPORT-4230-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v391_activate_v392_4230 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="81 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.390\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.391\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-15'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4229-capstone-v391\n"
        "    result: OK\n"
    )
    return head + block * duplicates


def _capstone(**overrides: object) -> dict:
    payload = {
        "experiment_id": 4229,
        "honest_verdict": (
            "complete: capstone_v391_oracle_distinct_verifier_ties_vote_with_headroom_null_"
            "oracle_TIES-VOTE-NULL_reward_HARNESS-DEFERRED_arc_levels17_flagged_skipped3_"
            "diffusiongemma_still_pending"
        ),
        "headline_outcome": "oracle_distinct_verifier_ties_vote_with_headroom_null",
        "oracle_distinct_status": "TIES-VOTE-NULL",
        "diffusiongemma_gate_resolvable": False,
        "learned_arc_verifier": {
            "accepted_rejected_n": {"accepted": 14, "rejected": 1782, "total": 1796},
            "metric_source": "computed_from_clean_model_oof_rows_and_clean_gate_task_rows",
            "model_type": "standardized_logistic_regression",
            "off_fold_auroc": 0.7790203623536957,
            "oof_row_n": 1796,
            "summary_artifact_status": "skipped_flagged_adversarial",
            "verifier_is_oracle": False,
            "wrong_majority_n": 5,
        },
        "oracle_distinct_frontier": {
            "gate_ran": True,
            "headroom_present": True,
            "honest_verdict": "complete: oracle_distinct_verifier_ties_vote_with_headroom",
            "matched_control_delta": 0.0,
            "matched_control_present": True,
            "n_tasks": 14,
            "oracle_at_k": 1.0,
            "oracle_distinct_beats_vote": False,
            "oracle_distinct_status": "TIES-VOTE-NULL",
            "pass_rates": {
                "verifier_at_1": 0.5714285714,
                "vote_at_1": 0.6428571429,
                "matched_control_at_1": 0.5714285714,
                "arbiter_override_at_1": 0.6428571429,
            },
            "verifier_is_oracle": False,
            "verifier_minus_vote_ci95": [-0.2142857143, 0.0],
            "verifier_minus_vote_delta": -0.0714285714,
        },
        "verifier_as_reward": {
            "honest_verdict": "progress: accumulating_verifier_reward_training_no_eval_yet",
            "status": "included",
            "verifier_is_oracle": True,
            "youden_j": 0.4137931034482759,
        },
        "arc_progress": {
            "honest_verdict": "success: incremental_progress_sc25-635fd71a_advanced_to_L3_total17",
            "levels_completed": 3,
            "new_levels_solved_this_task": 1,
            "total_arc_games_solved": 13,
            "total_arc_levels_solved": 17,
        },
        "live_solver_accuracy": {
            "honest_verdict": "complete: solver_completes_0_levels_live_lp85-305b61c3_efficiency_only",
            "levels_completed": 0,
            "live_env_reachable": True,
            "score": 0.0,
            "scorecard_levels_completed": 0,
            "solver_beats_floor_accuracy": False,
            "solver_beats_floor_efficiency": True,
            "solver_completes_level": False,
        },
        "flagged_artifacts_skipped": [
            {"experiment_id": 4220, "reason": "flagged_adversarial:true"},
            {"experiment_id": 4222, "reason": "flagged_adversarial:true"},
            {"experiment_id": 4223, "reason": "flagged_adversarial:true"},
        ],
        "total_arc_levels_solved": 17,
    }
    payload.update(overrides)
    return payload


def _build(**overrides: object) -> dict:
    payload = {
        "accepted_rejected_n": {"accepted": 14, "rejected": 1782, "total": 1796},
        "honest_verdict": "complete: oracle_distinct_arc_verifier_trained_auroc_0.7790",
        "model_specs": {
            "architecture": "class_weight_balanced_standardized_logistic_regression",
            "training_recipe": "accepted_and_rejected_arc_candidates_task_held_out",
        },
        "oracle_distinct_auroc": 0.778980279,
        "oracle_distinct_auroc_ci95": [0.6146676853, 0.9174508427],
        "positive_candidate_n": 14,
        "positive_sparsity_flag": True,
        "raw_candidate_n": 8041,
        "selector_trained": True,
        "stratified_task_n": 14,
        "verifier_is_oracle": False,
        "wrong_majority_n": 5,
        "flagged_adversarial": True,
        "corrigendum_pending": [
            {
                "kind": "TAUTOLOGY",
                "severity": "critical",
                "detail": "detector_row_n and raw_candidate_n agree exactly",
            }
        ],
    }
    payload.update(overrides)
    return payload


def _gate(**overrides: object) -> dict:
    payload = {
        "headline_outcome": "oracle_distinct_verifier_ties_vote_with_headroom",
        "headroom_exists": True,
        "honest_verdict": "complete: oracle_distinct_verifier_ties_vote_with_headroom",
        "matched_control_delta": 0.0,
        "n_tasks": 14,
        "oracle_at_k": 1.0,
        "oracle_distinct_beats_vote": False,
        "pass_rates": {
            "verifier_at_1": 0.5714285714,
            "vote_at_1": 0.6428571429,
            "matched_control_at_1": 0.5714285714,
            "arbiter_override_at_1": 0.6428571429,
        },
        "status": "complete",
        "verifier_is_oracle": False,
        "verifier_minus_vote_ci95": [-0.2142857143, 0.0],
        "verifier_minus_vote_delta": -0.0714285714,
    }
    payload.update(overrides)
    return payload


def _harness(**overrides: object) -> dict:
    payload = {
        "duration_s": 14.064674,
        "harness_smoke_passed": True,
        "honest_verdict": "complete: verifier_reward_lora_harness_smoke_passed",
        "preconditions": {
            "stable_checkpoint_path": "/repo/results/verifier_reward_3arm_lora_rft/checkpoint",
            "stable_checkpoint_readable": True,
        },
        "trainable_param_count": 5701632,
        "verifier_is_oracle": True,
        "flagged_adversarial": True,
        "corrigendum_pending": [
            {
                "kind": "DURATION_TOO_SHORT",
                "severity": "critical",
                "detail": "duration_s=14.064674 but artifact references compute-bound markers",
            }
        ],
    }
    payload.update(overrides)
    return payload


def _reward(**overrides: object) -> dict:
    checkpoint = "/repo/results/verifier_reward_3arm_lora_rft/checkpoint"
    payload = {
        "arm_corpus_sizes": {"A": 776, "B": 776, "C": 742, "D": 0},
        "duration_s": 36.694545,
        "honest_verdict": "progress: accumulating_verifier_reward_training_no_eval_yet",
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
            "status": "partial",
            "per_arm": {"A": {"status": "blocked_loss_without_grad"}},
            "used_detached_process": False,
        },
        "verifier_is_oracle": True,
        "verifier_label_carries_signal": False,
        "youden_j": 0.4137931034482759,
        "flagged_adversarial": True,
        "corrigendum_pending": [
            {
                "kind": "DURATION_TOO_SHORT",
                "severity": "critical",
                "detail": "duration_s=36.694545 but artifact references compute-bound markers",
            }
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
        "retired:\n- experiment_id: 4223\n  reason: flagged\n", encoding="utf-8"
    )
    (root / "research-roadmap.yaml").write_text("milestone: 2026.06.392\n", encoding="utf-8")
    (root / "tests" / "python").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "python" / "test_pipeline_extract.py").write_text(
        "def test_pipeline_extract():\n    assert True\n", encoding="utf-8"
    )
    (root / "tests" / "python" / "test_docs.py").write_text(
        "def test_docs():\n    assert True\n", encoding="utf-8"
    )
    _write_json(root / "results" / "experiment_4229_capstone_v391.json", _capstone())
    _write_json(
        root / "results" / "experiment_4221_oracle_distinct_arc_verifier_beats_vote.json",
        _gate(),
    )
    _write_json(
        root / "results" / "experiment_4220_oracle_distinct_arc_verifier_build_labeled.json",
        _build(),
    )
    _write_json(
        root / "results" / "experiment_4222_verifier_reward_lora_harness_fix_smoke.json",
        _harness(),
    )
    _write_json(
        root / "results" / "experiment_4223_verifier_as_reward_3arm_synchronous.json",
        _reward(
            stable_checkpoint_path=str(root / "results" / "verifier_reward_3arm_lora_rft" / "checkpoint"),
            preconditions={
                "arms_n_matched": True,
                "stable_checkpoint_path": str(
                    root / "results" / "verifier_reward_3arm_lora_rft" / "checkpoint"
                ),
                "stable_checkpoint_readable": True,
            },
        ),
    )
    (root / "results" / "verifier_reward_3arm_lora_rft" / "checkpoint").mkdir(
        parents=True, exist_ok=True
    )
    return root


def run_happy(root: Path) -> dict:
    out = mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5)
    return json.loads(out.read_text(encoding="utf-8"))


def test_req_report_4230_spec_declares_contract() -> None:
    """REQ-REPORT-4230: OpenSpec declares the archive truth contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    assert "REQ-REPORT-4230" in spec
    assert "SCENARIO-REPORT-4230" in spec
    assert "SCENARIO-REPORT-4230-BLOCKED-PRECONDITION" in spec
    assert "first clean oracle-distinct frontier read" in spec
    assert "`oracle_distinct_status=TIES-VOTE-NULL`" in spec
    assert "`verifier@1-vote@1=-0.0714`" in spec
    assert "per-candidate standardized logistic regression" in spec
    assert "DURATION-flagged short-circuits" in spec
    assert "total_levels_solved=17" in spec
    assert "DiffusionGemma as" in spec
    assert mod.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert mod.FIELD_PRINCIPLES["v391_close_state"] in spec
    assert mod.FIELD_PRINCIPLES["preconditions_checked"] in spec


def test_shared_helpers_and_archive_record_editing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4230: helper behavior is deterministic and YAML-safe."""

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

    close_state = mod.build_v391_close_state(
        {
            "4229": _capstone(),
            "4221": _gate(),
            "4220": _build(),
            "4222": _harness(),
            "4223": _reward(),
        },
        root=tmp_path,
    )
    assert mod.archive_record_count(_research_complete_text(duplicates=3)) == 3
    deduped, removed, action = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=3), close_state
    )
    assert (removed, action) == (2, "deduped")
    assert mod.archive_record_count(deduped) == 1
    assert "TIES-VOTE-NULL" in deduped
    assert "UNDER-POWERED + WEAKLY-BUILT" in deduped
    assert "ARC total_levels_solved=17" in deduped
    assert mod.yaml_parses(deduped)
    updated, removed2, action2 = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=1), close_state
    )
    assert (removed2, action2) == (0, "updated")
    unchanged, removed3, action3 = mod.dedupe_or_update_record(updated, close_state)
    assert (unchanged, removed3, action3) == (updated, 0, "unchanged")
    old_activation, removed4, action4 = mod.dedupe_or_update_record(
        "milestones:\n- id: 2026.06.391\n  activation_recorded: old\n  tasks:\n  - id: exp4229\n",
        close_state,
    )
    assert (removed4, action4) == (0, "updated")
    assert "activation_recorded: exp4230-archive-v391-activate-v392" in old_activation
    appended, removed5, action5 = mod.dedupe_or_update_record(
        "# history\nmilestones:\n- id: 2026.06.390\n  finding: prior\n", close_state
    )
    assert (removed5, action5) == (0, "appended")
    assert "activation_recorded: exp4230-archive-v391-activate-v392" in appended


def test_precondition_helpers_and_source_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-4230-BLOCKED-PRECONDITION: resource probes are explicit."""

    assert mod._milestone_from_text("name: no milestone\n") == "unknown"
    assert mod.read_active_milestone(tmp_path) == ("unknown", "research-roadmap.yaml")
    (tmp_path / "research-roadmap-next.yaml").write_text(
        "milestone: 2026.06.392\n", encoding="utf-8"
    )
    assert mod.read_active_milestone(tmp_path) == ("2026.06.392", "research-roadmap-next.yaml")
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

    extra = root / "tests" / "python" / "test_extra_4230.py"
    extra.write_text("def test_extra():\n    assert True\n", encoding="utf-8")
    original_git_lines = mod._git_lines
    monkeypatch.setattr(
        mod,
        "_git_lines",
        lambda root_path, args: ["tests/python/test_extra_4230.py"] if args[:1] == ["diff"] else [],
    )
    assert "tests/python/test_extra_4230.py" in mod.smart_subset_targets(root)
    monkeypatch.setattr(mod, "_run_command", lambda command, root_path: GREEN)
    assert mod.run_smart_subset(root).exit_code == 0
    monkeypatch.setattr(
        mod,
        "_run_command",
        lambda command, root_path: mod.CommandResult(command, 0, "a.py\n\nb.py\n", ""),
    )
    monkeypatch.setattr(mod, "_git_lines", original_git_lines)
    assert mod._git_lines(root, ["diff"]) == ["a.py", "b.py"]


def test_read_sources_and_build_v391_close_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4230: close-state records the .391 first clean null."""

    root = make_repo(tmp_path)
    sources = mod.read_v391_sources(root)
    assert sources["4229"]["oracle_distinct_status"] == "TIES-VOTE-NULL"
    assert sources["4221"]["headroom_exists"] is True
    assert sources["4220"]["positive_sparsity_flag"] is True
    assert sources["4223"]["duration_s"] == 36.694545
    cited = mod.build_cited_upstream(root)
    assert {item["experiment_id"] for item in cited} == {"4229", "4221", "4220", "4222", "4223"}
    assert all(item["sha256"] is None or mod.is_sha256(item["sha256"]) for item in cited)

    state = mod.build_v391_close_state(sources, root=root)
    assert state["summary"] == "oracle_distinct_clean_ties_vote_weak_build_reward_duration_arc17"
    assert state["oracle_distinct_gate_ran"] is True
    assert state["oracle_distinct_gate_clean"] is True
    assert state["oracle_distinct_status"] == "TIES-VOTE-NULL"
    assert state["oracle_distinct_beats_vote"] is False
    assert state["verifier_is_oracle"] is False
    assert state["verifier_minus_vote_delta"] == -0.0714
    assert state["verifier_minus_vote_ci95"] == [-0.214, 0.0]
    assert state["n_tasks"] == 14
    assert state["oracle_at_k"] == 1.0
    assert state["headroom_exists"] is True
    assert state["pass_rates"]["vote_at_1"] == 0.6429
    assert state["underpowered_first_clean_read"] is True
    assert state["not_settled_refutation"] is True
    assert state["weak_build_causes"] == [
        "isolated_per_candidate_logistic_regression",
        "extreme_class_imbalance_14_positive_1782_negative",
        "held_out_gate_n14_below_clt_floor",
    ]
    assert state["build_model_type"] == "standardized_logistic_regression"
    assert state["build_architecture"] == "class_weight_balanced_standardized_logistic_regression"
    assert state["accepted_rejected_n"] == {"accepted": 14, "rejected": 1782, "total": 1796}
    assert state["base_rate"] == 0.0078
    assert state["off_fold_auroc"] == 0.779
    assert state["positive_sparsity_flag"] is True
    assert state["build_flagged_adversarial"] is True
    assert state["build_corrigendum_kinds"] == ["TAUTOLOGY"]
    assert state["reward_infra_failures"] == {
        "exp4222_smoke_duration_s": 14.0647,
        "exp4223_three_arm_duration_s": 36.6945,
        "fourth_and_fifth_infra_short_circuit": True,
    }
    assert state["reward_duration_flagged"] is True
    assert state["reward_corpora"] == {"A": 776, "B": 776, "C": 742}
    assert state["reward_youden_j"] == 0.4138
    assert state["reward_checkpoint_intact"] is True
    assert state["reward_verifier_is_oracle"] is True
    assert state["total_levels_solved"] == 17
    assert state["total_games_solved"] == 13
    assert state["live_solver_levels_completed"] == 0
    assert state["live_solver_efficiency_only_no_level"] is True
    assert state["flagged_artifacts_skipped"] == [4220, 4222, 4223]
    assert state["diffusiongemma_status"] == "STILL-PENDING"
    assert state["v392_frame"] == mod.V392_FRAME

    fallback = mod.build_v391_close_state(
        {
            "4229": _capstone(learned_arc_verifier={}, flagged_artifacts_skipped="bad"),
            "4221": _gate(pass_rates="bad", verifier_minus_vote_ci95="bad"),
            "4220": _build(accepted_rejected_n={}, corrigendum_pending="bad"),
            "4222": _harness(preconditions={}),
            "4223": _reward(arm_corpus_sizes={}),
        },
        root=tmp_path,
    )
    assert fallback["accepted_rejected_n"] == {"accepted": 14, "rejected": 1782, "total": 1796}
    assert fallback["verifier_minus_vote_ci95"] == [-0.214, 0.0]
    assert fallback["pass_rates"] == {}
    assert fallback["reward_corpora"] == {"A": 776, "B": 776, "C": 742}
    assert fallback["flagged_artifacts_skipped"] == []
    assert mod._checkpoint_present(tmp_path, "", False) is False
    relative_checkpoint = tmp_path / "relative_checkpoint"
    relative_checkpoint.mkdir()
    assert mod._checkpoint_present(tmp_path, "relative_checkpoint", False) is True


def test_run_happy_path_writes_valid_artifact_and_updates_archive(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4230: complete path writes the terminal archive artifact."""

    root = make_repo(tmp_path, duplicates=2)
    artifact = run_happy(root)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["archived_milestone"] == "2026.06.391"
    assert artifact["activated_milestone"] == "2026.06.392"
    assert artifact["active_milestone_confirmed"] == "2026.06.392"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["research_complete_record_action"] == "deduped"
    assert artifact["research_complete_duplicates_removed"] == 1
    assert artifact["v391_close_state"]["oracle_distinct_gate_ran"] is True
    assert artifact["v391_close_state"]["off_fold_auroc"] == 0.779
    assert artifact["v391_close_state"]["reward_duration_flagged"] is True
    assert artifact["v391_close_state"]["total_levels_solved"] == 17
    assert (
        artifact["field_principles"]["v391_close_state"] == mod.FIELD_PRINCIPLES["v391_close_state"]
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.is_sha256(artifact["reproducibility_checksum"])
    complete_text = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(complete_text) == 1
    assert "TIES-VOTE-NULL" in complete_text
    assert "STRENGTHEN the oracle-distinct verifier" in complete_text
    mod.validate_artifact(artifact)


def test_run_real_pretest_branch_and_entrypoints_are_injectable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-4230: real pretest and CLI entrypoints can be substituted."""

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

    import carnot.experiment_4230_archive_v391_activate_v392 as entrypoint

    called: dict[str, Path] = {}

    def fake_run(root_path: Path = mod.REPO_ROOT) -> Path:
        called["root"] = Path(root_path)
        return root / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(entrypoint, "run", fake_run)
    assert entrypoint.main() == 0
    assert called["root"] == entrypoint.REPO_ROOT

    script_path = Path("results/experiment_4230_archive_v391_activate_v392.py")
    spec = importlib.util.spec_from_file_location("exp4230_archive_script", script_path)
    assert spec and spec.loader
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "run", fake_run)
    assert script.main() == 0


def test_run_blocked_preconditions_before_research_complete_edit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4230-BLOCKED-PRECONDITION: blocked paths do not fabricate success."""

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
    (root5 / "research-roadmap.yaml").write_text("milestone: 2026.06.391\n", encoding="utf-8")
    artifact5 = json.loads(mod.run(root5, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact5["honest_verdict"] == "blocked_v392_not_active"

    missing_sources = [
        ("experiment_4229_capstone_v391.json", "blocked_v391_capstone_missing"),
        (
            "experiment_4221_oracle_distinct_arc_verifier_beats_vote.json",
            "blocked_oracle_distinct_gate_missing",
        ),
        (
            "experiment_4220_oracle_distinct_arc_verifier_build_labeled.json",
            "blocked_oracle_distinct_build_missing",
        ),
        (
            "experiment_4222_verifier_reward_lora_harness_fix_smoke.json",
            "blocked_reward_harness_missing",
        ),
        (
            "experiment_4223_verifier_as_reward_3arm_synchronous.json",
            "blocked_reward_three_arm_missing",
        ),
    ]
    for filename, reason in missing_sources:
        root_missing = make_repo(tmp_path / reason)
        (root_missing / "results" / filename).unlink()
        artifact_missing = json.loads(
            mod.run(root_missing, pretest_result=GREEN).read_text(encoding="utf-8")
        )
        assert artifact_missing["honest_verdict"] == reason


def test_run_blocked_research_complete_edit_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4230: invalid archive edits are blocked before completion."""

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
    """REQ-REPORT-4230: complete and blocked builders keep schema shape."""

    root = make_repo(tmp_path)
    state = mod.build_v391_close_state(mod.read_v391_sources(root), root=root)
    complete = mod.build_complete_artifact(
        v391_close_state=state,
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
    """REQ-REPORT-4230: validation rejects artifacts that launder the .391 truth."""

    good = run_happy(make_repo(tmp_path))

    def set_path(obj: dict, path: list[str], value: object) -> None:
        cur = obj
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = value

    cases = [
        ("missing required fields", lambda a: a.pop("v391_close_state")),
        ("terminal-prefixed", lambda a: a.__setitem__("honest_verdict", "done")),
        ("field_principles must be a mapping", lambda a: a.__setitem__("field_principles", "x")),
        ("missing field principles", lambda a: a.__setitem__("field_principles", {})),
        (
            "principle must match REQ-REPORT-4230",
            lambda a: a["field_principles"].__setitem__("v391_close_state", "wrong"),
        ),
        ("archived milestone", lambda a: a.__setitem__("archived_milestone", "2026.06.999")),
        ("activated milestone", lambda a: a.__setitem__("activated_milestone", "2026.06.999")),
        ("research-complete YAML", lambda a: a.__setitem__("research_complete_yaml_parses", False)),
        ("exclusion manifest", lambda a: a.__setitem__("exclusion_manifest_parses", False)),
        ("pretest suite", lambda a: a.__setitem__("pretest_suite_green", False)),
        ("active milestone", lambda a: a.__setitem__("active_milestone_confirmed", "2026.06.391")),
        ("v391_close_state must be a mapping", lambda a: a.__setitem__("v391_close_state", "x")),
        ("gate ran", lambda a: set_path(a, ["v391_close_state", "oracle_distinct_gate_ran"], False)),
        (
            "status",
            lambda a: set_path(a, ["v391_close_state", "oracle_distinct_status"], "MOAT-WON"),
        ),
        (
            "delta",
            lambda a: set_path(a, ["v391_close_state", "verifier_minus_vote_delta"], 0.1),
        ),
        (
            "CI",
            lambda a: set_path(a, ["v391_close_state", "verifier_minus_vote_ci95"], [0.0, 0.1]),
        ),
        ("n_tasks", lambda a: set_path(a, ["v391_close_state", "n_tasks"], 30)),
        ("oracle@K", lambda a: set_path(a, ["v391_close_state", "oracle_at_k"], 0.5)),
        ("headroom", lambda a: set_path(a, ["v391_close_state", "headroom_exists"], False)),
        (
            "refutation",
            lambda a: set_path(a, ["v391_close_state", "not_settled_refutation"], False),
        ),
        (
            "accepted",
            lambda a: set_path(a, ["v391_close_state", "accepted_rejected_n"], {"total": 1}),
        ),
        ("AUROC", lambda a: set_path(a, ["v391_close_state", "off_fold_auroc"], 0.5)),
        ("base-rate", lambda a: set_path(a, ["v391_close_state", "base_rate"], 0.5)),
        (
            "TAUTOLOGY",
            lambda a: set_path(a, ["v391_close_state", "build_corrigendum_kinds"], []),
        ),
        (
            "reward duration",
            lambda a: set_path(a, ["v391_close_state", "reward_duration_flagged"], False),
        ),
        (
            "reward corpora",
            lambda a: set_path(a, ["v391_close_state", "reward_corpora"], {"A": 1, "B": 1, "C": 1}),
        ),
        ("Youden", lambda a: set_path(a, ["v391_close_state", "reward_youden_j"], 0.1)),
        ("checkpoint", lambda a: set_path(a, ["v391_close_state", "reward_checkpoint_intact"], False)),
        (
            "reward oracle",
            lambda a: set_path(a, ["v391_close_state", "reward_verifier_is_oracle"], False),
        ),
        ("ARC levels", lambda a: set_path(a, ["v391_close_state", "total_levels_solved"], 16)),
        ("ARC games", lambda a: set_path(a, ["v391_close_state", "total_games_solved"], 12)),
        (
            "live",
            lambda a: set_path(
                a, ["v391_close_state", "live_solver_efficiency_only_no_level"], False
            ),
        ),
        (
            "flagged",
            lambda a: set_path(a, ["v391_close_state", "flagged_artifacts_skipped"], [4220]),
        ),
        (
            "DiffusionGemma",
            lambda a: set_path(a, ["v391_close_state", "diffusiongemma_status"], "MET"),
        ),
        ("v392 frame", lambda a: set_path(a, ["v391_close_state", "v392_frame"], "redo")),
    ]
    for label, mutate in cases:
        artifact = copy.deepcopy(good)
        mutate(artifact)
        with pytest.raises(ValueError, match=label):
            mod.validate_artifact(artifact)
