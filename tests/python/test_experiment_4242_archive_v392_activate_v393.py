"""Tests for Exp 4242 `.392` archive / `.393` activation.

Spec refs: REQ-REPORT-4242, SCENARIO-REPORT-4242,
SCENARIO-REPORT-4242-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v392_activate_v393_4242 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="81 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.391\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.392\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-15'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4241-capstone-v392\n"
        "    result: OK\n"
    )
    return head + block * duplicates


def _capstone(**overrides: object) -> dict:
    payload = {
        "experiment_id": 4241,
        "honest_verdict": (
            "complete: capstone_v392_oracle_distinct_arc_null_is_data_sparsity_code_wins_"
            "oracle_ARC-NULL-IS-DATA-SPARSITY_reward_HARNESS-DEFERRED_arc_levels18_"
            "flagged_skipped2_diffusiongemma_resolvable"
        ),
        "headline_outcome": "oracle_distinct_arc_null_is_data_sparsity_code_wins",
        "oracle_distinct_status": "ARC-NULL-IS-DATA-SPARSITY",
        "code_disambiguation": {
            "ci95_excludes_zero": True,
            "code_oracle_distinct_beats_vote": True,
            "code_predictor_minus_vote_ci95": [0.00625, 0.0625],
            "code_predictor_minus_vote_delta": 0.03125,
            "code_status": "CODE-WON",
            "gate_ran": True,
            "headroom_present": True,
            "held_out_task_n": 160,
            "matched_control_delta": 0.00625,
            "off_fold_auroc": 0.9739318159,
            "oracle_at_k": 0.9625,
            "pass_rates": {
                "matched_control_at_1": 0.94375,
                "predictor_at_1": 0.95,
                "vote_at_1": 0.91875,
            },
            "verifier_is_oracle": False,
        },
        "arc_aggregator_gate": {
            "aggregator_minus_vote_ci95": [0.0, 0.0],
            "aggregator_minus_vote_delta": 0.0,
            "arc_status": "TIES-AT-POWER-NULL",
            "candidate_count": 28419,
            "gate_ran": True,
            "headroom_present": True,
            "held_out_task_n": 52,
            "oracle_at_k": 0.3653846154,
            "oracle_distinct_beats_vote": False,
            "pass_rates": {
                "aggregator_at_1": 0.1923076923,
                "margin_override_at_1": 0.1923076923,
                "matched_control_at_1": 0.1538461538,
                "vote_at_1": 0.1923076923,
            },
            "verifier_is_oracle": False,
            "wrong_majority_n": 9,
        },
        "arc_aggregator_model": {
            "accepted_rejected_n": {"accepted": 20, "rejected": 28399, "total": 28419},
            "build_artifact_status": "skipped_flagged_adversarial",
            "held_out_task_n": 52,
            "model_type": "standardized_logistic_regression_isotonic_calibrated",
            "off_fold_auroc": 0.8397117856262545,
            "oof_row_n": 28419,
            "verifier_is_oracle": False,
            "wrong_majority_n": 0,
        },
        "verifier_as_reward": {
            "b1_real_training_smoke": {
                "harness_smoke_passed": False,
                "status": "skipped_flagged_adversarial",
            },
            "blocked_at_layer": "conductor_pre_gate",
            "honest_verdict": "blocked_gate_check_failed",
            "live_lora_retired": False,
            "status": "included",
            "verifier_as_reward_status": "HARNESS-DEFERRED",
        },
        "arc_progress": {
            "honest_verdict": "success: incremental_progress_sc25-635fd71a_advanced_to_L4_total18",
            "levels_completed": 4,
            "total_arc_games_solved": 13,
            "total_arc_levels_solved": 18,
        },
        "live_solver_accuracy": {
            "honest_verdict": "complete: solver_completes_0_levels_live_lp85-305b61c3_efficiency_only",
            "levels_completed": 0,
            "solver_beats_floor_accuracy": False,
            "solver_beats_floor_efficiency": True,
        },
        "flagged_artifacts_skipped": [
            {"experiment_id": 4231, "reason": "flagged_adversarial:true"},
            {"experiment_id": 4234, "reason": "flagged_adversarial:true"},
        ],
        "sota_v393": {
            "flagged_for_v393": "bigger_arc_pool_full_set_encoder_agglm_aggregator_v393",
            "strongest_method_name": "Set-Encoder full cross-candidate attention",
        },
        "total_arc_levels_solved": 18,
        "diffusiongemma_gate_resolvable": True,
    }
    payload.update(overrides)
    return payload


def _code(**overrides: object) -> dict:
    payload = {
        "adversarial_verify": {
            "circular_moat_overclaim_clean": True,
            "flag_count": 0,
            "status": "clean",
        },
        "candidate_pool": {"candidate_n": 2294, "pass_rate": 0.6469049695, "positive_n": 1484},
        "ci95_excludes_zero": True,
        "code_oracle_distinct_beats_vote": True,
        "code_predictor_minus_vote_ci95": [0.00625, 0.0625],
        "code_predictor_minus_vote_delta": 0.03125,
        "headroom_exists": True,
        "held_out_task_n": 160,
        "honest_verdict": "complete: code_oracle_distinct_beats_vote",
        "matched_control_delta": 0.00625,
        "off_fold_auroc": 0.9739318159,
        "oracle_at_k": 0.9625,
        "pass_rates": {"matched_control_at_1": 0.94375, "predictor_at_1": 0.95, "vote_at_1": 0.91875},
        "status": "complete",
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _arc_gate(**overrides: object) -> dict:
    payload = {
        "aggregator_minus_vote_ci95": [0.0, 0.0],
        "aggregator_minus_vote_delta": 0.0,
        "candidate_count": 28419,
        "headroom_exists": True,
        "held_out_task_n": 52,
        "honest_verdict": "complete: oracle_distinct_aggregator_ties_vote_with_headroom_at_power",
        "oracle_at_k": 0.3653846154,
        "oracle_distinct_beats_vote": False,
        "pass_rates": {
            "aggregator_at_1": 0.1923076923,
            "margin_override_at_1": 0.1923076923,
            "matched_control_at_1": 0.1538461538,
            "vote_at_1": 0.1923076923,
        },
        "status": "complete",
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _lora_smoke(**overrides: object) -> dict:
    payload = {
        "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        "duration_s": 17.71837,
        "flagged_adversarial": True,
        "harness_smoke_passed": False,
        "honest_verdict": "blocked_lora_training_cannot_run_in_window",
        "preconditions": {
            "corpus_sizes": {"A": 776, "B": 776, "C": 742},
            "stable_checkpoint_readable": True,
        },
        "verifier_is_oracle": True,
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
        "retired:\n- experiment_id: 4234\n  reason: flagged\n", encoding="utf-8"
    )
    (root / "research-roadmap.yaml").write_text("milestone: 2026.06.393\n", encoding="utf-8")
    (root / "tests" / "python").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "python" / "test_pipeline_extract.py").write_text(
        "def test_pipeline_extract():\n    assert True\n", encoding="utf-8"
    )
    (root / "tests" / "python" / "test_docs.py").write_text(
        "def test_docs():\n    assert True\n", encoding="utf-8"
    )
    _write_json(root / "results" / "experiment_4241_capstone_v392.json", _capstone())
    _write_json(root / "results" / "experiment_4233_oracle_distinct_code_beats_vote.json", _code())
    _write_json(
        root / "results" / "experiment_4232_oracle_distinct_arc_aggregator_beats_vote.json",
        _arc_gate(),
    )
    _write_json(
        root / "results" / "experiment_4234_verifier_reward_lora_harness_real_training_smoke.json",
        _lora_smoke(),
    )
    return root


def run_happy(root: Path) -> dict:
    out = mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5)
    return json.loads(out.read_text(encoding="utf-8"))


def test_req_report_4242_spec_declares_contract() -> None:
    """REQ-REPORT-4242: OpenSpec declares the archive truth contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    assert "REQ-REPORT-4242" in spec
    assert "SCENARIO-REPORT-4242" in spec
    assert "SCENARIO-REPORT-4242-BLOCKED-PRECONDITION" in spec
    assert "first oracle-distinct win on CODE" in spec
    assert "`predictor@1-vote@1=+0.03125`" in spec
    assert "`ARC-NULL-IS-DATA-SPARSITY`" in spec
    assert "`20` positive" in spec
    assert "`HARNESS-DEFERRED`" in spec
    assert "DiffusionGemma as resolvable on the code win while ARC still ties" in spec
    assert mod.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert mod.FIELD_PRINCIPLES["v392_close_state"] in spec
    assert mod.FIELD_PRINCIPLES["preconditions_checked"] in spec


def test_helpers_and_archive_record_editing(tmp_path: Path) -> None:
    """REQ-REPORT-4242: helper behavior is deterministic and YAML-safe."""

    assert mod.yaml_parses("a: 1\n") is True
    assert mod.yaml_parses("a: : :\n- [\n") is False
    assert mod.duration_from(None, None) == 0.0001
    assert mod.payload_checksum({"a": 1}) == mod.payload_checksum(
        {"a": 1, "reproducibility_checksum": "old"}
    )
    assert mod.is_sha256("a" * 64) is True
    assert mod.is_sha256("z" * 64) is False
    out = tmp_path / "artifact.json"
    mod.write_payload(out, {"b": 2, "a": 1})
    assert out.read_text(encoding="utf-8").startswith('{\n  "a"')
    assert mod.read_active_milestone(tmp_path) == ("unknown", "research-roadmap.yaml")

    close_state = mod.build_v392_close_state(
        {
            "4241": _capstone(),
            "4233": _code(),
            "4232": _arc_gate(),
            "4234": _lora_smoke(),
        }
    )
    assert mod.archive_record_count(_research_complete_text(duplicates=3)) == 3
    deduped, removed, action = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=3), close_state
    )
    assert (removed, action) == (2, "deduped")
    assert mod.archive_record_count(deduped) == 1
    assert "CODE-WON" in deduped
    assert "DATA-SPARSITY" in deduped
    assert "ARC total_levels_solved=18" in deduped
    assert mod.yaml_parses(deduped)
    updated, removed2, action2 = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=1), close_state
    )
    assert (removed2, action2) == (0, "updated")
    unchanged, removed3, action3 = mod.dedupe_or_update_record(updated, close_state)
    assert (unchanged, removed3, action3) == (updated, 0, "unchanged")
    appended, removed4, action4 = mod.dedupe_or_update_record(
        "# history\nmilestones:\n- id: 2026.06.391\n  finding: prior\n", close_state
    )
    assert (removed4, action4) == (0, "appended")
    assert "activation_recorded: exp4242-archive-v392-activate-v393" in appended
    no_tasks = mod._insert_before_tasks(["  title: no tasks"], "  finding: x")
    assert no_tasks == ["  title: no tasks", "  finding: x"]
    added_finding, removed5, action5 = mod.dedupe_or_update_record(
        "milestones:\n- id: 2026.06.392\n  title: missing finding\n  tasks:\n  - id: exp4241\n",
        close_state,
    )
    assert (removed5, action5) == (0, "updated")
    assert "FIRST oracle-distinct CODE win" in added_finding


def test_read_sources_and_build_v392_close_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4242: close-state records the CODE win and ARC data-sparsity tie."""

    root = make_repo(tmp_path)
    sources = mod.read_v392_sources(root)
    assert sources["4241"]["oracle_distinct_status"] == "ARC-NULL-IS-DATA-SPARSITY"
    assert sources["4233"]["code_oracle_distinct_beats_vote"] is True
    assert sources["4232"]["held_out_task_n"] == 52
    assert sources["4234"]["harness_smoke_passed"] is False
    cited = mod.build_cited_upstream(root)
    assert {item["experiment_id"] for item in cited} == {"4241", "4233", "4232", "4234"}
    assert all(item["sha256"] is None or mod.is_sha256(item["sha256"]) for item in cited)

    state = mod.build_v392_close_state(sources)
    assert state["summary"] == "code_oracle_distinct_won_arc_data_sparsity_tie_reward_deferred_arc18"
    assert state["headline_outcome"] == "oracle_distinct_arc_null_is_data_sparsity_code_wins"
    assert state["oracle_distinct_status"] == "ARC-NULL-IS-DATA-SPARSITY"
    assert state["code_status"] == "CODE-WON"
    assert state["code_oracle_distinct_beats_vote"] is True
    assert state["code_predictor_minus_vote_delta"] == 0.03125
    assert state["code_predictor_minus_vote_ci95"] == [0.00625, 0.0625]
    assert state["code_ci95_excludes_zero"] is True
    assert state["code_oracle_at_k"] == 0.9625
    assert state["code_off_fold_auroc"] == 0.974
    assert state["code_held_out_task_n"] == 160
    assert state["code_verifier_is_oracle"] is False
    assert state["code_adversarial_clean"] is True
    assert state["arc_status"] == "TIES-AT-POWER-NULL"
    assert state["arc_aggregator_minus_vote_delta"] == 0.0
    assert state["arc_aggregator_minus_vote_ci95"] == [0.0, 0.0]
    assert state["arc_held_out_task_n"] == 52
    assert state["arc_oracle_at_k"] == 0.3654
    assert state["arc_headroom_present"] is True
    assert state["arc_data_sparsity_diagnosis"] is True
    assert state["arc_accepted_rejected_n"] == {"accepted": 20, "rejected": 28399, "total": 28419}
    assert state["arc_base_rate"] == 0.0007
    assert state["verifier_as_reward_status"] == "HARNESS-DEFERRED"
    assert state["exp4234_honest_verdict"] == "blocked_lora_training_cannot_run_in_window"
    assert state["exp4234_flagged_adversarial"] is True
    assert state["exp4235_blocked_at_layer"] == "conductor_pre_gate"
    assert state["live_lora_retired"] is False
    assert state["auto_retire_never_fired_because_b2_never_ran"] is True
    assert state["total_levels_solved"] == 18
    assert state["total_games_solved"] == 13
    assert state["live_solver_levels_completed"] == 0
    assert state["live_solver_efficiency_only_no_level"] is True
    assert state["flagged_artifacts_skipped"] == [4231, 4234]
    assert state["diffusiongemma_gate_resolvable_on_code"] is True
    assert state["arc_still_ties"] is True
    assert state["v393_frame"] == mod.V393_FRAME

    fallback = mod.build_v392_close_state(
        {
            "4241": _capstone(code_disambiguation="bad", flagged_artifacts_skipped="bad"),
            "4233": _code(pass_rates="bad", code_predictor_minus_vote_ci95="bad"),
            "4232": _arc_gate(pass_rates="bad", aggregator_minus_vote_ci95="bad"),
            "4234": _lora_smoke(preconditions={}),
        }
    )
    assert fallback["code_predictor_minus_vote_ci95"] == [0.00625, 0.0625]
    assert fallback["code_pass_rates"] == {}
    assert fallback["arc_aggregator_minus_vote_ci95"] == [0.0, 0.0]
    assert fallback["arc_pass_rates"] == {}
    assert fallback["flagged_artifacts_skipped"] == []


def test_run_happy_path_writes_valid_artifact_and_updates_archive(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4242: complete path writes the terminal archive artifact."""

    root = make_repo(tmp_path, duplicates=2)
    artifact = run_happy(root)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["archived_milestone"] == "2026.06.392"
    assert artifact["activated_milestone"] == "2026.06.393"
    assert artifact["active_milestone_confirmed"] == "2026.06.393"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["research_complete_record_action"] == "deduped"
    assert artifact["research_complete_duplicates_removed"] == 1
    assert artifact["v392_close_state"]["code_status"] == "CODE-WON"
    assert artifact["v392_close_state"]["arc_data_sparsity_diagnosis"] is True
    assert artifact["v392_close_state"]["total_levels_solved"] == 18
    assert (
        artifact["field_principles"]["v392_close_state"] == mod.FIELD_PRINCIPLES["v392_close_state"]
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.is_sha256(artifact["reproducibility_checksum"])
    complete_text = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(complete_text) == 1
    assert "FIRST oracle-distinct CODE win" in complete_text
    assert "LAND the ARC oracle-distinct win" in complete_text
    mod.validate_artifact(artifact)


def test_run_blocked_preconditions_before_research_complete_edit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4242-BLOCKED-PRECONDITION: blocked paths do not fabricate success."""

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
    (root5 / "research-roadmap.yaml").write_text("milestone: 2026.06.392\n", encoding="utf-8")
    artifact5 = json.loads(mod.run(root5, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact5["honest_verdict"] == "blocked_v393_not_active"

    missing_sources = [
        ("experiment_4241_capstone_v392.json", "blocked_v392_capstone_missing"),
        ("experiment_4233_oracle_distinct_code_beats_vote.json", "blocked_code_gate_missing"),
        (
            "experiment_4232_oracle_distinct_arc_aggregator_beats_vote.json",
            "blocked_arc_gate_missing",
        ),
        (
            "experiment_4234_verifier_reward_lora_harness_real_training_smoke.json",
            "blocked_lora_smoke_missing",
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
    """REQ-REPORT-4242: invalid archive edits are blocked before completion."""

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
    """REQ-REPORT-4242: schema validation and entrypoints preserve the contract."""

    root = make_repo(tmp_path)
    state = mod.build_v392_close_state(mod.read_v392_sources(root))
    complete = mod.build_complete_artifact(
        v392_close_state=state,
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

    import carnot.experiment_4242_archive_v392_activate_v393 as entrypoint

    called: dict[str, Path] = {}

    def fake_run(root_path: Path = mod.REPO_ROOT) -> Path:
        called["root"] = Path(root_path)
        return root / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(entrypoint, "run", fake_run)
    assert entrypoint.main() == 0
    assert called["root"] == entrypoint.REPO_ROOT

    script_path = Path("results/experiment_4242_archive_v392_activate_v393.py")
    spec = importlib.util.spec_from_file_location("exp4242_archive_script", script_path)
    assert spec and spec.loader
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "run", fake_run)
    assert script.main() == 0


def test_validate_artifact_rejection_guards(tmp_path: Path) -> None:
    """REQ-REPORT-4242: validation rejects artifacts that launder the .392 truth."""

    good = run_happy(make_repo(tmp_path))

    def set_path(obj: dict, path: list[str], value: object) -> None:
        cur = obj
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = value

    cases = [
        ("missing required fields", lambda a: a.pop("v392_close_state")),
        ("terminal-prefixed", lambda a: a.__setitem__("honest_verdict", "done")),
        ("field_principles must be a mapping", lambda a: a.__setitem__("field_principles", "x")),
        ("missing field principles", lambda a: a.__setitem__("field_principles", {})),
        (
            "principle must match REQ-REPORT-4242",
            lambda a: a["field_principles"].__setitem__("v392_close_state", "wrong"),
        ),
        ("archived milestone", lambda a: a.__setitem__("archived_milestone", "2026.06.999")),
        ("activated milestone", lambda a: a.__setitem__("activated_milestone", "2026.06.999")),
        ("research-complete YAML", lambda a: a.__setitem__("research_complete_yaml_parses", False)),
        ("exclusion manifest", lambda a: a.__setitem__("exclusion_manifest_parses", False)),
        ("pretest suite", lambda a: a.__setitem__("pretest_suite_green", False)),
        ("active milestone", lambda a: a.__setitem__("active_milestone_confirmed", "2026.06.392")),
        ("v392_close_state must be a mapping", lambda a: a.__setitem__("v392_close_state", "x")),
        ("code status", lambda a: set_path(a, ["v392_close_state", "code_status"], "TIED")),
        (
            "code delta",
            lambda a: set_path(a, ["v392_close_state", "code_predictor_minus_vote_delta"], 0.0),
        ),
        ("code CI", lambda a: set_path(a, ["v392_close_state", "code_predictor_minus_vote_ci95"], [0, 0])),
        ("code n", lambda a: set_path(a, ["v392_close_state", "code_held_out_task_n"], 159)),
        ("code oracle", lambda a: set_path(a, ["v392_close_state", "code_verifier_is_oracle"], True)),
        ("code clean", lambda a: set_path(a, ["v392_close_state", "code_adversarial_clean"], False)),
        ("ARC status", lambda a: set_path(a, ["v392_close_state", "oracle_distinct_status"], "CODE-WON")),
        ("ARC delta", lambda a: set_path(a, ["v392_close_state", "arc_aggregator_minus_vote_delta"], 0.1)),
        ("ARC n", lambda a: set_path(a, ["v392_close_state", "arc_held_out_task_n"], 14)),
        ("data sparsity", lambda a: set_path(a, ["v392_close_state", "arc_data_sparsity_diagnosis"], False)),
        (
            "accepted",
            lambda a: set_path(a, ["v392_close_state", "arc_accepted_rejected_n"], {"total": 1}),
        ),
        ("base-rate", lambda a: set_path(a, ["v392_close_state", "arc_base_rate"], 0.01)),
        ("reward", lambda a: set_path(a, ["v392_close_state", "verifier_as_reward_status"], "DONE")),
        ("live LoRA", lambda a: set_path(a, ["v392_close_state", "live_lora_retired"], True)),
        ("ARC levels", lambda a: set_path(a, ["v392_close_state", "total_levels_solved"], 17)),
        ("ARC games", lambda a: set_path(a, ["v392_close_state", "total_games_solved"], 12)),
        ("live", lambda a: set_path(a, ["v392_close_state", "live_solver_efficiency_only_no_level"], False)),
        ("flagged", lambda a: set_path(a, ["v392_close_state", "flagged_artifacts_skipped"], [4231])),
        (
            "DiffusionGemma",
            lambda a: set_path(a, ["v392_close_state", "diffusiongemma_gate_resolvable_on_code"], False),
        ),
        ("v393 frame", lambda a: set_path(a, ["v392_close_state", "v393_frame"], "redo")),
    ]
    for label, mutate in cases:
        artifact = copy.deepcopy(good)
        mutate(artifact)
        with pytest.raises(ValueError, match=label):
            mod.validate_artifact(artifact)
