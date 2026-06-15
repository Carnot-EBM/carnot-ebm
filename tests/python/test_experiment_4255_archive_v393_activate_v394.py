"""Tests for Exp 4255 `.393` archive / `.394` activation.

Spec refs: REQ-REPORT-4255, SCENARIO-REPORT-4255,
SCENARIO-REPORT-4255-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v393_activate_v394_4255 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="81 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.392\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.393\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-15'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4254-capstone-v393\n"
        "    result: OK\n"
    )
    return head + block * duplicates


def _capstone(**overrides: object) -> dict:
    payload = {
        "experiment_id": 4254,
        "honest_verdict": (
            "complete: capstone_v393_arc_oracle_distinct_set_encoder_beats_vote_first_arc_win_"
            "oracle_ARC-MOAT-WON_reward_LIVE-LORA-RETIRED-OFFLINE-PENDING_arc_levels19_"
            "flagged_skipped1_diffusiongemma_resolvable"
        ),
        "headline_outcome": "arc_oracle_distinct_set_encoder_beats_vote_first_arc_win",
        "oracle_distinct_status": "ARC-MOAT-WON",
        "verifier_as_reward_status": "LIVE-LORA-RETIRED-OFFLINE-PENDING",
        "diffusiongemma_gate_resolvable": True,
        "total_arc_levels_solved": 19,
        "arc_set_encoder_gate": {
            "arc_status": "ARC-MOAT-WON",
            "ci95_excludes_zero": True,
            "gate_ran": True,
            "headroom_present": True,
            "held_out_task_n": 52,
            "matched_control_delta": 0.4807692308,
            "matched_control_present": True,
            "oracle_at_k": 0.8269230769,
            "oracle_distinct_beats_vote": True,
            "pass_rates": {
                "matched_control_at_1": 0.2115384615,
                "set_encoder_at_1": 0.6923076923,
                "vote_at_1": 0.25,
            },
            "set_encoder_minus_vote_ci95": [0.3076923077, 0.5961538462],
            "set_encoder_minus_vote_delta": 0.4423076923,
            "verifier_is_oracle": False,
            "wrong_majority_n": 30,
        },
        "code_replication": {
            "code_replication_beats_vote": False,
            "code_status": "BLOCKED",
            "gate_ran": False,
            "held_out_task_n": 0,
            "honest_verdict": "blocked_code_second_corpus_missing",
            "replication_read": "blocked_code_second_corpus_missing",
            "verifier_is_oracle": False,
        },
        "verifier_as_reward": {
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: exp4247-verifier-reward-offline-"
                "harness-retire-livelora.harness_smoke_passed"
            ),
            "honest_verdict": "blocked_gate_check_failed",
            "live_lora_retired_recorded": True,
            "offline_a_vs_b_ran": False,
            "retirement_artifact_skipped": True,
            "retirement_artifact_status": "skipped_flagged_adversarial",
            "verifier_as_reward_status": "LIVE-LORA-RETIRED-OFFLINE-PENDING",
        },
        "arc_progress": {
            "honest_verdict": "success: incremental_progress_sc25-635fd71a_advanced_to_L5_total19",
            "levels_completed": 5,
            "new_levels_solved_this_task": 1,
            "prior_total_levels_solved": 18,
            "total_arc_games_solved": 13,
            "total_arc_levels_solved": 19,
        },
        "live_solver_accuracy": {
            "honest_verdict": "complete: solver_completes_0_levels_live_lp85-305b61c3_efficiency_only",
            "levels_completed": 0,
            "solver_beats_floor_accuracy": False,
            "solver_beats_floor_efficiency": True,
            "solver_completes_level": False,
        },
        "sota_v394": {
            "flagged_for_v394": "agglm_synthesize_corrected_grid_from_set_encoder_evidence_v394",
            "strongest_method_name": "Set-LLM permutation-invariant set architecture",
        },
        "flagged_artifacts_skipped": [
            {"experiment_id": 4247, "reason": "flagged_adversarial:true"}
        ],
    }
    payload.update(overrides)
    return payload


def _arc_win(**overrides: object) -> dict:
    payload = {
        "headroom_exists": True,
        "held_out_task_n": 52,
        "honest_verdict": "complete: arc_oracle_distinct_set_encoder_beats_vote",
        "oracle_at_k": 0.8269230769,
        "pass_rates": {
            "matched_control_at_1": 0.2115384615,
            "set_encoder_at_1": 0.6923076923,
            "vote_at_1": 0.25,
        },
        "random_seed": 4245,
        "set_encoder_minus_vote_ci95": [0.3076923077, 0.5961538462],
        "set_encoder_minus_vote_delta": 0.4423076923,
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _build(**overrides: object) -> dict:
    payload = {
        "aggregator_trained": True,
        "held_out_task_n": 52,
        "honest_verdict": "complete_arc_set_encoder_no_gain_over_logistic_auroc0.9633",
        "logistic_auroc": 0.9795019663,
        "oracle_distinct_auroc": 0.9633173387,
        "positive_candidate_n": 48,
        "set_encoder_vs_logistic_auroc_delta": -0.0161846276,
        "verifier_is_oracle": False,
        "wrong_majority_n": 30,
    }
    payload.update(overrides)
    return payload


def _code_replication(**overrides: object) -> dict:
    payload = {
        "code_replication_beats_vote": False,
        "held_out_task_n": 0,
        "honest_verdict": "blocked_code_second_corpus_missing",
        "replication_read": "blocked_code_second_corpus_missing",
        "status": "complete",
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _reward_retire(**overrides: object) -> dict:
    payload = {
        "corrigendum_pending": [
            {
                "kind": "DURATION_TOO_SHORT",
                "severity": "critical",
                "detail": "duration_s=12.7 but artifact references compute-bound markers",
            }
        ],
        "flagged_adversarial": True,
        "honest_verdict": "blocked_offline_reward_weighted_training_cannot_run_in_window",
        "live_lora_retired": True,
        "preconditions": {"stable_checkpoint_readable": True},
        "verifier_is_oracle": True,
    }
    payload.update(overrides)
    return payload


def _arc_progress(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "success: incremental_progress_sc25-635fd71a_advanced_to_L5_total19",
        "levels_completed": 5,
        "new_levels_solved_this_task": 1,
        "prior_total_levels_solved": 18,
        "real_env_confirmed": True,
        "total_games_solved": 13,
        "total_levels_solved": 19,
    }
    payload.update(overrides)
    return payload


def _live_solver(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: solver_completes_0_levels_live_lp85-305b61c3_efficiency_only",
        "live_env_metrics": {"levels_completed": 0, "score": 0.0},
        "solver_beats_floor": {
            "accuracy": {"beats": False, "solver_levels_completed": 0},
            "efficiency": {"beats": True, "solver_actions": 5},
        },
        "solver_completes_level": False,
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
        "retired:\n- experiment_id: 4247\n  reason: flagged\n", encoding="utf-8"
    )
    (root / "research-roadmap.yaml").write_text("milestone: 2026.06.394\n", encoding="utf-8")
    (root / "tests" / "python").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "python" / "test_pipeline_extract.py").write_text(
        "def test_pipeline_extract():\n    assert True\n", encoding="utf-8"
    )
    (root / "tests" / "python" / "test_docs.py").write_text(
        "def test_docs():\n    assert True\n", encoding="utf-8"
    )
    _write_json(root / "results" / "experiment_4254_capstone_v393.json", _capstone())
    _write_json(root / "results" / "experiment_4245_arc_set_encoder_beats_vote.json", _arc_win())
    _write_json(
        root / "results" / "experiment_4244_arc_set_encoder_aggregator_build.json",
        _build(),
    )
    _write_json(
        root / "results" / "experiment_4246_code_oracle_distinct_replication.json",
        _code_replication(),
    )
    _write_json(
        root / "results" / "experiment_4247_verifier_reward_offline_harness_retire_livelora.json",
        _reward_retire(),
    )
    _write_json(root / "results" / "experiment_4249_arc_incremental_progress.json", _arc_progress())
    _write_json(
        root / "results" / "experiment_4250_arc_live_env_solver_accuracy.json",
        _live_solver(),
    )
    return root


def run_happy(root: Path) -> dict:
    out = mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5)
    return json.loads(out.read_text(encoding="utf-8"))


def test_req_report_4255_spec_declares_contract() -> None:
    """REQ-REPORT-4255: OpenSpec declares the .393 close-state truth contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    assert "REQ-REPORT-4255" in spec
    assert "SCENARIO-REPORT-4255" in spec
    assert "SCENARIO-REPORT-4255-BLOCKED-PRECONDITION" in spec
    assert "first ARC oracle-distinct win" in spec
    assert "`set_encoder@1-vote@1=+0.4423`" in spec
    assert "single-seed on `n=52`" in spec
    assert "provenance-leak risk" in spec
    assert "rather than the set-encoder architecture" in spec
    assert "blocked_code_second_corpus_missing" in spec
    assert "seventh verifier-as-reward failure" in spec
    assert mod.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert mod.FIELD_PRINCIPLES["v393_close_state"] in spec
    assert mod.FIELD_PRINCIPLES["preconditions_checked"] in spec


def test_helpers_and_archive_record_editing(tmp_path: Path) -> None:
    """REQ-REPORT-4255: helper behavior is deterministic and YAML-safe."""

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

    close_state = mod.build_v393_close_state(
        {
            "4254": _capstone(),
            "4245": _arc_win(),
            "4244": _build(),
            "4246": _code_replication(),
            "4247": _reward_retire(),
            "4249": _arc_progress(),
            "4250": _live_solver(),
        }
    )
    assert mod.archive_record_count(_research_complete_text(duplicates=3)) == 3
    deduped, removed, action = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=3), close_state
    )
    assert (removed, action) == (2, "deduped")
    assert mod.archive_record_count(deduped) == 1
    assert "FIRST ARC oracle-distinct win" in deduped
    assert "HARDEN the win" in deduped
    assert mod.yaml_parses(deduped)
    updated, removed2, action2 = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=1), close_state
    )
    assert (removed2, action2) == (0, "updated")
    unchanged, removed3, action3 = mod.dedupe_or_update_record(updated, close_state)
    assert (unchanged, removed3, action3) == (updated, 0, "unchanged")
    appended, removed4, action4 = mod.dedupe_or_update_record(
        "# history\nmilestones:\n- id: 2026.06.392\n  finding: prior\n", close_state
    )
    assert (removed4, action4) == (0, "appended")
    assert "activation_recorded: exp4255-archive-v393-activate-v394" in appended
    no_tasks = mod._insert_before_tasks(["  title: no tasks"], "  finding: x")
    assert no_tasks == ["  title: no tasks", "  finding: x"]
    added_finding, removed5, action5 = mod.dedupe_or_update_record(
        "milestones:\n- id: 2026.06.393\n  title: missing finding\n  tasks:\n  - id: exp4254\n",
        close_state,
    )
    assert (removed5, action5) == (0, "updated")
    assert "FIRST ARC oracle-distinct win" in added_finding


def test_read_sources_and_build_v393_close_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4255: close-state records the ARC win and caveats."""

    root = make_repo(tmp_path)
    sources = mod.read_v393_sources(root)
    assert sources["4254"]["oracle_distinct_status"] == "ARC-MOAT-WON"
    assert sources["4245"]["set_encoder_minus_vote_delta"] == 0.4423076923
    assert sources["4244"]["oracle_distinct_auroc"] == 0.9633173387
    assert sources["4247"]["live_lora_retired"] is True
    cited = mod.build_cited_upstream(root)
    assert {item["experiment_id"] for item in cited} == {
        "4254",
        "4245",
        "4244",
        "4246",
        "4247",
        "4249",
        "4250",
    }
    assert all(item["sha256"] is None or mod.is_sha256(item["sha256"]) for item in cited)

    state = mod.build_v393_close_state(sources)
    assert state["summary"] == "first_arc_oracle_distinct_win_harden_before_scale"
    assert state["headline_outcome"] == "arc_oracle_distinct_set_encoder_beats_vote_first_arc_win"
    assert state["oracle_distinct_status"] == "ARC-MOAT-WON"
    assert state["arc_win_status"] == "ARC-MOAT-WON"
    assert state["set_encoder_minus_vote_delta"] == 0.4423
    assert state["set_encoder_minus_vote_ci95"] == [0.308, 0.596]
    assert state["ci95_excludes_zero"] is True
    assert state["exclusion_manifest_count"] == 0
    assert state["verifier_is_oracle"] is False
    assert state["oracle_at_k"] == 0.827
    assert state["held_out_task_n"] == 52
    assert state["single_seed_n52_caveat"] is True
    assert state["provenance_leak_risk_caveat"] is True
    assert state["win_from_grown_pool_not_set_encoder_caveat"] is True
    assert state["set_encoder_auroc"] == 0.963
    assert state["logistic_auroc"] == 0.98
    assert state["set_encoder_underperformed_logistic"] is True
    assert state["code_replication_status"] == "BLOCKED"
    assert state["code_replication_honest_verdict"] == "blocked_code_second_corpus_missing"
    assert state["verifier_as_reward_status"] == "LIVE-LORA-RETIRED-OFFLINE-PENDING"
    assert state["verifier_as_reward_seventh_failure"] is True
    assert state["exp4247_flagged_adversarial"] is True
    assert state["exp4247_critical_flags"] == ["DURATION_TOO_SHORT"]
    assert state["live_lora_retired"] is True
    assert state["total_levels_solved"] == 19
    assert state["live_solver_levels_completed"] == 0
    assert state["live_solver_efficiency_only_no_level"] is True
    assert state["diffusiongemma_gate_resolvable"] is True
    assert state["v394_frame"] == mod.V394_FRAME

    fallback = mod.build_v393_close_state(
        {
            "4254": _capstone(arc_set_encoder_gate="bad", flagged_artifacts_skipped="bad"),
            "4245": _arc_win(set_encoder_minus_vote_ci95="bad", pass_rates="bad"),
            "4244": _build(oracle_distinct_auroc="bad", logistic_auroc="bad"),
            "4246": _code_replication(),
            "4247": _reward_retire(corrigendum_pending="bad"),
            "4249": _arc_progress(),
            "4250": _live_solver(solver_beats_floor="bad"),
        }
    )
    assert fallback["set_encoder_minus_vote_ci95"] == [0.308, 0.596]
    assert fallback["pass_rates"] == {}
    assert fallback["set_encoder_auroc"] == 0.963
    assert fallback["logistic_auroc"] == 0.98
    assert fallback["exp4247_critical_flags"] == []
    malformed_flags = mod.build_v393_close_state(
        {
            "4254": _capstone(),
            "4245": _arc_win(),
            "4244": _build(),
            "4246": _code_replication(),
            "4247": _reward_retire(
                corrigendum_pending=[
                    "bad",
                    {"kind": "INFO_ONLY", "severity": "warning"},
                    {"kind": "DURATION_TOO_SHORT", "severity": "critical"},
                ]
            ),
            "4249": _arc_progress(),
            "4250": _live_solver(),
        }
    )
    assert malformed_flags["exp4247_critical_flags"] == ["DURATION_TOO_SHORT"]


def test_run_happy_path_writes_valid_artifact_and_updates_archive(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4255: complete path writes the terminal archive artifact."""

    root = make_repo(tmp_path, duplicates=2)
    artifact = run_happy(root)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["archived_milestone"] == "2026.06.393"
    assert artifact["activated_milestone"] == "2026.06.394"
    assert artifact["active_milestone_confirmed"] == "2026.06.394"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["research_complete_record_action"] == "deduped"
    assert artifact["research_complete_duplicates_removed"] == 1
    assert artifact["v393_close_state"]["arc_win_status"] == "ARC-MOAT-WON"
    assert artifact["v393_close_state"]["single_seed_n52_caveat"] is True
    assert artifact["v393_close_state"]["code_replication_status"] == "BLOCKED"
    assert (
        artifact["field_principles"]["v393_close_state"] == mod.FIELD_PRINCIPLES["v393_close_state"]
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.is_sha256(artifact["reproducibility_checksum"])
    complete_text = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(complete_text) == 1
    assert "FIRST ARC oracle-distinct win" in complete_text
    assert "HARDEN the win" in complete_text
    mod.validate_artifact(artifact)


def test_run_blocked_preconditions_before_research_complete_edit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4255-BLOCKED-PRECONDITION: blocked paths do not fabricate success."""

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
    (root5 / "research-roadmap.yaml").write_text("milestone: 2026.06.393\n", encoding="utf-8")
    artifact5 = json.loads(mod.run(root5, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact5["honest_verdict"] == "blocked_v394_not_active"

    missing_sources = [
        ("experiment_4254_capstone_v393.json", "blocked_v393_capstone_missing"),
        ("experiment_4245_arc_set_encoder_beats_vote.json", "blocked_arc_win_missing"),
        ("experiment_4244_arc_set_encoder_aggregator_build.json", "blocked_set_encoder_build_missing"),
        ("experiment_4246_code_oracle_distinct_replication.json", "blocked_code_replication_missing"),
        (
            "experiment_4247_verifier_reward_offline_harness_retire_livelora.json",
            "blocked_reward_retirement_missing",
        ),
        ("experiment_4249_arc_incremental_progress.json", "blocked_arc_progress_missing"),
        ("experiment_4250_arc_live_env_solver_accuracy.json", "blocked_live_solver_missing"),
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
    """REQ-REPORT-4255: invalid archive edits are blocked before completion."""

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
    """REQ-REPORT-4255: schema validation and entrypoints preserve the contract."""

    root = make_repo(tmp_path)
    state = mod.build_v393_close_state(mod.read_v393_sources(root))
    complete = mod.build_complete_artifact(
        v393_close_state=state,
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

    import carnot.experiment_4255_archive_v393_activate_v394 as entrypoint

    called: dict[str, Path] = {}

    def fake_run(root_path: Path = mod.REPO_ROOT) -> Path:
        called["root"] = Path(root_path)
        return root / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(entrypoint, "run", fake_run)
    assert entrypoint.main() == 0
    assert called["root"] == entrypoint.REPO_ROOT

    script_path = Path("results/experiment_4255_archive_v393_activate_v394.py")
    spec = importlib.util.spec_from_file_location("exp4255_archive_script", script_path)
    assert spec and spec.loader
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "run", fake_run)
    assert script.main() == 0


def test_validate_artifact_rejection_guards(tmp_path: Path) -> None:
    """REQ-REPORT-4255: validation rejects artifacts that launder the .393 truth."""

    good = run_happy(make_repo(tmp_path))

    def set_path(obj: dict, path: list[str], value: object) -> None:
        cur = obj
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = value

    cases = [
        ("missing required fields", lambda a: a.pop("v393_close_state")),
        ("terminal-prefixed", lambda a: a.__setitem__("honest_verdict", "done")),
        ("field_principles must be a mapping", lambda a: a.__setitem__("field_principles", "x")),
        ("missing field principles", lambda a: a.__setitem__("field_principles", {})),
        (
            "principle must match REQ-REPORT-4255",
            lambda a: a["field_principles"].__setitem__("v393_close_state", "wrong"),
        ),
        ("archived milestone", lambda a: a.__setitem__("archived_milestone", "2026.06.999")),
        ("activated milestone", lambda a: a.__setitem__("activated_milestone", "2026.06.999")),
        ("research-complete YAML", lambda a: a.__setitem__("research_complete_yaml_parses", False)),
        ("exclusion manifest", lambda a: a.__setitem__("exclusion_manifest_parses", False)),
        ("pretest suite", lambda a: a.__setitem__("pretest_suite_green", False)),
        ("active milestone", lambda a: a.__setitem__("active_milestone_confirmed", "2026.06.393")),
        ("v393_close_state must be a mapping", lambda a: a.__setitem__("v393_close_state", "x")),
        ("ARC win status", lambda a: set_path(a, ["v393_close_state", "arc_win_status"], "TIED")),
        ("ARC delta", lambda a: set_path(a, ["v393_close_state", "set_encoder_minus_vote_delta"], 0.0)),
        ("ARC CI", lambda a: set_path(a, ["v393_close_state", "set_encoder_minus_vote_ci95"], [0, 0])),
        ("ARC n", lambda a: set_path(a, ["v393_close_state", "held_out_task_n"], 51)),
        ("ARC oracle", lambda a: set_path(a, ["v393_close_state", "verifier_is_oracle"], True)),
        ("exclusion count", lambda a: set_path(a, ["v393_close_state", "exclusion_manifest_count"], 1)),
        ("single-seed caveat", lambda a: set_path(a, ["v393_close_state", "single_seed_n52_caveat"], False)),
        ("provenance caveat", lambda a: set_path(a, ["v393_close_state", "provenance_leak_risk_caveat"], False)),
        (
            "grown-pool caveat",
            lambda a: set_path(a, ["v393_close_state", "win_from_grown_pool_not_set_encoder_caveat"], False),
        ),
        ("set-encoder AUROC", lambda a: set_path(a, ["v393_close_state", "set_encoder_auroc"], 0.99)),
        ("logistic AUROC", lambda a: set_path(a, ["v393_close_state", "logistic_auroc"], 0.95)),
        ("code replication", lambda a: set_path(a, ["v393_close_state", "code_replication_status"], "WON")),
        (
            "reward",
            lambda a: set_path(a, ["v393_close_state", "verifier_as_reward_status"], "OFFLINE-REAL"),
        ),
        ("seventh failure", lambda a: set_path(a, ["v393_close_state", "verifier_as_reward_seventh_failure"], False)),
        ("flagged", lambda a: set_path(a, ["v393_close_state", "exp4247_flagged_adversarial"], False)),
        ("critical", lambda a: set_path(a, ["v393_close_state", "exp4247_critical_flags"], [])),
        ("live LoRA", lambda a: set_path(a, ["v393_close_state", "live_lora_retired"], False)),
        ("ARC levels", lambda a: set_path(a, ["v393_close_state", "total_levels_solved"], 18)),
        ("live", lambda a: set_path(a, ["v393_close_state", "live_solver_efficiency_only_no_level"], False)),
        ("DiffusionGemma", lambda a: set_path(a, ["v393_close_state", "diffusiongemma_gate_resolvable"], False)),
        ("v394 frame", lambda a: set_path(a, ["v393_close_state", "v394_frame"], "redo")),
    ]
    for label, mutate in cases:
        artifact = copy.deepcopy(good)
        mutate(artifact)
        with pytest.raises(ValueError, match=label):
            mod.validate_artifact(artifact)
