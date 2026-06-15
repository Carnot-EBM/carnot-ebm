"""Tests for Exp 4269 `.394` archive / `.395` activation.

Spec refs: REQ-REPORT-4269, SCENARIO-REPORT-4269,
SCENARIO-REPORT-4269-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v394_activate_v395_4269 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="87 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.393\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.394\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-15'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4268-capstone-v394\n"
        "    result: OK\n"
    )
    return head + block * duplicates


def _capstone(**overrides: object) -> dict:
    payload = {
        "experiment": 4268,
        "honest_verdict": (
            "complete: capstone_v394_within_pool_win_survived_but_cross_game_blocked_"
            "synthesis_excluded_preflight_excluded_arc19_live0_reward_out_of_band_"
            "code_corpus_specific_paper_ready_hardened_win_False_"
            "diffusiongemma_full_run_gate_False_excluded_3"
        ),
        "headline_outcome": (
            "within_pool_win_survived_but_cross_game_blocked_synthesis_excluded_"
            "preflight_excluded_arc19_live0_reward_out_of_band_code_corpus_specific_"
            "paper_ready"
        ),
        "hardened_win": False,
        "diffusiongemma_full_run_gate": False,
        "paper_ready": True,
        "hardening": {
            "hardened_win": False,
            "provenance_blind": {
                "honest_verdict": "complete: arc_set_encoder_win_survives_provenance_blind_audit",
                "win_survives_provenance_blind": True,
                "provenance_blind_delta": 0.3846153846,
                "provenance_blind_set_encoder_auroc": 0.990260169,
                "origin_probe_auroc": 0.9481745077,
                "held_out_task_n": 52,
                "oracle_at_k": 0.8269230769,
                "verifier_is_oracle": False,
            },
            "multiseed": {
                "honest_verdict": "complete: arc_oracle_distinct_win_replicates_multiseed",
                "oracle_distinct_win_replicates": True,
                "mean_delta": 0.4576923077,
                "independent_rescore_delta": 0.4423076923,
                "n_seeds": 5,
                "verifier_is_oracle": False,
            },
            "cross_game": {
                "honest_verdict": "blocked_arc_game_ids_unrecoverable",
                "status": "blocked_or_no_positive_delta",
                "cross_game_delta": None,
                "held_out_game_n": 0,
                "held_out_task_n": 0,
                "verifier_is_oracle": False,
            },
        },
        "extend_synthesis": {
            "status": "excluded_flagged_adversarial",
            "synthesis_breaks_oracle_ceiling": False,
        },
        "scale_up_readiness": {"status": "excluded_flagged_adversarial", "preflight_go": False},
        "arc_progress": {
            "honest_verdict": (
                "complete: incremental_progress_no_advance_sc25-635fd71a_L6_"
                "no_verifier_validated_level_up_candidate"
            ),
            "total_levels_solved": 19,
            "total_levels": 19,
            "levels_completed": 0,
            "status": "included",
        },
        "reward_decision": {
            "honest_verdict": "complete: ready_for_out_of_band_verifier_reward_training",
            "ready_for_out_of_band": True,
            "status": "reward_out_of_band",
            "verifier_as_reward_retired": False,
            "verifier_is_oracle": True,
        },
        "code_read": {
            "honest_verdict": "complete: code_oracle_distinct_replication_corpus_specific",
            "code_predictor_minus_vote_delta": -0.00625,
            "off_fold_auroc": 0.6967654987,
            "code_replication_beats_vote": False,
            "replication_read": "corpus_specific",
            "status": "code_corpus_specific",
            "verifier_is_oracle": False,
        },
        "flagged_artifacts_excluded": [
            {"experiment_id": 4259, "artifact_key": "4259_synthesis"},
            {"experiment_id": 4260, "artifact_key": "4260_preflight"},
            {"experiment_id": 4266, "artifact_key": "4266_registry"},
        ],
        "publication_gate": {"paper_ready": True, "unmet_gates": []},
        "unmet_gates": [],
    }
    payload.update(overrides)
    return payload


def _leak_audit(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: arc_set_encoder_win_survives_provenance_blind_audit",
        "win_survives_provenance_blind": True,
        "provenance_blind_delta": 0.3846153846,
        "provenance_blind_set_encoder_auroc": 0.990260169,
        "origin_probe_auroc": 0.9481745077,
        "held_out_task_n": 52,
        "oracle_at_k": 0.8269230769,
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _multiseed(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: arc_oracle_distinct_win_replicates_multiseed",
        "oracle_distinct_win_replicates": True,
        "mean_delta": 0.4576923077,
        "independent_rescore_delta": 0.4423076923,
        "independent_rescore_oracle_at_k": 0.8269230769,
        "n_seeds": 5,
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _cross_game(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "blocked_arc_game_ids_unrecoverable",
        "honest_read": "blocked",
        "cross_game_delta": None,
        "held_out_game_n": 0,
        "held_out_task_n": 0,
        "verifier_is_oracle": False,
        "model_specs": {"blocked_reason": "game_ids_unrecoverable"},
    }
    payload.update(overrides)
    return payload


def _synthesis(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: arc_synthesis_underperforms_selection",
        "synthesis_breaks_oracle_ceiling": False,
        "synthesis_minus_oracle_delta": -0.2826086957,
        "synthesis_minus_vote_delta": 0.347826087,
        "flagged_adversarial": True,
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _preflight(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "blocked_diffusiongemma_gguf_loader_failed",
        "preflight_go": False,
        "guidance_changes_selection": False,
        "full_run_cost_estimate_s": 0.0,
        "flagged_adversarial": True,
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _arc_progress(**overrides: object) -> dict:
    payload = {
        "honest_verdict": (
            "complete: incremental_progress_no_advance_sc25-635fd71a_L6_"
            "no_verifier_validated_level_up_candidate"
        ),
        "total_levels_solved": 19,
        "total_levels": 19,
        "levels_completed": 0,
        "verifier_validated": False,
    }
    payload.update(overrides)
    return payload


def _reward(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: ready_for_out_of_band_verifier_reward_training",
        "ready_for_out_of_band": True,
        "verifier_as_reward_retired": False,
        "verifier_is_oracle": True,
    }
    payload.update(overrides)
    return payload


def _code(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: code_oracle_distinct_replication_corpus_specific",
        "code_predictor_minus_vote_delta": -0.00625,
        "code_replication_beats_vote": False,
        "off_fold_auroc": 0.6967654987,
        "replication_read": "corpus_specific",
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
        "retired:\n- experiment_id: 4266\n  reason: flagged\n", encoding="utf-8"
    )
    (root / "research-roadmap.yaml").write_text("milestone: 2026.06.395\n", encoding="utf-8")
    _write_json(root / "results" / "experiment_4268_capstone_v394.json", _capstone())
    _write_json(
        root / "results" / "experiment_4256_arc_oracle_distinct_leak_audit.json",
        _leak_audit(),
    )
    _write_json(
        root / "results" / "experiment_4257_arc_oracle_distinct_multiseed_replication.json",
        _multiseed(),
    )
    _write_json(
        root / "results" / "experiment_4258_arc_oracle_distinct_cross_game_transfer.json",
        _cross_game(),
    )
    _write_json(root / "results" / "experiment_4259_arc_agglm_grid_synthesis.json", _synthesis())
    _write_json(
        root / "results" / "experiment_4260_diffusiongemma_energy_guided_preflight.json",
        _preflight(),
    )
    _write_json(root / "results" / "experiment_4261_arc_incremental_progress.json", _arc_progress())
    _write_json(
        root / "results" / "experiment_4263_verifier_as_reward_out_of_band_or_retire.json",
        _reward(),
    )
    _write_json(
        root / "results" / "experiment_4264_code_oracle_distinct_replication_retry.json",
        _code(),
    )
    return root


def run_happy(root: Path) -> dict:
    out = mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5)
    return json.loads(out.read_text(encoding="utf-8"))


def test_req_report_4269_spec_declares_contract() -> None:
    """REQ-REPORT-4269: OpenSpec declares the .394 close-state truth contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    assert "REQ-REPORT-4269" in spec
    assert "SCENARIO-REPORT-4269" in spec
    assert "SCENARIO-REPORT-4269-BLOCKED-PRECONDITION" in spec
    assert "win hardened 2-of-3" in spec
    assert "blocked-not-collapsed" in spec
    assert "`+0.385`" in spec
    assert "mean delta `+0.458`" in spec
    assert "`blocked_arc_game_ids_unrecoverable`" in spec
    assert "synthesis_minus_oracle=-0.283" in spec
    assert "ready_for_out_of_band=true" in spec
    assert "fresh-corpus delta `-0.006`" in spec
    assert mod.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert mod.FIELD_PRINCIPLES["v394_close_state"] in spec
    assert mod.FIELD_PRINCIPLES["preconditions_checked"] in spec


def test_helpers_and_archive_record_editing(tmp_path: Path) -> None:
    """REQ-REPORT-4269: helper behavior is deterministic and YAML-safe."""

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

    close_state = mod.build_v394_close_state(
        {
            "4268": _capstone(),
            "4256": _leak_audit(),
            "4257": _multiseed(),
            "4258": _cross_game(),
            "4259": _synthesis(),
            "4260": _preflight(),
            "4261": _arc_progress(),
            "4263": _reward(),
            "4264": _code(),
        }
    )
    assert mod.archive_record_count(_research_complete_text(duplicates=3)) == 3
    deduped, removed, action = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=3), close_state
    )
    assert (removed, action) == (2, "deduped")
    assert mod.archive_record_count(deduped) == 1
    assert "HARDENED 2 of 3" in deduped
    assert "CLOSE the cross-family OOD question" in deduped
    assert mod.yaml_parses(deduped)
    updated, removed2, action2 = mod.dedupe_or_update_record(
        _research_complete_text(duplicates=1), close_state
    )
    assert (removed2, action2) == (0, "updated")
    unchanged, removed3, action3 = mod.dedupe_or_update_record(updated, close_state)
    assert (unchanged, removed3, action3) == (updated, 0, "unchanged")
    appended, removed4, action4 = mod.dedupe_or_update_record(
        "# history\nmilestones:\n- id: 2026.06.393\n  finding: prior\n", close_state
    )
    assert (removed4, action4) == (0, "appended")
    assert "activation_recorded: exp4269-archive-v394-activate-v395" in appended
    assert mod._insert_before_tasks(["  title: no tasks"], "  finding: x") == [
        "  title: no tasks",
        "  finding: x",
    ]
    added_finding, removed5, action5 = mod.dedupe_or_update_record(
        "milestones:\n- id: 2026.06.394\n  title: missing finding\n  tasks:\n  - id: exp4268\n",
        close_state,
    )
    assert (removed5, action5) == (0, "updated")
    assert "HARDENED 2 of 3" in added_finding


def test_read_sources_and_build_v394_close_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4269: close-state records hardening and the blocked OOD gap."""

    root = make_repo(tmp_path)
    sources = mod.read_v394_sources(root)
    assert sources["4268"]["hardened_win"] is False
    assert sources["4256"]["win_survives_provenance_blind"] is True
    assert sources["4257"]["oracle_distinct_win_replicates"] is True
    assert sources["4258"]["honest_verdict"] == "blocked_arc_game_ids_unrecoverable"

    cited = mod.build_cited_upstream(root)
    assert {item["experiment_id"] for item in cited} == {
        "4268",
        "4256",
        "4257",
        "4258",
        "4259",
        "4260",
        "4261",
        "4263",
        "4264",
    }
    assert all(item["sha256"] is None or mod.is_sha256(item["sha256"]) for item in cited)

    state = mod.build_v394_close_state(sources)
    assert state["summary"] == "first_arc_oracle_distinct_win_hardened_2_of_3_ood_blocked"
    assert state["leak_audit_survived"] is True
    assert state["provenance_blind_delta"] == 0.385
    assert state["provenance_blind_auroc"] == 0.99
    assert state["multiseed_replicated"] is True
    assert state["multiseed_mean_delta"] == 0.458
    assert state["independent_rescore_delta"] == 0.442
    assert state["cross_game_ood_ran"] is False
    assert state["cross_game_block_reason"] == "game_ids_unrecoverable"
    assert state["cross_game_blocked_not_collapsed"] is True
    assert state["hardened_axes_passed"] == 2
    assert state["hardened_axes_total"] == 3
    assert state["hardened_win"] is False
    assert state["diffusiongemma_full_run_gate"] is False
    assert state["synthesis_breaks_oracle_ceiling"] is False
    assert state["synthesis_minus_oracle_delta"] == -0.283
    assert state["diffusiongemma_preflight_go"] is False
    assert state["diffusiongemma_block_reason"] == "blocked_diffusiongemma_gguf_loader_failed"
    assert state["total_levels_solved"] == 19
    assert state["reward_ready_for_out_of_band"] is True
    assert state["code_replication_read"] == "corpus_specific"
    assert state["code_predictor_minus_vote_delta"] == -0.006
    assert state["paper_ready"] is True
    assert state["v395_frame"] == mod.V395_FRAME

    fallback = mod.build_v394_close_state(
        {
            "4268": _capstone(hardening="bad", flagged_artifacts_excluded="bad"),
            "4256": _leak_audit(provenance_blind_delta="bad"),
            "4257": _multiseed(mean_delta="bad"),
            "4258": _cross_game(honest_verdict="blocked_arc_family_missing"),
            "4259": _synthesis(synthesis_minus_oracle_delta="bad"),
            "4260": _preflight(),
            "4261": _arc_progress(total_levels_solved="bad"),
            "4263": _reward(),
            "4264": _code(code_predictor_minus_vote_delta="bad"),
        }
    )
    assert fallback["provenance_blind_delta"] == 0.385
    assert fallback["multiseed_mean_delta"] == 0.458
    assert fallback["cross_game_block_reason"] == "arc_family_missing"
    assert fallback["synthesis_minus_oracle_delta"] == -0.283
    assert fallback["total_levels_solved"] == 19
    assert fallback["code_predictor_minus_vote_delta"] == -0.006
    assert fallback["flagged_artifacts_excluded"] == []


def test_run_happy_path_writes_valid_artifact_and_updates_archive(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4269: complete path writes the terminal archive artifact."""

    root = make_repo(tmp_path, duplicates=2)
    artifact = run_happy(root)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["archived_milestone"] == "2026.06.394"
    assert artifact["activated_milestone"] == "2026.06.395"
    assert artifact["active_milestone_confirmed"] == "2026.06.395"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["research_complete_record_action"] == "deduped"
    assert artifact["research_complete_duplicates_removed"] == 1
    assert artifact["v394_close_state"]["hardened_axes_passed"] == 2
    assert artifact["v394_close_state"]["cross_game_blocked_not_collapsed"] is True
    assert artifact["v394_close_state"]["hardened_win"] is False
    assert artifact["v394_close_state"]["paper_ready"] is True
    assert artifact["field_principles"]["v394_close_state"] == mod.FIELD_PRINCIPLES[
        "v394_close_state"
    ]
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.is_sha256(artifact["reproducibility_checksum"])
    complete_text = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(complete_text) == 1
    assert "HARDENED 2 of 3" in complete_text
    assert "game_ids_unrecoverable" in complete_text
    mod.validate_artifact(artifact)


def test_run_blocked_preconditions_before_research_complete_edit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4269-BLOCKED-PRECONDITION: blocked paths do not fabricate success."""

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
    (root5 / "research-roadmap.yaml").write_text("milestone: 2026.06.394\n", encoding="utf-8")
    artifact5 = json.loads(mod.run(root5, pretest_result=GREEN).read_text(encoding="utf-8"))
    assert artifact5["honest_verdict"] == "blocked_v395_not_active"

    missing_sources = [
        ("experiment_4268_capstone_v394.json", "blocked_v394_capstone_missing"),
        ("experiment_4256_arc_oracle_distinct_leak_audit.json", "blocked_leak_audit_missing"),
        (
            "experiment_4257_arc_oracle_distinct_multiseed_replication.json",
            "blocked_multiseed_replication_missing",
        ),
        (
            "experiment_4258_arc_oracle_distinct_cross_game_transfer.json",
            "blocked_cross_game_transfer_missing",
        ),
        ("experiment_4259_arc_agglm_grid_synthesis.json", "blocked_synthesis_missing"),
        (
            "experiment_4260_diffusiongemma_energy_guided_preflight.json",
            "blocked_diffusiongemma_preflight_missing",
        ),
        ("experiment_4261_arc_incremental_progress.json", "blocked_arc_progress_missing"),
        (
            "experiment_4263_verifier_as_reward_out_of_band_or_retire.json",
            "blocked_reward_out_of_band_missing",
        ),
        (
            "experiment_4264_code_oracle_distinct_replication_retry.json",
            "blocked_code_replication_missing",
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
    """REQ-REPORT-4269: invalid archive edits are blocked before completion."""

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
    """REQ-REPORT-4269: schema validation and entrypoints preserve the contract."""

    root = make_repo(tmp_path)
    state = mod.build_v394_close_state(mod.read_v394_sources(root))
    complete = mod.build_complete_artifact(
        v394_close_state=state,
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

    import carnot.experiment_4269_archive_v394_activate_v395 as entrypoint

    called: dict[str, Path] = {}

    def fake_run(root_path: Path = mod.REPO_ROOT) -> Path:
        called["root"] = Path(root_path)
        return root / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(entrypoint, "run", fake_run)
    assert entrypoint.main() == 0
    assert called["root"] == entrypoint.REPO_ROOT

    script_path = Path("results/experiment_4269_archive_v394_activate_v395.py")
    spec = importlib.util.spec_from_file_location("exp4269_archive_script", script_path)
    assert spec and spec.loader
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "run", fake_run)
    assert script.main() == 0


def test_validate_artifact_rejection_guards(tmp_path: Path) -> None:
    """REQ-REPORT-4269: validation rejects artifacts that launder the .394 truth."""

    good = run_happy(make_repo(tmp_path))

    def set_path(obj: dict, path: list[str], value: object) -> None:
        cur = obj
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = value

    cases = [
        ("missing required fields", lambda a: a.pop("v394_close_state")),
        ("terminal-prefixed", lambda a: a.__setitem__("honest_verdict", "done")),
        ("field_principles must be a mapping", lambda a: a.__setitem__("field_principles", "x")),
        ("missing field principles", lambda a: a.__setitem__("field_principles", {})),
        (
            "principle must match REQ-REPORT-4269",
            lambda a: a["field_principles"].__setitem__("v394_close_state", "wrong"),
        ),
        ("archived milestone", lambda a: a.__setitem__("archived_milestone", "2026.06.999")),
        ("activated milestone", lambda a: a.__setitem__("activated_milestone", "2026.06.999")),
        ("research-complete YAML", lambda a: a.__setitem__("research_complete_yaml_parses", False)),
        ("exclusion manifest", lambda a: a.__setitem__("exclusion_manifest_parses", False)),
        ("pretest suite", lambda a: a.__setitem__("pretest_suite_green", False)),
        ("active milestone", lambda a: a.__setitem__("active_milestone_confirmed", "2026.06.394")),
        ("v394_close_state must be a mapping", lambda a: a.__setitem__("v394_close_state", "x")),
        ("leak audit", lambda a: set_path(a, ["v394_close_state", "leak_audit_survived"], False)),
        ("leak delta", lambda a: set_path(a, ["v394_close_state", "provenance_blind_delta"], 0.0)),
        ("leak AUROC", lambda a: set_path(a, ["v394_close_state", "provenance_blind_auroc"], 0.1)),
        ("multi-seed", lambda a: set_path(a, ["v394_close_state", "multiseed_replicated"], False)),
        ("mean delta", lambda a: set_path(a, ["v394_close_state", "multiseed_mean_delta"], 0.0)),
        ("rescore", lambda a: set_path(a, ["v394_close_state", "independent_rescore_delta"], 0.0)),
        ("cross-game", lambda a: set_path(a, ["v394_close_state", "cross_game_ood_ran"], True)),
        (
            "game IDs",
            lambda a: set_path(a, ["v394_close_state", "cross_game_block_reason"], "win_collapsed"),
        ),
        (
            "blocked-not-collapsed",
            lambda a: set_path(a, ["v394_close_state", "cross_game_blocked_not_collapsed"], False),
        ),
        ("axes", lambda a: set_path(a, ["v394_close_state", "hardened_axes_passed"], 3)),
        ("hardened win", lambda a: set_path(a, ["v394_close_state", "hardened_win"], True)),
        ("DiffusionGemma gate", lambda a: set_path(a, ["v394_close_state", "diffusiongemma_full_run_gate"], True)),
        (
            "synthesis",
            lambda a: set_path(a, ["v394_close_state", "synthesis_breaks_oracle_ceiling"], True),
        ),
        ("oracle delta", lambda a: set_path(a, ["v394_close_state", "synthesis_minus_oracle_delta"], 0.0)),
        (
            "loader",
            lambda a: set_path(a, ["v394_close_state", "diffusiongemma_block_reason"], "ok"),
        ),
        ("ARC levels", lambda a: set_path(a, ["v394_close_state", "total_levels_solved"], 18)),
        (
            "reward",
            lambda a: set_path(a, ["v394_close_state", "reward_ready_for_out_of_band"], False),
        ),
        ("code", lambda a: set_path(a, ["v394_close_state", "code_replication_read"], "robust")),
        ("code delta", lambda a: set_path(a, ["v394_close_state", "code_predictor_minus_vote_delta"], 0.1)),
        ("paper", lambda a: set_path(a, ["v394_close_state", "paper_ready"], False)),
        ("v395 frame", lambda a: set_path(a, ["v394_close_state", "v395_frame"], "redo")),
    ]
    for label, mutate in cases:
        artifact = copy.deepcopy(good)
        mutate(artifact)
        with pytest.raises(ValueError, match=label):
            mod.validate_artifact(artifact)
