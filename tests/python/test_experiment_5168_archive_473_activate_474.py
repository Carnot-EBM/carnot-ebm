"""Tests for Exp 5168 archive .473 / activate .474 aggregation.

Spec refs: REQ-REPORT-5168, SCENARIO-REPORT-5168,
SCENARIO-REPORT-5168-DIRTY-RUNTIME.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5168_archive_473_activate_474 as mod


GREEN_VERIFY = mod.CommandResult(
    command=("python", "scripts/adversarial_verify.py"),
    exit_code=0,
    stdout='{"flags":[]}',
    stderr="",
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _task_yaml(milestone: str = "2026.07.474", *, last: int = 5180) -> str:
    tasks = "\n".join(
        f"- id: exp{exp_id}-task\n  milestone: {milestone}\n  title: task {exp_id}"
        for exp_id in range(5168, last + 1)
    )
    return f"milestone: {milestone}\ntasks:\n{tasks}\n"


def _vnext_text(milestone: str = "2026.07.474") -> str:
    return (
        "# Research Roadmap vNEXT\n\n"
        f"**Milestone:** `{milestone}`\n"
        "**Predecessor:** `2026.07.473`\n"
        "Same-day correction absorbed into this plan: exp5161 is now un-quarantined.\n"
    )


def _v473_payloads() -> dict[int, dict]:
    return {
        5156: {
            "experiment": "experiment_5156_archive_472_activate_473",
            "experiment_id": "exp5156-archive-472-activate-473",
            "honest_verdict": "complete_archive_472_closed_473_active_runtime_clean",
            "flagged_adversarial": True,
            "adversarial_verification": {
                "flags": [{"kind": "qd-random-mutation-ablation-omitted", "severity": "warn"}],
                "flagged_adversarial": True,
            },
            "v472_runtime_clean": True,
            "reproducibility_checksum": "sha256:" + "0" * 64,
        },
        5157: {
            "experiment": "experiment_5157_deepen_warmstart_replay_ablation_v473",
            "honest_verdict": "complete: warmstart_replay_ablation_gate_failed_honest_null_delta_0.0",
            "gate_passed": False,
            "warmstart_vs_cold_delta_median": 0.0,
            "games_tested": [
                {"game": f"g{i:02d}", "n_level_transitions_tested": 1}
                for i in range(20)
            ]
            + [{"game": "tr87", "n_level_transitions_tested": 12}],
            "per_transition_breakdown": [{"game": "tr87"} for _ in range(32)],
            "reproducibility_checksum": "sha256:" + "1" * 64,
        },
        5158: {
            "experiment": "experiment_5158_deepen_goal_energy_ranker_replay_v473",
            "honest_verdict": "complete: goal_energy_ranker_warmstart_gate_failed_improved_1_of_3",
            "gate_passed": False,
            "games_tested": [
                {"game": "lp85", "n_level_transitions_tested": 4},
                {"game": "sc25", "n_level_transitions_tested": 4},
                {"game": "tr87", "n_level_transitions_tested": 5},
            ],
            "games_improved_count": 1,
            "reciprocal_rank_cold": {"lp85": 1.0, "sc25": 0.339286, "tr87": 1.0},
            "reciprocal_rank_warmstart": {
                "lp85": 0.404167,
                "sc25": 0.345238,
                "tr87": 0.666667,
            },
            "reproducibility_checksum": "sha256:" + "2" * 64,
        },
        5159: {
            "experiment": 5159,
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: "
                "exp5157-deepen-warmstart-replay-ablation-v473.gate_passed "
                "(actual=False == expected=True)"
            ),
            "reproducibility_checksum": "sha256:" + "3" * 64,
        },
        5160: {
            "experiment": "experiment_5160_oracle_distinct_cross_corpus_closure_v473",
            "honest_verdict": (
                "success_arc_set_encoder_win_survives_cross_corpus_replication: "
                "set-encoder-vs-vote win survives corrected cross-corpus replication"
            ),
            "cross_corpus_delta": 0.5,
            "cross_corpus_delta_ci95": [0.5, 0.5],
            "second_pool_leak_audit_passed": True,
            "diffusiongemma_gate_updated_recommendation": "ungate_now",
            "game_id_misnomer_confirmed": True,
            "held_out_task_n": 24,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "4" * 64,
        },
        5161: {
            "experiment": "experiment_5161_gap4_protocol_execution_pilot",
            "experiment_id": 5161,
            "honest_verdict": (
                "complete_gap4_pilot_n60_direction_replicated_not_significant_scale_up_recommended"
            ),
            "flagged_adversarial": False,
            "duration_s": 5.58995,
            "pilot_n_achieved": 60,
            "replicated_prior_direction": True,
            "exact_test_discordant_wins": {
                "principle": "sign-test floor",
                "value": 4,
            },
            "exact_test_discordant_losses": 0,
            "exact_test_p_value_two_sided": 0.125,
            "exact_test_passes_min6_rule": {
                "principle": "requires at least six wins",
                "value": False,
            },
            "gap4_status_recommendation": {
                "principle": "feeds GAP-4 status",
                "value": "scale_up_recommended",
            },
            "linter_flag_corrigendum": "corrected same-day",
            "reproducibility_checksum": "sha256:" + "5" * 64,
        },
        5162: {
            "experiment_id": "experiment_5162_sota_ingestion_multilevel_v473",
            "honest_verdict": (
                "complete: zero new post-2026-07-02 primary findings; "
                "outcome-conditioned V474 references appended"
            ),
            "reproducibility_checksum": "sha256:" + "6" * 64,
        },
        5163: {
            "experiment": "experiment_5163_mmlu_pro_verifier_rescale_v473",
            "honest_verdict": {
                "principle": "terminal",
                "value": (
                    "complete_mmlu_pro_fewshot_verifier_vs_cheap_delta_+0.025_"
                    "CI95_[-0.125,0.175]_CI_includes_0"
                ),
            },
            "flagged_adversarial": True,
            "verifier_vs_cheap_delta": {"principle": "delta", "value": 0.025},
            "verifier_vs_cheap_delta_ci95": {
                "principle": "ci",
                "value": [-0.125, 0.175],
            },
            "still_underpowered": {"principle": "underpowered", "value": True},
            "fewshot_oracle_at_k": {"principle": "ceiling", "value": 0.5},
            "oracle_at_k_ceiling": 0.5,
            "reproducibility_checksum": "sha256:" + "7" * 64,
        },
        5164: {
            "experiment": "experiment_5164_retro_timing_falsezero_fix_v473",
            "honest_verdict": (
                "complete: module correctly reconstructs the .450 ground-truth case "
                "(214.6 wall minutes, 4 compute-bound arms) and returns non-zero timing "
                "for .467, .470, and .472 without modifying scripts/research_conductor.py"
            ),
            "m450_reconstruction_correct": True,
            "validated_milestones": [
                {
                    "milestone": "2026.06.450",
                    "reconstructed_wall_time_minutes": 214.6,
                    "reconstructed_compute_bound_count": 4,
                    "matches_known_good": True,
                },
                {"milestone": "2026.07.467", "matches_known_good": True},
                {"milestone": "2026.07.470", "matches_known_good": True},
                {"milestone": "2026.07.472", "matches_known_good": True},
            ],
            "tests_added": 6,
            "tests_passing": True,
            "research_conductor_py_modified": False,
            "reproducibility_checksum": "sha256:" + "8" * 64,
        },
        5165: {
            "experiment": "experiment_5165_generation_axis_retirement_hygiene_v473",
            "experiment_id": "exp5165-generation-axis-retirement-hygiene-v473",
            "honest_verdict": (
                "complete: generation_axis_exploration_signal_scope_retired_and_lint_load_bearing"
            ),
            "flagged_adversarial": False,
            "exclusion_manifest_entry_added": True,
            "entry_id": "generation_axis_exploration_signal_retired_exp5154_v473",
            "synthetic_match_check_passed": True,
            "false_positive_check_against_this_milestone": True,
            "reproducibility_checksum": "sha256:" + "9" * 64,
        },
        5166: {
            "experiment": "experiment_5166_hardware_continuity_board_timing",
            "experiment_id": "exp5166-hardware-continuity-board-timing-v473",
            "honest_verdict": (
                "complete_hardware_continuity_board_timing_gatemate:"
                "blocked_gatemate_dirtyjtag_idcode_no_speedup_claim"
            ),
            "boards_reachable_count": 2,
            "kv260_result": {"reachable": True, "hash_verified": True},
            "polarfire_result": {"reachable": True, "hash_verified": True},
            "gatemate_result": {
                "reachable": False,
                "blocked_reason": "blocked_gatemate_dirtyjtag_idcode",
                "timing_output": {"expected_idcode": "0x20000001"},
            },
            "no_speedup_claim": True,
            "reproducibility_checksum": "sha256:" + "a" * 64,
        },
        5167: {
            "experiment": "experiment_5167_capstone_v473",
            "experiment_id": "exp5167-capstone-v473",
            "honest_verdict": (
                "complete: v473 reconciled with DiffusionGemma ungated for future scaling, "
                "GAP-4 scale-up not filled, zero new ARC levels banked, and exp5161 "
                "excluded as flagged_adversarial."
            ),
            "flagged_adversarial": False,
            "flagged_adversarial_artifacts_excluded": {
                "principle": "excluded from headline",
                "value": ["exp5161-gap4-protocol-execution-pilot-v473"],
            },
            "gap4_status_reconciled": {
                "principle": "GAP-4 status",
                "value": "scale_up_recommended_not_filled_flagged_excluded_from_headline",
            },
            "registry_reconciliation": {
                "loadable": True,
                "reproducible_total_levels": 69,
                "reproducible_total_games": 24,
                "delta_from_exp5159": 0,
            },
            "reproducible_total_levels_delta": {
                "principle": "new banked levels",
                "value": 0,
            },
            "research_conductor_py_untouched_confirmed": {
                "principle": "conductor untouched",
                "value": True,
            },
            "reproducibility_checksum": "sha256:" + "b" * 64,
        },
    }


def make_repo(
    tmp_path: Path,
    *,
    active_valid: bool = True,
    vnext_valid: bool = True,
    registry_levels: int = 69,
    registry_games: int = 24,
    gap4_valid: bool = True,
    omit_results: set[int] | None = None,
) -> Path:
    root = tmp_path
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "research_conductor.py").write_text("# conductor\n", encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(
        _task_yaml() if active_valid else _task_yaml("2026.07.473", last=5170),
        encoding="utf-8",
    )
    vnext_path = root / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md"
    vnext_path.parent.mkdir(parents=True, exist_ok=True)
    vnext_path.write_text(_vnext_text() if vnext_valid else _vnext_text("2026.07.473"), encoding="utf-8")
    (root / "ops" / "arc_solve_registry.yaml").write_text(
        f"reproducible_total_levels: {registry_levels}\n"
        f"reproducible_total_games: {registry_games}\n",
        encoding="utf-8",
    )
    gap4_line = (
        "`status` stays **open -- FIRST POSITIVE LANDED, now with a bounded-scale "
        "(n=60) directional replication on top, still short of the significance floor.**"
        if gap4_valid
        else "`status` is **filled**."
    )
    (root / "ops" / "verifier_gaps.md").write_text(
        "### GAP-4: Exp 5161 .473 forward-protocol pilot (n=60, bounded scale)\n"
        f"{gap4_line}\n",
        encoding="utf-8",
    )
    for exp_id, payload in _v473_payloads().items():
        if omit_results and exp_id in omit_results:
            continue
        _write_json(root / mod.V473_RESULT_PATHS[exp_id], payload)
    return root


def clean_runtime_snapshot() -> mod.RuntimeSnapshot:
    return mod.RuntimeSnapshot(
        git_status_porcelain="",
        process_table=(
            "100 42 Ssl 03:50:42 python scripts/research_conductor.py --loop\n"
            "101 100 Ssl 00:00:59 codex exec --cd /repo -\n"
        ),
    )


def test_req_report_5168_spec_declares_archive_contract() -> None:
    """REQ-REPORT-5168: OpenSpec anchors the .473 archive and .474 activation contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    for marker in (
        "REQ-REPORT-5168",
        "SCENARIO-REPORT-5168",
        "SCENARIO-REPORT-5168-DIRTY-RUNTIME",
        "results/experiment_5168_archive_473_activate_474.json",
        "v473_runtime_clean",
        "exp5161_unquarantine_noted",
        "aggregation_from_upstream_artifacts",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle in spec


def test_scenario_report_5168_happy_path_archives_corrected_v473_truth(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5168: corrected .473 truth and clean .474 activation are preserved."""

    artifact = mod.build_artifact(
        root=make_repo(tmp_path),
        duration_s=1.25,
        run_date="20260702",
        verification=mod.verification_payload(GREEN_VERIFY),
        runtime_snapshot=clean_runtime_snapshot(),
        tests_run=["unit-test-placeholder"],
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == "exp5168-archive-473-activate-474"
    assert artifact["milestone"] == "2026.07.474"
    assert artifact["archived_milestone"] == "2026.07.473"
    assert artifact["honest_verdict"] == mod.COMPLETE_VERDICT
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["v473_runtime_clean"] is True
    assert artifact["exp5161_unquarantine_noted"] is True
    assert artifact["active_roadmap_ready"] is True
    assert artifact["vnext_ready"] is True
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    assert len(artifact["task_verdicts"]) == 12

    archive = {row["experiment_id"]: row for row in artifact["milestone_archive_summary"]}
    assert archive["exp5157-deepen-warmstart-replay-ablation-v473"]["classification"] == (
        "honest_null_warmstart_gate_failed"
    )
    assert archive["exp5157-deepen-warmstart-replay-ablation-v473"]["transition_count"] == 32
    assert archive["exp5157-deepen-warmstart-replay-ablation-v473"]["game_count"] == 21
    assert archive["exp5158-deepen-goal-energy-ranker-replay-v473"]["improved_games_count"] == 1
    assert archive["exp5158-deepen-goal-energy-ranker-replay-v473"]["target_games"] == [
        "lp85",
        "sc25",
        "tr87",
    ]
    assert archive["exp5159-deepen-live-levelup-attempt-v473"]["classification"] == (
        "blocked_upstream_gate_no_live_run"
    )
    assert archive["exp5160-oracle-distinct-cross-corpus-closure-v473"]["classification"] == (
        "real_cross_corpus_win_below_clt_floor"
    )
    assert archive["exp5160-oracle-distinct-cross-corpus-closure-v473"]["cross_corpus_delta_ci95"] == [
        0.5,
        0.5,
    ]
    assert archive["exp5160-oracle-distinct-cross-corpus-closure-v473"]["meets_clt_floor_n30"] is False
    assert archive["exp5161-gap4-protocol-execution-pilot-v473"]["flagged_adversarial"] is False
    assert archive["exp5161-gap4-protocol-execution-pilot-v473"]["exact_test_p_value_two_sided"] == 0.125
    assert archive["exp5163-mmlu-pro-verifier-rescale-v473"]["classification"] == (
        "underpowered_tautology_flagged_not_headline_clean"
    )
    assert archive["exp5164-retro-timing-falsezero-fix-v473"]["m450_wall_minutes"] == 214.6
    assert archive["exp5165-generation-axis-retirement-hygiene-v473"]["synthetic_match_check_passed"] is True
    assert archive["exp5166-hardware-continuity-board-timing-v473"]["boards_reachable_count"] == 2
    assert archive["exp5167-capstone-v473"]["classification"] == "capstone_stale_exp5161_exclusion"

    correction = artifact["capstone_stale_exclusions_corrected"]
    assert correction["capstone_excluded_task_ids"] == [
        "exp5161-gap4-protocol-execution-pilot-v473"
    ]
    assert correction["exp5161_removed_from_exclusion"] is True
    assert correction["live_exp5161_flagged_adversarial"] is False
    assert "exp5163-mmlu-pro-verifier-rescale-v473" in correction["not_headline_clean_task_ids"]

    assert artifact["arc_registry_reconciliation"]["drift_detected"] is False
    assert artifact["arc_registry_reconciliation"]["reproducible_total_levels"] == 69
    assert artifact["arc_registry_reconciliation"]["reproducible_total_games"] == 24
    assert artifact["gap4_status_reconciliation"]["drift_detected"] is False
    assert artifact["gap4_status_reconciliation"]["status_line_matches_capstone"] is True


def test_scenario_report_5168_dirty_runtime_gate_is_visible(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5168-DIRTY-RUNTIME: dirty handoff is recorded as a blocking gate."""

    dirty = mod.RuntimeSnapshot(
        git_status_porcelain=(
            " M ops/status.md\n"
            "?? python/carnot/experiment_5168_archive_473_activate_474.py\n"
            "?? results/experiment_5168_archive_473_activate_474.json\n"
        ),
        process_table=(
            "200 1 Ssl 02:00:00 python scripts/research_conductor.py --loop\n"
            "201 200 Ssl 00:00:59 codex exec --cd /repo -\n"
        ),
    )
    artifact = mod.build_artifact(
        root=make_repo(tmp_path),
        duration_s=1.0,
        run_date="20260702",
        verification=mod.verification_payload(GREEN_VERIFY),
        runtime_snapshot=dirty,
        tests_run=["unit-test-placeholder"],
    )

    mod.validate_artifact(artifact)
    assert artifact["v473_runtime_clean"] is False
    assert artifact["honest_verdict"] == mod.DIRTY_HANDOFF_VERDICT
    assert artifact["runtime_clean_details"]["non_transition_dirty_paths"] == ["ops/status.md"]
    assert artifact["runtime_clean_details"]["ignored_transition_dirty_paths"] == [
        "python/carnot/experiment_5168_archive_473_activate_474.py",
        "results/experiment_5168_archive_473_activate_474.json",
    ]
    assert artifact["runtime_clean_details"]["orphaned_conductor_processes"]


def test_scenario_report_5168_run_preserves_active_roadmap_and_conductor(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5168: activation records readiness without mutating live files."""

    root = make_repo(tmp_path)
    active_before = (root / "research-roadmap.yaml").read_text(encoding="utf-8")
    conductor_before = (root / "scripts" / "research_conductor.py").read_text(encoding="utf-8")

    output = mod.run(
        root=root,
        run_date="20260702",
        clock=iter([100.0, 101.0]).__next__,
        verification_runner=lambda path: GREEN_VERIFY,
        runtime_probe=lambda repo: clean_runtime_snapshot(),
        tests_run=["unit-test-placeholder"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"] == mod.COMPLETE_VERDICT
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    assert (root / "research-roadmap.yaml").read_text(encoding="utf-8") == active_before
    assert (root / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    ) == conductor_before
    mod.validate_artifact(artifact)


def test_req_report_5168_validation_edges_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-5168: validation fails closed and helpers expose readiness gaps."""

    valid = mod.build_artifact(
        root=make_repo(tmp_path / "valid"),
        duration_s=1.0,
        run_date="20260702",
        verification=mod.verification_payload(GREEN_VERIFY),
        runtime_snapshot=clean_runtime_snapshot(),
        tests_run=["unit-test-placeholder"],
    )
    mod.validate_artifact(valid)

    mutations = [
        ("experiment_id", "wrong"),
        ("milestone", "2026.07.473"),
        ("archived_milestone", "2026.07.472"),
        ("honest_verdict", "bad"),
        ("inference_substrate", "live_llm_inference"),
        ("duration_s", 0.0),
        ("source_artifacts_read", []),
        ("task_verdicts", []),
        ("milestone_archive_summary", []),
        ("v473_runtime_clean", "true"),
        ("runtime_clean_details", []),
        ("exp5161_unquarantine_noted", "true"),
        ("capstone_stale_exclusions_corrected", []),
        ("arc_registry_reconciliation", []),
        ("gap4_status_reconciliation", []),
        ("active_roadmap_ready", "true"),
        ("active_roadmap_modified", True),
        ("conductor_modified", True),
        ("flagged_adversarial", "false"),
        ("tests_run", []),
        ("reproducibility_checksum", "bad"),
    ]
    for key, value in mutations:
        payload = copy.deepcopy(valid)
        payload[key] = value
        with pytest.raises(ValueError):
            mod.validate_artifact(payload)

    payload = copy.deepcopy(valid)
    payload.pop("tests_run")
    with pytest.raises(ValueError, match="invalid Exp 5168 archive artifact"):
        mod.validate_artifact(payload)

    payload = copy.deepcopy(valid)
    payload["field_principles"]["tests_run"] = "wrong"
    with pytest.raises(ValueError, match="invalid Exp 5168 archive artifact"):
        mod.validate_artifact(payload)

    assert mod._roadmap_check(tmp_path / "missing.yaml")["ready"] is False
    (tmp_path / "poison.yaml").write_text("a: : :\n- [\n", encoding="utf-8")
    assert mod._roadmap_check(tmp_path / "poison.yaml")["parses"] is False
    assert mod._vnext_check(tmp_path / "missing.md")["ready"] is False
    assert mod._as_int(True, default=-7) == -7
    assert mod._as_int("not-an-int", default=-8) == -8
    assert mod._as_float(False) is None
    assert mod._as_float("not-a-float") is None
    assert mod._m450_validation({"validated_milestones": [{"milestone": "2026.07.472"}]}) == {}
    assert mod._registry_reconciliation(tmp_path / "missing-registry.yaml", {})["exists"] is False
    (tmp_path / "bad-registry.yaml").write_text("a: : :\n- [\n", encoding="utf-8")
    assert mod._registry_reconciliation(tmp_path / "bad-registry.yaml", {})["parses"] is False
    assert mod._gap4_status_reconciliation(tmp_path / "missing-gaps.md", {})["exists"] is False

    gated = mod.build_artifact(
        root=make_repo(tmp_path / "gated", active_valid=False),
        duration_s=1.0,
        run_date="20260702",
        verification=mod.verification_payload(GREEN_VERIFY),
        runtime_snapshot=clean_runtime_snapshot(),
        tests_run=["unit-test-placeholder"],
    )
    assert gated["honest_verdict"] == mod.ACTIVATION_GATED_VERDICT
    mod.validate_artifact(gated)

    missing_inputs = mod.build_artifact(
        root=make_repo(tmp_path / "missing_inputs", omit_results={5160}),
        duration_s=1.0,
        run_date="20260702",
        verification=mod.verification_payload(GREEN_VERIFY),
        runtime_snapshot=clean_runtime_snapshot(),
        tests_run=["unit-test-placeholder"],
    )
    assert missing_inputs["honest_verdict"] == mod.MISSING_INPUTS_VERDICT
    mod.validate_artifact(missing_inputs)

    drift = mod.build_artifact(
        root=make_repo(tmp_path / "drift", registry_levels=70, gap4_valid=False),
        duration_s=1.0,
        run_date="20260702",
        verification=mod.verification_payload(GREEN_VERIFY),
        runtime_snapshot=clean_runtime_snapshot(),
        tests_run=["unit-test-placeholder"],
    )
    assert drift["honest_verdict"] == mod.ACTIVATION_GATED_VERDICT
    assert drift["arc_registry_reconciliation"]["drift_detected"] is True
    assert drift["gap4_status_reconciliation"]["drift_detected"] is True
    mod.validate_artifact(drift)

    root = make_repo(tmp_path / "cli_repo")
    (root / "scripts" / "adversarial_verify.py").write_text(
        "import json\nprint(json.dumps({'flags': []}))\n",
        encoding="utf-8",
    )
    output = root / "module_cli_result.json"
    monkeypatch.setattr(mod, "capture_runtime_snapshot", lambda repo: clean_runtime_snapshot())
    assert mod.main(["--root", str(root), "--output", str(output), "--date", "20260702"]) == 0
    assert output.exists()
