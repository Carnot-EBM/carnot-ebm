"""Tests for Exp 5181 archive .474 / activate .475 aggregation.

Spec refs: REQ-REPORT-5181, SCENARIO-REPORT-5181,
SCENARIO-REPORT-5181-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_5181_archive_474_activate_475 as mod


CLEAN_PUBLICATION_GATE = {
    "paper_ready": True,
    "unmet_gates": [],
    "gates": {"G1": {"pass": True}, "G2": {"pass": True}, "G3": {"pass": True}, "G4": {"pass": True}},
}

CLEAN_LINT = mod.CommandResult(
    command=(".venv/bin/python", "scripts/exclusion_manifest_lint.py", "research-roadmap.yaml"),
    exit_code=0,
    stdout=(
        "Exclusion-manifest lint found 2 violation(s) in research-roadmap.yaml:\n"
        "WARNING violations (2, override present):\n"
        "All violations have operator_override -- activation would proceed with warnings.\n"
    ),
    stderr="",
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _wrapped(value: object) -> dict[str, object]:
    return {"principle": "test principle", "value": value}


def _payloads() -> dict[int, dict]:
    phase_d_sources = [
        "exp4940",
        "distributional_energy_verifier_musr",
        "exp5003",
        "exp5004",
        "exp5005",
        "exp5007",
        "exp5015",
        "exp5017",
        "exp5018",
        "exp5022",
        "exp5029",
        "exp5031",
        "exp5032",
        "exp5033",
        "exp5036",
        "exp5045",
        "exp5046",
        "exp5047",
        "exp5050",
        "exp5059",
        "exp5060",
        "exp5063",
        "exp5072",
        "exp5086",
        "exp5087",
        "exp5088",
        "exp5126",
        "exp5163",
    ]
    return {
        5168: {
            "experiment_id": "exp5168-archive-473-activate-474",
            "honest_verdict": "complete_archive_473_closed_474_active_runtime_clean_exp5161_unquarantined",
            "v473_runtime_clean": True,
            "exp5161_unquarantine_noted": True,
        },
        5169: {
            "experiment": "experiment_5169_adversarial_verify_qd_citation_scope_fix_v474",
            "honest_verdict": "complete: exp5156_resolves_clean_qd_citation_scope_fixed_warn_only_not_quarantine",
            "exp5156_resolved": _wrapped(True),
            "backfill_dry_run_summary": _wrapped(
                {
                    "artifacts_newly_unflagged_count": 4,
                    "artifacts_newly_flagged_count": 2,
                    "any_unexpected_unflag": False,
                }
            ),
        },
        5170: {
            "experiment_id": "exp5170-retire-phase-d-external-text-scorer-v474",
            "honest_verdict": "complete: phase_d_external_text_scorer_scope_retired_and_hidden_state_exception_preserved",
            "phase_d_artifacts_enumerated": phase_d_sources,
            "false_positive_check_against_exp5178": True,
            "synthetic_match_check_passed": True,
            "manifest_entry_audit": {"found": True, "errors": []},
            "lineage_stage_summary": {
                "cleanest_point_estimate": {"source_id": "exp5031", "delta_vs_tuned_sc": 0.08, "ci95": [0.0, 0.165]},
                "terminal_continuation": {
                    "source_id": "exp5163",
                    "delta": 0.025,
                    "ci95": [-0.125, 0.175],
                    "flagged_adversarial": True,
                },
            },
        },
        5171: {
            "experiment": "experiment_5171_harden_set_encoder_cross_corpus_n30_v474",
            "honest_verdict": "success_arc_set_encoder_cross_corpus_gate_passed_n30: gate passed at n>=30",
            "gate_passed": True,
            "held_out_task_n": 30,
            "cross_corpus_delta_n30": 0.5,
            "cross_corpus_delta_ci95_n30": [0.3333333333, 0.6666666667],
            "per_seed_deltas": [0.5, 0.5, 0.5, 0.5, 0.5],
            "random_seeds_used": [5171, 5172, 5173, 5174, 5175],
            "verifier_is_oracle": False,
        },
        5172: {
            "experiment_id": "experiment_5172_sota_ingestion_diffusion_hierarchical_search_v474",
            "honest_verdict": "complete: map_deep_read_recommends_map_pre_stage_if_phase_b_pruner_stalls",
            "bottom_line_recommendation_for_475": _wrapped("MAP should be prototyped next."),
            "map_paper_deep_read": _wrapped(
                {
                    "comparison_vs_relational_mask_pruner": (
                        "Falsifiable .475 gate: on CD82/SK48/SP80, run pruner-only, map-only, "
                        "and map-plus-pruner under the same 4000-expansion and reproduction-gated protocol."
                    )
                }
            ),
        },
        5173: {
            "experiment_id": "experiment_5173_diffusiongemma_energy_guided_diffusion_pilot_v474",
            "honest_verdict": "blocked_diffusiongemma_meta_tensor_bug_unresolved",
            "arm_rows": [],
            "preconditions": {
                "smoke": {
                    "success": False,
                    "error": "ValueError: Some modules are dispatched on the CPU or the disk before forward",
                    "tried": ["wrong auto class", "device_map=auto", "device_map=auto + max_memory"],
                }
            },
            "meta_tensor_bug_resolution": _wrapped("blocked_diffusiongemma_meta_tensor_bug_unresolved"),
            "corrigendum_pending": [{"kind": "CIRCULAR_MOAT_OVERCLAIM"}, {"kind": "MOAT_CLAIM_RIGOR"}],
            "verifier_is_oracle": _wrapped(True),
            "flagged_adversarial": False,
        },
        5174: {
            "experiment_id": "5174",
            "honest_verdict": (
                "complete: original three GAP-LIVE-INTEGRATION claims were stale; current code imports "
                "router/DSL, ships target_levels=3, and uses a nonzero submitted value_weight while Exp4652 "
                "records a tried-nonzero honest null; provenance audit finds 4/24 declared registry games "
                "live-self-discovery vs 20/24 development-proxy."
            ),
            "claim_router_dsl_unimported": _wrapped(False),
            "claim_target_levels_1": _wrapped(False),
            "claim_value_weight_0": _wrapped(False),
            "solve_provenance_audit": _wrapped(
                {
                    "live_agent_self_discovery_count": 4,
                    "development_proxy_count": 20,
                    "out_of_registry_declared_games": 24,
                    "row_level_counts": {"live_agent_self_discovery": 4, "development_proxy": 21},
                }
            ),
        },
        5175: {
            "experiment": "experiment_5175_gap4891_relational_mask_pruner_ab_v474",
            "honest_verdict": (
                "complete_relational_mask_pruner_prunes_edges_but_states_expanded_unchanged_no_level_bank_"
                "pruning_alone_does_not_close_enumeration_wall_MAP_map_then_act_next"
            ),
            "games_tested": ["cd82", "sk48", "sp80", "cn04"],
            "states_expanded_unpruned": {"cd82": 4000, "sk48": 4000, "sp80": 4000, "cn04": 4000},
            "states_expanded_pruned": {"cd82": 4000, "sk48": 4000, "sp80": 4000, "cn04": 4000},
            "move_pruned_edges": {"cd82": 358, "sk48": 22807, "sp80": 0, "cn04": 375},
            "levels_banked": [],
            "next_specific_lever": (
                "Prototype a MAP-style map-then-act / hierarchical pre-search stage that generates "
                "candidate subgoal trajectories before flat frontier enumeration."
            ),
        },
        5176: {
            "experiment": "experiment_5176_deepen_live_levelup_attempt_v474",
            "honest_verdict": "complete_blocked_no_validated_lever_from_b1_b2_zero_levels_banked",
            "levels_banked": [],
            "reproducible_levels_delta": 0,
            "lever_used": "none_available",
        },
        5177: {
            "experiment_id": 5177,
            "honest_verdict": "complete_gap4_scaleup_v474_n62_of_target180_floor_not_crossed_scale_up_recommended",
            "target_n": _wrapped(180),
            "achieved_n": _wrapped(62),
            "checkpoint_resume_used": _wrapped(True),
            "checkpoint_path": _wrapped("results/experiment_5177_gap4_scaleup_decentralization_tier_v474.checkpoint.json"),
            "exact_test_discordant_wins": _wrapped(4),
            "exact_test_discordant_losses": 0,
            "exact_test_p_value_two_sided": _wrapped(0.125),
            "exact_test_passes_min6_rule": _wrapped(False),
            "local_generator_arm_result": _wrapped({"status": "completed_real_local_generator_subset", "achieved_n": 5}),
            "gap4_status_recommendation": _wrapped("scale_up_recommended"),
        },
        5178: {
            "experiment_id": 5178,
            "honest_verdict": _wrapped(
                "complete_hidden_state_verifier_ties_tuned_sc_accuracy_point_lower_efficiency_loses_to_sc_"
                "extra_hidden_forward_wins_vs_llm_judge_no_decode_hidden0.000_sc0.333_delta-0.333"
            ),
            "hidden_state_access_feasible": _wrapped(True),
            "design_path_taken": _wrapped("trajselector_trained_probe: final-token hidden vectors only"),
            "tuned_sc_baseline_accuracy": _wrapped(0.333333),
            "hidden_state_verifier_accuracy": _wrapped(0.0),
            "accuracy_delta_ci95": _wrapped([-0.666667, 0.0]),
            "mcnemar_p_value": _wrapped(0.5),
            "pilot_n_questions": 6,
            "pilot_n_candidates": 48,
            "oracle_at_k_accuracy": 1.0,
            "verifier_is_oracle": _wrapped(False),
            "headroom_present": _wrapped(True),
            "flagged_adversarial": False,
        },
        5179: {
            "experiment_id": "exp5179-hardware-continuity-board-timing-v474",
            "honest_verdict": (
                "complete_hardware_continuity_board_timing_kv260:reachable_gatemate:"
                "blocked_gatemate_dirtyjtag_idcode_unresolved_after_diagnostics_polarfire:reachable_no_speedup_claim"
            ),
            "boards_reachable_count": 2,
            "kv260_result": {"reachable": True, "hash_verified": True},
            "polarfire_result": {"reachable": True, "hash_verified": True},
            "gatemate_result": {
                "reachable": False,
                "blocked_reason": "blocked_gatemate_dirtyjtag_idcode",
                "timing_output": {"expected_idcode": "0x20000001"},
            },
            "conductor_modified": False,
        },
        5180: {
            "experiment_id": "exp5180-capstone-v474",
            "honest_verdict": (
                "complete: v474 reconciled with no flagged headline artifacts after live verification, "
                "GAP-4891 and GAP-4 still open but sharpened, DiffusionGemma blocked before measurement, "
                "Phase D retirement clean, and zero new ARC levels banked."
            ),
            "flagged_adversarial": False,
            "flagged_adversarial_artifacts_excluded": _wrapped([]),
            "publication_gate": {"paper_ready": True, "unmet_gates": []},
            "registry_reconciliation": {
                "reproducible_total_levels": 69,
                "reproducible_total_games": 24,
                "delta_from_exp5175_exp5176": 0,
            },
            "phase_d_retirement_confirmed_clean": _wrapped(True),
        },
    }


def _manifest_entry() -> dict:
    return {
        "id": mod.PHASE_D_ENTRY_ID,
        "experiment_scope": (
            "external-TEXT-based energy/reward verifier scoring (LoRA-EBM holistic scorer / uPRM / "
            "EBRM style) vs. self-consistency on off-ARC reasoning corpora"
        ),
        "reason": (
            "hidden-state/internal-representation verifiers, ARC oracle-distinct verifier work, and "
            "the FoVer production ensemble are outside this retired scope."
        ),
        "experiment_ids": [item for item in mod.PHASE_D_RETIRED_EXP_IDS],
        "blocked_patterns": [
            "train lora ebm scorer v2 on off-arc reasoning corpus",
            "rerun uprm text scorer on off-arc reasoning corpus",
            "rerun ebrm external text reward scorer on off-arc reasoning corpus",
            "phase d external text scorer rerun",
        ],
    }


def make_repo(
    tmp_path: Path,
    *,
    active_roadmap: bool = True,
    manifest_entry: dict | None = None,
    omit_artifact: int | None = None,
) -> Path:
    root = tmp_path
    (root / "results").mkdir(parents=True)
    (root / "ops").mkdir(parents=True)
    (root / "_bmad").mkdir(parents=True)
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "research_conductor.py").write_text("# conductor\n", encoding="utf-8")
    tasks = [{"id": f"exp{exp_id}-task", "title": f"task {exp_id}"} for exp_id in range(5181, 5193)]
    if active_roadmap:
        tasks[0]["id"] = "exp5181-archive-474-activate-475"
        tasks[4]["id"] = "exp5185-map-landmark-prestage-gap4891-v475"
        tasks[6]["id"] = "exp5187-hidden-state-verifier-v2-v475"
    (root / "research-roadmap.yaml").write_text(
        yaml.safe_dump({"milestone": "2026.07.475" if active_roadmap else "2026.07.474", "tasks": tasks}),
        encoding="utf-8",
    )
    (root / "ops" / "exclusion_manifest.yaml").write_text(
        yaml.safe_dump({"retired_extras": [manifest_entry if manifest_entry is not None else _manifest_entry()]}),
        encoding="utf-8",
    )
    (root / "_bmad" / "architecture.md").write_text(
        "# Architecture\n\n**Last Reconciled:** 2026-05-16\n",
        encoding="utf-8",
    )
    for exp_id, payload in _payloads().items():
        if exp_id == omit_artifact:
            continue
        _write_json(root / mod.V474_RESULT_PATHS[exp_id], payload)
    return root


def test_req_report_5181_spec_declares_archive_contract() -> None:
    """REQ-REPORT-5181: OpenSpec anchors the .474 archive and .475 activation contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    for marker in (
        "REQ-REPORT-5181",
        "SCENARIO-REPORT-5181",
        "SCENARIO-REPORT-5181-BLOCKED-PRECONDITION",
        "results/experiment_5181_archive_474_activate_475.json",
        "v474_summary",
        "architecture_md_staleness_days",
        "aggregation_from_upstream_artifacts",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle in spec


def test_scenario_report_5181_happy_path_preserves_precise_v474_truth(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5181: precise .474 outcomes and .475 activation are recorded."""

    artifact = mod.build_artifact(
        root=make_repo(tmp_path),
        duration_s=1.25,
        run_date="20260703",
        publication_gate=CLEAN_PUBLICATION_GATE,
        exclusion_lint=CLEAN_LINT,
        tests_run=["unit-test-placeholder"],
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == "exp5181-archive-474-activate-475"
    assert artifact["milestone"] == "2026.07.475"
    assert artifact["archived_milestone"] == "2026.07.474"
    assert artifact["honest_verdict"]["value"] == mod.COMPLETE_VERDICT
    assert artifact["inference_substrate"]["value"] == "aggregation_from_upstream_artifacts"
    assert artifact["exclusion_manifest_confirmed_clean"]["value"] is True
    assert artifact["research_roadmap_yaml_activated"]["value"] is True
    assert artifact["architecture_md_staleness_days"]["value"] == 48
    assert artifact["research_conductor_modified"] is False

    summary = artifact["v474_summary"]["value"]
    for verdict in (
        "success_arc_set_encoder_cross_corpus_gate_passed_n30: gate passed at n>=30",
        "blocked_diffusiongemma_meta_tensor_bug_unresolved",
        "complete_gap4_scaleup_v474_n62_of_target180_floor_not_crossed_scale_up_recommended",
        "complete_blocked_no_validated_lever_from_b1_b2_zero_levels_banked",
    ):
        assert verdict in summary
    assert "28 source artifacts" in summary
    assert "27 retired exp* IDs" in summary
    assert "n=62/180" in summary
    assert "checkpoint missing on disk" in summary
    assert "69/24" in summary

    rows = {row["exp_id"]: row for row in artifact["v474_task_rows"]}
    assert len(rows) == 13
    assert rows[5171]["key_facts"]["gate_passed"] is True
    assert rows[5171]["key_facts"]["ci95"] == [0.3333333333, 0.6666666667]
    assert rows[5173]["key_facts"]["arm_rows"] == []
    assert rows[5175]["key_facts"]["move_pruned_edges"] == {"cd82": 358, "sk48": 22807, "sp80": 0, "cn04": 375}
    assert rows[5177]["key_facts"]["checkpoint_exists"] is False
    assert rows[5178]["key_facts"]["hidden_state_verifier_accuracy"] == 0.0
    assert rows[5180]["key_facts"]["flagged_adversarial_artifacts_excluded"] == []

    assert artifact["phase_d_manifest_audit"]["retired_exp_id_count"] == 27
    assert artifact["phase_d_manifest_audit"]["source_artifact_count"] == 28
    assert artifact["publication_gate"]["paper_ready"] is True
    assert artifact["publication_gate"]["unmet_gates"] == []


def test_scenario_report_5181_blocked_preconditions_are_visible(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5181-BLOCKED-PRECONDITION: failed inputs block clean handoff claims."""

    bad_lint = mod.CommandResult(
        command=CLEAN_LINT.command,
        exit_code=1,
        stdout="[HARD] exp5187-hidden-state-verifier-v2-v475 phase d external text scorer rerun",
        stderr="",
    )
    overbroad = copy.deepcopy(_manifest_entry())
    overbroad["reason"] = "external text scorer retired"

    cases = [
        mod.build_artifact(
            root=make_repo(tmp_path / "inactive", active_roadmap=False),
            duration_s=1.0,
            run_date="20260703",
            publication_gate=CLEAN_PUBLICATION_GATE,
            exclusion_lint=CLEAN_LINT,
            tests_run=["unit-test-placeholder"],
        ),
        mod.build_artifact(
            root=make_repo(tmp_path / "missing", omit_artifact=5171),
            duration_s=1.0,
            run_date="20260703",
            publication_gate=CLEAN_PUBLICATION_GATE,
            exclusion_lint=CLEAN_LINT,
            tests_run=["unit-test-placeholder"],
        ),
        mod.build_artifact(
            root=make_repo(tmp_path / "overbroad", manifest_entry=overbroad),
            duration_s=1.0,
            run_date="20260703",
            publication_gate=CLEAN_PUBLICATION_GATE,
            exclusion_lint=bad_lint,
            tests_run=["unit-test-placeholder"],
        ),
    ]

    for artifact in cases:
        mod.validate_artifact(artifact)
        assert artifact["honest_verdict"]["value"] == mod.BLOCKED_VERDICT
        assert artifact["clean_handoff"] is False
        assert artifact["failed_preconditions"]

    assert cases[0]["research_roadmap_yaml_activated"]["value"] is False
    assert cases[1]["source_artifact_audit"]["all_present"] is False
    assert cases[2]["exclusion_manifest_confirmed_clean"]["value"] is False


def test_req_report_5181_validation_edges_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-5181: validation fails closed and the CLI writes the artifact."""

    root = make_repo(tmp_path / "repo")
    artifact = mod.build_artifact(
        root=root,
        duration_s=1.0,
        run_date="20260703",
        publication_gate=CLEAN_PUBLICATION_GATE,
        exclusion_lint=CLEAN_LINT,
        tests_run=["unit-test-placeholder"],
    )
    mod.validate_artifact(artifact)

    mutations = [
        ("schema", "wrong"),
        ("experiment_id", "wrong"),
        ("milestone", "2026.07.474"),
        ("archived_milestone", "2026.07.473"),
        ("v474_summary", {"value": "", "principle": mod.FIELD_PRINCIPLES["v474_summary"]}),
        ("exclusion_manifest_confirmed_clean", {"value": "true", "principle": mod.FIELD_PRINCIPLES["exclusion_manifest_confirmed_clean"]}),
        ("research_roadmap_yaml_activated", {"value": "true", "principle": mod.FIELD_PRINCIPLES["research_roadmap_yaml_activated"]}),
        ("architecture_md_staleness_days", {"value": "48", "principle": mod.FIELD_PRINCIPLES["architecture_md_staleness_days"]}),
        ("inference_substrate", {"value": "live_llm_inference", "principle": mod.FIELD_PRINCIPLES["inference_substrate"]}),
        ("honest_verdict", {"value": "bad", "principle": mod.FIELD_PRINCIPLES["honest_verdict"]}),
        ("v474_task_rows", []),
        ("source_artifact_audit", {}),
        ("publication_gate", {"paper_ready": False, "unmet_gates": ["G2"]}),
        ("reproducibility_checksum", "bad"),
    ]
    for key, value in mutations:
        payload = copy.deepcopy(artifact)
        payload[key] = value
        with pytest.raises(ValueError):
            mod.validate_artifact(payload)

    payload = copy.deepcopy(artifact)
    payload["field_principles"]["v474_summary"] = "wrong"
    with pytest.raises(ValueError):
        mod.validate_artifact(payload)

    payload = copy.deepcopy(artifact)
    payload.pop("tests_run")
    with pytest.raises(ValueError):
        mod.validate_artifact(payload)

    payload = copy.deepcopy(artifact)
    payload["v474_summary"] = {"principle": "wrong", "value": "summary"}
    with pytest.raises(ValueError):
        mod.validate_artifact(payload)

    payload = copy.deepcopy(artifact)
    payload["v474_summary"] = {"principle": mod.FIELD_PRINCIPLES["v474_summary"]}
    with pytest.raises(ValueError):
        mod.validate_artifact(payload)

    assert mod._int(True, default=-7) == -7
    assert mod._int("not-an-int", default=-8) == -8
    assert mod._float(False) is None
    assert mod._float("not-a-float") is None

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json\n", encoding="utf-8")
    assert mod.read_json_mapping(bad_json)[1]["loadable"] is False
    list_json = tmp_path / "list.json"
    list_json.write_text("[]\n", encoding="utf-8")
    assert mod.read_json_mapping(list_json)[1]["error"] == "top-level JSON is not an object"

    poison_roadmap = tmp_path / "poison-roadmap.yaml"
    poison_roadmap.write_text("a: : :\n- [\n", encoding="utf-8")
    assert mod._roadmap_activation_check(poison_roadmap)["parses"] is False
    no_date_arch = tmp_path / "architecture-no-date.md"
    no_date_arch.write_text("# Architecture\n", encoding="utf-8")
    assert mod._architecture_staleness_days(no_date_arch, "20260703") == -1

    poison_manifest = tmp_path / "poison-manifest.yaml"
    poison_manifest.write_text("a: : :\n- [\n", encoding="utf-8")
    assert mod._manifest_audit(poison_manifest, [])["parses"] is False
    missing_entry_manifest = tmp_path / "missing-entry-manifest.yaml"
    missing_entry_manifest.write_text(yaml.safe_dump({"retired_extras": []}), encoding="utf-8")
    assert mod._manifest_audit(missing_entry_manifest, [])["entry_found"] is False
    bad_entry = copy.deepcopy(_manifest_entry())
    bad_entry["experiment_ids"] = ["exp5163"]
    bad_entry["experiment_scope"] = "hidden-state/internal-representation verifier"
    bad_manifest = tmp_path / "bad-entry-manifest.yaml"
    bad_manifest.write_text(yaml.safe_dump({"retired_extras": [bad_entry]}), encoding="utf-8")
    bad_audit = mod._manifest_audit(bad_manifest, [])
    assert "retired_exp_ids_mismatch" in bad_audit["errors"]
    assert "scope_not_external_text_off_arc" in bad_audit["errors"]

    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("# python\n", encoding="utf-8")
    assert mod._python_executable(tmp_path) == str(venv_python)
    failures = mod._failed_preconditions(
        source_audit={"all_present": True, "all_loadable": True},
        manifest_audit={"clean": True},
        lint_audit={"clean": True},
        roadmap_check={"activated": True},
        architecture_days=-1,
        publication_gate={"paper_ready": False, "unmet_gates": ["G2"]},
        conductor_modified=True,
    )
    assert failures == [
        "architecture_last_reconciled_unreadable",
        "publication_gate_not_ready",
        "scripts_research_conductor_py_modified",
    ]

    assert mod._architecture_staleness_days(tmp_path / "missing.md", "20260703") == -1
    assert mod._roadmap_activation_check(tmp_path / "missing.yaml")["activated"] is False
    assert mod._publication_gate_clean({"paper_ready": True}) is False
    assert mod._manifest_audit(tmp_path / "missing.yaml", [])["entry_found"] is False
    assert mod._command_clean(mod.CommandResult(("cmd",), 0, "warnings only", "")) is True
    assert mod._command_clean(mod.CommandResult(("cmd",), 0, "HARD violation", "")) is False

    output = root / "results" / "cli.json"
    monkeypatch.setattr(mod, "run_publication_gate", lambda repo: CLEAN_PUBLICATION_GATE)
    monkeypatch.setattr(mod, "run_exclusion_manifest_lint", lambda repo: CLEAN_LINT)
    assert mod.main(["--root", str(root), "--output", str(output), "--date", "20260703"]) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    mod.validate_artifact(written)
