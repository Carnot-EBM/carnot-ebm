"""Tests for REQ-CAPSTONE-5027 / SCENARIO-CAPSTONE-5027."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_5027_capstone_v462 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _arm_row(
    *,
    arm: str = "EBRM",
    arm_id: str = "D3",
    delta: float | None = None,
    ci: list[float] | None = None,
    execution_status: str = "blocked",
    source_experiment_id: int = 5019,
) -> JsonDict:
    return {
        "abstention_rate": None,
        "arm": arm,
        "arm_id": arm_id,
        "corpus": "MuSR",
        "delta_vs_tuned_sc": delta,
        "execution_status": execution_status,
        "genuine_tuned_sc_accuracy": None,
        "headroom_present": delta is not None,
        "honest_verdict": "blocked_gate_check_failed",
        "mcnemar_p": None,
        "n_questions": 0 if execution_status == "blocked" else 200,
        "oracle_at_k": None,
        "paired_ci95": ci,
        "scorer_trained": None,
        "selection_accuracy": None,
        "source_experiment_id": source_experiment_id,
        "verifier_is_oracle": False,
        "win_vs_tuned_sc": bool(delta and delta > 0),
    }


def _d5(
    *,
    decision: str = "EXECUTION-INCOMPLETE",
    moat_realized: bool = False,
    moat_retired_bounded: bool = False,
    efficiency_win: bool = False,
    best_arm: JsonDict | None = None,
) -> JsonDict:
    row = _arm_row()
    return {
        "experiment": "experiment_5022_moat_gate_resolution_v2",
        "experiment_id": 5022,
        "honest_verdict": "complete_moat_execution_incomplete_ebrm",
        "decision": decision,
        "moat_realized": moat_realized,
        "moat_retired_bounded": moat_retired_bounded,
        "efficiency_win": efficiency_win,
        "best_arm": best_arm,
        "per_arm_table": [row],
        "execution_incomplete_arms": [
            {
                "arm": "EBRM",
                "arm_id": "D3",
                "corpus": "MuSR",
                "execution_status": "blocked",
                "honest_verdict": "blocked_gate_check_failed",
                "source_experiment_id": 5019,
            }
        ],
        "diffusiongemma_gate_status": "STILL-PENDING",
        "diffusiongemma_gate_conditions_satisfied_off_arc": False,
        "diffusiongemma_activation": "not_activated",
        "flagged_arms_skipped": [
            {"arm": "LoRA-EBM", "arm_id": "D1", "experiment_id": 5017},
            {"arm": "uPRM", "arm_id": "D2", "experiment_id": 5018},
            {"arm": "cascade", "arm_id": "D6", "experiment_id": 5020},
            {
                "arm": "second-corpus-confirmation",
                "arm_id": "D4",
                "experiment_id": 5021,
            },
        ],
        "paper_summary": "D5 is execution-incomplete; D3 did not cleanly execute.",
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _d_arm(
    experiment_id: int,
    arm_id: str,
    *,
    flagged: bool = False,
    delta: float | None = 0.0,
    ci: list[float] | None = None,
    execution_status: str = "complete",
) -> JsonDict:
    return {
        "experiment": f"experiment_{experiment_id}_{arm_id.lower()}",
        "experiment_id": experiment_id,
        "honest_verdict": f"complete_{arm_id.lower()}",
        "flagged_adversarial": flagged,
        "delta_vs_tuned_sc": delta,
        "delta_vs_tuned_sc_second": delta,
        "paired_ci95": ci if ci is not None else [-0.03, 0.025],
        "paired_ci95_second": ci if ci is not None else [-0.03, 0.025],
        "mcnemar_p": 1.0,
        "mcnemar_p_second": 1.0,
        "headroom_present": delta is not None and execution_status == "complete",
        "oracle_at_k": 0.865 if delta is not None else None,
        "oracle_at_k_second": 0.87 if delta is not None else None,
        "genuine_tuned_sc_accuracy": 0.585 if delta is not None else None,
        "genuine_tuned_sc_accuracy_second": 0.6 if delta is not None else None,
        "trained_scorer_accuracy": 0.585 if delta is not None else None,
        "uprm_selection_accuracy": 0.585 if delta is not None else None,
        "second_corpus_accuracy": 0.6 if delta is not None else None,
        "n_questions": 200 if execution_status == "complete" else 0,
        "scorer_trained": delta is not None,
        "abstention_rate": 0.0 if delta is not None else None,
        "verifier_is_oracle": False,
        "execution_status": execution_status,
    }


def _artifacts(*, include_d5: bool = True) -> dict[str, JsonDict]:
    artifacts: dict[str, JsonDict] = {
        "B1_GENUINE_SC": {
            "experiment": "experiment_5015_genuine_sc_baseline_fix",
            "experiment_id": 5015,
            "honest_verdict": "success_genuine_sc_baseline_fixed_degeneracy_guard_shipped",
            "genuine_tuned_sc_accuracy": 0.585,
            "genuine_headroom_present": True,
            "oracle_at_k": 0.865,
            "degeneracy_guard_fires": True,
            "single_candidate_degenerate": False,
            "n_questions": 200,
        },
        "B2_LOGPROB_CACHE": {
            "experiment": "experiment_5016_shared_logprob_candidate_cache",
            "experiment_id": 5016,
            "honest_verdict": "blocked_generation_or_cache_error",
            "candidate_cache_built": False,
            "cache_jsonl_path": "results/experiment_5016_shared_logprob_candidate_cache_musr.jsonl",
            "n_cached_rows": 0,
            "n_questions": 0,
            "corpora_cached": [],
            "has_per_token_logprobs": False,
        },
        "D1_LORA_EBM": _d_arm(5017, "D1", flagged=True, delta=None),
        "D2_UPRM": _d_arm(5018, "D2", flagged=True, delta=None),
        "D3_EBRM": {
            "experiment": 5019,
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "exp5017-d1.scorer_trained failed",
            "blocked_at_layer": "conductor_pre_gate",
        },
        "D6_CASCADE": {
            **_d_arm(5020, "D6", flagged=True, delta=None),
            "honest_verdict": "blocked_judge_server",
            "cascade_accuracy": None,
            "judge_only_accuracy": None,
            "judge_call_fraction": None,
        },
        "D4_SECOND_CORPUS": _d_arm(5021, "D4", flagged=True, delta=None),
        "C_KV260": {
            "experiment": "experiment_5023_kv260_continuity",
            "honest_verdict": "success_kv260_reachable_overlay_loaded_energy_ok",
            "kv260_ssh_reachable": True,
            "loaded_overlay": "carnot_ising_v2_n64",
            "energy_smoke": {"success": True, "energy": -7, "expected_energy": -7},
            "uio_devices": ["/dev/uio0"],
            "xmutil_requires_sudo": True,
        },
        "E1_SOTA": {
            "honest_verdict": "success_sota_ingested_5_new_papers_mapped_to_phase_d",
            "new_arxiv_ids": ["2502.11250", "2605.11334", "2507.01951", "2605.30085", "2502.14356"],
            "next_milestone_candidates": [{"candidate": "Rerun D3"}],
            "sota_to_phase_d_mapping": [{"arxiv_id": "2502.11250"}],
            "d5_conditioning": {"decision": "EXECUTION-INCOMPLETE"},
        },
        "E2_SELF_PLAY": {
            "experiment": "experiment_5025_self_play_verifier_checkpoint",
            "honest_verdict": "success_self_play_checkpoint_refreshed",
            "verifier_checkpoint_refreshed": True,
            "checkpoint_path": "models/arc_verifier_lf52.json",
            "target_game": "lf52",
            "offline_reproduced": True,
            "reproduced_levels": 2,
            "solve_provenance": "live_agent_self_discovery",
            "flag_resolved": True,
        },
        "E3_ARC_LEVEL": {
            "experiment": "experiment_5026_levelup_attempt",
            "experiment_id": 5026,
            "honest_verdict": "complete_lp85_no_new_level_residual_no_grounded_l6_delta",
            "target_game": "lp85",
            "target_level": 6,
            "new_levels_banked": 0,
            "offline_reproduced": False,
            "reproduced_levels": 5,
            "reproducible_total_levels_after": 69,
            "live_path_reachable": True,
        },
    }
    if include_d5:
        artifacts["D5_MOAT_GATE"] = _d5()
    return artifacts


def _hashes() -> dict[str, str]:
    return {source: f"sha256:{source.lower()}" for source in mod.UPSTREAM_SOURCES}


def _direct_clean_artifacts(*, d1_delta: float | None, d2_delta: float | None) -> dict[str, JsonDict]:
    artifacts = _artifacts(include_d5=False)
    artifacts["D1_LORA_EBM"] = _d_arm(5017, "D1", delta=d1_delta)
    artifacts["D2_UPRM"] = _d_arm(5018, "D2", delta=d2_delta)
    artifacts["D3_EBRM"] = _d_arm(5019, "D3", delta=0.0)
    artifacts["D6_CASCADE"] = {
        **_d_arm(5020, "D6", delta=0.0),
        "cascade_accuracy": 0.6,
        "judge_only_accuracy": 0.6,
        "paired_ci95_cascade_vs_judge": [-0.01, 0.01],
    }
    artifacts["D4_SECOND_CORPUS"] = _d_arm(5021, "D4", delta=0.0)
    return artifacts


def test_req_capstone_5027_spec_declares_v462_scorecard_contract() -> None:
    """REQ-CAPSTONE-5027: OpenSpec declares the .462 scorecard contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5027") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH in section
    for field in mod.REQUIRED_FIELDS:
        assert field in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section


def test_scenario_capstone_5027_d5_authoritative_execution_incomplete_scorecard() -> None:
    """SCENARIO-CAPSTONE-5027: D5 drives execution-incomplete moat verdict."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "complete_capstone_v462_moat_execution_incomplete_ebrm"
    assert artifact["capstone_ready"] is True
    assert artifact["moat_verdict"]["state"] == "execution_incomplete"
    assert artifact["moat_verdict"]["source"] == "D5_MOAT_GATE"
    assert artifact["moat_verdict"]["execution_incomplete_arms"][0]["arm_id"] == "D3"
    assert artifact["best_arm_and_delta"]["arm_id"] == "D3"
    assert artifact["best_arm_and_delta"]["delta_vs_tuned_sc"] is None
    assert artifact["best_arm_and_delta"]["execution_status"] == "blocked"
    assert artifact["efficiency_win"] is False
    assert artifact["diffusiongemma_gate_status"] == {
        "status": "STILL-PENDING",
        "conditions_satisfied_off_arc": False,
        "activation": "not_activated",
        "operator_gated": True,
        "autonomously_flipped_to_met": False,
    }
    assert artifact["per_arm_table"] == [_d5()["per_arm_table"][0]]
    assert artifact["infra_rollup"]["b1_genuine_sc_baseline"]["built"] is True
    assert artifact["infra_rollup"]["b2_logprob_cache"]["cache_built"] is False
    assert artifact["hardware_rollup"]["kv260_reachable"] is True
    assert artifact["sota_ingestion_rollup"]["new_papers_ingested"] == 5
    assert artifact["self_play_rollup"]["checkpoint_refreshed"] is True
    assert artifact["arc_opportunistic_rollup"]["new_levels_banked"] == 0
    assert artifact["arc_deliverable_locked"] == {
        "locked": True,
        "deliverable": "levels_69_plus_publishable_fover_paper",
        "arc_work_mode": "opportunistic",
    }
    assert artifact["reproducible_total_levels"] == 69
    assert artifact["next_milestone_pointer"]["direction"] == "rerun_unexecuted_arm"
    assert artifact["next_milestone_pointer"]["milestone"] == "2026.06.463"
    assert {row["source"] for row in artifact["flagged_artifacts_skipped"]} == {
        "D1_LORA_EBM",
        "D2_UPRM",
        "D6_CASCADE",
        "D4_SECOND_CORPUS",
    }
    assert not {
        "D1_LORA_EBM",
        "D2_UPRM",
        "D6_CASCADE",
        "D4_SECOND_CORPUS",
    } & {row["source"] for row in artifact["cited_upstream_artifacts"]}
    assert {"D5_MOAT_GATE", "D3_EBRM", "B1_GENUINE_SC", "B2_LOGPROB_CACHE"}.issubset(
        {row["source"] for row in artifact["cited_upstream_artifacts"]}
    )
    assert all(row["sha256"].startswith("sha256:") for row in artifact["cited_upstream_artifacts"])
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert mod.validate_artifact(artifact) == []


def test_req_capstone_5027_flagged_direct_arms_and_blocked_arms_do_not_make_null() -> None:
    """REQ-CAPSTONE-5027: flagged or blocked D arms are failed executions, not clean nulls."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(include_d5=False),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        duration_s=0.001,
    )

    assert artifact["moat_verdict"]["d5_missing"] is True
    assert artifact["moat_verdict"]["state"] == "execution_incomplete"
    assert artifact["best_arm_and_delta"]["arm_id"] == "D3"
    assert artifact["best_arm_and_delta"]["execution_status"] == "blocked"
    assert all(row["arm_id"] == "D3" for row in artifact["per_arm_table"])
    assert {row["experiment_id"] for row in artifact["flagged_artifacts_skipped"]} == {
        5017,
        5018,
        5020,
        5021,
    }
    assert "D5 missing" in artifact["moat_verdict"]["summary"]
    assert mod.validate_artifact(artifact) == []

    realized_best = _arm_row(delta=0.12, ci=[0.02, 0.2], execution_status="complete")
    realized = mod.build_artifact(
        artifacts={**_artifacts(), "D5_MOAT_GATE": _d5(
            decision="MOAT-REALIZED",
            moat_realized=True,
            best_arm=realized_best,
        )},
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        duration_s=0.001,
    )
    assert realized["moat_verdict"]["state"] == "moat_realized"
    assert realized["next_milestone_pointer"]["direction"] == "scale_winning_construction"

    retired = mod.build_artifact(
        artifacts={**_artifacts(), "D5_MOAT_GATE": _d5(
            decision="MOAT-RETIRED-BOUNDED",
            moat_retired_bounded=True,
        )},
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        duration_s=0.001,
    )
    assert retired["moat_verdict"]["state"] == "moat_retired_bounded"
    assert retired["next_milestone_pointer"]["direction"] == "pivot_to_e1_sota_candidates"

    mixed = mod.build_artifact(
        artifacts={
            **_artifacts(),
            "D5_MOAT_GATE": {**_d5(decision="MIXED-SCOPED"), "execution_incomplete_arms": []},
        },
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        duration_s=0.001,
    )
    assert mixed["moat_verdict"]["state"] == "mixed_musr_scoped"
    assert mixed["next_milestone_pointer"]["direction"] == "tighten_strongest_arm"

    efficient_best = {
        **_arm_row(arm="cascade", arm_id="D6", delta=0.0, ci=[-0.01, 0.01], execution_status="complete", source_experiment_id=5020),
        "judge_call_fraction": 0.18,
    }
    efficient = mod.build_artifact(
        artifacts={**_artifacts(), "D5_MOAT_GATE": _d5(
            decision="MOAT-REALIZED-CASCADE-EFFICIENCY",
            moat_realized=True,
            efficiency_win=True,
            best_arm=efficient_best,
        )},
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        duration_s=0.001,
    )
    assert efficient["efficiency_win"] is True
    assert efficient["best_arm_and_delta"]["arm_id"] == "D6"


def test_req_capstone_5027_direct_fallback_verdict_branches() -> None:
    """REQ-CAPSTONE-5027: direct fallback handles clean D arms by measured state."""

    realized_artifacts = _direct_clean_artifacts(d1_delta=0.12, d2_delta=0.0)
    realized_artifacts["D1_LORA_EBM"]["paired_ci95"] = [0.02, 0.2]
    realized = mod.build_artifact(
        artifacts=realized_artifacts,
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        duration_s=0.001,
    )
    assert realized["moat_verdict"]["state"] == "moat_realized"
    assert realized["diffusiongemma_gate_status"]["conditions_satisfied_off_arc"] is True
    assert {row["arm_id"] for row in realized["per_arm_table"]} == {
        "D1",
        "D2",
        "D3",
        "D6",
        "D4",
    }

    retired = mod.build_artifact(
        artifacts=_direct_clean_artifacts(d1_delta=0.0, d2_delta=0.0),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        duration_s=0.001,
    )
    assert retired["moat_verdict"]["state"] == "moat_retired_bounded"
    assert retired["honest_verdict"] == (
        "complete_capstone_v462_moat_retired_bounded_lora_ebm_and_uprm_both_null"
    )

    mixed = mod.build_artifact(
        artifacts=_direct_clean_artifacts(d1_delta=None, d2_delta=0.0),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        duration_s=0.001,
    )
    assert mixed["moat_verdict"]["state"] == "mixed_musr_scoped"
    assert mixed["next_milestone_pointer"]["direction"] == "tighten_strongest_arm"
    assert mod._best_arm(
        {"D5_MOAT_GATE": {"execution_incomplete_arms": [[], {"arm": "EBRM", "arm_id": "D3"}]}},
        [],
    ) == {"arm": "EBRM", "arm_id": "D3"}
    assert mod._execution_status("D1_LORA_EBM", {"n_questions": 0}) == "blocked"


def test_scenario_capstone_5027_runtime_writes_result_with_absent_arms_recorded(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5027: run writes the deliverable and records absent inputs."""

    present = {
        key: value for key, value in _artifacts().items() if key not in {"C_KV260", "E3_ARC_LEVEL"}
    }
    for key, payload in present.items():
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text("schema_version: 1\nreproducible_total_levels: 69\n", encoding="utf-8")
    spec = tmp_path / mod.SPEC_RELATIVE_PATH
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text("REQ-CAPSTONE-5027\n", encoding="utf-8")

    artifact = mod.run_capstone(root=tmp_path)

    assert artifact["honest_verdict"] == "complete_capstone_v462_moat_execution_incomplete_ebrm"
    assert artifact["capstone_ready"] is True
    assert artifact["hardware_rollup"]["status"] == "missing_or_blocked"
    assert artifact["arc_opportunistic_rollup"]["status"] == "missing_or_blocked"
    assert artifact["preconditions_checked"]["upstream_artifacts"]["C_KV260"]["present"] is False
    assert (
        artifact["preconditions_checked"]["upstream_artifacts"]["E3_ARC_LEVEL"]["present"] is False
    )
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.validate_artifact(artifact) == []

    bad_registry_root = tmp_path / "bad_registry"
    _write_json(
        bad_registry_root / mod.UPSTREAM_SOURCES["D5_MOAT_GATE"].relative_path,
        _d5(),
    )
    _write_json(
        bad_registry_root / mod.UPSTREAM_SOURCES["B1_GENUINE_SC"].relative_path,
        _artifacts()["B1_GENUINE_SC"],
    )
    bad_registry = bad_registry_root / mod.REGISTRY_RELATIVE_PATH
    bad_registry.parent.mkdir(parents=True, exist_ok=True)
    bad_registry.write_text("schema_version: [unterminated\n", encoding="utf-8")
    bad_spec = bad_registry_root / mod.SPEC_RELATIVE_PATH
    bad_spec.parent.mkdir(parents=True, exist_ok=True)
    bad_spec.write_text("REQ-CAPSTONE-5027\n", encoding="utf-8")

    bad = mod.run_capstone(root=bad_registry_root)
    assert bad["preconditions_checked"]["registry"]["yaml_loadable"] is False
    assert bad["reproducible_total_levels"] == 0
    assert bad["capstone_ready"] is True


def test_scenario_capstone_5027_validation_guards_schema_drift() -> None:
    """SCENARIO-CAPSTONE-5027-FIELD-PRINCIPLES: validation guards schema drift."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        duration_s=0.001,
    )

    errors = set(
        mod.validate_artifact(
            {
                **artifact,
                "honest_verdict": "maybe",
                "inference_substrate": "live_llm_inference",
                "moat_verdict": [],
                "best_arm_and_delta": [],
                "efficiency_win": "false",
                "diffusiongemma_gate_status": [],
                "per_arm_table": {},
                "infra_rollup": [],
                "reproducible_total_levels": "69",
                "next_milestone_pointer": [],
                "preconditions_checked": [],
                "hardware_rollup": [],
                "sota_ingestion_rollup": [],
                "self_play_rollup": [],
                "arc_opportunistic_rollup": [],
                "arc_deliverable_locked": [],
                "random_seed": True,
                "flagged_artifacts_skipped": [{"experiment_id": 5017}],
                "cited_upstream_artifacts": [{"experiment_id": 5022}],
                "field_principles": {},
                "capstone_ready": "true",
                "reproducibility_checksum": "bad",
            }
        )
    )
    assert {
        "honest_verdict_missing_terminal_prefix",
        "invalid_inference_substrate",
        "invalid_moat_verdict",
        "invalid_best_arm_and_delta",
        "invalid_efficiency_win",
        "invalid_diffusiongemma_gate_status",
        "invalid_per_arm_table",
        "invalid_infra_rollup",
        "invalid_reproducible_total_levels",
        "invalid_next_milestone_pointer",
        "invalid_preconditions_checked",
        "invalid_hardware_rollup",
        "invalid_sota_ingestion_rollup",
        "invalid_self_play_rollup",
        "invalid_arc_opportunistic_rollup",
        "invalid_arc_deliverable_locked",
        "invalid_random_seed",
        "invalid_flagged_artifacts_skipped",
        "invalid_cited_upstream_artifacts",
        "missing_principle:honest_verdict",
        "invalid_capstone_ready",
        "invalid_reproducibility_checksum",
    }.issubset(errors)
    assert "missing_field:moat_verdict" in mod.validate_artifact(
        {key: value for key, value in artifact.items() if key != "moat_verdict"}
    )
    assert mod._source_for_experiment(9999) is None
    assert mod._best_arm({}, []) == {}
    assert mod._delta_slug(None) == "unknown"
    d5_only_skipped = {
        "D5_MOAT_GATE": {
            **_d5(),
            "flagged_arms_skipped": [
                [],
                {"arm": "LoRA-EBM", "arm_id": "D1", "experiment_id": 5017},
                {"arm": "unknown", "arm_id": "DX", "experiment_id": 9999},
            ],
        }
    }
    assert mod._flagged_artifacts_skipped(d5_only_skipped, _hashes()) == [
        {
            "source": "D1_LORA_EBM",
            "experiment_id": 5017,
            "path": "results/experiment_5017_lora_ebm_scorer_musr_v2.json",
            "reason": "flagged_adversarial",
            "sha256": "sha256:d1_lora_ebm",
            "honest_verdict": "",
        }
    ]
