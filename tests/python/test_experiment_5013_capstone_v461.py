"""Tests for REQ-CAPSTONE-5013 / SCENARIO-CAPSTONE-5013."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_5013_capstone_v461 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _d_arm(
    experiment_id: int,
    arm_id: str,
    *,
    flagged: bool = False,
    delta: float | None = 0.0,
    ci: list[float] | None = None,
) -> JsonDict:
    return {
        "experiment": f"experiment_{experiment_id}_{arm_id.lower()}",
        "honest_verdict": f"complete_{arm_id.lower()}",
        "flagged_adversarial": bool(flagged),
        "delta_vs_tuned_sc": delta,
        "paired_ci95": ci if ci is not None else [-0.03, 0.025],
        "mcnemar_p": 1.0,
        "headroom_present": delta is not None,
        "oracle_at_k": 0.93 if delta is not None else None,
        "tuned_sc_accuracy": 0.585 if delta is not None else None,
        "ebrm_selection_accuracy": 0.585 if delta is not None else None,
        "verifier_is_oracle": False,
    }


def _d5() -> JsonDict:
    row = {
        "arm": "EBRM",
        "arm_id": "D3",
        "corpus": "MuSR",
        "delta_vs_tuned_sc": 0.0,
        "headroom_present": True,
        "mcnemar_p": 1.0,
        "n_questions": 200,
        "oracle_at_k": 0.93,
        "paired_ci95": [-0.03, 0.025],
        "selection_accuracy": 0.585,
        "source_experiment_id": 5005,
        "tuned_sc_accuracy": 0.585,
        "verifier_is_oracle": False,
        "win_vs_tuned_sc": False,
    }
    return {
        "experiment": "experiment_5007_moat_gate_resolution",
        "honest_verdict": "complete_moat_scoped_no_realized_no_bounded_retirement",
        "decision": "MIXED-SCOPED",
        "moat_realized": False,
        "moat_retired_bounded": False,
        "best_arm": row,
        "per_arm_table": [row],
        "diffusiongemma_gate_status": "STILL-PENDING",
        "diffusiongemma_gate_conditions_satisfied_off_arc": False,
        "diffusiongemma_activation": "not_activated",
        "flagged_arms_skipped": [
            {"arm": "LoRA-EBM", "arm_id": "D1", "experiment_id": 5003},
            {"arm": "uPRM", "arm_id": "D2", "experiment_id": 5004},
            {"arm": "second-corpus-confirmation", "arm_id": "D4", "experiment_id": 5006},
        ],
        "paper_summary": "D5 is scoped rather than realized or retired.",
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _artifacts(*, include_d5: bool = True) -> dict[str, JsonDict]:
    artifacts: dict[str, JsonDict] = {
        "B1_HARNESS": {
            "experiment": "experiment_5002_moat_benchmark_harness",
            "experiment_id": 5002,
            "honest_verdict": "success_moat_harness_built_smoke_green",
            "oracle_distinctness_enforced": True,
            "headroom_present_smoke": True,
            "harness_module_path": "python/carnot/moat_benchmark_harness.py",
            "corpora_available": ["MuSR/murder_mysteries", "MMLU-Pro-hard"],
        },
        "D1_LORA_EBM": _d_arm(5003, "D1", flagged=True, delta=0.22, ci=[0.1, 0.3]),
        "D2_UPRM": _d_arm(5004, "D2", flagged=True, delta=None, ci=None),
        "D3_EBRM": _d_arm(5005, "D3"),
        "D4_SECOND_CORPUS": _d_arm(5006, "D4", flagged=True, delta=0.18, ci=[0.05, 0.31]),
        "B2_LINT": {
            "experiment": "experiment_5008_moat_oracle_distinct_lint",
            "honest_verdict": "success_moat_rigor_lint_shipped_fixtures_green.",
            "check_function_name": "check_moat_claim_rigor",
            "rules_implemented": [{"rule": "a"}, {"rule": "b"}],
            "fixtures_passed": {"clean": {"passed": True}},
        },
        "C_KV260": {
            "experiment": 5009,
            "honest_verdict": "success_kv260_reachable_overlay_loaded_energy_ok",
            "kv260_ssh_reachable": True,
            "loaded_overlay": "carnot_ising_v2_n64",
            "energy_smoke": {"success": True, "energy": -7, "expected_energy": -7},
            "xmutil_requires_sudo": True,
        },
        "E1_SOTA": {
            "honest_verdict": "success_sota_ingested_5_new_papers_mapped_to_phase_d",
            "new_arxiv_ids": [
                "2606.19818",
                "2606.09073",
                "2602.24040",
                "2510.20369",
                "2605.24005",
            ],
            "next_milestone_candidates": [{"candidate": "D3 uncertainty-aware EBRM rerun"}],
        },
        "E2_SELF_PLAY": {
            "experiment": "experiment_5011_self_play_verifier_checkpoint",
            "honest_verdict": "success_self_play_checkpoint_refreshed",
            "verifier_checkpoint_refreshed": True,
            "checkpoint_path": "models/arc_verifier_r11l.json",
            "target_game": "r11l",
            "offline_reproduced": True,
            "reproduced_levels": 2,
            "solve_provenance": "live_agent_self_discovery",
        },
        "E3_ARC_LEVEL": {
            "experiment": "experiment_5012_levelup_attempt",
            "experiment_id": 5012,
            "honest_verdict": "complete_cn04_no_new_level_residual_no_grounded_l4_delta",
            "target_game": "cn04",
            "new_levels_banked": 0,
            "offline_reproduced": False,
            "reproduced_levels": 3,
            "reproducible_total_levels_after": 69,
            "live_path_reachable": True,
        },
    }
    if include_d5:
        artifacts["D5_MOAT_GATE"] = _d5()
    return artifacts


def _hashes() -> dict[str, str]:
    return {source: f"sha256:{source.lower()}" for source in mod.UPSTREAM_SOURCES}


def test_req_capstone_5013_spec_declares_v461_scorecard_contract() -> None:
    """REQ-CAPSTONE-5013: OpenSpec declares the .461 scorecard contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5013") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH in section
    for field in mod.REQUIRED_FIELDS:
        assert field in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section


def test_scenario_capstone_5013_d5_authoritative_mixed_scorecard() -> None:
    """SCENARIO-CAPSTONE-5013: D5 drives the moat verdict and .462 pointer."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == (
        "complete_capstone_v461_moat_musr_scoped_ebrm_musr_delta_0p000"
    )
    assert artifact["capstone_ready"] is True
    assert artifact["moat_verdict"]["state"] == "mixed_musr_scoped"
    assert artifact["moat_verdict"]["source"] == "D5_MOAT_GATE"
    assert artifact["best_arm_and_delta"] == {
        "arm": "EBRM",
        "arm_id": "D3",
        "corpus": "MuSR",
        "delta_vs_tuned_sc": 0.0,
        "paired_ci95": [-0.03, 0.025],
        "headroom_present": True,
        "verifier_is_oracle": False,
        "win_vs_tuned_sc": False,
        "source_experiment_id": 5005,
    }
    assert artifact["diffusiongemma_gate_status"] == {
        "status": "STILL-PENDING",
        "conditions_satisfied_off_arc": False,
        "activation": "not_activated",
        "operator_gated": True,
        "autonomously_flipped_to_met": False,
    }
    assert artifact["per_arm_table"] == [_d5()["per_arm_table"][0]]
    assert artifact["infra_rollup"]["b1_moat_harness"]["built"] is True
    assert artifact["infra_rollup"]["b2_oracle_distinct_lint"]["shipped"] is True
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
    assert artifact["next_milestone_pointer"]["direction"] == "tighten_strongest_arm"
    assert artifact["next_milestone_pointer"]["milestone"] == "2026.06.462"
    assert {row["source"] for row in artifact["flagged_artifacts_skipped"]} == {
        "D1_LORA_EBM",
        "D2_UPRM",
        "D4_SECOND_CORPUS",
    }
    assert not {
        "D1_LORA_EBM",
        "D2_UPRM",
        "D4_SECOND_CORPUS",
    } & {row["source"] for row in artifact["cited_upstream_artifacts"]}
    assert {"D5_MOAT_GATE", "D3_EBRM", "B1_HARNESS", "B2_LINT"}.issubset(
        {row["source"] for row in artifact["cited_upstream_artifacts"]}
    )
    assert all(row["sha256"].startswith("sha256:") for row in artifact["cited_upstream_artifacts"])
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert mod.validate_artifact(artifact) == []


def test_req_capstone_5013_flagged_direct_arms_never_drive_fallback_headline() -> None:
    """REQ-CAPSTONE-5013: flagged D1-D4 arms are skipped before aggregation."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(include_d5=False),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        duration_s=0.001,
    )

    assert artifact["moat_verdict"]["d5_missing"] is True
    assert artifact["moat_verdict"]["state"] == "mixed_musr_scoped"
    assert artifact["best_arm_and_delta"]["arm_id"] == "D3"
    assert artifact["best_arm_and_delta"]["delta_vs_tuned_sc"] == 0.0
    assert all(row["arm_id"] == "D3" for row in artifact["per_arm_table"])
    assert {row["experiment_id"] for row in artifact["flagged_artifacts_skipped"]} == {
        5003,
        5004,
        5006,
    }
    assert all(row["source"] != "D1_LORA_EBM" for row in artifact["cited_upstream_artifacts"])
    assert "D5 missing" in artifact["moat_verdict"]["summary"]
    assert mod.validate_artifact(artifact) == []

    direct_realized = _artifacts(include_d5=False)
    direct_realized["D3_EBRM"] = _d_arm(5005, "D3", delta=0.12, ci=[0.02, 0.2])
    realized = mod.build_artifact(
        artifacts=direct_realized,
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        duration_s=0.001,
    )
    assert realized["moat_verdict"]["state"] == "moat_realized"
    assert realized["diffusiongemma_gate_status"]["conditions_satisfied_off_arc"] is True

    direct_retired = _artifacts(include_d5=False)
    direct_retired["D1_LORA_EBM"] = _d_arm(5003, "D1", flagged=False)
    direct_retired["D2_UPRM"] = _d_arm(5004, "D2", flagged=False)
    retired = mod.build_artifact(
        artifacts=direct_retired,
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        duration_s=0.001,
    )
    assert retired["moat_verdict"]["state"] == "moat_retired_bounded"
    assert retired["honest_verdict"] == (
        "complete_capstone_v461_moat_retired_bounded_lora_ebm_and_uprm_both_null"
    )


def test_scenario_capstone_5013_runtime_writes_result_with_absent_arms_recorded(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5013: run writes the deliverable and records absent inputs."""

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
    spec.write_text("REQ-CAPSTONE-5013\n", encoding="utf-8")

    artifact = mod.run_capstone(root=tmp_path)

    assert artifact["honest_verdict"] == (
        "complete_capstone_v461_moat_musr_scoped_ebrm_musr_delta_0p000"
    )
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
    _write_json(bad_registry_root / mod.UPSTREAM_SOURCES["D5_MOAT_GATE"].relative_path, _d5())
    _write_json(
        bad_registry_root / mod.UPSTREAM_SOURCES["B1_HARNESS"].relative_path,
        _artifacts()["B1_HARNESS"],
    )
    _write_json(
        bad_registry_root / mod.UPSTREAM_SOURCES["B2_LINT"].relative_path,
        _artifacts()["B2_LINT"],
    )
    bad_registry = bad_registry_root / mod.REGISTRY_RELATIVE_PATH
    bad_registry.parent.mkdir(parents=True, exist_ok=True)
    bad_registry.write_text("schema_version: [unterminated\n", encoding="utf-8")
    bad_spec = bad_registry_root / mod.SPEC_RELATIVE_PATH
    bad_spec.parent.mkdir(parents=True, exist_ok=True)
    bad_spec.write_text("REQ-CAPSTONE-5013\n", encoding="utf-8")

    bad = mod.run_capstone(root=bad_registry_root)
    assert bad["preconditions_checked"]["registry"]["yaml_loadable"] is False
    assert bad["reproducible_total_levels"] == 0


def test_scenario_capstone_5013_validation_and_pointer_branches() -> None:
    """SCENARIO-CAPSTONE-5013-FIELD-PRINCIPLES: validation guards schema drift."""

    realized_d5 = {
        **_d5(),
        "decision": "MOAT-REALIZED",
        "moat_realized": True,
        "moat_retired_bounded": False,
        "best_arm": {
            **_d5()["best_arm"],
            "delta_vs_tuned_sc": 0.12,
            "paired_ci95": [0.02, 0.2],
            "win_vs_tuned_sc": True,
        },
    }
    retired_d5 = {
        **_d5(),
        "decision": "MOAT-RETIRED-BOUNDED",
        "moat_realized": False,
        "moat_retired_bounded": True,
    }
    realized = mod.build_artifact(
        artifacts={**_artifacts(), "D5_MOAT_GATE": realized_d5},
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        duration_s=0.001,
    )
    retired = mod.build_artifact(
        artifacts={**_artifacts(), "D5_MOAT_GATE": retired_d5},
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        duration_s=0.001,
    )

    assert realized["moat_verdict"]["state"] == "moat_realized"
    assert realized["next_milestone_pointer"]["direction"] == "scale_winning_construction"
    assert retired["moat_verdict"]["state"] == "moat_retired_bounded"
    assert retired["next_milestone_pointer"]["direction"] == "pivot_to_e1_sota_candidates"

    errors = set(
        mod.validate_artifact(
            {
                **realized,
                "honest_verdict": "maybe",
                "inference_substrate": "live_llm_inference",
                "moat_verdict": [],
                "best_arm_and_delta": [],
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
                "flagged_artifacts_skipped": [{"experiment_id": 5003}],
                "cited_upstream_artifacts": [{"experiment_id": 5007}],
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
        {key: value for key, value in realized.items() if key != "moat_verdict"}
    )
    assert mod._source_for_experiment(9999) is None
    assert mod._best_arm({}, []) == {}
    assert mod._delta_slug(None) == "unknown"
    d5_only_skipped = {
        "D5_MOAT_GATE": {
            **_d5(),
            "flagged_arms_skipped": [
                [],
                {"arm": "LoRA-EBM", "arm_id": "D1", "experiment_id": 5003},
                {"arm": "unknown", "arm_id": "DX", "experiment_id": 9999},
            ],
        }
    }
    assert mod._flagged_artifacts_skipped(d5_only_skipped, _hashes()) == [
        {
            "source": "D1_LORA_EBM",
            "experiment_id": 5003,
            "path": "results/experiment_5003_lora_ebm_scorer_musr.json",
            "reason": "flagged_adversarial",
            "sha256": "sha256:d1_lora_ebm",
            "honest_verdict": "",
        }
    ]
