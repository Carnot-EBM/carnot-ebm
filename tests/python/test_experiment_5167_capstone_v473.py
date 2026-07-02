"""Tests for Exp 5167 V473 capstone reconciliation.

Spec refs: REQ-CAPSTONE-5167, SCENARIO-CAPSTONE-5167,
SCENARIO-CAPSTONE-5167-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5167_capstone_v473 as mod
from scripts import experiment_5167_capstone_v473 as script_mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _wrap(value: Any, principle: str = "fixture principle") -> dict[str, Any]:
    return {"principle": principle, "value": value}


def _base(experiment: str, verdict: Any, *, flagged: bool | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": experiment,
        "honest_verdict": verdict,
        "duration_s": 0.1,
        "inference_substrate": "fixture",
    }
    if flagged is not None:
        payload["flagged_adversarial"] = flagged
    return payload


def _payloads() -> dict[int, dict[str, Any]]:
    return {
        5157: {
            **_base(
                "experiment_5157_deepen_warmstart_replay_ablation_v473",
                "complete: warmstart_replay_ablation_gate_failed_honest_null_delta_0.0",
            ),
            "gate_passed": False,
            "warmstart_vs_cold_delta_median": 0.0,
            "actions_saved_pct_median": 0.0,
            "offline_reproduced": False,
        },
        5158: {
            **_base(
                "experiment_5158_deepen_goal_energy_ranker_replay_v473",
                "complete: goal_energy_ranker_warmstart_gate_failed_improved_1_of_3",
            ),
            "gate_passed": False,
            "games_improved_count": 1,
            "games_tested": [{"game": "lp85"}, {"game": "sc25"}, {"game": "tr87"}],
            "no_level_regression": True,
        },
        5159: {
            "experiment": 5159,
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: "
                "exp5157-deepen-warmstart-replay-ablation-v473.gate_passed"
            ),
        },
        5160: {
            **_base(
                "experiment_5160_oracle_distinct_cross_corpus_closure_v473",
                "success_arc_set_encoder_win_survives_cross_corpus_replication: "
                "set-encoder-vs-vote win survives corrected cross-corpus replication",
            ),
            "acceptance_gate": True,
            "cross_corpus_replication_passed": True,
            "cross_corpus_delta": 0.5,
            "cross_corpus_delta_ci95": [0.5, 0.5],
            "diffusiongemma_gate_updated_recommendation": "ungate_now",
            "headline_outcome": "arc_set_encoder_win_survives_cross_corpus_replication",
        },
        5161: {
            **_base(
                "experiment_5161_gap4_protocol_execution_pilot",
                "complete_gap4_pilot_n60_direction_replicated_not_significant_scale_up_recommended",
                flagged=True,
            ),
            "pilot_n_achieved": _wrap(60),
            "pilot_n_target": _wrap(60),
            "gap4_status_recommendation": _wrap("scale_up_recommended"),
            "exact_test_passes_min6_rule": _wrap(False),
            "exact_test_discordant_wins": _wrap(4),
            "exact_test_discordant_losses": 0,
        },
        5162: {
            **_base(
                "experiment_5162_sota_ingestion_multilevel_v473",
                "complete: zero new post-2026-07-02 primary findings; "
                "outcome-conditioned V474 references appended",
            ),
            "incremental_findings": _wrap([]),
            "bottom_line_recommendation": _wrap("do not scale the simple warm-start as-is"),
        },
        5163: {
            **_base(
                "experiment_5163_mmlu_pro_verifier_rescale_v473",
                _wrap(
                    "complete_mmlu_pro_fewshot_verifier_vs_cheap_delta_+0.025_"
                    "CI95_[-0.125,0.175]_CI_includes_0"
                ),
            ),
            "verifier_selection_accuracy": 0.175,
            "cheap_baseline_selection_accuracy": 0.15,
            "verifier_vs_cheap_delta": _wrap(0.025),
        },
        5164: {
            **_base(
                "experiment_5164_retro_timing_falsezero_fix_v473",
                "complete: module correctly reconstructs timing without modifying "
                "scripts/research_conductor.py",
            ),
            "research_conductor_py_modified": False,
            "tests_passing": True,
        },
        5165: {
            **_base(
                "experiment_5165_generation_axis_retirement_hygiene_v473",
                "complete: generation_axis_exploration_signal_scope_retired_and_lint_load_bearing",
                flagged=False,
            ),
            "known_issues_or_gaps_md_updated": True,
            "synthetic_match_check_passed": True,
        },
        5166: {
            **_base(
                "experiment_5166_hardware_continuity_board_timing",
                "complete_hardware_continuity_board_timing_gatemate:blocked_gatemate_dirtyjtag_"
                "idcode_no_speedup_claim",
            ),
            "boards_reachable_count": 2,
            "hardware_speedup_claimed": False,
            "no_speedup_claim": True,
        },
    }


def _make_repo(root: Path, *, omit: set[int] | None = None) -> None:
    omit = omit or set()
    payloads = _payloads()
    for source in mod.UPSTREAM_SOURCES:
        if source.experiment_number not in omit:
            _write_json(root / source.relative_path, payloads[source.experiment_number])
    _write_json(
        root / "ops" / "arc_solve_registry.yaml",
        {"reproducible_total_levels": 69, "reproducible_total_games": 24},
    )


def _reporter(path: Path) -> dict[str, Any]:
    critical = "5161" in path.name
    return {
        "artifact": str(path),
        "loaded": True,
        "flag_count": 3 if critical else 0,
        "max_severity": 2 if critical else -1,
        "flags": [
            {"kind": "DURATION_TOO_SHORT", "severity": "critical", "detail": "fixture"}
        ]
        if critical
        else [],
    }


def test_req_capstone_5167_spec_declares_v473_reconciliation_contract() -> None:
    """REQ-CAPSTONE-5167: OpenSpec declares the V473 capstone fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5167") :]

    for marker in (
        "REQ-CAPSTONE-5167",
        "SCENARIO-CAPSTONE-5167",
        "SCENARIO-CAPSTONE-5167-FIELD-PRINCIPLES",
        mod.EXPERIMENT_ID,
        str(mod.RESULT_RELATIVE_PATH),
        "levelup_guarantee_outcome_satisfied",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_capstone_5167_reconciles_v473_without_laundering_flagged(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5167: flagged upstreams are excluded from headline aggregation."""

    _make_repo(tmp_path)
    artifact = mod.build_artifact(
        root=tmp_path,
        run_date="20260702",
        duration_s=1.25,
        tests_run=["focused"],
        adversarial_reporter=_reporter,
        levelup_lint_result=mod.LevelupLintResult(
            exit_code=0,
            stdout="level-up attempts: 1\nOK: 1 >= 1",
            structurally_satisfied=True,
        ),
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["per_task_summary"]["principle"] == mod.FIELD_PRINCIPLES["per_task_summary"]
    assert len(artifact["per_task_summary"]["value"]) == 10
    assert artifact["flagged_adversarial_artifacts_excluded"]["value"] == [
        "exp5161-gap4-protocol-execution-pilot-v473"
    ]
    assert artifact["headline_eligible_task_ids"] == [
        source.task_id for source in mod.UPSTREAM_SOURCES if source.experiment_number != 5161
    ]
    assert artifact["gap4_status_reconciled"]["value"] == (
        "scale_up_recommended_not_filled_flagged_excluded_from_headline"
    )
    assert artifact["diffusiongemma_gate_reconciled"]["value"] == (
        "ungate_now_cross_corpus_replication_passed_no_scaling_run"
    )
    assert artifact["reproducible_total_levels_delta"]["value"] == 0
    assert artifact["registry_reconciliation"]["reproducible_total_levels"] == 69
    assert artifact["registry_reconciliation"]["reproducible_total_games"] == 24
    assert artifact["levelup_guarantee_structurally_satisfied"]["value"] is True
    assert artifact["levelup_guarantee_outcome_satisfied"]["value"] is False
    assert artifact["research_conductor_py_untouched_confirmed"]["value"] is True
    assert artifact["flagged_adversarial"] is False

    per_task = {row["task_id"]: row for row in artifact["per_task_summary"]["value"]}
    assert per_task["exp5159-deepen-live-levelup-attempt-v473"]["headline_outcome"] == (
        "gate_blocked_no_level_banked"
    )
    assert per_task["exp5161-gap4-protocol-execution-pilot-v473"]["headline_outcome"] == (
        "excluded_from_headline_aggregation_flagged_adversarial"
    )
    assert per_task["exp5163-mmlu-pro-verifier-rescale-v473"]["headline_outcome"] == (
        "mmlu_pro_verifier_delta_0.025_ci_includes_0"
    )


def test_req_capstone_5167_validation_and_run_edges(tmp_path: Path) -> None:
    """REQ-CAPSTONE-5167: schema validation rejects overclaims and stale checksums."""

    _make_repo(tmp_path, omit={5166})
    artifact = mod.build_artifact(
        root=tmp_path,
        run_date="20260702",
        duration_s=2.0,
        tests_run=["focused"],
        adversarial_reporter=_reporter,
        levelup_lint_result=mod.LevelupLintResult(
            exit_code=1,
            stdout="level-up attempts: 0\nFAIL",
            structurally_satisfied=False,
        ),
    )

    assert artifact["missing_artifacts"] == ["exp5166-hardware-continuity-board-timing-v473"]
    assert artifact["levelup_guarantee_structurally_satisfied"]["value"] is False
    assert artifact["levelup_guarantee_outcome_satisfied"]["value"] is False
    mod.validate_artifact(artifact)

    with pytest.raises(ValueError, match="missing required artifact fields"):
        mod.validate_artifact(
            {key: value for key, value in artifact.items() if key != "duration_s"}
        )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "done"})
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(artifact | {"inference_substrate": "live_llm_inference"})
    with pytest.raises(ValueError, match="field principle mismatch"):
        mod.validate_artifact(
            artifact
            | {"field_principles": artifact["field_principles"] | {"honest_verdict": "loose"}}
        )
    with pytest.raises(ValueError, match="flagged_adversarial"):
        mod.validate_artifact(artifact | {"flagged_adversarial": True})
    with pytest.raises(ValueError, match="research_conductor"):
        mod.validate_artifact(
            artifact | {"research_conductor_py_untouched_confirmed": _wrap(False)}
        )
    with pytest.raises(ValueError, match="levelup_guarantee_outcome_satisfied"):
        mod.validate_artifact(artifact | {"levelup_guarantee_outcome_satisfied": _wrap(True)})
    with pytest.raises(ValueError, match="tests_run"):
        mod.validate_artifact(artifact | {"tests_run": []})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(artifact | {"reproducibility_checksum": "stale"})

    assert mod.value_of(_wrap("x")) == "x"
    assert mod.value_of("x") == "x"
    assert mod.value_of({"principle": "missing"}) == {"principle": "missing"}
    assert mod.honest_verdict_text({"value": "complete_wrapped"}) == "complete_wrapped"
    assert mod.honest_verdict_text(None) == ""
    assert mod._number("3.5") == 3.5
    assert mod._number("not-a-number") is None
    assert mod._number(None) is None
    assert mod._number(True) is None

    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert mod.read_json_mapping(malformed)[1]["error"] == "malformed_json"
    not_mapping = tmp_path / "array.json"
    not_mapping.write_text("[]", encoding="utf-8")
    assert mod.read_json_mapping(not_mapping)[1]["error"] == "not_json_object"

    registry_path = tmp_path / "ops" / "arc_solve_registry.yaml"
    registry_path.unlink()
    assert mod.read_registry_totals(tmp_path)["loadable"] is False
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("- not-a-mapping\n", encoding="utf-8")
    assert mod.read_registry_totals(tmp_path)["reproducible_total_levels"] is None

    unknown_source = mod.UpstreamSource(9999, "exp9999-fixture", Path("results/x.json"))
    assert mod.headline_outcome(unknown_source, {"honest_verdict": "complete_fixture"}, []) == (
        "reconciled"
    )
    assert mod.gap4_status({}, []) == "still_open_missing_exp5161"
    assert mod.gap4_status({5161: {"gap4_status_recommendation": _wrap("filled")}}, []) == "filled"
    assert mod.gap4_status({5161: {"gap4_status_recommendation": _wrap("surprising")}}, []) == (
        "still_open"
    )
    assert mod.diffusiongemma_gate({}) == "still_gated_missing_exp5160"
    assert mod.diffusiongemma_gate({5160: {"diffusiongemma_gate_updated_recommendation": "keep_gated"}}) == (
        "keep_gated_cross_corpus_not_decision_grade"
    )

    out_path = mod.run(
        root=tmp_path,
        run_date="20260702",
        duration_s=2.5,
        tests_run=["run"],
        adversarial_reporter=_reporter,
        levelup_lint_result=mod.LevelupLintResult(
            exit_code=0,
            stdout="level-up attempts: 1\nOK: 1 >= 1",
            structurally_satisfied=True,
        ),
    )
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["duration_s"] == 2.5
    assert saved["tests_run"] == ["run"]
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)

    assert (
        script_mod.main(
            root=tmp_path,
            date="20260702",
            duration_s=3.0,
            tests_run=["script"],
            adversarial_reporter=_reporter,
            levelup_lint_result=mod.LevelupLintResult(
                exit_code=0,
                stdout="level-up attempts: 1\nOK: 1 >= 1",
                structurally_satisfied=True,
            ),
        )
        == out_path
    )
    assert (
        script_mod.main(
            ["--root", str(tmp_path), "--date", "20260702"],
            duration_s=3.5,
            tests_run=["script-argv"],
            adversarial_reporter=_reporter,
            levelup_lint_result=mod.LevelupLintResult(
                exit_code=0,
                stdout="level-up attempts: 1\nOK: 1 >= 1",
                structurally_satisfied=True,
            ),
        )
        == out_path
    )
