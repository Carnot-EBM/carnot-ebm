"""Focused tests for Exp4994 final held-out first-win readiness carry.

Spec refs: REQ-CAPSTONE-4994, SCENARIO-CAPSTONE-4994-CARRY-FINAL-FIRST-WIN,
SCENARIO-CAPSTONE-4994-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4994-FLAG-RESOLUTION.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4994_held_out_first_win_readiness as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _source(*, experiment_id: int = 4983) -> JsonDict:
    return {
        "duration_s": 673.601274,
        "experiment": f"experiment_{experiment_id}_heldout_first_win_readiness",
        "experiment_id": experiment_id,
        "first_win_baseline": 0.04,
        "first_win_delta_vs_baseline": 0.0,
        "first_win_rate_integrated": 0.04,
        "flag_resolved": True,
        "flagged_adversarial": False,
        "games_evaluated": 25,
        "games_remaining": 0,
        "generator_backend": "gpu0_cuda",
        "heldout_first_win_ci": {
            "bootstrap_resamples": 1000,
            "ci95": [0.0, 0.0],
            "method": "paired_percentile_bootstrap",
            "point": 0.0,
            "random_seed": 4605,
        },
        "heldout_first_win_ci_lower": 0.0,
        "heldout_first_win_delta_vs_baseline": 0.0,
        "heldout_first_win_rate": 0.04,
        "heldout_proxy_summary": {
            "heldout_first_win_rate": 0.04,
            "heldout_variant_attempts": 100,
            "source_artifact_path": "results/experiment_4605_live_integration_scored_agent.json",
        },
        "heldout_variant_attempt_floor": "B>=100",
        "heldout_variant_attempts": 100,
        "honest_verdict": "complete_heldout_first_win_0.04_full25_final_flag_resolved",
        "inference_substrate": "live_llm_inference",
        "model_specs": {
            "backend": "gpu0_cuda",
            "cuda_visible_devices": "0",
            "model_id": "unsloth/Qwen3.5-9B-MTP-GGUF",
            "name": "Qwen3.5-9B-MTP",
            "port": 8931,
            "server": "/home/ianblenke/.cache/llama.cpp-master/build/bin/llama-server",
            "serving_path": "GPU-0 CUDA llama-server",
        },
        "null_delta_methodology_note": (
            "The 0.04 agreement is a measured no-improvement null, not a TAUTOLOGY bug."
        ),
        "parity_test_green": True,
        "partial": False,
        "positive_control_passed": True,
        "preconditions_checked": {
            "ok": True,
            "generator_backend": "gpu0_cuda",
            "harness_present": True,
            "offline_arcade": True,
        },
        "random_seed": 4928,
        "solve_provenance": "development_proxy",
    }


def _write(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _warn_recheck(_path: Path) -> JsonDict:
    return {
        "flags": [
            {
                "kind": "TAUTOLOGY",
                "severity": "warn",
                "detail": "0.04==0.04 annotated null",
            }
        ],
        "summarize_exit_code": 1,
    }


def test_req_capstone_4994_spec_declares_final_carry_contract() -> None:
    """REQ-CAPSTONE-4994: OpenSpec declares the final A4 carry/readiness artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4994",
        "SCENARIO-CAPSTONE-4994-CARRY-FINAL-FIRST-WIN",
        "SCENARIO-CAPSTONE-4994-BLOCKED-PRECONDITION",
        "SCENARIO-CAPSTONE-4994-FLAG-RESOLUTION",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4994_carries_clean_exp4983_and_exp4972(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4994-CARRY-FINAL-FIRST-WIN: clean sources become final number."""

    primary_path = tmp_path / mod.PRIMARY_RESULT_RELATIVE_PATH
    secondary_path = tmp_path / mod.SECONDARY_RESULT_RELATIVE_PATH
    _write(primary_path, _source(experiment_id=4983))
    _write(secondary_path, _source(experiment_id=4972))

    artifact = mod.run(root=tmp_path, live_recheck=_warn_recheck)

    assert (
        artifact["honest_verdict"] == "complete_heldout_first_win_0.04_full25_final_flag_resolved"
    )
    assert artifact["heldout_first_win_rate"] == 0.04
    assert artifact["heldout_first_win_ci"]["ci95"] == [0.0, 0.0]
    assert artifact["games_evaluated"] == 25
    assert artifact["source_artifact"] == "exp4983/exp4972"
    assert artifact["source_artifacts"][0]["sha256"] == mod.file_sha256(primary_path)
    assert artifact["source_artifacts"][1]["sha256"] == mod.file_sha256(secondary_path)
    assert artifact["flag_resolved"] is True
    assert artifact["flagged_adversarial"] is False
    assert artifact["adversarial_verification"]["live_recheck"] == "warn"
    assert artifact["adversarial_verification"]["warn_count"] == 1
    assert artifact["positive_control_passed"] is True
    assert artifact["model_specs"]["name"] == "Qwen3.5-9B-MTP"
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["preconditions_checked"]["live_confirm_ran"] is False
    assert "TAUTOLOGY bug" in artifact["null_delta_methodology_note"]
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact


def test_scenario_capstone_4994_blocks_missing_or_unclean_sources(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4994-BLOCKED-PRECONDITION: missing source fabricates no rate."""

    missing = mod.run(root=tmp_path, live_recheck=_warn_recheck)

    assert missing["honest_verdict"] == "blocked_exp4983_artifact_missing"
    assert missing["heldout_first_win_rate"] is None
    assert missing["heldout_first_win_ci"] == {}
    assert missing["games_evaluated"] == 0
    assert missing["flag_resolved"] is False
    assert missing["preconditions_checked"]["blocked_resource"] == "exp4983_artifact_missing"
    assert missing["reproducibility_checksum"] == mod.payload_checksum(missing)
    assert mod.artifact_schema_errors(missing) == []

    primary_bad_root = tmp_path / "primary-bad"
    primary_bad = _source(experiment_id=4983)
    primary_bad["flag_resolved"] = False
    _write(primary_bad_root / mod.PRIMARY_RESULT_RELATIVE_PATH, primary_bad)

    primary_unclean = mod.run(root=primary_bad_root, live_recheck=_warn_recheck)

    assert primary_unclean["honest_verdict"] == "blocked_exp4983_flag_unresolved"
    assert primary_unclean["heldout_first_win_rate"] is None
    assert primary_unclean["preconditions_checked"]["blocked_resource"] == "exp4983_flag_unresolved"
    assert mod.artifact_schema_errors(primary_unclean) == []

    secondary_missing_root = tmp_path / "secondary-missing"
    _write(secondary_missing_root / mod.PRIMARY_RESULT_RELATIVE_PATH, _source(experiment_id=4983))

    secondary_missing = mod.run(root=secondary_missing_root, live_recheck=_warn_recheck)

    assert secondary_missing["honest_verdict"] == "blocked_exp4972_artifact_missing"
    assert secondary_missing["heldout_first_win_rate"] is None
    assert secondary_missing["preconditions_checked"]["blocked_resource"] == (
        "exp4972_artifact_missing"
    )
    assert mod.artifact_schema_errors(secondary_missing) == []

    _write(tmp_path / mod.PRIMARY_RESULT_RELATIVE_PATH, _source(experiment_id=4983))
    secondary = _source(experiment_id=4972)
    secondary["flag_resolved"] = False
    _write(tmp_path / mod.SECONDARY_RESULT_RELATIVE_PATH, secondary)

    unclean = mod.run(root=tmp_path, live_recheck=_warn_recheck)

    assert unclean["honest_verdict"] == "blocked_exp4972_flag_unresolved"
    assert unclean["heldout_first_win_rate"] is None
    assert unclean["preconditions_checked"]["blocked_resource"] == "exp4972_flag_unresolved"
    assert mod.artifact_schema_errors(unclean) == []


def test_scenario_capstone_4994_fallback_null_note_is_stamped(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4994-FLAG-RESOLUTION: unstated TAUTOLOGY text is restamped."""

    primary = _source(experiment_id=4983)
    primary["null_delta_methodology_note"] = "Measured no-improvement null."
    secondary = _source(experiment_id=4972)
    secondary["null_delta_methodology_note"] = "Independent confirm null."
    _write(tmp_path / mod.PRIMARY_RESULT_RELATIVE_PATH, primary)
    _write(tmp_path / mod.SECONDARY_RESULT_RELATIVE_PATH, secondary)

    artifact = mod.run(root=tmp_path, live_recheck=_warn_recheck)

    assert (
        artifact["honest_verdict"] == "complete_heldout_first_win_0.04_full25_final_flag_resolved"
    )
    assert "TAUTOLOGY bug" in artifact["null_delta_methodology_note"]
    assert "Primary note: Measured no-improvement null." in artifact["null_delta_methodology_note"]
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4994_blocks_critical_live_recheck(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4994-FLAG-RESOLUTION: critical live recheck blocks use."""

    _write(tmp_path / mod.PRIMARY_RESULT_RELATIVE_PATH, _source(experiment_id=4983))
    _write(tmp_path / mod.SECONDARY_RESULT_RELATIVE_PATH, _source(experiment_id=4972))

    artifact = mod.run(
        root=tmp_path,
        live_recheck=lambda _path: {
            "flags": [{"kind": "TRUE_LIVE_RECHECK", "severity": "critical", "detail": "bad"}],
            "summarize_exit_code": 2,
        },
    )

    assert artifact["honest_verdict"] == "blocked_exp4994_live_recheck_critical"
    assert artifact["flag_resolved"] is False
    assert artifact["flagged_adversarial"] is True
    assert artifact["heldout_first_win_rate"] is None
    assert artifact["triggering_rule_if_flagged"] == "TRUE_LIVE_RECHECK: bad"
    assert artifact["minimal_documented_source_fix_if_flagged"]
    assert artifact["preconditions_checked"]["critical_flags"] == [
        {"kind": "TRUE_LIVE_RECHECK", "severity": "critical", "detail": "bad"}
    ]
    assert mod.artifact_schema_errors(artifact) == []
