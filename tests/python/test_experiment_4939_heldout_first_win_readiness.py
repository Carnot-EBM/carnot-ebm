"""Focused tests for Exp4939 final held-out first-win carry.

Spec refs: REQ-CAPSTONE-4939, SCENARIO-CAPSTONE-4939,
SCENARIO-CAPSTONE-4939-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4939-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4939_held_out_first_win_readiness as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _source() -> JsonDict:
    return {
        "duration_s": 673.601274,
        "experiment": "experiment_4928_heldout_first_win_readiness",
        "experiment_id": 4928,
        "first_win_baseline": 0.04,
        "flag_resolved": True,
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
        "heldout_proxy_summary": {"heldout_variant_attempts": 100},
        "heldout_variant_attempt_floor": "B>=100",
        "heldout_variant_attempts": 100,
        "honest_verdict": "complete_heldout_first_win_0.04_full25_live_flag_resolved",
        "inference_substrate": "live_llm_inference",
        "model_specs": {
            "backend": "gpu0_cuda",
            "model_id": "unsloth/Qwen3.5-9B-MTP-GGUF",
            "name": "Qwen3.5-9B-MTP",
            "port": 8931,
            "serving_path": "GPU-0 CUDA llama-server",
        },
        "null_delta_methodology_note": (
            "Held-out first-win rate equals the 0.04 baseline with positive-control parity "
            "passed; this is a genuine no-improvement result."
        ),
        "parity_test_green": True,
        "partial": False,
        "positive_control_passed": True,
        "preconditions_checked": {
            "ok": True,
            "experiment_4917_21of25_ledger_present": True,
            "generator_backend": "gpu0_cuda",
            "offline_arcade": True,
        },
        "random_seed": 4928,
        "solve_provenance": "development_proxy",
    }


def test_req_capstone_4939_spec_declares_final_carry_contract() -> None:
    """REQ-CAPSTONE-4939: OpenSpec declares the final carry/readiness artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4939",
        "SCENARIO-CAPSTONE-4939",
        "SCENARIO-CAPSTONE-4939-BLOCKED-PRECONDITION",
        "SCENARIO-CAPSTONE-4939-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4939_carries_clean_exp4928_full25(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4939: clean Exp4928 is carried as the final go/no-go."""

    source_path = tmp_path / mod.SOURCE_RESULT_RELATIVE_PATH
    source_path.parent.mkdir(parents=True)
    source_path.write_text(json.dumps(_source(), indent=2), encoding="utf-8")

    artifact = mod.run(root=tmp_path, critical_flags=lambda _path: [])

    assert (
        artifact["honest_verdict"] == "complete_heldout_first_win_0.04_full25_final_flag_resolved"
    )
    assert artifact["heldout_first_win_rate"] == 0.04
    assert artifact["heldout_first_win_ci"]["ci95"] == [0.0, 0.0]
    assert artifact["games_evaluated"] == 25
    assert artifact["source_artifact"] == "exp4928"
    assert artifact["source_artifact_sha256"] == mod.file_sha256(source_path)
    assert artifact["flag_resolved"] is True
    assert artifact["triggering_rule_if_flagged"] == ""
    assert artifact["positive_control_passed"] is True
    assert artifact["model_specs"]["name"] == "Qwen3.5-9B-MTP"
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["solve_provenance"] == "development_proxy"
    assert "TAUTOLOGY bug" in artifact["null_delta_methodology_note"]
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact


def test_scenario_capstone_4939_blocks_missing_or_unclean_source(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4939-BLOCKED-PRECONDITION: no source means no fabricated rate."""

    missing = mod.run(root=tmp_path, critical_flags=lambda _path: [])

    assert missing["honest_verdict"] == "blocked_exp4928_artifact_missing"
    assert missing["heldout_first_win_rate"] is None
    assert missing["heldout_first_win_ci"] == {}
    assert missing["games_evaluated"] == 0
    assert missing["flag_resolved"] is False
    assert missing["preconditions_checked"]["blocked_resource"] == "exp4928_artifact_missing"
    assert missing["reproducibility_checksum"] == mod.payload_checksum(missing)
    assert mod.artifact_schema_errors(missing) == []

    source_path = tmp_path / mod.SOURCE_RESULT_RELATIVE_PATH
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source = _source()
    source["flag_resolved"] = False
    source_path.write_text(json.dumps(source, indent=2), encoding="utf-8")

    unclean = mod.run(root=tmp_path, critical_flags=lambda _path: [])

    assert unclean["honest_verdict"] == "blocked_exp4928_flag_unresolved"
    assert unclean["preconditions_checked"]["blocked_resource"] == "exp4928_flag_unresolved"
    assert unclean["heldout_first_win_rate"] is None
    assert unclean["source_artifact_sha256"] == mod.file_sha256(source_path)
    assert mod.artifact_schema_errors(unclean) == []


def test_scenario_capstone_4939_blocks_each_unclean_source_contract(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4939-BLOCKED-PRECONDITION: every source gate fails closed."""

    source_path = tmp_path / mod.SOURCE_RESULT_RELATIVE_PATH
    source_path.parent.mkdir(parents=True, exist_ok=True)

    cases = (
        ("games_evaluated", 24, "exp4928_not_full25"),
        ("heldout_first_win_rate", 0.08, "exp4928_rate_not_final_0_04"),
        ("positive_control_passed", False, "exp4928_positive_control_missing"),
        ("null_delta_methodology_note", "", "exp4928_methodology_stamp_missing"),
    )
    for field, value, reason in cases:
        source = _source()
        source[field] = value
        source_path.write_text(json.dumps(source, indent=2), encoding="utf-8")

        artifact = mod.run(root=tmp_path, critical_flags=lambda _path: [])

        assert artifact["honest_verdict"] == f"blocked_{reason}"
        assert artifact["preconditions_checked"]["blocked_resource"] == reason
        assert artifact["heldout_first_win_rate"] is None
        assert mod.artifact_schema_errors(artifact) == []

    source = _source()
    source["null_delta_methodology_note"] = "Already marked as not a TAUTOLOGY bug."
    source_path.write_text(json.dumps(source, indent=2), encoding="utf-8")

    clean = mod.run(root=tmp_path, critical_flags=lambda _path: [])

    assert clean["null_delta_methodology_note"] == source["null_delta_methodology_note"]
    assert mod.artifact_schema_errors(clean) == []


def test_scenario_capstone_4939_blocks_critical_live_recheck(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4939: critical adversarial recheck blocks the final carry."""

    source_path = tmp_path / mod.SOURCE_RESULT_RELATIVE_PATH
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(json.dumps(_source(), indent=2), encoding="utf-8")

    artifact = mod.run(
        root=tmp_path,
        critical_flags=lambda _path: [{"kind": "TRUE_LIVE_RECHECK", "detail": "critical"}],
    )

    assert artifact["honest_verdict"] == "blocked_exp4939_live_recheck_critical"
    assert artifact["triggering_rule_if_flagged"] == "TRUE_LIVE_RECHECK: critical"
    assert artifact["heldout_first_win_rate"] is None
    assert artifact["preconditions_checked"]["critical_flags"] == [
        {"kind": "TRUE_LIVE_RECHECK", "detail": "critical"}
    ]
    assert mod.artifact_schema_errors(artifact) == []
