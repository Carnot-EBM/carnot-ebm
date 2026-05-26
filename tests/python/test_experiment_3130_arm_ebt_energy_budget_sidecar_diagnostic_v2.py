"""Tests for Exp 3130 ARM/EBT energy-budget sidecar diagnostic v2.

Spec refs: REQ-VERIFY-3130, SCENARIO-VERIFY-3130.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import arm_ebt_energy_budget_sidecar_diagnostic_v2 as mod


GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"

REQUIRED_FIELDS = {
    "arm_ebt_energy_budget_sidecar_v2_ready",
    "model_specs",
    "selected_model_ids",
    "live_call_count",
    "exact_fixture_count",
    "deterministic_constraint_penalty_summary",
    "learned_quality_proxy_summary",
    "uncertainty_summary",
    "approximation_gap_summary",
    "model_identity_confound_audit",
    "correlation_metrics",
    "live_integration",
    "integration_blockers",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _base_sidecar_rows() -> list[dict[str, Any]]:
    return [
        {
            "fixture_id": "case-accept-arith",
            "task_family": "arithmetic_code_assertions",
            "exact_outcome": "accepted",
            "expected_action": "accept",
            "sidecar_action": "accept",
            "reject_or_repair_label": 0,
            "label_blind_feature_energy": 0.1,
            "replay_total_energy": 0.1,
            "feature_summary": {"label_blind_violation": 0.0, "surface_complexity": 0.5},
            "replay_score": {"confidence": 0.9},
        },
        {
            "fixture_id": "case-reject-arith",
            "task_family": "arithmetic_code_assertions",
            "exact_outcome": "rejected",
            "expected_action": "reject",
            "sidecar_action": "reject",
            "reject_or_repair_label": 1,
            "label_blind_feature_energy": 2.0,
            "replay_total_energy": 2.0,
            "feature_summary": {"label_blind_violation": 1.5, "surface_complexity": 0.5},
            "replay_score": {"confidence": 0.3},
        },
        {
            "fixture_id": "case-repair-json",
            "task_family": "repairable_invalid_candidates",
            "exact_outcome": "repairable",
            "expected_action": "reject",
            "sidecar_action": "reject",
            "reject_or_repair_label": 1,
            "label_blind_feature_energy": 1.4,
            "replay_total_energy": 1.4,
            "feature_summary": {"label_blind_violation": 1.0, "surface_complexity": 0.4},
            "replay_score": {"confidence": 0.45},
        },
    ]


def _write_sources(root: Path, *, include_exp3124: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("headline results need live provenance\n", encoding="utf-8")
    (root / "research-references.md").write_text("Distributional EBM uncertainty\n", encoding="utf-8")
    _write_json(
        root,
        mod.EXP3123_REL_PATH,
        {
            "artifact": "experiment_3123_sota_cache_preconditions_manifest_v2",
            "sota_cache_manifest_v2_ready": True,
            "mandatory_headline_model_ids": list(mod.MANDATORY_MODEL_IDS),
            "selected_headline_model_ids": [GEMMA26],
            "present_model_ids": [GEMMA26],
            "headline_claim_allowed": True,
            "gpu_preflight": {"cuda_available": True, "gpu_count": 2, "no_inference_run": True},
            "cache_inventory": [
                {
                    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "cache_status": "missing",
                    "role": "moe",
                    "path": None,
                },
                {
                    "hf_id": "unsloth/gemma-4-31B-it-GGUF",
                    "cache_status": "missing",
                    "role": "dense",
                    "path": None,
                },
                {
                    "hf_id": GEMMA26,
                    "cache_status": "resolved",
                    "role": "moe",
                    "path": "/tmp/gemma.gguf",
                },
            ],
        },
    )
    _write_json(
        root,
        mod.EXP3117_REL_PATH,
        {
            "artifact": "experiment_3117_ebt_arm_sidecar_score_correlation_boundary_v3",
            "sidecar_score_correlation_boundary_v3_ready": True,
            "exact_fixture_count": 3,
            "diagnostic_rows": _base_sidecar_rows(),
        },
    )
    if include_exp3124:
        _write_json(
            root,
            mod.EXP3124_REL_PATH,
            {
                "artifact": "experiment_3124_difficulty_stratified_live_sota_verifier_panel_v6",
                "difficulty_stratified_live_sota_panel_v6_ready": True,
                "selected_model_ids": [GEMMA26],
                "live_call_count": 2,
                "model_specs": [
                    {
                        "hf_id": GEMMA26,
                        "present": True,
                        "selected": True,
                        "cache_status": "resolved",
                        "role": "moe",
                    }
                ],
                "false_accept_rate": 0.5,
                "live_rows": [
                    {
                        "fixture_id": "case-accept-arith",
                        "model_id": GEMMA26,
                        "model_hash": "unit-hash",
                        "expected_action": "accept",
                        "live_decision": "accept",
                        "exact_answer_match": True,
                        "live_correct": True,
                        "fixture_family": "arithmetic_code_assertions",
                    },
                    {
                        "fixture_id": "case-reject-arith",
                        "model_id": GEMMA26,
                        "model_hash": "unit-hash",
                        "expected_action": "reject",
                        "live_decision": "accept",
                        "exact_answer_match": False,
                        "live_correct": False,
                        "fixture_family": "arithmetic_code_assertions",
                    },
                ],
            },
        )


def test_req_verify_3130_spec_anchor_exists() -> None:
    """REQ-VERIFY-3130: OpenSpec declares required artifact fields first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3130" in spec
    assert "SCENARIO-VERIFY-3130" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "deterministic_constraint_penalty_summary" in spec
    assert "live_integration=false" in spec


def test_scenario_verify_3130_builds_sidecar_diagnostic(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3130: energy budgets are diagnostic-only and auditable."""

    _write_sources(tmp_path)

    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=10.0,
        now_s=12.0,
        tests_run=["focused-unit"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["arm_ebt_energy_budget_sidecar_v2_ready"] is True
    assert artifact["selected_model_ids"] == [GEMMA26]
    assert artifact["live_call_count"] == 2
    assert artifact["exact_fixture_count"] == 3
    assert artifact["live_integration"] is False
    assert artifact["deterministic_constraint_penalty_summary"]["final_penalty"]["mean"] == pytest.approx(
        0.833333
    )
    assert artifact["learned_quality_proxy_summary"]["learned_model_score_available"] is False
    assert artifact["learned_quality_proxy_summary"]["quality_proxy"]["mean"] == pytest.approx(0.55)
    assert artifact["uncertainty_summary"]["uncertainty_proxy"]["mean"] == pytest.approx(0.45)
    assert artifact["approximation_gap_summary"]["accept_boundary_false_safe_count"] == 0
    assert artifact["model_identity_confound_audit"]["single_model_trace_only"] is True
    assert artifact["correlation_metrics"]["sidecar_energy"]["spearman_reject_or_repair"] > 0.0
    assert (
        artifact["correlation_metrics"]["quality_proxy"]["spearman_reject_or_repair"] < 0.0
    )
    assert artifact["correlation_metrics"]["by_fixture_family"][
        "arithmetic_code_assertions"
    ]["count"] == 2
    assert "exp3124 false_accept_rate=0.5" in artifact["integration_blockers"]
    assert artifact["inference_substrate"]["new_live_model_calls"] == 0
    assert artifact["inference_substrate"]["upstream_live_trace_count"] == 2
    assert artifact["duration_s"] == pytest.approx(2.0)
    assert artifact["tests_run"] == ["focused-unit"]
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)

    relative_output = mod.write_artifact(
        tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        started_s=20.0,
        now_s=21.0,
        tests_run=["relative-output"],
    )
    assert relative_output == tmp_path / mod.OUTPUT_REL_PATH


def test_req_verify_3130_blocks_without_trace_source(tmp_path: Path) -> None:
    """REQ-VERIFY-3130: missing mandated traces fail closed without live integration."""

    missing = mod.build_artifact(tmp_path, started_s=0.0, now_s=0.5, tests_run=["missing"])

    assert missing["arm_ebt_energy_budget_sidecar_v2_ready"] is False
    assert missing["live_call_count"] == 0
    assert missing["exact_fixture_count"] == 0
    assert missing["live_integration"] is False
    assert missing["honest_verdict"].startswith("blocked_")
    mod.validate_artifact(missing)

    invalid = tmp_path / mod.EXP3123_REL_PATH
    invalid.parent.mkdir(parents=True, exist_ok=True)
    invalid.write_text("not-json\n", encoding="utf-8")
    assert mod.read_json_object(invalid) == {}


def test_req_verify_3130_exp3124_optional_and_validation(tmp_path: Path) -> None:
    """REQ-VERIFY-3130: prior sidecar fixtures remain usable when Exp3124 is absent."""

    _write_sources(tmp_path, include_exp3124=False)
    artifact = mod.build_artifact(
        tmp_path,
        started_s=5.0,
        now_s=6.0,
        tests_run=["no-exp3124"],
    )

    assert artifact["arm_ebt_energy_budget_sidecar_v2_ready"] is True
    assert artifact["live_call_count"] == 0
    assert artifact["selected_model_ids"] == [GEMMA26]
    assert artifact["model_identity_confound_audit"]["live_trace_model_counts"] == {}
    assert artifact["inference_substrate"]["uses_exp3124_cached_live_traces"] is False
    assert artifact["source_artifacts"][5]["required"] is False
    mod.validate_artifact(artifact)

    incomplete_root = tmp_path / "incomplete"
    incomplete_root.mkdir()
    _write_json(
        incomplete_root,
        mod.EXP3117_REL_PATH,
        {
            "artifact": "experiment_3117_ebt_arm_sidecar_score_correlation_boundary_v3",
            "exact_fixture_count": 3,
            "diagnostic_rows": _base_sidecar_rows(),
        },
    )
    incomplete = mod.build_artifact(incomplete_root, started_s=1.0, now_s=2.0)
    assert incomplete["honest_verdict"].startswith("blocked_incomplete_diagnostic")

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="live_integration"):
        mod.validate_artifact(artifact | {"live_integration": True})
    with pytest.raises(ValueError, match="new_live_model_calls"):
        mod.validate_artifact(
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"new_live_model_calls": 1}
            }
        )
    with pytest.raises(ValueError, match="integration_blockers"):
        mod.validate_artifact(artifact | {"integration_blockers": []})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "ready"})

    assert mod.relative_path(tmp_path, tmp_path / "nested" / "artifact.json") == "nested/artifact.json"
    assert mod.relative_path(tmp_path, Path("/outside/artifact.json")) == "/outside/artifact.json"
    assert mod.numeric_summary([]) == {"count": 0, "finite": False}
    assert mod.scale01([2.0, 2.0]) == [0.0, 0.0]
    assert mod.confound_risk({"a": 1, "b": 1}, 2) == "lower_multiple_model_traces"

    legacy_summaries = {
        "model_identity_confound_audit": {"legacy_small_model_selected": True}
    }
    assert "legacy small model selected" in mod.integration_blockers(
        legacy_summaries,
        [],
        {},
        [],
    )
