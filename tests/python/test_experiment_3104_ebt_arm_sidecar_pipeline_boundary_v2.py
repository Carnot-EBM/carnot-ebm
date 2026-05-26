"""Tests for Exp 3104 EBT/ARM sidecar pipeline boundary v2.

Spec refs: REQ-VERIFY-3091, SCENARIO-VERIFY-3091.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from carnot.inference.ebt_arm_sidecar_adapter import (
    REQUIRED_SIDECAR_FIELDS,
    REPLAY_INFERENCE_SUBSTRATE,
    SidecarReplayScorer,
    example_sidecar_records,
    load_sidecar_schema,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPO_ROOT / "results" / "experiment_3104_ebt_arm_sidecar_pipeline_boundary_v2.json"
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)


def test_req_verify_3091_pipeline_adapter_imports_and_schema_loads() -> None:
    """REQ-VERIFY-3091: the sidecar adapter and schema remain importable."""

    module = importlib.import_module("carnot.inference.ebt_arm_sidecar_adapter")
    schema = load_sidecar_schema(REPO_ROOT)

    assert module.SidecarReplayScorer is SidecarReplayScorer
    assert REQUIRED_SIDECAR_FIELDS <= set(schema["required"])
    assert REPLAY_INFERENCE_SUBSTRATE["live_model_inference"] is False
    assert REPLAY_INFERENCE_SUBSTRATE["model_weights_loaded"] is False


def test_scenario_verify_3091_pipeline_boundary_is_cached_replay_only() -> None:
    """SCENARIO-VERIFY-3091: replay scoring uses cached rows, not live weights."""

    records = example_sidecar_records()
    scorer = SidecarReplayScorer(schema=load_sidecar_schema(REPO_ROOT))
    scores = [scorer.score(record) for record in records]

    assert [score.total_energy for score in scores] == [0.08, 30.9]
    assert scores == [scorer.score(record) for record in records]
    assert all(score.inference_substrate["live_model_inference"] is False for score in scores)
    assert all(score.inference_substrate["model_weights_loaded"] is False for score in scores)
    assert all(score.inference_substrate["generation_performed"] is False for score in scores)


def test_req_verify_3091_exp3104_artifact_records_no_integration_boundary() -> None:
    """REQ-VERIFY-3091: Exp 3104 records the boundary without overclaiming."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    assert artifact["sidecar_boundary_v2_ready"] is True
    assert artifact["adapter_schema_ready"] is True
    assert artifact["pipeline_adapter_importable"] is True
    assert artifact["targeted_tests_passed"] is True
    assert artifact["targeted_test_commands"]
    assert artifact["no_weight_update_claim"] is True
    assert artifact["no_live_model_integration_claim"] is True
    assert "failed" in artifact["full_suite_status"]
    assert "not rerun" in artifact["full_suite_status"]
    assert artifact["remaining_integration_blockers"]
    assert artifact["honest_verdict"].startswith(SUCCESS_PREFIXES)

    substrate = artifact["inference_substrate"]
    assert substrate["kind"] == "offline_sidecar_tests"
    assert substrate["live_model_inference"] is False
    assert substrate["live_llm_inference"] is False
    assert substrate["model_weights_loaded"] is False
    assert substrate["generation_performed"] is False
    assert substrate["speedup_claimed"] is False

    source_paths = {
        source["path"] if isinstance(source, dict) else source
        for source in artifact["source_artifacts"]
    }
    assert (
        "results/experiment_3091_ebt_arm_sidecar_adapter_schema_prototype_v1.json" in source_paths
    )
    assert "results/experiment_3094_capstone_v288.json" in source_paths
