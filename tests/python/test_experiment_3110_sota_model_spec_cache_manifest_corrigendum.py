"""Tests for Exp 3110 SOTA model-spec/cache manifest corrigendum.

Spec refs: REQ-REPORT-3110, SCENARIO-REPORT-3110.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from carnot.reporting import sota_model_spec_cache_manifest_corrigendum_3110 as mod


REQUIRED_FIELDS = {
    "sota_model_manifest_ready",
    "mandatory_headline_model_ids",
    "present_model_ids",
    "missing_model_ids",
    "cached_sota_pair_available",
    "selected_headline_model_ids",
    "smoke_test_model_ids",
    "headline_claim_allowed",
    "downstream_usage",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path, text: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_spec(hf_id: str, *, cached: bool, selected: bool = False) -> dict[str, Any]:
    return {
        "hf_id": hf_id,
        "cache_present": cached,
        "cached": cached,
        "cache_status": "cached" if cached else "cache_missing",
        "model_path": f"/cache/{hf_id.replace('/', '--')}.gguf" if cached else None,
        "selected": selected,
    }


def _exp3099_payload(*, selected_middle: bool = True, cached_middle: bool = True) -> dict[str, Any]:
    return {
        "artifact": "experiment_3099_local_sota_confidence_abstention_panel_v3",
        "abstention_panel_v3_ready": True,
        "cached_sota_pair": {"called": True, "ready": False, "model_ids": [], "result": None},
        "model_specs": [
            _model_spec(mod.QWEN_MOE_ID, cached=False),
            _model_spec(mod.GEMMA_MIDDLE_MOE_ID, cached=cached_middle, selected=selected_middle),
            _model_spec(mod.GEMMA_DENSE_ID, cached=False),
        ],
        "models_used": [mod.GEMMA_MIDDLE_MOE_ID] if selected_middle else [],
        "inference_substrate": {
            "executes_models": True,
            "kind": "local_sota_gguf_llama_cpp_or_blocked_preflight",
            "legacy_tiny_models_promoted": False,
        },
        "honest_verdict": "complete: abstention_panel_v3_ready=true",
    }


def _exp3100_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3100_z3_oracle_feedback_v2",
        "cached_sota_pair_status": {"available": False, "models": [], "error": None},
        "formal_feedback_v2_ready": False,
        "headline_blocked_reason": "cached_sota_pair_unavailable",
        "inference_substrate": {
            "cached_sota_pair_available": False,
            "cached_sota_pair_models": [],
            "live_llm_inference": False,
            "model_cache_status": [
                {"hf_id": mod.QWEN_MOE_ID, "cached": False, "model_path": None},
                {
                    "hf_id": mod.GEMMA_MIDDLE_MOE_ID,
                    "cached": True,
                    "model_path": f"/cache/{mod.GEMMA_MIDDLE_MOE_ID.replace('/', '--')}.gguf",
                },
                {"hf_id": mod.GEMMA_DENSE_ID, "cached": False, "model_path": None},
            ],
        },
        "model_specs": [
            _model_spec(mod.QWEN_MOE_ID, cached=False),
            _model_spec(mod.GEMMA_MIDDLE_MOE_ID, cached=True),
            _model_spec(mod.GEMMA_DENSE_ID, cached=False),
        ],
        "solver_only_success_count": 2,
        "honest_verdict": "complete_blocked_headline: formal_feedback_v2_ready=false",
    }


def _matrix_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3107_cross_corpus_matrix_v23",
        "matrix_v23_ready": True,
        "headline_model_spec_gaps": [
            {
                "row_id": "dot289:exp3099_local_sota_confidence_abstention_panel",
                "source_artifact": mod.EXP3099_REL_PATH.as_posix(),
                "present_model_ids": list(mod.MANDATORY_HEADLINE_MODEL_IDS),
                "missing_model_ids": [],
                "reason": "mandatory_headline_model_ids missing for live LLM artifact",
            }
        ],
        "honest_verdict": "complete: matrix_v23_ready=true",
    }


def _capstone_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3108_capstone_v289",
        "capstone_ready": True,
        "paper_ready": False,
        "headline_model_spec_gaps": _matrix_payload()["headline_model_spec_gaps"],
        "honest_verdict": "complete: capstone_ready=true; paper_ready=false",
    }


def _write_common_sources(
    root: Path,
    *,
    selected_middle: bool = True,
    cached_middle: bool = True,
) -> None:
    _write_json(
        root,
        mod.EXP3099_REL_PATH,
        _exp3099_payload(selected_middle=selected_middle, cached_middle=cached_middle),
    )
    _write_json(root, mod.EXP3100_REL_PATH, _exp3100_payload())
    _write_json(root, mod.MATRIX_V23_REL_PATH, _matrix_payload())
    _write_json(root, mod.CAPSTONE_V289_REL_PATH, _capstone_payload())
    _write_text(
        root,
        mod.EXPERIMENT_TEMPLATE_REL_PATH,
        "cached_sota_pair() fallback: Qwen/Qwen3.5-0.8B google/gemma-4-E4B-it\n",
    )


def test_req_report_3110_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3110: OpenSpec declares the manifest contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3110" in spec
    assert "SCENARIO-REPORT-3110" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3110_builds_manifest_without_pair_overclaim(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3110: one cached mandated model clears metadata without pair claims."""

    _write_common_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=3.0, now_s=4.25)
    sources = {row["path"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["sota_model_manifest_ready"] is True
    assert artifact["mandatory_headline_model_ids"] == list(mod.MANDATORY_HEADLINE_MODEL_IDS)
    assert artifact["present_model_ids"] == [mod.GEMMA_MIDDLE_MOE_ID]
    assert artifact["missing_model_ids"] == [mod.QWEN_MOE_ID, mod.GEMMA_DENSE_ID]
    assert artifact["cached_sota_pair_available"] is False
    assert artifact["selected_headline_model_ids"] == [mod.GEMMA_MIDDLE_MOE_ID]
    assert artifact["smoke_test_model_ids"] == list(mod.LEGACY_SMOKE_TEST_MODEL_IDS)
    assert artifact["headline_claim_allowed"] is True
    assert artifact["duration_s"] == 1.25
    assert artifact["honest_verdict"].startswith("complete:")

    solver_rules = artifact["downstream_usage"]["solver_only_tasks"]
    live_rules = artifact["downstream_usage"]["live_llm_headline_tasks"]
    pair_rules = artifact["downstream_usage"]["pair_or_comparative_headline_tasks"]
    legacy_rules = artifact["downstream_usage"]["legacy_small_models"]
    assert solver_rules["allowed_without_cached_sota_pair"] is True
    assert solver_rules["headline_claim_allowed_from_solver_only"] is False
    assert live_rules["minimum_selected_mandated_cached_models"] == 1
    assert live_rules["allowed_model_ids"] == [mod.GEMMA_MIDDLE_MOE_ID]
    assert live_rules["headline_claim_allowed"] is True
    assert pair_rules["headline_claim_allowed"] is False
    assert legacy_rules["headline_claim_allowed"] is False

    assert artifact["inference_substrate"] == {
        "kind": "corrigendum_from_checked_in_artifacts",
        "source": "exp3099_exp3100_matrix_v23_capstone_v289",
        "executes_models": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
        "cache_probe_performed": False,
        "local_repo_only": True,
    }
    assert sources[mod.EXP3099_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.EXP3099_REL_PATH
    )
    assert artifact["matrix_reported_model_spec_gaps"][0]["reason"] == (
        "mandatory_headline_model_ids missing for live LLM artifact"
    )


def test_req_report_3110_blocks_headline_when_no_selected_cached_model(tmp_path: Path) -> None:
    """REQ-REPORT-3110: live headline claims fail closed without a selected cached model."""

    _write_common_sources(tmp_path, selected_middle=False, cached_middle=False)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["sota_model_manifest_ready"] is True
    assert artifact["present_model_ids"] == [mod.GEMMA_MIDDLE_MOE_ID]
    assert artifact["selected_headline_model_ids"] == []
    assert artifact["headline_claim_allowed"] is False
    assert (
        artifact["downstream_usage"]["live_llm_headline_tasks"]["headline_claim_allowed"] is False
    )
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_report_3110_missing_sources_and_write_helpers(tmp_path: Path) -> None:
    """REQ-REPORT-3110: missing authorities block readiness and helpers are deterministic."""

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)
    missing_paths = [row["path"] for row in artifact["missing_source_artifacts"]]

    assert artifact["sota_model_manifest_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_sota_model_manifest_preconditions")
    assert mod.EXP3099_REL_PATH.as_posix() in missing_paths

    malformed = tmp_path / "bad.json"
    list_json = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_json.write_text("[1]\n", encoding="utf-8")
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod._rows_from_model_specs(
        "edge",
        [1, {}, {"hf_id": "legacy/model", "cache_status": "cached"}],
    ) == [
        {
            "hf_id": "legacy/model",
            "cached": True,
            "selected": False,
            "model_path": None,
            "cache_status": "cached",
            "source_field": "edge",
        }
    ]

    _write_common_sources(tmp_path)
    output = mod.write_artifact(tmp_path, started_s=2.0, now_s=2.5)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["sota_model_manifest_ready"] is True
    assert saved["duration_s"] == 0.5
