"""Tests for Exp 1435 DPO headline provenance audit.

Spec: REQ-LEARN-1435, SCENARIO-LEARN-1435.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import dpo_headline_provenance_audit as mod


def _exp1420_artifact(
    *,
    supported: bool = False,
    reranker_used: bool = True,
    reason: str = "trl_not_available_for_dpo",
) -> dict[str, Any]:
    return {
        "status": "complete",
        "dpo_full_finetune_performed": supported,
        "dpo_reranker_fallback_used": reranker_used,
        "headline_result_allowed": False,
        "direct_dpo_feasibility": {
            "supported": supported,
            "reason": reason,
            "packages_checked": {"trl": supported},
        },
        "feasibility_note": (
            "Direct DPO fine-tuning was not performed because local GGUF files are "
            "llama.cpp inference artifacts, not trainable TRL/PEFT model directories."
        ),
        "gguf_model_checks": [
            {
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "cached": True,
                "llama_cpp_inference_performed": False,
            }
        ],
        "honest_verdict": "gguf_dpo_unsupported_reranker_fallback_measured",
    }


def test_req_learn_1435_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1435-1: bootstrap artifact is visible before evidence scanning."""

    output_path = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(
        output_path,
        project_root=tmp_path,
        run_date="20260506",
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    assert written["model_specs"] == list(mod.MODEL_SPECS)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(written)
    assert written["headline_provenance_ready"] is False
    assert written["honest_verdict"] == "in_progress"


def test_req_learn_1435_extracts_exp1420_gguf_blocker() -> None:
    """REQ-LEARN-1435-3: Exp 1420's unsupported GGUF DPO reason is explicit."""

    summary = mod.summarize_exp1420(_exp1420_artifact(reason="trl_not_available_for_dpo"))

    assert summary["direct_dpo_supported"] is False
    assert summary["full_finetune_performed"] is False
    assert summary["reranker_fallback_used"] is True
    assert summary["unsupported_reason"] == "trl_not_available_for_dpo"
    assert summary["headline_result_allowed"] is False
    assert "llama.cpp inference artifacts" in summary["feasibility_note"]

    malformed = mod.summarize_exp1420({"direct_dpo_feasibility": "not-a-dict"})
    assert malformed["unsupported_reason"] == "unknown"


def test_req_learn_1435_local_scan_requires_trainable_adapter_tooling() -> None:
    """REQ-LEARN-1435-4: reranker and GGUF inference code is not adapter support."""

    evidence = {
        "pyproject.toml": "dependencies = ['jax', 'numpy']",
        "python/carnot/reporting/dpo_verified_pairs_probe.py": (
            "import trl\nGGUF files are inference artifacts, not trainable TRL/PEFT "
            "model directories. train_reranker_fallback()"
        ),
        "python/carnot/inference/sota_models.py": "resolve_cached_gguf via llama.cpp path",
        "CLAUDE.md": "unsloth/Qwen3.6-35B-A3B-GGUF is a mandated SOTA GGUF",
    }

    scan = mod.scan_local_support(evidence)

    assert scan["direct_gguf_finetune_supported"] is False
    assert scan["local_adapter_path_supported"] is False
    assert "reranker_fallback" in scan["signals_found"]
    assert "mandated_sota_gguf_ids" in scan["signals_found"]
    assert "gguf_inference_only_blocker" in scan["blockers"]
    assert "missing_declared_trl_peft_dependencies" in scan["blockers"]


def test_req_learn_1435_supported_adapter_path_documents_next_experiment() -> None:
    """REQ-LEARN-1435-4: a complete TRL/PEFT path is treated as adapter-capable."""

    evidence = {
        "pyproject.toml": "dependencies = ['trl', 'peft', 'transformers']",
        "python/carnot/training/dpo_lora.py": (
            "from trl import DPOTrainer\n"
            "from peft import LoraConfig, PeftModel\n"
            "from transformers import AutoModelForCausalLM\n"
            "def train_mandated_sota_adapter(): pass\n"
            "convert_lora_to_gguf"
        ),
    }

    artifact = mod.build_artifact(
        exp1420_artifact=_exp1420_artifact(supported=False),
        evidence_texts=evidence,
        evidence_paths_checked=list(evidence),
        project_root="/repo",
        run_date="20260506",
    )

    assert artifact["direct_gguf_finetune_supported"] is False
    assert artifact["local_adapter_path_supported"] is True
    assert artifact["headline_provenance_ready"] is True
    assert artifact["reranker_track_relabelled"] is False
    assert "minimal_exp1436_or_followon" in artifact["recommended_next_training_path"]

    direct_path = mod._recommended_path(direct_gguf=True, local_adapter=False)
    assert "direct GGUF trainer" in direct_path


def test_scenario_learn_1435_unsupported_gguf_relabels_reranker_track(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1435: unsupported local DPO keeps the track reranker-only."""

    exp1420_path = tmp_path / "results" / "experiment_1420_dpo_verified_pairs_1508.json"
    exp1420_path.parent.mkdir()
    exp1420_path.write_text(json.dumps(_exp1420_artifact()), encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("dependencies = ['jax', 'numpy']", encoding="utf-8")
    dpo_module = tmp_path / "python" / "carnot" / "reporting" / "dpo_verified_pairs_probe.py"
    dpo_module.parent.mkdir(parents=True)
    dpo_module.write_text("train_reranker_fallback\nGGUF files are inference artifacts", encoding="utf-8")
    sota_module = tmp_path / "python" / "carnot" / "inference" / "sota_models.py"
    sota_module.parent.mkdir(parents=True)
    sota_module.write_text("resolve_cached_gguf\nllama.cpp GGUF path", encoding="utf-8")
    output_path = tmp_path / "results" / mod.OUTPUT_FILE

    artifact = mod.run(
        out_path=output_path,
        exp1420_path=exp1420_path,
        project_root=tmp_path,
        evidence_paths=["pyproject.toml", str(dpo_module.relative_to(tmp_path)), str(sota_module.relative_to(tmp_path))],
        run_date="20260506",
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["status"] == "complete"
    assert artifact["direct_gguf_finetune_supported"] is False
    assert artifact["local_adapter_path_supported"] is False
    assert artifact["headline_provenance_ready"] is False
    assert artifact["reranker_track_relabelled"] is True
    assert artifact["honest_verdict"] == mod.RERANKER_ONLY_VERDICT
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert str(exp1420_path) in artifact["evidence_paths_checked"]


def test_req_learn_1435_load_json_rejects_non_object(tmp_path: Path) -> None:
    """REQ-LEARN-1435-3: source artifacts must be JSON objects."""

    source = tmp_path / "bad.json"
    source.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="expected JSON object"):
        mod.load_json(source)
