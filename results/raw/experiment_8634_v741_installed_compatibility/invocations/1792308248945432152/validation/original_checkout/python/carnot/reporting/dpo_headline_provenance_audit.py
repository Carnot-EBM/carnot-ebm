"""Exp 1435 DPO headline provenance audit.

This module does not train a model.  It reads Exp 1420's artifact and the local
supporting code/docs so Carnot can separate a useful preference-reranking signal
from a headline claim that would require an actual local SOTA GGUF adapter or
fine-tune path.

Spec: REQ-LEARN-1435, SCENARIO-LEARN-1435.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILE = "experiment_1435_dpo_headline_provenance_audit.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / OUTPUT_FILE
DEFAULT_EXP1420_PATH = REPO_ROOT / "results" / "experiment_1420_dpo_verified_pairs_1508.json"
RUN_DATE = "20260506"
SCHEMA = "dpo_headline_provenance_audit_v1"
RERANKER_ONLY_VERDICT = "dpo_headline_not_ready_reranker_only_until_adapter_or_conversion_tooling"
ADAPTER_READY_VERDICT = "local_adapter_path_supported_next_experiment_required_for_headline_dpo"

MODEL_SPECS: tuple[dict[str, str], ...] = (
    {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "primary_training_target_if_supported",
    },
    {
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "dense_training_target_if_supported",
    },
    {
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "moe_training_target_if_supported",
    },
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "model_specs",
    "direct_gguf_finetune_supported",
    "local_adapter_path_supported",
    "headline_provenance_ready",
    "reranker_track_relabelled",
    "recommended_next_training_path",
    "evidence_paths_checked",
    "honest_verdict",
)

DEFAULT_EVIDENCE_PATHS: tuple[str, ...] = (
    "python/carnot/reporting/dpo_verified_pairs_probe.py",
    "python/carnot/inference/sota_models.py",
    "python/carnot/training/grpo_v5.py",
    "python/carnot/training/grpo_v5_2.py",
    "python/carnot/training/grpo_vps_training.py",
    "python/carnot/training/sdpo_dense_reward.py",
    "scripts/experiment_template.py",
    "pyproject.toml",
    "CLAUDE.md",
    "research-references.md",
    "ops/status.md",
)


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path | str, artifact: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload


def load_json(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path!s}")
    return payload


def write_in_progress_artifact(
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1435-1: write the visible bootstrap artifact first."""

    return _write_json(
        out_path,
        {
            "schema": SCHEMA,
            "spec": ["REQ-LEARN-1435", "SCENARIO-LEARN-1435"],
            "artifact_metadata": {"project_root": str(project_root), "run_date": run_date},
            "run_date": run_date,
            "started_at": _timestamp(),
            "status": "in_progress",
            "model_specs": list(MODEL_SPECS),
            "direct_gguf_finetune_supported": None,
            "local_adapter_path_supported": None,
            "headline_provenance_ready": False,
            "reranker_track_relabelled": None,
            "recommended_next_training_path": None,
            "evidence_paths_checked": [],
            "honest_verdict": "in_progress",
        },
    )


def summarize_exp1420(exp1420_artifact: Mapping[str, Any]) -> dict[str, Any]:
    """REQ-LEARN-1435-3: extract the prior GGUF DPO blocker without relabeling it."""

    feasibility = exp1420_artifact.get("direct_dpo_feasibility") or {}
    if not isinstance(feasibility, Mapping):
        feasibility = {}
    return {
        "direct_dpo_supported": bool(feasibility.get("supported")),
        "unsupported_reason": str(feasibility.get("reason") or "unknown"),
        "packages_checked": dict(feasibility.get("packages_checked") or {}),
        "full_finetune_performed": bool(exp1420_artifact.get("dpo_full_finetune_performed")),
        "reranker_fallback_used": bool(exp1420_artifact.get("dpo_reranker_fallback_used")),
        "headline_result_allowed": bool(exp1420_artifact.get("headline_result_allowed")),
        "feasibility_note": str(exp1420_artifact.get("feasibility_note") or ""),
        "honest_verdict": str(exp1420_artifact.get("honest_verdict") or ""),
        "gguf_model_checks": list(exp1420_artifact.get("gguf_model_checks") or []),
    }


def _has_token(text: str, token: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", text) is not None


def scan_local_support(evidence_texts: Mapping[str, str]) -> dict[str, Any]:
    """REQ-LEARN-1435-4: conservatively classify the current local DPO support surface."""

    combined = "\n".join(evidence_texts.values()).lower()
    pyproject = "\n".join(
        text.lower() for path, text in evidence_texts.items() if path.endswith("pyproject.toml")
    )
    trl_declared = _has_token(pyproject, "trl")
    peft_declared = _has_token(pyproject, "peft")
    has_dpo_trainer = "dpotrainer" in combined
    has_lora = "loraconfig" in combined or _has_token(combined, "lora")
    has_hf_loader = "automodelforcausallm" in combined or _has_token(combined, "transformers")
    has_adapter_export = any(
        token in combined
        for token in (
            "convert_lora_to_gguf",
            "convert-hf-to-gguf",
            "merge_and_unload",
            "adapter_model.safetensors",
        )
    )
    direct_gguf = "direct_gguf_finetune_supported=true" in combined

    signals: list[str] = []
    blockers: list[str] = []
    if "train_reranker_fallback" in combined or "reranker fallback" in combined:
        signals.append("reranker_fallback")
    if "unsloth/qwen3.6-35b-a3b-gguf" in combined:
        signals.append("mandated_sota_gguf_ids")
    if "resolve_cached_gguf" in combined or "llama.cpp" in combined:
        signals.append("gguf_cache_resolution")
    if trl_declared:
        signals.append("trl_dependency_declared")
    if peft_declared:
        signals.append("peft_dependency_declared")
    if has_dpo_trainer:
        signals.append("dpo_trainer")
    if has_lora:
        signals.append("peft_lora_adapter")
    if has_hf_loader:
        signals.append("hf_trainable_model_loader")
    if has_adapter_export:
        signals.append("gguf_export_or_conversion")

    if "gguf files are inference artifacts" in combined or "not trainable trl/peft" in combined:
        blockers.append("gguf_inference_only_blocker")
    if not direct_gguf:
        blockers.append("direct_gguf_weight_update_not_supported")
    if not (trl_declared and peft_declared):
        blockers.append("missing_declared_trl_peft_dependencies")
    if not has_dpo_trainer:
        blockers.append("missing_dpo_trainer_implementation")
    if not has_lora:
        blockers.append("missing_peft_lora_adapter_implementation")
    if not has_adapter_export:
        blockers.append("missing_adapter_merge_or_gguf_conversion_step")

    local_adapter = all(
        [trl_declared, peft_declared, has_dpo_trainer, has_lora, has_hf_loader, has_adapter_export]
    )
    return {
        "direct_gguf_finetune_supported": direct_gguf,
        "local_adapter_path_supported": local_adapter,
        "signals_found": signals,
        "blockers": blockers,
    }


def _recommended_path(*, direct_gguf: bool, local_adapter: bool) -> str:
    if direct_gguf:
        return (
            "minimal_exp1436_or_followon: run a bounded DPO smoke update on one mandated "
            "SOTA GGUF target with the direct GGUF trainer, save the weight delta, then "
            "evaluate against Exp 1420 pairs before allowing headline wording."
        )
    if local_adapter:
        return (
            "minimal_exp1436_or_followon: run TRL/PEFT LoRA DPO on the trainable "
            "HF-format base corresponding to a mandated GGUF target, save the adapter, "
            "merge/export or convert it for local GGUF inference, then gate headline "
            "wording on live held-out evaluation."
        )
    return (
        "Keep Exp 1420 as a reranker-only benchmark. Add a supported HF-format "
        "base-model plus TRL/PEFT LoRA DPO path, and add adapter merge/export or "
        "GGUF conversion tooling, before making any headline DPO training claim."
    )


def build_artifact(
    *,
    exp1420_artifact: Mapping[str, Any],
    evidence_texts: Mapping[str, str],
    evidence_paths_checked: Sequence[str],
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1435-5: build the terminal provenance decision artifact."""

    exp1420_summary = summarize_exp1420(exp1420_artifact)
    local_scan = scan_local_support(evidence_texts)
    direct_gguf = bool(local_scan["direct_gguf_finetune_supported"])
    local_adapter = bool(local_scan["local_adapter_path_supported"])
    headline_ready = direct_gguf or local_adapter

    return {
        "schema": SCHEMA,
        "spec": ["REQ-LEARN-1435", "SCENARIO-LEARN-1435"],
        "artifact_metadata": {"project_root": str(project_root), "run_date": run_date},
        "run_date": run_date,
        "finished_at": _timestamp(),
        "status": "complete",
        "model_specs": list(MODEL_SPECS),
        "direct_gguf_finetune_supported": direct_gguf,
        "local_adapter_path_supported": local_adapter,
        "headline_provenance_ready": headline_ready,
        "reranker_track_relabelled": not headline_ready,
        "recommended_next_training_path": _recommended_path(
            direct_gguf=direct_gguf,
            local_adapter=local_adapter,
        ),
        "evidence_paths_checked": list(evidence_paths_checked),
        "honest_verdict": ADAPTER_READY_VERDICT if headline_ready else RERANKER_ONLY_VERDICT,
        "exp1420_blocker_summary": exp1420_summary,
        "local_support_scan": local_scan,
        "legacy_small_model_headline_disallowed": True,
    }


def read_evidence_texts(
    project_root: Path | str,
    evidence_paths: Sequence[str] = DEFAULT_EVIDENCE_PATHS,
) -> tuple[dict[str, str], list[str]]:
    """Read small, explicit evidence files and preserve missing paths in the audit trail."""

    root = Path(project_root)
    texts: dict[str, str] = {}
    checked: list[str] = []
    for evidence_path in evidence_paths:
        path = Path(evidence_path)
        candidate = path if path.is_absolute() else root / path
        checked.append(str(candidate))
        texts[str(candidate)] = candidate.read_text(encoding="utf-8") if candidate.exists() else ""
    return texts, checked


def run(
    *,
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    exp1420_path: Path | str = DEFAULT_EXP1420_PATH,
    project_root: str | Path = REPO_ROOT,
    evidence_paths: Sequence[str] = DEFAULT_EVIDENCE_PATHS,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Write bootstrap and final Exp 1435 artifacts, then return the final JSON."""

    write_in_progress_artifact(out_path, project_root=project_root, run_date=run_date)
    exp1420_artifact = load_json(exp1420_path)
    evidence_texts, checked = read_evidence_texts(project_root, evidence_paths)
    checked = [str(exp1420_path), *checked]
    artifact = build_artifact(
        exp1420_artifact=exp1420_artifact,
        evidence_texts=evidence_texts,
        evidence_paths_checked=checked,
        project_root=project_root,
        run_date=run_date,
    )
    return _write_json(out_path, artifact)
