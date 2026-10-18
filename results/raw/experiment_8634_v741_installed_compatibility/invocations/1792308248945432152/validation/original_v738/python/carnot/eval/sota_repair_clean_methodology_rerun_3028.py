"""Exp 3028 clean-methodology SOTA repair evidence builder.

Spec: REQ-CODE-3028, SCENARIO-CODE-3028.

This module is a methodology repair pass over the Exp 3016 live repair
transcripts. Exp 3027 correctly flagged that Exp 3016 missed top-level seed and
transcript-hash fields; the candidate rows and transcript files still contain
the missing per-task evidence. Exp 3028 makes that evidence explicit, then
checks it with deterministic validators that are separate from the LLM
generation transcript.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from carnot.eval import gated_sota_intent_preserving_repair_hard_set as repair
from carnot.eval import hard_code_stress_manifest as hard
from carnot.eval import metamorphic_repair_oracle_audit as metamorphic
from carnot.eval import repair_acceptance_controller as acceptance


JsonDict = dict[str, Any]
ClockFunc = Callable[[], float]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260525"
SCHEMA = "carnot.sota_repair_clean_methodology_rerun.v2"
ARTIFACT = "experiment_3028_sota_repair_clean_methodology_rerun_v2"
OUTPUT_REL_PATH = Path("results/experiment_3028_sota_repair_clean_methodology_rerun_v2.json")

EXP3013_REL_PATH = Path("results/experiment_3013_sota_gguf_logprob_telemetry_preflight_v1.json")
EXP3015_REL_PATH = Path("results/experiment_3015_cactus_style_repair_acceptance_controller_v1.json")
EXP3016_REL_PATH = Path(
    "results/experiment_3016_sota_repair_rerun_with_acceptance_controller_v1.json"
)
EXP3027_REL_PATH = Path(
    "results/experiment_3027_adversarial_flag_methodology_corrigendum_v1.json"
)
EXP3002_REL_PATH = Path("results") / metamorphic.ARTIFACT_FILENAME
CONTROLLER_CONFIG_REL_PATH = acceptance.CONFIG_REL_PATH
HARD_MANIFEST_REL_PATH = hard.DEFAULT_MANIFEST_REL_PATH
METAMORPHIC_MANIFEST_REL_PATH = metamorphic.METAMORPHIC_MANIFEST_REL_PATH
RAW_REL_DIR = Path("results/raw") / ARTIFACT

HEADLINE_MODEL_IDS: tuple[str, ...] = repair.HEADLINE_MODEL_IDS
SMOKE_ONLY_MODEL_IDS: tuple[str, ...] = repair.SMOKE_ONLY_MODEL_IDS

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "clean_repair_rerun_ready",
    "repair_controller_clean",
    "clean_repair_claim_promotable_candidate",
    "n_tasks",
    "n_live_transcripts",
    "model_specs",
    "legacy_smoke_only_used",
    "pass_at_1_delta",
    "pass_at_k_delta",
    "syntax_failure_rate_delta",
    "schema_failure_rate_delta",
    "false_accept_delta",
    "tautology_gate_clean",
    "intent_drift_count",
    "reproducibility_checksum",
    "inference_substrate",
    "honest_verdict",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and timing hooks for Exp 3028."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    hard_manifest_path: Path | None = None
    metamorphic_manifest_path: Path | None = None
    controller_config_path: Path | None = None
    started_at: float | None = None
    clock: ClockFunc = time.time
    tests_run: Sequence[str] = field(default_factory=tuple)

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / OUTPUT_REL_PATH

    def resolved_hard_manifest_path(self) -> Path:
        return self.hard_manifest_path or self.repo_root / HARD_MANIFEST_REL_PATH

    def resolved_metamorphic_manifest_path(self, exp3002: Mapping[str, Any]) -> Path:
        if self.metamorphic_manifest_path is not None:
            return self.metamorphic_manifest_path
        return _resolve_repo_path(
            self.repo_root,
            exp3002.get("metamorphic_manifest_path") or METAMORPHIC_MANIFEST_REL_PATH,
        )

    def resolved_controller_config_path(self, exp3015: Mapping[str, Any]) -> Path:
        if self.controller_config_path is not None:
            return self.controller_config_path
        return _resolve_repo_path(
            self.repo_root,
            exp3015.get("controller_config_path") or CONTROLLER_CONFIG_REL_PATH,
        )


def build_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """REQ-CODE-3028: build clean repair evidence from live transcript rows."""

    config = config or ExperimentConfig()
    started = config.start_time()
    sources = _load_sources(config.repo_root)
    exp3002 = sources["exp3002"]
    controller_config = _read_json_if_present(
        config.resolved_controller_config_path(sources["exp3015"])
    )
    hard_items = _load_hard_items(config)
    variants = _load_metamorphic_variants(config, exp3002)
    substrate = _inference_substrate(config, sources)
    candidate_evidence = _candidate_evidence_rows(
        config=config,
        exp3016=sources["exp3016"],
        controller_rule=_mapping(controller_config.get("selected_rule")),
        hard_items=hard_items,
        variants=variants,
        tautology_gate_clean=_tautology_gate_clean(exp3002),
    )
    precondition_checks = _precondition_checks(
        config=config,
        sources=sources,
        controller_config=controller_config,
        hard_items=hard_items,
        variants=variants,
        candidate_evidence=candidate_evidence,
    )
    selected_tasks = _selected_tasks(hard_items, candidate_evidence)
    baseline_rows = _baseline_rows(selected_tasks, variants)
    accepted_rows = [row for row in candidate_evidence if row["controller_accepted"]]
    baseline_metrics = _metric_summary(baseline_rows, selected_tasks)
    accept_all_metrics = _metric_summary(candidate_evidence, selected_tasks)
    repair_metrics = _metric_summary(accepted_rows, selected_tasks)
    deltas = _metric_deltas(baseline_metrics, repair_metrics)
    legacy_smoke_only_used = any(row["model_hf_id"] in SMOKE_ONLY_MODEL_IDS for row in candidate_evidence)
    headline_models_used = sorted(
        {row["model_hf_id"] for row in candidate_evidence if row["model_hf_id"] in HEADLINE_MODEL_IDS}
    )
    intent_drift_count = sum(1 for row in accepted_rows if row["intent_drift"])
    candidate_intent_drift_count = sum(1 for row in candidate_evidence if row["intent_drift"])
    n_live_transcripts = sum(1 for row in candidate_evidence if row["live_transcript_present"])
    complete_reconstruction = bool(candidate_evidence) and all(
        row["checker_evidence_complete"] for row in candidate_evidence
    )
    substrate["reconstruction_mode"] = (
        "exp3016_nested_live_transcripts" if complete_reconstruction else "blocked_incomplete_transcripts"
    )
    clean = _clean_gate(
        complete_reconstruction=complete_reconstruction,
        headline_models_used=headline_models_used,
        legacy_smoke_only_used=legacy_smoke_only_used,
        accepted_rows=accepted_rows,
        deltas=deltas,
        tautology_gate_clean=_tautology_gate_clean(exp3002),
        intent_drift_count=intent_drift_count,
    )
    blocked = not complete_reconstruction and not headline_models_used
    if not complete_reconstruction and headline_models_used:
        blocked = True
    artifact = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "clean_repair_rerun_ready": clean,
        "repair_controller_clean": clean,
        "clean_repair_claim_promotable_candidate": clean,
        "n_tasks": len(selected_tasks) if complete_reconstruction else 0,
        "n_live_transcripts": n_live_transcripts,
        "model_specs": _model_specs_list(sources["exp3013"], sources["exp3016"], headline_models_used),
        "legacy_smoke_only_used": legacy_smoke_only_used,
        "pass_at_1_delta": deltas["pass_at_1_delta"],
        "pass_at_k_delta": deltas["pass_at_k_delta"],
        "syntax_failure_rate_delta": deltas["syntax_failure_rate_delta"],
        "schema_failure_rate_delta": deltas["schema_failure_rate_delta"],
        "false_accept_delta": deltas["false_accept_delta"],
        "tautology_gate_clean": _tautology_gate_clean(exp3002),
        "intent_drift_count": intent_drift_count,
        "reproducibility_checksum": _sha256_payload(
            {
                "candidate_hashes": [row["candidate_patch_sha256"] for row in candidate_evidence],
                "transcript_hashes": [row["transcript_hash"] for row in candidate_evidence],
                "seeds": [row["random_seed"] for row in candidate_evidence],
                "deltas": deltas,
                "headline_models_used": headline_models_used,
            }
        ),
        "inference_substrate": substrate,
        "honest_verdict": _honest_verdict(clean=clean, blocked=blocked, n_tasks=len(selected_tasks)),
        "precondition_checks": precondition_checks,
        "candidate_evidence": candidate_evidence,
        "baseline_metrics": baseline_metrics,
        "accept_all_metrics": accept_all_metrics,
        "repair_metrics": repair_metrics,
        "headline_models_used": headline_models_used,
        "accepted_candidate_count": len(accepted_rows),
        "rejected_candidate_count": len(candidate_evidence) - len(accepted_rows),
        "candidate_intent_drift_count": candidate_intent_drift_count,
        "source_artifacts": _source_artifacts(config, sources),
        "march_information_asymmetry": {
            "candidate_source": "exp3016_live_transcripts_and_patch_files",
            "checker_source": "exp3015_controller_plus_deterministic_original_and_metamorphic_validators",
            "source_row_does_not_grade_itself": True,
        },
        "duration_s": _elapsed(config, started),
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
    }
    return artifact


def write_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build and persist the Exp 3028 deliverable JSON."""

    config = config or ExperimentConfig()
    artifact = build_artifact(config)
    output_path = config.artifact_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _load_sources(root: Path) -> dict[str, JsonDict]:
    return {
        "exp3013": _read_json_if_present(root / EXP3013_REL_PATH),
        "exp3015": _read_json_if_present(root / EXP3015_REL_PATH),
        "exp3016": _read_json_if_present(root / EXP3016_REL_PATH),
        "exp3027": _read_json_if_present(root / EXP3027_REL_PATH),
        "exp3002": _read_json_if_present(root / EXP3002_REL_PATH),
    }


def _candidate_evidence_rows(
    *,
    config: ExperimentConfig,
    exp3016: Mapping[str, Any],
    controller_rule: Mapping[str, Any],
    hard_items: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
    tautology_gate_clean: bool,
) -> list[JsonDict]:
    item_by_id = {str(item.get("item_id") or ""): item for item in hard_items}
    variants_by_source = _variants_by_source(variants)
    rows = []
    for index, source_row in enumerate(exp3016.get("candidate_evaluations") or []):
        if not isinstance(source_row, Mapping):
            continue
        item_id = str(source_row.get("item_id") or "")
        item = item_by_id.get(item_id, {})
        patch_path = _resolve_repo_path(config.repo_root, source_row.get("candidate_patch_path"))
        transcript_path = _resolve_repo_path(config.repo_root, source_row.get("live_transcript_path"))
        verifier_path = _resolve_repo_path(config.repo_root, source_row.get("verifier_log_path"))
        patch_text = _read_text_if_present(patch_path)
        transcript = _read_json_if_present(transcript_path)
        verifier = _read_json_if_present(verifier_path)
        syntax_success, syntax_errors = repair.syntax_diagnostics(patch_text)
        original = _run_original_check(item, patch_text)
        variant_outcomes = _run_variant_checks(
            item=item,
            patch_text=patch_text,
            variants=variants_by_source.get(item_id, []),
        )
        metamorphic_passed_all = bool(variant_outcomes) and all(row.passed for row in variant_outcomes)
        original_passed = bool(original.passed)
        false_accept = bool(original_passed and not metamorphic_passed_all)
        draft_intent = str(source_row.get("draft_intent") or transcript.get("draft_intent") or "")
        expected_behavior = str(item.get("expected_behavior") or "")
        intent_drift = not _intent_preserved(draft_intent, expected_behavior)
        model_id = str(source_row.get("model_hf_id") or "")
        live_transcript_present = transcript_path.is_file()
        transcript_hash = _sha256_file(transcript_path) if live_transcript_present else None
        patch_present = patch_path.is_file()
        verifier_present = verifier_path.is_file()
        checker_evidence_complete = bool(
            item
            and patch_present
            and live_transcript_present
            and verifier_present
            and source_row.get("seed") is not None
            and transcript_hash
        )
        computed = {
            "schema_valid": bool(source_row.get("schema_valid")),
            "syntax_success": syntax_success,
            "entry_point_present": _entry_point_present(patch_text, str(item.get("entry_point") or "")),
            "false_accept": false_accept,
            "intent_drift": intent_drift,
            "original_passed": original_passed,
            "metamorphic_passed_all": metamorphic_passed_all,
            "tautology_probe_clean": tautology_gate_clean,
            "checker_evidence_complete": checker_evidence_complete,
            "model_hf_id": model_id,
        }
        rejection_reasons = _controller_rejection_reasons(computed, controller_rule)
        rejection_reasons.extend(_methodology_rejection_reasons(computed))
        controller_accepted = not rejection_reasons
        final_patch_sha = _sha256_text(patch_text) if patch_text else None
        rows.append(
            {
                "item_id": item_id,
                "model_hf_id": model_id,
                "random_seed": int(source_row.get("seed") or 0),
                "draft_intent": draft_intent,
                "expected_behavior": expected_behavior,
                "failing_trace": transcript.get("failing_trace") or {},
                "candidate_patch_path": _path_string(config.repo_root, patch_path),
                "candidate_patch_sha256": final_patch_sha,
                "final_patch": patch_text,
                "final_patch_sha256": final_patch_sha,
                "validator_output": verifier,
                "original_verifier_output": original.as_dict(),
                "metamorphic_verifier_outputs": [row.as_dict() for row in variant_outcomes],
                "generation_duration_s": float(
                    source_row.get("generation_duration_s")
                    or transcript.get("generation_duration_s")
                    or 0.0
                ),
                "live_transcript_path": _path_string(config.repo_root, transcript_path),
                "live_transcript_present": live_transcript_present,
                "transcript_hash": transcript_hash,
                "transcript_hash_matches_source": bool(
                    transcript_hash and transcript_hash == source_row.get("transcript_sha256")
                ),
                "verifier_log_path": _path_string(config.repo_root, verifier_path),
                "schema_valid": bool(source_row.get("schema_valid")),
                "syntax_success": syntax_success,
                "syntax_errors": syntax_errors,
                "entry_point_present": computed["entry_point_present"],
                "original_passed": original_passed,
                "metamorphic_passed_all": metamorphic_passed_all,
                "metamorphic_variant_count": len(variant_outcomes),
                "passed": bool(original_passed and metamorphic_passed_all),
                "false_accept": false_accept,
                "tautology_probe_clean": tautology_gate_clean,
                "intent_drift": intent_drift,
                "checker_evidence_complete": checker_evidence_complete,
                "controller_accepted": controller_accepted,
                "rejection_reasons": rejection_reasons,
                "candidate_index": index,
                "reproducibility_checksum": _sha256_payload(
                    {
                        "item_id": item_id,
                        "model_hf_id": model_id,
                        "seed": source_row.get("seed"),
                        "patch_sha256": final_patch_sha,
                        "transcript_hash": transcript_hash,
                    }
                ),
            }
        )
    return rows


def _precondition_checks(
    *,
    config: ExperimentConfig,
    sources: Mapping[str, Mapping[str, Any]],
    controller_config: Mapping[str, Any],
    hard_items: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
    candidate_evidence: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    complete_reconstruction = bool(candidate_evidence) and all(
        row.get("checker_evidence_complete") for row in candidate_evidence
    )
    return [
        {
            "resource": "exp3027_corrigendum",
            "available": sources["exp3027"].get("methodology_corrigendum_ready") is True,
            "repair_rerun_required": bool(sources["exp3027"].get("repair_rerun_required")),
        },
        {
            "resource": "exp3013_sota_preconditions",
            "available": sources["exp3013"].get("sota_headline_ready") is True,
            "path": EXP3013_REL_PATH.as_posix(),
        },
        {
            "resource": "exp3015_acceptance_controller",
            "available": sources["exp3015"].get("acceptance_controller_ready") is True,
            "path": EXP3015_REL_PATH.as_posix(),
        },
        {
            "resource": "acceptance_controller_config",
            "available": controller_config.get("policy_type") == "transparent_grid_rule"
            and isinstance(controller_config.get("selected_rule"), Mapping),
            "path": _path_string(config.repo_root, config.resolved_controller_config_path(sources["exp3015"])),
        },
        {
            "resource": "hard_manifest",
            "available": bool(hard_items),
            "path": _path_string(config.repo_root, config.resolved_hard_manifest_path()),
        },
        {
            "resource": "metamorphic_manifest",
            "available": bool(variants),
            "path": _path_string(config.repo_root, config.resolved_metamorphic_manifest_path(sources["exp3002"])),
        },
        {
            "resource": "complete_transcript_reconstruction",
            "available": complete_reconstruction,
            "n_candidates": len(candidate_evidence),
            "n_complete": sum(1 for row in candidate_evidence if row.get("checker_evidence_complete")),
        },
    ]


def _baseline_rows(
    tasks: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    variants_by_source = _variants_by_source(variants)
    rows = []
    for item in tasks:
        original = hard.run_candidate_tests(item, "baseline_candidate")
        variant_outcomes = [
            hard.run_candidate_tests(variant, "baseline_candidate")
            for variant in variants_by_source.get(str(item.get("item_id") or ""), [])
        ]
        metamorphic_passed_all = bool(variant_outcomes) and all(row.passed for row in variant_outcomes)
        rows.append(
            {
                "item_id": str(item.get("item_id") or ""),
                "schema_valid": True,
                "syntax_success": True,
                "original_passed": bool(original.passed),
                "metamorphic_passed_all": metamorphic_passed_all,
                "passed": bool(original.passed and metamorphic_passed_all),
                "false_accept": bool(original.passed and not metamorphic_passed_all),
            }
        )
    return rows


def _metric_summary(
    rows: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
) -> JsonDict:
    by_item = {str(item.get("item_id") or ""): [] for item in tasks}
    for row in rows:
        by_item.setdefault(str(row.get("item_id") or ""), []).append(row)
    per_task = []
    for item in tasks:
        item_id = str(item.get("item_id") or "")
        task_rows = by_item.get(item_id, [])
        pass_vector = [bool(row.get("passed")) for row in task_rows]
        per_task.append(
            {
                "item_id": item_id,
                "pass_vector": pass_vector,
                "pass_at_1": 1.0 if pass_vector and pass_vector[0] else 0.0,
                "pass_at_k": 1.0 if any(pass_vector) else 0.0,
            }
        )
    return {
        "candidate_count": len(rows),
        "per_task_results": per_task,
        "pass_at_1": _mean([row["pass_at_1"] for row in per_task]),
        "pass_at_k": _mean([row["pass_at_k"] for row in per_task]),
        "schema_failure_rate": _rate(rows, lambda row: row.get("schema_valid") is False),
        "syntax_failure_rate": _rate(rows, lambda row: row.get("syntax_success") is False),
        "false_accept_rate": _rate(rows, lambda row: row.get("false_accept") is True),
    }


def _metric_deltas(baseline: Mapping[str, Any], repair_metrics: Mapping[str, Any]) -> JsonDict:
    return {
        "pass_at_1_delta": _delta(repair_metrics.get("pass_at_1"), baseline.get("pass_at_1")),
        "pass_at_k_delta": _delta(repair_metrics.get("pass_at_k"), baseline.get("pass_at_k")),
        "syntax_failure_rate_delta": _delta(
            repair_metrics.get("syntax_failure_rate"), baseline.get("syntax_failure_rate")
        ),
        "schema_failure_rate_delta": _delta(
            repair_metrics.get("schema_failure_rate"), baseline.get("schema_failure_rate")
        ),
        "false_accept_delta": _delta(
            repair_metrics.get("false_accept_rate"), baseline.get("false_accept_rate")
        ),
    }


def _clean_gate(
    *,
    complete_reconstruction: bool,
    headline_models_used: Sequence[str],
    legacy_smoke_only_used: bool,
    accepted_rows: Sequence[Mapping[str, Any]],
    deltas: Mapping[str, Any],
    tautology_gate_clean: bool,
    intent_drift_count: int,
) -> bool:
    return bool(
        complete_reconstruction
        and headline_models_used
        and not legacy_smoke_only_used
        and accepted_rows
        and _positive(deltas.get("pass_at_1_delta"))
        and _nonnegative(deltas.get("pass_at_k_delta"))
        and _nonpositive(deltas.get("syntax_failure_rate_delta"))
        and _nonpositive(deltas.get("schema_failure_rate_delta"))
        and _nonpositive(deltas.get("false_accept_delta"))
        and tautology_gate_clean
        and intent_drift_count == 0
    )


def _controller_rejection_reasons(row: Mapping[str, Any], rule: Mapping[str, Any]) -> list[str]:
    checks = [
        ("require_schema_valid", "schema_valid", row.get("schema_valid") is True),
        ("require_syntax_success", "syntax_success", row.get("syntax_success") is True),
        ("require_entry_point_present", "entry_point_present", row.get("entry_point_present") is True),
        ("require_false_accept_probe_clean", "false_accept", row.get("false_accept") is False),
        ("require_no_intent_drift", "intent_drift", row.get("intent_drift") is False),
        ("require_original_passed", "original_passed", row.get("original_passed") is True),
        (
            "require_metamorphic_passed_all",
            "metamorphic_passed_all",
            row.get("metamorphic_passed_all") is True,
        ),
        (
            "require_tautology_probe_clean",
            "tautology_probe_clean",
            row.get("tautology_probe_clean") is True,
        ),
    ]
    return [reason for flag, reason, passed in checks if rule.get(flag) and not passed]


def _methodology_rejection_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    model_id = str(row.get("model_hf_id") or "")
    if model_id in SMOKE_ONLY_MODEL_IDS:
        reasons.append("legacy_smoke_only_model")
    elif model_id not in HEADLINE_MODEL_IDS:
        reasons.append("non_headline_model")
    if row.get("checker_evidence_complete") is not True:
        reasons.append("missing_checker_evidence")
    return reasons


def _model_specs_list(
    exp3013: Mapping[str, Any],
    exp3016: Mapping[str, Any],
    headline_models_used: Sequence[str],
) -> list[JsonDict]:
    runnable = _mapping((exp3016.get("model_specs") or {}).get("runnable_headline_models"))
    if not runnable and isinstance((exp3016.get("model_specs") or {}).get("runnable_headline_models"), list):
        rows = (exp3016.get("model_specs") or {}).get("runnable_headline_models")
    else:
        rows = []
    cache_paths = _mapping((exp3013.get("cache_paths") or {}).get("headline_models"))
    checksums = _mapping(exp3016.get("model_checksums") or exp3013.get("model_checksums"))
    out = []
    for model_id in headline_models_used:
        row = _first_mapping(rows, "hf_id", model_id)
        checksum = _mapping(checksums.get(model_id))
        out.append(
            {
                "hf_id": model_id,
                "model_path": row.get("model_path") or cache_paths.get(model_id) or checksum.get("path"),
                "checksum": checksum.get("bounded_sha256") or checksum.get("sha256"),
            }
        )
    return out


def _inference_substrate(config: ExperimentConfig, sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    exp3013_evidence = _mapping(sources["exp3013"].get("precondition_evidence"))
    current_gpu = _nvidia_smi_inventory(config.repo_root)
    repo_commit = _git_commit(config.repo_root)
    return {
        "kind": "clean_repair_reconstruction",
        "recorded_before_model_load": True,
        "model_load_attempted": False,
        "live_repair_generation_run": False,
        "exp3027_repair_rerun_required": bool(sources["exp3027"].get("repair_rerun_required")),
        "exp3027_decision": _mapping(sources["exp3027"].get("repair_rerun_decision")).get("decision"),
        "cuda_available": _cuda_available_from_evidence(exp3013_evidence, current_gpu),
        "gpu_inventory": current_gpu or exp3013_evidence.get("gpu_inventory") or {},
        "free_vram_mib_total": _free_vram(exp3013_evidence, current_gpu),
        "repo_commit": repo_commit or _mapping(exp3013_evidence.get("repo_commit")).get("commit"),
        "python_environment": {
            "executable": sys.executable,
            "version": sys.version,
            "source": exp3013_evidence.get("python_environment") or {},
        },
        "gguf_cache_paths": (sources["exp3013"].get("cache_paths") or {}).get("headline_models") or {},
        "model_checksum_feasibility": exp3013_evidence.get("checksum_feasibility") or {},
    }


def _source_artifacts(config: ExperimentConfig, sources: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    del sources
    paths = [
        EXP3013_REL_PATH,
        EXP3015_REL_PATH,
        EXP3016_REL_PATH,
        EXP3027_REL_PATH,
        EXP3002_REL_PATH,
        config.resolved_hard_manifest_path(),
        config.resolved_metamorphic_manifest_path(_read_json_if_present(config.repo_root / EXP3002_REL_PATH)),
    ]
    out = []
    for path_value in paths:
        path = path_value if isinstance(path_value, Path) and path_value.is_absolute() else config.repo_root / path_value
        out.append(
            {
                "path": _path_string(config.repo_root, path),
                "present": path.is_file(),
                "sha256": _sha256_file(path) if path.is_file() else None,
            }
        )
    return out


def _selected_tasks(
    hard_items: Sequence[Mapping[str, Any]],
    candidate_evidence: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    wanted = {str(row.get("item_id") or "") for row in candidate_evidence}
    return [item for item in hard_items if str(item.get("item_id") or "") in wanted]


def _load_hard_items(config: ExperimentConfig) -> list[JsonDict]:
    path = config.resolved_hard_manifest_path()
    if not path.is_file():
        return []
    try:
        return [dict(item) for item in hard.load_manifest(path)]
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return []


def _load_metamorphic_variants(config: ExperimentConfig, exp3002: Mapping[str, Any]) -> list[JsonDict]:
    path = config.resolved_metamorphic_manifest_path(exp3002)
    if not path.is_file():
        return []
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return []


def _run_original_check(item: Mapping[str, Any], patch_text: str) -> hard.VerificationOutcome:
    if not item:
        return hard.VerificationOutcome(
            False,
            "repair_candidate",
            _sha256_text(patch_text),
            "",
            0,
            [],
            [{"test_id": "missing-item", "error_type": "MissingItem", "message": "item not found"}],
        )
    return hard.run_candidate_tests({**dict(item), "repair_candidate": patch_text}, "repair_candidate")


def _run_variant_checks(
    *,
    item: Mapping[str, Any],
    patch_text: str,
    variants: Sequence[Mapping[str, Any]],
) -> list[hard.VerificationOutcome]:
    out = []
    for variant in variants:
        adapted = metamorphic._adapt_candidate(
            patch_text,
            str(variant.get("source_entry_point") or item.get("entry_point") or ""),
            str(variant.get("entry_point") or ""),
        )
        out.append(hard.run_candidate_tests({**dict(variant), "repair_candidate": adapted}, "repair_candidate"))
    return out


def _variants_by_source(variants: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = {}
    for variant in variants:
        out.setdefault(str(variant.get("source_item_id") or ""), []).append(variant)
    return out


def _tautology_gate_clean(exp3002: Mapping[str, Any]) -> bool:
    rejected = exp3002.get("rejected_variants") or []
    return bool(
        exp3002.get("tautology_probe_ready") is True
        and any(row.get("reason") == "tautological_oracle_rejected" for row in rejected)
    )


def _intent_preserved(draft_intent: str, expected_behavior: str) -> bool:
    if not draft_intent.strip() or not expected_behavior.strip():
        return False
    draft_tokens = set(_content_tokens(draft_intent))
    expected_tokens = set(_content_tokens(expected_behavior))
    if not expected_tokens:
        return True
    overlap = len(draft_tokens.intersection(expected_tokens))
    return overlap >= min(2, len(expected_tokens))


def _content_tokens(text: str) -> list[str]:
    stop = {"the", "and", "into", "that", "with", "from", "each", "once", "return"}
    token = ""
    out: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            token += ch
        elif token:
            if len(token) > 2 and token not in stop:
                out.append(token)
            token = ""
    if token and len(token) > 2 and token not in stop:
        out.append(token)
    return out


def _entry_point_present(code: str, entry_point: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.FunctionDef) and node.name == entry_point for node in tree.body)


def _honest_verdict(*, clean: bool, blocked: bool, n_tasks: int) -> str:
    if clean:
        return f"complete: clean_repair_rerun_ready=true; n_tasks={n_tasks}"
    if blocked:
        return "blocked_sota_headline_model_unavailable: complete transcript reconstruction unavailable and no live headline panel executed"
    return f"complete_flagged: clean_repair_rerun_ready=false; n_tasks={n_tasks}"


def _read_json_if_present(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_text_if_present(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _resolve_repo_path(root: Path, value: Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else root / path


def _path_string(root: Path, path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()
    except ValueError:
        return path.resolve(strict=False).as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_payload(payload: Any) -> str:
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_mapping(rows: Sequence[Any], key: str, value: str) -> Mapping[str, Any]:
    for row in rows:
        if isinstance(row, Mapping) and row.get(key) == value:
            return row
    return {}


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def _rate(rows: Sequence[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]) -> float:
    return 0.0 if not rows else sum(1 for row in rows if predicate(row)) / len(rows)


def _delta(after: Any, before: Any) -> float:
    return float(after or 0.0) - float(before or 0.0)


def _positive(value: Any) -> bool:
    return isinstance(value, int | float) and value > 0


def _nonnegative(value: Any) -> bool:
    return isinstance(value, int | float) and value >= 0


def _nonpositive(value: Any) -> bool:
    return isinstance(value, int | float) and value <= 0


def _elapsed(config: ExperimentConfig, started: float) -> float:
    return round(max(0.0, config.clock() - started), 6)


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _nvidia_smi_inventory(root: Path) -> JsonDict:
    del root
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if result.returncode != 0:
        return {"available": False, "stderr_summary": result.stderr.strip()[:200]}
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 6:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mib": int(parts[2]),
                    "memory_used_mib": int(parts[3]),
                    "memory_free_mib": int(parts[4]),
                    "driver_version": parts[5],
                }
            )
    return {
        "available": bool(gpus),
        "gpus": gpus,
        "free_vram_mib_total": sum(gpu["memory_free_mib"] for gpu in gpus),
    }


def _cuda_available_from_evidence(exp3013_evidence: Mapping[str, Any], current_gpu: Mapping[str, Any]) -> bool:
    torch_cuda = _mapping(exp3013_evidence.get("torch_cuda"))
    gpu_inventory = _mapping(exp3013_evidence.get("gpu_inventory"))
    return bool(
        current_gpu.get("available") is True
        or torch_cuda.get("cuda_available") is True
        or gpu_inventory.get("available") is True
    )


def _free_vram(exp3013_evidence: Mapping[str, Any], current_gpu: Mapping[str, Any]) -> int:
    if isinstance(current_gpu.get("free_vram_mib_total"), int):
        return int(current_gpu["free_vram_mib_total"])
    gpu_inventory = _mapping(exp3013_evidence.get("gpu_inventory"))
    return int(gpu_inventory.get("free_vram_mib_total") or 0)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--test-run", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    args = _parse_args(argv)
    artifact = write_artifact(
        ExperimentConfig(output_path=args.output, tests_run=tuple(args.test_run))
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if not artifact["honest_verdict"].startswith("blocked") else 1


__all__ = [
    "ARTIFACT",
    "CONTROLLER_CONFIG_REL_PATH",
    "EXP3013_REL_PATH",
    "EXP3015_REL_PATH",
    "EXP3016_REL_PATH",
    "EXP3027_REL_PATH",
    "ExperimentConfig",
    "OUTPUT_REL_PATH",
    "RAW_REL_DIR",
    "REQUIRED_ARTIFACT_FIELDS",
    "build_artifact",
    "main",
    "write_artifact",
]
