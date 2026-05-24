"""Exp 3003 gated SOTA repair rerun with metamorphic false-accept evidence.

Spec: REQ-CODE-3003, SCENARIO-CODE-3003.

This module is the promotion gate after the SOTA cache refresh and the
metamorphic oracle audit. It does not let a repair row promote just because it
improves visible hard-set tests: every headline repair candidate is replayed
against the original hard-set validators and the Exp 3002 metamorphic variants,
then schema, syntax, false-accept, tautology, sample-size, and provenance gates
decide the terminal verdict.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from carnot.eval import hard_code_stress_manifest as hard
from carnot.eval import metamorphic_repair_oracle_audit as metamorphic
from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]
ClockFunc = Callable[[], float]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260524"
ARTIFACT_NAME = "experiment_3003_gated_sota_repair_metamorphic_false_accept_rerun_v1"
ARTIFACT_FILENAME = f"{ARTIFACT_NAME}.json"
SCHEMA = "carnot.gated_sota_repair_metamorphic_false_accept_rerun.v1"
EXP3001_FILENAME = "experiment_3001_sota_gguf_cache_carry_forward_checksum_refresh_v1.json"
EXP3002_FILENAME = "experiment_3002_metamorphic_repair_oracle_audit_v1.json"
EXP2991_FILENAME = "experiment_2991_gated_sota_intent_preserving_repair_hard_set_v1.json"
INFERENCE_SUBSTRATE = "live_llm_inference_with_metamorphic_replay"

HEADLINE_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
SMOKE_ONLY_MODEL_IDS: tuple[str, ...] = (
    "Qwen/Qwen3.5-0.8B",
    "unsloth/gemma-4-E4B-it-GGUF",
)
MIN_HEADLINE_TASKS = 20
RAW_REL_DIR = Path("results/raw") / ARTIFACT_NAME
VERIFIER_REL_DIR = Path("results/verifier_transcripts/experiment_3003")
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "repair_rerun_clean",
    "headline_result",
    "preconditions_checked",
    "n_tasks",
    "n_metamorphic_variants",
    "model_specs",
    "headline_models_used",
    "model_checksums",
    "pass_at_1_delta",
    "pass_at_k_delta",
    "false_accept_delta",
    "tautology_gate_clean",
    "syntax_failure_rate_delta",
    "live_transcript_paths",
    "verifier_log_paths",
    "honest_verdict",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and knobs for the Exp 3003 terminal gate."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    hard_manifest_path: Path | None = None
    metamorphic_manifest_path: Path | None = None
    exp2991_artifact_path: Path | None = None
    exp3001_artifact_path: Path | None = None
    exp3002_artifact_path: Path | None = None
    raw_dir: Path | None = None
    verifier_dir: Path | None = None
    n_tasks: int = 24
    tests_run: Sequence[str] = field(default_factory=tuple)
    started_at: float | None = None
    clock: ClockFunc = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / ARTIFACT_FILENAME

    def resolved_hard_manifest_path(self) -> Path:
        return self.hard_manifest_path or self.repo_root / hard.DEFAULT_MANIFEST_REL_PATH

    def resolved_exp2991_artifact_path(self) -> Path:
        return self.exp2991_artifact_path or self.repo_root / "results" / EXP2991_FILENAME

    def resolved_exp3001_artifact_path(self) -> Path:
        return self.exp3001_artifact_path or self.repo_root / "results" / EXP3001_FILENAME

    def resolved_exp3002_artifact_path(self) -> Path:
        return self.exp3002_artifact_path or self.repo_root / "results" / EXP3002_FILENAME

    def resolved_raw_dir(self) -> Path:
        return self.raw_dir or self.repo_root / RAW_REL_DIR

    def resolved_patch_dir(self) -> Path:
        return self.resolved_raw_dir() / "patches"

    def resolved_transcript_dir(self) -> Path:
        return self.resolved_raw_dir() / "transcripts"

    def resolved_verifier_dir(self) -> Path:
        return self.verifier_dir or self.repo_root / VERIFIER_REL_DIR


@dataclass(frozen=True)
class PreconditionReport:
    """Exp 3003 setup evidence gathered before repair promotion is evaluated."""

    checks: list[JsonDict]
    model_specs: JsonDict
    model_checksums: JsonDict
    exp3001_payload: JsonDict
    exp3002_payload: JsonDict


def build_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build the terminal Exp 3003 artifact."""

    config = config or ExperimentConfig()
    started = config.start_time()
    preconditions = _precondition_report(config)
    hard_report = _validate_hard_manifest(config)
    metamorphic_report = _validate_metamorphic_manifest(config, preconditions.exp3002_payload)
    checks = [*preconditions.checks, *hard_report["checks"], *metamorphic_report["checks"]]

    if not _checks_available(checks):
        return _blocked_artifact(config, started, checks, preconditions)

    tasks = list(hard_report["items"][: config.n_tasks])
    variants = _variants_for_tasks(metamorphic_report["variants"], tasks)
    if len(tasks) < MIN_HEADLINE_TASKS or not variants:
        checks = [
            *checks,
            {
                "resource": "exp3003_sample_size",
                "available": False,
                "n_tasks": len(tasks),
                "n_metamorphic_variants": len(variants),
                "minimum_tasks": MIN_HEADLINE_TASKS,
            },
        ]
        return _blocked_artifact(config, started, checks, preconditions)

    candidate_report = _load_exp2991_candidates(config, tasks)
    checks.append(candidate_report["check"])
    if not candidate_report["headline_candidates"]:
        return _blocked_artifact(config, started, checks, preconditions)

    baseline_rows = _baseline_rows(tasks, variants)
    candidate_rows = _candidate_rows(
        config,
        tasks,
        variants,
        candidate_report["headline_candidates"],
    )
    return _complete_artifact(
        config=config,
        started=started,
        preconditions=preconditions,
        precondition_checks=checks,
        tasks=tasks,
        variants=variants,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        smoke_only_candidate_count=int(candidate_report["smoke_only_candidate_count"]),
    )


def write_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build and persist the Exp 3003 terminal JSON artifact."""

    config = config or ExperimentConfig()
    artifact = build_artifact(config)
    path = config.artifact_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _precondition_report(config: ExperimentConfig) -> PreconditionReport:
    exp3001_path = config.resolved_exp3001_artifact_path()
    exp3002_path = config.resolved_exp3002_artifact_path()
    exp3001 = _read_json_if_present(exp3001_path)
    exp3002 = _read_json_if_present(exp3002_path)
    model_checksums = dict(exp3001.get("model_checksums") or {})
    cached_pair = _call_cached_sota_pair()
    checks = [
        {
            "resource": "exp3001_sota_cache_carry_forward",
            "available": exp3001_path.is_file()
            and exp3001.get("sota_headline_ready") is True
            and exp3001.get("preconditions_checked") is True,
            "path": str(_relative_or_absolute(config.repo_root, exp3001_path)),
            "sha256": _sha256_file(exp3001_path) if exp3001_path.is_file() else None,
        },
        {
            "resource": "exp3002_metamorphic_oracle_audit",
            "available": exp3002_path.is_file() and exp3002.get("metamorphic_oracle_ready") is True,
            "path": str(_relative_or_absolute(config.repo_root, exp3002_path)),
            "sha256": _sha256_file(exp3002_path) if exp3002_path.is_file() else None,
        },
    ]
    cuda_status = _cuda_status(exp3001)
    checks.append({"resource": "cuda_status", "available": cuda_status["available"], **cuda_status})
    checksum_models = [
        model_id
        for model_id in HEADLINE_MODEL_IDS
        if _checksum_available(model_checksums.get(model_id))
    ]
    checks.append(
        {
            "resource": "headline_model_checksum_available",
            "available": bool(checksum_models),
            "available_models": checksum_models,
        }
    )
    model_specs = {
        "headline_models": list(HEADLINE_MODEL_IDS),
        "smoke_only_models": list(SMOKE_ONLY_MODEL_IDS),
        "preferred_quantization": "Q4_K_M",
        "source_helpers": ["cached_sota_pair", "experiment_3001_cache_paths"],
        "cached_sota_pair_returned": bool(cached_pair),
        "exp3001_cache_paths": (exp3001.get("cache_paths") or {}).get("headline_models") or {},
    }
    return PreconditionReport(
        checks=checks,
        model_specs=model_specs,
        model_checksums=model_checksums,
        exp3001_payload=exp3001,
        exp3002_payload=exp3002,
    )


def _validate_hard_manifest(config: ExperimentConfig) -> JsonDict:
    path = config.resolved_hard_manifest_path()
    if not path.is_file():
        return {
            "items": [],
            "checks": [
                {
                    "resource": "hard_set_integrity",
                    "available": False,
                    "detail": f"missing {path}",
                }
            ],
        }
    items = hard.load_manifest(path)
    baseline = [hard.run_candidate_tests(item, "baseline_candidate") for item in items]
    reference = [hard.run_candidate_tests(item, "reference_solution") for item in items]
    ready = bool(
        len(items) >= MIN_HEADLINE_TASKS
        and all(bool(item.get("tests")) for item in items)
        and all(not row.passed for row in baseline)
        and all(row.passed for row in reference)
    )
    return {
        "items": items,
        "checks": [
            {
                "resource": "hard_set_integrity",
                "available": ready,
                "path": str(_relative_or_absolute(config.repo_root, path)),
                "sha256": _sha256_file(path),
                "n_items": len(items),
                "all_baseline_candidates_fail": all(not row.passed for row in baseline),
                "all_reference_solutions_pass": all(row.passed for row in reference),
            }
        ],
    }


def _validate_metamorphic_manifest(config: ExperimentConfig, exp3002: Mapping[str, Any]) -> JsonDict:
    path = _metamorphic_manifest_path(config, exp3002)
    if not path.is_file():
        return {
            "variants": [],
            "checks": [
                {
                    "resource": "metamorphic_manifest_integrity",
                    "available": False,
                    "detail": f"missing {path}",
                }
            ],
        }
    variants = _read_jsonl(path)
    reference = [hard.run_candidate_tests(variant, "reference_solution") for variant in variants]
    ready = bool(variants and all(row.passed for row in reference))
    return {
        "variants": variants,
        "checks": [
            {
                "resource": "metamorphic_manifest_integrity",
                "available": ready,
                "path": str(_relative_or_absolute(config.repo_root, path)),
                "sha256": _sha256_file(path),
                "n_metamorphic_variants": len(variants),
                "all_reference_solutions_pass": all(row.passed for row in reference),
            }
        ],
    }


def _metamorphic_manifest_path(config: ExperimentConfig, exp3002: Mapping[str, Any]) -> Path:
    if config.metamorphic_manifest_path is not None:
        return config.metamorphic_manifest_path
    rel = exp3002.get("metamorphic_manifest_path") or metamorphic.METAMORPHIC_MANIFEST_REL_PATH
    return config.repo_root / str(rel)


def _load_exp2991_candidates(config: ExperimentConfig, tasks: Sequence[Mapping[str, Any]]) -> JsonDict:
    path = config.resolved_exp2991_artifact_path()
    payload = _read_json_if_present(path)
    wanted = {str(item.get("item_id") or "") for item in tasks}
    headline: list[JsonDict] = []
    smoke_count = 0
    for index, row in enumerate(payload.get("candidate_evaluations") or []):
        item_id = str(row.get("item_id") or "")
        model_id = str(row.get("model_hf_id") or "")
        if item_id not in wanted:
            continue
        if model_id in SMOKE_ONLY_MODEL_IDS:
            smoke_count += 1
            continue
        if model_id not in HEADLINE_MODEL_IDS:
            continue
        patch_path = _resolve_repo_path(config.repo_root, row.get("candidate_patch_path"))
        transcript_path = _resolve_repo_path(config.repo_root, row.get("transcript_path"))
        if patch_path.is_file() and transcript_path.is_file():
            headline.append({**dict(row), "source_index": index})
    return {
        "headline_candidates": headline,
        "smoke_only_candidate_count": smoke_count,
        "check": {
            "resource": "exp2991_headline_candidate_provenance",
            "available": bool(headline),
            "path": str(_relative_or_absolute(config.repo_root, path)),
            "sha256": _sha256_file(path) if path.is_file() else None,
            "headline_candidate_count": len(headline),
            "smoke_only_candidate_count": smoke_count,
        },
    }


def _baseline_rows(
    tasks: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    variants_by_source = _variants_by_source(variants)
    rows: list[JsonDict] = []
    for item in tasks:
        source_id = str(item.get("item_id") or "")
        original = hard.run_candidate_tests(item, "baseline_candidate")
        variant_outcomes = [
            hard.run_candidate_tests(variant, "baseline_candidate")
            for variant in variants_by_source.get(source_id, ())
        ]
        rows.append(
            _combined_row(
                item=item,
                mode="baseline_candidate",
                candidate_code=str(item.get("baseline_candidate") or ""),
                original=original,
                variant_outcomes=variant_outcomes,
                schema_valid=True,
                syntax_success=True,
                model_id="",
            )
        )
    return rows


def _candidate_rows(
    config: ExperimentConfig,
    tasks: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    items_by_id = {str(item.get("item_id") or ""): item for item in tasks}
    variants_by_source = _variants_by_source(variants)
    out: list[JsonDict] = []
    for candidate_index, source in enumerate(candidates):
        item_id = str(source.get("item_id") or "")
        item = items_by_id[item_id]
        patch_path = _resolve_repo_path(config.repo_root, source.get("candidate_patch_path"))
        patch_code = patch_path.read_text(encoding="utf-8")
        original = hard.run_candidate_tests({**dict(item), "repair_candidate": patch_code}, "repair_candidate")
        variant_outcomes = []
        for variant in variants_by_source.get(item_id, ()):
            adapted = metamorphic._adapt_candidate(
                patch_code,
                str(variant.get("source_entry_point") or item.get("entry_point") or ""),
                str(variant.get("entry_point") or ""),
            )
            outcome = hard.run_candidate_tests({**dict(variant), "repair_candidate": adapted}, "repair_candidate")
            variant_outcomes.append(outcome)
        row = _combined_row(
            item=item,
            mode="headline_repair_candidate",
            candidate_code=patch_code,
            original=original,
            variant_outcomes=variant_outcomes,
            schema_valid=bool(source.get("schema_valid", True)),
            syntax_success=_candidate_syntax_success(source, patch_code),
            model_id=str(source.get("model_hf_id") or ""),
        )
        _write_candidate_evidence(config, row, source, item, patch_code, candidate_index)
        out.append(row)
    return out


def _combined_row(
    *,
    item: Mapping[str, Any],
    mode: str,
    candidate_code: str,
    original: hard.VerificationOutcome,
    variant_outcomes: Sequence[hard.VerificationOutcome],
    schema_valid: bool,
    syntax_success: bool,
    model_id: str,
) -> JsonDict:
    variants_all_pass = bool(variant_outcomes) and all(row.passed for row in variant_outcomes)
    combined_passed = bool(original.passed and variants_all_pass)
    false_accept = bool(original.passed and not combined_passed)
    return {
        "mode": mode,
        "item_id": str(item.get("item_id") or ""),
        "entry_point": str(item.get("entry_point") or ""),
        "model_hf_id": model_id,
        "candidate_sha256": _sha256_text(candidate_code),
        "schema_valid": bool(schema_valid),
        "syntax_success": bool(syntax_success),
        "original_passed": bool(original.passed),
        "metamorphic_passed_all": variants_all_pass,
        "metamorphic_variant_count": len(variant_outcomes),
        "metamorphic_pass_count": sum(1 for row in variant_outcomes if row.passed),
        "passed": combined_passed,
        "false_accept": false_accept,
        "original_verifier_output": original.as_dict(),
        "metamorphic_verifier_outputs": [row.as_dict() for row in variant_outcomes],
        "failing_assertions": list(original.failing_test_ids),
    }


def _write_candidate_evidence(
    config: ExperimentConfig,
    row: JsonDict,
    source: Mapping[str, Any],
    item: Mapping[str, Any],
    patch_code: str,
    candidate_index: int,
) -> None:
    token = _safe_token(f"{row['item_id']}_{row['model_hf_id']}_{candidate_index}")
    patch_path = config.resolved_patch_dir() / f"{token}.py"
    transcript_path = config.resolved_transcript_dir() / f"{token}.json"
    verifier_path = config.resolved_verifier_dir() / f"{token}.json"
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    verifier_path.parent.mkdir(parents=True, exist_ok=True)

    source_transcript = _resolve_repo_path(config.repo_root, source.get("transcript_path"))
    source_transcript_payload = _read_json_if_present(source_transcript)
    prompt = str(source_transcript_payload.get("prompt") or "")
    draft_intent = str(source.get("draft_intent") or source_transcript_payload.get("draft_intent") or "")
    generation_duration = float(
        source.get("generation_duration_s")
        or source_transcript_payload.get("generation_duration_s")
        or 0.0
    )
    patch_path.write_text(patch_code, encoding="utf-8")
    transcript = {
        "item_id": row["item_id"],
        "model_hf_id": row["model_hf_id"],
        "prompt": prompt,
        "prompt_sha256": _sha256_text(prompt),
        "draft_intent": draft_intent,
        "failing_trace": item.get("baseline_verification") or {},
        "final_patch_sha256": _sha256_text(patch_code),
        "generation_duration_s": generation_duration,
        "generation_backend": source.get("generation_backend"),
        "source_transcript_path": str(_relative_or_absolute(config.repo_root, source_transcript)),
        "source_transcript_sha256": _sha256_file(source_transcript) if source_transcript.is_file() else None,
    }
    verifier = {
        "item_id": row["item_id"],
        "model_hf_id": row["model_hf_id"],
        "original_verifier_output": row["original_verifier_output"],
        "metamorphic_verifier_outputs": row["metamorphic_verifier_outputs"],
        "false_accept": row["false_accept"],
        "passed": row["passed"],
    }
    transcript_path.write_text(json.dumps(transcript, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verifier_path.write_text(json.dumps(verifier, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    row["candidate_patch_path"] = str(_relative_or_absolute(config.repo_root, patch_path))
    row["transcript_path"] = str(_relative_or_absolute(config.repo_root, transcript_path))
    row["verifier_log_path"] = str(_relative_or_absolute(config.repo_root, verifier_path))
    row["live_transcript_path"] = str(_relative_or_absolute(config.repo_root, source_transcript))
    row["live_transcript_sha256"] = _sha256_file(source_transcript) if source_transcript.is_file() else None


def _complete_artifact(
    *,
    config: ExperimentConfig,
    started: float,
    preconditions: PreconditionReport,
    precondition_checks: list[JsonDict],
    tasks: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
    baseline_rows: list[JsonDict],
    candidate_rows: list[JsonDict],
    smoke_only_candidate_count: int,
) -> JsonDict:
    baseline = _metric_summary(baseline_rows, tasks)
    repair = _metric_summary(candidate_rows, tasks)
    deltas = _metric_deltas(baseline, repair)
    headline_models_used = sorted(
        {str(row["model_hf_id"]) for row in candidate_rows if row.get("model_hf_id") in HEADLINE_MODEL_IDS}
    )
    live_transcript_paths = _unique_paths(row.get("live_transcript_path") for row in candidate_rows)
    verifier_log_paths = _unique_paths(row.get("verifier_log_path") for row in candidate_rows)
    candidate_patch_paths = _unique_paths(row.get("candidate_patch_path") for row in candidate_rows)
    transcript_paths = _unique_paths(row.get("transcript_path") for row in candidate_rows)
    headline_result = bool(headline_models_used and live_transcript_paths)
    tautology_gate_clean = _tautology_gate_clean(preconditions.exp3002_payload)
    clean = _repair_rerun_clean(
        headline_result=headline_result,
        n_tasks=len(tasks),
        n_metamorphic_variants=len(variants),
        headline_models_used=headline_models_used,
        smoke_only_candidate_count=smoke_only_candidate_count,
        deltas=deltas,
        tautology_gate_clean=tautology_gate_clean,
    )
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT_NAME,
        "run_date": RUN_DATE,
        "repair_rerun_clean": clean,
        "headline_result": headline_result,
        "preconditions_checked": True,
        "n_tasks": len(tasks),
        "n_metamorphic_variants": len(variants),
        "model_specs": preconditions.model_specs,
        "headline_models_used": headline_models_used,
        "model_checksums": preconditions.model_checksums,
        "pass_at_1_delta": deltas["pass_at_1_delta"],
        "pass_at_k_delta": deltas["pass_at_k_delta"],
        "false_accept_delta": deltas["false_accept_delta"],
        "tautology_gate_clean": tautology_gate_clean,
        "syntax_failure_rate_delta": deltas["syntax_failure_rate_delta"],
        "schema_failure_rate_delta": deltas["schema_failure_rate_delta"],
        "live_transcript_paths": live_transcript_paths,
        "verifier_log_paths": verifier_log_paths,
        "honest_verdict": (
            "clean: metamorphic repair rerun gates passed"
            if clean
            else "flagged: metamorphic repair rerun did not clear gates"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "baseline_metrics": baseline,
        "repair_metrics": repair,
        "candidate_evaluations": candidate_rows,
        "baseline_evaluations": baseline_rows,
        "candidate_patch_paths": candidate_patch_paths,
        "transcript_paths": transcript_paths,
        "selected_item_ids": [str(item.get("item_id") or "") for item in tasks],
        "precondition_checks": precondition_checks,
        "source_artifacts": _source_artifacts(config),
        "exp2991_comparison": _exp2991_comparison(config),
        "smoke_only_candidate_count": smoke_only_candidate_count,
        "candidate_manifest_sha256": _sha256_payload(candidate_rows),
        "reproducibility_checksum": _sha256_payload(
            {
                "selected_item_ids": [str(item.get("item_id") or "") for item in tasks],
                "headline_models_used": headline_models_used,
                "deltas": deltas,
                "n_metamorphic_variants": len(variants),
            }
        ),
        "duration_s": _elapsed(config, started),
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
    }


def _blocked_artifact(
    config: ExperimentConfig,
    started: float,
    precondition_checks: list[JsonDict],
    preconditions: PreconditionReport,
) -> JsonDict:
    empty = _metric_summary([], [])
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT_NAME,
        "run_date": RUN_DATE,
        "repair_rerun_clean": False,
        "headline_result": False,
        "preconditions_checked": True,
        "n_tasks": 0,
        "n_metamorphic_variants": 0,
        "model_specs": preconditions.model_specs,
        "headline_models_used": [],
        "model_checksums": preconditions.model_checksums,
        "pass_at_1_delta": 0.0,
        "pass_at_k_delta": 0.0,
        "false_accept_delta": 0.0,
        "tautology_gate_clean": False,
        "syntax_failure_rate_delta": 0.0,
        "schema_failure_rate_delta": 0.0,
        "live_transcript_paths": [],
        "verifier_log_paths": [],
        "honest_verdict": "blocked: exp3003 preconditions not met",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "baseline_metrics": empty,
        "repair_metrics": empty,
        "candidate_evaluations": [],
        "baseline_evaluations": [],
        "candidate_patch_paths": [],
        "transcript_paths": [],
        "selected_item_ids": [],
        "precondition_checks": precondition_checks,
        "source_artifacts": _source_artifacts(config),
        "exp2991_comparison": _exp2991_comparison(config),
        "smoke_only_candidate_count": 0,
        "candidate_manifest_sha256": _sha256_payload([]),
        "reproducibility_checksum": _sha256_payload({"blocked": True}),
        "duration_s": _elapsed(config, started),
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
    }


def _metric_summary(rows: Sequence[Mapping[str, Any]], tasks: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_item: dict[str, list[Mapping[str, Any]]] = {str(item.get("item_id") or ""): [] for item in tasks}
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


def _metric_deltas(baseline: Mapping[str, Any], repair: Mapping[str, Any]) -> JsonDict:
    return {
        "pass_at_1_delta": _delta(repair.get("pass_at_1"), baseline.get("pass_at_1")),
        "pass_at_k_delta": _delta(repair.get("pass_at_k"), baseline.get("pass_at_k")),
        "schema_failure_rate_delta": _delta(
            repair.get("schema_failure_rate"),
            baseline.get("schema_failure_rate"),
        ),
        "syntax_failure_rate_delta": _delta(
            repair.get("syntax_failure_rate"),
            baseline.get("syntax_failure_rate"),
        ),
        "false_accept_delta": _delta(repair.get("false_accept_rate"), baseline.get("false_accept_rate")),
    }


def _repair_rerun_clean(
    *,
    headline_result: bool,
    n_tasks: int,
    n_metamorphic_variants: int,
    headline_models_used: Sequence[str],
    smoke_only_candidate_count: int,
    deltas: Mapping[str, Any],
    tautology_gate_clean: bool,
) -> bool:
    return bool(
        headline_result
        and n_tasks >= MIN_HEADLINE_TASKS
        and n_metamorphic_variants > 0
        and any(model_id in HEADLINE_MODEL_IDS for model_id in headline_models_used)
        and not (smoke_only_candidate_count > 0 and not headline_models_used)
        and _positive(deltas.get("pass_at_1_delta"))
        and _nonnegative(deltas.get("pass_at_k_delta"))
        and _nonpositive(deltas.get("schema_failure_rate_delta"))
        and _nonpositive(deltas.get("syntax_failure_rate_delta"))
        and _nonpositive(deltas.get("false_accept_delta"))
        and tautology_gate_clean
    )


def _tautology_gate_clean(exp3002: Mapping[str, Any]) -> bool:
    rejected = exp3002.get("rejected_variants") or []
    return bool(
        exp3002.get("tautology_probe_ready") is True
        and any(row.get("reason") == "tautological_oracle_rejected" for row in rejected)
    )


def _exp2991_comparison(config: ExperimentConfig) -> JsonDict:
    payload = _read_json_if_present(config.resolved_exp2991_artifact_path())
    return {
        "source_artifact": EXP2991_FILENAME,
        "headline_result": bool(payload.get("headline_result")),
        "repair_rerun_clean": bool(payload.get("repair_rerun_clean")),
        "n_tasks": int(payload.get("n_tasks") or 0),
        "headline_models_used": list(payload.get("headline_models_used") or []),
        "pass_at_1_delta": float(payload.get("pass_at_1_delta") or 0.0),
        "pass_at_k_delta": float(payload.get("pass_at_k_delta") or 0.0),
        "schema_failure_rate_delta": float(payload.get("schema_failure_rate_delta") or 0.0),
        "syntax_failure_rate_delta": float(payload.get("syntax_failure_rate_delta") or 0.0),
        "false_accept_delta": float(
            payload.get("false_accept_delta") or payload.get("verifier_false_accept_delta") or 0.0
        ),
    }


def _source_artifacts(config: ExperimentConfig) -> list[JsonDict]:
    paths = [
        config.resolved_exp3001_artifact_path(),
        config.resolved_exp3002_artifact_path(),
        config.resolved_exp2991_artifact_path(),
        config.resolved_hard_manifest_path(),
    ]
    return [
        {
            "path": str(_relative_or_absolute(config.repo_root, path)),
            "present": path.is_file(),
            "sha256": _sha256_file(path) if path.is_file() else None,
        }
        for path in paths
    ]


def _candidate_syntax_success(source: Mapping[str, Any], patch_code: str) -> bool:
    if "syntax_success" in source:
        return bool(source.get("syntax_success"))
    try:
        compile(patch_code, "<repair-candidate>", "exec")
    except SyntaxError:
        return False
    return True


def _variants_for_tasks(
    variants: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    wanted = {str(item.get("item_id") or "") for item in tasks}
    return [dict(variant) for variant in variants if str(variant.get("source_item_id") or "") in wanted]


def _variants_by_source(variants: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = {}
    for variant in variants:
        out.setdefault(str(variant.get("source_item_id") or ""), []).append(variant)
    return out


def _checks_available(checks: Sequence[Mapping[str, Any]]) -> bool:
    return all(bool(row.get("available")) for row in checks)


def _checksum_available(evidence: Any) -> bool:
    return isinstance(evidence, Mapping) and evidence.get("status") == "available" and bool(
        evidence.get("sha256") or evidence.get("bounded_sha256")
    )


def _cuda_status(exp3001: Mapping[str, Any]) -> JsonDict:
    evidence = exp3001.get("precondition_evidence") if isinstance(exp3001, Mapping) else {}
    if isinstance(evidence, Mapping):
        gpu_inventory = evidence.get("gpu_inventory")
        if isinstance(gpu_inventory, Mapping) and "available" in gpu_inventory:
            return {"available": bool(gpu_inventory.get("available")), "source": "exp3001_gpu_inventory"}
        torch_cuda = evidence.get("torch_cuda")
        if isinstance(torch_cuda, Mapping) and "cuda_available" in torch_cuda:
            return {"available": bool(torch_cuda.get("cuda_available")), "source": "exp3001_torch_cuda"}
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {
            "available": completed.returncode == 0 and bool(completed.stdout.strip()),
            "source": "nvidia-smi",
            "returncode": completed.returncode,
            "stdout_summary": completed.stdout[:500],
            "stderr_summary": completed.stderr[:500],
        }
    except Exception as exc:  # pragma: no cover - defensive host diagnostics.
        return {"available": False, "source": "nvidia-smi", "error": f"{type(exc).__name__}: {exc}"}


def _call_cached_sota_pair() -> list[JsonDict] | None:
    try:
        result = cached_sota_pair(gpu_indices=(0, 1))
    except Exception:
        return None
    return [dict(row) for row in result] if result else None


def _read_json_if_present(path: Path) -> JsonDict:
    return dict(json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_repo_path(root: Path, value: Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else root / path


def _unique_paths(values: Sequence[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _relative_or_absolute(root: Path, path: Path) -> Path:
    try:
        return path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return path.resolve(strict=False)


def _safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_payload(payload: Any) -> str:
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-tasks", type=int, default=24)
    parser.add_argument("--test-run", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    args = _parse_args(argv)
    artifact = write_artifact(
        ExperimentConfig(
            output_path=args.output,
            n_tasks=args.n_tasks,
            tests_run=tuple(args.test_run),
        )
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["preconditions_checked"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_FILENAME",
    "EXP2991_FILENAME",
    "EXP3001_FILENAME",
    "EXP3002_FILENAME",
    "HEADLINE_MODEL_IDS",
    "SMOKE_ONLY_MODEL_IDS",
    "ExperimentConfig",
    "build_artifact",
    "main",
    "write_artifact",
]
