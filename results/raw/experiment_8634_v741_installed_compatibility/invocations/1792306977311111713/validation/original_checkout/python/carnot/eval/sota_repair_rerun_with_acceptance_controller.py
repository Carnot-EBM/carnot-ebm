"""Exp 3016 local SOTA repair rerun with an acceptance controller.

Spec: REQ-CODE-3016, SCENARIO-CODE-3016.

This module composes the live hard-set repair harness with the deterministic
Exp 3015 acceptance policy. The important distinction from Exp 3003 is that
every candidate is still replayed against original and metamorphic validators,
but only candidates accepted by the transparent controller can contribute to
the promoted repair metrics.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot.eval import hard_code_stress_manifest as hard
from carnot.eval import metamorphic_repair_oracle_audit as metamorphic
from carnot.eval import repair_acceptance_controller as controller
from carnot.eval import gated_sota_intent_preserving_repair_hard_set as repair
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
RepairGenerator = Callable[
    [JsonDict, str, int, int, JsonDict],
    repair.GenerationOutcome,
]
ClockFunc = Callable[[], float]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260524"
ARTIFACT_NAME = "experiment_3016_sota_repair_rerun_with_acceptance_controller_v1"
ARTIFACT_FILENAME = f"{ARTIFACT_NAME}.json"
SCHEMA = "carnot.sota_repair_rerun_with_acceptance_controller.v1"
EXP3002_FILENAME = metamorphic.ARTIFACT_FILENAME
EXP3003_FILENAME = "experiment_3003_gated_sota_repair_metamorphic_false_accept_rerun_v1.json"
EXP3013_FILENAME = "experiment_3013_sota_gguf_logprob_telemetry_preflight_v1.json"
EXP3015_FILENAME = controller.ARTIFACT_FILENAME
CONFIG_REL_PATH = controller.CONFIG_REL_PATH
TAXONOMY_TABLE_REL_PATH = controller.TAXONOMY_TABLE_REL_PATH
RAW_REL_DIR = Path("results/raw") / ARTIFACT_NAME
VERIFIER_REL_DIR = Path("results/verifier_transcripts/experiment_3016")
INFERENCE_SUBSTRATE = "live_sota_gguf_repair_with_acceptance_controller"

HEADLINE_MODEL_IDS: tuple[str, ...] = repair.HEADLINE_MODEL_IDS
SMOKE_ONLY_MODEL_IDS: tuple[str, ...] = repair.SMOKE_ONLY_MODEL_IDS
DEFAULT_N_TASKS = 24
DEFAULT_MAX_TOKENS = repair.DEFAULT_MAX_TOKENS
DEFAULT_RANDOM_SEED = 301600
MIN_HEADLINE_TASKS = 20
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "repair_controller_clean",
    "headline_result",
    "preconditions_checked",
    "n_tasks",
    "n_metamorphic_variants",
    "model_specs",
    "headline_models_used",
    "model_checksums",
    "acceptance_controller_config_path",
    "pass_at_1_delta",
    "pass_at_k_delta",
    "false_accept_delta",
    "tautology_gate_clean",
    "syntax_failure_rate_delta",
    "schema_failure_rate_delta",
    "live_transcript_paths",
    "verifier_log_paths",
    "inference_substrate",
    "duration_s",
    "honest_verdict",
)

GenerationOutcome = repair.GenerationOutcome


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and knobs for the Exp 3016 controller-gated rerun."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    hard_manifest_path: Path | None = None
    metamorphic_manifest_path: Path | None = None
    exp3002_artifact_path: Path | None = None
    exp3003_artifact_path: Path | None = None
    exp3013_artifact_path: Path | None = None
    exp3015_artifact_path: Path | None = None
    controller_config_path: Path | None = None
    raw_dir: Path | None = None
    verifier_dir: Path | None = None
    n_tasks: int = DEFAULT_N_TASKS
    max_tokens: int = DEFAULT_MAX_TOKENS
    random_seed: int = DEFAULT_RANDOM_SEED
    max_headline_models: int = 1
    tests_run: Sequence[str] = field(default_factory=tuple)
    started_at: float | None = None
    clock: ClockFunc = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / ARTIFACT_FILENAME

    def resolved_hard_manifest_path(self) -> Path:
        return self.hard_manifest_path or self.repo_root / hard.DEFAULT_MANIFEST_REL_PATH

    def resolved_exp3002_artifact_path(self) -> Path:
        return self.exp3002_artifact_path or self.repo_root / "results" / EXP3002_FILENAME

    def resolved_exp3003_artifact_path(self) -> Path:
        return self.exp3003_artifact_path or self.repo_root / "results" / EXP3003_FILENAME

    def resolved_exp3013_artifact_path(self) -> Path:
        return self.exp3013_artifact_path or self.repo_root / "results" / EXP3013_FILENAME

    def resolved_exp3015_artifact_path(self) -> Path:
        return self.exp3015_artifact_path or self.repo_root / "results" / EXP3015_FILENAME

    def resolved_controller_config_path(self, exp3015: Mapping[str, Any]) -> Path:
        if self.controller_config_path is not None:
            return self.controller_config_path
        return _resolve_repo_path(
            self.repo_root,
            exp3015.get("controller_config_path") or CONFIG_REL_PATH,
        )

    def resolved_metamorphic_manifest_path(self, exp3002: Mapping[str, Any]) -> Path:
        if self.metamorphic_manifest_path is not None:
            return self.metamorphic_manifest_path
        return _resolve_repo_path(
            self.repo_root,
            exp3002.get("metamorphic_manifest_path") or metamorphic.METAMORPHIC_MANIFEST_REL_PATH,
        )

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
    """Setup evidence required before Exp 3016 may evaluate repair promotion."""

    checks: list[JsonDict]
    model_specs: JsonDict
    model_checksums: JsonDict
    controller_config: JsonDict
    controller_config_path: Path
    exp3002_payload: JsonDict
    exp3003_payload: JsonDict
    exp3013_payload: JsonDict
    exp3015_payload: JsonDict
    runnable_model_specs: list[JsonDict]
    telemetry_by_model: dict[str, JsonDict]


def build_artifact(
    config: ExperimentConfig | None = None,
    *,
    generator: RepairGenerator | None = None,
) -> JsonDict:
    """Build the terminal Exp 3016 artifact."""

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
                "resource": "exp3016_sample_size",
                "available": False,
                "n_tasks": len(tasks),
                "n_metamorphic_variants": len(variants),
                "minimum_tasks": MIN_HEADLINE_TASKS,
            },
        ]
        return _blocked_artifact(config, started, checks, preconditions)

    model_specs = preconditions.runnable_model_specs[: max(1, config.max_headline_models)]
    baseline_rows = _baseline_rows(tasks, variants)
    candidate_rows: list[JsonDict] = []
    for model_spec in model_specs:
        live_generator = generator or repair.llama_cpp_repair_generator(model_spec)
        for task_index, item in enumerate(tasks):
            seed = config.random_seed + task_index
            prompt = repair.repair_prompt(item)
            generation = live_generator(item, prompt, seed, config.max_tokens, model_spec)
            candidate_rows.append(
                _candidate_row(
                    config=config,
                    item=item,
                    variants=variants,
                    prompt=prompt,
                    seed=seed,
                    model_spec=model_spec,
                    generation=generation,
                    controller_rule=preconditions.controller_config.get("selected_rule") or {},
                    tautology_gate_clean=_tautology_gate_clean(preconditions.exp3002_payload),
                    telemetry=preconditions.telemetry_by_model.get(str(model_spec.get("hf_id") or ""), {}),
                    candidate_index=len(candidate_rows),
                )
            )

    return _complete_artifact(
        config=config,
        started=started,
        preconditions=preconditions,
        checks=checks,
        tasks=tasks,
        variants=variants,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
    )


def write_artifact(
    config: ExperimentConfig | None = None,
    *,
    generator: RepairGenerator | None = None,
) -> JsonDict:
    """Build and persist the Exp 3016 terminal JSON artifact."""

    config = config or ExperimentConfig()
    artifact = build_artifact(config, generator=generator)
    path = config.artifact_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _precondition_report(config: ExperimentConfig) -> PreconditionReport:
    exp3013_path = config.resolved_exp3013_artifact_path()
    exp3015_path = config.resolved_exp3015_artifact_path()
    exp3002_path = config.resolved_exp3002_artifact_path()
    exp3003_path = config.resolved_exp3003_artifact_path()
    exp3013 = _read_json_if_present(exp3013_path)
    exp3015 = _read_json_if_present(exp3015_path)
    exp3002 = _read_json_if_present(exp3002_path)
    exp3003 = _read_json_if_present(exp3003_path)
    config_path = config.resolved_controller_config_path(exp3015)
    controller_config = _read_json_if_present(config_path)
    model_checksums = dict(exp3013.get("model_checksums") or {})
    runnable = _runnable_model_specs(exp3013)
    telemetry_by_model = _telemetry_by_model(exp3013)
    checks = [
        {
            "resource": "exp3013_sota_logprob_telemetry",
            "available": exp3013_path.is_file()
            and exp3013.get("sota_headline_ready") is True
            and exp3013.get("sota_logprob_ready") is True
            and exp3013.get("preconditions_checked") is True,
            "path": str(_relative_or_absolute(config.repo_root, exp3013_path)),
            "sha256": _sha256_file(exp3013_path) if exp3013_path.is_file() else None,
        },
        {
            "resource": "exp3015_acceptance_controller",
            "available": exp3015_path.is_file() and exp3015.get("acceptance_controller_ready") is True,
            "path": str(_relative_or_absolute(config.repo_root, exp3015_path)),
            "sha256": _sha256_file(exp3015_path) if exp3015_path.is_file() else None,
        },
        {
            "resource": "cuda_cache_status",
            "available": _cuda_cache_ready(exp3013),
            "evidence": exp3013.get("precondition_evidence") or {},
        },
        {
            "resource": "headline_model_checksum_available",
            "available": any(_checksum_available(model_checksums.get(hf_id)) for hf_id in HEADLINE_MODEL_IDS),
            "available_models": [
                hf_id for hf_id in HEADLINE_MODEL_IDS if _checksum_available(model_checksums.get(hf_id))
            ],
        },
        {
            "resource": "headline_model_cache_available",
            "available": bool(runnable),
            "available_models": [str(row.get("hf_id") or "") for row in runnable],
        },
        {
            "resource": "acceptance_controller_config_available",
            "available": config_path.is_file()
            and controller_config.get("policy_type") == "transparent_grid_rule"
            and isinstance(controller_config.get("selected_rule"), Mapping)
            and controller_config.get("llm_judge_used") is False,
            "path": str(_relative_or_absolute(config.repo_root, config_path)),
            "sha256": _sha256_file(config_path) if config_path.is_file() else None,
        },
    ]
    model_specs = {
        "headline_models": list(HEADLINE_MODEL_IDS),
        "smoke_only_models": list(SMOKE_ONLY_MODEL_IDS),
        "preferred_quantization": "Q4_K_M",
        "source_helpers": ["cached_sota_pair", "resolve_cached_gguf", "exp3013_cache_paths"],
        "runnable_headline_models": runnable,
        "cache_paths": (exp3013.get("cache_paths") or {}).get("headline_models") or {},
    }
    return PreconditionReport(
        checks=checks,
        model_specs=model_specs,
        model_checksums=model_checksums,
        controller_config=controller_config,
        controller_config_path=config_path,
        exp3002_payload=exp3002,
        exp3003_payload=exp3003,
        exp3013_payload=exp3013,
        exp3015_payload=exp3015,
        runnable_model_specs=runnable,
        telemetry_by_model=telemetry_by_model,
    )


def _runnable_model_specs(exp3013: Mapping[str, Any]) -> list[JsonDict]:
    out: list[JsonDict] = []
    for row in _call_cached_sota_pair() or []:
        if row.get("hf_id") in HEADLINE_MODEL_IDS and row.get("model_path"):
            out.append(dict(row))
    cache_paths = (exp3013.get("cache_paths") or {}).get("headline_models") or {}
    for hf_id in HEADLINE_MODEL_IDS:
        path = cache_paths.get(hf_id) or resolve_cached_gguf(hf_id)
        if path and hf_id not in {str(row.get("hf_id") or "") for row in out}:
            out.append(
                {
                    "name": hf_id.split("/", 1)[-1].removesuffix("-GGUF"),
                    "hf_id": hf_id,
                    "gpu": len(out) % 2,
                    "model_path": str(path),
                }
            )
    return out


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
            }
        ],
    }


def _validate_metamorphic_manifest(config: ExperimentConfig, exp3002: Mapping[str, Any]) -> JsonDict:
    path = config.resolved_metamorphic_manifest_path(exp3002)
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
    ready = bool(
        exp3002.get("metamorphic_oracle_ready") is True
        and variants
        and all(row.passed for row in reference)
    )
    return {
        "variants": variants,
        "checks": [
            {
                "resource": "metamorphic_manifest_integrity",
                "available": ready,
                "path": str(_relative_or_absolute(config.repo_root, path)),
                "sha256": _sha256_file(path),
                "n_metamorphic_variants": len(variants),
            }
        ],
    }


def _baseline_rows(
    tasks: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    variants_by_source = _variants_by_source(variants)
    rows = []
    for item in tasks:
        source_id = str(item.get("item_id") or "")
        original = hard.run_candidate_tests(item, "baseline_candidate")
        variant_outcomes = [
            hard.run_candidate_tests(variant, "baseline_candidate")
            for variant in variants_by_source.get(source_id, ())
        ]
        rows.append(
            _combined_outcome_row(
                item=item,
                candidate_code=str(item.get("baseline_candidate") or ""),
                original=original,
                variant_outcomes=variant_outcomes,
                schema_valid=True,
                syntax_success=True,
                entry_point_present=True,
                model_id="",
            )
        )
    return rows


def _candidate_row(
    *,
    config: ExperimentConfig,
    item: Mapping[str, Any],
    variants: Sequence[Mapping[str, Any]],
    prompt: str,
    seed: int,
    model_spec: Mapping[str, Any],
    generation: repair.GenerationOutcome,
    controller_rule: Mapping[str, Any],
    tautology_gate_clean: bool,
    telemetry: Mapping[str, Any],
    candidate_index: int,
) -> JsonDict:
    parsed = repair.parse_repair_output(generation.text)
    syntax_success, syntax_errors = repair.syntax_diagnostics(parsed.final_patch)
    variants_by_source = _variants_by_source(variants)
    original = hard.run_candidate_tests(
        {**dict(item), "repair_candidate": parsed.final_patch},
        "repair_candidate",
    )
    variant_outcomes = []
    for variant in variants_by_source.get(str(item.get("item_id") or ""), ()):
        adapted = metamorphic._adapt_candidate(
            parsed.final_patch,
            str(variant.get("source_entry_point") or item.get("entry_point") or ""),
            str(variant.get("entry_point") or ""),
        )
        variant_outcomes.append(
            hard.run_candidate_tests({**dict(variant), "repair_candidate": adapted}, "repair_candidate")
        )
    entry_point = str(item.get("entry_point") or "")
    row = _combined_outcome_row(
        item=item,
        candidate_code=parsed.final_patch,
        original=original,
        variant_outcomes=variant_outcomes,
        schema_valid=parsed.schema_valid,
        syntax_success=syntax_success,
        entry_point_present=_entry_point_present(parsed.final_patch, entry_point),
        model_id=str(model_spec.get("hf_id") or ""),
    )
    row.update(
        {
            "mode": "acceptance_controlled_headline_repair",
            "seed": seed,
            "prompt_sha256": _sha256_text(prompt),
            "draft_intent": parsed.draft_intent,
            "schema_errors": list(parsed.schema_errors),
            "syntax_errors": syntax_errors,
            "generation_backend": generation.backend,
            "generation_backend_detail": generation.backend_detail,
            "generation_duration_s": float(generation.duration_s),
            "tokens_generated": int(generation.tokens_generated),
            "generation_error": generation.error,
            "tautology_probe_clean": tautology_gate_clean,
            "intent_drift_class": "clean" if row["passed"] or row["false_accept"] else "intent_drift",
            "deterministic_tests_executed": True,
            "telemetry_summary": dict(telemetry),
        }
    )
    reasons = _controller_rejection_reasons(row, controller_rule)
    row["controller_rejection_reasons"] = reasons
    row["controller_accepted"] = not reasons
    _write_candidate_evidence(config, row, item, prompt, generation.text, parsed.final_patch, candidate_index)
    return row


def _combined_outcome_row(
    *,
    item: Mapping[str, Any],
    candidate_code: str,
    original: hard.VerificationOutcome,
    variant_outcomes: Sequence[hard.VerificationOutcome],
    schema_valid: bool,
    syntax_success: bool,
    entry_point_present: bool,
    model_id: str,
) -> JsonDict:
    variants_all_pass = bool(variant_outcomes) and all(row.passed for row in variant_outcomes)
    passed = bool(original.passed and variants_all_pass)
    false_accept = bool(original.passed and not variants_all_pass)
    return {
        "item_id": str(item.get("item_id") or ""),
        "entry_point": str(item.get("entry_point") or ""),
        "model_hf_id": model_id,
        "candidate_sha256": _sha256_text(candidate_code),
        "schema_valid": bool(schema_valid),
        "syntax_success": bool(syntax_success),
        "entry_point_present": bool(entry_point_present),
        "original_passed": bool(original.passed),
        "metamorphic_passed_all": variants_all_pass,
        "metamorphic_variant_count": len(variant_outcomes),
        "metamorphic_pass_count": sum(1 for row in variant_outcomes if row.passed),
        "passed": passed,
        "false_accept": false_accept,
        "original_verifier_output": original.as_dict(),
        "metamorphic_verifier_outputs": [row.as_dict() for row in variant_outcomes],
        "failing_assertions": list(original.failing_test_ids),
    }


def _write_candidate_evidence(
    config: ExperimentConfig,
    row: JsonDict,
    item: Mapping[str, Any],
    prompt: str,
    raw_response: str,
    final_patch: str,
    candidate_index: int,
) -> None:
    token = _safe_token(f"{row['item_id']}_{row['model_hf_id']}_{candidate_index}")
    patch_path = config.resolved_patch_dir() / f"{token}.py"
    transcript_path = config.resolved_transcript_dir() / f"{token}.json"
    verifier_path = config.resolved_verifier_dir() / f"{token}.json"
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    verifier_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(final_patch, encoding="utf-8")
    transcript = {
        "item_id": row["item_id"],
        "model_hf_id": row["model_hf_id"],
        "prompt": prompt,
        "prompt_sha256": row["prompt_sha256"],
        "raw_response": raw_response,
        "draft_intent": row["draft_intent"],
        "failing_trace": item.get("baseline_verification") or {},
        "final_patch_sha256": row["candidate_sha256"],
        "generation_duration_s": row["generation_duration_s"],
        "generation_backend": row["generation_backend"],
        "telemetry_summary": row["telemetry_summary"],
    }
    verifier = {
        "item_id": row["item_id"],
        "model_hf_id": row["model_hf_id"],
        "original_verifier_output": row["original_verifier_output"],
        "metamorphic_verifier_outputs": row["metamorphic_verifier_outputs"],
        "controller_accepted": row["controller_accepted"],
        "controller_rejection_reasons": row["controller_rejection_reasons"],
        "deterministic_tests_executed": row["deterministic_tests_executed"],
        "false_accept": row["false_accept"],
        "passed": row["passed"],
    }
    transcript_path.write_text(json.dumps(transcript, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verifier_path.write_text(json.dumps(verifier, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    row["candidate_patch_path"] = str(_relative_or_absolute(config.repo_root, patch_path))
    row["live_transcript_path"] = str(_relative_or_absolute(config.repo_root, transcript_path))
    row["transcript_sha256"] = _sha256_file(transcript_path)
    row["verifier_log_path"] = str(_relative_or_absolute(config.repo_root, verifier_path))


def _complete_artifact(
    *,
    config: ExperimentConfig,
    started: float,
    preconditions: PreconditionReport,
    checks: list[JsonDict],
    tasks: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
    baseline_rows: list[JsonDict],
    candidate_rows: list[JsonDict],
) -> JsonDict:
    accepted_rows = [row for row in candidate_rows if row.get("controller_accepted") is True]
    baseline = _metric_summary(baseline_rows, tasks)
    accept_all = _metric_summary(candidate_rows, tasks)
    repair_metrics = _metric_summary(accepted_rows, tasks)
    deltas = _metric_deltas(baseline, repair_metrics)
    headline_models_used = sorted(
        {
            str(row.get("model_hf_id") or "")
            for row in candidate_rows
            if row.get("model_hf_id") in HEADLINE_MODEL_IDS
        }
    )
    smoke_only_models_used = sorted(
        {
            str(row.get("model_hf_id") or "")
            for row in candidate_rows
            if row.get("model_hf_id") in SMOKE_ONLY_MODEL_IDS
        }
    )
    live_transcript_paths = _unique_paths(row.get("live_transcript_path") for row in candidate_rows)
    verifier_log_paths = _unique_paths(row.get("verifier_log_path") for row in candidate_rows)
    candidate_patch_paths = _unique_paths(row.get("candidate_patch_path") for row in candidate_rows)
    tautology_gate_clean = _tautology_gate_clean(preconditions.exp3002_payload)
    headline_result = bool(headline_models_used and live_transcript_paths)
    clean = _repair_controller_clean(
        headline_result=headline_result,
        n_tasks=len(tasks),
        n_metamorphic_variants=len(variants),
        headline_models_used=headline_models_used,
        smoke_only_models_used=smoke_only_models_used,
        accepted_count=len(accepted_rows),
        deltas=deltas,
        tautology_gate_clean=tautology_gate_clean,
    )
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT_NAME,
        "run_date": RUN_DATE,
        "repair_controller_clean": clean,
        "headline_result": headline_result,
        "preconditions_checked": True,
        "n_tasks": len(tasks),
        "n_metamorphic_variants": len(variants),
        "model_specs": preconditions.model_specs,
        "headline_models_used": headline_models_used,
        "model_checksums": preconditions.model_checksums,
        "acceptance_controller_config_path": str(
            _relative_or_absolute(config.repo_root, preconditions.controller_config_path)
        ),
        "pass_at_1_delta": deltas["pass_at_1_delta"],
        "pass_at_k_delta": deltas["pass_at_k_delta"],
        "false_accept_delta": deltas["false_accept_delta"],
        "tautology_gate_clean": tautology_gate_clean,
        "syntax_failure_rate_delta": deltas["syntax_failure_rate_delta"],
        "schema_failure_rate_delta": deltas["schema_failure_rate_delta"],
        "live_transcript_paths": live_transcript_paths,
        "verifier_log_paths": verifier_log_paths,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _elapsed(config, started),
        "honest_verdict": (
            "complete: acceptance-controlled SOTA repair rerun gates passed"
            if clean
            else "complete_flagged: acceptance-controlled SOTA repair rerun did not clear gates"
        ),
        "baseline_metrics": baseline,
        "accept_all_metrics": accept_all,
        "repair_metrics": repair_metrics,
        "candidate_evaluations": candidate_rows,
        "baseline_evaluations": baseline_rows,
        "candidate_patch_paths": candidate_patch_paths,
        "accepted_candidate_count": len(accepted_rows),
        "rejected_candidate_count": len(candidate_rows) - len(accepted_rows),
        "smoke_only_models_used": smoke_only_models_used,
        "acceptance_controller_rule": dict(preconditions.controller_config.get("selected_rule") or {}),
        "accept_all_comparison": _accept_all_comparison(repair_metrics, accept_all),
        "exp3003_comparison": _exp3003_comparison(preconditions.exp3003_payload, repair_metrics),
        "precondition_checks": checks,
        "source_artifacts": _source_artifacts(config, preconditions),
        "selected_item_ids": [str(item.get("item_id") or "") for item in tasks],
        "telemetry_summaries": [dict(row.get("telemetry_summary") or {}) for row in candidate_rows],
        "candidate_manifest_sha256": _sha256_payload(candidate_rows),
        "reproducibility_checksum": _sha256_payload(
            {
                "selected_item_ids": [str(item.get("item_id") or "") for item in tasks],
                "headline_models_used": headline_models_used,
                "accepted_candidate_count": len(accepted_rows),
                "deltas": deltas,
            }
        ),
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
        "repair_controller_clean": False,
        "headline_result": False,
        "preconditions_checked": True,
        "n_tasks": 0,
        "n_metamorphic_variants": 0,
        "model_specs": preconditions.model_specs,
        "headline_models_used": [],
        "model_checksums": preconditions.model_checksums,
        "acceptance_controller_config_path": str(
            _relative_or_absolute(config.repo_root, preconditions.controller_config_path)
        )
        if preconditions.controller_config_path
        else "",
        "pass_at_1_delta": 0.0,
        "pass_at_k_delta": 0.0,
        "false_accept_delta": 0.0,
        "tautology_gate_clean": False,
        "syntax_failure_rate_delta": 0.0,
        "schema_failure_rate_delta": 0.0,
        "live_transcript_paths": [],
        "verifier_log_paths": [],
        "inference_substrate": "blocked_preconditions",
        "duration_s": _elapsed(config, started),
        "honest_verdict": "blocked: exp3016 preconditions not met",
        "baseline_metrics": empty,
        "accept_all_metrics": empty,
        "repair_metrics": empty,
        "candidate_evaluations": [],
        "baseline_evaluations": [],
        "candidate_patch_paths": [],
        "accepted_candidate_count": 0,
        "rejected_candidate_count": 0,
        "smoke_only_models_used": [],
        "acceptance_controller_rule": dict(preconditions.controller_config.get("selected_rule") or {}),
        "accept_all_comparison": _accept_all_comparison(empty, empty),
        "exp3003_comparison": _exp3003_comparison(preconditions.exp3003_payload, empty),
        "precondition_checks": precondition_checks,
        "source_artifacts": _source_artifacts(config, preconditions),
        "selected_item_ids": [],
        "telemetry_summaries": [],
        "candidate_manifest_sha256": _sha256_payload([]),
        "reproducibility_checksum": _sha256_payload({"blocked": True}),
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
    }


def _metric_summary(
    rows: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
) -> JsonDict:
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


def _metric_deltas(baseline: Mapping[str, Any], repair_metrics: Mapping[str, Any]) -> JsonDict:
    return {
        "pass_at_1_delta": _delta(repair_metrics.get("pass_at_1"), baseline.get("pass_at_1")),
        "pass_at_k_delta": _delta(repair_metrics.get("pass_at_k"), baseline.get("pass_at_k")),
        "schema_failure_rate_delta": _delta(
            repair_metrics.get("schema_failure_rate"),
            baseline.get("schema_failure_rate"),
        ),
        "syntax_failure_rate_delta": _delta(
            repair_metrics.get("syntax_failure_rate"),
            baseline.get("syntax_failure_rate"),
        ),
        "false_accept_delta": _delta(
            repair_metrics.get("false_accept_rate"),
            baseline.get("false_accept_rate"),
        ),
    }


def _repair_controller_clean(
    *,
    headline_result: bool,
    n_tasks: int,
    n_metamorphic_variants: int,
    headline_models_used: Sequence[str],
    smoke_only_models_used: Sequence[str],
    accepted_count: int,
    deltas: Mapping[str, Any],
    tautology_gate_clean: bool,
) -> bool:
    return bool(
        headline_result
        and n_tasks >= MIN_HEADLINE_TASKS
        and n_metamorphic_variants > 0
        and any(model_id in HEADLINE_MODEL_IDS for model_id in headline_models_used)
        and not (smoke_only_models_used and not headline_models_used)
        and accepted_count > 0
        and _positive(deltas.get("pass_at_1_delta"))
        and _nonnegative(deltas.get("pass_at_k_delta"))
        and _nonpositive(deltas.get("schema_failure_rate_delta"))
        and _nonpositive(deltas.get("syntax_failure_rate_delta"))
        and _nonpositive(deltas.get("false_accept_delta"))
        and tautology_gate_clean
    )


def _controller_rejection_reasons(row: Mapping[str, Any], rule: Mapping[str, Any]) -> list[str]:
    checks = [
        ("require_schema_valid", "schema_valid", row.get("schema_valid") is True),
        ("require_syntax_success", "syntax_success", row.get("syntax_success") is True),
        ("require_entry_point_present", "entry_point_present", row.get("entry_point_present") is True),
        (
            "require_false_accept_probe_clean",
            "false_accept",
            row.get("false_accept") is False,
        ),
        ("require_no_intent_drift", "intent_drift", row.get("intent_drift_class") == "clean"),
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


def _accept_all_comparison(repair_metrics: Mapping[str, Any], accept_all: Mapping[str, Any]) -> JsonDict:
    return {
        "pass_at_1_delta_vs_accept_all": _delta(repair_metrics.get("pass_at_1"), accept_all.get("pass_at_1")),
        "pass_at_k_delta_vs_accept_all": _delta(repair_metrics.get("pass_at_k"), accept_all.get("pass_at_k")),
        "false_accept_delta_vs_accept_all": _delta(
            repair_metrics.get("false_accept_rate"),
            accept_all.get("false_accept_rate"),
        ),
        "syntax_failure_rate_delta_vs_accept_all": _delta(
            repair_metrics.get("syntax_failure_rate"),
            accept_all.get("syntax_failure_rate"),
        ),
        "schema_failure_rate_delta_vs_accept_all": _delta(
            repair_metrics.get("schema_failure_rate"),
            accept_all.get("schema_failure_rate"),
        ),
    }


def _exp3003_comparison(exp3003: Mapping[str, Any], repair_metrics: Mapping[str, Any]) -> JsonDict:
    return {
        "source_artifact": EXP3003_FILENAME,
        "present": bool(exp3003),
        "headline_result": bool(exp3003.get("headline_result")),
        "repair_rerun_clean": bool(exp3003.get("repair_rerun_clean")),
        "n_tasks": int(exp3003.get("n_tasks") or 0),
        "pass_at_1_delta": float(exp3003.get("pass_at_1_delta") or 0.0),
        "pass_at_k_delta": float(exp3003.get("pass_at_k_delta") or 0.0),
        "false_accept_delta": float(exp3003.get("false_accept_delta") or 0.0),
        "syntax_failure_rate_delta": float(exp3003.get("syntax_failure_rate_delta") or 0.0),
        "schema_failure_rate_delta": float(exp3003.get("schema_failure_rate_delta") or 0.0),
        "controller_pass_at_1_minus_exp3003_delta": _delta(
            repair_metrics.get("pass_at_1"),
            (exp3003.get("repair_metrics") or {}).get("pass_at_1", 0.0),
        ),
    }


def _source_artifacts(config: ExperimentConfig, preconditions: PreconditionReport) -> list[JsonDict]:
    paths = [
        config.resolved_exp3013_artifact_path(),
        config.resolved_exp3015_artifact_path(),
        preconditions.controller_config_path,
        config.resolved_exp3002_artifact_path(),
        config.resolved_metamorphic_manifest_path(preconditions.exp3002_payload),
        config.resolved_exp3003_artifact_path(),
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


def _tautology_gate_clean(exp3002: Mapping[str, Any]) -> bool:
    rejected = exp3002.get("rejected_variants") or []
    return bool(
        exp3002.get("tautology_probe_ready") is True
        and any(row.get("reason") == "tautological_oracle_rejected" for row in rejected)
    )


def _cuda_cache_ready(exp3013: Mapping[str, Any]) -> bool:
    evidence = exp3013.get("precondition_evidence") or {}
    gpu = evidence.get("gpu_inventory") if isinstance(evidence, Mapping) else {}
    torch_cuda = evidence.get("torch_cuda") if isinstance(evidence, Mapping) else {}
    llama_cpp = evidence.get("llama_cpp") if isinstance(evidence, Mapping) else {}
    return bool(
        (
            isinstance(gpu, Mapping)
            and gpu.get("available") is True
            or isinstance(torch_cuda, Mapping)
            and torch_cuda.get("cuda_available") is True
        )
        and isinstance(llama_cpp, Mapping)
        and llama_cpp.get("llama_cpp_supports_gpu_offload") is True
    )


def _checksum_available(evidence: Any) -> bool:
    return isinstance(evidence, Mapping) and evidence.get("status") == "available" and bool(
        evidence.get("sha256") or evidence.get("bounded_sha256")
    )


def _call_cached_sota_pair() -> list[JsonDict] | None:
    try:
        result = cached_sota_pair(gpu_indices=(0, 1))
    except Exception:
        return None
    return [dict(row) for row in result] if result else None


def _telemetry_by_model(exp3013: Mapping[str, Any]) -> dict[str, JsonDict]:
    out: dict[str, JsonDict] = {}
    for row in exp3013.get("headline_models_attempted") or []:
        model_id = str(row.get("hf_id") or "")
        if model_id:
            out[model_id] = {
                "transcript_path": row.get("transcript_path"),
                "transcript_sha256": row.get("transcript_sha256"),
                "telemetry_observation": row.get("telemetry_observation") or {},
                "token_logprobs_exposed": bool(row.get("token_logprobs_exposed")),
                "topk_logprobs_exposed": bool(row.get("topk_logprobs_exposed")),
                "preflight_duration_s": float(row.get("duration_s") or 0.0),
            }
    return out


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


def _entry_point_present(code: str, entry_point: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.FunctionDef) and node.name == entry_point for node in tree.body)


def _checks_available(checks: Sequence[Mapping[str, Any]]) -> bool:
    return all(bool(row.get("available")) for row in checks)


def _read_json_if_present(path: Path) -> JsonDict:
    return dict(json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_repo_path(root: Path, value: Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else root / path


def _relative_or_absolute(root: Path, path: Path) -> Path:
    try:
        return path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return path.resolve(strict=False)


def _unique_paths(values: Sequence[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


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
    parser.add_argument("--n-tasks", type=int, default=DEFAULT_N_TASKS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-headline-models", type=int, default=1)
    parser.add_argument("--test-run", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    args = _parse_args(argv)
    artifact = write_artifact(
        ExperimentConfig(
            output_path=args.output,
            n_tasks=args.n_tasks,
            max_tokens=args.max_tokens,
            max_headline_models=args.max_headline_models,
            tests_run=tuple(args.test_run),
        )
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if not artifact["honest_verdict"].startswith("blocked:") else 1


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_FILENAME",
    "CONFIG_REL_PATH",
    "EXP3002_FILENAME",
    "EXP3003_FILENAME",
    "EXP3013_FILENAME",
    "EXP3015_FILENAME",
    "GenerationOutcome",
    "ExperimentConfig",
    "HEADLINE_MODEL_IDS",
    "INFERENCE_SUBSTRATE",
    "REQUIRED_ARTIFACT_FIELDS",
    "SMOKE_ONLY_MODEL_IDS",
    "build_artifact",
    "main",
    "write_artifact",
]
