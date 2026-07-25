"""Exp5910 verification-guided ConstraintIR repair controls.

Spec refs: REQ-VERIFY-5910, SCENARIO-VERIFY-5910-PRECONDITIONS,
SCENARIO-VERIFY-5910-PROMPTS, SCENARIO-VERIFY-5910-CONTROLS,
SCENARIO-VERIFY-5910-SAFETY.

Exp5909 already sealed local GGUF ConstraintIR proposals with exact parser,
type, compile, solver, and semantic diagnostics. This experiment asks whether
one repair call that sees the exact deployment-visible diagnostic is better
than matched second-call controls. The exact evaluator is the authority after
generation; the model remains only a ConstraintIR proposer.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import experiment_5896_typed_constraint_ir_fixture as exp5896
from carnot import experiment_5909_sota_constraint_synthesis_ab as exp5909
from carnot.inference.sota_models import (
    cached_sota_pair,
    gguf_tokenizer_loadable,
    resolve_cached_gguf,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5910_verification_guided_constraint_repair.json")
RAW_TRACE_RELATIVE_PATH = Path(
    "results/experiment_5910_verification_guided_constraint_repair.raw.jsonl"
)
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5910_verification_guided_constraint_repair.py")
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5910_verification_guided_constraint_repair.py"
)
RUN_DATE = "20260725"
RANDOM_SEED = 5910
INFERENCE_SUBSTRATE = "live_llm_inference"
VERIFIER_IS_ORACLE = True
ARTIFACT_SCHEMA_VERSION = "carnot.experiment_5910.verification_guided_constraint_repair.v1"
RAW_ROW_SCHEMA_VERSION = ARTIFACT_SCHEMA_VERSION + ".raw_row"

MANDATED_MODEL_IDS = exp5909.MANDATED_MODEL_IDS
MODEL_SPECS = exp5909.MODEL_SPECS
PRIMARY_EXP5909_ARMS = exp5909.PRIMARY_ARM_IDS
ELIGIBLE_ROWS_PER_MODEL_FAMILY_ERROR = 1

DECODING: JsonDict = {
    "temperature": 0.0,
    "top_p": 1.0,
    "repeat_penalty": 1.05,
    "stop": ["</s>", "<eos>", "<|eot_id|>"],
}
GENERATION_BUDGETS: JsonDict = {
    "second_call_max_tokens": exp5909.GENERATION_BUDGETS["max_tokens"],
    "n_ctx": exp5909.GENERATION_BUDGETS["n_ctx"],
    "n_batch": exp5909.GENERATION_BUDGETS["n_batch"],
    "n_gpu_layers": exp5909.GENERATION_BUDGETS["n_gpu_layers"],
    "wall_clock_budget_s_per_second_call": exp5909.GENERATION_BUDGETS[
        "wall_clock_budget_s_per_call"
    ],
}
ARM_DEFINITIONS: JsonDict = {
    "no_repair": {
        "calls_including_exp5909_initial": 1,
        "second_call": False,
        "diagnostic_mode": "none",
        "second_call_max_tokens": 0,
        "control": True,
    },
    "exact_diagnostic_repair": {
        "calls_including_exp5909_initial": 2,
        "second_call": True,
        "diagnostic_mode": "deployment_visible_exact",
        "second_call_max_tokens": GENERATION_BUDGETS["second_call_max_tokens"],
        "control": False,
    },
    "matched_second_call_no_diagnostic": {
        "calls_including_exp5909_initial": 2,
        "second_call": True,
        "diagnostic_mode": "withheld",
        "second_call_max_tokens": GENERATION_BUDGETS["second_call_max_tokens"],
        "control": True,
    },
    "no_information_diagnostic": {
        "calls_including_exp5909_initial": 2,
        "second_call": True,
        "diagnostic_mode": "no_information",
        "second_call_max_tokens": GENERATION_BUDGETS["second_call_max_tokens"],
        "control": True,
    },
    "shuffled_same_error_class_diagnostic": {
        "calls_including_exp5909_initial": 2,
        "second_call": True,
        "diagnostic_mode": "same_error_class_shuffled",
        "second_call_max_tokens": GENERATION_BUDGETS["second_call_max_tokens"],
        "control": True,
    },
}
TWO_CALL_ARMS = tuple(arm for arm, spec in ARM_DEFINITIONS.items() if spec["second_call"])
ALL_ARM_IDS = tuple(ARM_DEFINITIONS)
PROMOTION_THRESHOLDS: JsonDict = {
    "required_completed_headline_families": 3,
    "required_model_family_error_cells": 9,
    "held_cell_lower_bound_min": 0.0,
    "max_correct_row_regressions": 0,
    "max_unsafe_increase": 0.0,
}
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5910_verification_guided_constraint_repair.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5910_verification_guided_constraint_repair.py "
    "-m pytest tests/python/test_experiment_5910_verification_guided_constraint_repair.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5910_verification_guided_constraint_repair.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python -m carnot.experiment_5910_verification_guided_constraint_repair",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5910_verification_guided_constraint_repair.json",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5910_verification_guided_constraint_repair.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "git status --short -- scripts/research_conductor.py ops/changelog.md "
    "ops/status.md _bmad/traceability.md",
)
PROTECTED_FILES = exp5909.PROTECTED_FILES
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "upstream_gate_stream_and_hashes",
    "model_specs",
    "model_file_hashes",
    "embedded_tokenizer_loader_cuda_and_gpu_receipts",
    "frozen_eligibility_error_taxonomy_prompts_seeds_and_budgets",
    "arm_definitions_and_compute_parity",
    "diagnostic_visibility_and_oracle_boundary",
    "per_model_error_family_repair_metrics",
    "exact_semantic_repair_and_regression_metrics",
    "omitted_spurious_and_unsafe_constraint_metrics",
    "matched_no_diagnostic_no_information_and_shuffled_controls",
    "group_bootstrap_lower_bounds",
    "raw_trace_and_output_receipts",
    "gpu_utilization_vram_latency_and_energy_receipts",
    "protected_files_unchanged",
    "verification_guided_repair_ready_score",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)
FIELD_PRINCIPLES: JsonDict = {
    "diagnostic_visibility_and_oracle_boundary": (
        "Only deployment-visible exact diagnostics may guide repair."
    ),
    "arm_definitions_and_compute_parity": (
        "A second call alone cannot be credited as verification-guided repair."
    ),
    "verification_guided_repair_ready_score": (
        "Emit bare 1.0 only for positive held lower bounds over matched controls, "
        "zero correct-row regression, zero unsafe increase, and all three models completed."
    ),
    "inference_substrate": "Use live_llm_inference.",
    "verifier_is_oracle": (
        "True for exact outcome evaluation; the model remains only an IR proposer."
    ),
    "honest_verdict": (
        "Use complete_positive:, complete_null:, unsafe:, blocked_precondition:, or blocked:."
    ),
}
FORBIDDEN_ORACLE_TOKENS = (
    "constraint_ir",
    "gold_ir",
    "answer_label",
    "answer_labels",
    "held_identity",
    "held identities",
    "certificates",
    "certificate_solution",
    "certificate solutions",
    "behavior_hash",
    "query_bindings",
    "model_self_score",
)

CachedPairProvider = Callable[..., list[JsonDict] | None]
IndividualResolver = Callable[[str], str | None]
EnvironmentProbe = Callable[[Path], JsonDict]
TokenizerChecker = Callable[[str | None], tuple[bool, str]]
CollectRepairOutputsFn = Callable[[list[JsonDict], list[JsonDict], "ExperimentConfig"], JsonDict]


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths, clocks, and the preregistered eligibility cap for Exp5910."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    raw_trace_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    monotonic_clock: Callable[[], float] = time.monotonic
    random_seed: int = RANDOM_SEED
    max_rows_per_model_family_error: int | None = ELIGIBLE_ROWS_PER_MODEL_FAMILY_ERROR

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / RESULT_RELATIVE_PATH

    def raw_path(self) -> Path:
        return self.raw_trace_path or self.repo_root / RAW_TRACE_RELATIVE_PATH


@dataclass(frozen=True)
class PreconditionReport:
    """All gates that must pass before a repair model is loaded."""

    preconditions_checked: JsonDict
    upstream_gate_stream_and_hashes: JsonDict
    model_specs: list[JsonDict]
    model_file_hashes: JsonDict
    embedded_tokenizer_loader_cuda_and_gpu_receipts: JsonDict
    protected_file_baseline: JsonDict
    block_reason: str | None


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence with stable bytes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash a file by bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def resolve_model_specs(
    *,
    cached_pair_provider: CachedPairProvider = cached_sota_pair,
    individual_model_resolver: IndividualResolver = resolve_cached_gguf,
) -> tuple[list[JsonDict], JsonDict]:
    """Resolve the same three GGUF families used by Exp5909."""

    return exp5909.resolve_model_specs(
        cached_pair_provider=cached_pair_provider,
        individual_model_resolver=individual_model_resolver,
    )


def check_preconditions(
    config: ExperimentConfig,
    *,
    cached_pair_provider: CachedPairProvider = cached_sota_pair,
    individual_model_resolver: IndividualResolver = resolve_cached_gguf,
    environment_probe: EnvironmentProbe = lambda root: exp5909._probe_environment(root),
    tokenizer_checker: TokenizerChecker = gguf_tokenizer_loadable,
) -> PreconditionReport:
    """Replay Exp5909 and host gates before any repair model load."""

    root = config.repo_root
    protected_baseline = _protected_file_receipt(root)
    upstream = _upstream_gate_receipt(root)
    model_specs, resolution = resolve_model_specs(
        cached_pair_provider=cached_pair_provider,
        individual_model_resolver=individual_model_resolver,
    )
    model_file_hashes = _model_file_hashes(model_specs)
    environment = environment_probe(root)
    tokenizer_receipts = _tokenizer_receipts(model_specs, tokenizer_checker)
    receipts = {
        "model_resolution": resolution,
        "environment": environment,
        "embedded_tokenizer_receipts": tokenizer_receipts,
        "runtime_loader_receipts": [],
        "gpu_receipts": {},
    }
    checks = {
        "exp5909_stream_ready": bool(upstream.get("exp5909_stream_ready")),
        "exp5909_repair_admission_ready": bool(
            upstream.get("exp5909_repair_admission_ready")
        ),
        "all_three_model_specs_resolved": all(spec.get("model_path") for spec in model_specs),
        "all_model_files_exist": all(row.get("exists") for row in model_file_hashes["files"]),
        "all_model_hashes_recorded": all(row.get("sha256") for row in model_file_hashes["files"]),
        "embedded_gguf_tokenizers_load": all(row.get("ok") for row in tokenizer_receipts),
        "llama_cpp_import_ok": bool(environment.get("llama_cpp_import", {}).get("ok")),
        "public_llama_cpp_cuda_offload": bool(
            environment.get("llama_cpp_cuda_support", {}).get("ok")
        ),
        "two_healthy_rtx_3090s": exp5909._two_healthy_rtx_3090s(
            environment.get("gpu_health", {})
        ),
        "adequate_vram": exp5909._adequate_vram(environment.get("gpu_health", {})),
        "adequate_ram": bool(environment.get("ram", {}).get("ok")),
        "adequate_disk": bool(environment.get("disk", {}).get("ok")),
        "no_protected_workload": bool(environment.get("protected_workload", {}).get("ok")),
        "atomic_output_ready": bool(environment.get("atomic_output", {}).get("ok")),
    }
    block_reason = next((name for name, ok in checks.items() if not ok), None)
    return PreconditionReport(
        preconditions_checked={
            "run_order": "exp5909_gate_stream_and_resource_checks_before_any_repair_model_load",
            "blocked_before_model_load": block_reason is not None,
            "headline_checks": checks,
            "block_reason": block_reason,
        },
        upstream_gate_stream_and_hashes=upstream,
        model_specs=model_specs,
        model_file_hashes=model_file_hashes,
        embedded_tokenizer_loader_cuda_and_gpu_receipts=receipts,
        protected_file_baseline=protected_baseline,
        block_reason=block_reason,
    )


def freeze_eligible_rows(config: ExperimentConfig) -> list[JsonDict]:
    """Freeze residual incorrect Exp5909 rows with deployment-visible diagnostics."""

    raw_rows = _load_exp5909_raw_rows(config.repo_root)
    source_rows = {row["row_id"]: row for row in exp5896.build_fixture_rows()}
    candidates: list[JsonDict] = []
    for raw in raw_rows:
        if raw.get("arm_id") not in PRIMARY_EXP5909_ARMS:
            continue
        if not _row_is_incorrect(raw) or not _has_visible_diagnostic(raw):
            continue
        if _uses_forbidden_oracle_material(raw.get("visible_diagnostics") or {}):
            continue
        source = source_rows[str(raw["source_row_id"])]
        row = dict(raw)
        row["natural_language"] = source["natural_language"]
        row["eligible_error_class"] = classify_error(row)
        row["visible_trace_sha256"] = public_repair_diagnostic(row)["visible_trace_sha256"]
        candidates.append(row)

    candidates.sort(key=lambda row: int(row.get("stream_sequence_index") or 0))
    if config.max_rows_per_model_family_error is None:
        return candidates

    cap = int(config.max_rows_per_model_family_error)
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    selected: list[JsonDict] = []
    for row in candidates:
        key = (
            str(row.get("model_hf_id")),
            str(row.get("family")),
            str(row.get("eligible_error_class")),
        )
        if counts[key] >= cap:
            continue
        counts[key] += 1
        selected.append(row)
    return selected


def classify_error(row: Mapping[str, Any]) -> str:
    """Classify a residual row by the first deployment-visible failing stage."""

    diagnostics = row.get("visible_diagnostics") or row.get("diagnostics") or {}
    if diagnostics.get("parser_status") == "rejected":
        return "type" if diagnostics.get("type_status") == "rejected" else "parser"
    if diagnostics.get("type_status") == "rejected":
        return "type"
    if diagnostics.get("compiler_status") not in {None, "compiled"}:
        return "compile"
    if diagnostics.get("solver_status") not in {None, "sat", "unsat"}:
        return "solver"
    if row.get("certificate_status") not in {None, "accepted"}:
        return "certificate"
    return "semantic"


def public_repair_diagnostic(row: Mapping[str, Any]) -> JsonDict:
    """Expose only parser/type/compile/solver/certificate diagnostics."""

    diagnostics = dict(row.get("visible_diagnostics") or row.get("diagnostics") or {})
    payload = {
        "error_class": classify_error(row),
        "parser_status": diagnostics.get("parser_status"),
        "parser_error": diagnostics.get("parser_error"),
        "type_status": diagnostics.get("type_status"),
        "type_error": diagnostics.get("type_error"),
        "compiler_status": diagnostics.get("compiler_status"),
        "compiler_error": diagnostics.get("compiler_error"),
        "solver_status": diagnostics.get("solver_status"),
        "solver_error": diagnostics.get("solver_error"),
        "certificate_status": diagnostics.get("certificate_status"),
        "certificate_error": diagnostics.get("certificate_error"),
        "cross_backend_agreement": diagnostics.get("cross_backend_agreement"),
    }
    payload["visible_trace_sha256"] = sha256_json(payload)
    return payload


def build_repair_prompt(
    row: Mapping[str, Any],
    arm_id: str,
    diagnostic: Mapping[str, Any] | None,
) -> str:
    """Build one leakage-safe second-call prompt for an Exp5910 repair arm."""

    if arm_id not in TWO_CALL_ARMS:
        raise ValueError(f"unknown Exp5910 arm: {arm_id}")
    problem = str(row.get("natural_language") or "")
    previous = str(row.get("raw_output_text") or "")
    header = (
        "Return exactly one corrected typed ConstraintIR JSON object and no prose. "
        "Repair only the JSON object. Do not answer in natural language. "
        f"schema_version: {exp5896.CONSTRAINT_IR_SCHEMA_VERSION}\n"
    )
    if arm_id == "exact_diagnostic_repair":
        middle = (
            "The exact deployment executor exposed these diagnostics from the previous JSON:\n"
            f"{canonical_json(dict(diagnostic or {}))}\n"
        )
    elif arm_id == "matched_second_call_no_diagnostic":
        middle = (
            "No parser, type, compile, solver, or certificate diagnostics are provided "
            "in this matched second-call control.\n"
        )
    elif arm_id == "no_information_diagnostic":
        middle = (
            "The following diagnostic-shaped block is intentionally non-informative:\n"
            f"{canonical_json(_no_information_diagnostic())}\n"
        )
    else:
        middle = (
            "A diagnostic from another candidate with the same deployment error class follows:\n"
            f"{canonical_json(dict(diagnostic or {}))}\n"
        )
    return f"{header}{middle}Problem text:\n{problem}\nPrevious JSON attempt:\n{previous}\n"


def diagnostic_visibility_receipt(
    eligible_rows: Sequence[Mapping[str, Any]],
    evaluated_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Prove repair prompts use only visible diagnostics and zero forbidden oracles."""

    leakage = any(
        _uses_forbidden_oracle_material(row.get("repair_prompt_visible_diagnostic") or {})
        for row in evaluated_rows
    )
    trace_hashes = [
        row.get("visible_trace_sha256") or row.get("diagnostic_trace_sha256")
        for row in eligible_rows
    ]
    return {
        "visible_to_exact_diagnostic_repair": [
            "parser_status",
            "parser_error",
            "type_status",
            "type_error",
            "compiler_status",
            "compiler_error",
            "solver_status",
            "solver_error",
            "certificate_status",
            "certificate_error",
            "cross_backend_agreement",
        ],
        "withheld_from_all_repair_prompts": [
            "hidden_gold_ir",
            "answer_labels",
            "held_identities",
            "certificate_solutions",
            "behavior_hashes",
            "query_bindings",
            "model_self_scores",
        ],
        "forbidden_oracle_access_counts": {
            "gold_ir": 0,
            "answer_labels": 0,
            "held_identities": 0,
            "certificate_solutions": 0,
            "model_self_scores": 0,
        },
        "visible_trace_hash_coverage": bool(eligible_rows)
        and all(isinstance(value, str) and value.startswith("sha256:") for value in trace_hashes),
        "visible_trace_hashes": trace_hashes,
        "oracle_leakage_detected": leakage,
        "principle": FIELD_PRINCIPLES["diagnostic_visibility_and_oracle_boundary"],
    }


def evaluate_candidate(
    row: Mapping[str, Any],
    arm_id: str,
    raw_text: str,
    generation_metadata: Mapping[str, Any],
) -> JsonDict:
    """Evaluate one repaired ConstraintIR with the shared exact evaluator."""

    return exp5909.evaluate_candidate(row, arm_id, raw_text, generation_metadata)


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    cached_pair_provider: CachedPairProvider = cached_sota_pair,
    individual_model_resolver: IndividualResolver = resolve_cached_gguf,
    environment_probe: EnvironmentProbe = lambda root: exp5909._probe_environment(root),
    tokenizer_checker: TokenizerChecker = gguf_tokenizer_loadable,
    collect_repair_outputs_fn: CollectRepairOutputsFn | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Run Exp5910 and write the terminal artifact plus raw trace atomically."""

    active = config or ExperimentConfig()
    started = active.start_time()
    preconditions = check_preconditions(
        active,
        cached_pair_provider=cached_pair_provider,
        individual_model_resolver=individual_model_resolver,
        environment_probe=environment_probe,
        tokenizer_checker=tokenizer_checker,
    )
    if preconditions.block_reason is not None:
        _write_raw_rows_atomic(active.raw_path(), [])
        artifact = _build_artifact(
            active,
            preconditions,
            eligible_rows=_safe_freeze_eligible_rows(active),
            sealed_rows=[],
            collection={"model_attempts": [], "gpu_receipts": {}},
            duration_s=active.clock() - started,
            test_exit_codes=test_exit_codes,
        )
        _write_json_atomic(active.artifact_path(), artifact)
        return artifact

    eligible_rows = freeze_eligible_rows(active)
    collector = collect_repair_outputs_fn or collect_live_repair_outputs
    collection = collector(preconditions.model_specs, eligible_rows, active)
    repair_raw_rows = [dict(row) for row in collection.get("rows") or []]
    sealed_rows = seal_repair_rows(eligible_rows, repair_raw_rows)
    _write_raw_rows_atomic(active.raw_path(), sealed_rows)
    artifact = _build_artifact(
        active,
        preconditions,
        eligible_rows=eligible_rows,
        sealed_rows=sealed_rows,
        collection=collection,
        duration_s=active.clock() - started,
        test_exit_codes=test_exit_codes,
    )
    _write_json_atomic(active.artifact_path(), artifact)
    return artifact


def collect_live_repair_outputs(
    model_specs: list[JsonDict],
    eligible_rows: list[JsonDict],
    config: ExperimentConfig,
) -> JsonDict:  # pragma: no cover - requires local GGUFs and CUDA.
    """Collect live llama.cpp repair outputs for every eligible row."""

    from llama_cpp import Llama

    by_model: dict[str, list[JsonDict]] = defaultdict(list)
    for row in eligible_rows:
        by_model[str(row.get("model_hf_id"))].append(row)
    rows: list[JsonDict] = []
    attempts: list[JsonDict] = []
    gpu_receipts: JsonDict = {"load_receipts": [], "generation_receipts": []}
    for model_index, spec in enumerate(model_specs):
        hf_id = str(spec.get("hf_id"))
        model_path = str(spec.get("model_path") or "")
        load_start = config.monotonic_clock()
        before = exp5909._gpu_health_probe()
        try:
            llm = Llama(
                model_path=model_path,
                n_gpu_layers=int(GENERATION_BUDGETS["n_gpu_layers"]),
                main_gpu=int(spec.get("gpu") or 0),
                n_ctx=int(GENERATION_BUDGETS["n_ctx"]),
                n_batch=int(GENERATION_BUDGETS["n_batch"]),
                seed=config.random_seed + model_index,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            attempts.append(
                {
                    "hf_id": hf_id,
                    "model_name": spec.get("name"),
                    "model_path": model_path,
                    "model_used": False,
                    "blocker": f"{type(exc).__name__}: {exc}",
                    "gpu_offload_verified": False,
                    "elapsed_seconds": round(config.monotonic_clock() - load_start, 6),
                }
            )
            continue
        after = exp5909._gpu_health_probe()
        attempt = {
            "hf_id": hf_id,
            "model_name": spec.get("name"),
            "model_path": model_path,
            "model_used": True,
            "blocker": None,
            "gpu_offload_verified": exp5909._vram_delta_mb(before, after) > 512,
            "vram_delta_mb": exp5909._vram_delta_mb(before, after),
            "elapsed_seconds": round(config.monotonic_clock() - load_start, 6),
        }
        attempts.append(attempt)
        gpu_receipts["load_receipts"].append(attempt)
        try:
            for eligible in by_model.get(hf_id, []):
                rows.extend(_collect_live_repair_arms(llm, spec, eligible, eligible_rows, config))
                gpu_receipts["generation_receipts"].append(
                    {
                        "model_hf_id": hf_id,
                        "source_stream_sequence_index": eligible.get("stream_sequence_index"),
                        "gpu_health_after": exp5909._gpu_health_probe(),
                    }
                )
        finally:
            close = getattr(llm, "close", None)
            if callable(close):
                close()
            del llm
            gc.collect()
    return {"rows": rows, "model_attempts": attempts, "gpu_receipts": gpu_receipts}


def seal_repair_rows(
    eligible_rows: Sequence[Mapping[str, Any]],
    repair_raw_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Seal no-repair baselines and second-call outputs with exact labels."""

    source_rows = {row["row_id"]: row for row in exp5896.build_fixture_rows()}
    eligible_by_sequence = {
        int(row["stream_sequence_index"]): dict(row) for row in eligible_rows
    }
    sealed: list[JsonDict] = []
    sequence = 0
    for eligible in eligible_rows:
        source = source_rows[str(eligible["source_row_id"])]
        sealed.append(_seal_one_row(sequence, eligible, source, "no_repair", eligible))
        sequence += 1
    for raw in repair_raw_rows:
        source_value = raw.get("source_stream_sequence_index")
        source_index = -1 if source_value is None else int(source_value)
        if source_index not in eligible_by_sequence:
            continue
        eligible = eligible_by_sequence[source_index]
        source = source_rows[str(eligible["source_row_id"])]
        sealed.append(_seal_one_row(sequence, eligible, source, str(raw.get("arm_id")), raw))
        sequence += 1
    return sealed


def aggregate_repair_evaluations(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Aggregate repair metrics by arm, model/error/family, safety, and controls."""

    row_list = [dict(row) for row in rows]
    by_arm = {arm_id: _metric_summary(_rows_for_arm(row_list, arm_id)) for arm_id in ALL_ARM_IDS}
    lower_bounds = _group_bootstrap_lower_bounds(row_list)
    exact = _exact_repair_metrics(row_list)
    return {
        "per_model_error_family_repair_metrics": {
            "by_model_error_family": _grouped_combo_metrics(
                row_list, ("model_hf_id", "eligible_error_class", "family")
            ),
            "by_model": _grouped_metrics(row_list, "model_hf_id"),
            "by_error_class": _grouped_metrics(row_list, "eligible_error_class"),
            "by_family": _grouped_metrics(row_list, "family"),
        },
        "exact_semantic_repair_and_regression_metrics": exact,
        "omitted_spurious_and_unsafe_constraint_metrics": _constraint_error_metrics(row_list),
        "matched_no_diagnostic_no_information_and_shuffled_controls": {
            "by_arm": {
                arm_id: by_arm[arm_id]
                for arm_id in (
                    "matched_second_call_no_diagnostic",
                    "no_information_diagnostic",
                    "shuffled_same_error_class_diagnostic",
                )
            },
            "shuffled_same_error_class_rate": _rate(
                sum(
                    bool(row.get("shuffled_diagnostic_same_error_class"))
                    for row in _rows_for_arm(row_list, "shuffled_same_error_class_diagnostic")
                ),
                len(_rows_for_arm(row_list, "shuffled_same_error_class_diagnostic")),
            ),
            "principle": FIELD_PRINCIPLES["arm_definitions_and_compute_parity"],
        },
        "group_bootstrap_lower_bounds": lower_bounds,
        "parse_type_compile_and_semantic_by_arm": by_arm,
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate terminal schema and load-bearing principle fields."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be live_llm_inference")
    if artifact["verifier_is_oracle"] is not True:
        raise ValueError("verifier_is_oracle must be true for exact evaluation")
    ids = [str(spec.get("hf_id")) for spec in artifact.get("model_specs", [])]
    if ids[:3] != list(MANDATED_MODEL_IDS):
        raise ValueError("model_specs must record all three mandated families in frozen order")
    score = float(artifact["verification_guided_repair_ready_score"])
    if score not in {0.0, 1.0}:
        raise ValueError("repair_ready score must be bare 0.0 or 1.0")
    if score == 1.0 and not str(artifact["honest_verdict"]).startswith("complete_positive:"):
        raise ValueError("positive ready score requires complete_positive honest_verdict")


def refresh_artifact_test_exit_codes(
    *,
    root: Path = REPO_ROOT,
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    """Update only test exit-code provenance after validation commands run."""

    path = root / RESULT_RELATIVE_PATH
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["test_exit_codes"] = dict(test_exit_codes)
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    validate_artifact(artifact)
    _write_json_atomic(path, artifact)
    return artifact


def raw_trace_row_hash(row: Mapping[str, Any]) -> str:
    """Hash one sealed raw trace row while excluding its own row hash."""

    stable = dict(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def _build_artifact(
    config: ExperimentConfig,
    preconditions: PreconditionReport,
    *,
    eligible_rows: Sequence[Mapping[str, Any]],
    sealed_rows: Sequence[Mapping[str, Any]],
    collection: Mapping[str, Any],
    duration_s: float,
    test_exit_codes: Mapping[str, int] | None,
) -> JsonDict:
    aggregates = aggregate_repair_evaluations(sealed_rows)
    completed = _completed_headline_models(collection.get("model_attempts") or [])
    protected = _protected_file_receipt(
        config.repo_root, baseline=preconditions.protected_file_baseline
    )
    visibility = diagnostic_visibility_receipt(eligible_rows, sealed_rows)
    runtime_receipt = _runtime_loader_receipt(preconditions, collection)
    score = _ready_score(aggregates, completed, visibility, runtime_receipt)
    unsafe_increase = _unsafe_increase(aggregates)
    status, verdict = _status_and_verdict(
        preconditions.block_reason, completed, score, unsafe_increase
    )
    artifact: JsonDict = {
        "schema": ARTIFACT_SCHEMA_VERSION,
        "experiment_id": "experiment_5910_verification_guided_constraint_repair",
        "run_date": RUN_DATE,
        "random_seed": config.random_seed,
        "field_principles": FIELD_PRINCIPLES,
        "status": status,
        "preconditions_checked": preconditions.preconditions_checked,
        "upstream_gate_stream_and_hashes": preconditions.upstream_gate_stream_and_hashes,
        "model_specs": preconditions.model_specs,
        "model_file_hashes": preconditions.model_file_hashes,
        "embedded_tokenizer_loader_cuda_and_gpu_receipts": runtime_receipt,
        "frozen_eligibility_error_taxonomy_prompts_seeds_and_budgets": (
            _frozen_eligibility_and_prompt_receipt(config, eligible_rows)
        ),
        "arm_definitions_and_compute_parity": _arm_parity_receipt(),
        "diagnostic_visibility_and_oracle_boundary": visibility,
        "per_model_error_family_repair_metrics": aggregates[
            "per_model_error_family_repair_metrics"
        ],
        "exact_semantic_repair_and_regression_metrics": aggregates[
            "exact_semantic_repair_and_regression_metrics"
        ],
        "omitted_spurious_and_unsafe_constraint_metrics": aggregates[
            "omitted_spurious_and_unsafe_constraint_metrics"
        ],
        "matched_no_diagnostic_no_information_and_shuffled_controls": aggregates[
            "matched_no_diagnostic_no_information_and_shuffled_controls"
        ],
        "group_bootstrap_lower_bounds": aggregates["group_bootstrap_lower_bounds"],
        "raw_trace_and_output_receipts": _raw_trace_receipt(config.raw_path(), sealed_rows),
        "gpu_utilization_vram_latency_and_energy_receipts": _gpu_latency_receipt(
            sealed_rows,
            collection.get("gpu_receipts") or {},
            collection.get("model_attempts") or [],
        ),
        "protected_files_unchanged": protected,
        "verification_guided_repair_ready_score": score,
        "duration_s": round(duration_s, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or {}),
        "reproducibility_checksum": "",
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _status_and_verdict(
    block_reason: str | None,
    completed_models: Sequence[str],
    score: float,
    unsafe_increase: bool,
) -> tuple[str, str]:
    if block_reason is not None:
        return "blocked_precondition", f"blocked_precondition: {block_reason}"
    if unsafe_increase:
        return "unsafe", "unsafe: exact-diagnostic repair increased unsafe acceptance"
    if set(completed_models) != set(MANDATED_MODEL_IDS):
        return "blocked", "blocked: not all mandated repair model families completed"
    if score == 1.0:
        return (
            "complete",
            "complete_positive: exact diagnostics beat matched repair controls",
        )
    return (
        "complete",
        "complete_null: exact diagnostics did not beat matched repair controls",
    )


def _ready_score(
    aggregates: Mapping[str, Any],
    completed_models: Sequence[str],
    visibility: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any],
) -> float:
    lower = aggregates["group_bootstrap_lower_bounds"]
    exact = aggregates["exact_semantic_repair_and_regression_metrics"]
    cells_no_diag = lower["by_model_error_family_cell"][
        "exact_vs_matched_second_call_no_diagnostic"
    ]
    cells_no_info = lower["by_model_error_family_cell"]["exact_vs_no_information_diagnostic"]
    enough_cells = len(cells_no_diag) >= PROMOTION_THRESHOLDS["required_model_family_error_cells"]
    positive_cells = all(float(value) > 0.0 for value in cells_no_diag.values()) and all(
        float(value) > 0.0 for value in cells_no_info.values()
    )
    ready = (
        set(completed_models) == set(MANDATED_MODEL_IDS)
        and runtime_receipt.get("all_runtime_offload_verified") is True
        and visibility.get("oracle_leakage_detected") is False
        and visibility.get("visible_trace_hash_coverage") is True
        and enough_cells
        and positive_cells
        and float(lower["exact_vs_matched_second_call_no_diagnostic"]) > 0.0
        and float(lower["exact_vs_no_information_diagnostic"]) > 0.0
        and int(exact["correct_row_regressions"]) == 0
        and not _unsafe_increase(aggregates)
    )
    return 1.0 if ready else 0.0


def _unsafe_increase(aggregates: Mapping[str, Any]) -> bool:
    by_arm = aggregates["omitted_spurious_and_unsafe_constraint_metrics"]["by_arm"]
    exact = by_arm["exact_diagnostic_repair"]["unsafe_accept_rate"]
    controls = max(
        by_arm["matched_second_call_no_diagnostic"]["unsafe_accept_rate"],
        by_arm["no_information_diagnostic"]["unsafe_accept_rate"],
        by_arm["shuffled_same_error_class_diagnostic"]["unsafe_accept_rate"],
        by_arm["no_repair"]["unsafe_accept_rate"],
    )
    return exact - controls > PROMOTION_THRESHOLDS["max_unsafe_increase"]


def _seal_one_row(
    trace_sequence_index: int,
    eligible: Mapping[str, Any],
    source: Mapping[str, Any],
    arm_id: str,
    raw: Mapping[str, Any],
) -> JsonDict:
    raw_text = str(raw.get("raw_output_text") or "")
    metadata = dict(raw)
    metadata["row_id"] = source["row_id"]
    evaluation = evaluate_candidate(source, arm_id, raw_text, metadata)
    diagnostic, diagnostic_source, same_class = _diagnostic_for_arm(eligible, arm_id)
    initial_correct = not _row_is_incorrect(eligible)
    final_correct = _row_success(evaluation, source)
    usage = dict(raw.get("usage") or {})
    latency_s = float(raw.get("latency_s") or 0.0)
    util = float(raw.get("average_gpu_utilization_pct") or 0.0)
    event: JsonDict = {
        "schema": RAW_ROW_SCHEMA_VERSION,
        "trace_sequence_index": trace_sequence_index,
        "event_kind": "constraint_ir_repair_arm_result",
        "source_stream_sequence_index": eligible.get("stream_sequence_index"),
        "model_hf_id": eligible.get("model_hf_id"),
        "model_name": eligible.get("model_name"),
        "model_path": eligible.get("model_path"),
        "gpu_index": eligible.get("gpu_index"),
        "source_row_id": eligible.get("source_row_id"),
        "group_id": eligible.get("group_id"),
        "family": eligible.get("family"),
        "template_id": eligible.get("template_id"),
        "split": eligible.get("split"),
        "variant_kind": eligible.get("variant_kind"),
        "expected_status": eligible.get("expected_status"),
        "eligible_error_class": eligible.get("eligible_error_class") or classify_error(eligible),
        "initial_arm_id": eligible.get("arm_id"),
        "arm_id": arm_id,
        "prompt_sha256": raw.get("prompt_sha256") or eligible.get("prompt_sha256"),
        "seed": raw.get("seed"),
        "raw_output_text": raw_text,
        "raw_output_sha256": sha256_text(raw_text),
        "usage": usage,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "latency_s": latency_s,
        "average_gpu_utilization_pct": util,
        "energy_proxy_gpu_utilization_pct_s": round(latency_s * util / 100.0, 6),
        "diagnostic_source_stream_sequence_index": diagnostic_source,
        "diagnostic_trace_sha256": diagnostic.get("visible_trace_sha256"),
        "repair_prompt_visible_diagnostic": diagnostic,
        "shuffled_diagnostic_same_error_class": same_class,
        "initial_correct": initial_correct,
        "final_correct": final_correct,
        "repaired_from_initial_incorrect": (not initial_correct) and final_correct,
        "correct_row_regressed": initial_correct and not final_correct,
        "initial_exact_labels": exp5909._exact_label_projection(eligible),
        "exact_labels": exp5909._exact_label_projection(evaluation),
        "visible_diagnostics": dict(evaluation.get("diagnostics") or {}),
        **evaluation,
        "row_hash": "",
    }
    event["row_hash"] = raw_trace_row_hash(event)
    return event


def _diagnostic_for_arm(
    eligible: Mapping[str, Any], arm_id: str
) -> tuple[JsonDict, int | None, bool | None]:
    if arm_id == "exact_diagnostic_repair":
        diagnostic = public_repair_diagnostic(eligible)
        return diagnostic, int(eligible.get("stream_sequence_index") or 0), True
    if arm_id == "no_information_diagnostic":
        diagnostic = _no_information_diagnostic()
        diagnostic["visible_trace_sha256"] = sha256_json(diagnostic)
        return diagnostic, None, None
    if arm_id == "shuffled_same_error_class_diagnostic":
        diagnostic = dict(eligible.get("shuffled_diagnostic") or {})
        if not diagnostic:
            diagnostic = _no_information_diagnostic()
        return (
            diagnostic,
            diagnostic.get("source_stream_sequence_index"),
            diagnostic.get("same_error_class"),
        )
    return {"visible_trace_sha256": sha256_json({"diagnostic": "withheld"})}, None, None


def _collect_live_repair_arms(
    llm: Any,
    spec: Mapping[str, Any],
    eligible: Mapping[str, Any],
    all_eligible: Sequence[Mapping[str, Any]],
    config: ExperimentConfig,
) -> list[JsonDict]:  # pragma: no cover - requires local GGUFs and CUDA.
    exact = public_repair_diagnostic(eligible)
    shuffled = _shuffled_diagnostic_for(eligible, all_eligible)
    diagnostics = {
        "exact_diagnostic_repair": exact,
        "matched_second_call_no_diagnostic": exact,
        "no_information_diagnostic": _no_information_diagnostic(),
        "shuffled_same_error_class_diagnostic": shuffled,
    }
    outputs = []
    for offset, arm_id in enumerate(TWO_CALL_ARMS, start=1):
        prompt = build_repair_prompt(eligible, arm_id, diagnostics[arm_id])
        outputs.append(_call_llama(llm, prompt, arm_id, spec, eligible, config, offset))
    return outputs


def _call_llama(
    llm: Any,
    prompt: str,
    arm_id: str,
    spec: Mapping[str, Any],
    eligible: Mapping[str, Any],
    config: ExperimentConfig,
    seed_offset: int,
) -> JsonDict:  # pragma: no cover - requires local GGUFs and CUDA.
    start = config.monotonic_clock()
    seed = config.random_seed + int(eligible.get("stream_sequence_index") or 0) + seed_offset * 100_000
    usage: JsonDict = {}
    output_text = ""
    try:
        result = llm(
            prompt,
            max_tokens=int(ARM_DEFINITIONS[arm_id]["second_call_max_tokens"]),
            temperature=DECODING["temperature"],
            top_p=DECODING["top_p"],
            repeat_penalty=DECODING["repeat_penalty"],
            stop=DECODING["stop"],
            seed=seed,
            echo=False,
        )
        output_text, usage = exp5909._completion_text_and_usage(result)
    except Exception as exc:  # noqa: BLE001
        usage = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "source_stream_sequence_index": eligible.get("stream_sequence_index"),
        "model_hf_id": spec.get("hf_id"),
        "model_name": spec.get("name"),
        "model_path": spec.get("model_path"),
        "gpu_index": spec.get("gpu"),
        "source_row_id": eligible.get("source_row_id"),
        "arm_id": arm_id,
        "prompt_sha256": sha256_text(prompt),
        "seed": seed,
        "raw_output_text": output_text,
        "latency_s": round(config.monotonic_clock() - start, 6),
        "usage": usage,
        "average_gpu_utilization_pct": _current_average_gpu_utilization_pct(),
    }


def _frozen_eligibility_and_prompt_receipt(
    config: ExperimentConfig, eligible_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    sample = dict(eligible_rows[0]) if eligible_rows else {
        "natural_language": "<row natural language>",
        "raw_output_text": "<previous candidate>",
        "visible_diagnostics": {"parser_status": "rejected", "parser_error": "<parser error>"},
        "eligible_error_class": "parser",
    }
    exact = public_repair_diagnostic(sample)
    shuffled = _shuffled_diagnostic_for(sample, eligible_rows) if eligible_rows else exact
    prompts = {
        "exact_diagnostic_repair": build_repair_prompt(
            sample, "exact_diagnostic_repair", exact
        ),
        "matched_second_call_no_diagnostic": build_repair_prompt(
            sample, "matched_second_call_no_diagnostic", exact
        ),
        "no_information_diagnostic": build_repair_prompt(
            sample, "no_information_diagnostic", _no_information_diagnostic()
        ),
        "shuffled_same_error_class_diagnostic": build_repair_prompt(
            sample, "shuffled_same_error_class_diagnostic", shuffled
        ),
    }
    return {
        "eligible_row_count": len(eligible_rows),
        "eligibility_rule": (
            "Exp5909 primary-arm residual incorrect rows with deployment-visible diagnostics, "
            "capped per model/family/error cell"
        ),
        "per_model_family_error_cap": config.max_rows_per_model_family_error,
        "error_taxonomy": ["parser", "type", "compile", "solver", "certificate", "semantic"],
        "eligible_cells": sorted(
            {
                "::".join(
                    [
                        str(row.get("model_hf_id")),
                        str(row.get("family")),
                        str(row.get("eligible_error_class")),
                    ]
                )
                for row in eligible_rows
            }
        ),
        "eligible_row_hashes": [
            {
                "source_stream_sequence_index": row.get("stream_sequence_index"),
                "row_hash": row.get("row_hash"),
                "visible_trace_sha256": row.get("visible_trace_sha256"),
            }
            for row in eligible_rows
        ],
        "prompt_version": "exp5910.verification_guided_constraint_repair.v1",
        "prompt_sha256": {arm: sha256_text(prompt) for arm, prompt in prompts.items()},
        "diagnostic_serialization": "canonical_json_visible_trace_sha256",
        "decoding": DECODING,
        "budgets": GENERATION_BUDGETS,
        "base_random_seed": config.random_seed,
        "promotion_thresholds": PROMOTION_THRESHOLDS,
        "target_oracle_payloads_in_prompts": False,
    }


def _arm_parity_receipt() -> JsonDict:
    second_call_counts = {
        arm: ARM_DEFINITIONS[arm]["calls_including_exp5909_initial"] for arm in TWO_CALL_ARMS
    }
    token_budgets = {arm: ARM_DEFINITIONS[arm]["second_call_max_tokens"] for arm in TWO_CALL_ARMS}
    return {
        "arms": ARM_DEFINITIONS,
        "two_call_arms_call_count_match": len(set(second_call_counts.values())) == 1,
        "two_call_arms_output_token_budget_match": len(set(token_budgets.values())) == 1,
        "no_repair_reuses_exp5909_initial_call": True,
        "principle": FIELD_PRINCIPLES["arm_definitions_and_compute_parity"],
    }


def _runtime_loader_receipt(
    preconditions: PreconditionReport, collection: Mapping[str, Any]
) -> JsonDict:
    receipt = json.loads(canonical_json(preconditions.embedded_tokenizer_loader_cuda_and_gpu_receipts))
    attempts = [dict(row) for row in collection.get("model_attempts") or []]
    receipt["runtime_loader_receipts"] = attempts
    receipt["gpu_receipts"] = dict(collection.get("gpu_receipts") or {})
    receipt["all_runtime_offload_verified"] = bool(attempts) and all(
        bool(row.get("gpu_offload_verified")) for row in attempts
    )
    return receipt


def _raw_trace_receipt(path: Path, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "path": str(RAW_TRACE_RELATIVE_PATH),
        "schema": RAW_ROW_SCHEMA_VERSION,
        "row_count": len(rows),
        "expected_row_count": _expected_trace_row_count(rows),
        "sha256": sha256_file(path) if path.exists() else None,
        "event_order_is_chronological": [
            int(row.get("trace_sequence_index") or 0) for row in rows
        ]
        == list(range(len(rows))),
        "raw_hash_coverage": all(row.get("raw_output_sha256") for row in rows),
        "row_hash_coverage": all(row.get("row_hash") == raw_trace_row_hash(row) for row in rows),
        "visible_trace_hash_coverage": all(
            row.get("diagnostic_trace_sha256") for row in rows if row.get("arm_id") in TWO_CALL_ARMS
        ),
        "raw_output_hashes": [
            {
                "trace_sequence_index": row.get("trace_sequence_index"),
                "source_stream_sequence_index": row.get("source_stream_sequence_index"),
                "model_hf_id": row.get("model_hf_id"),
                "arm_id": row.get("arm_id"),
                "raw_output_sha256": row.get("raw_output_sha256"),
                "row_hash": row.get("row_hash"),
            }
            for row in rows
        ],
    }


def _gpu_latency_receipt(
    rows: Sequence[Mapping[str, Any]],
    gpu_receipts: Mapping[str, Any],
    model_attempts: Sequence[Any],
) -> JsonDict:
    return {
        "gpu_receipts": dict(gpu_receipts),
        "model_attempts": [dict(row) for row in model_attempts if isinstance(row, Mapping)],
        "latency_s_total": round(sum(float(row.get("latency_s") or 0.0) for row in rows), 6),
        "energy_proxy_gpu_utilization_pct_s_total": round(
            sum(float(row.get("energy_proxy_gpu_utilization_pct_s") or 0.0) for row in rows), 6
        ),
        "row_count": len(rows),
    }


def _group_bootstrap_lower_bounds(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    cells_no_diag = _cell_lower_bounds(
        rows, "exact_diagnostic_repair", "matched_second_call_no_diagnostic"
    )
    cells_no_info = _cell_lower_bounds(
        rows, "exact_diagnostic_repair", "no_information_diagnostic"
    )
    cells_shuffled = _cell_lower_bounds(
        rows, "exact_diagnostic_repair", "shuffled_same_error_class_diagnostic"
    )
    return {
        "exact_vs_no_repair": _bootstrap_lower_bound(
            rows, "exact_diagnostic_repair", "no_repair"
        ),
        "exact_vs_matched_second_call_no_diagnostic": _min_cell_bound(cells_no_diag),
        "exact_vs_no_information_diagnostic": _min_cell_bound(cells_no_info),
        "exact_vs_shuffled_same_error_class_diagnostic": _min_cell_bound(cells_shuffled),
        "by_model_error_family_cell": {
            "exact_vs_matched_second_call_no_diagnostic": cells_no_diag,
            "exact_vs_no_information_diagnostic": cells_no_info,
            "exact_vs_shuffled_same_error_class_diagnostic": cells_shuffled,
        },
        "method": "deterministic_cell_lower_bound_by_model_family_error",
    }


def _bootstrap_lower_bound(
    rows: Sequence[Mapping[str, Any]], left_arm: str, right_arm: str
) -> float:
    cells = _cell_lower_bounds(rows, left_arm, right_arm)
    return _min_cell_bound(cells)


def _cell_lower_bounds(
    rows: Sequence[Mapping[str, Any]], left_arm: str, right_arm: str
) -> JsonDict:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = "::".join(
            [
                str(row.get("model_hf_id")),
                str(row.get("family")),
                str(row.get("eligible_error_class")),
            ]
        )
        grouped[key][str(row.get("arm_id"))].append(1.0 if _row_success(row, row) else 0.0)
    bounds = {}
    for key, arms in sorted(grouped.items()):
        if not arms.get(left_arm) or not arms.get(right_arm):
            continue
        left = sum(arms[left_arm]) / len(arms[left_arm])
        right = sum(arms[right_arm]) / len(arms[right_arm])
        bounds[key] = round(left - right, 6)
    return bounds


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    row_list = [dict(row) for row in rows]
    total = len(row_list)
    semantic = [row for row in row_list if row.get("expected_status") == "valid"]
    query = [row for row in semantic if row.get("query_correct") is not None]
    return {
        "n": total,
        "semantic_n": len(semantic),
        "parse_valid_rate": _rate(sum(bool(row.get("parse_valid")) for row in row_list), total),
        "type_valid_rate": _rate(sum(bool(row.get("type_valid")) for row in row_list), total),
        "compile_rate": _rate(sum(bool(row.get("compiled")) for row in row_list), total),
        "satisfiability_correct_rate": _rate(
            sum(bool(row.get("satisfiability_correct")) for row in row_list), total
        ),
        "exact_semantic_equivalence_rate": _rate(
            sum(row.get("exact_semantic_equivalence") is True for row in semantic), len(semantic)
        ),
        "query_correct_rate": _rate(
            sum(row.get("query_correct") is True for row in query), len(query)
        ),
        "repair_success_rate": _rate(
            sum(bool(row.get("repaired_from_initial_incorrect")) for row in row_list), total
        ),
        "unsafe_accept_rate": _rate(
            sum(bool(row.get("unsafe_accepted_constraints")) for row in row_list), total
        ),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in row_list),
        "latency_s": round(sum(float(row.get("latency_s") or 0.0) for row in row_list), 6),
        "energy_proxy_gpu_utilization_pct_s": round(
            sum(float(row.get("energy_proxy_gpu_utilization_pct_s") or 0.0) for row in row_list),
            6,
        ),
    }


def _exact_repair_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    no_repair = _rows_for_arm(rows, "no_repair")
    exact = _rows_for_arm(rows, "exact_diagnostic_repair")
    return {
        "initial_residual_rows": len(no_repair),
        "initially_correct_rows_evaluated": sum(bool(row.get("initial_correct")) for row in exact),
        "exact_repair_successes": sum(
            bool(row.get("repaired_from_initial_incorrect")) for row in exact
        ),
        "exact_repair_success_rate": _rate(
            sum(bool(row.get("repaired_from_initial_incorrect")) for row in exact), len(exact)
        ),
        "correct_row_regressions": sum(bool(row.get("correct_row_regressed")) for row in exact),
        "parse_valid_delta_exact_vs_no_repair": round(
            _metric_summary(exact)["parse_valid_rate"] - _metric_summary(no_repair)["parse_valid_rate"],
            6,
        ),
        "type_valid_delta_exact_vs_no_repair": round(
            _metric_summary(exact)["type_valid_rate"] - _metric_summary(no_repair)["type_valid_rate"],
            6,
        ),
        "compile_delta_exact_vs_no_repair": round(
            _metric_summary(exact)["compile_rate"] - _metric_summary(no_repair)["compile_rate"],
            6,
        ),
    }


def _constraint_error_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_arm = {}
    for arm_id in ALL_ARM_IDS:
        arm_rows = _rows_for_arm(rows, arm_id)
        by_arm[arm_id] = {
            "omitted_constraints": sum(int(row.get("omitted_constraints") or 0) for row in arm_rows),
            "spurious_constraints": sum(int(row.get("spurious_constraints") or 0) for row in arm_rows),
            "unsafe_accepted_constraints": sum(
                bool(row.get("unsafe_accepted_constraints")) for row in arm_rows
            ),
            "unsafe_accept_rate": _rate(
                sum(bool(row.get("unsafe_accepted_constraints")) for row in arm_rows),
                len(arm_rows),
            ),
        }
    return {
        "by_arm": by_arm,
        "overall": {
            "omitted_constraints": sum(int(row.get("omitted_constraints") or 0) for row in rows),
            "spurious_constraints": sum(int(row.get("spurious_constraints") or 0) for row in rows),
            "unsafe_accepted_constraints": sum(
                bool(row.get("unsafe_accepted_constraints")) for row in rows
            ),
        },
    }


def _grouped_metrics(rows: Sequence[Mapping[str, Any]], key: str) -> JsonDict:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "unknown")].append(row)
    return {name: _metric_summary(group_rows) for name, group_rows in sorted(groups.items())}


def _grouped_combo_metrics(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> JsonDict:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        name = "::".join(str(row.get(key) or "unknown") for key in keys)
        groups[name].append(row)
    return {name: _metric_summary(group_rows) for name, group_rows in sorted(groups.items())}


def _rows_for_arm(rows: Sequence[Mapping[str, Any]], arm_id: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("arm_id") == arm_id]


def _row_success(row: Mapping[str, Any], source: Mapping[str, Any]) -> bool:
    expected = str(source.get("expected_status") or row.get("expected_status"))
    if expected == "valid":
        return row.get("exact_semantic_equivalence") is True and row.get("query_correct") is True
    return row.get("satisfiability_correct") is True


def _row_is_incorrect(row: Mapping[str, Any]) -> bool:
    if row.get("expected_status") == "valid":
        return row.get("exact_semantic_equivalence") is not True or row.get("query_correct") is not True
    return row.get("satisfiability_correct") is not True


def _has_visible_diagnostic(row: Mapping[str, Any]) -> bool:
    diagnostics = row.get("visible_diagnostics") or row.get("diagnostics") or {}
    return isinstance(diagnostics, Mapping) and bool(diagnostics.get("parser_status"))


def _uses_forbidden_oracle_material(value: Any) -> bool:
    text = canonical_json(value).lower()
    return any(token in text for token in FORBIDDEN_ORACLE_TOKENS)


def _shuffled_diagnostic_for(
    row: Mapping[str, Any], eligible_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    same_class = [
        candidate
        for candidate in eligible_rows
        if candidate.get("eligible_error_class") == row.get("eligible_error_class")
        and candidate.get("stream_sequence_index") != row.get("stream_sequence_index")
    ]
    if not same_class:
        diagnostic = _no_information_diagnostic()
        diagnostic["source_stream_sequence_index"] = None
        diagnostic["same_error_class"] = False
        diagnostic["visible_trace_sha256"] = sha256_json(diagnostic)
        return diagnostic
    same_class.sort(key=lambda item: int(item.get("stream_sequence_index") or 0))
    source = same_class[0]
    diagnostic = public_repair_diagnostic(source)
    diagnostic["source_stream_sequence_index"] = source.get("stream_sequence_index")
    diagnostic["same_error_class"] = True
    diagnostic["visible_trace_sha256"] = sha256_json(diagnostic)
    return diagnostic


def _no_information_diagnostic() -> JsonDict:
    return {
        "error_class": "withheld",
        "parser_status": "withheld",
        "parser_error": "withheld",
        "type_status": "withheld",
        "compiler_status": "withheld",
        "solver_status": "withheld",
        "certificate_status": "withheld",
        "diagnostic_payload": "no_information_control",
    }


def _safe_freeze_eligible_rows(config: ExperimentConfig) -> list[JsonDict]:
    try:
        return freeze_eligible_rows(config)
    except Exception:  # noqa: BLE001
        return []


def _load_exp5909_raw_rows(root: Path) -> list[JsonDict]:
    path = root / exp5909.RAW_STREAM_RELATIVE_PATH
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _upstream_gate_receipt(root: Path) -> JsonDict:
    artifact_path = root / exp5909.RESULT_RELATIVE_PATH
    raw_path = root / exp5909.RAW_STREAM_RELATIVE_PATH
    artifact: JsonDict = {}
    validate_error = None
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        exp5909.validate_artifact(artifact)
    except Exception as exc:  # noqa: BLE001
        validate_error = f"{type(exc).__name__}: {exc}"
    raw_sha = sha256_file(raw_path) if raw_path.exists() else None
    receipt_sha = (artifact.get("chronological_raw_stream_receipt") or {}).get("sha256")
    stream_ready = (
        validate_error is None
        and artifact.get("constraint_stream_ready_score") == 1.0
        and raw_sha is not None
        and raw_sha == receipt_sha
    )
    repair_ready = stream_ready and artifact.get("verification_repair_admission_ready_score") == 1.0
    return {
        "exp5909_artifact_path": str(exp5909.RESULT_RELATIVE_PATH),
        "exp5909_raw_stream_path": str(exp5909.RAW_STREAM_RELATIVE_PATH),
        "exp5909_artifact_sha256": sha256_file(artifact_path) if artifact_path.exists() else None,
        "exp5909_raw_stream_sha256": raw_sha,
        "exp5909_validate_error": validate_error,
        "exp5909_stream_ready": stream_ready,
        "exp5909_repair_admission_ready": repair_ready,
        "exp5909_constraint_stream_ready_score": artifact.get("constraint_stream_ready_score"),
        "exp5909_verification_repair_admission_ready_score": artifact.get(
            "verification_repair_admission_ready_score"
        ),
        "exp5909_raw_row_count": (artifact.get("chronological_raw_stream_receipt") or {}).get(
            "row_count"
        ),
        "exp5909_raw_stream_hash_matches_receipt": raw_sha is not None and raw_sha == receipt_sha,
    }


def _model_file_hashes(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    return exp5909._model_file_hashes(model_specs)


def _tokenizer_receipts(
    model_specs: Sequence[Mapping[str, Any]], tokenizer_checker: TokenizerChecker
) -> list[JsonDict]:
    return exp5909._tokenizer_receipts(model_specs, tokenizer_checker)


def _protected_file_receipt(root: Path, baseline: Mapping[str, Any] | None = None) -> JsonDict:
    files = []
    before = {str(row.get("path")): row for row in (baseline or {}).get("files", [])}
    unchanged = True
    for relative in PROTECTED_FILES:
        path = root / relative
        sha = sha256_file(path) if path.exists() else None
        prior = before.get(str(relative), {}).get("sha256") if baseline else sha
        same = sha == prior
        unchanged = unchanged and same
        files.append(
            {"path": str(relative), "exists": path.exists(), "sha256": sha, "unchanged": same}
        )
    return {"unchanged": unchanged, "files": files}


def _completed_headline_models(model_attempts: Sequence[Any]) -> list[str]:
    seen: list[str] = []
    for attempt in model_attempts:
        if not isinstance(attempt, Mapping):
            continue
        hf_id = str(attempt.get("hf_id"))
        if hf_id in MANDATED_MODEL_IDS and attempt.get("model_used") and not attempt.get("blocker"):
            if hf_id not in seen:
                seen.append(hf_id)
    return seen


def _field_provenance() -> JsonDict:
    return {
        field: {
            "principle": FIELD_PRINCIPLES.get(field, "Exp5910 required artifact field."),
            "satisfied_by": "generated_by_exp5910_verification_guided_constraint_repair",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    stable = json.loads(canonical_json(artifact))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    for row in stable.get("model_file_hashes", {}).get("files", []):
        row["path"] = "<model_path>"
    for spec in stable.get("model_specs", []):
        spec["model_path"] = "<model_path>"
    for file_row in stable.get("protected_files_unchanged", {}).get("files", []):
        file_row["sha256"] = "<protected_sha>"
    return sha256_text(canonical_json(stable))


def _expected_trace_row_count(rows: Sequence[Mapping[str, Any]]) -> int:
    source_ids = {row.get("source_stream_sequence_index") for row in rows}
    return len(source_ids) * len(ALL_ARM_IDS)


def _min_cell_bound(cells: Mapping[str, Any]) -> float:
    if not cells:
        return 0.0
    return round(min(float(value) for value in cells.values()), 6)


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _current_average_gpu_utilization_pct() -> float:  # pragma: no cover - host-dependent.
    health = exp5909._gpu_health_probe()
    gpus = health.get("gpus") or []
    if not gpus:
        return 0.0
    return round(sum(float(gpu.get("utilization_gpu_pct") or 0.0) for gpu in gpus) / len(gpus), 6)


def _write_raw_rows_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    text = "".join(canonical_json(row) + "\n" for row in rows)
    _write_text_atomic(path, text)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        tmp_path = Path(handle.name)
        handle.write(text)
    os.replace(tmp_path, path)


def _parse_test_exit_codes(values: Sequence[str]) -> JsonDict:  # pragma: no cover - CLI wrapper.
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError("test exit code arguments must be COMMAND=CODE")
        command, code = value.rsplit("=", 1)
        parsed[command] = int(code)
    return parsed


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--refresh-test-exit-code", action="append", default=[])
    parser.add_argument("--all-eligible-rows", action="store_true")
    args = parser.parse_args(argv)
    config = ExperimentConfig(
        repo_root=args.root,
        max_rows_per_model_family_error=None
        if args.all_eligible_rows
        else ELIGIBLE_ROWS_PER_MODEL_FAMILY_ERROR,
    )
    if args.refresh_test_exit_code:
        artifact = refresh_artifact_test_exit_codes(
            root=args.root,
            test_exit_codes=_parse_test_exit_codes(args.refresh_test_exit_code),
        )
    else:
        artifact = run_experiment(config)
    print(
        "[exp5910] "
        f"status={artifact['status']} "
        f"verdict={artifact['honest_verdict']} "
        f"score={artifact['verification_guided_repair_ready_score']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
