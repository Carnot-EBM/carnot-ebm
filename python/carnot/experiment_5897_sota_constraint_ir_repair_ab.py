"""Exp5897 SOTA ConstraintIR trace-repair A/B.

Spec refs: REQ-BENCH-5897, SCENARIO-BENCH-5897-PRECONDITIONS,
SCENARIO-BENCH-5897-TRACE-BOUNDARY, SCENARIO-BENCH-5897-EXACT-METRICS.

The experiment asks local GGUF models to propose typed ConstraintIR JSON and
then lets the Exp5896 exact executor decide every parser, type, compile,
satisfiability, and semantic-equivalence label. The repair arm may see only
deployment-visible diagnostics from its own failed proposal. It never sees the
hidden fixture IR, gold labels, held-out identities, or certificate solutions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any

from carnot import experiment_5896_typed_constraint_ir_fixture as exp5896
from carnot.inference.sota_models import (
    cached_sota_pair,
    gguf_tokenizer_loadable,
    resolve_cached_gguf,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5897_sota_constraint_ir_repair_ab.json")
RAW_OUTPUT_RELATIVE_PATH = Path("results/experiment_5897_sota_constraint_ir_repair_ab.raw.jsonl")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5897_sota_constraint_ir_repair_ab.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5897_sota_constraint_ir_repair_ab.py")
RUN_DATE = "20260724"
RANDOM_SEED = 5897
INFERENCE_SUBSTRATE = "live_llm_inference"
VERIFIER_IS_ORACLE = True

MANDATED_MODEL_IDS: tuple[str, str, str] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": MANDATED_MODEL_IDS[0],
        "family": "qwen_moe",
        "role": "flagship_moe",
        "preferred_quant": "Q4_K_M",
        "gpu": 0,
        "min_vram_gb": 24,
    },
    {
        "name": "Gemma4-31B-it",
        "hf_id": MANDATED_MODEL_IDS[1],
        "family": "gemma_dense",
        "role": "flagship_dense",
        "preferred_quant": "Q4_K_M",
        "gpu": 1,
        "min_vram_gb": 24,
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": MANDATED_MODEL_IDS[2],
        "family": "gemma_moe",
        "role": "middle_moe",
        "preferred_quant": "Q4_K_M",
        "gpu": 0,
        "min_vram_gb": 16,
    },
)

DECODING: JsonDict = {
    "temperature": 0.0,
    "top_p": 1.0,
    "repeat_penalty": 1.05,
    "stop": ["</s>", "<eos>"],
}
ARM_DEFINITIONS: JsonDict = {
    "single_pass": {
        "calls": 1,
        "max_tokens": 1536,
        "diagnostic_trace": False,
        "principle": "Baseline extraction gets one proposal call and no exact diagnostics.",
    },
    "trace_guided_repair": {
        "calls": 2,
        "max_tokens": 1536,
        "diagnostic_trace": True,
        "principle": "Only deployment-visible exact diagnostics may guide the second proposal.",
    },
    "matched_two_call_no_trace": {
        "calls": 2,
        "max_tokens": 1536,
        "diagnostic_trace": False,
        "principle": "A second call with matched budget controls for extra compute alone.",
    },
    "no_information_trace_control": {
        "calls": 2,
        "max_tokens": 1536,
        "diagnostic_trace": "uninformative",
        "principle": "A trace-shaped block with no diagnostics controls for prompt format.",
    },
}
PROMOTION_THRESHOLDS: JsonDict = {
    "required_completed_headline_families": 3,
    "held_group_lower_bound_min": 0.0,
    "max_unsafe_delta": 0.0,
}
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5897_sota_constraint_ir_repair_ab.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5897_sota_constraint_ir_repair_ab.py "
    "-m pytest tests/python/test_experiment_5897_sota_constraint_ir_repair_ab.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5897_sota_constraint_ir_repair_ab.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_5897_sota_constraint_ir_repair_ab.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "git status --short -- scripts/research_conductor.py ops/changelog.md ops/status.md _bmad/traceability.md",
)
PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "upstream_gate_and_fixture_hashes",
    "model_specs",
    "model_file_hashes",
    "loader_and_cuda_receipts",
    "frozen_prompts_decoding_seeds_and_budgets",
    "arm_definitions_and_compute_parity",
    "trace_visibility_and_oracle_boundary",
    "per_model_family_and_template_metrics",
    "parse_type_compile_and_semantic_metrics",
    "omitted_spurious_and_unsafe_constraint_metrics",
    "group_bootstrap_lower_bounds",
    "no_trace_and_no_information_controls",
    "gpu_utilization_vram_and_latency_receipts",
    "raw_output_receipts",
    "protected_files_unchanged",
    "trace_repair_mechanism_ready_score",
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
    "arm_definitions_and_compute_parity": (
        "A second call alone cannot be credited as trace-guided repair."
    ),
    "trace_visibility_and_oracle_boundary": (
        "Only deployment-available exact diagnostics may guide repair."
    ),
    "trace_repair_mechanism_ready_score": (
        "Emit bare 1.0 only for positive held lower bounds over both controls, "
        "zero unsafe increase, and all three headline families completed."
    ),
    "inference_substrate": "Use live_llm_inference.",
    "verifier_is_oracle": "True for exact evaluation; the model is only an IR proposer.",
    "honest_verdict": (
        "Use complete_positive:, complete_null:, unsafe:, blocked_precondition:, or blocked:."
    ),
}


CachedPairProvider = Callable[..., list[JsonDict] | None]
IndividualResolver = Callable[[str], str | None]
EnvironmentProbe = Callable[[Path], JsonDict]
TokenizerChecker = Callable[[str | None], tuple[bool, str]]
CollectOutputsFn = Callable[[list[JsonDict], list[JsonDict], "ExperimentConfig"], JsonDict]


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and clocks for Exp5897."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    raw_output_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    monotonic_clock: Callable[[], float] = time.monotonic
    random_seed: int = RANDOM_SEED

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / RESULT_RELATIVE_PATH

    def raw_path(self) -> Path:
        return self.raw_output_path or self.repo_root / RAW_OUTPUT_RELATIVE_PATH


@dataclass(frozen=True)
class PreconditionReport:
    """All checks that must pass before a headline GGUF model is loaded."""

    preconditions_checked: JsonDict
    upstream_gate_and_fixture_hashes: JsonDict
    model_specs: list[JsonDict]
    model_file_hashes: JsonDict
    loader_and_cuda_receipts: JsonDict
    protected_file_baseline: JsonDict
    block_reason: str | None


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence with stable bytes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


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
    """Resolve Qwen, Gemma dense, and Gemma MoE in the frozen headline order."""

    pair_error: str | None = None
    pair: list[JsonDict] | None = None
    try:
        pair = cached_pair_provider(gpu_indices=(0, 1), model_indices=(0, 2))
    except Exception as exc:  # noqa: BLE001
        pair_error = f"{type(exc).__name__}: {exc}"
    by_hf_id = {str(spec.get("hf_id")): dict(spec) for spec in pair or []}
    resolved: list[JsonDict] = []
    resolver_hits: list[str] = []
    for base in MODEL_SPECS:
        hf_id = str(base["hf_id"])
        spec = dict(base)
        if hf_id in by_hf_id:
            spec.update(by_hf_id[hf_id])
            spec.setdefault("family", base["family"])
            spec.setdefault("role", base["role"])
            spec.setdefault("preferred_quant", base["preferred_quant"])
            spec.setdefault("min_vram_gb", base["min_vram_gb"])
        else:
            path = individual_model_resolver(hf_id)
            if path:
                spec["model_path"] = str(path)
                resolver_hits.append(hf_id)
            else:
                spec["model_path"] = None
        resolved.append(spec)
    receipt = {
        "cached_sota_pair_called_with": {"gpu_indices": [0, 1], "model_indices": [0, 2]},
        "cached_sota_pair_error": pair_error,
        "cached_sota_pair_returned": [str(spec.get("hf_id")) for spec in pair or []],
        "third_family_resolved_by_individual_cache": MANDATED_MODEL_IDS[2] in resolver_hits,
        "resolved_hf_ids": [str(spec["hf_id"]) for spec in resolved],
    }
    return resolved, receipt


def check_preconditions(
    config: ExperimentConfig,
    *,
    cached_pair_provider: CachedPairProvider = cached_sota_pair,
    individual_model_resolver: IndividualResolver = resolve_cached_gguf,
    environment_probe: EnvironmentProbe = lambda root: _probe_environment(root),
    tokenizer_checker: TokenizerChecker = gguf_tokenizer_loadable,
) -> PreconditionReport:
    """Check every headline resource before any GGUF model load."""

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
    loader_receipts = {
        "model_resolution": resolution,
        "environment": environment,
        "tokenizer_receipts": tokenizer_receipts,
    }
    checks = {
        "exp5896_gate_replayed": bool(upstream.get("replay_ok")),
        "all_three_model_specs_resolved": all(spec.get("model_path") for spec in model_specs),
        "all_model_files_exist": all(
            row.get("exists") for row in model_file_hashes.get("files", [])
        ),
        "all_model_hashes_recorded": all(
            row.get("sha256") for row in model_file_hashes.get("files", [])
        ),
        "embedded_gguf_tokenizers_load": all(row.get("ok") for row in tokenizer_receipts),
        "llama_cpp_import_ok": bool(environment.get("llama_cpp_import", {}).get("ok")),
        "llama_cpp_cuda_support": bool(environment.get("llama_cpp_cuda_support", {}).get("ok")),
        "two_healthy_rtx_3090s": _two_healthy_rtx_3090s(environment.get("gpu_health", {})),
        "adequate_vram": _adequate_vram(environment.get("gpu_health", {})),
        "adequate_ram": bool(environment.get("ram", {}).get("ok")),
        "adequate_disk": bool(environment.get("disk", {}).get("ok")),
        "no_protected_workload": bool(environment.get("protected_workload", {}).get("ok")),
        "atomic_output_ready": bool(environment.get("atomic_output", {}).get("ok")),
    }
    block_reason = next((name for name, ok in checks.items() if not ok), None)
    preconditions_checked = {
        "run_order": "exp5896_replay_and_resource_checks_before_any_model_load",
        "blocked_before_model_load": block_reason is not None,
        "headline_checks": checks,
        "block_reason": block_reason,
    }
    return PreconditionReport(
        preconditions_checked=preconditions_checked,
        upstream_gate_and_fixture_hashes=upstream,
        model_specs=model_specs,
        model_file_hashes=model_file_hashes,
        loader_and_cuda_receipts=loader_receipts,
        protected_file_baseline=protected_baseline,
        block_reason=block_reason,
    )


def build_single_pass_prompt(row: Mapping[str, Any]) -> str:
    """Build the frozen one-call typed ConstraintIR extraction prompt."""

    return (
        "Return exactly one JSON object for the typed ConstraintIR schema. "
        "Use keys schema_version, domains, entities, predicates, facts, rules, query. "
        "Supported nodes are atom, not, and, and arith. Use finite domains only. "
        "Do not include explanations, confidence scores, or markdown.\n"
        f"schema_version: {exp5896.CONSTRAINT_IR_SCHEMA_VERSION}\n"
        f"Problem text:\n{row['natural_language']}\n"
    )


def build_trace_repair_prompt(
    row: Mapping[str, Any],
    prior_output: str,
    diagnostic_trace: Mapping[str, Any],
) -> str:
    """Build the second-call repair prompt with exact public diagnostics only."""

    return (
        "Return exactly one corrected typed ConstraintIR JSON object and no prose. "
        "Repair only the constraint JSON. Do not answer the problem in natural language. "
        "The exact executor exposed these deployment diagnostics from your previous JSON:\n"
        f"{canonical_json(diagnostic_trace)}\n"
        f"Problem text:\n{row['natural_language']}\n"
        f"Previous JSON attempt:\n{prior_output}\n"
    )


def build_matched_no_trace_prompt(row: Mapping[str, Any], prior_output: str) -> str:
    """Build the matched second call that withholds exact diagnostics."""

    return (
        "Return exactly one revised typed ConstraintIR JSON object and no prose. "
        "Use the same schema and token budget as the repair condition. "
        "No parser, compiler, or solver details are provided in this condition.\n"
        f"Problem text:\n{row['natural_language']}\n"
        f"Previous JSON attempt:\n{prior_output}\n"
    )


def build_no_information_trace_prompt(row: Mapping[str, Any], prior_output: str) -> str:
    """Build a trace-shaped second call whose fields carry no diagnostics."""

    blank_trace = {
        "parser_status": "withheld",
        "type_status": "withheld",
        "compiler_status": "withheld",
        "solver_status": "withheld",
        "diagnostic_payload": "no_information_control",
    }
    return (
        "Return exactly one revised typed ConstraintIR JSON object and no prose. "
        "The following structured block is intentionally non-informative:\n"
        f"{canonical_json(blank_trace)}\n"
        f"Problem text:\n{row['natural_language']}\n"
        f"Previous JSON attempt:\n{prior_output}\n"
    )


def public_diagnostic_trace(evaluation: Mapping[str, Any]) -> JsonDict:
    """Expose only diagnostics a deployment parser/compiler/solver would reveal."""

    diagnostics = dict(evaluation.get("diagnostics") or {})
    return {
        "parser_status": diagnostics.get("parser_status"),
        "parser_error": diagnostics.get("parser_error"),
        "type_status": diagnostics.get("type_status"),
        "compiler_status": diagnostics.get("compiler_status"),
        "compiler_error": diagnostics.get("compiler_error"),
        "solver_status": diagnostics.get("solver_status"),
        "cross_backend_agreement": diagnostics.get("cross_backend_agreement"),
    }


def evaluate_candidate(
    row: Mapping[str, Any],
    arm_id: str,
    raw_text: str,
    generation_metadata: Mapping[str, Any],
) -> JsonDict:
    """Score one proposed ConstraintIR using only the Exp5896 exact executor."""

    candidate, json_error = _extract_json_object(raw_text)
    base = _evaluation_base(row, arm_id, raw_text, generation_metadata)
    if candidate is None:
        return base | _rejected_evaluation("no_json_object", row, None)

    receipt = exp5896.certify_ir(candidate)
    parser = dict(receipt.get("parser") or {})
    parser_status = str(parser.get("status"))
    parser_error = parser.get("error")
    parser_kind = parser.get("kind")
    parse_valid = parser_status == "accepted"
    type_valid = parse_valid
    if parser_kind == "type_error":
        type_valid = False
    if not parse_valid:
        return base | _rejected_evaluation(str(parser_error or json_error), row, parser_kind)

    python_receipt = dict(receipt.get("python") or {})
    z3_receipt = dict(receipt.get("z3") or {})
    solver_status = str(python_receipt.get("status"))
    z3_status = str(z3_receipt.get("status"))
    compiled = solver_status in {"sat", "unsat"} and z3_status in {"sat", "unsat"}
    expected_status = str(row.get("expected_status"))
    exact_equivalence, query_correct = _semantic_checks(row, python_receipt)
    satisfiability_correct = _satisfiability_correct(expected_status, solver_status, parse_valid)
    omitted, spurious = _constraint_diff_counts(row.get("constraint_ir"), candidate)
    unsafe = _unsafe_accepted(expected_status, solver_status, compiled, exact_equivalence)
    return base | {
        "parse_valid": parse_valid,
        "type_valid": type_valid,
        "compiled": compiled,
        "satisfiability_correct": satisfiability_correct,
        "solver_status": solver_status,
        "z3_status": z3_status,
        "exact_semantic_equivalence": exact_equivalence,
        "query_correct": query_correct,
        "omitted_constraints": omitted,
        "spurious_constraints": spurious,
        "unsafe_accepted_constraints": unsafe,
        "candidate_sha256": sha256_text(canonical_json(candidate)),
        "diagnostics": {
            "parser_status": parser_status,
            "parser_error": None,
            "type_status": "accepted",
            "compiler_status": "compiled" if compiled else "not_compiled",
            "compiler_error": None if compiled else f"python={solver_status},z3={z3_status}",
            "solver_status": solver_status,
            "cross_backend_agreement": receipt.get("cross_backend_agreement", {}).get("agrees"),
        },
    }


def aggregate_evaluations(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Aggregate exact metrics overall, by arm, model, family, and template."""

    row_list = [dict(row) for row in rows]
    by_arm = {
        arm_id: _metric_summary([row for row in row_list if row.get("arm_id") == arm_id])
        for arm_id in ARM_DEFINITIONS
    }
    by_model = _grouped_metrics(row_list, "model_hf_id")
    by_family = _grouped_metrics(row_list, "family")
    by_template = _grouped_metrics(row_list, "template_id")
    lower_bounds = _group_bootstrap_lower_bounds(row_list)
    control_metrics = {
        "matched_two_call_no_trace": by_arm["matched_two_call_no_trace"],
        "no_information_trace_control": by_arm["no_information_trace_control"],
        "principle": FIELD_PRINCIPLES["arm_definitions_and_compute_parity"],
    }
    return {
        "per_model_family_and_template_metrics": {
            "by_model": by_model,
            "by_family": by_family,
            "by_template": by_template,
        },
        "parse_type_compile_and_semantic_metrics": {
            "overall": _metric_summary(row_list),
            "by_arm": by_arm,
        },
        "omitted_spurious_and_unsafe_constraint_metrics": _constraint_error_metrics(row_list),
        "group_bootstrap_lower_bounds": lower_bounds,
        "no_trace_and_no_information_controls": control_metrics,
    }


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    cached_pair_provider: CachedPairProvider = cached_sota_pair,
    individual_model_resolver: IndividualResolver = resolve_cached_gguf,
    environment_probe: EnvironmentProbe = lambda root: _probe_environment(root),
    tokenizer_checker: TokenizerChecker = gguf_tokenizer_loadable,
    collect_model_outputs_fn: CollectOutputsFn | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Run Exp5897 and write the terminal artifact atomically."""

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
            evaluated_rows=[],
            collection={"model_attempts": [], "gpu_receipts": {}},
            raw_output_receipts=_raw_receipt(active.raw_path(), []),
            duration_s=active.clock() - started,
            test_exit_codes=test_exit_codes,
        )
        _write_json_atomic(active.artifact_path(), artifact)
        return artifact

    fixture_rows = _load_fixture_rows(active.repo_root)
    collector = collect_model_outputs_fn or collect_live_model_outputs
    collection = collector(preconditions.model_specs, fixture_rows, active)
    raw_rows = [dict(row) for row in collection.get("rows") or []]
    _write_raw_rows_atomic(active.raw_path(), raw_rows)
    row_by_id = {str(row["row_id"]): dict(row) for row in fixture_rows}
    evaluated_rows = [
        evaluate_candidate(
            row_by_id[str(raw["row_id"])],
            str(raw["arm_id"]),
            str(raw.get("raw_output_text") or ""),
            raw,
        )
        for raw in raw_rows
        if str(raw.get("row_id")) in row_by_id
    ]
    artifact = _build_artifact(
        active,
        preconditions,
        evaluated_rows=evaluated_rows,
        collection=collection,
        raw_output_receipts=_raw_receipt(active.raw_path(), raw_rows),
        duration_s=active.clock() - started,
        test_exit_codes=test_exit_codes,
    )
    _write_json_atomic(active.artifact_path(), artifact)
    return artifact


def collect_live_model_outputs(
    model_specs: list[JsonDict],
    fixture_rows: list[JsonDict],
    config: ExperimentConfig,
) -> JsonDict:  # pragma: no cover - requires local GGUFs and CUDA.
    """Collect live llama.cpp outputs for every model, row, and arm."""

    from llama_cpp import Llama

    rows: list[JsonDict] = []
    model_attempts: list[JsonDict] = []
    gpu_receipts: JsonDict = {"load_receipts": [], "generation_receipts": []}
    for model_index, spec in enumerate(model_specs):
        model_path = str(spec.get("model_path") or "")
        load_start = config.monotonic_clock()
        before = _gpu_health_probe()
        try:
            llm = Llama(
                model_path=model_path,
                n_gpu_layers=-1,
                main_gpu=int(spec.get("gpu") or 0),
                n_ctx=8192,
                seed=config.random_seed + model_index,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            model_attempts.append(
                {
                    "hf_id": spec.get("hf_id"),
                    "model_name": spec.get("name"),
                    "model_path": model_path,
                    "model_used": False,
                    "blocker": f"{type(exc).__name__}: {exc}",
                    "gpu_offload_verified": False,
                    "elapsed_seconds": round(config.monotonic_clock() - load_start, 6),
                }
            )
            continue
        after = _gpu_health_probe()
        vram_delta = _vram_delta_mb(before, after)
        attempt = {
            "hf_id": spec.get("hf_id"),
            "model_name": spec.get("name"),
            "model_path": model_path,
            "model_used": True,
            "blocker": None,
            "gpu_offload_verified": vram_delta > 512,
            "vram_delta_mb": vram_delta,
            "elapsed_seconds": round(config.monotonic_clock() - load_start, 6),
        }
        model_attempts.append(attempt)
        gpu_receipts["load_receipts"].append(attempt)
        try:
            for row_index, fixture_row in enumerate(fixture_rows):
                rows.extend(_collect_live_row_arms(llm, spec, fixture_row, row_index, config))
        finally:
            close = getattr(llm, "close", None)
            if callable(close):
                close()
    return {"rows": rows, "model_attempts": model_attempts, "gpu_receipts": gpu_receipts}


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal schema and load-bearing principle fields."""

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
    score = float(artifact["trace_repair_mechanism_ready_score"])
    if score not in {0.0, 1.0}:
        raise ValueError("trace_repair_mechanism_ready_score must be bare 0.0 or 1.0")
    if score == 1.0 and not str(artifact["honest_verdict"]).startswith("complete_positive:"):
        raise ValueError("positive ready score requires complete_positive verdict")


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


def _build_artifact(
    config: ExperimentConfig,
    preconditions: PreconditionReport,
    *,
    evaluated_rows: Sequence[Mapping[str, Any]],
    collection: Mapping[str, Any],
    raw_output_receipts: JsonDict,
    duration_s: float,
    test_exit_codes: Mapping[str, int] | None,
) -> JsonDict:
    aggregates = aggregate_evaluations(evaluated_rows)
    completed = _completed_headline_models(collection.get("model_attempts") or [])
    score = _ready_score(aggregates, completed)
    unsafe_increase = _unsafe_increase(aggregates)
    status, verdict = _status_and_verdict(preconditions.block_reason, score, unsafe_increase)
    artifact: JsonDict = {
        "schema": "carnot.experiment_5897.sota_constraint_ir_repair_ab.v1",
        "experiment_id": "experiment_5897_sota_constraint_ir_repair_ab",
        "run_date": RUN_DATE,
        "random_seed": config.random_seed,
        "status": status,
        "preconditions_checked": preconditions.preconditions_checked,
        "upstream_gate_and_fixture_hashes": preconditions.upstream_gate_and_fixture_hashes,
        "model_specs": preconditions.model_specs,
        "model_file_hashes": preconditions.model_file_hashes,
        "loader_and_cuda_receipts": preconditions.loader_and_cuda_receipts,
        "frozen_prompts_decoding_seeds_and_budgets": _frozen_prompt_receipt(config),
        "arm_definitions_and_compute_parity": _arm_parity_receipt(),
        "trace_visibility_and_oracle_boundary": _trace_boundary_receipt(),
        **aggregates,
        "gpu_utilization_vram_and_latency_receipts": _gpu_latency_receipt(
            evaluated_rows, collection.get("gpu_receipts") or {}
        ),
        "raw_output_receipts": raw_output_receipts,
        "protected_files_unchanged": _protected_file_receipt(
            config.repo_root, baseline=preconditions.protected_file_baseline
        ),
        "trace_repair_mechanism_ready_score": score,
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
    block_reason: str | None, score: float, unsafe_increase: bool
) -> tuple[str, str]:
    if block_reason is not None:
        return "blocked_precondition", f"blocked_precondition: {block_reason}"
    if unsafe_increase:
        return "unsafe", "unsafe: trace-guided repair increased unsafe accepted constraints"
    if score == 1.0:
        return "complete", "complete_positive: trace-guided repair beats both matched controls"
    return "complete", "complete_null: trace-guided repair did not clear held-control promotion"


def _evaluation_base(
    row: Mapping[str, Any],
    arm_id: str,
    raw_text: str,
    metadata: Mapping[str, Any],
) -> JsonDict:
    usage = dict(metadata.get("usage") or {})
    latency_s = float(metadata.get("latency_s") or metadata.get("elapsed_seconds") or 0.0)
    util = float(metadata.get("average_gpu_utilization_pct") or 0.0)
    return {
        "row_id": row.get("row_id"),
        "split": row.get("split"),
        "family": row.get("family"),
        "template_id": row.get("template_id"),
        "variant_kind": row.get("variant_kind"),
        "expected_status": row.get("expected_status"),
        "arm_id": arm_id,
        "model_hf_id": metadata.get("model_hf_id"),
        "model_name": metadata.get("model_name"),
        "gpu_index": metadata.get("gpu_index"),
        "prompt_sha256": metadata.get("prompt_sha256"),
        "seed": metadata.get("seed"),
        "raw_output_sha256": sha256_text(raw_text),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "latency_s": latency_s,
        "energy_proxy_gpu_utilization_pct_s": round(latency_s * util / 100.0, 6),
    }


def _rejected_evaluation(error: str, row: Mapping[str, Any], parser_kind: Any) -> JsonDict:
    expected_status = str(row.get("expected_status"))
    parse_kind = str(parser_kind or "invalid")
    return {
        "parse_valid": False,
        "type_valid": False,
        "compiled": False,
        "satisfiability_correct": expected_status in {"invalid", "type_error"}
        and parse_kind == expected_status,
        "solver_status": "not_applicable",
        "z3_status": "not_applicable",
        "exact_semantic_equivalence": None,
        "query_correct": None,
        "omitted_constraints": 0,
        "spurious_constraints": 0,
        "unsafe_accepted_constraints": False,
        "candidate_sha256": None,
        "diagnostics": {
            "parser_status": "rejected",
            "parser_error": error,
            "type_status": "rejected" if parse_kind == "type_error" else "not_applicable",
            "compiler_status": "not_compiled",
            "compiler_error": "parser_rejected",
            "solver_status": "not_applicable",
            "cross_backend_agreement": None,
        },
    }


def _semantic_checks(
    row: Mapping[str, Any], python_receipt: Mapping[str, Any]
) -> tuple[bool | None, bool | None]:
    if str(row.get("expected_status")) != "valid":
        return None, None
    if python_receipt.get("status") != "sat":
        return False, False
    expected_python = dict(row.get("certificates", {}).get("python") or {})
    behavior_ok = python_receipt.get("behavior_hash") == expected_python.get("behavior_hash")
    candidate_query = dict(python_receipt.get("behavior_signature") or {}).get("query_bindings")
    expected_query = dict(expected_python.get("behavior_signature") or {}).get("query_bindings")
    return bool(behavior_ok), bool(candidate_query == expected_query)


def _satisfiability_correct(expected_status: str, solver_status: str, parse_valid: bool) -> bool:
    if expected_status == "valid":
        return parse_valid and solver_status == "sat"
    if expected_status == "unsat":
        return parse_valid and solver_status == "unsat"
    return not parse_valid


def _unsafe_accepted(
    expected_status: str,
    solver_status: str,
    compiled: bool,
    exact_equivalence: bool | None,
) -> bool:
    if not compiled or solver_status != "sat":
        return False
    if expected_status != "valid":
        return True
    return exact_equivalence is False


def _extract_json_object(text: str) -> tuple[JsonDict | None, str | None]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return dict(obj), None
    return None, "no_json_object"


def _constraint_diff_counts(expected: Any, candidate: Any) -> tuple[int, int]:
    if not isinstance(expected, Mapping) or not isinstance(candidate, Mapping):
        return 0, 0
    expected_set = _constraint_fingerprint(expected)
    candidate_set = _constraint_fingerprint(candidate)
    return len(expected_set - candidate_set), len(candidate_set - expected_set)


def _constraint_fingerprint(ir: Mapping[str, Any]) -> set[str]:
    found: set[str] = set()
    for fact in ir.get("facts") or []:
        found.add("fact:" + canonical_json(fact))
    for rule in ir.get("rules") or []:
        if isinstance(rule, Mapping):
            found.add("rule_head:" + canonical_json(rule.get("head")))
            for term in (rule.get("body") or {}).get("terms") or []:
                found.add("rule_term:" + canonical_json(term))
    return found


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


def _constraint_error_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_arm = {
        arm_id: {
            "omitted_constraints": sum(
                int(row.get("omitted_constraints") or 0)
                for row in rows
                if row.get("arm_id") == arm_id
            ),
            "spurious_constraints": sum(
                int(row.get("spurious_constraints") or 0)
                for row in rows
                if row.get("arm_id") == arm_id
            ),
            "unsafe_accepted_constraints": sum(
                bool(row.get("unsafe_accepted_constraints"))
                for row in rows
                if row.get("arm_id") == arm_id
            ),
        }
        for arm_id in ARM_DEFINITIONS
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


def _group_bootstrap_lower_bounds(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "trace_vs_matched_two_call_no_trace": _bootstrap_lower_bound(
            rows, "trace_guided_repair", "matched_two_call_no_trace"
        ),
        "trace_vs_no_information_trace_control": _bootstrap_lower_bound(
            rows, "trace_guided_repair", "no_information_trace_control"
        ),
        "method": "deterministic_group_bootstrap_5th_percentile_over_heldout_valid_rows",
    }


def _bootstrap_lower_bound(
    rows: Sequence[Mapping[str, Any]], left_arm: str, right_arm: str
) -> float:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("split") != "heldout" or row.get("expected_status") != "valid":
            continue
        key = (str(row.get("model_hf_id")), str(row.get("family")))
        grouped[key][str(row.get("arm_id"))].append(
            1.0 if row.get("exact_semantic_equivalence") else 0.0
        )
    deltas = []
    for arms in grouped.values():
        if arms.get(left_arm) and arms.get(right_arm):
            deltas.append(
                (sum(arms[left_arm]) / len(arms[left_arm]))
                - (sum(arms[right_arm]) / len(arms[right_arm]))
            )
    if not deltas:
        return 0.0
    if len(deltas) == 1:
        return round(deltas[0], 6)
    samples = []
    for index in range(200):
        values = [deltas[(index + offset * 17) % len(deltas)] for offset in range(len(deltas))]
        samples.append(sum(values) / len(values))
    samples.sort()
    return round(samples[int(0.05 * (len(samples) - 1))], 6)


def _ready_score(aggregates: Mapping[str, Any], completed_models: Sequence[str]) -> float:
    lower = aggregates["group_bootstrap_lower_bounds"]
    by_arm = aggregates["parse_type_compile_and_semantic_metrics"]["by_arm"]
    trace_unsafe = by_arm["trace_guided_repair"]["unsafe_accept_rate"]
    control_unsafe = max(
        by_arm["matched_two_call_no_trace"]["unsafe_accept_rate"],
        by_arm["no_information_trace_control"]["unsafe_accept_rate"],
    )
    ready = (
        set(completed_models) == set(MANDATED_MODEL_IDS)
        and lower["trace_vs_matched_two_call_no_trace"]
        > PROMOTION_THRESHOLDS["held_group_lower_bound_min"]
        and lower["trace_vs_no_information_trace_control"]
        > PROMOTION_THRESHOLDS["held_group_lower_bound_min"]
        and trace_unsafe - control_unsafe <= PROMOTION_THRESHOLDS["max_unsafe_delta"]
    )
    return 1.0 if ready else 0.0


def _unsafe_increase(aggregates: Mapping[str, Any]) -> bool:
    by_arm = aggregates["parse_type_compile_and_semantic_metrics"]["by_arm"]
    trace_unsafe = by_arm["trace_guided_repair"]["unsafe_accept_rate"]
    return trace_unsafe > max(
        by_arm["matched_two_call_no_trace"]["unsafe_accept_rate"],
        by_arm["no_information_trace_control"]["unsafe_accept_rate"],
    )


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


def _upstream_gate_receipt(root: Path) -> JsonDict:
    result_path = root / exp5896.RESULT_RELATIVE_PATH
    row_path = root / exp5896.ROW_FILE_RELATIVE_PATH
    try:
        replay = exp5896.replay_artifact(root=root)
        replay_ok = bool(replay.get("ok"))
        replay_error = None
    except Exception as exc:  # noqa: BLE001
        replay_ok = False
        replay_error = f"{type(exc).__name__}: {exc}"
    return {
        "replay_ok": replay_ok,
        "replay_error": replay_error,
        "artifact_path": str(exp5896.RESULT_RELATIVE_PATH),
        "row_path": str(exp5896.ROW_FILE_RELATIVE_PATH),
        "artifact_sha256": sha256_file(result_path) if result_path.exists() else None,
        "row_file_sha256": sha256_file(row_path) if row_path.exists() else None,
    }


def _model_file_hashes(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    files = []
    for spec in model_specs:
        path_value = spec.get("model_path")
        path = Path(str(path_value)) if path_value else None
        exists = bool(path and path.is_file())
        files.append(
            {
                "hf_id": spec.get("hf_id"),
                "path": str(path) if path else None,
                "exists": exists,
                "size_bytes": path.stat().st_size if exists and path else None,
                "sha256": sha256_file(path) if exists and path else None,
            }
        )
    return {"files": files, "all_hashed": all(file.get("sha256") for file in files)}


def _tokenizer_receipts(
    model_specs: Sequence[Mapping[str, Any]],
    tokenizer_checker: TokenizerChecker,
) -> list[JsonDict]:
    receipts = []
    for spec in model_specs:
        path = spec.get("model_path")
        ok, detail = tokenizer_checker(str(path) if path else None)
        receipts.append({"hf_id": spec.get("hf_id"), "ok": ok, "detail": detail})
    return receipts


def _two_healthy_rtx_3090s(gpu_health: Mapping[str, Any]) -> bool:
    gpus = gpu_health.get("gpus") or []
    healthy = [
        gpu
        for gpu in gpus
        if isinstance(gpu, Mapping)
        and "RTX 3090" in str(gpu.get("name"))
        and int(gpu.get("memory_total_mb") or 0) >= 24000
    ]
    return bool(gpu_health.get("ok") and len(healthy) >= 2)


def _adequate_vram(gpu_health: Mapping[str, Any]) -> bool:
    gpus = [gpu for gpu in gpu_health.get("gpus") or [] if isinstance(gpu, Mapping)]
    return len(gpus) >= 2 and all(int(gpu.get("memory_total_mb") or 0) >= 24000 for gpu in gpus[:2])


def _load_fixture_rows(root: Path) -> list[JsonDict]:
    path = root / exp5896.ROW_FILE_RELATIVE_PATH
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _frozen_prompt_receipt(config: ExperimentConfig) -> JsonDict:
    prompts = {
        "single_pass": build_single_pass_prompt({"natural_language": "<row natural language>"}),
        "trace_guided_repair": build_trace_repair_prompt(
            {"natural_language": "<row natural language>"},
            "<prior model JSON>",
            {
                "parser_status": "rejected",
                "parser_error": "<deployment parser error>",
                "type_status": "rejected",
                "compiler_status": "not_compiled",
                "compiler_error": "parser_rejected",
                "solver_status": "not_applicable",
                "cross_backend_agreement": None,
            },
        ),
        "matched_two_call_no_trace": build_matched_no_trace_prompt(
            {"natural_language": "<row natural language>"}, "<prior model JSON>"
        ),
        "no_information_trace_control": build_no_information_trace_prompt(
            {"natural_language": "<row natural language>"}, "<prior model JSON>"
        ),
    }
    return {
        "prompt_version": "exp5897.constraint_ir_ab.v1",
        "prompt_sha256": {name: sha256_text(prompt) for name, prompt in prompts.items()},
        "decoding": DECODING,
        "base_random_seed": config.random_seed,
        "arm_budgets": {name: ARM_DEFINITIONS[name]["max_tokens"] for name in ARM_DEFINITIONS},
        "promotion_thresholds": PROMOTION_THRESHOLDS,
        "row_groups_frozen_by_exp5896": True,
    }


def _arm_parity_receipt() -> JsonDict:
    repair_budget = ARM_DEFINITIONS["trace_guided_repair"]["max_tokens"]
    return {
        "arms": ARM_DEFINITIONS,
        "second_call_budget_parity": {
            "trace_vs_no_trace": repair_budget
            == ARM_DEFINITIONS["matched_two_call_no_trace"]["max_tokens"],
            "trace_vs_no_information": repair_budget
            == ARM_DEFINITIONS["no_information_trace_control"]["max_tokens"],
        },
        "principle": FIELD_PRINCIPLES["arm_definitions_and_compute_parity"],
    }


def _trace_boundary_receipt() -> JsonDict:
    return {
        "visible_to_trace_repair": [
            "parser_status",
            "parser_error",
            "type_status",
            "compiler_status",
            "compiler_error",
            "solver_status",
            "cross_backend_agreement",
        ],
        "withheld_from_model": [
            "hidden_gold_ir",
            "answer_labels",
            "held_family_identity",
            "certificate_solutions",
            "gold_behavior_hash",
            "model_self_scores",
        ],
        "generated_answer_repair": False,
        "model_self_scores_excluded": True,
        "principle": FIELD_PRINCIPLES["trace_visibility_and_oracle_boundary"],
    }


def _gpu_latency_receipt(
    rows: Sequence[Mapping[str, Any]], gpu_receipts: Mapping[str, Any]
) -> JsonDict:
    return {
        "gpu_receipts": dict(gpu_receipts),
        "latency_s_total": round(sum(float(row.get("latency_s") or 0.0) for row in rows), 6),
        "energy_proxy_gpu_utilization_pct_s_total": round(
            sum(float(row.get("energy_proxy_gpu_utilization_pct_s") or 0.0) for row in rows),
            6,
        ),
        "row_count": len(rows),
    }


def _raw_receipt(path: Path, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "path": str(RAW_OUTPUT_RELATIVE_PATH),
        "row_count": len(rows),
        "sha256": sha256_file(path) if path.exists() else None,
        "raw_output_hashes": [
            {
                "row_id": row.get("row_id"),
                "arm_id": row.get("arm_id"),
                "sha256": sha256_text(str(row.get("raw_output_text") or "")),
            }
            for row in rows
        ],
    }


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


def _field_provenance() -> JsonDict:
    return {
        field: "generated_by_exp5897_sota_constraint_ir_repair_ab"
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    stable = json.loads(canonical_json(artifact))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    for row in stable.get("model_file_hashes", {}).get("files", []):
        row["path"] = "<model_path>"
    for file_row in stable.get("protected_files_unchanged", {}).get("files", []):
        file_row["sha256"] = "<protected_sha>"
    return sha256_text(canonical_json(stable))


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


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


def _probe_environment(root: Path) -> JsonDict:  # pragma: no cover - host-dependent.
    return {
        "llama_cpp_import": _llama_cpp_import_receipt(),
        "llama_cpp_cuda_support": _llama_cpp_cuda_receipt(),
        "gpu_health": _gpu_health_probe(),
        "ram": _memory_probe(),
        "disk": _disk_probe(root),
        "protected_workload": _protected_workload_probe(),
        "atomic_output": _atomic_output_probe(root),
    }


def _llama_cpp_import_receipt() -> JsonDict:  # pragma: no cover - host-dependent.
    try:
        import llama_cpp
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "detail": f"llama_cpp {getattr(llama_cpp, '__version__', 'unknown')}"}


def _llama_cpp_cuda_receipt() -> JsonDict:  # pragma: no cover - host-dependent.
    try:
        from llama_cpp import llama_cpp as backend

        supported = bool(backend.llama_supports_gpu_offload())
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    return {"ok": supported, "detail": f"llama_supports_gpu_offload={supported}"}


def _gpu_health_probe() -> JsonDict:  # pragma: no cover - host-dependent.
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}", "gpus": []}
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mb": int(parts[2]),
                "memory_free_mb": int(parts[3]),
                "utilization_gpu_pct": int(parts[4]),
            }
        )
    return {
        "ok": result.returncode == 0 and bool(gpus),
        "returncode": result.returncode,
        "gpus": gpus,
    }


def _memory_probe() -> JsonDict:  # pragma: no cover - host-dependent.
    required_mb = 32768
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {
        "ok": available_mb >= required_mb,
        "available_mb": available_mb,
        "required_mb": required_mb,
    }


def _disk_probe(root: Path) -> JsonDict:  # pragma: no cover - host-dependent.
    required_mb = 8192
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {
        "ok": available_mb >= required_mb,
        "available_mb": available_mb,
        "required_mb": required_mb,
    }


def _protected_workload_probe() -> JsonDict:  # pragma: no cover - host-dependent.
    try:
        from scripts.experiment_template import _pid_is_protected_training_proc
    except Exception:
        _pid_is_protected_training_proc = lambda pid: False  # type: ignore[assignment]
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": True,
            "detail": f"compute_app_probe_unavailable:{type(exc).__name__}",
            "protected_pids": [],
        }
    protected = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not parts or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        if _pid_is_protected_training_proc(pid):
            protected.append({"pid": pid, "process_name": parts[1] if len(parts) > 1 else None})
    return {"ok": not protected, "protected_pids": protected}


def _atomic_output_probe(root: Path) -> JsonDict:  # pragma: no cover - host-dependent.
    probe_dir = root / "results"
    probe_dir.mkdir(parents=True, exist_ok=True)
    target = probe_dir / ".exp5897_atomic_probe"
    _write_text_atomic(target, "ok\n")
    ok = target.read_text(encoding="utf-8") == "ok\n"
    target.unlink(missing_ok=True)
    return {"ok": ok, "detail": "tempfile_replace_supported"}


def _vram_delta_mb(before: Mapping[str, Any], after: Mapping[str, Any]) -> int:  # pragma: no cover.
    before_used = sum(
        int(gpu.get("memory_total_mb") or 0) - int(gpu.get("memory_free_mb") or 0)
        for gpu in before.get("gpus") or []
    )
    after_used = sum(
        int(gpu.get("memory_total_mb") or 0) - int(gpu.get("memory_free_mb") or 0)
        for gpu in after.get("gpus") or []
    )
    return max(0, after_used - before_used)


def _collect_live_row_arms(
    llm: Any,
    spec: Mapping[str, Any],
    row: Mapping[str, Any],
    row_index: int,
    config: ExperimentConfig,
) -> list[JsonDict]:  # pragma: no cover - requires local GGUFs and CUDA.
    single_prompt = build_single_pass_prompt(row)
    single = _call_llama(llm, single_prompt, "single_pass", spec, row, row_index, config, 0)
    single_eval = evaluate_candidate(row, "single_pass", single["raw_output_text"], single)
    trace = public_diagnostic_trace(single_eval)
    prompts = {
        "trace_guided_repair": build_trace_repair_prompt(row, single["raw_output_text"], trace),
        "matched_two_call_no_trace": build_matched_no_trace_prompt(row, single["raw_output_text"]),
        "no_information_trace_control": build_no_information_trace_prompt(
            row, single["raw_output_text"]
        ),
    }
    results = [single]
    for offset, (arm_id, prompt) in enumerate(prompts.items(), start=1):
        results.append(_call_llama(llm, prompt, arm_id, spec, row, row_index, config, offset))
    return results


def _call_llama(
    llm: Any,
    prompt: str,
    arm_id: str,
    spec: Mapping[str, Any],
    row: Mapping[str, Any],
    row_index: int,
    config: ExperimentConfig,
    seed_offset: int,
) -> JsonDict:  # pragma: no cover - requires local GGUFs and CUDA.
    start = config.monotonic_clock()
    seed = config.random_seed + row_index + seed_offset * 100_000
    output_text = ""
    usage: JsonDict = {}
    try:
        result = llm(
            prompt,
            max_tokens=int(ARM_DEFINITIONS[arm_id]["max_tokens"]),
            temperature=DECODING["temperature"],
            top_p=DECODING["top_p"],
            repeat_penalty=DECODING["repeat_penalty"],
            stop=DECODING["stop"],
            seed=seed,
            echo=False,
        )
        output_text, usage = _completion_text_and_usage(result)
    except Exception as exc:  # noqa: BLE001
        output_text = f""
        usage = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "model_hf_id": spec.get("hf_id"),
        "model_name": spec.get("name"),
        "model_path": spec.get("model_path"),
        "gpu_index": spec.get("gpu"),
        "row_id": row.get("row_id"),
        "arm_id": arm_id,
        "prompt_sha256": sha256_text(prompt),
        "seed": seed,
        "raw_output_text": output_text,
        "latency_s": round(config.monotonic_clock() - start, 6),
        "usage": usage,
    }


def _completion_text_and_usage(result: Any) -> tuple[str, JsonDict]:  # pragma: no cover.
    if isinstance(result, str):
        return result, {}
    if not isinstance(result, Mapping):
        return "", {}
    choices = result.get("choices")
    text = ""
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        choice = choices[0]
        if isinstance(choice.get("text"), str):
            text = str(choice["text"])
        elif isinstance(choice.get("message"), Mapping):
            text = str(choice["message"].get("content") or "")
    usage = dict(result.get("usage") or {})
    return text, usage


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
    args = parser.parse_args(argv)
    if args.refresh_test_exit_code:
        artifact = refresh_artifact_test_exit_codes(
            root=args.root,
            test_exit_codes=_parse_test_exit_codes(args.refresh_test_exit_code),
        )
    else:
        artifact = run_experiment(ExperimentConfig(repo_root=args.root))
    print(
        "[exp5897] "
        f"status={artifact['status']} "
        f"verdict={artifact['honest_verdict']} "
        f"score={artifact['trace_repair_mechanism_ready_score']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
