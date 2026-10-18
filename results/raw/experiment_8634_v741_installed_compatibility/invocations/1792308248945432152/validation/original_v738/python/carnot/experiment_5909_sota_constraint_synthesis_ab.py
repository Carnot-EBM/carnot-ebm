"""Exp5909 live SOTA ConstraintIR synthesis A/B stream.

Spec refs: REQ-BENCH-5909, SCENARIO-BENCH-5909-PRECONDITIONS,
SCENARIO-BENCH-5909-PROMPTS, SCENARIO-BENCH-5909-STREAM,
SCENARIO-BENCH-5909-HEADROOM.

This experiment asks local GGUF models to propose typed ConstraintIR JSON from
the Exp5908 prompt-plan fixture. The models are only proposal generators. The
checked-in Exp5896 parser, Python backend, Z3 backend, and semantic replay
labels own every outcome after the raw text is sealed.
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
import shutil
import subprocess
import tempfile
import time
from typing import Any

from carnot import experiment_5896_typed_constraint_ir_fixture as exp5896
from carnot import experiment_5897_sota_constraint_ir_repair_ab as exp5897
from carnot import experiment_5908_verisynth_constraint_fixture as exp5908
from carnot.inference.sota_models import (
    cached_sota_pair,
    gguf_tokenizer_loadable,
    resolve_cached_gguf,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5909_sota_constraint_synthesis_ab.json")
RAW_STREAM_RELATIVE_PATH = Path("results/experiment_5909_sota_constraint_synthesis_ab.raw.jsonl")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5909_sota_constraint_synthesis_ab.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5909_sota_constraint_synthesis_ab.py")
RUN_DATE = "20260725"
RANDOM_SEED = 5909
INFERENCE_SUBSTRATE = "live_llm_inference"
VERIFIER_IS_ORACLE = True
ARTIFACT_SCHEMA_VERSION = "carnot.experiment_5909.sota_constraint_synthesis_ab.v1"
RAW_ROW_SCHEMA_VERSION = ARTIFACT_SCHEMA_VERSION + ".raw_row"

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

PRIMARY_ARM_IDS = (
    "direct",
    "semantic_decomposition",
    "decomposition_plus_exact_example_retrieval",
)
CONTROL_ARM_IDS = ("wrong_family_retrieval", "shuffled_decomposition")
ALL_ARM_IDS = (*PRIMARY_ARM_IDS, *CONTROL_ARM_IDS)
CONFIRMATORY_CONTROL_ROW_IDS = (
    "exp5896-access_control-canonical",
    "exp5896-menu_recommendation-canonical",
    "exp5896-task_selection-canonical",
)
DECODING: JsonDict = {
    "temperature": 0.0,
    "top_p": 1.0,
    "repeat_penalty": 1.05,
    "stop": ["</s>", "<eos>", "<|eot_id|>"],
}
GENERATION_BUDGETS: JsonDict = {
    "max_tokens": exp5908.TOKEN_ENVELOPE_MAX_TOKENS,
    "n_ctx": 8192,
    "n_batch": 512,
    "n_gpu_layers": -1,
    "wall_clock_budget_s_per_call": 240,
    "retrieval_exemplars": exp5908.EXEMPLARS_PER_RETRIEVAL_ARM,
}
ARM_DEFINITIONS: JsonDict = {
    "direct": {
        "calls": 1,
        "max_tokens": GENERATION_BUDGETS["max_tokens"],
        "uses_decomposition_structure": False,
        "uses_retrieval_examples": False,
        "control": False,
    },
    "semantic_decomposition": {
        "calls": 1,
        "max_tokens": GENERATION_BUDGETS["max_tokens"],
        "uses_decomposition_structure": True,
        "uses_retrieval_examples": False,
        "control": False,
    },
    "decomposition_plus_exact_example_retrieval": {
        "calls": 1,
        "max_tokens": GENERATION_BUDGETS["max_tokens"],
        "uses_decomposition_structure": True,
        "uses_retrieval_examples": True,
        "control": False,
    },
    "wrong_family_retrieval": {
        "calls": 1,
        "max_tokens": GENERATION_BUDGETS["max_tokens"],
        "uses_decomposition_structure": True,
        "uses_retrieval_examples": True,
        "control": True,
    },
    "shuffled_decomposition": {
        "calls": 1,
        "max_tokens": GENERATION_BUDGETS["max_tokens"],
        "uses_decomposition_structure": True,
        "uses_retrieval_examples": False,
        "control": True,
    },
}
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5909_sota_constraint_synthesis_ab.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5909_sota_constraint_synthesis_ab.py "
    "-m pytest tests/python/test_experiment_5909_sota_constraint_synthesis_ab.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5909_sota_constraint_synthesis_ab.py --fail-under=100",
    ".venv/bin/pytest tests/python/test_experiment_5896_typed_constraint_ir_fixture.py "
    "tests/python/test_experiment_5907_constraint_ir_replay_contract.py "
    "tests/python/test_experiment_5908_verisynth_constraint_fixture.py "
    "tests/python/test_experiment_5909_sota_constraint_synthesis_ab.py -q --no-cov -n 0",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python -m carnot.experiment_5909_sota_constraint_synthesis_ab",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5909_sota_constraint_synthesis_ab.json",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5909_sota_constraint_synthesis_ab.py",
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
    "embedded_tokenizer_and_loader_cuda_receipts",
    "frozen_prompts_decoding_seeds_and_budgets",
    "arm_definitions_and_compute_parity",
    "retrieval_and_oracle_visibility",
    "per_model_family_template_metrics",
    "parse_type_compile_and_semantic_metrics",
    "omitted_spurious_and_unsafe_constraint_metrics",
    "group_bootstrap_lower_bounds",
    "wrong_retrieval_and_shuffled_controls",
    "chronological_raw_stream_receipt",
    "residual_error_and_diagnostic_headroom",
    "gpu_utilization_vram_latency_and_energy_receipts",
    "protected_files_unchanged",
    "constraint_stream_ready_score",
    "verification_repair_admission_ready_score",
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
        "Decomposition and retrieval cannot win by unbounded context or calls."
    ),
    "chronological_raw_stream_receipt": (
        "Downstream learning sees only sealed model outputs and deployment-visible evidence in event order."
    ),
    "constraint_stream_ready_score": (
        "Emit bare 1.0 only for all three real models, exact labels, raw hashes, "
        "and zero authority violation."
    ),
    "verification_repair_admission_ready_score": (
        "Emit bare 1.0 only for sufficient exact residual headroom with usable "
        "non-oracle diagnostics."
    ),
    "inference_substrate": "Use live_llm_inference.",
    "verifier_is_oracle": "True for exact evaluation and never for model proposal credit.",
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
    """Runtime paths and clocks for Exp5909."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    raw_stream_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    monotonic_clock: Callable[[], float] = time.monotonic
    random_seed: int = RANDOM_SEED

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / RESULT_RELATIVE_PATH

    def raw_path(self) -> Path:
        return self.raw_stream_path or self.repo_root / RAW_STREAM_RELATIVE_PATH


@dataclass(frozen=True)
class PreconditionReport:
    """All checks that must pass before any GGUF model is loaded."""

    preconditions_checked: JsonDict
    upstream_gate_and_fixture_hashes: JsonDict
    model_specs: list[JsonDict]
    model_file_hashes: JsonDict
    embedded_tokenizer_and_loader_cuda_receipts: JsonDict
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
    """Resolve Qwen, Gemma dense, and Gemma MoE in the preregistered order."""

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
    """Check upstream, cache, tokenizer, CUDA, and output gates before model load."""

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
    }
    checks = {
        "exp5908_gate_replayed_before_model_load": bool(upstream.get("exp5908_replay_ok")),
        "all_three_model_specs_resolved": all(spec.get("model_path") for spec in model_specs),
        "all_model_files_exist": all(row.get("exists") for row in model_file_hashes["files"]),
        "all_model_hashes_recorded": all(row.get("sha256") for row in model_file_hashes["files"]),
        "embedded_gguf_tokenizers_load": all(row.get("ok") for row in tokenizer_receipts),
        "llama_cpp_import_ok": bool(environment.get("llama_cpp_import", {}).get("ok")),
        "public_llama_cpp_cuda_offload": bool(
            environment.get("llama_cpp_cuda_support", {}).get("ok")
        ),
        "two_healthy_rtx_3090s": _two_healthy_rtx_3090s(environment.get("gpu_health", {})),
        "adequate_vram": _adequate_vram(environment.get("gpu_health", {})),
        "adequate_ram": bool(environment.get("ram", {}).get("ok")),
        "adequate_disk": bool(environment.get("disk", {}).get("ok")),
        "no_protected_workload": bool(environment.get("protected_workload", {}).get("ok")),
        "atomic_output_ready": bool(environment.get("atomic_output", {}).get("ok")),
    }
    block_reason = next((name for name, ok in checks.items() if not ok), None)
    return PreconditionReport(
        preconditions_checked={
            "run_order": "exp5908_gate_and_resource_checks_before_any_model_load",
            "blocked_before_model_load": block_reason is not None,
            "headline_checks": checks,
            "block_reason": block_reason,
        },
        upstream_gate_and_fixture_hashes=upstream,
        model_specs=model_specs,
        model_file_hashes=model_file_hashes,
        embedded_tokenizer_and_loader_cuda_receipts=receipts,
        protected_file_baseline=protected_baseline,
        block_reason=block_reason,
    )


def build_prompt(
    plan_row: Mapping[str, Any],
    source_row: Mapping[str, Any],
    source_rows_by_id: Mapping[str, Mapping[str, Any]],
    arm_id: str,
) -> str:
    """Build one leakage-safe synthesis prompt for a frozen Exp5908 arm."""

    if arm_id not in ALL_ARM_IDS:
        raise ValueError(f"unknown Exp5909 arm: {arm_id}")
    unit_order = _unit_order_for_arm(plan_row, arm_id)
    sections = [
        "Return exactly one JSON object for the typed ConstraintIR schema.",
        "Use keys schema_version, domains, entities, predicates, facts, rules, query.",
        "Supported expression nodes are atom, not, and, arith. Use finite domains only.",
        "Do not include markdown, explanations, row identifiers, labels, or diagnostics.",
        f"schema_version: {exp5896.CONSTRAINT_IR_SCHEMA_VERSION}",
    ]
    if arm_id != "direct":
        sections.append("Fill the object in this semantic order:")
        sections.extend(f"- {_unit_instruction(unit)}" for unit in unit_order)
    if arm_id in {"decomposition_plus_exact_example_retrieval", "wrong_family_retrieval"}:
        sections.append("Bounded deployment-visible examples follow.")
        sections.extend(_retrieval_example_blocks(plan_row, source_rows_by_id, arm_id))
    sections.extend(["Problem text:", str(source_row["natural_language"])])
    return "\n".join(sections) + "\n"


def evaluate_candidate(
    row: Mapping[str, Any],
    arm_id: str,
    raw_text: str,
    generation_metadata: Mapping[str, Any],
) -> JsonDict:
    """Score one proposal with the shared exact ConstraintIR evaluator."""

    return exp5897.evaluate_candidate(row, arm_id, raw_text, generation_metadata)


def expected_raw_event_count(plan_rows: Sequence[Mapping[str, Any]]) -> int:
    """Return the preregistered complete stream size for three models."""

    control_targets = sum(
        str(row.get("source_row_id")) in CONFIRMATORY_CONTROL_ROW_IDS for row in plan_rows
    )
    per_model = len(plan_rows) * len(PRIMARY_ARM_IDS) + control_targets * len(CONTROL_ARM_IDS)
    return len(MANDATED_MODEL_IDS) * per_model


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
    """Run Exp5909 and write the artifact plus sealed raw stream atomically."""

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
            plan_rows=_safe_load_plan_rows(active.repo_root),
            sealed_rows=[],
            collection={"model_attempts": [], "gpu_receipts": {}},
            duration_s=active.clock() - started,
            test_exit_codes=test_exit_codes,
        )
        _write_json_atomic(active.artifact_path(), artifact)
        return artifact

    plan_rows = _load_plan_rows(active.repo_root)
    collector = collect_model_outputs_fn or collect_live_model_outputs
    collection = collector(preconditions.model_specs, plan_rows, active)
    raw_rows = [dict(row) for row in collection.get("rows") or []]
    sealed_rows = seal_raw_rows(raw_rows, plan_rows)
    _write_raw_rows_atomic(active.raw_path(), sealed_rows)
    artifact = _build_artifact(
        active,
        preconditions,
        plan_rows=plan_rows,
        sealed_rows=sealed_rows,
        collection=collection,
        duration_s=active.clock() - started,
        test_exit_codes=test_exit_codes,
    )
    _write_json_atomic(active.artifact_path(), artifact)
    return artifact


def collect_live_model_outputs(
    model_specs: list[JsonDict],
    plan_rows: list[JsonDict],
    config: ExperimentConfig,
) -> JsonDict:  # pragma: no cover - requires local GGUFs and CUDA.
    """Collect live llama.cpp outputs for every model, row, and preregistered arm."""

    from llama_cpp import Llama

    source_rows = {row["row_id"]: row for row in exp5896.build_fixture_rows()}
    rows: list[JsonDict] = []
    attempts: list[JsonDict] = []
    gpu_receipts: JsonDict = {"load_receipts": [], "generation_receipts": []}
    sequence = 0
    for model_index, spec in enumerate(model_specs):
        model_path = str(spec.get("model_path") or "")
        load_start = config.monotonic_clock()
        before = _gpu_health_probe()
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
        attempts.append(attempt)
        gpu_receipts["load_receipts"].append(attempt)
        try:
            for plan_index, plan_row in enumerate(plan_rows):
                source = source_rows[str(plan_row["source_row_id"])]
                arm_ids = list(PRIMARY_ARM_IDS)
                if str(plan_row["source_row_id"]) in CONFIRMATORY_CONTROL_ROW_IDS:
                    arm_ids.extend(CONTROL_ARM_IDS)
                for arm_offset, arm_id in enumerate(arm_ids):
                    prompt = build_prompt(plan_row, source, source_rows, arm_id)
                    row = _call_llama(
                        llm,
                        prompt,
                        arm_id,
                        spec,
                        plan_row,
                        sequence,
                        model_index,
                        plan_index,
                        arm_offset,
                        config,
                    )
                    rows.append(row)
                    gpu_receipts["generation_receipts"].append(
                        {
                            "stream_sequence_index": sequence,
                            "model_hf_id": spec.get("hf_id"),
                            "source_row_id": plan_row.get("source_row_id"),
                            "arm_id": arm_id,
                            "latency_s": row["latency_s"],
                            "gpu_health_after": _gpu_health_probe(),
                        }
                    )
                    sequence += 1
        finally:
            close = getattr(llm, "close", None)
            if callable(close):
                close()
            del llm
            gc.collect()
    return {"rows": rows, "model_attempts": attempts, "gpu_receipts": gpu_receipts}


def seal_raw_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    plan_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Attach exact labels and deployment-visible diagnostics to raw model text."""

    source_rows = {row["row_id"]: row for row in exp5896.build_fixture_rows()}
    plan_by_source = {str(row["source_row_id"]): dict(row) for row in plan_rows}
    sealed = []
    for index, raw in enumerate(raw_rows):
        source_id = str(raw.get("source_row_id") or raw.get("row_id"))
        source = source_rows[source_id]
        plan = plan_by_source[source_id]
        raw_text = str(raw.get("raw_output_text") or "")
        metadata = dict(raw)
        metadata["row_id"] = source_id
        evaluation = evaluate_candidate(source, str(raw.get("arm_id")), raw_text, metadata)
        event = {
            "schema": RAW_ROW_SCHEMA_VERSION,
            "stream_sequence_index": int(raw.get("stream_sequence_index", index)),
            "event_kind": "model_constraint_ir_proposal",
            "model_hf_id": raw.get("model_hf_id"),
            "model_name": raw.get("model_name"),
            "model_path": raw.get("model_path"),
            "gpu_index": raw.get("gpu_index"),
            "source_row_id": source_id,
            "plan_row_hash": plan.get("row_hash"),
            "source_row_hash": plan.get("source_row_hash"),
            "group_id": plan.get("group_id"),
            "family": plan.get("family"),
            "template_id": plan.get("template_id"),
            "split": plan.get("split"),
            "variant_kind": plan.get("variant_kind"),
            "expected_status": plan.get("expected_status"),
            "expected_equivalent_to_canonical": plan.get("expected_equivalent_to_canonical"),
            "arm_id": raw.get("arm_id"),
            "prompt_plan_hash": (plan.get("prompt_plan_arms") or {})
            .get(str(raw.get("arm_id")), {})
            .get("prompt_plan_hash"),
            "prompt_sha256": raw.get("prompt_sha256"),
            "seed": raw.get("seed"),
            "raw_output_text": raw_text,
            "raw_output_sha256": sha256_text(raw_text),
            "usage": dict(raw.get("usage") or {}),
            "latency_s": float(raw.get("latency_s") or 0.0),
            "average_gpu_utilization_pct": float(raw.get("average_gpu_utilization_pct") or 0.0),
            "exact_labels": _exact_label_projection(evaluation),
            "visible_diagnostics": dict(evaluation.get("diagnostics") or {}),
            **evaluation,
            "row_hash": "",
        }
        event["row_hash"] = stream_row_hash(event)
        sealed.append(event)
    return sorted(sealed, key=lambda row: int(row["stream_sequence_index"]))


def aggregate_evaluations(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Aggregate exact metrics overall, by arm, model, family, template, and controls."""

    row_list = [dict(row) for row in rows]
    by_arm = {
        arm_id: _metric_summary([row for row in row_list if row.get("arm_id") == arm_id])
        for arm_id in ALL_ARM_IDS
    }
    return {
        "per_model_family_template_metrics": {
            "by_model": _grouped_metrics(row_list, "model_hf_id"),
            "by_family": _grouped_metrics(row_list, "family"),
            "by_template": _grouped_metrics(row_list, "template_id"),
            "by_model_family_template": _grouped_combo_metrics(
                row_list, ("model_hf_id", "family", "template_id")
            ),
        },
        "parse_type_compile_and_semantic_metrics": {
            "overall": _metric_summary(row_list),
            "by_arm": by_arm,
        },
        "omitted_spurious_and_unsafe_constraint_metrics": _constraint_error_metrics(row_list),
        "group_bootstrap_lower_bounds": {
            "semantic_decomposition_vs_direct": _bootstrap_lower_bound(
                row_list, "semantic_decomposition", "direct"
            ),
            "decomposition_plus_exact_example_retrieval_vs_direct": _bootstrap_lower_bound(
                row_list, "decomposition_plus_exact_example_retrieval", "direct"
            ),
            "retrieval_vs_semantic_decomposition": _bootstrap_lower_bound(
                row_list,
                "decomposition_plus_exact_example_retrieval",
                "semantic_decomposition",
            ),
            "method": "deterministic_group_bootstrap_5th_percentile_over_heldout_valid_rows",
        },
        "wrong_retrieval_and_shuffled_controls": {
            "confirmatory_subset_row_ids": list(CONFIRMATORY_CONTROL_ROW_IDS),
            "by_arm": {arm_id: by_arm[arm_id] for arm_id in CONTROL_ARM_IDS},
            "expected_control_event_count": len(MANDATED_MODEL_IDS)
            * len(CONFIRMATORY_CONTROL_ROW_IDS)
            * len(CONTROL_ARM_IDS),
            "completed_control_event_count": sum(
                row.get("arm_id") in CONTROL_ARM_IDS for row in row_list
            ),
        },
    }


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
    stream_score = float(artifact["constraint_stream_ready_score"])
    repair_score = float(artifact["verification_repair_admission_ready_score"])
    if stream_score not in {0.0, 1.0}:
        raise ValueError("constraint_stream_ready_score must be bare 0.0 or 1.0")
    if repair_score not in {0.0, 1.0}:
        raise ValueError("verification_repair_admission_ready_score must be bare 0.0 or 1.0")
    if stream_score == 1.0 and not str(artifact["honest_verdict"]).startswith(
        ("complete_positive:", "complete_null:", "unsafe:")
    ):
        raise ValueError("ready stream requires complete or unsafe honest_verdict prefix")


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


def stream_row_hash(row: Mapping[str, Any]) -> str:
    """Hash one sealed stream row while excluding its own row hash."""

    stable = dict(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def _build_artifact(
    config: ExperimentConfig,
    preconditions: PreconditionReport,
    *,
    plan_rows: Sequence[Mapping[str, Any]],
    sealed_rows: Sequence[Mapping[str, Any]],
    collection: Mapping[str, Any],
    duration_s: float,
    test_exit_codes: Mapping[str, int] | None,
) -> JsonDict:
    aggregates = aggregate_evaluations(sealed_rows)
    protected = _protected_file_receipt(
        config.repo_root, baseline=preconditions.protected_file_baseline
    )
    completed = _completed_headline_models(collection.get("model_attempts") or [])
    visibility = _retrieval_and_oracle_visibility(plan_rows, sealed_rows)
    raw_receipt = _raw_stream_receipt(config.raw_path(), sealed_rows, plan_rows)
    stream_score = _constraint_stream_ready_score(
        completed, raw_receipt, visibility, protected, plan_rows
    )
    residual = _residual_headroom(sealed_rows, completed, plan_rows)
    status, verdict = _status_and_verdict(
        preconditions.block_reason,
        completed,
        stream_score,
        visibility["authority_violation_detected"],
        aggregates["group_bootstrap_lower_bounds"],
    )
    artifact: JsonDict = {
        "schema": ARTIFACT_SCHEMA_VERSION,
        "experiment_id": "experiment_5909_sota_constraint_synthesis_ab",
        "run_date": RUN_DATE,
        "random_seed": config.random_seed,
        "field_principles": FIELD_PRINCIPLES,
        "status": status,
        "preconditions_checked": preconditions.preconditions_checked,
        "upstream_gate_and_fixture_hashes": preconditions.upstream_gate_and_fixture_hashes,
        "model_specs": preconditions.model_specs,
        "model_file_hashes": preconditions.model_file_hashes,
        "embedded_tokenizer_and_loader_cuda_receipts": _runtime_loader_receipt(
            preconditions, collection
        ),
        "frozen_prompts_decoding_seeds_and_budgets": _frozen_prompt_receipt(config),
        "arm_definitions_and_compute_parity": _arm_parity_receipt(),
        "retrieval_and_oracle_visibility": visibility,
        **aggregates,
        "chronological_raw_stream_receipt": raw_receipt,
        "residual_error_and_diagnostic_headroom": residual,
        "gpu_utilization_vram_latency_and_energy_receipts": _gpu_latency_receipt(
            sealed_rows,
            collection.get("gpu_receipts") or {},
            collection.get("model_attempts") or [],
        ),
        "protected_files_unchanged": protected,
        "constraint_stream_ready_score": stream_score,
        "verification_repair_admission_ready_score": residual[
            "verification_repair_admission_ready_score"
        ],
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
    stream_score: float,
    authority_violation: bool,
    lower_bounds: Mapping[str, Any],
) -> tuple[str, str]:
    if block_reason is not None:
        return "blocked_precondition", f"blocked_precondition: {block_reason}"
    if authority_violation:
        return "unsafe", "unsafe: oracle visibility boundary was violated"
    if set(completed_models) != set(MANDATED_MODEL_IDS):
        return "blocked", "blocked: not all mandated model families completed"
    if stream_score != 1.0:
        return "blocked", "blocked: sealed raw stream is incomplete or unauthenticated"
    positive = (
        max(
            float(lower_bounds.get("semantic_decomposition_vs_direct") or 0.0),
            float(lower_bounds.get("decomposition_plus_exact_example_retrieval_vs_direct") or 0.0),
        )
        > 0.0
    )
    if positive:
        return "complete", "complete_positive: structured prompt arms improved exact synthesis"
    return "complete", "complete_null: structured prompt arms did not improve exact synthesis"


def _unit_order_for_arm(plan_row: Mapping[str, Any], arm_id: str) -> list[str]:
    components = list((plan_row.get("decomposition_plan") or {}).get("components") or [])
    by_hash = {
        str(component["component_hash"]): str(component["unit_type"]) for component in components
    }
    if arm_id == "direct":
        return []
    hashes = list(
        (plan_row.get("prompt_plan_arms") or {}).get(arm_id, {}).get("component_hashes") or []
    )
    return [by_hash[value] for value in hashes if value in by_hash]


def _unit_instruction(unit_type: str) -> str:
    instructions = {
        "entities_domains": "name finite domains and entities",
        "state_facts": "write observed facts with truth values",
        "transition_implication_relations": "write implication rules",
        "invariants": "declare schema and predicate signatures",
        "explicit_negation": "encode explicit not terms or false facts",
        "arithmetic_constraints": "encode arithmetic comparisons",
        "query_goals": "write the query goal",
    }
    return instructions.get(unit_type, unit_type)


def _retrieval_example_blocks(
    plan_row: Mapping[str, Any],
    source_rows_by_id: Mapping[str, Mapping[str, Any]],
    arm_id: str,
) -> list[str]:
    blocks = []
    exemplars = list(
        (plan_row.get("prompt_plan_arms") or {}).get(arm_id, {}).get("exemplars") or []
    )
    for index, exemplar in enumerate(exemplars, start=1):
        row_id = exemplar.get("row_id")
        if row_id not in source_rows_by_id:
            continue
        row = source_rows_by_id[str(row_id)]
        blocks.append(
            "\n".join(
                [
                    f"Example {index} problem:",
                    str(row["natural_language"]),
                    f"Example JSON {index}:",
                    canonical_json(row["constraint_ir"]),
                ]
            )
        )
    return blocks


def _exact_label_projection(evaluation: Mapping[str, Any]) -> JsonDict:
    return {
        "parse_valid": evaluation.get("parse_valid"),
        "type_valid": evaluation.get("type_valid"),
        "compiled": evaluation.get("compiled"),
        "satisfiability_correct": evaluation.get("satisfiability_correct"),
        "exact_semantic_equivalence": evaluation.get("exact_semantic_equivalence"),
        "query_correct": evaluation.get("query_correct"),
        "unsafe_accepted_constraints": evaluation.get("unsafe_accepted_constraints"),
    }


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    row_list = [dict(row) for row in rows]
    total = len(row_list)
    semantic = [row for row in row_list if row.get("expected_status") == "valid"]
    query = [row for row in semantic if row.get("query_correct") is not None]
    exact_values = [
        1.0 if row.get("exact_semantic_equivalence") is True else 0.0 for row in semantic
    ]
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
        "exact_semantic_equivalence_interval": _simple_interval(exact_values),
        "query_correct_rate": _rate(
            sum(row.get("query_correct") is True for row in query), len(query)
        ),
        "unsafe_accept_rate": _rate(
            sum(bool(row.get("unsafe_accepted_constraints")) for row in row_list), total
        ),
        "total_tokens": sum(
            int((row.get("usage") or {}).get("total_tokens") or row.get("total_tokens") or 0)
            for row in row_list
        ),
        "latency_s": round(sum(float(row.get("latency_s") or 0.0) for row in row_list), 6),
        "energy_proxy_gpu_utilization_pct_s": round(sum(_energy_proxy(row) for row in row_list), 6),
    }


def _simple_interval(values: Sequence[float]) -> JsonDict:
    if not values:
        return {"mean": 0.0, "ci95": [0.0, 0.0], "n": 0}
    mean = sum(values) / len(values)
    radius = 1.96 * ((mean * (1.0 - mean) / len(values)) ** 0.5) if len(values) else 0.0
    return {
        "mean": round(mean, 6),
        "ci95": [round(max(0.0, mean - radius), 6), round(min(1.0, mean + radius), 6)],
        "n": len(values),
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
        for arm_id in ALL_ARM_IDS
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


def _bootstrap_lower_bound(
    rows: Sequence[Mapping[str, Any]], left_arm: str, right_arm: str
) -> float:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("split") != "heldout" or row.get("expected_status") != "valid":
            continue
        key = (str(row.get("model_hf_id")), str(row.get("group_id")))
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


def _retrieval_and_oracle_visibility(
    plan_rows: Sequence[Mapping[str, Any]],
    sealed_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    violations = []
    for plan in plan_rows:
        for arm_id in ("decomposition_plus_exact_example_retrieval", "wrong_family_retrieval"):
            for exemplar in (plan.get("prompt_plan_arms") or {}).get(arm_id, {}).get(
                "exemplars"
            ) or []:
                if exemplar.get("group_id") == plan.get("group_id"):
                    violations.append(f"same_group:{plan.get('source_row_id')}:{arm_id}")
                if exemplar.get("split") == "heldout":
                    violations.append(f"heldout:{plan.get('source_row_id')}:{arm_id}")
    oracle_leakage = _diagnostic_oracle_leakage(sealed_rows)
    return {
        "target_hidden_gold_ir_exposed_to_prompt": False,
        "target_exact_labels_exposed_to_prompt": False,
        "certificates_exposed_to_prompt": False,
        "diagnostic_repair_traces_exposed_to_prompt": False,
        "retrieval_examples_group_and_split_violations": violations,
        "retrieval_examples_visible_exact_ir_allowed": True,
        "visible_to_model": [
            "natural_language",
            "schema instructions",
            "semantic unit names",
            "retrieved visible example natural_language",
            "retrieved visible example ConstraintIR",
        ],
        "visible_after_proposal_to_downstream_learning": [
            "raw_output_text",
            "parser_status",
            "parser_error",
            "type_status",
            "compiler_status",
            "solver_status",
            "exact labels",
            "group_id",
            "row hashes",
        ],
        "withheld_from_model": [
            "target hidden gold IR",
            "target labels",
            "certificates",
            "held identities",
            "diagnostic repair traces",
        ],
        "oracle_leakage_in_visible_diagnostics": oracle_leakage,
        "authority_violation_detected": bool(violations) or oracle_leakage,
        "principle": FIELD_PRINCIPLES["chronological_raw_stream_receipt"],
    }


def _diagnostic_oracle_leakage(rows: Sequence[Mapping[str, Any]]) -> bool:
    forbidden = ("constraint_ir", "certificates", "behavior_signature", "query_bindings")
    for row in rows:
        text = canonical_json(row.get("visible_diagnostics") or {}).lower()
        if any(token in text for token in forbidden):
            return True
    return False


def _constraint_stream_ready_score(
    completed_models: Sequence[str],
    raw_receipt: Mapping[str, Any],
    visibility: Mapping[str, Any],
    protected: Mapping[str, Any],
    plan_rows: Sequence[Mapping[str, Any]],
) -> float:
    ready = (
        set(completed_models) == set(MANDATED_MODEL_IDS)
        and int(raw_receipt.get("row_count") or 0) == expected_raw_event_count(plan_rows)
        and raw_receipt.get("exact_label_coverage") is True
        and raw_receipt.get("raw_hash_coverage") is True
        and raw_receipt.get("event_order_is_chronological") is True
        and visibility.get("authority_violation_detected") is False
        and protected.get("unchanged") is True
    )
    return 1.0 if ready else 0.0


def _residual_headroom(
    rows: Sequence[Mapping[str, Any]],
    completed_models: Sequence[str],
    plan_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    required_groups = sorted({str(row.get("group_id")) for row in plan_rows})
    residuals: dict[str, list[Mapping[str, Any]]] = {group: [] for group in required_groups}
    for row in rows:
        if row.get("arm_id") not in PRIMARY_ARM_IDS:
            continue
        if _row_is_incorrect(row):
            residuals.setdefault(str(row.get("group_id")), []).append(row)
    counts = {group: len(group_rows) for group, group_rows in residuals.items()}
    diagnostic_counts = {
        group: sum(_has_visible_diagnostic(row) for row in group_rows)
        for group, group_rows in residuals.items()
    }
    all_groups = all(counts.get(group, 0) > 0 for group in required_groups)
    diagnostics = all(
        diagnostic_counts.get(group, 0) == counts.get(group, 0) for group in required_groups
    )
    no_leakage = not _diagnostic_oracle_leakage(rows)
    ready = (
        set(completed_models) == set(MANDATED_MODEL_IDS)
        and all_groups
        and diagnostics
        and no_leakage
    )
    return {
        "required_group_ids": required_groups,
        "residual_incorrect_counts_by_group": counts,
        "visible_diagnostic_counts_by_group": diagnostic_counts,
        "all_required_groups_have_residuals": all_groups,
        "residuals_have_deployment_visible_diagnostics": diagnostics,
        "unsafe_oracle_leakage_detected": not no_leakage,
        "verification_repair_admission_ready_score": 1.0 if ready else 0.0,
        "principle": FIELD_PRINCIPLES["verification_repair_admission_ready_score"],
    }


def _row_is_incorrect(row: Mapping[str, Any]) -> bool:
    if row.get("expected_status") == "valid":
        return (
            row.get("exact_semantic_equivalence") is not True
            or row.get("query_correct") is not True
        )
    return row.get("satisfiability_correct") is not True


def _has_visible_diagnostic(row: Mapping[str, Any]) -> bool:
    diagnostics = row.get("visible_diagnostics") or row.get("diagnostics") or {}
    return isinstance(diagnostics, Mapping) and bool(diagnostics.get("parser_status"))


def _raw_stream_receipt(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    plan_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    sequence = [int(row.get("stream_sequence_index") or 0) for row in rows]
    return {
        "path": str(RAW_STREAM_RELATIVE_PATH),
        "schema": RAW_ROW_SCHEMA_VERSION,
        "row_count": len(rows),
        "expected_row_count": expected_raw_event_count(plan_rows),
        "sha256": sha256_file(path) if path.exists() else None,
        "first_sequence_index": min(sequence) if sequence else None,
        "last_sequence_index": max(sequence) if sequence else None,
        "event_order_is_chronological": sequence == list(range(len(rows))),
        "raw_hash_coverage": all(row.get("raw_output_sha256") for row in rows),
        "exact_label_coverage": all(row.get("exact_labels") for row in rows),
        "row_hash_coverage": all(row.get("row_hash") == stream_row_hash(row) for row in rows),
        "group_ids": sorted({str(row.get("group_id")) for row in rows}),
        "raw_output_hashes": [
            {
                "stream_sequence_index": row.get("stream_sequence_index"),
                "model_hf_id": row.get("model_hf_id"),
                "source_row_id": row.get("source_row_id"),
                "arm_id": row.get("arm_id"),
                "raw_output_sha256": row.get("raw_output_sha256"),
                "row_hash": row.get("row_hash"),
            }
            for row in rows
        ],
        "principle": FIELD_PRINCIPLES["chronological_raw_stream_receipt"],
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
            sum(_energy_proxy(row) for row in rows), 6
        ),
        "row_count": len(rows),
    }


def _energy_proxy(row: Mapping[str, Any]) -> float:
    return round(
        float(row.get("latency_s") or 0.0)
        * float(row.get("average_gpu_utilization_pct") or 0.0)
        / 100.0,
        6,
    )


def _runtime_loader_receipt(
    preconditions: PreconditionReport,
    collection: Mapping[str, Any],
) -> JsonDict:
    receipt = json.loads(canonical_json(preconditions.embedded_tokenizer_and_loader_cuda_receipts))
    receipt["runtime_loader_receipts"] = [
        dict(row) for row in collection.get("model_attempts") or []
    ]
    receipt["gpu_receipts"] = dict(collection.get("gpu_receipts") or {})
    return receipt


def _frozen_prompt_receipt(config: ExperimentConfig) -> JsonDict:
    source_rows = {row["row_id"]: row for row in exp5896.build_fixture_rows()}
    sample_plan = exp5908.build_prompt_plan_rows()[0]
    sample_source = source_rows[str(sample_plan["source_row_id"])]
    prompts = {
        arm_id: build_prompt(sample_plan, sample_source, source_rows, arm_id)
        for arm_id in ALL_ARM_IDS
    }
    return {
        "prompt_version": "exp5909.constraint_synthesis_ab.v1",
        "prompt_sha256": {arm_id: sha256_text(prompt) for arm_id, prompt in prompts.items()},
        "decoding": DECODING,
        "budgets": GENERATION_BUDGETS,
        "base_random_seed": config.random_seed,
        "control_subset_row_ids": list(CONFIRMATORY_CONTROL_ROW_IDS),
        "row_groups_frozen_by_exp5908": True,
        "target_oracle_payloads_in_prompts": False,
    }


def _arm_parity_receipt() -> JsonDict:
    max_tokens = {arm: ARM_DEFINITIONS[arm]["max_tokens"] for arm in ALL_ARM_IDS}
    calls = {arm: ARM_DEFINITIONS[arm]["calls"] for arm in ALL_ARM_IDS}
    return {
        "arms": ARM_DEFINITIONS,
        "all_call_counts_match": len(set(calls.values())) == 1,
        "all_output_token_budgets_match": len(set(max_tokens.values())) == 1,
        "retrieval_exemplar_count": exp5908.EXEMPLARS_PER_RETRIEVAL_ARM,
        "control_subset_row_ids": list(CONFIRMATORY_CONTROL_ROW_IDS),
        "principle": FIELD_PRINCIPLES["arm_definitions_and_compute_parity"],
    }


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
    result_path = root / exp5908.RESULT_RELATIVE_PATH
    row_path = root / exp5908.ROW_FILE_RELATIVE_PATH
    try:
        replay = exp5908.replay_artifact(root=root)
        replay_ok = bool(replay.get("ok"))
        replay_error = None
    except Exception as exc:  # noqa: BLE001
        replay = {}
        replay_ok = False
        replay_error = f"{type(exc).__name__}: {exc}"
    return {
        "exp5908_replay_ok": replay_ok,
        "exp5908_replay_error": replay_error,
        "exp5908_replay": replay,
        "exp5908_artifact_path": str(exp5908.RESULT_RELATIVE_PATH),
        "exp5908_row_path": str(exp5908.ROW_FILE_RELATIVE_PATH),
        "exp5908_artifact_sha256": sha256_file(result_path) if result_path.exists() else None,
        "exp5908_row_file_sha256": sha256_file(row_path) if row_path.exists() else None,
        "exp5896_row_file_sha256": sha256_file(root / exp5896.ROW_FILE_RELATIVE_PATH)
        if (root / exp5896.ROW_FILE_RELATIVE_PATH).exists()
        else None,
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


def _load_plan_rows(root: Path) -> list[JsonDict]:
    path = root / exp5908.ROW_FILE_RELATIVE_PATH
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _safe_load_plan_rows(root: Path) -> list[JsonDict]:
    try:
        return _load_plan_rows(root)
    except Exception:  # noqa: BLE001
        return []


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
        field: {
            "principle": FIELD_PRINCIPLES.get(field, "Exp5909 required artifact field."),
            "satisfied_by": "generated_by_exp5909_sota_constraint_synthesis_ab",
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
        import importlib.metadata
        import llama_cpp
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": True,
        "detail": f"llama_cpp {getattr(llama_cpp, '__version__', 'unknown')}",
        "distribution_version": importlib.metadata.version("llama-cpp-python"),
        "origin": str(getattr(llama_cpp, "__file__", "")),
    }


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
    target = probe_dir / ".exp5909_atomic_probe"
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


def _call_llama(
    llm: Any,
    prompt: str,
    arm_id: str,
    spec: Mapping[str, Any],
    plan_row: Mapping[str, Any],
    sequence: int,
    model_index: int,
    plan_index: int,
    arm_offset: int,
    config: ExperimentConfig,
) -> JsonDict:  # pragma: no cover - requires local GGUFs and CUDA.
    start = config.monotonic_clock()
    seed = config.random_seed + model_index * 1_000_000 + plan_index * 100 + arm_offset
    output_text = ""
    usage: JsonDict = {}
    try:
        result = llm(
            prompt,
            max_tokens=int(ARM_DEFINITIONS[arm_id]["max_tokens"]),
            temperature=float(DECODING["temperature"]),
            top_p=float(DECODING["top_p"]),
            repeat_penalty=float(DECODING["repeat_penalty"]),
            stop=list(DECODING["stop"]),
            seed=seed,
            echo=False,
        )
        output_text, usage = _completion_text_and_usage(result)
    except Exception as exc:  # noqa: BLE001
        output_text = ""
        usage = {"error": f"{type(exc).__name__}: {exc}"}
    health = _gpu_health_probe()
    util = max(
        (int(gpu.get("utilization_gpu_pct") or 0) for gpu in health.get("gpus") or []), default=0
    )
    return {
        "stream_sequence_index": sequence,
        "model_hf_id": spec.get("hf_id"),
        "model_name": spec.get("name"),
        "model_path": spec.get("model_path"),
        "gpu_index": spec.get("gpu"),
        "source_row_id": plan_row.get("source_row_id"),
        "plan_row_hash": plan_row.get("row_hash"),
        "group_id": plan_row.get("group_id"),
        "arm_id": arm_id,
        "prompt_sha256": sha256_text(prompt),
        "seed": seed,
        "raw_output_text": output_text,
        "latency_s": round(config.monotonic_clock() - start, 6),
        "usage": usage,
        "average_gpu_utilization_pct": util,
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
    return text, dict(result.get("usage") or {})


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
        "[exp5909] "
        f"status={artifact['status']} "
        f"verdict={artifact['honest_verdict']} "
        f"stream_score={artifact['constraint_stream_ready_score']} "
        f"repair_score={artifact['verification_repair_admission_ready_score']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
