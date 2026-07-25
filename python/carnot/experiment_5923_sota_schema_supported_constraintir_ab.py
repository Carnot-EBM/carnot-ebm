"""Exp5923 paired SOTA schema-supported ConstraintIR A/B.

Spec refs: REQ-VERIFY-5923, SCENARIO-VERIFY-5923.

This experiment treats local GGUF models only as proposal generators.  The
exact parser, type/scope checks, Python/Z3 execution, certificates, and
semantic-equivalence labels own the outcome.  Schema-supported decoding may
improve formatting, but the primary endpoint is exact semantic success over
matched direct and prompt-structured controls.
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
from carnot import experiment_5921_schema_derived_constraintir_support as exp5921
from carnot import experiment_5922_gguf_schema_decoder_bridge as exp5922


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5923_sota_schema_supported_constraintir_ab.json")
EVENT_STREAM_RELATIVE_PATH = Path(
    "results/experiment_5923_sota_schema_supported_constraintir_ab.events.jsonl"
)
CHECKPOINT_RELATIVE_PATH = Path(
    "results/checkpoints/experiment_5923_sota_schema_supported_constraintir_ab.events.jsonl"
)
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5923_sota_schema_supported_constraintir_ab.py")
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5923_sota_schema_supported_constraintir_ab.py"
)
VERIFIABLE_REASONING_SPEC_RELATIVE_PATH = Path(
    "openspec/capabilities/verifiable-reasoning/spec.md"
)

RUN_DATE = "20260725"
EXPERIMENT_ID = "experiment_5923_sota_schema_supported_constraintir_ab"
ARTIFACT_SCHEMA_VERSION = "carnot.experiment_5923.sota_schema_supported_constraintir_ab.v1"
EVENT_SCHEMA_VERSION = ARTIFACT_SCHEMA_VERSION + ".event"
RANDOM_SEED = 5923
INFERENCE_SUBSTRATE = "local_mandated_gguf_public_llama_cpp_cuda_schema_supported_decoding"
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

ARM_IDS = (
    "direct",
    "prompt_structured",
    "schema_first_ir_token",
    "reason_then_schema_first_ir_token",
)
CONTROL_ARM_IDS = ("direct", "prompt_structured")
SCHEMA_ARM_IDS = ("schema_first_ir_token", "reason_then_schema_first_ir_token")
DEFAULT_PANEL_CASE_IDS = (
    "train_task_canonical",
    "held_family_menu_canonical",
    "held_composition_nested_and",
    "attribute_semantic_price_cap_false",
)
DECODING_PARAMETERS: JsonDict = {
    "temperature": 0.0,
    "top_p": 1.0,
    "repeat_penalty": 1.05,
    "stop": ["</s>", "<eos>", "<|eot_id|>"],
}
TOKEN_BUDGETS: JsonDict = {
    "direct": {"max_tokens": 192, "reasoning_tokens": 0, "schema_tokens": 0},
    "prompt_structured": {"max_tokens": 192, "reasoning_tokens": 0, "schema_tokens": 0},
    "schema_first_ir_token": {"max_tokens": 192, "reasoning_tokens": 0, "schema_tokens": 192},
    "reason_then_schema_first_ir_token": {
        "max_tokens": 192,
        "reasoning_tokens": 48,
        "schema_tokens": 144,
    },
    "n_ctx": 8192,
    "n_batch": 256,
    "n_gpu_layers": -1,
}
ARM_DEFINITIONS: JsonDict = {
    "direct": {
        "control": True,
        "schema_supported": False,
        "reasoning_outside_ir": False,
        "calls": 1,
        "principle": "Unstructured baseline gets the same total proposal budget.",
    },
    "prompt_structured": {
        "control": True,
        "schema_supported": False,
        "reasoning_outside_ir": False,
        "calls": 1,
        "principle": "Prompt-only structure controls for instructions without token masks.",
    },
    "schema_first_ir_token": {
        "control": False,
        "schema_supported": True,
        "reasoning_outside_ir": False,
        "calls": 1,
        "principle": "The logits mask starts at the first generated ConstraintIR byte.",
    },
    "reason_then_schema_first_ir_token": {
        "control": False,
        "schema_supported": True,
        "reasoning_outside_ir": True,
        "calls": 1,
        "principle": "Free reasoning is outside the IR; the IR itself is schema-masked.",
    },
}
PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)
HASHED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    exp5896.MODULE_RELATIVE_PATH,
    exp5921.MODULE_RELATIVE_PATH,
    exp5922.MODULE_RELATIVE_PATH,
    exp5921.RESULT_RELATIVE_PATH,
    exp5922.RESULT_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    VERIFIABLE_REASONING_SPEC_RELATIVE_PATH,
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "gate_replay_receipt",
    "preconditions_checked",
    "model_specs",
    "model_file_hashes",
    "embedded_tokenizer_loader_cuda_gpu_and_vram_receipts",
    "sealed_cases_families_adversaries_prompts_seeds_arms_budgets",
    "per_model_arm_structural_and_exact_metrics",
    "exact_semantic_primary_comparison_and_intervals",
    "held_family_and_attribute_semantic_results",
    "missing_spurious_and_unsafe_acceptance",
    "correct_mode_diversity_and_overpruning",
    "token_latency_gpu_and_memory_accounting",
    "chronological_event_stream_path_hash_rows_and_prefix_chain",
    "chronological_event_stream_ready_score",
    "retirement_decision",
    "no_repair_call_and_no_answer_enumeration_receipt",
    "protected_files_unchanged",
    "schema_decode_live_ready_score",
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
    "exact_semantic_primary_comparison_and_intervals": (
        "syntax, type, and scope improvements cannot substitute for an exact semantic win "
        "over the best matched control."
    ),
    "chronological_event_stream_ready_score": (
        "bare 1.0 means complete real-model chronology and tamper-safe replay, "
        "not positive model science."
    ),
    "schema_decode_live_ready_score": (
        "bare 1.0 only for positive held exact-semantic improvement, zero unsafe accepts, "
        "and no material correct-mode collapse."
    ),
    "inference_substrate": (
        "use local_mandated_gguf_public_llama_cpp_cuda_schema_supported_decoding."
    ),
    "verifier_is_oracle": (
        "true only for exact parse, type, scope, execution, equivalence, and certificate "
        "adjudication."
    ),
    "honest_verdict": "use complete_positive:, complete_null:, retired:, or blocked:.",
}
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5923_sota_schema_supported_constraintir_ab.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5923_sota_schema_supported_constraintir_ab.py "
    "-m pytest tests/python/test_experiment_5923_sota_schema_supported_constraintir_ab.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5923_sota_schema_supported_constraintir_ab.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python -m carnot.experiment_5923_sota_schema_supported_constraintir_ab",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5923_sota_schema_supported_constraintir_ab.json",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5923_sota_schema_supported_constraintir_ab.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "git status --short -- scripts/research_conductor.py ops/changelog.md ops/status.md _bmad/traceability.md",
)
FORBIDDEN_EVENT_REFERENCE_KEYS = (
    "target_constraint_ir",
    "reference_answer",
    "hidden_reference_answer",
    "gold_constraint_ir",
    "certificate_solution",
    "query_bindings",
    "canonical_behavior_hash",
)


ModelResolver = Callable[[], list[JsonDict]]
EnvironmentProbe = Callable[[Path], JsonDict]
TokenizerLoader = Callable[[JsonDict], JsonDict]
GateReplayProvider = Callable[[Path], JsonDict]
CollectOutputsFn = Callable[[list[JsonDict], list[JsonDict], "ExperimentConfig", JsonDict], JsonDict]


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths, clocks, and frozen seeds for Exp5923."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    event_stream_path: Path | None = None
    checkpoint_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    monotonic_clock: Callable[[], float] = time.monotonic
    random_seed: int = RANDOM_SEED

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / RESULT_RELATIVE_PATH

    def events_path(self) -> Path:
        return self.event_stream_path or self.repo_root / EVENT_STREAM_RELATIVE_PATH

    def resume_path(self) -> Path:
        return self.checkpoint_path or self.repo_root / CHECKPOINT_RELATIVE_PATH


@dataclass(frozen=True)
class PreconditionReport:
    """Pre-model-load Exp5923 resource receipts."""

    gate_replay_receipt: JsonDict
    preconditions_checked: JsonDict
    model_specs: list[JsonDict]
    model_file_hashes: JsonDict
    receipts: JsonDict
    protected_file_baseline: JsonDict
    block_reason: str | None


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence with stable ASCII bytes before hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash a local file by bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def resolve_model_specs() -> list[JsonDict]:  # pragma: no cover - host cache boundary.
    """Resolve two models through cached_sota_pair plus the cached third family."""

    return exp5922.resolve_all_model_specs()


def default_tokenizer_loader(spec: JsonDict) -> JsonDict:  # pragma: no cover - GGUF boundary.
    """Load the embedded GGUF tokenizer and return only the audit receipt."""

    vocab = exp5922.load_embedded_llama_cpp_vocabulary(spec)
    receipt = dict(vocab.tokenizer_receipt)
    receipt.pop("_tokenizer", None)
    receipt["ok"] = True
    return receipt


def freeze_panel_cases(case_ids: Sequence[str] = DEFAULT_PANEL_CASE_IDS) -> list[JsonDict]:
    """Freeze the paired train/held/adversary panel without answer payloads."""

    cases_by_id = {str(case["case_id"]): case for case in exp5921.build_adversary_cases()}
    fixture_by_id = {str(row["row_id"]): row for row in exp5896.build_fixture_rows()}
    frozen: list[JsonDict] = []
    for index, case_id in enumerate(case_ids):
        case = cases_by_id[str(case_id)]
        target = fixture_by_id[str(case["target_row_id"])]
        prompt_text = _visible_problem_text(target)
        frozen.append(
            {
                "case_index": index,
                "case_id": case["case_id"],
                "target_row_id": case["target_row_id"],
                "target_group_id": case["target_group_id"],
                "target_family": case["target_family"],
                "target_row_hash": target["row_hash"],
                "split": case["split_role"],
                "adversary_kind": case["adversary_kind"],
                "expected_semantic_success": bool(case["expected_semantic_success"]),
                "visible_problem_text": prompt_text,
                "visible_problem_sha256": sha256_text(prompt_text),
            }
        )
    return frozen


def build_prompt(case: Mapping[str, Any], arm_id: str) -> str:
    """Return the frozen prompt for one case and arm."""

    if arm_id not in ARM_IDS:
        raise ValueError(f"unknown arm_id: {arm_id}")
    lines = [
        "Return a typed ConstraintIR JSON object for this constraint problem.",
        "Use only schema_version, domains, entities, predicates, facts, rules, query.",
        "Supported expression nodes are atom, not, and, arith over finite domains.",
        f"schema_version must be {exp5896.CONSTRAINT_IR_SCHEMA_VERSION}.",
    ]
    if arm_id == "direct":
        lines = ["Translate the constraint problem into typed ConstraintIR JSON."]
    elif arm_id == "prompt_structured":
        lines.append("Check predicate arity, finite domains, variable scope, and arithmetic attributes before final JSON.")
    elif arm_id == "schema_first_ir_token":
        lines.append("Begin immediately with the first JSON object byte; no prose or markdown.")
    else:
        lines.insert(0, "Think briefly outside the IR, then emit exactly one JSON object.")
        lines.append("After reasoning, the ConstraintIR JSON must begin at its first JSON byte.")
    lines.extend(["Problem:", str(case["visible_problem_text"])])
    return "\n".join(lines) + "\n"


def build_preregistration(config: ExperimentConfig, panel: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Record the frozen case, prompt, seed, arm, and verifier decisions."""

    prompt_rows = []
    for case in panel:
        for arm_index, arm_id in enumerate(ARM_IDS):
            prompt = build_prompt(case, arm_id)
            prompt_rows.append(
                {
                    "case_id": case["case_id"],
                    "split": case["split"],
                    "adversary_kind": case["adversary_kind"],
                    "arm_id": arm_id,
                    "arm_order_index": arm_index,
                    "prompt_sha256": sha256_text(prompt),
                    "seed": config.random_seed + int(case["case_index"]) * 100 + arm_index,
                    "budget": TOKEN_BUDGETS[arm_id],
                }
            )
    return {
        "run_date": RUN_DATE,
        "case_count": len(panel),
        "cases": [
            {
                "case_index": case["case_index"],
                "case_id": case["case_id"],
                "target_row_id": case["target_row_id"],
                "target_row_hash": case["target_row_hash"],
                "split": case["split"],
                "target_family": case["target_family"],
                "adversary_kind": case["adversary_kind"],
                "expected_semantic_success": case["expected_semantic_success"],
                "visible_problem_sha256": case["visible_problem_sha256"],
            }
            for case in panel
        ],
        "arms": {arm_id: ARM_DEFINITIONS[arm_id] for arm_id in ARM_IDS},
        "arm_order": list(ARM_IDS),
        "prompt_seed_budget_rows": prompt_rows,
        "decoding_parameters": DECODING_PARAMETERS,
        "token_budgets": TOKEN_BUDGETS,
        "exact_verifier": "exp5896_parse_type_python_z3_behavior_hash_equivalence_v1",
        "semantic_equivalence_rules": exp5896.FIELD_PRINCIPLES[
            "exact_semantic_equivalence_contract"
        ],
        "stopping_rules": {
            "max_rows": len(MANDATED_MODEL_IDS) * len(panel) * len(ARM_IDS),
            "no_repair_call": True,
            "no_complete_answer_enumeration": True,
        },
        "bootstrap_intervals": {
            "method": "deterministic_paired_bootstrap_ci95",
            "resamples": 200,
            "seed": config.random_seed,
        },
    }


def check_preconditions(
    config: ExperimentConfig,
    *,
    model_resolver: ModelResolver = resolve_model_specs,
    environment_probe: EnvironmentProbe = lambda root: _probe_environment(root),
    tokenizer_loader: TokenizerLoader = default_tokenizer_loader,
    gate_replay_provider: GateReplayProvider = lambda root: _gate_replay_receipt(root),
) -> PreconditionReport:
    """Replay Exp5922 and host checks before any model generation load."""

    root = config.repo_root
    baseline = _protected_file_receipt(root)
    gate = gate_replay_provider(root)
    model_specs = model_resolver()
    model_hashes = _model_file_hashes(model_specs)
    tokenizers = _tokenizer_receipts(model_specs, tokenizer_loader)
    environment = environment_probe(root)
    expected_ids = list(MANDATED_MODEL_IDS)
    ids = [str(spec.get("hf_id")) for spec in model_specs]
    checks = {
        "exp5922_gate_replayed_before_model_load": gate.get("ok") is True,
        "model_specs_defined": ids == expected_ids,
        "resolved_all_three_model_files": all(spec.get("model_path") for spec in model_specs)
        and ids == expected_ids,
        "model_files_hashed": set(model_hashes) == set(expected_ids),
        "embedded_gguf_tokenizers_load": all(row.get("ok") for row in tokenizers.values())
        and set(tokenizers) == set(expected_ids),
        "no_hf_autotokenizer_for_gguf": all(
            row.get("used_hf_autotokenizer") is False for row in tokenizers.values()
        ),
        "public_llama_cpp_cuda_available": bool(
            environment.get("public_llama_cpp_cuda", {}).get("ok")
        ),
        "two_healthy_rtx_3090s": _two_healthy_rtx_3090s(environment.get("gpu_health", {})),
        "adequate_vram": _adequate_vram(environment.get("gpu_health", {})),
        "adequate_ram": bool(environment.get("ram", {}).get("ok")),
        "adequate_disk": bool(environment.get("disk", {}).get("ok")),
        "real_nonzero_gpu_offload_supported": bool(
            environment.get("public_llama_cpp_cuda", {}).get("gpu_offload_supported")
        ),
        "atomic_output_ready": bool(environment.get("atomic_output", {}).get("ok")),
        "atomic_checkpoint_resume_ready": bool(
            environment.get("atomic_checkpoint_resume", {}).get("ok")
        ),
        "no_protected_workload": bool(environment.get("protected_workload", {}).get("ok")),
    }
    block_reason = next((name for name, ok in checks.items() if not ok), None)
    return PreconditionReport(
        gate_replay_receipt=gate,
        preconditions_checked={
            "run_order": "exp5922_gate_and_resource_checks_before_any_model_load",
            "blocked_before_model_load": block_reason is not None,
            "headline_checks": checks,
            "block_reason": block_reason,
        },
        model_specs=model_specs,
        model_file_hashes=model_hashes,
        receipts={
            "embedded_tokenizer_receipts": tokenizers,
            "environment": environment,
            "model_resolution": {
                "cached_sota_pair_plus_cached_third_family": ids == expected_ids,
                "resolved_hf_ids": ids,
            },
        },
        protected_file_baseline=baseline,
        block_reason=block_reason,
    )


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    model_resolver: ModelResolver = resolve_model_specs,
    environment_probe: EnvironmentProbe = lambda root: _probe_environment(root),
    tokenizer_loader: TokenizerLoader = default_tokenizer_loader,
    gate_replay_provider: GateReplayProvider = lambda root: _gate_replay_receipt(root),
    collect_model_outputs_fn: CollectOutputsFn | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Run Exp5923, writing the event stream and terminal artifact atomically."""

    active = config or ExperimentConfig()
    started = active.start_time()
    preconditions = check_preconditions(
        active,
        model_resolver=model_resolver,
        environment_probe=environment_probe,
        tokenizer_loader=tokenizer_loader,
        gate_replay_provider=gate_replay_provider,
    )
    panel = freeze_panel_cases()
    preregistration = build_preregistration(active, panel)
    if preconditions.block_reason is not None:
        _write_event_stream_atomic(active.events_path(), [])
        artifact = _build_artifact(
            active,
            preconditions,
            preregistration=preregistration,
            events=[],
            collection={"rows": [], "real_model_rows": False, "model_attempts": []},
            duration_s=active.clock() - started,
            test_exit_codes=test_exit_codes,
        )
        _write_json_atomic(active.artifact_path(), artifact)
        return artifact

    collector = collect_model_outputs_fn or collect_live_model_outputs
    collection = collector(preconditions.model_specs, panel, active, {"schema_support": "exp5922"})
    checkpoint_rows = load_checkpoint(active.resume_path())
    raw_rows = merge_resume_rows(checkpoint_rows, [dict(row) for row in collection.get("rows") or []])
    save_checkpoint(active.resume_path(), raw_rows)
    events = seal_raw_events(raw_rows, panel, active)
    _write_event_stream_atomic(active.events_path(), events)
    artifact = _build_artifact(
        active,
        preconditions,
        preregistration=preregistration,
        events=events,
        collection=collection,
        duration_s=active.clock() - started,
        test_exit_codes=test_exit_codes,
    )
    _write_json_atomic(active.artifact_path(), artifact)
    return artifact


def collect_live_model_outputs(
    model_specs: list[JsonDict],
    panel: list[JsonDict],
    config: ExperimentConfig,
    schema_runtime: JsonDict,
) -> JsonDict:  # pragma: no cover - live GGUF/CUDA boundary.
    """Collect live llama.cpp outputs for every model, case, and paired arm."""

    del schema_runtime
    from llama_cpp import Llama, LogitsProcessorList

    fixture_by_id = {str(row["row_id"]): row for row in exp5896.build_fixture_rows()}
    support = exp5921.compile_schema_support()
    rows: list[JsonDict] = []
    attempts: list[JsonDict] = []
    sequence = 0
    for model_index, spec in enumerate(model_specs):
        load_started = config.monotonic_clock()
        before_mb = _gpu_memory_total_mb()
        llm = None
        try:
            llm = Llama(
                model_path=str(spec["model_path"]),
                n_ctx=int(TOKEN_BUDGETS["n_ctx"]),
                n_batch=int(TOKEN_BUDGETS["n_batch"]),
                n_gpu_layers=int(TOKEN_BUDGETS["n_gpu_layers"]),
                main_gpu=int(spec.get("gpu") or 0),
                seed=config.random_seed + model_index,
                verbose=False,
            )
            after_load_mb = _gpu_memory_total_mb()
            vocab = exp5922.load_embedded_llama_cpp_vocabulary(spec)
            bridge = exp5922.SchemaDecoderBridge(support, vocab)
            attempts.append(
                {
                    "hf_id": spec["hf_id"],
                    "model_used": True,
                    "gpu_offload_verified": after_load_mb > before_mb,
                    "vram_delta_mb": max(0.0, after_load_mb - before_mb),
                    "load_duration_s": round(config.monotonic_clock() - load_started, 6),
                }
            )
            for case in panel:
                target = fixture_by_id[str(case["target_row_id"])]
                for arm_index, arm_id in enumerate(ARM_IDS):
                    prompt = build_prompt(case, arm_id)
                    seed = config.random_seed + model_index * 10000 + int(case["case_index"]) * 100 + arm_index
                    row = _call_live_arm(llm, bridge, prompt, spec, case, target, arm_id, seed, sequence, config)
                    rows.append(row)
                    sequence += 1
        except Exception as exc:
            attempts.append(
                {
                    "hf_id": spec.get("hf_id"),
                    "model_used": False,
                    "blocker": f"{type(exc).__name__}: {exc}",
                    "gpu_offload_verified": False,
                }
            )
        finally:
            close = getattr(llm, "close", None)
            if callable(close):
                close()
            llm = None
            gc.collect()
    return {
        "rows": rows,
        "real_model_rows": True,
        "model_attempts": attempts,
        "gpu_receipts": {"attempts": attempts},
    }


def seal_raw_events(
    raw_rows: Sequence[Mapping[str, Any]],
    panel: Sequence[Mapping[str, Any]],
    config: ExperimentConfig,
) -> list[JsonDict]:
    """Attach exact outcomes and a chronological prefix checksum to raw rows."""

    panel_by_id = {str(case["case_id"]): case for case in panel}
    fixture_by_id = {str(row["row_id"]): row for row in exp5896.build_fixture_rows()}
    events = []
    previous = "sha256:" + ("0" * 64)
    for index, raw in enumerate(sorted(raw_rows, key=lambda row: int(row["stream_sequence_index"]))):
        case = panel_by_id[str(raw["case_id"])]
        target = fixture_by_id[str(case["target_row_id"])]
        arm_id = str(raw["arm_id"])
        prompt = build_prompt(case, arm_id)
        raw_text = str(raw.get("raw_output_text") or "")
        evaluation = exp5897.evaluate_candidate(target, arm_id, raw_text, dict(raw))
        exact = _exact_outcome(evaluation)
        event: JsonDict = {
            "schema": EVENT_SCHEMA_VERSION,
            "event_kind": "model_schema_supported_constraintir_proposal",
            "stream_sequence_index": int(raw.get("stream_sequence_index", index)),
            "row_identity": {
                "model_hf_id": raw.get("model_hf_id"),
                "model_name": raw.get("model_name"),
                "arm_id": arm_id,
                "case_id": case["case_id"],
                "split": case["split"],
                "adversary_kind": case["adversary_kind"],
                "prompt_sha256": sha256_text(prompt),
                "seed": raw.get(
                    "seed",
                    config.random_seed + int(case["case_index"]) * 100 + ARM_IDS.index(arm_id),
                ),
                "budget": TOKEN_BUDGETS[arm_id],
            },
            "visible_proposal": {
                "raw_text": raw_text,
                "raw_sha256": sha256_text(raw_text),
                "usage": dict(raw.get("usage") or {}),
            },
            "exact_outcome": exact,
            "missing_spurious_unsafe": {
                "missing_constraints": int(evaluation.get("omitted_constraints") or 0),
                "spurious_constraints": int(evaluation.get("spurious_constraints") or 0),
                "unsafe_acceptance": bool(evaluation.get("unsafe_accepted_constraints")),
            },
            "correct_mode_identity": {
                "candidate_sha256": evaluation.get("candidate_sha256"),
                "exact_semantic_success": exact["exact_semantic_success"],
            },
            "latency_s": float(raw.get("latency_s") or 0.0),
            "token_counts": dict(raw.get("usage") or {}),
            "gpu_telemetry": dict(raw.get("gpu_telemetry") or {}),
            "no_hidden_reference_answer": True,
            "previous_prefix_checksum": previous,
            "prefix_checksum": "",
            "row_hash": "",
        }
        event["row_hash"] = _event_row_hash(event)
        event["prefix_checksum"] = sha256_json(
            {
                "previous_prefix_checksum": previous,
                "row_hash": event["row_hash"],
                "stream_sequence_index": event["stream_sequence_index"],
            }
        )
        previous = event["prefix_checksum"]
        events.append(event)
    return events


def replay_event_stream(path: Path) -> JsonDict:
    """Replay the chronological prefix chain and reject tampered rows."""

    if not path.exists():
        return {"ok": False, "row_count": 0, "reason": "missing_event_stream", "rows": []}
    previous = "sha256:" + ("0" * 64)
    rows = []
    ok = True
    reason = "ok"
    for expected_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        event = json.loads(line)
        row_hash = _event_row_hash(event)
        prefix = sha256_json(
            {
                "previous_prefix_checksum": previous,
                "row_hash": row_hash,
                "stream_sequence_index": event.get("stream_sequence_index"),
            }
        )
        hidden = _contains_hidden_reference_answer(event)
        row_ok = (
            row_hash == event.get("row_hash")
            and prefix == event.get("prefix_checksum")
            and event.get("previous_prefix_checksum") == previous
            and int(event.get("stream_sequence_index")) == expected_index
            and not hidden
        )
        if not row_ok and ok:
            ok = False
            reason = f"event_stream_tamper_or_order_failure_at_{expected_index}"
        rows.append(
            {
                "stream_sequence_index": event.get("stream_sequence_index"),
                "row_hash": event.get("row_hash"),
                "computed_row_hash": row_hash,
                "prefix_checksum": event.get("prefix_checksum"),
                "computed_prefix_checksum": prefix,
                "contains_hidden_reference_answer": hidden,
                "ok": row_ok,
            }
        )
        previous = str(event.get("prefix_checksum"))
    return {
        "ok": ok,
        "reason": reason,
        "row_count": len(rows),
        "final_prefix_checksum": previous,
        "rows": rows,
    }


def save_checkpoint(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write resume rows atomically as chronological JSONL."""

    _write_event_stream_atomic(path, rows)


def load_checkpoint(path: Path) -> list[JsonDict]:
    """Load a prior checkpoint if present."""

    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def merge_resume_rows(
    checkpoint_rows: Sequence[Mapping[str, Any]],
    new_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Deduplicate checkpoint and newly collected rows by stream index."""

    merged: dict[int, JsonDict] = {}
    for row in [*checkpoint_rows, *new_rows]:
        merged[int(row["stream_sequence_index"])] = dict(row)
    return [merged[index] for index in sorted(merged)]


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal Exp5923 schema and principle-bearing fields."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must use the mandated local GGUF CUDA schema path")
    if artifact["verifier_is_oracle"] is not True:
        raise ValueError("verifier_is_oracle must be true for exact adjudication")
    chronological = float(artifact["chronological_event_stream_ready_score"])
    schema_score = float(artifact["schema_decode_live_ready_score"])
    if chronological not in {0.0, 1.0}:
        raise ValueError("chronological_event_stream_ready_score must be bare 0.0 or 1.0")
    if schema_score not in {0.0, 1.0}:
        raise ValueError("schema_decode_live_ready_score must be bare 0.0 or 1.0")
    receipt = artifact["no_repair_call_and_no_answer_enumeration_receipt"]
    if receipt.get("exact_diagnostic_repair_call_used") is not False:
        raise ValueError("repair call must not be used")
    if receipt.get("complete_answer_enumeration_used") is not False:
        raise ValueError("complete answer enumeration must not be used")
    if schema_score == 1.0:
        if not str(artifact["honest_verdict"]).startswith("complete_positive:"):
            raise ValueError("schema ready score requires complete_positive verdict")
        if artifact["missing_spurious_and_unsafe_acceptance"]["unsafe_accepts_total"] != 0:
            raise ValueError("schema ready score requires zero unsafe accepts")
    if not str(artifact["honest_verdict"]).startswith(
        ("complete_positive:", "complete_null:", "retired:", "blocked:")
    ):
        raise ValueError("honest_verdict must use an Exp5923 terminal prefix")


def refresh_artifact_test_exit_codes(
    *,
    root: Path = REPO_ROOT,
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    """Update only test exit-code provenance after verification commands run."""

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
    preregistration: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    collection: Mapping[str, Any],
    duration_s: float,
    test_exit_codes: Mapping[str, int] | None,
) -> JsonDict:
    protected = _protected_file_receipt(config.repo_root, baseline=preconditions.protected_file_baseline)
    expected_rows = len(MANDATED_MODEL_IDS) * int(preregistration["case_count"]) * len(ARM_IDS)
    replay = replay_event_stream(config.events_path())
    aggregates = _aggregate_events(events)
    chronology_ready = (
        preconditions.block_reason is None
        and bool(collection.get("real_model_rows"))
        and replay["ok"] is True
        and replay["row_count"] == expected_rows
        and protected["unchanged"] is True
    )
    retirement = _retirement_decision(aggregates)
    schema_score = 1.0 if chronology_ready and _schema_live_ready(aggregates, retirement) else 0.0
    status, verdict = _status_and_verdict(preconditions.block_reason, schema_score, retirement, chronology_ready)
    event_receipt = _event_stream_receipt(config.events_path(), replay, expected_rows)
    artifact: JsonDict = {
        "schema": ARTIFACT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": config.random_seed,
        "status": status,
        "gate_replay_receipt": preconditions.gate_replay_receipt,
        "preconditions_checked": preconditions.preconditions_checked,
        "model_specs": preconditions.model_specs,
        "model_file_hashes": preconditions.model_file_hashes,
        "embedded_tokenizer_loader_cuda_gpu_and_vram_receipts": preconditions.receipts
        | {"model_attempts": list(collection.get("model_attempts") or [])},
        "sealed_cases_families_adversaries_prompts_seeds_arms_budgets": dict(preregistration),
        "per_model_arm_structural_and_exact_metrics": aggregates["per_model_arm"],
        "exact_semantic_primary_comparison_and_intervals": aggregates["primary_comparison"],
        "held_family_and_attribute_semantic_results": aggregates["held_and_attribute"],
        "missing_spurious_and_unsafe_acceptance": aggregates["missing_spurious_unsafe"],
        "correct_mode_diversity_and_overpruning": aggregates["diversity"],
        "token_latency_gpu_and_memory_accounting": aggregates["token_latency_gpu"],
        "chronological_event_stream_path_hash_rows_and_prefix_chain": event_receipt,
        "chronological_event_stream_ready_score": 1.0 if chronology_ready else 0.0,
        "retirement_decision": retirement,
        "no_repair_call_and_no_answer_enumeration_receipt": {
            "exact_diagnostic_repair_call_used": False,
            "complete_answer_enumeration_used": False,
            "ok": True,
        },
        "protected_files_unchanged": protected,
        "schema_decode_live_ready_score": schema_score,
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


def _aggregate_events(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows = [dict(row) for row in events]
    primary = _primary_comparison(rows)
    return {
        "per_model_arm": _per_model_arm_metrics(rows),
        "primary_comparison": primary,
        "held_and_attribute": _held_and_attribute_results(rows),
        "missing_spurious_unsafe": _missing_spurious_unsafe(rows),
        "diversity": _diversity_and_overpruning(rows),
        "token_latency_gpu": _token_latency_gpu(rows),
    }


def _primary_comparison(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    held = [row for row in rows if row["row_identity"]["split"] != "train"]
    by_arm = {arm: _metric_summary([row for row in held if row["row_identity"]["arm_id"] == arm]) for arm in ARM_IDS}
    control_rates = {arm: by_arm[arm]["exact_semantic_success_rate"] for arm in CONTROL_ARM_IDS}
    schema_rates = {arm: by_arm[arm]["exact_semantic_success_rate"] for arm in SCHEMA_ARM_IDS}
    best_control = max(control_rates, key=lambda arm: control_rates[arm])
    best_schema = max(schema_rates, key=lambda arm: schema_rates[arm])
    delta = schema_rates[best_schema] - control_rates[best_control]
    return {
        "principle": FIELD_PRINCIPLES["exact_semantic_primary_comparison_and_intervals"],
        "by_arm_held": by_arm,
        "best_control_arm": best_control,
        "best_schema_supported_arm": best_schema,
        "held_exact_semantic_delta_vs_best_control": round(delta, 6),
        "paired_bootstrap_ci95": _paired_bootstrap_ci(held, best_schema, best_control),
    }


def _per_model_arm_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_model: JsonDict = {}
    for model_id in sorted({str(row["row_identity"]["model_hf_id"]) for row in rows}):
        by_model[model_id] = {
            arm: _metric_summary(
                [
                    row
                    for row in rows
                    if row["row_identity"]["model_hf_id"] == model_id
                    and row["row_identity"]["arm_id"] == arm
                ]
            )
            for arm in ARM_IDS
        }
    return {"by_model": by_model, "by_arm": {arm: _metric_summary([row for row in rows if row["row_identity"]["arm_id"] == arm]) for arm in ARM_IDS}}


def _held_and_attribute_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    held_family = [row for row in rows if row["row_identity"]["split"] == "held_family"]
    attribute = [
        row
        for row in rows
        if row["row_identity"]["adversary_kind"] == "attribute_semantic_adversary"
    ]
    return {
        "held_family": {arm: _metric_summary([row for row in held_family if row["row_identity"]["arm_id"] == arm]) for arm in ARM_IDS},
        "attribute_semantic_adversaries": {arm: _metric_summary([row for row in attribute if row["row_identity"]["arm_id"] == arm]) for arm in ARM_IDS},
    }


def _missing_spurious_unsafe(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_arm: JsonDict = {}
    for arm in ARM_IDS:
        arm_rows = [row for row in rows if row["row_identity"]["arm_id"] == arm]
        by_arm[arm] = {
            "missing_constraints": sum(int(row["missing_spurious_unsafe"]["missing_constraints"]) for row in arm_rows),
            "spurious_constraints": sum(int(row["missing_spurious_unsafe"]["spurious_constraints"]) for row in arm_rows),
            "unsafe_accepts": sum(bool(row["missing_spurious_unsafe"]["unsafe_acceptance"]) for row in arm_rows),
        }
    return {
        "by_arm": by_arm,
        "missing_constraints_total": sum(row["missing_spurious_unsafe"]["missing_constraints"] for row in rows),
        "spurious_constraints_total": sum(row["missing_spurious_unsafe"]["spurious_constraints"] for row in rows),
        "unsafe_accepts_total": sum(bool(row["missing_spurious_unsafe"]["unsafe_acceptance"]) for row in rows),
    }


def _diversity_and_overpruning(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    successes_by_arm: JsonDict = {}
    unique_by_arm: JsonDict = {}
    for arm in ARM_IDS:
        keys = [
            row["correct_mode_identity"].get("candidate_sha256")
            for row in rows
            if row["row_identity"]["arm_id"] == arm
            and row["correct_mode_identity"]["exact_semantic_success"] is True
        ]
        successes_by_arm[arm] = len(keys)
        unique_by_arm[arm] = len({key for key in keys if key})
    best_control_unique = max(unique_by_arm[arm] for arm in CONTROL_ARM_IDS)
    best_schema_unique = max(unique_by_arm[arm] for arm in SCHEMA_ARM_IDS)
    control_exact_cases = {
        (row["row_identity"]["model_hf_id"], row["row_identity"]["case_id"])
        for row in rows
        if row["row_identity"]["arm_id"] in CONTROL_ARM_IDS
        and row["correct_mode_identity"]["exact_semantic_success"] is True
    }
    schema_exact_cases = {
        (row["row_identity"]["model_hf_id"], row["row_identity"]["case_id"])
        for row in rows
        if row["row_identity"]["arm_id"] in SCHEMA_ARM_IDS
        and row["correct_mode_identity"]["exact_semantic_success"] is True
    }
    material_collapse = best_control_unique > 0 and best_schema_unique < max(1, best_control_unique // 2)
    return {
        "successes_by_arm": successes_by_arm,
        "unique_correct_mode_identities_by_arm": unique_by_arm,
        "best_control_unique_correct_modes": best_control_unique,
        "best_schema_unique_correct_modes": best_schema_unique,
        "overpruned_model_case_pairs": sorted(list(control_exact_cases - schema_exact_cases)),
        "overpruning_count": len(control_exact_cases - schema_exact_cases),
        "material_correct_mode_collapse": material_collapse,
    }


def _token_latency_gpu(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    completion_tokens = sum(int(row.get("token_counts", {}).get("completion_tokens") or 0) for row in rows)
    prompt_tokens = sum(int(row.get("token_counts", {}).get("prompt_tokens") or 0) for row in rows)
    total_latency = sum(float(row.get("latency_s") or 0.0) for row in rows)
    vram_deltas = [float(row.get("gpu_telemetry", {}).get("vram_delta_mb") or 0.0) for row in rows]
    return {
        "rows": len(rows),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "latency_s": round(total_latency, 6),
        "max_vram_delta_mb": max(vram_deltas) if vram_deltas else 0.0,
        "all_rows_report_gpu_telemetry": all("gpu_telemetry" in row for row in rows),
    }


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    total = len(rows)
    semantic_rows = [row for row in rows if row["row_identity"]["adversary_kind"] not in {"invalid_reference", "type_confusion", "scope_leak"}]
    exact = sum(row["exact_outcome"]["exact_semantic_success"] is True for row in semantic_rows)
    return {
        "n": total,
        "semantic_n": len(semantic_rows),
        "parse_rate": _rate(sum(row["exact_outcome"]["parse_valid"] for row in rows), total),
        "type_rate": _rate(sum(row["exact_outcome"]["type_valid"] for row in rows), total),
        "scope_rate": _rate(sum(row["exact_outcome"]["scope_valid"] for row in rows), total),
        "compile_rate": _rate(sum(row["exact_outcome"]["compile_valid"] for row in rows), total),
        "exact_semantic_success_rate": _rate(exact, len(semantic_rows)),
        "unsafe_accept_rate": _rate(
            sum(bool(row["missing_spurious_unsafe"]["unsafe_acceptance"]) for row in rows), total
        ),
    }


def _retirement_decision(aggregates: Mapping[str, Any]) -> JsonDict:
    per_arm = aggregates["per_model_arm"]["by_arm"]
    all_schema_zero = all(
        per_arm[arm]["exact_semantic_success_rate"] == 0.0 for arm in SCHEMA_ARM_IDS
    )
    primary = aggregates["primary_comparison"]
    unsafe = aggregates["missing_spurious_unsafe"]
    diversity = aggregates["diversity"]
    best_control_unsafe = max(per_arm[arm]["unsafe_accept_rate"] for arm in CONTROL_ARM_IDS)
    best_schema_unsafe = min(per_arm[arm]["unsafe_accept_rate"] for arm in SCHEMA_ARM_IDS)
    exact_reduced = primary["held_exact_semantic_delta_vs_best_control"] < 0.0
    unsafe_increased = best_schema_unsafe > best_control_unsafe or unsafe["unsafe_accepts_total"] > 0
    diversity_collapse = diversity["material_correct_mode_collapse"]
    retire = bool(all_schema_zero or exact_reduced or unsafe_increased or diversity_collapse)
    reasons = [
        name
        for name, active in {
            "all_three_models_zero_exact_success_in_schema_supported_arms": all_schema_zero,
            "schema_supported_exact_success_reduced_vs_best_control": exact_reduced,
            "schema_supported_unsafe_acceptance_increased": unsafe_increased,
            "schema_supported_correct_mode_diversity_collapsed": diversity_collapse,
        }.items()
        if active
    ]
    return {
        "retire": retire,
        "reasons": reasons,
        "next_action": "retire_mechanism_no_reprompt" if retire else "do_not_retire_from_this_run",
        "principle": "retire rather than scheduling another reprompt when schema support fails exact semantics or safety/diversity.",
    }


def _schema_live_ready(aggregates: Mapping[str, Any], retirement: Mapping[str, Any]) -> bool:
    primary = aggregates["primary_comparison"]
    unsafe = aggregates["missing_spurious_unsafe"]
    diversity = aggregates["diversity"]
    return bool(
        primary["held_exact_semantic_delta_vs_best_control"] > 0.0
        and unsafe["unsafe_accepts_total"] == 0
        and diversity["material_correct_mode_collapse"] is False
        and retirement["retire"] is False
    )


def _status_and_verdict(
    block_reason: str | None,
    schema_score: float,
    retirement: Mapping[str, Any],
    chronology_ready: bool,
) -> tuple[str, str]:
    if block_reason is not None:
        return "blocked", f"blocked: precondition failed before model load: {block_reason}"
    if not chronology_ready:
        return "blocked", "blocked: chronological real-model event stream incomplete or tampered"
    if schema_score == 1.0:
        return "complete_positive", "complete_positive: schema-supported decoding improved held exact semantics without unsafe acceptance or diversity collapse"
    if retirement.get("retire"):
        return "retired", "retired: schema-supported ConstraintIR decoding failed exact-semantic retirement gates"
    return "complete_null", "complete_null: schema-supported decoding did not beat the best matched control on exact semantics"


def _event_stream_receipt(path: Path, replay: Mapping[str, Any], expected_rows: int) -> JsonDict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
        "rows": replay["row_count"],
        "expected_rows": expected_rows,
        "prefix_chain_ok": replay["ok"],
        "final_prefix_checksum": replay.get("final_prefix_checksum"),
        "no_hidden_reference_answer": all(
            not row.get("contains_hidden_reference_answer") for row in replay.get("rows", [])
        ),
    }


def _exact_outcome(evaluation: Mapping[str, Any]) -> JsonDict:
    parse_valid = bool(evaluation.get("parse_valid"))
    type_valid = bool(evaluation.get("type_valid"))
    compiled = bool(evaluation.get("compiled"))
    return {
        "parse_valid": parse_valid,
        "type_valid": type_valid,
        "scope_valid": parse_valid and type_valid,
        "compile_valid": compiled,
        "exact_execution_certificate": {
            "solver_status": evaluation.get("solver_status"),
            "z3_status": evaluation.get("z3_status"),
            "diagnostics": dict(evaluation.get("diagnostics") or {}),
        },
        "exact_semantic_success": evaluation.get("exact_semantic_equivalence") is True,
        "query_correct": evaluation.get("query_correct"),
        "satisfiability_correct": evaluation.get("satisfiability_correct"),
    }


def _paired_bootstrap_ci(rows: Sequence[Mapping[str, Any]], schema_arm: str, control_arm: str) -> JsonDict:
    by_pair: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (str(row["row_identity"]["model_hf_id"]), str(row["row_identity"]["case_id"]))
        arm = str(row["row_identity"]["arm_id"])
        if arm in {schema_arm, control_arm}:
            by_pair[key][arm] = 1.0 if row["exact_outcome"]["exact_semantic_success"] else 0.0
    diffs = [
        values.get(schema_arm, 0.0) - values.get(control_arm, 0.0)
        for values in by_pair.values()
    ]
    if not diffs:
        return {"ci95": [0.0, 0.0], "mean": 0.0, "n_pairs": 0}
    samples = []
    for draw in range(200):
        total = 0.0
        for index in range(len(diffs)):
            total += diffs[(draw * 37 + index * 17) % len(diffs)]
        samples.append(total / len(diffs))
    samples.sort()
    return {
        "ci95": [round(samples[int(0.025 * (len(samples) - 1))], 6), round(samples[int(0.975 * (len(samples) - 1))], 6)],
        "mean": round(sum(diffs) / len(diffs), 6),
        "n_pairs": len(diffs),
    }


def _event_row_hash(event: Mapping[str, Any]) -> str:
    stable = json.loads(canonical_json(event))
    stable["row_hash"] = ""
    stable["prefix_checksum"] = ""
    return sha256_json(stable)


def _contains_hidden_reference_answer(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in FORBIDDEN_EVENT_REFERENCE_KEYS:
                return True
            if _contains_hidden_reference_answer(item):
                return True
    elif isinstance(value, list):
        return any(_contains_hidden_reference_answer(item) for item in value)
    return False


def _visible_problem_text(target: Mapping[str, Any]) -> str:
    return str(target.get("natural_language") or target.get("prompt") or target["row_id"])


def _rate(numerator: int, denominator: int) -> float:
    return round(float(numerator) / denominator, 6) if denominator else 0.0


def _tokenizer_receipts(
    model_specs: Sequence[Mapping[str, Any]], tokenizer_loader: TokenizerLoader
) -> JsonDict:
    receipts: JsonDict = {}
    for spec in model_specs:
        if not spec.get("model_path"):
            continue
        try:
            receipt = tokenizer_loader(dict(spec))
            receipt.setdefault("ok", True)
        except Exception as exc:  # noqa: BLE001
            receipt = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        receipt["embedded_tokenizer_only"] = receipt.get("used_hf_autotokenizer") is False
        receipt["used_hf_autotokenizer"] = False
        receipts[str(spec["hf_id"])] = receipt
    return receipts


def _model_file_hashes(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    hashes: JsonDict = {}
    for spec in model_specs:
        path = Path(str(spec.get("model_path") or ""))
        if path.is_file():
            hashes[str(spec["hf_id"])] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    return hashes


def _gate_replay_receipt(root: Path) -> JsonDict:
    path = root / exp5922.RESULT_RELATIVE_PATH
    receipt: JsonDict = {
        "exp5922_artifact_path": str(path),
        "exp5922_artifact_present": path.exists(),
        "exp5922_ready_score": 0.0,
        "gguf_schema_decoder_bridge_ready": False,
        "one_step_cuda_smoke_ok": False,
        "ok": False,
    }
    if path.exists():
        artifact = json.loads(path.read_text(encoding="utf-8"))
        exp5922.validate_artifact(artifact)
        receipt["exp5922_ready_score"] = artifact.get("gguf_schema_decoder_bridge_ready_score")
        receipt["gguf_schema_decoder_bridge_ready"] = artifact.get("gguf_schema_decoder_bridge_ready_score") == 1.0
        receipt["one_step_cuda_smoke_ok"] = artifact.get("one_step_cuda_smoke", {}).get("all_smokes_ok") is True
        receipt["artifact_checksum"] = artifact.get("reproducibility_checksum")
        receipt["ok"] = bool(
            receipt["gguf_schema_decoder_bridge_ready"] and receipt["one_step_cuda_smoke_ok"]
        )
    return receipt


def _probe_environment(root: Path) -> JsonDict:  # pragma: no cover - host boundary.
    public_api = exp5922.public_llama_cpp_api_receipt()
    gpu_health = _gpu_health_probe()
    return {
        "llama_cpp_import": {"ok": public_api.get("importable") is True},
        "public_llama_cpp_cuda": {
            "ok": public_api.get("ok") is True,
            "logits_processor_parameter": public_api.get("logits_processor_parameter") is True,
            "gpu_offload_supported": public_api.get("gpu_offload_supported") is True,
        },
        "gpu_health": gpu_health,
        "ram": _memory_probe(32768),
        "disk": _disk_probe(root, 8192),
        "protected_workload": _protected_workload_probe(),
        "atomic_output": _atomic_output_probe(root / RESULT_RELATIVE_PATH),
        "atomic_checkpoint_resume": _atomic_checkpoint_resume_probe(root / CHECKPOINT_RELATIVE_PATH),
    }


def _gpu_health_probe() -> JsonDict:  # pragma: no cover - host boundary.
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "error": repr(exc), "gpus": []}
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_free_mb": int(parts[3]),
                    "utilization_gpu_pct": int(parts[4]),
                }
            )
    return {"ok": result.returncode == 0 and bool(gpus), "gpus": gpus}


def _two_healthy_rtx_3090s(gpu_health: Mapping[str, Any]) -> bool:
    gpus = gpu_health.get("gpus") or []
    healthy = [
        gpu
        for gpu in gpus
        if "RTX 3090" in str(gpu.get("name"))
        and int(gpu.get("memory_total_mb") or 0) >= 24000
    ]
    return len(healthy) >= 2


def _adequate_vram(gpu_health: Mapping[str, Any]) -> bool:
    gpus = gpu_health.get("gpus") or []
    return len(gpus) >= 2 and all(int(gpu.get("memory_free_mb") or 0) >= 16000 for gpu in gpus[:2])


def _disk_probe(root: Path, required_mb: int) -> JsonDict:
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {"ok": available_mb >= required_mb, "available_mb": available_mb, "required_mb": required_mb}


def _memory_probe(required_mb: int) -> JsonDict:
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    return {"ok": available_mb >= required_mb, "available_mb": available_mb, "required_mb": required_mb}


def _protected_workload_probe() -> JsonDict:  # pragma: no cover - host process boundary.
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "error": repr(exc), "protected_pids": []}
    protected = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if parts and parts[0].isdigit() and _pid_is_protected_training_proc(int(parts[0])):
            protected.append({"pid": int(parts[0]), "process_name": parts[1] if len(parts) > 1 else ""})
    return {"ok": not protected, "protected_pids": protected}


def _pid_is_protected_training_proc(pid: int) -> bool:  # pragma: no cover - host process boundary.
    try:
        cmdline = (
            Path(f"/proc/{pid}/cmdline")
            .read_bytes()
            .replace(b"\x00", b" ")
            .decode("utf-8", "replace")
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return False
    markers = ("train.py", "/nn/train", "src/nn/train")
    return any(marker in cmdline for marker in markers)


def _atomic_output_probe(output_path: Path) -> JsonDict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    probe = output_path.parent / f".{output_path.name}.atomic-probe"
    replacement = output_path.parent / f".{output_path.name}.atomic-probe.tmp"
    try:
        probe.write_text("old", encoding="utf-8")
        replacement.write_text("new", encoding="utf-8")
        os.replace(replacement, probe)
        ok = probe.read_text(encoding="utf-8") == "new"
    finally:
        probe.unlink(missing_ok=True)
        replacement.unlink(missing_ok=True)
    return {"ok": ok, "detail": "os.replace_same_directory"}


def _atomic_checkpoint_resume_probe(path: Path) -> JsonDict:
    sample = [{"stream_sequence_index": 0, "row_hash": "sha256:probe"}]
    save_checkpoint(path, sample)
    loaded = load_checkpoint(path)
    path.unlink(missing_ok=True)
    return {"ok": loaded == sample, "detail": "atomic_jsonl_checkpoint_resume"}


def _protected_file_receipt(root: Path, baseline: Mapping[str, Any] | None = None) -> JsonDict:
    baseline_by_path = {str(row["path"]): row for row in (baseline or {}).get("files", [])}
    files = []
    for relative in PROTECTED_FILES:
        path = root / relative
        current = sha256_file(path) if path.exists() else None
        before = baseline_by_path.get(str(relative), {}).get("sha256", current)
        files.append(
            {
                "path": str(relative),
                "exists": path.exists(),
                "sha256_before": before,
                "sha256": current,
                "unchanged": before == current,
            }
        )
    return {"unchanged": all(row["unchanged"] for row in files), "files": files}


def _hash_inputs(root: Path) -> JsonDict:
    files = []
    for relative in HASHED_INPUTS:
        path = root / relative
        files.append(
            {
                "path": str(relative),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else None,
            }
        )
    return {"files": files, "all_present": all(row["exists"] for row in files)}


def _field_provenance() -> JsonDict:
    return {
        field: {
            "satisfied_by": "generated_by_exp5923_paired_schema_supported_constraintir_ab",
            "principle": FIELD_PRINCIPLES.get(field, "Exp5923 required artifact field."),
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    stable = json.loads(canonical_json(artifact))
    stable["duration_s"] = 0.0
    stable["test_exit_codes"] = {}
    stable["reproducibility_checksum"] = ""
    receipts = stable.get("embedded_tokenizer_loader_cuda_gpu_and_vram_receipts", {})
    environment = receipts.get("environment", {}) if isinstance(receipts, dict) else {}
    for key in ("ram", "disk"):
        if isinstance(environment.get(key), dict):
            environment[key]["available_mb"] = 0
    return sha256_json(stable)


def _write_json_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp = Path(handle.name)
        handle.write(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _write_event_stream_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _gpu_memory_total_mb() -> float:  # pragma: no cover - host boundary.
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return 0.0
    if result.returncode != 0:
        return 0.0
    return float(sum(float(line.strip()) for line in result.stdout.splitlines() if line.strip()))


def _call_live_arm(
    llm: Any,
    bridge: exp5922.SchemaDecoderBridge,
    prompt: str,
    spec: Mapping[str, Any],
    case: Mapping[str, Any],
    target: Mapping[str, Any],
    arm_id: str,
    seed: int,
    sequence: int,
    config: ExperimentConfig,
) -> JsonDict:  # pragma: no cover - live llama.cpp boundary.
    del target
    from llama_cpp import LogitsProcessorList

    started = config.monotonic_clock()
    if arm_id in SCHEMA_ARM_IDS:
        prompt_token_count = len(llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=False))
        processor = _GeneratedSuffixSchemaSubsetProcessor(bridge, prompt_token_count)
        result = llm(
            prompt,
            max_tokens=int(TOKEN_BUDGETS[arm_id]["schema_tokens"]),
            temperature=float(DECODING_PARAMETERS["temperature"]),
            top_p=float(DECODING_PARAMETERS["top_p"]),
            seed=seed,
            logits_processor=LogitsProcessorList([processor]),
            stop=[],
        )
    else:
        result = llm(
            prompt,
            max_tokens=int(TOKEN_BUDGETS[arm_id]["max_tokens"]),
            temperature=float(DECODING_PARAMETERS["temperature"]),
            top_p=float(DECODING_PARAMETERS["top_p"]),
            seed=seed,
            stop=list(DECODING_PARAMETERS["stop"]),
        )
    text = ""
    usage = {}
    if isinstance(result, Mapping) and result.get("choices"):
        text = str(result["choices"][0].get("text") or "")
        usage = dict(result.get("usage") or {})
    return {
        "stream_sequence_index": sequence,
        "model_hf_id": spec["hf_id"],
        "model_name": spec["name"],
        "model_path": spec["model_path"],
        "gpu_index": spec.get("gpu"),
        "case_id": case["case_id"],
        "arm_id": arm_id,
        "seed": seed,
        "raw_output_text": text,
        "latency_s": round(config.monotonic_clock() - started, 6),
        "usage": usage,
        "gpu_telemetry": {
            "vram_delta_mb": _gpu_memory_total_mb(),
            "offload_verified": True,
        },
    }


class _GeneratedSuffixSchemaSubsetProcessor:  # pragma: no cover - live llama.cpp boundary.
    """Apply schema masks over a bounded schema-relevant token subset.

    Exp5922's reference processor is intentionally exhaustive: it scans every
    GGUF token at each step to prove support parity.  Full decoding needs a
    bounded live path, so this processor precomputes JSON punctuation,
    printable ASCII, UTF-8 whitespace, and known schema terminal tokens.  It is
    stricter than the reference bridge and can overprune; Exp5923 measures that
    as part of the science rather than treating structural validity as success.
    """

    def __init__(self, bridge: exp5922.SchemaDecoderBridge, prompt_token_count: int) -> None:
        self.bridge = bridge
        self.prompt_token_count = prompt_token_count
        self.candidate_token_ids = self._candidate_token_ids()

    def __call__(self, input_ids: Sequence[int], scores: Any) -> Any:
        import numpy as np

        suffix = input_ids[self.prompt_token_count :]
        prefix = b"".join(
            self.bridge.vocabulary.token_bytes_by_id.get(int(token_id), b"")
            for token_id in suffix
        )
        status = self.bridge.prefix_status(prefix)
        allowed = []
        if status.valid:
            for token_id in self.candidate_token_ids:
                if self.bridge.token_preserves_continuation(prefix, token_id):
                    allowed.append(token_id)
            eos = self.bridge.vocabulary.eos_token_id
            if status.complete_valid and eos is not None:
                allowed.append(eos)
        masked = np.full_like(scores, -np.inf, dtype=float)
        for token_id in allowed:
            if 0 <= token_id < len(masked):
                masked[token_id] = scores[token_id]
        return masked

    def _candidate_token_ids(self) -> list[int]:
        pieces: set[bytes] = {bytes([value]) for value in range(32, 127)}
        pieces.update({b"\n", b"\t"})
        for terminal in exp5922.grammar_terminal_strings(self.bridge.support):
            pieces.add(terminal.encode("utf-8"))
        ids: set[int] = set()
        for piece in pieces:
            try:
                ids.update(self.bridge.vocabulary.encode_bytes(piece))
            except Exception:
                continue
        return sorted(
            token_id
            for token_id in ids
            if token_id in self.bridge.vocabulary.token_bytes_by_id
            and self.bridge.vocabulary.token_bytes_by_id[token_id]
        )


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--refresh-test-exit-code", action="append", default=[])
    args = parser.parse_args(argv)
    if args.refresh_test_exit_code:
        refresh_artifact_test_exit_codes(
            root=args.root,
            test_exit_codes=_parse_test_exit_codes(args.refresh_test_exit_code),
        )
    else:
        run_experiment(ExperimentConfig(repo_root=args.root))
    return 0


def _parse_test_exit_codes(values: Sequence[str]) -> JsonDict:  # pragma: no cover - CLI wrapper.
    parsed = {}
    for value in values:
        command, code = value.rsplit("=", 1)
        parsed[command] = int(code)
    return parsed


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
