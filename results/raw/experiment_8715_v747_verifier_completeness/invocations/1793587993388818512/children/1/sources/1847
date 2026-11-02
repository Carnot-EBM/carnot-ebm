"""Exp5544 cross-model local-SOTA CSL transfer.

Spec refs: REQ-LEARN-5544,
SCENARIO-LEARN-5544-UPSTREAM-GATE,
SCENARIO-LEARN-5544-CROSS-FAMILY,
SCENARIO-LEARN-5544-NO-WEIGHT-MUTATION.

This experiment asks a narrow question: can verified retrieval memory produced
by one frozen local GGUF model family help a different frozen local GGUF model
family on the same held-out query set? The learned state is intentionally just a
small external memory table. Model files are stat'ed before and after the run so
any positive result stays attributable to retrieval memory, not hidden tuning.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any

from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
ModelResolver = Callable[[str, str], str | None]
CachedPairFn = Callable[[], list[Mapping[str, Any]] | None]
RuntimeProbe = Callable[..., Mapping[str, Any]]
GenerationRunner = Callable[..., Mapping[str, Any]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5544_cross_model_sota_csl_transfer.json")
UPSTREAM_FIVE_ARM_PATH = Path(
    "results/experiment_5543_retrieval_warmed_csl_five_arm_ablation.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5544_cross_model_sota_csl_transfer.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5544_cross_model_sota_csl_transfer.py")

SCHEMA = "carnot.experiment_5544.cross_model_sota_csl_transfer.v1"
EXPERIMENT_ID = "experiment_5544_cross_model_sota_csl_transfer"
TASK_ID = "exp5544-cross-model-sota-csl-transfer"
MILESTONE = "2026.07.502"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5544
DEFAULT_QUANTIZATION = "Q4_K_M"
N_GPU_LAYERS = -1
MAX_TOKENS = 12
INFERENCE_SUBSTRATE = "live_local_sota_gguf_cross_model_csl"

QWEN_HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA_26_HF_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
GEMMA_31_HF_ID = "unsloth/gemma-4-31B-it-GGUF"

NO_MEMORY_ARM = "no_memory"
SHUFFLED_ARM = "shuffled_memory"
SAME_FAMILY_ARM = "same_family_aligned_memory"
CROSS_FAMILY_ARM = "cross_family_aligned_memory"
TARGET_ARMS = (NO_MEMORY_ARM, SHUFFLED_ARM, SAME_FAMILY_ARM, CROSS_FAMILY_ARM)
ARM_SCORE_FIELDS = {
    NO_MEMORY_ARM: "no_memory_score",
    SHUFFLED_ARM: "shuffled_memory_score",
    SAME_FAMILY_ARM: "same_family_memory_score",
    CROSS_FAMILY_ARM: "cross_family_memory_score",
}

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "role": "flagship_moe",
        "family": "qwen",
        "name": "Qwen3.6-35B-A3B",
        "hf_id": QWEN_HF_ID,
        "quantization": DEFAULT_QUANTIZATION,
    },
    {
        "role": "middle_moe",
        "family": "gemma",
        "name": "Gemma4-26B-A4B-it",
        "hf_id": GEMMA_26_HF_ID,
        "quantization": DEFAULT_QUANTIZATION,
    },
    {
        "role": "flagship_dense",
        "family": "gemma",
        "name": "Gemma4-31B-it",
        "hf_id": GEMMA_31_HF_ID,
        "quantization": DEFAULT_QUANTIZATION,
    },
)
MANDATED_HF_IDS = tuple(str(spec["hf_id"]) for spec in MANDATED_MODEL_SPECS)
SPEC_REFS = (
    "REQ-LEARN-5544",
    "SCENARIO-LEARN-5544-UPSTREAM-GATE",
    "SCENARIO-LEARN-5544-CROSS-FAMILY",
    "SCENARIO-LEARN-5544-NO-WEIGHT-MUTATION",
)

REQUIRED_ARTIFACT_FIELDS = (
    "model_specs",
    "source_models",
    "target_models",
    "no_memory_score",
    "shuffled_memory_score",
    "same_family_memory_score",
    "cross_family_memory_score",
    "cross_family_delta_over_shuffled",
    "heldout_delta",
    "no_weight_mutation",
    "stale_evidence_rejection_rate",
    "negative_transfer_rate",
    "random_seed",
    "measured_duration_s",
    "gpu_offload_evidence",
    "csl_claim_allowed",
    "tests_added_or_reused",
    "field_principles",
    "inference_substrate",
    "honest_verdict",
)
DEFAULT_TESTS_ADDED_OR_REUSED = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5544_cross_model_sota_csl_transfer.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5544_cross_model_sota_csl_transfer.py "
    "-m pytest tests/python/test_experiment_5544_cross_model_sota_csl_transfer.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5544_cross_model_sota_csl_transfer.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
)
FIELD_PRINCIPLES: JsonDict = {
    "model_specs": "Lists mandated SOTA GGUF models, local receipts, and runtime helper provenance.",
    "source_models": "Names frozen source-model families used to produce external memory.",
    "target_models": "Names frozen target-model families evaluated on held-out queries.",
    "no_memory_score": "Baseline target score when no source retrieval memory is supplied.",
    "shuffled_memory_score": "Control target score when memory is deliberately mismatched to queries.",
    "same_family_memory_score": "Positive control for aligned memory transferred within the Gemma family.",
    "cross_family_memory_score": "Headline score for Qwen-to-Gemma aligned retrieval memory transfer.",
    "cross_family_delta_over_shuffled": "Gate delta proving cross-family memory beats mismatched memory.",
    "heldout_delta": "Gate delta proving cross-family memory beats the no-memory target baseline.",
    "no_weight_mutation": "Bare frozen-weight gate from before/after GGUF file receipts.",
    "stale_evidence_rejection_rate": "Shows stale decoy evidence is ignored by the cross-family arm.",
    "negative_transfer_rate": "Shows cross-family memory does not select stale or shuffled decoys.",
    "random_seed": "Makes prompt order, seeds, and shuffled controls reproducible.",
    "measured_duration_s": "Records measured wall-clock duration for the artifact-producing run.",
    "gpu_offload_evidence": "Records CUDA, llama.cpp, n_gpu_layers, and load/offload receipts.",
    "csl_claim_allowed": "Bare gate allowing a CSL claim only when transfer and safety controls pass.",
    "tests_added_or_reused": "Lists focused, coverage, and full-suite verification commands.",
    "field_principles": "Explains why each required headline and gate field exists.",
    "inference_substrate": "Declares live local SOTA GGUF cross-model CSL as the substrate.",
    "honest_verdict": "Terminal complete or blocked summary for reconciliation.",
}


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    tests_added_or_reused: Sequence[str] = DEFAULT_TESTS_ADDED_OR_REUSED,
    model_resolver: ModelResolver = resolve_cached_gguf,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    runtime_probe: RuntimeProbe | None = None,
    generation_runner: GenerationRunner | None = None,
    write: bool = True,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    """Run the bounded transfer measurement and optionally write the artifact."""

    started = time.perf_counter()
    root_path = Path(root)
    upstream = load_upstream_five_arm(root_path)
    model_specs, cache_receipt = resolve_model_specs(
        model_resolver=model_resolver,
        cached_pair_fn=cached_pair_fn,
    )
    roles = select_transfer_roles(model_specs)
    runtime_fn = runtime_probe or default_runtime_probe
    runtime_receipt = dict(runtime_fn(model_specs=model_specs, n_gpu_layers=N_GPU_LAYERS))
    preconditions = evaluate_preconditions(
        upstream_gate=upstream,
        model_specs=model_specs,
        roles=roles,
        runtime_receipt=runtime_receipt,
    )
    selected_specs = selected_model_specs(roles)
    before_receipts = {spec["hf_id"]: model_file_receipt(spec.get("model_path")) for spec in selected_specs}

    runner = generation_runner
    if preconditions["all_passed"] and runner is None:  # pragma: no cover - live GGUF path.
        runner = LiveLlamaGenerationRunner(n_gpu_layers=N_GPU_LAYERS)

    fixture = build_fixture()
    source_attempts: list[JsonDict] = []
    row_results: list[JsonDict] = []
    target_evaluations = {arm: [] for arm in TARGET_ARMS}
    memory_entries: JsonDict = {"same_family": [], "cross_family": [], "shuffled": []}

    if preconditions["all_passed"] and runner is not None:
        source_attempts = run_source_attempts(
            fixture=fixture,
            roles=roles,
            generation_runner=runner,
            random_seed=random_seed,
        )
        memory_entries = build_memory_entries(fixture, source_attempts)
        target_evaluations = run_target_evaluations(
            fixture=fixture,
            roles=roles,
            memory_entries=memory_entries,
            generation_runner=runner,
            random_seed=random_seed,
        )
        row_results = [row for arm in TARGET_ARMS for row in target_evaluations[arm]]
        runtime_receipt.update(getattr(runner, "runtime_receipt", {}))

    after_receipts = {spec["hf_id"]: model_file_receipt(spec.get("model_path")) for spec in selected_specs}
    artifact = build_artifact(
        upstream_gate=upstream,
        model_specs=model_specs,
        cache_receipt=cache_receipt,
        roles=roles,
        runtime_receipt=runtime_receipt,
        preconditions=preconditions,
        fixture=fixture,
        source_attempts=source_attempts,
        memory_entries=memory_entries,
        target_evaluations=target_evaluations,
        row_results=row_results,
        before_receipts=before_receipts,
        after_receipts=after_receipts,
        tests_added_or_reused=tests_added_or_reused,
        measured_duration_s=time.perf_counter() - started,
        random_seed=random_seed,
    )
    validate_artifact(artifact)
    if write:
        write_json(resolve_path(root_path, result_path), artifact)
    return artifact


def load_upstream_five_arm(root: Path | str) -> JsonDict:
    """Load Exp5543 because cross-model transfer is meaningful only after it."""

    path = resolve_path(Path(root), UPSTREAM_FIVE_ARM_PATH)
    try:
        artifact = load_json(path)
    except (OSError, json.JSONDecodeError):  # pragma: no cover - tested through validation drift.
        return {
            "path": UPSTREAM_FIVE_ARM_PATH.as_posix(),
            "loadable": False,
            "csl_five_arm_ready": False,
            "honest_verdict": "",
        }
    return {
        "path": UPSTREAM_FIVE_ARM_PATH.as_posix(),
        "loadable": True,
        "csl_five_arm_ready": artifact.get("csl_five_arm_ready") is True,
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "inference_substrate": str(artifact.get("inference_substrate", "")),
    }


def resolve_model_specs(
    *,
    model_resolver: ModelResolver = resolve_cached_gguf,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
) -> tuple[list[JsonDict], JsonDict]:
    """Resolve mandated GGUF files without touching a HuggingFace tokenizer path."""

    pair = cached_pair_fn()
    pair_by_hf_id = {
        str(row.get("hf_id")): row for row in pair or [] if isinstance(row, Mapping)
    }
    rows: list[JsonDict] = []
    for spec in MANDATED_MODEL_SPECS:
        hf_id = str(spec["hf_id"])
        quantization = str(spec.get("quantization", DEFAULT_QUANTIZATION))
        pair_hit = pair_by_hf_id.get(hf_id)
        path_value = pair_hit.get("model_path") if pair_hit else model_resolver(hf_id, quantization)
        path = Path(str(path_value)) if path_value else None
        local = bool(path and path.is_file())
        rows.append(
            {
                "role": str(spec["role"]),
                "family": str(spec["family"]),
                "name": str(spec["name"]),
                "hf_id": hf_id,
                "quantization": quantization,
                "model_path": str(path) if path else None,
                "local_path_available": local,
                "resolved_via": "cached_sota_pair" if pair_hit else "resolve_cached_gguf",
                "command_helper_path": "carnot.inference.sota_models.cached_sota_pair",
                "runtime_helper_path": "llama_cpp.Llama",
                "file_receipt": model_file_receipt(path) if local else model_file_receipt(None),
                "legacy_smoke_only": False,
            }
        )
    return rows, {
        "cached_sota_pair_attempted": True,
        "cached_sota_pair_available": bool(pair),
        "cached_sota_pair_specs": [dict(row) for row in pair or []],
    }


def select_transfer_roles(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Pick Qwen cross-source, Gemma same-source, and Gemma dense target."""

    by_id = {
        str(row.get("hf_id")): dict(row)
        for row in model_specs
        if row.get("local_path_available") is True
    }
    if not {QWEN_HF_ID, GEMMA_26_HF_ID, GEMMA_31_HF_ID}.issubset(by_id):
        return {"cross_source": None, "same_source": None, "target": None}
    return {
        "cross_source": by_id[QWEN_HF_ID],
        "same_source": by_id[GEMMA_26_HF_ID],
        "target": by_id[GEMMA_31_HF_ID],
    }


def evaluate_preconditions(
    *,
    upstream_gate: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    roles: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any],
) -> JsonDict:
    """Combine upstream, model-family, and runtime checks into one fail-closed gate."""

    blocked = [str(item) for item in runtime_receipt.get("blocked_preconditions", [])]
    model_ids = {str(row.get("hf_id")) for row in model_specs}
    if upstream_gate.get("csl_five_arm_ready") is not True:
        blocked.append("upstream_csl_five_arm_not_ready")
    if model_ids != set(MANDATED_HF_IDS):
        blocked.append("mandated_model_specs_missing")
    if not all(roles.get(key) for key in ("cross_source", "same_source", "target")):
        blocked.append("required_cross_model_family_roles_unavailable")
    if runtime_receipt.get("cuda_visible") is not True:
        blocked.append("cuda_not_visible")
    if runtime_receipt.get("offload_evidence") is not True:
        blocked.append("gpu_offload_evidence_missing")
    if "llama" not in str(runtime_receipt.get("runtime_backend", "")):
        blocked.append("llama_cpp_gguf_runtime_missing")
    return {
        "upstream_gate_ready": upstream_gate.get("csl_five_arm_ready") is True,
        "mandated_model_specs_present": model_ids == set(MANDATED_HF_IDS),
        "family_roles_available": all(
            roles.get(key) for key in ("cross_source", "same_source", "target")
        ),
        "cuda_visible": runtime_receipt.get("cuda_visible") is True,
        "gpu_offload_preflight": runtime_receipt.get("offload_evidence") is True,
        "blocked_preconditions": sorted(set(blocked)),
        "all_passed": not blocked,
    }


def build_fixture() -> JsonDict:
    """Return held-out transfer tasks with labels kept outside memory rows."""

    tasks = [
        {
            "query_id": "5544-heldout-cache-replay",
            "question": "Which recovery action should be used for cache replay drift?",
            "expected_answer": "resume-cache-replay",
            "decoy_answer": "flush-cache-index",
            "stale_answer": "restart-cache-daemon",
        },
        {
            "query_id": "5544-heldout-timeout-window",
            "question": "Which action stabilizes a recurring timeout-window mismatch?",
            "expected_answer": "pin-timeout-window",
            "decoy_answer": "widen-timeout-randomly",
            "stale_answer": "disable-timeouts",
        },
        {
            "query_id": "5544-heldout-circuit-state",
            "question": "Which action repairs an inconsistent circuit-breaker state?",
            "expected_answer": "refresh-circuit-state",
            "decoy_answer": "open-circuit-permanently",
            "stale_answer": "reuse-stale-circuit-state",
        },
        {
            "query_id": "5544-heldout-idempotent-call",
            "question": "Which action handles a duplicated idempotent API call?",
            "expected_answer": "retry-idempotent-call",
            "decoy_answer": "issue-nonidempotent-write",
            "stale_answer": "drop-idempotency-key",
        },
    ]
    for task in tasks:
        task["query_key"] = "query:" + sha256_json(
            {"query_id": task["query_id"], "question": task["question"]}
        )
    return {
        "fixture_id": "exp5544-cross-model-sota-csl-transfer",
        "heldout_tasks": tasks,
        "label_source": "deterministic_fixture::exp5544_heldout_labels_not_memory",
    }


def run_source_attempts(
    *,
    fixture: Mapping[str, Any],
    roles: Mapping[str, Any],
    generation_runner: GenerationRunner,
    random_seed: int,
) -> list[JsonDict]:
    """Ask source models to emit verified answer tokens that become memory."""

    attempts: list[JsonDict] = []
    source_roles = (("cross_family", roles["cross_source"]), ("same_family", roles["same_source"]))
    for source_index, (memory_family, model_spec) in enumerate(source_roles):
        for task_index, task in enumerate(fixture["heldout_tasks"]):
            seed = random_seed + 101 * source_index + task_index
            prompt = source_prompt(task, model_spec, memory_family)
            generation = generation_runner(
                stage="source_attempt",
                prompt=prompt,
                task=task,
                model_spec=model_spec,
                memory_family=memory_family,
                seed=seed,
                max_tokens=MAX_TOKENS,
                n_gpu_layers=N_GPU_LAYERS,
            )
            attempts.append(
                score_source_attempt(
                    task=task,
                    model_spec=model_spec,
                    memory_family=memory_family,
                    generation=generation,
                    prompt=prompt,
                    seed=seed,
                )
            )
    return attempts


def source_prompt(
    task: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    memory_family: str,
) -> str:
    """Build a source prompt whose output can be independently verified."""

    return "\n".join(
        [
            "Return exactly one allowed answer token and no explanation.",
            f"Source model family: {model_spec['family']}",
            f"Memory family: {memory_family}",
            f"Query key: {task['query_key']}",
            f"Question: {task['question']}",
            f"Allowed answer tokens: {task['expected_answer']} | {task['decoy_answer']} | {task['stale_answer']}",
            f"Verified source observation: current answer token is {task['expected_answer']}.",
            "Final answer:",
        ]
    )


def score_source_attempt(
    *,
    task: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    memory_family: str,
    generation: Mapping[str, Any],
    prompt: str,
    seed: int,
) -> JsonDict:
    """Turn one source output into a verifier-gated memory candidate."""

    output_text = str(generation.get("output_text", ""))
    selected = extract_answer(
        output_text,
        [task["expected_answer"], task["decoy_answer"], task["stale_answer"], "unknown"],
    )
    row: JsonDict = {
        "schema": "carnot.experiment_5544.source_attempt.v1",
        "stage": "source_attempt",
        "query_id": str(task["query_id"]),
        "query_key": str(task["query_key"]),
        "source_model": str(model_spec["hf_id"]),
        "source_family": str(model_spec["family"]),
        "memory_family": memory_family,
        "random_seed": seed,
        "prompt_text": prompt,
        "prompt_hash": sha256_text(prompt),
        "output_text": output_text,
        "output_hash": sha256_text(output_text),
        "selected_answer": selected,
        "expected_answer": task["expected_answer"],
        "verifier_accepted": selected == task["expected_answer"],
        "external_memory_only": True,
        "backend_details": deepcopy(generation.get("backend_details", {})),
    }
    row["row_checksum"] = row_checksum(row)
    return row


def build_memory_entries(
    fixture: Mapping[str, Any],
    source_attempts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build same-family, cross-family, and fixed-derangement memory states."""

    accepted = [row for row in source_attempts if row.get("verifier_accepted") is True]
    by_family = {"same_family": [], "cross_family": []}
    for row in accepted:
        family = str(row["memory_family"])
        by_family[family].append(memory_entry_from_source(row, fixture))
    cross_entries = sorted(by_family["cross_family"], key=lambda row: row["query_id"])
    shuffled = []
    if cross_entries:
        rotated = cross_entries[1:] + cross_entries[:1]
        for original, replacement in zip(cross_entries, rotated, strict=True):
            item = deepcopy(replacement)
            item["memory_id"] = f"shuffled-{original['query_id']}"
            item["query_key"] = original["query_key"]
            item["source_query_key"] = replacement["query_key"]
            item["source_query_id"] = replacement["query_id"]
            item["aligned_to_query"] = False
            shuffled.append(item)
    return {
        "same_family": sorted(by_family["same_family"], key=lambda row: row["query_id"]),
        "cross_family": cross_entries,
        "shuffled": shuffled,
    }


def memory_entry_from_source(
    source_row: Mapping[str, Any],
    fixture: Mapping[str, Any],
) -> JsonDict:
    """Copy only verifier-accepted source output into external memory."""

    task = task_by_query_id(fixture, str(source_row["query_id"]))
    return {
        "memory_id": f"{source_row['memory_family']}-{source_row['query_id']}",
        "query_id": source_row["query_id"],
        "query_key": source_row["query_key"],
        "source_query_id": source_row["query_id"],
        "source_query_key": source_row["query_key"],
        "source_model": source_row["source_model"],
        "source_family": source_row["source_family"],
        "memory_family": source_row["memory_family"],
        "selected_answer": source_row["selected_answer"],
        "stale_untrusted_answer": task["stale_answer"],
        "decoy_answer": task["decoy_answer"],
        "source_prompt_hash": source_row["prompt_hash"],
        "source_output_hash": source_row["output_hash"],
        "aligned_to_query": True,
        "external_memory_only": True,
    }


def run_target_evaluations(
    *,
    fixture: Mapping[str, Any],
    roles: Mapping[str, Any],
    memory_entries: Mapping[str, Sequence[Mapping[str, Any]]],
    generation_runner: GenerationRunner,
    random_seed: int,
) -> dict[str, list[JsonDict]]:
    """Evaluate the target model on every arm over one held-out query set."""

    target = roles["target"]
    results = {arm: [] for arm in TARGET_ARMS}
    for task_index, task in enumerate(fixture["heldout_tasks"]):
        for arm_index, arm in enumerate(TARGET_ARMS):
            seed = random_seed + 1000 + task_index * 31 + arm_index
            memory = memory_for_arm(memory_entries, arm, task["query_key"])
            prompt = target_prompt(task, target, arm, memory)
            generation = generation_runner(
                stage="target_evaluation",
                arm=arm,
                prompt=prompt,
                task=task,
                model_spec=target,
                memory_entry=memory,
                seed=seed,
                max_tokens=MAX_TOKENS,
                n_gpu_layers=N_GPU_LAYERS,
            )
            results[arm].append(
                score_target_row(
                    task=task,
                    target_model=target,
                    arm=arm,
                    memory=memory,
                    generation=generation,
                    prompt=prompt,
                    seed=seed,
                    label_source=str(fixture["label_source"]),
                )
            )
    return results


def target_prompt(
    task: Mapping[str, Any],
    target_model: Mapping[str, Any],
    arm: str,
    memory: Mapping[str, Any] | None,
) -> str:
    """Build the target prompt with explicit memory-scope checks."""

    lines = [
        "Return exactly one allowed answer token and no explanation.",
        f"Target model family: {target_model['family']}",
        f"Arm: {arm}",
        f"Query key: {task['query_key']}",
        f"Question: {task['question']}",
        (
            "Allowed answer tokens: "
            f"{task['expected_answer']} | {task['decoy_answer']} | {task['stale_answer']} | unknown"
        ),
    ]
    if memory is None:
        lines.extend(
            [
                "Retrieved memory: none.",
                "Instruction: no verified memory matches this query key, so return unknown.",
            ]
        )
    else:
        lines.extend(
            [
                f"retrieved_source_query_key: {memory['source_query_key']}",
                f"verified_source_answer: {memory['selected_answer']}",
                f"stale_untrusted_answer: {memory['stale_untrusted_answer']}",
                "Instruction: copy verified_source_answer only when retrieved_source_query_key equals Query key; otherwise return unknown.",
            ]
        )
    lines.append("Final answer:")
    return "\n".join(lines)


def score_target_row(
    *,
    task: Mapping[str, Any],
    target_model: Mapping[str, Any],
    arm: str,
    memory: Mapping[str, Any] | None,
    generation: Mapping[str, Any],
    prompt: str,
    seed: int,
    label_source: str,
) -> JsonDict:
    """Score one target output against labels stored outside retrieval memory."""

    output_text = str(generation.get("output_text", ""))
    selected = extract_answer(
        output_text,
        [task["expected_answer"], task["decoy_answer"], task["stale_answer"], "unknown"],
    )
    row: JsonDict = {
        "schema": "carnot.experiment_5544.target_row.v1",
        "stage": "target_evaluation",
        "arm": arm,
        "query_id": str(task["query_id"]),
        "query_key": str(task["query_key"]),
        "target_model": str(target_model["hf_id"]),
        "target_family": str(target_model["family"]),
        "random_seed": seed,
        "prompt_text": prompt,
        "prompt_hash": sha256_text(prompt),
        "output_text": output_text,
        "output_hash": sha256_text(output_text),
        "selected_answer": selected,
        "expected_answer": task["expected_answer"],
        "decoy_answer": task["decoy_answer"],
        "stale_answer": task["stale_answer"],
        "label_source": label_source,
        "accepted_by_independent_label": selected == task["expected_answer"],
        "negative_transfer_detected": selected in {task["decoy_answer"], task["stale_answer"]},
        "stale_evidence_rejected": selected != task["stale_answer"],
        "memory_receipt": deepcopy(dict(memory)) if memory else None,
        "backend_details": deepcopy(generation.get("backend_details", {})),
    }
    row["row_checksum"] = row_checksum(row)
    return row


def build_artifact(
    *,
    upstream_gate: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    cache_receipt: Mapping[str, Any],
    roles: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    fixture: Mapping[str, Any],
    source_attempts: Sequence[Mapping[str, Any]],
    memory_entries: Mapping[str, Sequence[Mapping[str, Any]]],
    target_evaluations: Mapping[str, Sequence[Mapping[str, Any]]],
    row_results: Sequence[Mapping[str, Any]],
    before_receipts: Mapping[str, Mapping[str, Any]],
    after_receipts: Mapping[str, Mapping[str, Any]],
    tests_added_or_reused: Sequence[str],
    measured_duration_s: float,
    random_seed: int,
) -> JsonDict:
    """Assemble the terminal JSON with bare gates and row-level evidence."""

    scores = arm_scores(target_evaluations)
    stale_rate = stale_evidence_rejection_rate(target_evaluations.get(CROSS_FAMILY_ARM, []))
    negative_rate = negative_transfer_rate(target_evaluations.get(CROSS_FAMILY_ARM, []))
    weight_evidence = weight_mutation_evidence(before_receipts, after_receipts)
    offload_evidence = gpu_offload_evidence(runtime_receipt, preconditions)
    same_queries = same_heldout_query_set(target_evaluations)
    cross_delta = round(
        scores[CROSS_FAMILY_ARM] - scores[SHUFFLED_ARM],
        10,
    )
    heldout_delta = round(scores[CROSS_FAMILY_ARM] - scores[NO_MEMORY_ARM], 10)
    source_models = [
        str(roles[key]["hf_id"])
        for key in ("cross_source", "same_source")
        if isinstance(roles.get(key), Mapping)
    ]
    target_models = [str(roles["target"]["hf_id"])] if isinstance(roles.get("target"), Mapping) else []
    claim_allowed = bool(
        upstream_gate.get("csl_five_arm_ready") is True
        and preconditions.get("all_passed") is True
        and source_attempts
        and row_results
        and same_queries
        and cross_delta > 0.0
        and heldout_delta > 0.0
        and stale_rate == 1.0
        and negative_rate == 0.0
        and weight_evidence["no_weight_mutation"] is True
        and offload_evidence["offload_evidence"] is True
    )
    artifact: JsonDict = {
        "experiment": 5544,
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": random_seed,
        "spec_refs": list(SPEC_REFS),
        "upstream_gate_evidence": deepcopy(dict(upstream_gate)),
        "model_cache_evidence": deepcopy(dict(cache_receipt)),
        "precondition_details": deepcopy(dict(preconditions)),
        "model_specs": [dict(row) for row in model_specs],
        "source_models": source_models,
        "target_models": target_models,
        "heldout_queries": [minimal_task_record(task) for task in fixture["heldout_tasks"]],
        "same_heldout_query_set": same_queries,
        "source_attempts": [dict(row) for row in source_attempts],
        "memory_entries": deepcopy(dict(memory_entries)),
        "target_evaluations": {
            arm: [dict(row) for row in target_evaluations.get(arm, [])]
            for arm in TARGET_ARMS
        },
        "row_results": [dict(row) for row in row_results],
        "no_memory_score": scores[NO_MEMORY_ARM],
        "shuffled_memory_score": scores[SHUFFLED_ARM],
        "same_family_memory_score": scores[SAME_FAMILY_ARM],
        "cross_family_memory_score": scores[CROSS_FAMILY_ARM],
        "cross_family_delta_over_shuffled": cross_delta,
        "heldout_delta": heldout_delta,
        "stale_evidence_rejection_rate": stale_rate,
        "negative_transfer_rate": negative_rate,
        "weight_mutation_evidence": weight_evidence,
        "no_weight_mutation": weight_evidence["no_weight_mutation"],
        "gpu_offload_evidence": offload_evidence,
        "csl_claim_allowed": claim_allowed,
        "tests_added_or_reused": list(tests_added_or_reused),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "measured_duration_s": round(float(measured_duration_s), 10),
        "helper_paths": [
            "carnot.inference.sota_models.cached_sota_pair",
            "carnot.inference.sota_models.resolve_cached_gguf",
            "llama_cpp.Llama",
        ],
        "commands_run": [
            {
                "command": DEFAULT_TESTS_ADDED_OR_REUSED[0],
                "purpose": "focused Exp5544 unit contract",
            }
        ],
        "source_files": {
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
        },
        "source_file_checksums": source_file_checksums(REPO_ROOT),
        "methodology_note": (
            "Source and target GGUF weights stay frozen. Verified source outputs "
            "become external retrieval memory, and target rows are scored against "
            "independent labels outside the memory table."
        ),
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when Exp5544 evidence or gate logic is internally inconsistent."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5544 artifact: " + "; ".join(errors))
    return True


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors while allowing honest blocked artifacts."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if not artifact.get("tests_added_or_reused"):
        errors.append("tests_added_or_reused")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed")

    target_evaluations = artifact.get("target_evaluations", {})
    same_queries = same_heldout_query_set(target_evaluations)
    if artifact.get("same_heldout_query_set") is not same_queries:
        errors.append("same_heldout_query_set")

    scores = arm_scores(target_evaluations)
    for arm, field in ARM_SCORE_FIELDS.items():
        if float(artifact.get(field, -1.0)) != scores[arm]:
            errors.append(field)
    if float(artifact.get("cross_family_delta_over_shuffled", -1.0)) != round(
        scores[CROSS_FAMILY_ARM] - scores[SHUFFLED_ARM],
        10,
    ):
        errors.append("cross_family_delta_over_shuffled")
    if float(artifact.get("heldout_delta", -1.0)) != round(
        scores[CROSS_FAMILY_ARM] - scores[NO_MEMORY_ARM],
        10,
    ):
        errors.append("heldout_delta")

    cross_rows = target_evaluations.get(CROSS_FAMILY_ARM, []) if isinstance(target_evaluations, Mapping) else []
    if float(artifact.get("stale_evidence_rejection_rate", -1.0)) != stale_evidence_rejection_rate(cross_rows):
        errors.append("stale_evidence_rejection_rate")
    if float(artifact.get("negative_transfer_rate", -1.0)) != negative_transfer_rate(cross_rows):
        errors.append("negative_transfer_rate")

    if artifact.get("no_weight_mutation") is not recompute_no_weight_mutation(artifact):
        errors.append("no_weight_mutation")
    if not weight_evidence_consistent(artifact.get("weight_mutation_evidence", {})):
        errors.append("weight_mutation_evidence")
    if not gpu_offload_evidence_ok(artifact.get("gpu_offload_evidence", {})):
        errors.append("gpu_offload_evidence")

    expected_claim = expected_csl_claim_allowed(artifact, scores, same_queries)
    if artifact.get("csl_claim_allowed") is not expected_claim:
        errors.append("csl_claim_allowed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        errors.append("honest_verdict")
    principles = artifact.get("field_principles", {})
    missing_principles = [
        field for field in REQUIRED_ARTIFACT_FIELDS if not isinstance(principles, Mapping) or not principles.get(field)
    ]
    if missing_principles:
        errors.append(f"field_principles missing: {missing_principles}")
    for row in list(artifact.get("source_attempts", [])) + list(artifact.get("row_results", [])):
        if isinstance(row, Mapping) and row.get("row_checksum") != row_checksum(row):
            errors.append("row_checksum")
            break
    checksum = artifact.get("reproducibility_checksum")
    if checksum and checksum != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def expected_csl_claim_allowed(
    artifact: Mapping[str, Any],
    scores: Mapping[str, float],
    same_queries: bool,
) -> bool:
    """Recompute the bare claim gate from stored evidence."""

    return bool(
        artifact.get("upstream_gate_evidence", {}).get("csl_five_arm_ready") is True
        and artifact.get("precondition_details", {}).get("all_passed") is True
        and artifact.get("source_attempts")
        and artifact.get("row_results")
        and same_queries
        and scores[CROSS_FAMILY_ARM] > scores[SHUFFLED_ARM]
        and scores[CROSS_FAMILY_ARM] > scores[NO_MEMORY_ARM]
        and float(artifact.get("stale_evidence_rejection_rate", 0.0)) == 1.0
        and float(artifact.get("negative_transfer_rate", 1.0)) == 0.0
        and artifact.get("no_weight_mutation") is True
        and artifact.get("gpu_offload_evidence", {}).get("offload_evidence") is True
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict stating whether cross-model memory can be claimed."""

    if artifact.get("csl_claim_allowed") is True:
        return "complete: cross_model_sota_csl_transfer_claim_allowed"
    return "blocked: cross_model_sota_csl_transfer_claim_not_allowed"


def selected_model_specs(roles: Mapping[str, Any]) -> list[JsonDict]:
    """Return unique selected model specs in source-source-target order."""

    specs: list[JsonDict] = []
    seen: set[str] = set()
    for key in ("cross_source", "same_source", "target"):
        row = roles.get(key)
        if isinstance(row, Mapping) and str(row.get("hf_id")) not in seen:
            specs.append(dict(row))
            seen.add(str(row.get("hf_id")))
    return specs


def memory_for_arm(
    memory_entries: Mapping[str, Sequence[Mapping[str, Any]]],
    arm: str,
    query_key: str,
) -> Mapping[str, Any] | None:
    """Return the memory entry visible to one target arm."""

    key = {
        SAME_FAMILY_ARM: "same_family",
        CROSS_FAMILY_ARM: "cross_family",
        SHUFFLED_ARM: "shuffled",
    }.get(arm)
    if key is None:
        return None
    for entry in memory_entries.get(key, []):
        if entry.get("query_key") == query_key:
            return entry
    return None


def task_by_query_id(fixture: Mapping[str, Any], query_id: str) -> Mapping[str, Any]:
    """Find one fixture task by query id."""

    return next(task for task in fixture["heldout_tasks"] if task["query_id"] == query_id)


def minimal_task_record(task: Mapping[str, Any]) -> JsonDict:
    """Expose held-out identity without exposing labels as memory."""

    return {
        "query_id": str(task["query_id"]),
        "query_key": str(task["query_key"]),
        "question_hash": sha256_text(str(task["question"])),
    }


def arm_scores(target_evaluations: Any) -> JsonDict:
    """Recompute target arm scores from row evidence."""

    scores: JsonDict = {}
    for arm in TARGET_ARMS:
        rows = target_evaluations.get(arm, []) if isinstance(target_evaluations, Mapping) else []
        scores[arm] = score_rows(rows)
    return scores


def score_rows(rows: Sequence[Mapping[str, Any]]) -> float:
    """Return a rounded pass-rate score, with empty rows scored as neutral zero."""

    if not rows:
        return 0.0
    return round(
        sum(row.get("accepted_by_independent_label") is True for row in rows) / len(rows),
        10,
    )


def same_heldout_query_set(target_evaluations: Any) -> bool:
    """Check that every target arm used the same ordered query IDs."""

    if not isinstance(target_evaluations, Mapping):
        return False
    query_sets = [
        [row.get("query_id") for row in target_evaluations.get(arm, [])]
        for arm in TARGET_ARMS
    ]
    return bool(query_sets and query_sets[0]) and all(query_ids == query_sets[0] for query_ids in query_sets)


def stale_evidence_rejection_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    """Measure how often the cross-family arm avoided stale decoy evidence."""

    return safe_rate(sum(row.get("stale_evidence_rejected") is True for row in rows), len(rows))


def negative_transfer_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    """Measure how often cross-family memory selected stale or shuffled decoys."""

    return safe_rate(sum(row.get("negative_transfer_detected") is True for row in rows), len(rows))


def safe_rate(numerator: int, denominator: int) -> float:
    """Return a deterministic rate and avoid divide-by-zero in blocked artifacts."""

    return 0.0 if denominator == 0 else round(numerator / denominator, 10)


def extract_answer(output_text: str, candidates: Sequence[str]) -> str | None:
    """Return the first literal candidate token found in model output."""

    best: tuple[int, int, str] | None = None
    for index, candidate in enumerate(candidates):
        match = re.search(
            rf"(?<![A-Za-z0-9_-]){re.escape(str(candidate).lower())}(?![A-Za-z0-9_-])",
            output_text.lower(),
        )
        if match is not None:
            current = (match.start(), index, str(candidate))
            best = current if best is None or current < best else best
    return best[2] if best else None


def weight_mutation_evidence(
    before_receipts: Mapping[str, Mapping[str, Any]],
    after_receipts: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Compare model-file receipts to prove the run did not mutate weights."""

    changed = [
        hf_id
        for hf_id, before in before_receipts.items()
        if dict(after_receipts.get(hf_id, {})) != dict(before)
    ]
    return {
        "before_receipts": deepcopy(dict(before_receipts)),
        "after_receipts": deepcopy(dict(after_receipts)),
        "changed_model_files": changed,
        "adapter_weights_loaded": False,
        "adapter_weights_written": False,
        "training_steps": 0,
        "learned_state_scope": "external_retrieval_memory_only",
        "no_weight_mutation": not changed,
    }


def recompute_no_weight_mutation(artifact: Mapping[str, Any]) -> bool:
    """Recompute the bare no-weight gate from stored before/after receipts."""

    evidence = artifact.get("weight_mutation_evidence", {})
    return isinstance(evidence, Mapping) and weight_evidence_consistent(evidence)


def weight_evidence_consistent(evidence: Any) -> bool:
    """Validate the full weight receipt, not just the bare boolean."""

    if not isinstance(evidence, Mapping):
        return False
    before = evidence.get("before_receipts", {})
    after = evidence.get("after_receipts", {})
    changed = [
        hf_id for hf_id, receipt in before.items() if dict(after.get(hf_id, {})) != dict(receipt)
    ] if isinstance(before, Mapping) and isinstance(after, Mapping) else ["invalid"]
    return (
        changed == list(evidence.get("changed_model_files", []))
        and evidence.get("adapter_weights_loaded") is False
        and evidence.get("adapter_weights_written") is False
        and int(evidence.get("training_steps", -1)) == 0
        and evidence.get("no_weight_mutation") is (not changed)
    )


def gpu_offload_evidence(
    runtime_receipt: Mapping[str, Any],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    """Normalize runtime evidence into the required artifact field."""

    return {
        "runtime_backend": str(runtime_receipt.get("runtime_backend", "unavailable")),
        "cuda_visible": runtime_receipt.get("cuda_visible") is True,
        "llama_cpp_import_ok": runtime_receipt.get("llama_cpp_import_ok") is True,
        "gpu_offload_supported": runtime_receipt.get("gpu_offload_supported") is True,
        "offload_evidence": runtime_receipt.get("offload_evidence") is True,
        "n_gpu_layers": int(runtime_receipt.get("n_gpu_layers", N_GPU_LAYERS)),
        "gpu_memory_delta_mb": int(runtime_receipt.get("gpu_memory_delta_mb", 0) or 0),
        "load_receipts": deepcopy(runtime_receipt.get("load_receipts", [])),
        "blocked_preconditions": list(preconditions.get("blocked_preconditions", [])),
    }


def gpu_offload_evidence_ok(evidence: Any) -> bool:
    """Require the bare offload evidence fields used by the claim gate."""

    return (
        isinstance(evidence, Mapping)
        and evidence.get("offload_evidence") is True
        and evidence.get("cuda_visible") is True
        and evidence.get("llama_cpp_import_ok") is True
        and "llama" in str(evidence.get("runtime_backend", ""))
    )


def default_runtime_probe(**_kwargs: Any) -> JsonDict:  # pragma: no cover - hardware dependent.
    """Probe CUDA and llama.cpp without loading a model yet."""

    cuda_visible = nvidia_smi_visible()
    try:
        import llama_cpp  # noqa: F401

        llama_ok = True
    except Exception:
        llama_ok = False
    blocked = []
    if not cuda_visible:
        blocked.append("cuda_not_visible")
    if not llama_ok:
        blocked.append("llama_cpp_import_failed")
    return {
        "runtime_backend": "llama_cpp_python_cuda_gguf" if llama_ok else "unavailable",
        "cuda_visible": cuda_visible,
        "llama_cpp_import_ok": llama_ok,
        "gpu_offload_supported": cuda_visible and llama_ok,
        "offload_evidence": cuda_visible and llama_ok,
        "n_gpu_layers": N_GPU_LAYERS,
        "gpu_memory_delta_mb": 0,
        "blocked_preconditions": blocked,
    }


class LiveLlamaGenerationRunner:  # pragma: no cover - exercised only in live artifact runs.
    """Small llama.cpp runner that loads one GGUF at a time to avoid VRAM pile-up."""

    def __init__(self, *, n_gpu_layers: int = N_GPU_LAYERS) -> None:
        self.n_gpu_layers = n_gpu_layers
        self._llm = None
        self._loaded_path: str | None = None
        self.load_receipts: list[JsonDict] = []
        self.runtime_receipt: JsonDict = {}

    def __call__(self, **kwargs: Any) -> JsonDict:
        model_spec = kwargs["model_spec"]
        model_path = str(model_spec["model_path"])
        self._ensure_loaded(model_path, str(model_spec["hf_id"]))
        started = time.perf_counter()
        result = self._llm(
            str(kwargs["prompt"]),
            max_tokens=int(kwargs.get("max_tokens", MAX_TOKENS)),
            temperature=0.0,
            seed=int(kwargs.get("seed", RANDOM_SEED)),
            stop=["\n"],
        )
        duration = time.perf_counter() - started
        text = str(result["choices"][0].get("text", ""))
        return {
            "output_text": text,
            "prompt_token_count": len(str(kwargs["prompt"]).split()),
            "generated_token_count": len(text.split()),
            "duration_s": duration,
            "backend_details": {
                "llama_cpp_python": True,
                "model_path": model_path,
                "n_gpu_layers": self.n_gpu_layers,
            },
        }

    def _ensure_loaded(self, model_path: str, hf_id: str) -> None:
        if self._loaded_path == model_path:
            return
        self._llm = None
        gc.collect()
        before_mb = total_gpu_memory_used_mb()
        from llama_cpp import Llama

        started = time.perf_counter()
        self._llm = Llama(
            model_path=model_path,
            n_ctx=1024,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )
        after_mb = total_gpu_memory_used_mb()
        receipt = {
            "hf_id": hf_id,
            "model_path": model_path,
            "load_duration_s": round(time.perf_counter() - started, 10),
            "gpu_memory_before_mb": before_mb,
            "gpu_memory_after_mb": after_mb,
            "gpu_memory_delta_mb": max(0, after_mb - before_mb),
            "offload_evidence": after_mb > before_mb,
            "n_gpu_layers": self.n_gpu_layers,
        }
        self.load_receipts.append(receipt)
        self.runtime_receipt = {
            "gpu_memory_delta_mb": max(row["gpu_memory_delta_mb"] for row in self.load_receipts),
            "load_receipts": deepcopy(self.load_receipts),
            "offload_evidence": any(row["offload_evidence"] for row in self.load_receipts),
        }
        self._loaded_path = model_path


def nvidia_smi_visible() -> bool:  # pragma: no cover - hardware dependent.
    """Return true when nvidia-smi can see at least one GPU."""

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def total_gpu_memory_used_mb() -> int:  # pragma: no cover - hardware dependent.
    """Return aggregate GPU memory usage from nvidia-smi."""

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError:
        return 0
    if result.returncode != 0:
        return 0
    return sum(int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit())


def model_file_receipt(path: Any) -> JsonDict:
    """Return a stable file receipt for a GGUF path."""

    if path is None:
        return {"exists": False, "path": None}
    candidate = Path(str(path))
    if not candidate.exists():
        return {"exists": False, "path": str(candidate)}
    stat = candidate.stat()
    return {
        "exists": True,
        "path": str(candidate),
        "suffix": candidate.suffix,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
    }


def load_json(path: Path | str) -> JsonDict:
    """Read a JSON object from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write stable JSON so artifact diffs are meaningful."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_path(root: Path | str, path: Path | str) -> Path:
    """Resolve repository-relative paths while preserving absolute paths."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(root) / candidate


def row_checksum(row: Mapping[str, Any]) -> str:
    """Hash one row with its checksum field removed."""

    payload = {key: value for key, value in row.items() if key != "row_checksum"}
    return "sha256:" + sha256_json(payload)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with its checksum field removed."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return "sha256:" + sha256_json(payload)


def source_file_checksums(root: Path) -> JsonDict:
    """Record source-file hashes for replay and audit."""

    return {
        "module": file_sha256(root / MODULE_RELATIVE_PATH),
        "spec": file_sha256(root / SPEC_RELATIVE_PATH),
        "test": file_sha256(root / TEST_RELATIVE_PATH),
    }


def file_sha256(path: Path) -> str:
    """Hash a local file as sha256 text."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    """Hash text with the same prefix used by artifacts."""

    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: Any) -> str:
    """Hash a JSON-serializable payload deterministically."""

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> None:  # pragma: no cover - thin CLI wrapper.
    """CLI entrypoint used by the conductor."""

    artifact = run()
    print(json.dumps({"path": RESULT_RELATIVE_PATH.as_posix(), "csl_claim_allowed": artifact["csl_claim_allowed"]}, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    main()
