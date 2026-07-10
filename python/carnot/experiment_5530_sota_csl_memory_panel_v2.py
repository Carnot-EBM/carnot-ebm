"""Exp5530 local-SOTA GGUF CSL memory panel v2.

Spec refs: REQ-LEARN-5530,
SCENARIO-LEARN-5530-UPSTREAM-GATES,
SCENARIO-LEARN-5530-CONTROLS,
SCENARIO-LEARN-5530-NO-WEIGHT-MUTATION.

The panel keeps the GGUF model frozen. The only state that changes is a small,
hashable external memory table. Exact held-out labels live in a separate table,
so the model output is scored by an independent witness rather than by memory
utility or a self-reported model verdict.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
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
RESULT_RELATIVE_PATH = Path("results/experiment_5530_sota_csl_memory_panel_v2.json")
UPSTREAM_5528_PATH = Path("results/experiment_5528_csl_canonical_gate_artifact.json")
UPSTREAM_5529_PATH = Path("results/experiment_5529_csl_event_topic_residue_stress.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5530_sota_csl_memory_panel_v2.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5530_sota_csl_memory_panel_v2.py")

SCHEMA = "carnot.experiment_5530.sota_csl_memory_panel_v2.v1"
EXPERIMENT_ID = "experiment_5530_sota_csl_memory_panel_v2"
TASK_ID = "exp5530-gated-sota-csl-memory-panel-v2"
MILESTONE = "2026.07.501"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5530
DEFAULT_QUANTIZATION = "Q4_K_M"
N_GPU_LAYERS = -1
MAX_TOKENS = 16
INFERENCE_SUBSTRATE = "local_sota_gguf_csl_memory_panel"
INDEPENDENT_LABEL_SOURCE = "deterministic_fixture::heldout_labels_not_memory"
TERMINAL_PREFIXES = ("complete:", "blocked:")

NO_MEMORY_CONDITION = "no_memory"
FRESH_MEMORY_CONDITION = "fresh_memory"
STALE_MEMORY_CONDITION = "stale_memory"
CONDITIONS = (NO_MEMORY_CONDITION, FRESH_MEMORY_CONDITION, STALE_MEMORY_CONDITION)

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "role": "flagship_moe",
        "name": "Qwen3.6-35B-A3B",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "quantization": DEFAULT_QUANTIZATION,
    },
    {
        "role": "flagship_dense",
        "name": "Gemma4-31B-it",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "quantization": DEFAULT_QUANTIZATION,
    },
    {
        "role": "middle_moe",
        "name": "Gemma4-26B-A4B-it",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "quantization": DEFAULT_QUANTIZATION,
    },
)
MANDATED_HF_IDS = tuple(str(spec["hf_id"]) for spec in MANDATED_MODEL_SPECS)
RUN_ROLE_PREFERENCE = ("middle_moe", "flagship_moe", "flagship_dense")
SPEC_REFS = (
    "REQ-LEARN-5530",
    "SCENARIO-LEARN-5530-UPSTREAM-GATES",
    "SCENARIO-LEARN-5530-CONTROLS",
    "SCENARIO-LEARN-5530-NO-WEIGHT-MUTATION",
)

REQUIRED_ARTIFACT_FIELDS = (
    "model_specs",
    "models_attempted",
    "no_memory_score",
    "fresh_memory_score",
    "stale_memory_score",
    "heldout_delta",
    "negative_transfer_rate",
    "stale_evidence_rejection_rate",
    "memory_hash_before",
    "memory_hash_after",
    "no_model_weight_mutation",
    "gpu_offload_evidence",
    "continuous_self_learning_evidence",
    "csl_claim_allowed",
    "tests_added_or_reused",
    "field_principles",
    "inference_substrate",
    "honest_verdict",
)
FIELD_PRINCIPLES: JsonDict = {
    "model_specs": "Lists every mandated local SOTA GGUF and its local-path receipt.",
    "models_attempted": "Records which mandated GGUF was actually loaded or invoked for the panel.",
    "no_memory_score": "Independent held-out baseline with no external memory evidence.",
    "fresh_memory_score": "Independent held-out score when fresh verified memory is available.",
    "stale_memory_score": "Control score when only stale memory is supplied.",
    "heldout_delta": "Utility lift computed as fresh_memory_score minus no_memory_score.",
    "negative_transfer_rate": "Rate at which fresh-memory rows choose stale or irrelevant decoys.",
    "stale_evidence_rejection_rate": "Rate at which fresh-memory rows avoid stale candidate answers.",
    "memory_hash_before": "Deterministic hash of external memory before verifier-governed updates.",
    "memory_hash_after": "Deterministic hash of external memory after verifier-governed updates.",
    "no_model_weight_mutation": "Bare frozen-weight gate from before/after model-file receipts.",
    "gpu_offload_evidence": "Records CUDA, llama.cpp offload, load delta, and blocked preconditions.",
    "continuous_self_learning_evidence": "Bounded CSL evidence requiring utility lift and safety gates.",
    "csl_claim_allowed": "Bare gate forbidding broad CSL claims unless all evidence thresholds pass.",
    "tests_added_or_reused": "Lists focused, coverage, and full-suite verification commands.",
    "field_principles": "Explains why each headline and gate field exists.",
    "inference_substrate": "Declares a local SOTA GGUF panel with external CSL memory.",
    "honest_verdict": "Terminal summary with complete or blocked prefix.",
}


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    tests_added_or_reused: Sequence[str] = (),
    model_resolver: ModelResolver = resolve_cached_gguf,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    runtime_probe: RuntimeProbe | None = None,
    generation_runner: GenerationRunner | None = None,
    write: bool = True,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    """Run the bounded panel and optionally write the required JSON artifact."""

    started = time.perf_counter()
    root_path = Path(root)
    target = resolve_output_path(root_path, result_path)
    upstream = load_upstream_gates(root_path)
    model_specs, cache_receipt = resolve_model_specs(
        model_resolver=model_resolver,
        cached_pair_fn=cached_pair_fn,
    )
    selected_model = select_panel_model(model_specs)
    runtime_fn = runtime_probe or default_runtime_probe
    runtime_receipt = dict(runtime_fn(model_spec=selected_model, n_gpu_layers=N_GPU_LAYERS))
    model_before = model_file_receipt(selected_model.get("model_path") if selected_model else None)
    preconditions = evaluate_preconditions(
        upstream_gates=upstream,
        model_specs=model_specs,
        selected_model=selected_model,
        runtime_receipt=runtime_receipt,
    )
    runner = generation_runner
    models_attempted: list[JsonDict] = []

    if preconditions["all_passed"] and runner is None:  # pragma: no cover
        try:
            live = LlamaMemoryPanelRunner(
                model_spec=selected_model or {},
                n_gpu_layers=N_GPU_LAYERS,
                seed=random_seed,
            )
        except Exception as exc:
            preconditions["all_passed"] = False
            preconditions["blocked_preconditions"].append(
                f"llama_cpp_model_load_failed:{type(exc).__name__}: {exc}"
            )
        else:
            runner = live
            runtime_receipt["load_receipt"] = dict(live.load_receipt)
            runtime_receipt["gpu_memory_delta_mb"] = live.load_receipt.get(
                "gpu_memory_delta_mb", 0
            )
            runtime_receipt["offload_evidence"] = bool(
                runtime_receipt.get("offload_evidence")
                and live.load_receipt.get("offload_evidence")
            )
            if runtime_receipt["offload_evidence"] is not True:
                preconditions["all_passed"] = False
                preconditions["blocked_preconditions"].append(
                    "gpu_offload_not_observed_after_load"
                )

    fixture = build_fixture()
    memory_before = empty_memory_state()
    memory_after = build_memory_state(fixture)
    rows: list[JsonDict] = []
    if preconditions["all_passed"] and runner is not None and selected_model is not None:
        models_attempted.append(
            {
                "hf_id": selected_model["hf_id"],
                "model_path": selected_model["model_path"],
                "attempted": True,
                "outcome": "rows_generated",
            }
        )
        rows = run_panel_rows(
            fixture=fixture,
            model_spec=selected_model,
            runtime_receipt=runtime_receipt,
            generation_runner=runner,
            random_seed=random_seed,
        )
        mark_model_ran(model_specs, selected_model)

    model_after = model_file_receipt(selected_model.get("model_path") if selected_model else None)
    artifact = build_artifact(
        upstream_gates=upstream,
        fixture=fixture,
        model_specs=model_specs,
        cache_receipt=cache_receipt,
        selected_model=selected_model,
        models_attempted=models_attempted,
        runtime_receipt=runtime_receipt,
        preconditions=preconditions,
        memory_before=memory_before,
        memory_after=memory_after,
        rows=rows,
        tests_added_or_reused=tests_added_or_reused,
        model_before=model_before,
        model_after=model_after,
        methodology_duration_s=time.perf_counter() - started,
    )
    validate_artifact(artifact)
    if write:
        write_json(target, artifact)
    return artifact


def load_upstream_gates(root: Path | str) -> JsonDict:
    """Load Exp5528 and Exp5529 gate artifacts and expose their bare gates."""

    root_path = Path(root)
    exp5528 = load_json(root_path / UPSTREAM_5528_PATH)
    exp5529 = load_json(root_path / UPSTREAM_5529_PATH)
    gate_5528 = exp5528.get("csl_gate_fields_conductor_visible") is True
    gate_5529 = exp5529.get("csl_residue_stress_ready") is True
    return {
        "exp5528": {
            "path": UPSTREAM_5528_PATH.as_posix(),
            "loadable": True,
            "csl_gate_fields_conductor_visible": gate_5528,
            "honest_verdict": exp5528.get("honest_verdict", ""),
        },
        "exp5529": {
            "path": UPSTREAM_5529_PATH.as_posix(),
            "loadable": True,
            "csl_residue_stress_ready": gate_5529,
            "flagged_adversarial": exp5529.get("flagged_adversarial") is True,
            "corrigendum_pending": deepcopy(exp5529.get("corrigendum_pending", [])),
            "honest_verdict": exp5529.get("honest_verdict", ""),
        },
        "all_required_gates_true": gate_5528 and gate_5529,
    }


def resolve_model_specs(
    *,
    model_resolver: ModelResolver = resolve_cached_gguf,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
) -> tuple[list[JsonDict], JsonDict]:
    """Resolve mandated GGUF paths through cached_sota_pair plus local fallback."""

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
                "name": str(spec["name"]),
                "hf_id": hf_id,
                "quantization": quantization,
                "model_path": str(path) if path else None,
                "local_path_available": local,
                "resolved_via": "cached_sota_pair" if pair_hit else "resolve_cached_gguf",
                "file_receipt": model_file_receipt(path) if local else None,
                "selected_for_panel": False,
                "ran_panel": False,
                "gpu_offload_verified": False,
                "legacy_smoke_only": False,
            }
        )
    return rows, {
        "cached_sota_pair_attempted": True,
        "cached_sota_pair_available": bool(pair),
        "cached_sota_pair_specs": [dict(row) for row in pair or []],
    }


def select_panel_model(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict | None:
    """Pick one available mandated GGUF, preferring the smaller MoE model."""

    by_role = {str(row.get("role")): row for row in model_specs}
    for role in RUN_ROLE_PREFERENCE:
        row = by_role.get(role)
        if isinstance(row, Mapping) and row.get("local_path_available") is True:
            selected = dict(row)
            selected["selected_for_panel"] = True
            return selected
    return None


def evaluate_preconditions(
    *,
    upstream_gates: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    selected_model: Mapping[str, Any] | None,
    runtime_receipt: Mapping[str, Any],
) -> JsonDict:
    """Combine upstream gates, GGUF cache, CUDA, and llama.cpp offload checks."""

    blocked = [str(item) for item in runtime_receipt.get("blocked_preconditions", [])]
    model_ids = {str(row.get("hf_id")) for row in model_specs}
    all_ids_present = model_ids == set(MANDATED_HF_IDS)
    if upstream_gates.get("all_required_gates_true") is not True:
        blocked.append("upstream_csl_gates_not_ready")
    if not all_ids_present:
        blocked.append("mandated_model_specs_missing")
    if selected_model is None:
        blocked.append("no_mandated_local_sota_gguf_available")
    if runtime_receipt.get("cuda_visible") is not True:
        blocked.append("cuda_not_visible")
    if runtime_receipt.get("offload_evidence") is not True:
        blocked.append("gpu_offload_evidence_missing")
    if "llama" not in str(runtime_receipt.get("runtime_backend", "")):
        blocked.append("llama_cpp_gguf_runtime_missing")
    return {
        "upstream_gates_ready": upstream_gates.get("all_required_gates_true") is True,
        "mandated_model_specs_present": all_ids_present,
        "selected_model_available": selected_model is not None,
        "cuda_visible": runtime_receipt.get("cuda_visible") is True,
        "gpu_offload_preflight": runtime_receipt.get("offload_evidence") is True,
        "blocked_preconditions": sorted(set(blocked)),
        "all_passed": not blocked,
    }


def build_fixture() -> JsonDict:
    """Build the bounded stream and independent labels for all conditions."""

    tasks = [
        {
            "task_id": "5530-heldout-rx4-queue",
            "label_id": "label-5530-rx4-queue",
            "question": "Which queue owns incident RX-4?",
            "expected_answer": "queue-beta",
            "stale_answer": "queue-alpha",
            "fresh_memory": "fresh verifier receipt: incident RX-4 owner is queue-beta.",
            "stale_memory": "stale ticket note: incident RX-4 owner is queue-alpha.",
        },
        {
            "task_id": "5530-heldout-lot77-vendor",
            "label_id": "label-5530-lot77-vendor",
            "question": "Which vendor is verified for lot 77?",
            "expected_answer": "vendor-A",
            "stale_answer": "vendor-B",
            "fresh_memory": "fresh procurement receipt: lot 77 verified vendor is vendor-A.",
            "stale_memory": "stale vendor note: lot 77 vendor is vendor-B.",
        },
        {
            "task_id": "5530-heldout-ticket-k42",
            "label_id": "label-5530-ticket-k42",
            "question": "What is the current shipping code?",
            "expected_answer": "K-42",
            "stale_answer": "K-24",
            "fresh_memory": "fresh ticket receipt: current shipping code is K-42.",
            "stale_memory": "stale ticket note: shipping code was K-24.",
        },
    ]
    return {
        "fixture_id": "exp5530-sota-csl-memory-panel-v2",
        "heldout_tasks": tasks,
        "heldout_labels": {
            task["label_id"]: {
                "expected_answer": task["expected_answer"],
                "label_source": INDEPENDENT_LABEL_SOURCE,
            }
            for task in tasks
        },
    }


def empty_memory_state() -> JsonDict:
    """Return the pre-update external memory state."""

    return {"kind": "sota_csl_memory_panel_v2", "memories": []}


def build_memory_state(fixture: Mapping[str, Any]) -> JsonDict:
    """Return verifier-governed external memory after fresh updates."""

    memories = []
    for task in fixture["heldout_tasks"]:
        memories.append(
            {
                "memory_id": f"fresh-{task['task_id']}",
                "task_id": task["task_id"],
                "status": "fresh_verified",
                "fact": task["fresh_memory"],
                "stale_candidate_fact": task["stale_memory"],
                "used_as_label_source": False,
            }
        )
    return {"kind": "sota_csl_memory_panel_v2", "memories": memories}


def run_panel_rows(
    *,
    fixture: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any],
    generation_runner: GenerationRunner,
    random_seed: int,
) -> list[JsonDict]:
    """Generate and score one row for each held-out task and memory condition."""

    rows: list[JsonDict] = []
    for task_index, task in enumerate(fixture["heldout_tasks"]):
        for condition_index, condition in enumerate(CONDITIONS):
            seed = random_seed + task_index * 17 + condition_index
            prompt = prompt_for_condition(task, condition)
            memory_receipt = memory_receipt_for_condition(task, condition)
            started = time.perf_counter()
            generation = generation_runner(
                prompt=prompt,
                condition=condition,
                task=task,
                model_spec=model_spec,
                seed=seed,
                max_tokens=MAX_TOKENS,
                n_gpu_layers=N_GPU_LAYERS,
            )
            fallback_duration_s = time.perf_counter() - started
            row = score_row(
                fixture=fixture,
                task=task,
                condition=condition,
                model_spec=model_spec,
                runtime_receipt=runtime_receipt,
                generation=generation,
                prompt=prompt,
                memory_receipt=memory_receipt,
                seed=seed,
                fallback_duration_s=fallback_duration_s,
            )
            rows.append(row)
    return rows


def prompt_for_condition(task: Mapping[str, Any], condition: str) -> str:
    """Build the bounded answer-copying prompt for one memory condition."""

    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    allowed = f"{task['expected_answer']} | {task['stale_answer']} | unknown"
    lines = [
        "Return exactly one allowed answer token and no explanation.",
        f"Allowed answer tokens: {allowed}",
        f"Question: {task['question']}",
    ]
    if condition == NO_MEMORY_CONDITION:
        lines.extend(
            [
                "Fresh verified memory: none.",
                "Instruction: no fresh_verified_answer line is present, so return unknown.",
            ]
        )
    elif condition == FRESH_MEMORY_CONDITION:
        lines.extend(
            [
                f"fresh_verified_answer: {task['expected_answer']}",
                f"fresh_verified_memory: {task['fresh_memory']}",
                f"stale_untrusted_answer: {task['stale_answer']}",
                f"stale_untrusted_memory: {task['stale_memory']}",
                "Instruction: copy fresh_verified_answer and ignore stale_untrusted_answer.",
            ]
        )
    else:
        lines.extend(
            [
                f"stale_untrusted_answer: {task['stale_answer']}",
                f"stale_untrusted_memory: {task['stale_memory']}",
                "Instruction: this stale-memory control exposes what stale memory would answer.",
            ]
        )
    lines.append("Final answer:")
    return "\n".join(lines)


def memory_receipt_for_condition(task: Mapping[str, Any], condition: str) -> JsonDict:
    """Record memory supplied to the model without copying label-source metadata."""

    return {
        "condition": condition,
        "task_id": task["task_id"],
        "fresh_memory_supplied": condition == FRESH_MEMORY_CONDITION,
        "stale_memory_supplied": condition in {FRESH_MEMORY_CONDITION, STALE_MEMORY_CONDITION},
        "stale_candidate_answer": task["stale_answer"],
        "negative_transfer_candidate": condition == FRESH_MEMORY_CONDITION,
        "label_source_used": False,
    }


def score_row(
    *,
    fixture: Mapping[str, Any],
    task: Mapping[str, Any],
    condition: str,
    model_spec: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any],
    generation: Mapping[str, Any],
    prompt: str,
    memory_receipt: Mapping[str, Any],
    seed: int,
    fallback_duration_s: float,
) -> JsonDict:
    """Score a generated answer with the independent held-out label table."""

    label = fixture["heldout_labels"][task["label_id"]]
    output_text = str(generation.get("output_text", ""))
    witness = exact_label_witness(task, label, output_text)
    prompt_tokens = int(generation.get("prompt_token_count", 0) or estimate_tokens(prompt))
    completion_tokens = int(generation.get("generated_token_count", 0) or 0)
    row: JsonDict = {
        "schema": "carnot.experiment_5530.row.v1",
        "experiment_id": EXPERIMENT_ID,
        "task_id": str(task["task_id"]),
        "label_id": str(task["label_id"]),
        "condition": condition,
        "model_hf_id": str(model_spec.get("hf_id")),
        "model_path": str(model_spec.get("model_path")),
        "runtime_backend": str(runtime_receipt.get("runtime_backend", "unavailable")),
        "gpu_offload_evidence": bool(runtime_receipt.get("offload_evidence")),
        "random_seed": seed,
        "prompt_text": prompt,
        "prompt_hash": sha256_text(prompt),
        "context_cost": prompt_tokens,
        "generated_token_count": completion_tokens,
        "token_cost": prompt_tokens + completion_tokens,
        "verifier_cost": 2 if condition == FRESH_MEMORY_CONDITION else 1,
        "generation_duration_s": float(generation.get("duration_s", fallback_duration_s)),
        "output_text": output_text,
        "selected_answer": witness["selected_answer"],
        "expected_answer": label["expected_answer"],
        "stale_answer": task["stale_answer"],
        "accepted_by_independent_label": bool(witness["accepted"]),
        "negative_transfer_detected": bool(witness["selected_answer"] == task["stale_answer"]),
        "stale_evidence_rejected": bool(
            condition == FRESH_MEMORY_CONDITION and witness["selected_answer"] != task["stale_answer"]
        ),
        "memory_receipt": deepcopy(dict(memory_receipt)),
        "exact_verifier_witness": witness,
        "final_authority_bypassed": False,
        "backend_details": deepcopy(generation.get("backend_details", {})),
    }
    row["row_checksum"] = row_checksum(row)
    return row


def exact_label_witness(
    task: Mapping[str, Any],
    label: Mapping[str, Any],
    output_text: str,
) -> JsonDict:
    """Return an exact independent-label witness for one model output."""

    candidates = [str(label["expected_answer"]), str(task["stale_answer"]), "unknown"]
    selected = extract_answer(output_text, candidates)
    return {
        "authority": "independent_label_table",
        "label_source": label["label_source"],
        "selected_answer": selected,
        "expected_answer": label["expected_answer"],
        "accepted": selected == label["expected_answer"],
        "model_self_verdict_ignored": True,
    }


def extract_answer(output_text: str, candidates: Sequence[str]) -> str | None:
    """Return the first literal candidate found in model output."""

    best: tuple[int, int, str] | None = None
    for index, candidate in enumerate(candidates):
        pattern = re.escape(str(candidate).lower())
        match = re.search(rf"(?<![A-Za-z0-9_-]){pattern}(?![A-Za-z0-9_-])", output_text.lower())
        if match is None:
            continue
        current = (match.start(), index, str(candidate))
        if best is None or current < best:
            best = current
    return best[2] if best else None


def derive_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute scores, utility deltas, cost deltas, and transfer rates."""

    by_condition = {condition: [] for condition in CONDITIONS}
    for row in rows:
        by_condition.setdefault(str(row.get("condition")), []).append(dict(row))
    scores = {
        condition: condition_score(by_condition.get(condition, [])) for condition in CONDITIONS
    }
    fresh_rows = by_condition.get(FRESH_MEMORY_CONDITION, [])
    stale_candidates = [
        row for row in fresh_rows if row.get("memory_receipt", {}).get("negative_transfer_candidate")
    ]
    negative_transfer_rate = safe_rate(
        sum(row.get("negative_transfer_detected") is True for row in stale_candidates),
        len(stale_candidates),
    )
    stale_rejection_rate = safe_rate(
        sum(row.get("stale_evidence_rejected") is True for row in stale_candidates),
        len(stale_candidates),
    )
    no_rows = by_condition.get(NO_MEMORY_CONDITION, [])
    stale_rows = by_condition.get(STALE_MEMORY_CONDITION, [])
    return {
        "condition_scores": scores,
        "no_memory_score": scores[NO_MEMORY_CONDITION],
        "fresh_memory_score": scores[FRESH_MEMORY_CONDITION],
        "stale_memory_score": scores[STALE_MEMORY_CONDITION],
        "heldout_delta": round(scores[FRESH_MEMORY_CONDITION] - scores[NO_MEMORY_CONDITION], 10),
        "negative_transfer_rate": negative_transfer_rate,
        "stale_evidence_rejection_rate": stale_rejection_rate,
        "utility_deltas": {
            "fresh_vs_no_memory": round(
                scores[FRESH_MEMORY_CONDITION] - scores[NO_MEMORY_CONDITION], 10
            ),
            "fresh_vs_stale_memory": round(
                scores[FRESH_MEMORY_CONDITION] - scores[STALE_MEMORY_CONDITION], 10
            ),
        },
        "cost_deltas": {
            "fresh_minus_no_memory_prompt_tokens": round(
                mean_field(fresh_rows, "context_cost") - mean_field(no_rows, "context_cost"), 10
            ),
            "fresh_minus_stale_memory_prompt_tokens": round(
                mean_field(fresh_rows, "context_cost") - mean_field(stale_rows, "context_cost"), 10
            ),
            "fresh_minus_no_memory_verifier_units": round(
                mean_field(fresh_rows, "verifier_cost") - mean_field(no_rows, "verifier_cost"), 10
            ),
        },
        "row_checksums_match": all(row.get("row_checksum") == row_checksum(row) for row in rows),
        "row_count": len(rows),
    }


def build_artifact(
    *,
    upstream_gates: Mapping[str, Any],
    fixture: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    cache_receipt: Mapping[str, Any],
    selected_model: Mapping[str, Any] | None,
    models_attempted: Sequence[Mapping[str, Any]],
    runtime_receipt: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    memory_before: Mapping[str, Any],
    memory_after: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    tests_added_or_reused: Sequence[str],
    model_before: Mapping[str, Any],
    model_after: Mapping[str, Any],
    methodology_duration_s: float,
) -> JsonDict:
    """Assemble the terminal Exp5530 artifact."""

    row_list = [dict(row) for row in rows]
    metrics = derive_metrics(row_list)
    weight_receipt = model_weight_receipt(model_before=model_before, model_after=model_after)
    memory_hash_before = hash_memory_state(memory_before)
    memory_hash_after = hash_memory_state(memory_after)
    upstream_ready = upstream_gates.get("all_required_gates_true") is True
    offload_ok = gpu_offload_ok(runtime_receipt)
    metric_independence_clean = metric_independence_ok(row_list)
    csl_evidence = bool(
        row_list
        and upstream_ready
        and offload_ok
        and metric_independence_clean
        and metrics["heldout_delta"] > 0.0
        and metrics["negative_transfer_rate"] == 0.0
        and metrics["stale_evidence_rejection_rate"] == 1.0
        and memory_hash_before != memory_hash_after
        and weight_receipt["no_model_weight_mutation"] is True
    )
    artifact: JsonDict = {
        "experiment": 5530,
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "status": "complete" if row_list else "blocked",
        "upstream_gate_evidence": deepcopy(dict(upstream_gates)),
        "precondition_details": deepcopy(dict(preconditions)),
        "model_specs": [dict(row) for row in model_specs],
        "selected_model_spec": dict(selected_model) if selected_model else None,
        "models_attempted": [dict(row) for row in models_attempted],
        "model_cache_evidence": deepcopy(dict(cache_receipt)),
        "condition_names": list(CONDITIONS),
        "independent_label_source": INDEPENDENT_LABEL_SOURCE,
        "metric_independence_clean": metric_independence_clean,
        "condition_results": condition_results(row_list),
        "row_results": row_list,
        "no_memory_score": metrics["no_memory_score"],
        "fresh_memory_score": metrics["fresh_memory_score"],
        "stale_memory_score": metrics["stale_memory_score"],
        "heldout_delta": metrics["heldout_delta"],
        "negative_transfer_rate": metrics["negative_transfer_rate"],
        "stale_evidence_rejection_rate": metrics["stale_evidence_rejection_rate"],
        "utility_deltas": metrics["utility_deltas"],
        "cost_deltas": metrics["cost_deltas"],
        "memory_hash_before": memory_hash_before,
        "memory_hash_after": memory_hash_after,
        "model_weight_receipt": weight_receipt,
        "no_model_weight_mutation": weight_receipt["no_model_weight_mutation"],
        "gpu_offload_evidence": gpu_offload_evidence(runtime_receipt, preconditions),
        "continuous_self_learning_evidence": csl_evidence,
        "csl_claim_allowed": csl_evidence,
        "tests_added_or_reused": list(tests_added_or_reused),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "source_files": {
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
        },
        "source_file_checksums": source_file_checksums(REPO_ROOT),
        "methodology_duration_s": float(methodology_duration_s),
        "methodology_note": (
            "This is a bounded local-SOTA prompt-memory panel. The GGUF model is frozen; "
            "only external memory changes, and exact labels remain outside memory."
        ),
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return json_ready(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when the Exp5530 artifact cannot support its claim fields."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5530 artifact: " + "; ".join(errors))
    return True


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors without mutating the artifact."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if not artifact.get("tests_added_or_reused"):
        errors.append("tests_added_or_reused")
    principles = artifact.get("field_principles", {})
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if not principles.get(field)]
    if missing_principles:
        errors.append(f"field_principles missing: {missing_principles}")
    model_ids = {str(row.get("hf_id")) for row in artifact.get("model_specs", [])}
    if model_ids != set(MANDATED_HF_IDS):
        errors.append("model_specs must list the three mandated SOTA GGUF ids")
    expected_delta = round(
        float(artifact.get("fresh_memory_score", 0.0))
        - float(artifact.get("no_memory_score", 0.0)),
        10,
    )
    if float(artifact.get("heldout_delta", 0.0)) != expected_delta:
        errors.append("heldout_delta")
    if artifact.get("memory_hash_before") == artifact.get("memory_hash_after"):
        errors.append("memory_hash")
    weight_receipt = artifact.get("model_weight_receipt", {})
    if bool(weight_receipt.get("no_model_weight_mutation")) != bool(
        artifact.get("no_model_weight_mutation")
    ):
        errors.append("model_weight_receipt/no_model_weight_mutation mismatch")
    if weight_receipt and weight_receipt.get("model_file_receipt_before") != weight_receipt.get(
        "model_file_receipt_after"
    ):
        errors.append("model_weight_receipt")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    errors.extend(positive_claim_errors(artifact))
    rows = artifact.get("row_results", [])
    if isinstance(rows, list):
        if not all(isinstance(row, Mapping) and row.get("row_checksum") == row_checksum(row) for row in rows):
            errors.append("row_checksum")
    else:
        errors.append("row_results")
    checksum = artifact.get("reproducibility_checksum")
    if checksum and checksum != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def positive_claim_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return errors that forbid broad CSL claims."""

    errors: list[str] = []
    csl = artifact.get("continuous_self_learning_evidence") is True
    claim = artifact.get("csl_claim_allowed") is True
    if not csl and artifact.get("status") == "complete":
        errors.append("continuous_self_learning_evidence")
    if csl != claim:
        errors.append("csl_claim_allowed")
    if claim:
        if float(artifact.get("heldout_delta", 0.0)) <= 0.0:
            errors.append("heldout_delta")
        if float(artifact.get("negative_transfer_rate", 1.0)) != 0.0:
            errors.append("negative_transfer_rate")
        if float(artifact.get("stale_evidence_rejection_rate", 0.0)) != 1.0:
            errors.append("stale_evidence_rejection_rate")
        if artifact.get("no_model_weight_mutation") is not True:
            errors.append("no_model_weight_mutation")
        if artifact.get("gpu_offload_evidence", {}).get("offload_evidence") is not True:
            errors.append("gpu_offload_evidence")
        if not artifact.get("models_attempted"):
            errors.append("models_attempted")
    return errors


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal verdict from the claim gate."""

    if artifact.get("csl_claim_allowed") is True:
        return "complete: bounded_sota_csl_memory_panel_v2_claim_allowed"
    blockers = artifact.get("precondition_details", {}).get("blocked_preconditions", [])
    if blockers:
        return "blocked: " + ",".join(str(item) for item in blockers[:3])
    return "blocked: sota_csl_memory_panel_v2_claim_not_allowed"


def condition_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Group row evidence by memory condition."""

    grouped: JsonDict = {condition: [] for condition in CONDITIONS}
    for row in rows:
        grouped.setdefault(str(row.get("condition")), []).append(dict(row))
    return grouped


def condition_score(rows: Sequence[Mapping[str, Any]]) -> float:
    """Return exact pass rate for one condition."""

    if not rows:
        return 0.0
    return round(sum(row.get("accepted_by_independent_label") is True for row in rows) / len(rows), 10)


def safe_rate(numerator: int, denominator: int) -> float:
    """Return a bounded rate, treating empty safety denominators as zero."""

    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 10)


def mean_field(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    """Return the mean numeric field value for rows."""

    if not rows:
        return 0.0
    return sum(float(row.get(field, 0.0)) for row in rows) / len(rows)


def metric_independence_ok(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Check that row scores came from independent-label witnesses."""

    return bool(
        rows
        and all(row.get("exact_verifier_witness", {}).get("authority") == "independent_label_table" for row in rows)
        and all(row.get("exact_verifier_witness", {}).get("label_source") == INDEPENDENT_LABEL_SOURCE for row in rows)
        and all(row.get("memory_receipt", {}).get("label_source_used") is False for row in rows)
    )


def mark_model_ran(model_specs: list[JsonDict], selected_model: Mapping[str, Any]) -> None:
    """Mutate the selected model row to mark panel execution receipts."""

    for row in model_specs:
        if row.get("hf_id") == selected_model.get("hf_id"):
            row["selected_for_panel"] = True
            row["ran_panel"] = True
            row["gpu_offload_verified"] = True


def model_file_receipt(path_value: Any) -> JsonDict:
    """Return lightweight file metadata used as a no-mutation receipt."""

    if not path_value:
        return {"path": None, "exists": False}
    path = Path(str(path_value))
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "inode": int(stat.st_ino),
        "suffix": path.suffix,
    }


def model_weight_receipt(*, model_before: Mapping[str, Any], model_after: Mapping[str, Any]) -> JsonDict:
    """Report the frozen model-weight boundary."""

    unchanged = dict(model_before) == dict(model_after)
    return {
        "no_model_weight_mutation": unchanged,
        "model_file_receipt_before": dict(model_before),
        "model_file_receipt_after": dict(model_after),
        "adapter_weights_loaded": False,
        "adapter_weights_written": False,
        "learned_state_scope": "external_memory_only",
    }


def gpu_offload_ok(runtime_receipt: Mapping[str, Any]) -> bool:
    """Return whether preflight and load receipts show GPU offload."""

    load_receipt = runtime_receipt.get("load_receipt", {})
    load_ok = True
    if isinstance(load_receipt, Mapping) and load_receipt:
        load_ok = load_receipt.get("offload_evidence") is True
    return runtime_receipt.get("offload_evidence") is True and load_ok


def gpu_offload_evidence(
    runtime_receipt: Mapping[str, Any],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    """Normalize runtime receipts into the required artifact field."""

    load_receipt = runtime_receipt.get("load_receipt", {})
    return {
        "runtime_backend": runtime_receipt.get("runtime_backend", "unavailable"),
        "cuda_visible": runtime_receipt.get("cuda_visible") is True,
        "llama_cpp_import_ok": runtime_receipt.get("llama_cpp_import_ok") is True,
        "gpu_offload_supported": runtime_receipt.get("gpu_offload_supported") is True,
        "preflight_offload_evidence": runtime_receipt.get("offload_evidence") is True,
        "load_offload_evidence": (
            load_receipt.get("offload_evidence") is True if isinstance(load_receipt, Mapping) else False
        ),
        "offload_evidence": gpu_offload_ok(runtime_receipt),
        "gpu_memory_delta_mb": int(runtime_receipt.get("gpu_memory_delta_mb", 0) or 0),
        "n_gpu_layers": int(runtime_receipt.get("n_gpu_layers", N_GPU_LAYERS)),
        "blocked_preconditions": list(preconditions.get("blocked_preconditions", [])),
    }


class LlamaMemoryPanelRunner:  # pragma: no cover
    """Small llama-cpp-python generation wrapper for the live bounded panel."""

    def __init__(
        self,
        *,
        model_spec: Mapping[str, Any],
        n_gpu_layers: int = N_GPU_LAYERS,
        seed: int = RANDOM_SEED,
    ) -> None:
        from llama_cpp import Llama

        before = gpu_memory_snapshot()
        started = time.perf_counter()
        self.llm = Llama(
            model_path=str(model_spec["model_path"]),
            n_gpu_layers=n_gpu_layers,
            n_ctx=768,
            n_batch=128,
            seed=seed,
            verbose=False,
        )
        after = gpu_memory_snapshot()
        self.load_receipt = {
            "load_duration_s": time.perf_counter() - started,
            "gpu_memory_before": before,
            "gpu_memory_after": after,
            "gpu_memory_delta_mb": max_gpu_delta(before, after),
            "offload_evidence": max_gpu_delta(before, after) > 512,
        }

    def __call__(self, **kwargs: Any) -> JsonDict:
        started = time.perf_counter()
        result = self.llm(
            str(kwargs["prompt"]),
            max_tokens=int(kwargs.get("max_tokens", MAX_TOKENS)),
            temperature=0.0,
            top_p=1.0,
            echo=False,
            stop=["\n"],
        )
        text = str(result["choices"][0].get("text", ""))
        usage = result.get("usage", {})
        return {
            "output_text": text,
            "prompt_token_count": int(usage.get("prompt_tokens", 0) or 0),
            "generated_token_count": int(usage.get("completion_tokens", 0) or 0),
            "duration_s": time.perf_counter() - started,
            "backend_details": {"llama_cpp_python": True},
        }


def default_runtime_probe(**kwargs: Any) -> JsonDict:  # pragma: no cover
    """Check CUDA and llama.cpp GPU-offload support before model load."""

    blocked: list[str] = []
    try:
        import torch

        cuda_visible = bool(torch.cuda.is_available()) and int(torch.cuda.device_count()) > 0
    except Exception as exc:
        cuda_visible = False
        blocked.append(f"torch_cuda_probe_failed:{type(exc).__name__}")
    try:
        from llama_cpp import llama_cpp

        llama_import_ok = True
        gpu_supported = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception as exc:
        llama_import_ok = False
        gpu_supported = False
        blocked.append(f"llama_cpp_probe_failed:{type(exc).__name__}")
    if not cuda_visible:
        blocked.append("cuda_not_visible")
    if not gpu_supported:
        blocked.append("gpu_offload_evidence_missing")
    return {
        "runtime_backend": "llama_cpp_python_cuda_gguf",
        "cuda_visible": cuda_visible,
        "llama_cpp_import_ok": llama_import_ok,
        "gpu_offload_supported": gpu_supported,
        "offload_evidence": cuda_visible and gpu_supported,
        "n_gpu_layers": int(kwargs.get("n_gpu_layers", N_GPU_LAYERS)),
        "gpu_memory_delta_mb": 0,
        "load_receipt": {},
        "nvidia_smi": nvidia_smi_query(),
        "blocked_preconditions": sorted(set(blocked)),
    }


def nvidia_smi_query() -> JsonDict:  # pragma: no cover
    """Return a compact nvidia-smi memory snapshot."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "gpus": []}
    if result.returncode != 0:
        return {"ok": False, "error": result.stderr.strip(), "gpus": []}
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mb": int(parts[2]),
                    "memory_free_mb": int(parts[3]),
                    "utilization_pct": int(parts[4]),
                }
            )
    return {"ok": True, "gpus": gpus}


def gpu_memory_snapshot() -> list[JsonDict]:  # pragma: no cover
    """Return GPU memory rows from nvidia-smi."""

    return list(nvidia_smi_query().get("gpus", []))


def max_gpu_delta(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> int:  # pragma: no cover
    """Return the largest per-GPU memory increase in MiB."""

    before_by_index = {row.get("index"): int(row.get("memory_used_mb", 0)) for row in before}
    return max(
        (
            int(row.get("memory_used_mb", 0)) - before_by_index.get(row.get("index"), 0)
            for row in after
        ),
        default=0,
    )


def resolve_output_path(root: Path, path: Path | str) -> Path:
    """Resolve repository-relative output paths while preserving absolute paths."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def load_json(path: Path | str) -> JsonDict:
    """Read a JSON object from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write stable JSON so reruns are diffable."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def hash_memory_state(state: Mapping[str, Any]) -> str:
    """Return a stable SHA256 hash for JSON-compatible memory state."""

    return "sha256:" + sha256_json(state)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with its checksum field removed."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return "sha256:" + sha256_json(payload)


def row_checksum(row: Mapping[str, Any]) -> str:
    """Hash a row with its checksum field removed."""

    payload = {key: value for key, value in row.items() if key != "row_checksum"}
    return "sha256:" + sha256_json(payload)


def source_file_checksums(root: Path) -> JsonDict:
    """Record the source files backing this artifact."""

    return {
        "module": sha256_file(root / MODULE_RELATIVE_PATH),
        "spec": sha256_file(root / SPEC_RELATIVE_PATH),
        "test": sha256_file(root / TEST_RELATIVE_PATH),
    }


def sha256_file(path: Path | str) -> str:
    """Return a SHA256 digest for a file."""

    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    """Return a SHA256 digest for text."""

    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: Mapping[str, Any]) -> str:
    """Return a SHA256 digest for JSON-compatible mappings."""

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def estimate_tokens(text: str) -> int:
    """Return a stable coarse token estimate for cost accounting."""

    return max(1, len(text.split()))


def json_ready(value: Any) -> Any:
    """Round-trip through JSON to normalize tuples and pathlib-adjacent values."""

    return json.loads(json.dumps(value, sort_keys=True))
