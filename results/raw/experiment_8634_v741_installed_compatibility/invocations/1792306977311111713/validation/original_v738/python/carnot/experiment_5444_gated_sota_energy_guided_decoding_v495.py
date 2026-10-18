#!/usr/bin/env python3
"""Exp5444 gated local SOTA verifier-potential decoding pilot.

Spec refs: REQ-SAFE-5444, SCENARIO-SAFE-5444.

This experiment is deliberately narrow: it compares unconstrained decoding,
grammar-only JSON decoding, and verifier-potential guided prefix decoding on
the deterministic Exp5443 fixtures.  The live model is only a candidate
generator.  Every candidate is judged by the same exact final verifiers used by
the fixture, so a model self-claim, a parseable JSON object, or a promising
prefix can never become final evidence by itself.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any

from carnot import experiment_5443_verifier_potential_prefix_fixture_v495 as exp5443
from carnot.inference.sota_models import resolve_cached_gguf


JsonDict = dict[str, Any]
ModelResolver = Callable[[str, str], str | None]
RuntimeProbe = Callable[..., JsonDict]
GenerationRunner = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5444_gated_sota_energy_guided_decoding_v495.json")
ROW_RESULTS_RELATIVE_PATH = Path(
    "results/experiment_5444_gated_sota_energy_guided_decoding_v495_rows.jsonl"
)
EXPERIMENT_ID = "experiment_5444_gated_sota_energy_guided_decoding_v495"
TASK_ID = "exp5444-v495-gated-sota-energy-guided-decoding"
MILESTONE = "2026.07.495"
RUN_DATE = "2026-07-08"
SCHEMA = "carnot.experiment_5444.gated_sota_energy_guided_decoding.v495"
SPEC_REFS = ("REQ-SAFE-5444", "SCENARIO-SAFE-5444")
RANDOM_SEED = 5444
DEFAULT_QUANTIZATION = "Q4_K_M"
DEFAULT_TOKEN_BUDGET = 96
DEFAULT_MAX_FIXTURES = 4
N_GPU_LAYERS = -1
INFERENCE_SUBSTRATE = "live_llm_inference"
BLOCKED_INFERENCE_SUBSTRATE = "blocked_preconditions_no_live_llm"
TERMINAL_PREFIXES = ("complete:", "blocked:")

CONDITION_NAMES = ("unconstrained", "grammar_only", "verifier_potential_guided")
GUIDED_CONDITION = "verifier_potential_guided"
BASELINE_CONDITIONS = ("unconstrained", "grammar_only")
UNSAFE_FALSE_ACCEPT_FAMILIES = {"schema_only_trap", "api_call_witness"}

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "role": "flagship_moe",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "quantization": DEFAULT_QUANTIZATION,
    },
    {
        "role": "flagship_dense",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "quantization": DEFAULT_QUANTIZATION,
    },
    {
        "role": "middle_moe",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "quantization": DEFAULT_QUANTIZATION,
    },
)
MANDATED_HF_IDS = tuple(str(spec["hf_id"]) for spec in MANDATED_MODEL_SPECS)
RUN_ROLE_PREFERENCE = ("middle_moe", "flagship_dense", "flagship_moe")

FIELD_PRINCIPLES: dict[str, str] = {
    "preconditions_checked": "compute-bound task must fail fast.",
    "model_specs": "mandated SOTA GGUF provenance.",
    "headline_required_any_of": "confirms at least one mandated SOTA model ran.",
    "runtime_backend": "GGUF/llama.cpp path, not transformers tokenizer path.",
    "gpu_offload_verified": "no CPU-only SOTA headline.",
    "fixture_count": "bounded evaluation size.",
    "condition_names": "baseline clarity.",
    "row_results_path": "inspectable evidence.",
    "exact_final_authority": "deterministic verifier authority.",
    "reward_evaluation_budget": "inference cost accounting.",
    "guided_validity_delta_vs_unconstrained": "utility measurement.",
    "guided_validity_delta_vs_grammar_only": "incremental utility measurement.",
    "semantic_false_accept_rate": "hallucination boundary.",
    "unsafe_false_accept_rate": "safety boundary.",
    "action_unreachability_rate": "action-reachability boundary.",
    "abstention_rate": "selective behavior boundary.",
    "metric_independence_checks_passed": "tautology prevention.",
    "verifier_guided_decoding_ready": "capstone evidence.",
    "inference_substrate": "real local model invocation.",
    "honest_verdict": 'terminal status; start with "complete:" or "blocked:".',
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)

JSON_GBNF = r'''
root ::= object
object ::= "{" ws ( string ":" ws value ("," ws string ":" ws value)* )? "}" ws
array ::= "[" ws ( value ("," ws value)* )? "]" ws
value ::= object | array | string | number | "true" ws | "false" ws | "null" ws
string ::= "\"" ( [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) )* "\"" ws
number ::= "-"? ( "0" | [1-9] [0-9]* ) ( "." [0-9]+ )? ( [eE] [+-]? [0-9]+ )? ws
ws ::= [ \t\n\r]*
'''


def run(
    *,
    root: Path | str = REPO_ROOT,
    artifact_path: Path | str | None = None,
    row_results_path: Path | str | None = None,
    fixture_artifact: Mapping[str, Any] | None = None,
    model_resolver: ModelResolver = resolve_cached_gguf,
    runtime_probe: RuntimeProbe | None = None,
    generation_runner: GenerationRunner | None = None,
    max_fixtures: int = DEFAULT_MAX_FIXTURES,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
    write: bool = True,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    """Run the gated pilot and optionally write the artifact and row JSONL."""

    root_path = Path(root)
    destination = _destination(root_path, artifact_path, RESULT_RELATIVE_PATH)
    rows_destination = _destination(root_path, row_results_path, ROW_RESULTS_RELATIVE_PATH)
    fixture_payload = (
        dict(fixture_artifact)
        if fixture_artifact is not None
        else _read_json(root_path / exp5443.RESULT_RELATIVE_PATH)
    )
    model_specs = resolve_model_specs(model_resolver)
    selected_model = select_headline_model(model_specs)
    runtime_fn = runtime_probe or default_runtime_probe
    runtime_receipt = runtime_fn(model_spec=selected_model, n_gpu_layers=N_GPU_LAYERS)
    preconditions = evaluate_preconditions(
        fixture_payload=fixture_payload,
        model_specs=model_specs,
        selected_model=selected_model,
        runtime_receipt=runtime_receipt,
    )

    live_runner: GenerationRunner | None = generation_runner
    runner_load_receipt: JsonDict = {}
    if preconditions["all_passed"] and live_runner is None:
        try:
            live = LlamaCppGenerationRunner(
                model_spec=selected_model or {},
                n_gpu_layers=N_GPU_LAYERS,
                seed=random_seed,
            )
        except Exception as exc:  # pragma: no cover - depends on local runtime/model.
            preconditions["all_passed"] = False
            preconditions["blocked_preconditions"].append(
                f"llama_cpp_model_load_failed:{type(exc).__name__}: {exc}"
            )
            live_runner = None
        else:
            live_runner = live
            runner_load_receipt = dict(live.load_receipt)
            runtime_receipt = {**runtime_receipt, "load_receipt": runner_load_receipt}
            if not bool(runner_load_receipt.get("offload_evidence")):
                preconditions["all_passed"] = False
                preconditions["blocked_preconditions"].append("gpu_offload_not_observed_after_load")

    if not preconditions["all_passed"] or live_runner is None:
        artifact = build_artifact(
            model_specs=model_specs,
            selected_model=selected_model,
            runtime_receipt=runtime_receipt,
            preconditions=preconditions,
            rows=[],
            row_results_path=rows_destination,
            tests_run=tests_run,
            blocked=True,
        )
        if write:
            _write_json(destination, artifact)
        validate_artifact(artifact)
        return artifact

    selected_fixtures = select_fixture_rows(fixture_payload, max_fixtures=max_fixtures)
    rows = run_condition_rows(
        fixtures=selected_fixtures,
        model_spec=selected_model or {},
        runtime_receipt=runtime_receipt,
        generation_runner=live_runner,
        random_seed=random_seed,
    )
    artifact = build_artifact(
        model_specs=_mark_model_ran(model_specs, selected_model),
        selected_model=selected_model,
        runtime_receipt=runtime_receipt,
        preconditions=preconditions,
        rows=rows,
        row_results_path=rows_destination,
        tests_run=tests_run,
        blocked=False,
    )
    if write:
        _write_jsonl(rows_destination, rows)
        validate_artifact(artifact)
        _write_json(destination, artifact)
    else:
        validate_artifact({**artifact, "row_results_path": str(rows_destination)})
    return artifact


def resolve_model_specs(model_resolver: ModelResolver = resolve_cached_gguf) -> list[JsonDict]:
    """Resolve the mandated local GGUF specs without using transformers tokenizers."""

    rows: list[JsonDict] = []
    for spec in MANDATED_MODEL_SPECS:
        hf_id = str(spec["hf_id"])
        quantization = str(spec.get("quantization", DEFAULT_QUANTIZATION))
        path_text = model_resolver(hf_id, quantization)
        path = Path(path_text) if path_text else None
        local = bool(path and path.is_file())
        rows.append(
            {
                "role": str(spec["role"]),
                "hf_id": hf_id,
                "quantization": quantization,
                "model_path": str(path) if path else None,
                "local_path_available": local,
                "file_receipt": _file_receipt(path) if local and path is not None else None,
                "runtime_backend": None,
                "n_gpu_layers": None,
                "gpu_offload_verified": False,
                "ran_headline": False,
                "legacy_smoke_only": False,
            }
        )
    return rows


def select_headline_model(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict | None:
    """Select one resolved mandated model, preferring the lower-active-parameter MoE."""

    by_role = {str(row.get("role")): row for row in model_specs}
    for role in RUN_ROLE_PREFERENCE:
        row = by_role.get(role)
        if isinstance(row, Mapping) and row.get("local_path_available") is True:
            return dict(row)
    return None


def default_runtime_probe(**kwargs: Any) -> JsonDict:  # pragma: no cover
    """Check CUDA and llama.cpp GPU-offload support before generation."""

    blocked: list[str] = []
    torch_cuda: JsonDict
    try:
        import torch  # noqa: PLC0415

        torch_cuda = {
            "import_ok": True,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
        if not torch_cuda["cuda_available"] or int(torch_cuda["device_count"]) <= 0:
            blocked.append("torch_cuda_unavailable")
    except Exception as exc:  # pragma: no cover - environment-dependent.
        torch_cuda = {"import_ok": False, "error": f"{type(exc).__name__}: {exc}"}
        blocked.append("torch_import_failed")

    llama_info: JsonDict
    try:
        from llama_cpp import llama_cpp  # noqa: PLC0415

        supports = bool(llama_cpp.llama_supports_gpu_offload())
        system_info_raw = llama_cpp.llama_print_system_info()
        if isinstance(system_info_raw, bytes):
            system_info = system_info_raw.decode("utf-8", "replace")
        else:
            system_info = str(system_info_raw)
        llama_info = {
            "import_ok": True,
            "gpu_offload_supported": supports,
            "system_info": system_info,
        }
        if not supports:
            blocked.append("llama_cpp_gpu_offload_unsupported")
    except Exception as exc:  # pragma: no cover - environment-dependent.
        llama_info = {"import_ok": False, "error": f"{type(exc).__name__}: {exc}"}
        blocked.append("llama_cpp_import_failed")

    nvidia_smi = _nvidia_smi_query()
    if not nvidia_smi.get("ok"):
        blocked.append("nvidia_smi_unavailable")

    return {
        "runtime_backend": "llama_cpp_python_cuda_gguf",
        "llama_cpp_import_ok": bool(llama_info.get("import_ok")),
        "cuda_visible": bool(torch_cuda.get("cuda_available"))
        and int(torch_cuda.get("device_count", 0)) > 0,
        "gpu_offload_supported": bool(llama_info.get("gpu_offload_supported")),
        "n_gpu_layers": int(kwargs.get("n_gpu_layers", N_GPU_LAYERS)),
        "offload_evidence": bool(llama_info.get("gpu_offload_supported"))
        and bool(torch_cuda.get("cuda_available")),
        "torch_cuda": torch_cuda,
        "llama_cpp": llama_info,
        "nvidia_smi": nvidia_smi,
        "blocked_preconditions": sorted(set(blocked)),
    }


def evaluate_preconditions(
    *,
    fixture_payload: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    selected_model: Mapping[str, Any] | None,
    runtime_receipt: Mapping[str, Any],
) -> JsonDict:
    """Combine the Exp5443, model-cache, CUDA, and offload gates."""

    blocked = list(runtime_receipt.get("blocked_preconditions", []))
    fixture_ready = fixture_payload.get("verifier_potential_fixture_ready") is True
    if not fixture_ready:
        blocked.append("exp5443_verifier_potential_fixture_not_ready")
    mandated_present = {str(row.get("hf_id")) for row in model_specs} == set(MANDATED_HF_IDS)
    if not mandated_present:
        blocked.append("mandated_model_specs_missing")
    if selected_model is None:
        blocked.append("no_mandated_local_gguf_model_path")
    if runtime_receipt.get("cuda_visible") is not True:
        blocked.append("cuda_not_visible")
    if runtime_receipt.get("offload_evidence") is not True:
        blocked.append("gpu_offload_evidence_missing")
    return {
        "fixture_ready": fixture_ready,
        "mandated_model_specs_present": mandated_present,
        "local_model_path_available": selected_model is not None,
        "cuda_visible": runtime_receipt.get("cuda_visible") is True,
        "runtime_backend": str(runtime_receipt.get("runtime_backend", "")),
        "gpu_offload_preflight": runtime_receipt.get("offload_evidence") is True,
        "blocked_preconditions": sorted(set(str(item) for item in blocked)),
        "all_passed": not blocked,
    }


def select_fixture_rows(
    fixture_payload: Mapping[str, Any],
    *,
    max_fixtures: int = DEFAULT_MAX_FIXTURES,
) -> list[JsonDict]:
    """Select a bounded set of Exp5443 rows with exact final verifier support."""

    rows = fixture_payload.get("fixture_rows")
    if not isinstance(rows, list):
        return []
    selected: list[JsonDict] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        verdict = _mapping(row.get("exact_final_verdict"))
        if verdict.get("verified") is not True:
            continue
        selected.append(copy.deepcopy(dict(row)))
        if len(selected) >= max_fixtures:
            break
    return selected


def run_condition_rows(
    *,
    fixtures: Sequence[Mapping[str, Any]],
    model_spec: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any],
    generation_runner: GenerationRunner,
    random_seed: int = RANDOM_SEED,
) -> list[JsonDict]:
    """Generate and score one row per fixture and condition."""

    rows: list[JsonDict] = []
    runtime_backend = str(runtime_receipt.get("runtime_backend", "llama_cpp_python_cuda_gguf"))
    for fixture_index, fixture in enumerate(fixtures):
        fixture_row = copy.deepcopy(dict(fixture))
        for condition_index, condition in enumerate(CONDITION_NAMES):
            seed = random_seed + fixture_index * 101 + condition_index
            prompt = build_prompt(fixture_row, condition=condition)
            prompt_hash = _sha256_text(prompt)
            prefix = _best_guided_prefix(fixture_row) if condition == GUIDED_CONDITION else None
            grammar = JSON_GBNF if condition in {"grammar_only", GUIDED_CONDITION} else None
            started = time.perf_counter()
            generation = generation_runner(
                prompt=prompt,
                condition=condition,
                fixture=fixture_row,
                model_spec=model_spec,
                runtime_backend=runtime_backend,
                grammar=grammar,
                prefix_seed=prefix,
                seed=seed,
                max_tokens=DEFAULT_TOKEN_BUDGET,
                n_gpu_layers=N_GPU_LAYERS,
            )
            elapsed = time.perf_counter() - started
            row = score_candidate_row(
                fixture=fixture_row,
                condition=condition,
                model_spec=model_spec,
                runtime_backend=runtime_backend,
                runtime_receipt=runtime_receipt,
                generation=generation,
                prompt_hash=prompt_hash,
                seed=seed,
                token_budget=DEFAULT_TOKEN_BUDGET,
                fallback_duration_s=elapsed,
                prefix_seed=prefix,
            )
            rows.append(row)
    return rows


def build_prompt(fixture: Mapping[str, Any], *, condition: str) -> str:
    """Build the condition-specific prompt while keeping exact checks hidden in code."""

    family = str(fixture.get("constraint_family"))
    lines = [
        "Return exactly one compact JSON object and no prose.",
        f"Fixture id: {fixture.get('fixture_id')}",
        f"Constraint family: {family}",
        f"Required keys: {json.dumps(fixture.get('required_keys', []), sort_keys=True)}",
        f"Allowed keys: {json.dumps(fixture.get('allowed_keys', []), sort_keys=True)}",
        _family_instruction(fixture, family),
    ]
    if condition == "unconstrained":
        lines.append("No grammar or prefix is applied in this condition.")
    elif condition == "grammar_only":
        lines.append("A JSON grammar constrains syntax only; satisfy the semantics yourself.")
    elif condition == GUIDED_CONDITION:
        prefix = _best_guided_prefix(fixture)
        lines.append("A deterministic verifier potential selected this partial JSON prefix:")
        lines.append(json.dumps(_mapping(prefix.get("fields")) if prefix else {}, sort_keys=True))
        lines.append("Complete or minimally repair the prefix so the final JSON satisfies all rules.")
    else:
        raise ValueError(f"unknown condition: {condition}")
    return "\n".join(lines)


def score_candidate_row(
    *,
    fixture: Mapping[str, Any],
    condition: str,
    model_spec: Mapping[str, Any],
    runtime_backend: str,
    runtime_receipt: Mapping[str, Any],
    generation: Mapping[str, Any],
    prompt_hash: str,
    seed: int,
    token_budget: int,
    fallback_duration_s: float,
    prefix_seed: Mapping[str, Any] | None,
) -> JsonDict:
    """Parse one model candidate and run the exact final verifier."""

    output_text = str(generation.get("output_text", ""))
    parsed, parse_error = extract_json_object(output_text)
    final_output = parsed if parsed is not None else {}
    exact_row = copy.deepcopy(dict(fixture))
    exact_row["final_output"] = final_output
    exact_verdict = exp5443.exact_final_verdict(exact_row)
    advisory_accept = parsed is not None
    reward_evaluations = _reward_evaluation_count(condition, prefix_seed)
    row: JsonDict = {
        "schema": "carnot.experiment_5444.row.v1",
        "experiment_id": EXPERIMENT_ID,
        "row_id": f"{fixture.get('row_id')}:{condition}",
        "fixture_row_id": str(fixture.get("row_id")),
        "fixture_id": str(fixture.get("fixture_id")),
        "constraint_family": str(fixture.get("constraint_family")),
        "condition": condition,
        "model_role": str(model_spec.get("role")),
        "model_hf_id": str(model_spec.get("hf_id")),
        "model_path": str(model_spec.get("model_path")),
        "runtime_backend": runtime_backend,
        "n_gpu_layers": int(runtime_receipt.get("n_gpu_layers", N_GPU_LAYERS)),
        "gpu_offload_evidence": bool(runtime_receipt.get("offload_evidence")),
        "random_seed": seed,
        "prompt_hash": prompt_hash,
        "token_budget": token_budget,
        "reward_evaluation_count": reward_evaluations,
        "generation_duration_s": float(generation.get("duration_s", fallback_duration_s)),
        "generated_token_count": int(generation.get("generated_token_count", 0) or 0),
        "output_text": output_text,
        "candidate_output": parsed,
        "parse_status": "parsed" if parsed is not None else "abstained",
        "parse_error": parse_error,
        "condition_advisory_accept": advisory_accept,
        "exact_final_verdict": exact_verdict,
        "accepted_by_final_authority": bool(exact_verdict["accepted"]),
        "final_authority_bypassed": False,
        "prefix_seed": copy.deepcopy(prefix_seed),
        "backend_details": copy.deepcopy(generation.get("backend_details", {})),
    }
    row["row_checksum"] = row_checksum(row)
    return row


def extract_json_object(text: str) -> tuple[JsonDict | None, str | None]:
    """Return the first JSON object in text, if one can be decoded."""

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        if isinstance(value, dict):
            return dict(value), None
    return None, locals().get("last_error", "no_json_object")


def derive_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute all aggregate metrics from candidate rows only."""

    row_list = [dict(row) for row in rows if isinstance(row, Mapping)]
    total = len(row_list)
    by_condition: dict[str, list[JsonDict]] = {name: [] for name in CONDITION_NAMES}
    for row in row_list:
        by_condition.setdefault(str(row.get("condition")), []).append(row)
    condition_metrics = {
        condition: _condition_metric(condition_rows)
        for condition, condition_rows in by_condition.items()
        if condition in CONDITION_NAMES
    }
    guided_rate = condition_metrics.get(GUIDED_CONDITION, {}).get("accepted_validity_rate", 0.0)
    unconstrained_rate = condition_metrics.get("unconstrained", {}).get(
        "accepted_validity_rate", 0.0
    )
    grammar_rate = condition_metrics.get("grammar_only", {}).get("accepted_validity_rate", 0.0)
    semantic_false_accepts = [
        row
        for row in row_list
        if _advisory_false_accept(row) and row.get("constraint_family") == "semantic_contradiction"
    ]
    unsafe_false_accepts = [
        row
        for row in row_list
        if _advisory_false_accept(row) and row.get("constraint_family") in UNSAFE_FALSE_ACCEPT_FAMILIES
    ]
    action_false_accepts = [
        row
        for row in row_list
        if _advisory_false_accept(row) and row.get("constraint_family") == "unreachable_tool_action"
    ]
    abstentions = [row for row in row_list if row.get("parse_status") == "abstained"]
    row_checksums_match = all(row.get("row_checksum") == row_checksum(row) for row in row_list)
    exact_authority_ok = all(
        _mapping(row.get("exact_final_verdict")).get("verified") is True
        and _mapping(row.get("exact_final_verdict")).get("authority") == "exact_final_verifier"
        and row.get("final_authority_bypassed") is False
        for row in row_list
    )
    predicate_support = {
        "accepted_validity": "exact_final_verdict.accepted is true",
        "semantic_false_accept": "advisory accept, exact reject, semantic family",
        "unsafe_false_accept": "advisory accept, exact reject, unsafe schema/API family",
        "action_unreachability": "advisory accept, exact reject, unreachable action family",
        "abstention": "parse_status is abstained",
        "guided_delta": "guided accepted-validity rate minus baseline accepted-validity rate",
    }
    metric_independence = bool(
        (not row_list or row_checksums_match)
        and exact_authority_ok
        and len(set(predicate_support.values())) == len(predicate_support)
    )
    return {
        "condition_metrics": condition_metrics,
        "accepted_validity_rate": _rate(
            [row for row in row_list if row.get("accepted_by_final_authority") is True], total
        ),
        "guided_validity_delta_vs_unconstrained": guided_rate - unconstrained_rate,
        "guided_validity_delta_vs_grammar_only": guided_rate - grammar_rate,
        "semantic_false_accept_rate": _rate(semantic_false_accepts, total),
        "unsafe_false_accept_rate": _rate(unsafe_false_accepts, total),
        "action_unreachability_rate": _rate(action_false_accepts, total),
        "abstention_rate": _rate(abstentions, total),
        "constraint_family_counts": dict(
            sorted(Counter(str(row.get("constraint_family")) for row in row_list).items())
        ),
        "metric_independence_checks_passed": metric_independence,
        "predicate_support": predicate_support,
        "row_checksums_match": row_checksums_match,
        "exact_authority_rows": sum(1 for row in row_list if _exact_authority_row(row)),
        "row_count": total,
    }


def derive_reward_budget(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute reward-evaluation accounting from candidate rows."""

    per_condition: dict[str, int] = defaultdict(int)
    per_row: list[JsonDict] = []
    for row in rows:
        condition = str(row.get("condition"))
        count = int(row.get("reward_evaluation_count", 0))
        per_condition[condition] += count
        per_row.append(
            {
                "row_id": row.get("row_id"),
                "fixture_id": row.get("fixture_id"),
                "condition": condition,
                "reward_evaluation_count": count,
            }
        )
    return {
        "unit": "one deterministic verifier-potential or exact-final verifier invocation",
        "total_reward_evaluations": sum(per_condition.values()),
        "per_condition": {condition: per_condition.get(condition, 0) for condition in CONDITION_NAMES},
        "per_row": per_row,
    }


def build_artifact(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    selected_model: Mapping[str, Any] | None,
    runtime_receipt: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    row_results_path: Path,
    tests_run: Sequence[str | Mapping[str, Any]],
    blocked: bool,
) -> JsonDict:
    """Assemble the terminal artifact from row records and preflight receipts."""

    row_list = [dict(row) for row in rows]
    metrics = derive_metrics(row_list)
    budget = derive_reward_budget(row_list)
    gpu_offload_verified = bool(
        preconditions.get("all_passed")
        and runtime_receipt.get("offload_evidence") is True
        and (runtime_receipt.get("load_receipt", {}).get("offload_evidence", True) is True)
    )
    ready = bool(
        not blocked
        and row_list
        and gpu_offload_verified
        and metrics["metric_independence_checks_passed"]
        and metrics["condition_metrics"].get(GUIDED_CONDITION, {}).get("count", 0)
        == len(row_list) // len(CONDITION_NAMES)
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "status": "complete" if ready else "blocked",
        "preconditions_checked": True,
        "precondition_details": copy.deepcopy(preconditions),
        "model_specs": [dict(row) for row in model_specs],
        "selected_model_spec": dict(selected_model) if selected_model is not None else None,
        "headline_required_any_of": list(MANDATED_HF_IDS),
        "runtime_backend": str(runtime_receipt.get("runtime_backend", "unavailable")),
        "runtime_receipt": copy.deepcopy(runtime_receipt),
        "gpu_offload_verified": gpu_offload_verified,
        "fixture_count": len({row.get("fixture_id") for row in row_list}),
        "condition_names": list(CONDITION_NAMES),
        "row_results_path": str(row_results_path),
        "exact_final_authority": True,
        "reward_evaluation_budget": budget,
        "guided_validity_delta_vs_unconstrained": metrics[
            "guided_validity_delta_vs_unconstrained"
        ],
        "guided_validity_delta_vs_grammar_only": metrics[
            "guided_validity_delta_vs_grammar_only"
        ],
        "semantic_false_accept_rate": metrics["semantic_false_accept_rate"],
        "unsafe_false_accept_rate": metrics["unsafe_false_accept_rate"],
        "action_unreachability_rate": metrics["action_unreachability_rate"],
        "abstention_rate": metrics["abstention_rate"],
        "accepted_validity_rate": metrics["accepted_validity_rate"],
        "metric_independence_checks_passed": metrics["metric_independence_checks_passed"],
        "verifier_guided_decoding_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE if ready else BLOCKED_INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(ready, preconditions, metrics),
        "condition_metrics": metrics["condition_metrics"],
        "constraint_family_counts": metrics["constraint_family_counts"],
        "metric_details": metrics,
        "field_principles": dict(FIELD_PRINCIPLES),
        "row_results": row_list,
        "row_checksums": [row.get("row_checksum") for row in row_list],
        "tests_run": _normalise_tests_run(tests_run),
        "research_conductor_modified": False,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp5444 artifact cannot support the decoding claim."""

    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema, row, precondition, and aggregate validation errors."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match REQ-SAFE-5444")
    if type(artifact.get("preconditions_checked")) is not bool:
        errors.append("preconditions_checked must be boolean")
    if not isinstance(artifact.get("model_specs"), list):
        errors.append("model_specs must be a list")
        model_specs: list[Mapping[str, Any]] = []
    else:
        model_specs = [row for row in artifact["model_specs"] if isinstance(row, Mapping)]
    model_ids = {str(row.get("hf_id")) for row in model_specs}
    if model_ids != set(MANDATED_HF_IDS):
        errors.append("mandated model_specs must include the three required GGUF IDs")
    if artifact.get("headline_required_any_of") != list(MANDATED_HF_IDS):
        errors.append("headline_required_any_of must list mandated SOTA IDs")
    if artifact.get("condition_names") != list(CONDITION_NAMES):
        errors.append("condition_names must match required baseline/guided order")
    if artifact.get("exact_final_authority") is not True:
        errors.append("exact_final_authority must be true")
    if type(artifact.get("gpu_offload_verified")) is not bool:
        errors.append("gpu_offload_verified must be boolean")
    if not isinstance(artifact.get("runtime_backend"), str):
        errors.append("runtime_backend must be a string")
    if not _bare_non_negative_int(artifact.get("fixture_count")):
        errors.append("fixture_count must be a non-negative integer")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or "\n" in verdict or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with complete: or blocked:")
    if artifact.get("research_conductor_modified") is not False:
        errors.append("scripts/research_conductor.py must not be modified")

    rows_value = artifact.get("row_results")
    if not isinstance(rows_value, list):
        errors.append("row_results must be a list")
        rows: list[JsonDict] = []
    else:
        rows = [dict(row) for row in rows_value if isinstance(row, Mapping)]
    errors.extend(_row_integrity_errors(rows))
    metrics = derive_metrics(rows)
    budget = derive_reward_budget(rows)
    errors.extend(_aggregate_errors(artifact, metrics, budget, rows))
    errors.extend(_row_path_errors(artifact, rows))

    ready = artifact.get("verifier_guided_decoding_ready") is True
    if ready:
        if artifact.get("status") != "complete":
            errors.append("verifier_guided_decoding_ready requires complete status")
        if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
            errors.append("complete artifact requires inference_substrate=live_llm_inference")
        if artifact.get("gpu_offload_verified") is not True:
            errors.append("verifier_guided_decoding_ready requires gpu_offload_verified")
        if not any(row.get("ran_headline") is True for row in model_specs):
            errors.append("headline_required_any_of requires at least one mandated model ran")
        if not rows:
            errors.append("verifier_guided_decoding_ready requires row evidence")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum must match artifact payload")
    return errors


class LlamaCppGenerationRunner:  # pragma: no cover
    """Small llama-cpp-python wrapper that loads one GGUF and generates bounded rows."""

    def __init__(
        self,
        *,
        model_spec: Mapping[str, Any],
        n_gpu_layers: int = N_GPU_LAYERS,
        seed: int = RANDOM_SEED,
    ) -> None:
        from llama_cpp import Llama  # noqa: PLC0415

        self.model_spec = dict(model_spec)
        self.n_gpu_layers = n_gpu_layers
        before = _gpu_memory_snapshot()
        started = time.perf_counter()
        self.llm = Llama(
            model_path=str(model_spec["model_path"]),
            n_gpu_layers=n_gpu_layers,
            n_ctx=2048,
            n_batch=256,
            seed=seed,
            verbose=False,
        )
        duration_s = time.perf_counter() - started
        after = _gpu_memory_snapshot()
        delta = _max_memory_delta_mb(before, after)
        self.load_receipt = {
            "load_duration_s": duration_s,
            "gpu_memory_before": before,
            "gpu_memory_after": after,
            "gpu_memory_delta_mb": delta,
            "offload_evidence": delta > 512,
            "n_gpu_layers": n_gpu_layers,
        }

    def __call__(self, **kwargs: Any) -> JsonDict:
        from llama_cpp import LlamaGrammar  # noqa: PLC0415

        grammar_text = kwargs.get("grammar")
        grammar = LlamaGrammar.from_string(str(grammar_text), verbose=False) if grammar_text else None
        started = time.perf_counter()
        result = self.llm.create_completion(
            prompt=str(kwargs["prompt"]),
            max_tokens=int(kwargs["max_tokens"]),
            temperature=0.0 if kwargs["condition"] != "unconstrained" else 0.2,
            top_p=0.95,
            seed=int(kwargs["seed"]),
            echo=False,
            stop=["\n\n"],
            grammar=grammar,
        )
        duration_s = time.perf_counter() - started
        choices = result.get("choices", []) if isinstance(result, Mapping) else []
        text = str(choices[0].get("text", "")) if choices else ""
        usage = result.get("usage", {}) if isinstance(result, Mapping) else {}
        return {
            "output_text": text,
            "duration_s": duration_s,
            "generated_token_count": int(usage.get("completion_tokens", 0) or 0),
            "backend_details": {
                "llama_cpp_create_completion": True,
                "grammar_used": grammar_text is not None,
            },
        }


def row_checksum(row: Mapping[str, Any]) -> str:
    """Hash one row while excluding the checksum field itself."""

    payload = {key: value for key, value in row.items() if key != "row_checksum"}
    return _sha256_json(payload)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact payload without the self-referential checksum."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return _sha256_json(payload)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """CLI entry point for producing the Exp5444 result artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifact-path", type=Path, default=None)
    parser.add_argument("--row-results-path", type=Path, default=None)
    parser.add_argument("--max-fixtures", type=int, default=DEFAULT_MAX_FIXTURES)
    args = parser.parse_args(argv)
    artifact = run(
        root=args.root,
        artifact_path=args.artifact_path,
        row_results_path=args.row_results_path,
        max_fixtures=args.max_fixtures,
        write=True,
    )
    print(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if artifact["verifier_guided_decoding_ready"] else 1


def _condition_metric(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    total = len(rows)
    accepted = [row for row in rows if row.get("accepted_by_final_authority") is True]
    abstained = [row for row in rows if row.get("parse_status") == "abstained"]
    return {
        "count": total,
        "accepted_count": len(accepted),
        "accepted_validity_rate": _rate(accepted, total),
        "abstention_rate": _rate(abstained, total),
    }


def _aggregate_errors(
    artifact: Mapping[str, Any],
    metrics: Mapping[str, Any],
    budget: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    scalar_fields = (
        "guided_validity_delta_vs_unconstrained",
        "guided_validity_delta_vs_grammar_only",
        "semantic_false_accept_rate",
        "unsafe_false_accept_rate",
        "action_unreachability_rate",
        "abstention_rate",
        "accepted_validity_rate",
    )
    for field in scalar_fields:
        if not _float_close(artifact.get(field), metrics.get(field)):
            errors.append(f"{field} must match row recomputation")
    if artifact.get("condition_metrics") != metrics.get("condition_metrics"):
        errors.append("condition_metrics must match row recomputation")
    if artifact.get("metric_details") != metrics:
        errors.append("metric_details must match row recomputation")
    if artifact.get("metric_independence_checks_passed") != metrics.get(
        "metric_independence_checks_passed"
    ):
        errors.append("metric_independence_checks_passed must match row recomputation")
    if artifact.get("constraint_family_counts") != metrics.get("constraint_family_counts"):
        errors.append("constraint_family_counts must match row recomputation")
    if artifact.get("reward_evaluation_budget") != budget:
        errors.append("reward_evaluation_budget must match row recomputation")
    if artifact.get("row_checksums") != [row.get("row_checksum") for row in rows]:
        errors.append("row_checksums must match row_results")
    if artifact.get("fixture_count") != len({row.get("fixture_id") for row in rows}):
        errors.append("fixture_count must match distinct row fixtures")
    return errors


def _row_integrity_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        verdict = _mapping(row.get("exact_final_verdict"))
        if verdict.get("verified") is not True or verdict.get("authority") != "exact_final_verifier":
            errors.append("row exact final authority must be exact_final_verifier")
        if row.get("final_authority_bypassed") is not False:
            errors.append("row exact final authority must not be bypassed")
        if row.get("row_checksum") != row_checksum(row):
            errors.append("row checksum must match row payload")
        if str(row.get("model_hf_id")) not in MANDATED_HF_IDS:
            errors.append("row model_hf_id must be mandated")
        if not str(row.get("model_path", "")).endswith(".gguf"):
            errors.append("row model_path must be a GGUF path")
    return errors


def _row_path_errors(artifact: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    path_text = artifact.get("row_results_path")
    if not isinstance(path_text, str) or not path_text:
        return ["row_results_path must be a non-empty string"]
    path = Path(path_text)
    if artifact.get("verifier_guided_decoding_ready") is True:
        if not path.is_file():
            errors.append("row_results_path must point to written row evidence")
            return errors
        try:
            disk_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"row_results_path is unreadable: {type(exc).__name__}: {exc}")
            return errors
        if disk_rows != list(rows):
            errors.append("row_results_path contents must match embedded row_results")
    return errors


def _mark_model_ran(
    model_specs: Sequence[Mapping[str, Any]],
    selected_model: Mapping[str, Any] | None,
) -> list[JsonDict]:
    selected_hf_id = str(selected_model.get("hf_id")) if selected_model else ""
    rows: list[JsonDict] = []
    for spec in model_specs:
        row = dict(spec)
        if str(row.get("hf_id")) == selected_hf_id:
            row["ran_headline"] = True
            row["runtime_backend"] = "llama_cpp_python_cuda_gguf"
            row["n_gpu_layers"] = N_GPU_LAYERS
            row["gpu_offload_verified"] = True
        rows.append(row)
    return rows


def _family_instruction(fixture: Mapping[str, Any], family: str) -> str:
    if family == "schema_only_trap":
        return "Use only the allowed keys. Do not add admin_override or any extra field."
    if family == "semantic_contradiction":
        return "If you include negated_object, it must differ from object."
    if family == "unreachable_tool_action":
        return "The tool must be allowed, order_state must be paid, and lock_active must be false."
    if family == "arithmetic_finite_domain":
        return "All numeric fields must be integers in range and sum must equal x + y."
    if family == "ontology_triple_update":
        return "The triple must match the ontology shape and must not create a cycle."
    if family == "api_call_witness":
        api = _mapping(fixture.get("api"))
        return f"Use an allowed API call and witness signature {api.get('expected_signature')}."
    if family == "benign":
        return "The evidence field must support claim_key and claim_value exactly."
    return "Satisfy the deterministic row constraints."


def _best_guided_prefix(fixture: Mapping[str, Any]) -> JsonDict | None:
    prefixes = [
        dict(prefix)
        for prefix in fixture.get("prefixes", [])
        if isinstance(prefix, Mapping) and prefix.get("accepted_by_potential") is True
    ]
    if not prefixes:
        return None
    return max(
        prefixes,
        key=lambda prefix: (
            len(prefix.get("accepted_functions", [])),
            -int(prefix.get("reward_evaluation_cost_units", 0)),
        ),
    )


def _reward_evaluation_count(condition: str, prefix_seed: Mapping[str, Any] | None) -> int:
    final_verifier_call = 1
    if condition != GUIDED_CONDITION or prefix_seed is None:
        return final_verifier_call
    return final_verifier_call + len(prefix_seed.get("potential_evaluations", []))


def _advisory_false_accept(row: Mapping[str, Any]) -> bool:
    return row.get("condition_advisory_accept") is True and row.get(
        "accepted_by_final_authority"
    ) is False


def _exact_authority_row(row: Mapping[str, Any]) -> bool:
    verdict = _mapping(row.get("exact_final_verdict"))
    return (
        verdict.get("verified") is True
        and verdict.get("authority") == "exact_final_verifier"
        and row.get("final_authority_bypassed") is False
    )


def _honest_verdict(
    ready: bool,
    preconditions: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> str:
    if ready:
        return "complete: local SOTA verifier-potential guided decoding pilot ready"
    blockers = list(preconditions.get("blocked_preconditions", []))
    if blockers:
        return "blocked: " + ",".join(str(item) for item in blockers)
    if metrics.get("metric_independence_checks_passed") is not True:
        return "blocked: metric independence checks failed"
    return "blocked: verifier-guided decoding readiness checks failed"


def _file_receipt(path: Path) -> JsonDict:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "suffix": path.suffix,
        "name": path.name,
        "head_64k_sha256": _hash_file_head(path, 64 * 1024),
    }


def _hash_file_head(path: Path, limit: int) -> str:
    with path.open("rb") as handle:
        return hashlib.sha256(handle.read(limit)).hexdigest()


def _nvidia_smi_query() -> JsonDict:  # pragma: no cover
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - environment-dependent.
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": result.returncode == 0,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "returncode": result.returncode,
    }


def _gpu_memory_snapshot() -> list[JsonDict]:  # pragma: no cover
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:  # pragma: no cover - environment-dependent.
        return []
    if result.returncode != 0:
        return []
    rows: list[JsonDict] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            rows.append({"index": int(parts[0]), "memory_used_mb": int(float(parts[1]))})
    return rows


def _max_memory_delta_mb(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> int:
    before_by_index = {int(row["index"]): int(row["memory_used_mb"]) for row in before}
    deltas = [
        int(row["memory_used_mb"]) - before_by_index.get(int(row["index"]), 0) for row in after
    ]
    return max(deltas) if deltas else 0


def _read_json(path: Path) -> JsonDict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _normalise_tests_run(value: Sequence[str | Mapping[str, Any]]) -> list[JsonDict]:
    rows = [_normalise_test_run(row) for row in value]
    return rows or [
        {
            "command": (
                ".venv/bin/pytest tests/python/"
                "test_experiment_5444_gated_sota_energy_guided_decoding_v495.py -q"
            ),
            "outcome": "not_recorded",
        }
    ]


def _normalise_test_run(value: str | Mapping[str, Any]) -> JsonDict:
    if isinstance(value, Mapping):
        return {
            "command": str(value.get("command", "")),
            "outcome": str(value.get("outcome", "not_recorded")),
        }
    return {"command": str(value), "outcome": "not_recorded"}


def _destination(root: Path, path: Path | str | None, default: Path) -> Path:
    if path is None:
        return root / default
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _bare_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _rate(items: Sequence[Any], denominator: int) -> float:
    return float(len(items) / denominator) if denominator else 0.0


def _float_close(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
