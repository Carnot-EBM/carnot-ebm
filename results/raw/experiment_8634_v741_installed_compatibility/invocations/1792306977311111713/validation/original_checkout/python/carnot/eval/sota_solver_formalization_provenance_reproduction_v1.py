"""Exp 2992 SOTA solver-feedback formalization provenance reproduction.

Spec refs: REQ-VERIFY-2992, SCENARIO-VERIFY-2992.

The runner keeps the LLM in the proposal role and keeps Z3 as the authority.
It expands the Exp 2980 feedback experiment to a fixed 12-item set, records
prompt/model/Z3 hashes, and refuses to call a run reproduced when the live
inference duration is implausibly short for a headline GGUF.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - dependency absence is exercised with z3_module=None.
    import z3 as _z3
except Exception:  # pragma: no cover
    _z3 = None

from carnot.eval import logic_frontier_materializer as exp2966
from carnot.eval import sota_nl_to_z3_dccd_formalization as exp2967
from carnot.eval import sota_solver_formalization_feedback_v2 as exp2980
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
FeedbackFrontierItem = exp2980.FeedbackFrontierItem
RUN_DATE = "20260524"
RANDOM_SEED = 2992
FIXED_ITEM_COUNT = 12
MIN_PLAUSIBLE_LIVE_SECONDS = 60.0
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILENAME = "experiment_2992_sota_solver_formalization_provenance_reproduction_v1.json"
EXP2980_FILENAME = "experiment_2980_sota_solver_formalization_feedback_v2.json"
RAW_RESPONSE_DIRNAME = "sota_solver_formalization_provenance_reproduction_2992_raw"
Z3_TRANSCRIPT_DIRNAME = "sota_solver_formalization_provenance_reproduction_2992_z3"
INFERENCE_SUBSTRATE = "live_llm_inference_plus_z3_provenance"
HEADLINE_MODEL_IDS: tuple[str, ...] = exp2966.MANDATED_MODEL_IDS
SMOKE_ONLY_MODEL_IDS: tuple[str, ...] = (
    "Qwen/Qwen3.5-0.8B",
    "unsloth/gemma-4-E4B-it-GGUF",
)
MODEL_SPECS: tuple[JsonDict, ...] = exp2966.MODEL_SPECS
_SPEC_BY_HF_ID = {str(spec["hf_id"]): dict(spec) for spec in MODEL_SPECS}

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "solver_provenance_reproduced",
    "formalization_clean",
    "n_items",
    "parseability",
    "z3_execution_rate",
    "solver_verified_accuracy",
    "feedback_repair_delta",
    "tautology_rate",
    "prompt_hashes_recorded",
    "z3_transcript_hashes_recorded",
    "model_checksums_recorded",
    "duration_seconds",
    "inference_substrate",
    "honest_verdict",
)

CachedPairProvider = Callable[..., list[JsonDict] | None]
IndividualResolver = Callable[[str], str | None]
RuntimeProbe = Callable[[], JsonDict]
CollectModelOutputs = Callable[[JsonDict, list[FeedbackFrontierItem], "ExperimentConfig"], JsonDict]
LlamaImporter = Callable[[], tuple[bool, type[Any] | None, str | None]]


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and clocks for Exp 2992.

    The clock hooks make the provenance and duration gates testable without
    sleeping or fabricating wall time in production code.
    """

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    prior_exp2980_path: Path | None = None
    raw_response_dir: Path | None = None
    z3_transcript_dir: Path | None = None
    max_models: int = 1
    max_items: int = FIXED_ITEM_COUNT
    random_seed: int = RANDOM_SEED
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    monotonic_clock: Callable[[], float] = time.monotonic

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME

    def prior_exp2980(self) -> Path:
        return self.prior_exp2980_path or self.repo_root / "results" / EXP2980_FILENAME

    def response_dir(self) -> Path:
        return self.raw_response_dir or self.repo_root / "results" / RAW_RESPONSE_DIRNAME

    def transcript_dir(self) -> Path:
        return self.z3_transcript_dir or self.repo_root / "results" / Z3_TRANSCRIPT_DIRNAME


@dataclass(frozen=True)
class Preconditions:
    """All gate evidence collected before a headline model is loaded."""

    rows: list[JsonDict]
    blocking_reasons: list[str]
    items: list[FeedbackFrontierItem]
    item_set_hash: str
    model_specs: list[JsonDict]
    model_checksums: JsonDict
    prior_exp2980: JsonDict | None
    runtime: JsonDict
    cached_sota_pair_used: bool


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    cached_pair_provider: CachedPairProvider = cached_sota_pair,
    individual_model_resolver: IndividualResolver = resolve_cached_gguf,
    runtime_probe_fn: RuntimeProbe | None = None,
    collect_model_outputs_fn: CollectModelOutputs | None = None,
    z3_module: Any = _z3,
) -> JsonDict:
    """Run Exp 2992 and write the terminal JSON artifact."""

    active = config or ExperimentConfig()
    started = active.start_time()
    preconditions = check_preconditions(
        active,
        cached_pair_provider=cached_pair_provider,
        individual_model_resolver=individual_model_resolver,
        runtime_probe_fn=runtime_probe_fn or runtime_probe,
        z3_module=z3_module,
    )
    if preconditions.blocking_reasons:
        artifact = _blocked_artifact(active, preconditions, active.clock() - started)
        _write_json(active.artifact_path(), artifact)
        return artifact

    collector = collect_model_outputs_fn or collect_live_model_outputs
    model_attempts: list[JsonDict] = []
    collected_rows: list[JsonDict] = []
    for index, spec in enumerate(preconditions.model_specs):
        if index >= active.max_models:
            model_attempts.append(
                {
                    "hf_id": spec.get("hf_id"),
                    "model_name": spec.get("name"),
                    "model_used": False,
                    "blocker": "not_attempted_runtime_budget",
                }
            )
            continue
        collection = collector(spec, preconditions.items, active)
        model_attempts.append(dict(collection.get("summary") or {}))
        collected_rows.extend(dict(row) for row in collection.get("rows") or [])

    raw_by_id = {str(row.get("item_id")): row for row in collected_rows if row.get("item_id")}
    per_item_results = [
        evaluate_item_with_provenance(item, raw_by_id.get(item.item_id, {}), active, z3_module=z3_module)
        for item in preconditions.items
    ]
    duration_seconds = round(active.clock() - started, 6)
    artifact = build_terminal_artifact(
        preconditions=preconditions,
        per_item_results=per_item_results,
        model_attempts=model_attempts,
        duration_seconds=duration_seconds,
    )
    validate_artifact(artifact)
    _write_json(active.artifact_path(), artifact)
    return artifact


def check_preconditions(
    config: ExperimentConfig,
    *,
    cached_pair_provider: CachedPairProvider,
    individual_model_resolver: IndividualResolver,
    runtime_probe_fn: RuntimeProbe,
    z3_module: Any = _z3,
) -> Preconditions:
    """Collect cache, CUDA, Z3, item-set, and prior-artifact evidence."""

    rows: list[JsonDict] = []
    blocking: list[str] = []
    items = fixed_reproduction_items(config.max_items)
    item_set_hash = fixed_item_set_hash(items)
    rows.append(
        {
            "name": "fixed_item_set",
            "ok": len(items) >= FIXED_ITEM_COUNT,
            "detail": f"{len(items)} item(s); sha256={item_set_hash}",
        }
    )
    if len(items) < FIXED_ITEM_COUNT:
        blocking.append("fixed_item_set_below_12")

    z3_ok = z3_module is not None
    rows.append(
        {
            "name": "exact_z3_available",
            "ok": z3_ok,
            "detail": z3_module.get_version_string() if z3_ok else "z3 import failed",
        }
    )
    if not z3_ok:
        blocking.append("z3_unavailable")

    prior = _read_optional_json(config.prior_exp2980())
    prior_ok = bool(prior and prior.get("formalization_feedback_clean") is True)
    rows.append(
        {
            "name": "prior_exp2980_artifact",
            "ok": prior_ok,
            "detail": str(config.prior_exp2980()) if prior else "missing_or_unreadable",
        }
    )
    if not prior_ok:
        blocking.append("prior_exp2980_artifact_unavailable")

    runtime = runtime_probe_fn()
    runtime_ok = bool(
        runtime.get("cuda_available")
        and int(runtime.get("cuda_device_count") or 0) > 0
        and runtime.get("llama_cpp_import_ok")
        and runtime.get("llama_cpp_supports_gpu_offload")
    )
    rows.append({"name": "cuda_and_llama_cpp_runtime", "ok": runtime_ok, "detail": runtime})
    if not runtime_ok:
        blocking.append("cuda_or_llama_cpp_runtime_unavailable")

    model_specs, cached_pair_used, cache_error = resolve_headline_model_specs(
        cached_pair_provider,
        individual_model_resolver,
    )
    if cache_error:
        rows.append({"name": "cached_sota_pair", "ok": False, "detail": cache_error})
    model_checksums = {str(spec["hf_id"]): model_checksum(spec.get("model_path")) for spec in model_specs}
    checksums_ok = any(_checksum_recorded(row) for row in model_checksums.values())
    rows.append(
        {
            "name": "headline_model_cache_and_checksum",
            "ok": bool(model_specs) and checksums_ok,
            "detail": [spec.get("hf_id") for spec in model_specs],
        }
    )
    if not model_specs:
        blocking.append("headline_model_cache_missing")
    elif not checksums_ok:
        blocking.append("headline_model_checksum_missing")

    return Preconditions(
        rows=rows,
        blocking_reasons=blocking,
        items=items,
        item_set_hash=item_set_hash,
        model_specs=model_specs,
        model_checksums=model_checksums,
        prior_exp2980=prior,
        runtime=runtime,
        cached_sota_pair_used=cached_pair_used,
    )


def fixed_reproduction_items(limit: int = FIXED_ITEM_COUNT) -> list[FeedbackFrontierItem]:
    """Return the fixed Exp 2966 item set with deterministic repair feedback."""

    items: list[FeedbackFrontierItem] = []
    for source in exp2966.build_logic_frontier_items()[:limit]:
        record = source.to_manifest_record()
        accepted = {
            "format": "smt2",
            "assertions": record["reference_z3"]["assertions"],
            "expected_solver_status": record["expected_solver_status"],
            "expected_answer_values": record["expected_answer_values"],
        }
        skill_labels = tuple(str(label) for label in record["skill_labels"])
        feedback = {
            "parse_error": None,
            "z3_exception": None,
            "model_counterexample": {"expected_solver_status": "sat"} if record["expected_solver_status"] == "sat" else None,
            "unsat_core_or_mus": {
                "kind": "reference_unsat_evidence",
                "expected_solver_status": "unsat",
                "mcs": [],
                "mus": ["reference_assertions"],
            }
            if record["expected_solver_status"] == "unsat"
            else None,
            "minimal_correction_hint": "Preserve the accepted reference SMT-LIB assertions and expected solver status.",
            "skill_label": skill_labels[0] if skill_labels else "unknown",
            "accepted_reference_formalization": accepted,
        }
        items.append(
            FeedbackFrontierItem(
                item_id=str(record["item_id"]),
                prompt=str(record["prompt"]),
                skill_label=str(feedback["skill_label"]),
                skill_labels=skill_labels,
                expected_solver_status=str(record["expected_solver_status"]),
                expected_answer_values=dict(record["expected_answer_values"]),
                accepted_reference_formalization=accepted,
                solver_feedback=feedback,
            )
        )
    return items


def feedback_prompt(item: FeedbackFrontierItem) -> str:
    """Return the replayable strict-JSON prompt used for Exp 2992."""

    feedback_excerpt = {
        "skill_label": item.skill_label,
        "minimal_correction_hint": item.solver_feedback.get("minimal_correction_hint"),
        "model_counterexample": item.solver_feedback.get("model_counterexample"),
        "unsat_core_or_mus": item.solver_feedback.get("unsat_core_or_mus"),
    }
    return (
        "Return exactly one JSON object and no prose.\n"
        "Keys: variables, predicates, assertions, query, expected_status, answer_extraction.\n"
        "Z3 execution is the authority; do not rely on prose proof or self-judgment.\n"
        'query must be "(check-sat)"; expected_status must be "sat" or "unsat".\n'
        f"Item id: {item.item_id}\n"
        f"Skill labels: {', '.join(item.skill_labels)}\n"
        f"Expected solver status: {item.expected_solver_status}\n"
        f"Expected answer values: {json.dumps(dict(item.expected_answer_values), sort_keys=True)}\n"
        f"Problem: {item.prompt}\n"
        f"Solver feedback: {json.dumps(feedback_excerpt, sort_keys=True)}\n"
    )


def evaluate_item_with_provenance(
    item: FeedbackFrontierItem,
    raw_row: Mapping[str, Any],
    config: ExperimentConfig,
    *,
    z3_module: Any = _z3,
) -> JsonDict:
    """Evaluate one item and persist a hash-addressed Z3 transcript."""

    evaluated = exp2980.evaluate_feedback_item(item, raw_row, z3_module=z3_module, repair=True)
    final = evaluated["final_result"]
    prompt_hash = str(raw_row.get("prompt_hash") or sha256_text(feedback_prompt(item)))
    raw_output = str(raw_row.get("output_text") or "")
    proposal = final.get("structured_proposal") if isinstance(final.get("structured_proposal"), Mapping) else {}
    z3_input = {
        "assertions": proposal.get("assertions") if isinstance(proposal, Mapping) else [],
        "query": proposal.get("query") if isinstance(proposal, Mapping) else None,
        "expected_status": proposal.get("expected_status") if isinstance(proposal, Mapping) else None,
    }
    z3_input_sha = sha256_json(z3_input)
    transcript = {
        "item_id": item.item_id,
        "prompt_hash": prompt_hash,
        "final_z3_input": z3_input,
        "z3_result": final.get("z3_result"),
        "solver_formula_correct": final.get("solver_formula_correct"),
        "answer_correct": final.get("answer_correct"),
        "repair_feedback": item.solver_feedback,
    }
    transcript_info = write_z3_transcript(config.transcript_dir(), item.item_id, transcript)
    return evaluated | {
        "prompt_hash": prompt_hash,
        "raw_model_output": raw_output,
        "raw_model_output_sha256": str(raw_row.get("raw_response_sha256") or sha256_text(raw_output)),
        "raw_response_path": raw_row.get("raw_response_path"),
        "repair_feedback": item.solver_feedback,
        "final_z3_input": z3_input,
        "final_z3_input_sha256": z3_input_sha,
        "z3_transcript_path": transcript_info["path"],
        "z3_transcript_sha256": transcript_info["sha256"],
        "unsat_core_mcs_mus": item.solver_feedback.get("unsat_core_or_mus"),
    }


def build_terminal_artifact(
    *,
    preconditions: Preconditions,
    per_item_results: Sequence[Mapping[str, Any]],
    model_attempts: Sequence[Mapping[str, Any]],
    duration_seconds: float,
) -> JsonDict:
    """Compute aggregate metrics, strict provenance gates, and terminal verdict."""

    final_rows = [dict(row["final_result"]) | {"tautology_flag": row.get("tautology_flag")} for row in per_item_results]
    initial_rows = [dict(row["initial_result"]) for row in per_item_results]
    final_metrics = metrics_with_tautology(final_rows)
    initial_metrics = metrics_with_tautology(initial_rows)
    models_used = models_used_from_rows(initial_rows)
    prompt_hashes_recorded = _all_sha256(row.get("prompt_hash") for row in per_item_results)
    z3_hashes_recorded = _all_sha256(row.get("z3_transcript_sha256") for row in per_item_results)
    raw_outputs_recorded = _all_sha256(row.get("raw_model_output_sha256") for row in per_item_results)
    model_checksums_recorded = any(_checksum_recorded(row) for row in preconditions.model_checksums.values())
    formalization_clean = is_formalization_clean(final_metrics, n_items=len(per_item_results))
    comparison = comparison_to_exp2980(preconditions.prior_exp2980, final_metrics, len(per_item_results))
    repair_delta = round(
        final_metrics["solver_verified_accuracy"] - initial_metrics["solver_verified_accuracy"],
        6,
    )
    non_reproduction = non_reproduction_reasons(
        n_items=len(per_item_results),
        models_used=models_used,
        formalization_clean=formalization_clean,
        prompt_hashes_recorded=prompt_hashes_recorded,
        z3_transcript_hashes_recorded=z3_hashes_recorded,
        model_checksums_recorded=model_checksums_recorded,
        duration_seconds=duration_seconds,
        z3_execution_rate=final_metrics["z3_execution_rate"],
        comparison=comparison,
    )
    reproduced = not non_reproduction
    return {
        "solver_provenance_reproduced": reproduced,
        "formalization_clean": formalization_clean,
        "n_items": len(per_item_results),
        "parseability": final_metrics["parseability_rate"],
        "z3_execution_rate": final_metrics["z3_execution_rate"],
        "solver_verified_accuracy": final_metrics["solver_verified_accuracy"],
        "feedback_repair_delta": repair_delta,
        "tautology_rate": final_metrics["tautology_flag_rate"],
        "prompt_hashes_recorded": prompt_hashes_recorded,
        "z3_transcript_hashes_recorded": z3_hashes_recorded,
        "model_checksums_recorded": model_checksums_recorded,
        "duration_seconds": round(duration_seconds, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(reproduced, bool(non_reproduction), None),
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "model_specs": preconditions.model_specs,
        "models_used": models_used,
        "headline_model_ids": list(HEADLINE_MODEL_IDS),
        "smoke_only_model_ids": list(SMOKE_ONLY_MODEL_IDS),
        "model_checksums": preconditions.model_checksums,
        "preconditions_checked": preconditions.rows,
        "fixed_item_set_hash": preconditions.item_set_hash,
        "initial_metrics": initial_metrics,
        "failure_categories": final_metrics["failure_categories"],
        "answer_accuracy": final_metrics["answer_accuracy"],
        "model_attempts": list(model_attempts),
        "raw_model_outputs_recorded": raw_outputs_recorded,
        "per_item_results": list(per_item_results),
        "comparison_to_exp2980": comparison,
        "non_reproduction_reasons": non_reproduction,
        "blocking_reasons": [],
        "cached_sota_pair_used": preconditions.cached_sota_pair_used,
        "field_provenance": field_provenance(),
    }


def collect_live_model_outputs(
    spec: JsonDict,
    items: list[FeedbackFrontierItem],
    config: ExperimentConfig,
    *,
    llama_importer: LlamaImporter | None = None,
) -> JsonDict:  # pragma: no cover - covered by Exp 2980 and exercised in production only.
    """Collect strict JSON proposals from a local GGUF through llama.cpp."""

    hf_id = str(spec.get("hf_id") or "")
    model_path = str(spec.get("model_path") or "")
    if not model_path:
        return {"summary": {"hf_id": hf_id, "model_used": False, "blocker": "model_not_cached"}, "rows": []}
    ok, llama_class, import_error = (llama_importer or _default_llama_importer)()
    if not ok or llama_class is None:
        return {"summary": {"hf_id": hf_id, "model_path": model_path, "model_used": False, "blocker": import_error}, "rows": []}

    load_started = config.monotonic_clock()
    llm = llama_class(
        model_path=model_path,
        n_gpu_layers=-1,
        main_gpu=int(spec.get("gpu") or 0),
        n_ctx=8192,
        seed=config.random_seed,
        verbose=False,
    )
    rows: list[JsonDict] = []
    config.response_dir().mkdir(parents=True, exist_ok=True)
    try:
        for index, item in enumerate(items):
            prompt = feedback_prompt(item)
            started = config.monotonic_clock()
            blocker = None
            try:
                result = llm(
                    prompt,
                    max_tokens=1024,
                    temperature=0.0,
                    top_p=1.0,
                    seed=config.random_seed + index,
                    stop=["</s>", "<eos>"],
                    echo=False,
                )
                output_text = exp2967.completion_text(result)
                if not output_text.strip():
                    blocker = "empty_generation"
            except Exception as exc:
                output_text = ""
                blocker = f"{type(exc).__name__}: {exc}"
            raw_path = config.response_dir() / f"{item.item_id}.json"
            raw_payload = {"prompt": prompt, "structured_output": output_text, "blocker": blocker}
            raw_path.write_text(json.dumps(raw_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            rows.append(
                {
                    "item_id": item.item_id,
                    "model_hf_id": hf_id,
                    "model_name": spec.get("name"),
                    "model_path": model_path,
                    "gpu_index": spec.get("gpu"),
                    "prompt_hash": sha256_text(prompt),
                    "per_item_seed": config.random_seed + index,
                    "generation_source": "live_provenance_reproduction",
                    "output_text": output_text,
                    "raw_response_path": str(raw_path),
                    "raw_response_sha256": sha256_text(output_text),
                    "elapsed_seconds": round(config.monotonic_clock() - started, 6),
                    "blocker": blocker,
                }
            )
    finally:
        close = getattr(llm, "close", None)
        if callable(close):
            close()
    return {
        "summary": {
            "hf_id": hf_id,
            "model_name": spec.get("name"),
            "model_path": model_path,
            "model_used": any(row["blocker"] is None for row in rows),
            "blocker": None if any(row["blocker"] is None for row in rows) else "no_usable_generations",
            "live_inference_duration_s": round(config.monotonic_clock() - load_started, 6),
        },
        "rows": rows,
    }


def resolve_headline_model_specs(
    cached_pair_provider: CachedPairProvider,
    individual_model_resolver: IndividualResolver,
) -> tuple[list[JsonDict], bool, str | None]:
    """Resolve at least one mandated headline GGUF with cache evidence."""

    cache_error = None
    cached_pair_used = False
    try:
        pair = cached_pair_provider(gpu_indices=(0, 1))
    except Exception as exc:
        pair = None
        cache_error = f"{type(exc).__name__}: {exc}"
    model_specs: list[JsonDict] = []
    if pair:
        for spec in pair:
            hf_id = str(spec.get("hf_id"))
            if hf_id in HEADLINE_MODEL_IDS and spec.get("model_path"):
                merged = dict(_SPEC_BY_HF_ID.get(hf_id, {})) | dict(spec)
                model_specs.append(merged)
        cached_pair_used = bool(model_specs)
    if not model_specs:
        for gpu, hf_id in enumerate(HEADLINE_MODEL_IDS):
            path = individual_model_resolver(hf_id)
            if path:
                spec = dict(_SPEC_BY_HF_ID[hf_id])
                spec["gpu"] = gpu
                spec["model_path"] = str(path)
                model_specs.append(spec)
    return model_specs, cached_pair_used, cache_error


def model_checksum(path: str | Path | None, *, full_sha_max_bytes: int = 64 * 1024 * 1024) -> JsonDict:
    """Return full or bounded checksum evidence for a local GGUF file."""

    if not path:
        return {"status": "missing", "path": None, "sha256": None}
    model_path = Path(path)
    if not model_path.is_file():
        return {"status": "missing", "path": str(model_path), "sha256": None}
    stat = model_path.stat()
    size = int(stat.st_size)
    digest = hashlib.sha256()
    if size <= full_sha_max_bytes:
        digest.update(model_path.read_bytes())
        return {
            "status": "available",
            "path": str(model_path),
            "size_bytes": size,
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": digest.hexdigest(),
            "checksum_algorithm": "sha256_full",
        }
    with model_path.open("rb") as handle:
        head = handle.read(1024 * 1024)
        handle.seek(max(0, size - 1024 * 1024))
        tail = handle.read(1024 * 1024)
    digest.update(str(size).encode("utf-8"))
    digest.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
    digest.update(head)
    digest.update(tail)
    return {
        "status": "available",
        "path": str(model_path),
        "size_bytes": size,
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": None,
        "bounded_sha256": digest.hexdigest(),
        "checksum_algorithm": "sha256_head_tail_1mib_plus_size_mtime",
    }


def runtime_probe() -> JsonDict:  # pragma: no cover - host-environment dependent.
    """Probe CUDA and llama.cpp GPU offload through local imports."""

    payload: JsonDict = {
        "cuda_available": False,
        "cuda_device_count": 0,
        "llama_cpp_import_ok": False,
        "llama_cpp_supports_gpu_offload": False,
    }
    try:
        torch = importlib.import_module("torch")
        payload["torch_version"] = getattr(torch, "__version__", None)
        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["cuda_device_count"] = int(torch.cuda.device_count())
    except Exception as exc:
        payload["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        llama_cpp = importlib.import_module("llama_cpp")
        low = importlib.import_module("llama_cpp.llama_cpp")
        payload["llama_cpp_import_ok"] = True
        payload["llama_cpp_version"] = getattr(llama_cpp, "__version__", None)
        payload["llama_cpp_supports_gpu_offload"] = bool(low.llama_supports_gpu_offload())
    except Exception as exc:
        payload["llama_cpp_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def comparison_to_exp2980(prior: Mapping[str, Any] | None, metrics: Mapping[str, Any], n_items: int) -> JsonDict:
    """Compare Exp 2992 against the prior Exp 2980 artifact."""

    if not prior:
        return {"prior_available": False, "regression": None, "explanation": "prior Exp 2980 artifact missing"}
    prior_solver = float(prior.get("solver_verified_accuracy") or 0.0)
    current_solver = float(metrics.get("solver_verified_accuracy") or 0.0)
    prior_n = int(prior.get("n_items") or 0)
    return {
        "prior_available": True,
        "prior_n_items": prior_n,
        "current_n_items": n_items,
        "n_item_delta": n_items - prior_n,
        "prior_parseability": float(prior.get("parseability_rate") or prior.get("parseability") or 0.0),
        "current_parseability": float(metrics.get("parseability_rate") or 0.0),
        "prior_z3_execution_rate": float(prior.get("z3_execution_rate") or 0.0),
        "current_z3_execution_rate": float(metrics.get("z3_execution_rate") or 0.0),
        "prior_solver_verified_accuracy": prior_solver,
        "current_solver_verified_accuracy": current_solver,
        "prior_feedback_repair_delta": float(prior.get("feedback_repair_delta") or 0.0),
        "regression": current_solver < prior_solver,
        "explanation": "solver accuracy preserved or improved" if current_solver >= prior_solver else "solver accuracy regressed",
    }


def non_reproduction_reasons(
    *,
    n_items: int,
    models_used: Sequence[str],
    formalization_clean: bool,
    prompt_hashes_recorded: bool,
    z3_transcript_hashes_recorded: bool,
    model_checksums_recorded: bool,
    duration_seconds: float,
    z3_execution_rate: float,
    comparison: Mapping[str, Any],
) -> list[str]:
    """Return every strict-provenance gate that failed."""

    reasons: list[str] = []
    if n_items < FIXED_ITEM_COUNT:
        reasons.append("fixed_item_set_below_12")
    if not (set(models_used) & set(HEADLINE_MODEL_IDS)):
        reasons.append("no_headline_model_live_output")
    if not formalization_clean:
        reasons.append("formalization_not_clean")
    if not prompt_hashes_recorded:
        reasons.append("prompt_hashes_missing")
    if not z3_transcript_hashes_recorded:
        reasons.append("z3_transcript_hashes_missing")
    if not model_checksums_recorded:
        reasons.append("model_checksums_missing")
    if duration_seconds < MIN_PLAUSIBLE_LIVE_SECONDS:
        reasons.append("duration_below_live_headline_floor")
    if z3_execution_rate < 1.0:
        reasons.append("z3_execution_not_complete")
    if comparison.get("regression") is True:
        reasons.append("solver_accuracy_regressed_vs_exp2980")
    return reasons


def is_formalization_clean(metrics: Mapping[str, Any], *, n_items: int) -> bool:
    """Return the explicit Exp 2992 solver-backed formalization gate."""

    return bool(
        n_items >= FIXED_ITEM_COUNT
        and float(metrics.get("parseability_rate") or 0.0) >= 0.50
        and float(metrics.get("z3_execution_rate") or 0.0) == 1.0
        and float(metrics.get("solver_verified_accuracy") or 0.0) >= 0.40
        and float(metrics.get("tautology_flag_rate") or 0.0) == 0.0
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp 2992 terminal contract is internally inconsistent."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not str(artifact.get("honest_verdict", "")).startswith(("reproduced:", "flagged:", "blocked:")):
        raise ValueError("honest_verdict must state reproduced, flagged, or blocked")
    if artifact.get("solver_provenance_reproduced"):
        if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
            raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")
        if float(artifact.get("duration_seconds") or 0.0) < MIN_PLAUSIBLE_LIVE_SECONDS:
            raise ValueError("implausible duration for reproduced live headline inference")
        if int(artifact.get("n_items") or 0) < FIXED_ITEM_COUNT:
            raise ValueError("reproduced provenance requires at least 12 fixed items")
        if not artifact.get("formalization_clean"):
            raise ValueError("reproduced provenance requires formalization_clean")
        for field in ("prompt_hashes_recorded", "z3_transcript_hashes_recorded", "model_checksums_recorded"):
            if not artifact.get(field):
                raise ValueError(f"reproduced provenance requires {field}")
        if float(artifact.get("z3_execution_rate") or 0.0) < 1.0:
            raise ValueError("reproduced provenance requires complete Z3 execution")


def metrics_with_tautology(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute aggregate formalization metrics plus tautology rate."""

    metrics = exp2967.aggregate_results(rows)
    tautology_count = sum(bool(row.get("tautology_flag")) for row in rows)
    metrics["tautology_flag_rate"] = round(tautology_count / len(rows), 6) if rows else 0.0
    metrics["failure_categories"] = dict(metrics["failure_categories"])
    metrics["failure_categories"]["tautology_flagged"] = tautology_count
    return metrics


def models_used_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return ordered model IDs that produced live, hashable output rows."""

    seen: list[str] = []
    for row in rows:
        hf_id = str(row.get("model_hf_id") or "")
        if hf_id and hf_id not in seen and row.get("raw_output_sha256") and not row.get("generation_blocker"):
            seen.append(hf_id)
    return seen


def fixed_item_set_hash(items: Sequence[FeedbackFrontierItem]) -> str:
    """Return a stable hash for the chosen item IDs, prompts, and references."""

    payload = [
        {
            "item_id": item.item_id,
            "prompt": item.prompt,
            "expected_solver_status": item.expected_solver_status,
            "expected_answer_values": dict(item.expected_answer_values),
            "accepted_reference_formalization": item.accepted_reference_formalization,
        }
        for item in items
    ]
    return sha256_json(payload)


def write_z3_transcript(transcript_dir: Path, item_id: str, transcript: Mapping[str, Any]) -> JsonDict:
    """Write one replayable Z3 transcript and return path/hash evidence."""

    transcript_dir.mkdir(parents=True, exist_ok=True)
    path = transcript_dir / f"{item_id}.json"
    encoded = json.dumps(transcript, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(encoded + b"\n")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def field_provenance() -> JsonDict:
    """Principle annotations copied into the artifact for audit readability."""

    return {
        "solver_provenance_reproduced": {
            "principle": "Downstream self-learning must gate on reproducible solver evidence.",
            "satisfied_by": "strict conjunction of model, prompt, Z3 transcript, duration, and metric gates",
        },
        "formalization_clean": {
            "principle": "Solver promotion must be explicit.",
            "satisfied_by": "Z3-backed aggregate gate, independent of LLM self-judgment",
        },
        "n_items": {
            "principle": "Sample size must exceed the .280 pilot.",
            "satisfied_by": f"fixed {FIXED_ITEM_COUNT}-item Exp 2966 subset",
        },
        "duration_seconds": {
            "principle": "Live inference timing must be plausible.",
            "satisfied_by": "real wall-clock delta; no sleep padding",
        },
    }


def honest_verdict(reproduced: bool, flagged: bool, blocked_reason: str | None) -> str:
    """Return the terminal verdict with one of the required prefixes."""

    if blocked_reason:
        return f"blocked: {blocked_reason}"
    if reproduced:
        return "reproduced: solver-feedback formalization gain reproduced with stricter Z3 provenance"
    if flagged:
        return "flagged: solver-feedback formalization did not meet stricter provenance gates"
    return "flagged: insufficient evidence for reproduction"


def sha256_text(text: str) -> str:
    """Return a SHA-256 digest for UTF-8 text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: Any) -> str:
    """Return a stable SHA-256 digest for JSON-serializable content."""

    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _blocked_artifact(config: ExperimentConfig, preconditions: Preconditions, duration_seconds: float) -> JsonDict:
    metrics = metrics_with_tautology([])
    reason = ",".join(preconditions.blocking_reasons)
    artifact = {
        "solver_provenance_reproduced": False,
        "formalization_clean": False,
        "n_items": len(preconditions.items),
        "parseability": metrics["parseability_rate"],
        "z3_execution_rate": metrics["z3_execution_rate"],
        "solver_verified_accuracy": metrics["solver_verified_accuracy"],
        "feedback_repair_delta": 0.0,
        "tautology_rate": metrics["tautology_flag_rate"],
        "prompt_hashes_recorded": False,
        "z3_transcript_hashes_recorded": False,
        "model_checksums_recorded": any(_checksum_recorded(row) for row in preconditions.model_checksums.values()),
        "duration_seconds": round(duration_seconds, 6),
        "inference_substrate": "blocked_precondition",
        "honest_verdict": honest_verdict(False, False, reason),
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "model_specs": preconditions.model_specs,
        "models_used": [],
        "headline_model_ids": list(HEADLINE_MODEL_IDS),
        "smoke_only_model_ids": list(SMOKE_ONLY_MODEL_IDS),
        "model_checksums": preconditions.model_checksums,
        "preconditions_checked": preconditions.rows,
        "fixed_item_set_hash": preconditions.item_set_hash,
        "initial_metrics": metrics,
        "failure_categories": metrics["failure_categories"],
        "answer_accuracy": metrics["answer_accuracy"],
        "model_attempts": [],
        "raw_model_outputs_recorded": False,
        "per_item_results": [],
        "comparison_to_exp2980": comparison_to_exp2980(preconditions.prior_exp2980, metrics, len(preconditions.items)),
        "non_reproduction_reasons": preconditions.blocking_reasons,
        "blocking_reasons": preconditions.blocking_reasons,
        "cached_sota_pair_used": preconditions.cached_sota_pair_used,
        "field_provenance": field_provenance(),
    }
    validate_artifact(artifact)
    return artifact


def _read_optional_json(path: Path) -> JsonDict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checksum_recorded(row: Mapping[str, Any]) -> bool:
    return bool(row.get("sha256") or row.get("bounded_sha256"))


def _all_sha256(values: Any) -> bool:
    values_list = list(values)
    return bool(values_list) and all(isinstance(value, str) and len(value) == 64 for value in values_list)


def _default_llama_importer() -> tuple[bool, type[Any] | None, str | None]:  # pragma: no cover
    try:
        from llama_cpp import Llama
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"
    return True, Llama, None


def main() -> int:  # pragma: no cover
    artifact = run_experiment()
    print(
        "[exp2992] "
        f"verdict={artifact['honest_verdict']} "
        f"items={artifact['n_items']} "
        f"parse={artifact['parseability']} "
        f"z3={artifact['z3_execution_rate']} "
        f"solver={artifact['solver_verified_accuracy']} "
        f"repair_delta={artifact['feedback_repair_delta']} "
        f"duration={artifact['duration_seconds']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
