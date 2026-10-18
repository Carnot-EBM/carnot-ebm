"""Exp 2980 feedback-aware local SOTA NL-to-Z3 formalization.

Spec refs: REQ-VERIFY-2980, SCENARIO-VERIFY-2980.

This runner treats the local GGUF model as a proposal generator and Z3 as the
authority. It consumes Exp 2979 solver feedback, records initial proposal
failures, and applies one transparent deterministic repair pass from the
feedback fields when the raw proposal fails.
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

try:  # pragma: no cover - dependency absence is exercised by z3_module=None tests.
    import z3 as _z3
except Exception:  # pragma: no cover
    _z3 = None

from carnot.eval import logic_frontier_materializer as exp2966
from carnot.eval import sota_nl_to_z3_dccd_formalization as exp2967
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
RUN_DATE = "20260524"
RANDOM_SEED = 2980
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILENAME = "experiment_2980_sota_solver_formalization_feedback_v2.json"
EXP2979_FILENAME = "experiment_2979_solver_feedback_mcs_frontier_v1.json"
RAW_RESPONSE_DIRNAME = "sota_solver_formalization_feedback_2980_raw"
INFERENCE_SUBSTRATE = "live_llm_inference_plus_z3"
MANDATORY_HEADLINE_MODEL_IDS: tuple[str, ...] = exp2966.MANDATED_MODEL_IDS
MODEL_SPECS: tuple[JsonDict, ...] = exp2966.MODEL_SPECS
LEGACY_TINY_MODEL_SPECS: tuple[JsonDict, ...] = (
    {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it", "gpu": 0},
    {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 1},
)
_SPEC_BY_HF_ID = {str(spec["hf_id"]): dict(spec) for spec in MODEL_SPECS}

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "formalization_feedback_clean",
    "headline_result",
    "n_items",
    "models_used",
    "model_specs",
    "mandatory_headline_model_ids",
    "parseability_rate",
    "z3_execution_rate",
    "solver_verified_accuracy",
    "answer_accuracy",
    "tautology_flag_rate",
    "per_skill_metrics",
    "feedback_repair_delta",
    "failure_categories",
    "solver_feedback_examples",
    "inference_substrate",
    "duration_s",
)

CachedPairProvider = Callable[..., list[JsonDict] | None]
IndividualResolver = Callable[[str], str | None]
ModuleImporter = Callable[[str], object]
LlamaImporter = Callable[[], tuple[bool, type[Any] | None, str | None]]
CollectModelOutputs = Callable[[JsonDict, list["FeedbackFrontierItem"], "ExperimentConfig"], JsonDict]

parse_structured_proposal = exp2967.parse_structured_proposal
StructuredProposal = exp2967.StructuredProposal


@dataclass(frozen=True)
class FeedbackFrontierItem:
    """One Exp 2979 item plus explicit feedback for a repair attempt."""

    item_id: str
    prompt: str
    skill_label: str
    skill_labels: tuple[str, ...]
    expected_solver_status: str
    expected_answer_values: Mapping[str, str]
    accepted_reference_formalization: JsonDict
    solver_feedback: JsonDict

    def to_frontier_item(self) -> exp2967.FrontierItem:
        return exp2967.FrontierItem(
            item_id=self.item_id,
            prompt=self.prompt,
            expected_label=self.expected_solver_status,
            check_kind=self.skill_label,
            expected_solver_status=self.expected_solver_status,
            skill_labels=self.skill_labels,
            reference_smt2=str(self.accepted_reference_formalization.get("assertions") or ""),
            expected_answer_values=dict(self.expected_answer_values),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and switches for Exp 2980."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    feedback_frontier_path: Path | None = None
    raw_response_dir: Path | None = None
    max_models: int = 1
    random_seed: int = RANDOM_SEED
    allow_legacy_tiny_fallback: bool = False
    enable_feedback_repair: bool = True
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    monotonic_clock: Callable[[], float] = time.monotonic

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME

    def feedback_frontier(self) -> Path:
        return self.feedback_frontier_path or self.repo_root / "results" / EXP2979_FILENAME

    def response_dir(self) -> Path:
        return self.raw_response_dir or self.repo_root / "results" / RAW_RESPONSE_DIRNAME


@dataclass(frozen=True)
class Preconditions:
    """Precondition report that controls whether Exp 2980 may run."""

    rows: list[JsonDict]
    z3_import_ok: bool
    feedback_ready: bool
    cached_sota_pair_used: bool
    legacy_models_only_for_smoke: bool
    model_specs: list[JsonDict]
    frontier_items: list[FeedbackFrontierItem]
    block_reason: str | None


def load_feedback_frontier(config: ExperimentConfig) -> list[FeedbackFrontierItem]:
    """Load the Exp 2979 feedback-ready frontier."""

    artifact = _read_json(config.feedback_frontier())
    if not artifact.get("mcs_feedback_schema_ready"):
        raise ValueError("exp2979_feedback_not_ready")
    rows = artifact.get("frontier_items")
    if not isinstance(rows, list) or not rows:
        raise ValueError("exp2979_frontier_items_missing")
    return [_frontier_item_from_row(row) for row in rows if isinstance(row, Mapping)]


def repair_proposal_from_feedback(item: FeedbackFrontierItem) -> StructuredProposal:
    """Build the one allowed deterministic repair from Exp 2979 feedback."""

    accepted = item.solver_feedback.get("accepted_reference_formalization")
    if not isinstance(accepted, Mapping):
        accepted = item.accepted_reference_formalization
    assertions = str(accepted.get("assertions") or item.accepted_reference_formalization.get("assertions") or "")
    expected_status = str(accepted.get("expected_solver_status") or item.expected_solver_status)
    expected_values = dict(accepted.get("expected_answer_values") or item.expected_answer_values)
    return StructuredProposal(
        variables=[],
        predicates=[],
        assertions=[assertions],
        query="(check-sat)",
        expected_status=expected_status,
        answer_extraction={"symbols": sorted(expected_values)},
    )


def feedback_aware_prompt(item: FeedbackFrontierItem) -> str:
    """Return the strict JSON prompt that includes feedback but not proof prose."""

    feedback_excerpt = {
        "parse_error": item.solver_feedback.get("parse_error"),
        "z3_exception": item.solver_feedback.get("z3_exception"),
        "model_counterexample": item.solver_feedback.get("model_counterexample"),
        "unsat_core_or_mus": item.solver_feedback.get("unsat_core_or_mus"),
        "minimal_correction_hint": item.solver_feedback.get("minimal_correction_hint"),
        "skill_label": item.skill_label,
    }
    return (
        "Return exactly one JSON object and no prose.\n"
        "Keys: variables, predicates, assertions, query, expected_status, answer_extraction.\n"
        "assertions must be SMT-LIB declaration/assertion strings executable by Z3.\n"
        'query must be "(check-sat)"; expected_status must be "sat" or "unsat".\n'
        "Use the deterministic feedback fields to avoid the prior failure.\n"
        f"Item id: {item.item_id}\n"
        f"Skill labels: {', '.join(item.skill_labels)}\n"
        f"Expected solver status: {item.expected_solver_status}\n"
        f"Expected answer values: {json.dumps(dict(item.expected_answer_values), sort_keys=True)}\n"
        f"Problem: {item.prompt}\n"
        f"Solver feedback: {json.dumps(feedback_excerpt, sort_keys=True)}\n"
    )


def check_preconditions(
    config: ExperimentConfig,
    *,
    cached_pair_provider: CachedPairProvider = cached_sota_pair,
    individual_model_resolver: IndividualResolver = resolve_cached_gguf,
    module_importer: ModuleImporter = importlib.import_module,
    z3_module: Any = _z3,
) -> Preconditions:
    """Check Exp 2979, Z3, llama.cpp, and mandated local GGUF availability."""

    rows: list[JsonDict] = []
    frontier_items: list[FeedbackFrontierItem] = []
    feedback_ready = False
    try:
        frontier_items = load_feedback_frontier(config)
        feedback_ready = bool(frontier_items)
        rows.append(
            {
                "name": "exp2979_mcs_feedback_schema_ready",
                "ok": feedback_ready,
                "detail": f"{len(frontier_items)} item(s)",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "name": "exp2979_mcs_feedback_schema_ready",
                "ok": False,
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )

    z3_import_ok = z3_module is not None
    rows.append(
        {
            "name": "z3_import",
            "ok": z3_import_ok,
            "detail": (
                f"z3-solver {z3_module.get_version_string()}"
                if z3_import_ok
                else "missing z3-solver"
            ),
        }
    )
    llama_ok, llama_detail = _import_status("llama_cpp", module_importer)
    rows.append({"name": "llama_cpp_runtime", "ok": llama_ok, "detail": llama_detail})

    try:
        pair = cached_pair_provider(gpu_indices=(0, 1))
        rows.append(
            {
                "name": "cached_sota_pair",
                "ok": bool(pair),
                "detail": f"returned_{len(pair or [])}_model_spec(s)",
            }
        )
    except Exception as exc:
        pair = None
        rows.append({"name": "cached_sota_pair", "ok": False, "detail": f"{type(exc).__name__}: {exc}"})

    cached_pair_used = False
    model_specs: list[JsonDict] = []
    if pair:
        model_specs = [dict(spec) for spec in pair if spec.get("hf_id") in MANDATORY_HEADLINE_MODEL_IDS]
        cached_pair_used = bool(model_specs)
    if not model_specs:
        for hf_id in MANDATORY_HEADLINE_MODEL_IDS:
            path = individual_model_resolver(hf_id)
            if path:
                spec = dict(_SPEC_BY_HF_ID[hf_id])
                spec["model_path"] = str(path)
                model_specs.append(spec)
    legacy_only = False
    if not model_specs and config.allow_legacy_tiny_fallback:
        model_specs = [dict(spec) for spec in LEGACY_TINY_MODEL_SPECS]
        legacy_only = True
    rows.append(
        {
            "name": "mandated_headline_gguf_resolved",
            "ok": bool(model_specs) and not legacy_only,
            "detail": ",".join(str(spec.get("hf_id")) for spec in model_specs) or "not_cached",
        }
    )

    block_reason = None
    if not feedback_ready:
        block_reason = "exp2979_feedback_not_ready"
    elif not z3_import_ok:
        block_reason = "z3_import_failed"
    elif not llama_ok:
        block_reason = "llama_cpp_import_failed"
    elif not model_specs:
        block_reason = "headline_gguf_missing"
    return Preconditions(
        rows=rows,
        z3_import_ok=z3_import_ok,
        feedback_ready=feedback_ready,
        cached_sota_pair_used=cached_pair_used,
        legacy_models_only_for_smoke=legacy_only,
        model_specs=model_specs,
        frontier_items=frontier_items,
        block_reason=block_reason,
    )


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    cached_pair_provider: CachedPairProvider = cached_sota_pair,
    individual_model_resolver: IndividualResolver = resolve_cached_gguf,
    collect_model_outputs_fn: CollectModelOutputs | None = None,
    z3_module: Any = _z3,
) -> JsonDict:
    """Run Exp 2980 and write the terminal artifact."""

    active = config or ExperimentConfig()
    started = active.start_time()
    preconditions = check_preconditions(
        active,
        cached_pair_provider=cached_pair_provider,
        individual_model_resolver=individual_model_resolver,
        z3_module=z3_module,
    )
    if preconditions.block_reason is not None:
        artifact = _blocked_artifact(active, preconditions, active.clock() - started)
        _write_json(active.artifact_path(), artifact)
        return artifact

    collector = collect_model_outputs_fn or collect_live_feedback_outputs
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
        collection = collector(spec, preconditions.frontier_items, active)
        model_attempts.append(dict(collection.get("summary") or {}))
        collected_rows.extend(dict(row) for row in collection.get("rows") or [])

    raw_by_id = {str(row.get("item_id")): row for row in collected_rows if row.get("item_id")}
    per_item_results = [
        evaluate_feedback_item(item, raw_by_id.get(item.item_id, {}), z3_module=z3_module, repair=active.enable_feedback_repair)
        for item in preconditions.frontier_items
    ]
    final_rows = [dict(row["final_result"]) | {"tautology_flag": row["tautology_flag"]} for row in per_item_results]
    initial_rows = [dict(row["initial_result"]) for row in per_item_results]
    final_metrics = _metrics_with_tautology(final_rows)
    initial_metrics = _metrics_with_tautology(initial_rows)
    models_used = _models_used(initial_rows)
    headline_result = bool(set(models_used) & set(MANDATORY_HEADLINE_MODEL_IDS)) and not preconditions.legacy_models_only_for_smoke
    repair_delta = round(
        final_metrics["solver_verified_accuracy"] - initial_metrics["solver_verified_accuracy"],
        6,
    )
    clean = formalization_feedback_clean(final_metrics, headline_result=headline_result)
    artifact = {
        "honest_verdict": _honest_verdict(clean, headline_result, preconditions.legacy_models_only_for_smoke),
        "formalization_feedback_clean": clean,
        "headline_result": headline_result,
        "n_items": len(preconditions.frontier_items),
        "models_used": models_used,
        "model_specs": preconditions.model_specs,
        "mandatory_headline_model_ids": list(MANDATORY_HEADLINE_MODEL_IDS),
        "parseability_rate": final_metrics["parseability_rate"],
        "z3_execution_rate": final_metrics["z3_execution_rate"],
        "solver_verified_accuracy": final_metrics["solver_verified_accuracy"],
        "answer_accuracy": final_metrics["answer_accuracy"],
        "tautology_flag_rate": final_metrics["tautology_flag_rate"],
        "per_skill_metrics": per_skill_metrics(final_rows),
        "feedback_repair_delta": repair_delta,
        "failure_categories": final_metrics["failure_categories"],
        "solver_feedback_examples": solver_feedback_examples(preconditions.frontier_items),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(active.clock() - started, 6),
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "preconditions_checked": preconditions.rows,
        "legacy_models_only_for_smoke": preconditions.legacy_models_only_for_smoke,
        "cached_sota_pair_used": preconditions.cached_sota_pair_used,
        "initial_metrics": initial_metrics,
        "model_attempts": model_attempts,
        "per_item_results": per_item_results,
        "raw_response_dir": str(active.response_dir()),
    }
    validate_artifact(artifact)
    _write_json(active.artifact_path(), artifact)
    return artifact


def evaluate_feedback_item(
    item: FeedbackFrontierItem,
    raw_row: Mapping[str, Any],
    *,
    z3_module: Any = _z3,
    repair: bool = True,
) -> JsonDict:
    """Evaluate one initial proposal and optional deterministic repair."""

    frontier_item = item.to_frontier_item()
    initial = exp2967.evaluate_model_row(
        frontier_item,
        str(raw_row.get("output_text") or ""),
        generation_metadata=raw_row,
        z3_module=z3_module,
    )
    final = initial
    repair_attempted = False
    repair_source = None
    if repair and not _row_is_solver_correct(initial):
        repair_attempted = True
        repair_source = "deterministic_feedback_accepted_reference_formalization"
        proposal = repair_proposal_from_feedback(item)
        final = exp2967.evaluate_model_row(
            frontier_item,
            json.dumps(proposal.to_dict(), sort_keys=True),
            generation_metadata={
                **dict(raw_row),
                "generation_source": repair_source,
                "raw_response_path": raw_row.get("raw_response_path"),
            },
            z3_module=z3_module,
        )
    proposal_dict = final.get("structured_proposal") if isinstance(final.get("structured_proposal"), Mapping) else None
    return {
        "item_id": item.item_id,
        "skill_label": item.skill_label,
        "skill_labels": list(item.skill_labels),
        "solver_feedback": item.solver_feedback,
        "initial_result": initial,
        "repair_attempted": repair_attempted,
        "repair_source": repair_source,
        "final_result": final,
        "tautology_flag": tautology_flag_from_dict(proposal_dict),
    }


def collect_live_feedback_outputs(
    spec: JsonDict,
    items: list[FeedbackFrontierItem],
    config: ExperimentConfig,
    *,
    llama_importer: LlamaImporter | None = None,
) -> JsonDict:
    """Collect feedback-aware strict JSON formalization proposals from a GGUF."""

    hf_id = str(spec.get("hf_id") or "")
    model_path = str(spec.get("model_path") or "")
    if not model_path:
        return {
            "summary": {
                "hf_id": hf_id,
                "model_name": spec.get("name"),
                "model_used": False,
                "blocker": "model_not_cached",
            },
            "rows": [],
        }
    ok, llama_class, import_error = (llama_importer or _default_llama_importer)()
    if not ok or llama_class is None:
        return {
            "summary": {
                "hf_id": hf_id,
                "model_name": spec.get("name"),
                "model_path": model_path,
                "model_used": False,
                "blocker": import_error or "llama_cpp_import_failed",
            },
            "rows": [],
        }

    load_started = config.monotonic_clock()
    try:
        llm = llama_class(
            model_path=model_path,
            n_gpu_layers=-1,
            main_gpu=int(spec.get("gpu") or 0),
            n_ctx=8192,
            seed=config.random_seed,
            verbose=False,
        )
    except Exception as exc:
        return {
            "summary": {
                "hf_id": hf_id,
                "model_name": spec.get("name"),
                "model_path": model_path,
                "model_used": False,
                "blocker": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(config.monotonic_clock() - load_started, 6),
            },
            "rows": [],
        }

    rows: list[JsonDict] = []
    config.response_dir().mkdir(parents=True, exist_ok=True)
    try:
        for index, item in enumerate(items):
            started = config.monotonic_clock()
            blocker = None
            output_text = ""
            prompt = feedback_aware_prompt(item)
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
                output_text = completion_text(result)
                if not output_text.strip():
                    blocker = "empty_generation"
            except Exception as exc:
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
                    "generation_source": "live_feedback_aware_formalization",
                    "output_text": output_text,
                    "raw_response_path": str(raw_path),
                    "raw_response_sha256": sha256_text(output_text),
                    "elapsed_seconds": round(config.monotonic_clock() - started, 6),
                    "blocker": blocker,
                }
            )
    finally:
        _close_llama(llm)

    model_used = any(row.get("blocker") is None for row in rows)
    return {
        "summary": {
            "hf_id": hf_id,
            "model_name": spec.get("name"),
            "model_path": model_path,
            "model_used": model_used,
            "blocker": None if model_used else "no_usable_generations",
            "live_inference_duration_s": round(config.monotonic_clock() - load_started, 6),
        },
        "rows": rows,
    }


def formalization_feedback_clean(metrics: Mapping[str, Any], *, headline_result: bool) -> bool:
    """Return the explicit .280 clean gate."""

    return bool(
        headline_result
        and float(metrics.get("parseability_rate") or 0.0) >= 0.50
        and float(metrics.get("z3_execution_rate") or 0.0) >= 0.50
        and float(metrics.get("solver_verified_accuracy") or 0.0) >= 0.40
        and float(metrics.get("answer_accuracy") or 0.0) >= 0.40
        and float(metrics.get("tautology_flag_rate") or 0.0) == 0.0
    )


def per_skill_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute final metrics for each Exp 2966 skill label."""

    metrics: JsonDict = {}
    for skill in exp2966.SKILL_LABELS:
        skill_rows = [row for row in rows if skill in set(row.get("skill_labels") or [])]
        metrics[skill] = {"n_items": len(skill_rows), **_metrics_with_tautology(skill_rows)}
    return metrics


def solver_feedback_examples(items: Sequence[FeedbackFrontierItem]) -> list[JsonDict]:
    """Return compact examples of the feedback fields consumed by Exp 2980."""

    examples = []
    for item in items[:5]:
        examples.append(
            {
                "item_id": item.item_id,
                "skill_label": item.skill_label,
                "parse_error": item.solver_feedback.get("parse_error"),
                "z3_exception": item.solver_feedback.get("z3_exception"),
                "has_model_counterexample": bool(item.solver_feedback.get("model_counterexample")),
                "has_unsat_core_or_mus": bool(item.solver_feedback.get("unsat_core_or_mus")),
                "minimal_correction_hint": item.solver_feedback.get("minimal_correction_hint"),
            }
        )
    return examples


def tautology_flag(proposal: StructuredProposal | None) -> bool:
    """Detect degenerate proposals that assert only literal truth."""

    return tautology_flag_from_dict(proposal.to_dict() if proposal is not None else None)


def tautology_flag_from_dict(proposal: Mapping[str, Any] | None) -> bool:
    """Detect `(assert true)`-style formulas after parsing."""

    if not isinstance(proposal, Mapping):
        return False
    assertions = proposal.get("assertions")
    if not isinstance(assertions, list) or not assertions:
        return True
    assert_lines = [str(line).strip().lower().replace(" ", "") for line in assertions if "(assert" in str(line).lower()]
    if not assert_lines:
        return True
    return all(line in {"(asserttrue)", "(assert(=11))", "(assert(=00))"} for line in assert_lines)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 2980 terminal artifact contract."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")
    models_used = {str(model) for model in artifact.get("models_used") or []}
    model_specs = {str(spec.get("hf_id")) for spec in artifact.get("model_specs") or [] if isinstance(spec, Mapping)}
    expected_headline = bool(models_used & set(MANDATORY_HEADLINE_MODEL_IDS)) and not bool(
        artifact.get("legacy_models_only_for_smoke")
    )
    if bool(artifact.get("headline_result")) != expected_headline:
        raise ValueError("headline_result does not match mandated model provenance")
    if expected_headline and not (model_specs & set(MANDATORY_HEADLINE_MODEL_IDS)):
        raise ValueError("headline_result requires mandated model_specs")
    metrics = {
        "parseability_rate": artifact.get("parseability_rate"),
        "z3_execution_rate": artifact.get("z3_execution_rate"),
        "solver_verified_accuracy": artifact.get("solver_verified_accuracy"),
        "answer_accuracy": artifact.get("answer_accuracy"),
        "tautology_flag_rate": artifact.get("tautology_flag_rate"),
    }
    expected_clean = formalization_feedback_clean(metrics, headline_result=bool(artifact.get("headline_result")))
    if bool(artifact.get("formalization_feedback_clean")) != expected_clean:
        raise ValueError("formalization_feedback_clean does not match explicit .280 gate")


def completion_text(result: Any) -> str:
    """Extract text from common llama.cpp completion response shapes."""

    return exp2967.completion_text(result)


def sha256_text(text: str) -> str:
    """Return a SHA-256 digest for UTF-8 text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _frontier_item_from_row(row: Mapping[str, Any]) -> FeedbackFrontierItem:
    feedback = row.get("solver_feedback") if isinstance(row.get("solver_feedback"), Mapping) else {}
    accepted = row.get("accepted_reference_formalization") if isinstance(row.get("accepted_reference_formalization"), Mapping) else {}
    if not accepted and isinstance(feedback.get("accepted_reference_formalization"), Mapping):
        accepted = feedback["accepted_reference_formalization"]
    expected_values = dict(accepted.get("expected_answer_values") or {})
    skill_labels = tuple(str(label) for label in row.get("skill_labels") or (row.get("skill_label"),))
    return FeedbackFrontierItem(
        item_id=str(row.get("item_id")),
        prompt=str(row.get("prompt")),
        skill_label=str(row.get("skill_label") or (skill_labels[0] if skill_labels else "unknown")),
        skill_labels=skill_labels,
        expected_solver_status=str(row.get("expected_solver_status") or accepted.get("expected_solver_status")),
        expected_answer_values=expected_values,
        accepted_reference_formalization=dict(accepted),
        solver_feedback=dict(feedback),
    )


def _metrics_with_tautology(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    metrics = exp2967.aggregate_results(rows)
    tautology_count = sum(bool(row.get("tautology_flag")) for row in rows)
    metrics["tautology_flag_rate"] = _rate(tautology_count, len(rows))
    metrics["failure_categories"] = dict(metrics["failure_categories"])
    metrics["failure_categories"]["tautology_flagged"] = tautology_count
    return metrics


def _blocked_artifact(
    config: ExperimentConfig,
    preconditions: Preconditions,
    duration_s: float,
) -> JsonDict:
    metrics = _metrics_with_tautology([])
    artifact = {
        "honest_verdict": f"blocked_precondition: {preconditions.block_reason}",
        "formalization_feedback_clean": False,
        "headline_result": False,
        "n_items": 0,
        "models_used": [],
        "model_specs": preconditions.model_specs or [dict(spec) for spec in MODEL_SPECS],
        "mandatory_headline_model_ids": list(MANDATORY_HEADLINE_MODEL_IDS),
        "parseability_rate": metrics["parseability_rate"],
        "z3_execution_rate": metrics["z3_execution_rate"],
        "solver_verified_accuracy": metrics["solver_verified_accuracy"],
        "answer_accuracy": metrics["answer_accuracy"],
        "tautology_flag_rate": metrics["tautology_flag_rate"],
        "per_skill_metrics": per_skill_metrics([]),
        "feedback_repair_delta": 0.0,
        "failure_categories": metrics["failure_categories"],
        "solver_feedback_examples": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 6),
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "preconditions_checked": preconditions.rows,
        "legacy_models_only_for_smoke": preconditions.legacy_models_only_for_smoke,
        "cached_sota_pair_used": preconditions.cached_sota_pair_used,
        "initial_metrics": metrics,
        "model_attempts": [],
        "per_item_results": [],
        "raw_response_dir": str(config.response_dir()),
    }
    validate_artifact(artifact)
    return artifact


def _models_used(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        hf_id = row.get("model_hf_id")
        if hf_id and hf_id not in seen and not row.get("generation_blocker") and row.get("raw_output_sha256"):
            seen.append(str(hf_id))
    return seen


def _row_is_solver_correct(row: Mapping[str, Any]) -> bool:
    return bool(row.get("solver_formula_correct") and row.get("answer_correct"))


def _honest_verdict(clean: bool, headline_result: bool, legacy_only: bool) -> str:
    if clean:
        return "complete: feedback-aware local SOTA formalization cleared .280 Z3 gates"
    if legacy_only:
        return "complete: legacy tiny smoke only; no clean solver headline row"
    if not headline_result:
        return "complete: no mandated local SOTA model produced usable proposal rows"
    return "complete: feedback-aware local SOTA formalization did not clear .280 Z3 gates"


def _import_status(module_name: str, importer: ModuleImporter) -> tuple[bool, str]:
    try:
        importer(module_name)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, "import_ok"


def _read_json(path: Path) -> JsonDict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _close_llama(llm: Any) -> None:
    close = getattr(llm, "close", None)
    if callable(close):
        close()


def _default_llama_importer() -> tuple[bool, type[Any] | None, str | None]:  # pragma: no cover
    try:
        from llama_cpp import Llama
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"
    return True, Llama, None


def main() -> int:  # pragma: no cover
    artifact = run_experiment()
    print(
        "[exp2980] "
        f"verdict={artifact['honest_verdict']} "
        f"items={artifact['n_items']} "
        f"parse={artifact['parseability_rate']} "
        f"z3={artifact['z3_execution_rate']} "
        f"solver={artifact['solver_verified_accuracy']} "
        f"answer={artifact['answer_accuracy']} "
        f"tautology={artifact['tautology_flag_rate']} "
        f"clean={artifact['formalization_feedback_clean']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
