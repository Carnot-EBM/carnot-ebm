"""Exp 2959 NL-to-Z3 execution repair mini benchmark.

Spec: REQ-BENCH-2959, SCENARIO-BENCH-2959.

This module repairs the blocked Exp 2931 path at the narrowest useful layer:
local SOTA GGUF text may contain a valid formalization fragment instead of the
exact top-level JSON object requested by the prompt.  The repair accepts only
that bounded schema fragment, rebuilds the strict JSON contract, and then lets
Z3 decide the modal answer.  Model prose remains evidence to parse, never proof.
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

from carnot.eval import llmeval_logic_z3_mini as exp2931
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
RUN_DATE = "20260524"
RANDOM_SEED = 2959
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILENAME = "experiment_2959_nl_to_z3_execution_repair_mini_v2.json"
SOURCE_ARTIFACT_FILENAME = exp2931.OUTPUT_FILENAME
SOURCE_RAW_DIRNAME = exp2931.RAW_RESPONSE_DIRNAME
INFERENCE_SUBSTRATE = "live_llm_inference"

MANDATED_MODEL_IDS: tuple[str, ...] = exp2931.MANDATED_MODEL_IDS
MODEL_SPECS: tuple[JsonDict, ...] = exp2931.MANDATED_MODEL_SPECS
_SPEC_BY_HF_ID = {str(spec["hf_id"]): dict(spec) for spec in MODEL_SPECS}
ALLOWED_ANSWERS = exp2931.ALLOWED_ANSWERS
FAILURE_CATEGORY_NAMES = (
    "unparseable",
    "z3_exception",
    "wrong_formula",
    "wrong_answer",
    "solver_verified_correct",
)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "model_specs",
    "headline_models_used",
    "z3_import_ok",
    "z3_execution_repaired",
    "n_items",
    "parseability_rate",
    "z3_execution_rate",
    "solver_verified_accuracy",
    "answer_accuracy",
    "failure_categories",
    "formalization_manifest_sha256",
    "duration_s",
)

CachedPairProvider = Callable[..., list[JsonDict] | None]
IndividualResolver = Callable[[str], str | None]
ModuleImporter = Callable[[str], object]
RawRowsProvider = Callable[[list[exp2931.LogicItem], "ExperimentConfig"], list[JsonDict]]


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths for the Exp 2959 repair artifact."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    source_artifact_path: Path | None = None
    source_raw_dir: Path | None = None
    max_items: int = 12
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME

    def source_artifact(self) -> Path:
        return self.source_artifact_path or self.repo_root / "results" / SOURCE_ARTIFACT_FILENAME

    def raw_dir(self) -> Path:
        return self.source_raw_dir or self.repo_root / "results" / SOURCE_RAW_DIRNAME


@dataclass(frozen=True)
class RepairedModelResponse:
    """Parsed or schema-repaired response ready for Z3 execution."""

    parseable: bool
    formalization: JsonDict | None
    answer: str | None
    error: str | None
    repair_applied: bool
    repair_note: str | None


@dataclass(frozen=True)
class Preconditions:
    """Local runtime and model-resolution precondition report."""

    rows: list[JsonDict]
    z3_import_ok: bool
    llama_cpp_import_ok: bool
    cached_pair_used: bool
    model_specs: list[JsonDict]


def selected_logic_items(max_items: int = 12) -> list[exp2931.LogicItem]:
    """Return the bounded 8-12 item pack used by the repaired mini benchmark."""

    if not 8 <= max_items <= 12:
        raise ValueError("Exp 2959 selects 8-12 logic items")
    items, _scope = exp2931.build_or_load_logic_items(cache_paths=())
    return items[:max_items]


def parse_or_repair_model_response(text: str) -> RepairedModelResponse:
    """Parse strict JSON or repair one bounded formalization fragment."""

    strict = exp2931.parse_model_response(text)
    if strict.parseable:
        return RepairedModelResponse(
            parseable=True,
            formalization=strict.formalization,
            answer=strict.answer,
            error=None,
            repair_applied=False,
            repair_note=None,
        )

    formalization = _extract_formalization_fragment(text)
    if formalization is None:
        return RepairedModelResponse(False, None, None, "no_repairable_formalization", False, None)

    answer = _extract_answer(text)
    if answer is None:
        return RepairedModelResponse(False, None, None, "answer_not_in_allowed_set", True, None)
    return RepairedModelResponse(
        parseable=True,
        formalization=formalization,
        answer=answer,
        error=None,
        repair_applied=True,
        repair_note="formalization_fragment_plus_answer_line",
    )


def evaluate_repaired_output(
    item: exp2931.LogicItem,
    raw_text: str,
    *,
    generation_metadata: Mapping[str, Any],
) -> JsonDict:
    """Evaluate one local SOTA proposal with parser repair and Z3 authority."""

    parsed = parse_or_repair_model_response(raw_text)
    raw_sha = exp2931.sha256_text(raw_text)
    if not parsed.parseable or parsed.formalization is None:
        return {
            "item_id": item.item_id,
            "gold_answer": item.gold_answer,
            "model_answer": parsed.answer,
            "solver_answer": None,
            "parseable": False,
            "parse_repair_applied": parsed.repair_applied,
            "parse_repair_note": parsed.repair_note,
            "parse_error": parsed.error,
            "z3_executed": False,
            "z3_result": _unexecuted_z3_result(parsed.error),
            "solver_formula_correct": False,
            "answer_correct": False,
            "failure_category": "unparseable",
            "parsed_formalization": None,
            "raw_output_sha256": raw_sha,
            **_generation_fields(generation_metadata),
        }

    z3_result = exp2931.execute_z3_checks(parsed.formalization)
    z3_executed = bool(z3_result.get("z3_executed"))
    solver_answer = z3_result.get("solver_answer")
    solver_formula_correct = bool(z3_executed and solver_answer == item.gold_answer)
    answer_correct = bool(parsed.answer == item.gold_answer)
    if not z3_executed:
        failure_category = "z3_exception"
    elif not solver_formula_correct:
        failure_category = "wrong_formula"
    elif not answer_correct:
        failure_category = "wrong_answer"
    else:
        failure_category = "solver_verified_correct"

    return {
        "item_id": item.item_id,
        "gold_answer": item.gold_answer,
        "model_answer": parsed.answer,
        "solver_answer": solver_answer,
        "parseable": True,
        "parse_repair_applied": parsed.repair_applied,
        "parse_repair_note": parsed.repair_note,
        "parse_error": None,
        "z3_executed": z3_executed,
        "z3_result": z3_result,
        "solver_formula_correct": solver_formula_correct,
        "answer_correct": answer_correct,
        "failure_category": failure_category,
        "parsed_formalization": (
            exp2931.canonical_formalization(parsed.formalization) if z3_executed else None
        ),
        "raw_output_sha256": raw_sha,
        **_generation_fields(generation_metadata),
    }


def aggregate_repair_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute separate parse, Z3 execution, solver-label, and answer rates."""

    counts = {name: 0 for name in FAILURE_CATEGORY_NAMES}
    for row in rows:
        category = str(row.get("failure_category") or "unparseable")
        counts[category if category in counts else "unparseable"] += 1
    if not rows:
        return {
            "parseability_rate": 0.0,
            "z3_execution_rate": 0.0,
            "solver_verified_accuracy": 0.0,
            "answer_accuracy": 0.0,
            "failure_categories": counts,
        }
    total = len(rows)
    return {
        "parseability_rate": _rate(sum(bool(row.get("parseable")) for row in rows), total),
        "z3_execution_rate": _rate(sum(bool(row.get("z3_executed")) for row in rows), total),
        "solver_verified_accuracy": _rate(
            sum(bool(row.get("solver_formula_correct")) for row in rows),
            total,
        ),
        "answer_accuracy": _rate(sum(bool(row.get("answer_correct")) for row in rows), total),
        "failure_categories": counts,
    }


def check_preconditions(
    *,
    cached_pair_provider: CachedPairProvider = cached_sota_pair,
    individual_model_resolver: IndividualResolver = resolve_cached_gguf,
    module_importer: ModuleImporter = importlib.import_module,
) -> Preconditions:
    """Check Z3, local GGUF runtime, and mandated model cache availability."""

    z3_ok, z3_detail = _import_status("z3", module_importer)
    llama_ok, llama_detail = _import_status("llama_cpp", module_importer)
    rows = [
        {"name": "z3_import", "ok": z3_ok, "detail": z3_detail},
        {"name": "llama_cpp_runtime", "ok": llama_ok, "detail": llama_detail},
    ]
    pair: list[JsonDict] | None
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
        rows.append(
            {
                "name": "cached_sota_pair",
                "ok": False,
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )

    if pair:
        specs = [dict(spec) for spec in pair if spec.get("hf_id") in MANDATED_MODEL_IDS]
        rows.append(
            {
                "name": "mandated_headline_model_resolved",
                "ok": bool(specs),
                "detail": "cached_sota_pair",
            }
        )
        return Preconditions(rows, z3_ok, llama_ok, True, specs or _fallback_model_specs())

    specs = []
    for hf_id in MANDATED_MODEL_IDS:
        path = individual_model_resolver(hf_id)
        if path:
            spec = dict(_SPEC_BY_HF_ID[hf_id])
            spec["model_path"] = str(path)
            specs.append(spec)
    rows.append(
        {
            "name": "mandated_headline_model_resolved",
            "ok": bool(specs),
            "detail": "single_cached_mandated_gguf" if specs else "not_cached",
        }
    )
    return Preconditions(rows, z3_ok, llama_ok, False, specs or _fallback_model_specs())


def load_prior_live_rows(config: ExperimentConfig, items: Sequence[exp2931.LogicItem]) -> list[JsonDict]:
    """Load prior Exp 2931 live GGUF raw responses when available."""

    source_path = config.source_artifact()
    raw_dir = config.raw_dir()
    if not source_path.is_file() or not raw_dir.is_dir():
        return []
    with source_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not payload.get("models_used"):
        return []
    model_attempt = _first_used_attempt(payload.get("model_attempts") or [])
    rows = []
    for index, item in enumerate(items):
        raw_path = raw_dir / f"{item.item_id}.json"
        if not raw_path.is_file():
            continue
        output_text = raw_path.read_text(encoding="utf-8")
        rows.append(
            {
                "item_id": item.item_id,
                "model_hf_id": (payload.get("models_used") or [None])[0],
                "model_name": model_attempt.get("model_name"),
                "model_path": model_attempt.get("model_path"),
                "gpu_index": model_attempt.get("gpu_index"),
                "prompt_hash": exp2931.sha256_text(item.prompt),
                "per_item_seed": exp2931.RANDOM_SEED + index,
                "generation_source": "prior_live_sota_llamacpp_logic_json",
                "output_text": output_text,
                "raw_response_path": str(raw_path),
                "elapsed_seconds": None,
                "blocker": None,
            }
        )
    return rows


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    cached_pair_provider: CachedPairProvider = cached_sota_pair,
    individual_model_resolver: IndividualResolver = resolve_cached_gguf,
    raw_rows_provider: RawRowsProvider | None = None,
) -> JsonDict:
    """Run the repaired mini benchmark and write the Exp 2959 artifact."""

    active = config or ExperimentConfig()
    started = active.start_time()
    items = selected_logic_items(active.max_items)
    preconditions = check_preconditions(
        cached_pair_provider=cached_pair_provider,
        individual_model_resolver=individual_model_resolver,
    )
    raw_rows = (
        raw_rows_provider(items, active)
        if raw_rows_provider is not None
        else load_prior_live_rows(active, items)
    )
    raw_by_id = {str(row.get("item_id")): dict(row) for row in raw_rows if row.get("item_id")}
    per_item_results = []
    for item in items:
        raw_row = raw_by_id.get(item.item_id)
        raw_text = "" if raw_row is None else str(raw_row.get("output_text") or "")
        per_item_results.append(
            evaluate_repaired_output(item, raw_text, generation_metadata=raw_row or {})
        )

    metrics = aggregate_repair_results(per_item_results)
    headline_models_used = _headline_models_used(per_item_results)
    z3_execution_repaired = bool(preconditions.z3_import_ok and metrics["z3_execution_rate"] > 0.0)
    item_manifest = [exp2931._item_manifest_row(item) for item in items]  # noqa: SLF001
    manifest_sha = formalization_manifest_sha256(item_manifest, per_item_results)
    artifact: JsonDict = {
        "honest_verdict": _honest_verdict(
            z3_import_ok=preconditions.z3_import_ok,
            headline_models_used=headline_models_used,
            z3_execution_repaired=z3_execution_repaired,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions.rows,
        "model_specs": preconditions.model_specs,
        "headline_models_used": headline_models_used,
        "z3_import_ok": preconditions.z3_import_ok,
        "z3_execution_repaired": z3_execution_repaired,
        "n_items": len(items),
        "parseability_rate": metrics["parseability_rate"],
        "z3_execution_rate": metrics["z3_execution_rate"],
        "solver_verified_accuracy": metrics["solver_verified_accuracy"],
        "answer_accuracy": metrics["answer_accuracy"],
        "failure_categories": metrics["failure_categories"],
        "formalization_manifest_sha256": manifest_sha,
        "duration_s": round(active.clock() - started, 6),
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "item_manifest": item_manifest,
        "per_item_results": per_item_results,
        "source_artifact": str(active.source_artifact()),
        "source_raw_dir": str(active.raw_dir()),
        "cached_sota_pair_used": preconditions.cached_pair_used,
    }
    _write_json(active.artifact_path(), artifact)
    return artifact


def formalization_manifest_sha256(
    item_manifest: Sequence[Mapping[str, Any]],
    per_item_results: Sequence[Mapping[str, Any]],
) -> str:
    """Hash the gold items, repaired formalizations, Z3 results, and categories."""

    payload = {
        "item_manifest": list(item_manifest),
        "repaired_results": [
            {
                "item_id": row.get("item_id"),
                "model_answer": row.get("model_answer"),
                "solver_answer": row.get("solver_answer"),
                "parsed_formalization": row.get("parsed_formalization"),
                "z3_result": row.get("z3_result"),
                "failure_category": row.get("failure_category"),
                "raw_output_sha256": row.get("raw_output_sha256"),
            }
            for row in per_item_results
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _extract_formalization_fragment(text: str) -> JsonDict | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("formalization"), dict):
            candidate = dict(obj["formalization"])
        elif isinstance(obj, dict):
            candidate = dict(obj)
        if _strict_formalization_shape(candidate):
            return candidate
    return None


def _strict_formalization_shape(candidate: Mapping[str, Any]) -> bool:
    return (
        isinstance(candidate.get("facts"), list)
        and isinstance(candidate.get("rules"), list)
        and isinstance(candidate.get("exclusions"), list)
        and isinstance(candidate.get("query"), list)
    )


def _extract_answer(text: str) -> str | None:
    for line in text.splitlines():
        if "answer" not in line.lower():
            continue
        answer = _answer_in_text(line)
        if answer is not None:
            return answer
    return _answer_in_text(text)


def _answer_in_text(text: str) -> str | None:
    lowered = text.lower()
    for answer in sorted(ALLOWED_ANSWERS, key=len, reverse=True):
        token = answer.lower()
        if token in lowered:
            return token
    return None


def _unexecuted_z3_result(error: str | None) -> JsonDict:
    return {
        "z3_executed": False,
        "z3_error": error,
        "knowledge_base_consistent": False,
        "possible": False,
        "necessary": False,
        "solver_answer": None,
    }


def _generation_fields(metadata: Mapping[str, Any]) -> JsonDict:
    return {
        "model_hf_id": metadata.get("model_hf_id"),
        "model_name": metadata.get("model_name"),
        "model_path": metadata.get("model_path"),
        "gpu_index": metadata.get("gpu_index"),
        "prompt_hash": metadata.get("prompt_hash"),
        "per_item_seed": metadata.get("per_item_seed"),
        "generation_source": metadata.get("generation_source"),
        "generation_blocker": metadata.get("blocker"),
        "raw_response_path": metadata.get("raw_response_path"),
        "elapsed_seconds": metadata.get("elapsed_seconds"),
    }


def _import_status(module_name: str, importer: ModuleImporter) -> tuple[bool, str]:
    try:
        importer(module_name)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, "import_ok"


def _fallback_model_specs() -> list[JsonDict]:
    return [dict(spec) for spec in MODEL_SPECS]


def _first_used_attempt(attempts: Sequence[Any]) -> JsonDict:
    for attempt in attempts:
        if isinstance(attempt, Mapping) and attempt.get("model_used") is True:
            return dict(attempt)
    return {}


def _headline_models_used(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    seen = []
    for row in rows:
        hf_id = row.get("model_hf_id")
        if hf_id in MANDATED_MODEL_IDS and hf_id not in seen and row.get("raw_output_sha256"):
            seen.append(str(hf_id))
    return seen


def _honest_verdict(
    *,
    z3_import_ok: bool,
    headline_models_used: Sequence[str],
    z3_execution_repaired: bool,
) -> str:
    if not z3_import_ok:
        return "blocked_z3_import_failed"
    if not headline_models_used:
        return "blocked_live_sota_proposals_missing"
    if not z3_execution_repaired:
        return "blocked_z3_execution_unrepaired"
    return "complete: local SOTA logic proposals accepted_or_rejected_by_z3"


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:  # pragma: no cover
    artifact = run_experiment()
    print(
        "[exp2959] "
        f"verdict={artifact['honest_verdict']} "
        f"items={artifact['n_items']} "
        f"parse={artifact['parseability_rate']} "
        f"z3={artifact['z3_execution_rate']} "
        f"solver={artifact['solver_verified_accuracy']} "
        f"answer={artifact['answer_accuracy']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
