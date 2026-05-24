"""Exp 2967 live SOTA DCCD NL-to-Z3 frontier formalization.

Spec: REQ-BENCH-2967, SCENARIO-BENCH-2967.

This experiment takes the exact-verifier frontier from Exp 2966 and asks a
mandated local GGUF model to emit a strict SMT-LIB-backed structured proposal.
The model is allowed to draft the formalization, but Z3 remains the authority:
every parseable proposal is executed, categorized, and reported by skill label.
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
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
RUN_DATE = "20260524"
RANDOM_SEED = 2967
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILENAME = "experiment_2967_sota_nl_to_z3_dccd_formalization_v1.json"
RAW_RESPONSE_DIRNAME = "sota_nl_to_z3_dccd_2967_raw"
INFERENCE_SUBSTRATE = "live_llm_inference"
BASELINE_PARSEABILITY_RATE = 0.083333
BASELINE_SOLVER_VERIFIED_ACCURACY = 0.0

MODEL_SPECS: tuple[JsonDict, ...] = exp2966.MODEL_SPECS
MANDATED_MODEL_IDS: tuple[str, ...] = exp2966.MANDATED_MODEL_IDS
SKILL_LABELS: tuple[str, ...] = exp2966.SKILL_LABELS
_SPEC_BY_HF_ID = {str(spec["hf_id"]): dict(spec) for spec in MODEL_SPECS}

FAILURE_CATEGORY_NAMES: tuple[str, ...] = (
    "unparseable",
    "z3_exception",
    "wrong_formula",
    "wrong_answer",
    "solver_verified_correct",
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "model_specs",
    "headline_models_used",
    "legacy_models_only_for_smoke",
    "n_items",
    "parseability_rate",
    "z3_execution_rate",
    "solver_verified_accuracy",
    "answer_accuracy",
    "baseline_parseability_rate",
    "baseline_solver_verified_accuracy",
    "skill_wise_metrics",
    "failure_categories",
    "formalization_delta_clean",
    "formalization_manifest_sha256",
    "duration_s",
)

CachedPairProvider = Callable[..., list[JsonDict] | None]
IndividualResolver = Callable[[str], str | None]
ModuleImporter = Callable[[str], object]
LlamaImporter = Callable[[], tuple[bool, type[Any] | None, str | None]]
CollectModelOutputs = Callable[[JsonDict, list["FrontierItem"], "ExperimentConfig"], JsonDict]


@dataclass(frozen=True)
class FrontierItem:
    """One Exp 2966 frontier item with reference labels for scoring proposals."""

    item_id: str
    prompt: str
    expected_label: str
    check_kind: str
    expected_solver_status: str
    skill_labels: tuple[str, ...]
    reference_smt2: str
    expected_answer_values: Mapping[str, str]


@dataclass(frozen=True)
class StructuredProposal:
    """Validated DCCD structured-output proposal ready for Z3 execution."""

    variables: list[JsonDict]
    predicates: list[JsonDict]
    assertions: list[str]
    query: str
    expected_status: str
    answer_extraction: JsonDict

    def to_dict(self) -> JsonDict:
        return {
            "variables": self.variables,
            "predicates": self.predicates,
            "assertions": self.assertions,
            "query": self.query,
            "expected_status": self.expected_status,
            "answer_extraction": self.answer_extraction,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and clocks for Exp 2967."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    frontier_artifact_path: Path | None = None
    raw_response_dir: Path | None = None
    max_models: int = 1
    random_seed: int = RANDOM_SEED
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    monotonic_clock: Callable[[], float] = time.monotonic

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME

    def frontier_artifact(self) -> Path:
        return self.frontier_artifact_path or self.repo_root / "results" / exp2966.OUTPUT_FILENAME

    def response_dir(self) -> Path:
        return self.raw_response_dir or self.repo_root / "results" / RAW_RESPONSE_DIRNAME


@dataclass(frozen=True)
class Preconditions:
    """Precondition report used to decide whether live inference may run."""

    rows: list[JsonDict]
    z3_import_ok: bool
    frontier_materialized: bool
    cached_sota_pair_used: bool
    model_specs: list[JsonDict]
    frontier_items: list[FrontierItem]
    block_reason: str | None


def load_frontier_items(config: ExperimentConfig) -> list[FrontierItem]:
    """Load the materialized Exp 2966 manifest and return its frontier items."""

    artifact_path = config.frontier_artifact()
    artifact = _read_json(artifact_path)
    if not artifact.get("logic_frontier_materialized"):
        raise ValueError("exp2966_not_materialized")
    manifest_path = Path(str(artifact.get("manifest_path") or ""))
    if not manifest_path.is_file():
        candidate = config.repo_root / manifest_path
        manifest_path = candidate if candidate.is_file() else manifest_path
    manifest = _read_json(manifest_path)
    return frontier_items_from_manifest(manifest)


def frontier_items_from_manifest(manifest: Mapping[str, Any]) -> list[FrontierItem]:
    """Convert the Exp 2966 manifest JSON into typed frontier records."""

    items: list[FrontierItem] = []
    for raw in manifest.get("items") or []:
        reference_z3 = raw.get("reference_z3") or {}
        items.append(
            FrontierItem(
                item_id=str(raw["item_id"]),
                prompt=str(raw["prompt"]),
                expected_label=str(raw["expected_label"]),
                check_kind=str(raw["check_kind"]),
                expected_solver_status=str(raw["expected_solver_status"]),
                skill_labels=tuple(str(label) for label in raw.get("skill_labels") or ()),
                reference_smt2=str(reference_z3.get("assertions") or ""),
                expected_answer_values=dict(raw.get("expected_answer_values") or {}),
            )
        )
    return items


def draft_prompt_for_item(item: FrontierItem) -> str:
    """Return the unconstrained draft prompt for the DCCD first pass."""

    return (
        "Draft a Z3 SMT-LIB formalization plan for this logic item. "
        "Identify variables, predicates, assertions, query, expected solver "
        "status, and any answer-extraction symbols. Do not solve in prose.\n"
        f"Item id: {item.item_id}\n"
        f"Skill labels: {', '.join(item.skill_labels)}\n"
        f"Expected check kind: {item.check_kind}\n"
        f"Problem: {item.prompt}\n"
    )


def structured_prompt_for_item(item: FrontierItem, draft_text: str) -> str:
    """Return the strict structured-output prompt conditioned on a draft."""

    return (
        "Return exactly one JSON object and no prose.\n"
        "The object must have keys: variables, predicates, assertions, query, "
        "expected_status, answer_extraction.\n"
        "variables: list of {name, sort}; predicates: list of {name, signature, returns}; "
        'assertions: list of SMT-LIB declaration/assertion strings; query: "(check-sat)"; '
        'expected_status: "sat" or "unsat"; answer_extraction: {symbols: [...]}.\n'
        "Z3 will execute assertions; your text is not proof.\n"
        f"Item id: {item.item_id}\n"
        f"Problem: {item.prompt}\n"
        f"Draft to condition on:\n{draft_text}\n"
    )


def parse_structured_proposal(text: str) -> tuple[StructuredProposal | None, str | None]:
    """Parse and validate the first strict DCCD structured-output object."""

    obj, error = _extract_json_object(text)
    if error is not None:
        return None, error
    if isinstance(obj.get("formalization"), Mapping):
        obj = dict(obj["formalization"])
    required = (
        "variables",
        "predicates",
        "assertions",
        "query",
        "expected_status",
        "answer_extraction",
    )
    missing = [name for name in required if name not in obj]
    if missing:
        return None, f"missing_schema_field:{','.join(missing)}"
    if not isinstance(obj["variables"], list):
        return None, "variables_not_list"
    if not isinstance(obj["predicates"], list):
        return None, "predicates_not_list"
    assertions = obj["assertions"]
    if (
        not isinstance(assertions, list)
        or not assertions
        or not all(isinstance(assertion, str) and assertion.strip() for assertion in assertions)
    ):
        return None, "assertions_not_nonempty_string_list"
    if not isinstance(obj["query"], str) or not obj["query"].strip():
        return None, "query_not_string"
    expected_status = obj["expected_status"]
    if not isinstance(expected_status, str) or expected_status.strip().lower() not in {
        "sat",
        "unsat",
    }:
        return None, "expected_status_not_sat_or_unsat"
    if not isinstance(obj["answer_extraction"], Mapping):
        return None, "answer_extraction_not_object"
    return (
        StructuredProposal(
            variables=[
                dict(value) if isinstance(value, Mapping) else {"value": value}
                for value in obj["variables"]
            ],
            predicates=[
                dict(value) if isinstance(value, Mapping) else {"value": value}
                for value in obj["predicates"]
            ],
            assertions=[str(assertion) for assertion in assertions],
            query=obj["query"].strip(),
            expected_status=expected_status.strip().lower(),
            answer_extraction=dict(obj["answer_extraction"]),
        ),
        None,
    )


def execute_structured_proposal(
    proposal: StructuredProposal,
    item: FrontierItem,
    *,
    z3_module: Any = _z3,
) -> JsonDict:
    """Execute the parseable proposal in Z3 and compare against Exp 2966 gold."""

    if z3_module is None:
        return _unexecuted_z3_result("z3_unavailable")
    try:
        solver = z3_module.Solver()
        solver.add(z3_module.parse_smt2_string("\n".join(proposal.assertions)))
        actual_status = str(solver.check())
    except Exception as exc:
        return _unexecuted_z3_result(f"{type(exc).__name__}: {exc}")

    expected_values = dict(item.expected_answer_values)
    actual_values: dict[str, str] = {}
    answer_extraction_matches = True
    if actual_status == "sat" and expected_values:
        model = solver.model()
        for symbol_name, expected_value in expected_values.items():
            actual_value = str(model.eval(z3_module.Int(symbol_name), model_completion=True))
            actual_values[symbol_name] = actual_value
            answer_extraction_matches = answer_extraction_matches and actual_value == expected_value
    elif expected_values:
        answer_extraction_matches = False
    return {
        "z3_executed": True,
        "z3_error": None,
        "actual_solver_status": actual_status,
        "expected_solver_status": item.expected_solver_status,
        "proposal_expected_status": proposal.expected_status,
        "solver_status_matches_expected": actual_status == item.expected_solver_status,
        "actual_answer_values": actual_values,
        "expected_answer_values": expected_values,
        "answer_extraction_matches_expected": answer_extraction_matches,
    }


def evaluate_model_row(
    item: FrontierItem,
    raw_text: str,
    *,
    generation_metadata: Mapping[str, Any],
    z3_module: Any = _z3,
) -> JsonDict:
    """Evaluate one model proposal with parsing, Z3 execution, and category labels."""

    proposal, parse_error = parse_structured_proposal(raw_text)
    raw_sha = sha256_text(raw_text)
    base = {
        "item_id": item.item_id,
        "expected_label": item.expected_label,
        "check_kind": item.check_kind,
        "expected_solver_status": item.expected_solver_status,
        "skill_labels": list(item.skill_labels),
        "raw_output_sha256": raw_sha,
        **_generation_fields(generation_metadata),
    }
    if proposal is None:
        return base | {
            "parseable": False,
            "parse_error": parse_error,
            "structured_proposal": None,
            "z3_executed": False,
            "z3_result": _unexecuted_z3_result(parse_error),
            "solver_formula_correct": False,
            "answer_correct": False,
            "failure_category": "unparseable",
        }

    z3_result = execute_structured_proposal(proposal, item, z3_module=z3_module)
    z3_executed = bool(z3_result.get("z3_executed"))
    solver_formula_correct = bool(
        z3_executed
        and z3_result.get("solver_status_matches_expected")
        and z3_result.get("answer_extraction_matches_expected")
    )
    answer_correct = proposal.expected_status == item.expected_solver_status
    if not z3_executed:
        failure_category = "z3_exception"
    elif not solver_formula_correct:
        failure_category = "wrong_formula"
    elif not answer_correct:
        failure_category = "wrong_answer"
    else:
        failure_category = "solver_verified_correct"
    return base | {
        "parseable": True,
        "parse_error": None,
        "structured_proposal": proposal.to_dict(),
        "z3_executed": z3_executed,
        "z3_result": z3_result,
        "solver_formula_correct": solver_formula_correct,
        "answer_correct": answer_correct,
        "failure_category": failure_category,
    }


def aggregate_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute aggregate parseability, execution, solver, answer, and failures."""

    counts = _failure_counts(rows)
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


def skill_wise_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute the same metrics after expanding each row by skill label."""

    metrics: JsonDict = {}
    for skill in SKILL_LABELS:
        skill_rows = [row for row in rows if skill in (row.get("skill_labels") or [])]
        aggregate = aggregate_results(skill_rows)
        metrics[skill] = {"n_items": len(skill_rows), **aggregate}
    return metrics


def formalization_delta_clean(metrics: Mapping[str, Any]) -> bool:
    """Return the explicit .278 delta gate result."""

    return bool(
        float(metrics.get("parseability_rate") or 0.0) >= 0.50
        and float(metrics.get("z3_execution_rate") or 0.0) >= 0.50
        and float(metrics.get("solver_verified_accuracy") or 0.0) > 0.0
    )


def check_preconditions(
    config: ExperimentConfig,
    *,
    cached_pair_provider: CachedPairProvider = cached_sota_pair,
    individual_model_resolver: IndividualResolver = resolve_cached_gguf,
    module_importer: ModuleImporter = importlib.import_module,
    z3_module: Any = _z3,
) -> Preconditions:
    """Check Exp 2966, Z3, llama.cpp, and mandated local GGUF availability."""

    rows: list[JsonDict] = []
    frontier_items: list[FrontierItem] = []
    frontier_materialized = False
    try:
        frontier_items = load_frontier_items(config)
        frontier_materialized = bool(frontier_items)
        rows.append(
            {
                "name": "exp2966_logic_frontier_materialized",
                "ok": frontier_materialized,
                "detail": f"{len(frontier_items)} item(s)",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "name": "exp2966_logic_frontier_materialized",
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

    cached_pair_used = False
    model_specs: list[JsonDict] = []
    if pair:
        model_specs = [dict(spec) for spec in pair if spec.get("hf_id") in MANDATED_MODEL_IDS]
        cached_pair_used = bool(model_specs)
    if not model_specs:
        for hf_id in MANDATED_MODEL_IDS:
            path = individual_model_resolver(hf_id)
            if path:
                spec = dict(_SPEC_BY_HF_ID[hf_id])
                spec["model_path"] = str(path)
                model_specs.append(spec)
    rows.append(
        {
            "name": "mandated_headline_gguf_resolved",
            "ok": bool(model_specs),
            "detail": ",".join(str(spec.get("hf_id")) for spec in model_specs) or "not_cached",
        }
    )

    block_reason = None
    if not frontier_materialized:
        block_reason = "exp2966_not_materialized"
    elif not z3_import_ok:
        block_reason = "z3_import_failed"
    elif not llama_ok:
        block_reason = "llama_cpp_import_failed"
    elif not model_specs:
        block_reason = "headline_gguf_missing"
    return Preconditions(
        rows=rows,
        z3_import_ok=z3_import_ok,
        frontier_materialized=frontier_materialized,
        cached_sota_pair_used=cached_pair_used,
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
    """Run the live DCCD formalization benchmark and write the artifact."""

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

    items = preconditions.frontier_items
    item_by_id = {item.item_id: item for item in items}
    collector = collect_model_outputs_fn or collect_live_dccd_outputs
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
        collection = collector(spec, items, active)
        model_attempts.append(dict(collection.get("summary") or {}))
        collected_rows.extend(dict(row) for row in collection.get("rows") or [])

    raw_by_id = {str(row.get("item_id")): row for row in collected_rows if row.get("item_id")}
    per_item_results = []
    for item_id, item in item_by_id.items():
        raw_row = raw_by_id.get(item_id, {})
        per_item_results.append(
            evaluate_model_row(
                item,
                str(raw_row.get("output_text") or ""),
                generation_metadata=raw_row,
                z3_module=z3_module,
            )
        )

    metrics = aggregate_results(per_item_results)
    item_manifest = [_item_manifest_row(item) for item in items]
    manifest_sha = formalization_manifest_sha256(item_manifest, per_item_results)
    delta_clean = formalization_delta_clean(metrics)
    artifact = {
        "honest_verdict": (
            "complete: local SOTA DCCD formalizations improve over .278 baseline"
            if delta_clean
            else "complete: local SOTA DCCD formalizations did not clear .278 delta gate"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions.rows,
        "model_specs": preconditions.model_specs or [dict(spec) for spec in MODEL_SPECS],
        "headline_models_used": _headline_models_used(per_item_results),
        "legacy_models_only_for_smoke": False,
        "n_items": len(items),
        "parseability_rate": metrics["parseability_rate"],
        "z3_execution_rate": metrics["z3_execution_rate"],
        "solver_verified_accuracy": metrics["solver_verified_accuracy"],
        "answer_accuracy": metrics["answer_accuracy"],
        "baseline_parseability_rate": BASELINE_PARSEABILITY_RATE,
        "baseline_solver_verified_accuracy": BASELINE_SOLVER_VERIFIED_ACCURACY,
        "skill_wise_metrics": skill_wise_metrics(per_item_results),
        "failure_categories": metrics["failure_categories"],
        "formalization_delta_clean": delta_clean,
        "formalization_manifest_sha256": manifest_sha,
        "duration_s": round(active.clock() - started, 6),
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "item_manifest": item_manifest,
        "per_item_results": per_item_results,
        "model_attempts": model_attempts,
        "cached_sota_pair_used": preconditions.cached_sota_pair_used,
        "raw_response_dir": str(active.response_dir()),
    }
    validate_artifact(artifact)
    _write_json(active.artifact_path(), artifact)
    return artifact


def collect_live_dccd_outputs(
    spec: JsonDict,
    items: list[FrontierItem],
    config: ExperimentConfig,
    *,
    llama_importer: LlamaImporter | None = None,
) -> JsonDict:
    """Collect DCCD draft and strict structured proposals from one local GGUF."""

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
            draft_text = ""
            output_text = ""
            try:
                draft_result = llm(
                    draft_prompt_for_item(item),
                    max_tokens=384,
                    temperature=0.2,
                    top_p=1.0,
                    seed=config.random_seed + index,
                    stop=["</s>", "<eos>"],
                    echo=False,
                )
                draft_text = completion_text(draft_result)
                structured_result = llm(
                    structured_prompt_for_item(item, draft_text),
                    max_tokens=1024,
                    temperature=0.0,
                    top_p=1.0,
                    seed=config.random_seed + 1000 + index,
                    stop=["</s>", "<eos>"],
                    echo=False,
                )
                output_text = completion_text(structured_result)
                if not output_text.strip():
                    blocker = "empty_generation"
            except Exception as exc:
                blocker = f"{type(exc).__name__}: {exc}"
            raw_path = config.response_dir() / f"{item.item_id}.json"
            raw_payload = {"draft_text": draft_text, "structured_output": output_text}
            raw_path.write_text(
                json.dumps(raw_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            rows.append(
                {
                    "item_id": item.item_id,
                    "model_hf_id": hf_id,
                    "model_name": spec.get("name"),
                    "model_path": model_path,
                    "gpu_index": spec.get("gpu"),
                    "prompt_hash": sha256_text(item.prompt),
                    "per_item_seed": config.random_seed + index,
                    "generation_source": "live_sota_dccd_structured_output",
                    "draft_text": draft_text,
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


def formalization_manifest_sha256(
    item_manifest: Sequence[Mapping[str, Any]],
    per_item_results: Sequence[Mapping[str, Any]],
) -> str:
    """Hash items, structured proposals, Z3 outcomes, skills, and categories."""

    payload = {
        "item_manifest": list(item_manifest),
        "proposal_results": [
            {
                "item_id": row.get("item_id"),
                "skill_labels": row.get("skill_labels"),
                "structured_proposal": row.get("structured_proposal"),
                "z3_result": row.get("z3_result"),
                "failure_category": row.get("failure_category"),
                "raw_output_sha256": row.get("raw_output_sha256"),
            }
            for row in per_item_results
        ],
    }
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal artifact fields used by conductor checks."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be live_llm_inference")
    recorded_ids = {str(spec.get("hf_id")) for spec in artifact.get("model_specs", [])}
    if not (recorded_ids & set(MANDATED_MODEL_IDS)):
        raise ValueError("model_specs must include at least one mandated headline GGUF")
    expected_delta = formalization_delta_clean(artifact)
    if bool(artifact["formalization_delta_clean"]) != expected_delta:
        raise ValueError("formalization_delta_clean does not match explicit gate")


def completion_text(result: Any) -> str:
    """Extract text from common llama.cpp completion response shapes."""

    if isinstance(result, str):
        return result
    if not isinstance(result, Mapping):
        return ""
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return ""
    text = choice.get("text")
    if isinstance(text, str):
        return text
    message = choice.get("message")
    if isinstance(message, Mapping) and isinstance(message.get("content"), str):
        return str(message["content"])
    return ""


def sha256_text(text: str) -> str:
    """Return a SHA-256 digest for UTF-8 text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _blocked_artifact(
    config: ExperimentConfig,
    preconditions: Preconditions,
    duration_s: float,
) -> JsonDict:
    metrics = aggregate_results([])
    artifact = {
        "honest_verdict": f"blocked_precondition: {preconditions.block_reason}",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions.rows,
        "model_specs": preconditions.model_specs or [dict(spec) for spec in MODEL_SPECS],
        "headline_models_used": [],
        "legacy_models_only_for_smoke": False,
        "n_items": 0,
        "parseability_rate": metrics["parseability_rate"],
        "z3_execution_rate": metrics["z3_execution_rate"],
        "solver_verified_accuracy": metrics["solver_verified_accuracy"],
        "answer_accuracy": metrics["answer_accuracy"],
        "baseline_parseability_rate": BASELINE_PARSEABILITY_RATE,
        "baseline_solver_verified_accuracy": BASELINE_SOLVER_VERIFIED_ACCURACY,
        "skill_wise_metrics": skill_wise_metrics([]),
        "failure_categories": metrics["failure_categories"],
        "formalization_delta_clean": False,
        "formalization_manifest_sha256": "",
        "duration_s": round(duration_s, 6),
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "item_manifest": [],
        "per_item_results": [],
        "model_attempts": [],
        "cached_sota_pair_used": preconditions.cached_sota_pair_used,
        "raw_response_dir": str(config.response_dir()),
    }
    validate_artifact(artifact)
    return artifact


def _item_manifest_row(item: FrontierItem) -> JsonDict:
    return {
        "item_id": item.item_id,
        "prompt": item.prompt,
        "expected_label": item.expected_label,
        "check_kind": item.check_kind,
        "expected_solver_status": item.expected_solver_status,
        "skill_labels": list(item.skill_labels),
        "expected_answer_values": dict(item.expected_answer_values),
    }


def _failure_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    counts = {name: 0 for name in FAILURE_CATEGORY_NAMES}
    for row in rows:
        category = str(row.get("failure_category") or "unparseable")
        counts[category if category in counts else "unparseable"] += 1
    return counts


def _extract_json_object(text: str) -> tuple[JsonDict, str | None]:
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
    return {}, "no_json_object"


def _unexecuted_z3_result(error: str | None) -> JsonDict:
    return {
        "z3_executed": False,
        "z3_error": error,
        "actual_solver_status": None,
        "expected_solver_status": None,
        "proposal_expected_status": None,
        "solver_status_matches_expected": False,
        "actual_answer_values": {},
        "expected_answer_values": {},
        "answer_extraction_matches_expected": False,
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
        "draft_sha256": sha256_text(str(metadata.get("draft_text") or "")),
        "raw_response_path": metadata.get("raw_response_path"),
        "elapsed_seconds": metadata.get("elapsed_seconds"),
    }


def _headline_models_used(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    seen = []
    for row in rows:
        hf_id = row.get("model_hf_id")
        if (
            hf_id in MANDATED_MODEL_IDS
            and hf_id not in seen
            and not row.get("generation_blocker")
            and row.get("raw_output_sha256")
        ):
            seen.append(str(hf_id))
    return seen


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
        "[exp2967] "
        f"verdict={artifact['honest_verdict']} "
        f"items={artifact['n_items']} "
        f"parse={artifact['parseability_rate']} "
        f"z3={artifact['z3_execution_rate']} "
        f"solver={artifact['solver_verified_accuracy']} "
        f"delta_clean={artifact['formalization_delta_clean']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
