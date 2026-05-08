"""Exp 1511 product-line solver oracle benchmark.

Spec: REQ-BENCH-1511, SCENARIO-BENCH-1511.

This benchmark keeps the product-line task small enough for exhaustive
enumeration.  That matters because product-line feature models are mostly a
feasibility problem: a model can parse the feature names correctly and still
choose a configuration that violates a requires/excludes clause.  The local
oracle therefore enumerates every bounded configuration before it scores the
model answer, so no LLM judgment is needed for headline metrics.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from carnot.eval import cctu_executable_constraint_microbenchmark as cctu

JsonDict = dict[str, Any]

RUN_DATE = "20260508"
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1511_product_line_solver_oracle_benchmark.json")
DEFAULT_MANIFEST_PATH = Path("results/product_line_solver_oracle_1511.jsonl")

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "name": "Qwen3.6-35B-A3B",
        "role": "flagship_moe_primary_feature_model_solver",
        "gpu": 0,
    },
    {
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "name": "Gemma4-31B-it",
        "role": "flagship_dense_secondary_feature_model_solver",
        "gpu": 1,
    },
    {
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "name": "Gemma4-26B-A4B-it",
        "role": "middle_moe_secondary_feature_model_solver",
        "gpu": 1,
    },
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "model_specs",
    "live_sota_model_inference_used",
    "product_line_benchmark_ready",
    "feature_models_defined",
    "cases_attempted",
    "parse_rate",
    "solver_oracle_ready",
    "feasibility_rate",
    "oracle_agreement_rate",
    "verifier_false_accept_rate",
    "benchmark_manifest_path",
    "models_used",
    "gpu_probe",
    "blockers",
    "honest_verdict",
)


@dataclass(frozen=True)
class FeatureModel:
    """Parsed feature-model structure used by the deterministic oracle."""

    model_id: str
    features: frozenset[str]
    mandatory: frozenset[str]
    optional: frozenset[str]
    requires: tuple[tuple[str, str], ...]
    excludes: tuple[tuple[str, str], ...]
    costs: dict[str, int]
    values: dict[str, int]


@dataclass(frozen=True)
class AnalysisOperation:
    """One bounded solver request over a feature model."""

    kind: str
    include: frozenset[str]
    budget: int | None = None


@dataclass(frozen=True)
class ProductLineCase:
    """One semi-formal product-line blueprint plus expected analysis operation."""

    case_id: str
    model: FeatureModel
    operation: AnalysisOperation
    blueprint_text: str
    prompt: str


@dataclass(frozen=True)
class FeasibilityCheck:
    """Boolean feasibility result with concrete violation reasons."""

    ok: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class OracleResult:
    """Deterministic exhaustive solution for one product-line case."""

    feasible_exists: bool
    feasible_count: int
    optimal_features: tuple[str, ...]
    optimal_cost: int
    optimal_value: int
    optimal_feature_sets: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class ParsedModelAnswer:
    """Structured model answer extracted from a free-form completion."""

    parse_ok: bool
    selected_features: tuple[str, ...]
    model_declared_accept: bool | None
    objective_cost: int | None = None
    objective_value: int | None = None
    parse_error: str | None = None


@dataclass(frozen=True)
class SelectionEvaluation:
    """Oracle classification for a parsed feature selection."""

    classification: str
    feasible: bool
    oracle_agrees: bool
    cost: int | None
    value: int | None
    reasons: tuple[str, ...]


CollectModelOutputsFn = Callable[[JsonDict, list[ProductLineCase]], JsonDict]
CachedPairFn = Callable[..., list[JsonDict] | None]
LlamaImporterFn = Callable[[], tuple[bool, type[Any] | None, str | None]]
ResolverFn = Callable[[str], str | None]


BLUEPRINT_TEXTS: tuple[str, ...] = (
    """CASE: plc-1511-retail-budget
MODEL: RetailCheckout
FEATURES:
- Store mandatory cost=0 value=0
- Catalog mandatory cost=1 value=2
- Checkout mandatory cost=2 value=4
- Loyalty optional cost=2 value=5
- Coupons optional cost=2 value=8
- Fraud optional cost=3 value=6
- ExpressShipping optional cost=4 value=7
- CryptoPay optional cost=3 value=6
REQUIRES:
- Coupons -> Loyalty
- ExpressShipping -> Fraud
- CryptoPay -> Fraud
EXCLUDES:
- CryptoPay x Coupons
OPERATION: max_value budget=12 include=Store
""",
    """CASE: plc-1511-drone-min-cost
MODEL: DronePackage
FEATURES:
- Drone mandatory cost=0 value=0
- Airframe mandatory cost=3 value=4
- Controller mandatory cost=2 value=3
- Camera optional cost=4 value=5
- Thermal optional cost=3 value=6
- NightMode optional cost=2 value=4
- LongRange optional cost=5 value=7
- HeavyLift optional cost=4 value=5
- EncryptedLink optional cost=2 value=3
REQUIRES:
- Thermal -> Camera
- NightMode -> Camera
- HeavyLift -> LongRange
EXCLUDES:
- HeavyLift x Camera
OPERATION: min_cost include=Thermal,NightMode
""",
    """CASE: plc-1511-clinic-budget
MODEL: ClinicPortal
FEATURES:
- Clinic mandatory cost=0 value=0
- Portal mandatory cost=2 value=3
- Scheduling mandatory cost=2 value=4
- SMS optional cost=1 value=2
- Telehealth optional cost=4 value=7
- Insurance optional cost=3 value=5
- Payments optional cost=2 value=4
- Analytics optional cost=4 value=6
- ResearchExport optional cost=3 value=5
REQUIRES:
- Telehealth -> SMS
- Payments -> Insurance
- ResearchExport -> Analytics
EXCLUDES:
- ResearchExport x Insurance
OPERATION: max_value budget=13 include=Telehealth
""",
    """CASE: plc-1511-vehicle-min-cost
MODEL: VehicleVariant
FEATURES:
- Vehicle mandatory cost=0 value=0
- Chassis mandatory cost=3 value=4
- Engine mandatory cost=4 value=5
- TowPackage optional cost=3 value=5
- Trailer optional cost=5 value=8
- Offroad optional cost=4 value=6
- SportTune optional cost=3 value=6
- Hybrid optional cost=4 value=7
REQUIRES:
- Trailer -> TowPackage
EXCLUDES:
- SportTune x TowPackage
- Hybrid x SportTune
OPERATION: min_cost include=Trailer,Offroad
""",
    """CASE: plc-1511-media-budget
MODEL: MediaApp
FEATURES:
- App mandatory cost=0 value=0
- Player mandatory cost=2 value=4
- Library mandatory cost=2 value=4
- DRM optional cost=2 value=4
- Store optional cost=3 value=5
- FamilySharing optional cost=2 value=4
- OfflineMode optional cost=3 value=7
- CloudSync optional cost=2 value=4
- Ads optional cost=1 value=-2
REQUIRES:
- Store -> DRM
- OfflineMode -> DRM
EXCLUDES:
- Ads x FamilySharing
OPERATION: max_value budget=11 include=OfflineMode
""",
    """CASE: plc-1511-factory-budget
MODEL: FactoryOps
FEATURES:
- Factory mandatory cost=0 value=0
- Sensors mandatory cost=3 value=4
- Dashboard mandatory cost=2 value=4
- PLCIntegration optional cost=4 value=6
- PredictiveMaintenance optional cost=5 value=9
- AIPlanner optional cost=4 value=8
- RemoteAccess optional cost=3 value=5
- AirGap optional cost=2 value=3
- AuditLog optional cost=2 value=4
REQUIRES:
- PredictiveMaintenance -> Sensors
- AIPlanner -> PredictiveMaintenance
EXCLUDES:
- AirGap x RemoteAccess
OPERATION: max_value budget=16 include=PredictiveMaintenance
""",
)


def parse_blueprint(text: str) -> ProductLineCase:
    """Parse the intentionally small semi-formal feature-model grammar."""

    case_id: str | None = None
    model_id: str | None = None
    section: str | None = None
    features: set[str] = set()
    mandatory: set[str] = set()
    optional: set[str] = set()
    requires: list[tuple[str, str]] = []
    excludes: list[tuple[str, str]] = []
    costs: dict[str, int] = {}
    values: dict[str, int] = {}
    operation: AnalysisOperation | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("CASE:"):
            case_id = line.split(":", 1)[1].strip()
            continue
        if line.startswith("MODEL:"):
            model_id = line.split(":", 1)[1].strip()
            continue
        if line in {"FEATURES:", "REQUIRES:", "EXCLUDES:"}:
            section = line[:-1]
            continue
        if line.startswith("OPERATION:"):
            operation = _parse_operation(line.split(":", 1)[1].strip())
            section = None
            continue
        if line.startswith("-"):
            if section is None:
                raise ValueError("feature before FEATURES or constraint section")
            item = line[1:].strip()
            if section == "FEATURES":
                name, kind, cost, value = _parse_feature_line(item)
                features.add(name)
                costs[name] = cost
                values[name] = value
                if kind == "mandatory":
                    mandatory.add(name)
                elif kind == "optional":
                    optional.add(name)
                else:
                    raise ValueError(f"unknown feature kind: {kind}")
            elif section == "REQUIRES":
                requires.append(_parse_constraint_pair(item, "->"))
            elif section == "EXCLUDES":
                excludes.append(_parse_constraint_pair(item, "x"))
            continue
        raise ValueError(f"unrecognised blueprint line: {line}")  # pragma: no cover

    if not case_id:
        raise ValueError("missing CASE")
    if not model_id:
        raise ValueError("missing MODEL")
    if not features:
        raise ValueError("missing FEATURES")  # pragma: no cover
    if operation is None:
        raise ValueError("missing OPERATION")  # pragma: no cover

    known = set(features)
    for source, target in [*requires, *excludes, *((feature, "") for feature in operation.include)]:
        if source and source not in known:
            raise ValueError(f"unknown feature reference: {source}")  # pragma: no cover
        if target and target not in known:
            raise ValueError(f"unknown feature reference: {target}")  # pragma: no cover

    model = FeatureModel(
        model_id=model_id,
        features=frozenset(features),
        mandatory=frozenset(mandatory),
        optional=frozenset(optional),
        requires=tuple(requires),
        excludes=tuple(excludes),
        costs=costs,
        values=values,
    )
    return ProductLineCase(
        case_id=case_id,
        model=model,
        operation=operation,
        blueprint_text=text,
        prompt=_build_prompt(case_id, text),
    )


def build_feature_model_cases() -> list[ProductLineCase]:
    """Return the fixed six-case product-line benchmark suite."""

    return [parse_blueprint(text) for text in BLUEPRINT_TEXTS]


def solve_case(case: ProductLineCase) -> OracleResult:
    """Exhaustively enumerate the bounded feature model and choose the optimum."""

    candidates: list[frozenset[str]] = []
    for subset_size in range(len(case.model.optional) + 1):
        for subset in itertools.combinations(sorted(case.model.optional), subset_size):
            selection = frozenset(case.model.mandatory | set(subset))
            if not is_selection_feasible(case.model, selection).ok:
                continue
            if not selection_satisfies_operation(case, selection).ok:
                continue
            candidates.append(selection)

    if not candidates:
        return OracleResult(False, 0, (), 0, 0, ())

    if case.operation.kind == "max_value":
        best_key = max(_max_value_key(case.model, selection) for selection in candidates)
        optimal = [
            selection
            for selection in candidates
            if _max_value_key(case.model, selection) == best_key
        ]
    elif case.operation.kind == "min_cost":
        best_key = min(_min_cost_key(case.model, selection) for selection in candidates)
        optimal = [
            selection
            for selection in candidates
            if _min_cost_key(case.model, selection) == best_key
        ]
    else:
        raise ValueError(f"unsupported operation: {case.operation.kind}")  # pragma: no cover

    optimal_sets = tuple(tuple(sorted(selection)) for selection in sorted(optimal, key=sorted))
    first = optimal_sets[0]
    return OracleResult(
        feasible_exists=True,
        feasible_count=len(candidates),
        optimal_features=first,
        optimal_cost=selection_cost(case.model, first),
        optimal_value=selection_value(case.model, first),
        optimal_feature_sets=optimal_sets,
    )


def is_selection_feasible(model: FeatureModel, selection: Iterable[str]) -> FeasibilityCheck:
    """Check product-line structural feasibility for one selected feature set."""

    selected = frozenset(selection)
    reasons: list[str] = []
    unknown = sorted(selected - model.features)
    missing_mandatory = sorted(model.mandatory - selected)
    if unknown:
        reasons.append(f"unknown:{','.join(unknown)}")
    if missing_mandatory:
        reasons.append(f"missing_mandatory:{','.join(missing_mandatory)}")
    for source, target in model.requires:
        if source in selected and target not in selected:
            reasons.append(f"requires:{source}->{target}")
    for left, right in model.excludes:
        if left in selected and right in selected:
            reasons.append(f"excludes:{left}x{right}")
    return FeasibilityCheck(ok=not reasons, reasons=tuple(reasons))


def selection_satisfies_operation(
    case: ProductLineCase, selection: Iterable[str]
) -> FeasibilityCheck:
    """Check analysis-operation hard constraints such as include and budget."""

    selected = frozenset(selection)
    reasons: list[str] = []
    missing_include = sorted(case.operation.include - selected)
    if missing_include:
        reasons.append(f"missing_include:{','.join(missing_include)}")
    if case.operation.budget is not None:
        cost = selection_cost(case.model, selected)
        if cost > case.operation.budget:
            reasons.append(f"budget:{cost}>{case.operation.budget}")
    return FeasibilityCheck(ok=not reasons, reasons=tuple(reasons))


def evaluate_selection(case: ProductLineCase, selection: Iterable[str]) -> SelectionEvaluation:
    """Classify a parsed feature selection against feasibility and oracle optimality."""

    selected = frozenset(selection)
    structural = is_selection_feasible(case.model, selected)
    operational = selection_satisfies_operation(case, selected)
    reasons = (*structural.reasons, *operational.reasons)
    if reasons:
        return SelectionEvaluation("infeasible", False, False, None, None, reasons)

    oracle = solve_case(case)
    canonical = tuple(sorted(selected))
    agrees = canonical in oracle.optimal_feature_sets
    return SelectionEvaluation(
        classification="oracle_agreement" if agrees else "wrong_or_suboptimal",
        feasible=True,
        oracle_agrees=agrees,
        cost=selection_cost(case.model, selected),
        value=selection_value(case.model, selected),
        reasons=() if agrees else ("not_oracle_optimal",),
    )


def parse_model_answer(text: str) -> ParsedModelAnswer:
    """Extract and validate the JSON answer shape expected from the model."""

    obj = cctu.extract_json_object(text)
    if obj is None:
        return ParsedModelAnswer(False, (), None, parse_error="no_json_object")
    selected = obj.get("selected_features")
    if not isinstance(selected, list):
        return ParsedModelAnswer(False, (), _declared_accept(obj), parse_error="selected_features_not_list")
    if not all(isinstance(feature, str) for feature in selected):
        return ParsedModelAnswer(False, (), _declared_accept(obj), parse_error="selected_feature_not_string")
    return ParsedModelAnswer(
        parse_ok=True,
        selected_features=tuple(sorted(dict.fromkeys(selected))),
        model_declared_accept=_declared_accept(obj),
        objective_cost=_optional_int(obj.get("objective_cost")),
        objective_value=_optional_int(obj.get("objective_value")),
    )


def compliant_answer_for_case(case: ProductLineCase) -> str:
    """Return a gold JSON answer used by tests and oracle sanity checks."""

    oracle = solve_case(case)
    payload = {
        "selected_features": list(oracle.optimal_features),
        "objective_cost": oracle.optimal_cost,
        "objective_value": oracle.optimal_value,
        "verifier": {"accept": True},
    }
    return json.dumps(payload, sort_keys=True)


def build_manifest_row(case: ProductLineCase, generation_row: JsonDict) -> JsonDict:
    """Join raw model output with deterministic oracle validation."""

    output_text = str(generation_row.get("output_text") or "")
    parsed = parse_model_answer(output_text)
    if not parsed.parse_ok:
        classification = "parse_failure"
        evaluation = SelectionEvaluation(classification, False, False, None, None, (parsed.parse_error or "parse_error",))
    else:
        evaluation = evaluate_selection(case, parsed.selected_features)
        classification = evaluation.classification

    false_accept = parsed.model_declared_accept is True and classification != "oracle_agreement"
    return {
        "case_id": case.case_id,
        "model_id": case.model.model_id,
        "operation": {
            "kind": case.operation.kind,
            "budget": case.operation.budget,
            "include": sorted(case.operation.include),
        },
        "prompt": case.prompt,
        "model_hf_id": generation_row.get("model_hf_id"),
        "model_name": generation_row.get("model_name"),
        "generation_source": generation_row.get("generation_source"),
        "elapsed_seconds": generation_row.get("elapsed_seconds"),
        "blocker": generation_row.get("blocker"),
        "model_output": output_text,
        "parse_result": {
            "parse_ok": parsed.parse_ok,
            "parse_error": parsed.parse_error,
            "selected_features": list(parsed.selected_features),
            "model_declared_accept": parsed.model_declared_accept,
            "objective_cost": parsed.objective_cost,
            "objective_value": parsed.objective_value,
        },
        "oracle_result": {
            "classification": evaluation.classification,
            "feasible": evaluation.feasible,
            "oracle_agrees": evaluation.oracle_agrees,
            "selection_cost": evaluation.cost,
            "selection_value": evaluation.value,
            "reasons": list(evaluation.reasons),
            "optimal_features": list(solve_case(case).optimal_features),
        },
        "verifier_result": {
            "accepted": classification == "oracle_agreement",
            "self_verifier_false_accept": false_accept,
        },
    }


def aggregate_manifest_metrics(rows: list[JsonDict]) -> JsonDict:
    """Compute parse, feasibility, agreement, and false-accept rates."""

    if not rows:
        return {
            "parse_rate": 0.0,
            "feasibility_rate": 0.0,
            "oracle_agreement_rate": 0.0,
            "verifier_false_accept_rate": 0.0,
            "classification_counts": {},
        }

    total = len(rows)
    parsed_rows = [row for row in rows if bool(row["parse_result"]["parse_ok"])]
    feasible_rows = [row for row in parsed_rows if bool(row["oracle_result"]["feasible"])]
    agreement_rows = [row for row in rows if bool(row["oracle_result"]["oracle_agrees"])]
    invalid_rows = [row for row in rows if not bool(row["oracle_result"]["oracle_agrees"])]
    false_accepts = [
        row for row in invalid_rows if bool(row["verifier_result"]["self_verifier_false_accept"])
    ]
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["oracle_result"]["classification"])
        counts[key] = counts.get(key, 0) + 1
    return {
        "parse_rate": round(len(parsed_rows) / total, 6),
        "feasibility_rate": round(len(feasible_rows) / len(parsed_rows), 6) if parsed_rows else 0.0,
        "oracle_agreement_rate": round(len(agreement_rows) / total, 6),
        "verifier_false_accept_rate": (
            round(len(false_accepts) / len(invalid_rows), 6) if invalid_rows else 0.0
        ),
        "classification_counts": counts,
    }


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Write the durable in-progress artifact required by the experiment prompt."""

    payload: JsonDict = {
        "status": "in_progress",
        "run_date": run_date,
        "model_specs": [spec["hf_id"] for spec in MANDATED_MODEL_SPECS],
        "live_sota_model_inference_used": False,
        "product_line_benchmark_ready": False,
        "feature_models_defined": 0,
        "cases_attempted": 0,
        "parse_rate": 0.0,
        "solver_oracle_ready": False,
        "feasibility_rate": 0.0,
        "oracle_agreement_rate": 0.0,
        "verifier_false_accept_rate": 0.0,
        "benchmark_manifest_path": _display_path(DEFAULT_MANIFEST_PATH),
        "models_used": [],
        "gpu_probe": {},
        "blockers": [],
        "honest_verdict": "in_progress: product-line benchmark artifact initialized",
    }
    _write_json(Path(output_path), payload)
    return payload


def run_benchmark(
    *,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    run_date: str = RUN_DATE,
    model_specs: Iterable[JsonDict] = MANDATED_MODEL_SPECS,
    collect_model_outputs_fn: CollectModelOutputsFn | None = None,
    cached_pair_fn: CachedPairFn | None = None,
    gpu_probe_fn: Callable[[], JsonDict] | None = None,
    max_models: int = 1,
    tests_run: list[str] | None = None,
) -> JsonDict:
    """Run the benchmark, then write the JSONL manifest and final artifact."""

    output = Path(output_path)
    manifest = Path(manifest_path)
    write_in_progress_artifact(output, run_date=run_date)

    cases = build_feature_model_cases()
    solver_oracle_ready = all(solve_case(case).feasible_exists for case in cases)
    specs, cached_pair_details, cached_pair_error = _resolve_headline_specs(
        [dict(spec) for spec in model_specs], cached_pair_fn
    )
    collector = collect_model_outputs_fn or collect_live_model_outputs
    rows: list[JsonDict] = []
    model_attempts: list[JsonDict] = []
    case_by_id = {case.case_id: case for case in cases}

    for index, spec in enumerate(specs):
        if index >= max_models:
            model_attempts.append(
                {
                    "hf_id": spec.get("hf_id"),
                    "model_name": spec.get("name"),
                    "model_used": False,
                    "blocker": "not_attempted_runtime_budget",
                }
            )
            continue
        collection = collector(spec, cases)
        summary = dict(collection.get("summary") or {})
        model_attempts.append(summary)
        for generation_row in collection.get("rows") or []:
            case = case_by_id.get(generation_row.get("case_id"))
            if case is not None:
                rows.append(build_manifest_row(case, generation_row))

    _write_jsonl(manifest, rows)
    metrics = aggregate_manifest_metrics(rows)
    models_used = [
        str(summary["hf_id"])
        for summary in model_attempts
        if summary.get("model_used") is True and summary.get("hf_id")
    ]
    mandated = {str(spec["hf_id"]) for spec in MANDATED_MODEL_SPECS}
    live_used = any(
        row.get("generation_source") == "live_sota_llamacpp"
        and not row.get("blocker")
        and row.get("model_hf_id") in mandated
        for row in rows
    )
    ready = solver_oracle_ready and bool(rows) and live_used
    blockers = _collect_blockers(model_attempts, cached_pair_error)
    status = "complete" if ready else "blocked"
    artifact: JsonDict = {
        "status": status,
        "run_date": run_date,
        "schema_version": 1,
        "model_specs": [spec["hf_id"] for spec in MANDATED_MODEL_SPECS],
        "live_sota_model_inference_used": bool(live_used),
        "product_line_benchmark_ready": bool(ready),
        "feature_models_defined": len(cases),
        "cases_attempted": len(rows),
        "parse_rate": metrics["parse_rate"],
        "solver_oracle_ready": bool(solver_oracle_ready),
        "feasibility_rate": metrics["feasibility_rate"],
        "oracle_agreement_rate": metrics["oracle_agreement_rate"],
        "verifier_false_accept_rate": metrics["verifier_false_accept_rate"],
        "benchmark_manifest_path": _display_path(manifest),
        "models_used": models_used,
        "gpu_probe": (gpu_probe_fn or probe_gpu)(),
        "blockers": blockers,
        "honest_verdict": (
            "complete: product-line solver oracle benchmark ready with live local SOTA GGUF rows"
            if ready
            else "complete_blocked: product-line solver oracle written but live SOTA rows unavailable"
        ),
        "classification_counts": metrics["classification_counts"],
        "model_attempts": model_attempts,
        "cached_sota_pair": cached_pair_details,
        "tests_run": list(tests_run or []),
    }
    _write_json(output, artifact)
    return artifact


def collect_live_model_outputs(
    spec: JsonDict,
    cases: list[ProductLineCase],
    *,
    resolver: ResolverFn | None = None,
    llama_importer: LlamaImporterFn | None = None,
    env_preparer: Callable[[], JsonDict] | None = None,
) -> JsonDict:
    """Collect raw outputs from one mandated local GGUF model through llama.cpp."""

    hf_id = str(spec.get("hf_id") or "")
    resolver_fn = resolver or _default_resolver
    model_path = spec.get("model_path") or resolver_fn(hf_id)
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

    env_details = (env_preparer or cctu.prepare_llama_environment)()
    ok, llama_class, import_error = (llama_importer or cctu._default_llama_importer)()
    if not ok or llama_class is None:
        return {
            "summary": {
                "hf_id": hf_id,
                "model_name": spec.get("name"),
                "model_path": str(model_path),
                "model_used": False,
                "blocker": import_error or "llama_cpp_import_failed",
                "env_details": env_details,
            },
            "rows": [],
        }

    llm = None
    rows: list[JsonDict] = []
    load_start = time.monotonic()
    try:
        llm = llama_class(
            model_path=str(model_path),
            n_gpu_layers=-1,
            main_gpu=int(spec.get("gpu") or 0),
            n_ctx=4096,
            seed=1511,
            verbose=False,
        )
    except Exception as exc:
        return {
            "summary": {
                "hf_id": hf_id,
                "model_name": spec.get("name"),
                "model_path": str(model_path),
                "model_used": False,
                "blocker": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.monotonic() - load_start, 6),
                "env_details": env_details,
            },
            "rows": [],
        }

    try:
        for case in cases:
            started = time.monotonic()
            try:
                result = llm(
                    case.prompt,
                    max_tokens=256,
                    temperature=0.0,
                    top_p=1.0,
                    stop=["</s>", "<eos>"],
                    echo=False,
                )
                output_text = cctu._completion_text(result)
                blocker = None if output_text.strip() else "empty_generation"
            except Exception as exc:
                output_text = ""
                blocker = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "case_id": case.case_id,
                    "model_hf_id": hf_id,
                    "model_name": spec.get("name"),
                    "model_path": str(model_path),
                    "generation_source": "live_sota_llamacpp",
                    "output_text": output_text,
                    "elapsed_seconds": round(time.monotonic() - started, 6),
                    "blocker": blocker,
                }
            )
    finally:
        cctu._close_llama(llm)

    model_used = any(row.get("blocker") is None for row in rows)
    return {
        "summary": {
            "hf_id": hf_id,
            "model_name": spec.get("name"),
            "model_path": str(model_path),
            "model_used": model_used,
            "blocker": None if model_used else "no_usable_generations",
            "env_details": env_details,
        },
        "rows": rows,
    }


def probe_gpu() -> JsonDict:  # pragma: no cover - host-specific probe.
    """Return a small GPU provenance snapshot without making it a hard dependency."""

    try:
        from scripts.experiment_template import (  # noqa: PLC0415
            _cuda_is_available,
            _detect_gpu_count_rocm_aware,
        )

        return {
            "cuda_available": _cuda_is_available(),
            "gpu_count": _detect_gpu_count_rocm_aware(),
        }
    except Exception as exc:
        return {"probe_error": f"{type(exc).__name__}: {exc}"}


def selection_cost(model: FeatureModel, selection: Iterable[str]) -> int:
    """Return the deterministic total cost for a feature selection."""

    return sum(model.costs[feature] for feature in selection)


def selection_value(model: FeatureModel, selection: Iterable[str]) -> int:
    """Return the deterministic total utility for a feature selection."""

    return sum(model.values[feature] for feature in selection)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the conductor and manual runs."""

    args = list(sys.argv[1:] if argv is None else argv)
    max_models = int(os.getenv("CARNOT_PRODUCT_LINE_1511_MAX_MODELS", "1"))
    if "--all-models" in args:
        max_models = len(MANDATED_MODEL_SPECS)
    artifact = run_benchmark(max_models=max_models)
    print(
        "[exp1511] "
        f"ready={artifact['product_line_benchmark_ready']} "
        f"feature_models={artifact['feature_models_defined']} "
        f"models={artifact['models_used']} "
        f"parse_rate={artifact['parse_rate']} "
        f"feasibility_rate={artifact['feasibility_rate']} "
        f"false_accept={artifact['verifier_false_accept_rate']}"
    )
    return 0


def _parse_feature_line(item: str) -> tuple[str, str, int, int]:
    parts = item.split()
    if len(parts) < 4:
        raise ValueError(f"malformed feature line: {item}")  # pragma: no cover
    name = parts[0]
    kind = parts[1]
    attrs = _parse_key_values(parts[2:])
    return name, kind, int(attrs["cost"]), int(attrs["value"])


def _parse_key_values(parts: list[str]) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            raise ValueError(f"malformed key/value: {part}")  # pragma: no cover
        key, value = part.split("=", 1)
        attrs[key] = value
    if "cost" not in attrs or "value" not in attrs:
        raise ValueError("feature missing cost/value")  # pragma: no cover
    return attrs


def _parse_constraint_pair(item: str, delimiter: str) -> tuple[str, str]:
    token = f" {delimiter} " if delimiter == "x" else delimiter
    if token not in item:
        raise ValueError(f"malformed constraint: {item}")  # pragma: no cover
    left, right = item.split(token, 1)
    return left.strip(), right.strip()


def _parse_operation(item: str) -> AnalysisOperation:
    parts = item.split()
    if not parts:
        raise ValueError("missing operation")  # pragma: no cover
    kind = parts[0]
    if kind not in {"max_value", "min_cost"}:
        raise ValueError(f"unsupported operation: {kind}")
    attrs: dict[str, str] = {}
    for part in parts[1:]:
        key, value = part.split("=", 1)
        attrs[key] = value
    include = frozenset(
        feature for feature in attrs.get("include", "").split(",") if feature
    )
    budget = int(attrs["budget"]) if "budget" in attrs else None
    return AnalysisOperation(kind=kind, include=include, budget=budget)


def _build_prompt(case_id: str, blueprint_text: str) -> str:
    schema = {
        "selected_features": ["include every selected feature, including mandatory/root features"],
        "objective_cost": "<integer total cost>",
        "objective_value": "<integer total value>",
        "verifier": {"accept": "<true only if the selection is feasible and optimal>"},
    }
    return (
        "Solve this bounded product-line feature-model case.\n"
        f"Case: {case_id}\n"
        "Return exactly one JSON object and no prose.\n"
        "The JSON object must follow this shape:\n"
        f"{json.dumps(schema, sort_keys=True)}\n\n"
        "Feature-model blueprint:\n"
        f"{blueprint_text.strip()}\n"
    )


def _max_value_key(model: FeatureModel, selection: frozenset[str]) -> tuple[int, int, tuple[str, ...]]:
    return (
        selection_value(model, selection),
        -selection_cost(model, selection),
        tuple(sorted(selection)),
    )


def _min_cost_key(model: FeatureModel, selection: frozenset[str]) -> tuple[int, int, tuple[str, ...]]:
    return (
        selection_cost(model, selection),
        -selection_value(model, selection),
        tuple(sorted(selection)),
    )


def _declared_accept(obj: JsonDict) -> bool | None:
    verifier = obj.get("verifier")
    if isinstance(verifier, dict) and isinstance(verifier.get("accept"), bool):
        return bool(verifier["accept"])
    return None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _resolve_headline_specs(
    specs: list[JsonDict], cached_pair_fn: CachedPairFn | None
) -> tuple[list[JsonDict], list[JsonDict], str | None]:
    pair_details: list[JsonDict] = []
    pair_error: str | None = None
    try:
        if cached_pair_fn is None:  # pragma: no cover - real cache path exercised by live run.
            from carnot.inference.sota_models import cached_sota_pair  # noqa: PLC0415

            pair = cached_sota_pair(gpu_indices=(0, 1))
        else:
            pair = cached_pair_fn(gpu_indices=(0, 1))
    except Exception as exc:  # pragma: no cover - depends on host cache/import state.
        pair = None
        pair_error = f"{type(exc).__name__}: {exc}"
    if pair:
        pair_details = [dict(item) for item in pair]
        paths = {item.get("hf_id"): item.get("model_path") for item in pair if item.get("model_path")}
        for spec in specs:
            if spec.get("hf_id") in paths:
                spec["model_path"] = paths[spec.get("hf_id")]
    return specs, pair_details, pair_error


def _collect_blockers(model_attempts: list[JsonDict], cached_pair_error: str | None) -> list[str]:
    blockers: list[str] = []
    if cached_pair_error:
        blockers.append(f"cached_sota_pair_error:{cached_pair_error}")  # pragma: no cover
    for attempt in model_attempts:
        blocker = attempt.get("blocker")
        if blocker and blocker != "not_attempted_runtime_budget" and str(blocker) not in blockers:
            blockers.append(str(blocker))
    return blockers


def _default_resolver(hf_id: str) -> str | None:  # pragma: no cover - thin external resolver.
    from carnot.inference.sota_models import resolve_cached_gguf  # noqa: PLC0415

    return resolve_cached_gguf(hf_id)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _display_path(path: Path | str) -> str:
    as_path = Path(path)
    try:
        return str(as_path.resolve().relative_to(_repo_root()))
    except ValueError:
        return str(as_path)


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - exercised by conductor.
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ARTIFACT_PATH",
    "DEFAULT_MANIFEST_PATH",
    "MANDATED_MODEL_SPECS",
    "REQUIRED_ARTIFACT_FIELDS",
    "AnalysisOperation",
    "FeatureModel",
    "OracleResult",
    "ParsedModelAnswer",
    "ProductLineCase",
    "SelectionEvaluation",
    "aggregate_manifest_metrics",
    "build_feature_model_cases",
    "build_manifest_row",
    "collect_live_model_outputs",
    "compliant_answer_for_case",
    "evaluate_selection",
    "is_selection_feasible",
    "main",
    "parse_blueprint",
    "parse_model_answer",
    "run_benchmark",
    "selection_cost",
    "selection_satisfies_operation",
    "selection_value",
    "solve_case",
    "write_in_progress_artifact",
]
