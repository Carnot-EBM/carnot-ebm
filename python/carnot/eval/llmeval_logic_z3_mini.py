"""Exp 2931 LLMEval-Logic-style local GGUF to Z3 mini benchmark.

Spec: REQ-BENCH-2931, SCENARIO-BENCH-2931.

This module is intentionally small and exact.  A mandated local GGUF is asked
to translate natural-language logic items into one JSON formalization.  Carnot
then treats Z3, not the model, as the authority for whether the query is
possible, necessary, or impossible.  The experiment keeps syntax parsing, Z3
execution, final answer correctness, and semantic faithfulness as separate
measurements so a correct-looking answer cannot hide a bad formalization.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import z3

from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
RUN_DATE = "20260523"
RANDOM_SEED = 2931
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILENAME = "experiment_2931_llmeval_logic_z3_mini_v1.json"
RAW_RESPONSE_DIRNAME = "llmeval_logic_z3_mini_2931_raw"
INFERENCE_SUBSTRATE = "live_llm_inference_plus_z3"
ALLOWED_ANSWERS = frozenset({"necessary", "possible", "impossible", "inconsistent"})
MANDATED_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {"name": "Qwen3.6-35B-A3B", "hf_id": MANDATED_MODEL_IDS[0], "gpu": 0},
    {"name": "Gemma4-31B-it", "hf_id": MANDATED_MODEL_IDS[1], "gpu": 0},
    {"name": "Gemma4-26B-A4B-it", "hf_id": MANDATED_MODEL_IDS[2], "gpu": 0},
)
_SPEC_BY_HF_ID = {str(spec["hf_id"]): spec for spec in MANDATED_MODEL_SPECS}
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "logic_verifier_mini_ready",
    "benchmark_scope",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
    "models_used",
    "n_items",
    "parseability_rate",
    "z3_execution_rate",
    "answer_accuracy",
    "formalization_faithfulness_rate",
    "per_item_results",
    "raw_response_dir",
    "inference_substrate",
    "duration_s",
    "run_date",
)

CachedPairProvider = Callable[..., list[JsonDict] | None]
IndividualResolver = Callable[[str], str | None]
LlamaImporter = Callable[[], tuple[bool, type[Any] | None, str | None]]
CollectModelOutputs = Callable[[JsonDict, list["LogicItem"], "ExperimentConfig"], JsonDict]


@dataclass(frozen=True)
class LogicItem:
    """One bounded natural-language logic item plus its gold Z3 formalization."""

    item_id: str
    problem: str
    gold_answer: str
    formalization: JsonDict
    prompt: str = ""


@dataclass(frozen=True)
class ParsedModelResponse:
    """Structured parse result for one model response."""

    parseable: bool
    formalization: JsonDict | None
    answer: str | None
    error: str | None


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for Exp 2931 artifact paths and live collection."""

    output_path: Path | None = None
    raw_response_dir: Path | None = None
    max_models: int = 1
    random_seed: int = RANDOM_SEED
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    monotonic_clock: Callable[[], float] = time.monotonic

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or REPO_ROOT / "results" / OUTPUT_FILENAME

    def response_dir(self) -> Path:
        return self.raw_response_dir or REPO_ROOT / "results" / RAW_RESPONSE_DIRNAME


def build_or_load_logic_items(
    cache_paths: Sequence[Path | str] | None = None,
) -> tuple[list[LogicItem], str]:
    """Load a local LLMEval-Logic cache when present, otherwise use a fixture."""

    paths = list(cache_paths) if cache_paths is not None else _default_cache_paths()
    loaded: list[LogicItem] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    loaded.append(_logic_item_from_record(json.loads(line)))
    if loaded:
        scope = (
            "local LLMEval-Logic cache subset; bounded to available Z3-checkable "
            f"records from {len(paths)} candidate path(s)"
        )
        return [_with_prompt(item) for item in loaded], scope

    items = [_with_prompt(item) for item in _fixture_items()]
    return (
        items,
        "forward-authored LLMEval-Logic-style mini fixture, not the full benchmark",
    )


def prompt_for_item(item: LogicItem) -> str:
    """Return the exact structured-output prompt used for one logic item."""

    return (
        "Translate the natural-language logic problem into bounded ground logic.\n"
        "Return exactly one JSON object and no prose.\n"
        "The JSON object must have this shape:\n"
        '{"formalization": {"facts": [["Predicate", "entity"]], '
        '"rules": [{"if": [["Predicate", "entity"]], "then": ["Predicate", "entity"]}], '
        '"exclusions": [[["Predicate", "entity"], ["Other", "entity"]]], '
        '"query": ["Predicate", "entity"]}, "answer": "necessary|possible|impossible"}\n'
        "Use only ground atoms; a rule is a Horn implication from all if-atoms to the then-atom.\n"
        f"Item id: {item.item_id}\n"
        f"Problem: {item.problem}\n"
    )


def canonical_formalization(formalization: Mapping[str, Any]) -> JsonDict:
    """Return a stable normal form for comparing model JSON with gold JSON."""

    facts = sorted(_canonical_atom(atom) for atom in formalization.get("facts", []))
    rules = [
        {
            "if": sorted(_canonical_atom(atom) for atom in rule.get("if", [])),
            "then": _canonical_atom(rule.get("then")),
        }
        for rule in formalization.get("rules", [])
    ]
    rules = sorted(rules, key=lambda rule: json.dumps(rule, sort_keys=True))
    exclusions = sorted(
        sorted(_canonical_atom(atom) for atom in pair)
        for pair in formalization.get("exclusions", [])
    )
    return {
        "facts": facts,
        "rules": rules,
        "exclusions": exclusions,
        "query": _canonical_atom(formalization.get("query")),
    }


def parse_model_response(text: str) -> ParsedModelResponse:
    """Extract and validate the first structured formalization JSON object."""

    obj, error = _extract_json_object(text)
    if error is not None:
        return ParsedModelResponse(False, None, None, error)
    formalization = obj.get("formalization")
    if not isinstance(formalization, dict):
        return ParsedModelResponse(False, None, None, "formalization_not_object")
    answer = obj.get("answer")
    if not isinstance(answer, str):
        return ParsedModelResponse(False, None, None, "answer_not_string")
    normalized_answer = answer.strip().lower()
    if normalized_answer not in ALLOWED_ANSWERS:
        return ParsedModelResponse(False, None, None, "answer_not_in_allowed_set")
    return ParsedModelResponse(True, dict(formalization), normalized_answer, None)


def execute_z3_checks(formalization: Mapping[str, Any]) -> JsonDict:
    """Compile the formalization to Z3 and check possible/necessary status."""

    try:
        atom_cache: dict[str, z3.BoolRef] = {}
        solver = z3.Solver()
        for atom in formalization.get("facts", []):
            solver.add(_atom_expr(atom, atom_cache))
        for rule in formalization.get("rules", []):
            antecedents = [_atom_expr(atom, atom_cache) for atom in rule.get("if", [])]
            premise = z3.And(*antecedents) if antecedents else z3.BoolVal(True)
            solver.add(z3.Implies(premise, _atom_expr(rule.get("then"), atom_cache)))
        for pair in formalization.get("exclusions", []):
            left, right = pair
            solver.add(z3.Not(z3.And(_atom_expr(left, atom_cache), _atom_expr(right, atom_cache))))
        query = _atom_expr(formalization.get("query"), atom_cache)
    except Exception as exc:
        return {
            "z3_executed": False,
            "z3_error": str(exc),
            "knowledge_base_consistent": False,
            "possible": False,
            "necessary": False,
            "solver_answer": None,
        }

    if solver.check() == z3.unsat:
        return {
            "z3_executed": True,
            "z3_error": None,
            "knowledge_base_consistent": False,
            "possible": False,
            "necessary": False,
            "solver_answer": "inconsistent",
        }

    solver.push()
    solver.add(query)
    possible = solver.check() == z3.sat
    solver.pop()
    solver.push()
    solver.add(z3.Not(query))
    necessary = solver.check() == z3.unsat
    solver.pop()
    if necessary:
        solver_answer = "necessary"
    elif possible:
        solver_answer = "possible"
    else:
        solver_answer = "impossible"
    return {
        "z3_executed": True,
        "z3_error": None,
        "knowledge_base_consistent": True,
        "possible": bool(possible),
        "necessary": bool(necessary),
        "solver_answer": solver_answer,
    }


def evaluate_raw_output(
    item: LogicItem,
    raw_text: str,
    *,
    generation_metadata: Mapping[str, Any],
) -> JsonDict:
    """Evaluate one raw model response against parser, Z3, and gold rubric."""

    parsed = parse_model_response(raw_text)
    raw_sha = sha256_text(raw_text)
    if not parsed.parseable or parsed.formalization is None:
        return {
            "item_id": item.item_id,
            "gold_answer": item.gold_answer,
            "model_answer": parsed.answer,
            "solver_answer": None,
            "parseable": False,
            "z3_executed": False,
            "answer_correct": False,
            "semantic_faithful": False,
            "violation_class": "parse_error",
            "parse_error": parsed.error,
            "parsed_formalization": None,
            "z3_result": {
                "z3_executed": False,
                "z3_error": parsed.error,
                "knowledge_base_consistent": False,
                "possible": False,
                "necessary": False,
                "solver_answer": None,
            },
            "raw_output_sha256": raw_sha,
            **_generation_fields(generation_metadata),
        }

    z3_result = execute_z3_checks(parsed.formalization)
    solver_answer = z3_result.get("solver_answer")
    semantic_faithful = (
        canonical_formalization(parsed.formalization) == canonical_formalization(item.formalization)
        if z3_result.get("z3_executed")
        else False
    )
    answer_correct = bool(
        z3_result.get("z3_executed")
        and parsed.answer == item.gold_answer
        and solver_answer == item.gold_answer
    )
    if not z3_result.get("z3_executed"):
        violation_class = "z3_error"
    elif answer_correct and semantic_faithful:
        violation_class = "none"
    elif not answer_correct:
        violation_class = "answer_wrong"
    else:
        violation_class = "semantic_mismatch"
    return {
        "item_id": item.item_id,
        "gold_answer": item.gold_answer,
        "model_answer": parsed.answer,
        "solver_answer": solver_answer,
        "parseable": True,
        "z3_executed": bool(z3_result.get("z3_executed")),
        "answer_correct": answer_correct,
        "semantic_faithful": semantic_faithful,
        "violation_class": violation_class,
        "parse_error": None,
        "parsed_formalization": (
            canonical_formalization(parsed.formalization)
            if z3_result.get("z3_executed")
            else None
        ),
        "z3_result": z3_result,
        "raw_output_sha256": raw_sha,
        **_generation_fields(generation_metadata),
    }


def aggregate_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute separate parse, Z3, answer, and faithfulness rates."""

    if not rows:
        return {
            "parseability_rate": 0.0,
            "z3_execution_rate": 0.0,
            "answer_accuracy": 0.0,
            "formalization_faithfulness_rate": 0.0,
        }
    total = len(rows)
    return {
        "parseability_rate": round(sum(bool(row["parseable"]) for row in rows) / total, 6),
        "z3_execution_rate": round(sum(bool(row["z3_executed"]) for row in rows) / total, 6),
        "answer_accuracy": round(sum(bool(row["answer_correct"]) for row in rows) / total, 6),
        "formalization_faithfulness_rate": round(
            sum(bool(row["semantic_faithful"]) for row in rows) / total,
            6,
        ),
    }


def compute_reproducibility_checksum(
    *,
    items: Sequence[Any],
    prompts: Sequence[str],
    model_specs: Sequence[Mapping[str, Any]],
    raw_outputs: Sequence[Any],
    parsed_formulas: Sequence[Any],
    z3_results: Sequence[Any],
) -> str:
    """Hash the full evidence surface that makes the artifact reproducible."""

    payload = {
        "random_seed": RANDOM_SEED,
        "items": items,
        "prompts": prompts,
        "model_specs": model_specs,
        "raw_outputs": raw_outputs,
        "parsed_formulas": parsed_formulas,
        "z3_results": z3_results,
    }
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def resolve_model_specs(
    *,
    cached_pair_provider: CachedPairProvider = cached_sota_pair,
    individual_model_resolver: IndividualResolver = resolve_cached_gguf,
) -> tuple[list[JsonDict], bool, str | None]:
    """Call cached_sota_pair first, then resolve only mandated GGUF IDs."""

    cache_error = None
    try:
        pair = cached_pair_provider(gpu_indices=(0, 1))
        if pair:
            specs = [dict(spec) for spec in pair if spec.get("hf_id") in MANDATED_MODEL_IDS]
            if specs:
                return specs, True, None
    except Exception as exc:
        cache_error = f"{type(exc).__name__}: {exc}"

    specs = []
    for hf_id in MANDATED_MODEL_IDS:
        path = individual_model_resolver(hf_id)
        if path:
            spec = dict(_SPEC_BY_HF_ID[hf_id])
            spec["model_path"] = str(path)
            specs.append(spec)
    return specs, False, cache_error


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    cached_pair_provider: CachedPairProvider = cached_sota_pair,
    individual_model_resolver: IndividualResolver = resolve_cached_gguf,
    collect_model_outputs_fn: CollectModelOutputs | None = None,
) -> JsonDict:
    """Run the mini benchmark and write the required deliverable JSON."""

    active = config or ExperimentConfig()
    started = active.start_time()
    items, benchmark_scope = build_or_load_logic_items()
    prompts = [item.prompt for item in items]
    specs, cached_pair_used, cache_error = resolve_model_specs(
        cached_pair_provider=cached_pair_provider,
        individual_model_resolver=individual_model_resolver,
    )
    model_specs = specs if specs else [dict(spec) for spec in MANDATED_MODEL_SPECS]
    rows: list[JsonDict] = []
    model_attempts: list[JsonDict] = []

    if specs:
        collector = collect_model_outputs_fn or collect_live_model_outputs
        item_by_id = {item.item_id: item for item in items}
        for index, spec in enumerate(specs):
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
            for generation_row in collection.get("rows") or []:
                item = item_by_id.get(generation_row.get("item_id"))
                if item is None:
                    continue
                rows.append(
                    evaluate_raw_output(
                        item,
                        str(generation_row.get("output_text") or ""),
                        generation_metadata=generation_row,
                    )
                )

    metrics = aggregate_results(rows)
    models_used = [
        str(attempt["hf_id"])
        for attempt in model_attempts
        if attempt.get("model_used") is True and attempt.get("hf_id") in MANDATED_MODEL_IDS
    ]
    ready = bool(models_used) and bool(rows) and metrics["z3_execution_rate"] == 1.0
    checksum = compute_reproducibility_checksum(
        items=[_item_manifest_row(item) for item in items],
        prompts=prompts,
        model_specs=model_specs,
        raw_outputs=[row["raw_output_sha256"] for row in rows],
        parsed_formulas=[row["parsed_formalization"] for row in rows],
        z3_results=[row["z3_result"] for row in rows],
    )
    artifact = {
        "honest_verdict": _honest_verdict(
            specs=specs,
            ready=ready,
            models_used=models_used,
            z3_execution_rate=metrics["z3_execution_rate"],
        ),
        "logic_verifier_mini_ready": ready,
        "benchmark_scope": benchmark_scope,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": checksum,
        "model_specs": model_specs,
        "models_used": models_used,
        "n_items": len(items),
        "parseability_rate": metrics["parseability_rate"],
        "z3_execution_rate": metrics["z3_execution_rate"],
        "answer_accuracy": metrics["answer_accuracy"],
        "formalization_faithfulness_rate": metrics["formalization_faithfulness_rate"],
        "per_item_results": rows,
        "raw_response_dir": str(active.response_dir()),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(active.clock() - started, 6),
        "run_date": RUN_DATE,
        "item_manifest": [_item_manifest_row(item) for item in items],
        "prompts": prompts,
        "cached_sota_pair_used": cached_pair_used,
        "cached_sota_pair_error": cache_error,
        "model_attempts": model_attempts,
    }
    _write_json(active.artifact_path(), artifact)
    return artifact


def collect_live_model_outputs(
    spec: JsonDict,
    items: list[LogicItem],
    config: ExperimentConfig,
    *,
    llama_importer: LlamaImporter | None = None,
) -> JsonDict:
    """Collect structured formalization JSON from one local GGUF model."""

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
            n_ctx=4096,
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
            try:
                result = llm(
                    item.prompt,
                    max_tokens=512,
                    temperature=0.0,
                    top_p=1.0,
                    seed=config.random_seed + index,
                    stop=["</s>", "<eos>"],
                    echo=False,
                )
                output_text = completion_text(result)
                blocker = None if output_text.strip() else "empty_generation"
            except Exception as exc:
                output_text = ""
                blocker = f"{type(exc).__name__}: {exc}"
            raw_path = config.response_dir() / f"{item.item_id}.json"
            raw_path.write_text(output_text, encoding="utf-8")
            rows.append(
                {
                    "item_id": item.item_id,
                    "model_hf_id": hf_id,
                    "model_name": spec.get("name"),
                    "model_path": model_path,
                    "gpu_index": spec.get("gpu"),
                    "prompt_hash": sha256_text(item.prompt),
                    "per_item_seed": config.random_seed + index,
                    "generation_source": "live_sota_llamacpp_logic_json",
                    "output_text": output_text,
                    "raw_response_path": str(raw_path),
                    "raw_response_sha256": sha256_text(output_text),
                    "elapsed_seconds": round(config.monotonic_clock() - started, 6),
                    "blocker": blocker,
                }
            )
    finally:
        _close_llama(llm)

    model_used = any(not row.get("blocker") for row in rows)
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


def completion_text(result: Any) -> str:
    """Extract text from common llama.cpp completion shapes."""

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


def _fixture_items() -> list[LogicItem]:
    items = [
        _item(
            "llmlogic-2931-001",
            "All cats are mammals. Milo is a cat. Is Milo necessarily a mammal?",
            "necessary",
            facts=[["Cat", "milo"]],
            rules=[{"if": [["Cat", "milo"]], "then": ["Mammal", "milo"]}],
            exclusions=[],
            query=["Mammal", "milo"],
        ),
        _item(
            "llmlogic-2931-002",
            "Nia is a student. Some students may be athletes, but no rule says Nia is one. Is Nia an athlete possible?",
            "possible",
            facts=[["Student", "nia"]],
            rules=[],
            exclusions=[],
            query=["Athlete", "nia"],
        ),
        _item(
            "llmlogic-2931-003",
            "No robot is organic. Rho is a robot. Is Rho organic possible?",
            "impossible",
            facts=[["Robot", "rho"]],
            rules=[],
            exclusions=[[["Robot", "rho"], ["Organic", "rho"]]],
            query=["Organic", "rho"],
        ),
        _item(
            "llmlogic-2931-004",
            "Every archivist is careful. Every careful person is trusted. Ivo is an archivist. Is Ivo trusted necessary?",
            "necessary",
            facts=[["Archivist", "ivo"]],
            rules=[
                {"if": [["Archivist", "ivo"]], "then": ["Careful", "ivo"]},
                {"if": [["Careful", "ivo"]], "then": ["Trusted", "ivo"]},
            ],
            exclusions=[],
            query=["Trusted", "ivo"],
        ),
        _item(
            "llmlogic-2931-005",
            "A token cannot be both active and revoked. Token tau is revoked. Can tau be active?",
            "impossible",
            facts=[["Revoked", "tau"]],
            rules=[],
            exclusions=[[["Active", "tau"], ["Revoked", "tau"]]],
            query=["Active", "tau"],
        ),
        _item(
            "llmlogic-2931-006",
            "Every pilot is licensed. Ada is licensed, but nothing says Ada is a pilot. Is Ada a pilot necessary?",
            "possible",
            facts=[["Licensed", "ada"]],
            rules=[{"if": [["Pilot", "ada"]], "then": ["Licensed", "ada"]}],
            exclusions=[],
            query=["Pilot", "ada"],
        ),
        _item(
            "llmlogic-2931-007",
            "If a sample is sterile then it is sealed. Sample s1 is sterile. Is s1 sealed necessary?",
            "necessary",
            facts=[["Sterile", "s1"]],
            rules=[{"if": [["Sterile", "s1"]], "then": ["Sealed", "s1"]}],
            exclusions=[],
            query=["Sealed", "s1"],
        ),
        _item(
            "llmlogic-2931-008",
            "No blue badge is temporary. Badge b7 is blue. Can b7 be temporary?",
            "impossible",
            facts=[["BlueBadge", "b7"]],
            rules=[],
            exclusions=[[["BlueBadge", "b7"], ["Temporary", "b7"]]],
            query=["Temporary", "b7"],
        ),
        _item(
            "llmlogic-2931-009",
            "Every audited build is reproducible. Build kappa is audited and signed. Is kappa reproducible necessary?",
            "necessary",
            facts=[["Audited", "kappa"], ["Signed", "kappa"]],
            rules=[{"if": [["Audited", "kappa"]], "then": ["Reproducible", "kappa"]}],
            exclusions=[],
            query=["Reproducible", "kappa"],
        ),
        _item(
            "llmlogic-2931-010",
            "No node can be both offline and reachable. Node n4 is offline. Can n4 be reachable?",
            "impossible",
            facts=[["Offline", "n4"]],
            rules=[],
            exclusions=[[["Offline", "n4"], ["Reachable", "n4"]]],
            query=["Reachable", "n4"],
        ),
        _item(
            "llmlogic-2931-011",
            "If a parcel is fragile and shipped, it must be padded. Parcel p2 is fragile and shipped. Is p2 padded necessary?",
            "necessary",
            facts=[["Fragile", "p2"], ["Shipped", "p2"]],
            rules=[
                {"if": [["Fragile", "p2"], ["Shipped", "p2"]], "then": ["Padded", "p2"]}
            ],
            exclusions=[],
            query=["Padded", "p2"],
        ),
        _item(
            "llmlogic-2931-012",
            "Kai is enrolled. There is no rule connecting enrollment to graduation. Is Kai graduated possible?",
            "possible",
            facts=[["Enrolled", "kai"]],
            rules=[],
            exclusions=[],
            query=["Graduated", "kai"],
        ),
    ]
    return items


def _item(
    item_id: str,
    problem: str,
    gold_answer: str,
    *,
    facts: list[list[str]],
    rules: list[JsonDict],
    exclusions: list[list[list[str]]],
    query: list[str],
) -> LogicItem:
    return LogicItem(
        item_id=item_id,
        problem=problem,
        gold_answer=gold_answer,
        formalization={
            "facts": facts,
            "rules": rules,
            "exclusions": exclusions,
            "query": query,
        },
    )


def _with_prompt(item: LogicItem) -> LogicItem:
    return LogicItem(
        item_id=item.item_id,
        problem=item.problem,
        gold_answer=item.gold_answer,
        formalization=item.formalization,
        prompt=prompt_for_item(item),
    )


def _logic_item_from_record(record: Mapping[str, Any]) -> LogicItem:
    return LogicItem(
        item_id=str(record["item_id"]),
        problem=str(record.get("problem", record.get("prompt", ""))),
        gold_answer=str(record["gold_answer"]).lower(),
        formalization=dict(record["formalization"]),
    )


def _item_manifest_row(item: LogicItem) -> JsonDict:
    return {
        "item_id": item.item_id,
        "problem": item.problem,
        "gold_answer": item.gold_answer,
        "formalization": canonical_formalization(item.formalization),
    }


def _default_cache_paths() -> list[Path]:
    candidates = [
        REPO_ROOT / "data" / "research" / "llmeval_logic.jsonl",
        REPO_ROOT / "data" / "research" / "llmeval_logic_mini.jsonl",
        REPO_ROOT / "data" / "llmeval_logic.jsonl",
    ]
    return [path for path in candidates if path.exists()]


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
            return obj, None
    return {}, "no_json_object"


def _atom_expr(atom: Any, cache: dict[str, z3.BoolRef]) -> z3.BoolRef:
    canonical = _canonical_atom(atom)
    key = "__".join(canonical)
    if key not in cache:
        cache[key] = z3.Bool(key)
    return cache[key]


def _canonical_atom(atom: Any) -> list[str]:
    if not isinstance(atom, list) or not atom:
        raise ValueError("atom_must_be_nonempty_list")
    if len(atom) < 2:
        raise ValueError("atom_needs_predicate_and_argument")
    if not all(isinstance(part, str) and part for part in atom):
        raise ValueError("atom_parts_must_be_nonempty_strings")
    return list(atom)


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


def _honest_verdict(
    *,
    specs: Sequence[Mapping[str, Any]],
    ready: bool,
    models_used: Sequence[str],
    z3_execution_rate: float,
) -> str:
    if not specs:
        return "blocked_sota_gguf_cache_missing"
    if ready:
        return "complete: llmeval-logic-style local GGUF formalizations checked by Z3"
    if not models_used:
        return "blocked_sota_runtime_unavailable"
    if z3_execution_rate < 1.0:
        return "blocked_z3_execution_incomplete"
    return "blocked_logic_verifier_mini_not_ready"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _default_llama_importer() -> tuple[bool, type[Any] | None, str | None]:  # pragma: no cover
    try:
        from llama_cpp import Llama  # noqa: PLC0415

        return True, Llama, None
    except Exception as exc:  # noqa: BLE001
        return False, None, f"{type(exc).__name__}: {exc}"


def _close_llama(llm: Any) -> None:
    close = getattr(llm, "close", None)
    if callable(close):
        close()


def main() -> int:  # pragma: no cover
    artifact = run_experiment()
    print(
        "[exp2931] "
        f"ready={artifact['logic_verifier_mini_ready']} "
        f"items={artifact['n_items']} "
        f"parse={artifact['parseability_rate']} "
        f"z3={artifact['z3_execution_rate']} "
        f"answer={artifact['answer_accuracy']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
