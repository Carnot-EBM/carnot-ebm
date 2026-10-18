"""Exp 3058 AquaForte-style LLM-guided SMT instantiation pilot.

Spec refs: REQ-VERIFY-3058,
           SCENARIO-VERIFY-3058,
           SCENARIO-VERIFY-3058-BLOCKED.

This module runs a deliberately tiny version of the AquaForte pattern:
a local mandated GGUF proposes quantified-SMT instantiations, then Z3 decides
whether those instantiations actually prove the target. The LLM is never the
authority. Invalid proposals stay in the row evidence, and a solver-only
fallback is evaluated on the same fixtures so guidance cannot replace formal
validation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from itertools import product
from pathlib import Path
import time
from typing import Any

from carnot.experiment_3043_verified_speculation_transcript_fingerprint import (
    _cuda_probe,
    _extract_text,
    _file_evidence,
    _gpu_inventory,
    _normalize_output,
    _python_environment,
    _repo_commit,
)
from carnot.inference.sota_models import SOTA_GGUF_MODELS, resolve_cached_gguf

try:  # pragma: no cover - dependency absence is exercised through injection.
    import z3 as _z3
except Exception:  # pragma: no cover
    _z3 = None


JsonDict = dict[str, Any]
ResolveGgufFn = Callable[[str, str], str | None]
LlamaFactory = Callable[..., Any]
ClockFn = Callable[[], float]
RepoCommitFn = Callable[[Path], str]

ARTIFACT = "experiment_3058_aquaforte_style_llm_guided_smt_pilot_v1"
ARTIFACT_FILENAME = f"{ARTIFACT}.json"
SCRIPT_FILENAME = f"{ARTIFACT}.py"
SCHEMA = "carnot.aquaforte_style_llm_guided_smt_pilot.v1"
RUN_DATE = "20260525"
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results") / ARTIFACT_FILENAME
PILOT_ROWS_REL_PATH = (
    Path("results") / "aquaforte_style_llm_guided_smt_pilot_3058" / "pilot_rows.jsonl"
)
EXP3057_REL_PATH = (
    Path("results") / "experiment_3057_local_sota_solution_verifier_gain_panel_v1.json"
)
EXACT_SOLVER_REL_PATH = Path("python/carnot/eval/aquaforte_style_llm_guided_smt_pilot_v1.py")
EXACT_SOLVER_PATH = f"{EXACT_SOLVER_REL_PATH.as_posix()}::validate_proposal_with_z3"
DEFAULT_SEED = 305800
DEFAULT_DECODE_CONFIG: JsonDict = {
    "max_tokens": 96,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "repeat_penalty": 1.05,
}
DEFAULT_LOAD_CONFIG: JsonDict = {
    "n_ctx": 1024,
    "n_batch": 64,
    "n_ubatch": 64,
    "n_gpu_layers": -1,
    "main_gpu": 0,
    "verbose": False,
}
MANDATED_MODEL_IDS = tuple(model["hf_id"] for model in SOTA_GGUF_MODELS)
REQUIRED_ARTIFACT_FIELDS = (
    "llm_guided_smt_pilot_ready",
    "formal_fallback_preserved",
    "guided_success_count",
    "solver_only_success_count",
    "unresolved_count",
    "invalid_llm_proposal_count",
    "models_used",
    "model_specs",
    "legacy_smoke_only_used",
    "exact_solver_path",
    "prompt_hashes",
    "inference_substrate",
    "honest_verdict",
)
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for the Exp 3058 pilot."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    rows_path: Path | None = None
    seed: int = DEFAULT_SEED
    preferred_quant: str = "Q4_K_M"
    decode_config: Mapping[str, Any] | None = None
    load_config: Mapping[str, Any] | None = None
    tests_run: Sequence[str] = ()

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / OUTPUT_REL_PATH

    def pilot_rows_path(self) -> Path:
        return self.rows_path or self.repo_root / PILOT_ROWS_REL_PATH

    def effective_decode_config(self) -> JsonDict:
        config = dict(DEFAULT_DECODE_CONFIG)
        if self.decode_config:
            config.update(dict(self.decode_config))
        return config

    def effective_load_config(self, gpu: int = 0) -> JsonDict:
        config = dict(DEFAULT_LOAD_CONFIG)
        if self.load_config:
            config.update(dict(self.load_config))
        config["main_gpu"] = int(gpu)
        return config


def build_smt_fixtures() -> list[JsonDict]:
    """Return tiny UF-inspired fixtures for exact instantiation checks."""

    return [
        _function_fixture("uf-inc-2", "f_inc", ("x",), [2], 3, _affine({"x": 1}, 1)),
        _function_fixture("uf-double-4", "f_double", ("x",), [4], 8, _affine({"x": 2}, 0)),
        _function_fixture("uf-offset-5", "g_offset", ("x",), [5], 3, _affine({"x": 1}, -2)),
        _function_fixture(
            "uf-add-2-3",
            "uf_add",
            ("x", "y"),
            [2, 3],
            5,
            _affine({"x": 1, "y": 1}, 0),
        ),
        _function_fixture("uf-square-3", "uf_square", ("x",), [3], 9, {"op": "square"}),
        {
            "fixture_id": "pred-chain-0-2",
            "kind": "predicate_chain",
            "predicate": "reachable",
            "variables": ["x"],
            "premises": [0],
            "target_arg": 2,
            "step": 1,
            "candidate_domain": [0, 1, 2],
            "max_solver_instantiations": 2,
        },
    ]


def parse_llm_instantiations(text: str, fixture: Mapping[str, Any]) -> JsonDict:
    """Parse generated text into concrete instantiation dictionaries."""

    parsed = _parse_json_object(text)
    if not parsed:
        return {"valid_parse": False, "instantiations": [], "parse_error": "json_object_missing"}
    fixture_id = parsed.get("fixture_id")
    if fixture_id is not None and str(fixture_id) != str(fixture["fixture_id"]):
        return {"valid_parse": False, "instantiations": [], "parse_error": "fixture_id_mismatch"}
    raw_instantiations = parsed.get("instantiations")
    if not isinstance(raw_instantiations, list):
        return {"valid_parse": False, "instantiations": [], "parse_error": "instantiations_not_list"}
    instantiations, errors = _normalise_instantiations(raw_instantiations, fixture)
    if errors or not instantiations:
        return {
            "valid_parse": False,
            "instantiations": instantiations,
            "parse_error": ";".join(errors) or "no_valid_instantiations",
        }
    return {"valid_parse": True, "instantiations": instantiations, "parse_error": ""}


def validate_proposal_with_z3(
    fixture: Mapping[str, Any],
    instantiations: Sequence[Mapping[str, int]],
    *,
    z3_module: Any = _z3,
) -> JsonDict:
    """Use Z3 to decide whether candidate instantiations prove the target."""

    if z3_module is None:
        return {
            "valid": False,
            "exact_checked": False,
            "exact_authority": "z3_unavailable",
            "solver_status": "not_run",
            "instantiations": [dict(row) for row in instantiations],
            "failure_reason": "z3_solver_unavailable",
        }
    normalised, errors = _normalise_instantiations(instantiations, fixture)
    if errors:
        return {
            "valid": False,
            "exact_checked": False,
            "exact_authority": "z3_solver",
            "solver_status": "not_run",
            "instantiations": normalised,
            "failure_reason": ";".join(errors),
        }
    if fixture["kind"] == "predicate_chain":
        return _validate_predicate_chain(fixture, normalised, z3_module)
    return _validate_function_value(fixture, normalised, z3_module)


def solver_only_fallback(fixture: Mapping[str, Any], *, z3_module: Any = _z3) -> JsonDict:
    """Enumerate tiny integer instantiations with Z3 as the only authority."""

    if z3_module is None:
        return {
            "valid": False,
            "exact_checked": False,
            "exact_authority": "z3_unavailable",
            "solver_status": "not_run",
            "instantiations": [],
            "attempt_count": 0,
        }
    variables = [str(name) for name in fixture["variables"]]
    domain = [int(value) for value in fixture["candidate_domain"]]
    max_instances = int(fixture.get("max_solver_instantiations", 1))
    attempt_count = 0
    last_result: JsonDict | None = None
    for size in range(1, max_instances + 1):
        for values in product(product(domain, repeat=len(variables)), repeat=size):
            instantiations = [dict(zip(variables, row, strict=True)) for row in values]
            attempt_count += 1
            result = validate_proposal_with_z3(fixture, instantiations, z3_module=z3_module)
            last_result = result
            if result["valid"]:
                return result | {"attempt_count": attempt_count}
    return (last_result or {}) | {"attempt_count": attempt_count}


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    resolve_gguf_func: ResolveGgufFn = resolve_cached_gguf,
    llama_factory: LlamaFactory | None = None,
    z3_module: Any = _z3,
    monotonic: ClockFn = time.monotonic,
    repo_commit_func: RepoCommitFn = _repo_commit,
) -> JsonDict:
    """Run the live guided SMT pilot and write the terminal artifact."""

    active = config or ExperimentConfig()
    started = monotonic()
    exp3057_ready = _exp3057_ready(active.repo_root)
    cache_resolution = _resolve_cache(resolve_gguf_func, active)
    selected_models = _select_models(cache_resolution)
    blocker = _precondition_blocker(exp3057_ready, z3_module, selected_models)
    if blocker:
        artifact = _blocked_artifact(
            config=active,
            cache_resolution=cache_resolution,
            blocker=blocker,
            duration_s=round(monotonic() - started, 6),
            exp3057_ready=exp3057_ready,
            exact_solver_available=z3_module is not None,
            repo_commit_func=repo_commit_func,
        )
        _write_json(active.artifact_path(), artifact)
        return artifact

    try:
        rows, prompt_hashes = _run_live_guidance(
            config=active,
            selected_model=selected_models[0],
            llama_factory=llama_factory or _default_llama_factory,
            z3_module=z3_module,
        )
        runtime_blocker = None
    except Exception as exc:
        rows = []
        prompt_hashes = []
        runtime_blocker = f"{type(exc).__name__}: {exc}"

    if rows:
        _write_jsonl(active.pilot_rows_path(), rows)
    artifact = _build_artifact(
        config=active,
        rows=rows,
        prompt_hashes=prompt_hashes,
        selected_models=selected_models,
        cache_resolution=cache_resolution,
        duration_s=round(monotonic() - started, 6),
        runtime_blocker=runtime_blocker,
        exp3057_ready=exp3057_ready,
        exact_solver_available=z3_module is not None,
        repo_commit_func=repo_commit_func,
    )
    validate_artifact(artifact)
    _write_json(active.artifact_path(), artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 3058 artifact violates the pilot contract."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("legacy_smoke_only_used") is not False:
        raise ValueError("legacy smoke evidence cannot satisfy REQ-VERIFY-3058")
    verdict = str(artifact.get("honest_verdict", ""))
    if artifact.get("llm_guided_smt_pilot_ready") is not True:
        if not verdict.startswith("blocked_"):
            raise ValueError("honest_verdict must disclose the blocked precondition")
        return
    if not artifact.get("model_specs"):
        raise ValueError("model_specs must be present when pilot is ready")
    if not artifact.get("prompt_hashes"):
        raise ValueError("prompt_hashes must be present when pilot is ready")
    if artifact.get("formal_fallback_preserved") is not True:
        raise ValueError("formal_fallback_preserved must be true when pilot is ready")
    if int(artifact.get("guided_success_count") or 0) <= 0:
        raise ValueError("guided_success_count must be positive when pilot is ready")
    if int(artifact.get("solver_only_success_count") or 0) <= 0:
        raise ValueError("solver_only_success_count must be positive when pilot is ready")
    if artifact.get("exact_solver_path") != EXACT_SOLVER_PATH:
        raise ValueError("exact_solver_path must name the Z3 validation function")
    if not set(_string_list(artifact.get("models_used"))).intersection(MANDATED_MODEL_IDS):
        raise ValueError("models_used must include a mandated local GGUF")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping) or substrate.get("local_gguf_inference") is not True:
        raise ValueError("inference_substrate must disclose local GGUF inference")
    if not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal success prefix")


def load_jsonl(path: Path) -> list[JsonDict]:
    """Load JSONL pilot rows written by this module."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _run_live_guidance(
    *,
    config: ExperimentConfig,
    selected_model: Mapping[str, Any],
    llama_factory: LlamaFactory,
    z3_module: Any,
) -> tuple[list[JsonDict], list[str]]:
    fixtures = build_smt_fixtures()
    decode_config = config.effective_decode_config()
    load_config = config.effective_load_config(int(selected_model.get("gpu", 0)))
    llm = llama_factory(model_path=str(selected_model["model_path"]), **load_config)
    rows: list[JsonDict] = []
    prompt_hashes: list[str] = []
    try:
        for fixture in fixtures:
            prompt = _guidance_prompt(fixture)
            raw = llm(prompt, **dict(decode_config), seed=config.seed)
            text = _normalize_output(_extract_text(raw))
            prompt_hash = _sha256_text(prompt)
            parsed = parse_llm_instantiations(text, fixture)
            guided_validation = validate_proposal_with_z3(
                fixture, parsed["instantiations"], z3_module=z3_module
            )
            fallback = solver_only_fallback(fixture, z3_module=z3_module)
            rows.append(
                {
                    "fixture_id": fixture["fixture_id"],
                    "fixture_kind": fixture["kind"],
                    "prompt_hash": prompt_hash,
                    "raw_output_hash": _sha256_text(text),
                    "parse": parsed,
                    "guided_validation": guided_validation,
                    "solver_only_fallback": fallback,
                    "guided_success": bool(guided_validation["valid"]),
                    "solver_only_success": bool(fallback["valid"]),
                    "exact_authority": "z3_solver",
                }
            )
            prompt_hashes.append(prompt_hash)
    finally:
        close = getattr(llm, "close", None)
        if callable(close):
            close()
    return rows, prompt_hashes


def _build_artifact(
    *,
    config: ExperimentConfig,
    rows: Sequence[Mapping[str, Any]],
    prompt_hashes: Sequence[str],
    selected_models: Sequence[Mapping[str, Any]],
    cache_resolution: Mapping[str, str | None],
    duration_s: float,
    runtime_blocker: str | None,
    exp3057_ready: bool,
    exact_solver_available: bool,
    repo_commit_func: RepoCommitFn,
) -> JsonDict:
    metrics = _metrics(rows)
    formal_fallback_preserved = _formal_fallback_preserved(rows, exact_solver_available)
    model_specs = [_model_spec(row) for row in selected_models] if runtime_blocker is None else []
    models_used = [str(row["hf_id"]) for row in selected_models] if runtime_blocker is None else []
    ready = (
        runtime_blocker is None
        and exp3057_ready
        and exact_solver_available
        and 4 <= len(rows) <= 8
        and bool(model_specs)
        and bool(prompt_hashes)
        and metrics["guided_success_count"] > 0
        and formal_fallback_preserved
    )
    artifact = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "llm_guided_smt_pilot_ready": ready,
        "formal_fallback_preserved": formal_fallback_preserved,
        "guided_success_count": metrics["guided_success_count"],
        "solver_only_success_count": metrics["solver_only_success_count"],
        "unresolved_count": metrics["unresolved_count"],
        "invalid_llm_proposal_count": metrics["invalid_llm_proposal_count"],
        "models_used": models_used,
        "model_specs": model_specs,
        "legacy_smoke_only_used": False,
        "exact_solver_path": EXACT_SOLVER_PATH,
        "prompt_hashes": list(prompt_hashes) if runtime_blocker is None else [],
        "inference_substrate": _substrate(
            config=config,
            cache_resolution=cache_resolution,
            selected_models=selected_models if runtime_blocker is None else [],
            duration_s=duration_s,
            exp3057_ready=exp3057_ready,
            exact_solver_available=exact_solver_available,
            repo_commit_func=repo_commit_func,
        ),
        "honest_verdict": _honest_verdict(ready, metrics, runtime_blocker),
        "fixture_count": len(rows),
        "pilot_rows_path": str(_relative_to(config.repo_root, config.pilot_rows_path())),
        "pilot_rows_sha256": _sha256_file(config.pilot_rows_path()) if rows else "",
        "decode_config": config.effective_decode_config(),
        "seed": config.seed,
        "tests_or_checks_run": list(config.tests_run),
        "runtime_blocker": runtime_blocker,
        "preconditions": {
            "exp3057_ready": exp3057_ready,
            "exact_solver_available": exact_solver_available,
            "mandated_gguf_resolved": bool(selected_models),
        },
        "guidance_vs_solver_only": {
            "guided_minus_solver_only_success_count": metrics["guided_success_count"]
            - metrics["solver_only_success_count"],
            "unresolved_reduction_count": 0,
            "fallback_only_reduction_count": 0,
            "guidance_helped": metrics["guided_success_count"]
            > metrics["solver_only_success_count"],
        },
        "pilot_rows": [dict(row) for row in rows],
        "source_context": {
            "exp3044": "results/experiment_3044_smt_sat_validator_tree_exactness_upgrade_v1.json",
            "exp3057": EXP3057_REL_PATH.as_posix(),
            "reference": "research-references.md#AquaForte",
        },
        "duration_s": duration_s,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _sha256_json(
        {
            "models_used": artifact["models_used"],
            "prompt_hashes": artifact["prompt_hashes"],
            "metrics": {
                key: artifact[key]
                for key in (
                    "guided_success_count",
                    "solver_only_success_count",
                    "unresolved_count",
                    "invalid_llm_proposal_count",
                )
            },
        }
    )
    return artifact


def _blocked_artifact(
    *,
    config: ExperimentConfig,
    cache_resolution: Mapping[str, str | None],
    blocker: str,
    duration_s: float,
    exp3057_ready: bool,
    exact_solver_available: bool,
    repo_commit_func: RepoCommitFn,
) -> JsonDict:
    artifact = _build_artifact(
        config=config,
        rows=[],
        prompt_hashes=[],
        selected_models=[],
        cache_resolution=cache_resolution,
        duration_s=duration_s,
        runtime_blocker=blocker,
        exp3057_ready=exp3057_ready,
        exact_solver_available=exact_solver_available,
        repo_commit_func=repo_commit_func,
    )
    validate_artifact(artifact)
    return artifact


def _metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    guided_success = sum(1 for row in rows if row.get("guided_success") is True)
    solver_success = sum(1 for row in rows if row.get("solver_only_success") is True)
    invalid = sum(1 for row in rows if row.get("guided_success") is not True)
    unresolved = sum(
        1
        for row in rows
        if row.get("guided_success") is not True and row.get("solver_only_success") is not True
    )
    return {
        "guided_success_count": guided_success,
        "solver_only_success_count": solver_success,
        "invalid_llm_proposal_count": invalid,
        "unresolved_count": unresolved,
    }


def _formal_fallback_preserved(rows: Sequence[Mapping[str, Any]], exact_solver_available: bool) -> bool:
    return bool(rows) and exact_solver_available and all(
        row.get("solver_only_fallback", {}).get("exact_authority") == "z3_solver"
        and row.get("solver_only_fallback", {}).get("exact_checked") is True
        for row in rows
    )


def _validate_function_value(
    fixture: Mapping[str, Any],
    instantiations: Sequence[Mapping[str, int]],
    z3_module: Any,
) -> JsonDict:
    int_sort = z3_module.IntSort()
    variables = [str(name) for name in fixture["variables"]]
    function = z3_module.Function(str(fixture["function"]), *([int_sort] * len(variables)), int_sort)
    solver = z3_module.Solver()
    for instantiation in instantiations:
        args = [z3_module.IntVal(int(instantiation[name])) for name in variables]
        solver.add(function(*args) == _expr_int_value(fixture["expr"], instantiation, z3_module))
    target_args = [z3_module.IntVal(int(value)) for value in fixture["target_args"]]
    target = function(*target_args) == z3_module.IntVal(int(fixture["target_value"]))
    solver.add(z3_module.Not(target))
    status = solver.check()
    return {
        "valid": status == z3_module.unsat,
        "exact_checked": True,
        "exact_authority": "z3_solver",
        "solver_status": _status_name(status, z3_module),
        "instantiations": [dict(row) for row in instantiations],
        "failure_reason": "" if status == z3_module.unsat else "proposal_does_not_prove_target",
    }


def _validate_predicate_chain(
    fixture: Mapping[str, Any],
    instantiations: Sequence[Mapping[str, int]],
    z3_module: Any,
) -> JsonDict:
    int_sort = z3_module.IntSort()
    predicate = z3_module.Function(str(fixture["predicate"]), int_sort, z3_module.BoolSort())
    solver = z3_module.Solver()
    for premise in fixture["premises"]:
        solver.add(predicate(z3_module.IntVal(int(premise))))
    step = int(fixture["step"])
    for instantiation in instantiations:
        x_value = int(instantiation["x"])
        solver.add(
            z3_module.Implies(
                predicate(z3_module.IntVal(x_value)),
                predicate(z3_module.IntVal(x_value + step)),
            )
        )
    solver.add(z3_module.Not(predicate(z3_module.IntVal(int(fixture["target_arg"])))))
    status = solver.check()
    return {
        "valid": status == z3_module.unsat,
        "exact_checked": True,
        "exact_authority": "z3_solver",
        "solver_status": _status_name(status, z3_module),
        "instantiations": [dict(row) for row in instantiations],
        "failure_reason": "" if status == z3_module.unsat else "proposal_does_not_prove_target",
    }


def _expr_int_value(expr: Mapping[str, Any], instantiation: Mapping[str, int], z3_module: Any) -> Any:
    if expr["op"] == "square":
        x_value = int(instantiation["x"])
        return z3_module.IntVal(x_value * x_value)
    total = int(expr.get("constant", 0))
    for name, coefficient in dict(expr["coefficients"]).items():
        total += int(coefficient) * int(instantiation[str(name)])
    return z3_module.IntVal(total)


def _normalise_instantiations(
    instantiations: Sequence[Mapping[str, Any]],
    fixture: Mapping[str, Any],
) -> tuple[list[dict[str, int]], list[str]]:
    variables = [str(name) for name in fixture["variables"]]
    rows: list[dict[str, int]] = []
    errors: list[str] = []
    for raw in instantiations:
        if not isinstance(raw, Mapping):
            errors.append("instantiation_not_object")
            continue
        source = dict(raw)
        if "var" in source and "value" in source and len(variables) == 1:
            source = {variables[0]: source["value"]}
        row: dict[str, int] = {}
        for name in variables:
            if name not in source:
                errors.append(f"missing_{name}")
                continue
            try:
                row[name] = int(source[name])
            except (TypeError, ValueError):
                errors.append(f"non_integer_{name}")
        if sorted(row) == sorted(variables):
            rows.append(row)
    return rows, errors


def _guidance_prompt(fixture: Mapping[str, Any]) -> str:
    variables = [str(name) for name in fixture["variables"]]
    shape_fields = ",".join(f'"{name}":INTEGER' for name in variables)
    return (
        "You propose quantified SMT instantiations. Z3 will validate them.\n"
        "Return only JSON with no markdown or prose.\n"
        f"Required shape: {{\"fixture_id\":\"{fixture['fixture_id']}\","
        f"\"instantiations\":[{{{shape_fields}}}]}}\n"
        "Use multiple instantiation objects only when one rule instance is not enough.\n"
        f"Allowed integer terms: {json.dumps(fixture['candidate_domain'])}\n"
        f"Problem: {_fixture_text(fixture)}\n"
        "JSON:"
    )


def _fixture_text(fixture: Mapping[str, Any]) -> str:
    if fixture["kind"] == "predicate_chain":
        return (
            f"forall x. {fixture['predicate']}(x) implies "
            f"{fixture['predicate']}(x+{fixture['step']}); "
            f"premise {fixture['predicate']}({fixture['premises'][0]}); "
            f"target {fixture['predicate']}({fixture['target_arg']})."
        )
    expr = fixture["expr"]
    if expr["op"] == "square":
        rhs = "x*x"
    else:
        terms = [
            f"{coefficient}*{name}" for name, coefficient in dict(expr["coefficients"]).items()
        ]
        if int(expr.get("constant", 0)):
            terms.append(str(expr["constant"]))
        rhs = " + ".join(terms)
    args = ", ".join(str(value) for value in fixture["target_args"])
    return (
        f"forall {', '.join(fixture['variables'])}. {fixture['function']}("
        f"{', '.join(fixture['variables'])}) = {rhs}; target "
        f"{fixture['function']}({args}) = {fixture['target_value']}."
    )


def _select_models(cache_resolution: Mapping[str, str | None]) -> list[JsonDict]:
    selected = []
    for index, model in enumerate(SOTA_GGUF_MODELS):
        path = cache_resolution.get(model["hf_id"])
        if path:
            selected.append(
                {
                    "name": model["name"],
                    "hf_id": model["hf_id"],
                    "model_path": path,
                    "gpu": min(index, 1),
                    "role": model["role"],
                    "family": _model_family(model["hf_id"]),
                }
            )
    return selected[:1]


def _resolve_cache(resolve_gguf_func: ResolveGgufFn, config: ExperimentConfig) -> dict[str, str | None]:
    return {hf_id: resolve_gguf_func(hf_id, config.preferred_quant) for hf_id in MANDATED_MODEL_IDS}


def _precondition_blocker(
    exp3057_ready: bool,
    z3_module: Any,
    selected_models: Sequence[Mapping[str, Any]],
) -> str:
    if not exp3057_ready:
        return "blocked_exp3057_not_ready: Exp 3057 is absent or not ready"
    if z3_module is None:
        return "blocked_exact_solver_unavailable: Z3 import failed"
    if not selected_models:
        return "blocked_sota_gguf_unavailable: no mandated local SOTA GGUF resolved"
    return ""


def _exp3057_ready(repo_root: Path) -> bool:
    path = repo_root / EXP3057_REL_PATH
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("solution_verifier_calibration_ready") is True


def _model_spec(model: Mapping[str, Any]) -> JsonDict:
    evidence = _file_evidence(model["model_path"], full_limit_bytes=512 * 1024 * 1024)
    return {
        "name": model["name"],
        "hf_id": model["hf_id"],
        "model_path": model["model_path"],
        "gpu": model["gpu"],
        "role": model["role"],
        "family": model["family"],
        "model_hash_or_cache_path": evidence.get("model_hash_or_cache_path"),
        "checksum_feasibility": {
            "method": evidence.get("method"),
            "full_sha256_feasible": bool(evidence.get("full_sha256_feasible")),
            "size_bytes": evidence.get("size_bytes"),
        },
    }


def _substrate(
    *,
    config: ExperimentConfig,
    cache_resolution: Mapping[str, str | None],
    selected_models: Sequence[Mapping[str, Any]],
    duration_s: float,
    exp3057_ready: bool,
    exact_solver_available: bool,
    repo_commit_func: RepoCommitFn,
) -> JsonDict:
    return {
        "runtime": "llama_cpp",
        "local_gguf_inference": bool(selected_models),
        "live_llm_inference": bool(selected_models),
        "exact_solver": "z3" if exact_solver_available else "unavailable",
        "exact_solver_path": EXACT_SOLVER_PATH,
        "exp3057_ready": exp3057_ready,
        "cuda_probe": _cuda_probe(),
        "gpu_inventory": _gpu_inventory(),
        "python_environment": _python_environment(),
        "repo_commit": repo_commit_func(config.repo_root),
        "gguf_cache_resolution": dict(cache_resolution),
        "selected_model_paths": [str(model["model_path"]) for model in selected_models],
        "decode_config": config.effective_decode_config(),
        "load_config": config.effective_load_config(),
        "seed": config.seed,
        "llm_substrate": {
            "runtime": "llama_cpp",
            "models_used": [str(model["hf_id"]) for model in selected_models],
        },
        "solver_substrate": {
            "authority": "z3_solver" if exact_solver_available else "unavailable",
            "solver_only_fallback_preserved": exact_solver_available,
        },
        "wall_clock_duration_s": duration_s,
    }


def _honest_verdict(
    ready: bool,
    metrics: Mapping[str, int],
    runtime_blocker: str | None,
) -> str:
    if ready:
        return (
            "complete: llm_guided_smt_pilot_ready=true; "
            f"guided_success_count={metrics['guided_success_count']}; "
            f"solver_only_success_count={metrics['solver_only_success_count']}; "
            f"invalid_llm_proposal_count={metrics['invalid_llm_proposal_count']}"
        )
    if runtime_blocker:
        if runtime_blocker.startswith("blocked_"):
            return runtime_blocker
        return f"blocked_sota_gguf_unavailable: live GGUF runtime failed: {runtime_blocker}"
    return "blocked_guided_smt_pilot_incomplete"


def _function_fixture(
    fixture_id: str,
    function: str,
    variables: Sequence[str],
    target_args: Sequence[int],
    target_value: int,
    expr: Mapping[str, Any],
) -> JsonDict:
    domain = sorted(set(range(0, 7)).union(int(value) for value in target_args))
    return {
        "fixture_id": fixture_id,
        "kind": "function_value",
        "function": function,
        "variables": list(variables),
        "target_args": [int(value) for value in target_args],
        "target_value": int(target_value),
        "expr": dict(expr),
        "candidate_domain": domain,
        "max_solver_instantiations": 1,
    }


def _affine(coefficients: Mapping[str, int], constant: int) -> JsonDict:
    return {"op": "affine", "coefficients": dict(coefficients), "constant": int(constant)}


def _parse_json_object(text: str) -> JsonDict:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return dict(value) if isinstance(value, Mapping) else {}
    return {}


def _status_name(status: Any, z3_module: Any) -> str:
    if status == z3_module.sat:
        return "sat"
    if status == z3_module.unsat:
        return "unsat"
    return "unknown"


def _model_family(hf_id: str) -> str:
    lowered = hf_id.lower()
    if "qwen" in lowered:
        return "qwen"
    if "gemma" in lowered:
        return "gemma"
    return hf_id.split("/", 1)[0].lower()


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _default_llama_factory(**kwargs: Any) -> Any:  # pragma: no cover - live hardware path.
    from llama_cpp import Llama  # noqa: PLC0415

    return Llama(**kwargs)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    return _sha256_text(json.dumps(dict(payload), sort_keys=True, separators=(",", ":")))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _relative_to(root: Path, path: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path
