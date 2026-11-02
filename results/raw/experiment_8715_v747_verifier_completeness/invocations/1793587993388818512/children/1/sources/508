"""Exp 3071 VERGE-style MCS SMT correction-feedback pilot.

Spec refs: REQ-VERIFY-3071,
           SCENARIO-VERIFY-3071,
           SCENARIO-VERIFY-3071-BLOCKED.

This module keeps the VERGE idea intentionally small: Z3 turns bad tiny
integer records into machine-readable correction feedback, a mandated local
GGUF proposes repaired records from that feedback alone, and Z3 validates the
repairs. The model is only a proposer; the exact solver remains the authority.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
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

ARTIFACT = "experiment_3071_verge_mcs_smt_correction_pilot_v1"
ARTIFACT_FILENAME = f"{ARTIFACT}.json"
SCRIPT_FILENAME = f"{ARTIFACT}.py"
SCHEMA = "carnot.verge_mcs_smt_correction_pilot.v1"
RUN_DATE = "20260525"
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results") / ARTIFACT_FILENAME
PILOT_ROWS_REL_PATH = Path("results") / "verge_mcs_smt_correction_pilot_3071" / "pilot_rows.jsonl"
EXACT_SOLVER_REL_PATH = Path("python/carnot/eval/verge_mcs_smt_correction_pilot_v1.py")
EXACT_SOLVER_PATH = f"{EXACT_SOLVER_REL_PATH.as_posix()}::validate_candidate_with_z3"
DEFAULT_SEED = 307100
DEFAULT_DECODE_CONFIG: JsonDict = {
    "max_tokens": 96,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "repeat_penalty": 1.05,
    "stop": ["\n\n"],
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
REQUIRED_ARTIFACT_FIELDS = (
    "mcs_feedback_ready",
    "formal_fallback_preserved",
    "mcs_count",
    "guided_success_count",
    "solver_only_success_count",
    "invalid_llm_proposal_count",
    "correction_subset_useful_count",
    "exact_solver_path",
    "models_used",
    "model_specs",
    "legacy_smoke_only_used",
    "prompt_hashes",
    "inference_substrate",
    "honest_verdict",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for the Exp 3071 correction-feedback pilot."""

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


def build_correction_fixtures() -> list[JsonDict]:
    """Return tiny fixtures spanning valid, invalid, incomplete, and bounded states."""

    sum_relation = _eq_affine("sum_relation", "total", {"a": 1, "b": 1})
    difference_relation = _eq_affine("difference_relation", "delta", {"left": 1, "right": -1})
    score_relation = _eq_affine("weighted_score", "score", {"p": 2, "q": 3})
    return [
        {
            "fixture_id": "sum-total-valid",
            "kind": "already_valid",
            "required_fields": ["a", "b", "total"],
            "candidate": {"a": 2, "b": 3, "total": 5},
            "mutable_fields": ["total"],
            "constraints": [sum_relation],
        },
        {
            "fixture_id": "sum-total-high",
            "kind": "invalid_assignment",
            "required_fields": ["a", "b", "total"],
            "candidate": {"a": 2, "b": 3, "total": 6},
            "mutable_fields": ["total"],
            "constraints": [sum_relation],
        },
        {
            "fixture_id": "sum-total-missing",
            "kind": "underconstrained_missing_field",
            "required_fields": ["a", "b", "total"],
            "candidate": {"a": 4, "b": 1},
            "mutable_fields": ["total"],
            "constraints": [sum_relation],
        },
        {
            "fixture_id": "bounded-x-high",
            "kind": "overconstrained_bound",
            "required_fields": ["x"],
            "candidate": {"x": 11},
            "mutable_fields": ["x"],
            "constraints": [
                {
                    "constraint_id": "x_bounds",
                    "op": "bounds",
                    "target": "x",
                    "lower": 0,
                    "upper": 10,
                }
            ],
        },
        {
            "fixture_id": "difference-delta-wrong",
            "kind": "invalid_assignment",
            "required_fields": ["left", "right", "delta"],
            "candidate": {"left": 9, "right": 4, "delta": 6},
            "mutable_fields": ["delta"],
            "constraints": [difference_relation],
        },
        {
            "fixture_id": "weighted-score-low",
            "kind": "invalid_assignment",
            "required_fields": ["p", "q", "score"],
            "candidate": {"p": 2, "q": 5, "score": 18},
            "mutable_fields": ["score"],
            "constraints": [score_relation],
        },
    ]


def validate_candidate_with_z3(
    fixture: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    z3_module: Any = _z3,
) -> JsonDict:
    """Validate a complete candidate with Z3 as the only authority."""

    if z3_module is None:
        return _validation_result(
            valid=False,
            exact_checked=False,
            exact_authority="z3_unavailable",
            solver_status="not_run",
            candidate=dict(candidate),
            failure_reason="z3_solver_unavailable",
        )
    normalised, errors = _normalise_candidate(candidate)
    if errors:
        return _validation_result(
            valid=False,
            exact_checked=False,
            exact_authority="z3_solver",
            solver_status="not_run",
            candidate=normalised,
            failure_reason=";".join(errors),
        )
    required = _string_list(fixture["required_fields"])
    missing = [field for field in required if field not in normalised]
    status = _candidate_solver_status(fixture, normalised, z3_module)
    if missing:
        return _validation_result(
            valid=False,
            exact_checked=True,
            exact_authority="z3_solver",
            solver_status=f"{status}_partial",
            candidate=normalised,
            failure_reason=f"missing_required_fields:{','.join(missing)}",
        )
    return _validation_result(
        valid=status == "sat",
        exact_checked=True,
        exact_authority="z3_solver",
        solver_status=status,
        candidate=normalised,
        failure_reason="" if status == "sat" else "candidate_violates_constraints",
    )


def generate_correction_feedback(
    fixture: Mapping[str, Any],
    *,
    z3_module: Any = _z3,
) -> JsonDict:
    """Generate exact correction feedback for one fixture."""

    validation = validate_candidate_with_z3(fixture, fixture["candidate"], z3_module=z3_module)
    if validation["valid"]:
        return {
            "fixture_id": fixture["fixture_id"],
            "feedback_type": "verified",
            "exact_authority": validation["exact_authority"],
            "exact_checked": validation["exact_checked"],
            "solver_status": validation["solver_status"],
            "validation": validation,
            "correction_subset": None,
        }
    if z3_module is None:
        return {
            "fixture_id": fixture["fixture_id"],
            "feedback_type": "unavailable",
            "exact_authority": "z3_unavailable",
            "exact_checked": False,
            "solver_status": "not_run",
            "validation": validation,
            "correction_subset": {
                "candidate_fields": [],
                "minimal_assignment_ids": [],
                "suggested_assignments": {},
                "failing_constraint_ids": [],
            },
        }
    normalised = dict(validation["candidate"])
    missing = [
        field for field in _string_list(fixture["required_fields"]) if field not in normalised
    ]
    if missing:
        subset = _refinement_subset(fixture, normalised, missing, z3_module)
        feedback_type = "refinement"
    else:
        subset = _minimal_correction_subset(fixture, normalised, z3_module)
        feedback_type = "mcs"
    return {
        "fixture_id": fixture["fixture_id"],
        "feedback_type": feedback_type,
        "exact_authority": "z3_solver",
        "exact_checked": True,
        "solver_status": "sat" if subset["suggested_assignments"] else validation["solver_status"],
        "validation": validation,
        "correction_subset": subset,
        "prompt_feedback": _prompt_feedback(fixture, feedback_type, subset),
    }


def solver_only_repair(feedback: Mapping[str, Any], fixture: Mapping[str, Any]) -> JsonDict:
    """Apply exact correction feedback directly and validate the repaired candidate."""

    candidate = dict(fixture["candidate"])
    candidate.update(dict(feedback["correction_subset"]["suggested_assignments"]))
    validation = validate_candidate_with_z3(fixture, candidate)
    return validation | {"candidate": candidate}


def parse_llm_candidate(text: str, fixture: Mapping[str, Any]) -> JsonDict:
    """Parse local-GGUF text into a candidate record."""

    parsed = _parse_json_object(text)
    if not parsed:
        return {"valid_parse": False, "candidate": {}, "parse_error": "json_object_missing"}
    fixture_id = parsed.get("fixture_id")
    if fixture_id is not None and str(fixture_id) != str(fixture["fixture_id"]):
        return {"valid_parse": False, "candidate": {}, "parse_error": "fixture_id_mismatch"}
    raw_candidate = parsed.get("candidate")
    if not isinstance(raw_candidate, Mapping):
        return {"valid_parse": False, "candidate": {}, "parse_error": "candidate_not_object"}
    candidate, errors = _normalise_candidate(raw_candidate)
    if errors:
        return {"valid_parse": False, "candidate": candidate, "parse_error": ";".join(errors)}
    return {"valid_parse": True, "candidate": candidate, "parse_error": ""}


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    resolve_gguf_func: ResolveGgufFn = resolve_cached_gguf,
    llama_factory: LlamaFactory | None = None,
    z3_module: Any = _z3,
    monotonic: ClockFn = time.monotonic,
    repo_commit_func: RepoCommitFn = _repo_commit,
) -> JsonDict:
    """Run the live correction-feedback pilot and write the terminal artifact."""

    active = config or ExperimentConfig()
    started = monotonic()
    cache_resolution = _resolve_cache(resolve_gguf_func, active)
    selected_models = _select_models(cache_resolution)
    blocker = _precondition_blocker(z3_module, selected_models)
    if blocker:
        artifact = _blocked_artifact(
            config=active,
            cache_resolution=cache_resolution,
            blocker=blocker,
            duration_s=round(monotonic() - started, 6),
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
        runtime_blocker = f"blocked_solver_or_sota_unavailable: {type(exc).__name__}: {exc}"
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
        exact_solver_available=z3_module is not None,
        repo_commit_func=repo_commit_func,
    )
    validate_artifact(artifact)
    _write_json(active.artifact_path(), artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 3071 artifact violates the correction-feedback contract."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("legacy_smoke_only_used") is not False:
        raise ValueError("legacy smoke evidence cannot satisfy REQ-VERIFY-3071")
    verdict = str(artifact.get("honest_verdict", ""))
    if artifact.get("mcs_feedback_ready") is not True:
        if not verdict.startswith("blocked_solver_or_sota_unavailable"):
            raise ValueError("honest_verdict must disclose the blocked precondition")
        return
    if not artifact.get("model_specs"):
        raise ValueError("model_specs must be present when MCS feedback is ready")
    if not artifact.get("prompt_hashes"):
        raise ValueError("prompt_hashes must be present when MCS feedback is ready")
    if artifact.get("formal_fallback_preserved") is not True:
        raise ValueError("formal_fallback_preserved must be true when ready")
    if int(artifact.get("mcs_count") or 0) <= 0:
        raise ValueError("mcs_count must be positive when ready")
    if int(artifact.get("guided_success_count") or 0) <= 0:
        raise ValueError("guided_success_count must be positive when ready")
    if int(artifact.get("solver_only_success_count") or 0) <= 0:
        raise ValueError("solver_only_success_count must be positive when ready")
    if int(artifact.get("correction_subset_useful_count") or 0) <= 0:
        raise ValueError("correction_subset_useful_count must be positive when ready")
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
    fixtures = build_correction_fixtures()
    fixture_by_id = {fixture["fixture_id"]: fixture for fixture in fixtures}
    decode_config = config.effective_decode_config()
    load_config = config.effective_load_config(int(selected_model.get("gpu", 0)))
    llm = llama_factory(model_path=str(selected_model["model_path"]), **load_config)
    rows: list[JsonDict] = []
    prompt_hashes: list[str] = []
    try:
        for fixture in fixtures:
            feedback = generate_correction_feedback(fixture, z3_module=z3_module)
            if feedback["feedback_type"] == "verified":
                continue
            prompt = _repair_prompt(feedback)
            raw = llm(prompt, **dict(decode_config), seed=config.seed)
            text = _normalize_output(_extract_text(raw))
            prompt_hash = _sha256_text(prompt)
            parsed = parse_llm_candidate(text, fixture_by_id[feedback["fixture_id"]])
            guided_validation = (
                validate_candidate_with_z3(fixture, parsed["candidate"], z3_module=z3_module)
                if parsed["valid_parse"]
                else _validation_result(
                    valid=False,
                    exact_checked=False,
                    exact_authority="z3_solver",
                    solver_status="not_run",
                    candidate=parsed["candidate"],
                    failure_reason=parsed["parse_error"],
                )
            )
            fallback = solver_only_repair(feedback, fixture)
            rows.append(
                {
                    "fixture_id": fixture["fixture_id"],
                    "fixture_kind": fixture["kind"],
                    "feedback_type": feedback["feedback_type"],
                    "correction_subset": feedback["correction_subset"],
                    "prompt_hash": prompt_hash,
                    "raw_output_hash": _sha256_text(text),
                    "parse": parsed,
                    "guided_validation": guided_validation,
                    "solver_only_fallback": fallback,
                    "guided_success": bool(guided_validation["valid"]),
                    "solver_only_success": bool(fallback["valid"]),
                    "correction_feedback_useful": bool(fallback["valid"]),
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
    exact_solver_available: bool,
    repo_commit_func: RepoCommitFn,
) -> JsonDict:
    metrics = _metrics(rows)
    formal_fallback_preserved = _formal_fallback_preserved(rows, exact_solver_available)
    model_specs = [_model_spec(row) for row in selected_models] if runtime_blocker is None else []
    models_used = [str(row["hf_id"]) for row in selected_models] if runtime_blocker is None else []
    ready = (
        runtime_blocker is None
        and exact_solver_available
        and bool(model_specs)
        and bool(prompt_hashes)
        and metrics["mcs_count"] > 0
        and metrics["guided_success_count"] > 0
        and metrics["solver_only_success_count"] > 0
        and metrics["correction_subset_useful_count"] > 0
        and formal_fallback_preserved
    )
    artifact = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "mcs_feedback_ready": ready,
        "formal_fallback_preserved": formal_fallback_preserved,
        "mcs_count": metrics["mcs_count"],
        "guided_success_count": metrics["guided_success_count"],
        "solver_only_success_count": metrics["solver_only_success_count"],
        "invalid_llm_proposal_count": metrics["invalid_llm_proposal_count"],
        "correction_subset_useful_count": metrics["correction_subset_useful_count"],
        "exact_solver_path": EXACT_SOLVER_PATH,
        "models_used": models_used,
        "model_specs": model_specs,
        "legacy_smoke_only_used": False,
        "prompt_hashes": list(prompt_hashes) if runtime_blocker is None else [],
        "inference_substrate": _substrate(
            config=config,
            cache_resolution=cache_resolution,
            selected_models=selected_models if runtime_blocker is None else [],
            duration_s=duration_s,
            exact_solver_available=exact_solver_available,
            repo_commit_func=repo_commit_func,
        ),
        "honest_verdict": _honest_verdict(ready, metrics, runtime_blocker),
        "fixture_count": len(build_correction_fixtures()),
        "pilot_rows_path": str(_relative_to(config.repo_root, config.pilot_rows_path())),
        "pilot_rows_sha256": _sha256_file(config.pilot_rows_path()) if rows else "",
        "decode_config": config.effective_decode_config(),
        "seed": config.seed,
        "tests_or_checks_run": list(config.tests_run),
        "runtime_blocker": runtime_blocker,
        "preconditions": {
            "exact_solver_available": exact_solver_available,
            "mandated_gguf_resolved": bool(selected_models),
            "mandated_gguf_loaded": runtime_blocker is None and bool(selected_models),
        },
        "guidance_vs_solver_only": {
            "guided_minus_solver_only_success_count": metrics["guided_success_count"]
            - metrics["solver_only_success_count"],
            "guidance_helped": metrics["guided_success_count"]
            > metrics["solver_only_success_count"],
        },
        "pilot_rows": [dict(row) for row in rows],
        "source_context": {
            "exp3044": "results/experiment_3044_smt_sat_validator_tree_exactness_upgrade_v1.json",
            "exp3058": "results/experiment_3058_aquaforte_style_llm_guided_smt_pilot_v1.json",
            "reference": "research-references.md#VERGE",
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
                    "mcs_count",
                    "guided_success_count",
                    "solver_only_success_count",
                    "invalid_llm_proposal_count",
                    "correction_subset_useful_count",
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
        exact_solver_available=exact_solver_available,
        repo_commit_func=repo_commit_func,
    )
    validate_artifact(artifact)
    return artifact


def _metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    guided_success = sum(1 for row in rows if row.get("guided_success") is True)
    solver_success = sum(1 for row in rows if row.get("solver_only_success") is True)
    useful = sum(1 for row in rows if row.get("correction_feedback_useful") is True)
    return {
        "mcs_count": len(rows),
        "guided_success_count": guided_success,
        "solver_only_success_count": solver_success,
        "invalid_llm_proposal_count": len(rows) - guided_success,
        "correction_subset_useful_count": useful,
    }


def _formal_fallback_preserved(
    rows: Sequence[Mapping[str, Any]], exact_solver_available: bool
) -> bool:
    return (
        bool(rows)
        and exact_solver_available
        and all(
            row.get("solver_only_fallback", {}).get("exact_authority") == "z3_solver"
            and row.get("solver_only_fallback", {}).get("exact_checked") is True
            for row in rows
        )
    )


def _refinement_subset(
    fixture: Mapping[str, Any],
    candidate: Mapping[str, int],
    missing: Sequence[str],
    z3_module: Any,
) -> JsonDict:
    model = _solve_model(fixture, candidate, z3_module)
    return {
        "candidate_fields": list(missing),
        "minimal_assignment_ids": [],
        "suggested_assignments": _model_assignments(model, missing, fixture, z3_module),
        "failing_constraint_ids": _constraint_ids(fixture),
        "refinement_reason": "missing_required_fields",
    }


def _minimal_correction_subset(
    fixture: Mapping[str, Any],
    candidate: Mapping[str, int],
    z3_module: Any,
) -> JsonDict:
    mutable = [field for field in _string_list(fixture["mutable_fields"]) if field in candidate]
    for size in range(1, len(mutable) + 1):
        for fields in combinations(mutable, size):
            kept = {key: value for key, value in candidate.items() if key not in fields}
            model = _solve_model(fixture, kept, z3_module)
            if model is not None:
                return {
                    "candidate_fields": list(fields),
                    "minimal_assignment_ids": [f"candidate.{field}" for field in fields],
                    "suggested_assignments": _model_assignments(model, fields, fixture, z3_module),
                    "failing_constraint_ids": _constraint_ids(fixture),
                }
    return {
        "candidate_fields": [],
        "minimal_assignment_ids": [],
        "suggested_assignments": {},
        "failing_constraint_ids": _constraint_ids(fixture),
    }


def _solve_model(
    fixture: Mapping[str, Any], candidate: Mapping[str, int], z3_module: Any
) -> Any | None:
    variables = _z3_variables(fixture, candidate, z3_module)
    solver = z3_module.Solver()
    solver.add(*_candidate_assertions(candidate, variables, z3_module))
    solver.add(*_constraint_assertions(fixture, variables, z3_module))
    if solver.check() == z3_module.sat:
        return solver.model()
    return None


def _model_assignments(
    model: Any | None,
    fields: Sequence[str],
    fixture: Mapping[str, Any],
    z3_module: Any,
) -> dict[str, int]:
    variables = _z3_variables(fixture, {}, z3_module)
    if model is None:
        return {}
    return {
        field: int(model.eval(variables[field], model_completion=True).as_long())
        for field in fields
    }


def _candidate_solver_status(
    fixture: Mapping[str, Any],
    candidate: Mapping[str, int],
    z3_module: Any,
) -> str:
    return "sat" if _solve_model(fixture, candidate, z3_module) is not None else "unsat"


def _z3_variables(
    fixture: Mapping[str, Any],
    candidate: Mapping[str, int],
    z3_module: Any,
) -> dict[str, Any]:
    names = set(_string_list(fixture["required_fields"])) | set(candidate)
    for constraint in _mapping_list(fixture["constraints"]):
        names.add(str(constraint["target"]))
        names.update(str(name) for name in dict(constraint.get("terms") or {}))
    return {name: z3_module.Int(name) for name in sorted(names)}


def _candidate_assertions(
    candidate: Mapping[str, int],
    variables: Mapping[str, Any],
    z3_module: Any,
) -> list[Any]:
    return [
        variables[field] == z3_module.IntVal(value) for field, value in sorted(candidate.items())
    ]


def _constraint_assertions(
    fixture: Mapping[str, Any],
    variables: Mapping[str, Any],
    z3_module: Any,
) -> list[Any]:
    rows: list[Any] = []
    for constraint in _mapping_list(fixture["constraints"]):
        op = str(constraint["op"])
        target = variables[str(constraint["target"])]
        if op == "bounds":
            rows.extend(
                [
                    target >= z3_module.IntVal(int(constraint["lower"])),
                    target <= z3_module.IntVal(int(constraint["upper"])),
                ]
            )
        else:
            expression = z3_module.IntVal(int(constraint.get("constant", 0)))
            for name, coefficient in sorted(dict(constraint["terms"]).items()):
                expression += int(coefficient) * variables[str(name)]
            rows.append(target == expression)
    return rows


def _repair_prompt(feedback: Mapping[str, Any]) -> str:
    prompt_feedback = feedback.get("prompt_feedback") or {}
    return (
        "You repair tiny integer candidates. Return only JSON, no markdown.\n"
        "Use only the correction feedback below; do not invent new fields.\n"
        "The candidate must include every key from complete_candidate_template.\n"
        f"Correction feedback only:\n{json.dumps(prompt_feedback, sort_keys=True)}\n"
        "Required shape: "
        f'{{"fixture_id":"{prompt_feedback.get("fixture_id")}",'
        f'"candidate":{json.dumps(prompt_feedback.get("complete_candidate_template", {}), sort_keys=True)}}}\n'
        "JSON:"
    )


def _prompt_feedback(
    fixture: Mapping[str, Any],
    feedback_type: str,
    subset: Mapping[str, Any],
) -> JsonDict:
    candidate = dict(fixture["candidate"])
    immutable = {
        field: value
        for field, value in candidate.items()
        if field not in set(_string_list(fixture["mutable_fields"]))
    }
    complete_candidate = immutable | dict(subset["suggested_assignments"])
    return {
        "fixture_id": fixture["fixture_id"],
        "feedback_type": feedback_type,
        "required_fields": list(fixture["required_fields"]),
        "immutable_assignments": immutable,
        "mutable_fields": list(fixture["mutable_fields"]),
        "candidate_fields_to_repair": list(subset["candidate_fields"]),
        "suggested_assignments": dict(subset["suggested_assignments"]),
        "complete_candidate_template": complete_candidate,
        "failing_constraint_ids": list(subset["failing_constraint_ids"]),
    }


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


def _resolve_cache(
    resolve_gguf_func: ResolveGgufFn, config: ExperimentConfig
) -> dict[str, str | None]:
    return {hf_id: resolve_gguf_func(hf_id, config.preferred_quant) for hf_id in MANDATED_MODEL_IDS}


def _precondition_blocker(z3_module: Any, selected_models: Sequence[Mapping[str, Any]]) -> str:
    if z3_module is None:
        return "blocked_solver_or_sota_unavailable: Z3 import failed"
    if not selected_models:
        return "blocked_solver_or_sota_unavailable: no mandated local SOTA GGUF resolved"
    return ""


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
    exact_solver_available: bool,
    repo_commit_func: RepoCommitFn,
) -> JsonDict:
    return {
        "runtime": "llama_cpp",
        "exact_solver": "z3" if exact_solver_available else "unavailable",
        "exact_solver_path": EXACT_SOLVER_PATH,
        "live_llm_inference": bool(selected_models),
        "local_gguf_inference": bool(selected_models),
        "legacy_smoke_only_used": False,
        "gguf_cache_resolution": dict(cache_resolution),
        "selected_model_paths": [str(row["model_path"]) for row in selected_models],
        "llm_substrate": {
            "runtime": "llama_cpp",
            "models_used": [str(row["hf_id"]) for row in selected_models],
        },
        "solver_substrate": {
            "authority": "z3_solver" if exact_solver_available else "unavailable",
            "solver_only_fallback_preserved": True,
        },
        "decode_config": config.effective_decode_config(),
        "load_config": config.effective_load_config(int(selected_models[0]["gpu"]))
        if selected_models
        else {},
        "seed": config.seed,
        "cuda_probe": _cuda_probe(),
        "gpu_inventory": _gpu_inventory(),
        "python_environment": _python_environment(),
        "repo_commit": repo_commit_func(config.repo_root),
        "wall_clock_duration_s": duration_s,
    }


def _honest_verdict(ready: bool, metrics: Mapping[str, int], runtime_blocker: str | None) -> str:
    if ready:
        return (
            "complete: mcs_feedback_ready=true; "
            f"mcs_count={metrics['mcs_count']}; "
            f"guided_success_count={metrics['guided_success_count']}; "
            f"solver_only_success_count={metrics['solver_only_success_count']}; "
            f"invalid_llm_proposal_count={metrics['invalid_llm_proposal_count']}"
        )
    return (
        runtime_blocker
        or "blocked_solver_or_sota_unavailable: correction feedback pilot incomplete"
    )


def _default_llama_factory(
    **kwargs: Any,
) -> Any:  # pragma: no cover - live path covered by artifact run.
    from llama_cpp import Llama

    return Llama(**kwargs)


def _validation_result(
    *,
    valid: bool,
    exact_checked: bool,
    exact_authority: str,
    solver_status: str,
    candidate: Mapping[str, Any],
    failure_reason: str,
) -> JsonDict:
    return {
        "valid": valid,
        "exact_checked": exact_checked,
        "exact_authority": exact_authority,
        "solver_status": solver_status,
        "candidate": dict(candidate),
        "failure_reason": failure_reason,
    }


def _eq_affine(
    constraint_id: str,
    target: str,
    terms: Mapping[str, int],
    constant: int = 0,
) -> JsonDict:
    return {
        "constraint_id": constraint_id,
        "op": "eq_affine",
        "target": target,
        "terms": dict(terms),
        "constant": constant,
    }


def _normalise_candidate(candidate: Mapping[str, Any]) -> tuple[dict[str, int], list[str]]:
    rows: dict[str, int] = {}
    errors: list[str] = []
    for field, value in dict(candidate).items():
        try:
            rows[str(field)] = int(value)
        except (TypeError, ValueError):
            errors.append(f"non_integer_{field}")
    return rows, errors


def _parse_json_object(text: str) -> JsonDict | None:
    start = text.find("{")
    if start < 0:
        return None
    try:
        parsed, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return dict(parsed) if isinstance(parsed, Mapping) else None


def _constraint_ids(fixture: Mapping[str, Any]) -> list[str]:
    return [str(row["constraint_id"]) for row in _mapping_list(fixture["constraints"])]


def _model_family(hf_id: str) -> str:
    lowered = hf_id.lower()
    if "qwen" in lowered:
        return "qwen"
    if "gemma" in lowered:
        return "gemma"
    return "unknown"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_to(root: Path, path: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def _mapping_list(value: Any) -> list[JsonDict]:
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, (list, tuple)) else []
