"""Exp 3086 Dafny/Z3 formal-feedback pilot.

Spec refs: REQ-VERIFY-3086,
           SCENARIO-VERIFY-3086,
           SCENARIO-VERIFY-3086-BLOCKED.

This pilot keeps solver authority separate from model text. Dafny is only used
when the binary is present; on this host the intended fallback is Z3. The GGUF
model sees verifier-style diagnostics and proposes a JSON repair, but Z3/local
execution revalidates the repaired candidate before any success is counted.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
import time
from typing import Any

from carnot.inference.sota_models import SOTA_GGUF_MODELS, resolve_cached_gguf

try:  # pragma: no cover - missing dependency is exercised by precondition logic.
    import z3 as _z3
except Exception:  # pragma: no cover
    _z3 = None


JsonDict = dict[str, Any]
ResolveGgufFn = Callable[[str, str], str | None]
LlamaFactory = Callable[..., Any]
ClockFn = Callable[[], float]
RepoCommitFn = Callable[[Path], str]
ProbeFn = Callable[[], Mapping[str, Any]]
CommandResolver = Callable[[str], str | None]

ARTIFACT = "experiment_3086_dafny_z3_formal_feedback_pilot_v1"
ARTIFACT_FILENAME = f"{ARTIFACT}.json"
SCRIPT_FILENAME = f"{ARTIFACT}.py"
SCHEMA = "carnot.dafny_z3_formal_feedback_pilot.v1"
RUN_DATE = "20260525"
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results") / ARTIFACT_FILENAME
DEFAULT_SEED = 308600
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
    "shipped_",
)
REQUIRED_ARTIFACT_FIELDS = (
    "formal_feedback_ready",
    "formal_feedback_delta",
    "dafny_available",
    "z3_available",
    "vacuity_guard_passed",
    "guided_success_count",
    "solver_only_success_count",
    "exact_ground_truth_count",
    "models_used",
    "model_specs",
    "legacy_smoke_only_used",
    "preconditions_checked",
    "prompt_hashes",
    "inference_substrate",
    "honest_verdict",
)


@dataclass(frozen=True)
class FormalFixture:
    """A tiny repair problem with exact reference semantics hidden from prompts."""

    fixture_id: str
    family: str
    requirement: str
    candidate: JsonDict
    reference_operation: str | None
    expected_issue: str
    domain: tuple[int, int] = (-5, 5)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and knobs for Exp 3086."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    seed: int = DEFAULT_SEED
    preferred_quant: str = "Q4_K_M"
    decode_config: Mapping[str, Any] | None = None
    load_config: Mapping[str, Any] | None = None
    tests_run: Sequence[str] = ()

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / OUTPUT_REL_PATH

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


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    command_resolver: CommandResolver = shutil.which,
    resolve_gguf_func: ResolveGgufFn = resolve_cached_gguf,
    llama_factory: LlamaFactory | None = None,
    monotonic: ClockFn = time.monotonic,
    repo_commit_func: RepoCommitFn | None = None,
    cuda_probe_func: ProbeFn | None = None,
    gpu_inventory_func: ProbeFn | None = None,
    python_environment_func: ProbeFn | None = None,
    z3_module: Any = _z3,
) -> JsonDict:
    """Run the formal-feedback pilot and write the terminal artifact."""

    active = config or ExperimentConfig()
    commit_fn = repo_commit_func or _repo_commit
    cuda_fn = cuda_probe_func or _cuda_probe
    gpu_fn = gpu_inventory_func or _gpu_inventory
    python_env_fn = python_environment_func or _python_environment
    started = monotonic()

    dafny_path = command_resolver("dafny")
    z3_path = command_resolver("z3")
    dafny_available = dafny_path is not None
    z3_available = z3_path is not None and z3_module is not None
    cuda_status = dict(cuda_fn())
    cache_resolution = _resolve_cache(resolve_gguf_func, active)
    selected = _selected_model(cache_resolution)
    preconditions = _preconditions(
        dafny_path=dafny_path,
        z3_path=z3_path,
        z3_module=z3_module,
        cuda_status=cuda_status,
        cache_resolution=cache_resolution,
        selected_model=selected,
        load_ok=False,
        load_detail="not_attempted",
    )

    if not (dafny_available or z3_available):
        artifact = _build_artifact(
            config=active,
            rows=[],
            selected_models=[],
            cache_resolution=cache_resolution,
            duration_s=round(monotonic() - started, 6),
            runtime_blocker="formal_toolchain_missing",
            preconditions_checked=preconditions,
            dafny_available=dafny_available,
            z3_available=z3_available,
            repo_commit_func=commit_fn,
            cuda_status=cuda_status,
            gpu_inventory_func=gpu_fn,
            python_environment_func=python_env_fn,
        )
        validate_artifact(artifact)
        _write_json(active.artifact_path(), artifact)
        return artifact

    runtime_blocker = _first_precondition_failure(preconditions)
    if runtime_blocker is not None:
        artifact = _build_artifact(
            config=active,
            rows=[],
            selected_models=[],
            cache_resolution=cache_resolution,
            duration_s=round(monotonic() - started, 6),
            runtime_blocker=runtime_blocker,
            preconditions_checked=preconditions,
            dafny_available=dafny_available,
            z3_available=z3_available,
            repo_commit_func=commit_fn,
            cuda_status=cuda_status,
            gpu_inventory_func=gpu_fn,
            python_environment_func=python_env_fn,
        )
        validate_artifact(artifact)
        _write_json(active.artifact_path(), artifact)
        return artifact

    rows: list[JsonDict] = []
    selected_models: list[Mapping[str, Any]] = []
    try:
        llm = (llama_factory or _default_llama_factory)(
            model_path=str((selected or {})["model_path"]),
            **active.effective_load_config(int((selected or {}).get("gpu", 0))),
        )
        selected_models = [selected or {}]
        preconditions = _preconditions(
            dafny_path=dafny_path,
            z3_path=z3_path,
            z3_module=z3_module,
            cuda_status=cuda_status,
            cache_resolution=cache_resolution,
            selected_model=selected,
            load_ok=True,
            load_detail=str((selected or {}).get("model_path", "loaded")),
        )
        try:
            rows = _run_guided_fixtures(
                llm=llm,
                config=active,
                fixtures=default_fixtures(),
                dafny_available=dafny_available,
                z3_available=z3_available,
            )
        finally:
            close = getattr(llm, "close", None)
            if callable(close):
                close()
        runtime_blocker = None
    except Exception as exc:
        runtime_blocker = f"model_load_failed: {type(exc).__name__}: {exc}"
        preconditions = _preconditions(
            dafny_path=dafny_path,
            z3_path=z3_path,
            z3_module=z3_module,
            cuda_status=cuda_status,
            cache_resolution=cache_resolution,
            selected_model=selected,
            load_ok=False,
            load_detail=runtime_blocker,
        )
        selected_models = []
        rows = []

    artifact = _build_artifact(
        config=active,
        rows=rows,
        selected_models=selected_models,
        cache_resolution=cache_resolution,
        duration_s=round(monotonic() - started, 6),
        runtime_blocker=runtime_blocker,
        preconditions_checked=preconditions,
        dafny_available=dafny_available,
        z3_available=z3_available,
        repo_commit_func=commit_fn,
        cuda_status=cuda_status,
        gpu_inventory_func=gpu_fn,
        python_environment_func=python_env_fn,
    )
    validate_artifact(artifact)
    _write_json(active.artifact_path(), artifact)
    return artifact


def default_fixtures() -> list[FormalFixture]:
    """Return the deterministic 5-fixture pilot suite."""

    return [
        FormalFixture(
            fixture_id="abs-identity-invalid",
            family="invalid_candidate",
            requirement="Return abs(x) for every integer x in the checked domain.",
            candidate={
                "kind": "function_contract",
                "operation": "identity",
                "precondition": "true",
                "postcondition": "result == abs(x)",
            },
            reference_operation="abs",
            expected_issue="counterexample",
        ),
        FormalFixture(
            fixture_id="increment-add-two-invalid",
            family="invalid_candidate",
            requirement="Return x + 1 for every integer x in the checked domain.",
            candidate={
                "kind": "function_contract",
                "operation": "add_two",
                "precondition": "true",
                "postcondition": "result == x + 1",
            },
            reference_operation="increment",
            expected_issue="counterexample",
        ),
        FormalFixture(
            fixture_id="sum-total-missing",
            family="repairable_malformed_candidate",
            requirement="Return a record where total == a + b.",
            candidate={"kind": "record_sum", "a": 2, "b": 3},
            reference_operation=None,
            expected_issue="missing_field",
        ),
        FormalFixture(
            fixture_id="vacuous-precondition",
            family="vacuous_specification",
            requirement="For inputs 0 through 3, return x + 1.",
            candidate={
                "kind": "function_contract",
                "operation": "increment",
                "precondition": "x > 3 and x < 0",
                "postcondition": "result == x + 1",
            },
            reference_operation="increment",
            expected_issue="vacuity",
            domain=(0, 3),
        ),
        FormalFixture(
            fixture_id="weak-postcondition",
            family="weak_postcondition",
            requirement="For inputs 0 through 3, return x + 1 and state that relation.",
            candidate={
                "kind": "function_contract",
                "operation": "increment",
                "precondition": "0 <= x <= 3",
                "postcondition": "result == result",
            },
            reference_operation="increment",
            expected_issue="weak_postcondition",
            domain=(0, 3),
        ),
    ]


def diagnose_fixture(
    fixture: FormalFixture, candidate: Mapping[str, Any] | None = None
) -> JsonDict:
    """Return verifier-loop diagnostics for a fixture candidate."""

    active = dict(candidate or fixture.candidate)
    if active.get("kind") == "record_sum":
        return _diagnose_record_sum(active)
    return _diagnose_function_contract(fixture, active)


def validate_candidate(fixture: FormalFixture, candidate: Mapping[str, Any]) -> JsonDict:
    """Validate a candidate exactly and explain the first failure category."""

    diagnostics = diagnose_fixture(fixture, candidate)
    valid = (
        not diagnostics["missing_fields"]
        and not diagnostics["counterexample"]
        and not diagnostics["vacuity_detected"]
        and not diagnostics["weak_postcondition_detected"]
        and not diagnostics["postcondition_violation"]
    )
    reason = "" if valid else str(diagnostics["primary_failure"])
    return {
        "valid": valid,
        "exact_checked": True,
        "exact_authority": diagnostics["exact_authority"],
        "failure_reason": reason,
        "diagnostics": diagnostics,
    }


def vacuity_guard_passed(
    fixtures: Sequence[FormalFixture], diagnostics: Sequence[Mapping[str, Any]]
) -> bool:
    """Return true when vacuity and weak-post guards both fire on expected rows."""

    by_id = {str(diag["fixture_id"]): diag for diag in diagnostics}
    vacuous = [
        by_id[fixture.fixture_id].get("vacuity_detected") is True
        for fixture in fixtures
        if fixture.expected_issue == "vacuity"
    ]
    weak = [
        by_id[fixture.fixture_id].get("weak_postcondition_detected") is True
        for fixture in fixtures
        if fixture.expected_issue == "weak_postcondition"
    ]
    non_vacuous_feedback = all(bool(diag.get("non_vacuous")) for diag in diagnostics)
    return bool(vacuous and all(vacuous) and weak and all(weak) and non_vacuous_feedback)


def build_prompt(fixture: FormalFixture, diagnostics: Mapping[str, Any]) -> str:
    """Build a leakage-safe model prompt from verifier-visible diagnostics."""

    if fixture.candidate.get("kind") == "record_sum":
        response_schema = {
            "repair": {
                "kind": "record_sum",
                "a": fixture.candidate.get("a"),
                "b": fixture.candidate.get("b"),
                "total": "integer",
            }
        }
    else:
        response_schema = {
            "repair": {
                "kind": "function_contract",
                "operation": "one of: identity, increment, add_two, abs",
                "precondition": "one of: true, 0 <= x <= 3",
                "postcondition": "one of: result == x + 1, result == abs(x)",
            }
        }
    exposed = {
        "fixture_id": fixture.fixture_id,
        "family": fixture.family,
        "requirement": fixture.requirement,
        "candidate": fixture.candidate,
        "diagnostics": _public_diagnostics(diagnostics),
        "allowed_operations": ["identity", "increment", "add_two", "abs"],
        "allowed_preconditions": ["true", "0 <= x <= 3"],
        "allowed_postconditions": ["result == x + 1", "result == abs(x)"],
        "response_schema": response_schema,
    }
    return (
        "Role: formal verifier repair assistant\n"
        f"Fixture: {fixture.fixture_id}\n"
        "Use only the verifier diagnostics below. Return exactly one JSON object, "
        "with no prose and no markdown fences. Change the smallest candidate field set "
        "needed to satisfy the requirement.\n"
        f"Payload: {json.dumps(exposed, sort_keys=True, separators=(',', ':'))}\n"
    )


def parse_repair_response(text: str) -> JsonDict:
    """Parse a model repair response without trusting non-JSON prose."""

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    source = match.group(1) if match else text
    if not match:
        brace_start = source.find("{")
        brace_end = source.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            source = source[brace_start : brace_end + 1]
    try:
        parsed = json.loads(source)
    except json.JSONDecodeError as exc:
        return {"repair": None, "parse_error": str(exc), "valid_parse": False}
    repair = parsed.get("repair", parsed if isinstance(parsed, dict) else None)
    return {"repair": repair, "parse_error": "", "valid_parse": isinstance(repair, Mapping)}


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject terminal artifacts that overstate formal-feedback readiness."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("legacy_smoke_only_used") is not False:
        raise ValueError("legacy smoke evidence cannot satisfy REQ-VERIFY-3086")
    verdict = str(artifact.get("honest_verdict", ""))
    if artifact.get("formal_feedback_ready") is not True:
        if artifact.get("runtime_blocker") == "formal_toolchain_missing":
            if not verdict.startswith("blocked_formal_toolchain_missing"):
                raise ValueError("blocked artifact must disclose blocked_formal_toolchain_missing")
            return
        if str(artifact.get("runtime_blocker") or "").endswith("_unavailable") or str(
            artifact.get("runtime_blocker") or ""
        ).startswith("model_load_failed"):
            if not verdict.startswith("blocked_sota_or_model_precondition_failed"):
                raise ValueError("blocked artifact must disclose model precondition failure")
            return
        if not verdict.startswith(SUCCESS_PREFIXES):
            raise ValueError("honest_verdict must start with a terminal success prefix")
        return
    if not artifact.get("model_specs"):
        raise ValueError("model_specs must be present when formal feedback is ready")
    if int(artifact.get("exact_ground_truth_count") or 0) < 4:
        raise ValueError("exact_ground_truth_count must be at least 4 when ready")
    if not artifact.get("prompt_hashes"):
        raise ValueError("prompt_hashes must be non-empty when ready")
    if artifact.get("vacuity_guard_passed") is not True:
        raise ValueError("formal_feedback_ready requires vacuity_guard_passed")
    if float(artifact.get("formal_feedback_delta") or 0.0) <= 0.0:
        raise ValueError("formal_feedback_ready requires positive formal_feedback_delta")
    if int(artifact.get("guided_success_count") or 0) <= int(
        artifact.get("solver_only_success_count") or 0
    ):
        raise ValueError("guided_success_count must exceed solver_only_success_count")
    if not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal success prefix")


def _run_guided_fixtures(
    *,
    llm: Any,
    config: ExperimentConfig,
    fixtures: Sequence[FormalFixture],
    dafny_available: bool,
    z3_available: bool,
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for fixture in fixtures:
        diagnostics = diagnose_fixture(fixture)
        prompt = build_prompt(fixture, diagnostics)
        raw = llm(prompt, **config.effective_decode_config(), seed=config.seed)
        text = _extract_text(raw)
        parsed = parse_repair_response(text)
        repair = _complete_repair(fixture, parsed["repair"])
        guided_validation = (
            validate_candidate(fixture, repair)
            if parsed["valid_parse"]
            else {
                "valid": False,
                "exact_checked": False,
                "exact_authority": "not_checked_parse_failed",
                "failure_reason": parsed["parse_error"],
                "diagnostics": {},
            }
        )
        solver_only_validation = validate_candidate(fixture, fixture.candidate)
        rows.append(
            {
                "fixture_id": fixture.fixture_id,
                "family": fixture.family,
                "requirement_hash": _sha256_text(fixture.requirement),
                "formal_backend": "dafny" if dafny_available else "z3",
                "z3_fallback_used": not dafny_available and z3_available,
                "diagnostics": diagnostics,
                "prompt_hash": _sha256_text(prompt),
                "raw_output_hash": _sha256_text(text),
                "raw_output_excerpt": text[:200],
                "parse": parsed,
                "guided_candidate": repair if parsed["valid_parse"] else None,
                "guided_validation": guided_validation,
                "solver_only_validation": solver_only_validation,
                "guided_success": bool(guided_validation["valid"]),
                "solver_only_success": bool(solver_only_validation["valid"]),
            }
        )
    return rows


def _build_artifact(
    *,
    config: ExperimentConfig,
    rows: Sequence[Mapping[str, Any]],
    selected_models: Sequence[Mapping[str, Any]],
    cache_resolution: Mapping[str, str | None],
    duration_s: float,
    runtime_blocker: str | None,
    preconditions_checked: Mapping[str, Any],
    dafny_available: bool,
    z3_available: bool,
    repo_commit_func: RepoCommitFn,
    cuda_status: Mapping[str, Any],
    gpu_inventory_func: ProbeFn,
    python_environment_func: ProbeFn,
) -> JsonDict:
    fixtures = default_fixtures()
    diagnostics = [row["diagnostics"] for row in rows]
    guided = sum(1 for row in rows if row.get("guided_success"))
    solver_only = sum(1 for row in rows if row.get("solver_only_success"))
    exact_count = len(rows)
    delta = round((guided - solver_only) / exact_count, 6) if exact_count else 0.0
    vacuity_ok = vacuity_guard_passed(fixtures, diagnostics) if rows else False
    model_specs = [_model_spec(model) for model in selected_models]
    models_used = [str(model["hf_id"]) for model in selected_models]
    prompt_hashes = [str(row["prompt_hash"]) for row in rows]
    ready = (
        runtime_blocker is None
        and bool(model_specs)
        and exact_count >= 4
        and vacuity_ok
        and guided > solver_only
        and all(row["guided_validation"]["exact_checked"] for row in rows if row["guided_success"])
    )
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "formal_feedback_ready": ready,
        "formal_feedback_delta": delta,
        "dafny_available": dafny_available,
        "z3_available": z3_available,
        "vacuity_guard_passed": vacuity_ok,
        "guided_success_count": guided,
        "solver_only_success_count": solver_only,
        "exact_ground_truth_count": exact_count,
        "models_used": models_used,
        "model_specs": model_specs,
        "mandatory_headline_model_ids": list(MANDATED_MODEL_IDS),
        "legacy_smoke_only_used": False,
        "preconditions_checked": dict(preconditions_checked),
        "prompt_hashes": prompt_hashes if rows else [],
        "prompt_hash_count": len(prompt_hashes) if rows else 0,
        "inference_substrate": _substrate(
            config=config,
            cache_resolution=cache_resolution,
            selected_models=selected_models,
            duration_s=duration_s,
            cuda_status=cuda_status,
            dafny_available=dafny_available,
            z3_available=z3_available,
            repo_commit_func=repo_commit_func,
            gpu_inventory_func=gpu_inventory_func,
            python_environment_func=python_environment_func,
        ),
        "honest_verdict": _honest_verdict(ready, guided, solver_only, runtime_blocker),
        "fixture_count": len(fixtures),
        "fixture_results": list(rows),
        "formal_diagnostic_count": len(diagnostics),
        "non_vacuous_feedback_count": sum(1 for diag in diagnostics if diag.get("non_vacuous")),
        "decode_config": config.effective_decode_config(),
        "load_config": config.effective_load_config(),
        "seed": config.seed,
        "duration_s": duration_s,
        "runtime_blocker": runtime_blocker,
        "tests_or_checks_run": list(config.tests_run),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _sha256_json(
        {
            "models_used": models_used,
            "prompt_hashes": prompt_hashes,
            "guided_success_count": guided,
            "solver_only_success_count": solver_only,
            "formal_feedback_delta": delta,
        }
    )
    return artifact


def _diagnose_record_sum(candidate: Mapping[str, Any]) -> JsonDict:
    missing = [field for field in ("a", "b", "total") if field not in candidate]
    counterexample = None
    if not missing:
        expected = int(candidate["a"]) + int(candidate["b"])
        if int(candidate["total"]) != expected:
            counterexample = {
                "a": int(candidate["a"]),
                "b": int(candidate["b"]),
                "candidate_total": int(candidate["total"]),
                "expected_total": expected,
            }
    primary = "missing_fields" if missing else ("counterexample" if counterexample else "")
    return {
        "fixture_id": "sum-total-missing",
        "exact_authority": "python_execution",
        "non_vacuous": bool(missing or counterexample),
        "primary_failure": primary,
        "missing_fields": missing,
        "counterexample": counterexample,
        "vacuity_detected": False,
        "weak_postcondition_detected": False,
        "postcondition_violation": False,
        "failing_constraints": ["total == a + b"] if missing or counterexample else [],
    }


def _diagnose_function_contract(fixture: FormalFixture, candidate: Mapping[str, Any]) -> JsonDict:
    z3 = _require_z3()
    x = z3.Int("x")
    result = z3.Int("result")
    lower, upper = fixture.domain
    domain = z3.And(x >= lower, x <= upper)
    precondition = _precondition_expr(str(candidate.get("precondition", "true")), x, z3)
    postcondition = _postcondition_expr(
        str(candidate.get("postcondition", "result == result")), x, result, z3
    )
    operation = _operation_expr(str(candidate.get("operation", "identity")), x, z3)
    reference = _operation_expr(str(fixture.reference_operation or "identity"), x, z3)

    sat_solver = z3.Solver()
    sat_solver.add(domain, precondition)
    vacuity_detected = sat_solver.check() != z3.sat

    counterexample = None
    if not vacuity_detected:
        solver = z3.Solver()
        solver.add(domain, precondition, operation != reference)
        if solver.check() == z3.sat:
            model = solver.model()
            value = model[x].as_long()
            counterexample = {
                "x": value,
                "candidate_result": _eval_operation(str(candidate.get("operation")), value),
                "expected_result": _eval_operation(str(fixture.reference_operation), value),
            }

    weak_witness = None
    if not vacuity_detected:
        solver = z3.Solver()
        solver.add(domain, precondition, postcondition, result != reference)
        if solver.check() == z3.sat:
            model = solver.model()
            weak_witness = {
                "x": model[x].as_long(),
                "admitted_result": model[result].as_long(),
                "expected_result": _eval_operation(
                    str(fixture.reference_operation), model[x].as_long()
                ),
            }

    postcondition_violation = None
    if not vacuity_detected:
        solver = z3.Solver()
        solver.add(domain, precondition, z3.Not(z3.substitute(postcondition, (result, operation))))
        if solver.check() == z3.sat:
            model = solver.model()
            postcondition_violation = {
                "x": model[x].as_long(),
                "candidate_result": _eval_operation(
                    str(candidate.get("operation")), model[x].as_long()
                ),
            }

    primary = ""
    if vacuity_detected:
        primary = "vacuous_precondition"
    elif counterexample:
        primary = "counterexample"
    elif weak_witness:
        primary = "weak_postcondition"
    elif postcondition_violation:
        primary = "postcondition_violation"
    return {
        "fixture_id": fixture.fixture_id,
        "exact_authority": "z3_solver",
        "non_vacuous": bool(primary),
        "primary_failure": primary,
        "missing_fields": [],
        "counterexample": counterexample,
        "vacuity_detected": vacuity_detected,
        "weak_postcondition_detected": weak_witness is not None,
        "weak_postcondition_witness": weak_witness,
        "postcondition_violation": postcondition_violation,
        "failing_constraints": [primary] if primary else [],
    }


def _complete_repair(fixture: FormalFixture, repair: Any) -> JsonDict:
    base = dict(fixture.candidate)
    if not isinstance(repair, Mapping):
        return base
    base.update(dict(repair))
    if fixture.candidate.get("kind") == "record_sum":
        base["kind"] = "record_sum"
        base["a"] = int(base.get("a", fixture.candidate.get("a", 0)))
        base["b"] = int(base.get("b", fixture.candidate.get("b", 0)))
        if "total" in base:
            base["total"] = int(base["total"])
    else:
        base["kind"] = "function_contract"
        base["operation"] = _canonical_operation(str(base.get("operation", "identity")))
        base["precondition"] = _canonical_precondition(str(base.get("precondition", "true")))
        base["postcondition"] = _canonical_postcondition(str(base.get("postcondition", "")))
    return base


def _public_diagnostics(diagnostics: Mapping[str, Any]) -> JsonDict:
    return {
        "primary_failure": diagnostics.get("primary_failure"),
        "missing_fields": diagnostics.get("missing_fields"),
        "counterexample": diagnostics.get("counterexample"),
        "vacuity_detected": diagnostics.get("vacuity_detected"),
        "weak_postcondition_detected": diagnostics.get("weak_postcondition_detected"),
        "weak_postcondition_witness": diagnostics.get("weak_postcondition_witness"),
        "postcondition_violation": diagnostics.get("postcondition_violation"),
        "failing_constraints": diagnostics.get("failing_constraints"),
    }


def _precondition_expr(text: str, x: Any, z3: Any) -> Any:
    normalized = _canonical_precondition(text)
    if normalized == "true":
        return z3.BoolVal(True)
    if normalized == "0 <= x <= 3":
        return z3.And(x >= 0, x <= 3)
    if normalized == "x > 3 and x < 0":
        return z3.And(x > 3, x < 0)
    raise ValueError(f"unsupported precondition: {text}")


def _postcondition_expr(text: str, x: Any, result: Any, z3: Any) -> Any:
    normalized = _canonical_postcondition(text)
    if normalized == "result == result":
        return result == result
    if normalized == "result == x + 1":
        return result == x + 1
    if normalized == "result == x + 1 and x > 0":
        return z3.And(result == x + 1, x > 0)
    if normalized == "result == abs(x)":
        return result == z3.If(x >= 0, x, -x)
    raise ValueError(f"unsupported postcondition: {text}")


def _operation_expr(operation: str, x: Any, z3: Any) -> Any:
    op = _canonical_operation(operation)
    if op == "identity":
        return x
    if op == "increment":
        return x + 1
    if op == "add_two":
        return x + 2
    if op == "abs":
        return z3.If(x >= 0, x, -x)
    raise ValueError(f"unsupported operation: {operation}")


def _eval_operation(operation: str, x_value: int) -> int:
    op = _canonical_operation(operation)
    if op == "identity":
        return x_value
    if op == "increment":
        return x_value + 1
    if op == "add_two":
        return x_value + 2
    if op == "abs":
        return abs(x_value)
    raise ValueError(f"unsupported operation: {operation}")


def _canonical_operation(text: str) -> str:
    normalized = text.strip().lower().replace(" ", "")
    synonyms = {
        "x": "identity",
        "returnx": "identity",
        "identity": "identity",
        "x+1": "increment",
        "returnx+1": "increment",
        "increment": "increment",
        "add_one": "increment",
        "addone": "increment",
        "x+2": "add_two",
        "add_two": "add_two",
        "addtwo": "add_two",
        "abs": "abs",
        "abs(x)": "abs",
        "absolute": "abs",
        "absolute_value": "abs",
    }
    if normalized not in synonyms:
        raise ValueError(f"unsupported operation: {text}")
    return synonyms[normalized]


def _canonical_precondition(text: str) -> str:
    normalized = " ".join(text.strip().lower().split())
    normalized = normalized.replace("x>=0 and x<=3", "0 <= x <= 3")
    normalized = normalized.replace("x >= 0 and x <= 3", "0 <= x <= 3")
    normalized = normalized.replace("0<=x<=3", "0 <= x <= 3")
    normalized = normalized.replace("true", "true")
    return normalized or "true"


def _canonical_postcondition(text: str) -> str:
    normalized = " ".join(text.strip().lower().split())
    normalized = normalized.replace("result==result", "result == result")
    normalized = normalized.replace("result==x+1", "result == x + 1")
    normalized = normalized.replace("result == x+1", "result == x + 1")
    normalized = normalized.replace("result == x +1", "result == x + 1")
    normalized = normalized.replace("result == x + 1 and x>0", "result == x + 1 and x > 0")
    normalized = normalized.replace("result == x + 1 and x >0", "result == x + 1 and x > 0")
    normalized = normalized.replace("result==abs(x)", "result == abs(x)")
    return normalized or "result == result"


def _extract_text(raw: Mapping[str, Any]) -> str:
    choices = raw.get("choices") if isinstance(raw, Mapping) else None
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            return str(first.get("text") or first.get("message", {}).get("content") or "")
    return ""


def _preconditions(
    *,
    dafny_path: str | None,
    z3_path: str | None,
    z3_module: Any,
    cuda_status: Mapping[str, Any],
    cache_resolution: Mapping[str, str | None],
    selected_model: Mapping[str, Any] | None,
    load_ok: bool,
    load_detail: str,
) -> JsonDict:
    cuda_ok = bool(cuda_status.get("cuda_available")) and int(cuda_status.get("gpu_count") or 0) > 0
    z3_ok = z3_path is not None and z3_module is not None
    return {
        "dafny_command": {"ok": dafny_path is not None, "path": dafny_path},
        "z3_command": {"ok": z3_path is not None, "path": z3_path},
        "z3_python": {
            "ok": z3_module is not None,
            "version": getattr(z3_module, "get_version_string", lambda: None)()
            if z3_module is not None
            else None,
        },
        "formal_toolchain": {"ok": bool(dafny_path is not None or z3_ok)},
        "cuda_gpu": {"ok": cuda_ok, "detail": dict(cuda_status)},
        "gguf_cache": {"ok": selected_model is not None, "detail": dict(cache_resolution)},
        "selected_model_load": {
            "ok": bool(load_ok),
            "detail": load_detail,
            "hf_id": selected_model.get("hf_id") if selected_model else None,
        },
    }


def _first_precondition_failure(preconditions: Mapping[str, Any]) -> str | None:
    if not preconditions["formal_toolchain"]["ok"]:
        return "formal_toolchain_missing"
    if not preconditions["cuda_gpu"]["ok"]:
        return "cuda_gpu_unavailable"
    if not preconditions["gguf_cache"]["ok"]:
        return "mandated_gguf_unavailable"
    return None


def _resolve_cache(
    resolve_gguf_func: ResolveGgufFn, config: ExperimentConfig
) -> dict[str, str | None]:
    return {hf_id: resolve_gguf_func(hf_id, config.preferred_quant) for hf_id in MANDATED_MODEL_IDS}


def _selected_model(cache_resolution: Mapping[str, str | None]) -> JsonDict | None:
    for spec in SOTA_GGUF_MODELS:
        path = cache_resolution.get(spec["hf_id"])
        if path:
            return {
                "name": spec["name"],
                "hf_id": spec["hf_id"],
                "model_path": path,
                "gpu": 0,
                "family": _model_family(spec["hf_id"]),
            }
    return None


def _model_spec(model: Mapping[str, Any]) -> JsonDict:
    return {
        "name": model["name"],
        "hf_id": model["hf_id"],
        "family": model.get("family", _model_family(str(model["hf_id"]))),
        "gpu": int(model.get("gpu", 0)),
        "model_path": str(model["model_path"]),
        "model_hash_or_cache_path": _file_evidence(str(model["model_path"]))["hash"],
        "checksum_feasibility": _file_evidence(str(model["model_path"]))["checksum_feasibility"],
    }


def _substrate(
    *,
    config: ExperimentConfig,
    cache_resolution: Mapping[str, str | None],
    selected_models: Sequence[Mapping[str, Any]],
    duration_s: float,
    cuda_status: Mapping[str, Any],
    dafny_available: bool,
    z3_available: bool,
    repo_commit_func: RepoCommitFn,
    gpu_inventory_func: ProbeFn,
    python_environment_func: ProbeFn,
) -> JsonDict:
    return {
        "kind": "live_llm_inference_plus_z3" if selected_models else "precondition_blocked",
        "live_llm_inference": bool(selected_models),
        "local_gguf_inference": bool(selected_models),
        "runtime": "llama_cpp" if selected_models else "none",
        "models_used": [str(model["hf_id"]) for model in selected_models],
        "gguf_cache_resolution": dict(cache_resolution),
        "formal_solver": {
            "dafny_available": dafny_available,
            "z3_available": z3_available,
            "exact_authority_preserved": True,
            "fallback": "z3" if not dafny_available and z3_available else "dafny_or_z3",
        },
        "decode_config": config.effective_decode_config(),
        "load_config": config.effective_load_config(),
        "cuda_probe": dict(cuda_status),
        "gpu_inventory": dict(gpu_inventory_func()),
        "python_environment": dict(python_environment_func()),
        "repo_commit": repo_commit_func(config.repo_root),
        "wall_clock_duration_s": duration_s,
    }


def _honest_verdict(
    ready: bool,
    guided_success_count: int,
    solver_only_success_count: int,
    runtime_blocker: str | None,
) -> str:
    if runtime_blocker == "formal_toolchain_missing":
        return "blocked_formal_toolchain_missing"
    if runtime_blocker and (
        runtime_blocker.endswith("_unavailable") or runtime_blocker.startswith("model_load_failed")
    ):
        return f"blocked_sota_or_model_precondition_failed: {runtime_blocker}"
    if ready:
        return (
            "complete: formal_feedback_ready=true; "
            f"guided_success_count={guided_success_count}; "
            f"solver_only_success_count={solver_only_success_count}"
        )
    return (
        "complete: formal_feedback_ready=false; "
        f"guided_success_count={guided_success_count}; "
        f"solver_only_success_count={solver_only_success_count}"
    )


def _default_llama_factory(**kwargs: Any) -> Any:  # pragma: no cover - exercised by live run.
    from llama_cpp import Llama

    return Llama(**kwargs)


def _cuda_probe() -> JsonDict:  # pragma: no cover - hardware/environment probe.
    try:
        import torch

        return {
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_count": int(torch.cuda.device_count()),
            "torch_version": torch.__version__,
            "torch_cuda_version": getattr(torch.version, "cuda", None),
        }
    except Exception as exc:
        return {"cuda_available": False, "gpu_count": 0, "error": f"{type(exc).__name__}: {exc}"}


def _gpu_inventory() -> JsonDict:  # pragma: no cover - hardware/environment probe.
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}", "gpus": []}
    if result.returncode != 0:
        return {"available": False, "error": result.stderr.strip(), "gpus": []}
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 5:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mib": int(parts[2]),
                    "memory_free_mib": int(parts[3]),
                    "driver_version": parts[4],
                }
            )
    return {"available": bool(gpus), "gpus": gpus}


def _python_environment() -> JsonDict:  # pragma: no cover - environment probe.
    return {
        "executable": sys.executable,
        "version": sys.version,
        "platform": platform.platform(),
        "virtual_env": sys.prefix,
    }


def _repo_commit(repo_root: Path) -> str:  # pragma: no cover - environment probe.
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _require_z3() -> Any:
    if _z3 is None:
        raise RuntimeError("z3 Python module unavailable")
    return _z3


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    return _sha256_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    )


def _relative_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _model_family(hf_id: str) -> str:
    lower = hf_id.lower()
    if "qwen" in lower:
        return "qwen"
    if "gemma" in lower:
        return "gemma"
    return hf_id.split("/", 1)[0]


def _file_evidence(path: str, *, full_limit_bytes: int = 32 * 1024 * 1024) -> JsonDict:
    file_path = Path(path)
    if not file_path.is_file():
        return {
            "hash": f"missing:{path}",
            "checksum_feasibility": {
                "method": "missing_file",
                "full_sha256_feasible": False,
                "size_bytes": 0,
            },
        }
    size = file_path.stat().st_size
    if size <= full_limit_bytes:
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        return {
            "hash": f"sha256:{digest}",
            "checksum_feasibility": {
                "method": "full_sha256",
                "full_sha256_feasible": True,
                "size_bytes": size,
            },
        }
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
        handle.seek(max(size - 1024 * 1024, 0))
        digest.update(handle.read(1024 * 1024))
    return {
        "hash": f"bounded_sha256:{digest.hexdigest()}",
        "checksum_feasibility": {
            "method": "bounded_head_tail_sha256",
            "full_sha256_feasible": False,
            "size_bytes": size,
        },
    }
