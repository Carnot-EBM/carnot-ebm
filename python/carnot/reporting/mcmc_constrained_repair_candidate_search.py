"""Exp 1429 MCMC-style constrained repair candidate search.

Spec: REQ-VERIFY-1429, SCENARIO-VERIFY-1429
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.pipeline.certificate_repair_executor import CertificateRepairRequest
from carnot.pipeline.mcmc_constrained_repair_search import (
    BoundedConstrainedRepairCandidateSearch,
    CandidateSearchConfig,
)
from carnot.reporting import certificate_llm_repair_executor_v1 as exp1414


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260506"
EXPERIMENT = "1429_mcmc_constrained_repair_candidate_search"
SCHEMA = "mcmc_constrained_repair_candidate_search"
DEFAULT_EXP1397_PATH = REPO_ROOT / "results" / "experiment_1397_fullscale_pipeline_v2_200cases.json"
DEFAULT_EXP1428_PATH = (
    REPO_ROOT / "results" / "experiment_1428_dccd_schema_constrained_repair_v2.json"
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "results" / "experiment_1429_mcmc_constrained_repair_candidate_search.json"
)
MIN_REPAIR_HINT_CASES = 20
DEFAULT_CANDIDATES_PER_CASE = 4

MODEL_SPECS: list[dict[str, str]] = [
    {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "primary_candidate_generator",
    },
    {
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "dense_fallback",
    },
    {
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "moe_fallback",
    },
]

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "model_specs",
    "candidate_search_complete",
    "cases_evaluated",
    "candidates_per_case",
    "mcmc_acceptance_rate",
    "repair_success_rate_one_candidate",
    "repair_success_rate_best_of_n",
    "energy_rerank_improved",
    "local_sota_model_used",
    "honest_verdict",
)

CachedPairFn = Callable[..., list[dict[str, Any]] | None]
GeneratorFactory = Callable[[dict[str, Any]], Callable[[str], str]]
ValidatorFn = Callable[[CertificateRepairRequest, Any], Mapping[str, Any]]
WriteObserver = Callable[[Path, dict[str, Any]], None]


def write_in_progress_artifact(
    path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    candidates_per_case: int = DEFAULT_CANDIDATES_PER_CASE,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """REQ-VERIFY-1429: persist bootstrap JSON before loading experiment inputs."""

    artifact = _base_artifact(
        project_root=project_root,
        run_date=run_date,
        status="in_progress",
        candidates_per_case=candidates_per_case,
    )
    artifact["honest_verdict"] = "in_progress"
    _write_json(Path(path), artifact, write_observer=write_observer)
    return artifact


def run_experiment(
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    exp1397_path: str | Path = DEFAULT_EXP1397_PATH,
    exp1428_path: str | Path = DEFAULT_EXP1428_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    cached_pair_fn: CachedPairFn | None = None,
    generator_factory: GeneratorFactory | None = None,
    validator: ValidatorFn | None = None,
    candidates_per_case: int = DEFAULT_CANDIDATES_PER_CASE,
    executor_runtime_mode: str = "live_local_sota_gguf",
    tests_run: Sequence[str] | None = None,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """Run Exp 1429 or write an honest blocked artifact for gates and blockers."""

    root = Path(project_root)
    output = _resolve(root, output_path)
    write_in_progress_artifact(
        output,
        project_root=root,
        run_date=run_date,
        candidates_per_case=candidates_per_case,
        write_observer=write_observer,
    )

    exp1428 = _read_json(_resolve(root, exp1428_path))
    if exp1428.get("repair_executor_v2_deployed") is not True:
        artifact = _blocked_artifact(
            project_root=root,
            run_date=run_date,
            candidates_per_case=candidates_per_case,
            tests_run=tests_run,
            honest_verdict="blocked_repair_v2_not_deployed",
            repair_v2_deployment_diagnostics=exp1428,
        )
        _write_json(output, artifact, write_observer=write_observer)
        return artifact

    resolved_specs, cache_diagnostics = resolve_model_specs(cached_pair_fn)
    selected_spec = select_candidate_model(resolved_specs)
    if selected_spec is None:
        artifact = _blocked_artifact(
            project_root=root,
            run_date=run_date,
            candidates_per_case=candidates_per_case,
            tests_run=tests_run,
            honest_verdict="blocked_sota_model_cache_unavailable",
            model_specs=resolved_specs,
            cache_diagnostics=cache_diagnostics,
        )
        _write_json(output, artifact, write_observer=write_observer)
        return artifact

    try:
        generator = (generator_factory or _default_generator_factory)(selected_spec)
    except Exception as exc:  # pragma: no cover - depends on local llama.cpp installation.
        diagnostics = dict(cache_diagnostics)
        diagnostics["generator_error"] = f"{type(exc).__name__}: {exc}"
        artifact = _blocked_artifact(
            project_root=root,
            run_date=run_date,
            candidates_per_case=candidates_per_case,
            tests_run=tests_run,
            honest_verdict="blocked_local_model_runtime_unavailable",
            model_specs=resolved_specs,
            cache_diagnostics=diagnostics,
        )
        _write_json(output, artifact, write_observer=write_observer)
        return artifact

    exp1397 = _read_json(_resolve(root, exp1397_path))
    repair_cases = exp1414.repair_requests_from_exp1397(exp1397)
    sample = repair_cases[:MIN_REPAIR_HINT_CASES]
    active_validator = validator or _default_validator
    search = BoundedConstrainedRepairCandidateSearch(
        generator=generator,
        model_spec=selected_spec,
        validator=active_validator,
        config=CandidateSearchConfig(candidates_per_case=candidates_per_case),
    )
    case_results = [search.search(request) for request in sample]
    cases_evaluated = len(case_results)
    total_candidates = sum(result.candidates_evaluated for result in case_results)
    accepted_candidates = sum(result.accepted_candidate_count for result in case_results)
    one_candidate_successes = sum(1 for result in case_results if result.one_candidate_success)
    best_of_n_successes = sum(1 for result in case_results if result.best_of_n_success)
    energy_rerank_improved = any(result.energy_rerank_improved for result in case_results)

    artifact = _base_artifact(
        project_root=root,
        run_date=run_date,
        status="complete",
        candidates_per_case=candidates_per_case,
    )
    artifact.update(
        {
            "model_specs": resolved_specs,
            "cache_diagnostics": cache_diagnostics,
            "candidate_search_complete": True,
            "repair_v2_deployed_confirmed": True,
            "repair_hint_cases_available": len(repair_cases),
            "cases_evaluated": cases_evaluated,
            "total_candidate_proposals": total_candidates,
            "accepted_candidate_count": accepted_candidates,
            "mcmc_acceptance_rate": _rate(accepted_candidates, total_candidates),
            "repair_success_rate_one_candidate": _rate(one_candidate_successes, cases_evaluated),
            "repair_success_rate_best_of_n": _rate(best_of_n_successes, cases_evaluated),
            "energy_rerank_improved": energy_rerank_improved,
            "local_sota_model_used": selected_spec.get("hf_id"),
            "local_sota_model_inference_used": executor_runtime_mode == "live_local_sota_gguf",
            "executor_runtime_mode": executor_runtime_mode,
            "candidate_search_results": [result.to_dict() for result in case_results],
            "tests_run": list(tests_run or []),
            "honest_verdict": _complete_verdict(
                cases_evaluated=cases_evaluated,
                one_candidate_successes=one_candidate_successes,
                best_of_n_successes=best_of_n_successes,
                executor_runtime_mode=executor_runtime_mode,
            ),
        }
    )
    _write_json(output, artifact, write_observer=write_observer)
    return artifact


def resolve_model_specs(
    cached_pair_fn: CachedPairFn | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve mandated SOTA GGUF specs and preserve Exp 1429 role names."""

    resolved, diagnostics = exp1414.resolve_model_specs(cached_pair_fn)
    resolved_by_hf = {str(spec.get("hf_id")): dict(spec) for spec in resolved}
    role_by_hf = {spec["hf_id"]: spec["role"] for spec in MODEL_SPECS}
    model_specs: list[dict[str, Any]] = []
    for required in MODEL_SPECS:
        spec = dict(required)
        inherited = resolved_by_hf.get(required["hf_id"])
        if inherited is not None:
            spec.update(inherited)
        spec["role"] = role_by_hf[required["hf_id"]]
        model_specs.append(spec)
    return model_specs, dict(diagnostics)


def select_candidate_model(model_specs: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Choose primary, then dense fallback, then MoE fallback from resolved specs."""

    for role in ("primary_candidate_generator", "dense_fallback", "moe_fallback"):
        for spec in model_specs:
            if spec.get("role") == role and spec.get("model_path"):
                return dict(spec)
    return None


def _base_artifact(
    *,
    project_root: str | Path,
    run_date: str,
    status: str,
    candidates_per_case: int,
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": status,
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "spec": ["REQ-VERIFY-1429", "SCENARIO-VERIFY-1429"],
            "source_experiments": ["exp1397", "exp1428"],
        },
        "model_specs": [dict(spec) for spec in MODEL_SPECS],
        "cache_diagnostics": {},
        "candidate_search_complete": False,
        "repair_v2_deployed_confirmed": False,
        "repair_hint_cases_available": 0,
        "cases_evaluated": 0,
        "candidates_per_case": candidates_per_case,
        "total_candidate_proposals": 0,
        "accepted_candidate_count": 0,
        "mcmc_acceptance_rate": 0.0,
        "repair_success_rate_one_candidate": 0.0,
        "repair_success_rate_best_of_n": 0.0,
        "energy_rerank_improved": False,
        "local_sota_model_used": None,
        "local_sota_model_inference_used": False,
        "executor_runtime_mode": "not_run",
        "candidate_search_results": [],
        "tests_run": [],
        "honest_verdict": "not_run",
    }


def _blocked_artifact(
    *,
    project_root: str | Path,
    run_date: str,
    candidates_per_case: int,
    tests_run: Sequence[str] | None,
    honest_verdict: str,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    cache_diagnostics: Mapping[str, Any] | None = None,
    repair_v2_deployment_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = _base_artifact(
        project_root=project_root,
        run_date=run_date,
        status="blocked",
        candidates_per_case=candidates_per_case,
    )
    artifact.update(
        {
            "model_specs": [dict(spec) for spec in (model_specs or MODEL_SPECS)],
            "cache_diagnostics": dict(cache_diagnostics or {}),
            "repair_v2_deployment_diagnostics": dict(repair_v2_deployment_diagnostics or {}),
            "tests_run": list(tests_run or []),
            "honest_verdict": honest_verdict,
        }
    )
    return artifact


def _complete_verdict(
    *,
    cases_evaluated: int,
    one_candidate_successes: int,
    best_of_n_successes: int,
    executor_runtime_mode: str,
) -> str:
    if cases_evaluated == 0:
        verdict = "complete_mcmc_constrained_repair_candidate_search_no_cases"
    elif best_of_n_successes > one_candidate_successes:
        verdict = "complete_mcmc_constrained_repair_candidate_search_improved"
    elif best_of_n_successes > 0:
        verdict = "complete_mcmc_constrained_repair_candidate_search_no_rate_improvement"
    else:
        verdict = "complete_mcmc_constrained_repair_candidate_search_no_successful_repairs"
    if executor_runtime_mode != "live_local_sota_gguf":
        verdict = f"{verdict}_prototype_no_headline_sota_claim"
    return verdict


def _default_validator(
    _request: CertificateRepairRequest,
    _candidate: Any,
) -> Mapping[str, Any]:  # pragma: no cover - live validation is experiment-specific.
    return {
        "constraint_passed": False,
        "semantic_result": "REPAIR_HINT",
        "repair_required": True,
        "false_acceptance": False,
        "fallback_reason": "no_validator_injected",
    }


def _default_generator_factory(
    model_spec: dict[str, Any],
) -> Callable[[str], str]:  # pragma: no cover - requires local llama.cpp runtime.
    return exp1414._default_generator_factory(model_spec)


def _rate(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(
    path: Path,
    artifact: dict[str, Any],
    *,
    write_observer: WriteObserver | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if write_observer is not None:
        write_observer(path, artifact)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(REPO_ROOT))
    parser.add_argument("--run-date", default=RUN_DATE)
    parser.add_argument("--exp1397-path", default=str(DEFAULT_EXP1397_PATH))
    parser.add_argument("--exp1428-path", default=str(DEFAULT_EXP1428_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--candidates-per-case", type=int, default=DEFAULT_CANDIDATES_PER_CASE)
    args = parser.parse_args(argv)
    run_experiment(
        project_root=Path(args.project_root),
        run_date=args.run_date,
        exp1397_path=Path(args.exp1397_path),
        exp1428_path=Path(args.exp1428_path),
        output_path=Path(args.output_path),
        candidates_per_case=args.candidates_per_case,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
