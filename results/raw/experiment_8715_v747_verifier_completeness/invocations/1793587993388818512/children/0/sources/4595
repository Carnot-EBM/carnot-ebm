"""Exp 1428 DCCD schema-constrained certificate repair executor v2.

Spec: REQ-VERIFY-1428, SCENARIO-VERIFY-1428
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.pipeline.certificate_repair_executor import CertificateRepairRequest
from carnot.pipeline.dccd_schema_constrained_repair import (
    DCCD_REJECTION_REASONS,
    DCCD_REPAIR_OUTPUT_SCHEMA,
    DraftConditionedSchemaRepairExecutor,
)
from carnot.reporting import certificate_llm_repair_executor_v1 as exp1414


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260506"
EXPERIMENT = "1428_dccd_schema_constrained_repair_v2"
SCHEMA = "dccd_schema_constrained_repair_v2"
DEFAULT_EXP1397_PATH = (
    REPO_ROOT / "results" / "experiment_1397_fullscale_pipeline_v2_200cases.json"
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "results" / "experiment_1428_dccd_schema_constrained_repair_v2.json"
)
MIN_REPAIR_HINT_CASES = 20

MODEL_SPECS = [dict(spec) for spec in exp1414.MODEL_SPECS]

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "model_specs",
    "local_sota_model_used",
    "repair_executor_v2_deployed",
    "repair_hint_cases_tested",
    "repaired_cases_successful",
    "repaired_case_success_rate",
    "schema_valid_rate",
    "semantic_acceptance_rate",
    "rejection_reason_counts",
    "tests_run",
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
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """REQ-VERIFY-1428: persist bootstrap JSON before source or model loading."""

    artifact = _base_artifact(project_root=project_root, run_date=run_date, status="in_progress")
    artifact["honest_verdict"] = "in_progress"
    _write_json(Path(path), artifact, write_observer=write_observer)
    return artifact


def run_experiment(
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    exp1397_path: str | Path = DEFAULT_EXP1397_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    cached_pair_fn: CachedPairFn | None = None,
    generator_factory: GeneratorFactory | None = None,
    validator: ValidatorFn | None = None,
    executor_runtime_mode: str = "live_local_sota_gguf",
    tests_run: Sequence[str] | None = None,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """Run Exp 1428 or write an honest blocked artifact for cache/runtime blockers."""

    root = Path(project_root)
    output = _resolve(root, output_path)
    write_in_progress_artifact(
        output,
        project_root=root,
        run_date=run_date,
        write_observer=write_observer,
    )

    exp1397 = _read_json(_resolve(root, exp1397_path))
    resolved_specs, cache_diagnostics = exp1414.resolve_model_specs(cached_pair_fn)
    selected_spec = exp1414.select_repair_model(resolved_specs)
    if selected_spec is None:
        artifact = _blocked_artifact(
            project_root=root,
            run_date=run_date,
            model_specs=resolved_specs,
            cache_diagnostics=cache_diagnostics,
            tests_run=tests_run,
            honest_verdict="blocked_sota_model_cache_unavailable",
            executor_runtime_mode=executor_runtime_mode,
            local_sota_model_used=None,
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
            model_specs=resolved_specs,
            cache_diagnostics=diagnostics,
            tests_run=tests_run,
            honest_verdict="blocked_local_model_runtime_unavailable",
            executor_runtime_mode=executor_runtime_mode,
            local_sota_model_used=None,
        )
        _write_json(output, artifact, write_observer=write_observer)
        return artifact

    repair_cases = exp1414.repair_requests_from_exp1397(exp1397)
    sample = repair_cases[:MIN_REPAIR_HINT_CASES]
    active_validator = validator or _default_validator
    executor = DraftConditionedSchemaRepairExecutor(
        generator=generator,
        model_spec=selected_spec,
        validator=active_validator,
    )
    results = [executor.attempt(request) for request in sample]
    tested = len(sample)
    accepted = [result for result in results if result.accepted]
    schema_valid = [result for result in results if result.schema_valid]
    semantic_accepted = [result for result in results if result.semantic_accepted]
    rejection_counts = _rejection_reason_counts(results)

    artifact = _base_artifact(project_root=root, run_date=run_date, status="complete")
    artifact.update(
        {
            "model_specs": resolved_specs,
            "cache_diagnostics": cache_diagnostics,
            "local_sota_model_used": selected_spec.get("hf_id"),
            "local_sota_model_inference_used": executor_runtime_mode == "live_local_sota_gguf",
            "executor_runtime_mode": executor_runtime_mode,
            "repair_hint_cases_available": len(repair_cases),
            "repair_hint_cases_tested": tested,
            "repaired_cases_successful": len(accepted),
            "repaired_case_success_rate": _rate(len(accepted), tested),
            "schema_valid_rate": _rate(len(schema_valid), tested),
            "semantic_acceptance_rate": _rate(len(semantic_accepted), tested),
            "rejection_reason_counts": rejection_counts,
            "tests_run": list(tests_run or []),
            "repair_results": [result.to_dict() for result in results],
            "honest_verdict": _complete_verdict(
                available=len(repair_cases),
                tested=tested,
                successful=len(accepted),
                executor_runtime_mode=executor_runtime_mode,
            ),
        }
    )
    _write_json(output, artifact, write_observer=write_observer)
    return artifact


def _base_artifact(*, project_root: str | Path, run_date: str, status: str) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": status,
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "spec": ["REQ-VERIFY-1428", "SCENARIO-VERIFY-1428"],
            "source_experiments": ["exp1397", "exp1427"],
        },
        "model_specs": [dict(spec) for spec in MODEL_SPECS],
        "cache_diagnostics": {},
        "local_sota_model_used": None,
        "local_sota_model_inference_used": False,
        "executor_runtime_mode": "not_run",
        "repair_executor_v2_deployed": True,
        "repair_hint_cases_available": 0,
        "repair_hint_cases_tested": 0,
        "repaired_cases_successful": 0,
        "repaired_case_success_rate": 0.0,
        "schema_valid_rate": 0.0,
        "semantic_acceptance_rate": 0.0,
        "rejection_reason_counts": _zero_rejection_counts(),
        "dccd_repair_output_schema": DCCD_REPAIR_OUTPUT_SCHEMA,
        "tests_run": [],
        "honest_verdict": "not_run",
    }


def _blocked_artifact(
    *,
    project_root: str | Path,
    run_date: str,
    model_specs: Sequence[Mapping[str, Any]],
    cache_diagnostics: Mapping[str, Any],
    tests_run: Sequence[str] | None,
    honest_verdict: str,
    executor_runtime_mode: str,
    local_sota_model_used: str | None,
) -> dict[str, Any]:
    artifact = _base_artifact(project_root=project_root, run_date=run_date, status="blocked")
    artifact.update(
        {
            "model_specs": [dict(spec) for spec in model_specs],
            "cache_diagnostics": dict(cache_diagnostics),
            "local_sota_model_used": local_sota_model_used,
            "executor_runtime_mode": executor_runtime_mode,
            "tests_run": list(tests_run or []),
            "honest_verdict": honest_verdict,
        }
    )
    return artifact


def _complete_verdict(
    *,
    available: int,
    tested: int,
    successful: int,
    executor_runtime_mode: str = "live_local_sota_gguf",
) -> str:
    if available == 0:
        return "complete_no_repair_hint_cases_available"
    if tested < MIN_REPAIR_HINT_CASES:
        return "complete_dccd_schema_constrained_repair_v2_short_sample"
    if successful > 0:
        if executor_runtime_mode != "live_local_sota_gguf":
            return (
                "complete_dccd_schema_constrained_repair_v2_nonzero_repairs_"
                "prototype_no_headline_sota_claim"
            )
        return "complete_dccd_schema_constrained_repair_v2_nonzero_repairs"
    return "complete_dccd_schema_constrained_repair_v2_no_successful_repairs"


def _rejection_reason_counts(results: Sequence[Any]) -> dict[str, int]:
    counts = Counter(
        result.rejection_reason for result in results if result.rejection_reason is not None
    )
    merged = _zero_rejection_counts()
    merged.update({str(reason): int(count) for reason, count in counts.items()})
    return dict(sorted(merged.items()))


def _zero_rejection_counts() -> dict[str, int]:
    return {reason: 0 for reason in DCCD_REJECTION_REASONS}


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
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args(argv)
    run_experiment(
        project_root=Path(args.project_root),
        run_date=args.run_date,
        exp1397_path=Path(args.exp1397_path),
        output_path=Path(args.output_path),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
