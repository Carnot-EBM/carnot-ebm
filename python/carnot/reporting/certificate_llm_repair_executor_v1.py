"""Exp 1414 bounded local LLM certificate repair executor.

Spec: REQ-VERIFY-1414, SCENARIO-VERIFY-1414
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.pipeline.certificate_repair_executor import (
    ALLOWED_REPAIR_OUTPUT_SCHEMA,
    BoundedLocalLLMCertificateRepairExecutor,
    CertificateRepairPipelineHook,
    CertificateRepairRequest,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260506"
EXPERIMENT = "1414_certificate_llm_repair_executor_v1"
SCHEMA = "certificate_llm_repair_executor_v1"
DEFAULT_EXP1397_PATH = (
    REPO_ROOT / "results" / "experiment_1397_fullscale_pipeline_v2_200cases.json"
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "results" / "experiment_1414_certificate_llm_repair_executor_v1.json"
)
MIN_REPAIR_HINT_CASES = 20

MODEL_SPECS: list[dict[str, str]] = [
    {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "primary_repair_model",
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
    "repair_executor_deployed",
    "repair_hint_cases_tested",
    "repaired_cases_successful",
    "repaired_case_success_rate",
    "semantic_equivalence_pass_rate_after_repair",
    "local_sota_model_used",
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
    """REQ-VERIFY-1414: persist bootstrap JSON before model or source loading."""

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
    tests_run: Sequence[str] | None = None,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """Run Exp 1414 or write an honest blocked artifact when local SOTA is absent."""

    root = Path(project_root)
    output = _resolve(root, output_path)
    write_in_progress_artifact(
        output,
        project_root=root,
        run_date=run_date,
        write_observer=write_observer,
    )

    exp1397 = _read_json(_resolve(root, exp1397_path))
    resolved_specs, cache_diagnostics = resolve_model_specs(cached_pair_fn)
    selected_spec = select_repair_model(resolved_specs)
    if selected_spec is None:
        artifact = _blocked_artifact(
            project_root=root,
            run_date=run_date,
            model_specs=resolved_specs,
            cache_diagnostics=cache_diagnostics,
            tests_run=tests_run,
            honest_verdict="blocked_sota_model_cache_unavailable",
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
        )
        _write_json(output, artifact, write_observer=write_observer)
        return artifact

    repair_cases = repair_requests_from_exp1397(exp1397)
    sample = repair_cases[:MIN_REPAIR_HINT_CASES]
    active_validator = validator or _default_validator
    executor = BoundedLocalLLMCertificateRepairExecutor(
        generator=generator,
        model_spec=selected_spec,
        validator=active_validator,
    )
    hook = CertificateRepairPipelineHook(executor=executor, enabled=True)
    results = [hook.attempt(request) for request in sample]
    accepted = [result for result in results if result is not None and result.accepted]
    semantic_pass = [
        result
        for result in accepted
        if result.validation_result.get("semantic_result") == "SAT"
        and result.validation_result.get("constraint_passed") is True
    ]
    tested = len(sample)
    artifact = _base_artifact(project_root=root, run_date=run_date, status="complete")
    artifact.update(
        {
            "model_specs": resolved_specs,
            "cache_diagnostics": cache_diagnostics,
            "repair_hint_cases_available": len(repair_cases),
            "repair_hint_cases_tested": tested,
            "repaired_cases_successful": len(accepted),
            "repaired_case_success_rate": _rate(len(accepted), tested),
            "semantic_equivalence_pass_rate_after_repair": _rate(len(semantic_pass), tested),
            "local_sota_model_used": selected_spec.get("hf_id"),
            "tests_run": list(tests_run or []),
            "repair_results": [_result_dict(result) for result in results if result is not None],
            "honest_verdict": _complete_verdict(
                tested=tested,
                successful=len(accepted),
                available=len(repair_cases),
            ),
        }
    )
    _write_json(output, artifact, write_observer=write_observer)
    return artifact


def resolve_model_specs(
    cached_pair_fn: CachedPairFn | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve mandated SOTA GGUF specs via `cached_sota_pair()` style lookup."""

    resolver = cached_pair_fn or _cached_sota_pair
    try:
        cached_pair = resolver(gpu_indices=(0, 1), preferred_quant="Q4_K_M")
    except Exception as exc:
        cached_pair = None
        resolver_error = f"{type(exc).__name__}: {exc}"
    else:
        resolver_error = None

    cached_by_hf = {str(spec.get("hf_id")): dict(spec) for spec in cached_pair or []}
    model_specs: list[dict[str, Any]] = []
    for required in MODEL_SPECS:
        resolved = dict(required)
        cached = cached_by_hf.get(required["hf_id"])
        resolved["cache_status"] = "available" if cached and cached.get("model_path") else "missing"
        if cached:
            resolved.update({key: value for key, value in cached.items() if key != "role"})
            resolved["role"] = required["role"]
        model_specs.append(resolved)

    diagnostics = {
        "cached_pair_available": bool(cached_pair),
        "cached_pair_hf_ids": [spec.get("hf_id") for spec in cached_pair or []],
        "missing_hf_ids": [
            spec["hf_id"] for spec in model_specs if spec.get("cache_status") != "available"
        ],
        "resolver_error": resolver_error,
    }
    return model_specs, diagnostics


def select_repair_model(model_specs: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Choose primary, then dense fallback, then MoE fallback from resolved specs."""

    role_order = ("primary_repair_model", "dense_fallback", "moe_fallback")
    for role in role_order:
        for spec in model_specs:
            if spec.get("role") == role and spec.get("model_path"):
                return dict(spec)
    return None


def repair_requests_from_exp1397(exp1397_artifact: Mapping[str, Any]) -> list[CertificateRepairRequest]:
    """Build executor requests from Exp 1397 generation, semantic, and repair rows."""

    generation_by_id = _rows_by_case_id(exp1397_artifact.get("generation_rows"))
    certificate_by_id = _rows_by_case_id(exp1397_artifact.get("certificate_rows"))
    semantic_by_id = _rows_by_case_id(exp1397_artifact.get("semantic_validation_rows"))
    scheduler_by_id = _rows_by_case_id(exp1397_artifact.get("scheduler_rows"))
    requests: list[CertificateRepairRequest] = []
    for repair in exp1397_artifact.get("repair_localization_rows") or []:
        if not isinstance(repair, Mapping) or not repair.get("repair_hint"):
            continue
        case_id = str(repair.get("case_id") or "")
        generation = generation_by_id.get(case_id, {})
        certificate = certificate_by_id.get(case_id, {})
        semantic = semantic_by_id.get(case_id, {})
        scheduler = scheduler_by_id.get(case_id, {})
        requests.append(
            CertificateRepairRequest(
                case_id=case_id,
                original_prompt=str(
                    generation.get("reasoning_text")
                    or generation.get("prompt")
                    or certificate.get("reasoning_text")
                    or ""
                ),
                current_certificate=_certificate_text(generation, certificate),
                repair_hint=str(repair.get("repair_hint") or ""),
                validator_error=_validator_feedback(
                    repair_row=repair,
                    semantic_row=semantic,
                    scheduler_row=scheduler,
                ),
                allowed_output_schema=ALLOWED_REPAIR_OUTPUT_SCHEMA,
            )
        )
    return requests


def _base_artifact(*, project_root: str | Path, run_date: str, status: str) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": status,
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "spec": ["REQ-VERIFY-1414", "SCENARIO-VERIFY-1414"],
            "source_experiments": ["exp1397", "exp1413"],
        },
        "model_specs": [dict(spec) for spec in MODEL_SPECS],
        "repair_executor_deployed": True,
        "repair_hint_cases_available": 0,
        "repair_hint_cases_tested": 0,
        "repaired_cases_successful": 0,
        "repaired_case_success_rate": 0.0,
        "semantic_equivalence_pass_rate_after_repair": 0.0,
        "local_sota_model_used": None,
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
) -> dict[str, Any]:
    artifact = _base_artifact(project_root=project_root, run_date=run_date, status="blocked")
    artifact.update(
        {
            "model_specs": [dict(spec) for spec in model_specs],
            "cache_diagnostics": dict(cache_diagnostics),
            "tests_run": list(tests_run or []),
            "honest_verdict": honest_verdict,
        }
    )
    return artifact


def _complete_verdict(*, tested: int, successful: int, available: int) -> str:
    if available == 0:
        return "complete_no_repair_hint_cases_available"
    if tested < MIN_REPAIR_HINT_CASES:
        return "complete_repair_executor_validated_on_available_short_sample"
    if successful > 0:
        return "complete_repair_executor_validated_on_sample"
    return "complete_repair_executor_no_successful_repairs"


def _result_dict(result: Any) -> dict[str, Any]:
    return {
        "case_id": result.case_id,
        "attempted": result.attempted,
        "accepted": result.accepted,
        "local_model_used": result.local_model_used,
        "fallback_reason": result.fallback_reason,
        "validation_result": result.validation_result,
        "runtime_s": result.runtime_s,
    }


def _certificate_text(
    generation_row: Mapping[str, Any],
    certificate_row: Mapping[str, Any],
) -> str:
    if generation_row.get("full_certificate_text"):
        return str(generation_row.get("full_certificate_text"))
    prefix = str(generation_row.get("certificate_prefix") or certificate_row.get("certificate_prefix") or "")
    body = str(generation_row.get("certificate_body") or certificate_row.get("certificate_body") or "")
    return f"{prefix}{body}"


def _validator_feedback(
    *,
    repair_row: Mapping[str, Any],
    semantic_row: Mapping[str, Any],
    scheduler_row: Mapping[str, Any],
) -> str:
    feedback = {
        "localized_constraint": repair_row.get("localized_constraint"),
        "minimal_local_change": repair_row.get("minimal_local_change"),
        "semantic_result": semantic_row.get("semantic_result"),
        "failure_reason": semantic_row.get("failure_reason"),
        "scheduler_action": scheduler_row.get("scheduler_action"),
        "repair_required": scheduler_row.get("repair_required"),
        "full_pipeline_pass": scheduler_row.get("full_pipeline_pass"),
    }
    return json.dumps(feedback, sort_keys=True)


def _rows_by_case_id(rows: object) -> dict[str, Mapping[str, Any]]:
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("case_id")): row
        for row in rows
        if isinstance(row, Mapping) and row.get("case_id") is not None
    }


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
    from carnot.pipeline.gemma4_quantized_loader import Gemma4QuantizedLoader

    loader = Gemma4QuantizedLoader(
        model_path=str(model_spec.get("model_path") or ""),
        n_gpu_layers=-1,
        max_tokens=384,
    )
    if not loader.load() or getattr(loader, "_stub_mode", False):
        raise RuntimeError("local GGUF model could not be loaded without stub mode")
    return loader.generate


def _cached_sota_pair(**kwargs: Any) -> list[dict[str, Any]] | None:  # pragma: no cover
    from carnot.inference.sota_models import cached_sota_pair

    return cached_sota_pair(**kwargs)


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
