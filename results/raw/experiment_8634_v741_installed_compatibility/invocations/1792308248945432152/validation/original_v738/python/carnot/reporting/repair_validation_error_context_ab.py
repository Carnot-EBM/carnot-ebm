"""Exp 1464 repair retry validation-error-context A/B evaluator.

Spec: REQ-VERIFY-1464, SCENARIO-VERIFY-1464
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.pipeline.certificate_repair_executor import (
    CertificateRepairRequest,
    validation_accepts_repair,
)
from carnot.pipeline.dccd_schema_constrained_repair import (
    DCCDRepairConfig,
    DCCDRepairOutputSchemaError,
    build_dccd_repair_prompt,
    build_dccd_retry_prompt,
    classify_dccd_rejection,
    parse_dccd_repair_model_output,
)
from carnot.reporting import certificate_llm_repair_executor_v1 as exp1414


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260507"
EXPERIMENT = "1464_repair_validation_error_context_ab"
SCHEMA = "repair_validation_error_context_ab_v1"
DEFAULT_EXP1397_PATH = REPO_ROOT / "results" / "experiment_1397_fullscale_pipeline_v2_200cases.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "experiment_1464_repair_validation_error_context_ab.json"
DEFAULT_MAX_CASES = 3

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "model_specs",
    "live_sota_model_inference_used",
    "validation_error_context_enabled",
    "cases_evaluated",
    "baseline_acceptance_rate",
    "context_acceptance_rate",
    "acceptance_delta_pp",
    "schema_validity_delta_pp",
    "semantic_correctness_delta_pp",
    "spilled_energy_diagnostic_available",
    "repair_executor_lineage_preserved",
    "repair_executor_lineage_retired",
    "commands_run",
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
    """REQ-VERIFY-1464: persist bootstrap JSON before model loading or A/B work."""

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
    max_cases: int = DEFAULT_MAX_CASES,
    commands_run: Sequence[str] | None = None,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """Run the same FoVer repair subset through baseline and context retries."""

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
            commands_run=commands_run,
            honest_verdict="blocked_sota_model_cache_unavailable",
        )
        _write_json(output, artifact, write_observer=write_observer)
        return artifact

    try:
        generator = (generator_factory or _default_generator_factory)(selected_spec)
    except Exception as exc:  # pragma: no cover - depends on local llama.cpp runtime.
        diagnostics = dict(cache_diagnostics)
        diagnostics["generator_error"] = f"{type(exc).__name__}: {exc}"
        artifact = _blocked_artifact(
            project_root=root,
            run_date=run_date,
            model_specs=resolved_specs,
            cache_diagnostics=diagnostics,
            commands_run=commands_run,
            honest_verdict="blocked_local_model_runtime_unavailable",
        )
        _write_json(output, artifact, write_observer=write_observer)
        return artifact

    repair_cases = exp1414.repair_requests_from_exp1397(exp1397)[: max(0, max_cases)]
    active_validator = validator or _default_validator
    config = DCCDRepairConfig(max_field_chars=900, max_output_chars=4000)
    per_case = [
        _evaluate_case(request, generator=generator, validator=active_validator, config=config)
        for request in repair_cases
    ]

    rates = _ab_rates(per_case)
    deltas = _ab_deltas(rates)
    improved = _lineage_metric_improved(deltas)
    live_inference = executor_runtime_mode == "live_local_sota_gguf" and bool(per_case)
    artifact = _base_artifact(project_root=root, run_date=run_date, status="complete")
    artifact.update(
        {
            "model_specs": resolved_specs,
            "cache_diagnostics": cache_diagnostics,
            "selected_model": _selected_model_evidence(selected_spec),
            "live_sota_model_inference_used": live_inference,
            "validation_error_context_enabled": True,
            "cases_evaluated": len(per_case),
            "case_ids_evaluated": [row["case_id"] for row in per_case],
            "baseline_acceptance_rate": rates["baseline_acceptance_rate"],
            "context_acceptance_rate": rates["context_acceptance_rate"],
            "baseline_schema_validity_rate": rates["baseline_schema_validity_rate"],
            "context_schema_validity_rate": rates["context_schema_validity_rate"],
            "baseline_semantic_correctness_rate": rates["baseline_semantic_correctness_rate"],
            "context_semantic_correctness_rate": rates["context_semantic_correctness_rate"],
            "baseline_false_acceptance_rate": rates["baseline_false_acceptance_rate"],
            "context_false_acceptance_rate": rates["context_false_acceptance_rate"],
            "acceptance_delta_pp": deltas["acceptance_delta_pp"],
            "schema_validity_delta_pp": deltas["schema_validity_delta_pp"],
            "semantic_correctness_delta_pp": deltas["semantic_correctness_delta_pp"],
            "false_acceptance_delta_pp": deltas["false_acceptance_delta_pp"],
            "spilled_energy_diagnostic_available": False,
            "spilled_energy_diagnostic_note": (
                "Unavailable: the bounded llama.cpp text-generation path did not "
                "request logits/logprobs, and Exp 1464 did not expand scope."
            ),
            "repair_executor_lineage_preserved": improved,
            "repair_executor_lineage_retired": not improved,
            "executor_runtime_mode": executor_runtime_mode,
            "live_inference_evidence": {
                "selected_hf_id": selected_spec.get("hf_id"),
                "model_path": selected_spec.get("model_path"),
                "gpu": selected_spec.get("gpu"),
                "generation_calls": len(per_case) * 3,
                "runtime_mode": executor_runtime_mode,
                "live_sota_model_inference_used": live_inference,
            },
            "per_case_results": per_case,
            "commands_run": list(commands_run or []),
            "honest_verdict": _complete_verdict(cases=len(per_case), improved=improved),
        }
    )
    _write_json(output, artifact, write_observer=write_observer)
    return artifact


def _evaluate_case(
    request: CertificateRepairRequest,
    *,
    generator: Callable[[str], str],
    validator: ValidatorFn,
    config: DCCDRepairConfig,
) -> dict[str, Any]:
    initial_prompt = build_dccd_repair_prompt(request, config)
    initial_raw = generator(initial_prompt)
    initial = _evaluate_output(request, initial_raw, validator)
    validation_error = initial["validation_error_message"] or "initial candidate did not fail"

    baseline_raw = generator(
        build_dccd_retry_prompt(
            request,
            failed_output=initial_raw,
            validation_error_message=validation_error,
            include_validation_error_context=False,
            config=config,
        )
    )
    context_raw = generator(
        build_dccd_retry_prompt(
            request,
            failed_output=initial_raw,
            validation_error_message=validation_error,
            include_validation_error_context=True,
            config=config,
        )
    )
    return {
        "case_id": request.case_id,
        "initial_failure": initial,
        "baseline": _evaluate_output(request, baseline_raw, validator),
        "context": _evaluate_output(request, context_raw, validator),
    }


def _evaluate_output(
    request: CertificateRepairRequest,
    raw_output: str,
    validator: ValidatorFn,
) -> dict[str, Any]:
    try:
        candidate = parse_dccd_repair_model_output(raw_output)
    except DCCDRepairOutputSchemaError as exc:
        return {
            "raw_output_preview": _preview(raw_output),
            "schema_valid": False,
            "semantic_correct": False,
            "accepted": False,
            "false_acceptance": False,
            "validation_result": {"error": str(exc)},
            "validation_error_message": f"schema_validation_failed: {exc}",
            "rejection_reason": "schema_validation_failed",
        }

    try:
        validation = dict(validator(request, candidate))
    except Exception as exc:  # pragma: no cover - validator integrations vary by experiment.
        return {
            "raw_output_preview": _preview(raw_output),
            "schema_valid": True,
            "semantic_correct": False,
            "accepted": False,
            "false_acceptance": False,
            "validation_result": {"error": f"{type(exc).__name__}: {exc}"},
            "validation_error_message": f"generation_or_validation_failed: {type(exc).__name__}: {exc}",
            "rejection_reason": "generation_or_validation_failed",
        }

    accepted = validation_accepts_repair(validation)
    rejection_reason = None if accepted else classify_dccd_rejection(
        "semantic_validation_failed",
        validation,
    )
    validation_error = "" if accepted else (
        f"semantic_validation_failed: {rejection_reason}; "
        f"validation_result={json.dumps(validation, sort_keys=True)}"
    )
    return {
        "raw_output_preview": _preview(raw_output),
        "schema_valid": True,
        "semantic_correct": accepted,
        "accepted": accepted,
        "false_acceptance": validation.get("false_acceptance") is True,
        "validation_result": validation,
        "validation_error_message": validation_error,
        "rejection_reason": rejection_reason,
    }


def _ab_rates(per_case: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        "baseline_acceptance_rate": _variant_rate(per_case, "baseline", "accepted"),
        "context_acceptance_rate": _variant_rate(per_case, "context", "accepted"),
        "baseline_schema_validity_rate": _variant_rate(per_case, "baseline", "schema_valid"),
        "context_schema_validity_rate": _variant_rate(per_case, "context", "schema_valid"),
        "baseline_semantic_correctness_rate": _variant_rate(per_case, "baseline", "semantic_correct"),
        "context_semantic_correctness_rate": _variant_rate(per_case, "context", "semantic_correct"),
        "baseline_false_acceptance_rate": _variant_rate(per_case, "baseline", "false_acceptance"),
        "context_false_acceptance_rate": _variant_rate(per_case, "context", "false_acceptance"),
    }


def _ab_deltas(rates: Mapping[str, float]) -> dict[str, float]:
    return {
        "acceptance_delta_pp": _pp(
            rates["context_acceptance_rate"] - rates["baseline_acceptance_rate"]
        ),
        "schema_validity_delta_pp": _pp(
            rates["context_schema_validity_rate"] - rates["baseline_schema_validity_rate"]
        ),
        "semantic_correctness_delta_pp": _pp(
            rates["context_semantic_correctness_rate"]
            - rates["baseline_semantic_correctness_rate"]
        ),
        "false_acceptance_delta_pp": _pp(
            rates["context_false_acceptance_rate"] - rates["baseline_false_acceptance_rate"]
        ),
    }


def _lineage_metric_improved(deltas: Mapping[str, float]) -> bool:
    return (
        deltas["acceptance_delta_pp"] > 0.0
        or deltas["schema_validity_delta_pp"] > 0.0
        or deltas["semantic_correctness_delta_pp"] > 0.0
        or deltas["false_acceptance_delta_pp"] < 0.0
    )


def _variant_rate(
    per_case: Sequence[Mapping[str, Any]],
    variant: str,
    field: str,
) -> float:
    count = sum(1 for row in per_case if row[variant].get(field) is True)
    return _rate(count, len(per_case))


def _base_artifact(*, project_root: str | Path, run_date: str, status: str) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": status,
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "spec": ["REQ-VERIFY-1464", "SCENARIO-VERIFY-1464"],
            "source_experiments": ["exp1397", "exp1427", "exp1428", "exp1463"],
        },
        "model_specs": [dict(spec) for spec in exp1414.MODEL_SPECS],
        "cache_diagnostics": {},
        "selected_model": {},
        "live_sota_model_inference_used": False,
        "validation_error_context_enabled": False,
        "cases_evaluated": 0,
        "case_ids_evaluated": [],
        "baseline_acceptance_rate": 0.0,
        "context_acceptance_rate": 0.0,
        "baseline_schema_validity_rate": 0.0,
        "context_schema_validity_rate": 0.0,
        "baseline_semantic_correctness_rate": 0.0,
        "context_semantic_correctness_rate": 0.0,
        "baseline_false_acceptance_rate": 0.0,
        "context_false_acceptance_rate": 0.0,
        "acceptance_delta_pp": 0.0,
        "schema_validity_delta_pp": 0.0,
        "semantic_correctness_delta_pp": 0.0,
        "false_acceptance_delta_pp": 0.0,
        "spilled_energy_diagnostic_available": False,
        "spilled_energy_diagnostic_note": "not_run",
        "repair_executor_lineage_preserved": False,
        "repair_executor_lineage_retired": False,
        "executor_runtime_mode": "not_run",
        "live_inference_evidence": {},
        "per_case_results": [],
        "commands_run": [],
        "honest_verdict": "not_run",
    }


def _blocked_artifact(
    *,
    project_root: str | Path,
    run_date: str,
    model_specs: Sequence[Mapping[str, Any]],
    cache_diagnostics: Mapping[str, Any],
    commands_run: Sequence[str] | None,
    honest_verdict: str,
) -> dict[str, Any]:
    artifact = _base_artifact(project_root=project_root, run_date=run_date, status="blocked")
    artifact.update(
        {
            "model_specs": [dict(spec) for spec in model_specs],
            "cache_diagnostics": dict(cache_diagnostics),
            "commands_run": list(commands_run or []),
            "honest_verdict": honest_verdict,
        }
    )
    return artifact


def _complete_verdict(*, cases: int, improved: bool) -> str:
    if cases == 0:
        return "complete_no_repair_hint_cases_available"
    if improved:
        return "complete_retry_context_improved_repair_executor_preserved"
    return "complete_no_retry_context_improvement_repair_executor_retired"


def _default_validator(
    _request: CertificateRepairRequest,
    _candidate: Any,
) -> Mapping[str, Any]:
    return {
        "constraint_passed": False,
        "semantic_result": "REPAIR_HINT",
        "repair_required": True,
        "false_acceptance": False,
        "fallback_reason": "no_validator_injected",
    }


def _default_generator_factory(
    model_spec: dict[str, Any],
) -> Callable[[str], str]:  # pragma: no cover - live GGUF inference is host-specific.
    _prepare_cuda_library_path(REPO_ROOT)
    from llama_cpp import Llama

    llm = Llama(
        model_path=str(model_spec["model_path"]),
        n_gpu_layers=-1 if int(model_spec.get("gpu", 0)) >= 0 else 0,
        main_gpu=max(int(model_spec.get("gpu", 0)), 0),
        n_ctx=2048,
        verbose=False,
    )

    def generate(prompt: str) -> str:
        completion = llm(
            prompt,
            max_tokens=128,
            temperature=0.0,
            echo=False,
            stop=["</s>", "<eos>"],
        )
        return _completion_text(completion).strip()

    return generate


def _prepare_cuda_library_path(project_root: Path) -> None:  # pragma: no cover
    site_packages = sorted((project_root / ".venv" / "lib").glob("python*/site-packages"))
    candidates: list[str] = []
    for site in site_packages:
        candidates.extend(
            [
                str(site / "nvidia" / "cuda_runtime" / "lib"),
                str(site / "nvidia" / "cublas" / "lib"),
            ]
        )
    existing = [path for path in candidates if Path(path).is_dir()]
    current = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part]
    deduped: list[str] = []
    for part in existing + current:
        if part not in deduped:
            deduped.append(part)
    os.environ["LD_LIBRARY_PATH"] = ":".join(deduped)


def _completion_text(completion: Any) -> str:  # pragma: no cover
    choices = completion.get("choices") if isinstance(completion, Mapping) else None
    if not choices:
        return ""
    first = choices[0]
    if isinstance(first, Mapping):
        return str(first.get("text") or first.get("message", {}).get("content") or "")
    return ""


def _selected_model_evidence(model_spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "hf_id": model_spec.get("hf_id"),
        "model_path": model_spec.get("model_path"),
        "gpu": model_spec.get("gpu"),
        "name": model_spec.get("name"),
        "role": model_spec.get("role"),
    }


def _preview(text: str, limit: int = 500) -> str:
    value = str(text or "")
    return value if len(value) <= limit else f"{value[: max(0, limit - 12)]}[truncated]"


def _rate(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _pp(value: float) -> float:
    return round(float(value) * 100.0, 6)


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
    parser.add_argument("--max-cases", type=int, default=DEFAULT_MAX_CASES)
    args = parser.parse_args(argv)
    started = time.monotonic()
    command = " ".join([sys.executable, "-m", "carnot.reporting.repair_validation_error_context_ab", *sys.argv[1:]])
    artifact = run_experiment(
        project_root=Path(args.project_root),
        run_date=args.run_date,
        exp1397_path=Path(args.exp1397_path),
        output_path=Path(args.output_path),
        max_cases=args.max_cases,
        commands_run=[command],
    )
    artifact["commands_run"].append(f"exp1464_wall_time_s={round(time.monotonic() - started, 6)}")
    _write_json(_resolve(Path(args.project_root), args.output_path), artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
