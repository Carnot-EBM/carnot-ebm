"""Exp 1419 full-scale pipeline v3 repair-executor rerun.

Spec: REQ-VERIFY-1419, SCENARIO-VERIFY-1419
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.reporting import certificate_llm_repair_executor_v1 as exp1414


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260506"
EXPERIMENT = "1419_fullscale_pipeline_v3_repair_executor"
SCHEMA = "fullscale_pipeline_v3_repair_executor_v1"
DEFAULT_EXP1397_PATH = (
    REPO_ROOT / "results" / "experiment_1397_fullscale_pipeline_v2_200cases.json"
)
DEFAULT_EXP1414_PATH = (
    REPO_ROOT / "results" / "experiment_1414_certificate_llm_repair_executor_v1.json"
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "results" / "experiment_1419_fullscale_pipeline_v3_repair_executor.json"
)
MIN_SOURCE_CASES = 200
EXP1397_BASELINE_FULL_PIPELINE_PASS_RATE = 0.305
HEADLINE_FULL_PIPELINE_TARGET = 0.40

MODEL_SPECS = [
    {"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF", "role": "primary_pipeline_repair_model"},
    {"hf_id": "unsloth/gemma-4-31B-it-GGUF", "role": "dense_fallback"},
    {"hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF", "role": "moe_fallback"},
]
PIPELINE_ROLE_BY_HF_ID = {spec["hf_id"]: spec["role"] for spec in MODEL_SPECS}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "model_specs",
    "cases_evaluated",
    "certificate_parse_rate",
    "semantic_validation_pass_rate",
    "repair_hint_cases_total",
    "repaired_cases_successful",
    "repair_success_rate",
    "full_pipeline_pass_rate",
    "full_pipeline_headline_gate_met",
    "honest_verdict",
)

CachedPairFn = exp1414.CachedPairFn
GeneratorFactory = exp1414.GeneratorFactory
ValidatorFn = exp1414.ValidatorFn
WriteObserver = Callable[[Path, dict[str, Any]], None]


def write_in_progress_artifact(
    path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """REQ-VERIFY-1419: persist bootstrap JSON before source or model loading."""

    artifact = _base_artifact(project_root=project_root, run_date=run_date, status="in_progress")
    artifact["honest_verdict"] = "in_progress"
    _write_json(Path(path), artifact, write_observer=write_observer)
    return artifact


def run_experiment(
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    exp1397_path: str | Path = DEFAULT_EXP1397_PATH,
    exp1414_path: str | Path = DEFAULT_EXP1414_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    cached_pair_fn: CachedPairFn | None = None,
    generator_factory: GeneratorFactory | None = None,
    validator: ValidatorFn | None = None,
    tests_run: Sequence[str] | None = None,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """Run the 200-case Exp 1397 replay with the Exp 1414 repair hook enabled."""

    root = Path(project_root)
    output = _resolve(root, output_path)
    write_in_progress_artifact(
        output,
        project_root=root,
        run_date=run_date,
        write_observer=write_observer,
    )

    exp1397_artifact = _read_json(_resolve(root, exp1397_path))
    exp1414_artifact = _read_json(_resolve(root, exp1414_path))
    source_metrics = _source_metrics(exp1397_artifact)
    repair_requests = exp1414.repair_requests_from_exp1397(exp1397_artifact)

    if exp1414_artifact.get("repair_executor_deployed") is not True:
        artifact = _blocked_artifact(
            project_root=root,
            run_date=run_date,
            source_metrics=source_metrics,
            repair_hint_cases_total=len(repair_requests),
            model_specs=MODEL_SPECS,
            blocker="exp1414_repair_executor_not_deployed",
            blocker_detail="Exp 1414 artifact did not set repair_executor_deployed=true.",
            exp1414_deployed=False,
            tests_run=tests_run,
        )
        _write_json(output, artifact, write_observer=write_observer)
        return artifact

    if source_metrics["cases_evaluated"] < MIN_SOURCE_CASES:
        artifact = _blocked_artifact(
            project_root=root,
            run_date=run_date,
            source_metrics=source_metrics,
            repair_hint_cases_total=len(repair_requests),
            model_specs=MODEL_SPECS,
            blocker="source_case_count_below_200",
            blocker_detail=(
                f"Exp 1397 source cases_evaluated={source_metrics['cases_evaluated']} "
                f"is below required {MIN_SOURCE_CASES}."
            ),
            exp1414_deployed=True,
            tests_run=tests_run,
        )
        _write_json(output, artifact, write_observer=write_observer)
        return artifact

    resolved_specs, cache_diagnostics = exp1414.resolve_model_specs(cached_pair_fn)
    selected_spec = exp1414.select_repair_model(resolved_specs)
    if selected_spec is None:
        artifact = _blocked_artifact(
            project_root=root,
            run_date=run_date,
            source_metrics=source_metrics,
            repair_hint_cases_total=len(repair_requests),
            model_specs=_pipeline_model_specs(resolved_specs),
            cache_diagnostics=cache_diagnostics,
            blocker="sota_model_cache_unavailable",
            blocker_detail="No mandated local SOTA GGUF model resolved with a model_path.",
            exp1414_deployed=True,
            tests_run=tests_run,
        )
        _write_json(output, artifact, write_observer=write_observer)
        return artifact

    try:
        generator = (generator_factory or exp1414._default_generator_factory)(selected_spec)
    except Exception as exc:  # pragma: no cover - exact runtime failures are tested via injection.
        artifact = _blocked_artifact(
            project_root=root,
            run_date=run_date,
            source_metrics=source_metrics,
            repair_hint_cases_total=len(repair_requests),
            model_specs=_pipeline_model_specs(resolved_specs),
            cache_diagnostics=cache_diagnostics,
            blocker="local_model_runtime_unavailable",
            blocker_detail=f"{type(exc).__name__}: {exc}",
            exp1414_deployed=True,
            actual_model_used=str(selected_spec.get("hf_id") or selected_spec.get("name") or ""),
            tests_run=tests_run,
        )
        _write_json(output, artifact, write_observer=write_observer)
        return artifact

    executor = exp1414.BoundedLocalLLMCertificateRepairExecutor(
        generator=generator,
        model_spec=selected_spec,
        validator=validator or exp1414._default_validator,
    )
    hook = exp1414.CertificateRepairPipelineHook(executor=executor, enabled=True)
    repair_results = [hook.attempt(request) for request in repair_requests]
    accepted_results = [
        result for result in repair_results if result is not None and result.accepted
    ]
    original_pass_ids = _original_pass_case_ids(exp1397_artifact)
    accepted_repair_ids = {result.case_id for result in accepted_results}
    final_pass_ids = original_pass_ids | accepted_repair_ids
    final_full_rate = _rate(len(final_pass_ids), source_metrics["cases_evaluated"])
    headline_met = final_full_rate >= HEADLINE_FULL_PIPELINE_TARGET

    artifact = _base_artifact(project_root=root, run_date=run_date, status="complete")
    artifact.update(
        {
            **source_metrics,
            "model_specs": _pipeline_model_specs(resolved_specs),
            "cache_diagnostics": cache_diagnostics,
            "repair_executor_enabled": True,
            "exp1414_repair_executor_deployed": True,
            "repair_hint_cases_total": len(repair_requests),
            "repaired_cases_successful": len(accepted_results),
            "repair_success_rate": _rate(len(accepted_results), len(repair_requests)),
            "original_full_pipeline_pass_cases": len(original_pass_ids),
            "final_full_pipeline_pass_cases": len(final_pass_ids),
            "full_pipeline_pass_rate": final_full_rate,
            "full_pipeline_delta_vs_exp1397": round(
                final_full_rate - EXP1397_BASELINE_FULL_PIPELINE_PASS_RATE, 6
            ),
            "full_pipeline_headline_gate_met": headline_met,
            "actual_model_used": selected_spec.get("hf_id"),
            "local_sota_model_used": selected_spec.get("hf_id"),
            "repair_results": [
                exp1414._result_dict(result) for result in repair_results if result is not None
            ],
            "tests_run": list(tests_run or []),
            "honest_verdict": _complete_verdict(headline_met=headline_met),
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
            "spec": ["REQ-VERIFY-1419", "SCENARIO-VERIFY-1419"],
            "source_experiments": ["exp1397", "exp1414"],
        },
        "model_specs": [dict(spec) for spec in MODEL_SPECS],
        "cases_evaluated": 0,
        "certificate_parse_rate": 0.0,
        "semantic_validation_pass_rate": 0.0,
        "repair_hint_cases_total": 0,
        "repaired_cases_successful": 0,
        "repair_success_rate": 0.0,
        "full_pipeline_pass_rate": 0.0,
        "full_pipeline_headline_gate_met": False,
        "headline_full_pipeline_target": HEADLINE_FULL_PIPELINE_TARGET,
        "exp1397_baseline_full_pipeline_pass_rate": EXP1397_BASELINE_FULL_PIPELINE_PASS_RATE,
        "full_pipeline_delta_vs_exp1397": round(-EXP1397_BASELINE_FULL_PIPELINE_PASS_RATE, 6),
        "repair_executor_enabled": False,
        "exp1414_repair_executor_deployed": False,
        "actual_model_used": None,
        "local_sota_model_used": None,
        "blocker": None,
        "blocker_detail": None,
        "tests_run": [],
        "honest_verdict": "not_run",
    }


def _blocked_artifact(
    *,
    project_root: str | Path,
    run_date: str,
    source_metrics: Mapping[str, Any],
    repair_hint_cases_total: int,
    model_specs: Sequence[Mapping[str, Any]],
    blocker: str,
    blocker_detail: str,
    exp1414_deployed: bool,
    cache_diagnostics: Mapping[str, Any] | None = None,
    actual_model_used: str | None = None,
    tests_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    artifact = _base_artifact(project_root=project_root, run_date=run_date, status="blocked")
    artifact.update(
        {
            **dict(source_metrics),
            "model_specs": [dict(spec) for spec in model_specs],
            "cache_diagnostics": dict(cache_diagnostics or {}),
            "repair_hint_cases_total": int(repair_hint_cases_total),
            "repair_success_rate": 0.0,
            "full_pipeline_headline_gate_met": False,
            "repair_executor_enabled": False,
            "exp1414_repair_executor_deployed": bool(exp1414_deployed),
            "actual_model_used": actual_model_used,
            "local_sota_model_used": actual_model_used,
            "blocker": blocker,
            "blocker_detail": blocker_detail,
            "tests_run": list(tests_run or []),
            "honest_verdict": f"blocked_{blocker}",
        }
    )
    return artifact


def _source_metrics(exp1397_artifact: Mapping[str, Any]) -> dict[str, Any]:
    scheduler_rows = _rows(exp1397_artifact.get("scheduler_rows"))
    certificate_rows = _rows(exp1397_artifact.get("certificate_rows"))
    semantic_rows = _rows(exp1397_artifact.get("semantic_validation_rows"))
    cases_evaluated = int(
        exp1397_artifact.get("cases_evaluated")
        or exp1397_artifact.get("total_fover_cases")
        or len(scheduler_rows)
        or len(certificate_rows)
    )
    original_passes = len(_original_pass_case_ids(exp1397_artifact))
    return {
        "cases_evaluated": cases_evaluated,
        "certificate_parse_rate": _metric_or_rows(
            exp1397_artifact.get("certificate_parse_rate"),
            rows=certificate_rows,
            key="parseable",
            denominator=cases_evaluated,
        ),
        "semantic_validation_pass_rate": _metric_or_rows(
            exp1397_artifact.get("semantic_validation_pass_rate"),
            rows=semantic_rows,
            key="constraint_passed",
            denominator=cases_evaluated,
        ),
        "full_pipeline_pass_rate": _metric_or_count(
            exp1397_artifact.get("full_pipeline_pass_rate"),
            numerator=original_passes,
            denominator=cases_evaluated,
        ),
    }


def _pipeline_model_specs(model_specs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for spec in model_specs:
        record = dict(spec)
        role = PIPELINE_ROLE_BY_HF_ID.get(str(record.get("hf_id") or ""))
        if role:
            record["role"] = role
        records.append(record)
    return records


def _metric_or_rows(
    value: object,
    *,
    rows: Sequence[Mapping[str, Any]],
    key: str,
    denominator: int,
) -> float:
    parsed = _float(value)
    if parsed is not None:
        return parsed
    return _rate(sum(1 for row in rows if row.get(key) is True), denominator)


def _metric_or_count(value: object, *, numerator: int, denominator: int) -> float:
    parsed = _float(value)
    if parsed is not None:
        return parsed
    return _rate(numerator, denominator)


def _original_pass_case_ids(exp1397_artifact: Mapping[str, Any]) -> set[str]:
    return {
        str(row.get("case_id"))
        for row in _rows(exp1397_artifact.get("scheduler_rows"))
        if row.get("case_id") is not None and row.get("full_pipeline_pass") is True
    }


def _complete_verdict(*, headline_met: bool) -> str:
    if headline_met:
        return "headline_full_pipeline_gate_met"
    return "not_headline_full_pipeline_below_0_40"


def _rows(rows: object) -> list[Mapping[str, Any]]:
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _float(value: object) -> float | None:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


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
    parser.add_argument("--exp1414-path", default=str(DEFAULT_EXP1414_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args(argv)
    run_experiment(
        project_root=Path(args.project_root),
        run_date=args.run_date,
        exp1397_path=Path(args.exp1397_path),
        exp1414_path=Path(args.exp1414_path),
        output_path=Path(args.output_path),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
