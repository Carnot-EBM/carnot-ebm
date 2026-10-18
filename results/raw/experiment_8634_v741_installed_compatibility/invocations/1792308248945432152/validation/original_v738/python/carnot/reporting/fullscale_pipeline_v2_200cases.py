"""Exp 1397 full-scale certificate and semantic repair pipeline v2.

Spec: REQ-VERIFY-1397, SCENARIO-VERIFY-1397.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.reporting import fullscale_certificate_semantic_repair_100cases as exp1382


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260506"
EXPERIMENT = "1397_fullscale_pipeline_v2_200cases"
SCHEMA = "fullscale_pipeline_v2_200cases_v1"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "experiment_1397_fullscale_pipeline_v2_200cases.json"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "results" / "exp1397_ckpt.json"
DEFAULT_FOVER_PATH = exp1382.DEFAULT_FOVER_PATH
DEFAULT_EXP1381_PATH = exp1382.DEFAULT_EXP1381_PATH
DEFAULT_EXP1396_PATH = (
    REPO_ROOT / "results" / "experiment_1396_semantic_validation_pass_rate_fix_v1.json"
)
TARGET_FOVER_CASES = 200
EXP1382_SEMANTIC_BASELINE = 0.59
EXP1382_FULL_PIPELINE_BASELINE = 0.29
HEADLINE_SEMANTIC_THRESHOLD = 0.70
HEADLINE_FULL_PIPELINE_THRESHOLD = 0.40
MANDATED_HEADLINE_MODEL_IDS = exp1382.MANDATED_HEADLINE_MODEL_IDS

FoVerPipelineCase = exp1382.FoVerPipelineCase
CertificateCase = exp1382.CertificateCase
CranePrompts = exp1382.CranePrompts
CraneGenerationResult = exp1382.CraneGenerationResult
GenerationFn = exp1382.GenerationFn
DviPredictor = exp1382.DviPredictor
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
WriteObserver = Callable[[Path, dict[str, Any]], None]

structural_tag = exp1382.structural_tag
certificate_body_for_state = exp1382.certificate_body_for_state

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "cases_evaluated",
    "models_used",
    "certificate_extract_count",
    "certificate_parse_rate",
    "semantic_validation_pass_rate",
    "full_pipeline_pass_rate",
    "semantic_validation_improvement_vs_exp1382",
    "full_pipeline_improvement_vs_exp1382",
    "headline_result_allowed",
    "honest_verdict",
)


def write_in_progress_artifact(
    path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """REQ-VERIFY-1397: persist the bootstrap artifact before any expensive work.

    This experiment is intended to be a publication-quality headline result.
    Writing the bootstrap artifact first makes interrupted GPU runs auditable:
    downstream tooling can distinguish "never started" from "started and was
    interrupted before model loading or corpus replay finished."
    """

    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "in_progress",
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "spec": ["REQ-VERIFY-1397", "SCENARIO-VERIFY-1397"],
            "source_experiments": ["exp1382", "exp1391", "exp1396"],
        },
        "cases_evaluated": 0,
        "MODEL_SPECS": [],
        "models_used": [],
        "certificate_extract_count": 0,
        "certificate_parse_rate": None,
        "semantic_validation_pass_rate": None,
        "full_pipeline_pass_rate": None,
        "semantic_validation_improvement_vs_exp1382": None,
        "full_pipeline_improvement_vs_exp1382": None,
        "headline_result_allowed": False,
        "honest_verdict": "in_progress",
    }
    _write_json(Path(path), artifact, write_observer=write_observer)
    return artifact


def build_fullscale_pipeline_v2_artifact(
    *,
    cases: Sequence[FoVerPipelineCase],
    model_specs: Sequence[Mapping[str, Any]] | None,
    dvi_checkpoint_path: str | Path,
    exp1396_artifact: Mapping[str, Any],
    dvi_predictor: DviPredictor | None = None,
    generation_fn: GenerationFn | None = None,
    run_date: str = RUN_DATE,
    project_root: str | Path = REPO_ROOT,
    checkpoint_path: str | Path | None = DEFAULT_CHECKPOINT_PATH,
    runtime_settings: Mapping[str, Any] | None = None,
    max_models: int = 1,
) -> dict[str, Any]:
    """Run the inherited full pipeline and adapt its result to Exp1397.

    Exp1396 changed the semantic validator, not the certificate parser or MCS
    stages.  Reusing Exp1382's runner keeps the measured pipeline comparable to
    the baseline while this wrapper changes only the sample size, prerequisite
    gate, result schema, and headline thresholds requested for Exp1397.
    """

    if exp1396_artifact.get("semantic_validation_improvement_measured") is not True:
        return _blocked_artifact(
            terminal_blocker="exp1396_semantic_validation_fix_not_confirmed",
            model_specs=model_specs or [],
            run_date=run_date,
            project_root=project_root,
            exp1396_confirmed=False,
        )

    pipeline_artifact = exp1382.build_fullscale_pipeline_artifact(
        cases=cases,
        model_specs=model_specs,
        dvi_checkpoint_path=dvi_checkpoint_path,
        dvi_predictor=dvi_predictor,
        generation_fn=generation_fn,
        run_date=run_date,
        project_root=project_root,
        checkpoint_path=checkpoint_path,
        runtime_settings=runtime_settings,
        max_models=max_models,
    )
    return finalize_exp1397_artifact(
        pipeline_artifact,
        model_specs=model_specs or [],
        exp1396_artifact=exp1396_artifact,
        run_date=run_date,
        project_root=project_root,
    )


def finalize_exp1397_artifact(
    pipeline_artifact: Mapping[str, Any],
    *,
    model_specs: Sequence[Mapping[str, Any]],
    exp1396_artifact: Mapping[str, Any],
    run_date: str = RUN_DATE,
    project_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Return the Exp1397 terminal artifact from an Exp1382-style pipeline run."""

    artifact = dict(pipeline_artifact)
    cases_evaluated = int(artifact.get("cases_evaluated") or artifact.get("total_fover_cases") or 0)
    semantic_rate = _float(artifact.get("semantic_validation_pass_rate"))
    full_rate = _float(artifact.get("full_pipeline_pass_rate"))
    certificate_parse_rate = _float(artifact.get("certificate_parse_rate"))
    certificate_extract_count = int(artifact.get("certificate_extract_count") or 0)
    exp1396_confirmed = exp1396_artifact.get("semantic_validation_improvement_measured") is True
    metric_gate = (
        semantic_rate >= HEADLINE_SEMANTIC_THRESHOLD
        and full_rate >= HEADLINE_FULL_PIPELINE_THRESHOLD
    )
    sota_gate = _sota_generation_gate(
        artifact,
        model_specs=model_specs,
        cases_evaluated=cases_evaluated,
    )
    terminal_blocker = artifact.get("terminal_blocker")
    headline_allowed = bool(
        artifact.get("status") == "complete"
        and exp1396_confirmed
        and metric_gate
        and sota_gate
        and terminal_blocker is None
    )
    models_used = [
        dict(row) for row in (artifact.get("models_used") or _model_records(model_specs))
    ]
    model_specs_field = [dict(spec) for spec in model_specs]

    metadata = dict(artifact.get("artifact_metadata") or {})
    metadata.update(
        {
            "project_root": str(project_root),
            "run_date": run_date,
            "spec": ["REQ-VERIFY-1397", "SCENARIO-VERIFY-1397"],
            "source_experiments": [
                "exp1382_fullscale_certificate_semantic_repair_100cases",
                "exp1391_fullscale_pipeline_failure_diagnosis",
                "exp1396_semantic_validation_pass_rate_fix_v1",
            ],
        }
    )
    artifact.update(
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "run_date": run_date,
            "artifact_metadata": metadata,
            "status": str(artifact.get("status") or "complete"),
            "cases_evaluated": cases_evaluated,
            "target_cases": TARGET_FOVER_CASES,
            "MODEL_SPECS": model_specs_field,
            "models_used": models_used,
            "certificate_extract_count": certificate_extract_count,
            "certificate_parse_rate": certificate_parse_rate,
            "semantic_validation_pass_rate": semantic_rate,
            "full_pipeline_pass_rate": full_rate,
            "semantic_validation_improvement_vs_exp1382": round(
                semantic_rate - EXP1382_SEMANTIC_BASELINE, 6
            ),
            "full_pipeline_improvement_vs_exp1382": round(
                full_rate - EXP1382_FULL_PIPELINE_BASELINE, 6
            ),
            "exp1382_baseline": {
                "semantic_validation_pass_rate": EXP1382_SEMANTIC_BASELINE,
                "full_pipeline_pass_rate": EXP1382_FULL_PIPELINE_BASELINE,
            },
            "exp1396_fix_confirmed": exp1396_confirmed,
            "headline_metric_gate_passed": metric_gate,
            "headline_sota_generation_gate_passed": sota_gate,
            "headline_semantic_validation_threshold": HEADLINE_SEMANTIC_THRESHOLD,
            "headline_full_pipeline_threshold": HEADLINE_FULL_PIPELINE_THRESHOLD,
            "headline_result_allowed": headline_allowed,
            "repair_engine": "VERGE_MCS_localization_v1",
            "semantic_validator": "NSVIF_Z3_plus_exp1396_fover_semantic_calibration",
        }
    )
    for row in models_used:
        selected = bool(row.get("selected_for_generation"))
        row["headline_result_allowed"] = bool(headline_allowed and selected)
        if not headline_allowed and selected:
            row["fallback_reason"] = _headline_blocker_label(
                exp1396_confirmed=exp1396_confirmed,
                semantic_rate=semantic_rate,
                full_rate=full_rate,
                sota_gate=sota_gate,
                terminal_blocker=str(terminal_blocker) if terminal_blocker else None,
            )
    artifact["models_used"] = models_used
    artifact["honest_verdict"] = _honest_verdict(
        exp1396_confirmed=exp1396_confirmed,
        semantic_rate=semantic_rate,
        full_rate=full_rate,
        sota_gate=sota_gate,
        terminal_blocker=str(terminal_blocker) if terminal_blocker else None,
        headline_result_allowed=headline_allowed,
    )
    return artifact


def run_experiment(
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    fover_path: str | Path = DEFAULT_FOVER_PATH,
    exp1381_path: str | Path = DEFAULT_EXP1381_PATH,
    exp1396_path: str | Path = DEFAULT_EXP1396_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    checkpoint_path: str | Path | None = DEFAULT_CHECKPOINT_PATH,
    cached_pair_fn: CachedPairFn | None = None,
    generation_fn: GenerationFn | None = None,
    dvi_predictor: DviPredictor | None = None,
    target_cases: int = TARGET_FOVER_CASES,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """Write progress, confirm Exp1396, resolve SOTA specs, and run 200 cases."""

    root = Path(project_root)
    output = _resolve(root, output_path)
    checkpoint = None if checkpoint_path is None else _resolve(root, checkpoint_path)
    write_in_progress_artifact(
        output,
        project_root=root,
        run_date=run_date,
        write_observer=write_observer,
    )

    exp1396_artifact = _read_json(_resolve(root, exp1396_path))
    if exp1396_artifact.get("semantic_validation_improvement_measured") is not True:
        artifact = _blocked_artifact(
            terminal_blocker="exp1396_semantic_validation_fix_not_confirmed",
            model_specs=[],
            run_date=run_date,
            project_root=root,
            exp1396_confirmed=False,
        )
        _write_json(output, artifact, write_observer=write_observer)
        return artifact

    model_specs = _resolve_model_specs(cached_pair_fn)
    exp1381_artifact = _read_json(_resolve(root, exp1381_path))
    dvi_checkpoint = _dvi_checkpoint_from_exp1381(exp1381_artifact)
    if dvi_checkpoint is None:
        artifact = _blocked_artifact(
            terminal_blocker="exp1381_dvi_checkpoint_not_deployed",
            model_specs=model_specs or [],
            run_date=run_date,
            project_root=root,
            exp1396_confirmed=True,
        )
        _write_json(output, artifact, write_observer=write_observer)
        return artifact

    cases = exp1382.load_fover_cases(_resolve(root, fover_path), target_cases=target_cases)
    artifact = build_fullscale_pipeline_v2_artifact(
        cases=cases,
        model_specs=model_specs,
        dvi_checkpoint_path=dvi_checkpoint,
        exp1396_artifact=exp1396_artifact,
        dvi_predictor=dvi_predictor,
        generation_fn=generation_fn,
        run_date=run_date,
        project_root=root,
        checkpoint_path=checkpoint,
    )
    _write_json(output, artifact, write_observer=write_observer)
    return artifact


def _resolve_model_specs(cached_pair_fn: CachedPairFn | None) -> list[dict[str, Any]] | None:
    resolver = cached_pair_fn or _cached_sota_pair
    try:
        return resolver(gpu_indices=(0, 1), preferred_quant="Q4_K_M")
    except Exception:
        return None


def _cached_sota_pair(**kwargs: Any) -> list[dict[str, Any]] | None:  # pragma: no cover
    from carnot.inference.sota_models import cached_sota_pair

    return cached_sota_pair(**kwargs)


def _blocked_artifact(
    *,
    terminal_blocker: str,
    model_specs: Sequence[Mapping[str, Any]],
    run_date: str,
    project_root: str | Path,
    exp1396_confirmed: bool,
) -> dict[str, Any]:
    artifact = {
        "status": "complete",
        "total_fover_cases": 0,
        "certificate_extract_count": 0,
        "certificate_parse_rate": 0.0,
        "semantic_validation_pass_rate": 0.0,
        "full_pipeline_pass_rate": 0.0,
        "models_used": _model_records(model_specs),
        "terminal_blocker": terminal_blocker,
    }
    return finalize_exp1397_artifact(
        artifact,
        model_specs=model_specs,
        exp1396_artifact={"semantic_validation_improvement_measured": exp1396_confirmed},
        run_date=run_date,
        project_root=project_root,
    )


def _sota_generation_gate(
    artifact: Mapping[str, Any],
    *,
    model_specs: Sequence[Mapping[str, Any]],
    cases_evaluated: int,
) -> bool:
    if not _model_specs_include_mandated_cached_sota(model_specs):
        return False

    rows = list(artifact.get("certificate_rows") or [])
    if rows:
        mandated_live_rows = [
            row
            for row in rows
            if row.get("generation_source") == "live_sota_llamacpp"
            and row.get("model_hf_id") in MANDATED_HEADLINE_MODEL_IDS
        ]
        return len(mandated_live_rows) == cases_evaluated and cases_evaluated > 0

    evidence = artifact.get("headline_gate_evidence")
    if isinstance(evidence, Mapping):
        mandated_count = int(evidence.get("mandated_live_generation_case_count") or 0)
        return bool(evidence.get("headline_result_allowed")) and mandated_count >= cases_evaluated
    return False


def _model_specs_include_mandated_cached_sota(model_specs: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        str(spec.get("hf_id") or "") in MANDATED_HEADLINE_MODEL_IDS and bool(spec.get("model_path"))
        for spec in model_specs
    )


def _model_records(model_specs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec in model_specs:
        records.append(
            {
                "name": spec.get("name"),
                "hf_id": spec.get("hf_id"),
                "gpu": spec.get("gpu"),
                "model_path": spec.get("model_path"),
                "selected_for_generation": False,
                "headline_result_allowed": False,
            }
        )
    return records


def _dvi_checkpoint_from_exp1381(exp1381_artifact: Mapping[str, Any]) -> str | None:
    if exp1381_artifact.get("dvi_deployed") is not True:
        return None
    raw_path = exp1381_artifact.get("dvi_checkpoint_path")
    if not raw_path:
        return None
    path = Path(str(raw_path))
    return str(path) if path.exists() else None


def _honest_verdict(
    *,
    exp1396_confirmed: bool,
    semantic_rate: float,
    full_rate: float,
    sota_gate: bool,
    terminal_blocker: str | None,
    headline_result_allowed: bool,
) -> str:
    if not exp1396_confirmed:
        return "blocked_exp1396_semantic_validation_fix_not_confirmed"
    if terminal_blocker:
        return f"blocked_{terminal_blocker}"
    if semantic_rate < HEADLINE_SEMANTIC_THRESHOLD:
        return "not_headline_semantic_validation_below_0_70"
    if full_rate < HEADLINE_FULL_PIPELINE_THRESHOLD:
        return "not_headline_full_pipeline_below_0_40"
    if not sota_gate:
        return "not_headline_sota_generation_provenance_missing"
    if headline_result_allowed:
        return (
            "headline_allowed_exp1397_semantic_"
            f"{_rate_label(semantic_rate)}_full_pipeline_{_rate_label(full_rate)}"
        )
    return "not_headline_unknown_gate_failure"


def _headline_blocker_label(
    *,
    exp1396_confirmed: bool,
    semantic_rate: float,
    full_rate: float,
    sota_gate: bool,
    terminal_blocker: str | None,
) -> str:
    if not exp1396_confirmed:
        return "exp1396_semantic_validation_fix_not_confirmed"
    if terminal_blocker:
        return terminal_blocker
    if semantic_rate < HEADLINE_SEMANTIC_THRESHOLD:
        return "semantic_validation_below_0_70"
    if full_rate < HEADLINE_FULL_PIPELINE_THRESHOLD:
        return "full_pipeline_below_0_40"
    if not sota_gate:
        return "sota_generation_provenance_missing"
    return "unknown_gate_failure"


def _write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    write_observer: WriteObserver | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    if write_observer is not None:
        write_observer(path, dict(payload))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _float(value: Any) -> float:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return 0.0


def _rate_label(value: float) -> str:
    return str(round(float(value), 6)).replace(".", "_")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by the conductor for the live GPU run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--run-date", default=RUN_DATE)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--target-cases", type=int, default=TARGET_FOVER_CASES)
    args = parser.parse_args(argv)

    artifact = run_experiment(
        project_root=args.project_root,
        run_date=args.run_date,
        output_path=args.output_path,
        target_cases=args.target_cases,
    )
    print(
        json.dumps(
            {
                "status": artifact.get("status"),
                "cases_evaluated": artifact.get("cases_evaluated"),
                "certificate_parse_rate": artifact.get("certificate_parse_rate"),
                "semantic_validation_pass_rate": artifact.get("semantic_validation_pass_rate"),
                "full_pipeline_pass_rate": artifact.get("full_pipeline_pass_rate"),
                "headline_result_allowed": artifact.get("headline_result_allowed"),
                "honest_verdict": artifact.get("honest_verdict"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
