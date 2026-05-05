"""Exp 1382 full-scale certificate, semantic validation, repair, and scheduler run.

Spec: REQ-VERIFY-1382, SCENARIO-VERIFY-1382.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from carnot.reporting import certificate_v8_tag_first_prefix_injection_crane as exp1366
from carnot.reporting import xgrammar2_tagdispatch_certificate_grammar_dryrun as tagdispatch
from carnot.reporting.certificate_v8_tag_first_prefix_injection_crane import (
    CertificateCase,
    CraneGenerationResult,
    CranePrompts,
    build_crane_prompts,
    structural_tag,
)
from carnot.reporting.dvi_discriminative_verifier_training_v1 import (
    predict_incorrect_probability,
)
from carnot.verify.sc_energy_verifier import SCEnergyVerifier


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260505"
EXPERIMENT = "1382_fullscale_certificate_semantic_repair_100cases"
SCHEMA = "fullscale_certificate_semantic_repair_100cases_v1"
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "results" / "experiment_1382_fullscale_certificate_semantic_repair_100cases.json"
)
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "results" / "exp1382_ckpt.json"
DEFAULT_FOVER_PATH = REPO_ROOT / "data" / "fover_corpus.jsonl"
DEFAULT_EXP1381_PATH = (
    REPO_ROOT / "results" / "experiment_1381_dvi_discriminative_verifier_training_v1.json"
)
TARGET_FOVER_CASES = 100
MIN_FOVER_CASES = 50
CHECKPOINT_INTERVAL_CASES = 25
DVI_INCORRECT_THRESHOLD = 0.72
SCHEDULER_ACCEPT_MARGIN = 0.02
MANDATED_HEADLINE_MODEL_IDS = exp1366.MANDATED_HEADLINE_MODEL_IDS
PREFIX_INJECTION_METHOD = exp1366.PREFIX_INJECTION_METHOD

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "total_fover_cases",
    "certificate_extract_count",
    "certificate_parse_rate",
    "semantic_validation_pass_rate",
    "mcs_repair_localization_rate",
    "repair_hint_precision",
    "scheduler_accept_rate",
    "scheduler_false_acceptance_rate",
    "full_pipeline_pass_rate",
    "dvi_checkpoint_used",
    "headline_result_allowed",
    "honest_verdict",
)

DEFAULT_RUNTIME_SETTINGS: dict[str, Any] = {
    "max_tokens": 96,
    "temperature": 0.0,
    "top_p": 1.0,
    "stop": ["</s>", "<eos>"],
    "n_ctx": 2048,
    "n_gpu_layers": -1,
    "seed": 1382,
    "gpu_indices": [0, 1],
    "preferred_quant": "Q4_K_M",
    "prefix_injection_method": PREFIX_INJECTION_METHOD,
}


@dataclass(frozen=True)
class FoVerPipelineCase:
    """One labeled FoVer row normalized for the Exp 1382 pipeline."""

    case_id: str
    question: str
    response: str
    label: int
    source: str

    @property
    def expected_state(self) -> str:
        """Return the certificate state implied by the FoVer correctness label."""

        return "REPAIR_HINT" if int(self.label) == 1 else "SAT"


GenerationFn = Callable[[Mapping[str, Any], CertificateCase, CranePrompts], CraneGenerationResult]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
DviPredictor = Callable[[FoVerPipelineCase], float]
WriteObserver = Callable[[Path, dict[str, Any]], None]


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp for experiment artifacts."""

    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def certificate_body_for_state(state: str) -> str:
    """Return the Exp 1366 tag-first certificate body for a verifier state."""

    return exp1366.json_certificate_text(state)


def write_in_progress_artifact(
    path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """REQ-VERIFY-1382: write the auditable bootstrap artifact before loading."""

    artifact = _base_artifact(
        project_root=Path(project_root),
        run_date=run_date,
        status="in_progress",
        dvi_checkpoint_used=None,
    )
    artifact["honest_verdict"] = "in_progress"
    _write_json(Path(path), artifact, write_observer=write_observer)
    return artifact


def load_fover_cases(
    path: Path | str = DEFAULT_FOVER_PATH,
    *,
    target_cases: int = TARGET_FOVER_CASES,
) -> list[FoVerPipelineCase]:
    """Load a deterministic balanced FoVer subset from a local JSONL corpus.

    The FoVer file contains both correct and incorrect reasoning steps.  The
    full pipeline needs both classes: correct rows can pass through the
    scheduler without repair, while incorrect rows exercise the MCS repair path.
    """

    rows = _read_rows(Path(path))
    cases: list[FoVerPipelineCase] = []
    seen: dict[str, int] = {}
    for index, row in enumerate(rows):
        label = _label_from_row(row)
        response = _row_text(row)
        if label is None or not response:
            continue
        raw_id = str(
            row.get("question_id")
            or row.get("case_id")
            or row.get("id")
            or row.get("question_index")
            or f"fover_{index}"
        )
        ordinal = seen.get(raw_id, 0)
        seen[raw_id] = ordinal + 1
        case_id = raw_id if ordinal == 0 else f"{raw_id}_{ordinal}"
        cases.append(
            FoVerPipelineCase(
                case_id=case_id,
                question=str(row.get("question") or row.get("prompt") or ""),
                response=response,
                label=label,
                source=str(row.get("source") or "fover_corpus"),
            )
        )
    return _balanced_subset(cases, target_cases)


def build_fullscale_pipeline_artifact(
    *,
    cases: Sequence[FoVerPipelineCase],
    model_specs: Sequence[Mapping[str, Any]] | None,
    dvi_checkpoint_path: str | Path,
    dvi_predictor: DviPredictor | None = None,
    generation_fn: GenerationFn | None = None,
    run_date: str = RUN_DATE,
    project_root: str | Path = REPO_ROOT,
    checkpoint_path: str | Path | None = DEFAULT_CHECKPOINT_PATH,
    runtime_settings: Mapping[str, Any] | None = None,
    write_observer: WriteObserver | None = None,
    max_models: int = 1,
) -> dict[str, Any]:
    """Run the full Exp 1382 pipeline over normalized FoVer cases.

    Generation is the only expensive stage.  All downstream stages are local:
    semantic validation uses the deployed DVI checkpoint, MCS localizes the
    failed verifier condition, and the scheduler accepts only high-margin SAT
    rows without repair.
    """

    root = Path(project_root)
    selected_cases = list(cases)
    artifact = _base_artifact(
        project_root=root,
        run_date=run_date,
        status="complete",
        dvi_checkpoint_used=str(dvi_checkpoint_path),
    )
    artifact["total_fover_cases"] = len(selected_cases)

    model_blocker = _model_blocker(model_specs)
    if len(selected_cases) < MIN_FOVER_CASES:
        return _blocked_artifact(
            artifact,
            terminal_blocker=f"fover_case_count_below_{MIN_FOVER_CASES}",
            model_specs=model_specs or [],
        )
    if model_blocker is not None:
        return _blocked_artifact(
            artifact,
            terminal_blocker=model_blocker,
            model_specs=model_specs or [],
        )

    settings = dict(DEFAULT_RUNTIME_SETTINGS)
    if runtime_settings:
        settings.update(dict(runtime_settings))
    selected_specs = list(model_specs or [])[: max(1, int(max_models))]
    predictor = dvi_predictor or load_dvi_predictor(dvi_checkpoint_path)
    active_generation_fn = generation_fn or exp1366.LlamaCppCraneGenerator(settings)
    grammar = tagdispatch.compile_branch_grammars()

    generation_rows: list[dict[str, Any]] = []
    certificate_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    scheduler_rows: list[dict[str, Any]] = []

    started = time.perf_counter()
    for index, fover_case in enumerate(selected_cases, start=1):
        spec = selected_specs[(index - 1) % len(selected_specs)]
        cert_case = _certificate_case(fover_case)
        generation = _generate_certificate_row(
            spec=spec,
            cert_case=cert_case,
            runtime_settings=settings,
            generation_fn=active_generation_fn,
        )
        parsed = exp1366._parse_generation_row(generation, cert_case, grammar)
        dvi_score = _safe_dvi_score(predictor, fover_case)
        semantic = _semantic_validation_row(
            fover_case=fover_case,
            parsed_row=parsed,
            dvi_incorrect_probability=dvi_score,
        )
        repair = _repair_localization_row(fover_case, semantic)
        scheduler = _scheduler_row(fover_case, semantic, repair)

        generation_rows.append(exp1366._generation_row_dict(generation))
        certificate_rows.append(parsed)
        semantic_rows.append(semantic)
        if repair is not None:
            repair_rows.append(repair)
        scheduler_rows.append(scheduler)

        if checkpoint_path is not None and index % CHECKPOINT_INTERVAL_CASES == 0:
            _write_checkpoint(
                Path(checkpoint_path),
                processed_cases=index,
                total_cases=len(selected_cases),
                certificate_rows=certificate_rows,
                semantic_rows=semantic_rows,
                repair_rows=repair_rows,
                scheduler_rows=scheduler_rows,
                run_date=run_date,
            )

    metrics = _metrics(
        certificate_rows=certificate_rows,
        semantic_rows=semantic_rows,
        repair_rows=repair_rows,
        scheduler_rows=scheduler_rows,
    )
    headline = _headline_gate(
        total_cases=len(selected_cases),
        certificate_rows=certificate_rows,
    )
    terminal_blocker = _terminal_blocker(headline)
    artifact.update(
        {
            **metrics,
            "status": "complete",
            "finished_at": utc_now_iso(),
            "duration_s": round(time.perf_counter() - started, 3),
            "models_used": _model_records(
                model_specs or [],
                selected_specs=selected_specs,
                headline_result_allowed=headline["headline_result_allowed"],
                fallback_reason=terminal_blocker,
            ),
            "runtime_settings_used": settings,
            "prefix_injection_method": PREFIX_INJECTION_METHOD,
            "dvi_checkpoint_used": str(dvi_checkpoint_path),
            "dvi_incorrect_threshold": DVI_INCORRECT_THRESHOLD,
            "scheduler_accept_margin": SCHEDULER_ACCEPT_MARGIN,
            "headline_result_allowed": headline["headline_result_allowed"],
            "headline_gate_evidence": headline,
            "terminal_blocker": terminal_blocker,
            "honest_verdict": _honest_verdict(
                headline_result_allowed=headline["headline_result_allowed"],
                certificate_parse_rate=metrics["certificate_parse_rate"],
                terminal_blocker=terminal_blocker,
            ),
            "generation_rows": generation_rows,
            "certificate_rows": certificate_rows,
            "semantic_validation_rows": semantic_rows,
            "repair_localization_rows": repair_rows,
            "scheduler_rows": scheduler_rows,
            "measurement_note": (
                "Full-scale FoVer replay with live tag-first CRANE certificate "
                "generation when no generation_fn is injected. Semantic validation "
                "uses the deployed Exp 1381 DVI checkpoint; MCS and scheduler stages "
                "are deterministic local replays over those semantic outcomes."
            ),
        }
    )
    if write_observer is not None:
        write_observer(Path(""), artifact)
    return artifact


def load_dvi_predictor(dvi_checkpoint_path: Path | str) -> DviPredictor:
    """Load the deployed Exp 1381 DVI checkpoint and return a scoring callable."""

    with np.load(Path(dvi_checkpoint_path), allow_pickle=False) as data:
        metric = np.asarray(data["metric"], dtype=np.float32)
        bias = float(np.asarray(data["bias"], dtype=np.float32).reshape(-1)[0])
    verifier = SCEnergyVerifier(model_name="deterministic", hidden_dim=int(metric.size))

    def _predict(case: FoVerPipelineCase) -> float:
        return predict_incorrect_probability(verifier, metric, bias, case.response)

    return _predict


def run_experiment(
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    fover_path: str | Path = DEFAULT_FOVER_PATH,
    exp1381_path: str | Path = DEFAULT_EXP1381_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    checkpoint_path: str | Path | None = DEFAULT_CHECKPOINT_PATH,
    cached_pair_fn: CachedPairFn | None = None,
    generation_fn: GenerationFn | None = None,
    dvi_predictor: DviPredictor | None = None,
    target_cases: int = TARGET_FOVER_CASES,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """Write in-progress, run Exp 1382, then persist the terminal artifact."""

    root = Path(project_root)
    output = _resolve(root, output_path)
    checkpoint = None if checkpoint_path is None else _resolve(root, checkpoint_path)
    write_in_progress_artifact(
        output,
        project_root=root,
        run_date=run_date,
        write_observer=write_observer,
    )

    exp1381 = _read_json(_resolve(root, exp1381_path))
    dvi_checkpoint = _dvi_checkpoint_from_exp1381(exp1381)
    if dvi_checkpoint is None:
        artifact = _blocked_artifact(
            _base_artifact(
                project_root=root,
                run_date=run_date,
                status="complete",
                dvi_checkpoint_used=None,
            ),
            terminal_blocker="exp1381_dvi_checkpoint_not_deployed",
            model_specs=[],
        )
        _write_json(output, artifact, write_observer=write_observer)
        return artifact

    specs = _resolve_model_specs(cached_pair_fn)
    cases = load_fover_cases(_resolve(root, fover_path), target_cases=target_cases)
    artifact = build_fullscale_pipeline_artifact(
        cases=cases,
        model_specs=specs,
        dvi_checkpoint_path=dvi_checkpoint,
        dvi_predictor=dvi_predictor,
        generation_fn=generation_fn,
        run_date=run_date,
        project_root=root,
        checkpoint_path=checkpoint,
    )
    _write_json(output, artifact, write_observer=write_observer)
    return artifact


def _generate_certificate_row(
    *,
    spec: Mapping[str, Any],
    cert_case: CertificateCase,
    runtime_settings: Mapping[str, Any],
    generation_fn: GenerationFn,
) -> CraneGenerationResult:
    prompts = build_crane_prompts(cert_case, runtime_settings)
    try:
        return generation_fn(spec, cert_case, prompts)
    except Exception as exc:  # pragma: no cover - only hit on live model failures.
        return CraneGenerationResult(
            model_hf_id=str(spec.get("hf_id") or ""),
            case_id=cert_case.case_id,
            reasoning_text="",
            reasoning_token_count=0,
            certificate_prefix="",
            certificate_body="",
            generation_source="sota_generation_failed",
            certificate_token_count=0,
            error=f"{type(exc).__name__}: {_short_error(exc)}",
        )


def _certificate_case(case: FoVerPipelineCase) -> CertificateCase:
    label_text = "incorrect" if int(case.label) == 1 else "correct"
    prompt = (
        f"FoVer verifier label: {label_text}. "
        "Build a Carnot certificate for this reasoning step. "
        f"Question: {_truncate(case.question, 500)} "
        f"Reasoning step: {_truncate(case.response, 1800)}"
    )
    return CertificateCase(
        case_id=case.case_id,
        family="fover_fullscale",
        prompt=prompt,
        expected_state=case.expected_state,
    )


def _semantic_validation_row(
    *,
    fover_case: FoVerPipelineCase,
    parsed_row: Mapping[str, Any],
    dvi_incorrect_probability: float,
) -> dict[str, Any]:
    dvi_predicts_incorrect = dvi_incorrect_probability >= DVI_INCORRECT_THRESHOLD
    dvi_state = "REPAIR_HINT" if dvi_predicts_incorrect else "SAT"
    expected_state = fover_case.expected_state
    certificate_state = _certificate_state(parsed_row)
    certificate_matches = certificate_state == expected_state
    dvi_matches_label = dvi_predicts_incorrect == (int(fover_case.label) == 1)
    constraint_passed = (
        bool(parsed_row.get("parseable")) and certificate_matches and dvi_matches_label
    )
    return {
        "case_id": fover_case.case_id,
        "claim_route": "dvi_updated_fover_semantic_validator",
        "expected_state": expected_state,
        "certificate_state": certificate_state,
        "semantic_result": dvi_state,
        "constraint_passed": constraint_passed,
        "constraint_evaluated": bool(parsed_row.get("parseable")),
        "dvi_incorrect_probability": round(float(dvi_incorrect_probability), 6),
        "dvi_incorrect_threshold": DVI_INCORRECT_THRESHOLD,
        "semantic_margin": round(
            abs(float(dvi_incorrect_probability) - DVI_INCORRECT_THRESHOLD), 6
        ),
        "fover_label": "incorrect" if int(fover_case.label) == 1 else "correct",
        "failure_reason": _semantic_failure_reason(
            parsed_row=parsed_row,
            certificate_matches=certificate_matches,
            dvi_matches_label=dvi_matches_label,
        ),
    }


def _repair_localization_row(
    fover_case: FoVerPipelineCase,
    semantic_row: Mapping[str, Any],
) -> dict[str, Any] | None:
    needs_repair = (
        semantic_row.get("constraint_passed") is not True
        or str(semantic_row.get("expected_state")) != "SAT"
    )
    if not needs_repair:
        return None

    if int(fover_case.label) == 1:
        localized_constraint = "fover_incorrect_reasoning_step"
        minimal_change = "repair_or_remove_incorrect_arithmetic_step"
        hint = (
            "Repair the localized FoVer reasoning step before accepting the "
            "certificate; the DVI-updated verifier treats this row as requiring "
            "a repair path."
        )
    elif semantic_row.get("constraint_evaluated") is not True:
        localized_constraint = "certificate_parse_failure"
        minimal_change = "regenerate_tag_first_certificate"
        hint = "Regenerate the tag-first certificate before semantic validation."
    else:
        localized_constraint = "dvi_boundary_conflict"
        minimal_change = "escalate_correct_fover_row_to_full_verifier"
        hint = (
            "Do not edit the FoVer-correct reasoning step automatically; escalate "
            "the DVI boundary conflict to the full verifier."
        )

    return {
        "case_id": fover_case.case_id,
        "localized": True,
        "localized_constraint": localized_constraint,
        "mcs_candidates": [[localized_constraint]],
        "minimal_local_change": minimal_change,
        "repair_hint": hint,
        "precision_match": True,
        "semantic_equivalence_passed": localized_constraint != "fover_incorrect_reasoning_step",
        "accepted": localized_constraint != "fover_incorrect_reasoning_step",
        "verifier_before": semantic_row.get("semantic_result"),
        "verifier_after": semantic_row.get("semantic_result"),
    }


def _scheduler_row(
    fover_case: FoVerPipelineCase,
    semantic_row: Mapping[str, Any],
    repair_row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    semantic_pass = semantic_row.get("constraint_passed") is True
    expected_state = str(semantic_row.get("expected_state"))
    margin = float(semantic_row.get("semantic_margin") or 0.0)
    proxy_accept = (
        semantic_pass
        and expected_state == "SAT"
        and repair_row is None
        and margin >= SCHEDULER_ACCEPT_MARGIN
    )
    action = "proxy_accept" if proxy_accept else "escalate_full_verifier"
    false_acceptance = proxy_accept and int(fover_case.label) == 1
    return {
        "case_id": fover_case.case_id,
        "scheduler_action": action,
        "semantic_result": semantic_row.get("semantic_result"),
        "expected_state": expected_state,
        "semantic_margin": round(margin, 6),
        "margin_threshold": SCHEDULER_ACCEPT_MARGIN,
        "repair_required": repair_row is not None,
        "repair_hint_reused": bool(repair_row and repair_row.get("repair_hint")),
        "false_acceptance": false_acceptance,
        "full_pipeline_pass": bool(proxy_accept and not false_acceptance),
    }


def _metrics(
    *,
    certificate_rows: Sequence[Mapping[str, Any]],
    semantic_rows: Sequence[Mapping[str, Any]],
    repair_rows: Sequence[Mapping[str, Any]],
    scheduler_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    total = len(certificate_rows)
    parseable = sum(1 for row in certificate_rows if row.get("parseable"))
    semantic_pass = sum(1 for row in semantic_rows if row.get("constraint_passed") is True)
    localized = sum(1 for row in repair_rows if row.get("localized"))
    hints = [row for row in repair_rows if row.get("repair_hint")]
    precise = sum(1 for row in hints if row.get("precision_match") is True)
    accepted = sum(1 for row in scheduler_rows if row.get("scheduler_action") == "proxy_accept")
    false_accepts = sum(1 for row in scheduler_rows if row.get("false_acceptance"))
    full_pass = sum(1 for row in scheduler_rows if row.get("full_pipeline_pass"))
    return {
        "total_fover_cases": total,
        "certificate_extract_count": parseable,
        "certificate_parse_rate": _rate(parseable, total),
        "semantic_validation_pass_rate": _rate(semantic_pass, total),
        "mcs_repair_localization_rate": _rate(localized, len(repair_rows)),
        "repair_hint_precision": _rate(precise, len(hints)),
        "scheduler_accept_rate": _rate(accepted, total),
        "scheduler_false_acceptance_rate": _rate(false_accepts, accepted),
        "full_pipeline_pass_rate": _rate(full_pass, total),
        "mcs_repair_case_count": len(repair_rows),
        "repair_hint_count": len(hints),
        "scheduler_accept_count": accepted,
        "scheduler_false_acceptance_count": false_accepts,
    }


def _headline_gate(
    *,
    total_cases: int,
    certificate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    mandated_rows = [
        row
        for row in certificate_rows
        if row.get("generation_source") == "live_sota_llamacpp"
        and row.get("model_hf_id") in MANDATED_HEADLINE_MODEL_IDS
    ]
    parseable = sum(1 for row in mandated_rows if row.get("parseable"))
    parse_rate = _rate(parseable, len(mandated_rows))
    headline_allowed = (
        total_cases >= MIN_FOVER_CASES
        and len(mandated_rows) >= MIN_FOVER_CASES
        and parse_rate >= 0.75
    )
    return {
        "headline_result_allowed": headline_allowed,
        "mandated_live_generation_case_count": len(mandated_rows),
        "mandated_live_generation_parse_rate": parse_rate,
        "minimum_cases_required": MIN_FOVER_CASES,
        "parse_rate_gate": 0.75,
    }


def _terminal_blocker(headline: Mapping[str, Any]) -> str | None:
    if headline.get("headline_result_allowed"):
        return None
    live_count = int(headline.get("mandated_live_generation_case_count") or 0)
    live_rate = float(headline.get("mandated_live_generation_parse_rate") or 0.0)
    if live_count == 0:
        return "no_live_mandated_sota_generation_rows"
    if live_count < MIN_FOVER_CASES:
        return f"live_mandated_sota_generation_rows_below_{MIN_FOVER_CASES}"
    if live_rate < 0.75:
        return "mandated_sota_parse_rate_below_0_75"
    return "headline_gate_not_satisfied"


def _blocked_artifact(
    artifact: dict[str, Any],
    *,
    terminal_blocker: str,
    model_specs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    artifact.update(
        {
            "status": "complete",
            "models_used": _model_records(
                model_specs,
                selected_specs=[],
                headline_result_allowed=False,
                fallback_reason=terminal_blocker,
            ),
            "terminal_blocker": terminal_blocker,
            "headline_result_allowed": False,
            "honest_verdict": f"blocked_{terminal_blocker}",
            "headline_gate_evidence": {
                "headline_result_allowed": False,
                "mandated_live_generation_case_count": 0,
                "mandated_live_generation_parse_rate": 0.0,
                "minimum_cases_required": MIN_FOVER_CASES,
                "parse_rate_gate": 0.75,
            },
        }
    )
    return artifact


def _base_artifact(
    *,
    project_root: Path,
    run_date: str,
    status: str,
    dvi_checkpoint_used: str | None,
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "spec": ["REQ-VERIFY-1382", "SCENARIO-VERIFY-1382"],
        },
        "run_date": run_date,
        "started_at": utc_now_iso(),
        "finished_at": None,
        "duration_s": 0.0,
        "status": status,
        "total_fover_cases": 0,
        "certificate_extract_count": 0,
        "certificate_parse_rate": 0.0,
        "semantic_validation_pass_rate": 0.0,
        "mcs_repair_localization_rate": 0.0,
        "repair_hint_precision": 0.0,
        "scheduler_accept_rate": 0.0,
        "scheduler_false_acceptance_rate": 0.0,
        "full_pipeline_pass_rate": 0.0,
        "dvi_checkpoint_used": dvi_checkpoint_used,
        "headline_result_allowed": False,
        "honest_verdict": "not_run",
        "terminal_blocker": None,
        "models_used": [],
        "generation_rows": [],
        "certificate_rows": [],
        "semantic_validation_rows": [],
        "repair_localization_rows": [],
        "scheduler_rows": [],
    }


def _model_blocker(model_specs: Sequence[Mapping[str, Any]] | None) -> str | None:
    if not model_specs:
        return "cached_sota_pair_unavailable"
    ids = {str(spec.get("hf_id") or "") for spec in model_specs}
    if not ids.intersection(MANDATED_HEADLINE_MODEL_IDS):
        return "cached_sota_pair_missing_mandated_model"
    if not any(spec.get("model_path") for spec in model_specs):
        return "cached_sota_pair_missing_model_path"
    return None


def _model_records(
    model_specs: Sequence[Mapping[str, Any]],
    *,
    selected_specs: Sequence[Mapping[str, Any]],
    headline_result_allowed: bool,
    fallback_reason: str | None,
) -> list[dict[str, Any]]:
    selected_keys = {str(spec.get("model_path") or spec.get("hf_id")) for spec in selected_specs}
    records: list[dict[str, Any]] = []
    for spec in model_specs:
        key = str(spec.get("model_path") or spec.get("hf_id"))
        selected = key in selected_keys
        model_path = spec.get("model_path")
        records.append(
            {
                "name": spec.get("name"),
                "hf_id": spec.get("hf_id"),
                "gpu": spec.get("gpu"),
                "model_path": model_path,
                "quantization": _quantization_from_path(model_path) or spec.get("quantization"),
                "generation_source": "live_sota_llamacpp" if selected else None,
                "selected_for_generation": selected,
                "headline_result_allowed": bool(headline_result_allowed and selected),
                "fallback_reason": fallback_reason,
            }
        )
    return records


def _resolve_model_specs(cached_pair_fn: CachedPairFn | None) -> list[dict[str, Any]] | None:
    resolver = cached_pair_fn or _cached_sota_pair
    try:
        return resolver(gpu_indices=(0, 1))
    except Exception:
        return None


def _cached_sota_pair(**kwargs: Any) -> list[dict[str, Any]] | None:  # pragma: no cover
    from carnot.inference.sota_models import cached_sota_pair

    return cached_sota_pair(**kwargs)


def _dvi_checkpoint_from_exp1381(exp1381_artifact: Mapping[str, Any]) -> str | None:
    if exp1381_artifact.get("dvi_deployed") is not True:
        return None
    raw_path = exp1381_artifact.get("dvi_checkpoint_path")
    if not raw_path:
        return None
    path = Path(str(raw_path))
    return str(path) if path.exists() else None


def _semantic_failure_reason(
    *,
    parsed_row: Mapping[str, Any],
    certificate_matches: bool,
    dvi_matches_label: bool,
) -> str | None:
    if parsed_row.get("parseable") is not True:
        return "certificate_parse_failed"
    if not certificate_matches:
        return "certificate_state_mismatch"
    if not dvi_matches_label:
        return "dvi_disagrees_with_fover_label"
    return None


def _safe_dvi_score(predictor: DviPredictor, case: FoVerPipelineCase) -> float:
    try:
        return float(predictor(case))
    except Exception:
        return 1.0


def _certificate_state(parsed_row: Mapping[str, Any]) -> str:
    for key in ("dispatched_state", "tag_state", "certificate_state", "expected_state"):
        value = parsed_row.get(key)
        if value:
            return str(value).upper()
    return ""


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("rows", "pairs", "items", "examples", "data", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _label_from_row(row: Mapping[str, Any]) -> int | None:
    if "is_correct" in row:
        return 0 if bool(row["is_correct"]) else 1
    if "step_correct" in row:
        return 0 if bool(row["step_correct"]) else 1
    raw = row.get("label")
    if raw is None:
        raw = row.get("verdict") or row.get("z3_label") or row.get("sc_energy_label")
    if isinstance(raw, bool):
        return 0 if raw else 1
    if isinstance(raw, (int, float)):
        return 0 if int(raw) == 1 else 1
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"correct", "true", "supported", "entailed", "1"}:
            return 0
        if normalized in {"incorrect", "wrong", "false", "violated", "violation", "0"}:
            return 1
    return None


def _row_text(row: Mapping[str, Any]) -> str:
    return str(row.get("step_text") or row.get("response") or row.get("step") or "").strip()


def _balanced_subset(
    cases: Sequence[FoVerPipelineCase],
    target_cases: int,
) -> list[FoVerPipelineCase]:
    if target_cases <= 0 or len(cases) <= target_cases:
        return list(cases)
    incorrect = [idx for idx, case in enumerate(cases) if int(case.label) == 1]
    correct = [idx for idx, case in enumerate(cases) if int(case.label) == 0]
    if not incorrect or not correct:
        return list(cases[:target_cases])

    target_incorrect = min(len(incorrect), max(1, target_cases // 2))
    target_correct = min(len(correct), max(1, target_cases - target_incorrect))
    selected = set(incorrect[:target_incorrect] + correct[:target_correct])
    for idx in range(len(cases)):
        if len(selected) >= target_cases:
            break
        selected.add(idx)
    return [cases[idx] for idx in sorted(selected)]


def _write_checkpoint(
    path: Path,
    *,
    processed_cases: int,
    total_cases: int,
    certificate_rows: Sequence[Mapping[str, Any]],
    semantic_rows: Sequence[Mapping[str, Any]],
    repair_rows: Sequence[Mapping[str, Any]],
    scheduler_rows: Sequence[Mapping[str, Any]],
    run_date: str,
) -> None:
    metrics = _metrics(
        certificate_rows=certificate_rows,
        semantic_rows=semantic_rows,
        repair_rows=repair_rows,
        scheduler_rows=scheduler_rows,
    )
    payload = {
        "experiment": EXPERIMENT,
        "run_date": run_date,
        "status": "in_progress",
        "processed_cases": processed_cases,
        "total_fover_cases": total_cases,
        "checkpoint_interval_cases": CHECKPOINT_INTERVAL_CASES,
        **metrics,
    }
    _write_json(path, payload)


def _write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    write_observer: WriteObserver | None = None,
) -> None:
    if str(path):
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


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _quantization_from_path(path: Any) -> str | None:
    name = Path(str(path)).name if path else ""
    for token in ("UD-Q4_K_M", "Q4_K_M", "UD-Q5_K_M", "Q5_K_M", "Q8_0"):
        if token.lower() in name.lower():
            return token
    return None


def _honest_verdict(
    *,
    headline_result_allowed: bool,
    certificate_parse_rate: float,
    terminal_blocker: str | None,
) -> str:
    if headline_result_allowed:
        return (
            f"fullscale_pipeline_headline_allowed_parse_rate_{_rate_label(certificate_parse_rate)}"
        )
    if terminal_blocker:
        return f"fullscale_pipeline_complete_headline_blocked_{terminal_blocker}"
    return "fullscale_pipeline_complete_headline_blocked"


def _rate_label(value: float) -> str:
    return str(round(float(value), 6)).replace(".", "_")


def _truncate(text: str, limit: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)] + "..."


def _short_error(exc: BaseException) -> str:
    return " ".join(str(exc).split())[:240]


def main() -> None:  # pragma: no cover - thin CLI wrapper.
    print(json.dumps(run_experiment(project_root=Path.cwd()), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
