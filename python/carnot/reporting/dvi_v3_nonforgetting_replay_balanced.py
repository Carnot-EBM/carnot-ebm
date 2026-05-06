"""Exp 1432 DVI v3 replay-heldout nonforgetting repair.

Exp 1415 already measured a positive DVI v3 AUROC delta, but it blocked
deployment because the fixed FR-11 memory-acceptance threshold promoted a small
set of replay demotions.  This module keeps the ranking measurement intact and
repairs only the deployment decision boundary: it calibrates the inherited SECL
acceptance threshold on one deterministic replay split, then audits a held-out
replay split before writing a deployable checkpoint.

Spec: REQ-VERIFY-1432, SCENARIO-VERIFY-1432.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from carnot.reporting import dvi_discriminative_verifier_training_v1 as dvi
from carnot.reporting import dvi_v3_1508_fresh_cases as base
from carnot.reporting import fr11_self_learning_v5 as fr11
from carnot.reporting import secl_discriminative_self_calibration as secl


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_VERIFY_DIR = REPO_ROOT / "python" / "carnot" / "verify"
DEFAULT_FOVER_PATH = REPO_ROOT / "data" / "fover_corpus.jsonl"

EXP1394_FILE = base.EXP1394_FILE
EXP1395_FILE = base.EXP1395_FILE
EXP1415_FILE = "experiment_1415_dvi_v3_1508_fresh_cases.json"
OUTPUT_FILE = "experiment_1432_dvi_v3_nonforgetting_replay_balanced.json"

DEFAULT_EXP1394_PATH = DEFAULT_RESULTS_DIR / EXP1394_FILE
DEFAULT_EXP1395_PATH = DEFAULT_RESULTS_DIR / EXP1395_FILE
DEFAULT_EXP1415_PATH = DEFAULT_RESULTS_DIR / EXP1415_FILE
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE
DEFAULT_CHECKPOINT_PATH = DEFAULT_VERIFY_DIR / "dvi_v3_nonforgetting_replay_balanced.pt"

EXPERIMENT = "1432_dvi_v3_nonforgetting_replay_balanced"
SCHEMA = "dvi_v3_nonforgetting_replay_balanced_v1"
RUN_DATE = "20260506"
FRESH_VERIFIED_CASE_COUNT = base.FRESH_VERIFIED_CASE_COUNT
DVI_V2_AUROC_DELTA_BASELINE = base.DVI_V2_AUROC_DELTA_BASELINE
MIN_NONFORGETTING_RATE = 0.99
TRAINING_METHOD = "dvi_v3_exp1415_weights_with_replay_heldout_threshold_calibration"

DEFAULT_SECL_THRESHOLD_CANDIDATES = (
    fr11.SECL_CONFIDENCE_THRESHOLD,
    fr11.SECL_CONFIDENCE_THRESHOLD + 0.000001,
    0.51,
    0.53,
    0.531,
    0.532,
    0.55,
    0.60,
    0.75,
    0.90,
    1.0,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "dvi_v3_deployed",
    "dvi_v3_auroc_delta",
    "dvi_v2_auroc_delta_baseline",
    "nonforgetting_rate",
    "replay_balance_applied",
    "threshold_calibration_applied",
    "fresh_cases_used",
    "tests_run",
    "honest_verdict",
)

run_dvi_v3_training = base.run_dvi_v3_training
measure_secl_preservation = base.measure_secl_preservation


@dataclass(frozen=True)
class FailureDiagnosis:
    """Structured diagnosis for why Exp 1415 did not deploy.

    The diagnosis is intentionally conservative.  A positive AUROC delta rules
    out a broad model-update regression as the primary blocker, while a
    nonforgetting-only deployment block points to the decision threshold used
    to promote or demote replay rows.
    """

    failure_mode: str
    auroc_improved_over_v2: bool
    nonforgetting_below_gate: bool
    evidence: list[str]


@dataclass(frozen=True)
class ReplaySplit:
    """Deterministic replay split used for calibration and held-out audit."""

    calibration: list[dvi.DviCase]
    holdout: list[dvi.DviCase]


@dataclass(frozen=True)
class ThresholdCalibrationResult:
    """Selected DVI/SECL acceptance thresholds for the repaired checkpoint."""

    dvi_incorrect_threshold: float
    secl_confidence_threshold: float
    calibration_nonforgetting_rate: float
    threshold_calibration_applied: bool


def _write_json(path: Path | str, artifact: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _metadata(project_root: str | Path, run_date: str) -> dict[str, str]:
    return {"project_root": str(project_root), "run_date": run_date}


def write_in_progress_artifact(
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    dvi_v2_auroc_delta_baseline: float = DVI_V2_AUROC_DELTA_BASELINE,
) -> dict[str, Any]:
    """REQ-VERIFY-1432: write a traceable bootstrap artifact first."""

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "status": "in_progress",
            "dvi_v3_deployed": False,
            "dvi_v3_auroc_delta": None,
            "dvi_v2_auroc_delta_baseline": round(float(dvi_v2_auroc_delta_baseline), 6),
            "nonforgetting_rate": None,
            "replay_balance_applied": False,
            "threshold_calibration_applied": False,
            "fresh_cases_used": 0,
            "tests_run": [],
            "honest_verdict": "in_progress",
            "fresh_llm_inference_used": False,
            "cpu_only": True,
        },
    )


def load_json(path: Path | str) -> dict[str, Any]:
    """Load a JSON artifact as an object."""

    return base.load_json(path)


def diagnose_exp1415_failure(
    exp1415_artifact: Mapping[str, Any],
    *,
    min_nonforgetting_rate: float = MIN_NONFORGETTING_RATE,
) -> FailureDiagnosis:
    """REQ-VERIFY-1432: classify the prior DVI v3 deployment failure."""

    dvi_v3_delta = _float(exp1415_artifact.get("dvi_v3_auroc_delta"))
    dvi_v2_delta = _float(
        exp1415_artifact.get("dvi_v2_auroc_delta_baseline"),
        DVI_V2_AUROC_DELTA_BASELINE,
    )
    nonforgetting_rate = _float(exp1415_artifact.get("nonforgetting_rate"), 1.0)
    block_reasons = [str(item) for item in exp1415_artifact.get("block_reasons", [])]
    auroc_improved = dvi_v3_delta >= dvi_v2_delta
    nonforgetting_below = nonforgetting_rate < float(min_nonforgetting_rate)

    evidence = [
        f"dvi_v3_auroc_delta={dvi_v3_delta:.6f}",
        f"dvi_v2_auroc_delta_baseline={dvi_v2_delta:.6f}",
        f"nonforgetting_rate={nonforgetting_rate:.6f}",
    ]
    evidence.extend(block_reasons)

    if auroc_improved and nonforgetting_below and "nonforgetting_below_gate" in block_reasons:
        mode = "thresholding"
    elif nonforgetting_below and not auroc_improved:
        mode = "model_update_drift"
    elif nonforgetting_below:
        mode = "sampling_imbalance"
    else:
        mode = "no_failure_detected"
    return FailureDiagnosis(
        failure_mode=mode,
        auroc_improved_over_v2=bool(auroc_improved),
        nonforgetting_below_gate=bool(nonforgetting_below),
        evidence=evidence,
    )


def split_replay_cases(
    replay_cases: Sequence[dvi.DviCase],
) -> ReplaySplit:
    """Split Exp 1395 replay demotions into calibration and held-out rows."""

    cases = list(replay_cases)
    if not cases:
        return ReplaySplit(calibration=[], holdout=[])
    calibration = [case for index, case in enumerate(cases) if index % 2 == 0]
    holdout = [case for index, case in enumerate(cases) if index % 2 == 1]
    if not holdout:
        holdout = list(calibration)
    return ReplaySplit(calibration=calibration, holdout=holdout)


def measure_nonforgetting_rate(
    *,
    replay_cases: Sequence[dvi.DviCase],
    metric: np.ndarray,
    bias: float,
    confidence_head: secl.HistogramECEConfidenceHead,
    dvi_incorrect_threshold: float,
    secl_confidence_threshold: float,
) -> float:
    """Measure replay demotions preserved under the selected thresholds."""

    return base.measure_nonforgetting_rate(
        replay_cases=replay_cases,
        metric=metric,
        bias=bias,
        confidence_head=confidence_head,
        incorrect_threshold=dvi_incorrect_threshold,
        secl_confidence_threshold=secl_confidence_threshold,
    )


def calibrate_threshold_for_nonforgetting(
    *,
    replay_cases: Sequence[dvi.DviCase],
    metric: np.ndarray,
    bias: float,
    confidence_head: secl.HistogramECEConfidenceHead,
    base_dvi_threshold: float = fr11.DVI_INCORRECT_THRESHOLD,
    base_secl_threshold: float = fr11.SECL_CONFIDENCE_THRESHOLD,
    candidate_secl_thresholds: Sequence[float] = DEFAULT_SECL_THRESHOLD_CANDIDATES,
    min_nonforgetting_rate: float = MIN_NONFORGETTING_RATE,
) -> ThresholdCalibrationResult:
    """Choose the smallest bounded SECL threshold that clears replay calibration.

    AUROC is a ranking metric, so calibrating the memory-acceptance threshold
    does not alter the Exp 1415 rank-order evidence.  The calibration only
    controls whether a replay demotion is promoted back into memory.
    """

    candidates = sorted(
        {
            min(1.0, max(0.0, float(threshold)))
            for threshold in (*candidate_secl_thresholds, float(base_secl_threshold))
        }
    )
    best_threshold = float(base_secl_threshold)
    best_rate = measure_nonforgetting_rate(
        replay_cases=replay_cases,
        metric=metric,
        bias=bias,
        confidence_head=confidence_head,
        dvi_incorrect_threshold=base_dvi_threshold,
        secl_confidence_threshold=best_threshold,
    )
    for threshold in candidates:
        rate = measure_nonforgetting_rate(
            replay_cases=replay_cases,
            metric=metric,
            bias=bias,
            confidence_head=confidence_head,
            dvi_incorrect_threshold=base_dvi_threshold,
            secl_confidence_threshold=threshold,
        )
        if rate >= float(min_nonforgetting_rate):
            return ThresholdCalibrationResult(
                dvi_incorrect_threshold=float(base_dvi_threshold),
                secl_confidence_threshold=float(threshold),
                calibration_nonforgetting_rate=float(rate),
                threshold_calibration_applied=not _same_threshold(threshold, base_secl_threshold),
            )
        if rate > best_rate:
            best_rate = rate
            best_threshold = threshold
    return ThresholdCalibrationResult(
        dvi_incorrect_threshold=float(base_dvi_threshold),
        secl_confidence_threshold=float(best_threshold),
        calibration_nonforgetting_rate=float(best_rate),
        threshold_calibration_applied=not _same_threshold(best_threshold, base_secl_threshold),
    )


def save_repaired_checkpoint(
    path: Path | str,
    *,
    training_result: base.DviV3TrainingResult,
    confidence_head: secl.HistogramECEConfidenceHead,
    calibration: ThresholdCalibrationResult,
    fresh_cases_used: int,
    replay_holdout_cases_used: int,
) -> bool:
    """Persist DVI v3 weights plus the calibrated acceptance thresholds."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        np.savez(
            handle,
            metric=np.asarray(training_result.metric, dtype=np.float32),
            bias=np.asarray([training_result.bias], dtype=np.float32),
            loss_history=np.asarray(training_result.loss_history, dtype=np.float32),
            secl_bin_values=np.asarray(confidence_head.bin_values, dtype=np.float32),
            secl_global_value=np.asarray([confidence_head.global_value], dtype=np.float32),
            secl_n_bins=np.asarray([confidence_head.n_bins], dtype=np.int32),
            dvi_incorrect_threshold=np.asarray(
                [calibration.dvi_incorrect_threshold],
                dtype=np.float32,
            ),
            secl_confidence_threshold=np.asarray(
                [calibration.secl_confidence_threshold],
                dtype=np.float32,
            ),
            fresh_cases_used=np.asarray([fresh_cases_used], dtype=np.int32),
            replay_holdout_cases_used=np.asarray([replay_holdout_cases_used], dtype=np.int32),
            training_method=np.asarray([TRAINING_METHOD]),
            source_checkpoint_path=np.asarray([training_result.source_checkpoint_path]),
        )
    return destination.exists()


def build_artifact(
    *,
    diagnosis: FailureDiagnosis,
    fresh_cases_used: int,
    replay_calibration_cases_used: int,
    replay_holdout_cases_used: int,
    dvi_v2_auroc_delta_baseline: float,
    training_result: base.DviV3TrainingResult,
    calibration: ThresholdCalibrationResult,
    nonforgetting_rate: float,
    secl_preservation: base.SECLPreservationResult,
    deployed: bool,
    checkpoint_path: Path | str,
    source_checkpoint_path: Path | str,
    started_at: str,
    duration_s: float,
    heldout_cases_used: int,
    tests_run: Sequence[str],
    block_reasons: Sequence[str],
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Build and validate the terminal Exp 1432 artifact."""

    delta = round(float(training_result.auroc_delta), 6)
    reasons = list(block_reasons)
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "started_at": started_at,
        "finished_at": datetime.now(tz=UTC).isoformat(),
        "duration_s": round(float(duration_s), 3),
        "status": "complete" if deployed else "blocked",
        "spec": ["REQ-VERIFY-1432", "SCENARIO-VERIFY-1432"],
        "source_artifacts": [
            f"results/{EXP1415_FILE}",
            f"results/{EXP1394_FILE}",
            f"results/{EXP1395_FILE}",
            "data/fover_corpus.jsonl",
        ],
        "source_artifact_aliases_missing": [
            "results/experiment_1394_dvi_v2_secl_59_cases.json",
            "results/experiment_1395_fr11_self_learning_v5_1508_cases.json",
        ],
        "source_checkpoint_path": str(source_checkpoint_path),
        "failure_diagnosis": diagnosis.failure_mode,
        "failure_diagnosis_evidence": list(diagnosis.evidence),
        "fresh_cases_used": int(fresh_cases_used),
        "fresh_verified_cases_used": int(fresh_cases_used),
        "fresh_case_source": "exp1395_memory_updates_promoted_dvi_v2_fover",
        "replay_balance_applied": replay_calibration_cases_used > 0
        and replay_holdout_cases_used > 0,
        "replay_calibration_cases_used": int(replay_calibration_cases_used),
        "replay_holdout_cases_used": int(replay_holdout_cases_used),
        "fover_heldout_cases_used": int(heldout_cases_used),
        "training_method": TRAINING_METHOD,
        "training_loss_history": [round(float(loss), 6) for loss in training_result.loss_history],
        "dvi_v2_auroc_delta_baseline": round(float(dvi_v2_auroc_delta_baseline), 6),
        "dvi_v3_baseline_auroc": round(float(training_result.baseline_auroc), 6),
        "dvi_v3_trained_auroc": round(float(training_result.trained_auroc), 6),
        "dvi_v3_auroc_delta": delta,
        "dvi_v3_auroc_nonregression_gate": delta >= float(dvi_v2_auroc_delta_baseline),
        "base_dvi_incorrect_threshold": fr11.DVI_INCORRECT_THRESHOLD,
        "base_secl_confidence_threshold": fr11.SECL_CONFIDENCE_THRESHOLD,
        "calibrated_dvi_incorrect_threshold": round(
            float(calibration.dvi_incorrect_threshold),
            6,
        ),
        "calibrated_secl_confidence_threshold": round(
            float(calibration.secl_confidence_threshold),
            6,
        ),
        "calibration_nonforgetting_rate": round(
            float(calibration.calibration_nonforgetting_rate),
            6,
        ),
        "nonforgetting_rate": round(float(nonforgetting_rate), 6),
        "min_nonforgetting_rate": MIN_NONFORGETTING_RATE,
        "threshold_calibration_applied": bool(calibration.threshold_calibration_applied),
        "secl_ece_before": round(float(secl_preservation.ece_before), 6),
        "secl_ece_after": round(float(secl_preservation.ece_after), 6),
        "secl_ece_reduction_pct_v3": round(float(secl_preservation.ece_reduction_pct), 6),
        "secl_ece_reduction_pct_preserved": bool(secl_preservation.preserved),
        "dvi_v3_deployed": bool(deployed),
        "dvi_v3_checkpoint_path": str(checkpoint_path) if deployed else None,
        "block_reasons": reasons,
        "tests_run": list(tests_run),
        "fresh_llm_inference_used": False,
        "cpu_only": True,
        "honest_verdict": _honest_verdict(deployed, reasons),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """REQ-VERIFY-1432: enforce required fields and deployment invariants."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] not in {"in_progress", "complete", "blocked"}:
        raise AssertionError(f"unsupported status: {artifact['status']}")
    if not isinstance(artifact["tests_run"], list):
        raise AssertionError("tests_run must be a list of command strings")
    if artifact["dvi_v3_deployed"]:
        path = artifact.get("dvi_v3_checkpoint_path")
        if not path or not Path(str(path)).exists():
            raise AssertionError("dvi_v3_deployed requires an existing checkpoint path")
        if float(artifact["nonforgetting_rate"]) < MIN_NONFORGETTING_RATE:
            raise AssertionError("dvi_v3_deployed requires held-out nonforgetting gate")
        if float(artifact["dvi_v3_auroc_delta"]) < float(artifact["dvi_v2_auroc_delta_baseline"]):
            raise AssertionError("dvi_v3_deployed requires AUROC nonregression")
    elif artifact["status"] == "blocked" and artifact.get("dvi_v3_checkpoint_path") is not None:
        raise AssertionError("blocked DVI v3 artifacts must not expose a deployed checkpoint")


def run(
    *,
    exp1394_path: Path | str = DEFAULT_EXP1394_PATH,
    exp1395_path: Path | str = DEFAULT_EXP1395_PATH,
    exp1415_path: Path | str = DEFAULT_EXP1415_PATH,
    fover_path: Path | str = DEFAULT_FOVER_PATH,
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    expected_fresh_count: int = FRESH_VERIFIED_CASE_COUNT,
    min_nonforgetting_rate: float = MIN_NONFORGETTING_RATE,
    candidate_secl_thresholds: Sequence[float] = DEFAULT_SECL_THRESHOLD_CANDIDATES,
    tests_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run Exp 1432 and deploy only when AUROC and held-out replay gates pass."""

    started_at = datetime.now(tz=UTC).isoformat()
    t0 = time.perf_counter()
    write_in_progress_artifact(out_path, project_root=project_root, run_date=run_date)

    exp1394 = load_json(exp1394_path)
    baseline_delta = _float(
        exp1394.get("dvi_v2_auroc_delta"),
        DVI_V2_AUROC_DELTA_BASELINE,
    )
    write_in_progress_artifact(
        out_path,
        project_root=project_root,
        run_date=run_date,
        dvi_v2_auroc_delta_baseline=baseline_delta,
    )
    exp1395 = load_json(exp1395_path)
    exp1415 = load_json(exp1415_path)
    diagnosis = diagnose_exp1415_failure(
        exp1415,
        min_nonforgetting_rate=min_nonforgetting_rate,
    )
    activation = fr11.activate_dvi_v2_checkpoint(exp1394)
    if not activation.active or activation.state is None:
        return _write_json(
            out_path,
            _blocked_without_training(
                project_root=project_root,
                run_date=run_date,
                started_at=started_at,
                duration_s=time.perf_counter() - t0,
                dvi_v2_auroc_delta_baseline=baseline_delta,
                tests_run=list(tests_run or []),
                block_reason=activation.blocker or "dvi_v2_checkpoint_inactive",
                diagnosis=diagnosis,
            ),
        )

    rows = dvi.load_jsonl_rows(fover_path)
    fresh_cases = base.load_fresh_verified_cases(
        exp1395,
        rows,
        expected_count=expected_fresh_count,
    )
    replay_cases = base.load_replay_cases(exp1395, rows)
    replay_split = split_replay_cases(replay_cases)
    _, holdout_rows = dvi.split_fover_rows(rows)
    confidence_head = secl.HistogramECEConfidenceHead(
        bin_values=np.asarray(activation.state.secl_bin_values, dtype=np.float64),
        global_value=float(activation.state.secl_global_value),
        n_bins=int(activation.state.secl_n_bins),
    )
    source_checkpoint = Path(activation.state.checkpoint_path)

    training_result = run_dvi_v3_training(
        fresh_cases=fresh_cases,
        holdout_rows=holdout_rows,
        source_checkpoint_path=source_checkpoint,
        n_epochs=base.N_EPOCHS,
    )
    calibration = calibrate_threshold_for_nonforgetting(
        replay_cases=replay_split.calibration,
        metric=training_result.metric,
        bias=training_result.bias,
        confidence_head=confidence_head,
        base_dvi_threshold=fr11.DVI_INCORRECT_THRESHOLD,
        base_secl_threshold=fr11.SECL_CONFIDENCE_THRESHOLD,
        candidate_secl_thresholds=candidate_secl_thresholds,
        min_nonforgetting_rate=min_nonforgetting_rate,
    )
    nonforgetting_rate = measure_nonforgetting_rate(
        replay_cases=replay_split.holdout,
        metric=training_result.metric,
        bias=training_result.bias,
        confidence_head=confidence_head,
        dvi_incorrect_threshold=calibration.dvi_incorrect_threshold,
        secl_confidence_threshold=calibration.secl_confidence_threshold,
    )
    secl_preservation = measure_secl_preservation(
        metric=training_result.metric,
        bias=training_result.bias,
        confidence_head=confidence_head,
        holdout_rows=holdout_rows,
        v2_ece_reduction_pct=_float(exp1394.get("secl_ece_reduction_pct")),
    )
    reasons = _deployment_block_reasons(
        dvi_v3_delta=training_result.auroc_delta,
        dvi_v2_delta_baseline=baseline_delta,
        nonforgetting_rate=nonforgetting_rate,
        min_nonforgetting_rate=min_nonforgetting_rate,
    )
    deployed = not reasons
    if deployed:
        deployed = save_repaired_checkpoint(
            checkpoint_path,
            training_result=training_result,
            confidence_head=confidence_head,
            calibration=calibration,
            fresh_cases_used=len(fresh_cases),
            replay_holdout_cases_used=len(replay_split.holdout),
        )
        if not deployed:
            reasons.append("dvi_v3_checkpoint_write_failed")

    artifact = build_artifact(
        diagnosis=diagnosis,
        fresh_cases_used=len(fresh_cases),
        replay_calibration_cases_used=len(replay_split.calibration),
        replay_holdout_cases_used=len(replay_split.holdout),
        dvi_v2_auroc_delta_baseline=baseline_delta,
        training_result=training_result,
        calibration=calibration,
        nonforgetting_rate=nonforgetting_rate,
        secl_preservation=secl_preservation,
        deployed=deployed and not reasons,
        checkpoint_path=checkpoint_path,
        source_checkpoint_path=source_checkpoint,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
        heldout_cases_used=len(holdout_rows),
        tests_run=list(tests_run or []),
        block_reasons=reasons,
        project_root=project_root,
        run_date=run_date,
    )
    return _write_json(out_path, artifact)


def _blocked_without_training(
    *,
    project_root: str | Path,
    run_date: str,
    started_at: str,
    duration_s: float,
    dvi_v2_auroc_delta_baseline: float,
    tests_run: Sequence[str],
    block_reason: str,
    diagnosis: FailureDiagnosis,
) -> dict[str, Any]:
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "started_at": started_at,
        "finished_at": datetime.now(tz=UTC).isoformat(),
        "duration_s": round(float(duration_s), 3),
        "status": "blocked",
        "spec": ["REQ-VERIFY-1432", "SCENARIO-VERIFY-1432"],
        "dvi_v3_deployed": False,
        "dvi_v3_auroc_delta": None,
        "dvi_v2_auroc_delta_baseline": round(float(dvi_v2_auroc_delta_baseline), 6),
        "nonforgetting_rate": None,
        "replay_balance_applied": False,
        "threshold_calibration_applied": False,
        "fresh_cases_used": 0,
        "failure_diagnosis": diagnosis.failure_mode,
        "failure_diagnosis_evidence": list(diagnosis.evidence),
        "dvi_v3_checkpoint_path": None,
        "block_reasons": [block_reason],
        "tests_run": list(tests_run),
        "fresh_llm_inference_used": False,
        "cpu_only": True,
        "honest_verdict": _honest_verdict(False, [block_reason]),
    }
    validate_artifact(artifact)
    return artifact


def _deployment_block_reasons(
    *,
    dvi_v3_delta: float,
    dvi_v2_delta_baseline: float,
    nonforgetting_rate: float,
    min_nonforgetting_rate: float,
) -> list[str]:
    reasons: list[str] = []
    if float(dvi_v3_delta) < float(dvi_v2_delta_baseline):
        reasons.append("dvi_v3_delta_below_dvi_v2_baseline")
    if float(nonforgetting_rate) < float(min_nonforgetting_rate):
        reasons.append("nonforgetting_below_gate")
    return reasons


def _honest_verdict(deployed: bool, block_reasons: Sequence[str]) -> str:
    if deployed:
        return "dvi_v3_deployed_replay_heldout_threshold_calibrated"
    if block_reasons:
        return "dvi_v3_blocked_" + "_and_".join(str(reason) for reason in block_reasons)
    return "dvi_v3_blocked_unknown_reason"


def _same_threshold(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= 1e-12


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), indent=2, sort_keys=True))
