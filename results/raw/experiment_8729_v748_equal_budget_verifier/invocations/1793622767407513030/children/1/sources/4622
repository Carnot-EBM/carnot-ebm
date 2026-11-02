"""Exp 1394 DVI v2 plus SECL combined verifier deployment.

This experiment reuses the deployed Exp 1381 DVI checkpoint as the starting
boundary, trains it on the larger Exp 1388 fresh verified set, then attaches the
Exp 1386 SECL histogram confidence head.  The checkpoint remains CPU-readable:
it is a NumPy ``.npz`` payload written to the requested ``.pt`` path so the
existing lightweight verifier loaders can inspect it without PyTorch.

Spec: REQ-VERIFY-1394, SCENARIO-VERIFY-1394.
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
from carnot.reporting import secl_discriminative_self_calibration as secl
from carnot.verify.sc_energy_verifier import SCEnergyVerifier


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_VERIFY_DIR = REPO_ROOT / "python" / "carnot" / "verify"
DEFAULT_FOVER_PATH = REPO_ROOT / "data" / "fover_corpus.jsonl"

EXP1381_FILE = "experiment_1381_dvi_discriminative_verifier_training_v1.json"
EXP1382_FILE = "experiment_1382_fullscale_certificate_semantic_repair_100cases.json"
EXP1388_FILE = "experiment_1388_fr11_self_learning_v4_dvi_grpo_integration.json"
OUTPUT_FILE = "experiment_1394_dvi_v2_secl_combined.json"

DEFAULT_EXP1381_PATH = DEFAULT_RESULTS_DIR / EXP1381_FILE
DEFAULT_EXP1382_PATH = DEFAULT_RESULTS_DIR / EXP1382_FILE
DEFAULT_EXP1388_PATH = DEFAULT_RESULTS_DIR / EXP1388_FILE
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE
DEFAULT_CHECKPOINT_PATH = DEFAULT_VERIFY_DIR / "dvi_v2_secl_combined_checkpoint.pt"

EXPERIMENT = "1394_dvi_v2_secl_combined"
SCHEMA = "dvi_v2_secl_combined_v1"
RUN_DATE = "20260506"
FRESH_CASE_COUNT = 59
N_EPOCHS = 20
TRAINING_METHOD = dvi.TRAINING_METHOD
SECL_METHOD = secl.TRAINING_METHOD

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "fresh_cases_used",
    "dvi_v2_baseline_auroc",
    "dvi_v2_trained_auroc",
    "dvi_v2_auroc_delta",
    "secl_ece_before",
    "secl_ece_after",
    "secl_ece_reduction_pct",
    "dvi_v2_deployed",
    "checkpoint_path",
    "honest_verdict",
)


@dataclass(frozen=True)
class SECLMeasurement:
    """Measured calibration outcome for the DVI v2 checkpoint.

    DVI v2 changes the discrimination boundary, so SECL must be measured
    against that trained boundary instead of reusing Exp 1386's old ECE values.
    The confidence head is returned with the metrics so the exact calibration
    state can be persisted into the combined checkpoint.
    """

    ece_before: float
    ece_after: float
    ece_reduction_pct: float
    confidence_head: secl.HistogramECEConfidenceHead
    calibration_cases_used: int
    negative_cases_used: int
    heldout_cases_used: int
    discriminative_signal_correlation: float


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
) -> dict[str, Any]:
    """REQ-VERIFY-1394: make Exp 1394 visible before any source loading."""

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "status": "in_progress",
            "fresh_cases_used": 0,
            "dvi_v2_baseline_auroc": None,
            "dvi_v2_trained_auroc": None,
            "dvi_v2_auroc_delta": None,
            "secl_ece_before": None,
            "secl_ece_after": None,
            "secl_ece_reduction_pct": None,
            "dvi_v2_deployed": False,
            "checkpoint_path": str(DEFAULT_CHECKPOINT_PATH),
            "honest_verdict": "in_progress",
            "fresh_llm_inference_used": False,
            "cpu_only": True,
        },
    )


def load_json(path: Path | str) -> dict[str, Any]:
    """Load a JSON artifact as a mapping and reject non-object payloads."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return payload


def fresh_case_ids_from_exp1388(exp1388_artifact: Mapping[str, Any]) -> list[str]:
    """REQ-VERIFY-1394: return the 59 DVI-only Exp 1382 promoted case IDs."""

    promoted = exp1388_artifact.get("memory_updates", {}).get("promoted", [])
    if not isinstance(promoted, Sequence) or isinstance(promoted, (str, bytes)):
        raise ValueError("Exp 1388 memory_updates.promoted must be a list")

    case_ids: list[str] = []
    for item in promoted:
        value = str(item)
        prefix = "dvi:exp1382:"
        if value.startswith(prefix):
            case_ids.append(value[len(prefix) :])

    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Exp 1388 DVI-only promoted IDs contain duplicates")

    expected = int(exp1388_artifact.get("fresh_verified_sample_count", FRESH_CASE_COUNT))
    if len(case_ids) != expected or len(case_ids) != FRESH_CASE_COUNT:
        raise ValueError(
            "Exp 1388 fresh DVI-only case count mismatch: "
            f"ids={len(case_ids)} expected={expected} required={FRESH_CASE_COUNT}"
        )
    return case_ids


def load_fresh_dvi_cases(
    exp1388_artifact: Mapping[str, Any],
    exp1382_artifact: Mapping[str, Any],
) -> list[dvi.DviCase]:
    """Load fresh verified DVI positives by joining Exp 1388 IDs to Exp 1382 rows."""

    ids = fresh_case_ids_from_exp1388(exp1388_artifact)
    rows_by_id = {
        str(row.get("case_id")): row
        for row in _rows(exp1382_artifact, ("semantic_validation_rows",))
    }

    cases: list[dvi.DviCase] = []
    for case_id in ids:
        row = rows_by_id.get(case_id)
        if row is None:
            raise ValueError(f"Exp 1382 semantic row missing for fresh case {case_id}")
        if row.get("constraint_passed") is not True:
            raise ValueError(f"Fresh case {case_id} is not constraint-passed in Exp 1382")
        cases.append(
            dvi.DviCase(
                case_id=case_id,
                text=fresh_case_text(row),
                label=0,
                source="exp1388_dvi_only_replay_exp1382",
            )
        )

    if len(cases) != FRESH_CASE_COUNT:
        raise ValueError(f"DVI v2 requires {FRESH_CASE_COUNT} fresh cases, got {len(cases)}")
    return cases


def fresh_case_text(row: Mapping[str, Any]) -> str:
    """Serialize semantic-validation evidence into verifier-visible text."""

    parts = [
        f"case_id={row.get('case_id')}",
        f"semantic_result={row.get('semantic_result')}",
        f"certificate_state={row.get('certificate_state')}",
        f"expected_state={row.get('expected_state')}",
        f"claim_route={row.get('claim_route')}",
        f"constraint_passed={row.get('constraint_passed')}",
        f"fover_label={row.get('fover_label')}",
        f"semantic_margin={row.get('semantic_margin')}",
        f"dvi_incorrect_probability={row.get('dvi_incorrect_probability')}",
    ]
    return " | ".join(part for part in parts if part and not part.endswith("None"))


def deployed_exp1381_checkpoint(exp1381_artifact: Mapping[str, Any]) -> Path:
    """Return the deployed Exp 1381 checkpoint path or raise a clear blocker."""

    if exp1381_artifact.get("dvi_deployed") is not True:
        raise ValueError("Exp 1381 artifact does not mark DVI as deployed")
    raw_path = exp1381_artifact.get("dvi_checkpoint_path")
    if not raw_path:
        raise ValueError("Exp 1381 artifact has no dvi_checkpoint_path")
    path = Path(str(raw_path))
    if not path.exists():
        raise ValueError(f"Exp 1381 DVI checkpoint does not exist: {path}")
    return path


def measure_secl_for_dvi_v2(
    *,
    checkpoint_path: Path | str,
    fresh_cases: Sequence[dvi.DviCase],
    train_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
) -> SECLMeasurement:
    """Apply the Exp 1386 SECL histogram recipe to the DVI v2 checkpoint."""

    state = secl.load_verifier_state(checkpoint_path)
    verifier = SCEnergyVerifier(model_name="deterministic", hidden_dim=int(state.metric.size))
    positive_cases = [
        secl.SECLCase(
            case_id=case.case_id,
            text=case.text,
            discriminative_signal=1.0,
            source=case.source,
        )
        for case in fresh_cases
    ]
    negative_cases = secl.select_negative_cases(train_rows, count=len(positive_cases))
    calibration_cases = [*positive_cases, *negative_cases]

    calibration_raw = [
        secl.raw_correct_probability(verifier, state, case.text) for case in calibration_cases
    ]
    calibration_signal = [case.discriminative_signal for case in calibration_cases]
    confidence_head = secl.train_ece_confidence_head(calibration_raw, calibration_signal)

    holdout_labels: list[int] = []
    raw_probs: list[float] = []
    for row in holdout_rows:
        label = secl.row_is_correct(row)
        text = secl.row_text(row)
        if label is None or not text:
            continue
        holdout_labels.append(1 if label else 0)
        raw_probs.append(secl.raw_correct_probability(verifier, state, text))

    calibrated_probs = confidence_head.predict(raw_probs)
    ece_before = secl.expected_calibration_error(holdout_labels, raw_probs)
    ece_after = secl.expected_calibration_error(holdout_labels, calibrated_probs)
    binary_signal = [1.0 if prob >= 0.5 else 0.0 for prob in raw_probs]
    signal_correlation = secl.pearson_correlation(binary_signal, holdout_labels)
    reduction = _ece_reduction_pct(ece_before, ece_after)
    return SECLMeasurement(
        ece_before=ece_before,
        ece_after=ece_after,
        ece_reduction_pct=reduction,
        confidence_head=confidence_head,
        calibration_cases_used=len(calibration_cases),
        negative_cases_used=len(negative_cases),
        heldout_cases_used=len(holdout_labels),
        discriminative_signal_correlation=signal_correlation,
    )


def save_combined_checkpoint(
    path: Path | str,
    *,
    dvi_result: dvi.DviTrainingResult,
    secl_measurement: SECLMeasurement,
    fresh_cases_used: int,
    source_checkpoint_path: Path | str,
) -> bool:
    """Persist DVI v2 weights plus the SECL confidence head in one checkpoint."""

    state = secl.load_verifier_state(dvi_result.checkpoint_path)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        np.savez(
            handle,
            metric=np.asarray(state.metric, dtype=np.float32),
            bias=np.asarray([state.bias], dtype=np.float32),
            loss_history=np.asarray(dvi_result.loss_history, dtype=np.float32),
            secl_bin_values=np.asarray(
                secl_measurement.confidence_head.bin_values,
                dtype=np.float32,
            ),
            secl_global_value=np.asarray(
                [secl_measurement.confidence_head.global_value],
                dtype=np.float32,
            ),
            secl_n_bins=np.asarray([secl_measurement.confidence_head.n_bins], dtype=np.int32),
            fresh_cases_used=np.asarray([fresh_cases_used], dtype=np.int32),
            training_method=np.asarray([TRAINING_METHOD]),
            secl_training_method=np.asarray([SECL_METHOD]),
            source_checkpoint_path=np.asarray([str(source_checkpoint_path)]),
        )
    return destination.exists()


def build_artifact(
    *,
    fresh_cases_used: int,
    negative_cases_used: int,
    dvi_result: dvi.DviTrainingResult,
    secl_measurement: SECLMeasurement,
    dvi_v2_deployed: bool,
    checkpoint_path: Path | str,
    source_checkpoint_path: Path | str,
    started_at: str,
    duration_s: float,
    train_rows_used: int,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Build and validate the terminal Exp 1394 artifact."""

    baseline = round(float(dvi_result.baseline_auroc), 6)
    trained = round(float(dvi_result.trained_auroc), 6)
    delta = round(trained - baseline, 6)
    ece_before = round(float(secl_measurement.ece_before), 6)
    ece_after = round(float(secl_measurement.ece_after), 6)
    ece_reduction = round(float(secl_measurement.ece_reduction_pct), 6)
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "started_at": started_at,
        "finished_at": datetime.now(tz=UTC).isoformat(),
        "duration_s": round(float(duration_s), 3),
        "status": "complete",
        "spec": ["REQ-VERIFY-1394", "SCENARIO-VERIFY-1394"],
        "source_artifacts": [
            f"results/{EXP1381_FILE}",
            f"results/{EXP1382_FILE}",
            f"results/{EXP1388_FILE}",
        ],
        "source_checkpoint_path": str(source_checkpoint_path),
        "fresh_case_source": "exp1388_dvi_only_replay_exp1382_promoted",
        "fresh_cases_used": int(fresh_cases_used),
        "negative_case_source": "fover_incorrect_reasoning_steps",
        "negative_cases_used": int(negative_cases_used),
        "negative_ratio_target": dvi.NEGATIVE_RATIO,
        "training_method": TRAINING_METHOD,
        "epochs_run": len(dvi_result.loss_history),
        "learning_rate": dvi.LEARNING_RATE,
        "l2_weight_decay": dvi.L2_WEIGHT_DECAY,
        "training_loss_history": [round(float(loss), 6) for loss in dvi_result.loss_history],
        "fover_split_seed": dvi.FOVER_SPLIT_SEED,
        "fover_train_rows_used": int(train_rows_used),
        "fover_heldout_cases_used": int(secl_measurement.heldout_cases_used),
        "dvi_v2_baseline_auroc": baseline,
        "dvi_v2_trained_auroc": trained,
        "dvi_v2_auroc_delta": delta,
        "secl_training_method": SECL_METHOD,
        "secl_calibration_cases_used": int(secl_measurement.calibration_cases_used),
        "secl_negative_cases_used": int(secl_measurement.negative_cases_used),
        "secl_confidence_head": secl_measurement.confidence_head.to_dict(),
        "secl_ece_before": ece_before,
        "secl_ece_after": ece_after,
        "secl_ece_reduction_pct": ece_reduction,
        "secl_discriminative_signal_correlation": round(
            float(secl_measurement.discriminative_signal_correlation),
            6,
        ),
        "dvi_v2_deployed": bool(dvi_v2_deployed),
        "checkpoint_path": str(checkpoint_path),
        "fresh_llm_inference_used": False,
        "cpu_only": True,
        "honest_verdict": _honest_verdict(delta, ece_reduction, dvi_v2_deployed),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """SCENARIO-VERIFY-1394: enforce required fields and deployment invariants."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] == "complete":
        if int(artifact["fresh_cases_used"]) != FRESH_CASE_COUNT:
            raise AssertionError(f"fresh_cases_used must be {FRESH_CASE_COUNT}")
        if artifact["dvi_v2_auroc_delta"] is None:
            raise AssertionError("complete DVI v2 artifact requires AUROC delta")
        if artifact["secl_ece_before"] is None or artifact["secl_ece_after"] is None:
            raise AssertionError("complete DVI v2 artifact requires SECL ECE metrics")
        if artifact["dvi_v2_deployed"] and not Path(str(artifact["checkpoint_path"])).exists():
            raise AssertionError("dvi_v2_deployed requires checkpoint_path to exist")


def run(
    *,
    exp1381_path: Path | str = DEFAULT_EXP1381_PATH,
    exp1382_path: Path | str = DEFAULT_EXP1382_PATH,
    exp1388_path: Path | str = DEFAULT_EXP1388_PATH,
    fover_path: Path | str = DEFAULT_FOVER_PATH,
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    n_epochs: int = N_EPOCHS,
) -> dict[str, Any]:
    """Run Exp 1394 end-to-end and deploy the combined verifier checkpoint."""

    started_at = datetime.now(tz=UTC).isoformat()
    t0 = time.perf_counter()
    write_in_progress_artifact(out_path, project_root=project_root, run_date=run_date)

    exp1381 = load_json(exp1381_path)
    exp1382 = load_json(exp1382_path)
    exp1388 = load_json(exp1388_path)
    source_checkpoint = deployed_exp1381_checkpoint(exp1381)
    fresh_cases = load_fresh_dvi_cases(exp1388, exp1382)
    rows = dvi.load_jsonl_rows(fover_path)
    train_rows, holdout_rows = dvi.split_fover_rows(rows)
    negative_cases = dvi.select_negative_cases(
        train_rows,
        positive_count=len(fresh_cases),
        ratio=dvi.NEGATIVE_RATIO,
    )

    dvi_result = dvi.run_dvi_training(
        positive_cases=fresh_cases,
        negative_cases=negative_cases,
        holdout_rows=holdout_rows,
        checkpoint_path=checkpoint_path,
        baseline_checkpoint_path=source_checkpoint,
        n_epochs=n_epochs,
        learning_rate=dvi.LEARNING_RATE,
        l2_weight_decay=dvi.L2_WEIGHT_DECAY,
    )
    secl_measurement = measure_secl_for_dvi_v2(
        checkpoint_path=checkpoint_path,
        fresh_cases=fresh_cases,
        train_rows=train_rows,
        holdout_rows=holdout_rows,
    )
    deployed = save_combined_checkpoint(
        checkpoint_path,
        dvi_result=dvi_result,
        secl_measurement=secl_measurement,
        fresh_cases_used=len(fresh_cases),
        source_checkpoint_path=source_checkpoint,
    )
    artifact = build_artifact(
        fresh_cases_used=len(fresh_cases),
        negative_cases_used=len(negative_cases),
        dvi_result=dvi_result,
        secl_measurement=secl_measurement,
        dvi_v2_deployed=deployed,
        checkpoint_path=checkpoint_path,
        source_checkpoint_path=source_checkpoint,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
        train_rows_used=len(train_rows),
        project_root=project_root,
        run_date=run_date,
    )
    return _write_json(out_path, artifact)


def _rows(artifact: Mapping[str, Any], keys: Sequence[str]) -> list[dict[str, Any]]:
    for key in keys:
        value = artifact.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _ece_reduction_pct(ece_before: float, ece_after: float) -> float:
    if float(ece_before) <= 0.0:
        return 0.0
    return (float(ece_before) - float(ece_after)) / float(ece_before) * 100.0


def _honest_verdict(delta: float, ece_reduction_pct: float, deployed: bool) -> str:
    if not deployed:
        return "dvi_v2_secl_combined_not_deployed"
    if delta > 0.0 and ece_reduction_pct > 0.0:
        return "dvi_v2_secl_combined_deployed_positive_auroc_delta_ece_reduced"
    if delta > 0.0:
        return "dvi_v2_secl_combined_deployed_positive_auroc_delta_no_ece_reduction"
    if ece_reduction_pct > 0.0:
        return "dvi_v2_secl_combined_deployed_no_auroc_gain_ece_reduced"
    return "dvi_v2_secl_combined_deployed_no_measured_gain"


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
