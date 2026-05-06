"""Exp 1415 DVI v3 update from Exp 1395's 1508 fresh verified FoVer cases.

DVI v2 promoted a much larger fresh FoVer slice in Exp 1395 than the 59-case
Exp 1394 training set.  This module treats those promoted IDs as labeled
discriminative supervision, initializes from the deployed Exp 1394 DVI v2 +
SECL checkpoint, and deploys v3 only if the measured AUROC delta beats the DVI
v2 delta while replay and calibration gates stay intact.

Spec: REQ-VERIFY-1415, SCENARIO-VERIFY-1415.
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
from carnot.reporting import fr11_self_learning_v5 as fr11
from carnot.reporting import secl_discriminative_self_calibration as secl
from carnot.verify.sc_energy_verifier import SCEnergyVerifier


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_VERIFY_DIR = REPO_ROOT / "python" / "carnot" / "verify"
DEFAULT_FOVER_PATH = REPO_ROOT / "data" / "fover_corpus.jsonl"

EXP1394_FILE = "experiment_1394_dvi_v2_secl_combined.json"
EXP1395_FILE = "experiment_1395_fr11_self_learning_v5.json"
OUTPUT_FILE = "experiment_1415_dvi_v3_1508_fresh_cases.json"

DEFAULT_EXP1394_PATH = DEFAULT_RESULTS_DIR / EXP1394_FILE
DEFAULT_EXP1395_PATH = DEFAULT_RESULTS_DIR / EXP1395_FILE
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE
DEFAULT_CHECKPOINT_PATH = DEFAULT_VERIFY_DIR / "dvi_v3_1508_fresh_cases_checkpoint.pt"

EXPERIMENT = "1415_dvi_v3_1508_fresh_cases"
SCHEMA = "dvi_v3_1508_fresh_cases_v1"
RUN_DATE = "20260506"
FRESH_VERIFIED_CASE_COUNT = 1508
DVI_V2_AUROC_DELTA_BASELINE = 0.011458
N_EPOCHS = 20
MIN_NONFORGETTING_RATE = 0.99
MIN_SECL_PRESERVATION_RATIO = 0.90
TRAINING_METHOD = "secl_style_bce_sc_energy_linear_adapter_v3_exp1395_labeled_fover"
PROMOTED_PREFIX = "dvi_v2:fover:"

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "fresh_verified_cases_used",
    "dvi_v2_auroc_delta_baseline",
    "dvi_v3_auroc_delta",
    "dvi_v3_deployed",
    "dvi_v3_checkpoint_path",
    "nonforgetting_rate",
    "secl_ece_reduction_pct_preserved",
    "tests_run",
    "honest_verdict",
)


@dataclass(frozen=True)
class DviV3TrainingResult:
    """Measured v3 boundary and the weights that may become the checkpoint.

    The metric and bias are kept out of the JSON artifact because they are model
    state rather than reporting metadata.  Keeping them here lets the deployment
    gate decide honestly whether to persist a checkpoint after all metrics are
    known.
    """

    baseline_auroc: float
    trained_auroc: float
    auroc_delta: float
    metric: np.ndarray
    bias: float
    loss_history: list[float]
    source_checkpoint_path: str


@dataclass(frozen=True)
class SECLPreservationResult:
    """Held-out ECE measurement after applying the inherited SECL head.

    DVI v3 changes the discriminator boundary, so merely copying the SECL head
    would be an unsupported claim unless the held-out ECE reduction is checked
    again.  This result records that check separately from the deployment gate.
    """

    ece_before: float
    ece_after: float
    ece_reduction_pct: float
    preserved: bool


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
    """REQ-VERIFY-1415: write the bootstrap artifact before source loading."""

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "status": "in_progress",
            "expected_fresh_verified_cases": FRESH_VERIFIED_CASE_COUNT,
            "fresh_verified_cases_used": 0,
            "dvi_v2_auroc_delta_baseline": round(float(dvi_v2_auroc_delta_baseline), 6),
            "dvi_v3_baseline_auroc": None,
            "dvi_v3_trained_auroc": None,
            "dvi_v3_auroc_delta": None,
            "dvi_v3_deployed": False,
            "dvi_v3_checkpoint_path": str(DEFAULT_CHECKPOINT_PATH),
            "nonforgetting_rate": None,
            "secl_ece_reduction_pct_preserved": False,
            "tests_run": [],
            "honest_verdict": "in_progress",
            "fresh_llm_inference_used": False,
            "cpu_only": True,
        },
    )


def load_json(path: Path | str) -> dict[str, Any]:
    """Load a JSON artifact and reject non-object payloads."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return payload


def fresh_case_ids_from_exp1395(
    exp1395_artifact: Mapping[str, Any],
    *,
    expected_count: int = FRESH_VERIFIED_CASE_COUNT,
) -> list[str]:
    """REQ-VERIFY-1415: return Exp 1395's promoted DVI v2 FoVer case IDs."""

    promoted = exp1395_artifact.get("memory_updates", {}).get("promoted", [])
    if not isinstance(promoted, Sequence) or isinstance(promoted, (str, bytes)):
        raise ValueError("Exp 1395 memory_updates.promoted must be a list")

    case_ids = _prefixed_case_ids(promoted)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Exp 1395 promoted DVI v2 FoVer IDs contain duplicates")

    artifact_count = _int(exp1395_artifact.get("fresh_verified_sample_count"), expected_count)
    if len(case_ids) != artifact_count or len(case_ids) != int(expected_count):
        raise ValueError(
            "Exp 1395 fresh verified count mismatch: "
            f"ids={len(case_ids)} artifact={artifact_count} expected={expected_count}"
        )
    return case_ids


def replay_case_ids_from_exp1395(
    exp1395_artifact: Mapping[str, Any],
    *,
    max_replay_cases: int | None = None,
) -> list[str]:
    """Return deterministic demoted replay IDs for the nonforgetting gate."""

    demoted = exp1395_artifact.get("memory_updates", {}).get("demoted", [])
    if not isinstance(demoted, Sequence) or isinstance(demoted, (str, bytes)):
        return []
    ids = _dedupe_preserving_order(_prefixed_case_ids(demoted))
    if max_replay_cases is None:
        return ids
    return ids[: max(0, int(max_replay_cases))]


def load_fresh_verified_cases(
    exp1395_artifact: Mapping[str, Any],
    fover_rows: Sequence[Mapping[str, Any]],
    *,
    expected_count: int = FRESH_VERIFIED_CASE_COUNT,
) -> list[dvi.DviCase]:
    """Join Exp 1395 fresh IDs back to FoVer rows with their correctness labels."""

    case_ids = fresh_case_ids_from_exp1395(exp1395_artifact, expected_count=expected_count)
    case_map = _fover_case_map(fover_rows)
    cases: list[dvi.DviCase] = []
    for case_id in case_ids:
        row = case_map.get(case_id)
        if row is None:
            raise ValueError(f"Exp 1395 fresh case missing from FoVer corpus: {case_id}")
        cases.append(_dvi_case_from_fover(row, source="exp1395_dvi_v2_secl_fresh_verified_fover"))
    return cases


def load_replay_cases(
    exp1395_artifact: Mapping[str, Any],
    fover_rows: Sequence[Mapping[str, Any]],
    *,
    max_replay_cases: int | None = None,
) -> list[dvi.DviCase]:
    """Load demoted Exp 1395 FoVer rows used to measure nonforgetting."""

    case_ids = replay_case_ids_from_exp1395(
        exp1395_artifact,
        max_replay_cases=max_replay_cases,
    )
    case_map = _fover_case_map(fover_rows)
    cases: list[dvi.DviCase] = []
    for case_id in case_ids:
        row = case_map.get(case_id)
        if row is not None:
            cases.append(_dvi_case_from_fover(row, source="exp1395_dvi_v2_secl_replay_demoted"))
    return cases


def run_dvi_v3_training(
    *,
    fresh_cases: Sequence[dvi.DviCase],
    holdout_rows: Sequence[Mapping[str, Any]],
    source_checkpoint_path: Path | str,
    n_epochs: int = N_EPOCHS,
) -> DviV3TrainingResult:
    """Train v3 on the full labeled Exp 1395 fresh set without saving yet."""

    positive_cases = [case for case in fresh_cases if int(case.label) == 0]
    negative_cases = [case for case in fresh_cases if int(case.label) == 1]
    if not positive_cases or not negative_cases:
        raise ValueError("DVI v3 fresh corpus must contain both correct and incorrect labels")

    state = secl.load_verifier_state(source_checkpoint_path)
    verifier = SCEnergyVerifier(model_name="deterministic", hidden_dim=int(state.metric.size))
    baseline_auroc = dvi.evaluate_auroc(verifier, state.metric, state.bias, holdout_rows)
    metric, bias, loss_history = dvi.train_bce_classifier(
        verifier=verifier,
        initial_metric=state.metric,
        initial_bias=state.bias,
        positive_cases=positive_cases,
        negative_cases=negative_cases,
        n_epochs=n_epochs,
        learning_rate=dvi.LEARNING_RATE,
        l2_weight_decay=dvi.L2_WEIGHT_DECAY,
    )
    trained_auroc = dvi.evaluate_auroc(verifier, metric, bias, holdout_rows)
    return DviV3TrainingResult(
        baseline_auroc=baseline_auroc,
        trained_auroc=trained_auroc,
        auroc_delta=trained_auroc - baseline_auroc,
        metric=metric,
        bias=bias,
        loss_history=loss_history,
        source_checkpoint_path=str(source_checkpoint_path),
    )


def measure_nonforgetting_rate(
    *,
    replay_cases: Sequence[dvi.DviCase],
    metric: np.ndarray,
    bias: float,
    confidence_head: secl.HistogramECEConfidenceHead,
    incorrect_threshold: float = fr11.DVI_INCORRECT_THRESHOLD,
    secl_confidence_threshold: float = fr11.SECL_CONFIDENCE_THRESHOLD,
) -> float:
    """Measure the fraction of Exp 1395 demoted replay rows that remain demoted."""

    if not replay_cases:
        return 1.0
    verifier = SCEnergyVerifier(model_name="deterministic", hidden_dim=int(metric.size))
    preserved = 0
    for case in replay_cases:
        action = memory_action_for_case(
            case,
            verifier=verifier,
            metric=metric,
            bias=bias,
            confidence_head=confidence_head,
            incorrect_threshold=incorrect_threshold,
            secl_confidence_threshold=secl_confidence_threshold,
        )
        if action == fr11.POLICY_DEMOTE:
            preserved += 1
    return float(preserved) / float(len(replay_cases))


def memory_action_for_case(
    case: dvi.DviCase,
    *,
    verifier: SCEnergyVerifier,
    metric: np.ndarray,
    bias: float,
    confidence_head: secl.HistogramECEConfidenceHead,
    incorrect_threshold: float,
    secl_confidence_threshold: float,
) -> str:
    """Apply the Exp 1395 DVI+SECL memory action rule to a single case."""

    incorrect_probability = dvi.predict_incorrect_probability(verifier, metric, bias, case.text)
    dvi_predicts_incorrect = incorrect_probability >= float(incorrect_threshold)
    semantic_result = fr11.STATE_REPAIR_HINT if dvi_predicts_incorrect else fr11.STATE_SAT
    certificate_state = fr11.STATE_REPAIR_HINT if int(case.label) == 1 else fr11.STATE_SAT
    predicted_state_probability = (
        incorrect_probability if dvi_predicts_incorrect else 1.0 - incorrect_probability
    )
    secl_confidence = float(confidence_head.predict([predicted_state_probability])[0])
    if semantic_result == certificate_state and secl_confidence >= float(secl_confidence_threshold):
        return fr11.POLICY_PROMOTE
    return fr11.POLICY_DEMOTE


def measure_secl_preservation(
    *,
    metric: np.ndarray,
    bias: float,
    confidence_head: secl.HistogramECEConfidenceHead,
    holdout_rows: Sequence[Mapping[str, Any]],
    v2_ece_reduction_pct: float,
    min_preservation_ratio: float = MIN_SECL_PRESERVATION_RATIO,
) -> SECLPreservationResult:
    """Re-measure held-out ECE with the inherited SECL head after v3 training."""

    verifier = SCEnergyVerifier(model_name="deterministic", hidden_dim=int(metric.size))
    labels: list[int] = []
    raw_probs: list[float] = []
    for row in holdout_rows:
        label = secl.row_is_correct(row)
        text = secl.row_text(row)
        if label is None or not text:
            continue
        labels.append(1 if label else 0)
        incorrect = dvi.predict_incorrect_probability(verifier, metric, bias, text)
        raw_probs.append(1.0 - incorrect)

    if not labels:
        return SECLPreservationResult(
            ece_before=0.0,
            ece_after=0.0,
            ece_reduction_pct=0.0,
            preserved=False,
        )

    calibrated = confidence_head.predict(raw_probs)
    ece_before = secl.expected_calibration_error(labels, raw_probs)
    ece_after = secl.expected_calibration_error(labels, calibrated)
    reduction = _ece_reduction_pct(ece_before, ece_after)
    baseline = float(v2_ece_reduction_pct)
    preserved = reduction > 0.0
    if baseline > 0.0:
        preserved = preserved and reduction >= baseline * float(min_preservation_ratio)
    return SECLPreservationResult(
        ece_before=ece_before,
        ece_after=ece_after,
        ece_reduction_pct=reduction,
        preserved=bool(preserved),
    )


def save_v3_checkpoint(
    path: Path | str,
    *,
    training_result: DviV3TrainingResult,
    confidence_head: secl.HistogramECEConfidenceHead,
    fresh_verified_cases_used: int,
) -> bool:
    """Persist the deployable v3 metric plus the inherited SECL confidence head."""

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
            fresh_verified_cases_used=np.asarray([fresh_verified_cases_used], dtype=np.int32),
            training_method=np.asarray([TRAINING_METHOD]),
            source_checkpoint_path=np.asarray([training_result.source_checkpoint_path]),
        )
    return destination.exists()


def build_artifact(
    *,
    fresh_verified_cases_used: int,
    replay_cases_used: int,
    dvi_v2_auroc_delta_baseline: float,
    training_result: DviV3TrainingResult,
    nonforgetting_rate: float,
    secl_preservation: SECLPreservationResult,
    deployed: bool,
    checkpoint_path: Path | str,
    source_checkpoint_path: Path | str,
    started_at: str,
    duration_s: float,
    train_cases_used: int,
    heldout_cases_used: int,
    tests_run: Sequence[str],
    block_reasons: Sequence[str] | None = None,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    expected_fresh_count: int = FRESH_VERIFIED_CASE_COUNT,
) -> dict[str, Any]:
    """Build the terminal Exp 1415 artifact with deploy or blocked status."""

    baseline_auroc = round(float(training_result.baseline_auroc), 6)
    trained_auroc = round(float(training_result.trained_auroc), 6)
    delta = round(float(training_result.auroc_delta), 6)
    reasons = list(block_reasons or [])
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "started_at": started_at,
        "finished_at": datetime.now(tz=UTC).isoformat(),
        "duration_s": round(float(duration_s), 3),
        "status": "complete" if deployed else "blocked",
        "spec": ["REQ-VERIFY-1415", "SCENARIO-VERIFY-1415"],
        "source_artifacts": [
            f"results/{EXP1394_FILE}",
            f"results/{EXP1395_FILE}",
            "data/fover_corpus.jsonl",
        ],
        "source_checkpoint_path": str(source_checkpoint_path),
        "expected_fresh_verified_cases": int(expected_fresh_count),
        "fresh_verified_cases_used": int(fresh_verified_cases_used),
        "fresh_case_source": "exp1395_memory_updates_promoted_dvi_v2_fover",
        "train_cases_used": int(train_cases_used),
        "replay_case_source": "exp1395_memory_updates_demoted_dvi_v2_fover",
        "replay_cases_used": int(replay_cases_used),
        "fover_split_seed": dvi.FOVER_SPLIT_SEED,
        "fover_heldout_cases_used": int(heldout_cases_used),
        "training_method": TRAINING_METHOD,
        "epochs_run": len(training_result.loss_history),
        "learning_rate": dvi.LEARNING_RATE,
        "l2_weight_decay": dvi.L2_WEIGHT_DECAY,
        "training_loss_history": [round(float(loss), 6) for loss in training_result.loss_history],
        "dvi_v2_auroc_delta_baseline": round(float(dvi_v2_auroc_delta_baseline), 6),
        "dvi_v3_baseline_auroc": baseline_auroc,
        "dvi_v3_trained_auroc": trained_auroc,
        "dvi_v3_auroc_delta": delta,
        "dvi_v3_delta_improved_over_v2": delta > float(dvi_v2_auroc_delta_baseline),
        "nonforgetting_rate": round(float(nonforgetting_rate), 6),
        "min_nonforgetting_rate": MIN_NONFORGETTING_RATE,
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
    """SCENARIO-VERIFY-1415: enforce required fields and deployment invariants."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] not in {"in_progress", "complete", "blocked"}:
        raise AssertionError(f"unsupported status: {artifact['status']}")
    if not isinstance(artifact["tests_run"], list):
        raise AssertionError("tests_run must be a list of command strings")
    if artifact["status"] in {"complete", "blocked"}:
        expected = int(artifact.get("expected_fresh_verified_cases", FRESH_VERIFIED_CASE_COUNT))
        if int(artifact["fresh_verified_cases_used"]) != expected:
            raise AssertionError(f"fresh_verified_cases_used must equal expected count {expected}")
        if artifact["dvi_v3_auroc_delta"] is None:
            raise AssertionError("terminal DVI v3 artifact requires AUROC delta")
        if artifact["nonforgetting_rate"] is None:
            raise AssertionError("terminal DVI v3 artifact requires nonforgetting_rate")
    if artifact["dvi_v3_deployed"]:
        path = artifact.get("dvi_v3_checkpoint_path")
        if not path or not Path(str(path)).exists():
            raise AssertionError("dvi_v3_deployed requires an existing checkpoint path")
    elif artifact["status"] == "blocked" and artifact.get("dvi_v3_checkpoint_path") is not None:
        raise AssertionError("blocked DVI v3 artifacts must not expose a deployed checkpoint")


def run(
    *,
    exp1394_path: Path | str = DEFAULT_EXP1394_PATH,
    exp1395_path: Path | str = DEFAULT_EXP1395_PATH,
    fover_path: Path | str = DEFAULT_FOVER_PATH,
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    expected_fresh_count: int = FRESH_VERIFIED_CASE_COUNT,
    dvi_v2_auroc_delta_baseline: float | None = None,
    min_nonforgetting_rate: float = MIN_NONFORGETTING_RATE,
    require_secl_preserved: bool = True,
    n_epochs: int = N_EPOCHS,
    tests_run: Sequence[str] | None = None,
    max_replay_cases: int | None = None,
) -> dict[str, Any]:
    """Run Exp 1415 end-to-end and deploy only if all v3 gates pass."""

    started_at = datetime.now(tz=UTC).isoformat()
    t0 = time.perf_counter()
    baseline_delta = (
        DVI_V2_AUROC_DELTA_BASELINE
        if dvi_v2_auroc_delta_baseline is None
        else float(dvi_v2_auroc_delta_baseline)
    )
    write_in_progress_artifact(
        out_path,
        project_root=project_root,
        run_date=run_date,
        dvi_v2_auroc_delta_baseline=baseline_delta,
    )

    exp1394 = load_json(exp1394_path)
    exp1395 = load_json(exp1395_path)
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
                expected_fresh_count=expected_fresh_count,
                block_reason=activation.blocker or "dvi_v2_checkpoint_inactive",
            ),
        )

    source_checkpoint = Path(activation.state.checkpoint_path)
    rows = dvi.load_jsonl_rows(fover_path)
    fresh_cases = load_fresh_verified_cases(
        exp1395,
        rows,
        expected_count=expected_fresh_count,
    )
    replay_cases = load_replay_cases(
        exp1395,
        rows,
        max_replay_cases=max_replay_cases,
    )
    _, holdout_rows = dvi.split_fover_rows(rows)
    confidence_head = secl.HistogramECEConfidenceHead(
        bin_values=np.asarray(activation.state.secl_bin_values, dtype=np.float64),
        global_value=float(activation.state.secl_global_value),
        n_bins=int(activation.state.secl_n_bins),
    )

    training_result = run_dvi_v3_training(
        fresh_cases=fresh_cases,
        holdout_rows=holdout_rows,
        source_checkpoint_path=source_checkpoint,
        n_epochs=n_epochs,
    )
    nonforgetting_rate = measure_nonforgetting_rate(
        replay_cases=replay_cases,
        metric=training_result.metric,
        bias=training_result.bias,
        confidence_head=confidence_head,
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
        secl_preserved=secl_preservation.preserved,
        require_secl_preserved=require_secl_preserved,
    )
    deployed = not reasons
    if deployed:
        deployed = save_v3_checkpoint(
            checkpoint_path,
            training_result=training_result,
            confidence_head=confidence_head,
            fresh_verified_cases_used=len(fresh_cases),
        )
        if not deployed:
            reasons.append("dvi_v3_checkpoint_write_failed")

    artifact = build_artifact(
        fresh_verified_cases_used=len(fresh_cases),
        replay_cases_used=len(replay_cases),
        dvi_v2_auroc_delta_baseline=baseline_delta,
        training_result=training_result,
        nonforgetting_rate=nonforgetting_rate,
        secl_preservation=secl_preservation,
        deployed=deployed and not reasons,
        checkpoint_path=checkpoint_path,
        source_checkpoint_path=source_checkpoint,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
        train_cases_used=len(fresh_cases),
        heldout_cases_used=len(holdout_rows),
        tests_run=list(tests_run or []),
        block_reasons=reasons,
        project_root=project_root,
        run_date=run_date,
        expected_fresh_count=expected_fresh_count,
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
    expected_fresh_count: int,
    block_reason: str,
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
        "spec": ["REQ-VERIFY-1415", "SCENARIO-VERIFY-1415"],
        "source_artifacts": [f"results/{EXP1394_FILE}", f"results/{EXP1395_FILE}"],
        "expected_fresh_verified_cases": int(expected_fresh_count),
        "fresh_verified_cases_used": 0,
        "dvi_v2_auroc_delta_baseline": round(float(dvi_v2_auroc_delta_baseline), 6),
        "dvi_v3_baseline_auroc": None,
        "dvi_v3_trained_auroc": None,
        "dvi_v3_auroc_delta": None,
        "dvi_v3_deployed": False,
        "dvi_v3_checkpoint_path": None,
        "nonforgetting_rate": None,
        "secl_ece_reduction_pct_preserved": False,
        "block_reasons": [block_reason],
        "tests_run": list(tests_run),
        "fresh_llm_inference_used": False,
        "cpu_only": True,
        "honest_verdict": _honest_verdict(False, [block_reason]),
    }
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    return artifact


def _deployment_block_reasons(
    *,
    dvi_v3_delta: float,
    dvi_v2_delta_baseline: float,
    nonforgetting_rate: float,
    min_nonforgetting_rate: float,
    secl_preserved: bool,
    require_secl_preserved: bool,
) -> list[str]:
    reasons: list[str] = []
    if float(dvi_v3_delta) <= float(dvi_v2_delta_baseline):
        reasons.append("dvi_v3_delta_not_improved")
    if float(nonforgetting_rate) < float(min_nonforgetting_rate):
        reasons.append("nonforgetting_below_gate")
    if require_secl_preserved and not secl_preserved:
        reasons.append("secl_ece_reduction_not_preserved")
    return reasons


def _fover_case_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, fr11.FoVerSelfLearningCase]:
    return {case.case_id: case for case in fr11.normalize_fover_cases(rows)}


def _dvi_case_from_fover(case: fr11.FoVerSelfLearningCase, *, source: str) -> dvi.DviCase:
    label = 1 if case.is_incorrect else 0
    text = " | ".join(
        part
        for part in (
            f"case_id={case.case_id}",
            f"question={case.question}",
            f"response={case.response}",
            f"fover_label={'incorrect' if case.is_incorrect else 'correct'}",
            f"certificate_state={case.certificate_state}",
            f"source={case.source}",
        )
        if part and not part.endswith("=")
    )
    return dvi.DviCase(case_id=case.case_id, text=text, label=label, source=source)


def _prefixed_case_ids(items: Sequence[Any]) -> list[str]:
    ids: list[str] = []
    for item in items:
        value = str(item)
        if value.startswith(PROMOTED_PREFIX):
            ids.append(value[len(PROMOTED_PREFIX) :])
    return ids


def _dedupe_preserving_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _ece_reduction_pct(ece_before: float, ece_after: float) -> float:
    before = float(ece_before)
    if before <= 0.0:
        return 0.0
    return (before - float(ece_after)) / before * 100.0


def _honest_verdict(deployed: bool, block_reasons: Sequence[str]) -> str:
    if deployed:
        return "dvi_v3_deployed_delta_improved_nonforgetting_secl_preserved"
    if block_reasons:
        return "dvi_v3_blocked_" + "_and_".join(str(reason) for reason in block_reasons)
    return "dvi_v3_blocked_unknown_reason"


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), indent=2, sort_keys=True))
