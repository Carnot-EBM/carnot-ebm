"""Exp 1386 SECL discriminative self-calibration for SC-Energy.

SECL uses the answer to the discriminative question, "is this case correct?",
as self-supervision for confidence calibration.  In Carnot that signal comes
from verifier acceptance for the Exp 1374 promoted semantic positives and from
FoVer contrastive negatives requested by the experiment prompt.  The calibrated
confidence head is intentionally small and CPU-only: fixed confidence bins are
assigned the empirical mean discriminative signal observed in the calibration
slice, which directly minimizes empirical ECE for those bins.

Spec: REQ-VERIFY-1386, SCENARIO-VERIFY-1386.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from carnot.verify.sc_energy_verifier import SCEnergyVerifier, _Pair


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_MODELS_DIR = REPO_ROOT / "python" / "carnot" / "models"
DEFAULT_FOVER_PATH = REPO_ROOT / "data" / "fover_corpus.jsonl"
DEFAULT_EXP1374_PATH = (
    DEFAULT_RESULTS_DIR
    / "experiment_1374_continuous_self_learning_v3_verifier_selected_or_csp_fallback.json"
)
DEFAULT_OUTPUT_PATH = (
    DEFAULT_RESULTS_DIR / "experiment_1386_secl_discriminative_self_calibration.json"
)
DEFAULT_SC_ENERGY_CHECKPOINT = DEFAULT_MODELS_DIR / "sc_energy_v2_regularized.pt"

EXPERIMENT = "1386_secl_discriminative_self_calibration"
SCHEMA = "secl_discriminative_self_calibration_v1"
RUN_DATE = "20260505"
TRAINING_METHOD = "secl_histogram_ece_confidence_head_v1"
FOVER_SPLIT_SEED = 1386
FOVER_TRAIN_FRACTION = 0.8
N_BINS = 10
NEGATIVES_PER_POSITIVE = 1
CALIBRATION_STREAM_TARGET = 32

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "verifier_targeted",
    "positive_cases_used",
    "ece_before",
    "ece_after",
    "ece_reduction_pct",
    "discriminative_signal_correlation",
    "calibration_cases_used",
    "secl_viable_for_dvi",
    "honest_verdict",
)


@dataclass(frozen=True)
class SECLCase:
    """One row used by the SECL confidence head.

    ``discriminative_signal`` is the self-supervision target: 1.0 means the
    discriminative verifier path says the reasoning case is correct, and 0.0
    means it is a contrastive incorrect case.
    """

    case_id: str
    text: str
    discriminative_signal: float
    source: str


@dataclass(frozen=True)
class VerifierState:
    """The linear SC-Energy state used to compute pre-calibration confidence."""

    metric: np.ndarray
    bias: float
    target_name: str
    checkpoint_path: str | None
    checkpoint_kind: str


@dataclass(frozen=True)
class HistogramECEConfidenceHead:
    """Fixed-bin confidence head trained by empirical ECE minimization.

    For a fixed set of bins, ECE depends on the gap between a bin's empirical
    accuracy and its mean predicted confidence.  Setting every calibrated
    confidence in that bin to the empirical discriminative-signal mean makes
    that bin's calibration gap zero on the calibration slice.
    """

    bin_values: np.ndarray
    global_value: float
    n_bins: int = N_BINS

    def predict(self, probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return calibrated correctness probabilities for raw confidences."""

        probs = np.asarray(probabilities, dtype=np.float64)
        clipped = np.clip(probs, 0.0, 1.0)
        indices = np.minimum(self.n_bins - 1, np.floor(clipped * self.n_bins).astype(int))
        return np.asarray(self.bin_values[indices], dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the head into JSON-friendly primitive values."""

        return {
            "type": "histogram_ece_confidence_head",
            "n_bins": int(self.n_bins),
            "global_value": round(float(self.global_value), 8),
            "bin_values": [round(float(value), 8) for value in self.bin_values],
        }


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
    """REQ-VERIFY-1386: write the bootstrap artifact before source loading."""

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "status": "in_progress",
            "verifier_targeted": None,
            "positive_cases_used": 0,
            "ece_before": None,
            "ece_after": None,
            "ece_reduction_pct": None,
            "discriminative_signal_correlation": None,
            "calibration_cases_used": 0,
            "secl_viable_for_dvi": False,
            "honest_verdict": "in_progress",
            "fresh_llm_inference_used": False,
            "cpu_only": True,
        },
    )


def load_positive_cases(exp1374_artifact: Mapping[str, Any]) -> list[SECLCase]:
    """Return Exp 1374 promoted semantic positives as SECL signal=correct."""

    if exp1374_artifact.get("path_used") != "primary_semantic_verified":
        return []

    cases: list[SECLCase] = []
    for index, row in enumerate(exp1374_artifact.get("variant_questions", [])):
        if not isinstance(row, Mapping):
            continue
        evidence = row.get("evidence_summary", {})
        if not isinstance(evidence, Mapping):
            evidence = {}
        accepted = bool(row.get("verifier_accepted")) and not bool(row.get("semantic_rejected"))
        promoted = str(row.get("memory_action", "promote")) == "promote"
        if not (accepted and promoted and evidence.get("constraint_passed") is True):
            continue
        case_id = str(row.get("case_id") or f"exp1374_positive_{index}")
        cases.append(
            SECLCase(
                case_id=case_id,
                text=_positive_case_text(row, evidence),
                discriminative_signal=1.0,
                source="exp1374_primary_semantic_verified",
            )
        )
    return cases


def _positive_case_text(row: Mapping[str, Any], evidence: Mapping[str, Any]) -> str:
    parts = [
        str(row.get("question") or ""),
        f"case_id={row.get('case_id')}",
        f"semantic_result={evidence.get('semantic_result')}",
        f"certificate_state={evidence.get('certificate_state')}",
        f"expected_state={evidence.get('expected_state')}",
        f"claim_route={evidence.get('claim_route')}",
        f"constraint_passed={evidence.get('constraint_passed')}",
    ]
    return " | ".join(part for part in parts if part and part != "None")


def load_fover_rows(path: Path | str) -> list[dict[str, Any]]:
    """Load FoVer rows from JSONL, a JSON list, or a dict with a row list."""

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            return [dict(row) for row in payload if isinstance(row, Mapping)]
        if isinstance(payload, Mapping):
            for key in ("rows", "items", "examples", "corpus"):
                rows = payload.get(key)
                if isinstance(rows, list):
                    return [dict(row) for row in rows if isinstance(row, Mapping)]
            return []

    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, Mapping):
            rows.append(dict(row))
    return rows


def row_text(row: Mapping[str, Any]) -> str:
    """Return the verifier-visible reasoning text from a FoVer row."""

    return str(row.get("step_text") or row.get("response") or row.get("step") or "").strip()


def row_is_correct(row: Mapping[str, Any]) -> bool | None:
    """Normalize FoVer schema variants into a binary correctness label."""

    if "is_correct" in row:
        return bool(row["is_correct"])
    if "step_correct" in row:
        return bool(row["step_correct"])
    raw = row.get("label", row.get("sc_energy_label", row.get("coherence_label")))
    if isinstance(raw, bool):
        return bool(raw)
    if isinstance(raw, str):
        label = raw.lower()
        if label in {"correct", "coherent", "true", "1"}:
            return True
        if label in {"incorrect", "incoherent", "wrong", "false", "0"}:
            return False
    return None


def split_fover_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = FOVER_SPLIT_SEED,
    train_fraction: float = FOVER_TRAIN_FRACTION,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split FoVer rows by question id so held-out rows are question-disjoint."""

    materialized = [dict(row) for row in rows if row_text(row) and row_is_correct(row) is not None]
    qids = sorted(
        {
            str(row.get("question_id", row.get("id", index)))
            for index, row in enumerate(materialized)
        }
    )
    if len(qids) < 2:
        split_at = max(1, int(train_fraction * len(materialized)))
        return materialized[:split_at], materialized[split_at:]

    rng = random.Random(seed)
    rng.shuffle(qids)
    split_at = min(len(qids) - 1, max(1, int(train_fraction * len(qids))))
    train_qids = set(qids[:split_at])
    train_rows = [
        row
        for index, row in enumerate(materialized)
        if str(row.get("question_id", row.get("id", index))) in train_qids
    ]
    holdout_rows = [
        row
        for index, row in enumerate(materialized)
        if str(row.get("question_id", row.get("id", index))) not in train_qids
    ]
    return train_rows, holdout_rows


def select_negative_cases(
    rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
) -> list[SECLCase]:
    """Select FoVer incorrect rows as contrastive SECL signal=incorrect."""

    cases: list[SECLCase] = []
    for index, row in enumerate(rows):
        if row_is_correct(row) is not False:
            continue
        text = row_text(row)
        if not text:
            continue
        cases.append(
            SECLCase(
                case_id=str(row.get("question_id") or row.get("id") or f"fover_negative_{index}"),
                text=text,
                discriminative_signal=0.0,
                source="fover_contrastive_negative",
            )
        )
        if len(cases) >= count:
            break
    if len(cases) < count:
        raise ValueError(f"FoVer negatives below SECL calibration need: {len(cases)} < {count}")
    return cases


def load_verifier_state(
    checkpoint_path: Path | str | None = DEFAULT_SC_ENERGY_CHECKPOINT,
) -> VerifierState:
    """Load the SC-Energy linear checkpoint or use the deterministic fallback."""

    source_path = Path(checkpoint_path) if checkpoint_path is not None else None
    if source_path is not None and source_path.exists():
        loaded = _load_npz_metric_checkpoint(source_path)
        if loaded is not None:
            metric, bias = loaded
            return VerifierState(
                metric=metric,
                bias=bias,
                target_name="SC-Energy verifier",
                checkpoint_path=str(source_path),
                checkpoint_kind="sc_energy_npz",
            )

    verifier = SCEnergyVerifier(model_name="deterministic", hidden_dim=128)
    return VerifierState(
        metric=verifier._metric.astype(np.float32).copy(),
        bias=float(verifier._bias),
        target_name="SC-Energy verifier",
        checkpoint_path=str(source_path) if source_path is not None else None,
        checkpoint_kind="initialized_sc_energy_deterministic",
    )


def _load_npz_metric_checkpoint(path: Path) -> tuple[np.ndarray, float] | None:
    try:
        data = np.load(path, allow_pickle=False)
    except Exception:
        return None
    if "metric" not in data.files:
        return None
    metric = np.asarray(data["metric"], dtype=np.float32)
    if metric.ndim != 1 or metric.size == 0:
        return None
    bias_array = (
        np.asarray(data["bias"], dtype=np.float32) if "bias" in data.files else np.array([0.0])
    )
    return metric, float(bias_array.reshape(-1)[0])


def raw_correct_probability(
    verifier: SCEnergyVerifier,
    state: VerifierState,
    text: str,
) -> float:
    """Return the pre-calibration SC-Energy probability that a step is correct."""

    feature = verifier._feature_for_pair(_Pair(response=text, context=""))
    incorrect_logit = float(state.bias + np.dot(state.metric, feature))
    incorrect_probability = _sigmoid_scalar(incorrect_logit)
    return float(1.0 - incorrect_probability)


def train_ece_confidence_head(
    raw_probabilities: Sequence[float],
    discriminative_signals: Sequence[float],
    *,
    n_bins: int = N_BINS,
) -> HistogramECEConfidenceHead:
    """Train the fixed-bin confidence head by minimizing empirical ECE."""

    probs = np.clip(np.asarray(raw_probabilities, dtype=np.float64), 0.0, 1.0)
    signals = np.clip(np.asarray(discriminative_signals, dtype=np.float64), 0.0, 1.0)
    if probs.shape != signals.shape or probs.size == 0:
        raise ValueError("raw_probabilities and discriminative_signals must be non-empty peers")

    global_value = float(signals.mean())
    values = np.full(int(n_bins), global_value, dtype=np.float64)
    indices = np.minimum(int(n_bins) - 1, np.floor(probs * int(n_bins)).astype(int))
    for bin_index in range(int(n_bins)):
        mask = indices == bin_index
        if np.any(mask):
            values[bin_index] = float(signals[mask].mean())
    return HistogramECEConfidenceHead(
        bin_values=values,
        global_value=global_value,
        n_bins=int(n_bins),
    )


def expected_calibration_error(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    n_bins: int = N_BINS,
) -> float:
    """Compute standard binwise Expected Calibration Error for correctness."""

    y = np.asarray(labels, dtype=np.float64)
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    if y.shape != probs.shape or y.size == 0:
        raise ValueError("labels and probabilities must be non-empty peers")

    total = float(y.size)
    ece = 0.0
    for bin_index in range(int(n_bins)):
        lower = bin_index / float(n_bins)
        upper = (bin_index + 1) / float(n_bins)
        if bin_index == int(n_bins) - 1:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)
        if not np.any(mask):
            continue
        ece += float(mask.sum()) / total * abs(float(y[mask].mean()) - float(probs[mask].mean()))
    return float(ece)


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Return Pearson correlation, using 0.0 for degenerate constant inputs."""

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("correlation inputs must be non-empty peers")
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def build_artifact(
    *,
    state: VerifierState,
    positive_cases_used: int,
    negative_cases_used: int,
    calibration_cases_used: int,
    calibration_stream_cases_seen: int,
    ece_before: float,
    ece_after: float,
    discriminative_signal_correlation: float,
    heldout_cases_used: int,
    train_rows_used: int,
    confidence_head: HistogramECEConfidenceHead,
    started_at: str,
    duration_s: float,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Build and validate the final Exp 1386 artifact."""

    reduction = _ece_reduction_pct(ece_before, ece_after)
    viable = bool(reduction > 10.0)
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "started_at": started_at,
        "finished_at": datetime.now(tz=UTC).isoformat(),
        "duration_s": round(float(duration_s), 3),
        "status": "complete",
        "spec": ["REQ-VERIFY-1386", "SCENARIO-VERIFY-1386"],
        "verifier_targeted": _verifier_target_string(state),
        "verifier_checkpoint_path": state.checkpoint_path,
        "verifier_checkpoint_kind": state.checkpoint_kind,
        "positive_cases_used": int(positive_cases_used),
        "negative_cases_used": int(negative_cases_used),
        "positive_case_source": "exp1374_primary_semantic_verified",
        "negative_case_source": "fover_contrastive_negative",
        "calibration_cases_used": int(calibration_cases_used),
        "calibration_stream_cases_seen": int(calibration_stream_cases_seen),
        "calibration_fraction_used": round(
            float(calibration_cases_used) / max(1.0, float(calibration_stream_cases_seen)),
            6,
        ),
        "training_method": TRAINING_METHOD,
        "confidence_head": confidence_head.to_dict(),
        "ece_before": round(float(ece_before), 6),
        "ece_after": round(float(ece_after), 6),
        "ece_reduction_pct": round(float(reduction), 6),
        "discriminative_signal_correlation": round(float(discriminative_signal_correlation), 6),
        "discriminative_signal_correlation_scope": (
            "heldout_fover_sc_energy_binary_correct_output_vs_ground_truth"
        ),
        "fover_split_seed": FOVER_SPLIT_SEED,
        "fover_train_rows_used": int(train_rows_used),
        "fover_heldout_cases_used": int(heldout_cases_used),
        "n_bins": N_BINS,
        "cpu_only": True,
        "fresh_llm_inference_used": False,
        "secl_viable_for_dvi": viable,
        "honest_verdict": _honest_verdict(reduction, discriminative_signal_correlation),
        "secl_training_procedure": (
            "SECL distills the discriminative 'is this correct?' signal into "
            "a confidence head. This run uses Exp 1374 promoted verifier-accepted "
            "positives plus FoVer contrastive negatives, then evaluates ECE on "
            "a question-disjoint held-out FoVer split."
        ),
    }
    validate_artifact(artifact)
    return artifact


def _verifier_target_string(state: VerifierState) -> str:
    if state.checkpoint_path:
        return f"{state.target_name} ({state.checkpoint_kind}: {state.checkpoint_path})"
    return f"{state.target_name} ({state.checkpoint_kind})"


def _ece_reduction_pct(ece_before: float, ece_after: float) -> float:
    before = float(ece_before)
    if before <= 0.0:
        return 0.0
    return (before - float(ece_after)) / before * 100.0


def _honest_verdict(reduction_pct: float, signal_correlation: float) -> str:
    if reduction_pct > 10.0 and signal_correlation > 0.0:
        return "secl_ece_reduced_signal_positive_viable_for_dvi"
    if reduction_pct > 10.0:
        return "secl_ece_reduced_but_signal_correlation_nonpositive_dvi_unproven"
    if reduction_pct > 0.0:
        return "secl_ece_reduced_below_dvi_gate"
    return "secl_no_heldout_ece_improvement"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Enforce the user-required Exp 1386 schema fields."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] == "complete":
        if artifact["positive_cases_used"] <= 0:
            raise AssertionError("complete SECL artifact requires positive cases")
        if artifact["calibration_cases_used"] < artifact["positive_cases_used"]:
            raise AssertionError("calibration cases must include the positive cases")
        expected_viable = float(artifact["ece_reduction_pct"]) > 10.0
        if bool(artifact["secl_viable_for_dvi"]) is not expected_viable:
            raise AssertionError("secl_viable_for_dvi must equal ece_reduction_pct > 10")


def run(
    *,
    exp1374_path: Path | str = DEFAULT_EXP1374_PATH,
    fover_path: Path | str = DEFAULT_FOVER_PATH,
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    checkpoint_path: Path | str | None = DEFAULT_SC_ENERGY_CHECKPOINT,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Run Exp 1386 end-to-end and write the complete SECL artifact."""

    started_at = datetime.now(tz=UTC).isoformat()
    t0 = time.perf_counter()
    write_in_progress_artifact(out_path, project_root=project_root, run_date=run_date)

    exp1374_artifact = json.loads(Path(exp1374_path).read_text(encoding="utf-8"))
    positive_cases = load_positive_cases(exp1374_artifact)
    rows = load_fover_rows(fover_path)
    train_rows, holdout_rows = split_fover_rows(rows)
    stream_rows = train_rows[: max(0, CALIBRATION_STREAM_TARGET - len(positive_cases))]
    calibration_stream_cases_seen = len(positive_cases) + len(stream_rows)
    negative_count = max(1, len(positive_cases) * NEGATIVES_PER_POSITIVE)
    try:
        negative_cases = select_negative_cases(stream_rows, count=negative_count)
    except ValueError:
        negative_cases = select_negative_cases(train_rows, count=negative_count)
        calibration_stream_cases_seen = len(positive_cases) + len(train_rows)

    calibration_cases = [*positive_cases, *negative_cases]
    state = load_verifier_state(checkpoint_path)
    verifier = SCEnergyVerifier(model_name="deterministic", hidden_dim=int(state.metric.size))

    calibration_raw = [
        raw_correct_probability(verifier, state, case.text) for case in calibration_cases
    ]
    calibration_signal = [case.discriminative_signal for case in calibration_cases]
    confidence_head = train_ece_confidence_head(calibration_raw, calibration_signal)

    holdout_labels: list[int] = []
    raw_probs: list[float] = []
    for row in holdout_rows:
        label = row_is_correct(row)
        text = row_text(row)
        if label is None or not text:
            continue
        holdout_labels.append(1 if label else 0)
        raw_probs.append(raw_correct_probability(verifier, state, text))
    calibrated_probs = confidence_head.predict(raw_probs)

    ece_before = expected_calibration_error(holdout_labels, raw_probs)
    ece_after = expected_calibration_error(holdout_labels, calibrated_probs)
    binary_signal = [1.0 if prob >= 0.5 else 0.0 for prob in raw_probs]
    signal_correlation = pearson_correlation(binary_signal, holdout_labels)

    artifact = build_artifact(
        state=state,
        positive_cases_used=len(positive_cases),
        negative_cases_used=len(negative_cases),
        calibration_cases_used=len(calibration_cases),
        calibration_stream_cases_seen=calibration_stream_cases_seen,
        ece_before=ece_before,
        ece_after=ece_after,
        discriminative_signal_correlation=signal_correlation,
        heldout_cases_used=len(holdout_labels),
        train_rows_used=len(train_rows),
        confidence_head=confidence_head,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
        project_root=project_root,
        run_date=run_date,
    )
    return _write_json(out_path, artifact)


def _sigmoid_scalar(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-max(-40.0, min(40.0, value)))))


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
