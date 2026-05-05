"""Exp 1381 DVI discriminative verifier training.

Spec: REQ-VERIFY-1381, SCENARIO-VERIFY-1381.
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
EXP1374_FILE = "experiment_1374_continuous_self_learning_v3_verifier_selected_or_csp_fallback.json"
OUTPUT_FILE = "experiment_1381_dvi_discriminative_verifier_training_v1.json"
DEFAULT_EXP1374_PATH = DEFAULT_RESULTS_DIR / EXP1374_FILE
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE
DEFAULT_CHECKPOINT_PATH = DEFAULT_MODELS_DIR / "dvi_checkpoint_v1.pt"

EXPERIMENT = "1381_dvi_discriminative_verifier_training_v1"
SCHEMA = "dvi_discriminative_verifier_training_v1"
RUN_DATE = "20260505"
TRAINING_METHOD = "secl_style_bce_sc_energy_linear_adapter_v1"
NEGATIVE_RATIO = 3
FOVER_SPLIT_SEED = 1381
N_EPOCHS = 10
LEARNING_RATE = 0.5
L2_WEIGHT_DECAY = 1e-4

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "positive_cases_used",
    "negative_cases_used",
    "training_method",
    "dvi_baseline_auroc",
    "dvi_trained_auroc",
    "dvi_auroc_delta",
    "dvi_checkpoint_path",
    "dvi_deployed",
    "discriminative_improvement_measured",
    "honest_verdict",
)


@dataclass(frozen=True)
class DviCase:
    """One discriminative training case.

    ``label`` follows the verifier-energy convention used in this module:
    0 means a semantically verified positive/correct case and 1 means a
    contrastive incorrect case.  Keeping the labels explicit makes the saved
    training method auditable instead of burying the class direction in code.
    """

    case_id: str
    text: str
    label: int
    source: str


@dataclass(frozen=True)
class VerifierState:
    """Linear SC-Energy verifier state used for DVI fine-tuning."""

    metric: np.ndarray
    bias: float
    source_checkpoint_path: str | None
    source_checkpoint_kind: str


@dataclass(frozen=True)
class DviTrainingResult:
    """Measured DVI result before artifact serialization."""

    baseline_auroc: float
    trained_auroc: float
    auroc_delta: float
    checkpoint_path: str
    checkpoint_saved: bool
    loss_history: list[float]
    source_checkpoint_path: str | None
    source_checkpoint_kind: str


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
    """REQ-VERIFY-1381: persist an auditable bootstrap artifact first."""

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "status": "in_progress",
            "positive_cases_used": 0,
            "negative_cases_used": 0,
            "training_method": TRAINING_METHOD,
            "dvi_baseline_auroc": None,
            "dvi_trained_auroc": None,
            "dvi_auroc_delta": None,
            "dvi_checkpoint_path": str(DEFAULT_CHECKPOINT_PATH),
            "dvi_deployed": False,
            "discriminative_improvement_measured": False,
            "honest_verdict": "in_progress",
            "fresh_llm_inference_used": False,
            "model_specs_required": False,
        },
    )


def load_positive_cases(exp1374_artifact: Mapping[str, Any]) -> list[DviCase]:
    """Load DVI positives from Exp 1374's primary semantic verified path.

    The DVI milestone is specifically about using fresh semantic verifier wins,
    not replay-only memory or CSP fallback rows.  This filter therefore accepts
    only promoted, verifier-accepted rows from the primary semantic path and
    serializes the semantic evidence into the text seen by the discriminator.
    """

    if exp1374_artifact.get("path_used") != "primary_semantic_verified":
        return []

    cases: list[DviCase] = []
    for index, row in enumerate(exp1374_artifact.get("variant_questions", [])):
        if not isinstance(row, Mapping):
            continue
        evidence = row.get("evidence_summary", {})
        if not isinstance(evidence, Mapping):
            evidence = {}
        accepted = bool(row.get("verifier_accepted")) and not bool(row.get("semantic_rejected"))
        promoted = str(row.get("memory_action", "promote")) == "promote"
        constraint_passed = evidence.get("constraint_passed") is True
        if not (accepted and promoted and constraint_passed):
            continue
        case_id = str(row.get("case_id") or f"exp1374_positive_{index}")
        cases.append(
            DviCase(
                case_id=case_id,
                text=_positive_case_text(row, evidence),
                label=0,
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
    constraints = evidence.get("text_constraints")
    if isinstance(constraints, Sequence) and not isinstance(constraints, (str, bytes)):
        parts.append("text_constraints=" + ",".join(str(item) for item in constraints))
    return " | ".join(part for part in parts if part and part != "None")


def load_jsonl_rows(path: Path | str) -> list[dict[str, Any]]:
    """Load FoVer-style JSONL rows, skipping blank or malformed lines."""

    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def row_text(row: Mapping[str, Any]) -> str:
    """Return the natural-language reasoning step from a FoVer row."""

    return str(row.get("step_text") or row.get("response") or row.get("step") or "").strip()


def is_incorrect_row(row: Mapping[str, Any]) -> bool:
    """Return whether a FoVer row is labeled incorrect across schema variants."""

    if "is_correct" in row:
        return not bool(row["is_correct"])
    if "step_correct" in row:
        return not bool(row["step_correct"])
    label = row.get("label") or row.get("sc_energy_label") or row.get("coherence_label")
    if isinstance(label, str):
        return label.lower() in {"incorrect", "incoherent", "wrong", "false", "0"}
    if isinstance(label, bool):
        return not label
    return False


def split_fover_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = FOVER_SPLIT_SEED,
    train_fraction: float = 0.8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split FoVer rows by question id so held-out rows are question-disjoint."""

    materialized = [dict(row) for row in rows if row_text(row)]
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
    positive_count: int,
    ratio: int = NEGATIVE_RATIO,
) -> list[DviCase]:
    """Select FoVer incorrect rows at the required positive:negative ratio."""

    target = max(0, int(positive_count) * int(ratio))
    cases: list[DviCase] = []
    for index, row in enumerate(rows):
        text = row_text(row)
        if not text or not is_incorrect_row(row):
            continue
        cases.append(
            DviCase(
                case_id=str(row.get("question_id") or row.get("id") or f"fover_negative_{index}"),
                text=text,
                label=1,
                source="fover_incorrect_reasoning_step",
            )
        )
        if len(cases) >= target:
            break
    if len(cases) < target:
        raise ValueError(
            f"FoVer negative cases below required 1:{ratio} ratio: "
            f"{len(cases)} available for {positive_count} positives"
        )
    return cases


def tie_aware_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute AUROC with 0.5 credit for ties and 0.5 for undefined splits."""

    pos = [float(score) for label, score in zip(labels, scores) if int(label) == 1]
    neg = [float(score) for label, score in zip(labels, scores) if int(label) == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for pos_score in pos:
        for neg_score in neg:
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def identify_current_verifier_checkpoint(
    models_dir: Path | str = DEFAULT_MODELS_DIR,
) -> Path | None:
    """Prefer a GS-KAN checkpoint, then the newest SC-Energy checkpoint."""

    root = Path(models_dir)
    if not root.exists():
        return None
    preferred_patterns = (
        "*gskan*.pt",
        "*gskan*.npz",
        "*gskan*.json",
        "*sc_energy*.pt",
        "*sc_energy*.npz",
    )
    for pattern in preferred_patterns:
        candidates = [
            path
            for path in root.glob(pattern)
            if path.is_file() and path.name != DEFAULT_CHECKPOINT_PATH.name
        ]
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime)
    return None


def load_verifier_state(
    *,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    baseline_checkpoint_path: Path | str | None = None,
) -> VerifierState:
    """Load the current verifier checkpoint or initialize a deterministic state."""

    source_path = (
        Path(baseline_checkpoint_path)
        if baseline_checkpoint_path is not None
        else identify_current_verifier_checkpoint(models_dir)
    )
    if source_path is not None:
        loaded = _load_npz_metric_checkpoint(source_path)
        if loaded is not None:
            metric, bias = loaded
            return VerifierState(
                metric=metric,
                bias=bias,
                source_checkpoint_path=str(source_path),
                source_checkpoint_kind="sc_energy_npz",
            )

    verifier = SCEnergyVerifier(model_name="deterministic", hidden_dim=128)
    return VerifierState(
        metric=verifier._metric.astype(np.float32).copy(),
        bias=float(verifier._bias),
        source_checkpoint_path=str(source_path) if source_path is not None else None,
        source_checkpoint_kind="initialized_sc_energy_deterministic",
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


def run_dvi_training(
    *,
    positive_cases: Sequence[DviCase],
    negative_cases: Sequence[DviCase],
    holdout_rows: Sequence[Mapping[str, Any]],
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    baseline_checkpoint_path: Path | str | None = None,
    n_epochs: int = N_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    l2_weight_decay: float = L2_WEIGHT_DECAY,
) -> DviTrainingResult:
    """Train the SC-Energy discriminator with SECL-style BCE supervision."""

    if not positive_cases:
        raise ValueError("DVI training requires at least one positive case")
    if not negative_cases:
        raise ValueError("DVI training requires at least one negative case")

    state = load_verifier_state(
        models_dir=models_dir,
        baseline_checkpoint_path=baseline_checkpoint_path,
    )
    verifier = SCEnergyVerifier(model_name="deterministic", hidden_dim=int(state.metric.size))
    baseline_auroc = evaluate_auroc(verifier, state.metric, state.bias, holdout_rows)
    metric, bias, loss_history = train_bce_classifier(
        verifier=verifier,
        initial_metric=state.metric,
        initial_bias=state.bias,
        positive_cases=positive_cases,
        negative_cases=negative_cases,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        l2_weight_decay=l2_weight_decay,
    )
    trained_auroc = evaluate_auroc(verifier, metric, bias, holdout_rows)
    save_dvi_checkpoint(
        checkpoint_path,
        metric=metric,
        bias=bias,
        loss_history=loss_history,
        source_checkpoint_path=state.source_checkpoint_path,
    )
    saved = Path(checkpoint_path).exists()
    return DviTrainingResult(
        baseline_auroc=baseline_auroc,
        trained_auroc=trained_auroc,
        auroc_delta=trained_auroc - baseline_auroc,
        checkpoint_path=str(checkpoint_path),
        checkpoint_saved=saved,
        loss_history=loss_history,
        source_checkpoint_path=state.source_checkpoint_path,
        source_checkpoint_kind=state.source_checkpoint_kind,
    )


def train_bce_classifier(
    *,
    verifier: SCEnergyVerifier,
    initial_metric: np.ndarray,
    initial_bias: float,
    positive_cases: Sequence[DviCase],
    negative_cases: Sequence[DviCase],
    n_epochs: int = N_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    l2_weight_decay: float = L2_WEIGHT_DECAY,
) -> tuple[np.ndarray, float, list[float]]:
    """Apply binary cross-entropy updates to the verifier's linear boundary."""

    cases = list(positive_cases) + list(negative_cases)
    features = np.stack([_feature_for_text(verifier, case.text) for case in cases]).astype(
        np.float32
    )
    labels = np.asarray([case.label for case in cases], dtype=np.float32)
    metric = np.asarray(initial_metric, dtype=np.float32).copy()
    bias = float(initial_bias)
    losses: list[float] = []
    epochs = max(N_EPOCHS, int(n_epochs))
    for _ in range(epochs):
        logits = bias + features @ metric
        probs = _sigmoid_array(logits)
        losses.append(_binary_cross_entropy(labels, probs))
        error = probs - labels
        grad_metric = features.T @ error / len(labels) + l2_weight_decay * metric
        grad_bias = float(np.mean(error))
        metric = (metric - learning_rate * grad_metric).astype(np.float32)
        bias -= learning_rate * grad_bias
    return metric, bias, losses


def evaluate_auroc(
    verifier: SCEnergyVerifier,
    metric: np.ndarray,
    bias: float,
    rows: Sequence[Mapping[str, Any]],
) -> float:
    """Measure held-out FoVer AUROC where higher score means more incorrect."""

    labels: list[int] = []
    scores: list[float] = []
    for row in rows:
        text = row_text(row)
        if not text:
            continue
        labels.append(1 if is_incorrect_row(row) else 0)
        scores.append(predict_incorrect_probability(verifier, metric, bias, text))
    return tie_aware_auroc(labels, scores)


def predict_incorrect_probability(
    verifier: SCEnergyVerifier,
    metric: np.ndarray,
    bias: float,
    text: str,
) -> float:
    """Return the BCE classifier probability that a reasoning step is incorrect."""

    feature = _feature_for_text(verifier, text)
    return float(_sigmoid_scalar(float(bias + np.dot(metric, feature))))


def _feature_for_text(verifier: SCEnergyVerifier, text: str) -> np.ndarray:
    return verifier._feature_for_pair(_Pair(response=text, context=""))


def _sigmoid_array(values: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))).astype(np.float32)


def _sigmoid_scalar(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-max(-40.0, min(40.0, value)))))


def _binary_cross_entropy(labels: np.ndarray, probs: np.ndarray) -> float:
    eps = 1e-9
    return float(
        -np.mean(labels * np.log(probs + eps) + (1.0 - labels) * np.log(1.0 - probs + eps))
    )


def save_dvi_checkpoint(
    path: Path | str,
    *,
    metric: np.ndarray,
    bias: float,
    loss_history: Sequence[float],
    source_checkpoint_path: str | None,
) -> None:
    """Save the DVI verifier checkpoint to the exact requested ``.pt`` path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        np.savez(
            handle,
            metric=np.asarray(metric, dtype=np.float32),
            bias=np.asarray([bias], dtype=np.float32),
            loss_history=np.asarray(loss_history, dtype=np.float32),
            training_method=np.asarray([TRAINING_METHOD]),
            source_checkpoint_path=np.asarray([source_checkpoint_path or ""]),
        )


def build_artifact(
    *,
    positive_cases_used: int,
    negative_cases_used: int,
    training_result: DviTrainingResult,
    heldout_cases_used: int,
    train_rows_used: int,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    started_at: str | None = None,
    duration_s: float = 0.0,
    n_epochs: int = N_EPOCHS,
) -> dict[str, Any]:
    """Build and validate the final Exp 1381 DVI artifact."""

    baseline = round(float(training_result.baseline_auroc), 6)
    trained = round(float(training_result.trained_auroc), 6)
    delta = round(trained - baseline, 6)
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "started_at": started_at or datetime.now(tz=UTC).isoformat(),
        "finished_at": datetime.now(tz=UTC).isoformat(),
        "duration_s": round(float(duration_s), 3),
        "status": "complete",
        "spec": ["REQ-VERIFY-1381", "SCENARIO-VERIFY-1381"],
        "positive_cases_used": int(positive_cases_used),
        "negative_cases_used": int(negative_cases_used),
        "positive_case_source": "exp1374_primary_semantic_verified",
        "negative_case_source": "fover_incorrect_reasoning_steps",
        "negative_ratio_target": NEGATIVE_RATIO,
        "training_method": TRAINING_METHOD,
        "training_loss_history": [round(float(loss), 6) for loss in training_result.loss_history],
        "epochs_run": len(training_result.loss_history),
        "learning_rate": LEARNING_RATE,
        "l2_weight_decay": L2_WEIGHT_DECAY,
        "fover_split_seed": FOVER_SPLIT_SEED,
        "fover_train_rows_used": int(train_rows_used),
        "fover_heldout_cases_used": int(heldout_cases_used),
        "dvi_source_checkpoint_path": training_result.source_checkpoint_path,
        "dvi_source_checkpoint_kind": training_result.source_checkpoint_kind,
        "dvi_baseline_auroc": baseline,
        "dvi_trained_auroc": trained,
        "dvi_auroc_delta": delta,
        "dvi_checkpoint_path": training_result.checkpoint_path,
        "dvi_deployed": bool(training_result.checkpoint_saved),
        "discriminative_improvement_measured": True,
        "fresh_llm_inference_used": False,
        "model_specs_required": False,
        "honest_verdict": _honest_verdict(delta),
        "measurement_note": (
            "No fresh LLM inference was used. DVI trained Carnot's SC-Energy "
            "verifier boundary on Exp 1374 semantic positives versus FoVer "
            "incorrect contrastive negatives, then measured AUROC on the same "
            "fixed held-out FoVer split before and after training."
        ),
    }
    validate_artifact(artifact)
    return artifact


def _honest_verdict(delta: float | None) -> str:
    if delta is None:
        return "dvi_training_no_auroc_delta_measured"
    if delta > 0:
        return "dvi_discriminative_improvement_measured_positive_delta"
    if delta < 0:
        return "dvi_discriminative_improvement_measured_negative_delta"
    return "dvi_discriminative_improvement_measured_flat_delta"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Enforce the user-required DVI artifact fields and deployment contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] == "complete":
        if artifact["positive_cases_used"] <= 0:
            raise AssertionError("complete DVI artifact requires positive cases")
        if artifact["negative_cases_used"] < artifact["positive_cases_used"] * NEGATIVE_RATIO:
            raise AssertionError("complete DVI artifact violates 1:3 positive:negative ratio")
        if artifact["discriminative_improvement_measured"] and artifact["dvi_auroc_delta"] is None:
            raise AssertionError("measured DVI improvement requires an AUROC delta")
        if artifact["dvi_deployed"] and not Path(str(artifact["dvi_checkpoint_path"])).exists():
            raise AssertionError("dvi_deployed requires checkpoint path to exist")


def run(
    *,
    exp1374_path: Path | str = DEFAULT_EXP1374_PATH,
    fover_path: Path | str = DEFAULT_FOVER_PATH,
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    baseline_checkpoint_path: Path | str | None = None,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    n_epochs: int = N_EPOCHS,
) -> dict[str, Any]:
    """Run Exp 1381 end-to-end and write the final DVI artifact."""

    started_at = datetime.now(tz=UTC).isoformat()
    t0 = time.perf_counter()
    write_in_progress_artifact(out_path, project_root=project_root, run_date=run_date)
    exp1374_artifact = json.loads(Path(exp1374_path).read_text(encoding="utf-8"))
    positive_cases = load_positive_cases(exp1374_artifact)
    rows = load_jsonl_rows(fover_path)
    train_rows, holdout_rows = split_fover_rows(rows)
    negative_cases = select_negative_cases(
        train_rows,
        positive_count=len(positive_cases),
        ratio=NEGATIVE_RATIO,
    )
    training_result = run_dvi_training(
        positive_cases=positive_cases,
        negative_cases=negative_cases,
        holdout_rows=holdout_rows,
        checkpoint_path=checkpoint_path,
        models_dir=models_dir,
        baseline_checkpoint_path=baseline_checkpoint_path,
        n_epochs=n_epochs,
    )
    artifact = build_artifact(
        positive_cases_used=len(positive_cases),
        negative_cases_used=len(negative_cases),
        training_result=training_result,
        heldout_cases_used=len(holdout_rows),
        train_rows_used=len(train_rows),
        project_root=project_root,
        run_date=run_date,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
        n_epochs=n_epochs,
    )
    return _write_json(out_path, artifact)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
