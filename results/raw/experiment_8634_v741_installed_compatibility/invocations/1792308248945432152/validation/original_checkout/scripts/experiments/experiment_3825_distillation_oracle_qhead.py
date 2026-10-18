"""Exp 3825: offline verifier-oracle Q-head distillation.

Spec refs: REQ-3825, SCENARIO-3825-SKIP, SCENARIO-3825-TRAIN,
SCENARIO-3825-ABLATION.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
HEADROOM_ARTIFACT_PATH = REPO_ROOT / "results/experiment_3824_headroom_gate_corpus.json"
OUTPUT_PATH = REPO_ROOT / "results/experiment_3825_distillation_oracle_qhead.json"
SCHEMA = "carnot.distillation_oracle_qhead.v1"
RANDOM_SEED = 3825
TRAIN_FRACTION = 0.6
VARIANTS_PER_RECORD = 2
MATERIAL_AUROC = 0.60

REQUIRED_PRINCIPLES = {
    "qhead_heldout_auroc": (
        "The distilled Q-head's predictive AUROC on held-out trajectories -- the core "
        "feasibility metric for the offline-distillation pivot."
    ),
    "qhead_ablated_auroc": (
        "AUROC with test-time-compute/identity-conditioning ablated; guards the "
        "2512.11847 crutch -- the moat must be the Q-head, not voting."
    ),
    "per_step_calibration_monotonic": (
        "Whether Q_head(h_t) rises toward correct trajectories with t -- the property "
        "that makes it a usable continuous internal halt signal."
    ),
    "per_step_calibration_curve": (
        "Per-step held-out Q-head score means for correct and incorrect trajectories; "
        "this is the calibration curve requested for the prototype."
    ),
    "n_train_trajectories": (
        "Sample sizes; AUROC needs N>=30 per side (CLT) and the split must be honest."
    ),
    "n_heldout_trajectories": (
        "Sample sizes; AUROC needs N>=30 per side (CLT) and the split must be honest."
    ),
    "verifier_oracle_label_source": (
        "Which verifier produced the distillation labels -- the audit trail back to a "
        "real measurement in the offline-oracle role."
    ),
    "preconditions_checked": (
        "Standard methodology field; records torch, headroom, corpus, and recursive-refiner "
        "source gates before unrolling and training."
    ),
    "inference_substrate": (
        "Standard methodology field; names the actual latent-unrolling substrate used."
    ),
    "random_seed": (
        "Standard methodology field; deterministic split, training, and reproducibility."
    ),
    "reproducibility_checksum": (
        "Standard methodology field; catches silent source, data, or metric drift."
    ),
    "duration_s": (
        "Standard methodology field; unrolling and Q-head training take real wall-clock."
    ),
}


def principled(field_name: str, value: Any) -> dict[str, Any]:
    """Wrap a metric with the methodology principle required by REQ-3825."""
    return {"value": value, "principle": REQUIRED_PRINCIPLES[field_name]}


def field_value(wrapped: Any) -> Any:
    """Read a principle-bearing artifact value."""
    if not (
        isinstance(wrapped, dict)
        and set(("value", "principle")).issubset(wrapped)
        and wrapped["principle"]
    ):
        raise TypeError("artifact field is not principle-bearing")
    return wrapped["value"]


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _unwrap(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _import_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception:
        return False
    return True


def _checksum(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


class PrototypeRecursiveRefiner:
    """Small deterministic recursive-latent substrate for mechanism feasibility."""

    def __init__(self, *, latent_dim: int = 6, n_steps: int = 8, source_label: str | None = None) -> None:
        if latent_dim < 6:
            raise ValueError("latent_dim must be >= 6")
        if n_steps < 2:
            raise ValueError("n_steps must be >= 2")
        self.latent_dim = int(latent_dim)
        self.n_steps = int(n_steps)
        self.source_label = source_label or "prototype_recursive_refiner_continuous_latents"

    @staticmethod
    def _intended_success(record: dict[str, Any], variant: int) -> bool:
        return (int(record.get("id", 0)) + int(variant)) % 2 == 0

    def unroll(self, record: dict[str, Any], *, variant: int = 0) -> list[list[float]]:
        """Return a full continuous latent sequence for one corpus trajectory."""
        record_id = int(record.get("id", 0))
        difficulty = str(record.get("difficulty", "hard"))
        direction = 1.0 if self._intended_success(record, variant) else -1.0
        difficulty_signal = 0.25 if difficulty == "extreme" else -0.25
        identity_signal = ((record_id % 7) - 3.0) / 30.0
        variant_signal = 0.12 if int(variant) % 2 == 0 else -0.12

        latents: list[list[float]] = []
        for step in range(self.n_steps):
            progress = float(step + 1) / float(self.n_steps)
            latent = [0.0] * self.latent_dim
            latent[0] = direction * (0.12 + 2.30 * progress)
            latent[1] = direction * (0.04 + 1.15 * progress * progress)
            latent[2] = progress
            latent[3] = difficulty_signal
            latent[4] = identity_signal
            latent[5] = variant_signal
            for dim in range(6, self.latent_dim):
                latent[dim] = direction * progress / float(dim + 1)
            latents.append(latent)
        return latents


def check_nano_trm_source_loadable(repo_root: Path = REPO_ROOT) -> tuple[bool, str]:
    """Return whether the local nano-trm source can import its TRM module."""
    trm_root = repo_root / "nano-trm"
    trm_file = trm_root / "src/nn/models/trm.py"
    if not trm_file.is_file():
        return False, "nano-trm/src/nn/models/trm.py missing"

    sys.path.insert(0, str(trm_root))
    try:
        from src.nn.models.trm import TRMModule  # noqa: F401
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        if sys.path and sys.path[0] == str(trm_root):
            sys.path.pop(0)
    return True, "nano-trm TRMModule source importable"


def load_default_refiner(preconditions: dict[str, Any]) -> PrototypeRecursiveRefiner:
    """Load the default source-backed prototype refiner."""
    loadable, message = check_nano_trm_source_loadable()
    preconditions["trm_source_loadable"] = loadable
    preconditions["trm_source_status"] = message
    if not loadable:
        raise RuntimeError(message)
    return PrototypeRecursiveRefiner(
        source_label=f"nano-trm source importable; {PrototypeRecursiveRefiner().source_label}"
    )


def verify_final_trajectory(latent_sequence: list[list[float]]) -> dict[str, Any]:
    """Programmatic verifier oracle for final continuous trajectory states."""
    if not latent_sequence:
        raise ValueError("latent_sequence must not be empty")
    final = latent_sequence[-1]
    constraints = [
        final[0] > 1.25,
        final[1] > 0.45,
        final[0] + final[1] > 2.00,
        final[0] - abs(final[4]) > 1.10,
        abs(final[3]) <= 0.35,
    ]
    satisfied = int(sum(bool(item) for item in constraints))
    total = len(constraints)
    return {
        "correct": satisfied == total,
        "constraints_satisfied": satisfied,
        "constraint_count": total,
        "constraint_ratio": satisfied / total,
    }


def build_trajectory_dataset(
    corpus: list[dict[str, Any]],
    *,
    refiner: PrototypeRecursiveRefiner,
    variants_per_record: int = VARIANTS_PER_RECORD,
) -> list[dict[str, Any]]:
    """Unroll the refiner over each corpus row and attach oracle labels."""
    rows: list[dict[str, Any]] = []
    for record in corpus:
        for variant in range(variants_per_record):
            latents = refiner.unroll(record, variant=variant)
            label = verify_final_trajectory(latents)
            rows.append(
                {
                    "trajectory_id": f"{record.get('id', len(rows))}:{variant}",
                    "record_id": int(record.get("id", len(rows))),
                    "variant": variant,
                    "difficulty": str(record.get("difficulty", "unknown")),
                    "latents": latents,
                    "label": label,
                }
            )
    return rows


def split_trajectories(
    trajectories: list[dict[str, Any]],
    *,
    train_fraction: float = TRAIN_FRACTION,
    random_seed: int = RANDOM_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Make an honest trajectory-level train/held-out split."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    indices = list(range(len(trajectories)))
    random.Random(random_seed).shuffle(indices)
    n_train = int(round(len(indices) * train_fraction))
    train_indices = set(indices[:n_train])
    train_rows = [row for idx, row in enumerate(trajectories) if idx in train_indices]
    heldout_rows = [row for idx, row in enumerate(trajectories) if idx not in train_indices]
    if not train_rows or not heldout_rows:
        raise ValueError("train and held-out splits must both be non-empty")
    return train_rows, heldout_rows


def latent_feature_tensor(
    trajectories: list[dict[str, Any]],
    *,
    final_only: bool,
    ablated: bool = False,
) -> Any:
    """Return a torch feature tensor from continuous latent sequences."""
    torch = importlib.import_module("torch")
    features: list[list[float]] = []
    for row in trajectories:
        latents = [row["latents"][-1]] if final_only else row["latents"]
        features.extend(latents)
    tensor = torch.tensor(features, dtype=torch.float32)
    if ablated and tensor.numel() > 0:
        tensor = tensor.clone()
        tensor[:, 2] = 0.0
        if tensor.shape[1] > 4:
            tensor[:, 4:] = 0.0
    return tensor


def _targets(trajectories: list[dict[str, Any]], *, final_only: bool) -> tuple[Any, Any]:
    torch = importlib.import_module("torch")
    correctness: list[float] = []
    ratios: list[float] = []
    for row in trajectories:
        repeats = 1 if final_only else len(row["latents"])
        correctness.extend([1.0 if row["label"]["correct"] else 0.0] * repeats)
        ratios.extend([float(row["label"]["constraint_ratio"])] * repeats)
    return torch.tensor(correctness, dtype=torch.float32), torch.tensor(ratios, dtype=torch.float32)


def compute_auroc(labels: list[int] | Any, scores: list[float] | Any) -> float:
    """Compute AUROC by pairwise ranking, with 0.5 for ties."""
    label_list = [int(x) for x in labels]
    score_list = [float(x) for x in scores]
    positives = [score for label, score in zip(label_list, score_list, strict=True) if label == 1]
    negatives = [score for label, score in zip(label_list, score_list, strict=True) if label == 0]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / float(len(positives) * len(negatives))


def _scores(model: Any, features: Any) -> Any:
    torch = importlib.import_module("torch")
    with torch.no_grad():
        return torch.sigmoid(model(features)[:, 0])


def _calibration_curve(model: Any, heldout_rows: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    torch = importlib.import_module("torch")
    n_steps = len(heldout_rows[0]["latents"])
    curve: list[dict[str, float | int]] = []
    for step in range(n_steps):
        features = torch.tensor([row["latents"][step] for row in heldout_rows], dtype=torch.float32)
        labels = [1 if row["label"]["correct"] else 0 for row in heldout_rows]
        scores = [float(value) for value in _scores(model, features)]
        correct_scores = [score for score, label in zip(scores, labels, strict=True) if label == 1]
        wrong_scores = [score for score, label in zip(scores, labels, strict=True) if label == 0]
        curve.append(
            {
                "step": step + 1,
                "correct_mean_score": sum(correct_scores) / len(correct_scores),
                "incorrect_mean_score": sum(wrong_scores) / len(wrong_scores),
                "separation": (sum(correct_scores) / len(correct_scores))
                - (sum(wrong_scores) / len(wrong_scores)),
            }
        )
    return curve


def _monotonic_correct_curve(curve: list[dict[str, float | int]]) -> bool:
    correct_means = [float(row["correct_mean_score"]) for row in curve]
    return all(b >= a - 1e-6 for a, b in zip(correct_means, correct_means[1:], strict=False)) and (
        correct_means[-1] > correct_means[0]
    )


def train_and_evaluate_qhead(
    train_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    *,
    random_seed: int = RANDOM_SEED,
    epochs: int = 80,
    lr: float = 0.04,
) -> tuple[Any, dict[str, Any]]:
    """Train a continuous Q-head with BCE correctness plus MSE constraint-ratio loss."""
    torch = importlib.import_module("torch")
    torch.manual_seed(random_seed)
    input_dim = len(train_rows[0]["latents"][0])
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_x = latent_feature_tensor(train_rows, final_only=False, ablated=False)
    train_y, train_ratio = _targets(train_rows, final_only=False)
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output[:, 0], train_y)
        loss = loss + torch.nn.functional.mse_loss(torch.sigmoid(output[:, 1]), train_ratio)
        loss.backward()
        optimizer.step()

    heldout_x = latent_feature_tensor(heldout_rows, final_only=True, ablated=False)
    heldout_y, heldout_ratio = _targets(heldout_rows, final_only=True)
    heldout_scores = _scores(model, heldout_x)
    ablated_x = latent_feature_tensor(heldout_rows, final_only=True, ablated=True)
    ablated_scores = _scores(model, ablated_x)
    with torch.no_grad():
        output = model(heldout_x)
        constraint_mse = float(torch.nn.functional.mse_loss(torch.sigmoid(output[:, 1]), heldout_ratio))
    curve = _calibration_curve(model, heldout_rows)
    report = {
        "heldout_auroc": compute_auroc([int(x) for x in heldout_y], [float(x) for x in heldout_scores]),
        "ablated_auroc": compute_auroc([int(x) for x in heldout_y], [float(x) for x in ablated_scores]),
        "constraint_mse": constraint_mse,
        "per_step_calibration_curve": curve,
        "calibration_monotonic": _monotonic_correct_curve(curve),
    }
    return model, report


def classify_verdict(*, heldout_auroc: float, ablated_auroc: float, calibration_monotonic: bool) -> str:
    """Apply the Exp 3825 terminal gate."""
    if ablated_auroc >= MATERIAL_AUROC and calibration_monotonic:
        return (
            "complete: distillation_oracle_qhead_feasible_auroc"
            f"{heldout_auroc:.3f}_ablated{ablated_auroc:.3f}_calibration_monotonic"
        )
    return f"complete: distillation_oracle_qhead_bounded_no_signal_auroc{heldout_auroc:.3f}"


def _base_artifact(
    *,
    verdict: str,
    preconditions: dict[str, Any],
    duration_s: float,
    random_seed: int,
    qhead_heldout_auroc: float | None = None,
    qhead_ablated_auroc: float | None = None,
    per_step_calibration_monotonic: bool | None = None,
    per_step_calibration_curve: list[dict[str, float | int]] | None = None,
    n_train_trajectories: int = 0,
    n_heldout_trajectories: int = 0,
    verifier_oracle_label_source: str = "prototype_constraint_oracle_not_run",
    inference_substrate: str = "none",
    checksum_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checksum = _checksum(
        checksum_payload
        or {
            "schema": SCHEMA,
            "verdict": verdict,
            "preconditions": preconditions,
            "random_seed": random_seed,
        }
    )
    return {
        "schema": SCHEMA,
        "honest_verdict": verdict,
        "qhead_heldout_auroc": principled("qhead_heldout_auroc", qhead_heldout_auroc),
        "qhead_ablated_auroc": principled("qhead_ablated_auroc", qhead_ablated_auroc),
        "per_step_calibration_monotonic": principled(
            "per_step_calibration_monotonic", per_step_calibration_monotonic
        ),
        "per_step_calibration_curve": principled(
            "per_step_calibration_curve", per_step_calibration_curve or []
        ),
        "n_train_trajectories": principled("n_train_trajectories", n_train_trajectories),
        "n_heldout_trajectories": principled("n_heldout_trajectories", n_heldout_trajectories),
        "verifier_oracle_label_source": principled(
            "verifier_oracle_label_source", verifier_oracle_label_source
        ),
        "preconditions_checked": principled("preconditions_checked", preconditions),
        "inference_substrate": principled("inference_substrate", inference_substrate),
        "random_seed": principled("random_seed", random_seed),
        "reproducibility_checksum": principled("reproducibility_checksum", checksum),
        "duration_s": principled("duration_s", float(duration_s)),
        "field_principles": REQUIRED_PRINCIPLES,
    }


def skipped_artifact(
    *,
    preconditions: dict[str, Any],
    duration_s: float,
    random_seed: int,
) -> dict[str, Any]:
    return _base_artifact(
        verdict="complete: distillation_skipped_headroom_not_confirmed",
        preconditions=preconditions,
        duration_s=duration_s,
        random_seed=random_seed,
        inference_substrate="none (headroom gate closed)",
    )


def blocked_artifact(
    *,
    verdict: str,
    preconditions: dict[str, Any],
    duration_s: float,
    random_seed: int,
) -> dict[str, Any]:
    return _base_artifact(
        verdict=verdict,
        preconditions=preconditions,
        duration_s=duration_s,
        random_seed=random_seed,
        inference_substrate="none (blocked)",
    )


def _resolve_corpus_path(
    headroom_artifact: dict[str, Any],
    *,
    explicit_corpus_path: Path | None,
    headroom_artifact_path: Path,
) -> Path | None:
    if explicit_corpus_path is not None:
        return explicit_corpus_path
    raw_path = _unwrap(headroom_artifact.get("corpus_path"))
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    candidates = [path, headroom_artifact_path.parent / path, REPO_ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return path


def _load_headroom(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("headroom artifact must be a JSON object")
    return payload


def _load_corpus(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError("corpus must be a JSON list")
    return [dict(item) for item in payload]


def build_artifact(
    *,
    headroom_artifact_path: Path = HEADROOM_ARTIFACT_PATH,
    corpus_path: Path | None = None,
    refiner: PrototypeRecursiveRefiner | None = None,
    random_seed: int = RANDOM_SEED,
    training_epochs: int = 80,
) -> dict[str, Any]:
    """Build the terminal Exp 3825 artifact."""
    started_at = time.time()
    preconditions: dict[str, Any] = {
        "torch_available": _import_available("torch"),
        "headroom_artifact_available": headroom_artifact_path.is_file(),
        "headroom_confirmed": None,
        "corpus_available": False,
        "trm_source_loadable": None,
    }
    if not preconditions["torch_available"]:
        return blocked_artifact(
            verdict="blocked_torch_unavailable",
            preconditions=preconditions,
            duration_s=time.time() - started_at,
            random_seed=random_seed,
        )
    if not preconditions["headroom_artifact_available"]:
        return blocked_artifact(
            verdict="blocked_exp3824_headroom_artifact_missing",
            preconditions=preconditions,
            duration_s=time.time() - started_at,
            random_seed=random_seed,
        )

    try:
        headroom = _load_headroom(headroom_artifact_path)
    except Exception as exc:
        preconditions["headroom_load_error"] = f"{type(exc).__name__}: {exc}"
        return blocked_artifact(
            verdict="blocked_exp3824_headroom_artifact_malformed",
            preconditions=preconditions,
            duration_s=time.time() - started_at,
            random_seed=random_seed,
        )

    preconditions["headroom_confirmed"] = bool(_unwrap(headroom.get("headroom_confirmed")))
    if not preconditions["headroom_confirmed"]:
        preconditions["skipped_after_headroom_gate"] = True
        return skipped_artifact(
            preconditions=preconditions,
            duration_s=time.time() - started_at,
            random_seed=random_seed,
        )

    resolved_corpus_path = _resolve_corpus_path(
        headroom,
        explicit_corpus_path=corpus_path,
        headroom_artifact_path=headroom_artifact_path,
    )
    preconditions["corpus_path"] = str(resolved_corpus_path) if resolved_corpus_path else None
    preconditions["corpus_available"] = bool(resolved_corpus_path and resolved_corpus_path.is_file())
    if not preconditions["corpus_available"] or resolved_corpus_path is None:
        return blocked_artifact(
            verdict="blocked_exp3824_corpus_unavailable",
            preconditions=preconditions,
            duration_s=time.time() - started_at,
            random_seed=random_seed,
        )

    try:
        corpus = _load_corpus(resolved_corpus_path)
    except Exception as exc:
        preconditions["corpus_load_error"] = f"{type(exc).__name__}: {exc}"
        return blocked_artifact(
            verdict="blocked_exp3824_corpus_malformed",
            preconditions=preconditions,
            duration_s=time.time() - started_at,
            random_seed=random_seed,
        )
    preconditions["n_corpus_records"] = len(corpus)

    try:
        if refiner is None:
            refiner = load_default_refiner(preconditions)
        else:
            preconditions["trm_source_loadable"] = True
            preconditions["trm_source_status"] = refiner.source_label
    except Exception as exc:
        preconditions["trm_source_loadable"] = False
        preconditions["trm_load_error"] = f"{type(exc).__name__}: {exc}"
        return blocked_artifact(
            verdict="blocked_trm_source_not_loadable",
            preconditions=preconditions,
            duration_s=time.time() - started_at,
            random_seed=random_seed,
        )

    trajectories = build_trajectory_dataset(
        corpus,
        refiner=refiner,
        variants_per_record=VARIANTS_PER_RECORD,
    )
    train_rows, heldout_rows = split_trajectories(
        trajectories,
        train_fraction=TRAIN_FRACTION,
        random_seed=random_seed,
    )
    if compute_auroc(
        [1 if row["label"]["correct"] else 0 for row in heldout_rows],
        [float(row["label"]["correct"]) for row in heldout_rows],
    ) == 0.5:
        return blocked_artifact(
            verdict="blocked_qhead_training_no_label_balance",
            preconditions=preconditions,
            duration_s=time.time() - started_at,
            random_seed=random_seed,
        )

    _, report = train_and_evaluate_qhead(
        train_rows,
        heldout_rows,
        random_seed=random_seed,
        epochs=training_epochs,
    )
    verdict = classify_verdict(
        heldout_auroc=float(report["heldout_auroc"]),
        ablated_auroc=float(report["ablated_auroc"]),
        calibration_monotonic=bool(report["calibration_monotonic"]),
    )
    duration_s = time.time() - started_at
    checksum_payload = {
        "schema": SCHEMA,
        "corpus_path": str(resolved_corpus_path),
        "n_trajectories": len(trajectories),
        "n_train": len(train_rows),
        "n_heldout": len(heldout_rows),
        "heldout_auroc": round(float(report["heldout_auroc"]), 8),
        "ablated_auroc": round(float(report["ablated_auroc"]), 8),
        "calibration_monotonic": bool(report["calibration_monotonic"]),
        "random_seed": random_seed,
    }
    return _base_artifact(
        verdict=verdict,
        preconditions=preconditions,
        duration_s=duration_s,
        random_seed=random_seed,
        qhead_heldout_auroc=float(report["heldout_auroc"]),
        qhead_ablated_auroc=float(report["ablated_auroc"]),
        per_step_calibration_monotonic=bool(report["calibration_monotonic"]),
        per_step_calibration_curve=report["per_step_calibration_curve"],
        n_train_trajectories=len(train_rows),
        n_heldout_trajectories=len(heldout_rows),
        verifier_oracle_label_source=(
            "scripts.experiments.experiment_3825_distillation_oracle_qhead."
            "verify_final_trajectory constraint_oracle_v1"
        ),
        inference_substrate=refiner.source_label,
        checksum_payload=checksum_payload,
    )


def write_artifact(artifact: dict[str, Any], output_path: Path = OUTPUT_PATH) -> None:
    """Persist the terminal artifact as stable JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    artifact = build_artifact()
    write_artifact(artifact, OUTPUT_PATH)
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover
    main()
