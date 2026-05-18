"""HALT cached-logprob hallucination-risk probe.

HALT uses residual probes on hidden states at question tokens.  The live SOTA
telemetry available to Carnot does not persist those hidden states, so this
Tier 0j prototype works from cached token logprobs and top-k alternatives:
proxy A uses top-k logprob variance, proxy B uses softmax(top-k) kurtosis, and
proxy C trains a small LogisticRegression probe on those logprob features.

Spec: REQ-TIER0-008, Exp 2394.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from carnot.verify.semantic_energy import binary_auroc

DEFAULT_MANIFEST_PATH = Path("results/live_sota_balanced_telemetry_manifest_1480.jsonl")
DEFAULT_OUTPUT_PATH = Path("results/experiment_2394_halt_tier0j.json")
DEFAULT_RANDOM_SEED = 42
SEMANTIC_ENERGY_BASELINE_AUROC = 0.685

_STAT_NAMES = ("mean", "std", "min", "max", "slope")
_FEATURE_GROUPS = (
    "topk_variance",
    "topk_gap",
    "topk_entropy",
    "topk_softmax_kurtosis",
    "topk_marginal_energy",
    "token_logprob",
    "token_nll",
)
HALT_FEATURE_NAMES = tuple(
    f"{group}_{stat_name}" for group in _FEATURE_GROUPS for stat_name in _STAT_NAMES
)

JsonDict = dict[str, Any]


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_series(values: Iterable[Any]) -> list[float]:
    return [number for value in values if (number := _finite_float(value)) is not None]


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = _mean(values)
    return float(math.sqrt(sum((value - center) ** 2 for value in values) / len(values)))


def _linear_slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    x_center = (len(values) - 1) / 2.0
    y_center = _mean(values)
    numerator = sum((idx - x_center) * (value - y_center) for idx, value in enumerate(values))
    denominator = sum((idx - x_center) ** 2 for idx in range(len(values)))
    return float(numerator / denominator) if denominator else 0.0


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {stat_name: 0.0 for stat_name in _STAT_NAMES}
    return {
        "mean": _mean(values),
        "std": _std(values),
        "min": float(min(values)),
        "max": float(max(values)),
        "slope": _linear_slope(values),
    }


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return float(maximum + math.log(sum(math.exp(value - maximum) for value in values)))


def _softmax_from_logprobs(logprobs: Sequence[float]) -> list[float]:
    local_logsum = _logsumexp(logprobs)
    return [float(math.exp(value - local_logsum)) for value in logprobs]


def _pearson_kurtosis(values: Sequence[float]) -> float:
    """Return ordinary kurtosis, not excess kurtosis, for a small probability vector.

    Proxy B's signal is the heaviness of the renormalized top-k probability
    distribution.  A zero-variance distribution has no usable tail information,
    so it contributes 0.0 instead of an infinite or NaN value.

    Spec: REQ-TIER0-008-2
    """
    if len(values) < 2:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    centered = array - float(np.mean(array))
    variance = float(np.mean(centered * centered))
    if variance <= 1e-12:
        return 0.0
    return float(np.mean(centered**4) / (variance * variance))


def _safe_sigmoid(value: float) -> float:
    bounded = max(min(float(value), 60.0), -60.0)
    return float(1.0 / (1.0 + math.exp(-bounded)))


def _topk_logprob_positions(entry: JsonDict) -> list[list[float]]:
    positions: list[list[float]] = []
    for alternatives in entry.get("top_logprobs") or []:
        if isinstance(alternatives, dict):
            values = _finite_series(alternatives.values())
            if values:
                positions.append(values)
    return positions


def extract_logprob_features(entry: JsonDict) -> dict[str, float]:
    """Extract fixed-width logprob features for HALT proxy A/B/C scoring.

    The feature vector deliberately uses logprob-shape statistics instead of
    prompt text or response text.  Counts are reported for diagnostics but are
    not part of `HALT_FEATURE_NAMES`, so the proxy C training matrix is driven by
    the observed probability geometry.

    Spec: REQ-TIER0-008-1, REQ-TIER0-008-2
    """
    positions = _topk_logprob_positions(entry)
    token_logprobs = _finite_series(entry.get("token_logprobs") or [])

    topk_variance: list[float] = []
    topk_gap: list[float] = []
    topk_entropy: list[float] = []
    topk_softmax_kurtosis: list[float] = []
    topk_marginal_energy: list[float] = []

    for logprobs in positions:
        sorted_logprobs = sorted(logprobs, reverse=True)
        probabilities = _softmax_from_logprobs(sorted_logprobs)
        topk_variance.append(float(np.var(np.asarray(sorted_logprobs, dtype=np.float64))))
        topk_gap.append(
            float(sorted_logprobs[0] - sorted_logprobs[1])
            if len(sorted_logprobs) > 1
            else 0.0
        )
        topk_entropy.append(
            float(-sum(prob * math.log(prob) for prob in probabilities if prob > 0.0))
        )
        topk_softmax_kurtosis.append(_pearson_kurtosis(probabilities))
        topk_mass = min(max(sum(math.exp(value) for value in sorted_logprobs), 0.0), 1.0)
        topk_marginal_energy.append(float(-math.log(max(topk_mass, 1e-12))))

    grouped = {
        "topk_variance": topk_variance,
        "topk_gap": topk_gap,
        "topk_entropy": topk_entropy,
        "topk_softmax_kurtosis": topk_softmax_kurtosis,
        "topk_marginal_energy": topk_marginal_energy,
        "token_logprob": token_logprobs,
        "token_nll": [-value for value in token_logprobs],
    }

    features: dict[str, float] = {
        "topk_position_count": float(len(positions)),
        "token_logprob_count": float(len(token_logprobs)),
    }
    for group_name, values in grouped.items():
        for stat_name, stat_value in _summary(values).items():
            features[f"{group_name}_{stat_name}"] = float(stat_value)
    return features


def _feature_matrix(entries: Sequence[JsonDict]) -> np.ndarray:
    rows = []
    for entry in entries:
        features = extract_logprob_features(entry)
        rows.append([features[name] for name in HALT_FEATURE_NAMES])
    return np.asarray(rows, dtype=np.float64)


def label_from_entry(entry: JsonDict) -> int:
    """Return 1 for hallucination/incorrect rows and 0 for factual/correct rows.

    Spec: REQ-TIER0-008-5
    """
    correctness = str(entry.get("correctness_label", "")).strip().lower()
    if correctness == "incorrect":
        return 1
    if correctness == "correct":
        return 0
    if entry.get("correct") is False:
        return 1
    if entry.get("correct") is True:
        return 0
    raise ValueError("entry does not contain a binary correctness label")


class HaltProbeDetector:
    """Tier 0j HALT proxy detector for cached logprob telemetry.

    Args:
        threshold: Decision threshold over `halt_risk_score`.  Scores are
            probabilities when proxy C is fitted and sigmoid-normalized proxy
            scores otherwise.
        random_seed: Seed passed to scikit-learn LogisticRegression for proxy C.
    """

    def __init__(self, threshold: float = 0.5, random_seed: int = DEFAULT_RANDOM_SEED) -> None:
        self.threshold = float(threshold)
        self.random_seed = int(random_seed)
        self._model: Any | None = None

    def fit(self, entries: Sequence[JsonDict], labels: Sequence[int]) -> HaltProbeDetector:
        """Fit proxy C, a small LogisticRegression probe over logprob features.

        Spec: REQ-TIER0-008-3
        """
        if len(entries) != len(labels):
            raise ValueError("entries and labels must have the same length")
        if len(set(int(label) for label in labels)) < 2:
            raise ValueError("labels must include both classes")

        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
        from sklearn.pipeline import make_pipeline  # noqa: PLC0415
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=self.random_seed,
                solver="liblinear",
            ),
        )
        model.fit(_feature_matrix(entries), np.asarray(labels, dtype=np.int64))
        self._model = model
        return self

    def compute_halt_score(self, entry: JsonDict) -> float:
        """Return a finite hallucination-risk score for one telemetry row.

        Spec: REQ-TIER0-008-1
        """
        score, _proxy = self._score_entry(entry)
        if not math.isfinite(score):
            raise ValueError("HALT risk score must be finite")
        return float(score)

    def verify(self, entry: JsonDict) -> JsonDict:
        """Return HALT proxy score, thresholded risk flag, and proxy identity.

        Spec: REQ-TIER0-008-4
        """
        score, proxy_used = self._score_entry(entry)
        return {
            "halt_risk_score": float(score),
            "is_high_risk": bool(score >= self.threshold),
            "proxy_used": proxy_used,
        }

    def _score_entry(self, entry: JsonDict) -> tuple[float, str]:
        features = extract_logprob_features(entry)
        if self._model is not None:
            vector = np.asarray([[features[name] for name in HALT_FEATURE_NAMES]], dtype=np.float64)
            probability = float(self._model.predict_proba(vector)[0][1])
            return min(max(probability, 0.0), 1.0), "C"

        if features["topk_position_count"] > 0.0:
            variance = features["topk_variance_mean"]
            kurtosis = features["topk_softmax_kurtosis_mean"]
            nll = features["token_nll_mean"]
            raw_score = 0.08 * (variance - 15.0) + 1.25 * (kurtosis - 3.0) + 0.50 * nll
            proxy = "A+B" if kurtosis > 0.0 else "A"
            return _safe_sigmoid(raw_score), proxy

        if features["token_logprob_count"] > 0.0:
            raw_score = features["token_nll_mean"] + 0.5 * features["token_nll_max"]
            return _safe_sigmoid(raw_score - 0.5), "token_logprobs"

        raise ValueError("entry must contain top_logprobs or token_logprobs")


def compute_halt_score(entry: JsonDict) -> float:
    """Convenience wrapper around the default training-free detector."""
    return HaltProbeDetector().compute_halt_score(entry)


def verify(entry: JsonDict) -> JsonDict:
    """Convenience wrapper around the default training-free detector."""
    return HaltProbeDetector().verify(entry)


def oof_halt_risk_scores(
    entries: Sequence[JsonDict],
    labels: Sequence[int],
    *,
    random_seed: int = DEFAULT_RANDOM_SEED,
    n_splits: int = 6,
) -> list[float]:
    """Return deterministic out-of-fold proxy C risk scores.

    The small 36-row Exp 2394 evaluation should not train and score the same row
    with the same LogisticRegression fit.  This helper therefore fits proxy C on
    stratified folds and reports the held-out probability for each row.

    Spec: REQ-TIER0-008-3, REQ-TIER0-008-5
    """
    if len(entries) != len(labels):
        raise ValueError("entries and labels must have the same length")
    label_counts = Counter(int(label) for label in labels)
    if len(label_counts) < 2:
        raise ValueError("labels must include both classes")
    fold_count = min(int(n_splits), min(label_counts.values()))
    if fold_count < 2:
        raise ValueError("at least two examples per class are required for out-of-fold scoring")

    from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415

    labels_array = np.asarray(labels, dtype=np.int64)
    scores = np.zeros(len(entries), dtype=np.float64)
    splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=random_seed)
    indices = np.arange(len(entries))
    for train_idx, test_idx in splitter.split(indices, labels_array):
        train_entries = [entries[idx] for idx in train_idx]
        train_labels = [int(labels_array[idx]) for idx in train_idx]
        detector = HaltProbeDetector(random_seed=random_seed).fit(train_entries, train_labels)
        for idx in test_idx:
            scores[idx] = detector.compute_halt_score(entries[int(idx)])
    return [float(score) for score in scores]


def _read_jsonl(path: Path, limit: int | None = None) -> list[JsonDict]:
    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def _sklearn_precondition() -> dict[str, Any]:
    try:
        import sklearn  # noqa: PLC0415
    except ModuleNotFoundError:
        return {"sklearn_importable": False, "sklearn_version": None}
    return {"sklearn_importable": True, "sklearn_version": sklearn.__version__}


def _preconditions(manifest_path: Path) -> dict[str, Any]:
    checked = _sklearn_precondition()
    checked["telemetry_manifest_present"] = manifest_path.is_file()
    checked["telemetry_manifest_path"] = str(manifest_path)
    checked["telemetry_fields"] = []
    if manifest_path.is_file():
        rows = _read_jsonl(manifest_path, limit=1)
        checked["telemetry_fields"] = list(rows[0].keys()) if rows else []
    return checked


def build_experiment_artifact(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    n_eval_examples: int = 36,
    random_seed: int = DEFAULT_RANDOM_SEED,
    semantic_energy_baseline: float = SEMANTIC_ENERGY_BASELINE_AUROC,
) -> JsonDict:
    """Evaluate HALT proxy C on the first 36 cached telemetry entries.

    Spec: REQ-TIER0-008-5
    """
    start = time.perf_counter()
    manifest = Path(manifest_path)
    checked = _preconditions(manifest)
    if not checked["telemetry_manifest_present"]:
        return {
            "status": "blocked",
            "honest_verdict": "blocked_telemetry_manifest_missing",
            "halt_k19j_validated": False,
            "halt_k19j_auroc": None,
            "halt_k19j_mean_risk_score": None,
            "halt_vs_semantic_energy_auroc_delta": None,
            "halt_proxy_used": None,
            "n_eval_examples": 0,
            "verifier_tier": "0j",
            "random_seed": random_seed,
            "duration_s": round(time.perf_counter() - start, 6),
            "preconditions_checked": checked,
        }
    if not checked["sklearn_importable"]:
        raise ModuleNotFoundError("scikit-learn is required for HALT proxy C")

    entries = _read_jsonl(manifest, limit=n_eval_examples)
    labels = [label_from_entry(entry) for entry in entries]
    scores = oof_halt_risk_scores(entries, labels, random_seed=random_seed)
    auroc = binary_auroc(labels, scores)
    mean_risk = float(np.mean(np.asarray(scores, dtype=np.float64))) if scores else 0.0
    nontrivial = len({round(score, 12) for score in scores}) > 1
    validated = bool(nontrivial and len(entries) >= 30 and math.isfinite(float(auroc)))
    duration_s = round(time.perf_counter() - start, 6)

    return {
        "status": "complete",
        "experiment": 2394,
        "title": "HALT Tier 0j cached-logprob proxy validation",
        "module_path": "python/carnot/verify/halt_probe.py",
        "spec_refs": ["REQ-TIER0-008"],
        "field_principles": {
            "honest_verdict": "Terminal-prefix required. complete: with AUROC result.",
            "halt_k19j_validated": (
                "True if HaltProbeDetector ran and produced non-trivial risk scores."
            ),
            "halt_k19j_auroc": (
                "Primary metric - compare with baseline 0.685. No fabrication."
            ),
            "halt_vs_semantic_energy_auroc_delta": (
                "HALT - SemanticEnergy delta. Positive = improvement over baseline."
            ),
            "halt_proxy_used": (
                "Records which proxy strategy was used (A/B/C) for reproducibility."
            ),
            "n_eval_examples": "Must be 36 for direct comparison with exp2351.",
            "verifier_tier": "Must be '0j'.",
            "random_seed": "Must be 42 for reproducibility.",
            "duration_s": "Guards against fabrication.",
            "preconditions_checked": "Records sklearn + telemetry manifest check.",
        },
        "honest_verdict": (
            "complete: HaltProbeDetector Tier 0j proxy C ran on "
            f"{len(entries)} cached telemetry entries; AUROC={float(auroc):.6f}."
        ),
        "halt_k19j_validated": validated,
        "halt_k19j_auroc": float(auroc),
        "halt_k19j_mean_risk_score": mean_risk,
        "halt_vs_semantic_energy_auroc_delta": float(auroc - semantic_energy_baseline),
        "semantic_energy_baseline_auroc": float(semantic_energy_baseline),
        "halt_proxy_used": "C",
        "halt_proxy_detail": "out_of_fold_logistic_regression_on_cached_logprob_features",
        "n_eval_examples": len(entries),
        "n_factual_examples": int(labels.count(0)),
        "n_hallucination_examples": int(labels.count(1)),
        "verifier_tier": "0j",
        "random_seed": int(random_seed),
        "duration_s": duration_s,
        "preconditions_checked": checked,
        "score_direction": "higher_score_means_more_hallucination_like",
        "score_field": "halt_risk_score",
        "score_summary": {
            "min": float(min(scores)),
            "max": float(max(scores)),
            "mean": mean_risk,
            "std": float(np.std(np.asarray(scores, dtype=np.float64))),
        },
        "source_artifact": str(manifest),
        "evaluation_design": (
            "Load the first 36 live SOTA balanced telemetry rows and compute "
            "stratified out-of-fold LogisticRegression risk scores from cached "
            "logprob-only HALT proxy features."
        ),
        "acceptance_gates": {
            "halt_k19j_validated": validated,
            "n_eval_examples_gte_30": len(entries) >= 30,
        },
    }


def write_experiment_artifact(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> JsonDict:
    artifact = build_experiment_artifact(manifest_path=manifest_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
