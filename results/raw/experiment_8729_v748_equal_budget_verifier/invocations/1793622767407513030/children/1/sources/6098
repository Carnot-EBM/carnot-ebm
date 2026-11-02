"""HIVE-style soft-vote ensemble over cached Tier 0 verifier scores.

The ensemble discovers the four Tier 0 verifier modules named by Exp 2398 and
uses whichever are importable in the local checkout.  It learns nonnegative
soft-vote weights from verifier score columns with LogisticRegression and
reports held-out scores from a stratified cross-validation loop.

Spec: REQ-TIER0-012, SCENARIO-TIER0-012, Exp 2398.
"""

from __future__ import annotations

import importlib
import json
import math
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MANIFEST_PATH = Path("results/live_sota_balanced_telemetry_manifest_1480.jsonl")
DEFAULT_OUTPUT_PATH = Path("results/experiment_2398_hive_ensemble.json")
DEFAULT_RANDOM_SEED = 42
SEMANTIC_ENERGY_BASELINE_AUROC = 0.685
HALLSCAN_REFERENCE_AUROC = 0.88

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class VerifierSpec:
    """Import and score metadata for one optional Tier 0 verifier."""

    name: str
    tier: str
    module_name: str


VERIFIER_SPECS: tuple[VerifierSpec, ...] = (
    VerifierSpec("freq_aware_attention", "0f", "carnot.verify.freq_aware_attention"),
    VerifierSpec("semantic_energy", "0g", "carnot.verify.semantic_energy"),
    VerifierSpec("laab_verifier", "0h", "carnot.verify.laab_verifier"),
    VerifierSpec("halt_probe", "0j", "carnot.verify.halt_probe"),
)


def _read_jsonl(path: Path, limit: int | None = None) -> list[JsonDict]:
    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def label_from_entry(entry: JsonDict) -> int:
    """Return 1 for incorrect/hallucination rows and 0 for correct rows."""

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


def _binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    label_array = np.asarray(labels)
    score_array = np.asarray(scores, dtype=np.float64)
    if label_array.shape[0] != score_array.shape[0]:
        raise ValueError("labels and scores must have the same length")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores must be finite")

    positive_scores = score_array[label_array == 1]
    negative_scores = score_array[label_array == 0]
    if positive_scores.size == 0 or negative_scores.size == 0:
        raise ValueError("labels must include at least one positive and one negative example")

    wins = 0.0
    for positive_score in positive_scores:
        wins += float(np.sum(positive_score > negative_scores))
        wins += 0.5 * float(np.sum(positive_score == negative_scores))
    return float(wins / (positive_scores.size * negative_scores.size))


def _finite_float(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _safe_sigmoid(value: float) -> float:
    bounded = max(min(float(value), 60.0), -60.0)
    return float(1.0 / (1.0 + math.exp(-bounded)))


def _soft_vote_weights_from_coefficients(coefficients: np.ndarray) -> np.ndarray:
    """Map LogisticRegression coefficients to normalized positive vote weights.

    LogisticRegression learns signed log-odds coefficients.  The HIVE soft vote
    needs nonnegative weights for `sum(weight_i * score_i) / sum(weights)`, so
    this uses a sigmoid activation and normalizes onto the probability simplex.

    Spec: REQ-TIER0-012-3
    """

    weights = np.asarray([_safe_sigmoid(float(value)) for value in coefficients], dtype=np.float64)
    if weights.size == 0 or not np.all(np.isfinite(weights)) or float(np.sum(weights)) <= 0.0:
        weights = np.ones_like(np.asarray(coefficients, dtype=np.float64))
    return weights / float(np.sum(weights))


def _normalize_by_train(
    train_scores: np.ndarray, test_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    minimum = np.min(train_scores, axis=0)
    maximum = np.max(train_scores, axis=0)
    span = maximum - minimum
    constant = np.isclose(span, 0.0)
    safe_span = np.where(constant, 1.0, span)
    train_normalized = (train_scores - minimum) / safe_span
    test_normalized = (test_scores - minimum) / safe_span
    train_normalized[:, constant] = 0.0
    test_normalized[:, constant] = 0.0
    return (
        np.clip(train_normalized, 0.0, 1.0),
        np.clip(test_normalized, 0.0, 1.0),
        minimum.astype(np.float64),
        maximum.astype(np.float64),
    )


def _stratified_fold_count(labels: Sequence[int], requested: int) -> int:
    label_counts = Counter(int(label) for label in labels)
    if len(label_counts) < 2:
        raise ValueError("labels must include both classes")
    fold_count = min(int(requested), min(label_counts.values()))
    if fold_count < 2:
        raise ValueError("at least two examples per class are required")
    return fold_count


class HiveEnsembleDetector:
    """HIVE-style learned soft-vote ensemble for cached Tier 0 verifier scores."""

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        random_seed: int = DEFAULT_RANDOM_SEED,
        n_splits: int = 5,
    ) -> None:
        self.threshold = float(threshold)
        self.random_seed = int(random_seed)
        self.n_splits = int(n_splits)
        self.available_verifiers: list[str] = []
        self.skipped_verifiers: dict[str, str] = {}
        self.verifier_weights_: dict[str, float] = {}
        self.score_min_: dict[str, float] = {}
        self.score_max_: dict[str, float] = {}
        self._halt_model: Any | None = None

    def discover_verifiers(self) -> list[str]:
        """Return importable verifier names and remember skipped modules."""

        available: list[str] = []
        skipped: dict[str, str] = {}
        for spec in VERIFIER_SPECS:
            try:
                importlib.import_module(spec.module_name)
            except Exception as exc:  # noqa: BLE001 - optional verifier discovery
                skipped[spec.name] = f"{exc.__class__.__name__}: {exc}"
            else:
                available.append(spec.name)
        self.available_verifiers = available
        self.skipped_verifiers = skipped
        return list(available)

    def collect_verifier_scores(
        self, entries: Sequence[JsonDict], labels: Sequence[int] | None = None
    ) -> dict[str, list[float]]:
        """Collect one score column per available verifier for diagnostics.

        HALT has a fitted proxy C path, so when labels are supplied this returns
        out-of-fold HALT risk scores.  Other available verifiers are
        training-free cached-telemetry scorers.

        Spec: REQ-TIER0-012-1
        """

        if not self.available_verifiers:
            self.discover_verifiers()

        scores: dict[str, list[float]] = {}
        for name in self.available_verifiers:
            if name == "halt_probe":
                if labels is None:
                    scores[name] = self._score_halt_training_free(entries)
                else:
                    scores[name] = self._score_halt_oof(entries, labels)
            else:
                scores[name] = self._score_static_verifier(name, entries)
            self._validate_score_column(name, scores[name], len(entries))
        return scores

    def fit(self, entries: Sequence[JsonDict], labels: Sequence[int]) -> HiveEnsembleDetector:
        """Fit final full-data weights for later single-entry `verify()` calls."""

        if not self.available_verifiers:
            self.discover_verifiers()
        if len(self.available_verifiers) < 2:
            raise ValueError("at least two verifiers are required for HIVE ensemble")

        raw_scores = self.collect_verifier_scores(entries, labels)
        names = list(raw_scores)
        matrix = np.asarray([raw_scores[name] for name in names], dtype=np.float64).T
        normalized, _unused, minimum, maximum = _normalize_by_train(matrix, matrix)
        weights, _model = self._fit_weight_model(normalized, labels)
        self.verifier_weights_ = {
            name: float(weight) for name, weight in zip(names, weights, strict=True)
        }
        self.score_min_ = {name: float(value) for name, value in zip(names, minimum, strict=True)}
        self.score_max_ = {name: float(value) for name, value in zip(names, maximum, strict=True)}

        if "halt_probe" in names:
            module = importlib.import_module("carnot.verify.halt_probe")
            self._halt_model = module.HaltProbeDetector(random_seed=self.random_seed).fit(
                list(entries), list(labels)
            )
        return self

    def verify(self, entry: JsonDict) -> JsonDict:
        """Return a HIVE ensemble score for one entry after `fit()`."""

        if not self.verifier_weights_:
            raise ValueError("HiveEnsembleDetector must be fitted before verify()")

        raw_scores = {
            name: self._score_one_verifier(name, entry)
            for name in self.verifier_weights_
        }
        weighted_sum = 0.0
        weight_total = 0.0
        for name, raw_score in raw_scores.items():
            minimum = self.score_min_[name]
            maximum = self.score_max_[name]
            if math.isclose(maximum, minimum):
                normalized = 0.0
            else:
                normalized = (raw_score - minimum) / (maximum - minimum)
            weight = self.verifier_weights_[name]
            weighted_sum += weight * min(max(float(normalized), 0.0), 1.0)
            weight_total += weight

        score = float(weighted_sum / weight_total) if weight_total > 0.0 else 0.0
        return {
            "hive_ensemble_score": score,
            "is_high_risk": bool(score >= self.threshold),
            "verifier_scores": raw_scores,
            "verifier_weights": dict(self.verifier_weights_),
            "n_verifiers_fused": len(self.verifier_weights_),
        }

    def evaluate(self, entries: Sequence[JsonDict], labels: Sequence[int]) -> JsonDict:
        """Evaluate held-out HIVE soft-vote AUROC on labeled entries."""

        if not self.available_verifiers:
            self.discover_verifiers()
        if len(self.available_verifiers) < 2:
            raise ValueError("at least two verifiers are required for HIVE ensemble")

        labels_array = np.asarray(labels, dtype=np.int64)
        static_scores = {
            name: self._score_static_verifier(name, entries)
            for name in self.available_verifiers
            if name != "halt_probe"
        }
        for name, scores in static_scores.items():
            self._validate_score_column(name, scores, len(entries))

        from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415

        fold_count = _stratified_fold_count(labels, self.n_splits)
        splitter = StratifiedKFold(
            n_splits=fold_count, shuffle=True, random_state=self.random_seed
        )
        indices = np.arange(len(entries))
        heldout_scores = np.zeros(len(entries), dtype=np.float64)
        fold_weights: list[dict[str, float]] = []
        fold_details: list[JsonDict] = []

        for fold_index, (train_idx, test_idx) in enumerate(
            splitter.split(indices, labels_array), start=1
        ):
            names: list[str] = []
            train_columns: list[list[float]] = []
            test_columns: list[list[float]] = []

            for name in self.available_verifiers:
                if name == "halt_probe":
                    train_scores, test_scores = self._score_halt_fold(
                        entries, labels, train_idx, test_idx
                    )
                else:
                    all_scores = static_scores[name]
                    train_scores = [all_scores[int(idx)] for idx in train_idx]
                    test_scores = [all_scores[int(idx)] for idx in test_idx]
                names.append(name)
                train_columns.append(train_scores)
                test_columns.append(test_scores)

            train_matrix = np.asarray(train_columns, dtype=np.float64).T
            test_matrix = np.asarray(test_columns, dtype=np.float64).T
            train_normalized, test_normalized, _minimum, _maximum = _normalize_by_train(
                train_matrix, test_matrix
            )
            weights, _model = self._fit_weight_model(
                train_normalized, labels_array[train_idx].tolist()
            )
            heldout_scores[test_idx] = test_normalized @ weights
            weight_map = {
                name: float(weight) for name, weight in zip(names, weights, strict=True)
            }
            fold_weights.append(weight_map)
            fold_details.append(
                {
                    "fold": fold_index,
                    "train_size": int(len(train_idx)),
                    "test_size": int(len(test_idx)),
                    "verifier_weights": weight_map,
                }
            )

        averaged_weights = self._average_fold_weights(fold_weights)
        raw_scores = self.collect_verifier_scores(entries, labels)
        base_aurocs = {
            name: float(_binary_auroc(labels, scores)) for name, scores in raw_scores.items()
        }
        return {
            "hive_ensemble_auroc": float(_binary_auroc(labels, heldout_scores.tolist())),
            "heldout_scores": [float(score) for score in heldout_scores],
            "verifier_weights": averaged_weights,
            "n_verifiers_fused": len(averaged_weights),
            "base_verifier_aurocs": base_aurocs,
            "fold_details": fold_details,
        }

    def _fit_weight_model(
        self, matrix: np.ndarray, labels: Sequence[int]
    ) -> tuple[np.ndarray, Any]:
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=self.random_seed,
            solver="liblinear",
        )
        model.fit(matrix, np.asarray(labels, dtype=np.int64))
        weights = _soft_vote_weights_from_coefficients(np.ravel(model.coef_[0]))
        return weights, model

    def _score_static_verifier(
        self, name: str, entries: Sequence[JsonDict]
    ) -> list[float]:
        if name == "freq_aware_attention":
            module = importlib.import_module("carnot.verify.freq_aware_attention")
            detector = module.FreqAwareAttentionDetector()
            return [float(detector.compute_freq_attn_score(entry)) for entry in entries]

        if name == "semantic_energy":
            module = importlib.import_module("carnot.verify.semantic_energy")
            detector = module.SemanticEnergyDetector()
            return [
                float(abs(detector.compute_energy(
                    module.top_logprobs_to_logit_vector(entry.get("top_logprobs") or [])
                )))
                for entry in entries
            ]

        if name == "laab_verifier":
            module = importlib.import_module("carnot.verify.laab_verifier")
            return [self._score_laab_entry(module, entry) for entry in entries]

        raise ValueError(f"unsupported static verifier: {name}")

    def _score_one_verifier(self, name: str, entry: JsonDict) -> float:
        if name == "halt_probe":
            if self._halt_model is not None:
                return float(self._halt_model.compute_halt_score(entry))
            return self._score_halt_training_free([entry])[0]
        return self._score_static_verifier(name, [entry])[0]

    def _score_halt_training_free(self, entries: Sequence[JsonDict]) -> list[float]:
        module = importlib.import_module("carnot.verify.halt_probe")
        detector = module.HaltProbeDetector(random_seed=self.random_seed)
        return [float(detector.compute_halt_score(entry)) for entry in entries]

    def _score_halt_oof(
        self, entries: Sequence[JsonDict], labels: Sequence[int]
    ) -> list[float]:
        module = importlib.import_module("carnot.verify.halt_probe")
        fold_count = min(6, _stratified_fold_count(labels, 6))
        return [
            float(score)
            for score in module.oof_halt_risk_scores(
                list(entries), list(labels), random_seed=self.random_seed, n_splits=fold_count
            )
        ]

    def _score_halt_fold(
        self,
        entries: Sequence[JsonDict],
        labels: Sequence[int],
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> tuple[list[float], list[float]]:
        module = importlib.import_module("carnot.verify.halt_probe")
        train_entries = [entries[int(idx)] for idx in train_idx]
        train_labels = [int(labels[int(idx)]) for idx in train_idx]
        train_fold_count = _stratified_fold_count(train_labels, self.n_splits)
        train_scores = [
            float(score)
            for score in module.oof_halt_risk_scores(
                train_entries,
                train_labels,
                random_seed=self.random_seed,
                n_splits=train_fold_count,
            )
        ]
        detector = module.HaltProbeDetector(random_seed=self.random_seed).fit(
            train_entries, train_labels
        )
        test_scores = [
            float(detector.compute_halt_score(entries[int(idx)])) for idx in test_idx
        ]
        return train_scores, test_scores

    def _score_laab_entry(self, module: Any, entry: JsonDict) -> float:
        for function_name in (
            "compute_laab_score",
            "laab_score_from_entry",
            "compute_logical_consistency_risk",
        ):
            function = getattr(module, function_name, None)
            if callable(function):
                result = function(entry)
                if isinstance(result, tuple):
                    result = result[0]
                return _finite_float(result, function_name)

        verifier_cls = getattr(module, "LaaBVerifier", None)
        if verifier_cls is not None:
            verifier = verifier_cls()
            for method_name in ("compute_laab_score", "compute_risk_score", "verify"):
                method = getattr(verifier, method_name, None)
                if callable(method):
                    result = method(entry)
                    return self._score_from_verifier_result(result)

        verify_function = getattr(module, "verify", None)
        if callable(verify_function):
            return self._score_from_verifier_result(verify_function(entry))

        raise ValueError("laab_verifier does not expose a supported score API")

    @staticmethod
    def _score_from_verifier_result(result: Any) -> float:
        if isinstance(result, int | float):
            return _finite_float(result, "laab_result")
        if not isinstance(result, dict):
            raise ValueError("verifier result must be numeric or dict-like")
        for field_name in (
            "laab_score",
            "laab_risk_score",
            "logical_consistency_risk",
            "risk_score",
            "score",
        ):
            if field_name in result:
                return _finite_float(result[field_name], field_name)
        raise ValueError("verifier result does not contain a supported score field")

    @staticmethod
    def _validate_score_column(name: str, scores: Sequence[float], expected: int) -> None:
        if len(scores) != expected:
            raise ValueError(f"{name} produced {len(scores)} scores for {expected} entries")
        if not np.all(np.isfinite(np.asarray(scores, dtype=np.float64))):
            raise ValueError(f"{name} produced non-finite scores")

    @staticmethod
    def _average_fold_weights(fold_weights: Sequence[dict[str, float]]) -> dict[str, float]:
        totals: Counter[str] = Counter()
        for weight_map in fold_weights:
            for name, value in weight_map.items():
                totals[name] += float(value)
        averaged = {name: value / len(fold_weights) for name, value in totals.items()}
        total = sum(averaged.values())
        if total <= 0.0:
            uniform = 1.0 / len(averaged)
            return {name: uniform for name in averaged}
        return {name: float(value / total) for name, value in averaged.items()}


def _sklearn_precondition() -> JsonDict:
    try:
        import sklearn  # noqa: PLC0415
    except ModuleNotFoundError:
        return {"sklearn_importable": False, "sklearn_version": None}
    return {"sklearn_importable": True, "sklearn_version": sklearn.__version__}


def _preconditions(manifest_path: Path) -> JsonDict:
    checked = _sklearn_precondition()
    checked["telemetry_manifest_present"] = manifest_path.is_file()
    checked["telemetry_manifest_path"] = str(manifest_path)
    detector = HiveEnsembleDetector()
    available = detector.discover_verifiers()
    checked["verifier_imports"] = {
        spec.name: spec.name in available for spec in VERIFIER_SPECS
    }
    checked["available_verifier_count"] = len(available)
    checked["available_verifiers"] = available
    checked["missing_verifiers"] = [
        spec.name for spec in VERIFIER_SPECS if spec.name not in available
    ]
    checked["skipped_verifier_errors"] = detector.skipped_verifiers
    checked["telemetry_fields"] = []
    if manifest_path.is_file():
        rows = _read_jsonl(manifest_path, limit=1)
        checked["telemetry_fields"] = list(rows[0].keys()) if rows else []
    return checked


def _blocked_artifact(
    *, honest_verdict: str, checked: JsonDict, start: float, random_seed: int
) -> JsonDict:
    return {
        "status": "blocked",
        "experiment": 2398,
        "honest_verdict": honest_verdict,
        "hive_ensemble_auroc": None,
        "hive_gap_closed_vs_hallscan": None,
        "ensemble_auroc_improved": False,
        "n_verifiers_fused": int(checked.get("available_verifier_count", 0)),
        "verifier_weights": {},
        "n_eval_examples": 0,
        "random_seed": int(random_seed),
        "duration_s": round(time.perf_counter() - start, 6),
        "preconditions_checked": checked,
        "acceptance_gates": {
            "ensemble_auroc_improved": False,
            "n_verifiers_fused_gte_2": int(checked.get("available_verifier_count", 0)) >= 2,
        },
    }


def build_experiment_artifact(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    n_eval_examples: int = 36,
    random_seed: int = DEFAULT_RANDOM_SEED,
    semantic_energy_baseline: float = SEMANTIC_ENERGY_BASELINE_AUROC,
) -> JsonDict:
    """Evaluate HIVE soft-vote fusion on the cached telemetry split.

    Spec: REQ-TIER0-012-4, REQ-TIER0-012-5
    """

    start = time.perf_counter()
    manifest = Path(manifest_path)
    checked = _preconditions(manifest)
    if checked["available_verifier_count"] < 2:
        return _blocked_artifact(
            honest_verdict="blocked_insufficient_verifiers",
            checked=checked,
            start=start,
            random_seed=random_seed,
        )
    if not checked["telemetry_manifest_present"]:
        return _blocked_artifact(
            honest_verdict="blocked_telemetry_manifest_missing",
            checked=checked,
            start=start,
            random_seed=random_seed,
        )
    if not checked["sklearn_importable"]:
        return _blocked_artifact(
            honest_verdict="blocked_sklearn_missing",
            checked=checked,
            start=start,
            random_seed=random_seed,
        )

    entries = _read_jsonl(manifest, limit=n_eval_examples)
    labels = [label_from_entry(entry) for entry in entries]
    detector = HiveEnsembleDetector(random_seed=random_seed, n_splits=5)
    detector.discover_verifiers()
    evaluation = detector.evaluate(entries, labels)
    detector.fit(entries, labels)

    auroc = float(evaluation["hive_ensemble_auroc"])
    duration_s = round(time.perf_counter() - start, 6)
    ensemble_improved = bool(auroc > float(semantic_energy_baseline))
    n_verifiers_fused = int(evaluation["n_verifiers_fused"])
    validated = bool(
        len(entries) == int(n_eval_examples)
        and n_verifiers_fused >= 2
        and math.isfinite(auroc)
    )

    return {
        "status": "complete",
        "experiment": 2398,
        "title": "HIVE-style learned soft-vote Tier 0 verifier ensemble",
        "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "module_path": "python/carnot/verify/hive_ensemble.py",
        "spec_refs": ["REQ-TIER0-012", "SCENARIO-TIER0-012"],
        "field_principles": {
            "honest_verdict": "Terminal-prefix required. complete: with AUROC.",
            "hive_ensemble_auroc": (
                "Primary metric. Honest result - may be below 0.88, that is a finding."
            ),
            "hive_gap_closed_vs_hallscan": (
                "0.88 - hive_ensemble_auroc. Negative = exceeded baseline. "
                "Key paper-v6 metric."
            ),
            "ensemble_auroc_improved": (
                "True if hive_ensemble_auroc > 0.685 (beats single-verifier baseline)."
            ),
            "n_verifiers_fused": "How many Tier 0 verifiers contributed to ensemble.",
            "verifier_weights": "Soft-vote weights per verifier - reproducibility.",
            "n_eval_examples": "Must be 36.",
            "random_seed": "Must be 42.",
            "duration_s": "Guards against fabrication.",
            "preconditions_checked": "Records verifier count + telemetry check.",
        },
        "honest_verdict": (
            "complete: HiveEnsembleDetector fused "
            f"{n_verifiers_fused} Tier 0 verifier scores on {len(entries)} cached "
            f"telemetry entries; AUROC={auroc:.6f}."
        ),
        "hive_ensemble_validated": validated,
        "hive_ensemble_auroc": auroc,
        "hive_gap_closed_vs_hallscan": float(HALLSCAN_REFERENCE_AUROC - auroc),
        "ensemble_auroc_improved": ensemble_improved,
        "semantic_energy_baseline_auroc": float(semantic_energy_baseline),
        "hallscan_reference_auroc": HALLSCAN_REFERENCE_AUROC,
        "hive_external_reference_auroc": 0.9236,
        "n_verifiers_fused": n_verifiers_fused,
        "available_verifiers": list(evaluation["verifier_weights"].keys()),
        "skipped_verifiers": checked["missing_verifiers"],
        "verifier_weights": evaluation["verifier_weights"],
        "base_verifier_aurocs": evaluation["base_verifier_aurocs"],
        "n_eval_examples": len(entries),
        "n_factual_examples": int(labels.count(0)),
        "n_hallucination_examples": int(labels.count(1)),
        "random_seed": int(random_seed),
        "duration_s": duration_s,
        "preconditions_checked": checked,
        "score_direction": "higher_score_means_more_hallucination_like",
        "score_field": "hive_ensemble_score",
        "weight_learning": (
            "stratified_5_fold_logistic_regression_with_sigmoid_coefficient_weights"
        ),
        "evaluation_design": (
            "Load the first 36 live SOTA balanced telemetry rows, collect available "
            "Tier 0 verifier scores, learn soft-vote weights inside each held-out "
            "fold, and compute weighted_score=sum(weight_i*score_i)/sum(weights)."
        ),
        "source_artifact": str(manifest),
        "heldout_score_summary": {
            "min": float(np.min(np.asarray(evaluation["heldout_scores"], dtype=np.float64))),
            "max": float(np.max(np.asarray(evaluation["heldout_scores"], dtype=np.float64))),
            "mean": float(np.mean(np.asarray(evaluation["heldout_scores"], dtype=np.float64))),
            "std": float(np.std(np.asarray(evaluation["heldout_scores"], dtype=np.float64))),
        },
        "fold_details": evaluation["fold_details"],
        "acceptance_gates": {
            "ensemble_auroc_improved": ensemble_improved,
            "n_verifiers_fused_gte_2": n_verifiers_fused >= 2,
        },
    }


def write_experiment_artifact(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> JsonDict:
    """Write the Exp 2398 HIVE ensemble deliverable JSON."""

    artifact = build_experiment_artifact(manifest_path=manifest_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


if __name__ == "__main__":
    print(json.dumps(write_experiment_artifact(), indent=2, sort_keys=True))
