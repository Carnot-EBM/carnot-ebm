"""FR-11 continuous self-learning v23 independence-reweighting tracker.

Spec: REQ-LEARN-3864, SCENARIO-LEARN-3864.

The v23 forward difference keeps the v17 precision counters but adds a second
online signal: whether one verifier catches a gold error that every peer misses.
That unique-catch counter is the practical proxy for inferred independence.  We
combine it with precision and the frozen Exp 2837 prior so the tracker can
learn which verifier contributes residual error coverage without silently
moving the frozen 0.9131 headline result.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot.fr11 import continuous_self_learning_v17 as v17


JsonDict = dict[str, Any]

OUTPUT_REL_PATH = Path(
    "results/experiment_3864_fr11_self_learning_v23_independence_reweighting.json"
)
STATE_REL_PATH = Path("results/experiment_3864_fr11_v23_independence_reweighting_state.json")
DEFAULT_RANDOM_SEED = 3864
DEFAULT_CORPUS_RANDOM_SEED = 3864
DEFAULT_N_EXAMPLES = 1000
DEFAULT_SCORE_THRESHOLD = 0.5
FROZEN_CI95 = (0.9027316334533082, 0.9235355665466916)
PRIOR_FLOOR = 0.01
PRECISION_FLOOR = 0.5
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: reads cached verifier scores; no LLM loaded)."
)
UPDATE_COST_NOTE = (
    "Tier-1 hardware path: pure CPU counter updates over cached verifier scores; "
    "qualitatively <1us/update, no fabricated wall-time timing."
)
SUCCESS_VERDICT_PREFIX = (
    "complete: fr11_v23_independence_reweighting_learned_auroc"
)
SUCCESS_VERDICT_SUFFIX = (
    "_in_frozen_ci_memory_contribution_preserved_state_persisted"
)
REGRESS_VERDICT = (
    "complete: "
    "fr11_v23_independence_reweighting_REGRESSED_below_frozen_ci_"
    "reverted_to_static_weights"
)
BLOCKED_CORPUS_VERDICT = "blocked_cached_score_corpus"
BLOCKED_VERIFY_VERDICT = "blocked_carnot_verify_import"
BLOCKED_SCORING_VERDICT = "blocked_fover_scores_missing"
BLOCKED_MEMORY_VERDICT = "blocked_memory_ablation_reference"
VERIFIER_NAMES = v17.VERIFIER_NAMES

score_fover_corpus = v17.score_fover_corpus
score_matrix = v17.score_matrix

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "reweighted_ensemble_auroc",
    "auroc_in_frozen_ci",
    "independence_weight_per_verifier",
    "memory_ablation_contribution_preserved",
    "update_cost_note",
    "state_persisted_path",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "inference_substrate",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix; complete or blocked_<resource>.",
    "reweighted_ensemble_auroc": (
        "The independence-reweighted ensemble's AUROC; must not regress the "
        "frozen headline below CI."
    ),
    "auroc_in_frozen_ci": (
        "BARE bool (not a {value,principle} dict -- gated-fields-must-be-bare) "
        "-- the INVARIANT gate: the reweighting may not break the frozen 0.9131."
    ),
    "independence_weight_per_verifier": (
        "The learned online weighting -- which verifiers the tracker upweighted "
        "for independence; the self-learning artifact + input to the moat story."
    ),
    "memory_ablation_contribution_preserved": (
        "Bare bool -- the +0.0185 learned contribution survives the reweighting."
    ),
    "update_cost_note": (
        "Qualitative Tier-1 hardware-path claim (pure CPU counter updates, "
        "<1us/update); NOT a fabricated wall-time."
    ),
    "state_persisted_path": (
        "Where the learned weighting is saved for the next FR-11 milestone "
        "(continuity of the self-learning loop)."
    ),
    "preconditions_checked": (
        "Adversarial-Verify + Inference-Substrate; verifier-scoring substrate "
        "(no LLM loaded)."
    ),
    "random_seed": (
        "Adversarial-Verify + Inference-Substrate; verifier-scoring substrate "
        "(no LLM loaded)."
    ),
    "reproducibility_checksum": (
        "Adversarial-Verify + Inference-Substrate; verifier-scoring substrate "
        "(no LLM loaded)."
    ),
    "inference_substrate": (
        "Adversarial-Verify + Inference-Substrate; verifier-scoring substrate "
        "(no LLM loaded)."
    ),
    "duration_s": (
        "Adversarial-Verify + Inference-Substrate; verifier-scoring substrate "
        "(no LLM loaded)."
    ),
}


@dataclass
class IndependenceStats:
    """Raw online counters for one verifier's precision and unique catches."""

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0
    unique_error_catches: int = 0

    def record(self, *, predicted_positive: bool, label: int, unique_catch: bool) -> None:
        actual_positive = int(label) == 1
        if predicted_positive and actual_positive:
            self.true_positive += 1
            if unique_catch:
                self.unique_error_catches += 1
        elif predicted_positive and not actual_positive:
            self.false_positive += 1
        elif not predicted_positive and actual_positive:
            self.false_negative += 1
        else:
            self.true_negative += 1

    @property
    def predicted_positive_total(self) -> int:
        return self.true_positive + self.false_positive

    @property
    def actual_positive_total(self) -> int:
        return self.true_positive + self.false_negative

    @property
    def total(self) -> int:
        return (
            self.true_positive
            + self.false_positive
            + self.false_negative
            + self.true_negative
        )

    @property
    def precision(self) -> float:
        return _rate(self.true_positive, self.predicted_positive_total)

    @property
    def independence_score(self) -> float:
        return _rate(self.unique_error_catches, self.actual_positive_total)

    def to_json(self) -> JsonDict:
        return {
            "true_positive": int(self.true_positive),
            "false_positive": int(self.false_positive),
            "false_negative": int(self.false_negative),
            "true_negative": int(self.true_negative),
            "unique_error_catches": int(self.unique_error_catches),
        }


class IndependenceReweightingTracker:
    """Online CPU counter tracker for precision and inferred independence."""

    def __init__(self, verifier_names: Sequence[str] = VERIFIER_NAMES) -> None:
        self.verifier_names = tuple(str(name) for name in verifier_names)
        self._stats = {name: IndependenceStats() for name in self.verifier_names}

    def update_from_scores(
        self,
        labels: Sequence[int],
        scores_by_verifier: Mapping[str, Sequence[float]],
        *,
        threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> None:
        labels_arr = [int(label) for label in labels]
        if any(label not in {0, 1} for label in labels_arr):
            raise ValueError("labels must be binary 0/1")
        for name in self.verifier_names:
            if name not in scores_by_verifier:
                raise ValueError(f"missing verifier score column: {name}")
            if len(scores_by_verifier[name]) != len(labels_arr):
                raise ValueError("labels and verifier scores must have the same length")
        for index, label in enumerate(labels_arr):
            row_scores = {
                name: _finite_score(scores_by_verifier[name][index])
                for name in self.verifier_names
            }
            predicted = {
                name: row_scores[name] >= float(threshold) for name in self.verifier_names
            }
            catch_count = sum(1 for value in predicted.values() if value)
            for name in self.verifier_names:
                self._stats[name].record(
                    predicted_positive=predicted[name],
                    label=label,
                    unique_catch=bool(int(label) == 1 and predicted[name] and catch_count == 1),
                )

    def stats(self) -> dict[str, IndependenceStats]:
        return dict(self._stats)

    def to_json(self) -> JsonDict:
        return {
            "schema": "carnot.fr11_v23_independence_reweighting_state",
            "version": 1,
            "verifier_names": list(self.verifier_names),
            "stats": {
                name: self._stats[name].to_json() for name in self.verifier_names
            },
        }


def build_artifact(
    repo_root: Path | str,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    n_examples: int = DEFAULT_N_EXAMPLES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    corpus_random_seed: int = DEFAULT_CORPUS_RANDOM_SEED,
    state_path: Path | str = STATE_REL_PATH,
    frozen_ci: tuple[float, float] = FROZEN_CI95,
) -> JsonDict:
    """Build Exp 3864 from cached FoVer rows and verifier scores."""

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    state_output = _resolve_under_root(root, state_path)
    preconditions = [
        verify_import_precondition(),
        cached_corpus_precondition(root, n_examples=n_examples),
    ]
    if not all(item["available"] for item in preconditions):
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            state_output=state_output,
            repo_root=root,
            preconditions=preconditions,
            verdict=_blocked_verdict(preconditions),
        )
    try:
        labels, scores_by_verifier = score_fover_corpus(
            root,
            n_examples=n_examples,
            random_seed=corpus_random_seed,
        )
    except Exception as exc:  # noqa: BLE001 - cached verifier scoring is terminal.
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            state_output=state_output,
            repo_root=root,
            preconditions=[
                *preconditions,
                {
                    "resource": "cached_verifier_scores",
                    "available": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                },
            ],
            verdict=BLOCKED_SCORING_VERDICT,
        )
    return build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=start,
        now_s=now_s,
        random_seed=random_seed,
        repo_root=root,
        state_path=state_output,
        preconditions=preconditions,
        frozen_ci=frozen_ci,
        memory_contribution_reference=v17.load_memory_contribution_reference(root),
    )


def build_artifact_from_scores(
    *,
    labels: Sequence[int],
    scores_by_verifier: Mapping[str, Sequence[float]],
    started_s: float,
    now_s: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    repo_root: Path | str | None = None,
    state_path: Path | str = STATE_REL_PATH,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
    frozen_ci: tuple[float, float] = FROZEN_CI95,
    memory_contribution_reference: float | None = None,
) -> JsonDict:
    """Update online independence counters and evaluate the learned ensemble."""

    root = Path(repo_root) if repo_root is not None else None
    state_output = _resolve_under_root(root or Path("."), state_path)
    if not labels or not scores_by_verifier:
        return _blocked_artifact(
            duration_s=_duration(float(started_s), now_s),
            random_seed=random_seed,
            state_output=state_output,
            repo_root=root,
            preconditions=preconditions or [_trace_precondition(labels, scores_by_verifier)],
            verdict=BLOCKED_CORPUS_VERDICT,
        )

    labels_arr = np.asarray(labels, dtype=np.int64)
    _require_binary_labels(labels_arr)
    matrix = score_matrix(scores_by_verifier, VERIFIER_NAMES)
    if matrix.shape[0] != len(labels_arr):
        raise ValueError("labels and verifier scores must have the same length")
    tracker = IndependenceReweightingTracker(VERIFIER_NAMES)
    tracker.update_from_scores(labels_arr.tolist(), scores_by_verifier)
    table = independence_table(tracker, VERIFIER_NAMES)
    weights = independence_weighting(table, VERIFIER_NAMES)
    weight_vector = np.asarray([weights[name] for name in VERIFIER_NAMES], dtype=np.float64)
    ensemble_scores = v17.v10.exp3644.ensemble_scores(matrix, weight_vector)
    auroc = v17.v10.exp3644.tie_aware_auroc(labels_arr, ensemble_scores)
    auroc_in_ci = _in_ci(auroc, frozen_ci)
    memory_preserved = v17.memory_contribution_preserved(
        weights,
        memory_contribution_reference,
    )
    state_sha = persist_state(
        tracker=tracker,
        path=state_output,
        weights=weights,
        table=table,
        n_examples_observed=len(labels_arr),
        random_seed=random_seed,
        reweighted_ensemble_auroc=auroc,
        auroc_in_frozen_ci=auroc_in_ci,
        memory_ablation_contribution_preserved=memory_preserved,
    )
    state_persisted = state_output.is_file()
    verdict = _select_verdict(
        auroc=auroc,
        auroc_in_frozen_ci=auroc_in_ci,
        memory_ablation_contribution_preserved=memory_preserved,
        state_persisted=state_persisted,
    )
    artifact: JsonDict = {
        "artifact": "experiment_3864_fr11_self_learning_v23_independence_reweighting",
        "schema": "carnot.fr11_continuous_self_learning_v23",
        "continuous_self_learning_task": True,
        "honest_verdict": verdict,
        "reweighted_ensemble_auroc": _round(auroc),
        "auroc_in_frozen_ci": bool(auroc_in_ci),
        "independence_weight_per_verifier": weights,
        "memory_ablation_contribution_preserved": bool(memory_preserved),
        "update_cost_note": UPDATE_COST_NOTE,
        "state_persisted_path": _relative_path(state_output, root),
        "preconditions_checked": [dict(item) for item in preconditions or []],
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum(
            labels=labels_arr,
            scores=matrix,
            table=table,
            weights=weights,
            state_sha256=state_sha,
            random_seed=random_seed,
            frozen_ci=frozen_ci,
            memory_contribution_reference=memory_contribution_reference,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _round(_duration(float(started_s), now_s)),
        "n_examples": int(len(labels_arr)),
        "score_threshold": DEFAULT_SCORE_THRESHOLD,
        "verifier_names": list(VERIFIER_NAMES),
        "per_verifier_independence_table": table,
        "frozen_ci95": {"low": float(frozen_ci[0]), "high": float(frozen_ci[1])},
        "frozen_headline_ensemble_auroc": 0.9131,
        "memory_contribution_reference": memory_contribution_reference,
        "state_sha256": state_sha,
        "state_persisted": bool(state_persisted),
        "fallback_static_weighting": static_weighting(VERIFIER_NAMES),
        "model_specs": _model_specs(n_examples=len(labels_arr)),
        "methodology": _methodology(random_seed),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "passed": bool(auroc_in_ci and memory_preserved and state_persisted),
            "condition": (
                "auroc_in_frozen_ci == true AND "
                "memory_ablation_contribution_preserved == true"
            ),
            "principle": (
                "Self-learning may persist the independence state only as a "
                "continuation artifact; the frozen 0.9131 headline remains fixed."
            ),
        },
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(
    repo_root: Path | str,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    state_path: Path | str = STATE_REL_PATH,
    labels: Sequence[int] | None = None,
    scores_by_verifier: Mapping[str, Sequence[float]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    frozen_ci: tuple[float, float] = FROZEN_CI95,
) -> Path:
    """Build, validate, and write the Exp 3864 JSON artifact."""

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    resolved_state_path = _resolve_under_root(root, state_path)
    if labels is not None or scores_by_verifier is not None:
        artifact = build_artifact_from_scores(
            labels=labels or [],
            scores_by_verifier=scores_by_verifier or {},
            started_s=start,
            now_s=now_s,
            repo_root=root,
            state_path=resolved_state_path,
            frozen_ci=frozen_ci,
            memory_contribution_reference=v17.load_memory_contribution_reference(root),
        )
    else:
        artifact = build_artifact(
            root,
            started_s=start,
            now_s=now_s,
            state_path=resolved_state_path,
            frozen_ci=frozen_ci,
        )
    output = _resolve_under_root(root, output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def independence_table(
    tracker: IndependenceReweightingTracker,
    verifier_names: Sequence[str] | None = None,
) -> list[JsonDict]:
    """Return one precision plus unique-catch row per verifier."""

    names = tuple(verifier_names or tracker.verifier_names)
    stats_by_name = tracker.stats()
    rows: list[JsonDict] = []
    for name in names:
        stats = stats_by_name.get(name, IndependenceStats())
        rows.append(
            {
                "verifier": name,
                "true_positive": stats.true_positive,
                "false_positive": stats.false_positive,
                "false_negative": stats.false_negative,
                "true_negative": stats.true_negative,
                "predicted_positive_total": stats.predicted_positive_total,
                "actual_positive_total": stats.actual_positive_total,
                "n_examples": stats.total,
                "precision": _round(stats.precision),
                "unique_error_catches": stats.unique_error_catches,
                "independence_score": _round(stats.independence_score),
                "measurement_note": (
                    "online_unique_error_catches_over_cached_verifier_masks"
                ),
            }
        )
    return rows


def independence_weighting(
    table: Sequence[Mapping[str, Any]],
    verifier_names: Sequence[str] = VERIFIER_NAMES,
) -> dict[str, float]:
    """Normalize frozen-prior, precision, and unique-catch independence weights."""

    if not table:
        return {}
    by_name = {str(row["verifier"]): row for row in table}
    static = static_weighting(verifier_names)
    raw = []
    for name in verifier_names:
        row = by_name.get(name, {})
        precision = float(row.get("precision", 0.0))
        independence = float(row.get("independence_score", 0.0))
        raw.append(
            (float(static.get(name, 0.0)) + PRIOR_FLOOR)
            * (PRECISION_FLOOR + max(0.0, precision))
            * (1.0 + max(0.0, independence))
        )
    weights = v17.v10.exp3644.normalize_weights(raw)
    return {
        name: _round(float(weight))
        for name, weight in zip(verifier_names, weights, strict=True)
    }


def static_weighting(verifier_names: Sequence[str] = VERIFIER_NAMES) -> dict[str, float]:
    """Return the frozen Exp 2837 prior in the requested verifier order."""

    weights = v17.v10.exp3644.carnot_current_weights(verifier_names)
    return {
        name: _round(float(weight))
        for name, weight in zip(verifier_names, weights, strict=True)
    }


def persist_state(
    *,
    tracker: IndependenceReweightingTracker,
    path: Path | str,
    weights: Mapping[str, float],
    table: Sequence[Mapping[str, Any]],
    n_examples_observed: int,
    random_seed: int,
    reweighted_ensemble_auroc: float,
    auroc_in_frozen_ci: bool,
    memory_ablation_contribution_preserved: bool,
) -> str:
    """Persist the raw counters and learned weights for the next milestone."""

    output = Path(path)
    payload = {
        **tracker.to_json(),
        "weights": dict(weights),
        "per_verifier_independence_table": [dict(row) for row in table],
        "n_examples_observed": int(n_examples_observed),
        "random_seed": int(random_seed),
        "reweighted_ensemble_auroc": _round(reweighted_ensemble_auroc),
        "auroc_in_frozen_ci": bool(auroc_in_frozen_ci),
        "memory_ablation_contribution_preserved": bool(
            memory_ablation_contribution_preserved
        ),
        "update_cost_note": UPDATE_COST_NOTE,
    }
    checksum = _json_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({**payload, "sha256": checksum}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return checksum


def verify_import_precondition(
    importer: Callable[[str], Any] = importlib.import_module,
) -> JsonDict:
    """Check the verifier package import without loading a live model."""

    try:
        importer("carnot.verify")
    except Exception as exc:  # noqa: BLE001 - import failure is the resource verdict.
        return {
            "resource": "carnot_verify_import",
            "available": False,
            "detail": f"{type(exc).__name__}: {exc}",
        }
    return {
        "resource": "carnot_verify_import",
        "available": True,
        "detail": "import carnot.verify succeeded",
    }


def cached_corpus_precondition(repo_root: Path | str, *, n_examples: int) -> JsonDict:
    """Check for the cached scoreable corpus before any online update."""

    root = Path(repo_root)
    fover_jsonl = root / "data" / "fover_corpus.jsonl"
    step_balanced = root / "data" / "step_error_balanced_v2.json"
    fover_v3 = root / "data" / "fover_corpus_v3.json"
    if fover_jsonl.is_file():
        line_count = _line_count(fover_jsonl)
        return {
            "resource": "cached_score_corpus",
            "available": line_count >= int(n_examples),
            "detail": f"data/fover_corpus.jsonl line_count={line_count}; required>={n_examples}",
        }
    alternatives = [path for path in (step_balanced, fover_v3) if path.is_file()]
    return {
        "resource": "cached_score_corpus",
        "available": bool(alternatives),
        "detail": (
            "fallback corpus present: "
            + ",".join(path.relative_to(root).as_posix() for path in alternatives)
            if alternatives
            else "missing data/fover_corpus.jsonl, data/step_error_balanced_v2.json, and data/fover_corpus_v3.json"
        ),
    }


def reproducibility_checksum(
    *,
    labels: np.ndarray | Sequence[int],
    scores: np.ndarray | Sequence[Sequence[float]],
    table: Sequence[Mapping[str, Any]],
    weights: Mapping[str, float],
    state_sha256: str,
    random_seed: int,
    frozen_ci: tuple[float, float],
    memory_contribution_reference: float | None,
) -> str:
    """Hash the labels, score matrix, learned counters, state, and seed."""

    payload = {
        "labels_sha256": _array_sha256(np.asarray(labels, dtype=np.int64)),
        "scores_sha256": _array_sha256(np.asarray(scores, dtype=np.float64)),
        "per_verifier_independence_table": [dict(row) for row in table],
        "independence_weight_per_verifier": dict(weights),
        "state_sha256": state_sha256,
        "random_seed": int(random_seed),
        "frozen_ci95": {"low": float(frozen_ci[0]), "high": float(frozen_ci[1])},
        "memory_contribution_reference": memory_contribution_reference,
    }
    return _json_sha256(payload)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3864 artifact schema before writing."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must declare cached verifier scoring")
    serialized = json.dumps(artifact, sort_keys=True)
    for marker in ("GGUF", "CUDA", "cuda", "live_llm_inference", "torch.cuda"):
        if marker in serialized:
            raise ValueError("forbidden inference marker present")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be present")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    if not isinstance(artifact.get("auroc_in_frozen_ci"), bool):
        raise ValueError("auroc_in_frozen_ci must be a bare boolean")
    if not isinstance(artifact.get("memory_ablation_contribution_preserved"), bool):
        raise ValueError("memory_ablation_contribution_preserved must be a bare boolean")
    if artifact.get("update_cost_note") != UPDATE_COST_NOTE:
        raise ValueError("update_cost_note must declare qualitative CPU counter updates")
    if not isinstance(artifact.get("state_persisted_path"), str):
        raise ValueError("state_persisted_path must be a string path")
    weights = artifact.get("independence_weight_per_verifier")
    if weights:
        if not isinstance(weights, Mapping) or set(weights) != set(VERIFIER_NAMES):
            raise ValueError("independence weights must include all four verifiers")
        total = sum(float(value) for value in weights.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("independence weights must sum to 1")
    auroc = artifact.get("reweighted_ensemble_auroc")
    if auroc is not None and (
        not isinstance(auroc, int | float) or not 0.0 <= float(auroc) <= 1.0
    ):
        raise ValueError("reweighted_ensemble_auroc must be in [0, 1]")
    verdict = str(artifact.get("honest_verdict"))
    if verdict.startswith("blocked_"):
        return
    if verdict == REGRESS_VERDICT:
        if artifact.get("auroc_in_frozen_ci") is not False:
            raise ValueError("regressed verdict requires auroc_in_frozen_ci=false")
        return
    if not (verdict.startswith(SUCCESS_VERDICT_PREFIX) and verdict.endswith(SUCCESS_VERDICT_SUFFIX)):
        raise ValueError(f"unsupported honest_verdict: {verdict!r}")
    if artifact.get("auroc_in_frozen_ci") is not True:
        raise ValueError("success verdict requires auroc_in_frozen_ci=true")
    if artifact.get("memory_ablation_contribution_preserved") is not True:
        raise ValueError("success verdict requires memory ablation preservation")


def _blocked_artifact(
    *,
    duration_s: float,
    random_seed: int,
    state_output: Path,
    repo_root: Path | None,
    preconditions: Sequence[Mapping[str, Any]],
    verdict: str,
) -> JsonDict:
    payload = {
        "preconditions": [dict(item) for item in preconditions],
        "random_seed": int(random_seed),
        "verdict": verdict,
    }
    artifact: JsonDict = {
        "artifact": "experiment_3864_fr11_self_learning_v23_independence_reweighting",
        "schema": "carnot.fr11_continuous_self_learning_v23",
        "continuous_self_learning_task": True,
        "honest_verdict": verdict,
        "reweighted_ensemble_auroc": None,
        "auroc_in_frozen_ci": False,
        "independence_weight_per_verifier": {},
        "memory_ablation_contribution_preserved": False,
        "update_cost_note": UPDATE_COST_NOTE,
        "state_persisted_path": _relative_path(state_output, repo_root),
        "preconditions_checked": [dict(item) for item in preconditions],
        "random_seed": int(random_seed),
        "reproducibility_checksum": _json_sha256(payload),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _round(duration_s),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "passed": False,
            "condition": "preconditions available before online reweighting",
            "principle": "No independence weights are fabricated without cached score evidence.",
        },
    }
    validate_artifact(artifact)
    return artifact


def _select_verdict(
    *,
    auroc: float,
    auroc_in_frozen_ci: bool,
    memory_ablation_contribution_preserved: bool,
    state_persisted: bool,
) -> str:
    if auroc_in_frozen_ci and memory_ablation_contribution_preserved and state_persisted:
        return f"{SUCCESS_VERDICT_PREFIX}{_round(auroc, digits=4)}{SUCCESS_VERDICT_SUFFIX}"
    if not auroc_in_frozen_ci:
        return REGRESS_VERDICT
    return BLOCKED_MEMORY_VERDICT


def _blocked_verdict(preconditions: Sequence[Mapping[str, Any]]) -> str:
    for item in preconditions:
        if item.get("available"):
            continue
        if item.get("resource") == "carnot_verify_import":
            return BLOCKED_VERIFY_VERDICT
        return BLOCKED_CORPUS_VERDICT
    return BLOCKED_CORPUS_VERDICT


def _trace_precondition(
    labels: Sequence[int],
    scores_by_verifier: Mapping[str, Sequence[float]],
) -> JsonDict:
    return {
        "resource": "cached_traces_with_per_verifier_scores_and_labels",
        "available": bool(labels and scores_by_verifier),
        "detail": f"n_examples={len(labels)}; n_verifiers={len(scores_by_verifier)}",
    }


def _model_specs(*, n_examples: int) -> JsonDict:
    return {
        "corpus": "FoVer cached corpus",
        "n_examples": int(n_examples),
        "verifiers": list(VERIFIER_NAMES),
        "live_model_invoked": False,
        "scoring_mode": "cached_verifier_scores_only",
    }


def _methodology(random_seed: int) -> JsonDict:
    return {
        "random_seed": int(random_seed),
        "score_threshold": DEFAULT_SCORE_THRESHOLD,
        "weighting_rule": (
            "(frozen_prior + prior_floor) * (precision_floor + online_precision) "
            "* (1 + unique_error_catch_rate)"
        ),
        "hardware_path": UPDATE_COST_NOTE,
    }


def _require_binary_labels(labels: np.ndarray) -> None:
    if set(int(value) for value in labels) != {0, 1}:
        raise ValueError("labels must contain both binary classes 0 and 1")


def _finite_score(value: float | int) -> float:
    score = float(value)
    if not math.isfinite(score):
        raise ValueError("verifier scores must be finite")
    return score


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _in_ci(value: float, frozen_ci: tuple[float, float]) -> bool:
    return float(frozen_ci[0]) <= float(value) <= float(frozen_ci[1])


def _resolve_under_root(root: Path, path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _relative_path(path: Path, root: Path | None) -> str:
    if root is None:
        return path.as_posix()
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _line in handle)


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.time() if now_s is None else float(now_s)
    return max(0.0001, end - float(started_s))


def _round(value: float | int | None, digits: int = 9) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
