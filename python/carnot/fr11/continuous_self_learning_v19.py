"""FR-11 continuous self-learning v19 Tier-3 predictive verification.

Spec: REQ-LEARN-3788, SCENARIO-LEARN-3788.

The v19 forward difference is a predictive head. It scores the cached FoVer
corpus with the four frozen headline verifiers, turns that verifier feature
signal into the 258-dimensional input expected by ``FR11ExtendedJEPA``, trains
the predictor on a train split, and reports held-out step-error AUROC. The
frozen 0.9131 scoring ensemble is not modified or replaced.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from carnot.fr11 import continuous_self_learning_v17 as v17


JsonDict = dict[str, Any]

OUTPUT_REL_PATH = Path(
    "results/experiment_3788_fr11_self_learning_v19_tier3_predictive.json"
)
PREDICTOR_STATE_REL_PATH = Path(
    "results/experiment_3788_fr11_v19_tier3_predictive_jepa_state.npz"
)
EXP2837_REL_PATH = Path("results/experiment_2837_fover_memory_leakage_v3.json")
DEFAULT_RANDOM_SEED = 3788
DEFAULT_CORPUS_RANDOM_SEED = 3788
DEFAULT_N_EXAMPLES = 1000
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
MEMORY_CONTRIBUTION_ROUNDED_MIN = 0.0185
FROZEN_HEADLINE_ROUNDED = 0.9131
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: reads cached verifier scores, no live model)."
)
SUCCESS_VERDICT_PREFIX = (
    "complete: fr11_v19_tier3_predictive_verifier_trained_predictive_auroc_"
)
SUCCESS_VERDICT_SUFFIX = (
    "_headline_ensemble_unchanged_memory_contribution_preserved_state_persisted"
)
BLOCKED_INTERPRETER_VERDICT = "blocked_interpreter_runtime"
BLOCKED_CORPUS_VERDICT = "blocked_fover_corpus_missing: no corpus to train on"
BLOCKED_SCORING_VERDICT = "blocked_fover_scores_missing: no corpus to train on"
VERIFIER_NAMES = v17.VERIFIER_NAMES
JEPA_EMBED_DIM = 258
JEPA_DOMAINS = ("arithmetic", "code", "logic")

score_fover_corpus = v17.score_fover_corpus
score_matrix = v17.score_matrix
probe_cached_trace_preconditions = v17.probe_cached_trace_preconditions

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "predictive_auroc",
    "train_test_split_sizes",
    "headline_ensemble_unchanged",
    "memory_contribution_preserved",
    "is_tier3_not_tier1_or_tier2",
    "tracker_state_persisted",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal prefix; the self-learning outcome; blocked_<resource> if a "
        "precondition failed."
    ),
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates (principle: reads cached "
        "verifier scores, no live model)."
    ),
    "predictive_auroc": (
        "The Tier-3 predictor's held-out AUROC -- the core deliverable; "
        "reported honestly even if below the full ensemble."
    ),
    "train_test_split_sizes": (
        "Sample-size hygiene -- the held-out split sizes so the AUROC is "
        "interpretable (and leakage is auditable)."
    ),
    "headline_ensemble_unchanged": (
        "BARE bool, true -- the frozen 0.9131 scoring ensemble is UNTOUCHED; "
        "Tier-3 adds a predictive head, it does not replace the discriminator."
    ),
    "memory_contribution_preserved": (
        "BARE bool -- the +0.0185 fr11_session_memory contribution survives "
        "(the load-bearing invariant)."
    ),
    "is_tier3_not_tier1_or_tier2": (
        "BARE bool, true -- confirms v19 is Tier-3 predictive verification, "
        "NOT a re-run of v17 (Tier-1) or v18 (Tier-2)."
    ),
    "tracker_state_persisted": (
        "BARE bool, true -- the Tier-3 predictor was persisted so a future "
        "milestone resumes it (continuous, not one-shot)."
    ),
    "model_specs": (
        "Names the corpus + 4 verifiers + the JEPA predictor -- honest substrate."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": (
        "Wall-clock plausibility floor; verifier-scoring + a small CPU JAX "
        "train is modest."
    ),
}


def build_artifact(
    repo_root: Path | str,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    n_examples: int = DEFAULT_N_EXAMPLES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    corpus_random_seed: int = DEFAULT_CORPUS_RANDOM_SEED,
    state_path: Path | str = PREDICTOR_STATE_REL_PATH,
    n_epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> JsonDict:
    """Build Exp 3788 from cached FoVer rows and verifier scores."""

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    state_output = _resolve_under_root(root, state_path)
    corpus_path = root / "data" / "fover_corpus.jsonl"
    preconditions = [
        _interpreter_precondition(),
        _fover_corpus_precondition(corpus_path, n_examples=n_examples),
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
    preconditions.extend(probe_cached_trace_preconditions(root, n_examples=n_examples))
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
    except Exception as exc:  # noqa: BLE001 - cached scoring failure is terminal.
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            state_output=state_output,
            repo_root=root,
            preconditions=[
                *preconditions,
                {
                    "resource": "cached_trace_scoring",
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
        state_path=state_output,
        repo_root=root,
        preconditions=preconditions,
        memory_contribution_reference=load_memory_contribution_reference(root),
        headline_ensemble_reference=load_headline_ensemble_reference(root),
        corpus_absolute_path=corpus_path.resolve(),
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        test_fraction=test_fraction,
    )


def build_artifact_from_scores(
    *,
    labels: Sequence[int],
    scores_by_verifier: Mapping[str, Sequence[float]],
    started_s: float,
    now_s: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    state_path: Path | str = PREDICTOR_STATE_REL_PATH,
    repo_root: Path | str | None = None,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
    memory_contribution_reference: float | None = None,
    headline_ensemble_reference: float | None = None,
    corpus_absolute_path: Path | str | None = None,
    n_epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> JsonDict:
    """Train the Tier-3 predictor and report held-out predictive AUROC."""

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

    run = train_tier3_predictor(
        labels=labels_arr.tolist(),
        scores_by_verifier=scores_by_verifier,
        random_seed=random_seed,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        test_fraction=test_fraction,
    )
    predictive_auroc = float(run["predictive_auroc"])
    split_sizes = dict(run["train_test_split_sizes"])
    state_sha = persist_predictor_state(
        run["predictor"],
        state_output,
        metadata={
            "random_seed": int(random_seed),
            "predictive_auroc": _round(predictive_auroc),
            "train_test_split_sizes": split_sizes,
            "verifier_names": list(VERIFIER_NAMES),
            "predictor": "FR11ExtendedJEPA",
            "input_dim": JEPA_EMBED_DIM,
        },
    )
    headline_ok = headline_ensemble_unchanged(headline_ensemble_reference)
    memory_ok = memory_contribution_preserved(memory_contribution_reference)
    verdict = _success_verdict(predictive_auroc)
    artifact: JsonDict = {
        "artifact": "experiment_3788_fr11_self_learning_v19_tier3_predictive",
        "schema": "carnot.fr11_continuous_self_learning_v19",
        "continuous_self_learning_task": True,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "predictive_auroc": _round(predictive_auroc),
        "train_test_split_sizes": split_sizes,
        "headline_ensemble_unchanged": bool(headline_ok),
        "memory_contribution_preserved": bool(memory_ok),
        "is_tier3_not_tier1_or_tier2": True,
        "tracker_state_persisted": state_output.is_file(),
        "model_specs": _model_specs(
            corpus_absolute_path,
            n_examples=len(labels_arr),
        ),
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum(
            labels=labels_arr,
            scores=matrix,
            train_indices=run["train_indices"],
            test_indices=run["test_indices"],
            test_probabilities=run["test_probabilities"],
            predictive_auroc=predictive_auroc,
            random_seed=random_seed,
            predictor_state_sha256=state_sha,
            memory_contribution_reference=memory_contribution_reference,
            headline_ensemble_reference=headline_ensemble_reference,
        ),
        "duration_s": _round(_duration(float(started_s), now_s)),
        "n_examples": int(len(labels_arr)),
        "n_samples": int(split_sizes["test"]),
        "verifier_names": list(VERIFIER_NAMES),
        "predictor_state_path": _relative_path(state_output, root),
        "predictor_state_sha256": state_sha,
        "predictor_training": {
            "train_macro_auroc": _round(
                float(run["train_metrics"].get("macro_auroc", 0.0))
            ),
            "per_domain_predictive_auroc": {
                key: _round(value)
                for key, value in run["per_domain_predictive_auroc"].items()
            },
            "n_epochs": int(n_epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(lr),
            "test_fraction": float(test_fraction),
        },
        "frozen_headline_ensemble_auroc": headline_ensemble_reference,
        "memory_contribution_reference": memory_contribution_reference,
        "methodology": _methodology(random_seed, corpus_absolute_path),
        "preconditions_checked": [dict(item) for item in preconditions or []],
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "passed": bool(headline_ok and memory_ok and state_output.is_file()),
            "condition": (
                "held-out predictive AUROC reported AND frozen 0.9131 ensemble "
                "unchanged AND Exp 2837 rounded memory contribution >= +0.0185 "
                "AND Tier-3 predictor state persisted"
            ),
            "principle": (
                "Tier-3 may be weaker than the scorer, but it must be a real "
                "additive predictor and must not regress the banked discriminator."
            ),
        },
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(
    repo_root: Path | str,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    state_path: Path | str = PREDICTOR_STATE_REL_PATH,
    labels: Sequence[int] | None = None,
    scores_by_verifier: Mapping[str, Sequence[float]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    n_epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> Path:
    """Build, validate, and write the Exp 3788 JSON artifact."""

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    resolved_state_path = _resolve_under_root(root, state_path)
    if labels is not None or scores_by_verifier is not None:
        artifact = build_artifact_from_scores(
            labels=labels or [],
            scores_by_verifier=scores_by_verifier or {},
            started_s=start,
            now_s=now_s,
            state_path=resolved_state_path,
            repo_root=root,
            memory_contribution_reference=load_memory_contribution_reference(root),
            headline_ensemble_reference=load_headline_ensemble_reference(root),
            corpus_absolute_path=(root / "data" / "fover_corpus.jsonl").resolve(),
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            test_fraction=test_fraction,
        )
    else:
        artifact = build_artifact(
            root,
            started_s=start,
            now_s=now_s,
            state_path=resolved_state_path,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            test_fraction=test_fraction,
        )
    output = _resolve_under_root(root, output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def build_predictor_pairs(
    labels: Sequence[int],
    scores_by_verifier: Mapping[str, Sequence[float]],
    *,
    verifier_names: Sequence[str] = VERIFIER_NAMES,
) -> list[JsonDict]:
    """Build ``FR11ExtendedJEPA`` training pairs from verifier score features."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    matrix = score_matrix(scores_by_verifier, verifier_names)
    if matrix.shape[0] != len(labels_arr):
        raise ValueError("labels and verifier scores must have the same length")
    pairs: list[JsonDict] = []
    for row, label in zip(matrix, labels_arr, strict=True):
        violated = bool(int(label) == 1)
        pair: JsonDict = {
            "embedding": verifier_feature_embedding(row, verifier_names=verifier_names),
        }
        for domain in JEPA_DOMAINS:
            pair[f"violated_{domain}"] = violated
        pairs.append(pair)
    return pairs


def verifier_feature_embedding(
    row_scores: Sequence[float],
    *,
    verifier_names: Sequence[str] = VERIFIER_NAMES,
) -> list[float]:
    """Return the 258-wide JEPA feature vector for one four-verifier row."""

    scores = np.asarray(row_scores, dtype=np.float64)
    names = tuple(verifier_names)
    if scores.shape != (len(names),):
        raise ValueError("row_scores must contain one value per verifier")
    if not np.isfinite(scores).all():
        raise ValueError("verifier score matrix must be finite")
    weights = v17.v10.exp3644.carnot_current_weights(names)
    ensemble = float(np.clip(scores @ weights, 0.0, 1.0))
    pairwise_diffs = [
        float(scores[i] - scores[j])
        for i in range(len(scores))
        for j in range(i + 1, len(scores))
    ]
    pairwise_products = [
        float(scores[i] * scores[j])
        for i in range(len(scores))
        for j in range(i + 1, len(scores))
    ]
    base = np.asarray(
        [
            *scores.tolist(),
            float(np.mean(scores)),
            float(np.std(scores)),
            float(np.min(scores)),
            float(np.max(scores)),
            ensemble,
            *pairwise_diffs,
            *pairwise_products,
            *(scores - 0.5).tolist(),
            *np.abs(scores - 0.5).tolist(),
            *(scores - ensemble).tolist(),
        ],
        dtype=np.float64,
    )
    channels = np.concatenate(
        [
            base,
            np.square(base),
            np.sqrt(np.abs(base)),
            np.sin(np.pi * base),
            np.cos(np.pi * base),
            np.tanh(2.0 * base),
        ]
    )
    if channels.size < JEPA_EMBED_DIM:
        repeats = int(math.ceil(JEPA_EMBED_DIM / float(channels.size)))
        channels = np.tile(channels, repeats)
    embedding = channels[:JEPA_EMBED_DIM].astype(np.float32)
    if not np.isfinite(embedding).all():  # pragma: no cover - finite transforms above.
        raise ValueError("predictor embedding must be finite")
    return embedding.tolist()


def train_tier3_predictor(
    *,
    labels: Sequence[int],
    scores_by_verifier: Mapping[str, Sequence[float]],
    random_seed: int = DEFAULT_RANDOM_SEED,
    n_epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> JsonDict:
    """Train ``FR11ExtendedJEPA`` and evaluate held-out step-error AUROC."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    _require_binary_labels(labels_arr)
    pairs = build_predictor_pairs(labels_arr.tolist(), scores_by_verifier)
    train_indices, test_indices = stratified_train_test_indices(
        labels_arr,
        test_fraction=test_fraction,
        random_seed=random_seed,
    )
    train_pairs = [pairs[int(index)] for index in train_indices]
    predictor = _new_predictor(random_seed)

    old_random_state = np.random.get_state()
    np.random.seed(int(random_seed) % (2**32 - 1))
    try:
        train_metrics = predictor.train(
            train_pairs,
            n_epochs=int(n_epochs),
            lr=float(lr),
            batch_size=int(batch_size),
            seed=int(random_seed),
        )
    finally:
        np.random.set_state(old_random_state)

    test_labels = labels_arr[test_indices]
    per_domain_probs: dict[str, np.ndarray] = {}
    for domain in JEPA_DOMAINS:
        per_domain_probs[domain] = np.asarray(
            [
                predictor.predict(pairs[int(index)]["embedding"])[domain]
                for index in test_indices
            ],
            dtype=np.float64,
        )
    test_probabilities = np.mean(
        np.column_stack([per_domain_probs[domain] for domain in JEPA_DOMAINS]),
        axis=1,
    )
    predictive_auroc = _roc_auc(test_labels, test_probabilities)
    per_domain_auroc = {
        domain: _roc_auc(test_labels, probs)
        for domain, probs in per_domain_probs.items()
    }
    return {
        "predictor": predictor,
        "predictive_auroc": float(predictive_auroc),
        "train_test_split_sizes": _split_sizes(labels_arr, train_indices, test_indices),
        "train_indices": train_indices,
        "test_indices": test_indices,
        "test_labels": test_labels,
        "test_probabilities": test_probabilities,
        "per_domain_predictive_auroc": per_domain_auroc,
        "train_metrics": dict(train_metrics),
    }


def stratified_train_test_indices(
    labels: Sequence[int] | np.ndarray,
    *,
    test_fraction: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic stratified train/test row indices."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    _require_binary_labels(labels_arr)
    fraction = float(test_fraction)
    if not 0.0 < fraction < 0.5:
        raise ValueError("test_fraction must be between 0 and 0.5")
    rng = np.random.default_rng(int(random_seed))
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in (0, 1):
        indices = np.flatnonzero(labels_arr == label)
        if len(indices) < 2:
            raise ValueError("each binary class needs at least two examples")
        rng.shuffle(indices)
        n_test = max(1, int(round(len(indices) * fraction)))
        n_test = min(n_test, len(indices) - 1)
        test_parts.append(indices[:n_test])
        train_parts.append(indices[n_test:])
    train_indices = np.concatenate(train_parts).astype(np.int64)
    test_indices = np.concatenate(test_parts).astype(np.int64)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices


def persist_predictor_state(
    predictor: Any,
    path: Path | str,
    *,
    metadata: Mapping[str, Any],
) -> str:
    """Persist JEPA predictor parameters and return the file checksum."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    params = _predictor_params(predictor)
    arrays = {key: np.asarray(value, dtype=np.float32) for key, value in params.items()}
    arrays["metadata_json"] = np.asarray(
        json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"))
    )
    np.savez_compressed(output, **arrays)
    return _file_sha256(output)


def load_predictor_state(path: Path | str) -> JsonDict:
    """Load a persisted JEPA predictor state."""

    import jax.numpy as jnp

    loaded = np.load(Path(path), allow_pickle=False)
    try:
        metadata = json.loads(str(loaded["metadata_json"].item()))
        predictor = _new_predictor(int(metadata.get("random_seed", 0)))
        params = {
            key: jnp.asarray(loaded[key], dtype=jnp.float32)
            for key in ("w1", "b1", "w2", "b2", "w3", "b3")
        }
        predictor._params = params
    finally:
        loaded.close()
    return {"predictor": predictor, "metadata": metadata}


def poison_predictor_state_for_test(source_path: Path | str, output_path: Path | str) -> None:
    """Write a deliberately changed predictor state for anti-poison tests."""

    loaded = np.load(Path(source_path), allow_pickle=False)
    try:
        arrays = {key: loaded[key] for key in loaded.files}
        arrays["b3"] = np.asarray(arrays["b3"], dtype=np.float32) + np.asarray(
            [7.0, 0.0, 0.0],
            dtype=np.float32,
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **arrays)
    finally:
        loaded.close()


def memory_contribution_preserved(reference: float | None) -> bool:
    """Return true when the Exp 2837 FR-11 memory contribution remains present."""

    if reference is None:
        return False
    return round(float(reference), 4) >= MEMORY_CONTRIBUTION_ROUNDED_MIN


def headline_ensemble_unchanged(reference: float | None) -> bool:
    """Return true when the frozen Exp 2837 headline AUROC remains 0.9131."""

    if reference is None:
        return False
    return round(float(reference), 4) == FROZEN_HEADLINE_ROUNDED


def load_memory_contribution_reference(repo_root: Path | str) -> float | None:
    """Load the Exp 2837 FR-11 memory contribution reference if present."""

    payload = _load_exp2837(repo_root)
    value = payload.get("learning_contribution") if payload else None
    return float(value) if isinstance(value, int | float) else None


def load_headline_ensemble_reference(repo_root: Path | str) -> float | None:
    """Load the Exp 2837 frozen headline ensemble AUROC if present."""

    payload = _load_exp2837(repo_root)
    value = payload.get("condition_a_production_auroc_mean") if payload else None
    return float(value) if isinstance(value, int | float) else None


def reproducibility_checksum(
    *,
    labels: np.ndarray | Sequence[int],
    scores: np.ndarray | Sequence[Sequence[float]],
    train_indices: np.ndarray | Sequence[int],
    test_indices: np.ndarray | Sequence[int],
    test_probabilities: np.ndarray | Sequence[float],
    predictive_auroc: float,
    random_seed: int,
    predictor_state_sha256: str,
    memory_contribution_reference: float | None,
    headline_ensemble_reference: float | None,
) -> str:
    """Hash the measured labels, features, split, predictions, and state."""

    payload = {
        "labels_sha256": _array_sha256(np.asarray(labels, dtype=np.int64)),
        "scores_sha256": _array_sha256(np.asarray(scores, dtype=np.float64)),
        "train_indices_sha256": _array_sha256(np.asarray(train_indices, dtype=np.int64)),
        "test_indices_sha256": _array_sha256(np.asarray(test_indices, dtype=np.int64)),
        "test_probabilities_sha256": _array_sha256(
            np.asarray(test_probabilities, dtype=np.float64)
        ),
        "predictive_auroc": _round(predictive_auroc),
        "random_seed": int(random_seed),
        "predictor_state_sha256": predictor_state_sha256,
        "memory_contribution_reference": memory_contribution_reference,
        "headline_ensemble_reference": headline_ensemble_reference,
    }
    return _json_sha256(payload)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3788 artifact schema before writing."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be present")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must declare verifier-scoring substrate")
    serialized = json.dumps(artifact, sort_keys=True)
    for marker in ("GGUF", "CUDA", "cuda", "live_llm_inference", "torch.cuda"):
        if marker in serialized:
            raise ValueError("forbidden inference marker present")
    if not isinstance(artifact.get("train_test_split_sizes"), Mapping):
        raise ValueError("train_test_split_sizes must be a mapping")
    if not isinstance(artifact.get("headline_ensemble_unchanged"), bool):
        raise ValueError("headline_ensemble_unchanged must be a bare boolean")
    if not isinstance(artifact.get("memory_contribution_preserved"), bool):
        raise ValueError("memory_contribution_preserved must be a bare boolean")
    if artifact.get("is_tier3_not_tier1_or_tier2") is not True:
        raise ValueError("is_tier3_not_tier1_or_tier2 must be true")
    if not isinstance(artifact.get("tracker_state_persisted"), bool):
        raise ValueError("tracker_state_persisted must be boolean")
    if not isinstance(artifact.get("duration_s"), int | float):
        raise ValueError("duration_s must be numeric")

    verdict = artifact.get("honest_verdict")
    if verdict in {
        BLOCKED_INTERPRETER_VERDICT,
        BLOCKED_CORPUS_VERDICT,
        BLOCKED_SCORING_VERDICT,
    }:
        if artifact.get("predictive_auroc") is not None:
            raise ValueError("blocked artifact must not fabricate predictive_auroc")
        if artifact.get("train_test_split_sizes") != {"train": 0, "test": 0}:
            raise ValueError("blocked artifact must not fabricate split sizes")
        if artifact.get("tracker_state_persisted") is not False:
            raise ValueError("blocked artifact must not fabricate predictor state")
        return

    if not isinstance(verdict, str) or not (
        verdict.startswith(SUCCESS_VERDICT_PREFIX)
        and verdict.endswith(SUCCESS_VERDICT_SUFFIX)
    ):
        raise ValueError(f"unsupported honest_verdict: {verdict!r}")
    auroc = artifact.get("predictive_auroc")
    if not isinstance(auroc, int | float) or not 0.0 <= float(auroc) <= 1.0:
        raise ValueError("predictive_auroc must be in [0, 1]")
    split_sizes = artifact.get("train_test_split_sizes")
    if int(split_sizes.get("train", 0)) <= 0 or int(split_sizes.get("test", 0)) <= 0:
        raise ValueError("train_test_split_sizes must include positive train and test")
    if float(auroc) == 1.0 and int(split_sizes.get("test", 0)) >= 50:
        raise ValueError("predictive_auroc is implausibly perfect on a non-trivial split")
    if artifact.get("headline_ensemble_unchanged") is not True:
        raise ValueError("headline_ensemble_unchanged must be true for success")
    if artifact.get("memory_contribution_preserved") is not True:
        raise ValueError("memory_contribution_preserved must be true for success")
    if artifact.get("tracker_state_persisted") is not True:
        raise ValueError("tracker_state_persisted must be true for success")
    specs = artifact.get("model_specs")
    if not isinstance(specs, Mapping) or specs.get("predictive_head") != "FR11ExtendedJEPA":
        raise ValueError("model_specs must name the FR11ExtendedJEPA predictor")
    if specs.get("verifiers") != list(VERIFIER_NAMES):
        raise ValueError("model_specs must name the four FoVer verifiers")


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
        "artifact": "experiment_3788_fr11_self_learning_v19_tier3_predictive",
        "schema": "carnot.fr11_continuous_self_learning_v19",
        "continuous_self_learning_task": True,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "predictive_auroc": None,
        "train_test_split_sizes": {"train": 0, "test": 0},
        "headline_ensemble_unchanged": False,
        "memory_contribution_preserved": False,
        "is_tier3_not_tier1_or_tier2": True,
        "tracker_state_persisted": False,
        "model_specs": _model_specs(None, n_examples=0),
        "random_seed": int(random_seed),
        "reproducibility_checksum": _json_sha256(payload),
        "duration_s": _round(duration_s),
        "n_examples": 0,
        "n_samples": 0,
        "verifier_names": list(VERIFIER_NAMES),
        "predictor_state_path": _relative_path(state_output, repo_root),
        "predictor_state_sha256": None,
        "predictor_training": {},
        "frozen_headline_ensemble_auroc": None,
        "memory_contribution_reference": None,
        "methodology": _methodology(random_seed, None),
        "preconditions_checked": [dict(item) for item in preconditions],
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "passed": False,
            "condition": "FoVer scores and labels are present before predictor training",
            "principle": "No Tier-3 predictive AUROC is emitted without corpus evidence.",
        },
    }
    validate_artifact(artifact)
    return artifact


def _interpreter_precondition() -> JsonDict:
    packages = ("jax", "numpy", "sklearn")
    loaded: list[str] = []
    missing: list[str] = []
    for package in packages:
        try:
            importlib.import_module(package)
            loaded.append(package)
        except Exception:  # noqa: BLE001 - reported as blocked precondition.
            missing.append(package)
    try:
        module = importlib.import_module("carnot.fr11.tier3_jepa")
        jepa_importable = callable(getattr(module, "FR11ExtendedJEPA", None))
    except Exception:  # noqa: BLE001 - reported as blocked precondition.
        jepa_importable = False
    executable = Path(sys.executable).as_posix()
    is_venv = ".venv/bin/python" in executable or executable.endswith("/.venv/bin/python")
    available = bool(is_venv and not missing and jepa_importable)
    return {
        "resource": "interpreter_runtime",
        "available": available,
        "detail": (
            f"executable={executable}; loaded={','.join(loaded)}; "
            f"missing={','.join(missing) or 'none'}; "
            f"FR11ExtendedJEPA_importable={jepa_importable}"
        ),
    }


def _fover_corpus_precondition(corpus_path: Path, *, n_examples: int) -> JsonDict:
    absolute = corpus_path.resolve()
    if not corpus_path.is_file():
        return {
            "resource": "fover_corpus_absolute_path",
            "available": False,
            "detail": f"{absolute}; missing; no corpus to train on",
        }
    n_rows = _line_count(corpus_path)
    return {
        "resource": "fover_corpus_absolute_path",
        "available": n_rows >= int(n_examples),
        "detail": f"{absolute}; line_count={n_rows}; required>={int(n_examples)}",
    }


def _blocked_verdict(preconditions: Sequence[Mapping[str, Any]]) -> str:
    for item in preconditions:
        if item.get("available"):
            continue
        resource = str(item.get("resource", "resource"))
        if resource == "interpreter_runtime":
            return BLOCKED_INTERPRETER_VERDICT
        if "fover_corpus" in resource:
            return BLOCKED_CORPUS_VERDICT
    return BLOCKED_SCORING_VERDICT


def _trace_precondition(
    labels: Sequence[int],
    scores_by_verifier: Mapping[str, Sequence[float]],
) -> JsonDict:
    return {
        "resource": "cached_traces_with_per_verifier_scores_and_labels",
        "available": bool(labels and scores_by_verifier),
        "detail": f"n_examples={len(labels)}; n_verifiers={len(scores_by_verifier)}",
    }


def _model_specs(corpus_absolute_path: Path | str | None, *, n_examples: int) -> JsonDict:
    return {
        "corpus": "FoVer cached corpus",
        "corpus_absolute_path": str(corpus_absolute_path) if corpus_absolute_path else None,
        "n_examples_requested": int(n_examples),
        "verifiers": list(VERIFIER_NAMES),
        "live_model_invoked": False,
        "scoring_mode": "cached_verifier_scores_only",
        "predictive_head": "FR11ExtendedJEPA",
        "predictive_head_module": "carnot.fr11.tier3_jepa",
        "predictor_input_dim": JEPA_EMBED_DIM,
        "predictor_outputs": list(JEPA_DOMAINS),
        "tier": "Tier-3 predictive verification",
        "ensemble_status": "frozen_0.9131_scoring_ensemble_unchanged",
        "execution_path": "cpu_validated_small_predictor_future_gpu_npu_path",
    }


def _methodology(random_seed: int, corpus_absolute_path: Path | str | None) -> JsonDict:
    return {
        "random_seed": int(random_seed),
        "corpus_absolute_path": str(corpus_absolute_path) if corpus_absolute_path else None,
        "scoring_protocol": "ops/reproduction-runbook-fover-headline.md",
        "domain_keys": list(VERIFIER_NAMES),
        "lineage": "v19_tier3_predictive_verification_not_v17_tier1_or_v18_tier2",
        "feature_builder": "four_verifier_scores_to_258_dim_FR11ExtendedJEPA_embedding",
        "predictor_api": "FR11ExtendedJEPA.__init__/train/predict/energy",
        "heldout_metric": "sklearn.metrics.roc_auc_score",
    }


def _new_predictor(seed: int) -> Any:
    module = importlib.import_module("carnot.fr11.tier3_jepa")
    predictor_cls = getattr(module, "FR11ExtendedJEPA")
    return predictor_cls(seed=int(seed))


def _predictor_params(predictor: Any) -> Mapping[str, Any]:
    params = getattr(predictor, "_params", None)
    if not isinstance(params, Mapping):
        raise ValueError("FR11ExtendedJEPA predictor parameters are unavailable")
    missing = [key for key in ("w1", "b1", "w2", "b2", "w3", "b3") if key not in params]
    if missing:
        raise ValueError(f"FR11ExtendedJEPA predictor parameters missing: {missing}")
    return params


def _roc_auc(labels: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray) -> float:
    metrics = importlib.import_module("sklearn.metrics")
    labels_arr = np.asarray(labels, dtype=np.int64)
    score_arr = np.asarray(scores, dtype=np.float64)
    _require_binary_labels(labels_arr)
    if len(labels_arr) != len(score_arr):
        raise ValueError("labels and scores must have the same length")
    return float(metrics.roc_auc_score(labels_arr, score_arr))


def _split_sizes(
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> JsonDict:
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    return {
        "train": int(len(train_indices)),
        "test": int(len(test_indices)),
        "train_positive": int(np.sum(train_labels == 1)),
        "train_negative": int(np.sum(train_labels == 0)),
        "test_positive": int(np.sum(test_labels == 1)),
        "test_negative": int(np.sum(test_labels == 0)),
    }


def _success_verdict(predictive_auroc: float) -> str:
    return f"{SUCCESS_VERDICT_PREFIX}{float(predictive_auroc):.4f}{SUCCESS_VERDICT_SUFFIX}"


def _load_exp2837(repo_root: Path | str) -> JsonDict | None:
    path = Path(repo_root) / EXP2837_REL_PATH
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


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


def _require_binary_labels(labels: np.ndarray) -> None:
    if set(int(value) for value in labels) != {0, 1}:
        raise ValueError("labels must contain both binary classes 0 and 1")


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.time() if now_s is None else float(now_s)
    return max(0.0001, end - float(started_s))


def _round(value: float | int | None, digits: int = 9) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
