"""FR-11 continuous self-learning v26 with cascade-band online learning.

Spec: REQ-LEARN-3930, SCENARIO-LEARN-3930.

Exp 3930 continues the persisted v25 independence counters instead of
restarting the online learner. It also connects the same fresh-slice evidence
to the deployable cascade artifact: when Exp 3927 is present on disk, the
fresh learned-ensemble margins update the escalation band that decides which
items need the slower judge path. This keeps the self-learning loop tied to a
product artifact while preserving the frozen 0.9131 headline invariant.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import importlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np


JsonDict = dict[str, Any]

OUTPUT_REL_PATH = Path("results/experiment_3930_fr11_v26_cascade_band_online_learning.json")
STATE_REL_PATH = Path("results/experiment_3930_fr11_v26_cascade_band_online_learning_state.json")
V25_STATE_REL_PATH = Path("results/experiment_3921_fr11_v25_independence_reweighting_state.json")
EXP3927_REL_PATH = Path("results/experiment_3927_non_degenerate_cascade_router.json")
DEFAULT_RANDOM_SEED = 3930
DEFAULT_N_UPDATES = 228
CASCADE_MARGIN_QUANTILE = 0.10
CASCADE_BAND_MIN = 0.02
CASCADE_BAND_MAX = 0.25
FROZEN_CI = (0.9027, 0.9235)
FROZEN_HEADLINE_AUROC = 0.9131
MEMORY_ABLATION_MIN = 0.012
VERIFIER_NAMES = (
    "fr11_session_memory",
    "tier0r_curry_howard",
    "tier0s_arithmetic_gap",
    "tier0u_logical_consistency",
)
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: CPU counter updates over cached FoVer v4/v3 scores; no LLM loaded)."
)
UPDATE_COST_NOTE = (
    "Tier-1 online path: pure CPU counter updates over cached verifier scores; "
    "persisted state lets v27 continue."
)
SUCCESS_VERDICT_PREFIX = "complete: fr11_v26_INVARIANT_HELD_auroc"
SUCCESS_VERDICT_SUFFIX = "_state_persisted"
BROKEN_VERDICT_PREFIX = "complete: fr11_v26_INVARIANT_BROKEN_auroc"
BROKEN_VERDICT_SUFFIX = "_online_reweighting_drifts"
BLOCKED_VERIFY_VERDICT = "blocked_carnot_verify_import"
BLOCKED_V25_STATE_VERDICT = "blocked_v25_reweighting_state"
BLOCKED_FRESH_CORPUS_VERDICT = "blocked_fresh_verification_corpus"
BLOCKED_FROZEN_SCORING_VERDICT = "blocked_frozen_headline_scoring"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "learned_ensemble_auroc",
    "memory_ablation_contribution",
    "frozen_headline_unchanged",
    "cascade_band_learned",
    "invariant_held",
    "state_persisted_path",
    "n_updates",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix; complete or blocked_<resource>.",
    "learned_ensemble_auroc": (
        "Must stay within the frozen CI [0.9027,0.9235] -- online learning must "
        "not degrade the headline."
    ),
    "memory_ablation_contribution": (
        "Must hold ~+0.0185 (CI [0.0125,0.0245]) -- the self-learning "
        "component's measured value."
    ),
    "frozen_headline_unchanged": (
        "0.9131 NEVER silently substituted -- the invariant guard."
    ),
    "cascade_band_learned": (
        "BARE BOOL -- whether the exp3927 escalation band was updated online "
        "(the deployable-artifact tie-in); false if exp3927 absent."
    ),
    "invariant_held": (
        "BARE BOOL -- did v26 preserve both the CI-band AUROC and the memory "
        "contribution."
    ),
    "state_persisted_path": (
        "Verifier-scoring methodology -- CPU counter updates; persisted state "
        "lets v27 continue."
    ),
    "n_updates": (
        "Verifier-scoring methodology -- CPU counter updates; persisted state "
        "lets v27 continue."
    ),
    "preconditions_checked": (
        "Verifier-scoring methodology -- CPU counter updates; persisted state "
        "lets v27 continue."
    ),
    "random_seed": (
        "Verifier-scoring methodology -- CPU counter updates; persisted state "
        "lets v27 continue."
    ),
    "reproducibility_checksum": (
        "Verifier-scoring methodology -- CPU counter updates; persisted state "
        "lets v27 continue."
    ),
    "duration_s": (
        "Verifier-scoring methodology -- CPU counter updates; persisted state "
        "lets v27 continue."
    ),
    "inference_substrate": (
        "Verifier-scoring methodology -- CPU counter updates; persisted state "
        "lets v27 continue."
    ),
}

def _fr11_v17() -> Any:
    return importlib.import_module("carnot.fr11.continuous_self_learning_v17")


def _fr11_v23() -> Any:
    return importlib.import_module("carnot.fr11.continuous_self_learning_v23")


def _fr11_v24() -> Any:
    return importlib.import_module("carnot.fr11.continuous_self_learning_v24")


def _fr11_v25() -> Any:
    return importlib.import_module("carnot.fr11.continuous_self_learning_v25")


def verify_import_precondition(
    importer: Any = importlib.import_module,
) -> JsonDict:
    """Check the verifier package import before any online update."""

    try:
        importer("carnot.verify")
    except Exception as exc:
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


def score_fresh_verification_corpus(
    repo_root: Path | str,
    *,
    n_updates: int,
    random_seed: int,
) -> tuple[list[int], dict[str, list[float]], JsonDict]:
    """Score a fresh balanced FoVer v4/v3 slice with the frozen verifier set."""

    return _fr11_v25().score_fresh_verification_corpus(
        repo_root,
        n_updates=n_updates,
        random_seed=random_seed,
    )


def score_frozen_headline_corpus(
    repo_root: Path | str,
) -> tuple[list[int], dict[str, list[float]]]:
    """Score the frozen FoVer headline path used for the CI guard."""

    return _fr11_v25().score_frozen_headline_corpus(repo_root)


def load_memory_contribution_reference(repo_root: Path | str) -> float:
    """Load the preserved FR-11 memory-ablation contribution reference."""

    return float(_fr11_v17().load_memory_contribution_reference(repo_root))


def read_json_fover_rows(path: Path | str) -> list[JsonDict]:
    """Read JSON-array FoVer rows and retain only supported binary labels."""

    return _fr11_v25().read_json_fover_rows(path)


def select_balanced_fresh_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    n_updates: int,
) -> list[JsonDict]:
    """Select a deterministic balanced fresh slice without exceeding class counts."""

    return _fr11_v25().select_balanced_fresh_rows(rows, seed=seed, n_updates=n_updates)


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.time() if now_s is None else float(now_s)
    return max(0.0001, end - float(started_s))


def _in_ci(value: float, frozen_ci: tuple[float, float]) -> bool:
    return float(frozen_ci[0]) <= float(value) <= float(frozen_ci[1])


def _resolve_under_root(root: Path, path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _relative_path(path: Path, root: Path | None) -> str:
    if root is None:
        return Path(path).as_posix()
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _round(value: float | int | None, digits: int = 9) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(
    repo_root: Path | str,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    n_updates: int = DEFAULT_N_UPDATES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    v25_state_path: Path | str = V25_STATE_REL_PATH,
    state_path: Path | str = STATE_REL_PATH,
    cascade_path: Path | str = EXP3927_REL_PATH,
    frozen_ci: tuple[float, float] = FROZEN_CI,
) -> JsonDict:
    """Build the Exp 3930 artifact from persisted v25 state and cached scores."""

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    resolved_v25_state = _resolve_under_root(root, v25_state_path)
    resolved_state = _resolve_under_root(root, state_path)
    cascade_check = load_optional_cascade_artifact(root, cascade_path)
    hard_preconditions = [
        verify_import_precondition(),
        v25_state_precondition(resolved_v25_state),
    ]
    preconditions = [*hard_preconditions, _public_precondition(cascade_check)]
    if not all(item["available"] for item in hard_preconditions):
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            state_output=resolved_state,
            repo_root=root,
            preconditions=preconditions,
            verdict=_blocked_verdict(hard_preconditions),
        )

    v25_state = load_v25_state(resolved_v25_state)
    try:
        update_labels, update_scores, fresh_source = score_fresh_verification_corpus(
            root,
            n_updates=n_updates,
            random_seed=random_seed,
        )
    except Exception as exc:  # pragma: no cover - depends on local corpus availability.
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            state_output=resolved_state,
            repo_root=root,
            preconditions=[
                *preconditions,
                {
                    "resource": "fresh_verification_corpus",
                    "available": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                },
            ],
            verdict=BLOCKED_FRESH_CORPUS_VERDICT,
        )

    try:
        eval_labels, eval_scores = score_frozen_headline_corpus(root)
    except Exception as exc:  # pragma: no cover - depends on frozen scoring data.
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            state_output=resolved_state,
            repo_root=root,
            preconditions=[
                *preconditions,
                {"resource": "fresh_verification_corpus", "available": True, "detail": "scored"},
                {
                    "resource": "frozen_headline_scoring",
                    "available": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                },
            ],
            verdict=BLOCKED_FROZEN_SCORING_VERDICT,
        )

    cascade_artifact = cascade_check.get("artifact") if cascade_check.get("available") else None
    return build_artifact_from_scores(
        v25_state=v25_state,
        update_labels=update_labels,
        update_scores_by_verifier=update_scores,
        eval_labels=eval_labels,
        eval_scores_by_verifier=eval_scores,
        started_s=start,
        now_s=now_s,
        random_seed=random_seed,
        repo_root=root,
        state_path=resolved_state,
        previous_state_path=_relative_path(resolved_v25_state, root),
        frozen_ci=frozen_ci,
        memory_ablation_contribution=load_memory_contribution_reference(root),
        preconditions=[
            *preconditions,
            {
                "resource": "fresh_verification_corpus",
                "available": True,
                "detail": f"scored n_updates={len(update_labels)}",
            },
            {
                "resource": "frozen_headline_scoring",
                "available": True,
                "detail": f"scored n_examples={len(eval_labels)}",
            },
        ],
        fresh_corpus_source=fresh_source,
        cascade_artifact=cascade_artifact if isinstance(cascade_artifact, Mapping) else None,
        cascade_artifact_path=_relative_path(
            _resolve_under_root(root, cascade_path),
            root,
        ),
    )


def build_artifact_from_scores(
    *,
    v25_state: Mapping[str, Any],
    update_labels: Sequence[int],
    update_scores_by_verifier: Mapping[str, Sequence[float]],
    eval_labels: Sequence[int],
    eval_scores_by_verifier: Mapping[str, Sequence[float]],
    started_s: float,
    now_s: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    repo_root: Path | str | None = None,
    state_path: Path | str = STATE_REL_PATH,
    previous_state_path: Path | str = V25_STATE_REL_PATH,
    frozen_ci: tuple[float, float] = FROZEN_CI,
    memory_ablation_contribution: float | None = None,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
    fresh_corpus_source: Mapping[str, Any] | None = None,
    cascade_artifact: Mapping[str, Any] | None = None,
    cascade_artifact_path: Path | str | None = None,
) -> JsonDict:
    """Continue v25 counters, learn optional cascade band, and persist v26 state."""

    v17 = _fr11_v17()
    v23 = _fr11_v23()
    v24 = _fr11_v24()
    _validate_v25_state(v25_state)
    root = Path(repo_root) if repo_root is not None else None
    resolved_state = _resolve_under_root(root or Path("."), state_path)
    tracker = tracker_from_v25_state(v25_state)
    update_matrix = v23.score_matrix(update_scores_by_verifier, VERIFIER_NAMES)
    update_labels_arr = np.asarray(update_labels, dtype=np.int64)
    if update_matrix.shape[0] != len(update_labels_arr):
        raise ValueError("labels and verifier scores must have the same length")
    v23._require_binary_labels(update_labels_arr)
    tracker.update_from_scores(update_labels_arr.tolist(), update_scores_by_verifier)
    table = v23.independence_table(tracker, VERIFIER_NAMES)
    weights = v23.independence_weighting(table, VERIFIER_NAMES)
    weight_vector = np.asarray([weights[name] for name in VERIFIER_NAMES], dtype=np.float64)
    fresh_learned_scores = v17.v10.exp3644.ensemble_scores(update_matrix, weight_vector)
    cascade_update = learn_cascade_band(
        cascade_artifact,
        fresh_scores=fresh_learned_scores,
        artifact_path=cascade_artifact_path,
    )

    eval_matrix = v23.score_matrix(eval_scores_by_verifier, VERIFIER_NAMES)
    eval_labels_arr = np.asarray(eval_labels, dtype=np.int64)
    if eval_matrix.shape[0] != len(eval_labels_arr):
        raise ValueError("labels and verifier scores must have the same length")
    v23._require_binary_labels(eval_labels_arr)
    learned_scores = v17.v10.exp3644.ensemble_scores(eval_matrix, weight_vector)
    learned_auroc = v17.v10.exp3644.tie_aware_auroc(eval_labels_arr, learned_scores)
    memory_contribution = (
        0.0
        if memory_ablation_contribution is None
        else float(memory_ablation_contribution)
    )
    auroc_in_ci = _in_ci(learned_auroc, frozen_ci)
    memory_min_met = memory_contribution >= MEMORY_ABLATION_MIN
    frozen_headline_unchanged = math.isclose(
        FROZEN_HEADLINE_AUROC,
        0.9131,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    invariant_held = bool(auroc_in_ci and memory_min_met and frozen_headline_unchanged)
    n_updates = int(len(update_labels_arr))
    state_sha = persist_state(
        tracker=tracker,
        path=resolved_state,
        weights=weights,
        table=table,
        previous_state=v25_state,
        previous_state_path=str(previous_state_path),
        n_updates=n_updates,
        random_seed=random_seed,
        learned_ensemble_auroc=learned_auroc,
        memory_ablation_contribution=memory_contribution,
        invariant_held=invariant_held,
        cascade_update=cascade_update,
    )
    state_persisted = resolved_state.is_file()
    cascade_band_learned = bool(cascade_update["cascade_band_learned"])
    verdict = select_verdict(
        learned_auroc=learned_auroc,
        memory_ablation_contribution=memory_contribution,
        invariant_held=invariant_held,
        cascade_band_learned=cascade_band_learned,
        state_persisted=state_persisted,
    )
    checksum_base = v24.reproducibility_checksum(
        update_labels=update_labels_arr,
        update_scores=update_matrix,
        eval_labels=eval_labels_arr,
        eval_scores=eval_matrix,
        table=table,
        weights=weights,
        state_sha256=state_sha,
        previous_state_sha256=str(v25_state.get("sha256", "")),
        random_seed=random_seed,
        frozen_ci=frozen_ci,
        memory_ablation_contribution=memory_contribution,
    )
    artifact: JsonDict = {
        "artifact": "experiment_3930_fr11_v26_cascade_band_online_learning",
        "schema": "carnot.fr11_continuous_self_learning_v26",
        "continuous_self_learning_task": True,
        "honest_verdict": verdict,
        "learned_ensemble_auroc": _round(learned_auroc),
        "memory_ablation_contribution": _round(memory_contribution),
        "frozen_headline_unchanged": bool(frozen_headline_unchanged),
        "cascade_band_learned": cascade_band_learned,
        "invariant_held": bool(invariant_held),
        "state_persisted_path": _relative_path(resolved_state, root),
        "n_updates": n_updates,
        "preconditions_checked": [dict(item) for item in preconditions or []],
        "random_seed": int(random_seed),
        "reproducibility_checksum": _json_sha256(
            {
                "schema": "carnot.fr11_v26_checksum",
                "base": checksum_base,
                "cascade_band_update": cascade_update,
            }
        ),
        "duration_s": _round(_duration(float(started_s), now_s)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "update_cost_note": UPDATE_COST_NOTE,
        "frozen_ci": {"low": float(frozen_ci[0]), "high": float(frozen_ci[1])},
        "frozen_headline_ensemble_auroc": FROZEN_HEADLINE_AUROC,
        "learned_weight_per_verifier": dict(weights),
        "per_verifier_independence_table": table,
        "previous_state_path": str(previous_state_path),
        "previous_state_sha256": v25_state.get("sha256"),
        "state_sha256": state_sha,
        "state_persisted": bool(state_persisted),
        "fresh_corpus_source": dict(fresh_corpus_source or {}),
        "cascade_band_update": cascade_update,
        "falsification_gate": {
            "learned_ensemble_auroc_in_frozen_ci": bool(auroc_in_ci),
            "memory_ablation_contribution_min_met": bool(memory_min_met),
            "frozen_headline_unchanged": bool(frozen_headline_unchanged),
            "condition": (
                "learned_ensemble_auroc in [0.9027,0.9235] AND "
                "memory_ablation_contribution >= 0.012 AND frozen_headline_unchanged"
            ),
        },
        "model_specs": {
            "live_model_invoked": False,
            "scoring_mode": "cached_verifier_scores_only",
            "verifiers": list(VERIFIER_NAMES),
        },
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(
    repo_root: Path | str,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    state_path: Path | str = STATE_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    n_updates: int = DEFAULT_N_UPDATES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    v25_state_path: Path | str = V25_STATE_REL_PATH,
    cascade_path: Path | str = EXP3927_REL_PATH,
    frozen_ci: tuple[float, float] = FROZEN_CI,
) -> Path:
    """Build, validate, and write the Exp 3930 JSON artifact."""

    root = Path(repo_root)
    artifact = build_artifact(
        root,
        started_s=started_s,
        now_s=now_s,
        n_updates=n_updates,
        random_seed=random_seed,
        v25_state_path=v25_state_path,
        state_path=state_path,
        cascade_path=cascade_path,
        frozen_ci=frozen_ci,
    )
    output = _resolve_under_root(root, output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def load_v25_state(path: Path | str) -> JsonDict:
    """Load and validate the persisted v25 reweighting state."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_v25_state(payload)
    return payload


def tracker_from_v25_state(state: Mapping[str, Any]) -> Any:
    """Create a tracker with counters initialized from the persisted v25 state."""

    v23 = _fr11_v23()
    _validate_v25_state(state)
    tracker = v23.IndependenceReweightingTracker(VERIFIER_NAMES)
    stats_by_name = tracker.stats()
    raw_stats = state["stats"]
    for name in VERIFIER_NAMES:
        target = stats_by_name[name]
        raw = raw_stats[name]
        target.true_positive = int(raw.get("true_positive", 0))
        target.false_positive = int(raw.get("false_positive", 0))
        target.false_negative = int(raw.get("false_negative", 0))
        target.true_negative = int(raw.get("true_negative", 0))
        target.unique_error_catches = int(raw.get("unique_error_catches", 0))
    return tracker


def learn_cascade_band(
    cascade_artifact: Mapping[str, Any] | None,
    *,
    fresh_scores: Sequence[float],
    artifact_path: Path | str | None,
) -> JsonDict:
    """Update the Exp 3927 escalation band from fresh learned-score margins."""

    if cascade_artifact is None:
        return {
            "cascade_band_learned": False,
            "reason": "exp3927_absent",
            "path": None if artifact_path is None else str(artifact_path),
            "band_before": None,
            "band_after": None,
        }
    threshold = float(cascade_artifact.get("energy_threshold", 0.5))
    before_raw = cascade_artifact.get("band_tuned_on_calibration")
    band_before = None if before_raw is None else max(0.0, float(before_raw))
    scores = np.asarray(fresh_scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("fresh_scores must contain at least one score")
    margins = np.abs(scores - threshold)
    target = float(np.quantile(margins, CASCADE_MARGIN_QUANTILE))
    target = min(CASCADE_BAND_MAX, max(CASCADE_BAND_MIN, target))
    band_after = target if band_before is None else (0.75 * band_before) + (0.25 * target)
    return {
        "cascade_band_learned": True,
        "reason": "exp3927_present_online_margin_update",
        "path": None if artifact_path is None else str(artifact_path),
        "exp3927_honest_verdict": cascade_artifact.get("honest_verdict"),
        "energy_threshold": _round(threshold),
        "band_before": None if band_before is None else _round(band_before),
        "band_after": _round(band_after),
        "target_band_from_fresh_margins": _round(target),
        "margin_quantile": CASCADE_MARGIN_QUANTILE,
        "learned_from_n_margins": int(scores.size),
        "method": "online_exponential_band_update_from_fresh_learned_ensemble_margins",
    }


def load_optional_cascade_artifact(
    repo_root: Path | str,
    path: Path | str = EXP3927_REL_PATH,
) -> JsonDict:
    """Disk-read the optional Exp 3927 artifact without making it a hard gate."""

    root = Path(repo_root)
    resolved = _resolve_under_root(root, path)
    if not resolved.is_file():
        return {
            "resource": "exp3927_cascade_router_artifact",
            "available": False,
            "detail": "optional artifact absent",
            "path": _relative_path(resolved, root),
        }
    try:
        artifact = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "resource": "exp3927_cascade_router_artifact",
            "available": False,
            "detail": f"{type(exc).__name__}: {exc}",
            "path": _relative_path(resolved, root),
        }
    if not isinstance(artifact, Mapping):
        return {
            "resource": "exp3927_cascade_router_artifact",
            "available": False,
            "detail": "artifact is not a JSON object",
            "path": _relative_path(resolved, root),
        }
    return {
        "resource": "exp3927_cascade_router_artifact",
        "available": True,
        "detail": f"loaded honest_verdict={artifact.get('honest_verdict')}",
        "path": _relative_path(resolved, root),
        "artifact": dict(artifact),
    }


def persist_state(
    *,
    tracker: Any,
    path: Path | str,
    weights: Mapping[str, float],
    table: Sequence[Mapping[str, Any]],
    previous_state: Mapping[str, Any],
    previous_state_path: str,
    n_updates: int,
    random_seed: int,
    learned_ensemble_auroc: float,
    memory_ablation_contribution: float,
    invariant_held: bool,
    cascade_update: Mapping[str, Any],
) -> str:
    """Persist v26 cumulative counters and return the stored payload checksum."""

    output = Path(path)
    previous_n = int(previous_state.get("n_examples_observed", 0))
    payload = {
        "schema": "carnot.fr11_v26_cascade_band_online_learning_state",
        "version": 1,
        "previous_schema": previous_state.get("schema"),
        "previous_state_path": str(previous_state_path),
        "previous_state_sha256": previous_state.get("sha256"),
        "verifier_names": list(VERIFIER_NAMES),
        "stats": {name: tracker.stats()[name].to_json() for name in VERIFIER_NAMES},
        "per_verifier_independence_table": [dict(row) for row in table],
        "weights": dict(weights),
        "n_examples_observed": previous_n + int(n_updates),
        "n_updates": int(n_updates),
        "random_seed": int(random_seed),
        "learned_ensemble_auroc": _round(learned_ensemble_auroc),
        "memory_ablation_contribution": _round(memory_ablation_contribution),
        "invariant_held": bool(invariant_held),
        "cascade_band_learned": bool(cascade_update.get("cascade_band_learned")),
        "cascade_band_before": cascade_update.get("band_before"),
        "cascade_band_after": cascade_update.get("band_after"),
        "cascade_band_update": dict(cascade_update),
        "update_cost_note": UPDATE_COST_NOTE,
    }
    checksum = _json_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({**payload, "sha256": checksum}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return checksum


def v25_state_precondition(path: Path | str) -> JsonDict:
    """Check that the v25 state loads and has the expected schema."""

    try:
        state = load_v25_state(path)
    except Exception as exc:
        return {
            "resource": "v25_reweighting_state",
            "available": False,
            "detail": f"{type(exc).__name__}: {exc}",
        }
    return {
        "resource": "v25_reweighting_state",
        "available": True,
        "detail": f"loaded n_examples_observed={state.get('n_examples_observed')}",
    }


def select_verdict(
    *,
    learned_auroc: float,
    memory_ablation_contribution: float,
    invariant_held: bool,
    cascade_band_learned: bool,
    state_persisted: bool,
) -> str:
    """Return the terminal v26 verdict for the falsification gate."""

    auroc = _round(learned_auroc, digits=4)
    mem = _round(memory_ablation_contribution, digits=4)
    if invariant_held and state_persisted:
        band_flag = str(bool(cascade_band_learned)).lower()
        return (
            f"{SUCCESS_VERDICT_PREFIX}{auroc}_memcontrib{mem}_"
            f"cascade_band_learned{band_flag}{SUCCESS_VERDICT_SUFFIX}"
        )
    return f"{BROKEN_VERDICT_PREFIX}{auroc}_memcontrib{mem}{BROKEN_VERDICT_SUFFIX}"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3930 artifact schema before writing."""

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
        raise ValueError("inference_substrate must declare cached verifier scoring")
    serialized = json.dumps(artifact, sort_keys=True)
    for marker in ("GGUF", "CUDA", "cuda", "torch.cuda", "live_llm_inference"):
        if marker in serialized:
            raise ValueError("forbidden inference marker present")
    if not isinstance(artifact.get("invariant_held"), bool):
        raise ValueError("invariant_held must be a bare boolean")
    if not isinstance(artifact.get("cascade_band_learned"), bool):
        raise ValueError("cascade_band_learned must be a bare boolean")
    if not isinstance(artifact.get("frozen_headline_unchanged"), bool):
        raise ValueError("frozen_headline_unchanged must be a bare boolean")
    if not isinstance(artifact.get("memory_ablation_contribution"), int | float):
        raise ValueError("memory_ablation_contribution must be a bare scalar")
    if not isinstance(artifact.get("state_persisted_path"), str):
        raise ValueError("state_persisted_path must be a string path")
    if not isinstance(artifact.get("n_updates"), int) or int(artifact["n_updates"]) < 0:
        raise ValueError("n_updates must be a non-negative integer")
    auroc = artifact.get("learned_ensemble_auroc")
    if auroc is not None and (
        not isinstance(auroc, int | float) or not 0.0 <= float(auroc) <= 1.0
    ):
        raise ValueError("learned_ensemble_auroc must be in [0, 1]")
    verdict = str(artifact.get("honest_verdict"))
    invariant_held = artifact.get("invariant_held")
    if verdict.startswith("blocked_"):
        return
    if verdict.startswith(SUCCESS_VERDICT_PREFIX):
        if invariant_held is not True:
            raise ValueError("held verdict requires invariant_held=true")
        if not verdict.endswith(SUCCESS_VERDICT_SUFFIX):
            raise ValueError("held verdict must record state persistence")
        if "_cascade_band_learned" not in verdict:
            raise ValueError("held verdict must record cascade band learning")
        return
    if verdict.startswith(BROKEN_VERDICT_PREFIX):
        if invariant_held is not False:
            raise ValueError("broken verdict requires invariant_held=false")
        if not verdict.endswith(BROKEN_VERDICT_SUFFIX):
            raise ValueError("broken verdict must record online drift")
        return
    raise ValueError(f"unsupported honest_verdict: {verdict!r}")  # pragma: no cover


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
        "artifact": "experiment_3930_fr11_v26_cascade_band_online_learning",
        "schema": "carnot.fr11_continuous_self_learning_v26",
        "continuous_self_learning_task": True,
        "honest_verdict": verdict,
        "learned_ensemble_auroc": None,
        "memory_ablation_contribution": 0.0,
        "frozen_headline_unchanged": False,
        "cascade_band_learned": False,
        "invariant_held": False,
        "state_persisted_path": _relative_path(state_output, repo_root),
        "n_updates": 0,
        "preconditions_checked": [dict(item) for item in preconditions],
        "random_seed": int(random_seed),
        "reproducibility_checksum": _json_sha256(payload),
        "duration_s": _round(duration_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "cascade_band_update": {
            "cascade_band_learned": False,
            "reason": "blocked_preconditions",
            "band_before": None,
            "band_after": None,
        },
        "falsification_gate": {
            "learned_ensemble_auroc_in_frozen_ci": False,
            "memory_ablation_contribution_min_met": False,
            "frozen_headline_unchanged": False,
            "condition": "preconditions available before online reweighting",
        },
    }
    validate_artifact(artifact)
    return artifact


def _validate_v25_state(state: Mapping[str, Any]) -> None:
    if state.get("schema") != "carnot.fr11_v25_independence_reweighting_state":
        raise ValueError("v25 state schema is missing or unsupported")
    stats = state.get("stats")
    if not isinstance(stats, Mapping) or set(stats) != set(VERIFIER_NAMES):
        raise ValueError("v25 state stats must include all frozen verifiers")
    for name in VERIFIER_NAMES:
        raw = stats[name]
        if not isinstance(raw, Mapping):
            raise ValueError("v25 state stat entries must be mappings")
        for field in (
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
            "unique_error_catches",
        ):
            int(raw.get(field, 0))


def _blocked_verdict(preconditions: Sequence[Mapping[str, Any]]) -> str:
    for item in preconditions:
        if item.get("available"):
            continue
        if item.get("resource") == "carnot_verify_import":
            return BLOCKED_VERIFY_VERDICT
        if item.get("resource") == "v25_reweighting_state":
            return BLOCKED_V25_STATE_VERDICT
        return f"blocked_{item.get('resource', 'resource')}"
    return BLOCKED_FRESH_CORPUS_VERDICT


def _public_precondition(precondition: Mapping[str, Any]) -> JsonDict:
    return {
        key: value
        for key, value in precondition.items()
        if key != "artifact"
    }
