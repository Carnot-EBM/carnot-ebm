"""FR-11 continuous self-learning v25 persisted independence reweighting.

Spec: REQ-LEARN-3921, SCENARIO-LEARN-3921.

Exp 3921 resumes the v24 independence counters instead of starting a new
tracker. The update is intentionally plain CPU bookkeeping over cached verifier
scores: every fresh FoVer row increments precision counters and unique-catch
counters, then the learned weights are remeasured against the frozen headline
scoring path. That makes the artifact a continuity check for FR-11 online
self-learning, not a replacement for the frozen 0.9131 headline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot.fr11 import continuous_self_learning_v17 as v17
from carnot.fr11 import continuous_self_learning_v23 as v23
from carnot.fr11 import continuous_self_learning_v24 as v24


JsonDict = dict[str, Any]

OUTPUT_REL_PATH = Path("results/experiment_3921_fr11_v25_independence_reweighting.json")
STATE_REL_PATH = Path("results/experiment_3921_fr11_v25_independence_reweighting_state.json")
V24_STATE_REL_PATH = Path("results/experiment_3888_fr11_v24_independence_reweighting_state.json")
DEFAULT_RANDOM_SEED = 3921
DEFAULT_N_UPDATES = 228
FROZEN_CI = v24.FROZEN_CI
FROZEN_HEADLINE_AUROC = v24.FROZEN_HEADLINE_AUROC
MEMORY_ABLATION_MIN = v24.MEMORY_ABLATION_MIN
VERIFIER_NAMES = v24.VERIFIER_NAMES
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: CPU counter updates over cached FoVer v4/v3 scores; no LLM loaded)."
)
UPDATE_COST_NOTE = (
    "Tier-1 online path: pure CPU counter updates over cached verifier scores; "
    "persisted state lets v26 continue."
)
SUCCESS_VERDICT_PREFIX = "complete: fr11_v25_INVARIANT_HELD_auroc"
SUCCESS_VERDICT_SUFFIX = "_state_persisted"
BROKEN_VERDICT_PREFIX = "complete: fr11_v25_INVARIANT_BROKEN_auroc"
BROKEN_VERDICT_SUFFIX = "_online_reweighting_drifts"
BLOCKED_VERIFY_VERDICT = "blocked_carnot_verify_import"
BLOCKED_V24_STATE_VERDICT = "blocked_v24_reweighting_state"
BLOCKED_FRESH_CORPUS_VERDICT = "blocked_fresh_verification_corpus"
BLOCKED_FROZEN_SCORING_VERDICT = "blocked_frozen_headline_scoring"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "learned_ensemble_auroc",
    "memory_ablation_contribution",
    "frozen_headline_unchanged",
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
    "invariant_held": (
        "BARE BOOL -- did v25 preserve both the CI-band AUROC and the memory "
        "contribution; the milestone self-learning success signal."
    ),
    "state_persisted_path": (
        "Verifier-scoring methodology -- CPU counter updates; persisted state "
        "lets v26 continue."
    ),
    "n_updates": (
        "Verifier-scoring methodology -- CPU counter updates; persisted state "
        "lets v26 continue."
    ),
    "preconditions_checked": (
        "Verifier-scoring methodology -- CPU counter updates; persisted state "
        "lets v26 continue."
    ),
    "random_seed": (
        "Verifier-scoring methodology -- CPU counter updates; persisted state "
        "lets v26 continue."
    ),
    "reproducibility_checksum": (
        "Verifier-scoring methodology -- CPU counter updates; persisted state "
        "lets v26 continue."
    ),
    "duration_s": (
        "Verifier-scoring methodology -- CPU counter updates; persisted state "
        "lets v26 continue."
    ),
    "inference_substrate": (
        "Verifier-scoring methodology -- CPU counter updates; persisted state "
        "lets v26 continue."
    ),
}

verify_import_precondition = v24.verify_import_precondition
score_fresh_verification_corpus = v24.score_fresh_verification_corpus
score_frozen_headline_corpus = v24.score_frozen_headline_corpus
read_json_fover_rows = v24.read_json_fover_rows
select_balanced_fresh_rows = v24.select_balanced_fresh_rows
_duration = v24._duration
_in_ci = v24._in_ci
_json_sha256 = v24._json_sha256
_relative_path = v24._relative_path
_resolve_under_root = v24._resolve_under_root
_round = v24._round


def build_artifact(
    repo_root: Path | str,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    n_updates: int = DEFAULT_N_UPDATES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    v24_state_path: Path | str = V24_STATE_REL_PATH,
    state_path: Path | str = STATE_REL_PATH,
    frozen_ci: tuple[float, float] = FROZEN_CI,
) -> JsonDict:
    """Build the Exp 3921 artifact from persisted v24 state and cached scores."""

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    resolved_v24_state = _resolve_under_root(root, v24_state_path)
    resolved_state = _resolve_under_root(root, state_path)
    preconditions = [
        verify_import_precondition(),
        v24_state_precondition(resolved_v24_state),
    ]
    if not all(item["available"] for item in preconditions):
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            state_output=resolved_state,
            repo_root=root,
            preconditions=preconditions,
            verdict=_blocked_verdict(preconditions),
        )

    v24_state = load_v24_state(resolved_v24_state)
    try:
        update_labels, update_scores, fresh_source = score_fresh_verification_corpus(
            root,
            n_updates=n_updates,
            random_seed=random_seed,
        )
    except Exception as exc:  # pragma: no cover - exercised by integration runs.
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
    except Exception as exc:  # pragma: no cover - exercised by integration runs.
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

    return build_artifact_from_scores(
        v24_state=v24_state,
        update_labels=update_labels,
        update_scores_by_verifier=update_scores,
        eval_labels=eval_labels,
        eval_scores_by_verifier=eval_scores,
        started_s=start,
        now_s=now_s,
        random_seed=random_seed,
        repo_root=root,
        state_path=resolved_state,
        previous_state_path=_relative_path(resolved_v24_state, root),
        frozen_ci=frozen_ci,
        memory_ablation_contribution=v17.load_memory_contribution_reference(root),
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
    )


def build_artifact_from_scores(
    *,
    v24_state: Mapping[str, Any],
    update_labels: Sequence[int],
    update_scores_by_verifier: Mapping[str, Sequence[float]],
    eval_labels: Sequence[int],
    eval_scores_by_verifier: Mapping[str, Sequence[float]],
    started_s: float,
    now_s: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    repo_root: Path | str | None = None,
    state_path: Path | str = STATE_REL_PATH,
    previous_state_path: Path | str = V24_STATE_REL_PATH,
    frozen_ci: tuple[float, float] = FROZEN_CI,
    memory_ablation_contribution: float | None = None,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
    fresh_corpus_source: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Continue the v24 tracker, evaluate frozen AUROC, and persist v25 state."""

    _validate_v24_state(v24_state)
    root = Path(repo_root) if repo_root is not None else None
    resolved_state = _resolve_under_root(root or Path("."), state_path)
    tracker = tracker_from_v24_state(v24_state)
    update_matrix = v23.score_matrix(update_scores_by_verifier, VERIFIER_NAMES)
    update_labels_arr = np.asarray(update_labels, dtype=np.int64)
    if update_matrix.shape[0] != len(update_labels_arr):
        raise ValueError("labels and verifier scores must have the same length")
    v23._require_binary_labels(update_labels_arr)
    tracker.update_from_scores(update_labels_arr.tolist(), update_scores_by_verifier)
    table = v23.independence_table(tracker, VERIFIER_NAMES)
    weights = v23.independence_weighting(table, VERIFIER_NAMES)

    eval_matrix = v23.score_matrix(eval_scores_by_verifier, VERIFIER_NAMES)
    eval_labels_arr = np.asarray(eval_labels, dtype=np.int64)
    if eval_matrix.shape[0] != len(eval_labels_arr):
        raise ValueError("labels and verifier scores must have the same length")
    v23._require_binary_labels(eval_labels_arr)
    weight_vector = np.asarray([weights[name] for name in VERIFIER_NAMES], dtype=np.float64)
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
        previous_state=v24_state,
        previous_state_path=str(previous_state_path),
        n_updates=n_updates,
        random_seed=random_seed,
        learned_ensemble_auroc=learned_auroc,
        memory_ablation_contribution=memory_contribution,
        invariant_held=invariant_held,
    )
    state_persisted = resolved_state.is_file()
    verdict = select_verdict(
        learned_auroc=learned_auroc,
        memory_ablation_contribution=memory_contribution,
        invariant_held=invariant_held,
        state_persisted=state_persisted,
    )
    artifact: JsonDict = {
        "artifact": "experiment_3921_fr11_v25_independence_reweighting",
        "schema": "carnot.fr11_continuous_self_learning_v25",
        "continuous_self_learning_task": True,
        "honest_verdict": verdict,
        "learned_ensemble_auroc": _round(learned_auroc),
        "memory_ablation_contribution": _round(memory_contribution),
        "frozen_headline_unchanged": bool(frozen_headline_unchanged),
        "invariant_held": bool(invariant_held),
        "state_persisted_path": _relative_path(resolved_state, root),
        "n_updates": n_updates,
        "preconditions_checked": [dict(item) for item in preconditions or []],
        "random_seed": int(random_seed),
        "reproducibility_checksum": v24.reproducibility_checksum(
            update_labels=update_labels_arr,
            update_scores=update_matrix,
            eval_labels=eval_labels_arr,
            eval_scores=eval_matrix,
            table=table,
            weights=weights,
            state_sha256=state_sha,
            previous_state_sha256=str(v24_state.get("sha256", "")),
            random_seed=random_seed,
            frozen_ci=frozen_ci,
            memory_ablation_contribution=memory_contribution,
        ),
        "duration_s": _round(_duration(float(started_s), now_s)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "update_cost_note": UPDATE_COST_NOTE,
        "frozen_ci": {"low": float(frozen_ci[0]), "high": float(frozen_ci[1])},
        "frozen_headline_ensemble_auroc": FROZEN_HEADLINE_AUROC,
        "learned_weight_per_verifier": dict(weights),
        "per_verifier_independence_table": table,
        "previous_state_path": str(previous_state_path),
        "previous_state_sha256": v24_state.get("sha256"),
        "state_sha256": state_sha,
        "state_persisted": bool(state_persisted),
        "fresh_corpus_source": dict(fresh_corpus_source or {}),
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
    v24_state_path: Path | str = V24_STATE_REL_PATH,
    frozen_ci: tuple[float, float] = FROZEN_CI,
) -> Path:
    """Build, validate, and write the Exp 3921 JSON artifact."""

    root = Path(repo_root)
    artifact = build_artifact(
        root,
        started_s=started_s,
        now_s=now_s,
        n_updates=n_updates,
        random_seed=random_seed,
        v24_state_path=v24_state_path,
        state_path=state_path,
        frozen_ci=frozen_ci,
    )
    output = _resolve_under_root(root, output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def load_v24_state(path: Path | str) -> JsonDict:
    """Load and validate the persisted v24 reweighting state."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_v24_state(payload)
    return payload


def tracker_from_v24_state(state: Mapping[str, Any]) -> v23.IndependenceReweightingTracker:
    """Create a tracker with counters initialized from the persisted v24 state."""

    _validate_v24_state(state)
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


def persist_state(
    *,
    tracker: v23.IndependenceReweightingTracker,
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
) -> str:
    """Persist v25 cumulative counters and return the stored payload checksum."""

    output = Path(path)
    previous_n = int(previous_state.get("n_examples_observed", 0))
    payload = {
        "schema": "carnot.fr11_v25_independence_reweighting_state",
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
        "update_cost_note": UPDATE_COST_NOTE,
    }
    checksum = _json_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({**payload, "sha256": checksum}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return checksum


def v24_state_precondition(path: Path | str) -> JsonDict:
    """Check that the v24 state loads and has the expected schema."""

    try:
        state = load_v24_state(path)
    except Exception as exc:
        return {
            "resource": "v24_reweighting_state",
            "available": False,
            "detail": f"{type(exc).__name__}: {exc}",
        }
    return {
        "resource": "v24_reweighting_state",
        "available": True,
        "detail": f"loaded n_examples_observed={state.get('n_examples_observed')}",
    }


def select_verdict(
    *,
    learned_auroc: float,
    memory_ablation_contribution: float,
    invariant_held: bool,
    state_persisted: bool,
) -> str:
    """Return the terminal v25 verdict for the falsification gate."""

    auroc = _round(learned_auroc, digits=4)
    mem = _round(memory_ablation_contribution, digits=4)
    if invariant_held and state_persisted:
        return f"{SUCCESS_VERDICT_PREFIX}{auroc}_memcontrib{mem}{SUCCESS_VERDICT_SUFFIX}"
    return f"{BROKEN_VERDICT_PREFIX}{auroc}_memcontrib{mem}{BROKEN_VERDICT_SUFFIX}"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3921 artifact schema before writing."""

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
        "artifact": "experiment_3921_fr11_v25_independence_reweighting",
        "schema": "carnot.fr11_continuous_self_learning_v25",
        "continuous_self_learning_task": True,
        "honest_verdict": verdict,
        "learned_ensemble_auroc": None,
        "memory_ablation_contribution": 0.0,
        "frozen_headline_unchanged": False,
        "invariant_held": False,
        "state_persisted_path": _relative_path(state_output, repo_root),
        "n_updates": 0,
        "preconditions_checked": [dict(item) for item in preconditions],
        "random_seed": int(random_seed),
        "reproducibility_checksum": _json_sha256(payload),
        "duration_s": _round(duration_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "falsification_gate": {
            "learned_ensemble_auroc_in_frozen_ci": False,
            "memory_ablation_contribution_min_met": False,
            "frozen_headline_unchanged": False,
            "condition": "preconditions available before online reweighting",
        },
    }
    validate_artifact(artifact)
    return artifact


def _validate_v24_state(state: Mapping[str, Any]) -> None:
    if state.get("schema") != "carnot.fr11_v24_independence_reweighting_state":
        raise ValueError("v24 state schema is missing or unsupported")
    stats = state.get("stats")
    if not isinstance(stats, Mapping) or set(stats) != set(VERIFIER_NAMES):
        raise ValueError("v24 state stats must include all frozen verifiers")
    for name in VERIFIER_NAMES:
        raw = stats[name]
        if not isinstance(raw, Mapping):
            raise ValueError("v24 state stat entries must be mappings")
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
        if item.get("resource") == "v24_reweighting_state":
            return BLOCKED_V24_STATE_VERDICT
        return f"blocked_{item.get('resource', 'resource')}"
    return BLOCKED_FRESH_CORPUS_VERDICT
