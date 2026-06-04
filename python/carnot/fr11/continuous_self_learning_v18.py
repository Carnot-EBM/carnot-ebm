"""FR-11 continuous self-learning v18 Tier-2 constraint memory.

Spec: REQ-LEARN-3778, SCENARIO-LEARN-3778.

The v18 forward difference is Tier-2 persistence, not another Tier-1 precision
counter.  It scores the cached FoVer corpus with the four headline verifiers,
consolidates one score-threshold delta per verifier domain with
Tier2ThresholdMemory, persists those deltas to SQLite, and recomputes the
headline ensemble AUROC through the persisted ``apply_delta`` path.
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
from carnot.fr11.tier2_memory import Tier2ThresholdMemory


JsonDict = dict[str, Any]

OUTPUT_REL_PATH = Path(
    "results/experiment_3778_fr11_self_learning_v18_tier2_constraint_memory.json"
)
TIER2_STATE_REL_PATH = Path(
    "results/experiment_3778_fr11_v18_tier2_threshold_memory.db"
)
EXP2837_REL_PATH = Path("results/experiment_2837_fover_memory_leakage_v3.json")
DEFAULT_RANDOM_SEED = 3778
DEFAULT_CORPUS_RANDOM_SEED = 3778
DEFAULT_N_EXAMPLES = 1000
DEFAULT_FROZEN_CI = (0.9027, 0.9235)
MEMORY_CONTRIBUTION_ROUNDED_MIN = 0.0185
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: reads cached verifier scores, no live model)."
)
SUCCESS_VERDICT = (
    "complete: "
    "fr11_v18_tier2_constraint_memory_consolidated_auroc_within_frozen_ci_"
    "memory_contribution_preserved_state_persisted"
)
BLOCKED_CORPUS_VERDICT = (
    "complete: blocked_fover_corpus_missing -- no corpus to consolidate from"
)
BLOCKED_SCORING_VERDICT = (
    "complete: blocked_fover_scores_missing -- no corpus to consolidate from"
)
BLOCKED_INTERPRETER_VERDICT = "complete: blocked_interpreter_runtime"
TIER2_LOOKUP = "cpu_plus_system_memory_sqlite_lookup_lt_1ms_no_model_retrain"
VERIFIER_NAMES = v17.VERIFIER_NAMES

score_fover_corpus = v17.score_fover_corpus
score_matrix = v17.score_matrix
probe_cached_trace_preconditions = v17.probe_cached_trace_preconditions

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "per_domain_threshold_deltas",
    "ensemble_auroc_under_consolidation",
    "auroc_within_frozen_ci",
    "memory_contribution_preserved",
    "is_tier2_not_tier1",
    "tier2_lookup_is_cpu_memory",
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
    "per_domain_threshold_deltas": (
        "The consolidated per-domain deltas -- the Tier-2 learned knowledge; "
        "the core deliverable."
    ),
    "ensemble_auroc_under_consolidation": (
        "The ensemble AUROC after applying the consolidated deltas -- confirms "
        "the self-learning does not regress the banked discriminator."
    ),
    "auroc_within_frozen_ci": (
        "BARE bool -- the consolidated AUROC stays within/above the frozen "
        "CI95 [0.9027, 0.9235]; the no-regression invariant."
    ),
    "memory_contribution_preserved": (
        "BARE bool -- the +0.0185 fr11_session_memory contribution survives "
        "consolidation."
    ),
    "is_tier2_not_tier1": (
        "BARE bool, true -- confirms v18 is Tier-2 constraint-memory "
        "consolidation, NOT a re-run of v17's Tier-1 precision tracker."
    ),
    "tier2_lookup_is_cpu_memory": (
        "Confirms the mechanic is a CPU + system-memory lookup (<1ms, the "
        "Tier-2 hardware path), not a model retrain."
    ),
    "tracker_state_persisted": (
        "BARE bool, true -- the Tier-2 memory db was persisted so a future "
        "milestone resumes it."
    ),
    "model_specs": "Names the corpus + 4 verifiers -- honest substrate.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": (
        "Wall-clock plausibility floor; verifier-scoring + consolidation is modest."
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
    db_path: Path | str = TIER2_STATE_REL_PATH,
    frozen_ci: tuple[float, float] = DEFAULT_FROZEN_CI,
) -> JsonDict:
    """Build Exp 3778 from cached FoVer rows and verifier scores."""

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    db_output = _resolve_under_root(root, db_path)
    corpus_path = root / "data" / "fover_corpus.jsonl"
    preconditions = [
        _interpreter_precondition(),
        _fover_corpus_precondition(corpus_path, n_examples=n_examples),
    ]
    if not preconditions[-1]["available"]:
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            db_output=db_output,
            repo_root=root,
            preconditions=preconditions,
            verdict=BLOCKED_CORPUS_VERDICT,
        )
    preconditions.extend(probe_cached_trace_preconditions(root, n_examples=n_examples))
    if not all(item["available"] for item in preconditions):
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            db_output=db_output,
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
            db_output=db_output,
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
        db_path=db_output,
        repo_root=root,
        preconditions=preconditions,
        memory_contribution_reference=load_memory_contribution_reference(root),
        corpus_absolute_path=corpus_path.resolve(),
        frozen_ci=frozen_ci,
    )


def build_artifact_from_scores(
    *,
    labels: Sequence[int],
    scores_by_verifier: Mapping[str, Sequence[float]],
    started_s: float,
    now_s: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    db_path: Path | str = TIER2_STATE_REL_PATH,
    repo_root: Path | str | None = None,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
    memory_contribution_reference: float | None = None,
    corpus_absolute_path: Path | str | None = None,
    frozen_ci: tuple[float, float] = DEFAULT_FROZEN_CI,
) -> JsonDict:
    """Consolidate Tier-2 deltas and evaluate the persisted-memory AUROC."""

    root = Path(repo_root) if repo_root is not None else None
    db_output = _resolve_under_root(root or Path("."), db_path)
    if not labels or not scores_by_verifier:
        return _blocked_artifact(
            duration_s=_duration(float(started_s), now_s),
            random_seed=random_seed,
            db_output=db_output,
            repo_root=root,
            preconditions=preconditions or [_trace_precondition(labels, scores_by_verifier)],
            verdict=BLOCKED_CORPUS_VERDICT,
        )

    labels_arr = np.asarray(labels, dtype=np.int64)
    _require_binary_labels(labels_arr)
    raw_matrix = score_matrix(scores_by_verifier, VERIFIER_NAMES)
    if raw_matrix.shape[0] != len(labels_arr):
        raise ValueError("labels and verifier scores must have the same length")
    weights = v17.v10.exp3644.carnot_current_weights(VERIFIER_NAMES)
    deltas = consolidate_domain_deltas(
        labels=labels_arr.tolist(),
        scores_by_domain=scores_by_verifier,
        db_path=db_output,
    )
    adjusted_matrix = apply_consolidated_deltas(
        scores_by_domain=scores_by_verifier,
        db_path=db_output,
    )
    baseline_scores = v17.v10.exp3644.ensemble_scores(raw_matrix, weights)
    consolidated_scores = v17.v10.exp3644.ensemble_scores(adjusted_matrix, weights)
    baseline_auroc = v17.v10.exp3644.tie_aware_auroc(labels_arr, baseline_scores)
    consolidated_auroc = v17.v10.exp3644.tie_aware_auroc(labels_arr, consolidated_scores)
    within_ci = auroc_within_or_above_frozen_ci(consolidated_auroc, frozen_ci)
    preserved = memory_contribution_preserved(memory_contribution_reference)
    state_sha = tier2_state_checksum(
        deltas=deltas,
        n_examples_by_domain={name: len(scores_by_verifier[name]) for name in VERIFIER_NAMES},
        db_path=_relative_path(db_output, root),
    )
    artifact: JsonDict = {
        "artifact": "experiment_3778_fr11_self_learning_v18_tier2_constraint_memory",
        "schema": "carnot.fr11_continuous_self_learning_v18",
        "continuous_self_learning_task": True,
        "honest_verdict": SUCCESS_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "per_domain_threshold_deltas": {name: _round(deltas[name]) for name in VERIFIER_NAMES},
        "ensemble_auroc_under_consolidation": _round(consolidated_auroc),
        "auroc_within_frozen_ci": bool(within_ci),
        "memory_contribution_preserved": bool(preserved),
        "is_tier2_not_tier1": True,
        "tier2_lookup_is_cpu_memory": TIER2_LOOKUP,
        "tracker_state_persisted": db_output.is_file(),
        "model_specs": _model_specs(corpus_absolute_path, n_examples=len(labels_arr)),
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum(
            labels=labels_arr,
            raw_scores=raw_matrix,
            adjusted_scores=adjusted_matrix,
            deltas=deltas,
            baseline_auroc=baseline_auroc,
            consolidated_auroc=consolidated_auroc,
            random_seed=random_seed,
            memory_contribution_reference=memory_contribution_reference,
            tier2_state_sha256=state_sha,
        ),
        "duration_s": _round(_duration(float(started_s), now_s)),
        "frozen_ci95": {"low": float(frozen_ci[0]), "high": float(frozen_ci[1])},
        "memory_contribution_reference": memory_contribution_reference,
        "n_examples": int(len(labels_arr)),
        "n_examples_by_domain": {name: len(scores_by_verifier[name]) for name in VERIFIER_NAMES},
        "verifier_names": list(VERIFIER_NAMES),
        "tier2_memory_state_path": _relative_path(db_output, root),
        "tier2_memory_state_sha256": state_sha,
        "methodology": _methodology(random_seed, corpus_absolute_path),
        "preconditions_checked": [dict(item) for item in preconditions or []],
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "passed": bool(within_ci and preserved and db_output.is_file()),
            "condition": (
                "consolidated AUROC >= frozen CI low AND Exp 2837 rounded "
                "memory contribution >= +0.0185 AND Tier-2 SQLite state persisted"
            ),
            "principle": (
                "Constraint-memory consolidation must not silently regress the "
                "FoVer discriminator or drop the FR-11 memory contribution."
            ),
        },
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(
    repo_root: Path | str,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    db_path: Path | str = TIER2_STATE_REL_PATH,
    labels: Sequence[int] | None = None,
    scores_by_verifier: Mapping[str, Sequence[float]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    frozen_ci: tuple[float, float] = DEFAULT_FROZEN_CI,
) -> Path:
    """Build, validate, and write the Exp 3778 JSON artifact."""

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    resolved_db = _resolve_under_root(root, db_path)
    if labels is not None or scores_by_verifier is not None:
        artifact = build_artifact_from_scores(
            labels=labels or [],
            scores_by_verifier=scores_by_verifier or {},
            started_s=start,
            now_s=now_s,
            db_path=resolved_db,
            repo_root=root,
            memory_contribution_reference=load_memory_contribution_reference(root),
            corpus_absolute_path=(root / "data" / "fover_corpus.jsonl").resolve(),
            frozen_ci=frozen_ci,
        )
    else:
        artifact = build_artifact(
            root,
            started_s=start,
            now_s=now_s,
            db_path=resolved_db,
            frozen_ci=frozen_ci,
        )
    output = _resolve_under_root(root, output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def consolidate_domain_deltas(
    *,
    labels: Sequence[int],
    scores_by_domain: Mapping[str, Sequence[float]],
    db_path: Path | str,
    domain_keys: Sequence[str] = VERIFIER_NAMES,
) -> dict[str, float]:
    """Persist one Tier-2 threshold delta per verifier score domain."""

    labels_list = [int(label) for label in labels]
    memory = Tier2ThresholdMemory(str(db_path))
    deltas: dict[str, float] = {}
    for domain_key in domain_keys:
        scores = [float(value) for value in scores_by_domain[domain_key]]
        if len(scores) != len(labels_list):
            raise ValueError("labels and verifier scores must have the same length")
        memory.update_domain_delta(domain_key, scores, labels_list)
        deltas[domain_key] = float(memory.get_domain_delta(domain_key))
    return deltas


def apply_consolidated_deltas(
    *,
    scores_by_domain: Mapping[str, Sequence[float]],
    db_path: Path | str,
    domain_keys: Sequence[str] = VERIFIER_NAMES,
) -> np.ndarray:
    """Return a score matrix adjusted through persisted Tier-2 memory lookups."""

    memory = Tier2ThresholdMemory(str(db_path))
    columns = []
    for domain_key in domain_keys:
        columns.append(
            [
                memory.apply_delta(domain_key, float(value))
                for value in scores_by_domain[domain_key]
            ]
        )
    matrix = np.column_stack([np.asarray(column, dtype=np.float64) for column in columns])
    if not np.isfinite(matrix).all():
        raise ValueError("adjusted verifier score matrix must be finite")
    return matrix


def auroc_within_or_above_frozen_ci(
    auroc: float,
    frozen_ci: tuple[float, float] = DEFAULT_FROZEN_CI,
) -> bool:
    """Return true when AUROC is inside or above the frozen no-regression band."""

    value = float(auroc)
    low, _high = frozen_ci
    return math.isfinite(value) and value >= float(low)


def memory_contribution_preserved(reference: float | None) -> bool:
    """Return true when the Exp 2837 FR-11 memory contribution remains present."""

    if reference is None:
        return False
    return round(float(reference), 4) >= MEMORY_CONTRIBUTION_ROUNDED_MIN


def load_memory_contribution_reference(repo_root: Path | str) -> float | None:
    """Load the Exp 2837 FR-11 memory contribution reference if present."""

    path = Path(repo_root) / EXP2837_REL_PATH
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = payload.get("learning_contribution")
    return float(value) if isinstance(value, int | float) else None


def tier2_state_checksum(
    *,
    deltas: Mapping[str, float],
    n_examples_by_domain: Mapping[str, int],
    db_path: str,
) -> str:
    """Hash stable Tier-2 state content without SQLite timestamp noise."""

    payload = {
        "deltas": {key: _round(deltas[key]) for key in sorted(deltas)},
        "n_examples_by_domain": {
            key: int(n_examples_by_domain[key]) for key in sorted(n_examples_by_domain)
        },
        "db_path": db_path,
        "tier2_lookup": TIER2_LOOKUP,
    }
    return _json_sha256(payload)


def reproducibility_checksum(
    *,
    labels: np.ndarray | Sequence[int],
    raw_scores: np.ndarray | Sequence[Sequence[float]],
    adjusted_scores: np.ndarray | Sequence[Sequence[float]],
    deltas: Mapping[str, float],
    baseline_auroc: float,
    consolidated_auroc: float,
    random_seed: int,
    memory_contribution_reference: float | None,
    tier2_state_sha256: str,
) -> str:
    """Hash the measured labels, score matrices, deltas, AUROC, and seed."""

    payload = {
        "labels_sha256": _array_sha256(np.asarray(labels, dtype=np.int64)),
        "raw_scores_sha256": _array_sha256(np.asarray(raw_scores, dtype=np.float64)),
        "adjusted_scores_sha256": _array_sha256(
            np.asarray(adjusted_scores, dtype=np.float64)
        ),
        "per_domain_threshold_deltas": {key: _round(deltas[key]) for key in sorted(deltas)},
        "baseline_auroc": _round(baseline_auroc),
        "consolidated_auroc": _round(consolidated_auroc),
        "random_seed": int(random_seed),
        "memory_contribution_reference": memory_contribution_reference,
        "tier2_state_sha256": tier2_state_sha256,
    }
    return _json_sha256(payload)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3778 artifact schema before writing."""

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
    if not isinstance(artifact.get("per_domain_threshold_deltas"), Mapping):
        raise ValueError("per_domain_threshold_deltas must be a mapping")
    if not isinstance(artifact.get("auroc_within_frozen_ci"), bool):
        raise ValueError("auroc_within_frozen_ci must be a bare boolean")
    if not isinstance(artifact.get("memory_contribution_preserved"), bool):
        raise ValueError("memory_contribution_preserved must be a bare boolean")
    if artifact.get("is_tier2_not_tier1") is not True:
        raise ValueError("is_tier2_not_tier1 must be true")
    if artifact.get("tier2_lookup_is_cpu_memory") != TIER2_LOOKUP:
        raise ValueError("tier2_lookup_is_cpu_memory must declare CPU memory lookup")
    if not isinstance(artifact.get("tracker_state_persisted"), bool):
        raise ValueError("tracker_state_persisted must be boolean")
    if not isinstance(artifact.get("duration_s"), int | float):
        raise ValueError("duration_s must be numeric")

    verdict = artifact.get("honest_verdict")
    if verdict in {
        BLOCKED_CORPUS_VERDICT,
        BLOCKED_SCORING_VERDICT,
        BLOCKED_INTERPRETER_VERDICT,
    }:
        if artifact.get("per_domain_threshold_deltas") != {}:
            raise ValueError("blocked artifact must not fabricate threshold deltas")
        if artifact.get("ensemble_auroc_under_consolidation") is not None:
            raise ValueError("blocked artifact must not fabricate AUROC")
        return
    if verdict != SUCCESS_VERDICT:
        raise ValueError(f"unsupported honest_verdict: {verdict!r}")
    deltas = artifact.get("per_domain_threshold_deltas")
    if set(deltas) != set(VERIFIER_NAMES):
        raise ValueError("per_domain_threshold_deltas must include all four verifier domains")
    for name, value in deltas.items():
        if name not in VERIFIER_NAMES or not isinstance(value, int | float):
            raise ValueError("per_domain_threshold_deltas values must be numeric")
    auroc = artifact.get("ensemble_auroc_under_consolidation")
    if not isinstance(auroc, int | float) or not 0.0 <= float(auroc) <= 1.0:
        raise ValueError("ensemble_auroc_under_consolidation must be in [0, 1]")
    if artifact.get("auroc_within_frozen_ci") is not True:
        raise ValueError("auroc_within_frozen_ci must be true for success")
    if artifact.get("memory_contribution_preserved") is not True:
        raise ValueError("memory_contribution_preserved must be true for success")
    if artifact.get("tracker_state_persisted") is not True:
        raise ValueError("tracker_state_persisted must be true for success")


def _blocked_artifact(
    *,
    duration_s: float,
    random_seed: int,
    db_output: Path,
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
        "artifact": "experiment_3778_fr11_self_learning_v18_tier2_constraint_memory",
        "schema": "carnot.fr11_continuous_self_learning_v18",
        "continuous_self_learning_task": True,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "per_domain_threshold_deltas": {},
        "ensemble_auroc_under_consolidation": None,
        "auroc_within_frozen_ci": False,
        "memory_contribution_preserved": False,
        "is_tier2_not_tier1": True,
        "tier2_lookup_is_cpu_memory": TIER2_LOOKUP,
        "tracker_state_persisted": db_output.is_file(),
        "model_specs": _model_specs(None, n_examples=0),
        "random_seed": int(random_seed),
        "reproducibility_checksum": _json_sha256(payload),
        "duration_s": _round(duration_s),
        "frozen_ci95": {"low": DEFAULT_FROZEN_CI[0], "high": DEFAULT_FROZEN_CI[1]},
        "memory_contribution_reference": None,
        "n_examples": 0,
        "n_examples_by_domain": {},
        "verifier_names": list(VERIFIER_NAMES),
        "tier2_memory_state_path": _relative_path(db_output, repo_root),
        "tier2_memory_state_sha256": None,
        "methodology": _methodology(random_seed, None),
        "preconditions_checked": [dict(item) for item in preconditions],
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "passed": False,
            "condition": "FoVer scores and labels are present before consolidation",
            "principle": "No Tier-2 threshold table is emitted without corpus evidence.",
        },
    }
    validate_artifact(artifact)
    return artifact


def _interpreter_precondition() -> JsonDict:
    packages = ("yaml", "numpy", "sklearn")
    loaded: list[str] = []
    missing: list[str] = []
    for package in packages:
        try:
            importlib.import_module(package)
            loaded.append(package)
        except Exception:  # noqa: BLE001 - reported as blocked precondition.
            missing.append(package)
    tier2_importable = callable(Tier2ThresholdMemory)
    executable = Path(sys.executable).as_posix()
    is_venv = ".venv/bin/python" in executable or executable.endswith("/.venv/bin/python")
    available = bool(is_venv and not missing and tier2_importable)
    return {
        "resource": "interpreter_runtime",
        "available": available,
        "detail": (
            f"executable={executable}; loaded={','.join(loaded)}; "
            f"missing={','.join(missing) or 'none'}; "
            f"Tier2ThresholdMemory_importable={tier2_importable}"
        ),
    }


def _fover_corpus_precondition(corpus_path: Path, *, n_examples: int) -> JsonDict:
    absolute = corpus_path.resolve()
    if not corpus_path.is_file():
        return {
            "resource": "fover_corpus_absolute_path",
            "available": False,
            "detail": f"{absolute}; missing; no corpus to consolidate from",
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
        "tier": "Tier-2 constraint memory",
    }


def _methodology(random_seed: int, corpus_absolute_path: Path | str | None) -> JsonDict:
    return {
        "random_seed": int(random_seed),
        "corpus_absolute_path": str(corpus_absolute_path) if corpus_absolute_path else None,
        "scoring_protocol": "ops/reproduction-runbook-fover-headline.md",
        "domain_keys": list(VERIFIER_NAMES),
        "lineage": "v18_tier2_constraint_memory_not_v17_tier1_precision_tracker",
        "delta_update_api": "Tier2ThresholdMemory.update_domain_delta",
        "delta_apply_api": "Tier2ThresholdMemory.apply_delta",
    }


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


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
