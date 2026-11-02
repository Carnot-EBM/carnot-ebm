"""Exp 3680 dependency-aware FoVer dual-condition integrity.

This module measures the Exp 3667 dependency-aware weighting under the same
memory-ablation protocol that produced the frozen FoVer headline. The important
distinction is that this is still verifier replay over cached FoVer rows: the
verifiers score only the row text/candidate content, while labels are used
afterward for AUROC, dependency-aware weight fitting, and significance tests.

Spec: REQ-VERIFY-3680, SCENARIO-VERIFY-3680.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot.eval import fover_dual_condition_integrity_v4 as exp2850
from carnot.eval.fover_memory_leakage_v3 import (
    CONDITION_ARCHITECTURE_ONLY,
    CONDITION_PRODUCTION,
    DEFAULT_RANDOM_SEEDS,
    ConditionScoringError,
    _fr11_memory_score,
    _label_to_int,
    _load_fr11_memory_index,
    _read_fover_rows,
    _score_text_verifiers,
    _select_balanced_subset,
    _subset_sha,
    discover_fr11_state_files,
    state_files_restored_sha_match,
    temporarily_move_state_files,
)
from carnot.verify import correlation_aware_weighting_paradox_diagnosis as exp3656
from carnot.verify import dependency_aware_weighting_clean as exp3667
from carnot.verify import weaver_peer_comparison_v3 as exp3644


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3680_dependency_aware_dual_condition_integrity.json")
EXP2850_REL_PATH = Path("results/experiment_2850_fover_dual_condition_integrity_v4.json")
DEFAULT_N_EXAMPLES = 1000
DEFAULT_RANDOM_SEED = DEFAULT_RANDOM_SEEDS[0]
DEFAULT_BOOTSTRAP_REPS = 200
FROZEN_HEADLINE_AUROC = 0.9131
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates (principle: scores the cached "
    "FoVer corpus under the dual-condition protocol; no LLM load)."
)

SUCCESS_CONFIRMED = (
    "complete: dependency_aware_g1_rigor_confirmed_headline_candidate_exceeds_frozen_0_9131"
)
SUCCESS_NO_GAIN = (
    "complete: dependency_aware_no_significant_gain_under_g1_protocol_frozen_headline_stands"
)
BLOCKED_VERDICT = "complete: blocked_fover_corpus_or_dependency_weighting_or_g1_source_unavailable"
TERMINAL_VERDICTS = (SUCCESS_CONFIRMED, SUCCESS_NO_GAIN, BLOCKED_VERDICT)
OUTCOME_CATEGORIES = (
    "g1_rigor_confirmed",
    "no_significant_gain_under_protocol",
    "blocked",
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "production_auroc_dependency_aware",
    "production_auroc_carnot_current",
    "frozen_headline_auroc",
    "production_auroc_ci",
    "learning_contribution_dependency_aware",
    "dependency_vs_carnot_delta_ci",
    "delong_p_dependency_vs_carnot",
    "n_seeds",
    "n_examples",
    "adversarial_verify_clean",
    "leak_free",
    "dependency_aware_g1_rigor_confirmed",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": INFERENCE_SUBSTRATE,
    "production_auroc_dependency_aware": (
        "The candidate-new-headline number: dependency-aware AUROC in the "
        "production condition under the frozen G1 protocol -- one field, no alias."
    ),
    "production_auroc_carnot_current": (
        "Carnot's current weighting under the IDENTICAL protocol -- the "
        "in-artifact bar; one field, no alias."
    ),
    "frozen_headline_auroc": (
        "The frozen 0.9131 -- recorded verbatim so the candidate is explicitly "
        "compared, never silently substituted."
    ),
    "production_auroc_ci": (
        "Pooled bootstrap CI of the dependency-aware production AUROC -- a "
        "candidate headline needs a CI like the frozen one."
    ),
    "learning_contribution_dependency_aware": (
        "The architecture-only memory-ablation delta recomputed under the new "
        "weighting -- the FR-11 contribution must be re-measured for a re-freeze."
    ),
    "dependency_vs_carnot_delta_ci": (
        "Paired delta + bootstrap CI of dependency-aware minus Carnot under the protocol."
    ),
    "delong_p_dependency_vs_carnot": (
        "DeLong paired significance under the protocol -- a point estimate alone "
        "cannot move a headline."
    ),
    "n_seeds": "Replication count (>=5, matching the frozen G1 source).",
    "n_examples": "Sample-size rigor (FoVer n>=1000).",
    "adversarial_verify_clean": (
        "True iff the artifact passes adversarial_verify with no TAUTOLOGY/critical "
        "flag -- a G1 candidate must be adversarial-clean."
    ),
    "leak_free": (
        "True iff verifiers scored (input, candidate) only and AUROC < 0.99 -- a "
        "leaked AUROC cannot be a headline."
    ),
    "dependency_aware_g1_rigor_confirmed": (
        "BARE bool. True iff dependency-aware production AUROC > frozen 0.9131 "
        "AND delta CI excludes 0 AND DeLong p<0.05 AND adversarial_verify_clean "
        "AND leak_free -- the G1-candidate gate. STORE AS BARE true/false -- "
        "gates exp3681."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


@dataclass(frozen=True)
class OutcomeClassification:
    """Terminal verdict and bare downstream gate for one measured outcome."""

    category: str
    terminal_verdict: str
    dependency_aware_g1_rigor_confirmed: bool


@dataclass(frozen=True)
class ConditionScoreRows:
    """Verifier score columns for one seed under both G1 memory conditions.

    The row labels are retained for AUROC and weight fitting, but the verifier
    columns themselves must be produced from the FoVer input/candidate text
    only. Keeping both conditions in one object makes row parity auditable.
    """

    seed: int
    labels: Sequence[int]
    production_scores_by_verifier: Mapping[str, Sequence[float]]
    architecture_scores_by_verifier: Mapping[str, Sequence[float]]
    production_state_visible_count: int = 0
    architecture_state_visible_count: int = 0
    subset_sha256: str | None = None
    architecture_subset_sha256: str | None = None


@dataclass(frozen=True)
class _ConditionVerifierScores:
    labels: list[int]
    scores_by_verifier: dict[str, list[float]]
    state_visible_count: int
    subset_sha256: str


def classify_outcome(
    *,
    blocked: bool,
    dependency_aware_auroc: float | None,
    frozen_headline_auroc: float,
    delta_ci: Mapping[str, Any] | None,
    delong_p: float | None,
    adversarial_verify_clean: bool,
    leak_free: bool,
) -> OutcomeClassification:
    """Map measured G1-rigor statistics onto the allowed terminal outcomes."""

    if blocked:
        return OutcomeClassification("blocked", BLOCKED_VERDICT, False)
    ci = list((delta_ci or {}).get("ci95") or [])
    ci_excludes_zero_positive = len(ci) == 2 and float(ci[0]) > 0.0
    confirmed = (
        dependency_aware_auroc is not None
        and float(dependency_aware_auroc) > float(frozen_headline_auroc)
        and ci_excludes_zero_positive
        and delong_p is not None
        and float(delong_p) < 0.05
        and bool(adversarial_verify_clean)
        and bool(leak_free)
    )
    if confirmed:
        return OutcomeClassification("g1_rigor_confirmed", SUCCESS_CONFIRMED, True)
    return OutcomeClassification("no_significant_gain_under_protocol", SUCCESS_NO_GAIN, False)


def compute_leak_free(
    *,
    verifier_scoring_input_candidate_only: bool,
    production_auroc_dependency_aware: float | None,
    n_examples: int,
) -> bool:
    """Return false for label-scoring leaks or implausible ceiling AUROC."""

    if not verifier_scoring_input_candidate_only:
        return False
    if production_auroc_dependency_aware is None:
        return False
    if int(n_examples) >= 1000 and float(production_auroc_dependency_aware) >= 0.99:
        return False
    return True


def build_artifact(
    repo_root: Path,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    n_examples: int = DEFAULT_N_EXAMPLES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    bootstrap_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPS,
    adversarial_verify_clean: bool = False,
) -> dict[str, Any]:
    """Build the Exp 3680 artifact from local FoVer rows or fail closed."""

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    preconditions = probe_preconditions(root, n_examples=n_examples)
    if not all(item["available"] for item in preconditions):
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            preconditions=preconditions,
        )

    try:
        exp2850_source = load_exp2850_source_artifact(root)
        state_files = discover_fr11_state_files(root)
        rows = [
            score_dual_condition_rows(
                root,
                seed=int(seed),
                n_examples=n_examples,
                state_files=state_files,
            )
            for seed in random_seeds
        ]
    except Exception as exc:  # noqa: BLE001 - terminal artifact must not half-claim.
        failed = [
            *preconditions,
            {
                "resource": "dual_condition_scoring",
                "available": False,
                "detail": f"{type(exc).__name__}: {exc}",
            },
        ]
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            preconditions=failed,
        )

    return build_artifact_from_condition_rows(
        rows,
        started_s=start,
        now_s=now_s,
        random_seed=random_seed,
        bootstrap_seeds=bootstrap_seeds,
        n_bootstrap=n_bootstrap,
        adversarial_verify_clean=adversarial_verify_clean,
        preconditions=preconditions,
        exp2850_source=exp2850_source,
    )


def probe_preconditions(repo_root: Path, *, n_examples: int) -> list[dict[str, Any]]:
    """Check the exact resources needed before scoring any G1 condition."""

    root = Path(repo_root)
    fover_path = root / "data" / "fover_corpus.jsonl"
    labeled_count = exp2850._count_labeled_fover_rows(fover_path)
    checks: list[dict[str, Any]] = [
        {
            "resource": "fover_corpus",
            "available": fover_path.is_file() and labeled_count >= int(n_examples),
            "detail": f"labeled_rows={labeled_count}; required>={int(n_examples)}",
        }
    ]

    try:
        smoke = _score_text_verifiers(["1 + 1 = 2"])
        scoring_available = set(smoke) == set(exp3644.VERIFIER_NAMES[1:])
        detail = "loaded=" + ",".join(sorted(smoke))
    except Exception as exc:  # noqa: BLE001 - precondition diagnostics belong in artifact.
        scoring_available = False
        detail = f"{type(exc).__name__}: {exc}"
    checks.append(
        {
            "resource": "four_exp2837_scoring_verifiers",
            "available": scoring_available,
            "detail": detail,
        }
    )

    try:
        state_files = discover_fr11_state_files(root)
        memory_index = _load_fr11_memory_index(root) if state_files else {}
        memory_loaded = bool(memory_index.get("question_ids") or memory_index.get("prompt_token_sets"))
        memory_detail = f"state_files={len(state_files)}; memory_loaded={memory_loaded}"
    except Exception as exc:  # noqa: BLE001
        state_files = []
        memory_loaded = False
        memory_detail = f"{type(exc).__name__}: {exc}"
    checks.append(
        {
            "resource": "fr11_session_memory_state",
            "available": bool(state_files) and memory_loaded,
            "detail": memory_detail,
        }
    )

    dependency_functions = (
        exp3656.fit_dependency_aware_weights,
        exp3656.learn_dependency_graph,
        exp3656.dependency_aware_crossfit_scores,
        exp3667.score_weighting_panel,
    )
    checks.append(
        {
            "resource": "exp3667_dependency_aware_implementation",
            "available": all(callable(func) for func in dependency_functions),
            "detail": "dependency graph, crossfit scores, and clean panel importable",
        }
    )

    try:
        source = load_exp2850_source_artifact(root)
        source_detail = (
            f"n_seeds={source.get('n_seeds')}; "
            f"condition_a={source.get('condition_a_production_auroc_mean')}"
        )
        source_available = True
    except Exception as exc:  # noqa: BLE001
        source_detail = f"{type(exc).__name__}: {exc}"
        source_available = False
    checks.append(
        {
            "resource": "exp2850_g1_source_artifact",
            "available": source_available,
            "detail": source_detail,
        }
    )
    return checks


def load_exp2850_source_artifact(repo_root: Path) -> dict[str, Any]:
    """Load the frozen-protocol source so protocol parity is explicit."""

    path = Path(repo_root) / EXP2850_REL_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    for field in (
        "condition_a_production_auroc_mean",
        "condition_b_architecture_only_auroc_mean",
        "n_seeds",
        "n_examples",
    ):
        if field not in payload:
            raise ValueError(f"Exp 2850 source artifact is missing {field}")
    return payload


def score_dual_condition_rows(
    repo_root: Path,
    *,
    seed: int,
    n_examples: int,
    state_files: Sequence[Mapping[str, object]],
) -> ConditionScoreRows:
    """Score one seed under production and memory-ablation conditions."""

    root = Path(repo_root)
    production = score_condition_verifier_rows(
        root,
        seed=seed,
        n_examples=n_examples,
        condition=CONDITION_PRODUCTION,
        require_no_state=False,
    )
    backup_root = Path("/tmp") / f"carnot_exp3680_fr11_state_backup_{os.getpid()}_{seed}"
    with temporarily_move_state_files(root, state_files, backup_root):
        architecture = score_condition_verifier_rows(
            root,
            seed=seed,
            n_examples=n_examples,
            condition=CONDITION_ARCHITECTURE_ONLY,
            require_no_state=True,
        )
    if production.labels != architecture.labels:
        raise ConditionScoringError("production and architecture labels diverged")
    if production.subset_sha256 != architecture.subset_sha256:
        raise ConditionScoringError("production and architecture subsets diverged")
    if not state_files_restored_sha_match(root, state_files):
        raise ConditionScoringError("FR-11 state restore SHA256 mismatch")
    return ConditionScoreRows(
        seed=int(seed),
        labels=production.labels,
        production_scores_by_verifier=production.scores_by_verifier,
        architecture_scores_by_verifier=architecture.scores_by_verifier,
        production_state_visible_count=production.state_visible_count,
        architecture_state_visible_count=architecture.state_visible_count,
        subset_sha256=production.subset_sha256,
        architecture_subset_sha256=architecture.subset_sha256,
    )


def score_condition_verifier_rows(
    repo_root: Path,
    *,
    seed: int,
    n_examples: int,
    condition: str,
    require_no_state: bool,
) -> _ConditionVerifierScores:
    """Return four verifier score columns for one FoVer condition.

    Condition B keeps the FR-11 verifier name but fills it with zero evidence.
    That preserves the same four-column weighting interface while representing
    the memory-ablation truth: no session-memory state is visible.
    """

    root = Path(repo_root)
    state_visible_count = len(discover_fr11_state_files(root))
    if require_no_state and state_visible_count != 0:
        raise ConditionScoringError(
            f"architecture-only condition saw {state_visible_count} FR-11 state files"
        )
    rows = _select_balanced_subset(
        _read_fover_rows(root / "data" / "fover_corpus.jsonl"),
        seed=int(seed),
        n_examples=int(n_examples),
    )
    labels = [_label_to_int(row["label"]) for row in rows]
    texts = [str(row.get("step_text", "")) for row in rows]
    text_scores = _score_text_verifiers(texts)
    if condition == CONDITION_PRODUCTION:
        memory_index = _load_fr11_memory_index(root)
        fr11_scores = [_fr11_memory_score(row, memory_index) for row in rows]
    elif condition == CONDITION_ARCHITECTURE_ONLY:
        fr11_scores = [0.0 for _row in rows]
    else:
        raise ConditionScoringError(f"unknown condition: {condition}")

    scores_by_verifier = {
        "fr11_session_memory": [float(value) for value in fr11_scores],
        "tier0r_curry_howard": [float(value) for value in text_scores["tier0r_curry_howard"]],
        "tier0s_arithmetic_gap": [float(value) for value in text_scores["tier0s_arithmetic_gap"]],
        "tier0u_logical_consistency": [
            float(value) for value in text_scores["tier0u_logical_consistency"]
        ],
    }
    return _ConditionVerifierScores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        state_visible_count=state_visible_count,
        subset_sha256=_subset_sha(rows),
    )


def build_artifact_from_condition_rows(
    condition_rows: Sequence[ConditionScoreRows],
    *,
    started_s: float,
    now_s: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    bootstrap_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPS,
    adversarial_verify_clean: bool = False,
    verifier_scoring_input_candidate_only: bool = True,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
    exp2850_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble Exp 3680 statistics from already-scored condition rows."""

    rows = list(condition_rows)
    if not rows:
        raise ValueError("at least one condition row panel is required")

    per_seed: list[dict[str, Any]] = []
    pooled_labels: list[np.ndarray] = []
    pooled_dependency: list[np.ndarray] = []
    pooled_carnot: list[np.ndarray] = []
    production_dep_aurocs: list[float] = []
    production_carnot_aurocs: list[float] = []
    architecture_dep_aurocs: list[float] = []
    names = list(exp3644.VERIFIER_NAMES)

    for row in rows:
        labels = np.asarray(row.labels, dtype=np.int64)
        production_matrix = exp3656.score_matrix(row.production_scores_by_verifier, names)
        architecture_matrix = exp3656.score_matrix(row.architecture_scores_by_verifier, names)
        if production_matrix.shape[0] != len(labels) or architecture_matrix.shape[0] != len(labels):
            raise ValueError("labels and verifier scores must have the same length")
        _require_binary_labels(labels)

        production_scores = exp3667.score_weighting_panel(
            labels=labels,
            score_matrix=production_matrix,
            verifier_names=names,
            random_seed=int(row.seed),
        )
        architecture_scores = exp3667.score_weighting_panel(
            labels=labels,
            score_matrix=architecture_matrix,
            verifier_names=names,
            random_seed=int(row.seed),
        )
        dep_prod = exp3644.tie_aware_auroc(labels, production_scores["dependency_aware_proper"])
        carnot_prod = exp3644.tie_aware_auroc(labels, production_scores["carnot_current"])
        dep_arch = exp3644.tie_aware_auroc(labels, architecture_scores["dependency_aware_proper"])
        production_dep_aurocs.append(dep_prod)
        production_carnot_aurocs.append(carnot_prod)
        architecture_dep_aurocs.append(dep_arch)
        pooled_labels.append(labels)
        pooled_dependency.append(np.asarray(production_scores["dependency_aware_proper"], dtype=np.float64))
        pooled_carnot.append(np.asarray(production_scores["carnot_current"], dtype=np.float64))
        per_seed.append(
            {
                "seed": int(row.seed),
                "n_examples": int(len(labels)),
                "production_auroc_dependency_aware": _round_metric(dep_prod),
                "production_auroc_carnot_current": _round_metric(carnot_prod),
                "architecture_auroc_dependency_aware": _round_metric(dep_arch),
                "learning_contribution_dependency_aware": _round_metric(dep_prod - dep_arch),
                "production_state_visible_count": int(row.production_state_visible_count),
                "architecture_state_visible_count": int(row.architecture_state_visible_count),
                "subset_sha256": row.subset_sha256,
                "architecture_subset_sha256": row.architecture_subset_sha256 or row.subset_sha256,
            }
        )

    pooled_label_arr = np.concatenate(pooled_labels)
    pooled_dep_arr = np.concatenate(pooled_dependency)
    pooled_carnot_arr = np.concatenate(pooled_carnot)
    production_auroc_dependency = float(np.mean(production_dep_aurocs))
    production_auroc_carnot = float(np.mean(production_carnot_aurocs))
    architecture_auroc_dependency = float(np.mean(architecture_dep_aurocs))
    production_ci = exp3667.bootstrap_auroc_ci(
        pooled_label_arr,
        pooled_dep_arr,
        seeds=bootstrap_seeds,
        n_bootstrap=n_bootstrap,
    )
    delta_ci = exp3667.paired_delta_ci(
        pooled_label_arr,
        pooled_dep_arr,
        pooled_carnot_arr,
        seeds=bootstrap_seeds,
        n_bootstrap=n_bootstrap,
    )
    delong = exp3667.paired_delong_test(pooled_label_arr, pooled_dep_arr, pooled_carnot_arr)
    n_examples = int(min(len(row.labels) for row in rows))
    leak_free = compute_leak_free(
        verifier_scoring_input_candidate_only=verifier_scoring_input_candidate_only,
        production_auroc_dependency_aware=production_auroc_dependency,
        n_examples=n_examples,
    )
    classification = classify_outcome(
        blocked=False,
        dependency_aware_auroc=production_auroc_dependency,
        frozen_headline_auroc=FROZEN_HEADLINE_AUROC,
        delta_ci=delta_ci,
        delong_p=delong["p_value"],
        adversarial_verify_clean=adversarial_verify_clean,
        leak_free=leak_free,
    )
    source = dict(exp2850_source or {})
    artifact = {
        "artifact": "experiment_3680_dependency_aware_dual_condition_integrity",
        "schema": "carnot.dependency_aware_dual_condition_integrity.v1",
        "honest_verdict": classification.terminal_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "production_auroc_dependency_aware": _round_metric(production_auroc_dependency),
        "production_auroc_carnot_current": _round_metric(production_auroc_carnot),
        "frozen_headline_auroc": FROZEN_HEADLINE_AUROC,
        "production_auroc_ci": production_ci,
        "learning_contribution_dependency_aware": _round_metric(
            production_auroc_dependency - architecture_auroc_dependency
        ),
        "dependency_vs_carnot_delta_ci": delta_ci,
        "delong_p_dependency_vs_carnot": _round_p(float(delong["p_value"])),
        "n_seeds": int(len(rows)),
        "n_examples": n_examples,
        "adversarial_verify_clean": bool(adversarial_verify_clean),
        "leak_free": bool(leak_free),
        "dependency_aware_g1_rigor_confirmed": (
            classification.dependency_aware_g1_rigor_confirmed
        ),
        "random_seed": int(random_seed),
        "random_seeds_used": [int(row.seed) for row in rows],
        "bootstrap_seeds": [int(seed) for seed in bootstrap_seeds],
        "n_bootstrap_per_seed": int(n_bootstrap),
        "reproducibility_checksum": reproducibility_checksum(
            rows=rows,
            verifier_names=names,
            random_seed=random_seed,
            bootstrap_seeds=bootstrap_seeds,
            exp2850_source=source,
        ),
        "duration_s": _round_metric(_duration(float(started_s), now_s)),
        "architecture_auroc_dependency_aware": _round_metric(architecture_auroc_dependency),
        "production_auroc_dependency_aware_vs_frozen_headline_delta": _round_metric(
            production_auroc_dependency - FROZEN_HEADLINE_AUROC
        ),
        "verifier_names": names,
        "verifier_scoring_input_candidate_only": bool(verifier_scoring_input_candidate_only),
        "per_seed_results": per_seed,
        "delong_dependency_vs_carnot": delong,
        "exp2850_protocol_source": {
            "path": EXP2850_REL_PATH.as_posix(),
            "condition_a_production_auroc_mean": source.get(
                "condition_a_production_auroc_mean"
            ),
            "condition_b_architecture_only_auroc_mean": source.get(
                "condition_b_architecture_only_auroc_mean"
            ),
            "n_seeds": source.get("n_seeds"),
            "n_examples": source.get("n_examples"),
            "reproducibility_checksum": source.get("reproducibility_checksum"),
        },
        "dependency_aware_training_protocol": {
            "method": "stratified_crossfit_graph_sparse_signed_fisher_weights",
            "folds": exp3667.DEFAULT_CROSSFIT_FOLDS,
            "learned_on_labels_for_weight_fit": True,
            "verifier_scores_use_gold_label": False,
            "dependency_reference": (
                "Learning Dependency Structures for Weak Supervision Models "
                "(arXiv:1903.05844)"
            ),
        },
        "de_tautology_note": (
            "Each conceptually distinct AUROC is stored under one top-level field; "
            "the frozen 0.9131 remains a comparison constant, not a replacement."
        ),
        "leak_guard": {
            "red_flag_rule": "AUROC >= 0.99 on n>=1000 sets leak_free=false",
            "verifiers_score_input_candidate_only": bool(verifier_scoring_input_candidate_only),
            "triggered": not bool(leak_free),
        },
        "acceptance_gate": {
            "condition": (
                "production_auroc_dependency_aware present AND "
                "production_auroc_carnot_current present AND "
                "adversarial_verify_clean == true AND leak_free == true AND n_seeds >= 5"
            ),
            "principle": (
                "A G1-rigor headline candidate requires both weightings measured "
                "under the frozen dual-condition protocol, >=5 seeds, "
                "adversarial-clean and leak-free -- otherwise it is not comparable "
                "to the frozen G1 source."
            ),
            "passed": bool(
                production_auroc_dependency is not None
                and production_auroc_carnot is not None
                and adversarial_verify_clean
                and leak_free
                and len(rows) >= 5
            ),
        },
        "preconditions_checked": [dict(item) for item in preconditions or []],
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def reproducibility_checksum(
    *,
    rows: Sequence[ConditionScoreRows],
    verifier_names: Sequence[str],
    random_seed: int,
    bootstrap_seeds: Sequence[int],
    exp2850_source: Mapping[str, Any],
) -> str:
    """Hash score columns, seeds, verifier order, and frozen-protocol source."""

    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(int(row.seed)).encode("ascii"))
        digest.update(np.ascontiguousarray(row.labels, dtype=np.int64).tobytes())
        digest.update(
            np.ascontiguousarray(
                exp3656.score_matrix(row.production_scores_by_verifier, verifier_names),
                dtype=np.float64,
            ).tobytes()
        )
        digest.update(
            np.ascontiguousarray(
                exp3656.score_matrix(row.architecture_scores_by_verifier, verifier_names),
                dtype=np.float64,
            ).tobytes()
        )
    digest.update(json.dumps(list(verifier_names), separators=(",", ":")).encode("utf-8"))
    digest.update(str(int(random_seed)).encode("ascii"))
    digest.update(json.dumps([int(seed) for seed in bootstrap_seeds]).encode("ascii"))
    digest.update(
        json.dumps(
            {
                "condition_a_production_auroc_mean": exp2850_source.get(
                    "condition_a_production_auroc_mean"
                ),
                "condition_b_architecture_only_auroc_mean": exp2850_source.get(
                    "condition_b_architecture_only_auroc_mean"
                ),
                "reproducibility_checksum": exp2850_source.get("reproducibility_checksum"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3680 schema and the downstream bare bool contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be present")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    verdict = artifact.get("honest_verdict")
    if verdict not in TERMINAL_VERDICTS:
        raise ValueError(f"unsupported honest_verdict: {verdict!r}")
    for field in (
        "adversarial_verify_clean",
        "leak_free",
        "dependency_aware_g1_rigor_confirmed",
    ):
        if type(artifact.get(field)) is not bool:
            raise ValueError(f"{field} must be a bare boolean")
    if not isinstance(artifact.get("duration_s"), int | float):
        raise ValueError("duration_s must be numeric")
    if verdict == BLOCKED_VERDICT:
        return
    if int(artifact.get("n_seeds", 0)) < 5:
        raise ValueError("n_seeds must be at least 5")
    if int(artifact.get("n_examples", 0)) < DEFAULT_N_EXAMPLES:
        raise ValueError(f"n_examples must be at least {DEFAULT_N_EXAMPLES}")
    for field in (
        "production_auroc_dependency_aware",
        "production_auroc_carnot_current",
        "frozen_headline_auroc",
        "architecture_auroc_dependency_aware",
    ):
        _validate_auroc_field(artifact, field)
    _validate_ci(artifact.get("production_auroc_ci"), "production_auroc_ci")
    _validate_ci(artifact.get("dependency_vs_carnot_delta_ci"), "dependency_vs_carnot_delta_ci")
    delong_p = artifact.get("delong_p_dependency_vs_carnot")
    if not _is_finite_number(delong_p) or not 0.0 <= float(delong_p) <= 1.0:
        raise ValueError("delong_p_dependency_vs_carnot must be finite and in [0, 1]")


def write_artifact(
    repo_root: Path,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, self-verify, and write the Exp 3680 terminal JSON artifact."""

    root = Path(repo_root)
    artifact = build_artifact(root, started_s=started_s, now_s=now_s)
    target = root / output_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if artifact["honest_verdict"] != BLOCKED_VERDICT:
        report = run_adversarial_verify_report(target)
        clean = adversarial_report_is_clean(report)
        classification = classify_outcome(
            blocked=False,
            dependency_aware_auroc=artifact["production_auroc_dependency_aware"],
            frozen_headline_auroc=artifact["frozen_headline_auroc"],
            delta_ci=artifact["dependency_vs_carnot_delta_ci"],
            delong_p=artifact["delong_p_dependency_vs_carnot"],
            adversarial_verify_clean=clean,
            leak_free=bool(artifact["leak_free"]),
        )
        artifact["adversarial_verify_clean"] = clean
        artifact["dependency_aware_g1_rigor_confirmed"] = (
            classification.dependency_aware_g1_rigor_confirmed
        )
        artifact["honest_verdict"] = classification.terminal_verdict
        artifact["acceptance_gate"]["passed"] = bool(
            artifact["production_auroc_dependency_aware"] is not None
            and artifact["production_auroc_carnot_current"] is not None
            and clean
            and artifact["leak_free"]
            and artifact["n_seeds"] >= 5
        )
        artifact["adversarial_verify_report"] = {
            "flag_count": int(report.get("flag_count", 0)),
            "max_severity": report.get("max_severity"),
            "flags": list(report.get("flags") or []),
        }
        validate_artifact(artifact)
        target.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def run_adversarial_verify_report(path: Path) -> dict[str, Any]:  # pragma: no cover
    """Run the repository adversarial verifier without shelling through JSON text."""

    script_path = REPO_ROOT / "scripts" / "adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.verify_artifact(Path(path)))


def adversarial_report_is_clean(report: Mapping[str, Any]) -> bool:
    """True when adversarial verification found no TAUTOLOGY or critical flag."""

    for flag in list(report.get("flags") or []):
        kind = str(dict(flag).get("kind", ""))
        severity = str(dict(flag).get("severity", ""))
        if kind == "TAUTOLOGY" or severity == "critical":
            return False
    return True


def _blocked_artifact(
    *,
    duration_s: float,
    random_seed: int,
    preconditions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = json.dumps(
        {"preconditions": [dict(item) for item in preconditions], "random_seed": random_seed},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    artifact: dict[str, Any] = {
        "artifact": "experiment_3680_dependency_aware_dual_condition_integrity",
        "schema": "carnot.dependency_aware_dual_condition_integrity.v1",
        "honest_verdict": BLOCKED_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "production_auroc_dependency_aware": None,
        "production_auroc_carnot_current": None,
        "frozen_headline_auroc": FROZEN_HEADLINE_AUROC,
        "production_auroc_ci": None,
        "learning_contribution_dependency_aware": None,
        "dependency_vs_carnot_delta_ci": None,
        "delong_p_dependency_vs_carnot": None,
        "n_seeds": 0,
        "n_examples": 0,
        "adversarial_verify_clean": False,
        "leak_free": False,
        "dependency_aware_g1_rigor_confirmed": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": hashlib.sha256(payload).hexdigest(),
        "duration_s": _round_metric(duration_s),
        "preconditions_checked": [dict(item) for item in preconditions],
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "condition": (
                "production_auroc_dependency_aware present AND "
                "production_auroc_carnot_current present AND "
                "adversarial_verify_clean == true AND leak_free == true AND n_seeds >= 5"
            ),
            "principle": (
                "A G1-rigor headline candidate requires both weightings measured "
                "under the frozen dual-condition protocol, >=5 seeds, "
                "adversarial-clean and leak-free -- otherwise it is not comparable "
                "to the frozen G1 source."
            ),
            "passed": False,
        },
    }
    validate_artifact(artifact)
    return artifact


def _require_binary_labels(labels: np.ndarray) -> None:
    values = set(np.asarray(labels, dtype=np.int64).tolist())
    if values != {0, 1}:
        raise ValueError("labels must contain both binary classes 0 and 1")


def _validate_auroc_field(artifact: Mapping[str, Any], field: str) -> None:
    value = artifact.get(field)
    if not _is_finite_number(value) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")


def _validate_ci(value: Any, field: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    point = value.get("point")
    ci = value.get("ci95")
    if not _is_finite_number(point) or not isinstance(ci, list) or len(ci) != 2:
        raise ValueError(f"{field} must include point and ci95")
    if not all(_is_finite_number(item) for item in ci):
        raise ValueError(f"{field} bounds must be finite")
    if not float(ci[0]) <= float(point) <= float(ci[1]):
        raise ValueError(f"{field} must contain its point estimate")


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value))


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.time() if now_s is None else float(now_s)
    return max(0.0, end - float(started_s))


def _round_metric(value: float | int | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _round_p(value: float | int | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if numeric != 0.0 and abs(numeric) < 1e-6:
        return float(f"{numeric:.6g}")
    return round(numeric, 6)
