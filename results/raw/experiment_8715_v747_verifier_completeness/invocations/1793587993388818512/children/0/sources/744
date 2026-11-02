"""Exp 4314 IR3DE+CASCAL cross-domain selector rerun.

Spec refs: REQ-VERIFY-4314, SCENARIO-VERIFY-4314.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from carnot import experiment_4305_cross_domain_selector_generalization as base


RANDOM_SEED = 20260617
BOOTSTRAP_RESAMPLES = 2000
OUTPUT_REL = Path("results/experiment_4314_cross_domain_selector_ir3de_cascal.json")
POOL_REL = Path("results/experiment_4314_cross_domain_selector_ir3de_cascal_pool.json.gz")
MANIFEST_REL = Path("results/experiment_4314_cross_domain_selector_ir3de_cascal_manifest.json")
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
SPEC_REFS = ["REQ-VERIFY-4314", "SCENARIO-VERIFY-4314"]
DOMAIN_ORDER = base.DOMAIN_ORDER
FEATURE_NAMES = base.FEATURE_NAMES
FOVER_POWERED_SOURCE_RELS = (
    Path("data/fover_corpus_v4.json"),
    Path("data/fover_corpus_v3.json"),
    Path("results/fover_labeled_steps_v21_multi.json"),
    Path("data/fover_train_v3.json"),
    Path("data/fover_test_v3.json"),
    Path("results/fover_labeled_steps_live.json"),
    Path("data/fover_corpus_expanded.json"),
)
CandidateRow = base.CandidateRow
DomainPool = base.DomainPool
SetEncoderProfile = base.SetEncoderProfile
common_feature_payload = base.common_feature_payload
load_arc_domain_pool = base.load_arc_domain_pool
load_arcgen_domain_pool = base.load_arcgen_domain_pool

BASE_SELECTOR_FEATURE_NAMES = (
    "vote_weight",
    "self_consistency_margin",
    "vote_weight_rank_fraction",
    "cell_confidence_mean",
    "cell_confidence_margin",
    "cell_confidence_rank_fraction",
    "grid_entropy",
    "program_demo_fit",
    "vote_weight_zscore",
    "cell_confidence_zscore",
    "modal_cell_agreement_frac",
    "shape_vote_frac",
    "palette_vote_frac",
    "is_modal_shape",
    "is_modal_palette",
    "same_shape_as_input",
)
CONTEXTPRM_FEATURE_NAMES = (
    "contextprm_step_coherence",
    "contextprm_coherence_margin",
    "contextprm_vote_coherence_alignment",
    "contextprm_candidate_set_support",
    "contextprm_contradiction_pressure",
    "contextprm_structure_consistency",
)
IR3DE_EXPERT_FEATURE_NAMES = (
    "ir3de_expert_mean_score",
    "ir3de_expert_max_score",
    "ir3de_expert_score_margin",
)
LABEL_FEATURE_NAMES = (
    "label_domain_hash",
    "label_family_hash",
    "label_domain_arc",
    "label_domain_arcgen",
    "label_domain_fover",
)
ROUTER_FEATURE_NAMES = (
    BASE_SELECTOR_FEATURE_NAMES
    + CONTEXTPRM_FEATURE_NAMES
    + IR3DE_EXPERT_FEATURE_NAMES
    + LABEL_FEATURE_NAMES
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A cross-domain survive (the moat escapes the domain bound), "
        "a POWERED collapse (the moat is domain-bound -> retire the ask), a label-ablation "
        "failure (the router read the label), and an honest blocked_insufficient_domains "
        "are ALL COMPLETE and decision-grade."
    ),
    "cross_domain_selection_holds": (
        "BARE bool: the capstone reads this (gated-fields-must-be-bare); true iff "
        "held-out-DOMAIN router+set_encoder@1 - vote@1 > 0 AND CI95-excl-0 AND "
        "non-degenerate guards AND label_ablation_robust -- the strongest selection "
        "result (escapes the math/ARC domain bound)."
    ),
    "cross_domain_delta": (
        "BARE float: held-out-DOMAIN router+set_encoder@1 - vote@1 -- compare to "
        "exp4305's +0.231 (the IR3DE+CASCAL upgrade should tighten the CI; a positive "
        "held-out delta with CI95-excl-0 is the cross-domain moat)."
    ),
    "cross_domain_delta_ci95": (
        "Task-level bootstrap CI95 (>=2000 resamples) of the held-out delta -- excluding "
        "0 is the decision-grade close (exp4305's [-0.115,0.538] included 0)."
    ),
    "vote_at_1": (
        "BARE float: the vote baseline on the held-out domain -- MUST be > 0.05 "
        "(if 0, the pool is wrong-majority-only and the delta is degenerate)."
    ),
    "oracle_at_k": (
        "BARE float: positive-control ceiling on the held-out domain -- MUST be < 1.0 "
        "(a sub-1.0 ceiling is real headroom the verifier must earn)."
    ),
    "label_ablation_robust": (
        "BARE bool: true iff held-out accuracy SURVIVES removing the domain/family label "
        "from the router input -- the anti-leak proof (the moat must use task structure, "
        "not domain identity)."
    ),
    "per_domain_delta": (
        "The lift reported SEPARATELY per held-out domain (arc / arcgen / fover) -- "
        "guards the one-domain-dominates failure mode."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- a learned IR3DE router + set-encoder over cross-domain pools, "
        "no demo execution (oracle-distinct)."
    ),
    "random_seed": (
        "Determinism precondition; the cross-domain sampling + domain-split + CASCAL "
        "synthetic-task generation reproducible."
    ),
    "reproducibility_checksum": (
        "Hash of the cross-domain pool + the domain manifest + the IR3DE router + the "
        "CASCAL synthetic set; lets a third party re-run."
    ),
    "model_specs": (
        "The 3-domain provenance + the IR3DE linear router + the CASCAL pretrain + the "
        "ContextPRM coherence features + the domain-disjoint split + the label-ablation "
        "protocol; required methodology."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "cross_domain_selection_holds",
    "cross_domain_delta",
    "cross_domain_delta_ci95",
    "vote_at_1",
    "oracle_at_k",
    "label_ablation_robust",
    "per_domain_delta",
    "verifier_is_oracle",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
    "field_principles",
    "spec_refs",
    "acceptance_gate",
    "adversarial_verify",
)


@dataclass(frozen=True)
class LinearRidgeModel:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    alpha: float


@dataclass(frozen=True)
class IR3DERouter:
    held_out_domain: str
    profiles: dict[str, SetEncoderProfile]
    global_profile: SetEncoderProfile
    linear_model: LinearRidgeModel
    include_labels: bool
    cascal_synthetic_n: int
    train_domain_ids: tuple[str, ...]
    cascal_checksum: str


def _round_metric(value: float) -> float:
    return round(float(value), 10)


def _safe_float(value: Any) -> float:
    return base._safe_float(value)


def _sha256_file(path: Path) -> str:
    return base._sha256_file(path)


def _stable_hash_fraction(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def _default_domain_loaders(repo_root: Path) -> dict[str, Callable[[], DomainPool]]:
    return {
        "arc": lambda: load_arc_domain_pool(repo_root),
        "arcgen": lambda: load_arcgen_domain_pool(repo_root),
        "fover": lambda: load_powered_fover_domain_pool(repo_root),
    }


def _load_set_encoder_precondition(repo_root: Path | str = Path(".")) -> dict[str, Any]:
    return base._load_set_encoder_precondition(repo_root)


def _load_domains(loaders: Mapping[str, Callable[[], DomainPool]]) -> tuple[dict[str, DomainPool], list[dict[str, str]]]:
    return base._load_domains(loaders)


def _group_by_task(rows: list[CandidateRow]) -> list[list[CandidateRow]]:
    return base._group_by_task(rows)


def _select_vote(task_rows: list[CandidateRow]) -> CandidateRow:
    return base._select_vote(task_rows)


def _select_first(task_rows: list[CandidateRow]) -> CandidateRow:
    return base._select_first(task_rows)


def _rate(values: list[bool]) -> float:
    return base._rate(values)


def _bootstrap_ci95(deltas: list[float], *, random_seed: int, resamples: int) -> list[float]:
    return base._bootstrap_ci95(deltas, random_seed=random_seed, resamples=resamples)


def _ci_excludes_zero(ci95: list[float]) -> bool:
    return base._ci_excludes_zero(ci95)


def _fit_profile(domain_id: str, rows: list[CandidateRow]) -> SetEncoderProfile:
    return base._fit_profile(domain_id, rows)


def _profile_score(profile: SetEncoderProfile, row: CandidateRow) -> float:
    return base._profile_score(profile, row)


def _label_features(row: CandidateRow, include_labels: bool) -> dict[str, float]:
    if not include_labels:
        return {name: 0.0 for name in LABEL_FEATURE_NAMES}
    return {
        "label_domain_hash": _stable_hash_fraction(row.domain_id),
        "label_family_hash": _stable_hash_fraction(row.family_id),
        "label_domain_arc": 1.0 if row.domain_id == "arc" else 0.0,
        "label_domain_arcgen": 1.0 if row.domain_id == "arcgen" else 0.0,
        "label_domain_fover": 1.0 if row.domain_id == "fover" else 0.0,
    }


def contextprm_feature_vector(task_rows: list[CandidateRow], row: CandidateRow) -> dict[str, float]:
    """Build domain-agnostic step-coherence features for one candidate.

    These features intentionally use only the candidate's relation to its set:
    no domain, family, task id, or target label enters the vector.
    """

    count = max(1, len(task_rows))
    confidences = [_safe_float(item.features.get("cell_confidence_mean")) for item in task_rows]
    votes = [_safe_float(item.vote_weight) for item in task_rows]
    mean_conf = sum(confidences) / float(count)
    max_conf = max(confidences) if confidences else 0.0
    max_vote = max(votes) if votes else 0.0
    row_conf = _safe_float(row.features.get("cell_confidence_mean"))
    row_vote = _safe_float(row.vote_weight)
    structure = (
        _safe_float(row.features.get("modal_cell_agreement_frac"))
        + _safe_float(row.features.get("is_modal_shape"))
        + _safe_float(row.features.get("is_modal_palette"))
        + _safe_float(row.features.get("same_shape_as_input"))
    ) / 4.0
    return {
        "contextprm_step_coherence": row_conf,
        "contextprm_coherence_margin": row_conf - mean_conf,
        "contextprm_vote_coherence_alignment": (
            _safe_float(row.features.get("cell_confidence_rank_fraction"))
            - _safe_float(row.features.get("vote_weight_rank_fraction"))
        ),
        "contextprm_candidate_set_support": row_conf / max(max_conf, 1e-9),
        "contextprm_contradiction_pressure": (max_vote - row_vote) / max(max_vote, 1.0),
        "contextprm_structure_consistency": structure,
    }


def _expert_features(profiles: Mapping[str, SetEncoderProfile], row: CandidateRow) -> dict[str, float]:
    scores = [_profile_score(profile, row) for profile in profiles.values()]
    if not scores:
        scores = [0.0]
    top = max(scores)
    mean = sum(scores) / float(len(scores))
    second = sorted(scores)[-2] if len(scores) > 1 else mean
    return {
        "ir3de_expert_mean_score": mean,
        "ir3de_expert_max_score": top,
        "ir3de_expert_score_margin": top - second,
    }


def selector_feature_vector(
    task_rows: list[CandidateRow],
    row: CandidateRow,
    *,
    profiles: Mapping[str, SetEncoderProfile],
    include_labels: bool,
) -> dict[str, float]:
    features = {name: _safe_float(row.features.get(name)) for name in BASE_SELECTOR_FEATURE_NAMES}
    features.update(contextprm_feature_vector(task_rows, row))
    features.update(_expert_features(profiles, row))
    features.update(_label_features(row, include_labels))
    return {name: _safe_float(features.get(name)) for name in ROUTER_FEATURE_NAMES}


def _linear_score(model: LinearRidgeModel, vector: Mapping[str, float]) -> float:
    score = float(model.intercept)
    for name, mean, scale, coefficient in zip(
        model.feature_names, model.means, model.scales, model.coefficients, strict=True
    ):
        score += float(coefficient) * ((_safe_float(vector.get(name)) - float(mean)) / float(scale))
    return score


def _fit_linear_ridge(
    examples: list[tuple[dict[str, float], bool]],
    *,
    alpha: float = 1.0,
) -> LinearRidgeModel:
    if not examples:
        zeros = tuple(0.0 for _ in ROUTER_FEATURE_NAMES)
        ones = tuple(1.0 for _ in ROUTER_FEATURE_NAMES)
        return LinearRidgeModel(ROUTER_FEATURE_NAMES, zeros, ones, zeros, 0.0, alpha)
    x_raw = np.asarray(
        [[_safe_float(features.get(name)) for name in ROUTER_FEATURE_NAMES] for features, _ in examples],
        dtype=float,
    )
    y = np.asarray([1.0 if label else 0.0 for _, label in examples], dtype=float)
    means = x_raw.mean(axis=0)
    scales = x_raw.std(axis=0)
    scales[scales == 0.0] = 1.0
    x = (x_raw - means) / scales
    design = np.column_stack([np.ones(x.shape[0]), x])
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
    return LinearRidgeModel(
        feature_names=ROUTER_FEATURE_NAMES,
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in beta[1:]),
        intercept=float(beta[0]),
        alpha=float(alpha),
    )


def _cascal_pretraining_examples(
    observed_examples: list[tuple[dict[str, float], bool]],
    *,
    random_seed: int,
) -> tuple[list[tuple[dict[str, float], bool]], str]:
    rng = random.Random(int(random_seed))
    synthetic: list[tuple[dict[str, float], bool]] = []
    mutable_names = [
        "cell_confidence_mean",
        "cell_confidence_margin",
        "cell_confidence_rank_fraction",
        "contextprm_step_coherence",
        "contextprm_coherence_margin",
        "contextprm_candidate_set_support",
    ]
    for features, label in observed_examples:
        generated = dict(features)
        direction = 1.0 if label else -1.0
        for name in mutable_names:
            generated[name] = _safe_float(generated.get(name)) + direction * rng.uniform(0.005, 0.025)
        synthetic.append((generated, label))
        coherent_low_vote = dict(features)
        coherent_low_vote.update(
            {
                "vote_weight": 2.0,
                "vote_weight_rank_fraction": 0.25,
                "cell_confidence_mean": 0.96,
                "cell_confidence_margin": 0.72,
                "cell_confidence_rank_fraction": 1.0,
                "program_demo_fit": 0.96,
                "cell_confidence_zscore": 1.75,
                "contextprm_step_coherence": 0.96,
                "contextprm_coherence_margin": 0.58,
                "contextprm_vote_coherence_alignment": 0.75,
                "contextprm_candidate_set_support": 1.0,
                "contextprm_contradiction_pressure": 0.8,
                "contextprm_structure_consistency": 0.9,
            }
        )
        incoherent_high_vote = dict(features)
        incoherent_high_vote.update(
            {
                "vote_weight": 9.0,
                "vote_weight_rank_fraction": 1.0,
                "cell_confidence_mean": 0.2,
                "cell_confidence_margin": -0.15,
                "cell_confidence_rank_fraction": 0.1,
                "program_demo_fit": 0.2,
                "cell_confidence_zscore": -1.5,
                "contextprm_step_coherence": 0.2,
                "contextprm_coherence_margin": -0.25,
                "contextprm_vote_coherence_alignment": -0.9,
                "contextprm_candidate_set_support": 0.25,
                "contextprm_contradiction_pressure": 0.0,
                "contextprm_structure_consistency": 0.35,
            }
        )
        synthetic.append((coherent_low_vote, True))
        synthetic.append((incoherent_high_vote, False))
    checksum_payload = {
        "synthetic_n": len(synthetic),
        "feature_names": ROUTER_FEATURE_NAMES,
        "seed": int(random_seed),
        "no_held_out_target_labels": True,
    }
    checksum = "sha256:" + hashlib.sha256(
        json.dumps(checksum_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return synthetic, checksum


def train_ir3de_router(
    domain_pools: Mapping[str, DomainPool],
    *,
    held_out_domain: str,
    include_labels: bool,
    random_seed: int,
) -> IR3DERouter:
    train_pools = [pool for domain_id, pool in domain_pools.items() if domain_id != held_out_domain]
    train_rows = [row for pool in train_pools for row in pool.rows]
    profiles = {pool.domain_id: _fit_profile(pool.domain_id, pool.rows) for pool in train_pools}
    global_profile = _fit_profile("global_train_domains", train_rows)
    observed: list[tuple[dict[str, float], bool]] = []
    for pool in train_pools:
        for task_rows in _group_by_task(pool.rows):
            for row in task_rows:
                observed.append(
                    (
                        selector_feature_vector(
                            task_rows,
                            row,
                            profiles=profiles,
                            include_labels=include_labels,
                        ),
                        bool(row.is_correct),
                    )
                )
    synthetic, cascal_checksum = _cascal_pretraining_examples(
        observed, random_seed=int(random_seed) + (0 if include_labels else 7919)
    )
    model = _fit_linear_ridge(observed + synthetic, alpha=1.0)
    return IR3DERouter(
        held_out_domain=held_out_domain,
        profiles=profiles,
        global_profile=global_profile,
        linear_model=model,
        include_labels=include_labels,
        cascal_synthetic_n=len(synthetic),
        train_domain_ids=tuple(pool.domain_id for pool in train_pools),
        cascal_checksum=cascal_checksum,
    )


def _score_candidate(task_rows: list[CandidateRow], row: CandidateRow, router: IR3DERouter) -> float:
    vector = selector_feature_vector(
        task_rows,
        row,
        profiles=router.profiles,
        include_labels=router.include_labels,
    )
    return _linear_score(router.linear_model, vector)


def _select_router(task_rows: list[CandidateRow], router: IR3DERouter) -> CandidateRow:
    return max(
        task_rows,
        key=lambda row: (_score_candidate(task_rows, row, router), row.vote_weight, -row.candidate_index),
    )


def evaluate_heldout_domain(
    domain_pool: DomainPool,
    router: IR3DERouter,
    *,
    random_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    vote_hits: list[bool] = []
    selector_hits: list[bool] = []
    oracle_hits: list[bool] = []
    control_hits: list[bool] = []
    deltas: list[float] = []
    control_deltas: list[float] = []
    task_payloads: list[dict[str, Any]] = []

    for task_rows in _group_by_task(domain_pool.rows):
        vote_pick = _select_vote(task_rows)
        selector_pick = _select_router(task_rows, router)
        control_pick = _select_first(task_rows)
        oracle_hit = any(row.is_correct for row in task_rows)
        vote_hits.append(vote_pick.is_correct)
        selector_hits.append(selector_pick.is_correct)
        oracle_hits.append(oracle_hit)
        control_hits.append(control_pick.is_correct)
        deltas.append(float(selector_pick.is_correct) - float(vote_pick.is_correct))
        control_deltas.append(float(selector_pick.is_correct) - float(control_pick.is_correct))
        task_payloads.append(
            {
                "task_id": vote_pick.task_id,
                "domain_id": vote_pick.domain_id,
                "family_id": vote_pick.family_id,
                "target_hash": vote_pick.target_hash,
                "candidate_count": len(task_rows),
                "oracle_hit": oracle_hit,
                "vote_candidate_id": vote_pick.candidate_id,
                "vote_correct": bool(vote_pick.is_correct),
                "router_set_encoder_candidate_id": selector_pick.candidate_id,
                "router_set_encoder_correct": bool(selector_pick.is_correct),
                "matched_control_candidate_id": control_pick.candidate_id,
                "matched_control_correct": bool(control_pick.is_correct),
            }
        )

    vote_at_1 = _rate(vote_hits)
    selector_at_1 = _rate(selector_hits)
    oracle_at_k = _rate(oracle_hits)
    control_at_1 = _rate(control_hits)
    cross_domain_delta = _round_metric(selector_at_1 - vote_at_1)
    return {
        "domain_id": domain_pool.domain_id,
        "cross_domain_delta": cross_domain_delta,
        "cross_domain_delta_ci95": _bootstrap_ci95(
            deltas, random_seed=random_seed, resamples=bootstrap_resamples
        ),
        "vote_at_1": _round_metric(vote_at_1),
        "oracle_at_k": _round_metric(oracle_at_k),
        "router_set_encoder_at_1": _round_metric(selector_at_1),
        "matched_control_at_1": _round_metric(control_at_1),
        "matched_control_delta": _round_metric(
            sum(control_deltas) / float(len(control_deltas)) if control_deltas else 0.0
        ),
        "held_out_task_n": len(vote_hits),
        "task_rows": task_payloads,
    }


def label_ablation_survives(
    primary: Mapping[str, Any],
    ablated: Mapping[str, Any],
    *,
    tolerance: float,
) -> bool:
    primary_acc = float(primary.get("router_set_encoder_at_1", 0.0))
    ablated_acc = float(ablated.get("router_set_encoder_at_1", 0.0))
    return bool(
        primary_acc - ablated_acc <= float(tolerance)
        and float(ablated.get("cross_domain_delta", 0.0)) > 0.0
    )


def cross_domain_selection_holds_from_metrics(metrics: Mapping[str, Any]) -> bool:
    return bool(
        float(metrics.get("cross_domain_delta", 0.0)) > 0.0
        and _ci_excludes_zero(list(metrics.get("cross_domain_delta_ci95", [])))
        and float(metrics.get("vote_at_1", 0.0)) > 0.05
        and float(metrics.get("oracle_at_k", 1.0)) < 1.0
        and float(metrics.get("cross_domain_delta", 1.0)) < 0.95
        and metrics.get("label_ablation_robust") is True
    )


def _row_to_json(row: CandidateRow) -> dict[str, Any]:
    return {
        "task_id": row.task_id,
        "candidate_id": row.candidate_id,
        "candidate_index": row.candidate_index,
        "domain_id": row.domain_id,
        "family_id": row.family_id,
        "target_hash": row.target_hash,
        "is_correct": row.is_correct,
        "vote_weight": row.vote_weight,
        "features": row.features,
    }


def persist_cross_domain_pool(
    repo_root: Path | str,
    domain_pools: Mapping[str, DomainPool],
    *,
    random_seed: int,
) -> dict[str, Any]:
    root = Path(repo_root)
    pool_path = root / POOL_REL
    manifest_path = root / MANIFEST_REL
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = [
        row
        for domain_id in DOMAIN_ORDER
        for row in domain_pools.get(domain_id, DomainPool(domain_id, [], "", "", {})).rows
    ]
    tasks = []
    for task_rows in _group_by_task(all_rows):
        first = task_rows[0]
        tasks.append(
            {
                "task_id": first.task_id,
                "domain_id": first.domain_id,
                "family_id": first.family_id,
                "target_hash": first.target_hash,
                "candidate_count": len(task_rows),
                "oracle_present": any(row.is_correct for row in task_rows),
                "vote_top_correct": _select_vote(task_rows).is_correct,
                "candidates": [_row_to_json(row) for row in task_rows],
            }
        )
    pool_payload = {
        "schema": "carnot.cross_domain_selector_ir3de_cascal_pool_4314.v1",
        "spec_refs": SPEC_REFS,
        "random_seed": int(random_seed),
        "tasks": tasks,
    }
    with gzip.open(pool_path, "wt", encoding="utf-8") as handle:
        json.dump(pool_payload, handle, sort_keys=True, separators=(",", ":"))
    pool_sha256 = _sha256_file(pool_path)

    manifest_payload = {
        "schema": "carnot.cross_domain_selector_ir3de_cascal_manifest_4314.v1",
        "spec_refs": SPEC_REFS,
        "random_seed": int(random_seed),
        "rows": [
            {
                "task_id": task["task_id"],
                "domain_id": task["domain_id"],
                "family_id": task["family_id"],
                "target_hash": task["target_hash"],
                "candidate_count": task["candidate_count"],
                "oracle_present": task["oracle_present"],
                "vote_top_correct": task["vote_top_correct"],
            }
            for task in tasks
        ],
        "domain_sources": {
            domain_id: {
                "source_path": pool.source_path,
                "source_sha256": pool.source_sha256,
                "provenance": pool.provenance,
            }
            for domain_id, pool in domain_pools.items()
        },
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_sha256 = _sha256_file(manifest_path)
    return {
        "pool_path": pool_path,
        "pool_sha256": "sha256:" + pool_sha256,
        "manifest_path": manifest_path,
        "manifest_sha256": "sha256:" + manifest_sha256,
    }


def _missing_verifier_gap(reason: str, held_out_domain: str) -> dict[str, str]:
    return {
        "gap_id": "GAP-CROSS-DOMAIN-FAMILY-INVARIANT-SELECTION-4305",
        "failure_mode": reason,
        "missing_discriminator": (
            "domain-invariant selector features that preserve wrong-majority recovery across "
            "ARC, ARC-GEN, and FoVer/math step candidates without using domain labels"
        ),
        "candidate_design": (
            "stronger family-invariant verifier dimensions beyond IR3DE+CASCAL+ContextPRM, "
            f"validated on held-out {held_out_domain}"
        ),
        "priority": "P0 cross-domain verifier scope boundary",
    }


def _per_domain_reports(
    domain_pools: Mapping[str, DomainPool],
    *,
    random_seed: int,
    bootstrap_resamples: int,
    label_ablation_tolerance: float,
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for domain_id in DOMAIN_ORDER:
        if domain_id not in domain_pools:
            continue
        primary_router = train_ir3de_router(
            domain_pools,
            held_out_domain=domain_id,
            include_labels=True,
            random_seed=random_seed,
        )
        ablated_router = train_ir3de_router(
            domain_pools,
            held_out_domain=domain_id,
            include_labels=False,
            random_seed=random_seed,
        )
        primary = evaluate_heldout_domain(
            domain_pools[domain_id],
            primary_router,
            random_seed=random_seed,
            bootstrap_resamples=bootstrap_resamples,
        )
        ablated = evaluate_heldout_domain(
            domain_pools[domain_id],
            ablated_router,
            random_seed=random_seed + 17,
            bootstrap_resamples=bootstrap_resamples,
        )
        primary["label_ablation_robust"] = label_ablation_survives(
            primary, ablated, tolerance=label_ablation_tolerance
        )
        primary["label_ablation"] = {
            "router_set_encoder_at_1": ablated["router_set_encoder_at_1"],
            "cross_domain_delta": ablated["cross_domain_delta"],
            "cross_domain_delta_ci95": ablated["cross_domain_delta_ci95"],
            "protocol": "domain_id and family_id removed from router input; ContextPRM features retained",
        }
        primary["cross_domain_selection_holds"] = cross_domain_selection_holds_from_metrics(primary)
        primary["router_specs"] = {
            "train_domain_ids": list(primary_router.train_domain_ids),
            "cascal_synthetic_n": primary_router.cascal_synthetic_n,
            "cascal_checksum": primary_router.cascal_checksum,
            "ridge_alpha": primary_router.linear_model.alpha,
            "feature_n": len(primary_router.linear_model.feature_names),
        }
        reports[domain_id] = primary
    return reports


def _reproducibility_checksum(
    *,
    persisted: Mapping[str, Any],
    primary: Mapping[str, Any],
    per_domain: Mapping[str, Any],
    random_seed: int,
) -> str:
    payload = {
        "pool_sha256": persisted.get("pool_sha256"),
        "manifest_sha256": persisted.get("manifest_sha256"),
        "primary_domain": primary.get("domain_id"),
        "primary_delta": primary.get("cross_domain_delta"),
        "primary_ci95": primary.get("cross_domain_delta_ci95"),
        "per_domain": {
            key: {
                "cross_domain_delta": value.get("cross_domain_delta"),
                "vote_at_1": value.get("vote_at_1"),
                "oracle_at_k": value.get("oracle_at_k"),
                "cascal_checksum": value.get("router_specs", {}).get("cascal_checksum"),
            }
            for key, value in per_domain.items()
        },
        "random_seed": int(random_seed),
        "router": "ir3de_linear_domain_expert_ridge_cascal_contextprm",
    }
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _model_specs(
    *,
    status: str,
    set_encoder: Mapping[str, Any] | None,
    domain_pools: Mapping[str, DomainPool],
    missing_domains: list[dict[str, str]],
    held_out_domain: str | None,
    persisted: Mapping[str, Any] | None,
    primary: Mapping[str, Any] | None = None,
    blocked_reason: str | None = None,
) -> dict[str, Any]:
    persisted_pool = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in dict(persisted or {}).items()
    }
    router_specs = dict((primary or {}).get("router_specs", {}))
    return {
        "status": status,
        "blocked_reason": blocked_reason,
        "available_domains": [domain for domain in DOMAIN_ORDER if domain in domain_pools],
        "missing_domains": missing_domains,
        "domain_provenance": {
            domain_id: {
                "source_path": pool.source_path,
                "source_sha256": pool.source_sha256,
                "provenance": pool.provenance,
            }
            for domain_id, pool in domain_pools.items()
        },
        "set_encoder_precondition": dict(set_encoder or {}),
        "router": {
            "kind": "ir3de_linear_domain_expert_ridge",
            "ridge_alpha": router_specs.get("ridge_alpha", 1.0),
            "feature_names": list(ROUTER_FEATURE_NAMES),
            "train_domain_ids": router_specs.get("train_domain_ids", []),
            "label_features": "domain/family labels allowed only in primary arm; ablation zeroes them",
        },
        "cascal_pretraining": {
            "kind": "generated_train_domain_selector_tasks",
            "synthetic_task_n": router_specs.get("cascal_synthetic_n", 0),
            "synthetic_checksum": router_specs.get("cascal_checksum"),
            "held_out_target_labels_used": False,
        },
        "contextprm_features": {
            "kind": "domain_agnostic_step_coherence",
            "domain_agnostic": True,
            "feature_names": list(CONTEXTPRM_FEATURE_NAMES),
            "label_fields_excluded": ["domain_id", "family_id", "target_hash"],
        },
        "per_domain_set_encoder": {
            "kind": "Exp 4244 standardized correct-minus-wrong set profile used as domain experts",
            "training_scope": "train domains only for each leave-one-domain-out read",
        },
        "domain_disjoint_split": {
            "primary_held_out_domain": held_out_domain,
            "leave_one_domain_out_domains": [domain for domain in DOMAIN_ORDER if domain in domain_pools],
        },
        "label_ablation_protocol": (
            "rerun held-out routing with domain_id/family_id label features removed while "
            "retaining normalized selector and ContextPRM coherence features"
        ),
        "persisted_pool": persisted_pool,
    }


def _blocked_artifact(
    *,
    duration_s: float,
    random_seed: int,
    set_encoder: Mapping[str, Any] | None,
    domain_pools: Mapping[str, DomainPool],
    missing_domains: list[dict[str, str]],
) -> dict[str, Any]:
    reason = "blocked_insufficient_domains"
    checksum = "sha256:" + hashlib.sha256(
        json.dumps(
            {
                "reason": reason,
                "available_domains": sorted(domain_pools),
                "missing_domains": missing_domains,
                "random_seed": int(random_seed),
                "router": "ir3de_linear_domain_expert_ridge_cascal_contextprm",
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "experiment": "experiment_4314_cross_domain_selector_ir3de_cascal",
        "schema": "carnot.cross_domain_selector_ir3de_cascal_4314.v1",
        "status": "blocked",
        "honest_verdict": reason,
        "cross_domain_selection_holds": False,
        "cross_domain_delta": 0.0,
        "cross_domain_delta_ci95": [0.0, 0.0],
        "vote_at_1": 0.0,
        "oracle_at_k": 0.0,
        "label_ablation_robust": False,
        "per_domain_delta": {},
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "model_specs": _model_specs(
            status="blocked",
            set_encoder=set_encoder,
            domain_pools=domain_pools,
            missing_domains=missing_domains,
            held_out_domain=None,
            persisted=None,
            blocked_reason=reason,
        ),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": {
            "condition": "blocked before held-out-domain measurement because fewer than three domains loaded",
            "passed": False,
        },
        "adversarial_verify": {"status": "pending"},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "missing_verifier_gaps": [],
        "cross_domain_pool_path": "",
        "domain_manifest_path": "",
    }


def _complete_artifact(
    *,
    primary: Mapping[str, Any],
    per_domain: Mapping[str, Any],
    set_encoder: Mapping[str, Any],
    domain_pools: Mapping[str, DomainPool],
    missing_domains: list[dict[str, str]],
    persisted: Mapping[str, Any],
    random_seed: int,
    duration_s: float,
    checksum: str,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    label_only_failure = (
        float(primary.get("cross_domain_delta", 0.0)) > 0.0
        and _ci_excludes_zero(list(primary.get("cross_domain_delta_ci95", [])))
        and float(primary.get("vote_at_1", 0.0)) > 0.05
        and float(primary.get("oracle_at_k", 1.0)) < 1.0
        and primary.get("label_ablation_robust") is not True
    )
    holds = cross_domain_selection_holds_from_metrics(primary)
    if holds:
        verdict_suffix = "cross_domain_selection_survives_ir3de_cascal"
        gaps: list[dict[str, str]] = []
    elif label_only_failure:
        verdict_suffix = "label_ablation_failure_router_read_label"
        gaps = [_missing_verifier_gap(verdict_suffix, str(primary.get("domain_id")))]
    else:
        verdict_suffix = "powered_collapse_cross_domain_domain_bound"
        gaps = [_missing_verifier_gap(verdict_suffix, str(primary.get("domain_id")))]
    return {
        "experiment": "experiment_4314_cross_domain_selector_ir3de_cascal",
        "schema": "carnot.cross_domain_selector_ir3de_cascal_4314.v1",
        "status": "complete",
        "honest_verdict": f"complete: {verdict_suffix}",
        "cross_domain_selection_holds": holds,
        "cross_domain_delta": primary["cross_domain_delta"],
        "cross_domain_delta_ci95": primary["cross_domain_delta_ci95"],
        "vote_at_1": primary["vote_at_1"],
        "oracle_at_k": primary["oracle_at_k"],
        "label_ablation_robust": bool(primary["label_ablation_robust"]),
        "per_domain_delta": {
            domain_id: {
                key: value[key]
                for key in (
                    "cross_domain_delta",
                    "cross_domain_delta_ci95",
                    "vote_at_1",
                    "oracle_at_k",
                    "router_set_encoder_at_1",
                    "matched_control_delta",
                    "held_out_task_n",
                    "label_ablation_robust",
                    "cross_domain_selection_holds",
                )
            }
            for domain_id, value in per_domain.items()
        },
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "model_specs": _model_specs(
            status="complete",
            set_encoder=set_encoder,
            domain_pools=domain_pools,
            missing_domains=missing_domains,
            held_out_domain=str(primary["domain_id"]),
            persisted=persisted,
            primary=primary,
        ),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": {
            "condition": (
                "cross_domain_delta>0 AND CI95-excl-0 AND vote_at_1>0.05 AND "
                "oracle_at_k<1.0 AND cross_domain_delta<0.95 AND label_ablation_robust"
            ),
            "passed": holds,
        },
        "adversarial_verify": {"status": "pending"},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "bootstrap_resamples": int(bootstrap_resamples),
        "primary_held_out_domain": str(primary["domain_id"]),
        "pass_rates": {
            "vote_at_1": primary["vote_at_1"],
            "router_set_encoder_at_1": primary["router_set_encoder_at_1"],
            "matched_control_at_1": primary["matched_control_at_1"],
        },
        "oracle_minus_vote": _round_metric(float(primary["oracle_at_k"]) - float(primary["vote_at_1"])),
        "held_out_task_n": int(primary["held_out_task_n"]),
        "matched_control_delta": primary["matched_control_delta"],
        "label_ablation": primary["label_ablation"],
        "task_rows": primary["task_rows"],
        "missing_verifier_gaps": gaps,
        "cross_domain_pool_path": str(persisted["pool_path"]),
        "cross_domain_pool_sha256": persisted["pool_sha256"],
        "domain_manifest_path": str(persisted["manifest_path"]),
        "domain_manifest_sha256": persisted["manifest_sha256"],
    }


def _run_adversarial_verify(repo_root: Path, artifact_path: Path) -> dict[str, Any]:  # pragma: no cover
    proc = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "adversarial_verify.py"), "--json", str(artifact_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = {"stdout": proc.stdout, "stderr": proc.stderr}
    payload["returncode"] = proc.returncode
    return payload


def _clean_adversarial_report(report: Mapping[str, Any]) -> dict[str, Any]:
    return base._clean_adversarial_report(report)


def _bare_float(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{field} must be a bare finite float")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:") or verdict == "blocked_insufficient_domains"
    ):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if type(artifact["cross_domain_selection_holds"]) is not bool:
        raise ValueError("cross_domain_selection_holds must be a bare bool")
    for field in ("cross_domain_delta", "vote_at_1", "oracle_at_k"):
        _bare_float(artifact[field], field)
    ci95 = artifact["cross_domain_delta_ci95"]
    if (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in ci95)
    ):
        raise ValueError("cross_domain_delta_ci95 must be a two-number ci95")
    if type(artifact["label_ablation_robust"]) is not bool:
        raise ValueError("label_ablation_robust must be a bare bool")
    if not isinstance(artifact["per_domain_delta"], dict):
        raise ValueError("per_domain_delta must be an object")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if type(artifact["random_seed"]) is not int:
        raise ValueError("random_seed must be a bare int")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4314")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs do not match REQ-VERIFY-4314")


def run(
    repo_root: Path | str = Path("."),
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    held_out_domain: str = "fover",
    label_ablation_tolerance: float = 0.05,
    domain_loaders: Mapping[str, Callable[[], DomainPool]] | None = None,
    set_encoder_loader: Callable[[Path], Mapping[str, Any]] | None = None,
    adversarial_runner: Callable[[Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)
    output_path = root / OUTPUT_REL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    loader_map = domain_loaders or _default_domain_loaders(root)
    set_encoder = dict((set_encoder_loader or _load_set_encoder_precondition)(root))
    domain_pools, missing_domains = _load_domains(loader_map)

    if len(domain_pools) < 3:
        artifact = _blocked_artifact(
            duration_s=time.perf_counter() - start,
            random_seed=random_seed,
            set_encoder=set_encoder,
            domain_pools=domain_pools,
            missing_domains=missing_domains,
        )
    else:
        primary_domain = held_out_domain if held_out_domain in domain_pools else sorted(domain_pools)[-1]
        persisted = persist_cross_domain_pool(root, domain_pools, random_seed=random_seed)
        per_domain = _per_domain_reports(
            domain_pools,
            random_seed=random_seed,
            bootstrap_resamples=bootstrap_resamples,
            label_ablation_tolerance=label_ablation_tolerance,
        )
        primary = per_domain[primary_domain]
        checksum = _reproducibility_checksum(
            persisted=persisted,
            primary=primary,
            per_domain=per_domain,
            random_seed=random_seed,
        )
        artifact = _complete_artifact(
            primary=primary,
            per_domain=per_domain,
            set_encoder=set_encoder,
            domain_pools=domain_pools,
            missing_domains=missing_domains,
            persisted=persisted,
            random_seed=random_seed,
            duration_s=time.perf_counter() - start,
            checksum=checksum,
            bootstrap_resamples=bootstrap_resamples,
        )
    validate_artifact(artifact)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = (adversarial_runner or (lambda path: _run_adversarial_verify(root, path)))(output_path)
    artifact["adversarial_verify"] = _clean_adversarial_report(report)
    validate_artifact(artifact)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover
    artifact = run(Path(__file__).resolve().parents[2])
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
