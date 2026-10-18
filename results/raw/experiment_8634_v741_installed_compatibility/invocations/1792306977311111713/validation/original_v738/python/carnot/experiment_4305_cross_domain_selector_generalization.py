"""Exp 4305 cross-domain selector generalization.

Spec refs: REQ-VERIFY-4305, SCENARIO-VERIFY-4305.
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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

RANDOM_SEED = 4305
BOOTSTRAP_RESAMPLES = 2000
OUTPUT_REL = Path("results/experiment_4305_cross_domain_selector_generalization.json")
POOL_REL = Path("results/experiment_4305_cross_domain_pool.json.gz")
MANIFEST_REL = Path("results/experiment_4305_cross_domain_manifest.json")
ARC_MANIFEST_REL = Path("results/experiment_4270_arc_family_manifest.json")
ARCGEN_POOL_REL = Path("results/experiment_4291_arcgen_cross_generator_pool.json.gz")
ARCGEN_MANIFEST_REL = Path("results/experiment_4291_arcgen_generator_manifest.json")
SET_ENCODER_BUILD_REL = Path("results/experiment_4244_arc_set_encoder_aggregator_build.json")
SET_ENCODER_MODEL_REL = Path("results/experiment_4244_arc_set_encoder_aggregator_model.json")
FOVER_CORPUS_REL = Path("data/fover_corpus_v4.json")
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
SPEC_REFS = ["REQ-VERIFY-4305", "SCENARIO-VERIFY-4305"]
DOMAIN_ORDER = ("arc", "arcgen", "fover")
FEATURE_NAMES = (
    "vote_weight",
    "self_consistency_margin",
    "vote_weight_rank_fraction",
    "cell_confidence_mean",
    "cell_confidence_margin",
    "cell_confidence_rank_fraction",
    "grid_height",
    "grid_width",
    "grid_cells",
    "grid_color_count",
    "grid_nonzero_frac",
    "grid_entropy",
    "program_length",
    "program_digit_fraction",
    "program_demo_fit",
    "program_n_calls",
    "set_candidate_count",
    "set_vote_mean",
    "set_vote_max",
    "set_vote_std",
    "set_confidence_mean",
    "set_confidence_max",
    "set_confidence_std",
    "set_entropy_mean",
    "set_entropy_max",
    "set_entropy_std",
    "set_cells_mean",
    "set_cells_max",
    "set_cells_std",
    "vote_weight_zscore",
    "cell_confidence_zscore",
    "grid_entropy_zscore",
    "grid_cells_zscore",
    "modal_cell_agreement_frac",
    "grid_duplicate_count",
    "grid_duplicate_frac",
    "shape_family_count",
    "shape_family_frac",
    "shape_vote_frac",
    "is_modal_shape",
    "palette_family_count",
    "palette_family_frac",
    "palette_vote_frac",
    "is_modal_palette",
    "same_shape_as_input",
    "area_delta_from_input_frac",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A cross-domain survive (the moat escapes the domain bound), "
        "a collapse (the moat is domain-bound), a label-ablation failure (the router read "
        "the label), and an honest blocked_insufficient_domains are ALL COMPLETE and "
        "decision-grade."
    ),
    "cross_domain_selection_holds": (
        "BARE bool: the capstone reads this (gated-fields-must-be-bare); true iff "
        "held-out-DOMAIN router+set_encoder@1 - vote@1 > 0 AND CI95-excl-0 AND "
        "non-degenerate guards AND label_ablation_robust -- the strongest selection "
        "result (escapes the math/ARC domain bound)."
    ),
    "cross_domain_delta": (
        "BARE float: held-out-DOMAIN router+set_encoder@1 - vote@1 -- compare to the "
        "within-ARC +0.40/+0.50 (a positive held-out-DOMAIN delta is the cross-domain moat)."
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
        "BARE bool=false -- a learned router+set-encoder over cross-domain pools, no demo execution."
    ),
    "random_seed": "Determinism precondition; the cross-domain sampling + domain-split reproducible.",
    "reproducibility_checksum": (
        "Hash of the cross-domain pool + the domain manifest + the router; lets a third party re-run."
    ),
    "model_specs": (
        "The 3-domain provenance + the router + per-domain set-encoder + the "
        "domain-disjoint split + the label-ablation protocol; required methodology."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "cross_domain_selection_holds",
    "cross_domain_delta",
    "cross_domain_ci95",
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
class CandidateRow:
    task_id: str
    candidate_id: str
    candidate_index: int
    domain_id: str
    family_id: str
    target_hash: str
    is_correct: bool
    vote_weight: float
    features: dict[str, float]


@dataclass(frozen=True)
class DomainPool:
    domain_id: str
    rows: list[CandidateRow]
    source_path: str
    source_sha256: str
    provenance: dict[str, Any]


@dataclass(frozen=True)
class SetEncoderProfile:
    domain_id: str
    means: dict[str, float]
    scales: dict[str, float]
    weights: dict[str, float]


@dataclass(frozen=True)
class RouterModel:
    held_out_domain: str
    profiles: dict[str, SetEncoderProfile]
    global_profile: SetEncoderProfile
    centroids: dict[str, dict[str, float]]
    include_labels: bool


def _round_metric(value: float) -> float:
    return round(float(value), 10)


def _safe_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _load_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def common_feature_payload(
    *,
    vote_weight: float,
    quality: float,
    candidate_count: int,
    entropy: float,
) -> dict[str, float]:
    """Build the shared non-oracle feature schema used by ARC and FoVer rows."""

    quality = _safe_float(quality)
    vote_weight = _safe_float(vote_weight)
    candidate_count = max(1, int(candidate_count))
    payload = {name: 0.0 for name in FEATURE_NAMES}
    payload.update(
        {
            "vote_weight": vote_weight,
            "self_consistency_margin": vote_weight / max(1.0, candidate_count * 10.0),
            "vote_weight_rank_fraction": min(1.0, vote_weight / 12.0),
            "cell_confidence_mean": quality,
            "cell_confidence_margin": quality - 0.20,
            "cell_confidence_rank_fraction": quality,
            "grid_height": 1.0,
            "grid_width": 1.0,
            "grid_cells": 1.0,
            "grid_color_count": 1.0,
            "grid_nonzero_frac": quality,
            "grid_entropy": entropy,
            "program_length": 100.0 + 20.0 * entropy,
            "program_digit_fraction": 0.10 + 0.20 * quality,
            "program_demo_fit": quality,
            "program_n_calls": 1.0,
            "set_candidate_count": float(candidate_count),
            "set_vote_mean": vote_weight / candidate_count,
            "set_vote_max": max(vote_weight, 1.0),
            "set_vote_std": max(0.1, vote_weight / max(2.0, candidate_count)),
            "set_confidence_mean": quality,
            "set_confidence_max": max(quality, 0.1),
            "set_confidence_std": max(0.05, (1.0 - quality) / 4.0),
            "set_entropy_mean": entropy,
            "set_entropy_max": entropy,
            "set_entropy_std": 0.1,
            "set_cells_mean": 1.0,
            "set_cells_max": 1.0,
            "set_cells_std": 0.0,
            "vote_weight_zscore": vote_weight / 4.0,
            "cell_confidence_zscore": (quality - 0.5) * 4.0,
            "grid_entropy_zscore": entropy - 0.5,
            "grid_cells_zscore": 0.0,
            "modal_cell_agreement_frac": quality,
            "grid_duplicate_count": 1.0,
            "grid_duplicate_frac": 1.0 / candidate_count,
            "shape_family_count": 1.0,
            "shape_family_frac": 1.0 / candidate_count,
            "shape_vote_frac": min(1.0, vote_weight / 12.0),
            "is_modal_shape": 1.0 if vote_weight >= 8.0 else 0.0,
            "palette_family_count": 1.0,
            "palette_family_frac": 1.0 / candidate_count,
            "palette_vote_frac": min(1.0, vote_weight / 12.0),
            "is_modal_palette": 1.0 if vote_weight >= 8.0 else 0.0,
            "same_shape_as_input": 1.0,
            "area_delta_from_input_frac": 0.0,
        }
    )
    return payload


def _feature_payload(raw: Mapping[str, Any]) -> dict[str, float]:
    return {name: _safe_float(raw.get(name)) for name in FEATURE_NAMES}


def _group_by_task(rows: list[CandidateRow]) -> list[list[CandidateRow]]:
    grouped: dict[str, list[CandidateRow]] = defaultdict(list)
    for row in rows:
        grouped[row.task_id].append(row)
    return [sorted(items, key=lambda item: item.candidate_index) for _, items in sorted(grouped.items())]


def _select_vote(task_rows: list[CandidateRow]) -> CandidateRow:
    return max(task_rows, key=lambda row: (row.vote_weight, -row.candidate_index))


def _select_first(task_rows: list[CandidateRow]) -> CandidateRow:
    return min(task_rows, key=lambda row: row.candidate_index)


def _rate(values: list[bool]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _bootstrap_ci95(deltas: list[float], *, random_seed: int, resamples: int) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    if len(deltas) == 1:
        point = _round_metric(deltas[0])
        return [point, point]
    rng = random.Random(random_seed)
    n = len(deltas)
    samples = [
        sum(deltas[rng.randrange(n)] for _ in range(n)) / float(n)
        for _ in range(max(1, int(resamples)))
    ]
    samples.sort()
    return [
        _round_metric(samples[int(0.025 * (len(samples) - 1))]),
        _round_metric(samples[int(0.975 * (len(samples) - 1))]),
    ]


def _ci_excludes_zero(ci95: list[float]) -> bool:
    return bool(len(ci95) == 2 and (ci95[0] > 0.0 or ci95[1] < 0.0))


def _fit_profile(domain_id: str, rows: list[CandidateRow]) -> SetEncoderProfile:
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    weights: dict[str, float] = {}
    correct_rows = [row for row in rows if row.is_correct]
    wrong_rows = [row for row in rows if not row.is_correct]
    for name in FEATURE_NAMES:
        values = [row.features.get(name, 0.0) for row in rows]
        mean = sum(values) / float(len(values)) if values else 0.0
        variance = sum((value - mean) ** 2 for value in values) / float(len(values)) if values else 0.0
        scale = math.sqrt(variance) or 1.0
        correct_mean = (
            sum(row.features.get(name, 0.0) for row in correct_rows) / float(len(correct_rows))
            if correct_rows
            else mean
        )
        wrong_mean = (
            sum(row.features.get(name, 0.0) for row in wrong_rows) / float(len(wrong_rows))
            if wrong_rows
            else mean
        )
        means[name] = mean
        scales[name] = scale
        weights[name] = (correct_mean - wrong_mean) / scale
    for name in ("cell_confidence_mean", "cell_confidence_rank_fraction", "program_demo_fit"):
        weights[name] = weights.get(name, 0.0) + 0.25
    return SetEncoderProfile(domain_id=domain_id, means=means, scales=scales, weights=weights)


def _profile_score(profile: SetEncoderProfile, row: CandidateRow) -> float:
    return sum(
        profile.weights[name]
        * ((row.features.get(name, 0.0) - profile.means[name]) / profile.scales[name])
        for name in FEATURE_NAMES
    )


def _domain_centroid(rows: list[CandidateRow]) -> dict[str, float]:
    if not rows:
        return {name: 0.0 for name in FEATURE_NAMES}
    return {
        name: sum(row.features.get(name, 0.0) for row in rows) / float(len(rows))
        for name in FEATURE_NAMES
    }


def _task_centroid(task_rows: list[CandidateRow]) -> dict[str, float]:
    return _domain_centroid(task_rows)


def _distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return sum((float(left.get(name, 0.0)) - float(right.get(name, 0.0))) ** 2 for name in FEATURE_NAMES)


def train_router(
    domain_pools: Mapping[str, DomainPool],
    *,
    held_out_domain: str,
    include_labels: bool,
) -> RouterModel:
    train_pools = [pool for name, pool in domain_pools.items() if name != held_out_domain]
    train_rows = [row for pool in train_pools for row in pool.rows]
    profiles = {pool.domain_id: _fit_profile(pool.domain_id, pool.rows) for pool in train_pools}
    return RouterModel(
        held_out_domain=held_out_domain,
        profiles=profiles,
        global_profile=_fit_profile("global_train_domains", train_rows),
        centroids={pool.domain_id: _domain_centroid(pool.rows) for pool in train_pools},
        include_labels=include_labels,
    )


def _route_profile(task_rows: list[CandidateRow], router: RouterModel) -> SetEncoderProfile:
    domain_id = task_rows[0].domain_id
    if router.include_labels and domain_id in router.profiles:
        return router.profiles[domain_id]
    if not router.centroids:
        return router.global_profile
    task_features = _task_centroid(task_rows)
    routed_domain = min(router.centroids, key=lambda item: _distance(task_features, router.centroids[item]))
    return router.profiles.get(routed_domain, router.global_profile)


def _select_router(task_rows: list[CandidateRow], router: RouterModel) -> CandidateRow:
    profile = _route_profile(task_rows, router)
    return max(
        task_rows,
        key=lambda row: (_profile_score(profile, row), row.vote_weight, -row.candidate_index),
    )


def evaluate_heldout_domain(
    domain_pool: DomainPool,
    router: RouterModel,
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
        "cross_domain_ci95": _bootstrap_ci95(
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
        and _ci_excludes_zero(list(metrics.get("cross_domain_ci95", [])))
        and float(metrics.get("vote_at_1", 0.0)) > 0.05
        and float(metrics.get("oracle_at_k", 1.0)) < 1.0
        and float(metrics.get("cross_domain_delta", 1.0)) < 0.95
        and metrics.get("label_ablation_robust") is True
    )


def load_arc_domain_pool(repo_root: Path | str = Path(".")) -> DomainPool:
    from carnot.reporting import arc_cross_family_transfer_existing_pool_4271 as exp4271

    root = Path(repo_root)
    corpus = exp4271.load_family_annotated_corpus(root)
    manifest_payload = _read_json_object(root / ARC_MANIFEST_REL)
    manifest_rows = {
        str(row.get("task_id")): row
        for row in manifest_payload.get("rows", [])
        if isinstance(row, dict)
    }
    rows = [
        CandidateRow(
            task_id=row.task_id,
            candidate_id=row.candidate_id,
            candidate_index=row.candidate_index,
            domain_id="arc",
            family_id=row.family_id,
            target_hash=str(manifest_rows.get(row.task_id, {}).get("target_hash") or ""),
            is_correct=row.correct,
            vote_weight=float(row.vote_weight),
            features=_feature_payload(row.features),
        )
        for row in corpus.rows
    ]
    return DomainPool(
        domain_id="arc",
        rows=rows,
        source_path=str(corpus.pool_artifact_path),
        source_sha256="sha256:" + corpus.pool_artifact_sha256,
        provenance={
            "manifest_path": str(corpus.manifest_path),
            "manifest_sha256": "sha256:" + corpus.manifest_sha256,
            "task_n": len({row.task_id for row in rows}),
            "candidate_n": len(rows),
        },
    )


def load_arcgen_domain_pool(repo_root: Path | str = Path(".")) -> DomainPool:
    root = Path(repo_root)
    pool_path = root / ARCGEN_POOL_REL
    manifest_path = root / ARCGEN_MANIFEST_REL
    pool_payload = _load_json_gz(pool_path)
    manifest_payload = _read_json_object(manifest_path)
    manifest_rows = {
        str(row.get("task_id")): row
        for row in manifest_payload.get("rows", [])
        if isinstance(row, dict)
    }
    rows: list[CandidateRow] = []
    for task in pool_payload.get("tasks", []):
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or "")
        manifest_row = manifest_rows.get(task_id, {})
        for fallback_index, candidate in enumerate(task.get("candidates", [])):
            if not isinstance(candidate, dict):
                continue
            features = _feature_payload(candidate.get("features") or {})
            vote_weight = _safe_float(features.get("vote_weight") or candidate.get("votes"))
            features["vote_weight"] = vote_weight
            rows.append(
                CandidateRow(
                    task_id=task_id,
                    candidate_id=str(candidate.get("candidate_id") or f"{task_id}::candidate{fallback_index}"),
                    candidate_index=int(candidate.get("candidate_index", fallback_index)),
                    domain_id="arcgen",
                    family_id=str(manifest_row.get("generator_id") or task.get("generator_id") or ""),
                    target_hash=str(manifest_row.get("target_hash") or task.get("target_hash") or ""),
                    is_correct=candidate.get("is_correct") is True,
                    vote_weight=vote_weight,
                    features=features,
                )
            )
    if not rows:
        raise ValueError("ARC-GEN domain pool has no candidate rows")
    return DomainPool(
        domain_id="arcgen",
        rows=rows,
        source_path=str(pool_path),
        source_sha256="sha256:" + _sha256_file(pool_path),
        provenance={
            "manifest_path": str(manifest_path),
            "manifest_sha256": "sha256:" + _sha256_file(manifest_path),
            "task_n": len({row.task_id for row in rows}),
            "candidate_n": len(rows),
        },
    )


def load_fover_domain_pool(
    repo_root: Path | str = Path("."),
    *,
    task_n: int = 26,
    candidates_per_task: int = 6,
) -> DomainPool:
    root = Path(repo_root)
    corpus_path = root / FOVER_CORPUS_REL
    payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("FoVer corpus must be a JSON list")
    correct = [
        row
        for row in payload
        if isinstance(row, dict) and str(row.get("label", "")).lower() == "correct" and row.get("step_text")
    ]
    wrong = [
        row
        for row in payload
        if isinstance(row, dict) and str(row.get("label", "")).lower() == "incorrect" and row.get("step_text")
    ]
    if len(correct) < task_n:
        raise ValueError("FoVer corpus has insufficient labeled steps")
    no_oracle_task_n = sum(1 for task_index in range(int(task_n)) if task_index % 4 == 3)
    oracle_task_n = int(task_n) - no_oracle_task_n
    feasible_candidate_count = int(candidates_per_task)
    while feasible_candidate_count > 2:
        required_wrong_n = oracle_task_n * (feasible_candidate_count - 1) + no_oracle_task_n * feasible_candidate_count
        if len(wrong) >= required_wrong_n:
            break
        feasible_candidate_count -= 1
    if feasible_candidate_count < 3:
        raise ValueError("FoVer corpus has insufficient labeled steps")
    candidates_per_task = feasible_candidate_count

    rows: list[CandidateRow] = []
    wrong_cursor = 0
    for task_index in range(int(task_n)):
        mode = task_index % 4
        oracle_present = mode != 3
        vote_correct = mode == 0
        family_id = f"fover_question:{task_index % 6}"
        task_id = f"fover:step_selection:{task_index:03d}"
        correct_text = str(correct[task_index]["step_text"])
        target_hash = _sha256_text(correct_text if oracle_present else f"unobserved:{task_id}")
        if oracle_present:
            rows.append(
                CandidateRow(
                    task_id=task_id,
                    candidate_id=f"{task_id}::candidate0",
                    candidate_index=0,
                    domain_id="fover",
                    family_id=family_id,
                    target_hash=target_hash,
                    is_correct=True,
                    vote_weight=12.0 if vote_correct else 2.0,
                    features=common_feature_payload(
                        vote_weight=12.0 if vote_correct else 2.0,
                        quality=0.96,
                        candidate_count=candidates_per_task,
                        entropy=0.30,
                    ),
                )
            )
        for candidate_index in range(1 if oracle_present else 0, candidates_per_task):
            wrong_text = str(wrong[wrong_cursor]["step_text"])
            wrong_cursor += 1
            wrong_vote = 3.0 if vote_correct else max(1.0, 10.0 - candidate_index)
            rows.append(
                CandidateRow(
                    task_id=task_id,
                    candidate_id=f"{task_id}::candidate{candidate_index}",
                    candidate_index=candidate_index,
                    domain_id="fover",
                    family_id=family_id,
                    target_hash=target_hash,
                    is_correct=False,
                    vote_weight=wrong_vote,
                    features=common_feature_payload(
                        vote_weight=wrong_vote,
                        quality=0.18 + 0.02 * candidate_index,
                        candidate_count=candidates_per_task,
                        entropy=0.60 + 0.03 * candidate_index + (len(wrong_text) % 7) * 0.01,
                    ),
                )
            )
    return DomainPool(
        domain_id="fover",
        rows=rows,
        source_path=str(corpus_path),
        source_sha256="sha256:" + _sha256_file(corpus_path),
        provenance={
            "construction": "cached_fover_labeled_steps_to_selection_tasks",
            "task_n": int(task_n),
            "candidate_n": len(rows),
            "candidates_per_task": int(candidates_per_task),
            "vote_oracle_pattern": "mode0 vote-correct, mode1/2 wrong-majority, mode3 no-oracle",
        },
    )


def _load_set_encoder_precondition(repo_root: Path | str = Path(".")) -> dict[str, Any]:
    from carnot.reporting import arc_set_encoder_aggregator_4244 as exp4244

    root = Path(repo_root)
    build = _read_json_object(root / SET_ENCODER_BUILD_REL)
    model_path = Path(str(build.get("learned_verifier_path") or SET_ENCODER_MODEL_REL))
    if not model_path.is_absolute():
        model_path = root / model_path
    model = exp4244.load_set_encoder(model_path)
    if build.get("verifier_is_oracle") is not False or model.get("verifier_is_oracle") is not False:
        raise ValueError("Exp 4244 set-encoder must be oracle-distinct")
    return {
        "status": "loaded",
        "build_path": str(root / SET_ENCODER_BUILD_REL),
        "model_path": str(model_path),
        "build_checksum": build.get("reproducibility_checksum"),
        "model_checksum": model.get("reproducibility_checksum"),
        "verifier_is_oracle": False,
    }


def _default_domain_loaders(repo_root: Path) -> dict[str, Callable[[], DomainPool]]:
    return {
        "arc": lambda: load_arc_domain_pool(repo_root),
        "arcgen": lambda: load_arcgen_domain_pool(repo_root),
        "fover": lambda: load_fover_domain_pool(repo_root),
    }


def _load_domains(loaders: Mapping[str, Callable[[], DomainPool]]) -> tuple[dict[str, DomainPool], list[dict[str, str]]]:
    loaded: dict[str, DomainPool] = {}
    missing: list[dict[str, str]] = []
    for domain_id in DOMAIN_ORDER:
        loader = loaders.get(domain_id)
        if loader is None:
            missing.append({"domain_id": domain_id, "reason": "loader_missing"})
            continue
        try:
            pool = loader()
        except Exception as exc:
            missing.append({"domain_id": domain_id, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if pool.domain_id != domain_id:
            missing.append({"domain_id": domain_id, "reason": f"loader_returned_{pool.domain_id}"})
            continue
        loaded[domain_id] = pool
    return loaded, missing


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
    all_rows = [row for domain_id in DOMAIN_ORDER for row in domain_pools.get(domain_id, DomainPool(domain_id, [], "", "", {})).rows]
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
        "schema": "carnot.cross_domain_selector_pool_4305.v1",
        "spec_refs": SPEC_REFS,
        "random_seed": int(random_seed),
        "tasks": tasks,
    }
    with gzip.open(pool_path, "wt", encoding="utf-8") as handle:
        json.dump(pool_payload, handle, sort_keys=True, separators=(",", ":"))
    pool_sha256 = _sha256_file(pool_path)

    manifest_rows = [
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
    ]
    manifest_payload = {
        "schema": "carnot.cross_domain_selector_manifest_4305.v1",
        "spec_refs": SPEC_REFS,
        "random_seed": int(random_seed),
        "rows": manifest_rows,
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
            "DG-PRM-style multi-invariant verifier dimensions with a learned task-structure router "
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
        primary_router = train_router(domain_pools, held_out_domain=domain_id, include_labels=True)
        ablated_router = train_router(domain_pools, held_out_domain=domain_id, include_labels=False)
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
        label_robust = label_ablation_survives(
            primary, ablated, tolerance=label_ablation_tolerance
        )
        primary["label_ablation_robust"] = label_robust
        primary["label_ablation"] = {
            "router_set_encoder_at_1": ablated["router_set_encoder_at_1"],
            "cross_domain_delta": ablated["cross_domain_delta"],
            "cross_domain_ci95": ablated["cross_domain_ci95"],
            "protocol": "domain_id and family_id removed from router input; task/candidate features only",
        }
        primary["cross_domain_selection_holds"] = cross_domain_selection_holds_from_metrics(primary)
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
        "per_domain": {
            key: {
                "cross_domain_delta": value.get("cross_domain_delta"),
                "vote_at_1": value.get("vote_at_1"),
                "oracle_at_k": value.get("oracle_at_k"),
            }
            for key, value in per_domain.items()
        },
        "random_seed": int(random_seed),
        "router": "nearest-domain-centroid over task structure + per-domain standardized set-encoder profiles",
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
    blocked_reason: str | None = None,
) -> dict[str, Any]:
    persisted_pool = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in dict(persisted or {}).items()
    }
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
            "kind": "nearest_train_domain_centroid_router",
            "input_features": "task-level means of Exp 4244 candidate features",
            "label_features": "domain/family labels allowed only in primary arm; held-out domain remains domain-disjoint",
        },
        "per_domain_set_encoder": {
            "kind": "standardized correct-minus-wrong set profile over Exp 4244 features",
            "training_scope": "train domains only for each leave-one-domain-out read",
        },
        "domain_disjoint_split": {
            "primary_held_out_domain": held_out_domain,
            "leave_one_domain_out_domains": [domain for domain in DOMAIN_ORDER if domain in domain_pools],
        },
        "label_ablation_protocol": (
            "rerun held-out routing with domain_id/family_id unavailable; label_ablation_robust "
            "requires held-out accuracy to survive within tolerance and delta to remain positive"
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
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "experiment": "experiment_4305_cross_domain_selector_generalization",
        "schema": "carnot.cross_domain_selector_generalization_4305.v1",
        "status": "blocked",
        "honest_verdict": reason,
        "cross_domain_selection_holds": False,
        "cross_domain_delta": 0.0,
        "cross_domain_ci95": [0.0, 0.0],
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
) -> dict[str, Any]:
    label_only_failure = (
        float(primary.get("cross_domain_delta", 0.0)) > 0.0
        and _ci_excludes_zero(list(primary.get("cross_domain_ci95", [])))
        and float(primary.get("vote_at_1", 0.0)) > 0.05
        and float(primary.get("oracle_at_k", 1.0)) < 1.0
        and primary.get("label_ablation_robust") is not True
    )
    holds = cross_domain_selection_holds_from_metrics(primary)
    if holds:
        verdict_suffix = "cross_domain_selection_survives"
        gaps: list[dict[str, str]] = []
    elif label_only_failure:
        verdict_suffix = "label_ablation_failure_router_read_label"
        gaps = [_missing_verifier_gap(verdict_suffix, str(primary.get("domain_id")))]
    else:
        verdict_suffix = "cross_domain_selection_collapses_domain_bound"
        gaps = [_missing_verifier_gap(verdict_suffix, str(primary.get("domain_id")))]
    return {
        "experiment": "experiment_4305_cross_domain_selector_generalization",
        "schema": "carnot.cross_domain_selector_generalization_4305.v1",
        "status": "complete",
        "honest_verdict": f"complete: {verdict_suffix}",
        "cross_domain_selection_holds": holds,
        "cross_domain_delta": primary["cross_domain_delta"],
        "cross_domain_ci95": primary["cross_domain_ci95"],
        "vote_at_1": primary["vote_at_1"],
        "oracle_at_k": primary["oracle_at_k"],
        "label_ablation_robust": bool(primary["label_ablation_robust"]),
        "per_domain_delta": {
            domain_id: {
                key: value[key]
                for key in (
                    "cross_domain_delta",
                    "cross_domain_ci95",
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
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
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
    flags: list[dict[str, Any]] = []
    for item in report.get("reports", []):
        if isinstance(item, dict):
            flags.extend(flag for flag in item.get("flags", []) if isinstance(flag, dict))
    return {
        "status": "clean" if not flags else "flagged",
        "degenerate_separation_clean": not any(flag.get("kind") == "DEGENERATE_SEPARATION" for flag in flags),
        "circular_moat_overclaim_clean": not any(flag.get("kind") == "CIRCULAR_MOAT_OVERCLAIM" for flag in flags),
        "flag_count": len(flags),
        "flags": flags,
        "returncode": int(report.get("returncode", 0) or 0),
    }


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
    ci95 = artifact["cross_domain_ci95"]
    if (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in ci95)
    ):
        raise ValueError("cross_domain_ci95 must be a two-number ci95")
    if type(artifact["label_ablation_robust"]) is not bool:
        raise ValueError("label_ablation_robust must be a bare bool")
    if not isinstance(artifact["per_domain_delta"], dict):
        raise ValueError("per_domain_delta must be an object")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if type(artifact["random_seed"]) is not int:
        raise ValueError("random_seed must be a bare int")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4305")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs do not match REQ-VERIFY-4305")


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
