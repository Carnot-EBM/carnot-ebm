"""Exp 4306 powered cross-domain self-learning CI.

Spec refs: REQ-VERIFY-4306, SCENARIO-VERIFY-4306.

This runner reuses the Exp 4295 controller-learning mechanisms on the Exp 4305
cross-domain pool. It updates only CPU-side selector counters and memory
contexts after each already-scored family. No model weights are trained,
fine-tuned, or mutated.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_4305_cross_domain_selector_generalization as exp4305
from carnot.reporting import arc_cross_family_online_adaptation_4273 as exp4273
from carnot.reporting import arc_cross_family_transfer_existing_pool_4271 as exp4271
from carnot.reporting import self_learning_tier2_fixed_retrieval_4295 as exp4295


RANDOM_SEED = 4306
BOOTSTRAP_RESAMPLES = 2000
RETRIEVAL_K = 3
OUTPUT_REL = Path("results/experiment_4306_self_learning_powered_ci_cross_domain.json")
ENTRYPOINT_REL = Path("results/experiment_4306_self_learning_powered_ci_cross_domain.py")
CROSS_DOMAIN_ARTIFACT_REL = exp4305.OUTPUT_REL
CROSS_DOMAIN_POOL_REL = exp4305.POOL_REL
CROSS_DOMAIN_MANIFEST_REL = exp4305.MANIFEST_REL
PRIOR_SELF_LEARNING_REL = exp4295.OUTPUT_REL
BLOCKED_POOLS_VERDICT = "blocked_pools_missing"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
SPEC_REFS = ["REQ-VERIFY-4306", "SCENARIO-VERIFY-4306"]
ARM_KEYS = ("static", "online", "tier2_memory", "tier2_retrieval")
ADAPTIVE_ARM_KEYS = ("online", "tier2_memory", "tier2_retrieval")

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A powered 'online adaptation helps' (CI95-excl-0), "
        "a powered null (static is genuinely the ceiling), and an honest "
        "blocked_pools_missing are ALL COMPLETE and decision-grade."
    ),
    "online_adaptation_helps": (
        "BARE bool: the capstone reads this (gated-fields-must-be-bare); true iff "
        "best-adaptive - static > 0 AND its powered CI95 excludes 0 -- cheap "
        "Tier-1/Tier-2 adaptation beats the static selector, decision-grade "
        "(not the .397 unpowered delta)."
    ),
    "best_adaptive_minus_static_delta": (
        "BARE float: (best of online/tier2-memory/tier2-retrieval) - static -- "
        "the load-bearing self-learning gain (compare to the .397 +0.067)."
    ),
    "best_adaptive_minus_static_ci95": (
        "Powered bootstrap CI95 (>=2000 resamples) of the best-adaptive-minus-static "
        "delta -- excluding 0 makes 'helps' decision-grade; including 0 retires the ask."
    ),
    "arm_deltas": (
        "Per-arm cross-family delta (static / online / tier2_memory / tier2_retrieval) "
        "-- shows WHICH adaptation tier helps (and that tier-2 is no longer a no-op)."
    ),
    "positive_control_headroom": (
        "Oracle-minus-vote headroom on the same cross-domain tasks -- the "
        "FALSE_NEGATIVE_RISK guard that makes a null informative instead of degenerate."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- online/retrieval reweighting of a learned selector, "
        "NO executable oracle, NO weight mutation."
    ),
    "random_seed": "Determinism precondition for the adaptation order + bootstrap.",
    "reproducibility_checksum": (
        "Hash of the arms + the cross-domain pools + the bootstrap; lets a third party re-run."
    ),
    "model_specs": (
        "The four arms + the cross-domain pools + the retrieval mechanism + the "
        "bootstrap protocol; required methodology."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "online_adaptation_helps",
    "best_adaptive_minus_static_delta",
    "best_adaptive_minus_static_ci95",
    "arm_deltas",
    "positive_control_headroom",
    "verifier_is_oracle",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
    "field_principles",
    "spec_refs",
    "acceptance_gate",
    "adversarial_verify",
)


class BlockedRun(RuntimeError):
    """Expected precondition failure that still writes a terminal artifact."""

    def __init__(self, missing_pools: Sequence[Mapping[str, Any]] | str | None = None) -> None:
        super().__init__(BLOCKED_POOLS_VERDICT)
        if isinstance(missing_pools, str):
            self.missing_pools = [{"reason": missing_pools}]
        else:
            self.missing_pools = [dict(item) for item in (missing_pools or [])]


@dataclass(frozen=True)
class ExperimentInputs:
    corpus: exp4271.FamilyAnnotatedCorpus
    static_task_rows: list[dict[str, Any]]
    feature_names: list[str]
    cross_domain_pool_path: Path
    cross_domain_pool_sha256: str
    domain_manifest_path: Path
    domain_manifest_sha256: str
    domain_sources: dict[str, dict[str, Any]]
    upstream_artifacts: dict[str, dict[str, Any]]
    input_notes: list[str]


def _round_metric(value: float) -> float:
    return round(float(value), 10)


def _bare_float(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{field} must be a bare float")


def _validate_ci_pair(value: Any, field: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise ValueError(f"{field} must contain two-number ci95 lists")


def _ci_excludes_zero(ci95: Sequence[float]) -> bool:
    return bool(len(ci95) == 2 and (float(ci95[0]) > 0.0 or float(ci95[1]) < 0.0))


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _read_json_gz_object(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _qualified_family_id(domain_id: str, family_id: str) -> str:
    family = str(family_id or "unknown_family")
    return family if family.startswith(f"{domain_id}:") else f"{domain_id}:{family}"


def _domain_sources(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = manifest.get("domain_sources")
    if not isinstance(raw, dict):
        return {}
    return {
        str(domain_id): dict(source)
        for domain_id, source in raw.items()
        if isinstance(source, dict)
    }


def _resolve_source_path(repo_root: Path, source_path: Any) -> Path:
    path = Path(str(source_path or ""))
    return path if path.is_absolute() else repo_root / path


def _missing_source_paths(repo_root: Path, sources: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for domain_id in exp4305.DOMAIN_ORDER:
        source = sources.get(domain_id)
        if source is None:
            missing.append({"domain_id": domain_id, "path": "domain_source_missing"})
            continue
        source_path = _resolve_source_path(repo_root, source.get("source_path"))
        if not source_path.exists():
            missing.append({"domain_id": domain_id, "path": str(source_path)})
    return missing


def _candidate_from_json(task: Mapping[str, Any], candidate: Mapping[str, Any]) -> exp4305.CandidateRow:
    domain_id = str(candidate.get("domain_id") or task.get("domain_id") or "")
    features_raw = candidate.get("features") if isinstance(candidate.get("features"), dict) else {}
    return exp4305.CandidateRow(
        task_id=str(candidate.get("task_id") or task.get("task_id") or ""),
        candidate_id=str(candidate.get("candidate_id") or ""),
        candidate_index=int(candidate.get("candidate_index", 0) or 0),
        domain_id=domain_id,
        family_id=str(candidate.get("family_id") or task.get("family_id") or ""),
        target_hash=str(candidate.get("target_hash") or task.get("target_hash") or ""),
        is_correct=candidate.get("is_correct") is True,
        vote_weight=exp4273._safe_float(candidate.get("vote_weight")),
        features={
            name: exp4273._safe_float(features_raw.get(name))
            for name in exp4305.FEATURE_NAMES
        },
    )


def _domain_pools_from_payload(
    pool_payload: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, exp4305.DomainPool]:
    by_domain: dict[str, list[exp4305.CandidateRow]] = defaultdict(list)
    for task in pool_payload.get("tasks", []):
        if not isinstance(task, dict):
            continue
        for candidate in task.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            row = _candidate_from_json(task, candidate)
            if row.domain_id:
                by_domain[row.domain_id].append(row)
    return {
        domain_id: exp4305.DomainPool(
            domain_id=domain_id,
            rows=rows,
            source_path=str(sources.get(domain_id, {}).get("source_path") or ""),
            source_sha256=str(sources.get(domain_id, {}).get("source_sha256") or ""),
            provenance=dict(sources.get(domain_id, {}).get("provenance") or {}),
        )
        for domain_id, rows in by_domain.items()
        if rows
    }


def _corpus_from_domain_pools(
    domain_pools: Mapping[str, exp4305.DomainPool],
    *,
    pool_path: Path,
    pool_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
) -> exp4271.FamilyAnnotatedCorpus:
    rows: list[exp4271.FamilyAnnotatedRow] = []
    task_family_ids: dict[str, str] = {}
    task_folds: dict[str, int] = {}
    fold_by_domain = {domain_id: index for index, domain_id in enumerate(exp4305.DOMAIN_ORDER)}
    for domain_id in exp4305.DOMAIN_ORDER:
        pool = domain_pools.get(domain_id)
        if pool is None:
            continue
        for item in pool.rows:
            family_id = _qualified_family_id(item.domain_id, item.family_id)
            rows.append(
                exp4271.FamilyAnnotatedRow(
                    task_id=item.task_id,
                    family_id=family_id,
                    fold=fold_by_domain.get(item.domain_id, 0),
                    candidate_id=item.candidate_id,
                    candidate_index=item.candidate_index,
                    correct=item.is_correct,
                    features=dict(item.features),
                    vote_weight=float(item.vote_weight),
                )
            )
            task_family_ids[item.task_id] = family_id
            task_folds[item.task_id] = fold_by_domain.get(item.domain_id, 0)
    return exp4271.FamilyAnnotatedCorpus(
        rows=rows,
        task_family_ids=task_family_ids,
        task_folds=task_folds,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256.removeprefix("sha256:"),
        pool_artifact_path=pool_path,
        pool_artifact_sha256=pool_sha256.removeprefix("sha256:"),
        upstream_checksum=pool_sha256,
        held_out_family_n=len(set(task_family_ids.values())),
        held_out_task_n=len(task_family_ids),
        candidate_n=len(rows),
    )


def _static_rows_from_cross_domain_reports(
    domain_pools: Mapping[str, exp4305.DomainPool],
    *,
    random_seed: int,
    bootstrap_resamples: int,
) -> list[dict[str, Any]]:
    reports = exp4305._per_domain_reports(
        domain_pools,
        random_seed=random_seed,
        bootstrap_resamples=bootstrap_resamples,
        label_ablation_tolerance=0.05,
    )
    static_rows: list[dict[str, Any]] = []
    for domain_id in exp4305.DOMAIN_ORDER:
        report = reports.get(domain_id)
        if not report:
            continue
        for row in report.get("task_rows", []):
            if not isinstance(row, dict):
                continue
            static_rows.append(
                {
                    "task_id": str(row.get("task_id") or ""),
                    "family_id": _qualified_family_id(domain_id, str(row.get("family_id") or "")),
                    "fold": exp4305.DOMAIN_ORDER.index(domain_id),
                    "vote_candidate_id": str(row.get("vote_candidate_id") or ""),
                    "vote_correct": bool(row.get("vote_correct")),
                    "set_encoder_candidate_id": str(row.get("router_set_encoder_candidate_id") or ""),
                    "set_encoder_correct": bool(row.get("router_set_encoder_correct")),
                }
            )
    return static_rows


def _optional_artifact(repo_root: Path, rel_path: Path) -> dict[str, Any]:
    path = repo_root / rel_path
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    try:
        return _read_json_object(path)
    except Exception as exc:  # pragma: no cover - defensive live artifact metadata.
        return {"status": "unreadable", "path": str(path), "error": f"{type(exc).__name__}: {exc}"}


def load_inputs(repo_root: Path | str = Path(".")) -> ExperimentInputs:
    root = Path(repo_root)
    pool_path = root / CROSS_DOMAIN_POOL_REL
    manifest_path = root / CROSS_DOMAIN_MANIFEST_REL
    missing = [
        {"path": str(path)}
        for path in (pool_path, manifest_path)
        if not path.exists()
    ]
    if missing:
        raise BlockedRun(missing)

    try:
        pool_payload = _read_json_gz_object(pool_path)
        manifest_payload = _read_json_object(manifest_path)
    except Exception as exc:
        raise BlockedRun([{"path": str(pool_path), "reason": f"{type(exc).__name__}: {exc}"}]) from exc

    sources = _domain_sources(manifest_payload)
    missing_sources = _missing_source_paths(root, sources)
    domain_pools = _domain_pools_from_payload(pool_payload, sources)
    missing_domains = [
        {"domain_id": domain_id, "path": "cross_domain_pool_rows_missing"}
        for domain_id in exp4305.DOMAIN_ORDER
        if domain_id not in domain_pools
    ]
    if missing_sources or missing_domains:
        raise BlockedRun([*missing_sources, *missing_domains])

    pool_sha256 = _sha256_file(pool_path)
    manifest_sha256 = _sha256_file(manifest_path)
    corpus = _corpus_from_domain_pools(
        domain_pools,
        pool_path=pool_path.resolve(),
        pool_sha256=pool_sha256,
        manifest_path=manifest_path.resolve(),
        manifest_sha256=manifest_sha256,
    )
    static_rows = _static_rows_from_cross_domain_reports(
        domain_pools,
        random_seed=RANDOM_SEED,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
    )
    if not corpus.rows or not static_rows:
        raise BlockedRun([{"path": str(pool_path), "reason": "empty_cross_domain_stream"}])
    return ExperimentInputs(
        corpus=corpus,
        static_task_rows=static_rows,
        feature_names=list(exp4305.FEATURE_NAMES),
        cross_domain_pool_path=pool_path.resolve(),
        cross_domain_pool_sha256=pool_sha256,
        domain_manifest_path=manifest_path.resolve(),
        domain_manifest_sha256=manifest_sha256,
        domain_sources={domain_id: dict(sources[domain_id]) for domain_id in exp4305.DOMAIN_ORDER},
        upstream_artifacts={
            "experiment_4305": _optional_artifact(root, CROSS_DOMAIN_ARTIFACT_REL),
            "experiment_4295": _optional_artifact(root, PRIOR_SELF_LEARNING_REL),
        },
        input_notes=["exp4305_cross_domain_pool_loaded", "exp4295_adaptation_components_imported"],
    )


def _group_by_task(
    rows: Sequence[exp4271.FamilyAnnotatedRow],
) -> dict[str, list[exp4271.FamilyAnnotatedRow]]:
    grouped: dict[str, list[exp4271.FamilyAnnotatedRow]] = defaultdict(list)
    for row in rows:
        grouped[row.task_id].append(row)
    return {
        task_id: sorted(task_rows, key=lambda item: item.candidate_index)
        for task_id, task_rows in grouped.items()
    }


def positive_control_headroom(corpus: exp4271.FamilyAnnotatedCorpus) -> dict[str, Any]:
    vote_hits: list[bool] = []
    oracle_hits: list[bool] = []
    for task_rows in _group_by_task(corpus.rows).values():
        vote_pick = max(task_rows, key=lambda row: (row.vote_weight, -row.candidate_index))
        vote_hits.append(bool(vote_pick.correct))
        oracle_hits.append(any(row.correct for row in task_rows))
    vote_at_1 = exp4273._rate(vote_hits)
    oracle_at_k = exp4273._rate(oracle_hits)
    return {
        "passed": bool(oracle_at_k > vote_at_1),
        "vote_at_1": _round_metric(vote_at_1),
        "oracle_at_k": _round_metric(oracle_at_k),
        "oracle_minus_vote": _round_metric(oracle_at_k - vote_at_1),
        "task_n": len(vote_hits),
        "family_n": len(set(corpus.task_family_ids.values())),
    }


def _best_adaptive_arm(arm_deltas: Mapping[str, float]) -> str:
    order = {arm: index for index, arm in enumerate(ADAPTIVE_ARM_KEYS)}
    return max(ADAPTIVE_ARM_KEYS, key=lambda arm: (float(arm_deltas[arm]), -order[arm]))


def measure_powered_adaptation(
    inputs: ExperimentInputs,
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    retrieval_k: int = RETRIEVAL_K,
) -> dict[str, Any]:
    """SCENARIO-VERIFY-4306: run the four adaptive arms on cross-domain families."""

    raw = exp4295.measure_self_learning(
        inputs.corpus,
        inputs.static_task_rows,
        feature_names=inputs.feature_names,
        arcgen_used=True,
        random_seed=random_seed,
        bootstrap_resamples=bootstrap_resamples,
        retrieval_k=retrieval_k,
    )
    arm_deltas = {
        "static": _round_metric(raw["static_cross_family_delta"]),
        "online": _round_metric(raw["online_cross_family_delta"]),
        "tier2_memory": _round_metric(raw["tier2_memory_cross_family_delta"]),
        "tier2_retrieval": _round_metric(raw["tier2_retrieval_cross_family_delta"]),
    }
    best_arm = _best_adaptive_arm(arm_deltas)
    best_delta = _round_metric(arm_deltas[best_arm] - arm_deltas["static"])
    ci_by_arm = {
        arm: list(raw["adaptive_minus_static_ci95"][arm])
        for arm in ADAPTIVE_ARM_KEYS
    }
    best_ci = ci_by_arm[best_arm]
    helps = bool(best_delta > 0.0 and _ci_excludes_zero(best_ci))
    return {
        "online_adaptation_helps": helps,
        "best_adaptive_arm": best_arm,
        "best_adaptive_minus_static_delta": best_delta,
        "best_adaptive_minus_static_ci95": best_ci,
        "arm_deltas": arm_deltas,
        "arm_minus_static_ci95": ci_by_arm,
        "positive_control_headroom": positive_control_headroom(inputs.corpus),
        "adaptation_curve": raw["adaptation_curve"],
        "pass_rates": raw["pass_rates"],
        "adaptive_diagnostics": raw["adaptive_diagnostics"],
        "tier2_diagnostics": raw["tier2_diagnostics"],
        "precision_table": raw["precision_table"],
        "tier2_memory_table": raw["tier2_memory_table"],
        "family_stream_order": raw["family_stream_order"],
        "task_rows": raw["task_rows"],
        "held_out_family_n": raw["held_out_family_n"],
        "held_out_task_n": raw["held_out_task_n"],
        "candidate_count": inputs.corpus.candidate_n,
        "bootstrap_resamples": int(bootstrap_resamples),
        "retrieval_k": int(retrieval_k),
    }


def _checksum_payload(inputs: ExperimentInputs, metrics: Mapping[str, Any], random_seed: int) -> dict[str, Any]:
    return {
        "arm_deltas": metrics.get("arm_deltas"),
        "best_adaptive_arm": metrics.get("best_adaptive_arm"),
        "best_adaptive_minus_static_ci95": metrics.get("best_adaptive_minus_static_ci95"),
        "best_adaptive_minus_static_delta": metrics.get("best_adaptive_minus_static_delta"),
        "bootstrap_resamples": metrics.get("bootstrap_resamples"),
        "domain_manifest_sha256": inputs.domain_manifest_sha256,
        "family_stream_order": metrics.get("family_stream_order"),
        "pool_sha256": inputs.cross_domain_pool_sha256,
        "positive_control_headroom": metrics.get("positive_control_headroom"),
        "random_seed": int(random_seed),
        "retrieval_k": metrics.get("retrieval_k"),
    }


def reproducibility_checksum(
    inputs: ExperimentInputs,
    metrics: Mapping[str, Any],
    *,
    random_seed: int,
) -> str:
    raw = json.dumps(
        _checksum_payload(inputs, metrics, random_seed),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _blocked_checksum(missing_pools: Sequence[Mapping[str, Any]], random_seed: int) -> str:
    raw = json.dumps(
        {"missing_pools": list(missing_pools), "random_seed": int(random_seed)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _model_specs(inputs: ExperimentInputs, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "complete",
        "arms": {
            "static": {
                "selector": "frozen_cross_domain_router_set_encoder_rows",
                "model_training": False,
                "weight_mutation": False,
            },
            "tier1_online": {
                "selector": "Exp4295 CPU per-arm precision tracker",
                "counter_update_only": True,
                "model_training": False,
                "weight_mutation": False,
                "hardware_path": "CPU counter updates; FPGA-friendly integer counters",
            },
            "tier2_memory": {
                "selector": "Exp4295 nearest-family selector-pattern cache",
                "model_training": False,
                "weight_mutation": False,
                "hardware_path": "CPU/system-memory cache lookup; FPGA-friendly pattern match",
            },
            "tier2_retrieval": {
                "selector": "Exp4295 retrieval-only nearest seen-family curated context",
                "retrieval_k": int(metrics.get("retrieval_k", RETRIEVAL_K)),
                "model_training": False,
                "weight_mutation": False,
                "hardware_path": "CPU/system-memory nearest-neighbor lookup",
            },
        },
        "cross_domain_pools": {
            "pool_path": str(inputs.cross_domain_pool_path),
            "pool_sha256": inputs.cross_domain_pool_sha256,
            "manifest_path": str(inputs.domain_manifest_path),
            "manifest_sha256": inputs.domain_manifest_sha256,
            "domain_sources": inputs.domain_sources,
            "input_notes": inputs.input_notes,
        },
        "retrieval_mechanism": {
            "kind": "retrieval_only_curated_context_no_weight_mutation",
            "nearest_neighbor_space": "family mean candidate-feature profile",
            "current_family_feedback": "not visible until after that family is scored",
        },
        "bootstrap_protocol": {
            "unit": "family",
            "method": "paired percentile bootstrap",
            "resamples": int(metrics.get("bootstrap_resamples", BOOTSTRAP_RESAMPLES)),
            "best_arm_selection": "highest adaptive cross-family delta before CI read",
            "random_seed": RANDOM_SEED,
        },
        "upstream_artifacts": inputs.upstream_artifacts,
        "verifier_is_oracle": False,
    }


def _blocked_model_specs(missing_pools: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "status": "blocked",
        "blocked_reason": BLOCKED_POOLS_VERDICT,
        "missing_pools": list(missing_pools),
        "arms": {
            "static": {"model_training": False, "weight_mutation": False},
            "tier1_online": {"counter_update_only": True, "model_training": False, "weight_mutation": False},
            "tier2_memory": {"model_training": False, "weight_mutation": False},
            "tier2_retrieval": {"model_training": False, "weight_mutation": False},
        },
        "bootstrap_protocol": {"resamples": 0, "unit": "family"},
        "verifier_is_oracle": False,
    }


def _complete_artifact(
    *,
    inputs: ExperimentInputs,
    metrics: dict[str, Any],
    checksum: str,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    headroom = metrics["positive_control_headroom"]
    if metrics["online_adaptation_helps"]:
        suffix = "powered_cross_domain_online_adaptation_helps"
    elif headroom.get("passed") is True:
        suffix = "powered_cross_domain_static_is_the_ceiling"
    else:
        suffix = "cross_domain_positive_control_headroom_missing"
    acceptance_passed = bool(
        int(metrics["bootstrap_resamples"]) >= BOOTSTRAP_RESAMPLES
        and headroom.get("passed") is True
    )
    return {
        "experiment": "experiment_4306_self_learning_powered_ci_cross_domain",
        "schema": "carnot.self_learning_powered_ci_cross_domain_4306.v1",
        "status": "complete",
        "honest_verdict": f"complete: {suffix}",
        **metrics,
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "model_specs": _model_specs(inputs, metrics),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": {
            "condition": (
                "online_adaptation_helps + best_adaptive_minus_static_delta + "
                "best_adaptive_minus_static_ci95 + arm_deltas reported with a "
                "positive-control headroom check AND verifier_is_oracle=false, "
                "OR an honest blocked_pools_missing verdict"
            ),
            "passed": acceptance_passed,
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "adversarial_verify": {"status": "pending"},
    }


def _blocked_artifact(
    *,
    missing_pools: Sequence[Mapping[str, Any]],
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4306_self_learning_powered_ci_cross_domain",
        "schema": "carnot.self_learning_powered_ci_cross_domain_4306.v1",
        "status": "blocked",
        "honest_verdict": BLOCKED_POOLS_VERDICT,
        "online_adaptation_helps": False,
        "best_adaptive_minus_static_delta": 0.0,
        "best_adaptive_minus_static_ci95": [0.0, 0.0],
        "best_adaptive_arm": "online",
        "arm_deltas": {arm: 0.0 for arm in ARM_KEYS},
        "arm_minus_static_ci95": {arm: [0.0, 0.0] for arm in ADAPTIVE_ARM_KEYS},
        "positive_control_headroom": {
            "passed": False,
            "vote_at_1": 0.0,
            "oracle_at_k": 0.0,
            "oracle_minus_vote": 0.0,
            "task_n": 0,
            "family_n": 0,
        },
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": _blocked_checksum(missing_pools, random_seed),
        "model_specs": _blocked_model_specs(missing_pools),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": {
            "condition": "honest blocked_pools_missing verdict is complete when pools are missing",
            "passed": False,
        },
        "adversarial_verify": {"status": "pending"},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "adaptation_curve": [],
        "pass_rates": {},
        "adaptive_diagnostics": {},
        "tier2_diagnostics": {},
        "precision_table": {},
        "tier2_memory_table": {},
        "family_stream_order": [],
        "task_rows": [],
        "held_out_family_n": 0,
        "held_out_task_n": 0,
        "candidate_count": 0,
        "bootstrap_resamples": 0,
        "retrieval_k": RETRIEVAL_K,
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
        "circular_moat_overclaim_clean": not any(
            flag.get("kind") == "CIRCULAR_MOAT_OVERCLAIM" for flag in flags
        ),
        "false_negative_risk_clean": not any(
            flag.get("kind") == "FALSE_NEGATIVE_RISK" for flag in flags
        ),
        "degenerate_separation_clean": not any(
            flag.get("kind") == "DEGENERATE_SEPARATION" for flag in flags
        ),
        "flag_count": len(flags),
        "flags": flags,
        "returncode": int(report.get("returncode", 0) or 0),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:") or verdict == BLOCKED_POOLS_VERDICT
    ):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if type(artifact["online_adaptation_helps"]) is not bool:
        raise ValueError("online_adaptation_helps must be a bare bool")
    _bare_float(
        artifact["best_adaptive_minus_static_delta"],
        "best_adaptive_minus_static_delta",
    )
    _validate_ci_pair(
        artifact["best_adaptive_minus_static_ci95"],
        "best_adaptive_minus_static_ci95",
    )
    arm_deltas = artifact["arm_deltas"]
    if not isinstance(arm_deltas, dict) or tuple(arm_deltas.keys()) != ARM_KEYS:
        raise ValueError("arm_deltas must contain static/online/tier2_memory/tier2_retrieval")
    for arm in ARM_KEYS:
        _bare_float(arm_deltas[arm], f"arm_deltas.{arm}")
    if not isinstance(artifact["positive_control_headroom"], dict):
        raise ValueError("positive_control_headroom must be an object")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if type(artifact["random_seed"]) is not int:
        raise ValueError("random_seed must be a bare int")
    if not isinstance(artifact["model_specs"], dict):
        raise ValueError("model_specs must be an object")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4306")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs do not match REQ-VERIFY-4306")
    expected_helps = bool(
        float(artifact["best_adaptive_minus_static_delta"]) > 0.0
        and _ci_excludes_zero(artifact["best_adaptive_minus_static_ci95"])
    )
    if artifact["online_adaptation_helps"] is not expected_helps:
        raise ValueError("online_adaptation_helps must match delta>0 and CI95-excl-0")


def run(
    repo_root: Path | str = Path("."),
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    retrieval_k: int = RETRIEVAL_K,
    adversarial_runner: Callable[[Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)
    output_path = root / OUTPUT_REL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        inputs = load_inputs(root)
        metrics = measure_powered_adaptation(
            inputs,
            random_seed=random_seed,
            bootstrap_resamples=bootstrap_resamples,
            retrieval_k=retrieval_k,
        )
        checksum = reproducibility_checksum(inputs, metrics, random_seed=random_seed)
        artifact = _complete_artifact(
            inputs=inputs,
            metrics=metrics,
            checksum=checksum,
            random_seed=random_seed,
            duration_s=time.perf_counter() - start,
        )
    except BlockedRun as exc:
        artifact = _blocked_artifact(
            missing_pools=exc.missing_pools,
            random_seed=random_seed,
            duration_s=time.perf_counter() - start,
        )
    validate_artifact(artifact)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = (
        adversarial_runner(output_path)
        if adversarial_runner is not None
        else _run_adversarial_verify(root, output_path)
    )
    artifact["adversarial_verify"] = _clean_adversarial_report(report)
    validate_artifact(artifact)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover
    artifact = run(Path(__file__).resolve().parents[2])
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
