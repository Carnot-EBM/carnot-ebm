"""Exp 4283 repowered ARC self-learning adaptation.

Spec refs: REQ-VERIFY-4283, SCENARIO-VERIFY-4283.

This is a controller-learning experiment, not model training. The Tier-1 arm
updates CPU precision counters, and the Tier-2 arm caches family selection
patterns. Both are prequential: a family's labels become usable only after that
family has already been scored.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
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

from carnot.reporting import arc_cross_family_online_adaptation_4273 as exp4273
from carnot.reporting import arc_cross_family_transfer_existing_pool_4271 as exp4271
from carnot.reporting import arc_set_encoder_aggregator_4244 as exp4244


RANDOM_SEED = 4283
BOOTSTRAP_RESAMPLES = 2000
V395_FAMILY_N = 52
ARCGEN_THIN_FAMILY_N = 4
OUTPUT_REL = Path("results/experiment_4283_self_learning_repowered_arcgen.json")
ENTRYPOINT_REL = Path("results/experiment_4283_self_learning_repowered_arcgen.py")
SET_ENCODER_BUILD_REL = exp4271.SET_ENCODER_BUILD_REL
SET_ENCODER_MODEL_REL = exp4271.SET_ENCODER_MODEL_REL
ORIGINAL_CROSS_FAMILY_REL = exp4271.OUTPUT_REL
ORIGINAL_ONLINE_REL = exp4273.OUTPUT_REL
ARCGEN_STRESS_REL = Path("results/experiment_4282_arcgen_cross_family_stress.json")
ARCGEN_POOL_REL = Path("results/experiment_4282_arcgen_candidate_pool.json.gz")
ARCGEN_MANIFEST_REL = Path("results/experiment_4282_arcgen_family_manifest.json")
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
TIER1_COUNTER_UPDATE = exp4273.TIER1_COUNTER_UPDATE
TIER2_MEMORY_UPDATE = "cpu_system_memory_nearest_family_pattern_cache_no_model_retrain"
BLOCKED_INPUTS_VERDICT = "blocked_self_learning_repowered_inputs_missing"
STATIC_ARM = exp4273.STATIC_ARM
VOTE_ARM = exp4273.VOTE_ARM
SPEC_REFS = ["REQ-VERIFY-4283", "SCENARIO-VERIFY-4283"]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An online/Tier-2 gain AND a now-POWERED static-is-the-ceiling "
        "null are BOTH COMPLETE -- both are valid self-learning findings (the powered null "
        "retires the ask)."
    ),
    "online_adaptation_helps": (
        "BARE bool: true iff the best adaptive selector (online or Tier-2) exceeds static "
        "AND the (adaptive-static) CI95 excludes 0 with power -- the self-learning value "
        "verdict the .395 n-limit could not settle."
    ),
    "static_cross_family_delta": (
        "BARE float: the static set-encoder's held-out-family delta -- the no-adaptation baseline."
    ),
    "online_cross_family_delta": (
        "BARE float: the online-reweighted selector's held-out-family delta -- the Tier-1 "
        "self-learning result, now on a powered family set."
    ),
    "tier2_cross_family_delta": (
        "BARE float: the constraint-memory (Tier-2) selector's delta -- a distinct "
        "self-learning mechanism beyond online reweighting."
    ),
    "family_count_vs_v395": (
        "The held-out family count vs the .395 n=52 -- quantifies the power gain (the .395 "
        "CI touched 0 because n was small; a non-touching CI here needs more families)."
    ),
    "adaptation_curve": (
        "Per-family (adaptive - static) gain as families stream in -- shows whether adaptation "
        "compounds (the continuous-self-learning signature) or is flat."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- learned selector with online weight updates + a pattern cache, "
        "no demo execution and no fine-tuning."
    ),
    "random_seed": "Determinism precondition; the family stream order + reweighting reproducible.",
    "reproducibility_checksum": (
        "Hash of the folds + reweighting + memory trace; lets a third party re-run."
    ),
    "model_specs": (
        "The Tier-1 online-reweighting rule + the Tier-2 memory rule + the combined family "
        "stream; required methodology."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "online_adaptation_helps",
    "static_cross_family_delta",
    "online_cross_family_delta",
    "tier2_cross_family_delta",
    "adaptive_minus_static_ci95",
    "family_count_vs_v395",
    "adaptation_curve",
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
    """Expected input failure that still writes a terminal artifact."""


@dataclass(frozen=True)
class ExperimentInputs:
    corpus: exp4271.FamilyAnnotatedCorpus
    static_task_rows: list[dict[str, Any]]
    build_artifact: dict[str, Any]
    model_artifact: dict[str, Any]
    original_artifact: dict[str, Any]
    arcgen_artifact: dict[str, Any] | None
    arcgen_used: bool
    input_notes: list[str]


@dataclass
class PatternMemory:
    family_best_arm: dict[str, str]

    def arm_for(self, nearest_family: str | None) -> str:
        if nearest_family is None:
            return STATIC_ARM
        return self.family_best_arm.get(nearest_family, STATIC_ARM)

    def record_family(self, family_id: str, arm_hits: Mapping[str, Sequence[bool]]) -> None:
        if not arm_hits:
            self.family_best_arm[family_id] = STATIC_ARM
            return
        self.family_best_arm[family_id] = max(
            sorted(arm_hits),
            key=lambda arm: (exp4273._rate([bool(hit) for hit in arm_hits[arm]]), arm != STATIC_ARM),
        )

    def to_json(self) -> dict[str, str]:
        return dict(sorted(self.family_best_arm.items()))


def _round_metric(value: float) -> float:
    return round(float(value), 10)


def _read_json_object(path: Path) -> dict[str, Any]:  # pragma: no cover - exercised by live run.
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _sha256_file(path: Path) -> str:  # pragma: no cover - exercised by live run.
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_required_path(repo_root: Path, value: Any, fallback: Path) -> Path:  # pragma: no cover
    candidate = Path(str(value or fallback))
    resolved = candidate if candidate.is_absolute() else repo_root / candidate
    if not resolved.exists():
        raise BlockedRun(BLOCKED_INPUTS_VERDICT)
    return resolved


def _feature_names(model_artifact: Mapping[str, Any]) -> list[str]:
    raw = model_artifact.get("feature_names", exp4244.FEATURE_NAMES)
    return [str(name) for name in raw if str(name)]


def _static_rows_by_task(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if task_id:
            out[task_id] = dict(row)
    return out


def _static_row_for_task(
    task_id: str,
    family_id: str,
    task_rows: Sequence[exp4271.FamilyAnnotatedRow],
) -> dict[str, Any]:
    vote = max(task_rows, key=lambda row: (row.vote_weight, -row.candidate_index))
    return {
        "task_id": task_id,
        "family_id": family_id,
        "fold": task_rows[0].fold if task_rows else 0,
        "vote_candidate_id": vote.candidate_id,
        "vote_correct": bool(vote.correct),
        "set_encoder_candidate_id": vote.candidate_id,
        "set_encoder_correct": bool(vote.correct),
    }


def _rows_from_arcgen_pool(
    pool_payload: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
) -> exp4271.FamilyAnnotatedCorpus:
    task_family_ids = {str(row["task_id"]): str(row["family_id"]) for row in manifest_rows}
    task_folds = {str(row["task_id"]): int(row.get("fold", 0)) for row in manifest_rows}
    family_rows: list[exp4271.FamilyAnnotatedRow] = []
    for task in pool_payload.get("tasks", []):
        task_id = str(task.get("task_id") or "")
        if task_id not in task_family_ids:
            continue
        for candidate in task.get("candidates", []):
            features = {
                str(key): exp4273._safe_float(value)
                for key, value in dict(candidate.get("features", {})).items()
            }
            family_rows.append(
                exp4271.FamilyAnnotatedRow(
                    task_id=task_id,
                    family_id=task_family_ids[task_id],
                    fold=task_folds[task_id],
                    candidate_id=str(candidate.get("candidate_id") or ""),
                    candidate_index=int(candidate.get("candidate_index", 0) or 0),
                    correct=bool(candidate.get("is_correct")),
                    features=features,
                    vote_weight=exp4273._safe_float(candidate.get("votes")),
                )
            )
    return exp4271.FamilyAnnotatedCorpus(
        rows=family_rows,
        task_family_ids=task_family_ids,
        task_folds=task_folds,
        manifest_path=Path(ARCGEN_MANIFEST_REL),
        manifest_sha256="",
        pool_artifact_path=Path(ARCGEN_POOL_REL),
        pool_artifact_sha256="",
        upstream_checksum=str(pool_payload.get("reproducibility_checksum") or ""),
        held_out_family_n=len(set(task_family_ids.values())),
        held_out_task_n=len(task_family_ids),
        candidate_n=len(family_rows),
    )


def _load_arcgen_component(repo_root: Path) -> tuple[
    exp4271.FamilyAnnotatedCorpus,
    list[dict[str, Any]],
    dict[str, Any],
] | None:  # pragma: no cover - exercised by live run.
    artifact_path = repo_root / ARCGEN_STRESS_REL
    pool_path = repo_root / ARCGEN_POOL_REL
    manifest_path = repo_root / ARCGEN_MANIFEST_REL
    if not artifact_path.exists() or not pool_path.exists() or not manifest_path.exists():
        return None
    artifact = _read_json_object(artifact_path)
    manifest = _read_json_object(manifest_path)
    task_rows = artifact.get("task_rows")
    manifest_rows = manifest.get("rows")
    if not isinstance(task_rows, list) or not isinstance(manifest_rows, list):
        return None
    family_n = len({str(row.get("family_id") or "") for row in manifest_rows if row.get("family_id")})
    if family_n < ARCGEN_THIN_FAMILY_N:
        return None
    with gzip.open(pool_path, "rt", encoding="utf-8") as handle:
        pool_payload = json.load(handle)
    if not isinstance(pool_payload, dict):
        return None
    corpus = _rows_from_arcgen_pool(pool_payload, manifest_rows)
    corpus = exp4271.FamilyAnnotatedCorpus(
        rows=corpus.rows,
        task_family_ids=corpus.task_family_ids,
        task_folds=corpus.task_folds,
        manifest_path=manifest_path.resolve(),
        manifest_sha256=_sha256_file(manifest_path),
        pool_artifact_path=pool_path.resolve(),
        pool_artifact_sha256=_sha256_file(pool_path),
        upstream_checksum=str(pool_payload.get("reproducibility_checksum") or ""),
        held_out_family_n=corpus.held_out_family_n,
        held_out_task_n=corpus.held_out_task_n,
        candidate_n=corpus.candidate_n,
    )
    return corpus, [dict(row) for row in task_rows], artifact


def _merge_corpora(
    original: exp4271.FamilyAnnotatedCorpus,
    arcgen: exp4271.FamilyAnnotatedCorpus | None,
) -> exp4271.FamilyAnnotatedCorpus:
    rows = list(original.rows)
    task_family_ids = dict(original.task_family_ids)
    task_folds = dict(original.task_folds)
    if arcgen is not None:
        rows.extend(arcgen.rows)
        task_family_ids.update(arcgen.task_family_ids)
        task_folds.update(arcgen.task_folds)
    checksum_payload = {
        "original_manifest": original.manifest_sha256,
        "original_pool": original.pool_artifact_sha256,
        "arcgen_manifest": arcgen.manifest_sha256 if arcgen is not None else None,
        "arcgen_pool": arcgen.pool_artifact_sha256 if arcgen is not None else None,
    }
    raw = json.dumps(checksum_payload, sort_keys=True, separators=(",", ":"))
    merged_checksum = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return exp4271.FamilyAnnotatedCorpus(
        rows=rows,
        task_family_ids=task_family_ids,
        task_folds=task_folds,
        manifest_path=original.manifest_path,
        manifest_sha256=merged_checksum,
        pool_artifact_path=original.pool_artifact_path,
        pool_artifact_sha256=merged_checksum,
        upstream_checksum=merged_checksum,
        held_out_family_n=len(set(task_family_ids.values())),
        held_out_task_n=len(task_family_ids),
        candidate_n=len(rows),
    )


def load_inputs(repo_root: Path | str = Path(".")) -> ExperimentInputs:  # pragma: no cover
    root = Path(repo_root)
    try:
        build = _read_json_object(root / SET_ENCODER_BUILD_REL)
        model_path = _resolve_required_path(root, build.get("learned_verifier_path"), SET_ENCODER_MODEL_REL)
        model = exp4244.load_set_encoder(model_path)
        original_corpus = exp4271.load_family_annotated_corpus(root)
        original_cross = _read_json_object(root / ORIGINAL_CROSS_FAMILY_REL)
        original_online = _read_json_object(root / ORIGINAL_ONLINE_REL)
    except Exception as exc:
        raise BlockedRun(BLOCKED_INPUTS_VERDICT) from exc
    if build.get("aggregator_trained") is not True or build.get("verifier_is_oracle") is not False:
        raise BlockedRun(BLOCKED_INPUTS_VERDICT)
    if model.get("verifier_is_oracle") is not False:
        raise BlockedRun(BLOCKED_INPUTS_VERDICT)
    if not isinstance(original_cross.get("task_rows"), list):
        raise BlockedRun(BLOCKED_INPUTS_VERDICT)

    arcgen_component = _load_arcgen_component(root)
    arcgen_corpus: exp4271.FamilyAnnotatedCorpus | None = None
    arcgen_rows: list[dict[str, Any]] = []
    arcgen_artifact: dict[str, Any] | None = None
    notes = ["exp4270_4273_recovered_family_stream_loaded"]
    if arcgen_component is None:
        notes.append("arcgen_absent_or_thin_fallback_to_v395_manifest")
    else:
        arcgen_corpus, arcgen_rows, arcgen_artifact = arcgen_component
        notes.append("exp4282_arcgen_family_stream_appended")
    corpus = _merge_corpora(original_corpus, arcgen_corpus)
    return ExperimentInputs(
        corpus=corpus,
        static_task_rows=[dict(row) for row in original_cross["task_rows"]] + arcgen_rows,
        build_artifact=build,
        model_artifact=model,
        original_artifact=original_online,
        arcgen_artifact=arcgen_artifact,
        arcgen_used=arcgen_component is not None,
        input_notes=notes,
    )


def _task_groups(
    rows: Iterable[exp4271.FamilyAnnotatedRow],
) -> dict[str, list[exp4271.FamilyAnnotatedRow]]:
    grouped: dict[str, list[exp4271.FamilyAnnotatedRow]] = defaultdict(list)
    for row in rows:
        grouped[row.task_id].append(row)
    return {task_id: sorted(items, key=lambda item: item.candidate_index) for task_id, items in grouped.items()}


def _seeded_family_order(family_ids: Iterable[str], *, random_seed: int) -> list[str]:
    families = sorted(set(family_ids))
    rng = __import__("random").Random(random_seed)
    rng.shuffle(families)
    return families


def _family_count_vs_v395(stream_order: Sequence[str], *, arcgen_used: bool) -> dict[str, Any]:
    arcgen_family_n = sum(1 for family_id in stream_order if family_id.startswith("arcgen_native_task:"))
    held_out_family_n = len(stream_order)
    powered = bool(arcgen_used and held_out_family_n > V395_FAMILY_N)
    if powered:
        read = "repowered_with_arcgen_family_gain"
    elif arcgen_used:
        read = "arcgen_present_but_still_under_powered"
    else:
        read = "fallback_still_under_powered"
    return {
        "v395_family_n": V395_FAMILY_N,
        "held_out_family_n": held_out_family_n,
        "power_gain_family_n": held_out_family_n - V395_FAMILY_N,
        "original_family_n": held_out_family_n - arcgen_family_n,
        "arcgen_family_n": arcgen_family_n,
        "arcgen_used": bool(arcgen_used),
        "powered": powered,
        "read": read,
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _pick_tier2(
    nominees: Mapping[str, exp4271.FamilyAnnotatedRow],
    memory_arm: str,
) -> exp4271.FamilyAnnotatedRow:
    return nominees.get(memory_arm) or nominees[STATIC_ARM]


def measure_self_learning(
    corpus: exp4271.FamilyAnnotatedCorpus,
    static_task_rows: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str],
    arcgen_used: bool,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """SCENARIO-VERIFY-4283: score static, online, and Tier-2 prequential arms."""

    grouped_tasks = _task_groups(corpus.rows)
    static_by_task = _static_rows_by_task(static_task_rows)
    filtered_features = [name for name in feature_names if any(name in row.features for row in corpus.rows)]
    arms = exp4273._arm_names(filtered_features)
    profiles = exp4273._family_profiles(corpus, filtered_features)
    family_to_tasks: dict[str, list[str]] = defaultdict(list)
    for task_id, family_id in sorted(corpus.task_family_ids.items()):
        if task_id in grouped_tasks:
            family_to_tasks[family_id].append(task_id)
    stream_order = _seeded_family_order(family_to_tasks, random_seed=random_seed)

    tracker = exp4273.PrecisionTracker()
    memory = PatternMemory(family_best_arm={})
    seen_families: list[str] = []
    family_vote_rates: list[float] = []
    family_static_rates: list[float] = []
    family_online_rates: list[float] = []
    family_tier2_rates: list[float] = []
    family_online_minus_static: list[float] = []
    family_tier2_minus_static: list[float] = []
    family_best_minus_static: list[float] = []
    adaptation_curve: list[dict[str, Any]] = []
    task_rows_payload: list[dict[str, Any]] = []

    for stream_index, family_id in enumerate(stream_order):
        nearest = exp4273._nearest_seen_family(family_id, profiles, seen_families)
        arm_weights = {arm: tracker.weight_for(arm, nearest) for arm in arms}
        memory_arm = memory.arm_for(nearest)
        family_arm_hits: dict[str, list[bool]] = {arm: [] for arm in arms}
        vote_hits: list[bool] = []
        static_hits: list[bool] = []
        online_hits: list[bool] = []
        tier2_hits: list[bool] = []

        for task_id in family_to_tasks[family_id]:
            task_rows = grouped_tasks[task_id]
            static_row = static_by_task.get(task_id) or _static_row_for_task(task_id, family_id, task_rows)
            nominees = exp4273._arm_nominees(task_rows, static_row, filtered_features)
            online_pick = exp4273._online_pick(task_rows, nominees, arm_weights, static_row)
            tier2_pick = _pick_tier2(nominees, memory_arm)
            vote_hit = bool(static_row.get("vote_correct"))
            static_hit = bool(static_row.get("set_encoder_correct"))
            online_hit = bool(online_pick.correct)
            tier2_hit = bool(tier2_pick.correct)
            vote_hits.append(vote_hit)
            static_hits.append(static_hit)
            online_hits.append(online_hit)
            tier2_hits.append(tier2_hit)
            for arm, nominee in nominees.items():
                family_arm_hits[arm].append(bool(nominee.correct))
            task_rows_payload.append(
                {
                    "task_id": task_id,
                    "family_id": family_id,
                    "static_candidate_id": str(static_row.get("set_encoder_candidate_id") or ""),
                    "static_correct": static_hit,
                    "vote_candidate_id": str(static_row.get("vote_candidate_id") or ""),
                    "vote_correct": vote_hit,
                    "online_candidate_id": online_pick.candidate_id,
                    "online_correct": online_hit,
                    "tier2_candidate_id": tier2_pick.candidate_id,
                    "tier2_correct": tier2_hit,
                    "tier2_memory_arm": memory_arm,
                    "nearest_seen_family": nearest,
                }
            )

        static_rate = exp4273._rate(static_hits)
        online_rate = exp4273._rate(online_hits)
        tier2_rate = exp4273._rate(tier2_hits)
        vote_rate = exp4273._rate(vote_hits)
        online_gain = online_rate - static_rate
        tier2_gain = tier2_rate - static_rate
        best_gain = max(online_gain, tier2_gain)
        family_vote_rates.append(vote_rate)
        family_static_rates.append(static_rate)
        family_online_rates.append(online_rate)
        family_tier2_rates.append(tier2_rate)
        family_online_minus_static.append(online_gain)
        family_tier2_minus_static.append(tier2_gain)
        family_best_minus_static.append(best_gain)
        adaptation_curve.append(
            {
                "stream_index": stream_index,
                "family_id": family_id,
                "nearest_seen_family": nearest,
                "task_n": len(static_hits),
                "static_at_1": _round_metric(static_rate),
                "online_at_1": _round_metric(online_rate),
                "tier2_at_1": _round_metric(tier2_rate),
                "vote_at_1": _round_metric(vote_rate),
                "online_minus_static_gain": _round_metric(online_gain),
                "tier2_minus_static_gain": _round_metric(tier2_gain),
                "best_adaptive_minus_static_gain": _round_metric(best_gain),
                "cumulative_best_adaptive_minus_static_gain": _round_metric(_mean(family_best_minus_static)),
                "tier2_memory_arm": memory_arm,
                "top_arm_weights_before": exp4273._top_arm_weights(arm_weights),
            }
        )
        tracker.record_family(family_id, family_arm_hits)
        memory.record_family(family_id, family_arm_hits)
        seen_families.append(family_id)

    static_delta = _round_metric(_mean(family_static_rates) - _mean(family_vote_rates))
    online_delta = _round_metric(_mean(family_online_rates) - _mean(family_vote_rates))
    tier2_delta = _round_metric(_mean(family_tier2_rates) - _mean(family_vote_rates))
    online_ci = exp4273._bootstrap_ci95(
        family_online_minus_static,
        random_seed=random_seed,
        resamples=bootstrap_resamples,
    )
    tier2_ci = exp4273._bootstrap_ci95(
        family_tier2_minus_static,
        random_seed=random_seed + 1,
        resamples=bootstrap_resamples,
    )
    count_read = _family_count_vs_v395(stream_order, arcgen_used=arcgen_used)
    online_gain = _mean(family_online_minus_static)
    tier2_gain = _mean(family_tier2_minus_static)
    best_gain, best_ci, best_arm = (
        (tier2_gain, tier2_ci, "tier2")
        if tier2_gain > online_gain
        else (online_gain, online_ci, "online")
    )
    helps = bool(best_gain > 0.0 and exp4273._ci_excludes_zero(best_ci) and count_read["powered"])
    return {
        "online_adaptation_helps": helps,
        "static_cross_family_delta": static_delta,
        "online_cross_family_delta": online_delta,
        "tier2_cross_family_delta": tier2_delta,
        "adaptive_minus_static_ci95": {
            "online": online_ci,
            "tier2": tier2_ci,
            "best_adaptive_arm": best_arm,
        },
        "family_count_vs_v395": count_read,
        "adaptation_curve": adaptation_curve,
        "pass_rates": {
            "vote_family_mean_at_1": _round_metric(_mean(family_vote_rates)),
            "static_family_mean_at_1": _round_metric(_mean(family_static_rates)),
            "online_family_mean_at_1": _round_metric(_mean(family_online_rates)),
            "tier2_family_mean_at_1": _round_metric(_mean(family_tier2_rates)),
        },
        "adaptive_diagnostics": {
            "online_minus_static_family_mean": _round_metric(online_gain),
            "tier2_minus_static_family_mean": _round_metric(tier2_gain),
            "best_minus_static_family_mean": _round_metric(best_gain),
            "family_level_bootstrap": True,
            "powered_bootstrap": bool(count_read["powered"]),
        },
        "held_out_family_n": len(stream_order),
        "held_out_task_n": len(task_rows_payload),
        "bootstrap_resamples": int(bootstrap_resamples),
        "tier1_counter_update": TIER1_COUNTER_UPDATE,
        "tier2_memory_update": TIER2_MEMORY_UPDATE,
        "precision_table": tracker.precision_table(),
        "tier2_memory_table": memory.to_json(),
        "task_rows": task_rows_payload,
        "feature_arm_n": len(filtered_features),
        "selector_arm_n": len(arms),
        "family_stream_order": stream_order,
    }


def _model_specs(inputs: ExperimentInputs, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "family_stream_protocol": {
            "split_unit": "family_id",
            "stream_order_policy": "seeded_shuffle_of_combined_family_ids",
            "combined_family_stream": metrics.get("family_stream_order", []),
            "family_count_vs_v395": metrics.get("family_count_vs_v395", {}),
            "input_notes": list(inputs.input_notes),
        },
        "tier1_online_reweighting_rule": {
            "arms": "static_set_encoder + vote_weight + per-feature candidate selectors",
            "counter_update": TIER1_COUNTER_UPDATE,
            "precision_prior": exp4273.PRECISION_PRIOR,
            "nearest_family_weighting": "0.5 * global_precision + 0.5 * nearest_seen_family_precision",
            "current_family_feedback": "not used until after every task in that family is scored",
            "fine_tuning": False,
            "model_training": False,
            "hardware_path": "pure_cpu_counter_updates_lt_1us_per_arm_update",
        },
        "tier2_memory_rule": {
            "cache": "per-family best selector arm pattern",
            "lookup": "reuse nearest already-seen family's cached selector arm",
            "miss_policy": "static_set_encoder",
            "current_family_feedback": "cached only after every task in that family is scored",
            "fine_tuning": False,
            "model_training": False,
            "hardware_path": "CPU/system-memory cache lookup, FPGA-friendly pattern match",
        },
        "upstream_artifacts": {
            "exp4273_checksum": inputs.original_artifact.get("reproducibility_checksum"),
            "exp4282_checksum": (inputs.arcgen_artifact or {}).get("reproducibility_checksum"),
            "exp4244_build_checksum": inputs.build_artifact.get("reproducibility_checksum"),
            "exp4244_model_checksum": inputs.model_artifact.get("reproducibility_checksum"),
        },
        "set_encoder_config": inputs.build_artifact.get("model_specs", {}),
    }


def reproducibility_checksum(
    *,
    inputs: ExperimentInputs,
    metrics: Mapping[str, Any],
    random_seed: int,
) -> str:
    payload = {
        "adaptive_minus_static_ci95": metrics.get("adaptive_minus_static_ci95"),
        "adaptation_curve": metrics.get("adaptation_curve"),
        "family_count_vs_v395": metrics.get("family_count_vs_v395"),
        "family_stream_order": metrics.get("family_stream_order"),
        "manifest_sha256": inputs.corpus.manifest_sha256,
        "model_checksum": inputs.model_artifact.get("reproducibility_checksum"),
        "pool_artifact_sha256": inputs.corpus.pool_artifact_sha256,
        "random_seed": int(random_seed),
        "static_cross_family_delta": metrics.get("static_cross_family_delta"),
        "online_cross_family_delta": metrics.get("online_cross_family_delta"),
        "tier2_cross_family_delta": metrics.get("tier2_cross_family_delta"),
        "tier2_memory_table": metrics.get("tier2_memory_table"),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _blocked_checksum(reason: str, random_seed: int) -> str:
    raw = json.dumps({"random_seed": int(random_seed), "reason": reason}, sort_keys=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _terminal_artifact(reason: str, *, random_seed: int, duration_s: float) -> dict[str, Any]:
    return {
        "experiment": "experiment_4283_self_learning_repowered_arcgen",
        "schema": "carnot.self_learning_repowered_arcgen_4283.v1",
        "status": "blocked" if reason.startswith("blocked_") else "complete",
        "honest_verdict": reason,
        "online_adaptation_helps": False,
        "static_cross_family_delta": 0.0,
        "online_cross_family_delta": 0.0,
        "tier2_cross_family_delta": 0.0,
        "adaptive_minus_static_ci95": {
            "online": [0.0, 0.0],
            "tier2": [0.0, 0.0],
            "best_adaptive_arm": "online",
        },
        "family_count_vs_v395": {
            "v395_family_n": V395_FAMILY_N,
            "held_out_family_n": 0,
            "power_gain_family_n": -V395_FAMILY_N,
            "original_family_n": 0,
            "arcgen_family_n": 0,
            "arcgen_used": False,
            "powered": False,
            "read": "blocked_no_family_stream",
        },
        "adaptation_curve": [],
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": _blocked_checksum(reason, random_seed),
        "model_specs": {
            "status": "blocked",
            "blocked_reason": reason,
            "tier1_online_reweighting_rule": {"fine_tuning": False, "model_training": False},
            "tier2_memory_rule": {"fine_tuning": False, "model_training": False},
        },
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "pass_rates": {},
        "adaptive_diagnostics": {
            "online_minus_static_family_mean": 0.0,
            "tier2_minus_static_family_mean": 0.0,
            "best_minus_static_family_mean": 0.0,
            "family_level_bootstrap": True,
            "powered_bootstrap": False,
        },
        "held_out_family_n": 0,
        "held_out_task_n": 0,
        "bootstrap_resamples": 0,
        "tier1_counter_update": TIER1_COUNTER_UPDATE,
        "tier2_memory_update": TIER2_MEMORY_UPDATE,
        "precision_table": {},
        "tier2_memory_table": {},
        "task_rows": [],
        "duration_s": round(float(duration_s), 6),
        "adversarial_verify": {"status": "pending"},
    }


def _complete_artifact(
    *,
    inputs: ExperimentInputs,
    metrics: dict[str, Any],
    checksum: str,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    if metrics["online_adaptation_helps"]:
        read = "adaptive_self_learning_improves_generalization"
    elif metrics["family_count_vs_v395"]["powered"]:
        read = "powered_static_is_the_ceiling_for_self_learning"
    else:
        read = "fallback_still_under_powered_static_ceiling_unsettled"
    return {
        "experiment": "experiment_4283_self_learning_repowered_arcgen",
        "schema": "carnot.self_learning_repowered_arcgen_4283.v1",
        "status": "complete",
        "honest_verdict": f"complete: {read}",
        **metrics,
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "model_specs": _model_specs(inputs, metrics),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "candidate_count": inputs.corpus.candidate_n,
        "duration_s": round(float(duration_s), 6),
        "adversarial_verify": {"status": "pending"},
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
    circular_clean = not any(flag.get("kind") == "CIRCULAR_MOAT_OVERCLAIM" for flag in flags)
    return {
        "status": "clean" if not flags else "flagged",
        "circular_moat_overclaim_clean": circular_clean,
        "flag_count": len(flags),
        "flags": flags,
        "returncode": int(report.get("returncode", 0) or 0),
    }


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


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:") or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if type(artifact["online_adaptation_helps"]) is not bool:
        raise ValueError("online_adaptation_helps must be a bare bool")
    for field in ("static_cross_family_delta", "online_cross_family_delta", "tier2_cross_family_delta"):
        _bare_float(artifact[field], field)
    ci95 = artifact["adaptive_minus_static_ci95"]
    if not isinstance(ci95, dict):
        raise ValueError("adaptive_minus_static_ci95 must be an object")
    _validate_ci_pair(ci95.get("online"), "adaptive_minus_static_ci95")
    _validate_ci_pair(ci95.get("tier2"), "adaptive_minus_static_ci95")
    if not isinstance(artifact["family_count_vs_v395"], dict):
        raise ValueError("family_count_vs_v395 must be an object")
    if not isinstance(artifact["adaptation_curve"], list):
        raise ValueError("adaptation_curve must be a list")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if type(artifact["random_seed"]) is not int:
        raise ValueError("random_seed must be a bare int")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4283")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs do not match REQ-VERIFY-4283")


def run(
    repo_root: Path | str = Path("."),
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    adversarial_runner: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)
    output_path = root / OUTPUT_REL
    try:
        inputs = load_inputs(root)
        metrics = measure_self_learning(
            inputs.corpus,
            inputs.static_task_rows,
            feature_names=_feature_names(inputs.model_artifact),
            arcgen_used=inputs.arcgen_used,
            random_seed=random_seed,
            bootstrap_resamples=bootstrap_resamples,
        )
        checksum = reproducibility_checksum(inputs=inputs, metrics=metrics, random_seed=random_seed)
        artifact = _complete_artifact(
            inputs=inputs,
            metrics=metrics,
            checksum=checksum,
            random_seed=random_seed,
            duration_s=time.perf_counter() - start,
        )
    except BlockedRun as exc:
        artifact = _terminal_artifact(
            str(exc) or BLOCKED_INPUTS_VERDICT,
            random_seed=random_seed,
            duration_s=time.perf_counter() - start,
        )
    validate_artifact(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raw_report = (
        adversarial_runner(output_path)
        if adversarial_runner is not None
        else _run_adversarial_verify(root, output_path)
    )
    artifact["adversarial_verify"] = _clean_adversarial_report(raw_report)
    validate_artifact(artifact)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover - exercised by result entrypoint.
    repo_root = Path(__file__).resolve().parents[3]
    print(json.dumps(run(repo_root), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
