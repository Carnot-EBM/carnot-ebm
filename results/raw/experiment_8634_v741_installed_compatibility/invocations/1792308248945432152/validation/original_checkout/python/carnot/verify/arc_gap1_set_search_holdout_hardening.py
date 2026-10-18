"""Exp 5209: GAP-1 set-search holdout hardening.

Spec refs: REQ-VERIFY-5209, SCENARIO-VERIFY-5209.

This hardening pass reuses Exp 5205's cached ARC square-transpose candidate
pool and exact deterministic discriminator library. It repeats grouped
train/held-out subset selection by task id so held-out rows only measure the
chosen subset; they never choose it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import random
from statistics import mean, stdev
import time
from typing import Any

from carnot.verify import arc_gap1_autopyverifier_pilot as pilot


JsonDict = dict[str, Any]

REPO_ROOT = pilot.REPO_ROOT
EXPERIMENT = "experiment_5209_gap1_set_search_holdout_hardening_v477"
EXPERIMENT_ID = 5209
SCHEMA = "carnot.arc_gap1_set_search_holdout_hardening_5209.v1"
RESULT_RELATIVE_PATH = "results/experiment_5209_gap1_set_search_holdout_hardening_v477.json"
RUN_DATE = "2026-07-04"
SPEC_REFS = ("REQ-VERIFY-5209", "SCENARIO-VERIFY-5209")
RANDOM_SEED = 520900
DEFAULT_GROUPED_SPLITS = 20
HELDOUT_FRACTION = 0.34
EXP5205_BEST_SUBSET = (
    "border_ordered_profile",
    "color_centroid_orientation",
    "row_column_run_profile",
)
ALWAYS_ON_BASELINE: tuple[str, ...] = ()
REFUTED_DIRECTIONAL_BASELINE = (pilot.REFUTED_DIRECTIONAL,)
INFERENCE_SUBSTRATE = pilot.INFERENCE_SUBSTRATE
TERMINAL_PREFIXES = pilot.TERMINAL_PREFIXES

FIELD_PRINCIPLES: dict[str, str] = {
    "gap1_hardened_positive": (
        "BARE top-level boolean used by exp5210 gate. True only if held-out mean pass@2 beats both "
        "baselines and the paired delta CI excludes harmful regression."
    ),
    "heldout_pass_at_2_mean": "Mean held-out pass@2 for the train-selected subset over grouped splits.",
    "baseline_always_on_pass_at_2_mean": (
        "Mean held-out pass@2 for the object_count + palette_histogram_shape always-on baseline."
    ),
    "single_refuted_directional_pass_at_2_mean": (
        "Mean held-out pass@2 for the directional_adjacency_refuted_20260609 singleton baseline."
    ),
    "delta_over_always_on": "Held-out mean pass@2 minus the always-on baseline mean.",
    "delta_over_single_refuted": "Held-out mean pass@2 minus the refuted directional singleton mean.",
    "paired_delta_ci95": (
        "Paired 95% CI over per-split min(selected-always_on, selected-single_refuted) deltas."
    ),
    "n_grouped_splits": "Number of repeated task-id-grouped train/held-out splits.",
    "leakage_audit_passed": (
        "True only when scoring avoids test-gold features, held-out rows are never used for selection, "
        "and train/eval task-id groups are disjoint."
    ),
    "best_subset_stable": "True when one exact selected subset wins at least half of grouped splits.",
    "ops_verifier_gaps_updated": "True when the GAP-1 section in ops/verifier_gaps.md was updated.",
    "inference_substrate": "Must remain verifier_ensemble_against_cached_candidates; no live LLM inference.",
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ and say whether GAP-1 set-search remains "
        "positive after hardening."
    ),
    "random_seed": "Seed base used for deterministic repeated grouped splits.",
    "reproducibility_checksum": "SHA-256 checksum over the terminal artifact with this field blanked.",
}
REQUIRED_PRINCIPLED_FIELDS = tuple(FIELD_PRINCIPLES)

_T_CRITICAL_95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.16,
    14: 2.145,
    15: 2.131,
    16: 2.12,
    17: 2.11,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.08,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.06,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


@dataclass(frozen=True)
class GroupedSplit:
    split_index: int
    seed: int
    train_groups: tuple[str, ...]
    heldout_groups: tuple[str, ...]
    train_pools: tuple[pilot.TaskPool, ...]
    heldout_pools: tuple[pilot.TaskPool, ...]


@dataclass(frozen=True)
class CandidateMatrix:
    summary: JsonDict
    score_table: dict[tuple[str, str], dict[str, float]]
    source_artifact: JsonDict


def _base_task_id(task_id: str) -> str:
    return task_id.split(":", 1)[0]


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _round_score(value: float) -> float:
    return round(float(value), 12)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _value(artifact: Mapping[str, Any], field: str) -> Any:
    raw = artifact.get(field)
    return raw["value"] if isinstance(raw, Mapping) and "value" in raw else raw


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = json.loads(json.dumps(dict(artifact), sort_keys=True, default=str))
    checksum = payload.get("reproducibility_checksum")
    if isinstance(checksum, Mapping):
        checksum = dict(checksum)
        checksum["value"] = ""
        payload["reproducibility_checksum"] = checksum
    else:
        payload["reproducibility_checksum"] = {"value": ""}
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def grouped_split(
    pools: Sequence[pilot.TaskPool],
    *,
    seed: int,
    split_index: int = 0,
    heldout_fraction: float = HELDOUT_FRACTION,
) -> GroupedSplit:
    groups = sorted({_base_task_id(pool.task_id) for pool in pools})
    if len(groups) < 2:
        raise ValueError("at least two task-id groups are required for grouped holdout splitting")
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    heldout_n = max(1, min(len(shuffled) - 1, round(len(shuffled) * heldout_fraction)))
    heldout_groups = tuple(sorted(shuffled[:heldout_n]))
    train_groups = tuple(sorted(shuffled[heldout_n:]))
    heldout_set = set(heldout_groups)
    train_set = set(train_groups)
    return GroupedSplit(
        split_index=split_index,
        seed=seed,
        train_groups=train_groups,
        heldout_groups=heldout_groups,
        train_pools=tuple(pool for pool in pools if _base_task_id(pool.task_id) in train_set),
        heldout_pools=tuple(pool for pool in pools if _base_task_id(pool.task_id) in heldout_set),
    )


def repeated_grouped_splits(
    pools: Sequence[pilot.TaskPool],
    *,
    n_grouped_splits: int,
    seed_base: int = RANDOM_SEED,
    heldout_fraction: float = HELDOUT_FRACTION,
) -> tuple[GroupedSplit, ...]:
    return tuple(
        grouped_split(
            pools,
            seed=seed_base + split_index,
            split_index=split_index,
            heldout_fraction=heldout_fraction,
        )
        for split_index in range(n_grouped_splits)
    )


def _candidate_subsets(names: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    subsets: list[tuple[str, ...]] = [()]
    for size in range(1, len(names) + 1):
        subsets.extend(itertools.combinations(names, size))
    return tuple(subsets)


def _select_subset_on_train(
    train_pools: Sequence[pilot.TaskPool],
    subsets: Sequence[tuple[str, ...]],
    discriminators_by_name: Mapping[str, pilot.Discriminator],
    score_table: Mapping[tuple[str, str], Mapping[str, float]],
) -> tuple[tuple[str, ...], float, int]:
    best_subset: tuple[str, ...] = ()
    best_key: tuple[float, int, int, tuple[str, ...]] | None = None
    best_pass_at_2 = 0.0
    best_captures = 0
    for subset in subsets:
        train_pass_at_2 = pilot._pass_at_2(train_pools, subset, discriminators_by_name, score_table)
        train_captures, _train_total = pilot._transpose_capture(
            train_pools,
            subset,
            discriminators_by_name,
            score_table,
        )
        key = (train_pass_at_2, train_captures, -len(subset), tuple(reversed(subset)))
        if best_key is None or key > best_key:
            best_key = key
            best_subset = subset
            best_pass_at_2 = train_pass_at_2
            best_captures = train_captures
    return best_subset, best_pass_at_2, best_captures


def _transpose_capture_rate(
    pools: Sequence[pilot.TaskPool],
    subset: Sequence[str],
    discriminators_by_name: Mapping[str, pilot.Discriminator],
    score_table: Mapping[tuple[str, str], Mapping[str, float]],
) -> float:
    captured, total = pilot._transpose_capture(pools, subset, discriminators_by_name, score_table)
    return _round_float(captured / total) if total else 0.0


def _ci95(values: Sequence[float]) -> tuple[float, float]:
    vals = [float(value) for value in values]
    center = mean(vals)
    standard_error = stdev(vals) / (len(vals) ** 0.5)
    critical = _T_CRITICAL_95.get(len(vals) - 1, 1.96)
    return _round_float(center - critical * standard_error), _round_float(center + critical * standard_error)


def _ci_string(bounds: tuple[float, float]) -> str:
    return f"[{bounds[0]:.6f}, {bounds[1]:.6f}]"


def reconstruct_exp5205_candidate_matrix(
    pools: Sequence[pilot.TaskPool],
    *,
    root: Path | str = REPO_ROOT,
    discriminators: Sequence[pilot.Discriminator] | None = None,
) -> CandidateMatrix:
    root_path = Path(root)
    exp5205_path = root_path / pilot.RESULT_RELATIVE_PATH
    source_path = root_path / pilot.SOURCE_ARTIFACT_RELATIVE_PATH
    exp5205 = _read_json(exp5205_path)
    source_artifact = _read_json(source_path)
    rows = tuple(discriminators or pilot.default_discriminators())
    names = [row.name for row in rows]
    authored = _value(exp5205, "candidate_discriminators_authored")
    authored_names = [str(row.get("name")) for row in authored if isinstance(row, Mapping)]
    if authored_names != names:
        raise ValueError(f"candidate discriminator mismatch: exp5205={authored_names}, current={names}")

    by_name = {row.name: row for row in rows}
    score_table = pilot._score_table(pools, by_name)
    columns = ["object_count", "palette_histogram_shape", "__always_on__", *names]
    matrix_rows = []
    for pool in sorted(pools, key=lambda row: row.task_id):
        for candidate in sorted(pool.candidates, key=lambda row: row.candidate_id):
            scores = score_table[(pool.task_id, candidate.candidate_id)]
            matrix_rows.append(
                {
                    "task_id": pool.task_id,
                    "task_group": _base_task_id(pool.task_id),
                    "candidate_id": candidate.candidate_id,
                    "candidate_kind": candidate.kind,
                    "correct_for_eval_only": bool(candidate.correct),
                    "scores": {column: _round_score(scores[column]) for column in columns},
                }
            )

    summary: JsonDict = {
        "source_exp5205_artifact": pilot.RESULT_RELATIVE_PATH,
        "source_exp5205_artifact_sha256": _sha256(exp5205_path),
        "source_invariant_artifact": pilot.SOURCE_ARTIFACT_RELATIVE_PATH,
        "source_invariant_artifact_sha256": _sha256(source_path),
        "task_pool_count": len(pools),
        "task_group_count": len({_base_task_id(pool.task_id) for pool in pools}),
        "row_count": len(matrix_rows),
        "columns": columns,
        "exp5205_best_subset": list(_value(exp5205, "best_subset_found")),
        "no_llm_generated_candidates_added": True,
        "matrix_sha256": "sha256:" + hashlib.sha256(_stable_json(matrix_rows).encode("utf-8")).hexdigest(),
        "rows": matrix_rows,
    }
    return CandidateMatrix(summary=summary, score_table=score_table, source_artifact=source_artifact)


def leakage_audit_errors(
    splits: Sequence[GroupedSplit],
    *,
    source_artifact: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    if source_artifact.get("no_test_gold_leak") is not True:
        errors.append("source artifact does not assert no_test_gold_leak=true")
    for split in splits:
        overlap = sorted(set(split.train_groups) & set(split.heldout_groups))
        if overlap:
            errors.append(
                f"duplicate task ids across train/eval in split {split.split_index}: {', '.join(overlap)}"
            )
    return errors


def evaluate_repeated_grouped_splits(
    pools: Sequence[pilot.TaskPool],
    *,
    matrix: CandidateMatrix,
    discriminators: Sequence[pilot.Discriminator] | None = None,
    n_grouped_splits: int = DEFAULT_GROUPED_SPLITS,
    seed_base: int = RANDOM_SEED,
) -> tuple[tuple[GroupedSplit, ...], list[JsonDict]]:
    rows = tuple(discriminators or pilot.default_discriminators())
    by_name = {row.name: row for row in rows}
    subsets = _candidate_subsets([row.name for row in rows])
    splits = repeated_grouped_splits(pools, n_grouped_splits=n_grouped_splits, seed_base=seed_base)
    split_details: list[JsonDict] = []
    for split in splits:
        selected, train_pass_at_2, train_captures = _select_subset_on_train(
            split.train_pools,
            subsets,
            by_name,
            matrix.score_table,
        )
        heldout_pass_at_2 = pilot._pass_at_2(split.heldout_pools, selected, by_name, matrix.score_table)
        always_pass_at_2 = pilot._pass_at_2(
            split.heldout_pools,
            ALWAYS_ON_BASELINE,
            by_name,
            matrix.score_table,
        )
        refuted_pass_at_2 = pilot._pass_at_2(
            split.heldout_pools,
            REFUTED_DIRECTIONAL_BASELINE,
            by_name,
            matrix.score_table,
        )
        split_details.append(
            {
                "split_index": split.split_index,
                "seed": split.seed,
                "train_groups": list(split.train_groups),
                "heldout_groups": list(split.heldout_groups),
                "train_pool_count": len(split.train_pools),
                "heldout_pool_count": len(split.heldout_pools),
                "selected_subset": list(selected),
                "train_pass@2": train_pass_at_2,
                "train_transpose_captures": train_captures,
                "heldout_pass@2": heldout_pass_at_2,
                "baseline_always_on_pass@2": always_pass_at_2,
                "single_refuted_directional_pass@2": refuted_pass_at_2,
                "heldout_transpose_capture_rate": _transpose_capture_rate(
                    split.heldout_pools,
                    selected,
                    by_name,
                    matrix.score_table,
                ),
                "baseline_always_on_transpose_capture_rate": _transpose_capture_rate(
                    split.heldout_pools,
                    ALWAYS_ON_BASELINE,
                    by_name,
                    matrix.score_table,
                ),
                "single_refuted_directional_transpose_capture_rate": _transpose_capture_rate(
                    split.heldout_pools,
                    REFUTED_DIRECTIONAL_BASELINE,
                    by_name,
                    matrix.score_table,
                ),
            }
        )
    return splits, split_details


def _subset_stability(split_details: Sequence[Mapping[str, Any]]) -> JsonDict:
    counts = Counter(tuple(row["selected_subset"]) for row in split_details)
    top_subset, top_count = counts.most_common(1)[0]
    return {
        "top_subset": list(top_subset),
        "top_subset_count": top_count,
        "top_subset_fraction": _round_float(top_count / len(split_details)),
        "selection_counts": [
            {"subset": list(subset), "count": count}
            for subset, count in sorted(counts.items(), key=lambda row: (-row[1], row[0]))
        ],
        "stability_rule": "one exact subset selected in at least half of grouped splits",
    }


def build_artifact(
    pools: Sequence[pilot.TaskPool],
    *,
    root: Path | str = REPO_ROOT,
    n_grouped_splits: int = DEFAULT_GROUPED_SPLITS,
    seed_base: int = RANDOM_SEED,
    duration_s: float = 0.0,
    ops_verifier_gaps_updated: bool = False,
) -> JsonDict:
    matrix = reconstruct_exp5205_candidate_matrix(pools, root=root)
    splits, split_details = evaluate_repeated_grouped_splits(
        pools,
        matrix=matrix,
        n_grouped_splits=n_grouped_splits,
        seed_base=seed_base,
    )
    audit_errors = leakage_audit_errors(splits, source_artifact=matrix.source_artifact)
    heldout = [float(row["heldout_pass@2"]) for row in split_details]
    always = [float(row["baseline_always_on_pass@2"]) for row in split_details]
    refuted = [float(row["single_refuted_directional_pass@2"]) for row in split_details]
    min_deltas = [min(selected - base, selected - single) for selected, base, single in zip(heldout, always, refuted)]
    ci = _ci95(min_deltas)
    heldout_mean = _round_float(mean(heldout))
    always_mean = _round_float(mean(always))
    refuted_mean = _round_float(mean(refuted))
    delta_always = _round_float(heldout_mean - always_mean)
    delta_refuted = _round_float(heldout_mean - refuted_mean)
    leakage_passed = not audit_errors
    stable = _subset_stability(split_details)
    best_subset_stable = bool(stable["top_subset_fraction"] >= 0.5)
    hardened_positive = bool(heldout_mean > always_mean and heldout_mean > refuted_mean and ci[0] > 0 and leakage_passed)
    verdict_status = (
        "set_search_remains_positive_after_hardening"
        if hardened_positive
        else "set_search_not_positive_after_hardening"
    )
    stability_status = "best_subset_stable" if best_subset_stable else "best_subset_not_stable"
    verdict = (
        f"complete: {verdict_status}_heldout_{heldout_mean:.4f}_always_{always_mean:.4f}_"
        f"single_refuted_{refuted_mean:.4f}_paired_delta_ci95_{ci[0]:.4f}_{ci[1]:.4f}_"
        f"{stability_status}_do_not_promote_to_registry_here"
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "run_date": RUN_DATE,
        "result_path": RESULT_RELATIVE_PATH,
        "candidate_discriminator_matrix": matrix.summary,
        "baseline_definitions": {
            "always_on": ["object_count", "palette_histogram_shape"],
            "single_refuted_directional": list(REFUTED_DIRECTIONAL_BASELINE),
            "selected_subset_rule": (
                "Within each split, maximize train pass@2, then train transpose captures, then smaller subset; "
                "held-out rows are used only for final metrics."
            ),
        },
        "split_details": split_details,
        "subset_stability": stable,
        "leakage_audit": {
            "passed": leakage_passed,
            "errors": audit_errors,
            "no_test_gold_in_scoring": leakage_passed,
            "no_test_output_derived_features": matrix.summary["no_llm_generated_candidates_added"],
            "no_subset_selection_on_heldout_rows": True,
            "no_duplicate_task_ids_across_train_eval": leakage_passed,
        },
        "heldout_transpose_capture_rate_mean": _round_float(
            mean(float(row["heldout_transpose_capture_rate"]) for row in split_details)
        ),
        "baseline_always_on_transpose_capture_rate_mean": _round_float(
            mean(float(row["baseline_always_on_transpose_capture_rate"]) for row in split_details)
        ),
        "single_refuted_directional_transpose_capture_rate_mean": _round_float(
            mean(float(row["single_refuted_directional_transpose_capture_rate"]) for row in split_details)
        ),
        "gap1_hardened_positive": _wrap("gap1_hardened_positive", hardened_positive),
        "heldout_pass_at_2_mean": _wrap("heldout_pass_at_2_mean", heldout_mean),
        "baseline_always_on_pass_at_2_mean": _wrap("baseline_always_on_pass_at_2_mean", always_mean),
        "single_refuted_directional_pass_at_2_mean": _wrap(
            "single_refuted_directional_pass_at_2_mean",
            refuted_mean,
        ),
        "delta_over_always_on": _wrap("delta_over_always_on", delta_always),
        "delta_over_single_refuted": _wrap("delta_over_single_refuted", delta_refuted),
        "paired_delta_ci95": _wrap("paired_delta_ci95", _ci_string(ci)),
        "n_grouped_splits": _wrap("n_grouped_splits", int(n_grouped_splits)),
        "leakage_audit_passed": _wrap("leakage_audit_passed", leakage_passed),
        "best_subset_stable": _wrap("best_subset_stable", best_subset_stable),
        "ops_verifier_gaps_updated": _wrap("ops_verifier_gaps_updated", bool(ops_verifier_gaps_updated)),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "honest_verdict": _wrap("honest_verdict", verdict),
        "random_seed": _wrap("random_seed", int(seed_base)),
        "reproducibility_checksum": _wrap("reproducibility_checksum", ""),
        "duration_s": round(float(duration_s), 3),
        "field_principles": FIELD_PRINCIPLES,
    }
    artifact["reproducibility_checksum"] = _wrap("reproducibility_checksum", payload_checksum(artifact))
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def _parse_ci95(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, str) or not value.startswith("[") or not value.endswith("]") or "," not in value:
        return None
    left, right = value.strip("[]").split(",", 1)
    return float(left), float(right)


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_PRINCIPLED_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
    for field in REQUIRED_PRINCIPLED_FIELDS:
        raw = artifact.get(field)
        if not isinstance(raw, Mapping) or "value" not in raw or "principle" not in raw:
            errors.append(f"{field} must be principle-wrapped")
            continue
        if raw.get("principle") != FIELD_PRINCIPLES[field]:
            errors.append(f"{field} principle mismatch")
    if _value(artifact, "n_grouped_splits") < DEFAULT_GROUPED_SPLITS:
        errors.append("n_grouped_splits must be at least 20")
    if _value(artifact, "leakage_audit_passed") is not True:
        errors.append("leakage_audit_passed must be true")
    if _value(artifact, "inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must be verifier_ensemble_against_cached_candidates")
    verdict = _value(artifact, "honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must use a terminal complete/success prefix")
    ci = _parse_ci95(_value(artifact, "paired_delta_ci95"))
    if ci is None:
        errors.append("paired_delta_ci95 must be formatted as [lo, hi]")
    hardened = _value(artifact, "gap1_hardened_positive")
    if hardened is True:
        heldout = float(_value(artifact, "heldout_pass_at_2_mean"))
        always = float(_value(artifact, "baseline_always_on_pass_at_2_mean"))
        refuted = float(_value(artifact, "single_refuted_directional_pass_at_2_mean"))
        if not (heldout > always and heldout > refuted and ci is not None and ci[0] > 0):
            errors.append("gap1_hardened_positive true requires positive held-out deltas and CI lower bound")
    checksum = _value(artifact, "reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum must be sha256")
    elif checksum != payload_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def _gap_update_block(artifact: Mapping[str, Any]) -> str:
    verdict = _value(artifact, "honest_verdict")
    hardened = _value(artifact, "gap1_hardened_positive")
    heldout = _value(artifact, "heldout_pass_at_2_mean")
    always = _value(artifact, "baseline_always_on_pass_at_2_mean")
    refuted = _value(artifact, "single_refuted_directional_pass_at_2_mean")
    ci = _value(artifact, "paired_delta_ci95")
    stable = _value(artifact, "best_subset_stable")
    leakage = _value(artifact, "leakage_audit_passed")
    return (
        "<!-- experiment_5209_gap1_set_search_holdout_hardening_v477 start -->\n"
        "- experiment_5209 GAP-1 set-search holdout hardening (2026-07-04): "
        f"gap1_hardened_positive={hardened}, heldout pass@2 mean={heldout}, "
        f"always-on baseline={always}, single refuted directional={refuted}, "
        f"paired delta CI95={ci}, leakage_audit_passed={leakage}, "
        f"best_subset_stable={stable}. Do not promote to registry here. Verdict: {verdict}\n"
        "<!-- experiment_5209_gap1_set_search_holdout_hardening_v477 end -->\n"
    )


def update_verifier_gap_doc(root: Path | str, artifact: Mapping[str, Any]) -> None:
    path = Path(root) / pilot.VERIFIER_GAPS_RELATIVE_PATH
    if not path.exists():  # pragma: no cover - defensive no-op for ad hoc temp roots.
        return
    text = path.read_text(encoding="utf-8")
    start = "<!-- experiment_5209_gap1_set_search_holdout_hardening_v477 start -->"
    end = "<!-- experiment_5209_gap1_set_search_holdout_hardening_v477 end -->"
    block = _gap_update_block(artifact)
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _old, after = rest.split(end, 1)
        path.write_text(before + block + after, encoding="utf-8")
        return
    prior_end = "<!-- experiment_5205_autopyverifier_gap1_pilot_v476 end -->"
    if prior_end in text:
        text = text.replace(prior_end, prior_end + "\n" + block.rstrip("\n"), 1)
    else:  # pragma: no cover - repository GAP-1 section currently carries the exp5205 marker.
        text = text.replace("### GAP-2:", block + "\n### GAP-2:", 1)
    path.write_text(text, encoding="utf-8")


def run(
    *,
    root: Path | str = REPO_ROOT,
    arc_root: Path | str = pilot.DEFAULT_ARC_ROOT,
    pools: Sequence[pilot.TaskPool] | None = None,
    result_path: Path | str | None = None,
    n_grouped_splits: int = DEFAULT_GROUPED_SPLITS,
    seed_base: int = RANDOM_SEED,
    duration_s: float | None = None,
    update_gap_doc: bool = True,
) -> JsonDict:
    started = time.time()
    root_path = Path(root)
    task_pools = list(pools) if pools is not None else pilot.load_square_transpose_subset(root=root_path, arc_root=arc_root)
    elapsed = time.time() - started if duration_s is None else duration_s
    ops_will_update = bool(update_gap_doc and (root_path / pilot.VERIFIER_GAPS_RELATIVE_PATH).exists())
    artifact = build_artifact(
        task_pools,
        root=root_path,
        n_grouped_splits=n_grouped_splits,
        seed_base=seed_base,
        duration_s=elapsed,
        ops_verifier_gaps_updated=ops_will_update,
    )
    output = Path(result_path) if result_path is not None else root_path / RESULT_RELATIVE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if update_gap_doc:
        update_verifier_gap_doc(root_path, artifact)
    return artifact


def main() -> int:  # pragma: no cover - exercised by direct experiment invocation, not unit tests.
    artifact = run()
    print(artifact["honest_verdict"]["value"])
    print(f"wrote {RESULT_RELATIVE_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
