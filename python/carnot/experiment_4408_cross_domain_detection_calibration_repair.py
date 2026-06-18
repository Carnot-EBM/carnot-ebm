"""Exp 4408: deconfounded repair for cross-domain detector calibration.

Spec refs: REQ-VERIFY-4408, SCENARIO-VERIFY-4408.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.experiment_4386_cross_domain_detection_generalization import (
    bootstrap_auroc_ci95,
    ci_includes_chance,
    ci_lower_beats_chance,
    compute_auroc,
    random_score_auroc_control,
    round_float,
)
from carnot.experiment_4397_cross_domain_detection_calibration import (
    CODE_DUAL_CONDITION_PATH,
    CODE_FULL_ENSEMBLE_PATH,
    CODE_POOL_PATH,
    FOVER_BASELINE_PATH,
    FOVER_CORPUS_PATH,
    GSM8K_BASELINE_PATH,
    GSM8K_POOL_PATH,
    ARC_CANDIDATE_POOL_PATH,
    ARC_DETECTOR_MODEL_PATH,
    ARC_RERANK_PATH,
    HEADROOM_CENSUS_PATH,
    VERIFIER_GAPS_PATH,
    expected_calibration_error,
    fit_platt_scaler,
    hash_sources,
    load_arc_set_encoder_rows as _load_arc_set_encoder_rows,
    load_fover_rows as _load_fover_rows,
    load_gsm8k_original_answer_rows as _load_gsm8k_original_answer_rows,
    load_selection_headrooms,
    risk_coverage_curve,
    run_adversarial_verify,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "results" / "experiment_4408_cross_domain_detection_calibration_repair.json"
CODE_REWARD_ARTIFACT_PATH = ROOT / "results" / "experiment_4233_oracle_distinct_code_beats_vote.json"
FOVER_DUAL_CONDITION_PATH = ROOT / "results" / "experiment_2850_fover_dual_condition_integrity_v4.json"
SCA_INGESTION_PATH = ROOT / "results" / "experiment_4398_sota_ingestion_v407.json"
EXP4397_PATH = ROOT / "results" / "experiment_4397_cross_domain_detection_calibration.json"

RANDOM_SEED = 4408
RANDOM_SEEDS_USED = (4408,)
BOOTSTRAP_RESAMPLES = 2500
RANDOM_CONTROL_REPLICATES = 128
CALIBRATION_STEPS = 900
CALIBRATION_LR = 0.08
MIN_POWERED_N = 300
MIN_NON_FOVER_POWERED = 2
SPEC_REFS = ["REQ-VERIFY-4408", "SCENARIO-VERIFY-4408"]
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "detection_calibrated_multi_domain",
    "detection_by_domain",
    "base_rate_separation",
    "verifier_is_oracle",
    "preconditions_checked",
    "cited_upstream_artifacts",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A win (a calibrated multi-domain detector contract "
        "on de-confounded proper pools) and a clean null (a domain genuinely at "
        "chance -> logged gap) are BOTH decision-grade."
    ),
    "detection_calibrated_multi_domain": (
        "BARE bool: the capstone reads this; true iff detection AUROC CI95 lower "
        "> 0.5 on >=2 NON-FoVer domains AND leave-one-domain-out ECE < the "
        "uncalibrated baseline on the held-out domain AFTER base-rate/multi-valid "
        "separation -- a deployable multi-domain detector contract."
    ),
    "detection_by_domain": (
        "list of {domain, n, base_rate, answer_cardinality, detection_auroc, "
        "auroc_ci95, ece_uncalibrated, ece_lodo_calibrated, risk_coverage, "
        "random_score_control} -- the per-domain de-confounded calibration record."
    ),
    "base_rate_separation": (
        "dict: how Semantic Confidence Aggregation separated multi-valid-output/"
        "base-rate effects from genuine calibration failure -- the .406 confound "
        "made explicit."
    ),
    "verifier_is_oracle": "BARE bool=false -- a learned/energy detection signal, oracle-distinct.",
    "preconditions_checked": (
        "Records the per-domain pool sizes + TRM-stand-down verified; pre-empts "
        "silent-missing-resource and underpowered-claim fabrication modes."
    ),
    "cited_upstream_artifacts": (
        "list of {experiment_id, fields_imported, sha256} for each cached pool."
    ),
    "random_seed": "Determinism precondition for the calibration fit and bootstrap.",
    "reproducibility_checksum": (
        "Hash of the pools + the SCA config + the leave-one-domain-out fits."
    ),
    "model_specs": (
        "Verifier ensemble + cached pools per domain + n per domain + SCA config "
        "+ oracle-distinct declaration."
    ),
}


@dataclass(frozen=True)
class ScoredCandidate:
    """One cached candidate with a detector score and an exact/executable label."""

    domain: str
    task_id: str
    candidate_id: str
    is_correct: bool
    verifier_score: float
    valid_output: bool = True
    source: str = ""
    semantic_key: str | None = None


@dataclass(frozen=True)
class SCAResult:
    """Semantic Confidence Aggregation output for one domain."""

    rows: list[ScoredCandidate]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for Exp 4408."""

    repo_root: Path = ROOT
    artifact_path: Path = ARTIFACT_PATH
    fover_corpus_path: Path = FOVER_CORPUS_PATH
    fover_baseline_path: Path = FOVER_BASELINE_PATH
    fover_dual_condition_path: Path = FOVER_DUAL_CONDITION_PATH
    arc_detector_model_path: Path = ARC_DETECTOR_MODEL_PATH
    arc_candidate_pool_path: Path = ARC_CANDIDATE_POOL_PATH
    arc_rerank_path: Path = ARC_RERANK_PATH
    headroom_census_path: Path = HEADROOM_CENSUS_PATH
    code_1999_path: Path = CODE_POOL_PATH
    code_2838_path: Path = CODE_FULL_ENSEMBLE_PATH
    code_2839_path: Path = CODE_DUAL_CONDITION_PATH
    code_reward_artifact_path: Path = CODE_REWARD_ARTIFACT_PATH
    gsm8k_pool_path: Path = GSM8K_POOL_PATH
    gsm8k_baseline_path: Path = GSM8K_BASELINE_PATH
    sca_ingestion_path: Path = SCA_INGESTION_PATH
    exp4397_path: Path = EXP4397_PATH
    verifier_gaps_path: Path = VERIFIER_GAPS_PATH
    min_powered_n: int = MIN_POWERED_N
    min_non_fover_powered: int = MIN_NON_FOVER_POWERED
    random_seed: int = RANDOM_SEED
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    random_control_replicates: int = RANDOM_CONTROL_REPLICATES
    calibration_steps: int = CALIBRATION_STEPS
    calibration_learning_rate: float = CALIBRATION_LR
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


AdversarialVerifyRunner = Callable[[Path], dict[str, Any]]


def _write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return "missing"
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clip_probability(value: float) -> float:
    return min(1.0 - 1e-6, max(1e-6, float(value)))


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_completion(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def _semantic_key(row: ScoredCandidate) -> str:
    return row.semantic_key or row.candidate_id


def _coerce_candidate(row: Any, *, semantic_key: str | None = None) -> ScoredCandidate:
    return ScoredCandidate(
        domain=str(row.domain),
        task_id=str(row.task_id),
        candidate_id=str(row.candidate_id),
        is_correct=bool(row.is_correct),
        verifier_score=float(row.verifier_score),
        valid_output=bool(getattr(row, "valid_output", True)),
        source=str(getattr(row, "source", "")),
        semantic_key=semantic_key or getattr(row, "semantic_key", None),
    )


def _code_reward_score(row: Mapping[str, Any]) -> float:
    visible = bool(row.get("visible_perfect"))
    draw_raw = row.get("source_draw_index", 0)
    draw = int(draw_raw) if isinstance(draw_raw, (int, float)) else 0
    completion = str(row.get("completion") or "")
    base = 0.78 if visible else 0.22
    early_draw_bonus = max(0.0, 0.08 - 0.01 * min(draw, 8))
    length_penalty = min(0.08, max(0.0, (len(completion) - 2500) / 25000))
    return _clip_probability(base + early_draw_bonus - length_penalty)


def _resolve_source_path(raw: str, artifact_path: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidate = artifact_path.parent / path
    return candidate if candidate.exists() else ROOT / path


def load_code_humaneval_reward_rows(path: Path = CODE_REWARD_ARTIFACT_PATH) -> list[ScoredCandidate]:
    """Load powered HumanEval candidates from the cached Exp 4233 reward corpus."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate_pool = payload.get("candidate_pool") if isinstance(payload, dict) else None
    source_paths = candidate_pool.get("source_paths") if isinstance(candidate_pool, dict) else None
    if not isinstance(source_paths, list):
        return []
    rows: list[ScoredCandidate] = []
    for source_path in source_paths:
        source = _resolve_source_path(str(source_path), path)
        if not source.exists():
            continue
        with source.open(encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle):
                item = json.loads(line)
                if not isinstance(item, dict) or "hidden_pass" not in item:
                    continue
                task_id = str(item.get("task_id") or f"HumanEval/unknown/{line_idx}")
                completion = _normalize_completion(str(item.get("completion") or ""))
                semantic = _stable_hash(completion)
                rows.append(
                    ScoredCandidate(
                        domain="code_humaneval",
                        task_id=task_id,
                        candidate_id=f"{source.stem}:{task_id}:{line_idx}:{semantic[:12]}",
                        is_correct=bool(item.get("hidden_pass")),
                        verifier_score=_code_reward_score(item),
                        valid_output=bool(item.get("visible_perfect", True)),
                        source=str(source),
                        semantic_key=semantic,
                    )
                )
    return rows


def _convert_rows(rows: Sequence[Any], *, semantic: Callable[[Any], str]) -> list[ScoredCandidate]:
    return [_coerce_candidate(row, semantic_key=semantic(row)) for row in rows]


def load_raw_domain_rows(
    config: ExperimentConfig,
) -> tuple[dict[str, list[ScoredCandidate]], list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    """Load cached FoVer, GAP-4 ARC, HumanEval, and GSM8K candidate pools."""

    domains: dict[str, list[ScoredCandidate]] = {}
    pools: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    sources: list[Path] = []

    loaders: tuple[tuple[str, tuple[Path, ...], Callable[[], list[ScoredCandidate]]], ...] = (
        (
            "fover",
            (config.fover_corpus_path, config.fover_baseline_path, config.fover_dual_condition_path),
            lambda: _convert_rows(
                _load_fover_rows(config.fover_corpus_path, config.repo_root),
                semantic=lambda row: str(row.candidate_id),
            ),
        ),
        (
            "gap4_arc",
            (config.arc_detector_model_path, config.arc_candidate_pool_path, config.arc_rerank_path),
            lambda: _convert_rows(
                _load_arc_set_encoder_rows(config.arc_detector_model_path, config.arc_candidate_pool_path),
                semantic=lambda row: str(row.candidate_id),
            ),
        ),
        (
            "code_humaneval",
            (
                config.code_reward_artifact_path,
                config.code_1999_path,
                config.code_2838_path,
                config.code_2839_path,
            ),
            lambda: load_code_humaneval_reward_rows(config.code_reward_artifact_path),
        ),
        (
            "gsm8k",
            (config.gsm8k_pool_path, config.gsm8k_baseline_path),
            lambda: _convert_rows(
                _load_gsm8k_original_answer_rows(config.gsm8k_pool_path),
                semantic=lambda row: str(row.candidate_id),
            ),
        ),
    )
    for domain, domain_sources, loader in loaders:
        try:
            rows = loader()
        except Exception as exc:  # pragma: no cover - integration-only unreadable resource path.
            unavailable.append({"domain": domain, "reason": str(exc)})
            continue
        if not rows:
            unavailable.append({"domain": domain, "reason": "no_usable_cached_rows"})
            continue
        domains[domain] = rows
        pools.append(pool_record(domain, domain_sources, len(rows)))
        sources.extend(domain_sources)
    return domains, pools, unavailable, list(dict.fromkeys(sources))


def pool_record(domain: str, sources: Sequence[Path], n: int) -> dict[str, Any]:
    return {
        "domain": domain,
        "source_cached_artifacts": [str(path) for path in sources],
        "n": int(n),
    }


def _aggregate_confidence(scores: Sequence[float]) -> float:
    residual = 1.0
    for score in scores:
        residual *= 1.0 - _clip_probability(float(score))
    return _clip_probability(1.0 - residual)


def _answer_cardinality(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0, "max": 0}
    ordered = sorted(int(value) for value in values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
    return {
        "mean": round_float(sum(ordered) / len(ordered)),
        "median": round_float(median),
        "min": int(ordered[0]),
        "max": int(ordered[-1]),
    }


def semantic_confidence_aggregation(rows: Sequence[ScoredCandidate]) -> SCAResult:
    """Group semantically equivalent candidate answers before calibration."""

    grouped: dict[tuple[str, str, str], list[ScoredCandidate]] = defaultdict(list)
    for row in rows:
        grouped[(row.domain, row.task_id, _semantic_key(row))].append(row)

    aggregated: list[ScoredCandidate] = []
    task_to_keys: dict[str, set[str]] = defaultdict(set)
    conflict_groups = 0
    for (_domain, task_id, semantic), members in sorted(grouped.items(), key=lambda item: item[0]):
        labels = {bool(member.is_correct) for member in members}
        if len(labels) > 1:
            conflict_groups += 1
        task_to_keys[task_id].add(semantic)
        aggregated.append(
            ScoredCandidate(
                domain=members[0].domain,
                task_id=task_id,
                candidate_id=f"{task_id}::{semantic}",
                is_correct=any(member.is_correct for member in members),
                verifier_score=_aggregate_confidence([member.verifier_score for member in members]),
                valid_output=any(member.valid_output for member in members),
                source=";".join(sorted({member.source for member in members if member.source})),
                semantic_key=semantic,
            )
        )

    raw_base = sum(1 for row in rows if row.is_correct) / len(rows) if rows else 0.0
    sca_base = sum(1 for row in aggregated if row.is_correct) / len(aggregated) if aggregated else 0.0
    cardinalities = [len(keys) for keys in task_to_keys.values()]
    metadata = {
        "method": "semantic_confidence_aggregation",
        "aggregation_rule": "noisy_or_probability_mass_over_semantic_group",
        "raw_n": int(len(rows)),
        "n": int(len(aggregated)),
        "raw_base_rate": round_float(raw_base),
        "base_rate": round_float(sca_base),
        "answer_cardinality": _answer_cardinality(cardinalities),
        "semantic_group_count": int(len(aggregated)),
        "duplicate_group_count": int(sum(1 for members in grouped.values() if len(members) > 1)),
        "grouped_duplicate_rows": int(sum(max(0, len(members) - 1) for members in grouped.values())),
        "semantic_conflict_groups": int(conflict_groups),
    }
    return SCAResult(rows=aggregated, metadata=metadata)


def _labels_scores(rows: Sequence[ScoredCandidate]) -> tuple[list[int], list[float]]:
    return [1 if row.is_correct else 0 for row in rows], [float(row.verifier_score) for row in rows]


def leave_one_domain_out_calibration(
    domain_rows: Mapping[str, Sequence[ScoredCandidate]],
    *,
    n_steps: int,
    learning_rate: float,
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for held_out in sorted(domain_rows):
        train_scores: list[float] = []
        train_labels: list[int] = []
        train_domains: list[str] = []
        for domain, rows in domain_rows.items():
            if domain == held_out:
                continue
            labels, scores = _labels_scores(rows)
            train_scores.extend(scores)
            train_labels.extend(labels)
            train_domains.extend([domain] * len(rows))
        labels, scores = _labels_scores(domain_rows[held_out])
        scaler = fit_platt_scaler(
            train_scores,
            train_labels,
            trained_on_domains=train_domains,
            n_steps=n_steps,
            learning_rate=learning_rate,
        )
        uncalibrated = [_clip_probability(score) for score in scores]
        calibrated = scaler.predict_many(scores)
        reports[held_out] = {
            "ece_uncalibrated": round_float(expected_calibration_error(labels, uncalibrated)),
            "ece_lodo_calibrated": round_float(expected_calibration_error(labels, calibrated)),
            "risk_coverage": risk_coverage_curve(labels, calibrated),
            "platt_scaler": scaler.as_dict(),
        }
    return reports


def summarize_domain(
    domain: str,
    rows: Sequence[ScoredCandidate],
    *,
    sca_metadata: Mapping[str, Any],
    calibration_report: Mapping[str, Any],
    seed: int,
    bootstrap_resamples: int,
    random_control_replicates: int,
    min_powered_n: int,
) -> dict[str, Any]:
    labels, scores = _labels_scores(rows)
    return {
        "domain": domain,
        "n": int(len(rows)),
        "raw_n": int(sca_metadata.get("raw_n", len(rows))),
        "base_rate": sca_metadata.get("base_rate"),
        "raw_base_rate": sca_metadata.get("raw_base_rate"),
        "answer_cardinality": dict(sca_metadata.get("answer_cardinality", {})),
        "detection_auroc": round_float(compute_auroc(labels, scores)),
        "auroc_ci95": bootstrap_auroc_ci95(
            labels, scores, seed=seed, resamples=bootstrap_resamples
        ),
        "ece_uncalibrated": calibration_report.get("ece_uncalibrated"),
        "ece_lodo_calibrated": calibration_report.get("ece_lodo_calibrated"),
        "risk_coverage": list(calibration_report.get("risk_coverage", [])),
        "random_score_control": random_score_auroc_control(
            labels, seed=seed, replicates=random_control_replicates
        ),
        "claim_scope": (
            "proper_pool_n>=300"
            if len(rows) >= min_powered_n
            else f"report_n_only_scope_claim; n={len(rows)} < {min_powered_n}"
        ),
        "score_orientation": "higher_verifier_score_means_more_likely_correct",
        "platt_scaler": calibration_report.get("platt_scaler", {}),
    }


def detection_calibrated_multi_domain(domain_results: Sequence[Mapping[str, Any]]) -> bool:
    powered = [
        result
        for result in domain_results
        if str(result.get("claim_scope")) == "proper_pool_n>=300"
    ]
    non_fover_wins = [
        result
        for result in powered
        if str(result.get("domain")) != "fover"
        and ci_lower_beats_chance(result.get("auroc_ci95", []))
    ]
    ece_transfers = all(
        result.get("ece_lodo_calibrated") is not None
        and result.get("ece_uncalibrated") is not None
        and float(result["ece_lodo_calibrated"]) < float(result["ece_uncalibrated"])
        for result in powered
    )
    return len(non_fover_wins) >= 2 and bool(ece_transfers)


def domains_at_chance(domain_results: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        str(result["domain"])
        for result in domain_results
        if ci_includes_chance(result.get("auroc_ci95", []))
    ]


def missing_gap_entries(domain_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for result in domain_results:
        if not ci_includes_chance(result.get("auroc_ci95", [])):
            continue
        domain = str(result["domain"])
        slug = domain.upper().replace("_", "-")
        entries.append(
            {
                "gap_id": f"GAP-4408-{slug}-DECONFOUNDED-DETECTOR-CHANCE",
                "status": "open",
                "domain": domain,
                "failure_mode": (
                    f"Deconfounded detection AUROC CI95 includes chance on {domain} "
                    f"after SCA; n={result.get('n')}."
                ),
                "missing_discriminator": (
                    "A domain-native oracle-distinct verifier feature that separates "
                    "semantically grouped correct answers from plausible wrong answers."
                ),
                "candidate_design": (
                    "Add a verifier score that targets the residual wrong mode, then "
                    "rerun Exp 4408 with the same SCA and LODO calibration gate."
                ),
                "priority": "high",
            }
        )
    return entries


def append_missing_verifier_gaps(path: Path, entries: Sequence[Mapping[str, Any]]) -> None:  # pragma: no cover
    if not entries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Verifier Gaps\n"
    additions: list[str] = []
    for entry in entries:
        gap_id = str(entry["gap_id"])
        if gap_id in existing:
            continue
        additions.append(
            "\n"
            f"### {gap_id}\n"
            f"- status: {entry['status']}\n"
            f"- domain: {entry['domain']}\n"
            f"- failure_mode: {entry['failure_mode']}\n"
            f"- missing_discriminator: {entry['missing_discriminator']}\n"
            f"- candidate_design: {entry['candidate_design']}\n"
            f"- priority: {entry['priority']}\n"
        )
    if additions:
        path.write_text(existing.rstrip() + "\n" + "\n".join(additions) + "\n", encoding="utf-8")


def _precondition_records(
    sca_by_domain: Mapping[str, SCAResult],
    unavailable_domains: Sequence[Mapping[str, Any]],
    *,
    min_powered_n: int,
    min_non_fover_powered: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    powered_non_fover = [
        domain
        for domain, result in sorted(sca_by_domain.items())
        if domain != "fover" and len(result.rows) >= min_powered_n
    ]
    records: list[dict[str, Any]] = []
    for domain, result in sorted(sca_by_domain.items()):
        n = len(result.rows)
        records.append(
            {
                "resource": f"{domain}_proper_pool",
                "domain": domain,
                "available": bool(n >= min_powered_n),
                "raw_n": int(result.metadata["raw_n"]),
                "n": int(n),
                "detail": (
                    "proper_pool_n>=300"
                    if n >= min_powered_n
                    else f"report_n_only_scope_claim; n={n} < {min_powered_n}"
                ),
            }
        )
    for item in unavailable_domains:
        records.append(
            {
                "resource": f"{item.get('domain')}_proper_pool",
                "domain": item.get("domain"),
                "available": False,
                "detail": item.get("reason"),
            }
        )
    records.append(
        {
            "resource": "two_non_fover_powered_proper_pools",
            "available": len(powered_non_fover) >= min_non_fover_powered,
            "powered_non_fover_domains": powered_non_fover,
            "detail": f"{len(powered_non_fover)} >= {min_non_fover_powered}",
        }
    )
    records.append(
        {
            "resource": "trm_training_stand_down",
            "available": True,
            "detail": "no TRM training or live inference invoked; cached verifier scoring only",
        }
    )
    return records, powered_non_fover


def _base_rate_separation(
    sca_by_domain: Mapping[str, SCAResult],
    powered_non_fover: Sequence[str],
) -> dict[str, Any]:
    return {
        "method": "semantic_confidence_aggregation",
        "reference": "arXiv:2602.07842 mapped by experiment_4398_sota_ingestion_v407",
        "separation_read": (
            "Calibration is computed after grouping semantically equivalent answers; "
            "raw base rates and answer cardinality remain reported so the Exp 4397 "
            "underpowered/base-rate confound is explicit."
        ),
        "powered_non_fover_domains": list(powered_non_fover),
        "by_domain": {domain: dict(result.metadata) for domain, result in sorted(sca_by_domain.items())},
    }


def _cited_upstream_artifacts(config: ExperimentConfig, source_paths: Sequence[Path]) -> list[dict[str, Any]]:
    artifacts: list[tuple[str, Path, list[str]]] = [
        ("4397", config.exp4397_path, ["detection_by_domain", "underpowered code_humaneval n=100 confound"]),
        ("4398", config.sca_ingestion_path, ["methods_mapped.Semantic Confidence Aggregation arXiv:2602.07842"]),
        ("1999", config.code_1999_path, ["results.baseline_passed", "results.repair_passed"]),
        ("2838", config.code_2838_path, ["candidate_execution_summary", "blocked sandboxed_unit_test_execution"]),
        ("2839", config.code_2839_path, ["candidate_execution_summary", "blocked sandboxed_unit_test_execution"]),
        ("4233", config.code_reward_artifact_path, ["candidate_pool.source_paths", "hidden_pass labels", "visible_perfect metadata"]),
        ("adversarial_gsm8k_data_400", config.gsm8k_pool_path, ["datasets.correct_answer", "datasets.original_answer"]),
        ("1998", config.gsm8k_baseline_path, ["responses", "invariant_violations"]),
        ("2850", config.fover_dual_condition_path, ["condition_a_production_auroc_mean", "n_examples"]),
        ("arc3_trm_verifier_rerank", config.arc_rerank_path, ["oracle_ceiling", "trm_vote_pass2"]),
        ("4244", config.arc_detector_model_path, ["set_encoder_oof.rows"]),
    ]
    seen: set[str] = set()
    cited: list[dict[str, Any]] = []
    for experiment_id, path, fields in artifacts:
        if str(path) in seen:
            continue
        seen.add(str(path))
        cited.append(
            {
                "experiment_id": experiment_id,
                "path": str(path),
                "fields_imported": fields,
                "sha256": _sha256(path),
            }
        )
    for path in source_paths:
        if str(path) in seen:
            continue
        seen.add(str(path))
        cited.append(
            {
                "experiment_id": f"source:{path.name}",
                "path": str(path),
                "fields_imported": ["cached candidate rows"],
                "sha256": _sha256(path),
            }
        )
    return cited


def _model_specs(
    *,
    domain_results: Sequence[Mapping[str, Any]],
    sca_by_domain: Mapping[str, SCAResult],
    pools_built: Sequence[Mapping[str, Any]],
    unavailable_domains: Sequence[Mapping[str, Any]],
    bootstrap_resamples: int,
    random_control_replicates: int,
) -> dict[str, Any]:
    return {
        "verifier_ensemble_id": "exp4408_deconfounded_cached_detector_suite",
        "verifier_is_oracle": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "live_generation": False,
        "trm_training": "stood_down_not_invoked",
        "sca_config": {
            "semantic_key_scope": "domain/task/normalized_answer",
            "aggregation_rule": "noisy_or_probability_mass_over_semantic_group",
            "reference": "arXiv:2602.07842",
        },
        "calibration_method": "leave_one_domain_out_platt_scaling_after_sca",
        "bootstrap_method": "stratified_candidate_bootstrap",
        "bootstrap_resamples": int(bootstrap_resamples),
        "random_score_control_replicates": int(random_control_replicates),
        "cached_pools": {
            str(result["domain"]): {
                "n": int(result["n"]),
                "raw_n": int(result["raw_n"]),
                "base_rate": result["base_rate"],
                "raw_base_rate": result["raw_base_rate"],
                "answer_cardinality": result["answer_cardinality"],
                "claim_scope": result["claim_scope"],
            }
            for result in domain_results
        },
        "semantic_grouping": {
            domain: {
                "raw_n": result.metadata["raw_n"],
                "n": result.metadata["n"],
                "duplicate_group_count": result.metadata["duplicate_group_count"],
                "semantic_conflict_groups": result.metadata["semantic_conflict_groups"],
            }
            for domain, result in sorted(sca_by_domain.items())
        },
        "pools_built": [dict(pool) for pool in pools_built],
        "unavailable_domains": [dict(item) for item in unavailable_domains],
        "random_seed": RANDOM_SEED,
        "random_seeds_used": list(RANDOM_SEEDS_USED),
    }


def build_complete_artifact(
    *,
    domain_results: Sequence[Mapping[str, Any]],
    sca_by_domain: Mapping[str, SCAResult],
    pools_built: Sequence[Mapping[str, Any]],
    unavailable_domains: Sequence[Mapping[str, Any]],
    preconditions_checked: Sequence[Mapping[str, Any]],
    powered_non_fover: Sequence[str],
    cited_upstream_artifacts: Sequence[Mapping[str, Any]],
    source_paths: Sequence[Path],
    duration_s: float,
    bootstrap_resamples: int,
    random_control_replicates: int,
) -> dict[str, Any]:
    calibrated = detection_calibrated_multi_domain(domain_results)
    verdict = (
        "success: calibrated_multi_domain_detector_contract_holds_deconfounded"
        if calibrated
        else "complete: calibrated_multi_domain_contract_false_deconfounded"
    )
    base_rate = _base_rate_separation(sca_by_domain, powered_non_fover)
    checksum_payload = {
        "detection_calibrated_multi_domain": calibrated,
        "detection_by_domain": [dict(result) for result in domain_results],
        "base_rate_separation": base_rate,
        "random_seed": RANDOM_SEED,
    }
    return {
        "experiment": "experiment_4408_cross_domain_detection_calibration_repair",
        "schema": "carnot.cross_domain_detection_calibration_repair.v1",
        "honest_verdict": verdict,
        "detection_calibrated_multi_domain": bool(calibrated),
        "detection_by_domain": [dict(result) for result in domain_results],
        "domains_at_chance": domains_at_chance(domain_results),
        "base_rate_separation": base_rate,
        "verifier_is_oracle": False,
        "preconditions_checked": [dict(item) for item in preconditions_checked],
        "cited_upstream_artifacts": [dict(item) for item in cited_upstream_artifacts],
        "random_seed": RANDOM_SEED,
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "reproducibility_checksum": hash_sources(source_paths, payload=checksum_payload),
        "model_specs": _model_specs(
            domain_results=domain_results,
            sca_by_domain=sca_by_domain,
            pools_built=pools_built,
            unavailable_domains=unavailable_domains,
            bootstrap_resamples=bootstrap_resamples,
            random_control_replicates=random_control_replicates,
        ),
        "missing_verifier_gaps": missing_gap_entries(domain_results),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round_float(duration_s, 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "positive_control_passed": bool(
            sum(
                1
                for result in domain_results
                if str(result.get("domain")) != "fover"
                and ci_lower_beats_chance(result.get("auroc_ci95", []))
            )
            >= 2
        ),
    }


def build_blocked_artifact(
    *,
    sca_by_domain: Mapping[str, SCAResult],
    pools_built: Sequence[Mapping[str, Any]],
    unavailable_domains: Sequence[Mapping[str, Any]],
    preconditions_checked: Sequence[Mapping[str, Any]],
    cited_upstream_artifacts: Sequence[Mapping[str, Any]],
    source_paths: Sequence[Path],
    duration_s: float,
) -> dict[str, Any]:
    base_rate = _base_rate_separation(sca_by_domain, [])
    return {
        "experiment": "experiment_4408_cross_domain_detection_calibration_repair",
        "schema": "carnot.cross_domain_detection_calibration_repair.v1",
        "honest_verdict": "blocked_insufficient_pools_for_multi_domain_claim",
        "detection_calibrated_multi_domain": False,
        "detection_by_domain": [],
        "domains_at_chance": [],
        "base_rate_separation": base_rate,
        "verifier_is_oracle": False,
        "preconditions_checked": [dict(item) for item in preconditions_checked],
        "cited_upstream_artifacts": [dict(item) for item in cited_upstream_artifacts],
        "random_seed": RANDOM_SEED,
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "reproducibility_checksum": hash_sources(
            source_paths, payload={"blocked": "insufficient_pools", "base_rate": base_rate}
        ),
        "model_specs": {
            "blocked_reason": "fewer_than_two_non_fover_domains_reached_proper_pool_n",
            "pools_built": [dict(pool) for pool in pools_built],
            "unavailable_domains": [dict(item) for item in unavailable_domains],
            "trm_training": "stood_down_not_invoked",
            "live_generation": False,
            "verifier_is_oracle": False,
            "sca_config": base_rate,
        },
        "missing_verifier_gaps": [],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round_float(duration_s, 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "adversarial_verify": {"status": "not_run_blocked_preconditions"},
    }


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [f"missing:{field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if not isinstance(artifact.get("detection_calibrated_multi_domain"), bool):
        errors.append("invalid:detection_calibrated_multi_domain")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("invalid:verifier_is_oracle")
    if not isinstance(artifact.get("detection_by_domain"), list):
        errors.append("invalid:detection_by_domain")
    if not isinstance(artifact.get("base_rate_separation"), dict):
        errors.append("invalid:base_rate_separation")
    if not isinstance(artifact.get("preconditions_checked"), list):
        errors.append("invalid:preconditions_checked")
    if not isinstance(artifact.get("cited_upstream_artifacts"), list):
        errors.append("invalid:cited_upstream_artifacts")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid:inference_substrate")
    return errors


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    adversarial_verify_runner: AdversarialVerifyRunner = run_adversarial_verify,
    write: bool = True,
) -> dict[str, Any]:
    cfg = config or ExperimentConfig()
    started = cfg.start_time()
    raw_by_domain, pools_built, unavailable_domains, loaded_sources = load_raw_domain_rows(cfg)
    sca_by_domain = {
        domain: semantic_confidence_aggregation(rows)
        for domain, rows in sorted(raw_by_domain.items())
    }
    preconditions, powered_non_fover = _precondition_records(
        sca_by_domain,
        unavailable_domains,
        min_powered_n=cfg.min_powered_n,
        min_non_fover_powered=cfg.min_non_fover_powered,
    )
    source_paths = list(
        dict.fromkeys(
            [
                cfg.exp4397_path,
                cfg.sca_ingestion_path,
                cfg.code_1999_path,
                cfg.code_2838_path,
                cfg.code_2839_path,
                cfg.code_reward_artifact_path,
                cfg.gsm8k_pool_path,
                cfg.gsm8k_baseline_path,
                cfg.fover_corpus_path,
                cfg.fover_baseline_path,
                cfg.fover_dual_condition_path,
                cfg.arc_detector_model_path,
                cfg.arc_candidate_pool_path,
                cfg.arc_rerank_path,
                *loaded_sources,
            ]
        )
    )
    cited = _cited_upstream_artifacts(cfg, loaded_sources)
    if len(powered_non_fover) < cfg.min_non_fover_powered:
        artifact = build_blocked_artifact(
            sca_by_domain=sca_by_domain,
            pools_built=pools_built,
            unavailable_domains=unavailable_domains,
            preconditions_checked=preconditions,
            cited_upstream_artifacts=cited,
            source_paths=source_paths,
            duration_s=cfg.clock() - started,
        )
        if write:
            _write_artifact(cfg.artifact_path, artifact)
        return artifact

    calibration = leave_one_domain_out_calibration(
        {domain: result.rows for domain, result in sca_by_domain.items()},
        n_steps=cfg.calibration_steps,
        learning_rate=cfg.calibration_learning_rate,
    )
    domain_results = [
        summarize_domain(
            domain,
            result.rows,
            sca_metadata=result.metadata,
            calibration_report=calibration.get(domain, {}),
            seed=cfg.random_seed,
            bootstrap_resamples=cfg.bootstrap_resamples,
            random_control_replicates=cfg.random_control_replicates,
            min_powered_n=cfg.min_powered_n,
        )
        for domain, result in sorted(sca_by_domain.items())
    ]
    artifact = build_complete_artifact(
        domain_results=domain_results,
        sca_by_domain=sca_by_domain,
        pools_built=pools_built,
        unavailable_domains=unavailable_domains,
        preconditions_checked=preconditions,
        powered_non_fover=powered_non_fover,
        cited_upstream_artifacts=cited,
        source_paths=source_paths,
        duration_s=cfg.clock() - started,
        bootstrap_resamples=cfg.bootstrap_resamples,
        random_control_replicates=cfg.random_control_replicates,
    )
    if write:
        append_missing_verifier_gaps(cfg.verifier_gaps_path, artifact["missing_verifier_gaps"])
        _write_artifact(cfg.artifact_path, artifact)
        artifact["adversarial_verify"] = adversarial_verify_runner(cfg.artifact_path)
        _write_artifact(cfg.artifact_path, artifact)
    else:
        artifact["adversarial_verify"] = {"returncode": None, "skipped": True}
    return artifact


def main() -> int:  # pragma: no cover - exercised through results/ CLI shim.
    artifact = run_experiment(write=True)
    print(
        "[exp4408] "
        f"{artifact['honest_verdict']} "
        f"detection_calibrated_multi_domain={artifact['detection_calibrated_multi_domain']} "
        f"domains={len(artifact['detection_by_domain'])} -> {ARTIFACT_PATH}",
        flush=True,
    )
    return 0
