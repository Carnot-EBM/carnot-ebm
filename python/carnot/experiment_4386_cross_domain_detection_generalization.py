"""Exp 4386: cross-domain detector generalization on cached non-FoVer pools.

Spec refs: REQ-VERIFY-4386, SCENARIO-VERIFY-4386.
"""

from __future__ import annotations

import bisect
import gzip
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - the repository test environment has numpy.
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "results" / "experiment_4386_cross_domain_detection_generalization.json"
REGISTRY_PATH = ROOT / "ops" / "verifier_registry.yaml"
FOVER_BASELINE_PATH = ROOT / "results" / "experiment_4375_verifier_as_detector_measurement.json"
DETECTOR_CONFIG_PATH = (
    ROOT / "results" / "experiment_4381_biprm_detector_localization_abstention.json"
)
HEADROOM_CENSUS_PATH = ROOT / "results" / "experiment_4175_headroom_gate_executable_census.json"
ARC_RERANK_PATH = ROOT / "results" / "arc3_trm_verifier_rerank.json"
ARC_DETECTOR_MODEL_PATH = ROOT / "results" / "experiment_4244_arc_set_encoder_aggregator_model.json"
ARC_CANDIDATE_POOL_PATH = ROOT / "results" / "experiment_4243_arc_candidate_pool_grow_pool.json.gz"
CODE_POOL_PATH = ROOT / "results" / "experiment_1999_code_verification_humaneval.json"
GSM8K_POOL_PATH = ROOT / "results" / "adversarial_gsm8k_data_400.json"
VERIFIER_GAPS_PATH = ROOT / "ops" / "verifier_gaps.md"

RANDOM_SEED = 4386
RANDOM_SEEDS_USED = (4386,)
BOOTSTRAP_RESAMPLES = 2500
RANDOM_CONTROL_REPLICATES = 128
MIN_DOMAIN_CANDIDATES = 1000
SPEC_REFS = ["REQ-VERIFY-4386", "SCENARIO-VERIFY-4386"]
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "detector_generalizes_cross_domain",
    "detection_by_domain",
    "domains_at_chance",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A generalization win (detection beats chance on a "
        "NON-FoVer domain -- the verifier-domain-expansion step) and an honest "
        "null (detection is FoVer-bound -> a logged missing-verifier gap = the "
        "product backlog) are BOTH decision-grade."
    ),
    "detector_generalizes_cross_domain": (
        "BARE bool: the capstone reads this (gated-fields-must-be-bare); true "
        "iff detection AUROC CI95 lower bound > 0.5 on >=1 NON-FoVer domain -- "
        "the detection capability is a GENERAL Carnot property, not FoVer-specific."
    ),
    "detection_by_domain": (
        "list of {domain, detection_auroc, auroc_ci95, selection_headroom, n, "
        "base_rate} -- the per-domain detection-vs-selection DIVERGENCE "
        "(detect where you cannot select), the honest cross-domain reading."
    ),
    "domains_at_chance": (
        "list[str]: domains where detection AUROC CI95 includes 0.5 -- each "
        "LOGGED as a missing-verifier gap (the product backlog the operator "
        "directed us to grow)."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- a learned/energy detection signal scored against "
        "cached candidates, oracle-distinct."
    ),
    "preconditions_checked": (
        "Records the non-FoVer cached pool + ensemble + TRM-stand-down verified; "
        "pre-empts the silent-missing-resource fabrication mode."
    ),
    "random_seed": (
        "Determinism precondition for the (stochastic) ensemble scoring + the bootstrap."
    ),
    "reproducibility_checksum": (
        "Hash of the cross-domain pools + the ensemble config + the AUROC "
        "computation; lets a third party re-run."
    ),
    "model_specs": (
        "The verifier ensemble + the non-FoVer corpora + the selection-headroom "
        "source + n per domain; required methodology."
    ),
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One prerequisite checked before cached cross-domain scoring starts."""

    resource: str
    available: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "resource": self.resource,
            "available": bool(self.available),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ScoredCandidate:
    """One cached non-FoVer candidate with an oracle-distinct detector score."""

    domain: str
    task_id: str
    candidate_id: str
    is_correct: bool
    verifier_score: float
    valid_output: bool = True
    source: str = ""


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for Exp 4386."""

    repo_root: Path = ROOT
    artifact_path: Path = ARTIFACT_PATH
    registry_path: Path = REGISTRY_PATH
    fover_baseline_path: Path = FOVER_BASELINE_PATH
    detector_config_path: Path = DETECTOR_CONFIG_PATH
    headroom_census_path: Path = HEADROOM_CENSUS_PATH
    arc_rerank_path: Path = ARC_RERANK_PATH
    arc_detector_model_path: Path = ARC_DETECTOR_MODEL_PATH
    arc_candidate_pool_path: Path = ARC_CANDIDATE_POOL_PATH
    code_pool_path: Path = CODE_POOL_PATH
    gsm8k_pool_path: Path = GSM8K_POOL_PATH
    verifier_gaps_path: Path = VERIFIER_GAPS_PATH
    min_domain_candidates: int = MIN_DOMAIN_CANDIDATES
    random_seed: int = RANDOM_SEED
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    random_control_replicates: int = RANDOM_CONTROL_REPLICATES
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


AdversarialVerifyRunner = Callable[[Path], dict[str, Any]]


def round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return round(float(value), digits)


def _read_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def hash_sources(source_paths: Sequence[Path], *, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for path in sorted({Path(path) for path in source_paths}, key=lambda item: str(item)):
        digest.update(str(path).encode("utf-8"))
        if not path.exists():
            digest.update(b"\0MISSING\0")
            continue
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def compute_auroc(labels: Sequence[int | bool], scores: Sequence[float]) -> float:
    """Compute AUROC where label 1 means correct and higher score means correct."""

    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    int_labels = [int(label) for label in labels]
    n_pos = sum(int_labels)
    n_neg = len(int_labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC requires both positive and negative labels")

    ranked = sorted(enumerate(scores), key=lambda item: float(item[1]))
    ranks = [0.0] * len(scores)
    cursor = 0
    while cursor < len(ranked):
        end = cursor + 1
        while end < len(ranked) and float(ranked[end][1]) == float(ranked[cursor][1]):
            end += 1
        avg_rank = (cursor + 1 + end) / 2.0
        for offset in range(cursor, end):
            ranks[ranked[offset][0]] = avg_rank
        cursor = end
    pos_rank_sum = sum(rank for rank, label in zip(ranks, int_labels, strict=True) if label == 1)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _bootstrap_auroc_ci95_numpy(
    labels: Sequence[int | bool],
    scores: Sequence[float],
    *,
    seed: int,
    resamples: int,
) -> list[float | None]:
    label_array = np.asarray([int(label) for label in labels], dtype=np.int8)  # type: ignore[union-attr]
    score_array = np.asarray([float(score) for score in scores], dtype=float)  # type: ignore[union-attr]
    pos_scores = score_array[label_array == 1]
    neg_scores = score_array[label_array == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0 or resamples <= 0:
        return [None, None]
    rng = np.random.default_rng(seed)  # type: ignore[union-attr]
    values: list[float] = []
    denom = float(len(pos_scores) * len(neg_scores))
    for _idx in range(resamples):
        pos_sample = pos_scores[rng.integers(0, len(pos_scores), len(pos_scores))]
        neg_sample = np.sort(neg_scores[rng.integers(0, len(neg_scores), len(neg_scores))])  # type: ignore[union-attr]
        left = np.searchsorted(neg_sample, pos_sample, side="left")  # type: ignore[union-attr]
        right = np.searchsorted(neg_sample, pos_sample, side="right")  # type: ignore[union-attr]
        values.append(float((left.sum() + 0.5 * (right - left).sum()) / denom))
    values.sort()
    lo = int(0.025 * (len(values) - 1))
    hi = int(0.975 * (len(values) - 1))
    return [round_float(values[lo]), round_float(values[hi])]


def _bootstrap_auroc_ci95_python(
    labels: Sequence[int | bool],
    scores: Sequence[float],
    *,
    seed: int,
    resamples: int,
) -> list[float | None]:  # pragma: no cover - numpy path is exercised in CI.
    pos_scores = [
        float(score) for label, score in zip(labels, scores, strict=True) if int(label) == 1
    ]
    neg_scores = [
        float(score) for label, score in zip(labels, scores, strict=True) if int(label) == 0
    ]
    if not pos_scores or not neg_scores or resamples <= 0:
        return [None, None]
    rng = random.Random(seed)
    values: list[float] = []
    denom = float(len(pos_scores) * len(neg_scores))
    for _idx in range(resamples):
        pos_sample = [pos_scores[rng.randrange(len(pos_scores))] for _ in pos_scores]
        neg_sample = sorted(neg_scores[rng.randrange(len(neg_scores))] for _ in neg_scores)
        wins = 0.0
        for score in pos_sample:
            left = bisect.bisect_left(neg_sample, score)
            right = bisect.bisect_right(neg_sample, score)
            wins += left + 0.5 * (right - left)
        values.append(wins / denom)
    values.sort()
    lo = int(0.025 * (len(values) - 1))
    hi = int(0.975 * (len(values) - 1))
    return [round_float(values[lo]), round_float(values[hi])]


def bootstrap_auroc_ci95(
    labels: Sequence[int | bool],
    scores: Sequence[float],
    *,
    seed: int,
    resamples: int,
) -> list[float | None]:
    """Return a stratified bootstrap CI95 for candidate-level detection AUROC."""

    if len(labels) != len(scores) or len({int(label) for label in labels}) < 2:
        return [None, None]
    if np is not None:
        return _bootstrap_auroc_ci95_numpy(labels, scores, seed=seed, resamples=resamples)
    return _bootstrap_auroc_ci95_python(  # pragma: no cover
        labels, scores, seed=seed, resamples=resamples
    )


def ci_lower_beats_chance(ci95: Sequence[float | None]) -> bool:
    return bool(ci95 and ci95[0] is not None and float(ci95[0]) > 0.5)


def ci_includes_chance(ci95: Sequence[float | None]) -> bool:
    return bool(
        len(ci95) >= 2
        and ci95[0] is not None
        and ci95[1] is not None
        and float(ci95[0]) <= 0.5 <= float(ci95[1])
    )


def random_score_auroc_control(
    labels: Sequence[int | bool],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    """Average random-score AUROC over replicates so the control stays near 0.5."""

    rng = random.Random(seed)
    values: list[float] = []
    for _idx in range(replicates):
        random_scores = [rng.random() for _label in labels]
        values.append(compute_auroc(labels, random_scores))
    return {
        "auroc": round_float(sum(values) / len(values) if values else None),
        "replicates": int(replicates),
        "seed": int(seed),
    }


def _is_valid_grid(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    width: int | None = None
    for row in value:
        if not isinstance(row, list) or not row:
            return False
        width = len(row) if width is None else width
        if len(row) != width:
            return False
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, (int, float)):
                return False
            if int(cell) != cell or not 0 <= int(cell) <= 9:
                return False
    return True


def load_arc_valid_grid_map(pool_path: Path) -> dict[str, bool]:
    payload = _read_json(pool_path)
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    valid: dict[str, bool] = {}
    if not isinstance(tasks, list):
        return valid
    for task in tasks:
        if not isinstance(task, dict):
            continue
        for candidate in task.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            candidate_id = candidate.get("candidate_id")
            if candidate_id is None:
                continue
            valid[str(candidate_id)] = _is_valid_grid(candidate.get("grid"))
    return valid


def load_arc_set_encoder_rows(model_path: Path, candidate_pool_path: Path) -> list[ScoredCandidate]:
    """Load cached ARC set-encoder out-of-fold scores as detector rows."""

    payload = _read_json(model_path)
    if not isinstance(payload, dict):
        raise ValueError("ARC detector model payload must be a JSON object")
    if payload.get("verifier_is_oracle") is True:
        raise ValueError("ARC detector model is marked verifier_is_oracle=true")
    oof = payload.get("set_encoder_oof")
    rows = oof.get("rows") if isinstance(oof, dict) else payload.get("oof_rows")
    if not isinstance(rows, list):
        raise ValueError("ARC detector model missing cached out-of-fold rows")
    valid_map = load_arc_valid_grid_map(candidate_pool_path) if candidate_pool_path.exists() else {}
    scored: list[ScoredCandidate] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        correct = row.get("correct")
        score = row.get("score")
        if not isinstance(correct, bool) or not isinstance(score, (int, float)):
            continue
        candidate_id = str(row.get("candidate_id") or f"arc_candidate_{idx}")
        task_id = str(row.get("task_id") or "unknown_task")
        scored.append(
            ScoredCandidate(
                domain="gap4_arc",
                task_id=task_id,
                candidate_id=candidate_id,
                is_correct=correct,
                verifier_score=float(score),
                valid_output=valid_map.get(candidate_id, True),
                source=str(model_path),
            )
        )
    if not scored:
        raise ValueError("ARC detector model had no usable scored rows")
    return scored


def _generic_scored_rows_from_pool(domain: str, path: Path) -> list[ScoredCandidate]:
    payload = _read_json(path)
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(tasks, list):
        return []
    rows: list[ScoredCandidate] = []
    for task_idx, task in enumerate(tasks):
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or task.get("task") or f"task_{task_idx}")
        candidates = task.get("candidates") or task.get("cands")
        if not isinstance(candidates, list):
            continue
        for cand_idx, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            correct = candidate.get("is_correct", candidate.get("correct"))
            score = candidate.get("verifier_score", candidate.get("score"))
            if not isinstance(correct, bool) or not isinstance(score, (int, float)):
                continue
            rows.append(
                ScoredCandidate(
                    domain=domain,
                    task_id=task_id,
                    candidate_id=str(candidate.get("candidate_id") or f"{task_id}::{cand_idx}"),
                    is_correct=correct,
                    verifier_score=float(score),
                    valid_output=True,
                    source=str(path),
                )
            )
    return rows


def _code_pool_unavailable_reason(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        payload = _read_json(path)
    except Exception as exc:
        return f"unreadable: {exc}"
    results = payload.get("results") if isinstance(payload, dict) else None
    if isinstance(results, list):
        return (
            f"labeled_candidates={2 * len(results)}; missing candidate source text "
            "or cached verifier_score per output"
        )
    return "no_humaneval_mbpp_candidate_rows"


def _gsm8k_pool_unavailable_reason(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        payload = _read_json(path)
    except Exception as exc:
        return f"unreadable: {exc}"
    if isinstance(payload, dict) and "datasets" in payload:
        return "datasets_present_but_no_multicandidate_verifier_scores"
    return "no_gsm8k_candidate_rows"


def load_available_domain_rows(
    config: ExperimentConfig,
) -> tuple[dict[str, list[ScoredCandidate]], list[dict[str, Any]], list[Path]]:
    """Load every cached non-FoVer scored pool available for Exp 4386."""

    domains: dict[str, list[ScoredCandidate]] = {}
    unavailable: list[dict[str, Any]] = []
    source_paths: list[Path] = []

    try:
        arc_rows = load_arc_set_encoder_rows(
            config.arc_detector_model_path,
            config.arc_candidate_pool_path,
        )
        domains["gap4_arc"] = arc_rows
        source_paths.extend([config.arc_detector_model_path, config.arc_candidate_pool_path])
    except Exception as exc:
        unavailable.append({"domain": "gap4_arc", "reason": str(exc)})

    code_rows = _generic_scored_rows_from_pool("code_humaneval_mbpp", config.code_pool_path)
    if code_rows:
        domains["code_humaneval_mbpp"] = code_rows
        source_paths.append(config.code_pool_path)
    else:
        unavailable.append(
            {
                "domain": "code_humaneval_mbpp",
                "reason": _code_pool_unavailable_reason(config.code_pool_path),
            }
        )

    gsm_rows = _generic_scored_rows_from_pool("gsm8k", config.gsm8k_pool_path)
    if gsm_rows:
        domains["gsm8k"] = gsm_rows
        source_paths.append(config.gsm8k_pool_path)
    else:
        unavailable.append(
            {"domain": "gsm8k", "reason": _gsm8k_pool_unavailable_reason(config.gsm8k_pool_path)}
        )

    return domains, unavailable, source_paths


def load_selection_headrooms(headroom_path: Path, arc_rerank_path: Path) -> dict[str, float]:
    headrooms = {"gap4_arc": 0.0, "code_humaneval_mbpp": 0.0, "gsm8k": 0.0}
    if headroom_path.exists():
        payload = _read_json(headroom_path)
        per_domain = payload.get("per_domain_headroom") if isinstance(payload, dict) else {}
        if isinstance(per_domain, dict):
            code = per_domain.get("code")
            math_domain = per_domain.get("math")
            arc_legacy = per_domain.get("sudoku")
            if isinstance(code, dict):
                headrooms["code_humaneval_mbpp"] = float(
                    code.get("selectable_headroom", 0.0) or 0.0
                )
            if isinstance(math_domain, dict):
                headrooms["gsm8k"] = float(math_domain.get("selectable_headroom", 0.0) or 0.0)
            if isinstance(arc_legacy, dict):
                headrooms["gap4_arc"] = float(arc_legacy.get("selectable_headroom", 0.0) or 0.0)
    if arc_rerank_path.exists():
        payload = _read_json(arc_rerank_path)
        if isinstance(payload, dict):
            oracle = payload.get("oracle_ceiling")
            if isinstance(oracle, dict):
                oracle_at_k = oracle.get("pass@2", oracle.get("pass@1000"))
                vote = payload.get("trm_vote_pass2")
                if isinstance(oracle_at_k, (int, float)) and isinstance(vote, (int, float)):
                    headrooms["gap4_arc"] = float(oracle_at_k) - float(vote)
    return {key: round(float(value), 10) for key, value in headrooms.items()}


def summarize_domain(
    domain: str,
    rows: Sequence[ScoredCandidate],
    *,
    selection_headroom: float,
    seed: int,
    bootstrap_resamples: int,
    random_control_replicates: int,
    min_candidates: int = MIN_DOMAIN_CANDIDATES,
) -> dict[str, Any]:
    labels = [1 if row.is_correct else 0 for row in rows]
    scores = [float(row.verifier_score) for row in rows]
    auroc = compute_auroc(labels, scores)
    ci95 = bootstrap_auroc_ci95(labels, scores, seed=seed, resamples=bootstrap_resamples)
    valid_rows = [row for row in rows if row.is_correct or row.valid_output]
    valid_auroc: float | None = None
    if len({int(row.is_correct) for row in valid_rows}) == 2:
        valid_auroc = compute_auroc(
            [1 if row.is_correct else 0 for row in valid_rows],
            [row.verifier_score for row in valid_rows],
        )
    summary = {
        "domain": domain,
        "detection_auroc": round_float(auroc),
        "auroc_ci95": ci95,
        "selection_headroom": round_float(selection_headroom),
        "n": int(len(rows)),
        "base_rate": round_float(sum(labels) / max(1, len(labels))),
        "random_score_auroc_control": random_score_auroc_control(
            labels,
            seed=seed,
            replicates=random_control_replicates,
        ),
        "valid_but_wrong_restricted_auroc": round_float(valid_auroc),
        "valid_but_wrong_restricted_n": int(len(valid_rows)),
        "valid_wrong_negative_n": int(sum(1 for row in valid_rows if not row.is_correct)),
        "score_orientation": "higher_verifier_score_means_more_likely_correct",
        "claim_scope": (
            "n>=1000"
            if len(rows) >= min_candidates
            else f"underpowered_n={len(rows)}; report_n_only_scope_claim"
        ),
    }
    return summary


def detector_generalizes(domain_results: Sequence[dict[str, Any]]) -> bool:
    return any(ci_lower_beats_chance(result.get("auroc_ci95", [])) for result in domain_results)


def domains_at_chance(domain_results: Sequence[dict[str, Any]]) -> list[str]:
    return [
        str(result["domain"])
        for result in domain_results
        if ci_includes_chance(result.get("auroc_ci95", []))
    ]


def missing_gap_entries(domain_results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for result in domain_results:
        if not ci_includes_chance(result.get("auroc_ci95", [])):
            continue
        domain = str(result["domain"])
        domain_slug = domain.upper().replace("_", "-")
        headroom = float(result.get("selection_headroom") or 0.0)
        entries.append(
            {
                "gap_id": f"GAP-4386-{domain_slug}-DETECTOR-CHANCE",
                "status": "open",
                "domain": domain,
                "failure_mode": (
                    f"Detection AUROC CI95 includes chance on {domain} "
                    f"while selection_headroom={round_float(headroom)}."
                ),
                "missing_discriminator": (
                    "A domain-specific oracle-distinct verifier feature that separates "
                    "correct cached outputs from plausible wrong outputs without using "
                    "the executable oracle."
                ),
                "candidate_design": (
                    "Train or add a domain verifier over cached accepted/rejected "
                    "candidate features, then rerun Exp 4386 with the same AUROC-vs-headroom gate."
                ),
                "priority": "high" if headroom >= 0.10 else "medium",
            }
        )
    return entries


def append_missing_verifier_gaps(path: Path, entries: Sequence[dict[str, Any]]) -> None:
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


def _json_artifact_check(path: Path, resource: str, required_key: str) -> PreconditionCheck:
    if not path.exists():
        return PreconditionCheck(resource, False, "missing")
    try:
        payload = _read_json(path)
    except Exception as exc:
        return PreconditionCheck(resource, False, f"unreadable: {exc}")
    ok = isinstance(payload, dict) and required_key in payload
    return PreconditionCheck(
        resource,
        ok,
        f"{required_key} present" if ok else f"missing {required_key}",
    )


def check_preconditions(
    config: ExperimentConfig,
    domain_rows: dict[str, list[ScoredCandidate]],
    unavailable_domains: Sequence[dict[str, Any]],
) -> list[PreconditionCheck]:
    checks: list[PreconditionCheck] = []
    registry_ok = config.registry_path.exists() and "verifier_id" in config.registry_path.read_text(
        encoding="utf-8"
    )
    checks.append(
        PreconditionCheck(
            "verifier_registry",
            registry_ok,
            "loaded verifier registry" if registry_ok else "missing or malformed verifier registry",
        )
    )
    checks.append(
        _json_artifact_check(
            config.fover_baseline_path, "fover_detector_baseline", "detector_auroc"
        )
    )
    checks.append(
        _json_artifact_check(config.detector_config_path, "biprm_detector_config", "model_specs")
    )
    checks.append(
        _json_artifact_check(
            config.headroom_census_path, "selection_headroom_census", "per_domain_headroom"
        )
    )
    checks.append(
        _json_artifact_check(config.arc_rerank_path, "gap4_arc_rerank_summary", "oracle_ceiling")
    )
    scored_detail = (
        ", ".join(f"{domain}:n={len(rows)}" for domain, rows in sorted(domain_rows.items()))
        if domain_rows
        else "none; "
        + "; ".join(f"{item['domain']}={item['reason']}" for item in unavailable_domains)
    )
    checks.append(
        PreconditionCheck("non_fover_cached_scored_pool", bool(domain_rows), scored_detail)
    )
    checks.append(
        PreconditionCheck(
            "trm_training_stand_down",
            True,
            "not invoked; Exp 4386 scores cached candidates and cached detector outputs only",
        )
    )
    return checks


def _model_specs(
    *,
    domain_results: Sequence[dict[str, Any]],
    unavailable_domains: Sequence[dict[str, Any]],
    source_paths: Sequence[Path],
    bootstrap_resamples: int,
    random_control_replicates: int,
) -> dict[str, Any]:
    return {
        "verifier_ensemble_id": "cross_domain_cached_detector_suite",
        "ensemble_registry_path": str(REGISTRY_PATH),
        "score_sources": {
            "gap4_arc": str(ARC_DETECTOR_MODEL_PATH),
            "fover_detector_baseline": str(FOVER_BASELINE_PATH),
            "biprm_detector_config": str(DETECTOR_CONFIG_PATH),
        },
        "non_fover_corpora": {
            str(result["domain"]): {
                "n": int(result["n"]),
                "base_rate": result["base_rate"],
                "selection_headroom": result["selection_headroom"],
                "claim_scope": result.get("claim_scope"),
            }
            for result in domain_results
        },
        "unavailable_domains": list(unavailable_domains),
        "selection_headroom_source": str(HEADROOM_CENSUS_PATH),
        "arc_selection_headroom_summary": str(ARC_RERANK_PATH),
        "source_paths": [str(path) for path in source_paths],
        "bootstrap_method": "stratified_candidate_bootstrap",
        "bootstrap_resamples": int(bootstrap_resamples),
        "random_score_control_replicates": int(random_control_replicates),
        "random_seed": RANDOM_SEED,
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "trm_training": "stood_down_not_invoked",
        "live_generation": False,
        "verifier_is_oracle": False,
    }


def build_complete_artifact(
    *,
    domain_results: Sequence[dict[str, Any]],
    unavailable_domains: Sequence[dict[str, Any]],
    preconditions_checked: list[dict[str, Any]],
    source_paths: Sequence[Path],
    duration_s: float,
    bootstrap_resamples: int,
    random_control_replicates: int,
    model_specs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    generalizes = detector_generalizes(domain_results)
    chance_domains = domains_at_chance(domain_results)
    if generalizes:
        verdict = "success: detector_generalizes_cross_domain_non_fover"
    elif domain_results:
        verdict = "complete: detector_fover_bound_non_fover_domains_at_chance"
    else:
        verdict = "blocked_no_non_fover_cached_pool"
    checksum_payload = {
        "detection_by_domain": list(domain_results),
        "domains_at_chance": chance_domains,
        "detector_generalizes_cross_domain": generalizes,
        "bootstrap_resamples": bootstrap_resamples,
        "random_control_replicates": random_control_replicates,
        "random_seed": RANDOM_SEED,
    }
    specs = model_specs or _model_specs(
        domain_results=domain_results,
        unavailable_domains=unavailable_domains,
        source_paths=source_paths,
        bootstrap_resamples=bootstrap_resamples,
        random_control_replicates=random_control_replicates,
    )
    return {
        "experiment": "experiment_4386_cross_domain_detection_generalization",
        "schema": "carnot.cross_domain_detection_generalization.v1",
        "honest_verdict": verdict,
        "detector_generalizes_cross_domain": bool(generalizes),
        "detection_by_domain": [dict(result) for result in domain_results],
        "domains_at_chance": chance_domains,
        "verifier_is_oracle": False,
        "preconditions_checked": preconditions_checked,
        "random_seed": RANDOM_SEED,
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "bootstrap_resamples": int(bootstrap_resamples),
        "reproducibility_checksum": hash_sources(source_paths, payload=checksum_payload),
        "model_specs": specs,
        "unavailable_domains": list(unavailable_domains),
        "missing_verifier_gaps": missing_gap_entries(domain_results),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round_float(duration_s, 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
    }


def build_blocked_artifact(
    *,
    preconditions_checked: list[dict[str, Any]],
    source_paths: Sequence[Path],
    duration_s: float,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4386_cross_domain_detection_generalization",
        "schema": "carnot.cross_domain_detection_generalization.v1",
        "honest_verdict": "blocked_no_non_fover_cached_pool",
        "detector_generalizes_cross_domain": False,
        "detection_by_domain": [],
        "domains_at_chance": [],
        "verifier_is_oracle": False,
        "preconditions_checked": preconditions_checked,
        "random_seed": RANDOM_SEED,
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "reproducibility_checksum": hash_sources(
            source_paths, payload={"blocked": "blocked_no_non_fover_cached_pool"}
        ),
        "model_specs": {
            "blocked_reason": "no_non_fover_cached_scored_pool",
            "trm_training": "stood_down_not_invoked",
            "live_generation": False,
            "verifier_is_oracle": False,
        },
        "unavailable_domains": [],
        "missing_verifier_gaps": [],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round_float(duration_s, 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "adversarial_verify": {"status": "not_run_blocked_preconditions"},
    }


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    errors = [f"missing:{field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if not isinstance(artifact.get("detector_generalizes_cross_domain"), bool):
        errors.append("invalid:detector_generalizes_cross_domain")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("invalid:verifier_is_oracle")
    if not isinstance(artifact.get("detection_by_domain"), list):
        errors.append("invalid:detection_by_domain")
    if not isinstance(artifact.get("domains_at_chance"), list):
        errors.append("invalid:domains_at_chance")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid:inference_substrate")
    return errors


def run_adversarial_verify(path: Path, repo_root: Path = ROOT) -> dict[str, Any]:
    script = repo_root / "scripts" / "adversarial_verify.py"
    if not script.is_file():
        return {"returncode": None, "flags": [], "stderr": "scripts/adversarial_verify.py missing"}
    proc = subprocess.run(
        [sys.executable, str(script), str(path)],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    return {
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _configured_source_paths(
    config: ExperimentConfig, loaded_source_paths: Sequence[Path]
) -> list[Path]:
    paths = [
        config.registry_path,
        config.fover_baseline_path,
        config.detector_config_path,
        config.headroom_census_path,
        config.arc_rerank_path,
        config.code_pool_path,
        config.gsm8k_pool_path,
    ]
    paths.extend(loaded_source_paths)
    return list(dict.fromkeys(paths))


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    adversarial_verify_runner: AdversarialVerifyRunner = run_adversarial_verify,
    write: bool = True,
) -> dict[str, Any]:
    """Run Exp 4386 and optionally write the terminal JSON artifact."""

    cfg = config or ExperimentConfig()
    started = cfg.start_time()
    domain_rows, unavailable_domains, loaded_sources = load_available_domain_rows(cfg)
    source_paths = _configured_source_paths(cfg, loaded_sources)
    checks = check_preconditions(cfg, domain_rows, unavailable_domains)
    preconditions = [check.as_dict() for check in checks]
    if not all(check.available for check in checks):
        artifact = build_blocked_artifact(
            preconditions_checked=preconditions,
            source_paths=source_paths,
            duration_s=cfg.clock() - started,
        )
        if write:
            _write_artifact(cfg.artifact_path, artifact)
        return artifact

    headrooms = load_selection_headrooms(cfg.headroom_census_path, cfg.arc_rerank_path)
    domain_results = [
        summarize_domain(
            domain,
            rows,
            selection_headroom=headrooms.get(domain, 0.0),
            seed=cfg.random_seed,
            bootstrap_resamples=cfg.bootstrap_resamples,
            random_control_replicates=cfg.random_control_replicates,
            min_candidates=cfg.min_domain_candidates,
        )
        for domain, rows in sorted(domain_rows.items())
    ]
    model_specs = _model_specs(
        domain_results=domain_results,
        unavailable_domains=unavailable_domains,
        source_paths=source_paths,
        bootstrap_resamples=cfg.bootstrap_resamples,
        random_control_replicates=cfg.random_control_replicates,
    )
    artifact = build_complete_artifact(
        domain_results=domain_results,
        unavailable_domains=unavailable_domains,
        preconditions_checked=preconditions,
        source_paths=source_paths,
        duration_s=cfg.clock() - started,
        bootstrap_resamples=cfg.bootstrap_resamples,
        random_control_replicates=cfg.random_control_replicates,
        model_specs=model_specs,
    )
    if write:
        append_missing_verifier_gaps(cfg.verifier_gaps_path, artifact["missing_verifier_gaps"])
        _write_artifact(cfg.artifact_path, artifact)
        artifact["adversarial_verify"] = adversarial_verify_runner(cfg.artifact_path)
        _write_artifact(cfg.artifact_path, artifact)
    else:
        artifact["adversarial_verify"] = {"returncode": None, "skipped": True}
    return artifact


def main() -> int:  # pragma: no cover - exercised through the results/ CLI shim.
    artifact = run_experiment(write=True)
    print(
        "[exp4386] "
        f"{artifact['honest_verdict']} "
        f"detector_generalizes_cross_domain={artifact['detector_generalizes_cross_domain']} "
        f"domains={len(artifact['detection_by_domain'])} -> {ARTIFACT_PATH}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
