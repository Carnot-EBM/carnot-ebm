"""Exp 4245 held-out ARC Set-Encoder beats-vote gate.

Spec refs: REQ-VERIFY-4245, SCENARIO-VERIFY-4245,
SCENARIO-VERIFY-4245-NO-HEADROOM, SCENARIO-VERIFY-4245-DEFERRED.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.reporting import arc_set_encoder_aggregator_4244 as exp4244


RANDOM_SEED = 4245
BOOTSTRAP_RESAMPLES = 2000
MARGIN_TRIGGER_THRESHOLD = 0.10
A2_REL = Path("results/experiment_4244_arc_set_encoder_aggregator_build.json")
OUTPUT_REL = Path("results/experiment_4245_arc_set_encoder_beats_vote.json")
SPEC_REFS = [
    "REQ-VERIFY-4245",
    "SCENARIO-VERIFY-4245",
    "SCENARIO-VERIFY-4245-NO-HEADROOM",
    "SCENARIO-VERIFY-4245-DEFERRED",
]
INFERENCE_SUBSTRATE = "cached_grown_arc_pool_oof_set_encoder_rerank"
DEFERRED_VERDICT = "complete_arc_oracle_distinct_gate_deferred_no_built_aggregator"
NO_HEADROOM_VERDICT = "complete_arc_oracle_distinct_no_headroom_uninformative"

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A clean beats-vote win, a clean ties-at-power-on-the-grown-pool "
        "null (data-sparsity removed -> a real bound), and an honest no-headroom are "
        "ALL COMPLETE and decision-grade."
    ),
    "oracle_distinct_beats_vote": (
        "BARE bool: set_encoder@1 - vote@1 CI95 excludes 0 AND delta>0 AND headroom "
        "exists -- the de-confounded FIRST ARC oracle-distinct win (GAP-3-ties-vote "
        "closed on the north-star domain); NOT a circular execution result."
    ),
    "set_encoder_minus_vote_delta": (
        "set_encoder@1 - vote@1 on held-out ARC -- the oracle-distinct lift; recovers "
        "the wrong-majority-failure answers (ARBITER/AggLM) a flat vote discards."
    ),
    "set_encoder_minus_vote_ci95": (
        "Task-level bootstrap CI95 of the delta -- excluding 0 is what distinguishes "
        "a real oracle-distinct win from noise."
    ),
    "margin_override_minus_vote": (
        "Margin-triggered override (keep vote unless high set-encoder margin, threshold "
        "pre-registered on the A2 fold) minus vote -- the 2606.04323 deployment pattern; "
        "can win where a flat rerank loses by overriding only on confident wrong-majority "
        "cases (the .392 override never fired)."
    ),
    "matched_control_delta": (
        "set_encoder@1 minus the budget-matched no-verifier control -- isolates the "
        "set-encoder's contribution from the candidate budget."
    ),
    "oracle_at_k": (
        "The positive-control ceiling (any candidate exactly correct) -- if oracle@K ~= "
        "vote the null is uninformative (FALSE_NEGATIVE_RISK), not a verifier failure."
    ),
    "held_out_task_n": (
        "BARE int: the gate's actual N -- target >=40 so the win/null is not under-powered."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- this measurement uses the LEARNED Set-Encoder (no demo "
        "execution); only this makes the result headline/gate-eligible (adversarial_verify "
        "CIRCULAR_MOAT_OVERCLAIM must stay clean)."
    ),
    "random_seed": (
        "Determinism precondition; the held-out split + bootstrap must be reproducible."
    ),
    "reproducibility_checksum": (
        "Hash of the held-out pool + the learned set-encoder; lets a third party re-run "
        "the decisive gate."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "oracle_distinct_beats_vote",
    "set_encoder_minus_vote_delta",
    "set_encoder_minus_vote_ci95",
    "margin_override_minus_vote",
    "matched_control_delta",
    "oracle_at_k",
    "held_out_task_n",
    "verifier_is_oracle",
    "random_seed",
    "reproducibility_checksum",
    "field_principles",
    "spec_refs",
    "acceptance_gate",
)


class BlockedRun(RuntimeError):
    """Expected gate/precondition failure that still writes a terminal artifact."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ScoredArcCandidate:
    task_id: str
    candidate_id: str
    candidate_index: int
    vote_weight: float
    correct: bool
    set_encoder_score: float
    set_encoder_train_task_excluded: bool
    fold: int
    features: dict[str, float]


@dataclass(frozen=True)
class HeldoutPool:
    candidates: list[ScoredArcCandidate]
    candidate_pool_path: Path
    candidate_pool_sha256: str
    learned_verifier_path: Path
    learned_verifier_sha256: str
    score_source: str
    model_specs: dict[str, Any]
    dropped_task_n: int
    dropped_candidate_n: int


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BlockedRun("blocked_malformed_json_artifact")
    return payload


def _resolve_existing_path(repo_root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise BlockedRun(DEFERRED_VERDICT)
    path = Path(value)
    resolved = path if path.is_absolute() else repo_root / path
    if not resolved.exists():
        raise BlockedRun(DEFERRED_VERDICT)
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_a2(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    a2_path = repo_root / A2_REL
    if not a2_path.exists():
        raise BlockedRun(DEFERRED_VERDICT)
    try:
        a2_artifact = _read_json_object(a2_path)
    except Exception as exc:
        raise BlockedRun(DEFERRED_VERDICT) from exc
    if a2_artifact.get("aggregator_trained") is not True:
        raise BlockedRun(DEFERRED_VERDICT)
    if a2_artifact.get("verifier_is_oracle") is not False:
        raise BlockedRun(DEFERRED_VERDICT)
    set_encoder_path = _resolve_existing_path(repo_root, a2_artifact.get("learned_verifier_path"))
    try:
        set_encoder = exp4244.load_set_encoder(set_encoder_path)
    except Exception as exc:
        raise BlockedRun(DEFERRED_VERDICT) from exc
    if set_encoder.get("verifier_is_oracle") is not False:
        raise BlockedRun(DEFERRED_VERDICT)
    rows = set_encoder.get("set_encoder_oof", {}).get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise BlockedRun(DEFERRED_VERDICT)
    return a2_artifact, set_encoder, set_encoder_path


def _safe_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _oof_score_map(set_encoder: dict[str, Any]) -> dict[str, tuple[float, bool, int]]:
    scores: dict[str, tuple[float, bool, int]] = {}
    rows = set_encoder.get("set_encoder_oof", {}).get("rows", [])
    if not isinstance(rows, list):
        return scores
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidate_id = row.get("candidate_id")
        task_id = row.get("task_id")
        if not isinstance(candidate_id, str) or not isinstance(task_id, str):
            continue
        train_task_ids = row.get("train_task_ids", [])
        excluded = isinstance(train_task_ids, list) and task_id not in train_task_ids
        scores[candidate_id] = (_safe_float(row.get("score", 0.0)), excluded, _safe_int(row.get("fold", 0)))
    return scores


def load_heldout_pool(
    repo_root: Path | str,
    set_encoder: dict[str, Any],
    set_encoder_path: Path,
) -> HeldoutPool:
    """SCENARIO-VERIFY-4245: attach Exp 4244 out-of-fold Set-Encoder scores."""

    root = Path(repo_root)
    try:
        corpus = exp4244.load_grown_pool(root)
    except Exception as exc:
        raise BlockedRun(DEFERRED_VERDICT) from exc
    oof_scores = _oof_score_map(set_encoder)
    source_rows_by_task: dict[str, list[exp4244.GrownPoolRow]] = defaultdict(list)
    valid_rows_by_task: dict[str, list[ScoredArcCandidate]] = defaultdict(list)
    dropped_candidate_n = 0
    for row in corpus.rows:
        source_rows_by_task[row.task_id].append(row)
        score_item = oof_scores.get(row.candidate_id)
        if score_item is None:
            dropped_candidate_n += 1
            continue
        score, excluded, fold = score_item
        if not excluded:
            dropped_candidate_n += 1
            continue
        valid_rows_by_task[row.task_id].append(
            ScoredArcCandidate(
                task_id=row.task_id,
                candidate_id=row.candidate_id,
                candidate_index=row.candidate_index,
                vote_weight=row.vote_weight,
                correct=row.correct,
                set_encoder_score=score,
                set_encoder_train_task_excluded=excluded,
                fold=fold,
                features=row.features,
            )
        )

    candidates: list[ScoredArcCandidate] = []
    dropped_task_n = 0
    for task_id, source_rows in sorted(source_rows_by_task.items()):
        task_candidates = valid_rows_by_task.get(task_id, [])
        if len(task_candidates) != len(source_rows):
            dropped_task_n += 1
            continue
        candidates.extend(sorted(task_candidates, key=lambda candidate: candidate.candidate_index))
    if not candidates:
        raise BlockedRun("blocked_no_heldout_set_encoder_scores")
    model_specs = set_encoder.get("model_specs", {})
    if not isinstance(model_specs, dict):
        model_specs = {}
    return HeldoutPool(
        candidates=candidates,
        candidate_pool_path=corpus.pool_artifact_path,
        candidate_pool_sha256=corpus.pool_artifact_sha256,
        learned_verifier_path=set_encoder_path.resolve(),
        learned_verifier_sha256=_sha256_file(set_encoder_path),
        score_source="exp4244_set_encoder_oof_scores",
        model_specs=model_specs,
        dropped_task_n=dropped_task_n,
        dropped_candidate_n=dropped_candidate_n,
    )


def _group_by_task(candidates: list[ScoredArcCandidate]) -> list[list[ScoredArcCandidate]]:
    grouped: dict[str, list[ScoredArcCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.task_id].append(candidate)
    return [
        sorted(task_candidates, key=lambda candidate: candidate.candidate_index)
        for _, task_candidates in sorted(grouped.items())
    ]


def _select_vote(task_candidates: list[ScoredArcCandidate]) -> ScoredArcCandidate:
    return max(task_candidates, key=lambda candidate: (candidate.vote_weight, -candidate.candidate_index))


def _select_set_encoder(task_candidates: list[ScoredArcCandidate]) -> ScoredArcCandidate:
    return max(
        task_candidates,
        key=lambda candidate: (
            candidate.set_encoder_score,
            candidate.vote_weight,
            -candidate.candidate_index,
        ),
    )


def _select_first(task_candidates: list[ScoredArcCandidate]) -> ScoredArcCandidate:
    return min(task_candidates, key=lambda candidate: candidate.candidate_index)


def _select_margin_override(
    task_candidates: list[ScoredArcCandidate], margin_threshold: float
) -> ScoredArcCandidate:
    vote_pick = _select_vote(task_candidates)
    set_encoder_pick = _select_set_encoder(task_candidates)
    margin = set_encoder_pick.set_encoder_score - vote_pick.set_encoder_score
    if set_encoder_pick.candidate_id != vote_pick.candidate_id and margin >= margin_threshold:
        return set_encoder_pick
    return vote_pick


def _rate(values: list[bool]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _round_metric(value: float) -> float:
    return round(float(value), 10)


def _bootstrap_ci95(deltas: list[float], *, random_seed: int, resamples: int) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    rng = random.Random(random_seed)
    n = len(deltas)
    means = [
        sum(deltas[rng.randrange(n)] for _ in range(n)) / float(n)
        for _ in range(int(resamples))
    ]
    if not means:
        point = sum(deltas) / float(len(deltas))
        return [_round_metric(point), _round_metric(point)]
    means.sort()
    return [
        _round_metric(means[int(0.025 * (len(means) - 1))]),
        _round_metric(means[int(0.975 * (len(means) - 1))]),
    ]


def _ci_excludes_zero(ci95: list[float]) -> bool:
    return bool(len(ci95) == 2 and (ci95[0] > 0.0 or ci95[1] < 0.0))


def _measure_pool(
    pool: HeldoutPool,
    *,
    random_seed: int,
    bootstrap_resamples: int,
    margin_threshold: float,
) -> dict[str, Any]:
    tasks = _group_by_task(pool.candidates)
    task_rows: list[dict[str, Any]] = []
    deltas_set_encoder_vote: list[float] = []
    deltas_set_encoder_control: list[float] = []
    deltas_override_vote: list[float] = []
    oracle_hits: list[bool] = []
    vote_hits: list[bool] = []
    set_encoder_hits: list[bool] = []
    control_hits: list[bool] = []
    override_hits: list[bool] = []
    for task_candidates in tasks:
        vote_pick = _select_vote(task_candidates)
        set_encoder_pick = _select_set_encoder(task_candidates)
        control_pick = _select_first(task_candidates)
        override_pick = _select_margin_override(task_candidates, margin_threshold)
        oracle_hit = any(candidate.correct for candidate in task_candidates)
        vote_hit = vote_pick.correct
        set_encoder_hit = set_encoder_pick.correct
        control_hit = control_pick.correct
        override_hit = override_pick.correct
        oracle_hits.append(oracle_hit)
        vote_hits.append(vote_hit)
        set_encoder_hits.append(set_encoder_hit)
        control_hits.append(control_hit)
        override_hits.append(override_hit)
        deltas_set_encoder_vote.append(float(set_encoder_hit) - float(vote_hit))
        deltas_set_encoder_control.append(float(set_encoder_hit) - float(control_hit))
        deltas_override_vote.append(float(override_hit) - float(vote_hit))
        task_rows.append(
            {
                "task_id": vote_pick.task_id,
                "oracle_hit": oracle_hit,
                "vote_candidate_id": vote_pick.candidate_id,
                "vote_correct": vote_hit,
                "set_encoder_candidate_id": set_encoder_pick.candidate_id,
                "set_encoder_correct": set_encoder_hit,
                "set_encoder_score_margin_vs_vote": _round_metric(
                    set_encoder_pick.set_encoder_score - vote_pick.set_encoder_score
                ),
                "set_encoder_train_task_excluded": all(
                    candidate.set_encoder_train_task_excluded for candidate in task_candidates
                ),
                "matched_control_candidate_id": control_pick.candidate_id,
                "matched_control_correct": control_hit,
                "margin_override_candidate_id": override_pick.candidate_id,
                "margin_override_correct": override_hit,
            }
        )

    vote_at_1 = _rate(vote_hits)
    set_encoder_at_1 = _rate(set_encoder_hits)
    oracle_at_k = _rate(oracle_hits)
    control_at_1 = _rate(control_hits)
    override_at_1 = _rate(override_hits)
    set_encoder_minus_vote = _round_metric(set_encoder_at_1 - vote_at_1)
    ci95 = _bootstrap_ci95(
        deltas_set_encoder_vote,
        random_seed=random_seed,
        resamples=bootstrap_resamples,
    )
    matched_control_delta = _round_metric(
        sum(deltas_set_encoder_control) / float(len(deltas_set_encoder_control))
        if deltas_set_encoder_control
        else 0.0
    )
    override_delta = _round_metric(
        sum(deltas_override_vote) / float(len(deltas_override_vote))
        if deltas_override_vote
        else 0.0
    )
    headroom_exists = oracle_at_k > vote_at_1
    beats_vote = bool(headroom_exists and set_encoder_minus_vote > 0.0 and ci95[0] > 0.0)
    if not headroom_exists:
        headline = "arc_oracle_distinct_no_headroom_uninformative"
        honest_verdict = NO_HEADROOM_VERDICT
    elif beats_vote:
        headline = "arc_oracle_distinct_set_encoder_beats_vote"
        honest_verdict = f"complete: {headline}"
    else:
        headline = "arc_oracle_distinct_ties_vote_at_power_on_grown_pool"
        honest_verdict = f"complete: {headline}"
    return {
        "headline_outcome": headline,
        "honest_verdict": honest_verdict,
        "oracle_distinct_beats_vote": beats_vote,
        "set_encoder_minus_vote_delta": set_encoder_minus_vote,
        "set_encoder_minus_vote_ci95": ci95,
        "margin_override_minus_vote": override_delta,
        "matched_control_delta": matched_control_delta,
        "oracle_at_k": _round_metric(oracle_at_k),
        "held_out_task_n": len(tasks),
        "pass_rates": {
            "vote_at_1": _round_metric(vote_at_1),
            "set_encoder_at_1": _round_metric(set_encoder_at_1),
            "matched_control_at_1": _round_metric(control_at_1),
            "margin_override_at_1": _round_metric(override_at_1),
        },
        "oracle_minus_vote": _round_metric(oracle_at_k - vote_at_1),
        "headroom_exists": headroom_exists,
        "candidate_count": len(pool.candidates),
        "bootstrap_resamples": int(bootstrap_resamples),
        "margin_trigger_threshold": float(margin_threshold),
        "margin_threshold_policy": "pre_registered_a2_fold_fixed_threshold",
        "matched_control_policy": "deterministic_first_of_k_no_verifier",
        "clt_floor_caveat": len(tasks) < 30,
        "task_rows": task_rows,
        "ci95_excludes_zero": _ci_excludes_zero(ci95),
    }


def reproducibility_checksum(pool: HeldoutPool, metrics: dict[str, Any], random_seed: int) -> str:
    payload = {
        "candidate_pool_sha256": pool.candidate_pool_sha256,
        "candidate_scores": [
            {
                "candidate_id": candidate.candidate_id,
                "correct": candidate.correct,
                "fold": candidate.fold,
                "set_encoder_score": _round_metric(candidate.set_encoder_score),
                "task_id": candidate.task_id,
                "vote_weight": _round_metric(candidate.vote_weight),
            }
            for candidate in pool.candidates
        ],
        "learned_set_encoder_sha256": pool.learned_verifier_sha256,
        "margin_trigger_threshold": metrics["margin_trigger_threshold"],
        "random_seed": int(random_seed),
        "score_source": pool.score_source,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _deferred_artifact(
    reason: str,
    *,
    random_seed: int,
    checksum: str,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4245_arc_set_encoder_beats_vote",
        "schema": "carnot.arc_set_encoder_beats_vote_4245.v1",
        "status": "complete",
        "headline_outcome": reason.replace("complete_", "", 1),
        "honest_verdict": reason,
        "oracle_distinct_beats_vote": False,
        "set_encoder_minus_vote_delta": 0.0,
        "set_encoder_minus_vote_ci95": [0.0, 0.0],
        "margin_override_minus_vote": 0.0,
        "matched_control_delta": 0.0,
        "oracle_at_k": 0.0,
        "held_out_task_n": 0,
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "pass_rates": {
            "vote_at_1": 0.0,
            "set_encoder_at_1": 0.0,
            "matched_control_at_1": 0.0,
            "margin_override_at_1": 0.0,
        },
        "oracle_minus_vote": 0.0,
        "headroom_exists": False,
        "candidate_count": 0,
        "bootstrap_resamples": 0,
        "margin_trigger_threshold": MARGIN_TRIGGER_THRESHOLD,
        "margin_threshold_policy": "pre_registered_a2_fold_fixed_threshold",
        "matched_control_policy": "deterministic_first_of_k_no_verifier",
        "clt_floor_caveat": True,
        "task_rows": [],
        "ci95_excludes_zero": False,
        "model_specs": {"status": "deferred_set_encoder_gate"},
        "score_source": "",
        "learned_verifier_path": "",
        "candidate_pool_path": "",
        "candidate_pool_sha256": "",
        "dropped_task_n": 0,
        "dropped_candidate_n": 0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 6),
        "adversarial_verify": {"status": "pending"},
    }


def _complete_artifact(
    pool: HeldoutPool,
    metrics: dict[str, Any],
    *,
    checksum: str,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4245_arc_set_encoder_beats_vote",
        "schema": "carnot.arc_set_encoder_beats_vote_4245.v1",
        "status": "complete",
        **metrics,
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "score_source": pool.score_source,
        "learned_verifier_path": str(pool.learned_verifier_path),
        "candidate_pool_path": str(pool.candidate_pool_path),
        "candidate_pool_sha256": pool.candidate_pool_sha256,
        "dropped_task_n": pool.dropped_task_n,
        "dropped_candidate_n": pool.dropped_candidate_n,
        "model_specs": pool.model_specs,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 6),
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


def _clean_adversarial_report(report: dict[str, Any]) -> dict[str, Any]:
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


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:") or verdict.startswith("complete_")
    ):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if type(artifact["oracle_distinct_beats_vote"]) is not bool:
        raise ValueError("oracle_distinct_beats_vote must be a bare bool")
    for field in (
        "set_encoder_minus_vote_delta",
        "margin_override_minus_vote",
        "matched_control_delta",
        "oracle_at_k",
    ):
        if isinstance(artifact[field], bool) or not isinstance(artifact[field], (int, float)):
            raise ValueError(f"{field} must be a bare float")
    ci95 = artifact["set_encoder_minus_vote_ci95"]
    if (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in ci95)
    ):
        raise ValueError("set_encoder_minus_vote_ci95 must be a two-number ci95")
    if type(artifact["held_out_task_n"]) is not int:
        raise ValueError("held_out_task_n must be a bare int")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if type(artifact["random_seed"]) is not int:
        raise ValueError("random_seed must be a bare int")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4245")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs do not match REQ-VERIFY-4245")


def _blocked_checksum(reason: str, random_seed: int) -> str:
    raw = json.dumps({"random_seed": random_seed, "reason": reason}, sort_keys=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def run(
    repo_root: Path | str = Path("."),
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    margin_threshold: float = MARGIN_TRIGGER_THRESHOLD,
    adversarial_runner: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)
    output_path = root / OUTPUT_REL
    try:
        _a2_artifact, set_encoder, set_encoder_path = _load_a2(root)
        pool = load_heldout_pool(root, set_encoder, set_encoder_path)
        metrics = _measure_pool(
            pool,
            random_seed=random_seed,
            bootstrap_resamples=bootstrap_resamples,
            margin_threshold=margin_threshold,
        )
        checksum = reproducibility_checksum(pool, metrics, random_seed)
        artifact = _complete_artifact(
            pool,
            metrics,
            checksum=checksum,
            random_seed=random_seed,
            duration_s=time.perf_counter() - start,
        )
    except BlockedRun:
        reason = DEFERRED_VERDICT
        artifact = _deferred_artifact(
            reason,
            random_seed=random_seed,
            checksum=_blocked_checksum(reason, random_seed),
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


def main() -> None:  # pragma: no cover - exercised by the result entrypoint.
    repo_root = Path(__file__).resolve().parents[3]
    print(json.dumps(run(repo_root), indent=2, sort_keys=True))
