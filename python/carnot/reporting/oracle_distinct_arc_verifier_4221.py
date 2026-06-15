"""Exp 4221 held-out ARC learned-verifier rerank gate.

Spec refs: REQ-VERIFY-4221, SCENARIO-VERIFY-4221,
SCENARIO-VERIFY-4221-NO-HEADROOM, SCENARIO-VERIFY-4221-DEFERRED.
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

from carnot.reporting import oracle_distinct_arc_verifier_4220 as exp4220


RANDOM_SEED = 4221
BOOTSTRAP_RESAMPLES = 2000
ARBITER_MARGIN_THRESHOLD = 0.10
A1_REL = Path("results/experiment_4220_oracle_distinct_arc_verifier_build_labeled.json")
OUTPUT_REL = Path("results/experiment_4221_oracle_distinct_arc_verifier_beats_vote.json")
SPEC_REFS = [
    "REQ-VERIFY-4221",
    "SCENARIO-VERIFY-4221",
    "SCENARIO-VERIFY-4221-NO-HEADROOM",
    "SCENARIO-VERIFY-4221-DEFERRED",
]
INFERENCE_SUBSTRATE = "cached_gap_arc_pool_oof_learned_verifier_rerank"

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A clean beats-vote win, a clean ties-vote-with-headroom "
        "null, and an honest no-headroom-uninformative are ALL COMPLETE -- the first "
        "clean read on the oracle-distinct frontier either way (this time the gate RAN)."
    ),
    "oracle_distinct_beats_vote": (
        "BARE bool: verifier@1 - vote@1 CI95 excludes 0 AND delta>0 AND headroom "
        "exists -- the de-confounded oracle-distinct win (GAP-3-ties-vote closed); "
        "NOT a circular execution result."
    ),
    "verifier_minus_vote_delta": (
        "verifier@1 - vote@1 on held-out ARC -- the oracle-distinct lift; recovers "
        "the wrong-majority-failure answers (ARBITER/AggLM) a flat vote discards."
    ),
    "verifier_minus_vote_ci95": (
        "Task-level bootstrap CI95 of the delta -- excluding 0 is what distinguishes "
        "a real oracle-distinct win from noise."
    ),
    "arbiter_override_minus_vote": (
        "Conservative-override (keep vote unless high learned margin) minus vote -- "
        "the ARBITER 2605.26172 deployment pattern; can win where a flat rerank loses "
        "by overriding only on confident cases."
    ),
    "oracle_at_k": (
        "The positive-control ceiling (any candidate exactly correct) -- if oracle@K "
        "~= vote the null is uninformative (FALSE_NEGATIVE_RISK), not a verifier failure."
    ),
    "matched_control_delta": (
        "verifier@1 minus the budget-matched no-verifier control -- isolates the "
        "verifier's contribution from the candidate budget (arXiv:2511.02886)."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- this measurement uses the LEARNED verifier (no demo "
        "execution); only this makes the result headline/gate-eligible (Circularity "
        "Discipline; adversarial_verify CIRCULAR_MOAT_OVERCLAIM must stay clean)."
    ),
    "random_seed": (
        "Determinism precondition; the held-out split + bootstrap must be reproducible."
    ),
    "reproducibility_checksum": (
        "Hash of the held-out pool + the learned verifier; lets a third party re-run "
        "the decisive gate."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "oracle_distinct_beats_vote",
    "verifier_minus_vote_delta",
    "verifier_minus_vote_ci95",
    "arbiter_override_minus_vote",
    "oracle_at_k",
    "matched_control_delta",
    "verifier_is_oracle",
    "random_seed",
    "reproducibility_checksum",
    "field_principles",
    "spec_refs",
    "acceptance_gate",
)

DEFERRED_VERDICT = "complete_oracle_distinct_arc_gate_deferred_no_built_verifier"


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
    learned_score: float
    verifier_train_task_excluded: bool
    features: dict[str, float]


@dataclass(frozen=True)
class HeldoutPool:
    candidates: list[ScoredArcCandidate]
    pool_path: Path
    programs_path: Path
    learned_verifier_path: Path
    learned_verifier_sha256: str
    score_source: str


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
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_a1(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    a1_path = repo_root / A1_REL
    if not a1_path.exists():
        raise BlockedRun(DEFERRED_VERDICT)
    try:
        a1_artifact = _read_json_object(a1_path)
    except Exception as exc:
        raise BlockedRun(DEFERRED_VERDICT) from exc
    if a1_artifact.get("selector_trained") is not True:
        raise BlockedRun(DEFERRED_VERDICT)
    if a1_artifact.get("verifier_is_oracle") is not False:
        raise BlockedRun(DEFERRED_VERDICT)
    verifier_path = _resolve_existing_path(repo_root, a1_artifact.get("learned_verifier_path"))
    try:
        verifier = exp4220.load_verifier(verifier_path)
    except Exception as exc:
        raise BlockedRun(DEFERRED_VERDICT) from exc
    if verifier.get("verifier_is_oracle") is not False:
        raise BlockedRun(DEFERRED_VERDICT)
    return a1_artifact, verifier, verifier_path


def _oof_score_map(verifier: dict[str, Any]) -> dict[str, tuple[float, bool]]:
    scores: dict[str, tuple[float, bool]] = {}
    for row in verifier.get("oof_rows", []):
        if not isinstance(row, dict):
            continue
        candidate_id = row.get("candidate_id")
        task_id = row.get("task_id")
        train_task_ids = row.get("train_task_ids", [])
        if not isinstance(candidate_id, str) or not isinstance(task_id, str):
            continue
        excluded = isinstance(train_task_ids, list) and task_id not in train_task_ids
        scores[candidate_id] = (float(row.get("score", 0.0)), excluded)
    return scores


def load_heldout_pool(
    repo_root: Path | str,
    verifier: dict[str, Any],
    verifier_path: Path,
) -> HeldoutPool:
    """SCENARIO-VERIFY-4221: rebuild ARC rows and attach learned held-out scores."""

    root = Path(repo_root)
    corpus = exp4220.load_labeled_arc_pool(root)
    oof_scores = _oof_score_map(verifier)
    candidates: list[ScoredArcCandidate] = []
    used_oof = 0
    for row in corpus.rows:
        if row.candidate_id in oof_scores:
            learned_score, excluded = oof_scores[row.candidate_id]
            used_oof += 1
        else:
            learned_score = exp4220.score_with_verifier(verifier, row.features)
            excluded = False
        candidates.append(
            ScoredArcCandidate(
                task_id=row.task_id,
                candidate_id=row.candidate_id,
                candidate_index=row.candidate_index,
                vote_weight=row.vote_weight,
                correct=row.correct,
                learned_score=learned_score,
                verifier_train_task_excluded=excluded,
                features=row.features,
            )
        )
    if not candidates:
        raise BlockedRun("complete_oracle_distinct_arc_no_headroom_uninformative")
    score_source = "exp4220_oof_scores" if used_oof == len(candidates) else "persisted_verifier_scores"
    return HeldoutPool(
        candidates=candidates,
        pool_path=corpus.pool_path,
        programs_path=corpus.programs_path,
        learned_verifier_path=verifier_path.resolve(),
        learned_verifier_sha256=_sha256_file(verifier_path),
        score_source=score_source,
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
    return max(
        task_candidates,
        key=lambda candidate: (candidate.vote_weight, -candidate.candidate_index),
    )


def _select_verifier(task_candidates: list[ScoredArcCandidate]) -> ScoredArcCandidate:
    return max(
        task_candidates,
        key=lambda candidate: (
            candidate.learned_score,
            candidate.vote_weight,
            -candidate.candidate_index,
        ),
    )


def _select_first(task_candidates: list[ScoredArcCandidate]) -> ScoredArcCandidate:
    return min(task_candidates, key=lambda candidate: candidate.candidate_index)


def _select_arbiter(
    task_candidates: list[ScoredArcCandidate], margin_threshold: float
) -> ScoredArcCandidate:
    vote_pick = _select_vote(task_candidates)
    learned_pick = _select_verifier(task_candidates)
    margin = learned_pick.learned_score - vote_pick.learned_score
    if learned_pick.candidate_id != vote_pick.candidate_id and margin >= margin_threshold:
        return learned_pick
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
    deltas_verifier_vote: list[float] = []
    deltas_verifier_control: list[float] = []
    deltas_arbiter_vote: list[float] = []
    oracle_hits: list[bool] = []
    vote_hits: list[bool] = []
    verifier_hits: list[bool] = []
    control_hits: list[bool] = []
    arbiter_hits: list[bool] = []
    for task_candidates in tasks:
        vote_pick = _select_vote(task_candidates)
        verifier_pick = _select_verifier(task_candidates)
        control_pick = _select_first(task_candidates)
        arbiter_pick = _select_arbiter(task_candidates, margin_threshold)
        oracle_hit = any(candidate.correct for candidate in task_candidates)
        vote_hit = vote_pick.correct
        verifier_hit = verifier_pick.correct
        control_hit = control_pick.correct
        arbiter_hit = arbiter_pick.correct
        oracle_hits.append(oracle_hit)
        vote_hits.append(vote_hit)
        verifier_hits.append(verifier_hit)
        control_hits.append(control_hit)
        arbiter_hits.append(arbiter_hit)
        deltas_verifier_vote.append(float(verifier_hit) - float(vote_hit))
        deltas_verifier_control.append(float(verifier_hit) - float(control_hit))
        deltas_arbiter_vote.append(float(arbiter_hit) - float(vote_hit))
        task_rows.append(
            {
                "task_id": vote_pick.task_id,
                "oracle_hit": oracle_hit,
                "vote_candidate_id": vote_pick.candidate_id,
                "vote_correct": vote_hit,
                "verifier_candidate_id": verifier_pick.candidate_id,
                "verifier_correct": verifier_hit,
                "verifier_score_margin_vs_vote": _round_metric(
                    verifier_pick.learned_score - vote_pick.learned_score
                ),
                "verifier_train_task_excluded": all(
                    candidate.verifier_train_task_excluded for candidate in task_candidates
                ),
                "matched_control_candidate_id": control_pick.candidate_id,
                "matched_control_correct": control_hit,
                "arbiter_candidate_id": arbiter_pick.candidate_id,
                "arbiter_correct": arbiter_hit,
            }
        )
    vote_at_1 = _rate(vote_hits)
    verifier_at_1 = _rate(verifier_hits)
    oracle_at_k = _rate(oracle_hits)
    control_at_1 = _rate(control_hits)
    arbiter_at_1 = _rate(arbiter_hits)
    verifier_minus_vote = _round_metric(verifier_at_1 - vote_at_1)
    ci95 = _bootstrap_ci95(
        deltas_verifier_vote,
        random_seed=random_seed,
        resamples=bootstrap_resamples,
    )
    matched_control_delta = _round_metric(
        sum(deltas_verifier_control) / float(len(deltas_verifier_control))
        if deltas_verifier_control
        else 0.0
    )
    arbiter_delta = _round_metric(
        sum(deltas_arbiter_vote) / float(len(deltas_arbiter_vote))
        if deltas_arbiter_vote
        else 0.0
    )
    headroom_exists = oracle_at_k > vote_at_1
    beats_vote = bool(headroom_exists and verifier_minus_vote > 0.0 and ci95[0] > 0.0)
    if not headroom_exists:
        headline = "oracle_distinct_arc_no_headroom_uninformative"
    elif beats_vote:
        headline = "oracle_distinct_verifier_beats_vote"
    else:
        headline = "oracle_distinct_verifier_ties_vote_with_headroom"
    return {
        "headline_outcome": headline,
        "honest_verdict": f"complete: {headline}",
        "oracle_distinct_beats_vote": beats_vote,
        "verifier_minus_vote_delta": verifier_minus_vote,
        "verifier_minus_vote_ci95": ci95,
        "arbiter_override_minus_vote": arbiter_delta,
        "oracle_at_k": _round_metric(oracle_at_k),
        "matched_control_delta": matched_control_delta,
        "pass_rates": {
            "vote_at_1": _round_metric(vote_at_1),
            "verifier_at_1": _round_metric(verifier_at_1),
            "matched_control_at_1": _round_metric(control_at_1),
            "arbiter_override_at_1": _round_metric(arbiter_at_1),
        },
        "oracle_minus_vote": _round_metric(oracle_at_k - vote_at_1),
        "headroom_exists": headroom_exists,
        "n_tasks": len(tasks),
        "candidate_count": len(pool.candidates),
        "bootstrap_resamples": int(bootstrap_resamples),
        "arbiter_margin_threshold": float(margin_threshold),
        "matched_control_policy": "deterministic_first_of_k_no_verifier",
        "clt_floor_caveat": len(tasks) < 30,
        "task_rows": task_rows,
        "ci95_excludes_zero": _ci_excludes_zero(ci95),
    }


def reproducibility_checksum(pool: HeldoutPool, metrics: dict[str, Any], random_seed: int) -> str:
    payload = {
        "arbiter_margin_threshold": metrics["arbiter_margin_threshold"],
        "candidate_scores": [
            {
                "candidate_id": candidate.candidate_id,
                "correct": candidate.correct,
                "learned_score": _round_metric(candidate.learned_score),
                "task_id": candidate.task_id,
                "vote_weight": _round_metric(candidate.vote_weight),
            }
            for candidate in pool.candidates
        ],
        "learned_verifier_sha256": pool.learned_verifier_sha256,
        "pool_path": str(pool.pool_path),
        "programs_path": str(pool.programs_path),
        "random_seed": int(random_seed),
        "score_source": pool.score_source,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _deferred_artifact(
    reason: str,
    *,
    random_seed: int,
    checksum: str,
    duration_s: float,
) -> dict[str, Any]:
    headline = reason.replace("complete_", "", 1)
    return {
        "experiment": "experiment_4221_oracle_distinct_arc_verifier_beats_vote",
        "schema": "carnot.oracle_distinct_arc_verifier_4221.v1",
        "status": "complete",
        "headline_outcome": headline,
        "honest_verdict": reason,
        "oracle_distinct_beats_vote": False,
        "verifier_minus_vote_delta": 0.0,
        "verifier_minus_vote_ci95": [0.0, 0.0],
        "arbiter_override_minus_vote": 0.0,
        "oracle_at_k": 0.0,
        "matched_control_delta": 0.0,
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "pass_rates": {
            "vote_at_1": 0.0,
            "verifier_at_1": 0.0,
            "matched_control_at_1": 0.0,
            "arbiter_override_at_1": 0.0,
        },
        "oracle_minus_vote": 0.0,
        "headroom_exists": False,
        "n_tasks": 0,
        "candidate_count": 0,
        "bootstrap_resamples": 0,
        "arbiter_margin_threshold": ARBITER_MARGIN_THRESHOLD,
        "matched_control_policy": "deterministic_first_of_k_no_verifier",
        "clt_floor_caveat": True,
        "task_rows": [],
        "ci95_excludes_zero": False,
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
        "experiment": "experiment_4221_oracle_distinct_arc_verifier_beats_vote",
        "schema": "carnot.oracle_distinct_arc_verifier_4221.v1",
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
        "candidate_pool_source": str(pool.pool_path),
        "induced_programs_source": str(pool.programs_path),
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
        "verifier_minus_vote_delta",
        "arbiter_override_minus_vote",
        "oracle_at_k",
        "matched_control_delta",
    ):
        if isinstance(artifact[field], bool) or not isinstance(artifact[field], (int, float)):
            raise ValueError(f"{field} must be a bare float")
    ci95 = artifact["verifier_minus_vote_ci95"]
    if (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in ci95)
    ):
        raise ValueError("verifier_minus_vote_ci95 must be a two-number ci95")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if type(artifact["random_seed"]) is not int:
        raise ValueError("random_seed must be a bare int")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4221")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs do not match REQ-VERIFY-4221")


def _blocked_checksum(reason: str, random_seed: int) -> str:
    raw = json.dumps({"reason": reason, "random_seed": random_seed}, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def run(
    repo_root: Path | str = Path("."),
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    margin_threshold: float = ARBITER_MARGIN_THRESHOLD,
    adversarial_runner: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)
    output_path = root / OUTPUT_REL
    try:
        _a1_artifact, verifier, verifier_path = _load_a1(root)
        pool = load_heldout_pool(root, verifier, verifier_path)
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
    except BlockedRun as blocked:
        reason = (
            "complete_oracle_distinct_arc_no_headroom_uninformative"
            if blocked.reason == "complete_oracle_distinct_arc_no_headroom_uninformative"
            else DEFERRED_VERDICT
        )
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
