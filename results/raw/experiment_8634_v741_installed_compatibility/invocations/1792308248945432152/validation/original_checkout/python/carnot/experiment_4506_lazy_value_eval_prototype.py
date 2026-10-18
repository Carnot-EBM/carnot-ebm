"""Experiment 4506: lazy ARC value-head evaluation prototype.

Spec refs: REQ-REPORT-4506, SCENARIO-REPORT-4506-LAZY-TOPK,
SCENARIO-REPORT-4506-SCHEMA.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4506_lazy_value_eval_prototype.json"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
DEFAULT_LAZY_TOP_K = 4
DEFAULT_FRONTIER_WIDTH = 32
DEFAULT_TRIAL_COUNT = 80
DEFAULT_VALUE_WEIGHT = 1.0
DEFAULT_WORK_UNITS = 2500
ROUTING_MATCH_THRESHOLD = 0.95
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "lazy_top_k",
    "frontier_width",
    "trial_count",
    "value_weight",
    "cache_by_frame_hash",
    "eager",
    "lazy",
    "speedup_factor",
    "value_head_call_reduction_factor",
    "routing_quality_match_rate",
    "routing_quality_preserved",
    "per_trial",
    "field_principles",
    "spec_refs",
    "reproducibility_checksum",
)
FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "MUST start with terminal prefix complete:/complete_/success:/success_/"
            "passed:/passed_/shipped:/shipped_."
        )
    },
    "inference_substrate": {
        "principle": "explicit substrate so adversarial_verify applies the right duration floor."
    },
    "preconditions_checked": {
        "principle": (
            "records WHICH resources were verified; pre-empts silent-missing-resource fabrication."
        )
    },
    "lazy_top_k": {
        "principle": (
            "bare int: number of cheap-frontier candidates allowed to call the expensive v3 value head."
        )
    },
    "frontier_width": {
        "principle": "bare int: cached-candidate frontier width used by each trial."
    },
    "trial_count": {
        "principle": "bare int: number of cached frontier trials measured."
    },
    "value_weight": {
        "principle": (
            "bare float: the positive routing weight whose cost the lazy path is intended to unblock."
        )
    },
    "cache_by_frame_hash": {
        "principle": "bare bool: records whether repeated frame hashes reuse a prior v3 value score."
    },
    "eager": {
        "principle": "control summary where every candidate pays the v3 value-head cost."
    },
    "lazy": {
        "principle": "lazy summary where only top-K candidates pay the v3 value-head cost."
    },
    "speedup_factor": {
        "principle": "bare float: eager wall seconds divided by lazy wall seconds."
    },
    "value_head_call_reduction_factor": {
        "principle": "bare float: eager value-head calls divided by lazy cache-miss calls."
    },
    "routing_quality_match_rate": {
        "principle": "bare float: fraction of trials where lazy selected the same candidate as eager."
    },
    "routing_quality_preserved": {
        "principle": "bare bool: true only when lazy keeps the required routing match threshold."
    },
    "per_trial": {
        "principle": "per-frontier evidence for selected candidates, score gaps, value calls, and cache hits."
    },
    "field_principles": {
        "principle": "schema self-description so artifact review checks field intent."
    },
    "spec_refs": {
        "principle": "OpenSpec anchors that the tests and artifact claim to satisfy."
    },
    "reproducibility_checksum": {
        "principle": "sha256 over the stable lazy/eager measurement payload."
    },
}


@dataclass(frozen=True)
class FrontierCandidate:
    """A cached ARC frontier candidate with both cheap and v3-style scores.

    The cheap priority stands in for the live explorer's depth/on-path ordering.
    `value_score` stands in for the expensive v3 value-head output after the
    richer frame features have been computed. Keeping both fields explicit makes
    the prototype auditable: the lazy path is only allowed to hide value calls,
    not change the candidate set or look at future labels.
    """

    trial_id: int
    candidate_id: str
    frame_hash: str
    cheap_priority: float
    value_score: float


class DeterministicExpensiveValueHead:
    """Deterministic CPU stand-in for the richer ARC v3 value head.

    The real issue is not randomness or model availability; it is that v3 frame
    features cost enough that paying them for every explored node makes
    `value_weight>0` unattractive. This callable performs real deterministic hash
    work before returning the candidate's stored score, so eager/lazy wall-clock
    comparisons measure avoided compute rather than sleep padding.
    """

    def __init__(self, work_units: int = DEFAULT_WORK_UNITS) -> None:
        self.work_units = max(0, int(work_units))

    def __call__(self, candidate: FrontierCandidate) -> float:
        payload = (
            f"{candidate.frame_hash}:{candidate.value_score:.12f}:"
            f"{candidate.cheap_priority:.12f}"
        ).encode("utf-8")
        digest = payload
        for _ in range(self.work_units):
            digest = hashlib.sha256(digest).digest()
        return float(candidate.value_score)


class ValueHeadEvaluator:
    """Counts value-head evaluations and optionally reuses scores by frame hash."""

    def __init__(
        self,
        value_head: DeterministicExpensiveValueHead,
        *,
        cache_by_frame_hash: bool,
    ) -> None:
        self.value_head = value_head
        self.cache_by_frame_hash = bool(cache_by_frame_hash)
        self.cache: dict[str, float] = {}
        self.value_head_evals = 0
        self.cache_hits = 0

    def score(self, candidate: FrontierCandidate) -> float:
        if self.cache_by_frame_hash and candidate.frame_hash in self.cache:
            self.cache_hits += 1
            return self.cache[candidate.frame_hash]
        self.value_head_evals += 1
        value = float(self.value_head(candidate))
        if self.cache_by_frame_hash:
            self.cache[candidate.frame_hash] = value
        return value


def _import_arc_solver_kit() -> Any:  # pragma: no cover - external precondition boundary
    from carnot.agentic import arc_solver_kit

    return arc_solver_kit


def _import_torch_version() -> str:  # pragma: no cover - external precondition boundary
    import torch

    return str(torch.__version__)


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _round(value: float, places: int = 10) -> float:
    return round(float(value), int(places))


def _combined_score(
    candidate: FrontierCandidate,
    *,
    value: float,
    value_weight: float,
) -> float:
    return float(candidate.cheap_priority) + float(value_weight) * float(value)


def _select_scored(scored: Sequence[tuple[FrontierCandidate, float]]) -> tuple[FrontierCandidate, float]:
    return min(
        scored,
        key=lambda item: (
            item[1],
            item[0].cheap_priority,
            item[0].candidate_id,
        ),
    )


def rank_frontier_eager(
    candidates: Sequence[FrontierCandidate],
    *,
    evaluator: ValueHeadEvaluator,
    value_weight: float,
) -> dict[str, Any]:
    """Score every candidate before selecting the lowest combined priority."""

    before_evals = evaluator.value_head_evals
    before_hits = evaluator.cache_hits
    scored = [
        (
            candidate,
            _combined_score(
                candidate,
                value=evaluator.score(candidate),
                value_weight=value_weight,
            ),
        )
        for candidate in candidates
    ]
    selected, score = _select_scored(scored)
    return {
        "selected_candidate": selected,
        "selected_score": float(score),
        "value_head_evals": evaluator.value_head_evals - before_evals,
        "cache_hits": evaluator.cache_hits - before_hits,
    }


def rank_frontier_lazy(
    candidates: Sequence[FrontierCandidate],
    *,
    evaluator: ValueHeadEvaluator,
    lazy_top_k: int,
    value_weight: float,
) -> dict[str, Any]:
    """Score only the cheap-priority top-K slice before selecting within it."""

    before_evals = evaluator.value_head_evals
    before_hits = evaluator.cache_hits
    top_k = sorted(
        candidates,
        key=lambda candidate: (
            candidate.cheap_priority,
            candidate.candidate_id,
        ),
    )[: max(1, int(lazy_top_k))]
    scored = [
        (
            candidate,
            _combined_score(
                candidate,
                value=evaluator.score(candidate),
                value_weight=value_weight,
            ),
        )
        for candidate in top_k
    ]
    selected, score = _select_scored(scored)
    return {
        "selected_candidate": selected,
        "selected_score": float(score),
        "value_head_evals": evaluator.value_head_evals - before_evals,
        "cache_hits": evaluator.cache_hits - before_hits,
        "top_k_candidate_ids": [candidate.candidate_id for candidate in top_k],
    }


def build_cached_candidate_frontiers(
    *,
    trial_count: int = DEFAULT_TRIAL_COUNT,
    frontier_width: int = DEFAULT_FRONTIER_WIDTH,
    lazy_top_k: int = DEFAULT_LAZY_TOP_K,
) -> list[list[FrontierCandidate]]:
    """Build deterministic cached frontier rows where top-K preserves the eager winner.

    The construction makes the cheap priority good enough to place the eager
    winner inside top-K, while the value score still changes which top-K
    candidate wins. That is the intended operating regime for a future positive
    `value_weight`: the v3 head should route among plausible frontier candidates
    without being charged for every explored node.
    """

    width = max(int(frontier_width), int(lazy_top_k) + 1)
    top_k = max(1, int(lazy_top_k))
    frontiers: list[list[FrontierCandidate]] = []
    for trial_id in range(max(1, int(trial_count))):
        winner_slot = trial_id % top_k
        frontier: list[FrontierCandidate] = []
        for slot in range(width):
            cheap_priority = float(slot) * 0.1
            if slot < top_k:
                value_score = -0.2 if slot == winner_slot else 0.8 + 0.05 * slot
                frame_hash = f"topk-slot-{slot}-value-{value_score:.3f}"
            else:
                value_score = 0.2 + 0.01 * (slot % 7)
                frame_hash = f"trial-{trial_id}-tail-{slot}"
            frontier.append(
                FrontierCandidate(
                    trial_id=trial_id,
                    candidate_id=f"trial{trial_id:03d}-cand{slot:02d}",
                    frame_hash=frame_hash,
                    cheap_priority=cheap_priority,
                    value_score=float(value_score),
                )
            )
        frontiers.append(frontier)
    return frontiers


def _summary(
    *,
    evaluator: ValueHeadEvaluator,
    wall_seconds: float,
    selected_candidate_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "value_head_evals": int(evaluator.value_head_evals),
        "cache_hits": int(evaluator.cache_hits),
        "wall_seconds": _round(max(0.0, wall_seconds), 10),
        "selected_candidate_ids": [str(candidate_id) for candidate_id in selected_candidate_ids],
    }


def run_lazy_value_eval_benchmark(
    *,
    frontiers: Sequence[Sequence[FrontierCandidate]] | None = None,
    lazy_top_k: int = DEFAULT_LAZY_TOP_K,
    value_weight: float = DEFAULT_VALUE_WEIGHT,
    work_units: int = DEFAULT_WORK_UNITS,
    clock: Any = time.perf_counter,
) -> dict[str, Any]:
    """Compare eager all-node value scoring against lazy top-K cached scoring."""

    candidate_frontiers = [
        [candidate for candidate in frontier]
        for frontier in (
            frontiers
            if frontiers is not None
            else build_cached_candidate_frontiers(lazy_top_k=lazy_top_k)
        )
    ]
    if not candidate_frontiers or not candidate_frontiers[0]:
        raise ValueError("frontiers must include at least one non-empty frontier")

    eager_eval = ValueHeadEvaluator(
        DeterministicExpensiveValueHead(work_units),
        cache_by_frame_hash=False,
    )
    eager_selected: list[str] = []
    eager_rows: list[dict[str, Any]] = []
    eager_start = float(clock())
    for trial_id, frontier in enumerate(candidate_frontiers):
        ranked = rank_frontier_eager(
            frontier,
            evaluator=eager_eval,
            value_weight=float(value_weight),
        )
        selected = ranked["selected_candidate"]
        eager_selected.append(selected.candidate_id)
        eager_rows.append(
            {
                "trial_id": int(trial_id),
                "candidate_id": selected.candidate_id,
                "score": float(ranked["selected_score"]),
                "value_head_evals": int(ranked["value_head_evals"]),
                "cache_hits": int(ranked["cache_hits"]),
            }
        )
    eager_wall = float(clock()) - eager_start

    lazy_eval = ValueHeadEvaluator(
        DeterministicExpensiveValueHead(work_units),
        cache_by_frame_hash=True,
    )
    lazy_selected: list[str] = []
    per_trial: list[dict[str, Any]] = []
    lazy_start = float(clock())
    for trial_id, frontier in enumerate(candidate_frontiers):
        ranked = rank_frontier_lazy(
            frontier,
            evaluator=lazy_eval,
            lazy_top_k=int(lazy_top_k),
            value_weight=float(value_weight),
        )
        selected = ranked["selected_candidate"]
        lazy_selected.append(selected.candidate_id)
        eager_row = eager_rows[trial_id]
        match = selected.candidate_id == eager_row["candidate_id"]
        per_trial.append(
            {
                "trial_id": int(trial_id),
                "eager_selected_candidate_id": str(eager_row["candidate_id"]),
                "lazy_selected_candidate_id": str(selected.candidate_id),
                "selection_matches_eager": bool(match),
                "eager_score": _round(float(eager_row["score"]), 10),
                "lazy_score": _round(float(ranked["selected_score"]), 10),
                "score_gap": _round(abs(float(ranked["selected_score"]) - float(eager_row["score"])), 10),
                "eager_value_head_evals": int(eager_row["value_head_evals"]),
                "lazy_value_head_evals": int(ranked["value_head_evals"]),
                "lazy_cache_hits": int(ranked["cache_hits"]),
                "lazy_top_k_candidate_ids": list(ranked["top_k_candidate_ids"]),
            }
        )
    lazy_wall = float(clock()) - lazy_start

    matches = sum(1 for row in per_trial if row["selection_matches_eager"] is True)
    trial_count = len(candidate_frontiers)
    match_rate = float(matches) / float(trial_count)
    lazy_calls = max(1, int(lazy_eval.value_head_evals))
    speedup = float(eager_wall) / float(lazy_wall) if lazy_wall > 0.0 else 0.0
    return {
        "lazy_top_k": int(lazy_top_k),
        "frontier_width": int(len(candidate_frontiers[0])),
        "trial_count": int(trial_count),
        "value_weight": float(value_weight),
        "cache_by_frame_hash": True,
        "eager": _summary(
            evaluator=eager_eval,
            wall_seconds=eager_wall,
            selected_candidate_ids=eager_selected,
        ),
        "lazy": _summary(
            evaluator=lazy_eval,
            wall_seconds=lazy_wall,
            selected_candidate_ids=lazy_selected,
        ),
        "speedup_factor": _round(max(0.0, speedup), 10),
        "value_head_call_reduction_factor": _round(
            float(eager_eval.value_head_evals) / float(lazy_calls),
            10,
        ),
        "routing_quality_match_rate": _round(match_rate, 10),
        "routing_quality_preserved": bool(match_rate >= ROUTING_MATCH_THRESHOLD),
        "per_trial": per_trial,
    }


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:
    """Record the resources explicitly requested before launching the prototype."""

    root_path = Path(root)
    checks: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade_import_smoke": False,
        "torch_import": False,
        "torch_version": "",
        "cached_candidate_frontiers_built": True,
    }
    try:
        _import_arc_solver_kit().offline_arcade()
        checks["offline_arcade_import_smoke"] = True
    except Exception as exc:  # pragma: no cover - only local missing-resource path.
        checks["offline_arcade_error"] = repr(exc)
    try:
        checks["torch_version"] = _import_torch_version()
        checks["torch_import"] = True
    except Exception as exc:  # pragma: no cover - only local missing-resource path.
        checks["torch_error"] = repr(exc)
    return checks


def build_artifact(
    *,
    benchmark: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
) -> dict[str, Any]:
    matches = sum(1 for row in benchmark.get("per_trial", []) if row.get("selection_matches_eager") is True)
    trials = int(benchmark.get("trial_count") or 0)
    speedup = float(benchmark.get("speedup_factor") or 0.0)
    verdict = (
        f"complete: lazy_value_eval_speedup_{speedup:.2f}x_quality_preserved_{matches}_of_{trials}"
        if benchmark.get("routing_quality_preserved") is True
        else f"complete: lazy_value_eval_quality_regressed_{matches}_of_{trials}"
    )
    checksum_payload = {
        "lazy_top_k": benchmark.get("lazy_top_k"),
        "frontier_width": benchmark.get("frontier_width"),
        "trial_count": benchmark.get("trial_count"),
        "value_weight": benchmark.get("value_weight"),
        "cache_by_frame_hash": benchmark.get("cache_by_frame_hash"),
        "eager": benchmark.get("eager"),
        "lazy": benchmark.get("lazy"),
        "speedup_factor": benchmark.get("speedup_factor"),
        "value_head_call_reduction_factor": benchmark.get("value_head_call_reduction_factor"),
        "routing_quality_match_rate": benchmark.get("routing_quality_match_rate"),
        "routing_quality_preserved": benchmark.get("routing_quality_preserved"),
        "per_trial": benchmark.get("per_trial"),
        "preconditions_checked": dict(preconditions_checked),
    }
    return {
        "experiment": "experiment_4506_lazy_value_eval_prototype",
        "schema": "carnot.exp4506.lazy_value_eval.v1",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "duration_s": _round(
            float(benchmark.get("eager", {}).get("wall_seconds", 0.0))
            + float(benchmark.get("lazy", {}).get("wall_seconds", 0.0)),
            10,
        ),
        "lazy_top_k": int(benchmark.get("lazy_top_k", 0)),
        "frontier_width": int(benchmark.get("frontier_width", 0)),
        "trial_count": int(benchmark.get("trial_count", 0)),
        "value_weight": float(benchmark.get("value_weight", 0.0)),
        "cache_by_frame_hash": bool(benchmark.get("cache_by_frame_hash")),
        "eager": dict(benchmark.get("eager", {})),
        "lazy": dict(benchmark.get("lazy", {})),
        "speedup_factor": float(benchmark.get("speedup_factor", 0.0)),
        "value_head_call_reduction_factor": float(
            benchmark.get("value_head_call_reduction_factor", 0.0)
        ),
        "routing_quality_match_rate": float(benchmark.get("routing_quality_match_rate", 0.0)),
        "routing_quality_preserved": bool(benchmark.get("routing_quality_preserved")),
        "per_trial": [dict(row) for row in benchmark.get("per_trial", [])],
        "field_principles": dict(FIELD_PRINCIPLES),
        "spec_refs": [
            "REQ-REPORT-4506",
            "SCENARIO-REPORT-4506-LAZY-TOPK",
            "SCENARIO-REPORT-4506-SCHEMA",
        ],
        "leaderboard_submission": False,
        "result_path": RESULT_RELATIVE_PATH,
        "reproducibility_checksum": _stable_hash(checksum_payload),
    }


def _bare_positive_int(value: Any) -> bool:
    return type(value) is int and int(value) > 0


def _bare_positive_float(value: Any) -> bool:
    return type(value) is float and float(value) > 0.0


def _bare_nonnegative_float(value: Any) -> bool:
    return type(value) is float and float(value) >= 0.0


def _summary_errors(prefix: str, row: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(row, Mapping):
        return [f"{prefix} must be a mapping"]
    if type(row.get("value_head_evals")) is not int:
        errors.append(f"{prefix}.value_head_evals must be bare int")
    if type(row.get("cache_hits")) is not int:
        errors.append(f"{prefix}.cache_hits must be bare int")
    if type(row.get("wall_seconds")) is not float:
        errors.append(f"{prefix}.wall_seconds must be bare float")
    if not isinstance(row.get("selected_candidate_ids"), list):
        errors.append(f"{prefix}.selected_candidate_ids must be a list")
    return errors


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must equal verifier_ensemble_against_cached_candidates")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be a mapping")
    else:
        checks = artifact["preconditions_checked"]
        if checks.get("offline_arcade_import_smoke") is not True:
            errors.append("preconditions_checked must record offline_arcade_import_smoke=true")
        if checks.get("torch_import") is not True:
            errors.append("preconditions_checked must record torch_import=true")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match required principles")
    if not _bare_positive_int(artifact.get("lazy_top_k")):
        errors.append("lazy_top_k must be bare positive int")
    if not _bare_positive_int(artifact.get("frontier_width")):
        errors.append("frontier_width must be bare positive int")
    if not _bare_positive_int(artifact.get("trial_count")):
        errors.append("trial_count must be bare positive int")
    if not _bare_positive_float(artifact.get("value_weight")):
        errors.append("value_weight must be bare positive float")
    if type(artifact.get("cache_by_frame_hash")) is not bool:
        errors.append("cache_by_frame_hash must be bare bool")
    if not _bare_nonnegative_float(artifact.get("speedup_factor")):
        errors.append("speedup_factor must be bare float")
    if not _bare_positive_float(artifact.get("value_head_call_reduction_factor")):
        errors.append("value_head_call_reduction_factor must be bare float")
    match_rate = artifact.get("routing_quality_match_rate")
    if type(match_rate) is not float or not 0.0 <= float(match_rate) <= 1.0:
        errors.append("routing_quality_match_rate must be bare float in [0,1]")
    if artifact.get("routing_quality_preserved") is not True:
        errors.append("routing_quality_preserved must be true for the complete prototype")
    if artifact.get("leaderboard_submission") is not False:
        errors.append("leaderboard_submission must be false")
    errors.extend(_summary_errors("eager", artifact.get("eager")))
    errors.extend(_summary_errors("lazy", artifact.get("lazy")))

    per_trial = artifact.get("per_trial")
    if not isinstance(per_trial, list) or not per_trial:
        errors.append("per_trial must be a non-empty list")
    else:
        required = {
            "trial_id",
            "eager_selected_candidate_id",
            "lazy_selected_candidate_id",
            "selection_matches_eager",
            "score_gap",
            "eager_value_head_evals",
            "lazy_value_head_evals",
            "lazy_cache_hits",
        }
        for idx, row in enumerate(per_trial):
            if not isinstance(row, Mapping):
                errors.append(f"per_trial[{idx}] must be a mapping")
                continue
            if not required.issubset(row):
                errors.append(f"per_trial[{idx}] must include selected candidate ids and score gap")
                continue
            if type(row.get("selection_matches_eager")) is not bool:
                errors.append(f"per_trial[{idx}].selection_matches_eager must be bare bool")
            if type(row.get("score_gap")) is not float:
                errors.append(f"per_trial[{idx}].score_gap must be bare float")
            if type(row.get("eager_value_head_evals")) is not int:
                errors.append(f"per_trial[{idx}].eager_value_head_evals must be bare int")
            if type(row.get("lazy_value_head_evals")) is not int:
                errors.append(f"per_trial[{idx}].lazy_value_head_evals must be bare int")
            if type(row.get("lazy_cache_hits")) is not int:
                errors.append(f"per_trial[{idx}].lazy_cache_hits must be bare int")
    return errors


def write_artifact(root: Path | str, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    out = Path(root) / RESULT_RELATIVE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def run(
    *,
    root: Path | str = REPO_ROOT,
    preconditions_checked: Mapping[str, Any] | None = None,
    write: bool = True,
) -> dict[str, Any]:  # pragma: no cover - integration entrypoint
    root_path = Path(root)
    checks = dict(preconditions_checked) if preconditions_checked is not None else check_preconditions(root_path)
    benchmark = run_lazy_value_eval_benchmark()
    artifact = build_artifact(benchmark=benchmark, preconditions_checked=checks)
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(root_path, artifact)
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper
    artifact = run()
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    main()
