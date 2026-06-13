"""Exp 4139 decisive Sudoku verifier graft with oracle separation.

Spec refs: REQ-LEARN-4139, SCENARIO-LEARN-4139-NO-HEADROOM,
SCENARIO-LEARN-4139-RERANK, SCENARIO-LEARN-4139-RFT-OR-DEFER.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from carnot import experiment_4107_nanotrm_mechanism_smoke as exp4107
from carnot import experiment_4109_carnot_verifier_graft_sudoku as exp4109
from carnot import experiment_4138_sudoku_accumulate_pass4_convergence_check as exp4138


CandidateSample = exp4109.CandidateSample
CandidatePool = exp4109.CandidatePool

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4139_decisive_verifier_graft_sudoku.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_EXP4138_ARTIFACT = REPO_ROOT / "results" / exp4138.RESULT_FILENAME
DEFAULT_DATA_DIR = REPO_ROOT / "nano-trm" / "data" / "sudoku_extreme_1k_aug_1k"
DEFAULT_HELDOUT_SPLIT = "_valsmall"
RANDOM_SEED = 4139
DEFAULT_MAX_PUZZLES = 64
DEFAULT_K_CANDIDATES = 8
NEAR_FAITHFUL_THRESHOLD = 0.80
PUBLISHED_TARGET = 0.87
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
SPEC_REFS = [
    "REQ-LEARN-4139",
    "SCENARIO-LEARN-4139-NO-HEADROOM",
    "SCENARIO-LEARN-4139-RERANK",
    "SCENARIO-LEARN-4139-RFT-OR-DEFER",
]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "headroom_present",
    "oracle_vs_vote_gap",
    "executable_verifier_is_oracle",
    "executable_oracle_upper_bound",
    "ensemble_rerank_lift_vs_vote",
    "rft_vs_ablation_delta",
    "graft_deferred",
    "verifier_value_added",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest deferral, an A~=B null WITH headroom present, "
        "OR an uninformative no-headroom result are all COMPLETE."
    ),
    "headroom_present": (
        "Bare bool: oracle(best-of-K) > vote pass@1. The positive control -- a "
        "verifier verdict is only meaningful when this is True (else FALSE_NEGATIVE_RISK)."
    ),
    "oracle_vs_vote_gap": (
        "pass@K oracle minus vote pass@1; the size of the selectable headroom any "
        "reranker could capture."
    ),
    "executable_verifier_is_oracle": (
        "Bare bool (true on unique-solution Sudoku): flags that the executable-validity "
        "rerank == oracle by construction, so the capstone NEVER reads "
        "executable_oracle_upper_bound as a generalization claim."
    ),
    "executable_oracle_upper_bound": (
        "The executable-validity rerank lift -- the UPPER BOUND (tautological == "
        "oracle gap), reported for context, explicitly NOT the verifier-value result."
    ),
    "ensemble_rerank_lift_vs_vote": (
        "The NON-oracle Carnot energy/text-stat ensemble rerank lift vs vote, with "
        "CI -- the TRANSFERABLE-verifier headline (the .382 finding was no single "
        "verifier beat vote; this tests the weighted ensemble). Only meaningful "
        "when headroom_present."
    ),
    "rft_vs_ablation_delta": (
        "The de-confounded A-vs-B held-out delta with CI95 (RFT branch only): "
        "isolates the verifier LABEL's training contribution from the vote LABEL's "
        "-- the load-bearing, non-tautological contrast."
    ),
    "graft_deferred": (
        "Bare bool: True if val<0.80 (RFT arm not run). Prevents a meaningless graft "
        "dressed as a result."
    ),
    "verifier_value_added": (
        "Bare bool: did the NON-oracle ensemble rerank OR the RFT de-confound beat "
        "the vote WITH headroom present? THE headline answer + the DiffusionGemma "
        "gate. Defined on the transferable verifier + the LABEL contrast, NEVER "
        "on the executable-oracle rerank."
    ),
    "preconditions_checked": (
        "Records the baseline checkpoint + CUDA verified; pre-empts silent-missing-resource fabrication."
    ),
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One Exp 4139 runtime resource check."""

    resource: str
    available: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BaselineContext:
    """Exp 4138 baseline evidence that decides the Exp 4139 branch."""

    artifact_path: Path
    stable_checkpoint_path: Path
    val_exact_accuracy: float | None
    matches_published_087: bool
    near_faithful_080: bool
    estimated_passes_to_converge: int | None
    raw: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_path": str(self.artifact_path),
            "stable_checkpoint_path": str(self.stable_checkpoint_path),
            "val_exact_accuracy": self.val_exact_accuracy,
            "matches_published_087": self.matches_published_087,
            "near_faithful_080": self.near_faithful_080,
            "estimated_passes_to_converge": self.estimated_passes_to_converge,
        }


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str) and value.strip():
        try:
            number = float(value)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and math.isfinite(value) and value >= 0 and value.is_integer():
        return int(value)
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _checks_to_dicts(checks: Sequence[PreconditionCheck | Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [check.to_dict() if isinstance(check, PreconditionCheck) else dict(check) for check in checks]


def _all_preconditions_available(checks: Sequence[PreconditionCheck | Mapping[str, Any]]) -> bool:
    return all(bool(check.available) if isinstance(check, PreconditionCheck) else check.get("available") is True for check in checks)


def _metric_has_ci(metric: Mapping[str, Any]) -> bool:
    ci95 = metric.get("ci95")
    return (
        isinstance(ci95, Sequence)
        and not isinstance(ci95, (str, bytes))
        and len(ci95) == 2
        and _float_or_none(ci95[0]) is not None
        and _float_or_none(ci95[1]) is not None
    )


def _bootstrap_ci(differences: Sequence[float], *, random_seed: int, resamples: int) -> list[float]:
    if not differences:
        return [0.0, 0.0]
    rng = random.Random(int(random_seed))
    n = len(differences)
    draws = []
    for _ in range(max(int(resamples), 1)):
        draws.append(sum(float(differences[rng.randrange(n)]) for _item in range(n)) / n)
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return [round(float(lo), 6), round(float(hi), 6)]


def _is_exact(candidate_tokens: Sequence[int], label_tokens: Sequence[int]) -> bool:
    return tuple(int(token) for token in candidate_tokens) == tuple(int(token) for token in label_tokens)


def _last_trajectory_val(payload: Mapping[str, Any]) -> float | None:
    trajectory = payload.get("val_trajectory_383")
    if not isinstance(trajectory, Sequence) or isinstance(trajectory, (str, bytes)):
        return None
    for entry in reversed(trajectory):
        if isinstance(entry, Mapping):
            value = _float_or_none(entry.get("val_exact_accuracy"))
            if value is not None:
                return value
    return None


def load_baseline_context(path: str | Path) -> BaselineContext:
    """REQ-LEARN-4139: defensively load the Exp 4138 baseline gate."""

    artifact_path = Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Exp 4138 artifact must be a JSON object")
    stable_value = payload.get("stable_checkpoint_path")
    stable = Path(stable_value) if isinstance(stable_value, str) else exp4138.DEFAULT_STABLE_DIR / "last.ckpt"
    val = _float_or_none(payload.get("val_exact_accuracy"))
    if val is None:
        val = _last_trajectory_val(payload)
    return BaselineContext(
        artifact_path=artifact_path,
        stable_checkpoint_path=stable,
        val_exact_accuracy=val,
        matches_published_087=payload.get("matches_published_087") is True,
        near_faithful_080=payload.get("near_faithful_080") is True,
        estimated_passes_to_converge=_int_or_none(payload.get("estimated_passes_to_converge")),
        raw=dict(payload),
    )


def baseline_runs_rft_arm(baseline: BaselineContext) -> bool:
    """SCENARIO-LEARN-4139-RFT-OR-DEFER: gate the RFT branch at near-faithful."""

    return baseline.matches_published_087 or baseline.near_faithful_080 or (
        baseline.val_exact_accuracy is not None and baseline.val_exact_accuracy >= NEAR_FAITHFUL_THRESHOLD
    )


def build_384_estimate(baseline: BaselineContext | None) -> dict[str, Any]:
    """SCENARIO-LEARN-4139-RFT-OR-DEFER: preserve the .384 convergence estimate."""

    if baseline is None:
        return {
            "destination": ".384",
            "target_val_exact_accuracy": PUBLISHED_TARGET,
            "current_val_exact_accuracy": None,
            "estimated_additional_passes": None,
            "basis": "missing_exp4138_baseline",
        }
    if baseline.val_exact_accuracy is not None and baseline.val_exact_accuracy >= NEAR_FAITHFUL_THRESHOLD:
        estimate = 0
        basis = "already_near_faithful"
    elif baseline.estimated_passes_to_converge is not None:
        estimate = baseline.estimated_passes_to_converge
        basis = "exp4138_estimated_passes_to_converge"
    else:
        estimate = None
        basis = "exp4138_missing_or_config_blocked_estimate"
    return {
        "destination": ".384",
        "target_val_exact_accuracy": PUBLISHED_TARGET,
        "current_val_exact_accuracy": baseline.val_exact_accuracy,
        "estimated_additional_passes": estimate,
        "basis": basis,
    }


def check_preconditions(
    baseline: BaselineContext,
    *,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4107._default_cuda_checker,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
) -> list[PreconditionCheck]:
    """REQ-LEARN-4139: record baseline artifact, checkpoint, and CUDA checks."""

    checks = [
        PreconditionCheck("exp4138_artifact", baseline.artifact_path.is_file(), str(baseline.artifact_path)),
        PreconditionCheck(
            "stable_checkpoint_path",
            bool(str(baseline.stable_checkpoint_path)),
            str(baseline.stable_checkpoint_path),
        ),
    ]
    if baseline.stable_checkpoint_path.is_file():
        try:
            checkpoint_ok, checkpoint_detail = checkpoint_loader(baseline.stable_checkpoint_path)
        except Exception as exc:
            checkpoint_ok, checkpoint_detail = False, f"{type(exc).__name__}: {exc}"
    else:
        checkpoint_ok = False
        checkpoint_detail = f"missing: {baseline.stable_checkpoint_path}"
    checks.append(PreconditionCheck("baseline_checkpoint", bool(checkpoint_ok), str(checkpoint_detail)))

    try:
        cuda_ok, cuda_detail = cuda_checker()
    except Exception as exc:
        cuda_ok, cuda_detail = False, f"{type(exc).__name__}: {exc}"
    checks.append(PreconditionCheck("cuda_available", bool(cuda_ok), str(cuda_detail)))
    return checks


def _box_dims(grid_size: int) -> tuple[int, int]:
    return exp4109._box_dims(grid_size)  # pylint: disable=protected-access


def _unit_duplicate_energy(values: Sequence[int], allowed: set[int]) -> float:
    counts = {digit: 0 for digit in allowed}
    out_of_range = 0
    for value in values:
        digit = int(value)
        if digit in counts:
            counts[digit] += 1
        else:
            out_of_range += 1
    missing_or_duplicate = sum(abs(count - 1) for count in counts.values())
    return float(missing_or_duplicate + out_of_range) / max(float(2 * len(allowed)), 1.0)


def non_oracle_ensemble_score(
    puzzle_tokens: Sequence[int],
    candidate_tokens: Sequence[int],
    *,
    grid_size: int = 9,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4139-RERANK: score without executable exact-validity checks."""

    puzzle = exp4109.decode_tokens(puzzle_tokens, grid_size=grid_size)
    candidate = exp4109.decode_tokens(candidate_tokens, grid_size=grid_size)
    allowed = set(range(1, grid_size + 1))
    box_rows, box_cols = _box_dims(grid_size)

    range_penalty = 0.0
    valid_digit_count = 0
    histogram = {digit: 0 for digit in allowed}
    for row in candidate:
        for value in row:
            digit = int(value)
            if digit in allowed:
                valid_digit_count += 1
                histogram[digit] += 1
            else:
                range_penalty += min(abs(digit - 1), abs(digit - grid_size), 1.0)

    clue_error = 0.0
    clue_count = 0
    for row in range(grid_size):
        for col in range(grid_size):
            clue = int(puzzle[row][col])
            if clue > 0:
                clue_count += 1
                clue_error += ((int(candidate[row][col]) - clue) / max(grid_size - 1, 1)) ** 2

    unit_energy = 0.0
    unit_count = 0
    for row in range(grid_size):
        unit_count += 1
        unit_energy += _unit_duplicate_energy(candidate[row], allowed)
    for col in range(grid_size):
        unit_count += 1
        unit_energy += _unit_duplicate_energy([candidate[row][col] for row in range(grid_size)], allowed)
    for box_row in range(0, grid_size, box_rows):
        for box_col in range(0, grid_size, box_cols):
            unit_count += 1
            values = [
                candidate[row][col]
                for row in range(box_row, box_row + box_rows)
                for col in range(box_col, box_col + box_cols)
            ]
            unit_energy += _unit_duplicate_energy(values, allowed)

    normalized_energy = (
        (range_penalty / max(grid_size * grid_size, 1))
        + (clue_error / max(clue_count, 1))
        + (unit_energy / max(unit_count, 1))
    )
    energy_score = 1.0 / (1.0 + normalized_energy)
    histogram_error = sum(abs(count - grid_size) for count in histogram.values()) / max(2 * grid_size * grid_size, 1)
    digit_text_score = max(0.0, 1.0 - histogram_error)
    digit_text_score = 0.7 * digit_text_score + 0.3 * (valid_digit_count / max(grid_size * grid_size, 1))
    ensemble = (0.72 * energy_score) + (0.28 * digit_text_score)
    return {
        "ensemble_score": round(float(ensemble), 9),
        "continuous_sudoku_energy": round(float(energy_score), 9),
        "digit_text_statistics": round(float(digit_text_score), 9),
        "uses_exact_validity_check": False,
    }


def select_ensemble_candidate(pool: CandidatePool) -> tuple[CandidateSample, dict[str, Any]]:
    """SCENARIO-LEARN-4139-RERANK: select by non-oracle energy/text-stat ensemble."""

    if not pool.candidates:
        raise ValueError("candidate pool is empty")
    grouped = exp4109._vote_groups(pool.candidates)  # pylint: disable=protected-access
    vote_count_by_tokens = {tokens: int(row["count"]) for tokens, row in grouped.items()}
    best: tuple[tuple[float, float, int, int], CandidateSample, dict[str, Any]] | None = None
    for index, candidate in enumerate(pool.candidates):
        score = non_oracle_ensemble_score(pool.puzzle_tokens, candidate.tokens)
        key = (
            float(score["ensemble_score"]),
            float(candidate.trm_score),
            int(vote_count_by_tokens[candidate.token_tuple()]),
            -index,
        )
        if best is None or key > best[0]:
            best = (key, candidate, score)
    assert best is not None
    return best[1], best[2]


def _mean(values: Sequence[bool]) -> float:
    return sum(bool(value) for value in values) / len(values) if values else 0.0


def evaluate_reranks(
    pools: Sequence[CandidatePool],
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = 2000,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4139-RERANK: report oracle upper bound and ensemble rerank."""

    rows: list[dict[str, Any]] = []
    oracle_diffs: list[float] = []
    executable_diffs: list[float] = []
    ensemble_diffs: list[float] = []
    for pool in pools:
        vote = exp4109.select_vote_candidate(pool.candidates)
        executable, executable_score = exp4109.select_verifier_candidate(pool)
        ensemble, ensemble_score = select_ensemble_candidate(pool)
        vote_correct = _is_exact(vote.tokens, pool.label_tokens)
        executable_correct = _is_exact(executable.tokens, pool.label_tokens)
        ensemble_correct = _is_exact(ensemble.tokens, pool.label_tokens)
        oracle_correct = any(_is_exact(candidate.tokens, pool.label_tokens) for candidate in pool.candidates)
        oracle_diffs.append(float(oracle_correct) - float(vote_correct))
        executable_diffs.append(float(executable_correct) - float(vote_correct))
        ensemble_diffs.append(float(ensemble_correct) - float(vote_correct))
        rows.append(
            {
                "puzzle_id": pool.puzzle_id,
                "vote_sample_id": vote.sample_id,
                "executable_sample_id": executable.sample_id,
                "ensemble_sample_id": ensemble.sample_id,
                "vote_correct": vote_correct,
                "oracle_correct": oracle_correct,
                "executable_correct": executable_correct,
                "ensemble_correct": ensemble_correct,
                "executable_score": executable_score.to_dict(),
                "ensemble_score": ensemble_score,
            }
        )

    vote_acc = _mean([row["vote_correct"] for row in rows])
    oracle_acc = _mean([row["oracle_correct"] for row in rows])
    executable_acc = _mean([row["executable_correct"] for row in rows])
    ensemble_acc = _mean([row["ensemble_correct"] for row in rows])
    oracle_gap = round(float(oracle_acc - vote_acc), 6)
    executable_delta = round(float(executable_acc - vote_acc), 6)
    ensemble_delta = round(float(ensemble_acc - vote_acc), 6)
    headroom = oracle_gap > 0.0
    return {
        "metric": "pass@1_exact_accuracy",
        "n_puzzles": len(rows),
        "vote_pass_at_1": round(float(vote_acc), 6),
        "oracle_pass_at_k": round(float(oracle_acc), 6),
        "headroom_present": bool(headroom),
        "false_negative_risk": not headroom,
        "oracle_vs_vote_gap": oracle_gap,
        "executable_verifier_is_oracle": True,
        "executable_oracle_upper_bound": {
            "metric": "pass@1_exact_accuracy",
            "vote_pass_at_1": round(float(vote_acc), 6),
            "executable_oracle_pass_at_1": round(float(executable_acc), 6),
            "delta": executable_delta,
            "ci95": _bootstrap_ci(executable_diffs, random_seed=random_seed, resamples=bootstrap_resamples),
            "interpretation": "oracle_upper_bound_not_verifier_value",
            "executable_verifier_is_oracle": True,
            "tautological_equals_oracle_gap": executable_delta == oracle_gap,
        },
        "ensemble_rerank_lift_vs_vote": {
            "metric": "pass@1_exact_accuracy",
            "vote_pass_at_1": round(float(vote_acc), 6),
            "ensemble_pass_at_1": round(float(ensemble_acc), 6),
            "delta": ensemble_delta,
            "ci95": _bootstrap_ci(ensemble_diffs, random_seed=random_seed + 1, resamples=bootstrap_resamples),
            "score_components": ["continuous_sudoku_energy", "digit_text_statistics"],
            "uses_exact_validity_check": False,
            "meaningful": bool(headroom),
            "status": "measured" if headroom else "uninterpretable_no_headroom",
        },
        "per_puzzle": rows,
        "oracle_ci95": _bootstrap_ci(oracle_diffs, random_seed=random_seed + 2, resamples=bootstrap_resamples),
    }


def deferred_rft_delta(baseline: BaselineContext | None) -> dict[str, Any]:
    """SCENARIO-LEARN-4139-RFT-OR-DEFER: mark the RFT arm as honestly deferred."""

    return {
        "metric": "heldout_exact_accuracy",
        "n_matched": 0,
        "a_exact_accuracy": 0.0,
        "b_exact_accuracy": 0.0,
        "delta": 0.0,
        "ci95": [0.0, 0.0],
        "status": "deferred_baseline_below_0.80",
        "current_val_exact_accuracy": None if baseline is None else baseline.val_exact_accuracy,
        "estimated_passes_to_converge_for_384": build_384_estimate(baseline),
    }


def _not_run_rft_delta(status: str) -> dict[str, Any]:
    return {
        "metric": "heldout_exact_accuracy",
        "n_matched": 0,
        "a_exact_accuracy": 0.0,
        "b_exact_accuracy": 0.0,
        "delta": 0.0,
        "ci95": [0.0, 0.0],
        "status": status,
    }


def _empty_rerank_metrics(status: str) -> dict[str, Any]:
    return {
        "metric": "pass@1_exact_accuracy",
        "n_puzzles": 0,
        "vote_pass_at_1": 0.0,
        "oracle_pass_at_k": 0.0,
        "headroom_present": False,
        "false_negative_risk": True,
        "oracle_vs_vote_gap": 0.0,
        "executable_verifier_is_oracle": True,
        "executable_oracle_upper_bound": {
            "metric": "pass@1_exact_accuracy",
            "vote_pass_at_1": 0.0,
            "executable_oracle_pass_at_1": 0.0,
            "delta": 0.0,
            "ci95": [0.0, 0.0],
            "interpretation": "oracle_upper_bound_not_verifier_value",
            "executable_verifier_is_oracle": True,
            "tautological_equals_oracle_gap": True,
            "status": status,
        },
        "ensemble_rerank_lift_vs_vote": {
            "metric": "pass@1_exact_accuracy",
            "vote_pass_at_1": 0.0,
            "ensemble_pass_at_1": 0.0,
            "delta": 0.0,
            "ci95": [0.0, 0.0],
            "score_components": ["continuous_sudoku_energy", "digit_text_statistics"],
            "uses_exact_validity_check": False,
            "meaningful": False,
            "status": status,
        },
        "per_puzzle": [],
        "oracle_ci95": [0.0, 0.0],
    }


def _metric_beats_vote(metric: Mapping[str, Any]) -> bool:
    ci95 = metric.get("ci95")
    try:
        return (
            isinstance(ci95, Sequence)
            and not isinstance(ci95, (str, bytes))
            and len(ci95) == 2
            and float(metric.get("delta", 0.0)) > 0.0
            and float(ci95[0]) > 0.0
        )
    except (TypeError, ValueError):
        return False


def verifier_value_added(
    *,
    headroom_present: bool,
    ensemble_rerank_lift_vs_vote: Mapping[str, Any],
    rft_vs_ablation_delta: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    """REQ-LEARN-4139: compute value only from transferable verifier signals."""

    if not headroom_present:
        return False, []
    basis: list[str] = []
    if _metric_beats_vote(ensemble_rerank_lift_vs_vote):
        basis.append("ensemble_rerank_lift_vs_vote")
    if _metric_beats_vote(rft_vs_ablation_delta):
        basis.append("rft_vs_ablation_delta")
    return bool(basis), basis


def _artifact_verdict(*, headroom_present: bool, graft_deferred: bool, value_added: bool, basis: Sequence[str]) -> str:
    if not headroom_present:
        return "complete: uninformative_no_headroom_false_negative_risk"
    if value_added:
        if graft_deferred:
            return "success: verifier_value_added_ensemble_with_rft_deferred"
        joined = "_and_".join(basis) if basis else "transferable_verifier"
        return f"success: verifier_value_added_{joined}"
    if graft_deferred:
        return "complete: graft_deferred_baseline_below_0.80"
    return "complete: A~=B null with headroom present"


def compute_reproducibility_checksum(
    *,
    baseline: BaselineContext | None,
    heldout_ids: Sequence[str],
    corpora: Mapping[str, Any] | None = None,
) -> str:
    """REQ-LEARN-4139: hash baseline, held-out ids, and label rows."""

    payload = {
        "schema": "carnot.experiment_4139.decisisive_graft.v1",
        "baseline": None if baseline is None else baseline.to_dict(),
        "heldout_ids": list(heldout_ids),
        "corpora": _jsonable(corpora or {}),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _acceptance_gate(artifact: Mapping[str, Any]) -> bool:
    if str(artifact.get("honest_verdict", "")).startswith("blocked_"):
        return False
    if type(artifact.get("headroom_present")) is not bool:
        return False
    if artifact.get("executable_verifier_is_oracle") is not True:
        return False
    if not isinstance(artifact.get("ensemble_rerank_lift_vs_vote"), Mapping):
        return False
    if artifact.get("graft_deferred") is True:
        return isinstance(artifact.get("estimated_passes_to_converge_for_384"), Mapping)
    return _metric_has_ci(dict(artifact.get("rft_vs_ablation_delta", {})))


def build_result_artifact(
    *,
    baseline: BaselineContext,
    rerank_metrics: Mapping[str, Any],
    rft_vs_ablation_delta: Mapping[str, Any],
    preconditions_checked: Sequence[PreconditionCheck | Mapping[str, Any]],
    duration_s: float,
    candidate_source: str,
    k_candidates: int,
    n_candidate_pools: int,
    corpora_summary: Mapping[str, Any] | None = None,
    random_seed: int = RANDOM_SEED,
    reproducibility_checksum: str | None = None,
) -> dict[str, Any]:
    """REQ-LEARN-4139: build the decisive graft artifact."""

    headroom = bool(rerank_metrics.get("headroom_present"))
    graft_deferred = not baseline_runs_rft_arm(baseline)
    ensemble_metric = dict(rerank_metrics.get("ensemble_rerank_lift_vs_vote", {}))
    value_added, basis = verifier_value_added(
        headroom_present=headroom,
        ensemble_rerank_lift_vs_vote=ensemble_metric,
        rft_vs_ablation_delta=rft_vs_ablation_delta,
    )
    heldout_ids = [
        str(row.get("puzzle_id"))
        for row in rerank_metrics.get("per_puzzle", [])
        if isinstance(row, Mapping)
    ]
    artifact: dict[str, Any] = {
        "experiment": "experiment_4139_decisive_verifier_graft_sudoku",
        "schema": "carnot.experiment_4139_decisive_verifier_graft_sudoku.v1",
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": _artifact_verdict(
            headroom_present=headroom,
            graft_deferred=graft_deferred,
            value_added=value_added,
            basis=basis,
        ),
        "headroom_present": headroom,
        "false_negative_risk": bool(rerank_metrics.get("false_negative_risk", not headroom)),
        "oracle_vs_vote_gap": float(rerank_metrics.get("oracle_vs_vote_gap", 0.0)),
        "executable_verifier_is_oracle": True,
        "executable_oracle_upper_bound": _jsonable(rerank_metrics.get("executable_oracle_upper_bound", {})),
        "ensemble_rerank_lift_vs_vote": _jsonable(ensemble_metric),
        "rft_vs_ablation_delta": _jsonable(rft_vs_ablation_delta),
        "graft_deferred": graft_deferred,
        "verifier_value_added": bool(value_added),
        "verifier_value_added_basis": basis,
        "preconditions_checked": _checks_to_dicts(preconditions_checked),
        "baseline_val_exact_accuracy": baseline.val_exact_accuracy,
        "baseline_matches_published_087": baseline.matches_published_087,
        "baseline_near_faithful_080": baseline.near_faithful_080,
        "stable_checkpoint_path": str(baseline.stable_checkpoint_path),
        "baseline_artifact_path": str(baseline.artifact_path),
        "baseline": baseline.to_dict(),
        "estimated_passes_to_converge_for_384": build_384_estimate(baseline),
        "vote_pass_at_1": rerank_metrics.get("vote_pass_at_1", 0.0),
        "oracle_pass_at_k": rerank_metrics.get("oracle_pass_at_k", 0.0),
        "rerank_metric": "pass@1_exact_accuracy",
        "per_puzzle": _jsonable(rerank_metrics.get("per_puzzle", [])),
        "candidate_source": candidate_source,
        "k_candidates_per_puzzle": int(k_candidates),
        "n_candidate_pools": int(n_candidate_pools),
        "corpus_summary": _jsonable(corpora_summary or {}),
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum
        or compute_reproducibility_checksum(baseline=baseline, heldout_ids=heldout_ids),
        "duration_s": round(float(duration_s), 3),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate_passed": False,
    }
    artifact["acceptance_gate_passed"] = _acceptance_gate(artifact)
    validate_artifact(artifact)
    return artifact


def build_blocked_artifact(
    honest_verdict: str,
    *,
    baseline: BaselineContext | None,
    preconditions_checked: Sequence[PreconditionCheck | Mapping[str, Any]],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """REQ-LEARN-4139: fail closed when the baseline checkpoint cannot be used."""

    rerank = _empty_rerank_metrics("not_run_preconditions_failed")
    artifact = {
        "experiment": "experiment_4139_decisive_verifier_graft_sudoku",
        "schema": "carnot.experiment_4139_decisive_verifier_graft_sudoku.v1",
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": honest_verdict,
        "headroom_present": False,
        "false_negative_risk": True,
        "oracle_vs_vote_gap": 0.0,
        "executable_verifier_is_oracle": True,
        "executable_oracle_upper_bound": rerank["executable_oracle_upper_bound"],
        "ensemble_rerank_lift_vs_vote": rerank["ensemble_rerank_lift_vs_vote"],
        "rft_vs_ablation_delta": _not_run_rft_delta("not_run_preconditions_failed"),
        "graft_deferred": True,
        "verifier_value_added": False,
        "verifier_value_added_basis": [],
        "preconditions_checked": _checks_to_dicts(preconditions_checked),
        "baseline_val_exact_accuracy": None if baseline is None else baseline.val_exact_accuracy,
        "baseline_matches_published_087": False if baseline is None else baseline.matches_published_087,
        "baseline_near_faithful_080": False if baseline is None else baseline.near_faithful_080,
        "stable_checkpoint_path": None if baseline is None else str(baseline.stable_checkpoint_path),
        "baseline_artifact_path": None if baseline is None else str(baseline.artifact_path),
        "baseline": None if baseline is None else baseline.to_dict(),
        "estimated_passes_to_converge_for_384": build_384_estimate(baseline),
        "candidate_source": "none_preconditions_failed",
        "k_candidates_per_puzzle": 0,
        "n_candidate_pools": 0,
        "random_seed": int(random_seed),
        "reproducibility_checksum": compute_reproducibility_checksum(baseline=baseline, heldout_ids=[]),
        "duration_s": round(float(duration_s), 3),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate_passed": False,
    }
    validate_artifact(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return explicit schema errors for the Exp 4139 deliverable."""

    errors: list[str] = []
    for field_name in REQUIRED_ARTIFACT_FIELDS:
        if field_name not in artifact:
            errors.append(f"missing required field {field_name}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed or blocked")

    if type(artifact.get("headroom_present")) is not bool:
        errors.append("headroom_present must be a bare bool")
    gap = _float_or_none(artifact.get("oracle_vs_vote_gap"))
    if gap is None:
        errors.append("oracle_vs_vote_gap must be numeric")
    if artifact.get("executable_verifier_is_oracle") is not True:
        errors.append("executable_verifier_is_oracle must be true for unique-solution Sudoku")
    if type(artifact.get("graft_deferred")) is not bool:
        errors.append("graft_deferred must be a bare bool")
    if type(artifact.get("verifier_value_added")) is not bool:
        errors.append("verifier_value_added must be a bare bool")
    if artifact.get("headroom_present") is False and artifact.get("verifier_value_added") is True:
        errors.append("verifier_value_added requires headroom_present")

    for field_name in ("executable_oracle_upper_bound", "ensemble_rerank_lift_vs_vote", "rft_vs_ablation_delta"):
        metric = artifact.get(field_name)
        if not isinstance(metric, Mapping):
            errors.append(f"{field_name} must be an object")
            continue
        if "delta" not in metric:
            errors.append(f"{field_name}.delta is required")
        if not _metric_has_ci(metric):
            errors.append(f"{field_name}.ci95 must have two numeric bounds")

    executable = artifact.get("executable_oracle_upper_bound")
    if isinstance(executable, Mapping) and gap is not None:
        executable_delta = _float_or_none(executable.get("delta"))
        if executable_delta is not None and abs(executable_delta - gap) > 1e-9:
            errors.append("executable_oracle_upper_bound.delta must equal oracle_vs_vote_gap")
    ensemble = artifact.get("ensemble_rerank_lift_vs_vote")
    if isinstance(ensemble, Mapping) and ensemble.get("uses_exact_validity_check") is True:
        errors.append("ensemble_rerank_lift_vs_vote must not use exact validity")

    basis = artifact.get("verifier_value_added_basis", [])
    if isinstance(basis, Sequence) and not isinstance(basis, (str, bytes)):
        if any("executable" in str(item) for item in basis):
            errors.append("verifier_value_added_basis must not include executable oracle")
    else:
        errors.append("verifier_value_added_basis must be a list")

    preconditions = artifact.get("preconditions_checked")
    if not isinstance(preconditions, list):
        errors.append("preconditions_checked must be a list")
    elif any(
        not isinstance(item, Mapping) or "resource" not in item or "available" not in item
        for item in preconditions
    ):
        errors.append("preconditions_checked entries must include resource and available")

    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles must be an object")
    else:
        for field_name, principle in FIELD_PRINCIPLES.items():
            if principles.get(field_name) != principle:
                errors.append(f"field_principles.{field_name} mismatch")

    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
        errors.append("duration_s must be numeric")
    if "acceptance_gate_passed" in artifact and type(artifact.get("acceptance_gate_passed")) is not bool:
        errors.append("acceptance_gate_passed must be a bare bool")
    checksum = artifact.get("reproducibility_checksum")
    if checksum is not None and not (
        isinstance(checksum, str) and checksum.startswith("sha256:") and len(checksum) == 71
    ):
        errors.append("reproducibility_checksum must be sha256-prefixed")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def write_artifact(path: str | Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Write the stable Exp 4139 JSON artifact."""

    validate_artifact(artifact)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _jsonable(artifact)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return json.loads(output_path.read_text(encoding="utf-8"))


def sample_checkpoint_candidate_pools(  # pragma: no cover - live GPU/checkpoint path.
    *,
    baseline: BaselineContext,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    split: str = DEFAULT_HELDOUT_SPLIT,
    max_puzzles: int = DEFAULT_MAX_PUZZLES,
    k_candidates: int = DEFAULT_K_CANDIDATES,
    random_seed: int = RANDOM_SEED,
) -> list[CandidatePool]:
    """REQ-LEARN-4139: sample K candidate grids from the stable TRM checkpoint."""

    return exp4109.sample_checkpoint_candidate_pools(
        checkpoint_path=baseline.stable_checkpoint_path,
        data_dir=data_dir,
        split=split,
        max_puzzles=max_puzzles,
        k_candidates=k_candidates,
        random_seed=random_seed,
    )


def _summarize_corpora(corpora: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(corpora.get("rows", []))
    return {
        "arm_a": corpora.get("arm_a"),
        "arm_b": corpora.get("arm_b"),
        "n_matched": int(corpora.get("n_matched", 0)),
        "skipped_no_verifier_valid": len(corpora.get("skipped_no_verifier_valid", [])),
        "a_exact_count": sum(bool(row.get("a_exact")) for row in rows if isinstance(row, Mapping)),
        "b_exact_count": sum(bool(row.get("b_exact")) for row in rows if isinstance(row, Mapping)),
    }


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    output_path: str | Path = DEFAULT_OUTPUT,
    exp4138_artifact_path: str | Path | None = None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    heldout_split: str = DEFAULT_HELDOUT_SPLIT,
    max_puzzles: int = DEFAULT_MAX_PUZZLES,
    k_candidates: int = DEFAULT_K_CANDIDATES,
    bootstrap_resamples: int = 2000,
    random_seed: int = RANDOM_SEED,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4107._default_cuda_checker,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
    candidate_pool_provider: Callable[[BaselineContext], Sequence[CandidatePool]] | None = None,
    rft_runner: Callable[[BaselineContext, dict[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run Exp 4139 and write the decisive verifier-graft artifact."""

    started = time.time()
    root = Path(repo_root)
    baseline_path = Path(exp4138_artifact_path) if exp4138_artifact_path is not None else root / "results" / exp4138.RESULT_FILENAME
    try:
        baseline = load_baseline_context(baseline_path)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        artifact = build_blocked_artifact(
            "blocked_exp4138_baseline_missing",
            baseline=None,
            preconditions_checked=[PreconditionCheck("exp4138_artifact", False, str(baseline_path))],
            duration_s=time.time() - started,
            random_seed=random_seed,
        )
        return write_artifact(output_path, artifact)

    checks = check_preconditions(baseline, cuda_checker=cuda_checker, checkpoint_loader=checkpoint_loader)
    checkpoint_check = next((check for check in checks if check.resource == "baseline_checkpoint"), None)
    if checkpoint_check is None or not checkpoint_check.available:
        artifact = build_blocked_artifact(
            "blocked_baseline_checkpoint_missing",
            baseline=baseline,
            preconditions_checked=checks,
            duration_s=time.time() - started,
            random_seed=random_seed,
        )
        return write_artifact(output_path, artifact)
    if not _all_preconditions_available(checks):
        artifact = build_blocked_artifact(
            "blocked_exp4139_preconditions_missing",
            baseline=baseline,
            preconditions_checked=checks,
            duration_s=time.time() - started,
            random_seed=random_seed,
        )
        return write_artifact(output_path, artifact)

    try:
        if candidate_pool_provider is not None:
            pools = list(candidate_pool_provider(baseline))
            candidate_source = "provided_candidate_pool"
        else:  # pragma: no cover - live checkpoint sampling path.
            pools = sample_checkpoint_candidate_pools(
                baseline=baseline,
                data_dir=data_dir,
                split=heldout_split,
                max_puzzles=max_puzzles,
                k_candidates=k_candidates,
                random_seed=random_seed,
            )
            candidate_source = "trm_checkpoint_final_logits_k_sampling"
    except Exception as exc:  # pragma: no cover - defensive live failure path.
        artifact = build_blocked_artifact(
            "blocked_candidate_sampling_failed",
            baseline=baseline,
            preconditions_checked=[
                *checks,
                PreconditionCheck("candidate_sampling", False, f"{type(exc).__name__}: {exc}"),
            ],
            duration_s=time.time() - started,
            random_seed=random_seed,
        )
        return write_artifact(output_path, artifact)

    rerank_metrics = evaluate_reranks(pools, random_seed=random_seed, bootstrap_resamples=bootstrap_resamples)
    corpora: dict[str, Any] = {}
    if baseline_runs_rft_arm(baseline):
        corpora = exp4109.build_matched_corpora(pools)
        if rft_runner is not None:
            rft_delta = dict(rft_runner(baseline, corpora))
        else:
            rft_delta = exp4109.evaluate_label_arms(
                corpora,
                random_seed=random_seed + 3,
                bootstrap_resamples=bootstrap_resamples,
            )
    else:
        rft_delta = deferred_rft_delta(baseline)
    heldout_ids = [pool.puzzle_id for pool in pools]
    checksum = compute_reproducibility_checksum(
        baseline=baseline,
        heldout_ids=heldout_ids,
        corpora=corpora,
    )
    artifact = build_result_artifact(
        baseline=baseline,
        rerank_metrics=rerank_metrics,
        rft_vs_ablation_delta=rft_delta,
        preconditions_checked=checks,
        duration_s=time.time() - started,
        candidate_source=candidate_source,
        k_candidates=k_candidates,
        n_candidate_pools=len(pools),
        corpora_summary=_summarize_corpora(corpora) if corpora else {},
        random_seed=random_seed,
        reproducibility_checksum=checksum,
    )
    return write_artifact(output_path, artifact)


def main() -> None:  # pragma: no cover - thin CLI wrapper.
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
