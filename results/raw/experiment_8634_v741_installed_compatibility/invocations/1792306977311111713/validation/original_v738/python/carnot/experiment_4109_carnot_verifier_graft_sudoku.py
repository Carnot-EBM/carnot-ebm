"""Exp 4109 Sudoku verifier graft over nano-trm.

Spec refs: REQ-LEARN-4109, SCENARIO-LEARN-4109-RERANK,
SCENARIO-LEARN-4109-RFT.
"""

from __future__ import annotations

from carnot.serialization_safety import safe_torch_load

import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4109_carnot_verifier_graft_sudoku.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_EXP4108_ARTIFACT = (
    REPO_ROOT / "results" / "experiment_4108_nanotrm_sudoku_extreme_baseline.json"
)
DEFAULT_EXP4107_ARTIFACT = REPO_ROOT / "results" / "experiment_4107_nanotrm_mechanism_smoke.json"
DEFAULT_DATA_DIR = REPO_ROOT / "nano-trm" / "data" / "sudoku_extreme_1k_aug_1k"
RANDOM_SEED = 4109
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "rerank_lift_vs_vote",
    "rft_vs_ablation_delta",
    "verifier_value_added",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest A~=B (verifier adds nothing even where it "
        "discriminates, e.g. base near ceiling) is a COMPLETE, decision-grade verdict."
    ),
    "rerank_lift_vs_vote": (
        "pass@1 lift from verifier-reranking vs TRM-vote with CI; confirms the "
        "executable verifier discriminates on Sudoku (the contrast to .379's "
        "ARC-grid anti-discrimination)."
    ),
    "rft_vs_ablation_delta": (
        "The de-confounded A-vs-B held-out delta with CI: isolates the verifier "
        "LABEL's training contribution from generic self-training."
    ),
    "verifier_value_added": (
        "Bare bool: did the verifier graft beat the vote ablation (A>B, "
        "CI-excl-0)? The milestone's headline answer on whether "
        "verifier-as-reward is real on an executable domain."
    ),
    "preconditions_checked": (
        "Records the checkpoint + CUDA verified; pre-empts the silent-missing-resource "
        "fabrication mode."
    ),
    "random_seed": "Determinism precondition for reproducing the graft.",
    "reproducibility_checksum": "Hash of the corpus + split; catches silent drift.",
}


@dataclass(frozen=True)
class CandidateSample:
    """One sampled Sudoku solution candidate from a TRM candidate pool."""

    sample_id: str
    tokens: Sequence[int]
    trm_score: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def token_tuple(self) -> tuple[int, ...]:
        return tuple(int(token) for token in self.tokens)


@dataclass(frozen=True)
class CandidatePool:
    """All sampled candidates for one held-out Sudoku puzzle."""

    puzzle_id: str
    puzzle_tokens: Sequence[int]
    label_tokens: Sequence[int]
    candidates: Sequence[CandidateSample]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SudokuVerifierScore:
    """Executable Sudoku verifier result for one candidate grid."""

    exact_valid: bool
    satisfied_constraints: int
    total_constraints: int
    normalized_score: float
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreconditionCheck:
    """One Exp 4109 runtime resource check."""

    resource: str
    available: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CheckpointChoice:
    """The checkpoint Exp 4109 will graft the verifier onto."""

    checkpoint_path: Path | None
    source_experiment: str
    limitation: str

    def to_dict(self) -> dict[str, object | None]:
        return {
            "checkpoint_path": None if self.checkpoint_path is None else str(self.checkpoint_path),
            "source_experiment": self.source_experiment,
            "limitation": self.limitation,
        }


def encode_grid(grid: Sequence[Sequence[int]]) -> list[int]:
    """Encode a Sudoku grid using nano-trm tokens: empty=2, digit d=d+2."""

    return [2 if int(value) == 0 else int(value) + 2 for row in grid for value in row]


def decode_tokens(tokens: Sequence[int], *, grid_size: int = 9) -> list[list[int]]:
    """Decode nano-trm tokens to Sudoku digits, where token 2 becomes empty 0."""

    flat = [int(token) - 2 for token in tokens]
    expected = grid_size * grid_size
    if len(flat) != expected:
        raise ValueError(f"expected {expected} tokens for {grid_size}x{grid_size} Sudoku")
    return [flat[row * grid_size : (row + 1) * grid_size] for row in range(grid_size)]


def _box_dims(grid_size: int) -> tuple[int, int]:
    if grid_size == 4:
        return 2, 2
    if grid_size == 6:
        return 2, 3
    if grid_size == 9:
        return 3, 3
    side = int(grid_size**0.5)
    if side * side == grid_size:
        return side, side
    raise ValueError(f"unsupported Sudoku grid size: {grid_size}")


def _unit_ok(values: Sequence[int], allowed: set[int]) -> bool:
    return set(int(value) for value in values) == allowed


def score_sudoku_candidate(
    puzzle_tokens: Sequence[int],
    candidate_tokens: Sequence[int],
    *,
    grid_size: int = 9,
) -> SudokuVerifierScore:
    """REQ-LEARN-4109: score exact Sudoku constraint satisfaction."""

    puzzle = decode_tokens(puzzle_tokens, grid_size=grid_size)
    candidate = decode_tokens(candidate_tokens, grid_size=grid_size)
    allowed = set(range(1, grid_size + 1))
    box_rows, box_cols = _box_dims(grid_size)
    satisfied = 0
    total = grid_size * grid_size
    failures: set[str] = set()

    for row in range(grid_size):
        for col in range(grid_size):
            value = candidate[row][col]
            if value in allowed:
                satisfied += 1
            else:
                failures.add("range")

    for row in range(grid_size):
        for col in range(grid_size):
            clue = puzzle[row][col]
            if clue > 0:
                total += 1
                if candidate[row][col] == clue:
                    satisfied += 1
                else:
                    failures.add("clue")

    for row in range(grid_size):
        total += 1
        if _unit_ok(candidate[row], allowed):
            satisfied += 1
        else:
            failures.add("row")

    for col in range(grid_size):
        total += 1
        if _unit_ok([candidate[row][col] for row in range(grid_size)], allowed):
            satisfied += 1
        else:
            failures.add("column")

    for box_row in range(0, grid_size, box_rows):
        for box_col in range(0, grid_size, box_cols):
            total += 1
            values = [
                candidate[row][col]
                for row in range(box_row, box_row + box_rows)
                for col in range(box_col, box_col + box_cols)
            ]
            if _unit_ok(values, allowed):
                satisfied += 1
            else:
                failures.add("box")

    return SudokuVerifierScore(
        exact_valid=satisfied == total,
        satisfied_constraints=int(satisfied),
        total_constraints=int(total),
        normalized_score=round(float(satisfied / total), 6),
        failure_reasons=tuple(sorted(failures)),
    )


def _is_exact(candidate_tokens: Sequence[int], label_tokens: Sequence[int]) -> bool:
    return tuple(int(token) for token in candidate_tokens) == tuple(
        int(token) for token in label_tokens
    )


def _vote_groups(candidates: Sequence[CandidateSample]) -> dict[tuple[int, ...], dict[str, Any]]:
    grouped: dict[tuple[int, ...], dict[str, Any]] = {}
    for index, candidate in enumerate(candidates):
        key = candidate.token_tuple()
        if key not in grouped:
            grouped[key] = {
                "count": 0,
                "score_sum": 0.0,
                "first_index": index,
                "sample": candidate,
            }
        grouped[key]["count"] += 1
        grouped[key]["score_sum"] += float(candidate.trm_score)
    return grouped


def select_vote_candidate(candidates: Sequence[CandidateSample]) -> CandidateSample:
    """Select the TRM-vote candidate by majority, then mean TRM score."""

    if not candidates:
        raise ValueError("candidate pool is empty")
    grouped = _vote_groups(candidates)
    winner = max(
        grouped.values(),
        key=lambda item: (
            int(item["count"]),
            float(item["score_sum"]) / max(int(item["count"]), 1),
            -int(item["first_index"]),
        ),
    )
    return winner["sample"]


def select_verifier_candidate(pool: CandidatePool) -> tuple[CandidateSample, SudokuVerifierScore]:
    """Select the best candidate by executable Sudoku verifier score."""

    if not pool.candidates:
        raise ValueError("candidate pool is empty")
    grouped = _vote_groups(pool.candidates)
    vote_count_by_tokens = {tokens: int(row["count"]) for tokens, row in grouped.items()}
    best: tuple[tuple[float, int, float, int], CandidateSample, SudokuVerifierScore] | None = None
    for index, candidate in enumerate(pool.candidates):
        score = score_sudoku_candidate(pool.puzzle_tokens, candidate.tokens)
        key = (
            float(score.normalized_score),
            int(score.exact_valid),
            float(vote_count_by_tokens[candidate.token_tuple()]),
            float(candidate.trm_score),
            -index,
        )
        if best is None or key > best[0]:
            best = (key, candidate, score)
    assert best is not None
    return best[1], best[2]


def _bootstrap_ci(differences: Sequence[float], *, random_seed: int, resamples: int) -> list[float]:
    if not differences:
        return [0.0, 0.0]
    rng = random.Random(random_seed)
    n = len(differences)
    draws = []
    for _ in range(max(int(resamples), 1)):
        draws.append(sum(differences[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return [round(float(lo), 6), round(float(hi), 6)]


def evaluate_rerank(
    pools: Sequence[CandidatePool],
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = 2000,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4109-RERANK: compare verifier rerank against TRM vote."""

    rows: list[dict[str, Any]] = []
    differences: list[float] = []
    for pool in pools:
        vote = select_vote_candidate(pool.candidates)
        verifier, verifier_score = select_verifier_candidate(pool)
        vote_score = score_sudoku_candidate(pool.puzzle_tokens, vote.tokens)
        vote_correct = _is_exact(vote.tokens, pool.label_tokens)
        verifier_correct = _is_exact(verifier.tokens, pool.label_tokens)
        oracle_correct = any(
            _is_exact(candidate.tokens, pool.label_tokens) for candidate in pool.candidates
        )
        differences.append(float(verifier_correct) - float(vote_correct))
        rows.append(
            {
                "puzzle_id": pool.puzzle_id,
                "vote_sample_id": vote.sample_id,
                "verifier_sample_id": verifier.sample_id,
                "vote_correct": vote_correct,
                "verifier_correct": verifier_correct,
                "oracle_correct": oracle_correct,
                "vote_verifier_score": vote_score.to_dict(),
                "selected_verifier_score": verifier_score.to_dict(),
            }
        )

    n = len(rows)
    vote_acc = sum(row["vote_correct"] for row in rows) / n if n else 0.0
    verifier_acc = sum(row["verifier_correct"] for row in rows) / n if n else 0.0
    oracle_acc = sum(row["oracle_correct"] for row in rows) / n if n else 0.0
    delta = verifier_acc - vote_acc
    return {
        "metric": "pass@1_exact_accuracy",
        "n_puzzles": n,
        "vote_pass_at_1": round(float(vote_acc), 6),
        "verifier_pass_at_1": round(float(verifier_acc), 6),
        "oracle_ceiling_pass_at_1": round(float(oracle_acc), 6),
        "delta": round(float(delta), 6),
        "ci95": _bootstrap_ci(differences, random_seed=random_seed, resamples=bootstrap_resamples),
        "per_puzzle": rows,
    }


def build_matched_corpora(pools: Sequence[CandidatePool]) -> dict[str, Any]:
    """SCENARIO-LEARN-4109-RFT: build N-matched verifier and vote label rows."""

    rows: list[dict[str, Any]] = []
    skipped_no_verifier_valid: list[str] = []
    for pool in pools:
        vote = select_vote_candidate(pool.candidates)
        verifier, verifier_score = select_verifier_candidate(pool)
        if not verifier_score.exact_valid:
            skipped_no_verifier_valid.append(pool.puzzle_id)
            continue
        vote_score = score_sudoku_candidate(pool.puzzle_tokens, vote.tokens)
        rows.append(
            {
                "puzzle_id": pool.puzzle_id,
                "a_sample_id": verifier.sample_id,
                "b_sample_id": vote.sample_id,
                "a_exact": _is_exact(verifier.tokens, pool.label_tokens),
                "b_exact": _is_exact(vote.tokens, pool.label_tokens),
                "a_verifier_score": verifier_score.to_dict(),
                "b_verifier_score": vote_score.to_dict(),
            }
        )
    return {
        "arm_a": "verifier_certified",
        "arm_b": "vote_certified",
        "n_matched": len(rows),
        "rows": rows,
        "skipped_no_verifier_valid": skipped_no_verifier_valid,
    }


def evaluate_label_arms(
    corpora: Mapping[str, Any],
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = 2000,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4109-RFT: report A-vs-B held-out label delta."""

    rows = list(corpora.get("rows", []))
    if not rows:
        return {
            "metric": "heldout_exact_accuracy",
            "n_matched": 0,
            "a_exact_accuracy": 0.0,
            "b_exact_accuracy": 0.0,
            "delta": 0.0,
            "ci95": [0.0, 0.0],
            "status": "no_matched_verifier_valid_labels",
        }
    differences = [float(row["a_exact"]) - float(row["b_exact"]) for row in rows]
    a_acc = sum(bool(row["a_exact"]) for row in rows) / len(rows)
    b_acc = sum(bool(row["b_exact"]) for row in rows) / len(rows)
    ci95 = _bootstrap_ci(differences, random_seed=random_seed, resamples=bootstrap_resamples)
    delta = a_acc - b_acc
    if delta > 0.0 and ci95[0] > 0.0:
        status = "ci95_excludes_zero"
    elif delta < 0.0 and ci95[1] < 0.0:
        status = "negative_ci95_excludes_zero"
    else:
        status = "honest_null_ci95_includes_zero"
    return {
        "metric": "heldout_exact_accuracy",
        "n_matched": len(rows),
        "a_exact_accuracy": round(float(a_acc), 6),
        "b_exact_accuracy": round(float(b_acc), 6),
        "delta": round(float(delta), 6),
        "ci95": ci95,
        "status": status,
    }


def verifier_value_added(rft_delta: Mapping[str, Any]) -> bool:
    """Return the bare Exp 4109 headline bool."""

    ci95 = rft_delta.get("ci95")
    return (
        isinstance(ci95, Sequence)
        and len(ci95) == 2
        and float(rft_delta.get("delta", 0.0)) > 0.0
        and float(ci95[0]) > 0.0
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def compute_reproducibility_checksum(
    corpora: Mapping[str, Any], *, heldout_ids: Sequence[str]
) -> str:
    """REQ-LEARN-4109: hash corpus rows and held-out split identifiers."""

    payload = {
        "schema": "carnot.experiment_4109.corpus_split.v1",
        "corpora": _jsonable(corpora),
        "heldout_ids": list(heldout_ids),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _load_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def resolve_checkpoint_choice(
    exp4108_artifact_path: str | Path = DEFAULT_EXP4108_ARTIFACT,
    exp4107_artifact_path: str | Path = DEFAULT_EXP4107_ARTIFACT,
) -> CheckpointChoice:
    """REQ-LEARN-4109: prefer Exp 4108 checkpoint and fall back to Exp 4107."""

    exp4108_path = Path(exp4108_artifact_path)
    exp4108 = _load_json_object(exp4108_path)
    if exp4108 is not None:
        checkpoint_value = exp4108.get("checkpoint_path")
        if isinstance(checkpoint_value, str) and Path(checkpoint_value).is_file():
            limitation = (
                "exp4108_reproducing_checkpoint"
                if exp4108.get("matches_published_087") is True
                else "exp4108_partial_baseline_matches_published_087_false"
            )
            return CheckpointChoice(Path(checkpoint_value), "exp4108", limitation)

    exp4107 = _load_json_object(Path(exp4107_artifact_path))
    if exp4107 is not None:
        checkpoint_value = exp4107.get("checkpoint_path")
        if isinstance(checkpoint_value, str) and Path(checkpoint_value).is_file():
            return CheckpointChoice(
                Path(checkpoint_value),
                "exp4107",
                "exp4108_checkpoint_missing_fell_back_to_exp4107_smoke",
            )
    return CheckpointChoice(None, "none", "no_exp4108_or_exp4107_checkpoint_available")


def _default_cuda_checker() -> tuple[
    bool, str
]:  # pragma: no cover - imports torch and probes host GPU.
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # pragma: no cover - depends on runtime.
        return False, f"{type(exc).__name__}: {exc}"
    available = bool(torch.cuda.is_available())
    detail = f"torch.cuda.is_available()={available}"
    if available:
        detail += f"; device={torch.cuda.get_device_name(0)}"
    return available, detail


def check_preconditions(
    *,
    exp4108_artifact_path: str | Path = DEFAULT_EXP4108_ARTIFACT,
    exp4107_artifact_path: str | Path = DEFAULT_EXP4107_ARTIFACT,
    cuda_checker: Callable[[], tuple[bool, str]] = _default_cuda_checker,
) -> tuple[list[PreconditionCheck], CheckpointChoice]:
    """REQ-LEARN-4109: record checkpoint and CUDA preconditions."""

    choice = resolve_checkpoint_choice(exp4108_artifact_path, exp4107_artifact_path)
    checks = [
        PreconditionCheck(
            "checkpoint_path",
            choice.checkpoint_path is not None,
            choice.limitation if choice.checkpoint_path is None else str(choice.checkpoint_path),
        )
    ]
    try:
        cuda_available, cuda_detail = cuda_checker()
    except Exception as exc:
        cuda_available, cuda_detail = False, f"{type(exc).__name__}: {exc}"
    checks.append(PreconditionCheck("cuda_available", bool(cuda_available), str(cuda_detail)))
    return checks, choice


def _candidate_score_from_logprobs(
    log_probs: Any, selected: Any
) -> float:  # pragma: no cover - torch live sampler helper.
    gathered = log_probs.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
    return float(gathered.mean().detach().cpu().item())


def sample_checkpoint_candidate_pools(  # pragma: no cover - live GPU/checkpoint path.
    *,
    checkpoint_path: str | Path,
    repo_root: str | Path = REPO_ROOT,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    split: str = "val",
    max_puzzles: int = 128,
    k_candidates: int = 8,
    batch_size: int = 64,
    random_seed: int = RANDOM_SEED,
    temperature: float = 1.0,
) -> list[CandidatePool]:
    """Sample K candidate grids from the Exp 4108 TRM checkpoint."""

    import numpy as np  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    root = Path(repo_root)
    nano_root = root / "nano-trm"
    for path in (nano_root, nano_root / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from src.nn.models.trm import TRMModule  # pylint: disable=import-outside-toplevel

    print(f"[exp4109] loading checkpoint={checkpoint_path}", flush=True)
    checkpoint = safe_torch_load(
        Path(checkpoint_path), map_location="cpu", allow_unsafe_pickle=True
    )
    model = TRMModule(**checkpoint["hyper_parameters"])
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    split_dir = Path(data_dir) / split
    inputs = np.load(split_dir / "all__inputs.npy", mmap_mode="r")
    labels = np.load(split_dir / "all__labels.npy", mmap_mode="r")
    identifiers = np.load(split_dir / "all__puzzle_identifiers.npy")
    n = min(int(max_puzzles), int(len(inputs)))
    torch.manual_seed(int(random_seed))
    pools: list[CandidatePool] = []

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        print(
            f"[exp4109] sampling TRM candidates split={split} rows={start}:{stop} k={k_candidates}",
            flush=True,
        )
        batch = {
            "input": torch.as_tensor(inputs[start:stop], dtype=torch.long, device=device),
            "output": torch.as_tensor(labels[start:stop], dtype=torch.long, device=device),
            "puzzle_identifiers": torch.as_tensor(
                identifiers[start:stop], dtype=torch.long, device=device
            ),
        }
        with torch.no_grad():
            carry = model.initial_carry(batch)
            outputs = None
            for _step in range(int(model.hparams.N_supervision_val)):
                carry, outputs = model.forward(carry, batch)
                if bool(carry.halted.all()):
                    break
            assert outputs is not None
            logits = outputs["logits"].float() / max(float(temperature), 1e-6)
            log_probs = logits.log_softmax(dim=-1)
            probs = logits.softmax(dim=-1)
            argmax = logits.argmax(dim=-1)
            sampled = [argmax]
            flat_probs = probs.reshape(-1, probs.shape[-1])
            for _ in range(max(int(k_candidates) - 1, 0)):
                sampled.append(torch.multinomial(flat_probs, 1).reshape(stop - start, -1))

        for offset in range(stop - start):
            candidates: list[CandidateSample] = []
            for sample_index, selected in enumerate(sampled):
                tokens = selected[offset].detach().cpu().numpy().astype(int).tolist()
                score = _candidate_score_from_logprobs(log_probs[offset], selected[offset])
                source = "argmax" if sample_index == 0 else "multinomial"
                candidates.append(
                    CandidateSample(
                        sample_id=f"{split}:{start + offset}:k{sample_index}",
                        tokens=tokens,
                        trm_score=score,
                        metadata={"source": source, "temperature": float(temperature)},
                    )
                )
            pools.append(
                CandidatePool(
                    puzzle_id=f"{split}:{start + offset}",
                    puzzle_tokens=inputs[start + offset].astype(int).tolist(),
                    label_tokens=labels[start + offset].astype(int).tolist(),
                    candidates=candidates,
                    metadata={"puzzle_identifier": int(identifiers[start + offset])},
                )
            )
    return pools


def _summarize_corpora(corpora: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(corpora.get("rows", []))
    return {
        "arm_a": corpora.get("arm_a"),
        "arm_b": corpora.get("arm_b"),
        "n_matched": int(corpora.get("n_matched", 0)),
        "skipped_no_verifier_valid": len(corpora.get("skipped_no_verifier_valid", [])),
        "a_exact_count": sum(bool(row.get("a_exact")) for row in rows),
        "b_exact_count": sum(bool(row.get("b_exact")) for row in rows),
    }


def build_result_artifact(
    *,
    rerank_metrics: Mapping[str, Any],
    rft_delta: Mapping[str, Any],
    corpus_summary: Mapping[str, Any],
    preconditions_checked: Sequence[Mapping[str, Any]],
    checkpoint_choice: CheckpointChoice,
    random_seed: int,
    reproducibility_checksum: str,
    duration_s: float,
    native_training_launched: bool,
    candidate_source: str = "trm_checkpoint_final_logits_k_sampling",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate the Exp 4109 result artifact."""

    value_added = verifier_value_added(rft_delta)
    if value_added:
        verdict = "complete: verifier_value_added_A_gt_B_ci95_excludes_zero"
    elif rft_delta.get("status") == "no_matched_verifier_valid_labels":
        verdict = "complete: honest_null_no_verifier_valid_training_labels"
    else:
        verdict = "complete: honest_null_A_approx_B_verifier_adds_nothing"

    cold_accuracy = (
        float(rerank_metrics.get("vote_pass_at_1", 0.0))
        if native_training_launched
        else float(rft_delta.get("b_exact_accuracy", rerank_metrics.get("vote_pass_at_1", 0.0)))
    )
    a_vs_cold_lift = {
        "metric": "heldout_exact_accuracy",
        "a_exact_accuracy": float(rft_delta.get("a_exact_accuracy", 0.0)),
        "cold_exact_accuracy": cold_accuracy,
        "delta": round(float(rft_delta.get("a_exact_accuracy", 0.0)) - cold_accuracy, 6),
        "comparison_basis": (
            "native_trm_heldout_accuracy"
            if native_training_launched
            else "matched_vote_proxy_because_native_training_not_launched"
        ),
    }
    artifact: dict[str, Any] = {
        "experiment": "experiment_4109_carnot_verifier_graft_sudoku",
        "schema": "carnot.experiment_4109_carnot_verifier_graft_sudoku.v1",
        "honest_verdict": verdict,
        "rerank_lift_vs_vote": _jsonable(rerank_metrics),
        "rft_vs_ablation_delta": _jsonable(rft_delta),
        "a_vs_cold_lift": a_vs_cold_lift,
        "verifier_value_added": bool(value_added),
        "preconditions_checked": [_jsonable(check) for check in preconditions_checked],
        "checkpoint_choice": checkpoint_choice.to_dict(),
        "checkpoint_path": None
        if checkpoint_choice.checkpoint_path is None
        else str(checkpoint_choice.checkpoint_path),
        "checkpoint_source_experiment": checkpoint_choice.source_experiment,
        "baseline_limitation": checkpoint_choice.limitation,
        "candidate_source": candidate_source,
        "native_training_launched": bool(native_training_launched),
        "native_training_limitation": (
            None
            if native_training_launched
            else "native nano-trm full fine-tuning not launched in this bounded run; "
            "A-vs-B is the verifier-label versus vote-label deconfound."
        ),
        "corpus_summary": _jsonable(corpus_summary),
        "acceptance_gate_passed": _has_reported_ci(rerank_metrics) and _has_reported_ci(rft_delta),
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum,
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": round(float(duration_s), 3),
        "spec_refs": ["REQ-LEARN-4109", "SCENARIO-LEARN-4109-RERANK", "SCENARIO-LEARN-4109-RFT"],
    }
    if extra:
        artifact.update(_jsonable(extra))
    validate_artifact(artifact)
    return artifact


def _has_reported_ci(metric: Mapping[str, Any]) -> bool:
    ci95 = metric.get("ci95")
    return isinstance(ci95, Sequence) and len(ci95) == 2


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return explicit schema errors for the Exp 4109 deliverable."""

    errors: list[str] = []
    for field_name in REQUIRED_ARTIFACT_FIELDS:
        if field_name not in artifact:
            errors.append(f"missing required field {field_name}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed")

    for field_name in ("rerank_lift_vs_vote", "rft_vs_ablation_delta"):
        metric = artifact.get(field_name)
        if not isinstance(metric, Mapping):
            errors.append(f"{field_name} must be an object")
            continue
        if "delta" not in metric:
            errors.append(f"{field_name}.delta is required")
        ci95 = metric.get("ci95")
        if not (isinstance(ci95, Sequence) and len(ci95) == 2):
            errors.append(f"{field_name}.ci95 must have two bounds")

    if type(artifact.get("verifier_value_added")) is not bool:
        errors.append("verifier_value_added must be a bare bool")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be a bare int")
    checksum = artifact.get("reproducibility_checksum")
    if not (isinstance(checksum, str) and checksum.startswith("sha256:") and len(checksum) == 71):
        errors.append("reproducibility_checksum must be sha256-prefixed")
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
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def write_artifact(path: str | Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Write the stable Exp 4109 JSON artifact."""

    validate_artifact(artifact)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _jsonable(artifact)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return json.loads(output_path.read_text(encoding="utf-8"))


def run_experiment(
    *,
    output_path: str | Path = DEFAULT_OUTPUT,
    exp4108_artifact_path: str | Path = DEFAULT_EXP4108_ARTIFACT,
    exp4107_artifact_path: str | Path = DEFAULT_EXP4107_ARTIFACT,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    max_puzzles: int = 128,
    k_candidates: int = 8,
    bootstrap_resamples: int = 2000,
    random_seed: int = RANDOM_SEED,
    cuda_checker: Callable[[], tuple[bool, str]] = _default_cuda_checker,
    candidate_pool_provider: Callable[[CheckpointChoice], Sequence[CandidatePool]] | None = None,
    native_training_launched: bool = False,
) -> dict[str, Any]:
    """Run the bounded Exp 4109 verifier-graft measurement and write JSON."""

    started = time.time()
    checks, choice = check_preconditions(
        exp4108_artifact_path=exp4108_artifact_path,
        exp4107_artifact_path=exp4107_artifact_path,
        cuda_checker=cuda_checker,
    )
    if choice.checkpoint_path is None or any(not check.available for check in checks):
        empty_rerank = {
            "metric": "pass@1_exact_accuracy",
            "n_puzzles": 0,
            "vote_pass_at_1": 0.0,
            "verifier_pass_at_1": 0.0,
            "oracle_ceiling_pass_at_1": 0.0,
            "delta": 0.0,
            "ci95": [0.0, 0.0],
            "per_puzzle": [],
        }
        empty_delta = {
            "metric": "heldout_exact_accuracy",
            "n_matched": 0,
            "a_exact_accuracy": 0.0,
            "b_exact_accuracy": 0.0,
            "delta": 0.0,
            "ci95": [0.0, 0.0],
            "status": "blocked_preconditions_missing",
        }
        checksum = compute_reproducibility_checksum({"rows": []}, heldout_ids=[])
        artifact = build_result_artifact(
            rerank_metrics=empty_rerank,
            rft_delta=empty_delta,
            corpus_summary={"n_matched": 0},
            preconditions_checked=[check.to_dict() for check in checks],
            checkpoint_choice=choice,
            random_seed=random_seed,
            reproducibility_checksum=checksum,
            duration_s=time.time() - started,
            native_training_launched=native_training_launched,
            candidate_source="none_preconditions_missing",
        )
        artifact["honest_verdict"] = "blocked_exp4109_preconditions_missing"
        write_artifact(output_path, artifact)
        return artifact

    if candidate_pool_provider is not None:
        pools = list(candidate_pool_provider(choice))
        candidate_source = "provided_candidate_pool"
    else:  # pragma: no cover - live checkpoint sampling path.
        pools = sample_checkpoint_candidate_pools(
            checkpoint_path=choice.checkpoint_path,
            data_dir=data_dir,
            max_puzzles=max_puzzles,
            k_candidates=k_candidates,
            random_seed=random_seed,
        )
        candidate_source = "trm_checkpoint_final_logits_k_sampling"

    rerank = evaluate_rerank(
        pools, random_seed=random_seed, bootstrap_resamples=bootstrap_resamples
    )
    corpora = build_matched_corpora(pools)
    rft_delta = evaluate_label_arms(
        corpora, random_seed=random_seed + 1, bootstrap_resamples=bootstrap_resamples
    )
    heldout_ids = [pool.puzzle_id for pool in pools]
    checksum = compute_reproducibility_checksum(corpora, heldout_ids=heldout_ids)
    artifact = build_result_artifact(
        rerank_metrics=rerank,
        rft_delta=rft_delta,
        corpus_summary=_summarize_corpora(corpora),
        preconditions_checked=[check.to_dict() for check in checks],
        checkpoint_choice=choice,
        random_seed=random_seed,
        reproducibility_checksum=checksum,
        duration_s=time.time() - started,
        native_training_launched=native_training_launched,
        candidate_source=candidate_source,
        extra={
            "k_candidates_per_puzzle": int(k_candidates),
            "n_candidate_pools": len(pools),
            "dataset_dir": str(data_dir),
        },
    )
    return write_artifact(output_path, artifact)


def main() -> None:  # pragma: no cover - CLI wrapper.
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
