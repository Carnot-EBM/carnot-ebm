"""Exp 4099 TRM-pool verifier discrimination probe.

Spec refs: REQ-LEARN-4099, SCENARIO-LEARN-4099,
SCENARIO-LEARN-4099-UNDERPOWERED.
"""

from __future__ import annotations

import glob
import hashlib
import json
import random
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_FILENAME = "experiment_4099_trm_pool_verifier_discrimination_probe.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
TRM_ROOT = Path("/home/ianblenke/trm_src")
DEFAULT_CACHED_ARTIFACTS = (
    Path("results/trm_verifier_rerank_opportunity.json"),
    Path("results/arc3_trm_verifier_rerank_gap1.json"),
)
RANDOM_SEED = 4099
BOOTSTRAP_RESAMPLES = 2000
POOL_TARGET_N = 100
INFERENCE_SUBSTRATE = "offline_saved_trm_candidate_grid_rerank_zero_codex"
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "verifier_beats_trm_vote",
    "captured_pp_directional",
    "pool_n_tasks",
    "underpowered",
    "per_reranker",
    "oracle_ceiling",
    "n_tasks_scored",
    "random_seed",
    "reproducibility_checksum",
)
RERANKERS = (
    "TRM_VOTE",
    "DEMO_FIT",
    "AUG_INVARIANCE",
    "K_OF_N_AGREEMENT",
    "MIN_HAMMING",
    "STACK_DEMO_AUG",
    "STACK_ALL",
)


@dataclass(frozen=True)
class CandidateGrid:
    """One unique TRM candidate grid after de-augmentation and vote grouping."""

    candidate_id: str
    grid: list[list[int]]
    vote_count: int
    avg_q: float
    correct: bool


@dataclass(frozen=True)
class TaskCandidatePool:
    """One ARC test example with public demos and TRM candidate grids."""

    task_id: str
    train_pairs: list[dict[str, Any]]
    test_input: Any
    candidates: list[CandidateGrid]
    source: str


@dataclass(frozen=True)
class ScoredCandidate:
    """A candidate grid plus model-free verifier signals."""

    candidate_id: str
    candidate: CandidateGrid
    signals: dict[str, float]


@dataclass(frozen=True)
class PreconditionCheck:
    """One resource check recorded before offline replay."""

    resource: str
    available: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SavedPoolSpec:
    """Location of saved TRM outputs and the ARC puzzle metadata they use."""

    name: str
    data_path: Path
    preds_glob: str


DEFAULT_POOL_SPECS = (
    SavedPoolSpec(
        "arc1_latent_saved_outputs",
        TRM_ROOT / "data" / "arc1concept-aug-1000",
        str(TRM_ROOT / "eval_out" / "arc_v1_latent" / "step_0_all_preds.*"),
    ),
    SavedPoolSpec(
        "arc2_saved_outputs",
        TRM_ROOT / "data" / "arc2concept-aug-1000",
        str(TRM_ROOT / "eval_out" / "arc_v2" / "step_0_all_preds.*"),
    ),
)


def _grid_array(grid: Any) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError("grid must be two-dimensional")
    return arr


def _grid_hash(grid: Any) -> str:
    arr = _grid_array(grid)
    payload = repr(tuple(arr.shape)).encode("ascii") + arr.tobytes()
    return hashlib.sha1(payload).hexdigest()


def _safe_grid_hash(grid: Any) -> str:
    try:
        return _grid_hash(grid)
    except Exception:
        encoded = json.dumps(grid, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        return "invalid:" + hashlib.sha1(encoded).hexdigest()


def _normalized_hamming(left: Any, right: Any) -> float:
    try:
        left_arr = _grid_array(left)
        right_arr = _grid_array(right)
    except Exception:
        return 1.0
    if left_arr.shape != right_arr.shape or left_arr.size == 0:
        return 1.0
    return float(np.mean(left_arr != right_arr))


def _first_seen_color_canonical(grid: Any) -> tuple[tuple[int, ...], ...]:
    arr = _grid_array(grid)
    remap: dict[int, int] = {}
    next_id = 0
    rows: list[tuple[int, ...]] = []
    for row in arr.tolist():
        out_row = []
        for value in row:
            color = int(value)
            if color not in remap:
                remap[color] = next_id
                next_id += 1
            out_row.append(remap[color])
        rows.append(tuple(out_row))
    return tuple(rows)


def _d4_color_signature(grid: Any) -> str:
    arr = _grid_array(grid)
    variants = []
    for turns in range(4):
        rotated = np.rot90(arr, turns)
        variants.append(_first_seen_color_canonical(rotated))
        variants.append(_first_seen_color_canonical(np.fliplr(rotated)))
    return repr(min(variants))


def _demo_output_grids(train_pairs: Sequence[Mapping[str, Any]]) -> list[Any]:
    return [pair["output"] for pair in train_pairs if isinstance(pair, Mapping) and "output" in pair]


def _demo_fit_energy(grid: Any, train_pairs: Sequence[Mapping[str, Any]]) -> float:
    demo_outputs = _demo_output_grids(train_pairs)
    if not demo_outputs:
        return 1.0
    energies = [_normalized_hamming(grid, demo) for demo in demo_outputs]
    return round(float(min(energies)), 6)


def _augmentation_signature_energy(grid: Any, train_pairs: Sequence[Mapping[str, Any]]) -> float:
    demo_outputs = _demo_output_grids(train_pairs)
    if not demo_outputs:
        return 1.0
    try:
        candidate_signature = _d4_color_signature(grid)
        demo_signatures = {_d4_color_signature(demo) for demo in demo_outputs}
    except Exception:
        return 1.0
    return 0.0 if candidate_signature in demo_signatures else 1.0


def _consensus_grid(candidates: Sequence[CandidateGrid]) -> list[list[int]] | None:
    if not candidates:
        return None
    winner = max(candidates, key=lambda item: (item.vote_count, item.avg_q, item.candidate_id))
    return winner.grid


def score_task_candidates(task: TaskCandidatePool) -> list[ScoredCandidate]:
    """REQ-LEARN-4099: compute model-free signals for a task's TRM grids."""

    max_votes = max((candidate.vote_count for candidate in task.candidates), default=1)
    consensus = _consensus_grid(task.candidates)
    scored: list[ScoredCandidate] = []
    for order, candidate in enumerate(task.candidates):
        vote_fraction = candidate.vote_count / max(max_votes, 1)
        signals = {
            "demo_fit_energy": _demo_fit_energy(candidate.grid, task.train_pairs),
            "augmentation_invariance_energy": _augmentation_signature_energy(candidate.grid, task.train_pairs),
            "agreement_count": float(candidate.vote_count),
            "vote_fraction": round(float(vote_fraction), 6),
            "min_hamming_energy": _normalized_hamming(candidate.grid, consensus) if consensus is not None else 1.0,
            "avg_q": float(candidate.avg_q),
            "candidate_order": float(order),
        }
        scored.append(ScoredCandidate(candidate.candidate_id, candidate, signals))
    return scored


def _sort_key(row: ScoredCandidate, reranker: str) -> tuple[float | str, ...]:
    s = row.signals
    if reranker == "TRM_VOTE":
        return (-s["agreement_count"], -s["avg_q"], s["candidate_order"], row.candidate_id)
    if reranker == "DEMO_FIT":
        return (s["demo_fit_energy"], s["candidate_order"], row.candidate_id)
    if reranker == "AUG_INVARIANCE":
        return (s["augmentation_invariance_energy"], s["demo_fit_energy"], s["candidate_order"], row.candidate_id)
    if reranker == "K_OF_N_AGREEMENT":
        return (-s["agreement_count"], s["demo_fit_energy"], s["candidate_order"], row.candidate_id)
    if reranker == "MIN_HAMMING":
        return (s["min_hamming_energy"], s["demo_fit_energy"], -s["agreement_count"], s["candidate_order"], row.candidate_id)
    if reranker == "STACK_DEMO_AUG":
        return (s["demo_fit_energy"], s["augmentation_invariance_energy"], s["candidate_order"], row.candidate_id)
    if reranker == "STACK_ALL":
        return (
            s["demo_fit_energy"],
            s["augmentation_invariance_energy"],
            s["min_hamming_energy"],
            -s["agreement_count"],
            s["candidate_order"],
            row.candidate_id,
        )
    raise ValueError(f"unknown reranker: {reranker}")


def rank_candidates(scored: Sequence[ScoredCandidate], reranker: str) -> list[ScoredCandidate]:
    """REQ-LEARN-4099: rank candidates by one deterministic signal stack."""

    return sorted(scored, key=lambda row: _sort_key(row, reranker))


def _pass_at_k(ranked: Sequence[ScoredCandidate], k: int) -> bool:
    return any(row.candidate.correct for row in ranked[:k])


def _mean(values: Sequence[bool]) -> float:
    return round(sum(int(value) for value in values) / len(values), 4) if values else 0.0


def _bootstrap_ci(diffs: Sequence[float], *, n_boot: int, seed: int) -> tuple[float, float]:
    if not diffs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(diffs)
    samples = []
    for _ in range(n_boot):
        samples.append(sum(diffs[rng.randrange(n)] for _idx in range(n)) / n)
    samples.sort()
    lo = samples[int(0.025 * (n_boot - 1))]
    hi = samples[int(0.975 * (n_boot - 1))]
    return (round(float(lo), 4), round(float(hi), 4))


def _pool_checksum(pool: Sequence[TaskCandidatePool]) -> str:
    stable = []
    for task in sorted(pool, key=lambda item: item.task_id):
        stable.append(
            {
                "task_id": task.task_id,
                "source": task.source,
                "test_input_hash": _safe_grid_hash(task.test_input),
                "train_output_hashes": [_safe_grid_hash(grid) for grid in _demo_output_grids(task.train_pairs)],
                "candidates": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "grid_hash": _safe_grid_hash(candidate.grid),
                        "vote_count": candidate.vote_count,
                        "avg_q": round(candidate.avg_q, 8),
                        "correct": candidate.correct,
                    }
                    for candidate in sorted(task.candidates, key=lambda item: item.candidate_id)
                ],
            }
        )
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _field_principles() -> dict[str, str]:
    return {
        "honest_verdict": "Terminal-prefixed verdict; honest no-win outcomes are complete findings.",
        "verifier_beats_trm_vote": "Bare bool strict gate: true only for a CI-separated pass@2 win over TRM_VOTE.",
        "captured_pp_directional": "Best non-TRM reranker's point estimate versus TRM_VOTE pass@2.",
        "pool_n_tasks": "Actual number of TRM test examples reranked.",
        "underpowered": "Bare bool marking pools below the requested 100-task target.",
        "per_reranker": "Per-reranker pass@1, pass@2, captured_pp, and paired bootstrap CI.",
        "oracle_ceiling": "Perfect-selector pass@2 ceiling from candidate-pool coverage.",
        "n_tasks_scored": "Same counted selection unit as pool_n_tasks.",
        "random_seed": "Deterministic bootstrap and ordering seed.",
        "reproducibility_checksum": "SHA-256 content hash of the replayed TRM candidate pool.",
    }


def build_probe_artifact(
    pool: Sequence[TaskCandidatePool],
    *,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    random_seed: int = RANDOM_SEED,
    preconditions_checked: Sequence[PreconditionCheck] = (),
    duration_s: float = 0.0,
) -> dict[str, Any]:
    """REQ-LEARN-4099: evaluate rerankers and build the terminal artifact."""

    scored_by_task = [(task, score_task_candidates(task)) for task in pool if task.candidates]
    task_count = len(scored_by_task)
    per_task: list[dict[str, Any]] = []
    per_reranker: dict[str, dict[str, Any]] = {}
    pass1_by_reranker: dict[str, list[bool]] = {name: [] for name in RERANKERS}
    pass2_by_reranker: dict[str, list[bool]] = {name: [] for name in RERANKERS}
    oracle_hits = []

    for task, scored in scored_by_task:
        task_row: dict[str, Any] = {"task_id": task.task_id, "source": task.source, "n_candidates": len(scored)}
        oracle_hit = any(row.candidate.correct for row in scored)
        oracle_hits.append(oracle_hit)
        for reranker in RERANKERS:
            ranked = rank_candidates(scored, reranker)
            p1 = _pass_at_k(ranked, 1)
            p2 = _pass_at_k(ranked, 2)
            pass1_by_reranker[reranker].append(p1)
            pass2_by_reranker[reranker].append(p2)
            task_row[f"{reranker}_pass@1"] = p1
            task_row[f"{reranker}_pass@2"] = p2
        per_task.append(task_row)

    baseline_pass2 = pass2_by_reranker["TRM_VOTE"]
    for offset, reranker in enumerate(RERANKERS):
        pass1 = _mean(pass1_by_reranker[reranker])
        pass2 = _mean(pass2_by_reranker[reranker])
        diffs = [
            float(int(this) - int(base))
            for this, base in zip(pass2_by_reranker[reranker], baseline_pass2, strict=True)
        ]
        ci = _bootstrap_ci(diffs, n_boot=n_boot, seed=random_seed + offset)
        per_reranker[reranker] = {
            "pass@1": pass1,
            "pass@2": pass2,
            "captured_pp": round(float(sum(diffs) / len(diffs)), 4) if diffs else 0.0,
            "captured_pp_ci95": [ci[0], ci[1]],
        }

    verifier_names = [name for name in RERANKERS if name != "TRM_VOTE"]
    best_name = max(
        verifier_names,
        key=lambda name: (
            per_reranker[name]["captured_pp"],
            per_reranker[name]["pass@2"],
            name,
        ),
    )
    best = per_reranker[best_name]
    captured = float(best["captured_pp"])
    ci_low, ci_high = best["captured_pp_ci95"]
    strict_win = captured > 0.0 and ci_low > 0.0
    underpowered = task_count < POOL_TARGET_N
    verdict_status = "verifier_beats_trm_vote" if strict_win else "no_verifier_beats_trm_vote"
    verdict = (
        f"complete: {verdict_status}_best_{best_name}_captured_{captured:.4f}_"
        f"ci95_{ci_low:.4f}_{ci_high:.4f}_n{task_count}_underpowered_{str(underpowered).lower()}"
    )
    artifact: dict[str, Any] = {
        "experiment": "experiment_4099_trm_pool_verifier_discrimination_probe",
        "schema": "carnot.experiment_4099_trm_pool_verifier_discrimination_probe.v1",
        "honest_verdict": verdict,
        "verifier_beats_trm_vote": strict_win,
        "captured_pp_directional": round(captured, 4),
        "pool_n_tasks": task_count,
        "underpowered": underpowered,
        "per_reranker": per_reranker,
        "oracle_ceiling": {"pass@1": _mean(oracle_hits), "pass@2": _mean(oracle_hits)},
        "n_tasks_scored": task_count,
        "random_seed": int(random_seed),
        "reproducibility_checksum": _pool_checksum([task for task, _scored in scored_by_task]),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "n_codex_calls": 0,
        "bootstrap_resamples": int(n_boot),
        "best_reranker": best_name,
        "trm_vote_pass2": per_reranker["TRM_VOTE"]["pass@2"],
        "preconditions_checked": [check.to_dict() for check in preconditions_checked],
        "pool_sources": sorted({task.source for task, _scored in scored_by_task}),
        "pool_target_n_tasks": POOL_TARGET_N,
        "duration_s": round(float(duration_s), 3),
        "field_principles": _field_principles(),
        "spec_refs": ["REQ-LEARN-4099", "SCENARIO-LEARN-4099", "SCENARIO-LEARN-4099-UNDERPOWERED"],
        "per_task": per_task,
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def build_blocked_artifact(
    reason: str,
    *,
    preconditions_checked: Sequence[PreconditionCheck],
    duration_s: float = 0.0,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "experiment": "experiment_4099_trm_pool_verifier_discrimination_probe",
        "schema": "carnot.experiment_4099_trm_pool_verifier_discrimination_probe.v1",
        "honest_verdict": reason,
        "verifier_beats_trm_vote": False,
        "captured_pp_directional": 0.0,
        "pool_n_tasks": 0,
        "underpowered": True,
        "per_reranker": {},
        "oracle_ceiling": {"pass@1": 0.0, "pass@2": 0.0},
        "n_tasks_scored": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": hashlib.sha256(reason.encode("utf-8")).hexdigest(),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "n_codex_calls": 0,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "preconditions_checked": [check.to_dict() for check in preconditions_checked],
        "pool_sources": [],
        "pool_target_n_tasks": POOL_TARGET_N,
        "duration_s": round(float(duration_s), 3),
        "field_principles": _field_principles(),
        "spec_refs": ["REQ-LEARN-4099"],
        "per_task": [],
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed")
    for field in ("verifier_beats_trm_vote", "underpowered"):
        if not isinstance(artifact.get(field), bool):
            errors.append(f"{field} must be a bare bool")
    if not isinstance(artifact.get("captured_pp_directional"), (float, int)) or isinstance(
        artifact.get("captured_pp_directional"), bool
    ):
        errors.append("captured_pp_directional must be numeric")
    for field in ("pool_n_tasks", "n_tasks_scored", "random_seed"):
        if not isinstance(artifact.get(field), int) or isinstance(artifact.get(field), bool):
            errors.append(f"{field} must be a bare int")
    if not isinstance(artifact.get("reproducibility_checksum"), str):
        errors.append("reproducibility_checksum must be a string")
    oracle = artifact.get("oracle_ceiling")
    if not isinstance(oracle, Mapping) or not isinstance(oracle.get("pass@2"), (float, int)):
        errors.append("oracle_ceiling must include numeric pass@2")
    per_reranker = artifact.get("per_reranker")
    if not isinstance(per_reranker, Mapping):
        errors.append("per_reranker must be a dict")
    else:
        required = {"pass@1", "pass@2", "captured_pp", "captured_pp_ci95"}
        for row in per_reranker.values():
            ci = row.get("captured_pp_ci95") if isinstance(row, Mapping) else None
            if not isinstance(row, Mapping) or not required.issubset(row) or not isinstance(ci, list) or len(ci) != 2:
                errors.append("per_reranker entries must include pass@1, pass@2, captured_pp, captured_pp_ci95")
                break
    if artifact.get("inference_substrate") not in (None, INFERENCE_SUBSTRATE):
        errors.append("inference_substrate must declare offline saved TRM candidate rerank")
    return errors


def _check_json_artifact(path: Path, resource: str) -> PreconditionCheck:
    if not path.exists() or path.stat().st_size <= 0:
        return PreconditionCheck(resource, False, f"missing or empty: {path}")
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return PreconditionCheck(resource, False, f"{type(exc).__name__}: {exc}")
    return PreconditionCheck(resource, True, f"loaded {path}")


def check_preconditions(*, repo_root: str | Path = REPO_ROOT) -> tuple[list[PreconditionCheck], str | None]:
    root = Path(repo_root)
    checks = [
        _check_json_artifact(root / relative, relative.stem)
        for relative in DEFAULT_CACHED_ARTIFACTS
    ]
    missing = next((check for check in checks if not check.available), None)
    return checks, "blocked_trm_pool_missing" if missing else None


def _load_saved_pool_spec(spec: SavedPoolSpec) -> list[TaskCandidatePool]:  # pragma: no cover - external TRM dump loader
    if str(TRM_ROOT) not in sys.path:
        sys.path.insert(0, str(TRM_ROOT))
    import torch  # pylint: disable=import-outside-toplevel
    from dataset.build_arc_dataset import arc_grid_to_np, grid_hash  # pylint: disable=import-outside-toplevel
    from evaluators.arc import ARC  # pylint: disable=import-outside-toplevel
    from types import SimpleNamespace  # pylint: disable=import-outside-toplevel

    evaluator = ARC(data_path=str(spec.data_path), eval_metadata=SimpleNamespace(blank_identifier_id=0))
    for shard in sorted(glob.glob(spec.preds_glob)):
        payload = torch.load(shard, map_location="cpu")
        if "q_halt_logits" not in payload:
            continue
        evaluator.update_batch(
            {"inputs": payload["inputs"], "puzzle_identifiers": payload["puzzle_identifiers"]},
            {"preds": payload["preds"], "q_halt_logits": payload["q_halt_logits"]},
        )

    tasks: list[TaskCandidatePool] = []
    for task_name, puzzle in sorted(evaluator.test_puzzles.items()):
        for test_index, pair in enumerate(puzzle["test"]):
            input_hash = grid_hash(arc_grid_to_np(pair["input"]))
            label_hash = grid_hash(arc_grid_to_np(pair["output"]))
            preds = evaluator._local_preds.get(task_name, {}).get(input_hash)  # noqa: SLF001
            if not preds:
                continue
            grouped: dict[str, list[float]] = {}
            for pred_hash, q_value in preds:
                grouped.setdefault(pred_hash, [0.0, 0.0])
                grouped[pred_hash][0] += 1.0
                grouped[pred_hash][1] += float(q_value)
            candidates = []
            for pred_hash, (count, q_sum) in sorted(grouped.items()):
                grid = evaluator._local_hmap[pred_hash].astype(int).tolist()  # noqa: SLF001
                vote_count = int(count)
                candidates.append(
                    CandidateGrid(
                        candidate_id=f"{spec.name}:{task_name}:{test_index}:{pred_hash}",
                        grid=grid,
                        vote_count=vote_count,
                        avg_q=float(q_sum / max(vote_count, 1)),
                        correct=pred_hash == label_hash,
                    )
                )
            tasks.append(
                TaskCandidatePool(
                    task_id=f"{spec.name}:{task_name}:{test_index}",
                    train_pairs=list(puzzle["train"]),
                    test_input=pair["input"],
                    candidates=candidates,
                    source=spec.name,
                )
            )
    return tasks


def load_default_trm_pool(  # pragma: no cover - external TRM dump loader
    pool_specs: Sequence[SavedPoolSpec] = DEFAULT_POOL_SPECS,
) -> list[TaskCandidatePool]:
    pool: list[TaskCandidatePool] = []
    for spec in pool_specs:
        pool.extend(_load_saved_pool_spec(spec))
    return pool


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    output_path: str | Path | None = None,
    pool_loader: Callable[[], Sequence[TaskCandidatePool]] = load_default_trm_pool,
) -> dict[str, Any]:
    started = time.time()
    checks, blocker = check_preconditions(repo_root=repo_root)
    out_path = Path(output_path) if output_path is not None else Path(repo_root) / "results" / RESULT_FILENAME
    if blocker:
        artifact = build_blocked_artifact(blocker, preconditions_checked=checks, duration_s=time.time() - started)
        _write_json(out_path, artifact)
        return artifact
    try:
        pool = list(pool_loader())
    except Exception as exc:
        checks = [*checks, PreconditionCheck("saved_trm_candidate_grid_pool", False, f"{type(exc).__name__}: {exc}")]
        artifact = build_blocked_artifact("blocked_trm_pool_missing", preconditions_checked=checks, duration_s=time.time() - started)
        _write_json(out_path, artifact)
        return artifact
    if not pool:
        checks = [*checks, PreconditionCheck("saved_trm_candidate_grid_pool", False, "no task candidates loaded")]
        artifact = build_blocked_artifact("blocked_trm_pool_missing", preconditions_checked=checks, duration_s=time.time() - started)
    else:
        checks = [
            *checks,
            PreconditionCheck(
                "saved_trm_candidate_grid_pool",
                True,
                f"loaded {len(pool)} task examples from saved TRM outputs",
            ),
        ]
        artifact = build_probe_artifact(pool, preconditions_checked=checks, duration_s=time.time() - started)
    _write_json(out_path, artifact)
    return artifact


def main() -> None:  # pragma: no cover - CLI shim
    artifact = run_experiment()
    print(artifact["honest_verdict"])
    print(f"wrote {DEFAULT_OUTPUT}")


if __name__ == "__main__":  # pragma: no cover - CLI shim
    main()
