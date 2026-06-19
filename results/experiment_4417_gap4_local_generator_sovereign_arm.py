#!/usr/bin/env python3
"""Exp 4417: GAP-4 local-generator sovereign forward arm.

This runner is deliberately replay-based. It verifies that a local open-weight
GGUF exists and that cached local-GGUF program samples are available, then
re-scores those samples with the GAP-4 execution gate. It does not train TRM,
fine-tune a generator, call Codex, or submit to any leaderboard.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
RESULTS_DIR = ROOT / "results"
for _path in (PYTHON_DIR, RESULTS_DIR):
    if str(_path) not in sys.path:  # pragma: no cover - import bootstrap.
        sys.path.insert(0, str(_path))

from carnot.agentic.gap4_graded_execution_gate import (  # noqa: E402
    DEFAULT_HIGH_VOTE_GUARD_THRESHOLD,
    DEFAULT_TAU,
    gated_rank_indices,
    hit_indices,
    pass_at_k,
    select_guarded_graded_candidate,
    vote_rank_indices,
)
from experiment_4188_sovereign_local_generator_gap4_self_distill import (  # noqa: E402
    _to_grid_list,
    demo_fit,
    safe_transform_from_code,
)


ARTIFACT_PATH = RESULTS_DIR / "experiment_4417_gap4_local_generator_sovereign_arm.json"
POOL_PATH = RESULTS_DIR / "arc3_gap3_stage2_eval_pool.json.gz"
VOTE_BASELINE_PATH = RESULTS_DIR / "arc3_trm_verifier_rerank.json"
CHECKPOINT_PATH = RESULTS_DIR / "decentralization_moe_qwen35a3b_arc1_k8.checkpoint.json"

RANDOM_SEED = 12345
K_CONSISTENCY = 2
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

LOCAL_MODEL_CANDIDATES = [
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "moe",
    },
    {
        "name": "Gemma4-31B-it",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "dense",
    },
    {
        "name": "Gemma4-12B-it",
        "hf_id": "unsloth/gemma-4-12B-it-GGUF",
        "role": "dense",
    },
]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A sovereign gate that holds (local generator + graded gate >= vote) "
        "and a clean null (local generator cannot drive the gate -> decentralization gap) are "
        "BOTH decision-grade."
    ),
    "sovereign_gap4_gate_holds": (
        "BARE bool: the capstone reads this; true iff a LOCAL-generator demo-fit + "
        "graded-min-hamming + k-consistency gate holds pass@2 >= vote with 0 pass@2 losses "
        "on the matched pool (CI95) -- the decentralization/sovereignty tier of the GAP-4 "
        "forward protocol."
    ),
    "local_generator_coverage": (
        "BARE float: the demo-perfect program rate from the LOCAL generator (vs exp4069's ~0.23) "
        "-- the raw sovereign-induction signal separate from the gate's pass@2 effect."
    ),
    "pass2_vs_vote": (
        "dict: {vote_pass2, gated_pass2, delta, delta_ci95, pass2_vote_wins_lost, "
        "graded_gate_fires} -- the gate's pass@2 effect on the matched pool with the "
        "no-verifier control."
    ),
    "verifier_is_oracle": (
        "BARE bool=true -- the demo-fit execution gate IS the oracle (execution-grounded); "
        "this is the SOVEREIGNTY question, NOT an oracle-distinct moat headline."
    ),
    "preconditions_checked": (
        "Records the cached-pool + the cached local generator + TRM-stand-down verified; "
        "pre-empts the silent-missing-resource fabrication mode."
    ),
    "random_seed": "Determinism precondition for the local induction + the graded gate + the bootstrap.",
    "reproducibility_checksum": (
        "Hash of the local-generator induced programs + the gate config + the pool + the control; "
        "lets a third party re-run."
    ),
    "model_specs": (
        "The LOCAL open-weight generator GGUF (the sovereign inducer) + the cached TRM pool + "
        "the graded-gate config + the vote baseline + n; required methodology + the "
        "decentralization (local/open) declaration."
    ),
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_pool(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("candidate pool missing entries list")
    return entries


def _load_vote_baseline(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rankers = payload.get("rankers")
    vote = rankers.get("TRM_VOTE") if isinstance(rankers, dict) else None
    pass2 = payload.get("trm_vote_pass2")
    if isinstance(vote, dict) and isinstance(vote.get("pass@2"), int | float):
        pass2 = vote["pass@2"]
    if not isinstance(pass2, int | float):
        raise ValueError("vote baseline missing TRM_VOTE pass@2")
    return {"artifact_pass2": float(pass2), "payload": payload}


def _load_checkpoint(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("local checkpoint missing tasks map")
    return payload


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _grid_hash(grid: Any) -> str:
    return hashlib.sha256(json.dumps(grid, sort_keys=True).encode("utf-8")).hexdigest()


def _draw_index(sample: dict[str, Any]) -> int:
    value = sample.get("draw_index", 0)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def resolve_cached_local_gguf(cache_root: Path | str | None = None) -> dict[str, Any] | None:
    root = Path(cache_root) if cache_root is not None else Path.home() / ".cache" / "huggingface" / "hub"
    for spec in LOCAL_MODEL_CANDIDATES:  # pragma: no branch - exits on the first cached allowed GGUF.
        model_dir = root / f"models--{spec['hf_id'].replace('/', '--')}"
        if not model_dir.exists():
            continue
        hits = [
            path
            for path in sorted(model_dir.rglob("*.gguf"))
            if ".no_exist" not in path.parts and path.is_file()
        ]
        if hits:  # pragma: no branch - real run intentionally returns the first cached allowed model.
            return {**spec, "model_path": str(hits[0])}
    return None


def llama_vocab_preflight(model_path: str) -> bool:  # pragma: no cover - real GGUF dependency.
    try:
        from llama_cpp import Llama

        llm = Llama(model_path=model_path, vocab_only=True, verbose=False)
        return bool(llm.tokenize(b"test"))
    except Exception:
        return False


def blocked_artifact(
    *,
    verdict: str,
    preconditions: list[dict[str, Any]],
    duration_s: float,
    model_specs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = {
        "experiment": "experiment_4417_gap4_local_generator_sovereign_arm",
        "schema": "carnot.experiment_4417_gap4_local_generator_sovereign_arm.v1",
        "title": "GAP-4 local generator sovereign forward arm",
        "honest_verdict": verdict,
        "sovereign_gap4_gate_holds": False,
        "local_generator_coverage": 0.0,
        "pass2_vs_vote": {
            "vote_pass2": 0.0,
            "gated_pass2": 0.0,
            "delta": 0.0,
            "delta_ci95": [0.0, 0.0],
            "pass2_vote_wins_lost": 0,
            "graded_gate_fires": 0,
        },
        "verifier_is_oracle": True,
        "preconditions_checked": preconditions,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "model_specs": model_specs or {},
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(duration_s, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
    }
    validate_artifact(artifact)
    return artifact


def _cached_pool_precondition(pool_path: Path, vote_baseline_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]] | None, dict[str, Any] | None]:
    try:
        entries = _load_pool(pool_path)
        vote = _load_vote_baseline(vote_baseline_path)
    except Exception as exc:
        return (
            {
                "resource": "cached_trm_pool_and_vote_baseline",
                "available": False,
                "pool_path": str(pool_path),
                "vote_baseline_path": str(vote_baseline_path),
                "error": type(exc).__name__,
            },
            None,
            None,
        )
    return (
        {
            "resource": "cached_trm_pool_and_vote_baseline",
            "available": True,
            "pool_path": str(pool_path),
            "vote_baseline_path": str(vote_baseline_path),
            "n_entries": len(entries),
            "vote_baseline_artifact_pass2": round(float(vote["artifact_pass2"]), 4),
        },
        entries,
        vote,
    )


def _local_generator_precondition(
    *,
    cache_root: Path | str | None,
    resolver: Callable[[Path | str | None], dict[str, Any] | None],
    preflight_fn: Callable[[str], bool],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    spec = resolver(cache_root)
    ok = bool(spec and str(spec.get("model_path", "")).endswith(".gguf") and preflight_fn(str(spec["model_path"])))
    return (
        {
            "resource": "local_open_weight_generator_gguf",
            "available": ok,
            "allowed_hf_ids": [row["hf_id"] for row in LOCAL_MODEL_CANDIDATES],
            "selected_hf_id": spec.get("hf_id") if spec else None,
            "model_path": spec.get("model_path") if spec else None,
            "loader": "llama_cpp.Llama(model_path=..., vocab_only=True)",
        },
        spec if ok else None,
    )


def _verified_predictions_for_task(
    entry: dict[str, Any],
    samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    records: list[dict[str, Any]] = []
    has_demo_perfect = False
    for sample in sorted(samples, key=_draw_index):
        code = sample.get("code")
        if not isinstance(code, str) or not code.strip():
            continue
        fn = safe_transform_from_code(code)
        if fn is None:
            continue
        fit = float(demo_fit(fn, entry.get("demos", [])))
        if fit < 1.0:
            continue
        has_demo_perfect = True
        pred = _to_grid_list(fn(entry.get("test_input")))
        if pred is None:
            continue
        records.append(
            {
                "task": entry["task"],
                "draw_index": _draw_index(sample),
                "pred_grid": pred,
                "pred_hash": _grid_hash(pred),
                "code_sha256": hashlib.sha256(code.strip().encode("utf-8")).hexdigest(),
            }
        )
    return records, has_demo_perfect


def build_local_program_records(
    entries: list[dict[str, Any]],
    checkpoint_payload: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    tasks = checkpoint_payload["tasks"]
    by_task: dict[str, dict[str, Any]] = {}
    for entry in entries:
        by_task.setdefault(entry["task"], entry)

    records_by_task: dict[str, list[dict[str, Any]]] = {}
    demo_perfect_tasks: set[str] = set()
    for task, entry in sorted(by_task.items()):
        samples = tasks.get(task, [])
        if not isinstance(samples, list):
            samples = []
        records, has_demo_perfect = _verified_predictions_for_task(entry, samples)
        records_by_task[task] = records
        if has_demo_perfect:
            demo_perfect_tasks.add(task)
    return records_by_task, demo_perfect_tasks


def _k_consistent_prediction(records: list[dict[str, Any]], *, k: int) -> dict[str, Any] | None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["pred_hash"]].append(record)
    eligible = [rows for rows in grouped.values() if len(rows) >= k]
    if not eligible:
        return None
    eligible.sort(key=lambda rows: min(row["draw_index"] for row in rows))
    rows = sorted(eligible[0], key=lambda row: row["draw_index"])
    return {
        "pred_grid": rows[0]["pred_grid"],
        "pred_hash": rows[0]["pred_hash"],
        "agreement_count": len(rows),
        "draw_indices": [row["draw_index"] for row in rows],
    }


def score_k_consistent_gate(
    entries: list[dict[str, Any]],
    records_by_task: dict[str, list[dict[str, Any]]],
    *,
    k_consistency: int = K_CONSISTENCY,
    tau: float = DEFAULT_TAU,
    high_vote_guard_threshold: int | float | None = DEFAULT_HIGH_VOTE_GUARD_THRESHOLD,
) -> dict[str, Any]:
    vote_rankings = [vote_rank_indices(entry["candidates"]) for entry in entries]
    gated_rankings: list[list[int]] = []
    selections: list[dict[str, Any]] = []
    k_consistent_entries = 0
    for entry in entries:
        prediction = _k_consistent_prediction(records_by_task.get(entry["task"], []), k=k_consistency)
        if prediction is None:
            selections.append(
                {
                    "task_id": entry["task"],
                    "gate_fired": False,
                    "selected_index": None,
                    "reason": "k_consistency_not_met",
                    "agreement_count": 0,
                }
            )
            gated_rankings.append(vote_rank_indices(entry["candidates"]))
            continue
        k_consistent_entries += 1
        selection = select_guarded_graded_candidate(
            entry["candidates"],
            prediction=prediction["pred_grid"],
            demo_fit=1.0,
            task_id=entry["task"],
            tau=tau,
            high_vote_guard_threshold=high_vote_guard_threshold,
            agreement_confidence_label=True,
        )
        selection["agreement_count"] = prediction["agreement_count"]
        selection["pred_hash"] = prediction["pred_hash"]
        selection["draw_indices"] = prediction["draw_indices"]
        selections.append(selection)
        gated_rankings.append(gated_rank_indices(entry["candidates"], selection["selected_index"]))

    vote_hits = hit_indices(entries, vote_rankings, 2)
    gated_hits = hit_indices(entries, gated_rankings, 2)
    return {
        "vote_rankings": vote_rankings,
        "gated_rankings": gated_rankings,
        "selections": selections,
        "pass1_vote": pass_at_k(entries, vote_rankings, 1),
        "pass1_gate": pass_at_k(entries, gated_rankings, 1),
        "pass2_vote": pass_at_k(entries, vote_rankings, 2),
        "pass2_gate": pass_at_k(entries, gated_rankings, 2),
        "recovered": len(gated_hits - vote_hits),
        "lost": len(vote_hits - gated_hits),
        "graded_gate_fires": sum(1 for row in selections if row["gate_fired"]),
        "guard_block_count": sum(1 for row in selections if row.get("guard_blocked")),
        "k_consistent_entries": k_consistent_entries,
    }


def _paired_hit_vectors(
    entries: list[dict[str, Any]],
    vote_rankings: list[list[int]],
    gated_rankings: list[list[int]],
) -> tuple[list[int], list[int]]:
    vote_hits = []
    gated_hits = []
    for entry, vote_order, gate_order in zip(entries, vote_rankings, gated_rankings, strict=True):
        candidates = entry["candidates"]
        vote_hits.append(int(any(candidates[i].get("correct", False) for i in vote_order[:2])))
        gated_hits.append(int(any(candidates[i].get("correct", False) for i in gate_order[:2])))
    return vote_hits, gated_hits


def bootstrap_delta_ci95(
    vote_hits: list[int],
    gated_hits: list[int],
    *,
    seed: int = RANDOM_SEED,
    n_bootstrap: int = 2000,
) -> list[float]:
    if len(vote_hits) != len(gated_hits):
        raise ValueError("paired hit vectors must have equal length")
    n = len(vote_hits)
    if n == 0:
        return [0.0, 0.0]
    rng = random.Random(seed)
    deltas = []
    for _ in range(max(1, n_bootstrap)):
        total = 0
        for _j in range(n):
            idx = rng.randrange(n)
            total += gated_hits[idx] - vote_hits[idx]
        deltas.append(total / n)
    deltas.sort()
    lo = deltas[int(0.025 * (len(deltas) - 1))]
    hi = deltas[int(0.975 * (len(deltas) - 1))]
    return [round(lo, 4), round(hi, 4)]


def _reproducibility_checksum(
    *,
    checkpoint_path: Path,
    pool_path: Path,
    vote_baseline_path: Path,
    model_spec: dict[str, Any],
    k_consistency: int,
    tau: float,
) -> str:
    blob = json.dumps(
        {
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "pool_sha256": _sha256_file(pool_path),
            "vote_baseline_sha256": _sha256_file(vote_baseline_path),
            "generator_hf_id": model_spec.get("hf_id"),
            "generator_path": model_spec.get("model_path"),
            "k_consistency": k_consistency,
            "tau": tau,
            "seed": RANDOM_SEED,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _verdict(*, holds: bool, coverage: float, delta: float, fires: int, lost: int) -> str:
    if holds and fires > 0 and delta > 0:
        return f"success: sovereign_gap4_local_gate_holds_pass2_delta_{delta}_cov_{coverage}_fires_{fires}"
    if holds:
        return f"complete: sovereign_gap4_local_gate_holds_flat_cov_{coverage}_fires_{fires}_lost_{lost}"
    return f"complete: clean_null_local_generator_k_consistency_gap_cov_{coverage}_fires_{fires}_lost_{lost}"


def build_artifact(
    *,
    entries: list[dict[str, Any]],
    vote_baseline: dict[str, Any],
    checkpoint_payload: dict[str, Any],
    model_spec: dict[str, Any],
    preconditions: list[dict[str, Any]],
    pool_path: Path,
    vote_baseline_path: Path,
    checkpoint_path: Path,
    duration_s: float,
    n_bootstrap: int = 2000,
    k_consistency: int = K_CONSISTENCY,
    tau: float = DEFAULT_TAU,
) -> dict[str, Any]:
    records_by_task, demo_perfect_tasks = build_local_program_records(entries, checkpoint_payload)
    unique_tasks = {entry["task"] for entry in entries}
    coverage = round(len(demo_perfect_tasks) / max(1, len(unique_tasks)), 4)
    scored = score_k_consistent_gate(entries, records_by_task, k_consistency=k_consistency, tau=tau)
    vote_hits, gated_hits = _paired_hit_vectors(
        entries,
        scored["vote_rankings"],
        scored["gated_rankings"],
    )
    delta = round(scored["pass2_gate"] - scored["pass2_vote"], 4)
    delta_ci95 = bootstrap_delta_ci95(
        vote_hits,
        gated_hits,
        seed=RANDOM_SEED,
        n_bootstrap=n_bootstrap,
    )
    holds = bool(scored["pass2_gate"] >= scored["pass2_vote"] and scored["lost"] == 0)

    artifact = {
        "experiment": "experiment_4417_gap4_local_generator_sovereign_arm",
        "schema": "carnot.experiment_4417_gap4_local_generator_sovereign_arm.v1",
        "title": "GAP-4 local generator sovereign forward arm",
        "honest_verdict": _verdict(
            holds=holds,
            coverage=coverage,
            delta=delta,
            fires=scored["graded_gate_fires"],
            lost=scored["lost"],
        ),
        "sovereign_gap4_gate_holds": holds,
        "local_generator_coverage": coverage,
        "pass2_vs_vote": {
            "vote_pass2": scored["pass2_vote"],
            "gated_pass2": scored["pass2_gate"],
            "delta": delta,
            "delta_ci95": delta_ci95,
            "pass2_vote_wins_lost": scored["lost"],
            "graded_gate_fires": scored["graded_gate_fires"],
        },
        "pass1_vs_vote": {
            "vote_pass1": scored["pass1_vote"],
            "gated_pass1": scored["pass1_gate"],
            "delta": round(scored["pass1_gate"] - scored["pass1_vote"], 4),
        },
        "verifier_is_oracle": True,
        "preconditions_checked": preconditions,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _reproducibility_checksum(
            checkpoint_path=checkpoint_path,
            pool_path=pool_path,
            vote_baseline_path=vote_baseline_path,
            model_spec=model_spec,
            k_consistency=k_consistency,
            tau=tau,
        ),
        "model_specs": {
            "generator_model": model_spec.get("name"),
            "generator_hf_id": model_spec.get("hf_id"),
            "generator_gguf_path": model_spec.get("model_path"),
            "generator_declaration": "local_open_weight_gguf_cached_checkpoint_replay",
            "source_checkpoint_path": str(checkpoint_path),
            "source_checkpoint_schema": checkpoint_payload.get("schema"),
            "source_checkpoint_k_samples_per_task": checkpoint_payload.get("k_samples_per_task"),
            "cached_trm_pool_path": str(pool_path),
            "cached_vote_baseline_path": str(vote_baseline_path),
            "vote_baseline_artifact_pass2": round(float(vote_baseline["artifact_pass2"]), 4),
            "matched_vote_control": "recomputed from candidate votes on the matched pool",
            "graded_gate_tau": tau,
            "k_consistency": k_consistency,
            "vote_aware_guard_threshold": DEFAULT_HIGH_VOTE_GUARD_THRESHOLD,
            "n_entries": len(entries),
            "n_unique_tasks": len(unique_tasks),
            "no_trm_training": True,
            "leaderboard_submission": False,
        },
        "k_consistency_details": {
            "k_consistent_entries": scored["k_consistent_entries"],
            "guard_block_count": scored["guard_block_count"],
            "recovered": scored["recovered"],
            "lost": scored["lost"],
            "demo_perfect_unique_tasks": len(demo_perfect_tasks),
            "unique_tasks": len(unique_tasks),
        },
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(duration_s, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    required = {
        "honest_verdict",
        "sovereign_gap4_gate_holds",
        "local_generator_coverage",
        "pass2_vs_vote",
        "verifier_is_oracle",
        "preconditions_checked",
        "random_seed",
        "reproducibility_checksum",
        "model_specs",
        "duration_s",
        "inference_substrate",
    }
    missing = required - artifact.keys()
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("success:") or verdict.startswith("complete:") or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if not isinstance(artifact["sovereign_gap4_gate_holds"], bool):
        raise ValueError("sovereign_gap4_gate_holds must be a bare bool")
    if not isinstance(artifact["verifier_is_oracle"], bool):
        raise ValueError("verifier_is_oracle must be a bare bool")
    if not isinstance(artifact["local_generator_coverage"], float):
        raise ValueError("local_generator_coverage must be a bare float")
    pass2 = artifact["pass2_vs_vote"]
    if not isinstance(pass2, dict):
        raise ValueError("pass2_vs_vote must be a dict")
    for field in ("vote_pass2", "gated_pass2", "delta"):
        if not isinstance(pass2.get(field), float):
            raise ValueError(f"pass2_vs_vote.{field} must be a bare float")
    if not (
        isinstance(pass2.get("delta_ci95"), list)
        and len(pass2["delta_ci95"]) == 2
        and all(isinstance(value, float) for value in pass2["delta_ci95"])
    ):
        raise ValueError("pass2_vs_vote.delta_ci95 must be a two-float list")
    for field in ("pass2_vote_wins_lost", "graded_gate_fires"):
        value = pass2.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"pass2_vs_vote.{field} must be a bare int")
    if not isinstance(artifact["preconditions_checked"], list):
        raise ValueError("preconditions_checked must be a list")
    for row in artifact["preconditions_checked"]:
        if "available" in row and not isinstance(row["available"], bool):
            raise ValueError("precondition availability must be a bare bool")
    if not isinstance(artifact["random_seed"], int) or isinstance(artifact["random_seed"], bool):
        raise ValueError("random_seed must be a bare int")
    if not isinstance(artifact["reproducibility_checksum"], str):
        raise ValueError("reproducibility_checksum must be a string")
    if not isinstance(artifact["model_specs"], dict):
        raise ValueError("model_specs must be a dict")
    if not isinstance(artifact["duration_s"], float):
        raise ValueError("duration_s must be a bare float")


def run(
    *,
    artifact_path: Path = ARTIFACT_PATH,
    pool_path: Path = POOL_PATH,
    vote_baseline_path: Path = VOTE_BASELINE_PATH,
    checkpoint_path: Path = CHECKPOINT_PATH,
    cache_root: Path | str | None = None,
    gguf_resolver: Callable[[Path | str | None], dict[str, Any] | None] = resolve_cached_local_gguf,
    gguf_preflight_fn: Callable[[str], bool] = llama_vocab_preflight,
    n_bootstrap: int = 2000,
) -> dict[str, Any]:
    started = time.time()
    artifact_path = Path(artifact_path)
    pool_path = Path(pool_path)
    vote_baseline_path = Path(vote_baseline_path)
    checkpoint_path = Path(checkpoint_path)

    pool_precondition, entries, vote_baseline = _cached_pool_precondition(pool_path, vote_baseline_path)
    if entries is None or vote_baseline is None:
        artifact = blocked_artifact(
            verdict="blocked_cached_pool_unavailable",
            preconditions=[pool_precondition],
            duration_s=time.time() - started,
        )
        _write_json(artifact_path, artifact)
        return artifact

    local_precondition, model_spec = _local_generator_precondition(
        cache_root=cache_root,
        resolver=gguf_resolver,
        preflight_fn=gguf_preflight_fn,
    )
    preconditions = [
        pool_precondition,
        local_precondition,
        {
            "resource": "trm_training_stood_down",
            "available": True,
            "evidence": "runner is offline replay; no TRM or generator training entrypoint is invoked",
        },
    ]
    if model_spec is None:
        artifact = blocked_artifact(
            verdict="blocked_local_generator_not_cached",
            preconditions=preconditions,
            duration_s=time.time() - started,
        )
        _write_json(artifact_path, artifact)
        return artifact

    try:
        checkpoint_payload = _load_checkpoint(checkpoint_path)
    except Exception as exc:
        preconditions.append(
            {
                "resource": "cached_local_generator_checkpoint",
                "available": False,
                "path": str(checkpoint_path),
                "error": type(exc).__name__,
            }
        )
        artifact = blocked_artifact(
            verdict="blocked_local_generator_not_cached",
            preconditions=preconditions,
            duration_s=time.time() - started,
            model_specs=model_spec,
        )
        _write_json(artifact_path, artifact)
        return artifact

    preconditions.append(
        {
            "resource": "cached_local_generator_checkpoint",
            "available": True,
            "path": str(checkpoint_path),
            "n_tasks": len(checkpoint_payload["tasks"]),
        }
    )
    artifact = build_artifact(
        entries=entries,
        vote_baseline=vote_baseline,
        checkpoint_payload=checkpoint_payload,
        model_spec=model_spec,
        preconditions=preconditions,
        pool_path=pool_path,
        vote_baseline_path=vote_baseline_path,
        checkpoint_path=checkpoint_path,
        duration_s=time.time() - started,
        n_bootstrap=n_bootstrap,
    )
    _write_json(artifact_path, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--pool", type=Path, default=POOL_PATH)
    parser.add_argument("--vote-baseline", type=Path, default=VOTE_BASELINE_PATH)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args(argv)
    run(
        artifact_path=args.artifact,
        pool_path=args.pool,
        vote_baseline_path=args.vote_baseline,
        checkpoint_path=args.checkpoint,
        cache_root=args.cache_root,
        n_bootstrap=args.n_bootstrap,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint.
    raise SystemExit(main())
