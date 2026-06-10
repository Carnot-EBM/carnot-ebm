"""Exp 4012 GAP-4 LOCAL best-of-N generator arm.

Spec refs: REQ-VERIFY-4012, SCENARIO-VERIFY-4012.

This runner raises only the local generator sampling budget from Exp 4002. The verifier side remains
the GAP-4 model-free path: restricted execution, demo-fit filtering, min-Hamming candidate snap, and
vote-primary gated pass@2 scoring. If the local GGUF or llama.cpp runtime is absent, the run blocks;
it never falls back to DSL-only or codex.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from arc3_gap4_rule_exec_verifier import (  # noqa: E402
    _extract_code,
    _fmt_grid,
    demo_fit,
    norm_hamming,
    safe_transform_from_code,
)
from experiment_4002_gap4_local_generator_arm import (  # noqa: E402
    CODEX_GATED_PASS2_REF,
    CODEX_REF,
    ORACLE_PASS2_REF,
    POOL,
    SEED,
    VOTE_PASS2_REF,
    codex_reference_cost,
    ghash,
    load_local_llama,
    resolve_local_gguf,
    score_pool,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "results" / "experiment_4012_gap4_local_best_of_n.json"
CHECKPOINT = REPO_ROOT / "results" / "experiment_4012_gap4_local_best_of_n.checkpoint.json"

INFERENCE_SUBSTRATE = "live_local_gguf_llama_cpp_best_of_n"
BASELINE_3ATTEMPT_COVERAGE = 0.2581
DEFAULT_K = 8
SNAP_TAU = 0.005

LOCAL_MODELS: dict[str, dict[str, str]] = {
    "gemma12": {"hf_id": "unsloth/gemma-4-12B-it-GGUF", "name": "gemma-4-12B"},
    "gemma26": {"hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF", "name": "gemma-4-26B-A4B"},
}
PREFERRED_MODEL_KEYS = ("gemma12", "gemma26")

REQUIRED_FIELDS = [
    "local_demo_perfect_coverage_bestofn",
    "k_samples_per_task",
    "local_gated_pass2",
    "local_beats_vote",
    "coverage_gain_vs_3attempt",
    "local_model_used",
    "cost_local_seconds",
    "cost_codex_seconds_ref",
    "cost_verifier_seconds",
    "ci95_local_minus_vote",
    "verifier_side_unchanged",
    "missing_verifier_gaps",
    "preconditions_checked",
    "random_seed",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
]

FIELD_PRINCIPLES = {
    "local_demo_perfect_coverage_bestofn": (
        "BARE FLOAT -- fraction of tasks with >=1 demo-perfect program under best-of-N "
        "(the gap-closing datum; vs <=3-attempt 0.2581 and codex 0.94)."
    ),
    "k_samples_per_task": "BARE INT -- the best-of-N draw count actually used per task.",
    "local_gated_pass2": (
        "BARE FLOAT -- the best-of-N local gated rerank pass@2 "
        "(vs vote 0.4516, oracle 0.6129, codex 0.5806)."
    ),
    "local_beats_vote": (
        "BARE BOOL -- does best-of-N local gated rerank beat vote pass@2 with a CI excluding 0."
    ),
    "coverage_gain_vs_3attempt": (
        "BARE FLOAT -- best-of-N coverage minus the <=3-attempt 0.2581 baseline."
    ),
    "local_model_used": "Which SOTA local GGUF was the inducer.",
    "cost_local_seconds": "Per-task wall-cost of the best-of-N local inducer.",
    "cost_codex_seconds_ref": "Per-task wall-cost of the codex tier reference.",
    "cost_verifier_seconds": "Per-task wall-cost of the model-free verifier.",
    "ci95_local_minus_vote": "Bootstrap 95% CI of (best-of-N local gated - vote) pass@2.",
    "verifier_side_unchanged": (
        "BARE BOOL -- the model-free verifier primitives were reused unchanged."
    ),
    "missing_verifier_gaps": "Which tasks best-of-N still could not induce.",
    "preconditions_checked": "list of {resource, available} records GGUF cache + llama_cpp checks.",
    "random_seed": "Reproducibility seed.",
    "honest_verdict": "Terminal-prefix verdict.",
    "duration_s": "Wall-clock seconds.",
    "inference_substrate": "The live local-GGUF substrate.",
}


def select_local_model(
    model_key: str = "auto",
    resolver: Callable[[str], str | None] = resolve_local_gguf,
) -> dict[str, str] | None:
    keys = PREFERRED_MODEL_KEYS if model_key == "auto" else (model_key,)
    for key in keys:
        spec = LOCAL_MODELS.get(key)
        if spec is None:
            continue
        model_path = resolver(spec["hf_id"])
        if model_path:
            return {**spec, "model_key": key, "model_path": str(model_path)}
    return None


def _pool_and_verifier_loadable(pool_path: Path | str) -> bool:
    try:
        with gzip.open(pool_path, "rt", encoding="utf-8") as handle:
            json.load(handle)
        return callable(safe_transform_from_code) and callable(score_pool)
    except Exception:
        return False


def check_preconditions(
    *,
    model_key: str,
    pool_path: Path | str,
    resolver: Callable[[str], str | None] = resolve_local_gguf,
    llama_available_override: bool | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
    chosen = select_local_model(model_key, resolver=resolver)
    if llama_available_override is None:
        try:
            import llama_cpp  # noqa: F401

            llama_ok = True
        except Exception:
            llama_ok = False
    else:
        llama_ok = bool(llama_available_override)

    pool_ok = _pool_and_verifier_loadable(pool_path)
    preconditions = [
        {
            "resource": "local_gguf_cached",
            "available": chosen is not None,
            "selected_model": chosen["name"] if chosen else None,
        },
        {"resource": "llama_cpp", "available": llama_ok},
        {"resource": "arc1_pool_and_verifier_primitives", "available": pool_ok},
    ]
    return preconditions, chosen


def blocker_from_preconditions(preconditions: list[dict[str, Any]]) -> str | None:
    by_resource = {row["resource"]: bool(row["available"]) for row in preconditions}
    if not by_resource.get("local_gguf_cached", False):
        return "blocked_local_gguf_not_cached"
    if not by_resource.get("llama_cpp", False):
        return "blocked_llama_cpp_unavailable"
    if not by_resource.get("arc1_pool_and_verifier_primitives", False):
        return "blocked_eval_pool_unreadable"
    return None


def demo_only_prompt(
    demos: list[dict[str, Any]],
    *,
    task_name: str | None = None,
    test_input: Any | None = None,
) -> str:
    _ = task_name, test_input
    lines = [
        "You are solving an ARC puzzle. The demonstration pairs below share one transformation rule.",
        "",
    ]
    for i, pair in enumerate(demos):
        inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
        lines.append(f"Demo {i + 1} INPUT ({inp.shape[0]}x{inp.shape[1]}):\n{_fmt_grid(inp)}")
        lines.append(f"Demo {i + 1} OUTPUT ({out.shape[0]}x{out.shape[1]}):\n{_fmt_grid(out)}\n")
    lines.append(
        "Write exactly one generic Python function:\n"
        "    def transform(grid):\n"
        "        # grid is a 2D numpy int array with colors 0-9; return a 2D numpy int array.\n"
        "Use only the demonstrated rule. np is already provided. Do not import or access files. "
        "Output only one ```python code block."
    )
    return "\n".join(lines)


class IndependentLocalSampler:
    SYSTEM = (
        "You are an expert Python programmer solving ARC tasks. Infer the rule from demos and "
        "output only a ```python block containing def transform(grid)."
    )

    def __init__(
        self,
        llama: Any,
        *,
        max_tokens: int = 2048,
        base_seed: int = SEED,
        base_temperature: float = 0.25,
    ) -> None:
        self._llama = llama
        self.max_tokens = max_tokens
        self.base_seed = base_seed
        self.base_temperature = base_temperature

    def __call__(self, prompt: str, draw_index: int) -> tuple[str, float]:
        temperature = round(min(0.95, self.base_temperature + 0.05 * (draw_index % 8)), 3)
        seed = self.base_seed + 1009 * draw_index
        t0 = time.time()
        try:
            out = self._llama.create_chat_completion(
                messages=[
                    {"role": "system", "content": self.SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=temperature,
                top_p=0.95,
                seed=seed,
            )
            text = out["choices"][0]["message"]["content"] or ""
        except Exception as exc:  # pragma: no cover - defensive around live model failures.
            text = f"__local_error__:{type(exc).__name__}"
        return text, round(time.time() - t0, 2)


def induce_task_samples(
    task_name: str,
    demos: list[dict[str, Any]],
    sampler: Callable[[str, int], tuple[str, float]],
    *,
    k: int,
) -> list[dict[str, Any]]:
    prompt = demo_only_prompt(demos, task_name=task_name)
    samples: list[dict[str, Any]] = []
    for draw_index in range(k):
        raw, local_s = sampler(prompt, draw_index)
        code = _extract_code(raw)
        if code is None:
            samples.append(
                {
                    "task": task_name,
                    "draw_index": draw_index,
                    "status": "no_code",
                    "demo_fit": 0.0,
                    "demo_perfect": False,
                    "local_s": local_s,
                    "code": None,
                }
            )
            continue
        fn = safe_transform_from_code(code)
        if fn is None:
            samples.append(
                {
                    "task": task_name,
                    "draw_index": draw_index,
                    "status": "unsafe_or_uncompilable",
                    "demo_fit": 0.0,
                    "demo_perfect": False,
                    "local_s": local_s,
                    "code": code,
                }
            )
            continue
        fit = demo_fit(fn, demos)
        samples.append(
            {
                "task": task_name,
                "draw_index": draw_index,
                "status": "graded",
                "demo_fit": round(fit, 4),
                "demo_perfect": bool(fit >= 1.0),
                "local_s": local_s,
                "code_len": len(code),
                "code": code,
            }
        )
    return samples


def _checkpoint_payload(
    tasks: dict[str, list[dict[str, Any]]],
    *,
    k: int,
    model_name: str,
) -> dict[str, Any]:
    return {
        "schema": "carnot.experiment_4012_gap4_local_best_of_n.checkpoint.v1",
        "k_samples_per_task": k,
        "local_model_used": model_name,
        "tasks": tasks,
    }


def _load_checkpoint(
    checkpoint_path: Path | None,
    *,
    k: int,
    model_name: str,
) -> dict[str, list[dict[str, Any]]]:
    if checkpoint_path is None or not checkpoint_path.exists():
        return {}
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - malformed checkpoint is treated as absent.
        return {}
    if payload.get("k_samples_per_task") != k or payload.get("local_model_used") != model_name:
        return {}
    tasks = payload.get("tasks")
    return tasks if isinstance(tasks, dict) else {}


def _save_checkpoint(
    checkpoint_path: Path | None,
    tasks: dict[str, list[dict[str, Any]]],
    *,
    k: int,
    model_name: str,
) -> None:
    if checkpoint_path is None:
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(_checkpoint_payload(tasks, k=k, model_name=model_name), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _entries_by_task(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_task.setdefault(entry["task"], []).append(entry)
    return by_task


def induce_pool_best_of_n(
    entries: list[dict[str, Any]],
    sampler: Callable[[str, int], tuple[str, float]],
    *,
    k: int,
    checkpoint_path: Path | None,
    model_name: str,
    started_s: float | None = None,
    max_wall_s: float | None = None,
) -> dict[str, list[dict[str, Any]]]:
    by_task = _entries_by_task(entries)
    samples_by_task = _load_checkpoint(checkpoint_path, k=k, model_name=model_name)
    for task_name in sorted(by_task):
        cached = samples_by_task.get(task_name)
        if isinstance(cached, list) and len(cached) >= k:
            samples_by_task[task_name] = cached[:k]
            continue
        if (
            started_s is not None
            and max_wall_s is not None
            and time.time() - started_s >= max_wall_s
        ):  # pragma: no cover - exercised only by live timeout runs.
            break
        samples_by_task[task_name] = induce_task_samples(
            task_name, by_task[task_name][0]["demos"], sampler, k=k
        )
        _save_checkpoint(checkpoint_path, samples_by_task, k=k, model_name=model_name)
    return samples_by_task


def _reexecute_sample_for_entry(sample: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    rec = dict(sample)
    rec["pred_hash"] = None
    rec["pred_grid"] = None
    if not sample.get("demo_perfect") or not sample.get("code"):
        return rec
    fn = safe_transform_from_code(sample["code"])
    pred = fn(entry["test_input"]) if fn is not None else None
    if pred is None:
        return rec
    rec["pred_hash"] = ghash(pred)
    rec["pred_grid"] = pred.tolist()
    return rec


def _best_candidate_snap(
    entry: dict[str, Any],
    pred_grid: Any,
    *,
    tau: float,
) -> tuple[float, int, list[list[int]], str] | None:
    options = []
    for candidate in entry["candidates"]:
        hamming = norm_hamming(candidate["grid"], pred_grid)
        if hamming <= tau:
            options.append((hamming, -int(candidate["votes"]), candidate["grid"], ghash(candidate["grid"])))
    if not options:
        return None
    hamming, neg_votes, grid, grid_hash = min(options, key=lambda item: (item[0], item[1], item[3]))
    return hamming, -neg_votes, grid, grid_hash


def _empty_program(entry: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
    best_fit = max((float(sample.get("demo_fit", 0.0)) for sample in samples), default=0.0)
    return {
        "task": entry["task"],
        "demo_fit": round(best_fit, 4),
        "demo_perfect": False,
        "pred_hash": None,
        "pred_grid": None,
        "n_calls": len(samples),
        "local_seconds": round(sum(float(sample.get("local_s", 0.0)) for sample in samples), 2),
        "history": samples,
        "code": None,
        "snap_hamming": None,
    }


def build_entry_programs(
    entries: list[dict[str, Any]],
    samples_by_task: dict[str, list[dict[str, Any]]],
    *,
    tau: float = SNAP_TAU,
) -> dict[int, dict[str, Any]]:
    prog_by_entry_id: dict[int, dict[str, Any]] = {}
    for entry in entries:
        samples = list(samples_by_task.get(entry["task"], []))
        executed = [
            _reexecute_sample_for_entry(sample, entry)
            for sample in samples
            if sample.get("demo_perfect")
        ]
        executed = [rec for rec in executed if rec.get("pred_grid") is not None]
        if not executed:
            prog_by_entry_id[id(entry)] = _empty_program(entry, samples)
            continue

        snap_options = []
        for rec in executed:
            snap = _best_candidate_snap(entry, rec["pred_grid"], tau=tau)
            if snap is None:
                continue
            hamming, votes, grid, grid_hash = snap
            snap_options.append((hamming, -votes, int(rec["draw_index"]), rec, grid, grid_hash))

        if snap_options:
            hamming, neg_votes, _draw_index, rec, grid, grid_hash = min(
                snap_options, key=lambda item: (item[0], item[1], item[2])
            )
            selected = dict(rec)
            selected["pred_grid"] = grid
            selected["pred_hash"] = grid_hash
            selected["snap_hamming"] = round(float(hamming), 6)
            selected["snap_candidate_votes"] = -neg_votes
        else:
            selected = dict(min(executed, key=lambda rec: int(rec["draw_index"])))
            selected["snap_hamming"] = None
            selected["snap_candidate_votes"] = None

        selected["n_calls"] = len(samples)
        selected["local_seconds"] = round(
            sum(float(sample.get("local_s", 0.0)) for sample in samples), 2
        )
        selected["history"] = samples
        prog_by_entry_id[id(entry)] = selected
    return prog_by_entry_id


def score_best_of_n_pool(
    entries: list[dict[str, Any]],
    prog_by_entry_id: dict[int, dict[str, Any]],
    *,
    seed: int = SEED,
) -> dict[str, Any]:
    return score_pool(entries, prog_by_entry_id, seed=seed)


def missing_verifier_gap_tasks(
    entries: list[dict[str, Any]],
    samples_by_task: dict[str, list[dict[str, Any]]],
    codex_ref_path: Path = CODEX_REF,
) -> list[str]:
    local_perfect = {
        task for task, samples in samples_by_task.items() if any(s.get("demo_perfect") for s in samples)
    }
    try:
        ref = json.loads(codex_ref_path.read_text(encoding="utf-8"))
        codex_perfect = {row["task"] for row in ref.get("per_task", []) if row.get("demo_perfect")}
    except Exception:
        codex_perfect = {entry["task"] for entry in entries}
    return sorted(codex_perfect - local_perfect)


def _fmt(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _verdict(local_beats_vote: bool, coverage: float, pass2: float, model_name: str) -> str:
    model_slug = model_name.replace("/", "_")
    if local_beats_vote:
        return (
            "success: gap4_local_bestofn_beats_vote_pass2"
            + _fmt(pass2)
            + "_cov"
            + _fmt(coverage)
            + "_inducer"
            + model_slug
        )
    return (
        "complete: gap4_local_bestofn_cov"
        + _fmt(coverage)
        + "_pass2"
        + _fmt(pass2)
        + "_below_codex"
    )


def _is_bare_float(value: Any) -> bool:
    return isinstance(value, float) and not isinstance(value, bool)


def validate_artifact(artifact: dict[str, Any]) -> None:
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:")
        or verdict.startswith("success:")
        or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must use a terminal prefix")
    if not isinstance(artifact["k_samples_per_task"], int) or isinstance(
        artifact["k_samples_per_task"], bool
    ):
        raise ValueError("k_samples_per_task must be a bare int")
    for field in ("local_beats_vote", "verifier_side_unchanged"):
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare bool")
    for field in (
        "local_demo_perfect_coverage_bestofn",
        "local_gated_pass2",
        "coverage_gain_vs_3attempt",
        "cost_local_seconds",
        "cost_codex_seconds_ref",
        "cost_verifier_seconds",
        "duration_s",
    ):
        if not _is_bare_float(artifact[field]):
            raise ValueError(f"{field} must be a bare float")
    if not isinstance(artifact["random_seed"], int) or isinstance(artifact["random_seed"], bool):
        raise ValueError("random_seed must be a bare int")
    if not (isinstance(artifact["ci95_local_minus_vote"], list) and len(artifact["ci95_local_minus_vote"]) == 2):
        raise ValueError("ci95_local_minus_vote must be a 2-element list")
    for field in ("local_model_used", "honest_verdict", "inference_substrate"):
        if not isinstance(artifact[field], str):
            raise ValueError(f"{field} must be a string")
    for field in ("missing_verifier_gaps", "preconditions_checked"):
        if not isinstance(artifact[field], list):
            raise ValueError(f"{field} must be a list")


def blocked_artifact(
    verdict: str,
    chosen_model: dict[str, str] | None,
    preconditions: list[dict[str, Any]],
    duration_s: float,
    *,
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    artifact = {
        "experiment": "experiment_4012_gap4_local_best_of_n",
        "schema": "carnot.experiment_4012_gap4_local_best_of_n.v1",
        "title": "GAP-4 local best-of-N open-weight generator arm",
        "local_demo_perfect_coverage_bestofn": 0.0,
        "k_samples_per_task": k,
        "local_gated_pass2": 0.0,
        "local_beats_vote": False,
        "coverage_gain_vs_3attempt": round(0.0 - BASELINE_3ATTEMPT_COVERAGE, 4),
        "local_model_used": chosen_model["name"] if chosen_model else "none",
        "cost_local_seconds": 0.0,
        "cost_codex_seconds_ref": codex_reference_cost(),
        "cost_verifier_seconds": 0.0,
        "ci95_local_minus_vote": [0.0, 0.0],
        "verifier_side_unchanged": True,
        "missing_verifier_gaps": [],
        "preconditions_checked": preconditions,
        "random_seed": SEED,
        "honest_verdict": verdict,
        "duration_s": round(duration_s, 2),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
    }
    validate_artifact(artifact)
    return artifact


def _sample_summary(samples_by_task: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    summary = []
    for task, samples in sorted(samples_by_task.items()):
        summary.append(
            {
                "task": task,
                "n_samples": len(samples),
                "n_demo_perfect": sum(1 for sample in samples if sample.get("demo_perfect")),
                "best_demo_fit": round(
                    max((float(sample.get("demo_fit", 0.0)) for sample in samples), default=0.0),
                    4,
                ),
                "local_seconds": round(sum(float(sample.get("local_s", 0.0)) for sample in samples), 2),
                "statuses": [sample.get("status") for sample in samples],
            }
        )
    return summary


def build_complete_artifact(
    *,
    entries: list[dict[str, Any]],
    samples_by_task: dict[str, list[dict[str, Any]]],
    prog_by_entry_id: dict[int, dict[str, Any]],
    scored: dict[str, Any],
    chosen_model: dict[str, str],
    preconditions: list[dict[str, Any]],
    verifier_seconds: float,
    started_s: float,
    now_s: float,
    k: int,
    codex_ref_path: Path,
) -> dict[str, Any]:
    coverage = round(scored["n_perfect"] / max(1, scored["n"]), 4)
    coverage_gain = round(coverage - BASELINE_3ATTEMPT_COVERAGE, 4)
    ci = scored["ci95_gated_vs_vote"]
    local_beats_vote = bool(scored["g2"] > scored["vote2"] and ci[0] > 0.0)
    total_local_s = sum(
        float(sample.get("local_s", 0.0))
        for samples in samples_by_task.values()
        for sample in samples
    )
    n_unique = len({entry["task"] for entry in entries})
    checksum_blob = json.dumps(
        {
            task: [(sample.get("draw_index"), sample.get("code") or "") for sample in samples]
            for task, samples in sorted(samples_by_task.items())
        },
        sort_keys=True,
    )
    artifact = {
        "experiment": "experiment_4012_gap4_local_best_of_n",
        "schema": "carnot.experiment_4012_gap4_local_best_of_n.v1",
        "title": "GAP-4 local best-of-N open-weight generator arm",
        "local_demo_perfect_coverage_bestofn": coverage,
        "k_samples_per_task": k,
        "local_gated_pass2": scored["g2"],
        "local_beats_vote": local_beats_vote,
        "coverage_gain_vs_3attempt": coverage_gain,
        "local_model_used": chosen_model["name"],
        "cost_local_seconds": round(total_local_s / max(1, len(samples_by_task)), 2),
        "cost_codex_seconds_ref": codex_reference_cost(codex_ref_path),
        "cost_verifier_seconds": round(verifier_seconds / max(1, scored["n"]), 4),
        "ci95_local_minus_vote": ci,
        "verifier_side_unchanged": True,
        "missing_verifier_gaps": missing_verifier_gap_tasks(entries, samples_by_task, codex_ref_path),
        "preconditions_checked": preconditions,
        "random_seed": SEED,
        "honest_verdict": _verdict(local_beats_vote, coverage, scored["g2"], chosen_model["name"]),
        "duration_s": round(now_s - started_s, 2),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "n_entries": scored["n"],
        "n_unique_tasks": n_unique,
        "n_local_demo_perfect": scored["n_perfect"],
        "total_local_samples": sum(len(samples) for samples in samples_by_task.values()),
        "total_local_seconds": round(total_local_s, 2),
        "references": {
            "vote_pass2": VOTE_PASS2_REF,
            "oracle_pass2": ORACLE_PASS2_REF,
            "codex_gated_pass2": CODEX_GATED_PASS2_REF,
            "local_3attempt_demo_perfect_coverage": BASELINE_3ATTEMPT_COVERAGE,
        },
        "rankers": scored["rankers"],
        "gates": scored["gates"],
        "headroom_recovered_tasks": scored["headroom_recovered"],
        "vote_wins_lost_tasks": scored["vote_wins_lost"],
        "per_task": scored["per_task"],
        "per_task_sample_summary": _sample_summary(samples_by_task),
        "model_specs": {
            "generator_model": chosen_model["name"],
            "generator_hf_id": chosen_model["hf_id"],
            "generator_gguf_path": chosen_model["model_path"],
            "verifier": (
                "model-free: demo-fit exact-reproduction gate + restricted-namespace execution + "
                "min-hamming snap + content-hash candidate match"
            ),
        },
        "selected_programs_checksum": hashlib.sha256(checksum_blob.encode()).hexdigest()[:16],
        "field_principles": FIELD_PRINCIPLES,
    }
    validate_artifact(artifact)
    return artifact


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(
    *,
    model_key: str = "auto",
    pool_path: Path | str = POOL,
    output_path: Path = OUTPUT,
    codex_ref_path: Path = CODEX_REF,
    checkpoint_path: Path | None = CHECKPOINT,
    k: int = DEFAULT_K,
    limit: int = 0,
    n_ctx: int = 16384,
    max_wall_s: float = 7200.0,
    sampler: Callable[[str, int], tuple[str, float]] | None = None,
    resolver: Callable[[str], str | None] = resolve_local_gguf,
    llama_available_override: bool | None = None,
    write: bool = True,
) -> dict[str, Any]:
    started = time.time()
    preconditions, chosen_model = check_preconditions(
        model_key=model_key,
        pool_path=pool_path,
        resolver=resolver,
        llama_available_override=llama_available_override,
    )
    blocker = blocker_from_preconditions(preconditions)
    if blocker:
        artifact = blocked_artifact(blocker, chosen_model, preconditions, time.time() - started, k=k)
        if write:
            _write_json(output_path, artifact)
        print(f"-> {artifact['honest_verdict']}", flush=True)
        return artifact

    with gzip.open(pool_path, "rt", encoding="utf-8") as handle:
        pool = json.load(handle)
    entries = pool["entries"]
    if limit:
        entries = entries[:limit]

    if sampler is None:  # pragma: no cover - loads the live multi-GB model.
        llama = load_local_llama(chosen_model["model_path"], n_ctx=n_ctx, seed=SEED)
        sampler = IndependentLocalSampler(llama, base_seed=SEED)

    print(
        f"[exp4012] LOCAL inducer={chosen_model['name']} k={k} over {len(entries)} entries "
        f"({len({entry['task'] for entry in entries})} unique tasks)",
        flush=True,
    )
    samples_by_task = induce_pool_best_of_n(
        entries,
        sampler,
        k=k,
        checkpoint_path=checkpoint_path,
        model_name=chosen_model["name"],
        started_s=started,
        max_wall_s=max_wall_s,
    )
    prog_by_entry_id = build_entry_programs(entries, samples_by_task, tau=SNAP_TAU)

    verifier_t0 = time.time()
    scored = score_best_of_n_pool(entries, prog_by_entry_id, seed=SEED)
    verifier_seconds = time.time() - verifier_t0

    artifact = build_complete_artifact(
        entries=entries,
        samples_by_task=samples_by_task,
        prog_by_entry_id=prog_by_entry_id,
        scored=scored,
        chosen_model=chosen_model,
        preconditions=preconditions,
        verifier_seconds=verifier_seconds,
        started_s=started,
        now_s=time.time(),
        k=k,
        codex_ref_path=codex_ref_path,
    )
    if write:
        _write_json(output_path, artifact)
    print(f"-> {artifact['honest_verdict']}", flush=True)
    print(
        f"   coverage={artifact['local_demo_perfect_coverage_bestofn']} "
        f"gain={artifact['coverage_gain_vs_3attempt']} "
        f"gated_pass2={artifact['local_gated_pass2']} "
        f"beats_vote={artifact['local_beats_vote']} CI={artifact['ci95_local_minus_vote']}",
        flush=True,
    )
    return artifact


def main() -> None:  # pragma: no cover - exercised by the required script command.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["auto", *LOCAL_MODELS], default="auto")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--n-ctx", type=int, default=16384)
    parser.add_argument("--max-wall-s", type=float, default=7200.0)
    args = parser.parse_args()
    if args.k < 1 or args.k > 16:
        raise SystemExit("--k must be between 1 and 16")
    run(model_key=args.model, k=args.k, limit=args.limit, n_ctx=args.n_ctx, max_wall_s=args.max_wall_s)


if __name__ == "__main__":  # pragma: no cover
    main()
