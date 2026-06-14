#!/usr/bin/env python3
"""Exp 4188: sovereign local GAP-4 generator plus self-distillation corpus.

This runner measures the local open-weight generator side of GAP-4.  It checks
that the mandated SOTA GGUF pair is cached, replays or induces local GGUF
`def transform(grid)` samples from demos only, re-verifies demo perfection with
the mechanical GAP-4 execution verifier, reranks with the hardened Exp 4187
graded gate, and banks verifier-labeled demo-perfect programs for later
distillation.  It never calls Codex or any closed-weight generator.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
EXPERIMENTS_DIR = ROOT / "scripts" / "experiments"
for _path in (PYTHON_DIR, EXPERIMENTS_DIR, ROOT):
    if str(_path) not in sys.path:  # pragma: no cover - import-path bootstrap.
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


ARTIFACT_PATH = ROOT / "results" / "experiment_4188_sovereign_local_generator_gap4_self_distill.json"
CORPUS_PATH = ROOT / "results" / "experiment_4188_sovereign_local_generator_gap4_self_distill_corpus.jsonl"
POOL_PATH = ROOT / "results" / "arc3_gap3_stage2_eval_pool.json.gz"
CHECKPOINT_PATH = ROOT / "results" / "decentralization_moe_qwen35a3b_arc1_k8.checkpoint.json"
HARDENED_GATE_PATH = ROOT / "results" / "experiment_4187_gap4_graded_execution_gate_hardening.json"
PRIOR_NULL_PATH = ROOT / "results" / "experiment_4069_decentralization_moe_sync.json"

RANDOM_SEED = 12345
DEFAULT_K = 8
CODEX_DEMO_PERFECT = 29
CODEX_TOTAL = 31
CODEX_HARDENED_PASS2 = 0.5806
EXEC_TIMEOUT_S = 5.0

_FORBIDDEN = (
    "__import__",
    "open(",
    "eval(",
    "exec(",
    "compile(",
    "subprocess",
    "os.",
    "sys.",
    "import os",
    "import sys",
    "import subprocess",
    "socket",
    "shutil",
    "Path(",
    "getattr(",
    "setattr(",
    "globals(",
    "locals(",
    "type(",
    "np.load",
    "np.save",
    "np.fromfile",
    "np.memmap",
    ".tofile",
)

_SAFE_BUILTIN_NAMES = [
    "range",
    "len",
    "min",
    "max",
    "abs",
    "enumerate",
    "zip",
    "sum",
    "sorted",
    "list",
    "dict",
    "set",
    "tuple",
    "int",
    "float",
    "bool",
    "map",
    "filter",
    "reversed",
    "any",
    "all",
    "round",
    "isinstance",
    "str",
]

REQUIRED_FIELDS = [
    "honest_verdict",
    "local_induction_rate",
    "sovereign_pool_pass2",
    "self_distillation_corpus_size",
    "no_closed_weight_call",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "preconditions_checked",
    "field_principles",
    "duration_s",
    "inference_substrate",
]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest 'local generator under-induces vs codex' is a "
        "COMPLETE, decision-grade verdict (it bounds the sovereignty claim)."
    ),
    "local_induction_rate": (
        "demo-perfect / total with a LOCAL generator vs the codex 29/31 -- the "
        "decentralization-rule-1 measurement."
    ),
    "sovereign_pool_pass2": (
        "pool-rerank pass@2 with the local generator + hardened gate vs TRM vote -- "
        "does a fully-sovereign stack recover ARC headroom?"
    ),
    "self_distillation_corpus_size": (
        "Count of verifier-labeled demo-perfect programs banked for distillation -- "
        "the FR-11 self-learning loop where the energy function is the ground-truth reward."
    ),
    "no_closed_weight_call": (
        "Bare bool: zero codex/closed-weight calls were made (decentralization rule 1 -- "
        "Carnot works end-to-end on local open models)."
    ),
    "model_specs": (
        "The local GGUF generator invoked; required methodology for a live-LLM artifact."
    ),
    "random_seed": "Determinism precondition; the induction + verification must be reproducible.",
    "reproducibility_checksum": (
        "Hash of the task subset + generator config; catches silent drift."
    ),
}


def _numpy_only_import(name: str, *args: Any, **kwargs: Any) -> Any:
    if name == "numpy" or name.startswith("numpy."):
        return __import__(name, *args, **kwargs)
    raise ImportError(f"import of {name!r} is blocked in the GAP-4 sandbox")


def _safe_builtins() -> dict[str, Any]:
    import builtins as _builtins

    out = {name: getattr(_builtins, name) for name in _SAFE_BUILTIN_NAMES}
    out["__import__"] = _numpy_only_import
    return out


def safe_transform_from_code(code: str) -> Callable[[Any], np.ndarray | None] | None:
    for token in _FORBIDDEN:
        if re.search(r"(?<![A-Za-z0-9_])" + re.escape(token), code):
            return None
    body = "\n".join(
        line for line in code.splitlines() if not line.strip().startswith(("import ", "from "))
    )
    namespace = {"np": np, "numpy": np, "__builtins__": _safe_builtins()}
    try:
        exec(body, namespace)
    except Exception:
        return None
    fn = namespace.get("transform")
    if not callable(fn):
        return None

    def _call(grid: Any) -> np.ndarray | None:
        try:
            out = np.asarray(fn(np.asarray(grid, dtype=np.int64).copy()), dtype=np.int64)
        except Exception:
            return None
        if out.ndim != 2 or out.size == 0 or out.shape[0] > 30 or out.shape[1] > 30:
            return None
        if out.min() < 0 or out.max() > 9:
            return None
        return out

    def wrapped(grid: Any) -> np.ndarray | None:
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            return executor.submit(_call, grid).result(timeout=EXEC_TIMEOUT_S)
        except TimeoutError:  # pragma: no cover - defensive around live generated loops.
            return None
        finally:
            executor.shutdown(wait=False)

    return wrapped


def demo_fit(fn: Callable[[Any], np.ndarray | None], demos: list[dict[str, Any]]) -> float:
    hits = 0
    for pair in demos:
        out = fn(pair["input"])
        if out is not None and np.array_equal(out, np.asarray(pair["output"], dtype=np.int64)):
            hits += 1
    return hits / max(1, len(demos))


def _default_cached_sota_pair() -> list[dict[str, Any]] | None:  # pragma: no cover
    from scripts.experiment_template import cached_sota_pair  # noqa: PLC0415

    return cached_sota_pair()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_pool(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("pool artifact missing entries list")
    return entries


def _load_checkpoint(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("checkpoint artifact missing tasks map")
    return payload


def _gguf_pair_available(specs: list[dict[str, Any]] | None) -> bool:
    return bool(
        isinstance(specs, list)
        and len(specs) >= 2
        and all(str(spec.get("model_path", "")).endswith(".gguf") for spec in specs[:2])
    )


def _select_generator_spec(
    specs: list[dict[str, Any]],
    checkpoint_model: str | None = None,
) -> dict[str, Any]:
    if checkpoint_model:
        for spec in specs:
            if spec.get("name") == checkpoint_model:
                return dict(spec)
    for spec in specs:
        if spec.get("hf_id") == "unsloth/Qwen3.6-35B-A3B-GGUF":
            return dict(spec)
    return dict(specs[0])


def _check_preconditions(
    *,
    specs: list[dict[str, Any]] | None,
    pool_path: Path,
    checkpoint_path: Path,
    hardened_gate_path: Path,
) -> list[dict[str, Any]]:
    return [
        {
            "resource": "sota_gguf_pair_cached",
            "available": _gguf_pair_available(specs),
            "models": [
                {
                    "name": spec.get("name"),
                    "hf_id": spec.get("hf_id"),
                    "model_path": spec.get("model_path"),
                }
                for spec in (specs or [])[:2]
            ],
        },
        {"resource": "arc1_candidate_pool", "available": pool_path.exists(), "path": str(pool_path)},
        {
            "resource": "local_gguf_samples_checkpoint",
            "available": checkpoint_path.exists(),
            "path": str(checkpoint_path),
        },
        {
            "resource": "hardened_gap4_gate_artifact",
            "available": hardened_gate_path.exists(),
            "path": str(hardened_gate_path),
        },
    ]


def _blocker(preconditions: list[dict[str, Any]]) -> str | None:
    by_resource = {row["resource"]: bool(row["available"]) for row in preconditions}
    if not by_resource.get("sota_gguf_pair_cached", False):
        return "blocked_model_not_cached_sota_gguf"
    if not by_resource.get("arc1_candidate_pool", False):
        return "blocked_arc1_candidate_pool_missing"
    if not by_resource.get("hardened_gap4_gate_artifact", False):
        return "blocked_hardened_gap4_gate_missing"
    return None


def blocked_artifact(
    *,
    verdict: str,
    preconditions: list[dict[str, Any]],
    duration_s: float,
    model_specs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = {
        "experiment": "experiment_4188_sovereign_local_generator_gap4_self_distill",
        "schema": "carnot.experiment_4188_sovereign_local_generator_gap4_self_distill.v1",
        "title": "Sovereign local GAP-4 generator with hardened execution gate",
        "honest_verdict": verdict,
        "local_induction_rate": {
            "demo_perfect": 0,
            "total": 0,
            "rate": 0.0,
            "codex_reference": {
                "demo_perfect": CODEX_DEMO_PERFECT,
                "total": CODEX_TOTAL,
                "rate": round(CODEX_DEMO_PERFECT / CODEX_TOTAL, 4),
            },
        },
        "sovereign_pool_pass2": {
            "TRM_VOTE": 0.0,
            "LOCAL_HARDENED_GATE": 0.0,
            "delta_vs_vote": 0.0,
            "codex_hardened_reference": CODEX_HARDENED_PASS2,
            "recovered": 0,
            "lost": 0,
        },
        "self_distillation_corpus_size": 0,
        "no_closed_weight_call": True,
        "model_specs": model_specs or {},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "preconditions_checked": preconditions,
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(duration_s, 6),
        "inference_substrate": "local_gguf_precondition_blocked",
    }
    validate_artifact(artifact)
    return artifact


def _sample_draw_index(sample: dict[str, Any]) -> int:
    value = sample.get("draw_index", 0)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _to_grid_list(value: Any) -> list[list[int]] | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim != 2 or arr.size == 0:
        return None
    return [[int(cell) for cell in row] for row in arr.tolist()]


def _verified_records_for_task(
    task: str,
    demos: list[dict[str, Any]],
    samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    records: list[dict[str, Any]] = []
    best_fit = 0.0
    seen_codes: set[str] = set()
    for sample in sorted(samples, key=_sample_draw_index):
        code = sample.get("code")
        if not isinstance(code, str) or not code.strip():
            continue
        fn = safe_transform_from_code(code)
        if fn is None:
            continue
        fit = float(demo_fit(fn, demos))
        best_fit = max(best_fit, fit)
        if fit < 1.0:
            continue
        clean_code = code.strip()
        if clean_code in seen_codes:
            continue
        seen_codes.add(clean_code)
        records.append(
            {
                "task": task,
                "draw_index": _sample_draw_index(sample),
                "demo_fit": 1.0,
                "code": clean_code,
                "local_s": float(sample.get("local_s", 0.0) or 0.0),
                "_fn": fn,
            }
        )
    return records, best_fit


def build_verified_programs(
    entries: list[dict[str, Any]],
    samples_by_task: dict[str, list[dict[str, Any]]],
    *,
    generator_spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_task: dict[str, dict[str, Any]] = {}
    for entry in entries:
        by_task.setdefault(entry["task"], entry)

    verified_by_task: dict[str, list[dict[str, Any]]] = {}
    best_fit_by_task: dict[str, float] = {}
    corpus: list[dict[str, Any]] = []
    for task in sorted(by_task):
        entry = by_task[task]
        samples = samples_by_task.get(task, [])
        verified, best_fit = _verified_records_for_task(task, entry["demos"], samples)
        verified_by_task[task] = verified
        best_fit_by_task[task] = best_fit
        for rec in verified:
            corpus.append(
                {
                    "task": task,
                    "demo_pairs": entry["demos"],
                    "program": rec["code"],
                    "verifier_label": "demo_perfect",
                    "demo_fit": rec["demo_fit"],
                    "source_draw_index": rec["draw_index"],
                    "generator_model": generator_spec.get("name"),
                    "generator_hf_id": generator_spec.get("hf_id"),
                    "reward_energy": "execution_consistency_demo_exact",
                }
            )

    programs: list[dict[str, Any]] = []
    for entry in entries:
        verified = verified_by_task.get(entry["task"], [])
        if not verified:
            programs.append(
                {
                    "task": entry["task"],
                    "demo_fit": round(best_fit_by_task.get(entry["task"], 0.0), 4),
                    "demo_perfect": False,
                    "pred_grid": None,
                    "code": None,
                    "source_draw_index": None,
                }
            )
            continue
        selected = verified[0]
        pred = selected["_fn"](entry["test_input"])
        programs.append(
            {
                "task": entry["task"],
                "demo_fit": 1.0,
                "demo_perfect": True,
                "pred_grid": _to_grid_list(pred),
                "code": selected["code"],
                "source_draw_index": selected["draw_index"],
            }
        )
    return programs, corpus


def score_hardened_gate(
    entries: list[dict[str, Any]],
    programs: list[dict[str, Any]],
    *,
    tau: float = DEFAULT_TAU,
    high_vote_guard_threshold: int | float | None = DEFAULT_HIGH_VOTE_GUARD_THRESHOLD,
) -> dict[str, Any]:
    if len(entries) != len(programs):
        raise ValueError("entries/programs length mismatch")
    vote_rankings = [vote_rank_indices(entry["candidates"]) for entry in entries]
    gated_rankings: list[list[int]] = []
    selections: list[dict[str, Any]] = []
    for entry, program in zip(entries, programs, strict=True):
        if entry["task"] != program["task"]:
            raise ValueError(f"task mismatch: {entry['task']} != {program['task']}")
        selection = select_guarded_graded_candidate(
            entry["candidates"],
            prediction=program.get("pred_grid"),
            demo_fit=program.get("demo_fit"),
            task_id=entry["task"],
            tau=tau,
            high_vote_guard_threshold=high_vote_guard_threshold,
        )
        selections.append(selection)
        gated_rankings.append(gated_rank_indices(entry["candidates"], selection["selected_index"]))

    vote_hits = hit_indices(entries, vote_rankings, 2)
    gated_hits = hit_indices(entries, gated_rankings, 2)
    pass2_vote = pass_at_k(entries, vote_rankings, 2)
    pass2_gate = pass_at_k(entries, gated_rankings, 2)
    return {
        "pass_at_1": {
            "TRM_VOTE": pass_at_k(entries, vote_rankings, 1),
            "LOCAL_HARDENED_GATE": pass_at_k(entries, gated_rankings, 1),
        },
        "pass_at_2": {
            "TRM_VOTE": pass2_vote,
            "LOCAL_HARDENED_GATE": pass2_gate,
        },
        "recovered": len(gated_hits - vote_hits),
        "lost": len(vote_hits - gated_hits),
        "gate_fire_count": sum(1 for selection in selections if selection["gate_fired"]),
        "guard_block_count": sum(1 for selection in selections if selection["guard_blocked"]),
        "selection_details": selections,
    }


def _write_corpus(path: Path, corpus: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in corpus)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def _file_sha(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reproducibility_checksum(
    *,
    entries: list[dict[str, Any]],
    generator_spec: dict[str, Any],
    checkpoint_path: Path,
    seed: int,
) -> str:
    task_subset = [
        {
            "task": entry["task"],
            "n_candidates": len(entry.get("candidates", [])),
            "n_demos": len(entry.get("demos", [])),
        }
        for entry in entries
    ]
    blob = json.dumps(
        {
            "checkpoint_sha256": _file_sha(checkpoint_path),
            "generator_hf_id": generator_spec.get("hf_id"),
            "generator_model_path": generator_spec.get("model_path"),
            "seed": seed,
            "task_subset": task_subset,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _verdict(rate: float, pass2_gate: float, pass2_vote: float, corpus_size: int) -> str:
    if pass2_gate > pass2_vote:
        return (
            "success: sovereign_local_gap4_recovers_headroom_"
            f"pass2{pass2_gate}_cov{rate}_corpus{corpus_size}"
        )
    return (
        "complete: local_generator_underinduces_vs_codex_"
        f"cov{rate}_pass2{pass2_gate}_corpus{corpus_size}"
    )


def _complete_artifact(
    *,
    entries: list[dict[str, Any]],
    programs: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    scored: dict[str, Any],
    generator_spec: dict[str, Any],
    preconditions: list[dict[str, Any]],
    checkpoint_payload: dict[str, Any],
    checkpoint_path: Path,
    corpus_path: Path,
    duration_s: float,
) -> dict[str, Any]:
    total = len(entries)
    n_demo_perfect = sum(1 for program in programs if program.get("demo_perfect"))
    rate = round(n_demo_perfect / max(1, total), 4)
    pass2_vote = scored["pass_at_2"]["TRM_VOTE"]
    pass2_gate = scored["pass_at_2"]["LOCAL_HARDENED_GATE"]
    artifact = {
        "experiment": "experiment_4188_sovereign_local_generator_gap4_self_distill",
        "schema": "carnot.experiment_4188_sovereign_local_generator_gap4_self_distill.v1",
        "title": "Sovereign local GAP-4 generator with hardened execution gate",
        "honest_verdict": _verdict(rate, pass2_gate, pass2_vote, len(corpus)),
        "local_induction_rate": {
            "demo_perfect": n_demo_perfect,
            "total": total,
            "rate": rate,
            "codex_reference": {
                "demo_perfect": CODEX_DEMO_PERFECT,
                "total": CODEX_TOTAL,
                "rate": round(CODEX_DEMO_PERFECT / CODEX_TOTAL, 4),
            },
        },
        "sovereign_pool_pass2": {
            "TRM_VOTE": pass2_vote,
            "LOCAL_HARDENED_GATE": pass2_gate,
            "delta_vs_vote": round(pass2_gate - pass2_vote, 4),
            "codex_hardened_reference": CODEX_HARDENED_PASS2,
            "recovered": scored["recovered"],
            "lost": scored["lost"],
            "pass_at_1": scored["pass_at_1"],
            "gate_fire_count": scored["gate_fire_count"],
            "guard_block_count": scored["guard_block_count"],
        },
        "self_distillation_corpus_size": len(corpus),
        "self_distillation_corpus_path": str(corpus_path),
        "no_closed_weight_call": True,
        "model_specs": {
            "generator_model": generator_spec.get("name"),
            "generator_hf_id": generator_spec.get("hf_id"),
            "generator_gguf_path": generator_spec.get("model_path"),
            "generator_source": "local_gguf_checkpoint_replay",
            "source_checkpoint_path": str(checkpoint_path),
            "source_checkpoint_schema": checkpoint_payload.get("schema"),
            "source_checkpoint_k_samples_per_task": checkpoint_payload.get("k_samples_per_task"),
            "hardened_gate": "experiment_4187_gap4_graded_execution_gate_hardening",
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _reproducibility_checksum(
            entries=entries,
            generator_spec=generator_spec,
            checkpoint_path=checkpoint_path,
            seed=RANDOM_SEED,
        ),
        "reproducibility_checksum_sources": [str(checkpoint_path), str(POOL_PATH)],
        "preconditions_checked": preconditions,
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(duration_s, 6),
        "inference_substrate": "local_gguf_checkpoint_replay_plus_deterministic_hardened_gate",
        "n_tasks": total,
        "n_unique_tasks": len({entry["task"] for entry in entries}),
        "programs_reverified": [
            {
                "task": program["task"],
                "demo_perfect": bool(program["demo_perfect"]),
                "demo_fit": program["demo_fit"],
                "has_prediction": program["pred_grid"] is not None,
            }
            for program in programs
        ],
        "prior_local_moe_null": str(PRIOR_NULL_PATH),
    }
    validate_artifact(artifact)
    return artifact


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
    if not isinstance(artifact["no_closed_weight_call"], bool):
        raise ValueError("no_closed_weight_call must be a bare bool")
    if artifact["no_closed_weight_call"] is not True:
        raise ValueError("no_closed_weight_call must be true")
    if not isinstance(artifact["self_distillation_corpus_size"], int) or isinstance(
        artifact["self_distillation_corpus_size"], bool
    ):
        raise ValueError("self_distillation_corpus_size must be a bare int")
    if not isinstance(artifact["random_seed"], int) or isinstance(artifact["random_seed"], bool):
        raise ValueError("random_seed must be a bare int")
    for field in ("local_induction_rate", "sovereign_pool_pass2", "model_specs", "field_principles"):
        if not isinstance(artifact[field], dict):
            raise ValueError(f"{field} must be a dict")
    if not isinstance(artifact["preconditions_checked"], list):
        raise ValueError("preconditions_checked must be a list")
    if not isinstance(artifact["reproducibility_checksum"], str):
        raise ValueError("reproducibility_checksum must be a string")
    if not isinstance(artifact["duration_s"], float):
        raise ValueError("duration_s must be a bare float")
    if not isinstance(artifact["inference_substrate"], str):
        raise ValueError("inference_substrate must be a string")


def _live_samples_if_checkpoint_missing(  # pragma: no cover
    entries: list[dict[str, Any]],
    generator_spec: dict[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:  # pragma: no cover - loads a live multi-GB GGUF only when no checkpoint exists.
    from experiment_4012_gap4_local_best_of_n import (  # noqa: PLC0415
        IndependentLocalSampler,
        induce_pool_best_of_n,
        load_local_llama,
    )

    llama = load_local_llama(str(generator_spec["model_path"]), seed=RANDOM_SEED)
    sampler = IndependentLocalSampler(llama, base_seed=RANDOM_SEED)
    tasks = induce_pool_best_of_n(
        entries,
        sampler,
        k=DEFAULT_K,
        checkpoint_path=checkpoint_path,
        model_name=str(generator_spec["name"]),
    )
    return {
        "schema": "carnot.experiment_4188.live_generated_checkpoint.v1",
        "k_samples_per_task": DEFAULT_K,
        "local_model_used": generator_spec["name"],
        "tasks": tasks,
    }


def run(
    *,
    artifact_path: Path = ARTIFACT_PATH,
    corpus_path: Path = CORPUS_PATH,
    pool_path: Path = POOL_PATH,
    checkpoint_path: Path = CHECKPOINT_PATH,
    hardened_gate_path: Path = HARDENED_GATE_PATH,
    cached_pair_fn: Callable[[], list[dict[str, Any]] | None] = _default_cached_sota_pair,
) -> dict[str, Any]:
    started = time.time()
    artifact_path = Path(artifact_path)
    corpus_path = Path(corpus_path)
    pool_path = Path(pool_path)
    checkpoint_path = Path(checkpoint_path)
    hardened_gate_path = Path(hardened_gate_path)
    specs = cached_pair_fn()
    preconditions = _check_preconditions(
        specs=specs,
        pool_path=pool_path,
        checkpoint_path=checkpoint_path,
        hardened_gate_path=hardened_gate_path,
    )
    blocker = _blocker(preconditions)
    if blocker:
        artifact = blocked_artifact(
            verdict=blocker,
            preconditions=preconditions,
            duration_s=time.time() - started,
        )
        _write_json(artifact_path, artifact)
        print(f"-> {artifact['honest_verdict']}", flush=True)
        return artifact

    checkpoint_payload = _load_checkpoint(checkpoint_path) if checkpoint_path.exists() else {}
    generator_spec = _select_generator_spec(
        specs or [],
        checkpoint_model=checkpoint_payload.get("local_model_used"),
    )
    entries = _load_pool(pool_path)
    if checkpoint_payload:
        samples_by_task = checkpoint_payload["tasks"]
    else:  # pragma: no cover - live GGUF fallback when no local checkpoint exists.
        checkpoint_payload = _live_samples_if_checkpoint_missing(
            entries,
            generator_spec,
            checkpoint_path,
        )
        samples_by_task = checkpoint_payload["tasks"]

    programs, corpus = build_verified_programs(
        entries,
        samples_by_task,
        generator_spec=generator_spec,
    )
    scored = score_hardened_gate(entries, programs)
    _write_corpus(corpus_path, corpus)
    artifact = _complete_artifact(
        entries=entries,
        programs=programs,
        corpus=corpus,
        scored=scored,
        generator_spec=generator_spec,
        preconditions=preconditions,
        checkpoint_payload=checkpoint_payload,
        checkpoint_path=checkpoint_path,
        corpus_path=corpus_path,
        duration_s=time.time() - started,
    )
    _write_json(artifact_path, artifact)
    print(f"-> {artifact['honest_verdict']}", flush=True)
    print(
        "   local_induction="
        f"{artifact['local_induction_rate']['demo_perfect']}/"
        f"{artifact['local_induction_rate']['total']} "
        f"pass2={artifact['sovereign_pool_pass2']['LOCAL_HARDENED_GATE']} "
        f"vote={artifact['sovereign_pool_pass2']['TRM_VOTE']} "
        f"corpus={artifact['self_distillation_corpus_size']}",
        flush=True,
    )
    return artifact


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--pool", type=Path, default=POOL_PATH)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--hardened-gate", type=Path, default=HARDENED_GATE_PATH)
    args = parser.parse_args(argv)
    run(
        artifact_path=args.artifact,
        corpus_path=args.corpus,
        pool_path=args.pool,
        checkpoint_path=args.checkpoint,
        hardened_gate_path=args.hardened_gate,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
