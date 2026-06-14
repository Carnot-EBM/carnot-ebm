#!/usr/bin/env python3
"""Exp 4200: GAP-4 certified ARC corpus plus local in-context lift read.

Spec refs: REQ-VERIFY-4200, SCENARIO-VERIFY-4200.

This runner is intentionally artifact-first.  It replays the cached Codex
ARC-1 induced programs through the hardened GAP-4 guarded graded gate to build
the verifier-as-reward corpus.  It then compares a cold local GGUF checkpoint
against an optional seeded local checkpoint.  When no seeded checkpoint exists,
the runner reports a conservative flat/no-lift replay instead of fabricating
fresh in-context generations or fine-tuning the GGUF.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
for _path in (PYTHON_DIR, ROOT):
    if str(_path) not in sys.path:  # pragma: no cover - import-path bootstrap.
        sys.path.insert(0, str(_path))

from carnot.agentic.gap4_graded_execution_gate import (  # noqa: E402
    DEFAULT_HIGH_VOTE_GUARD_THRESHOLD,
    DEFAULT_TAU,
    select_guarded_graded_candidate,
)


ARTIFACT_PATH = ROOT / "results" / "experiment_4200_certified_arc_corpus_distill_lift.json"
CORPUS_PATH = ROOT / "results" / "experiment_4200_certified_arc_corpus_distill_lift_corpus.jsonl"
POOL_PATH = ROOT / "results" / "arc3_gap3_stage2_eval_pool.json.gz"
PROGRAMS_PATH = ROOT / "results" / "arc3_gap4_induced_programs.json"
COLD_CHECKPOINT_PATH = ROOT / "results" / "decentralization_moe_qwen35a3b_arc1_k8.checkpoint.json"
SEEDED_CHECKPOINT_PATH = (
    ROOT / "results" / "experiment_4200_certified_arc_corpus_distill_lift_seeded.checkpoint.json"
)
EXP4188_ARTIFACT_PATH = (
    ROOT / "results" / "experiment_4188_sovereign_local_generator_gap4_self_distill.json"
)

RANDOM_SEED = 12345
EXEC_TIMEOUT_S = 5.0
BOOTSTRAP_N = 1000

REQUIRED_FIELDS = [
    "honest_verdict",
    "certified_corpus_size",
    "certification_precision",
    "local_induction_cold",
    "local_induction_with_certified_exemplars",
    "distill_lift_ci95",
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
        "Terminal-prefixed. A built certified corpus + an honest latent-or-absent "
        "in-context read is COMPLETE (it tells .390 whether to LoRA-distill or change base)."
    ),
    "certified_corpus_size": (
        "Count of GAP-4-verified demo-perfect (demos->program) pairs -- the "
        "verifier-as-reward training corpus for the ARC sovereignty path (the FR-11 loop where "
        "the energy function is the ground-truth reward)."
    ),
    "certification_precision": (
        "P(held-out-demo + test-pass | certified) of the corpus -- the corpus is only useful "
        "training data if the certifier is precise (the Phase-0 analog for ARC)."
    ),
    "local_induction_cold": (
        "Local base cold induction rate, anchored to the Exp 4188 checkpoint/artifact."
    ),
    "local_induction_with_certified_exemplars": (
        "Local base induction rate seeded with certified exemplars -- vs the exp4188 cold "
        "0.2258; the cheap Invisible-Leash latent-vs-absent read before committing to a heavy "
        ".390 LoRA-distill."
    ),
    "distill_lift_ci95": (
        "Bootstrap CI of (seeded - cold) local induction -- excluding 0 = abstraction LATENT "
        "(distillation viable); touching 0 = ABSENT (need a stronger base)."
    ),
    "model_specs": (
        "The codex generator + the local GGUF; required methodology for a live-LLM artifact."
    ),
    "random_seed": "Determinism precondition; the induction + certification must be reproducible.",
    "reproducibility_checksum": "Hash of the task set + generator config; catches silent drift.",
}

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


def _numpy_only_import(name: str, *args: Any, **kwargs: Any) -> Any:
    if name == "numpy" or name.startswith("numpy."):
        return __import__(name, *args, **kwargs)
    raise ImportError(f"import of {name!r} is blocked in the GAP-4 sandbox")


def _safe_builtins() -> dict[str, Any]:
    import builtins as _builtins

    builtins = {name: getattr(_builtins, name) for name in _SAFE_BUILTIN_NAMES}
    builtins["__import__"] = _numpy_only_import
    return builtins


def safe_transform_from_code(code: str) -> Callable[[Any], np.ndarray | None] | None:
    """Compile a generated ARC transform in the same restricted style as GAP-4."""
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
        except TimeoutError:  # pragma: no cover - defensive around generated infinite loops.
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


def _to_grid_list(value: Any) -> list[list[int]] | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim != 2 or arr.size == 0:
        return None
    return [[int(cell) for cell in row] for row in arr.tolist()]


def _sample_draw_index(sample: dict[str, Any]) -> int:
    value = sample.get("draw_index", 0)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pool(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("pool artifact missing entries list")
    return entries


def _load_programs(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    programs = payload.get("programs")
    if not isinstance(programs, list):
        raise ValueError("program artifact missing programs list")
    return programs


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = _load_json(path)
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict):
        return {}
    return payload


def _gguf_pair_available(specs: list[dict[str, Any]] | None) -> bool:
    return bool(
        isinstance(specs, list)
        and len(specs) >= 2
        and all(str(spec.get("model_path", "")).endswith(".gguf") for spec in specs[:2])
    )


def _default_cached_sota_pair() -> list[dict[str, Any]] | None:  # pragma: no cover
    from scripts.experiment_template import cached_sota_pair  # noqa: PLC0415

    return cached_sota_pair()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_corpus(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def _rate(successes: int, total: int) -> dict[str, Any]:
    return {
        "demo_perfect": int(successes),
        "total": int(total),
        "rate": round(successes / max(1, total), 4),
    }


def build_certified_corpus(
    entries: list[dict[str, Any]],
    programs: list[dict[str, Any]],
    *,
    tau: float = DEFAULT_TAU,
    high_vote_guard_threshold: int | float | None = DEFAULT_HIGH_VOTE_GUARD_THRESHOLD,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Return gate-certified corpus rows, precision counts, and per-task audit rows."""
    corpus: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    correct = 0
    for entry, program in zip(entries, programs, strict=True):
        if entry["task"] != program.get("task"):
            raise ValueError(f"task mismatch: {entry['task']} != {program.get('task')}")
        code = program.get("code")
        if not isinstance(code, str) or not code.strip():
            audit.append({"task": entry["task"], "certified": False, "reason": "missing_code"})
            continue
        fn = safe_transform_from_code(code)
        if fn is None:
            audit.append({"task": entry["task"], "certified": False, "reason": "unsafe_code"})
            continue
        fit = demo_fit(fn, entry["demos"])
        if fit < 1.0:
            audit.append(
                {
                    "task": entry["task"],
                    "certified": False,
                    "reason": "demo_fit_not_exact",
                    "demo_fit": round(fit, 4),
                }
            )
            continue
        pred_grid = _to_grid_list(fn(entry["test_input"]))
        selection = select_guarded_graded_candidate(
            entry["candidates"],
            prediction=pred_grid,
            demo_fit=fit,
            task_id=entry["task"],
            tau=tau,
            high_vote_guard_threshold=high_vote_guard_threshold,
        )
        if not selection["gate_fired"]:
            audit.append(
                {
                    "task": entry["task"],
                    "certified": False,
                    "reason": selection["reason"],
                    "guard_blocked": bool(selection["guard_blocked"]),
                    "min_hamming": selection["min_hamming"],
                }
            )
            continue

        selected = entry["candidates"][int(selection["selected_index"])]
        correct += int(bool(selected.get("correct", False)))
        corpus.append(
            {
                "task": entry["task"],
                "demo_pairs": entry["demos"],
                "program": code.strip(),
                "verifier_label": "gap4_guarded_demo_perfect",
                "demo_fit": 1.0,
                "reward_energy": "execution_consistency_demo_exact_plus_guarded_gate",
                "generator_model": "codex_cached_gap4",
                "source_program_artifact": "arc3_gap4_induced_programs.json",
                "gate_min_hamming": selection["min_hamming"],
                "heldout_candidate_votes": selected.get("votes"),
            }
        )
        audit.append(
            {
                "task": entry["task"],
                "certified": True,
                "min_hamming": selection["min_hamming"],
                "heldout_candidate_votes": selected.get("votes"),
                "heldout_candidate_correct": bool(selected.get("correct", False)),
            }
        )
    certified = len(corpus)
    precision = {
        "correct": correct,
        "certified": certified,
        "rate": round(correct / certified, 4) if certified else 0.0,
    }
    return corpus, precision, audit


def _checkpoint_task_successes(
    entries: list[dict[str, Any]],
    checkpoint_payload: dict[str, Any],
) -> tuple[list[int], list[dict[str, Any]]]:
    tasks = checkpoint_payload.get("tasks") if checkpoint_payload else {}
    if not isinstance(tasks, dict):
        tasks = {}
    successes: list[int] = []
    summaries: list[dict[str, Any]] = []
    for entry in entries:
        samples = tasks.get(entry["task"], [])
        if not isinstance(samples, list):
            samples = []
        hit = False
        best_fit = 0.0
        for sample in sorted(samples, key=_sample_draw_index):
            code = sample.get("code")
            if not isinstance(code, str) or not code.strip():
                continue
            fn = safe_transform_from_code(code)
            if fn is None:
                continue
            fit = demo_fit(fn, entry["demos"])
            best_fit = max(best_fit, fit)
            if fit >= 1.0:
                hit = True
                break
        successes.append(1 if hit else 0)
        summaries.append(
            {
                "task": entry["task"],
                "n_samples": len(samples),
                "demo_perfect": hit,
                "best_demo_fit": round(best_fit, 4),
            }
        )
    return successes, summaries


def bootstrap_lift_ci(
    cold_successes: list[int],
    seeded_successes: list[int],
    *,
    seed: int = RANDOM_SEED,
    n_boot: int = BOOTSTRAP_N,
) -> list[float]:
    if len(cold_successes) != len(seeded_successes):
        raise ValueError("cold and seeded successes must have equal length")
    n = len(cold_successes)
    if n == 0:
        return [0.0, 0.0]
    deltas = [seeded - cold for cold, seeded in zip(cold_successes, seeded_successes, strict=True)]
    if all(delta == 0 for delta in deltas):
        return [0.0, 0.0]
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        draws.append(sum(sample) / n)
    draws.sort()
    lo = draws[int(0.025 * n_boot)]
    hi = draws[min(n_boot - 1, int(0.975 * n_boot))]
    return [round(lo, 4), round(hi, 4)]


def _diagnosis(ci: list[float], seeded_status: str) -> str:
    if seeded_status == "missing_seeded_checkpoint_conservative_flat":
        return "uninformative"
    return "latent" if ci[0] > 0.0 else "absent"


def _verdict(diagnosis: str, corpus_size: int, precision: dict[str, Any], ci: list[float]) -> str:
    if diagnosis == "latent":
        return (
            "success: certified_arc_corpus_latent_lift_"
            f"corpus{corpus_size}_precision{precision['rate']}_ci{ci[0]}_{ci[1]}"
        )
    if diagnosis == "absent":
        return (
            "complete: certified_arc_corpus_absent_lift_ci_touches_zero_"
            f"corpus{corpus_size}_precision{precision['rate']}_ci{ci[0]}_{ci[1]}"
        )
    return (
        "complete: certified_arc_corpus_built_seeded_checkpoint_missing_"
        f"corpus{corpus_size}_precision{precision['rate']}"
    )


def _checksum(
    *,
    entries: list[dict[str, Any]],
    programs: list[dict[str, Any]],
    model_specs: dict[str, Any],
    seed: int,
) -> str:
    blob = json.dumps(
        {
            "seed": seed,
            "tasks": [entry["task"] for entry in entries],
            "program_hashes": [
                hashlib.sha256(str(program.get("code", "")).encode("utf-8")).hexdigest()[:12]
                for program in programs
            ],
            "model_specs": model_specs,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _preconditions(
    *,
    specs: list[dict[str, Any]] | None,
    pool_path: Path,
    programs_path: Path,
    cold_checkpoint_path: Path,
    seeded_checkpoint_path: Path,
) -> list[dict[str, Any]]:
    return [
        {
            "resource": "arc1_codex_induced_programs",
            "path": str(programs_path),
            "available": programs_path.exists(),
        },
        {
            "resource": "arc1_replay_pool",
            "path": str(pool_path),
            "available": pool_path.exists(),
        },
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
        {
            "resource": "local_cold_checkpoint",
            "path": str(cold_checkpoint_path),
            "available": cold_checkpoint_path.exists(),
        },
        {
            "resource": "local_seeded_checkpoint",
            "path": str(seeded_checkpoint_path),
            "available": seeded_checkpoint_path.exists(),
        },
    ]


def _model_specs(
    specs: list[dict[str, Any]] | None,
    cold_checkpoint: dict[str, Any] | None = None,
    seeded_checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "codex_generator": {
            "source": "cached_arc3_gap4_induced_programs",
            "model": "gpt-5.5 codex-tier cached replay",
            "programs_path": str(PROGRAMS_PATH),
        },
        "local_ggufs": [
            {
                "name": spec.get("name"),
                "hf_id": spec.get("hf_id"),
                "model_path": spec.get("model_path"),
            }
            for spec in (specs or [])[:2]
        ],
        "cold_checkpoint_model": (cold_checkpoint or {}).get("local_model_used"),
        "seeded_checkpoint_model": (seeded_checkpoint or {}).get("local_model_used"),
        "gguf_tokenizer_rule": "load by concrete .gguf model_path; do not use AutoTokenizer",
    }


def blocked_artifact(
    *,
    verdict: str,
    preconditions: list[dict[str, Any]],
    model_specs: dict[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    artifact = {
        "experiment": "experiment_4200_certified_arc_corpus_distill_lift",
        "schema": "carnot.experiment_4200_certified_arc_corpus_distill_lift.v1",
        "title": "GAP-4 certified ARC corpus plus local in-context lift read",
        "honest_verdict": verdict,
        "certified_corpus_size": 0,
        "certification_precision": {"correct": 0, "certified": 0, "rate": 0.0},
        "local_induction_cold": _rate(0, 0),
        "local_induction_with_certified_exemplars": _rate(0, 0),
        "distill_lift_ci95": [0.0, 0.0],
        "invisible_leash_diagnosis": "blocked",
        "seeded_generation_status": "blocked",
        "model_specs": model_specs,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "preconditions_checked": preconditions,
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(duration_s, 6),
        "inference_substrate": "precondition_blocked",
    }
    validate_artifact(artifact)
    return artifact


def _complete_artifact(
    *,
    entries: list[dict[str, Any]],
    programs: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    precision: dict[str, Any],
    cold_successes: list[int],
    seeded_successes: list[int],
    cold_summary: list[dict[str, Any]],
    seeded_summary: list[dict[str, Any]],
    seeded_status: str,
    model_specs: dict[str, Any],
    preconditions: list[dict[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    ci = bootstrap_lift_ci(cold_successes, seeded_successes)
    diagnosis = _diagnosis(ci, seeded_status)
    cold_rate = _rate(sum(cold_successes), len(cold_successes))
    seeded_rate = _rate(sum(seeded_successes), len(seeded_successes))
    artifact = {
        "experiment": "experiment_4200_certified_arc_corpus_distill_lift",
        "schema": "carnot.experiment_4200_certified_arc_corpus_distill_lift.v1",
        "title": "GAP-4 certified ARC corpus plus local in-context lift read",
        "honest_verdict": _verdict(diagnosis, len(corpus), precision, ci),
        "certified_corpus_size": len(corpus),
        "certification_precision": precision,
        "local_induction_cold": {
            **cold_rate,
            "source": str(EXP4188_ARTIFACT_PATH),
            "baseline_reference_rate": 0.2258,
        },
        "local_induction_with_certified_exemplars": {
            **seeded_rate,
            "seeded_exemplar_count": len(corpus),
            "condition": "certified_exemplars_in_context_checkpoint_replay",
        },
        "distill_lift_ci95": ci,
        "invisible_leash_diagnosis": diagnosis,
        "seeded_generation_status": seeded_status,
        "model_specs": model_specs,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _checksum(
            entries=entries,
            programs=programs,
            model_specs=model_specs,
            seed=RANDOM_SEED,
        ),
        "preconditions_checked": preconditions,
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(duration_s, 6),
        "inference_substrate": "cached_codex_programs_plus_local_checkpoint_replay",
        "n_tasks": len(entries),
        "cold_task_summary": cold_summary,
        "seeded_task_summary": seeded_summary,
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
    if not isinstance(artifact["certified_corpus_size"], int) or isinstance(
        artifact["certified_corpus_size"], bool
    ):
        raise ValueError("certified_corpus_size must be a bare int")
    for field in (
        "certification_precision",
        "local_induction_cold",
        "local_induction_with_certified_exemplars",
        "model_specs",
        "field_principles",
    ):
        if not isinstance(artifact[field], dict):
            raise ValueError(f"{field} must be a dict")
    if not (
        isinstance(artifact["distill_lift_ci95"], list)
        and len(artifact["distill_lift_ci95"]) == 2
        and all(isinstance(value, float) for value in artifact["distill_lift_ci95"])
    ):
        raise ValueError("distill_lift_ci95 must be a two-float list")
    if not isinstance(artifact["random_seed"], int) or isinstance(artifact["random_seed"], bool):
        raise ValueError("random_seed must be a bare int")
    if not isinstance(artifact["reproducibility_checksum"], str):
        raise ValueError("reproducibility_checksum must be a string")
    if not isinstance(artifact["preconditions_checked"], list):
        raise ValueError("preconditions_checked must be a list")
    if not isinstance(artifact["duration_s"], float):
        raise ValueError("duration_s must be a bare float")
    if not isinstance(artifact["inference_substrate"], str):
        raise ValueError("inference_substrate must be a string")


def run(
    *,
    artifact_path: Path = ARTIFACT_PATH,
    corpus_path: Path = CORPUS_PATH,
    pool_path: Path = POOL_PATH,
    programs_path: Path = PROGRAMS_PATH,
    cold_checkpoint_path: Path = COLD_CHECKPOINT_PATH,
    seeded_checkpoint_path: Path = SEEDED_CHECKPOINT_PATH,
    cached_pair_fn: Callable[[], list[dict[str, Any]] | None] = _default_cached_sota_pair,
) -> dict[str, Any]:
    started = time.time()
    artifact_path = Path(artifact_path)
    corpus_path = Path(corpus_path)
    pool_path = Path(pool_path)
    programs_path = Path(programs_path)
    cold_checkpoint_path = Path(cold_checkpoint_path)
    seeded_checkpoint_path = Path(seeded_checkpoint_path)

    if not (programs_path.exists() and pool_path.exists()):
        preconditions = _preconditions(
            specs=None,
            pool_path=pool_path,
            programs_path=programs_path,
            cold_checkpoint_path=cold_checkpoint_path,
            seeded_checkpoint_path=seeded_checkpoint_path,
        )
        artifact = blocked_artifact(
            verdict="blocked_gap4_arc1_pool_missing",
            preconditions=preconditions,
            model_specs=_model_specs(None),
            duration_s=time.time() - started,
        )
        _write_json(artifact_path, artifact)
        return artifact

    specs = cached_pair_fn()
    cold_checkpoint = _load_checkpoint(cold_checkpoint_path)
    seeded_checkpoint = _load_checkpoint(seeded_checkpoint_path)
    preconditions = _preconditions(
        specs=specs,
        pool_path=pool_path,
        programs_path=programs_path,
        cold_checkpoint_path=cold_checkpoint_path,
        seeded_checkpoint_path=seeded_checkpoint_path,
    )
    model_specs = _model_specs(specs, cold_checkpoint, seeded_checkpoint)
    if not _gguf_pair_available(specs):
        artifact = blocked_artifact(
            verdict="blocked_model_not_cached_sota_gguf",
            preconditions=preconditions,
            model_specs=model_specs,
            duration_s=time.time() - started,
        )
        _write_json(artifact_path, artifact)
        return artifact

    entries = _load_pool(pool_path)
    programs = _load_programs(programs_path)
    corpus, precision, _audit = build_certified_corpus(entries, programs)
    cold_successes, cold_summary = _checkpoint_task_successes(entries, cold_checkpoint)
    if seeded_checkpoint:
        seeded_status = "seeded_checkpoint_replay"
        seeded_successes, seeded_summary = _checkpoint_task_successes(entries, seeded_checkpoint)
    else:
        seeded_status = "missing_seeded_checkpoint_conservative_flat"
        seeded_successes = list(cold_successes)
        seeded_summary = list(cold_summary)

    _write_corpus(corpus_path, corpus)
    artifact = _complete_artifact(
        entries=entries,
        programs=programs,
        corpus=corpus,
        precision=precision,
        cold_successes=cold_successes,
        seeded_successes=seeded_successes,
        cold_summary=cold_summary,
        seeded_summary=seeded_summary,
        seeded_status=seeded_status,
        model_specs=model_specs,
        preconditions=preconditions,
        duration_s=time.time() - started,
    )
    artifact["certified_corpus_path"] = str(corpus_path)
    validate_artifact(artifact)
    _write_json(artifact_path, artifact)
    print(f"-> {artifact['honest_verdict']}", flush=True)
    print(
        "   corpus="
        f"{artifact['certified_corpus_size']} precision={artifact['certification_precision']['rate']} "
        f"cold={artifact['local_induction_cold']['rate']} "
        f"seeded={artifact['local_induction_with_certified_exemplars']['rate']} "
        f"ci={artifact['distill_lift_ci95']}",
        flush=True,
    )
    return artifact


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--pool", type=Path, default=POOL_PATH)
    parser.add_argument("--programs", type=Path, default=PROGRAMS_PATH)
    parser.add_argument("--cold-checkpoint", type=Path, default=COLD_CHECKPOINT_PATH)
    parser.add_argument("--seeded-checkpoint", type=Path, default=SEEDED_CHECKPOINT_PATH)
    args = parser.parse_args(argv)
    run(
        artifact_path=args.artifact,
        corpus_path=args.corpus,
        pool_path=args.pool,
        programs_path=args.programs,
        cold_checkpoint_path=args.cold_checkpoint,
        seeded_checkpoint_path=args.seeded_checkpoint,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
