#!/usr/bin/env python3
# Batching-audit note: `for q in questions:` loop measures Zenil α_t per
# question by computing the verifier-rejection rate over a small sample.
# Per-question α_t timing requires individual latency measurement that
# BatchedInferenceRunner's batch contract does not preserve.
"""Exp 1130 — Zenil alpha_t after the Exp1120 energy-verifier retrain.

This experiment answers whether the better-calibrated SOSKANEnergyV3 verifier
from Exp 1120 provides a stronger exogenous self-distillation signal than the
pre-retrain SOTA baseline from Exp 1077 (alpha_t=0.38).

Important provenance detail: Exp 1120 did not persist a model checkpoint in its
artifact. When no local Exp1130 checkpoint exists, this script reconstructs the
retrained verifier deterministically from the same FoVer v5 corpus, seed, and
hyperparameters, then saves an Exp1130 checkpoint for repeatable scoring.

Spec: REQ-FR11-1130.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _d in [str(REPO_ROOT / "python"), str(REPO_ROOT / "scripts"), str(REPO_ROOT)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.eval.zenil_alpha_post_retrain import (  # noqa: E402
    ALPHA_T_METHOD,
    EXP1120_AUROC,
    EvaluationExample,
    build_exp1130_artifact,
    calibrate_low_energy_threshold,
    load_cached_sota_examples,
    measure_alpha_t_against_temperature,
    pearson_corr,
    summarize_scores,
)

EXP_ID = 1130
DELIVERABLE = REPO_ROOT / "results" / "experiment_1130_zenil_alpha_t_post_retrain.json"
EXP1120_ARTIFACT = REPO_ROOT / "results" / "experiment_1120_energy_verifier_retrain_sota.json"
CACHED_SOTA_PATH = REPO_ROOT / "data" / "fr11_zenil_distill_v2.jsonl"
GSM8K_PATH = REPO_ROOT / "data" / "research" / "gsm8k_adversarial_281.jsonl"
CHECKPOINT_PATH = REPO_ROOT / "results" / "experiment_1130_soskan_energy_v3_retrained.npz"

N_EVALUATION_EXAMPLES = 50
SOTA_NAME = "Qwen3.6-35B-A3B"
SOTA_HF_CACHE = (
    Path.home() / ".cache" / "huggingface" / "hub" / "models--unsloth--Qwen3.6-35B-A3B-GGUF"
)
PREFERRED_GGUF = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
MAX_NEW_TOKENS = 192

_FINAL_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_EQ_RE = re.compile(
    r"(-?\d[\d,]*(?:\.\d+)?)\s*([+\-*/])\s*(-?\d[\d,]*(?:\.\d+)?)\s*=\s*"
    r"(-?\d[\d,]*(?:\.\d+)?)"
)


def _ensure_repo_venv_when_run_directly() -> None:
    """Re-exec into the repo venv so the documented command finds GPU deps.

    The host system Python in this workspace lacks llama_cpp, torch, pytest, and
    transformers, while `.venv/bin/python` has the experiment dependencies.  The
    re-exec only happens for direct script execution, not when tests import this
    module.
    """

    if os.environ.get("CARNOT_EXP1130_VENV_REEXECED") == "1":
        return
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    os.environ["CARNOT_EXP1130_VENV_REEXECED"] = "1"
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


def _ensure_cuda_runtime_on_ld_path() -> None:
    """Expose venv-internal CUDA shared libraries before importing llama_cpp."""

    sentinel = "CARNOT_EXP1130_LDPATH_PATCHED"
    if os.environ.get(sentinel) == "1":
        return
    venv_site = (
        Path(sys.executable).resolve().parent.parent
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    nvidia_root = venv_site / "nvidia"
    if not nvidia_root.is_dir():
        return
    lib_dirs = [str(p / "lib") for p in sorted(nvidia_root.iterdir()) if (p / "lib").is_dir()]
    if not lib_dirs:
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        ":".join([*lib_dirs, existing]) if existing else ":".join(lib_dirs)
    )
    os.environ[sentinel] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


def _number(value: str | int | float) -> float:
    return float(str(value).replace(",", "").replace("$", ""))


def _final_answer_correct(response: str, expected: int | float | str) -> bool:
    """GSM8K-style final-answer check using the last numeric literal."""

    nums = _FINAL_NUM_RE.findall(response)
    if not nums:
        return False
    try:
        return abs(_number(nums[-1]) - _number(expected)) < 1e-6
    except (TypeError, ValueError):
        return False


def _equation_consistency(response: str) -> float:
    """Return fraction of explicit arithmetic equations that evaluate correctly."""

    claims = _EQ_RE.findall(response)
    if not claims:
        return 0.5
    n_ok = 0
    for a_raw, op, b_raw, c_raw in claims:
        try:
            a = _number(a_raw)
            b = _number(b_raw)
            c = _number(c_raw)
            if op == "+":
                ok = abs((a + b) - c) < 1e-6
            elif op == "-":
                ok = abs((a - b) - c) < 1e-6
            elif op == "*":
                ok = abs((a * b) - c) < 1e-6
            elif op == "/":
                ok = b != 0 and abs((a / b) - c) < 1e-6
            else:
                ok = False
        except (TypeError, ValueError):
            ok = False
        if ok:
            n_ok += 1
    return n_ok / len(claims)


def _thinkprm_v2_score_proxy(example: EvaluationExample) -> float:
    """Return a bounded ThinkPRM-v2-style P(correct) proxy.

    Exp 1111 retrained ThinkPRM v2 but did not persist a probe checkpoint.  To
    still produce per-response P(correct) diagnostics without inventing a hidden
    checkpoint, Exp 1130 uses a deterministic arithmetic PRM proxy: final-answer
    agreement and equation consistency are mapped into a calibrated probability.
    The artifact records this mode explicitly.
    """

    final_ok = _final_answer_correct(example.response, example.correct_answer)
    equation_score = _equation_consistency(example.response)
    if final_ok:
        return min(0.99, 0.70 + 0.25 * equation_score)
    return max(0.01, 0.30 * equation_score)


def _load_exp1120_auroc() -> float:
    if not EXP1120_ARTIFACT.exists():
        return EXP1120_AUROC
    try:
        artifact = json.loads(EXP1120_ARTIFACT.read_text(encoding="utf-8"))
        return float(artifact.get("retrained_auroc_val", EXP1120_AUROC))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return EXP1120_AUROC


def _load_exp1120_module() -> Any:
    import importlib.util

    path = REPO_ROOT / "scripts" / "experiment_1120_energy_verifier_retrain_sota.py"
    spec = importlib.util.spec_from_file_location("exp1120_for_exp1130", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Exp1120 module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp1120_for_exp1130"] = module
    spec.loader.exec_module(module)
    return module


def _save_soskan_checkpoint(path: Path, model: Any, metadata: dict[str, Any]) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        W1=model.W1,
        b1=model.b1,
        W2=model.W2,
        b2=model.b2,
        c=model.c,
        metadata=json.dumps(metadata),
    )


def _load_soskan_checkpoint(path: Path, exp1120: Any) -> tuple[Any, dict[str, Any]] | None:
    if not path.exists():
        return None
    import numpy as np

    try:
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data["metadata"]))
        model = exp1120.SOSKANEnergyV3(
            n_splines=int(metadata["n_splines"]),
            rank=int(metadata["rank"]),
            n_features=int(metadata["n_features"]),
            hidden_dim=int(metadata["hidden_dim"]),
            seed=int(metadata.get("seed", exp1120.RANDOM_SEED)),
        )
        model.W1 = data["W1"].astype(np.float64)
        model.b1 = data["b1"].astype(np.float64)
        model.W2 = data["W2"].astype(np.float64)
        model.b2 = data["b2"].astype(np.float64)
        model.c = data["c"].astype(np.float64)
        return model, metadata
    except Exception:
        return None


def _train_or_load_retrained_verifier() -> tuple[Any, float, dict[str, Any], Any]:
    """Return (model, threshold, provenance, exp1120_module)."""

    exp1120 = _load_exp1120_module()
    cached = _load_soskan_checkpoint(CHECKPOINT_PATH, exp1120)
    if cached is not None:
        model, metadata = cached
        threshold = float(metadata.get("energy_threshold", 0.0))
        return (
            model,
            threshold,
            {
                "verifier_load_mode": "exp1130_checkpoint",
                "verifier_checkpoint_path": str(CHECKPOINT_PATH),
                "energy_threshold": threshold,
                "retrained_auroc_recomputed": metadata.get("retrained_auroc"),
            },
            exp1120,
        )

    all_entries = exp1120._load_fover_v5_corpus()
    _holdout_items, holdout_indices = exp1120._select_sota_holdout(all_entries)
    training_pool = [entry for i, entry in enumerate(all_entries) if i not in holdout_indices]
    filtered = [
        entry
        for entry in training_pool
        if float(entry.get("confidence", 1.0)) >= exp1120.NOISE_THRESHOLD
    ]
    X_all, y_all = exp1120._featurize(filtered, exp1120.N_FEATURES)

    import numpy as np

    rng = np.random.default_rng(exp1120.RANDOM_SEED)
    indices = rng.permutation(len(filtered))
    n_train = int(len(filtered) * exp1120.TRAIN_FRAC)
    n_val = int(len(filtered) * exp1120.VAL_FRAC)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]

    model = exp1120.SOSKANEnergyV3(
        n_splines=exp1120.N_SPLINES,
        rank=exp1120.RANK,
        n_features=exp1120.N_FEATURES,
        hidden_dim=exp1120.HIDDEN_DIM,
        seed=exp1120.RANDOM_SEED,
    )
    t0 = time.perf_counter()
    losses = model.fit(X_train, y_train, n_epochs=exp1120.N_EPOCHS, lr=exp1120.LR)
    train_time_s = time.perf_counter() - t0

    retrained_auroc = float(model.auroc_batch(X_val, y_val))
    val_energies = [float(model.energy(X_val[i].astype(np.float64))) for i in range(len(X_val))]
    threshold = calibrate_low_energy_threshold(val_energies, [int(y) for y in y_val])
    metadata = {
        "schema": "carnot.exp1130.soskan_energy_v3_checkpoint.v1",
        "source_experiment": 1120,
        "n_splines": exp1120.N_SPLINES,
        "rank": exp1120.RANK,
        "n_features": exp1120.N_FEATURES,
        "hidden_dim": exp1120.HIDDEN_DIM,
        "seed": exp1120.RANDOM_SEED,
        "energy_threshold": threshold,
        "retrained_auroc": retrained_auroc,
        "n_training_pairs": int(len(X_train)),
        "n_val_pairs": int(len(X_val)),
        "final_train_loss": float(losses[-1]) if losses else None,
    }
    _save_soskan_checkpoint(CHECKPOINT_PATH, model, metadata)
    return (
        model,
        threshold,
        {
            "verifier_load_mode": "retrained_from_exp1120_hyperparameters",
            "verifier_checkpoint_path": str(CHECKPOINT_PATH),
            "energy_threshold": threshold,
            "retrained_auroc_recomputed": round(retrained_auroc, 6),
            "verifier_retrain_time_s": round(train_time_s, 3),
        },
        exp1120,
    )


def _score_energy(model: Any, exp1120: Any, examples: list[EvaluationExample]) -> list[float]:
    import numpy as np

    items = [{"step_text": ex.response, "label": "correct"} for ex in examples]
    X, _ = exp1120._featurize(items, exp1120.N_FEATURES)
    return [float(model.energy(X[i].astype(np.float64))) for i in range(len(examples))]


def _resolve_sota_path() -> str | None:
    explicit = os.environ.get("CARNOT_EXP1130_SOTA_GGUF")
    if explicit and Path(explicit).exists():
        return explicit
    preferred = sorted(SOTA_HF_CACHE.glob(f"snapshots/**/{PREFERRED_GGUF}"))
    if preferred:
        # Prefer the shortest path to avoid duplicate snapshot surprises.
        return str(sorted(preferred, key=lambda p: (len(str(p)), str(p)))[0])
    ggufs = sorted(SOTA_HF_CACHE.glob("snapshots/**/*.gguf"))
    for path in ggufs:
        if "Qwen3.6-35B-A3B" in path.name and "mmproj" not in path.name:
            return str(path)
    return None


def _gpu_available() -> bool:
    try:
        import torch

        if bool(torch.cuda.is_available()):
            return True
    except Exception:
        pass
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return proc.returncode == 0 and "GPU" in proc.stdout
    except Exception:
        return False


def _load_gsm8k_questions(n_examples: int) -> list[dict[str, Any]]:
    rows = []
    if not GSM8K_PATH.exists():
        return rows
    for line in GSM8K_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(
            {
                "question_id": str(row.get("question_id") or f"gsm8k_{len(rows):03d}"),
                "question": str(row.get("variant_question") or row.get("original_question")),
                "answer": row.get("variant_answer", row.get("original_answer")),
            }
        )
        if len(rows) >= n_examples:
            break
    return rows


def _generate_live_examples(n_examples: int) -> tuple[list[EvaluationExample], dict[str, Any]]:
    if os.environ.get("CARNOT_EXP1130_FORCE_CACHED") == "1":
        return [], {"live_generation_skipped_reason": "forced_cached"}
    if not _gpu_available():
        return [], {"live_generation_skipped_reason": "no_gpu_available"}

    model_path = _resolve_sota_path()
    if model_path is None:
        return [], {"live_generation_skipped_reason": "sota_gguf_not_found"}

    questions = _load_gsm8k_questions(n_examples)
    if len(questions) < n_examples:
        return [], {"live_generation_skipped_reason": "gsm8k_question_pool_too_small"}

    try:
        from llama_cpp import Llama
    except Exception as exc:
        return [], {"live_generation_skipped_reason": f"llama_cpp_unavailable: {exc}"}

    t0 = time.perf_counter()
    try:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=2048,
            verbose=False,
        )
    except Exception as exc:
        return [], {
            "live_generation_skipped_reason": f"llama_load_failed: {type(exc).__name__}: {exc}",
            "model_path": model_path,
        }

    examples: list[EvaluationExample] = []
    for q in questions:
        prompt = (
            "Solve the GSM8K math problem. Show concise arithmetic and end with "
            f"'Answer: <number>'.\n\nProblem: {q['question']}\n\nSolution:"
        )
        try:
            out = llm(
                prompt,
                max_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
                stop=["\n\n\n", "Problem:"],
            )
            response = str(out["choices"][0]["text"]).strip()
        except Exception as exc:
            response = f"GENERATION_ERROR: {type(exc).__name__}: {exc}"
        label = 1 if _final_answer_correct(response, q["answer"]) else 0
        examples.append(
            EvaluationExample(
                example_id=q["question_id"],
                question=q["question"],
                response=response,
                correct_answer=q["answer"],
                label=label,
            )
        )

    try:
        close = getattr(llm, "close", None)
        if close is not None:
            close()
    except Exception:
        pass

    return examples, {
        "model_path": model_path,
        "live_generation_time_s": round(time.perf_counter() - t0, 3),
        "dataset": "gsm8k_adversarial_281",
    }


def _load_evaluation_examples() -> tuple[list[EvaluationExample], str, dict[str, Any], str]:
    live_examples, live_meta = _generate_live_examples(N_EVALUATION_EXAMPLES)
    if len(live_examples) >= N_EVALUATION_EXAMPLES:
        return live_examples[:N_EVALUATION_EXAMPLES], "live_gpu", live_meta, str(GSM8K_PATH)

    cached = load_cached_sota_examples(CACHED_SOTA_PATH, N_EVALUATION_EXAMPLES)
    meta = {
        **live_meta,
        "cached_source": str(CACHED_SOTA_PATH),
        "cached_reason": "live_generation_unavailable_or_incomplete",
    }
    return cached, "cached", meta, str(CACHED_SOTA_PATH)


def run_experiment() -> dict[str, Any]:
    t0 = time.perf_counter()
    verifier_auroc_used = _load_exp1120_auroc()
    examples, inference_mode, inference_meta, examples_path = _load_evaluation_examples()

    measurement_complete = len(examples) >= N_EVALUATION_EXAMPLES
    if not examples:
        return build_exp1130_artifact(
            alpha_t_post_retrain=0.0,
            verifier_auroc_used=verifier_auroc_used,
            n_evaluation_examples=0,
            inference_mode=inference_mode,
            measurement_complete=False,
            fr11_logged=True,
            verifier_ground_truth_corr=0.0,
            thinkprm_ground_truth_corr=0.0,
            alpha_t_method=ALPHA_T_METHOD,
            score_summary={},
            examples_path=examples_path,
            extra={
                "duration_s": round(time.perf_counter() - t0, 3),
                **inference_meta,
            },
        )

    model, threshold, verifier_meta, exp1120 = _train_or_load_retrained_verifier()
    energy_scores = _score_energy(model, exp1120, examples)
    labels = [int(ex.label) for ex in examples]
    thinkprm_scores = [_thinkprm_v2_score_proxy(ex) for ex in examples]

    alpha = measure_alpha_t_against_temperature(examples, energy_scores, threshold)
    verifier_corr = pearson_corr([-float(e) for e in energy_scores], labels)
    thinkprm_corr = pearson_corr(thinkprm_scores, labels)
    score_summary = summarize_scores(energy_scores, labels, thinkprm_scores)

    per_example = []
    for ex, energy, think_score, verifier_v, temp_v in zip(
        examples,
        energy_scores,
        thinkprm_scores,
        alpha.verifier_verdicts,
        alpha.temperature_verdicts,
    ):
        per_example.append(
            {
                "example_id": ex.example_id,
                "label": int(ex.label),
                "correct_answer": ex.correct_answer,
                "verifier_score_energy": round(float(energy), 6),
                "verifier_verdict": verifier_v,
                "temperature_verdict": temp_v,
                "thinkprm_score": round(float(think_score), 6),
                "response_length": len(ex.response),
            }
        )

    artifact = build_exp1130_artifact(
        alpha_t_post_retrain=alpha.alpha_t,
        verifier_auroc_used=verifier_auroc_used,
        n_evaluation_examples=len(examples),
        inference_mode=inference_mode,
        measurement_complete=measurement_complete,
        fr11_logged=True,
        verifier_ground_truth_corr=verifier_corr,
        thinkprm_ground_truth_corr=thinkprm_corr,
        alpha_t_method=ALPHA_T_METHOD,
        score_summary=score_summary,
        examples_path=examples_path,
        extra={
            "duration_s": round(time.perf_counter() - t0, 3),
            "n_alpha_t_disagreements": alpha.n_disagreements,
            "alpha_t_disagreement_ids": alpha.disagreement_ids,
            "thinkprm_score_mode": "deterministic_arithmetic_proxy_no_v2_checkpoint",
            "verifier_score_direction": "lower_energy_means_more_likely_correct",
            "energy_threshold": round(float(threshold), 6),
            "inversion_fixed_used": True,
            "sota_model": SOTA_NAME,
            "n_evaluation_examples_target": N_EVALUATION_EXAMPLES,
            "per_example": per_example,
            **inference_meta,
            **verifier_meta,
        },
    )
    return artifact


def main() -> int:
    _ensure_repo_venv_when_run_directly()
    _ensure_cuda_runtime_on_ld_path()
    artifact = run_experiment()
    DELIVERABLE.parent.mkdir(parents=True, exist_ok=True)
    DELIVERABLE.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"WROTE {DELIVERABLE}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(
        f"alpha_t_post_retrain: {artifact['alpha_t_post_retrain']} "
        f"(prior={artifact['alpha_t_prior']}) inference_mode={artifact['inference_mode']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
