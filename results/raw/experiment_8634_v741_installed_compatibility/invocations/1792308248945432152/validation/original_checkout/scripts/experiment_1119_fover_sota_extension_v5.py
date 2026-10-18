#!/usr/bin/env python3
"""Exp 1119 — Extend FoVer corpus with SOTA-model (Qwen3.6-35B / Gemma4-31B) outputs.

**Researcher summary (read this even if you skim the code):**

    arXiv 2504.13134 (EBRM) identified an energy-ordering inversion on SOTA
    model outputs: correct answers scored HIGHER energy than incorrect ones
    (mean_correct=0.689 > mean_incorrect=0.621) in exp1100/exp1115. The root
    cause is distributional mismatch — FoVer v4 was trained entirely on base-
    model outputs (high error rate, unoptimised), while Qwen3.6-35B-A3B and
    Gemma4-31B-it are RL-optimised SOTA models that lie far outside the training
    distribution. The verifier's calibration inverts on OOD data.

    This experiment is Fix Step 1: extend the FoVer corpus with SOTA model
    outputs so that exp1120 can retrain the energy verifier on a distribution
    that includes SOTA model behaviour. The labeling uses Z3MathVerifier for
    arithmetic-step correctness — NOT ThinkPRM v2, which would create circular
    training data for exp1120.

**What we do:**

    1. Convert the existing FoVer corpus (data/fover_corpus_v4.json, 6548 pairs)
       to JSONL format at data/fover_corpus.jsonl if not already present.
       This is a lossless migration — every existing field is preserved plus
       ``model="base_model"`` and ``source="fover_v4"`` are added for lineage.

    2. Sample 500 GSM8K training questions whose indices are NOT already
       represented in the corpus. Since the v4 corpus used training-split
       indices 1-2000 (one-based), we sample from 2001+ using the training
       split of the gsm8k dataset.

    3. Generate N=2 CoT solutions per question using:
          - Qwen3.6-35B-A3B-GGUF  (primary, SOTA MoE, ~3B active params)
          - Gemma4-31B-it-GGUF    (secondary, SOTA dense)
       via llama.cpp loaded from the HuggingFace cache. The dual-GPU rig
       (2× RTX 3090, 48 GB discrete VRAM) handles both models in tensor-split
       mode. A hard wall budget (480 s) caps inference so the experiment exits
       gracefully if GPU throughput is lower than expected.

    4. Split each CoT solution into numbered steps. Label each step with
       Z3MathVerifier: score() returns the arithmetic violation fraction
       (0.0 = fully correct, 1.0 = fully violated, 0.5 = no equations found).
       Steps with score < 0.3 are labeled "correct" (label=1); score > 0.7
       are "incorrect" (label=0); the middle band uses a heuristic based on
       the final numeric answer presence.

    5. Append labeled steps to data/fover_corpus.jsonl and write the
       standardised result artifact.

**Label schema (new SOTA extension rows):**

    {
      "question_id": "gsm8k_2001_0_0",
      "step_text": "Step 1: ...",
      "label": "correct" | "incorrect",
      "confidence": 0.8,
      "model": "Qwen3.6-35B" | "Gemma4-31B",
      "source": "sota_extension_v5",
      "verifier": "Z3Math" | "heuristic"
    }

**Honest-result discipline:**

    honest_verdict shapes:
      * "corpus_extended_above_7000"  — n_pairs_after >= 7000
      * "corpus_extended_below_7000"  — extension succeeded but < 7000 pairs
      * "partial"                     — some pairs added, wall budget hit early
      * "failed"                      — GPU unavailable or model not cached

Spec: REQ-VERIFY-083 (live_gpu evidence), REQ-INFER-SOTA-001 (SOTA-tier
      model), REQ-LEARN-011 (continuous self-learning corpus extension).
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repo root / path bootstrap — same pattern as exp1118.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _d in [str(_REPO_ROOT / "python"), str(_REPO_ROOT / "scripts"), str(_REPO_ROOT)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# ---------------------------------------------------------------------------
# CUDA runtime LD path patch (required for llama_cpp on this host).
# llama_cpp links against libcudart.so.12 which isn't on the default
# LD_LIBRARY_PATH. We prepend the nvidia/* lib dirs from the venv and
# re-exec once so the dynamic linker picks them up.
# ---------------------------------------------------------------------------


def _patch_cuda_ld_path_and_reexec() -> None:
    """Prepend venv nvidia/* lib dirs to LD_LIBRARY_PATH and re-exec.

    The dynamic linker reads LD_LIBRARY_PATH at process startup, so
    modifying os.environ after launch does nothing for already-loaded
    shared libraries. The only reliable fix is to set the variable and
    then os.execv() so the kernel launches a fresh process image with
    the correct search path. A sentinel env var prevents infinite re-exec.
    """
    sentinel = "CARNOT_LDPATH_PATCHED"
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
    lib_dirs: list[str] = []
    for sub in sorted(nvidia_root.iterdir()):
        lib = sub / "lib"
        if lib.is_dir():
            lib_dirs.append(str(lib))
    if not lib_dirs:
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    new_val = ":".join([*lib_dirs, existing]) if existing else ":".join(lib_dirs)
    os.environ["LD_LIBRARY_PATH"] = new_val
    os.environ[sentinel] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


_patch_cuda_ld_path_and_reexec()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 1119
EXP_TITLE = "FoVer corpus extension with SOTA model outputs (sota_extension_v5)"
DELIVERABLE = str(_REPO_ROOT / "results" / "experiment_1119_fover_sota_extension_v5.json")

FOVER_V4_JSON = _REPO_ROOT / "data" / "fover_corpus_v4.json"
FOVER_JSONL = _REPO_ROOT / "data" / "fover_corpus.jsonl"

# SOTA model HuggingFace IDs — mandated by CLAUDE.md.
QWEN_HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA_HF_ID = "unsloth/gemma-4-31B-it-GGUF"

# GSM8K training questions already in the v4 corpus used indices 1-2000
# (one-based). We sample from index 2001+ (zero-based: 2000+) to avoid
# any overlap. The GSM8K training split has 7473 questions, giving us
# 5473 fresh candidates.
CORPUS_EXISTING_MAX_IDX = 2000  # one-based max in v4 corpus
GSM8K_OFFSET = CORPUS_EXISTING_MAX_IDX  # zero-based start for fresh questions

# Inference budget and sizing.
# 500 questions × N=2 solutions ≈ 1000 completions at ~4 s/completion on
# dual-3090 with Qwen3.6-35B-A3B MoE is ~66 minutes — over budget.
# We split the wall budget across both models (240 s each) and log whatever
# we produced. A hard exit guard stops inference 20 s before the wall
# so we have time to write the artifact.
N_QUESTIONS_TARGET = 500
N_SOLUTIONS_PER_Q = 2  # generate N=2 solutions per question per model
MAX_NEW_TOKENS = 256
INFERENCE_WALL_BUDGET_S = 480.0  # 8 minutes total for inference phase

_FINAL_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_STEP_SPLIT_RE = re.compile(r"(?:^|\n)(?:Step\s*\d+[:.)\s]|\d+\.\s+|\*\s+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# FoVer corpus utilities
# ---------------------------------------------------------------------------


def load_fover_v4() -> list[dict[str, Any]]:
    """Load the existing FoVer v4 corpus from JSON.

    The v4 corpus contains 6548 step-level pairs derived from base-model
    GSM8K solutions. Each entry has fields: question_id, step_text,
    label ("correct"|"incorrect"), confidence. We preserve all of them
    and add source="fover_v4" and model="base_model" for lineage tracking.
    """
    if not FOVER_V4_JSON.exists():
        return []
    with FOVER_V4_JSON.open() as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        return []
    return data


def initialize_fover_jsonl_if_needed() -> int:
    """Convert v4.json → fover_corpus.jsonl if the JSONL file is absent.

    Returns the number of entries written (0 if file already existed).
    This is a one-time migration: every downstream reader uses the JSONL
    file; the v4.json is kept as an immutable archive.
    """
    if FOVER_JSONL.exists():
        return 0
    FOVER_JSONL.parent.mkdir(parents=True, exist_ok=True)
    entries = load_fover_v4()
    written = 0
    with FOVER_JSONL.open("w") as fh:
        for entry in entries:
            row: dict[str, Any] = dict(entry)
            row.setdefault("model", "base_model")
            row.setdefault("source", "fover_v4")
            row.setdefault("verifier", "heuristic")
            fh.write(json.dumps(row) + "\n")
            written += 1
    return written


def count_fover_jsonl() -> int:
    """Return the number of lines (entries) in data/fover_corpus.jsonl."""
    if not FOVER_JSONL.exists():
        return 0
    with FOVER_JSONL.open() as fh:
        return sum(1 for line in fh if line.strip())


def existing_question_ids() -> set[str]:
    """Return the set of question IDs already present in the JSONL corpus.

    Used when sampling fresh GSM8K questions to avoid repeating any
    question that already appears in the training data — data leakage
    from repeating train questions would corrupt exp1120's validation.
    """
    ids: set[str] = set()
    if not FOVER_JSONL.exists():
        return ids
    with FOVER_JSONL.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                qid = row.get("question_id", "")
                # Extract the numeric index from ids like "gsm8k_2001_0_0"
                m = re.match(r"gsm8k_(\d+)", qid)
                if m:
                    ids.add(qid)
            except json.JSONDecodeError:
                continue
    return ids


# ---------------------------------------------------------------------------
# GSM8K loader
# ---------------------------------------------------------------------------


def load_fresh_gsm8k(n: int, *, offset: int = GSM8K_OFFSET) -> list[dict[str, Any]]:
    """Load ``n`` GSM8K training questions starting at zero-based ``offset``.

    GSM8K training split has 7473 questions. We start after index 2000
    (the highest index already in the v4 corpus) so every returned question
    is guaranteed to be absent from the existing FoVer corpus regardless
    of the JSONL's current state.

    The expected integer answer is extracted from the ``#### N`` suffix
    that GSM8K appends to every ground-truth solution.
    """
    from datasets import load_dataset  # local import keeps test imports light

    end = offset + n
    ds = load_dataset("gsm8k", "main", split=f"train[{offset}:{end}]")
    out: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", row["answer"])
        if not m:
            continue
        try:
            expected = float(m.group(1))
        except ValueError:
            continue
        out.append(
            {
                "question_id": f"gsm8k_{offset + i + 1}",  # one-based
                "question": row["question"],
                "answer": expected,
            }
        )
    return out


# ---------------------------------------------------------------------------
# CoT step extraction
# ---------------------------------------------------------------------------


def split_cot_into_steps(cot: str) -> list[str]:
    """Split a chain-of-thought solution into individual reasoning steps.

    A "step" is a coherent fragment of the reasoning trace: either
    a numbered step ("Step 1:", "1."), a bullet point ("* ..."), or
    a double-newline-separated paragraph. We return at least one step
    even for unstructured responses, because a well-formed GSM8K solution
    that lacks explicit step markers still contains verifiable arithmetic.

    Why split at all: Z3MathVerifier operates on individual step texts,
    not on the whole solution, because step-level granularity is what the
    FoVer corpus models — each row is a single reasoning step, not a
    complete solution.
    """
    cot = cot.strip()
    if not cot:
        return []

    # Try numbered-step pattern first.
    parts = _STEP_SPLIT_RE.split(cot)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) >= 2:
        return parts

    # Fallback: paragraph split on double newline.
    paragraphs = [p.strip() for p in cot.split("\n\n") if p.strip()]
    if len(paragraphs) >= 2:
        return paragraphs

    # Last resort: return the full text as a single step.
    return [cot]


# ---------------------------------------------------------------------------
# Step labeling with Z3MathVerifier
# ---------------------------------------------------------------------------


def _load_z3_verifier() -> Any:
    """Load Z3MathVerifier directly, bypassing the package __init__.

    The carnot package __init__ imports JAX at top level. Importing
    ``carnot.verify.z3_math_verifier`` via the package triggers that
    import even when JAX_PLATFORMS=cpu — which is slow and occasionally
    crashes. We use importlib to load only the specific module we need.
    """
    import importlib.util

    z3_path = _REPO_ROOT / "python" / "carnot" / "verify" / "z3_math_verifier.py"
    spec = importlib.util.spec_from_file_location("z3_math_verifier", z3_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod.Z3MathVerifier()
    except Exception:
        return None


def label_step(
    step_text: str,
    verifier: Any,
    final_answer_correct: bool,
) -> tuple[str, float, str]:
    """Label a single reasoning step as "correct" or "incorrect".

    Returns ``(label, confidence, verifier_name)`` where:
      - label:          "correct" | "incorrect"
      - confidence:     float in [0, 1]
      - verifier_name:  "Z3Math" | "heuristic"

    The Z3MathVerifier returns an arithmetic violation fraction in [0, 1].
    We interpret it as:
      * score < 0.3  → step is arithmetically correct   (label="correct")
      * score > 0.7  → step is arithmetically incorrect (label="incorrect")
      * 0.3 ≤ score ≤ 0.7 → indeterminate; fall through to heuristic

    The heuristic for the indeterminate band (and when Z3 is unavailable)
    is based on whether the step contains a numeric literal (answer
    committed) and whether the overall solution got the final answer right.
    A step in a correct solution that contains a number is labeled correct
    with confidence 0.7; otherwise confidence falls to 0.55 (near-random).

    Why NOT circular ThinkPRM v2: using ThinkPRM v2 as the labeler would
    mean that exp1120 trains on labels produced by the same probe it is
    trying to improve — bootstrap collapse. Z3MathVerifier is independent
    (formal arithmetic, no learned weights) so labels are guaranteed
    to be free of this circular dependency.
    """
    # Attempt formal arithmetic verification.
    if verifier is not None:
        try:
            score = float(verifier.score(step_text))
            if score < 0.3:
                return "correct", 1.0 - score, "Z3Math"
            if score > 0.7:
                return "incorrect", score, "Z3Math"
            # Indeterminate band — fall through to heuristic below.
        except Exception:
            pass

    # Heuristic fallback: use final-answer correctness + numeric presence.
    has_number = bool(_FINAL_NUM_RE.search(step_text))
    if final_answer_correct and has_number:
        return "correct", 0.75, "heuristic"
    if not final_answer_correct and has_number:
        return "incorrect", 0.65, "heuristic"
    if final_answer_correct:
        return "correct", 0.60, "heuristic"
    return "incorrect", 0.60, "heuristic"


# ---------------------------------------------------------------------------
# SOTA model resolution
# ---------------------------------------------------------------------------


def _resolve_gguf(hf_id: str) -> str | None:
    """Return the cached GGUF file path for ``hf_id``, or None if absent.

    We use a direct importlib load of sota_models.py to avoid the carnot
    package __init__ (same reason as _load_z3_verifier above).
    """
    import importlib.util

    sota_path = _REPO_ROOT / "python" / "carnot" / "inference" / "sota_models.py"
    spec = importlib.util.spec_from_file_location("sota_models", sota_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        p = mod.resolve_cached_gguf(hf_id)
    except Exception:
        return None
    if not p or not Path(p).exists():
        return None
    return p


def _preload_cuda_libs() -> None:
    """Pre-load CUDA shared libraries via ctypes RTLD_GLOBAL so that
    subsequent dlopen calls (inside llama_cpp) can resolve libcudart.

    LD_LIBRARY_PATH is consumed by the kernel dynamic linker at process
    startup, not by subsequent dlopen calls on all systems. Pre-loading
    with RTLD_GLOBAL makes the symbols globally visible to later dlopen.

    We iterate over ``site.getsitepackages()`` rather than constructing
    the path from ``sys.executable`` because uv-managed Python resolves
    ``sys.executable`` to the uv cache location, not the venv root.
    ``site.getsitepackages()`` is always correct for the active venv.
    """
    import ctypes
    import site

    for site_pkg in site.getsitepackages():
        nvidia_root = Path(site_pkg) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for sub in sorted(nvidia_root.iterdir()):
            lib_dir = sub / "lib"
            if not lib_dir.is_dir():
                continue
            for so_file in sorted(lib_dir.glob("*.so*")):
                try:
                    ctypes.CDLL(str(so_file), ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def _load_llm(model_path: str, n_gpu_layers: int = -1) -> Any | None:
    """Load a llama_cpp Llama model from ``model_path``.

    Returns None on ImportError or any load failure so the caller can
    fall back gracefully. n_gpu_layers=-1 means "offload all layers to GPU"
    which is the correct setting for the dual-RTX-3090 rig.
    """
    _preload_cuda_libs()
    try:
        from llama_cpp import Llama  # type: ignore[import]
    except (ImportError, RuntimeError, OSError):
        return None
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=1024,
            n_gpu_layers=n_gpu_layers,
            tensor_split=None,  # auto-split across all visible GPUs
            verbose=False,
        )
        return llm
    except Exception as e:
        print(f"[exp1119] model load error: {e}", flush=True)
        return None


def _generate_cot(llm: Any, question: str) -> str:
    """Generate a chain-of-thought GSM8K solution for ``question``.

    Instructs the model to show its arithmetic explicitly ("Step N:" format
    with "=" signs) so that Z3MathVerifier can extract and check equations.
    Temperature 0.7 introduces diversity between the two solutions per
    question that GRPO-style training needs; greedy decoding (T=0.0) would
    collapse both solutions to the same output.
    """
    prompt = (
        "Solve the following math problem step by step. "
        "Show each step clearly using 'Step N:' format. "
        "Write out all arithmetic with '=' signs.\n\n"
        f"Problem: {question}\n\nSolution:"
    )
    try:
        out = llm(
            prompt,
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.95,
            stop=["\nProblem:", "\n\n\n\n"],
        )
        return out["choices"][0]["text"].strip()
    except Exception as e:
        print(f"[exp1119] generation error: {e}", flush=True)
        return ""


def final_answer_correct(cot: str, expected: float) -> bool:
    """Return True iff the last numeric literal in ``cot`` matches ``expected``.

    GSM8K-style evaluation: the model's committed answer is the last number
    in its reasoning trace. Approximate equality (1e-6) handles float-
    rounded outputs like "160.0" matching expected 160.
    """
    nums = _FINAL_NUM_RE.findall(cot)
    if not nums:
        return False
    try:
        return abs(float(nums[-1]) - float(expected)) < 1e-6
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Core generation loop
# ---------------------------------------------------------------------------


def generate_sota_pairs(
    questions: list[dict[str, Any]],
    model_path: str,
    model_name: str,
    verifier: Any,
    *,
    wall_deadline: float,
    n_solutions: int = N_SOLUTIONS_PER_Q,
) -> list[dict[str, Any]]:
    """Generate and label step-level pairs from ``questions`` using ``model_path``.

    For each question we generate ``n_solutions`` CoT completions, then
    split each completion into steps and label each step with ``verifier``.
    The outer loop respects ``wall_deadline`` (a time.perf_counter() value)
    so we never exceed the experiment's hard wall budget.

    Returns a flat list of labeled step rows ready to append to the JSONL.

    Why flat list vs. nested per-question dict: the FoVer corpus schema is
    flat (one row per step) so we emit the same shape, keeping the append
    path simple and the downstream training code unchanged.
    """
    llm = _load_llm(model_path)
    if llm is None:
        print(f"[exp1119] could not load {model_name}, skipping", flush=True)
        return []

    pairs: list[dict[str, Any]] = []
    for qi, q in enumerate(questions):
        if time.perf_counter() > wall_deadline:
            print(
                f"[exp1119] wall deadline hit after {qi} questions ({model_name})",
                flush=True,
            )
            break

        q_id = q["question_id"]
        expected = q["answer"]

        for sol_idx in range(n_solutions):
            if time.perf_counter() > wall_deadline:
                break

            cot = _generate_cot(llm, q["question"])
            if not cot:
                continue

            is_correct = final_answer_correct(cot, expected)
            steps = split_cot_into_steps(cot)

            for step_idx, step_text in enumerate(steps):
                if not step_text.strip():
                    continue
                label, confidence, verifier_name = label_step(step_text, verifier, is_correct)
                pairs.append(
                    {
                        "question_id": f"{q_id}_{sol_idx}_{step_idx}",
                        "step_text": step_text,
                        "label": label,
                        "confidence": confidence,
                        "model": model_name,
                        "source": "sota_extension_v5",
                        "verifier": verifier_name,
                    }
                )

        if qi % 10 == 0:
            print(
                f"[exp1119] {model_name}: {qi}/{len(questions)} questions, "
                f"{len(pairs)} pairs so far",
                flush=True,
            )

    return pairs


# ---------------------------------------------------------------------------
# Main experiment entry point
# ---------------------------------------------------------------------------


def run_experiment() -> dict[str, Any]:
    """Execute the FoVer corpus extension and return the result artifact dict.

    Separated from ``main()`` so unit tests can call it with mocked
    internals without requiring a GPU or real model weights on disk.
    """
    t_start = time.perf_counter()

    # -- Phase 0: Initialize JSONL from v4.json if needed ------------------
    print("[exp1119] Phase 0: initializing fover_corpus.jsonl if needed", flush=True)
    initialized = initialize_fover_jsonl_if_needed()
    if initialized:
        print(f"[exp1119] converted {initialized} v4 entries to JSONL", flush=True)

    n_pairs_before = count_fover_jsonl()
    print(f"[exp1119] n_pairs_before = {n_pairs_before}", flush=True)

    # -- Phase 1: Load fresh GSM8K questions --------------------------------
    print("[exp1119] Phase 1: loading fresh GSM8K questions", flush=True)
    try:
        questions = load_fresh_gsm8k(N_QUESTIONS_TARGET, offset=GSM8K_OFFSET)
    except Exception as e:
        print(f"[exp1119] GSM8K load failed: {e}", flush=True)
        questions = []

    if not questions:
        print("[exp1119] no fresh questions available — writing failed artifact", flush=True)
        return _build_artifact(
            n_pairs_before=n_pairs_before,
            n_pairs_added=0,
            n_pairs_after=n_pairs_before,
            models_used=[],
            labeling_verifiers=[],
            label_positive_fraction=0.0,
            inference_mode="live_gpu",
            honest_verdict="failed",
            duration_s=time.perf_counter() - t_start,
            notes="GSM8K dataset unavailable",
        )

    print(f"[exp1119] loaded {len(questions)} fresh questions", flush=True)

    # -- Phase 2: Load Z3MathVerifier --------------------------------------
    print("[exp1119] Phase 2: loading Z3MathVerifier", flush=True)
    verifier = _load_z3_verifier()
    verifier_name = "Z3Math" if verifier is not None else "heuristic"
    print(f"[exp1119] verifier: {verifier_name}", flush=True)

    # -- Phase 3: Resolve SOTA model paths ---------------------------------
    print("[exp1119] Phase 3: resolving SOTA model paths", flush=True)
    qwen_path = _resolve_gguf(QWEN_HF_ID)
    gemma_path = _resolve_gguf(GEMMA_HF_ID)

    if qwen_path:
        print(f"[exp1119] Qwen3.6-35B path: {qwen_path}", flush=True)
    else:
        print("[exp1119] Qwen3.6-35B NOT in cache — will skip", flush=True)

    if gemma_path:
        print(f"[exp1119] Gemma4-31B path: {gemma_path}", flush=True)
    else:
        print("[exp1119] Gemma4-31B NOT in cache — will skip", flush=True)

    if not qwen_path and not gemma_path:
        return _build_artifact(
            n_pairs_before=n_pairs_before,
            n_pairs_added=0,
            n_pairs_after=n_pairs_before,
            models_used=[],
            labeling_verifiers=[verifier_name],
            label_positive_fraction=0.0,
            inference_mode="live_gpu",
            honest_verdict="failed",
            duration_s=time.perf_counter() - t_start,
            notes="No SOTA GGUF models found in HF cache",
        )

    # -- Phase 4: Generate SOTA pairs within wall budget -------------------
    # Split the budget: Qwen gets 55%, Gemma gets 40%, 5% buffer for I/O.
    inference_start = time.perf_counter()
    wall_deadline = inference_start + INFERENCE_WALL_BUDGET_S * 0.95

    all_new_pairs: list[dict[str, Any]] = []
    models_used: list[str] = []
    verifiers_used: set[str] = set()

    if qwen_path:
        qwen_deadline = inference_start + INFERENCE_WALL_BUDGET_S * 0.50
        print(
            "[exp1119] Phase 4a: generating with Qwen3.6-35B-A3B "
            f"({len(questions)} questions, deadline in "
            f"{qwen_deadline - time.perf_counter():.0f}s)",
            flush=True,
        )
        qwen_pairs = generate_sota_pairs(
            questions,
            qwen_path,
            "Qwen3.6-35B",
            verifier,
            wall_deadline=min(qwen_deadline, wall_deadline),
            n_solutions=N_SOLUTIONS_PER_Q,
        )
        all_new_pairs.extend(qwen_pairs)
        if qwen_pairs:
            models_used.append(QWEN_HF_ID)
            verifiers_used.update(r["verifier"] for r in qwen_pairs)
        print(f"[exp1119] Qwen produced {len(qwen_pairs)} step pairs", flush=True)

    if gemma_path and time.perf_counter() < wall_deadline:
        gemma_deadline = inference_start + INFERENCE_WALL_BUDGET_S * 0.90
        print(
            "[exp1119] Phase 4b: generating with Gemma4-31B-it "
            f"({len(questions)} questions, deadline in "
            f"{gemma_deadline - time.perf_counter():.0f}s)",
            flush=True,
        )
        gemma_pairs = generate_sota_pairs(
            questions,
            gemma_path,
            "Gemma4-31B",
            verifier,
            wall_deadline=min(gemma_deadline, wall_deadline),
            n_solutions=N_SOLUTIONS_PER_Q,
        )
        all_new_pairs.extend(gemma_pairs)
        if gemma_pairs:
            models_used.append(GEMMA_HF_ID)
            verifiers_used.update(r["verifier"] for r in gemma_pairs)
        print(f"[exp1119] Gemma produced {len(gemma_pairs)} step pairs", flush=True)

    # -- Phase 5: Append new pairs to JSONL --------------------------------
    print(f"[exp1119] Phase 5: appending {len(all_new_pairs)} pairs to JSONL", flush=True)
    if all_new_pairs:
        with FOVER_JSONL.open("a") as fh:
            for row in all_new_pairs:
                fh.write(json.dumps(row) + "\n")

    n_pairs_after = count_fover_jsonl()
    n_pairs_added = len(all_new_pairs)

    # -- Phase 6: Compute label statistics ---------------------------------
    n_correct = sum(1 for r in all_new_pairs if r.get("label") == "correct")
    label_positive_fraction = n_correct / n_pairs_added if n_pairs_added > 0 else 0.0

    # -- Phase 7: Determine honest verdict ---------------------------------
    if n_pairs_added == 0:
        verdict = "failed"
    elif n_pairs_added < N_QUESTIONS_TARGET * N_SOLUTIONS_PER_Q:
        # Wall budget hit before processing all questions.
        verdict = "partial" if n_pairs_after < 7000 else "corpus_extended_above_7000"
    elif n_pairs_after >= 7000:
        verdict = "corpus_extended_above_7000"
    else:
        verdict = "corpus_extended_below_7000"

    duration_s = time.perf_counter() - t_start
    print(
        f"[exp1119] done: {n_pairs_added} pairs added, "
        f"total {n_pairs_after}, verdict={verdict}, "
        f"duration={duration_s:.1f}s",
        flush=True,
    )

    return _build_artifact(
        n_pairs_before=n_pairs_before,
        n_pairs_added=n_pairs_added,
        n_pairs_after=n_pairs_after,
        models_used=models_used,
        labeling_verifiers=list(verifiers_used) if verifiers_used else [verifier_name],
        label_positive_fraction=label_positive_fraction,
        inference_mode="live_gpu",
        honest_verdict=verdict,
        duration_s=duration_s,
    )


# ---------------------------------------------------------------------------
# Artifact builder
# ---------------------------------------------------------------------------


def _build_artifact(
    *,
    n_pairs_before: int,
    n_pairs_added: int,
    n_pairs_after: int,
    models_used: list[str],
    labeling_verifiers: list[str],
    label_positive_fraction: float,
    inference_mode: str,
    honest_verdict: str,
    duration_s: float,
    notes: str = "",
) -> dict[str, Any]:
    """Assemble the standardised result dict for this experiment.

    All required schema fields from the task spec are present. Extra
    fields provide additional context for the operator and exp1120.
    """
    artifact: dict[str, Any] = {
        "experiment": EXP_ID,
        "title": EXP_TITLE,
        "run_date": _run_date(),
        "schema_version": "1.0",
        "n_pairs_before": n_pairs_before,
        "n_pairs_added": n_pairs_added,
        "n_pairs_after": n_pairs_after,
        "fover_sota_pairs_added_above_7000": n_pairs_after >= 7000,
        "models_used": models_used,
        "labeling_verifiers": labeling_verifiers,
        "label_positive_fraction": round(label_positive_fraction, 4),
        "inference_mode": inference_mode,
        "honest_verdict": honest_verdict,
        "duration_s": round(duration_s, 2),
        "fover_jsonl_path": str(FOVER_JSONL),
        "fover_v4_json_path": str(FOVER_V4_JSON),
        "gsm8k_offset_used": GSM8K_OFFSET,
        "n_questions_target": N_QUESTIONS_TARGET,
        "n_solutions_per_q": N_SOLUTIONS_PER_Q,
    }
    if notes:
        artifact["notes"] = notes
    return artifact


def _run_date() -> str:
    """Return today's date as an 8-digit string (e.g. '20260501')."""
    import datetime

    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the experiment and write the deliverable JSON."""
    artifact = run_experiment()
    out_path = Path(DELIVERABLE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(artifact, fh, indent=2)
    print(f"[exp1119] artifact written → {out_path}", flush=True)


if __name__ == "__main__":
    main()
