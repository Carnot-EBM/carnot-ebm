#!/usr/bin/env python3
"""Exp 3448 (P0.1 corpus builder): resumable GSM8K generation corpus.

**The decoupling fix for the P0.1 idle-timeout.** P0.1 — "does energy-based
selection/voting on continuous latents BEAT plain token-sampling
self-consistency at equal compute?" — is the single most important test in the
project and has failed to land THREE times. The most recent failure (exp3437)
was NOT scientific: it was a 1201-second idle-timeout. A single in-session job
tried to do live 35B generation over ``200 x k`` samples AND score
energy/self-consistency, ran silently past the agent's ~20-minute
wall-clock+idle budget, and produced no artifact at all.

This script does ONLY the expensive half — generating candidate solutions from
the SOTA GGUF — and writes them to an append-only, RESUMABLE corpus at
``data/p01_gsm8k_generations.jsonl``. A downstream scoring task (exp3449) then
consumes that corpus with NO live model and thus no idle-timeout risk. The
builder:

  * checkpoints ONE completed problem at a time (append immediately, never
    buffer to the end), so an interruption loses at most one problem;
  * prints a one-line progress message after EVERY problem, so the subprocess is
    never silent for the ~20 minutes that killed exp3437;
  * respects an ~18-minute wall-time budget and exits clean with whatever it
    finished (a partial corpus is progress, not failure);
  * resumes from whatever it already wrote, so the corpus accumulates across
    milestones toward the n=120 x k=6 (+greedy) target.

It captures per-token logprobs for every generation (greedy + k sampled) because
the downstream self-certainty Best-of-N selector (arXiv:2502.18581) needs the
mean chosen-token confidence. CRITICAL — how the logprobs are captured: the GGUF
is loaded with the FAST default (``logits_all=False``) and we read each generated
token's logprob directly from the C context via ``llama_get_logits_ith(ctx, -1)``
inside a low-level ``llm.generate`` loop. This was a measured engineering choice,
not an aesthetic one:

  * The high-level ``llm(prompt, logprobs=1)`` path RAISES
    ``ValueError: logprobs is not supported for models created with
    logits_all=False`` — that exception, swallowed by exp3426's bare ``except``,
    is the real root cause of its all-null sampled-candidate answers (a 0.0
    self-consistency accuracy). exp3426 loaded WITHOUT ``logits_all`` yet asked
    for logprobs, so every sampled candidate threw and came back empty.
  * Loading WITH ``logits_all=True`` (so the high-level logprobs path works)
    makes llama.cpp materialise a full-vocab (n_ctx x 262144) score buffer and
    softmax every position each step — CPU-bound, GPU idle, ~3 tok/s. A measured
    run produced ZERO completed problems in 140s. Unusable for a 120-problem
    corpus.

  The C-API extraction gives BOTH: ~80 tok/s GPU-bound generation AND exact
  per-token logprobs (greedy mean ~= -0.11, verified to match the slow
  logits_all=True reference to 3 decimals).

Spec: REQ-KONA-3448, SCENARIO-KONA-3448, SCENARIO-KONA-3448-RESUME,
SCENARIO-KONA-3448-BLOCKED.
Run: .venv/bin/python scripts/experiment_3448_p01_generation_corpus_builder_v1.py
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.inference.sota_models import resolve_cached_gguf  # noqa: E402
from carnot.phase3.energy_descent_premise import load_gsm8k_subset  # noqa: E402
from carnot.phase3.p01_generation_corpus import (  # noqa: E402
    build_corpus_row,
    completed_problem_ids,
    corpus_reproducibility_checksum,
    derive_corpus_verdict,
    make_sample,
    read_corpus_rows,
    warmup_self_consistency_check,
)

DELIVERABLE_PATH = "results/experiment_3448_p01_generation_corpus_builder_v1.json"
CORPUS_OUT_PATH = project_root / "data" / "p01_gsm8k_generations.jsonl"
SOURCE_CORPUS_PATH = project_root / "data" / "research" / "gsm8k_adversarial_281.jsonl"
# Fastest SOTA MoE (gemma-4-26B-A4B, ~4B active) per the task spec — quickest
# headline-eligible local model, so the per-problem wall time stays bounded.
MODEL_HF_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"

# Tunables (env-overridable so a smoke run is cheap; defaults are the real run).
N_PROBLEMS = int(os.environ.get("EXP3448_N", "120"))
K_SAMPLES = int(os.environ.get("EXP3448_K", "6"))
MAX_TOKENS = int(os.environ.get("EXP3448_MAXTOK", "320"))
SEED = int(os.environ.get("EXP3448_SEED", "20260530"))
SAMPLE_TEMPERATURE = float(os.environ.get("EXP3448_TEMP", "0.8"))
# ~18-minute wall budget; we stop generating new problems once we cross it and
# finalize cleanly. Override smaller for a quick verification run.
WALL_BUDGET_S = float(os.environ.get("EXP3448_WALL_BUDGET_S", "1020"))

PROMPT_TEMPLATE = (
    "Solve this math word problem. Show brief reasoning, then on the final line "
    "write the answer as: #### <number>\n\nProblem: {q}\n\nSolution:"
)


def _field_principles() -> dict[str, str]:
    """Per-field principle annotations (CLAUDE.md Principle-Annotated Fields)."""
    return {
        "honest_verdict": "Terminal verdict must start with complete:/success:/"
        "passed:/shipped_.",
        "inference_substrate": "live_llm_inference: candidates really load + run the GGUF.",
        "corpus_path": "data/p01_gsm8k_generations.jsonl — the cached corpus exp3449 consumes.",
        "n_problems_completed": "problems with a full set of generations in the corpus "
        "after this run.",
        "n_problems_target": "the target (120); n_completed < target -> the builder resumes "
        "next milestone.",
        "k_samples": "sampled generations per problem (6); the matched-compute budget for "
        "the scoring task.",
        "per_sample_logprobs_captured": "boolean: mean-token confidence stored per "
        "generation, required for self-certainty BoN downstream.",
        "self_consistency_non_degenerate": "warm-up gate: SC accuracy >= greedy AND > 0.30 "
        "— proves the per-sample answer extraction works (the exp3426 0.0-bug guard).",
        "warmup_self_consistency_accuracy": "majority-vote accuracy on the warm-up batch.",
        "warmup_greedy_accuracy": "greedy accuracy on the warm-up batch.",
        "model_specs": "the actual GGUF invoked (gemma-4-26B-A4B-it-GGUF).",
        "random_seed": "determinism precondition for reproducibility.",
        "reproducibility_checksum": "content hash of corpus split + model + seed.",
        "duration_s": "real live MoE generation takes wall time; 60s floor — a sub-60s "
        "duration is the fabrication signal.",
    }


_STOP_STRINGS = ("\nProblem:", "\n\nProblem")


def _last_token_logprob(ctx, n_vocab: int, token: int) -> float:
    """log P(token) from the context's final-position logits (logits_all-free).

    ``llama_get_logits_ith(ctx, -1)`` returns the logits of the LAST decoded
    position — always computed by llama.cpp even when ``logits_all=False`` — so we
    can score the just-sampled token without the full-vocab-every-position tax.
    We subtract a numerically-stable log-sum-exp to convert the raw logit to a
    proper log-probability (the self-certainty proxy the scoring task needs).
    """
    import llama_cpp  # noqa: PLC0415 — bound lazily; only the GPU path needs it
    import numpy as np  # noqa: PLC0415 — heavy import kept local to the GPU path

    ptr = llama_cpp.llama_get_logits_ith(ctx, -1)
    logits = np.ctypeslib.as_array(
        ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)), shape=(n_vocab,)
    ).astype(np.float64)
    m = float(logits.max())
    lse = m + math.log(float(np.exp(logits - m).sum()))
    return float(logits[token] - lse)


def _generate(llm, prompt: str, *, temperature: float, seed: int):
    """One bounded llama.cpp generation. Returns (text, token_logprobs).

    Uses the low-level ``llm.generate`` token stream and reads each chosen
    token's logprob straight from the C context (see the module docstring for the
    full rationale: the high-level ``logprobs=1`` path raises with the fast load,
    and ``logits_all=True`` is ~25x slower). The RNG is reseeded per call so a
    given (prompt, seed) is reproducible while distinct sample seeds give variety.
    Generation stops at EOS/EOT, the ``MAX_TOKENS`` cap, or a stop string (whose
    text is trimmed). Returns ('', []) on any failure so a single bad generation
    is stored as a no-answer miss rather than crashing the resumable run.
    """
    try:
        ctx = getattr(llm, "ctx", None) or getattr(getattr(llm, "_ctx", None), "ctx", None)
        try:
            n_vocab = llm.n_vocab()
        except Exception:  # pragma: no cover - binding-version-dependent
            n_vocab = llm._model.n_vocab()

        llm.reset()
        if hasattr(llm, "set_seed"):
            llm.set_seed(seed)
        tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
        eos = {llm.token_eos()}
        try:
            eos.add(llm.token_eot())
        except Exception:  # pragma: no cover - some models lack a distinct EOT
            pass

        out_tokens: list[int] = []
        token_logprobs: list[float] = []
        for token in llm.generate(
            tokens,
            temp=temperature,
            top_p=0.95 if temperature > 0 else 1.0,
            top_k=0,
        ):
            token_logprobs.append(_last_token_logprob(ctx, n_vocab, token))
            out_tokens.append(token)
            if token in eos or len(out_tokens) >= MAX_TOKENS:
                break
            text = llm.detokenize(out_tokens).decode("utf-8", "ignore")
            for stop in _STOP_STRINGS:
                idx = text.find(stop)
                if idx != -1:
                    return text[:idx], token_logprobs
        return llm.detokenize(out_tokens).decode("utf-8", "ignore"), token_logprobs
    except Exception:  # pragma: no cover - inference-environment-dependent
        return "", []


def main() -> None:
    tmpl = ExperimentTemplate(
        exp_id=3448,
        title="P0.1 resumable GSM8K generation-corpus builder",
        deliverable=DELIVERABLE_PATH,
        requires_gpu=True,
        seed=SEED,
    )
    tmpl.setup()
    start = time.time()
    principles = _field_principles()

    def _emit(payload: dict, status: str) -> None:
        artifact = tmpl.build_result(payload, status=status)
        Path(DELIVERABLE_PATH).write_text(json.dumps(artifact, indent=2))
        tmpl.assert_deliverable_written()

    def _emit_block(verdict: str, detail: str, pre: list[dict]) -> None:
        _emit(
            {
                "honest_verdict": verdict,
                "inference_substrate": "live_llm_inference",
                "corpus_path": str(CORPUS_OUT_PATH.relative_to(project_root)),
                "block_detail": detail,
                "preconditions_checked": pre,
                "random_seed": SEED,
                "duration_s": round(time.time() - start, 3),
                "field_provenance": principles,
            },
            "blocked",
        )
        print(f"BLOCKED: {verdict} — {detail}")

    # ----- Step 0: PRECONDITIONS (before any generation) -----
    pre: list[dict] = []

    # (a) CUDA
    try:
        import torch  # noqa: PLC0415

        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False
    pre.append({"resource": "cuda", "available": cuda_ok})
    if not cuda_ok:
        _emit_block("blocked_cuda_unavailable", "torch.cuda.is_available() is False", pre)
        return

    # (c) real GSM8K corpus with integer labels (cheap; check before model load)
    corpus_ok = SOURCE_CORPUS_PATH.exists()
    pre.append({"resource": "real_task_corpus", "available": corpus_ok})
    if not corpus_ok:
        _emit_block(
            "blocked_real_task_corpus_missing",
            f"source corpus not found at {SOURCE_CORPUS_PATH}",
            pre,
        )
        return

    # (b) SOTA GGUF loads via the GGUF path (embedded tokenizer; NOT AutoTokenizer)
    model_path = resolve_cached_gguf(MODEL_HF_ID)
    gguf_ok = model_path is not None and os.path.exists(model_path)
    if gguf_ok:
        try:
            import llama_cpp  # noqa: PLC0415

            probe = llama_cpp.Llama(model_path=model_path, vocab_only=True, verbose=False)
            probe.tokenize(b"What is 2+2?")
            del probe
        except Exception as exc:  # pragma: no cover - inference-environment-dependent
            gguf_ok = False
            pre.append({"resource": "sota_gguf_tokenizer", "available": False})
            _emit_block(
                "blocked_sota_gguf_tokenizer_unavailable",
                f"GGUF embedded-tokenizer probe failed for {MODEL_HF_ID}: {exc}",
                pre,
            )
            return
    pre.append({"resource": "sota_gguf_tokenizer", "available": gguf_ok})
    if not gguf_ok:
        _emit_block(
            "blocked_sota_gguf_tokenizer_unavailable",
            f"GGUF for {MODEL_HF_ID} not cached",
            pre,
        )
        return

    # ----- Step 1: load the fixed, documented GSM8K split -----
    problems = load_gsm8k_subset(SOURCE_CORPUS_PATH, n=N_PROBLEMS, seed=SEED)
    problem_order = [p.problem_id for p in problems]

    # ----- Step 2: RESUME — skip already-completed problems -----
    done_ids = completed_problem_ids(CORPUS_OUT_PATH, k_samples=K_SAMPLES)
    remaining = [p for p in problems if p.problem_id not in done_ids]
    print(
        f"[exp3448] target n={N_PROBLEMS} k={K_SAMPLES}; "
        f"{len(done_ids)} already complete on disk, {len(remaining)} remaining"
    )

    # Load the full model only if there is work to do (resume may find all done).
    llm = None
    if remaining:
        try:
            import llama_cpp  # noqa: PLC0415

            # logits_all is LEFT AT ITS FAST DEFAULT (False). We do NOT use the
            # high-level logprobs path (it raises with this load); instead
            # ``_generate`` reads each token's logprob from the C context via
            # ``llama_get_logits_ith(ctx, -1)``. logits_all=True would make the
            # high-level path work but forces a full-vocab softmax over all
            # positions every step — a CPU-bound pathology (GPU idle, ~3 tok/s)
            # that produced ZERO completed problems in 140s of an earlier run.
            llm = llama_cpp.Llama(
                model_path=model_path,
                n_gpu_layers=-1,
                n_ctx=2048,
                seed=SEED,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - inference-environment-dependent
            _emit_block(
                "blocked_sota_gguf_tokenizer_unavailable",
                f"llama.cpp failed to load {MODEL_HF_ID}: {exc}",
                pre,
            )
            return

    # ----- Steps 3-7: generate + checkpoint each remaining problem -----
    CORPUS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_done_this_run = 0
    budget_hit = False
    for problem in remaining:
        # Step 7: wall-time budget — stop launching new problems near the ceiling.
        if time.time() - start >= WALL_BUDGET_S:
            budget_hit = True
            print(
                f"[exp3448] wall-time budget {WALL_BUDGET_S:.0f}s reached; "
                f"finalizing with a partial corpus (progress, not failure)"
            )
            break

        prompt = PROMPT_TEMPLATE.format(q=problem.question)

        # 1 greedy generation (temp 0) + k sampled generations (temp ~0.8).
        g_text, g_lps = _generate(llm, prompt, temperature=0.0, seed=SEED)
        greedy = make_sample(g_text, g_lps)
        samples = []
        for k in range(K_SAMPLES):
            s_text, s_lps = _generate(
                llm, prompt, temperature=SAMPLE_TEMPERATURE, seed=SEED + 1000 * (k + 1)
            )
            samples.append(make_sample(s_text, s_lps))

        row = build_corpus_row(
            problem_id=problem.problem_id,
            question=problem.question,
            gold=problem.answer,
            greedy=greedy,
            samples=samples,
            temperature=SAMPLE_TEMPERATURE,
        )
        # Step 4: CHECKPOINT — append immediately so an interruption loses <=1 problem.
        with open(CORPUS_OUT_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        n_done_this_run += 1

        # Step 5: PROGRESS — one line per problem (kills the idle-timeout risk).
        n_total_done = len(done_ids) + n_done_this_run
        elapsed = int(time.time() - start)
        print(
            f"[exp3448] problem {n_total_done}/{N_PROBLEMS} done "
            f"({K_SAMPLES} samples + greedy), elapsed {elapsed}s"
        )

    # ----- Step 6 + 8: warm-up self-check + artifact over the full corpus -----
    all_rows = read_corpus_rows(CORPUS_OUT_PATH)
    # Only count rows for problems in THIS split (a corpus could hold stale ids).
    split_ids = set(problem_order)
    corpus_rows = [r for r in all_rows if str(r.get("problem_id")) in split_ids]
    n_completed = len({str(r["problem_id"]) for r in corpus_rows if "problem_id" in r})

    warmup = warmup_self_consistency_check(corpus_rows)
    verdict = derive_corpus_verdict(n_completed, N_PROBLEMS)
    checksum = corpus_reproducibility_checksum(
        corpus_path=SOURCE_CORPUS_PATH,
        model_path=model_path,
        seed=SEED,
        n_target=N_PROBLEMS,
        k_samples=K_SAMPLES,
    )
    # per_sample_logprobs_captured: did every completed row store mean confidence
    # for each sampled generation? (the field the scoring task depends on)
    logprobs_captured = bool(corpus_rows) and all(
        all(s.get("mean_token_logprob") is not None for s in (r.get("samples") or []))
        for r in corpus_rows
    )

    payload = {
        "honest_verdict": verdict,
        "inference_substrate": "live_llm_inference",
        "task_name": "GSM8K (original questions, exp281 corpus), held-out shuffled split",
        "corpus_path": str(CORPUS_OUT_PATH.relative_to(project_root)),
        "n_problems_completed": n_completed,
        "n_problems_target": N_PROBLEMS,
        "n_problems_this_run": n_done_this_run,
        "k_samples": K_SAMPLES,
        "max_tokens": MAX_TOKENS,
        "sample_temperature": SAMPLE_TEMPERATURE,
        "per_sample_logprobs_captured": logprobs_captured,
        "self_consistency_non_degenerate": warmup.non_degenerate,
        "warmup_self_consistency_accuracy": warmup.self_consistency_accuracy,
        "warmup_greedy_accuracy": warmup.greedy_accuracy,
        "warmup_n_problems": warmup.n_problems,
        "warmup_extraction_examples": warmup.examples,
        "wall_budget_s": WALL_BUDGET_S,
        "wall_budget_hit": budget_hit,
        "acceptance_gate_g1_corpus_usable": (n_completed >= 30) and logprobs_captured,
        "random_seed": SEED,
        "reproducibility_checksum": checksum,
        "duration_s": round(time.time() - start, 3),
        "preconditions_checked": pre,
        "model_specs": [
            {
                "name": "Gemma4-26B-A4B-it",
                "hf_id": MODEL_HF_ID,
                "model_path": model_path,
                "quantization": "Q4_K_M",
            }
        ],
        "field_provenance": principles,
    }
    _emit(payload, "success")
    print(
        f"DONE: {verdict}\n"
        f"  n_completed={n_completed}/{N_PROBLEMS} (this run +{n_done_this_run}) "
        f"k={K_SAMPLES} logprobs_captured={logprobs_captured}\n"
        f"  warmup: SC={warmup.self_consistency_accuracy} greedy={warmup.greedy_accuracy} "
        f"non_degenerate={warmup.non_degenerate} (n={warmup.n_problems})\n"
        f"  dur={payload['duration_s']}s"
    )


if __name__ == "__main__":
    main()
