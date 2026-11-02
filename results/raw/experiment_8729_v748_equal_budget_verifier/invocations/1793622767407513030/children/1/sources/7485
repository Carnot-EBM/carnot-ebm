#!/usr/bin/env python3
"""Exp 3471 (P0.1 headroom corpus): hard-math generation corpus with per-step traces.

**Why this script exists (plain-language summary).** P0.1 — "does energy-based
selection/voting BEAT plain self-consistency at equal compute?" — kept coming back
a TIE, and the reason was the benchmark, not the selector. On GSM8K a strong model
gets self-consistency (SC) accuracy ~0.91 — almost a CEILING — so there is no
accuracy left for ANY re-ranker to recover on top of the majority vote (exp3460
measured the trained-energy vote tying SC exactly). A selector can only be shown
to help where SC has HEADROOM: where SC is materially below 1.0.

The process-reward literature (arXiv:2602.11570 "PRIME", +8-9% on AIME from
process-aware verification) shows the regime where verifier selection beats SC is
HARD math, scored as a PROCESS reward over the per-STEP reasoning trace. This
script builds exactly that substrate: a NEW cached corpus on a hard-math benchmark
(MATH Level 5) whose SC lands in the HEADROOM band [0.4, 0.7], capturing 1 greedy +
k=6 sampled generations per problem, EACH carrying:

  * the raw text,
  * the extracted ``\\boxed{}`` final answer + a correctness label,
  * the mean-token logprob (self-certainty proxy), and
  * a parsed list of discrete reasoning STEPS — the NEW capability over the GSM8K
    builders, so the FoVer step-error verifier (the 0.9131 ensemble) can be scored
    as a PROCESS reward by the downstream crux (exp3472/3473/3475).

It reuses the proven decoupled-generation discipline (generation ONLY — no scoring,
so it cannot idle-timeout the way the monolithic exp3437 did): it RESUMES from
whatever is already on disk, appends ONE completed problem at a time, prints a
progress line after EVERY problem (so the subprocess is never silent), respects a
~22-minute wall-time budget, and exits clean (code 0) with whatever it finished. It
reuses exp3448's exact C-API logprob-extraction generation primitive (same fast
GGUF load, same per-token logprob capture) so every row is homogeneous with the
GSM8K corpus's confidence signal; only the MATH prompt, the higher max-token cap,
the math answer/step parsing, and the headroom-band self-check are new.

The pure verdict/gate/parse logic lives in
:mod:`carnot.phase3.p01_headroom_corpus` (math extraction, step parsing, the
[0.4,0.7] band check, the four-band verdict, the two gates) and is unit-tested
without a GPU.

Spec: REQ-KONA-3471, SCENARIO-KONA-3471, SCENARIO-KONA-3471-RESUME,
SCENARIO-KONA-3471-NO-HEADROOM.
Run: .venv/bin/python scripts/experiment_3471_p01_headroom_corpus_builder_hard_math_v1.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

DELIVERABLE_PATH = (
    "results/experiment_3471_p01_headroom_corpus_builder_hard_math_v1.json"
)
CORPUS_OUT_PATH = project_root / "data" / "p01_hardmath_generations.jsonl"
# Fastest SOTA MoE (gemma-4-26B-A4B, ~4B active) — the SAME model the .318/.319
# GSM8K builders used, so the two corpora are comparable.
MODEL_HF_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"

# Tunables (env-overridable so a smoke run is cheap; defaults are the real run).
N_TARGET = int(os.environ.get("EXP3471_N", "80"))
K_SAMPLES = int(os.environ.get("EXP3471_K", "6"))
# Hard math needs a longer chain than GSM8K; cap at 512.
MAX_TOKENS = int(os.environ.get("EXP3471_MAXTOK", "512"))
SEED = int(os.environ.get("EXP3471_SEED", "20260530"))
SAMPLE_TEMPERATURE = float(os.environ.get("EXP3471_TEMP", "0.8"))
# ~22-minute wall budget (per the task spec); stop launching new problems once we
# cross it and finalize cleanly. Override smaller for a quick verification run.
WALL_BUDGET_S = float(os.environ.get("EXP3471_WALL_BUDGET_S", "1320"))
# MATH difficulty level(s). Default Level 5 (hardest available -> most headroom).
# A re-invocation can switch to "4,5" or "4" if the first split lands out of band.
_LEVEL_ENV = os.environ.get("EXP3471_LEVELS", "5")
LEVELS = {f"Level {lv.strip()}" for lv in _LEVEL_ENV.split(",") if lv.strip()}
# The MATH subjects cached locally (each contributes its test split).
MATH_SUBJECTS = os.environ.get(
    "EXP3471_SUBJECTS", "algebra,number_theory,prealgebra"
).split(",")
# Log a running SC estimate once this many problems are on disk (warm-up probe).
WARMUP_PROBE_N = int(os.environ.get("EXP3471_WARMUP_PROBE_N", "10"))

# Generation reuses exp3448's exact primitive, which reads its MAX_TOKENS from its
# OWN module global bound at import. Export the higher hard-math cap BEFORE
# importing exp3448 so its ``_generate`` honours 512, not the GSM8K default 320.
os.environ["EXP3448_MAXTOK"] = str(MAX_TOKENS)

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from scripts.experiment_3448_p01_generation_corpus_builder_v1 import (  # noqa: E402
    _generate,
)
from carnot.inference.sota_models import resolve_cached_gguf  # noqa: E402
from carnot.phase3.p01_headroom_corpus import (  # noqa: E402
    DEFAULT_BENCHMARK_ID,
    build_headroom_row,
    build_math_problems,
    corpus_problem_ids,
    derive_headroom_verdict,
    headroom_acceptance_gates,
    headroom_reproducibility_checksum,
    headroom_warmup_check,
    make_headroom_sample,
    read_corpus_rows,
)

# A MATH-specific prompt: ask for step-by-step reasoning AND a \boxed{} answer, so
# both the per-step trace parser and the answer extractor have something to read.
PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. Show your reasoning, with one "
    "step per line, then give the final answer inside \\boxed{{}}.\n\n"
    "Problem: {q}\n\nSolution:"
)


def _field_principles() -> dict[str, str]:
    """Per-field principle annotations (CLAUDE.md Principle-Annotated Fields)."""
    return {
        "honest_verdict": "Terminal verdict must start with complete:/success:/"
        "passed:/shipped_.",
        "inference_substrate": "live_llm_inference: candidates really load + run the GGUF.",
        "corpus_path": "data/p01_hardmath_generations.jsonl — the headroom corpus "
        "exp3472/3473/3475 consume.",
        "benchmark_id": "the exact hard-math dataset id + split + level + seed — the "
        "new substrate (unlike GSM8K's 0.908 ceiling).",
        "n_problems_completed": "problems with a full generation set after this run.",
        "n_problems_target": "the target (>=80 for headline-eligibility); n < target "
        "-> resume next milestone.",
        "n_problems_added_this_run": "problems newly generated this invocation (proves "
        "resume worked, distinct from total).",
        "k_samples": "sampled generations per problem (6) — the matched-compute budget.",
        "per_step_traces_captured": "boolean: each generation carries a parsed step list "
        "for PROCESS-reward scoring (the .320 new capability).",
        "per_sample_logprobs_captured": "boolean: mean-token confidence stored per "
        "generation.",
        "warmup_self_consistency_accuracy": "majority-vote accuracy over the corpus — "
        "MUST land in the headroom band for the corpus to be useful.",
        "self_consistency_in_headroom_band": "boolean: SC in [0.4,0.7] — the precondition "
        "that makes P0.1 testable (unlike GSM8K's 0.908 ceiling).",
        "warmup_greedy_accuracy": "greedy accuracy over the corpus.",
        "model_specs": "the actual GGUF invoked (gemma-4-26B-A4B-it-GGUF).",
        "random_seed": "determinism precondition for reproducibility.",
        "reproducibility_checksum": "content hash of benchmark split + model + seed.",
        "duration_s": "real live MoE generation takes wall time; 60s floor — sub-60s is "
        "the fabrication signal.",
    }


def _load_math_records() -> list[dict]:
    """Download the cached MATH subjects' test splits as plain record dicts.

    Kept here (not in the unit-tested module) because it depends on ``datasets``
    and the on-disk HuggingFace cache. Each record carries ``problem`` / ``level``
    / ``type`` / ``solution`` — exactly what :func:`build_math_problems` filters.
    Subjects that fail to load are skipped (a partial benchmark is still usable);
    an empty result is treated as "benchmark unavailable" by the caller.
    """
    from datasets import load_dataset  # noqa: PLC0415 — heavy import, GPU-free path

    records: list[dict] = []
    for subject in MATH_SUBJECTS:
        subject = subject.strip()
        if not subject:
            continue
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", subject, split="test")
        except Exception as exc:  # pragma: no cover - dataset-cache-dependent
            print(f"[exp3471] WARN: subject {subject!r} failed to load: {exc}")
            continue
        for row in ds:
            records.append(
                {
                    "problem": row.get("problem"),
                    "level": row.get("level"),
                    "type": row.get("type"),
                    "solution": row.get("solution"),
                }
            )
    return records


def main() -> None:
    tmpl = ExperimentTemplate(
        exp_id=3471,
        title="P0.1 hard-math headroom corpus builder with per-step traces (v1)",
        deliverable=DELIVERABLE_PATH,
        requires_gpu=True,
        seed=SEED,
    )
    tmpl.setup()
    start = time.time()
    principles = _field_principles()
    benchmark_id = (
        f"{DEFAULT_BENCHMARK_ID} | subjects={'+'.join(s.strip() for s in MATH_SUBJECTS)}"
        f" | levels={'+'.join(sorted(LEVELS))} | seed={SEED}"
    )

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
                "benchmark_id": benchmark_id,
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

    # (a) CUDA available.
    try:
        import torch  # noqa: PLC0415

        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False
    pre.append({"resource": "cuda", "available": cuda_ok})
    if not cuda_ok:
        _emit_block("complete: blocked_cuda_unavailable", "torch.cuda.is_available() is False", pre)
        return

    # (b) SOTA GGUF loads via the GGUF path (embedded tokenizer; NOT AutoTokenizer
    #     on a -GGUF repo id, per the 2026-05-29 GGUF tokenizer rule).
    model_path = resolve_cached_gguf(MODEL_HF_ID)
    gguf_ok = model_path is not None and os.path.exists(model_path)
    if gguf_ok:
        try:
            import llama_cpp  # noqa: PLC0415

            probe = llama_cpp.Llama(model_path=model_path, vocab_only=True, verbose=False)
            probe.tokenize(b"What is 2+2?")
            del probe
        except Exception as exc:  # pragma: no cover - inference-environment-dependent
            pre.append({"resource": "sota_gguf_tokenizer", "available": False})
            _emit_block(
                "complete: blocked_sota_gguf_tokenizer_unavailable",
                f"GGUF embedded-tokenizer probe failed for {MODEL_HF_ID}: {exc}",
                pre,
            )
            return
    pre.append({"resource": "sota_gguf_tokenizer", "available": gguf_ok})
    if not gguf_ok:
        _emit_block(
            "complete: blocked_sota_gguf_tokenizer_unavailable",
            f"GGUF for {MODEL_HF_ID} not cached",
            pre,
        )
        return

    # ----- Step 1: load the HARD-MATH split (MATH Level 5) -----
    records = _load_math_records()
    problems = build_math_problems(records, levels=LEVELS, n=N_TARGET, seed=SEED)
    benchmark_ok = len(problems) > 0
    pre.append({"resource": "hard_math_benchmark", "available": benchmark_ok})
    if not benchmark_ok:
        _emit_block(
            "complete: blocked_hard_math_benchmark_unavailable",
            f"no MATH problems at levels {sorted(LEVELS)} from subjects {MATH_SUBJECTS}",
            pre,
        )
        return
    problem_order = [p.problem_id for p in problems]
    split_ids = set(problem_order)
    print(
        f"[exp3471] benchmark={benchmark_id}; {len(problems)} problems loaded "
        f"at levels {sorted(LEVELS)}"
    )

    # ----- Step 3: RESUME — skip already-completed problems -----
    done_ids = corpus_problem_ids(CORPUS_OUT_PATH, k_samples=K_SAMPLES)
    n_prior = len(done_ids & split_ids)
    remaining = [p for p in problems if p.problem_id not in done_ids]
    print(
        f"[exp3471] target n={N_TARGET} k={K_SAMPLES}; "
        f"{n_prior} already complete on disk, {len(remaining)} remaining"
    )

    # Load the full model only if there is work to do (resume may find all done).
    llm = None
    if remaining:
        try:
            import llama_cpp  # noqa: PLC0415

            # logits_all left at its FAST default (False); exp3448's ``_generate``
            # reads per-token logprobs from the C context (see its module docstring).
            llm = llama_cpp.Llama(
                model_path=model_path,
                n_gpu_layers=-1,
                n_ctx=4096,
                seed=SEED,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - inference-environment-dependent
            _emit_block(
                "complete: blocked_sota_gguf_tokenizer_unavailable",
                f"llama.cpp failed to load {MODEL_HF_ID}: {exc}",
                pre,
            )
            return

    # ----- Steps 4-6: generate + checkpoint each remaining problem -----
    CORPUS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_done_this_run = 0
    budget_hit = False
    warmup_logged = False
    for problem in remaining:
        # Step 7: wall-time budget — stop launching new problems near the ceiling.
        if time.time() - start >= WALL_BUDGET_S:
            budget_hit = True
            print(
                f"[exp3471] wall-time budget {WALL_BUDGET_S:.0f}s reached; "
                f"finalizing with a partial corpus (progress, not failure)"
            )
            break

        prompt = PROMPT_TEMPLATE.format(q=problem.question)

        # Step 4: 1 greedy (temp 0) + k sampled (temp ~0.8) generations.
        g_text, g_lps = _generate(llm, prompt, temperature=0.0, seed=SEED)
        greedy = make_headroom_sample(g_text, g_lps)
        samples = []
        for k in range(K_SAMPLES):
            s_text, s_lps = _generate(
                llm, prompt, temperature=SAMPLE_TEMPERATURE, seed=SEED + 1000 * (k + 1)
            )
            samples.append(make_headroom_sample(s_text, s_lps))

        row = build_headroom_row(
            problem_id=problem.problem_id,
            question=problem.question,
            gold=problem.answer,
            level=problem.level,
            greedy=greedy,
            samples=samples,
            temperature=SAMPLE_TEMPERATURE,
        )
        # Step 5: CHECKPOINT — append immediately so an interruption loses <=1 problem.
        with open(CORPUS_OUT_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        n_done_this_run += 1

        # Step 6: PROGRESS — one line per problem (kills the idle-timeout risk).
        n_total_done = n_prior + n_done_this_run
        elapsed = int(time.time() - start)
        print(
            f"[exp3471] problem {n_total_done}/{N_TARGET} done "
            f"({K_SAMPLES} samples + greedy, {len(greedy.steps)} greedy steps), "
            f"elapsed {elapsed}s"
        )

        # Step 2 (warm-up probe, folded in): once enough problems are on disk for
        # this split, log the running SC headroom band so a too-easy/too-hard
        # split is visible early, well before the wall-time budget.
        if not warmup_logged and n_total_done >= WARMUP_PROBE_N:
            probe_rows = [
                r
                for r in read_corpus_rows(CORPUS_OUT_PATH)
                if str(r.get("problem_id")) in split_ids
            ]
            probe = headroom_warmup_check(probe_rows)
            warmup_logged = True
            print(
                f"[exp3471] WARM-UP SC probe @ n={probe.n_problems}: "
                f"SC={probe.self_consistency_accuracy} "
                f"in_band={probe.in_band} (target band [0.4,0.7])"
            )

    # ----- Steps 8-9: full-corpus warm-up self-check + artifact -----
    all_rows = read_corpus_rows(CORPUS_OUT_PATH)
    # Only count rows for problems in THIS split (a corpus could hold stale ids).
    corpus_rows = [r for r in all_rows if str(r.get("problem_id")) in split_ids]
    n_completed = len({str(r["problem_id"]) for r in corpus_rows if "problem_id" in r})
    n_added = max(0, n_completed - n_prior)

    warmup = headroom_warmup_check(corpus_rows)
    verdict = derive_headroom_verdict(
        n_completed, warmup.self_consistency_accuracy, warmup.in_band
    )
    checksum = headroom_reproducibility_checksum(
        benchmark_id=benchmark_id,
        model_path=model_path,
        seed=SEED,
        n_target=N_TARGET,
        k_samples=K_SAMPLES,
        levels=LEVELS,
    )
    # per_sample_logprobs_captured: did every completed row store a mean confidence
    # for each sampled generation? (the field the self-certainty selector needs)
    logprobs_captured = bool(corpus_rows) and all(
        all(s.get("mean_token_logprob") is not None for s in (r.get("samples") or []))
        for r in corpus_rows
    )
    # per_step_traces_captured: does every completed row carry a parsed step list on
    # the greedy generation AND every sample? (the NEW .320 process-reward field)
    per_step_captured = bool(corpus_rows) and all(
        isinstance((r.get("greedy") or {}).get("steps"), list)
        and all(isinstance(s.get("steps"), list) for s in (r.get("samples") or []))
        for r in corpus_rows
    )
    gates = headroom_acceptance_gates(warmup.in_band, n_completed, per_step_captured)

    payload = {
        "honest_verdict": verdict,
        "inference_substrate": "live_llm_inference",
        "task_name": "MATH (Hendrycks) Level 5 hard-math split, held-out shuffled",
        "benchmark_id": benchmark_id,
        "corpus_path": str(CORPUS_OUT_PATH.relative_to(project_root)),
        "n_problems_completed": n_completed,
        "n_problems_target": N_TARGET,
        "n_problems_added_this_run": n_added,
        "n_problems_prior": n_prior,
        "k_samples": K_SAMPLES,
        "max_tokens": MAX_TOKENS,
        "sample_temperature": SAMPLE_TEMPERATURE,
        "levels": sorted(LEVELS),
        "per_step_traces_captured": per_step_captured,
        "per_sample_logprobs_captured": logprobs_captured,
        "warmup_self_consistency_accuracy": warmup.self_consistency_accuracy,
        "self_consistency_in_headroom_band": warmup.in_band,
        "warmup_greedy_accuracy": warmup.greedy_accuracy,
        "warmup_n_problems": warmup.n_problems,
        "warmup_extraction_examples": warmup.examples,
        "wall_budget_s": WALL_BUDGET_S,
        "wall_budget_hit": budget_hit,
        "acceptance_gate_g1_headroom_confirmed": gates["g1_headroom_confirmed"],
        "acceptance_gate_g2_scorable": gates["g2_scorable"],
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
        f"  n_completed={n_completed}/{N_TARGET} (this run +{n_added}, prior {n_prior}) "
        f"k={K_SAMPLES}\n"
        f"  per_step_traces={per_step_captured} logprobs={logprobs_captured}\n"
        f"  gates: G1_headroom_confirmed={gates['g1_headroom_confirmed']} "
        f"G2_scorable={gates['g2_scorable']}\n"
        f"  warmup: SC={warmup.self_consistency_accuracy} greedy={warmup.greedy_accuracy} "
        f"in_band={warmup.in_band} (n={warmup.n_problems})\n"
        f"  dur={payload['duration_s']}s"
    )


if __name__ == "__main__":
    main()
