#!/usr/bin/env python3
"""Exp 3459 (P0.1 corpus extend v2): resume exp3448's corpus toward n=120.

**Why this script exists (plain-language summary).** P0.1 — "does energy-based
selection/voting BEAT plain token-sampling self-consistency at equal compute?" —
is the single most important test in the project. Exp 3448 built the first,
RESUMABLE half of its evidence: it generated candidate GSM8K solutions from the
SOTA GGUF and appended them to ``data/p01_gsm8k_generations.jsonl`` one completed
problem at a time, then exited clean on its ~18-minute wall-time budget with
``n=47/120`` problems. That clean exit was BY DESIGN — a partial corpus is
progress, not failure — so the builder is meant to be RE-INVOKED across milestones
until the corpus is large enough.

This is that re-invocation (v2). It does ONLY generation (no scoring, so it cannot
idle-timeout the way the monolithic exp3437 did), RESUMES from whatever is already
on disk (reads the completed problem ids and SKIPS them), and extends the corpus
toward the n=120 target — crossing the ``>=80`` HEADLINE-eligible threshold the
downstream crux (exp3460) needs. It reuses exp3448's exact generation primitives
(same prompt, same C-API logprob extraction, same stop strings) and the SAME
GSM8K split, seed, model, and sampling parameters exp3448 documented, so every
extended row is HOMOGENEOUS with the originals. It:

  * checkpoints ONE completed problem at a time (append immediately), so an
    interruption loses at most one problem;
  * prints a one-line progress message after EVERY problem, so the subprocess is
    never silent for the ~20 minutes that killed exp3437;
  * respects an ~18-minute wall-time budget and exits clean (code 0) with whatever
    it finished;
  * reports ``n_problems_added_this_run`` (distinct from the running total) so the
    artifact proves the resume actually generated new work.

The pure verdict/gate logic lives in :mod:`carnot.phase3.p01_corpus_extend` (v2
bands + G1/G2 gates) and the generation/resume machinery in
:mod:`carnot.phase3.p01_generation_corpus`; both are unit-tested without a GPU.

Spec: REQ-KONA-3459, SCENARIO-KONA-3459, SCENARIO-KONA-3459-RESUME-MONOTONE.
Run: .venv/bin/python scripts/experiment_3459_p01_generation_corpus_extend_to_120_v2.py
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
    "results/experiment_3459_p01_generation_corpus_extend_to_120_v2.json"
)
EXP3448_ARTIFACT = (
    project_root
    / "results"
    / "experiment_3448_p01_generation_corpus_builder_v1.json"
)
CORPUS_OUT_PATH = project_root / "data" / "p01_gsm8k_generations.jsonl"
SOURCE_CORPUS_PATH = (
    project_root / "data" / "research" / "gsm8k_adversarial_281.jsonl"
)


def _read_exp3448_params() -> dict:
    """Read the homogeneity-defining params from the exp3448 artifact.

    The extended rows MUST be generated with the SAME GSM8K split (seed), model,
    sample count, sampling temperature, max-tokens cap, and target as exp3448, or
    the corpus stops being a single comparable population. We read those values
    from exp3448's own artifact (the authoritative record of what it ran) and fall
    back to exp3448's documented defaults only if the artifact is unreadable.
    """
    defaults = {
        "random_seed": 20260530,
        "k_samples": 6,
        "max_tokens": 320,
        "sample_temperature": 0.8,
        "n_problems_target": 120,
        "model_hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
    }
    try:
        art = json.loads(EXP3448_ARTIFACT.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - artifact-presence-dependent
        return defaults
    specs = art.get("model_specs") or [{}]
    return {
        "random_seed": int(art.get("random_seed", defaults["random_seed"])),
        "k_samples": int(art.get("k_samples", defaults["k_samples"])),
        "max_tokens": int(art.get("max_tokens", defaults["max_tokens"])),
        "sample_temperature": float(
            art.get("sample_temperature", defaults["sample_temperature"])
        ),
        "n_problems_target": int(
            art.get("n_problems_target", defaults["n_problems_target"])
        ),
        "model_hf_id": specs[0].get("hf_id", defaults["model_hf_id"]),
    }


_P = _read_exp3448_params()
SEED = int(os.environ.get("EXP3459_SEED", str(_P["random_seed"])))
K_SAMPLES = int(os.environ.get("EXP3459_K", str(_P["k_samples"])))
MAX_TOKENS = int(os.environ.get("EXP3459_MAXTOK", str(_P["max_tokens"])))
SAMPLE_TEMPERATURE = float(
    os.environ.get("EXP3459_TEMP", str(_P["sample_temperature"]))
)
N_TARGET = int(os.environ.get("EXP3459_N", str(_P["n_problems_target"])))
MODEL_HF_ID = _P["model_hf_id"]
# ~18-minute wall budget (exp3448's proven 1020s); we stop launching new problems
# once we cross it and finalize cleanly. Env-overridable for a quick smoke run.
WALL_BUDGET_S = float(os.environ.get("EXP3459_WALL_BUDGET_S", "1020"))

# Generation must be byte-for-byte homogeneous with exp3448, so we reuse its exact
# primitives. exp3448's ``_generate`` reads ``MAX_TOKENS`` from its OWN module
# global; export the env it honours BEFORE importing so the cap matches what we
# read from the artifact (default 320 == exp3448's default — this is belt-and-
# suspenders for the case where the artifact recorded a different cap).
os.environ.setdefault("EXP3448_MAXTOK", str(MAX_TOKENS))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from scripts.experiment_3448_p01_generation_corpus_builder_v1 import (  # noqa: E402
    PROMPT_TEMPLATE,
    _generate,
)
from carnot.inference.sota_models import resolve_cached_gguf  # noqa: E402
from carnot.phase3.energy_descent_premise import load_gsm8k_subset  # noqa: E402
from carnot.phase3.p01_generation_corpus import (  # noqa: E402
    build_corpus_row,
    completed_problem_ids,
    corpus_reproducibility_checksum,
    make_sample,
    read_corpus_rows,
    warmup_self_consistency_check,
)
from carnot.phase3.p01_corpus_extend import (  # noqa: E402
    added_this_run,
    derive_extend_verdict,
    extend_acceptance_gates,
)


def _field_principles() -> dict[str, str]:
    """Per-field principle annotations (CLAUDE.md Principle-Annotated Fields)."""
    return {
        "honest_verdict": "Terminal verdict must start with complete:/success:/"
        "passed:/shipped_.",
        "inference_substrate": "live_llm_inference: candidates really load + run the GGUF.",
        "corpus_path": "data/p01_gsm8k_generations.jsonl — the cached corpus "
        "exp3460/3461/3464 consume.",
        "n_problems_completed": "TOTAL problems with a full generation set after this "
        "run (was 47).",
        "n_problems_target": "the target (120); n < target -> resume again next milestone.",
        "n_problems_added_this_run": "problems newly generated this invocation (proves "
        "resume worked, distinct from total).",
        "k_samples": "sampled generations per problem (6) — the matched-compute budget "
        "for scoring.",
        "per_sample_logprobs_captured": "boolean: mean-token confidence stored per "
        "generation.",
        "self_consistency_non_degenerate": "full-corpus warm-up gate: SC >= greedy AND "
        "> 0.30 (the exp3426 0.0-bug guard).",
        "warmup_self_consistency_accuracy": "majority-vote accuracy over the full corpus.",
        "warmup_greedy_accuracy": "greedy accuracy over the full corpus.",
        "model_specs": "the actual GGUF invoked (gemma-4-26B-A4B-it-GGUF).",
        "random_seed": "determinism precondition for reproducibility.",
        "reproducibility_checksum": "content hash of corpus split + model + seed.",
        "duration_s": "real live MoE generation takes wall time; 60s floor — sub-60s is "
        "the fabrication signal.",
    }


def main() -> None:
    tmpl = ExperimentTemplate(
        exp_id=3459,
        title="P0.1 generation-corpus resume-and-extend toward n=120 (v2)",
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

    # (a) CUDA available.
    try:
        import torch  # noqa: PLC0415

        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False
    pre.append({"resource": "cuda", "available": cuda_ok})
    if not cuda_ok:
        _emit_block("blocked_cuda_unavailable", "torch.cuda.is_available() is False", pre)
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

    # (c) cached corpus present. If MISSING, we FALL BACK to a fresh n=0 build
    #     using exp3448's documented split + seed (the resume contract degrades
    #     gracefully to a fresh start) — NOT a hard block, since a fresh build is
    #     still productive. We record the state so the artifact is honest.
    corpus_present = CORPUS_OUT_PATH.exists()
    pre.append({"resource": "p01_corpus_present", "available": corpus_present})
    if not corpus_present:
        print(
            "[exp3459] data/p01_gsm8k_generations.jsonl missing — "
            "falling back to a fresh n=0 build (exp3448 split + seed)"
        )

    # The source GSM8K split must also exist to define the problems at all.
    source_ok = SOURCE_CORPUS_PATH.exists()
    pre.append({"resource": "real_task_corpus", "available": source_ok})
    if not source_ok:
        _emit_block(
            "blocked_real_task_corpus_missing",
            f"source GSM8K split not found at {SOURCE_CORPUS_PATH}",
            pre,
        )
        return

    # ----- Step 1: load the EXACT exp3448 GSM8K split + seed -----
    problems = load_gsm8k_subset(SOURCE_CORPUS_PATH, n=N_TARGET, seed=SEED)
    problem_order = [p.problem_id for p in problems]
    split_ids = set(problem_order)

    # ----- Step 2: RESUME — count prior, skip already-completed problems -----
    done_ids = completed_problem_ids(CORPUS_OUT_PATH, k_samples=K_SAMPLES)
    n_prior = len(done_ids & split_ids)
    remaining = [p for p in problems if p.problem_id not in done_ids]
    print(
        f"[exp3459] target n={N_TARGET} k={K_SAMPLES}; "
        f"{n_prior} already complete on disk, {len(remaining)} remaining"
    )

    # Load the full model only if there is work to do (resume may find all done).
    llm = None
    if remaining:
        try:
            import llama_cpp  # noqa: PLC0415

            # logits_all left at its FAST default (False); exp3448's ``_generate``
            # reads per-token logprobs from the C context, so we get GPU-bound
            # generation AND exact logprobs (see the exp3448 module docstring).
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

    # ----- Steps 3-6: generate + checkpoint each remaining problem -----
    CORPUS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_done_this_run = 0
    budget_hit = False
    for problem in remaining:
        # Step 6: wall-time budget — stop launching new problems near the ceiling.
        if time.time() - start >= WALL_BUDGET_S:
            budget_hit = True
            print(
                f"[exp3459] wall-time budget {WALL_BUDGET_S:.0f}s reached; "
                f"finalizing with a partial corpus (progress, not failure)"
            )
            break

        prompt = PROMPT_TEMPLATE.format(q=problem.question)

        # Step 3: 1 greedy (temp 0) + k sampled (temp ~0.8) generations.
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
        n_total_done = n_prior + n_done_this_run
        elapsed = int(time.time() - start)
        print(
            f"[exp3459] problem {n_total_done}/{N_TARGET} done "
            f"({K_SAMPLES} samples + greedy), elapsed {elapsed}s"
        )

    # ----- Steps 7-8: full-corpus warm-up self-check + artifact -----
    all_rows = read_corpus_rows(CORPUS_OUT_PATH)
    # Only count rows for problems in THIS split (a corpus could hold stale ids).
    corpus_rows = [r for r in all_rows if str(r.get("problem_id")) in split_ids]
    n_completed = len({str(r["problem_id"]) for r in corpus_rows if "problem_id" in r})
    n_added = added_this_run(n_completed, n_prior)

    warmup = warmup_self_consistency_check(corpus_rows)
    verdict = derive_extend_verdict(n_completed, N_TARGET)
    checksum = corpus_reproducibility_checksum(
        corpus_path=SOURCE_CORPUS_PATH,
        model_path=model_path,
        seed=SEED,
        n_target=N_TARGET,
        k_samples=K_SAMPLES,
    )
    # per_sample_logprobs_captured: did every completed row store a mean confidence
    # for each sampled generation? (the field the scoring task depends on)
    logprobs_captured = bool(corpus_rows) and all(
        all(s.get("mean_token_logprob") is not None for s in (r.get("samples") or []))
        for r in corpus_rows
    )
    gates = extend_acceptance_gates(n_completed, logprobs_captured)

    payload = {
        "honest_verdict": verdict,
        "inference_substrate": "live_llm_inference",
        "task_name": "GSM8K (original questions, exp281 corpus), held-out shuffled split",
        "corpus_path": str(CORPUS_OUT_PATH.relative_to(project_root)),
        "n_problems_completed": n_completed,
        "n_problems_target": N_TARGET,
        "n_problems_added_this_run": n_added,
        "n_problems_prior": n_prior,
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
        "acceptance_gate_g1_corpus_not_regressed": gates["g1_corpus_not_regressed"],
        "acceptance_gate_g2_headline_eligible": gates["g2_headline_eligible"],
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
        f"k={K_SAMPLES} logprobs_captured={logprobs_captured}\n"
        f"  gates: G1_not_regressed={gates['g1_corpus_not_regressed']} "
        f"G2_headline_eligible={gates['g2_headline_eligible']}\n"
        f"  warmup: SC={warmup.self_consistency_accuracy} greedy={warmup.greedy_accuracy} "
        f"non_degenerate={warmup.non_degenerate} (n={warmup.n_problems})\n"
        f"  dur={payload['duration_s']}s"
    )


if __name__ == "__main__":
    main()
