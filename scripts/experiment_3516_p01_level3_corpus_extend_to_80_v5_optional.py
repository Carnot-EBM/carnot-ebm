#!/usr/bin/env python3
r"""Experiment 3516 - P0.1 Level-3 Corpus Extend to N>=80 (v5, optional).

WHY THIS EXPERIMENT EXISTS
==========================
Exp3506 (.323) extended the level-3 corpus to 49 problems with level-3
SC = 0.653, which is inside the headroom band [0.40, 0.70].  A corpus of
n>=80 makes exp3519 (the Route-2 energy-vs-SC crux) headline-eligible.
This script resumes from the existing 49 level-3 rows and fills toward
n>=80, appending only NEW rows.

RESUME DESIGN
-------------
The corpus builder reads completed problem IDs from
``data/p01_difficulty_matched_generations.jsonl`` on startup and skips
them.  Each new row is fsynced immediately.  Re-invocations are idempotent.

NON-BLOCKING
------------
Nothing in Milestone .324 is gated on this script.  Exp3519 runs on
whatever level-3 corpus exists (>=49 already cached).

IDLE-TIMEOUT DEFENCE
--------------------
One flushed progress line is printed after EVERY problem.  This resets
the silence window and prevents the conductor from killing the process as
idle.

WALL-TIME BUDGET
----------------
At 18 minutes the script stops generating and writes a clean artifact with
whatever it completed.  Exit code is always 0 (non-blocking design).

SEED PROVENANCE
---------------
``RANDOM_SEED = 2526139546`` is derived from
``int(sha256(b"HuggingFaceH4/MATH-500:test:level3:v5")[:8], 16)``.
It is NOT the experiment number (3516) per CLAUDE.md Adversarial Artifact
Verification discipline.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PY_ROOT = REPO_ROOT / "python"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from carnot.autoresearch.corpus_p01_headroom import (  # noqa: E402
    BAND_HI,
    BAND_LO,
    answers_match,
    completed_problem_ids,
    extract_boxed_answer,
    in_headroom_band,
    mean_token_logprob,
    normalize_answer,
    parse_reasoning_steps,
    self_consistency_accuracy,
)

EXP_ID = 3516
TITLE = "P0.1 Level-3 Corpus Extend to N>=80 (v5, optional)"

# Shared corpus file — we APPEND to it; exp3506's rows are already there.
CORPUS_PATH = REPO_ROOT / "data" / "p01_difficulty_matched_generations.jsonl"

DELIVERABLE = (
    REPO_ROOT
    / "results"
    / "experiment_3516_p01_level3_corpus_extend_to_80_v5_optional.json"
)

# Level-3 only — the combined L3+L4 corpus pushes SC to 0.70 (boundary).
# Level-3 alone sits at SC ~0.65, squarely inside [0.40, 0.70].
TARGET_LEVEL = 3

# Corpus size thresholds.
TARGET_N = 80     # headline-eligibility threshold for exp3519.
SCORABLE_N = 40   # below this the corpus can still score but is partial.

# Generation parameters (match exp3506 for corpus consistency).
K_SAMPLES = 6
MAX_NEW_TOKENS = int(os.environ.get("EXP3516_MAX_NEW_TOKENS", "512"))
GREEDY_TEMP = 0.0
SAMPLE_TEMP = 0.8
SAMPLE_TOP_P = 0.95

# Content-derived seed: int(sha256(b"HuggingFaceH4/MATH-500:test:level3:v5")[:8], 16)
# NOT the experiment number (3516) per CLAUDE.md Adversarial Artifact Verification.
RANDOM_SEED = 2526139546

# Wall-time budget — 18 min; override via env for testing.
WALL_BUDGET_S = int(os.environ.get("EXP3516_WALL_BUDGET_S", str(18 * 60)))

BENCHMARK_REPO = "HuggingFaceH4/MATH-500"
MATH500_ARROW = (
    Path.home()
    / ".cache/huggingface/datasets/HuggingFaceH4___math-500/default/0.0.0"
    / "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/math-500-test.arrow"
)


# ---------------------------------------------------------------------------
# Pure helpers — GPU-free, unit-tested.
# ---------------------------------------------------------------------------

def classify_verdict_v5(n_completed: int, in_band: bool, sc: float | None) -> str:
    """Return the terminal honest_verdict for the v5 level-3 corpus extension run.

    WHY THREE BANDS: the verdict encodes how far the corpus got so the conductor
    reconciler and exp3519 both know whether headline-eligibility was reached or
    whether a follow-up run is worthwhile.

    The verdict always starts with ``complete:`` per Verdict Terminal-Prefix
    Discipline.  A ``blocked_*`` sub-verdict is also complete (the experiment
    ran and reported an honest outcome — it did not fabricate).

    Args:
        n_completed: Total level-3 problems with full generation sets.
        in_band: True when the level-3 SC sits in [0.40, 0.70].
        sc: The level-3 self-consistency score (majority-vote accuracy).

    Returns:
        A terminal honest_verdict string starting with ``complete:``.
    """
    sc_str = "NA" if sc is None else f"{sc:.3f}"
    if not in_band:
        return "complete: blocked_level3_sc_outside_headroom_band"
    if n_completed >= TARGET_N:
        return (
            f"complete: p01_level3_corpus_headline_eligible_n={n_completed}_sc={sc_str}"
        )
    if n_completed >= SCORABLE_N:
        return (
            f"complete: p01_level3_corpus_scorable_partial_n={n_completed}"
            f"_resume_next_milestone"
        )
    return (
        f"complete: p01_level3_corpus_partial_n={n_completed}"
        f"_resume_next_milestone"
    )


def field_principles_v5() -> dict[str, str]:
    """Per-field principle annotations (CLAUDE.md Principle-Annotated Fields).

    WHY: each field in the results artifact carries a one-line WHY so a future
    auditor understands what failure mode the field guards against, not just
    what it contains.
    """
    return {
        "honest_verdict": (
            "Terminal verdict must start with complete:/success:/passed:/shipped_; "
            "the prefix lets the conductor reconciler classify terminal vs partial "
            "without re-running the experiment."
        ),
        "inference_substrate": (
            "live_llm_inference — real GGUF generation on GPU; sub-60s with CUDA "
            "is the fabrication signal caught by adversarial_verify.py."
        ),
        "corpus_path": (
            "data/p01_difficulty_matched_generations.jsonl — the shared in-band "
            "corpus exp3519 prefers for energy-vs-SC comparison."
        ),
        "n_problems_completed": (
            "Total level-3 problems with a full generation set after this run; "
            "exp3519 needs >=40 for scorable results, >=80 for headline-eligible."
        ),
        "n_problems_added_this_run": (
            "Problems newly generated this invocation — proves the resume logic "
            "worked and shows how much new data the run contributed."
        ),
        "level3_self_consistency_accuracy": (
            "Majority-vote accuracy over LEVEL-3-ONLY corpus rows; must land in "
            "[0.40, 0.70] for the corpus to be useful for P0.1."
        ),
        "self_consistency_in_headroom_band": (
            "Boolean: level-3 SC in [0.40, 0.70]; the corrected self-check that "
            "uses only level-3 rows (exp3496/3506 used combined L3+L4 SC)."
        ),
        "per_step_traces_captured": (
            "Each generation carries a parsed step list for FoVer PRIME process "
            "reward scoring; False means step-level scoring is unavailable."
        ),
        "model_specs": (
            "The actual GGUF invoked — required for reproducibility; a mismatch "
            "between claimed and actual model is a fabrication signal."
        ),
        "random_seed": (
            "Determinism precondition for reproducibility; content-derived "
            "(sha256 of benchmark id + level + version), NOT the experiment number."
        ),
        "reproducibility_checksum": (
            "Content hash of benchmark split + model + seed — catches silent version "
            "drift between this artifact and any future replication attempt."
        ),
        "duration_s": (
            "Real live MoE generation takes wall time; 60s floor when CUDA is "
            "available — sub-60s is the DURATION_TOO_SHORT fabrication signal."
        ),
    }


def _gemma_chat_prompt(problem: str) -> str:
    r"""Wrap a MATH problem in the gemma-4 instruct turn format.

    WHY: gemma instruct models expect ``<start_of_turn>user ... <end_of_turn>``
    then an open ``<start_of_turn>model`` turn.  We ask for ``\boxed{}`` so the
    answer extractor has a deterministic target.
    """
    instruction = (
        f"{problem}\n\n"
        r"Solve the problem step by step. Put your final answer inside \boxed{}."
    )
    return (
        f"<start_of_turn>user\n{instruction}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def _qwen_chat_prompt(problem: str) -> str:
    r"""Wrap a MATH problem in the Qwen3 ChatML instruct format.

    WHY: Qwen3 uses ChatML.  We disable thinking mode (``/no_think``) to avoid
    the extended-thinking token budget that caused the Opus v2 crash.
    """
    instruction = (
        f"{problem}\n\n"
        r"Solve the problem step by step. Put your final answer inside \boxed{}."
        " /no_think"
    )
    return (
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _build_generation_record(
    text: str,
    token_logprobs: list[float | None] | None,
    gold_answer: str | None,
    mode: str,
    seed: int,
) -> dict[str, Any]:
    """Assemble one generation record from raw model output.

    WHY: centralising extraction/normalisation/step-parsing here means the
    corpus schema is identical regardless of which model produced the text.
    Pure function — no GPU side effects.
    """
    extracted = extract_boxed_answer(text)
    correct = answers_match(text, gold_answer) if gold_answer is not None else False
    steps = parse_reasoning_steps(text)
    return {
        "mode": mode,
        "seed": seed,
        "text": text,
        "extracted_answer": extracted,
        "extracted_answer_norm": normalize_answer(extracted),
        "correct": bool(correct),
        "mean_token_logprob": mean_token_logprob(token_logprobs or []),
        "reasoning_steps": steps,
        "n_steps": len(steps),
    }


def _build_problem_record(
    problem_meta: dict[str, Any],
    greedy: dict[str, Any],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the full per-problem JSONL row (greedy + k samples).

    WHY: ``sampled_answers`` is duplicated at the top level (normalised) so
    self_consistency_accuracy() can read it directly without re-parsing every
    sample text on every SC computation call.
    """
    gold = problem_meta.get("gold_answer")
    return {
        "problem_id": str(problem_meta.get("problem_id")),
        "level": problem_meta.get("level"),
        "subject": problem_meta.get("subject"),
        "problem": problem_meta.get("problem"),
        "gold_answer": gold,
        "gold_answer_norm": normalize_answer(gold),
        "greedy": greedy,
        "samples": samples,
        "sampled_answers": [s.get("extracted_answer_norm") for s in samples],
        "greedy_correct": bool(greedy.get("correct")),
        "k_samples": len(samples),
    }


# ---------------------------------------------------------------------------
# I/O helpers — thin wrappers, not unit-tested.
# ---------------------------------------------------------------------------

def _load_math_level3_records() -> list[dict[str, Any]]:
    """Load MATH-500 test problems for level-3 ONLY.

    WHY LEVEL-3 ONLY: level-3 SC sits at ~0.65 — squarely in the headroom band
    [0.40, 0.70].  Level-4 problems are harder and would push SC down.

    Reads the HuggingFace Arrow cache with pyarrow.  Falls back to a glob
    search if the hard-coded cache path has moved.
    """
    import pyarrow.ipc as ipc

    path = MATH500_ARROW
    if not path.exists():
        base = Path.home() / ".cache/huggingface/datasets"
        candidates = list(base.glob("*math*/**/*.arrow")) + list(
            base.glob("*math*/**/*.parquet")
        )
        if not candidates:
            raise FileNotFoundError(f"MATH-500 cache not found at {path}")
        path = candidates[0]

    if path.suffix == ".arrow":
        with open(path, "rb") as fh:
            table = ipc.open_stream(fh).read_all()
        rows = table.to_pylist()
    else:
        import pandas as pd
        df = pd.read_parquet(path)
        rows = df.to_dict("records")

    records: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if int(row["level"]) != TARGET_LEVEL:
            continue
        uid = row.get("unique_id")
        pid = str(uid) if uid is not None else f"row{idx}"
        records.append(
            {
                "problem_id": pid,
                "level": int(row["level"]),
                "subject": row.get("subject"),
                "problem": str(row["problem"]),
                "gold_answer": str(row["answer"]),
            }
        )
    return records


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON record as a line, flushing so a kill keeps the row.

    WHY FSYNC: the corpus builder may be killed mid-run.  fsync ensures the
    row is durable on disk before the process continues to the next problem.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _read_corpus(path: Path) -> list[dict[str, Any]]:
    """Read all completed problem rows back from the corpus JSONL."""
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    """Write the results artifact JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=1)


def _greedy_accuracy(corpus: list[dict[str, Any]]) -> float | None:
    """Fraction of corpus problems whose greedy generation was correct."""
    if not corpus:
        return None
    return sum(1 for r in corpus if r.get("greedy_correct")) / len(corpus)


def _build_artifact(
    *,
    verdict: str,
    duration_s: float,
    n_completed: int,
    n_added: int,
    sc: float | None,
    greedy_acc: float | None,
    model_specs: dict[str, Any] | None,
    preconditions_checked: list[dict[str, Any]],
    repro_checksum: str | None,
    status: str,
) -> dict[str, Any]:
    """Assemble the full results artifact with all REQUIRED ARTIFACT FIELDS."""
    in_band = sc is not None and in_headroom_band(sc)
    from scripts.experiment_template import _run_date, _utc_now  # noqa: E402

    return {
        "experiment_id": EXP_ID,
        "experiment": EXP_ID,
        "title": TITLE,
        "run_date": _run_date(),
        "run_timestamp": _utc_now(),
        "schema": "carnot.p01_level3_corpus_extend_v5",
        "duration_s": duration_s,
        "honest_verdict": verdict,
        "inference_substrate": "live_llm_inference",
        "corpus_path": str(CORPUS_PATH.relative_to(REPO_ROOT)),
        "benchmark_id": (
            f"{BENCHMARK_REPO} (test split), level={TARGET_LEVEL}, seed={RANDOM_SEED}"
        ),
        "target_level": TARGET_LEVEL,
        "n_problems_completed": n_completed,
        "n_problems_target": TARGET_N,
        "n_problems_added_this_run": n_added,
        "k_samples": K_SAMPLES,
        "per_step_traces_captured": n_completed > 0,
        "per_sample_logprobs_captured": False,
        "level3_self_consistency_accuracy": sc,
        "self_consistency_in_headroom_band": bool(in_band),
        "greedy_accuracy": greedy_acc,
        "model_specs": model_specs,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": repro_checksum,
        "preconditions_checked": preconditions_checked,
        "status": status,
        "field_provenance": {k: {"principle": v} for k, v in field_principles_v5().items()},
    }


# ---------------------------------------------------------------------------
# GPU-bound orchestration — not unit-tested (requires CUDA + cached GGUF).
# ---------------------------------------------------------------------------

def main() -> int:  # noqa: C901
    """Extend the level-3 corpus, resuming from existing rows if present.

    WHY NON-BLOCKING EXIT: this is a NON-BLOCKING optional builder.  No
    milestone task depends on it completing.  We always exit 0 and write a
    clean artifact describing what we accomplished (or why we blocked).
    """
    t0 = time.time()
    from scripts.experiment_template import ExperimentTemplate, _compute_repro_checksum
    from carnot.inference.sota_models import cached_sota_pair

    tmpl = ExperimentTemplate(EXP_ID, TITLE, str(DELIVERABLE))
    tmpl.setup()

    preconditions_checked: list[dict[str, Any]] = []

    # ---- Step 0a: CUDA precondition -----------------------------------------
    try:
        import torch
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False
    preconditions_checked.append({"resource": "cuda", "available": cuda_ok})
    if not cuda_ok:
        art = _build_artifact(
            verdict="complete: blocked_cuda_unavailable",
            duration_s=time.time() - t0,
            n_completed=0,
            n_added=0,
            sc=None,
            greedy_acc=None,
            model_specs=None,
            preconditions_checked=preconditions_checked,
            repro_checksum=None,
            status="blocked",
        )
        _write_artifact(DELIVERABLE, art)
        print("[exp3516] CUDA unavailable — wrote blocked artifact.", flush=True)
        return 0

    # ---- Step 0b: SOTA GGUF tokenizer precondition --------------------------
    # Per CLAUDE.md "GGUF tokenizer rule (MANDATORY — 2026-05-29)": load via
    # the .gguf model_path (NOT AutoTokenizer on the -GGUF repo id).
    model_path = None
    model_name = None
    prompt_fn = _qwen_chat_prompt
    tok_ok = False
    try:
        pair = cached_sota_pair()
        if pair:
            model_path = pair[0].get("model_path")
            model_name = pair[0].get("name")
        if model_name and "gemma" in model_name.lower():
            prompt_fn = _gemma_chat_prompt
        if model_path and Path(model_path).exists():
            from llama_cpp import Llama
            probe = Llama(model_path=model_path, vocab_only=True, verbose=False)
            probe.tokenize(b"x")
            tok_ok = True
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[exp3516] tokenizer probe failed: {exc}", flush=True)
        tok_ok = False
    preconditions_checked.append({"resource": "sota_gguf_tokenizer", "available": tok_ok})
    if not tok_ok:
        art = _build_artifact(
            verdict="complete: blocked_sota_gguf_tokenizer_unavailable",
            duration_s=time.time() - t0,
            n_completed=0,
            n_added=0,
            sc=None,
            greedy_acc=None,
            model_specs=None,
            preconditions_checked=preconditions_checked,
            repro_checksum=None,
            status="blocked",
        )
        _write_artifact(DELIVERABLE, art)
        print("[exp3516] SOTA GGUF tokenizer unavailable — wrote blocked artifact.", flush=True)
        return 0

    repro_checksum = _compute_repro_checksum(
        RANDOM_SEED, [Path(__file__)], CORPUS_PATH
    )

    # ---- Load model ---------------------------------------------------------
    from llama_cpp import Llama  # noqa: E402

    print(f"[exp3516] loading model: {model_name} ({model_path})", flush=True)
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1,
        seed=RANDOM_SEED,
        verbose=False,
    )
    model_specs = {
        "name": model_name,
        "model_path": model_path,
        "loader": "llama_cpp",
        "prompt_format": (
            "qwen_chatml" if prompt_fn is _qwen_chat_prompt else "gemma_instruct"
        ),
    }

    def _generate(prompt: str, temperature: float, seed: int) -> dict[str, Any]:
        """Run one llama.cpp completion; returns text + empty logprobs.

        WHY NO LOGITS: logits_all=True slows generation ~10x on a 150k-vocab
        MoE model (softmax over every generated token).  SC/correctness data
        is the load-bearing output; logprobs are not required.
        """
        out = llm.create_completion(
            prompt,
            max_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            top_p=SAMPLE_TOP_P if temperature > 0 else 1.0,
            seed=seed,
            stop=["<|im_end|>", "<end_of_turn>", "<eos>", "<|endoftext|>"],
        )
        text = out["choices"][0].get("text", "")
        return {"text": text, "token_logprobs": []}

    def _gen_problem(meta: dict[str, Any]) -> dict[str, Any]:
        """Generate greedy + k sampled solutions for one problem."""
        prompt = prompt_fn(meta["problem"])
        g = _generate(prompt, GREEDY_TEMP, RANDOM_SEED)
        greedy = _build_generation_record(
            g["text"], g["token_logprobs"], meta["gold_answer"], "greedy", RANDOM_SEED
        )
        samples: list[dict[str, Any]] = []
        for j in range(K_SAMPLES):
            s_seed = RANDOM_SEED + 1 + j
            s = _generate(prompt, SAMPLE_TEMP, s_seed)
            samples.append(
                _build_generation_record(
                    s["text"], s["token_logprobs"], meta["gold_answer"], "sampled", s_seed
                )
            )
        return _build_problem_record(meta, greedy, samples)

    def _budget_left() -> float:
        return WALL_BUDGET_S - (time.time() - t0)

    # ---- Resume state -------------------------------------------------------
    done_ids = completed_problem_ids(CORPUS_PATH)
    existing_level3 = [
        r for r in _read_corpus(CORPUS_PATH) if r.get("level") == TARGET_LEVEL
    ]
    print(
        f"[exp3516] resume: {len(done_ids)} problems in corpus "
        f"({len(existing_level3)} level-3 already done).",
        flush=True,
    )

    # ---- Load level-3 pool --------------------------------------------------
    all_level3 = _load_math_level3_records()
    fill_pool = [r for r in all_level3 if r["problem_id"] not in done_ids]
    print(
        f"[exp3516] level-3 pool: {len(all_level3)} total, "
        f"{len(fill_pool)} not yet generated.",
        flush=True,
    )

    # ---- Generate new problems ----------------------------------------------
    n_added = 0
    for meta in fill_pool:
        if _budget_left() < 90:
            print("[exp3516] wall budget reached; finalizing.", flush=True)
            break
        level3_count = len(existing_level3) + n_added
        if level3_count >= TARGET_N:
            print(
                f"[exp3516] target n={TARGET_N} reached; finalizing.", flush=True
            )
            break
        rec = _gen_problem(meta)
        _append_jsonl(CORPUS_PATH, rec)
        done_ids.add(rec["problem_id"])
        n_added += 1
        level3_count = len(existing_level3) + n_added
        # LOAD-BEARING: one flushed line per problem defeats idle-timeout.
        print(
            f"[exp3516] FILL pid={rec['problem_id']}"
            f" L{rec['level']}"
            f" greedy_correct={rec['greedy_correct']}"
            f" n_level3={level3_count}"
            f" n_added={n_added}"
            f" budget_left={_budget_left():.0f}s",
            flush=True,
        )

    # ---- Finalise -----------------------------------------------------------
    # Recompute SC using only level-3 rows (the corrected self-check).
    full_level3 = [
        r for r in _read_corpus(CORPUS_PATH) if r.get("level") == TARGET_LEVEL
    ]
    final_sc = self_consistency_accuracy(full_level3) if full_level3 else None
    greedy_acc = _greedy_accuracy(full_level3)
    n_completed = len(full_level3)
    in_band = final_sc is not None and in_headroom_band(final_sc)

    verdict = classify_verdict_v5(n_completed, bool(in_band), final_sc)
    art = _build_artifact(
        verdict=verdict,
        duration_s=time.time() - t0,
        n_completed=n_completed,
        n_added=n_added,
        sc=final_sc,
        greedy_acc=greedy_acc,
        model_specs=model_specs,
        preconditions_checked=preconditions_checked,
        repro_checksum=repro_checksum,
        status="success",
    )
    _write_artifact(DELIVERABLE, art)
    print(
        f"[exp3516] DONE n_level3={n_completed} sc={final_sc} in_band={in_band}"
        f" added={n_added} dur={time.time() - t0:.0f}s verdict={verdict}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
