#!/usr/bin/env python3
r"""Experiment 3530 - P0.1 Route 2 Selectable-Headroom Corpus Build v1.

WHY THIS EXPERIMENT EXISTS
==========================
Every Route-2 energy-vs-SC test (including exp3519) has failed for the same
structural reason: the test corpus has NO selectable headroom.  Selectable
headroom exists for a problem when the correct answer IS present among the k
sampled candidates but is NOT the SC majority.  Without this property the
oracle accuracy <= SC accuracy, so any energy-vs-SC comparison is
uninformative (FALSE_NEGATIVE_RISK per CLAUDE.md "Adversarial Artifact
Verification").

This script builds a POSITIVE CONTROL corpus by:
1.  Targeting harder difficulty (MATH level 4-5) where SC drops and
    minority-correct answers become more common.
2.  Applying a HEADROOM FILTER: keep a problem iff the correct answer appears
    in the k=8 sampled candidates AND is NOT the SC majority.
3.  Writing only the KEPT problems to data/p01_selectable_headroom_corpus.jsonl.

For the kept corpus:
  - oracle_accuracy = 1.0  (by construction: correct answer always present)
  - self_consistency_accuracy = 0.0  (by construction: SC majority always wrong)
  - selectable_headroom = 1.0  (strictly > 0, target >= 0.05 per task spec)

The acceptance gate is: oracle_exceeds_sc == True AND n_problems_kept >= 40.
Once this corpus exists, exp3531 can run a meaningful Route-2 energy-vs-SC
test: does energy-based reranking recover the correct answer (which IS present
in candidates) better than the SC majority (which IS wrong)?

REFERENCES
----------
ThinkPRM (arXiv:2504.16828): PRM beats SC where selectable headroom exists.
Self-certainty BoN (arXiv:2502.18581): BoN selection on math problems.
MoB (arXiv:2511.18630): mixture-of-branching high-reward completion selection.
exp3519 (.324): oracle 0.475 <= SC 0.5 — the FALSE_NEGATIVE_RISK this fixes.

RESUME DESIGN
-------------
The corpus builder reads completed problem IDs from
``data/p01_selectable_headroom_corpus.jsonl`` on startup and skips them.
Each kept row is fsynced immediately.  Re-invocations are idempotent.

IDLE-TIMEOUT DEFENCE
--------------------
One flushed progress line is printed after EVERY problem (kept or skipped).
This resets the silence window and prevents the conductor from killing the
process as idle.

WALL-TIME BUDGET
----------------
At ~22 minutes the script stops generating and writes a clean artifact with
whatever it completed.  Exit code is always 0.

SEED PROVENANCE
---------------
RANDOM_SEED = int(sha256(b"HuggingFaceH4/MATH-500:test:level4-5:selectable_headroom:v1")[:8], 16)
= 2323699563 — NOT the experiment number (3530) per CLAUDE.md Adversarial
Artifact Verification discipline.
"""
from __future__ import annotations

import hashlib
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
    answers_match,
    completed_problem_ids,
    extract_boxed_answer,
    mean_token_logprob,
    normalize_answer,
    parse_reasoning_steps,
    self_consistency_accuracy,
)

EXP_ID = 3530
TITLE = "P0.1 Route 2 Selectable-Headroom Corpus Build v1"

# New corpus path — does NOT overwrite the level-3 corpus.
CORPUS_PATH = REPO_ROOT / "data" / "p01_selectable_headroom_corpus.jsonl"

DELIVERABLE = (
    REPO_ROOT
    / "results"
    / "experiment_3530_p01_route2_selectable_headroom_corpus_build_v1.json"
)

# Harder difficulty: level 4-5 where SC drops and minority-correct answers
# become common, enabling the selectable headroom property.
TARGET_LEVELS = {4, 5}

# Acceptance gate: at least this many kept problems to be headline-eligible.
TARGET_N = 40

# Generation parameters — k=8 per task spec ("k>=8 sampled").
K_SAMPLES = 8
MAX_NEW_TOKENS = int(os.environ.get("EXP3530_MAX_NEW_TOKENS", "768"))
GREEDY_TEMP = 0.0
SAMPLE_TEMP = 0.8
SAMPLE_TOP_P = 0.95

# Content-derived seed: int(sha256(b"HuggingFaceH4/MATH-500:test:level4-5:selectable_headroom:v1")[:8], 16)
# NOT 3530 per CLAUDE.md Adversarial Artifact Verification discipline.
_SEED_SRC = b"HuggingFaceH4/MATH-500:test:level4-5:selectable_headroom:v1"
RANDOM_SEED: int = int(hashlib.sha256(_SEED_SRC).hexdigest()[:8], 16)
assert RANDOM_SEED == 2323699563, f"seed mismatch: {RANDOM_SEED}"  # guard against hasher change

# Wall-time budget — 22 min; override via env for testing.
WALL_BUDGET_S = int(os.environ.get("EXP3530_WALL_BUDGET_S", str(22 * 60)))

BENCHMARK_REPO = "HuggingFaceH4/MATH-500"
MATH500_ARROW = (
    Path.home()
    / ".cache/huggingface/datasets/HuggingFaceH4___math-500/default/0.0.0"
    / "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/math-500-test.arrow"
)


# ---------------------------------------------------------------------------
# Pure helpers — GPU-free, unit-tested.
# ---------------------------------------------------------------------------

def has_selectable_headroom(record: dict[str, Any]) -> bool:
    """Return True when the problem has selectable headroom.

    WHY: Selectable headroom is the positive-control property that makes a
    Route-2 energy-vs-SC comparison informative.  A problem has selectable
    headroom when:
      1. The correct answer IS present among the k sampled candidates
         (at least one sample produced the right answer — an oracle reranker
         CAN recover it).
      2. The SC majority is NOT the correct answer (majority voting fails —
         there IS something to recover over SC).

    If condition 1 fails: oracle accuracy = 0 for this problem; the reranker
    can't recover what was never generated.
    If condition 2 fails: SC already gets it right; headroom = 0.

    Both conditions together guarantee: oracle > SC for this problem.

    Args:
        record: A problem record with ``sampled_answers`` (list of normalized
                answer strings) and ``gold_answer_norm`` (normalized gold).

    Returns:
        True iff both conditions above are satisfied.
    """
    gold = record.get("gold_answer_norm")
    if gold is None:
        return False
    sampled_answers = record.get("sampled_answers") or []
    # Condition 1: correct answer present in at least one sample.
    correct_present = any(
        normalize_answer(a) == normalize_answer(gold)
        for a in sampled_answers
        if a is not None
    )
    if not correct_present:
        return False
    # Condition 2: SC majority ≠ correct answer.
    counts: dict[str, int] = {}
    for a in sampled_answers:
        if a is not None:
            counts[a] = counts.get(a, 0) + 1
    if not counts:
        return False
    majority = max(counts, key=lambda k: counts[k])
    return normalize_answer(majority) != normalize_answer(gold)


def compute_corpus_stats(
    kept_records: list[dict[str, Any]],
) -> dict[str, float | bool]:
    """Compute oracle accuracy, SC accuracy, and selectable headroom over the kept corpus.

    WHY: We compute these explicitly over the kept corpus rather than reporting
    theoretical values (1.0 / 0.0 / 1.0) so any code-path bug in the filter
    immediately shows up as a deviation from the expected values.  A kept
    corpus with SC > 0 or oracle < 1.0 signals a filter logic error.

    For a correct run:
      - oracle_accuracy  = 1.0  (by construction: every kept problem has the
                                  correct answer present in samples)
      - self_consistency_accuracy = 0.0  (by construction: every kept problem
                                           has SC majority wrong)
      - selectable_headroom = 1.0

    Args:
        kept_records: Problem rows already filtered by ``has_selectable_headroom``.

    Returns:
        Dict with ``oracle_accuracy``, ``self_consistency_accuracy``,
        ``selectable_headroom``, and ``oracle_exceeds_sc``.
    """
    if not kept_records:
        return {
            "oracle_accuracy": 0.0,
            "self_consistency_accuracy": 0.0,
            "selectable_headroom": 0.0,
            "oracle_exceeds_sc": False,
        }
    # Oracle: fraction where correct answer is present in samples.
    oracle_correct = sum(1 for r in kept_records if _oracle_is_correct(r))
    oracle_acc = oracle_correct / len(kept_records)
    # SC: majority-vote accuracy.
    sc_acc = self_consistency_accuracy(kept_records)
    headroom = oracle_acc - sc_acc
    return {
        "oracle_accuracy": oracle_acc,
        "self_consistency_accuracy": sc_acc,
        "selectable_headroom": headroom,
        "oracle_exceeds_sc": bool(oracle_acc > sc_acc),
    }


def _oracle_is_correct(record: dict[str, Any]) -> bool:
    """Return True when the correct answer appears in at least one sample.

    WHY: Oracle accuracy = "what if we could always pick the correct answer
    when it is present in the k candidates?"  A problem where no sample is
    correct contributes 0 to oracle accuracy.
    """
    gold = record.get("gold_answer_norm")
    if gold is None:
        return False
    return any(
        normalize_answer(a) == normalize_answer(gold)
        for a in (record.get("sampled_answers") or [])
        if a is not None
    )


def classify_verdict_3530(
    n_kept: int,
    oracle_acc: float,
    sc_acc: float,
) -> str:
    """Return the terminal honest_verdict for the selectable-headroom corpus build.

    WHY THREE CASES: the verdict encodes whether the acceptance gate was met,
    what the headroom outcome was, or whether the model failed to produce any
    selectable headroom at all.  Always starts with ``complete:`` per
    Verdict Terminal-Prefix Discipline.

    Args:
        n_kept: Number of problems that passed the selectable-headroom filter.
        oracle_acc: Oracle accuracy over the kept corpus.
        sc_acc: SC majority accuracy over the kept corpus.

    Returns:
        A terminal honest_verdict string starting with ``complete:``.
    """
    if n_kept >= TARGET_N and oracle_acc > sc_acc:
        return (
            f"complete: p01_selectable_headroom_corpus_built"
            f"_n={n_kept}"
            f"_oracle={oracle_acc:.3f}"
            f"_exceeds_sc={sc_acc:.3f}"
        )
    if n_kept > 0 and oracle_acc > sc_acc:
        return (
            f"complete: p01_selectable_headroom_corpus_partial"
            f"_n={n_kept}"
            f"_below_target_{TARGET_N}"
            f"_resume_next_milestone"
        )
    if n_kept == 0:
        return (
            "complete: p01_no_selectable_headroom_even_at_level4_5"
            "_sc_majority_is_near_optimal_route2_premise_bounded"
        )
    # oracle <= sc edge case (should not happen with correct filter).
    return (
        f"complete: p01_selectable_headroom_filter_anomaly"
        f"_n={n_kept}_oracle={oracle_acc:.3f}_sc={sc_acc:.3f}"
    )


def field_principles_3530() -> dict[str, str]:
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
            "is the DURATION_TOO_SHORT fabrication signal in adversarial_verify.py."
        ),
        "corpus_path": (
            "data/p01_selectable_headroom_corpus.jsonl — the new positive-control "
            "corpus exp3531 reads for its Route-2 energy-vs-SC test."
        ),
        "n_problems_kept": (
            "Number of problems that passed the selectable-headroom filter; "
            "exp3531 needs >=40 for a meaningful Route-2 verdict."
        ),
        "self_consistency_accuracy": (
            "SC majority-vote accuracy over the KEPT corpus; should be 0.0 by "
            "construction (we only keep problems where SC majority is wrong) — "
            "any non-zero value signals a filter logic error."
        ),
        "oracle_accuracy": (
            "Accuracy if the correct answer is always selected when present; "
            "should be 1.0 by construction (we only keep problems where correct "
            "answer IS present in samples) — any sub-1.0 value signals a filter "
            "logic error."
        ),
        "selectable_headroom": (
            "oracle_accuracy - self_consistency_accuracy; MUST be > 0 (ideally "
            "1.0 for the kept corpus) — this is the positive-control property "
            "exp3519 lacked."
        ),
        "oracle_exceeds_sc": (
            "Boolean: oracle STRICTLY > SC; the FALSE_NEGATIVE_RISK precondition "
            "for a meaningful Route-2 energy-vs-SC test."
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
            "Determinism precondition for reproducibility; content-derived via "
            "sha256 of benchmark id + level + version, NOT the experiment number."
        ),
        "reproducibility_checksum": (
            "Content hash of benchmark split + model + seed — catches silent "
            "version drift between this artifact and any future replication."
        ),
        "duration_s": (
            "Real live MoE generation takes wall time; 60s floor when CUDA is "
            "available — sub-60s is the DURATION_TOO_SHORT fabrication signal."
        ),
    }


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
    ``has_selectable_headroom()`` and ``self_consistency_accuracy()`` can read
    it directly without re-parsing every sample text on every call.
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
        "has_selectable_headroom": None,  # filled by caller after filter
    }


def _build_artifact(
    *,
    verdict: str,
    duration_s: float,
    n_attempted: int,
    n_kept: int,
    n_added_this_run: int,
    oracle_acc: float | None,
    sc_acc: float | None,
    headroom: float | None,
    oracle_exceeds_sc: bool,
    per_step_traces: bool,
    model_specs: dict[str, Any] | None,
    preconditions_checked: list[dict[str, Any]],
    repro_checksum: str | None,
    status: str,
) -> dict[str, Any]:
    """Assemble the full results artifact with all REQUIRED ARTIFACT FIELDS."""
    from scripts.experiment_template import _run_date, _utc_now  # noqa: E402

    return {
        "experiment_id": EXP_ID,
        "experiment": EXP_ID,
        "title": TITLE,
        "run_date": _run_date(),
        "run_timestamp": _utc_now(),
        "schema": "carnot.p01_selectable_headroom_corpus_v1",
        "duration_s": duration_s,
        "honest_verdict": verdict,
        "inference_substrate": "live_llm_inference",
        "corpus_path": str(CORPUS_PATH.relative_to(REPO_ROOT)),
        "benchmark_id": (
            f"{BENCHMARK_REPO} (test split), levels={sorted(TARGET_LEVELS)}, "
            f"seed={RANDOM_SEED}"
        ),
        "target_levels": sorted(TARGET_LEVELS),
        "n_problems_attempted": n_attempted,
        "n_problems_kept": n_kept,
        "n_problems_added_this_run": n_added_this_run,
        "n_problems_target": TARGET_N,
        "k_samples": K_SAMPLES,
        "per_step_traces_captured": per_step_traces,
        "per_sample_logprobs_captured": False,
        "self_consistency_accuracy": sc_acc,
        "oracle_accuracy": oracle_acc,
        "selectable_headroom": headroom,
        "oracle_exceeds_sc": oracle_exceeds_sc,
        "model_specs": model_specs,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": repro_checksum,
        "preconditions_checked": preconditions_checked,
        "status": status,
        "field_provenance": {
            k: {"principle": v} for k, v in field_principles_3530().items()
        },
    }


# ---------------------------------------------------------------------------
# I/O helpers — thin wrappers, not unit-tested.
# ---------------------------------------------------------------------------

def _load_math_level45_records() -> list[dict[str, Any]]:
    """Load MATH-500 test problems for levels 4 and 5 only.

    WHY LEVEL 4-5: these are harder than level-3 (SC drops to ~30-45%) so
    minority-correct answers are more common, enabling the selectable-headroom
    property that level-3 (SC ~0.65) rarely exhibits.

    Reads the HuggingFace Arrow cache with pyarrow.  Falls back to a glob
    search if the hard-coded cache path has moved.
    """
    import pyarrow.ipc as ipc  # noqa: E402

    path = MATH500_ARROW
    if not path.exists():
        base = Path.home() / ".cache/huggingface/datasets"
        candidates = list(base.glob("*math*/**/*.arrow")) + list(
            base.glob("*math*/**/*.parquet")
        )
        # Prefer the MATH-500 HuggingFaceH4 dataset over other math datasets.
        hf_candidates = [c for c in candidates if "HuggingFaceH4" in str(c)]
        path = hf_candidates[0] if hf_candidates else (candidates[0] if candidates else path)

    if not path.exists():
        raise FileNotFoundError(f"MATH-500 cache not found at {path}")

    if path.suffix == ".arrow":
        with open(path, "rb") as fh:
            table = ipc.open_stream(fh).read_all()
        rows = table.to_pylist()
    else:
        import pandas as pd  # noqa: E402
        df = pd.read_parquet(path)
        rows = df.to_dict("records")

    records: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        level = int(row["level"])
        if level not in TARGET_LEVELS:
            continue
        uid = row.get("unique_id")
        pid = str(uid) if uid is not None else f"row{idx}"
        records.append(
            {
                "problem_id": pid,
                "level": level,
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
    """Read all completed problem rows from the corpus JSONL."""
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


# ---------------------------------------------------------------------------
# GPU-bound orchestration — not unit-tested (requires CUDA + cached GGUF).
# ---------------------------------------------------------------------------

def _gemma_chat_prompt(problem: str) -> str:
    r"""Wrap a MATH problem in the gemma-4 instruct turn format."""
    instruction = (
        f"{problem}\n\n"
        r"Solve the problem step by step. Put your final answer inside \boxed{}."
    )
    return (
        f"<start_of_turn>user\n{instruction}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def _qwen_chat_prompt(problem: str) -> str:
    r"""Wrap a MATH problem in the Qwen3 ChatML instruct format."""
    instruction = (
        f"{problem}\n\n"
        r"Solve the problem step by step. Put your final answer inside \boxed{}."
        " /no_think"
    )
    return (
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def main() -> int:  # noqa: C901
    """Build the selectable-headroom corpus, resuming from existing rows if present.

    WHY NON-BLOCKING EXIT: this is a non-blocking corpus builder.  We always
    exit 0 and write a clean artifact describing what we accomplished (or why
    we blocked).
    """
    t0 = time.time()
    from scripts.experiment_template import ExperimentTemplate, _compute_repro_checksum  # noqa: E402
    from carnot.inference.sota_models import cached_sota_pair  # noqa: E402

    tmpl = ExperimentTemplate(EXP_ID, TITLE, str(DELIVERABLE))
    tmpl.setup()

    preconditions_checked: list[dict[str, Any]] = []

    # ---- Step 0a: CUDA precondition -----------------------------------------
    try:
        import torch  # noqa: E402
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False
    preconditions_checked.append({"resource": "cuda", "available": cuda_ok})
    if not cuda_ok:
        art = _build_artifact(
            verdict="complete: blocked_cuda_unavailable",
            duration_s=time.time() - t0,
            n_attempted=0,
            n_kept=0,
            n_added_this_run=0,
            oracle_acc=None,
            sc_acc=None,
            headroom=None,
            oracle_exceeds_sc=False,
            per_step_traces=False,
            model_specs=None,
            preconditions_checked=preconditions_checked,
            repro_checksum=None,
            status="blocked",
        )
        _write_artifact(DELIVERABLE, art)
        print("[exp3530] CUDA unavailable — wrote blocked artifact.", flush=True)
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
            from llama_cpp import Llama  # noqa: E402
            probe = Llama(model_path=model_path, vocab_only=True, verbose=False)
            probe.tokenize(b"x")
            tok_ok = True
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[exp3530] tokenizer probe failed: {exc}", flush=True)
        tok_ok = False
    preconditions_checked.append({"resource": "sota_gguf_tokenizer", "available": tok_ok})
    if not tok_ok:
        art = _build_artifact(
            verdict="complete: blocked_sota_gguf_tokenizer_unavailable",
            duration_s=time.time() - t0,
            n_attempted=0,
            n_kept=0,
            n_added_this_run=0,
            oracle_acc=None,
            sc_acc=None,
            headroom=None,
            oracle_exceeds_sc=False,
            per_step_traces=False,
            model_specs=None,
            preconditions_checked=preconditions_checked,
            repro_checksum=None,
            status="blocked",
        )
        _write_artifact(DELIVERABLE, art)
        print("[exp3530] SOTA GGUF tokenizer unavailable — wrote blocked artifact.", flush=True)
        return 0

    repro_checksum = _compute_repro_checksum(
        RANDOM_SEED, [Path(__file__)], CORPUS_PATH
    )

    # ---- Load model ---------------------------------------------------------
    from llama_cpp import Llama  # noqa: E402

    print(f"[exp3530] loading model: {model_name} ({model_path})", flush=True)
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1,
        seed=RANDOM_SEED,
        verbose=False,
    )
    model_specs: dict[str, Any] = {
        "name": model_name,
        "model_path": model_path,
        "loader": "llama_cpp",
        "prompt_format": (
            "qwen_chatml" if prompt_fn is _qwen_chat_prompt else "gemma_instruct"
        ),
    }

    def _generate(prompt: str, temperature: float, seed: int) -> dict[str, Any]:
        """Run one llama.cpp completion; returns text + empty logprobs."""
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
                    s["text"], s["token_logprobs"], meta["gold_answer"],
                    "sampled", s_seed,
                )
            )
        return _build_problem_record(meta, greedy, samples)

    def _budget_left() -> float:
        return WALL_BUDGET_S - (time.time() - t0)

    # ---- Resume state -------------------------------------------------------
    done_ids = completed_problem_ids(CORPUS_PATH)
    existing_kept = _read_corpus(CORPUS_PATH)
    print(
        f"[exp3530] resume: {len(done_ids)} problems in corpus already kept.",
        flush=True,
    )

    # ---- Load level-4/5 pool ------------------------------------------------
    all_records = _load_math_level45_records()
    fill_pool = [r for r in all_records if r["problem_id"] not in done_ids]
    print(
        f"[exp3530] level-4/5 pool: {len(all_records)} total, "
        f"{len(fill_pool)} not yet attempted.",
        flush=True,
    )

    # ---- Generate and filter ------------------------------------------------
    n_attempted = 0
    n_added_this_run = 0
    kept_this_run: list[dict[str, Any]] = []

    for meta in fill_pool:
        if _budget_left() < 90:
            print("[exp3530] wall budget reached; finalizing.", flush=True)
            break
        rec = _gen_problem(meta)
        n_attempted += 1
        headroom_flag = has_selectable_headroom(rec)
        rec["has_selectable_headroom"] = headroom_flag

        # Running stats for the progress line.
        all_kept = existing_kept + kept_this_run
        if all_kept:
            run_stats = compute_corpus_stats(all_kept)
            run_oracle = run_stats["oracle_accuracy"]
            run_sc = run_stats["self_consistency_accuracy"]
        else:
            run_oracle = 0.0
            run_sc = 0.0

        if headroom_flag:
            _append_jsonl(CORPUS_PATH, rec)
            done_ids.add(rec["problem_id"])
            kept_this_run.append(rec)
            n_added_this_run += 1

        n_kept_total = len(existing_kept) + len(kept_this_run)
        # LOAD-BEARING: one flushed line per problem defeats idle-timeout.
        print(
            f"[exp3530] pid={rec['problem_id']}"
            f" L{rec['level']}"
            f" kept={headroom_flag}"
            f" n_kept={n_kept_total}"
            f" n_tried={n_attempted}"
            f" oracle={run_oracle:.3f}"
            f" sc={run_sc:.3f}"
            f" gap={run_oracle - run_sc:.3f}"
            f" budget_left={_budget_left():.0f}s",
            flush=True,
        )

        if n_kept_total >= TARGET_N * 2:
            # Generous stopping criterion — twice the target is more than enough.
            print(
                f"[exp3530] n_kept={n_kept_total} >= {TARGET_N * 2}; stopping.",
                flush=True,
            )
            break

    # ---- Finalise -----------------------------------------------------------
    final_kept = _read_corpus(CORPUS_PATH)
    n_kept_final = len(final_kept)
    if final_kept:
        stats = compute_corpus_stats(final_kept)
        oracle_acc = stats["oracle_accuracy"]
        sc_acc = stats["self_consistency_accuracy"]
        headroom_val = stats["selectable_headroom"]
        oracle_exceeds_sc = bool(stats["oracle_exceeds_sc"])
    else:
        oracle_acc = 0.0
        sc_acc = 0.0
        headroom_val = 0.0
        oracle_exceeds_sc = False

    verdict = classify_verdict_3530(n_kept_final, oracle_acc, sc_acc)
    art = _build_artifact(
        verdict=verdict,
        duration_s=time.time() - t0,
        n_attempted=n_attempted + len(done_ids) - n_added_this_run,
        n_kept=n_kept_final,
        n_added_this_run=n_added_this_run,
        oracle_acc=oracle_acc,
        sc_acc=sc_acc,
        headroom=headroom_val,
        oracle_exceeds_sc=oracle_exceeds_sc,
        per_step_traces=n_kept_final > 0,
        model_specs=model_specs,
        preconditions_checked=preconditions_checked,
        repro_checksum=repro_checksum,
        status="success",
    )
    _write_artifact(DELIVERABLE, art)
    print(
        f"[exp3530] DONE n_kept={n_kept_final} oracle={oracle_acc:.3f} sc={sc_acc:.3f}"
        f" headroom={headroom_val:.3f} dur={time.time() - t0:.0f}s"
        f" verdict={verdict}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
