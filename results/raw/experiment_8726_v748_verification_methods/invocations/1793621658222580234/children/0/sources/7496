#!/usr/bin/env python3
r"""Experiment 3483 - P0.1 Difficulty-Matched Corpus Builder (v2, MATH-500 L3-4).

WHY THIS EXPERIMENT EXISTS (verbose, for non-EBM engineers)
==========================================================
P0.1 is the project's oldest unrun existential question:

    "Does picking the answer with the LOWEST ENERGY (our verifier's score)
     beat just taking the MAJORITY VOTE among several sampled answers, when
     both are given the SAME compute budget?"

If energy-based selection cannot beat self-consistency (majority vote) at
equal compute, the whole "verifier as a second pair of eyes" thesis is in
question. So this is load-bearing.

THE CORPUS PROBLEM (what .319 and .320 found)
---------------------------------------------
A selector can only help when there is HEADROOM: the sampled answers must
DISAGREE enough that picking the right one matters, but the right answer must
appear OFTEN enough that a good selector can find it.

  - GSM8K   : self-consistency ~0.908 -> CEILING. No room for a selector.
  - MATH-L5 : self-consistency ~0.265 -> FLOOR. Majority vote is noise.

The headroom sweet-spot (per arXiv:2504.16828 ThinkPRM) is MATH-500 levels
3-4, where self-consistency lands in [0.40, 0.70]. exp3471 (v1) targeted
Level 5 and found the floor; THIS builder (v2) targets levels 3-4 with an
ADAPTIVE warm-up that probes self-consistency per level and selects the
level(s) that land in band. It also captures per-STEP reasoning traces so the
FoVer step-error verifier can later be scored as a PROCESS reward (PRIME
arXiv:2602.11570).

WHAT THIS SCRIPT DOES
---------------------
Generation-only corpus builder. For each MATH problem it generates one greedy
(temperature 0) plus k sampled (temperature ~0.8) solutions, extracts the
final ``\boxed{}`` answer, labels correctness against the gold answer, records
the mean-token logprob (confidence), and parses reasoning steps. It APPENDS one
JSONL row per completed problem so it can RESUME on re-invocation and cannot
idle-timeout. It respects a wall-time budget and EXITS CLEAN with whatever it
completes.

The pure, GPU-free helpers (answer extraction/normalization, majority vote,
self-consistency, band classification, mean logprob, resume) live in
``python/carnot/autoresearch/corpus_p01_headroom.py`` and are unit-tested
independently of the GPU. This script holds only the GPU-bound orchestration
plus a handful of pure assembly helpers that are also importable for tests.

See results/experiment_3483_*.json for the artifact this run produces.
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

EXP_ID = 3483
TITLE = "P0.1 Difficulty-Matched Corpus Builder (v2, MATH-500 L3-4)"
CORPUS_PATH = REPO_ROOT / "data" / "p01_difficulty_matched_generations.jsonl"
DELIVERABLE = (
    REPO_ROOT
    / "results"
    / "experiment_3483_p01_difficulty_matched_corpus_builder_v2.json"
)

# Difficulty-matched corpus parameters.
TARGET_N = 80              # headline-eligibility threshold
SCORABLE_N = 40           # G2 "scorable partial" threshold
K_SAMPLES = 6             # sampled generations per problem (matched compute)
MAX_NEW_TOKENS = 512      # per CLAUDE.md task cap
GREEDY_TEMP = 0.0
SAMPLE_TEMP = 0.8
SAMPLE_TOP_P = 0.95
RANDOM_SEED = 3483
WARMUP_PER_LEVEL = int(os.environ.get("EXP3483_WARMUP_PER_LEVEL", "8"))
CANDIDATE_LEVELS = (3, 4)
# Wall-time budget. Default 22 minutes per the task spec; overridable for a
# shorter first invocation (the resume path picks up where this leaves off).
WALL_BUDGET_S = int(os.environ.get("EXP3483_WALL_BUDGET_S", str(22 * 60)))

# MATH-500 local cache (read parquet directly; load_dataset has an ImportError
# in this venv). The benchmark id is recorded in the artifact for provenance.
BENCHMARK_REPO = "HuggingFaceH4/MATH-500"
MATH500_PARQUET = (
    Path.home()
    / ".cache/huggingface/datasets/MATH-500/test/0000.parquet"
)


def field_principles() -> dict[str, str]:
    """Per-field principle annotations (CLAUDE.md Principle-Annotated Fields).

    Each artifact field carries a one-line WHY so a future auditor (human or
    AI) understands what failure mode the field guards against, not just what
    it contains.
    """
    return {
        "honest_verdict": "Terminal verdict must start with complete:/success:/passed:/shipped_.",
        "inference_substrate": "live_llm_inference - real GGUF generation on GPU; 60s floor.",
        "corpus_path": "data/p01_difficulty_matched_generations.jsonl - the in-band corpus exp3484/3485/3487 consume.",
        "benchmark_id": "the exact MATH-500 dataset id + level(s) + seed.",
        "selected_levels": "the MATH level(s) chosen by the adaptive warm-up to land SC in band.",
        "per_level_probe_sc": "warm-up SC per candidate level - evidence the chosen level is in band.",
        "n_problems_completed": "problems with a full generation set after this run.",
        "n_problems_target": "the target (>=80 for headline-eligibility).",
        "n_problems_added_this_run": "problems newly generated this invocation (proves resume worked).",
        "k_samples": "sampled generations per problem - the matched-compute budget.",
        "per_step_traces_captured": "each generation carries a parsed step list for PROCESS-reward scoring.",
        "per_sample_logprobs_captured": "mean-token confidence stored per generation.",
        "warmup_self_consistency_accuracy": "majority-vote accuracy - MUST land in [0.40,0.70].",
        "self_consistency_in_headroom_band": "boolean: SC in [0.40,0.70] - the precondition that makes P0.1 testable.",
        "warmup_greedy_accuracy": "greedy accuracy over the corpus.",
        "model_specs": "the actual GGUF invoked (26B default or 31B fallback).",
        "random_seed": "determinism precondition for reproducibility.",
        "reproducibility_checksum": "content hash of benchmark split + model + seed.",
        "duration_s": "real live MoE generation takes wall time; 60s floor - sub-60s is the fabrication signal.",
    }


def gemma_chat_prompt(problem: str) -> str:
    r"""Wrap a MATH problem in the gemma-4 instruct turn format.

    gemma instruct models expect ``<start_of_turn>user ... <end_of_turn>`` then
    an open ``<start_of_turn>model`` turn. llama.cpp adds the BOS token itself.
    We ask explicitly for the final answer inside ``\boxed{}`` so the extractor
    has a deterministic target.
    """
    instruction = (
        f"{problem}\n\n"
        r"Solve the problem step by step. Put your final answer inside \boxed{}."
    )
    return (
        f"<start_of_turn>user\n{instruction}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def build_generation_record(
    text: str,
    token_logprobs: list[float | None] | None,
    gold_answer: str | None,
    mode: str,
    seed: int,
) -> dict[str, Any]:
    """Assemble one generation's record from raw model output.

    Pure (no GPU): given the completion ``text``, its per-token logprobs, and
    the gold answer, produce the structured row consumed by the corpus. We use
    the shared helpers so the extraction/normalization/step-parsing logic is
    identical to what the unit tests verify.
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


def build_problem_record(
    problem_meta: dict[str, Any],
    greedy: dict[str, Any],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the full per-problem JSONL row (greedy + k samples).

    ``sampled_answers`` is duplicated at the top level (normalized) so the
    self-consistency computation can read it directly without re-parsing every
    sample, matching the contract of ``self_consistency_accuracy``.
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


def classify_verdict(n_completed: int, in_band: bool, sc: float | None) -> str:
    """Return the terminal honest_verdict string for the run.

    The verdict starts with ``complete:`` per Verdict Terminal-Prefix
    Discipline. The branch encodes how far the corpus got: headline-eligible
    (>=80), scorable (40-79), partial (<40), or no in-band split found.
    """
    sc_str = "NA" if sc is None else f"{sc:.3f}"
    if not in_band:
        return "complete: blocked_no_in_band_split_found_sc_outside_band"
    if n_completed >= TARGET_N:
        return (
            f"complete: p01_difficulty_matched_corpus_headline_eligible_"
            f"n={n_completed}_sc={sc_str}"
        )
    if n_completed >= SCORABLE_N:
        return (
            f"complete: p01_difficulty_matched_corpus_scorable_partial_"
            f"n={n_completed}_resume_next_milestone"
        )
    return (
        f"complete: p01_difficulty_matched_corpus_partial_"
        f"n={n_completed}_resume_next_milestone"
    )


# ---------------------------------------------------------------------------
# GPU-bound orchestration (not unit-tested; requires CUDA + a cached GGUF).
# ---------------------------------------------------------------------------
def _load_math_records() -> list[dict[str, Any]]:
    """Load MATH-500 test problems for the candidate levels as record dicts.

    Reads the cached parquet directly (the ``datasets`` loader has an
    ImportError in this venv). Each record carries a stable ``problem_id``
    (the dataset's ``unique_id``) used for resume de-duplication.
    """
    import pandas as pd

    path = MATH500_PARQUET
    if not path.exists():
        # Fall back to a glob if the canonical path moved.
        candidates = list(
            (Path.home() / ".cache/huggingface/datasets").glob(
                "**/MATH-500/**/*.parquet"
            )
        )
        if not candidates:
            raise FileNotFoundError(f"MATH-500 parquet not found at {path}")
        path = candidates[0]
    df = pd.read_parquet(path)
    records: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        if int(row["level"]) not in CANDIDATE_LEVELS:
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
    """Append one JSON record as a line, flushing so a kill keeps the row."""
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


def _write(path: Path, artifact: dict[str, Any]) -> None:
    """Write the artifact JSON deliverable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=1)


def _build_artifact(
    *,
    verdict: str,
    duration_s: float,
    selected_levels: list[int],
    per_level_probe_sc: dict[str, float],
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
        "schema": "carnot.p01_difficulty_matched_corpus_v2",
        "duration_s": duration_s,
        "honest_verdict": verdict,
        "inference_substrate": "live_llm_inference",
        "corpus_path": str(CORPUS_PATH.relative_to(REPO_ROOT)),
        "benchmark_id": (
            f"{BENCHMARK_REPO} (test split), levels={selected_levels or list(CANDIDATE_LEVELS)}, "
            f"seed={RANDOM_SEED}"
        ),
        "selected_levels": selected_levels,
        "per_level_probe_sc": per_level_probe_sc,
        "n_problems_completed": n_completed,
        "n_problems_target": TARGET_N,
        "n_problems_added_this_run": n_added,
        "k_samples": K_SAMPLES,
        "per_step_traces_captured": n_completed > 0,
        "per_sample_logprobs_captured": n_completed > 0,
        "warmup_self_consistency_accuracy": sc,
        "self_consistency_in_headroom_band": bool(in_band),
        "warmup_greedy_accuracy": greedy_acc,
        "model_specs": model_specs,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": repro_checksum,
        "preconditions_checked": preconditions_checked,
        "status": status,
        "field_provenance": {k: {"principle": v} for k, v in field_principles().items()},
    }


def main() -> int:  # noqa: C901 - top-level orchestration is intentionally linear
    """Build the difficulty-matched corpus, resuming if it already exists."""
    t0 = time.time()
    from scripts.experiment_template import ExperimentTemplate, cached_sota_pair

    tmpl = ExperimentTemplate(EXP_ID, TITLE, str(DELIVERABLE))
    tmpl.setup()

    preconditions_checked: list[dict[str, Any]] = []

    # ---- Step 0a: CUDA precondition --------------------------------------
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
            selected_levels=[],
            per_level_probe_sc={},
            n_completed=0,
            n_added=0,
            sc=None,
            greedy_acc=None,
            model_specs=None,
            preconditions_checked=preconditions_checked,
            repro_checksum=None,
            status="blocked",
        )
        _write(DELIVERABLE, art)
        print("[exp3483] CUDA unavailable - wrote blocked artifact.")
        return 0

    # ---- Step 0b: SOTA GGUF tokenizer precondition (GGUF path, not HF) ----
    model_path = None
    model_name = None
    tok_ok = False
    try:
        pair = cached_sota_pair()
        if pair:
            model_path = pair[0].get("model_path")
            model_name = pair[0].get("name")
        if model_path and Path(model_path).exists():
            from llama_cpp import Llama

            probe = Llama(model_path=model_path, vocab_only=True, verbose=False)
            probe.tokenize(b"x")
            tok_ok = True
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[exp3483] tokenizer probe failed: {exc}")
        tok_ok = False
    preconditions_checked.append(
        {"resource": "sota_gguf_tokenizer", "available": tok_ok}
    )
    if not tok_ok:
        art = _build_artifact(
            verdict="complete: blocked_sota_gguf_tokenizer_unavailable",
            duration_s=time.time() - t0,
            selected_levels=[],
            per_level_probe_sc={},
            n_completed=0,
            n_added=0,
            sc=None,
            greedy_acc=None,
            model_specs=None,
            preconditions_checked=preconditions_checked,
            repro_checksum=None,
            status="blocked",
        )
        _write(DELIVERABLE, art)
        print("[exp3483] SOTA GGUF tokenizer unavailable - wrote blocked artifact.")
        return 0

    from scripts.experiment_template import _compute_repro_checksum

    repro_checksum = _compute_repro_checksum(
        RANDOM_SEED, [Path(__file__)], CORPUS_PATH
    )

    # ---- Model loader ----------------------------------------------------
    from llama_cpp import Llama

    print(f"[exp3483] loading model: {model_name} ({model_path})")
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1,
        seed=RANDOM_SEED,
        logits_all=True,  # required so create_completion returns token logprobs
        verbose=False,
    )
    model_specs = {"name": model_name, "model_path": model_path, "loader": "llama_cpp"}

    def _generate(prompt: str, temperature: float, seed: int) -> dict[str, Any]:
        """Run one llama.cpp completion and return text + token logprobs."""
        out = llm.create_completion(
            prompt,
            max_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            top_p=SAMPLE_TOP_P if temperature > 0 else 1.0,
            seed=seed,
            logprobs=1,
            stop=["<end_of_turn>", "<eos>"],
        )
        ch = out["choices"][0]
        text = ch.get("text", "")
        tlp = (ch.get("logprobs") or {}).get("token_logprobs") or []
        return {"text": text, "token_logprobs": tlp}

    def _gen_problem(meta: dict[str, Any]) -> dict[str, Any]:
        """Generate greedy + k sampled solutions for one problem."""
        prompt = gemma_chat_prompt(meta["problem"])
        g = _generate(prompt, GREEDY_TEMP, RANDOM_SEED)
        greedy = build_generation_record(
            g["text"], g["token_logprobs"], meta["gold_answer"], "greedy", RANDOM_SEED
        )
        samples: list[dict[str, Any]] = []
        for j in range(K_SAMPLES):
            s_seed = RANDOM_SEED + 1 + j
            s = _generate(prompt, SAMPLE_TEMP, s_seed)
            samples.append(
                build_generation_record(
                    s["text"], s["token_logprobs"], meta["gold_answer"], "sampled", s_seed
                )
            )
        return build_problem_record(meta, greedy, samples)

    def _budget_left() -> float:
        return WALL_BUDGET_S - (time.time() - t0)

    # ---- Load problems + resume state ------------------------------------
    all_records = _load_math_records()
    by_level: dict[int, list[dict[str, Any]]] = {lv: [] for lv in CANDIDATE_LEVELS}
    for rec in all_records:
        by_level.setdefault(rec["level"], []).append(rec)

    done_ids = completed_problem_ids(CORPUS_PATH)
    print(f"[exp3483] resume: {len(done_ids)} problems already in corpus.")

    per_level_probe_sc: dict[str, float] = {}

    # ---- ADAPTIVE WARM-UP -------------------------------------------------
    # Probe each candidate level on WARMUP_PER_LEVEL problems (skipping any
    # already in the corpus), generating full greedy+k so the warm-up rows are
    # reusable as corpus entries. Persist each completed problem immediately.
    n_added = 0
    probe_records_by_level: dict[int, list[dict[str, Any]]] = {}

    for level in CANDIDATE_LEVELS:
        if _budget_left() < 120:
            print(f"[exp3483] budget low before probing L{level}; stopping warm-up.")
            break
        pool = [r for r in by_level.get(level, []) if r["problem_id"] not in done_ids]
        probe_recs: list[dict[str, Any]] = []
        for meta in pool[:WARMUP_PER_LEVEL]:
            if _budget_left() < 90:
                print(f"[exp3483] budget low mid-probe L{level}; stopping.")
                break
            rec = _gen_problem(meta)
            _append_jsonl(CORPUS_PATH, rec)
            done_ids.add(rec["problem_id"])
            probe_recs.append(rec)
            n_added += 1
            mv_ok = "?"
            print(
                f"[exp3483] PROBE L{level} #{len(probe_recs)} pid={rec['problem_id']} "
                f"greedy_correct={rec['greedy_correct']} "
                f"left={_budget_left():.0f}s (mv={mv_ok})"
            )
        probe_records_by_level[level] = probe_recs
        if probe_recs:
            sc_level = self_consistency_accuracy(probe_recs)
            per_level_probe_sc[str(level)] = round(sc_level, 4)
            print(f"[exp3483] L{level} probe SC = {sc_level:.3f} (n={len(probe_recs)})")

    # Compute the 3+4 mix probe SC from the combined probe records.
    combined_probe = [r for recs in probe_records_by_level.values() for r in recs]
    if len(probe_records_by_level) >= 2 and combined_probe:
        per_level_probe_sc["3+4"] = round(self_consistency_accuracy(combined_probe), 4)

    # ---- Select the in-band level/mix ------------------------------------
    selected_levels: list[int] = []
    for level in CANDIDATE_LEVELS:
        sc = per_level_probe_sc.get(str(level))
        if sc is not None and in_headroom_band(sc):
            selected_levels = [level]
            break
    if not selected_levels:
        mix_sc = per_level_probe_sc.get("3+4")
        if mix_sc is not None and in_headroom_band(mix_sc):
            selected_levels = list(CANDIDATE_LEVELS)

    # If nothing is in band with the default model, the honest move is to stop
    # and resume next milestone (a 31B fallback re-probe is left for a future
    # invocation to keep this run within budget). Record the probe evidence.
    if not selected_levels:
        corpus_now = _read_corpus(CORPUS_PATH)
        overall_sc = (
            self_consistency_accuracy(corpus_now) if corpus_now else None
        )
        art = _build_artifact(
            verdict="complete: blocked_no_in_band_split_found_sc_outside_band",
            duration_s=time.time() - t0,
            selected_levels=[],
            per_level_probe_sc=per_level_probe_sc,
            n_completed=len(corpus_now),
            n_added=n_added,
            sc=overall_sc,
            greedy_acc=_greedy_accuracy(corpus_now),
            model_specs=model_specs,
            preconditions_checked=preconditions_checked,
            repro_checksum=repro_checksum,
            status="blocked",
        )
        _write(DELIVERABLE, art)
        print(
            f"[exp3483] no in-band split (per_level={per_level_probe_sc}); "
            f"wrote blocked artifact."
        )
        return 0

    print(f"[exp3483] selected levels: {selected_levels}")

    # ---- Fill remaining problems from the selected level pool ------------
    fill_pool: list[dict[str, Any]] = []
    for level in selected_levels:
        fill_pool.extend(
            r for r in by_level.get(level, []) if r["problem_id"] not in done_ids
        )

    for meta in fill_pool:
        if _budget_left() < 90:
            print("[exp3483] wall budget reached; finalizing.")
            break
        if len(_read_corpus(CORPUS_PATH)) >= TARGET_N:
            print("[exp3483] target N reached; finalizing.")
            break
        rec = _gen_problem(meta)
        _append_jsonl(CORPUS_PATH, rec)
        done_ids.add(rec["problem_id"])
        n_added += 1
        print(
            f"[exp3483] FILL pid={rec['problem_id']} L{rec['level']} "
            f"greedy_correct={rec['greedy_correct']} "
            f"n_added={n_added} left={_budget_left():.0f}s"
        )

    # ---- Finalize: recompute full-corpus SC self-check -------------------
    # Only count rows from the selected level(s) toward the in-band corpus.
    full_corpus = [
        r for r in _read_corpus(CORPUS_PATH) if r.get("level") in selected_levels
    ]
    final_sc = self_consistency_accuracy(full_corpus) if full_corpus else None
    greedy_acc = _greedy_accuracy(full_corpus)
    n_completed = len(full_corpus)
    in_band = final_sc is not None and in_headroom_band(final_sc)

    verdict = classify_verdict(n_completed, bool(in_band), final_sc)
    art = _build_artifact(
        verdict=verdict,
        duration_s=time.time() - t0,
        selected_levels=selected_levels,
        per_level_probe_sc=per_level_probe_sc,
        n_completed=n_completed,
        n_added=n_added,
        sc=final_sc,
        greedy_acc=greedy_acc,
        model_specs=model_specs,
        preconditions_checked=preconditions_checked,
        repro_checksum=repro_checksum,
        status="success",
    )
    _write(DELIVERABLE, art)
    print(
        f"[exp3483] DONE n={n_completed} sc={final_sc} in_band={in_band} "
        f"added={n_added} dur={time.time() - t0:.0f}s verdict={verdict}"
    )
    return 0


def _greedy_accuracy(corpus: list[dict[str, Any]]) -> float | None:
    """Fraction of corpus problems whose greedy generation was correct."""
    if not corpus:
        return None
    return sum(1 for r in corpus if r.get("greedy_correct")) / len(corpus)


if __name__ == "__main__":
    raise SystemExit(main())
