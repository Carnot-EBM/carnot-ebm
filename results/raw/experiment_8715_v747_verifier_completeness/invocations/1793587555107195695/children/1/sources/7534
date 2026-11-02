#!/usr/bin/env python3
r"""Experiment 3552 - P0.1 Route 2 Headroom Corpus Greedy-Wrong Construction v3.

WHY THIS EXPERIMENT EXISTS
==========================
Every Route-2 energy-vs-SC test has failed because SC is near-optimal and
the mode IS the answer.
This script builds a positive-control corpus that GUARANTEES headroom AT THE GREEDY LEVEL:
keep a problem iff its GREEDY (temp-0) answer is WRONG but at least one of k>=16 
sampled candidates is CORRECT.
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

EXP_ID = 3552
TITLE = "P0.1 Route 2 Headroom Corpus Greedy-Wrong Construction v3"

CORPUS_PATH = REPO_ROOT / "data" / "p01_greedy_wrong_headroom_corpus.jsonl"

DELIVERABLE = (
    REPO_ROOT
    / "results"
    / "experiment_3552_p01_route2_headroom_corpus_greedy_wrong_construction_v3.json"
)

TARGET_N = 40
K_SAMPLES = 16
MAX_NEW_TOKENS = int(os.environ.get("EXP3552_MAX_NEW_TOKENS", "768"))
GREEDY_TEMP = 0.0
SAMPLE_TEMP = 0.8
SAMPLE_TOP_P = 0.95

_SEED_SRC = b"HuggingFaceH4/MATH-500:test:greedy_wrong_headroom:v3"
RANDOM_SEED: int = int(hashlib.sha256(_SEED_SRC).hexdigest()[:8], 16)

WALL_BUDGET_S = int(os.environ.get("EXP3552_WALL_BUDGET_S", str(22 * 60)))

BENCHMARK_REPO = "HuggingFaceH4/MATH-500"
MATH500_ARROW = (
    Path.home()
    / ".cache/huggingface/datasets/HuggingFaceH4___math-500/default/0.0.0"
    / "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/math-500-test.arrow"
)

def has_selectable_headroom(record: dict[str, Any]) -> bool:
    gold = record.get("gold_answer_norm")
    if gold is None:
        return False
    
    # GREEDY answer must be wrong
    greedy = record.get("greedy", {})
    greedy_ans = greedy.get("extracted_answer_norm")
    if greedy_ans == gold:
        return False

    sampled_answers = record.get("sampled_answers") or []
    # >=1 sampled candidate is CORRECT
    correct_present = any(
        normalize_answer(a) == normalize_answer(gold)
        for a in sampled_answers
        if a is not None
    )
    if not correct_present:
        return False

    return True

def compute_corpus_stats(
    kept_records: list[dict[str, Any]],
) -> dict[str, float | bool]:
    if not kept_records:
        return {
            "greedy_accuracy": 0.0,
            "oracle_accuracy": 0.0,
            "self_consistency_accuracy": 0.0,
            "selectable_headroom": 0.0,
            "oracle_exceeds_sc": False,
        }
    oracle_correct = sum(1 for r in kept_records if _oracle_is_correct(r))
    oracle_acc = oracle_correct / len(kept_records)
    
    greedy_correct = sum(1 for r in kept_records if r.get("greedy_correct"))
    greedy_acc = greedy_correct / len(kept_records)
    
    sc_acc = self_consistency_accuracy(kept_records)
    headroom = oracle_acc - sc_acc
    return {
        "greedy_accuracy": greedy_acc,
        "oracle_accuracy": oracle_acc,
        "self_consistency_accuracy": sc_acc,
        "selectable_headroom": headroom,
        "oracle_exceeds_sc": bool(oracle_acc > sc_acc),
    }

def _oracle_is_correct(record: dict[str, Any]) -> bool:
    gold = record.get("gold_answer_norm")
    if gold is None:
        return False
    return any(
        normalize_answer(a) == normalize_answer(gold)
        for a in (record.get("sampled_answers") or [])
        if a is not None
    )

def classify_verdict_3552(
    n_kept: int,
    oracle_acc: float,
    sc_acc: float,
) -> str:
    if n_kept >= TARGET_N and oracle_acc > sc_acc:
        return (
            f"complete: p01_greedy_wrong_headroom_corpus_built"
            f"_n={n_kept}"
            f"_oracle_{oracle_acc:.3f}"
            f"_exceeds_sc_{sc_acc:.3f}"
        )
    if n_kept > 0 and oracle_acc > sc_acc:
        return (
            f"complete: p01_greedy_wrong_headroom_corpus_partial"
            f"_n={n_kept}"
            f"_below_target_{TARGET_N}"
            f"_resume_next_milestone"
        )
    return "complete: p01_sc_tracks_oracle_even_when_greedy_wrong_route2_premise_terminally_bounded_on_nl_math"

def field_principles_3552() -> dict[str, str]:
    return {
        "honest_verdict": "complete:/success:/passed:/shipped_ prefix.",
        "inference_substrate": "live_llm_inference",
        "corpus_path": "data/p01_greedy_wrong_headroom_corpus.jsonl — the new positive-control corpus exp3553 reads.",
        "construction_criterion": "string: 'greedy-wrong AND >=1 of k>=16 sampled correct' — documents WHY headroom exists by construction (distinct from exp3530's difficulty-band filter).",
        "k_candidates": ">=16 — doubled vs exp3530 so minority-correct answers surface.",
        "n_problems_kept": "problems with the greedy-wrong-recoverable property (target >=40 scorable).",
        "greedy_accuracy": "temp-0 greedy accuracy over the kept corpus — 0 by construction (every kept problem is greedy-wrong).",
        "self_consistency_accuracy": "SC majority-vote accuracy over the kept corpus — the baseline to beat.",
        "oracle_accuracy": "accuracy if the correct answer is always selected when present among k — the reranker upper bound.",
        "selectable_headroom": "oracle_accuracy - self_consistency_accuracy — MUST be > 0; the property exp3530 could not construct.",
        "oracle_exceeds_sc": "boolean: oracle STRICTLY > SC — the precondition for a meaningful Route-2 test.",
        "per_step_traces_captured": "boolean: each generation carries a parsed step list (for the step->final aggregation scorer).",
        "model_specs": "the actual GGUF invoked (26B default or 31B/35B fallback).",
        "random_seed": "determinism; content-derived, not the experiment number.",
        "reproducibility_checksum": "content hash of benchmark split + model + seed.",
        "duration_s": "real live MoE generation with k>=16 takes wall time; 60s floor when CUDA is available.",
    }

def _build_generation_record(
    text: str,
    token_logprobs: list[float | None] | None,
    gold_answer: str | None,
    mode: str,
    seed: int,
) -> dict[str, Any]:
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
        "has_selectable_headroom": None,
    }

def _build_artifact(
    *,
    verdict: str,
    duration_s: float,
    n_kept: int,
    greedy_acc: float,
    oracle_acc: float,
    sc_acc: float,
    headroom: float,
    oracle_exceeds_sc: bool,
    per_step_traces: bool,
    model_specs: dict[str, Any] | None,
    repro_checksum: str | None,
) -> dict[str, Any]:
    return {
        "honest_verdict": verdict,
        "inference_substrate": "live_llm_inference",
        "corpus_path": str(CORPUS_PATH.relative_to(REPO_ROOT)),
        "construction_criterion": "greedy-wrong AND >=1 of k>=16 sampled correct",
        "k_candidates": K_SAMPLES,
        "n_problems_kept": n_kept,
        "greedy_accuracy": greedy_acc,
        "self_consistency_accuracy": sc_acc,
        "oracle_accuracy": oracle_acc,
        "selectable_headroom": headroom,
        "oracle_exceeds_sc": oracle_exceeds_sc,
        "per_step_traces_captured": per_step_traces,
        "model_specs": model_specs,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": repro_checksum,
        "duration_s": duration_s,
        "field_provenance": {
            k: {"principle": v} for k, v in field_principles_3552().items()
        },
    }

def _load_math_records() -> list[dict[str, Any]]:
    import pyarrow.ipc as ipc  # noqa: E402

    path = MATH500_ARROW
    if not path.exists():
        base = Path.home() / ".cache/huggingface/datasets"
        candidates = list(base.glob("*math*/**/*.arrow")) + list(
            base.glob("*math*/**/*.parquet")
        )
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
        uid = row.get("unique_id")
        pid = str(uid) if uid is not None else f"row{idx}"
        records.append(
            {
                "problem_id": pid,
                "level": row.get("level"),
                "subject": row.get("subject"),
                "problem": str(row["problem"]),
                "gold_answer": str(row["answer"]),
            }
        )
    return records

def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

def _read_corpus(path: Path) -> list[dict[str, Any]]:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=1)

def _gemma_chat_prompt(problem: str) -> str:
    instruction = (
        f"{problem}\n\n"
        r"Solve the problem step by step. Put your final answer inside \boxed{}."
    )
    return (
        f"<start_of_turn>user\n{instruction}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

def _qwen_chat_prompt(problem: str) -> str:
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
    t0 = time.time()
    from scripts.experiment_template import ExperimentTemplate, _compute_repro_checksum  # noqa: E402
    from carnot.inference.sota_models import cached_sota_pair  # noqa: E402

    tmpl = ExperimentTemplate(EXP_ID, TITLE, str(DELIVERABLE))
    tmpl.setup()

    try:
        import torch  # noqa: E402
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False

    if not cuda_ok:
        art = _build_artifact(
            verdict="complete: blocked_cuda_unavailable",
            duration_s=time.time() - t0,
            n_kept=0,
            greedy_acc=0.0,
            oracle_acc=0.0,
            sc_acc=0.0,
            headroom=0.0,
            oracle_exceeds_sc=False,
            per_step_traces=False,
            model_specs=None,
            repro_checksum=None,
        )
        _write_artifact(DELIVERABLE, art)
        print("[exp3552] CUDA unavailable — wrote blocked artifact.", flush=True)
        return 0

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
    except Exception as exc:  # pragma: no cover
        print(f"[exp3552] tokenizer probe failed: {exc}", flush=True)
        tok_ok = False

    if not tok_ok:
        art = _build_artifact(
            verdict="complete: blocked_sota_gguf_tokenizer_unavailable",
            duration_s=time.time() - t0,
            n_kept=0,
            greedy_acc=0.0,
            oracle_acc=0.0,
            sc_acc=0.0,
            headroom=0.0,
            oracle_exceeds_sc=False,
            per_step_traces=False,
            model_specs=None,
            repro_checksum=None,
        )
        _write_artifact(DELIVERABLE, art)
        print("[exp3552] SOTA GGUF tokenizer unavailable — wrote blocked artifact.", flush=True)
        return 0

    repro_checksum = _compute_repro_checksum(
        RANDOM_SEED, [Path(__file__)], CORPUS_PATH
    )

    from llama_cpp import Llama  # noqa: E402
    print(f"[exp3552] loading model: {model_name} ({model_path})", flush=True)
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
        "prompt_format": "qwen_chatml" if prompt_fn is _qwen_chat_prompt else "gemma_instruct",
    }

    def _generate(prompt: str, temperature: float, seed: int) -> dict[str, Any]:
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

    done_ids = completed_problem_ids(CORPUS_PATH)
    existing_kept = _read_corpus(CORPUS_PATH)
    print(f"[exp3552] resume: {len(done_ids)} problems in corpus already kept.", flush=True)

    all_records = _load_math_records()
    fill_pool = [r for r in all_records if r["problem_id"] not in done_ids]
    print(f"[exp3552] pool: {len(all_records)} total, {len(fill_pool)} not yet attempted.", flush=True)

    n_attempted = 0
    kept_this_run: list[dict[str, Any]] = []

    for meta in fill_pool:
        if _budget_left() < 90:
            print("[exp3552] wall budget reached; finalizing.", flush=True)
            break
        rec = _gen_problem(meta)
        n_attempted += 1
        headroom_flag = has_selectable_headroom(rec)
        rec["has_selectable_headroom"] = headroom_flag

        all_kept = existing_kept + kept_this_run
        if all_kept:
            run_stats = compute_corpus_stats(all_kept)
            run_oracle = run_stats["oracle_accuracy"]
            run_sc = run_stats["self_consistency_accuracy"]
            run_greedy = run_stats["greedy_accuracy"]
        else:
            run_oracle = 0.0
            run_sc = 0.0
            run_greedy = 0.0

        if headroom_flag:
            _append_jsonl(CORPUS_PATH, rec)
            done_ids.add(rec["problem_id"])
            kept_this_run.append(rec)

        n_kept_total = len(existing_kept) + len(kept_this_run)
        print(
            f"[exp3552] pid={rec['problem_id']}"
            f" L{rec['level']}"
            f" kept={headroom_flag}"
            f" n_kept={n_kept_total}"
            f" n_tried={n_attempted}"
            f" greedy={run_greedy:.3f}"
            f" oracle={run_oracle:.3f}"
            f" sc={run_sc:.3f}"
            f" gap={run_oracle - run_sc:.3f}"
            f" budget_left={_budget_left():.0f}s",
            flush=True,
        )

        if n_kept_total >= TARGET_N * 2:
            print(f"[exp3552] n_kept={n_kept_total} >= {TARGET_N * 2}; stopping.", flush=True)
            break

    final_kept = _read_corpus(CORPUS_PATH)
    n_kept_final = len(final_kept)
    if final_kept:
        stats = compute_corpus_stats(final_kept)
        greedy_acc = stats["greedy_accuracy"]
        oracle_acc = stats["oracle_accuracy"]
        sc_acc = stats["self_consistency_accuracy"]
        headroom_val = stats["selectable_headroom"]
        oracle_exceeds_sc = bool(stats["oracle_exceeds_sc"])
    else:
        greedy_acc = 0.0
        oracle_acc = 0.0
        sc_acc = 0.0
        headroom_val = 0.0
        oracle_exceeds_sc = False

    verdict = classify_verdict_3552(n_kept_final, oracle_acc, sc_acc)
    art = _build_artifact(
        verdict=verdict,
        duration_s=time.time() - t0,
        n_kept=n_kept_final,
        greedy_acc=greedy_acc,
        oracle_acc=oracle_acc,
        sc_acc=sc_acc,
        headroom=headroom_val,
        oracle_exceeds_sc=oracle_exceeds_sc,
        per_step_traces=n_kept_final > 0,
        model_specs=model_specs,
        repro_checksum=repro_checksum,
    )
    _write_artifact(DELIVERABLE, art)
    print(
        f"[exp3552] DONE n_kept={n_kept_final} oracle={oracle_acc:.3f} sc={sc_acc:.3f}"
        f" headroom={headroom_val:.3f} dur={time.time() - t0:.0f}s"
        f" verdict={verdict}",
        flush=True,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
