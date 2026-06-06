#!/usr/bin/env python3
"""DRAFT — Verifier efficiency head-to-head (the verifier-proof EFFICIENCY axis).

WHY (2026-06-06 strategic reframe, ops/north-star.md §5; memory
`hybrid-pragmatic-architecture`): energy-as-generator is closed-negative; Carnot is
the hybrid's energy VERIFIER, and the verifier's value is unproven. The operator's
win condition is EFFICIENCY-PARITY: the energy verifier earns its place if it is
"equally effective as the LM at lower cost/latency" (no accuracy edge required,
though accuracy upside is pursued). The ACCURACY/moat axis already has a harness
(`carnot.eval.moat_scissor_in_distribution`, residual_catch_rate). This script is
the MISSING EFFICIENCY axis: on one common in-distribution corpus, score with three
verifiers and report, per method, BOTH accuracy (AUROC + bootstrap CI, for parity)
AND cost (per-item wall-clock + generated tokens, for the ratio). Target headline:
"parity at 10-100x cheaper."

THREE VERIFIERS (the baselines to match/beat):
  1. ENERGY  = Carnot ensemble (score_carnot_ensemble): text verifiers (Tier0r/
     Tier0u) + FR-11 memory, a forward-pass/symbolic score, NO LLM generation.
  2. SELF    = terse LLM self-verification (score_reasoner_with_llama_cpp,
     max_tokens=10): the generator model asked YES/NO per step. The moat comparison.
  3. JUDGE   = CoT LLM-as-judge (this module, max_tokens~256): the model reasons
     then verdicts. The "expensive but maybe more accurate" baseline. The efficiency
     comparison — ENERGY should match its accuracy at a fraction of the cost.

THE EFFICIENCY METRIC (new first-class result): for each method, median per-item
latency (ms), total generated tokens, and the cost RATIO vs ENERGY. A clean result
is "ENERGY AUROC within CI of JUDGE/SELF, at Nx lower latency/tokens." Report cost
HONESTLY (matched hardware; ENERGY's neural sub-verifiers are not free; JUDGE cost
scales with CoT length).

INFERENCE SUBSTRATE: verifier_scoring_against_cached_candidates + live llama.cpp
(JUDGE/SELF generate). Duration floor: real (the LLM passes take real wall-clock).

================================  STATUS: DRAFT  ================================
This reuses the REAL functions from
`carnot.eval.verifier_error_independence_scissor_at_scale` and
`carnot.eval.moat_scissor_in_distribution`. ONE wiring point must be confirmed by a
smoke test before a real run: `_panel_from_exp3884()` (the corpus-row -> FoVerPanel
adapter — verify the text/label field names against the live exp3884 corpus). Run
`--smoke` first (1-3 items) to confirm all three scorers return aligned scores, THEN
the full run. Do NOT report numbers from this until the smoke test is green.
================================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

OUTPUT_REL = Path("results/experiment_verifier_efficiency_headtohead.json")
EXP3884_ARTIFACT_REL = Path("results/experiment_3884_in_distribution_error_rich_corpus.json")
INFERENCE_SUBSTRATE = "verifier_scoring_against_cached_candidates_plus_live_llama_cpp"

# Cached SOTA GGUF candidates (CLAUDE.md "SOTA Local Models"); prefer the lightweight
# gemma-4-12B for the many per-item LLM calls, fall back to the larger MoE.
GGUF_GLOBS = (
    "~/.cache/huggingface/hub/models--unsloth--gemma-4-12B-it-GGUF/**/*.gguf",
    "~/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/**/*.gguf",
    "~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/**/*.gguf",
)

FIELD_PRINCIPLES = {
    "energy_auroc": "Accuracy of the cheap forward-pass verifier — the parity target.",
    "judge_auroc": "Accuracy of the expensive CoT LLM-judge — the baseline ENERGY must match.",
    "auroc_parity_within_ci": "True if ENERGY's AUROC CI overlaps JUDGE's — 'equally effective' (operator win condition).",
    "cost_ratio_judge_over_energy_latency": "JUDGE median latency / ENERGY median latency — the efficiency win ('Nx cheaper').",
    "cost_ratio_judge_over_energy_tokens": "JUDGE generates tokens; ENERGY generates 0 — the structural cost asymmetry.",
    "duration_s": "Real wall-clock; live llama.cpp passes take real time (fabrication floor).",
    "inference_substrate": "Declares verifier-scoring + live llama.cpp per Inference-Substrate Discipline.",
    "preconditions_checked": "Records which resources were verified before any scoring (no fabrication on missing resource).",
    "honest_verdict": "Terminal-prefixed self-declared state for the conductor reconciler.",
}


# --------------------------------------------------------------------------- #
# Preconditions (Pre-Launch Discipline — check BEFORE any scoring; never fabricate)
# --------------------------------------------------------------------------- #
@dataclass
class Precondition:
    resource: str
    available: bool
    detail: str


def _glob_first(pattern: str) -> str | None:
    import glob

    hits = sorted(glob.glob(os.path.expanduser(pattern), recursive=True))
    return hits[0] if hits else None


def check_preconditions() -> tuple[list[Precondition], dict[str, Any]]:
    checks: list[Precondition] = []
    ctx: dict[str, Any] = {}

    art = REPO_ROOT / EXP3884_ARTIFACT_REL
    checks.append(Precondition("exp3884_corpus_artifact", art.is_file(), str(EXP3884_ARTIFACT_REL)))

    try:
        import llama_cpp  # noqa: F401

        checks.append(Precondition("llama_cpp_import", True, "import llama_cpp OK"))
    except Exception as exc:  # noqa: BLE001
        checks.append(Precondition("llama_cpp_import", False, repr(exc)))

    model_path = next((p for g in GGUF_GLOBS if (p := _glob_first(g))), None)
    checks.append(Precondition("cached_sota_gguf", model_path is not None, model_path or "no GGUF in cache"))
    ctx["model_path"] = model_path

    try:
        import torch

        cuda = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
        checks.append(Precondition("cuda", cuda, f"cuda={cuda}"))
    except Exception as exc:  # noqa: BLE001
        checks.append(Precondition("cuda", False, repr(exc)))

    try:
        from carnot.eval.verifier_error_independence_scissor_at_scale import (  # noqa: F401
            FoVerPanel,
            parse_reasoner_error_score,
            reasoner_self_verify_prompt,
            score_carnot_ensemble,
            score_reasoner_with_llama_cpp,
        )
        from carnot.eval.moat_scissor_in_distribution import load_exp3884_panel  # noqa: F401

        checks.append(Precondition("carnot_verify_import", True, "scissor + moat_scissor import OK"))
    except Exception as exc:  # noqa: BLE001
        checks.append(Precondition("carnot_verify_import", False, repr(exc)))

    return checks, ctx


# --------------------------------------------------------------------------- #
# Panel adapter (THE WIRING POINT — confirm field names in the smoke test)
# --------------------------------------------------------------------------- #
def _panel_from_exp3884(repo_root: Path, limit: int | None):
    """Load the exp3884 corpus -> FoVerPanel(rows, labels, texts).

    DRAFT: reuses moat_scissor.load_exp3884_panel for the corpus + provenance, then
    extracts (text, label) per row. Confirm the field names against the live corpus
    in --smoke (the corpus row schema is the one thing not verified here)."""
    from carnot.eval.moat_scissor_in_distribution import load_exp3884_panel
    from carnot.eval.verifier_error_independence_scissor_at_scale import FoVerPanel

    exp_panel = load_exp3884_panel(repo_root)  # Exp3884Panel: rows + disk-backed scores
    rows = list(getattr(exp_panel, "rows"))
    if limit:
        rows = rows[:limit]

    def _text(row: dict[str, Any]) -> str:
        for k in ("step_text", "text", "step", "candidate_text", "reasoning_step"):
            if isinstance(row.get(k), str):
                return row[k]
        raise KeyError(f"no step-text field in row keys={list(row)[:12]}")

    def _label(row: dict[str, Any]) -> int:
        # 1 == this step is an ERROR (matches parse_reasoner_error_score convention)
        for k in ("is_error", "error", "label", "incorrect", "gold_incorrect"):
            if k in row:
                return int(bool(row[k]))
        raise KeyError(f"no label field in row keys={list(row)[:12]}")

    labels = tuple(_label(r) for r in rows)
    texts = tuple(_text(r) for r in rows)
    panel = FoVerPanel(rows=tuple(rows), labels=labels, texts=texts)
    return panel, exp_panel


# --------------------------------------------------------------------------- #
# JUDGE baseline (new) — CoT LLM-as-judge, cost-instrumented
# --------------------------------------------------------------------------- #
def _judge_prompt(step_text: str) -> str:
    return (
        "You are a careful math-reasoning checker. Think briefly step by step about "
        "whether the following reasoning step is correct, then on the LAST line output "
        "exactly 'VERDICT: YES' if correct or 'VERDICT: NO' if it contains an error.\n\n"
        f"Step: {step_text}\n"
    )


def _parse_judge(response: str) -> int:
    tail = response.strip().lower().splitlines()[-1] if response.strip() else ""
    # 1 == error (NO == contains an error)
    if "verdict:" in tail:
        return 1 if "no" in tail.split("verdict:", 1)[1] else 0
    return 1 if "no" in tail else 0


def score_llm_judge_cot(panel, model_specs: dict[str, Any], *, max_tokens: int = 256,
                        llama_factory: Callable[..., Any] | None = None):
    """Live CoT LLM-as-judge. Returns (error_scores, raw, total_tokens)."""
    if llama_factory is None:
        from llama_cpp import Llama

        llama_factory = Llama
    llm = llama_factory(model_path=str(model_specs["model_path"]), n_gpu_layers=-1,
                        n_ctx=2048, n_batch=256, verbose=False)
    scores, raw, tok = [], [], 0
    for text in panel.texts:
        out = llm(_judge_prompt(text), max_tokens=max_tokens, temperature=0.0, stop=["\n\n"])
        resp = str(out["choices"][0]["text"])
        raw.append(resp)
        tok += int(out.get("usage", {}).get("completion_tokens", len(resp.split())))
        scores.append(_parse_judge(resp))
    return tuple(scores), tuple(raw), tok


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    pos = [s for l, s in zip(labels, scores) if l == 1]
    neg = [s for l, s in zip(labels, scores) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _auroc_ci(labels, scores, *, seed: int = 0, resamples: int = 1000) -> tuple[float, float, float]:
    import random

    rng = random.Random(seed)
    n = len(labels)
    boots = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        a = _auroc([labels[i] for i in idx], [scores[i] for i in idx])
        if a == a:  # not nan
            boots.append(a)
    boots.sort()
    lo = boots[int(0.025 * len(boots))] if boots else float("nan")
    hi = boots[int(0.975 * len(boots)) - 1] if boots else float("nan")
    return _auroc(labels, scores), lo, hi


def _timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1-3 item wiring check; do NOT report numbers")
    ap.add_argument("--limit", type=int, default=None, help="cap corpus items (None = all)")
    ap.add_argument("--judge-max-tokens", type=int, default=256)
    ap.add_argument("--out", default=str(OUTPUT_REL))
    args = ap.parse_args(argv)
    if args.smoke and args.limit is None:
        args.limit = 3

    t_start = time.perf_counter()
    checks, ctx = check_preconditions()
    pc = [{"resource": c.resource, "available": c.available} for c in checks]
    missing = [c for c in checks if not c.available]
    if missing:
        verdict = "blocked_" + "_".join(sorted(c.resource for c in missing))
        art = {"experiment": "verifier_efficiency_headtohead", "honest_verdict": verdict,
               "preconditions_checked": pc, "inference_substrate": "blocked_precondition",
               "duration_s": time.perf_counter() - t_start, "field_principles": FIELD_PRINCIPLES,
               "blocked_detail": {c.resource: c.detail for c in missing}}
        Path(REPO_ROOT / args.out).write_text(json.dumps(art, indent=2))
        print(f"BLOCKED: {verdict}", file=sys.stderr)
        for c in missing:
            print(f"  - {c.resource}: {c.detail}", file=sys.stderr)
        return 1

    from carnot.eval.verifier_error_independence_scissor_at_scale import (
        score_carnot_ensemble, score_reasoner_with_llama_cpp,
    )

    panel, _exp_panel = _panel_from_exp3884(REPO_ROOT, args.limit)
    labels = list(panel.labels)
    model_specs = {"model_path": ctx["model_path"], "loader": "llama_cpp.Llama"}

    # --- three verifiers, each cost-instrumented ---
    energy, t_energy = _timed(lambda: score_carnot_ensemble(panel, REPO_ROOT))
    self_, t_self = _timed(lambda: score_reasoner_with_llama_cpp(panel, model_specs, max_tokens=10))
    judge_scores, t_judge = _timed(lambda: score_llm_judge_cot(panel, model_specs,
                                                               max_tokens=args.judge_max_tokens))
    judge_err, _judge_raw, judge_tok = judge_scores

    n = len(labels)
    energy_scores = list(energy.scores)
    self_err = list(self_.error_scores)

    e_auroc, e_lo, e_hi = _auroc_ci(labels, energy_scores)
    j_auroc, j_lo, j_hi = _auroc_ci(labels, [float(x) for x in judge_err])
    s_auroc, s_lo, s_hi = _auroc_ci(labels, [float(x) for x in self_err])
    parity = (e_hi >= j_lo) and (j_hi >= e_lo)  # CIs overlap

    art = {
        "experiment": "verifier_efficiency_headtohead",
        "honest_verdict": ("smoke_wiring_check_only_not_headline"
                           if args.smoke else
                           f"complete_efficiency_headtohead_parity_{str(parity).lower()}"
                           f"_costratio_lat_{(t_judge / max(t_energy,1e-9)):.1f}x"),
        "n_items": n,
        "energy_auroc": e_auroc, "energy_auroc_ci95": [e_lo, e_hi],
        "judge_auroc": j_auroc, "judge_auroc_ci95": [j_lo, j_hi],
        "self_verify_auroc": s_auroc, "self_verify_auroc_ci95": [s_lo, s_hi],
        "auroc_parity_within_ci": parity,
        "cost_energy_total_s": t_energy, "cost_self_total_s": t_self, "cost_judge_total_s": t_judge,
        "cost_energy_per_item_ms": 1000 * t_energy / max(n, 1),
        "cost_judge_per_item_ms": 1000 * t_judge / max(n, 1),
        "cost_ratio_judge_over_energy_latency": t_judge / max(t_energy, 1e-9),
        "judge_completion_tokens_total": judge_tok,
        "energy_completion_tokens_total": 0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": pc,
        "model_path": ctx["model_path"],
        "duration_s": time.perf_counter() - t_start,
        "field_principles": FIELD_PRINCIPLES,
        "draft_note": "Confirm _panel_from_exp3884 field names via --smoke before citing numbers.",
    }
    Path(REPO_ROOT / args.out).write_text(json.dumps(art, indent=2))
    print(json.dumps({k: v for k, v in art.items() if k not in ("preconditions_checked", "field_principles")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
