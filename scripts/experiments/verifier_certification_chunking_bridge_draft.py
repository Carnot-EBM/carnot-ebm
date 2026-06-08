"""DRAFT (Path 2 first step): the CHUNKING BRIDGE between v2 and v3.

v2 certified free-form p01 traces by FINE reasoning_steps -> 55% precision @ 24%
recall. v3 certified the FoVer corpus in its native multi-line CHUNK granularity ->
96.7% precision. The hypothesis: the gap is GRANULARITY -- the verifier is calibrated
on FoVer paragraph-sized step-chunks, and v2's fine steps were the wrong size. This
re-chunks the SAME p01 generated traces at FoVer granularity (paragraph split) and
re-measures, holding everything else identical to v2.

If precision climbs toward v3, the format transfer is a solvable chunking problem and
RFT (Path 2 proper) is clearly worth the fine-tune-harness build. If it stays ~v2,
the transfer is harder than the in-format result suggested.

CAVEAT: p01 has only TRACE-level gold (is_correct), so this is trace-level
certification (certify iff all chunks clean / max-chunk-error <= t) -- noisier than
per-step. The clean comparison is vs v2 (identical setup, only the chunking differs).

  .venv/bin/python scripts/experiments/verifier_certification_chunking_bridge_draft.py
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from carnot.eval.verifier_error_independence_scissor_at_scale import (
    FoVerPanel,
    score_carnot_ensemble,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACES = REPO_ROOT / "data" / "p01_difficulty_matched_generations_flattened_v2.jsonl"
OUT = REPO_ROOT / "results" / "verifier_certification_chunking_bridge.json"

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _fover_chunks(text: str) -> list[str]:
    """Split a free-form trace into FoVer-granularity (paragraph-sized) chunks."""

    body = _THINK.sub("", text).strip()
    # Paragraph split (FoVer step_text chunks are multi-line paragraphs).
    raw = re.split(r"\n\s*\n", body)
    chunks = []
    for c in raw:
        s = c.strip()
        if len(s) >= 12 and re.search(r"[a-zA-Z0-9]", s):
            chunks.append(s)
    return chunks


def _load(path: Path, limit: int | None = None) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ic = r.get("is_correct")
            chunks = _fover_chunks(str(r.get("text") or ""))
            if ic is None or not chunks:
                continue
            out.append({"chunks": chunks, "is_correct": bool(ic)})
            if limit and len(out) >= limit:
                break
    return out


def run(limit: int | None = None, write: bool = True) -> dict:
    traces = _load(TRACES, limit=limit)
    n_tr = len(traces)
    gold = [1 if t["is_correct"] else 0 for t in traces]
    base_rate = sum(gold) / n_tr if n_tr else 0.0
    mean_chunks = sum(len(t["chunks"]) for t in traces) / n_tr if n_tr else 0.0

    chunk_texts, owner = [], []
    for ti, t in enumerate(traces):
        for c in t["chunks"]:
            chunk_texts.append(c)
            owner.append(ti)

    panel = FoVerPanel(
        rows=tuple({"idx": i} for i in range(len(chunk_texts))),
        labels=tuple(0 for _ in chunk_texts),
        texts=tuple(chunk_texts),
        panel_sha256=hashlib.sha256("".join(chunk_texts).encode("utf-8")).hexdigest(),
    )
    scoring = score_carnot_ensemble(panel, REPO_ROOT)
    err = [float(s) for s in scoring.scores]
    pred = [int(p) for p in scoring.error_preds]

    trace_max_err = [0.0] * n_tr
    trace_any_flag = [0] * n_tr
    for i, ti in enumerate(owner):
        trace_max_err[ti] = max(trace_max_err[ti], err[i])
        if pred[i] == 1:
            trace_any_flag[ti] = 1

    clean = [i for i in range(n_tr) if trace_any_flag[i] == 0]
    a_prec = (sum(gold[i] for i in clean) / len(clean)) if clean else None
    a_rec = (sum(gold[i] for i in clean) / (sum(gold) or 1)) if clean else None

    lo, hi = min(trace_max_err), max(trace_max_err)
    grid = [lo + (hi - lo) * k / 20 for k in range(21)]
    sweep = []
    for t in grid:
        cert = [i for i in range(n_tr) if trace_max_err[i] <= t]
        if not cert:
            sweep.append({"threshold": round(t, 4), "n_certified": 0,
                          "precision": None, "recall_of_correct": None})
            continue
        tp = sum(gold[i] for i in cert)
        sweep.append({"threshold": round(t, 4), "n_certified": len(cert),
                      "precision": round(tp / len(cert), 4),
                      "recall_of_correct": round(tp / (sum(gold) or 1), 4)})
    usable = [p for p in sweep if p["recall_of_correct"] and p["recall_of_correct"] >= 0.20]
    rft_point = max(usable, key=lambda p: p["precision"]) if usable else None

    gate_pass = bool(rft_point and rft_point["precision"] and rft_point["precision"] >= 0.85)
    v2_prec = 0.5542  # the v2 RFT-point precision @ 24% recall, for direct comparison
    verdict = (
        f"complete: chunking_bridge_{'RECOVERS' if gate_pass else 'partial'}"
        f"_baserate{base_rate:.3f}_rftprec{rft_point['precision'] if rft_point else 'na'}"
        f"_vs_v2_{v2_prec}_meanchunks{mean_chunks:.1f}"
    )
    artifact = {
        "experiment": "verifier_certification_chunking_bridge_draft",
        "title": "verifier_certification_chunking_bridge",
        "honest_verdict": verdict,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "n_traces": n_tr,
        "n_chunks_scored": len(chunk_texts),
        "mean_chunks_per_trace": round(mean_chunks, 2),
        "gold_correct_base_rate": round(base_rate, 4),
        "all_clean_rule": {"n_certified": len(clean),
                           "precision": None if a_prec is None else round(a_prec, 4),
                           "recall_of_correct": None if a_rec is None else round(a_rec, 4)},
        "rft_operating_point": rft_point,
        "v2_finegrained_rft_precision_for_comparison": v2_prec,
        "precision_recall_sweep": sweep,
        "phase1_gate": "trace-level certification precision >= 0.85 at recall >= 0.20",
        "phase1_gate_pass": gate_pass,
        "interpretation": (
            "Compare rft precision to v2's 0.554 (same setup, finer chunks). A clear climb "
            "toward v3's in-format 0.967 confirms granularity/format transfer is the lever and "
            "RFT (Path 2) is viable. Flat ~v2 means trace-level transfer is harder than the "
            "per-step in-format result; note trace-level gold is noisier than FoVer per-step."
        ),
    }
    if write:
        OUT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", "utf-8")
    return artifact


if __name__ == "__main__":
    art = run()
    print(f"-> {art['honest_verdict']}")
    print(f"   n_traces={art['n_traces']} chunks={art['n_chunks_scored']} "
          f"mean_chunks/trace={art['mean_chunks_per_trace']} (v2 was ~11 fine steps)")
    print(f"   base_rate={art['gold_correct_base_rate']}")
    print(f"   all-clean: {art['all_clean_rule']}")
    print(f"   RFT point: {art['rft_operating_point']}  (v2 was 0.554 @ 24% recall)")
    print(f"   Phase-1 gate: {'PASS' if art['phase1_gate_pass'] else 'FAIL'}")
