"""PHASE D (outer-loop bootstrap on GPU-1): regenerate MuSR candidate REASONING TRACES.

WHY: the conductor's PHASE D D1/D2 tasks skeleton (D1 blocked on a 404 base `Qwen/Qwen3.5-1.7B`;
D2 blocked on an empty logprob cache), so the decisive moat question never gets tested. The cached
checkpoints (results/distributional_energy_verifier_musr_checkpoints/) saved only answer labels, NOT
the candidates' reasoning text -- so a trained verifier has nothing to learn from. This regenerates the
K reasoning traces per question and PERSISTS them, so the companion trainer
(exp_phase_d_musr_verifier_train.py) can train an ORACLE-DISTINCT verifier and test beats-genuine-SC
(0.585, headroom_present=True per exp5015) on MuSR -- the cheap prompted proxy already failed (0.515-0.535),
the open question is whether a TRAINED verifier captures the unrealized headroom.

Reuses the EXACT generation harness from exp_distributional_energy_verifier_musr.py (same proposer,
prompt, K=8, gold parsing) so it is comparable. Resumable (skips saved q files; resume-not-restart per
the powering-run discipline). Proposer = Qwen3.5-9B-MTP GGUF on GPU-1 (outer-loop's GPU). Detached job.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "musr_traces"
N_Q = 200
K = 8
SEED = 20260630

# import the proven helpers from the original experiment (same generation behavior)
_spec = importlib.util.spec_from_file_location(
    "dev_musr", str(REPO / "scripts" / "experiments" / "exp_distributional_energy_verifier_musr.py")
)
dev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dev)


def _log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    # GPU-1 (outer-loop's), distinct port so it never collides with the conductor's GPU-0 generator.
    os.environ.setdefault("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")
    proposer = LocalGGUFProposer(
        repo_substr="Qwen3.5-9B-MTP", kv_quant="q8_0", no_think_prefix="/no_think\n",
        port=int(os.environ.get("CARNOT_PHASE_D_LLM_PORT", "8922")),
    )
    if not (proposer._healthy() or proposer._ensure_server()):
        _log("PROPOSER UNHEALTHY -- aborting (no fabrication)")
        return 1
    _log("proposer healthy")

    from datasets import load_dataset
    rows = list(load_dataset("TAUR-Lab/MuSR")["murder_mysteries"])[:N_Q]
    _log(f"loaded {len(rows)} MuSR murder_mysteries questions")

    t0 = time.time()
    done = 0
    for qi, row in enumerate(rows):
        out_f = OUT / f"q{qi:04d}.json"
        if out_f.exists():
            done += 1
            continue
        narrative = row.get("narrative") or row.get("context") or ""
        question = row.get("question") or ""
        choices = dev._choices(row)
        gold = row.get("answer_choice") or (choices[row["answer_index"]] if row.get("answer_index") is not None and choices else None)
        cands = []
        for k in range(K):
            c = dev._gen_candidate(proposer, narrative, question, choices, SEED + qi * 100 + k)
            c["correct"] = int(c.get("answer") is not None and gold is not None and c["answer"] == gold)
            cands.append(c)
        rec = {
            "q": qi, "question": question, "narrative": narrative[:8000], "choices": choices,
            "gold": gold, "n_candidates": len(cands), "candidates": cands,
        }
        out_f.write_text(json.dumps(rec))
        done += 1
        n_corr = sum(c["correct"] for c in cands)
        if qi % 10 == 0 or qi < 3:
            _log(f"q{qi:04d}: {len(cands)} traces, {n_corr}/{K} correct, "
                 f"{done}/{N_Q} done, {time.time()-t0:.0f}s elapsed")

    _log(f"TRACE-REGEN COMPLETE: {done}/{N_Q} questions, {time.time()-t0:.0f}s")
    print("PHASE_D_TRACE_REGEN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
