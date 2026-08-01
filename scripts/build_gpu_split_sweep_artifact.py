#!/usr/bin/env python3
"""Build the analysis artifact for the 2026-07-31 GPU layer-split sweep.

The measurement harness (`results/arc_gpu_split_sweep_20260731/harness/gpu_split_measure.py`)
emits a bare list of five per-arm rows. This turns those rows into the artifact the project's
linters and readers expect: declared substrate, falsifiable gates evaluated against the
measured numbers rather than asserted, and the caveats stated at the same volume as the win.

Rebuild:
    .venv/bin/python scripts/build_gpu_split_sweep_artifact.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
D = REPO / "results" / "arc_gpu_split_sweep_20260731"
OUT = REPO / "results" / "outer_loop_arc_gpu_layer_split_sweep_20260731.json"

# Wall clock of the run, from the log file's create/modify stamps. Recorded rather than
# recomputed because the per-arm load_s + gen_wall_s (110.0 s total) EXCLUDES health polling,
# teardown and model eviction between arms, and would understate the run by ~2x.
RUN_START = "2026-07-31T22:59:41-04:00"
RUN_END = "2026-07-31T23:03:01-04:00"
DURATION_S = 199.8


def rows() -> list[dict]:
    src = D / "sweep_run.log"
    return [json.loads(x) for x in src.read_text().splitlines() if x.startswith("{")]


def main() -> int:
    by = {r["arm"]: r for r in rows()}

    def peak(a: str) -> int:
        return max(by[a]["residency"]["per_card_mib"].values())

    def cmp(base: str, arm: str) -> dict:
        A, B = by[base], by[arm]
        pa, pb = peak(base), peak(arm)
        ta, tb = A["decode_tok_s"], B["decode_tok_s"]
        return {
            "baseline_arm": base,
            "arm": arm,
            "decode_tok_s": {
                "baseline": ta,
                "arm": tb,
                "pct_change": round(100 * (tb - ta) / ta, 1),
            },
            "peak_per_card_mib": {
                "baseline": pa,
                "arm": pb,
                "delta_mib": pb - pa,
                "pct_change": round(100 * (pb - pa) / pa, 1),
            },
            "aggregate_mib": {
                "baseline": A["residency"]["total_mib"],
                "arm": B["residency"]["total_mib"],
                "delta_mib": B["residency"]["total_mib"] - A["residency"]["total_mib"],
            },
            "imbalance_mib": B["residency"]["max_minus_min_mib"],
            "imbalance_pct_of_card_share": round(100 * B["residency"]["max_minus_min_mib"] / pb, 2),
        }

    split32 = cmp("single_ctx32768", "split2_ctx32768")
    split81 = cmp("single_ctx81920", "split2_ctx81920")
    cpu32 = cmp("single_ctx32768", "cpu12_ctx32768")

    # ---- gates, fixed in PLAN.md BEFORE the run ------------------------------------------
    g_free = (
        abs(split32["decode_tok_s"]["pct_change"]) <= 10
        and abs(split81["decode_tok_s"]["pct_change"]) <= 10
    )
    g_halved = (
        split32["peak_per_card_mib"]["pct_change"] <= -40
        and split81["peak_per_card_mib"]["pct_change"] <= -40
    )
    g_even = (
        split32["imbalance_pct_of_card_share"] < 5 and split81["imbalance_pct_of_card_share"] < 5
    )

    art = {
        "experiment": "outer_loop_arc_gpu_layer_split_sweep_20260731",
        "schema": "carnot.gpu_layer_split_sweep.v1",
        "question": (
            "Does `-sm layer` across both RTX 3090s give near-single-GPU throughput at roughly "
            "half the per-device VRAM, and does it do so EVENLY? The only VRAM lever the ARC "
            "agent uses today is `-ot ...=CPU`, which spills dense FFN weights to host RAM."
        ),
        "honest_verdict": "complete_layer_split_is_free_and_dominates_the_cpu_offload_lever",
        "acceptance_gate_splitting_is_free": g_free,
        "acceptance_gate_per_card_roughly_halved": g_halved,
        "acceptance_gate_split_is_even": g_even,
        "acceptance_gate_passed": bool(g_free and g_halved and g_even),
        "gate_principles": {
            "acceptance_gate_splitting_is_free": (
                "decode within +-10% of the single-GPU arm. Distinguishes 'splitting costs "
                "nothing' from 'splitting costs less than CPU offload' -- only the former "
                "justifies making it the default lever."
            ),
            "acceptance_gate_per_card_roughly_halved": (
                "peak per-card <= -40%. A split that does not actually halve per-card residency "
                "buys no headroom and the whole exercise is moot."
            ),
            "acceptance_gate_split_is_even": (
                "imbalance < 5% of a card's share. `-ts 1,1` is passed explicitly because "
                "llama.cpp's DEFAULT splits by AVAILABLE VRAM, so a busy card would skew it; "
                "this gate is what proves the explicit ratio took effect."
            ),
        },
        "headline": (
            "Layer-splitting is FREE: +1.0% decode at n_ctx 32768 and +0.6% at the shipped "
            "81920, while cutting peak per-card residency 47.2% / 45.9% (-9640 / -10516 MiB). "
            "The CPU-offload lever in use today costs 59.8% of decode throughput to save only "
            "10.5% of per-card VRAM. Layer split strictly dominates it on both axes."
        ),
        "the_caveat_that_must_travel_with_the_headline": (
            "AGGREGATE VRAM across both cards goes UP when split: +984 MiB at 32768 and +1704 "
            "MiB at 81920, because each card carries its own compute buffers and non-layer "
            "state. The win is PER-CARD headroom, not total memory. On a single-GPU host there "
            "is no win at all -- there is a small loss. Anyone quoting 'halves VRAM' without "
            "this qualifier is overstating it."
        ),
        "operational_consequence": (
            "The shipped n_ctx 81920 DOES fit one 3090 (22900 MiB of 24576), but with only 1676 "
            "MiB spare -- tight enough that the MTP draft head plus any fragmentation is a real "
            "risk. Split leaves 12192 MiB free per card. That, not raw throughput, is the "
            "reason to prefer it."
        ),
        "comparisons": {
            "split_vs_single_ctx32768": split32,
            "split_vs_single_ctx81920": split81,
            "cpu_offload_vs_single_ctx32768": cpu32,
        },
        "observation_fixed_not_proportional_imbalance": (
            "The imbalance is 164 MiB in BOTH split arms, despite their per-card shares "
            "differing by ~1.6 GB. That is a fixed asymmetry (one card carries extra non-layer "
            "state, e.g. output/embedding tensors), not a proportional effect -- so it should "
            "not grow with context or model size. Recorded as an observation; not separately "
            "verified against the tensor map."
        ),
        "prefill_collapses_worse_than_decode_under_cpu_offload": (
            "The CPU-offload arm's prefill fell 908 -> 195 tok/s (-78.5%), worse than its "
            "decode fall (-59.8%). The 2026-07-28 characterisation of the lever's cost as "
            "'58-86% of decode throughput' therefore understates its impact on prompt-heavy "
            "work, which ARC induction is."
        ),
        "control_arm_reading": (
            "Arm 1 was written to reproduce the 2026-07-28 figures of 36.07 tok/s @ 21416 MiB. "
            "Those were measured on Q4_K_M; this ran on the newly-shipped QAT UD-Q4_K_XL. VRAM "
            "landed at 20428 MiB (-988), matching the 993 MiB model-file size difference and "
            "the head-to-head's independent 20430 MiB to within 2 MiB. Decode landed at 38.51 "
            "tok/s, +6.8% -- INSIDE the +-10% window, i.e. the throughput target effectively "
            "DID reproduce, slightly faster as a smaller model should be. An earlier draft of "
            "PLAN.md predicted both halves would fail to reproduce; that was corrected before "
            "the run (commit b44fb77cf) and the run confirms the correction was right."
        ),
        "what_this_does_NOT_establish": (
            "(1) Nothing here was measured on Kaggle's 4xL4, which is the substrate the scored "
            "submission actually runs on -- 4-way split across a different interconnect is "
            "unmeasured. (2) Only decode and prefill rate were measured; induction QUALITY "
            "under split was not, though there is no mechanism by which layer placement would "
            "change logits. (3) One run per arm, no repeats, so the ~1% throughput differences "
            "are within plausible run-to-run noise and should NOT be read as split being "
            "genuinely faster -- only as not-slower."
        ),
        "arms": rows(),
        "n_arms": len(by),
        # model_specs (not just `model`) because adversarial_verify's METHODOLOGY_MISSING check
        # looks for model_specs/target_model on any compute-bound artifact. The GGUF here is
        # genuinely loaded and generated from in every arm -- this is not a vestigial marker.
        "model_specs": [
            {
                "name": "gemma-4-31B-it-qat",
                "hf_id": "unsloth/gemma-4-31B-it-qat-GGUF",
                "file": "gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
                "size_gib": 16.10,
                "quantisation": "UD-Q4_K_XL (QAT)",
                "invoked": True,
                "resolved_from": (
                    "arc_executable_world_model.ARC_LIVE_GENERATOR_* -- the shipped pin, not a "
                    "hardcoded path, so this sweep tracks the generator if it is repinned"
                ),
            }
        ],
        "serving_config": {
            "kv_quant": "q8_0",
            "n_gpu_layers": 999,
            "n_predict": 256,
            "temperature": 0.0,
            "cache_prompt": False,
            "mtp_draft_head": False,
            "mtp_note": (
                "No draft head in any arm. Speculation would confound a decode-rate comparison, "
                "since acceptance rate varies with content. The MTP head is what the freed "
                "per-card headroom is FOR, but it is deliberately not measured here."
            ),
        },
        "hardware": "2x NVIDIA GeForce RTX 3090 (24576 MiB each), both idle at run start",
        "inference_substrate": "live_llm_inference",
        "duration_s": DURATION_S,
        "run_start": RUN_START,
        "run_end": RUN_END,
        "duration_note": (
            "Wall clock of the whole sweep. The per-arm load_s + gen_wall_s sums to only 110.0 "
            "s; that excludes health polling, teardown and eviction between arms and would "
            "understate the run by ~2x, so it is not used as duration_s."
        ),
        "random_seed": 0,
        "random_seed_note": (
            "temperature=0.0 and cache_prompt=false on a fixed prompt, so generation is "
            "deterministic and no seed is consulted. The seed field is 0 to satisfy the "
            "methodology check honestly rather than inventing a value that steers nothing."
        ),
        "verifier_is_oracle": False,
        "verifier_is_oracle_note": (
            "No verifier claim is made. This is a throughput/residency measurement, not a "
            "verifier-value experiment; the field is present because the linter requires it."
        ),
        "solve_provenance": "development_proxy",
        "preconditions_checked": [
            {"resource": "both_3090s_idle", "available": True},
            {"resource": "carnot_conductor_stopped", "available": True},
            {"resource": "llama_server_binary", "available": True},
            {"resource": "port_8971_clear", "available": True},
        ],
        "provenance": {
            "harness": "results/arc_gpu_split_sweep_20260731/harness/gpu_split_measure.py",
            "plan": "results/arc_gpu_split_sweep_20260731/PLAN.md",
            "raw_log": "results/arc_gpu_split_sweep_20260731/sweep_run.log",
            "raw_rows": "results/arc_gpu_split_sweep_20260731/gpu_split_results.json",
            "builder": "scripts/build_gpu_split_sweep_artifact.py",
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
            ).stdout.strip(),
        },
    }
    art["reproducibility_checksum"] = hashlib.sha256(
        json.dumps(art, sort_keys=True, default=str).encode()
    ).hexdigest()
    OUT.write_text(json.dumps(art, indent=2) + "\n")

    print(
        f"  gates: free={g_free} halved={g_halved} even={g_even} -> passed={art['acceptance_gate_passed']}"
    )
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
