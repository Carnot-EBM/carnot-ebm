"""Assemble the scored artifact for the ARC multimodal fidelity gate.

Reads the raw outputs produced by this run (tower_proof.json, fidelity_raw.json,
escalate_raw.json), DERIVES the verdict from the numbers rather than asserting one,
and writes results/arc_multimodal_fidelity_20260801/.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import numpy as np

SD = Path(
    "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/mmgate"
)
REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")
OUTDIR = REPO / "results" / "arc_multimodal_fidelity_20260801"

_HUB = "/home/ianblenke/.cache/huggingface/hub/models--unsloth--gemma-4-31B-it-GGUF/snapshots"
BASE_GGUF = f"{_HUB}/f130ba51393346288f5862e30e9586b9b021513f/gemma-4-31B-it-Q4_K_M.gguf"
MMPROJ = f"{_HUB}/c1ac76e99d5513b141e8adde7288b85c3f9c32ec/mmproj-F16.gguf"
MMPROJ_SHA = "6edcca228213c28d3567a35d22f849eea52d8360875093851959adf5d2f270eb"
SEED = 20260801
BATCH = 8  # probes per request -- the CLUSTER unit, see cluster_stats()

# BARS. Provenance matters and is recorded in the artifact, because one of these was
# fixed in advance and one was not.
#   REPLACE_BAR is PRE-REGISTERED: fidelity.py design decision 3 fixes it as the
#     escalation rule ("only if the charitable case clears 0.90...") and decision 1
#     names ">=90%" as the usable-for-exact-match-induction threshold, both written
#     before the run.
#   SUPPLEMENT_BAR is POST-HOC. It appears only in this file, which was written after
#     the numbers existed. It is retained for continuity but it is NO LONGER what
#     decides `can_supplement_text` -- that now rests on the pre-registered chance
#     rate instead. See supplement_basis in build().
REPLACE_BAR = 0.90
SUPPLEMENT_BAR = 0.25
SUPPLEMENT_BAR_IS_POST_HOC = True
CHANCE = 0.0625


# ---------------------------------------------------------------------------------
# Reconstruct the exact grid and probe list the run used, from the recorded seed.
# This is not a convenience: the raw file stores predictions POSITIONALLY, so every
# derived statistic below needs the ground truth in the same order. Reconstructing it
# and checking it against the recorded sha256s also proves the raw file and this
# builder are talking about the same experiment.
def _rebuild_probes(meta: dict):
    rng = np.random.default_rng(SEED)
    order = np.repeat(np.arange(16), 4)
    rng.shuffle(order)
    grid = np.repeat(np.repeat(order.reshape(8, 8), 8, axis=0), 8, axis=1).astype(np.uint8)
    probes = []
    for v in range(16):
        rs, cs = np.where(grid == v)
        for i in rng.choice(len(rs), size=8, replace=False):
            probes.append((int(rs[i]), int(cs[i]), int(v)))
    rng.shuffle(probes)
    if hashlib.sha256(grid.tobytes()).hexdigest() != meta["grid_sha256"]:
        raise SystemExit("REFUSING TO BUILD: reconstructed grid does not match meta.grid_sha256")
    if hashlib.sha256(json.dumps(probes).encode()).hexdigest() != meta["probes_sha256"]:
        raise SystemExit("REFUSING TO BUILD: reconstructed probes do not match meta.probes_sha256")
    return grid, probes


def mcnemar_exact(pred_a, pred_b, truth) -> dict:
    """Exact two-sided McNemar (a sign test on the discordant pairs).

    COMPUTED, NOT ASSERTED. The previous version of this file carried all five
    p-values as hardcoded literals, and two of them were written `0.0` -- a value an
    exact test can never return. The true values are 4.59e-07 and 4.00e-08. Nothing on
    disk computed them, so nothing could catch them.

    `min_reachable_p` is reported next to each: with b+c discordant pairs the smallest
    p the test can produce is 2/2^(b+c). Quoting it stops a reader treating a small p
    as more resolution than the design can deliver.
    """
    b = sum(1 for t, x, y in zip(truth, pred_a, pred_b, strict=True) if x == t and y != t)
    c = sum(1 for t, x, y in zip(truth, pred_a, pred_b, strict=True) if x != t and y == t)
    n = b + c
    if n == 0:
        return {"a_only_correct": 0, "b_only_correct": 0, "p_value": 1.0, "min_reachable_p": 1.0}
    # two-sided exact binomial at p=0.5
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    p = min(1.0, 2 * tail)
    return {
        "a_only_correct": b,
        "b_only_correct": c,
        "n_discordant": n,
        "p_value": float(f"{p:.6g}"),
        "min_reachable_p": float(f"{2 / (2**n):.3g}"),
        "significant_at_0.05": bool(p < 0.05),
    }


def cluster_stats(pred, truth) -> dict:
    """Batch-level (cluster-robust) summary.

    WHY THE PROBE-LEVEL INTERVALS IN THE FIRST DRAFT WERE TOO NARROW. The 128 probes
    were NOT asked independently: they went out in 16 requests of 8, and each request
    is answered by one chain-of-thought that either locates the grid correctly for the
    whole batch or does not. The measured per-batch counts are wildly bimodal (px16:
    8,7,2,0,8,8,0,0,1,8,1,1,0,8,0,7), so the effective sample size is nearer 16
    clusters than 128 probes. Treating probes as independent understates every interval
    and overstates every p-value's confidence.

    Reports the design effect (observed variance of the per-batch count over the
    binomial expectation) and a cluster-robust t-interval over the 16 batch means.
    """
    n_b = len(pred) // BATCH
    counts = [
        sum(1 for i in range(b * BATCH, (b + 1) * BATCH) if pred[i] == truth[i]) for b in range(n_b)
    ]
    phat = sum(counts) / len(pred)
    var_obs = statistics.variance(counts) if n_b > 1 else 0.0
    var_bin = BATCH * phat * (1 - phat)
    deff = (var_obs / var_bin) if var_bin > 0 else None
    means = [c / BATCH for c in counts]
    se = (statistics.stdev(means) / math.sqrt(n_b)) if n_b > 1 else 0.0
    t95 = 2.131  # t(0.975, df=15)
    return {
        "per_batch_correct_out_of_8": counts,
        "n_clusters": n_b,
        "design_effect": round(deff, 2) if deff else None,
        "effective_n": round(len(pred) / deff, 1) if deff else None,
        "exact_match_ci95_cluster_robust": [
            round(max(0.0, phat - t95 * se), 4),
            round(min(1.0, phat + t95 * se), 4),
        ],
    }


def shift_contamination(pred, truth) -> dict:
    """Quantify how far the retired positional-fallback parser shifted this run.

    A wrong prediction that equals the truth of an EARLIER probe in the same batch is
    the signature of a misaligned parse; the same comparison against LATER probes is
    the null, since nothing can shift answers backwards. The excess of forward over
    backward hits therefore bounds how many probes were scored against the wrong
    question -- all of which were counted WRONG, which is why every exact_match here is
    a lower bound rather than an overstatement.
    """
    fwd = bwd = 0
    for i, p in enumerate(pred):
        if p is None:
            continue
        lo, hi = (i // BATCH) * BATCH, (i // BATCH + 1) * BATCH
        for k in (1, 2, 3):
            if i - k >= lo and truth[i - k] == p and p != truth[i]:
                fwd += 1
            if i + k < hi and truth[i + k] == p and p != truth[i]:
                bwd += 1
    n_ok = sum(1 for t, p in zip(truth, pred, strict=True) if p == t)
    excess = max(0, fwd - bwd)
    return {
        "forward_shift_hits_k1_3": fwd,
        "backward_shift_hits_k1_3_null": bwd,
        "excess_over_null": excess,
        "exact_match_lower_bound": round(n_ok / len(pred), 4),
        "exact_match_optimistic_ceiling_if_every_shifted_probe_was_really_correct": round(
            min(1.0, (n_ok + excess) / len(pred)), 4
        ),
    }


def wilson(k: int, n: int) -> list[float]:
    """95% Wilson score interval -- correct near 0 and 1, where the normal approx is not."""
    if n == 0:
        return [0.0, 0.0]
    z = 1.959963985
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def confusion_note(conf: dict) -> dict:
    """Surface the index pairs this gate was specifically commissioned to look for.

    4/14 and 5/15 are the pairs `to_ascii()` already merges today (it keeps only the
    last decimal digit), so a vision tower reproducing the SAME collapse would be a
    second, independent instance of the same bug class.
    """

    def g(a, b):
        return conf.get(f"{a}->{b}", 0) + conf.get(f"{b}->{a}", 0)

    return {
        "pair_4_14_confusions": g(4, 14),
        "pair_5_15_confusions": g(5, 15),
        "pair_1_11_confusions": g(1, 11),
        "pair_0_10_confusions": g(0, 10),
        "last_digit_collapse_pairs_total": g(4, 14) + g(5, 15) + g(1, 11) + g(0, 10),
        "all_confusions_sorted": conf,
    }


def _assert_real_measurement(src: dict, name: str) -> None:
    """Refuse to build an artifact out of anything that is not a real measurement.

    WHY THIS GUARD EXISTS, stated plainly because it is embarrassing and instructive:
    while this run was still in flight, placeholder raw files were written to these
    exact paths in order to dry-run the artifact pipeline. They were deleted straight
    away, but for a few minutes the canonical input paths held FABRICATED numbers, and
    nothing in this script could have told the difference. Publishing a scored artifact
    built from those would have been precisely the fabrication class the project's
    adversarial-verify layer exists to catch -- and it would have been self-inflicted,
    not caught, because the numbers were plausible by construction.

    So the guard is structural rather than a promise to be careful: a real run records
    a per-probe `predictions` list and a non-trivial wall-clock, and the placeholder
    could not have either. Fail CLOSED -- raise, never warn-and-continue.
    """
    for r in src.get("results", []):
        preds = r.get("predictions")
        if not isinstance(preds, list) or len(preds) != r.get("cells_probed"):
            raise SystemExit(
                f"REFUSING TO BUILD: {name} scheme {r.get('scheme')!r} has no per-probe "
                f"predictions list matching cells_probed. This does not look like a real "
                f"measurement. Re-run the harness; do not hand-edit the raw file."
            )
        if not r.get("duration_s", 0) > 0:
            raise SystemExit(
                f"REFUSING TO BUILD: {name} scheme {r.get('scheme')!r} has no duration."
            )


def build():
    proof = json.loads((SD / "tower_proof.json").read_text())
    fid = json.loads((SD / "fidelity_raw.json").read_text())
    _assert_real_measurement(fid, "fidelity_raw.json")
    esc_p = SD / "escalate_raw.json"
    esc = json.loads(esc_p.read_text()) if esc_p.exists() else None
    if esc is not None:
        _assert_real_measurement(esc, "escalate_raw.json")

    _, probes = _rebuild_probes(fid["meta"])
    truth = [t for _, _, t in probes]
    preds = {r["scheme"]: r["predictions"] for r in fid["results"]}

    renderings = []
    # TAG CORRECTED ka59 -> lp85. The escalation grid is lp85's reset frame; escalate.py
    # sets GAME = "lp85" and documents rejecting ka59 (whose frame has a single cell each
    # of colours 0 and 5). ka59 was read during candidate selection only.
    for src, tag in [(fid, "synthetic_blocks"), (esc, "real_lp85_frame")]:
        if src is None:
            continue
        ctrl = next((r for r in src["results"] if "text" in r["scheme"]), None)
        for r in src["results"]:
            is_ctrl = "text" in r["scheme"]
            row = {
                "scheme": f"{tag}:{r['scheme']}",
                "grid_kind": tag,
                "is_negative_control": is_ctrl,
                "px_per_cell": r.get("px_per_cell"),
                "image_pixels": r.get("image_pixels"),
                "image_bytes": r.get("image_bytes"),
                "cells_probed": r["cells_probed"],
                "n_correct": r["n_correct"],
                "exact_match": r["exact_match"],
                "exact_match_is_a_lower_bound": True,
                "exact_match_lower_bound_reason": (
                    "the harness that produced this number used a positional-fallback parser "
                    "that ingested row and column indices as colour predictions; a misparsed "
                    "probe is scored WRONG, so the true value is >= this one. See "
                    "shift_contamination_by_scheme and post_review_corrections."
                ),
                "exact_match_ci95_wilson": wilson(r["n_correct"], r["cells_probed"]),
                "exact_match_ci95_wilson_note": (
                    "PROBE-LEVEL and therefore TOO NARROW -- it assumes 128 independent probes, "
                    "but they were asked in 16 batched requests. Use "
                    "exact_match_ci95_cluster_robust."
                ),
                # Cluster-robust stats need the ground truth in probe order, which is
                # reconstructed only for the synthetic arm (the real-frame arm has its
                # own grid and never produced a measurement).
                **(cluster_stats(preds[r["scheme"]], truth) if tag == "synthetic_blocks" else {}),
                "exact_match_over_parsed": r["exact_match_over_parsed"],
                "n_unparseable": r["n_unparseable"],
                "n_batches_truncated_after_retry": r["n_batches_truncated_after_retry"],
                "text_control_exact_match": ctrl["exact_match"] if ctrl else None,
                "text_control_scheme": ctrl["scheme"] if ctrl else None,
                "vision_minus_text_control": (
                    None
                    if (is_ctrl or not ctrl)
                    else round(r["exact_match"] - ctrl["exact_match"], 4)
                ),
                "confusions": confusion_note(r["confusions"]),
                "duration_s": r["duration_s"],
            }
            renderings.append(row)

    imgs = [r for r in renderings if not r["is_negative_control"]]
    ctrls = [r for r in renderings if r["is_negative_control"]]
    real_imgs = [r for r in imgs if r["grid_kind"] == "real_lp85_frame"]
    real_ctrl = [r for r in ctrls if r["grid_kind"] == "real_lp85_frame"]

    best_txt = max((r["exact_match"] for r in ctrls), default=0.0)
    real_best_img = max((r["exact_match"] for r in real_imgs), default=None)
    real_best_txt = max((r["exact_match"] for r in real_ctrl), default=None)

    # ---- THE GATE, derived from the numbers ----
    # can_replace_text: images stand IN PLACE OF the exact RLE text. The bar is
    #   absolute (>=0.90) because the induced engine is graded by EXACT match and
    #   writes Python comparing specific cells to specific integers.
    # can_supplement_text: images are added ALONGSIDE the text. Needs only that the
    #   vision path carry real signal, well clear of the 1/16 = 6.25% chance rate.
    #
    # WHICH GRID DECIDES `can_replace`, AND WHY THE MISSING REAL-FRAME ARM DOES NOT
    # LEAVE THIS UNDECIDED. The synthetic blocks grid is the CHARITABLE case by
    # construction: 8x8-cell solid blocks in 16 maximally-separated nameable colours,
    # both far easier than a real ARC frame. So its score is an UPPER BOUND on the real
    # frame. When that upper bound is itself below the bar, `can_replace` is settled a
    # fortiori -- a harder grid cannot rescue it -- and that is exactly the
    # pre-registered escalation rule in fidelity.py: escalate only if the charitable
    # case PASSES. A failure needs no escalation. If the charitable case HAD passed,
    # real-frame evidence would be REQUIRED, and its absence would force can_replace to
    # False rather than allow a pass on charitable evidence alone.
    charitable_best = max(
        (r["exact_match"] for r in imgs if r["grid_kind"] == "synthetic_blocks"), default=0.0
    )
    if charitable_best < REPLACE_BAR:
        can_replace = False  # settled by the upper bound; no real-frame arm needed
        replace_basis = (
            f"charitable synthetic upper bound {charitable_best:.3f} < {REPLACE_BAR} bar, so a "
            "real ARC frame (strictly harder) cannot clear it either"
        )
    else:
        can_replace = bool(real_best_img is not None and real_best_img >= REPLACE_BAR)
        replace_basis = (
            "charitable case cleared the bar, so the decision rests on the real-frame arm"
            + ("" if real_best_img is not None else " -- which was NOT run, so replace is refused")
        )
    # can_supplement, RE-ANCHORED after review. It previously tested `best_img >= 0.25`,
    # a bar that exists only in this file and was written after the numbers were in --
    # so both of the artifact's positive conclusions rested on a threshold chosen with
    # the answer visible. The replacement uses only pre-registered quantities: the
    # 6.25% chance rate (design decision 1), tested against the CLUSTER-ROBUST lower
    # confidence bound rather than the point estimate, so it also survives the
    # batch-clustering correction. This is a strictly harder test than the retired one
    # and it does not depend on any number chosen after the fact.
    best_img_row = max(imgs, key=lambda r: r["exact_match"]) if imgs else None
    img_lcb = (best_img_row or {}).get("exact_match_ci95_cluster_robust", [0.0, 0.0])[0]
    txt_row = ctrls[0] if ctrls else None
    txt_lcb = (txt_row or {}).get("exact_match_ci95_cluster_robust", [0.0, 0.0])[0]
    can_supplement = bool(img_lcb > CHANCE)
    supplement_basis = (
        f"the best image scheme's cluster-robust 95% lower bound is {img_lcb:.3f}, above the "
        f"pre-registered {CHANCE} chance rate, so the vision path carries real information. "
        "This does NOT license replacing the text, and it is not a claim that images help "
        "induction -- only that the channel is not noise. Anchored on the pre-registered "
        f"chance rate; the {SUPPLEMENT_BAR} supplement_bar in this file is POST-HOC and no "
        "longer decides anything. The measured value is also a LOWER bound (parser defect), "
        "so this conclusion can only strengthen under correction."
    )

    # THE NEGATIVE CONTROL'S VETO. If the text arm also collapses, the question format
    # is what failed and NO conclusion about vision is licensed either way. Same
    # re-anchoring: pre-registered chance rate, cluster-robust bound.
    control_informative = bool(txt_lcb > CHANCE)

    total_probe_s = sum(r["duration_s"] for r in renderings)

    art = {
        "experiment": "arc_multimodal_fidelity_gate_20260801",
        "experiment_id": "arc_multimodal_fidelity_20260801",
        "title": "Can gemma-4-31B-it's vision tower read ARC colour indices back exactly?",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema": "carnot.arc_multimodal_fidelity_gate.v1",
        "milestone": "outer-loop 2026-08-01",
        "inference_substrate": "live_llm_inference",
        "inference_substrate_note": (
            "A real llama.cpp llama-server process loaded the 18.3 GB gemma-4-31B-it Q4_K_M GGUF "
            "plus the 1.14 GiB mmproj-F16 vision projector onto RTX 3090 index 1 (about 22 GB "
            "resident, confirmed by nvidia-smi against the launched PID) and served full "
            "autoregressive generation for every probe. This is neither a cached-candidate "
            "rescoring nor an aggregation of upstream artifacts."
        ),
        "random_seed": SEED,
        "duration_s": round(total_probe_s, 2),
        "model_specs": {
            "generator": "unsloth/gemma-4-31B-it-GGUF",
            "weights_file": BASE_GGUF,
            "weights_bytes": os.path.getsize(BASE_GGUF),
            "vision_projector_file": MMPROJ,
            "vision_projector_bytes": os.path.getsize(MMPROJ),
            "vision_projector_sha256": MMPROJ_SHA,
            "server_build": "llama.cpp b9606-9b4dae81f (CUDA build)",
            "gpu": (
                "RTX 3090 index 1 via CUDA_VISIBLE_DEVICES=1; "
                "GPU 0 left untouched (conductor-owned)"
            ),
            "n_ctx": 16384,
            "parallel_slots": 1,
            "kv_cache": "q8_0",
            "temperature": 0.0,
            "port": 18831,
        },
        "preconditions_checked": [
            {
                "resource": "gpu1_free_before_launch",
                "available": True,
                "evidence": "nvidia-smi before launch: GPU1 2 MiB used, no compute processes",
            },
            {"resource": "base_gguf_cached", "available": True, "evidence": BASE_GGUF},
            {
                "resource": "mmproj_f16_downloaded",
                "available": True,
                "evidence": f"{MMPROJ} sha256={MMPROJ_SHA} bytes={os.path.getsize(MMPROJ)}",
            },
            {
                "resource": "llama_server_supports_mmproj",
                "available": True,
                "evidence": "llama-server --help lists -mm/--mmproj (build b9606-9b4dae81f)",
            },
            {
                "resource": "vision_tower_loaded_and_actually_used",
                "available": True,
                "evidence": "four-layer server_proof below",
            },
        ],
        # ---------------- THE MANDATORY PROOF THE TOWER WAS USED ----------------
        "mmproj_loaded": True,
        "server_proof": {
            "layer1_server_log": (
                "srv load_model: loaded multimodal model, '.../mmproj-F16.gguf' ; "
                "srv load_model: [mtmd] estimated worst-case memory usage of mmproj is 1290.66 MiB"
            ),
            "layer2_props_endpoint": {
                "modalities": {"vision": True, "video": True, "audio": False},
                "total_slots": 1,
            },
            "layer3_gpu_residency": (
                "nvidia-smi --query-compute-apps showed the launched server PID holding ~22 GB on "
                "GPU 1 while GPU 0 stayed at 2 MiB"
            ),
            "layer4_differential_test": {
                "why_this_is_the_decisive_layer": (
                    "llama.cpp will happily serve a request whose image it ignored, so a load-time "
                    "log line and a /props capability flag cannot rule out a silent text-only "
                    "fallback. Under such a fallback every fidelity number below would really be "
                    "measuring 'guess a colour index with no information' while being reported as "
                    "a vision result. So four BYTE-IDENTICAL text prompts were sent, each paired "
                    "with a DIFFERENT solid-colour image. No text-only model can separate them."
                ),
                "cases": proof["cases"],
                "no_image_answer": proof["no_image_answer"],
                "n_correct": proof["n_correct"],
                "n_distinct_answers": proof["n_distinct_answers"],
                "conclusion": (
                    f"{proof['n_correct']}/4 correct with {proof['n_distinct_answers']} distinct "
                    "answers to identical prompts, so the image is the only channel carrying the "
                    "difference. The same prompt with NO image answered "
                    f"{proof['no_image_answer']!r} "
                    "-- a guess, which confirms the text alone is uninformative."
                ),
                "vision_path_proven_live": proof["vision_path_proven_live"],
            },
        },
        # ---------------- RESULTS ----------------
        "renderings": renderings,
        # FOUR DISTINCT SCOPED MEASUREMENTS -- deliberately NOT accompanied by any
        # "best over everything" aggregate. An earlier draft also emitted
        # headline_best_image_exact_match / headline_best_text_control_exact_match as
        # maxima over all scopes, and a structural pre-check caught the consequence:
        # whenever the real-frame arm happened to BE the maximum, the aggregate and the
        # scoped field held byte-identical floats, which adversarial_verify correctly
        # flagged as a CRITICAL TAUTOLOGY. It was right to. Two top-level "metrics" that
        # are the same measurement under two names is a reporting bug, so the duplicate
        # aggregates were removed rather than exempted.
        "synthetic_best_image_exact_match": max(
            (r["exact_match"] for r in imgs if r["grid_kind"] == "synthetic_blocks"), default=None
        ),
        "synthetic_text_control_exact_match": next(
            (r["exact_match"] for r in ctrls if r["grid_kind"] == "synthetic_blocks"), None
        ),
        "real_frame_best_image_exact_match": real_best_img,
        "real_frame_text_control_exact_match": real_best_txt,
        "chance_rate_16_way": CHANCE,
        "replace_bar": REPLACE_BAR,
        "replace_bar_provenance": "PRE-REGISTERED in fidelity.py design decisions 1 and 3",
        "supplement_bar": SUPPLEMENT_BAR,
        "supplement_bar_is_post_hoc": SUPPLEMENT_BAR_IS_POST_HOC,
        "supplement_bar_provenance": (
            "POST-HOC. Defined only in build_artifact.py, which was written after the raw "
            "numbers existed. RETIRED as a decision rule: can_supplement_text is now decided "
            "by the pre-registered chance rate instead. Retained only so the earlier basis "
            "stays on the record."
        ),
        "can_replace_text": can_replace,
        "can_replace_text_basis": replace_basis,
        "can_supplement_text": can_supplement,
        "can_supplement_text_basis": supplement_basis,
        "negative_control_informative": control_informative,
        "negative_control_basis": (
            f"the shipped RLE text arm scored {best_txt:.3f} on the identical paired probes "
            f"(cluster-robust 95% lower bound {txt_lcb:.3f}), above the {CHANCE} chance rate, so "
            "'read the cell at (row, col)' is a demonstrably answerable question format. A low "
            "vision score therefore cannot be dismissed as a question-format artifact -- the "
            "confound the control exists to rule out is ruled out."
        ),
        # PAIRED SIGNIFICANCE. The probes are the same cells in the same order across
        # schemes, so the correct test is McNemar on the discordant pairs, NOT a
        # comparison of the two Wilson intervals -- overlapping CIs on PAIRED data
        # routinely hide a real difference, and non-overlapping ones can overstate it.
        #
        # ALL FIVE ARE NOW COMPUTED FROM `predictions`. They were previously hardcoded
        # literals, and two of them read `p_value: 0.0` -- impossible for an exact test.
        # The true values are 4.59e-07 and 4.00e-08.
        #
        # TWO CAVEATS TRAVEL WITH EVERY p HERE, and the px16-vs-text one cannot be read
        # without them:
        #   (1) PROBE-LEVEL, so anticonservative. The probes are clustered in batches of
        #       8 (measured design effect 3.3-7.0), so the effective sample is nearer 16
        #       clusters than 128 probes and these p-values are optimistic.
        #   (2) THE TWO ARMS ARE DIFFERENTIALLY BIASED by the retired parser. Forward
        #       shift-hits, which index how much each arm lost to misalignment, are
        #       HIGHER in the text control (66 against a null of 14) than in px16 (41
        #       against 13). Correcting the parser would therefore be expected to raise
        #       the text arm MORE than px16 -- so this contrast is not merely
        #       underpowered, its point estimate is biased in an unknown net direction.
        # Consequently the honest reading of px16-vs-text is NOT ESTABLISHED EITHER WAY,
        # not "statistically indistinguishable". Claiming equivalence from p=0.103 would
        # be accepting a null from a non-significant test on two biased measurements.
        "paired_significance_tests_mcnemar_exact": {
            "image_px16_vs_text_rle": {
                **mcnemar_exact(preds["image_px16"], preds["text_rle"], truth),
                "a_is": "image_px16",
                "b_is": "text_rle",
                "reading": (
                    "NOT ESTABLISHED EITHER WAY. Do NOT claim images beat the RLE text "
                    "encoding, and do NOT claim they are equivalent to it: this p is "
                    "probe-level (anticonservative under batch clustering) and the two arms "
                    "carry different amounts of parser-shift bias."
                ),
            },
            "image_px16_vs_image_px8": {
                **mcnemar_exact(preds["image_px16"], preds["image_px8"], truth),
                "a_is": "image_px16",
                "b_is": "image_px8",
            },
            "image_px16_vs_image_px1": {
                **mcnemar_exact(preds["image_px16"], preds["image_px1"], truth),
                "a_is": "image_px16",
                "b_is": "image_px1",
            },
            "text_rle_vs_image_px1": {
                **mcnemar_exact(preds["text_rle"], preds["image_px1"], truth),
                "a_is": "text_rle",
                "b_is": "image_px1",
            },
            "text_rle_vs_image_px8": {
                **mcnemar_exact(preds["text_rle"], preds["image_px8"], truth),
                "a_is": "text_rle",
                "b_is": "image_px8",
            },
            "_caveats": [
                "probe-level: anticonservative under the measured batch clustering",
                "arms are differentially biased by the retired positional-fallback parser",
            ],
        },
        # THE PARSER-SHIFT DIAGNOSTIC. This is what demotes every exact_match to a lower
        # bound and what makes the px16-vs-text contrast unsafe.
        "shift_contamination_by_scheme": {
            sch: shift_contamination(preds[sch], truth) for sch in preds
        },
        "images_beat_text_claim_supported": False,
        "images_equal_text_claim_supported": False,
        "last_digit_collapse_hypothesis_supported": False,
        "last_digit_collapse_hypothesis_note": (
            "The gate was commissioned partly to see whether the vision tower would reproduce the "
            "SAME collapse that arc_executable_world_model.to_ascii() already has -- it renders "
            "each cell as str(int(v))[-1], merging 4/14, 5/15, 1/11 and 0/10. It does NOT. Summed "
            "over all four schemes and 512 probes, those four pairs account for 12 confusions "
            "total, no more than unrelated pairs such as 4->6 or 12->15, and the single largest "
            "confusion in every scheme is a non-collapse pair. So the tower's errors are ordinary "
            "perceptual confusions, not a digit-truncation artifact. This is a clean negative on "
            "the striking-result hypothesis and should be reported as such rather than quietly "
            "dropped."
        ),
        # ---------------- DISCIPLINE FIELDS ----------------
        # ---- FALSE-NEGATIVE-RISK CONTROLS ----
        # A null result here ("the tower cannot read ARC colour indices") is a real
        # possible outcome of this gate, and a null claim is only worth anything if the
        # apparatus was demonstrably capable of producing a positive. Two independent
        # positive controls were run for exactly that reason, and both are recorded
        # rather than asserted:
        #   (a) the differential tower proof -- four solid-colour images read back
        #       correctly and distinctly through the same server, same projector, same
        #       prompt path. So a null on fine grids cannot be "the vision path was
        #       dead" or "the harness never sent the image".
        #   (b) the RLE text control -- the identical questions over the shipped text
        #       encoding, which bounds how much of any failure belongs to the question
        #       format rather than to the modality.
        # THIRD CONTROL: the RENDERER is not the lossy component. Each emitted PNG was
        # read back, nearest-neighbour-subsampled at its own px_per_cell, mapped
        # through the palette, and hashed: all three reproduce the source grid's
        # sha256 EXACTLY (d05eec86...). So the images genuinely contain every colour
        # index, and any read-back failure is attributable to the vision tower rather
        # than to smoothing, resampling or palette drift on our side. Without this a
        # low score would have an innocent alternative explanation.
        "render_losslessness_verified": True,
        "render_losslessness_evidence": (
            "rendered_px1.png / rendered_px8.png / rendered_px16.png each decode back to the "
            "source grid sha256 recorded in raw_per_scheme_measurements.json meta.grid_sha256"
        ),
        "positive_control_passed": bool(proof["vision_path_proven_live"]),
        "positive_control_description": (
            "Differential vision proof: 4 byte-identical prompts with 4 different solid-colour "
            "images returned 4 distinct correct colour names through the same server and prompt "
            "path used for every measurement below. The apparatus can demonstrably produce a "
            "positive, so a null on fine grids is a capability limit, not a dead code path."
        ),
        "false_negative_risk_checked": True,
        "verifier_is_oracle": False,
        "verifier_is_oracle_note": (
            "Not a verifier-value, moat or efficiency claim. This measures a model's vision "
            "encoder against a known ground-truth grid; no verifier is credited and no gate is "
            "flipped by this artifact."
        ),
        "solve_provenance": "development_proxy",
        "solve_provenance_note": (
            "No ARC level was solved, attempted, or claimed, and no game was played scored or "
            "online. The only use of the arcade was OFFLINE, reading reset frames out of the "
            "local environment_files/ as ground-truth grid data: ka59 and lp85 were both read "
            "while choosing an escalation grid, lp85 was selected, and in the event no probe "
            "was run against either -- every number in this artifact comes from the synthetic "
            "grid."
        ),
        "sample_size_justification": (
            "128 paired probe cells per scheme on the synthetic grid (8 per colour, all 16 "
            "indices represented). NO REAL-FRAME ARM WAS MEASURED -- an earlier version of this "
            "sentence also claimed '128 on the real ka59 frame (stratified over the 7 colours "
            "that frame actually contains)', which was doubly wrong: no real-frame probe ever "
            "completed, and the staged escalation grid was lp85 with 11 colours, not ka59 with "
            "7. Nothing downstream consumed that sentence -- real_frame_best_image_exact_match "
            "is null and every rendering row is synthetic -- but it was a number quoted in an "
            "artifact that no measurement backed, which is the exact failure this project's "
            "fabrication gate exists to catch, so it is corrected here and recorded in "
            "post_review_corrections rather than quietly deleted. For the synthetic arm: "
            "worst-case 95% Wilson half-width is about 9 percentage points at p=0.5 -- but see "
            "exact_match_ci95_cluster_robust, which is the interval to use, since the probes "
            "are clustered in 16 batched requests (design effect 3.3-7.0, effective n 18-38). "
            "Even cluster-robust, the sample separates the 1/16 = 6.25% chance rate from the "
            "0.90 usable rate, which is the only distinction this gate must resolve; it is NOT "
            "enough to resolve a few-point difference between two schemes, and no such claim is "
            "made. The precision floor and the rendering parameters were fixed before any "
            "result was inspected; the 0.25 supplement bar was NOT, and no longer decides "
            "anything."
        ),
        "methodology_note": (
            "Probes are PAIRED: the identical (row, col) list in the same order is used for every "
            "scheme including the text control, so schemes differ only in how the grid was "
            "presented. Temperature 0, fixed seed. gemma-4 emits a mandatory chain-of-thought, so "
            "a reply truncated at the token cap (finish_reason=length) returns an EMPTY answer "
            "that is indistinguishable from a wrong one; each such batch is retried once at an "
            "11k-token budget and any residual truncation is reported as n_unparseable. Both a "
            "strict exact_match over all probes and a diagnostic exact_match_over_parsed are "
            "given so 'misread the colour' stays distinguishable from 'never finished thinking'. "
            "TWO earlier attempts at this run were DISCARDED rather than reported, and neither "
            "contributed a single number to this artifact. (1) Two copies of the harness were "
            "accidentally launched at once and their concurrent requests shared one server KV "
            "pool; the run was killed and the server relaunched with --parallel 1. (2) The "
            "restarted run was found to be paying a 7000-token dead attempt plus an "
            "11000-token retry on most text batches -- about 530s of decode each against ~120s "
            "when the first attempt succeeds -- so the first-attempt budget was raised to 9000 "
            "and the run restarted again. That change is uniform across all schemes and can "
            "only affect whether an answer is REACHED, never whether it is correct. Every "
            "number here comes from the single final sequential run."
        ),
        "limitations": [
            "The index-to-RGB palette is ours, not ARC's. The environment_files define colour "
            "INDICES only and never RGB, so no official palette was recoverable. We chose 16 "
            "maximally-separated, individually-nameable colours and put the legend in the "
            "prompt, which makes every number here an UPPER BOUND: a real ARC palette with "
            "closer shades can only do worse.",
            "The synthetic grid is 8x8-cell solid blocks -- far coarser than a real ARC frame -- "
            "so every score here is a charitable UPPER bound, not an estimate of real-frame "
            "performance.",
            "A real-ARC-frame escalation WAS built and staged (lp85's reset frame, chosen over "
            "ka59 because ka59's frame has only one cell each of colours 0 and 5; lp85 carries 11 "
            "colours with >=32 cells each and still contains both 4/14 and 5/15). It was LAUNCHED "
            "AND THEN TERMINATED BEFORE ANY BATCH COMPLETED -- run_all.sh invokes escalate.py "
            "unconditionally and run_all.log ends with its lp85 metadata block and the line "
            "'--- TEXT CONTROL (shipped RLE encoding), REAL frame ---' with no batch lines after "
            "it and no ALL_DONE. An earlier version of this limitation said 'DELIBERATELY NOT "
            "RUN', which the run log contradicts; corrected here. NO real-frame probe produced an "
            "answer, so no real-frame number exists. The reasoning for stopping it is unchanged "
            "and still sound: at the measured ~200s/batch it needed roughly 2.5 more hours and "
            "could not change the gate, because the charitable arm already scored 0.461, far "
            "under the 0.90 bar, and a harder grid can only score lower. It WOULD be required "
            "before any future claim that images can carry ARC grids. Concretely it remains "
            "unmeasured how px16 and the text control compare on a real frame, whose fine "
            "structure plausibly hurts vision more than it hurts the run-length text -- and note "
            "that even HERE, on the grid that most favours vision, the two are NOT ESTABLISHED "
            "either way (McNemar p=0.103, probe-level and differentially parser-biased).",
            "One generator, one quantisation (Q4_K_M), one projector (F16). A different quant or "
            "the F32 projector could differ.",
            "Read-back fidelity is necessary for image-fed induction but not sufficient: this "
            "gate does not measure whether images improve induced-engine heldout accuracy.",
            "Read-back is a single-cell query. It does not establish that the tower preserves "
            "whole-grid structure well enough to predict a next grid exactly.",
            "The px=8 strict score is dominated by NON-ANSWERS, not misreads: 10 of its 16 "
            "batches exhausted even the retry budget mid-chain-of-thought, leaving only 48 of 128 "
            "probes with any answer. Its strict 0.156 and its parsed-only 0.417 therefore differ "
            "by a lot, and the honest reading is that at 512x512 the model frequently fails to "
            "TERMINATE rather than that it misreads the colour. Both numbers are reported; "
            "neither alone characterises that scheme.",
            "Scores are per-batch bimodal in every arm (px16: 8,7,2,0,8,8,0,0,1,8,1,1,0,8,0,7). "
            "AN EARLIER VERSION OF THIS LIMITATION ATTRIBUTED THAT TO THE MODEL'S SPATIAL "
            "INDEXING. That attribution was wrong, and wrong in the worst direction -- it "
            "published a harness bug as a model-capability finding. The bimodality is "
            "substantially the retired positional-fallback parser losing alignment: forward "
            "shift-hits (a wrong prediction equal to an earlier probe's truth in the same batch, "
            "the misparse signature) run 3.44-4.64 per batch in the batches scoring <=2/8 and "
            "0.00-0.14 per batch in those scoring >=6/8. In the text arm 11 of 16 batches "
            "returned a full 8 parsed answers of which <=2 were correct -- the positional-scoop "
            "signature exactly. Whether the model ALSO has a spatial-indexing weakness is "
            "genuinely unresolved and needs the corrected parser to answer; this experiment "
            "cannot separate parser misalignment, spatial indexing and colour discrimination.",
            "The 128 probes are NOT independent: they were asked in 16 requests of 8, and each "
            "request is answered by one chain-of-thought. Measured design effects are 3.3-7.0, "
            "so effective n is 18-38, not 128. Cluster-robust intervals are reported per "
            "rendering; the probe-level Wilson intervals are retained but are too narrow. The "
            "McNemar p-values are probe-level and correspondingly anticonservative.",
            "EVERY exact_match in this artifact is a LOWER BOUND, not a point estimate. The "
            "harness parsed replies with a positional fallback that scooped row and column "
            "numbers as colour predictions whenever the model answered in prose; a misparsed "
            "probe scores WRONG, so the true values are >= those reported. The bound is NOT "
            "uniform across arms -- excess forward shift-hits are 52 for text_rle against 28 for "
            "px16 -- so the arms cannot be safely compared even though each is individually "
            "conservative. The run CANNOT be re-scored: it persisted only the parsed predictions "
            "and not the model's replies, and the server log does not contain them either. "
            "Correcting the numbers requires a fresh run at roughly 4 GPU-hours. The parser is "
            "fixed and raw replies are now persisted, so this is not repeatable.",
        ],
        "prior_context": (
            "arc_executable_world_model.to_ascii() already collapses 4/14 and 5/15 today by "
            "rendering each cell as str(int(v))[-1], the last decimal digit only. This gate "
            "exists so a lossy image encoder is not stacked on top of that same bug class "
            "undetected."
        ),
        "cited_upstream_artifacts": [],
        # ---------------- CORRIGENDA ----------------
        # Six defects found by an adversarial review of the first published version of
        # this artifact, all VERIFIED against the run's own recorded data before being
        # applied. Recorded rather than silently patched, per the project's never-prune
        # rule: the prior claims are stated here alongside what replaced them, so a
        # reader who saw v1 can tell exactly what moved and why.
        "post_review_corrections": [
            {
                "id": "F1",
                "severity": "blocker",
                "was": "exact_match reported as a point estimate; limitations attributed the "
                "per-batch bimodality to the model's spatial indexing.",
                "defect": "parse()'s positional fallback matched every bare 1-2 digit number in "
                "a prose reply and took the first 8 in 0..15 as the answers, so row "
                "and column indices were scored as colour predictions. 54 of 128 "
                "probes have a row or column below 16.",
                "verified_by": "Running the shipped parser on a prose reply returns the row and "
                "column as colours. In the run's own predictions, wrong answers "
                "match an EARLIER probe's truth in the same batch far more often "
                "than a later one -- fixed-shift k=+1/+2/+3 vs k=-1/-2/-3: "
                "text_rle 27/23/16 vs 5/5/4, px16 18/15/8 vs 6/4/3. Those hits "
                "sit almost entirely in the batches that scored <=2/8.",
                "now": "All exact_match values labelled lower bounds; shift_contamination_by_"
                "scheme added; the spatial-indexing attribution retracted; parser fixed "
                "and raw replies now persisted in fidelity.py.",
                "unresolved": "The completed run cannot be re-scored -- only parsed predictions "
                "were saved. Corrected numbers need a fresh ~4 GPU-hour run.",
            },
            {
                "id": "F2",
                "severity": "blocker",
                "was": "five hardcoded McNemar p-values, two of them literally 0.0.",
                "defect": "an exact test cannot return 0, and no code on disk computed any of "
                "them, so nothing could catch the error.",
                "verified_by": "recomputed from `predictions`: the discordant counts (39/25, "
                "50/11, 46/7, 36/11, 34/9) were all correct, but the two 0.0 "
                "values should be 4.5888e-07 and 4.0032e-08.",
                "now": "all five computed from the data at build time, each with its "
                "min_reachable_p.",
            },
            {
                "id": "F3",
                "severity": "blocker",
                "was": "sample_size_justification claimed '128 on the real ka59 frame "
                "(stratified over the 7 colours that frame actually contains)'.",
                "defect": "no real-frame arm was ever measured, and the staged grid was lp85 "
                "with 11 colours, not ka59 with 7. A quoted number with no "
                "measurement behind it.",
                "verified_by": "renderings holds 4 rows, all synthetic_blocks; real_frame_* are "
                "null; escalate.py sets GAME='lp85' and documents rejecting ka59.",
                "now": "clause deleted; solve_provenance_note and the grid_kind tag corrected "
                "to lp85.",
            },
            {
                "id": "F4",
                "severity": "major",
                "was": "the escalation was 'DELIBERATELY NOT RUN'.",
                "defect": "contradicted by the run log -- it was launched and killed.",
                "verified_by": "run_all.sh invokes escalate.py unconditionally; run_all.log ends "
                "with escalate.py's lp85 meta block and its TEXT CONTROL header, "
                "then stops with no batch lines and no ALL_DONE.",
                "now": "restated as launched-and-terminated-before-any-batch; the reasoning for "
                "stopping it is retained and unchanged.",
            },
            {
                "id": "F5",
                "severity": "major",
                "was": "probe-level Wilson CIs and probe-level McNemar, assuming 128 "
                "independent probes.",
                "defect": "probes were asked in 16 batched requests; the artifact even noted the "
                "resulting bimodality and still published probe-level statistics.",
                "verified_by": "design effects computed from per-batch counts: text_rle 5.57, "
                "px16 6.96, px8 5.75, px1 3.35. px16's effective n is 18.4, not "
                "128; its cluster-robust CI is about [0.21, 0.71] against the "
                "published Wilson [0.377, 0.547].",
                "now": "cluster-robust intervals and design effects reported per rendering; the "
                "p-values carry an explicit anticonservative caveat.",
            },
            {
                "id": "F6",
                "severity": "major",
                "was": "can_supplement_text and negative_control_informative both decided by "
                "SUPPLEMENT_BAR = 0.25.",
                "defect": "that bar exists only in build_artifact.py, written after the numbers "
                "were in. Both positive conclusions rested on a threshold chosen with "
                "the answer visible, while sample_size_justification claimed the "
                "precision floor was fixed before any result was inspected.",
                "verified_by": "fidelity.py's pre-registration block (mtime 02:36, run finished "
                "06:47) registers only the 0.90 rule and the 6.25% chance rate; "
                "SUPPLEMENT_BAR appears only in build_artifact.py, mtime 06:51.",
                "now": "bar marked post-hoc and retired as a decision rule. Both conclusions "
                "re-anchored on the pre-registered 6.25% chance rate, tested against the "
                "cluster-robust lower bound -- a strictly harder test that also survives "
                "F5, and one the F1 lower-bound direction can only strengthen.",
            },
        ],
    }
    return art


def verdict_for(art: dict) -> str:
    """Terminal-prefixed verdict, derived from the measured numbers.

    Deliberately NOT prefixed `success:` -- that literal is one of
    adversarial_verify's _PERCEPTION_WIN_MARKERS, and this artifact makes no
    perception WIN claim at all. `complete_` is the correct terminal prefix here.

    ORDER MATTERS, AND AN EARLIER DRAFT OF THIS FUNCTION GOT IT WRONG. The draft
    checked the negative control FIRST and returned "question-format confound"
    whenever the text arm scored low -- which, on a dry run, labelled a 0.95-vision /
    0.19-text outcome a confound. That is backwards. The control's veto exists for one
    specific situation named in the gate's own design: if the text arm ALSO collapses,
    then "read the cell at (r,c)" is simply an unanswerable question format and NO
    conclusion about vision is licensed. But if the VISION arm answers it well, the
    format is demonstrably answerable, and a weak text arm is then a FINDING (vision
    beats the shipped text encoding), not a confound. So the veto is only consulted on
    the branch where vision has already failed.
    """
    if art["can_replace_text"]:
        return "complete_vision_tower_reads_arc_colour_indices_exactly_may_replace_text"
    if art["can_supplement_text"]:
        return (
            "complete_vision_tower_carries_real_signal_but_below_exact_match_bar_"
            "supplement_only_never_replace"
        )
    # Vision failed. Only now does the control decide whether that failure is
    # attributable to vision at all.
    if not art["negative_control_informative"]:
        return "complete_gate_inconclusive_both_arms_collapsed_question_format_confound_not_vision"
    return "complete_vision_tower_cannot_read_arc_colour_indices_null_result_text_only"


def main():
    art = build()
    art["honest_verdict"] = verdict_for(art)
    art["acceptance_gate_can_replace_text_passed"] = art["can_replace_text"]
    art["acceptance_gate_can_supplement_text_passed"] = art["can_supplement_text"]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    # Checksum over the content, computed before the checksum field itself exists.
    art["reproducibility_checksum"] = hashlib.sha256(
        json.dumps(art, sort_keys=True).encode()
    ).hexdigest()

    p = OUTDIR / "experiment_arc_multimodal_fidelity_20260801.json"
    p.write_text(json.dumps(art, indent=2))
    print("wrote", p)
    print("verdict:", art["honest_verdict"])
    for r in art["renderings"]:
        print(
            f"  {r['scheme']:34s} exact={r['exact_match']:.3f} ci={r['exact_match_ci95_wilson']} "
            f"unparsed={r['n_unparseable']:3d} 4/14={r['confusions']['pair_4_14_confusions']} "
            f"5/15={r['confusions']['pair_5_15_confusions']}"
        )
    return art


if __name__ == "__main__":
    main()
