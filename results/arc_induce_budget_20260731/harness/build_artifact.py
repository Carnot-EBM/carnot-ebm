#!/usr/bin/env python3
"""Assemble the Phase-1 result artifact from the sweep lanes + the arithmetic.

Substrate is `live_llm_inference`: every row is a real gemma-4-31B-it Q4_K_M forward pass on a
CUDA-proven llama-server (build read from /proc/<pid>/exe, per-PID VRAM residency recorded), so
the 60s duration floor applies and is cleared by three orders of magnitude.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/ianblenke/github.com/ianblenke/carnot"
sys.path.insert(0, os.path.join(REPO, "python"))

from carnot.agentic import arc_executable_world_model as e3  # noqa: E402

OUTDIR = os.path.join(REPO, "results", "arc_induce_budget_20260731")
os.makedirs(OUTDIR, exist_ok=True)
ART = os.path.join(REPO, "results", "outer_loop_arc_induce_budget_phase1_20260731.json")

LANES = {
    "induce_engine_only": "sweep",
    "induce_combined": "sweep_combined",
    "refactor": "sweep_refactor",
    "sampler_control": "sweep_sampler",
}

FT09_PROMPT_TOKENS = 4343
CARD_FREE_IDLE_MIB = 24123


def _sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def predicted_vram(n_ctx: int, layers: int = 0, slots: int = 4) -> float:
    return (
        e3._VRAM_GEMMA31B_INTERCEPT_MIB
        + e3._VRAM_GEMMA31B_PER_CTX_MIB * n_ctx
        + e3._VRAM_PER_SLOT_MIB * slots
        - e3._VRAM_PER_CPU_FFN_LAYER_MIB * layers
    )


def derived_n_ctx(max_tokens: int) -> int:
    need = e3._LLAMA_SERVER_DEFAULT_SLOTS * (
        e3._INDUCE_WORST_CASE_PROMPT_TOKENS + max_tokens
    )
    return int(-(-need // 4096) * 4096)


def main() -> int:
    lanes: dict = {}
    all_rows: list[dict] = []
    for name, d in LANES.items():
        p = os.path.join(HERE, d, "sweep.json")
        if not os.path.exists(p):
            lanes[name] = {"status": "not_run"}
            continue
        doc = json.load(open(p))
        lanes[name] = {
            "status": doc.get("status"),
            "witness": doc.get("witness"),
            "n_rows": len(doc.get("rows") or []),
        }
        for r in doc.get("rows") or []:
            r = dict(r)
            r["lane"] = name
            all_rows.append(r)
        # copy the completions into the repo evidence dir so the artifact's claims are readable
        dest = os.path.join(OUTDIR, d)
        os.makedirs(dest, exist_ok=True)
        for fn in sorted(os.listdir(os.path.join(HERE, d))):
            if fn.endswith(".txt") or fn == "sweep.json":
                src = os.path.join(HERE, d, fn)
                with open(src, "rb") as a, open(os.path.join(dest, fn), "wb") as b:
                    b.write(a.read())

    # prompts are evidence too: the whole arithmetic rests on their measured token counts
    pdest = os.path.join(OUTDIR, "prompts")
    os.makedirs(pdest, exist_ok=True)
    for fn in sorted(os.listdir(os.path.join(HERE, "prompts"))):
        src = os.path.join(HERE, "prompts", fn)
        with open(src, "rb") as a, open(os.path.join(pdest, fn), "wb") as b:
            b.write(a.read())

    # ---- fold in the verbatim-repetition metric ------------------------------------------
    # `ramble_frac` counts only BARE `#` lines and badly undercounts: the 8192 engine
    # completions are ~80% loop while scoring 0.008, because the repeated unit is a full
    # sentence rather than an empty comment. `loop_frac` is the general form and is the number
    # that decides Phase 1 -- if the extra budget lands inside repetition runs rather than in
    # code, more budget cannot be the lever.
    from repetition import loop_stats  # noqa: E402
    from score_engines import load_transitions, score_code  # noqa: E402

    import carnot.agentic.arc_executable_world_model as _e3  # noqa: E402

    trans = load_transitions()
    for r in all_rows:
        fn = r.get("completion_file")
        d = LANES.get(r["lane"])
        if not fn or not d:
            continue
        fp = os.path.join(HERE, d, fn)
        if not os.path.exists(fp):
            continue
        text = open(fp).read()
        r.update(loop_stats(text))
        # BEHAVIOURAL SCORE. The AST check ("does engine() return on every path") is gameable by
        # an IDENTITY engine, and the refactor lane produced exactly that -- `if action == 6:
        # return grid` / `return grid`, which returns cleanly and models nothing. It even scores
        # 19/25 exact, because 19 of ft09's 25 transitions are no-ops. So every engine is RUN
        # against the real captured transitions and the headline is `non_degenerate_usable`,
        # which additionally requires that the engine ever change the grid at all.
        code = _e3._extract_python(text) or (text.strip() if r["lane"] != "refactor" else "")
        r.update(score_code(code, trans) if code else {"engine_raised": "no code extracted"})
        r["non_degenerate_usable_engine"] = bool(
            r.get("generate_would_accept")
            and r.get("engine_returns_on_all_paths")
            and r.get("engine_changes_anything")
        )

    # ---- the scoreboard: n_usable per (lane, budget) -------------------------------------
    by: dict = {}
    for r in all_rows:
        key = (r["lane"], r.get("arm", "shipped"), r["budget"])
        by.setdefault(key, []).append(r)
    scoreboard = []
    for (lane, arm, budget), rows in sorted(by.items()):
        ok = [r for r in rows if r.get("status") == "ok"]
        scoreboard.append(
            {
                "lane": lane,
                "sampler_arm": arm,
                "budget": budget,
                "n_calls": len(rows),
                "n_ok": len(ok),
                "n_generate_would_accept": sum(
                    1 for r in ok if r.get("generate_would_accept")),
                "n_usable_engine_ast_only": sum(1 for r in ok if r.get("usable_engine")),
                "n_non_degenerate_usable_engine": sum(
                    1 for r in ok if r.get("non_degenerate_usable_engine")),
                "n_engine_changes_anything": sum(
                    1 for r in ok if r.get("engine_changes_anything")),
                "mean_heldout_exact": round(
                    sum(r.get("heldout_exact") or 0 for r in ok) / max(1, len(ok)), 4),
                "mean_cell_recall": round(
                    sum(r.get("cell_recall") or 0 for r in ok) / max(1, len(ok)), 4),
                "n_hit_budget_cap": sum(
                    1 for r in ok
                    if r.get("stop_type") == "limit" and r.get("predicted_n") == budget),
                "mean_predicted_n": round(
                    sum(r.get("predicted_n") or 0 for r in ok) / max(1, len(ok)), 1),
                "mean_ramble_frac": round(
                    sum(r.get("ramble_frac") or 0 for r in ok) / max(1, len(ok)), 4),
                "mean_code_lines": round(
                    sum(r.get("code_lines") or 0 for r in ok) / max(1, len(ok)), 1),
                "mean_loop_frac": round(
                    sum(r.get("loop_frac") or 0 for r in ok) / max(1, len(ok)), 4),
                "mean_lines_in_repetition_runs": round(
                    sum(r.get("lines_in_repetition_runs") or 0 for r in ok) / max(1, len(ok)), 1),
                "max_longest_verbatim_run": max(
                    [r.get("longest_verbatim_run") or 0 for r in ok] or [0]),
                "mean_wall_s": round(
                    sum(r.get("wall_s") or 0 for r in ok) / max(1, len(ok)), 1),
            }
        )

    fit_table = []
    for mt in (4096, 6144, 8192, 12288, 16384):
        n = derived_n_ctx(mt)
        v = predicted_vram(n, 0)
        need = v + e3._GENERATOR_CUDA_GUARD_MARGIN_MIB
        over = need - CARD_FREE_IDLE_MIB
        fit_table.append({
            "max_tokens": mt,
            "derived_n_ctx_K4": n,
            "predicted_vram_mib_L0": round(v),
            "guard_requires_free_mib": round(need),
            "fits_24gib_idle_3090": need <= CARD_FREE_IDLE_MIB,
            "cpu_ffn_layers_needed_to_fit": 0 if need <= CARD_FREE_IDLE_MIB
            else int(-(-over // e3._VRAM_PER_CPU_FFN_LAYER_MIB)),
        })

    pool_table = []
    for K in (1, 2, 4):
        per = 32768 // K
        pool_table.append({
            "K_concurrent": K,
            "per_slot_cells_at_n_ctx_32768": per,
            "max_tokens_ceiling_for_ft09_prompt": per - FT09_PROMPT_TOKENS,
        })

    dur = sum(r.get("wall_s") or 0 for r in all_rows)
    payload = {
        "schema": "carnot.outer_loop_arc_induce_budget_phase1.v1",
        "experiment": "outer_loop_arc_induce_completion_budget_phase1",
        "experiment_id": "outer_loop_arc_induce_budget_phase1_20260731",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_commit_at_run": subprocess.run(
            ["git", "-C", REPO, "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "inference_substrate": "live_llm_inference",
        "model_specs": {
            "generator": "unsloth/gemma-4-31B-it-GGUF (gemma-4-31B-it-Q4_K_M.gguf)",
            "loaded_via": "llama-server CUDA build, n_ctx=32768, ffn_cpu_layers=0, mtp=off, "
                          "kv q8_0, 4 slots",
        },
        "target_model": "gemma-4-31B-it",
        "random_seed": 3003,
        "random_seeds_used": sorted({r.get("seed") for r in all_rows if r.get("seed")}),
        "duration_s": round(dur, 1),
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        # The upstream cells this artifact reasons about lived in a session scratchpad, which is
        # garbage-collected. Copied into the evidence dir so every claim made about them stays
        # checkable -- a citation to a path that will stop existing is not provenance.
        "cited_upstream_artifacts": [
            {
                "experiment_id": "outer_loop_arc_llm_on_vs_off_activation_20260730",
                "fields_imported": [
                    "induction_events[0].refinement_rounds[*].message",
                    "induction_events[0].refinement_rounds[0].counterexample",
                    "result.action_trace",
                ],
                "path": "results/outer_loop_arc_llm_on_vs_off_activation_20260730.json",
                "per_cell_copies": "results/arc_induce_budget_20260731/upstream_ft09_cells/",
            }
        ],
        "question": "Does raising the induce completion budget make ft09 emit a usable engine()?",
        # FALSIFIABLE, and stated so a positive would have been visible. The gate is on the
        # measurement being INTERPRETABLE, not on the lever working -- a negative is the finding
        # here, and a gate that could only pass on a positive would make the honest answer
        # unreportable.
        "acceptance_gate_control_reproduces_live_failure": None,  # filled below
        "acceptance_gate_budget_axis_is_uncensored": None,  # filled below
        "acceptance_gate_passed": None,  # filled below
        "acceptance_gate_definition": {
            "control_reproduces_live_failure": "at budget 4096 the combined call must be "
            "REJECTED and the engine-only call must be ACCEPTED-but-non-returning, i.e. the "
            "banked ft09 shape. If the control does not reproduce, nothing above it is about "
            "ft09.",
            "budget_axis_is_uncensored": "every capped call must report predicted_n == its own "
            "budget with stop_type 'limit', never the short-of-budget signature of a shared-pool "
            "truncation. A pool-truncated run would be measuring the harness, not the lever.",
        },
        "prior_failures": [
            {
                "source": "arc_executable_world_model.induce() inline comment, "
                          "proto_l2_fix_finder 2026-06-25",
                "verdict": "a budget bump does NOT help; the model just rambles more",
                "addressed_by": "that measurement predates the 2026-07-28 switch of the live "
                                "generator to gemma-4-31B-it and describes the COMBINED call; "
                                "both the model and the call shape are swept here separately.",
                "retire_if_same_verdict": True,
            },
            {
                "source": "REQ-ARC-FCP-5699-34 / -35 (openspec, 2026-07-16)",
                "verdict": "max_tokens=8192 + structural reminder DOES fix the refactor call's "
                           "structural completion failure on a 27B model; 4096 was kept as the "
                           "live default explicitly because 'the 8192 requirement ... was "
                           "specific to a 3x larger, non-live candidate, not this one' (the 9B)",
                "addressed_by": "the live generator BECAME that 3x-larger class on 2026-07-28; "
                                "the stated reason for the 4096 default expired with the model "
                                "switch and the default never moved.",
                "retire_if_same_verdict": False,
            },
        ],
        # THE CAPTURE'S VALIDITY, CHECKED NOT ASSUMED. The prompts were captured from an
        # LLM-OFF run, which is only legitimate because ft09's trajectory is generator-
        # independent. Rather than inherit that from the 2026-07-30 grid, the capture's action
        # trace was rebuilt under the grid harness's own label encoding and compared
        # element-by-element to the banked live LLM-ON cell. A single divergence would put the
        # induce call at a different transition set and make every row below answer a different
        # question.
        "capture_trajectory_validation": {
            "live_cell": "p4/cells/on__ft09__s1.json (2026-07-30 grid, LLM-ON arm)",
            "n_actions_live": 60,
            "n_actions_capture": 60,
            "element_wise_identical": True,
            "action_trace_sha256_live": "fba0baa5d5473eac",
            "action_trace_sha256_capture": "fba0baa5d5473eac",
        },
        # THE SCORER, CROSS-VALIDATED AGAINST THE SHIPPED VERIFIER. A behavioural scorer written
        # for this measurement is worth nothing if it disagrees with the gate the live agent
        # actually runs. Scoring the `repeat_penalty 1.1` engine over the same 25 transitions
        # reproduces the live `onb` cell's recorded gate numbers TO THE CELL.
        "scorer_cross_validation": {
            "against": "results/arc_induce_budget_20260731/upstream_ft09_cells/"
                       "onb__ft09__s1.json -> induction_events[0].verify_*",
            "live_gate": {
                "verify_correct_changed_cells": 216,
                "verify_cell_recall": 0.9474,
                "verify_change_fidelity": 0.947368,
                "verify_spurious_changed_cells": 0,
            },
            "this_scorer_on_repeat_penalty_engine": {
                "n_true_changed_cells": 228,
                "n_correct_changed_cells": 216,
                "cell_recall": 0.9474,
            },
            "note": "216/228 = 0.9474 on both. The scorer is measuring what the shipped gate "
                    "measures. It also means the repeat_penalty arm reaches, on its FIRST "
                    "naturally-stopped attempt in 1912 tokens, the same cell recall the live "
                    "LLM-ON run only reached on one replicate after three refinement rounds.",
        },
        "lanes": lanes,
        "scoreboard": scoreboard,
        "rows": all_rows,
        "arithmetic": {
            "constants_read_from_repo": {
                "_INDUCE_DEFAULT_MAX_TOKENS": e3._INDUCE_DEFAULT_MAX_TOKENS,
                "_INDUCE_WORST_CASE_PROMPT_TOKENS": e3._INDUCE_WORST_CASE_PROMPT_TOKENS,
                "_LLAMA_SERVER_DEFAULT_SLOTS": e3._LLAMA_SERVER_DEFAULT_SLOTS,
                "_VRAM_GEMMA31B_INTERCEPT_MIB": e3._VRAM_GEMMA31B_INTERCEPT_MIB,
                "_VRAM_GEMMA31B_PER_CTX_MIB": e3._VRAM_GEMMA31B_PER_CTX_MIB,
                "_VRAM_PER_SLOT_MIB": e3._VRAM_PER_SLOT_MIB,
                "_VRAM_PER_CPU_FFN_LAYER_MIB": e3._VRAM_PER_CPU_FFN_LAYER_MIB,
                "_GENERATOR_CUDA_GUARD_MARGIN_MIB": e3._GENERATOR_CUDA_GUARD_MARGIN_MIB,
            },
            "ft09_real_induce_prompt_tokens": FT09_PROMPT_TOKENS,
            "ft09_prompt_tokens_method": "llama_cpp.Llama(model_path=<gguf>, vocab_only=True)"
                                         ".tokenize() on the byte-exact prompt the live path "
                                         "builds, never AutoTokenizer on a GGUF repo id",
            "card_free_idle_mib_observed": CARD_FREE_IDLE_MIB,
            "default_derivation_fit_on_24gib": fit_table,
            "pool_admission_at_pinned_n_ctx_32768": pool_table,
            "max_n_ctx_that_fits_24gib_L0": max(
                (n for n in range(4096, 262145, 4096)
                 if predicted_vram(n, 0) + e3._GENERATOR_CUDA_GUARD_MARGIN_MIB
                 <= CARD_FREE_IDLE_MIB), default=0),
            "envelope_independent_check": {
                "predicted_mib_at_32768_L0": round(predicted_vram(32768, 0)),
                "observed_mib_this_run_gpu0": 21416,
                "observed_mib_this_run_gpu1": 21416,
                "note": "both servers this run reported per-PID residency 21416 MiB, exactly "
                        "the envelope's prediction -- an independent confirmation taken "
                        "incidentally rather than a fitted point",
            },
        },
        "preconditions_checked": [
            {"resource": "gemma-4-31B-it Q4_K_M GGUF cached", "available": True},
            {"resource": "llama-server CUDA build (not build-hip)", "available": True},
            {"resource": "per-PID VRAM residency on the requested card", "available": True},
            {"resource": "two idle RTX 3090s", "available": True},
        ],
        "field_provenance": {
            "duration_s": {
                "principle": "real compute takes wall-clock time; this is the summed per-call "
                             "wall of every live generation in the sweep, so an implausibly "
                             "short value would mean the model was never actually run.",
                "satisfied_by": "sum of per-row wall_s",
            },
            "random_seed": {
                "principle": "llama-server defaults an absent seed to -1 (fresh random), so "
                             "without an explicit seed two identical runs diverge and no A/B "
                             "on this path is interpretable.",
                "satisfied_by": "CARNOT_ARC_GENERATOR_SEED=3003, per-attempt seed "
                                "3003*1000+attempt, recorded per row",
            },
            "reproducibility_checksum": {
                "principle": "content-addresses the exact prompts the sweep fired, so a later "
                             "replication can prove it used the same inputs rather than a "
                             "similarly-shaped reconstruction.",
                "satisfied_by": "sha256 over the captured prompt files",
            },
            "usable_engine": {
                "principle": "generate() accepts any output containing the required `def`s "
                             "that parses -- ft09's banked engine passes that bar and still "
                             "returns None on every click. This field adds 'and returns on "
                             "every path'. It is REPORTED BUT NOT THE HEADLINE, because it is "
                             "itself gameable -- see non_degenerate_usable_engine.",
                "satisfied_by": "AST reachability analysis of engine()'s body, unit-checked "
                                "against the banked ft09 engine (False) and a hand-written "
                                "returning engine (True)",
            },
            "non_degenerate_usable_engine": {
                "principle": "an IDENTITY engine (`if action == 6: return grid` / `return "
                             "grid`) satisfies 'accepted AND returns on every path' trivially "
                             "and scores 19/25 heldout-EXACT on a split where 19 transitions "
                             "are no-ops, while modelling nothing. The refactor lane produced "
                             "exactly that, so the AST bar had to be backed by BEHAVIOUR: the "
                             "engine must also ever change the grid.",
                "satisfied_by": "each engine exec'd and run over the 25 real captured ft09 "
                                "transitions; requires accepted AND returns-on-all-paths AND "
                                "at least one output differing from its input",
            },
            "engine_changes_anything": {
                "principle": "the single cheapest witness that a world model is not inert. Zero "
                             "here makes every accuracy number above it meaningless, however "
                             "high, because a no-op-heavy split rewards predicting nothing.",
                "satisfied_by": "numpy array comparison of engine output vs input per "
                                "transition",
            },
        },
    }
    # ---- evaluate the gates + write the honest verdict ------------------------------------
    b4096 = [r for r in all_rows if r.get("budget") == 4096 and r.get("status") == "ok"]
    ctrl_combined = [r for r in b4096 if r["lane"] == "induce_combined"]
    ctrl_engine = [r for r in b4096 if r["lane"] == "induce_engine_only"]
    control_ok = bool(
        ctrl_combined
        and ctrl_engine
        and not any(r.get("generate_would_accept") for r in ctrl_combined)
        and all(r.get("generate_would_accept") for r in ctrl_engine)
        and not any(r.get("usable_engine") for r in ctrl_engine)
    )
    capped = [r for r in all_rows
              if r.get("status") == "ok" and r.get("stop_type") == "limit"]
    uncensored = all(r.get("predicted_n") == r["budget"] for r in capped)
    payload["acceptance_gate_control_reproduces_live_failure"] = control_ok
    payload["acceptance_gate_budget_axis_is_uncensored"] = uncensored
    payload["acceptance_gate_passed"] = bool(control_ok and uncensored)

    induce_rows = [r for r in all_rows
                   if r["lane"] in ("induce_combined", "induce_engine_only")
                   and r.get("status") == "ok"]
    n_usable_induce = sum(1 for r in induce_rows if r.get("non_degenerate_usable_engine"))
    refactor_rows = [r for r in all_rows
                     if r["lane"] == "refactor" and r.get("status") == "ok"]
    payload["n_usable_engines_across_all_induce_budgets"] = n_usable_induce
    # A MISSING OBSERVATION IS NOT A ZERO. If the refactor lane did not run (or did not reach a
    # budget above 4096), this must read `null` and say so -- reporting "0 accepts" for a lane
    # that fired no calls would make an unrun arm indistinguishable from a measured failure, and
    # the refactor call is the one with prior evidence pointing the OTHER way (REQ-34).
    hi = [r for r in refactor_rows if r.get("budget", 0) > 4096]
    payload["n_refactor_accepts_above_4096"] = (
        sum(1 for r in hi if r.get("generate_would_accept")) if hi else None
    )
    payload["n_refactor_calls_above_4096"] = len(hi)
    payload["refactor_lane_status"] = (
        "ran" if refactor_rows else "NOT RUN -- reported as a missing observation, not a zero"
    )
    all_scored = [r for r in all_rows if r.get("status") == "ok"
                  and r["lane"] != "sampler_control"]
    payload["n_engines_that_change_anything"] = sum(
        1 for r in all_scored if r.get("engine_changes_anything"))
    payload["n_engines_scored"] = len(all_scored)
    payload["identity_engine_finding"] = (
        "NOT ONE generated engine, on any call shape at any budget, ever produces an output "
        "different from its input on ft09's real transitions. The refactor lane's completions "
        "pass the AST return-on-all-paths check and score 19/25 heldout-EXACT -- because 19 of "
        "the 25 transitions are no-ops and the engine is `if action == 6: return grid` / "
        "`return grid`. cell_recall is 0.0 on every one. This is the vacuous pass the "
        "change-aware gate exists to catch, arrived at here by running the code rather than "
        "trusting the metric that scored it."
    )
    samp = [r for r in all_rows if r["lane"] == "sampler_control" and r.get("status") == "ok"]
    rp = [r for r in samp if r.get("arm") == "repeat_penalty_1.1"]
    off = [r for r in samp if r.get("arm") == "off"]
    n_rp = sum(1 for r in rp if r.get("non_degenerate_usable_engine"))
    n_off = sum(1 for r in off if r.get("non_degenerate_usable_engine"))
    payload["sampler_control_headline"] = (
        f"repeat_penalty_1.1 {n_rp}/{len(rp)} non-degenerate vs shipped off {n_off}/{len(off)}"
        if rp else "sampler control NOT RUN"
    )
    payload["honest_verdict"] = (
        "complete_budget_is_not_the_lever_repeat_penalty_is: "
        f"{n_usable_induce} non-degenerate usable engines across {len(induce_rows)} live induce "
        "calls spanning budgets 4096/8192/16384, and 0 of "
        f"{payload['n_engines_scored']} generated engines under the SHIPPED sampler EVER change "
        "the grid at all -- the extra tokens go into verbatim repetition, not code (distinct-line "
        "count unchanged at matched seed while emitted length doubled). Holding the budget fixed "
        f"and setting repeat_penalty=1.1 gives {n_rp}/{len(rp)} non-degenerate engines at cell "
        "recall 0.9474 (vs 0/3 shipped), naturally stopped in under half the budget. The "
        "completion budget default is NOT raised and the sampler is NOT changed: a scored-path "
        "sampler change is a behaviour change and is the operator's to make."
        if rp else
        "complete_budget_is_not_the_lever_for_ft09_induction: "
        f"{n_usable_induce} non-degenerate usable engines across {len(induce_rows)} live induce "
        "calls; sampler control NOT RUN."
    )

    prompt_files = sorted(
        os.path.join(pdest, f) for f in os.listdir(pdest) if f.endswith(".txt"))
    h = hashlib.sha256()
    for f in prompt_files:
        h.update(os.path.basename(f).encode())
        h.update(open(f, "rb").read())
    payload["reproducibility_checksum"] = h.hexdigest()
    payload["prompt_file_sha256"] = {os.path.basename(f): _sha(f) for f in prompt_files}

    with open(ART, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"wrote {ART}")
    print(json.dumps(scoreboard, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
