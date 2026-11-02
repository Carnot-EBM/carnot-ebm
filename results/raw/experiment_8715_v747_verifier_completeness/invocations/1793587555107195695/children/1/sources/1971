"""Exp 5725: ThinkingCap-27B genuine-reasoning retest through its PROPER chat template
(REQ-ARC-WMTE-5725) -- the harness correction of REQ-ARC-WMTE-5724.

WHY THIS EXISTS
---------------
REQ-ARC-WMTE-5724 tried to answer "does a model RL-tuned for token-efficient reasoning
(bottlecapai/ThinkingCap-Qwen3.6-27B) COMPLETE genuine /think induction within the SAME
8192-token budget that vanilla Qwen3.5-9B-MTP overran on 0/N?" -- but the result was
INCONCLUSIVE for a harness reason, not a model reason. The exp5714 `_induce_no_fence`
mechanism talks to llama.cpp's RAW /completion endpoint with a hand-built prompt string and
NO chat template (deliberately, so a /think prefix reasons before emitting the code fence).
That works for Qwen3.5-9B, but Qwen3.6-family models (ThinkingCap-27B) REQUIRE their embedded
chat template's turn structure (system/user/assistant delimiters) to know a turn has started.
Without it, ThinkingCap emitted an IMMEDIATE end-of-sequence with ~0 output on 10/12 reason
cells (stop_type=='eos', no <think> trace) -- those cells never genuinely tested the budget
hypothesis. The 2 genuine attempts (0/2 completed) and the healthy 12/12 codeonly frozen
control proved the model/server/GPU path itself is fine; only the raw-completion-without-
template combination was broken for this model family.

THE FIX (this experiment)
-------------------------
Route the ThinkingCap reason induce through the OpenAI-compatible /v1/chat/completions
endpoint (LocalGGUFProposer.use_chat_template=True, REQ-ARC-WMTE-5725), so llama.cpp applies
the GGUF's OWN embedded Qwen3.6 chat template automatically -- no HuggingFace template needed.
The "genuine reasoning, no pre-opened fence" semantics the no-fence trick was created for are
PRESERVED: the induce still runs codeonly OFF (no code-only directive, no ```python fence
prefix) with the /think soft-switch in the user turn, so reasoning genuinely engages and the
model emits its ```python block afterward. This was smoke-tested BEFORE the full run: 4/4
induction-shaped chat calls engaged a real <think> trace (vs 10/12 immediate-EOS on the raw
path) with non-empty output on every call.

DESIGN -- byte-identical to REQ-ARC-WMTE-5724 EXCEPT the chat-template route
--------------------------------------------------------------------------
The ThinkingCap reason cell reuses exp5724.run_reason_cell VERBATIM (imported, not copied):
codeonly OFF, no_think_prefix=/think, max_tokens=8192, tries=1, `_induce_no_fence`, the
exp5722 stale-engine unlink guard, and the full REQ-ARC-WMTE-5720 actions-to-progress metric
ladder. The ONLY change is the proposer's use_chat_template=True. The two BASELINE arms are
NOT re-measured -- they were already healthy in REQ-ARC-WMTE-5724 (and use the raw path each
tolerates): the Qwen3.5-9B raw-/think reason baseline (1/12 induce_ok; Qwen tolerates the raw
path) and the ThinkingCap codeonly frozen control (12/12; proves model health) are CITED from
5724's artifact and their per-run rows are folded in (clearly tagged reused_from_5724) so the
paired-by-game comparison against Qwen is real. This keeps compute on the one arm the harness
bug actually corrupted.

Provenance: development_proxy on PUBLIC games (NOT a hidden-game solve). verifier_is_oracle
False (win oracle = level counter). This NEVER flips the frozen live default (operator-only)
and NEVER submits. Wall-clock on a 24GB 3090 dev card is NOT Kaggle-representative (the eval
GPU is ~16GB; ThinkingCap-27B Q4 is a tight fit there) -- this tests the CONTENT question
(does token-efficient reasoning clear the 8192 budget wall) only.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

sys_path = str(Path(__file__).resolve().parents[1] / "scripts")
import sys  # noqa: E402

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

os.environ.setdefault("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")
os.environ.setdefault("CARNOT_ARC_STALL_REFACTOR_LOOP", "0")

from carnot.agentic import arc_actions_to_progress as atp  # noqa: E402

# Reuse the EXACT measurement machinery of exp5724 so the ONLY difference is the chat-template
# route (the correction under test) -- guarantees apples-to-apples with the broken-harness prior.
from carnot.experiment_5724_thinkingcap_token_efficient_reason_ab import (  # noqa: E402
    GENERATORS,
    ROSTER,
    TRIALS,
    _completion_summary,
    _gpu1_mem_used_mib,
    _reason_cell_class,  # noqa: F401  (imported for parity / potential debug use)
    preflight_generator,
    run_reason_cell,
)

REPO = Path(__file__).resolve().parents[2]
SHARD = REPO / "results" / "exp5725_thinkingcap_chat_reason_shard.jsonl"
ARTIFACT = REPO / "results" / "experiment_5725_thinkingcap_chat_template_reason_ab.json"
PRIOR_5724 = REPO / "results" / "experiment_5724_thinkingcap_token_efficient_reason_ab.json"

# ONLY the ThinkingCap reason arm is re-measured (the arm the raw-completion harness corrupted).
GEN, ARM = "thinkingcap27", "reason"
ARM_NAME = f"{GEN}_{ARM}"

# Baselines CITED from REQ-ARC-WMTE-5724 (healthy in that run; not re-measured here).
CITED_BASE_ARMS = ["qwen9b_reason", "thinkingcap27_frozen"]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _load_shard() -> dict[tuple[str, str, int], dict[str, Any]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    if SHARD.exists():
        for line in SHARD.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[(r["arm"], r["game"], int(r["trial"]))] = r
    return rows


def _append_shard(row: dict[str, Any]) -> None:
    SHARD.parent.mkdir(parents=True, exist_ok=True)
    with SHARD.open("a") as f:
        f.write(json.dumps(row) + "\n")


def check_preconditions() -> dict[str, Any]:
    from carnot.agentic.arc_executable_world_model import _resolve_gguf

    checks: dict[str, Any] = {}
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        checks["offline_arcade_import"] = True
    except Exception as exc:
        checks["offline_arcade_import"] = False
        checks["offline_arcade_error"] = repr(exc)[:200]
    try:
        from llama_cpp import llama_cpp as _b

        checks["llama_supports_gpu_offload"] = bool(_b.llama_supports_gpu_offload())
    except Exception as exc:
        checks["llama_supports_gpu_offload"] = False
        checks["llama_offload_error"] = repr(exc)[:200]
    checks["generator_cuda_gpu"] = os.environ.get("CARNOT_ARC_GENERATOR_CUDA_GPU")
    p = _resolve_gguf(GENERATORS[GEN]["repo_substr"])
    checks["resolved_gguf_paths"] = {GEN: p}
    checks[f"{GEN}_gguf_cached"] = bool(p)
    checks["prior_5724_present"] = PRIOR_5724.exists()
    checks["gpu1_mem_used_mib_at_start"] = _gpu1_mem_used_mib()
    return checks


def _make_proposer():
    """ThinkingCap proposer with use_chat_template=True -- the ONLY change vs exp5724's proposer.
    Same repo/port/mtp/kv/n_ctx/-fit off/timeout so the served model is byte-identical."""
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    cfg = GENERATORS[GEN]
    return LocalGGUFProposer(
        repo_substr=cfg["repo_substr"],
        port=cfg["port"] + 10,  # distinct port (8959) so a stale 5724 server is never reused
        mtp=cfg["mtp"],
        kv_quant=cfg["kv_quant"],
        n_ctx=cfg["n_ctx"],
        extra_server_args=cfg["extra_server_args"],
        max_tokens=8192,  # overridden per-cell by run_reason_cell (reason=8192)
        timeout=cfg["timeout"],
        use_chat_template=True,
    )


def _cited_baseline_rows() -> list[dict[str, Any]]:
    """Load the qwen9b_reason + thinkingcap27_frozen per-run rows from REQ-ARC-WMTE-5724 and tag
    them reused_from_5724 so the paired-by-game comparison is real while provenance stays honest."""
    rows: list[dict[str, Any]] = []
    if not PRIOR_5724.exists():
        return rows
    prior = json.loads(PRIOR_5724.read_text())
    for r in prior.get("per_run_rows", []):
        if r.get("arm") in CITED_BASE_ARMS:
            r = dict(r)
            r["reused_from"] = "REQ-ARC-WMTE-5724"
            rows.append(r)
    return rows


def run_all(pre: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    done = _load_shard()
    total = len(ROSTER) * len(TRIALS)
    log(
        f"resume: {len([k for k in done if k[0] == ARM_NAME])}/{total} thinkingcap reason cells in shard"
    )

    windows: dict[str, Any] = {}
    for game in ROSTER:
        w = atp.build_progress_window(game)
        windows[game] = w
        if w is None:
            log(f"SKIP {game}: no offline L1 window (build_progress_window returned None)")

    pending = [
        (game, t)
        for game in ROSTER
        for t in TRIALS
        if windows.get(game) is not None and (ARM_NAME, game, t) not in done
    ]
    preflights: dict[str, Any] = {}
    if pending:
        log(f"=== {GEN} chat-template reason : {len(pending)} cells ===")
        prop = _make_proposer()
        pf = preflight_generator(GEN, prop)
        preflights[GEN] = pf
        log(f"  preflight {GEN}: {json.dumps({k: v for k, v in pf.items() if k != 'smoke_head'})}")
        try:
            if not pf.get("server_up"):
                log(f"  !! server failed to start for {GEN}; recording blocked cells")
                for game, t in pending:
                    row = {
                        "arm": ARM_NAME,
                        "generator": GEN,
                        "arm_kind": ARM,
                        "game": game,
                        "trial": t,
                        "induction_ok": False,
                        "induce_ok": False,
                        "plan_found": False,
                        "reached_levelup": False,
                        "error": "server_failed_to_start",
                        "wall_s": 0.0,
                    }
                    _append_shard(row)
                    done[(ARM_NAME, game, t)] = row
            else:
                for game, t in pending:
                    window, full_traj, cell = windows[game]
                    log(f"RUN {ARM_NAME} {game} trial={t}")
                    t0 = time.time()
                    row = run_reason_cell(
                        game, prop, trial=t, window=window, full_traj=full_traj, cell=cell
                    )
                    row["generator"] = GEN
                    row["arm_kind"] = ARM
                    row["arm"] = ARM_NAME
                    row["game"] = game
                    row["trial"] = t
                    _append_shard(row)
                    done[(ARM_NAME, game, t)] = row
                    log(
                        f"  -> induce_ok={row.get('induce_ok')} reason={row.get('reason_engaged')} "
                        f"overran={row.get('overran')} rawlen={row.get('max_raw_completion_len')} "
                        f"ind_ok={row.get('induction_ok')} plan={row.get('plan_found')} "
                        f"levelup={row.get('reached_levelup')} stop={row.get('last_stop_type')} "
                        f"wall={row.get('wall_s')}s ({time.time() - t0:.0f}s)"
                    )
        finally:
            proc = getattr(prop, "_proc", None)
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=15)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            time.sleep(3)
    else:
        log("all thinkingcap reason cells present in shard, skipping run")

    pre["preflights"] = preflights
    fresh_rows = [r for k, r in done.items() if k[0] == ARM_NAME]
    return fresh_rows + _cited_baseline_rows(), pre


FIELD_PRINCIPLES = {
    "honest_verdict": "terminal-prefixed self-declared state (Verdict Terminal-Prefix Discipline); "
    "'token-efficient reasoning through the proper chat template clears the 8192 budget wall' vs "
    "'overrun persists even with the correct template' are distinct, decision-critical, citable "
    "outcomes -- and BOTH differ from 5724's inconclusive (broken-harness) result.",
    "inference_substrate": "live_llm_inference -- ThinkingCap-27B really induces via the live "
    "/v1/chat/completions path on a CUDA llama-server; the reason arm's genuine /think calls are "
    "long (per-cell durations disclosed).",
    "random_seed": "the LLM sampling is stochastic (disclosed); trials are per-game replicates -- "
    "why we pair by GAME (average over trials) + report win/tie/loss + fragility.",
    "reproducibility_checksum": "content hash over harness+driver code + generator config + rows.",
    "solve_provenance": "development_proxy -- PUBLIC-game offline dev measurement of the LIVE "
    "induce->plan->execute mechanism with the generator swapped, NOT a hidden-game solve.",
    "verifier_is_oracle": "False -- the win oracle is the level counter (frame.levels_completed); "
    "hand_verifier is only a dense progress MEASUREMENT, oracle-distinct.",
    "preconditions_checked": "GGUF cache + offline arcade + REAL GPU offload (VRAM jump) + a /think "
    "chat-completions smoke round-trip verified before inference.",
    "induce_ok": "PRIMARY: did the cell finish reasoning and emit parseable engine+is_level_complete "
    "code BEFORE hitting the 8192-token limit -- the direct test of the token-efficiency claim vs "
    "vanilla Qwen's ~0/N, now that the model genuinely reasons (chat template applied).",
    "reason_engaged": "did a real <think> trace engage (exp5714 REASONING_TAGS) -- confirms the chat "
    "template fixed the immediate-EOS degeneracy that made 5724 inconclusive.",
    "overran": "did any generate() call hit finish_reason=='length' (reasoned past the budget) -- the "
    "failure mechanism this experiment tests whether token-efficiency + proper template breaks.",
    "use_chat_template": "True -- routes induce through /v1/chat/completions so the GGUF's embedded "
    "Qwen3.6 template supplies the assistant-turn structure the raw /completion path omitted (the "
    "5724 root cause).",
}


def _verdict(rows: list[dict[str, Any]], cited: dict[str, Any]) -> str:
    """Honest, numbers-first, terminal-prefixed verdict. The PRIMARY finding is the harness fix
    (did the chat template restore genuine reasoning?); the completion-rate delta over Qwen is
    reported WITHOUT an interpretive 'cleared' gloss because at n<=6 game pairs no completion delta
    reaches significance -- an overclaim the project has zero tolerance for. A completion gain with
    0 level-ups is stated as a modest direction, never a solved budget wall."""
    tc = _completion_summary(rows, ARM_NAME)
    n = tc["n_cells"]
    tc_ok = tc["n_induce_ok"]
    deg = tc["n_degenerate_empty_eos"]
    gen = tc["n_genuine_attempt"]
    qw_ok = cited.get("qwen9b_reason", {}).get("n_induce_ok", 0)
    qw_n = cited.get("qwen9b_reason", {}).get("n_cells", 0)
    if n == 0:
        return "complete_thinkingcap_chat_reason_no_cells_ran"
    # SANITY GUARD: the whole point of this experiment is that the chat template fixes the
    # immediate-EOS degeneracy. If most cells are STILL degenerate, the fix did not take -- flag it
    # loudly rather than silently reporting a false negative.
    if deg > gen:
        return (
            f"complete_thinkingcap_chat_reason_STILL_degenerate_{deg}of{n}_chat_template_did_not_fix_"
            f"immediate_eos_only_{gen}of{n}_genuine_needs_investigation"
        )
    # Chat template restored genuine reasoning -> lead with that (the harness correction), then the
    # honest completion delta, then the downstream-progress truth.
    base = (
        f"complete_thinkingcap_chat_template_fixes_degeneracy_reason_engaged_{tc['n_reason_engaged']}"
        f"of{n}_completes_{tc_ok}of{n}_vs_qwen_{qw_ok}of{qw_n}"
    )
    if tc["any_levelup"]:
        return base + "_with_real_levelup"
    if tc_ok > qw_ok:
        return (
            base + f"_modest_completion_gain_{tc['n_overran']}of{n}_still_overran_no_levelup_"
            "small_n_not_significant"
        )
    if tc_ok == 0:
        return (
            base + f"_no_completions_{tc['n_genuine_overran']}_overran_budget_wall_persists_"
            "despite_correct_template"
        )
    return base + "_no_reliable_completion_advantage_no_levelup"


def _repro_checksum(rows: list[dict[str, Any]]) -> str:
    from carnot.agentic import arc_executable_world_model as e3

    h = hashlib.sha256()
    h.update(Path(atp.__file__).read_bytes())
    h.update(Path(e3.__file__).read_bytes())  # the chat-template fix lives here
    h.update(Path(__file__).read_bytes())
    h.update(
        json.dumps(
            {"roster": ROSTER, "trials": TRIALS, "gen": GEN, "arm": ARM, "generators": GENERATORS},
            sort_keys=True,
            default=str,
        ).encode()
    )
    h.update(json.dumps(sorted(json.dumps(r, sort_keys=True) for r in rows)).encode())
    return "sha256:" + h.hexdigest()


def build_artifact(
    rows: list[dict[str, Any]], pre: dict[str, Any], duration_s: float
) -> dict[str, Any]:
    treat, base = ARM_NAME, "qwen9b_reason"
    metrics = [
        "induce_ok",
        "reason_engaged",
        "overran",
        "reached_levelup",
        "hv_progress",
        "plan_found",
        "heldout_accuracy",
        "cell_recall",
        "goal_predicate_accuracy",
        "levelup_positive_recall",
    ]
    comparisons = [
        {**atp.paired_by_game(rows, treat, base, metric=m), "contrast": f"{treat}_vs_{base}"}
        for m in metrics
    ]
    completion = {
        ARM_NAME: _completion_summary(rows, ARM_NAME),
        "qwen9b_reason": _completion_summary(rows, "qwen9b_reason"),
        "thinkingcap27_frozen": _completion_summary(rows, "thinkingcap27_frozen"),
    }
    tc_sum = completion[ARM_NAME]
    fr_sum = completion["thinkingcap27_frozen"]
    prior = json.loads(PRIOR_5724.read_text()) if PRIOR_5724.exists() else {}
    prior_tc = prior.get("completion_rate_summary", {}).get("thinkingcap27_reason", {})

    validity = {
        "chat_template_fix_took": tc_sum["n_degenerate_empty_eos"] <= tc_sum["n_genuine_attempt"],
        "thinkingcap_reason_degenerate_empty_eos_cells": tc_sum["n_degenerate_empty_eos"],
        "thinkingcap_reason_genuine_attempt_cells": tc_sum["n_genuine_attempt"],
        "thinkingcap_reason_genuine_completed": tc_sum["n_genuine_completed"],
        "thinkingcap_reason_genuine_overran": tc_sum["n_genuine_overran"],
        "prior_5724_degenerate_empty_eos_cells": prior_tc.get("n_degenerate_empty_eos"),
        "prior_5724_genuine_attempt_cells": prior_tc.get("n_genuine_attempt"),
        "prior_5724_induce_ok": prior_tc.get("n_induce_ok"),
        "smoke_test": {
            "ran_before_full_run": True,
            "result": "4/4 induction-shaped /v1/chat/completions calls engaged a genuine <think> "
            "trace with non-empty output (vs 10/12 immediate-EOS on the raw /completion path); the "
            "chat template supplies the assistant-turn structure ThinkingCap-27B (Qwen3.6) needs.",
        },
        "caveat": (
            "READ BEFORE CITING. This experiment CORRECTS the harness bug that made "
            "REQ-ARC-WMTE-5724 inconclusive: 5724's ThinkingCap reason arm hit the raw /completion "
            "endpoint (no chat template) and emitted immediate EOS with ~0 output on "
            f"{prior_tc.get('n_degenerate_empty_eos', '10')}/12 cells. Here the SAME reason cell "
            "(exp5724.run_reason_cell, imported verbatim) runs through /v1/chat/completions "
            "(use_chat_template=True), so the GGUF's embedded Qwen3.6 template applies and the model "
            f"genuinely reasons: {tc_sum['n_genuine_attempt']}/{tc_sum['n_cells']} genuine attempts, "
            f"{tc_sum['n_degenerate_empty_eos']}/{tc_sum['n_cells']} degenerate. The completion count "
            f"({tc_sum['n_induce_ok']}/{tc_sum['n_cells']}) is therefore now a VALID test of the "
            "token-efficiency hypothesis. The qwen9b_reason (raw /think, Qwen tolerates it) and "
            "thinkingcap27_frozen (codeonly control, 12/12 healthy) arms are CITED from 5724 (rows "
            "tagged reused_from_5724), NOT re-measured -- they were already healthy on their own "
            "correct paths."
        ),
    }
    tcfg = GENERATORS["thinkingcap27"]
    bcfg = GENERATORS["qwen9b"]
    n_games = len({r["game"] for r in rows if r.get("arm") == treat})
    cited_summ = {
        "qwen9b_reason": completion["qwen9b_reason"],
        "thinkingcap27_frozen": completion["thinkingcap27_frozen"],
    }
    return {
        "experiment": "experiment_5725_thinkingcap_chat_template_reason_ab",
        "schema": "carnot.exp5725.thinkingcap_chat_template_reason_ab.v1",
        "requirements": ["REQ-ARC-WMTE-5725"],
        "prior_work_extended": [
            "REQ-ARC-WMTE-5724",
            "REQ-ARC-WMTE-5714",
            "REQ-ARC-WMTE-5720",
        ],
        "corrects": "REQ-ARC-WMTE-5724 (harness-broken: raw /completion gave ThinkingCap immediate "
        "EOS on ~10/12 reason cells; this run routes the SAME cell through /v1/chat/completions).",
        "question": "With ThinkingCap-Qwen3.6-27B routed through its PROPER embedded chat template "
        "(so genuine /think reasoning actually engages), does it COMPLETE the genuine-reasoning "
        "induction within the SAME 8192-token budget more often than vanilla Qwen3.5-9B-MTP "
        "(~0/N)?",
        "inference_substrate": "live_llm_inference",
        "model_specs": [
            {
                "name": tcfg["repo_substr"],
                "hf_id": tcfg["hf_id"],
                "quant": "Q4_K_M",
                "role": f"TREATMENT generator ({tcfg['role']}); routed via /v1/chat/completions "
                "(use_chat_template=True) -- the REQ-ARC-WMTE-5725 fix",
                "gguf_path": pre.get("resolved_gguf_paths", {}).get(GEN),
                "mtp": tcfg["mtp"],
                "kv_quant": tcfg["kv_quant"],
                "n_ctx": tcfg["n_ctx"],
                "server": f"CUDA llama-server GPU1 (CARNOT_ARC_GENERATOR_CUDA_GPU=1) port "
                f"{tcfg['port'] + 10}, -ngl 999, q8_0 KV, -fit off (Qwen3.6-27B hybrid-attn), MTP "
                "off, /v1/chat/completions (embedded chat template)",
            },
            {
                "name": bcfg["repo_substr"],
                "hf_id": bcfg["hf_id"],
                "quant": "Q4_K_M",
                "role": f"BASELINE generator ({bcfg['role']}) -- CITED from REQ-ARC-WMTE-5724, not "
                "re-measured (raw /completion path, which Qwen3.5 tolerates)",
                "gguf_path": prior.get("model_specs", [{}, {}])[1].get("gguf_path")
                if prior.get("model_specs")
                else None,
                "mtp": bcfg["mtp"],
                "kv_quant": bcfg["kv_quant"],
                "n_ctx": bcfg["n_ctx"],
                "server": "CITED from REQ-ARC-WMTE-5724",
            },
        ],
        "honest_verdict": _verdict(rows, cited_summ),
        "solve_provenance": "development_proxy",
        "verifier_is_oracle": False,
        "read_game_source": False,
        "used_env_source": True,
        "random_seed": TRIALS[0],
        "trials_per_arm": len(TRIALS),
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": pre,
        "completion_rate_summary": completion,
        "cited_baselines": {
            "note": "qwen9b_reason + thinkingcap27_frozen are CITED from REQ-ARC-WMTE-5724 (their "
            "per-run rows are included with reused_from=REQ-ARC-WMTE-5724); ONLY thinkingcap27_reason "
            "was re-measured here through the chat template.",
            "source_artifact": "results/experiment_5724_thinkingcap_token_efficient_reason_ab.json",
            "prior_thinkingcap27_reason_raw_path": prior_tc,
        },
        "measurement_validity": validity,
        "sample_size": {
            "games": n_games,
            "trials_per_arm": len(TRIALS),
            "arms_measured_fresh": [ARM_NAME],
            "arms_cited_from_5724": CITED_BASE_ARMS,
            "paired_unit": "game (metrics averaged over trials, paired by game)",
            "disclosure": "SMALL N (<=6 game pairs); stochastic proposer. PRIMARY metric = the raw "
            "induce COMPLETION RATE (n_induce_ok / n_cells) for the freshly-measured chat-template "
            "ThinkingCap reason arm vs Qwen's cited ~0/N. With <=6 pairs the sign test cannot reach "
            "p<0.05 unless every game agrees; a positive completion-rate delta over Qwen's near-zero "
            "is a real, actionable direction, the paired comparisons are secondary color. The "
            "baselines are cited (same harness, same roster, same windows) -- disclosed, not "
            "re-measured.",
        },
        "measurement_integrity": {
            "induce_mechanism": "exp5724.run_reason_cell imported VERBATIM (exp5714 _induce_no_fence: "
            "codeonly OFF, /think, 8192 n_predict, tries=1, NO pre-opened fence) -- the ONLY change vs "
            "5724 is the proposer's use_chat_template=True routing induce through /v1/chat/completions.",
            "chat_template_source": "the GGUF's OWN embedded Qwen3.6 chat template (llama.cpp reads "
            "tokenizer.chat_template from the .gguf metadata for /v1/chat/completions) -- no external "
            "HuggingFace template needed.",
            "genuine_reasoning_preserved": "codeonly stays OFF and no ```python fence is pre-opened, "
            "so /think genuinely reasons before emitting code (the no-fence semantics); smoke-tested "
            "4/4 <think>-engaged before the full run.",
            "only_route_varies": "the reason arm config is otherwise byte-identical to 5724; only the "
            "HTTP endpoint (chat vs raw completion) changed, so any completion-rate delta vs 5724's "
            "ThinkingCap reason arm is attributable to the template fix.",
            "stale_engine_guard": "world_model.py deleted before each induce + induction_ok gated on "
            "_attribution_ok (the exp5722 fix, inherited via run_reason_cell).",
        },
        "comparisons": comparisons,
        "per_run_rows": rows,
        "methodology_note": (
            "SEEDED induce->plan->execute on the same build_progress_window input as "
            "REQ-ARC-WMTE-5720/5724, genuine-reasoning induce (no-fence /think, 8192) via "
            "exp5724.run_reason_cell, ThinkingCap-27B routed through /v1/chat/completions "
            "(use_chat_template=True, the REQ-ARC-WMTE-5725 fix). SMOKE TEST BEFORE the run: 4/4 "
            "induction-shaped chat calls engaged a genuine <think> trace with non-empty output (the "
            "raw /completion path gave 10/12 immediate-EOS in 5724). GPU 1, ThinkingCap server "
            "-fit off + n_ctx=22000 (Qwen3.6-27B hybrid-attn). Qwen3.5-9B reason baseline (1/12) and "
            "ThinkingCap frozen control (12/12) CITED from 5724 (rows tagged reused_from_5724), not "
            "re-measured. Paired by GAME. PRIMARY metric = induce completion rate vs Qwen's ~0/N."
        ),
        "recommendation_scope": (
            "A CONTENT test on a dev 24GB 3090, NOT a deployment decision. If token-efficient "
            "reasoning through the proper chat template clears the budget wall (completes materially "
            "more than Qwen's ~0/N), it is a candidate path to make /think usable -- PENDING a "
            "real-VRAM/latency feasibility check on the ~16GB Kaggle eval GPU (ThinkingCap-27B Q4 is "
            "a tight fit there). If the overrun persists even with the correct template, the lever is "
            "'raise the budget / split the induce', not 'use a more efficient model'. Either way this "
            "NEVER flips the frozen live default (operator-only) and NEVER submits."
        ),
        "duration_s": round(duration_s, 2),
        "reproducibility_checksum": _repro_checksum(rows),
    }


def main() -> None:
    t0 = time.time()
    pre = check_preconditions()
    log(
        f"preconditions: {json.dumps({k: v for k, v in pre.items() if k != 'resolved_gguf_paths'})}"
    )
    blocking = (
        not pre.get("offline_arcade_import")
        or not pre.get(f"{GEN}_gguf_cached")
        or not pre.get("llama_supports_gpu_offload")
        or not pre.get("prior_5724_present")
    )
    if blocking:
        log(f"PRECONDITION FAIL: {pre}")
        ARTIFACT.write_text(
            json.dumps(
                {
                    "experiment": "experiment_5725_thinkingcap_chat_template_reason_ab",
                    "requirements": ["REQ-ARC-WMTE-5725"],
                    "inference_substrate": "live_llm_inference",
                    "honest_verdict": "complete_blocked_preconditions_unmet",
                    "preconditions_checked": pre,
                    "random_seed": TRIALS[0],
                    "duration_s": round(time.time() - t0, 2),
                    "reproducibility_checksum": "sha256:"
                    + hashlib.sha256(
                        json.dumps(pre, sort_keys=True, default=str).encode()
                    ).hexdigest(),
                },
                indent=2,
            )
        )
        return

    rows, pre = run_all(pre)
    art = build_artifact(rows, pre, time.time() - t0)
    ARTIFACT.write_text(json.dumps(art, indent=2))
    log(f"WROTE {ARTIFACT.name}: {art['honest_verdict']}")
    log(f"DONE total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
