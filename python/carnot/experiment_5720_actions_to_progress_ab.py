"""Exp 5720/5721: re-test the MTP+reasoning and playbook-retrieval A/Bs with a REAL
actions-to-progress metric (REQ-ARC-WMTE-5720, REQ-ARC-WMTE-5721).

Both prior A/Bs -- REQ-ARC-WMTE-5714 (`/think` vs `/no_think`) and
REQ-ARC-WMTE-5716/5717/5718 (playbook exemplars, exp5719) -- were inconclusive on a
single-shot induction-quality proxy that was floored + high-variance and did not
measure real progress. This drives the ACTUAL scored `E3AgentPolicy` cascade on the
offline arcade for a bounded solve, per arm, and measures whether the capability change
helps the agent make REAL PROGRESS: a level-up (`frame.levels_completed`), a dense
`hand_verifier` goal-distance reduction, and induction quality aggregated over the run's
multiple induction attempts. See python/carnot/agentic/arc_actions_to_progress.py.

RESUMABLE: every (game, seed, arm) run is appended to a JSONL shard as it completes, so
an interrupted run resumes without redoing finished cells (the single-synchronous-resume-
accumulate pattern, not a split build/collect). Re-running skips cells already in the shard.

Substrate: live_llm_inference (the real Qwen3.5-9B-MTP GGUF induces on a CUDA llama-server).
Provenance: development_proxy on PUBLIC games -- NOT a hidden-game self-discovery solve; we
NEVER flip the frozen live default and NEVER submit. The dense progress proxy reads the live
runtime game object via the adapter's public hand_verifier (used_env_source=True), never a
game's .py source (read_game_source=False). The win oracle is the level counter, never a
heuristic (verifier_is_oracle=False).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from carnot.agentic import arc_actions_to_progress as atp  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SHARD = REPO / "results" / "exp5720_atp_shard.jsonl"
REASON_ARTIFACT = REPO / "results" / "experiment_5720_actions_to_progress_reason_ab.json"
RETRIEVAL_ARTIFACT = REPO / "results" / "experiment_5721_actions_to_progress_retrieval_ab.json"

# Roster: adaptered public games SELECTED by a fast induction-disabled probe
# (scratchpad/select_roster.py) to (a) reliably reach the stall->induce phase within
# budget and (b) have a NON-DEGENERATE hand_verifier that the agent can actually REDUCE
# (best_hv < start_hv) -- the precondition for the dense progress proxy to discriminate.
# Excluded games whose hand_verifier was flat/degenerate (cn04, sp80, sk48, ka59) or only
# increased (ar25, dc22, cd82). lp85 additionally solves L0->L1 (a live level-up signal).
ROSTER = ["ls20", "tr87", "lp85", "g50t", "m0r0", "ft09"]
SEEDS = [5720, 5721]
# frozen doubles as the "none" control for the retrieval question (byte-identical config).
ARMS = ["frozen", "reason", "retrieval", "static"]

# GPU split (launch with CARNOT_ARC_GENERATOR_CUDA_GPU=1 and BOTH GPUs visible): the induction
# llama-server runs on GPU1 (port 8933), the retrieval arm's in-process embedding GGUF loads on
# the free GPU0 -- separate cards so the second ~6GB load never contends/OOMs against the server
# (co-locating both on one card silently no-oped retrieval to mode="retrieval_unavailable").
TRIALS = [0, 1]  # LLM sampling is stochastic; trials are per-game replicates for variance
CUDA_PORT = 8933

TERMINAL = ("complete:", "complete_", "success:", "success_", "passed:", "passed_", "shipped:", "shipped_")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _load_shard() -> dict[tuple[str, int, str], dict[str, Any]]:
    rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    if SHARD.exists():
        for line in SHARD.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[(r["game"], int(r["trial"]), r["arm"])] = r
    return rows


def _append_shard(row: dict[str, Any]) -> None:
    SHARD.parent.mkdir(parents=True, exist_ok=True)
    with SHARD.open("a") as f:
        f.write(json.dumps(row) + "\n")


def check_preconditions() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        checks["offline_arcade_import"] = True
    except Exception as exc:
        checks["offline_arcade_import"] = False
        checks["offline_arcade_error"] = repr(exc)[:200]
    gguf = list((Path.home() / ".cache/huggingface/hub").glob(
        "models--unsloth--Qwen3.5-9B-MTP-GGUF/snapshots/*/Qwen3.5-9B-Q4_K_M.gguf"))
    checks["qwen3.5_9b_mtp_gguf_cached"] = bool(gguf)
    try:
        from carnot.agentic.arc_playbook_retrieval import load_index

        load_index()
        checks["playbook_index_built"] = True
    except Exception as exc:
        checks["playbook_index_built"] = False
        checks["playbook_index_error"] = repr(exc)[:200]
    return checks


def run_all() -> list[dict[str, Any]]:
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    # One-shot induce (no bounded-refinement loop): keeps each run bounded + FAIR across arms
    # (every arm gets exactly one induce attempt, so the slow /think arm is not truncated
    # mid-refinement), and makes the induced-model quality directly single-shot-comparable to
    # the prior REQ-ARC-WMTE-5714/5719 experiments. The induce still routes through the live
    # _induce_and_plan path (codeonly fence / think / retrieval wiring all active).
    os.environ["CARNOT_ARC_STALL_REFACTOR_LOOP"] = "0"

    prop = LocalGGUFProposer(
        repo_substr="Qwen3.5-9B-MTP", port=CUDA_PORT, mtp=True, kv_quant="q8_0",
        no_think_prefix="/no_think\n", max_tokens=4096, timeout=600,
    )
    done = _load_shard()
    total = len(ROSTER) * len(TRIALS) * len(ARMS)
    log(f"resume: {len(done)}/{total} cells already in shard")
    windows: dict[str, Any] = {}
    for game in ROSTER:
        w = windows.get(game)
        if w is None:
            w = atp.build_progress_window(game)
            windows[game] = w
            if w is None:
                log(f"SKIP {game}: no offline L1 window (build_progress_window returned None)")
                continue
        window, full_traj, cell = w
        for trial in TRIALS:
            for arm in ARMS:
                key = (game, trial, arm)
                if key in done:
                    continue
                log(f"RUN {game} trial={trial} arm={arm}")
                t0 = time.time()
                res = atp.run_seeded_progress(
                    game, arm, proposer=prop, trial=trial,
                    window=window, full_traj=full_traj, cell=cell,
                )
                row = res.to_row()
                _append_shard(row)
                done[key] = row
                log(f"  -> ind_ok={row['induction_ok']} plan={row['plan_found']} "
                    f"levelup={row['reached_levelup']} hv={row['hv_progress']} "
                    f"heldout={row['heldout_accuracy']} cellrec={row['cell_recall']} "
                    f"goalpred={row['goal_predicate_accuracy']} lurec={row['levelup_positive_recall']} "
                    f"MODE={row['playbook_injection_mode']} wall={row['wall_s']}s ({time.time()-t0:.0f}s)")
    return list(done.values())


# Primary progress signals (decisive but sparse) first, then the denser induction-quality
# discriminators the metric falls back to when plans/level-ups are floored.
PROGRESS_METRICS = ["reached_levelup", "hv_progress", "plan_found"]
INDUCTION_METRICS = ["heldout_accuracy", "cell_recall", "goal_predicate_accuracy",
                     "levelup_positive_recall"]
ALL_METRICS = PROGRESS_METRICS + INDUCTION_METRICS


def _repro_checksum(rows: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    h.update(Path(atp.__file__).read_bytes())
    h.update(json.dumps({"roster": ROSTER, "trials": TRIALS, "arms": ARMS,
                         "stall_refactor_loop": 0}, sort_keys=True).encode())
    h.update(json.dumps(sorted([json.dumps(r, sort_keys=True) for r in rows])).encode())
    return "sha256:" + h.hexdigest()


FIELD_PRINCIPLES = {
    "honest_verdict": "terminal-prefixed self-declared state (Verdict Terminal-Prefix Discipline).",
    "inference_substrate": "live_llm_inference -- the real Qwen3.5-9B-MTP GGUF induces via the live "
                           "E3AgentPolicy._induce_and_plan path every arm; 60s floor applies.",
    "random_seed": "the LLM sampling is stochastic (disclosed); trials are per-game replicates -- "
                   "why we pair by GAME (average over trials) and report win/tie/loss + outlier-fragility.",
    "reproducibility_checksum": "content hash over harness code + config + all rows catches drift.",
    "solve_provenance": "development_proxy -- PUBLIC-game offline dev measurement of the LIVE induce->"
                        "plan->execute mechanism, NOT a hidden-game self-discovery solve.",
    "verifier_is_oracle": "False -- the win oracle is the level counter (frame.levels_completed); "
                          "hand_verifier is only a dense progress MEASUREMENT, oracle-distinct.",
    "preconditions_checked": "GGUF + offline arcade + playbook index verified before any inference.",
    "metric_ladder": "PRIMARY = reached_levelup / hv_progress / plan_found (real downstream progress "
                     "from the induced model's OWN plan executed in the real env). When those are "
                     "floored (the frozen 9B's dynamics induction is usually too weak to plan a "
                     "level-up), the denser induction-quality discriminators (heldout_accuracy, "
                     "cell_recall, goal_predicate_accuracy, levelup_positive_recall) -- the SAME "
                     "signals the prior single-shot experiments used, now on the live path + averaged "
                     "over trials -- carry the comparison. An all-floored result is a valid, terminal "
                     "finding (the reasoning/retrieval knob is not the bottleneck).",
}


def _comparisons_for(rows: list[dict[str, Any]], treat: str, base: str,
                     contrast: str) -> list[dict[str, Any]]:
    return [{**atp.paired_by_game(rows, treat, base, metric=m), "contrast": contrast}
            for m in ALL_METRICS]


def build_artifact(*, exp_id: str, req: str, question: str, comparisons: list[dict[str, Any]],
                   rows: list[dict[str, Any]], arms_used: list[str], verdict: str,
                   preconditions: dict[str, Any], duration_s: float) -> dict[str, Any]:
    used = [r for r in rows if r["arm"] in arms_used]
    n_games = len({r["game"] for r in used})
    n_pairs_max = max((c.get("n_game_pairs", 0) for c in comparisons), default=0)
    # honesty stamp: did the retrieval arm actually inject, and did any plan/level-up ever form?
    retr_modes = sorted({r.get("playbook_injection_mode") for r in used
                         if r.get("arm") == "retrieval"})
    any_plan = any(r.get("plan_found") for r in used)
    any_levelup = any(r.get("reached_levelup") for r in used)
    return {
        "experiment": exp_id,
        "schema": f"carnot.{exp_id}.actions_to_progress_ab.v1",
        "requirements": [req, "REQ-ARC-WMTE-5720"],
        "prior_work_extended": ["REQ-ARC-WMTE-5714", "REQ-ARC-WMTE-5716", "REQ-ARC-WMTE-5717",
                                "REQ-ARC-WMTE-5718"],
        "question": question,
        "inference_substrate": "live_llm_inference",
        "model_specs": [{
            "name": "Qwen3.5-9B-MTP", "hf_id": "unsloth/Qwen3.5-9B-MTP-GGUF", "quant": "Q4_K_M",
            "role": "E3AgentPolicy world-model induction proposer (the frozen live generator)",
            "server": f"CUDA llama-server GPU0 port {CUDA_PORT}, -ngl 999, q8_0 KV, MTP self-draft on",
            "mtp_note": "MTP (draft-mtp speculative decoding) is content-neutral (speed only); the "
                        "induction-content arms differ in the request (codeonly fence / think / "
                        "retrieval / n_predict), not the MTP flag.",
            "gpu_substrate_note": "run on ONE discrete RTX 3090 (GPU0) for dev throughput -- server + "
                                  "retrieval embedder + CNN engine co-located to avoid cross-card OOM. "
                                  "Induction CONTENT is identical to the iGPU frozen live stack "
                                  "(same model/quant/prompt); wall-clock is NOT Kaggle-representative.",
        }],
        "honest_verdict": verdict,
        "solve_provenance": "development_proxy",
        "verifier_is_oracle": False,
        "read_game_source": False,
        "used_env_source": True,
        "random_seed": TRIALS[0],
        "trials_per_arm": len(TRIALS),
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": preconditions,
        "sample_size": {
            "games": n_games, "trials_per_arm": len(TRIALS), "arms": arms_used,
            "paired_unit": "game (metrics averaged over trials, paired by game)",
            "max_game_pairs": n_pairs_max,
            "disclosure": "SMALL N (<=6 game pairs); stochastic proposer. Report paired game "
                          "win/tie/loss + exact sign-test p + outlier-fragility, NOT a bare mean. "
                          "With <=6 pairs the sign test cannot reach p<0.05 unless every game agrees.",
        },
        "measurement_integrity": {
            "retrieval_injection_modes_observed": retr_modes,
            "retrieval_actually_fired": any(m == "retrieval" for m in retr_modes),
            "any_plan_found_across_all_runs": any_plan,
            "any_real_levelup_across_all_runs": any_levelup,
            "note": "If retrieval_actually_fired is False, the retrieval arm silently no-oped "
                    "(embedder unavailable) and its contrast is NOT a valid test. If "
                    "any_plan_found is False, the PRIMARY progress metrics are fully floored and "
                    "only the induction-quality ladder carries signal -- disclosed, not hidden.",
        },
        "comparisons": comparisons,
        "per_run_rows": used,
        "methodology_note": (
            "SEEDED induce->plan->execute: seed a real level-up-straddling transition window (reuses "
            "exp5717.build_window, the SAME input the prior A/Bs used), then DIRECT-induce with EXPLICIT "
            "per-arm injection (the exp5719 mechanism -- codeonly/think via env+proposer, retrieval via "
            "the live _retrieve_playbook_block, then proposer.induce() directly) so induction ALWAYS "
            "fires and the arm's exact prompt is guaranteed applied (the live _induce_and_plan auto-"
            "arming was unreliable: its TTT-CNN tier short-circuits the LLM fall-through where retrieval "
            "is armed, and the full-explore driver did not reliably reach a stall -- e.g. ls20 did 0 "
            "inductions). Then use the LIVE plan_in_model planner + execute the plan against a fresh "
            "real offline env, checking for a REAL level-up (frame.levels_completed advance) + tracking "
            "the dense hand_verifier goal-distance. Paired by GAME (metrics averaged over trials). ONLY "
            "the downstream metric changed vs the prior single-shot work; the induction input is identical."
        ),
        "duration_s": round(duration_s, 2),
        "reproducibility_checksum": _repro_checksum(rows),
    }


def _verdict(prefix: str, comps: list[dict[str, Any]]) -> str:
    """Terminal-prefixed verdict. Prefer the primary progress metric (reached_levelup); if it is
    fully floored (no pairs / all zero), fall back to the densest induction discriminator
    (heldout_accuracy) and say so, so the verdict never over-claims a progress signal that isn't there."""
    def pick(metric):
        return next((c for c in comps if c.get("metric") == metric and c.get("n_game_pairs")), None)

    primary = pick("reached_levelup")
    floored = primary is None or (primary.get("mean_treat") == 0 and primary.get("mean_base") == 0)
    driver = pick("heldout_accuracy") if floored else primary
    if driver is None:
        return f"complete_{prefix}_no_comparable_pairs_metric_unavailable"
    tag = "heldout_only_progress_floored" if floored else "reached_levelup"
    p, md, frag = driver.get("sign_test_p"), driver.get("mean_delta", 0.0), driver.get("outlier_fragile")
    if p is not None and p < 0.05 and not frag and abs(md) > 1e-9:
        direction = "helps" if md > 0 else "hurts"
        return f"complete_{prefix}_{tag}_{direction}_signif_delta_{md}"
    return (f"complete_{prefix}_{tag}_no_reliable_signal_delta_{md}_signp_{p}_fragile_{bool(frag)}")


def main() -> None:
    t0 = time.time()
    pre = check_preconditions()
    if not pre.get("offline_arcade_import") or not pre.get("qwen3.5_9b_mtp_gguf_cached"):
        log(f"PRECONDITION FAIL: {pre}")
        for path, exp, req in ((REASON_ARTIFACT, "experiment_5720_actions_to_progress_reason_ab",
                                "REQ-ARC-WMTE-5720"),
                               (RETRIEVAL_ARTIFACT, "experiment_5721_actions_to_progress_retrieval_ab",
                                "REQ-ARC-WMTE-5721")):
            path.write_text(json.dumps({
                "experiment": exp, "requirements": [req], "inference_substrate": "live_llm_inference",
                "honest_verdict": "complete_blocked_preconditions_unmet", "preconditions_checked": pre,
                "random_seed": TRIALS[0], "duration_s": round(time.time() - t0, 2),
                "reproducibility_checksum": "sha256:" + hashlib.sha256(
                    json.dumps(pre, sort_keys=True).encode()).hexdigest(),
            }, indent=2))
        return

    rows = run_all()

    # Reason question: reason vs frozen.
    reason_comps = _comparisons_for(rows, "reason", "frozen", "reason_vs_frozen")
    reason_art = build_artifact(
        exp_id="experiment_5720_actions_to_progress_reason_ab", req="REQ-ARC-WMTE-5720",
        question="Does removing the codeonly fence + genuine /think reasoning help actions-to-progress "
                 "vs the frozen codeonly+/no_think default? (re-test of REQ-ARC-WMTE-5714)",
        comparisons=reason_comps, rows=rows, arms_used=["frozen", "reason"],
        verdict=_verdict("think_vs_frozen", reason_comps), preconditions=pre,
        duration_s=time.time() - t0)
    REASON_ARTIFACT.write_text(json.dumps(reason_art, indent=2))
    log(f"WROTE {REASON_ARTIFACT.name}: {reason_art['honest_verdict']}")

    # Retrieval question: retrieval vs none(frozen); static vs none; retrieval vs static.
    retr_comps = (_comparisons_for(rows, "retrieval", "frozen", "retrieval_vs_none")
                  + _comparisons_for(rows, "static", "frozen", "static_vs_none")
                  + _comparisons_for(rows, "retrieval", "static", "retrieval_vs_static"))
    hv_retr_none = [c for c in retr_comps
                    if c.get("contrast") == "retrieval_vs_none" and c.get("n_game_pairs")]
    retr_art = build_artifact(
        exp_id="experiment_5721_actions_to_progress_retrieval_ab", req="REQ-ARC-WMTE-5721",
        question="Does playbook-exemplar retrieval injection help actions-to-progress vs no injection "
                 "(and vs static injection)? (re-test of REQ-ARC-WMTE-5716/5717/5718 / exp5719)",
        comparisons=retr_comps, rows=rows, arms_used=["frozen", "retrieval", "static"],
        verdict=_verdict("retrieval_vs_none", hv_retr_none), preconditions=pre,
        duration_s=time.time() - t0)
    RETRIEVAL_ARTIFACT.write_text(json.dumps(retr_art, indent=2))
    log(f"WROTE {RETRIEVAL_ARTIFACT.name}: {retr_art['honest_verdict']}")
    log(f"DONE total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
