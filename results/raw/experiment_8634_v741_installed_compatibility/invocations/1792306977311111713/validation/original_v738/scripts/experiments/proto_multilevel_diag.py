"""MULTI-LEVEL DEEPENING DIAGNOSTIC: locate WHERE the live agent's L1->L2 transition stalls.

Context (.428 B1): live first-win rate ~0.59 but live_multi_level_solve_rate ~0.04 -- the agent
reaches the FIRST level-up on a fresh game but rarely deepens to L2/L3. Every existing harness
(exp4628) used target_levels=1 AND broke at the first level-up, so NOTHING has ever instrumented
the L1->L2 transition in the live agent. This does.

Method: instantiate the LIVE SUBMITTED-config E3AgentPolicy (real CNN action-effect expansion prior +
value head + candidate router = the .427 bridge-crossing config that actually runs on Kaggle) but with
target_levels=5 and NO early break, on games where L2+ is KNOWN reachable (we reproduce them offline).
Roll out to a generous budget; record the action index at EACH level-up. This separates two hypotheses:
  (H1) real capability wall -- exploration+energy guidance simply cannot find the 2nd win, OR
  (H2) no-headroom measurement artifact -- the L2 transition needs the LLM proposer (goal induction)
       that the matched-offline noop arm disables, so a noop run stalling at L1 is EXPECTED not a wall.

Arm: NoOpProposer (exploration + CNN prior + value head, NO LLM induction). If exploration ALONE
deepens on some games -> H1 is wrong for those (exploration suffices). If it universally stalls at L1
-> the 2nd-win generation needs goal-induction (the proposer), pointing the lever at the proposer not
the explorer. CPU-forced (the value-head/CNN are small; no LLM in the noop arm).
"""

from __future__ import annotations

import json
import os
import sys
import time

# ARM selection. noop = exploration + CNN expansion prior + value head only (CPU, no LLM induction).
# real = the LIVE submitted agent with the real LocalGGUFProposer (LLM goal/rule induction); needs a GPU.
_ARM = os.environ.get("MULTILEVEL_ARM", "noop")
if _ARM != "real":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from arcengine import GameAction  # noqa: E402
from carnot.agentic import arc_solver_kit as kit  # noqa: E402
from carnot.agentic.arc_competition_agent import E3AgentPolicy, _level_of  # noqa: E402
from carnot.experiment_4628_dense_curiosity_progress_loop import _NoOpProposer  # noqa: E402


def _gid(arc, short):
    for e in arc.get_environments():
        g = getattr(e, "game_id", "")
        if g.split("-")[0] == short:
            return str(g)
    raise RuntimeError(f"{short} unavailable")


# Clean-port Qwen proposer: the live _proposer() hardcodes port 8919, which a persistent gemma-4-12B
# server squats on this box -> our first real-arm run silently reused GEMMA, not Qwen (the confound).
# Construct the proposer EXPLICITLY with the SAME live config but a free port so it spawns a fresh Qwen
# server. MULTILEVEL_PORT overrides the port; MULTILEVEL_QWEN_GGUF overrides the model path.
_QWEN_GGUF = os.environ.get(
    "MULTILEVEL_QWEN_GGUF",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-MTP-GGUF/snapshots/"
        "9716a636ee4bddc3fed678220b7a33dd2a4160ae/Qwen3.5-9B-Q4_K_M.gguf"
    ),
)
_PORT = int(os.environ.get("MULTILEVEL_PORT", "8920"))


def _clean_qwen_proposer():
    """Real Qwen3.5-9B-MTP proposer on a FREE port (matches arc_competition_agent._proposer config)."""
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    return LocalGGUFProposer(
        repo_substr="Qwen3.5-9B-MTP",
        model_path=_QWEN_GGUF if os.path.exists(_QWEN_GGUF) else None,
        port=_PORT,
        # mtp is DELIBERATELY NOT PASSED. This line used to read
        # `mtp=(os.environ.get("CARNOT_ARC_MTP", "1") != "0")` -- a literal "1" that is NOT the
        # project's canonical local default (`ARC_LIVE_GENERATOR_MTP_DEFAULT` is "0"). With
        # CARNOT_ARC_MTP unset that handed the proposer mtp=True, which at the shipped n_ctx 81920
        # needs ~14 offloaded FFN layers on a 24 GB card -- past the auto-fit cap, so the VRAM guard
        # declines CUDA, the generator falls back to the ~2 tok/s iGPU, every induce times out, and
        # the run proceeds LLM-OFF while still reporting itself LLM-on. Omitting the argument lets
        # `LocalGGUFProposer.mtp`'s own default factory (`_mtp_default_on()`) answer, which reads
        # the SAME env var against the canonical constant -- identical override behaviour, correct
        # default, and one place to change it.
        kv_quant="q8_0",
        no_think_prefix="/no_think\n",
        max_tokens=2560,
        n_gpu_layers=int(os.environ.get("CARNOT_ARC_NGL", "999")),
    )


def _slim_attempts(attempts):
    """Keep the diagnostic fields from each induction attempt; drop bulky round/counterexample blobs to
    a count + the first counterexample kind (enough to see WHY without exploding the artifact)."""
    out = []
    for a in attempts:
        cxs = a.get("counterexamples") or []
        # per-round held-out accuracy: is the induced model CLOSE to the gate (gate-fixable) or garbage?
        rounds = a.get("refinement_rounds") or []
        round_summ = [
            {
                "round": r.get("round"),
                "action": r.get("action"),
                "proposer_ok": r.get("proposer_ok"),
                "heldout_accuracy": r.get("heldout_accuracy"),
                "heldout_threshold": r.get("heldout_threshold"),
                "accepted": r.get("accepted_by_heldout_verifier"),
                "plan_length": r.get("plan_length"),
                "plan_reaches_goal": r.get("plan_reaches_goal"),
                "skipped": r.get("skipped"),
                "cx_kind": (r.get("counterexample") or {}).get("kind"),
            }
            for r in rounds
            if isinstance(r, dict)
        ]
        out.append(
            {
                "reason": a.get("reason"),
                "goal_level": a.get("goal_level"),
                "skipped": a.get("skipped")
                or ("" if a.get("planned") else "planned_false_no_skip"),
                "planned": bool(a.get("planned")),
                "plan_length": a.get("plan_length"),
                "heldout_accuracy": a.get("heldout_accuracy") or a.get("verify_accuracy"),
                "verify_cell_recall": a.get("verify_cell_recall"),
                "refinement_rounds_used": a.get("refinement_rounds_used"),
                "n_goal_candidates": len(a.get("goal_candidate_names") or []),
                "n_dynamics_candidates": len(a.get("dynamics_candidate_names") or []),
                "n_counterexamples": len(cxs),
                "first_counterexample_kind": (
                    cxs[0].get("kind") if cxs and isinstance(cxs[0], dict) else None
                ),
                "transition_count": a.get("transition_count"),
                "rounds": round_summ,
            }
        )
    return out


def _skip_histogram(attempts):
    hist = {}
    for a in attempts:
        key = a.get("skipped") or ("planned_ok" if a.get("planned") else "planned_false_no_skip")
        hist[key] = hist.get(key, 0) + 1
    return hist


def diagnose(arc, short, budget, target_levels=5):
    """Roll the LIVE submitted-config policy to budget; record per-level-up action idx + induction decomp."""
    gid = _gid(arc, short)
    env = arc.make(gid, scorecard_id=arc.open_scorecard())
    # SUBMITTED defaults EXCEPT: target_levels raised so it won't stop at L1.
    # noop arm -> NoOpProposer (exploration-only); real arm -> clean Qwen proposer on a free port.
    proposer = _clean_qwen_proposer() if _ARM == "real" else _NoOpProposer()
    policy = E3AgentPolicy(gid, proposer=proposer, target_levels=int(target_levels))
    frames: list = []
    latest = None
    start_level = None
    reached = 0
    actions = 0
    levelup_at = {}  # level -> action index where first reached
    t0 = time.time()
    for _ in range(int(budget)):
        if policy.is_done(frames, latest):
            break
        kind, data = policy.next_move(frames, latest)
        if kind == "RESET":
            latest = env.reset()
        elif kind is None:
            break
        else:
            latest = env.step(getattr(GameAction, f"ACTION{kind}"), data=data)
            actions += 1
        if latest is None:
            break
        lvl = _level_of(latest)
        if start_level is None:
            start_level = lvl
        rel = lvl - (start_level or 0)
        if rel > reached:
            reached = rel
            levelup_at.setdefault(rel, actions)
        frames.append(latest)
    return {
        "game": short,
        "max_rel_level": int(reached),
        "levelup_at_action": {str(k): v for k, v in sorted(levelup_at.items())},
        "actions_used": actions,
        "budget": int(budget),
        "exhausted_budget": actions >= int(budget) - 1,
        "explorer_explored_out": bool(getattr(policy.explorer, "explored_out", False)),
        "state_coverage": int(len(getattr(policy.explorer, "graph", {}) or {})),
        # induction diagnostics: did the explore->induce->execute cascade actually fire?
        "induction_fired": bool(getattr(policy, "induced", False)),
        "final_phase": str(getattr(policy, "phase", "?")),
        "transitions_collected": int(len(getattr(policy, "transitions", []) or [])),
        "plan_len": int(len(getattr(policy, "plan", []) or [])),
        # FULL induction decomposition -- WHY each induction produced (or failed to produce) a plan.
        # Each attempt records: reason, skipped(=missing_root/no_transitions/proposer_failed/
        # heldout_transition_verification_failed/no_reachable_plan_after_refinement/exception), planned,
        # plan_length, heldout_accuracy, goal/dynamics_candidate_names, refinement_rounds_used, counterexamples.
        "n_induction_attempts": int(len(getattr(policy, "induction_attempts", []) or [])),
        "induction_attempts": _slim_attempts(getattr(policy, "induction_attempts", []) or []),
        "induction_skip_reasons": _skip_histogram(getattr(policy, "induction_attempts", []) or []),
        "stalled_at_L1": reached == 1,
        "reached_L2_plus": reached >= 2,
        "wall_s": round(time.time() - t0, 1),
    }


def main() -> int:
    # Multi-level-reachable games (repro_levels>=2 per registry): exploration-only deepening candidates.
    games = (
        sys.argv[1].split(",")
        if len(sys.argv) > 1
        else ["vc33", "sc25", "tn36", "cd82", "sp80", "lp85", "su15", "tu93", "m0r0"]
    )
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    arc = kit.offline_arcade()
    out = {"budget": budget, "arm": _ARM, "games": {}}
    _outfile = f"results/proto_multilevel_diag_{_ARM}.json"
    print(
        f"== multi-level deepening diag: LIVE E3 (arm={_ARM}, target_levels=5), budget={budget} ==",
        flush=True,
    )
    for short in games:
        try:
            r = diagnose(arc, short, budget)
        except Exception as e:
            out["games"][short] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  {short:6} ERROR {type(e).__name__}: {e}", flush=True)
            continue
        out["games"][short] = r
        print(
            f"  {short:6} maxL={r['max_rel_level']} at={r['levelup_at_action']} "
            f"acts={r['actions_used']}/{budget} n_induce={r.get('n_induction_attempts')} "
            f"skips={r.get('induction_skip_reasons')} plan={r['plan_len']} "
            f"({r['wall_s']}s)",
            flush=True,
        )
    reached2 = sorted(g for g, v in out["games"].items() if v.get("reached_L2_plus"))
    stalled = sorted(g for g, v in out["games"].items() if v.get("stalled_at_L1"))
    out["reached_L2_plus"] = reached2
    out["stalled_at_L1"] = stalled
    out["VERDICT"] = (
        "EXPLORATION_ALONE_DEEPENS_on_" + ",".join(reached2)
        if reached2
        else "EXPLORATION_NEVER_DEEPENS_2nd_win_needs_goal_induction_proposer"
    )
    json.dump(out, open(_outfile, "w"), indent=2)
    print(
        f"\nreached_L2+={reached2}  stalled_at_L1={stalled}\nVERDICT={out['VERDICT']}\n-> {_outfile}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
