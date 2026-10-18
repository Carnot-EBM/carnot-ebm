"""MEASURE: does the combined fix stack (default-ON truncation + goal-repair loop) actually bank an
lp85 L2 -- reproduction-gated?

Background: two fixes shipped 2026-06-25:
  (1) default-ON code-only truncation fix -> the L2 induction now EMITS valid code (vs truncating to
      0 code at 450s);
  (2) goal-repair loop -> a degenerate induced is_level_complete (return False / unreachable) is
      replaced by the exemplar-derived nonzero-count fallback so the planner gets a satisfiable goal.

Before these, the live lp85 binary arm was: maxL=1, n_induce=2, goal_satisfiable=False (degenerate
goal). This probe re-runs the SAME single arm (lp85, binary, budget 400) with both fixes active and
asks the decisive question: does the repair FIRE (goal_predicate_satisfiable flips True), and does
lp85 actually reach a REPRODUCED L2 (the offline reproduction gate -- the only real-level signal)?

Honest prediction (to be confirmed/refuted): the fixes unblock PLANNING (goal_satisfiable becomes
True via repair) but lp85 L2 still does NOT reproduce, because the loose nonzero-count fallback is
not lp85's real win (marker_pair_shape_alignment). That would prove the remaining wall is goal
QUALITY, pointing at the perception-grounded structural-alignment goal.

Single game, single arm (NOT the 4-arm A/B that hung on sc25). Warm Qwen :8920 reused.
Integrity: inference_substrate=live_llm_inference; solve_provenance=development_proxy (offline dev
twin); verifier_is_oracle=false.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Ensure we are testing the DEFAULT path (no opt-out); default-ON is shipped, so unset is correct.
os.environ.pop("CARNOT_ARC_CODEONLY_INDUCE", None)

PORT = 8920
URL = f"http://127.0.0.1:{PORT}"
QWEN = os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-MTP-GGUF/snapshots/"
    "9716a636ee4bddc3fed678220b7a33dd2a4160ae/Qwen3.5-9B-Q4_K_M.gguf"
)
RESULT = REPO / "results" / "proto_l2_measure_fixstack.json"
GAME = "lp85"
BUDGET = int(os.environ.get("MEASURE_BUDGET", "400"))


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def preconditions() -> dict:
    pc = {"qwen_gguf": os.path.isfile(QWEN)}
    try:
        with urllib.request.urlopen(URL + "/health", timeout=4) as r:
            pc["server_health"] = json.load(r).get("status") == "ok"
        with urllib.request.urlopen(URL + "/props", timeout=5) as r:
            mp = json.load(r).get("model_path", "")
        pc["serving_qwen"] = "qwen" in mp.lower()
    except Exception as ex:
        pc["server_health"] = False
        pc["err"] = str(ex)[:120]
    return pc


def main() -> int:
    t0 = time.time()
    pc = preconditions()
    log(f"PRECONDITIONS: {pc}")
    if not (pc.get("qwen_gguf") and pc.get("server_health") and pc.get("serving_qwen")):
        RESULT.write_text(json.dumps({
            "experiment_id": "proto_l2_measure_fixstack",
            "honest_verdict": "blocked_preconditions",
            "preconditions_checked": pc,
            "inference_substrate": "live_llm_inference",
            "solve_provenance": "development_proxy",
            "verifier_is_oracle": False,
            "duration_s": round(time.time() - t0, 2),
        }, indent=2))
        log("BLOCKED: preconditions not met")
        return 0

    import carnot.agentic.arc_solver_kit as kit  # noqa: E402
    from scripts.experiments.proto_graded_goal_bias_ab import _run_arm  # noqa: E402

    arc = kit.offline_arcade()
    log(f"Running {GAME} binary arm (default fix stack: truncation default-ON + goal-repair), budget={BUDGET}...")
    arm = _run_arm(arc, GAME, graded=False, port=PORT, budget=BUDGET)

    max_depth = int(arm.get("max_depth_reached", 0))
    l2_repro = bool(arm.get("l2_offline_reproduced", False))
    gps_any = bool(arm.get("goal_predicate_satisfiable_any", False))
    n_induce = int(arm.get("n_induction_attempts", 0))
    exemplar_any = bool(arm.get("exemplar_present_on_any_induction", False))
    # did the goal-repair fire? infer from induction_summary (goal_repaired surfaced per round if present)
    induction_summary = arm.get("induction_summary", []) or []
    repaired_seen = any(
        (att.get("goal_repaired") or "exemplar_nonzero_count_fallback" in str(att))
        for att in induction_summary
    )

    # also read the final induced world model (to show the is_level_complete the model wrote)
    wm_path = REPO / "results" / "arc_e3" / GAME / "world_model.py"
    wm_tail = ""
    if wm_path.exists():
        txt = wm_path.read_text()
        idx = txt.find("def is_level_complete")
        wm_tail = txt[idx:idx + 400] if idx >= 0 else txt[-400:]

    # verdict
    if l2_repro:
        verdict = "success_lp85_L2_reproduced_with_fix_stack"
        summary = (f"lp85 reached a REPRODUCED L2 (max_depth={max_depth}) with the fix stack. "
                   "The truncation+goal-repair fixes DID bank a real level. (Refutes the prediction.)")
    elif max_depth >= 2:
        verdict = "complete_lp85_reached_L2_but_not_reproduced"
        summary = (f"lp85 reached L2 in-run (max_depth={max_depth}) but the reproduction gate did NOT "
                   "confirm it -- not a real bank. Goal quality still insufficient.")
    elif gps_any:
        verdict = "complete_repair_unblocked_planning_no_L2"
        summary = (f"The goal-repair UNBLOCKED planning (goal_satisfiable_any={gps_any}, "
                   f"repaired_seen={repaired_seen}) but lp85 stayed at L{max_depth} -- the loose "
                   "nonzero-count fallback is not lp85's real (shape-alignment) win. CONFIRMS the "
                   "remaining wall is goal QUALITY -> perception-grounded structural goal is the lever.")
    else:
        verdict = "complete_no_change_still_L1_goal_unsatisfiable"
        summary = (f"lp85 stayed L{max_depth}, goal_satisfiable_any={gps_any} -- the repair did NOT "
                   "fire or the induction still failed. Needs investigation.")

    artifact = {
        "experiment_id": "proto_l2_measure_fixstack",
        "honest_verdict": verdict,
        "verdict_summary": summary,
        "game": GAME,
        "budget": BUDGET,
        "max_depth_reached": max_depth,
        "l2_offline_reproduced": l2_repro,
        "goal_predicate_satisfiable_any": gps_any,
        "exemplar_present_on_any_induction": exemplar_any,
        "n_induction_attempts": n_induce,
        "goal_repair_fired_inferred": repaired_seen,
        "induced_is_level_complete_tail": wm_tail,
        "induction_summary": induction_summary,
        "arm_wall_s": arm.get("wall_s"),
        "pre_fix_baseline": "maxL=1, n_induce=2, goal_satisfiable=False (degenerate goal) -- the live "
                            "binary arm before the truncation + goal-repair fixes",
        "inference_substrate": "live_llm_inference",
        "solve_provenance": "development_proxy",
        "verifier_is_oracle": False,
        "random_seed": 4729,
        "duration_s": round(time.time() - t0, 2),
        "model_port": PORT,
        "model_path": QWEN,
    }
    RESULT.write_text(json.dumps(artifact, indent=2, default=str))
    log(f"VERDICT: {verdict}")
    log(f"SUMMARY: {summary}")
    log(f"  max_depth={max_depth} l2_reproduced={l2_repro} goal_satisfiable_any={gps_any} "
        f"repair_fired={repaired_seen} n_induce={n_induce}")
    log(f"  induced is_level_complete tail:\n{wm_tail[:200]}")
    log(f"DONE in {artifact['duration_s']}s -> {RESULT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
