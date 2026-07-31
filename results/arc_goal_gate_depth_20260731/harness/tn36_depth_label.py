#!/usr/bin/env python3
"""PHASE 3 VERIFICATION -- does tn36 now report `goal_unreached_within_depth`?

The claim under test is narrow and must be checked against the REAL thing, not a scale model:
the 2026-07-30 gate audit recorded tn36's `on` cell rejected as `degenerate_goal_predicate` with
`termination: queue_exhausted` and the detail "the reachable set was searched exhaustively
(frontier empty)". That detail is false -- the search was stopped by `max_depth=40` with the
chain still generating -- and this probe re-runs the SHIPPED gate against the SAME engine and the
SAME root grid to confirm the label now says so.

HOW THE ROOT GRID IS RECOVERED, and why it is the real one. Reused verbatim from the audit's own
Stage 1 (`p4/p2_stage1_planprobe.py`): Phase 1 measured the LLM-ON and LLM-OFF action traces to be
BYTE-IDENTICAL on tn36 (both A/A floors exactly 0), and the offline env is deterministic, so
replaying the OFF arm with an inert proposer reproduces the ON arm's state sequence and therefore
the root grid the ON arm induced against. The recovered grid's sha256 is compared against the
`f328c951a03d248d` the audit recorded; a mismatch invalidates the whole probe and is reported as
such rather than worked around.

The engine is READ from the audit's preserved per-cell store (md5 6d96491f…, 5256 B) and copied
into a scratch e3 dir before loading, so `results/arc_e3` and the audit's own store are never
written.

The root grid is also PERSISTED here, because the audit had to spend a 37s replay to recover
something that is a fixed deterministic constant -- and the next question about tn36 should not
have to spend it again.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import time

SCRATCH = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/ianblenke/github.com/ianblenke/carnot"

GAME = "tn36"
ARM = "on"
SEED = 1
EXPECTED_ROOT_SHA = "f328c951a03d248d"
EXPECTED_ENGINE_MD5 = "6d96491f80bec0319828ba1a04f5841e"

OUT = os.path.join(SCRATCH, "p5_tn36_depth_label.json")
E3_DIR = os.path.join(SCRATCH, "p5_e3", f"{GAME}__{ARM}__s{SEED}")
os.makedirs(E3_DIR, exist_ok=True)
os.environ["CARNOT_ARC_E3_DIR"] = E3_DIR
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

sys.path.insert(0, os.path.join(REPO, "python"))

ANON = "hg" + hashlib.sha256(f"{GAME}|heldout".encode()).hexdigest()[:6]
SRC_ENGINE = os.path.join(SCRATCH, "p4", "e3", f"{ARM}__{GAME}__s{SEED}", ANON, "world_model.py")

_T0 = time.monotonic()
CAPTURED: list[dict] = []


class _InertProposer:
    def __init__(self) -> None:
        self.n_induce_calls = 0
        self.no_think_prefix = ""
        self.max_tokens = 0
        self.tries = 0
        self.include_playbook_exemplars = False
        self.timeout = 0

    def induce(self, *_a, **_kw):  # noqa: ANN002,ANN003
        self.n_induce_calls += 1
        return False, "p5_depth_label_probe_no_generator"

    def world_model_candidates(self, _game=None):  # noqa: ANN001
        return []


def main() -> int:
    import numpy as np
    from carnot.agentic import arc_actions_to_progress as atp
    from carnot.agentic import arc_competition_agent as aca
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_llm_reinduction as reinduction

    rec: dict = {
        "probe": "phase3_tn36_depth_label",
        "game": GAME,
        "arm": ARM,
        "seed": SEED,
        "anon_game_id": ANON,
        "src_engine_path": SRC_ENGINE,
        "src_engine_exists": os.path.exists(SRC_ENGINE),
    }
    if not rec["src_engine_exists"]:
        rec["status"] = "blocked_no_stored_engine"
        pathlib.Path(OUT).write_text(json.dumps(rec, indent=2, sort_keys=True))
        print(json.dumps({"status": rec["status"]}))
        return 0

    with open(SRC_ENGINE, "rb") as fh:
        raw = fh.read()
    rec["src_engine_md5"] = hashlib.md5(raw).hexdigest()
    rec["src_engine_md5_matches_audit"] = rec["src_engine_md5"] == EXPECTED_ENGINE_MD5
    rec["src_engine_bytes"] = len(raw)

    dst_dir = os.path.join(E3_DIR, ANON)
    os.makedirs(dst_dir, exist_ok=True)
    with open(os.path.join(dst_dir, "world_model.py"), "wb") as fh:
        fh.write(raw)
    loaded = e3.load_engine(ANON)
    if loaded is None:
        rec["status"] = "blocked_engine_would_not_load"
        pathlib.Path(OUT).write_text(json.dumps(rec, indent=2, sort_keys=True))
        return 0
    engine = getattr(loaded, "engine", None) or (loaded[0] if isinstance(loaded, tuple) else None)
    is_done = getattr(loaded, "is_level_complete", None) or (
        loaded[1] if isinstance(loaded, tuple) and len(loaded) > 1 else None
    )

    base_policy = aca.E3AgentPolicy

    class _CapturePolicy(base_policy):  # type: ignore[misc,valid-type]
        def __init__(self, game_id, *a, **kw):  # noqa: ANN001,ANN002,ANN003
            super().__init__(ANON, *a, **kw)

        def _induce_and_plan(self):  # noqa: ANN201
            CAPTURED.append(
                {
                    "root_grid": None
                    if self.root_grid is None
                    else np.asarray(self.root_grid).tolist(),
                    "cell": self.cell,
                }
            )
            return super()._induce_and_plan()

    aca.E3AgentPolicy = _CapturePolicy
    atp.run_bounded_progress(
        GAME,
        "frozen_gemma_pin" if "frozen_gemma_pin" in atp.ARM_CONFIGS else "base",
        proposer=_InertProposer(),
        seed=SEED,
        budget=60,
        max_inductions=2,
        wall_s=600.0,
        explore_budget=24,
    )
    rec["n_captured_inductions"] = len(CAPTURED)
    if not CAPTURED or CAPTURED[0]["root_grid"] is None:
        rec["status"] = "blocked_no_root_grid_captured"
        pathlib.Path(OUT).write_text(json.dumps(rec, indent=2, sort_keys=True))
        return 0

    root = np.asarray(CAPTURED[0]["root_grid"])
    rec["root_grid_shape"] = list(root.shape)
    rec["root_grid_sha256"] = hashlib.sha256(root.tobytes()).hexdigest()[:16]
    rec["root_grid_sha256_matches_audit"] = rec["root_grid_sha256"] == EXPECTED_ROOT_SHA

    # Persist the root grid so nobody has to spend another replay to ask tn36 a question.
    np.save(os.path.join(SCRATCH, "p5_tn36_root_grid.npy"), root)

    # --- THE GATE, at the live parameters, exactly as the plain path calls it ---------------
    gate = reinduction._goal_satisfiability_check(engine=engine, goal=is_done, start_grid=root)
    rec["gate"] = {k: v for k, v in gate.items() if k != "counterexample"}
    rec["gate_counterexample"] = dict(gate.get("counterexample") or {})

    # The PRE-CHANGE label is derivable exactly from the same fields, because the old code keyed
    # on the budget alone: kind = within_budget if engine_calls >= max_nodes else degenerate.
    _budget_spent = int(gate["engine_calls"]) >= int(gate["max_nodes"])
    rec["label_before_this_commit"] = (
        "goal_unreached_within_budget" if _budget_spent else "degenerate_goal_predicate"
    )
    rec["label_after_this_commit"] = str(rec["gate_counterexample"].get("kind"))
    rec["label_changed"] = rec["label_before_this_commit"] != rec["label_after_this_commit"]
    rec["decision_unchanged_satisfiable_false"] = gate.get("satisfiable") is False

    # --- The corroboration: is the goal actually REACHABLE past the cap? ---------------------
    # If it is, `degenerate_goal_predicate` was an accusation against a correct predicate.
    for cap in (40, 80):
        _d: dict = {}
        t = time.monotonic()
        try:
            plan = e3.plan_in_model(
                engine, is_done, root, max_nodes=300000, max_depth=cap, diagnostics=_d
            )
            err = None
        except Exception as exc:  # noqa: BLE001
            plan, err = None, f"{type(exc).__name__}: {exc}"[:200]
        rec[f"plan_at_max_depth_{cap}"] = {
            "found": bool(plan),
            "length": len(plan) if plan else 0,
            "error": err,
            "diagnostics": dict(_d),
            "wall_s": round(time.monotonic() - t, 3),
        }

    # --- How deep does the goal actually sit? ------------------------------------------------
    # The gate's own probe, with the cap lifted, reports `first_true_depth` directly.
    deep_gate = reinduction._goal_satisfiability_check(
        engine=engine, goal=is_done, start_grid=root, max_nodes=300000, max_depth=200
    )
    rec["gate_with_cap_lifted"] = {
        "satisfiable": deep_gate.get("satisfiable"),
        "first_true_depth": deep_gate.get("first_true_depth"),
        "reachable_grids_evaluated": deep_gate.get("reachable_grids_evaluated"),
    }

    rec["status"] = "ok"
    rec["wall_s"] = round(time.monotonic() - _T0, 1)
    pathlib.Path(OUT).write_text(json.dumps(rec, indent=2, sort_keys=True, default=str))
    print(
        json.dumps(
            {
                "engine_md5_ok": rec["src_engine_md5_matches_audit"],
                "root_sha_ok": rec["root_grid_sha256_matches_audit"],
                "before": rec["label_before_this_commit"],
                "after": rec["label_after_this_commit"],
                "satisfiable": gate.get("satisfiable"),
                "engine_calls": gate.get("engine_calls"),
                "grids": gate.get("reachable_grids_evaluated"),
                "depth_truncated_nodes": gate.get("depth_truncated_nodes"),
                "termination": gate.get("termination"),
                "plan_d40": rec["plan_at_max_depth_40"]["found"],
                "plan_d80": (
                    rec["plan_at_max_depth_80"]["found"],
                    rec["plan_at_max_depth_80"]["length"],
                ),
                "first_true_depth": rec["gate_with_cap_lifted"]["first_true_depth"],
                "wall": rec["wall_s"],
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
