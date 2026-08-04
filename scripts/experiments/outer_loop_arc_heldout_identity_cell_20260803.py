#!/usr/bin/env python3
"""ONE CELL of the leave-one-game-out ADAPTER-FREE / held-out-identity measurement.

WHAT THIS IS. CLAUDE.md's "ARC-AGI-3 Generalization-Testing Floor" (2026-07-17) asks how far
the LIVE agent gets on a game when its per-game knowledge is removed and it must rely on the
reusable scaffolding alone. This runs exactly one (game, arm, seed) cell of that measurement and
writes one JSON row. The driver
(`outer_loop_arc_heldout_identity_driver_20260803.py`) fans these out.

WHY A SUBPROCESS PER CELL, not a loop in one process. `arc_executable_world_model.E3_DIR`
resolves from `CARNOT_ARC_E3_DIR` at IMPORT time, and the two arms need DIFFERENT engine stores
(the control gets a copy of the game's previously-induced engine, which a hidden game does not
have; the held-out arm gets an empty store). One process cannot hold both. A subprocess per cell
also means a policy crash is one lost cell, recorded by name, rather than a lost sweep.

THE TWO ARMS -- what actually differs, and what deliberately does not:

  control_identity_on   policy_game_id = the REAL game id. Every id-keyed lookup resolves.
                        E3_DIR pre-seeded with a COPY of results/arc_e3/<game>/.
  heldout_identity_off  policy_game_id = a synthetic id the registry has never seen
                        ("hg" + sha256(game|heldout)[:6]). Every id-keyed lookup misses.
                        E3_DIR empty -- no previously-induced engine, as for a hidden game.

Everything else is byte-identical between arms: same seed, same 400-action budget (the shipped
`CarnotAgent.MAX_ACTIONS`), the same LLM-off stub proposer, the same env running the REAL game.

`explore_budget` is deliberately passed as None so each arm gets the budget ITS OWN knowledge
state routes it to (`_route_explore_budget`: 24 when the registry records a goal-distance
mechanic class, else 80). Passing an explicit 24 -- which is `run_bounded_progress`'s default --
would force the control's routed budget onto the held-out arm and silently cancel one of the
seven leaks this measurement exists to remove. That is a treatment, not a confound.

WHY NO GPU, AND WHY THAT IS NOT A SHORTCUT. The LLM induction tier installs 0 plans (0 of 136
llm_on induce attempts installed one), and a measured llm-off arm was BIT-IDENTICAL to its
matched llm_on control on first_win / actions / reached_level / actions_to_first_levelup across
74/74 matched cells. The stub therefore reproduces the live trajectory while touching no GPU --
GPU 1 belongs to a concurrent workflow. The stub COUNTS its invocations so "the LLM tier never
fired" and "the LLM tier fired and returned nothing" stay distinguishable.

DELIVERY, NOT AVAILABILITY. Each id-keyed lookup site is instrumented at the CALLEE and records
the id it was called with plus the calling frame. A leak that is present in the import closure
but never called during a run cannot explain a difference, and a leak whose absence never
changes the trajectory is inert -- both are reported rather than assumed.

EVIDENCE IS READ-ONLY. results/arc_e3/ is copied OUT of, never written to; every write this
process makes lands in its own temp directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

# The shipped live action budget (`arc_competition_agent.CarnotAgent.MAX_ACTIONS`). Named here so
# the cell records what it used rather than leaving a bare 400 in an argparse default.
SHIPPED_MAX_ACTIONS = 400


def heldout_id(game: str) -> str:
    """The synthetic identity handed to the POLICY in the held-out arm.

    Deterministic (so a rerun is comparable) and structurally unlike any real ARC game id, so a
    registry / survey / adapter lookup cannot accidentally hit. The env keeps running the REAL
    game -- only the policy's notion of *which* game this is changes.
    """
    return "hg" + hashlib.sha256(f"{game}|heldout".encode()).hexdigest()[:6]


class LLMOffStubProposer:
    """A proposer that is present, configurable, and never generates.

    `induce` returns (False, reason) -- the SAME shape a real generator returns when it fails --
    so `E3AgentPolicy._induce_and_plan` takes its ordinary `proposer_failed_or_missing_root`
    branch instead of crashing. It is NOT a raising stub: at a 400-action budget the induce path
    is genuinely reached, and a raising stub would convert every such cell into an `error` row
    and destroy the measurement.

    Every call is counted. That is what separates "the LLM tier never fired" (n_induce_calls == 0)
    from "it fired and installed nothing" (n_induce_calls > 0) -- a distinction the prior
    llm_on/llm_off equivalence rests on and which must not be assumed here.
    """

    def __init__(self) -> None:
        self.no_think_prefix: Any = None
        self.max_tokens: Any = None
        self.tries: Any = None
        self.include_playbook_exemplars: Any = False
        self.n_induce_calls = 0
        self.n_other_calls = 0

    def induce(self, *args: Any, **kwargs: Any) -> tuple[bool, str]:
        self.n_induce_calls += 1
        return (False, "llm_off_stub_no_generator")

    def liveness_witness(self) -> dict[str, Any]:
        return {"llm_enabled": False, "stub": True, "n_induce_calls": self.n_induce_calls}

    def __getattr__(self, name: str) -> Any:
        # Anything else the policy probes for is absent, which is the honest state of a
        # generator-free run. Recorded so an unexpected probe is visible, not silent.
        if name.startswith("__"):
            raise AttributeError(name)
        object.__getattribute__(self, "__dict__")["n_other_calls"] = (
            object.__getattribute__(self, "__dict__").get("n_other_calls", 0) + 1
        )
        raise AttributeError(name)


def install_delivery_probes() -> dict[str, dict[str, Any]]:
    """Wrap each id-keyed lookup CALLEE and record what it was actually called with.

    Returns a mutable ledger the caller reads after the run. Keyed by site; each entry carries
    the call count, the distinct id arguments observed, and the first calling frame (file:line)
    so a site's reachability is read off the stack rather than inferred from the import graph.

    Patching happens on the DEFINING module object. Every consumer of these sites in the scored
    path holds the MODULE (``import carnot.agentic.arc_solve_learning as arc_solve_learning``,
    ``e3.load_engine``, ``rag.infer_query_mechanic_tags``), not a directly-bound function, so a
    module-attribute patch is seen by all of them. This is asserted, not assumed: a site whose
    count stays 0 for every cell is reported as NOT DELIVERED rather than quietly credited.
    """
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_playbook_retrieval as rag
    from carnot.agentic import arc_primitive_library as plib
    from carnot.agentic import arc_solve_learning as learning
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic import arc_strategy_router as router

    ledger: dict[str, dict[str, Any]] = {}

    def wrap(mod: Any, name: str, site: str, id_pos: int = 0, id_kw: str | None = None) -> None:
        fn = getattr(mod, name, None)
        if fn is None:
            ledger[site] = {"calls": 0, "ids": [], "first_caller": None, "missing": True}
            return
        rec: dict[str, Any] = {"calls": 0, "ids": [], "first_caller": None, "missing": False}
        ledger[site] = rec

        def wrapped(*a: Any, **k: Any) -> Any:
            rec["calls"] += 1
            ident: Any = None
            if id_kw is not None and id_kw in k:
                ident = k[id_kw]
            elif len(a) > id_pos:
                ident = a[id_pos]
            ident = str(ident)[:40]
            if ident not in rec["ids"]:
                rec["ids"].append(ident)
            if rec["first_caller"] is None:
                stack = traceback.extract_stack()
                # [-1] is this wrapper; [-2] is the real caller.
                if len(stack) >= 2:
                    fr = stack[-2]
                    rec["first_caller"] = f"{Path(fr.filename).name}:{fr.lineno}"
            return fn(*a, **k)

        wrapped.__name__ = getattr(fn, "__name__", name)
        setattr(mod, name, wrapped)

    # A1/A5 survey features + transfer routing ------------------------------------------------
    wrap(learning, "recommend_approach", "A1_A5_recommend_approach")
    wrap(learning, "_survey_features", "A1_survey_features")
    # A2 registry mechanic-class route --------------------------------------------------------
    wrap(router, "route_for_game", "A2_route_for_game")
    # A3 the target game's own registry digest used as the retrieval QUERY ---------------------
    wrap(plib, "game_digest", "A3_game_digest")
    # A4 documented-primitive retrieval --------------------------------------------------------
    wrap(plib, "retrieve_primitives", "A4_retrieve_primitives")
    # A6 per-game literal branches in the primitive-operator selector --------------------------
    wrap(kit, "select_primitive_operators", "A6_select_primitive_operators", id_kw="game")
    # B1 playbook mechanic tags ----------------------------------------------------------------
    wrap(rag, "infer_query_mechanic_tags", "B1_infer_query_mechanic_tags", id_kw="game")
    # B3 previously-induced per-game engine store ----------------------------------------------
    wrap(e3, "load_engine", "B3_load_engine")
    return ledger


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", required=True)
    ap.add_argument("--arm", required=True, choices=["control_identity_on", "heldout_identity_off"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--budget", type=int, default=SHIPPED_MAX_ACTIONS)
    ap.add_argument("--wall-s", type=float, default=1800.0)
    ap.add_argument("--out", required=True, help="output JSON row path")
    args = ap.parse_args(argv)

    t0 = time.time()
    row: dict[str, Any] = {
        "game": args.game,
        "arm": args.arm,
        "seed": args.seed,
        "budget": args.budget,
        "shipped_max_actions": SHIPPED_MAX_ACTIONS,
        "policy_game_id": None,
        "status": "started",
    }

    # ---- private engine store, set BEFORE the first carnot import (E3_DIR is import-time) ----
    tmp = tempfile.mkdtemp(prefix=f"arc_heldout_{args.game}_{args.arm}_")
    os.environ["CARNOT_ARC_E3_DIR"] = tmp
    # NO GPU. GPU 1 is owned by a concurrent workflow; this run needs no generator at all.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")

    seeded_engine = False
    if args.arm == "control_identity_on":
        # The control keeps the per-game asset a hidden game does not have. results/arc_e3 is
        # EVIDENCE: copied FROM, never written TO -- every write lands in `tmp`.
        src = REPO / "results" / "arc_e3" / args.game
        if src.is_dir():
            shutil.copytree(src, Path(tmp) / args.game)
            seeded_engine = True
    row["e3_dir"] = tmp
    row["e3_seeded_with_prior_engine"] = seeded_engine

    sys.path.insert(0, str(REPO / "python"))
    import logging

    logging.disable(logging.INFO)

    try:
        from carnot.agentic import arc_executable_world_model as e3

        resolved = str(getattr(e3, "E3_DIR", ""))
        row["e3_dir_resolved"] = resolved
        if os.path.abspath(resolved) != os.path.abspath(tmp):
            # FAIL CLOSED. A cell whose engine store is not the private one would be reading and
            # possibly WRITING the tracked evidence store, so it must not run at all.
            row["status"] = "blocked_e3_dir_not_honoured"
            Path(args.out).write_text(json.dumps(row, indent=2))
            return 2

        ledger = install_delivery_probes()

        from carnot.agentic import arc_actions_to_progress as atp

        pid = None if args.arm == "control_identity_on" else heldout_id(args.game)
        row["policy_game_id"] = pid if pid is not None else args.game

        stub = LLMOffStubProposer()
        res = atp.run_bounded_progress(
            args.game,
            "frozen_gemma_pin",
            proposer=stub,
            seed=args.seed,
            budget=args.budget,
            # The action budget is the ONLY intended bound. An induction cap would truncate the
            # arms at different points (they induce at different rates), which would make the
            # comparison a cap artifact.
            max_inductions=10**9,
            wall_s=args.wall_s,
            # Routed by each arm's OWN knowledge state -- see the module docstring.
            explore_budget=None,
            policy_game_id=pid,
        )
        r = res.to_row(include_events=False, include_trace=True)
        trace = r.pop("action_trace", []) or []
        row["result"] = r
        row["n_trace"] = len(trace)
        # THE VACUITY GUARD. If the two arms turn out to behave identically, the whole
        # measurement is either "removing the leaks changed nothing" or "nothing was removed",
        # and the difference between those two readings decides whether the result means
        # anything at all. Summary fields agreeing is suggestive; the ORDERED ACTION SEQUENCE
        # agreeing is decisive -- two different trajectories can share an action count, a level
        # count and a noop fraction. Stored as a digest rather than the trace so the row stays
        # small, plus a short prefix so a near-miss can be localized by eye.
        row["trace_sha256"] = hashlib.sha256("\n".join(trace).encode()).hexdigest()
        row["trace_head"] = trace[:12]
        row["stub_n_induce_calls"] = stub.n_induce_calls
        row["leak_delivery"] = ledger

        # ---- REPRODUCTION GATE (CLAUDE.md ARC Solve Reproducibility) -------------------------
        # A live-recorded trajectory is not a banked level. Replay it against a FRESH offline env
        # with the GENERIC label-only apply -- no adapter knowledge -- and report what it actually
        # reaches. Only reproduced levels are counted as the primary metric.
        gate: Any = None
        if (r.get("levels_gained") or 0) >= 1 and trace:
            try:
                from carnot.agentic import arc_solver_kit as kit

                gate = kit.reproduce(
                    args.game,
                    trace,
                    atp.replay_apply,
                    claimed_level=r.get("reached_level"),
                )
            except Exception as exc:  # a gate failure is a datum, not a silent pass
                gate = {"error": f"{type(exc).__name__}: {exc}"[:300], "reproduced": False}
        row["reproduction_gate"] = gate
        row["banked_levels"] = (
            int(r.get("reached_level") or 0)
            if (gate is not None and gate.get("reproduced"))
            else (0 if (r.get("levels_gained") or 0) >= 1 else 0)
        )
        row["reproduced"] = bool(gate.get("reproduced")) if isinstance(gate, dict) else False
        row["status"] = "ok"
    except Exception as exc:
        row["status"] = "blocked_cell_exception"
        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
        row["traceback"] = traceback.format_exc()[-1500:]
    finally:
        row["cell_wall_s"] = round(time.time() - t0, 2)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(row, indent=2))
        shutil.rmtree(tmp, ignore_errors=True)

    print(
        json.dumps(
            {
                "game": args.game,
                "arm": args.arm,
                "seed": args.seed,
                "status": row["status"],
                "levels_gained": (row.get("result") or {}).get("levels_gained"),
                "reproduced": row.get("reproduced"),
                "actions": (row.get("result") or {}).get("total_actions"),
                "wall_s": row["cell_wall_s"],
            }
        ),
        flush=True,
    )
    return 0 if row["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
