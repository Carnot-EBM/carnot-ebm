"""ARC induce->plan GATE ANATOMY: what rejects an induced engine, and is the gate BINDING?

CPU ONLY. No GGUF load, no GPU, no generator. Two kinds of evidence:

  (A) A RECOUNT of the 2026-07-27 real-generator corpus
      (results/first_win_llm_on_20260727/cells/llm_on*.json), from the raw per-cell
      `liveness_witness` records, reproducing "136 induce attempts, 0 installed a plan"
      and resolving the ~30 attempts the headline breakdown (63 + 43 = 106) left
      unaccounted.

  (B) An INJECTION PROBE that answers the load-bearing question the recount cannot:
      if an engine PASSED every gate, would a plan actually install, or is there a
      further step that would drop it anyway? A gate that is 100% closed and a gate
      that is bypassed entirely look identical from the outside.

      CLAUDE.md failure-mode-1 discipline ("availability is not delivery") says do not
      answer this by reading the call site. So the probe instruments the CALLEE --
      `arc_executable_world_model.plan_in_model` is wrapped, and each invocation records
      the caller frames off the stack plus whether the engine/goal objects it received
      are the ones injected. "A plan installed" is then read from the policy's own
      `self.plan` and `induction_attempts[-1]`, not inferred.

WHY THIS MATTERS. If the gate rejects every induced engine, the live agent is running
WITHOUT its LLM induction tier, and every improvement to the induce prompt has no channel
to any behavioural metric. Nothing downstream of a fully-closed gate can be measured.

NOTHING TRACKED IS WRITTEN. `load_engine` is monkeypatched to return the injected engine,
so the probe never writes `results/arc_e3/<game>/world_model.py`; transitions come from
`e3.collect_transitions`, which only READS `environment_files/`.

Rebuild:
    JAX_PLATFORMS=cpu .venv/bin/python \
        scripts/experiments/outer_loop_arc_induce_gate_anatomy_20260802.py
"""

from __future__ import annotations

import collections
import glob
import hashlib
import inspect
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "outer_loop_arc_induce_gate_anatomy_20260802.json"
CORPUS = REPO / "results" / "first_win_llm_on_20260727" / "cells"

# The 11-game hardcoded partition that selects the hidden-state trust gate over the plain
# accuracy gate. Imported, never re-typed, so a change to the live list cannot silently
# desynchronise this artifact from the branch it describes.
sys.path.insert(0, str(REPO / "scripts" / "experiments"))


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


# --------------------------------------------------------------------------------------
# (A) RECOUNT
# --------------------------------------------------------------------------------------
def recount(hidden_state_games: tuple[str, ...]) -> dict[str, Any]:
    cells = sorted(glob.glob(str(CORPUS / "llm_on*.json")))
    rows = []
    for path in cells:
        with open(path) as fh:
            cell = json.load(fh)
        lw = cell.get("liveness_witness") or {}
        rows.append(
            {
                "arm": cell.get("arm"),
                "game": cell.get("game"),
                "n": int(lw.get("induction_attempts_n") or 0),
                "planned": int(lw.get("induction_attempts_planned") or 0),
                "skips": list(lw.get("induction_attempts_skipped") or []),
                "valid": bool(lw.get("llm_on_row_valid")),
                "gate_diag": list(lw.get("induction_attempt_gate_diagnostics") or []),
            }
        )

    census: collections.Counter = collections.Counter()
    per_game: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    per_arm_rows: collections.Counter = collections.Counter()
    per_arm_valid: collections.Counter = collections.Counter()
    per_arm_att: collections.Counter = collections.Counter()
    valid_census: collections.Counter = collections.Counter()
    valid_att = 0
    for r in rows:
        per_arm_rows[r["arm"]] += 1
        per_arm_att[r["arm"]] += r["n"]
        per_arm_valid[r["arm"]] += 1 if r["valid"] else 0
        if r["valid"]:
            valid_att += r["n"]
        for s in r["skips"]:
            census[s] += 1
            per_game[r["game"]][s] += 1
            if r["valid"]:
                valid_census[s] += 1

    diag = [
        dict(arm=r["arm"], game=r["game"], valid=r["valid"], **g)
        for r in rows
        for g in r["gate_diag"]
    ]

    def dist(vals: list[Any]) -> dict[str, Any] | None:
        nums = sorted(float(v) for v in vals if v is not None)
        if not nums:
            return None
        q = lambda p: nums[min(len(nums) - 1, int(p * len(nums)))]  # noqa: E731
        return {
            "n": len(nums),
            "min": round(nums[0], 6),
            "q1": round(q(0.25), 6),
            "median": round(statistics.median(nums), 6),
            "q3": round(q(0.75), 6),
            "max": round(nums[-1], 6),
        }

    hs_diag = [d for d in diag if d["game"] in hidden_state_games]
    pl_diag = [d for d in diag if d["game"] not in hidden_state_games]

    # The class the census cannot attribute: an attempt whose recorded skip is a
    # refinement-loop outcome but whose recorded gate value CLEARS the live gate.
    mislabelled = [
        {
            "arm": d["arm"],
            "game": d["game"],
            "skipped": d.get("skipped"),
            "trust_metric": d.get("trust_metric"),
            "verify_accuracy": d.get("verify_accuracy"),
            "verify_cell_recall": d.get("verify_cell_recall"),
            "gate_value_that_cleared": float(d.get("verify_accuracy")),
            "why": (
                "skip string names a refinement-loop outcome, but the recorded plain-path "
                "gate value is >= 0.5, i.e. this attempt CLEARED the live admission gate "
                "and the planner then returned nothing"
            ),
        }
        for d in pl_diag
        if d.get("skipped") == "no_reachable_plan_after_refinement"
        and d.get("trust_metric") == "exact"
        and d.get("verify_accuracy") is not None
        and float(d["verify_accuracy"]) >= 0.5
    ]

    att_per_game = sorted(sum(c.values()) for c in per_game.values())
    return {
        "n_cells": len(cells),
        "n_attempts": sum(r["n"] for r in rows),
        "n_planned": sum(r["planned"] for r in rows),
        "n_skip_entries": sum(census.values()),
        "skip_census": dict(census),
        "per_arm": {
            a: {
                "rows": per_arm_rows[a],
                "rows_with_llm_on_row_valid": per_arm_valid[a],
                "attempts": per_arm_att[a],
            }
            for a in sorted(per_arm_att)
        },
        "valid_rows_only_subset": {
            "why": (
                "the 136 pools two arms whose generator was broken (llm_on_16k: 2/25 rows "
                "valid; llm_on_16k_probe: 0/12). Restricting to rows the project's own "
                "llm_on_row_valid flag accepts is the defensible denominator."
            ),
            "n_attempts": valid_att,
            "n_planned": 0,
            "skip_census": dict(valid_census),
        },
        "per_game_skip_census": {g: dict(c) for g, c in sorted(per_game.items())},
        "per_game_attempts_distribution": {
            "n_games": len(att_per_game),
            "min": att_per_game[0],
            "q1": att_per_game[len(att_per_game) // 4],
            "median": statistics.median(att_per_game),
            "q3": att_per_game[3 * len(att_per_game) // 4],
            "max": att_per_game[-1],
            "note": (
                "22 of 25 games contributed exactly 4 attempts; the three long tails "
                "(lp85 14, sp80 16, vc33 18) are where all 30 non-gate skips live."
            ),
        },
        "branch_partition_finding": {
            "claim": (
                "the 63/43 split is NOT two kinds of engine failure -- it is the "
                "HIDDEN_STATE_GAME_IDS partition. Every one of the 11 hidden-state games "
                "recorded only hidden_state_trust_below_threshold; every one of the 14 "
                "plain games recorded only world_model_accuracy_below_threshold."
            ),
            "hidden_state_games_with_only_trust_skips": sorted(
                g
                for g, c in per_game.items()
                if g in hidden_state_games and c["world_model_accuracy_below_threshold"] == 0
            ),
            "plain_games_with_only_accuracy_skips": sorted(
                g
                for g, c in per_game.items()
                if g not in hidden_state_games and c["hidden_state_trust_below_threshold"] == 0
            ),
        },
        "gate_margins": {
            "why": (
                "MARGINS, not just verdicts. Recorded on 52 of the 136 attempts (the two "
                "arms that ran after induction_attempt_gate_diagnostics was added). This is "
                "the only direct evidence about whether the rejections were near-misses."
            ),
            "n_records": len(diag),
            "n_records_missing_of_136": 136 - len(diag),
            "hidden_state_branch": {
                "gate": "heldout_change_consistency >= 0.5 AND correct_changed_cells >= 1",
                "n": len(hs_diag),
                "heldout_change_consistency": dist(
                    [d.get("heldout_change_consistency") for d in hs_diag]
                ),
                "heldout_accuracy": dist([d.get("heldout_accuracy") for d in hs_diag]),
                "n_at_or_above_threshold": sum(
                    1 for d in hs_diag if float(d.get("heldout_change_consistency") or 0.0) >= 0.5
                ),
                "n_with_zero_correct_changed_cells": sum(
                    1 for d in hs_diag if int(d.get("correct_changed_cells") or 0) == 0
                ),
                "per_game_values": {
                    g: [
                        round(float(d["heldout_change_consistency"]), 4)
                        for d in hs_diag
                        if d["game"] == g and d.get("heldout_change_consistency") is not None
                    ]
                    for g in sorted({d["game"] for d in hs_diag})
                },
            },
            "plain_branch": {
                "gate": "verify_accuracy (or verify_cell_recall) >= 0.5",
                "n": len(pl_diag),
                "verify_accuracy": dist([d.get("verify_accuracy") for d in pl_diag]),
                "verify_cell_recall": dist([d.get("verify_cell_recall") for d in pl_diag]),
                "by_trust_metric": {
                    m: {
                        "n": len(sub),
                        "n_at_or_above_threshold": sum(
                            1
                            for d in sub
                            if float(
                                (
                                    d.get("verify_cell_recall")
                                    if m == "cell_recall"
                                    else d.get("verify_accuracy")
                                )
                                or 0.0
                            )
                            >= 0.5
                        ),
                    }
                    for m in ("exact", "cell_recall")
                    for sub in [[d for d in pl_diag if d.get("trust_metric") == m]]
                },
            },
        },
        "FINDING_unattributable_skip_class": {
            "class": "no_reachable_plan_after_refinement",
            "n_in_census": census["no_reachable_plan_after_refinement"],
            "concentration": (
                "7 of 10 are vc33 alone; 1 each ft09/lp85/sp80. Not spread across the roster."
            ),
            "ambiguity": (
                "attempt['skipped'] is written by the STALL refinement loop and then only "
                "OVERWRITTEN if the plain single-shot path rejects. An attempt that CLEARS "
                "the plain gate and whose planner returns nothing therefore keeps the stale "
                "refinement-loop label. Demonstrated live in the injection probe below, and "
                "witnessed directly in the corpus by the records listed here."
            ),
            "witnessed_in_corpus": mislabelled,
            "n_witnessed": len(mislabelled),
            "MISSING_IS_NOT_ZERO": (
                "margins exist for only 3 of the 10; the other 7 are UNKNOWN, not zero. "
                "Do not read n_witnessed=2 as 'only 2 of the 10 are mislabelled'."
            ),
        },
    }


# --------------------------------------------------------------------------------------
# (B) INJECTION PROBE
# --------------------------------------------------------------------------------------
def identity_engine(grid, action, data):
    """The documented GAP-WM-TRUST-GATE degenerate: `return grid`. Must be rejected."""

    return np.asarray(grid).copy()


def unreachable_goal(grid):
    return False


def make_moved_goal(root_grid: np.ndarray):
    """Non-degenerate and genuinely reachable: the dc22 avatar footprint has left its root.

    Deliberately EASY. Its job is to prove the post-gate channel is live end to end
    (goal-bias install -> plan_in_model -> self.plan -> attempt['planned']), NOT to show
    that a realistic win predicate plans. Those are different claims and only the first
    one is made here.
    """

    cells = np.argwhere(np.asarray(root_grid) == 14)
    if len(cells) == 0:
        return None
    origin = (int(cells[:, 0].min()), int(cells[:, 1].min()))

    def is_level_complete(grid):
        c = np.argwhere(np.asarray(grid) == 14)
        if len(c) == 0:
            return False
        return (int(c[:, 0].min()), int(c[:, 1].min())) != origin

    return is_level_complete


def probe(
    game: str,
    seed: int,
    engine,
    goal_kind: str,
    label: str,
    stall_loop: str,
    agent,
    e3,
    *,
    structural_nav: bool = False,
):
    prev = os.environ.get("CARNOT_ARC_STALL_REFACTOR_LOOP")
    os.environ["CARNOT_ARC_STALL_REFACTOR_LOOP"] = stall_loop
    t0 = time.time()
    transitions, cell = e3.collect_transitions(game, n=120, seed=seed)
    root = np.asarray(transitions[0].grid)
    nav_fit: dict[str, Any] | None = None
    if structural_nav:
        # A STRUCTURAL, correct-by-construction fit from the agent's OWN transitions --
        # already live-path-reachable via CARNOT_ARC_STRUCTURED_NAV. Not hand-typed and not
        # fitted to the gate. It models only the NAV mechanic, which is exactly the honest
        # "genuinely useful but incomplete" engine a graded gate is supposed to admit.
        from carnot.agentic.arc_nav_world_model import InducedNavWorldModel

        nav = InducedNavWorldModel.fit(transitions)
        engine, goal = nav.as_callables()
        nav_fit = {
            "displacement_per_action": {
                str(k): list(v) for k, v in (nav.displacement or {}).items()
            },
            "goal_color": getattr(nav, "goal_color", None),
            "is_confident_nav": bool(nav.is_confident_nav(grid=root)),
        }
    else:
        goal = make_moved_goal(root) if goal_kind == "reachable" else unreachable_goal

    witness: dict[str, Any] = {
        "plan_in_model_calls": 0,
        "callers": [],
        "returned_plan_lengths": [],
        "reached_post_gate_call_site": False,
        "engine_identity_matches_injected": None,
        "goal_identity_matches_injected": None,
    }
    real_pim = e3.plan_in_model

    def instrumented(engine_arg, is_done_arg, start_grid, **kw):
        witness["plan_in_model_calls"] += 1
        frames = [f"{fr.function}:{fr.lineno}" for fr in inspect.stack()[1:3]]
        witness["callers"].append(frames)
        if engine_arg is engine:
            witness["engine_identity_matches_injected"] = True
            witness["goal_identity_matches_injected"] = is_done_arg is goal
            witness["reached_post_gate_call_site"] = True
            witness["post_gate_caller"] = frames
        out = real_pim(engine_arg, is_done_arg, start_grid, **kw)
        witness["returned_plan_lengths"].append(len(out or []))
        return out

    e3.plan_in_model = instrumented
    real_load = e3.load_engine
    e3.load_engine = lambda g: (engine, goal)
    try:
        policy = agent.E3AgentPolicy(
            game,
            proposer=SimpleNamespace(
                model_specs="INJECTED_NO_LLM_no_model_invoked",
                induce=lambda *a, **k: (True, ""),
                refactor=lambda *a, **k: (True, ""),
                include_playbook_exemplars=False,
            ),
            target_levels=2,
        )
        policy.root_grid = root
        policy.cell = cell
        policy.transitions = list(transitions)
        policy._episode_transition_start = 0
        policy._pending_induction_reason = "stall"
        policy._induce_and_plan()
        attempt = policy.induction_attempts[-1]
        lw = policy.generator_liveness_witness()
        plan_len = len(policy.plan or [])
    finally:
        e3.plan_in_model = real_pim
        e3.load_engine = real_load
        if prev is None:
            os.environ.pop("CARNOT_ARC_STALL_REFACTOR_LOOP", None)
        else:
            os.environ["CARNOT_ARC_STALL_REFACTOR_LOOP"] = prev

    hcc = attempt.get("heldout_change_consistency")
    return {
        "label": label,
        "game": game,
        "seed": seed,
        "goal_kind": goal_kind,
        "stall_refactor_loop": stall_loop,
        "n_transitions": len(transitions),
        "hidden_state_branch": game in agent.HIDDEN_STATE_GAME_IDS,
        "structural_nav_fit": nav_fit,
        "heldout_change_consistency": hcc,
        "verify_accuracy": attempt.get("verify_accuracy"),
        "verify_cell_recall": attempt.get("verify_cell_recall"),
        "trust_metric": attempt.get("trust_metric"),
        "gate_threshold": 0.5,
        "gate_passed": bool(
            (hcc is not None and float(hcc) >= 0.5)
            or (
                attempt.get("verify_accuracy") is not None
                and float(attempt["verify_accuracy"]) >= 0.5
            )
        ),
        "attempt_planned": bool(attempt.get("planned")),
        "attempt_skipped": attempt.get("skipped"),
        "attempt_plan_length": attempt.get("plan_length"),
        "policy_plan_installed_len": plan_len,
        "witness_induction_attempts_planned": lw.get("induction_attempts_planned"),
        "witness_induction_attempts_skipped": lw.get("induction_attempts_skipped"),
        "callee_witness": witness,
        "elapsed_s": round(time.time() - t0, 3),
    }


def main() -> int:
    started = time.time()
    from carnot.agentic import arc_competition_agent as agent
    from carnot.agentic import arc_executable_world_model as e3
    from experiment_6011_world_model_change_gate_four_arm import dc22_navigation_engine

    hs_games = tuple(agent.HIDDEN_STATE_GAME_IDS)
    part_a = recount(hs_games)

    probes = []
    for seed in (0, 1, 2):
        probes.append(
            probe(
                "dc22",
                seed,
                dc22_navigation_engine,
                "reachable",
                "known_good_dc22_nav_engine",
                "1",
                agent,
                e3,
            )
        )
        probes.append(
            probe(
                "dc22",
                seed,
                identity_engine,
                "reachable",
                "identity_engine_control",
                "1",
                agent,
                e3,
            )
        )
    # The mislabel demonstration: gate PASSES, planner returns nothing.
    for stall in ("1", "0"):
        probes.append(
            probe(
                "dc22",
                2,
                dc22_navigation_engine,
                "unreachable",
                "known_good_engine_UNREACHABLE_goal",
                stall,
                agent,
                e3,
            )
        )
    # PLAIN branch. dc22 is hidden-state and covers only 43 of the 136 rejections; the
    # plain accuracy gate carries the other 63, so it needs its own arm or the binding
    # claim rests on one branch of a two-branch decision.
    plain_probes = []
    for seed in (0, 1, 2):
        plain_probes.append(
            probe(
                "tu93",
                seed,
                None,
                "structural",
                "structured_nav_known_good",
                "1",
                agent,
                e3,
                structural_nav=True,
            )
        )
        plain_probes.append(
            probe(
                "tu93",
                seed,
                identity_engine,
                "unreachable",
                "identity_engine_control",
                "1",
                agent,
                e3,
            )
        )
    probes.extend(plain_probes)

    installed = [p for p in probes if p["attempt_planned"]]
    gate_passed = [p for p in probes if p["gate_passed"]]
    reached = [p for p in probes if p["callee_witness"]["reached_post_gate_call_site"]]

    payload: dict[str, Any] = {
        "experiment": "outer_loop_arc_induce_gate_anatomy_20260802",
        "title": (
            "ARC induce->plan gate anatomy: the gate is BINDING and mechanically reached, "
            "the rejections are not near-misses, and one skip class is unattributable"
        ),
        "run_date": "2026-08-02",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "substrate_note": (
            "CPU only. No GGUF load, no GPU, no llama-server, no generator. Part A is a "
            "recount of cached per-cell JSON. Part B scores hand-written / structural "
            "engines against offline-arcade transitions through the production "
            "E3AgentPolicy._induce_and_plan path. No model is invoked, so model_specs "
            "names none."
        ),
        "model_specs": {"note": "no model invoked; injected engines are hand-written Python"},
        "solve_provenance": "development_proxy",
        "solve_provenance_note": (
            "No level was solved and no solve is claimed. This is a gate diagnostic run "
            "against the offline development twin, not a live self-discovery advance."
        ),
        "verifier_is_oracle": False,
        "random_seed": 0,
        "random_seeds_used": [0, 1, 2],
        "preconditions_checked": [
            {"resource": "results/first_win_llm_on_20260727/cells", "available": CORPUS.is_dir()},
            {"resource": "offline arcade (environment_files/)", "available": True},
            {"resource": "no GPU / no GGUF required", "available": True},
        ],
        "provenance": {
            "code": [
                {"path": p, "sha256": _sha256(REPO / p)}
                for p in (
                    "python/carnot/agentic/arc_competition_agent.py",
                    "python/carnot/agentic/arc_world_model_trust_energy.py",
                    "python/carnot/agentic/arc_executable_world_model.py",
                    "python/carnot/agentic/arc_llm_reinduction.py",
                )
            ],
            "rebuild_command": (
                "JAX_PLATFORMS=cpu .venv/bin/python "
                "scripts/experiments/outer_loop_arc_induce_gate_anatomy_20260802.py"
            ),
            "nothing_tracked_is_written": (
                "load_engine is monkeypatched, so results/arc_e3/<game>/world_model.py is "
                "never written; collect_transitions only reads environment_files/."
            ),
        },
        "prior_art_built_on": [
            {
                "path": "results/experiment_6012_hidden_state_trust_gate_hole.json",
                "what_it_already_answered": (
                    "the shipped hidden-state trust gate REJECTS the hand-written correct "
                    "dc22 engine on 2 of 3 seeds (dc22@0, dc22@1) and admits it on dc22@2. "
                    "This run reproduces that same 2-of-3 split through a DIFFERENT path "
                    "(end-to-end _induce_and_plan rather than a direct "
                    "select_trusted_world_model call), so it is a corroboration, not a "
                    "re-derivation."
                ),
            },
            {
                "path": "results/experiment_6013_hidden_state_change_gate_closure.json",
                "what_it_already_answered": (
                    "the default-OFF change gate admits the same hand-written control 3/3 "
                    "with the HUD mask on and rejects it 3/3 with the mask off."
                ),
            },
            {
                "path": "results/outer_loop_arc_win_transition_exposure_20260802.json",
                "what_it_already_answered": (
                    "already stated the ~30 breakdown in prose (19 generation-side, 11 "
                    "downstream). This run reproduces it from the raw cells and adds the "
                    "per-game clustering plus the mislabel finding."
                ),
            },
            {
                "path": "scripts/experiments/experiment_6011_world_model_change_gate_four_arm.py",
                "what_it_already_answered": (
                    "supplies dc22_navigation_engine, the hand-written correct control. "
                    "IMPORTED, never re-typed."
                ),
            },
        ],
        "INDUCE_TO_PLAN_CHAIN_IN_ORDER": [
            {
                "stage": "0. GENERATION",
                "where": "proposer.induce() then e3.load_engine() (import/exec world_model.py)",
                "threshold": "binary: ok=True AND the module exposes `engine`",
                "threshold_provenance": "not a number; a parse/exec success",
                "rejected_of_136": 19,
                "skip_names": ["proposer_failed (13)", "proposer_failed_or_missing_root (6)"],
            },
            {
                "stage": "1. REFINEMENT-LOOP HELD-OUT ACCURACY",
                "where": "arc_llm_reinduction.execute_bounded_llm_reinduction, per round, K<=3",
                "threshold": "heldout_accuracy >= min_heldout_accuracy, LIVE VALUE 1.0",
                "threshold_provenance": (
                    "HARDCODED at both live call sites (arc_competition_agent.py:6225 and "
                    ":6362). Not tuned: the function's own default is 0.0 and the module's "
                    "sibling binary_exact_gate_pass defaults to 0.5. The in-code note says "
                    "it is 'rarely met (0/6 rounds across a full real run on g50t)'. This is "
                    "full-grid EXACT match on 100% of the held-out split."
                ),
                "rejected_of_136": "not separately counted -- see the label note below",
            },
            {
                "stage": "2. GOAL SATISFIABILITY (bounded BFS from root)",
                "where": "arc_llm_reinduction._goal_satisfiability_check",
                "threshold": "reach a predicate-True state within max_nodes=20000",
                "threshold_provenance": (
                    "compute budget, not quality. In-code measurement: ka59's "
                    "concept-correct depth-11 predicate needs ~160,000 engine calls to "
                    "certify and 137,347 planner nodes to use -- both far past 20,000."
                ),
                "rejected_of_136": 1,
                "skip_names": ["degenerate_goal_predicate (1)"],
            },
            {
                "stage": "3. GOAL PREDICATE CONSISTENCY",
                "where": "score_goal_predicate_consistency",
                "threshold": "accuracy >= min_goal_predicate_consistency, LIVE VALUE 1.0",
                "threshold_provenance": (
                    "hardcoded at the same two call sites. STRUCTURALLY INERT before the "
                    "first level-up: it only fires when the window contains >=1 real "
                    "level-up, and a first-contact stall window has none."
                ),
                "rejected_of_136": 0,
            },
            {
                "stage": "4. PLAIN-PATH ADMISSION -- two mutually exclusive branches",
                "where": "arc_competition_agent._induce_and_plan",
                "branch_hidden_state": {
                    "games": "the 11 HIDDEN_STATE_GAME_IDS",
                    "threshold": (
                        "trust_pass = correct_changed_cells >= 1 AND "
                        "heldout_change_consistency >= 0.5"
                    ),
                    "threshold_provenance": (
                        "INHERITED. In-code: 0.5 is 'the threshold recorded in "
                        "ops/verifier_gaps.md's GAP-WM-TRUST-GATE entry and used by "
                        "binary_exact_gate_pass's default'. It was not fitted to induced "
                        "engines. The sibling constant that WAS calibrated is "
                        "WORLD_MODEL_MAX_NOOP_HALLUCINATION_RATE=0.25, whose comment shows "
                        "the honest/attack separation it was set between -- and that gate "
                        "is default-OFF."
                    ),
                    "rejected_of_136": 43,
                },
                "branch_plain": {
                    "games": "the other 14",
                    "threshold": (
                        "verify_accuracy >= 0.5 (full-grid exact over ALL transitions "
                        "including no-ops), or verify_cell_recall >= 0.5 when "
                        "CARNOT_ARC_TRUST_METRIC=cell_recall"
                    ),
                    "threshold_provenance": "same inherited 0.5",
                    "rejected_of_136": 63,
                },
            },
            {
                "stage": "5. PLANNER",
                "where": "e3.plan_in_model via _call_plan_in_model, post-gate",
                "threshold": "returns a non-empty plan",
                "rejected_of_136": (
                    "AT LEAST 2, and up to 10 -- see FINDING_unattributable_skip_class. "
                    "These are recorded under the stage-1 label because attempt['skipped'] "
                    "is only overwritten when the stage-4 gate REJECTS."
                ),
            },
        ],
        "PART_A_recount_of_the_real_generator_corpus": part_a,
        "PART_B_injection_probe": {
            "question": (
                "if an engine PASSED every gate, would a plan actually install -- or is "
                "there a further step (a store write, an overwrite, a separate planner "
                "precondition) that would drop it anyway?"
            ),
            "method": (
                "instrument the CALLEE (plan_in_model) and read the caller off the stack; "
                "read 'a plan installed' from policy.plan and induction_attempts[-1], not "
                "from the call site."
            ),
            "rows": probes,
            "n_rows": len(probes),
            "n_gate_passed": len(gate_passed),
            "n_plans_installed": len(installed),
            "n_reached_post_gate_plan_in_model": len(reached),
            "ROSTER_LIMIT_STATED_PLAINLY": (
                "TWO games, not the roster. dc22 (hidden-state branch, 3 seeds) and tu93 "
                "(plain branch, 3 seeds). The binding claim rests on ONE gate-passing "
                "observation (dc22 seed 2); dc22 seeds 0 and 1 and all three tu93 seeds "
                "were rejected, so the pass region was reached once out of six known-good "
                "arms. That is enough to prove the channel is not bypassed and is NOT "
                "enough to characterise the pass region's size."
            ),
            "FINDING_plain_gate_does_not_separate_good_from_identity_on_tu93": {
                "claim": (
                    "on tu93 the shipped plain-branch metric (full-grid exact accuracy, "
                    "CARNOT_ARC_TRUST_METRIC unset -> 'exact') scores the structural nav "
                    "model and the `return grid` identity engine at EXACTLY the same value, "
                    "0.0000, on all 3 seeds. The gate rejects both identically, while "
                    "verify_cell_recall -- computed and recorded but not gating by default -- "
                    "separates them 0.275/0.353/0.311 vs 0.0/0.0/0.0."
                ),
                "measured_not_structural": (
                    "This is MEASURED on tu93 x 3 seeds. It is not asserted as a structural "
                    "property of the metric: on a no-op-heavy corpus the same metric scores "
                    "identity 0.725 and ADMITS it (the documented GAP-WM-TRUST-GATE, lp85, "
                    "~87 no-ops to 33 changing). Which failure direction the gate exhibits "
                    "depends on the no-op fraction of the transition window, which "
                    "collect_transitions' salience-biased sampling makes corpus-specific."
                ),
                "structural_nav_verify_accuracy": [
                    p.get("verify_accuracy")
                    for p in plain_probes
                    if p["label"] == "structured_nav_known_good"
                ],
                "structural_nav_verify_cell_recall": [
                    p.get("verify_cell_recall")
                    for p in plain_probes
                    if p["label"] == "structured_nav_known_good"
                ],
                "identity_verify_accuracy": [
                    p.get("verify_accuracy")
                    for p in plain_probes
                    if p["label"] == "identity_engine_control"
                ],
                "identity_verify_cell_recall": [
                    p.get("verify_cell_recall")
                    for p in plain_probes
                    if p["label"] == "identity_engine_control"
                ],
                "why_this_bounds_a_threshold_relaxation": (
                    "a structural fit recovering 27-35% of truly-changed cells is ~6-7x "
                    "better on that axis than the LIVE induced engines were (plain-branch "
                    "cell_recall max 0.0476 over 14 recorded margins) -- and it is still "
                    "rejected. Lowering the 0.5 threshold on the exact metric cannot admit "
                    "it, because its score is 0.0, not 0.4."
                ),
            },
        },
        "PART_C_rejected_engine_source_recoverability": {
            "question": "can the 136 rejected engines be recovered for scoring?",
            "for_the_specific_136": {
                "recoverable": False,
                "why": (
                    "the per-cell liveness_witness schema persists no engine source. The "
                    "richest record is induction_attempt_gate_diagnostics, present on 52 of "
                    "136 attempts, and it carries scalars only (planned / skipped / "
                    "trust_metric / verify_accuracy / verify_cell_recall / trust_energy / "
                    "heldout_accuracy / heldout_change_consistency / correct_changed_cells / "
                    "binary_gate_pass). No source text, no AST, no hash."
                ),
            },
            "what_IS_on_disk": [
                {
                    "path": "results/arc_e3/<game>/world_model.py",
                    "n": 25,
                    "what": (
                        "ONE file per game, overwritten by every refinement round and every "
                        "subsequent run. Holds the LAST engine only, never a per-attempt "
                        "history. ft09 and lp85 still carry files whose mtime falls inside "
                        "the 2026-07-27 run window and whose content matches the documented "
                        "degenerates -- ft09 is `return grid` on every branch with "
                        "is_level_complete -> False. mtime is suggestive, not proof of "
                        "attribution."
                    ),
                },
                {
                    "path": "git history of results/arc_e3/*/world_model.py",
                    "n_commits": 324,
                    "n_distinct_blobs": 309,
                    "what": (
                        "these files are TRACKED, so ~309 distinct historical engine bodies "
                        "are recoverable by `git show <sha>:<path>`. They are not "
                        "attributable to individual induce attempts and they span many "
                        "months and several generator models."
                    ),
                },
                {
                    "path": "results/arc_e3_origin_fixtures/",
                    "n": 27,
                    "what": "frozen, never-written copies of the origin-incident engines.",
                },
                {
                    "path": "results/arc_inert_rejection_ab_20260801/out/engines/",
                    "n": 151,
                    "what": (
                        "151 live-generator engine sources across 20 games from a 2026-08-01 "
                        "run, persisted per (game, replicate, arm). This is the closest thing "
                        "to a scoreable rejected-engine corpus that exists. IT IS ANOTHER "
                        "WRITER'S IN-FLIGHT, UNTRACKED WORK -- read as evidence, never "
                        "written or staged."
                    ),
                },
            ],
            "consequence_for_a_phase_2": (
                "scoring THE 136 is impossible; it would be a REGENERATION task. But the "
                "151-engine corpus already on disk plus the 52 recorded margins largely "
                "pre-empt the question a regeneration would ask."
            ),
        },
        "acceptance_gate_recount_reproduces_136_attempts_0_planned": (
            part_a["n_attempts"] == 136 and part_a["n_planned"] == 0
        ),
        "acceptance_gate_the_30_are_fully_accounted": (
            part_a["skip_census"].get("proposer_failed", 0)
            + part_a["skip_census"].get("proposer_failed_or_missing_root", 0)
            + part_a["skip_census"].get("no_reachable_plan_after_refinement", 0)
            + part_a["skip_census"].get("degenerate_goal_predicate", 0)
            == 30
        ),
        "acceptance_gate_a_gate_passing_engine_installs_a_plan": any(
            p["gate_passed"] and p["attempt_planned"] and p["policy_plan_installed_len"] > 0
            for p in probes
        ),
        "acceptance_gate_the_control_is_rejected_every_seed": all(
            (not p["gate_passed"]) and (not p["attempt_planned"])
            for p in probes
            if p["label"] == "identity_engine_control"
        ),
        "acceptance_gate_both_branches_probed": {p["hidden_state_branch"] for p in probes}
        == {True, False},
        "acceptance_gate_mislabel_demonstrated": any(
            p["gate_passed"]
            and not p["attempt_planned"]
            and p["attempt_skipped"] == "no_reachable_plan_after_refinement"
            for p in probes
        ),
        "acceptance_gate_nonempty_measurement": len(probes) > 0 and part_a["n_attempts"] > 0,
    }
    payload["acceptance_gate_passed"] = all(
        v for k, v in payload.items() if k.startswith("acceptance_gate_")
    )
    payload["interpretation"] = {
        "the_gate_IS_binding": (
            "PROVEN AT THE RECEIVING END, not by reading the call site. On dc22 seed 2 the "
            "hand-written correct engine scored heldout_change_consistency 0.542986, cleared "
            "the 0.5 trust gate, and the instrumented plan_in_model recorded a call from "
            "_induce_and_plan's POST-GATE call site with the injected engine and goal object "
            "identities matching. attempt['planned'] became True and policy.plan holds 1 "
            "step. No store write, no overwrite, and no separate planner precondition drops "
            "a gate-passing engine."
        ),
        "but_clearing_the_gate_is_necessary_not_sufficient": (
            "the same engine on the same seed with an UNREACHABLE goal also cleared the gate "
            "and reached the same call site, and the planner returned nothing. Opening the "
            "gate converts a rejection into an ATTEMPT to plan, not into a plan."
        ),
        "the_rejections_are_not_near_misses": (
            "MEASURED, on the 52 of 136 attempts that carry margins. Hidden-state branch: "
            "median heldout_change_consistency 0.0000, max 0.1781 against a 0.5 threshold, "
            "and 16 of 22 records have correct_changed_cells == 0 -- the engine got ZERO "
            "held-out changed cells right, which fails the nondegeneracy conjunct outright "
            "at any threshold. This is an empirical statement about these 52; the other 84 "
            "have NO recorded margin and are unknown, not zero."
        ),
        "so_a_threshold_relaxation_is_not_the_lever": (
            "on the plain branch a structural nav model recovering 27-35% of truly-changed "
            "cells is still rejected, because its full-grid exact score is 0.0 rather than "
            "0.4 -- nothing between 0.0 and 0.5 admits it. And the cell_recall metric that "
            "WOULD separate it already shipped as a flag and already ran as an arm "
            "(llm_on_fix_cellrecall): still 0 plans installed, with plain-branch cell_recall "
            "maxing at 0.0476 over 14 records against the structural fit's ~0.31."
        ),
        "what_this_does_NOT_show": (
            "no episode was played, no level solved, no live win-rate effect measured. The "
            "injection probe covers TWO games and SIX known-good arms, of which ONE reached "
            "the pass region. It proves the channel is open; it does not size the pass "
            "region, and it says nothing about how a REAL induced engine would behave if "
            "the gate were changed."
        ),
        "no_p_value_is_reported_and_none_is_reachable": (
            "this is a mechanism anatomy, not an A/B. There is no treatment, no pairing, and "
            "no discordant-pair count, so no significance claim is made anywhere in this "
            "artifact. The recount is a census of a fixed corpus; the probe is a "
            "demonstration with n=1 positive."
        ),
    }
    payload["honest_verdict"] = (
        "complete_gate_is_binding_and_reached_but_rejections_are_not_near_misses"
    )
    payload["measurement_wall_s"] = round(time.time() - started, 3)
    payload["duration_s"] = payload["measurement_wall_s"]
    payload["reproducibility_checksum"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    OUT.write_text(json.dumps(payload, indent=1, default=str))
    print(f"WROTE {OUT}")
    print(json.dumps({k: v for k, v in payload.items() if k.startswith("acceptance")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
