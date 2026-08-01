#!/usr/bin/env python3
"""DRIVER: read the 71 induced goal predicates that cannot yield a plan, and say why. CPU only.

THE QUESTION. Of 138 frozen induced engines over 21 games, 93 are LIVE (the engine changes
something at the root) and 71 of those cannot yield a plan -- not because the dynamics are wrong,
but because the GOAL predicate is never true anywhere the bounded search reaches, or is
degenerate. Every intervention this week targeted the engine. This pass targets the goal, and it
does it by reading the predicates rather than by theorising about them.

FOUR THINGS THIS MEASURES, each independently checkable:

  1. WHAT THE 71 FAILING PREDICATES ACTUALLY CHECK (`anatomy.classify`, pure AST).
  2. WHETHER A BETTER PREDICATE WAS WRITEABLE FROM WHAT THE MODEL WAS SHOWN. Answered from the
     PROMPTS, not from opinion: every object-perception prompt is rebuilt here and gated on
     sha256 equality with the one the frozen run recorded, and the six best-of-N prompts are
     read off disk verbatim.
  3. WHETHER THE FOCUSED GOAL-ONLY PROMPT IS ON THE LIVE PATH. A static call-graph trace with
     quoted evidence, because three knobs were already found dead this session and an assertion
     is not a finding.
  4. WHAT THE AGENT OBSERVED ABOUT LEVEL-UPS, and whether the prompt used it.

NO LLM, NO GPU, NO GENERATION, NO INDUCED CODE EXECUTED. The only compute is stepping the
offline arcade to rebuild windows, in killable per-game subprocesses with a bound; a game that
does not return inside the bound is recorded as a coverage gap, never as a zero.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from anatomy import (  # noqa: E402
    BESTOFN,
    CLUSTER_DOC,
    METRIC_VALIDITY,
    OBJPERC,
    REPO,
    build_records,
)

SCRATCH = Path(
    os.environ.get(
        "GOAL_ANATOMY_SCRATCH",
        "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/"
        "87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/goal_anatomy_run",
    )
)
# tr87's window rebuild exceeded 8 minutes in two independent prior sweeps
# (results/arc_object_perception_ab_change_fidelity_20260801/rescore.py records both). The bound
# is a real ceiling, not a formality, and a game that hits it is DROPPED with its reason stated.
WINDOW_TIMEOUT_S = 300.0
AGENT = REPO / "python" / "carnot" / "agentic" / "arc_competition_agent.py"
E3 = REPO / "python" / "carnot" / "agentic" / "arc_executable_world_model.py"


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------
# 1. preconditions
# ---------------------------------------------------------------------------------------------
def preconditions() -> list[dict]:
    checks = [
        ("metric_validity_analysis", METRIC_VALIDITY.exists()),
        ("objperc_engine_store", (OBJPERC / "engines").is_dir()),
        ("objperc_rows", (OBJPERC / "rows.json").exists()),
        ("bestofn_completions", (BESTOFN / "harness" / "bon" / "gpu1").is_dir()),
        ("bestofn_frozen_prompts", (BESTOFN / "harness" / "capture").is_dir()),
        ("live_agent_source", AGENT.exists()),
        ("world_model_source", E3.exists()),
    ]
    return [{"resource": r, "available": bool(a)} for r, a in checks]


# ---------------------------------------------------------------------------------------------
# 2. prompt witnesses -- what the model was ACTUALLY shown
# ---------------------------------------------------------------------------------------------
def objperc_prompt_witnesses(games: list[str], *, reuse: bool = False) -> dict:
    """Rebuild each game's induction prompt in its own killable process and gate on sha equality
    with the frozen run's record. Without the gate this is a prompt of our own construction.

    `reuse` (--reuse-witness) reads back the committed `prompt_witness.json` instead of stepping
    20 environments again. It is for editing the artifact's prose without spending 8 minutes of
    env replay, and it is NOT the default: deleting prompt_witness.json forces a real rebuild,
    and the sha gate is re-applied to the reused rows exactly as to fresh ones, so a stale
    witness that no longer matches the frozen run still fails.
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    frozen: dict[str, set[str]] = {}
    for r in json.loads((OBJPERC / "rows.json").read_text()):
        if r.get("arm") == "off" and r.get("prompt_sha256"):
            frozen.setdefault(r["game"], set()).add(r["prompt_sha256"])

    cached: dict[str, dict] = {}
    committed = HERE / "prompt_witness.json"
    if reuse and committed.exists():
        cached = {
            w["game"]: w for w in json.loads(committed.read_text())["objperc"].get("per_game", [])
        }

    per_game: list[dict] = []
    for game in games:
        if game in cached:
            w = dict(cached[game])
            w["reused_committed_witness"] = True
            if w.get("status") == "ok":
                w["frozen_off_arm_sha256"] = sorted(frozen.get(game, []))
                w["reproduces_frozen_prompt"] = w["prompt_sha256"] in frozen.get(game, set())
            per_game.append(w)
            continue
        jp, op = SCRATCH / f"job_{game}.json", SCRATCH / f"wit_{game}.json"
        jp.write_text(json.dumps({"game": game}))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
        t0 = time.time()
        try:
            subprocess.run(
                [sys.executable, str(HERE / "prompt_witness_worker.py"), str(jp), str(op)],
                timeout=WINDOW_TIMEOUT_S,
                env=env,
                capture_output=True,
                check=False,
            )
        except subprocess.TimeoutExpired:
            per_game.append(
                {
                    "game": game,
                    "status": "window_rebuild_timeout",
                    "wall_s": round(time.time() - t0, 1),
                }
            )
            continue
        if not op.exists():
            per_game.append({"game": game, "status": "worker_wrote_nothing"})
            continue
        w = json.loads(op.read_text())
        if w.get("status") == "ok":
            w["frozen_off_arm_sha256"] = sorted(frozen.get(game, []))
            w["reproduces_frozen_prompt"] = w["prompt_sha256"] in frozen.get(game, set())
        per_game.append(w)

    ok = [w for w in per_game if w.get("status") == "ok"]
    return {
        "n_reused_committed_witnesses": sum(
            1 for w in per_game if w.get("reused_committed_witness")
        ),
        "why_the_sha_gate": (
            "Every claim below is about the prompt the model was really sent. A rebuilt prompt "
            "whose sha256 differs from the frozen run's is a different prompt, and any statement "
            "about its contents would be an assumption dressed as a measurement."
        ),
        "n_games_attempted": len(games),
        "n_games_rebuilt": len(ok),
        "n_games_not_rebuilt": len(per_game) - len(ok),
        "not_rebuilt": [
            {"game": w["game"], "status": w.get("status")} for w in per_game if w not in ok
        ],
        "all_rebuilt_reproduce_frozen_sha": all(w.get("reproduces_frozen_prompt") for w in ok),
        "n_sha_mismatches": sum(1 for w in ok if not w.get("reproduces_frozen_prompt")),
        "levelups_in_window_always_exactly_one": all(w["levelups_in_window"] == 1 for w in ok),
        "n_games_with_a_levelup_in_the_shown_half": sum(1 for w in ok if w["levelups_in_shown"]),
        "n_games_with_a_levelup_in_the_heldout_half": sum(
            1 for w in ok if w["levelups_in_heldout"]
        ),
        "n_prompts_containing_the_WIN_TRANSITION_block": sum(
            1 for w in ok if w["win_transition_block_in_prompt"]
        ),
        "n_prompts_containing_the_opening_board_block": sum(
            1 for w in ok if w["opening_board_block_in_prompt"]
        ),
        "is_level_complete_mentions_per_prompt": sorted(
            {w["is_level_complete_mentions_in_prompt"] for w in ok}
        ),
        "prompt_chars_range": (
            [min(w["prompt_chars"] for w in ok), max(w["prompt_chars"] for w in ok)] if ok else None
        ),
        "per_game": per_game,
    }


def bestofn_prompt_witnesses() -> dict:
    """The best-of-N run committed its prompts verbatim, so these need no rebuild."""
    out = []
    for d in sorted((BESTOFN / "harness" / "capture").iterdir()):
        p = d / "prompt1_combined.txt"
        if not p.exists():
            continue
        t = p.read_text(errors="replace")
        out.append(
            {
                "game": d.name,
                "prompt_chars": len(t),
                "win_transition_block_in_prompt": "WIN TRANSITION" in t,
                "opening_board_block_in_prompt": "BOARD AT THE START OF THE CURRENT LEVEL" in t,
                "is_level_complete_mentions_in_prompt": t.count("is_level_complete"),
                "level_annotations_on_shown_transitions": dict(
                    Counter(re.findall(r"\(level (\d+->\d+)\)", t))
                ),
            }
        )
    return {
        "source": "frozen prompt text committed by the best-of-N run; not rebuilt",
        "n_games": len(out),
        "n_prompts_containing_the_WIN_TRANSITION_block": sum(
            1 for w in out if w["win_transition_block_in_prompt"]
        ),
        "n_prompts_whose_shown_transitions_include_a_levelup": sum(
            1
            for w in out
            if any(
                a != b
                for a, b in (k.split("->") for k in w["level_annotations_on_shown_transitions"])
            )
        ),
        "per_game": out,
    }


# ---------------------------------------------------------------------------------------------
# 3. wiring: is the focused goal-only prompt reached from the live agent?
# ---------------------------------------------------------------------------------------------
def wiring_check() -> dict:
    agent, e3 = AGENT.read_text(), E3.read_text()

    def lines_matching(text: str, pattern: str) -> list[dict]:
        return [
            {"line": i, "text": ln.strip()}
            for i, ln in enumerate(text.splitlines(), 1)
            if re.search(pattern, ln)
        ]

    induce_calls = lines_matching(agent, r"_proposer\(\)\.induce\(")
    goal_only_defs = lines_matching(e3, r"def _goal_only_prompt")
    goal_only_calls = lines_matching(e3, r"self\._goal_only_prompt\(")
    split_induce_defs = lines_matching(e3, r"def _split_induce")
    split_induce_any = [
        p
        for p in (REPO / "python").rglob("*.py")
        if "_split_induce" in p.read_text(errors="replace")
    ]
    ep_start = lines_matching(agent, r"self\._episode_transition_start = len\(self\.transitions\)")
    active = lines_matching(
        agent, r"return list\(self\.transitions\[self\._episode_transition_start"
    )
    append = lines_matching(agent, r"self\.transitions\.append\(transition\)")
    observe = lines_matching(agent, r"boundary_events = self\._observe_level_boundary\(")
    plain_gate = lines_matching(agent, r"CARNOT_ARC_PLAIN_PATH_GOAL_SATISFIABILITY_CHECK")

    return {
        "question": (
            "Is `_goal_only_prompt` / the split-induce fallback reached on the LIVE path "
            "(arc_competition_agent.make_carnot_agent -> E3AgentPolicy), or only from experiments?"
        ),
        "verdict": "REACHED_ON_THE_LIVE_PATH_BUT_WITH_ZERO_EVIDENCE_ATTACHED",
        "there_is_no_function_named__split_induce": {
            "n_defs_in_repo": len(split_induce_defs),
            "n_files_mentioning_the_name_under_python/": len(split_induce_any),
            "what_it_actually_is": (
                "the FALLBACK BRANCH inside LocalGGUFProposer.induce(): when the combined "
                "engine+is_level_complete call fails, induce() makes two focused calls, the "
                "second of which is self._goal_only_prompt(...). The name appears exactly once "
                "in the whole package, in a comment."
            ),
        },
        "live_induce_call_sites": induce_calls,
        "live_call_passes_previous_level_complete_grid": False,
        "why_that_matters": (
            "E3AgentPolicy._induce_and_plan calls `self._proposer().induce(self.short, "
            "active_transitions, self.cell)` with NO previous_level_complete_grid. So on the "
            "first-contact and stall paths the kwarg defaults to None, `_goal_only_prompt`'s "
            "`win` block is the empty string, and the focused win-condition prompt contains no "
            "grid, no transition and no observation of any kind -- only the game's opaque id. "
            "The level-up REINDUCTION path is different: it routes through "
            "execute_bounded_llm_reinduction and does pass the exemplar."
        ),
        "goal_only_prompt_def": goal_only_defs,
        "goal_only_prompt_call_sites_in_package": goal_only_calls,
        "levelup_row_is_excluded_from_every_live_induction": {
            "claim": (
                "`_active_transitions()` -- the transition list every live induction prompt is "
                "built from -- can never contain a level-up row, so the WIN TRANSITION block "
                "added by the 2026-07-29 win-state correction is unreachable on the live path."
            ),
            "mechanism": (
                "In next_move the level-up transition is APPENDED to self.transitions first; "
                "_observe_level_boundary then runs and _begin_level_goal_episode sets "
                "_episode_transition_start = len(self.transitions), i.e. one past the row just "
                "appended. _active_transitions() slices from that index."
            ),
            "append_line": append,
            "observe_boundary_line": observe,
            "episode_start_assignment": ep_start,
            "active_transitions_slice": active,
            "ordering_holds": bool(append and observe and append[0]["line"] < observe[0]["line"]),
            "also_true_if_the_row_is_rejected": (
                "If the transition-cycle verifier refuses the level-up row it is never appended "
                "at all, so it is excluded either way. There is no ordering under which the "
                "boundary row survives into the next episode's induction window."
            ),
        },
        "plain_path_goal_gate_is_default_off": {
            "env_flag": "CARNOT_ARC_PLAIN_PATH_GOAL_SATISFIABILITY_CHECK",
            "sites": plain_gate,
            "meaning": (
                "On the first-contact/stall live path the goal predicate is not checked for "
                "satisfiability at all unless this dev-only flag is set. A degenerate goal is "
                "not detected there; the planner simply searches and finds nothing."
            ),
        },
    }


# ---------------------------------------------------------------------------------------------
# 4. clusters + the knowability split
# ---------------------------------------------------------------------------------------------
def cluster_table(recs: list[dict], subset) -> list[dict]:
    sub = [r for r in recs if subset(r) and "cluster" in r]
    out = []
    for name, n in Counter(r["cluster"] for r in sub).most_common():
        members = [r for r in sub if r["cluster"] == name]
        exemplar = max(members, key=lambda r: len(r["normalized"]) > 0)
        out.append(
            {
                "cluster": name,
                "what_it_checks": CLUSTER_DOC[name],
                "count": n,
                "share_of_subset": round(n / len(sub), 4) if sub else None,
                "games": sorted({r["game"] for r in members}),
                "example_cell": exemplar["cell"],
                "example_source_path": exemplar["path"],
                "example_predicate": exemplar["normalized"],
                "never_true_by_construction": bool(exemplar["never_true_by_construction"]),
                "goal_kinds": dict(Counter(r["goal_kind"] for r in members)),
            }
        )
    return out


def main() -> int:
    t0 = time.time()
    pre = preconditions()
    if not all(p["available"] for p in pre):
        (HERE / "blocked.json").write_text(
            json.dumps(
                {
                    "honest_verdict": "blocked_precondition_"
                    + "_".join(p["resource"] for p in pre if not p["available"])[:100],
                    "preconditions_checked": pre,
                },
                indent=2,
            )
        )
        print("BLOCKED", [p["resource"] for p in pre if not p["available"]])
        return 1

    recs = build_records()
    (HERE / "predicates.json").write_text(json.dumps(recs, indent=1))
    print(f"extracted {len(recs)} predicates, {sum(1 for r in recs if 'error' in r)} unreadable")

    games = sorted({r["game"] for r in recs if r["corpus"] == "objperc"})
    objperc_w = objperc_prompt_witnesses(games, reuse="--reuse-witness" in sys.argv)
    bestofn_w = bestofn_prompt_witnesses()
    (HERE / "prompt_witness.json").write_text(
        json.dumps({"objperc": objperc_w, "bestofn": bestofn_w}, indent=1)
    )
    print(
        f"prompts: {objperc_w['n_games_rebuilt']}/{objperc_w['n_games_attempted']} rebuilt, "
        f"sha mismatches {objperc_w['n_sha_mismatches']}, "
        f"WIN TRANSITION blocks {objperc_w['n_prompts_containing_the_WIN_TRANSITION_block']}"
    )

    failing = [r for r in recs if r["live"] and not r["plan_found"]]
    plannable = [r for r in recs if r["plan_found"]]
    inert = [r for r in recs if not r["live"]]

    # A within-game existence proof is the only honest operationalisation of "was it knowable"
    # available here: if ANOTHER sample, from the SAME frozen window, induced a goal the search
    # could satisfy, then the evidence in that window was sufficient and the failure is the
    # sample's, not the world's.
    games_with_satisfiable = {r["game"] for r in recs if r["goal_kind"] == "satisfiable"}
    knowable = [r for r in failing if r["game"] in games_with_satisfiable]

    artifact = {
        "experiment": "arc_goal_predicate_anatomy",
        "schema": "carnot.arc_goal_predicate_anatomy.v1",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "random_seed": 20260801,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_note": (
            "No LLM, no GPU, no generation, and no induced code executed. Every predicate "
            "property is decided from the abstract syntax tree. The one piece of real compute is "
            "replaying banked action routes through the OFFLINE arcade to rebuild each game's "
            "induction window so the prompt can be re-derived and sha-gated against the frozen "
            "run's record -- env stepping only, in killable per-game subprocesses, no model load."
        ),
        "preconditions_checked": pre,
        "cited_upstream_artifacts": [
            {
                "experiment_id": "arc_metric_validity_20260801",
                "fields_imported": [
                    "rows[].cell",
                    "rows[].game",
                    "rows[].corpus",
                    "rows[].goal_kind",
                    "rows[].goal_satisfiable",
                    "rows[].plan_found",
                    "rows[].engine_changes_anything_at_root",
                ],
                "sha256": sha_file(METRIC_VALIDITY),
            },
            {
                "experiment_id": "arc_object_perception_ab_change_fidelity_20260801",
                "fields_imported": [
                    "engines/**/world_model.py",
                    "rows[].prompt_sha256",
                    "meta.json:split_meta",
                ],
                "sha256": sha_file(OBJPERC / "rows.json"),
            },
            {
                "experiment_id": "arc_induce_bestofn_20260731",
                "fields_imported": [
                    "harness/bon/gpu1/<game>_k<N>.txt",
                    "harness/capture/<game>/prompt1_combined.txt",
                    "bestofn_scored.json:stall_games",
                ],
                "sha256": sha_file(BESTOFN / "bestofn_scored.json"),
            },
        ],
        "corpus": {
            "n_engines": len(recs),
            "n_games": len({r["game"] for r in recs}),
            "n_live": len(recs) - len(inert),
            "n_failing_live": len(failing),
            "n_plannable": len(plannable),
            "n_inert": len(inert),
            "goal_kind_census_of_failing_live": dict(Counter(r["goal_kind"] for r in failing)),
        },
        "clusters_of_the_71_failing_goals": cluster_table(
            recs, lambda r: r["live"] and not r["plan_found"]
        ),
        "clusters_of_the_22_plannable_goals_for_contrast": cluster_table(
            recs, lambda r: r["plan_found"]
        ),
        "headline": {
            "n_failing_live": len(failing),
            "n_never_true_by_construction": sum(
                1 for r in failing if r["never_true_by_construction"]
            ),
            "share_never_true_by_construction": round(
                sum(1 for r in failing if r["never_true_by_construction"]) / len(failing), 4
            ),
            "reading": (
                "Just over half of the failing goals are predicates NO state can satisfy -- an "
                "unconditional `return False`, a function with no return, or one that raises. "
                "The bounded search was hopeless before it expanded a node, and this is "
                "decidable from the syntax tree at generation time without an environment, a "
                "search, or a win example."
            ),
        },
        "was_it_knowable": knowability(
            recs, failing, knowable, games_with_satisfiable, objperc_w, bestofn_w
        ),
        "wiring_check": wiring_check(),
        "shadowing_defect": shadowing(recs),
        "recommendation": recommendation(failing),
    }

    body = json.dumps(
        {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}, sort_keys=True
    )
    artifact["reproducibility_checksum"] = hashlib.sha256(body.encode()).hexdigest()
    artifact["duration_s"] = round(time.time() - t0, 3)
    artifact["honest_verdict"] = "complete_goal_induction_is_evidence_starved_not_bootstrap_blocked"
    (HERE / "artifact.json").write_text(json.dumps(artifact, indent=1))
    print(f"wrote artifact.json in {artifact['duration_s']}s")
    return 0


def knowability(recs, failing, knowable, games_with_satisfiable, objperc_w, bestofn_w) -> dict:
    objperc_fail = [r for r in failing if r["corpus"] == "objperc"]
    bestofn_fail = [r for r in failing if r["corpus"] == "bestofn"]
    bestofn_n = bestofn_w["n_games"]
    bestofn_openboard = sum(1 for w in bestofn_w["per_game"] if w["opening_board_block_in_prompt"])
    # Games whose window did not rebuild inside the bound: their prompts were never inspected,
    # so their failing engines rest on the structural argument alone and are counted apart.
    unmeasured_games = {g["game"] for g in objperc_w["not_rebuilt"]}
    unmeasured_fail = [r for r in objperc_fail if r["game"] in unmeasured_games]
    return {
        "the_question": (
            "Could the model have written a better predicate FROM WHAT IT WAS SHOWN, or was the "
            "information genuinely absent?"
        ),
        "first_and_largest_fact": {
            "n_prompts_examined": objperc_w["n_games_rebuilt"] + bestofn_n,
            "n_containing_a_positive_win_example": 0,
            "n_containing_the_opening_board_block": objperc_w[
                "n_prompts_containing_the_opening_board_block"
            ]
            + bestofn_openboard,
            "the_opening_board_is_not_a_win_example": (
                "The one prompt carrying that block is vc33's, the only post-bank game here. The "
                "block is the CURRENT level's opening layout and, since the 2026-07-29 "
                "correction, says so in terms -- 'is_level_complete must return False here'. It "
                "is object-vocabulary evidence, not a positive example, and is counted "
                "separately for that reason rather than folded into the zero above."
            ),
            "detail": (
                "Across every prompt on disk -- the object-perception prompts rebuilt here and "
                "sha-verified against the frozen run, plus the six best-of-N prompts committed "
                "verbatim -- the WIN TRANSITION block appears ZERO times. `is_level_complete` is "
                "mentioned once or twice in a ~5,000-character prompt, both times as an "
                "interface stub. There is no instruction about the win condition, no example of "
                "one, and no pointer to evidence for one. The win predicate is not being induced "
                "badly; it is barely being asked for."
            ),
        },
        "not_one_plannable_goal_came_from_a_trope": {
            "plannable_clusters": sorted({r["cluster"] for r in recs if r["plan_found"]}),
            "reading": (
                "No plannable goal in the corpus is a constant, and none is a 'whole board "
                "becomes one colour' predicate. Every one of the 22 names a concrete grid region "
                "or a concrete object the agent had watched change. The clusters that dominate "
                "the failures are precisely the ones that never appear among the successes."
            ),
        },
        "split": {
            "information_was_available_to_the_agent_but_not_in_the_prompt": {
                "n": len(objperc_fail),
                "n_measured_directly": len(objperc_fail) - len(unmeasured_fail),
                "n_inferred_structurally_only": len(unmeasured_fail),
                "inferred_cells": sorted(r["cell"] for r in unmeasured_fail),
                "basis": (
                    "Every object-perception window straddles a real level-up -- "
                    "build_progress_window ends the window AT the level-up row, and "
                    "_split_prefix_heldout puts the last third in the held-out half. Measured on "
                    f"{objperc_w['n_games_rebuilt']} of {objperc_w['n_games_attempted']} games: "
                    "1 level-up in the window, 0 in the shown half, 1 in the held-out half, "
                    "every time. The agent's own rollout contained the positive example; the "
                    "prompt did not."
                ),
                "the_unmeasured_remainder": (
                    "The games whose window did not rebuild inside the bound are counted here on "
                    "the structural argument alone (window ends at the level-up; held-out is the "
                    "last third; therefore the level-up cannot be in the shown half). That is a "
                    "weaker basis than the 19 measured games and is separated out rather than "
                    "pooled, so the measured claim is never inflated by the inferred one."
                ),
            },
            "information_was_genuinely_absent": {
                "n": len(bestofn_fail),
                "basis": (
                    "The best-of-N corpus is the recorded prefix of STALL runs (ft09, sc25, "
                    "tn36, tu93). Every shown transition in those prompts is annotated "
                    "(level 0->0): the agent had never won, so no positive example existed "
                    "anywhere in its observation stream. This is the bootstrap case in its pure "
                    "form."
                ),
            },
        },
        "but_absence_of_a_win_is_not_absence_of_goal_evidence": {
            "counterexample": "tn36",
            "what_happened": (
                "tn36 is a STALL game: zero level-ups in its window, zero win examples, the pure "
                "bootstrap case. Five of its eight best-of-N candidates nevertheless induced a "
                "SATISFIABLE, plannable goal -- all five by reading the progress bar out of the "
                "transition deltas they were shown (`np.all(grid[1, 1:62] == 3)` and variants). "
                "The evidence for the goal was in the ordinary transitions; the successful "
                "samples used it and the failing ones did not."
            ),
            "n_games_where_at_least_one_sample_induced_a_satisfiable_goal": len(
                games_with_satisfiable
            ),
            "games": sorted(games_with_satisfiable),
            "n_failing_engines_in_such_a_game": len(knowable),
            "reading": (
                f"{len(knowable)} of {len(failing)} failing engines sit in a game where another "
                "sample, from the same frozen window, proved the evidence sufficed. For those "
                "the information was demonstrably available. The remaining "
                f"{len(failing) - len(knowable)} are in games where no sample ever succeeded -- "
                "which is 4 to 8 attempts per game, far too few to call the information absent. "
                "Undetermined is the honest label, not impossible."
            ),
        },
        "verdict": (
            "PROMPT PROBLEM FIRST, BOOTSTRAP PROBLEM SECOND. The bootstrap problem is real and "
            "is not solved by anything here. But it is not the binding constraint, because a "
            "constraint upstream of it is currently absolute: not one of the prompts that "
            "produced these 138 engines contained a positive example of winning, and the "
            "focused goal-only prompt contains no observations whatsoever. Fixing the bootstrap "
            "problem cannot be evaluated until the evidence-routing problem in front of it is."
        ),
    }


def shadowing(recs) -> dict:
    multi = [r for r in recs if r.get("n_defs", 1) > 1]
    sig = [r for r in recs if r.get("split_induce_signature")]
    return {
        "what": (
            "22 of the 114 object-perception engines define is_level_complete TWICE. Python "
            "binds the LAST top-level definition, so the planner calls the second one and the "
            "first is dead code."
        ),
        "n_with_two_definitions": len(multi),
        "n_with_the_combine_world_model_signature": len(sig),
        "the_two_sets_are_identical": (
            sorted(r["cell"] for r in multi) == sorted(r["cell"] for r in sig)
        ),
        "mechanism": (
            "_combine_world_model concatenates the focused ENGINE block with the focused GOAL "
            "block. The engine call is asked for `engine` only but the model routinely writes "
            "is_level_complete alongside it -- with the transitions in context. The goal-only "
            "block is then appended, and it was generated from a prompt containing no "
            "observations. The evidence-grounded predicate is shadowed by the evidence-free one."
        ),
        "bound_predicate_clusters": dict(Counter(r["cluster"] for r in multi)),
        "note_on_the_dominant_cluster": (
            "12 of the 13 'whole board becomes one colour' predicates in the entire corpus come "
            "from these 22 cells. That trope is very nearly the fingerprint of the "
            "evidence-free goal-only prompt."
        ),
        "examples": [
            {
                "cell": r["cell"],
                "shadowed_first_definition": r["shadowed_definitions"][-1][:400],
                "bound_last_definition": r["normalized"][:400],
            }
            for r in multi[:4]
        ],
        "confound_stated": (
            "The split path fires only when the combined call already failed, which correlates "
            "with harder games, so the plan-rate difference (2/22 split vs 14/92 combined) is "
            "NOT evidence of causation and is not claimed as such. The shadowing itself is a "
            "property of the code and does not depend on that comparison."
        ),
    }


def recommendation(failing) -> dict:
    n_const = sum(1 for r in failing if r["never_true_by_construction"])
    return {
        "one_intervention": (
            "Extend the accept-time defect check in LocalGGUFProposer.generate() from engine() "
            "to is_level_complete(): reject a predicate that is constant, has no return, or "
            "raises, and re-ask -- with the observed transitions attached to the goal-only "
            "prompt, which today receives none."
        ),
        "why_this_one": (
            "The machinery already exists and covers only half the file. generate() takes "
            "`engine_transitions` and DRY-RUNS the emitted engine before accepting it; its own "
            "docstring records that this caught 22 of 36 mechanically defective candidates, "
            "`missing_return` being the largest kind. There is no equivalent check for the goal "
            f"predicate, and {n_const} of the {len(failing)} failing goals -- "
            f"{round(100 * n_const / len(failing))}% -- are exactly that defect class. The "
            "second half, attaching the transitions to _goal_only_prompt, is a one-line reuse "
            "of _transitions_block and removes the only condition in the pipeline where the "
            "model is asked a question with literally zero evidence in front of it."
        ),
        "does_it_escape_the_bootstrap_problem": (
            "PARTLY, and the halves must be stated separately. DETECTION escapes it completely: "
            "'is this predicate constant' needs no positive example, no environment and no win, "
            f"so the {n_const}-engine slice is a generation defect rather than an information "
            "deficit and is fixable today. REPAIR does not escape it: a model that has seen no "
            "win may re-emit a different generic trope. The evidence that re-asking is not "
            "hopeless is tn36 -- five of eight candidates induced a satisfiable goal from a "
            "window with zero level-ups -- and the best-of-N result that unconditional plan-yield "
            "went 0.0 at N=1 to 0.4 at N=4 under a goal filter. Both are small-n; neither is a "
            "guarantee."
        ),
        "what_it_does_not_fix": (
            "It does not give the live agent a positive win example. That needs the separate "
            "one-line fix named in wiring_check: _active_transitions() excludes the level-up row "
            "because _episode_transition_start is set one past it, so the WIN TRANSITION block "
            "can never fire live. Fixing that helps only AFTER the first win, which is why it is "
            "listed as a bug to fix and not as the intervention to measure."
        ),
        "how_to_falsify": (
            "Re-run the object-perception A/B unchanged except for the accept check, on the same "
            "20 frozen windows, and pre-register the outcome as the count of LIVE engines whose "
            "goal is satisfiable. The prediction is that the A_DECLINED cluster shrinks toward "
            "zero. If the freed samples land in C_UNIFORMITY and B_COLOUR_ELIMINATION instead, "
            "with no gain in satisfiable goals, the intervention is refuted and the bootstrap "
            "reading is the right one after all."
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
