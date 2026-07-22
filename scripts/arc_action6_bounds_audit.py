#!/usr/bin/env python3
"""ARC-AGI-3 generalization-testing floor task (class 2): harden arc_solver_kit.reproduce()
against the offline/live ACTION6 bounds gap, and audit all 25 banked submission trajectories
against the new check.

**Researcher summary:**
    The offline arcade (arc_agi's LocalEnvironmentWrapper) silently accepts and routes any
    ACTION6 click, including coordinates outside the [0,63]x[0,63] range the live
    arcprize.org API enforces. A solve route can therefore reproduce cleanly offline while
    being un-submittable live -- this happened for real to lf52's original L9 route (22
    out-of-bounds clicks, x up to 132), discovered only at live-submission time and fixed
    reactively (commit 5ca2a999b, 2026-07-17). `arc_solver_kit.reproduce()` -- THE canonical
    offline reproduction gate every future solve is supposed to pass through per CLAUDE.md's
    "ARC Solve Reproducibility Discipline" -- now flags this class of gap itself, so a future
    solve for any game (not just lf52) is caught before it is ever promoted into a submission
    package. This script (a) unit-verifies the new check against a synthetic out-of-bounds
    case, then (b) real, non-mocked replays all 25 currently-banked submission trajectories
    through the hardened `reproduce()` to confirm none currently regress against the new
    check (a genuine measurement, not an assumption).

**Detailed explanation for engineers:**
    This is a "reusable-primitive hardening" task under CLAUDE.md's "ARC-AGI-3
    Generalization-Testing Floor" (task class 2): the gap was surfaced by a real prior
    measurement (the 2026-07-17 live re-validation, see ops/known-issues.md and
    project_arc_final_sprint_state.md memory), and the fix strengthens a shared primitive in
    arc_solver_kit.py -- not a per-game patch -- so it helps on ANY future game, adaptered or
    not. This is NOT a solve artifact (no new level is claimed for any game), so the
    solve_provenance contract in CLAUDE.md's "ARC Live-Path Reachability Discipline" does not
    apply -- explicitly noted rather than silently omitted.

Spec: openspec/capabilities/arc-world-model-trust-energy/spec.md REQ-ARC-WMTE-5820
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

from carnot.agentic import arc_solver_kit as kit  # noqa: E402
from carnot.agentic.arc_agi3_live_adapter import _game_action  # noqa: E402
from carnot.agentic.arc_game_adapters import _json_action_label, get_adapter  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANKED_DIR = os.path.join(REPO_ROOT, "results", "arc3_live_banked_trajectories")
OUTPUT_PATH = os.path.join(REPO_ROOT, "results", "outer_loop_action6_bounds_audit_20260722.json")


def _generic_apply(env, label, frame):
    """Raw-replay apply used for ALL 25 games (see `_apply_for`'s docstring for why per-game
    GameAdapter.apply is deliberately NOT used here).

    Mirrors the pattern every real adapter's own `apply` already uses internally (e.g.
    arc_game_adapters.py's `_lf52`/`_re86`): decode the JSON label, dispatch the action id +
    data straight to env.step(). No per-game hand-tuning required for a pure bounds audit.
    """
    del frame
    step = json.loads(label)
    from arcengine.enums import GameAction

    return env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))


def _apply_for(game: str):
    """Always use the generic raw-replay apply, never a game's own GameAdapter.apply.

    Discovered by running this audit for real (not assumed up front): several adapters'
    apply() implementations parse their OWN internal search-label dialect (e.g. a bare
    action-id string, or a differently-templated click label) rather than the standard
    `_json_action_label` JSON-string format this script constructs from the raw banked
    `{"action":int,"data":{"x":..,"y":..}}` entries -- using get_adapter(game).apply here
    raised real errors for ar25/ka59 (int() on a raw JSON string) and cd82/lp85 (KeyError
    'x', a different label shape). _generic_apply directly interprets exactly the format
    this script produces, uniformly, for every game -- it doesn't need any per-game
    adapter's own idiosyncratic label parsing, which exists to interpret THAT adapter's own
    internally-generated search labels, not externally-supplied raw action replay.
    """
    del game
    return _generic_apply


def _labels_from_bank(raw_solution: list[dict]) -> tuple[list[str], bool]:
    """Convert a banked trajectory's structured entries into _json_action_label strings.

    Most games' banks are {"action": int, "data": {"x":.., "y":..}}. lp85's bank is
    ACTION6-only and omits the "action" key entirely (every entry is bare {"x":.., "y":..}) --
    confirmed by direct inspection, not assumed; handled explicitly here rather than papering
    over a KeyError, and the fallback is reported back via the returned bool so it shows up in
    the artifact rather than being silently applied.
    """
    used_implicit_action6_fallback = False
    labels = []
    for entry in raw_solution:
        if "action" in entry:
            labels.append(_json_action_label(entry["action"], entry.get("data")))
        elif {"x", "y"} <= entry.keys():
            used_implicit_action6_fallback = True
            labels.append(_json_action_label(6, {"x": entry["x"], "y": entry["y"]}))
        else:
            raise ValueError(f"unrecognized banked trajectory entry shape: {entry!r}")
    return labels, used_implicit_action6_fallback


def _run_synthetic_regression_check() -> dict:
    """Confirm the new gate actually detects a deliberately out-of-bounds click before
    trusting the audit below -- a check that would silently pass 'all clean' for the wrong
    reason (a broken detector) is worse than no check at all."""
    solution = [
        _json_action_label(1),
        _json_action_label(6, {"x": 999, "y": 5}),
    ]
    result = kit.reproduce("lf52", solution, get_adapter("lf52").apply, claimed_level=None)
    ok = (
        result["checked_action6_clicks"] == 1
        and result["any_oob_action6_clicks"] is True
        and result["oob_action6_clicks"] == [{"index": 1, "x": 999, "y": 5}]
    )
    return {"synthetic_oob_detected": ok, "raw_result": result}


def main() -> None:
    t0 = time.monotonic()

    synthetic = _run_synthetic_regression_check()
    if not synthetic["synthetic_oob_detected"]:
        artifact = {
            "experiment": "outer_loop_action6_bounds_audit_20260722",
            "schema": "carnot.arc_action6_bounds_audit.v1",
            "honest_verdict": "blocked_synthetic_regression_check_failed_detector_broken",
            "duration_s": round(time.monotonic() - t0, 3),
            "synthetic_regression_check": synthetic,
        }
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(artifact, f, indent=2, sort_keys=True)
        print("BLOCKED: synthetic regression check failed -- see", OUTPUT_PATH)
        sys.exit(1)

    banked_games = sorted(f[:-5] for f in os.listdir(BANKED_DIR) if f.endswith(".json"))
    per_game = []
    total_checked_clicks = 0
    total_oob_clicks = 0
    games_with_no_adapter = []

    for game in banked_games:
        with open(os.path.join(BANKED_DIR, game + ".json")) as f:
            bank = json.load(f)
        raw_solution = bank["solution"]
        labels, used_implicit_action6_fallback = _labels_from_bank(raw_solution)

        adapter = get_adapter(game)
        if adapter is None:
            games_with_no_adapter.append(game)
        apply_fn = _apply_for(game)

        try:
            result = kit.reproduce(game, labels, apply_fn, claimed_level=None)
            entry = {
                "game": game,
                "banked_action_count": bank.get("action_count", len(raw_solution)),
                "used_implicit_action6_fallback": used_implicit_action6_fallback,
                "checked_action6_clicks": result["checked_action6_clicks"],
                "any_oob_action6_clicks": result["any_oob_action6_clicks"],
                "oob_action6_clicks": result["oob_action6_clicks"],
                "replay_error": None,
            }
            total_checked_clicks += result["checked_action6_clicks"]
            total_oob_clicks += len(result["oob_action6_clicks"])
        except Exception as exc:  # noqa: BLE001 - genuinely want to record ANY replay failure, not just OOB
            entry = {
                "game": game,
                "banked_action_count": bank.get("action_count", len(raw_solution)),
                "used_implicit_action6_fallback": used_implicit_action6_fallback,
                "checked_action6_clicks": None,
                "any_oob_action6_clicks": None,
                "oob_action6_clicks": None,
                "replay_error": f"{type(exc).__name__}: {exc}",
            }
        per_game.append(entry)

    duration_s = round(time.monotonic() - t0, 3)
    replay_errors = [e for e in per_game if e["replay_error"] is not None]
    any_corpus_oob = any(
        e["any_oob_action6_clicks"] for e in per_game if e["any_oob_action6_clicks"] is not None
    )

    checksum_input = json.dumps(
        [{"game": e["game"], "oob": e["oob_action6_clicks"]} for e in per_game], sort_keys=True
    ).encode()
    reproducibility_checksum = hashlib.sha256(checksum_input).hexdigest()

    if replay_errors:
        verdict = "complete_audit_ran_with_replay_errors_see_replay_errors_field"
    elif any_corpus_oob:
        verdict = "complete_hardening_shipped_and_corpus_audit_found_oob_clicks_needing_followup"
    else:
        verdict = "complete_hardening_shipped_and_all_25_banked_trajectories_clean"

    artifact = {
        "experiment": "outer_loop_action6_bounds_audit_20260722",
        "schema": "carnot.arc_action6_bounds_audit.v1",
        "run_date": "2026-07-22",
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "solve_provenance_note": (
            "N/A -- this is not a solve artifact. No new level is claimed for any game; this "
            "audits and hardens the reproduction-gate infrastructure itself (arc_solver_kit.reproduce())."
        ),
        "random_seed": None,
        "random_seed_note": (
            "Deterministic replay of pre-existing banked action sequences against the "
            "deterministic offline simulator; no RNG invoked by this driver."
        ),
        "duration_s": duration_s,
        "duration_note": (
            f"Real, non-mocked offline replay of all {len(banked_games)} banked trajectories "
            f"(sum {sum(e['banked_action_count'] for e in per_game)} actions) through the "
            "actual environment_files game simulators via arc_solver_kit.reproduce()."
        ),
        "honest_verdict": verdict,
        "narrative": (
            "The OFFLINE arcade (arc_agi LocalEnvironmentWrapper.step()) never validates "
            "ACTION6 click coordinates, even though the installed arcengine dependency already "
            "declares the live API's own bound (ComplexAction.x/y: Field(ge=0, le=63)) -- that "
            "validation is wired into the live HTTP handler only, never into the local/offline "
            "path. arc_solver_kit.reproduce() (THE canonical offline reproduction gate) now "
            "reuses arcengine's own bound to flag any ACTION6 click a solve route depends on "
            "that would be rejected live, additively (oob_action6_clicks / "
            "any_oob_action6_clicks / checked_action6_clicks fields) without changing existing "
            "reproduced/reached_level semantics. Verified working via a synthetic regression "
            "check (a deliberately out-of-bounds click IS detected) before trusting the corpus "
            "audit below."
        ),
        "synthetic_regression_check": synthetic,
        "corpus_audit": {
            "games_audited": len(banked_games),
            "games_with_no_registered_game_adapter": games_with_no_adapter,
            "total_action6_clicks_checked": total_checked_clicks,
            "total_oob_action6_clicks_found": total_oob_clicks,
            "any_corpus_oob_clicks_found": any_corpus_oob,
            "replay_errors": replay_errors,
            "per_game": per_game,
        },
        "reproducibility_checksum": reproducibility_checksum,
        "preconditions_checked": [
            {
                "resource": "arc_solver_kit_reproduce_action6_bounds_hardening_shipped",
                "available": True,
                "note": "python/carnot/agentic/arc_solver_kit.py _action6_click_from_label / "
                "_action6_out_of_live_bounds / reproduce() edited before this run.",
            },
            {
                "resource": "banked_trajectories_directory_present",
                "available": os.path.isdir(BANKED_DIR),
                "note": f"{len(banked_games)} game trajectory files found in {BANKED_DIR}.",
            },
        ],
        "acceptance_gates": [
            {
                "condition": "synthetic_oob_detected == true",
                "principle": "A regression check that would silently pass 'all clean' because "
                "the detector itself is broken is worse than no check at all -- this gate "
                "must fail loudly before the corpus audit below is trusted.",
                "passed": synthetic["synthetic_oob_detected"],
            }
        ],
        "field_provenance": {
            "oob_action6_clicks": {
                "principle": "Records exactly which banked actions would 400 live, so a "
                "future fix (like lf52's 2026-07-17 in-bounds route) has a precise target "
                "instead of a vague 'something in this route fails live'.",
            }
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print(f"Wrote {OUTPUT_PATH}")
    print(f"verdict: {verdict}")
    print(
        f"games_audited={len(banked_games)} total_action6_clicks_checked={total_checked_clicks} "
        f"total_oob_action6_clicks_found={total_oob_clicks} replay_errors={len(replay_errors)}"
    )


if __name__ == "__main__":
    main()
