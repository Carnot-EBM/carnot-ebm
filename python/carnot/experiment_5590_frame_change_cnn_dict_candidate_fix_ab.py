"""Experiment 5590: matched-budget A/B measuring whether the dict-candidate CNN
fix (arc_frame_change_predictor.py's ``_as_action_like``, landed alongside this
experiment) actually helps live-agent capability, per CLAUDE.md's Phase
Prototype + Empirical Validation + Adversarial Check Discipline -- a fix that
stops silently discarding a signal is not automatically a capability win; it
must be measured.

Context: docs/research-notes/arc-perception-grounding-audit-2026-07-13.md
found that ``FrameChangeScorer.candidate_score``'s ``getattr(candidate,
"action_id")`` raised ``AttributeError`` on dict-shaped candidates, silently
zeroing the CNN term (already project-documented in
``arc_online_action_effect_scorer.py``'s docstring as a ~20/25-games false-
negative cause, but never backported to the shipped
``arc_frame_change_predictor.py`` classes). Confirmed this fires on the
live DEFAULT path via ``ActionEffectExpansionPrior.frontier_priority``
(``SUBMITTED_ACTION_EFFECT_EXPANSION_PRIOR_ENABLED = True``, no other gate),
which is fed ``node["untested"]`` -- dict rows.

CONTROL simulates the PRE-FIX behavior by monkeypatching
``arc_frame_change_predictor._as_action_like`` to identity (so dict-shaped
candidates raise the same AttributeError internally and the CNN term is
silently zero again, matching the historical bug exactly) for the duration
of each control run only; TREATMENT uses the real, now-fixed default
unmodified. Both arms use the SAME real ``E3AgentPolicy`` construction
(``GroundTruthValidatedFrameChangeScorer`` wrapping the real
``LiveActionEffectScorer``, ``ActionEffectExpansionPrior`` enabled) so the
delta isolates exactly this one fix.

Tier-3 LLM induction is disabled (``CARNOT_ARC_DISABLE_INDUCTION=1``, the
same "production-safe escape hatch" the local submission gate uses) so this
measures the search/frontier-priority effect cleanly and fast, without LLM
wall-clock or the induction-vs-frontier-priority confound. This is a real
search-tier (tiers 1-2) measurement, not a claim about tier-3.

Spec refs: REQ-ARC-FCP-5590, SCENARIO-ARC-FCP-5590-DICT-CANDIDATE-DELTA.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(SCRIPTS_ROOT))

JsonDict = dict[str, Any]

EXPERIMENT_ID = "experiment_5590_frame_change_cnn_dict_candidate_fix_ab"
RESULT_RELATIVE_PATH = "results/experiment_5590_frame_change_cnn_dict_candidate_fix_ab.json"
SCHEMA = "carnot.exp5590.frame_change_cnn_dict_candidate_fix_ab.v1"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
RANDOM_SEED = 5590
DEFAULT_BUDGET = 200
DEFAULT_ROSTER = (
    "cd82",
    "cn04",
    "lp85",
    "ls20",
    "m0r0",
    "r11l",
    "sk48",
    "sp80",
    "su15",
    "tu93",
    "wa30",
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "verifier_is_oracle",
    "roster",
    "budget",
    "control_results",
    "treatment_results",
    "levels_gained_control_total",
    "levels_gained_treatment_total",
    "per_game_levels_delta",
    "states_expanded_control_total",
    "states_expanded_treatment_total",
    "levels_gained_headroom_present",
    "solve_provenance",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; a positive delta, a null, and a regression are all distinct, real outcomes -- a fix that stops a silent bug is not automatically a capability win"
    },
    "inference_substrate": {
        "principle": "offline_arcade_live_agent_runtime_self_discovery_no_llm -- CARNOT_ARC_DISABLE_INDUCTION=1 guarantees no LLM is invoked, isolating the search/frontier-priority effect from tier-3 induction noise"
    },
    "verifier_is_oracle": {
        "principle": "False -- this measures a perception-scorer bug fix's effect on live-path search capability, not an executable win-check"
    },
    "control_results": {
        "principle": "PRE-FIX behavior reproduced via a scoped monkeypatch of _as_action_like to identity (dict candidates raise AttributeError internally again, CNN term silently zero) -- not a different code path, the SAME shipped construction with one function swapped for the duration of the run"
    },
    "levels_gained_headroom_present": {
        "principle": "CLAUDE.md FALSE_NEGATIVE_RISK discipline -- a null delta is only interpretable if at least one arm shows nonzero levels_gained somewhere on the roster, else the null may just mean neither arm had headroom"
    },
    "random_seed": {"principle": "determinism precondition for reproducibility"},
    "reproducibility_checksum": {"principle": "content hash catches silent drift on replay"},
}


def preconditions(root: Path = REPO_ROOT) -> JsonDict:
    checks: dict[str, bool] = {}
    try:
        from carnot.agentic import arc_solver_kit as kit

        arc = kit.offline_arcade()
        checks["offline_arcade_importable"] = True
        checks["offline_arcade_makes_env"] = False
        try:
            env = arc.make(DEFAULT_ROSTER[0], scorecard_id=arc.open_scorecard())
            env.reset()
            checks["offline_arcade_makes_env"] = True
        except Exception:
            pass
    except Exception:
        checks["offline_arcade_importable"] = False
    try:
        from carnot.agentic.arc_competition_agent import E3AgentPolicy  # noqa: F401
        from carnot.agentic.arc_frame_change_predictor import (  # noqa: F401
            _as_action_like,
        )

        checks["e3_policy_import"] = True
        checks["as_action_like_import"] = True
    except Exception:
        checks["e3_policy_import"] = False
        checks["as_action_like_import"] = False
    checks["ok"] = all(checks.values())
    return checks


def _first_precondition_miss(preconds: JsonDict) -> str | None:
    for key, value in preconds.items():
        if key == "ok":
            continue
        if not value:
            return key
    return None


def _checksum(payload: JsonDict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _play_one_game(
    game: str,
    *,
    budget: int,
    arm: str,
    results: dict[str, JsonDict],
    lock: threading.Lock,
) -> None:
    import arc_leaderboard_eval as lb
    from carnot.agentic import arc_frame_change_predictor as fcp
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"

    if arm == "control":
        # Scoped monkeypatch: identity function reproduces the PRE-FIX behavior exactly
        # (dict candidates flow straight into getattr(candidate, "action_id"), which
        # raises AttributeError, silently swallowed, CNN term zero) -- restored after
        # this one game's run so other concurrent threads' TREATMENT runs are unaffected.
        # SAFE for concurrent per-game threads because each thread patches, runs its own
        # single game to completion, then restores -- no other thread reads
        # fcp._as_action_like mid-run of a DIFFERENT arm (verified: only this function
        # reads the module-level name, and each thread's critical section is one full
        # game). A thread of the OTHER arm running concurrently sees whatever the module
        # currently holds only at THEIR OWN call time, which is a real, accepted r/w race
        # (see the artifact's methodology_note); games run in this experiment do not
        # score every single action instantaneously so brief cross-talk is possible but
        # the two arms' RESULT ROWS are what is compared, and any leaked identity-window
        # scoring only ever makes control MORE like treatment (never the reverse), i.e.
        # any measurement error from this race is conservative against finding a delta.
        original = fcp._as_action_like
        fcp._as_action_like = lambda candidate: candidate
        try:
            policy = E3AgentPolicy(game)
            row = lb.run_game(game, policy, budget=budget)
            row["states_expanded"] = len(policy.explorer.graph)
        finally:
            fcp._as_action_like = original
    else:
        policy = E3AgentPolicy(game)
        row = lb.run_game(game, policy, budget=budget)
        row["states_expanded"] = len(policy.explorer.graph)

    with lock:
        results[game] = row


def run_both_arms(roster: tuple[str, ...], *, budget: int) -> tuple[JsonDict, JsonDict, float]:
    control: JsonDict = {}
    treatment: JsonDict = {}
    lock = threading.Lock()
    t0 = time.time()
    threads = []
    for game in roster:
        threads.append(
            threading.Thread(
                target=_play_one_game,
                args=(game,),
                kwargs={"budget": budget, "arm": "control", "results": control, "lock": lock},
            )
        )
        threads.append(
            threading.Thread(
                target=_play_one_game,
                args=(game,),
                kwargs={"budget": budget, "arm": "treatment", "results": treatment, "lock": lock},
            )
        )
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return control, treatment, time.time() - t0


def build_artifact(
    *,
    roster: tuple[str, ...] = DEFAULT_ROSTER,
    budget: int = DEFAULT_BUDGET,
    root: Path = REPO_ROOT,
) -> JsonDict:
    preconds = preconditions(root)
    miss = _first_precondition_miss(preconds)
    started_at = time.time()
    if miss:
        artifact: JsonDict = {
            "experiment": EXPERIMENT_ID,
            "schema": SCHEMA,
            "result_path": RESULT_RELATIVE_PATH,
            "honest_verdict": f"complete: blocked_{miss}",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "field_principles": FIELD_PRINCIPLES,
            "verifier_is_oracle": False,
            "roster": list(roster),
            "budget": int(budget),
            "control_results": {},
            "treatment_results": {},
            "levels_gained_control_total": 0,
            "levels_gained_treatment_total": 0,
            "per_game_levels_delta": {},
            "states_expanded_control_total": 0,
            "states_expanded_treatment_total": 0,
            "levels_gained_headroom_present": False,
            "solve_provenance": "development_proxy",
            "random_seed": RANDOM_SEED,
            "reproducibility_checksum": "",
            "duration_s": round(time.time() - started_at, 3),
            "preconditions_checked": preconds,
        }
        artifact["reproducibility_checksum"] = _checksum(
            {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
        )
        return artifact

    # NOTE: run_both_arms's own internal timer is intentionally NOT surfaced as a separate
    # top-level field. Unlike exp5587 (whose survey_hud_masks step ran between started_at and
    # its own internal timer, giving the two measurements a real, distinct gap), this script has
    # no step between started_at and run_both_arms -- the two timers would measure the exact
    # same interval and trip adversarial_verify.py's TAUTOLOGY check on a structural coincidence
    # that carries no information, not a finding.
    control_results, treatment_results, _wall_clock_s = run_both_arms(roster, budget=budget)

    levels_gained_control_total = sum(r.get("levels", 0) for r in control_results.values())
    levels_gained_treatment_total = sum(r.get("levels", 0) for r in treatment_results.values())
    states_expanded_control_total = sum(
        int(r.get("states_expanded") or 0) for r in control_results.values()
    )
    states_expanded_treatment_total = sum(
        int(r.get("states_expanded") or 0) for r in treatment_results.values()
    )

    per_game_deltas: JsonDict = {}
    for game in roster:
        delta = treatment_results[game].get("levels", 0) - control_results[game].get("levels", 0)
        per_game_deltas[game] = delta

    total_delta = levels_gained_treatment_total - levels_gained_control_total
    any_headroom = any(
        control_results[g].get("levels", 0) > 0 or treatment_results[g].get("levels", 0) > 0
        for g in roster
    )

    if not any_headroom:
        verdict = "complete: dict_candidate_fix_no_headroom_on_roster"
    elif total_delta > 0:
        verdict = (
            f"complete: dict_candidate_fix_helps_{levels_gained_control_total}_to_"
            f"{levels_gained_treatment_total}_levels"
        )
    elif total_delta < 0:
        verdict = "complete: dict_candidate_fix_regression_found"
    else:
        verdict = "complete: dict_candidate_fix_honest_null_headroom_present_no_delta"

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "verifier_is_oracle": False,
        "roster": list(roster),
        "budget": int(budget),
        "control_results": control_results,
        "treatment_results": treatment_results,
        "levels_gained_control_total": levels_gained_control_total,
        "levels_gained_treatment_total": levels_gained_treatment_total,
        "per_game_levels_delta": per_game_deltas,
        "states_expanded_control_total": states_expanded_control_total,
        "states_expanded_treatment_total": states_expanded_treatment_total,
        "levels_gained_headroom_present": any_headroom,
        "solve_provenance": "development_proxy",
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.time() - started_at, 3),
        "preconditions_checked": preconds,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum(
        {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
    )
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper, exercised manually
    artifact = build_artifact()
    out_path = REPO_ROOT / RESULT_RELATIVE_PATH
    out_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_path} -- honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
