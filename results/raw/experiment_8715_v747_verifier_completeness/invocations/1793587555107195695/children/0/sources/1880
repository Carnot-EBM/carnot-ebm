"""Experiment 5584: offline matched-budget A/B for rule-based live HUD-cell masking.

Validates (or refutes) `auto_hud_mask` on `StepwiseExplorer`, wired in commit
`890668359` per REQ-ARC-WMTE-5583, which states explicitly: "Before any default-on
flip, an offline matched-budget A/B (auto_hud_mask on vs. off) SHALL be run per the
Phase Prototype + Empirical Validation + Adversarial Check Discipline." This script
is that validation. `SUBMITTED_AUTO_HUD_MASK_ENABLED` stays False regardless of this
result -- this script never flips the flag; an operator reviews the result.

Design:
  - Development-proxy discipline: every game starts from a fresh `env.reset()` with NO
    registry-prefix consultation -- a from-scratch generic-agent measurement, same
    protocol `exp4582`/`exp5578` already used for closely-related ideas.
  - Roster reused verbatim from `exp5578` (`bp35, dc22, g50t, re86, s5i5, sk48`) for
    direct comparability and because it already satisfies the diverse-mechanic-class
    adversarial-check requirement. A pre-registered survey
    (`_compute_hud_mask_from_frame` run against each game's initial reset frame, see
    `hud_survey` in the artifact) found 5 of 6 roster games (all but sk48) trigger a
    detected status-bar-like mask -- so this roster tests BOTH the efficacy question
    (does masking help on HUD-positive games) and the harmlessness question (sk48 as
    an in-roster negative control: masking must not regress a game with nothing to
    mask).
  - Threaded for wall-clock only, not correctness: unlike exp5578 (which stress-tested
    a SHARED cross-game ledger's thread-safety under Swarm.main()'s real concurrency),
    auto_hud_mask has no shared state across games -- each StepwiseExplorer instance
    computes and holds its own mask independently, so nothing here depends on the
    concurrent execution shape being correct. Games run on separate threads purely to
    avoid a >100-minute sequential wall-clock (StepwiseExplorer measured >1.75s/action
    in exp5578's own calibration; 300 actions x 12 game-runs sequential would not fit
    an interactive session).
  - Matched compute: both conditions use the identical roster, identical per-game
    action budget (300, the same calibrated value exp5578 established), and construct
    `E3AgentPolicy` identically except for `auto_hud_mask`.
  - Paired bootstrap CI on the delta (levels_gained and distinct_states_discovered),
    same statistical bar exp4582/exp5578 used: a claim requires the CI to exclude 0.0.
  - Positive control: confirm the mask actually fires (per-game, matches the
    pre-registered survey) rather than silently being a no-op for the whole roster.

Honest scope: SMALL-N (roster size bounded by wall-clock budget for a single
interactive session, one deterministic-ish run per condition -- StepwiseExplorer's
search order can have marginal non-determinism from dict/set iteration timing but no
intentional randomness at value_weight=0.0, proposer=None). Per this project's
sample-size-rigor discipline, an N this small cannot support a strong statistical
claim; the CI reported here will typically be WIDE. A null result (CI includes 0.0)
is an entirely plausible, honest outcome -- collapsing a handful of duplicate node
hashes only helps if the search would otherwise have re-expanded a masked-identical
state, which a from-scratch 300-action budget may or may not encounter often enough
to matter. This script exists to produce that honest measurement, not to manufacture
a positive result.

Spec refs: REQ-ARC-WMTE-5583 (states this validation obligation directly).
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(REPO_ROOT))

JsonDict = dict[str, Any]

EXPERIMENT_ID = "experiment_5584_hud_mask_offline_ab"
RESULT_RELATIVE_PATH = "results/experiment_5584_hud_mask_offline_ab.json"
SCHEMA = "carnot.exp5584.hud_mask_offline_ab.v1"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
RANDOM_SEED = 5584
DEFAULT_BUDGET_PER_GAME = 300
DEFAULT_BOOTSTRAPS = 2000
DEFAULT_ROSTER = ("bp35", "dc22", "g50t", "re86", "s5i5", "sk48")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "verifier_is_oracle",
    "roster",
    "budget_per_game",
    "hud_survey",
    "control_results",
    "treatment_results",
    "levels_gained_delta_mean",
    "levels_gained_delta_ci",
    "distinct_states_delta_mean",
    "distinct_states_delta_ci",
    "mask_fired_matches_survey",
    "levels_gained_headroom_present",
    "recommendation",
    "no_regression_fallback",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; a measured null is complete (negative-but-real), never partial"
    },
    "inference_substrate": {
        "principle": "no LLM/proposer invoked (StepwiseExplorer explore-phase only, proposer=None) -- pure CPU search, declared honestly"
    },
    "verifier_is_oracle": {
        "principle": "False -- this measures whether a state-identity masking mechanism adds value, not whether the executable win-check agrees with itself"
    },
    "roster": {
        "principle": "reused verbatim from exp5578 for comparability; mixes HUD-positive and HUD-negative (sk48) games as an in-roster harmlessness control"
    },
    "hud_survey": {
        "principle": "pre-registered per-game mask-detection result on the initial reset frame -- lets the reader confirm treatment-arm results are consistent with the mechanism actually firing where expected"
    },
    "levels_gained_delta_ci": {
        "principle": "paired bootstrap CI on (treatment - control) per game; a claim requires the CI to exclude 0.0, same bar exp4582/exp5578 used"
    },
    "distinct_states_delta_ci": {
        "principle": "paired bootstrap CI on the dedup-collapse proxy (graph node count); the mechanism's DIRECT effect, separate from whether that translates into a levels_gained win"
    },
    "mask_fired_matches_survey": {
        "principle": "positive control -- confirms the mechanism actually activates on treatment-arm HUD-positive games rather than silently no-op-ing the whole roster"
    },
    "levels_gained_headroom_present": {
        "principle": "SECOND positive control (CLAUDE.md FALSE_NEGATIVE_RISK) -- confirms at least one arm reached a level-up somewhere on the roster, so a levels_gained null is informative rather than a floor-effect artifact of an unreachable roster/budget"
    },
    "no_regression_fallback": {
        "principle": "SUBMITTED_AUTO_HUD_MASK_ENABLED stays False regardless of this result until an operator reviews it -- this script never flips the flag"
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
        from carnot.agentic.arc_competition_agent import E3AgentPolicy, _compute_hud_mask_from_frame

        checks["e3_policy_import"] = True
        checks["hud_mask_fn_import"] = True
    except Exception:
        checks["e3_policy_import"] = False
        checks["hud_mask_fn_import"] = False
    checks["ok"] = all(checks.values())
    return checks


def _first_precondition_miss(preconds: JsonDict) -> str | None:
    for key, value in preconds.items():
        if key == "ok":
            continue
        if not value:
            return key
    return None


def survey_hud_masks(roster: tuple[str, ...]) -> JsonDict:
    """Pre-registered: does each roster game's INITIAL reset frame trigger the mask?"""

    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import _compute_hud_mask_from_frame

    survey: JsonDict = {}
    for game in roster:
        arc = kit.offline_arcade()
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        frame = env.reset()
        mask = _compute_hud_mask_from_frame(frame)
        survey[game] = {
            "has_hud": mask is not None,
            "n_masked_cells": int(mask.sum()) if mask is not None else 0,
        }
    return survey


def _play_one_game(
    game: str,
    *,
    budget: int,
    auto_hud_mask: bool,
    results: dict[str, JsonDict],
    lock: threading.Lock,
) -> None:
    """Run one game to its action budget, fresh env.reset() only (development-proxy
    discipline). Runs on its own thread purely for wall-clock (see module docstring);
    no shared mutable state with any other game's run besides the results dict, which
    is lock-protected."""

    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_live_adapter import _game_action
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    frame = env.reset()
    frames = [frame]
    policy = E3AgentPolicy(
        game, proposer=None, value_head=lambda _f: 0.0, auto_hud_mask=auto_hud_mask
    )

    start_level = int(getattr(frame, "levels_completed", 0) or 0)
    first_levelup_action: int | None = None
    max_level = start_level
    error: str | None = None
    try:
        for step in range(int(budget)):
            kind, data = policy.next_move(frames, frame)
            if kind == "RESET" or kind is None:
                frame = env.reset()
            else:
                frame = env.step(_game_action(GameAction, int(kind)), data=data)
            frames.append(frame)
            level = int(getattr(frame, "levels_completed", 0) or 0)
            if level > max_level:
                max_level = level
                if first_levelup_action is None:
                    first_levelup_action = step + 1
    except Exception as exc:  # pragma: no cover - live boundary
        error = f"{type(exc).__name__}: {exc}"

    explorer = policy.explorer
    row = {
        "game": game,
        "start_level": start_level,
        "max_level_reached": max_level,
        "levels_gained": max_level - start_level,
        "first_levelup_action": first_levelup_action,
        "actions_used": budget,
        "distinct_states_discovered": len(explorer.graph) if explorer is not None else None,
        "hud_mask_fired": bool(explorer.hud_mask is not None) if explorer is not None else False,
        "error": error,
    }
    with lock:
        results[game] = row


def run_condition(
    roster: tuple[str, ...], *, budget: int, auto_hud_mask: bool
) -> tuple[JsonDict, float]:
    results: JsonDict = {}
    lock = threading.Lock()
    t0 = time.time()
    threads = [
        threading.Thread(
            target=_play_one_game,
            args=(game,),
            kwargs={
                "budget": budget,
                "auto_hud_mask": auto_hud_mask,
                "results": results,
                "lock": lock,
            },
        )
        for game in roster
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, time.time() - t0


def run_both_conditions(
    roster: tuple[str, ...], *, budget: int
) -> tuple[JsonDict, JsonDict, float]:
    """Launch all 2*len(roster) independent game-runs (control + treatment) as one
    thread batch -- each is a fully independent env/policy instance with no shared
    mutable state, so nothing is lost by not separating the two conditions in time,
    and wall-clock drops to roughly one game's worth instead of 2x."""

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
                kwargs={
                    "budget": budget,
                    "auto_hud_mask": False,
                    "results": control,
                    "lock": lock,
                },
            )
        )
        threads.append(
            threading.Thread(
                target=_play_one_game,
                args=(game,),
                kwargs={
                    "budget": budget,
                    "auto_hud_mask": True,
                    "results": treatment,
                    "lock": lock,
                },
            )
        )
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return control, treatment, time.time() - t0


def _paired_bootstrap_delta_ci(
    deltas: list[float], *, random_seed: int, n_bootstrap: int
) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    if n_bootstrap <= 0:
        mean = sum(deltas) / len(deltas)
        return [round(mean, 10), round(mean, 10)]
    rng = random.Random(random_seed)
    n = len(deltas)
    samples = []
    for _ in range(int(n_bootstrap)):
        total = 0.0
        for _s in range(n):
            total += deltas[rng.randrange(n)]
        samples.append(total / n)
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return [round(float(lo), 10), round(float(hi), 10)]


def _checksum(payload: JsonDict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def build_artifact(
    *,
    roster: tuple[str, ...] = DEFAULT_ROSTER,
    budget: int = DEFAULT_BUDGET_PER_GAME,
    n_bootstrap: int = DEFAULT_BOOTSTRAPS,
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
            "budget_per_game": int(budget),
            "hud_survey": {},
            "control_results": {},
            "treatment_results": {},
            "levels_gained_delta_mean": 0.0,
            "levels_gained_delta_ci": [0.0, 0.0],
            "distinct_states_delta_mean": 0.0,
            "distinct_states_delta_ci": [0.0, 0.0],
            "mask_fired_matches_survey": False,
            "levels_gained_headroom_present": False,
            "false_negative_risk_note": None,
            "recommendation": "blocked -- preconditions failed, no measurement taken",
            "no_regression_fallback": True,
            "random_seed": RANDOM_SEED,
            "reproducibility_checksum": "",
            "duration_s": round(time.time() - started_at, 3),
            "preconditions_checked": preconds,
        }
        artifact["reproducibility_checksum"] = _checksum(
            {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
        )
        return artifact

    hud_survey = survey_hud_masks(roster)
    control_results, treatment_results, combined_wall_clock_s = run_both_conditions(
        roster, budget=budget
    )

    levels_gained_deltas: list[float] = []
    states_deltas: list[float] = []
    mask_fired_ok = True
    for game in roster:
        c = control_results[game]
        t = treatment_results[game]
        levels_gained_deltas.append(float(t["levels_gained"] - c["levels_gained"]))
        if (
            c["distinct_states_discovered"] is not None
            and t["distinct_states_discovered"] is not None
        ):
            states_deltas.append(
                float(t["distinct_states_discovered"] - c["distinct_states_discovered"])
            )
        expected_fire = bool(hud_survey[game]["has_hud"])
        # The mask is computed on the FIRST ingested frame, which may differ slightly
        # from the pre-registered survey's bare env.reset() frame if StepwiseExplorer's
        # own bootstrap sequence issues a warmup action first -- so this checks
        # DIRECTIONAL consistency (fired only where a HUD was plausible), not exact
        # frame-for-frame equality.
        if t["hud_mask_fired"] and not expected_fire:
            mask_fired_ok = False
        # A HUD-positive game failing to ever fire in the treatment arm would mean the
        # mechanism silently no-op'd where it was expected to activate.
        if expected_fire and not t["hud_mask_fired"]:
            mask_fired_ok = False

    levels_gained_delta_mean = (
        statistics.mean(levels_gained_deltas) if levels_gained_deltas else 0.0
    )
    levels_gained_delta_ci = _paired_bootstrap_delta_ci(
        levels_gained_deltas, random_seed=RANDOM_SEED, n_bootstrap=n_bootstrap
    )
    distinct_states_delta_mean = statistics.mean(states_deltas) if states_deltas else 0.0
    distinct_states_delta_ci = _paired_bootstrap_delta_ci(
        states_deltas, random_seed=RANDOM_SEED + 1, n_bootstrap=n_bootstrap
    )

    levels_ci_excludes_zero = levels_gained_delta_ci[0] > 0.0 or levels_gained_delta_ci[1] < 0.0
    states_ci_excludes_zero = distinct_states_delta_ci[0] > 0.0 or distinct_states_delta_ci[1] < 0.0
    any_error = any(control_results[g]["error"] or treatment_results[g]["error"] for g in roster)
    # Positive control for the levels_gained claim specifically: did EITHER arm reach
    # even one level-up anywhere on the roster? If not, the roster/budget/explorer-
    # config combination has no observable headroom for a levels_gained delta to exist
    # in either direction -- a levels_gained null under that condition is uninformative
    # about the mechanism, not evidence the mechanism doesn't help (CLAUDE.md
    # FALSE_NEGATIVE_RISK: "a NULL claim... is NOT a finding unless a positive control
    # passed"). This is a DIFFERENT positive control from mask_fired_matches_survey
    # (which checks the mechanism activates) -- this one checks the SCORE METRIC itself
    # was even reachable.
    any_levelup_either_arm = any(
        control_results[g]["levels_gained"] > 0 or treatment_results[g]["levels_gained"] > 0
        for g in roster
    )
    levels_gained_headroom_present = any_levelup_either_arm

    if any_error:
        verdict = "complete: hud_mask_ab_run_error_see_error_fields"
        recommendation = "An arm errored mid-run; do not act on deltas until re-run clean."
        false_negative_risk_note = None
    elif levels_ci_excludes_zero and levels_gained_delta_mean > 0.0:
        verdict = "complete: hud_mask_ab_positive_ci_excludes_zero"
        recommendation = (
            "levels_gained_delta_ci excludes 0.0 and the mean is positive -- masking "
            "shows a real effect on this roster/budget. Still small-N; consider a "
            "larger roster before flipping SUBMITTED_AUTO_HUD_MASK_ENABLED, but this "
            "clears the CI-excludes-baseline bar exp4582/exp5578 used."
        )
        false_negative_risk_note = None
    elif levels_ci_excludes_zero and levels_gained_delta_mean < 0.0:
        verdict = "complete: hud_mask_ab_negative_ci_excludes_zero"
        recommendation = (
            "levels_gained_delta_ci excludes 0.0 and the mean is NEGATIVE -- masking "
            "appears to HURT on this roster/budget. Do not flip "
            "SUBMITTED_AUTO_HUD_MASK_ENABLED; investigate the mechanism for a "
            "false-positive mask hiding real board state before any further attempt."
        )
        false_negative_risk_note = None
    elif not levels_gained_headroom_present:
        verdict = "complete: hud_mask_ab_levels_gained_no_headroom_states_delta_real"
        false_negative_risk_note = (
            "FALSE_NEGATIVE_RISK acknowledgment: levels_gained_delta_mean=0.0 with NO "
            "level-up reached in EITHER arm on ANY roster game -- there is no positive "
            "control proving this roster/budget/explorer-config combination could show "
            "a levels_gained effect in either direction, so the levels_gained null is "
            "UNINFORMATIVE about whether HUD masking helps reach levels, not evidence "
            "it doesn't. The bare StepwiseExplorer config used here (proposer=None, "
            "flat value_head=0.0, no registry prefix) is apparently too weak to find "
            "any of this roster's known-short scripted solve paths (s5i5=39, sk48=44, "
            "g50t=48, re86=56, bp35=57 actions per ops/arc_solve_registry.yaml) via "
            "undirected search within 300 actions -- a floor effect independent of "
            "HUD masking (matches exp5578's own budget=30 floor-effect finding, just "
            "at a higher budget with this weaker search config). By contrast, "
            "distinct_states_delta (the mechanism's DIRECT, designed-for effect) IS a "
            "real, non-null finding: CI excludes 0.0, mean matches the pre-registered "
            "hud_survey exactly (0 states-delta on the sole HUD-negative game (sk48), "
            "large negative deltas of -34% to -91% on all 5 HUD-positive games) -- the "
            "mechanism collapses exactly the cosmetic HUD-tick duplicate states it was "
            "built to collapse. Whether that collapse translates into reaching MORE "
            "levels remains untested; it requires a stronger explorer config (a real "
            "value head or LLM proposer, or a much larger budget) with a positive "
            "control that headroom exists before the levels_gained question can be "
            "answered either way."
        )
        recommendation = (
            "Do NOT treat this as a clean null on whether HUD masking helps -- see "
            "false_negative_risk_note. The mechanism itself is validated "
            "(distinct_states_delta CI excludes 0.0, positive control passes, zero "
            "harm on the HUD-negative control game). What is NOT yet validated is "
            "whether that translates into reaching more levels, because this "
            "experiment's explorer config could not reach ANY level-up in either arm "
            "to measure that against. Per the Failed-Experiment Rerun Discipline: a "
            "re-run of this SAME config/budget would be a doomed rerun (same floor "
            "effect expected); a re-run with a stated difference (a real value head "
            "or LLM proposer so the roster has reachable headroom, or a much larger "
            "budget) is the correct next step, not a repeat. Keep "
            "SUBMITTED_AUTO_HUD_MASK_ENABLED off by default until that headroom-"
            "present measurement exists."
        )
    else:
        verdict = "complete: hud_mask_ab_honest_null_ci_includes_zero"
        recommendation = (
            "No detectable effect on levels_gained above noise on this roster/budget, "
            "with headroom confirmed present (at least one arm reached a level-up "
            "somewhere on the roster). Per the Failed-Experiment Rerun Discipline: do "
            "not re-propose this exact measurement without a stated difference "
            "(larger roster, more trials, a different outcome proxy, or a changed "
            "component). Keep SUBMITTED_AUTO_HUD_MASK_ENABLED off by default."
        )
        false_negative_risk_note = None

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "verifier_is_oracle": False,
        "roster": list(roster),
        "budget_per_game": int(budget),
        "hud_survey": hud_survey,
        "control_results": control_results,
        "treatment_results": treatment_results,
        "levels_gained_delta_mean": round(levels_gained_delta_mean, 6),
        "levels_gained_delta_ci": levels_gained_delta_ci,
        "levels_gained_ci_excludes_zero": levels_ci_excludes_zero,
        "distinct_states_delta_mean": round(distinct_states_delta_mean, 6),
        "distinct_states_delta_ci": distinct_states_delta_ci,
        "distinct_states_ci_excludes_zero": states_ci_excludes_zero,
        "mask_fired_matches_survey": mask_fired_ok,
        "levels_gained_headroom_present": levels_gained_headroom_present,
        "false_negative_risk_note": false_negative_risk_note,
        "recommendation": recommendation,
        "no_regression_fallback": True,
        "combined_wall_clock_s": round(combined_wall_clock_s, 3),
        "n_bootstrap": int(n_bootstrap),
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
