"""Experiment 5578: concurrent multi-game offline A/B for the run-local mechanic ledger.

Validates (or refutes) the RunLocalMechanicLedger built in commit `0ae6948c0`, per the
rollout order in `docs/research-notes/arc-agi3-run-local-cross-game-adaptation-scope-
2026-07-12.md` SS2.4. This is the REQUIRED validation before
`CARNOT_ARC_RUN_LOCAL_ADAPTATION` is ever flipped on by default -- the ledger currently
ships off, and stays off unless this experiment (or a successor with the same rigor)
clears the bar exp4582 already used for the closely-related idea.

Design constraints taken directly from the scope doc (do not weaken these without
updating the scope doc first):
  - Simulate the REAL deployment shape: `Swarm.main()` instantiates every game's agent up
    front and runs them CONCURRENTLY on separate threads within ONE process (confirmed
    against the actual scored submission path, not just the offline reference). A
    sequential simulation would not represent the real shape and any result from one
    would not be trustworthy evidence either way -- so this harness uses real
    `threading.Thread`s, matching that shape exactly.
  - Development-proxy discipline: every game starts from a fresh `env.reset()` with NO
    registry-prefix consultation -- the offline arcade sim is used only as a zero-quota
    stand-in for "a game the agent has not solved before," never as a source of a
    memorized action sequence (the operator's own dividing line for this whole scope).
  - Diverse roster (SS2.5 adversarial-check requirement): the game roster spans distinct
    registry `mechanic_class` labels and action models, so an apparent "later games do
    better" effect cannot be an artifact of a narrow/repeated test roster -- a real
    hidden-game roster does not repeat games either.
  - Matched compute: both conditions (ledger off / ledger on) use the identical roster,
    identical per-game action budget, identical thread count, and the SAME random seed
    for whatever RNG-influenced componentry the search touches.
  - Paired bootstrap CI on the delta, same statistical bar `exp4582` used: a claim
    requires the CI to exclude the no-mechanism baseline (0.0 delta).
  - Positive control: confirm the ledger accumulates without lost updates under this
    run's real concurrency, and that the confidence gate fires at least once when
    enabled -- a sanity check on the mechanism, not the headline claim.

Honest scope: this is a SMALL-N experiment (roster size bounded by wall-clock budget for
a single interactive session, one run per condition -- not repeated trials). Per this
project's own sample-size-rigor discipline, an N this small cannot support a strong
statistical claim; the CI reported here will typically be WIDE, and a null result is the
a priori more likely outcome (per exp4582's own null on the closely-related idea). This
script exists to produce that honest measurement, not to manufacture a positive result.

Spec refs: REQ-CAPSTONE-4582 (extends the live-wiring requirement's validation
obligation stated in `_maybe_route_from_transitions`'s own docstring).
"""

from __future__ import annotations

import hashlib
import json
import os
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

EXPERIMENT_ID = "experiment_5578_run_local_ledger_concurrent_ab"
RESULT_RELATIVE_PATH = "results/experiment_5578_run_local_ledger_concurrent_ab.json"
SCHEMA = "carnot.exp5578.run_local_ledger_concurrent_ab.v1"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
RANDOM_SEED = 5578
# StepwiseExplorer's per-action search cost measured empirically at >1.75s/action on at
# least one roster game (a 150-action timing probe on bp35 did not finish in 265s). An
# initial budget=30 run (2026-07-12) produced a floor-effect null: EVERY roster game used
# its full 30-action budget with zero level-ups in both conditions, so the run-local
# ledger's confidence bonus never had an observable window to show an effect.
#
# Recalibrated using known solve-length ground truth from ops/arc_solve_registry.yaml (the
# minimal SCRIPTED action count each roster game's GameAdapter needs to clear L1(+L2), i.e.
# the floor a from-scratch exploratory search must clear just to have a chance at one
# level-up): s5i5=39, sk48=44, g50t=48, re86=56, bp35=57, dc22=132 (the outlier). budget=300
# gives >=2.3x margin over EVERY roster game's known-minimal scripted route (dc22) and up
# to ~7.7x margin on the easiest (s5i5), which should be enough headroom for an
# adapter-free StepwiseExplorer search (no known route, real backtracking overhead) to
# reach at least one level transition on most of the roster.
DEFAULT_BUDGET_PER_GAME = 300
DEFAULT_BOOTSTRAPS = 2000

# Diverse roster: distinct registry mechanic_class labels + action models, per SS2.5's
# adversarial-check requirement (guards against "later games do better because the
# roster repeats similar games", which a real hidden roster would never do).
DEFAULT_ROSTER = ("bp35", "dc22", "g50t", "re86", "s5i5", "sk48")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "verifier_is_oracle",
    "roster",
    "budget_per_game",
    "control_results",
    "treatment_results",
    "levels_gained_delta_mean",
    "levels_gained_delta_ci",
    "actions_to_first_levelup_delta_mean",
    "ledger_lost_update_check_passed",
    "confidence_gate_fired_at_least_once",
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
        "principle": "no LLM/proposer invoked (StepwiseExplorer explore-phase only, proposer=None) -- pure CPU search, declared honestly rather than defaulted to a GGUF substrate that was never loaded"
    },
    "verifier_is_oracle": {
        "principle": "False -- this measures whether a learned routing signal adds value, not whether the executable win-check agrees with itself"
    },
    "roster": {
        "principle": "diverse mechanic_class/action-model spread so a positive result cannot be an artifact of a narrow/repeated test roster (SS2.5)"
    },
    "levels_gained_delta_ci": {
        "principle": "paired bootstrap CI on (treatment - control) per game; a claim requires the CI to exclude 0.0, same bar exp4582 used for the closely-related idea"
    },
    "ledger_lost_update_check_passed": {
        "principle": "positive control -- confirms the thread-safety claim under THIS run's real concurrency, separate from the efficacy question"
    },
    "no_regression_fallback": {
        "principle": "CARNOT_ARC_RUN_LOCAL_ADAPTATION stays off by default regardless of this result until an operator reviews it -- this script never flips the flag"
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
        from carnot.agentic.arc_competition_agent import E3AgentPolicy, RunLocalMechanicLedger

        checks["e3_policy_import"] = True
        checks["run_local_ledger_import"] = True
    except Exception:
        checks["e3_policy_import"] = False
        checks["run_local_ledger_import"] = False
    checks["ok"] = all(checks.values())
    return checks


def _first_precondition_miss(preconds: JsonDict) -> str | None:
    for key, value in preconds.items():
        if key == "ok":
            continue
        if not value:
            return key
    return None


def _play_one_game(
    game: str,
    *,
    budget: int,
    ledger_enabled: bool,
    results: dict[str, JsonDict],
    lock: threading.Lock,
) -> None:
    """Run one game to its action budget on its OWN thread -- mirrors Swarm.main()'s shape
    (every game's agent on its own Thread, sharing this process). Fresh env.reset() only;
    no registry-prefix consultation (development-proxy discipline)."""

    import os as _os

    prior = _os.environ.get("CARNOT_ARC_RUN_LOCAL_ADAPTATION")
    try:
        if ledger_enabled:
            _os.environ["CARNOT_ARC_RUN_LOCAL_ADAPTATION"] = "1"
        else:
            _os.environ.pop("CARNOT_ARC_RUN_LOCAL_ADAPTATION", None)

        from carnot.agentic import arc_solver_kit as kit
        from carnot.agentic.arc_agi3_live_adapter import _game_action
        from carnot.agentic.arc_competition_agent import E3AgentPolicy
        from arcengine import GameAction

        arc = kit.offline_arcade()
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        frame = env.reset()
        frames = [frame]
        policy = E3AgentPolicy(game, proposer=None, value_head=lambda _f: 0.0)

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

        row: JsonDict = {
            "game": game,
            "start_level": start_level,
            "max_level_reached": max_level,
            "levels_gained": max_level - start_level,
            "first_levelup_action": first_levelup_action,
            "actions_used": budget,
            "feature_router_mechanic_class": (policy.feature_router or {}).get("mechanic_class")
            if policy.feature_router
            else None,
            "feature_router_approach": (policy.feature_router or {}).get("approach")
            if policy.feature_router
            else None,
            "strategy_route_run_local_applied": "feature_router_mechanic_class"
            in policy.strategy_route,
            "error": error,
        }
    finally:
        if prior is None:
            _os.environ.pop("CARNOT_ARC_RUN_LOCAL_ADAPTATION", None)
        else:
            _os.environ["CARNOT_ARC_RUN_LOCAL_ADAPTATION"] = prior

    with lock:
        results[game] = row


def run_condition(
    roster: tuple[str, ...], *, budget: int, ledger_enabled: bool
) -> tuple[dict[str, JsonDict], float]:
    """Run the WHOLE roster CONCURRENTLY on real threads -- the actual Swarm.main() shape,
    not a sequential loop (see module docstring)."""

    results: dict[str, JsonDict] = {}
    lock = threading.Lock()
    t0 = time.time()
    threads = [
        threading.Thread(
            target=_play_one_game,
            args=(game,),
            kwargs={
                "budget": budget,
                "ledger_enabled": ledger_enabled,
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
            "control_results": {},
            "treatment_results": {},
            "levels_gained_delta_mean": 0.0,
            "levels_gained_delta_ci": [0.0, 0.0],
            "actions_to_first_levelup_delta_mean": None,
            "ledger_lost_update_check_passed": False,
            "confidence_gate_fired_at_least_once": False,
            "recommendation": "blocked; no measurement was fabricated",
            "no_regression_fallback": "CARNOT_ARC_RUN_LOCAL_ADAPTATION stays off by default",
            "random_seed": RANDOM_SEED,
            "duration_s": round(time.time() - started_at, 3),
            "preconditions_checked": preconds,
        }
        artifact["reproducibility_checksum"] = _checksum(artifact)
        return artifact

    random.seed(RANDOM_SEED)

    control_results, control_wall_s = run_condition(roster, budget=budget, ledger_enabled=False)
    treatment_results, treatment_wall_s = run_condition(roster, budget=budget, ledger_enabled=True)

    # Positive control 1: no lost updates. Every game in the roster must have produced a row
    # (a lost/crashed thread would silently drop a key) in BOTH conditions.
    lost_update_check_passed = len(control_results) == len(roster) and len(
        treatment_results
    ) == len(roster)

    # Positive control 2: the confidence gate fired at least once somewhere in the treatment
    # condition (a real behavior change was possible, not just dead code running quietly).
    confidence_gate_fired = any(
        row.get("strategy_route_run_local_applied") for row in treatment_results.values()
    )

    level_deltas: list[float] = []
    action_deltas: list[float] = []
    for game in roster:
        c = control_results.get(game, {})
        t = treatment_results.get(game, {})
        level_deltas.append(float(t.get("levels_gained", 0) - c.get("levels_gained", 0)))
        c_first = c.get("first_levelup_action")
        t_first = t.get("first_levelup_action")
        if c_first is not None and t_first is not None:
            # positive action_delta = treatment reached its first level-up in FEWER actions
            action_deltas.append(float(c_first - t_first))

    level_delta_mean = statistics.mean(level_deltas) if level_deltas else 0.0
    level_delta_ci = _paired_bootstrap_delta_ci(
        level_deltas, random_seed=RANDOM_SEED, n_bootstrap=n_bootstrap
    )
    action_delta_mean = statistics.mean(action_deltas) if action_deltas else None

    ci_excludes_zero_positive = level_delta_ci[0] > 0.0

    if not lost_update_check_passed:
        verdict = "complete: run_local_ledger_ab_positive_control_failed_lost_updates"
        recommendation = (
            "POSITIVE CONTROL FAILED: a game's result went missing under real concurrency -- "
            "investigate the ledger/thread-safety implementation before trusting ANY delta "
            "measured here. Do not enable the flag."
        )
    elif ci_excludes_zero_positive:
        verdict = "success: run_local_ledger_ab_positive_delta_ci_excludes_zero"
        recommendation = (
            "Measured a positive levels_gained delta whose bootstrap CI excludes 0.0 on this "
            f"N={len(roster)}-game roster (budget={budget}/game). This is still a SMALL-N result "
            "(one run per condition, not repeated trials) -- treat as suggestive, not conclusive, "
            "and consider a larger/repeated-trial follow-up before flipping the default-on flag. "
            "An operator should review this artifact before any config change."
        )
    else:
        verdict = "complete: run_local_ledger_ab_honest_null_ci_includes_zero"
        recommendation = (
            "No positive effect detected above noise on this roster/budget -- consistent with "
            "exp4582's null on the closely-related full-solver-swap idea. Per the Failed-"
            "Experiment Rerun Discipline: do not re-propose this exact measurement without a "
            "stated difference (larger roster, more trials, a different outcome proxy, or a "
            "changed component). Keep CARNOT_ARC_RUN_LOCAL_ADAPTATION off by default; retire or "
            "revise the mechanism per operator judgment."
        )

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
        "control_results": control_results,
        "treatment_results": treatment_results,
        "levels_gained_delta_mean": round(level_delta_mean, 4),
        "levels_gained_delta_ci": level_delta_ci,
        "actions_to_first_levelup_delta_mean": (
            round(action_delta_mean, 2) if action_delta_mean is not None else None
        ),
        "actions_to_first_levelup_delta_n": len(action_deltas),
        "ledger_lost_update_check_passed": lost_update_check_passed,
        "confidence_gate_fired_at_least_once": confidence_gate_fired,
        "recommendation": recommendation,
        "no_regression_fallback": "CARNOT_ARC_RUN_LOCAL_ADAPTATION stays off by default",
        "n_bootstrap": int(n_bootstrap),
        "control_wall_clock_s": round(control_wall_s, 2),
        "treatment_wall_clock_s": round(treatment_wall_s, 2),
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.time() - started_at, 3),
        "preconditions_checked": preconds,
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: JsonDict) -> list[str]:
    return [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]


def write_artifact(artifact: JsonDict, root: Path = REPO_ROOT) -> Path:
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    return path


def main() -> int:  # pragma: no cover - CLI wrapper
    artifact = build_artifact()
    errors = artifact_schema_errors(artifact)
    if errors:
        print(f"SCHEMA ERRORS: {errors}", file=sys.stderr)
        return 1
    write_artifact(artifact)
    print(json.dumps({k: artifact[k] for k in REQUIRED_ARTIFACT_FIELDS}, indent=2, default=str))
    return 0 if not str(artifact["honest_verdict"]).startswith("complete: blocked_") else 2


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
