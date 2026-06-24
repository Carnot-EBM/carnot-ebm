"""Experiment 4710: online action-effect learning -- arm comparison.

Spec refs: REQ-ARC-OAE-4710, SCENARIO-ARC-OAE-4710.

WHY THIS EXPERIMENT
-------------------
The exp4605 baseline measures the FROZEN cross-game scorer (LiveActionEffectScorer from the
exp4629 checkpoint + PersistentAEM). That scorer is static: it cannot adapt to hidden games
whose action mechanics differ from the 25 public games it was trained on.

This experiment compares ARMS:
  "frozen"             -- the exact exp4605 baseline (no online learning)
  "online-scratch"     -- fresh random CNN trained online from the agent's own transitions
  "online-warm"        -- exp4629 checkpoint CNN + online fine-tuning per episode
  "online-warm-propose"-- same + coordinate proposals injected into the explorer candidate set

The operative metric is first_win_rate on the exp4605 held-out first-win harness (offline,
CPU, no LLM, color-permuted variants of the 25 public games). An online arm is a win if its
first_win_rate >= frozen's first_win_rate AND the CI lower bound > 0.

WHY WRITE TO OUR OWN FILE (not the exp4605 path):
The exp4605 path is the conductor's gate artifact. Overwriting it with our online-arm result
would invalidate the gate measurement that the conductor already accepted. Each arm writes to
results/experiment_4710_online_action_learning_arms_<arm>.json so the two measurements coexist.

USAGE
-----
  CARNOT_ARC_ONLINE_ARM=frozen .venv/bin/python -m carnot.experiment_4710_online_action_learning_arms
  CARNOT_ARC_ONLINE_ARM=online-warm .venv/bin/python -m carnot.experiment_4710_online_action_learning_arms

The DEFAULT_BUDGET is inherited from exp4605. To run a faster smoke (B=25), set:
  CARNOT_ARC_ONLINE_BUDGET=25 CARNOT_ARC_ONLINE_ARM=frozen ...
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import carnot.experiment_4605_live_integration_scored_agent as mod

EXPERIMENT = "experiment_4710_online_action_learning_arms"
SCHEMA = "carnot.exp4710.online_action_learning_arms.v1"
RANDOM_SEED = 4710
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
SOLVE_PROVENANCE = "development_proxy"

RESULT_TEMPLATE = "results/experiment_4710_online_action_learning_arms_{arm}.json"

SPEC_REFS = ["REQ-ARC-OAE-4710", "SCENARIO-ARC-OAE-4710"]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "arm": {
        "principle": (
            "Which scorer arm was measured. The 'frozen' arm is the byte-identical exp4605 "
            "baseline. Online arms adapt the CNN from the agent's own transitions per episode."
        )
    },
    "first_win_rate": {
        "principle": (
            "Fraction of variant attempts that solved at least one level (offline reproduction "
            "gate pass). The primary metric for arm comparison."
        )
    },
    "bare_first_win_rate": {
        "principle": (
            "First-win rate of the bare control (no scorer, no value head, no router) run in "
            "parallel. A sanity check: the integrated scorer must not REGRESS below bare."
        )
    },
    "scorer_diagnostics": {
        "principle": (
            "Aggregate observed/fits/errors from the online scorer across all variant attempts. "
            "For the frozen arm these should all be zero. For online arms, observed>0 and fits>0 "
            "are required to confirm the online loop actually ran."
        )
    },
    "delta_integrated_minus_frozen": {
        "principle": (
            "first_win_rate(arm) - first_win_rate(frozen arm). Positive means the online arm "
            "improved over frozen. Populated only when comparing to a loaded frozen artifact."
        )
    },
    "duration_s": {
        "principle": (
            "Wall-clock time from start to end of the measurement. Captures the overhead of "
            "online fitting (should be small relative to the exploration budget)."
        )
    },
    "honest_verdict": {
        "principle": (
            "Terminal prefix required: complete: / success: / etc. Describes whether the arm "
            "completed successfully and the direction of the finding."
        )
    },
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates: the offline env is the 'verifier'; "
            "no live LLM inference is used. Duration floor is 1s, not 60s."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "False: the scorer is oracle-DISTINCT (learned CNN + PersistentAEM, NOT the game's "
            "own level-up oracle). A true value would indicate circularity."
        )
    },
    "solve_provenance": {
        "principle": (
            "development_proxy: solves are produced by the offline development twin (arc_loop "
            "offline sim + GameAdapter), NOT by the scored Kaggle E3 cascade. Not headline."
        )
    },
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """REQ-ARC-OAE-4710: content-addressed checksum matching exp4605 convention."""
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + _sha256(payload)


class _OnlineScorerFactory:
    """Build fresh online scorers for each (game, variant) attempt.

    WHY a class (not a closure): closure variables would share mutable state across
    concurrent or sequential calls. A class makes the state explicit and reset() safe.

    WHY reset() on env RESET: after an env.reset() the game starts a new level from the
    initial state. The CNN's prediction cache (keyed by frame hash) is stale because the
    layout may differ. The observation buffer contains transitions from the previous level
    which are now stale. We clear both. Warm weights survive -- they encode action-effect
    knowledge from earlier levels that may transfer.

    WHY ONE fresh scorer per (game, variant): each variant has different color permutations.
    The CNN's per-pixel click heatmap could theoretically overfit to the color palette of one
    variant, confusing the next. A fresh scorer per attempt is the conservative choice.
    """

    def __init__(self, arm: str, root: Path) -> None:
        self.arm = arm
        self.root = root
        self._current_scorer: Any = None
        self._agg_observed: int = 0
        self._agg_fits: int = 0
        self._agg_errors: int = 0

    def new_scorer(self) -> Any:
        """Build a fresh scorer for a new (game, variant) attempt."""
        from carnot.agentic.arc_online_action_effect_scorer import build_online_scorer

        scorer = build_online_scorer(self.arm, self.root)
        self._current_scorer = scorer
        return scorer

    def record_reset(self) -> None:
        """Call scorer.reset() if the current scorer supports it (online arms only)."""
        if self._current_scorer is not None and hasattr(self._current_scorer, "reset"):
            self._current_scorer.reset()

    def accumulate_diagnostics(self) -> None:
        """Pull diagnostics from the current scorer and add to aggregates."""
        if self._current_scorer is not None and hasattr(self._current_scorer, "diagnostics"):
            diag = self._current_scorer.diagnostics()
            self._agg_observed += int(diag.get("observed") or 0)
            self._agg_fits += int(diag.get("fits") or 0)
            self._agg_errors += int(diag.get("errors") or 0)

    def aggregate_diagnostics(self) -> dict[str, Any]:
        return {
            "observed": self._agg_observed,
            "fits": self._agg_fits,
            "errors": self._agg_errors,
        }


def _make_variant_runner_factory(
    arm: str, root: Path, scorer_factory: _OnlineScorerFactory
) -> mod.VariantRunnerFactory:
    """Build a VariantRunnerFactory for the given arm.

    WHY cloning the run_variant_attempt loop instead of calling it directly: the loop needs
    to (a) inject a fresh online scorer, (b) call scorer.reset() at each env RESET, and
    (c) accumulate diagnostics. The original function's policy is built via _policy_for_mode()
    which always creates the DEFAULT scorer. We cannot patch _policy_for_mode without editing
    the exp4605 module (which we must not do). So we replicate the loop with our injection.

    For the "bare" mode we DO call the original _policy_for_mode("bare") -- that branch is
    unchanged (no scorer injection needed for the bare control).
    """

    def runner(mode: str) -> mod.VariantRunner:
        def run_attempt(
            game: str, spec: Mapping[str, Any], budget: int
        ) -> dict[str, Any]:
            if mode == "bare":
                # Bare control: use the original unmodified function path.
                return dict(mod.run_variant_attempt("bare", game, spec, budget))
            # Integrated arm with online scorer.
            return dict(_run_online_variant_attempt(arm, game, spec, budget, scorer_factory, root))

        return run_attempt

    return runner


def _run_online_variant_attempt(
    arm: str,
    game: str,
    spec: Mapping[str, Any],
    budget: int,
    scorer_factory: _OnlineScorerFactory,
    root: Path,
) -> dict[str, Any]:
    """Run one variant attempt with a fresh online scorer injected into the policy.

    This is a close clone of mod.run_variant_attempt (lines 755-847) with three changes:
    1. The policy is built with frame_change_scorer = scorer_factory.new_scorer().
    2. On every env RESET we call scorer_factory.record_reset() to flush per-level state.
    3. After the attempt, scorer_factory.accumulate_diagnostics() pulls the online counters.
    """
    try:
        from arcengine import GameAction
        from carnot.agentic import arc_solver_kit as kit
        from carnot.agentic.arc_competition_agent import E3AgentPolicy
        from carnot.agentic.arc_variant_generator import VariantEnv
    except ImportError as exc:
        # Missing ARC engine dependency -- return a blocked result, never crash the harness.
        return {
            "game": game,
            "variant_signature": spec.get("variant_signature", ""),
            "variant": int(spec.get("variant", 0)),
            "kind": spec.get("kind", "color"),
            "reflect": spec.get("reflect"),
            "attempted": False,
            "solved": False,
            "first_win": False,
            "reached_level": 0,
            "actions": 0,
            "actions_to_first_levelup": None,
            "solution_labels": [],
            "reproduction_gate": {"reproduced": False},
            "blocked_reason": f"import_error: {exc}",
            "policy_mode": "integrated",
        }

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    env = VariantEnv(env, game, int(spec["variant"]), reflect=spec.get("reflect"))

    # Fresh scorer for this (game, variant) pair.
    scorer = scorer_factory.new_scorer()

    proposer = mod._NoOpProposer()
    policy = E3AgentPolicy(
        game,
        proposer=proposer,
        target_levels=mod._submitted_target_levels(),
        value_weight=mod._submitted_value_weight(),
        frame_change_scorer=scorer,
    )

    frames: list[Any] = []
    latest = None
    labels: list[str] = []
    actions = 0
    start_level: int | None = None
    reached = 0
    actions_to_first: int | None = None

    for _index in range(int(budget)):
        if policy.is_done(frames, latest):
            break
        kind, data = policy.next_move(frames, latest)
        if kind == "RESET":
            latest = env.reset()
            scorer_factory.record_reset()  # flush per-level buffer/cache on reset
            if labels:
                labels.append("RESET")
        elif kind is None:
            break
        else:
            latest = env.step(getattr(GameAction, f"ACTION{kind}"), data=data)
            labels.append(
                json.dumps({"action": int(kind), "data": data}, sort_keys=True, separators=(",", ":"))
            )
            actions += 1
        if start_level is None:
            start_level = mod._level_of_frame(latest)
        reached = mod._level_of_frame(latest)
        if start_level is not None and reached > start_level:
            if actions_to_first is None:
                actions_to_first = actions
            break
        frames.append(latest)
        if latest is None:
            break

    claimed = reached if start_level is not None and reached > start_level else 0
    gate: dict[str, Any] = {
        "game": game,
        "claimed_level": claimed,
        "reached_level": 0,
        "reproduced": False,
        "mode": "offline_reproduction_gate_no_solution",
    }
    if claimed > 0 and labels:
        gate = dict(
            kit.reproduce(
                game,
                labels,
                lambda env_obj, label, _f=None: (
                    env_obj.reset()
                    if label == "RESET"
                    else env_obj.step(
                        getattr(GameAction, f"ACTION{json.loads(label)['action']}"),
                        data=json.loads(label).get("data"),
                    )
                ),
                claimed_level=claimed,
            )
        )
    solved = bool(gate.get("reproduced")) and int(gate.get("reached_level") or 0) >= claimed >= 1

    # Pull online diagnostics AFTER the attempt completes.
    scorer_factory.accumulate_diagnostics()

    return {
        "game": game,
        "variant_signature": spec["variant_signature"],
        "variant": int(spec["variant"]),
        "kind": spec["kind"],
        "reflect": spec.get("reflect"),
        "attempted": True,
        "solved": solved,
        "first_win": solved,
        "reached_level": int(gate.get("reached_level") or reached) if solved else reached,
        "actions": actions,
        "actions_to_first_levelup": actions_to_first if solved else None,
        "solution_labels": labels if solved else [],
        "reproduction_gate": gate,
        "blocked_reason": "",
        "policy_mode": "integrated",
    }


def run_arm(
    arm: str,
    *,
    root: Path = REPO_ROOT,
    budget: int | None = None,
    public_games: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run a single arm measurement and return the artifact dict."""
    started = time.time()
    root_path = Path(root)

    effective_budget = budget
    if effective_budget is None:
        env_budget = os.environ.get("CARNOT_ARC_ONLINE_BUDGET", "").strip()
        if env_budget:
            try:
                effective_budget = int(env_budget)
            except ValueError:
                pass
    if effective_budget is None:
        effective_budget = mod.DEFAULT_BUDGET

    games = list(
        public_games
        if public_games is not None
        else (mod._public_games(root_path) if hasattr(mod, "_public_games") else [])
    )
    if not games:
        raise RuntimeError(
            "No public games found. Is environment_files/ present in the repo root?"
        )

    variant_ids = mod.resolve_variant_ids(None)
    scorer_factory = _OnlineScorerFactory(arm, root_path)
    factory = _make_variant_runner_factory(arm, root_path, scorer_factory)

    integrated_measurement, bare_measurement = mod.measure_policy_pair(
        public_games=games,
        variant_ids=variant_ids,
        budget=effective_budget,
        variant_runner_factory=factory,
    )

    duration_s = max(1.0, time.time() - started)

    first_win_rate = float(integrated_measurement.get("first_win_rate") or 0.0)
    bare_first_win_rate = float(bare_measurement.get("first_win_rate") or 0.0)

    per_game_solved = sorted(
        {
            str(attempt["game"])
            for attempt in integrated_measurement.get("variant_attempts", [])
            if attempt.get("first_win") or attempt.get("solved")
        }
    )

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "arm": arm,
        "first_win_rate": first_win_rate,
        "bare_first_win_rate": bare_first_win_rate,
        "variant_attempts": list(integrated_measurement.get("variant_attempts", [])),
        "per_game_solved": per_game_solved,
        "scorer_diagnostics": scorer_factory.aggregate_diagnostics(),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "solve_provenance": SOLVE_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": round(duration_s, 3),
        "field_principles": {k: v["principle"] for k, v in FIELD_PRINCIPLES.items()},
    }

    # Honest verdict.
    if first_win_rate > bare_first_win_rate:
        verdict = f"complete: arm={arm} first_win_rate={first_win_rate:.4f} > bare={bare_first_win_rate:.4f}"
    elif first_win_rate == bare_first_win_rate:
        verdict = f"complete: arm={arm} first_win_rate={first_win_rate:.4f} matches bare (null delta)"
    else:
        verdict = (
            f"complete: arm={arm} first_win_rate={first_win_rate:.4f} below bare={bare_first_win_rate:.4f}"
        )
    artifact["honest_verdict"] = verdict
    artifact["reproducibility_checksum"] = payload_checksum(artifact)

    return artifact


def main() -> int:
    """CLI entry point -- reads arm from CARNOT_ARC_ONLINE_ARM env var (default: frozen)."""
    arm = os.environ.get("CARNOT_ARC_ONLINE_ARM", "frozen").strip()
    valid_arms = ("frozen", "online-scratch", "online-warm", "online-warm-propose")
    if arm not in valid_arms:
        print(
            f"ERROR: CARNOT_ARC_ONLINE_ARM={arm!r} is not a valid arm. "
            f"Valid arms: {', '.join(valid_arms)}",
            file=sys.stderr,
        )
        return 1

    print(f"[exp4710] Running arm={arm!r} ...")
    t0 = time.time()
    artifact = run_arm(arm)
    elapsed = time.time() - t0
    print(f"[exp4710] Done in {elapsed:.1f}s. first_win_rate={artifact['first_win_rate']:.4f}")
    print(f"[exp4710] scorer_diagnostics={artifact['scorer_diagnostics']}")
    print(f"[exp4710] honest_verdict={artifact['honest_verdict']}")

    result_path = REPO_ROOT / RESULT_TEMPLATE.format(arm=arm.replace("-", "_"))
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[exp4710] Artifact written to {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
