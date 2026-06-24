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


class _ArmScorerFactory:
    """Builds a fresh online scorer per integrated (game, variant) attempt and retains each so
    aggregate online diagnostics can be summed after the sweep.

    WHY monkeypatch ``_policy_for_mode`` instead of cloning ``run_variant_attempt``: a hand-clone
    of the harness loop is brittle -- an earlier clone silently LOST lp85's solve (it ran too few
    actions) and reported a false 0.0 baseline. Patching ``mod._policy_for_mode`` at RUNTIME (this
    process only; restored in a ``finally``) keeps the REAL ``run_variant_attempt`` loop intact, so
    the frozen arm reproduces the exp4605 0.04 baseline EXACTLY and ONLY the scorer differs across
    arms. This is a standard test-injection seam, NOT an edit to the exp4605 module.

    WHY ONE fresh scorer per integrated attempt: ``_policy_for_mode`` is called once per attempt
    inside ``run_variant_attempt``, so building a fresh online scorer there gives per-game-per-
    episode learning (the leader's per-game adaptation) without cross-game bleed.

    WHY NO per-RESET reset: the explorer issues env RESETs mid-episode to backtrack and try other
    branches of the SAME level -- the online dynamics knowledge is still valid there, so wiping it
    on every RESET would be wrong. The leader resets only on a level-UP (score increase); within a
    first-win attempt (which breaks at the first level-up) there is no level-up before the win, so
    no reset is needed and the scorer accumulates over the whole pre-win exploration -- exactly the
    online-within-episode signal we want.
    """

    def __init__(self, arm: str, root: Path) -> None:
        self.arm = arm
        self.root = Path(root)
        self.scorers: list[Any] = []

    def build_integrated_scorer(self) -> Any:
        from carnot.agentic.arc_online_action_effect_scorer import build_online_scorer

        scorer = build_online_scorer(self.arm, self.root)
        self.scorers.append(scorer)
        return scorer

    def aggregate_diagnostics(self) -> dict[str, Any]:
        observed = fits = errors = 0
        for scorer in self.scorers:
            if hasattr(scorer, "diagnostics"):
                diag = scorer.diagnostics()
                observed += int(diag.get("observed") or 0)
                fits += int(diag.get("fits") or 0)
                errors += int(diag.get("errors") or 0)
        return {
            "observed": observed,
            "fits": fits,
            "errors": errors,
            "n_scorers": len(self.scorers),
        }


def run_arm(
    arm: str,
    *,
    root: Path = REPO_ROOT,
    budget: int | None = None,
    public_games: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run a single arm measurement and return the artifact dict.

    Uses the REAL exp4605 ``measure_policy_pair`` / ``run_variant_attempt`` loop with a runtime
    monkeypatch of ``_policy_for_mode`` that injects the arm's online scorer into the integrated
    policy. The ``frozen`` arm patches NOTHING (it falls through to the real default scorer), so it
    is byte-identical to the committed exp4605 baseline (first_win_rate=0.04).
    """
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
    factory = _ArmScorerFactory(arm, root_path)

    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    _orig_policy_for_mode = mod._policy_for_mode

    def _patched_policy_for_mode(mode: str, game: str):
        # The bare control and the frozen arm both use the REAL default policy path (no online
        # scorer) -- this is what guarantees the frozen arm == the 0.04 exp4605 baseline.
        if mode == "bare" or arm == "frozen":
            return _orig_policy_for_mode(mode, game)
        return E3AgentPolicy(
            game,
            proposer=mod._NoOpProposer(),
            target_levels=mod._submitted_target_levels(),
            value_weight=mod._submitted_value_weight(),
            frame_change_scorer=factory.build_integrated_scorer(),
        )

    mod._policy_for_mode = _patched_policy_for_mode
    try:
        integrated_measurement, bare_measurement = mod.measure_policy_pair(
            public_games=games,
            variant_ids=variant_ids,
            budget=effective_budget,
            variant_runner_factory=mod.default_variant_runner_factory,
        )
    finally:
        mod._policy_for_mode = _orig_policy_for_mode

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
        "scorer_diagnostics": factory.aggregate_diagnostics(),
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
