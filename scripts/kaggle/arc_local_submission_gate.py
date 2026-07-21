"""LOCAL SUBMISSION GATE (operator directive 2026-06-20): never waste a 1/day Kaggle submission slot on a
config that is a LOCAL REGRESSION.

Before any submit, this measures the CURRENT submitted-default agent (make_carnot_agent -> E3AgentPolicy,
via `arc_leaderboard_eval.py --policy e3`, frame-only) on a fixed game set and compares it to the best
VERIFIED baseline (`ops/arc-submission-baseline.json`) on the two things the leaderboard actually rewards:
  (1) solve-rate (solved games), and (2) ACTION EFFICIENCY (median actions on solved games -- the score
  is (human/agent)^2, so a "solve" that burns the whole action budget scores ~0).
PASS only if the current config is NON-INFERIOR on BOTH. This is what catches the regressions we already
hit: value_weight=5 (1/8 solved, slow) and the E3+v3 cascade (3 solved, ~7700 actions/solve vs bare BFS's
21 on lp85). It is a LOCAL (25-public-game) proxy, NOT a leaderboard predictor -- its only claim is
"don't submit a config locally WORSE than the last verified one."

Exit 0 = PASS (safe to submit), 1 = FAIL (regression -> refuse), 2 = could not measure.
CLI:  --check (default)            run the gate, print verdict, set exit code
      --update-baseline            overwrite the baseline with the CURRENT measurement (after a verified
                                   improvement + an actual successful submit)
      --policy e3|explorer         which policy to measure (default e3 = the submitted default)
      --budget N (8000)  --cap S (115)  --json
"""

import argparse
import inspect
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from typing import Any

REPO = Path(__file__).resolve().parents[2]
BASELINE = REPO / "ops" / "arc-submission-baseline.json"
EVAL = REPO / "scripts" / "arc_leaderboard_eval.py"
CANONICAL_GAME_SET = ("lp85", "m0r0", "sp80", "vc33", "cd82", "ft09", "su15", "ls20")
CANONICAL_BASELINE_ACTIONS_BY_GAME = {"lp85": 7792, "m0r0": 7789, "sp80": 7724, "vc33": 7731}
CANONICAL_CORE_GAMES = tuple(CANONICAL_BASELINE_ACTIONS_BY_GAME)
CANONICAL_BASELINE_MEDIAN_ACTIONS = 7760.0
CANONICAL_LP85_PER_LEVEL_EFFICIENCY_FLOOR = 2.0069
CANONICAL_ACTION_FIELD = "actions"
CANONICAL_ACTION_METRIC = {
    "field": CANONICAL_ACTION_FIELD,
    "definition": "total_actions_on_solved_games",
}
HEADROOM_BUDGET_CANDIDATES = (8000, 12000, 16000, 24000)
DEFAULT_BUDGET = 8000
INDUCTION_DISABLE_ENV = "CARNOT_ARC_DISABLE_INDUCTION"
OFFLINE_GATE_DISABLE_INDUCTION = True
PROPOSER_PARITY_GUARD = "offline_live_proposer_config_parity"
LIVE_PROPOSER_KIND = "LocalGGUFProposer"
# 4 reliably-solvable games (the bare-BFS solves) + 4 controls. Small so the gate runs in a couple minutes.
GATE_GAMES = list(CANONICAL_GAME_SET)
_LINE = re.compile(
    r"live=L(\d+)\s*\(\+(\d+)\)\s*actions=\s*(\d+)"
    r"(?:\s*eff=([\d.]+))?"
    r"(?:\s*nav_reset=(\d+)\s*nav_fwhr=([\d.]+))?"
)
EFFICIENCY_SLACK = (
    1.10  # allow 10% worse median actions before calling it a regression (FALLBACK metric)
)
EFFICIENCY_DROP_SLACK = (
    0.97  # PRIMARY: fail if CORE per-level efficiency drops below 97% of baseline
)
BIG_ACTIONS = 10**9


def _gate_policy_class(policy: str) -> str:
    if policy == "e3":
        return "E3AgentPolicy"
    if policy == "explorer":
        return "StepwiseExplorer"
    return str(policy)


def submitted_agent_proposer_config(
    submitted_agent_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the proposer/induction config implied by the submitted agent path."""

    if submitted_agent_config is None:
        from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG

        submitted_agent_config = SUBMITTED_AGENT_CONFIG
    policy = str(submitted_agent_config.get("policy") or "")
    cascade = bool(submitted_agent_config.get("cascade"))
    induction_enabled = bool(policy == "E3AgentPolicy" and cascade)
    return {
        "policy": policy,
        "cascade": cascade,
        "induction_enabled": induction_enabled,
        "proposer_kind": LIVE_PROPOSER_KIND if induction_enabled else None,
        "proposer_source": "E3AgentPolicy._proposer" if induction_enabled else "none",
        "disable_induction_env": "unset",
        "submitted_config_source": "carnot.agentic.arc_competition_agent.SUBMITTED_AGENT_CONFIG",
    }


def offline_gate_proposer_config(
    *,
    policy: str,
    disable_induction: bool = OFFLINE_GATE_DISABLE_INDUCTION,
) -> dict[str, Any]:
    """Return the local gate's effective proposer/induction config."""

    policy_class = _gate_policy_class(policy)
    cascade = policy_class == "E3AgentPolicy"
    induction_enabled = bool(cascade and not disable_induction)
    return {
        "policy": policy_class,
        "gate_policy_arg": str(policy),
        "cascade": cascade,
        "induction_enabled": induction_enabled,
        "proposer_kind": LIVE_PROPOSER_KIND if induction_enabled else None,
        "proposer_source": "E3AgentPolicy._proposer" if induction_enabled else "none",
        "disable_induction_env": "1" if disable_induction else "unset",
        "lower_bound_note": (
            "offline_core_efficiency_is_lower_bound_when_mismatch_true" if disable_induction else ""
        ),
    }


def _proposer_divergence_detail(
    field: str,
    offline_config: Mapping[str, Any],
    submitted_config: Mapping[str, Any],
) -> str:
    if (
        field == "induction_enabled"
        and offline_config.get(field) is False
        and submitted_config.get(field) is True
        and offline_config.get("disable_induction_env") == "1"
    ):
        return (
            f"offline gate sets {INDUCTION_DISABLE_ENV}=1, disabling induction/proposer; "
            "submitted E3 leaves the escape hatch unset and can call E3AgentPolicy._proposer()"
        )
    if field == "proposer_kind":
        return (
            "offline and submitted paths use different proposer availability; "
            "a None offline proposer means the measured core_efficiency is a bare-explorer lower bound"
        )
    return "offline effective config differs from submitted agent config"


def proposer_config_parity_report(
    *,
    offline_config: Mapping[str, Any],
    submitted_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare gate and submitted proposer config without loading any model."""

    offline = dict(offline_config)
    submitted = dict(submitted_config)
    divergence: list[dict[str, Any]] = []
    for field in ("policy", "cascade", "induction_enabled", "proposer_kind"):
        if offline.get(field) != submitted.get(field):
            divergence.append(
                {
                    "field": field,
                    "offline": offline.get(field),
                    "submitted": submitted.get(field),
                    "detail": _proposer_divergence_detail(field, offline, submitted),
                }
            )
    return {
        "parity_guard": PROPOSER_PARITY_GUARD,
        "proposer_config_mismatch": bool(divergence),
        "proposer_config_divergence": divergence,
        "offline_config": offline,
        "submitted_config": submitted,
    }


def attach_proposer_config_parity(
    measurement: Mapping[str, Any],
    *,
    policy: str,
    disable_induction: bool = OFFLINE_GATE_DISABLE_INDUCTION,
    submitted_agent_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach the offline/live proposer parity result to a gate measurement."""

    report = proposer_config_parity_report(
        offline_config=offline_gate_proposer_config(
            policy=policy,
            disable_induction=disable_induction,
        ),
        submitted_config=submitted_agent_proposer_config(submitted_agent_config),
    )
    out = dict(measurement)
    out.update(
        {
            "proposer_config_mismatch": report["proposer_config_mismatch"],
            "proposer_config_divergence": report["proposer_config_divergence"],
            "proposer_config_parity": report,
            "offline_effective_proposer_config": report["offline_config"],
            "submitted_effective_proposer_config": report["submitted_config"],
        }
    )
    return out


def _measure_game(
    game: str,
    policy: str,
    budget: int,
    cap: int,
    *,
    disable_induction: bool = OFFLINE_GATE_DISABLE_INDUCTION,
) -> dict:
    import os

    cmd = [
        str(REPO / ".venv" / "bin" / "python"),
        str(EVAL),
        "--policy",
        policy,
        "--games",
        "oracle",
        "--only",
        game,
        "--budget",
        str(budget),
    ]
    # Measure the SEARCH/efficiency of the tier-1 explorer cleanly: disable the LLM induction tier so the
    # gate doesn't pay the local llama-server spawn (irrelevant to a search regression; a one-time cost
    # under the real 12h eval). Production submission does NOT set this -> induction runs normally there.
    env = dict(os.environ)
    if disable_induction:
        env[INDUCTION_DISABLE_ENV] = "1"
    else:
        env.pop(INDUCTION_DISABLE_ENV, None)
    # 2026-07-15 fix: this function runs GATE_GAMES.__len__() (currently 8) of these SIMULTANEOUSLY
    # via measure()'s ThreadPoolExecutor(max_workers=8). Each subprocess independently imports
    # numpy/scipy/torch, which default to spawning one OpenMP/OpenBLAS thread PER CORE when no
    # thread-count env var is set (confirmed directly: a single arc_leaderboard_eval.py process
    # spawns exactly 24 threads on this 24-core box). 8 unconstrained subprocesses -> up to 192
    # threads contending for 24 cores -- severe, fully self-inflicted oversubscription that made
    # the gate time out 3 consecutive runs (identical 1/8-solved, 7-timed-out results each time,
    # independent of the concurrent research conductor's own load) even after two real underlying
    # performance bugs were fixed and individually verified fast (REQ-ARC-FCP-5591-3,
    # REQ-CAPSTONE-4556-2). Each game-eval subprocess does small-grid, Python-level search work --
    # it does not benefit from multi-threaded BLAS at this scale -- so pinning every math library
    # to 1 thread per subprocess lets all 8 run truly in parallel within the 24-core budget instead
    # of starving each other.
    for _threads_env in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[_threads_env] = "1"
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=cap, cwd=str(REPO), env=env)
        m = None
        for ln in p.stdout.splitlines():
            if game in ln and "live=L" in ln:
                m = _LINE.search(ln)
        if not m:
            return {
                "game": game,
                "timed_out": False,
                "solved": False,
                "actions": None,
                "efficiency": None,
                "per_level_efficiency": None,
                "levels": 0,
                "deepest_level_reached": 0,
                "reset_replay_steps": 0,
                "forward_walk_hit_rate": 0.0,
            }
        reached, levels, actions = int(m.group(1)), int(m.group(2)), int(m.group(3))
        eff = float(m.group(4)) if m.group(4) is not None else None
        reset_steps = int(m.group(5)) if m.group(5) is not None else 0
        hit_rate = float(m.group(6)) if m.group(6) is not None else 0.0
        return {
            "game": game,
            "timed_out": False,
            "solved": levels >= 1,
            "actions": actions,
            "efficiency": eff,
            "per_level_efficiency": eff,
            "levels": levels,
            "deepest_level_reached": reached,
            "reset_replay_steps": reset_steps,
            "forward_walk_hit_rate": hit_rate,
        }
    except subprocess.TimeoutExpired:
        return {
            "game": game,
            "timed_out": True,
            "solved": False,
            "actions": None,
            "efficiency": None,
            "per_level_efficiency": None,
            "levels": 0,
            "deepest_level_reached": 0,
            "reset_replay_steps": 0,
            "forward_walk_hit_rate": 0.0,
        }


def _normalize_game_row(row: dict) -> dict:
    out = dict(row)
    if out.get("per_level_efficiency") is None and out.get("efficiency") is not None:
        out["per_level_efficiency"] = float(out["efficiency"])
    if out.get("deepest_level_reached") is None:
        out["deepest_level_reached"] = int(out.get("reached", out.get("levels") or 0) or 0)
    out["reset_replay_steps"] = int(out.get("reset_replay_steps") or 0)
    out["forward_walk_hit_rate"] = float(out.get("forward_walk_hit_rate") or 0.0)
    return out


def _call_measure_game(
    game: str,
    policy: str,
    budget: int,
    cap: int,
    *,
    disable_induction: bool,
) -> dict:
    try:
        signature = inspect.signature(_measure_game)
    except (TypeError, ValueError):
        accepts_disable_induction = True
    else:
        parameters = signature.parameters.values()
        accepts_disable_induction = any(
            parameter.name == "disable_induction" or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
    if accepts_disable_induction:
        return _measure_game(
            game,
            policy,
            budget,
            cap,
            disable_induction=disable_induction,
        )
    return _measure_game(game, policy, budget, cap)


def measure(
    policy: str,
    budget: int,
    cap: int,
    *,
    disable_induction: bool = OFFLINE_GATE_DISABLE_INDUCTION,
) -> dict:
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = [
            _normalize_game_row(row)
            for row in ex.map(
                lambda g: _call_measure_game(
                    g,
                    policy,
                    budget,
                    cap,
                    disable_induction=disable_induction,
                ),
                GATE_GAMES,
            )
        ]
    solved = [r for r in rows if r["solved"]]
    acts = [r["actions"] for r in solved if r["actions"] is not None]
    per_level_efficiency_by_game = {
        r["game"]: float(r["per_level_efficiency"])
        for r in rows
        if r.get("per_level_efficiency") is not None
    }
    navigation_by_game = {
        r["game"]: {
            "reset_replay_steps": int(r.get("reset_replay_steps") or 0),
            "forward_walk_hit_rate": float(r.get("forward_walk_hit_rate") or 0.0),
        }
        for r in rows
    }
    measurement = {
        "policy": policy,
        "games": GATE_GAMES,
        "per_game": rows,
        "action_metric": dict(CANONICAL_ACTION_METRIC),
        "solved_count": len(solved),
        "median_actions_on_solved": (median(acts) if acts else None),
        "median_actions_on_core": median(
            [
                _actions_by_game({"per_game": rows, "action_metric": CANONICAL_ACTION_METRIC}).get(
                    game, BIG_ACTIONS
                )
                for game in CANONICAL_CORE_GAMES
            ]
        ),
        "total_actions_on_solved": (sum(acts) if acts else None),
        "timed_out_count": sum(1 for r in rows if r["timed_out"]),
        # CORE set-containment keys (2026-06-20): the verdict compares the SAME games across
        # baseline/treatment, so a knife-edge marginal solve flipping due to order-perturbation
        # noise (A1/A2 lesson: a 5%-recall prune that removes ~nothing still reshuffled the chaotic
        # ~7800-action trajectory and dropped m0r0) is NOT counted as a regression. The CORE is the
        # set of games the BASELINE solves; a lever must preserve every CORE solve (solve-rate
        # non-regression by set, not by count) and cut median actions ON THAT FIXED SET.
        "solved_games": sorted(r["game"] for r in solved),
        "actions_by_game": {r["game"]: r["actions"] for r in solved if r["actions"] is not None},
        # PER-LEVEL efficiency (2026-06-20): the REAL leaderboard score is per-level
        # sum(min(human/agent_per_level,1)^2), NOT total actions. The eval now emits it (eff=); this is the
        # PRIMARY metric the verdict judges (median actions is demoted to a wall-clock/compute-budget proxy).
        "efficiency_by_game": {
            r["game"]: r["efficiency"] for r in solved if r.get("efficiency") is not None
        },
        "per_level_efficiency_by_game": per_level_efficiency_by_game,
        "deepest_level_by_game": {
            r["game"]: int(r.get("deepest_level_reached") or 0) for r in rows
        },
        "navigation_by_game": navigation_by_game,
        "core_efficiency": round(
            sum(
                r["efficiency"]
                for r in rows
                if r["game"] in CANONICAL_CORE_GAMES and r.get("efficiency") is not None
            ),
            4,
        ),
    }
    return attach_proposer_config_parity(
        measurement,
        policy=policy,
        disable_induction=disable_induction,
    )


def _efficiency_by_game(measurement: dict) -> dict[str, float]:
    eff = measurement.get("efficiency_by_game")
    if isinstance(eff, dict) and eff:
        return {str(g): float(v) for g, v in eff.items() if v is not None}
    per_level = measurement.get("per_level_efficiency_by_game")
    if isinstance(per_level, dict) and per_level:
        return {str(g): float(v) for g, v in per_level.items() if v is not None}
    out: dict[str, float] = {}
    for row in measurement.get("per_game", []) or []:
        if isinstance(row, dict) and row.get("efficiency") is not None:
            out[str(row["game"])] = float(row["efficiency"])
        elif isinstance(row, dict) and row.get("per_level_efficiency") is not None:
            out[str(row["game"])] = float(row["per_level_efficiency"])
    return out


def _navigation_by_game(measurement: dict) -> dict[str, dict[str, float]]:
    nav = measurement.get("navigation_by_game")
    if isinstance(nav, dict) and nav:
        return {
            str(game): {
                "reset_replay_steps": int((value or {}).get("reset_replay_steps") or 0),
                "forward_walk_hit_rate": float((value or {}).get("forward_walk_hit_rate") or 0.0),
            }
            for game, value in nav.items()
            if isinstance(value, dict)
        }
    out: dict[str, dict[str, float]] = {}
    for row in measurement.get("per_game", []) or []:
        if not isinstance(row, dict) or row.get("game") is None:
            continue
        diagnostics = row.get("navigation_diagnostics")
        if not isinstance(diagnostics, dict):
            diagnostics = row
        out[str(row["game"])] = {
            "reset_replay_steps": int(diagnostics.get("reset_replay_steps") or 0),
            "forward_walk_hit_rate": float(diagnostics.get("forward_walk_hit_rate") or 0.0),
        }
    return out


def _action_metric_field(measurement: dict) -> str:
    metric = measurement.get("action_metric")
    if isinstance(metric, dict) and metric.get("field"):
        return str(metric["field"])
    if measurement.get("action_field"):
        return str(measurement["action_field"])
    rows = measurement.get("per_game") or []
    if any(isinstance(r, dict) and CANONICAL_ACTION_FIELD in r for r in rows):
        return CANONICAL_ACTION_FIELD
    if any(isinstance(r, dict) and "actions_to_first_levelup" in r for r in rows):
        return "actions_to_first_levelup"
    return CANONICAL_ACTION_FIELD


def _actions_by_game(measurement: dict) -> dict[str, int]:
    actions = measurement.get("actions_by_game")
    if isinstance(actions, dict) and actions:
        return {str(game): int(value) for game, value in actions.items() if value is not None}
    field = _action_metric_field(measurement)
    out: dict[str, int] = {}
    for row in measurement.get("per_game", []) or []:
        if not isinstance(row, dict) or row.get("solved") is not True:
            continue
        value = row.get(field)
        if value is not None:
            out[str(row["game"])] = int(value)
    return out


def _baseline_core(base: dict) -> set[str]:
    base_acts = _actions_by_game(base)
    return set(base.get("solved_games") or base_acts.keys())


def _metric_compatibility_error(cur: dict, base: dict) -> str | None:
    cur_field = _action_metric_field(cur)
    base_field = _action_metric_field(base)
    if cur_field != base_field:
        return (
            f"REGRESSION: action metric mismatch treatment={cur_field} baseline={base_field}; "
            "baseline and treatment must use the identical field"
        )
    return None


def validate_canonical_baseline(base: dict) -> dict:
    errors: list[str] = []
    games = list(base.get("games") or [])
    base_acts = _actions_by_game(base)
    core = sorted(_baseline_core(base))
    computed_median = median([base_acts.get(game, BIG_ACTIONS) for game in CANONICAL_CORE_GAMES])
    reported_median = base.get("median_actions_on_solved")
    # Game-set + CORE-set are metric-agnostic (the cherry-pick guard): always enforced.
    if games != list(CANONICAL_GAME_SET):
        errors.append(f"baseline games must equal the fixed 8-game set {list(CANONICAL_GAME_SET)}")
    if core != sorted(CANONICAL_CORE_GAMES):
        errors.append(f"baseline CORE must equal {sorted(CANONICAL_CORE_GAMES)}")
    if _efficiency_by_game(base):
        # NEW per-level efficiency baseline (the real metric, 2026-06-20): the retired total-actions 7760
        # control no longer applies (re-baselining with the per-level eval legitimately moves total
        # actions). The cherry-pick guard here is "efficiency present for every CORE game" so the baseline
        # cannot silently drop the real metric. Do NOT assert the 7760 total-actions control.
        eff = _efficiency_by_game(base)
        missing = [g for g in CANONICAL_CORE_GAMES if g not in eff]
        if missing:
            errors.append(f"baseline missing per-level efficiency for CORE games {missing}")
        lp85_efficiency = eff.get("lp85")
        if (
            lp85_efficiency is not None
            and float(lp85_efficiency) < CANONICAL_LP85_PER_LEVEL_EFFICIENCY_FLOOR
        ):
            errors.append(
                "baseline lp85 per-level efficiency "
                f"{float(lp85_efficiency):.4f} below floor "
                f"{CANONICAL_LP85_PER_LEVEL_EFFICIENCY_FLOOR:.4f}"
            )
    else:
        # LEGACY total-actions baseline (no efficiency, e.g. unit fixtures): keep the 7760 cherry-pick guard.
        if _action_metric_field(base) != CANONICAL_ACTION_FIELD:
            errors.append("baseline action metric must use total actions")
        if {
            game: base_acts.get(game) for game in CANONICAL_CORE_GAMES
        } != CANONICAL_BASELINE_ACTIONS_BY_GAME:
            errors.append("baseline CORE action map moved from the verified 7760 control")
        if float(computed_median) != CANONICAL_BASELINE_MEDIAN_ACTIONS:
            errors.append("computed baseline CORE median must remain 7760")
        if reported_median is None or float(reported_median) != CANONICAL_BASELINE_MEDIAN_ACTIONS:
            errors.append("reported baseline median_actions_on_solved must remain 7760")
    return {
        "ok": not errors,
        "errors": errors,
        "canonical_game_set": list(CANONICAL_GAME_SET),
        "canonical_baseline_median_actions": CANONICAL_BASELINE_MEDIAN_ACTIONS,
        "canonical_lp85_per_level_efficiency_floor": CANONICAL_LP85_PER_LEVEL_EFFICIENCY_FLOOR,
        "core_games": sorted(CANONICAL_CORE_GAMES),
        "action_metric_field": CANONICAL_ACTION_FIELD,
    }


def _verdict(cur: dict, base: dict) -> tuple[bool, str]:
    """CORE set-containment verdict (2026-06-20 redesign).

    The OLD verdict compared raw solved_COUNT, so a lever that merely reordered the chaotic
    near-budget search and flipped one knife-edge solve 4<->3 FAILed automatically regardless of
    merit (A1/A2 both died this way with their positive_control passing -> the metric, not the
    lever, was broken). The NEW verdict:
      * CORE := the games the BASELINE solves. A lever must preserve EVERY core solve (set
        containment, not count) -- this is the only relaxation that still FAILs a config that
        trades core solves for fringe ones (e.g. A2 swapping 3 core for 2 fringe).
      * median actions is measured on the FIXED CORE denominator (+inf for any core game the
        treatment failed to solve), so savings are credited on the same games, never gamed by
        dropping a hard one.
      * new solves OUTSIDE core are a reported BONUS, NEVER netted against a core loss.
    A legacy fallback (raw count) keeps the gate working against an old baseline JSON until the
    next `--update-baseline` persists the CORE keys.
    """
    metric_error = _metric_compatibility_error(cur, base)
    if metric_error:
        return False, metric_error
    # Reconstruct baseline per-game actions from the new key, else from the legacy per_game rows.
    base_acts = _actions_by_game(base)
    core = _baseline_core(base)
    if not core:
        # No baseline solves recorded at all -> fall back to the legacy count check.
        bs, cs = base.get("solved_count", 0), cur["solved_count"]
        return (
            cs >= bs and cs > 0,
            f"legacy count check: solved {cs} vs baseline {bs} (run --update-baseline for CORE)",
        )
    cur_solved = set(cur.get("solved_games") or [])
    lost = sorted(core - cur_solved)
    if lost:
        return False, f"REGRESSION: lost CORE solves {lost} (core={sorted(core)})"
    bonus = sorted(cur_solved - core)  # extra solves: reported, NEVER netted against a core loss

    # PRIMARY metric: per-level efficiency on CORE -- the REAL leaderboard score, sum over solved levels of
    # min(human_actions/agent_actions_for_level, 1)^2 (HIGHER is better). This replaces the median-actions
    # check, which measured TOTAL actions and scored an efficient-but-over-running solve at ~0 (the lp85
    # bug: solved L1 in 20 actions == human-class but ran to 7792 hunting unreachable levels -> old metric
    # said 0; per-level metric says 0.72 and the over-run is a wall-clock cost, not a score cost).
    base_eff = _efficiency_by_game(base)
    cur_eff = _efficiency_by_game(cur)
    if base_eff and not cur_eff:
        # The baseline uses the real per-level metric but the treatment emitted no efficiency (broken
        # eval / missing eff=). Refuse -- do NOT silently fall back to the retired total-actions metric,
        # which could PASS a real efficiency regression (adversarial review SF-2, 2026-06-20).
        return False, (
            "REGRESSION: could not measure per-level efficiency for the current config "
            "(baseline has it, treatment does not); refusing to judge on the retired "
            "total-actions fallback"
        )
    if base_eff and cur_eff:
        bce = round(sum(base_eff.get(g, 0.0) for g in core), 4)
        cce = round(sum(cur_eff.get(g, 0.0) for g in core), 4)
        # PER-GAME non-inferiority (SF-3, adversarial review): the CORE efficiency sum is lopsided (one
        # game ~ 100% of it), so a sum-only check lets a regression on the dominant game hide behind a
        # tiny gain elsewhere. Guard each CORE game whose baseline efficiency is non-trivial (>0.01).
        regressed = [
            g
            for g in core
            if base_eff.get(g, 0.0) > 0.01
            and cur_eff.get(g, 0.0) < base_eff.get(g, 0.0) * EFFICIENCY_DROP_SLACK
        ]
        if regressed:
            return False, (
                f"REGRESSION: CORE games lost per-level efficiency {regressed} "
                f"(per-game non-inferiority; the REAL leaderboard metric)"
            )
        if bce > 0 and cce < bce * EFFICIENCY_DROP_SLACK:
            return False, (
                f"REGRESSION: CORE per-level efficiency sum {cce} < baseline {bce} "
                f"(the REAL leaderboard metric: min((human/agent)^2*100,115), index-weighted)"
            )
        tag = "IMPROVED" if (cce > bce or bonus) else "non-inferior"
        msg = f"PASS ({tag}): CORE per-level efficiency {cce} vs baseline {bce}"
        if bonus:
            msg += f"; BONUS solves {bonus}"
        return True, msg

    # FALLBACK (no per-level efficiency in cur+base -- legacy baseline / unit fixtures): median TOTAL
    # actions, a wall-clock proxy. `--update-baseline` re-measures with the new eval and persists
    # efficiency, which activates the primary metric above.
    cur_acts = _actions_by_game(cur)
    cm = median([cur_acts.get(g, BIG_ACTIONS) for g in core])
    bm = median([base_acts.get(g, BIG_ACTIONS) for g in core])
    if cm > bm * EFFICIENCY_SLACK:
        return False, (
            f"REGRESSION: CORE median actions {cm} > baseline {bm} x{EFFICIENCY_SLACK} "
            f"(wall-clock fallback; re-run --update-baseline to use per-level efficiency)"
        )
    tag = "IMPROVED" if (cm < bm or bonus) else "non-inferior"
    msg = f"PASS ({tag}): CORE {sorted(core)} median actions {cm} vs baseline {bm}"
    if bonus:
        msg += f"; BONUS solves {bonus}"
    return True, msg


def dashboard_row(cur: dict, base: dict, *, lever: str) -> dict:
    ok, msg = _verdict(cur, base)
    core = _baseline_core(base)
    cur_solved = set(cur.get("solved_games") or [])
    cur_acts = _actions_by_game(cur)
    base_acts = _actions_by_game(base)
    cur_median = median([cur_acts.get(game, BIG_ACTIONS) for game in core]) if core else None
    base_median = median([base_acts.get(game, BIG_ACTIONS) for game in core]) if core else None
    bonus = sorted(cur_solved - core)
    nav_warning = _nav_regression_warning(cur, base, core=core)
    return {
        "lever": lever,
        "metric_action_field": _action_metric_field(cur),
        "canonical_game_set": list(CANONICAL_GAME_SET),
        "core_games": sorted(core),
        "median_actions_on_core": cur_median,
        "baseline_median_actions_on_core": base_median,
        "actions_saved_vs_baseline": (
            None if cur_median is None or base_median is None else float(base_median - cur_median)
        ),
        "core_solves_preserved": core.issubset(cur_solved),
        "lost_core_solves": sorted(core - cur_solved),
        "bonus_solves": bonus,
        "verdict_pass": ok,
        "verdict": msg,
        "navigation_by_game": _navigation_by_game(cur),
        "baseline_navigation_by_game": _navigation_by_game(base),
        "nav_regression_warning": nav_warning,
        "nav_metric_role": "secondary_wall_clock_warning_not_score_metric",
        "proposer_config_mismatch": bool(cur.get("proposer_config_mismatch")),
        "proposer_config_divergence": cur.get("proposer_config_divergence") or [],
    }


def positive_control(base: dict) -> dict:
    base_acts = _actions_by_game(base)
    core = _baseline_core(base)
    improved = {game: max(1, int(base_acts[game]) - 1000) for game in core}
    base_eff = _efficiency_by_game(base)
    improved_eff = None
    if base_eff:
        improved_eff = {game: float(base_eff.get(game, 0.0)) for game in core}
        for game in sorted(core):
            if game != "lp85":
                improved_eff[game] = round(improved_eff.get(game, 0.0) + 0.1, 4)
                break
        else:
            improved_eff["lp85"] = round(improved_eff.get("lp85", 0.0) + 0.1, 4)
    cur = {
        "games": list(CANONICAL_GAME_SET),
        "action_metric": dict(CANONICAL_ACTION_METRIC),
        "solved_count": len(improved),
        "solved_games": sorted(improved),
        "actions_by_game": improved,
        "median_actions_on_solved": median(improved.values()),
    }
    if improved_eff is not None:
        cur["efficiency_by_game"] = improved_eff
        cur["per_level_efficiency_by_game"] = improved_eff
        cur["core_efficiency"] = round(sum(improved_eff.values()), 4)
    row = dashboard_row(cur, base, lever="positive_control")
    return {
        "passed": bool(
            row["verdict_pass"]
            and row["median_actions_on_core"] < row["baseline_median_actions_on_core"]
        ),
        "dashboard_row": row,
    }


def _nav_regression_warning(cur: dict, base: dict, *, core: set[str] | None = None) -> str:
    cur_nav = _navigation_by_game(cur)
    base_nav = _navigation_by_game(base)
    if not cur_nav or not base_nav:
        return ""
    cur_actions = _actions_by_game(cur)
    base_actions = _actions_by_game(base)
    games = sorted(core or set(base_nav))
    regressed = [
        game
        for game in games
        if cur_actions.get(game) == base_actions.get(game)
        and cur_nav.get(game, {}).get("reset_replay_steps", 0)
        > base_nav.get(game, {}).get("reset_replay_steps", 0)
    ]
    if not regressed:
        return ""
    return (
        f"WARN: reset_replay_steps increased for {regressed} at equal actions "
        "(secondary wall-clock signal; per-level efficiency remains the score metric)"
    )


def _solved_set(measurement: dict) -> set[str]:
    solved = measurement.get("solved_games")
    if solved is not None:
        return {str(game) for game in solved}
    return {
        str(row["game"]) for row in measurement.get("per_game", []) if row.get("solved") is True
    }


def select_headroom_budget(
    by_budget: dict[int, dict], *, candidates=HEADROOM_BUDGET_CANDIDATES
) -> tuple[int, list[dict]]:
    rows = []
    selected = None
    for budget in candidates:
        comparison_budget = int(budget * 1.5)
        solved = sorted(_solved_set(by_budget.get(int(budget), {})))
        comparison_solved = sorted(_solved_set(by_budget.get(comparison_budget, {})))
        stable = bool(solved and solved == comparison_solved)
        rows.append(
            {
                "budget": int(budget),
                "comparison_budget": comparison_budget,
                "solved_games": solved,
                "comparison_solved_games": comparison_solved,
                "stable_vs_1_5x": stable,
            }
        )
        if selected is None and stable:
            selected = int(budget)
    return (selected or DEFAULT_BUDGET), rows


def measure_headroom_budget(
    policy: str, cap: int, *, candidates=HEADROOM_BUDGET_CANDIDATES
) -> dict:
    measurements: dict[int, dict] = {}
    rows: list[dict] = []
    selected = DEFAULT_BUDGET
    for index, budget in enumerate(candidates):
        comparison_budget = int(budget * 1.5)
        for needed in (int(budget), comparison_budget):
            if needed not in measurements:
                measurements[needed] = measure(policy, needed, cap)
        selected, rows = select_headroom_budget(
            measurements,
            candidates=tuple(int(candidate) for candidate in candidates[: index + 1]),
        )
        if rows and rows[-1]["stable_vs_1_5x"]:
            break
    return {
        "selected_default_budget": selected,
        "measured": True,
        "rows": rows,
        "measurements_by_budget": {
            str(budget): measurements[budget] for budget in sorted(measurements)
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--policy", default="e3")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--cap", type=int, default=115)
    ap.add_argument("--lever", default="submitted_default")
    ap.add_argument("--json", action="store_true")
    return ap


def main() -> int:
    a = _build_parser().parse_args()

    print(
        f"[gate] measuring current submitted config (policy={a.policy}) on {len(GATE_GAMES)} games "
        f"(budget {a.budget}, {a.cap}s cap each)...",
        flush=True,
    )
    cur = measure(a.policy, a.budget, a.cap)

    if a.update_baseline:
        # 2026-07-15 fix: `cur["solved_games"]` is EVERY game that solved this measurement,
        # including bonus solves outside CANONICAL_CORE_GAMES (the --check path already handles
        # this correctly -- bonus solves are reported separately, never netted against core; see
        # _verdict()'s `bonus = sorted(cur_solved - core)`). But naively persisting the full
        # solved_games as the NEW baseline's identity broke validate_canonical_baseline's
        # cherry-pick guard the first time a measurement had bonus solves (7/8 solved: 4 core +
        # cd82/ft09/su15 bonus) -- the guard requires the baseline's core to be EXACTLY
        # CANONICAL_CORE_GAMES, and 7 games != 4. The baseline's job is to anchor "core" to the
        # canonical 4 regardless of how many bonus games a given measurement happened to solve;
        # only core solves belong in solved_games's role as the core-identity field.
        candidate = {
            **cur,
            "solved_games": sorted(set(cur.get("solved_games") or []) & set(CANONICAL_CORE_GAMES)),
            "note": "verified baseline (update only after a real improvement + successful submit)",
        }
        guard = validate_canonical_baseline(candidate)
        if not guard["ok"]:
            print(
                f"[gate] REFUSED baseline update: canonical baseline guard failed {guard['errors']}"
            )
            if a.json:
                print(
                    json.dumps(
                        {
                            "pass": False,
                            "verdict": f"REGRESSION: canonical baseline guard failed {guard['errors']}",
                            "baseline_guard": guard,
                            "proposer_config_mismatch": cur.get("proposer_config_mismatch"),
                            "proposer_config_divergence": cur.get("proposer_config_divergence"),
                            "proposer_config_parity": cur.get("proposer_config_parity"),
                            "current": cur,
                        },
                        indent=2,
                    )
                )
            return 1
        BASELINE.write_text(json.dumps(candidate, indent=2))
        print(
            f"[gate] baseline UPDATED: solved {cur['solved_count']}, "
            f"median actions/solve {cur['median_actions_on_solved']}"
        )
        return 0

    if not BASELINE.exists():
        print(
            f"[gate] NO baseline at {BASELINE} -- run --update-baseline once on a trusted config first."
        )
        return 2
    base = json.loads(BASELINE.read_text())
    guard = validate_canonical_baseline(base)
    if not guard["ok"]:
        ok, msg = False, f"REGRESSION: canonical baseline guard failed {guard['errors']}"
        row = dashboard_row(cur, base, lever=a.lever)
        row["verdict_pass"] = False
        row["verdict"] = msg
    else:
        ok, msg = _verdict(cur, base)
        row = dashboard_row(cur, base, lever=a.lever)
    if a.json:
        print(
            json.dumps(
                {
                    "pass": ok,
                    "verdict": msg,
                    "lever_dashboard_row": row,
                    "baseline_guard": guard,
                    "proposer_config_mismatch": cur.get("proposer_config_mismatch"),
                    "proposer_config_divergence": cur.get("proposer_config_divergence"),
                    "proposer_config_parity": cur.get("proposer_config_parity"),
                    "current": cur,
                    "baseline": {
                        k: base.get(k)
                        for k in (
                            "solved_count",
                            "median_actions_on_solved",
                            "total_actions_on_solved",
                        )
                    },
                },
                indent=2,
            )
        )
    else:
        print(
            f"[gate] current : solved {cur['solved_count']}, median actions/solve "
            f"{cur['median_actions_on_solved']}, timed_out {cur['timed_out_count']}"
        )
        print(
            f"[gate] baseline: solved {base.get('solved_count')}, median actions/solve "
            f"{base.get('median_actions_on_solved')}"
        )
        print(f"[gate] {'PASS' if ok else 'FAIL'}: {msg}")
        print(
            f"[gate] lever {a.lever}: median_actions_on_core={row['median_actions_on_core']}, "
            f"core_solves_preserved={row['core_solves_preserved']}, bonus={row['bonus_solves']}"
        )
        if cur.get("proposer_config_mismatch"):
            print(f"[gate] proposer_config_mismatch=true: {cur.get('proposer_config_divergence')}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
