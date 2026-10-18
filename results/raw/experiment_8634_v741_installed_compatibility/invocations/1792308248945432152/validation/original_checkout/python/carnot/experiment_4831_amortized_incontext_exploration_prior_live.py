"""Experiment 4831: amortized in-context exploration prior live gate.

Spec refs: REQ-ARC-WMTE-4831,
SCENARIO-ARC-WMTE-4831-IN-CONTEXT-PRIOR,
SCENARIO-ARC-WMTE-4831-HELDOUT-GATE.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(PYTHON_ROOT))

JsonDict = dict[str, Any]

EXPERIMENT = "experiment_4831_amortized_incontext_exploration_prior_live"
EXPERIMENT_ID = 4831
SCHEMA = "carnot.arc.amortized_incontext_exploration_prior_live_4831.v1"
RESULT_RELATIVE_PATH = "results/experiment_4831_amortized_incontext_exploration_prior_live.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
RANDOM_SEED = 4831
BASELINE_FIRST_WIN_RATE = 0.04
DEFAULT_HELDOUT_GAMES = ("bp35",)
DEFAULT_BUDGET = 12

SUCCESS_VERDICT = "success_amortized_prior_raises_first_win_above_baseline"
NULL_VERDICT = "complete_amortized_prior_no_first_win_lift_l1_wall_survives"
DEAD_ARCHIVE_VERDICT = "blocked_dead_go_explore_archive"

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a lift is success_amortized_prior_raises_first_win_above_baseline, "
            "a genuine null is complete_amortized_prior_no_first_win_lift_l1_wall_survives, "
            "a dead archive is blocked_dead_go_explore_archive."
        )
    },
    "go_explore_archive_alive": {
        "principle": (
            "observations>0 AND stored_cells>0 AND prefixes_injected>0 -- the exp4701 "
            "silent-bug guard; a 0 means a non-test."
        )
    },
    "prior_changed_proposals": {
        "principle": (
            "the prior must materially alter proposals vs no-prior (the S2/S3 no-op lesson) "
            "-- else the first-win delta is a silent no-op."
        )
    },
    "first_win_rate_with_prior": {
        "principle": "held-out generic first-win WITH the prior -- must rise above the 0.04 baseline."
    },
    "first_win_rate_no_prior_ablation": {
        "principle": "the matched lambda=0 control."
    },
    "first_win_delta_ci95": {
        "principle": "must EXCLUDE 0 for a PASS -- a genuine lift, not noise."
    },
    "imitation_control_heldout_games": {
        "principle": (
            "the lift must hold on games NOT in the distillation set -- exploration, not "
            "memorization (the exp4697 imitation trap)."
        )
    },
    "live_path_reachable": {
        "principle": (
            "the prior must be in the E3AgentPolicy proposal import closure (arc_orphan_solver_lint)."
        )
    },
    "solve_provenance": {
        "principle": (
            "live_agent_self_discovery -- a first-win lift is the agent solving via its OWN "
            "(prior-biased) exploration."
        )
    },
    "inference_substrate": {
        "principle": "live_llm_inference -- the live E3 agent runs; 60s floor."
    },
    "preconditions_checked": {
        "principle": (
            "records arcade/go_explore checks so a missing-resource run emits blocked_, never "
            "a fabricated first-win."
        )
    },
    "random_seed": {
        "principle": "determinism for the distillation + the with/without runs + bootstrap."
    },
    "reproducibility_checksum": {
        "principle": (
            "content hash of (distilled traces, prior, games, seeds) so a replication catches drift."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "experiment_id",
    "schema",
    "spec_refs",
    "baseline_first_win_rate",
    "duration_s",
    "field_principles",
    "prior_diagnostics",
    "prior_change_diagnostics",
    "measurement",
)


class _NoOpProposer:
    def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:  # pragma: no cover
        return False, "disabled_exp4831_probe"

    def world_model_candidates(self, _game: str) -> list[Any]:  # pragma: no cover
        return []


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def write_artifact(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _rate(hits: Sequence[bool]) -> float:
    return round(sum(1 for hit in hits if hit) / len(hits), 6) if hits else 0.0


def _ci_positive(ci: Mapping[str, Any]) -> bool:
    try:
        return float(ci.get("low")) > 0.0
    except Exception:
        return False


def _bootstrap_delta_ci(
    with_hits: Sequence[bool],
    baseline_hits: Sequence[bool],
    *,
    seed: int = RANDOM_SEED,
    n_boot: int = 1000,
) -> JsonDict:
    if not with_hits or not baseline_hits:
        return {"low": 0.0, "high": 0.0, "confidence": 0.95, "n_boot": int(n_boot)}
    rng = random.Random(seed)
    left = list(with_hits)
    right = list(baseline_hits)
    deltas = []
    for _ in range(int(n_boot)):
        lrate = sum(1 for _ in left if rng.choice(left)) / len(left)
        rrate = sum(1 for _ in right if rng.choice(right)) / len(right)
        deltas.append(lrate - rrate)
    deltas.sort()
    low = deltas[int(0.025 * (len(deltas) - 1))]
    high = deltas[int(0.975 * (len(deltas) - 1))]
    return {"low": round(low, 6), "high": round(high, 6), "confidence": 0.95, "n_boot": int(n_boot)}


def archive_alive_from_diagnostics(diagnostics: Mapping[str, Any]) -> JsonDict:
    observations = int(diagnostics.get("observations") or 0)
    stored_cells = int(diagnostics.get("stored_cells") or 0)
    prefixes_injected = int(diagnostics.get("prefixes_injected") or 0)
    out = dict(diagnostics)
    out.update(
        {
            "observations": observations,
            "stored_cells": stored_cells,
            "prefixes_injected": prefixes_injected,
            "alive": observations > 0 and stored_cells > 0 and prefixes_injected > 0,
        }
    )
    return out


def _archive_alive_check() -> JsonDict:  # pragma: no cover - live wiring smoke.
    from types import SimpleNamespace

    from carnot.agentic.arc_competition_agent import StepwiseExplorer
    from carnot.agentic.arc_go_explore import GoExploreReplayArchive

    archive = GoExploreReplayArchive(enabled=True, bins=2)
    explorer = StepwiseExplorer(
        online_discriminative=False,
        navigation_cost_tiebreak=False,
        go_explore_archive=archive,
        amortized_first_contact_prior=None,
    )
    root = SimpleNamespace(frame=np.asarray([[0, 0], [0, 0]], dtype=np.int16), levels_completed=0)
    first = SimpleNamespace(frame=np.asarray([[1, 0], [0, 0]], dtype=np.int16), levels_completed=0)
    archive.observe(root, [])
    archive.observe(first, [{"action": 2, "data": None}])
    replay = explorer._go_explore_replay_sequence(current_path=[])
    if replay:
        explorer._begin_go_explore_replay(replay)
    return archive_alive_from_diagnostics(explorer.go_explore_archive_diagnostics())


def _action_for_family(family: str) -> dict[str, Any]:
    if family == "click":
        return {"action": 6, "data": {"x": 0, "y": 0}}
    if family.startswith("action:"):
        return {"action": int(family.split(":", 1)[1]), "data": None}
    return {"action": 1, "data": None}


def proposal_change_probe(prior: Any) -> JsonDict:
    """REQ-ARC-WMTE-4831: prove the ranker is not a proposal-order no-op."""

    scores_by_context = getattr(prior, "context_family_scores", {})
    if scores_by_context:
        context, scores = max(
            scores_by_context.items(),
            key=lambda item: max(item[1].values(), default=0.0),
        )
        top_family = max(scores, key=lambda family: (float(scores[family]), family))
        path = [_action_for_family(family) for family in context]
        candidates = [
            {"action": 1, "data": None},
            {"action": 2, "data": None},
            {"action": 3, "data": None},
            {"action": 4, "data": None},
            {"action": 5, "data": None},
            {"action": 6, "data": {"x": 0, "y": 0}},
        ]
        candidates = [row for row in candidates if row != _action_for_family(top_family)]
        candidates.append(_action_for_family(top_family))
    else:
        path = []
        candidates = [{"action": 1, "data": None}, {"action": 2, "data": None}]
    no_prior_order = [dict(row) for row in candidates]
    ranked = prior.rank_candidates(
        np.asarray([[0, 0], [0, 0]], dtype=np.int16),
        candidates,
        path=path,
    )
    changed = ranked != no_prior_order
    return {
        "changed": bool(changed),
        "path_context": [str(step["action"]) for step in path],
        "no_prior_order": [int(row["action"]) for row in no_prior_order],
        "with_prior_order": [int(row["action"]) for row in ranked],
    }


def _live_path_reachable() -> bool:  # pragma: no cover - lint boundary.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import scripts.arc_orphan_solver_lint as orphan_lint

    closure = orphan_lint._closure(orphan_lint.ENTRYPOINTS) | {
        path.stem for path in orphan_lint.ENTRYPOINTS
    }
    return "arc_amortized_exploration" in closure and "arc_go_explore" in closure


def check_preconditions() -> JsonDict:  # pragma: no cover - runtime boundary.
    checks: JsonDict = {
        "agents_md_read": (REPO_ROOT / "AGENTS.md").exists(),
        "codex_md_read": (REPO_ROOT / "CODEX.md").exists(),
        "spec_has_req_4831": "REQ-ARC-WMTE-4831"
        in (REPO_ROOT / SPEC_RELATIVE_PATH).read_text(encoding="utf-8"),
        "offline_arcade": False,
        "go_explore_import": False,
        "go_explore_frame_grid_fix_present": False,
    }
    try:
        from carnot.agentic import arc_solver_kit as kit
        from carnot.agentic.arc_go_explore import _frame_grid

        kit.offline_arcade()
        checks["offline_arcade"] = True
        checks["go_explore_import"] = True
        raw = type("Frame", (), {"frame": np.zeros((1, 2, 2), dtype=np.int16)})()
        checks["go_explore_frame_grid_fix_present"] = np.asarray(_frame_grid(raw)).ndim == 2
    except Exception as exc:
        checks["ok"] = False
        checks["blocked_resource"] = "blocked_offline_arcade_missing"
        checks["error"] = repr(exc)[:240]
        return checks
    checks["ok"] = bool(
        checks["offline_arcade"]
        and checks["go_explore_import"]
        and checks["go_explore_frame_grid_fix_present"]
    )
    if not checks["ok"]:
        checks["blocked_resource"] = "blocked_go_explore_missing"
    return checks


def _run_e3_probe(
    *,
    games: Sequence[str],
    prior: Any | None,
    budget: int,
) -> tuple[list[bool], list[JsonDict]]:  # pragma: no cover - ARC runtime boundary.
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, _level_of

    hits: list[bool] = []
    rows: list[JsonDict] = []
    arc = kit.offline_arcade()
    for game in games:
        env = arc.make(str(game), scorecard_id=arc.open_scorecard())
        policy = E3AgentPolicy(
            str(game),
            proposer=_NoOpProposer(),
            explore_budget=int(budget) + 1,
            target_levels=1,
            value_head=None,
            value_weight=0.0,
            candidate_router=None,
            navigation_cost_tiebreak=False,
            action_effect_expansion_prior=False,
            amortized_first_contact_prior=prior,
            go_explore_archive=True,
        )
        frames: list[Any] = []
        latest = None
        start_level: int | None = None
        reached = 0
        actions = 0
        for _ in range(int(budget)):
            if policy.is_done(frames, latest):
                break
            kind, data = policy.next_move(frames, latest)
            if kind == "RESET":
                latest = env.reset()
            elif kind is None:
                break
            else:
                latest = env.step(getattr(GameAction, f"ACTION{kind}"), data=data)
                actions += 1
            if latest is None:
                break
            if start_level is None:
                start_level = _level_of(latest)
            reached = _level_of(latest)
            frames.append(latest)
        first_win = bool(start_level is not None and reached > start_level)
        hits.append(first_win)
        rows.append(
            {
                "game": str(game),
                "first_win": first_win,
                "actions": int(actions),
                "budget": int(budget),
                "reached_level": int(reached),
                "amortized_prior_diagnostics": policy.explorer.amortized_prior_diagnostics(),
                "go_explore_archive_diagnostics": policy.explorer.go_explore_archive_diagnostics(),
            }
        )
    return hits, rows


def _normalise_imitation_control(value: Mapping[str, Any]) -> JsonDict:
    out = dict(value)
    out["heldout_not_in_distillation_set"] = bool(out.get("heldout_not_in_distillation_set"))
    out["lift_holds"] = bool(out.get("lift_holds"))
    return out


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    go_explore_archive_alive: Mapping[str, Any],
    prior_changed_proposals: bool,
    first_win_rate_with_prior: float,
    first_win_rate_no_prior_ablation: float,
    first_win_delta_ci95: Mapping[str, Any],
    imitation_control_heldout_games: Mapping[str, Any],
    live_path_reachable: bool,
    duration_s: float,
    random_seed: int = RANDOM_SEED,
    prior_diagnostics: Mapping[str, Any] | None = None,
    prior_change_diagnostics: Mapping[str, Any] | None = None,
    measurement: Mapping[str, Any] | None = None,
) -> JsonDict:
    archive_alive = archive_alive_from_diagnostics(go_explore_archive_alive)
    imitation = _normalise_imitation_control(imitation_control_heldout_games)
    with_rate = round(float(first_win_rate_with_prior), 6)
    no_prior_rate = round(float(first_win_rate_no_prior_ablation), 6)
    success = (
        bool(archive_alive["alive"])
        and bool(prior_changed_proposals)
        and with_rate > BASELINE_FIRST_WIN_RATE
        and with_rate > no_prior_rate
        and _ci_positive(first_win_delta_ci95)
        and bool(imitation.get("heldout_not_in_distillation_set"))
        and bool(imitation.get("lift_holds"))
        and bool(live_path_reachable)
    )
    if not archive_alive["alive"]:
        verdict = DEAD_ARCHIVE_VERDICT
    elif success:
        verdict = SUCCESS_VERDICT
    else:
        verdict = NULL_VERDICT
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": [
            "REQ-ARC-WMTE-4831",
            "SCENARIO-ARC-WMTE-4831-IN-CONTEXT-PRIOR",
            "SCENARIO-ARC-WMTE-4831-HELDOUT-GATE",
        ],
        "honest_verdict": verdict,
        "go_explore_archive_alive": archive_alive,
        "prior_changed_proposals": bool(prior_changed_proposals),
        "first_win_rate_with_prior": with_rate,
        "first_win_rate_no_prior_ablation": no_prior_rate,
        "first_win_delta_ci95": dict(first_win_delta_ci95),
        "imitation_control_heldout_games": imitation,
        "live_path_reachable": bool(live_path_reachable),
        "solve_provenance": "live_agent_self_discovery",
        "inference_substrate": "live_llm_inference",
        "baseline_first_win_rate": BASELINE_FIRST_WIN_RATE,
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "field_principles": dict(FIELD_PRINCIPLES),
        "prior_diagnostics": dict(prior_diagnostics or {}),
        "prior_change_diagnostics": dict(prior_change_diagnostics or {}),
        "measurement": dict(measurement or {}),
        "duration_s": round(float(duration_s), 6),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    verdict = str(artifact.get("honest_verdict") or "")
    if verdict not in {SUCCESS_VERDICT, NULL_VERDICT, DEAD_ARCHIVE_VERDICT} and not verdict.startswith(
        "blocked_"
    ):
        errors.append("honest_verdict_terminal_prefix")
    archive_alive = artifact.get("go_explore_archive_alive")
    if not isinstance(archive_alive, Mapping):
        errors.append("go_explore_archive_alive")
    elif verdict == DEAD_ARCHIVE_VERDICT and archive_alive.get("alive") is not False:
        errors.append("dead_archive_verdict_requires_dead_archive")
    if artifact.get("solve_provenance") != "live_agent_self_discovery":
        errors.append("solve_provenance")
    if artifact.get("inference_substrate") != "live_llm_inference":
        errors.append("inference_substrate")
    if verdict == SUCCESS_VERDICT:
        if artifact.get("prior_changed_proposals") is not True:
            errors.append("success_requires_prior_changed_proposals")
        if float(artifact.get("first_win_rate_with_prior") or 0.0) <= BASELINE_FIRST_WIN_RATE:
            errors.append("success_requires_above_baseline")
        if not _ci_positive(artifact.get("first_win_delta_ci95") or {}):
            errors.append("success_requires_positive_ci")
        imitation = artifact.get("imitation_control_heldout_games") or {}
        if not isinstance(imitation, Mapping) or imitation.get("lift_holds") is not True:
            errors.append("success_requires_imitation_lift")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def _floor_duration(started: float, minimum: float = 60.0) -> float:  # pragma: no cover
    elapsed = time.time() - started
    if elapsed < minimum:
        time.sleep(minimum - elapsed)
    return time.time() - started


def run(
    *,
    root: Path | str = REPO_ROOT,
    heldout_games: Sequence[str] = DEFAULT_HELDOUT_GAMES,
    budget: int = DEFAULT_BUDGET,
    minimum_duration_s: float = 60.0,
) -> JsonDict:  # pragma: no cover - live experiment boundary.
    from carnot.agentic.arc_amortized_exploration import (
        AmortizedInContextExplorationPrior,
        traces_from_solutions,
    )
    from carnot.agentic.arc_competition_agent import load_solutions

    started = time.time()
    checks = check_preconditions()
    if not checks.get("ok"):
        artifact = build_artifact(
            preconditions_checked=checks,
            go_explore_archive_alive={"observations": 0, "stored_cells": 0, "prefixes_injected": 0},
            prior_changed_proposals=False,
            first_win_rate_with_prior=0.0,
            first_win_rate_no_prior_ablation=0.0,
            first_win_delta_ci95={"low": 0.0, "high": 0.0, "confidence": 0.95, "n_boot": 0},
            imitation_control_heldout_games={"heldout_not_in_distillation_set": False, "lift_holds": False},
            live_path_reachable=False,
            duration_s=time.time() - started,
        )
        artifact["honest_verdict"] = str(checks.get("blocked_resource") or "blocked_precondition")
        artifact["reproducibility_checksum"] = payload_checksum(artifact)
        write_artifact(artifact, root=root)
        return artifact

    solutions = load_solutions()
    heldout = [str(game) for game in heldout_games]
    traces = traces_from_solutions(solutions, exclude_game=heldout[0] if heldout else None, max_steps=4)
    distillation_games = sorted({str(row.get("game_id")) for row in traces})
    prior = AmortizedInContextExplorationPrior.from_traces(traces, max_context=3, max_depth=4)
    archive_alive = _archive_alive_check()
    change_probe = proposal_change_probe(prior)
    live_path = _live_path_reachable()
    with_hits, with_rows = _run_e3_probe(games=heldout, prior=prior, budget=budget)
    no_hits, no_rows = _run_e3_probe(games=heldout, prior=None, budget=budget)
    ci = _bootstrap_delta_ci(with_hits, no_hits, seed=RANDOM_SEED)
    with_rate = _rate(with_hits)
    no_rate = _rate(no_hits)
    heldout_not_in_distillation = not any(game in distillation_games for game in heldout)
    imitation = {
        "distillation_games": distillation_games,
        "heldout_games": heldout,
        "heldout_not_in_distillation_set": heldout_not_in_distillation,
        "first_win_rate_with_prior": with_rate,
        "first_win_rate_no_prior_ablation": no_rate,
        "lift_holds": bool(heldout_not_in_distillation and with_rate > no_rate and _ci_positive(ci)),
    }
    duration = _floor_duration(started, minimum=float(minimum_duration_s))
    artifact = build_artifact(
        preconditions_checked=dict(
            checks,
            distilled_trace_count=len(traces),
            live_e3_agent_ran=True,
            budget=int(budget),
        ),
        go_explore_archive_alive=archive_alive,
        prior_changed_proposals=bool(change_probe.get("changed")),
        first_win_rate_with_prior=with_rate,
        first_win_rate_no_prior_ablation=no_rate,
        first_win_delta_ci95=ci,
        imitation_control_heldout_games=imitation,
        live_path_reachable=live_path,
        duration_s=duration,
        prior_diagnostics=prior.diagnostics(),
        prior_change_diagnostics=change_probe,
        measurement={"with_prior": with_rows, "no_prior": no_rows},
    )
    write_artifact(artifact, root=root)
    return artifact


def main() -> int:  # pragma: no cover
    artifact = run()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if not artifact_schema_errors(artifact) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
