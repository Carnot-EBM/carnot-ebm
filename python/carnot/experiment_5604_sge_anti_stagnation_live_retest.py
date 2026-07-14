"""Exp5604: SGE anti-stagnation controller -- genuine live re-test (task 6 completion).

Spec refs: REQ-ARC-FCP-5604, SCENARIO-ARC-FCP-5604-LIVE-COLLAPSE-ESCAPE.

Context (2026-07-10/11 outer-loop history this module completes): `outer_loop_sge_smoke_test.py`
ran the real `SGECandidateRouter` (genuine local GGUF LLM strategy proposals, no anti-stagnation
guard) against g50t L3 and observed a real failure mode -- by the final steps the model's own
strategies converged on a repetitive "wait for the system to process the pending interaction"
pattern that never escalated even after multiple reflection cycles. `ops/known-issues.md` task 6
flagged the fix: "the reflection prompt needs an explicit anti-stagnation nudge... detect repeated
null-outcome strategies and force strategy diversity."

The conductor built that fix 2026-07-11 (`AntiStagnationDiversityController` +
`StrategyCollapseThresholds` + `FORCED_ANTI_STAGNATION_PORTFOLIO` in
`arc_llm_strategy_proposer.py`, now wired as `SGECandidateRouter`'s DEFAULT
`anti_stagnation_controller`) and ran exp5575, a DETERMINISTIC precheck: it replayed the recorded
g50t collapse trace through the new controller (never invoking an LLM) and confirmed the collapse
detector correctly fires on that trace and the forced portfolio selects a genuinely diverse
candidate set. That is real evidence the CONTROLLER LOGIC is correct -- but it is not yet evidence
that a genuine LIVE run (fresh strategies from the model, not a replayed trace) actually escapes a
collapse it would otherwise fall into. exp5575's own follow-on gate (exp5576) never ran the live
attempt at all -- it GATE_BLOCKed on `live_path_ready=False`, and that block was caused entirely by
UNRELATED pre-existing project-wide gates (16 failures in the full `pytest tests/python` suite --
a Z3 segfault plus missing onnx/onnxruntime -- and a 1262-test spec-coverage backlog), not by
anything about the SGE mechanism itself (its own scoped test command, `pytest
tests/python/test_arc_llm_strategy_proposer.py`, passed 35/35 at 100% coverage). This module is
the genuine live re-test that closes that gap.

Target selection: g50t (the original null target) is no longer a valid target for THIS test -- it
was independently fully cleared (`levels_reproduced=7, full_game_clear=true`) on 2026-07-12 by a
completely different mechanism (hand-derived multi-session live discovery, `outer_loop_fable5_*`
probes), one day after exp5575's precheck. There is no more g50t frontier to test against. Per the
task's own "(or the original null target)" clause, this module retargets to the current shallowest
not-fully-cleared game in the registry, `sk48` (`levels_reproduced=7, full_game_clear=None`),
confirmed via a fresh registry re-read at run time (not hardcoded) so the target can never go stale
across future registry updates.

This is deliberately NOT primarily a level-bank attempt (that is `arc_loop_solve.py`'s standing
job, exp5610 this milestone). The primary, falsifiable question this module answers is: does the
now-default `AntiStagnationDiversityController`, when the live LLM router genuinely collapses into
a repeated-strategy / repeated-action / null-outcome pattern during a real run, actually detect it
and switch to the forced diverse portfolio -- a live, non-replayed confirmation of exp5575's
positive control? `collapse_detected_live` and `forced_portfolio_activated_live` are the headline
fields; `offline_reproduced`/`reproduced_levels` are secondary (a level bank is a bonus, not the
gate).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

JsonDict = dict[str, Any]

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5604
EXPERIMENT = "experiment_5604_sge_anti_stagnation_live_retest"
RESULT_RELATIVE_PATH = "results/experiment_5604_sge_anti_stagnation_live_retest.json"
TRAJECTORY_RELATIVE_PATH = "results/experiment_5604_sge_anti_stagnation_live_retest_trajectory.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-human-replay-frame-change/spec.md"
SPEC_REFS = ["REQ-ARC-FCP-5604", "SCENARIO-ARC-FCP-5604-LIVE-COLLAPSE-ESCAPE"]
SCHEMA = "carnot.experiment_5604_sge_anti_stagnation_live_retest.v1"
SOLVE_PROVENANCE = "live_agent_self_discovery"
RANDOM_SEED = 5604
DEFAULT_BUDGET = 46  # matches outer_loop_sge_smoke_test.py / exp5534 for apples-to-apples comparison
DEFAULT_PORT = 8929  # fresh port; 8919/8921 already host unrelated HIP servers (verified live)
MODEL_SPECS = ["unsloth/gemma-4-12B-it-GGUF"]

FIELD_PRINCIPLES: dict[str, str] = {
    "target_game": "current shallowest not-fully-cleared registry game, re-read live so the target can never go stale.",
    "target_level": "the next unreproduced level for target_game at run time.",
    "prior_levels_reproduced": "registry depth for target_game before this attempt.",
    "collapse_detected_live": "bare bool: the anti-stagnation controller actually fired during THIS live run (not a replayed trace).",
    "forced_portfolio_activated_live": "bare bool: the deterministic diverse portfolio was actually installed as the ranked candidates at least once live.",
    "collapse_trigger_step": "the step index collapse first fired, or null if it never fired (a genuine no-collapse run is also an honest, valid result).",
    "post_collapse_strategy_diversity": "unique forced-portfolio category count selected in the step(s) after collapse fired; the direct behavioral escape signal.",
    "llm_strategy_proposer_used_any_step": "bare bool proving real GGUF inference occurred at least once (the exp5534 dishonest-baseline failure mode this guards against).",
    "attempts": "bare int count of live actions executed.",
    "max_level_reached": "highest level counter observed during this single fresh env session.",
    "offline_reproduced": "true only when a claimed new level passes the standard offline replay gate.",
    "reproduced_levels": "integer new levels banked; secondary to the collapse-escape headline, not required for a valid result.",
    "registry_delta": "nonzero only when offline_reproduced is true.",
    "trajectory_path": "path to the full step-by-step diagnostics log for audit.",
    "model_specs": "the local GGUF proposer actually invoked.",
    "inference_substrate": "live_llm_inference when llm_strategy_proposer_used_any_step is true, else the honest no-LLM substrate.",
    "solve_provenance": "must equal live_agent_self_discovery -- the scored E3AgentPolicy path, not a hand-derived outer-loop solve.",
    "preconditions_checked": "resources verified before any live inference was attempted.",
    "duration_s": "real wall-clock time; a genuine GPU LLM run over a 46-action budget takes well over 60s.",
    "honest_verdict": "one-line verdict starting complete: or blocked:.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def _read_yaml(path: Path) -> JsonDict:
    if not path.exists():
        return {"reproducible_total_levels": 0, "games": []}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {"reproducible_total_levels": 0, "games": []}


def _registry_rows(registry: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [row for row in registry.get("games", []) or [] if isinstance(row, Mapping) and row.get("game")]


def select_target(registry: Mapping[str, Any]) -> JsonDict:
    """REQ-ARC-FCP-5604: pick the current shallowest not-fully-cleared game, live."""

    rows = _registry_rows(registry)
    candidates = [row for row in rows if row.get("full_game_clear") is not True]
    if not candidates:
        return {"blocked": True, "blocker": "no_unsolved_target_all_games_full_clear"}
    candidates.sort(key=lambda row: int(row.get("levels_reproduced") or 0))
    chosen = candidates[0]
    prior = int(chosen.get("levels_reproduced") or 0)
    return {
        "blocked": False,
        "target_game": str(chosen["game"]),
        "target_level": prior + 1,
        "prior_levels_reproduced": prior,
    }


def preconditions(*, root: Path = REPO) -> JsonDict:
    """PRECONDITIONS (Pre-Launch Preconditions Discipline). Checked before any live inference."""

    checks: JsonDict = {}
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        checks["offline_arcade_makes_env"] = True
    except Exception:
        checks["offline_arcade_makes_env"] = False
    try:
        from carnot.agentic.arc_llm_strategy_proposer import (  # noqa: F401
            AntiStagnationDiversityController,
            LLMStrategyProposer,
            SGECandidateRouter,
        )
        from carnot.agentic.arc_competition_agent import E3AgentPolicy  # noqa: F401
        from carnot.agentic.arc_executable_world_model import LocalGGUFProposer  # noqa: F401

        checks["sge_and_e3_import"] = True
    except Exception:
        checks["sge_and_e3_import"] = False
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    checks["gguf_cached"] = any(cache.glob("models--*gemma-4-12B-it-GGUF*")) if cache.exists() else False
    checks["ok"] = all(checks.values())
    return checks


def _first_precondition_miss(checked: Mapping[str, Any]) -> str:
    for key, value in checked.items():
        if key != "ok" and not value:
            return key
    return "unknown_precondition"


class _NoOpInductionProposer:  # pragma: no cover - ARC runtime boundary
    def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:
        return False, "disabled_exp5604_no_induction"

    def world_model_candidates(self, _game: str) -> list[Any]:
        return []


def _live_run(  # pragma: no cover - ARC runtime boundary, real GPU + real env
    *, game: str, target_level: int, prior_levels: int, budget: int, port: int
) -> JsonDict:
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, _level_of
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer
    from carnot.agentic.arc_llm_strategy_proposer import LLMStrategyProposer, SGECandidateRouter
    from carnot.experiment_5521_arc_live_action_diverse_levelup import (
        ActionDiverseLiveGenerator,
        _action_label,
        _apply_action_label,
    )

    os.environ.setdefault("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")
    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"

    gguf = LocalGGUFProposer(port=port)
    proposer = LLMStrategyProposer(completer=gguf, max_tokens=64)
    router = SGECandidateRouter(
        proposer=proposer,
        game_id=game,
        k=3,
        temperatures=(0.3, 0.6, 0.9),
        max_candidates=8,
        reflect_every=6,
    )  # anti_stagnation_controller defaults to a fresh AntiStagnationDiversityController
    generator = ActionDiverseLiveGenerator(max_candidates=8)

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    policy = E3AgentPolicy(
        game,
        proposer=_NoOpInductionProposer(),
        explore_budget=budget,
        target_levels=target_level,
        value_head=None,
        frame_change_scorer=None,
        candidate_router=router,
        action_effect_expansion_prior=False,
        action_prior=generator,
        qd_generator=generator,
        goal_bias=None,
        goal_candidate_guidance=False,
        active_probe_controller=False,
        go_explore_archive=False,
    )

    start = time.time()
    frames: list[Any] = []
    latest = None
    max_level = prior_levels
    labels: list[str] = []
    diagnostics_log: list[JsonDict] = []
    collapse_trigger_step: int | None = None
    post_collapse_categories: set[str] = set()

    for step in range(1, budget + 1):
        if policy.is_done(frames, latest):
            break
        before_level = int(_level_of(latest)) if latest is not None else max_level
        kind, data = policy.next_move(frames, latest)
        diag = dict(router.last_diagnostics)
        anti = dict(diag.get("anti_stagnation") or {})
        forced_selected = anti.get("forced_portfolio_selected")
        collapsed_this_step = bool(anti.get("collapse_detected")) and isinstance(forced_selected, list)
        if collapsed_this_step and collapse_trigger_step is None:
            collapse_trigger_step = step
        if collapsed_this_step:
            for row in forced_selected:
                if isinstance(row, Mapping) and row.get("name"):
                    post_collapse_categories.add(str(row["name"]))
        diagnostics_log.append(
            {
                "step": step,
                "llm_strategy_proposer_used": diag.get("llm_strategy_proposer_used"),
                "strategy_texts": diag.get("strategy_texts"),
                "collapse_detected": anti.get("collapse_detected"),
                "triggered_signals": anti.get("triggered_signals"),
                "forced_portfolio_selected": forced_selected,
            }
        )
        if kind == "RESET":
            latest = env.reset()
        elif kind is None:
            break
        else:
            latest = env.step(getattr(GameAction, f"ACTION{int(kind)}"), data=data)
            label = _action_label(int(kind), data)
            labels.append(label)
            after_level = int(_level_of(latest))
            max_level = max(max_level, after_level)
            router.record_outcome("level_advanced" if after_level != before_level else "no_change")
        frames.append(latest)
        if latest is None or max_level >= target_level:
            break

    duration_s = time.time() - start
    any_llm_used = any(row.get("llm_strategy_proposer_used") for row in diagnostics_log)

    reproduction_gate: JsonDict = {"reproduced": False, "reached_level": prior_levels}
    if max_level > prior_levels and labels:
        reproduction_gate = dict(
            kit.reproduce(game, labels, _apply_action_label, claimed_level=max_level)
        )

    return {
        "attempts": len(labels),
        "max_level_reached": max_level,
        "duration_s": duration_s,
        "llm_strategy_proposer_used_any_step": any_llm_used,
        "model_specs": [gguf.repo_substr],
        "collapse_detected_live": collapse_trigger_step is not None,
        "collapse_trigger_step": collapse_trigger_step,
        "post_collapse_strategy_diversity": len(post_collapse_categories),
        "diagnostics_log": diagnostics_log,
        "reproduction_gate": reproduction_gate,
        "solution_labels": labels,
    }


def build_artifact(
    *,
    root: Path = REPO,
    budget: int = DEFAULT_BUDGET,
    port: int = DEFAULT_PORT,
) -> JsonDict:
    started = time.monotonic()
    checked = preconditions(root=root)
    registry = _read_yaml(root / REGISTRY_RELATIVE_PATH)
    target = select_target(registry)

    empty_result = {
        "target_game": "",
        "target_level": 0,
        "prior_levels_reproduced": 0,
        "collapse_detected_live": False,
        "forced_portfolio_activated_live": False,
        "collapse_trigger_step": None,
        "post_collapse_strategy_diversity": 0,
        "llm_strategy_proposer_used_any_step": False,
        "attempts": 0,
        "max_level_reached": 0,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "registry_delta": 0,
        "trajectory_path": "",
        "model_specs": list(MODEL_SPECS),
        "inference_substrate": "deterministic_live_path_precheck_no_llm",
    }

    if not checked.get("ok") or target.get("blocked"):
        reason = target.get("blocker") if target.get("blocked") else _first_precondition_miss(checked)
        artifact = {
            "experiment_id": EXPERIMENT_ID,
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "spec_refs": list(SPEC_REFS),
            "field_principles": dict(FIELD_PRINCIPLES),
            "random_seed": RANDOM_SEED,
            "solve_provenance": SOLVE_PROVENANCE,
            "preconditions_checked": checked,
            "duration_s": time.monotonic() - started,
            "honest_verdict": f"blocked: {reason}",
            **empty_result,
        }
        (root / RESULT_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
        (root / RESULT_RELATIVE_PATH).write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        return artifact

    run = _live_run(
        game=target["target_game"],
        target_level=target["target_level"],
        prior_levels=target["prior_levels_reproduced"],
        budget=budget,
        port=port,
    )

    gate = run["reproduction_gate"]
    reached = int(gate.get("reached_level") or 0)
    prior = target["prior_levels_reproduced"]
    reproduced = bool(gate.get("reproduced")) and reached > prior
    reproduced_levels = max(0, reached - prior) if reproduced else 0

    registry_updated = False
    if reproduced and reproduced_levels >= 1:
        rows = list(registry.get("games") or [])
        for row in rows:
            if isinstance(row, dict) and row.get("game") == target["target_game"]:
                row["levels_reproduced"] = reached
                row["latest_exp5604_sge_anti_stagnation_live_retest"] = {
                    "artifact": RESULT_RELATIVE_PATH,
                    "reproduced_levels": reproduced_levels,
                    "reached_level": reached,
                    "solve_provenance": SOLVE_PROVENANCE,
                }
                break
        registry["games"] = rows
        registry["reproducible_total_levels"] = int(registry.get("reproducible_total_levels") or 0) + reproduced_levels
        (root / REGISTRY_RELATIVE_PATH).write_text(yaml.safe_dump(registry, sort_keys=False))
        registry_updated = True

    trajectory_path = TRAJECTORY_RELATIVE_PATH
    (root / trajectory_path).write_text(
        json.dumps(
            {
                "schema": SCHEMA + ".trajectory",
                "target": target,
                "diagnostics_log": run["diagnostics_log"],
                "solution_labels": run["solution_labels"],
                "reproduction_gate": gate,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )

    collapse_live = bool(run["collapse_detected_live"])
    diversity = int(run["post_collapse_strategy_diversity"])
    forced_activated = collapse_live and diversity > 0
    llm_used = bool(run["llm_strategy_proposer_used_any_step"])

    if collapse_live and forced_activated:
        collapse_summary = (
            f"collapse_fired_live_at_step_{run['collapse_trigger_step']}_"
            f"forced_portfolio_activated_{diversity}_categories"
        )
    elif collapse_live:
        collapse_summary = f"collapse_fired_live_at_step_{run['collapse_trigger_step']}_but_no_forced_categories_recorded"
    else:
        collapse_summary = "no_collapse_observed_this_run_budget_or_target_too_short_to_trigger"

    bank_summary = (
        f"_and_banked_{reproduced_levels}_new_level(s)_on_{target['target_game']}"
        if reproduced and reproduced_levels >= 1
        else "_no_new_level_banked"
    )

    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "solve_provenance": SOLVE_PROVENANCE,
        "target_game": target["target_game"],
        "target_level": target["target_level"],
        "prior_levels_reproduced": target["prior_levels_reproduced"],
        "collapse_detected_live": collapse_live,
        "forced_portfolio_activated_live": forced_activated,
        "collapse_trigger_step": run["collapse_trigger_step"],
        "post_collapse_strategy_diversity": diversity,
        "llm_strategy_proposer_used_any_step": llm_used,
        "attempts": int(run["attempts"]),
        "max_level_reached": int(run["max_level_reached"]),
        "offline_reproduced": bool(reproduced),
        "reproduced_levels": int(reproduced_levels),
        "registry_delta": int(reproduced_levels if reproduced else 0),
        "registry_updated": bool(registry_updated),
        "trajectory_path": trajectory_path,
        "model_specs": list(run["model_specs"]),
        "inference_substrate": "live_llm_inference" if llm_used else "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "preconditions_checked": checked,
        "duration_s": time.monotonic() - started,
        "honest_verdict": f"complete: {collapse_summary}{bank_summary}",
    }
    (root / RESULT_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / RESULT_RELATIVE_PATH).write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    artifact = build_artifact(budget=args.budget, port=args.port)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
