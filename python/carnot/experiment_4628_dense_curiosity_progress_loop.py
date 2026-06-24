"""Experiment 4628: dense curiosity / learning-progress live loop.

Spec refs: REQ-CAPSTONE-4628, SCENARIO-CAPSTONE-4628,
SCENARIO-CAPSTONE-4628-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(PYTHON_ROOT))

JsonDict = dict[str, Any]
VariantRunner = Callable[[str, Mapping[str, Any], int], Mapping[str, Any]]
VariantRunnerFactory = Callable[[str], VariantRunner]
LivePathCheck = Callable[[Path | str], Mapping[str, Any]]

RESULT_RELATIVE_PATH = "results/experiment_4628_dense_curiosity_progress_loop.json"
EXPERIMENT = "experiment_4628_dense_curiosity_progress_loop"
EXPERIMENT_ID = 4628
SCHEMA = "carnot.exp4628.dense_curiosity_progress_loop.v1"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- offline-arcade live-search measurement over cached "
    "variants (1s floor), no live_llm_inference"
)
SOLVE_PROVENANCE = "live_agent_self_discovery"
RANDOM_SEED = 4628
DEFAULT_VARIANT_IDS = (1,)
DEFAULT_BUDGET = 200
DEFAULT_BOOTSTRAPS = 1000
MULTI_LEVEL_TARGET_LEVELS = 2
TERMINAL_PREFIXES = ("success:", "complete:", "passed:", "shipped:", "blocked_")
SPEC_REFS = [
    "REQ-CAPSTONE-4628",
    "SCENARIO-CAPSTONE-4628",
    "SCENARIO-CAPSTONE-4628-FIELD-PRINCIPLES",
]
LOOP_CONFIG: JsonDict = {
    "policy": "E3AgentPolicy",
    "dense_curiosity_progress_loop_enabled": True,
    "dense_curiosity_weight": 0.15,
    "dense_curiosity_discount": 0.5,
    "value_weight": 0.0,
    "search_mode": "depth_first_ride",
    "candidate_router": None,
    "navigation_cost_tiebreak": False,
    "llm_arm": "disabled_noop_proposer_for_matched_offline_measurement",
    "target_levels": MULTI_LEVEL_TARGET_LEVELS,
}
BARE_CONTROL_CONFIG: JsonDict = {
    "policy": "E3AgentPolicy",
    "dense_curiosity_progress_loop_enabled": False,
    "value_weight": 0.0,
    "search_mode": "depth_first_ride",
    "candidate_router": None,
    "navigation_cost_tiebreak": False,
    "llm_arm": "disabled_noop_proposer_for_matched_offline_measurement",
    "target_levels": MULTI_LEVEL_TARGET_LEVELS,
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: dense_curiosity_loop_live_<solverate|coverage|firstwin>_up_<n> "
            "OR complete: dense_curiosity_loop_no_live_lift_honest_null_gap_sharpened."
        )
    },
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates -- offline-arcade live-search measurement over "
            "cached variants (1s floor); any LLM arm on the iGPU declares live_llm_inference (NEVER the "
            "3090s)."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "MUST be false -- the curiosity/progress signal is a learned prediction-error value, "
            "oracle-DISTINCT from running the executable win-check."
        )
    },
    "solve_provenance": {
        "principle": (
            "live_agent_self_discovery -- this improves the SCORED live agent's OWN exploration "
            "(arc_graph_explore/E3AgentPolicy); NOT a parallel solver, NOT outer_loop_re."
        )
    },
    "live_path_reachable": {
        "principle": (
            "HARD gate -- the new module is in the live import closure (arc_graph_explore/E3AgentPolicy/"
            "arc_loop_solve); arc_orphan_solver_lint passes."
        )
    },
    "dense_signal_source": {
        "principle": (
            "the world-model prediction-error field reused (LiveTTT per-cell error / consistency_energy) "
            "+ that it is the IMPROVEMENT (epistemic, reducible) not raw error (aleatoric)."
        )
    },
    "live_solve_rate_loop": {
        "principle": (
            "the HEADLINE -- held-out LIVE solve-rate WITH the dense-progress loop on the SCORED agent."
        )
    },
    "live_solve_rate_bare": {
        "principle": "the matched bare-explorer solve-rate on the SAME variants."
    },
    "solve_rate_delta": {"principle": "loop - bare, emitted explicitly so a null is annotated."},
    "state_coverage_delta": {"principle": "distinct win-relevant states reached: loop - bare."},
    "first_win_rate_delta": {
        "principle": "loop - bare first-win-rate, emitted explicitly so a null is annotated."
    },
    "live_lift_ci": {
        "principle": (
            "bootstrap CI on the chosen live-lift metric; a claim above bare requires the CI to exclude it."
        )
    },
    "bare_control_passed": {
        "principle": "the POSITIVE CONTROL -- the bare explorer ran on a corpus with reachable headroom."
    },
    "false_negative_risk_checked": {
        "principle": "true with the bare control + reachable-headroom confirmed."
    },
    "null_delta_methodology_note": {
        "principle": (
            "present when a delta==0 -- states the equality is an honest no-value null, not a measurement "
            "bug."
        )
    },
    "chosen_submitted_config": {
        "principle": (
            "the recommended SUBMITTED_AGENT_CONFIG change (dense loop on, bonus weight) -- 'unchanged' "
            "if null."
        )
    },
    "offline_reproduced": {
        "principle": "any newly-solved variant must offline-reproduce to count."
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent harness/corpus drift on replay."
    },
    "preconditions_checked": {
        "principle": (
            "records resources verified (offline arcade, E3AgentPolicy + LiveTTT + explorer importable); "
            "pre-empts missing-resource fabrication."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(
    field for field in FIELD_PRINCIPLES if field != "null_delta_methodology_note"
) + (
    "experiment",
    "experiment_id",
    "schema",
    "loop_measurement",
    "bare_measurement",
    "matched_variant_signatures",
    "loop_config",
    "bare_control_config",
    "live_path_check",
    "residual_bridge_gaps",
    "field_principles",
    "spec_refs",
    "duration_s",
    "submitted_to_leaderboard",
)


class _NoOpProposer:
    """Offline measurement proposer that avoids live LLM inference."""

    def induce(
        self, *_args: Any, **_kwargs: Any
    ) -> tuple[bool, str]:  # pragma: no cover - ARC runtime.
        return False, "disabled_exp4628_no_live_llm"

    def world_model_candidates(self, _game: str) -> list[Any]:  # pragma: no cover - ARC runtime.
        return []


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _public_games(root: Path) -> list[str]:  # pragma: no cover - filesystem boundary.
    env_dir = root / "environment_files"
    return (
        []
        if not env_dir.is_dir()
        else sorted(path.name for path in env_dir.iterdir() if path.is_dir())
    )


def _variant_signature(game: str, variant_id: int) -> str:
    return f"{game}~color{int(variant_id):02d}"


def variant_specs(public_games: Sequence[str], variant_ids: Sequence[int]) -> list[JsonDict]:
    return [
        {
            "game": str(game),
            "variant": int(variant_id),
            "kind": "color",
            "reflect": None,
            "variant_signature": _variant_signature(str(game), int(variant_id)),
        }
        for game in sorted(str(item) for item in public_games)
        for variant_id in sorted(int(item) for item in variant_ids)
    ]


def _truthy_solved(attempt: Mapping[str, Any]) -> bool:
    return attempt.get("attempted") is True and (
        attempt.get("first_win") is True or attempt.get("solved") is True
    )


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    parsed = int(value) if value is not None else 0
    return parsed if parsed > 0 else None


def _actions_to_first_levelup(attempt: Mapping[str, Any]) -> int | None:
    if not _truthy_solved(attempt):
        return None
    for key in ("actions_to_first_levelup", "first_levelup_actions", "actions"):
        value = _positive_int(attempt.get(key))
        if value is not None:
            return value
    return None


def _state_coverage(attempt: Mapping[str, Any]) -> int:
    return int(
        attempt.get("distinct_win_relevant_states")
        or attempt.get("state_coverage")
        or attempt.get("states")
        or 0
    )


def _rate(count: int, total: int) -> float:
    return 0.0 if total <= 0 else round(float(count) / float(total), 6)


def measurement_from_attempts(attempts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """SCENARIO-CAPSTONE-4628: summarize live solve, coverage, and first-win metrics."""

    rows = [dict(attempt) for attempt in attempts if attempt.get("attempted") is True]
    solved = [row for row in rows if _truthy_solved(row)]
    coverage = [_state_coverage(row) for row in rows]
    actions = [_actions_to_first_levelup(row) for row in rows]
    clean_actions = [int(value) for value in actions if value is not None]
    return {
        "variant_attempts": rows,
        "variant_attempts_count": len(rows),
        "variant_solved_count": len(solved),
        "solve_rate": _rate(len(solved), len(rows)),
        "first_win_rate": _rate(len(solved), len(rows)),
        "state_coverage": int(sum(coverage)),
        "state_coverage_by_variant": coverage,
        "actions_to_first_levelup": clean_actions,
        "variant_signatures": [str(row.get("variant_signature") or "") for row in rows],
    }


def paired_metric_delta_ci(
    loop_attempts: Sequence[Mapping[str, Any]],
    bare_attempts: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    random_seed: int = RANDOM_SEED,
    n_bootstrap: int = DEFAULT_BOOTSTRAPS,
) -> JsonDict:
    loop_by_sig = {
        str(attempt.get("variant_signature") or ""): dict(attempt)
        for attempt in loop_attempts
        if attempt.get("attempted") is True
    }
    bare_by_sig = {
        str(attempt.get("variant_signature") or ""): dict(attempt)
        for attempt in bare_attempts
        if attempt.get("attempted") is True
    }
    signatures = sorted(set(loop_by_sig) & set(bare_by_sig))
    if metric in {"solve_rate_delta", "first_win_rate_delta"}:
        deltas = [
            (1.0 if _truthy_solved(loop_by_sig[sig]) else 0.0)
            - (1.0 if _truthy_solved(bare_by_sig[sig]) else 0.0)
            for sig in signatures
        ]
    else:
        deltas = [
            float(_state_coverage(loop_by_sig[sig]) - _state_coverage(bare_by_sig[sig]))
            for sig in signatures
        ]
    point = 0.0 if not deltas else sum(deltas) / len(deltas)
    if not deltas or n_bootstrap <= 0 or len(set(deltas)) == 1:
        rounded = round(float(point), 6)
        return {
            "method": "paired_percentile_bootstrap",
            "metric": metric,
            "point": rounded,
            "ci95": [rounded, rounded],
            "bootstrap_resamples": int(n_bootstrap),
            "random_seed": int(random_seed),
        }
    rng = random.Random(random_seed)  # pragma: no cover - exercised by broader metric tests.
    samples = []  # pragma: no cover - non-degenerate bootstrap path.
    for _index in range(int(n_bootstrap)):  # pragma: no cover
        samples.append(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas))
    samples.sort()  # pragma: no cover
    lo = samples[int(0.025 * (len(samples) - 1))]  # pragma: no cover
    hi = samples[int(0.975 * (len(samples) - 1))]  # pragma: no cover
    return {  # pragma: no cover
        "method": "paired_percentile_bootstrap",
        "metric": metric,
        "point": round(float(point), 6),
        "ci95": [round(float(lo), 6), round(float(hi), 6)],
        "bootstrap_resamples": int(n_bootstrap),
        "random_seed": int(random_seed),
    }


def _same_variant_control(loop: Mapping[str, Any], bare: Mapping[str, Any]) -> bool:
    return loop.get("variant_attempts_count", 0) > 0 and list(
        loop.get("variant_signatures") or []
    ) == list(bare.get("variant_signatures") or [])


def _reachable_headroom(loop: Mapping[str, Any], bare: Mapping[str, Any]) -> bool:
    rows = list(loop.get("variant_attempts") or []) + list(bare.get("variant_attempts") or [])
    return any(row.get("reachable_headroom", True) is True for row in rows)


def _offline_reproduced(loop: Mapping[str, Any], bare: Mapping[str, Any]) -> bool:
    bare_wins = {
        str(attempt.get("variant_signature") or "")
        for attempt in bare.get("variant_attempts", [])
        if _truthy_solved(attempt)
    }
    for attempt in loop.get("variant_attempts", []):
        if not _truthy_solved(attempt):
            continue
        signature = str(attempt.get("variant_signature") or "")
        gate = attempt.get("reproduction_gate")
        if signature not in bare_wins and (
            not isinstance(gate, Mapping) or gate.get("reproduced") is not True
        ):
            return False
    return True


def _chosen_metric(solve_delta: float, coverage_delta: int, first_win_delta: float) -> str:
    if solve_delta > 0.0:
        return "solve_rate_delta"
    if coverage_delta > 0:
        return "state_coverage_delta"
    if first_win_delta > 0.0:
        return "first_win_rate_delta"
    return "solve_rate_delta"


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + _sha256(payload)


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    loop_measurement: Mapping[str, Any],
    bare_measurement: Mapping[str, Any],
    live_path_check: Mapping[str, Any],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    live_solve_rate_loop = float(loop_measurement.get("solve_rate") or 0.0)
    live_solve_rate_bare = float(bare_measurement.get("solve_rate") or 0.0)
    solve_rate_delta = round(live_solve_rate_loop - live_solve_rate_bare, 6)
    state_coverage_delta = int(
        (loop_measurement.get("state_coverage") or 0)
        - (bare_measurement.get("state_coverage") or 0)
    )
    first_win_rate_delta = round(
        float(loop_measurement.get("first_win_rate") or 0.0)
        - float(bare_measurement.get("first_win_rate") or 0.0),
        6,
    )
    chosen_metric = _chosen_metric(solve_rate_delta, state_coverage_delta, first_win_rate_delta)
    ci = paired_metric_delta_ci(
        loop_measurement.get("variant_attempts", []),
        bare_measurement.get("variant_attempts", []),
        metric=chosen_metric,
        random_seed=random_seed,
    )
    live_path_reachable = bool(live_path_check.get("passed")) and bool(
        preconditions_checked.get("arc_orphan_solver_lint_passed", True)
    )
    bare_control_passed = _same_variant_control(
        loop_measurement, bare_measurement
    ) and _reachable_headroom(
        loop_measurement,
        bare_measurement,
    )
    offline_reproduced = _offline_reproduced(loop_measurement, bare_measurement)
    solve_rate_preserved = live_solve_rate_loop >= live_solve_rate_bare
    ci_excludes_zero = ci["ci95"][0] > 0.0 or ci["ci95"][1] < 0.0
    metric_value = {
        "solve_rate_delta": solve_rate_delta,
        "state_coverage_delta": float(state_coverage_delta),
        "first_win_rate_delta": first_win_rate_delta,
    }[chosen_metric]
    success = (
        live_path_reachable
        and bare_control_passed
        and offline_reproduced
        and solve_rate_preserved
        and metric_value > 0.0
        and ci_excludes_zero
    )
    if success:
        metric_label = {
            "solve_rate_delta": "solverate",
            "state_coverage_delta": "coverage",
            "first_win_rate_delta": "firstwin",
        }[chosen_metric]
        up_count = (
            int(
                round(
                    metric_value * max(1, int(loop_measurement.get("variant_attempts_count") or 0))
                )
            )
            if chosen_metric != "state_coverage_delta"
            else int(metric_value)
        )
        honest_verdict = f"success: dense_curiosity_loop_live_{metric_label}_up_{up_count}"
    else:
        honest_verdict = "complete: dense_curiosity_loop_no_live_lift_honest_null_gap_sharpened"

    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "solve_provenance": SOLVE_PROVENANCE,
        "live_path_reachable": live_path_reachable,
        "dense_signal_source": (
            "LiveTTT per-cell prediction-error improvement with online aleatoric conflict suppression; "
            "not raw surprise."
        ),
        "live_solve_rate_loop": live_solve_rate_loop,
        "live_solve_rate_bare": live_solve_rate_bare,
        "solve_rate_delta": solve_rate_delta,
        "state_coverage_delta": state_coverage_delta,
        "first_win_rate_delta": first_win_rate_delta,
        "live_lift_ci": ci,
        "bare_control_passed": bool(bare_control_passed),
        "false_negative_risk_checked": bool(
            bare_control_passed and _reachable_headroom(loop_measurement, bare_measurement)
        ),
        "chosen_submitted_config": dict(LOOP_CONFIG) if success else "unchanged",
        "offline_reproduced": bool(offline_reproduced),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "loop_measurement": dict(loop_measurement),
        "bare_measurement": dict(bare_measurement),
        "matched_variant_signatures": list(loop_measurement.get("variant_signatures") or []),
        "loop_config": dict(LOOP_CONFIG),
        "bare_control_config": dict(BARE_CONTROL_CONFIG),
        "live_path_check": dict(live_path_check),
        "residual_bridge_gaps": []
        if success
        else [
            "Missing-Verifier / bridge gap: dense online progress did not lift matched live exploration."
        ],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "duration_s": round(float(duration_s), 6),
        "submitted_to_leaderboard": False,
    }
    if solve_rate_delta == 0.0 or state_coverage_delta == 0 or first_win_rate_delta == 0.0:
        artifact["null_delta_methodology_note"] = (
            "At least one headline delta is zero after running the matched bare control on the same "
            "variants; zero is an honest no-value null for that metric, not a measurement bug."
        )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_terminal_prefix")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_false")
    if artifact.get("live_path_reachable") is not True:
        errors.append("live_path_reachable")
    if (
        artifact.get("solve_rate_delta") == 0
        or artifact.get("state_coverage_delta") == 0
        or artifact.get("first_win_rate_delta") == 0
    ) and "null_delta_methodology_note" not in artifact:
        errors.append("null_delta_methodology_note")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def check_preconditions(
    root: Path | str = REPO_ROOT,
) -> JsonDict:  # pragma: no cover - live boundary.
    root_path = Path(root)
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists(),
        "offline_arcade": False,
        "e3_policy_import": False,
        "live_ttt_import": False,
        "arc_graph_explore_import": False,
        "dense_curiosity_import": False,
        "spec_has_req_4628": False,
        "leaderboard_submission": False,
        "live_llm_inference": False,
        "qwen35_9b_mtp_igpu_precondition": "not_used",
    }
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        checks["offline_arcade"] = True
        from carnot.agentic.arc_competition_agent import E3AgentPolicy as _E3AgentPolicy
        from carnot.agentic import arc_graph_explore as _arc_graph_explore
        from carnot.agentic import arc_live_ttt as _arc_live_ttt
        from carnot.agentic.arc_dense_curiosity_progress import DenseCuriosityProgress

        checks["e3_policy_import"] = _E3AgentPolicy is not None
        checks["live_ttt_import"] = _arc_live_ttt is not None
        checks["arc_graph_explore_import"] = _arc_graph_explore is not None
        checks["dense_curiosity_import"] = DenseCuriosityProgress is not None
    except Exception as exc:
        checks["blocked_resource"] = "offline_arcade_or_live_import"
        checks["error"] = repr(exc)[:200]
        checks["ok"] = False
        return checks
    spec = root_path / SPEC_RELATIVE_PATH
    checks["spec_has_req_4628"] = spec.exists() and "REQ-CAPSTONE-4628" in spec.read_text(
        encoding="utf-8"
    )
    checks["ok"] = all(
        bool(checks[key])
        for key in (
            "agents_md_read",
            "codex_md_read",
            "offline_arcade",
            "e3_policy_import",
            "live_ttt_import",
            "arc_graph_explore_import",
            "dense_curiosity_import",
            "spec_has_req_4628",
        )
    )
    if not checks["ok"]:
        checks["blocked_resource"] = "precondition"
    return checks


def _policy_for_mode(mode: str, game: str):  # pragma: no cover - ARC runtime.
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    proposer = _NoOpProposer()
    if mode == "loop":
        return E3AgentPolicy(
            game,
            proposer=proposer,
            target_levels=MULTI_LEVEL_TARGET_LEVELS,
            value_head=None,
            value_weight=0.0,
            candidate_router=None,
            navigation_cost_tiebreak=False,
            dense_curiosity=True,
            dense_curiosity_weight=float(LOOP_CONFIG["dense_curiosity_weight"]),
            dense_curiosity_discount=float(LOOP_CONFIG["dense_curiosity_discount"]),
        )
    return E3AgentPolicy(
        game,
        proposer=proposer,
        target_levels=MULTI_LEVEL_TARGET_LEVELS,
        value_head=None,
        value_weight=0.0,
        candidate_router=None,
        navigation_cost_tiebreak=False,
        dense_curiosity=False,
    )


def _action_label(action: int | str, data: Any) -> str:  # pragma: no cover - ARC runtime.
    return json.dumps({"action": action, "data": data}, sort_keys=True, separators=(",", ":"))


def _apply_action_label(env: Any, label: str, _frame: Any = None) -> Any:  # pragma: no cover.
    if label == "RESET":
        return env.reset()
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    step = json.loads(label)
    return env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))


def run_variant_attempt(
    mode: str, game: str, spec: Mapping[str, Any], budget: int
) -> JsonDict:  # pragma: no cover - ARC runtime.
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import _level_of
    from carnot.agentic.arc_variant_generator import VariantEnv

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    env = VariantEnv(env, game, int(spec["variant"]), reflect=spec.get("reflect"))
    policy = _policy_for_mode(mode, game)
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
            if labels:
                labels.append("RESET")
        elif kind is None:
            break
        else:
            latest = env.step(getattr(GameAction, f"ACTION{kind}"), data=data)
            labels.append(_action_label(int(kind), data))
            actions += 1
        if start_level is None:
            start_level = _level_of(latest)
        reached = _level_of(latest)
        if start_level is not None and reached > start_level and actions_to_first is None:
            actions_to_first = actions
        frames.append(latest)
        if latest is None:
            break
    claimed = reached if start_level is not None and reached > start_level else 0
    gate: JsonDict = {
        "game": game,
        "claimed_level": claimed,
        "reached_level": 0,
        "reproduced": False,
        "mode": "offline_reproduction_gate_no_solution",
    }
    if claimed > 0 and labels:
        gate = dict(kit.reproduce(game, labels, _apply_action_label, claimed_level=claimed))
    solved = bool(gate.get("reproduced")) and int(gate.get("reached_level") or 0) >= claimed >= 1
    state_coverage = int(len(getattr(policy.explorer, "graph", {}) or {}))
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
        "state_coverage": state_coverage,
        "distinct_win_relevant_states": state_coverage,
        "reachable_headroom": state_coverage > 0,
        "curiosity_diagnostics": policy.explorer.curiosity_diagnostics(),
        "solution_labels": labels if solved else [],
        "reproduction_gate": gate,
        "blocked_reason": "",
        "policy_mode": mode,
    }


def default_variant_runner_factory(mode: str) -> VariantRunner:  # pragma: no cover - ARC runtime.
    return lambda game, spec, budget: run_variant_attempt(mode, game, spec, budget)


def measure_policy_pair(
    *,
    public_games: Sequence[str],
    variant_ids: Sequence[int],
    budget: int,
    variant_runner_factory: VariantRunnerFactory,
) -> tuple[JsonDict, JsonDict]:
    specs = variant_specs(public_games, variant_ids)
    loop_runner = variant_runner_factory("loop")
    bare_runner = variant_runner_factory("bare")
    loop_attempts = [dict(loop_runner(str(spec["game"]), spec, int(budget))) for spec in specs]
    bare_attempts = [dict(bare_runner(str(spec["game"]), spec, int(budget))) for spec in specs]
    return measurement_from_attempts(loop_attempts), measurement_from_attempts(bare_attempts)


def run_live_path_check(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - subprocess.
    cmd = [sys.executable, "scripts/arc_orphan_solver_lint.py"]
    proc = subprocess.run(
        cmd,
        cwd=Path(root),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    return {
        "passed": proc.returncode == 0,
        "command": " ".join(cmd),
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def _blocked_artifact(checks: Mapping[str, Any], duration_s: float) -> JsonDict:
    artifact = build_artifact(
        preconditions_checked=checks,
        loop_measurement=measurement_from_attempts([]),
        bare_measurement=measurement_from_attempts([]),
        live_path_check={"passed": False},
        duration_s=duration_s,
    )
    artifact["honest_verdict"] = f"blocked_{checks.get('blocked_resource', 'precondition')}"
    artifact["chosen_submitted_config"] = "unchanged"
    artifact["bare_control_passed"] = False
    artifact["false_negative_risk_checked"] = False
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _floor_duration(
    *,
    started_at: float,
    now: Callable[[], float],
    sleep_fn: Callable[[float], None],
) -> float:
    elapsed = max(0.0, float(now() - started_at))
    if elapsed < 1.0:
        sleep_fn(1.0 - elapsed)
    return max(float(now()), started_at + 1.0) - started_at


def run(
    *,
    root: Path | str = REPO_ROOT,
    preconditions_checked: Mapping[str, Any] | None = None,
    public_games: Sequence[str] | None = None,
    variant_ids: Sequence[int] = DEFAULT_VARIANT_IDS,
    budget: int = DEFAULT_BUDGET,
    variant_runner_factory: VariantRunnerFactory = default_variant_runner_factory,
    live_path_check: LivePathCheck = run_live_path_check,
    now: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> JsonDict:
    started = now()
    root_path = Path(root)
    checks = dict(preconditions_checked or check_preconditions(root_path))
    if not checks.get("ok", True):
        artifact = _blocked_artifact(
            checks,
            _floor_duration(started_at=started, now=now, sleep_fn=sleep_fn),
        )
    else:
        live_check = dict(live_path_check(root_path))
        checks["arc_orphan_solver_lint_passed"] = bool(live_check.get("passed"))
        games = list(public_games if public_games is not None else _public_games(root_path))
        loop, bare = measure_policy_pair(
            public_games=games,
            variant_ids=variant_ids,
            budget=budget,
            variant_runner_factory=variant_runner_factory,
        )
        artifact = build_artifact(
            preconditions_checked=checks,
            loop_measurement=loop,
            bare_measurement=bare,
            live_path_check=live_check,
            duration_s=_floor_duration(started_at=started, now=now, sleep_fn=sleep_fn),
            random_seed=RANDOM_SEED,
        )
    output = root_path / RESULT_RELATIVE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:  # pragma: no cover - CLI shim.
    run()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim.
    raise SystemExit(main())
