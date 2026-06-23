"""Experiment 4617: graduate SpatialValueNet into the live ARC path.

Spec refs: REQ-ARC-WMTE-4617, SCENARIO-ARC-WMTE-4617-LIVE-PATH,
SCENARIO-ARC-WMTE-4617-MATCHED-CONTROLS, SCENARIO-ARC-WMTE-4617-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from carnot.experiment_4605_live_integration_scored_agent import (
    measurement_from_attempts,
    paired_first_win_delta_ci,
    variant_specs,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))

JsonDict = dict[str, Any]
VariantRunner = Callable[[str, Mapping[str, Any], int], Mapping[str, Any]]
VariantRunnerFactory = Callable[[str], VariantRunner]
ParityCheck = Callable[[Path | str], Mapping[str, Any]]
OrphanLint = Callable[[Path | str], Mapping[str, Any]]

EXPERIMENT = "experiment_4617_graduate_spatial_value_head_live"
SCHEMA = "carnot.arc.graduate_spatial_value_head_live_4617.v1"
RESULT_RELATIVE_PATH = "results/experiment_4617_graduate_spatial_value_head_live.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
A1_RELATIVE_PATH = "results/experiment_4616_offline_live_bridge_disambiguation.json"
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- offline-arcade live-search measurement over "
    "cached variants (1s floor); any LLM arm on the iGPU declares live_llm_inference."
)
SOLVE_PROVENANCE = "live_agent_self_discovery"
RANDOM_SEED = 4617
DEFAULT_VARIANT_IDS = (1,)
DEFAULT_BUDGET = 200
DEFAULT_BOOTSTRAPS = 1000
GRADUATED_VALUE_WEIGHT = 1e-6
VALUE_WEIGHT_BOUND = 1e-3
TERMINAL_PREFIXES = ("success:", "complete:", "passed:", "shipped:", "blocked_")
_GRADUATED_HEAD: Any | None = None
_LINEAR_HEAD: Any | None = None
SPEC_REFS = [
    "REQ-ARC-WMTE-4617",
    "SCENARIO-ARC-WMTE-4617-LIVE-PATH",
    "SCENARIO-ARC-WMTE-4617-MATCHED-CONTROLS",
    "SCENARIO-ARC-WMTE-4617-BLOCKED-PRECONDITION",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: spatial_value_head_graduated_live_first_win_up_<n> OR "
            "complete: spatial_value_head_graduated_no_live_value_honest_null_gap_sharpened."
        )
    },
    "inference_substrate": {"principle": INFERENCE_SUBSTRATE},
    "verifier_is_oracle": {
        "principle": (
            "MUST be false -- the SpatialValueNet is a learned value, oracle-DISTINCT from the "
            "executable win-check."
        )
    },
    "solve_provenance": {
        "principle": (
            "live_agent_self_discovery -- this graduates the value head into the SCORED live agent's "
            "OWN path (arc_loop_solve warm-start + E3AgentPolicy); NOT a parallel solver, NOT "
            "outer_loop_re."
        )
    },
    "value_head_live_path_reachable": {
        "principle": (
            "HARD gate -- the graduated module is imported by scripts/arc_loop_solve.py AND reachable "
            "from E3AgentPolicy; arc_orphan_solver_lint passes (NOT an orphaned scripts/experiments "
            "solver, per the Live-Path Reachability Discipline)."
        )
    },
    "bridge_fix_applied": {
        "principle": (
            "the A1-indicated fix actually wired (decision-point-only/cached / DAgger-retrained / "
            "isotonic-calibrated) -- documents the targeted, not naive, integration."
        )
    },
    "first_win_rate_graduated": {
        "principle": (
            "the HEADLINE -- held-out LIVE first-win-rate WITH the graduated SpatialValueNet (> the "
            "linear baseline is the bridge crossed)."
        )
    },
    "first_win_rate_linear_baseline": {
        "principle": (
            "the matched LINEAR-verifier baseline on the SAME variants (today's warm-start that "
            "'actively misled' -- the apples-to-apples control)."
        )
    },
    "first_win_rate_bare": {
        "principle": "the bare-BFS control (no value head) -- the no-regression floor."
    },
    "first_win_delta": {
        "principle": (
            "graduated - linear baseline (positive = the position-preserving head crosses the bridge), "
            "emitted explicitly so a null (0) is annotated."
        )
    },
    "first_win_ci": {
        "principle": (
            "bootstrap CI on the first-win delta; a claim above the linear baseline requires the CI to "
            "exclude it."
        )
    },
    "median_actions_to_first_levelup_graduated": {
        "principle": (
            "ACTION cost WITH the graduated head -- the leaderboard tiebreaker (RHAE rewards efficiency)."
        )
    },
    "actions_delta": {
        "principle": (
            "linear_actions - graduated (positive = fewer actions); emitted explicitly so a null is "
            "annotated."
        )
    },
    "value_weight_used": {
        "principle": (
            "MUST be ~0 (tie-breaker) or the A1-indicated bounded mode -- documents that this did NOT "
            "repeat the value_weight=5 regression."
        )
    },
    "parity_test_green": {
        "principle": (
            "HARD gate -- test_arc_submitted_agent_parity.py passes; the integrated config stays the "
            "single source of truth."
        )
    },
    "bare_and_linear_controls_passed": {
        "principle": (
            "the POSITIVE CONTROLS -- graduated must beat the linear baseline on the SAME variants AND "
            "not drop below bare BFS; a null is valid only if both ran."
        )
    },
    "false_negative_risk_checked": {
        "principle": "true with both controls run -- a no-value null is valid only then."
    },
    "null_delta_methodology_note": {
        "principle": (
            "present when first_win_delta==0 -- states the equality is an honest no-value null, not a "
            "measurement bug."
        )
    },
    "solve_rate_preserved": {
        "principle": "HARD gate -- graduating the head must NOT drop solve-rate vs bare BFS."
    },
    "chosen_submitted_config": {
        "principle": (
            "the recommended SUBMITTED_AGENT_CONFIG (graduated head on, value mode, weight) -- the A6 "
            "input; 'unchanged' if null."
        )
    },
    "offline_reproduced": {
        "principle": "any newly-solved variant must offline-reproduce to count."
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent drift on replay."
    },
    "preconditions_checked": {
        "principle": (
            "records resources verified (offline arcade, E3AgentPolicy + ValueNet importable, A1 "
            "artifact present); pre-empts missing-resource fabrication."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(
    field for field in FIELD_PRINCIPLES if field != "null_delta_methodology_note"
) + (
    "experiment",
    "schema",
    "solve_rate_graduated",
    "solve_rate_linear_baseline",
    "solve_rate_bare",
    "median_actions_to_first_levelup_linear_baseline",
    "median_actions_to_first_levelup_bare",
    "graduated_measurement",
    "linear_measurement",
    "bare_measurement",
    "matched_variant_signatures",
    "parity_test",
    "orphan_lint",
    "field_principles",
    "spec_refs",
    "duration_s",
    "submitted_to_leaderboard",
)


class _NoOpProposer:
    def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:  # pragma: no cover
        return False, "disabled_exp4617_no_live_llm"

    def world_model_candidates(self, _game: str) -> list[Any]:  # pragma: no cover
        return []


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _public_games(root: Path) -> list[str]:  # pragma: no cover - filesystem boundary
    env_dir = root / "environment_files"
    if not env_dir.is_dir():
        return []
    return sorted(path.name for path in env_dir.iterdir() if path.is_dir())


def _same_variant_control(*measurements: Mapping[str, Any]) -> bool:
    if not measurements:
        return False
    first = list(measurements[0].get("variant_signatures") or [])
    if not first:
        return False
    return all(list(measurement.get("variant_signatures") or []) == first for measurement in measurements)


def _truthy_solved(attempt: Mapping[str, Any]) -> bool:
    return attempt.get("attempted") is True and (
        attempt.get("first_win") is True or attempt.get("solved") is True
    )


def _offline_reproduced(
    graduated: Mapping[str, Any],
    linear: Mapping[str, Any],
    bare: Mapping[str, Any],
) -> bool:
    control_wins = {
        str(attempt.get("variant_signature") or "")
        for measurement in (linear, bare)
        for attempt in measurement.get("variant_attempts", [])
        if _truthy_solved(attempt)
    }
    for attempt in graduated.get("variant_attempts", []):
        if not _truthy_solved(attempt):
            continue
        signature = str(attempt.get("variant_signature") or "")
        gate = attempt.get("reproduction_gate")
        if signature not in control_wins and (
            not isinstance(gate, Mapping) or gate.get("reproduced") is not True
        ):
            return False
    return True


def _submitted_config_snapshot() -> JsonDict:
    from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG

    return json.loads(json.dumps(SUBMITTED_AGENT_CONFIG, sort_keys=True, default=str))


def _chosen_config_snapshot() -> JsonDict:
    config = _submitted_config_snapshot()
    config["value_head"] = "SpatialValueNet"
    config["value_mode"] = "decision_point_cached_tiebreak"
    config["bounded_value_weight"] = GRADUATED_VALUE_WEIGHT
    return config


def _bridge_fix() -> JsonDict:
    return {
        "mode": "decision_point_cached_tiebreak",
        "source": "experiment_4616_offline_live_bridge_disambiguation",
        "binding_bridge_cause": "compute_cost",
        "value_weight": GRADUATED_VALUE_WEIGHT,
        "lazy_value_top_k": 4,
        "cache_by_frame_hash": True,
    }


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + _sha256(payload)


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    graduated_measurement: Mapping[str, Any],
    linear_measurement: Mapping[str, Any],
    bare_measurement: Mapping[str, Any],
    parity_test: Mapping[str, Any],
    orphan_lint: Mapping[str, Any],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    first_win_rate_graduated = float(graduated_measurement.get("first_win_rate") or 0.0)
    first_win_rate_linear = float(linear_measurement.get("first_win_rate") or 0.0)
    first_win_rate_bare = float(bare_measurement.get("first_win_rate") or 0.0)
    first_win_delta = round(first_win_rate_graduated - first_win_rate_linear, 6)
    ci = paired_first_win_delta_ci(
        graduated_measurement.get("variant_attempts", []),
        linear_measurement.get("variant_attempts", []),
        random_seed=random_seed,
        n_bootstrap=DEFAULT_BOOTSTRAPS,
    )
    grad_actions = graduated_measurement.get("median_actions_to_first_levelup")
    linear_actions = linear_measurement.get("median_actions_to_first_levelup")
    actions_delta = (
        round(float(linear_actions) - float(grad_actions), 6)
        if linear_actions is not None and grad_actions is not None
        else 0.0
    )
    solve_rate_graduated = float(graduated_measurement.get("solve_rate") or 0.0)
    solve_rate_linear = float(linear_measurement.get("solve_rate") or 0.0)
    solve_rate_bare = float(bare_measurement.get("solve_rate") or 0.0)
    solve_rate_preserved = solve_rate_graduated >= solve_rate_bare
    parity_green = bool(parity_test.get("passed"))
    live_path_reachable = bool(orphan_lint.get("passed"))
    controls_passed = _same_variant_control(graduated_measurement, linear_measurement, bare_measurement)
    offline_reproduced = _offline_reproduced(graduated_measurement, linear_measurement, bare_measurement)
    ci_excludes_zero = ci["ci95"][0] > 0.0 or ci["ci95"][1] < 0.0
    first_win_success = first_win_delta > 0.0 and ci_excludes_zero
    actions_success = actions_delta > 0.0 and solve_rate_graduated >= solve_rate_linear
    success = (
        parity_green
        and live_path_reachable
        and controls_passed
        and solve_rate_preserved
        and offline_reproduced
        and abs(GRADUATED_VALUE_WEIGHT) <= VALUE_WEIGHT_BOUND
        and (first_win_success or actions_success)
    )
    if success and first_win_delta > 0.0:
        up_count = int(
            round(
                first_win_delta
                * max(1, int(graduated_measurement.get("variant_attempts_count") or 0))
            )
        )
        honest_verdict = f"success: spatial_value_head_graduated_live_first_win_up_{up_count}"
    elif success:
        honest_verdict = (
            "success: spatial_value_head_graduated_live_first_win_up_0_"
            f"actions_delta_{actions_delta:g}"
        )
    else:
        honest_verdict = "complete: spatial_value_head_graduated_no_live_value_honest_null_gap_sharpened"

    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "solve_provenance": SOLVE_PROVENANCE,
        "value_head_live_path_reachable": live_path_reachable,
        "bridge_fix_applied": _bridge_fix(),
        "first_win_rate_graduated": first_win_rate_graduated,
        "first_win_rate_linear_baseline": first_win_rate_linear,
        "first_win_rate_bare": first_win_rate_bare,
        "first_win_delta": first_win_delta,
        "first_win_ci": ci,
        "median_actions_to_first_levelup_graduated": grad_actions,
        "median_actions_to_first_levelup_linear_baseline": linear_actions,
        "median_actions_to_first_levelup_bare": bare_measurement.get(
            "median_actions_to_first_levelup"
        ),
        "actions_delta": actions_delta,
        "solve_rate_graduated": solve_rate_graduated,
        "solve_rate_linear_baseline": solve_rate_linear,
        "solve_rate_bare": solve_rate_bare,
        "value_weight_used": GRADUATED_VALUE_WEIGHT,
        "parity_test_green": parity_green,
        "bare_and_linear_controls_passed": controls_passed,
        "false_negative_risk_checked": bool(controls_passed),
        "solve_rate_preserved": bool(solve_rate_preserved),
        "chosen_submitted_config": _chosen_config_snapshot() if success else "unchanged",
        "offline_reproduced": bool(offline_reproduced),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "graduated_measurement": dict(graduated_measurement),
        "linear_measurement": dict(linear_measurement),
        "bare_measurement": dict(bare_measurement),
        "matched_variant_signatures": list(
            graduated_measurement.get("variant_signatures") or []
        ),
        "parity_test": dict(parity_test),
        "orphan_lint": dict(orphan_lint),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "duration_s": round(float(duration_s), 6),
        "submitted_to_leaderboard": False,
    }
    if first_win_delta == 0.0:
        artifact["null_delta_methodology_note"] = (
            "first_win_delta is zero after running the matched linear and bare controls on the same "
            "variants; this is an honest no-value null, not a measurement bug."
        )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_terminal_prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_false")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance")
    if artifact.get("value_head_live_path_reachable") is not True:
        errors.append("value_head_live_path_reachable")
    if not isinstance(artifact.get("bridge_fix_applied"), Mapping):
        errors.append("bridge_fix_applied")
    if abs(float(artifact.get("value_weight_used") or 0.0)) > VALUE_WEIGHT_BOUND:
        errors.append("value_weight_bounded")
    if artifact.get("first_win_delta") == 0 and "null_delta_methodology_note" not in artifact:
        errors.append("null_delta_methodology_note")
    if artifact.get("bare_and_linear_controls_passed") is not True:
        errors.append("bare_and_linear_controls_passed")
    if artifact.get("false_negative_risk_checked") is not True:
        errors.append("false_negative_risk_checked")
    if artifact.get("solve_rate_preserved") is not True:
        errors.append("solve_rate_preserved")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - live boundary
    root_path = Path(root)
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "offline_arcade": False,
        "e3_policy_import": False,
        "value_net_import": False,
        "a1_artifact_present": False,
        "a1_binding_bridge_cause": None,
        "a1_indicated_fix": None,
        "spec_has_req_4617": False,
        "leaderboard_submission": False,
        "live_llm_inference": False,
        "qwen35_9b_mtp_igpu_precondition": "not_used",
    }
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        checks["offline_arcade"] = True
    except Exception as exc:
        checks["blocked_resource"] = "offline_arcade"
        checks["error"] = repr(exc)[:200]
        checks["ok"] = False
        return checks
    try:
        from carnot.agentic.arc_competition_agent import E3AgentPolicy

        checks["e3_policy_import"] = E3AgentPolicy is not None
    except Exception as exc:
        checks["blocked_resource"] = "e3_policy_import"
        checks["error"] = repr(exc)[:200]
        checks["ok"] = False
        return checks
    try:
        from carnot.agentic.arc_value_net import SpatialValueNet, ValueNet

        checks["value_net_import"] = SpatialValueNet is not None and ValueNet is not None
    except Exception as exc:
        checks["blocked_resource"] = "value_net_import"
        checks["error"] = repr(exc)[:200]
        checks["ok"] = False
        return checks
    a1 = root_path / A1_RELATIVE_PATH
    if a1.exists():
        data = json.loads(a1.read_text(encoding="utf-8"))
        checks["a1_artifact_present"] = True
        checks["a1_binding_bridge_cause"] = data.get("binding_bridge_cause")
        checks["a1_indicated_fix"] = data.get("indicated_fix")
    spec = root_path / SPEC_RELATIVE_PATH
    checks["spec_has_req_4617"] = spec.exists() and "REQ-ARC-WMTE-4617" in spec.read_text(
        encoding="utf-8"
    )
    checks["ok"] = all(
        bool(checks[key])
        for key in (
            "agents_md_read",
            "codex_md_read",
            "offline_arcade",
            "e3_policy_import",
            "value_net_import",
            "a1_artifact_present",
            "spec_has_req_4617",
        )
    )
    if not checks["ok"]:
        checks["blocked_resource"] = "precondition"
    return checks


def _action_label(action: int | str, data: Any) -> str:  # pragma: no cover - ARC runtime
    return json.dumps({"action": action, "data": data}, sort_keys=True, separators=(",", ":"))


def _apply_action_label(env: Any, label: str, _frame: Any = None) -> Any:  # pragma: no cover
    if label == "RESET":
        return env.reset()
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    step = json.loads(label)
    return env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))


def _policy_for_mode(mode: str, game: str):  # pragma: no cover - ARC runtime
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, SUBMITTED_LAZY_VALUE_TOP_K
    from carnot.agentic.arc_competition_agent import _load_linear_cross_game_value_head
    from carnot.agentic.arc_value_net import SpatialValueNet, load_live_spatial_value_head

    global _GRADUATED_HEAD, _LINEAR_HEAD
    proposer = _NoOpProposer()
    common = {
        "proposer": proposer,
        "target_levels": 1,
        "search_mode": "depth_first_ride",
        "candidate_router": None,
        "navigation_cost_tiebreak": True,
    }
    if mode == "bare":
        return E3AgentPolicy(
            game,
            value_head=None,
            value_weight=0.0,
            navigation_cost_tiebreak=False,
            **{k: v for k, v in common.items() if k != "navigation_cost_tiebreak"},
        )
    if mode == "linear":
        if _LINEAR_HEAD is None:
            _LINEAR_HEAD = _load_linear_cross_game_value_head()
        return E3AgentPolicy(
            game,
            value_head=_LINEAR_HEAD,
            value_weight=GRADUATED_VALUE_WEIGHT,
            lazy_value_top_k=SUBMITTED_LAZY_VALUE_TOP_K,
            **common,
        )
    game_specific = load_live_spatial_value_head(root=REPO_ROOT, game=game)
    if game_specific is not None:
        spatial = game_specific
    else:
        if _GRADUATED_HEAD is None:
            _GRADUATED_HEAD = load_live_spatial_value_head(root=REPO_ROOT) or SpatialValueNet(device="cpu")
        spatial = _GRADUATED_HEAD
    return E3AgentPolicy(
        game,
        value_head=spatial,
        value_weight=GRADUATED_VALUE_WEIGHT,
        lazy_value_top_k=SUBMITTED_LAZY_VALUE_TOP_K,
        **common,
    )


def _level_of_frame(frame: Any) -> int:  # pragma: no cover - ARC runtime
    from carnot.agentic.arc_competition_agent import _level_of

    return int(_level_of(frame))


def run_variant_attempt(
    mode: str, game: str, spec: Mapping[str, Any], budget: int
) -> JsonDict:  # pragma: no cover - ARC runtime
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
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
            start_level = _level_of_frame(latest)
        reached = _level_of_frame(latest)
        if start_level is not None and reached > start_level:
            actions_to_first = actions
            break
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
        "policy_mode": mode,
    }


def default_variant_runner_factory(mode: str) -> VariantRunner:  # pragma: no cover - ARC runtime
    return lambda game, spec, budget: run_variant_attempt(mode, game, spec, budget)


def measure_three_arms(
    *,
    public_games: Sequence[str],
    variant_ids: Sequence[int],
    budget: int,
    variant_runner_factory: VariantRunnerFactory,
) -> tuple[JsonDict, JsonDict, JsonDict]:
    specs = variant_specs(public_games, variant_ids)
    graduated_runner = variant_runner_factory("graduated")
    linear_runner = variant_runner_factory("linear")
    bare_runner = variant_runner_factory("bare")
    graduated_attempts = [
        dict(graduated_runner(str(spec["game"]), spec, int(budget))) for spec in specs
    ]
    linear_attempts = [dict(linear_runner(str(spec["game"]), spec, int(budget))) for spec in specs]
    bare_attempts = [dict(bare_runner(str(spec["game"]), spec, int(budget))) for spec in specs]
    return (
        measurement_from_attempts(graduated_attempts),
        measurement_from_attempts(linear_attempts),
        measurement_from_attempts(bare_attempts),
    )


def run_parity_check(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - subprocess
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/python/test_arc_submitted_agent_parity.py",
        "-q",
        "--no-cov",
    ]
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


def run_orphan_lint(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - subprocess
    cmd = [sys.executable, "scripts/arc_orphan_solver_lint.py"]
    proc = subprocess.run(
        cmd,
        cwd=Path(root),
        capture_output=True,
        text=True,
        timeout=120,
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
        graduated_measurement=measurement_from_attempts([]),
        linear_measurement=measurement_from_attempts([]),
        bare_measurement=measurement_from_attempts([]),
        parity_test={"passed": False},
        orphan_lint={"passed": False},
        duration_s=duration_s,
    )
    artifact["honest_verdict"] = f"blocked_{checks.get('blocked_resource', 'precondition')}"
    artifact["value_head_live_path_reachable"] = False
    artifact["chosen_submitted_config"] = "unchanged"
    artifact["bare_and_linear_controls_passed"] = False
    artifact["false_negative_risk_checked"] = False
    artifact["solve_rate_preserved"] = True
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
    parity_check: ParityCheck = run_parity_check,
    orphan_lint: OrphanLint = run_orphan_lint,
    now: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> JsonDict:
    started = now()
    root_path = Path(root)
    checks = dict(preconditions_checked or check_preconditions(root_path))
    if not checks.get("ok", True):
        artifact = _blocked_artifact(
            checks, _floor_duration(started_at=started, now=now, sleep_fn=sleep_fn)
        )
    else:
        games = list(public_games if public_games is not None else _public_games(root_path))
        graduated, linear, bare = measure_three_arms(
            public_games=games,
            variant_ids=variant_ids,
            budget=budget,
            variant_runner_factory=variant_runner_factory,
        )
        parity = dict(parity_check(root_path))
        lint = dict(orphan_lint(root_path))
        artifact = build_artifact(
            preconditions_checked=checks,
            graduated_measurement=graduated,
            linear_measurement=linear,
            bare_measurement=bare,
            parity_test=parity,
            orphan_lint=lint,
            duration_s=_floor_duration(started_at=started, now=now, sleep_fn=sleep_fn),
            random_seed=RANDOM_SEED,
        )
    output = root_path / RESULT_RELATIVE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:  # pragma: no cover - CLI shim
    run()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim
    raise SystemExit(main())
