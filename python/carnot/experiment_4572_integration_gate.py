"""Experiment 4572: .422 submitted two-metric integration gate.

Spec refs: REQ-ARC-WMTE-4572, SCENARIO-ARC-WMTE-4572.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(REPO_ROOT))


JsonDict = dict[str, Any]
ActionRunner = Callable[..., Mapping[str, Any]]
TransferRunner = Callable[..., Mapping[str, Any]]

RESULT_RELATIVE_PATH = "results/experiment_4572_integration_gate.json"
EXPERIMENT_ID = "experiment_4572_integration_gate"
SCHEMA = "carnot.arc_integration_gate_4572.v1"
RANDOM_SEED = 4572
GENERIC_TRANSFER_BASELINE = 0.04
DEFAULT_ACTION_BUDGET = 8000
DEFAULT_TRANSFER_BUDGET = 200
DEFAULT_VARIANT_IDS = (1,)
CORE_GAMES = ("lp85", "m0r0", "sp80", "vc33")
GATE_GAMES = ("lp85", "m0r0", "sp80", "vc33", "cd82", "ft09", "su15", "ls20")
REQUIREMENTS = ("REQ-ARC-WMTE-4572",)
SCENARIOS = ("SCENARIO-ARC-WMTE-4572",)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- offline arcade end-to-end "
    "(if the integrated run invokes the LLM proposer, declare live_llm_inference + add the "
    "Qwen3.5-9B-MTP model precondition); a CNN predictor forward pass is NOT "
    "live_llm_inference."
)
UPSTREAM_ARTIFACTS = {
    "a1_clickability_predictor": "results/experiment_4568_clickability_action_effect_predictor.json",
    "a2_verifier_guided_expansion": "results/experiment_4569_verifier_guided_expansion.json",
    "a4_hidden_field_state_probe": "results/experiment_4571_hidden_field_state_probe_ka59.json",
}

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; success: integrated_actions_to_levelup_below_blind_or_generic_transfer_above_0.04 "
        "OR complete: no_lever_raises_a_metric_honest_null."
    ),
    "inference_substrate": INFERENCE_SUBSTRATE,
    "median_actions_to_first_levelup_integrated": (
        "the SUBMITTED-config action efficiency after wiring winners vs the blind baseline -- "
        "the leaderboard scoring lever."
    ),
    "generic_transfer_rate_integrated": (
        "the SUBMITTED-config held-out variant transfer after wiring winners vs baseline 0.04 -- "
        "the leaderboard-honest HEADLINE."
    ),
    "levers_integrated": (
        "names which of A1/A2/A4 were wired -- traceable to their measured deltas; [] is an honest null."
    ),
    "additivity_checked": (
        "integrated metric vs the naive sum of isolated A1/A2 deltas -- surfaces a destructive "
        "interaction instead of burying it."
    ),
    "core_solves_preserved": (
        "integration must preserve every CORE solve (set-containment); a dropped solve FAILS the lever."
    ),
    "heldout_solve_rate": "the real transfer signal; integration should not regress it.",
    "ready_for_operator_submit": (
        "True if the integrated config is a CORE-preserved improvement on either metric worth a "
        "1/day submission slot; the task NEVER submits (operator-only)."
    ),
    "false_negative_risk_checked": (
        "an honest null is only valid with both baselines measured the same way."
    ),
    "random_seed": "determinism precondition for reproducibility.",
    "reproducibility_checksum": "catches silent drift on replay.",
    "preconditions_checked": "records resources verified; pre-empts missing-resource fabrication.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "field_principles",
    "requirements",
    "scenarios",
    "result_path",
    "median_actions_to_first_levelup_baseline",
    "generic_transfer_baseline",
    "baseline_action_measurement",
    "integrated_action_measurement",
    "generic_transfer_measurement",
    "upstream_decision",
    "per_game_deepest_level_reached",
    "submitted_agent_config",
    "operator_submission_performed",
    "action_budget",
    "generic_transfer_budget",
    "duration_s",
)


def _stable_checksum(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return _stable_checksum(payload)


def _read_json(path: Path) -> JsonDict:  # pragma: no cover - filesystem boundary
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_upstream_artifacts(root: Path | str = REPO_ROOT) -> dict[str, JsonDict]:  # pragma: no cover
    root_path = Path(root)
    return {name: _read_json(root_path / relative) for name, relative in UPSTREAM_ARTIFACTS.items()}


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - live boundary
    root_path = Path(root)
    spec_path = root_path / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade_import": False,
        "a1_artifact_present": (root_path / UPSTREAM_ARTIFACTS["a1_clickability_predictor"]).exists(),
        "a2_artifact_present": (
            root_path / UPSTREAM_ARTIFACTS["a2_verifier_guided_expansion"]
        ).exists(),
        "a4_artifact_present": (root_path / UPSTREAM_ARTIFACTS["a4_hidden_field_state_probe"]).exists(),
        "spec_has_req_4572": "REQ-ARC-WMTE-4572" in spec_text,
        "leaderboard_submission": False,
    }
    try:
        from carnot.agentic import arc_solver_kit

        arc_solver_kit.offline_arcade()
        checks["offline_arcade_import"] = True
    except Exception as exc:
        checks["offline_arcade_error"] = repr(exc)
    checks["ok"] = bool(checks["offline_arcade_import"])
    return checks


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_metric(value: Any) -> float:
    return round(float(value or 0.0), 4)


def _ci_positive(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and _float_or_none(value[0]) is not None
        and _float_or_none(value[1]) is not None
        and float(value[0]) > 0.0
    )


def _levels_from_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {str(game): int(level or 0) for game, level in value.items()}


def _core_levels_or_zero(levels: Mapping[str, int]) -> dict[str, int]:
    return {game: int(levels.get(game, 0)) for game in CORE_GAMES}


def _levels_from_nested(
    artifact: Mapping[str, Any],
    *,
    outer_keys: Sequence[str],
    direct_keys: Sequence[str],
) -> dict[str, int]:
    for key in direct_keys:
        value = artifact.get(key)
        if isinstance(value, Mapping):
            levels = _levels_from_mapping(value.get("deepest_level_by_game"))
            if levels:
                return levels
    deepest = artifact.get("deepest_level_reached_per_core_game")
    if isinstance(deepest, Mapping):
        for key in outer_keys:
            levels = _levels_from_mapping(deepest.get(key))
            if levels:
                return levels
    return {game: 0 for game in CORE_GAMES}


def _core_regressions(control: Mapping[str, int], treatment: Mapping[str, int]) -> list[str]:
    return [game for game in CORE_GAMES if int(treatment.get(game, 0)) < int(control.get(game, 0))]


def _corrigendum_kinds(artifact: Mapping[str, Any]) -> list[str]:
    rows = artifact.get("corrigendum_pending")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    return [
        str(row["kind"])
        for row in rows
        if isinstance(row, Mapping) and row.get("kind") is not None
    ]


def _permitted_flagged_null(artifact: Mapping[str, Any]) -> bool:
    if artifact.get("flagged_adversarial") is not True:
        return False
    note = str(artifact.get("null_delta_methodology_note") or "")
    explicit_deltas = (
        _float_or_none(artifact.get("actions_delta")),
        _float_or_none(artifact.get("transfer_delta")),
        _float_or_none(artifact.get("generic_transfer_delta")),
    )
    return bool(_corrigendum_kinds(artifact) == ["TAUTOLOGY"] and 0.0 in explicit_deltas and note)


def _flag_status(artifact: Mapping[str, Any]) -> str:
    if artifact.get("flagged_adversarial") is not True:
        return "clean"
    if _permitted_flagged_null(artifact):
        return "permitted_flagged_null"
    return "rejected_flagged_adversarial"


def _reject(
    rejected: dict[str, dict[str, Any]],
    key: str,
    summary: Mapping[str, Any],
    reason: str,
) -> None:
    rejected[key] = {**dict(summary), "reason": reason}


def _a1_control_levels(artifact: Mapping[str, Any]) -> dict[str, int]:
    return _core_levels_or_zero(
        _levels_from_nested(
            artifact,
            outer_keys=("baseline", "control", "blind"),
            direct_keys=("baseline_measurement", "control_measurement"),
        )
    )


def _a1_treatment_levels(artifact: Mapping[str, Any]) -> dict[str, int]:
    return _core_levels_or_zero(
        _levels_from_nested(
            artifact,
            outer_keys=("with_predictor", "predictor", "treatment"),
            direct_keys=("predictor_measurement", "best_measurement"),
        )
    )


def _a2_control_levels(artifact: Mapping[str, Any]) -> dict[str, int]:
    return _core_levels_or_zero(
        _levels_from_nested(
            artifact,
            outer_keys=("baseline", "control", "without_expansion"),
            direct_keys=("baseline_measurement", "control_measurement"),
        )
    )


def _a2_treatment_levels(artifact: Mapping[str, Any]) -> dict[str, int]:
    return _core_levels_or_zero(
        _levels_from_nested(
            artifact,
            outer_keys=("with_expansion", "verifier_expansion", "treatment"),
            direct_keys=("verifier_expansion_measurement", "best_measurement"),
        )
    )


def _a4_control_levels(artifact: Mapping[str, Any]) -> dict[str, int]:
    return _core_levels_or_zero(
        _levels_from_nested(
            artifact,
            outer_keys=("baseline", "control"),
            direct_keys=("baseline_measurement", "control_measurement"),
        )
    )


def _a4_treatment_levels(artifact: Mapping[str, Any]) -> dict[str, int]:
    return _core_levels_or_zero(
        _levels_from_nested(
            artifact,
            outer_keys=("hidden_field_state", "new_bank", "treatment"),
            direct_keys=("hidden_field_measurement", "best_measurement"),
        )
    )


def _summary_for_a1(artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "honest_verdict": artifact.get("honest_verdict"),
        "flag_status": _flag_status(artifact),
        "median_actions_to_first_levelup_baseline": _float_or_none(
            artifact.get("median_actions_to_first_levelup_baseline")
        ),
        "median_actions_to_first_levelup_with_predictor": _float_or_none(
            artifact.get("median_actions_to_first_levelup_with_predictor")
        ),
        "actions_delta": _round_metric(artifact.get("actions_delta")),
        "actions_delta_ci": artifact.get("actions_delta_ci"),
        "positive_control_passed": artifact.get("positive_control_passed"),
        "solve_rate_preserved": artifact.get("solve_rate_preserved"),
    }


def _summary_for_a2(artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "honest_verdict": artifact.get("honest_verdict"),
        "generic_transfer_rate_baseline": _float_or_none(
            artifact.get("generic_transfer_rate_baseline")
        ),
        "generic_transfer_rate_with_expansion": _float_or_none(
            artifact.get("generic_transfer_rate_with_expansion")
        ),
        "transfer_delta": _round_metric(artifact.get("transfer_delta")),
        "transfer_ci": artifact.get("transfer_ci"),
        "random_priority_control_passed": artifact.get("random_priority_control_passed"),
        "solve_rate_preserved": artifact.get("solve_rate_preserved"),
    }


def _summary_for_a4(artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "honest_verdict": artifact.get("honest_verdict"),
        "offline_reproduced": artifact.get("offline_reproduced"),
        "reproduced_levels": int(artifact.get("reproduced_levels") or 0),
        "registry_updated": artifact.get("registry_updated"),
        "state_disambiguation_control_passed": artifact.get(
            "state_disambiguation_control_passed"
        ),
    }


def select_integrated_levers(
    a1_artifact: Mapping[str, Any],
    a2_artifact: Mapping[str, Any],
    a4_artifact: Mapping[str, Any],
) -> JsonDict:
    """REQ-ARC-WMTE-4572: select only A1/A2/A4 levers that raised a real metric."""

    accepted: list[str] = []
    rejected: dict[str, JsonDict] = {}
    a1_summary = _summary_for_a1(a1_artifact)
    a2_summary = _summary_for_a2(a2_artifact)
    a4_summary = _summary_for_a4(a4_artifact)

    a1_key = "A1_clickability_predictor:action_efficiency"
    a1_regressions = _core_regressions(_a1_control_levels(a1_artifact), _a1_treatment_levels(a1_artifact))
    if a1_summary["flag_status"] == "rejected_flagged_adversarial":
        _reject(rejected, a1_key, a1_summary, "flagged_adversarial")
    elif a1_regressions:
        _reject(rejected, a1_key, {**a1_summary, "lost_core_level_games": a1_regressions}, "core_level_regression")
    elif float(a1_summary["actions_delta"]) <= 0.0:
        _reject(rejected, a1_key, a1_summary, "no_action_efficiency_gain")
    elif not _ci_positive(a1_artifact.get("actions_delta_ci")):
        _reject(rejected, a1_key, a1_summary, "actions_delta_ci_missing")
    elif a1_artifact.get("positive_control_passed") is not True:
        _reject(rejected, a1_key, a1_summary, "positive_control_failed")
    elif a1_artifact.get("solve_rate_preserved") is not True:
        _reject(rejected, a1_key, a1_summary, "solve_rate_not_preserved")
    else:
        accepted.append(a1_key)

    a2_key = "A2_verifier_guided_expansion:generic_transfer"
    a2_regressions = _core_regressions(_a2_control_levels(a2_artifact), _a2_treatment_levels(a2_artifact))
    a2_rate = _float_or_none(a2_artifact.get("generic_transfer_rate_with_expansion"))
    if _flag_status(a2_artifact) == "rejected_flagged_adversarial":
        _reject(rejected, a2_key, a2_summary, "flagged_adversarial")
    elif a2_regressions:
        _reject(rejected, a2_key, {**a2_summary, "lost_core_level_games": a2_regressions}, "core_level_regression")
    elif a2_artifact.get("random_priority_control_passed") is not True:
        _reject(rejected, a2_key, a2_summary, "random_priority_control_failed")
    elif a2_rate is None or a2_rate <= GENERIC_TRANSFER_BASELINE:
        _reject(rejected, a2_key, a2_summary, "no_generic_transfer_gain")
    elif not _ci_positive(a2_artifact.get("transfer_ci")):
        _reject(rejected, a2_key, a2_summary, "transfer_ci_missing")
    elif a2_artifact.get("solve_rate_preserved") is not True:
        _reject(rejected, a2_key, a2_summary, "solve_rate_not_preserved")
    elif a2_artifact.get("offline_reproduced") is not True:
        _reject(rejected, a2_key, a2_summary, "offline_reproduction_missing")
    else:
        accepted.append(a2_key)

    a4_key = "A4_hidden_field_state_probe:new_bank"
    a4_regressions = _core_regressions(_a4_control_levels(a4_artifact), _a4_treatment_levels(a4_artifact))
    a4_banked = bool(
        a4_artifact.get("offline_reproduced") is True
        and int(a4_artifact.get("reproduced_levels") or 0) > 0
    )
    if _flag_status(a4_artifact) == "rejected_flagged_adversarial":
        _reject(rejected, a4_key, a4_summary, "flagged_adversarial")
    elif a4_regressions:
        _reject(rejected, a4_key, {**a4_summary, "lost_core_level_games": a4_regressions}, "core_level_regression")
    elif not a4_banked:
        _reject(rejected, a4_key, a4_summary, "no_new_offline_bank")
    elif a4_artifact.get("core_solves_preserved") is False:
        _reject(rejected, a4_key, a4_summary, "core_solves_not_preserved")
    elif a4_artifact.get("registry_updated") is not True:
        _reject(rejected, a4_key, a4_summary, "registry_update_missing")
    else:
        accepted.append(a4_key)

    isolated_deltas = {
        "A1_clickability_predictor": {
            "actions_delta": 0.0
            if a1_summary["flag_status"] == "rejected_flagged_adversarial"
            else float(a1_summary["actions_delta"]),
            "generic_transfer_delta": 0.0,
        },
        "A2_verifier_guided_expansion": {
            "actions_delta": 0.0,
            "generic_transfer_delta": float(a2_summary["transfer_delta"]),
        },
        "A4_hidden_field_state_probe": {
            "actions_delta": 0.0,
            "generic_transfer_delta": 0.0,
        },
    }
    return {
        "accepted_levers": accepted,
        "rejected_levers": rejected,
        "upstream_summaries": {
            "A1_clickability_predictor": a1_summary,
            "A2_verifier_guided_expansion": a2_summary,
            "A4_hidden_field_state_probe": a4_summary,
        },
        "isolated_deltas": isolated_deltas,
        "naive_actions_delta": _round_metric(
            sum(row["actions_delta"] for row in isolated_deltas.values())
        ),
        "naive_generic_transfer_delta": _round_metric(
            sum(row["generic_transfer_delta"] for row in isolated_deltas.values())
        ),
    }


def _median(values: Sequence[float]) -> float | None:
    return None if not values else float(statistics.median(values))


def run_action_efficiency_measurement(
    *,
    root: Path | str = REPO_ROOT,
    policy: str,
    games: Sequence[str] = GATE_GAMES,
    budget: int = DEFAULT_ACTION_BUDGET,
    disable_induction: bool = True,
) -> JsonDict:  # pragma: no cover - slow ARC runtime boundary
    from scripts import arc_leaderboard_eval as live_eval

    old_disable = os.environ.get("CARNOT_ARC_DISABLE_INDUCTION")
    if disable_induction:
        os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    else:
        os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
    try:
        with ThreadPoolExecutor(max_workers=min(8, len(games) or 1)) as pool:
            rows = list(
                pool.map(
                    lambda game: live_eval.run_game(
                        str(game),
                        live_eval._build_policy(str(policy), str(game)),
                        budget=int(budget),
                    ),
                    games,
                )
            )
    finally:
        if old_disable is None:
            os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        else:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disable

    firsts = [
        float(row["actions_to_first_levelup"])
        for row in rows
        if row.get("actions_to_first_levelup") is not None
    ]
    solved_games = [str(row["game"]) for row in rows if int(row.get("levels") or 0) >= 1]
    return {
        "policy": str(policy),
        "games": [str(game) for game in games],
        "budget": int(budget),
        "disable_induction": bool(disable_induction),
        "measurement_source": "scripts.arc_leaderboard_eval.run_game",
        "per_game": rows,
        "solved_games": solved_games,
        "actions_to_first_levelup_by_game": {
            str(row["game"]): int(row["actions_to_first_levelup"])
            for row in rows
            if row.get("actions_to_first_levelup") is not None
        },
        "median_actions_to_first_levelup": _median(firsts),
        "root": str(Path(root)),
    }


def run_generic_transfer_measurement(
    *,
    root: Path | str = REPO_ROOT,
    budget: int = DEFAULT_TRANSFER_BUDGET,
) -> JsonDict:  # pragma: no cover - slow ARC runtime boundary
    from carnot import experiment_4550_honest_sprint_metric as exp4550

    root_path = Path(root)
    preconditions = exp4550.check_preconditions(root_path)
    return exp4550.measure_generic_transfer_over_variants(
        public_games=preconditions.get("offline_env_public_games") or [],
        variant_ids=DEFAULT_VARIANT_IDS,
        budget=int(budget),
        variant_runner=exp4550.default_variant_runner,
        random_seed=RANDOM_SEED,
    )


def _submitted_agent_config() -> JsonDict:
    from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG

    return dict(SUBMITTED_AGENT_CONFIG)


def _metric_action_median(measurement: Mapping[str, Any]) -> float | None:
    return _float_or_none(measurement.get("median_actions_to_first_levelup"))


def _solved_games(measurement: Mapping[str, Any]) -> set[str]:
    solved = measurement.get("solved_games")
    if isinstance(solved, Sequence) and not isinstance(solved, (str, bytes)):
        return {str(game) for game in solved}
    return {
        str(row["game"])
        for row in measurement.get("per_game", []) or []
        if isinstance(row, Mapping)
        and (row.get("solved") is True or int(row.get("levels") or 0) >= 1)
    }


def _per_game_deepest_level(measurement: Mapping[str, Any]) -> dict[str, int]:
    out = {game: 0 for game in GATE_GAMES}
    for row in measurement.get("per_game", []) or []:
        if not isinstance(row, Mapping) or row.get("game") is None:
            continue
        out[str(row["game"])] = int(
            row.get("deepest_level_reached", row.get("reached", row.get("levels") or 0)) or 0
        )
    return out


def _transfer_rate(measurement: Mapping[str, Any]) -> float:
    rate = _float_or_none(measurement.get("generic_transfer_rate_over_variants"))
    if rate is not None:
        return round(rate, 10)
    attempted = int(measurement.get("variant_attempts_count") or 0)
    solved = int(measurement.get("variant_solved_count") or 0)
    return 0.0 if attempted <= 0 else round(float(solved) / float(attempted), 10)


def _false_negative_risk_checked(
    *,
    baseline_action_measurement: Mapping[str, Any],
    integrated_action_measurement: Mapping[str, Any],
    transfer_measurement: Mapping[str, Any],
) -> bool:
    return bool(
        _metric_action_median(baseline_action_measurement) is not None
        and _metric_action_median(integrated_action_measurement) is not None
        and baseline_action_measurement.get("measurement_source")
        == integrated_action_measurement.get("measurement_source")
        and int(transfer_measurement.get("variant_attempts_count") or 0) > 0
    )


def _additivity_checked(
    *,
    upstream_decision: Mapping[str, Any],
    baseline_actions: float,
    integrated_actions: float,
    integrated_generic_transfer: float,
) -> JsonDict:
    integrated_actions_delta = _round_metric(baseline_actions - integrated_actions)
    integrated_transfer_delta = _round_metric(
        integrated_generic_transfer - GENERIC_TRANSFER_BASELINE
    )
    naive_actions_delta = _round_metric(upstream_decision.get("naive_actions_delta"))
    naive_generic_delta = _round_metric(upstream_decision.get("naive_generic_transfer_delta"))
    return {
        "metrics": ["median_actions_to_first_levelup", "generic_transfer_rate_over_variants"],
        "isolated_deltas": dict(upstream_decision.get("isolated_deltas") or {}),
        "naive_actions_delta": naive_actions_delta,
        "integrated_actions_delta": integrated_actions_delta,
        "actions_interaction_delta": _round_metric(integrated_actions_delta - naive_actions_delta),
        "naive_generic_transfer_delta": naive_generic_delta,
        "integrated_generic_transfer_delta": integrated_transfer_delta,
        "generic_interaction_delta": _round_metric(integrated_transfer_delta - naive_generic_delta),
    }


def _honest_verdict(
    *,
    actions_improved: bool,
    transfer_improved: bool,
    core_solves_preserved: bool,
    levers_integrated: Sequence[str],
) -> str:
    if (actions_improved or transfer_improved) and core_solves_preserved and bool(levers_integrated):
        return "success: integrated_actions_to_levelup_below_blind_or_generic_transfer_above_0.04"
    return "complete: no_lever_raises_a_metric_honest_null"


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    upstream_decision: Mapping[str, Any],
    baseline_action_measurement: Mapping[str, Any],
    integrated_action_measurement: Mapping[str, Any],
    transfer_measurement: Mapping[str, Any],
    random_seed: int,
    duration_s: float | None,
) -> JsonDict:
    """SCENARIO-ARC-WMTE-4572: assemble the terminal two-metric integration artifact."""

    baseline_actions = _metric_action_median(baseline_action_measurement)
    integrated_actions = _metric_action_median(integrated_action_measurement)
    if baseline_actions is None or integrated_actions is None:
        baseline_actions = float("inf")
        integrated_actions = float("inf")
    generic_transfer = _transfer_rate(transfer_measurement)
    levers_integrated = list(upstream_decision.get("accepted_levers") or [])
    core_solves_preserved = set(CORE_GAMES).issubset(_solved_games(integrated_action_measurement))
    actions_improved = integrated_actions < baseline_actions
    transfer_improved = generic_transfer > GENERIC_TRANSFER_BASELINE
    ready = bool(
        levers_integrated and core_solves_preserved and (actions_improved or transfer_improved)
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "honest_verdict": _honest_verdict(
            actions_improved=actions_improved,
            transfer_improved=transfer_improved,
            core_solves_preserved=core_solves_preserved,
            levers_integrated=levers_integrated,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "scenarios": list(SCENARIOS),
        "preconditions_checked": dict(preconditions_checked),
        "median_actions_to_first_levelup_baseline": baseline_actions,
        "median_actions_to_first_levelup_integrated": integrated_actions,
        "generic_transfer_baseline": GENERIC_TRANSFER_BASELINE,
        "generic_transfer_rate_integrated": generic_transfer,
        "levers_integrated": levers_integrated,
        "additivity_checked": _additivity_checked(
            upstream_decision=upstream_decision,
            baseline_actions=baseline_actions,
            integrated_actions=integrated_actions,
            integrated_generic_transfer=generic_transfer,
        ),
        "core_solves_preserved": bool(core_solves_preserved),
        "heldout_solve_rate": generic_transfer,
        "ready_for_operator_submit": ready,
        "false_negative_risk_checked": _false_negative_risk_checked(
            baseline_action_measurement=baseline_action_measurement,
            integrated_action_measurement=integrated_action_measurement,
            transfer_measurement=transfer_measurement,
        ),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "baseline_action_measurement": dict(baseline_action_measurement),
        "integrated_action_measurement": dict(integrated_action_measurement),
        "generic_transfer_measurement": dict(transfer_measurement),
        "upstream_decision": dict(upstream_decision),
        "per_game_deepest_level_reached": _per_game_deepest_level(integrated_action_measurement),
        "submitted_agent_config": _submitted_agent_config(),
        "operator_submission_performed": False,
        "action_budget": DEFAULT_ACTION_BUDGET,
        "generic_transfer_budget": DEFAULT_TRANSFER_BUDGET,
        "result_path": RESULT_RELATIVE_PATH,
        "duration_s": None if duration_s is None else float(duration_s),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _blocked_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    random_seed: int,
    duration_s: float | None,
) -> JsonDict:
    missing = [
        key for key in ("offline_arcade_import",) if preconditions_checked.get(key) is not True
    ]
    reason = "_".join(missing) if missing else "unknown_resource"
    empty_action = {
        "measurement_source": "blocked",
        "per_game": [],
        "solved_games": list(CORE_GAMES),
        "median_actions_to_first_levelup": 0.0,
    }
    artifact = build_artifact(
        preconditions_checked=preconditions_checked,
        upstream_decision={
            "accepted_levers": [],
            "rejected_levers": {},
            "upstream_summaries": {},
            "isolated_deltas": {},
            "naive_actions_delta": 0.0,
            "naive_generic_transfer_delta": 0.0,
        },
        baseline_action_measurement=empty_action,
        integrated_action_measurement=empty_action,
        transfer_measurement={
            "variant_specs": [],
            "variant_attempts": [],
            "variant_attempts_count": 1,
            "variant_solved_count": 0,
            "generic_transfer_rate_over_variants": GENERIC_TRANSFER_BASELINE,
        },
        random_seed=random_seed,
        duration_s=duration_s,
    )
    artifact["honest_verdict"] = f"blocked_{reason}"
    artifact["false_negative_risk_checked"] = False
    artifact["ready_for_operator_submit"] = False
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    blocked = isinstance(verdict, str) and verdict.startswith("blocked_")
    success = isinstance(verdict, str) and verdict.startswith("success:")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must match")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match REQ-ARC-WMTE-4572")
    if not isinstance(artifact.get("median_actions_to_first_levelup_integrated"), (int, float)):
        errors.append("median_actions_to_first_levelup_integrated must be numeric")
    transfer = artifact.get("generic_transfer_rate_integrated")
    if not isinstance(transfer, (int, float)) or not 0.0 <= float(transfer) <= 1.0:
        errors.append("generic_transfer_rate_integrated must be numeric in [0,1]")
    if not isinstance(artifact.get("levers_integrated"), list):
        errors.append("levers_integrated must be a list")
    if not isinstance(artifact.get("additivity_checked"), Mapping):
        errors.append("additivity_checked must be a mapping")
    if not isinstance(artifact.get("core_solves_preserved"), bool):
        errors.append("core_solves_preserved must be bool")
    if not isinstance(artifact.get("heldout_solve_rate"), (int, float)):
        errors.append("heldout_solve_rate must be numeric")
    if artifact.get("ready_for_operator_submit") is True and not success:
        errors.append("ready_for_operator_submit cannot be true without success")
    if success:
        actions_up = float(artifact.get("median_actions_to_first_levelup_integrated") or 0.0) < float(
            artifact.get("median_actions_to_first_levelup_baseline") or 0.0
        )
        transfer_up = float(artifact.get("generic_transfer_rate_integrated") or 0.0) > GENERIC_TRANSFER_BASELINE
        if not actions_up and not transfer_up:
            errors.append("success requires action or generic-transfer lift")
        if artifact.get("core_solves_preserved") is not True:
            errors.append("success requires core_solves_preserved=true")
        if not artifact.get("levers_integrated"):
            errors.append("success requires an integrated lever")
    if not blocked and artifact.get("false_negative_risk_checked") is not True:
        errors.append("false_negative_risk_checked must be true for complete/success artifacts")
    if artifact.get("operator_submission_performed") is not False:
        errors.append("operator_submission_performed must be false")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be a mapping")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum must be sha256-prefixed")
    elif checksum != payload_checksum(artifact):
        errors.append("reproducibility_checksum must match artifact content")
    return errors


def write_artifact(artifact: Mapping[str, Any], root: Path | str = REPO_ROOT) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(
    *,
    root: Path | str = REPO_ROOT,
    write: bool = True,
    preconditions_checked: Mapping[str, Any] | None = None,
    load_upstream_artifacts: Callable[[Path], dict[str, JsonDict]] = load_upstream_artifacts,
    action_runner: ActionRunner = run_action_efficiency_measurement,
    transfer_runner: TransferRunner = run_generic_transfer_measurement,
    random_seed: int = RANDOM_SEED,
    now: Callable[[], float] = time.monotonic,
) -> JsonDict:
    """REQ-ARC-WMTE-4572: run both .422 integration metrics and write the artifact."""

    root_path = Path(root)
    started = float(now())
    duration = lambda: max(0.0, float(now()) - started)
    preconditions = (
        dict(preconditions_checked)
        if preconditions_checked is not None
        else check_preconditions(root_path)
    )
    if preconditions.get("offline_arcade_import") is not True:
        artifact = _blocked_artifact(
            preconditions_checked=preconditions,
            random_seed=random_seed,
            duration_s=duration(),
        )
    else:
        upstream = load_upstream_artifacts(root_path)
        decision = select_integrated_levers(
            upstream.get("a1_clickability_predictor", {}),
            upstream.get("a2_verifier_guided_expansion", {}),
            upstream.get("a4_hidden_field_state_probe", {}),
        )
        baseline_action = action_runner(
            root=root_path,
            policy="explorer",
            games=GATE_GAMES,
            budget=DEFAULT_ACTION_BUDGET,
            disable_induction=True,
        )
        integrated_action = action_runner(
            root=root_path,
            policy="e3",
            games=GATE_GAMES,
            budget=DEFAULT_ACTION_BUDGET,
            disable_induction=True,
        )
        transfer_measurement = transfer_runner(root=root_path, budget=DEFAULT_TRANSFER_BUDGET)
        artifact = build_artifact(
            preconditions_checked=preconditions,
            upstream_decision=decision,
            baseline_action_measurement=baseline_action,
            integrated_action_measurement=integrated_action,
            transfer_measurement=transfer_measurement,
            random_seed=random_seed,
            duration_s=duration(),
        )
        errors = artifact_schema_errors(artifact)
        if errors:
            raise ValueError("; ".join(errors))
    if write:
        write_artifact(artifact, root=root_path)
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper
    artifact = run()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
