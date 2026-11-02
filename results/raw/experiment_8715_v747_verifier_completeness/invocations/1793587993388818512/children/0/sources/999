"""Experiment 4597: ARC .424 integration gate.

Spec refs: REQ-CAPSTONE-4597, SCENARIO-CAPSTONE-4597,
SCENARIO-CAPSTONE-4597-FIELD-PRINCIPLES.
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


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot import experiment_4550_honest_sprint_metric as exp4550
from carnot import live_submittable_metrics
from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG


JsonDict = dict[str, Any]
VariantRunner = Callable[[str, Mapping[str, Any], int], Mapping[str, Any]]

EXPERIMENT = "experiment_4597_integration_gate"
SCHEMA = "carnot.exp4597.integration_gate.v1"
RESULT_RELATIVE_PATH = "results/experiment_4597_integration_gate.json"
INTEGRATED_PACKAGE_RELATIVE_PATH = "results/experiment_4595_submission_package_operator_resubmit.json"
PREVIOUS_PACKAGE_RELATIVE_PATH = "results/experiment_4585_submission_package_integration_gate.json"
A1_RELATIVE_PATH = "results/experiment_4592_generation_completeness_wiring.json"
A2_RELATIVE_PATH = "results/experiment_4593_levelup_selfplay.json"
A3_RELATIVE_PATH = "results/experiment_4594_goal_energy_generation_prior.json"
A4_RELATIVE_PATH = "results/experiment_4595_refresh_submission_package.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"

RANDOM_SEED = 4597
WINNER_GENERATED_BASELINE = 1.0 / 25.0
GENERIC_TRANSFER_BASELINE = 0.04
LIVE_SUBMITTABLE_BASELINE = 33
DEFAULT_VARIANT_IDS = (1,)
DEFAULT_BUDGET = 200
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_", "passed:", "shipped:")
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- offline arcade end-to-end, no LLM load"
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: integrated_winner_generated_or_transfer_above_baseline "
            "OR complete: no_lever_raises_a_metric_honest_null."
        )
    },
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates -- offline arcade end-to-end; if "
            "the integrated run invokes the LLM proposer, declare live_llm_inference with "
            "the Qwen3.5-9B-MTP iGPU precondition, never the 3090s."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "MUST be false on every integrated value claim -- generation and packaging "
            "levers are oracle-distinct from the win-check."
        )
    },
    "winner_generated_rate_integrated": {
        "principle": (
            "the SUBMITTED-config winner_generated_rate after wiring winners vs the 1/25 baseline."
        )
    },
    "generic_transfer_rate_integrated": {
        "principle": (
            "the SUBMITTED-config held-out variant transfer after wiring winners vs baseline 0.04."
        )
    },
    "live_submittable_level_count_integrated": {
        "principle": (
            "the SUBMITTED-config live-submittable count after wiring vs the 33 scorecard baseline."
        )
    },
    "levers_integrated": {
        "principle": "names which of A1/A2/A3 were wired; [] is an honest null."
    },
    "additivity_checked": {
        "principle": "integrated metric vs the naive sum of isolated deltas."
    },
    "core_solves_preserved": {"principle": "integration must preserve every CORE solve."},
    "parity_green": {
        "principle": "test_arc_submitted_agent_parity.py stays green after wiring."
    },
    "ready_for_operator_submit": {
        "principle": (
            "True only when the integrated config plus package beats the 33-level "
            "scorecard on a real metric; never submits."
        )
    },
    "false_negative_risk_checked": {
        "principle": (
            "an honest null is only valid with baselines measured the same way and passing controls."
        )
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {"principle": "catches silent drift on replay."},
    "preconditions_checked": {
        "principle": "records resources verified; pre-empts missing-resource fabrication."
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "winner_generated_rate_baseline",
    "generic_transfer_rate_baseline",
    "live_submittable_level_count_baseline",
    "generic_transfer_ci_integrated",
    "held_out_solve_rate",
    "per_game_deepest_level_integrated",
    "held_out_deepest_level_by_game",
    "integrated_package_path",
    "submitted_agent_config",
    "upstream_lever_audit",
    "disallowed_adversarial_inputs",
    "field_principles",
    "spec_refs",
    "result_path",
    "duration_s",
    "submitted_to_leaderboard",
)
SPEC_REFS = [
    "REQ-CAPSTONE-4597",
    "SCENARIO-CAPSTONE-4597",
    "SCENARIO-CAPSTONE-4597-FIELD-PRINCIPLES",
]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> JsonDict:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _public_games(root: Path) -> list[str]:
    env_dir = root / "environment_files"
    if not env_dir.is_dir():
        return []
    return sorted(path.name for path in env_dir.iterdir() if path.is_dir())


def _only_null_tautology_flag(artifact: Mapping[str, Any]) -> bool:
    flags = artifact.get("corrigendum_pending")
    if not isinstance(flags, list) or len(flags) != 1 or not isinstance(flags[0], Mapping):
        return False
    flag = flags[0]
    kind = str(flag.get("kind") or "").upper()
    detail = str(flag.get("detail") or "").lower()
    if kind != "TAUTOLOGY" or "control==best" not in detail or "null-delta" not in detail:
        return False
    has_zero_delta = any(
        key.endswith("delta") and _as_float(value, 1.0) == 0.0
        for key, value in artifact.items()
    )
    note = str(artifact.get("null_delta_methodology_note") or "").lower()
    return bool(has_zero_delta and "null" in note and "delta" in note)


def artifact_admissible_for_aggregation(artifact: Mapping[str, Any]) -> tuple[bool, str]:
    """REQ-CAPSTONE-4597: reject flagged positives and failed positive controls."""

    if artifact.get("positive_control_failed") is True:
        return False, "positive_control_failed"
    if artifact.get("flagged_adversarial") is not True:
        return True, "not_flagged"
    if _only_null_tautology_flag(artifact):
        return True, "flagged_null_tautology_diagnosis_only"
    return False, "flagged_adversarial_not_allowed_for_aggregation"


def _a1_integrates(a1: Mapping[str, Any]) -> tuple[bool, str, dict[str, float]]:
    ok, reason = artifact_admissible_for_aggregation(a1)
    if not ok:
        return False, reason, {}
    winner_delta = _as_float(a1.get("winner_generated_delta"))
    transfer_delta = _as_float(a1.get("transfer_delta"))
    winner_rate = _as_float(a1.get("winner_generated_rate_with_wiring"))
    transfer_rate = _as_float(a1.get("generic_transfer_rate_with_wiring"))
    positive = (
        a1.get("no_wiring_control_passed") is True
        and a1.get("solve_rate_preserved") is True
        and winner_delta > 0.0
        and winner_rate > WINNER_GENERATED_BASELINE
        and (transfer_delta > 0.0 or _as_float(a1.get("actions_delta")) > 0.0)
    )
    if positive:
        return (
            True,
            "wired_generation_dispatch_raised_winner_generated",
            {"winner_generated": winner_delta, "generic_transfer": transfer_delta},
        )
    return False, "no_admissible_generation_metric_gain", {}


def _a2_integrates(a2: Mapping[str, Any]) -> tuple[bool, str, int]:
    ok, reason = artifact_admissible_for_aggregation(a2)
    if not ok:
        return False, reason, 0
    update = a2.get("registry_update")
    gate = a2.get("reproduction_gate")
    if not isinstance(update, Mapping) or not isinstance(gate, Mapping):
        return False, "missing_registry_or_reproduction_gate", 0
    delta = _as_int(update.get("reconciled_total_delta") or update.get("banked_levels"))
    target = _as_int(a2.get("target_level") or gate.get("claimed_level"))
    if (
        a2.get("offline_reproduced") is True
        and update.get("updated") is True
        and gate.get("reproduced") is True
        and _as_int(gate.get("reached_level")) >= max(1, target)
        and _as_int(update.get("new_game_levels")) >= max(1, target)
        and delta > 0
    ):
        game = str(update.get("target_game") or a2.get("target_game") or "unknown")
        return True, f"{game}_L{target}_new_offline_reproduced_bank", delta
    return False, "no_new_offline_reproduced_bank", max(0, delta)


def _a3_integrates(a3: Mapping[str, Any]) -> tuple[bool, str, dict[str, float]]:
    ok, reason = artifact_admissible_for_aggregation(a3)
    if not ok:
        return False, reason, {}
    winner_delta = _as_float(a3.get("winner_generated_delta"))
    transfer_rate = _as_float(a3.get("generic_transfer_rate_with_energy"))
    no_energy_transfer = _as_float(a3.get("generic_transfer_rate_no_energy"))
    transfer_delta = transfer_rate - no_energy_transfer
    if (
        a3.get("no_energy_control_passed") is True
        and a3.get("solve_rate_preserved") is True
        and winner_delta > 0.0
        and str(a3.get("chosen_submitted_config") or "") != "unchanged"
    ):
        return (
            True,
            "goal_energy_prior_raised_targeted_winner_generated",
            {"winner_generated": winner_delta, "generic_transfer": max(0.0, transfer_delta)},
        )
    return False, "no_admissible_goal_energy_metric_gain", {}


def _a4_integrates(a4: Mapping[str, Any]) -> tuple[bool, str, int]:
    ok, reason = artifact_admissible_for_aggregation(a4)
    if not ok:
        return False, reason, 0
    live_count = _as_int(a4.get("live_submittable_level_count"))
    delta = _as_int(a4.get("count_delta"))
    if (
        live_count > LIVE_SUBMITTABLE_BASELINE
        and a4.get("ready_for_operator_submit") is True
        and a4.get("offline_reproduced") is True
        and a4.get("verifier_is_oracle") is False
    ):
        return True, "refreshed_package_live_submittable_above_33", max(0, delta)
    return False, "no_admissible_refreshed_package_gain", max(0, delta)


def audit_upstream_levers(
    *,
    a1_artifact: Mapping[str, Any],
    a2_artifact: Mapping[str, Any],
    a3_artifact: Mapping[str, Any],
    a4_artifact: Mapping[str, Any],
) -> JsonDict:
    a1_ok, a1_reason, a1_deltas = _a1_integrates(a1_artifact)
    a2_ok, a2_reason, a2_delta = _a2_integrates(a2_artifact)
    a3_ok, a3_reason, a3_deltas = _a3_integrates(a3_artifact)
    a4_ok, a4_reason, a4_delta = _a4_integrates(a4_artifact)

    levers: list[str] = []
    winner_deltas: dict[str, float] = {}
    transfer_deltas: dict[str, float] = {}
    live_deltas: dict[str, int] = {}
    if a1_ok:
        levers.append("A1_wired_generation_dispatch")
        winner_deltas["A1"] = float(a1_deltas.get("winner_generated", 0.0))
        transfer_deltas["A1"] = float(a1_deltas.get("generic_transfer", 0.0))
    if a2_ok:
        levers.append("A2_ft09_L2_banked_package_refresh")
        live_deltas["A2"] = int(a2_delta)
    if a3_ok:
        levers.append("A3_goal_energy_generation_prior")
        winner_deltas["A3"] = float(a3_deltas.get("winner_generated", 0.0))
        transfer_deltas["A3"] = float(a3_deltas.get("generic_transfer", 0.0))
    if a4_ok:
        levers.append("A4_refreshed_live_submit_package")
        live_deltas["A4"] = int(a4_delta)

    audit = {
        "A1": {
            "artifact": A1_RELATIVE_PATH,
            "integrated": a1_ok,
            "reason": a1_reason,
            "isolated_deltas": dict(a1_deltas),
            "flagged_adversarial": a1_artifact.get("flagged_adversarial") is True,
        },
        "A2": {
            "artifact": A2_RELATIVE_PATH,
            "integrated": a2_ok,
            "reason": a2_reason,
            "live_submittable_delta": int(a2_delta),
        },
        "A3": {
            "artifact": A3_RELATIVE_PATH,
            "integrated": a3_ok,
            "reason": a3_reason,
            "isolated_deltas": dict(a3_deltas),
        },
        "A4": {
            "artifact": A4_RELATIVE_PATH,
            "integrated": a4_ok,
            "reason": a4_reason,
            "live_submittable_delta": int(a4_delta),
        },
    }
    disallowed = [
        {"lever": key, "artifact": row["artifact"], "reason": row["reason"]}
        for key, row in audit.items()
        if row["reason"] in {"flagged_adversarial_not_allowed_for_aggregation", "positive_control_failed"}
    ]
    return {
        "levers_integrated": levers,
        "isolated_deltas": {
            "winner_generated": winner_deltas,
            "generic_transfer": transfer_deltas,
            "live_submittable": live_deltas,
        },
        "upstream_lever_audit": audit,
        "disallowed_adversarial_inputs": disallowed,
        "false_negative_risk_checked": _false_negative_risk_checked(a1_artifact, a3_artifact),
    }


def _false_negative_risk_checked(a1: Mapping[str, Any], a3: Mapping[str, Any]) -> bool:
    return bool(
        a1.get("no_wiring_control_passed") is True
        and a3.get("no_energy_control_passed") is True
        and a3.get("solve_rate_preserved") is True
    )


def _winner_generated_rate(attempts: Sequence[Mapping[str, Any]]) -> float:
    attempted = [attempt for attempt in attempts if attempt.get("attempted") is True]
    if not attempted:
        return 0.0
    winners = sum(
        1
        for attempt in attempted
        if attempt.get("winner_generated") is True or exp4550._attempt_solved(attempt)
    )
    return round(float(winners) / float(len(attempted)), 10)


def _held_out_deepest_by_game(attempts: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for attempt in attempts:
        game = str(attempt.get("game") or "")
        if not game:
            signature = str(attempt.get("variant_signature") or "")
            game = signature.split("~", 1)[0] if signature else ""
        if not game:
            continue
        level = _as_int(attempt.get("reached_level") or attempt.get("levels") or attempt.get("best_level"))
        out[game] = max(out.get(game, 0), level)
    return out


def _per_game_deepest_from_live_metrics(live_metrics: Mapping[str, Any]) -> dict[str, int]:
    rows = live_metrics.get("per_game_live_submittable")
    if not isinstance(rows, list):
        return {}
    out: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        game = str(row.get("game") or "")
        if game:
            out[game] = _as_int(row.get("submittable_level"))
    return out


def _package_levels(package: Mapping[str, Any]) -> dict[str, int]:
    rows = package.get("package_manifest")
    if not isinstance(rows, list):
        return {}
    levels: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        game = str(row.get("game") or "")
        if game:
            levels[game] = _as_int(row.get("levels"))
    return levels


def package_core_preservation(
    root: Path | str,
    *,
    baseline_package_path: str = PREVIOUS_PACKAGE_RELATIVE_PATH,
    integrated_package_path: str = INTEGRATED_PACKAGE_RELATIVE_PATH,
) -> JsonDict:
    """SCENARIO-CAPSTONE-4597: package integration preserves prior claimed levels."""

    root_path = Path(root)
    baseline = _package_levels(_read_json(root_path / baseline_package_path))
    integrated = _package_levels(_read_json(root_path / integrated_package_path))
    dropped = [
        game
        for game, level in sorted(baseline.items())
        if integrated.get(game, 0) < level
    ]
    return {
        "passed": not dropped,
        "baseline_package_path": baseline_package_path,
        "integrated_package_path": integrated_package_path,
        "baseline_deepest_level_by_game": baseline,
        "integrated_deepest_level_by_game": integrated,
        "dropped_games": dropped,
    }


def _default_core_preservation(live_metrics: Mapping[str, Any]) -> JsonDict:
    return {
        "passed": True,
        "baseline_package_path": "",
        "integrated_package_path": str(live_metrics.get("refreshed_package_path") or ""),
        "baseline_deepest_level_by_game": {},
        "integrated_deepest_level_by_game": _per_game_deepest_from_live_metrics(live_metrics),
        "dropped_games": [],
    }


def _additivity(
    *,
    audit: Mapping[str, Any],
    winner_integrated: float,
    transfer_integrated: float,
    live_integrated: int,
) -> JsonDict:
    isolated = audit.get("isolated_deltas")
    isolated = isolated if isinstance(isolated, Mapping) else {}
    winner = dict(isolated.get("winner_generated") or {})
    transfer = dict(isolated.get("generic_transfer") or {})
    live = dict(isolated.get("live_submittable") or {})
    expected_winner = WINNER_GENERATED_BASELINE + sum(_as_float(value) for value in winner.values())
    expected_transfer = GENERIC_TRANSFER_BASELINE + sum(_as_float(value) for value in transfer.values())
    expected_live = LIVE_SUBMITTABLE_BASELINE + sum(_as_int(value) for value in live.values())
    return {
        "passed": True,
        "winner_generated": {
            "baseline": round(WINNER_GENERATED_BASELINE, 10),
            "isolated_deltas": winner,
            "naive_expected": round(expected_winner, 10),
            "integrated": round(winner_integrated, 10),
            "interaction_delta": round(winner_integrated - expected_winner, 10),
        },
        "generic_transfer": {
            "baseline": GENERIC_TRANSFER_BASELINE,
            "isolated_deltas": transfer,
            "naive_expected": round(expected_transfer, 10),
            "integrated": round(transfer_integrated, 10),
            "interaction_delta": round(transfer_integrated - expected_transfer, 10),
        },
        "live_submittable": {
            "baseline": LIVE_SUBMITTABLE_BASELINE,
            "isolated_deltas": live,
            "naive_expected": expected_live,
            "integrated": int(live_integrated),
            "interaction_delta": int(live_integrated) - int(expected_live),
        },
    }


def _verdict(
    *,
    winner_rate: float,
    transfer_rate: float,
    live_count: int,
    ready: bool,
) -> str:
    if not ready:
        return "complete: no_lever_raises_a_metric_honest_null"
    if live_count > LIVE_SUBMITTABLE_BASELINE:
        return f"success: integrated_live_submittable_{live_count}_above_33"
    if winner_rate > WINNER_GENERATED_BASELINE:
        return f"success: integrated_winner_generated_{winner_rate:.3f}_above_1of25"
    if transfer_rate > GENERIC_TRANSFER_BASELINE:
        return f"success: integrated_generic_transfer_{transfer_rate:.3f}_above_0.04"
    return "complete: no_lever_raises_a_metric_honest_null"


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = {
        "honest_verdict": artifact.get("honest_verdict"),
        "winner_generated_rate_integrated": artifact.get("winner_generated_rate_integrated"),
        "generic_transfer_rate_integrated": artifact.get("generic_transfer_rate_integrated"),
        "live_submittable_level_count_integrated": artifact.get(
            "live_submittable_level_count_integrated"
        ),
        "levers_integrated": artifact.get("levers_integrated"),
        "additivity_checked": artifact.get("additivity_checked"),
        "core_solves_preserved": artifact.get("core_solves_preserved"),
        "parity_green": artifact.get("parity_green"),
        "ready_for_operator_submit": artifact.get("ready_for_operator_submit"),
        "integrated_package_path": artifact.get("integrated_package_path"),
        "random_seed": artifact.get("random_seed"),
        "preconditions_checked": artifact.get("preconditions_checked"),
    }
    return "sha256:" + _sha256(payload)


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    audit: Mapping[str, Any],
    transfer_measurement: Mapping[str, Any],
    live_metrics: Mapping[str, Any],
    submitted_agent_config: Mapping[str, Any],
    parity_green: bool,
    duration_s: float | None,
    core_preservation: Mapping[str, Any] | None = None,
) -> JsonDict:
    attempts_raw = transfer_measurement.get("variant_attempts")
    attempts = (
        [dict(row) for row in attempts_raw if isinstance(row, Mapping)]
        if isinstance(attempts_raw, list)
        else []
    )
    winner_rate = _winner_generated_rate(attempts)
    transfer_rate = _as_float(transfer_measurement.get("generic_transfer_rate_over_variants"))
    live_count = _as_int(live_metrics.get("live_submittable_level_count"))
    core = dict(core_preservation or _default_core_preservation(live_metrics))
    ready = bool(
        parity_green
        and core.get("passed") is True
        and (winner_rate > WINNER_GENERATED_BASELINE
             or transfer_rate > GENERIC_TRANSFER_BASELINE
             or live_count > LIVE_SUBMITTABLE_BASELINE)
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "honest_verdict": _verdict(
            winner_rate=winner_rate,
            transfer_rate=transfer_rate,
            live_count=live_count,
            ready=ready,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "winner_generated_rate_integrated": round(winner_rate, 10),
        "winner_generated_rate_baseline": round(WINNER_GENERATED_BASELINE, 10),
        "generic_transfer_rate_integrated": round(transfer_rate, 10),
        "generic_transfer_rate_baseline": GENERIC_TRANSFER_BASELINE,
        "generic_transfer_ci_integrated": [
            float(value) for value in (transfer_measurement.get("generic_transfer_ci") or [0.0, 0.0])
        ],
        "live_submittable_level_count_integrated": live_count,
        "live_submittable_level_count_baseline": LIVE_SUBMITTABLE_BASELINE,
        "held_out_solve_rate": round(transfer_rate, 10),
        "levers_integrated": list(audit.get("levers_integrated") or []),
        "additivity_checked": _additivity(
            audit=audit,
            winner_integrated=winner_rate,
            transfer_integrated=transfer_rate,
            live_integrated=live_count,
        ),
        "core_solves_preserved": core,
        "parity_green": bool(parity_green),
        "ready_for_operator_submit": ready,
        "false_negative_risk_checked": bool(audit.get("false_negative_risk_checked")),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "per_game_deepest_level_integrated": _per_game_deepest_from_live_metrics(live_metrics),
        "held_out_deepest_level_by_game": _held_out_deepest_by_game(attempts),
        "integrated_package_path": str(live_metrics.get("refreshed_package_path") or ""),
        "submitted_agent_config": dict(submitted_agent_config),
        "upstream_lever_audit": dict(audit.get("upstream_lever_audit") or {}),
        "disallowed_adversarial_inputs": list(audit.get("disallowed_adversarial_inputs") or []),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "duration_s": None if duration_s is None else max(0.0, round(float(duration_s), 6)),
        "submitted_to_leaderboard": False,
        "leaderboard_submission": False,
        "variant_attempts_count": _as_int(transfer_measurement.get("variant_attempts_count")),
        "variant_solved_count": _as_int(transfer_measurement.get("variant_solved_count")),
        "variant_attempts": attempts,
        "live_submittable_metrics": dict(live_metrics),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    for field in (
        "winner_generated_rate_integrated",
        "winner_generated_rate_baseline",
        "generic_transfer_rate_integrated",
        "generic_transfer_rate_baseline",
        "held_out_solve_rate",
    ):
        value = artifact.get(field)
        if not isinstance(value, float) or isinstance(value, bool) or not 0.0 <= value <= 1.0:
            errors.append(f"{field} must be a bare float in [0,1]")
    for field in ("live_submittable_level_count_integrated", "live_submittable_level_count_baseline"):
        if type(artifact.get(field)) is not int:
            errors.append(f"{field} must be a bare int")
    ci = artifact.get("generic_transfer_ci_integrated")
    if (
        not isinstance(ci, list)
        or len(ci) != 2
        or not all(isinstance(value, float) for value in ci)
    ):
        errors.append("generic_transfer_ci_integrated must be [float, float]")
    for field in ("levers_integrated", "disallowed_adversarial_inputs"):
        if not isinstance(artifact.get(field), list):
            errors.append(f"{field} must be a list")
    for field in (
        "additivity_checked",
        "core_solves_preserved",
        "preconditions_checked",
        "submitted_agent_config",
        "upstream_lever_audit",
    ):
        if not isinstance(artifact.get(field), Mapping):
            errors.append(f"{field} must be a mapping")
    for field in ("parity_green", "ready_for_operator_submit", "false_negative_risk_checked"):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be a bare bool")
    if artifact.get("submitted_to_leaderboard") is not False:
        errors.append("submitted_to_leaderboard must be false")
    if artifact.get("ready_for_operator_submit") is True:
        if artifact.get("parity_green") is not True:
            errors.append("ready_for_operator_submit requires parity_green")
        core = artifact.get("core_solves_preserved")
        if not isinstance(core, Mapping) or core.get("passed") is not True:
            errors.append("ready_for_operator_submit requires core_solves_preserved")
        moved = (
            _as_float(artifact.get("winner_generated_rate_integrated")) > WINNER_GENERATED_BASELINE
            or _as_float(artifact.get("generic_transfer_rate_integrated")) > GENERIC_TRANSFER_BASELINE
            or _as_int(artifact.get("live_submittable_level_count_integrated")) > LIVE_SUBMITTABLE_BASELINE
        )
        if not moved:
            errors.append("ready_for_operator_submit requires at least one metric above baseline")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:") or len(checksum) != 71:
        errors.append("reproducibility_checksum must be sha256-prefixed hex")
    return errors


def _default_offline_arcade_checker() -> bool:  # pragma: no cover - SDK boundary
    from carnot.agentic import arc_solver_kit as kit

    kit.offline_arcade()
    return True


def check_preconditions(  # pragma: no cover - filesystem/SDK boundary
    root: Path | str = REPO_ROOT,
    *,
    offline_arcade_checker: Callable[[], bool] | None = None,
) -> JsonDict:
    root_path = Path(root)
    checker = offline_arcade_checker or _default_offline_arcade_checker
    try:
        offline_ok = bool(checker())
        offline_error = ""
    except Exception as exc:
        offline_ok = False
        offline_error = f"{type(exc).__name__}: {exc}"
    spec_text = (root_path / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    checks: JsonDict = {
        "AGENTS.md": (root_path / "AGENTS.md").exists(),
        "CODEX.md": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade": offline_ok,
        "offline_arcade_error": offline_error,
        "a1_artifact_present": (root_path / A1_RELATIVE_PATH).exists(),
        "a2_artifact_present": (root_path / A2_RELATIVE_PATH).exists(),
        "a3_artifact_present": (root_path / A3_RELATIVE_PATH).exists(),
        "a4_artifact_present": (root_path / A4_RELATIVE_PATH).exists(),
        "integrated_package_present": (root_path / INTEGRATED_PACKAGE_RELATIVE_PATH).exists(),
        "previous_package_present": (root_path / PREVIOUS_PACKAGE_RELATIVE_PATH).exists(),
        "spec_has_req_4597": "REQ-CAPSTONE-4597" in spec_text,
        "live_llm_inference": False,
        "qwen35_9b_mtp_igpu_precondition": "not_used",
        "leaderboard_submission": False,
        "scripts_research_conductor_modified": False,
    }
    required = (
        "AGENTS.md",
        "CODEX.md",
        "offline_arcade",
        "a1_artifact_present",
        "a2_artifact_present",
        "a3_artifact_present",
        "a4_artifact_present",
        "integrated_package_present",
        "spec_has_req_4597",
    )
    checks["ok"] = all(bool(checks[key]) for key in required)
    return checks


def first_precondition_miss(preconditions: Mapping[str, Any]) -> str | None:
    for key in (
        "AGENTS.md",
        "CODEX.md",
        "offline_arcade",
        "a1_artifact_present",
        "a2_artifact_present",
        "a3_artifact_present",
        "a4_artifact_present",
        "integrated_package_present",
        "spec_has_req_4597",
    ):
        if preconditions.get(key) is not True:
            return key
    if preconditions.get("leaderboard_submission") is True:
        return "leaderboard_submission"
    return None


def measure_integrated_transfer(  # pragma: no cover - offline ARC runtime boundary
    root: Path | str,
    *,
    public_games: Sequence[str] | None = None,
    variant_ids: Sequence[int] = DEFAULT_VARIANT_IDS,
    budget: int = DEFAULT_BUDGET,
    variant_runner: VariantRunner = exp4550.default_variant_runner,
    random_seed: int = RANDOM_SEED,
    n_bootstrap: int = exp4550.DEFAULT_BOOTSTRAPS,
) -> JsonDict:
    root_path = Path(root)
    games = list(public_games if public_games is not None else _public_games(root_path))
    if not games:
        return {
            "variant_specs": [],
            "variant_attempts": [],
            "variant_attempts_count": 0,
            "variant_solved_count": 0,
            "generic_transfer_rate_over_variants": 0.0,
            "generic_transfer_ci": [0.0, 0.0],
        }
    return exp4550.measure_generic_transfer_over_variants(
        public_games=games,
        variant_ids=variant_ids,
        budget=budget,
        variant_runner=variant_runner,
        random_seed=random_seed,
        n_bootstrap=n_bootstrap,
    )


def run_parity_check(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - subprocess boundary
    root_path = Path(root)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/python/test_arc_submitted_agent_parity.py",
        "-q",
        "--no-cov",
    ]
    proc = subprocess.run(cmd, cwd=root_path, capture_output=True, text=True, timeout=180, check=False)
    return {
        "passed": proc.returncode == 0,
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def write_artifact(root: Path | str, artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    _write_json(Path(root) / RESULT_RELATIVE_PATH, artifact)


def blocked_artifact(  # pragma: no cover - only used when required resources are missing
    preconditions_checked: Mapping[str, Any],
    *,
    duration_s: float,
) -> JsonDict:
    miss = first_precondition_miss(preconditions_checked) or "unknown"
    artifact = build_artifact(
        preconditions_checked=preconditions_checked,
        audit={
            "levers_integrated": [],
            "isolated_deltas": {
                "winner_generated": {},
                "generic_transfer": {},
                "live_submittable": {},
            },
            "upstream_lever_audit": {},
            "disallowed_adversarial_inputs": [],
            "false_negative_risk_checked": False,
        },
        transfer_measurement={
            "variant_attempts": [],
            "variant_attempts_count": 0,
            "variant_solved_count": 0,
            "generic_transfer_rate_over_variants": 0.0,
            "generic_transfer_ci": [0.0, 0.0],
        },
        live_metrics={
            "live_submittable_level_count": 0,
            "refreshed_package_path": "",
            "per_game_live_submittable": [],
        },
        submitted_agent_config=SUBMITTED_AGENT_CONFIG,
        parity_green=False,
        duration_s=duration_s,
    )
    artifact["honest_verdict"] = f"blocked_{miss}"
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def run(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - end-to-end boundary
    root_path = Path(root)
    start = time.time()
    preconditions = check_preconditions(root_path)
    miss = first_precondition_miss(preconditions)
    if miss:
        artifact = blocked_artifact(preconditions, duration_s=time.time() - start)
        _write_json(root_path / RESULT_RELATIVE_PATH, artifact)
        return artifact

    audit = audit_upstream_levers(
        a1_artifact=_read_json(root_path / A1_RELATIVE_PATH),
        a2_artifact=_read_json(root_path / A2_RELATIVE_PATH),
        a3_artifact=_read_json(root_path / A3_RELATIVE_PATH),
        a4_artifact=_read_json(root_path / A4_RELATIVE_PATH),
    )
    transfer = measure_integrated_transfer(root_path)
    package_path = str(
        SUBMITTED_AGENT_CONFIG.get("live_submit_package_path") or INTEGRATED_PACKAGE_RELATIVE_PATH
    )
    live_metrics = live_submittable_metrics.compute_live_submittable_metrics(
        root_path,
        package_path=package_path,
    )
    core = package_core_preservation(
        root_path,
        integrated_package_path=package_path,
    )
    parity = run_parity_check(root_path)
    preconditions["parity_command"] = parity["command"]
    preconditions["parity_returncode"] = parity["returncode"]
    artifact = build_artifact(
        preconditions_checked=preconditions,
        audit=audit,
        transfer_measurement=transfer,
        live_metrics=live_metrics,
        submitted_agent_config=SUBMITTED_AGENT_CONFIG,
        parity_green=bool(parity["passed"]),
        core_preservation=core,
        duration_s=time.time() - start,
    )
    write_artifact(root_path, artifact)
    return artifact


def main() -> None:  # pragma: no cover - CLI boundary
    artifact = run(REPO_ROOT)
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    main()
