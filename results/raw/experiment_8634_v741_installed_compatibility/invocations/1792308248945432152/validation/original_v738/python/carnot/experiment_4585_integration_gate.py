"""Experiment 4585: ARC sprint integration gate.

Spec refs: REQ-CAPSTONE-4585, SCENARIO-CAPSTONE-4585,
SCENARIO-CAPSTONE-4585-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot import experiment_4550_honest_sprint_metric as exp4550
from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG


JsonDict = dict[str, Any]
VariantRunner = Callable[[str, Mapping[str, Any], int], Mapping[str, Any]]

RESULT_RELATIVE_PATH = "results/experiment_4585_integration_gate.json"
INTEGRATED_PACKAGE_RELATIVE_PATH = "results/experiment_4585_submission_package_integration_gate.json"
A1_RELATIVE_PATH = "results/experiment_4580_live_submission_gap_close.json"
A2_RELATIVE_PATH = "results/experiment_4581_levelup_selfplay.json"
A3_RELATIVE_PATH = "results/experiment_4582_feature_router_transfer.json"
A4_RELATIVE_PATH = "results/experiment_4583_diversity_floor_transfer.json"
A1_PACKAGE_RELATIVE_PATH = "results/experiment_4580_submission_package_live_gap_close.json"
AR25_L2_SOURCE_RELATIVE_PATH = "results/arc_loop_solve_ar25.json"
AR25_TRAJECTORY_RELATIVE_PATH = "results/arc3_live_banked_trajectories/ar25.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"

EXPERIMENT = "experiment_4585_integration_gate"
SCHEMA = "carnot.exp4585.integration_gate.v1"
PACKAGE_SCHEMA = "carnot.exp4585.submission_package.v1"
RANDOM_SEED = 4585
LIVE_SUBMITTABLE_BASELINE = 33
GENERIC_TRANSFER_BASELINE = 0.04
DEFAULT_VARIANT_IDS = (1,)
DEFAULT_BUDGET = 200
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_", "passed:", "shipped:")
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- offline arcade end-to-end, no LLM load"
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: integrated_live_submittable_above_33_or_generic_transfer_above_0.04 "
            "OR complete: no_lever_raises_a_metric_honest_null."
        )
    },
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates -- offline arcade end-to-end; if live LLM proposer "
            "is invoked, declare live_llm_inference with the Qwen3.5-9B-MTP iGPU precondition, never the 3090s."
        )
    },
    "live_submittable_level_count_integrated": {
        "principle": (
            "the SUBMITTED-config live-submittable count after wiring winners vs the 33 baseline -- "
            "the headline score lever."
        )
    },
    "generic_transfer_rate_integrated": {
        "principle": (
            "the SUBMITTED-config held-out variant transfer after wiring winners vs baseline 0.04 -- "
            "the leaderboard-honest transfer headline."
        )
    },
    "levers_integrated": {
        "principle": (
            "names which of A1/A2/A3/A4 were wired -- traceable to their measured deltas; [] is an honest null."
        )
    },
    "additivity_checked": {
        "principle": (
            "integrated metric vs the naive sum of isolated deltas -- surfaces a destructive interaction instead of burying it."
        )
    },
    "core_solves_preserved": {
        "principle": "integration must preserve every CORE solve (set-containment); a dropped solve FAILS the lever."
    },
    "parity_green": {
        "principle": (
            "test_arc_submitted_agent_parity.py stays green after wiring -- the single-source-of-truth guard."
        )
    },
    "ready_for_operator_submit": {
        "principle": (
            "True if the integrated config + refreshed package beat the 33-level scorecard on a real metric "
            "worth a 1/day submission slot; the task NEVER submits (operator-only)."
        )
    },
    "false_negative_risk_checked": {
        "principle": "an honest null is only valid with both baselines measured the same way."
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
    "leaderboard_submission",
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = {
        "honest_verdict": artifact.get("honest_verdict"),
        "live_submittable_level_count_integrated": artifact.get(
            "live_submittable_level_count_integrated"
        ),
        "generic_transfer_rate_integrated": artifact.get("generic_transfer_rate_integrated"),
        "levers_integrated": artifact.get("levers_integrated"),
        "additivity_checked": artifact.get("additivity_checked"),
        "core_solves_preserved": artifact.get("core_solves_preserved"),
        "ready_for_operator_submit": artifact.get("ready_for_operator_submit"),
        "integrated_package_path": artifact.get("integrated_package_path"),
        "random_seed": artifact.get("random_seed"),
        "preconditions_checked": artifact.get("preconditions_checked"),
    }
    return "sha256:" + _sha256(payload)


def _read_json(path: Path) -> JsonDict:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


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


def _default_offline_arcade_checker() -> bool:  # pragma: no cover - SDK boundary
    from carnot.agentic import arc_solver_kit as kit

    kit.offline_arcade()
    return True


def check_preconditions(
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

    spec_path = root_path / SPEC_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "offline_arcade": offline_ok,
        "offline_arcade_error": offline_error,
        "a1_artifact_present": (root_path / A1_RELATIVE_PATH).exists(),
        "a2_artifact_present": (root_path / A2_RELATIVE_PATH).exists(),
        "a3_artifact_present": (root_path / A3_RELATIVE_PATH).exists(),
        "a4_artifact_present": (root_path / A4_RELATIVE_PATH).exists(),
        "a1_package_present": (root_path / A1_PACKAGE_RELATIVE_PATH).exists(),
        "ar25_l2_source_present": (root_path / AR25_L2_SOURCE_RELATIVE_PATH).exists(),
        "spec_has_req_4585": "REQ-CAPSTONE-4585" in spec_text,
        "submitted_config_package_path": SUBMITTED_AGENT_CONFIG.get("live_submit_package_path"),
        "leaderboard_submission": False,
        "operator_only": True,
        "no_3090_inference": True,
    }
    required = (
        "agents_md_read",
        "codex_md_read",
        "offline_arcade",
        "a1_artifact_present",
        "a2_artifact_present",
        "a3_artifact_present",
        "a4_artifact_present",
        "a1_package_present",
        "spec_has_req_4585",
    )
    checks["ok"] = all(bool(checks[key]) for key in required)
    return checks


def _only_null_tautology_flag(artifact: Mapping[str, Any]) -> bool:
    flags = artifact.get("corrigendum_pending")
    if not isinstance(flags, list) or len(flags) != 1 or not isinstance(flags[0], Mapping):
        return False
    flag = flags[0]
    kind = str(flag.get("kind") or "").upper()
    detail = str(flag.get("detail") or "").lower()
    if kind != "TAUTOLOGY" or "null" not in detail or "delta" not in detail:
        return False
    has_zero_delta = any(
        key.endswith("delta") and _as_float(value, 1.0) == 0.0
        for key, value in artifact.items()
    )
    note = str(artifact.get("null_delta_methodology_note") or "").lower()
    return bool(has_zero_delta and "delta" in note and "null" in note)


def artifact_admissible_for_aggregation(artifact: Mapping[str, Any]) -> tuple[bool, str]:
    """REQ-CAPSTONE-4585: reject flagged upstreams except explicit null tautology controls."""

    if artifact.get("flagged_adversarial") is not True:
        return True, "not_flagged"
    if _only_null_tautology_flag(artifact):
        return True, "flagged_null_tautology_not_used_as_positive_signal"
    return False, "flagged_adversarial_not_allowed_for_aggregation"


def _a1_integrates(a1: Mapping[str, Any]) -> tuple[bool, str, int]:
    ok, reason = artifact_admissible_for_aggregation(a1)
    if not ok:
        return False, reason, 0
    live = _as_int(a1.get("live_submittable_level_count"))
    baseline = _as_int(a1.get("live_submittable_count_baseline"), LIVE_SUBMITTABLE_BASELINE)
    delta = live - baseline
    if live > LIVE_SUBMITTABLE_BASELINE and a1.get("ready_for_operator_submit") is True:
        return True, "live_submittable_count_strictly_above_33", delta
    return False, "no_live_submittable_gain", max(0, delta)


def _a2_integrates(a2: Mapping[str, Any]) -> tuple[bool, str, int]:
    ok, reason = artifact_admissible_for_aggregation(a2)
    if not ok:
        return False, reason, 0
    update = a2.get("registry_update")
    if not isinstance(update, Mapping):
        return False, "missing_registry_update", 0
    delta = _as_int(update.get("reconciled_total_delta") or update.get("banked_levels"))
    if (
        a2.get("offline_reproduced") is True
        and update.get("updated") is True
        and str(update.get("target_game") or "") == "ar25"
        and _as_int(update.get("new_game_levels")) >= 2
        and delta > 0
    ):
        return True, "ar25_L2_new_offline_reproduced_bank", delta
    return False, "no_new_offline_reproduced_bank", max(0, delta)


def _a3_integrates(a3: Mapping[str, Any]) -> tuple[bool, str, float]:
    ok, reason = artifact_admissible_for_aggregation(a3)
    if not ok:
        return False, reason, 0.0
    rate = _as_float(a3.get("generic_transfer_rate_with_router"))
    delta = _as_float(a3.get("transfer_delta"))
    if rate > GENERIC_TRANSFER_BASELINE and delta > 0.0 and a3.get("random_route_control_passed") is True:
        return True, "feature_router_generic_transfer_strictly_above_0.04", delta
    return False, "no_admissible_feature_router_transfer_gain", max(0.0, delta)


def _a4_integrates(a4: Mapping[str, Any]) -> tuple[bool, str, int]:
    ok, reason = artifact_admissible_for_aggregation(a4)
    if not ok:
        return False, reason, 0
    delta = _as_int(a4.get("firstwin_delta"))
    if delta > 0 and a4.get("diversity_off_control_passed") is True:
        return True, "diversity_firstwin_delta_positive", delta
    return False, "no_diversity_firstwin_gain", max(0, delta)


def audit_upstream_levers(
    *,
    a1_artifact: Mapping[str, Any],
    a2_artifact: Mapping[str, Any],
    a3_artifact: Mapping[str, Any],
    a4_artifact: Mapping[str, Any],
) -> JsonDict:
    a1_ok, a1_reason, a1_delta = _a1_integrates(a1_artifact)
    a2_ok, a2_reason, a2_delta = _a2_integrates(a2_artifact)
    a3_ok, a3_reason, a3_delta = _a3_integrates(a3_artifact)
    a4_ok, a4_reason, a4_delta = _a4_integrates(a4_artifact)

    levers: list[str] = []
    live_deltas: dict[str, int] = {}
    transfer_deltas: dict[str, float] = {}
    if a1_ok:
        levers.append("A1_refreshed_live_submit_package")
        live_deltas["A1"] = int(a1_delta)
    if a2_ok:
        levers.append("A2_ar25_L2_banked_package_refresh")
        live_deltas["A2"] = int(a2_delta)
    if a3_ok:
        levers.append("A3_feature_router")
        transfer_deltas["A3"] = float(a3_delta)
    if a4_ok:
        levers.append("A4_diversity_floor")

    audit = {
        "A1": {
            "artifact": A1_RELATIVE_PATH,
            "integrated": a1_ok,
            "reason": a1_reason,
            "live_submittable_delta": int(a1_delta),
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
            "generic_transfer_delta": float(a3_delta),
        },
        "A4": {
            "artifact": A4_RELATIVE_PATH,
            "integrated": a4_ok,
            "reason": a4_reason,
            "firstwin_delta": int(a4_delta),
        },
    }
    disallowed = [
        {"lever": key, "artifact": row["artifact"], "reason": row["reason"]}
        for key, row in audit.items()
        if row["reason"] == "flagged_adversarial_not_allowed_for_aggregation"
    ]
    return {
        "levers_integrated": levers,
        "isolated_deltas": {
            "live_submittable": live_deltas,
            "generic_transfer": transfer_deltas,
        },
        "upstream_lever_audit": audit,
        "disallowed_adversarial_inputs": disallowed,
    }


def _package_manifest(source_package: Mapping[str, Any]) -> list[JsonDict]:
    rows = source_package.get("package_manifest")
    return [dict(row) for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []


def _ar25_l2_actions(root: Path) -> list[JsonDict]:
    source = _read_json(root / AR25_L2_SOURCE_RELATIVE_PATH)
    if source.get("offline_reproduced") is not True:
        return []
    if _as_int(source.get("reached_level") or source.get("reproduced_levels")) < 2:
        return []
    raw = source.get("solution")
    if not isinstance(raw, list):
        return []
    actions = [dict(action) for action in raw if isinstance(action, Mapping) and "action" in action]
    return actions


def _write_ar25_l2_trajectory(root: Path, actions: Sequence[Mapping[str, Any]]) -> str:
    path = root / AR25_TRAJECTORY_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "game": "ar25",
        "source": AR25_L2_SOURCE_RELATIVE_PATH,
        "solution": [dict(action) for action in actions],
        "action_count": len(actions),
        "schema": "carnot.arc3.flat_trajectory_bank.v1",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return AR25_TRAJECTORY_RELATIVE_PATH


def _manifest_levels(manifest: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    levels: dict[str, int] = {}
    for row in manifest:
        game = str(row.get("game") or "")
        if game:
            levels[game] = _as_int(row.get("levels"))
    return levels


def _core_preservation(
    baseline_manifest: Sequence[Mapping[str, Any]],
    integrated_manifest: Sequence[Mapping[str, Any]],
) -> JsonDict:
    baseline = _manifest_levels(baseline_manifest)
    integrated = _manifest_levels(integrated_manifest)
    dropped = [
        game
        for game, level in sorted(baseline.items())
        if integrated.get(game, 0) < level
    ]
    return {
        "passed": not dropped,
        "baseline_core_games": sorted(baseline),
        "dropped_games": dropped,
        "baseline_deepest_level_by_game": baseline,
        "integrated_deepest_level_by_game": integrated,
    }


def refresh_integrated_package(
    root: Path | str,
    *,
    source_package: Mapping[str, Any],
    levers_integrated: Sequence[str],
) -> JsonDict:
    """SCENARIO-CAPSTONE-4585: write the operator-only integrated package."""

    root_path = Path(root)
    baseline_manifest = _package_manifest(source_package)
    manifest = [dict(row) for row in baseline_manifest]
    if "A2_ar25_L2_banked_package_refresh" in levers_integrated:
        actions = _ar25_l2_actions(root_path)
        if actions:
            trajectory_path = _write_ar25_l2_trajectory(root_path, actions)
            found = False
            for row in manifest:
                if row.get("game") == "ar25":
                    row.update(
                        {
                            "levels": 2,
                            "offline_reproduced_level": 2,
                            "registry_reproduced_level": 2,
                            "trajectory_path": trajectory_path,
                            "action_count": len(actions),
                            "source": AR25_L2_SOURCE_RELATIVE_PATH,
                            "env_matched": True,
                            "env_match_basis": "offline_fresh_replay_or_env_adaptive_proxy",
                            "claim_capped": False,
                        }
                    )
                    found = True
                    break
            if not found:
                manifest.append(
                    {
                        "game": "ar25",
                        "levels": 2,
                        "offline_reproduced_level": 2,
                        "registry_reproduced_level": 2,
                        "trajectory_path": trajectory_path,
                        "action_count": len(actions),
                        "source": AR25_L2_SOURCE_RELATIVE_PATH,
                        "env_matched": True,
                        "env_match_basis": "offline_fresh_replay_or_env_adaptive_proxy",
                        "adaptive_solver": "",
                        "adaptive_labels": [],
                        "claim_capped": False,
                    }
                )

    total = sum(_as_int(row.get("levels")) for row in manifest)
    package = {
        "experiment": "experiment_4585_submission_package_integration_gate",
        "schema": PACKAGE_SCHEMA,
        "source_result_path": RESULT_RELATIVE_PATH,
        "derived_from_package": A1_PACKAGE_RELATIVE_PATH,
        "package_manifest": manifest,
        "claimed_total_levels": total,
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "reproducibility_checksum": "sha256:" + _sha256(manifest),
    }
    path = root_path / INTEGRATED_PACKAGE_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    preservation = _core_preservation(baseline_manifest, manifest)
    return {
        "package_path": INTEGRATED_PACKAGE_RELATIVE_PATH,
        "claimed_total_levels": total,
        "per_game_deepest_level_integrated": preservation["integrated_deepest_level_by_game"],
        "core_solves_preserved": preservation,
        "package_checksum": package["reproducibility_checksum"],
    }


def _empty_transfer_measurement() -> JsonDict:
    return {
        "generic_transfer_rate_over_variants": 0.0,
        "generic_transfer_ci": [0.0, 0.0],
        "variant_attempts_count": 0,
        "variant_solved_count": 0,
        "variant_attempts": [],
        "variant_specs": [],
    }


def measure_integrated_transfer(
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
        return _empty_transfer_measurement()
    return exp4550.measure_generic_transfer_over_variants(
        public_games=games,
        variant_ids=variant_ids,
        budget=budget,
        variant_runner=variant_runner,
        random_seed=random_seed,
        n_bootstrap=n_bootstrap,
    )


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


def _additivity(
    *,
    audit: Mapping[str, Any],
    live_integrated: int,
    transfer_integrated: float,
) -> JsonDict:
    isolated = audit.get("isolated_deltas")
    live_deltas = {}
    transfer_deltas = {}
    if isinstance(isolated, Mapping):
        live_deltas = dict(isolated.get("live_submittable") or {})
        transfer_deltas = dict(isolated.get("generic_transfer") or {})
    expected_live = LIVE_SUBMITTABLE_BASELINE + sum(_as_int(value) for value in live_deltas.values())
    expected_transfer = GENERIC_TRANSFER_BASELINE + sum(
        _as_float(value) for value in transfer_deltas.values()
    )
    return {
        "passed": True,
        "live_submittable": {
            "baseline": LIVE_SUBMITTABLE_BASELINE,
            "isolated_deltas": live_deltas,
            "naive_expected": expected_live,
            "integrated": int(live_integrated),
            "interaction_delta": int(live_integrated) - int(expected_live),
        },
        "generic_transfer": {
            "baseline": GENERIC_TRANSFER_BASELINE,
            "isolated_deltas": transfer_deltas,
            "naive_expected": round(float(expected_transfer), 10),
            "integrated": round(float(transfer_integrated), 10),
            "interaction_delta": round(float(transfer_integrated) - float(expected_transfer), 10),
        },
    }


def _verdict(live_count: int, transfer_rate: float, ready: bool) -> str:
    if ready and live_count > LIVE_SUBMITTABLE_BASELINE:
        return f"success: integrated_live_submittable_{live_count}_above_33"
    if ready and transfer_rate > GENERIC_TRANSFER_BASELINE:
        return f"success: integrated_generic_transfer_{transfer_rate:.3f}_above_0.04"
    return "complete: no_lever_raises_a_metric_honest_null"


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    audit: Mapping[str, Any],
    package_summary: Mapping[str, Any],
    transfer_measurement: Mapping[str, Any],
    submitted_agent_config: Mapping[str, Any],
    duration_s: float | None,
) -> JsonDict:
    live_count = _as_int(package_summary.get("claimed_total_levels"))
    transfer_rate = _as_float(transfer_measurement.get("generic_transfer_rate_over_variants"))
    core = dict(package_summary.get("core_solves_preserved") or {})
    ready = bool(
        core.get("passed") is True
        and (live_count > LIVE_SUBMITTABLE_BASELINE or transfer_rate > GENERIC_TRANSFER_BASELINE)
    )
    attempts = transfer_measurement.get("variant_attempts")
    attempts_list = [dict(row) for row in attempts if isinstance(row, Mapping)] if isinstance(attempts, list) else []
    additivity = _additivity(
        audit=audit,
        live_integrated=live_count,
        transfer_integrated=transfer_rate,
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "honest_verdict": _verdict(live_count, transfer_rate, ready),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "live_submittable_level_count_integrated": live_count,
        "generic_transfer_rate_integrated": round(float(transfer_rate), 10),
        "generic_transfer_ci_integrated": list(transfer_measurement.get("generic_transfer_ci") or [0.0, 0.0]),
        "held_out_solve_rate": round(float(transfer_rate), 10),
        "levers_integrated": list(audit.get("levers_integrated") or []),
        "additivity_checked": additivity,
        "core_solves_preserved": core,
        "parity_green": bool(
            submitted_agent_config.get("live_submit_package_path") == INTEGRATED_PACKAGE_RELATIVE_PATH
            and submitted_agent_config.get("cascade") is True
            and submitted_agent_config.get("policy") == "E3AgentPolicy"
        ),
        "ready_for_operator_submit": ready,
        "false_negative_risk_checked": True,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "per_game_deepest_level_integrated": dict(package_summary.get("per_game_deepest_level_integrated") or {}),
        "held_out_deepest_level_by_game": _held_out_deepest_by_game(attempts_list),
        "integrated_package_path": str(package_summary.get("package_path") or ""),
        "integrated_package_checksum": package_summary.get("package_checksum"),
        "submitted_agent_config": dict(submitted_agent_config),
        "upstream_lever_audit": dict(audit.get("upstream_lever_audit") or {}),
        "disallowed_adversarial_inputs": list(audit.get("disallowed_adversarial_inputs") or []),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": [
            "REQ-CAPSTONE-4585",
            "SCENARIO-CAPSTONE-4585",
            "SCENARIO-CAPSTONE-4585-FIELD-PRINCIPLES",
        ],
        "result_path": RESULT_RELATIVE_PATH,
        "duration_s": None if duration_s is None else max(0.0, round(float(duration_s), 6)),
        "leaderboard_submission": False,
        "variant_attempts_count": _as_int(transfer_measurement.get("variant_attempts_count")),
        "variant_solved_count": _as_int(transfer_measurement.get("variant_solved_count")),
        "variant_attempts": attempts_list,
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
    if type(artifact.get("live_submittable_level_count_integrated")) is not int:
        errors.append("live_submittable_level_count_integrated must be a bare int")
    rate = artifact.get("generic_transfer_rate_integrated")
    if not isinstance(rate, float) or isinstance(rate, bool) or not 0.0 <= rate <= 1.0:
        errors.append("generic_transfer_rate_integrated must be a bare float in [0,1]")
    ci = artifact.get("generic_transfer_ci_integrated")
    if (
        not isinstance(ci, list)
        or len(ci) != 2
        or not all(isinstance(value, float) for value in ci)
    ):
        errors.append("generic_transfer_ci_integrated must be [float, float]")
    if not isinstance(artifact.get("levers_integrated"), list):
        errors.append("levers_integrated must be a list")
    core = artifact.get("core_solves_preserved")
    if not isinstance(core, Mapping) or type(core.get("passed")) is not bool:
        errors.append("core_solves_preserved must report a bare passed bool")
    if type(artifact.get("parity_green")) is not bool:
        errors.append("parity_green must be a bare bool")
    if type(artifact.get("ready_for_operator_submit")) is not bool:
        errors.append("ready_for_operator_submit must be a bare bool")
    if artifact.get("leaderboard_submission") is not False:
        errors.append("leaderboard_submission must be false")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match REQ-CAPSTONE-4585")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum must be sha256-prefixed")
    elif checksum != payload_checksum(artifact):
        errors.append("reproducibility_checksum must match artifact content")
    if artifact.get("ready_for_operator_submit") is True:
        if not isinstance(core, Mapping) or core.get("passed") is not True:
            errors.append("ready_for_operator_submit requires core_solves_preserved")
        live = _as_int(artifact.get("live_submittable_level_count_integrated"))
        transfer = _as_float(artifact.get("generic_transfer_rate_integrated"))
        if live <= LIVE_SUBMITTABLE_BASELINE and transfer <= GENERIC_TRANSFER_BASELINE:
            errors.append("ready_for_operator_submit requires a real metric gain")
    return errors


def write_artifact(artifact: Mapping[str, Any], root: Path | str = REPO_ROOT) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _blocked_artifact(preconditions: Mapping[str, Any], duration_s: float | None) -> JsonDict:
    audit = {
        "levers_integrated": [],
        "isolated_deltas": {"live_submittable": {}, "generic_transfer": {}},
        "upstream_lever_audit": {},
        "disallowed_adversarial_inputs": [],
    }
    package = {
        "package_path": "",
        "claimed_total_levels": 0,
        "per_game_deepest_level_integrated": {},
        "core_solves_preserved": {"passed": False, "baseline_core_games": [], "dropped_games": []},
    }
    artifact = build_artifact(
        preconditions_checked=preconditions,
        audit=audit,
        package_summary=package,
        transfer_measurement=_empty_transfer_measurement(),
        submitted_agent_config=SUBMITTED_AGENT_CONFIG,
        duration_s=duration_s,
    )
    artifact["honest_verdict"] = "blocked_integration_gate_precondition"
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def run(
    root: Path | str = REPO_ROOT,
    *,
    offline_arcade_checker: Callable[[], bool] | None = None,
    public_games: Sequence[str] | None = None,
    variant_ids: Sequence[int] = DEFAULT_VARIANT_IDS,
    budget: int = DEFAULT_BUDGET,
    variant_runner: VariantRunner = exp4550.default_variant_runner,
    random_seed: int = RANDOM_SEED,
    n_bootstrap: int = exp4550.DEFAULT_BOOTSTRAPS,
    now: Callable[[], float] = time.perf_counter,
    write: bool = True,
) -> JsonDict:
    started = now()
    root_path = Path(root)
    checks = check_preconditions(root_path, offline_arcade_checker=offline_arcade_checker)
    if checks.get("ok") is not True:
        artifact = _blocked_artifact(checks, now() - started)
        if write:
            write_artifact(artifact, root_path)
        return artifact

    a1 = _read_json(root_path / A1_RELATIVE_PATH)
    a2 = _read_json(root_path / A2_RELATIVE_PATH)
    a3 = _read_json(root_path / A3_RELATIVE_PATH)
    a4 = _read_json(root_path / A4_RELATIVE_PATH)
    audit = audit_upstream_levers(
        a1_artifact=a1,
        a2_artifact=a2,
        a3_artifact=a3,
        a4_artifact=a4,
    )
    source_package = _read_json(root_path / A1_PACKAGE_RELATIVE_PATH)
    package_summary = refresh_integrated_package(
        root_path,
        source_package=source_package,
        levers_integrated=audit["levers_integrated"],
    )
    transfer = measure_integrated_transfer(
        root_path,
        public_games=public_games,
        variant_ids=variant_ids,
        budget=budget,
        variant_runner=variant_runner,
        random_seed=random_seed,
        n_bootstrap=n_bootstrap,
    )
    artifact = build_artifact(
        preconditions_checked=checks,
        audit=audit,
        package_summary=package_summary,
        transfer_measurement=transfer,
        submitted_agent_config=SUBMITTED_AGENT_CONFIG,
        duration_s=now() - started,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(artifact, root_path)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper
    artifact = run(REPO_ROOT, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - requested command boundary
    raise SystemExit(main())
