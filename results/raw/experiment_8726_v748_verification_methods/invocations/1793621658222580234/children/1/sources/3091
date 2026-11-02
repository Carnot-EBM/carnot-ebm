"""Qualify goal-safe HUD deduplication through the live ARC planner wrapper.

REQ-ARC-WMTE-7639 keeps both planner features disabled by default. This CPU
experiment measures only the current OFF path and the guarded HUD-dedup path.
It reuses Experiment 10013 replay helpers and makes no model call.
"""

from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import yaml

from carnot import experiment_10012_gate_usefulness as exp12
from carnot import experiment_10013_planner_dedup_tiebreak as exp13
from carnot.agentic import arc_executable_world_model as e3
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)

EXPERIMENT_ID = 7639
MILESTONE = "2026.09.666"
RUN_DATE = "20260925"
REQUIREMENT_ID = "REQ-ARC-WMTE-7639"
SCHEMA = "carnot.experiment_7639_v666_arc_goal_dedup.v1"
RANDOM_SEED = 7639
MODEL_SPECS: list[dict[str, Any]] = []

RESULT_REL = Path("results/experiment_7639_v666_arc_goal_dedup.json")
RAW_REL = Path("results/raw/experiment_7639_v666_arc_goal_dedup")
WINDOW_MANIFEST_REL = RAW_REL / "windows.json"
MODULE_REL = Path("python/carnot/experiment_7639_v666_arc_goal_dedup.py")
TEST_REL = Path("tests/python/test_experiment_7639_v666_arc_goal_dedup.py")
WRAPPER_REL = Path("scripts/experiments/experiment_7639_v666_arc_goal_dedup.py")
WORLD_MODEL_REL = Path("python/carnot/agentic/arc_executable_world_model.py")
AGENT_REL = Path("python/carnot/agentic/arc_competition_agent.py")
SPEC_REL = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
REGISTRY_REL = Path("ops/arc_solve_registry.yaml")
SOURCE_ARTIFACT_REL = Path("results/experiment_10013_planner_dedup_tiebreak.json")


@dataclass(frozen=True)
class MeasurementArm:
    """One pre-registered planner environment with novelty always absent."""

    name: str
    environment: Mapping[str, str]


MEASUREMENT_ARMS = (
    MeasurementArm("OFF", {}),
    MeasurementArm("HUD_DEDUP", {"CARNOT_ARC_PLAN_HUD_DEDUP": "1"}),
)
_PLANNER_ENV_KEYS = ("CARNOT_ARC_PLAN_HUD_DEDUP", "CARNOT_ARC_PLAN_GOAL_TIEBREAK")

FIELD_PRINCIPLES = {
    "honest_verdict": "Completion and benefit are separate claims.",
    "verdict_class": "A closed class prevents readiness from masquerading as benefit.",
    "flagged_adversarial": "A terminal reader result must remain attached to the evidence.",
    "gate_check_summary": "Blocked evidence names exact operands; completed evidence names failed gates.",
    "acceptance_gate_results": "Validity, readiness, benefit, utility, retention, and freshness stay separate.",
    "rows": "Absolute per-unit arm operands permit independent reduction.",
    "sample_size_budget": "Arms and seeds do not multiply independent windows.",
    "preconditions_checked": "Authenticated inputs prevent fabricated replacement work.",
    "inference_substrate": "Current CPU replay is distinct from historical model generation.",
    "inference_substrate_class": "No model load has no generation-duration floor.",
    "MODEL_SPECS": "An empty list declares that current work invoked no model.",
    "model_invoked": "Inherited generated engines do not count as current inference.",
    "execution_venue": "The closed venue enum identifies host orchestration.",
    "execution_host": "The actual hostname stays separate from the closed venue enum.",
    "execution_device_uuid": "Null records that no accelerator device was bound.",
    "execution_owned_pid": "The owned producer PID binds current host execution.",
    "phase_spans": "Disjoint measured stages expose completed and pending work.",
    "invocation_counts": "Current loads, forwards, generations, and tokens remain separate.",
    "duration_s": "Monotonic current duration is never inherited or padded.",
    "random_seed": "Each stochastic purpose has an explicit deterministic seed.",
    "reproducibility_checksum": "One digest binds configuration, evidence, and reduction.",
    "source_artifact_hashes": "Producers and planned outputs are never confused.",
    "validation_receipts": "Command exits and log hashes bind validation claims.",
    "verifier_is_oracle": "Exact fixtures cannot establish an oracle-distinct learned advantage.",
    "planner_goal_guard_ready_score": "One requires fail-first regressions, parity, and truthful telemetry.",
    "arc_window_manifest_path": "A frozen identity-only manifest prevents outcome-driven selection.",
    "solve_provenance": "Public scripted replay is a development proxy, not solve credit.",
    "mask_equivalence_limits": "Observed stability cannot prove unseen or hidden-state equivalence.",
    "production_defaults_changed": "Research arms do not authorize default promotion.",
}


def progress(started: float, phase: str, event: str, **detail: Any) -> None:
    """Emit a flushed phase boundary with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(detail.items()))
    print(
        f"[exp7639] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} {suffix}".rstrip(),
        flush=True,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _window_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only provenance fields that exist before either arm runs."""

    fields = (
        "pair_id",
        "game",
        "engine_family",
        "variant",
        "provenance_cohort",
        "source_path",
        "source_sha256",
        "source_status",
        "model_family",
        "think_mode",
        "token_budget",
        "is_control",
    )
    return {key: _jsonable(row.get(key)) for key in fields if key in row}


def build_window_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select exposed controls and attempts without reading arm outcomes."""

    identities: dict[str, dict[str, Any]] = {}
    for source in rows:
        identity = _window_identity(source)
        pair_id = str(identity.get("pair_id") or "")
        if pair_id and identity.get("provenance_cohort") == "stall_window":
            identities[pair_id] = identity
    census = sorted(
        identities.values(), key=lambda row: (str(row.get("game")), str(row["pair_id"]))
    )
    experts = [row for row in census if row.get("engine_family") == "EXPERT"]
    eligible = [row for row in census if row.get("engine_family") not in {"EXPERT", "IDENTITY"}]
    by_game: dict[str, list[dict[str, Any]]] = {}
    for row in eligible:
        by_game.setdefault(str(row.get("game")), []).append(row)
    enough = len(eligible) >= 20 and len(by_game) >= 8
    if enough:
        selected: list[dict[str, Any]] = []
        depth = 0
        while len(selected) < 20:
            added = False
            for game in sorted(by_game):
                game_rows = by_game[game]
                if depth < len(game_rows):
                    selected.append(game_rows[depth])
                    added = True
                    if len(selected) == 20:
                        break
            if not added:  # pragma: no cover - census size check makes this defensive.
                break
            depth += 1
    else:
        selected = eligible
    manifest: dict[str, Any] = {
        "schema": "carnot.exp7639.window_manifest.v1",
        "selection_rule": (
            "Sort stall-window identities by game and pair_id; retain every EXPERT control; "
            "select additional attempts round-robin by game to 20, or the entire census "
            "when either the count or eight-game breadth is unavailable."
        ),
        "selection_uses_arm_outcomes": False,
        "outcome_fields_forbidden": [
            "real_level_up",
            "plan_found",
            "planner_engine_calls",
            "planner_wall_s",
            "real_actions_used",
        ],
        "target_expert_controls": 10,
        "target_additional_windows": 20,
        "target_additional_games": 8,
        "eligible_additional_census": len(eligible),
        "eligible_additional_game_count": len(by_game),
        "exposed_expert_controls": experts,
        "additional_attempt_windows": selected,
        "additional_game_count": len({str(row.get("game")) for row in selected}),
        "breadth_sufficient": enough,
        "insufficient_breadth_reason": (
            None
            if enough
            else (
                f"eligible={len(eligible)} games={len(by_game)}; required at least 20 "
                "windows across 8 games, so the complete eligible census was retained"
            )
        ),
        "exp7640_selection_frozen": True,
    }
    manifest["manifest_checksum"] = canonical_hash(
        {key: value for key, value in manifest.items() if key != "manifest_checksum"}
    )
    return manifest


def synthetic_regression_row(unit: str, arm: str, *, passed: bool) -> dict[str, Any]:
    """Build one exact development-proxy regression row."""

    return {
        "unit_id": unit,
        "unit_kind": "exact_regression_fixture",
        "arm": arm,
        "absolute_metric": int(bool(passed)),
        "numerator": int(bool(passed)),
        "denominator": 1,
        "seed": RANDOM_SEED,
        "direction": "higher_is_better",
        "censored": False,
        "raw_provenance": "current_process_actual_plan_in_model",
        "passed": bool(passed),
    }


@contextlib.contextmanager
def arm_environment(arm: MeasurementArm) -> Iterator[None]:
    """Isolate both planner controls for one measurement arm."""

    prior = {key: os.environ.get(key) for key in _PLANNER_ENV_KEYS}
    for key in _PLANNER_ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update(arm.environment)
    try:
        yield
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def independent_reduce(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce paired absolute metrics without multiplying window count by arms."""

    materialized = [dict(row) for row in rows]
    units = {str(row.get("unit_id") or row.get("pair_id")) for row in materialized}
    by_arm: dict[str, dict[str, Any]] = {}
    for arm in (item.name for item in MEASUREMENT_ARMS):
        arm_rows = [row for row in materialized if row.get("arm") == arm]
        by_arm[arm] = {
            "row_count": len(arm_rows),
            "uncensored_count": sum(not bool(row.get("censored")) for row in arm_rows),
            "level_up_numerator": sum(bool(row.get("real_level_up")) for row in arm_rows),
            "level_up_denominator": sum(not bool(row.get("censored")) for row in arm_rows),
            "plan_found_numerator": sum(bool(row.get("plan_found")) for row in arm_rows),
            "planner_engine_calls": sum(
                int(row.get("planner_engine_calls") or 0) for row in arm_rows
            ),
            "real_actions_used": sum(int(row.get("real_actions_used") or 0) for row in arm_rows),
        }
    regressions = []
    indexed = {
        (str(row.get("unit_id") or row.get("pair_id")), str(row.get("arm"))): row
        for row in materialized
    }
    for unit in sorted(units):
        off = indexed.get((unit, "OFF"))
        hud = indexed.get((unit, "HUD_DEDUP"))
        if off and off.get("real_level_up") and (not hud or not hud.get("real_level_up")):
            regressions.append(unit)
    return {
        "independent_unit_count": len(units),
        "row_count": len(materialized),
        "by_arm": by_arm,
        "off_winner_regressions": regressions,
        "off_winner_retention_passed": not regressions,
        "censored_row_count": sum(bool(row.get("censored")) for row in materialized),
    }


def _gate(passed: bool, condition: str, principle: str, observed: Any) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "condition": condition,
        "principle": principle,
        "observed": _jsonable(observed),
    }


def _receipt_passed(receipt: Mapping[str, Any]) -> bool:
    return receipt.get("passed") is True or receipt.get("exit_code") == 0


def build_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    preconditions_checked: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, Any] | None = None,
    regression_evidence: Mapping[str, Any] | None = None,
    flagged_adversarial: bool = False,
) -> dict[str, Any]:
    """Build a terminal null: this task qualifies readiness, not hidden-game benefit."""

    materialized = [deepcopy(dict(row)) for row in rows]
    reduction = independent_reduce(materialized)
    regression = deepcopy(dict(regression_evidence or {}))
    exact_rows = [row for row in materialized if row.get("unit_kind") == "exact_regression_fixture"]
    fail_first_passed = bool(exact_rows) and all(row.get("passed") is True for row in exact_rows)
    flags_off_parity = regression.get("flags_off_parity", fail_first_passed) is True
    telemetry_truthful = regression.get("telemetry_truthful", fail_first_passed) is True
    validation_passed = all(_receipt_passed(row) for row in validation_receipts)
    if not validation_receipts:
        validation_passed = True
    preconditions_passed = all(
        row.get("passed", row.get("available")) is True for row in preconditions_checked
    )
    validity = bool(preconditions_passed and fail_first_passed and not flagged_adversarial)
    readiness = bool(validity and flags_off_parity and telemetry_truthful and validation_passed)
    retention = bool(manifest.get("manifest_checksum"))
    freshness = bool(source_artifact_hashes or not validation_receipts)
    probability_benefit = False
    utility = False
    gates = {
        "validity": _gate(
            validity,
            "all exact regressions and authenticated preconditions pass",
            "Invalid or adversarial evidence cannot open downstream gates.",
            {"fail_first_passed": fail_first_passed, "flagged_adversarial": flagged_adversarial},
        ),
        "readiness": _gate(
            readiness,
            "fail-first regressions, flags-off parity, telemetry, and validation pass",
            "A reusable guard is ready only when protected defaults and diagnostics remain exact.",
            {"flags_off_parity": flags_off_parity, "telemetry_truthful": telemetry_truthful},
        ),
        "probability_benefit": _gate(
            probability_benefit,
            "independent hidden-game evidence supports a probability benefit",
            "Development proxies and exposed controls cannot establish hidden-game probability lift.",
            "not_claimed",
        ),
        "utility": _gate(
            utility,
            "benefit exceeds measured planner and action cost",
            "Utility stays closed when probability benefit is not established.",
            "not_claimed",
        ),
        "retention": _gate(
            retention,
            "the deterministic Exp7640 manifest is content-addressed",
            "A later measurement must use the same independent windows.",
            manifest.get("manifest_checksum"),
        ),
        "freshness": _gate(
            freshness,
            "current source hashes and fresh command receipts are present",
            "Inherited evidence cannot substitute for current validation.",
            bool(source_artifact_hashes),
        ),
    }
    ready_score = int(readiness)
    verdict_class = "null" if readiness else "disqualified"
    verdict = (
        "complete_null_planner_goal_guard_ready_no_hidden_game_benefit_claim"
        if readiness
        else "complete_disqualified_planner_goal_guard_validation_failed"
    )
    invocation_counts = {
        **ZERO_INVOCATION_COUNTS,
        "forward_calls_attempted": 0,
        "forward_calls_completed": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "requirement_id": REQUIREMENT_ID,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": bool(flagged_adversarial),
        "gate_check_summary": {
            "operational_passed": readiness,
            "failed_operational_checks": [
                name
                for name in ("validity", "readiness", "retention", "freshness")
                if gates[name]["passed"] is not True
            ],
            "benefit_intentionally_unclaimed": True,
        },
        "acceptance_gate_results": gates,
        "rows": materialized,
        "independent_reduction": reduction,
        "sample_size_budget": {
            "intended_independent_units": 30,
            "observed_independent_units": reduction["independent_unit_count"],
            "excluded_independent_units": max(0, 30 - reduction["independent_unit_count"]),
            "censored_independent_units": len(
                {
                    str(row.get("unit_id") or row.get("pair_id"))
                    for row in materialized
                    if row.get("censored")
                }
            ),
            "repeated_arms_views_and_seeds_multiply_sample_size": False,
        },
        "preconditions_checked": deepcopy(list(preconditions_checked)),
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "historical_models": ["Qwen3.5-9B-MTP", "Qwen3.8-27B-GGUF"],
        "model_invoked": False,
        "execution_venue": "host",
        "execution_host": os.uname().nodename,
        "execution_device_uuid": None,
        "execution_owned_pid": os.getpid(),
        "phase_spans": deepcopy(list(phase_spans)),
        "invocation_counts": invocation_counts,
        "duration_s": round(float(duration_s), 6),
        "random_seed": {"window_selection": RANDOM_SEED, "fixture_order": RANDOM_SEED},
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes or {})),
        "validation_receipts": deepcopy(list(validation_receipts)),
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "planner_goal_guard_ready_score": ready_score,
        "arc_window_manifest_path": str(WINDOW_MANIFEST_REL),
        "window_manifest": deepcopy(dict(manifest)),
        "solve_provenance": "development_proxy",
        "new_game_level_solve_credit": False,
        "mask_equivalence_limits": [
            "The guard proves only that masked cells were stable in the active observed transitions.",
            "Unseen masked values and hidden state can still alias; the mask remains default off.",
            "A full-grid terminal check does not prove non-terminal transition equivalence.",
        ],
        "regression_evidence": regression,
        "production_defaults_changed": False,
        "hud_dedup_default_enabled": False,
        "novelty_tiebreak_default_enabled": False,
        "external_publication_authorized": False,
        "kaggle_submission_attempted": False,
        "generator_training_attempted": False,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = canonical_hash(artifact)
    return artifact


def _candidate(action: int = 1) -> dict[str, Any]:
    return {"action": action, "data": None}


def _transition(before: np.ndarray, after: np.ndarray) -> e3.Transition:
    return e3.Transition(before, 1, None, after, 0, 0)


def measure_regression_evidence() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the reviewed failures through the real planner and scored wrapper."""

    rows: list[dict[str, Any]] = []
    original_candidates = e3._model_candidates
    e3._model_candidates = lambda _grid: [_candidate()]  # type: ignore[assignment]
    try:
        terminal_root = np.zeros((1, 2), dtype=np.int16)

        def terminal_engine(_grid, _action, _data):
            return np.array([[0, 7]], dtype=np.int16)

        for arm in MEASUREMENT_ARMS:
            diagnostics: dict[str, Any] = {}
            plan = e3.plan_in_model(
                terminal_engine,
                lambda grid: int(grid[0, 1]) == 7,
                terminal_root,
                diagnostics=diagnostics,
                dedup_mask=(np.array([[False, True]]) if arm.name == "HUD_DEDUP" else None),
            )
            passed = (
                plan == [_candidate()] and diagnostics.get("termination_reason") == "plan_found"
            )
            row = synthetic_regression_row("terminal_duplicate", arm.name, passed=passed)
            row.update(
                {
                    "plan_found": bool(plan),
                    "plan_length": len(plan or []),
                    "planner_engine_calls": int(diagnostics.get("nodes_expanded") or 0),
                    "real_actions_used": 0,
                    "real_level_up": False,
                    "diagnostics": diagnostics,
                }
            )
            rows.append(row)

        counter_root = np.zeros((2, 3), dtype=np.int16)
        observed = counter_root.copy()
        observed[0, :2] = 4
        observed[1, 0] = 9
        observed[1, 2] = 1
        counter_mask = np.zeros_like(counter_root, dtype=bool)
        counter_mask[1, 2] = True

        def counter_engine(grid, _action, _data):
            out = np.asarray(grid).copy()
            out[1, 2] += 1
            return out

        for arm in MEASUREMENT_ARMS:
            policy = exp13.ScoredPlannerPolicyStub(
                counter_mask,
                cell=1,
                transitions=[_transition(counter_root, observed)],
            )
            diagnostics = {}
            with arm_environment(arm):
                plan = policy._call_plan_in_model(
                    e3.plan_in_model,
                    counter_engine,
                    lambda grid: int(grid[1, 2]) == 3,
                    counter_root,
                    diagnostics=diagnostics,
                    goal_energy_override=lambda _grid: 1.0,
                )
            expected_status = "refused" if arm.name == "HUD_DEDUP" else None
            passed = plan == [_candidate()] * 3 and (
                expected_status is None
                or diagnostics.get("planner_hud_dedup_mask_status") == expected_status
            )
            row = synthetic_regression_row("intermediate_counter", arm.name, passed=passed)
            row.update(
                {
                    "plan_found": bool(plan),
                    "plan_length": len(plan or []),
                    "planner_engine_calls": int(diagnostics.get("nodes_expanded") or 0),
                    "real_actions_used": 0,
                    "real_level_up": False,
                    "diagnostics": diagnostics,
                }
            )
            rows.append(row)

        stable_after = counter_root.copy()
        stable_after[0, 0] = 1
        stable_mask = np.zeros_like(counter_root, dtype=bool)
        stable_mask[1, :] = True
        policy = exp13.ScoredPlannerPolicyStub(
            stable_mask,
            cell=1,
            transitions=[_transition(counter_root, stable_after)],
        )
        telemetry: dict[str, Any] = {}
        with arm_environment(MEASUREMENT_ARMS[1]):
            e3._model_candidates = lambda _grid: []  # type: ignore[assignment]
            applied: dict[str, Any] = {}
            policy._call_plan_in_model(
                e3.plan_in_model,
                lambda grid, _action, _data: grid,
                lambda _grid: False,
                counter_root,
                diagnostics=applied,
                goal_energy_override=lambda _grid: 1.0,
            )
            telemetry["applied"] = applied.get("planner_hud_dedup_mask_status") == "applied"

            invalid = exp13.ScoredPlannerPolicyStub(stable_mask, cell=1, transitions=[])
            invalid_diag: dict[str, Any] = {}
            invalid._call_plan_in_model(
                e3.plan_in_model,
                lambda grid, _action, _data: grid,
                lambda _grid: False,
                np.zeros((1, 3), dtype=np.int16),
                diagnostics=invalid_diag,
                goal_energy_override=lambda _grid: 1.0,
            )
            telemetry["invalid_shape"] = (
                invalid_diag.get("planner_hud_dedup_planner_reason") == "shape_mismatch"
            )

            no_mask = exp13.ScoredPlannerPolicyStub(None, cell=1, transitions=[])
            no_mask_diag: dict[str, Any] = {}
            no_mask._call_plan_in_model(
                e3.plan_in_model,
                lambda grid, _action, _data: grid,
                lambda _grid: False,
                counter_root,
                diagnostics=no_mask_diag,
                goal_energy_override=lambda _grid: 1.0,
            )
            telemetry["no_mask"] = no_mask_diag.get("planner_hud_dedup_mask_status") == "unresolved"

            def old_planner(_engine, _goal, _grid):
                return []

            signature_diag: dict[str, Any] = {}
            policy._call_plan_in_model(
                old_planner,
                object(),
                lambda _grid: False,
                counter_root,
                diagnostics=signature_diag,
                goal_energy_override=lambda _grid: 1.0,
            )
            telemetry["callable_signature"] = (
                signature_diag.get("planner_hud_dedup_planner_reason")
                == "callable_signature_rejected"
            )

            def ignores_mask(_engine, _goal, _grid, **_kwargs):
                return []

            policy._call_plan_in_model(
                ignores_mask,
                object(),
                lambda _grid: False,
                counter_root,
                diagnostics=applied,
                goal_energy_override=lambda _grid: 1.0,
            )
            telemetry["restart"] = (
                applied.get("planner_hud_dedup_planner_reason") == "planner_did_not_report_use"
            )
        telemetry_truthful = all(telemetry.values())
        return rows, {
            "flags_off_parity": all(row["passed"] for row in rows if row.get("arm") == "OFF"),
            "telemetry_truthful": telemetry_truthful,
            "telemetry_cases": telemetry,
            "terminal_duplicate_passed": all(
                row["passed"] for row in rows if row.get("unit_id") == "terminal_duplicate"
            ),
            "intermediate_counter_passed": all(
                row["passed"] for row in rows if row.get("unit_id") == "intermediate_counter"
            ),
        }
    finally:
        e3._model_candidates = original_candidates  # type: ignore[assignment]


def _candidate_identity(candidate: exp12.Candidate) -> dict[str, Any]:
    return {
        "pair_id": candidate.pair_id,
        "game": candidate.game,
        "engine_family": candidate.engine_family,
        "variant": candidate.variant,
        "provenance_cohort": "stall_window",
        "source_path": candidate.source_path,
        "source_sha256": candidate.source_sha256,
        "source_status": candidate.source_status,
        "model_family": candidate.model_family,
        "think_mode": candidate.think_mode,
        "token_budget": candidate.token_budget,
        "is_control": candidate.is_control,
    }


def _registry_rows(root: Path) -> dict[str, dict[str, Any]]:  # pragma: no cover - integration
    payload = yaml.safe_load((root / REGISTRY_REL).read_text(encoding="utf-8"))
    return {str(row["game"]): dict(row) for row in payload.get("games", [])}


def run_selected_windows(
    root: Path,
    raw_root: Path,
    *,
    started: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:  # pragma: no cover - integration
    """Run the frozen identity-only census through the generic scored wrapper."""

    paths = exp12.exp10.EvidencePaths.under(root)
    reports = exp12.exp10.load_control_reports(paths)
    specs = {
        game: exp12.exp10.load_window(game, index, reports[game], paths)
        for index, game in enumerate(exp12.WINDOWS)
    }
    candidates = exp12.load_candidates(root, paths)
    manifest = build_window_manifest(
        [_candidate_identity(candidate) for game in exp12.WINDOWS for candidate in candidates[game]]
    )
    registry = _registry_rows(root)
    for group in ("exposed_expert_controls", "additional_attempt_windows"):
        for row in manifest[group]:
            registered = registry.get(str(row.get("game")), {})
            row["registry_precheck"] = {
                "present": bool(registered),
                "reproducibility": registered.get("reproducibility"),
                "levels_reproduced": registered.get("levels_reproduced"),
                "full_game_clear": registered.get("full_game_clear"),
                "new_solve_credit_allowed": False,
            }
    manifest["manifest_checksum"] = canonical_hash(
        {key: value for key, value in manifest.items() if key != "manifest_checksum"}
    )
    atomic_json(raw_root / "windows.json", manifest)
    selected = [*manifest["exposed_expert_controls"], *manifest["additional_attempt_windows"]]
    by_pair = {
        candidate.pair_id: candidate for game in exp12.WINDOWS for candidate in candidates[game]
    }
    rows: list[dict[str, Any]] = []
    unit_index = 0
    for game in sorted({str(row["game"]) for row in selected}):
        spec = specs[game]
        actions = exp12.load_episode_actions(root, game)
        alignment = exp12.induction_alignment(root, game, actions)
        frame_mask, replay_record = exp13.replay_main_window_live_mask(
            root,
            game,
            actions,
            int(alignment["enclosing_action_index"]),
        )
        preview, _ = exp12.rebuild_arm_start_state(
            root, game, spec, actions, alignment, exp12.LIVE_SCORED
        )
        window_record = exp13.live_mask_window_record(
            window_id=f"experiment_10010:{game}",
            cohort="stall_window",
            frame_mask=frame_mask,
            replay_record=replay_record,
            planner_frame=preview.frame,
            cell=spec.cell,
            transitions=spec.rows,
        )
        for identity in [row for row in selected if row["game"] == game]:
            candidate = by_pair[str(identity["pair_id"])]
            unit_index += 1
            for arm in MEASUREMENT_ARMS:
                progress(
                    started,
                    "measurement",
                    "before_benchmark",
                    arm=arm.name,
                    unit=f"{unit_index}/{len(selected)}",
                    pair=candidate.pair_id,
                )
                rebuilt, evidence = exp12.rebuild_arm_start_state(
                    root, game, spec, actions, alignment, exp12.LIVE_SCORED
                )
                if evidence.get("mode") != "root_after_reset":
                    from carnot.agentic.arc_agi3_world_model import grid_of

                    rebuilt = exp12.rebuild_reset_state_from_env(
                        env=rebuilt.env,
                        expected_level=rebuilt.level,
                        grid_reader=lambda frame, cell=spec.cell: e3.to_logical(
                            grid_of(frame), cell
                        ),
                        expected_grid=np.asarray(spec.rows[0].grid),
                    )
                    evidence = exp13._state_record(rebuilt)
                with arm_environment(arm):
                    execution = (
                        exp13.plan_and_execute(
                            candidate,
                            rebuilt,
                            arm=arm,  # type: ignore[arg-type]
                            cell=spec.cell,
                            frame_mask=frame_mask,
                            transitions=spec.rows,
                            window_mask_record=window_record,
                        )
                        if rebuilt.recoverable
                        else exp13._unrecoverable_execution(rebuilt.reason)
                    )
                row = exp13._pair_arm_row(
                    candidate,
                    arm,  # type: ignore[arg-type]
                    execution,
                    provenance_cohort="stall_window",
                    control_category="expert_live_planner",
                    state_rebuild=evidence,
                )
                row.update(
                    {
                        "unit_id": candidate.pair_id,
                        "unit_kind": "existing_attempt_window",
                        "absolute_metrics": {
                            "plan_found": bool(row.get("plan_found")),
                            "real_level_up": bool(row.get("real_level_up")),
                            "planner_engine_calls": int(row.get("planner_engine_calls") or 0),
                            "real_actions_used": int(row.get("real_actions_used") or 0),
                            "planner_wall_s": float(row.get("planner_wall_s") or 0.0),
                        },
                        "numerator": int(bool(row.get("real_level_up"))),
                        "denominator": int(rebuilt.recoverable),
                        "seed": RANDOM_SEED,
                        "direction": "more_level_ups_then_fewer_calls_and_actions",
                        "censored": not bool(rebuilt.recoverable),
                        "raw_provenance": {
                            "source_path": candidate.source_path,
                            "source_sha256": candidate.source_sha256,
                            "window": f"experiment_10010:{game}",
                        },
                        "registry_precheck": identity["registry_precheck"],
                        "solve_provenance": "development_proxy",
                    }
                )
                rows.append(row)
                atomic_json(raw_root / "rows" / f"{candidate.pair_id}__{arm.name}.json", row)
                progress(
                    started,
                    "measurement",
                    "after_benchmark",
                    arm=arm.name,
                    calls=row.get("planner_engine_calls"),
                    level_up=row.get("real_level_up"),
                    pair=candidate.pair_id,
                )
    return rows, manifest


_NAMED_INPUTS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    WORLD_MODEL_REL,
    AGENT_REL,
    Path("python/carnot/experiment_10013_planner_dedup_tiebreak.py"),
    Path("docs/research-notes/gate-usefulness-2026-09-24.md"),
    SOURCE_ARTIFACT_REL,
    REGISTRY_REL,
    SPEC_REL,
    MODULE_REL,
    TEST_REL,
    WRAPPER_REL,
)


def collect_preconditions(
    root: Path, private_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:  # pragma: no cover - integration
    """Authenticate named inputs and task-owned resources before measurement."""

    checks: list[dict[str, Any]] = []
    hashes: dict[str, Any] = {"authenticated_sources": {}, "missing_inputs": []}
    for relative in _NAMED_INPUTS:
        path = root / relative
        exists = path.is_file()
        owned = exists and path.stat().st_uid == os.getuid()
        passed = bool(exists and owned)
        checks.append(
            {
                "check": "named_input_authentication",
                "upstream": "worktree",
                "path": relative.as_posix(),
                "field": "is_file_and_current_user_owned",
                "operator": "==",
                "expected": True,
                "observed": passed,
                "passed": passed,
            }
        )
        if exists:
            hashes["authenticated_sources"][relative.as_posix()] = sha256_file(path)
        else:
            hashes["missing_inputs"].append(relative.as_posix())
    environment_files = exp12.resolve_environment_files(root)
    checks.append(
        {
            "check": "offline_environment_custody",
            "upstream": "ARC offline simulator",
            "path": str(environment_files),
            "field": "is_dir",
            "operator": "==",
            "expected": True,
            "observed": environment_files.is_dir(),
            "passed": environment_files.is_dir(),
        }
    )
    requirement_present = REQUIREMENT_ID in (root / SPEC_REL).read_text(encoding="utf-8")
    checks.append(
        {
            "check": "capability_requirement",
            "upstream": "OpenSpec",
            "path": SPEC_REL.as_posix(),
            "field": REQUIREMENT_ID,
            "operator": "contains",
            "expected": True,
            "observed": requirement_present,
            "passed": requirement_present,
        }
    )
    exclusion = yaml.safe_load((root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8"))
    retired_ids = {
        int(row["experiment_id"])
        for row in exclusion.get("retired", [])
        if str(row.get("experiment_id", "")).isdigit()
    }
    not_retired = EXPERIMENT_ID not in retired_ids
    checks.append(
        {
            "check": "exclusion_manifest",
            "upstream": "ops/exclusion_manifest.yaml",
            "path": "retired[].experiment_id",
            "field": str(EXPERIMENT_ID),
            "operator": "not in",
            "expected": True,
            "observed": not_retired,
            "passed": not_retired,
        }
    )
    registry = _registry_rows(root)
    registry_ready = all(
        game in registry and registry[game].get("reproducibility") == "reproduced"
        for game in exp12.WINDOWS
    )
    checks.append(
        {
            "check": "registry_precheck",
            "upstream": REGISTRY_REL.as_posix(),
            "path": "games",
            "field": "selected_targets_already_reproduced",
            "operator": "==",
            "expected": True,
            "observed": registry_ready,
            "passed": registry_ready,
        }
    )
    private = private_root.resolve()
    private_owned = private.is_relative_to(Path(tempfile.gettempdir()).resolve())
    checks.append(
        {
            "check": "resource_ownership",
            "upstream": "current_exp7639_process",
            "path": str(private),
            "field": "task_owned_private_root",
            "operator": "==",
            "expected": True,
            "observed": private_owned,
            "passed": private_owned,
        }
    )
    checks.append(
        {
            "check": "model_call_declaration",
            "upstream": "current_exp7639_task",
            "path": MODULE_REL.as_posix(),
            "field": "MODEL_SPECS",
            "operator": "==",
            "expected": [],
            "observed": MODEL_SPECS,
            "passed": MODEL_SPECS == [],
        }
    )
    hashes["planned_outputs_not_inputs"] = [str(RESULT_REL), str(WINDOW_MANIFEST_REL)]
    return checks, hashes


def build_blocked_artifact(
    failed: Mapping[str, Any],
    *,
    checks: Sequence[Mapping[str, Any]],
    duration_s: float,
    source_hashes: Mapping[str, Any],
) -> dict[str, Any]:  # pragma: no cover - external block
    manifest = build_window_manifest([])
    artifact = build_artifact(
        rows=[],
        manifest=manifest,
        preconditions_checked=checks,
        duration_s=duration_s,
        phase_spans=[],
        validation_receipts=[],
        source_artifact_hashes=source_hashes,
    )
    reason = str(failed.get("check") or "precondition").replace(" ", "_")
    artifact["honest_verdict"] = f"complete_blocked_{reason}"
    artifact["verdict_class"] = "blocked"
    artifact["planner_goal_guard_ready_score"] = 0
    artifact["gate_check_summary"] = {
        key: failed.get(key)
        for key in ("check", "upstream", "path", "field", "operator", "expected", "observed")
    }
    artifact["reproducibility_checksum"] = ""
    artifact["reproducibility_checksum"] = canonical_hash(artifact)
    return artifact


def affected_validation_manifest() -> dict[str, Any]:
    """Freeze exact tests, coverage module, and static paths before validation."""

    return {
        "schema": "carnot.exp7639.affected_validation_manifest.v1",
        "tests": [TEST_REL.as_posix()],
        "changed_modules": [MODULE_REL.as_posix()],
        "static_paths": [
            WORLD_MODEL_REL.as_posix(),
            AGENT_REL.as_posix(),
            WRAPPER_REL.as_posix(),
        ],
        "coverage_required_percent": 100,
        "serial": True,
        "repository_addopts_disabled": True,
    }


def build_validation_commands(
    root: Path, private_root: Path
) -> list[CommandSpec]:  # pragma: no cover - integration
    return build_scoped_commands(
        root,
        [TEST_REL.as_posix()],
        [MODULE_REL.as_posix()],
        static_paths=[WORLD_MODEL_REL.as_posix(), AGENT_REL.as_posix(), WRAPPER_REL.as_posix()],
        basetemp=private_root / "pytest",
        coverage_file=private_root / "coverage" / ".coverage",
    )


def build_e2e_commands(root: Path, private_root: Path) -> list[CommandSpec]:
    """Declare affected ARC E2Es plus the private no-induction E3 smoke."""

    pytest = str(root / ".venv/bin/pytest")
    python = str(root / ".venv/bin/python")
    common = ("-n", "0", "-o", "addopts=", "--no-cov")
    targets = {
        "e2e_009": ("tests/python/test_arc_induction_state_persistence.py",),
        "e2e_011": ("tests/python/test_arc_decision_telemetry.py",),
        "e2e_013": (
            "tests/python/test_arc_decision_telemetry.py",
            "tests/python/test_experiment_7491_e6_timed_live_profile.py",
            "tests/python/test_experiment_7531_b2_induction_gate_measurement.py",
            "tests/python/test_semif_arc_readout_eval.py",
        ),
    }
    commands = [
        CommandSpec(
            name,
            (pytest, *common, f"--basetemp={private_root / name}", *paths, "-q"),
            name.upper().replace("_", "-"),
            900.0,
        )
        for name, paths in targets.items()
    ]
    foreign = private_root / "foreign-cwd"
    foreign.mkdir(parents=True, exist_ok=True)
    commands.append(
        CommandSpec(
            "no_induction_e3_cpu_smoke",
            (
                "/usr/bin/env",
                "-C",
                str(foreign),
                "CARNOT_ARC_DISABLE_INDUCTION=1",
                "JAX_PLATFORMS=cpu",
                python,
                "-u",
                str(root / "scripts/arc_loop_solve.py"),
                "--mechanism",
                "e3",
                "--game",
                "r11l",
                "--max-actions",
                "12",
                "--output",
                str(foreign / "r11l-smoke.json"),
            ),
            "private no-induction scored E3 CPU smoke",
            300.0,
        )
    )
    return commands


def validate_artifact(value: Mapping[str, Any]) -> list[str]:
    """Validate terminal shape without trusting producer summaries."""

    errors: list[str] = []
    required = {
        "honest_verdict",
        "verdict_class",
        "flagged_adversarial",
        "acceptance_gate_results",
        "rows",
        "sample_size_budget",
        "preconditions_checked",
        "inference_substrate",
        "inference_substrate_class",
        "MODEL_SPECS",
        "phase_spans",
        "invocation_counts",
        "reproducibility_checksum",
        "source_artifact_hashes",
        "validation_receipts",
        "field_principles",
        "planner_goal_guard_ready_score",
        "arc_window_manifest_path",
        "mask_equivalence_limits",
        "production_defaults_changed",
    }
    errors.extend(sorted(required - set(value)))
    if not str(value.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict_terminal_prefix")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class")
    if value.get("MODEL_SPECS") != [] or value.get("model_invoked") is not False:
        errors.append("no_model_load_declaration")
    if (
        value.get("execution_venue") != "host"
        or not value.get("execution_host")
        or value.get("execution_device_uuid") is not None
        or not isinstance(value.get("execution_owned_pid"), int)
        or value.get("execution_owned_pid", 0) <= 0
    ):
        errors.append("execution_identity")
    arms = {str(row.get("arm")) for row in value.get("rows", [])}
    if arms and not arms <= {"OFF", "HUD_DEDUP"}:
        errors.append("measurement_arms")
    copied = deepcopy(dict(value))
    observed = copied.pop("reproducibility_checksum", None)
    copied["reproducibility_checksum"] = ""
    if observed != canonical_hash(copied):
        errors.append("reproducibility_checksum")
    return errors


def cold_replay(path: Path) -> list[str]:
    """Reload exact bytes and recompute schema plus checksum in a fresh process."""

    return validate_artifact(json.loads(path.read_text(encoding="utf-8")))


def independent_replay(path: Path) -> list[str]:
    """Recompute arm totals from raw rows without producer summaries."""

    value = json.loads(path.read_text(encoding="utf-8"))
    observed = independent_reduce(value.get("rows", []))
    return [] if observed == value.get("independent_reduction") else ["independent_reduction"]


def build_terminal_commands(root: Path, candidate: Path) -> list[CommandSpec]:
    """Declare bounded fresh readers for the exact private terminal candidate."""

    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER_REL)
    common = ("--date", RUN_DATE)
    return [
        CommandSpec(
            "declared_entrypoint_validate",
            (python, "-u", wrapper, *common, "--validate", str(candidate)),
            "exact terminal candidate",
            300.0,
        ),
        CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", wrapper, *common, "--cold-replay", str(candidate)),
            "exact terminal candidate",
            300.0,
        ),
        CommandSpec(
            "independent_reduction",
            (python, "-u", wrapper, *common, "--independent-reduce", str(candidate)),
            "exact terminal candidate",
            300.0,
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", str(root / "scripts/adversarial_verify.py"), str(candidate)),
            "exact terminal candidate",
            300.0,
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                str(root / "scripts/verdict_row_consistency_lint.py"),
                "--strict",
                str(candidate),
            ),
            "exact terminal candidate",
            300.0,
        ),
    ]


def _record_phase(
    spans: list[dict[str, Any]], name: str, phase_started: float, task_started: float, **extra: Any
) -> None:  # pragma: no cover - integration
    ended = time.monotonic()
    spans.append(
        {
            "name": name,
            "started_elapsed_s": round(phase_started - task_started, 6),
            "ended_elapsed_s": round(ended - task_started, 6),
            "duration_s": round(ended - phase_started, 6),
            "completed_units": int(extra.pop("completed_units", 1)),
            "pending_operations": list(extra.pop("pending_operations", [])),
            "checkpoint_position": extra.pop("checkpoint_position", name),
            **extra,
        }
    )


def _all_passed(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(_receipt_passed(row) for row in rows)


def _copy_receipts_after_exit(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    raw_root: Path,
    group: str,
) -> list[dict[str, Any]]:  # pragma: no cover - integration
    copied: list[dict[str, Any]] = []
    for index, source_row in enumerate(rows):
        row = deepcopy(dict(source_row))
        source = Path(str(row["log_path"]))
        if not source.is_absolute():
            source = root / source
        destination = raw_root / "validation" / group / f"{index:02d}_{row['name']}.log"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        row["private_log_path"] = str(source)
        row["private_log_sha256"] = sha256_file(source)
        row["log_path"] = destination.relative_to(root).as_posix()
        row["log_sha256"] = sha256_file(destination)
        row["copied_after_process_exit"] = True
        copied.append(row)
    return copied


def run_experiment(
    repo_root: Path,
    run_date: str,
    output_path: Path,
) -> dict[str, Any]:  # pragma: no cover - declared integration entrypoint
    """Execute the CPU-only measurement, validation, readers, and publication."""

    started = time.monotonic()
    root = repo_root.resolve()
    expected_root = Path(__file__).resolve().parents[2]
    if root != expected_root:
        raise ValueError(f"root_mismatch:{root}:{expected_root}")
    if run_date != RUN_DATE:
        raise ValueError(f"run_date_mismatch:{run_date}:{RUN_DATE}")
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7639-v666-")).resolve()
    spans: list[dict[str, Any]] = []
    progress(started, "startup", "begin", root=root, private_root=private_root)

    phase_started = time.monotonic()
    progress(started, "preconditions", "before")
    checks, source_hashes = collect_preconditions(root, private_root)
    failed = [row for row in checks if row.get("passed") is not True]
    progress(started, "preconditions", "after", failed=len(failed), passed=not failed)
    _record_phase(spans, "preconditions", phase_started, started, completed_units=len(checks))
    if failed:
        artifact = build_blocked_artifact(
            failed[0],
            checks=checks,
            duration_s=time.monotonic() - started,
            source_hashes=source_hashes,
        )
        progress(started, "publish", "before_atomic_blocked", output=output_path)
        atomic_json(root / output_path, artifact)
        progress(started, "publish", "after_atomic_blocked", output=output_path)
        return artifact

    raw_root = root / RAW_REL
    raw_root.mkdir(parents=True, exist_ok=True)
    atomic_json(raw_root / "affected_validation_manifest.json", affected_validation_manifest())

    phase_started = time.monotonic()
    progress(started, "regression_fixtures", "before_benchmark")
    regression_rows, regression = measure_regression_evidence()
    progress(
        started,
        "regression_fixtures",
        "after_benchmark",
        passed=all(row["passed"] for row in regression_rows),
        units=len(regression_rows),
    )
    _record_phase(
        spans,
        "regression_fixtures",
        phase_started,
        started,
        completed_units=len(regression_rows),
    )

    phase_started = time.monotonic()
    progress(started, "measurement", "before_benchmark")
    measurement_rows, manifest = run_selected_windows(root, raw_root, started=started)
    progress(started, "measurement", "after_benchmark", rows=len(measurement_rows))
    _record_phase(
        spans,
        "measurement",
        phase_started,
        started,
        completed_units=len(measurement_rows),
    )
    rows = [*regression_rows, *measurement_rows]

    phase_started = time.monotonic()
    progress(started, "scoped_validation", "before_subprocess")
    (private_root / "coverage").mkdir(parents=True, exist_ok=True)
    (private_root / "pytest").mkdir(parents=True, exist_ok=True)
    validation_private = run_commands(
        root,
        build_validation_commands(root, private_root),
        log_dir=private_root / "logs" / "affected",
        extra_env={
            "JAX_PLATFORMS": "cpu",
            "COVERAGE_FILE": str(private_root / "coverage/.coverage"),
        },
        heartbeat_s=60.0,
    )
    progress(
        started,
        "scoped_validation",
        "after_subprocess",
        passed=_all_passed(validation_private),
        units=len(validation_private),
    )
    _record_phase(
        spans,
        "scoped_validation",
        phase_started,
        started,
        completed_units=len(validation_private),
    )

    phase_started = time.monotonic()
    progress(started, "arc_e2e", "before_subprocess")
    e2e_private = run_commands(
        root,
        build_e2e_commands(root, private_root),
        log_dir=private_root / "logs" / "e2e",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=60.0,
    )
    progress(
        started,
        "arc_e2e",
        "after_subprocess",
        passed=_all_passed(e2e_private),
        units=len(e2e_private),
    )
    _record_phase(
        spans,
        "arc_e2e",
        phase_started,
        started,
        completed_units=len(e2e_private),
    )

    validation = _copy_receipts_after_exit(root, validation_private, raw_root, "affected")
    e2e = _copy_receipts_after_exit(root, e2e_private, raw_root, "e2e")
    source_hashes["current_producers"] = {
        path.as_posix(): sha256_file(root / path)
        for path in (MODULE_REL, TEST_REL, WRAPPER_REL, WORLD_MODEL_REL, AGENT_REL, SPEC_REL)
    }
    candidate_path = private_root / "terminal_candidate.json"
    candidate = build_artifact(
        rows=rows,
        manifest=manifest,
        preconditions_checked=checks,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        validation_receipts=[*validation, *e2e],
        source_artifact_hashes=source_hashes,
        regression_evidence=regression,
    )
    errors = validate_artifact(candidate)
    if errors:
        raise ValueError("terminal_candidate_invalid:" + ",".join(errors))
    atomic_json(candidate_path, candidate)

    phase_started = time.monotonic()
    progress(
        started,
        "terminal_readers",
        "before_subprocess",
        candidate_sha256=sha256_file(candidate_path),
    )
    terminal_private = run_commands(
        root,
        build_terminal_commands(root, candidate_path),
        log_dir=private_root / "logs" / "terminal",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=60.0,
    )
    progress(
        started,
        "terminal_readers",
        "after_subprocess",
        passed=_all_passed(terminal_private),
        units=len(terminal_private),
    )
    _record_phase(
        spans,
        "terminal_readers",
        phase_started,
        started,
        completed_units=len(terminal_private),
    )
    terminal = _copy_receipts_after_exit(root, terminal_private, raw_root, "terminal")
    adversarial = next((row for row in terminal if row.get("name") == "adversarial_verify"), {})
    flagged = adversarial.get("passed") is not True
    final = build_artifact(
        rows=rows,
        manifest=manifest,
        preconditions_checked=checks,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        validation_receipts=[*validation, *e2e, *terminal],
        source_artifact_hashes=source_hashes,
        regression_evidence=regression,
        flagged_adversarial=flagged,
    )
    final_errors = validate_artifact(final)
    if final_errors:
        raise ValueError("terminal_artifact_invalid:" + ",".join(final_errors))
    progress(started, "publish", "before_atomic", output=output_path)
    atomic_json(raw_root / "terminal_validation_receipts.json", {"receipts": terminal})
    atomic_json(root / output_path, final)
    progress(started, "publish", "after_atomic", verdict=final["honest_verdict"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_REL)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI
    args = parse_args(argv)
    if args.validate is not None:
        errors = validate_artifact(json.loads(args.validate.read_text(encoding="utf-8")))
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.cold_replay is not None:
        errors = cold_replay(args.cold_replay)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        errors = independent_replay(args.independent_reduce)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    output = args.output if args.output.is_absolute() else args.repo_root / args.output
    run_experiment(args.repo_root, args.date, output.relative_to(args.repo_root))
    return 0
