"""Measure the V667 ARC scored wrapper on the complete frozen source census.

REQ-REPORT-7652. This is CPU development-proxy work; inherited engines are
replayed, no current model is loaded, and production flags stay unchanged.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from carnot import experiment_10012_gate_usefulness as exp12
from carnot import experiment_10013_planner_dedup_tiebreak as exp13
from carnot import experiment_7639_v666_arc_goal_dedup as exp39
from carnot.agentic import arc_executable_world_model as e3
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)

ROOT = Path(__file__).resolve().parents[2]
RESULT = Path("results/experiment_7652_v667_arc_wrapper_measurement.json")
RAW = Path("results/raw/experiment_7652_v667_arc_wrapper_measurement")
SOURCE = Path("results/experiment_10013_planner_dedup_tiebreak.json")
GUARD = Path("results/experiment_7645_v667_arc_validation_requalification.json")
MODULE = Path("python/carnot/experiment_7652_v667_arc_wrapper_measurement.py")
TEST = Path("tests/python/test_experiment_7652_v667_arc_wrapper_measurement.py")
WRAPPER = Path("scripts/experiments/experiment_7652_v667_arc_wrapper_measurement.py")
SPEC = Path("openspec/capabilities/research-reporting/spec.md")
REGISTRY = Path("ops/arc_solve_registry.yaml")
RUN_DATE = "20260925"
ARMS = exp39.MEASUREMENT_ARMS
MODEL_SPECS: list[dict[str, Any]] = []
INDUCED = {"THINK", "CODEONLY", "QWEN38"}


def progress(start: float, phase: str, event: str, **details: Any) -> None:
    """Print a flushed, monotonic boundary before and after long work."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7652] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} {suffix}",
        flush=True,
    )


def freeze_census(source_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """SCENARIO-REPORT-7652-CENSUS copies only pre-arm identity fields."""

    names = (
        "pair_id",
        "game",
        "engine_family",
        "variant",
        "source_sha256",
        "source_path",
        "source_status",
        "model_family",
        "think_mode",
        "token_budget",
        "is_control",
    )
    groups: dict[str, dict[str, dict[str, Any]]] = {
        "induced_engine_windows": {},
        "expert_controls": {},
        "identity_controls": {},
    }
    for row in source_rows:
        if row.get("provenance_cohort") != "stall_window":
            continue
        family = str(row.get("engine_family"))
        group = (
            "expert_controls"
            if family == "EXPERT"
            else "identity_controls"
            if family == "IDENTITY"
            else "induced_engine_windows"
            if family in INDUCED
            else None
        )
        if group is None:
            continue
        identity = {key: row.get(key) for key in names}
        pair = str(identity["pair_id"])
        if pair in groups[group] and groups[group][pair] != identity:
            raise ValueError(f"source_identity_conflict:{pair}")
        groups[group][pair] = identity
    manifest = {
        key: sorted(value.values(), key=lambda row: (str(row["game"]), str(row["pair_id"])))
        for key, value in groups.items()
    }
    manifest["selection_rule"] = (
        "all unique Exp10013 stall-window source identities; no arm outcome field"
    )
    manifest["selection_uses_arm_outcomes"] = False
    manifest["manifest_checksum"] = canonical_hash(manifest)
    return manifest


def check_goal_guard(guard: Mapping[str, Any], path: str) -> dict[str, Any] | None:
    """SCENARIO-REPORT-7652-GUARD names the first exact failed operand."""

    expected = {
        "honest_verdict": "complete_null_arc_goal_guard_ready_no_hidden_game_benefit",
        "planner_goal_guard_ready_score": 1,
        "flagged_adversarial": False,
        "verdict_class": "null",
    }
    for field, wanted in expected.items():
        observed = guard.get(field)
        if observed != wanted:
            return {
                "check": "exp7645_current_goal_guard",
                "upstream": "Experiment 7645",
                "path": path,
                "field": field,
                "operator": "==",
                "expected": wanted,
                "observed": observed,
            }
    return None


def validate_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """SCENARIO-REPORT-7652-REPLAY rejects unsafe applied-mask telemetry."""

    errors: list[str] = []
    for row in rows:
        mask = row.get("mask") or {}
        diagnostics = row.get("planner_diagnostics") or {}
        if mask.get("status") == "applied" and (
            mask.get("planner_reason") == "shape_mismatch"
            or mask.get("resolution_reason") == "shape_mismatch"
        ):
            errors.append("mask_mismatch_applied")
        if (
            row.get("real_level_up")
            or row.get("plan_found")
            or (mask.get("status") == "applied" and int(row.get("planner_engine_calls") or 0) > 0)
        ) and int(diagnostics.get("goal_evaluations") or 0) < 1:
            errors.append("goal_check_skipped")
    return sorted(set(errors))


def reduce_pairs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce one independent window once, with uncertainty clustered by game."""

    paired: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    controls = {"EXPERT": 0, "IDENTITY": 0}
    totals = {
        arm.name: {
            "successes": 0,
            "engine_calls": 0,
            "real_actions": 0,
            "wall_s": 0.0,
            "goal_checks": 0,
            "duplicate_skips": 0,
            "plan_length": 0,
        }
        for arm in ARMS
    }
    for row in rows:
        family = str(row.get("engine_family"))
        if family in controls:
            controls[family] += 1
            continue
        if family not in INDUCED:
            continue
        unit, arm = str(row.get("unit_id")), str(row.get("arm"))
        if arm in paired[unit]:
            raise ValueError(f"duplicate_arm:{unit}:{arm}")
        paired[unit][arm] = row
        if arm in totals:
            total = totals[arm]
            total["successes"] += int(bool(row.get("real_level_up")))
            total["engine_calls"] += int(row.get("planner_engine_calls") or 0)
            total["real_actions"] += int(row.get("real_actions_used") or 0)
            total["wall_s"] += float(row.get("planner_wall_s") or 0)
            total["goal_checks"] += int(
                (row.get("planner_diagnostics") or {}).get("goal_evaluations") or 0
            )
            total["duplicate_skips"] += int(
                (row.get("planner_diagnostics") or {}).get("hud_dedup_states_merged") or 0
            )
            total["plan_length"] += int(row.get("plan_length") or 0)
    gained = lost = usable = 0
    by_game: dict[str, list[int]] = defaultdict(list)
    for unit, arms in paired.items():
        if set(arms) != {"OFF", "HUD_DEDUP"}:
            raise ValueError(f"unpaired_window:{unit}")
        off, hud = arms["OFF"], arms["HUD_DEDUP"]
        if off.get("censored") or hud.get("censored"):
            continue
        usable += 1
        difference = int(bool(hud.get("real_level_up"))) - int(bool(off.get("real_level_up")))
        gained += int(difference > 0)
        lost += int(difference < 0)
        by_game[str(off.get("game"))].append(difference)
    game_values = [sum(values) / len(values) for _, values in sorted(by_game.items())]
    if game_values:
        rng = random.Random(7652)
        draws = sorted(
            sum(rng.choice(game_values) for _ in game_values) / len(game_values)
            for _ in range(2000)
        )
        ci95 = [draws[49], draws[1949]]
    else:
        ci95 = [0.0, 0.0]
    return {
        "induced_windows": len(paired),
        "usable_induced_windows": usable,
        "induced_new_successes": gained,
        "induced_lost_successes": lost,
        "development_proxy_improvement": gained >= 1 and lost == 0,
        "game_clusters": len(game_values),
        "ci95_game_cluster": ci95,
        "by_arm": totals,
        "expert_control_rows": controls["EXPERT"],
        "identity_control_rows": controls["IDENTITY"],
        "per_game_delta": {
            game: sum(values) / len(values) for game, values in sorted(by_game.items())
        },
    }


def _check(
    name: str, upstream: str, path: Path, field: str, observed: Any, expected: Any = True
) -> dict[str, Any]:
    return {
        "check": name,
        "upstream": upstream,
        "path": str(path),
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def collect_preconditions(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Authenticate actual inputs and resources, never the planned output."""

    required = [
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("scripts/experiment_template.py"),
        Path("python/carnot/reporting/current_work_receipt.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("python/carnot/experiment_7639_v666_arc_goal_dedup.py"),
        Path("python/carnot/experiment_10013_planner_dedup_tiebreak.py"),
        Path("python/carnot/agentic/arc_competition_agent.py"),
        Path("python/carnot/agentic/arc_solver_kit.py"),
        SPEC,
        REGISTRY,
        SOURCE,
        GUARD,
        MODULE,
        TEST,
        WRAPPER,
    ]
    checks = [
        _check(
            "named_input_authentication",
            "worktree",
            relative,
            "is_file_and_current_user_owned",
            (root / relative).is_file() and (root / relative).stat().st_uid == os.getuid()
            if (root / relative).is_file()
            else False,
        )
        for relative in required
    ]
    hashes = {
        "producer_files": {
            str(path): sha256_file(root / path) for path in required if (root / path).is_file()
        },
        "pre_gate_receipts": {},
        "missing_inputs": [str(path) for path in required if not (root / path).is_file()],
        "planned_outputs_not_inputs": [str(RESULT)],
    }
    if (root / GUARD).is_file():
        guard = json.loads((root / GUARD).read_text())
        failed = check_goal_guard(guard, str(GUARD))
        checks.append(
            {
                **(
                    failed
                    or _check(
                        "exp7645_current_goal_guard",
                        "Experiment 7645",
                        GUARD,
                        "planner_goal_guard_ready_score",
                        1,
                        1,
                    )
                ),
                "passed": failed is None,
            }
        )
        hashes["pre_gate_receipts"][str(GUARD)] = sha256_file(root / GUARD)
    env_dir = exp12.resolve_environment_files(root)
    checks.append(
        _check(
            "offline_environment_custody",
            "ARC offline simulator",
            env_dir,
            "is_dir",
            env_dir.is_dir(),
        )
    )
    checks.append(
        _check(
            "engine_call_budget",
            "LIVE_SCORED",
            Path("python/carnot/experiment_10012_gate_usefulness.py"),
            "max_nodes",
            exp12.LIVE_SCORED.max_nodes,
            20000,
        )
    )
    checks.append(
        _check(
            "novelty_tiebreak_default",
            "production planner",
            Path("python/carnot/agentic/arc_executable_world_model.py"),
            "CARNOT_ARC_PLAN_GOAL_TIEBREAK",
            os.environ.get("CARNOT_ARC_PLAN_GOAL_TIEBREAK") is None,
        )
    )
    registry = yaml.safe_load((root / REGISTRY).read_text()) if (root / REGISTRY).is_file() else {}
    reproduced = {
        str(row.get("game"))
        for row in registry.get("games", [])
        if row.get("reproducibility") == "reproduced"
    }
    checks.append(
        _check(
            "registry_precheck",
            "ARC solve registry",
            REGISTRY,
            "selected_targets_reproduced",
            set(exp12.WINDOWS) <= reproduced,
        )
    )
    exclusion = (
        yaml.safe_load((root / "ops/exclusion_manifest.yaml").read_text())
        if (root / "ops/exclusion_manifest.yaml").is_file()
        else {}
    )
    retired = {str(row.get("experiment_id")) for row in exclusion.get("retired", [])}
    checks.append(
        _check(
            "exclusion_manifest",
            "ops/exclusion_manifest.yaml",
            Path("ops/exclusion_manifest.yaml"),
            "7652_not_retired",
            "7652" not in retired,
        )
    )
    return checks, hashes


def prepare_windows(
    root: Path,
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]
]:  # pragma: no cover - live integration
    """Freeze all source identities and authenticate current candidate bytes."""

    source = json.loads((root / SOURCE).read_text())
    manifest = freeze_census(source["per_pair_arm_rows"])
    paths = exp12.exp10.EvidencePaths.under(root)
    reports = exp12.exp10.load_control_reports(paths)
    specs = {
        game: exp12.exp10.load_window(game, index, reports[game], paths)
        for index, game in enumerate(exp12.WINDOWS)
    }
    candidate_sets = exp12.load_candidates(root, paths)
    candidates = {
        candidate.pair_id: candidate for group in candidate_sets.values() for candidate in group
    }
    failures = []
    for group in ("induced_engine_windows", "expert_controls", "identity_controls"):
        for identity in manifest[group]:
            pair = identity["pair_id"]
            candidate = candidates.get(pair)
            actual = None if candidate is None else candidate.source_sha256
            wanted = identity["source_sha256"]
            if candidate is None or actual != wanted:
                failures.append(
                    _check(
                        "source_window_authentication",
                        "Exp10013 identity and current source",
                        Path(str(identity["source_path"])),
                        f"{pair}.source_sha256",
                        actual,
                        wanted,
                    )
                )
    return manifest, {"specs": specs, "candidates": candidates}, {"paths": paths}, failures


def measure_windows(
    root: Path, raw: Path, manifest: Mapping[str, Any], prepared: Mapping[str, Any], started: float
) -> list[dict[str, Any]]:  # pragma: no cover - live integration
    """Replay one identical scored start per pair and arm, checkpointing rows."""

    specs, candidates = prepared["specs"], prepared["candidates"]
    registry = exp39._registry_rows(root)
    selected = [
        *manifest["induced_engine_windows"],
        *manifest["expert_controls"],
        *manifest["identity_controls"],
    ]
    rows: list[dict[str, Any]] = []
    for index, game in enumerate(sorted({str(item["game"]) for item in selected}), 1):
        progress(started, "game", "before_benchmark", game=game, completed=f"{index - 1}/10")
        spec = specs[game]
        actions = exp12.load_episode_actions(root, game)
        alignment = exp12.induction_alignment(root, game, actions)
        frame_mask, replay = exp13.replay_main_window_live_mask(
            root, game, actions, int(alignment["enclosing_action_index"])
        )
        preview, _ = exp12.rebuild_arm_start_state(
            root, game, spec, actions, alignment, exp12.LIVE_SCORED
        )
        mask_record = exp13.live_mask_window_record(
            window_id=f"experiment_10010:{game}",
            cohort="stall_window",
            frame_mask=frame_mask,
            replay_record=replay,
            planner_frame=preview.frame,
            cell=spec.cell,
            transitions=spec.rows,
        )
        for identity in [item for item in selected if item["game"] == game]:
            candidate = candidates[str(identity["pair_id"])]
            for arm in ARMS:
                progress(
                    started,
                    "measurement",
                    "before_benchmark",
                    game=game,
                    pair=candidate.pair_id,
                    arm=arm.name,
                    completed=len(rows),
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
                with exp39.arm_environment(arm):
                    execution = (
                        exp13.plan_and_execute(
                            candidate,
                            rebuilt,
                            arm=arm,
                            cell=spec.cell,
                            frame_mask=frame_mask,
                            transitions=spec.rows,
                            window_mask_record=mask_record,
                        )
                        if rebuilt.recoverable
                        else exp13._unrecoverable_execution(rebuilt.reason)
                    )
                row = exp13._pair_arm_row(
                    candidate,
                    arm,
                    execution,
                    provenance_cohort="stall_window",
                    control_category="expert_live_planner"
                    if candidate.is_control
                    else "induced_engine",
                    state_rebuild=evidence,
                )
                usable = bool(rebuilt.recoverable and candidate.source is not None)
                row.update(
                    {
                        "unit_id": candidate.pair_id,
                        "unit_kind": "existing_attempt_window",
                        "numerator": int(bool(row.get("real_level_up"))),
                        "denominator": int(usable),
                        "censored": not usable,
                        "exclusion": None
                        if usable
                        else (
                            "source_absent_in_frozen_census"
                            if candidate.source is None
                            else rebuilt.reason
                        ),
                        "raw_provenance": {
                            "source_path": candidate.source_path,
                            "source_sha256": candidate.source_sha256,
                            "source_artifact": str(SOURCE),
                            "start_sha256": rebuilt.expected_sha256,
                        },
                        "registry_precheck": {
                            "reproducibility": registry[game].get("reproducibility"),
                            "new_solve_credit_allowed": False,
                        },
                        "solve_provenance": "development_proxy",
                        "per_game_adapter_routes_enabled": False,
                        "seed": 7652,
                    }
                )
                rows.append(row)
                atomic_json(raw / "rows" / f"{candidate.pair_id}__{arm.name}.json", row)
                progress(
                    started,
                    "measurement",
                    "after_benchmark",
                    game=game,
                    pair=candidate.pair_id,
                    arm=arm.name,
                    calls=row.get("planner_engine_calls"),
                    success=row.get("real_level_up"),
                    completed=len(rows),
                )
        progress(started, "game", "after_benchmark", game=game, completed=f"{index}/10")
    return rows


def _gate(passed: bool, principle: str, operands: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "principle": principle, "measured_operands": operands}


def build_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    spans: Sequence[Mapping[str, Any]],
    duration: float,
    flagged: bool = False,
) -> dict[str, Any]:  # pragma: no cover - live integration
    """Build a terminal record from absolute operands, not inherited verdicts."""

    raw_rows = deepcopy(list(rows))
    errors = validate_rows(raw_rows)
    reduction = (
        reduce_pairs(raw_rows)
        if raw_rows
        else {
            "induced_windows": 0,
            "usable_induced_windows": 0,
            "induced_new_successes": 0,
            "induced_lost_successes": 0,
            "development_proxy_improvement": False,
            "ci95_game_cluster": [0.0, 0.0],
            "by_arm": {},
            "per_game_delta": {},
            "expert_control_rows": 0,
            "identity_control_rows": 0,
            "game_clusters": 0,
        }
    )
    failed_checks = [dict(check) for check in checks if check.get("passed") is not True]
    receipt_failures = [
        str(row.get("name"))
        for row in receipts
        if row.get("passed") is not True or row.get("exit_code") != 0
    ]
    frozen_units = sum(
        len(manifest.get(group, []))
        for group in ("induced_engine_windows", "expert_controls", "identity_controls")
    )
    measurement_complete = (
        not errors
        and len(raw_rows) == 2 * frozen_units
        and len({(str(row.get("unit_id")), str(row.get("arm"))) for row in raw_rows})
        == len(raw_rows)
        and reduction["induced_windows"] == len(manifest.get("induced_engine_windows", []))
    )
    valid = not failed_checks and not errors and not receipt_failures and not flagged
    ready = valid and measurement_complete
    benefit = ready and reduction["development_proxy_improvement"]
    if failed_checks:
        verdict_class = "blocked"
        verdict = "complete_blocked_" + str(failed_checks[0]["check"])
        summary: Any = {
            key: failed_checks[0].get(key)
            for key in ("check", "upstream", "path", "field", "operator", "expected", "observed")
        }
    elif receipt_failures or errors or flagged:
        verdict_class = "disqualified"
        verdict = "complete_disqualified_wrapper_validation_failed"
        summary = {
            "validation_failures": receipt_failures,
            "row_errors": errors,
            "flagged_adversarial": flagged,
        }
    elif benefit:
        verdict_class = "positive"
        verdict = "complete_positive_development_proxy_wrapper_improvement"
        summary = {
            "induced_new_successes": reduction["induced_new_successes"],
            "induced_lost_successes": reduction["induced_lost_successes"],
            "hidden_game_groups": 0,
        }
    else:
        verdict_class = "null"
        verdict = "complete_null_arc_wrapper_no_induced_engine_gain"
        summary = {
            "induced_new_successes": reduction["induced_new_successes"],
            "induced_lost_successes": reduction["induced_lost_successes"],
            "hidden_game_groups": 0,
        }
    per_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        per_game[str(row.get("game"))].append(
            {
                "window": row.get("unit_id"),
                "arm": row.get("arm"),
                "mask": row.get("mask"),
                "real_level_up": row.get("real_level_up"),
                "plan_found": row.get("plan_found"),
                "goal_checks": (row.get("planner_diagnostics") or {}).get("goal_evaluations", 0),
                "duplicate_skips": (row.get("planner_diagnostics") or {}).get(
                    "hud_dedup_states_merged", 0
                ),
                "engine_calls": row.get("planner_engine_calls"),
                "plan_length": row.get("plan_length"),
                "real_actions": row.get("real_actions_used"),
                "wall_s": row.get("planner_wall_s"),
                "solver_provenance": row.get("engine_family"),
                "denominator": row.get("denominator"),
            }
        )
    gate_results = {
        "validity": _gate(
            not failed_checks and not errors and not flagged,
            "Authentic inputs and safe full-grid goal checks are necessary for validity.",
            {"failed_preconditions": len(failed_checks), "row_errors": errors, "flagged": flagged},
        ),
        "readiness": _gate(
            ready,
            "All frozen windows and required current validation must pass.",
            {
                "observed_induced": reduction["induced_windows"],
                "expected_induced": len(manifest.get("induced_engine_windows", [])),
                "failed_receipts": receipt_failures,
            },
        ),
        "probability_benefit": _gate(
            False,
            "Development replay cannot estimate hidden-game success probability.",
            {"hidden_game_groups": 0, "development_proxy_gain": reduction["induced_new_successes"]},
        ),
        "utility": _gate(
            benefit,
            "One added induced success without loss must justify measured compute and actions in this proxy.",
            {
                "gain": reduction["induced_new_successes"],
                "loss": reduction["induced_lost_successes"],
                "costs": reduction["by_arm"],
            },
        ),
        "retention": _gate(
            ready,
            "Every identity-only source window remains accounted exactly once.",
            {
                "frozen_induced": len(manifest.get("induced_engine_windows", [])),
                "observed_induced": reduction["induced_windows"],
            },
        ),
        "freshness": _gate(
            bool(hashes.get("producer_files")) and bool(receipts),
            "Current source hashes and actual command exits bind this invocation.",
            {"source_files": len(hashes.get("producer_files", {})), "receipts": len(receipts)},
        ),
    }
    artifact: dict[str, Any] = {
        "schema": "carnot.experiment_7652_v667_arc_wrapper_measurement.v1",
        "experiment_id": 7652,
        "milestone": "2026.09.667",
        "run_date": RUN_DATE,
        "requirement_id": "REQ-REPORT-7652",
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": flagged,
        "gate_check_summary": summary,
        "acceptance_gate_results": gate_results,
        "rows": raw_rows,
        "per_game_results": dict(per_game),
        "independent_reduction": reduction,
        "sample_size_budget": {
            "intended_independent_groups": len(manifest.get("induced_engine_windows", []))
            + len(manifest.get("expert_controls", []))
            + len(manifest.get("identity_controls", [])),
            "observed_independent_groups": len({str(row.get("unit_id")) for row in raw_rows}),
            "eligible_induced_groups": reduction["usable_induced_windows"],
            "excluded_groups": 0,
            "censored_groups": len(
                {str(row.get("unit_id")) for row in raw_rows if row.get("censored")}
            ),
            "exposure_limit": "public offline windows; no hidden game",
            "repeated_seeds_views_and_orderings_increase_sample_size": False,
        },
        "preconditions_checked": deepcopy(list(checks)),
        "inference_substrate": "CPU offline ARC scored-wrapper replay of inherited engine source",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "historical_models": ["Qwen3.5-9B-MTP"],
        "model_invoked": False,
        "invocation_counts": {
            "loads_attempted": 0,
            "loads_completed": 0,
            "forward_calls_attempted": 0,
            "forward_calls_completed": 0,
            "generations_attempted": 0,
            "generations_completed": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "host": os.uname().nodename,
            "device_uuid": None,
            "owned_pid": os.getpid(),
        },
        "phase_spans": deepcopy(list(spans)),
        "duration_s": round(duration, 6),
        "random_seed": {"game_cluster_bootstrap": 7652, "arm_order": "OFF_then_HUD_DEDUP"},
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(hashes)),
        "validation_receipts": deepcopy(list(receipts)),
        "verifier_is_oracle": True,
        "solve_provenance": "development_proxy",
        "new_game_level_solve_credit": False,
        "production_defaults_changed": False,
        "per_game_adapter_routes_enabled": False,
        "wrapper_measurement_complete_score": int(measurement_complete),
        "wrapper_benefit_score": {
            "induced_engine_accuracy": {
                "new_successes": reduction["induced_new_successes"],
                "lost_successes": reduction["induced_lost_successes"],
                "ci95_game_cluster": reduction["ci95_game_cluster"],
            },
            "induced_engine_efficiency": reduction["by_arm"],
            "expert_control_findings": {
                "rows": reduction["expert_control_rows"],
                "hidden_game_credit": 0,
            },
            "identity_control_findings": {
                "rows": reduction["identity_control_rows"],
                "hidden_game_credit": 0,
            },
            "engine_call_budget": {
                "configured_max_nodes": 20000,
                "observed_max_engine_calls": max(
                    (int(row.get("planner_engine_calls") or 0) for row in raw_rows),
                    default=0,
                ),
                "rows_above_configured_limit": sum(
                    int(row.get("planner_engine_calls") or 0) > 20000 for row in raw_rows
                ),
            },
        },
        "window_manifest": deepcopy(dict(manifest)),
        "field_principles": {
            key: "Current measured operands govern this field; inherited readiness cannot establish benefit."
            for key in (
                "honest_verdict",
                "verdict_class",
                "flagged_adversarial",
                "gate_check_summary",
                "acceptance_gate_results",
                "rows",
                "per_game_results",
                "sample_size_budget",
                "preconditions_checked",
                "inference_substrate",
                "inference_substrate_class",
                "MODEL_SPECS",
                "model_invoked",
                "execution_venue",
                "phase_spans",
                "duration_s",
                "random_seed",
                "reproducibility_checksum",
                "source_artifact_hashes",
                "validation_receipts",
                "verifier_is_oracle",
                "wrapper_measurement_complete_score",
                "wrapper_benefit_score",
                "solve_provenance",
            )
        },
    }
    artifact["reproducibility_checksum"] = canonical_hash(artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any]) -> list[str]:
    """Cold-check exact stored operands without trusting producer summaries."""

    errors = validate_rows(value.get("rows", []))
    if not str(value.get("honest_verdict", "")).startswith("complete_"):
        errors.append("terminal_verdict")
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
        errors.append("no_model_load")
    if value.get("execution_venue") != "host":
        errors.append("execution_venue")
    copied = deepcopy(dict(value))
    observed = copied.get("reproducibility_checksum")
    copied["reproducibility_checksum"] = ""
    if canonical_hash(copied) != observed:
        errors.append("reproducibility_checksum")
    rows = value.get("rows", [])
    if rows:
        try:
            if reduce_pairs(rows) != value.get("independent_reduction"):
                errors.append("independent_reduction")
        except ValueError as exc:
            errors.append(str(exc))
    return sorted(set(errors))


def independent_replay(path: Path) -> list[str]:
    """Recompute game-cluster and arm totals in a separate process."""

    value = json.loads(path.read_text())
    return (
        []
        if reduce_pairs(value["rows"]) == value["independent_reduction"]
        else ["independent_reduction"]
    )


def _span(
    spans: list[dict[str, Any]], name: str, started: float, phase_start: float, units: int
) -> None:
    now = time.monotonic()
    spans.append(
        {
            "name": name,
            "started_elapsed_s": round(phase_start - started, 6),
            "ended_elapsed_s": round(now - started, 6),
            "duration_s": round(now - phase_start, 6),
            "completed_units": units,
            "checkpoint_position": name,
        }
    )


def _copy_receipts(
    root: Path, raw: Path, group: str, receipts: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:  # pragma: no cover - live integration
    copied = []
    for index, receipt in enumerate(receipts):
        row = dict(receipt)
        source = Path(str(row["log_path"]))
        if not source.is_absolute():
            source = root / source
        target = raw / "validation" / group / f"{index:02d}_{row['name']}.log"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        row["log_path"] = target.relative_to(root).as_posix()
        row["log_sha256"] = sha256_file(target)
        copied.append(row)
    return copied


def validation_manifest() -> dict[str, Any]:
    """Freeze exact affected paths before any validation child starts."""

    return {
        "tests": [str(TEST)],
        "changed_modules": [str(MODULE)],
        "static_paths": [str(WRAPPER), "python/carnot/experiment_10013_planner_dedup_tiebreak.py"],
        "additional_mypy_modules": ["python/carnot/experiment_10013_planner_dedup_tiebreak.py"],
        "coverage_required_percent": 100,
        "serial": True,
        "repository_addopts_disabled": True,
    }


def validation_commands(
    root: Path, private: Path
) -> list[CommandSpec]:  # pragma: no cover - live integration
    manifest = validation_manifest()
    commands = build_scoped_commands(
        root,
        manifest["tests"],
        manifest["changed_modules"],
        static_paths=manifest["static_paths"],
        basetemp=private / "pytest",
        coverage_file=private / "coverage" / ".coverage",
    )
    commands.append(
        CommandSpec(
            "scored_wrapper_mypy",
            (str(root / ".venv/bin/mypy"), "python/carnot/experiment_10013_planner_dedup_tiebreak.py"),
            "changed scored wrapper",
            300.0,
        )
    )
    commands.append(
        CommandSpec(
            "full_python_suite",
            (str(root / ".venv/bin/pytest"), "tests/python", "-q"),
            "required full Python suite",
            1800.0,
        )
    )
    return commands


def e2e_commands(
    root: Path, private: Path
) -> list[CommandSpec]:  # pragma: no cover - live integration
    pytest = str(root / ".venv/bin/pytest")
    return [
        CommandSpec(
            "e2e_arc_scored_wrapper",
            (
                pytest,
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={private / 'e2e-scored'}",
                "tests/python/test_arc_induction_state_persistence.py",
                "tests/python/test_arc_decision_telemetry.py",
                "-q",
            ),
            "E2E-009 and E2E-011 scored ARC policy",
            900.0,
        )
    ]


def terminal_commands(
    root: Path, candidate: Path
) -> list[CommandSpec]:  # pragma: no cover - live integration
    py = str(root / ".venv/bin/python")
    entry = str(root / WRAPPER)
    return [
        CommandSpec(
            "declared_entrypoint_validate",
            (py, "-u", entry, "--date", RUN_DATE, "--validate", str(candidate)),
            "exact private candidate",
            300.0,
        ),
        CommandSpec(
            "fresh_process_cold_replay",
            (py, "-u", entry, "--date", RUN_DATE, "--cold-replay", str(candidate)),
            "exact private candidate",
            300.0,
        ),
        CommandSpec(
            "independent_reduction",
            (py, "-u", entry, "--date", RUN_DATE, "--independent-reduce", str(candidate)),
            "exact private candidate",
            300.0,
        ),
        CommandSpec(
            "adversarial_verify",
            (py, "-u", str(root / "scripts/adversarial_verify.py"), str(candidate)),
            "exact private candidate",
            300.0,
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (
                py,
                "-u",
                str(root / "scripts/verdict_row_consistency_lint.py"),
                "--strict",
                str(candidate),
            ),
            "exact private candidate",
            300.0,
        ),
    ]


def run_experiment(
    repo_root: Path, run_date: str, output: Path
) -> dict[str, Any]:  # pragma: no cover - declared integration
    """Run CPU replay, bounded children, exact readers, then atomic publication."""

    started = time.monotonic()
    root = repo_root.resolve()
    progress(started, "startup", "begin", root=root)
    if root != ROOT or run_date != RUN_DATE:
        raise ValueError(f"run_contract_mismatch:{root}:{run_date}")
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7652-")).resolve()
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    spans: list[dict[str, Any]] = []
    phase = time.monotonic()
    progress(started, "preconditions", "before")
    checks, hashes = collect_preconditions(root)
    progress(started, "preconditions", "after", failed=sum(not row["passed"] for row in checks))
    _span(spans, "preconditions", started, phase, len(checks))
    if any(not row["passed"] for row in checks):
        artifact = build_artifact(
            rows=[],
            manifest={},
            checks=checks,
            hashes=hashes,
            receipts=[],
            spans=spans,
            duration=time.monotonic() - started,
        )
        atomic_json(output, artifact)
        return artifact

    phase = time.monotonic()
    progress(started, "census", "before")
    manifest, prepared, _, source_failures = prepare_windows(root)
    checks.extend(source_failures)
    atomic_json(raw / "windows.json", manifest)
    atomic_json(raw / "affected_validation_manifest.json", validation_manifest())
    progress(
        started,
        "census",
        "after",
        induced=len(manifest["induced_engine_windows"]),
        controls=len(manifest["expert_controls"]) + len(manifest["identity_controls"]),
        source_failures=len(source_failures),
    )
    _span(spans, "census", started, phase, len(manifest["induced_engine_windows"]))
    if source_failures:
        artifact = build_artifact(
            rows=[],
            manifest=manifest,
            checks=checks,
            hashes=hashes,
            receipts=[],
            spans=spans,
            duration=time.monotonic() - started,
        )
        atomic_json(output, artifact)
        return artifact

    phase = time.monotonic()
    progress(started, "measurement", "before_benchmark")
    rows = measure_windows(root, raw, manifest, prepared, started)
    progress(started, "measurement", "after_benchmark", rows=len(rows))
    _span(spans, "measurement", started, phase, len(rows))

    (private / "pytest").mkdir(parents=True, exist_ok=True)
    (private / "coverage").mkdir(parents=True, exist_ok=True)
    phase = time.monotonic()
    progress(started, "validation", "before_subprocess")
    validation = run_commands(
        root,
        validation_commands(root, private),
        log_dir=private / "logs" / "affected",
        extra_env={
            "JAX_PLATFORMS": "cpu",
            "COVERAGE_FILE": str(private / "coverage" / ".coverage"),
        },
        heartbeat_s=60,
    )
    progress(
        started,
        "validation",
        "after_subprocess",
        failed=sum(not row["passed"] for row in validation),
    )
    _span(spans, "validation", started, phase, len(validation))
    phase = time.monotonic()
    progress(started, "e2e", "before_subprocess")
    e2e = run_commands(
        root,
        e2e_commands(root, private),
        log_dir=private / "logs" / "e2e",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=60,
    )
    progress(started, "e2e", "after_subprocess", failed=sum(not row["passed"] for row in e2e))
    _span(spans, "e2e", started, phase, len(e2e))
    receipts = [
        *_copy_receipts(root, raw, "affected", validation),
        *_copy_receipts(root, raw, "e2e", e2e),
    ]
    hashes["producer_files"].update(
        {
            str(path): sha256_file(root / path)
            for path in (
                MODULE,
                TEST,
                WRAPPER,
                SPEC,
                Path("python/carnot/experiment_10013_planner_dedup_tiebreak.py"),
            )
        }
    )
    hashes["pre_gate_receipts"][str(RAW / "windows.json")] = sha256_file(raw / "windows.json")
    candidate_path = private / "terminal_candidate.json"
    candidate = build_artifact(
        rows=rows,
        manifest=manifest,
        checks=checks,
        hashes=hashes,
        receipts=receipts,
        spans=spans,
        duration=time.monotonic() - started,
    )
    atomic_json(candidate_path, candidate)
    phase = time.monotonic()
    progress(
        started,
        "terminal_readers",
        "before_subprocess",
        candidate_sha256=sha256_file(candidate_path),
    )
    terminal = run_commands(
        root,
        terminal_commands(root, candidate_path),
        log_dir=private / "logs" / "terminal",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=60,
    )
    progress(
        started,
        "terminal_readers",
        "after_subprocess",
        failed=sum(not row["passed"] for row in terminal),
    )
    _span(spans, "terminal_readers", started, phase, len(terminal))
    terminal_receipts = _copy_receipts(root, raw, "terminal", terminal)
    atomic_json(
        raw / "terminal_validation_receipts.json",
        {"candidate_sha256": sha256_file(candidate_path), "receipts": terminal_receipts},
    )
    flagged = next(
        (
            row["passed"] is not True
            for row in terminal_receipts
            if row["name"] == "adversarial_verify"
        ),
        True,
    )
    final = build_artifact(
        rows=rows,
        manifest=manifest,
        checks=checks,
        hashes=hashes,
        receipts=[*receipts, *terminal_receipts],
        spans=spans,
        duration=time.monotonic() - started,
        flagged=flagged,
    )
    if validate_artifact(final):
        raise ValueError(f"terminal_artifact_invalid:{validate_artifact(final)}")
    progress(started, "publication", "before_atomic", output=output)
    atomic_json(output, final)
    progress(started, "publication", "after_atomic", verdict=final["honest_verdict"])
    return final


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    args = parser.parse_args(argv)
    if args.date != RUN_DATE:
        raise ValueError("run_date_mismatch")
    if args.validate or args.cold_replay:
        value = json.loads((args.validate or args.cold_replay).read_text())
        errors = validate_artifact(value)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce:
        errors = independent_replay(args.independent_reduce)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    output = args.output if args.output.is_absolute() else ROOT / args.output
    run_experiment(ROOT, args.date, output)
    return 0
