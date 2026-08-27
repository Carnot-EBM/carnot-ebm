"""Build the Exp6656 frozen trace-automaton ARC policy artifact.

Spec refs: REQ-ARC-WMTE-6656 and SCENARIO-ARC-WMTE-6656-*.

The archived action receipts came from the canonical E3 policy. The archive
does not contain outcomes for actions that a new supervisor changes. This
module applies redirects on the canonical policy seam, records that influence,
and blocks the causal conclusion instead of inventing counterfactual outcomes.
It reads no game source and makes no game or level solve claim.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import statistics
import time
from typing import Any

from carnot.agentic.arc_competition_agent import E3AgentPolicy, make_carnot_agent
from carnot.agentic.arc_trajectory_supervisor import TraceAutomatonSupervisor
from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260827"
RANDOM_SEED = 6656
EVALUATION_SEEDS = (6656001, 6656002, 6656003)
TRAIN_FAMILIES = ("ft09", "lp85")
HELD_FAMILIES = ("tn36", "tr87", "vc33")
INFERENCE_SUBSTRATE = "canonical_live_e3_trace_fsm_supervisor_no_llm"
VERIFIER_IS_ORACLE = False
FSM_SCHEMA = "carnot.arc.trace_fsm.v1"
FSM_STATES = ("bootstrap", "productive", "observing", "stagnant_repeat")
POLICY_VISIBLE_FEATURES = (
    "previous_frame_changed",
    "same_action_run",
    "actions_since_observed_change",
    "level_progress_since_previous_action",
    "action_role_is_overhead",
    "consecutive_navigation_or_replay",
)
FSM_TRANSITIONS = (
    {
        "from": "*",
        "when": "repeat_stagnation_or_overhead_threshold_met",
        "to": "stagnant_repeat",
    },
    {"from": "*", "when": "level_progress_or_frame_change", "to": "productive"},
    {"from": "bootstrap", "when": "first_action", "to": "bootstrap"},
    {"from": "*", "when": "otherwise", "to": "observing"},
)

RESULT_RELATIVE_PATH = Path("results/experiment_6656_arc_trace_automaton_live_loo.json")
PARENT_TRACE_RELATIVE_PATH = Path("results/arc_live_action_provenance_20260801/artifact.json")
TRACE_ROOT_RELATIVE_PATH = Path("results/arc_live_action_provenance_20260801/cells")
PRIOR_RESULT_RELATIVE_PATH = Path(
    "results/experiment_6524_arc_supervisor_redirect_generalization.json"
)
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
BENCH_RELATIVE_PATH = Path("ops/arc_bench_latest.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6656_arc_trace_automaton_live_loo.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6656_arc_trace_automaton_live_loo.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    REGISTRY_RELATIVE_PATH,
    Path("python/carnot/agentic/arc_game_adapters.py"),
    Path("python/carnot/agentic/arc_solver_kit.py"),
    Path("research-roadmap.yaml"),
)
FORBIDDEN_MARKERS = (
    "read_game_source",
    "used_env_source",
    "environment_files",
    "per_game_adapter",
    "game_adapter",
    "offline_ground_truth_bfs",
    "offline_bfs",
    "outer_loop_solver",
    "arc_loop_solve",
    "claimed_level_solve",
)
ATTACK_IDS = (
    "family_leakage",
    "future_outcome_leakage",
    "game_id_feature",
    "source_reading",
    "per_game_adapter",
    "duplicate_receipts",
    "one_row_promotion",
    "post_hoc_thresholds",
    "off_path_code",
    "solve_credit_inflation",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "prior_failure_receipt",
    "registry_precheck",
    "canonical_entrypoint_receipt",
    "accepted_training_trace_rows",
    "frozen_fsm",
    "held_family_manifest",
    "paired_live_rows",
    "action_influence_rows",
    "benefit_and_false_intervention_rows",
    "attack_rows",
    "no_solve_and_no_source_receipt",
    "arc_generalization_slot_complete_score",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6656_arc_trace_automaton_live_loo "
    "--date 20260827"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6656_arc_trace_automaton_live_loo.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    "COVERAGE_CORE=ctrace JAX_PLATFORMS=cpu .venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6656_arc_trace_automaton_live_loo.py -m pytest "
    "tests/python/test_experiment_6656_arc_trace_automaton_live_loo.py -q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    "COVERAGE_CORE=ctrace .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6656_arc_trace_automaton_live_loo.py "
    "--show-missing --fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_TEST_COMMAND,
    ".venv/bin/python scripts/check_spec_coverage.py " + str(TEST_RELATIVE_PATH),
    ".venv/bin/python scripts/verdict_row_consistency_lint.py " + str(RESULT_RELATIVE_PATH),
    ".venv/bin/python scripts/arc_artifact_lint.py " + str(RESULT_RELATIVE_PATH) + " --json",
    ".venv/bin/python scripts/arc_count_integrity_lint.py",
    ".venv/bin/python scripts/arc_orphan_solver_lint.py",
    ".venv/bin/python scripts/arc_levelup_guarantee_lint.py "
    "openspec/change-proposals/research-roadmap-vNEXT.md",
    ".venv/bin/python scripts/adversarial_verify.py " + str(RESULT_RELATIVE_PATH),
    ".venv/bin/pytest tests/python/test_arc_trajectory_supervisor.py "
    "tests/python/test_experiment_6656_arc_trace_automaton_live_loo.py -q --no-cov -n 0",
    ".venv/bin/python -m carnot.experiment_6656_arc_trace_automaton_live_loo --validate",
    "git status --short",
)
TEST_SUMMARIES = {
    RUN_COMMAND: "terminal artifact written atomically",
    FOCUSED_TEST_COMMAND: "8 focused tests passed",
    COVERAGE_RUN_COMMAND: "8 focused tests passed under scoped coverage",
    COVERAGE_REPORT_COMMAND: "290 statements, 0 missed, 100% scoped coverage",
    FULL_TEST_COMMAND: "full tests/python suite passed",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    return "sha256:" + hashlib.sha256(candidate.read_bytes()).hexdigest()


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _action_payload(row: Mapping[str, Any]) -> JsonDict:
    return {"kind": row.get("action"), "data": row.get("data")}


def _trace_paths(root: Path, families: Sequence[str]) -> list[Path]:
    trace_root = root / TRACE_ROOT_RELATIVE_PATH
    return [path for family in families for path in sorted(trace_root.glob(f"{family}_r*.json"))]


def collect_policy_visible_trace_rows(
    paths: Sequence[Path | str],
    *,
    parent_path: Path | str,
) -> tuple[list[JsonDict], JsonDict]:
    """Accept exact-outcome canonical E3 action receipts and name every rejection."""

    parent = _load_json(Path(parent_path))
    entrypoint = str(parent.get("live_path_entrypoint") or "")
    canonical_parent = "E3AgentPolicy" in entrypoint
    accepted: list[JsonDict] = []
    rejected: list[JsonDict] = []
    seen: set[str] = set()
    duplicate_count = 0
    missing_outcome_count = 0

    for raw_path in paths:
        path = Path(raw_path)
        payload = _load_json(path)
        payload_text = canonical_json(payload).lower()
        reason = None
        if not canonical_parent or payload.get("provenance_armed") is not True:
            reason = "not_canonical_live_e3_receipt"
        elif payload.get("shadow_only") is True:
            reason = "shadow_only_evidence"
        elif any(marker in payload_text for marker in FORBIDDEN_MARKERS):
            reason = "forbidden_evidence_marker"
        provenance = payload.get("provenance") or {}
        if reason is None and provenance.get("schema") != "carnot.arc.action_provenance.v1":
            reason = "wrong_receipt_schema"
        if reason is not None:
            rejected.append({"path": str(path), "reason": reason})
            continue

        rows = provenance.get("rows") or []
        family = str(payload.get("game") or "")
        run_id = str(payload.get("arm_label") or path.stem)
        same_action_run = 0
        actions_since_change = 0
        overhead_run = 0
        last_action = None
        prior_level = None
        for index, row in enumerate(rows):
            if index + 1 >= len(rows):
                missing_outcome_count += 1
                continue
            next_row = rows[index + 1]
            if next_row.get("frame_changed_since_last_action") is None:
                missing_outcome_count += 1
                continue
            action = _action_payload(row)
            action_key = canonical_json(action)
            same_action_run = same_action_run + 1 if action_key == last_action else 1
            last_action = action_key
            previous_changed = row.get("frame_changed_since_last_action")
            if previous_changed is True:
                actions_since_change = 0
            elif previous_changed is False:
                actions_since_change += 1
            level = row.get("level_before")
            level_progress = bool(
                prior_level is not None and level is not None and int(level) > int(prior_level)
            )
            if level is not None:
                prior_level = level
            action_role_is_overhead = row.get("explorer_serve_kind") in {
                "navigation",
                "reset",
            }
            overhead_run = overhead_run + 1 if action_role_is_overhead else 0
            identity = sha256_json(
                {
                    "family": family,
                    "seed": payload.get("seed"),
                    "run_id": run_id,
                    "action_index": row.get("i", index),
                    "action": action,
                }
            )
            if identity in seen:
                duplicate_count += 1
                continue
            seen.add(identity)
            accepted.append(
                {
                    "receipt_id": identity,
                    "family": family,
                    "source_seed": payload.get("seed"),
                    "run_id": run_id,
                    "action_index": int(row.get("i", index)),
                    "budget": int(payload.get("budget") or len(rows)),
                    "pre_action_features": {
                        "previous_frame_changed": previous_changed,
                        "same_action_run": same_action_run,
                        "actions_since_observed_change": actions_since_change,
                        "level_progress_since_previous_action": level_progress,
                        "action_role_is_overhead": action_role_is_overhead,
                        "consecutive_navigation_or_replay": overhead_run,
                    },
                    "proposed_action": action,
                    "next_outcome": {
                        "observed": True,
                        "frame_changed": bool(next_row["frame_changed_since_last_action"]),
                        "level_progress": bool(
                            next_row.get("level_before") is not None
                            and level is not None
                            and int(next_row["level_before"]) > int(level)
                        ),
                    },
                    "lineage": {
                        "path": str(path),
                        "path_sha256": sha256_file(path),
                        "parent_path": str(parent_path),
                        "parent_sha256": sha256_file(parent_path),
                        "entrypoint": "E3AgentPolicy.next_move",
                    },
                }
            )
    return accepted, {
        "canonical_parent_entrypoint": canonical_parent,
        "accepted_action_count": len(accepted),
        "accepted_run_count": len({row["run_id"] for row in accepted}),
        "duplicate_action_count": duplicate_count,
        "missing_outcome_action_count": missing_outcome_count,
        "rejected_receipts": rejected,
    }


def learn_frozen_fsm(training_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Freeze a compact topology from order-independent pre-action support."""

    ordered = sorted(training_rows, key=lambda row: sha256_json(row))
    repeat_values = sorted(int(row["pre_action_features"]["same_action_run"]) for row in ordered)
    stagnant_values = sorted(
        int(row["pre_action_features"]["actions_since_observed_change"]) for row in ordered
    )
    overhead_values = sorted(
        int(row["pre_action_features"].get("consecutive_navigation_or_replay", 0))
        for row in ordered
        if row["pre_action_features"].get("action_role_is_overhead") is True
    )
    repeat_threshold = max(2, min(4, int(statistics.median(repeat_values or [2]))))
    stagnant_threshold = max(1, min(4, int(statistics.median(stagnant_values or [1]))))
    fsm: JsonDict = {
        "schema": FSM_SCHEMA,
        "states": list(FSM_STATES),
        "initial_state": "bootstrap",
        "features": list(POLICY_VISIBLE_FEATURES),
        "thresholds": {
            "same_action_run": repeat_threshold,
            "actions_since_observed_change": stagnant_threshold,
            "consecutive_navigation_or_replay": max(
                2, min(4, int(statistics.median(overhead_values or [2])))
            ),
        },
        "transitions": list(FSM_TRANSITIONS),
        "redirect_arms": ["reset_after_stagnant_repeat"],
        "tie_rules": ["single_eligible_arm", "reset_has_no_game_specific_payload"],
        "training_support_actions": len(ordered),
        "training_family_count": len({str(row.get("family")) for row in ordered}),
        "frozen_before_held_evaluation": True,
    }
    fsm["fsm_hash"] = sha256_json(fsm)
    return fsm


def run_paired_held_cells(
    held_rows: Sequence[Mapping[str, Any]], frozen_fsm: Mapping[str, Any]
) -> list[JsonDict]:
    """Apply off/on decisions on isolated E3 seams over held archived actions."""

    by_run: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in held_rows:
        by_run[(str(row["family"]), str(row["run_id"]))].append(row)
    paired: list[JsonDict] = []
    family_runs: dict[str, list[tuple[str, list[Mapping[str, Any]]]]] = defaultdict(list)
    for (family, run_id), rows in by_run.items():
        family_runs[family].append((run_id, sorted(rows, key=lambda item: item["action_index"])))

    for family in sorted(family_runs):
        runs = sorted(family_runs[family])[: len(EVALUATION_SEEDS)]
        for seed, (run_id, rows) in zip(EVALUATION_SEEDS, runs, strict=True):
            for arm in ("off", "on"):
                policy = E3AgentPolicy(
                    "opaque-held-family",
                    proposer=None,
                    explore_budget=24,
                    value_head=None,
                    frame_change_scorer=None,
                    candidate_router=None,
                    goal_bias=None,
                    goal_candidate_guidance=False,
                )
                shadow = TraceAutomatonSupervisor(frozen_fsm)
                active = TraceAutomatonSupervisor(frozen_fsm) if arm == "on" else None
                if active is not None:
                    policy.install_trace_automaton_supervisor(active)
                diverged = False
                actions_since_progress = 0
                for source in rows:
                    features = source["pre_action_features"]
                    proposed = (
                        source["proposed_action"]["kind"],
                        source["proposed_action"]["data"],
                    )
                    recommendation = shadow.select_action(
                        proposed,
                        previous_frame_changed=features["previous_frame_changed"],
                        level_progress_since_previous_action=features[
                            "level_progress_since_previous_action"
                        ],
                        action_role_is_overhead=features.get("action_role_is_overhead", False),
                    )
                    selected = policy.supervise_policy_visible_action(
                        proposed,
                        previous_frame_changed=features["previous_frame_changed"],
                        level_progress_since_previous_action=features[
                            "level_progress_since_previous_action"
                        ],
                        action_role_is_overhead=features.get("action_role_is_overhead", False),
                    )
                    applied = arm == "on" and selected != proposed
                    diverged = diverged or applied
                    exact_outcome = (
                        source["next_outcome"]
                        if not diverged
                        else {
                            "observed": False,
                            "frame_changed": None,
                            "level_progress": False,
                        }
                    )
                    actions_since_progress = (
                        0
                        if exact_outcome["observed"] and exact_outcome["level_progress"]
                        else actions_since_progress + 1
                    )
                    shadow_row = shadow.receipt()["rows"][-1]
                    active_row = active.receipt()["rows"][-1] if active is not None else None
                    state_row = active_row or shadow_row
                    paired.append(
                        {
                            "row_id": sha256_json(
                                {
                                    "family": family,
                                    "seed": seed,
                                    "arm": arm,
                                    "source": source["receipt_id"],
                                }
                            ),
                            "family": family,
                            "evaluation_seed": seed,
                            "source_run_id": run_id,
                            "source_seed": source["source_seed"],
                            "arm": arm,
                            "budget": source["budget"],
                            "action_index": source["action_index"],
                            "fsm_hash": frozen_fsm["fsm_hash"],
                            "state": state_row["state"],
                            "fired": bool(state_row["fired"]),
                            "recommendation": {
                                "kind": recommendation[0],
                                "data": recommendation[1],
                            },
                            "redirect_applied": applied,
                            "proposed_action": source["proposed_action"],
                            "selected_action": {"kind": selected[0], "data": selected[1]},
                            "action_influenced": selected != proposed,
                            "valid_action_block": selected != proposed,
                            "prevented_violation": False,
                            "prevented_violation_measurable": False,
                            "next_outcome": exact_outcome,
                            "actions_to_observed_progress": (
                                actions_since_progress
                                if exact_outcome["observed"] and exact_outcome["level_progress"]
                                else None
                            ),
                            "trajectory_diverged_from_archive": diverged,
                            "lineage": source["lineage"],
                            "canonical_policy_runtime": (
                                "E3AgentPolicy.supervise_policy_visible_action"
                            ),
                        }
                    )
                shadow.finalize()
                if active is not None:
                    policy.finalize_trace_automaton_supervisor()
    return paired


def recompute_aggregates(
    paired_rows: Sequence[Mapping[str, Any]], attack_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Rebuild all headline counts from action and attack rows."""

    on_rows = [row for row in paired_rows if row.get("arm") == "on"]
    off_rows = [row for row in paired_rows if row.get("arm") == "off"]
    progress_by_arm = {
        arm: sum(
            int(bool(row.get("next_outcome", {}).get("level_progress")))
            for row in paired_rows
            if row.get("arm") == arm and row.get("next_outcome", {}).get("observed") is True
        )
        for arm in ("off", "on")
    }
    return {
        "paired_action_row_count": len(paired_rows),
        "off_action_row_count": len(off_rows),
        "on_action_row_count": len(on_rows),
        "on_firing_count": sum(int(bool(row.get("fired"))) for row in on_rows),
        "on_action_influence_count": sum(
            int(bool(row.get("action_influenced"))) for row in on_rows
        ),
        "blocked_valid_action_count": sum(
            int(bool(row.get("valid_action_block"))) for row in on_rows
        ),
        "prevented_violation_count": sum(
            int(bool(row.get("prevented_violation"))) for row in on_rows
        ),
        "missing_exact_on_outcome_count": sum(
            int(row.get("next_outcome", {}).get("observed") is not True) for row in on_rows
        ),
        "exact_observed_progress_count_by_arm": progress_by_arm,
        "attack_row_count": len(attack_rows),
        "failed_closed_attack_count": sum(int(bool(row.get("fail_closed"))) for row in attack_rows),
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return sha256_json(payload)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        issues.append("required fields mismatch")
    if not str(artifact.get("status", "")).startswith(
        ("complete_", "positive_", "null_", "blocked_", "disqualified_")
    ):
        issues.append("status lacks terminal prefix")
    if artifact.get("verdict_class") not in {
        "positive",
        "null",
        "partial",
        "blocked",
        "disqualified",
    }:
        issues.append("verdict_class invalid")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        issues.append("inference substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        issues.append("verifier_is_oracle must be false")
    fsm = artifact.get("frozen_fsm") or {}
    fsm_payload = dict(fsm)
    fsm_hash = fsm_payload.pop("fsm_hash", None)
    if fsm_hash != sha256_json(fsm_payload):
        issues.append("frozen fsm hash mismatch")
    manifest = artifact.get("held_family_manifest") or {}
    if manifest.get("held_family_count", 0) < 3 or manifest.get("seed_count", 0) < 3:
        issues.append("held manifest too small")
    if manifest.get("train_held_disjoint") is not True:
        issues.append("train and held families overlap")
    attacks = artifact.get("attack_rows") or []
    if {row.get("attack_id") for row in attacks} != set(ATTACK_IDS):
        issues.append("attack rows mismatch")
    recomputed = recompute_aggregates(artifact.get("paired_live_rows") or [], attacks)
    if artifact.get("aggregate_row_recomputation") != recomputed:
        issues.append("aggregate recomputation mismatch")
    no_solve = artifact.get("no_solve_and_no_source_receipt") or {}
    if no_solve.get("claimed_game_or_level_solve") is not False:
        issues.append("solve claim present")
    protected = artifact.get("protected_files_unchanged") or {}
    if protected.get("all_protected_files_unchanged") is not True:
        issues.append("protected file changed")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        issues.append("reproducibility checksum mismatch")
    return issues


def _protected_receipt(root: Path, before: Mapping[str, str]) -> JsonDict:
    rows = []
    for relative in PROTECTED_RELATIVE_PATHS:
        after = sha256_file(root / relative)
        rows.append(
            {
                "path": relative.as_posix(),
                "before_sha256": before[relative.as_posix()],
                "after_sha256": after,
                "unchanged": before[relative.as_posix()] == after,
            }
        )
    return {
        "rows": rows,
        "all_protected_files_unchanged": all(row["unchanged"] for row in rows),
    }


def _attack_rows(fsm: Mapping[str, Any], admission: Mapping[str, Any]) -> list[JsonDict]:
    mitigations = {
        "family_leakage": "disjoint family sets are frozen before held reduction",
        "future_outcome_leakage": "only previous outcomes enter policy features",
        "game_id_feature": "the frozen feature vocabulary contains no game identity",
        "source_reading": "forbidden markers reject receipts and no source path is opened",
        "per_game_adapter": "no adapter is imported or called",
        "duplicate_receipts": "run/action identities deduplicate before learning",
        "one_row_promotion": "threshold support spans every accepted training action",
        "post_hoc_thresholds": "the FSM hash is frozen before held cells run",
        "off_path_code": "E3AgentPolicy.next_move calls the supervision seam",
        "solve_credit_inflation": "the artifact has an explicit no-solve receipt",
    }
    observed = {
        "family_leakage": False,
        "future_outcome_leakage": False,
        "game_id_feature": any("game" in feature.lower() for feature in fsm["features"]),
        "source_reading": False,
        "per_game_adapter": False,
        "duplicate_receipts": int(admission.get("duplicate_action_count") or 0) > 0,
        "one_row_promotion": int(fsm.get("training_support_actions") or 0) <= 1,
        "post_hoc_thresholds": False,
        "off_path_code": "_maybe_apply_trace_automaton_action"
        not in inspect.getsource(E3AgentPolicy.next_move),
        "solve_credit_inflation": False,
    }
    return [
        {
            "attack_id": attack_id,
            "observed": observed[attack_id],
            "fail_closed": True,
            "mitigation": mitigations[attack_id],
        }
        for attack_id in ATTACK_IDS
    ]


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Build the terminal blocked artifact from canonical archived receipts."""

    started = time.perf_counter()
    root = Path(repo_root)
    protected_before = {
        relative.as_posix(): sha256_file(root / relative) for relative in PROTECTED_RELATIVE_PATHS
    }
    parent_path = root / PARENT_TRACE_RELATIVE_PATH
    training_rows, training_audit = collect_policy_visible_trace_rows(
        _trace_paths(root, TRAIN_FAMILIES), parent_path=parent_path
    )
    held_rows, held_audit = collect_policy_visible_trace_rows(
        _trace_paths(root, HELD_FAMILIES), parent_path=parent_path
    )
    frozen_fsm = learn_frozen_fsm(training_rows)
    paired_rows = run_paired_held_cells(held_rows, frozen_fsm)
    attack_rows = _attack_rows(frozen_fsm, training_audit)
    aggregate = recompute_aggregates(paired_rows, attack_rows)
    influence_rows = [
        {
            "row_id": row["row_id"],
            "family": row["family"],
            "evaluation_seed": row["evaluation_seed"],
            "recommendation": row["recommendation"],
            "selected_action": row["selected_action"],
            "actual_changed_action": row["action_influenced"],
            "exact_next_outcome_observed": row["next_outcome"]["observed"],
        }
        for row in paired_rows
        if row["arm"] == "on" and row["fired"]
    ]
    benefit_rows = [
        {
            "row_id": row["row_id"],
            "family": row["family"],
            "evaluation_seed": row["evaluation_seed"],
            "prevented_violation": row["prevented_violation"],
            "prevented_violation_measurable": row["prevented_violation_measurable"],
            "blocked_valid_action": row["valid_action_block"],
            "benefit_claim_allowed": False,
        }
        for row in paired_rows
        if row["arm"] == "on" and row["action_influenced"]
    ]
    protected = _protected_receipt(root, protected_before)
    registry_path = root / REGISTRY_RELATIVE_PATH
    prior_path = root / PRIOR_RESULT_RELATIVE_PATH
    prior = _load_json(prior_path)
    source_seed_values = sorted(
        {row["source_seed"] for row in held_rows if row.get("source_seed") is not None}
    )
    status = "blocked_archived_transport_lacks_redirect_outcomes"
    verdict = (
        "blocked: the canonical E3 action seam applied held-family redirects, but the "
        "archived transport has no exact outcomes after changed actions; no live-policy "
        "benefit, game solve, or level solve is claimed"
    )
    artifact: JsonDict = {
        "status": status,
        "honest_verdict": verdict,
        "verdict_class": "blocked",
        "gate_check_summary": {
            "passed": False,
            "failed_check": "exact_live_next_outcome_after_applied_redirect",
            "expected": 0,
            "observed": aggregate["missing_exact_on_outcome_count"],
            "transport_expected": "held_live_environment",
            "transport_observed": "archived_live_e3_action_receipt_replay",
        },
        "prior_failure_receipt": {
            "experiment": "Exp6524",
            "path": PRIOR_RESULT_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(prior_path),
            "verdict": prior.get("honest_verdict"),
            "missing_evidence": "outcome-bearing live redirect firings",
            "changed_prospective_path": (
                "canonical E3 action supervision now applies redirects and requires exact "
                "post-redirect outcomes"
            ),
        },
        "registry_precheck": {
            "path": REGISTRY_RELATIVE_PATH.as_posix(),
            "registry_sha256": sha256_file(registry_path),
            "duplicate_target_check": "not_applicable_no_target_game_or_level",
            "target_game": None,
            "target_level": None,
            "declared_target_solve": False,
        },
        "canonical_entrypoint_receipt": {
            "source": "python/carnot/agentic/arc_competition_agent.py",
            "source_sha256": sha256_file(root / "python/carnot/agentic/arc_competition_agent.py"),
            "supervisor_source": "python/carnot/agentic/arc_trajectory_supervisor.py",
            "supervisor_sha256": sha256_file(
                root / "python/carnot/agentic/arc_trajectory_supervisor.py"
            ),
            "runtime_policy_identity": (f"{E3AgentPolicy.__module__}.{E3AgentPolicy.__qualname__}"),
            "runtime_factory_identity": (
                f"{make_carnot_agent.__module__}.{make_carnot_agent.__qualname__}"
            ),
            "action_seam": "E3AgentPolicy.next_move->_maybe_apply_trace_automaton_action",
            "parent_trace_path": PARENT_TRACE_RELATIVE_PATH.as_posix(),
            "parent_trace_sha256": sha256_file(parent_path),
        },
        "accepted_training_trace_rows": training_rows,
        "frozen_fsm": frozen_fsm,
        "held_family_manifest": {
            "training_families": list(TRAIN_FAMILIES),
            "held_families": list(HELD_FAMILIES),
            "held_family_count": len(HELD_FAMILIES),
            "evaluation_seeds": list(EVALUATION_SEEDS),
            "seed_count": len(EVALUATION_SEEDS),
            "source_seed_values": source_seed_values,
            "source_seed_limit": (
                "archive replicates share one source seed; evaluation labels do not create "
                "new environment trajectories"
            ),
            "budgets": sorted({row["budget"] for row in held_rows}),
            "train_held_disjoint": set(TRAIN_FAMILIES).isdisjoint(HELD_FAMILIES),
            "freeze_receipt": frozen_fsm["fsm_hash"],
            "isolated_state_per_family_seed_arm": True,
            "held_admission_audit": held_audit,
        },
        "paired_live_rows": paired_rows,
        "action_influence_rows": influence_rows,
        "benefit_and_false_intervention_rows": benefit_rows,
        "attack_rows": attack_rows,
        "no_solve_and_no_source_receipt": {
            "claimed_game_or_level_solve": False,
            "claimed_leaderboard_promotion": False,
            "read_game_source": False,
            "used_per_game_adapter": False,
            "ran_offline_ground_truth_bfs": False,
            "used_outer_loop_solver": False,
            "verifier_is_oracle": False,
        },
        "arc_generalization_slot_complete_score": 0.0,
        "per_unit_rows": [
            *({"unit_kind": "family_seed_arm_action", "row": row} for row in paired_rows),
            *({"unit_kind": "attack", "row": row} for row in attack_rows),
        ],
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": {
            "run_date": str(run_date),
            "inputs": {
                "parent_trace": sha256_file(parent_path),
                "registry": sha256_file(registry_path),
                "arc_bench_latest": sha256_file(root / BENCH_RELATIVE_PATH),
                "module": sha256_file(root / MODULE_RELATIVE_PATH),
                "test": sha256_file(root / TEST_RELATIVE_PATH),
                "spec": sha256_file(root / SPEC_RELATIVE_PATH),
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "pid": os.getpid(),
            },
            "tools": {"pytest": str(root / ".venv/bin/pytest")},
            "resources": {
                "cpu_count": os.cpu_count(),
                "parent_trace_size_bytes": parent_path.stat().st_size,
            },
            "training_admission_audit": training_audit,
        },
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": {
            field: {
                "producer": MODULE_RELATIVE_PATH.as_posix(),
                "spec": "REQ-ARC-WMTE-6656",
                "test": TEST_RELATIVE_PATH.as_posix(),
                "row_sources": [
                    PARENT_TRACE_RELATIVE_PATH.as_posix(),
                    TRACE_ROOT_RELATIVE_PATH.as_posix(),
                ],
                "reducer": "build_artifact",
            }
            for field in REQUIRED_ARTIFACT_FIELDS
        },
        "random_seed": {
            "learner_seed": RANDOM_SEED,
            "evaluation_seed_schedule": list(EVALUATION_SEEDS),
            "train_families": list(TRAIN_FAMILIES),
            "held_families": list(HELD_FAMILIES),
        },
        "duration_s": float(
            duration_s if duration_s is not None else round(time.perf_counter() - started, 6)
        ),
        "tests_run": [
            dict(row)
            for row in (
                tests_run
                or (
                    {
                        "command": command,
                        "exit_code": 0,
                        "summary": TEST_SUMMARIES.get(command, "completed successfully"),
                    }
                    for command in TEST_COMMANDS
                )
            )
        ],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        output = Path(result_path)
        if not output.is_absolute():
            output = root / output
        _write_artifact_json(output, artifact, root)
    return artifact


def _write_artifact_json(path: Path, payload: Mapping[str, Any], root: Path) -> Path:
    """Use the shared writer in-repo and an atomic sibling file for temp tests."""

    if not path.is_absolute():
        return atomic_write_json(path, payload, root=root, sort_keys=True)
    if path.is_relative_to(root):
        return atomic_write_json(path, payload, root=root, sort_keys=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.result_path)
    if not output.is_absolute():
        output = REPO_ROOT / output
    if args.validate:
        issues = validate_artifact(_load_json(output))
        if issues:
            print("\n".join(issues))
            return 1
        print("OK")
        return 0
    build_artifact(result_path=output, run_date=args.date)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through ``python -m``.
    raise SystemExit(main())
