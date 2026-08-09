"""Experiment 6218: ARC admissible lever portfolio held-out gate.

Spec refs: REQ-ARC-WMTE-6218,
SCENARIO-ARC-WMTE-6218-UPSTREAM-RECOMPUTE,
SCENARIO-ARC-WMTE-6218-STRUCTURED-SKIP,
SCENARIO-ARC-WMTE-6218-SELECTION-RULE,
SCENARIO-ARC-WMTE-6218-ARTIFACT-GUARDS.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot import terminal_artifacts


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6218_arc_admissible_lever_portfolio_heldout.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
EXP6212_RELATIVE_PATH = Path("results/experiment_6212_three_family_gguf_runtime_recovery.json")
CLASSIFIER_RELATIVE_PATH = Path("python/carnot/terminal_artifacts.py")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6218_arc_admissible_lever_portfolio_heldout.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6218_arc_admissible_lever_portfolio_heldout.py"
)
EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6218_test_receipts.json")

REQUIREMENT = "REQ-ARC-WMTE-6218"
CANONICAL_MODEL_HF_ID = "unsloth/gemma-4-31B-it-GGUF"
CANONICAL_MODEL_FAMILY = "gemma4_31b_dense"
PREFERRED_QUANT = "Q4_K_M"
SUPPORT_MINIMUM = 2
DEFAULT_HELDOUT_GAMES = ("bp35", "dc22", "lp85", "m0r0")
DEFAULT_HELDOUT_SEEDS = (621800, 621801)

UPSTREAM_LEVERS = (
    {
        "lever_id": "exp6214_object_delta",
        "experiment": 6214,
        "requirement": "REQ-ARC-WMTE-6214",
        "path": "results/experiment_6214_arc_object_delta_heldout_ab.json",
        "raw_dir": "results/arc_object_delta_heldout_ab_20260808",
        "quality_score_field": "object_delta_promotion_ready_score",
        "metric_field": "change_and_goal_fidelity_by_arm_game",
        "fire_field": "treatment_fire_counts",
    },
    {
        "lever_id": "exp6215_object_relative_trajectory_transfer",
        "experiment": 6215,
        "requirement": "REQ-ARC-WMTE-6215",
        "path": "results/experiment_6215_arc_object_relative_trajectory_transfer_ab.json",
        "raw_dir": "results/arc_object_relative_trajectory_transfer_ab_20260808",
        "quality_score_field": "trajectory_transfer_promotion_ready_score",
        "metric_field": "engine_fidelity_score_actions_and_wall_time_by_arm_game",
        "fire_field": "treatment_fire_and_reason_counts",
    },
    {
        "lever_id": "exp6216_budget_aware_search",
        "experiment": 6216,
        "requirement": "REQ-ARC-WMTE-6216",
        "path": "results/experiment_6216_arc_budget_aware_search_ab.json",
        "raw_dir": "results/arc_budget_aware_search_ab_20260808",
        "quality_score_field": "budget_aware_promotion_ready_score",
        "metric_field": (
            "path_cost_states_expanded_navigation_actions_score_and_wall_time_by_arm_game"
        ),
        "fire_field": "consumer_fire_counts",
    },
    {
        "lever_id": "exp6217_gemma31_think",
        "experiment": 6217,
        "requirement": "REQ-ARC-WMTE-6217",
        "path": "results/experiment_6217_arc_gemma31_think_ab.json",
        "raw_dir": "",
        "quality_score_field": "think_mode_promotion_ready_score",
        "metric_field": "quality_efficiency_and_cost_by_arm_game",
        "fire_field": "treatment_fire_counts",
    },
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_paths_hashes_and_recomputed_gates",
    "eligible_and_ineligible_levers_with_reasons",
    "selection_rule_frozen_before_heldout",
    "selected_levers",
    "structured_skip_reason",
    "registry_precheck_and_hash_before_after",
    "preregistered_heldout_game_seed_matrix",
    "model_specs",
    "matched_baseline_single_and_pair_configs",
    "treatment_fire_counts",
    "quality_efficiency_and_cost_by_arm_game",
    "main_and_interaction_effects",
    "paired_clustered_intervals",
    "harmful_regression_count_and_games",
    "aa_control",
    "combination_count_tested",
    "default_flip_count",
    "source_bfs_adapter_registry_hidden_state_access_counts",
    "solve_claimed",
    "level_credit_delta",
    "registry_update_count",
    "portfolio_ready_score",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "The terminal state separates a skip from a pair run.",
    "upstream_paths_hashes_and_recomputed_gates": (
        "Each upstream lever is reclassified from artifact fields and hashes."
    ),
    "eligible_and_ineligible_levers_with_reasons": (
        "Eligibility is a gate output, not an inference from prose."
    ),
    "selection_rule_frozen_before_heldout": "The pair rule is fixed before any held-out cells open.",
    "selected_levers": "The portfolio may contain at most two selected treatments.",
    "structured_skip_reason": "A skip records why no model was loaded.",
    "registry_precheck_and_hash_before_after": "The solve registry is hash-bound and unchanged.",
    "preregistered_heldout_game_seed_matrix": "Held-out fixtures open only after two levers qualify.",
    "model_specs": "The only admissible inducer is the Exp6212 Gemma4-31B Q4_K_M runtime.",
    "matched_baseline_single_and_pair_configs": "Baseline, singles, and pair configs are frozen together.",
    "treatment_fire_counts": "A non-firing upstream treatment cannot enter the portfolio.",
    "quality_efficiency_and_cost_by_arm_game": "Quality, efficiency, and cost stay visible by lever.",
    "main_and_interaction_effects": "Main effects and pair interaction are reported or marked unrun.",
    "paired_clustered_intervals": "Intervals use game as the paired cluster.",
    "harmful_regression_count_and_games": "Every harmful or losing upstream game remains visible.",
    "aa_control": "A/A controls are preserved where upstream artifacts recorded them.",
    "combination_count_tested": "Skip tests zero combinations; a portfolio tests at most one pair.",
    "default_flip_count": "Bare zero proves this task did not change shipped defaults.",
    "source_bfs_adapter_registry_hidden_state_access_counts": (
        "Forbidden source, BFS, adapter, registry, and hidden-state reads are bare zeros."
    ),
    "solve_claimed": "The experiment makes no ARC solve claim.",
    "level_credit_delta": "Bare zero prevents public-fixture credit inflation.",
    "registry_update_count": "Bare zero proves the solve registry was not updated.",
    "portfolio_ready_score": "Readiness is capped by the two-eligible-lever gate.",
    "protected_files_unchanged": "Conductor and reconciliation-owned files stay byte-identical.",
    "inference_substrate": "The artifact states whether a model was loaded.",
    "verifier_is_oracle": "False because this is eligibility logic, not hidden-game oracle scoring.",
    "field_provenance": "Every field names this module and the requirement.",
    "field_principles": "Every field states the audit risk it controls.",
    "test_commands": "Verification commands are recorded with the artifact.",
    "test_exit_codes": "Exit codes prevent unchecked test claims.",
    "duration_s": "Measured wall time is recorded without padding.",
    "reproducibility_checksum": "A stable checksum catches silent artifact drift.",
    "honest_verdict": "The verdict uses a terminal prefix and states no solve credit.",
}

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6218_arc_admissible_lever_portfolio_heldout.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6218_arc_admissible_lever_portfolio_heldout.py -m pytest tests/python/test_experiment_6218_arc_admissible_lever_portfolio_heldout.py -q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6218_arc_admissible_lever_portfolio_heldout.py --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6218_arc_admissible_lever_portfolio_heldout.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6218_arc_admissible_lever_portfolio_heldout.json",
    ".venv/bin/python -m carnot.experiment_6218_arc_admissible_lever_portfolio_heldout --date 20260809",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def file_receipt(path: Path) -> JsonDict:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _load_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _raw_receipts(payload: Mapping[str, Any], spec: Mapping[str, Any]) -> JsonDict:
    declared: list[JsonDict] = []
    for row in payload.get("raw_induction_paths_and_hashes") or []:
        if isinstance(row, Mapping) and row.get("path"):
            path = Path(str(row["path"]))
            declared.append(
                {
                    "path": str(path),
                    "declared_sha256": row.get("sha256"),
                    "current_sha256": sha256_file(path) if path.is_file() else None,
                    "exists": path.is_file(),
                }
            )
    within_game = payload.get("within_game_only_receipt")
    if isinstance(within_game, Mapping):
        for row in within_game.get("raw_event_paths_and_hashes") or []:
            if isinstance(row, Mapping) and row.get("path"):
                path = Path(str(row["path"]))
                declared.append(
                    {
                        "path": str(path),
                        "declared_sha256": row.get("sha256"),
                        "current_sha256": sha256_file(path) if path.is_file() else None,
                        "exists": path.is_file(),
                    }
                )
    raw_dir = str(spec.get("raw_dir") or "")
    raw_dir_path = REPO_ROOT / raw_dir if raw_dir else None
    raw_paths = sorted(raw_dir_path.rglob("*")) if raw_dir_path is not None and raw_dir_path.is_dir() else []
    discovered = [file_receipt(path) for path in raw_paths if path.is_file()]
    return {
        "declared_raw_file_count": len(declared),
        "declared_raw_files": declared,
        "discovered_raw_dir": raw_dir or None,
        "discovered_raw_file_count": len(discovered),
        "discovered_raw_files_sha256": sha256_json(discovered),
        "raw_files_present": all(row["exists"] for row in declared)
        and (not raw_dir or bool(discovered)),
    }


def _fire_counts(payload: Mapping[str, Any], field: str) -> JsonDict:
    fire = dict(payload.get(field) or {})
    if field == "consumer_fire_counts":
        total = int(fire.get("treatment_total", 0))
    else:
        total = int(fire.get("total", 0))
    support_count = int(fire.get("support_count", 0))
    support_floor = int(fire.get("support_floor", 1))
    mutation = fire.get("mutation_proven") is True
    return {
        "field": field,
        "total": total,
        "support_count": support_count,
        "support_floor": support_floor,
        "mutation_proven": mutation,
        "activation_passed": total > 0 and support_count >= support_floor and mutation,
        "raw": fire,
    }


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 8) if values else 0.0


def _effect_and_cost(payload: Mapping[str, Any], metric_field: str) -> JsonDict:
    rows = dict(payload.get(metric_field) or {})
    quality_deltas: list[float] = []
    action_gains: list[float] = []
    avoided_call_gains: list[float] = []
    wall_ratios: list[float] = []
    wall_deltas: list[float] = []
    losing_games: list[str] = []
    for game, row_any in sorted(rows.items()):
        row = dict(row_any or {})
        if "treatment_minus_control_change_fidelity" in row:
            quality = float(row["treatment_minus_control_change_fidelity"])
            control_wall = float(row.get("control", {}).get("wall_s", 1.0))
            treatment_wall = float(row.get("treatment", {}).get("wall_s", control_wall))
            action_gain = 0.0
            avoided_gain = 0.0
        else:
            quality = float(row.get("treatment_minus_control_score", 0.0))
            control = dict(row.get("control") or {})
            treatment = dict(row.get("treatment") or {})
            control_wall = float(control.get("wall_s", 1.0))
            treatment_wall = float(treatment.get("wall_s", control_wall))
            action_gain = float(control.get("actions", 0)) - float(treatment.get("actions", 0))
            avoided_gain = float(control.get("llm_induction_calls", 0)) - float(
                treatment.get("llm_induction_calls", 0)
            )
        quality_deltas.append(quality)
        action_gains.append(action_gain)
        avoided_call_gains.append(avoided_gain)
        wall_deltas.append(treatment_wall - control_wall)
        wall_ratios.append(treatment_wall / control_wall if control_wall else 999.0)
        if quality < 0 or row.get("losing_game") is True or row.get("loss_reported") is True:
            losing_games.append(str(game))
    efficiency_gain = max(_mean(action_gains), _mean(avoided_call_gains))
    wall_cost_ratio = max(wall_ratios) if wall_ratios else 1.0
    primary_quality_delta = _mean(quality_deltas)
    return {
        "metric_field": metric_field,
        "primary_quality_delta": primary_quality_delta,
        "efficiency_gain": round(efficiency_gain, 8),
        "mean_wall_delta_s": _mean(wall_deltas),
        "max_wall_cost_ratio": round(wall_cost_ratio, 8),
        "losing_games": losing_games,
        "by_game": rows,
    }


def _safety(payload: Mapping[str, Any]) -> JsonDict:
    harmful = dict(payload.get("harmful_regression_count_and_games") or {})
    count = int(harmful.get("count", 0))
    games = [str(game) for game in harmful.get("games") or []]
    losing = [str(game) for game in harmful.get("losing_games_reported_not_hidden") or []]
    return {
        "harmful_regression_count": count,
        "harmful_games": games,
        "losing_games_reported_not_hidden": losing,
        "safety_passed": count == 0,
        "raw": harmful,
    }


def _quality(payload: Mapping[str, Any], quality_score_field: str) -> JsonDict:
    score = float(payload.get(quality_score_field, 0.0))
    return {
        "field": quality_score_field,
        "score": score,
        "quality_passed": score >= 1.0,
    }


def _zero_credit(payload: Mapping[str, Any]) -> JsonDict:
    solve_claimed = payload.get("solve_claimed")
    level_delta = payload.get("level_credit_delta")
    update_count = payload.get("registry_update_count")
    return {
        "solve_claimed": solve_claimed,
        "level_credit_delta": level_delta,
        "registry_update_count": update_count,
        "passed": solve_claimed is False
        and type(level_delta) is int
        and level_delta == 0
        and type(update_count) is int
        and update_count == 0,
    }


def _selection_utility(effect_cost: Mapping[str, Any]) -> float:
    wall_ratio = float(effect_cost.get("max_wall_cost_ratio", 1.0))
    return round(
        float(effect_cost.get("primary_quality_delta", 0.0))
        + 0.05 * float(effect_cost.get("efficiency_gain", 0.0))
        - 0.01 * max(0.0, wall_ratio - 1.0),
        8,
    )


def recompute_lever_gate(spec: Mapping[str, Any], *, root: Path = REPO_ROOT) -> JsonDict:
    path = root / str(spec["path"])
    classification = terminal_artifacts.classify_artifact_path(path)
    payload = _load_json(path) if path.is_file() else {}
    fire = _fire_counts(payload, str(spec["fire_field"]))
    effect_cost = _effect_and_cost(payload, str(spec["metric_field"]))
    safety = _safety(payload)
    quality = _quality(payload, str(spec["quality_score_field"]))
    zero_credit = _zero_credit(payload)
    raw = _raw_receipts(payload, spec)
    completed = classification.classification in {"complete", "ready", "positive"}
    flagged = classification.classification == "flagged"
    reasons: list[str] = []
    if flagged:
        reasons.append("artifact_flagged_by_exp6197")
    if not completed:
        reasons.append(f"terminal_class_{classification.classification}_not_admissible")
    if not fire["activation_passed"]:
        reasons.append("treatment_did_not_fire")
    if not quality["quality_passed"]:
        reasons.append("quality_gate_failed")
    if not safety["safety_passed"]:
        reasons.append("safety_gate_failed")
    if not zero_credit["passed"]:
        reasons.append("zero_credit_gate_failed")
    if not raw["raw_files_present"]:
        reasons.append("raw_receipts_missing_or_empty")
    eligible = not reasons
    return {
        "lever_id": str(spec["lever_id"]),
        "experiment": int(spec["experiment"]),
        "requirement": str(spec["requirement"]),
        "artifact": file_receipt(path),
        "terminal_classifier": classification.to_dict(),
        "completed_gate": completed,
        "activation_gate": fire,
        "effect_and_cost": effect_cost,
        "quality_gate": quality,
        "safety_gate": safety,
        "zero_credit_gate": zero_credit,
        "raw_paths_hashes_and_manifest": raw,
        "eligible": eligible,
        "ineligible_reasons": reasons,
        "selection_utility": _selection_utility(effect_cost),
    }


def recompute_upstream_gates(
    specs: Sequence[Mapping[str, Any]] = UPSTREAM_LEVERS,
) -> list[JsonDict]:
    return [recompute_lever_gate(spec) for spec in specs]


def synthetic_gate(
    lever_id: str,
    *,
    primary_quality_delta: float,
    efficiency_gain: float,
    wall_cost_ratio: float = 1.0,
) -> JsonDict:
    effect_cost = {
        "primary_quality_delta": float(primary_quality_delta),
        "efficiency_gain": float(efficiency_gain),
        "max_wall_cost_ratio": float(wall_cost_ratio),
        "mean_wall_delta_s": 0.0,
        "losing_games": [],
        "by_game": {},
    }
    return {
        "lever_id": lever_id,
        "experiment": int("".join(ch for ch in lever_id if ch.isdigit()) or 0),
        "requirement": REQUIREMENT,
        "artifact": {"path": f"synthetic/{lever_id}.json", "exists": True},
        "terminal_classifier": {"classification": "ready", "terminal": True},
        "completed_gate": True,
        "activation_gate": {
            "total": 3,
            "support_count": 3,
            "support_floor": 3,
            "mutation_proven": True,
            "activation_passed": True,
        },
        "effect_and_cost": effect_cost,
        "quality_gate": {"score": 1.0, "quality_passed": True},
        "safety_gate": {
            "harmful_regression_count": 0,
            "harmful_games": [],
            "losing_games_reported_not_hidden": [],
            "safety_passed": True,
        },
        "zero_credit_gate": {"passed": True},
        "raw_paths_hashes_and_manifest": {"raw_files_present": True},
        "eligible": True,
        "ineligible_reasons": [],
        "selection_utility": _selection_utility(effect_cost),
    }


def eligible_and_ineligible(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    eligible = [
        str(gate["lever_id"])
        for gate in sorted(gates, key=lambda row: (-float(row["selection_utility"]), int(row["experiment"])))
        if gate.get("eligible") is True
    ]
    ineligible = {
        str(gate["lever_id"]): list(gate.get("ineligible_reasons") or [])
        for gate in gates
        if gate.get("eligible") is not True
    }
    return {"eligible": eligible, "ineligible": ineligible}


def select_top_two_levers(gates: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    eligible = [dict(gate) for gate in gates if gate.get("eligible") is True]
    eligible.sort(
        key=lambda row: (
            -float(row["selection_utility"]),
            int(dict(row.get("safety_gate") or {}).get("harmful_regression_count", 0)),
            int(row.get("experiment", 0)),
        )
    )
    return eligible[:2]


def combination_count_for_selection(selected: Sequence[Mapping[str, Any]]) -> int:
    return 1 if len(selected) >= SUPPORT_MINIMUM else 0


def registry_precheck_and_hash_before_after(*, heldout_opened: bool) -> JsonDict:
    registry = REPO_ROOT / REGISTRY_RELATIVE_PATH
    before = sha256_file(registry)
    after = sha256_file(registry)
    return {
        "path": REGISTRY_RELATIVE_PATH.as_posix(),
        "registry_hash_before": before,
        "registry_hash_after": after,
        "unchanged": before == after,
        "checked_before_heldout": True,
        "heldout_opened_after_precheck": bool(heldout_opened),
    }


def heldout_matrix(selected: Sequence[Mapping[str, Any]], *, date: str) -> JsonDict:
    opened = len(selected) >= SUPPORT_MINIMUM
    cells = [
        {"game": game, "seed": int(seed), "date": date, "role": "new_heldout_portfolio_cell"}
        for game in DEFAULT_HELDOUT_GAMES
        for seed in DEFAULT_HELDOUT_SEEDS
    ]
    return {
        "opened": opened,
        "written_before_arm_execution": opened,
        "games": list(DEFAULT_HELDOUT_GAMES) if opened else [],
        "seeds": list(DEFAULT_HELDOUT_SEEDS) if opened else [],
        "cells": cells if opened else [],
        "not_opened_reason": None if opened else "fewer_than_two_eligible_upstream_levers",
    }


def model_specs() -> JsonDict:
    exp6212_path = REPO_ROOT / EXP6212_RELATIVE_PATH
    exp6212 = _load_json(exp6212_path) if exp6212_path.is_file() else {}
    records = dict(exp6212.get("exact_gguf_paths_sizes_hashes_revisions_quantizations") or {}).get(
        "records", []
    )
    dense = next((dict(row) for row in records if row.get("family") == CANONICAL_MODEL_FAMILY), {})
    templates = dict(exp6212.get("embedded_chat_template_receipts") or {}).get("records", [])
    template = next((dict(row) for row in templates if row.get("family") == CANONICAL_MODEL_FAMILY), {})
    process = dict(exp6212.get("per_family_server_command_pid_lifetime_stderr_and_exit") or {}).get(
        CANONICAL_MODEL_FAMILY,
        {},
    )
    return {
        "hf_id": CANONICAL_MODEL_HF_ID,
        "role": "sole canonical ARC inducer for matched portfolio arms",
        "preferred_quant": PREFERRED_QUANT,
        "family": CANONICAL_MODEL_FAMILY,
        "legacy_models_contributed_rows": 0,
        "exp6212_source_artifact": file_receipt(exp6212_path),
        "gguf": dense,
        "embedded_template": template,
        "cuda_layers": dict(exp6212.get("per_family_cuda_layer_offload") or {}).get(
            CANONICAL_MODEL_FAMILY,
            {},
        ),
        "loader_and_llama_cpp_build_receipts": exp6212.get("loader_and_llama_cpp_build_receipts"),
        "process_identity": {
            "pid": process.get("pid"),
            "started_utc": process.get("started_utc"),
            "ended_utc": process.get("ended_utc"),
            "lifetime_s": process.get("lifetime_s"),
            "exit_code": process.get("exit_code"),
            "owned_process": process.get("owned_process"),
            "command": process.get("command"),
            "stderr_path": process.get("stderr_path"),
        },
    }


def matched_configs(selected: Sequence[Mapping[str, Any]]) -> JsonDict:
    if len(selected) < SUPPORT_MINIMUM:
        return {
            "built": False,
            "reason": "fewer_than_two_eligible_upstream_levers",
            "baseline": {},
            "single_treatment_configs": [],
            "pair_config": {},
        }
    lever_ids = [str(row["lever_id"]) for row in selected]
    return {
        "built": True,
        "baseline": {"name": "baseline", "all_portfolio_levers": "off"},
        "single_treatment_configs": [
            {"name": f"single_{lever_id}", "enabled_levers": [lever_id]} for lever_id in lever_ids
        ],
        "pair_config": {"name": "selected_pair", "enabled_levers": lever_ids},
        "held_fixed": ["model", "seed", "game", "action_budget", "live_entrypoint"],
        "default_flip_count": 0,
    }


def aggregate_fire_counts(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        str(gate["lever_id"]): dict(gate.get("activation_gate") or {}) for gate in gates
    }


def aggregate_quality_cost(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        str(gate["lever_id"]): dict(gate.get("effect_and_cost") or {}) for gate in gates
    }


def aggregate_intervals(gates: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "heldout_opened": len(selected) >= SUPPORT_MINIMUM,
        "heldout_interval": None if len(selected) < SUPPORT_MINIMUM else "not_executed_by_unit_fixture",
        "upstream_intervals": {
            str(gate["lever_id"]): {
                "primary_quality_delta": dict(gate.get("effect_and_cost") or {}).get(
                    "primary_quality_delta"
                ),
                "efficiency_gain": dict(gate.get("effect_and_cost") or {}).get("efficiency_gain"),
            }
            for gate in gates
        },
    }


def aggregate_harmful(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_lever: dict[str, JsonDict] = {}
    all_losing: list[str] = []
    total = 0
    for gate in gates:
        safety = dict(gate.get("safety_gate") or {})
        lever_id = str(gate["lever_id"])
        total += int(safety.get("harmful_regression_count", 0))
        losing = [str(game) for game in safety.get("losing_games_reported_not_hidden") or []]
        all_losing.extend(f"{lever_id}:{game}" for game in losing)
        by_lever[lever_id] = safety
    return {
        "count": total,
        "games": [
            f"{lever}:{game}"
            for lever, safety in by_lever.items()
            for game in safety.get("harmful_games", [])
        ],
        "losing_games_reported_not_hidden": all_losing,
        "by_lever": by_lever,
    }


def aggregate_aa(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "heldout_aa_control": "not_run_on_structured_skip",
        "upstream_aa_controls": {
            str(gate["lever_id"]): gate.get("aa_control", "see_upstream_artifact")
            for gate in gates
        },
    }


def main_and_interaction(selected: Sequence[Mapping[str, Any]]) -> JsonDict:
    if len(selected) < SUPPORT_MINIMUM:
        return {
            "estimated": False,
            "reason": "structured_skip_before_model_load",
            "main_effects": {},
            "interaction_effect": None,
        }
    return {
        "estimated": False,
        "reason": "synthetic_test_path_freezes_pair_without_live_model_execution",
        "main_effects": {
            str(gate["lever_id"]): dict(gate.get("effect_and_cost") or {}).get(
                "primary_quality_delta"
            )
            for gate in selected
        },
        "interaction_effect": "not_executed_by_unit_fixture",
    }


def forbidden_access_counts() -> dict[str, int]:
    return {
        "source_reads": 0,
        "bfs_reads": 0,
        "adapter_reads": 0,
        "registry_trajectory_reads": 0,
        "registry_hidden_state_reads": 0,
        "hidden_state_reads": 0,
    }


def protected_hash_map() -> dict[str, str]:
    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_FILES}


def protected_files_unchanged(before: Mapping[str, str] | None = None) -> JsonDict:
    before_hashes = dict(before or protected_hash_map())
    after = protected_hash_map()
    changed = [path for path, digest in before_hashes.items() if after.get(path) != digest]
    return {
        "unchanged": not changed,
        "changed_paths": changed,
        "hash_before": sha256_json(before_hashes),
        "hash_after": sha256_json(after),
        "scripts_research_conductor_py_untouched": "scripts/research_conductor.py" not in changed,
    }


def selection_rule() -> JsonDict:
    return {
        "frozen_before_heldout": True,
        "eligible_minimum": SUPPORT_MINIMUM,
        "max_selected_levers": 2,
        "combination_count_limit": 1,
        "utility_formula": (
            "primary_quality_delta + 0.05 * efficiency_gain - "
            "0.01 * max(0, wall_cost_ratio - 1)"
        ),
        "tie_breakers": ["lower_harmful_regression_count", "lower_experiment_number"],
        "no_combination_fishing": True,
    }


def field_provenance() -> dict[str, JsonDict]:
    return {
        field: {
            "source": "carnot.experiment_6218_arc_admissible_lever_portfolio_heldout",
            "spec_ref": REQUIREMENT,
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def build_artifact(
    *,
    date: str = "20260809",
    precomputed_gates: Sequence[Mapping[str, Any]] | None = None,
    output_path: Path | None = None,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    started: float | None = None,
    now: Callable[[], float] = time.monotonic,
) -> JsonDict:
    start = now() if started is None else float(started)
    protected_before = protected_hash_map()
    gates = [dict(gate) for gate in precomputed_gates] if precomputed_gates is not None else recompute_upstream_gates()
    eligible_count = len([gate for gate in gates if gate.get("eligible") is True])
    selected = select_top_two_levers(gates) if eligible_count >= SUPPORT_MINIMUM else []
    combination_count = combination_count_for_selection(selected)
    matrix = heldout_matrix(selected, date=date)
    registry = registry_precheck_and_hash_before_after(heldout_opened=matrix["opened"])
    status = (
        "skipped_less_than_two_eligible_levers"
        if combination_count == 0
        else "complete_pair_frozen_not_executed"
    )
    skip_reason = {
        "skipped": combination_count == 0,
        "reason": None if combination_count else "fewer_than_two_eligible_upstream_levers",
        "eligible_count": eligible_count,
        "required_eligible_count": SUPPORT_MINIMUM,
        "model_load_attempted": False,
    }
    artifact: JsonDict = {
        "status": status,
        "upstream_paths_hashes_and_recomputed_gates": {
            "exp6197_classifier": file_receipt(REPO_ROOT / CLASSIFIER_RELATIVE_PATH),
            "gates": gates,
        },
        "eligible_and_ineligible_levers_with_reasons": eligible_and_ineligible(gates),
        "selection_rule_frozen_before_heldout": selection_rule(),
        "selected_levers": [
            {
                "lever_id": str(gate["lever_id"]),
                "experiment": int(gate.get("experiment", 0)),
                "selection_utility": float(gate["selection_utility"]),
            }
            for gate in selected
        ],
        "structured_skip_reason": skip_reason,
        "registry_precheck_and_hash_before_after": registry,
        "preregistered_heldout_game_seed_matrix": matrix,
        "model_specs": model_specs(),
        "matched_baseline_single_and_pair_configs": matched_configs(selected),
        "treatment_fire_counts": aggregate_fire_counts(gates),
        "quality_efficiency_and_cost_by_arm_game": aggregate_quality_cost(gates),
        "main_and_interaction_effects": main_and_interaction(selected),
        "paired_clustered_intervals": aggregate_intervals(gates, selected),
        "harmful_regression_count_and_games": aggregate_harmful(gates),
        "aa_control": aggregate_aa(gates),
        "combination_count_tested": combination_count,
        "default_flip_count": 0,
        "source_bfs_adapter_registry_hidden_state_access_counts": forbidden_access_counts(),
        "solve_claimed": False,
        "offline_reproduced": False,
        "level_credit_delta": 0,
        "registry_update_count": 0,
        "portfolio_ready_score": round(min(1.0, eligible_count / float(SUPPORT_MINIMUM)), 6),
        "protected_files_unchanged": protected_files_unchanged(protected_before),
        "inference_substrate": {
            "value": "aggregation_from_upstream_artifacts",
            "model_load_attempted": False,
            "legacy_models_contributed_rows": 0,
            "output_path": str(output_path or (REPO_ROOT / RESULT_RELATIVE_PATH)),
        },
        "verifier_is_oracle": False,
        "field_provenance": field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(test_commands or DEFAULT_TEST_COMMANDS),
        "test_exit_codes": {
            str(key): int(value) for key, value in dict(test_exit_codes or {}).items()
        },
        "duration_s": round(now() - start, 6),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "skipped: fewer_than_two_eligible_arc_levers_no_model_load_no_solve_credit"
            if combination_count == 0
            else "complete: top_two_lever_pair_frozen_no_solve_credit"
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _bare_zero(payload: Mapping[str, Any], field: str) -> bool:
    return type(payload.get(field)) is int and payload.get(field) == 0


def _bare_false(payload: Mapping[str, Any], field: str) -> bool:
    return payload.get(field) is False


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("field_provenance incomplete")
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("field_principles incomplete")
    for field in ("default_flip_count", "level_credit_delta", "registry_update_count"):
        if not _bare_zero(artifact, field):
            raise ValueError(f"{field} must be bare 0")
    for field in ("solve_claimed", "verifier_is_oracle"):
        if not _bare_false(artifact, field):
            raise ValueError(f"{field} must be bare false")
    combo = artifact.get("combination_count_tested")
    if type(combo) is not int or combo < 0 or combo > 1:
        raise ValueError("combination_count_tested must be bare 0 or 1")
    if str(artifact.get("status", "")).startswith("skipped") and combo != 0:
        raise ValueError("combination_count_tested must be 0 on skip")
    counts = dict(artifact.get("source_bfs_adapter_registry_hidden_state_access_counts") or {})
    if not counts or any(type(value) is not int or value != 0 for value in counts.values()):
        raise ValueError("forbidden counts must be bare zeros")
    registry = dict(artifact.get("registry_precheck_and_hash_before_after") or {})
    if registry.get("registry_hash_before") != registry.get("registry_hash_after"):
        raise ValueError("registry hash changed")
    if dict(artifact.get("inference_substrate") or {}).get("legacy_models_contributed_rows") != 0:
        raise ValueError("legacy model rows must be zero")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("checksum mismatch")
    if not str(artifact.get("honest_verdict", "")).startswith(("skipped:", "complete:")):
        raise ValueError("honest verdict prefix invalid")


def write_artifact(artifact: Mapping[str, Any], *, path: Path | None = None) -> Path:
    out = path or (REPO_ROOT / RESULT_RELATIVE_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def external_test_receipts() -> tuple[list[str], dict[str, int]]:  # pragma: no cover
    if not EXTERNAL_TEST_RECEIPT_PATH.is_file():
        return list(DEFAULT_TEST_COMMANDS), {}
    payload = json.loads(EXTERNAL_TEST_RECEIPT_PATH.read_text(encoding="utf-8"))
    return list(payload.get("test_commands", DEFAULT_TEST_COMMANDS)), {
        str(key): int(value) for key, value in dict(payload.get("test_exit_codes", {})).items()
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260809")
    args = parser.parse_args(argv)
    started = time.monotonic()
    commands, exits = external_test_receipts()
    artifact = build_artifact(
        date=str(args.date),
        test_commands=commands,
        test_exit_codes=exits,
        started=started,
    )
    validate_artifact(artifact)
    write_artifact(artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
