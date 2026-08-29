"""Run the paired ARC object-table fetch-on-demand experiment.

The science rows compare the current inline object table with the same prompt
without that table. Both arms keep the production selfparse tools. The module
records induction quality on held transitions. It never plays a scored game and
never claims a level solve.

Spec refs: REQ-ARC-WMTE-6753 and SCENARIO-ARC-WMTE-6753-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

from carnot.experiment_6752_arc_code_carrying_tool_preflight import (
    MODEL_SPECS as PREFLIGHT_MODEL_SPECS,
    canonical_json,
    nvidia_smi_inventory,
    sha256_file,
    sha256_json,
    sha256_text,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "results/experiment_6753_object_table_fetch_on_demand_ab.json"
PREFLIGHT_PATH = REPO_ROOT / "results/experiment_6752_arc_code_carrying_tool_preflight.json"
PRIOR_PATH = (
    REPO_ROOT / "results/outer_loop_arc_object_perception_heldout_ab_change_fidelity_20260801.json"
)
REGISTRY_PATH = REPO_ROOT / "ops/arc_solve_registry.yaml"
CODEX_PATH = REPO_ROOT / "CODEX.md"
SCHEMA = "carnot.experiment_6753.object_table_fetch_on_demand_ab.v1"
RUN_DATE = "20260829"
CONTEXT_REQUESTED = 32_768
NONINFERIORITY_MARGIN = 0.259909
BOOTSTRAP_SEED = 6_753
BOOTSTRAP_RESAMPLES = 20_000
BASELINE_ARM = "table_inline"
TREATMENT_ARM = "fetch_on_demand"
ARMS = (BASELINE_ARM, TREATMENT_ARM)
INFERENCE_SUBSTRATE = "production E3AgentPolicy local CUDA GGUF"
PRODUCTION_ROUTE = (
    "make_carnot_agent/E3AgentPolicy._proposer/LocalGGUFProposer/"
    "induce_with_tool_loop/selfparse/dispatch_tool"
)

GAME_IDS = (
    "ls20",
    "s5i5",
    "tu93",
    "cn04",
    "m0r0",
    "sk48",
    "ar25",
    "tr87",
    "g50t",
    "re86",
    "bp35",
    "sb26",
    "lf52",
    "su15",
    "lp85",
    "cd82",
    "wa30",
    "sc25",
    "tn36",
    "ka59",
)
SEEDS = (6_100, 6_101, 6_102)

MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        **dict(PREFLIGHT_MODEL_SPECS[0]),
        "role": "immutable_scored_arc_generator",
    },
    {
        **dict(PREFLIGHT_MODEL_SPECS[1]),
        "role": "mandated_same_task_transport_canary",
    },
)

SCIENCE_BUDGETS: JsonDict = {
    "tool_turns": 12,
    "think_tokens_per_turn": 3_072,
    "completion_tokens_per_turn": 4_096,
    "timeout_s": 2_400,
    "early_stop_after_non_improving": 2,
    "stall_turn_cap": 0,
    "force_engine_turn": 3,
}
SIDECAR_BUDGETS: JsonDict = {
    "tool_turns": 1,
    "think_tokens_per_turn": 512,
    "completion_tokens_per_turn": 1_024,
    "timeout_s": 1_200,
    "early_stop_after_non_improving": 2,
    "stall_turn_cap": 0,
    "force_engine_turn": 1,
}

VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}

FIELD_PRINCIPLES: JsonDict = {
    "schema": "A versioned shape lets readers reject incompatible evidence.",
    "experiment": "The experiment number binds the result to REQ-ARC-WMTE-6753.",
    "title": "The title states the paired non-inferiority question.",
    "run_date": "The requested planning date fixes the evidence period.",
    "status": "The status distinguishes a complete, partial, or blocked run.",
    "field_principles": "Every top-level field states why it exists.",
    "inference_substrate": "The substrate excludes remote, CPU, and helper-only substitutes.",
    "duration_s": "A monotonic interval shows real elapsed work.",
    "random_seed": "All arm seeds and the analysis seed make random choices repeatable.",
    "reproducibility_checksum": "One digest binds design, prompts, provenance, and rows.",
    "models_used": "Exact model IDs, roles, paths, and hashes prevent substitution.",
    "live_model_invoked": "The flag distinguishes live evidence from a dry artifact.",
    "context_requested": "Every worker owns the fixed 32768-cell context request.",
    "gpu_receipts": "Per-load device, layer, and memory facts prove CUDA inference.",
    "rows": "The full fixed denominator keeps failures and sidecars visible.",
    "frozen_design": "The design records choices made before live rows run.",
    "source_receipts": "Input hashes bind the preflight, prior evidence, and registry.",
    "preconditions_checked": "Fail-closed host and evidence checks explain admission.",
    "mean_prompt_token_savings": "The paired baseline-minus-treatment result measures savings.",
    "fetch_rate": "The row-derived rate reports whether treatment used find_objects.",
    "useful_fetch_rate": "The strict rate needs returned evidence and a later valid engine.",
    "change_fidelity_delta": "Treatment-minus-baseline held change fidelity is the main effect.",
    "change_fidelity_ci95": "A game-clustered interval supports the non-inferiority gate.",
    "transition_utility_delta": "The paired net changed-cell utility is a second quality view.",
    "noninferiority_margin": "The frozen prior within-arm mean spread is the allowed loss.",
    "harmful_regressions": "Games below the frozen margin remain visible.",
    "paired_analysis": "Per-game and per-seed details make the reducer reproducible.",
    "adoption_gate_conditions": "Named booleans make adoption independent of completion.",
    "adoption_gate_passed": "Adoption needs completion, savings, and non-inferiority.",
    "object_table_ab_completed": "Completion needs every valid science and sidecar row.",
    "solve_claim": "False prevents induction evidence from becoming a level-solve claim.",
    "gate_check_summary": "Blocked artifacts name the failed check and observed value.",
    "verdict_class": "A closed class supports machine-readable terminal handling.",
    "honest_verdict": "The terminal prefix states the owned result without inflation.",
}


def row_checksum(row: Mapping[str, Any]) -> str:
    """Hash a row without its self-referential checksum field."""

    return sha256_json({key: value for key, value in row.items() if key != "row_sha256"})


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash an artifact without its self-referential checksum field."""

    return sha256_json(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def frozen_design() -> JsonDict:
    """Return the complete design that is fixed before model inference."""

    return {
        "requirement": "REQ-ARC-WMTE-6753",
        "question": (
            "Is table-absent plus production find_objects non-inferior to the inline table "
            "with positive realized prompt-token savings?"
        ),
        "game_ids": list(GAME_IDS),
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "arm_order_rule": "alternate by (game_index + seed_index) parity",
        "context_requested": CONTEXT_REQUESTED,
        "science_budgets": deepcopy(SCIENCE_BUDGETS),
        "sidecar_budgets": deepcopy(SIDECAR_BUDGETS),
        "generator_configuration": {
            "science_model": MODEL_SPECS[0]["model_id"],
            "sidecar_model": MODEL_SPECS[1]["model_id"],
            "mtp": False,
            "kv_quant": "q8_0",
            "tool_transport": "selfparse",
            "production_route": PRODUCTION_ROUTE,
        },
        "pairing_unit": "game_and_seed; average seeds inside each game before inference",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "noninferiority_source_field": "NOISE_FLOOR_within_arm_replicate_spread.mean_spread",
        "adoption_conditions": [
            "all planned rows have valid provenance",
            "mean_prompt_token_savings > 0",
            "change_fidelity_ci95.lower >= -noninferiority_margin",
            "solve_claim is false",
        ],
        "solve_target": None,
        "solve_claim": False,
    }


def science_plan() -> list[JsonDict]:
    """Return 120 ordered Qwen3.8 rows with alternating within-pair order."""

    rows: list[JsonDict] = []
    for game_index, game in enumerate(GAME_IDS):
        for seed_index, seed in enumerate(SEEDS):
            order = ARMS if (game_index + seed_index) % 2 == 0 else tuple(reversed(ARMS))
            for arm in order:
                rows.append(
                    {
                        "row_id": f"science:{game}:{seed}:{arm}",
                        "row_kind": "science",
                        "game": game,
                        "seed": seed,
                        "arm": arm,
                        "arm_order": list(order),
                        "model_id": MODEL_SPECS[0]["model_id"],
                    }
                )
    return rows


def sidecar_plan() -> list[JsonDict]:
    """Return two bounded Qwen3.6 rows for the first frozen game and seed."""

    return [
        {
            "row_id": f"sidecar:{GAME_IDS[0]}:{SEEDS[0]}:{arm}",
            "row_kind": "transport_sidecar",
            "game": GAME_IDS[0],
            "seed": SEEDS[0],
            "arm": arm,
            "arm_order": list(ARMS),
            "model_id": MODEL_SPECS[1]["model_id"],
        }
        for arm in ARMS
    ]


def _budgets_for(planned: Mapping[str, Any]) -> Mapping[str, Any]:
    """Select the pre-registered budget for one planned row."""

    return SIDECAR_BUDGETS if planned.get("row_kind") == "transport_sidecar" else SCIENCE_BUDGETS


def worker_environment(
    base: Mapping[str, str], model: Mapping[str, Any], planned: Mapping[str, Any]
) -> dict[str, str]:
    """Build the owned environment before constructing a production proposer."""

    budgets = _budgets_for(planned)
    env = dict(base)
    env.update(
        {
            "CARNOT_ARC_OBJECT_PERCEPTION": (
                "1" if planned["arm"] == BASELINE_ARM else "0"
            ),
            "CARNOT_ARC_INDUCE_N_CTX": str(CONTEXT_REQUESTED),
            "CARNOT_ARC_INDUCE_TOOL_LOOP": "selfparse",
            "CARNOT_ARC_INDUCE_TOOL_TURNS": str(budgets["tool_turns"]),
            "CARNOT_ARC_INDUCE_TOOL_THINK_BUDGET": str(
                budgets["think_tokens_per_turn"]
            ),
            "CARNOT_ARC_INDUCE_MAX_TOKENS": str(budgets["completion_tokens_per_turn"]),
            "CARNOT_ARC_INDUCE_TIMEOUT": str(budgets["timeout_s"]),
            "CARNOT_ARC_INDUCE_TOOL_EARLY_STOP": str(
                budgets["early_stop_after_non_improving"]
            ),
            "CARNOT_ARC_INDUCE_TOOL_STALL_TURNS": str(budgets["stall_turn_cap"]),
            "CARNOT_ARC_INDUCE_TOOL_FORCE_ENGINE_TURN": str(budgets["force_engine_turn"]),
            "CARNOT_ARC_GENERATOR_SEED": str(planned["seed"]),
            "CARNOT_ARC_GENERATOR_CUDA_GPU": str(model.get("device_index", 0)),
            "CARNOT_ARC_GENERATOR_REQUIRE_CUDA": "1",
            "CARNOT_ARC_MTP": "0",
            "CARNOT_ARC_KV_QUANT": "q8_0",
            "CARNOT_ARC_GGUF_PATH": str(model["model_path"]),
            "PYTHONPATH": str(REPO_ROOT / "python")
            + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        }
    )
    return env


def prompt_isolation_receipt(
    inline_prompt: str,
    absent_prompt: str,
    inline_object_table: str,
    tool_schemas: str,
) -> JsonDict:
    """Prove that removing the table is the only prompt-byte difference."""

    if inline_prompt.count(inline_object_table) != 1:
        raise ValueError("inline object table must occur exactly once")
    if inline_prompt.count(tool_schemas) != 1 or absent_prompt.count(tool_schemas) != 1:
        raise ValueError("tool schemas must occur exactly once in both prompts")
    expected_absent = inline_prompt.replace(inline_object_table, "", 1)
    if expected_absent != absent_prompt:
        raise ValueError("treatment changed more than the inline object table")
    return {
        "only_object_table_removed": True,
        "inline_prompt_sha256": sha256_text(inline_prompt),
        "absent_prompt_sha256": sha256_text(absent_prompt),
        "inline_object_table_sha256": sha256_text(inline_object_table),
        "inline_object_table_chars": len(inline_object_table),
        "tool_schema_sha256": sha256_text(tool_schemas),
    }


def fetch_accounting(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count strict useful fetches from retained production tool events."""

    attempts = [event for event in events if event.get("parsed_tool") == "find_objects"]
    successes = [
        event
        for event in attempts
        if isinstance(event.get("dispatch_result"), Mapping)
        and event["dispatch_result"].get("ok") is True
    ]
    entered = False
    useful = 0
    for fetch in successes:
        if not fetch.get("bounded_response"):
            continue
        later_submissions = [
            event
            for event in events
            if int(event.get("turn", -1)) > int(fetch.get("turn", -1))
            and event.get("parsed_tool") == "run_engine_on_transitions"
        ]
        if later_submissions:
            entered = True
        if any(
            isinstance(event.get("dispatch_result"), Mapping)
            and event["dispatch_result"].get("ok") is True
            for event in later_submissions
        ):
            useful += 1
    return {
        "find_objects_attempts": len(attempts),
        "find_objects_successes": len(successes),
        "fetched_evidence_entered_later_reasoning": entered,
        "useful_fetches": useful,
    }


def transition_result(
    *,
    n_heldout: int,
    true_changed_cells: int,
    correct_changed_cells: int,
    spurious_changed_cells: int,
) -> JsonDict:
    """Return raw held-transition counts and their fixed net-change utility."""

    denominator = max(1, int(true_changed_cells))
    utility = (int(correct_changed_cells) - int(spurious_changed_cells)) / denominator
    raw = {
        "n_heldout": int(n_heldout),
        "true_changed_cells": int(true_changed_cells),
        "correct_changed_cells": int(correct_changed_cells),
        "spurious_changed_cells": int(spurious_changed_cells),
    }
    return {"transition_result": raw, "transition_utility": round(float(utility), 9)}


def _bootstrap_ci(values: Sequence[float], n_resamples: int) -> list[float]:
    """Return a fixed-seed percentile interval over independent game means."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 1:
        value = float(array[0])
        return [value, value]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(array, size=(int(n_resamples), array.size), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def paired_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    game_ids: Sequence[str] = GAME_IDS,
    seeds: Sequence[int] = SEEDS,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> JsonDict:
    """Pair exact Qwen3.8 game-seed arms and exclude every sidecar row."""

    indexed: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("row_kind") != "science":
            continue
        key = (str(row.get("game")), int(row.get("seed", -1)), str(row.get("arm")))
        if key in indexed:
            raise ValueError(f"duplicate science row: {key}")
        indexed[key] = row
    pair_rows: list[JsonDict] = []
    per_game: JsonDict = {}
    for game in game_ids:
        game_pairs = []
        for seed in seeds:
            baseline_key = (str(game), int(seed), BASELINE_ARM)
            treatment_key = (str(game), int(seed), TREATMENT_ARM)
            if baseline_key not in indexed:
                raise ValueError(f"missing science row: {baseline_key}")
            if treatment_key not in indexed:
                raise ValueError(f"missing science row: {treatment_key}")
            baseline = indexed[baseline_key]
            treatment = indexed[treatment_key]
            pair = {
                "game": str(game),
                "seed": int(seed),
                "baseline_row_id": baseline.get("row_id"),
                "treatment_row_id": treatment.get("row_id"),
                "change_fidelity_delta": float(treatment["change_fidelity"])
                - float(baseline["change_fidelity"]),
                "transition_utility_delta": float(treatment["transition_utility"])
                - float(baseline["transition_utility"]),
                "prompt_token_savings": int(baseline["prompt_tokens"])
                - int(treatment["prompt_tokens"]),
                "treatment_fetched": int(treatment.get("find_objects_attempts", 0)) > 0,
                "treatment_useful_fetch": int(treatment.get("useful_fetches", 0)) > 0,
            }
            pair_rows.append(pair)
            game_pairs.append(pair)
        per_game[str(game)] = {
            "n_seed_pairs": len(game_pairs),
            "change_fidelity_delta": float(
                np.mean([pair["change_fidelity_delta"] for pair in game_pairs])
            ),
            "transition_utility_delta": float(
                np.mean([pair["transition_utility_delta"] for pair in game_pairs])
            ),
            "prompt_token_savings": float(
                np.mean([pair["prompt_token_savings"] for pair in game_pairs])
            ),
        }
    game_deltas = [per_game[str(game)]["change_fidelity_delta"] for game in game_ids]
    interval = _bootstrap_ci(game_deltas, n_resamples)
    treatment_pairs = len(pair_rows)
    harmful = [
        {
            "game": str(game),
            "change_fidelity_delta": per_game[str(game)]["change_fidelity_delta"],
        }
        for game in game_ids
        if per_game[str(game)]["change_fidelity_delta"] < -NONINFERIORITY_MARGIN
    ]
    return {
        "n_games_paired": len(game_ids),
        "n_seed_pairs": len(pair_rows),
        "per_seed_pair": pair_rows,
        "per_game": per_game,
        "change_fidelity_delta": float(np.mean(game_deltas)),
        "change_fidelity_ci95": interval,
        "transition_utility_delta": float(
            np.mean([per_game[str(game)]["transition_utility_delta"] for game in game_ids])
        ),
        "mean_prompt_token_savings": float(
            np.mean([pair["prompt_token_savings"] for pair in pair_rows])
        ),
        "fetch_rate": sum(pair["treatment_fetched"] for pair in pair_rows) / treatment_pairs,
        "useful_fetch_rate": sum(pair["treatment_useful_fetch"] for pair in pair_rows)
        / treatment_pairs,
        "harmful_regressions": harmful,
        "noninferiority_passed": interval[0] >= -NONINFERIORITY_MARGIN,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": int(n_resamples),
    }


def row_evidence_errors(row: Mapping[str, Any]) -> list[str]:
    """Name provenance defects that make a live row invalid."""

    errors: list[str] = []
    kind = row.get("row_kind")
    expected_model = MODEL_SPECS[0]["model_id"] if kind == "science" else MODEL_SPECS[1]["model_id"]
    if row.get("model_id") != expected_model:
        errors.append("model_id")
    if row.get("production_route") != PRODUCTION_ROUTE:
        errors.append("production_route")
    if row.get("context_requested") != CONTEXT_REQUESTED:
        errors.append("context_requested")
    if int(row.get("context_observed_by_model") or 0) < CONTEXT_REQUESTED:
        errors.append("context_observed_by_model")
    gpu = row.get("gpu_receipt") if isinstance(row.get("gpu_receipt"), Mapping) else {}
    layers = gpu.get("gpu_layers") if isinstance(gpu.get("gpu_layers"), Mapping) else {}
    if int(layers.get("offloaded") or 0) <= 0:
        errors.append("cuda_offload")
    if row.get("live_model_invoked") is not True:
        errors.append("live_model_invoked")
    if not str(row.get("raw_prompt_sha256") or "").startswith("sha256:"):
        errors.append("raw_prompt_sha256")
    if int(row.get("prompt_tokens") or 0) <= 0:
        errors.append("prompt_tokens")
    if row.get("solve_claim") is not False:
        errors.append("solve_claim")
    if row.get("failure_class") is not None:
        errors.append("failure_class")
    if kind == "science":
        if not isinstance(row.get("change_fidelity"), (int, float)):
            errors.append("change_fidelity")
        if not isinstance(row.get("transition_utility"), (int, float)):
            errors.append("transition_utility")
    if row.get("row_sha256") != row_checksum(row):
        errors.append("row_sha256")
    return errors


def completion_and_adoption(
    rows: Sequence[Mapping[str, Any]],
    *,
    science_plan_rows: Sequence[Mapping[str, Any]] | None = None,
    sidecar_plan_rows: Sequence[Mapping[str, Any]] | None = None,
    game_ids: Sequence[str] = GAME_IDS,
    seeds: Sequence[int] = SEEDS,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> JsonDict:
    """Reduce completion and adoption as separate fail-closed decisions."""

    planned = list(science_plan_rows or science_plan()) + list(sidecar_plan_rows or sidecar_plan())
    expected_ids = [str(item["row_id"]) for item in planned]
    by_id: dict[str, Mapping[str, Any]] = {}
    duplicate_ids: list[str] = []
    for row in rows:
        row_id = str(row.get("row_id"))
        if row_id in by_id:
            duplicate_ids.append(row_id)
        by_id[row_id] = row
    missing_ids = [row_id for row_id in expected_ids if row_id not in by_id]
    unexpected_ids = [row_id for row_id in by_id if row_id not in set(expected_ids)]
    invalid_rows = {
        row_id: row_evidence_errors(by_id[row_id])
        for row_id in expected_ids
        if row_id in by_id and row_evidence_errors(by_id[row_id])
    }
    completed = not missing_ids and not unexpected_ids and not duplicate_ids and not invalid_rows
    if completed:
        analysis = paired_statistics(
            rows, game_ids=game_ids, seeds=seeds, n_resamples=n_resamples
        )
        positive_savings = analysis["mean_prompt_token_savings"] > 0
        noninferior = analysis["noninferiority_passed"] is True
    else:
        analysis = {
            "n_games_paired": 0,
            "n_seed_pairs": 0,
            "per_seed_pair": [],
            "per_game": {},
            "change_fidelity_delta": None,
            "change_fidelity_ci95": None,
            "transition_utility_delta": None,
            "mean_prompt_token_savings": None,
            "fetch_rate": None,
            "useful_fetch_rate": None,
            "harmful_regressions": [],
            "noninferiority_passed": False,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": int(n_resamples),
        }
        positive_savings = False
        noninferior = False
    conditions = {
        "all_planned_rows_valid": completed,
        "positive_prompt_token_savings": positive_savings,
        "change_fidelity_noninferior": noninferior,
        "solve_claim_false": all(row.get("solve_claim") is False for row in rows),
    }
    return {
        **analysis,
        "row_completion_receipt": {
            "planned": len(expected_ids),
            "observed": len(rows),
            "missing_row_ids": missing_ids,
            "unexpected_row_ids": unexpected_ids,
            "duplicate_row_ids": duplicate_ids,
            "invalid_rows": invalid_rows,
        },
        "adoption_gate_conditions": conditions,
        "adoption_gate_passed": all(conditions.values()),
        "object_table_ab_completed": completed,
        "solve_claim": False,
    }


def _load_json(path: Path) -> JsonDict:
    """Read one required JSON object from disk."""

    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def resolve_model_specs() -> list[JsonDict]:
    """Reuse the exact model paths and hashes that Exp6752 proved live."""

    preflight = _load_json(PREFLIGHT_PATH)
    by_id = {row.get("model_id"): row for row in preflight.get("models_used", [])}
    resolved = []
    for spec in MODEL_SPECS:
        prior = by_id.get(spec["model_id"], {})
        path = Path(str(prior.get("model_path") or ""))
        present = path.is_file() and path.name == spec["filename"]
        current_hash = sha256_file(path) if present else None
        resolved.append(
            {
                **dict(spec),
                "resolved": present,
                "model_path": str(path),
                "model_size_bytes": path.stat().st_size if present else 0,
                "model_sha256": current_hash,
                "exp6752_model_sha256": prior.get("model_sha256"),
                "required_vram_mb": prior.get("required_vram_mb"),
            }
        )
    return resolved


def live_preflight(models: list[JsonDict]) -> JsonDict:
    """Check all frozen evidence, exact models, CUDA, context, and no-solve gates."""

    checks: list[JsonDict] = []
    source_receipts: JsonDict = {}
    try:
        preflight = _load_json(PREFLIGHT_PATH)
        prior = _load_json(PRIOR_PATH)
        registry_text = REGISTRY_PATH.read_text()
        codex_text = CODEX_PATH.read_text()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        checks.append(
            {
                "check": "required_source_artifacts",
                "expected": True,
                "observed": f"{type(exc).__name__}: {exc}",
                "passed": False,
            }
        )
        return {"all_passed": False, "checks": checks, "source_receipts": source_receipts}
    source_receipts = {
        "exp6752": {"path": str(PREFLIGHT_PATH), "sha256": sha256_file(PREFLIGHT_PATH)},
        "prior_20260801": {"path": str(PRIOR_PATH), "sha256": sha256_file(PRIOR_PATH)},
        "solve_registry": {"path": str(REGISTRY_PATH), "sha256": sha256_text(registry_text)},
        "codex_instructions": {"path": str(CODEX_PATH), "sha256": sha256_text(codex_text)},
    }
    checks.append(
        {
            "check": "exp6752_ready",
            "expected": True,
            "observed": preflight.get("arc_context_tool_preflight_ready"),
            "passed": preflight.get("arc_context_tool_preflight_ready") is True,
        }
    )
    checks.append(
        {
            "check": "exp6752_owned_32k_cuda",
            "expected": {"context": CONTEXT_REQUESTED, "cuda_offload": True},
            "observed": [
                {
                    "model_id": row.get("model_id"),
                    "context": row.get("context_observed_by_model"),
                    "offloaded": (row.get("gpu_layers") or {}).get("offloaded"),
                }
                for row in preflight.get("rows", [])
            ],
            "passed": len(preflight.get("rows", [])) == 2
            and all(
                int(row.get("context_observed_by_model") or 0) >= CONTEXT_REQUESTED
                and int((row.get("gpu_layers") or {}).get("offloaded") or 0) > 0
                for row in preflight.get("rows", [])
            ),
        }
    )
    prior_content = prior.get("preregistration", {}).get("content", {})
    observed_roster = prior_content.get("roster")
    observed_seeds = [int(prior.get("random_seed", -1)) + index for index in range(3)]
    observed_margin = prior.get("NOISE_FLOOR_within_arm_replicate_spread", {}).get(
        "mean_spread"
    )
    checks.extend(
        [
            {
                "check": "frozen_games",
                "expected": list(GAME_IDS),
                "observed": observed_roster,
                "passed": observed_roster == list(GAME_IDS),
            },
            {
                "check": "frozen_seeds",
                "expected": list(SEEDS),
                "observed": observed_seeds,
                "passed": observed_seeds == list(SEEDS),
            },
            {
                "check": "frozen_within_arm_noise_floor",
                "expected": NONINFERIORITY_MARGIN,
                "observed": observed_margin,
                "passed": observed_margin == NONINFERIORITY_MARGIN,
            },
            {
                "check": "task_owned_context",
                "expected": CONTEXT_REQUESTED,
                "observed": frozen_design()["context_requested"],
                "passed": frozen_design()["context_requested"] == CONTEXT_REQUESTED,
            },
        ]
    )
    try:
        from llama_cpp import llama_cpp

        cuda_offload: Any = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception as exc:  # noqa: BLE001 - observed import failure belongs in the artifact
        cuda_offload = f"{type(exc).__name__}: {exc}"
    checks.append(
        {
            "check": "llama_cpp_cuda_offload",
            "expected": True,
            "observed": cuda_offload,
            "passed": cuda_offload is True,
        }
    )
    preflight_models = {row.get("model_id"): row for row in preflight.get("models_used", [])}
    for model in models:
        prior_model = preflight_models.get(model["model_id"], {})
        exact = (
            model.get("resolved") is True
            and model.get("model_path") == prior_model.get("model_path")
            and model.get("model_sha256") == prior_model.get("model_sha256")
            and Path(str(model.get("model_path"))).name == model.get("filename")
        )
        checks.append(
            {
                "check": f"cached_exact_model:{model['model_id']}",
                "expected": {
                    "path": prior_model.get("model_path"),
                    "sha256": prior_model.get("model_sha256"),
                },
                "observed": {
                    "path": model.get("model_path"),
                    "sha256": model.get("model_sha256"),
                    "resolved": model.get("resolved"),
                },
                "passed": exact,
            }
        )
    registry_ok = (
        bool(registry_text.strip())
        and "6753" not in registry_text
        and len(GAME_IDS) == len(set(GAME_IDS))
        and frozen_design()["solve_target"] is None
    )
    checks.append(
        {
            "check": "registry_no_new_or_duplicate_solve_target",
            "expected": {"experiment_target": None, "experiment_absent": True},
            "observed": {
                "experiment_target": frozen_design()["solve_target"],
                "experiment_absent": "6753" not in registry_text,
                "registry_sha256": sha256_text(registry_text),
            },
            "passed": registry_ok,
        }
    )
    architecture_fresh = "2026-08-26" in codex_text
    checks.append(
        {
            "check": "architecture_map_fresh",
            "expected": "last reconciled 2026-08-26 or later",
            "observed": "2026-08-26" if architecture_fresh else None,
            "passed": architecture_fresh,
        }
    )
    inventory = nvidia_smi_inventory()
    device_zero = next(
        (row for row in inventory.get("devices", []) if row.get("index") == 0), {}
    )
    required_vram = max(int(model.get("required_vram_mb") or 0) for model in models)
    observed_free_vram = int(device_zero.get("memory_free_mb") or 0)
    checks.append(
        {
            "check": "cuda_device_available",
            "expected": {"device_index": 0, "free_mb_at_least": required_vram},
            "observed": {
                "device": device_zero,
                "free_mb": observed_free_vram,
            },
            "passed": required_vram > 0 and observed_free_vram >= required_vram,
        }
    )
    return {
        "all_passed": all(check.get("passed") is True for check in checks),
        "checks": checks,
        "source_receipts": source_receipts,
        "gpu_inventory": inventory,
    }


def _failed_row(planned: Mapping[str, Any], model: Mapping[str, Any], failure: str) -> JsonDict:
    """Keep one planned denominator row when a precondition blocks inference."""

    row = {
        **dict(planned),
        "model_role": model.get("role"),
        "model_path": model.get("model_path"),
        "model_sha256": model.get("model_sha256"),
        "production_route": PRODUCTION_ROUTE,
        "context_requested": CONTEXT_REQUESTED,
        "context_observed_by_model": None,
        "gpu_receipt": None,
        "live_model_invoked": False,
        "raw_prompt_sha256": None,
        "prompt_tokens": None,
        "tool_events": [],
        **fetch_accounting([]),
        "transition_result": None,
        "change_fidelity": None,
        "transition_utility": None,
        "duration_s": 0.0,
        "actions": [],
        "stop_reason": "preflight_blocked",
        "failure_class": failure,
        "solve_claim": False,
    }
    row["row_sha256"] = row_checksum(row)
    return row


def _gpu_receipt(proposer: Any, device_index: int, peak_vram_mb: int) -> JsonDict:
    """Read runtime context, CUDA layers, and device identity from production receipts."""

    props = proposer.server_props() or {}
    total = int(props.get("total_slots") or proposer.observed_total_slots() or 1)
    model_path = proposer.observed_model_path()
    from carnot.experiment_6752_arc_code_carrying_tool_preflight import _gpu_layers_from_log

    log_path = getattr(proposer, "_stderr_log_path", None)
    try:
        log_text = Path(log_path).read_text(errors="replace") if log_path else ""
    except OSError:
        log_text = ""
    layers = _gpu_layers_from_log(log_text, int(proposer.n_gpu_layers))
    inventory = nvidia_smi_inventory()
    device = next(
        (row for row in inventory.get("devices", []) if row.get("index") == device_index), {}
    )
    return {
        "context_observed_by_model": proposer.observed_n_ctx(),
        "model_path_observed": model_path,
        "total_slots": total,
        "assigned_device": {
            "physical_index": device_index,
            "uuid": device.get("uuid"),
            "name": device.get("name"),
        },
        "gpu_layers": layers,
        "peak_vram_mb": int(peak_vram_mb),
        "server_pid": getattr(getattr(proposer, "_proc", None), "pid", None),
    }


def _pid_vram_mb(pid: int | None) -> int:
    """Return used device memory for the owned llama-server process."""

    if not pid:
        return 0
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0
    values = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            if int(parts[0]) == int(pid):
                values.append(int(parts[1]))
        except ValueError:
            continue
    return max(values, default=0)


def _score_written_engine(game: str, held: Sequence[Any], output_root: Path) -> JsonDict:
    """Score the generated engine on the frozen held tail, or return a model failure."""

    from carnot.agentic.arc_executable_world_model import WorldModelVerifier
    from carnot.agentic.arc_induction_tools import _exec_candidate

    world_model_path = output_root / game / "world_model.py"
    if not world_model_path.is_file():
        return {"ok": False, "failure_class": "model_no_world_model_file"}
    code = world_model_path.read_text()
    engine, error = _exec_candidate(code, "engine")
    if engine is None:
        return {"ok": False, "failure_class": f"model_engine_unusable:{error}"}
    verifier = WorldModelVerifier(list(held)).score(engine)
    true_changed = sum(
        int(np.count_nonzero(np.asarray(item.grid) != np.asarray(item.next_grid)))
        for item in held
        if item.level_after <= item.level_before
    )
    utility = transition_result(
        n_heldout=len(held),
        true_changed_cells=true_changed,
        correct_changed_cells=int(verifier.correct_changed_cells),
        spurious_changed_cells=int(verifier.spurious_changed_cells),
    )
    return {
        "ok": True,
        "engine_sha256": sha256_text(code),
        "change_fidelity": round(float(verifier.change_fidelity), 9),
        **utility,
        "verifier_metrics": {
            "accuracy": round(float(verifier.accuracy), 9),
            "cell_recall": round(float(verifier.cell_recall), 9),
            "change_accuracy": round(float(verifier.change_accuracy), 9),
            "n_changing": int(verifier.n_changing),
            "n_changes_correct": int(verifier.n_changes_correct),
        },
    }


def _build_windows(game_ids: Sequence[str]) -> dict[str, JsonDict]:
    """Rebuild the exact progress windows and explicit held tails used in 2026-08-01."""

    from carnot.agentic.arc_actions_to_progress import build_progress_window
    from carnot.agentic.arc_world_model_trust_energy import _split_prefix_heldout

    windows: dict[str, JsonDict] = {}
    for game in game_ids:
        built = build_progress_window(game)
        if built is None:
            raise RuntimeError(f"frozen progress window unavailable: {game}")
        window, _full, cell = built
        shown, held = _split_prefix_heldout(list(window))
        windows[game] = {"shown": shown, "held": held, "cell": int(cell)}
    return windows


def _public_policy_proposer(game: str, proposer: Any) -> Any:
    """Construct the public production agent and return its E3 proposer seam."""

    from carnot.agentic.arc_competition_agent import E3AgentPolicy, make_carnot_agent

    class BaseAgent:
        def __init__(self, game_id: str) -> None:
            self.game_id = game_id

    agent_type = make_carnot_agent(BaseAgent, cascade=True, proposer=proposer)
    agent = agent_type(game)
    if not isinstance(agent._policy, E3AgentPolicy):
        raise RuntimeError("public factory did not construct E3AgentPolicy")
    selected = agent._policy._proposer()
    if selected is not proposer:
        raise RuntimeError("E3AgentPolicy substituted the frozen proposer")
    return selected


def _run_live_row(
    planned: Mapping[str, Any],
    model: Mapping[str, Any],
    proposer: Any,
    window: Mapping[str, Any],
    output_root: Path,
) -> tuple[JsonDict, str]:
    """Run one planned prompt through the production agent, loop, and verifier."""

    from carnot.agentic import arc_induction_tool_loop as loop

    env = worker_environment(os.environ, model, planned)
    os.environ.update(env)
    events: list[JsonDict] = []
    prompts: list[str] = []
    responses: list[JsonDict] = []
    original_loop = loop.induce_with_tool_loop
    original_post = loop._post_chat

    def instrumented_loop(*args: Any, **kwargs: Any) -> tuple[bool, str]:
        kwargs["tool_event_sink"] = events
        return original_loop(*args, **kwargs)

    def instrumented_post(*args: Any, **kwargs: Any) -> JsonDict:
        messages = args[1] if len(args) > 1 else kwargs["messages"]
        if not prompts:
            prompts.append(str(messages[0]["content"]))
        raw = original_post(*args, **kwargs)
        responses.append(raw)
        return raw

    loop.induce_with_tool_loop = instrumented_loop
    loop._post_chat = instrumented_post
    started_ns = time.monotonic_ns()
    failure: str | None = None
    note = ""
    try:
        world_model_path = output_root / str(planned["game"]) / "world_model.py"
        world_model_path.unlink(missing_ok=True)
        selected = _public_policy_proposer(str(planned["game"]), proposer)
        if planned.get("row_kind") == "transport_sidecar":
            ok, note = loop.induce_with_tool_loop(
                selected,
                str(planned["game"]),
                list(window["shown"]),
                int(window["cell"]),
            )
        else:
            ok, note = selected.induce(
                str(planned["game"]), list(window["shown"]), int(window["cell"])
            )
        if not ok:
            failure = "model_induction_failed"
    except Exception as exc:  # noqa: BLE001 - a live row must retain its terminal failure
        failure = f"live_row_exception:{type(exc).__name__}:{exc}"[:500]
    finally:
        loop.induce_with_tool_loop = original_loop
        loop._post_chat = original_post
    duration_s = round((time.monotonic_ns() - started_ns) / 1_000_000_000, 6)
    prompt = prompts[0] if prompts else ""
    stats = deepcopy(getattr(proposer, "last_tool_loop_stats", {}))
    prompt_tokens = None
    if responses:
        usage = responses[0].get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
    if not prompt_tokens:
        values = stats.get("prompt_tokens_per_turn") or []
        prompt_tokens = values[0] if values else None
    if planned.get("row_kind") == "transport_sidecar":
        scored: JsonDict = {
            "ok": True,
            "transition_result": None,
            "change_fidelity": None,
            "transition_utility": None,
            "verifier_metrics": None,
            "engine_sha256": None,
        }
        if responses and failure == "model_induction_failed":
            failure = None
    else:
        scored = _score_written_engine(str(planned["game"]), window["held"], output_root)
        if not scored.get("ok") and failure is None:
            failure = str(scored.get("failure_class"))
    process = getattr(proposer, "_proc", None)
    peak_vram = _pid_vram_mb(getattr(process, "pid", None))
    gpu = _gpu_receipt(proposer, int(model.get("device_index", 0)), peak_vram)
    accounting = fetch_accounting(events)
    row = {
        **dict(planned),
        "model_role": model.get("role"),
        "model_path": model.get("model_path"),
        "model_sha256": model.get("model_sha256"),
        "production_route": PRODUCTION_ROUTE,
        "context_requested": CONTEXT_REQUESTED,
        "context_observed_by_model": gpu.pop("context_observed_by_model"),
        "gpu_receipt": gpu,
        "live_model_invoked": bool(responses),
        "raw_prompt_sha256": sha256_text(prompt) if prompt else None,
        "prompt_tokens": int(prompt_tokens) if prompt_tokens else None,
        "tool_events": events,
        **accounting,
        "tool_loop_stats": stats,
        "transition_result": scored.get("transition_result"),
        "change_fidelity": scored.get("change_fidelity"),
        "transition_utility": scored.get("transition_utility"),
        "verifier_metrics": scored.get("verifier_metrics"),
        "engine_sha256": scored.get("engine_sha256"),
        "duration_s": duration_s,
        "actions": [],
        "stop_reason": stats.get("terminated_by") or note or "induce_terminal",
        "failure_class": failure,
        "solve_claim": False,
    }
    row["row_sha256"] = row_checksum(row)
    return row, prompt


def run_live_batch(
    model: Mapping[str, Any],
    planned_rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint_path: Path | None = None,
) -> list[JsonDict]:
    """Run one model load over its ordered rows and retain every row receipt."""

    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer, _free_port
    from carnot.agentic.arc_induction_tools import render_tool_schemas_for_prompt
    games = tuple(dict.fromkeys(str(row["game"]) for row in planned_rows))
    windows = _build_windows(games)
    output_root = Path(os.environ["CARNOT_ARC_E3_DIR"])
    budgets = _budgets_for(planned_rows[0])
    proposer = LocalGGUFProposer(
        repo_substr=str(model["repo_substr"]),
        model_path=str(model["model_path"]),
        n_ctx=CONTEXT_REQUESTED,
        max_tokens=int(budgets["completion_tokens_per_turn"]),
        timeout=int(budgets["timeout_s"]),
        port=_free_port(),
        mtp=False,
    )
    rows: list[JsonDict] = []
    prompts: dict[tuple[str, int, str], str] = {}
    try:
        for planned in planned_rows:
            row, prompt = _run_live_row(
                planned,
                model,
                proposer,
                windows[str(planned["game"])],
                output_root,
            )
            rows.append(row)
            prompts[(str(planned["game"]), int(planned["seed"]), str(planned["arm"]))] = prompt
            key = (str(planned["game"]), int(planned["seed"]))
            if all((key[0], key[1], arm) in prompts for arm in ARMS):
                from carnot.agentic.arc_executable_world_model import objects_block

                object_table = (
                    "OBJECT STRUCTURE (same frames, connected-component view -- use object "
                    "shape ids to track objects across the deltas above):\n"
                    + objects_block(windows[key[0]]["shown"])
                )
                try:
                    receipt = prompt_isolation_receipt(
                        prompts[(key[0], key[1], BASELINE_ARM)],
                        prompts[(key[0], key[1], TREATMENT_ARM)],
                        object_table,
                        render_tool_schemas_for_prompt(),
                    )
                except ValueError as exc:
                    receipt = {"only_object_table_removed": False, "error": str(exc)}
                for candidate in rows:
                    if candidate.get("game") == key[0] and candidate.get("seed") == key[1]:
                        candidate["prompt_isolation_receipt"] = deepcopy(receipt)
                        if not receipt.get("only_object_table_removed"):
                            candidate["failure_class"] = "prompt_arm_isolation_failed"
                        candidate["row_sha256"] = row_checksum(candidate)
            if checkpoint_path is not None:
                _atomic_write(checkpoint_path, rows)
    finally:
        proposer.stop()
    return rows


def run_model_batch_subprocess(
    model: Mapping[str, Any], planned_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Run one owned model batch in a fresh process and read its retained rows."""

    with tempfile.TemporaryDirectory(prefix="carnot-exp6753-") as temp_dir:
        root = Path(temp_dir)
        input_path = root / "job.json"
        output_path = root / "rows.json"
        input_path.write_text(
            json.dumps({"model": dict(model), "planned_rows": list(planned_rows)}, indent=2) + "\n"
        )
        env = worker_environment(os.environ, model, planned_rows[0])
        env["CARNOT_ARC_E3_DIR"] = str(root / "e3")
        command = [
            sys.executable,
            "-m",
            "carnot.experiment_6753_object_table_fetch_on_demand_ab",
            "--worker-job",
            str(input_path),
            "--worker-output",
            str(output_path),
        ]
        timeout_s = sum(int(_budgets_for(row)["timeout_s"]) for row in planned_rows) + 900
        try:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            partial = json.loads(output_path.read_text()) if output_path.is_file() else []
            seen = {row.get("row_id") for row in partial if isinstance(row, dict)}
            failure = f"worker_process_failed:{type(exc).__name__}:{exc}"[:1000]
            partial.extend(
                _failed_row(row, model, failure)
                for row in planned_rows
                if row.get("row_id") not in seen
            )
            return partial
        if result.returncode != 0 or not output_path.is_file():
            failure = (
                f"worker_process_failed:returncode={result.returncode}:"
                f"stderr={result.stderr[-1000:]}"
            )
            return [_failed_row(row, model, failure) for row in planned_rows]
        value = json.loads(output_path.read_text())
        if not isinstance(value, list):
            raise ValueError("worker output must be a row list")
        return value


def _atomic_write(path: Path, value: Any) -> None:
    """Replace a JSON file only after a complete temporary file is ready."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=False)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def build_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
    preflight: Mapping[str, Any],
    started_ns: int,
    finished_ns: int,
) -> JsonDict:
    """Build one terminal artifact from retained rows and frozen inputs."""

    blocked = preflight.get("all_passed") is not True
    if blocked:
        reduction = completion_and_adoption([])
        reduction["row_completion_receipt"] = {
            "planned": len(science_plan()) + len(sidecar_plan()),
            "observed": len(rows),
            "missing_row_ids": [],
            "unexpected_row_ids": [],
            "duplicate_row_ids": [],
            "invalid_rows": {str(row.get("row_id")): [str(row.get("failure_class"))] for row in rows},
        }
        failed = next(
            (check for check in preflight.get("checks", []) if check.get("passed") is not True),
            {"check": "unknown_preflight"},
        )
        verdict_class = "blocked"
        honest_verdict = f"complete_blocked_object_table_ab:{failed.get('check')}"
        gate_summary = list(preflight.get("checks", []))
    else:
        reduction = completion_and_adoption(rows)
        gate_summary = [
            {"check": key, "observed": value, "passed": value is True}
            for key, value in reduction["adoption_gate_conditions"].items()
        ]
        if reduction["object_table_ab_completed"]:
            verdict_class = "positive" if reduction["adoption_gate_passed"] else "null"
            honest_verdict = (
                "complete_object_table_ab_adopt_fetch_on_demand"
                if reduction["adoption_gate_passed"]
                else "complete_object_table_ab_do_not_adopt"
            )
        else:
            verdict_class = "partial"
            honest_verdict = "complete_partial_object_table_ab_rows_incomplete"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": 6753,
        "title": "Inline object table versus production find_objects fetch on demand",
        "run_date": RUN_DATE,
        "status": "complete" if reduction["object_table_ab_completed"] else verdict_class,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(0, finished_ns - started_ns) / 1_000_000_000, 6),
        "random_seed": {"science_game_arm_seeds": list(SEEDS), "bootstrap": BOOTSTRAP_SEED},
        "reproducibility_checksum": None,
        "models_used": [dict(model) for model in models],
        "live_model_invoked": any(row.get("live_model_invoked") is True for row in rows),
        "context_requested": CONTEXT_REQUESTED,
        "gpu_receipts": [
            {
                "row_id": row.get("row_id"),
                "model_id": row.get("model_id"),
                "receipt": deepcopy(row.get("gpu_receipt")),
            }
            for row in rows
            if row.get("gpu_receipt") is not None
        ],
        "rows": [dict(row) for row in rows],
        "frozen_design": frozen_design(),
        "source_receipts": deepcopy(preflight.get("source_receipts", {})),
        "preconditions_checked": deepcopy(dict(preflight)),
        "mean_prompt_token_savings": reduction["mean_prompt_token_savings"],
        "fetch_rate": reduction["fetch_rate"],
        "useful_fetch_rate": reduction["useful_fetch_rate"],
        "change_fidelity_delta": reduction["change_fidelity_delta"],
        "change_fidelity_ci95": reduction["change_fidelity_ci95"],
        "transition_utility_delta": reduction["transition_utility_delta"],
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "harmful_regressions": reduction["harmful_regressions"],
        "paired_analysis": reduction,
        "adoption_gate_conditions": reduction["adoption_gate_conditions"],
        "adoption_gate_passed": reduction["adoption_gate_passed"],
        "object_table_ab_completed": reduction["object_table_ab_completed"],
        "solve_claim": False,
        "gate_check_summary": gate_summary,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return every terminal artifact contract error without changing evidence."""

    errors: list[str] = []
    if artifact.get("schema") != SCHEMA:
        errors.append("schema")
    if set(artifact) - set(artifact.get("field_principles", {})):
        errors.append("field_principles_incomplete")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("context_requested") != CONTEXT_REQUESTED:
        errors.append("context_requested")
    if artifact.get("solve_claim") is not False:
        errors.append("solve_claim")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class")
    rows = artifact.get("rows") if isinstance(artifact.get("rows"), list) else []
    expected_ids = [row["row_id"] for row in science_plan() + sidecar_plan()]
    if [row.get("row_id") for row in rows] != expected_ids:
        errors.append("row_denominator_or_order")
    blocked = artifact.get("verdict_class") == "blocked"
    if blocked:
        if artifact.get("object_table_ab_completed") is not False:
            errors.append("blocked_completion")
        if not all(str(row.get("failure_class") or "").startswith("preflight_blocked:") for row in rows):
            errors.append("blocked_rows")
    else:
        reduction = completion_and_adoption(rows)
        for field in (
            "mean_prompt_token_savings",
            "fetch_rate",
            "useful_fetch_rate",
            "change_fidelity_delta",
            "change_fidelity_ci95",
            "transition_utility_delta",
            "adoption_gate_passed",
            "object_table_ab_completed",
        ):
            if artifact.get(field) != reduction.get(field):
                errors.append(f"reduction:{field}")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def run(
    *,
    result_path: Path = RESULT_PATH,
    resolver: Callable[[], list[JsonDict]] = resolve_model_specs,
    preflight_fn: Callable[[list[JsonDict]], JsonDict] = live_preflight,
    worker_runner: Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]]], list[JsonDict]] = (
        run_model_batch_subprocess
    ),
    clock: Callable[[], int] = time.monotonic_ns,
) -> JsonDict:
    """Run fail-closed gates, then the scored batch and separate sidecar batch."""

    started_ns = clock()
    models = resolver()
    preflight = preflight_fn(models)
    science = science_plan()
    sidecars = sidecar_plan()
    if preflight.get("all_passed") is True:
        rows = worker_runner(models[0], science) + worker_runner(models[1], sidecars)
    else:
        failed = next(
            (check for check in preflight.get("checks", []) if check.get("passed") is not True),
            {"check": "unknown_preflight"},
        )
        failure = f"preflight_blocked:{failed.get('check')}"
        rows = [_failed_row(row, models[0], failure) for row in science]
        rows.extend(_failed_row(row, models[1], failure) for row in sidecars)
    finished_ns = clock()
    artifact = build_artifact(
        rows=rows,
        models=models,
        preflight=preflight,
        started_ns=started_ns,
        finished_ns=finished_ns,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"invalid Exp6753 artifact: {errors}")
    _atomic_write(result_path, artifact)
    return artifact


def _worker_entry(job_path: Path, output_path: Path) -> int:
    """Run one owned model batch inside its fresh worker process."""

    job = _load_json(job_path)
    rows = run_live_batch(job["model"], job["planned_rows"], checkpoint_path=output_path)
    _atomic_write(output_path, rows)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the parent experiment or one explicitly owned model worker."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-job", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args(argv)
    if args.worker_job is not None:
        if args.worker_output is None:
            parser.error("--worker-job requires --worker-output")
        return _worker_entry(args.worker_job, args.worker_output)
    artifact = run()
    print(
        json.dumps(
            {
                "artifact": str(RESULT_PATH),
                "completed": artifact["object_table_ab_completed"],
                "adopted": artifact["adoption_gate_passed"],
                "verdict": artifact["honest_verdict"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository script is the public entry point
    raise SystemExit(main())
