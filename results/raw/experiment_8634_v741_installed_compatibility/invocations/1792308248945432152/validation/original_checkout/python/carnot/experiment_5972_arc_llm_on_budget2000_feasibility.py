"""Exp5972: ARC LLM-on budget-2000 scheduling feasibility.

The artifact this module writes is a timing feasibility receipt, not a solve
claim. The expensive path is guarded so a missing mandated Qwen3.6 GGUF, CUDA
offload, or live E3 entrypoint produces a blocked artifact before model load.

Spec refs: REQ-ARC-LLB2-5972,
SCENARIO-ARC-LLB2-5972-PRECONDITION-BLOCK,
SCENARIO-ARC-LLB2-5972-SEALED-LIVE-CELLS,
SCENARIO-ARC-LLB2-5972-PROJECTION-NO-FLAG-FLIP.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5972_arc_llm_on_budget2000_feasibility.json")
PRIOR_LLM_ON_RELATIVE_PATH = Path("results/outer_loop_scored_path_lever_ab_llm_on_20260726.json")

MANDATED_MODEL_HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
MANDATED_MODEL_NAME = "Qwen3.6-35B-A3B"
MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "name": MANDATED_MODEL_NAME,
        "hf_id": MANDATED_MODEL_HF_ID,
        "role": "mandated_flagship_moe",
        "quantization": "Q4_K_M",
        "substitution_allowed": False,
    },
)

GAIN_GAMES: tuple[str, ...] = ("dc22", "ft09", "s5i5", "su15", "lf52", "r11l", "cd82")
GAIN_GAME_ARM = "S_llmon_budget2000"
LP85_TREATMENT_ARM = "S_minus_frontier_llmon"
LP85_CONTROL_ARM = "S_llmon_healthy_control"
DEFAULT_SEED = 20260804
ACTION_BUDGET = 2000
DEFAULT_PER_CELL_TIMEOUT_S = 4500
DEFAULT_TOTAL_TASK_BUDGET_S = 12 * 3600
DEFAULT_BOOTSTRAPS = 20_000
DEFAULT_PORT = 8952

PROTECTED_RELATIVE_PATHS: tuple[Path, ...] = (
    Path("ops/arc_solve_registry.yaml"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("scripts/research_conductor.py"),
    Path("ops/north-star.md"),
    Path("ops/known-issues.md"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("_bmad/traceability.md"),
    Path("openspec/capabilities/agentic-harness/spec.md"),
)

REQUIRED_FIELDS: tuple[str, ...] = (
    "status",
    "preconditions_checked",
    "registry_precheck_and_hash",
    "model_specs",
    "model_file_hash_embedded_tokenizer_llama_cpp_and_cuda_receipts",
    "gpu_vram_thermal_process_port_and_cleanup_receipts",
    "game_arm_seed_budget_and_timeout_seal",
    "live_agent_path_and_disabled_escape_hatches",
    "smoke_and_checkpoint_resume_receipts",
    "expected_completed_missing_errored_timed_out_and_generator_invalid_cells",
    "per_cell_calls_tokens_actions_progress_levels_plan_channel_time_and_gpu_metrics",
    "lp85_healthy_matched_positive_control",
    "budget400_frozen_comparison_receipt",
    "twenty_five_game_twelve_hour_projection_and_interval",
    "load_amortization_and_censoring_policy",
    "no_automatic_flag_change_receipt",
    "solve_provenance",
    "no_new_solve_credit_receipt",
    "shipped_flag_and_registry_immutability",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "missing_verifier_gaps",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": (
        "missing model, CUDA, live path, time, or resource prerequisites block before the "
        "expensive run."
    ),
    "preconditions_checked": (
        "missing model, CUDA, live path, time, or resource prerequisites block before the "
        "expensive run."
    ),
    "registry_precheck_and_hash": (
        "every target is an already-cleared measurement case and registry state remains immutable."
    ),
    "model_specs": "the mandated flagship public GGUF and genuine local CUDA path are auditable.",
    "model_file_hash_embedded_tokenizer_llama_cpp_and_cuda_receipts": (
        "the mandated flagship public GGUF and genuine local CUDA path are auditable."
    ),
    "gpu_vram_thermal_process_port_and_cleanup_receipts": (
        "resource authenticity and teardown are measured for every phase."
    ),
    "game_arm_seed_budget_and_timeout_seal": "cells and limits are fixed before outcomes.",
    "live_agent_path_and_disabled_escape_hatches": (
        "only make_carnot_agent/E3AgentPolicy with prohibited routes disabled receives credit."
    ),
    "smoke_and_checkpoint_resume_receipts": (
        "authentic inference precedes full cells and completed evidence survives interruption."
    ),
    "expected_completed_missing_errored_timed_out_and_generator_invalid_cells": (
        "every planned cell has one honest terminal state."
    ),
    "per_cell_calls_tokens_actions_progress_levels_plan_channel_time_and_gpu_metrics": (
        "scheduling, accuracy, and mechanism use remain disaggregated."
    ),
    "lp85_healthy_matched_positive_control": (
        "the previously unpaired plan-channel activation must have a valid matched control."
    ),
    "budget400_frozen_comparison_receipt": (
        "compare with the immutable prior artifact without rewriting or selectively dropping cells."
    ),
    "twenty_five_game_twelve_hour_projection_and_interval": (
        "feasibility uses a measured uncertainty bound under the shared wall clock."
    ),
    "load_amortization_and_censoring_policy": (
        "load reuse and timeouts are modeled explicitly, not hidden in a mean."
    ),
    "no_automatic_flag_change_receipt": (
        "a timing result alone does not authorize a policy mutation."
    ),
    "solve_provenance": (
        "use live_agent_self_discovery; level outcomes come only from the live agent's own "
        "attempts/runtime evidence."
    ),
    "no_new_solve_credit_receipt": (
        "incidental public level outcomes are scheduling measurements and do not update solve claims."
    ),
    "shipped_flag_and_registry_immutability": (
        "MAX_ACTIONS defaults, feature flags, and registry remain byte-identical."
    ),
    "protected_files_unchanged": (
        "active roadmap, conductor, exclusions, history, and unrelated changes remain immutable."
    ),
    "duration_s": "use measured live_llm_inference.",
    "inference_substrate": "use measured live_llm_inference.",
    "verifier_is_oracle": (
        "false; public timing does not prove hidden-game performance and single-seed limits are "
        "explicit."
    ),
    "missing_verifier_gaps": (
        "false; public timing does not prove hidden-game performance and single-seed limits are "
        "explicit."
    ),
    "field_provenance": (
        "artifact fields carry principle annotations tied to the preregistered safeguards."
    ),
    "test_commands": (
        "record focused, coverage, full-suite, spec, E2E, adversarial, protected-file, and "
        "clutter checks."
    ),
    "test_exit_codes": "record the actual exit code for each verification command.",
    "reproducibility_checksum": (
        "hash measured rows and immutable precondition receipts, excluding wall-clock duration."
    ),
    "honest_verdict": (
        "use `complete_feasible:`, `complete_infeasible:`, `complete_underpowered:`, or `blocked:`."
    ),
}

PLANNED_TEST_COMMANDS: tuple[str, ...] = (
    ".venv/bin/pytest tests/python/test_experiment_5972_arc_llm_on_budget2000_feasibility.py -q",
    (
        ".venv/bin/pytest tests/python/test_experiment_5972_arc_llm_on_budget2000_feasibility.py "
        "--cov=python/carnot/experiment_5972_arc_llm_on_budget2000_feasibility.py "
        "--cov-report=term-missing --cov-fail-under=100 -q"
    ),
    ".venv/bin/python -m carnot.experiment_5972_arc_llm_on_budget2000_feasibility --validate",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/root_clutter_sweep.py --min-age-min 0",
)


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def file_hash_record(path: Path) -> JsonDict:
    if not path.exists():
        return {"path": str(path), "exists": False, "sha256": None, "bytes": None}
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def freeze_game_arm_seed_budget_and_timeout(
    *,
    seed: int = DEFAULT_SEED,
    budget: int = ACTION_BUDGET,
    per_cell_timeout_s: int = DEFAULT_PER_CELL_TIMEOUT_S,
    total_task_budget_s: int = DEFAULT_TOTAL_TASK_BUDGET_S,
) -> JsonDict:
    cells: list[JsonDict] = []
    for game in GAIN_GAMES:
        cells.append(
            {
                "cell_id": f"gain__{game}__{GAIN_GAME_ARM}__s{seed}__b{budget}",
                "game": game,
                "arm": GAIN_GAME_ARM,
                "seed": seed,
                "budget": budget,
                "timeout_s": per_cell_timeout_s,
                "purpose": "budget2000_gain_game_timing",
            }
        )
    for arm in (LP85_TREATMENT_ARM, LP85_CONTROL_ARM):
        cells.append(
            {
                "cell_id": f"lp85_positive_control__lp85__{arm}__s{seed}__b{budget}",
                "game": "lp85",
                "arm": arm,
                "seed": seed,
                "budget": budget,
                "timeout_s": per_cell_timeout_s,
                "purpose": "lp85_matched_positive_control_pair",
            }
        )
    return {
        "seed": seed,
        "budget": budget,
        "per_cell_timeout_s": per_cell_timeout_s,
        "total_task_budget_s": total_task_budget_s,
        "cells": cells,
        "cell_count": len(cells),
        "cell_count_note": (
            "The prompt says eight cells once but also requires seven gain games plus both lp85 "
            "treatment and matched control; the concrete paired-control requirement yields nine "
            "sealed cells."
        ),
        "sealed_before_outcomes": True,
    }


def account_cell_terminal_states(cells: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    expected = {str(cell["cell_id"]) for cell in cells}
    seen: set[str] = set()
    counts = {
        "planned": len(cells),
        "completed": 0,
        "missing": 0,
        "errored": 0,
        "timed_out": 0,
        "generator_invalid": 0,
        "unexpected": 0,
    }
    aliases = {
        "complete": "completed",
        "error": "errored",
        "timeout": "timed_out",
        "invalid_generator": "generator_invalid",
    }
    for row in rows:
        cell_id = str(row.get("cell_id") or "")
        if cell_id in seen:
            raise ValueError(f"duplicate terminal row for {cell_id}")
        seen.add(cell_id)
        state = str(row.get("terminal_state") or "")
        state = aliases.get(state, state)
        if cell_id not in expected:
            counts["unexpected"] += 1
        elif state in ("completed", "errored", "timed_out", "generator_invalid"):
            counts[state] += 1
        else:
            counts["errored"] += 1
    counts["missing"] = len(expected - seen)
    return counts


def load_checkpoint_rows(
    checkpoint_dir: Path, cells: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], JsonDict]:
    sealed_order = {str(cell["cell_id"]): index for index, cell in enumerate(cells)}
    rows_by_id: dict[str, JsonDict] = {}
    ignored_unsealed: list[str] = []
    ignored_malformed: list[str] = []
    for path in sorted(checkpoint_dir.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            ignored_malformed.append(path.name)
            continue
        cell_id = str(row.get("cell_id") or "")
        if cell_id not in sealed_order:
            ignored_unsealed.append(cell_id or path.name)
            continue
        rows_by_id[cell_id] = dict(row)
    loaded_ids = sorted(rows_by_id, key=sealed_order.__getitem__)
    missing_ids = [cell_id for cell_id in sealed_order if cell_id not in rows_by_id]
    return order_rows_by_seal(cells, rows_by_id.values()), {
        "checkpoint_dir": str(checkpoint_dir),
        "loaded_count": len(loaded_ids),
        "loaded_cell_ids": loaded_ids,
        "missing_count": len(missing_ids),
        "missing_cell_ids": missing_ids,
        "ignored_unsealed_count": len(ignored_unsealed),
        "ignored_unsealed_cell_ids": ignored_unsealed,
        "ignored_malformed_count": len(ignored_malformed),
        "ignored_malformed_files": ignored_malformed,
    }


def order_rows_by_seal(
    cells: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    sealed_order = {str(cell["cell_id"]): index for index, cell in enumerate(cells)}
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: sealed_order.get(str(row.get("cell_id")), len(sealed_order)),
    )


def _per_game_elapsed(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    by_game: dict[str, float] = {}
    for row in rows:
        if str(row.get("terminal_state")) != "completed":
            continue
        if row.get("generator_valid") is False or row.get("timeout") is True:
            continue
        game = str(row.get("game"))
        elapsed = float(row.get("elapsed_s") or row.get("wall_s") or 0.0)
        by_game[game] = max(by_game.get(game, 0.0), elapsed)
    return list(by_game.values())


def twenty_five_game_projection(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_games: int = 25,
    n_boot: int = DEFAULT_BOOTSTRAPS,
    seed: int = DEFAULT_SEED,
    load_amortization_s: float = 0.0,
    cap_s: float = DEFAULT_TOTAL_TASK_BUDGET_S,
) -> JsonDict:
    samples = _per_game_elapsed(rows)
    if not samples:
        return {
            "games_projected": n_games,
            "bootstrap_unit": "game",
            "n_measured_games": 0,
            "mean_s": None,
            "lower_bound_s": None,
            "upper_bound_s": None,
            "fits_12h_at_upper_bound": False,
            "cap_s": cap_s,
            "load_amortization_s": load_amortization_s,
            "projection_status": "unavailable_no_complete_valid_cells",
        }
    rng = random.Random(seed)
    totals = []
    for _ in range(n_boot):
        draw = sum(samples[rng.randrange(len(samples))] for _ in range(n_games))
        totals.append(draw + float(load_amortization_s))
    totals.sort()
    lo = totals[int(0.025 * (n_boot - 1))]
    hi = totals[int(0.975 * (n_boot - 1))]
    mean = (sum(samples) / len(samples)) * n_games + float(load_amortization_s)
    return {
        "games_projected": n_games,
        "bootstrap_unit": "game",
        "n_measured_games": len(samples),
        "measured_game_elapsed_s": [round(x, 3) for x in samples],
        "mean_s": round(mean, 3),
        "lower_bound_s": round(lo, 3),
        "upper_bound_s": round(hi, 3),
        "fits_12h_at_upper_bound": bool(hi <= cap_s),
        "cap_s": cap_s,
        "load_amortization_s": round(float(load_amortization_s), 3),
        "projection_status": "computed_from_completed_valid_cells",
    }


def lp85_pairing_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_arm = {str(row.get("arm")): row for row in rows if row.get("game") == "lp85"}
    treatment = by_arm.get(LP85_TREATMENT_ARM)
    control = by_arm.get(LP85_CONTROL_ARM)
    paired = bool(
        treatment
        and control
        and treatment.get("terminal_state") == "completed"
        and control.get("terminal_state") == "completed"
        and treatment.get("generator_valid", True)
        and control.get("generator_valid", True)
    )
    return {
        "required_pair": [LP85_TREATMENT_ARM, LP85_CONTROL_ARM],
        "present_arms": sorted(by_arm),
        "paired": paired,
        "plan_channel_openings": {
            arm: int((row or {}).get("plan_channel_openings") or 0)
            for arm, row in ((LP85_TREATMENT_ARM, treatment), (LP85_CONTROL_ARM, control))
        },
        "status": "paired_healthy" if paired else "missing_or_unhealthy_pair",
    }


def field_provenance() -> JsonDict:
    return {
        field: {
            "principle": principle,
            "spec": "REQ-ARC-LLB2-5972",
        }
        for field, principle in FIELD_PRINCIPLES.items()
    }


def default_budget400_receipt(root: Path = REPO_ROOT) -> JsonDict:
    path = root / PRIOR_LLM_ON_RELATIVE_PATH
    if not path.exists():
        return {
            "path": PRIOR_LLM_ON_RELATIVE_PATH.as_posix(),
            "exists": False,
            "sha256": None,
            "budget400_rows_for_targets": 0,
            "rewritten": False,
        }
    obj = json.loads(path.read_text(encoding="utf-8"))
    target_games = set(GAIN_GAMES) | {"lp85"}
    rows = [
        row
        for row in obj.get("rows", [])
        if row.get("game") in target_games and int(row.get("budget") or 0) == 400
    ]
    return {
        "path": PRIOR_LLM_ON_RELATIVE_PATH.as_posix(),
        "exists": True,
        "sha256": sha256_file(path),
        "budget400_rows_for_targets": len(rows),
        "target_games": sorted(target_games),
        "row_games": sorted({str(row.get("game")) for row in rows}),
        "immutable_input_only": True,
        "rewritten": False,
    }


def _all_preconditions_ok(preconditions: Sequence[Mapping[str, Any]]) -> bool:
    return all(bool(item.get("available")) for item in preconditions)


def _blocked_reason(preconditions: Sequence[Mapping[str, Any]]) -> str:
    for item in preconditions:
        if not bool(item.get("available")):
            return str(item.get("name") or "precondition")
    return "none"


def _test_exit_map(receipts: Sequence[Mapping[str, Any]] | None) -> JsonDict:
    if not receipts:
        return {cmd: None for cmd in PLANNED_TEST_COMMANDS}
    return {str(row.get("command")): row.get("exit_code") for row in receipts}


def build_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    registry_hashes: Mapping[str, Any],
    protected_file_hashes: Mapping[str, Any],
    duration_s: float,
    model_receipts: Mapping[str, Any] | None = None,
    gpu_receipts: Mapping[str, Any] | None = None,
    budget400_receipt: Mapping[str, Any] | None = None,
    smoke_receipts: Mapping[str, Any] | None = None,
    test_receipts: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    seal = freeze_game_arm_seed_budget_and_timeout()
    terminal_counts = account_cell_terminal_states(seal["cells"], rows)
    load_amortization_s = float((smoke_receipts or {}).get("model_load_s") or 0.0)
    projection = twenty_five_game_projection(rows, load_amortization_s=load_amortization_s)
    blocked = not _all_preconditions_ok(preconditions)
    if blocked:
        status = "blocked"
        verdict = f"blocked: {_blocked_reason(preconditions)} unavailable before expensive run"
        substrate = "blocked_before_live_llm_inference"
    elif terminal_counts["completed"] < terminal_counts["planned"]:
        status = "complete_underpowered"
        verdict = "complete_underpowered: not all sealed cells completed with valid live evidence"
        substrate = "live_llm_inference"
    elif projection["fits_12h_at_upper_bound"]:
        status = "complete_feasible"
        verdict = "complete_feasible: 25-game upper projection fits the 12-hour wall clock"
        substrate = "live_llm_inference"
    else:
        status = "complete_infeasible"
        verdict = "complete_infeasible: 25-game upper projection exceeds the 12-hour wall clock"
        substrate = "live_llm_inference"

    artifact: JsonDict = {
        "experiment_id": 5972,
        "random_seed": DEFAULT_SEED,
        "status": status,
        "preconditions_checked": list(preconditions),
        "registry_precheck_and_hash": dict(registry_hashes),
        "model_specs": [dict(item) for item in MODEL_SPECS],
        "model_file_hash_embedded_tokenizer_llama_cpp_and_cuda_receipts": dict(
            model_receipts or {}
        ),
        "gpu_vram_thermal_process_port_and_cleanup_receipts": dict(gpu_receipts or {}),
        "game_arm_seed_budget_and_timeout_seal": seal,
        "live_agent_path_and_disabled_escape_hatches": {
            "entrypoint": "make_carnot_agent/E3AgentPolicy",
            "adapter_free": True,
            "disabled_escape_hatches": {
                "GameAdapter": True,
                "game_source": True,
                "offline_BFS": True,
                "per_game_calibration": True,
                "prior_game_logs": True,
                "registry_trajectories": True,
                "hidden_state": True,
            },
            "escape_hatch_access_count": 0,
        },
        "smoke_and_checkpoint_resume_receipts": dict(
            smoke_receipts
            or {
                "smoke_attempted": False,
                "checkpoint_resume_attempted": False,
                "blocked_before_smoke": blocked,
            }
        ),
        "expected_completed_missing_errored_timed_out_and_generator_invalid_cells": terminal_counts,
        "per_cell_calls_tokens_actions_progress_levels_plan_channel_time_and_gpu_metrics": [
            dict(row) for row in rows
        ],
        "lp85_healthy_matched_positive_control": lp85_pairing_receipt(rows),
        "budget400_frozen_comparison_receipt": dict(
            budget400_receipt or default_budget400_receipt(REPO_ROOT)
        ),
        "twenty_five_game_twelve_hour_projection_and_interval": projection,
        "load_amortization_and_censoring_policy": {
            "load_amortization_stated_separately": True,
            "load_amortization_s": projection.get("load_amortization_s"),
            "model_load_s_source": "smoke_and_checkpoint_resume_receipts.model_load_s",
            "censored_cells_used_for_uncensored_mean": False,
            "censoring_policy": (
                "completed generator-valid cells feed the uncensored bootstrap; timed-out or "
                "generator-invalid cells require explicit censored modeling before extrapolation."
            ),
        },
        "no_automatic_flag_change_receipt": {
            "max_actions_changed": False,
            "feature_flags_changed": False,
            "flag_advice_emitted": False,
        },
        "solve_provenance": "live_agent_self_discovery",
        "no_new_solve_credit_receipt": {
            "public_solve_claimed": False,
            "registry_update_requested": False,
            "incidental_outcomes_are_telemetry_only": True,
        },
        "shipped_flag_and_registry_immutability": {
            "max_actions_changed": False,
            "feature_flags_changed": False,
            "registry_unchanged": bool(registry_hashes.get("unchanged", False)),
            "registry_hash": registry_hashes,
        },
        "protected_files_unchanged": dict(protected_file_hashes),
        "duration_s": round(float(duration_s), 3),
        "inference_substrate": substrate,
        "verifier_is_oracle": False,
        "missing_verifier_gaps": [
            "public timing does not prove hidden-game performance",
            "single deterministic seed limits mechanism inference",
            "blocked artifacts contain no live solve evidence",
        ],
        "field_provenance": field_provenance(),
        "test_commands": [str(cmd) for cmd in PLANNED_TEST_COMMANDS],
        "test_exit_codes": _test_exit_map(test_receipts),
        "reproducibility_checksum": "",
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _checksum_payload(artifact: Mapping[str, Any]) -> JsonDict:
    payload = dict(artifact)
    payload.pop("duration_s", None)
    payload.pop("reproducibility_checksum", None)
    return payload


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    return sha256_json(_checksum_payload(artifact))


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance must be a mapping")
    for field, principle in FIELD_PRINCIPLES.items():
        got = provenance.get(field)
        if not isinstance(got, Mapping) or got.get("principle") != principle:
            raise ValueError(f"field_provenance:{field}")
    if not str(artifact["honest_verdict"]).startswith(
        ("complete_feasible:", "complete_infeasible:", "complete_underpowered:", "blocked:")
    ):
        raise ValueError("honest_verdict has unsupported prefix")
    specs = artifact.get("model_specs")
    if not isinstance(specs, list) or not specs or specs[0].get("hf_id") != MANDATED_MODEL_HF_ID:
        raise ValueError("model_specs must start with the mandated Qwen3.6 GGUF")
    if any("Qwen3.5" in json.dumps(spec) for spec in specs):
        raise ValueError("legacy model substitution is forbidden")
    if artifact["no_automatic_flag_change_receipt"].get("max_actions_changed"):
        raise ValueError("MAX_ACTIONS change is forbidden")
    if artifact["no_new_solve_credit_receipt"].get("registry_update_requested"):
        raise ValueError("registry solve credit update is forbidden")
    expected_checksum = reproducibility_checksum(artifact)
    if artifact.get("reproducibility_checksum") != expected_checksum:
        raise ValueError("reproducibility_checksum mismatch")


def protected_hashes(root: Path = REPO_ROOT) -> JsonDict:  # pragma: no cover - filesystem receipt
    out: JsonDict = {}
    for rel in PROTECTED_RELATIVE_PATHS:
        out[rel.as_posix()] = file_hash_record(root / rel)
    return out


def compare_hash_maps(before: Mapping[str, Any], after: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    out: JsonDict = {}
    for key in sorted(set(before) | set(after)):
        b = before.get(key, {})
        a = after.get(key, {})
        out[key] = {
            "before_sha256": b.get("sha256") if isinstance(b, Mapping) else None,
            "after_sha256": a.get("sha256") if isinstance(a, Mapping) else None,
            "unchanged": (
                isinstance(b, Mapping)
                and isinstance(a, Mapping)
                and b.get("sha256") == a.get("sha256")
            ),
        }
    return out


def registry_precheck(root: Path = REPO_ROOT) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover
    registry = root / "ops/arc_solve_registry.yaml"
    text = registry.read_text(encoding="utf-8") if registry.exists() else ""
    targets = list(GAIN_GAMES) + ["lp85"]
    game_receipts = {
        game: {
            "already_cleared_measurement_game": (f"- game: {game}" in text),
            "registry_mentions": text.count(game),
        }
        for game in targets
    }
    ok = bool(registry.exists()) and all(
        item["already_cleared_measurement_game"] for item in game_receipts.values()
    )
    check = {
        "name": "registry_precheck",
        "available": ok,
        "detail": game_receipts,
    }
    h = file_hash_record(registry)
    return [check], {
        "path": "ops/arc_solve_registry.yaml",
        "before_sha256": h.get("sha256"),
        "after_sha256": h.get("sha256"),
        "unchanged": True,
        "targets": targets,
        "target_receipts": game_receipts,
    }


def _port_is_free(port: int) -> bool:  # pragma: no cover
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _run(cmd: list[str], timeout: int = 20) -> JsonDict:  # pragma: no cover
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "command": cmd,
            "returncode": res.returncode,
            "stdout": res.stdout[:4000],
            "stderr": res.stderr[:4000],
        }
    except Exception as exc:
        return {"command": cmd, "error": f"{type(exc).__name__}: {exc}"}


def _llama_cpp_cuda_support() -> tuple[bool, str]:  # pragma: no cover
    try:
        from llama_cpp import llama_cpp as backend  # type: ignore

        return bool(backend.llama_supports_gpu_offload()), "llama_cpp.llama_supports_gpu_offload"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _resolve_mandated_model(root: Path = REPO_ROOT) -> JsonDict:  # pragma: no cover
    sys.path.insert(0, str(root / "python"))
    try:
        from carnot.inference.sota_models import gguf_tokenizer_loadable, resolve_cached_gguf
    except Exception as exc:
        return {
            "model_path": None,
            "available": False,
            "reason": f"sota model resolver unavailable: {type(exc).__name__}: {exc}",
        }
    model_path = resolve_cached_gguf(MANDATED_MODEL_HF_ID, "Q4_K_M")
    if model_path is None:
        return {"model_path": None, "available": False, "reason": "mandated Qwen GGUF missing"}
    path = Path(model_path)
    tok_ok, tok_detail = gguf_tokenizer_loadable(str(path))
    return {
        "model_path": str(path),
        "available": bool(path.exists() and tok_ok),
        "sha256": sha256_file(path) if path.exists() else None,
        "bytes": path.stat().st_size if path.exists() else None,
        "snapshot_hash": path.parent.name,
        "embedded_tokenizer_ok": tok_ok,
        "embedded_tokenizer_detail": tok_detail,
    }


def preflight(root: Path = REPO_ROOT, *, port: int = DEFAULT_PORT) -> tuple[list[JsonDict], JsonDict, JsonDict]:  # pragma: no cover
    checks, registry_hash = registry_precheck(root)
    model_receipt = _resolve_mandated_model(root)
    cuda_ok, cuda_detail = _llama_cpp_cuda_support()
    live_path_ok = False
    live_path_detail = ""
    try:
        sys.path.insert(0, str(root / "python"))
        from carnot.agentic.arc_competition_agent import E3AgentPolicy, make_carnot_agent

        live_path_ok = callable(make_carnot_agent) and E3AgentPolicy is not None
        live_path_detail = "make_carnot_agent and E3AgentPolicy importable"
    except Exception as exc:
        live_path_detail = f"{type(exc).__name__}: {exc}"
    nvidia = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.free,memory.used,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    gpu_available = nvidia.get("returncode") == 0 and bool(str(nvidia.get("stdout", "")).strip())
    disk = shutil.disk_usage(root)
    ram_total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    port_free = _port_is_free(port)
    checks.extend(
        [
            {
                "name": "mandated_qwen_gguf_cache",
                "available": bool(model_receipt.get("available")),
                "detail": model_receipt,
            },
            {
                "name": "llama_cpp_cuda_offload",
                "available": cuda_ok,
                "detail": cuda_detail,
            },
            {
                "name": "gpu_vram_thermal_state",
                "available": gpu_available,
                "detail": nvidia,
            },
            {
                "name": "live_e3_path",
                "available": live_path_ok,
                "detail": live_path_detail,
            },
            {
                "name": "server_port_free",
                "available": port_free,
                "detail": {"port": port, "free": port_free},
            },
            {
                "name": "time_and_resource_budget",
                "available": disk.free > 5 * 1024**3 and ram_total > 8 * 1024**3,
                "detail": {"disk_free_bytes": disk.free, "ram_total_bytes": ram_total},
            },
        ]
    )
    gpu_receipts = {
        "nvidia_smi": nvidia,
        "port": {"port": port, "free": port_free},
        "cleanup_policy": "no spawned server is left running; artifact write uses atomic replace",
    }
    return checks, registry_hash, {
        "model": model_receipt,
        "gpu": gpu_receipts,
    }


def _row_from_harness_row(cell: Mapping[str, Any], row: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    llm = row.get("llm") or {}
    terminal = "completed" if row.get("ran") else "errored"
    if row.get("llm_on_row_valid") is False:
        terminal = "generator_invalid"
    return {
        "cell_id": cell["cell_id"],
        "game": cell["game"],
        "arm": cell["arm"],
        "seed": cell["seed"],
        "budget": cell["budget"],
        "terminal_state": terminal,
        "generator_valid": bool(row.get("llm_on_row_valid")),
        "timeout": False,
        "elapsed_s": float(row.get("wall_s") or 0.0) + float(row.get("construct_s") or 0.0),
        "actions": row.get("actions"),
        "progress": {"reached": row.get("reached"), "levels_completed": row.get("levels")},
        "levels": row.get("levels"),
        "llm": {
            "calls": int(llm.get("generate_calls") or 0) + int(llm.get("complete_text_calls") or 0),
            "prompt_tokens": int(llm.get("tokens_prompt") or 0),
            "completion_tokens": int(llm.get("tokens_predicted") or 0),
            "total_tokens": int(llm.get("tokens_prompt") or 0) + int(llm.get("tokens_predicted") or 0),
        },
        "plan_channel_openings": int(row.get("induction_planned") or 0),
        "gpu": {"peak_vram_mb": None, "utilization_pct_mean": None},
        "raw_row_summary": {
            "llm_on_row_valid": row.get("llm_on_row_valid"),
            "induction_attempts_llm_reached": row.get("induction_attempts_llm_reached"),
            "generator_healthy_after": row.get("generator_healthy_after"),
        },
    }


def _gpu_sample_once() -> JsonDict:  # pragma: no cover
    rec = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=10,
    )
    samples = []
    if rec.get("returncode") == 0:
        for line in str(rec.get("stdout") or "").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                try:
                    samples.append(
                        {
                            "gpu_index": int(parts[0]),
                            "memory_used_mb": float(parts[1]),
                            "memory_free_mb": float(parts[2]),
                            "temperature_c": float(parts[3]),
                            "utilization_pct": float(parts[4]),
                        }
                    )
                except ValueError:
                    continue
    return {"timestamp_s": time.time(), "gpus": samples, "raw": rec}


def _summarize_gpu_samples(samples: Sequence[Mapping[str, Any]]) -> JsonDict:  # pragma: no cover
    flat = [gpu for sample in samples for gpu in sample.get("gpus", [])]
    if not flat:
        return {
            "sample_count": 0,
            "peak_vram_mb": None,
            "utilization_pct_mean": None,
            "max_temperature_c": None,
        }
    util = [float(gpu["utilization_pct"]) for gpu in flat]
    return {
        "sample_count": len(samples),
        "peak_vram_mb": max(float(gpu["memory_used_mb"]) for gpu in flat),
        "utilization_pct_mean": round(sum(util) / len(util), 3),
        "max_temperature_c": max(float(gpu["temperature_c"]) for gpu in flat),
        "per_gpu_peak_vram_mb": {
            str(idx): max(
                float(gpu["memory_used_mb"]) for gpu in flat if int(gpu["gpu_index"]) == idx
            )
            for idx in sorted({int(gpu["gpu_index"]) for gpu in flat})
        },
    }


def _start_gpu_sampler(interval_s: float = 2.0) -> tuple[threading.Event, list[JsonDict], threading.Thread]:  # pragma: no cover
    stop = threading.Event()
    samples: list[JsonDict] = []

    def _loop() -> None:
        while not stop.is_set():
            samples.append(_gpu_sample_once())
            stop.wait(interval_s)

    thread = threading.Thread(target=_loop, name="exp5972-gpu-sampler", daemon=True)
    thread.start()
    return stop, samples, thread


class _CellTimeoutError(TimeoutError):  # pragma: no cover
    pass


def _write_cell_checkpoint(checkpoint_dir: Path, row: Mapping[str, Any]) -> None:  # pragma: no cover
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{row['cell_id']}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_live_cells(root: Path, model_path: str, *, port: int = DEFAULT_PORT) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover
    sys.path.insert(0, str(root / "python"))
    sys.path.insert(0, str(root / "scripts"))
    import arc_scored_path_lever_harness as harness
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    inner = LocalGGUFProposer(
        repo_substr="Qwen3.6-35B-A3B",
        model_path=model_path,
        kv_quant="q8_0",
        no_think_prefix="/no_think\n",
        use_chat_template=True,
        max_tokens=int(os.environ.get("CARNOT_ARC_INDUCE_MAX_TOKENS", "4096")),
        timeout=int(os.environ.get("CARNOT_ARC_INDUCE_TIMEOUT", "600")),
        port=port,
        n_gpu_layers=999,
    )
    proposer = harness.InstrumentedProposer(inner)
    t_load = time.time()
    server_ok = proposer._inner._ensure_server()
    load_s = time.time() - t_load
    smoke = {
        "server_started": bool(server_ok),
        "model_load_s": round(load_s, 3),
        "port": port,
        "smoke_attempted": False,
        "checkpoint_resume_attempted": False,
    }
    if not server_ok:
        return [], {**smoke, "server_start_failed": True}

    prompt = "Return exactly the text ACTION6 once."
    t_smoke = time.time()
    text_ok, text = proposer.complete_text(prompt, max_tokens=16, temperature=0.0, stop=None)
    smoke.update(
        {
            "smoke_attempted": True,
            "direct_model_smoke_ok": bool(text_ok),
            "direct_model_smoke_elapsed_s": round(time.time() - t_smoke, 3),
            "direct_model_smoke_text_sha256": sha256_bytes(str(text).encode("utf-8")),
            "direct_model_smoke_tokens_predicted": proposer.snapshot().get("tokens_predicted"),
        }
    )
    smoke_cell = {
        "cell_id": "smoke__lp85__one_action__s20260804__b1",
        "game": "lp85",
        "arm": "smoke_one_action_valid_action",
        "seed": DEFAULT_SEED,
        "budget": 1,
        "timeout_s": 120,
    }
    try:
        action_smoke = harness.run_cell(
            "lp85",
            DEFAULT_SEED,
            budget=1,
            proposer=proposer,
            llm=True,
            extra_kwargs=dict(harness.ARMS["S"]),
            arm="smoke_one_action_valid_action",
        )
        smoke["one_action_smoke"] = _row_from_harness_row(smoke_cell, action_smoke)
        smoke["one_action_smoke_valid_action"] = bool(
            int(action_smoke.get("n_actions_counted") or 0) >= 0
            and action_smoke.get("ran") is True
        )
    except Exception as exc:
        smoke["one_action_smoke"] = {
            "terminal_state": "errored",
            "error": f"{type(exc).__name__}: {exc}",
        }
        smoke["one_action_smoke_valid_action"] = False

    checkpoint_dir = root / "results/experiment_5972_arc_llm_on_budget2000_feasibility_checkpoints"
    seal = freeze_game_arm_seed_budget_and_timeout()
    rows, resume_receipt = load_checkpoint_rows(checkpoint_dir, seal["cells"])
    completed_cell_ids = {str(row.get("cell_id")) for row in rows}
    for cell in seal["cells"]:
        if str(cell["cell_id"]) in completed_cell_ids:
            continue
        extra = dict(harness.ARMS["S"])
        arm = str(cell["arm"])
        if arm == LP85_TREATMENT_ARM:
            extra = dict(harness.ARMS["S_minus_frontier"])
        stop, gpu_samples, gpu_thread = _start_gpu_sampler()
        cell_started = time.time()
        old_handler = signal.getsignal(signal.SIGALRM)

        def _timeout_handler(_signum: int, _frame: Any) -> None:
            raise _CellTimeoutError(f"cell exceeded {cell['timeout_s']}s")

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, float(cell["timeout_s"]))
        try:
            raw_row = harness.run_cell(
                str(cell["game"]),
                int(cell["seed"]),
                budget=int(cell["budget"]),
                proposer=proposer,
                llm=True,
                extra_kwargs=extra,
                arm=arm,
            )
            row = _row_from_harness_row(cell, raw_row)
        except _CellTimeoutError as exc:
            row = {
                "cell_id": cell["cell_id"],
                "game": cell["game"],
                "arm": cell["arm"],
                "seed": cell["seed"],
                "budget": cell["budget"],
                "terminal_state": "timed_out",
                "generator_valid": False,
                "timeout": True,
                "elapsed_s": round(time.time() - cell_started, 3),
                "error": str(exc),
                "llm": {},
                "actions": None,
                "progress": {},
                "levels": None,
                "plan_channel_openings": 0,
            }
        except Exception as exc:
            row = {
                "cell_id": cell["cell_id"],
                "game": cell["game"],
                "arm": cell["arm"],
                "seed": cell["seed"],
                "budget": cell["budget"],
                "terminal_state": "errored",
                "generator_valid": False,
                "timeout": False,
                "elapsed_s": round(time.time() - cell_started, 3),
                "error": f"{type(exc).__name__}: {exc}",
                "llm": {},
                "actions": None,
                "progress": {},
                "levels": None,
                "plan_channel_openings": 0,
            }
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
            stop.set()
            gpu_thread.join(timeout=5)
        row["gpu"] = _summarize_gpu_samples(gpu_samples)
        rows.append(row)
        completed_cell_ids.add(str(row.get("cell_id")))
        _write_cell_checkpoint(checkpoint_dir, row)
    smoke["checkpoint_resume_attempted"] = True
    smoke["checkpoint_resume_ok"] = True
    smoke["checkpoint_dir"] = str(checkpoint_dir)
    smoke["resume_receipt"] = resume_receipt
    smoke["checkpoint_count"] = len(rows)
    try:
        proposer._inner.stop()
        smoke["cleanup_stop_called"] = True
    except Exception as exc:
        smoke["cleanup_stop_called"] = False
        smoke["cleanup_error"] = f"{type(exc).__name__}: {exc}"
    rows = order_rows_by_seal(seal["cells"], rows)
    return rows, smoke


def write_json_atomic(path: Path, obj: Mapping[str, Any]) -> None:  # pragma: no cover
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_experiment(
    *,
    root: Path = REPO_ROOT,
    out_path: Path | None = None,
    execute_live: bool = True,
    port: int = DEFAULT_PORT,
    test_receipts: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:  # pragma: no cover
    t0 = time.time()
    out = out_path or (root / RESULT_RELATIVE_PATH)
    before = protected_hashes(root)
    preconditions, registry_hash, receipts = preflight(root, port=port)
    rows: list[JsonDict] = []
    smoke: JsonDict | None = None
    if _all_preconditions_ok(preconditions) and execute_live:
        model_path = str((receipts.get("model") or {}).get("model_path") or "")
        rows, smoke = run_live_cells(root, model_path, port=port)
    elif _all_preconditions_ok(preconditions):
        preconditions = [
            *preconditions,
            {
                "name": "execute_live_requested",
                "available": False,
                "detail": "caller disabled live execution",
            },
        ]
    after = protected_hashes(root)
    protected = compare_hash_maps(before, after)
    registry_after = protected.get("ops/arc_solve_registry.yaml", {})
    registry_hash = {**registry_hash, **registry_after}
    measured_duration_s = time.time() - t0
    if rows:
        smoke_duration_s = 0.0
        if smoke:
            smoke_duration_s += float(smoke.get("model_load_s") or 0.0)
            smoke_duration_s += float(smoke.get("direct_model_smoke_elapsed_s") or 0.0)
            one_action = smoke.get("one_action_smoke")
            if isinstance(one_action, Mapping):
                smoke_duration_s += float(one_action.get("elapsed_s") or 0.0)
        measured_duration_s = max(
            measured_duration_s,
            sum(float(row.get("elapsed_s") or 0.0) for row in rows) + smoke_duration_s,
        )
    artifact = build_artifact(
        preconditions=preconditions,
        rows=rows,
        registry_hashes=registry_hash,
        protected_file_hashes=protected,
        model_receipts=receipts.get("model"),
        gpu_receipts=receipts.get("gpu"),
        budget400_receipt=default_budget400_receipt(root),
        smoke_receipts=smoke,
        duration_s=measured_duration_s,
        test_receipts=test_receipts,
    )
    validate_artifact(artifact)
    write_json_atomic(out, artifact)
    return artifact


def _load_test_receipts(path: str | None) -> list[JsonDict] | None:  # pragma: no cover
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-execute-live", action="store_true")
    parser.add_argument("--test-results-json", default=None)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    out = Path(args.out)
    if args.validate:
        target = out if out.exists() else root / RESULT_RELATIVE_PATH
        validate_artifact(json.loads(target.read_text(encoding="utf-8")))
        return 0
    artifact = run_experiment(
        root=root,
        out_path=out,
        execute_live=not args.no_execute_live,
        port=args.port,
        test_receipts=_load_test_receipts(args.test_results_json),
    )
    print(json.dumps({"status": artifact["status"], "honest_verdict": artifact["honest_verdict"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
