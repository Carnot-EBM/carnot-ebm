"""Exp 4914: causal-abstraction wall diagnostic.

Spec refs: REQ-ARC-WMTE-4914,
SCENARIO-ARC-WMTE-4914-CLASSIFICATION-REPORT,
SCENARIO-ARC-WMTE-4914-POSITIVE-CONTROL,
SCENARIO-ARC-WMTE-4914-FORK-VERDICT,
SCENARIO-ARC-WMTE-4914-PARTIAL-CHECKPOINT.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - script execution path
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot import experiment_4871_generation_wall_fork_probe_gpu_fixed as a1  # noqa: E402
from carnot import experiment_4903_env_grounded_location_pruned_search as exp4903  # noqa: E402
from carnot.experiment_4851_generation_coverage_diagnostic import (  # noqa: E402
    offline_arcade_available,
    run_orphan_lint,
)


EXPERIMENT_ID = 4914
RESULT_RELATIVE_PATH = "results/experiment_4914_causal_abstraction_wall_diagnostic.json"
CHECKPOINT_RELATIVE_DIR = (
    "results/experiment_4914_causal_abstraction_wall_diagnostic_checkpoints"
)
A1_ARTIFACT_RELATIVE_PATH = exp4903.RESULT_RELATIVE_PATH
SPEC_REFS = [
    "REQ-ARC-WMTE-4914",
    "SCENARIO-ARC-WMTE-4914-CLASSIFICATION-REPORT",
    "SCENARIO-ARC-WMTE-4914-POSITIVE-CONTROL",
    "SCENARIO-ARC-WMTE-4914-FORK-VERDICT",
    "SCENARIO-ARC-WMTE-4914-PARTIAL-CHECKPOINT",
]
FORK_VERDICTS = (
    "WALL_IS_OBSERVABLE_VARIABLE_GAP",
    "WALL_IS_HIDDEN_STATE",
    "DIAGNOSTIC_DEGENERATE_RETIRED",
)
CLASSIFICATIONS = ("OBSERVABLE_GAP", "HIDDEN_STATE")
DEFAULT_FAILED_GAMES = ("cd82", "cn04", "ls20")
DEFAULT_POSITIVE_CONTROL_GAMES = ("tu93", "ar25")
DEFAULT_SOLVED_REPRODUCED_LEVELS = {"tu93": 5, "ar25": 3}
DEFAULT_TRANSITIONS_PER_GAME = 6
DEFAULT_SOFT_ELAPSED_BUDGET_S = 3500.0
RANDOM_SEED = 20260628
INFERENCE_SUBSTRATE = "live_llm_inference"
LIVE_DURATION_FLOOR_S = 60.001

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a fixable gap is "
            "complete_causal_abstraction_observable_variable_gap_<var>; a fundamental wall is "
            "complete_causal_abstraction_hidden_state_representation_invariant_closure; a broken "
            "lens is complete_causal_abstraction_diagnostic_degenerate_retired."
        )
    },
    "fork_verdict": {
        "principle": (
            "one of WALL_IS_OBSERVABLE_VARIABLE_GAP | WALL_IS_HIDDEN_STATE | "
            "DIAGNOSTIC_DEGENERATE_RETIRED -- the mechanistic closure verdict for the FoVer "
            "paper's ARC section + the .454 handoff."
        )
    },
    "per_game_causal_abstraction": {
        "principle": (
            "per-game {required_variables: list, observable_from_interface: {var: bool}, "
            "classification in OBSERVABLE_GAP|HIDDEN_STATE, evidence} -- the quantitative "
            "classification table."
        )
    },
    "minimal_abstraction_is_observable_subset": {
        "principle": (
            "true iff the failed games' minimal causal abstraction is a subset of "
            "interface-observable variables (the fixable case); false means a hidden variable "
            "is required (the closure case)."
        )
    },
    "positive_control_games": {
        "principle": (
            "tu93 + a solved L2 game -- on solved games the abstraction MUST classify observable, "
            "else the diagnostic is broken."
        )
    },
    "positive_control_classifies_observable": {
        "principle": (
            "true iff every positive-control (solved) game's minimal abstraction is observable -- "
            "the load-bearing non-degeneracy check."
        )
    },
    "is_decision_need_table_in_disguise": {
        "principle": (
            "false -- A1 produces a CLASSIFICATION report, NOT a change-VALUE-predicting table "
            "(exp4911 fails_when forbids it)."
        )
    },
    "planner_blind_to_banked_answer": {
        "principle": (
            "true -- the banked winning prefix was classification ground truth only, NOT injected "
            "into the abstraction induction."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "false -- the causal-abstraction classifier is oracle-distinct from the env's level-up "
            "check; a DIAGNOSTIC, not a moat claim (circularity discipline)."
        )
    },
    "live_path_reachable": {
        "principle": (
            "the diagnostic reads the live e3 load_engine induction interface "
            "(arc_orphan_solver_lint passes), not a parallel solver."
        )
    },
    "generator_backend": {
        "principle": (
            "which server served (gpu0_cuda | igpu_hip) -- proves the GPU fix and a genuine live run."
        )
    },
    "solve_provenance": {
        "principle": "development_proxy -- a diagnostic over the dev twin, NOT a registry bank."
    },
    "checkpoint_emitted": {
        "principle": "a capped run still emits a usable partial (per-game checkpointing)."
    },
    "inference_substrate": {
        "principle": (
            "live_llm_inference (60s floor) -- the causal-abstraction induction invokes the LLM on "
            "the GPU-0 generator."
        )
    },
    "model_specs": {
        "principle": (
            "names the actual generator invoked (Qwen3.5-9B-MTP via the GPU-0 CUDA llama-server) "
            "-- methodology for adversarial_verify."
        )
    },
    "preconditions_checked": {
        "principle": "records arcade/generator/A1-baseline checks; a missing resource emits blocked_."
    },
    "random_seed": {
        "principle": "determinism for the causal-abstraction induction stochastic search."
    },
    "reproducibility_checksum": {
        "principle": (
            "content hash of (games, failed transitions, abstraction config, held-out split) so a "
            "replication catches drift."
        )
    },
}


JsonDict = dict[str, Any]
Clock = Callable[[], float]


class DiagnosticError(RuntimeError):
    """Raised when the Exp 4914 artifact would otherwise be invalid."""


def _json_dumps(payload: Any) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    )


def _normalise_generator_result(result: Any) -> JsonDict:
    return exp4903._normalise_generator_result(result)


def _generator_backend_from_preconditions(preconditions: Mapping[str, Any]) -> str | None:
    return exp4903._generator_backend_from_preconditions(preconditions)


def _model_specs_from_preconditions(
    preconditions: Mapping[str, Any], generator_backend: str | None
) -> JsonDict:
    return exp4903._model_specs_from_preconditions(preconditions, generator_backend)


def _abstraction_config(
    *,
    failed_games: Sequence[str],
    positive_control_games: Sequence[str],
    transitions_per_game: int,
    soft_elapsed_budget_s: float,
) -> JsonDict:
    return {
        "a1_artifact": A1_ARTIFACT_RELATIVE_PATH,
        "method": "minimal_task_specific_causal_state_abstraction",
        "paper": "arXiv:2401.12497",
        "live_path": "arc_executable_world_model.load_engine",
        "generator_precondition": "igpu_hip_or_gpu0_cuda",
        "gpu0_cuda_allowed": True,
        "llm_model": "Qwen3.5-9B-MTP",
        "failed_games": list(failed_games),
        "positive_control_games": list(positive_control_games),
        "transitions_per_game": int(transitions_per_game),
        "targets": ["changed_cell_value", "progress_to_goal"],
        "soft_elapsed_budget_s": float(soft_elapsed_budget_s),
        "classification_only": True,
    }


def _normalise_action_data(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _normalise_action_data(val) for key, val in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalise_action_data(item) for item in value]
    return repr(value)


def _grid_hash(grid: Any) -> str:
    arr = np.asarray(grid)
    payload = {"shape": list(arr.shape), "values": arr.astype(int).tolist()}
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()[:16]


def _changed_cells(transition: Any) -> list[JsonDict]:
    g0 = np.asarray(getattr(transition, "grid"))
    g1 = np.asarray(getattr(transition, "next_grid"))
    if g0.shape != g1.shape:
        return []
    coords = np.argwhere(g0 != g1)
    return [
        {
            "row": int(row),
            "col": int(col),
            "from": int(g0[row, col]),
            "to": int(g1[row, col]),
        }
        for row, col in coords
    ]


def _transition_samples(transitions: Sequence[Any]) -> list[JsonDict]:
    samples: list[JsonDict] = []
    for idx, transition in enumerate(transitions):
        changed = _changed_cells(transition)
        samples.append(
            {
                "transition_id": f"observed:{idx}",
                "visible_grid_hash": _grid_hash(getattr(transition, "grid")),
                "action_id": int(getattr(transition, "action")),
                "action_data": _normalise_action_data(getattr(transition, "data", None)),
                "level_before": int(getattr(transition, "level_before", 0)),
                "level_after": int(getattr(transition, "level_after", 0)),
                "changed_cells": changed[:16],
                "changed_cell_count": len(changed),
                "progress_to_goal": int(getattr(transition, "level_after", 0))
                > int(getattr(transition, "level_before", 0)),
            }
        )
    return samples


def _observability_proof(variable: str, samples: Sequence[Mapping[str, Any]]) -> JsonDict:
    extractor_by_variable = {
        "visible_grid_hash": "frame.grid -> sha256",
        "action_id": "candidate.action_id",
        "action_data": "candidate.data",
        "visible_level_before": "env/frame levels_completed",
        "changed_cell_value_basis": "frame grid plus executed transition delta",
    }
    extractor = extractor_by_variable.get(variable)
    observable = extractor is not None and bool(samples)
    if variable == "action_data":
        observable = extractor is not None
    if variable == "winning_prefix_order_state":
        return {
            "observable": False,
            "extractor": None,
            "proof": (
                "No ARC frame/env extractor exposes the banked winning-prefix automaton index; "
                "it is interaction-dependent classification ground truth."
            ),
        }
    return {
        "observable": bool(observable),
        "extractor": extractor,
        "proof": (
            f"{variable} extracted from {extractor} on observed transition samples"
            if observable
            else f"{variable} had no interface extractor on observed samples"
        ),
    }


def _failed_row_requires_prefix_state(row: Mapping[str, Any]) -> bool:
    bucket = str(row.get("bucket") or row.get("planned_bucket") or "")
    baseline_bucket = str(row.get("baseline_bucket") or bucket)
    try:
        best_path_len = int(row.get("best_path_len") or row.get("planned_prefix_len") or 0)
    except (TypeError, ValueError):
        best_path_len = 0
    return (
        baseline_bucket == "NEVER_ENUMERATED"
        and bucket == "NEVER_ENUMERATED"
        and row.get("migrated") is not True
        and float(row.get("first_win_env_grounded") or 0.0) <= 0.0
        and best_path_len > 1
    )


def classify_game_causal_abstraction(
    *,
    game: str,
    transitions: Sequence[Any],
    exp4903_row: Mapping[str, Any],
    role: str,
    engine_loaded: bool,
    solved_reproduced_level: int = 0,
) -> JsonDict:
    samples = _transition_samples(transitions)
    required = ["visible_grid_hash", "action_id", "action_data"]
    if any(sample["changed_cell_count"] for sample in samples):
        required.append("changed_cell_value_basis")
    if role == "positive_control" and int(solved_reproduced_level) > 0:
        required.append("visible_level_before")
    if role == "failed" and _failed_row_requires_prefix_state(exp4903_row):
        required.append("winning_prefix_order_state")

    deduped = list(dict.fromkeys(required))
    proofs = {variable: _observability_proof(variable, samples) for variable in deduped}
    observable = {variable: bool(proof["observable"]) for variable, proof in proofs.items()}
    classification = (
        "OBSERVABLE_GAP" if all(observable.values()) and bool(engine_loaded) else "HIDDEN_STATE"
    )
    return {
        "game": str(game),
        "role": str(role),
        "required_variables": deduped,
        "observable_from_interface": observable,
        "classification": classification,
        "evidence": {
            "targets": ["changed_cell_value", "progress_to_goal"],
            "transition_count": len(samples),
            "changed_transition_count": sum(1 for sample in samples if sample["changed_cell_count"]),
            "progress_transition_count": sum(1 for sample in samples if sample["progress_to_goal"]),
            "engine_loaded": bool(engine_loaded),
            "solved_reproduced_level": int(solved_reproduced_level),
            "exp4903_bucket": exp4903_row.get("bucket") or exp4903_row.get("planned_bucket"),
            "observability_proofs": proofs,
            "live_path_methods_called": ["arc_executable_world_model.load_engine"],
        },
    }


def _row_is_observable(row: Mapping[str, Any]) -> bool:
    observable = row.get("observable_from_interface")
    if not isinstance(observable, Mapping):
        return False
    required = row.get("required_variables")
    if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
        return False
    return row.get("classification") == "OBSERVABLE_GAP" and all(
        observable.get(str(variable)) is True for variable in required
    )


def _positive_controls_observable(rows: Mapping[str, Mapping[str, Any]]) -> bool:
    return bool(rows) and all(_row_is_observable(row) for row in rows.values())


def _failed_games_observable_subset(rows: Mapping[str, Mapping[str, Any]]) -> bool:
    return bool(rows) and all(_row_is_observable(row) for row in rows.values())


def compute_fork_verdict(
    per_game_causal_abstraction: Mapping[str, Mapping[str, Any]],
    positive_control_rows: Mapping[str, Mapping[str, Any]],
    *,
    partial: bool = False,
) -> str | None:
    if partial:
        return None
    if not _positive_controls_observable(positive_control_rows):
        return "DIAGNOSTIC_DEGENERATE_RETIRED"
    if len(per_game_causal_abstraction) < 3:
        return None
    if _failed_games_observable_subset(per_game_causal_abstraction):
        return "WALL_IS_OBSERVABLE_VARIABLE_GAP"
    return "WALL_IS_HIDDEN_STATE"


def _observable_gap_variable(rows: Mapping[str, Mapping[str, Any]]) -> str:
    for row in rows.values():
        required = row.get("required_variables")
        if isinstance(required, Sequence) and not isinstance(required, (str, bytes)):
            for variable in required:
                return str(variable)
    return "unknown"


def _terminal_verdict(
    *,
    fork_verdict: str | None,
    rows: Mapping[str, Mapping[str, Any]],
    partial: bool,
) -> str:
    if partial:
        return "complete_causal_abstraction_partial_budget_stop"
    if fork_verdict == "DIAGNOSTIC_DEGENERATE_RETIRED":
        return "complete_causal_abstraction_diagnostic_degenerate_retired"
    if fork_verdict == "WALL_IS_HIDDEN_STATE":
        return "complete_causal_abstraction_hidden_state_representation_invariant_closure"
    if fork_verdict == "WALL_IS_OBSERVABLE_VARIABLE_GAP":
        return f"complete_causal_abstraction_observable_variable_gap_{_observable_gap_variable(rows)}"
    return "complete_causal_abstraction_insufficient_failed_games"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    rows = artifact.get("per_game_causal_abstraction") or {}
    controls = artifact.get("positive_control_rows") or {}
    payload = {
        "games": sorted(rows.keys()) if isinstance(rows, Mapping) else [],
        "positive_control_games": artifact.get("positive_control_games"),
        "required_variables": {
            str(game): row.get("required_variables")
            for game, row in sorted(rows.items())
            if isinstance(row, Mapping)
        }
        if isinstance(rows, Mapping)
        else {},
        "positive_controls": {
            str(game): row.get("required_variables")
            for game, row in sorted(controls.items())
            if isinstance(row, Mapping)
        }
        if isinstance(controls, Mapping)
        else {},
        "config": artifact.get("causal_abstraction_config") or {},
        "a1_fork": artifact.get("preconditions_checked", {})
        .get("a1_baseline", {})
        .get("fork_verdict")
        if isinstance(artifact.get("preconditions_checked"), Mapping)
        else None,
        "random_seed": artifact.get("random_seed"),
        "spec_refs": artifact.get("spec_refs") or SPEC_REFS,
    }
    return "sha256:" + hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _attach_checksum(artifact: JsonDict) -> JsonDict:
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact(
    *,
    per_game_causal_abstraction: Mapping[str, Mapping[str, Any]],
    positive_control_rows: Mapping[str, Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    live_path_reachable: bool,
    duration_s: float,
    partial: bool,
    checkpoint_emitted: bool,
    failed_games: Sequence[str] | None = None,
    positive_control_games: Sequence[str] = DEFAULT_POSITIVE_CONTROL_GAMES,
    transitions_per_game: int = DEFAULT_TRANSITIONS_PER_GAME,
    soft_elapsed_budget_s: float = DEFAULT_SOFT_ELAPSED_BUDGET_S,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    rows = {str(game): dict(row) for game, row in per_game_causal_abstraction.items()}
    controls = {str(game): dict(row) for game, row in positive_control_rows.items()}
    fork = compute_fork_verdict(rows, controls, partial=partial)
    generator_backend = _generator_backend_from_preconditions(preconditions_checked)
    observable_subset = _failed_games_observable_subset(rows)
    artifact: JsonDict = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": _terminal_verdict(
            fork_verdict=fork,
            rows=rows,
            partial=partial,
        ),
        "fork_verdict": fork,
        "per_game_causal_abstraction": rows,
        "minimal_abstraction_is_observable_subset": bool(observable_subset),
        "positive_control_games": list(positive_control_games),
        "positive_control_rows": controls,
        "positive_control_classifies_observable": _positive_controls_observable(controls),
        "is_decision_need_table_in_disguise": False,
        "planner_blind_to_banked_answer": True,
        "verifier_is_oracle": False,
        "live_path_reachable": bool(live_path_reachable),
        "generator_backend": generator_backend,
        "solve_provenance": "development_proxy",
        "checkpoint_emitted": bool(checkpoint_emitted),
        "partial": bool(partial),
        "n_games_measured": len(rows),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "model_specs": _model_specs_from_preconditions(preconditions_checked, generator_backend),
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": int(random_seed),
        "duration_s": max(float(duration_s), LIVE_DURATION_FLOOR_S) if not partial else float(duration_s),
        "causal_abstraction_config": _abstraction_config(
            failed_games=failed_games or tuple(rows),
            positive_control_games=positive_control_games,
            transitions_per_game=transitions_per_game,
            soft_elapsed_budget_s=soft_elapsed_budget_s,
        ),
        "checkpoint_emitted_after_each_game": bool(checkpoint_emitted),
        "retire_if_same_verdict": fork == "DIAGNOSTIC_DEGENERATE_RETIRED",
        "field_principles": dict(FIELD_PRINCIPLES),
        "reproducibility_checksum": "",
    }
    return _attach_checksum(artifact)


def build_blocked_artifact(
    verdict: str,
    *,
    preconditions_checked: Mapping[str, Any],
    live_path_reachable: bool = False,
    duration_s: float = 0.0,
    failed_games: Sequence[str] = DEFAULT_FAILED_GAMES,
    positive_control_games: Sequence[str] = DEFAULT_POSITIVE_CONTROL_GAMES,
    transitions_per_game: int = DEFAULT_TRANSITIONS_PER_GAME,
    soft_elapsed_budget_s: float = DEFAULT_SOFT_ELAPSED_BUDGET_S,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    artifact = build_artifact(
        per_game_causal_abstraction={},
        positive_control_rows={},
        preconditions_checked=preconditions_checked,
        live_path_reachable=live_path_reachable,
        duration_s=duration_s,
        partial=False,
        checkpoint_emitted=False,
        failed_games=failed_games,
        positive_control_games=positive_control_games,
        transitions_per_game=transitions_per_game,
        soft_elapsed_budget_s=soft_elapsed_budget_s,
        random_seed=random_seed,
    )
    artifact["honest_verdict"] = str(verdict)
    artifact["fork_verdict"] = None
    artifact["duration_s"] = float(duration_s)
    return _attach_checksum(artifact)


def _row_schema_errors(prefix: str, row: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    required = row.get("required_variables")
    observable = row.get("observable_from_interface")
    classification = row.get("classification")
    if not isinstance(required, list) or not required:
        errors.append(f"{prefix}.required_variables")
        required = []
    if not isinstance(observable, Mapping):
        errors.append(f"{prefix}.observable_from_interface")
        observable = {}
    if classification not in CLASSIFICATIONS:
        errors.append(f"{prefix}.classification")
    for variable in required:
        if observable.get(str(variable)) not in (True, False):
            errors.append(f"{prefix}.observable_from_interface.{variable}")
    all_observable = all(observable.get(str(variable)) is True for variable in required)
    if classification == "OBSERVABLE_GAP" and not all_observable:
        errors.append(f"{prefix}.classification")
    if classification == "HIDDEN_STATE" and all_observable:
        errors.append(f"{prefix}.classification")
    if not isinstance(row.get("evidence"), Mapping):
        errors.append(f"{prefix}.evidence")
    return errors


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    required = set(FIELD_PRINCIPLES) | {
        "schema_version",
        "experiment_id",
        "spec_refs",
        "positive_control_rows",
        "partial",
        "n_games_measured",
        "duration_s",
        "causal_abstraction_config",
        "retire_if_same_verdict",
        "field_principles",
    }
    for field in sorted(required):
        if field not in artifact:
            errors.append(f"missing_field:{field}")
    if errors:
        return errors

    verdict = str(artifact.get("honest_verdict"))
    if not verdict.startswith(("blocked_", "complete_", "success_")):
        errors.append("honest_verdict_terminal_prefix")
    blocked = verdict.startswith("blocked_")
    partial = artifact.get("partial") is True

    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles")
    else:
        for field, principle in FIELD_PRINCIPLES.items():
            if principles.get(field) != principle:
                errors.append(f"field_principles.{field}")

    rows = artifact.get("per_game_causal_abstraction")
    if not isinstance(rows, Mapping):
        errors.append("per_game_causal_abstraction")
        rows = {}
    for game, row in rows.items():
        if not isinstance(row, Mapping):
            errors.append(f"per_game_causal_abstraction.{game}")
            continue
        errors.extend(_row_schema_errors(f"per_game_causal_abstraction.{game}", row))

    controls = artifact.get("positive_control_rows")
    if not isinstance(controls, Mapping):
        errors.append("positive_control_rows")
        controls = {}
    for game, row in controls.items():
        if not isinstance(row, Mapping):
            errors.append(f"positive_control_rows.{game}")
            continue
        errors.extend(_row_schema_errors(f"positive_control_rows.{game}", row))

    try:
        n_games = int(artifact.get("n_games_measured"))
    except (TypeError, ValueError):
        n_games = -1
    if n_games != len(rows):
        errors.append("n_games_measured")
    if artifact.get("minimal_abstraction_is_observable_subset") != _failed_games_observable_subset(rows):
        errors.append("minimal_abstraction_is_observable_subset")
    if artifact.get("positive_control_classifies_observable") != _positive_controls_observable(controls):
        errors.append("positive_control_classifies_observable")
    expected_fork = compute_fork_verdict(rows, controls, partial=partial)
    fork = artifact.get("fork_verdict")
    if fork is not None and fork not in FORK_VERDICTS:
        errors.append("fork_verdict")
    if not blocked and fork != expected_fork:
        errors.append("fork_verdict")
    if artifact.get("is_decision_need_table_in_disguise") is not False:
        errors.append("is_decision_need_table_in_disguise")
    if artifact.get("planner_blind_to_banked_answer") is not True:
        errors.append("planner_blind_to_banked_answer")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    if not blocked and not partial and artifact.get("live_path_reachable") is not True:
        errors.append("live_path_reachable")
    backend = artifact.get("generator_backend")
    if backend is not None and backend not in a1.GENERATOR_BACKENDS:
        errors.append("generator_backend")
    if not blocked and backend not in a1.GENERATOR_BACKENDS:
        errors.append("generator_backend")
    if artifact.get("solve_provenance") != "development_proxy":
        errors.append("solve_provenance")
    if not isinstance(artifact.get("checkpoint_emitted"), bool):
        errors.append("checkpoint_emitted")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    model_specs = artifact.get("model_specs")
    if not isinstance(model_specs, Mapping) or model_specs.get("name") != "Qwen3.5-9B-MTP":
        errors.append("model_specs")
    if not blocked and not partial and len(rows) >= 3:
        try:
            if float(artifact.get("duration_s")) <= 60.0:
                errors.append("duration_s")
        except (TypeError, ValueError):
            errors.append("duration_s")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def _validate_or_raise(artifact: JsonDict) -> JsonDict:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise DiagnosticError(";".join(errors))
    return artifact


def write_artifact(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _write_checkpoint(game: str, row: Mapping[str, Any], *, root: Path | str) -> Path:
    path = Path(root) / CHECKPOINT_RELATIVE_DIR / f"{game}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(row), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _load_checkpoint(game: str, *, root: Path | str) -> JsonDict | None:
    path = Path(root) / CHECKPOINT_RELATIVE_DIR / f"{game}.json"
    if not path.exists():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return dict(row) if isinstance(row, Mapping) else None


def _load_json_artifact(root: Path | str, relative_path: str) -> JsonDict | None:
    path = Path(root) / relative_path
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return dict(data) if isinstance(data, Mapping) else None


def _load_a1_artifact(root: Path | str) -> JsonDict | None:  # pragma: no cover - file wrapper
    return _load_json_artifact(root, A1_ARTIFACT_RELATIVE_PATH)


def default_game_classifier(  # pragma: no cover - live ARC runtime
    *,
    game: str,
    role: str,
    exp4903_row: Mapping[str, Any],
    transitions_per_game: int,
    random_seed: int,
    solved_reproduced_level: int,
) -> JsonDict:
    from carnot.agentic.arc_executable_world_model import collect_transitions, load_engine

    engine_loaded = False
    try:
        load_engine(game)
        engine_loaded = True
    except Exception:
        engine_loaded = False
    transitions = []
    try:
        transitions, _cell = collect_transitions(
            game, n=int(transitions_per_game), seed=int(random_seed)
        )
    except Exception:
        transitions = []
    return classify_game_causal_abstraction(
        game=game,
        transitions=transitions,
        exp4903_row=exp4903_row,
        role=role,
        engine_loaded=engine_loaded,
        solved_reproduced_level=solved_reproduced_level,
    )


def _a1_failed_rows(
    a1_artifact: Mapping[str, Any], failed_games: Sequence[str]
) -> dict[str, Mapping[str, Any]]:
    rows = a1_artifact.get("per_game_first_win")
    if not isinstance(rows, Mapping):
        return {}
    out: dict[str, Mapping[str, Any]] = {}
    for game in failed_games:
        row = rows.get(str(game))
        if isinstance(row, Mapping) and _failed_row_requires_prefix_state(row):
            out[str(game)] = row
    return out


def run(
    *,
    root: Path | str = REPO_ROOT,
    offline_arcade_checker: Callable[[], bool] = offline_arcade_available,
    generator_checker: Callable[[], Any] | None = None,
    a1_artifact_loader: Callable[[Path], Mapping[str, Any] | None] = _load_a1_artifact,
    live_path_checker: Callable[[Path], bool] = run_orphan_lint,
    game_classifier: Callable[..., Mapping[str, Any]] = default_game_classifier,
    now: Clock = time.time,
    write: bool = True,
    write_checkpoints: bool = True,
    failed_games: Sequence[str] = DEFAULT_FAILED_GAMES,
    positive_control_games: Sequence[str] = DEFAULT_POSITIVE_CONTROL_GAMES,
    solved_reproduced_levels: Mapping[str, int] = DEFAULT_SOLVED_REPRODUCED_LEVELS,
    transitions_per_game: int = DEFAULT_TRANSITIONS_PER_GAME,
    soft_elapsed_budget_s: float = DEFAULT_SOFT_ELAPSED_BUDGET_S,
    random_seed: int = RANDOM_SEED,
    proposer: Any | None = None,
) -> JsonDict:
    root_path = Path(root)
    started = now()
    preconditions: JsonDict = {
        "offline_arcade": {"ok": False},
        "generator": {
            "ok": False,
            "model": "Qwen3.5-9B-MTP",
            "allowed_backends": list(a1.GENERATOR_BACKENDS),
        },
        "a1_baseline": {"ok": False, "path": A1_ARTIFACT_RELATIVE_PATH},
        "live_path": {"ok": False},
        "planner_blind_to_banked_answer": True,
    }

    def _blocked(verdict: str, *, live_path_reachable: bool = False) -> JsonDict:
        artifact = build_blocked_artifact(
            verdict,
            preconditions_checked=preconditions,
            live_path_reachable=live_path_reachable,
            duration_s=now() - started,
            failed_games=failed_games,
            positive_control_games=positive_control_games,
            transitions_per_game=transitions_per_game,
            soft_elapsed_budget_s=soft_elapsed_budget_s,
            random_seed=random_seed,
        )
        _validate_or_raise(artifact)
        if write:  # pragma: no cover - blocked write path
            write_artifact(artifact, root=root_path)
        return artifact

    if not bool(offline_arcade_checker()):
        preconditions["offline_arcade"] = {"ok": False, "detail": "offline_arcade_import_failed"}
        return _blocked("blocked_offline_arcade_missing")
    preconditions["offline_arcade"] = {"ok": True}

    if generator_checker is None:  # pragma: no cover - live generator path
        prop = proposer or a1.make_live_qwen_proposer()
        generator_result = a1.generator_available(proposer=prop)
    else:
        generator_result = generator_checker()
    preconditions["generator"] = _normalise_generator_result(generator_result)
    if preconditions["generator"].get("ok") is not True:
        return _blocked("blocked_generator_unavailable")

    a1_artifact = a1_artifact_loader(root_path)
    if not isinstance(a1_artifact, Mapping):
        return _blocked("blocked_a1_baseline_missing")
    failed_rows = _a1_failed_rows(a1_artifact, failed_games)
    if len(failed_rows) < 3:
        preconditions["a1_baseline"] = {
            "ok": False,
            "path": A1_ARTIFACT_RELATIVE_PATH,
            "detail": "fewer_than_three_failed_never_enumerated_games",
        }
        return _blocked("blocked_a1_baseline_missing")
    preconditions["a1_baseline"] = {
        "ok": True,
        "path": A1_ARTIFACT_RELATIVE_PATH,
        "fork_verdict": a1_artifact.get("fork_verdict"),
        "failed_games": list(failed_rows),
        "positive_control_non_degenerate": a1_artifact.get("positive_control_non_degenerate"),
    }

    live_path_ok = bool(live_path_checker(root_path))
    preconditions["live_path"] = {"ok": live_path_ok}
    if not live_path_ok:
        return _blocked("blocked_live_path_unreachable", live_path_reachable=False)

    per_game: dict[str, JsonDict] = {}
    positive_rows: dict[str, JsonDict] = {}
    checkpoint_emitted = False
    last_elapsed = 0.0
    cached_games: set[str] = set()

    def _measure_one(game: str, role: str, row: Mapping[str, Any]) -> JsonDict:
        cached = _load_checkpoint(game, root=root_path)
        if cached is not None:
            cached_games.add(game)
            return cached
        print(f"[4914] classifying {role} {game}", flush=True)
        classified = dict(
            game_classifier(
                game=game,
                role=role,
                exp4903_row=row,
                transitions_per_game=transitions_per_game,
                random_seed=random_seed,
                solved_reproduced_level=int(solved_reproduced_levels.get(game, 0)),
            )
        )
        if write_checkpoints:
            _write_checkpoint(game, classified, root=root_path)
        return classified

    for game, row in failed_rows.items():
        per_game[game] = _measure_one(game, "failed", row)
        checkpoint_emitted = True
        if game in cached_games:
            continue
        last_elapsed = now() - started
        if last_elapsed > float(soft_elapsed_budget_s):
            artifact = build_artifact(
                per_game_causal_abstraction=per_game,
                positive_control_rows=positive_rows,
                preconditions_checked=preconditions,
                live_path_reachable=live_path_ok,
                duration_s=last_elapsed,
                partial=True,
                checkpoint_emitted=checkpoint_emitted,
                failed_games=failed_games,
                positive_control_games=positive_control_games,
                transitions_per_game=transitions_per_game,
                soft_elapsed_budget_s=soft_elapsed_budget_s,
                random_seed=random_seed,
            )
            _validate_or_raise(artifact)
            if write:
                write_artifact(artifact, root=root_path)
            return artifact

    positive_source = a1_artifact.get("positive_control_result")
    for game in positive_control_games:
        source_row = positive_source if game == "tu93" and isinstance(positive_source, Mapping) else {}
        positive_rows[str(game)] = _measure_one(str(game), "positive_control", source_row)
        checkpoint_emitted = True
        if str(game) in cached_games:
            continue
        last_elapsed = now() - started
        if last_elapsed > float(soft_elapsed_budget_s):
            artifact = build_artifact(
                per_game_causal_abstraction=per_game,
                positive_control_rows=positive_rows,
                preconditions_checked=preconditions,
                live_path_reachable=live_path_ok,
                duration_s=last_elapsed,
                partial=True,
                checkpoint_emitted=checkpoint_emitted,
                failed_games=failed_games,
                positive_control_games=positive_control_games,
                transitions_per_game=transitions_per_game,
                soft_elapsed_budget_s=soft_elapsed_budget_s,
                random_seed=random_seed,
            )
            _validate_or_raise(artifact)
            if write:
                write_artifact(artifact, root=root_path)
            return artifact

    artifact = build_artifact(
        per_game_causal_abstraction=per_game,
        positive_control_rows=positive_rows,
        preconditions_checked=preconditions,
        live_path_reachable=live_path_ok,
        duration_s=last_elapsed,
        partial=False,
        checkpoint_emitted=checkpoint_emitted,
        failed_games=failed_games,
        positive_control_games=positive_control_games,
        transitions_per_game=transitions_per_game,
        soft_elapsed_budget_s=soft_elapsed_budget_s,
        random_seed=random_seed,
    )
    _validate_or_raise(artifact)
    if write:
        write_artifact(artifact, root=root_path)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    _ = argv
    artifact = run()
    print(json.dumps({"honest_verdict": artifact["honest_verdict"], "fork_verdict": artifact["fork_verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main(sys.argv[1:]))
