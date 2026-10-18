"""Exp 4872: CEGIS refinement for induced executable ARC world models.

Spec refs: REQ-ARC-WMTE-4872,
SCENARIO-ARC-WMTE-4872-A1-LOW-ACCURACY-GATE,
SCENARIO-ARC-WMTE-4872-REPAIR-ACCEPTANCE,
SCENARIO-ARC-WMTE-4872-TRULY-HELDOUT-DELTA.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - script execution path
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot import experiment_4871_generation_wall_fork_probe_gpu_fixed as a1  # noqa: E402
from carnot.experiment_4851_generation_coverage_diagnostic import (  # noqa: E402
    offline_arcade_available,
    run_orphan_lint,
)


EXPERIMENT_ID = 4872
RESULT_RELATIVE_PATH = "results/experiment_4872_cegis_world_model_refinement.json"
CHECKPOINT_RELATIVE_DIR = "results/experiment_4872_cegis_world_model_refinement_checkpoints"
A1_RESULT_RELATIVE_PATH = a1.RESULT_RELATIVE_PATH
SPEC_REFS = [
    "REQ-ARC-WMTE-4872",
    "SCENARIO-ARC-WMTE-4872-A1-LOW-ACCURACY-GATE",
    "SCENARIO-ARC-WMTE-4872-REPAIR-ACCEPTANCE",
    "SCENARIO-ARC-WMTE-4872-TRULY-HELDOUT-DELTA",
]
DEFAULT_HELDOUT_GAMES = a1.HELDOUT_GAMES
DEFAULT_HELDOUT_TRANSITIONS = 24
DEFAULT_OBSERVED_PREFIX_TRANSITIONS = 24
DEFAULT_MAX_REPAIR_COUNTEREXAMPLES = 4
DEFAULT_MAX_ROUNDS = 1
DEFAULT_BOOTSTRAP_ITERATIONS = 1000
DEFAULT_SOFT_ELAPSED_BUDGET_S = 4200.0
HIGH_ACCURACY_THRESHOLD = 0.5
RANDOM_SEED = 20260627
INFERENCE_SUBSTRATE = "live_llm_inference"

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a real lift is success_cegis_engine_accuracy_lift_<delta>; "
            "a null is complete_cegis_no_heldout_accuracy_lift_residual_<cause>."
        )
    },
    "cegis_heldout_accuracy_delta_median": {
        "principle": (
            "median (refined - baseline) held-out transition accuracy -- did CEGIS MOVE "
            "the inducer wall?"
        )
    },
    "cegis_heldout_accuracy_delta_ci95": {
        "principle": (
            "bootstrap CI95 of the delta; PASS requires it to exclude 0 (not a noise lift)."
        )
    },
    "per_game_accuracy_delta": {
        "principle": (
            "per-game {baseline, refined, delta, counterexamples_fixed, cegis_rounds} -- "
            "the quantitative intervention table."
        )
    },
    "delta_on_truly_heldout_split": {
        "principle": (
            "true -- the re-measure split is DISJOINT from the repair counterexamples "
            "(B1 audits; else it is a tautology)."
        )
    },
    "positive_control_passed": {
        "principle": (
            "a known-correctable misprediction was fixed -> a flat null is a real ceiling, "
            "not a harness no-op."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "false -- the held-out transition score is oracle-distinct from the env's "
            "level-up check (circularity discipline)."
        )
    },
    "live_path_reachable": {
        "principle": (
            "the refinement improves the live e3 induction/repair path "
            "(arc_orphan_solver_lint passes), not a parallel solver."
        )
    },
    "solve_provenance": {
        "principle": (
            "development_proxy -- an inducer-accuracy measurement, NOT a banked level."
        )
    },
    "checkpoint_emitted": {
        "principle": (
            "a capped run still emits a usable partial (per-game + per-round checkpointing)."
        )
    },
    "inference_substrate": {
        "principle": (
            "live_llm_inference (60s floor) -- CEGIS repair invokes the LLM on the GPU-0 "
            "generator."
        )
    },
    "model_specs": {
        "principle": (
            "names the actual generator (Qwen3.5-9B-MTP via the GPU-0 CUDA llama-server)."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records arcade/generator/A1-baseline checks; a missing resource emits blocked_."
        )
    },
    "random_seed": {"principle": "determinism for the CEGIS stochastic repair search."},
    "reproducibility_checksum": {
        "principle": (
            "content hash of (games, A1 baseline, CEGIS config, held-out split) so a "
            "replication catches drift."
        )
    },
}


JsonDict = dict[str, Any]
Clock = Callable[[], float]


class DiagnosticError(RuntimeError):
    """Raised when the Exp 4872 artifact would otherwise be invalid."""


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def load_a1_artifact(root: Path | str = REPO_ROOT) -> JsonDict | None:
    path = Path(root) / A1_RESULT_RELATIVE_PATH
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return dict(data) if isinstance(data, Mapping) else None


def _normalise_generator_result(result: Any) -> JsonDict:
    return a1._normalise_generator_result(result)


def _generator_backend_from_preconditions(preconditions: Mapping[str, Any]) -> str | None:
    return a1._generator_backend_from_preconditions(preconditions)


def _model_specs_from_preconditions(
    preconditions: Mapping[str, Any], generator_backend: str | None
) -> JsonDict:
    return a1._model_specs_from_preconditions(preconditions, generator_backend)


def _cegis_config(
    *,
    heldout_transitions: int,
    observed_prefix_transitions: int,
    max_repair_counterexamples: int,
    max_rounds: int,
    soft_elapsed_budget_s: float,
    heldout_games: Sequence[str],
    bootstrap_iterations: int,
) -> JsonDict:
    return {
        "live_path": "arc_executable_world_model.load_engine/refactor -> E3 executable program",
        "llm_model": "Qwen3.5-9B-MTP",
        "generator_precondition": "igpu_hip_or_gpu0_cuda",
        "gpu0_cuda_allowed": True,
        "a1_artifact": A1_RESULT_RELATIVE_PATH,
        "a1_low_accuracy_threshold": HIGH_ACCURACY_THRESHOLD,
        "heldout_transitions": int(heldout_transitions),
        "observed_prefix_transitions": int(observed_prefix_transitions),
        "max_repair_counterexamples": int(max_repair_counterexamples),
        "max_rounds": int(max_rounds),
        "soft_elapsed_budget_s": float(soft_elapsed_budget_s),
        "heldout_games": list(heldout_games),
        "bootstrap_iterations": int(bootstrap_iterations),
    }


def _transition_id(index: int) -> int:
    return int(index)


def score_engine(engine: Callable[..., Any], transitions: Sequence[Any]) -> float:
    from carnot.agentic import arc_executable_world_model as e3

    return float(e3.WorldModelVerifier(list(transitions)).score(engine).accuracy)


def _correct_transition_indices(engine: Callable[..., Any], transitions: Sequence[Any]) -> set[int]:
    from carnot.agentic import arc_executable_world_model as e3

    score = e3.WorldModelVerifier(list(transitions)).score(engine, max_mismatch=len(transitions) + 1)
    mismatches = {int(row["i"]) for row in score.mismatches if "i" in row}
    return set(range(len(transitions))) - mismatches


def evaluate_repair_acceptance(
    *,
    previous_engine: Callable[..., Any],
    repaired_engine: Callable[..., Any],
    repair_counterexamples: Sequence[Any],
    observed_prefix: Sequence[Any],
) -> JsonDict:
    previous_repair_correct = _correct_transition_indices(previous_engine, repair_counterexamples)
    repaired_repair_correct = _correct_transition_indices(repaired_engine, repair_counterexamples)
    fixed = sorted(repaired_repair_correct - previous_repair_correct)
    previous_observed = score_engine(previous_engine, observed_prefix)
    repaired_observed = score_engine(repaired_engine, observed_prefix)
    observed_regressed = repaired_observed + 1e-12 < previous_observed
    accepted = bool(fixed) and not observed_regressed
    return {
        "accepted": accepted,
        "fixed_count": len(fixed),
        "fixed_transition_ids": [_transition_id(index) for index in fixed],
        "repair_accuracy_before": round(score_engine(previous_engine, repair_counterexamples), 6),
        "repair_accuracy_after": round(score_engine(repaired_engine, repair_counterexamples), 6),
        "observed_prefix_accuracy_before": round(previous_observed, 6),
        "observed_prefix_accuracy_after": round(repaired_observed, 6),
        "observed_regressed": bool(observed_regressed),
    }


def select_repair_and_remeasure_splits(
    *,
    engine: Callable[..., Any],
    heldout_transitions: Sequence[Any],
    max_repair_counterexamples: int,
    seed: int,
) -> JsonDict:
    correct = _correct_transition_indices(engine, heldout_transitions)
    failures = [i for i in range(len(heldout_transitions)) if i not in correct]
    rng = random.Random(seed)
    rng.shuffle(failures)
    repair_indices = sorted(failures[: max(0, int(max_repair_counterexamples))])
    remeasure_indices = [i for i in range(len(heldout_transitions)) if i not in set(repair_indices)]
    return {
        "repair_indices": repair_indices,
        "remeasure_indices": remeasure_indices,
        "repair_counterexamples": [heldout_transitions[i] for i in repair_indices],
        "remeasure_transitions": [heldout_transitions[i] for i in remeasure_indices],
    }


def _delta_values(per_game_accuracy_delta: Mapping[str, Mapping[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in per_game_accuracy_delta.values():
        if not isinstance(row, Mapping):
            continue
        try:
            values.append(float(row["delta"]))
        except (KeyError, TypeError, ValueError):
            continue
    return values


def bootstrap_ci95(values: Sequence[float], *, iterations: int, seed: int) -> list[float | None]:
    vals = [float(value) for value in values]
    if not vals:
        return [None, None]
    if len(set(vals)) == 1:
        value = round(vals[0], 6)
        return [value, value]
    rng = random.Random(seed)
    samples: list[float] = []
    count = max(1, int(iterations))
    for _ in range(count):
        draw = [rng.choice(vals) for _ in vals]
        samples.append(float(median(draw)))
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return [round(float(lo), 6), round(float(hi), 6)]


def _split_is_disjoint(row: Mapping[str, Any]) -> bool:
    repair = {int(item) for item in row.get("repair_transition_ids") or []}
    remeasure = {int(item) for item in row.get("remeasure_transition_ids") or []}
    return bool(remeasure) and repair.isdisjoint(remeasure)


def _all_rows_disjoint(per_game_accuracy_delta: Mapping[str, Mapping[str, Any]]) -> bool:
    if not per_game_accuracy_delta:
        return True
    return all(_split_is_disjoint(row) for row in per_game_accuracy_delta.values())


def _terminal_verdict(
    *,
    median_delta: float | None,
    ci95: Sequence[float | None],
    n_games: int,
    split_disjoint: bool,
    positive_control_passed: bool,
    partial: bool,
) -> str:
    if partial:
        return "complete_cegis_no_heldout_accuracy_lift_residual_partial_budget_stop"
    if n_games < 3:
        return "complete_cegis_no_heldout_accuracy_lift_residual_too_few_games"
    if not split_disjoint:
        return "complete_cegis_no_heldout_accuracy_lift_residual_split_not_disjoint"
    if not positive_control_passed:
        return "complete_cegis_no_heldout_accuracy_lift_residual_positive_control_failed"
    lo = ci95[0] if len(ci95) >= 1 else None
    hi = ci95[1] if len(ci95) >= 2 else None
    if median_delta is None or median_delta <= 0.0:
        return "complete_cegis_no_heldout_accuracy_lift_residual_nonpositive_delta"
    if lo is None or hi is None or float(lo) <= 0.0 <= float(hi):
        return "complete_cegis_no_heldout_accuracy_lift_residual_ci_includes_zero"
    return f"success_cegis_engine_accuracy_lift_{float(median_delta):.6f}"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    rows = artifact.get("per_game_accuracy_delta") or {}
    a1_baselines = {
        game: row.get("a1_artifact_baseline")
        for game, row in sorted(rows.items())
        if isinstance(row, Mapping)
    }
    split = {
        game: {
            "repair": list(row.get("repair_transition_ids") or []),
            "remeasure": list(row.get("remeasure_transition_ids") or []),
        }
        for game, row in sorted(rows.items())
        if isinstance(row, Mapping)
    }
    payload = {
        "games": sorted(rows.keys()) if isinstance(rows, Mapping) else [],
        "a1_baselines": a1_baselines,
        "cegis_config": artifact.get("cegis_config") or {},
        "heldout_split": split,
        "random_seed": artifact.get("random_seed"),
        "spec_refs": artifact.get("spec_refs") or SPEC_REFS,
    }
    return "sha256:" + hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _attach_checksum(artifact: JsonDict) -> JsonDict:
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_blocked_artifact(
    verdict: str,
    *,
    preconditions_checked: Mapping[str, Any],
    live_path_reachable: bool = False,
    duration_s: float = 0.0,
    random_seed: int = RANDOM_SEED,
    heldout_transitions: int = DEFAULT_HELDOUT_TRANSITIONS,
    observed_prefix_transitions: int = DEFAULT_OBSERVED_PREFIX_TRANSITIONS,
    max_repair_counterexamples: int = DEFAULT_MAX_REPAIR_COUNTEREXAMPLES,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    soft_elapsed_budget_s: float = DEFAULT_SOFT_ELAPSED_BUDGET_S,
    heldout_games: Sequence[str] = DEFAULT_HELDOUT_GAMES,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
) -> JsonDict:
    generator_backend = _generator_backend_from_preconditions(preconditions_checked)
    artifact: JsonDict = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": str(verdict),
        "cegis_heldout_accuracy_delta_median": None,
        "cegis_heldout_accuracy_delta_ci95": [None, None],
        "per_game_accuracy_delta": {},
        "delta_on_truly_heldout_split": True,
        "positive_control_passed": False,
        "verifier_is_oracle": False,
        "live_path_reachable": bool(live_path_reachable),
        "solve_provenance": "development_proxy",
        "checkpoint_emitted": False,
        "partial": False,
        "n_games_measured": 0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "model_specs": _model_specs_from_preconditions(preconditions_checked, generator_backend),
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": int(random_seed),
        "cegis_config": _cegis_config(
            heldout_transitions=heldout_transitions,
            observed_prefix_transitions=observed_prefix_transitions,
            max_repair_counterexamples=max_repair_counterexamples,
            max_rounds=max_rounds,
            soft_elapsed_budget_s=soft_elapsed_budget_s,
            heldout_games=heldout_games,
            bootstrap_iterations=bootstrap_iterations,
        ),
        "retire_if_same_verdict": True,
        "duration_s": float(duration_s),
        "field_principles": dict(FIELD_PRINCIPLES),
        "reproducibility_checksum": "",
    }
    return _attach_checksum(artifact)


def build_artifact(
    *,
    per_game_accuracy_delta: Mapping[str, Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    live_path_reachable: bool,
    duration_s: float,
    partial: bool,
    checkpoint_emitted: bool,
    positive_control_passed: bool | None = None,
    random_seed: int = RANDOM_SEED,
    heldout_transitions: int = DEFAULT_HELDOUT_TRANSITIONS,
    observed_prefix_transitions: int = DEFAULT_OBSERVED_PREFIX_TRANSITIONS,
    max_repair_counterexamples: int = DEFAULT_MAX_REPAIR_COUNTEREXAMPLES,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    soft_elapsed_budget_s: float = DEFAULT_SOFT_ELAPSED_BUDGET_S,
    heldout_games: Sequence[str] = DEFAULT_HELDOUT_GAMES,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
) -> JsonDict:
    rows = {str(game): dict(row) for game, row in per_game_accuracy_delta.items()}
    deltas = _delta_values(rows)
    med = round(float(median(deltas)), 6) if deltas else None
    ci95 = bootstrap_ci95(deltas, iterations=bootstrap_iterations, seed=random_seed)
    split_disjoint = _all_rows_disjoint(rows)
    positive = (
        any(int(row.get("counterexamples_fixed") or 0) > 0 for row in rows.values())
        if positive_control_passed is None
        else bool(positive_control_passed)
    )
    generator_backend = _generator_backend_from_preconditions(preconditions_checked)
    artifact: JsonDict = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": _terminal_verdict(
            median_delta=med,
            ci95=ci95,
            n_games=len(rows),
            split_disjoint=split_disjoint,
            positive_control_passed=positive,
            partial=partial,
        ),
        "cegis_heldout_accuracy_delta_median": med,
        "cegis_heldout_accuracy_delta_ci95": ci95,
        "per_game_accuracy_delta": rows,
        "delta_on_truly_heldout_split": split_disjoint,
        "positive_control_passed": positive,
        "verifier_is_oracle": False,
        "live_path_reachable": bool(live_path_reachable),
        "solve_provenance": "development_proxy",
        "checkpoint_emitted": bool(checkpoint_emitted),
        "partial": bool(partial),
        "n_games_measured": len(rows),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "model_specs": _model_specs_from_preconditions(preconditions_checked, generator_backend),
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": int(random_seed),
        "cegis_config": _cegis_config(
            heldout_transitions=heldout_transitions,
            observed_prefix_transitions=observed_prefix_transitions,
            max_repair_counterexamples=max_repair_counterexamples,
            max_rounds=max_rounds,
            soft_elapsed_budget_s=soft_elapsed_budget_s,
            heldout_games=heldout_games,
            bootstrap_iterations=bootstrap_iterations,
        ),
        "retire_if_same_verdict": True,
        "duration_s": float(duration_s),
        "field_principles": dict(FIELD_PRINCIPLES),
        "reproducibility_checksum": "",
    }
    return _attach_checksum(artifact)


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:  # pragma: no cover - defensive validator
    errors: list[str] = []
    required = set(FIELD_PRINCIPLES) | {
        "schema_version",
        "experiment_id",
        "spec_refs",
        "partial",
        "n_games_measured",
        "cegis_config",
        "retire_if_same_verdict",
        "duration_s",
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

    rows = artifact.get("per_game_accuracy_delta")
    if not isinstance(rows, Mapping):
        errors.append("per_game_accuracy_delta")
        rows = {}
    for game, row in rows.items():
        if not isinstance(row, Mapping):
            errors.append(f"per_game_accuracy_delta.{game}")
            continue
        for key in ("baseline", "refined", "delta"):
            try:
                value = float(row.get(key))
            except (TypeError, ValueError):
                errors.append(f"per_game_accuracy_delta.{game}.{key}")
                continue
            if key != "delta" and not 0.0 <= value <= 1.0:
                errors.append(f"per_game_accuracy_delta.{game}.{key}")
        for key in ("counterexamples_fixed", "cegis_rounds"):
            try:
                if int(row.get(key)) < 0:
                    errors.append(f"per_game_accuracy_delta.{game}.{key}")
            except (TypeError, ValueError):
                errors.append(f"per_game_accuracy_delta.{game}.{key}")
        if not _split_is_disjoint(row):
            errors.append(f"per_game_accuracy_delta.{game}.heldout_split")

    try:
        n_games_measured = int(artifact.get("n_games_measured"))
    except (TypeError, ValueError):
        n_games_measured = -1
    if n_games_measured != len(rows):
        errors.append("n_games_measured")
    if artifact.get("delta_on_truly_heldout_split") != _all_rows_disjoint(rows):
        errors.append("delta_on_truly_heldout_split")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    if artifact.get("solve_provenance") != "development_proxy":
        errors.append("solve_provenance")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if not isinstance(artifact.get("checkpoint_emitted"), bool):
        errors.append("checkpoint_emitted")
    if artifact.get("retire_if_same_verdict") is not True:
        errors.append("retire_if_same_verdict")
    if not blocked and not partial and len(rows) >= 3 and artifact.get("live_path_reachable") is not True:
        errors.append("live_path_reachable")
    model_specs = artifact.get("model_specs")
    if not isinstance(model_specs, Mapping) or model_specs.get("name") != "Qwen3.5-9B-MTP":
        errors.append("model_specs")
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


def refine_game_with_live_cegis(  # pragma: no cover - live ARC/LLM boundary
    *,
    game: str,
    a1_row: Mapping[str, Any],
    proposer: Any,
    heldout_transition_budget: int = DEFAULT_HELDOUT_TRANSITIONS,
    observed_prefix_budget: int = DEFAULT_OBSERVED_PREFIX_TRANSITIONS,
    max_repair_counterexamples: int = DEFAULT_MAX_REPAIR_COUNTEREXAMPLES,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    random_seed: int = RANDOM_SEED,
    root: Path | str = REPO_ROOT,
    round_checkpoint: Callable[[str, Mapping[str, Any]], Any] | None = None,
) -> JsonDict:
    from carnot.agentic import arc_executable_world_model as e3

    root_path = Path(root)
    seed_base = int(random_seed) + sum(ord(ch) for ch in str(game))
    observed_prefix, cell = e3.collect_transitions(
        game, n=int(observed_prefix_budget), warmup=False, seed=seed_base
    )
    heldout, _heldout_cell = e3.collect_transitions(
        game, n=int(heldout_transition_budget), warmup=False, seed=seed_base + 1009
    )
    world_model_path = root_path / "results" / "arc_e3" / game / "world_model.py"
    if not world_model_path.exists():
        ok, note = proposer.induce(game, observed_prefix, cell)
        if not ok:
            return {
                "game": game,
                "baseline": 0.0,
                "refined": 0.0,
                "delta": 0.0,
                "a1_artifact_baseline": float(a1_row.get("engine_heldout_accuracy") or 0.0),
                "counterexamples_fixed": 0,
                "cegis_rounds": 0,
                "accepted_repairs": 0,
                "repair_transition_ids": [],
                "remeasure_transition_ids": list(range(len(heldout))),
                "observed_prefix_accuracy_before": 0.0,
                "observed_prefix_accuracy_after": 0.0,
                "repair_counterexample_count": 0,
                "remeasure_transition_count": len(heldout),
                "rounds": [{"round": 0, "accepted": False, "residual": str(note)[:160]}],
                "residual": "initial_induction_failed",
            }

    current_engine, _is_done = e3.load_engine(game)
    split = select_repair_and_remeasure_splits(
        engine=current_engine,
        heldout_transitions=heldout,
        max_repair_counterexamples=max_repair_counterexamples,
        seed=seed_base + 202,
    )
    repair_counterexamples = list(split["repair_counterexamples"])
    remeasure_transitions = list(split["remeasure_transitions"])
    repair_ids = list(split["repair_indices"])
    remeasure_ids = list(split["remeasure_indices"])
    baseline = score_engine(current_engine, remeasure_transitions)
    observed_before = score_engine(current_engine, observed_prefix)
    rounds: list[JsonDict] = []
    total_fixed = 0
    accepted_repairs = 0
    previous_code = world_model_path.read_text(encoding="utf-8") if world_model_path.exists() else ""

    for round_index in range(1, int(max_rounds) + 1):
        if not repair_counterexamples:
            rounds.append({"round": round_index, "accepted": False, "residual": "no_repair_counterexamples"})
            break
        vr = e3.WorldModelVerifier(repair_counterexamples).score(current_engine)
        ok, note = proposer.refactor(game, vr)
        if not ok:
            rounds.append({"round": round_index, "accepted": False, "residual": str(note)[:160]})
            if previous_code:
                world_model_path.write_text(previous_code, encoding="utf-8")
            continue
        try:
            repaired_engine, _goal = e3.load_engine(game)
            acceptance = evaluate_repair_acceptance(
                previous_engine=current_engine,
                repaired_engine=repaired_engine,
                repair_counterexamples=repair_counterexamples,
                observed_prefix=observed_prefix,
            )
        except Exception as exc:
            acceptance = {
                "accepted": False,
                "fixed_count": 0,
                "observed_regressed": True,
                "residual": repr(exc)[:160],
            }
        acceptance["round"] = round_index
        acceptance["proposer_note"] = str(note)[:160]
        rounds.append(dict(acceptance))
        if acceptance.get("accepted") is True:
            current_engine = repaired_engine
            accepted_repairs += 1
            total_fixed += int(acceptance.get("fixed_count") or 0)
            previous_code = world_model_path.read_text(encoding="utf-8")
        elif previous_code:
            world_model_path.write_text(previous_code, encoding="utf-8")
        if round_checkpoint is not None:
            round_checkpoint(
                game,
                {
                    "game": game,
                    "rounds": list(rounds),
                    "counterexamples_fixed": int(total_fixed),
                    "accepted_repairs": int(accepted_repairs),
                },
            )

    refined = score_engine(current_engine, remeasure_transitions)
    observed_after = score_engine(current_engine, observed_prefix)
    return {
        "game": str(game),
        "baseline": round(float(baseline), 6),
        "refined": round(float(refined), 6),
        "delta": round(float(refined - baseline), 6),
        "a1_artifact_baseline": round(float(a1_row.get("engine_heldout_accuracy") or 0.0), 6),
        "counterexamples_fixed": int(total_fixed),
        "cegis_rounds": len(rounds),
        "accepted_repairs": int(accepted_repairs),
        "repair_transition_ids": repair_ids,
        "remeasure_transition_ids": remeasure_ids,
        "observed_prefix_accuracy_before": round(float(observed_before), 6),
        "observed_prefix_accuracy_after": round(float(observed_after), 6),
        "repair_counterexample_count": len(repair_counterexamples),
        "remeasure_transition_count": len(remeasure_transitions),
        "rounds": rounds,
        "residual": "accepted_repair" if accepted_repairs else "no_accepted_repair",
    }


def _a1_precondition(a1_artifact: Mapping[str, Any] | None) -> JsonDict:
    if not isinstance(a1_artifact, Mapping):
        return {"ok": False, "detail": "missing_or_malformed"}
    per_game = a1_artifact.get("per_game_fork")
    try:
        median_accuracy = float(a1_artifact.get("median_engine_heldout_accuracy"))
    except (TypeError, ValueError):
        return {"ok": False, "detail": "missing_median_engine_heldout_accuracy"}
    if not isinstance(per_game, Mapping) or not per_game:
        return {
            "ok": False,
            "detail": "missing_per_game_baselines",
            "median_engine_heldout_accuracy": median_accuracy,
        }
    low = median_accuracy < HIGH_ACCURACY_THRESHOLD
    return {
        "ok": bool(low),
        "detail": "ok" if low else "a1_not_inducer_ceiling",
        "median_engine_heldout_accuracy": median_accuracy,
        "per_game_baselines_present": True,
        "n_games": len(per_game),
    }


def run(
    *,
    root: Path | str = REPO_ROOT,
    offline_arcade_checker: Callable[[], bool] = offline_arcade_available,
    generator_checker: Callable[[], Any] | None = None,
    a1_artifact_loader: Callable[[Path], Mapping[str, Any] | None] = load_a1_artifact,
    live_path_checker: Callable[[Path], bool] = run_orphan_lint,
    game_refiner: Callable[..., Mapping[str, Any]] = refine_game_with_live_cegis,
    now: Clock = time.time,
    write: bool = True,
    write_checkpoints: bool = True,
    heldout_games: Sequence[str] = DEFAULT_HELDOUT_GAMES,
    heldout_transition_budget: int = DEFAULT_HELDOUT_TRANSITIONS,
    observed_prefix_budget: int = DEFAULT_OBSERVED_PREFIX_TRANSITIONS,
    max_repair_counterexamples: int = DEFAULT_MAX_REPAIR_COUNTEREXAMPLES,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    soft_elapsed_budget_s: float = DEFAULT_SOFT_ELAPSED_BUDGET_S,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
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
        "a1_baseline": {"ok": False},
        "live_path": {"ok": False},
    }

    def _blocked(verdict: str, *, live_path_reachable: bool = False) -> JsonDict:
        artifact = build_blocked_artifact(
            verdict,
            preconditions_checked=preconditions,
            live_path_reachable=live_path_reachable,
            duration_s=now() - started,
            random_seed=random_seed,
            heldout_transitions=heldout_transition_budget,
            observed_prefix_transitions=observed_prefix_budget,
            max_repair_counterexamples=max_repair_counterexamples,
            max_rounds=max_rounds,
            soft_elapsed_budget_s=soft_elapsed_budget_s,
            heldout_games=heldout_games,
            bootstrap_iterations=bootstrap_iterations,
        )
        _validate_or_raise(artifact)
        if write:
            write_artifact(artifact, root=root_path)
        return artifact

    if not bool(offline_arcade_checker()):
        preconditions["offline_arcade"] = {"ok": False}
        return _blocked("blocked_offline_arcade_missing")
    preconditions["offline_arcade"] = {"ok": True}

    prop = proposer
    if generator_checker is None:
        prop = prop or a1.make_live_qwen_proposer()
        generator_result = a1.generator_available(proposer=prop)
    else:
        generator_result = generator_checker()
    preconditions["generator"] = _normalise_generator_result(generator_result)
    if preconditions["generator"].get("ok") is not True:
        return _blocked("blocked_generator_unavailable")

    a1_artifact = a1_artifact_loader(root_path)
    a1_precondition = _a1_precondition(a1_artifact)
    preconditions["a1_baseline"] = a1_precondition
    if a1_precondition.get("ok") is not True:
        detail = a1_precondition.get("detail")
        verdict = (
            "blocked_a1_not_inducer_ceiling"
            if detail == "a1_not_inducer_ceiling"
            else "blocked_a1_baseline_missing"
        )
        return _blocked(verdict)

    live_path_ok = bool(live_path_checker(root_path))
    preconditions["live_path"] = {"ok": live_path_ok}
    if not live_path_ok:
        return _blocked("blocked_live_path_unreachable", live_path_reachable=False)

    prop = prop or a1.make_live_qwen_proposer()
    a1_rows = dict(a1_artifact.get("per_game_fork") or {})  # type: ignore[union-attr]
    rows: dict[str, JsonDict] = {}
    checkpoint_emitted = False
    partial = False

    def _round_checkpoint(game: str, row: Mapping[str, Any]) -> None:
        nonlocal checkpoint_emitted
        if write_checkpoints:
            _write_checkpoint(game, row, root=root_path)
            checkpoint_emitted = True

    for game in heldout_games:
        if game not in a1_rows:
            continue
        cached = _load_checkpoint(game, root=root_path)
        if cached is not None and "delta" in cached:
            rows[str(game)] = cached
            checkpoint_emitted = True
            continue
        print(f"[4872] refining {game} ({len(rows) + 1}/{len(heldout_games)})", flush=True)
        row = dict(
            game_refiner(
                game=str(game),
                a1_row=a1_rows[game],
                proposer=prop,
                heldout_transition_budget=heldout_transition_budget,
                observed_prefix_budget=observed_prefix_budget,
                max_repair_counterexamples=max_repair_counterexamples,
                max_rounds=max_rounds,
                random_seed=random_seed,
                root=root_path,
                round_checkpoint=_round_checkpoint,
            )
        )
        rows[str(game)] = row
        if write_checkpoints:
            _write_checkpoint(str(game), row, root=root_path)
            checkpoint_emitted = True
        elapsed = now() - started
        print(
            "[4872] "
            f"{game}: baseline={row.get('baseline')} refined={row.get('refined')} "
            f"delta={row.get('delta')} fixed={row.get('counterexamples_fixed')} "
            f"elapsed_s={elapsed:.1f}",
            flush=True,
        )
        if elapsed >= float(soft_elapsed_budget_s):
            partial = True
            break

    artifact = build_artifact(
        per_game_accuracy_delta=rows,
        preconditions_checked=preconditions,
        live_path_reachable=True,
        duration_s=now() - started,
        partial=partial,
        checkpoint_emitted=checkpoint_emitted,
        positive_control_passed=None,
        random_seed=random_seed,
        heldout_transitions=heldout_transition_budget,
        observed_prefix_transitions=observed_prefix_budget,
        max_repair_counterexamples=max_repair_counterexamples,
        max_rounds=max_rounds,
        soft_elapsed_budget_s=soft_elapsed_budget_s,
        heldout_games=heldout_games,
        bootstrap_iterations=bootstrap_iterations,
    )
    _validate_or_raise(artifact)
    if write:
        write_artifact(artifact, root=root_path)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI boundary
    _ = argv
    artifact = run(
        max_rounds=int(os.environ.get("CARNOT_ARC_4872_MAX_ROUNDS", str(DEFAULT_MAX_ROUNDS))),
        heldout_transition_budget=int(
            os.environ.get("CARNOT_ARC_4872_HELDOUT_TRANSITIONS", str(DEFAULT_HELDOUT_TRANSITIONS))
        ),
        observed_prefix_budget=int(
            os.environ.get(
                "CARNOT_ARC_4872_OBSERVED_PREFIX_TRANSITIONS",
                str(DEFAULT_OBSERVED_PREFIX_TRANSITIONS),
            )
        ),
        max_repair_counterexamples=int(
            os.environ.get(
                "CARNOT_ARC_4872_MAX_REPAIR_COUNTEREXAMPLES",
                str(DEFAULT_MAX_REPAIR_COUNTEREXAMPLES),
            )
        ),
        bootstrap_iterations=int(
            os.environ.get(
                "CARNOT_ARC_4872_BOOTSTRAP_ITERATIONS",
                str(DEFAULT_BOOTSTRAP_ITERATIONS),
            )
        ),
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main(sys.argv[1:]))
