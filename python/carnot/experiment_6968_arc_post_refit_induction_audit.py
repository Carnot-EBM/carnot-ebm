"""Audit one frozen live ARC engine without collecting or changing game data.

Spec refs: REQ-ARC-WMTE-6968 and SCENARIO-ARC-WMTE-6968-*.

The live run can finish without preserving the grids that produced an engine.
That case is blocked evidence, not a license to collect a different rollout.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np


JsonDict = dict[str, Any]

EXPERIMENT_ID = 6968
RANDOM_SEED = 6_968_202_609_04
BOOTSTRAP_RESAMPLES = 10_000
TARGET_ENGINE_SHA16 = "da8ffce3e3910d6d"
TARGET_N_CTX = 98_304
INFERENCE_SUBSTRATE = "fresh_process_replay_of_frozen_live_agent_engine"
REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
SCORER_PATH = Path("scripts/arc_e3_induced_model_quality.py")
STRUCTURAL_SCORER_PATH = Path("python/carnot/agentic/arc_executable_world_model.py")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
RUNS_PATH = Path("results/arc_leaderboard_eval_runs")
ATTEMPTS_PATH = Path("results/arc_e3/r11l/attempts")
OUTPUT_PATH = Path("results/experiment_6968_arc_post_refit_induction_audit.json")

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "selected_run_provenance",
    "engine_hash",
    "prompt_hash",
    "transition_hash",
    "scorer_hash",
    "rows",
    "per_transition_rows",
    "split_rows",
    "purity_rows",
    "engine_execution_rows",
    "control_rows",
    "prefix_exact_accuracy",
    "heldout_exact_accuracy",
    "heldout_changing_accuracy",
    "heldout_noop_accuracy",
    "generalization_gap",
    "paired_control_delta_rows",
    "memorization_signature_rows",
    "arc_induction_audit_complete_score",
    "arc_induction_generalization_positive_score",
    "solve_claimed",
    "level_claimed",
    "registry_updated",
    "submitted_to_leaderboard",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "A reason for each required field makes the evidence contract reviewable.",
    "preconditions_checked": "Exact checks prevent absent live evidence from becoming an inferred result.",
    "inference_substrate": "The declaration separates frozen engine replay from new model generation.",
    "duration_s": "Measured wall time shows how much audit work ran.",
    "source_artifact_hashes": "Content hashes bind every source used by the audit.",
    "selected_run_provenance": "Environment and time select the run without using its score.",
    "engine_hash": "The full digest binds results to the archived 170-line program.",
    "prompt_hash": "The prompt digest identifies what the frozen engine was allowed to see.",
    "transition_hash": "The transition digest detects any later row or ordering change.",
    "scorer_hash": "The scorer digest binds metric meaning to exact source code.",
    "rows": "Run-discovery rows expose accepted and rejected candidates.",
    "per_transition_rows": "Complete source rows let reviewers recompute every metric.",
    "split_rows": "Explicit memberships make the shown-prefix boundary reproducible.",
    "purity_rows": "Per-row visibility checks reveal prompt or feedback leakage.",
    "engine_execution_rows": "Fresh-process receipts show that the frozen program executed.",
    "control_rows": "Matched controls prevent a weak engine from earning credit against no baseline.",
    "prefix_exact_accuracy": "Shown-row fit measures transcription and cannot prove transfer.",
    "heldout_exact_accuracy": "Unshown exact predictions measure whole-transition transfer.",
    "heldout_changing_accuracy": "Changing rows prevent no-op prevalence from inflating quality.",
    "heldout_noop_accuracy": "No-op rows reveal hallucinated changes separately.",
    "generalization_gap": "The prefix-minus-held-out gap exposes fit that does not transfer.",
    "paired_control_delta_rows": "Paired transition deltas preserve the uncertainty unit.",
    "memorization_signature_rows": "Structural and behavioral signals distinguish lookup from rules.",
    "arc_induction_audit_complete_score": "Completion requires every source and prediction row to terminate.",
    "arc_induction_generalization_positive_score": "Credit requires leakage-free superiority above the paired interval floor.",
    "solve_claimed": "Engine quality is not a game solve and this field must remain false.",
    "level_claimed": "Transition quality is not a level completion and this field must remain false.",
    "registry_updated": "A read-only audit must not change the solve registry.",
    "submitted_to_leaderboard": "Offline replay cannot imply a leaderboard submission.",
    "random_seed": "The fixed seed makes the paired bootstrap reproducible.",
    "reproducibility_checksum": "A timing-free digest detects scientific-content drift.",
    "gate_check_summary": "Failed checks include expected and observed values for repair.",
    "verifier_is_oracle": "False records that the verifier scores observations but does not create them.",
    "verdict_class": "A closed class prevents blocked or null evidence from reading as positive.",
    "honest_verdict": "A terminal prefix makes the outcome machine-classifiable.",
}


def canonical_json(value: Any) -> bytes:
    """Return stable JSON bytes for hashes and exact comparison keys."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    """Return one SHA-256 digest with the repository prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str | None:
    """Hash a file and keep a missing source explicit."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def _read_json_object(path: Path) -> JsonDict | None:
    """Read a JSON object while treating malformed external evidence as absent."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _gate(check: str, expected: Any, observed: Any, *, passed: bool | None = None) -> JsonDict:
    """Record both sides of one fail-closed comparison."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": expected == observed if passed is None else bool(passed),
    }


def _failed_gates(checks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Reduce failed checks to the diagnostic schema required by the task."""

    return [
        {
            "failed_check": row.get("check"),
            "expected_value": row.get("expected_value"),
            "observed_value": row.get("observed_value"),
        }
        for row in checks
        if row.get("passed") is not True
    ]


def _iter_refinement_rounds(game_row: Mapping[str, Any]):
    """Yield each recorded induction round without trusting aggregate scores."""

    diagnostics = game_row.get("policy_diagnostics", {})
    attempts = diagnostics.get("induction_attempts", []) if isinstance(diagnostics, Mapping) else []
    for attempt_index, attempt in enumerate(attempts if isinstance(attempts, list) else []):
        if not isinstance(attempt, Mapping):
            continue
        rounds = attempt.get("refinement_rounds", [])
        for round_index, row in enumerate(rounds if isinstance(rounds, list) else []):
            if isinstance(row, Mapping):
                yield attempt_index, round_index, attempt, row


def _manifest_engine_timestamp(repo_root: Path, engine_sha16: str) -> str | None:
    """Read the durable archive timestamp for one content-addressed engine."""

    manifest = repo_root / ATTEMPTS_PATH / "manifest.jsonl"
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    matches = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, Mapping) and row.get("sha256_16") == engine_sha16:
            matches.append(str(row.get("ts")))
    return matches[0] if len(matches) == 1 else None


def discover_run_candidates(
    repo_root: Path, *, target_engine_sha16: str = TARGET_ENGINE_SHA16
) -> list[JsonDict]:
    """Find target-engine run rows by content, environment, and recorded times."""

    candidates: list[JsonDict] = []
    run_dir = repo_root / RUNS_PATH
    if not run_dir.is_dir():
        return candidates
    manifest_timestamp = _manifest_engine_timestamp(repo_root, target_engine_sha16)
    for path in sorted(run_dir.glob("*.json")):
        document = _read_json_object(path)
        if document is None:
            continue
        per_game = document.get("per_game", [])
        for game_index, game_row in enumerate(per_game if isinstance(per_game, list) else []):
            if not isinstance(game_row, Mapping) or game_row.get("game") != "r11l":
                continue
            provenance = game_row.get("generator_provenance", {})
            n_ctx = provenance.get("n_ctx") if isinstance(provenance, Mapping) else None
            for attempt_index, round_index, attempt, round_row in _iter_refinement_rounds(game_row):
                source_hash = round_row.get("engine_source_sha256", {})
                sha16 = source_hash.get("sha256_16") if isinstance(source_hash, Mapping) else None
                if sha16 != target_engine_sha16:
                    continue
                candidates.append(
                    {
                        "path": str(path.relative_to(repo_root)),
                        "run_sha256": sha256_path(path),
                        "complete": document.get("complete") is True,
                        "policy": document.get("policy"),
                        "game": "r11l",
                        "game_index": game_index,
                        "n_ctx": n_ctx,
                        "run_started_at": document.get("run_started_at"),
                        "run_completed_at": document.get("run_completed_at"),
                        "run_file_mtime_ns": path.stat().st_mtime_ns,
                        "engine_emitted_at": round_row.get("engine_emitted_at")
                        or manifest_timestamp,
                        "engine_sha256_16": sha16,
                        "engine_chars": source_hash.get("chars"),
                        "attempt_index": attempt_index,
                        "round_index": round_index,
                        "attempt_reason": attempt.get("reason"),
                        "prompt_sha256": round_row.get("prompt_sha256"),
                        "transition_source_path": round_row.get("transition_source_path"),
                    }
                )
    return candidates


def select_run(candidates: Sequence[Mapping[str, Any]]) -> tuple[JsonDict | None, str]:
    """Select the earliest qualifying target row without consulting any score."""

    eligible = [
        dict(row)
        for row in candidates
        if row.get("complete") is True
        and row.get("policy") == "e3"
        and row.get("game") == "r11l"
        and row.get("n_ctx") == TARGET_N_CTX
    ]
    if not eligible:
        return None, "no_completed_target_run"
    eligible.sort(
        key=lambda row: (
            str(row.get("engine_emitted_at") or "~"),
            str(row.get("run_started_at") or "~"),
            str(row.get("path")),
            int(row.get("attempt_index", 0)),
            int(row.get("round_index", 0)),
        )
    )
    return eligible[0], "selected_by_environment_engine_hash_and_timestamp"


def resolve_engine(
    repo_root: Path, target_engine_sha16: str = TARGET_ENGINE_SHA16
) -> tuple[Path | None, list[JsonDict]]:
    """Resolve exactly one archived file whose bytes match its name digest."""

    attempt_dir = repo_root / ATTEMPTS_PATH
    rows: list[JsonDict] = []
    for path in sorted(attempt_dir.glob(f"wm_*__{target_engine_sha16}.py")):
        digest = sha256_path(path)
        matches = bool(digest and digest.removeprefix("sha256:").startswith(target_engine_sha16))
        rows.append(
            {
                "path": str(path.relative_to(repo_root)),
                "sha256": digest,
                "name_hash_matches_bytes": matches,
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
        )
    valid = [repo_root / row["path"] for row in rows if row["name_hash_matches_bytes"]]
    return (valid[0] if len(valid) == 1 and len(rows) == 1 else None), rows


def _safe_relative_source(repo_root: Path, raw_path: Any) -> Path | None:
    """Resolve a recorded source only when it stays inside the selected checkout."""

    if not isinstance(raw_path, str) or not raw_path or Path(raw_path).is_absolute():
        return None
    path = (repo_root / raw_path).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError:
        return None
    return path


def rebuild_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    prompt_row_ids: Sequence[Any],
    repair_feedback_row_ids: Sequence[Any],
) -> tuple[JsonDict, list[JsonDict], list[JsonDict]]:
    """Rebuild the exact prompt prefix and reject feedback leakage."""

    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row.get("index", -1)))
    all_ids = [str(row.get("transition_id")) for row in ordered]
    prompt_ids = [str(value) for value in prompt_row_ids]
    feedback_ids = {str(value) for value in repair_feedback_row_ids}
    prompt_set = set(prompt_ids)
    heldout_ids = [row_id for row_id in all_ids if row_id not in prompt_set]
    split = {
        "shown_row_ids": prompt_ids,
        "heldout_row_ids": heldout_ids,
        "shown_count": len(prompt_ids),
        "heldout_count": len(heldout_ids),
    }
    purity = [
        {
            "transition_id": row_id,
            "prompt_visible": row_id in prompt_set,
            "repair_feedback_visible": row_id in feedback_ids,
            "pure": row_id not in prompt_set and row_id not in feedback_ids,
        }
        for row_id in heldout_ids
    ]
    checks = [
        _gate("unique_transition_ids", len(all_ids), len(set(all_ids))),
        _gate(
            "prompt_ids_present_once",
            prompt_ids,
            [row_id for row_id in all_ids if row_id in prompt_set],
        ),
        _gate("shown_rows_form_ordered_prefix", prompt_ids, all_ids[: len(prompt_ids)]),
        _gate("heldout_row_count_positive", ">=1", len(heldout_ids), passed=len(heldout_ids) >= 1),
        _gate(
            "heldout_rows_absent_from_repair_feedback",
            [],
            sorted(set(heldout_ids) & feedback_ids),
        ),
    ]
    return split, purity, _failed_gates(checks)


def _array(value: Any) -> np.ndarray:
    """Convert one grid through a strict two-dimensional integer boundary."""

    array = np.asarray(value, dtype=np.int64)
    if array.ndim != 2:
        raise ValueError("two_dimensional_grid_required")
    return array


def score_prediction(
    transition: Mapping[str, Any], prediction: Any, *, exception: str | None = None
) -> JsonDict:
    """Score one prediction with exact, changed-cell, and no-op channels."""

    source = _array(transition.get("grid"))
    target = _array(transition.get("next_grid"))
    changing = not np.array_equal(source, target)
    level_up = int(transition.get("level_after", 0)) > int(transition.get("level_before", 0))
    base = {
        "transition_id": str(transition.get("transition_id")),
        "status": "exception" if exception else "complete",
        "terminal": True,
        "exception": exception,
        "changing": changing,
        "level_up_renderer_row": level_up,
        "dynamics_eligible": not level_up,
    }
    if exception is not None:
        return {
            **base,
            "exact_transition_correct": False,
            "exact_cell_accuracy": 0.0,
            "changed_cell_recall": 0.0 if changing else None,
            "changed_cell_precision": 0.0 if changing else None,
            "changing_transition_correct": False if changing and not level_up else None,
            "noop_correct": False if not changing and not level_up else None,
        }
    try:
        predicted = _array(prediction)
    except (TypeError, ValueError) as error:
        return score_prediction(
            transition,
            None,
            exception=f"{type(error).__name__}: {error}",
        )
    shape_matches = predicted.shape == target.shape
    exact = bool(shape_matches and np.array_equal(predicted, target))
    if not shape_matches:
        cell_accuracy = 0.0
        recall = 0.0 if changing else None
        precision = 0.0 if changing else None
    else:
        cell_accuracy = float(np.mean(predicted == target))
        true_change = source != target
        predicted_change = predicted != source
        correct_changed = (predicted == target) & true_change
        recall = float(correct_changed.sum() / true_change.sum()) if changing else None
        precision = (
            float(correct_changed.sum() / predicted_change.sum())
            if changing and predicted_change.any()
            else (0.0 if changing else None)
        )
    return {
        **base,
        "exact_transition_correct": exact,
        "exact_cell_accuracy": cell_accuracy,
        "changed_cell_recall": recall,
        "changed_cell_precision": precision,
        "changing_transition_correct": exact if changing and not level_up else None,
        "noop_correct": exact if not changing and not level_up else None,
    }


def _deny_generated_side_effects(event: str, args: tuple[Any, ...]) -> None:  # pragma: no cover
    """Deny writes, network use, and subprocess escape after dependencies load."""

    if event == "open" and len(args) >= 2:
        mode = args[1]
        if isinstance(mode, str) and any(marker in mode for marker in "wax+"):
            raise PermissionError("generated_engine_write_denied")
        if isinstance(mode, int):
            write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
            if mode & write_flags:
                raise PermissionError("generated_engine_write_denied")
    if event in {"socket.connect", "socket.bind", "subprocess.Popen", "os.system"}:
        raise PermissionError("generated_engine_external_effect_denied")


def _worker_replay(
    engine_path: Path, rows: Sequence[Mapping[str, Any]]
) -> JsonDict:  # pragma: no cover
    """Load generated code after the audit hook and return only JSON-safe outputs."""

    sys.addaudithook(_deny_generated_side_effects)
    spec = importlib.util.spec_from_file_location("frozen_arc_engine", engine_path)
    if spec is None or spec.loader is None:
        raise ImportError("engine_module_spec_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    engine = getattr(module, "engine")
    outputs = []
    for row in rows:
        try:
            prediction = engine(
                _array(row.get("grid")).copy(),
                int(row.get("action", 0)),
                row.get("data"),
            )
            outputs.append({"prediction": _array(prediction).tolist(), "exception": None})
        except Exception as error:  # noqa: BLE001 - generated code failures are measurements.
            outputs.append(
                {
                    "prediction": None,
                    "exception": f"{type(error).__name__}: {str(error)[:240]}",
                }
            )
    return {"worker_pid": os.getpid(), "outputs": outputs}


def _worker_main(argv: Sequence[str]) -> int:  # pragma: no cover
    """Serve the private fresh-process protocol used by the parent audit."""

    payload = json.loads(sys.stdin.read())
    result = _worker_replay(Path(argv[1]), payload["rows"])
    sys.stdout.write(json.dumps(result))
    return 0


def execute_engine_fresh(
    engine_path: Path, rows: Sequence[Mapping[str, Any]], *, timeout_s: float = 10.0
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Replay one frozen engine in a new process with side effects denied."""

    started = time.monotonic()
    payload_rows = [
        {
            "transition_id": str(row.get("transition_id")),
            "grid": row.get("grid"),
            "next_grid": row.get("next_grid"),
            "action": row.get("action"),
            "data": row.get("data"),
            "level_before": row.get("level_before", 0),
            "level_after": row.get("level_after", 0),
        }
        for row in rows
    ]
    process: subprocess.Popen[str] | None = None
    result: JsonDict = {}
    process_error: str | None = None
    with tempfile.TemporaryDirectory(prefix="carnot-exp6968-") as temporary:
        temp_root = Path(temporary)
        copied_engine = temp_root / "frozen_engine.py"
        shutil.copyfile(engine_path, copied_engine)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(copied_engine)]
        environment = {
            "PATH": os.defpath,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        }
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=temp_root,
            env=environment,
        )
        try:
            stdout, stderr = process.communicate(
                json.dumps({"rows": payload_rows}), timeout=timeout_s
            )
            if process.returncode == 0:
                parsed = json.loads(stdout)
                result = parsed if isinstance(parsed, dict) else {}
            else:
                process_error = f"worker_exit_{process.returncode}: {stderr[:240]}"
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            process_error = f"TimeoutExpired: {timeout_s}s"
        except json.JSONDecodeError as error:
            process_error = f"JSONDecodeError: {error}"
    outputs = result.get("outputs", []) if isinstance(result, Mapping) else []
    scored: list[JsonDict] = []
    for index, row in enumerate(rows):
        output_present = index < len(outputs) and isinstance(outputs[index], Mapping)
        output = outputs[index] if output_present else {}
        exception = (
            output.get("exception") if output_present else process_error or "worker_missing_output"
        )
        score = score_prediction(
            row,
            output.get("prediction"),
            exception=str(exception) if exception else None,
        )
        score["source_index"] = int(row.get("index", index))
        scored.append(score)
    execution = [
        {
            "fresh_process": True,
            "restricted_process": True,
            "write_guard_enabled": True,
            "network_guard_enabled": True,
            "parent_pid": os.getpid(),
            "worker_pid": result.get("worker_pid")
            if result
            else (process.pid if process else None),
            "returncode": process.returncode if process else None,
            "timeout_s": timeout_s,
            "duration_s": round(time.monotonic() - started, 6),
            "rows_requested": len(rows),
            "rows_returned": len(outputs),
            "process_error": process_error,
            "terminal": len(scored) == len(rows) and all(row["terminal"] for row in scored),
        }
    ]
    return scored, execution


def _delta_signature(row: Mapping[str, Any]) -> tuple[tuple[int, int, int], ...]:
    """Encode the absolute changed-cell writes of one shown transition."""

    source = _array(row.get("grid"))
    target = _array(row.get("next_grid"))
    if source.shape != target.shape:
        return ()
    return tuple(
        (int(r), int(c), int(target[r, c])) for r, c in np.argwhere(source != target).tolist()
    )


def _apply_delta(grid: Any, delta: Sequence[Sequence[int]]) -> list[list[int]]:
    """Apply one prefix-derived absolute delta to a new input grid."""

    prediction = _array(grid).copy()
    for row, column, value in delta:
        if 0 <= int(row) < prediction.shape[0] and 0 <= int(column) < prediction.shape[1]:
            prediction[int(row), int(column)] = int(value)
    return prediction.tolist()


def _transition_key(row: Mapping[str, Any]) -> bytes:
    """Key row-table lookup on input, action, and action data only."""

    return canonical_json(
        {
            "grid": row.get("grid"),
            "action": row.get("action"),
            "data": row.get("data"),
        }
    )


def score_controls(
    shown_rows: Sequence[Mapping[str, Any]], heldout_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Score four prefix-only controls on identical held-out rows."""

    changing_deltas = [_delta_signature(row) for row in shown_rows]
    changing_deltas = [delta for delta in changing_deltas if delta]
    if changing_deltas:
        counts = Counter(changing_deltas)
        constant_delta = min(counts, key=lambda delta: (-counts[delta], delta))
    else:
        constant_delta = ()
    table = {_transition_key(row): row.get("next_grid") for row in shown_rows}
    control_rows: list[JsonDict] = []
    for heldout in heldout_rows:
        source = _array(heldout.get("grid"))
        nearest: tuple[int, str, Mapping[str, Any]] | None = None
        for shown in shown_rows:
            shown_grid = _array(shown.get("grid"))
            distance = (
                int(np.sum(source != shown_grid)) if source.shape == shown_grid.shape else 10**12
            )
            candidate = (distance, str(shown.get("transition_id")), shown)
            if nearest is None or candidate[:2] < nearest[:2]:
                nearest = candidate
        nearest_prediction = (
            _apply_delta(source, _delta_signature(nearest[2])) if nearest else source.tolist()
        )
        predictions = {
            "identity": source.tolist(),
            "constant_delta": _apply_delta(source, constant_delta),
            "nearest_shown_delta": nearest_prediction,
            "row_table_memorization": table.get(_transition_key(heldout), source.tolist()),
        }
        for control, prediction in predictions.items():
            scored = score_prediction(heldout, prediction)
            scored["control"] = control
            scored["terminal"] = True
            control_rows.append(scored)
    return control_rows


def _mean_boolean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    """Average measured booleans while preserving an empty channel as null."""

    values = [float(bool(row[field])) for row in rows if row.get(field) is not None]
    return float(np.mean(values)) if values else None


def detect_memorization(
    engine_source: str,
    engine_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Report source-table and prefix-collapse signatures without making them gates."""

    heldout_ids = {str(row.get("transition_id")) for row in control_rows}
    prefix = [row for row in engine_rows if str(row.get("transition_id")) not in heldout_ids]
    heldout = [row for row in engine_rows if str(row.get("transition_id")) in heldout_ids]
    prefix_accuracy = _mean_boolean(prefix, "exact_transition_correct")
    heldout_accuracy = _mean_boolean(heldout, "exact_transition_correct")
    table_rows = [row for row in control_rows if row.get("control") == "row_table_memorization"]
    table_by_id = {str(row.get("transition_id")): row for row in table_rows}
    comparisons = [
        bool(row.get("exact_transition_correct"))
        == bool(table_by_id[str(row.get("transition_id"))].get("exact_transition_correct"))
        for row in heldout
        if str(row.get("transition_id")) in table_by_id
    ]
    literal_markers = ("_DELTA_STRS", "_ROW_TABLE", "literal move templates")
    return [
        {
            "signature": "literal_row_or_delta_table",
            "detected": any(marker in engine_source for marker in literal_markers),
            "evidence": [marker for marker in literal_markers if marker in engine_source],
        },
        {
            "signature": "prefix_to_heldout_collapse",
            "detected": bool(
                prefix_accuracy is not None
                and heldout_accuracy is not None
                and prefix_accuracy > heldout_accuracy
            ),
            "prefix_exact_accuracy": prefix_accuracy,
            "heldout_exact_accuracy": heldout_accuracy,
            "gap": (
                None
                if prefix_accuracy is None or heldout_accuracy is None
                else prefix_accuracy - heldout_accuracy
            ),
        },
        {
            "signature": "heldout_behavior_matches_row_table_control",
            "detected": bool(comparisons and all(comparisons)),
            "matching_rows": sum(comparisons),
            "rows_compared": len(comparisons),
        },
    ]


def paired_control_deltas(
    engine_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = RANDOM_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> list[JsonDict]:
    """Pair changing-transition accuracy against the strongest fixed control."""

    changing_engine = {
        str(row.get("transition_id")): row
        for row in engine_rows
        if row.get("changing_transition_correct") is not None
    }
    by_control: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in control_rows:
        if row.get("changing_transition_correct") is None:
            continue
        by_control.setdefault(str(row.get("control")), {})[str(row.get("transition_id"))] = row
    if not changing_engine or not by_control:
        return []
    control_scores = {
        name: float(
            np.mean(
                [
                    bool(rows[row_id].get("changing_transition_correct"))
                    for row_id in changing_engine
                    if row_id in rows
                ]
            )
        )
        for name, rows in by_control.items()
        if all(row_id in rows for row_id in changing_engine)
    }
    if not control_scores:
        return []
    strongest = min(control_scores, key=lambda name: (-control_scores[name], name))
    ids = sorted(changing_engine)
    deltas = np.asarray(
        [
            float(bool(changing_engine[row_id].get("changing_transition_correct")))
            - float(bool(by_control[strongest][row_id].get("changing_transition_correct")))
            for row_id in ids
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    samples = rng.choice(deltas, size=(resamples, len(deltas)), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5]).tolist()
    return [
        {
            "transition_id": row_id,
            "strongest_control": strongest,
            "engine_correct": bool(changing_engine[row_id]["changing_transition_correct"]),
            "control_correct": bool(by_control[strongest][row_id]["changing_transition_correct"]),
            "paired_delta": float(deltas[index]),
            "mean_paired_delta": float(deltas.mean()),
            "interval_low": float(low),
            "interval_high": float(high),
            "interval_level": 0.95,
            "cluster_unit": "transition",
            "bootstrap_resamples": resamples,
            "random_seed": seed,
        }
        for index, row_id in enumerate(ids)
    ]


def _source_hashes(repo_root: Path) -> JsonDict:
    """Hash stable audit dependencies even when a live receipt is absent."""

    paths = {
        "existing_induction_quality_scorer": SCORER_PATH,
        "existing_world_model_scorer": STRUCTURAL_SCORER_PATH,
        "attempt_manifest": ATTEMPTS_PATH / "manifest.jsonl",
        "solve_registry": REGISTRY_PATH,
    }
    return {
        name: {"path": str(path), "sha256": sha256_path(repo_root / path)}
        for name, path in paths.items()
    }


def _scorer_hash(repo_root: Path) -> str | None:
    """Bind the existing scorer and this audit's added per-row reducer together."""

    existing = repo_root / SCORER_PATH
    if not existing.is_file():
        return None
    return sha256_bytes(existing.read_bytes() + b"\0" + Path(__file__).read_bytes())


def _stable_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash scientific content while excluding timing and operating-system process IDs."""

    value = json.loads(json.dumps(artifact))
    value.pop("duration_s", None)
    value.pop("reproducibility_checksum", None)
    for row in value.get("engine_execution_rows", []):
        if isinstance(row, dict):
            row.pop("duration_s", None)
            row.pop("parent_pid", None)
            row.pop("worker_pid", None)
    return sha256_bytes(canonical_json(value))


def _empty_artifact(repo_root: Path) -> JsonDict:
    """Create the full schema before any external precondition can fail."""

    engine_path, engine_rows = resolve_engine(repo_root)
    engine_hash = sha256_path(engine_path) if engine_path else None
    return {
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": 0.0,
        "source_artifact_hashes": _source_hashes(repo_root),
        "selected_run_provenance": {},
        "engine_hash": engine_hash,
        "prompt_hash": None,
        "transition_hash": None,
        "scorer_hash": _scorer_hash(repo_root),
        "rows": [],
        "per_transition_rows": [],
        "split_rows": [],
        "purity_rows": [],
        "engine_execution_rows": [],
        "control_rows": [],
        "prefix_exact_accuracy": None,
        "heldout_exact_accuracy": None,
        "heldout_changing_accuracy": None,
        "heldout_noop_accuracy": None,
        "generalization_gap": None,
        "paired_control_delta_rows": [],
        "memorization_signature_rows": [],
        "arc_induction_audit_complete_score": 0,
        "arc_induction_generalization_positive_score": 0,
        "solve_claimed": False,
        "level_claimed": False,
        "registry_updated": False,
        "submitted_to_leaderboard": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": None,
        "gate_check_summary": [],
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_arc_post_refit_induction_audit",
        "_engine_resolution_rows": engine_rows,
    }


def _finish_blocked(artifact: JsonDict, checks: list[JsonDict], started: float) -> JsonDict:
    """Finish a blocked artifact without leaving private construction fields."""

    artifact["preconditions_checked"] = checks
    artifact["gate_check_summary"] = _failed_gates(checks)
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact.pop("_engine_resolution_rows", None)
    artifact["reproducibility_checksum"] = _stable_checksum(artifact)
    return artifact


def build_artifact(
    *,
    date: str,
    repo_root: Path = REPO_ROOT,
    target_engine_sha16: str = TARGET_ENGINE_SHA16,
) -> JsonDict:
    """Build a positive, null, or blocked audit from frozen local evidence."""

    del date  # The execution date is fixed by the command and does not select evidence.
    started = time.monotonic()
    artifact = _empty_artifact(repo_root)
    candidates = discover_run_candidates(repo_root, target_engine_sha16=target_engine_sha16)
    artifact["rows"] = candidates
    selected, selection_reason = select_run(candidates)
    engine_path, engine_resolution = resolve_engine(repo_root, target_engine_sha16)
    artifact["engine_hash"] = sha256_path(engine_path) if engine_path else None
    artifact["selected_run_provenance"] = {
        "selection_reason": selection_reason,
        "target_engine_sha256_16": target_engine_sha16,
        "target_n_ctx": TARGET_N_CTX,
        "candidate_count": len(candidates),
        "selected": selected,
        "engine_resolution_rows": engine_resolution,
    }
    if selected is not None:
        artifact["source_artifact_hashes"]["selected_run"] = {
            "path": selected["path"],
            "sha256": selected.get("run_sha256"),
        }
    if engine_path is not None:
        artifact["source_artifact_hashes"]["engine"] = {
            "path": str(engine_path.relative_to(repo_root)),
            "sha256": artifact["engine_hash"],
        }
    checks = [
        _gate("existing_scorer", True, (repo_root / SCORER_PATH).is_file()),
        _gate(
            "completed_target_run_count",
            1,
            sum(
                row.get("complete") is True
                and row.get("n_ctx") == TARGET_N_CTX
                and row.get("policy") == "e3"
                for row in candidates
            ),
        ),
        _gate(
            "one_content_matched_engine",
            1,
            sum(row["name_hash_matches_bytes"] for row in engine_resolution),
        ),
        _gate("unambiguous_engine_path", True, engine_path is not None),
        _gate(
            "selection_timestamps_recorded",
            True,
            bool(
                selected and selected.get("engine_emitted_at") and selected.get("run_file_mtime_ns")
            ),
        ),
    ]
    if selected is None or engine_path is None or _failed_gates(checks):
        checks.append(_gate("immutable_transition_source", True, False))
        return _finish_blocked(artifact, checks, started)

    transition_path = _safe_relative_source(repo_root, selected.get("transition_source_path"))
    source = _read_json_object(transition_path) if transition_path else None
    checks.append(_gate("immutable_transition_source", True, source is not None))
    if source is None or transition_path is None:
        return _finish_blocked(artifact, checks, started)

    transition_hash = sha256_path(transition_path)
    prompt_hash = source.get("prompt_sha256")
    artifact["prompt_hash"] = prompt_hash
    artifact["transition_hash"] = transition_hash
    artifact["source_artifact_hashes"].update(
        {
            "transition_source": {
                "path": str(transition_path.relative_to(repo_root)),
                "sha256": transition_hash,
            },
        }
    )
    checks.extend(
        [
            _gate(
                "transition_source_engine_hash",
                artifact["engine_hash"],
                source.get("attempt_engine_sha256"),
            ),
            _gate("run_prompt_hash_matches_source", selected.get("prompt_sha256"), prompt_hash),
            _gate(
                "source_transition_rows_present",
                ">=1",
                len(source.get("rows", [])),
                passed=bool(source.get("rows")),
            ),
        ]
    )
    source_rows = source.get("rows", [])
    if _failed_gates(checks) or not isinstance(source_rows, list):
        return _finish_blocked(artifact, checks, started)

    try:
        for row in source_rows:
            _array(row.get("grid"))
            _array(row.get("next_grid"))
    except (AttributeError, TypeError, ValueError) as error:
        checks.append(
            _gate("source_transition_schema", "valid_2d_grids", f"{type(error).__name__}: {error}")
        )
        return _finish_blocked(artifact, checks, started)

    split, purity, split_failures = rebuild_split(
        source_rows,
        prompt_row_ids=source.get("prompt_row_ids", []),
        repair_feedback_row_ids=source.get("repair_feedback_row_ids", []),
    )
    artifact["split_rows"] = [split]
    artifact["purity_rows"] = purity
    if split_failures:
        checks.append(_gate("split_purity", [], split_failures))
        return _finish_blocked(artifact, checks, started)
    checks.append(_gate("split_purity", True, True))

    shown_ids = set(split["shown_row_ids"])
    normalized_rows = []
    for source_row in sorted(source_rows, key=lambda row: int(row.get("index", -1))):
        row = dict(source_row)
        row["split"] = "shown_prefix" if str(row.get("transition_id")) in shown_ids else "heldout"
        row["row_sha256"] = sha256_bytes(canonical_json(source_row))
        normalized_rows.append(row)
    artifact["per_transition_rows"] = normalized_rows
    engine_rows, execution_rows = execute_engine_fresh(engine_path, normalized_rows)
    for score, source_row in zip(engine_rows, normalized_rows, strict=True):
        score["split"] = source_row["split"]
    execution_receipt = execution_rows[0]
    artifact["engine_execution_rows"] = [
        {
            **row,
            "fresh_process": execution_receipt["fresh_process"],
            "restricted_process": execution_receipt["restricted_process"],
            "write_guard_enabled": execution_receipt["write_guard_enabled"],
            "network_guard_enabled": execution_receipt["network_guard_enabled"],
            "parent_pid": execution_receipt["parent_pid"],
            "worker_pid": execution_receipt["worker_pid"],
        }
        for row in engine_rows
    ]
    shown = [row for row in normalized_rows if row["split"] == "shown_prefix"]
    heldout = [row for row in normalized_rows if row["split"] == "heldout"]
    heldout_ids = {str(row["transition_id"]) for row in heldout}
    artifact["control_rows"] = score_controls(shown, heldout)
    artifact["prefix_exact_accuracy"] = _mean_boolean(
        [row for row in engine_rows if row["split"] == "shown_prefix"],
        "exact_transition_correct",
    )
    artifact["heldout_exact_accuracy"] = _mean_boolean(
        [row for row in engine_rows if str(row["transition_id"]) in heldout_ids],
        "exact_transition_correct",
    )
    artifact["heldout_changing_accuracy"] = _mean_boolean(
        [row for row in engine_rows if str(row["transition_id"]) in heldout_ids],
        "changing_transition_correct",
    )
    artifact["heldout_noop_accuracy"] = _mean_boolean(
        [row for row in engine_rows if str(row["transition_id"]) in heldout_ids],
        "noop_correct",
    )
    if (
        artifact["prefix_exact_accuracy"] is not None
        and artifact["heldout_exact_accuracy"] is not None
    ):
        artifact["generalization_gap"] = (
            artifact["prefix_exact_accuracy"] - artifact["heldout_exact_accuracy"]
        )
    artifact["paired_control_delta_rows"] = paired_control_deltas(
        [row for row in engine_rows if str(row["transition_id"]) in heldout_ids],
        artifact["control_rows"],
    )
    artifact["memorization_signature_rows"] = detect_memorization(
        engine_path.read_text(encoding="utf-8"), engine_rows, artifact["control_rows"]
    )
    terminal = (
        len(engine_rows) == len(normalized_rows)
        and all(row.get("terminal") is True for row in engine_rows)
        and len(artifact["control_rows"]) == 4 * len(heldout)
        and all(row.get("terminal") is True for row in artifact["control_rows"])
        and all(row.get("terminal") is True for row in execution_rows)
    )
    artifact["arc_induction_audit_complete_score"] = int(terminal)
    paired = artifact["paired_control_delta_rows"]
    positive = bool(
        terminal
        and paired
        and all(row.get("pure") is True for row in purity)
        and artifact["heldout_changing_accuracy"] is not None
        and paired[0]["mean_paired_delta"] > 0
        and paired[0]["interval_low"] > 0
    )
    artifact["arc_induction_generalization_positive_score"] = int(positive)
    artifact["verdict_class"] = "positive" if positive else "null"
    artifact["honest_verdict"] = (
        "positive_arc_post_refit_engine_generalizes_beyond_controls"
        if positive
        else "null_arc_post_refit_engine_does_not_clear_generalization_gate"
    )
    artifact["preconditions_checked"] = checks
    artifact["gate_check_summary"] = []
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact.pop("_engine_resolution_rows", None)
    artifact["reproducibility_checksum"] = _stable_checksum(artifact)
    return artifact


def write_json_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    """Replace the artifact atomically so readers never see a partial JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the read-only audit and write its terminal artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260904")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args(argv)
    output = Path(args.output)
    if not output.is_absolute():
        output = REPO_ROOT / output
    artifact = build_artifact(date=args.date, repo_root=REPO_ROOT)
    write_json_atomic(output, artifact)
    print(f"wrote {output}")
    print(artifact["honest_verdict"])
    return 0 if artifact["verdict_class"] != "blocked" else 1


if __name__ == "__main__":  # pragma: no cover - exercised as a fresh worker or thin CLI.
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        raise SystemExit(_worker_main(sys.argv[1:]))
    raise SystemExit(main())
