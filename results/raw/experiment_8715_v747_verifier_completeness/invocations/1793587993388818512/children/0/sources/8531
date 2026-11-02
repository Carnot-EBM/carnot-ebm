"""Exp 4093 OFF-ARC demo-fit precision replay.

Spec refs: REQ-VERIFY-4093, SCENARIO-VERIFY-4093.

This is an offline replay over cached code candidates. It measures the
candidate-level precision question directly:

    P(hidden-pass | visible-pass)

and then re-measures that precision after a cheap public-derived input-mutation
agreement filter. No Codex or local GGUF calls are made by this script.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ARTIFACT = REPO_ROOT / "results" / "experiment_4068_offarc_transfer_power_sync.json"
SOURCE_CHECKPOINT = REPO_ROOT / "results" / "offarc_power_sync_gemma12b_evalplus_k5.checkpoint.json"
LEGACY_CHECKPOINT = REPO_ROOT / "results" / "experiment_4045_offarc_transfer_power.checkpoint.json"
OUTPUT = REPO_ROOT / "results" / "experiment_4093_offarc_demofit_precision_transfer.json"

RANDOM_SEED = 4093
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
DOMAIN_GENERAL_PRECISION_FLOOR = 0.68
EXEC_TIMEOUT_S = 0.25

REQUIRED_ARTIFACT_FIELDS = [
    "honest_verdict",
    "demofit_precision_raw",
    "demofit_precision_filtered",
    "filter_recall",
    "primitive_is_domain_general",
    "n_tasks_scored",
    "n_codex_calls",
    "random_seed",
    "reproducibility_checksum",
    "inference_substrate",
]

FIELD_PRINCIPLES = {
    "demofit_precision_raw": (
        "Bare float: P(hidden-pass | visible-pass) over cached code candidates."
    ),
    "demofit_precision_filtered": (
        "Bare float: P(hidden-pass | visible-pass and public-derived mutation agreement)."
    ),
    "filter_recall": "Bare float: retained visible-pass candidates divided by all visible-pass candidates.",
    "primitive_is_domain_general": (
        "Bare bool: true only when off-ARC headroom exists and raw demo-fit precision clears "
        "the ARC-reference precision floor."
    ),
    "n_codex_calls": "Must remain zero: Exp 4093 is offline replay over cached candidates.",
    "inference_substrate": "Must be verifier_ensemble_against_cached_candidates.",
}


@dataclass(frozen=True)
class MutationProbe:
    func_name: str
    args: tuple[Any, ...]
    expected: Any


Executor = Callable[[str, str, tuple[Any, ...], float], tuple[Any, Exception | None]]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cached_candidate_pool(
    candidate_artifact_path: Path = SOURCE_ARTIFACT,
    candidate_checkpoint_path: Path = SOURCE_CHECKPOINT,
    legacy_checkpoint_path: Path = LEGACY_CHECKPOINT,
) -> tuple[dict[str, list[dict[str, Any]]], Path | None]:
    """Load the largest cached pool available without generating candidates."""
    for path in (candidate_artifact_path, candidate_checkpoint_path, legacy_checkpoint_path):
        if not path.exists():
            continue
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        pool = _pool_from_payload(payload)
        if pool:
            return pool, path
    return {}, None


def _pool_from_payload(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    candidate_pool = payload.get("candidate_pool")
    if isinstance(candidate_pool, dict):
        return _normalize_pool(candidate_pool)
    evaluations = payload.get("evaluations_by_task")
    if isinstance(evaluations, dict):
        return _normalize_pool(evaluations)
    return {}


def _normalize_pool(pool: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    normalized: dict[str, list[dict[str, Any]]] = {}
    for task_id, rows in pool.items():
        if not isinstance(rows, list):
            continue
        task_rows = [row for row in rows if isinstance(row, dict)]
        if task_rows:
            normalized[str(task_id)] = task_rows
    return normalized


def sandbox_importable() -> bool:  # pragma: no cover - exercised through run injection.
    try:
        from carnot.verify import sandbox  # noqa: F401

        return True
    except Exception:
        return False


def local_exec_with_timeout(
    code: str,
    func_name: str,
    args: tuple[Any, ...],
    timeout: float,
) -> tuple[Any, Exception | None]:  # pragma: no cover - integration execution path.
    """Execute cached candidate code with a real wall-clock timeout."""

    def _timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"execution timed out after {timeout}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)
    namespace: dict[str, Any] = {}
    try:
        signal.signal(signal.SIGALRM, _timeout)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, timeout))
        exec(code, namespace)  # noqa: S102 - intentional cached generated-code replay.
        func = namespace.get(func_name)
        if func is None:
            return None, NameError(f"Function '{func_name}' not found in code")
        return func(*copy.deepcopy(args)), None
    except Exception as exc:
        return None, exc
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])


def load_default_mutation_probes(  # pragma: no cover - exercised by required script run.
    task_limit: int = 160,
) -> dict[str, list[MutationProbe]]:
    import offarc_power_evalplus_run as evalplus_runner

    tasks, _skipped = evalplus_runner.load_code_tasks(limit=task_limit)
    raw_rows = _evalplus_raw_rows_by_task()
    return build_mutation_probes(tasks, raw_rows)


def _evalplus_raw_rows_by_task() -> dict[str, tuple[dict[str, Any], str]]:  # pragma: no cover
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    rows: dict[str, tuple[dict[str, Any], str]] = {}
    for task_id, row in get_human_eval_plus().items():
        rows[str(task_id)] = (dict(row), "evalplus_humaneval")
    for task_id, row in get_mbpp_plus().items():
        rows[str(task_id)] = (dict(row), "evalplus_mbpp")
    return rows


def build_mutation_probes(
    tasks: list[Any],
    raw_rows_by_task: dict[str, tuple[dict[str, Any], str]],
    *,
    max_probes_per_task: int = 4,
) -> dict[str, list[MutationProbe]]:
    import offarc_power_evalplus_run as evalplus_runner

    probes: dict[str, list[MutationProbe]] = {}
    for task in tasks:
        raw = _raw_row_for_task(str(task.task_id), raw_rows_by_task)
        if raw is None:
            probes[str(task.task_id)] = []
            continue
        raw_row, corpus = raw
        canonical_code = evalplus_runner._canonical_code(raw_row, corpus=corpus)
        task_probes: list[MutationProbe] = []
        for visible in list(task.visible_tests):
            mutated_args = tuple(
                evalplus_runner.base._mutate_public_arg(arg) for arg in visible.args
            )
            if mutated_args == tuple(visible.args):
                continue
            ok, expected = evalplus_runner._canonical_expected(
                canonical_code,
                visible.func_name,
                mutated_args,
            )
            if ok:
                task_probes.append(
                    MutationProbe(
                        func_name=visible.func_name,
                        args=mutated_args,
                        expected=expected,
                    )
                )
            if len(task_probes) >= max_probes_per_task:
                break
        probes[str(task.task_id)] = task_probes
    return probes


def _raw_row_for_task(
    task_id: str,
    raw_rows_by_task: dict[str, tuple[dict[str, Any], str]],
) -> tuple[dict[str, Any], str] | None:
    for alias in _task_aliases(task_id):
        if alias in raw_rows_by_task:
            return raw_rows_by_task[alias]
    return None


def _task_aliases(task_id: str) -> list[str]:
    aliases = [task_id]
    if task_id.startswith("Mbpp/"):
        aliases.append("mbpp-" + task_id.split("/", 1)[1])
    if task_id.startswith("mbpp-"):
        aliases.append("Mbpp/" + task_id.split("-", 1)[1])
    return aliases


def score_candidate_pool(
    candidate_pool: dict[str, list[dict[str, Any]]],
    *,
    mutation_probes_by_task: dict[str, list[MutationProbe]],
    executor: Executor = local_exec_with_timeout,
    timeout: float = EXEC_TIMEOUT_S,
) -> dict[str, Any]:
    raw_visible = 0
    raw_hidden = 0
    filtered_visible = 0
    filtered_hidden = 0
    hidden_fail_headroom = 0
    n_tasks_scored = 0
    n_mutation_probe_tasks = 0
    per_task: list[dict[str, Any]] = []

    for task_id in sorted(candidate_pool):
        rows = candidate_pool[task_id]
        scoreable = [
            row
            for row in rows
            if isinstance(row.get("visible_passes"), list)
            and isinstance(row.get("hidden_passes"), list)
        ]
        if not scoreable:
            continue
        n_tasks_scored += 1
        probes = mutation_probes_by_task.get(task_id, [])
        if probes:
            n_mutation_probe_tasks += 1
        task_raw_visible = 0
        task_raw_hidden = 0
        task_filtered_visible = 0
        task_filtered_hidden = 0

        for row in scoreable:
            if not _all_passed(row.get("visible_passes")):
                continue
            hidden_pass = _all_passed(row.get("hidden_passes"))
            raw_visible += 1
            task_raw_visible += 1
            if hidden_pass:
                raw_hidden += 1
                task_raw_hidden += 1
            else:
                hidden_fail_headroom += 1
            if _mutation_agrees(row, probes, executor=executor, timeout=timeout):
                filtered_visible += 1
                task_filtered_visible += 1
                if hidden_pass:
                    filtered_hidden += 1
                    task_filtered_hidden += 1

        per_task.append(
            {
                "task_id": task_id,
                "n_candidates": len(rows),
                "n_visible_pass_candidates": task_raw_visible,
                "n_visible_hidden_pass_candidates": task_raw_hidden,
                "n_filtered_candidates": task_filtered_visible,
                "n_filtered_hidden_pass_candidates": task_filtered_hidden,
                "n_mutation_probes": len(probes),
            }
        )

    raw_precision = _rate(raw_hidden, raw_visible)
    filtered_precision = _rate(filtered_hidden, filtered_visible)
    filter_recall = _rate(filtered_visible, raw_visible)
    return {
        "demofit_precision_raw": raw_precision,
        "demofit_precision_filtered": filtered_precision,
        "filter_recall": filter_recall,
        "filter_raises_precision": filtered_visible > 0
        and filtered_precision > raw_precision + 1e-12,
        "n_tasks_scored": n_tasks_scored,
        "n_mutation_probe_tasks": n_mutation_probe_tasks,
        "n_candidates": sum(len(rows) for rows in candidate_pool.values()),
        "n_visible_pass_candidates": raw_visible,
        "n_visible_hidden_pass_candidates": raw_hidden,
        "n_visible_hidden_fail_candidates": hidden_fail_headroom,
        "n_filtered_candidates": filtered_visible,
        "n_filtered_hidden_pass_candidates": filtered_hidden,
        "headroom_present": hidden_fail_headroom > 0,
        "per_task": per_task,
    }


def _mutation_agrees(
    row: dict[str, Any],
    probes: list[MutationProbe],
    *,
    executor: Executor,
    timeout: float,
) -> bool:
    if not probes:
        return False
    code = str(row.get("code") or "")
    if not code:
        return False
    for probe in probes:
        result, error = executor(code, probe.func_name, probe.args, timeout)
        if error is not None or result != probe.expected:
            return False
    return True


def _all_passed(values: Any) -> bool:
    return isinstance(values, list) and bool(values) and all(bool(value) for value in values)


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def build_blocked_artifact(
    honest_verdict: str,
    *,
    preconditions_checked: list[dict[str, Any]],
    duration_s: float,
    candidate_pool_source: Path | None = None,
) -> dict[str, Any]:
    return build_artifact(
        metrics={
            "demofit_precision_raw": 0.0,
            "demofit_precision_filtered": 0.0,
            "filter_recall": 0.0,
            "filter_raises_precision": False,
            "n_tasks_scored": 0,
            "n_mutation_probe_tasks": 0,
            "n_candidates": 0,
            "n_visible_pass_candidates": 0,
            "n_visible_hidden_pass_candidates": 0,
            "n_visible_hidden_fail_candidates": 0,
            "n_filtered_candidates": 0,
            "n_filtered_hidden_pass_candidates": 0,
            "headroom_present": False,
            "per_task": [],
        },
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        candidate_pool_source=candidate_pool_source,
        honest_verdict=honest_verdict,
    )


def build_artifact(
    *,
    metrics: dict[str, Any],
    preconditions_checked: list[dict[str, Any]],
    duration_s: float,
    candidate_pool_source: Path | None,
    honest_verdict: str | None = None,
) -> dict[str, Any]:
    raw = float(metrics["demofit_precision_raw"])
    filtered = float(metrics["demofit_precision_filtered"])
    headroom_present = bool(metrics["headroom_present"])
    if honest_verdict is None:
        honest_verdict = _verdict(metrics)
    artifact: dict[str, Any] = {
        "experiment": "experiment_4093_offarc_demofit_precision_transfer",
        "schema": "carnot.experiment_4093_offarc_demofit_precision_transfer.v1",
        "honest_verdict": honest_verdict,
        "demofit_precision_raw": raw,
        "demofit_precision_filtered": filtered,
        "filter_recall": float(metrics["filter_recall"]),
        "filter_raises_precision": bool(metrics["filter_raises_precision"]),
        "primitive_is_domain_general": bool(
            headroom_present and raw >= DOMAIN_GENERAL_PRECISION_FLOOR
        ),
        "n_tasks_scored": int(metrics["n_tasks_scored"]),
        "n_codex_calls": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 2),
        "candidate_pool_source": str(candidate_pool_source) if candidate_pool_source else None,
        "source_candidate_artifact": str(SOURCE_ARTIFACT),
        "source_candidate_checkpoint": str(SOURCE_CHECKPOINT),
        "legacy_candidate_checkpoint": str(LEGACY_CHECKPOINT),
        "preconditions_checked": preconditions_checked,
        "filter_definition": (
            "visible-pass plus deterministic public-derived input-mutation probe agreement; "
            "hidden labels are used only after filtering for scoring"
        ),
        "domain_general_precision_floor": DOMAIN_GENERAL_PRECISION_FLOOR,
        "n_candidates": int(metrics["n_candidates"]),
        "n_visible_pass_candidates": int(metrics["n_visible_pass_candidates"]),
        "n_visible_hidden_pass_candidates": int(metrics["n_visible_hidden_pass_candidates"]),
        "n_visible_hidden_fail_candidates": int(metrics["n_visible_hidden_fail_candidates"]),
        "n_filtered_candidates": int(metrics["n_filtered_candidates"]),
        "n_filtered_hidden_pass_candidates": int(metrics["n_filtered_hidden_pass_candidates"]),
        "n_mutation_probe_tasks": int(metrics["n_mutation_probe_tasks"]),
        "headroom_present": headroom_present,
        "per_task": metrics["per_task"],
        "field_principles": FIELD_PRINCIPLES,
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def _verdict(metrics: dict[str, Any]) -> str:
    raw = float(metrics["demofit_precision_raw"])
    filtered = float(metrics["demofit_precision_filtered"])
    if int(metrics["n_visible_pass_candidates"]) == 0:
        return "complete: offarc_no_headroom_no_visible_pass_candidates"
    if not bool(metrics["headroom_present"]):
        return "complete: offarc_no_headroom_visible_pass_equals_hidden_pass"
    if bool(metrics["filter_raises_precision"]):
        return f"complete: offarc_demofit_precision_{raw:.2f}_filter_raises_to_{filtered:.2f}"
    return f"complete: offarc_demofit_precision_{raw:.2f}_filter_no_raise_{filtered:.2f}"


def _artifact_checksum(artifact: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"reproducibility_checksum", "duration_s", "field_principles"}
    }
    encoded = json.dumps(payload, sort_keys=True, default=repr)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_artifact(artifact: dict[str, Any]) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:") or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must use a terminal prefix")
    for field in ("demofit_precision_raw", "demofit_precision_filtered", "filter_recall"):
        if not isinstance(artifact[field], float) or isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare float")
    if not isinstance(artifact["primitive_is_domain_general"], bool):
        raise ValueError("primitive_is_domain_general must be a bare bool")
    for field in ("n_tasks_scored", "n_codex_calls", "random_seed"):
        if not isinstance(artifact[field], int) or isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare int")
    if artifact["n_codex_calls"] != 0:
        raise ValueError("n_codex_calls must remain zero for offline replay")
    if (
        not isinstance(artifact["reproducibility_checksum"], str)
        or not artifact["reproducibility_checksum"]
    ):
        raise ValueError("reproducibility_checksum must be non-empty")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be cached candidates")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(
    *,
    candidate_artifact_path: Path = SOURCE_ARTIFACT,
    candidate_checkpoint_path: Path = SOURCE_CHECKPOINT,
    legacy_checkpoint_path: Path = LEGACY_CHECKPOINT,
    output_path: Path = OUTPUT,
    sandbox_importer: Callable[[], bool] = sandbox_importable,
    task_probe_builder: Callable[[], dict[str, list[MutationProbe]]] = load_default_mutation_probes,
    executor: Executor = local_exec_with_timeout,
) -> dict[str, Any]:
    started = time.time()
    preconditions: list[dict[str, Any]] = []
    pool, source = load_cached_candidate_pool(
        candidate_artifact_path,
        candidate_checkpoint_path,
        legacy_checkpoint_path,
    )
    preconditions.append(
        {
            "resource": "candidate_pool_cache",
            "available": bool(pool),
            "source": str(source) if source else None,
        }
    )
    sandbox_ok = sandbox_importer()
    preconditions.append({"resource": "sandbox_importable", "available": sandbox_ok})
    if not pool:
        artifact = build_blocked_artifact(
            "blocked_cached_candidate_pool_missing",
            preconditions_checked=preconditions,
            duration_s=time.time() - started,
            candidate_pool_source=source,
        )
        validate_artifact(artifact)
        _write_json(output_path, artifact)
        return artifact
    if not sandbox_ok:
        artifact = build_blocked_artifact(
            "blocked_sandbox_unavailable",
            preconditions_checked=preconditions,
            duration_s=time.time() - started,
            candidate_pool_source=source,
        )
        validate_artifact(artifact)
        _write_json(output_path, artifact)
        return artifact

    try:
        mutation_probes = task_probe_builder()
        preconditions.append(
            {
                "resource": "public_mutation_probes",
                "available": True,
                "n_tasks": len(mutation_probes),
            }
        )
    except Exception as exc:
        preconditions.append(
            {
                "resource": "public_mutation_probes",
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        artifact = build_blocked_artifact(
            "blocked_public_mutation_probes_unavailable",
            preconditions_checked=preconditions,
            duration_s=time.time() - started,
            candidate_pool_source=source,
        )
        validate_artifact(artifact)
        _write_json(output_path, artifact)
        return artifact

    metrics = score_candidate_pool(
        pool,
        mutation_probes_by_task=mutation_probes,
        executor=executor,
    )
    artifact = build_artifact(
        metrics=metrics,
        preconditions_checked=preconditions,
        duration_s=time.time() - started,
        candidate_pool_source=source,
    )
    validate_artifact(artifact)
    _write_json(output_path, artifact)
    return artifact


def main() -> None:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--candidate-artifact", type=Path, default=SOURCE_ARTIFACT)
    parser.add_argument("--candidate-checkpoint", type=Path, default=SOURCE_CHECKPOINT)
    parser.add_argument("--legacy-checkpoint", type=Path, default=LEGACY_CHECKPOINT)
    args = parser.parse_args()
    artifact = run(
        candidate_artifact_path=args.candidate_artifact,
        candidate_checkpoint_path=args.candidate_checkpoint,
        legacy_checkpoint_path=args.legacy_checkpoint,
        output_path=args.output,
    )
    print(f"-> {artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    main()
