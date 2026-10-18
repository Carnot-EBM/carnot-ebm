"""Exp 2920 OpenComputer-style local state verifier harness.

Spec: REQ-VERIFY-2920, SCENARIO-VERIFY-2920.

This module models agentic verification as state inspection rather than answer
judging.  Each synthetic task has a small local state surface, and each
verifier reads that surface directly to assign partial credit.  That makes the
failure evidence auditable: a future agent benchmark can point to the exact
file, JSON field, SQLite row, or filesystem path that caused the rejection.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

RUN_DATE = "20260523"
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILENAME = "experiment_2920_opencomputer_style_state_verifier_harness_v1.json"
MANIFEST_FILENAME = "opencomputer_state_verifier_manifest_2920.json"
VERIFIER_SOURCE_PATH = "python/carnot/eval/opencomputer_state_verifier_harness.py"
VERIFIER_ENTRYPOINT = "carnot.eval.opencomputer_state_verifier_harness.verify_task_state"
INFERENCE_SUBSTRATE = "deterministic_state_verifier"
PARTIAL_CREDIT_FIELDS = (
    "passed",
    "score",
    "earned_points",
    "max_points",
    "checks",
    "violations",
    "failure_localization",
)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "state_verifier_harness_ready",
    "task_manifest_path",
    "n_state_tasks",
    "verifier_source_paths",
    "golden_state_pass_rate",
    "negative_state_reject_rate",
    "partial_credit_fields",
    "failure_localization_examples",
    "llm_judge_used",
    "inference_substrate",
    "duration_s",
    "run_date",
)
FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix verdict tells the conductor whether this is a complete local harness artifact.",
    "state_verifier_harness_ready": "True only when all golden fixtures pass and all negative fixtures are rejected.",
    "task_manifest_path": "Future agent runners need a stable manifest path instead of hard-coded Python fixtures.",
    "n_state_tasks": "The task count keeps the harness scope auditable and prevents benchmark-size overclaiming.",
    "verifier_source_paths": "Auditors can inspect the deterministic verifier implementation that produced the scores.",
    "golden_state_pass_rate": "Golden fixtures confirm the verifier accepts known-good state.",
    "negative_state_reject_rate": "Negative fixtures confirm the verifier rejects localized state defects.",
    "partial_credit_fields": "Partial credit exposes what passed and failed instead of compressing evidence into one bit.",
    "failure_localization_examples": "Localized examples show where a future agent should repair the state.",
    "llm_judge_used": "Must remain false because this harness verifies app state, not prose quality.",
    "inference_substrate": "Declares that no LLM inference occurred; outputs come from deterministic state checks.",
    "duration_s": "Measured wall-clock runtime with no padding.",
    "run_date": "Pins the artifact to the conductor run date.",
}


@dataclass(frozen=True)
class CheckSpec:
    """One check in a deterministic state verifier.

    The check carries its own point value and localization target so the
    verifier can explain partial credit without guessing after the fact.
    """

    check_id: str
    description: str
    points: float
    localized_to: str


@dataclass(frozen=True)
class StateTask:
    """One tiny software-world task with observable local state."""

    task_id: str
    task_type: str
    instruction: str
    observable_kind: str
    observable_paths: tuple[str, ...]
    checks: tuple[CheckSpec, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for writing the Exp 2920 manifest and artifact."""

    output_path: Path | None = None
    manifest_path: Path | None = None
    tests_run: Sequence[str] = ()
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or REPO_ROOT / "results" / OUTPUT_FILENAME

    def task_manifest_path(self) -> Path:
        return self.manifest_path or REPO_ROOT / "results" / MANIFEST_FILENAME


def build_state_tasks() -> list[StateTask]:
    """Return the fixed four-task state-verifier harness."""

    return [
        StateTask(
            task_id="state-json-config-001",
            task_type="json_file_transform",
            instruction="Enable search in config/app.json, cap max_items at 25, and preserve service_id.",
            observable_kind="json_file",
            observable_paths=("config/app.json",),
            checks=(
                CheckSpec(
                    "search_enabled",
                    "features.search.enabled is true",
                    1.0,
                    "config/app.json:features.search.enabled",
                ),
                CheckSpec(
                    "max_items_25",
                    "limits.max_items equals 25",
                    1.0,
                    "config/app.json:limits.max_items",
                ),
                CheckSpec(
                    "service_id_preserved",
                    "service_id remains demo-search",
                    1.0,
                    "config/app.json:service_id",
                ),
            ),
        ),
        StateTask(
            task_id="state-sqlite-tasks-001",
            task_type="sqlite_row_edit",
            instruction="Mark the ship-harness task done, leave write-docs open, and keep an audit note.",
            observable_kind="sqlite_database",
            observable_paths=("data/tasks.sqlite",),
            checks=(
                CheckSpec(
                    "ship_harness_done",
                    "ship-harness row status is done",
                    1.0,
                    "data/tasks.sqlite:tasks[ship-harness].status",
                ),
                CheckSpec(
                    "write_docs_open",
                    "write-docs row status is todo",
                    1.0,
                    "data/tasks.sqlite:tasks[write-docs].status",
                ),
                CheckSpec(
                    "audit_note_present",
                    "ship-harness row records deterministic-verifier audit note",
                    1.0,
                    "data/tasks.sqlite:tasks[ship-harness].audit_note",
                ),
            ),
        ),
        StateTask(
            task_id="state-filesystem-cleanup-001",
            task_type="filesystem_state",
            instruction="Write the summary, preserve the archive marker, and remove the temporary cache file.",
            observable_kind="filesystem_tree",
            observable_paths=(
                "workspace/summary.txt",
                "workspace/archive/keep.txt",
                "workspace/tmp/cache.bin",
            ),
            checks=(
                CheckSpec(
                    "summary_written",
                    "workspace/summary.txt has the expected text",
                    1.0,
                    "workspace/summary.txt",
                ),
                CheckSpec(
                    "archive_preserved",
                    "workspace/archive/keep.txt still exists",
                    1.0,
                    "workspace/archive/keep.txt",
                ),
                CheckSpec(
                    "cache_removed",
                    "workspace/tmp/cache.bin is absent",
                    1.0,
                    "workspace/tmp/cache.bin",
                ),
            ),
        ),
        StateTask(
            task_id="state-jsonl-inventory-001",
            task_type="jsonl_inventory",
            instruction="Normalize inventory/items.jsonl so widget counts are exact and records stay sorted by sku.",
            observable_kind="jsonl_file",
            observable_paths=("inventory/items.jsonl",),
            checks=(
                CheckSpec(
                    "widget_a_count",
                    "widget-a count is 3",
                    1.0,
                    "inventory/items.jsonl:sku=widget-a",
                ),
                CheckSpec(
                    "widget_b_count",
                    "widget-b count is 7",
                    1.0,
                    "inventory/items.jsonl:sku=widget-b",
                ),
                CheckSpec(
                    "sorted_by_sku", "records are sorted by sku", 1.0, "inventory/items.jsonl"
                ),
            ),
        ),
    ]


def write_task_manifest(tasks: Sequence[StateTask], path: Path | str) -> JsonDict:
    """Write a future-consumable manifest for local state-verifier tasks."""

    payload: JsonDict = {
        "schema": "carnot.opencomputer_state_verifier_manifest.v1",
        "run_date": RUN_DATE,
        "n_state_tasks": len(tasks),
        "partial_credit_fields": list(PARTIAL_CREDIT_FIELDS),
        "tasks": [_task_to_manifest_row(task) for task in tasks],
    }
    _write_json(Path(path), payload)
    return payload


def materialize_task_state(task: StateTask, variant: str, workspace_root: Path | str) -> Path:
    """Create the golden or negative observable state for one task."""

    if variant not in {"golden", "negative"}:
        raise ValueError(f"unknown state variant: {variant}")
    state_root = Path(workspace_root) / task.task_id
    if state_root.exists():
        shutil.rmtree(state_root)
    state_root.mkdir(parents=True)
    {
        "json_file_transform": _materialize_json_file_transform,
        "sqlite_row_edit": _materialize_sqlite_row_edit,
        "filesystem_state": _materialize_filesystem_state,
        "jsonl_inventory": _materialize_jsonl_inventory,
    }[task.task_type](state_root, variant)
    return state_root


def verify_task_state(task: StateTask, state_root: Path | str) -> JsonDict:
    """Verify one task by inspecting its local state directly."""

    root = Path(state_root)
    if not root.exists():
        max_points = float(sum(check.points for check in task.checks))
        violation = {
            "check_id": "state_root_exists",
            "localized_to": str(root),
            "detail": "state root does not exist",
        }
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "passed": False,
            "score": 0.0,
            "earned_points": 0.0,
            "max_points": max_points,
            "checks": [],
            "violations": [violation],
            "failure_localization": [violation],
        }
    checks = {
        "json_file_transform": _verify_json_file_transform,
        "sqlite_row_edit": _verify_sqlite_row_edit,
        "filesystem_state": _verify_filesystem_state,
        "jsonl_inventory": _verify_jsonl_inventory,
    }[task.task_type](task, root)
    return _verification_result(task, checks)


def pass_rate(results: Sequence[Mapping[str, Any]]) -> float:
    """Return the fraction of verifier rows that passed."""

    return sum(1 for result in results if result.get("passed") is True) / len(results)


def reject_rate(results: Sequence[Mapping[str, Any]]) -> float:
    """Return the fraction of verifier rows that were rejected."""

    return sum(1 for result in results if result.get("passed") is not True) / len(results)


def build_experiment_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build the Exp 2920 state-verifier harness artifact."""

    config = config or ExperimentConfig()
    started = config.start_time()
    tasks = build_state_tasks()
    manifest_path = config.task_manifest_path()
    manifest = write_task_manifest(tasks, manifest_path)
    golden_results, negative_results = _run_fixture_matrix(tasks)
    golden_rate = pass_rate(golden_results)
    negative_rate = reject_rate(negative_results)
    ready = len(tasks) == 4 and golden_rate == 1.0 and negative_rate == 1.0
    artifact: JsonDict = {
        "artifact": "experiment_2920_opencomputer_style_state_verifier_harness_v1",
        "schema": "carnot.opencomputer_state_verifier_harness.v1",
        "honest_verdict": (
            "complete: deterministic OpenComputer-style state verifier harness ready"
            if ready
            else "blocked_state_verifier_harness_not_ready"
        ),
        "state_verifier_harness_ready": ready,
        "task_manifest_path": str(manifest_path),
        "n_state_tasks": len(tasks),
        "verifier_source_paths": [VERIFIER_SOURCE_PATH],
        "golden_state_pass_rate": golden_rate,
        "negative_state_reject_rate": negative_rate,
        "partial_credit_fields": list(PARTIAL_CREDIT_FIELDS),
        "failure_localization_examples": _failure_examples(negative_results),
        "llm_judge_used": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": max(0.0, config.clock() - started),
        "run_date": RUN_DATE,
        "task_manifest": manifest,
        "golden_state_results": golden_results,
        "negative_state_results": negative_results,
        "tests_run": list(config.tests_run),
        "field_principles": dict(FIELD_PRINCIPLES),
        "methodology_note": (
            "Pass/reject rates are fixture sanity checks over constructed golden and negative "
            "states; they are not benchmark accuracy claims."
        ),
        "manifest_checksum": _checksum(manifest),
    }
    return artifact


def write_experiment_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build and persist the Exp 2920 deliverable JSON."""

    config = config or ExperimentConfig()
    artifact = build_experiment_artifact(config)
    _write_json(config.artifact_path(), artifact)
    return artifact


def _materialize_json_file_transform(state_root: Path, variant: str) -> None:
    config_path = state_root / "config" / "app.json"
    payload = {
        "service_id": "demo-search",
        "features": {"search": {"enabled": variant == "golden"}, "legacy": False},
        "limits": {"max_items": 25},
    }
    _write_json(config_path, payload)


def _materialize_sqlite_row_edit(state_root: Path, variant: str) -> None:
    db_path = state_root / "data" / "tasks.sqlite"
    db_path.parent.mkdir(parents=True)
    status = "done" if variant == "golden" else "todo"
    with sqlite3.connect(db_path) as db:
        db.execute("CREATE TABLE tasks (slug TEXT PRIMARY KEY, status TEXT, audit_note TEXT)")
        db.execute(
            "INSERT INTO tasks VALUES (?, ?, ?)",
            ("ship-harness", status, "deterministic-verifier"),
        )
        db.execute("INSERT INTO tasks VALUES (?, ?, ?)", ("write-docs", "todo", "preserved"))


def _materialize_filesystem_state(state_root: Path, variant: str) -> None:
    workspace = state_root / "workspace"
    (workspace / "archive").mkdir(parents=True)
    (workspace / "summary.txt").write_text("state verifier complete\n", encoding="utf-8")
    (workspace / "archive" / "keep.txt").write_text("keep\n", encoding="utf-8")
    if variant == "negative":
        (workspace / "tmp").mkdir()
        (workspace / "tmp" / "cache.bin").write_bytes(b"stale cache")


def _materialize_jsonl_inventory(state_root: Path, variant: str) -> None:
    count_b = 7 if variant == "golden" else 5
    rows = [{"sku": "widget-a", "count": 3}, {"sku": "widget-b", "count": count_b}]
    inventory_path = state_root / "inventory" / "items.jsonl"
    inventory_path.parent.mkdir(parents=True)
    inventory_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _verify_json_file_transform(task: StateTask, state_root: Path) -> list[JsonDict]:
    config = json.loads((state_root / "config" / "app.json").read_text(encoding="utf-8"))
    return [
        _check(task.checks[0], config["features"]["search"]["enabled"] is True, "expected true"),
        _check(task.checks[1], config["limits"]["max_items"] == 25, "expected 25"),
        _check(task.checks[2], config["service_id"] == "demo-search", "expected demo-search"),
    ]


def _verify_sqlite_row_edit(task: StateTask, state_root: Path) -> list[JsonDict]:
    with sqlite3.connect(state_root / "data" / "tasks.sqlite") as db:
        rows = {
            slug: {"status": status, "audit_note": audit_note}
            for slug, status, audit_note in db.execute(
                "SELECT slug, status, audit_note FROM tasks ORDER BY slug"
            )
        }
    ship = rows["ship-harness"]
    docs = rows["write-docs"]
    return [
        _check(task.checks[0], ship["status"] == "done", "expected status=done"),
        _check(task.checks[1], docs["status"] == "todo", "expected status=todo"),
        _check(
            task.checks[2],
            ship["audit_note"] == "deterministic-verifier",
            "expected deterministic-verifier",
        ),
    ]


def _verify_filesystem_state(task: StateTask, state_root: Path) -> list[JsonDict]:
    workspace = state_root / "workspace"
    summary = workspace / "summary.txt"
    archive = workspace / "archive" / "keep.txt"
    cache = workspace / "tmp" / "cache.bin"
    return [
        _check(
            task.checks[0],
            summary.read_text(encoding="utf-8") == "state verifier complete\n",
            "expected summary text",
        ),
        _check(task.checks[1], archive.exists(), "expected archive marker"),
        _check(task.checks[2], not cache.exists(), "expected cache file to be absent"),
    ]


def _verify_jsonl_inventory(task: StateTask, state_root: Path) -> list[JsonDict]:
    rows = [
        json.loads(line)
        for line in (state_root / "inventory" / "items.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    by_sku = {row["sku"]: row["count"] for row in rows}
    ordered = [row["sku"] for row in rows]
    return [
        _check(task.checks[0], by_sku["widget-a"] == 3, "expected count=3"),
        _check(task.checks[1], by_sku["widget-b"] == 7, "expected count=7"),
        _check(task.checks[2], ordered == sorted(ordered), "expected ascending sku order"),
    ]


def _check(spec: CheckSpec, passed: bool, detail: str) -> JsonDict:
    return {
        "check_id": spec.check_id,
        "description": spec.description,
        "passed": bool(passed),
        "points": spec.points,
        "earned_points": spec.points if passed else 0.0,
        "localized_to": spec.localized_to,
        "detail": "ok" if passed else detail,
    }


def _verification_result(task: StateTask, checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    max_points = float(sum(check["points"] for check in checks))
    earned_points = float(sum(check["earned_points"] for check in checks))
    violations = [
        {
            "check_id": str(check["check_id"]),
            "localized_to": str(check["localized_to"]),
            "detail": str(check["detail"]),
        }
        for check in checks
        if check["passed"] is not True
    ]
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "passed": not violations,
        "score": earned_points / max_points,
        "earned_points": earned_points,
        "max_points": max_points,
        "checks": list(checks),
        "violations": violations,
        "failure_localization": violations,
    }


def _run_fixture_matrix(tasks: Sequence[StateTask]) -> tuple[list[JsonDict], list[JsonDict]]:
    golden_results: list[JsonDict] = []
    negative_results: list[JsonDict] = []
    with tempfile.TemporaryDirectory(prefix="carnot-state-verifier-") as tmp:
        root = Path(tmp)
        for task in tasks:
            golden_results.append(
                verify_task_state(task, materialize_task_state(task, "golden", root / "golden"))
            )
            negative_results.append(
                verify_task_state(task, materialize_task_state(task, "negative", root / "negative"))
            )
    return golden_results, negative_results


def _failure_examples(results: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "task_id": str(result["task_id"]),
            "check_id": str(violation["check_id"]),
            "localized_to": str(violation["localized_to"]),
            "detail": str(violation["detail"]),
        }
        for result in results
        for violation in result["violations"][:1]
    ]


def _task_to_manifest_row(task: StateTask) -> JsonDict:
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "instruction": task.instruction,
        "observable_state": {
            "kind": task.observable_kind,
            "root_relative": True,
            "paths": list(task.observable_paths),
        },
        "verifier": {
            "source_path": VERIFIER_SOURCE_PATH,
            "entrypoint": VERIFIER_ENTRYPOINT,
            "inference_substrate": INFERENCE_SUBSTRATE,
            "acceptance_object": "local_state_snapshot",
        },
        "partial_credit_fields": list(PARTIAL_CREDIT_FIELDS),
        "checks": [
            {
                "check_id": check.check_id,
                "description": check.description,
                "points": check.points,
                "localized_to": check.localized_to,
            }
            for check in task.checks
        ],
    }


def _checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "FIELD_PRINCIPLES",
    "INFERENCE_SUBSTRATE",
    "MANIFEST_FILENAME",
    "OUTPUT_FILENAME",
    "PARTIAL_CREDIT_FIELDS",
    "REQUIRED_ARTIFACT_FIELDS",
    "RUN_DATE",
    "CheckSpec",
    "ExperimentConfig",
    "StateTask",
    "build_experiment_artifact",
    "build_state_tasks",
    "materialize_task_state",
    "pass_rate",
    "reject_rate",
    "verify_task_state",
    "write_experiment_artifact",
    "write_task_manifest",
]
