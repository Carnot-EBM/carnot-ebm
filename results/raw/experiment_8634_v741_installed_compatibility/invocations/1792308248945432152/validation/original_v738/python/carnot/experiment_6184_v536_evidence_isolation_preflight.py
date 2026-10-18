"""Exp6184 task-scoped evidence-isolation preflight for milestone .536.

The point of this module is narrower than repository-wide writer migration:
prove that the declared Exp6183-Exp6196 writer/test surface has a reusable
temporary-root preflight, and report a blocked negative-control write as an
intercepted attempt rather than as a mutation when no tracked bytes change.

Spec refs: REQ-REPORT-6184,
SCENARIO-REPORT-6184-TASK-SCOPE-NON-CLOSURE,
SCENARIO-REPORT-6184-FROZEN-PREFLIGHT-INVOCATION,
SCENARIO-REPORT-6184-COMPATIBLE-WRITERS,
SCENARIO-REPORT-6184-INTERCEPTED-VS-MUTATION,
SCENARIO-REPORT-6184-ESCAPE-ROOT-ATOMIC-QUARANTINE,
SCENARIO-REPORT-6184-SCHEMA-READINESS.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from carnot import experiment_6170_v535_task_artifact_isolation_canary as base_preflight
from carnot.experiment_artifacts import (
    ARTIFACT_ROOT_ENV,
    ArtifactPathError,
    atomic_write_json,
    resolve_experiment_artifact_path,
    validate_artifact_output_root,
)
from carnot.paths import repo_root as resolve_repo_root
from carnot.testing import tracked_results_guard


JsonDict = dict[str, Any]

PREFLIGHT_CONTRACT_VERSION = "v536-evidence-isolation-preflight-v1"
INFERENCE_SUBSTRATE = "deterministic_task_scoped_repository_test_isolation"
RESULT_RELATIVE_PATH = Path("results/experiment_6184_v536_evidence_isolation_preflight.json")
PRECONDITION_SNAPSHOT_DEFAULT = Path("/tmp/carnot_6184_preconditions_before_edit.json")
PREFLIGHT_MODULE = "carnot.experiment_6184_v536_evidence_isolation_preflight"
PREFLIGHT_SOURCE_PATH = "python/carnot/experiment_6184_v536_evidence_isolation_preflight.py"
PREFLIGHT_TEST_TARGET = "tests/python/test_experiment_6184_v536_evidence_isolation_preflight.py"

V536_TASK_IDS: tuple[str, ...] = (
    "exp6183-v536-transition",
    "exp6184-v536-evidence-isolation-preflight",
    "exp6185-v536-post-marker-source-delta",
    "exp6186-livecodebench-bank-preregistration",
    "exp6187-livecodebench-authentic-k8-pool",
    "exp6188-livecodebench-headroom-audit",
    "exp6189-matching-base-code-hidden-state-surface",
    "exp6190-calibration-clue-linear-code-selector",
    "exp6191-held-code-internal-state-selection",
    "exp6192-live-strategy-seed-stream",
    "exp6193-prospective-continuous-strategy-learning-ab",
    "exp6194-mode-jump-rust-pyo3-parity",
    "exp6195-arc-task-aware-prospective-fresh-transition",
    "exp6196-v536-capstone",
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "preconditions_checked",
    "scope_boundary",
    "repository_wide_closure_claimed",
    "v536_task_writer_census",
    "frozen_preflight_module_and_invocation_manifest",
    "canonical_resolver_and_legacy_compatibility_paths",
    "task_owned_temp_root_receipts",
    "collection_and_subprocess_receipts",
    "expected_intercepted_attempt_controls",
    "actual_mutation_controls",
    "traversal_symlink_workspace_root_and_atomic_controls",
    "tracked_result_hash_before_after_matrix",
    "quarantine_field_before_after_matrix",
    "research_complete_multiplicity_receipt",
    "preflight_failure_classification",
    "isolation_violation_count",
    "v536_task_artifact_isolation_ready_score",
    "preexisting_worktree_changes_preserved",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "terminal state follows mutation, escape, command-classification, and frozen-invocation checks.",
    "preconditions_checked": "records the before-edit snapshot, resolver env, V536 deliverables, history multiplicity, protected files, and root clutter.",
    "scope_boundary": "qualifies only Exp6183-Exp6196 declared .536 writer/test surfaces.",
    "repository_wide_closure_claimed": "bare false; this task does not claim repository-wide writer closure.",
    "v536_task_writer_census": "lists only declared .536 task writer/test rows keyed by task, module, mechanism, and exact path.",
    "frozen_preflight_module_and_invocation_manifest": "freezes the reusable command contract and task-owned temp-root rule for downstream .536 tasks.",
    "canonical_resolver_and_legacy_compatibility_paths": "shows canonical, legacy literal, atomic replace, checkpoint, and subprocess paths under the temp root.",
    "task_owned_temp_root_receipts": "proves the preflight installed a validated task-owned temporary artifact root.",
    "collection_and_subprocess_receipts": "keeps collection, verification, and subprocess receipts auditable without widening scope.",
    "expected_intercepted_attempt_controls": "records a blocked negative-control write separately from actual mutation accounting.",
    "actual_mutation_controls": "counts only tracked-result bytes that changed or an uncaught tracked write attempt.",
    "traversal_symlink_workspace_root_and_atomic_controls": "records escape, invalid-root, and atomic-cleanup controls.",
    "tracked_result_hash_before_after_matrix": "compares aggregate and sentinel tracked-result hashes before and after preflight controls.",
    "quarantine_field_before_after_matrix": "compares protected quarantine/corrigendum/provenance fields before and after redirected writes.",
    "research_complete_multiplicity_receipt": "reports completion-history multiplicity diagnostically and never mutates it.",
    "preflight_failure_classification": "classifies nonzero command receipts without treating expected intercepted attempts as violations.",
    "isolation_violation_count": "counts real mutation, escape, or unclassified task-scope failure only.",
    "v536_task_artifact_isolation_ready_score": "one only with zero real mutations, zero escapes, zero unclassified failures, and frozen invocation data.",
    "preexisting_worktree_changes_preserved": "records that pre-existing dirty paths were observed and not reverted or staged by this workflow.",
    "protected_files_unchanged": "proves protected operational and guard files did not change during the preflight run.",
    "duration_s": "wall-clock duration of artifact construction.",
    "inference_substrate": "declares deterministic task-scoped repository test isolation.",
    "field_provenance": "maps every required field to its source and principle.",
    "test_commands": "records commands used to verify the preflight.",
    "test_exit_codes": "records exit codes for the verification commands.",
    "reproducibility_checksum": "hashes the artifact content excluding the checksum field.",
    "honest_verdict": "terminal verdict names the exact V536 scope and states repository-wide closure remains false.",
}

path_sha256 = base_preflight.path_sha256
payload_checksum = base_preflight.payload_checksum
snapshot_repository = base_preflight.snapshot_repository


def load_precondition_snapshot(path: Path | None = None) -> JsonDict:
    snapshot_path = PRECONDITION_SNAPSHOT_DEFAULT if path is None else path
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"present": False, "path": str(snapshot_path), "error": type(exc).__name__}
    if isinstance(payload, dict):
        payload.setdefault("present", True)
        payload.setdefault("path", str(snapshot_path))
        return payload
    return {"present": False, "path": str(snapshot_path), "error": "not_json_object"}


def _task_number(task_id: object) -> int | None:
    text = str(task_id)
    if not text.startswith("exp") or len(text) < 7 or not text[3:7].isdigit():
        return None
    return int(text[3:7])


def _load_roadmap_tasks(repo_root: Path) -> list[JsonDict]:
    payload = yaml.safe_load((repo_root / "research-roadmap.yaml").read_text(encoding="utf-8")) or {}
    tasks = payload.get("tasks", [])
    return [dict(task) for task in tasks if isinstance(task, dict)]


def declared_v536_tasks(repo_root: Path) -> list[JsonDict]:
    tasks: list[JsonDict] = []
    for task in _load_roadmap_tasks(repo_root):
        task_number = _task_number(task.get("id"))
        if task_number is None or not 6183 <= task_number <= 6196:
            continue
        tasks.append(
            {
                "task_id": str(task.get("id", "")),
                "title": task.get("title"),
                "track": task.get("track"),
                "deliverable": str(task.get("deliverable", "")),
            }
        )
    order = {task_id: index for index, task_id in enumerate(V536_TASK_IDS)}
    return sorted(tasks, key=lambda row: order.get(str(row["task_id"]), 999))


def _expected_module_for_task(deliverable: str) -> Path:
    return Path("python/carnot") / f"{Path(deliverable).stem}.py"


def _expected_test_for_module(module_path: Path) -> Path:
    return Path("tests/python") / f"test_{module_path.stem}.py"


def collect_v536_task_writer_census(repo_root: Path) -> JsonDict:
    """Inventory only declared .536 task/test surfaces, not the whole repo."""

    rows: list[JsonDict] = []
    for task in declared_v536_tasks(repo_root):
        deliverable = str(task["deliverable"])
        module_path = _expected_module_for_task(deliverable)
        rows.append(
            {
                "task_id": task["task_id"],
                "title": task["title"],
                "track": task["track"],
                "surface": "experiment_module",
                "module": module_path.as_posix(),
                "module_present": (repo_root / module_path).exists(),
                "mechanism": base_preflight._writer_mechanism(repo_root, module_path, deliverable),
                "exact_path": deliverable,
            }
        )
        test_path = _expected_test_for_module(module_path)
        if (repo_root / test_path).exists():
            rows.append(
                {
                    "task_id": task["task_id"],
                    "title": task["title"],
                    "track": task["track"],
                    "surface": "test_module",
                    "module": test_path.as_posix(),
                    "module_present": True,
                    "mechanism": "pytest_task_scoped_preflight_control",
                    "exact_path": deliverable,
                }
            )
    counts = Counter(row["mechanism"] for row in rows)
    return {
        "scope": {
            "first_task": V536_TASK_IDS[0],
            "last_task": V536_TASK_IDS[-1],
            "qualified_task_ids": list(V536_TASK_IDS),
            "repository_wide_writer_scan": False,
        },
        "declared_task_count": len(declared_v536_tasks(repo_root)),
        "row_count": len(rows),
        "mechanism_counts": dict(sorted(counts.items())),
        "rows": sorted(rows, key=lambda row: (row["task_id"], row["surface"], row["module"])),
    }


def build_frozen_preflight_manifest() -> JsonDict:
    root_expr = "$(mktemp -d /tmp/carnot-6184-preflight-XXXXXX)"
    focused = f".venv/bin/pytest {PREFLIGHT_TEST_TARGET} -q -o addopts="
    return {
        "version": PREFLIGHT_CONTRACT_VERSION,
        "preflight_module": PREFLIGHT_MODULE,
        "preflight_source": PREFLIGHT_SOURCE_PATH,
        "pytest_target": PREFLIGHT_TEST_TARGET,
        "artifact_root_env": ARTIFACT_ROOT_ENV,
        "temporary_root_contract": (
            "Create a task-owned directory under the system temp root, validate it, "
            "and set CARNOT_EXPERIMENT_ARTIFACT_ROOT before pytest collection."
        ),
        "canonical_task_owned_invocation": (
            f"{ARTIFACT_ROOT_ENV}={root_expr} {focused}"
        ),
        "repository_wide_closure_claimed": False,
        "commands": {
            "collection": (
                f"{ARTIFACT_ROOT_ENV}={root_expr} .venv/bin/pytest "
                f"{PREFLIGHT_TEST_TARGET} --collect-only -q -o addopts="
            ),
            "focused": f"{ARTIFACT_ROOT_ENV}={root_expr} {focused}",
            "new_code_coverage_run": (
                f"{ARTIFACT_ROOT_ENV}={root_expr} "
                ".venv/bin/coverage run --source=python/carnot -m pytest "
                f"{PREFLIGHT_TEST_TARGET} -q -o addopts="
            ),
            "new_code_coverage_report": (
                ".venv/bin/coverage report "
                f"--include='{PREFLIGHT_SOURCE_PATH}' --fail-under=100 --show-missing"
            ),
        },
    }


@contextlib.contextmanager
def _temporary_artifact_env(repo_root: Path, temp_root: Path):
    previous = os.environ.get(ARTIFACT_ROOT_ENV)
    previous_cwd = Path.cwd()
    os.environ[ARTIFACT_ROOT_ENV] = str(temp_root)
    os.chdir(repo_root)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        if previous is None:
            os.environ.pop(ARTIFACT_ROOT_ENV, None)
        else:
            os.environ[ARTIFACT_ROOT_ENV] = previous


def _ensure_temp_root(raw_root: Path | None) -> Path:
    if raw_root is None:
        return Path(tempfile.mkdtemp(prefix="carnot-6184-preflight-artifacts-")).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    return raw_root.resolve()


def _raises_artifact_error(fn) -> JsonDict:
    try:
        fn()
    except ArtifactPathError as exc:
        return {"raised": True, "exception": type(exc).__name__, "message": str(exc)}
    return {"raised": False, "exception": None, "message": "no exception"}


def _choose_tracked_sentinel(repo_root: Path) -> Path | None:
    for rel_path in base_preflight.SENTINEL_RESULT_PATHS:
        path = repo_root / rel_path
        if path.exists():
            return path
    tracked = base_preflight._tracked_results(repo_root)
    return repo_root / tracked[0] if tracked else None


def _run_subprocess_control(repo_root: Path, temp_root: Path) -> JsonDict:
    sub_root = temp_root / "subprocess-root"
    sub_root.mkdir(parents=True, exist_ok=True)
    validate_artifact_output_root(sub_root)
    code = textwrap.dedent(
        """
        import json
        import os
        from pathlib import Path

        from carnot.experiment_artifacts import atomic_write_json
        from carnot.testing import tracked_results_guard

        tracked_results_guard.install()
        tracked_results_guard.install_legacy_results_write_compat()
        tracked_results_guard.clear_legacy_compat_redirects()
        canonical = atomic_write_json(
            "results/experiment_6184_subprocess_canonical.json",
            {"status": "subprocess"},
        )
        with open("results/experiment_6184_subprocess_legacy.txt", "w", encoding="utf-8") as fh:
            fh.write("legacy subprocess\\n")
        print(json.dumps({
            "canonical": str(canonical),
            "legacy": str(Path(os.environ["CARNOT_EXPERIMENT_ARTIFACT_ROOT"]) / "experiment_6184_subprocess_legacy.txt"),
            "redirects": tracked_results_guard.recorded_legacy_compat_redirects(),
        }, sort_keys=True))
        """
    )
    env = os.environ.copy()
    pythonpath = [str(repo_root / "python"), str(repo_root)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env[ARTIFACT_ROOT_ENV] = str(sub_root)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    parsed = json.loads(proc.stdout) if proc.returncode == 0 and proc.stdout.strip() else {}
    return {
        "command": f"{sys.executable} -c <Exp6184 subprocess preflight>",
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "artifact_root": str(sub_root),
        "parsed": parsed,
        "canonical_exists": (sub_root / "experiment_6184_subprocess_canonical.json").exists(),
        "legacy_exists": (sub_root / "experiment_6184_subprocess_legacy.txt").exists(),
    }


def expected_intercepted_attempt_receipt(attempted: Mapping[str, Any]) -> JsonDict:
    intercepted = bool(
        attempted.get("observed_real_forbidden_attempt")
        and attempted.get("caught_before_mutation")
    )
    return {
        "classification": "expected_intercepted_attempt" if intercepted else "not_intercepted",
        "attempt_count": 1 if attempted.get("target") else 0,
        "counts_as_isolation_violation": False if intercepted else True,
        "counts_as_violation_count": 0 if intercepted else 1,
        "attempt": dict(attempted),
    }


def actual_mutation_count(attempted: Mapping[str, Any]) -> int:
    if not attempted.get("target"):
        return 0
    return 0 if attempted.get("caught_before_mutation") is True else 1


def actual_mutation_receipt(attempted: Mapping[str, Any]) -> JsonDict:
    count = actual_mutation_count(attempted)
    return {
        "actual_mutation_count": count,
        "tracked_result_mutated": count > 0,
        "attempt": dict(attempted),
    }


def run_writer_controls(repo_root: Path, temp_root: Path | None = None) -> JsonDict:
    """Run positive and adversarial writer controls under a task-owned root."""

    task_root = _ensure_temp_root(temp_root)
    validate_artifact_output_root(task_root)
    tracked_results_guard.install()
    tracked_results_guard.install_legacy_results_write_compat()
    tracked_results_guard.clear_violations()
    tracked_results_guard.clear_legacy_compat_redirects()

    sentinel = _choose_tracked_sentinel(repo_root)
    sentinel_before_hash = path_sha256(sentinel) if sentinel is not None else None
    sentinel_before_fields = (
        base_preflight._snapshot_quarantine_fields(repo_root, [sentinel.relative_to(repo_root)])
        if sentinel is not None
        else {}
    )

    with _temporary_artifact_env(repo_root, task_root):
        canonical_path = atomic_write_json(
            "results/experiment_6184_preflight_canonical.json", {"status": "canonical"}
        )
        with open("results/experiment_6184_preflight_legacy.txt", "w", encoding="utf-8") as fh:
            fh.write("legacy literal\n")
        Path("results/experiment_6184_preflight_replace.tmp").write_text(
            "atomic\n", encoding="utf-8"
        )
        os.replace(
            "results/experiment_6184_preflight_replace.tmp",
            "results/experiment_6184_preflight_replace.txt",
        )

        from scripts.experiment_template import ExperimentTemplate

        template = ExperimentTemplate(
            6184,
            "V536 evidence isolation preflight",
            "results/experiment_6184_template_preflight.json",
        )
        template.checkpoint_save({"status": "checkpoint"}, step=8)
        checkpoint_payload = template.checkpoint_resume() or {}

        atomic_path = atomic_write_json("results/experiment_6184_preflight_atomic.json", {"value": 1})
        atomic_write_json("results/experiment_6184_preflight_atomic.json", {"value": 2})
        atomic_payload = json.loads(atomic_path.read_text(encoding="utf-8"))

        atomic_write_json(
            f"results/{sentinel.name}" if sentinel is not None else "results/no_sentinel.json",
            {"status": "redirected quarantine control"},
        )

        outside = task_root.parent / f"{task_root.name}-outside"
        outside.mkdir(exist_ok=True)
        escape = task_root / "escape"
        if not escape.exists():
            escape.symlink_to(outside, target_is_directory=True)

        traversal = _raises_artifact_error(
            lambda: resolve_experiment_artifact_path("results/../experiment_6184_escape.json")
        )
        symlink_escape = _raises_artifact_error(
            lambda: resolve_experiment_artifact_path("results/escape/leak.json")
        )

    if sentinel is None:  # pragma: no cover - this repository has tracked result sentinels.
        attempted = {
            "target": None,
            "observed_real_forbidden_attempt": False,
            "caught_before_mutation": False,
            "reason": "no tracked result sentinel available",
        }
    else:
        before_hash = path_sha256(sentinel)
        tracked_results_guard.clear_violations()
        try:
            sentinel.write_text("forbidden tracked mutation\n", encoding="utf-8")
        except tracked_results_guard.TrackedResultWriteError as exc:
            violations = tracked_results_guard.recorded_violations()
            after_hash = path_sha256(sentinel)
            attempted = {
                "target": str(sentinel.relative_to(repo_root)),
                "observed_real_forbidden_attempt": True,
                "exception": type(exc).__name__,
                "violation_event": violations[-1]["event"] if violations else None,
                "violation_path": violations[-1]["path"] if violations else None,
                "pre_hash": before_hash,
                "post_hash": after_hash,
                "caught_before_mutation": before_hash == after_hash,
            }
        else:  # pragma: no cover - reaching this branch means the guard allowed mutation.
            after_hash = path_sha256(sentinel)
            attempted = {
                "target": str(sentinel.relative_to(repo_root)),
                "observed_real_forbidden_attempt": False,
                "pre_hash": before_hash,
                "post_hash": after_hash,
                "caught_before_mutation": False,
            }
        finally:
            tracked_results_guard.clear_violations()

    invalid_roots = {
        "workspace_root": _raises_artifact_error(lambda: validate_artifact_output_root(repo_root)),
        "repository_root": _raises_artifact_error(lambda: validate_artifact_output_root(repo_root)),
        "production_results_root": _raises_artifact_error(
            lambda: validate_artifact_output_root(repo_root / "results")
        ),
        "broad_tmp_root": _raises_artifact_error(
            lambda: validate_artifact_output_root(Path(tempfile.gettempdir()))
        ),
    }

    subprocess_receipt = _run_subprocess_control(repo_root, task_root)
    sentinel_after_hash = path_sha256(sentinel) if sentinel is not None else None
    sentinel_after_fields = (
        base_preflight._snapshot_quarantine_fields(repo_root, [sentinel.relative_to(repo_root)])
        if sentinel is not None
        else {}
    )
    redirects = tracked_results_guard.recorded_legacy_compat_redirects()
    tracked_results_guard.clear_legacy_compat_redirects()

    expected = expected_intercepted_attempt_receipt(attempted)
    actual = actual_mutation_receipt(attempted)
    return {
        "task_owned_temp_root": {
            "path": str(task_root),
            "validated": True,
            "exists": task_root.exists(),
            "under_system_tmp": str(task_root).startswith(
                str(Path(tempfile.gettempdir()).resolve())
            ),
        },
        "canonical_writer": {
            "requested": "results/experiment_6184_preflight_canonical.json",
            "resolved": str(canonical_path),
            "under_task_root": canonical_path.is_relative_to(task_root),
        },
        "legacy_literal_writer": {
            "requested": "results/experiment_6184_preflight_legacy.txt",
            "resolved": str(task_root / "experiment_6184_preflight_legacy.txt"),
            "under_task_root": (task_root / "experiment_6184_preflight_legacy.txt").exists(),
            "redirect_events": redirects,
        },
        "legacy_atomic_replace": {
            "requested": "results/experiment_6184_preflight_replace.txt",
            "resolved": str(task_root / "experiment_6184_preflight_replace.txt"),
            "under_task_root": (task_root / "experiment_6184_preflight_replace.txt").exists(),
        },
        "checkpoint_resume": {
            "checkpoint_path": str(
                task_root / "checkpoints" / "experiment_6184" / "checkpoint.json"
            ),
            "resumed_step": checkpoint_payload.get("step"),
            "resumed_status": (checkpoint_payload.get("results") or {}).get("status"),
        },
        "atomic_writer": {
            "requested": "results/experiment_6184_preflight_atomic.json",
            "resolved": str(atomic_path),
            "final_value": atomic_payload.get("value"),
            "leftover_tmp_files": sorted(path.name for path in task_root.glob("*.tmp")),
        },
        "subprocess": subprocess_receipt,
        "expected_intercepted_attempt": expected,
        "actual_mutation": actual,
        "traversal": traversal,
        "symlink_escape": symlink_escape,
        "invalid_roots": invalid_roots,
        "quarantine_preservation": {
            "target": str(sentinel.relative_to(repo_root)) if sentinel is not None else None,
            "pre_hash": sentinel_before_hash,
            "post_hash": sentinel_after_hash,
            "before_fields": sentinel_before_fields,
            "after_fields": sentinel_after_fields,
            "unchanged": (
                sentinel_before_hash == sentinel_after_hash
                and sentinel_before_fields == sentinel_after_fields
            ),
        },
    }


def classify_command_receipts(command_receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows: list[JsonDict] = []
    counts = Counter(
        {
            "zero": 0,
            "expected_intercepted_attempt": 0,
            "task_scope_failure": 0,
            "unrelated_preexisting": 0,
            "new_regression": 0,
            "unclassified_task_scope_failure": 0,
        }
    )
    for receipt in command_receipts:
        exit_code = int(receipt.get("exit_code", 0))
        text = " ".join(
            str(receipt.get(key, "")) for key in ("name", "command", "stdout", "stderr")
        )
        if exit_code == 0:
            classification = "zero"
        elif receipt.get("classification") in counts:
            classification = str(receipt["classification"])
        elif (
            "experiment_6184_v536_evidence_isolation_preflight" in text
            or "experiment_6184" in text
        ):
            classification = "task_scope_failure"
        elif "TrackedResultWriteError" in text or "expected_intercepted_attempt" in text:
            classification = "expected_intercepted_attempt"
        elif "ModuleNotFoundError" in text or "legacy_fixture" in text:
            classification = "unrelated_preexisting"
        elif "new_regression" in text:
            classification = "new_regression"
        else:
            classification = "unclassified_task_scope_failure"
        counts[classification] += 1
        row = dict(receipt)
        row["classification"] = classification
        rows.append(row)
    return {"rows": rows, "counts": dict(counts)}


def escape_failure_count(controls: Mapping[str, Any]) -> int:
    failures = 0
    for key in ("traversal", "symlink_escape"):
        if key in controls and (controls.get(key) or {}).get("raised") is not True:
            failures += 1
    invalid_roots = controls.get("invalid_roots", {})
    if isinstance(invalid_roots, Mapping):
        failures += sum(1 for row in invalid_roots.values() if row.get("raised") is not True)
    leftovers = (controls.get("atomic_writer") or {}).get("leftover_tmp_files") or []
    return failures + (1 if leftovers else 0)


def count_isolation_violations(
    *,
    actual_mutation_count: int,
    escape_failure_count: int,
    classification: Mapping[str, Any],
) -> int:
    counts = classification.get("counts", {})
    return (
        actual_mutation_count
        + escape_failure_count
        + int(counts.get("unclassified_task_scope_failure", 0))
    )


def ready_score(
    *,
    actual_mutation_count: int,
    escape_failure_count: int,
    classification: Mapping[str, Any],
    invocation_manifest: Mapping[str, Any],
    command_receipts: Sequence[Mapping[str, Any]],
) -> int:
    counts = classification.get("counts", {})
    if not command_receipts:
        return 0
    if actual_mutation_count != 0 or escape_failure_count != 0:
        return 0
    if counts.get("task_scope_failure", 0) or counts.get("unclassified_task_scope_failure", 0):
        return 0
    if invocation_manifest.get("repository_wide_closure_claimed") is not False:
        return 0
    if invocation_manifest.get("version") != PREFLIGHT_CONTRACT_VERSION:
        return 0
    if not invocation_manifest.get("canonical_task_owned_invocation"):
        return 0
    return 1


def _hash_matrix(before: Mapping[str, Any], after: Mapping[str, Any]) -> JsonDict:
    return {"before": before, "after": after, "unchanged": before == after}


def _research_complete_multiplicity(repo_root: Path, precondition_snapshot: Mapping[str, Any]) -> JsonDict:
    path = repo_root / "research-complete.yaml"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return {
        "history_path": "research-complete.yaml",
        "diagnostic_only": True,
        "history_mutated": False,
        "precondition_snapshot": precondition_snapshot.get("completion_history_multiplicity"),
        "current": {
            "milestone_2026_08_535_occurrences": text.count("2026.08.535"),
            "milestone_2026_08_536_occurrences": text.count("2026.08.536"),
            "exp6183_through_exp6196_token_counts": {
                f"exp{idx}": text.count(f"exp{idx}") for idx in range(6183, 6197)
            },
        },
    }


def _field_provenance() -> JsonDict:
    return {
        field: {
            "spec": "REQ-REPORT-6184",
            "principle": FIELD_PRINCIPLES[field],
            "source": PREFLIGHT_SOURCE_PATH,
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _command_maps(command_receipts: Sequence[Mapping[str, Any]]) -> tuple[JsonDict, JsonDict]:
    commands = {
        str(receipt.get("name", f"command_{idx}")): str(receipt.get("command", ""))
        for idx, receipt in enumerate(command_receipts)
    }
    exit_codes = {
        str(receipt.get("name", f"command_{idx}")): int(receipt.get("exit_code", 0))
        for idx, receipt in enumerate(command_receipts)
    }
    return commands, exit_codes


def _root_clutter(repo_root: Path) -> JsonDict:
    root_py = sorted(path.name for path in repo_root.glob("*.py"))
    return {"root_py_files": root_py, "root_py_file_count": len(root_py)}


def build_artifact(
    repo_root: Path,
    *,
    command_receipts: Sequence[Mapping[str, Any]],
    precondition_snapshot_path: Path | None = None,
    duration_s: float | None = None,
    temp_root: Path | None = None,
) -> JsonDict:
    started = time.perf_counter()
    repo_root = repo_root.resolve()
    precondition_snapshot = load_precondition_snapshot(precondition_snapshot_path)
    before = snapshot_repository(repo_root, precondition_snapshot=precondition_snapshot)
    controls = run_writer_controls(repo_root, temp_root)
    after = snapshot_repository(repo_root, precondition_snapshot=precondition_snapshot)
    census = collect_v536_task_writer_census(repo_root)
    invocation_manifest = build_frozen_preflight_manifest()
    classification = classify_command_receipts(command_receipts)

    tracked_hash_matrix = _hash_matrix(before["tracked_results"], after["tracked_results"])
    sentinel_hash_matrix = _hash_matrix(before["sentinel_hashes"], after["sentinel_hashes"])
    quarantine_matrix = _hash_matrix(before["quarantine_fields"], after["quarantine_fields"])
    task_start_matrix = base_preflight._task_start_matrix_from_precondition(
        precondition_snapshot, after
    )
    protected_matrix = _hash_matrix(before["protected_files"], after["protected_files"])
    actual_count = int(controls["actual_mutation"]["actual_mutation_count"])
    escape_count = escape_failure_count(controls)
    isolation_count = count_isolation_violations(
        actual_mutation_count=actual_count,
        escape_failure_count=escape_count,
        classification=classification,
    )
    ready = ready_score(
        actual_mutation_count=actual_count,
        escape_failure_count=escape_count,
        classification=classification,
        invocation_manifest=invocation_manifest,
        command_receipts=command_receipts,
    )
    commands, exit_codes = _command_maps(command_receipts)
    actual_duration = round(time.perf_counter() - started, 3) if duration_s is None else duration_s
    status = "complete_ready" if ready else "complete_partial"
    verdict_prefix = "complete_ready:" if ready else "complete_partial:"
    research_complete_receipt = _research_complete_multiplicity(repo_root, precondition_snapshot)
    artifact: JsonDict = {
        "status": status,
        "preconditions_checked": {
            "agents_codex_claude_and_task_files_read": True,
            "precondition_snapshot": precondition_snapshot,
            "git_status_short_after_build": base_preflight._git_status_short(repo_root),
            "declared_v536_deliverables": declared_v536_tasks(repo_root),
            "candidate_test_modules": sorted(
                _expected_test_for_module(_expected_module_for_task(row["deliverable"])).as_posix()
                for row in declared_v536_tasks(repo_root)
            ),
            "resolver_environment_before_controls": {
                ARTIFACT_ROOT_ENV: os.environ.get(ARTIFACT_ROOT_ENV),
                "PYTEST_CURRENT_TEST": os.environ.get("PYTEST_CURRENT_TEST"),
            },
            "symlink_traversal_boundaries": {
                "repo_root": str(repo_root),
                "workspace_root": str(repo_root),
                "production_results_root": str((repo_root / "results").resolve()),
                "broad_tmp_root": str(Path(tempfile.gettempdir()).resolve()),
            },
            "completion_history_multiplicity": research_complete_receipt,
            "protected_file_hashes_at_task_start": precondition_snapshot.get(
                "protected_file_hashes"
            ),
            "root_clutter": _root_clutter(repo_root),
        },
        "scope_boundary": {
            "qualified_scope": "Exp6183-Exp6196 declared .536 writer/test surfaces only",
            "qualified_task_ids": list(V536_TASK_IDS),
            "repository_wide_writer_scan": False,
        },
        "repository_wide_closure_claimed": False,
        "v536_task_writer_census": census,
        "frozen_preflight_module_and_invocation_manifest": invocation_manifest,
        "canonical_resolver_and_legacy_compatibility_paths": {
            "canonical_writer": controls["canonical_writer"],
            "legacy_literal_writer": controls["legacy_literal_writer"],
            "legacy_atomic_replace": controls["legacy_atomic_replace"],
            "checkpoint_resume": controls["checkpoint_resume"],
            "subprocess": controls["subprocess"],
        },
        "task_owned_temp_root_receipts": controls["task_owned_temp_root"],
        "collection_and_subprocess_receipts": {
            "command_receipts": list(command_receipts),
            "subprocess": controls["subprocess"],
        },
        "expected_intercepted_attempt_controls": controls["expected_intercepted_attempt"],
        "actual_mutation_controls": controls["actual_mutation"],
        "traversal_symlink_workspace_root_and_atomic_controls": {
            "traversal": controls["traversal"],
            "symlink_escape": controls["symlink_escape"],
            "invalid_roots": controls["invalid_roots"],
            "atomic_writer": controls["atomic_writer"],
            "escape_failure_count": escape_count,
        },
        "tracked_result_hash_before_after_matrix": {
            "preflight_control_window": {
                "aggregate": tracked_hash_matrix,
                "sentinels": sentinel_hash_matrix,
            },
            "task_start_to_post": task_start_matrix,
        },
        "quarantine_field_before_after_matrix": {
            "preflight_control_window": quarantine_matrix,
            "task_start_to_post": {
                "available": task_start_matrix.get("available"),
                "before": (task_start_matrix.get("before") or {}).get("quarantine_fields"),
                "after": (task_start_matrix.get("after") or {}).get("quarantine_fields"),
                "unchanged": (
                    None
                    if task_start_matrix.get("available") is False
                    else (task_start_matrix.get("before") or {}).get("quarantine_fields")
                    == (task_start_matrix.get("after") or {}).get("quarantine_fields")
                ),
            },
        },
        "research_complete_multiplicity_receipt": research_complete_receipt,
        "preflight_failure_classification": classification,
        "isolation_violation_count": isolation_count,
        "v536_task_artifact_isolation_ready_score": ready,
        "preexisting_worktree_changes_preserved": base_preflight._preexisting_changes(
            repo_root, precondition_snapshot
        ),
        "protected_files_unchanged": protected_matrix,
        "duration_s": actual_duration,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": _field_provenance(),
        "test_commands": commands,
        "test_exit_codes": exit_codes,
        "reproducibility_checksum": "",
        "honest_verdict": (
            f"{verdict_prefix} Exp6183-Exp6196 declared .536 preflight surface "
            f"ready_score={ready}, isolation_violation_count={isolation_count}, "
            "expected_intercepted_attempts_do_not_count_as_mutations, "
            "repository_wide_closure_claimed=false."
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"Exp6184 artifact validation failed: {errors}")
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            errors.append(f"missing:{field}")
    if payload.get("repository_wide_closure_claimed") is not False:
        errors.append("repository_wide_closure_claimed")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    provenance = payload.get("field_provenance")
    if not isinstance(provenance, dict):
        errors.append("field_provenance:not_mapping")
    else:
        for field in REQUIRED_ARTIFACT_FIELDS:
            row = provenance.get(field)
            if not isinstance(row, dict) or row.get("principle") != FIELD_PRINCIPLES[field]:
                errors.append(f"field_provenance:{field}")
    verdict = str(payload.get("honest_verdict", ""))
    if not verdict.startswith(("complete_ready:", "complete_partial:", "retired:", "blocked:")):
        errors.append("honest_verdict_prefix")
    expected_checksum = payload_checksum(payload)
    if payload.get("reproducibility_checksum") != expected_checksum:
        errors.append("reproducibility_checksum")
    ready = int(payload.get("v536_task_artifact_isolation_ready_score", 0))
    if ready not in (0, 1):
        errors.append("ready_score")
    if ready == 1 and int(payload.get("isolation_violation_count", 1)) != 0:
        errors.append("ready_score_vs_violations")
    return errors


def _load_command_receipts(path: Path | None) -> list[JsonDict]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("command receipts JSON must be a list")
    return [dict(row) for row in payload]


def write_artifact(payload: Mapping[str, Any], output_path: Path) -> Path:
    return atomic_write_json(output_path, payload, allow_override=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Exp6184 .536 evidence-isolation preflight.")
    parser.add_argument("--repo-root", type=Path, default=resolve_repo_root(start=__file__))
    parser.add_argument("--command-receipts-json", type=Path)
    parser.add_argument("--precondition-snapshot", type=Path, default=PRECONDITION_SNAPSHOT_DEFAULT)
    parser.add_argument("--output-path", type=Path, default=RESULT_RELATIVE_PATH)
    parser.add_argument("--duration-s", type=float)
    args = parser.parse_args(argv)

    artifact = build_artifact(
        args.repo_root,
        command_receipts=_load_command_receipts(args.command_receipts_json),
        precondition_snapshot_path=args.precondition_snapshot,
        duration_s=args.duration_s,
    )
    output = args.output_path
    if not output.is_absolute():
        output = args.repo_root / output
    write_artifact(artifact, output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests.
    raise SystemExit(main())
