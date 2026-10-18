"""Exp6170 task-scoped artifact-isolation canary for milestone .535.

The point of this module is deliberately narrow: prove that the declared
Exp6169-Exp6182 writer/test surface can run through the existing artifact
resolver and tracked-results guard without rewriting committed evidence. It is
not a repository-wide writer migration.

Spec refs: REQ-REPORT-6170,
SCENARIO-REPORT-6170-TASK-SCOPE-NON-CLOSURE,
SCENARIO-REPORT-6170-FROZEN-CANARY-INVOCATION,
SCENARIO-REPORT-6170-COMPATIBLE-WRITERS,
SCENARIO-REPORT-6170-ADVERSARIAL-CONTROLS,
SCENARIO-REPORT-6170-QUARANTINE-PRESERVATION,
SCENARIO-REPORT-6170-SCHEMA-READINESS.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
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

CANARY_CONTRACT_VERSION = "v1"
INFERENCE_SUBSTRATE = "deterministic_task_scoped_repository_test_isolation"
RESULT_RELATIVE_PATH = Path("results/experiment_6170_v535_task_artifact_isolation_canary.json")
PRECONDITION_SNAPSHOT_DEFAULT = Path("/tmp/carnot_6170_preconditions_before_edit.json")
CANARY_MODULE = "carnot.experiment_6170_v535_task_artifact_isolation_canary"
CANARY_TEST_TARGET = "tests/python/test_experiment_6170_v535_task_artifact_isolation_canary.py"
CANARY_SOURCE_PATH = "python/carnot/experiment_6170_v535_task_artifact_isolation_canary.py"

V535_TASK_IDS: tuple[str, ...] = (
    "exp6169-v535-transition",
    "exp6170-v535-task-artifact-isolation-canary",
    "exp6171-v535-source-delta-ingestion",
    "exp6172-current-rule-quarantine-determination",
    "exp6173-cctu-item-bank-preregistration",
    "exp6174-cctu-authentic-k8-pool",
    "exp6175-cctu-headroom-audit",
    "exp6176-hidden-state-surface-qualification",
    "exp6177-clue-latent-selector-freeze",
    "exp6178-held-internal-state-selection",
    "exp6179-retention-safe-continuous-strategy-learning-ab",
    "exp6180-exp6166-reproducibility-adjudication",
    "exp6181-arc-logo-shortcut-audit",
    "exp6182-v535-capstone-reconciliation",
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "preconditions_checked",
    "scope_boundary_and_repository_wide_closure_claimed",
    "v535_task_writer_census",
    "frozen_canary_module_and_invocation_manifest",
    "canonical_resolver_and_legacy_compatibility_paths",
    "task_owned_temp_root_receipts",
    "collection_and_subprocess_receipts",
    "attempted_tracked_write_controls",
    "traversal_symlink_workspace_root_and_atomic_controls",
    "tracked_result_hash_before_after_matrix",
    "quarantine_field_before_after_matrix",
    "canary_failure_classification",
    "isolation_violation_count",
    "v535_task_artifact_isolation_ready_score",
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
    "status": "terminal state is derived from tracked-mutation checks, canary failures, command classification, and invocation freeze.",
    "preconditions_checked": "records the before-edit snapshot, dirty worktree, tracked hashes, roadmap declarations, resolver env, boundaries, and protected files.",
    "scope_boundary_and_repository_wide_closure_claimed": "qualifies only Exp6169-Exp6182 and records repository-wide closure as bare false.",
    "v535_task_writer_census": "lists only declared .535 task writer/test surfaces keyed by task, module, mechanism, and exact path.",
    "frozen_canary_module_and_invocation_manifest": "freezes the command contract downstream .535 tasks can reuse.",
    "canonical_resolver_and_legacy_compatibility_paths": "shows canonical, legacy literal, atomic, checkpoint, and subprocess paths under the temp root.",
    "task_owned_temp_root_receipts": "proves the canary installed a validated task-owned temporary artifact root.",
    "collection_and_subprocess_receipts": "keeps command and subprocess receipts auditable without implying repo-wide closure.",
    "attempted_tracked_write_controls": "records a real tracked write attempt caught before bytes changed.",
    "traversal_symlink_workspace_root_and_atomic_controls": "records escape, invalid-root, and atomic-cleanup controls.",
    "tracked_result_hash_before_after_matrix": "compares aggregate and sentinel tracked-result hashes before and after canary controls.",
    "quarantine_field_before_after_matrix": "compares protected quarantine/corrigendum/provenance fields before and after redirected writes.",
    "canary_failure_classification": "classifies nonzero commands as canary-scope, unrelated pre-existing, new regression, or unclassified.",
    "isolation_violation_count": "counts tracked mutations and canary-scope failures.",
    "v535_task_artifact_isolation_ready_score": "one only with zero mutations, zero canary-scope failures, frozen invocation, and classified nonzero commands.",
    "preexisting_worktree_changes_preserved": "records that pre-existing dirty paths were observed and not reverted or staged by this workflow.",
    "protected_files_unchanged": "proves protected operational and guard files did not change during the canary run.",
    "duration_s": "wall-clock duration of artifact construction.",
    "inference_substrate": "declares deterministic task-scoped repository test isolation.",
    "field_provenance": "maps every required field to its source and principle.",
    "test_commands": "records commands used to verify the canary.",
    "test_exit_codes": "records exit codes for the verification commands.",
    "reproducibility_checksum": "hashes the artifact content excluding the checksum field.",
    "honest_verdict": "terminal verdict names the exact qualified scope without repository-wide closure.",
}

QUARANTINE_FIELDS: tuple[str, ...] = (
    "flagged_adversarial",
    "corrigendum_pending",
    "corrigendum_note",
    "flagged_adversarial_restoration_note",
    "flagged_adversarial_restored_fields",
    "inference_substrate",
    "inference_mode",
    "solve_provenance",
    "solve_provenance_note",
    "inference_substrate_correction_note",
    "inference_substrate_original_invalid_value",
)

SENTINEL_RESULT_PATHS: tuple[Path, ...] = (
    Path("results/experiment_1938_nrgpt_loss_probe.json"),
    Path("results/experiment_2085_pem_sudoku_eval.json"),
    Path("results/experiment_6143_test_artifact_isolation.json"),
)

PROTECTED_FILE_PATHS: tuple[Path, ...] = (
    Path("scripts/research_conductor.py"),
    Path("python/carnot/experiment_artifacts.py"),
    Path("python/carnot/testing/tracked_results_guard.py"),
    Path("tests/python/conftest.py"),
    Path("openspec/capabilities/research-reporting/spec.md"),
    Path("research-roadmap.yaml"),
    Path("research-roadmap-next.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
)


def path_sha256(path: Path) -> str | None:
    """Return a worktree SHA-256 digest, or None when the file is absent."""

    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Return the stable checksum used inside the Exp6170 artifact."""

    normalized = json.loads(json.dumps(payload, sort_keys=True, default=str))
    normalized.pop("reproducibility_checksum", None)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _git(repo_root: Path, args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _git_or_empty(repo_root: Path, args: Sequence[str]) -> str:
    try:
        return _git(repo_root, args)
    except Exception:
        return ""


def _git_status_short(repo_root: Path) -> list[str]:
    return _git_or_empty(repo_root, ["status", "--short"]).splitlines()


def _status_paths(status_lines: Iterable[str]) -> list[str]:
    paths: list[str] = []
    for line in status_lines:
        if len(line) < 4:
            continue
        paths.append(line[3:])
    return paths


def _staged_paths(status_lines: Iterable[str]) -> set[str]:
    staged: set[str] = set()
    for line in status_lines:
        if len(line) >= 4 and line[0] not in (" ", "?"):
            staged.add(line[3:])
    return staged


def _tracked_results(repo_root: Path) -> list[Path]:
    out = _git_or_empty(repo_root, ["ls-files", "results"])
    if out:
        return [Path(row) for row in out.splitlines() if row]
    results = repo_root / "results"
    if not results.exists():
        return []
    return sorted(path.relative_to(repo_root) for path in results.rglob("*") if path.is_file())


def _aggregate_digest(repo_root: Path, paths: Sequence[Path]) -> JsonDict:
    digest = hashlib.sha256()
    for rel_path in sorted(paths, key=lambda p: p.as_posix()):
        file_digest = path_sha256(repo_root / rel_path) or "MISSING"
        digest.update(rel_path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("utf-8"))
        digest.update(b"\n")
    return {"count": len(paths), "sha256": digest.hexdigest()}


def _read_json_object(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_unreadable_or_non_json": type(exc).__name__}
    if not isinstance(payload, dict):
        return {"_non_object_json": type(payload).__name__}
    return payload


def _snapshot_quarantine_fields(repo_root: Path, rel_paths: Sequence[Path]) -> dict[str, JsonDict]:
    matrix: dict[str, JsonDict] = {}
    for rel_path in rel_paths:
        payload = _read_json_object(repo_root / rel_path)
        matrix[rel_path.as_posix()] = {
            field: payload[field] for field in QUARANTINE_FIELDS if field in payload
        }
        if "_unreadable_or_non_json" in payload or "_non_object_json" in payload:
            matrix[rel_path.as_posix()] = payload
    return matrix


def _snapshot_hashes(repo_root: Path, rel_paths: Sequence[Path]) -> dict[str, JsonDict]:
    return {
        rel_path.as_posix(): {
            "exists": (repo_root / rel_path).exists(),
            "sha256": path_sha256(repo_root / rel_path),
        }
        for rel_path in rel_paths
    }


def _sentinel_paths(repo_root: Path, precondition_snapshot: Mapping[str, Any] | None) -> list[Path]:
    candidates = list(SENTINEL_RESULT_PATHS)
    if precondition_snapshot:
        for raw in precondition_snapshot.get("sentinel_result_hashes", {}):
            candidates.append(Path(raw))
    seen: set[str] = set()
    result: list[Path] = []
    for rel_path in candidates:
        key = rel_path.as_posix()
        if key not in seen and (repo_root / rel_path).exists():
            seen.add(key)
            result.append(rel_path)
    return result


def snapshot_repository(
    repo_root: Path,
    *,
    precondition_snapshot: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Snapshot only the evidence Exp6170 is responsible for preserving."""

    tracked = _tracked_results(repo_root)
    sentinels = _sentinel_paths(repo_root, precondition_snapshot)
    return {
        "tracked_results": _aggregate_digest(repo_root, tracked),
        "sentinel_hashes": _snapshot_hashes(repo_root, sentinels),
        "quarantine_fields": _snapshot_quarantine_fields(repo_root, sentinels),
        "protected_files": _snapshot_hashes(repo_root, PROTECTED_FILE_PATHS),
    }


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


def _task_number(task_id: str) -> int | None:
    if not task_id.startswith("exp") or len(task_id) < 7:
        return None
    try:
        return int(task_id[3:7])
    except ValueError:
        return None


def _load_roadmap_tasks(repo_root: Path) -> list[JsonDict]:
    path = repo_root / "research-roadmap.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    tasks = payload.get("tasks", [])
    return [task for task in tasks if isinstance(task, dict)]


def declared_v535_tasks(repo_root: Path) -> list[JsonDict]:
    tasks: list[JsonDict] = []
    for task in _load_roadmap_tasks(repo_root):
        task_id = str(task.get("id", ""))
        task_num = _task_number(task_id)
        if task_num is None or not 6169 <= task_num <= 6182:
            continue
        tasks.append(
            {
                "task_id": task_id,
                "title": task.get("title"),
                "track": task.get("track"),
                "deliverable": str(task.get("deliverable", "")),
            }
        )
    order = {task_id: index for index, task_id in enumerate(V535_TASK_IDS)}
    return sorted(tasks, key=lambda row: order.get(str(row["task_id"]), 999))


def _expected_module_for_task(task_id: str, deliverable: str) -> Path:
    if task_id == "exp6169-v535-transition":
        return Path("python/carnot/experiment_6169_transition_v535.py")
    if task_id == "exp6170-v535-task-artifact-isolation-canary":
        return Path(CANARY_SOURCE_PATH)
    stem = Path(deliverable).stem
    return Path("python/carnot") / f"{stem}.py"


def _expected_test_for_module(module_path: Path) -> Path:
    return Path("tests/python") / f"test_{module_path.stem}.py"


def _writer_mechanism(repo_root: Path, module_path: Path, deliverable: str) -> str:
    path = repo_root / module_path
    if not path.exists():
        return "declared_deliverable_pending_module"
    text = path.read_text(encoding="utf-8", errors="replace")
    if "atomic_write_json" in text or "resolve_experiment_artifact_path" in text:
        return "canonical_artifact_writer"
    if deliverable in text and ("open(" in text or ".write_text(" in text):
        return "legacy_literal_writer"
    return "declared_module_without_detected_writer"


def collect_v535_task_writer_census(repo_root: Path) -> JsonDict:
    """Inventory only declared .535 task/test surfaces, not the whole repo."""

    rows: list[JsonDict] = []
    for task in declared_v535_tasks(repo_root):
        task_id = str(task["task_id"])
        deliverable = str(task["deliverable"])
        module_path = _expected_module_for_task(task_id, deliverable)
        rows.append(
            {
                "task_id": task_id,
                "title": task["title"],
                "track": task["track"],
                "surface": "experiment_module",
                "module": module_path.as_posix(),
                "module_present": (repo_root / module_path).exists(),
                "mechanism": _writer_mechanism(repo_root, module_path, deliverable),
                "exact_path": deliverable,
            }
        )
        test_path = _expected_test_for_module(module_path)
        if (repo_root / test_path).exists():
            rows.append(
                {
                    "task_id": task_id,
                    "title": task["title"],
                    "track": task["track"],
                    "surface": "test_module",
                    "module": test_path.as_posix(),
                    "module_present": True,
                    "mechanism": "pytest_task_scoped_canary_control",
                    "exact_path": deliverable,
                }
            )
    counts = Counter(row["mechanism"] for row in rows)
    return {
        "scope": {
            "first_task": V535_TASK_IDS[0],
            "last_task": V535_TASK_IDS[-1],
            "qualified_task_ids": list(V535_TASK_IDS),
            "repository_wide_writer_scan": False,
        },
        "declared_task_count": len(declared_v535_tasks(repo_root)),
        "row_count": len(rows),
        "mechanism_counts": dict(sorted(counts.items())),
        "rows": sorted(rows, key=lambda row: (row["task_id"], row["surface"], row["module"])),
    }


def build_frozen_invocation_manifest() -> JsonDict:
    return {
        "version": CANARY_CONTRACT_VERSION,
        "canary_module": CANARY_MODULE,
        "canary_source": CANARY_SOURCE_PATH,
        "pytest_target": CANARY_TEST_TARGET,
        "artifact_root_env": ARTIFACT_ROOT_ENV,
        "temporary_root_contract": (
            "Set CARNOT_EXPERIMENT_ARTIFACT_ROOT to a validated task-owned directory "
            "under the system temp root before pytest collection."
        ),
        "repository_wide_closure_claimed": False,
        "commands": {
            "collection": f".venv/bin/pytest {CANARY_TEST_TARGET} --collect-only -q -o addopts=",
            "focused": f".venv/bin/pytest {CANARY_TEST_TARGET} -q -o addopts=",
            "new_code_coverage_run": (
                ".venv/bin/coverage run --source=python/carnot -m pytest "
                f"{CANARY_TEST_TARGET} -q -o addopts="
            ),
            "new_code_coverage_report": (
                ".venv/bin/coverage report "
                f"--include='{CANARY_SOURCE_PATH}' --fail-under=100 --show-missing"
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
        return Path(tempfile.mkdtemp(prefix="carnot-6170-canary-artifacts-")).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    return raw_root.resolve()


def _raises_artifact_error(fn) -> JsonDict:
    try:
        fn()
    except ArtifactPathError as exc:
        return {"raised": True, "exception": type(exc).__name__, "message": str(exc)}
    return {"raised": False, "exception": None, "message": "no exception"}


def _choose_tracked_sentinel(repo_root: Path) -> Path | None:
    for rel_path in SENTINEL_RESULT_PATHS:
        path = repo_root / rel_path
        if path.exists():
            return path
    tracked = _tracked_results(repo_root)
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
            "results/experiment_6170_subprocess_canonical.json",
            {"status": "subprocess"},
        )
        with open("results/experiment_6170_subprocess_legacy.txt", "w", encoding="utf-8") as fh:
            fh.write("legacy subprocess\\n")
        print(json.dumps({
            "canonical": str(canonical),
            "legacy": str(Path(os.environ["CARNOT_EXPERIMENT_ARTIFACT_ROOT"]) / "experiment_6170_subprocess_legacy.txt"),
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
    parsed: JsonDict | None = None
    if proc.returncode == 0 and proc.stdout.strip():
        parsed = json.loads(proc.stdout)
    return {
        "command": f"{sys.executable} -c <subprocess canary>",
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "artifact_root": str(sub_root),
        "parsed": parsed,
        "canonical_exists": (sub_root / "experiment_6170_subprocess_canonical.json").exists(),
        "legacy_exists": (sub_root / "experiment_6170_subprocess_legacy.txt").exists(),
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
        _snapshot_quarantine_fields(repo_root, [sentinel.relative_to(repo_root)])
        if sentinel is not None
        else {}
    )

    with _temporary_artifact_env(repo_root, task_root):
        canonical_path = atomic_write_json(
            "results/experiment_6170_canary_canonical.json", {"status": "canonical"}
        )
        with open("results/experiment_6170_canary_legacy.txt", "w", encoding="utf-8") as fh:
            fh.write("legacy literal\n")
        Path("results/experiment_6170_canary_replace.tmp").write_text("atomic\n", encoding="utf-8")
        os.replace(
            "results/experiment_6170_canary_replace.tmp",
            "results/experiment_6170_canary_replace.txt",
        )

        from scripts.experiment_template import ExperimentTemplate

        template = ExperimentTemplate(
            6170,
            "V535 artifact isolation canary",
            "results/experiment_6170_template_canary.json",
        )
        template.checkpoint_save({"status": "checkpoint"}, step=7)
        checkpoint_payload = template.checkpoint_resume() or {}

        atomic_path = atomic_write_json("results/experiment_6170_canary_atomic.json", {"value": 1})
        atomic_write_json("results/experiment_6170_canary_atomic.json", {"value": 2})
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
            lambda: resolve_experiment_artifact_path("results/../experiment_6170_escape.json")
        )
        symlink_escape = _raises_artifact_error(
            lambda: resolve_experiment_artifact_path("results/escape/leak.json")
        )

    attempted: JsonDict
    if sentinel is None:  # pragma: no cover - this scoped repo canary has tracked sentinels.
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
        else:  # pragma: no cover - reaching this branch would mean the guard mutated evidence.
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
        _snapshot_quarantine_fields(repo_root, [sentinel.relative_to(repo_root)])
        if sentinel is not None
        else {}
    )
    redirects = tracked_results_guard.recorded_legacy_compat_redirects()
    tracked_results_guard.clear_legacy_compat_redirects()

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
            "requested": "results/experiment_6170_canary_canonical.json",
            "resolved": str(canonical_path),
            "under_task_root": canonical_path.is_relative_to(task_root),
        },
        "legacy_literal_writer": {
            "requested": "results/experiment_6170_canary_legacy.txt",
            "resolved": str(task_root / "experiment_6170_canary_legacy.txt"),
            "under_task_root": (task_root / "experiment_6170_canary_legacy.txt").exists(),
            "redirect_events": redirects,
        },
        "legacy_atomic_replace": {
            "requested": "results/experiment_6170_canary_replace.txt",
            "resolved": str(task_root / "experiment_6170_canary_replace.txt"),
            "under_task_root": (task_root / "experiment_6170_canary_replace.txt").exists(),
        },
        "checkpoint_resume": {
            "checkpoint_path": str(
                task_root / "checkpoints" / "experiment_6170" / "checkpoint.json"
            ),
            "resumed_step": checkpoint_payload.get("step"),
            "resumed_status": (checkpoint_payload.get("results") or {}).get("status"),
        },
        "atomic_writer": {
            "requested": "results/experiment_6170_canary_atomic.json",
            "resolved": str(atomic_path),
            "final_value": atomic_payload.get("value"),
            "leftover_tmp_files": sorted(path.name for path in task_root.glob("*.tmp")),
        },
        "subprocess": subprocess_receipt,
        "attempted_tracked_write": attempted,
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
            "canary_scope": 0,
            "unrelated_preexisting": 0,
            "new_regression": 0,
            "unclassified": 0,
        }
    )
    for receipt in command_receipts:
        exit_code = int(receipt.get("exit_code", 0))
        text = " ".join(
            str(receipt.get(key, "")) for key in ("name", "command", "stdout", "stderr")
        )
        if exit_code == 0:
            classification = "zero"
        elif receipt.get("classification") in {
            "canary_scope",
            "unrelated_preexisting",
            "new_regression",
            "unclassified",
        }:
            classification = str(receipt["classification"])
        elif (
            "experiment_6170_v535_task_artifact_isolation_canary" in text
            or "TrackedResultWriteError" in text
            or "tracked result evidence" in text
        ):
            classification = "canary_scope"
        elif "ModuleNotFoundError" in text or "legacy_fixture" in text:
            classification = "unrelated_preexisting"
        elif "new_regression" in text:
            classification = "new_regression"
        else:
            classification = "unclassified"
        counts[classification] += 1
        row = dict(receipt)
        row["classification"] = classification
        rows.append(row)
    return {"rows": rows, "counts": dict(counts)}


def ready_score(
    *,
    tracked_mutation_count: int,
    classification: Mapping[str, Any],
    invocation_manifest: Mapping[str, Any],
    command_receipts: Sequence[Mapping[str, Any]],
) -> int:
    counts = classification.get("counts", {})
    if not command_receipts:
        return 0
    if tracked_mutation_count != 0:
        return 0
    if counts.get("canary_scope", 0) or counts.get("unclassified", 0):
        return 0
    if invocation_manifest.get("repository_wide_closure_claimed") is not False:
        return 0
    if invocation_manifest.get("version") != CANARY_CONTRACT_VERSION:
        return 0
    return 1


def _matrix_unchanged(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    return before == after


def _hash_matrix(before: Mapping[str, Any], after: Mapping[str, Any]) -> JsonDict:
    return {"before": before, "after": after, "unchanged": _matrix_unchanged(before, after)}


def _task_start_matrix_from_precondition(
    precondition_snapshot: Mapping[str, Any],
    after: Mapping[str, Any],
) -> JsonDict:
    if not precondition_snapshot.get("present"):
        return {"available": False, "unchanged": None, "reason": "precondition snapshot missing"}
    before = {
        "tracked_results": {
            "count": precondition_snapshot.get("tracked_results_count"),
            "sha256": precondition_snapshot.get("tracked_results_aggregate_sha256"),
        },
        "sentinel_hashes": {
            path: {"exists": True, "sha256": digest}
            for path, digest in dict(
                precondition_snapshot.get("sentinel_result_hashes", {})
            ).items()
        },
        "quarantine_fields": dict(precondition_snapshot.get("quarantine_fields", {})),
    }
    after_sentinels = {
        path: after["sentinel_hashes"].get(path)
        for path in before["sentinel_hashes"]
        if path in after["sentinel_hashes"]
    }
    after_quarantine = {
        path: after["quarantine_fields"].get(path)
        for path in before["quarantine_fields"]
        if path in after["quarantine_fields"]
    }
    after_view = {
        "tracked_results": after["tracked_results"],
        "sentinel_hashes": after_sentinels,
        "quarantine_fields": after_quarantine,
    }
    return {"available": True, **_hash_matrix(before, after_view)}


def _preexisting_changes(
    repo_root: Path,
    precondition_snapshot: Mapping[str, Any],
) -> JsonDict:
    initial_status = list(precondition_snapshot.get("git_status_short", []))
    current_status = _git_status_short(repo_root)
    initial_staged = _staged_paths(initial_status)
    current_staged = _staged_paths(current_status)
    initial_paths = set(_status_paths(initial_status))
    current_paths = set(_status_paths(current_status))
    return {
        "precondition_snapshot_path": precondition_snapshot.get("path"),
        "task_start_git_status_short": initial_status,
        "current_git_status_short": current_status,
        "task_start_modified_paths": sorted(initial_paths),
        "newly_staged_paths": sorted(current_staged - initial_staged),
        "preexisting_paths_still_present_or_existing": sorted(
            path for path in initial_paths if path in current_paths or (repo_root / path).exists()
        ),
        "no_revert_or_restore_performed": True,
    }


def _field_provenance() -> JsonDict:
    return {
        field: {
            "spec": "REQ-REPORT-6170",
            "principle": FIELD_PRINCIPLES[field],
            "source": CANARY_SOURCE_PATH,
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
    census = collect_v535_task_writer_census(repo_root)
    invocation_manifest = build_frozen_invocation_manifest()
    classification = classify_command_receipts(command_receipts)

    tracked_hash_matrix = _hash_matrix(before["tracked_results"], after["tracked_results"])
    sentinel_hash_matrix = _hash_matrix(before["sentinel_hashes"], after["sentinel_hashes"])
    quarantine_matrix = _hash_matrix(before["quarantine_fields"], after["quarantine_fields"])
    task_start_matrix = _task_start_matrix_from_precondition(precondition_snapshot, after)
    protected_matrix = _hash_matrix(before["protected_files"], after["protected_files"])
    task_start_unchanged = task_start_matrix.get("unchanged")
    tracked_mutation_count = (
        0
        if tracked_hash_matrix["unchanged"]
        and sentinel_hash_matrix["unchanged"]
        and task_start_unchanged is not False
        else 1
    )
    canary_failures = int(classification["counts"].get("canary_scope", 0))
    isolation_violation_count = tracked_mutation_count + canary_failures
    ready = ready_score(
        tracked_mutation_count=tracked_mutation_count,
        classification=classification,
        invocation_manifest=invocation_manifest,
        command_receipts=command_receipts,
    )
    commands, exit_codes = _command_maps(command_receipts)
    actual_duration = round(time.perf_counter() - started, 3) if duration_s is None else duration_s
    status = "complete_ready" if ready else "complete_partial"
    verdict_prefix = "complete_ready:" if ready else "complete_partial:"
    artifact: JsonDict = {
        "status": status,
        "preconditions_checked": {
            "agents_codex_claude_and_task_files_read": True,
            "precondition_snapshot": precondition_snapshot,
            "research_roadmap_next_present": (repo_root / "research-roadmap-next.yaml").exists(),
            "resolver_environment_before_controls": {
                ARTIFACT_ROOT_ENV: os.environ.get(ARTIFACT_ROOT_ENV),
                "PYTEST_CURRENT_TEST": os.environ.get("PYTEST_CURRENT_TEST"),
            },
            "declared_v535_deliverables": declared_v535_tasks(repo_root),
            "candidate_test_modules": sorted(
                row["module"] for row in census["rows"] if row["surface"] == "test_module"
            ),
            "symlink_traversal_boundaries": {
                "repo_root": str(repo_root),
                "production_results_root": str((repo_root / "results").resolve()),
                "workspace_root": str(repo_root),
                "broad_tmp_root": str(Path(tempfile.gettempdir()).resolve()),
            },
        },
        "scope_boundary_and_repository_wide_closure_claimed": {
            "qualified_scope": "Exp6169-Exp6182 declared .535 writer/test surfaces only",
            "qualified_task_ids": list(V535_TASK_IDS),
            "repository_wide_closure_claimed": False,
        },
        "v535_task_writer_census": census,
        "frozen_canary_module_and_invocation_manifest": invocation_manifest,
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
        "attempted_tracked_write_controls": controls["attempted_tracked_write"],
        "traversal_symlink_workspace_root_and_atomic_controls": {
            "traversal": controls["traversal"],
            "symlink_escape": controls["symlink_escape"],
            "invalid_roots": controls["invalid_roots"],
            "atomic_writer": controls["atomic_writer"],
        },
        "tracked_result_hash_before_after_matrix": {
            "canary_control_window": {
                "aggregate": tracked_hash_matrix,
                "sentinels": sentinel_hash_matrix,
            },
            "task_start_to_post": task_start_matrix,
        },
        "quarantine_field_before_after_matrix": {
            "canary_control_window": quarantine_matrix,
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
        "canary_failure_classification": classification,
        "isolation_violation_count": isolation_violation_count,
        "v535_task_artifact_isolation_ready_score": ready,
        "preexisting_worktree_changes_preserved": _preexisting_changes(
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
            f"{verdict_prefix} Exp6169-Exp6182 declared .535 canary surface "
            f"ready_score={ready}, isolation_violation_count={isolation_violation_count}, "
            "repository_wide_closure_claimed=false."
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"Exp6170 artifact validation failed: {errors}")
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            errors.append(f"missing:{field}")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    scope = payload.get("scope_boundary_and_repository_wide_closure_claimed")
    if not isinstance(scope, dict) or scope.get("repository_wide_closure_claimed") is not False:
        errors.append("repository_wide_closure_claimed")
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
    ready = int(payload.get("v535_task_artifact_isolation_ready_score", 0))
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
    parser = argparse.ArgumentParser(description="Run Exp6170 .535 artifact-isolation canary.")
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
