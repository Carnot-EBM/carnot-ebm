"""Exp6298 terminal-evidence preflight linter.

Spec refs: REQ-INFRA-6298, SCENARIO-INFRA-6298-1,
SCENARIO-INFRA-6298-2, SCENARIO-INFRA-6298-3,
SCENARIO-INFRA-6298-4, SCENARIO-INFRA-6298-5,
SCENARIO-INFRA-6298-6.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import json
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any

from carnot.experiment_artifacts import atomic_write_json
from carnot.terminal_artifacts import canonical_json, path_sha256, payload_sha256
from carnot.terminal_evidence_preflight import (
    ARTIFACT_QA_LINT_TESTS_SUBSTRATE,
    DEFAULT_GATE_FIELDS,
    FAILURE_TAXONOMY,
    FOCUSED_TEST_COMMAND,
    V542_FAILURE_FIXTURES,
    build_synthetic_fixture_manifest,
    evaluate_fixture_manifest,
    replay_v542_failure_fixtures,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "exp6298-terminal-evidence-preflight-linter"
SCHEMA = "carnot.experiment_6298.terminal_evidence_preflight_linter.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6298_terminal_evidence_preflight_linter.json")
SYNTHETIC_FIXTURE_MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_6298_terminal_evidence_preflight_synthetic_fixtures.json"
)
INFERENCE_SUBSTRATE = ARTIFACT_QA_LINT_TESTS_SUBSTRATE
RANDOM_SEED = 6298

SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("results/experiment_6288_partial_atom_evidence_adapter.json"),
    Path("results/experiment_6289_flagship_exact_state_refinement_benchmark.json"),
    Path("results/experiment_6290_revocable_atomic_repair_memory.json"),
    Path("results/experiment_6296_v542_adversarial_capstone.json"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/conductor_gates.py"),
    Path("python/carnot/terminal_artifacts.py"),
    Path("python/carnot/terminal_evidence_preflight.py"),
    Path("python/carnot/experiment_6298_terminal_evidence_preflight_linter.py"),
    Path("scripts/determination_preservation_lint.py"),
    Path("openspec/capabilities/research-harnesses/spec.md"),
    Path("ops/e2e-test-plan.md"),
    Path("tests/python/test_experiment_6298_terminal_evidence_preflight_linter.py"),
)
PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("results/experiment_6288_partial_atom_evidence_adapter.json"),
    Path("results/experiment_6289_flagship_exact_state_refinement_benchmark.json"),
    Path("results/experiment_6290_revocable_atomic_repair_memory.json"),
    Path("results/experiment_6296_v542_adversarial_capstone.json"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/conductor_gates.py"),
    Path("python/carnot/terminal_artifacts.py"),
    Path("scripts/determination_preservation_lint.py"),
    Path("ops/e2e-test-plan.md"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "failure_taxonomy",
    "source_paths_and_hashes",
    "v542_fixture_paths_hashes_and_expected_classes",
    "synthetic_fixture_manifest_path_and_hash",
    "required_field_checks",
    "terminal_prefix_checks",
    "field_principle_coverage_checks",
    "substrate_duration_and_methodology_checks",
    "test_command_and_exit_code_checks",
    "gate_field_type_checks",
    "determination_preservation_checks",
    "clean_fixture_accept_count",
    "bad_fixture_reject_count",
    "false_accept_count",
    "false_reject_count",
    "cli_contract",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "terminal_evidence_preflight_ready_score",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "The report closes only after fixture replay and protected hashes pass.",
    "failure_taxonomy": "Stable class names make each rejection auditable.",
    "source_paths_and_hashes": "Source hashes bind the linter to the exact code and inputs.",
    "v542_fixture_paths_hashes_and_expected_classes": "Prior failures stay immutable inputs.",
    "synthetic_fixture_manifest_path_and_hash": "Synthetic fixtures need a stable manifest.",
    "required_field_checks": "Missing fields are the cheapest fabrication signal.",
    "terminal_prefix_checks": "A terminal prefix prevents partial verdict laundering.",
    "field_principle_coverage_checks": "Each required field must carry the reason it exists.",
    "substrate_duration_and_methodology_checks": "Compute claims need plausible time and receipts.",
    "test_command_and_exit_code_checks": "Recorded verification failures must stay visible.",
    "gate_field_type_checks": "Gates must read exact bare values of the expected type.",
    "determination_preservation_checks": "Review determinations must not silently disappear.",
    "clean_fixture_accept_count": "The linter must accept known-good terminal evidence.",
    "bad_fixture_reject_count": "The linter must reject known-bad terminal evidence.",
    "false_accept_count": "Bare zero proves bad fixtures did not pass.",
    "false_reject_count": "Bare zero proves the clean fixture did not fail.",
    "cli_contract": "The CLI must be reproducible and separate from the conductor.",
    "protected_files_unchanged": "Protected hashes prove this run did not rewrite inputs.",
    "preconditions_checked": "The run records git state, hashes, and fixture setup first.",
    "inference_substrate": "This task runs artifact QA lint tests, not model inference.",
    "verifier_is_oracle": "The linter checks evidence quality and is not an answer oracle.",
    "field_provenance": "Every required field cites the evidence source that produced it.",
    "field_principles": "Every required field states the guard principle it serves.",
    "test_commands": "The report lists the bounded project-owned verification commands.",
    "test_exit_codes": "The report preserves exit codes without converting failures to passes.",
    "duration_s": "Wall time records the QA cost without padding.",
    "random_seed": "A fixed seed makes the synthetic fixture manifest reproducible.",
    "reproducibility_checksum": "A normalized checksum detects silent payload drift.",
    "terminal_evidence_preflight_ready_score": "The score is one only with zero false counts.",
    "honest_verdict": "The verdict states the terminal result with a safe prefix.",
}

COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null --branch "
    "--include=python/carnot/terminal_evidence_preflight.py,"
    "python/carnot/experiment_6298_terminal_evidence_preflight_linter.py "
    "-m pytest "
    "tests/python/test_experiment_6298_terminal_evidence_preflight_linter.py "
    "-q --no-cov -n 0"
)
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_COMMAND,
    (
        ".venv/bin/coverage report --rcfile=/dev/null "
        "--include=python/carnot/terminal_evidence_preflight.py,"
        "python/carnot/experiment_6298_terminal_evidence_preflight_linter.py "
        "--fail-under=100 --show-missing"
    ),
    (
        ".venv/bin/ruff check python/carnot/terminal_evidence_preflight.py "
        "python/carnot/experiment_6298_terminal_evidence_preflight_linter.py "
        "tests/python/test_experiment_6298_terminal_evidence_preflight_linter.py"
    ),
    (
        ".venv/bin/ruff format --check python/carnot/terminal_evidence_preflight.py "
        "python/carnot/experiment_6298_terminal_evidence_preflight_linter.py "
        "tests/python/test_experiment_6298_terminal_evidence_preflight_linter.py"
    ),
    (
        ".venv/bin/python scripts/check_spec_coverage.py "
        "tests/python/test_experiment_6298_terminal_evidence_preflight_linter.py"
    ),
    ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml",
    ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml",
    ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    ".venv/bin/python scripts/determination_preservation_lint.py --all",
    "sed -n 1,220p ops/e2e-test-plan.md",
    ".venv/bin/pytest tests/python -q",
    f".venv/bin/python scripts/adversarial_verify.py {RESULT_RELATIVE_PATH.as_posix()}",
)
COMMAND_TIMEOUTS_S = {
    ".venv/bin/pytest tests/python -q": 3600,
    COVERAGE_COMMAND: 600,
}


def payload_checksum(report: JsonMap) -> str:
    normalized = json.loads(canonical_json(report))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return payload_sha256(normalized)


def git_status_lines(root: Path) -> list[str]:  # pragma: no cover - shell edge.
    proc = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return [f"git_status_failed:{proc.returncode}:{proc.stderr.strip()}"]
    return [line for line in proc.stdout.splitlines() if line.strip()]


def hash_paths(root: Path, paths: Sequence[Path]) -> JsonDict:
    return {
        path.as_posix(): {"present": (root / path).exists(), "sha256": path_sha256(root / path)}
        for path in paths
    }


def protected_hashes(root: Path, paths: Sequence[Path] = PROTECTED_RELATIVE_PATHS) -> JsonDict:
    return {path.as_posix(): path_sha256(root / path) for path in paths}


def protected_files_unchanged(
    root: Path,
    before: JsonMap,
    paths: Sequence[Path] = PROTECTED_RELATIVE_PATHS,
) -> JsonDict:
    after = protected_hashes(root, paths)
    rows = {
        path: {
            "before_sha256": before.get(path),
            "after_sha256": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {"unchanged": all(row["unchanged"] for row in rows.values()), "paths": rows}


def _field_provenance() -> JsonDict:
    sources = sorted(path.as_posix() for path in SOURCE_RELATIVE_PATHS)
    return {
        field: {"principle": FIELD_PRINCIPLES[field], "sources": sources}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _collect_check(rows: Sequence[JsonMap], key: str) -> JsonDict:
    return {str(row.get("fixture_id") or ""): row.get(key) for row in rows}


def _v542_manifest(rows: Sequence[JsonMap]) -> list[JsonDict]:
    expected_by_id = {str(row["fixture_id"]): row for row in V542_FAILURE_FIXTURES}
    out: list[JsonDict] = []
    for row in rows:
        fixture_id = str(row.get("fixture_id") or "")
        expected = expected_by_id.get(fixture_id, {})
        out.append(
            {
                "fixture_id": fixture_id,
                "path": row.get("path"),
                "path_sha256": row.get("path_sha256"),
                "expected_accept": expected.get("expected_accept"),
                "expected_failure_classes": expected.get("expected_failure_classes", []),
                "observed_accept": row.get("accepted"),
                "observed_failure_classes": row.get("failure_classes", []),
            }
        )
    return out


def _failure_taxonomy(rows: Sequence[JsonMap]) -> JsonDict:
    counts = Counter(cls for row in rows for cls in row.get("failure_classes", []))
    return {
        "definitions": dict(FAILURE_TAXONOMY),
        "observed_counts": dict(sorted(counts.items())),
        "taxonomy_is_closed_for_report": True,
    }


def write_synthetic_fixture_manifest(
    manifest: JsonMap,
    root: Path = REPO_ROOT,
    *,
    env: Mapping[str, str] | None = None,
) -> Path:
    return atomic_write_json(
        SYNTHETIC_FIXTURE_MANIFEST_RELATIVE_PATH,
        manifest,
        root=root,
        env=env,
        sort_keys=False,
    )


def _manifest_path_hash(file_sha256: str | None, manifest: JsonMap) -> JsonDict:
    fixtures = manifest.get("fixtures")
    fixture_rows = fixtures if isinstance(fixtures, list) else []
    fixture_ids = [
        str(row.get("fixture_id") or "") for row in fixture_rows if isinstance(row, Mapping)
    ]
    return {
        "path": SYNTHETIC_FIXTURE_MANIFEST_RELATIVE_PATH.as_posix(),
        "file_sha256": file_sha256,
        "payload_sha256": payload_sha256(manifest),
        "fixture_ids": fixture_ids,
        "gate_fields": list(DEFAULT_GATE_FIELDS),
    }


def _test_exits(command_rows: Sequence[JsonMap]) -> JsonDict:
    return {
        str(row.get("command") or ""): int(row.get("exit_code") or 0)
        for row in command_rows
        if row.get("command")
    }


def _any_command_failed(command_rows: Sequence[JsonMap]) -> bool:
    return any(int(row.get("exit_code") or 0) != 0 for row in command_rows)


def build_report(
    root: Path = REPO_ROOT,
    *,
    date: str,
    command_receipts: Sequence[JsonMap] | None = None,
    before_hashes: JsonMap | None = None,
    synthetic_manifest_file_sha256: str | None = None,
    git_status_before: Sequence[str] | None = None,
    git_status_after_tests: Sequence[str] | None = None,
    started_at: float | None = None,
) -> JsonDict:
    started = time.perf_counter() if started_at is None else started_at
    before = dict(protected_hashes(root) if before_hashes is None else before_hashes)
    status_before = list(git_status_lines(root) if git_status_before is None else git_status_before)
    manifest = build_synthetic_fixture_manifest()
    synthetic = evaluate_fixture_manifest(manifest)
    synthetic_rows = [dict(row) for row in synthetic["fixture_results"]]
    v542_rows = replay_v542_failure_fixtures(root)
    all_rows = [*v542_rows, *synthetic_rows]
    false_accept_count = sum(
        1 for row in all_rows if row.get("accepted") is True and row.get("expected_accept") is False
    )
    false_reject_count = sum(
        1 for row in all_rows if row.get("accepted") is False and row.get("expected_accept") is True
    )
    protected = protected_files_unchanged(root, before)
    command_rows = [dict(row) for row in (command_receipts or [])]
    command_failure = _any_command_failed(command_rows)
    ready = (
        synthetic["clean_fixture_accept_count"] == 1
        and false_accept_count == 0
        and false_reject_count == 0
        and protected["unchanged"] is True
        and not command_failure
    )
    report: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "status": "complete" if ready else "blocked",
        "failure_taxonomy": _failure_taxonomy(all_rows),
        "source_paths_and_hashes": hash_paths(root, SOURCE_RELATIVE_PATHS),
        "v542_fixture_paths_hashes_and_expected_classes": _v542_manifest(v542_rows),
        "synthetic_fixture_manifest_path_and_hash": _manifest_path_hash(
            synthetic_manifest_file_sha256, manifest
        ),
        "required_field_checks": _collect_check(all_rows, "required_field_check"),
        "terminal_prefix_checks": _collect_check(all_rows, "terminal_prefix_check"),
        "field_principle_coverage_checks": _collect_check(
            all_rows, "field_principle_coverage_check"
        ),
        "substrate_duration_and_methodology_checks": _collect_check(
            all_rows, "substrate_duration_and_methodology_check"
        ),
        "test_command_and_exit_code_checks": _collect_check(
            all_rows, "test_command_and_exit_code_check"
        ),
        "gate_field_type_checks": _collect_check(all_rows, "gate_field_type_check"),
        "determination_preservation_checks": _collect_check(
            all_rows, "determination_preservation_check"
        ),
        "clean_fixture_accept_count": int(synthetic["clean_fixture_accept_count"]),
        "bad_fixture_reject_count": sum(
            1
            for row in all_rows
            if row.get("expected_accept") is False and row.get("accepted") is False
        ),
        "false_accept_count": int(false_accept_count),
        "false_reject_count": int(false_reject_count),
        "cli_contract": {
            "run_command": (
                ".venv/bin/python -m "
                "carnot.experiment_6298_terminal_evidence_preflight_linter --date YYYYMMDD"
            ),
            "actual_date_arg": date,
            "standalone_from_conductor": True,
            "does_not_modify_research_conductor": True,
            "does_not_execute_artifact_supplied_commands": True,
            "default_gate_fields": list(DEFAULT_GATE_FIELDS),
        },
        "protected_files_unchanged": protected,
        "preconditions_checked": {
            "git_status_before": status_before,
            "git_status_after_tests": list(git_status_after_tests or []),
            "protected_hashes_before_artifact_write": before,
            "prior_artifact_hashes_frozen": {row["path"]: row["path_sha256"] for row in v542_rows},
            "synthetic_manifest_payload_sha256": payload_sha256(manifest),
            "source_hashes_frozen": hash_paths(root, SOURCE_RELATIVE_PATHS),
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": [str(row.get("command") or "") for row in command_rows]
        or list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": _test_exits(command_rows),
        "duration_s": time.perf_counter() - started,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "terminal_evidence_preflight_ready_score": 1.0 if ready else 0.0,
        "honest_verdict": (
            "complete: terminal evidence preflight rejected V542 and synthetic bad fixtures "
            "with zero false accepts and zero false rejects"
            if ready
            else "blocked: terminal evidence preflight found a false count, command failure, "
            "or protected-file hash change"
        ),
    }
    report["reproducibility_checksum"] = payload_checksum(report)
    return report


def validate_report(report: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in report:
            errors.append(f"missing required field: {field}")
    principles = report.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles is not a mapping")
        principles = {}
    provenance = report.get("field_provenance")
    if not isinstance(provenance, Mapping):
        errors.append("field_provenance is not a mapping")
        provenance = {}
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in principles:
            errors.append(f"missing field_principles entry: {field}")
        if field not in provenance:
            errors.append(f"missing field_provenance entry: {field}")
    if report.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("wrong inference_substrate")
    if report.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if not (type(report.get("false_accept_count")) is int and report["false_accept_count"] == 0):
        errors.append("false_accept_count must be bare integer 0")
    if not (type(report.get("false_reject_count")) is int and report["false_reject_count"] == 0):
        errors.append("false_reject_count must be bare integer 0")
    if report.get("terminal_evidence_preflight_ready_score") != 1.0:
        errors.append("terminal_evidence_preflight_ready_score must be 1.0")
    protected = report.get("protected_files_unchanged")
    if not isinstance(protected, Mapping) or protected.get("unchanged") is not True:
        errors.append("protected_files_unchanged must be true")
    verdict = str(report.get("honest_verdict") or "")
    if not verdict.startswith(("complete:", "success:", "passed:", "shipped:")):
        errors.append("honest_verdict lacks accepted Exp6298 prefix")
    checksum = report.get("reproducibility_checksum")
    if isinstance(checksum, str) and checksum.startswith("sha256:"):
        if checksum != payload_checksum(report):
            errors.append("reproducibility_checksum mismatch")
    else:
        errors.append("reproducibility_checksum missing")
    return errors


def run_command(
    command: str, root: Path, timeout_s: int | None = None
) -> JsonDict:  # pragma: no cover - shell wrapper.
    try:
        proc = subprocess.run(
            shlex.split(command),
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "exit_code": 124,
            "classification": "timeout",
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
        }
    except FileNotFoundError as exc:
        return {
            "command": command,
            "exit_code": 127,
            "classification": "command_not_found",
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }
    return {
        "command": command,
        "exit_code": proc.returncode,
        "classification": "passed" if proc.returncode == 0 else f"nonzero_exit_{proc.returncode}",
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def run_default_commands(root: Path) -> list[JsonDict]:  # pragma: no cover - shell wrapper.
    return [
        run_command(command, root, COMMAND_TIMEOUTS_S.get(command))
        for command in DEFAULT_TEST_COMMANDS
    ]


def write_report(
    report: JsonMap,
    root: Path = REPO_ROOT,
    *,
    env: Mapping[str, str] | None = None,
) -> Path:
    errors = validate_report(report)
    if errors:
        raise ValueError("invalid Exp6298 report: " + "; ".join(errors))
    return atomic_write_json(RESULT_RELATIVE_PATH, report, root=root, env=env, sort_keys=False)


def run_experiment(
    root: Path,
    date: str,
    *,
    run_commands: bool,
) -> JsonDict:  # pragma: no cover - shell wrapper.
    started = time.perf_counter()
    before = protected_hashes(root)
    status_before = git_status_lines(root)
    manifest = build_synthetic_fixture_manifest()
    manifest_path = write_synthetic_fixture_manifest(manifest, root)
    manifest_hash = path_sha256(manifest_path)
    preliminary = build_report(
        root,
        date=date,
        command_receipts=[],
        before_hashes=before,
        synthetic_manifest_file_sha256=manifest_hash,
        git_status_before=status_before,
        started_at=started,
    )
    write_report(preliminary, root)
    commands = run_default_commands(root) if run_commands else []
    final = build_report(
        root,
        date=date,
        command_receipts=commands,
        before_hashes=before,
        synthetic_manifest_file_sha256=manifest_hash,
        git_status_before=status_before,
        git_status_after_tests=git_status_lines(root),
        started_at=started,
    )
    write_report(final, root)
    return final


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now(UTC).strftime("%Y%m%d"))
    parser.add_argument("--no-run-commands", action="store_true")
    args = parser.parse_args(argv)
    report = run_experiment(REPO_ROOT, args.date, run_commands=not args.no_run_commands)
    print(json.dumps(report, indent=2, sort_keys=False))
    return 0 if report["status"] in {"complete", "blocked"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
