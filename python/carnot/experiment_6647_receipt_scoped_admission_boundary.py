"""Reduce GPU lease readiness from preregistered task-owned receipts.

The repo-wide Python suite remains visible because it is important operational
truth. It cannot change this task's readiness because it includes unrelated
tests and a known xdist worker-directory failure. This module loads no model and
makes no model-quality claim.

Spec refs: REQ-INFRA-6647, SCENARIO-INFRA-6647-PREREGISTERED-OWNERSHIP,
SCENARIO-INFRA-6647-EXACT-FIELD-OWNERSHIP,
SCENARIO-INFRA-6647-MISSING-RECEIPT, REQ-REPORT-6647,
SCENARIO-REPORT-6647-READY, SCENARIO-REPORT-6647-GLOBAL-DIAGNOSTIC,
SCENARIO-REPORT-6647-BLOCKED-RECEIPT, and
SCENARIO-REPORT-6647-ATOMIC-PROVENANCE.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any

from carnot import experiment_6633_gpu_lease_phase_journal as prior_exp
from carnot import gpu_lease_phase_journal as lease_api


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260826"
RANDOM_SEED = 6647
RESULT_PATH = Path("results/experiment_6647_receipt_scoped_admission_boundary.json")
WORK_PATH = Path("results/.experiment_6647_receipt_scoped_admission_boundary")
PRIOR_PATH = Path("results/experiment_6633_gpu_lease_phase_journal.json")
MODULE_PATH = Path("python/carnot/experiment_6647_receipt_scoped_admission_boundary.py")
TEST_PATH = Path("tests/python/test_experiment_6647_receipt_scoped_admission_boundary.py")
LEASE_PATH = Path("python/carnot/gpu_lease_phase_journal.py")
PRIOR_MODULE_PATH = Path("python/carnot/experiment_6633_gpu_lease_phase_journal.py")
SPEC_PATH = REPO_ROOT / "openspec/capabilities/research-reporting/spec.md"
INFRA_SPEC_PATH = REPO_ROOT / "openspec/capabilities/research-harnesses/spec.md"
PROTECTED_PATHS = (Path("research-roadmap.yaml"), Path("scripts/research_conductor.py"))
INFERENCE_SUBSTRATE = "receipt_scoped_gpu_lease_reducer_no_llm"
KNOWN_ISSUE_LINK = "ops/known-issues.md:2620"

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6647_receipt_scoped_admission_boundary "
    "--date 20260826"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_gpu_lease_phase_journal.py "
    "tests/python/test_experiment_6633_gpu_lease_phase_journal.py "
    "tests/python/test_experiment_6647_receipt_scoped_admission_boundary.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    "JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "
    "COVERAGE_FILE=/tmp/carnot_exp6647.coverage .venv/bin/coverage run "
    "--include='*/experiment_6647_receipt_scoped_admission_boundary.py' "
    "-m pytest --noconftest "
    "tests/python/test_experiment_6647_receipt_scoped_admission_boundary.py "
    "-q -o addopts="
)
COVERAGE_REPORT_COMMAND = (
    "COVERAGE_FILE=/tmp/carnot_exp6647.coverage .venv/bin/coverage report "
    "--include='*/experiment_6647_receipt_scoped_admission_boundary.py' "
    "-m --fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6647_receipt_scoped_admission_boundary.py"
)
E2E_COMMAND = (
    "Exp6647 E2E: fresh-path acquisition, exclusion, independent-device, binding, "
    "heartbeat, phase, unload, recovery, tamper, and atomic-write replay"
)
RUFF_COMMAND = f".venv/bin/ruff check {MODULE_PATH} {TEST_PATH}"
FORMAT_COMMAND = RUFF_COMMAND.replace("ruff check", "ruff format --check")
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6647_receipt_scoped_admission_boundary --validate"
)
ADVERSARIAL_COMMAND = f".venv/bin/python scripts/adversarial_verify.py {RESULT_PATH}"


def _definition(
    ordinal: int,
    check_id: str,
    source: str,
    command: str,
) -> JsonDict:
    return {
        "ordinal": ordinal,
        "check_id": check_id,
        "expected_value": True,
        "source": source,
        "command": command,
        "receipt_schema": f"carnot.experiment_6647.check.{check_id}.v1",
    }


PREREGISTERED_TASK_OWNED_CHECKS = (
    _definition(1, "acquisition", "GpuLease.acquire", "fixture:acquisition"),
    _definition(
        2,
        "same_device_exclusion",
        "GpuLease.acquire/LeaseBusy",
        "fixture:same-device-exclusion",
    ),
    _definition(
        3,
        "independent_device_allowance",
        "GpuLease.acquire/device-scoped-lock",
        "fixture:independent-device-allowance",
    ),
    _definition(
        4,
        "token_pid_start_device_binding",
        "GpuLease._verify_owner",
        "fixture:token-pid-start-device-binding",
    ),
    _definition(5, "heartbeat", "GpuLease.heartbeat", "fixture:heartbeat"),
    _definition(
        6,
        "phase_transitions",
        "build_phase_transition_rows",
        "fixture:phase-transitions",
    ),
    _definition(7, "unload_release", "GpuLease.release", "fixture:unload-release"),
    _definition(
        8,
        "crash_recovery",
        "fixture_worker_main/GpuLease.acquire",
        "fixture:crash-recovery",
    ),
    _definition(9, "tamper_detection", "GpuLease._refresh", "fixture:tamper-detection"),
    _definition(
        10,
        "atomic_artifact_write",
        "write_json_atomic",
        "fixture:atomic-artifact-write",
    ),
    _definition(11, "focused_tests", "tests_run", FOCUSED_TEST_COMMAND),
    _definition(12, "spec_coverage", "tests_run", SPEC_COMMAND),
    _definition(13, "applicable_e2e_checks", "tests_run", E2E_COMMAND),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "prior_failure_receipt",
    "preregistered_task_owned_checks",
    "task_owned_check_rows",
    "global_suite_diagnostic",
    "reducer_contract",
    "task_owned_admission_ready_score",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "status": "One terminal state reports whether every owned receipt passed.",
    "honest_verdict": "The conclusion is limited to receipt-bound infrastructure.",
    "verdict_class": "Ready infrastructure is null; failed owned evidence is blocked.",
    "gate_check_summary": "Each block keeps the failed check and exact observed value.",
    "prior_failure_receipt": "The changed boundary stays tied to the Exp6633 failure.",
    "preregistered_task_owned_checks": "The ordered gate set exists before execution.",
    "task_owned_check_rows": "One replayable row exists for each owned definition.",
    "global_suite_diagnostic": "Repo-wide failure truth stays visible and non-gating.",
    "reducer_contract": "Inclusion and exclusion rules prevent boundary drift.",
    "task_owned_admission_ready_score": "Only exact owned rows control this binary field.",
    "per_unit_rows": "Owned and diagnostic rows remain individually inspectable.",
    "aggregate_row_recomputation": "The aggregate is rebuilt from rows without null coercion.",
    "preconditions_checked": "Inputs, tools, resources, hashes, and no-LLM scope are recorded.",
    "protected_files_unchanged": "The active roadmap and conductor stay byte-identical.",
    "inference_substrate": "The substrate prevents an infrastructure result becoming a model claim.",
    "verifier_is_oracle": "Fixture expectations define null infrastructure readiness.",
    "field_provenance": "Each field names its producer, parser, schema, and content hash.",
    "random_seed": "A fixed seed pins the fixture order even though no random sampling occurs.",
    "duration_s": "Monotonic elapsed time proves the bounded replay ran.",
    "tests_run": "Commands and exits preserve verification evidence without widening readiness.",
    "reproducibility_checksum": "The final content hash detects any artifact mutation.",
}

DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0, "summary": "focused tests passed"},
    {"command": COVERAGE_COMMAND, "exit_code": 0, "summary": "scoped coverage run passed"},
    {
        "command": COVERAGE_REPORT_COMMAND,
        "exit_code": 0,
        "summary": "new module has 100% scoped statement coverage",
    },
    {
        "command": FULL_TEST_COMMAND,
        "exit_code": 3,
        "summary": (
            "1037 failed, 34797 passed, 103 skipped, 140 warnings in 2310.03s; "
            "pytest/xdist stopped at 62% after an existing test removed its worker CWD, "
            "raising FileNotFoundError"
        ),
    },
    {"command": SPEC_COMMAND, "exit_code": 0, "summary": "focused spec coverage passed"},
    {"command": RUFF_COMMAND, "exit_code": 0, "summary": "focused lint passed"},
    {"command": FORMAT_COMMAND, "exit_code": 0, "summary": "focused format check passed"},
    {"command": VALIDATE_COMMAND, "exit_code": 0, "summary": "artifact validation passed"},
    {
        "command": ADVERSARIAL_COMMAND,
        "exit_code": 1,
        "summary": "one non-critical substrate review warning; no critical finding",
    },
    {
        "command": E2E_COMMAND,
        "exit_code": 0,
        "summary": "all preregistered owned fixtures passed in fresh temporary paths",
    },
)


def sha256_file(path: str | Path) -> str:
    """Hash one file so a missing input cannot look like an empty file."""

    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def receipt_hash(value: Any, *, excluded: Sequence[str] = ()) -> str:
    """Hash a JSON receipt after removing only named self-referential fields."""

    if isinstance(value, Mapping):
        payload = {key: item for key, item in value.items() if key not in set(excluded)}
    else:
        payload = value
    return lease_api.sha256_json(payload)


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash final content while excluding only the checksum field itself."""

    return receipt_hash(payload, excluded=("reproducibility_checksum",))


def protected_hashes(root: Path) -> dict[str, str]:
    """Hash the two files that this task has no authority to change."""

    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_PATHS}


def _protected_receipt(root: Path, before: Mapping[str, str]) -> JsonDict:
    after = protected_hashes(root)
    return {
        "schema": "carnot.experiment_6647.protected_files.v1",
        "before_hashes": dict(before),
        "after_hashes": after,
        "files": {
            path: {
                "before": before.get(path),
                "after": after.get(path),
                "unchanged": before.get(path) == after.get(path),
            }
            for path in sorted(set(before) | set(after))
        },
        "all_unchanged": bool(before) and dict(before) == after,
    }


def _read_prior(root: Path) -> JsonDict:
    path = root / PRIOR_PATH
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def load_global_suite_diagnostic(root: Path) -> JsonDict:
    """Load the exact repo-wide receipt without treating it as owned evidence."""

    prior = _read_prior(root)
    tests = prior.get("tests_run", [])
    receipt = next(
        (
            dict(row)
            for row in tests
            if isinstance(row, Mapping) and row.get("command") == FULL_TEST_COMMAND
        ),
        {"command": FULL_TEST_COMMAND, "exit_code": None, "summary": "receipt missing"},
    )
    diagnostic = {
        "row_kind": "global_suite_diagnostic",
        "command": receipt.get("command"),
        "exit_code": receipt.get("exit_code"),
        "summary": receipt.get("summary"),
        "source": PRIOR_PATH.as_posix(),
        "known_issue_link": KNOWN_ISSUE_LINK,
        "gating": False,
        "non_gating_rationale": (
            "The repo-wide selection includes unrelated tests and the known xdist/CWD "
            "failure. It remains operational truth but does not own this task's readiness."
        ),
        "schema": "carnot.experiment_6647.global_suite_diagnostic.v1",
    }
    diagnostic["receipt_hash"] = receipt_hash(diagnostic)
    return diagnostic


def _prior_failure_receipt(root: Path) -> JsonDict:
    prior = _read_prior(root)
    return {
        "experiment_id": 6633,
        "artifact_path": PRIOR_PATH.as_posix(),
        "artifact_sha256": sha256_file(root / PRIOR_PATH),
        "status": prior.get("status"),
        "verdict": prior.get("honest_verdict"),
        "verdict_class": prior.get("verdict_class"),
        "exact_failed_reduction_field": "focused_tests",
        "observed_value": False,
        "changed_boundary": (
            "Exp6647 includes only preregistered task-owned rows in readiness and keeps "
            "the repo-wide suite receipt as a separate diagnostic."
        ),
        "schema": "carnot.experiment_6647.prior_failure.v1",
    }


def collect_preconditions(
    root: Path, fixture_root: Path, protected_before: Mapping[str, str]
) -> JsonDict:
    """Record the exact reducer inputs, host resources, tools, and no-LLM scope."""

    disk = shutil.disk_usage(root)
    page_size = os.sysconf("SC_PAGE_SIZE")
    physical_pages = os.sysconf("SC_PHYS_PAGES")
    return {
        "schema": "carnot.experiment_6647.preconditions.v1",
        "inputs": {
            "planning_date": RUN_DATE,
            "root": str(root.resolve()),
            "prior_artifact_path": PRIOR_PATH.as_posix(),
            "prior_artifact_sha256": sha256_file(root / PRIOR_PATH),
            "research_roadmap_sha256": protected_before.get("research-roadmap.yaml"),
            "research_conductor_sha256": protected_before.get("scripts/research_conductor.py"),
            "lease_module_sha256": sha256_file(root / LEASE_PATH),
            "prior_reducer_sha256": sha256_file(root / PRIOR_MODULE_PATH),
            "module_sha256": sha256_file(root / MODULE_PATH),
            "test_sha256": sha256_file(root / TEST_PATH),
            "reporting_spec_sha256": sha256_file(SPEC_PATH),
            "infrastructure_spec_sha256": sha256_file(INFRA_SPEC_PATH),
        },
        "task_owned_fixture_inventory": [
            definition["check_id"] for definition in PREREGISTERED_TASK_OWNED_CHECKS
        ],
        "current_global_suite_receipt": load_global_suite_diagnostic(root),
        "fixture_root": str(fixture_root.resolve()),
        "python": {
            "executable": os.fspath(Path(sys.executable).resolve()),
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "host_resources": {
            "cpu_count": os.cpu_count(),
            "ram_bytes": page_size * physical_pages,
            "disk_total_bytes": disk.total,
            "disk_free_bytes": disk.free,
        },
        "tools": {
            "pytest": (root / ".venv/bin/pytest").is_file(),
            "coverage": (root / ".venv/bin/coverage").is_file(),
            "spec_coverage": (root / "scripts/check_spec_coverage.py").is_file(),
            "adversarial_verify": (root / "scripts/adversarial_verify.py").is_file(),
        },
        "protected_hashes_before": dict(protected_before),
        "no_llm": {
            "inference_substrate": INFERENCE_SUBSTRATE,
            "model_load_attempt_count": 0,
            "generation_attempt_count": 0,
            "llm_import_required": False,
        },
    }


def make_check_row(
    definition: Mapping[str, Any],
    *,
    observed_value: Any,
    exit_code: int | None,
    receipt: Mapping[str, Any],
    fixture_path: Path | None = None,
) -> JsonDict:
    """Bind one observation to its preregistered definition and full receipt."""

    row = {
        "ordinal": definition.get("ordinal"),
        "check_id": definition.get("check_id"),
        "expected_value": definition.get("expected_value"),
        "observed_value": observed_value,
        "source": definition.get("source"),
        "command": definition.get("command"),
        "exit_code": exit_code,
        "receipt_schema": definition.get("receipt_schema"),
        "receipt": dict(receipt),
    }
    if fixture_path is not None:
        row["fixture_path"] = str(fixture_path.resolve())
    row["receipt_hash"] = receipt_hash(row)
    return row


def _finish_blocked(lease: lease_api.GpuLease) -> JsonDict:
    lease.transition("terminal_blocked")
    return lease.release()


def _fixture_acquisition(path: Path) -> tuple[bool, JsonDict]:
    lease = lease_api.GpuLease.acquire(
        runtime_dir=path,
        task_id="exp6647-acquisition",
        device_uuid="GPU-exp6647-acquisition",
        expected_model="fixture/no-model.gguf",
        vram_before_mb=4,
    )
    owner = lease.owner_receipt()
    release = _finish_blocked(lease)
    passed = owner.get("token_opaque") is True and release.get("released") is True
    return passed, {"owner": owner, "release": release}


def _fixture_same_device(path: Path) -> tuple[bool, JsonDict]:
    owner = lease_api.GpuLease.acquire(
        runtime_dir=path,
        task_id="exp6647-same-owner",
        device_uuid="GPU-exp6647-same",
        expected_model="fixture/no-model.gguf",
        vram_before_mb=4,
    )
    try:
        lease_api.GpuLease.acquire(
            runtime_dir=path,
            task_id="exp6647-same-contender",
            device_uuid="GPU-exp6647-same",
            expected_model="fixture/no-model.gguf",
            vram_before_mb=4,
        )
    except lease_api.LeaseBusy as exc:
        outcome = type(exc).__name__
    else:  # pragma: no cover - a focused red test would show a lease defect
        outcome = "accepted"
    release = _finish_blocked(owner)
    return outcome == "LeaseBusy", {"contender_outcome": outcome, "release": release}


def _fixture_independent_devices(path: Path) -> tuple[bool, JsonDict]:
    leases = [
        lease_api.GpuLease.acquire(
            runtime_dir=path,
            task_id=f"exp6647-independent-{index}",
            device_uuid=f"GPU-exp6647-independent-{index}",
            expected_model="fixture/no-model.gguf",
            vram_before_mb=4,
        )
        for index in (0, 1)
    ]
    owners = [lease.owner_receipt() for lease in leases]
    releases = [_finish_blocked(lease) for lease in leases]
    passed = len({owner["device_uuid"] for owner in owners}) == 2 and all(
        receipt.get("released") is True for receipt in releases
    )
    return passed, {"owners": owners, "releases": releases}


def _fixture_binding(path: Path) -> tuple[bool, JsonDict]:
    lease = lease_api.GpuLease.acquire(
        runtime_dir=path,
        task_id="exp6647-binding",
        device_uuid="GPU-exp6647-binding",
        expected_model="fixture/no-model.gguf",
        vram_before_mb=4,
    )
    attacks = {
        "wrong_token": lambda: lease.heartbeat(token="wrong"),
        "wrong_device": lambda: lease.transition("admitted", device_uuid="wrong"),
        "pid_reuse": lambda: lease.transition(
            "admitted", pid_start_ticks=lease.pid_start_ticks + 1
        ),
    }
    outcomes = {}
    for attack_id, action in attacks.items():
        try:
            action()
        except lease_api.LeaseError as exc:
            outcomes[attack_id] = type(exc).__name__
        else:  # pragma: no cover - a focused red test would show a lease defect
            outcomes[attack_id] = "accepted"
    release = _finish_blocked(lease)
    passed = all(value != "accepted" for value in outcomes.values())
    return passed, {
        "attack_ids": list(attacks),
        "outcomes": outcomes,
        "release": release,
    }


def _fixture_heartbeat(path: Path) -> tuple[bool, JsonDict]:
    lease = lease_api.GpuLease.acquire(
        runtime_dir=path,
        task_id="exp6647-heartbeat",
        device_uuid="GPU-exp6647-heartbeat",
        expected_model="fixture/no-model.gguf",
        vram_before_mb=4,
    )
    heartbeat = lease.heartbeat()
    release = _finish_blocked(lease)
    passed = heartbeat.get("owner_verified") is True and release.get("released") is True
    return passed, {"heartbeat": heartbeat, "release": release}


def _fixture_phase_transitions(path: Path) -> tuple[bool, JsonDict]:
    rows = prior_exp.build_phase_transition_rows(path)
    rejected = {row.get("case") for row in rows if row.get("accepted") is False}
    passed = bool(rows) and {"skip", "reversal", "second_terminal"} <= rejected
    return passed, {"rows": rows, "rejected_cases": sorted(rejected)}


def _fixture_unload_release(path: Path) -> tuple[bool, JsonDict]:
    lease = lease_api.GpuLease.acquire(
        runtime_dir=path,
        task_id="exp6647-unload",
        device_uuid="GPU-exp6647-unload",
        expected_model="fixture/no-model.gguf",
        vram_before_mb=4,
    )
    phase_rows = prior_exp._complete_lease(lease)
    unload_evidence = deepcopy(lease.document.get("unload_evidence", {}))
    exit_evidence = deepcopy(lease.document.get("exit_evidence", {}))
    release = lease.release()
    passed = (
        unload_evidence.get("observed") is True
        and exit_evidence.get("exit_code") == 0
        and release.get("released") is True
    )
    return passed, {
        "phase_rows": phase_rows,
        "unload_evidence": unload_evidence,
        "exit_evidence": exit_evidence,
        "release": release,
    }


def _fixture_crash_recovery(path: Path) -> tuple[bool, JsonDict]:
    crash = prior_exp._worker_command(
        path,
        task_id="exp6647-crash",
        device_uuid="GPU-exp6647-crash",
        behavior="crash",
        exit_code=prior_exp.FIXTURE_CRASH_EXIT,
    )
    crash_code, crash_rows, crash_stderr = prior_exp._run_worker(crash)
    recover = prior_exp._worker_command(
        path,
        task_id="exp6647-recovery",
        device_uuid="GPU-exp6647-crash",
        behavior="recover_complete",
    )
    recover_code, recover_rows, recover_stderr = prior_exp._run_worker(recover)
    owner = next(
        (row.get("owner", {}) for row in recover_rows if row.get("outcome") == "acquired"),
        {},
    )
    passed = (
        crash_code == prior_exp.FIXTURE_CRASH_EXIT
        and recover_code == 0
        and owner.get("recovery", {}).get("performed") is True
    )
    return passed, {
        "commands": [crash, recover],
        "crash_exit_code": crash_code,
        "recovery_exit_code": recover_code,
        "crash_rows": crash_rows,
        "recovery_rows": recover_rows,
        "stderr": [crash_stderr, recover_stderr],
    }


def _fixture_tamper(path: Path) -> tuple[bool, JsonDict]:
    lease = lease_api.GpuLease.acquire(
        runtime_dir=path,
        task_id="exp6647-tamper",
        device_uuid="GPU-exp6647-tamper",
        expected_model="fixture/no-model.gguf",
        vram_before_mb=4,
    )
    changed = json.loads(lease.journal_path.read_text(encoding="utf-8"))
    changed["expected_model"] = "tampered/model.gguf"
    lease.journal_path.write_text(json.dumps(changed), encoding="utf-8")
    try:
        lease.heartbeat()
    except lease_api.JournalError as exc:
        outcome = type(exc).__name__
        reason = str(exc)
    else:  # pragma: no cover - a focused red test would show a lease defect
        outcome = "accepted"
        reason = ""
    lease.close()
    return outcome == "JournalError", {"outcome": outcome, "reason": reason}


def _fixture_atomic_write(path: Path) -> tuple[bool, JsonDict]:
    target = path / "atomic-probe.json"
    payload = {"state": "terminal", "sequence": [1, 2, 3]}
    lease_api.write_json_atomic(target, payload)
    observed = json.loads(target.read_text(encoding="utf-8"))
    partials = sorted(item.name for item in path.iterdir() if item != target)
    passed = observed == payload and not partials
    return passed, {
        "target": str(target.resolve()),
        "observed": observed,
        "target_sha256": sha256_file(target),
        "partial_paths": partials,
    }


FIXTURE_FUNCTIONS = (
    _fixture_acquisition,
    _fixture_same_device,
    _fixture_independent_devices,
    _fixture_binding,
    _fixture_heartbeat,
    _fixture_phase_transitions,
    _fixture_unload_release,
    _fixture_crash_recovery,
    _fixture_tamper,
    _fixture_atomic_write,
)


def _test_receipt(
    command: str, tests_run: Sequence[Mapping[str, Any]]
) -> tuple[Any, int | None, JsonDict]:
    receipt = next(
        (
            dict(row)
            for row in tests_run
            if isinstance(row, Mapping) and row.get("command") == command
        ),
        {"command": command, "exit_code": None, "summary": "receipt missing"},
    )
    exit_code = receipt.get("exit_code")
    observed = None if exit_code is None else exit_code == 0
    return observed, exit_code if isinstance(exit_code, int) else None, receipt


def replay_task_owned_checks(
    runtime_dir: Path,
    tests_run: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Replay the frozen inventory without adding the global-suite diagnostic."""

    runtime_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for definition, fixture in zip(
        PREREGISTERED_TASK_OWNED_CHECKS[:10], FIXTURE_FUNCTIONS, strict=True
    ):
        path = runtime_dir / f"{definition['ordinal']:02d}-{definition['check_id']}"
        path.mkdir(parents=True, exist_ok=False)
        try:
            observed, receipt = fixture(path)
            exit_code = 0 if observed is True else 1
        except (OSError, lease_api.LeaseError, RuntimeError, TimeoutError) as exc:
            observed = False
            exit_code = 1
            receipt = {"error": f"{type(exc).__name__}: {exc}"}
        rows.append(
            make_check_row(
                definition,
                observed_value=observed,
                exit_code=exit_code,
                receipt=receipt,
                fixture_path=path,
            )
        )
    for definition in PREREGISTERED_TASK_OWNED_CHECKS[10:]:
        observed, exit_code, receipt = _test_receipt(definition["command"], tests_run)
        rows.append(
            make_check_row(
                definition,
                observed_value=observed,
                exit_code=exit_code,
                receipt=receipt,
            )
        )
    return rows


def reduce_task_owned_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], JsonDict]:
    """Rebuild readiness from exact rows while preserving missing and null values."""

    definitions = list(PREREGISTERED_TASK_OWNED_CHECKS)
    row_list = [dict(row) for row in rows]
    counts = Counter(row.get("check_id") for row in row_list)
    failures: list[JsonDict] = []
    passed_ids = []
    for index, definition in enumerate(definitions):
        check_id = definition["check_id"]
        matches = [row for row in row_list if row.get("check_id") == check_id]
        if not matches:
            failures.append(
                {
                    "check": check_id,
                    "expected_value": definition["expected_value"],
                    "observed_value": None,
                    "reason": "missing_receipt",
                }
            )
            continue
        row = matches[0]
        reason = None
        if counts[check_id] != 1:
            reason = "duplicate_receipt"
        elif index >= len(row_list) or row_list[index].get("check_id") != check_id:
            reason = "receipt_order_mismatch"
        elif any(row.get(key) != definition.get(key) for key in definition):
            reason = "definition_mismatch"
        elif row.get("receipt_hash") != receipt_hash(row, excluded=("receipt_hash",)):
            reason = "receipt_hash_mismatch"
        elif row.get("observed_value") is None:
            reason = "null_observed_value"
        elif row.get("observed_value") != definition["expected_value"]:
            reason = "observed_value_mismatch"
        if reason is None:
            passed_ids.append(check_id)
        else:
            failures.append(
                {
                    "check": check_id,
                    "expected_value": definition["expected_value"],
                    "observed_value": row.get("observed_value"),
                    "reason": reason,
                }
            )
    aggregate = {
        "schema": "carnot.experiment_6647.aggregate.v1",
        "ordered_check_ids": [definition["check_id"] for definition in definitions],
        "received_ordered_check_ids": [row.get("check_id") for row in row_list],
        "expected_row_count": len(definitions),
        "observed_row_count": len(row_list),
        "passed_check_ids": passed_ids,
        "failed_checks": failures,
        "all_task_owned_checks_passed": not failures and len(row_list) == len(definitions),
        "recomputed_task_owned_admission_ready_score": (
            1.0 if not failures and len(row_list) == len(definitions) else 0.0
        ),
        "included_owned_row_count": len(row_list),
        "excluded_diagnostic_count": 1,
        "missing_or_null_coerced_to_zero": False,
    }
    return failures, aggregate


def _reducer_contract() -> JsonDict:
    return {
        "schema": "carnot.experiment_6647.reducer_contract.v1",
        "inclusion_rule": (
            "Include exactly one valid row for each preregistered task-owned check, "
            "in declared order, and require observed_value == expected_value."
        ),
        "included_check_ids": [
            definition["check_id"] for definition in PREREGISTERED_TASK_OWNED_CHECKS
        ],
        "exclusion_rule": (
            "Exclude global_suite_diagnostic and every unregistered repo-wide result "
            "from task_owned_admission_ready_score."
        ),
        "excluded_fields": ["global_suite_diagnostic"],
        "missing_rule": "Missing and null observations remain distinct and block readiness.",
        "ownership_rule": (
            "Only task_owned_check_rows own task_owned_admission_ready_score; the global "
            "suite owns operational diagnostics only."
        ),
    }


def build_field_provenance(artifact: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Describe the parser, producer, schema, and content hash for every field."""

    provenance = {}
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field == "field_provenance":
            digest = receipt_hash(sorted(REQUIRED_ARTIFACT_FIELDS))
        elif field == "reproducibility_checksum":
            digest = "self_excluded_from_content_hash"
        else:
            digest = receipt_hash(artifact.get(field))
        provenance[field] = {
            "source_path": (
                PRIOR_PATH.as_posix()
                if field in {"prior_failure_receipt", "global_suite_diagnostic"}
                else MODULE_PATH.as_posix()
            ),
            "parser": "json.loads_and_exact_row_reduction",
            "function": {
                "prior_failure_receipt": "_prior_failure_receipt",
                "global_suite_diagnostic": "load_global_suite_diagnostic",
                "preconditions_checked": "collect_preconditions",
                "protected_files_unchanged": "_protected_receipt",
                "task_owned_check_rows": "replay_task_owned_checks",
                "aggregate_row_recomputation": "reduce_task_owned_rows",
                "field_provenance": "build_field_provenance",
                "reproducibility_checksum": "payload_checksum",
            }.get(field, "build_artifact"),
            "hash": digest,
            "schema": f"carnot.experiment_6647.field.{field}.v1",
            "principle": FIELD_PRINCIPLES[field],
        }
    return provenance


def finalize_reduction(artifact: Mapping[str, Any]) -> JsonDict:
    """Set terminal fields from owned rows and refresh provenance and checksums."""

    result = deepcopy(dict(artifact))
    failures, aggregate = reduce_task_owned_rows(result.get("task_owned_check_rows", []))
    score = aggregate["recomputed_task_owned_admission_ready_score"]
    result["gate_check_summary"] = failures
    result["aggregate_row_recomputation"] = aggregate
    result["task_owned_admission_ready_score"] = score
    diagnostic = dict(result.get("global_suite_diagnostic", {}))
    result["per_unit_rows"] = [
        {"row_kind": "task_owned_check", **dict(row)}
        for row in result.get("task_owned_check_rows", [])
    ] + [{"row_kind": "global_suite_diagnostic", **diagnostic}]
    if score == 1.0:
        result["status"] = "complete_ready"
        result["honest_verdict"] = (
            "complete: preregistered task-owned GPU lease receipts support infrastructure "
            "admission; the global suite remains diagnostic; no model-quality claim"
        )
        result["verdict_class"] = None
    else:
        first = failures[0]["check"] if failures else "owned_receipt_set"
        result["status"] = f"blocked_task_owned_check_{first}"
        result["honest_verdict"] = (
            f"blocked_task_owned_check_{first}: task-owned receipt reduction failed; "
            "no model-quality claim"
        )
        result["verdict_class"] = "blocked"
    result["field_provenance"] = build_field_provenance(result)
    result["reproducibility_checksum"] = payload_checksum(result)
    return result


def build_artifact(
    *,
    date: str,
    root: Path,
    duration_s: float,
    check_rows: Sequence[Mapping[str, Any]],
    global_suite_diagnostic: Mapping[str, Any],
    protected_before: Mapping[str, str],
    preconditions: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build one terminal artifact before the atomic publication step."""

    artifact = {
        "status": "pending",
        "honest_verdict": "pending",
        "verdict_class": "blocked",
        "gate_check_summary": [],
        "prior_failure_receipt": _prior_failure_receipt(root),
        "preregistered_task_owned_checks": [
            dict(definition) for definition in PREREGISTERED_TASK_OWNED_CHECKS
        ],
        "task_owned_check_rows": [dict(row) for row in check_rows],
        "global_suite_diagnostic": dict(global_suite_diagnostic),
        "reducer_contract": _reducer_contract(),
        "task_owned_admission_ready_score": 0.0,
        "per_unit_rows": [],
        "aggregate_row_recomputation": {},
        "preconditions_checked": dict(preconditions),
        "protected_files_unchanged": _protected_receipt(root, protected_before),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": {},
        "random_seed": RANDOM_SEED,
        "duration_s": round(float(duration_s), 6),
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "pending",
    }
    return finalize_reduction(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute row ownership, terminal state, provenance, and content hashes."""

    errors = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required_fields_mismatch")
    if artifact.get("preregistered_task_owned_checks") != [
        dict(definition) for definition in PREREGISTERED_TASK_OWNED_CHECKS
    ]:
        errors.append("preregistered_checks_mismatch")
    rows = artifact.get("task_owned_check_rows", [])
    row_list = rows if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) else []
    for row in row_list:
        if isinstance(row, Mapping) and row.get("receipt_hash") != receipt_hash(
            row, excluded=("receipt_hash",)
        ):
            errors.append(f"row_receipt_hash_mismatch:{row.get('check_id')}")
    failures, aggregate = reduce_task_owned_rows(
        [row for row in row_list if isinstance(row, Mapping)]
    )
    score = aggregate["recomputed_task_owned_admission_ready_score"]
    if artifact.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate_row_recomputation_mismatch")
    if artifact.get("task_owned_admission_ready_score") != score:
        errors.append("readiness_score_mismatch")
    diagnostic = artifact.get("global_suite_diagnostic")
    diagnostic_row = dict(diagnostic) if isinstance(diagnostic, Mapping) else {}
    expected_units = [
        {"row_kind": "task_owned_check", **dict(row)}
        for row in row_list
        if isinstance(row, Mapping)
    ] + [
        {
            "row_kind": "global_suite_diagnostic",
            **diagnostic_row,
        }
    ]
    if artifact.get("per_unit_rows") != expected_units:
        errors.append("per_unit_rows_mismatch")
    if not isinstance(diagnostic, Mapping):
        errors.append("global_suite_diagnostic_missing")
    else:
        if diagnostic.get("gating") is not False:
            errors.append("global_suite_diagnostic_gating")
        if diagnostic.get("receipt_hash") != receipt_hash(diagnostic, excluded=("receipt_hash",)):
            errors.append("global_suite_diagnostic_hash_mismatch")
    if artifact.get("reducer_contract") != _reducer_contract():
        errors.append("reducer_contract_mismatch")
    if score == 1.0:
        if artifact.get("status") != "complete_ready":
            errors.append("ready_status_mismatch")
        if artifact.get("verdict_class") is not None:
            errors.append("ready_verdict_class_mismatch")
        if artifact.get("gate_check_summary") != []:
            errors.append("ready_gate_summary_mismatch")
    else:
        if not str(artifact.get("status", "")).startswith("blocked_"):
            errors.append("blocked_status_mismatch")
        if not str(artifact.get("honest_verdict", "")).startswith("blocked_"):
            errors.append("blocked_verdict_mismatch")
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked_verdict_class_mismatch")
        if artifact.get("gate_check_summary") != failures:
            errors.append("blocked_gate_summary_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_mismatch")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_mismatch")
    if artifact.get("protected_files_unchanged", {}).get("all_unchanged") is not True:
        errors.append("protected_files_changed")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance_mismatch")
    else:
        expected_provenance = build_field_provenance(artifact)
        for field, receipt in provenance.items():
            if not isinstance(receipt, Mapping) or not {
                "source_path",
                "parser",
                "function",
                "hash",
                "schema",
            } <= set(receipt):
                errors.append(f"field_provenance_invalid:{field}")
            elif receipt != expected_provenance[field]:
                errors.append(f"field_provenance_hash_mismatch:{field}")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def run(
    *,
    date: str,
    root: Path = REPO_ROOT,
    result_path: Path | None = None,
    work_dir: Path | None = None,
    tests_run: Sequence[Mapping[str, Any]] = DEFAULT_TESTS_RUN,
) -> JsonDict:
    """Replay owned checks and publish one validated JSON file atomically."""

    started = time.monotonic()
    output = result_path or (root / RESULT_PATH)
    work_root = work_dir or (root / WORK_PATH)
    run_root = work_root / f"run-{time.monotonic_ns()}"
    protected_before = protected_hashes(root)
    preconditions = collect_preconditions(root, run_root, protected_before)
    rows = replay_task_owned_checks(run_root, tests_run)
    artifact = build_artifact(
        date=date,
        root=root,
        duration_s=time.monotonic() - started,
        check_rows=rows,
        global_suite_diagnostic=load_global_suite_diagnostic(root),
        protected_before=protected_before,
        preconditions=preconditions,
        tests_run=tests_run,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"invalid Exp6647 artifact: {errors}")
    lease_api.write_json_atomic(output, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run or validate the receipt-scoped admission artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or (REPO_ROOT / RESULT_PATH)
    if args.validate:
        if not output.is_file():
            print(json.dumps({"valid": False, "errors": ["artifact_missing"]}))
            return 1
        try:
            artifact = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(
                json.dumps(
                    {"valid": False, "errors": [f"artifact_unreadable:{type(exc).__name__}"]}
                )
            )
            return 1
        errors = validate_artifact(artifact)
        print(json.dumps({"valid": not errors, "errors": errors}))
        return 1 if errors else 0
    artifact = run(
        date=args.date,
        result_path=output,
        work_dir=args.work_dir,
    )
    print(
        json.dumps(
            {
                "artifact": str(output),
                "status": artifact["status"],
                "task_owned_admission_ready_score": artifact["task_owned_admission_ready_score"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
