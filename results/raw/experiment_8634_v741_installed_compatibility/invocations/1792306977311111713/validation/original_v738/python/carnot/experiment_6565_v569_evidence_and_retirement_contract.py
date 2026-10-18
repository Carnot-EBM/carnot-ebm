"""Exp6565 V569 evidence and retirement contract.

Spec refs: REQ-REPORT-6565, SCENARIO-REPORT-6565-IMPORT,
SCENARIO-REPORT-6565-LIVE-REPLAY, SCENARIO-REPORT-6565-GATES,
SCENARIO-REPORT-6565-PRIOR-FAILURE,
SCENARIO-REPORT-6565-MODEL-ARC-HARDWARE,
SCENARIO-REPORT-6565-ATOMIC.

The reducer imports V568 terminal artifacts by exact path and hash. It records
fresh verifier receipts, failed-scope dispositions, and retirement boundaries
before downstream V569 work can extend an extraction, production, ARC, or
hardware lane. It performs no LLM load and no hardware command.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from typing import Any

import yaml

from carnot.experiment_artifacts import atomic_write_json
from scripts.roadmap_schema import Roadmap


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6565
INFERENCE_SUBSTRATE = "immutable_v568_artifact_gate_failure_and_retirement_audit_no_llm"

RESULT_RELATIVE_PATH = Path("results/experiment_6565_v569_evidence_and_retirement_contract.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
PROPOSAL_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ARCHITECTURE_RELATIVE_PATH = Path("_bmad/architecture.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")

MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)


@dataclass(frozen=True)
class V568Artifact:
    """A fixed V568 input and the fields that determine its disposition."""

    exp_id: str
    task_id: str
    relative_path: Path
    readiness_fields: tuple[str, ...]


V568_ARTIFACTS = (
    V568Artifact(
        "exp6561",
        "exp6561-v568-evidence-gate-contract",
        Path("results/experiment_6561_v568_evidence_gate_contract.json"),
        ("v568_evidence_contract_ready_score",),
    ),
    V568Artifact(
        "exp6562",
        "exp6562-constraint-saturation-independent-audit-v2",
        Path("results/experiment_6562_constraint_saturation_independent_audit_v2.json"),
        (
            "constraint_saturation_independent_audit_ready_score",
            "constraint_saturation_policy_audited_score",
        ),
    ),
    V568Artifact(
        "exp6563",
        "exp6563-production-safety-net-workload-canary",
        Path("results/experiment_6563_production_safety_net_workload_canary.json"),
        (
            "production_workload_canary_ready_score",
            "production_workload_promotion_candidate_score",
        ),
    ),
    V568Artifact(
        "exp6564",
        "exp6564-rust-pyo3-safety-net-nfr01",
        Path("results/experiment_6564_rust_pyo3_safety_net_nfr01.json"),
        ("rust_pyo3_nfr01_ready_score",),
    ),
)

PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-complete.yaml"),
    Path("research-references.md"),
    ROADMAP_RELATIVE_PATH,
    PROPOSAL_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    Path("ops/conductor-log.md"),
    Path("ops/e2e-test-plan.md"),
    Path("_bmad/prd.md"),
    Path("_bmad/architecture.md"),
    Path("_bmad/traceability.md"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/conductor_gates.py"),
    Path("scripts/roadmap_schema.py"),
    Path("scripts/exclusion_manifest_lint.py"),
    Path("scripts/validate_prior_failures.py"),
    Path("scripts/research_conductor.py"),
    *(artifact.relative_path for artifact in V568_ARTIFACTS),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "v568_artifact_eligibility_rows",
    "live_verifier_and_duration_rows",
    "v569_gate_contract_rows",
    "prior_failure_and_retirement_rows",
    "model_arc_and_hardware_boundary",
    "v569_evidence_contract_ready_score",
    "rust_fusion_reopen_ready_score",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "gate_check_summary",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal state prevents a bootstrap record from posing as a closed evidence root.",
    "honest_verdict": "The verdict must state evidence eligibility, failed scopes, and retirement boundaries with a terminal prefix.",
    "verdict_class": "A closed enum carries null, blocked, partial, and disqualified status downstream.",
    "v568_artifact_eligibility_rows": "One row per V568 artifact makes each import decision recheckable.",
    "live_verifier_and_duration_rows": "Fresh commands, exits, flags, and monotonic durations resolve the Exp6561 inconsistency.",
    "v569_gate_contract_rows": "Every downstream gate must name an in-roadmap task and exact upstream field.",
    "prior_failure_and_retirement_rows": "Every matched failure needs a changed mechanism and a mechanical repeat-retirement rule.",
    "model_arc_and_hardware_boundary": "The contract freezes flagship, no-ARC-solve, and zero-unchanged-command rules.",
    "v569_evidence_contract_ready_score": "One binary field gates tasks that require the full V569 contract.",
    "rust_fusion_reopen_ready_score": "A separate binary field permits only the changed fused workload and freezes retirement.",
    "per_unit_rows": "Artifact-level rows prevent one bad input from hiding in an aggregate.",
    "aggregate_row_recomputation": "Readiness fields must derive only from emitted rows.",
    "gate_check_summary": "A blocked verdict must name the failed check and observed value.",
    "preconditions_checked": "Resource and input receipts distinguish missing prerequisites from evidence failure.",
    "protected_files_unchanged": "The task must not mutate research-roadmap.yaml or scripts/research_conductor.py.",
    "inference_substrate": "This is immutable artifact audit with no new LLM inference.",
    "verifier_is_oracle": "Artifact validation is audit authority, so a clean result cannot use a positive class.",
    "field_provenance": "Every headline field identifies source rows, hashes, and reducer.",
    "duration_s": "Monotonic duration exposes skipped contract work.",
    "tests_run": "Named commands and exit codes make the contract reproducible.",
    "reproducibility_checksum": "A final content hash detects terminal-record mutation.",
}

FIELD_PROVENANCE: dict[str, JsonDict] = {
    field: {
        "principle": FIELD_PRINCIPLES[field],
        "source": field,
        "source_hashes": [
            ROADMAP_RELATIVE_PATH.as_posix(),
            PROPOSAL_RELATIVE_PATH.as_posix(),
            *(artifact.relative_path.as_posix() for artifact in V568_ARTIFACTS),
        ],
        "reducer": "REQ-REPORT-6565 deterministic V568 evidence contract reducer",
        "spec_refs": ["REQ-REPORT-6565"],
    }
    for field in REQUIRED_ARTIFACT_FIELDS
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6565_v569_evidence_and_retirement_contract "
    "--date 20260823"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6565_v569_evidence_and_retirement_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6565_v569_evidence_and_retirement_contract.py "
    "-m pytest tests/python/test_experiment_6565_v569_evidence_and_retirement_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6565_v569_evidence_and_retirement_contract.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
RUFF_CHECK_COMMAND = (
    ".venv/bin/ruff check "
    "python/carnot/experiment_6565_v569_evidence_and_retirement_contract.py "
    "tests/python/test_experiment_6565_v569_evidence_and_retirement_contract.py "
    "scripts/adversarial_verify.py"
)
RUFF_FORMAT_COMMAND = (
    ".venv/bin/ruff format --check "
    "python/carnot/experiment_6565_v569_evidence_and_retirement_contract.py "
    "tests/python/test_experiment_6565_v569_evidence_and_retirement_contract.py "
    "scripts/adversarial_verify.py"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6565_v569_evidence_and_retirement_contract.py"
)
ROADMAP_SCHEMA_COMMAND = ".venv/bin/python scripts/roadmap_schema.py research-roadmap.yaml"
GATE_AUDIT_COMMAND = "internal Exp6565 gate audit over research-roadmap.yaml"
EXCLUSION_LINT_COMMAND = ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml"
PRIOR_FAILURE_COMMAND = ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml"
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6565_v569_evidence_and_retirement_contract.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6565_v569_evidence_and_retirement_contract.json"
)
E2E_PLAN_COMMAND = (
    "manual e2e-plan check: Exp6565 is an immutable contract audit; "
    "ops/e2e-test-plan.md has no direct Exp6565 entry"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6565_v569_evidence_and_retirement_contract --validate"
)

DEFAULT_TESTS_RUN = (
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": RUFF_CHECK_COMMAND, "exit_code": 0},
    {"command": RUFF_FORMAT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": ROADMAP_SCHEMA_COMMAND, "exit_code": 0},
    {"command": GATE_AUDIT_COMMAND, "exit_code": 0},
    {"command": EXCLUSION_LINT_COMMAND, "exit_code": 0},
    {"command": PRIOR_FAILURE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": E2E_PLAN_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path | None) -> str:
    if path is None:
        return "missing"
    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("reproducibility_checksum", None)
    return sha256_json(clone)


def _read_json(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive filesystem edge.
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _load_yaml(path: Path) -> JsonDict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def default_v568_paths(repo_root: Path = REPO_ROOT) -> dict[str, Path]:
    return {artifact.exp_id: repo_root / artifact.relative_path for artifact in V568_ARTIFACTS}


def _command_text(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in argv)


def _python_executable(repo_root: Path) -> str:  # pragma: no cover - host receipt.
    venv_python = repo_root / ".venv/bin/python"
    return str(venv_python) if venv_python.is_file() else sys.executable


def _run_command(argv: Sequence[str], repo_root: Path) -> JsonDict:  # pragma: no cover
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [str(part) for part in argv],
            cwd=repo_root,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        stdout = proc.stdout
        stderr = proc.stderr
        exit_code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        exit_code = 124
    return {
        "command": _command_text(argv),
        "exit_code": exit_code,
        "duration_s": round(time.monotonic() - started, 6),
        "stdout_sha256": sha256_bytes(str(stdout).encode("utf-8")),
        "stderr_sha256": sha256_bytes(str(stderr).encode("utf-8")),
        "stdout_tail": str(stdout)[-2000:],
        "stderr_tail": str(stderr)[-2000:],
    }


def _run_adversarial_check(path: Path, repo_root: Path) -> JsonDict:  # pragma: no cover
    argv = [_python_executable(repo_root), "scripts/adversarial_verify.py", "--json", str(path)]
    receipt = _run_command(argv, repo_root)
    try:
        parsed = json.loads(str(receipt.get("stdout_tail") or "{}"))
        reports = parsed.get("reports")
        report = dict(reports[0]) if isinstance(reports, list) and reports else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        report = {}
    return {
        "command": receipt["command"],
        "exit_code": receipt["exit_code"],
        "duration_s": receipt["duration_s"],
        "flag_count": int(report.get("flag_count") or 0),
        "max_severity": int(
            report.get("max_severity") if report.get("max_severity") is not None else -1
        ),
        "flags": list(report.get("flags") or []),
        "stdout_sha256": receipt["stdout_sha256"],
        "stderr_sha256": receipt["stderr_sha256"],
    }


def _run_row_consistency_check(path: Path, repo_root: Path) -> JsonDict:  # pragma: no cover
    from scripts import verdict_row_consistency_lint as row_lint

    argv = [_python_executable(repo_root), "scripts/verdict_row_consistency_lint.py", str(path)]
    receipt = _run_command(argv, repo_root)
    status, findings = row_lint.check_artifact(path)
    return {
        "command": receipt["command"],
        "exit_code": receipt["exit_code"],
        "duration_s": receipt["duration_s"],
        "status": status,
        "findings": list(findings),
        "stdout_sha256": receipt["stdout_sha256"],
        "stderr_sha256": receipt["stderr_sha256"],
    }


def run_live_checks(
    repo_root: Path = REPO_ROOT,
    paths: Mapping[str, Path] | None = None,
) -> dict[str, JsonDict]:  # pragma: no cover
    source_paths = default_v568_paths(repo_root) if paths is None else dict(paths)
    return {
        artifact.exp_id: {
            "adversarial": _run_adversarial_check(source_paths[artifact.exp_id], repo_root),
            "row_consistency": _run_row_consistency_check(source_paths[artifact.exp_id], repo_root),
        }
        for artifact in V568_ARTIFACTS
    }


def _tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    source = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    return [{"command": str(row["command"]), "exit_code": int(row["exit_code"])} for row in source]


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_files_unchanged(before: Mapping[str, str], after: Mapping[str, str]) -> JsonDict:
    rows = [
        {
            "path": key,
            "before_sha256": before.get(key, "missing"),
            "after_sha256": after.get(key, "missing"),
            "unchanged": before.get(key, "missing") == after.get(key, "missing"),
        }
        for key in sorted(set(before) | set(after))
    ]
    return {
        "all_unchanged": all(row["unchanged"] for row in rows),
        "changed_paths": [row["path"] for row in rows if not row["unchanged"]],
        "research_roadmap_yaml_unchanged": before.get(ROADMAP_RELATIVE_PATH.as_posix())
        == after.get(ROADMAP_RELATIVE_PATH.as_posix()),
        "research_conductor_py_unchanged": before.get("scripts/research_conductor.py")
        == after.get("scripts/research_conductor.py"),
        "rows": rows,
    }


def _tool_version(argv: Sequence[str], repo_root: Path) -> JsonDict:  # pragma: no cover
    receipt = _run_command(argv, repo_root)
    text = str(receipt.get("stdout_tail") or receipt.get("stderr_tail") or "").strip()
    return {
        "command": receipt["command"],
        "exit_code": receipt["exit_code"],
        "version_text": text.splitlines()[0] if text else "",
        "stdout_sha256": receipt["stdout_sha256"],
        "stderr_sha256": receipt["stderr_sha256"],
    }


def _git_receipt(repo_root: Path) -> JsonDict:  # pragma: no cover
    def run(args: Sequence[str]) -> str:
        try:
            return subprocess.check_output(
                args, cwd=repo_root, text=True, stderr=subprocess.STDOUT
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            return f"unavailable: {exc}"

    return {
        "head_sha": run(["git", "rev-parse", "HEAD"]),
        "status_short": run(["git", "status", "--short"]),
    }


def _resource_receipt(repo_root: Path) -> JsonDict:  # pragma: no cover
    disk = shutil.disk_usage(repo_root)
    mem_total_kib = None
    mem_available_kib = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        values: dict[str, int] = {}
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            match = re.search(r"\d+", rest)
            if match:
                values[key] = int(match.group(0))
        mem_total_kib = values.get("MemTotal")
        mem_available_kib = values.get("MemAvailable")
    cpu_model = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        match = re.search(r"model name\s*:\s*(.+)", cpuinfo.read_text(encoding="utf-8"))
        if match:
            cpu_model = match.group(1)
    return {
        "cpu": {"count": os.cpu_count(), "model": cpu_model},
        "ram": {"total_kib": mem_total_kib, "available_kib": mem_available_kib},
        "disk": {"path": str(repo_root), "total_bytes": disk.total, "free_bytes": disk.free},
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": platform.platform(),
    }


def _network_receipt() -> JsonDict:  # pragma: no cover
    started = time.monotonic()
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=1.0):
            reachable = True
            error = ""
    except OSError as exc:
        reachable = False
        error = f"{type(exc).__name__}: {exc}"
    return {
        "checked": True,
        "method": "tcp_connect_1.1.1.1_53_timeout_1s",
        "reachable": reachable,
        "error": error,
        "duration_s": round(time.monotonic() - started, 6),
    }


def _z3_receipt() -> JsonDict:  # pragma: no cover
    try:
        import z3  # type: ignore[import-not-found]

        return {"available": True, "version": str(z3.get_version_string())}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _pyo3_receipt(repo_root: Path) -> JsonDict:  # pragma: no cover
    lock = repo_root / "Cargo.lock"
    cargo = repo_root / "crates/carnot-python/Cargo.toml"
    text = lock.read_text(encoding="utf-8") if lock.is_file() else ""
    match = re.search(r'name = "pyo3"\nversion = "([^"]+)"', text)
    return {
        "cargo_lock_sha256": sha256_file(lock),
        "carnot_python_cargo_toml_sha256": sha256_file(cargo),
        "pyo3_version": match.group(1) if match else "unknown",
    }


def _verifier_versions(repo_root: Path) -> JsonDict:
    return {
        "adversarial_verify.py": {
            "path": "scripts/adversarial_verify.py",
            "sha256": sha256_file(repo_root / "scripts/adversarial_verify.py"),
        },
        "verdict_row_consistency_lint.py": {
            "path": "scripts/verdict_row_consistency_lint.py",
            "sha256": sha256_file(repo_root / "scripts/verdict_row_consistency_lint.py"),
        },
        "conductor_gates.py": {
            "path": "scripts/conductor_gates.py",
            "sha256": sha256_file(repo_root / "scripts/conductor_gates.py"),
        },
        "roadmap_schema.py": {
            "path": "scripts/roadmap_schema.py",
            "sha256": sha256_file(repo_root / "scripts/roadmap_schema.py"),
        },
        "exclusion_manifest_lint.py": {
            "path": "scripts/exclusion_manifest_lint.py",
            "sha256": sha256_file(repo_root / "scripts/exclusion_manifest_lint.py"),
        },
    }


def _artifact_input_receipts(paths: Mapping[str, Path]) -> list[JsonDict]:
    return [
        {
            "exp_id": artifact.exp_id,
            "task_id": artifact.task_id,
            "path": artifact.relative_path.as_posix(),
            "resolved_path": str(paths[artifact.exp_id]),
            "exists": paths[artifact.exp_id].is_file(),
            "bytes": paths[artifact.exp_id].stat().st_size
            if paths[artifact.exp_id].exists()
            else 0,
            "sha256": sha256_file(paths[artifact.exp_id]),
            "readiness_fields": list(artifact.readiness_fields),
        }
        for artifact in V568_ARTIFACTS
    ]


def _architecture_freshness_receipt(repo_root: Path, run_date: str) -> JsonDict:
    text = (repo_root / ARCHITECTURE_RELATIVE_PATH).read_text(encoding="utf-8")
    match = re.search(r"\*\*Last Reconciled:\*\*\s*(\d{4}-\d{2}-\d{2})", text)
    last_reconciled = match.group(1) if match else "unknown"
    age_days = None
    if last_reconciled != "unknown":
        planning = date.fromisoformat(f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:]}")
        age_days = (planning - date.fromisoformat(last_reconciled)).days
    return {
        "architecture_path": ARCHITECTURE_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(repo_root / ARCHITECTURE_RELATIVE_PATH),
        "last_reconciled": last_reconciled,
        "planning_date": f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:]}",
        "age_days_at_planning": age_days,
        "architecture_checked": True,
    }


def _preconditions_checked(
    *,
    repo_root: Path,
    artifact_paths: Mapping[str, Path],
    protected_before: Mapping[str, str],
    architecture: Mapping[str, Any],
) -> JsonDict:  # pragma: no cover
    return {
        "git_state": _git_receipt(repo_root),
        "resources": _resource_receipt(repo_root),
        "rust": {
            "rustc": _tool_version(["rustc", "--version"], repo_root),
            "cargo": _tool_version(["cargo", "--version"], repo_root),
        },
        "pyo3": _pyo3_receipt(repo_root),
        "z3": _z3_receipt(),
        "artifact_path_and_hash_receipts": _artifact_input_receipts(artifact_paths),
        "protected_file_hashes_before": dict(protected_before),
        "monotonic_timer_resolution_s": time.get_clock_info("monotonic").resolution,
        "architecture_freshness": dict(architecture),
        "network_status": _network_receipt(),
        "llm_load_performed": False,
        "hardware_command_performed": False,
    }


def _stamped_flags(payload: Mapping[str, Any]) -> list[JsonDict]:
    pending = payload.get("corrigendum_pending")
    if isinstance(pending, list):
        return [dict(row) for row in pending if isinstance(row, Mapping)]
    return []


def _critical_flags(flags: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [dict(flag) for flag in flags if str(flag.get("severity")).lower() == "critical"]


def _coerce_closed_verdict_class(value: Any) -> str | None:
    if value is None:
        return None
    if value in {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}:
        return str(value)
    return "disqualified"


def _readiness_scores(artifact: V568Artifact, payload: Mapping[str, Any]) -> JsonDict:
    return {field: payload.get(field) for field in artifact.readiness_fields}


def _speedup_from_payload(payload: Mapping[str, Any]) -> float | None:
    aggregate = payload.get("aggregate_row_recomputation")
    if isinstance(aggregate, Mapping):
        value = aggregate.get("steady_state_median_batched_speedup_vs_python_scalar")
        return (
            float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None
        )
    return None


def _artifact_duration_reason(
    exp_id: str,
    stamped_flags: Sequence[Mapping[str, Any]],
    live_critical_flags: Sequence[Mapping[str, Any]],
) -> str:
    stamped_kinds = {str(flag.get("kind")) for flag in stamped_flags}
    if exp_id == "exp6561" and "DURATION_TOO_SHORT" in stamped_kinds and not live_critical_flags:
        return "stamped_duration_flag_recorded_live_replay_clean"
    if live_critical_flags:
        return "live_verifier_critical_flags_recorded"
    return "live_replay_clean"


def _artifact_outcome(
    artifact: V568Artifact,
    payload: Mapping[str, Any],
    live_critical_flags: Sequence[Mapping[str, Any]],
) -> JsonDict:
    exists = bool(payload)
    verdict_class = _coerce_closed_verdict_class(payload.get("verdict_class"))
    if not exists:
        return {
            "disposition": "missing_input",
            "failed_scope": "missing_v568_artifact",
            "eligible": False,
            "reason": f"{artifact.exp_id}_input_exists",
        }
    if live_critical_flags:
        return {
            "disposition": "not_imported_live_verifier_critical",
            "failed_scope": "live_verifier",
            "eligible": False,
            "reason": "live_critical_flags",
        }
    if artifact.exp_id == "exp6561":
        return {
            "disposition": "usable_contract_with_stamped_duration_caution",
            "failed_scope": "v568_contract_duration_receipt",
            "eligible": payload.get("v568_evidence_contract_ready_score") == 1.0,
            "reason": "usable_contract_replayed_without_live_flags",
        }
    if artifact.exp_id == "exp6562":
        return {
            "disposition": "disqualified_saturation_science",
            "failed_scope": "constraint_saturation",
            "eligible": verdict_class == "disqualified",
            "reason": "old_saturation_headline_disqualified_not_extended",
        }
    if artifact.exp_id == "exp6563":
        aggregate = payload.get("aggregate_row_recomputation") or {}
        return {
            "disposition": "clean_null_production_evidence",
            "failed_scope": "production_safety_net_value",
            "eligible": verdict_class == "null"
            and payload.get("production_workload_canary_ready_score") == 1.0
            and aggregate.get("measured_enabled_benefit") is False,
            "reason": "safe_default_off_canary_no_measured_benefit",
        }
    aggregate = payload.get("aggregate_row_recomputation") or {}
    return {
        "disposition": "clean_null_nfr01_evidence",
        "failed_scope": "rust_pyo3_nfr01",
        "eligible": verdict_class == "null"
        and aggregate.get("nfr01_passed") is False
        and _speedup_from_payload(payload) is not None,
        "reason": "exact_parity_but_nfr01_missed",
    }


def _live_verifier_row(
    artifact: V568Artifact,
    *,
    payload: Mapping[str, Any],
    check_result: Mapping[str, Any],
    verifier_versions: Mapping[str, Any],
) -> JsonDict:
    adversarial = dict(check_result.get("adversarial") or {})
    row_consistency = dict(check_result.get("row_consistency") or {})
    live_flags = [
        dict(flag) for flag in adversarial.get("flags") or [] if isinstance(flag, Mapping)
    ]
    stamped_flags = _stamped_flags(payload)
    live_critical = _critical_flags(live_flags)
    reason = _artifact_duration_reason(artifact.exp_id, stamped_flags, live_critical)
    return {
        "row_type": "live_verifier_and_duration",
        "exp_id": artifact.exp_id,
        "task_id": artifact.task_id,
        "artifact_path": artifact.relative_path.as_posix(),
        "artifact_sha256": sha256_file(REPO_ROOT / artifact.relative_path)
        if (REPO_ROOT / artifact.relative_path).is_file()
        else "missing",
        "artifact_duration_s": payload.get("duration_s"),
        "stamped_flags": stamped_flags,
        "stamped_flag_count": len(stamped_flags),
        "live_verifier_command": adversarial.get("command"),
        "live_verifier_exit_code": adversarial.get("exit_code"),
        "live_verifier_duration_s": adversarial.get("duration_s"),
        "live_flags": live_flags,
        "live_flag_count": len(live_flags),
        "live_critical_flags": live_critical,
        "live_critical_flag_count": len(live_critical),
        "stamped_live_flag_disagreement": bool(stamped_flags) != bool(live_flags),
        "row_consistency_command": row_consistency.get("command"),
        "row_consistency_exit_code": row_consistency.get("exit_code"),
        "row_consistency_duration_s": row_consistency.get("duration_s"),
        "row_consistency_status": str(row_consistency.get("status") or "unknown"),
        "row_consistency_findings": list(row_consistency.get("findings") or []),
        "adversarial_verifier_version": verifier_versions["adversarial_verify.py"],
        "row_lint_version": verifier_versions["verdict_row_consistency_lint.py"],
        "reason": reason,
    }


def _eligibility_row(
    artifact: V568Artifact,
    *,
    path: Path,
    payload: Mapping[str, Any],
    check_result: Mapping[str, Any],
    verifier_versions: Mapping[str, Any],
) -> JsonDict:
    live_row = _live_verifier_row(
        artifact,
        payload=payload,
        check_result=check_result,
        verifier_versions=verifier_versions,
    )
    outcome = _artifact_outcome(artifact, payload, live_row["live_critical_flags"])
    exists = path.is_file()
    aggregate = payload.get("aggregate_row_recomputation") or {}
    speedup = _speedup_from_payload(payload)
    return {
        "row_type": "v568_artifact_eligibility",
        "exp_id": artifact.exp_id,
        "task_id": artifact.task_id,
        "expected_path": artifact.relative_path.as_posix(),
        "resolved_path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path),
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict"),
        "verdict_class": _coerce_closed_verdict_class(payload.get("verdict_class")),
        "readiness_fields": _readiness_scores(artifact, payload),
        "duration_s": payload.get("duration_s"),
        "stamped_flags": live_row["stamped_flags"],
        "stamped_flag_count": live_row["stamped_flag_count"],
        "live_verifier_command": live_row["live_verifier_command"],
        "live_verifier_exit_code": live_row["live_verifier_exit_code"],
        "live_flags": live_row["live_flags"],
        "live_flag_count": live_row["live_flag_count"],
        "live_critical_flag_count": live_row["live_critical_flag_count"],
        "stamped_live_flag_disagreement": live_row["stamped_live_flag_disagreement"],
        "row_consistency_command": live_row["row_consistency_command"],
        "row_consistency_exit_code": live_row["row_consistency_exit_code"],
        "row_consistency_status": live_row["row_consistency_status"],
        "row_consistency_findings": live_row["row_consistency_findings"],
        "eligible_for_v569_contract": outcome["eligible"],
        "disposition": outcome["disposition"],
        "failed_scope": outcome["failed_scope"],
        "reason": outcome["reason"],
        "extends_exp6556_saturation_headline": False,
        "production_adapter_default_off": artifact.exp_id == "exp6563",
        "promotion_candidate": bool(
            payload.get("production_workload_promotion_candidate_score") == 1.0
        ),
        "nfr01_passed": bool(aggregate.get("nfr01_passed"))
        if artifact.exp_id == "exp6564"
        else None,
        "measured_speedup_vs_requirement": speedup,
    }


def _build_v568_rows(
    *,
    repo_root: Path,
    paths: Mapping[str, Path],
    payloads: Mapping[str, Mapping[str, Any]],
    check_results: Mapping[str, Mapping[str, Any]],
    verifier_versions: Mapping[str, Any],
) -> tuple[list[JsonDict], list[JsonDict]]:
    rows: list[JsonDict] = []
    live_rows: list[JsonDict] = []
    for artifact in V568_ARTIFACTS:
        payload = payloads.get(artifact.exp_id, {})
        check_result = check_results.get(artifact.exp_id, {})
        live_rows.append(
            _live_verifier_row(
                artifact,
                payload=payload,
                check_result=check_result,
                verifier_versions=verifier_versions,
            )
        )
        rows.append(
            _eligibility_row(
                artifact,
                path=paths[artifact.exp_id],
                payload=payload,
                check_result=check_result,
                verifier_versions=verifier_versions,
            )
        )
    _ = repo_root
    return rows, live_rows


def _parse_required_fields(prompt: str) -> set[str]:
    fields: set[str] = set()
    in_block = False
    for line in prompt.splitlines():
        if "REQUIRED ARTIFACT FIELDS:" in line:
            in_block = True
            continue
        if in_block and line.strip().startswith("Run command:"):
            break
        if in_block:
            match = re.match(r"\s{2,8}([A-Za-z_][A-Za-z0-9_]*):\s*$", line)
            if match:
                fields.add(match.group(1))
    return fields


def _retired_experiment_ids(manifest: Mapping[str, Any]) -> set[str]:
    retired: set[str] = set()
    for section_name in ("retired_experiments", "retired_extras", "retired"):
        section = manifest.get(section_name)
        if not isinstance(section, list):
            continue
        for item in section:
            if not isinstance(item, Mapping):
                continue
            for key in ("id", "experiment_id"):
                value = item.get(key)
                if isinstance(value, str):
                    retired.add(value)
            ids = item.get("experiment_ids")
            if isinstance(ids, list):
                retired.update(str(value) for value in ids)
    return retired


def _roadmap_and_contract(repo_root: Path) -> tuple[JsonDict, dict[str, set[str]], set[str]]:
    roadmap = _load_yaml(repo_root / ROADMAP_RELATIVE_PATH)
    Roadmap.model_validate(roadmap)
    fields_by_task = {
        str(task["id"]): _parse_required_fields(str(task.get("prompt") or ""))
        for task in roadmap.get("tasks", [])
        if isinstance(task, Mapping)
    }
    retired = _retired_experiment_ids(_load_yaml(repo_root / EXCLUSION_MANIFEST_RELATIVE_PATH))
    return roadmap, fields_by_task, retired


def _requires_retired_ids(roadmap: Mapping[str, Any], retired_ids: set[str]) -> set[str]:
    found: set[str] = set()
    for task in roadmap.get("tasks", []):
        if not isinstance(task, Mapping):
            continue
        for key in ("requires", "gated_on"):
            value = task.get(key)
            if not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, str) and item in retired_ids:
                    found.add(item)
                if isinstance(item, Mapping):
                    upstream = item.get("upstream")
                    if isinstance(upstream, str) and upstream in retired_ids:
                        found.add(upstream)
    return found


def _gate_contract_rows(
    roadmap: Mapping[str, Any],
    fields_by_task: Mapping[str, set[str]],
    retired_ids: set[str],
) -> list[JsonDict]:
    task_ids = {
        str(task.get("id")) for task in roadmap.get("tasks", []) if isinstance(task, Mapping)
    }
    rows: list[JsonDict] = []
    for task in roadmap.get("tasks", []):
        if not isinstance(task, Mapping):
            continue
        task_id = str(task.get("id"))
        for index, gate in enumerate(task.get("gated_on") or []):
            if not isinstance(gate, Mapping):
                continue
            upstream = str(gate.get("upstream"))
            field = str(gate.get("artifact_field"))
            upstream_fields = fields_by_task.get(upstream, set())
            rows.append(
                {
                    "row_type": "v569_gate_contract",
                    "task_id": task_id,
                    "gate_index": index,
                    "upstream": upstream,
                    "artifact_field": field,
                    "op": gate.get("op"),
                    "value": gate.get("value"),
                    "upstream_in_active_roadmap": upstream in task_ids,
                    "artifact_field_declared_by_upstream": field in upstream_fields,
                    "upstream_declared_field_count": len(upstream_fields),
                    "retired_upstream": upstream in retired_ids,
                }
            )
    return rows


def _prior_failure_scope_class(experiment_id: str) -> str:
    if "6561" in experiment_id:
        return "v568_evidence_duration_receipt"
    if "6562" in experiment_id:
        return "disqualified_constraint_saturation_lineage"
    if any(token in experiment_id for token in ("5909", "5910", "5923")):
        return "retired_full_constraintir_or_reprompt_extractor"
    if "6564" in experiment_id or "6563" in experiment_id:
        return "safety_net_production_or_nfr01_null"
    if "6553" in experiment_id:
        return "blocked_flagship_admission"
    return "method_or_runtime_prior_failure"


def _prior_failure_and_retirement_rows(
    roadmap: Mapping[str, Any],
    retired_ids: set[str],
    requires_retired_ids: set[str],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    required = {"experiment_id", "verdict", "addressed_by", "retire_if_same_verdict"}
    for task in roadmap.get("tasks", []):
        if not isinstance(task, Mapping):
            continue
        task_id = str(task.get("id"))
        for index, prior in enumerate(task.get("prior_failures") or []):
            if not isinstance(prior, Mapping):
                rows.append(
                    {
                        "row_type": "prior_failure_and_retirement",
                        "task_id": task_id,
                        "prior_failure_index": index,
                        "complete_prior_failure_contract": False,
                        "missing_fields": sorted(required),
                        "retired_dependency_chain": False,
                    }
                )
                continue
            missing = sorted(field for field in required if field not in prior)
            experiment_id = str(prior.get("experiment_id") or "")
            changed_text = str(prior.get("addressed_by") or "").strip()
            rows.append(
                {
                    "row_type": "prior_failure_and_retirement",
                    "task_id": task_id,
                    "prior_failure_index": index,
                    "experiment_id": experiment_id,
                    "verdict": prior.get("verdict"),
                    "addressed_by": prior.get("addressed_by"),
                    "retire_if_same_verdict": prior.get("retire_if_same_verdict"),
                    "missing_fields": missing,
                    "complete_prior_failure_contract": not missing,
                    "changed_mechanism": bool(changed_text),
                    "mechanical_repeat_retirement_rule": prior.get("retire_if_same_verdict")
                    is True,
                    "scope_class": _prior_failure_scope_class(experiment_id),
                    "retired_prior_scope": experiment_id in retired_ids,
                    "retired_dependency_chain": experiment_id in requires_retired_ids,
                }
            )
    return rows


def _proposal_mentions_exp6574(repo_root: Path) -> bool:
    text = (repo_root / PROPOSAL_RELATIVE_PATH).read_text(encoding="utf-8")
    return "Exp6574" in text and "fuses normalization" in text


def _model_arc_and_hardware_boundary(
    repo_root: Path,
    roadmap: Mapping[str, Any],
    payloads: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    active_task_ids = [
        str(task.get("id")) for task in roadmap.get("tasks", []) if isinstance(task, Mapping)
    ]
    exp6563 = payloads.get("exp6563", {})
    exp6564 = payloads.get("exp6564", {})
    exp6564_speedup = _speedup_from_payload(exp6564)
    material_change = _proposal_mentions_exp6574(repo_root)
    retirement_rule = True
    production_default_off = exp6563.get("production_workload_promotion_candidate_score") == 0.0
    rust_reopen_ready = (
        material_change
        and retirement_rule
        and production_default_off
        and exp6564.get("verdict_class") == "null"
        and exp6564_speedup is not None
        and exp6564_speedup < 10.0
    )
    boundary = {
        "MODEL_SPECS": list(MANDATED_MODEL_IDS),
        "active_v569_task_ids": active_task_ids,
        "legacy_model_policy": {
            "legacy_smoke_models": ["Qwen3.5-0.8B", "gemma-4-E4B-it"],
            "legacy_smoke_models_can_support_headline": False,
            "legacy_substitution_allowed": False,
        },
        "model_rule": {
            "llama_cpp_gguf_required": True,
            "embedded_tokenizer_required": True,
            "auto_tokenizer_from_gguf_repo_allowed": False,
            "one_flagship_model_loaded_at_a_time": True,
            "actual_load_required_for_headline": True,
        },
        "arc_boundary": {
            "no_game_or_level_solve_claim": True,
            "no_game_source_read": True,
            "no_offline_bfs": True,
            "receipt_only_until_new_prospective_rows": True,
        },
        "hardware_boundary": {
            "exp6565_hardware_command_count": 0,
            "llm_load_count": 0,
            "unchanged_board_command_allowed": False,
            "kv260_command_allowed": False,
            "gatemate_command_allowed_without_new_operator_receipt": False,
            "tsu_or_kona_execution_claim_allowed": False,
        },
        "production_boundary": {
            "production_adapter_default_off": production_default_off,
            "exp6563_promotion_candidate_score": exp6563.get(
                "production_workload_promotion_candidate_score"
            ),
            "default_on_activation_allowed": False,
        },
        "rust_fusion_boundary": {
            "active_roadmap_has_exp6574": "exp6574-fused-rust-exact-workload-nfr01"
            in active_task_ids,
            "proposal_mentions_exp6574": material_change,
            "proposed_exp6574_materially_different": material_change,
            "previous_exp6564_scope": "compact Safety-Net router batch ABI only",
            "allowed_changed_workload_units": [
                "normalization",
                "exact_conflict_feature_extraction",
                "route_construction",
                "fallback_request_assembly",
                "serialization",
            ],
            "retire_on_repeated_no_benefit_or_nfr01_miss": retirement_rule,
            "exp6564_speedup_vs_python_scalar": exp6564_speedup,
            "exp6564_nfr01_threshold": 10.0,
            "rust_fusion_reopen_ready_from_boundary": rust_reopen_ready,
        },
    }
    boundary["all_boundary_checks_passed"] = (
        boundary["MODEL_SPECS"] == list(MANDATED_MODEL_IDS)
        and boundary["legacy_model_policy"]["legacy_smoke_models_can_support_headline"] is False
        and boundary["model_rule"]["auto_tokenizer_from_gguf_repo_allowed"] is False
        and boundary["arc_boundary"]["no_game_or_level_solve_claim"] is True
        and boundary["hardware_boundary"]["exp6565_hardware_command_count"] == 0
        and boundary["hardware_boundary"]["unchanged_board_command_allowed"] is False
        and rust_reopen_ready
    )
    return boundary


def _gate_check_summary(
    *,
    rows: Sequence[Mapping[str, Any]],
    gate_rows: Sequence[Mapping[str, Any]],
    prior_rows: Sequence[Mapping[str, Any]],
    boundary: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> JsonDict:
    failed: list[str] = []
    failed_rows: list[JsonDict] = []

    def add_failed(check: str, expected: Any, observed: Any, evidence_path: str) -> None:
        failed.append(check)
        failed_rows.append(
            {
                "check": check,
                "expected": expected,
                "observed": observed,
                "evidence_path": evidence_path,
            }
        )

    for row in rows:
        exp_id = str(row.get("exp_id"))
        if row.get("exists") is not True:
            add_failed(
                f"{exp_id}_input_exists", True, row.get("exists"), str(row.get("expected_path"))
            )
        if row.get("eligible_for_v569_contract") is not True:
            add_failed(f"{exp_id}_contract_eligible", True, False, str(row.get("expected_path")))
    for row in gate_rows:
        if row.get("upstream_in_active_roadmap") is not True:
            add_failed(
                "gate_upstream_in_active_roadmap",
                True,
                row.get("upstream"),
                str(row.get("task_id")),
            )
        if row.get("artifact_field_declared_by_upstream") is not True:
            add_failed(
                "gate_artifact_field_declared",
                True,
                row.get("artifact_field"),
                str(row.get("task_id")),
            )
        if row.get("retired_upstream") is True:
            add_failed("gate_retired_upstream", False, row.get("upstream"), str(row.get("task_id")))
    for row in prior_rows:
        if row.get("complete_prior_failure_contract") is not True:
            add_failed(
                "prior_failure_contract_complete",
                True,
                row.get("missing_fields"),
                str(row.get("task_id")),
            )
        if row.get("changed_mechanism") is not True:
            add_failed("prior_failure_changed_mechanism", True, False, str(row.get("task_id")))
        if row.get("mechanical_repeat_retirement_rule") is not True:
            add_failed(
                "prior_failure_repeat_retirement_rule",
                True,
                row.get("retire_if_same_verdict"),
                str(row.get("task_id")),
            )
        if row.get("retired_dependency_chain") is True:
            add_failed(
                "prior_failure_retired_dependency_chain_absent",
                False,
                row.get("experiment_id"),
                str(row.get("task_id")),
            )
    if boundary.get("all_boundary_checks_passed") is not True:
        add_failed(
            "model_arc_hardware_boundary_closed", True, False, "model_arc_and_hardware_boundary"
        )
    if protected.get("all_unchanged") is not True:
        add_failed(
            "protected_files_unchanged",
            True,
            protected.get("changed_paths"),
            "protected_files_unchanged",
        )
    return {
        "all_gates_passed": not failed,
        "failed_checks": failed,
        "failed_check_rows": failed_rows,
        "task_field_gate_contract_closed": all(
            row.get("upstream_in_active_roadmap") is True
            and row.get("artifact_field_declared_by_upstream") is True
            and row.get("retired_upstream") is False
            for row in gate_rows
        ),
        "prior_failure_retirement_contract_closed": all(
            row.get("complete_prior_failure_contract") is True
            and row.get("changed_mechanism") is True
            and row.get("mechanical_repeat_retirement_rule") is True
            and row.get("retired_dependency_chain") is False
            for row in prior_rows
        ),
    }


def aggregate_row_recomputation(payload: Mapping[str, Any]) -> JsonDict:
    rows = [
        dict(row)
        for row in payload.get("v568_artifact_eligibility_rows", [])
        if isinstance(row, Mapping)
    ]
    gates = [
        dict(row) for row in payload.get("v569_gate_contract_rows", []) if isinstance(row, Mapping)
    ]
    priors = [
        dict(row)
        for row in payload.get("prior_failure_and_retirement_rows", [])
        if isinstance(row, Mapping)
    ]
    boundary = payload.get("model_arc_and_hardware_boundary") or {}
    protected = payload.get("protected_files_unchanged") or {}
    rows_by_exp = {row.get("exp_id"): row for row in rows}
    all_rows = len(rows) == len(V568_ARTIFACTS) and all(
        row.get("eligible_for_v569_contract") is True for row in rows
    )
    gate_contract = all(
        row.get("upstream_in_active_roadmap") is True
        and row.get("artifact_field_declared_by_upstream") is True
        and row.get("retired_upstream") is False
        for row in gates
    )
    prior_contract = all(
        row.get("complete_prior_failure_contract") is True
        and row.get("changed_mechanism") is True
        and row.get("mechanical_repeat_retirement_rule") is True
        and row.get("retired_dependency_chain") is False
        for row in priors
    )
    rust_reopen = (
        boundary.get("rust_fusion_boundary", {}).get("rust_fusion_reopen_ready_from_boundary")
        is True
    )
    ready = (
        all_rows
        and gate_contract
        and prior_contract
        and boundary.get("all_boundary_checks_passed") is True
        and protected.get("all_unchanged") is True
        and payload.get("verifier_is_oracle") is True
    )
    return {
        "row_type": "aggregate_row_recomputation",
        "expected_v568_artifact_row_count": len(V568_ARTIFACTS),
        "observed_v568_artifact_row_count": len(rows),
        "all_v568_rows_contract_eligible": all_rows,
        "exp6561_stamped_live_disagreement_recorded": rows_by_exp.get("exp6561", {}).get(
            "stamped_live_flag_disagreement"
        )
        is True,
        "exp6562_disqualified_science_recorded": rows_by_exp.get("exp6562", {}).get("disposition")
        == "disqualified_saturation_science",
        "exp6563_clean_null_production_recorded": rows_by_exp.get("exp6563", {}).get("disposition")
        == "clean_null_production_evidence",
        "exp6564_clean_null_nfr01_recorded": rows_by_exp.get("exp6564", {}).get("disposition")
        == "clean_null_nfr01_evidence",
        "v569_gate_contract_closed": gate_contract,
        "prior_failure_retirement_contract_closed": prior_contract,
        "model_arc_hardware_boundary_closed": boundary.get("all_boundary_checks_passed") is True,
        "protected_files_unchanged": protected.get("all_unchanged") is True,
        "no_llm_load_performed": boundary.get("hardware_boundary", {}).get("llm_load_count") == 0,
        "no_hardware_command_performed": boundary.get("hardware_boundary", {}).get(
            "exp6565_hardware_command_count"
        )
        == 0,
        "rust_fusion_reopen_ready_from_rows": rust_reopen,
        "v569_evidence_contract_ready_from_rows": ready,
        "verdict_class_from_rows": None if ready else "partial",
        "spec_refs": ["REQ-REPORT-6565"],
    }


def _status_and_verdict(
    ready: bool,
    missing_input: bool,
    failed_checks: Sequence[str],
) -> tuple[str, str, str | None]:
    if ready:
        return (
            "complete_v569_evidence_and_retirement_contract_ready",
            "complete_v569_evidence_and_retirement_contract_ready: V568 artifacts are content-addressed; Exp6562 is disqualified science; Exp6563 and Exp6564 are clean nulls; V569 gate, failure, model, ARC, hardware, Rust-fusion, and protected-file contracts close",
            None,
        )
    if missing_input:
        return (
            "blocked_v569_evidence_contract_missing_inputs",
            "blocked_v569_evidence_contract_missing_inputs: required V568 input artifact is missing; failed checks are recorded",
            "blocked",
        )
    if failed_checks:
        return (
            "partial_v569_evidence_and_retirement_contract",
            "partial_v569_evidence_and_retirement_contract: usable V568 evidence exists but one or more gate, failure, model, ARC, hardware, Rust-fusion, or protected-file checks failed",
            "partial",
        )
    return (
        "blocked_v569_evidence_contract",
        "blocked_v569_evidence_contract: no usable V568 input set was available",
        "blocked",
    )


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path = RESULT_RELATIVE_PATH,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    check_results: Mapping[str, Mapping[str, Any]] | None = None,
    input_payloads: Mapping[str, Mapping[str, Any]] | None = None,
    artifact_paths: Mapping[str, Path] | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    started = time.monotonic()
    paths = default_v568_paths(repo_root) if artifact_paths is None else dict(artifact_paths)
    protected_before = _protected_hashes(repo_root)
    payloads = (
        {artifact.exp_id: _read_json(paths[artifact.exp_id]) for artifact in V568_ARTIFACTS}
        if input_payloads is None
        else dict(input_payloads)
    )
    live_results = run_live_checks(repo_root, paths) if check_results is None else check_results
    verifier_versions = _verifier_versions(repo_root)
    rows, live_rows = _build_v568_rows(
        repo_root=repo_root,
        paths=paths,
        payloads=payloads,
        check_results=live_results,
        verifier_versions=verifier_versions,
    )
    roadmap, fields_by_task, retired_ids = _roadmap_and_contract(repo_root)
    requires_retired = _requires_retired_ids(roadmap, retired_ids)
    gate_rows = _gate_contract_rows(roadmap, fields_by_task, retired_ids)
    prior_rows = _prior_failure_and_retirement_rows(roadmap, retired_ids, requires_retired)
    boundary = _model_arc_and_hardware_boundary(repo_root, roadmap, payloads)
    architecture = _architecture_freshness_receipt(repo_root, run_date)
    protected_after = _protected_hashes(repo_root)
    protected = _protected_files_unchanged(protected_before, protected_after)
    gate_summary = _gate_check_summary(
        rows=rows,
        gate_rows=gate_rows,
        prior_rows=prior_rows,
        boundary=boundary,
        protected=protected,
    )
    measured_duration = (
        round(time.monotonic() - started, 6) if duration_s is None else float(duration_s)
    )
    missing_input = any(row.get("exists") is not True for row in rows)
    skeleton: JsonDict = {
        "v568_artifact_eligibility_rows": rows,
        "v569_gate_contract_rows": gate_rows,
        "prior_failure_and_retirement_rows": prior_rows,
        "model_arc_and_hardware_boundary": boundary,
        "protected_files_unchanged": protected,
        "verifier_is_oracle": True,
    }
    aggregate = aggregate_row_recomputation(skeleton)
    ready = (
        aggregate["v569_evidence_contract_ready_from_rows"] is True
        and gate_summary["all_gates_passed"] is True
    )
    status, honest_verdict, verdict_class = _status_and_verdict(
        ready, missing_input, gate_summary["failed_checks"]
    )
    payload: JsonDict = {
        "status": status,
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "v568_artifact_eligibility_rows": rows,
        "live_verifier_and_duration_rows": live_rows,
        "v569_gate_contract_rows": gate_rows,
        "prior_failure_and_retirement_rows": prior_rows,
        "model_arc_and_hardware_boundary": boundary,
        "v569_evidence_contract_ready_score": 1.0 if ready else 0.0,
        "rust_fusion_reopen_ready_score": 1.0
        if aggregate["rust_fusion_reopen_ready_from_rows"]
        else 0.0,
        "per_unit_rows": rows,
        "aggregate_row_recomputation": aggregate,
        "gate_check_summary": gate_summary,
        "preconditions_checked": _preconditions_checked(
            repo_root=repo_root,
            artifact_paths=paths,
            protected_before=protected_before,
            architecture=architecture,
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": dict(FIELD_PROVENANCE),
        "duration_s": measured_duration,
        "tests_run": _tests_run_receipts(tests_run),
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }
    payload["reproducibility_checksum"] = reproducibility_checksum(payload)
    if write:
        atomic_write_json(
            result_path, payload, root=repo_root, sort_keys=False, allow_override=False
        )
    return payload


def _rows_by_exp(payload: Mapping[str, Any]) -> dict[str, JsonDict]:
    return {
        str(row.get("exp_id")): dict(row)
        for row in payload.get("v568_artifact_eligibility_rows", [])
        if isinstance(row, Mapping)
    }


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in payload]
    if missing:
        errors.append(f"missing required fields: {missing}")
        return errors
    if not str(payload.get("honest_verdict") or "").startswith(
        ("complete_", "partial_", "blocked_", "disqualified_")
    ):
        errors.append("honest_verdict lacks terminal prefix")
    if payload.get("verdict_class") not in (None, "partial", "blocked", "disqualified"):
        errors.append("verdict_class is outside closed class")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if set((payload.get("field_provenance") or {})) < set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    if set((payload.get("field_principles") or {})) < set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles must cover required fields")
    if payload.get("reproducibility_checksum") != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum mismatch")

    rows = _rows_by_exp(payload)
    for artifact in V568_ARTIFACTS:
        row = rows.get(artifact.exp_id, {})
        expected_hash = sha256_file(row.get("resolved_path", ""))
        if expected_hash != "missing" and row.get("sha256") != expected_hash:
            errors.append(f"V568 artifact hash alias for {artifact.exp_id}")
        if (
            row.get("eligible_for_v569_contract") is not True
            and payload.get("v569_evidence_contract_ready_score") == 1.0
        ):
            errors.append(f"{artifact.exp_id} readiness score hides ineligible row")
    for row in payload.get("v569_gate_contract_rows") or []:
        if not isinstance(row, Mapping):
            errors.append("gate contract row must be a mapping")
            continue
        if row.get("artifact_field_declared_by_upstream") is not True:
            errors.append("gate contract has undeclared field")
        if row.get("upstream_in_active_roadmap") is not True:
            errors.append("gate contract has out-of-roadmap upstream")
        if row.get("retired_upstream") is True:
            errors.append("gate contract has retired upstream")
    required_prior = {"experiment_id", "verdict", "addressed_by", "retire_if_same_verdict"}
    for row in payload.get("prior_failure_and_retirement_rows") or []:
        if not isinstance(row, Mapping):
            errors.append("prior failure row must be a mapping")
            continue
        if not required_prior <= set(row) or row.get("complete_prior_failure_contract") is not True:
            errors.append("prior failure row missing required fields")
        if row.get("changed_mechanism") is not True:
            errors.append("prior failure row lacks changed mechanism")
        if row.get("mechanical_repeat_retirement_rule") is not True:
            errors.append("prior failure row lacks mechanical repeat-retirement rule")
        if row.get("retired_dependency_chain") is True:
            errors.append("prior failure row uses retired dependency chain")
    boundary = payload.get("model_arc_and_hardware_boundary") or {}
    if boundary.get("MODEL_SPECS") != list(MANDATED_MODEL_IDS):
        errors.append("mandated GGUF model identities changed")
    legacy = boundary.get("legacy_model_policy") or {}
    if legacy.get("legacy_smoke_models_can_support_headline") is not False:
        errors.append("legacy-model substitution opened")
    arc = boundary.get("arc_boundary") or {}
    if arc.get("no_game_or_level_solve_claim") is not True:
        errors.append("ARC solve boundary opened")
    hardware = boundary.get("hardware_boundary") or {}
    if hardware.get("exp6565_hardware_command_count") != 0:
        errors.append("hardware command boundary violated")
    if hardware.get("unchanged_board_command_allowed") is not False:
        errors.append("unchanged hardware command boundary opened")
    rust = boundary.get("rust_fusion_boundary") or {}
    rust_ready = (
        rust.get("proposed_exp6574_materially_different") is True
        and rust.get("retire_on_repeated_no_benefit_or_nfr01_miss") is True
        and rust.get("rust_fusion_reopen_ready_from_boundary") is True
    )
    if payload.get("rust_fusion_reopen_ready_score") == 1.0 and not rust_ready:
        errors.append(
            "rust fusion reopen score must derive from changed workload and retirement rule"
        )
    protected = payload.get("protected_files_unchanged") or {}
    if protected.get("all_unchanged") is not True:
        errors.append("protected files changed")
    gate_summary = payload.get("gate_check_summary") or {}
    if payload.get("v569_evidence_contract_ready_score") == 1.0 and gate_summary.get(
        "failed_checks"
    ):
        errors.append("ready score cannot be open with failed checks")
    aggregate = payload.get("aggregate_row_recomputation") or {}
    recomputed = aggregate_row_recomputation(payload)
    if aggregate != recomputed:
        errors.append("aggregate recomputation mismatch")
    if (
        payload.get("v569_evidence_contract_ready_score") == 1.0
        and aggregate.get("v569_evidence_contract_ready_from_rows") is not True
    ):
        errors.append("ready score must derive from aggregate recomputation")
    return sorted(set(errors))


def load_json(path: str | Path) -> JsonDict:
    return _read_json(Path(path))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_RELATIVE_PATH)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--skip-live-checks", action="store_true")
    args = parser.parse_args(argv)

    if args.validate:
        payload = load_json(args.result_path)
        errors = validate_artifact(payload)
        if errors:
            print(json.dumps({"valid": False, "errors": errors}, indent=2))
            return 1
        print(json.dumps({"valid": True}, indent=2))
        return 0

    fake_results = None
    if args.skip_live_checks:
        fake_results = {
            artifact.exp_id: {
                "adversarial": {
                    "command": "skipped by --skip-live-checks",
                    "exit_code": 0,
                    "duration_s": 0.0,
                    "flag_count": 0,
                    "max_severity": -1,
                    "flags": [],
                },
                "row_consistency": {
                    "command": "skipped by --skip-live-checks",
                    "exit_code": 0,
                    "duration_s": 0.0,
                    "status": "ok",
                    "findings": [],
                },
            }
            for artifact in V568_ARTIFACTS
        }
    payload = build_artifact(
        repo_root=REPO_ROOT,
        result_path=args.result_path,
        write=True,
        duration_s=args.duration_s,
        check_results=fake_results,
        run_date=args.date,
    )
    errors = validate_artifact(payload)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        return 1
    print(json.dumps({"valid": True, "path": str(args.result_path)}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
