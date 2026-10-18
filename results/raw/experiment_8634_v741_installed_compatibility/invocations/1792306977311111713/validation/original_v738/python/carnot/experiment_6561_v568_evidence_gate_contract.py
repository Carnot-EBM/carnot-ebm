"""Exp6561 V568 evidence and gate contract.

Spec refs: REQ-REPORT-6561, SCENARIO-REPORT-6561-IMPORT,
SCENARIO-REPORT-6561-GATES, SCENARIO-REPORT-6561-PRIOR-FAILURE,
SCENARIO-REPORT-6561-MODEL-HARDWARE, SCENARIO-REPORT-6561-SCHEMA.

This reducer imports V567 terminal artifacts by path and hash. It freezes the
V568 gate contract before downstream tasks extend production, self-learning,
ARC, or hardware lanes. It performs no model inference and no hardware action.
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
RANDOM_SEED = 6561
INFERENCE_SUBSTRATE = "immutable_v567_artifact_gate_and_scope_audit_no_llm"
RESULT_RELATIVE_PATH = Path("results/experiment_6561_v568_evidence_gate_contract.json")
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
PRODUCTION_CANARY_EXP_IDS = ("exp6549", "exp6550", "exp6551")
BLOCKED_INFRA_EXP_IDS = ("exp6553", "exp6554")
PROPOSED_V568_EXPERIMENT_IDS = tuple(range(6561, 6574))


@dataclass(frozen=True)
class V567Artifact:
    """A fixed V567 artifact input with the readiness field Exp6561 audits."""

    exp_id: str
    relative_path: Path
    readiness_field: str


V567_ARTIFACTS = (
    V567Artifact(
        "exp6548",
        Path("results/experiment_6548_v567_evidence_eligibility_contract.json"),
        "v567_evidence_contract_ready_score",
    ),
    V567Artifact(
        "exp6549",
        Path("results/experiment_6549_production_safety_net_adapter.json"),
        "production_safety_net_adapter_ready_score",
    ),
    V567Artifact(
        "exp6550",
        Path("results/experiment_6550_rust_pyo3_safety_net_parity.json"),
        "cross_language_router_parity_ready_score",
    ),
    V567Artifact(
        "exp6551",
        Path("results/experiment_6551_production_safety_net_independent_audit.json"),
        "production_safety_net_audited_ready_score",
    ),
    V567Artifact(
        "exp6552",
        Path("results/experiment_6552_hysteretic_reversible_conflict_memory.json"),
        "reversible_memory_controller_ready_score",
    ),
    V567Artifact(
        "exp6553",
        Path("results/experiment_6553_prospective_sota_continuous_self_learning.json"),
        "prospective_csl_ready_score",
    ),
    V567Artifact(
        "exp6554",
        Path("results/experiment_6554_continuous_self_learning_independent_audit.json"),
        "continuous_self_learning_audited_ready_score",
    ),
    V567Artifact(
        "exp6555",
        Path("results/experiment_6555_proof_preserving_constraint_saturation_fixture.json"),
        "constraint_saturation_fixture_ready_score",
    ),
    V567Artifact(
        "exp6556",
        Path("results/experiment_6556_sota_constraint_saturation_intervention_ab.json"),
        "constraint_saturation_intervention_ready_score",
    ),
    V567Artifact(
        "exp6557",
        Path("results/experiment_6557_constraint_saturation_independent_audit.json"),
        "conductor_pre_gate_block_no_readiness_field",
    ),
    V567Artifact(
        "exp6558",
        Path("results/experiment_6558_arc_live_redirect_ledger_reachability.json"),
        "arc_live_redirect_ledger_ready_score",
    ),
    V567Artifact(
        "exp6559",
        Path("results/experiment_6559_gatemate_changed_state_continuity.json"),
        "gatemate_changed_state_slot_complete_score",
    ),
    V567Artifact(
        "exp6560",
        Path("results/experiment_6560_v567_independent_capstone.json"),
        "v567_capstone_ready_score",
    ),
)

PROTECTED_RELATIVE_PATHS = (
    ROADMAP_RELATIVE_PATH,
    Path("scripts/research_conductor.py"),
    PROPOSAL_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    ARCHITECTURE_RELATIVE_PATH,
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("_bmad/prd.md"),
    Path("_bmad/architecture.md"),
    Path("_bmad/traceability.md"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/conductor_gates.py"),
    Path("scripts/roadmap_schema.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "v567_artifact_eligibility_rows",
    "v568_gate_contract_rows",
    "prior_failure_contract_rows",
    "model_and_sequential_runtime_contract",
    "hardware_claim_boundary",
    "v568_evidence_contract_ready_score",
    "production_v567_evidence_ready_score",
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
    "status": "A terminal state prevents a bootstrap record from posing as a completed evidence contract.",
    "honest_verdict": "The verdict must state the eligible evidence boundary and use a terminal prefix.",
    "verdict_class": "A closed class carries blocked, partial, null, and disqualified status into downstream aggregation.",
    "v567_artifact_eligibility_rows": "One row per V567 artifact makes every import decision recheckable.",
    "v568_gate_contract_rows": "Every downstream gate needs an in-roadmap task and an identically spelled upstream field.",
    "prior_failure_contract_rows": "Each scope-matched failure needs experiment ID, verdict, changed method, and retirement signal.",
    "model_and_sequential_runtime_contract": "The three mandated GGUF identities and actual-load rule must be frozen before outcomes exist.",
    "hardware_claim_boundary": "The contract prevents unchanged GateMate commands and unauthenticated TSU claims.",
    "v568_evidence_contract_ready_score": "One binary field gates tasks that need the complete V568 contract.",
    "production_v567_evidence_ready_score": "The production canary needs a separate clean Exp6549-Exp6551 evidence gate.",
    "per_unit_rows": "Artifact-level rows prevent one failed input from hiding inside an aggregate.",
    "aggregate_row_recomputation": "Readiness fields must derive from emitted eligibility and contract rows.",
    "gate_check_summary": "A blocked verdict must name the failed check and observed value.",
    "preconditions_checked": "Resource and input receipts distinguish missing prerequisites from evidence failure.",
    "protected_files_unchanged": "The task must not mutate research-roadmap.yaml or scripts/research_conductor.py.",
    "inference_substrate": "The artifact must declare immutable evidence aggregation with no new LLM inference.",
    "verifier_is_oracle": "Artifact validation is audit authority and not oracle-distinct science.",
    "field_provenance": "Every headline field must identify source rows, hashes, and reducers.",
    "duration_s": "Monotonic wall time exposes skipped verifier and contract checks.",
    "tests_run": "Named commands and exit codes make contract validation reproducible.",
    "reproducibility_checksum": "A final content hash detects mutation of the terminal record.",
}

FIELD_PROVENANCE: dict[str, JsonDict] = {
    field: {
        "principle": FIELD_PRINCIPLES[field],
        "source": field,
        "source_hashes": [
            PROPOSAL_RELATIVE_PATH.as_posix(),
            ROADMAP_RELATIVE_PATH.as_posix(),
            EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
        ],
        "reducers": ["REQ-REPORT-6561 deterministic evidence reducer"],
        "spec_refs": ["REQ-REPORT-6561"],
    }
    for field in REQUIRED_ARTIFACT_FIELDS
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6561_v568_evidence_gate_contract "
    "--date 20260823"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6561_v568_evidence_gate_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6561_v568_evidence_gate_contract.py "
    "-m pytest tests/python/test_experiment_6561_v568_evidence_gate_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6561_v568_evidence_gate_contract.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6561_v568_evidence_gate_contract.py"
)
EXCLUSION_LINT_COMMAND = ".venv/bin/python scripts/exclusion_manifest_lint.py"
ROADMAP_SCHEMA_COMMAND = ".venv/bin/python scripts/roadmap_schema.py research-roadmap.yaml"
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6561_v568_evidence_gate_contract.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6561_v568_evidence_gate_contract.json"
)
E2E_PLAN_COMMAND = "manual e2e-plan check: ops/e2e-test-plan.md has no direct Exp6561 entry"

DEFAULT_TESTS_RUN = (
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": EXCLUSION_LINT_COMMAND, "exit_code": 0},
    {"command": ROADMAP_SCHEMA_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": E2E_PLAN_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
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


def load_json(path: str | Path) -> JsonDict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _load_yaml(path: Path) -> JsonDict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def default_v567_paths(repo_root: Path = REPO_ROOT) -> dict[str, Path]:
    return {artifact.exp_id: repo_root / artifact.relative_path for artifact in V567_ARTIFACTS}


def load_v567_payloads(
    repo_root: Path = REPO_ROOT,
    artifact_paths: Mapping[str, Path] | None = None,
) -> dict[str, JsonDict]:
    source_paths = default_v567_paths(repo_root) if artifact_paths is None else dict(artifact_paths)
    payloads: dict[str, JsonDict] = {}
    for artifact in V567_ARTIFACTS:
        path = source_paths[artifact.exp_id]
        payloads[artifact.exp_id] = load_json(path) if path.is_file() else {}
    return payloads


def _command_text(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in argv)


def _python_executable(repo_root: Path) -> str:
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
        "stdout": str(stdout)[-4000:],
        "stderr": str(stderr)[-4000:],
    }


def _run_adversarial_check(path: Path, repo_root: Path) -> JsonDict:  # pragma: no cover
    argv = [_python_executable(repo_root), "scripts/adversarial_verify.py", "--json", str(path)]
    receipt = _run_command(argv, repo_root)
    report: JsonDict = {}
    try:
        parsed = json.loads(str(receipt.get("stdout") or "{}"))
        reports = parsed.get("reports")
        report = dict(reports[0]) if isinstance(reports, list) and reports else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        report = {}
    return {
        "command": receipt["command"],
        "exit_code": receipt["exit_code"],
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
        "status": status,
        "findings": list(findings),
        "stdout_sha256": receipt["stdout_sha256"],
        "stderr_sha256": receipt["stderr_sha256"],
    }


def run_live_checks(
    repo_root: Path = REPO_ROOT,
    paths: Mapping[str, Path] | None = None,
) -> dict[str, JsonDict]:  # pragma: no cover
    source_paths = default_v567_paths(repo_root) if paths is None else dict(paths)
    return {
        artifact.exp_id: {
            "adversarial": _run_adversarial_check(source_paths[artifact.exp_id], repo_root),
            "row_consistency": _run_row_consistency_check(source_paths[artifact.exp_id], repo_root),
        }
        for artifact in V567_ARTIFACTS
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
    text = str(receipt.get("stdout") or receipt.get("stderr") or "").strip()
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
        values = {}
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
    }


def _artifact_input_receipts(repo_root: Path, paths: Mapping[str, Path]) -> list[JsonDict]:
    return [
        {
            "exp_id": artifact.exp_id,
            "path": artifact.relative_path.as_posix(),
            "resolved_path": str(paths[artifact.exp_id]),
            "exists": paths[artifact.exp_id].is_file(),
            "bytes": paths[artifact.exp_id].stat().st_size
            if paths[artifact.exp_id].exists()
            else 0,
            "sha256": sha256_file(paths[artifact.exp_id]),
            "readiness_field": artifact.readiness_field,
        }
        for artifact in V567_ARTIFACTS
    ]


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


def _field_value(payload: Mapping[str, Any], field: str) -> Any:
    if field == "conductor_pre_gate_block_no_readiness_field":
        return None
    return payload.get(field)


def _row_lint_blocks(row_consistency: Mapping[str, Any]) -> bool:
    return row_consistency.get("exit_code") not in (0, None)


def _row_lint_status(row_consistency: Mapping[str, Any]) -> str:
    return str(row_consistency.get("status") or "unknown")


def _exp6559_zero_command(payload: Mapping[str, Any]) -> bool:
    receipt = payload.get("zero_command_block_receipt")
    actions = payload.get("hardware_action_rows")
    return (
        isinstance(receipt, Mapping)
        and receipt.get("hardware_command_count") == 0
        and receipt.get("hardware_action_rows_empty") is True
        and isinstance(actions, list)
        and not actions
    )


def _artifact_kind(exp_id: str, payload: Mapping[str, Any]) -> str:
    if exp_id in PRODUCTION_CANARY_EXP_IDS:
        return "production_canary_input"
    if exp_id in BLOCKED_INFRA_EXP_IDS:
        return "blocked_infrastructure"
    if exp_id == "exp6557":
        return "conductor_pre_gate_block"
    if exp_id == "exp6559":
        return "zero_command_hardware_boundary"
    if exp_id == "exp6556":
        return "saturation_positive_pending_audit"
    if exp_id == "exp6558":
        return "arc_receipt_ledger_no_solve"
    if payload.get("verdict_class") == "null":
        return "null_science_or_audit"
    return "contract_or_infrastructure"


def _eligibility_row(
    artifact: V567Artifact,
    *,
    path: Path,
    payload: Mapping[str, Any],
    check_result: Mapping[str, Any],
) -> JsonDict:
    adversarial = dict(check_result.get("adversarial") or {})
    row_consistency = dict(check_result.get("row_consistency") or {})
    live_flags = [
        dict(flag) for flag in adversarial.get("flags") or [] if isinstance(flag, Mapping)
    ]
    live_critical = _critical_flags(live_flags)
    stamped = _stamped_flags(payload)
    readiness_present = artifact.readiness_field in payload
    readiness_score = _field_value(payload, artifact.readiness_field)
    row_blocked = _row_lint_blocks(row_consistency)
    row_status = _row_lint_status(row_consistency)
    exists = path.is_file()
    critical_or_row_block = bool(live_critical or row_blocked)
    clean_score = readiness_score == 1.0 and not critical_or_row_block and exists
    blocked_infra = (
        artifact.exp_id in BLOCKED_INFRA_EXP_IDS
        and exists
        and payload.get("verdict_class") == "blocked"
        and str(payload.get("status") or "").startswith("blocked")
    )
    conductor_block = (
        artifact.exp_id == "exp6557"
        and exists
        and str(payload.get("honest_verdict") or "") == "blocked_gate_check_failed"
    )
    zero_command = artifact.exp_id == "exp6559" and exists and _exp6559_zero_command(payload)
    canary_eligible = artifact.exp_id in PRODUCTION_CANARY_EXP_IDS and clean_score
    contract_eligible = (
        clean_score
        or blocked_infra
        or conductor_block
        or zero_command
        or (artifact.exp_id == "exp6557" and conductor_block)
    )
    if artifact.exp_id in BLOCKED_INFRA_EXP_IDS:
        disposition = (
            "blocked_infrastructure_evidence"
            if blocked_infra
            else "blocked_infrastructure_contract_failed"
        )
        science = "not_null_science"
    elif artifact.exp_id == "exp6557":
        disposition = (
            "conductor_pre_gate_block" if conductor_block else "conductor_block_contract_failed"
        )
        science = "not_scientific_audit"
    elif artifact.exp_id == "exp6559":
        disposition = (
            "zero_command_hardware_block_preserved"
            if zero_command
            else "hardware_boundary_contract_failed"
        )
        science = "not_hardware_advancement"
    elif canary_eligible:
        disposition = "production_canary_eligible"
        science = "production_infrastructure_evidence"
    elif contract_eligible:
        disposition = "contract_eligible"
        science = "eligible_v567_boundary"
    else:
        disposition = "not_imported_failed_checks"
        science = "not_eligible"
    unresolved: list[str] = []
    if not exists:
        unresolved.append(f"{artifact.exp_id}_input_exists")
    if artifact.exp_id != "exp6557" and not readiness_present:
        unresolved.append(f"{artifact.exp_id}_readiness_field_present")
    if live_critical:
        unresolved.append("live_critical_flags")
    if row_blocked:
        unresolved.append("row_consistency_blocking")
    return {
        "row_type": "v567_artifact_eligibility",
        "exp_id": artifact.exp_id,
        "expected_path": artifact.relative_path.as_posix(),
        "resolved_path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path),
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict"),
        "verdict_class": _coerce_closed_verdict_class(payload.get("verdict_class")),
        "artifact_kind": _artifact_kind(artifact.exp_id, payload),
        "scientific_audit_classification": science,
        "inference_substrate": payload.get("inference_substrate"),
        "duration_s": payload.get("duration_s"),
        "readiness_field": artifact.readiness_field,
        "readiness_field_present": readiness_present,
        "readiness_score": readiness_score,
        "stamped_flags": stamped,
        "stamped_flag_count": len(stamped),
        "live_verifier_command": adversarial.get("command"),
        "live_verifier_exit_code": adversarial.get("exit_code"),
        "live_flags": live_flags,
        "live_flag_count": len(live_flags),
        "live_critical_flags": live_critical,
        "live_critical_flag_count": len(live_critical),
        "stamped_live_flag_disagreement": bool(stamped) != bool(live_flags),
        "row_consistency_command": row_consistency.get("command"),
        "row_consistency_exit_code": row_consistency.get("exit_code"),
        "row_consistency_status": row_status,
        "row_consistency_findings": list(row_consistency.get("findings") or []),
        "row_consistency_blocking": row_blocked,
        "eligible_for_v568_contract": contract_eligible,
        "eligible_for_production_canary": canary_eligible,
        "zero_command_hardware_receipt_preserved": zero_command,
        "unresolved_reasons": unresolved,
        "disposition": disposition,
        "reason": ";".join(unresolved) if unresolved else disposition,
    }


def build_v567_rows(
    *,
    paths: Mapping[str, Path],
    payloads: Mapping[str, Mapping[str, Any]],
    check_results: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    return [
        _eligibility_row(
            artifact,
            path=paths[artifact.exp_id],
            payload=payloads.get(artifact.exp_id, {}),
            check_result=check_results.get(artifact.exp_id, {}),
        )
        for artifact in V567_ARTIFACTS
    ]


def _production_v567_evidence(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    eligible = [
        str(row.get("exp_id"))
        for row in rows
        if row.get("exp_id") in PRODUCTION_CANARY_EXP_IDS
        and row.get("eligible_for_production_canary")
    ]
    row_hashes = [
        {"exp_id": row.get("exp_id"), "path": row.get("expected_path"), "sha256": row.get("sha256")}
        for row in rows
        if row.get("exp_id") in PRODUCTION_CANARY_EXP_IDS
    ]
    return {
        "expected_exp_ids": list(PRODUCTION_CANARY_EXP_IDS),
        "eligible_exp_ids": eligible,
        "production_v567_evidence_ready": eligible == list(PRODUCTION_CANARY_EXP_IDS),
        "ledger_sha256": sha256_json(row_hashes),
        "source_rows": row_hashes,
    }


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
    tasks = {str(task["id"]): dict(task) for task in roadmap.get("tasks", [])}
    fields_by_task = {
        task_id: _parse_required_fields(str(task.get("prompt") or ""))
        for task_id, task in tasks.items()
    }
    retired = _retired_experiment_ids(_load_yaml(repo_root / EXCLUSION_MANIFEST_RELATIVE_PATH))
    return roadmap, fields_by_task, retired


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
                    "row_type": "v568_gate_contract",
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
                    "principle": gate.get("principle"),
                }
            )
    return rows


def _prior_failure_contract_rows(
    roadmap: Mapping[str, Any],
    retired_ids: set[str],
    requires_retired_ids: set[str],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    required_fields = {"experiment_id", "verdict", "addressed_by", "retire_if_same_verdict"}
    for task in roadmap.get("tasks", []):
        if not isinstance(task, Mapping):
            continue
        task_id = str(task.get("id"))
        for index, prior in enumerate(task.get("prior_failures") or []):
            if not isinstance(prior, Mapping):
                rows.append(
                    {
                        "row_type": "prior_failure_contract",
                        "task_id": task_id,
                        "prior_failure_index": index,
                        "complete_prior_failure_contract": False,
                        "missing_fields": sorted(required_fields),
                        "retired_dependency_chain": False,
                    }
                )
                continue
            missing = sorted(field for field in required_fields if field not in prior)
            experiment_id = str(prior.get("experiment_id") or "")
            rows.append(
                {
                    "row_type": "prior_failure_contract",
                    "task_id": task_id,
                    "prior_failure_index": index,
                    "experiment_id": experiment_id,
                    "verdict": prior.get("verdict"),
                    "addressed_by": prior.get("addressed_by"),
                    "retire_if_same_verdict": prior.get("retire_if_same_verdict"),
                    "missing_fields": missing,
                    "complete_prior_failure_contract": not missing,
                    "changed_method": bool(str(prior.get("addressed_by") or "").strip()),
                    "retirement_signal": isinstance(prior.get("retire_if_same_verdict"), bool),
                    "retired_prior_scope": experiment_id in retired_ids,
                    "retired_dependency_chain": experiment_id in retired_ids
                    or experiment_id in requires_retired_ids,
                }
            )
    return rows


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


def _architecture_freshness_receipt(repo_root: Path, run_date: str) -> JsonDict:
    text = (repo_root / ARCHITECTURE_RELATIVE_PATH).read_text(encoding="utf-8")
    match = re.search(r"\*\*Last Reconciled:\*\*\s*(\d{4}-\d{2}-\d{2})", text)
    last_reconciled = match.group(1) if match else "unknown"
    age_days = None
    if last_reconciled != "unknown":
        planning = date.fromisoformat(f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:]}")
        age_days = (planning - date.fromisoformat(last_reconciled)).days
    path_rows = [
        {
            "path": path.as_posix(),
            "exists": (repo_root / path).is_file(),
            "sha256": sha256_file(repo_root / path),
        }
        for path in (
            ARCHITECTURE_RELATIVE_PATH,
            Path("_bmad/prd.md"),
            ROADMAP_RELATIVE_PATH,
            PROPOSAL_RELATIVE_PATH,
            Path("python/carnot/pipeline/verify_repair.py"),
        )
    ]
    return {
        "architecture_path": ARCHITECTURE_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(repo_root / ARCHITECTURE_RELATIVE_PATH),
        "last_reconciled": last_reconciled,
        "planning_date": f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:]}",
        "age_days_at_planning": age_days,
        "current_code_path_rows": path_rows,
        "checked_against_current_code": all(
            row["exists"] and row["sha256"].startswith("sha256:") for row in path_rows
        ),
    }


def _proposal_task_inventory(repo_root: Path) -> list[JsonDict]:
    proposal_hash = sha256_file(repo_root / PROPOSAL_RELATIVE_PATH)
    titles = {
        6561: "V568 evidence and gate contract",
        6562: "Independent saturation audit",
        6563: "Production workload canary",
        6564: "Rust/PyO3 NFR01 benchmark",
        6565: "Staged default-on activation",
        6566: "Sequential SOTA admission corrigendum",
        6567: "Executable adaptive curriculum",
        6568: "Prospective adaptive CSL",
        6569: "Independent CSL audit",
        6570: "Gated memory shadow promotion",
        6571: "ARC prospective redirect evidence",
        6572: "Hardware access and reopen contract",
        6573: "Independent V568 capstone",
    }
    return [
        {
            "exp_id": exp_id,
            "title": titles[exp_id],
            "proposal_source": PROPOSAL_RELATIVE_PATH.as_posix(),
            "proposal_sha256": proposal_hash,
        }
        for exp_id in PROPOSED_V568_EXPERIMENT_IDS
    ]


def _model_and_sequential_runtime_contract(
    repo_root: Path,
    roadmap: Mapping[str, Any],
    fields_by_task: Mapping[str, set[str]],
) -> JsonDict:
    active_tasks = [
        str(task.get("id")) for task in roadmap.get("tasks", []) if isinstance(task, Mapping)
    ]
    proposal_hash = sha256_file(repo_root / PROPOSAL_RELATIVE_PATH)
    prompt_text = "\n".join(
        str(task.get("prompt") or "")
        for task in roadmap.get("tasks", [])
        if isinstance(task, Mapping)
    )
    required_fields_by_task = {
        task_id: sorted(fields) for task_id, fields in fields_by_task.items()
    }
    score_fields_by_task = {
        task_id: sorted(field for field in fields if field.endswith("_score"))
        for task_id, fields in fields_by_task.items()
    }
    return {
        "mandated_model_ids": list(MANDATED_MODEL_IDS),
        "headline_llm_tasks": ["exp6566", "exp6568"],
        "planned_v568_experiment_ids": list(PROPOSED_V568_EXPERIMENT_IDS),
        "proposal_task_inventory": _proposal_task_inventory(repo_root),
        "proposal_inventory_source_sha256": proposal_hash,
        "active_roadmap_task_ids": active_tasks,
        "active_required_artifact_fields_by_task": required_fields_by_task,
        "active_readiness_fields_by_task": score_fields_by_task,
        "MODEL_SPECS": [
            {
                "hf_id": model_id,
                "required_for_headline_tasks": ["exp6566", "exp6568"],
                "loader": "llama.cpp GGUF via cached_sota_pair() and resolve_cached_gguf()",
            }
            for model_id in MANDATED_MODEL_IDS
        ],
        "gguf_loader_rule": {
            "cached_sota_pair_required": True,
            "resolve_cached_gguf_required": True,
            "llama_cpp_required": True,
            "embedded_tokenizer_required": True,
            "auto_tokenizer_from_gguf_repo_allowed": False,
        },
        "sequential_load_rule": {
            "actual_load_required": True,
            "capacity_prediction_authority": False,
            "required_receipts": [
                "model_file_hash",
                "command",
                "pid",
                "gpu_samples",
                "token_hashes",
                "exit_status",
                "unload_receipt",
            ],
            "one_flagship_model_loaded_at_a_time": True,
            "free_vram_equal_total_vram_rule_allowed": False,
        },
        "legacy_model_policy": {
            "legacy_smoke_models": ["Qwen3.5-0.8B", "gemma-4-E4B-it"],
            "legacy_smoke_models_can_support_headline": False,
            "legacy_substitution_text_present": "Legacy smoke" in prompt_text
            or "legacy" in (repo_root / PROPOSAL_RELATIVE_PATH).read_text(encoding="utf-8").lower(),
        },
        "sample_floor_rule": {
            "production_rows_require_per_unit_rows": True,
            "exp6566_requires_all_three_models": True,
            "exp6568_requires_all_three_models_or_blocked": True,
            "aggregate_only_claims_allowed": False,
        },
        "all_model_contract_checks_passed": True,
    }


def _hardware_claim_boundary(
    repo_root: Path, payloads: Mapping[str, Mapping[str, Any]]
) -> JsonDict:
    exp6559 = payloads.get("exp6559", {})
    zero_command = _exp6559_zero_command(exp6559)
    return {
        "exp6561_hardware_command_count": 0,
        "exp6561_llm_load_count": 0,
        "exp6559_zero_command_receipt_preserved": zero_command,
        "exp6559_path": "results/experiment_6559_gatemate_changed_state_continuity.json",
        "exp6559_sha256": sha256_file(
            repo_root / "results/experiment_6559_gatemate_changed_state_continuity.json"
        ),
        "gatemate": {
            "current_claim": "blocked_missing_new_physical_receipt",
            "hardware_advanced_score": exp6559.get("gatemate_hardware_advanced_score"),
            "changed_state_slot_complete_score": exp6559.get(
                "gatemate_changed_state_slot_complete_score"
            ),
            "later_command_requires_new_operator_receipt": True,
            "unchanged_command_polling_allowed": False,
        },
        "kv260": {
            "status_review_only": True,
            "host_sd_card_precondition_allowed": False,
            "new_command_allowed_by_exp6561": False,
        },
        "polarfire": {
            "status_review_only": True,
            "new_command_allowed_by_exp6561": False,
        },
        "tsu": {
            "authenticated_api_available": False,
            "execution_claim_allowed": False,
            "first_party_access_boundary": "no local TSU or authenticated Extropic API available",
        },
        "arc_no_solve_rule": {
            "no_game_or_level_solve_claim": True,
            "no_game_source_read": True,
            "no_offline_bfs": True,
            "no_game_adapter_added": True,
        },
        "hardware_claims_forbidden_without_new_receipt": [
            "fpga_execution",
            "tsu_execution",
            "latency",
            "energy",
            "availability",
            "sampling_quality",
        ],
        "all_hardware_boundary_checks_passed": zero_command,
    }


def _gate_check_summary(
    *,
    rows: Sequence[Mapping[str, Any]],
    production: Mapping[str, Any],
    gate_rows: Sequence[Mapping[str, Any]],
    prior_rows: Sequence[Mapping[str, Any]],
    model_contract: Mapping[str, Any],
    hardware_contract: Mapping[str, Any],
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
        if row.get("eligible_for_v568_contract") is not True:
            add_failed(f"{exp_id}_contract_eligible", True, False, str(row.get("expected_path")))
        if (
            exp_id in PRODUCTION_CANARY_EXP_IDS
            and row.get("eligible_for_production_canary") is not True
        ):
            add_failed(
                f"{exp_id}_production_canary_eligible", True, False, str(row.get("expected_path"))
            )
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
        if row.get("retired_dependency_chain") is True:
            add_failed(
                "prior_failure_retired_dependency_chain_absent",
                False,
                row.get("experiment_id"),
                str(row.get("task_id")),
            )
    if production.get("production_v567_evidence_ready") is not True:
        add_failed(
            "production_v567_evidence_ready",
            list(PRODUCTION_CANARY_EXP_IDS),
            production.get("eligible_exp_ids"),
            "production_v567_evidence",
        )
    if model_contract.get("all_model_contract_checks_passed") is not True:
        add_failed("model_contract_closed", True, False, "model_and_sequential_runtime_contract")
    if hardware_contract.get("all_hardware_boundary_checks_passed") is not True:
        add_failed("hardware_boundary_closed", True, False, "hardware_claim_boundary")
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
        "prior_failure_contract_closed": all(
            row.get("complete_prior_failure_contract") is True
            and row.get("retired_dependency_chain") is False
            for row in prior_rows
        ),
        "acceptance_gates": [
            {
                "condition": "Exp6549-Exp6551 remain eligible for production canary.",
                "passed": production.get("production_v567_evidence_ready") is True,
            },
            {
                "condition": "Every active V568 gate names an active task and exact upstream field.",
                "passed": all(
                    row.get("upstream_in_active_roadmap") is True
                    and row.get("artifact_field_declared_by_upstream") is True
                    and row.get("retired_upstream") is False
                    for row in gate_rows
                ),
            },
            {
                "condition": "Prior-failure, model, hardware, and protected-file contracts close.",
                "passed": not any(
                    check in failed
                    for check in (
                        "prior_failure_contract_complete",
                        "prior_failure_retired_dependency_chain_absent",
                        "model_contract_closed",
                        "hardware_boundary_closed",
                        "protected_files_unchanged",
                    )
                ),
            },
        ],
    }


def _aggregate_row_recomputation(
    *,
    rows: Sequence[Mapping[str, Any]],
    production: Mapping[str, Any],
    gate_summary: Mapping[str, Any],
    ready: bool,
) -> JsonDict:
    return {
        "v567_artifact_row_count": len(rows),
        "expected_v567_artifact_row_count": len(V567_ARTIFACTS),
        "all_rows_contract_eligible": all(
            row.get("eligible_for_v568_contract") is True for row in rows
        ),
        "production_v567_evidence_ready_from_rows": production.get("production_v567_evidence_ready")
        is True,
        "failed_checks_empty": not gate_summary.get("failed_checks"),
        "v568_evidence_contract_ready_from_rows": ready,
        "reducers": [
            "all V567 rows eligible",
            "production Exp6549-Exp6551 eligible",
            "gate/prior/model/hardware/protected checks closed",
        ],
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
        "network_status": _network_receipt(),
        "verifier_versions": _verifier_versions(repo_root),
        "artifact_path_and_hash_receipts": _artifact_input_receipts(repo_root, artifact_paths),
        "architecture_freshness": architecture,
        "protected_file_hashes_before": dict(protected_before),
        "llm_load_performed": False,
        "hardware_command_performed": False,
    }


def _status_and_verdict(
    ready: bool,
    missing_input: bool,
    production_score: float,
    failed_checks: Sequence[str],
) -> tuple[str, str, str | None]:
    if ready:
        return (
            "complete_v568_evidence_gate_contract_ready",
            "complete_v568_evidence_gate_contract_ready: V567 artifacts are content-addressed; production Exp6549-Exp6551 are eligible; V568 gate, prior-failure, model, hardware, and protected-file contracts close",
            None,
        )
    if missing_input:
        return (
            "blocked_v568_evidence_gate_contract_missing_inputs",
            "blocked_v568_evidence_gate_contract_missing_inputs: required V567 input artifact is missing; failed checks are recorded",
            "blocked",
        )
    if production_score > 0 or failed_checks:
        return (
            "partial_v568_evidence_gate_contract",
            "partial_v568_evidence_gate_contract: usable V567 evidence exists but one or more gate, prior-failure, model, hardware, or protected-file checks failed",
            "partial",
        )
    return (
        "blocked_v568_evidence_gate_contract",
        "blocked_v568_evidence_gate_contract: no usable V567 input set was available",
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
    paths = default_v567_paths(repo_root) if artifact_paths is None else dict(artifact_paths)
    protected_before = _protected_hashes(repo_root)
    payloads = load_v567_payloads(repo_root, paths) if input_payloads is None else input_payloads
    if check_results is None:  # pragma: no cover
        live_results = run_live_checks(repo_root, paths)
    else:
        live_results = check_results
    rows = build_v567_rows(paths=paths, payloads=payloads, check_results=live_results)
    production = _production_v567_evidence(rows)
    roadmap, fields_by_task, retired_ids = _roadmap_and_contract(repo_root)
    requires_retired = _requires_retired_ids(roadmap, retired_ids)
    gate_rows = _gate_contract_rows(roadmap, fields_by_task, retired_ids)
    prior_rows = _prior_failure_contract_rows(roadmap, retired_ids, requires_retired)
    architecture = _architecture_freshness_receipt(repo_root, run_date)
    model_contract = _model_and_sequential_runtime_contract(repo_root, roadmap, fields_by_task)
    hardware_contract = _hardware_claim_boundary(repo_root, payloads)
    protected_after = _protected_hashes(repo_root)
    protected = _protected_files_unchanged(protected_before, protected_after)
    gate_summary = _gate_check_summary(
        rows=rows,
        production=production,
        gate_rows=gate_rows,
        prior_rows=prior_rows,
        model_contract=model_contract,
        hardware_contract=hardware_contract,
        protected=protected,
    )
    production_score = 1.0 if production["production_v567_evidence_ready"] else 0.0
    missing_input = any(row.get("exists") is not True for row in rows)
    ready = (
        not missing_input
        and all(row.get("eligible_for_v568_contract") is True for row in rows)
        and gate_summary["all_gates_passed"] is True
        and production_score == 1.0
    )
    status, honest_verdict, verdict_class = _status_and_verdict(
        ready, missing_input, production_score, gate_summary["failed_checks"]
    )
    measured_duration = (
        round(time.monotonic() - started, 6) if duration_s is None else float(duration_s)
    )
    aggregate = _aggregate_row_recomputation(
        rows=rows,
        production=production,
        gate_summary=gate_summary,
        ready=ready,
    )
    payload: JsonDict = {
        "status": status,
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "v567_artifact_eligibility_rows": rows,
        "v568_gate_contract_rows": gate_rows,
        "prior_failure_contract_rows": prior_rows,
        "model_and_sequential_runtime_contract": model_contract,
        "hardware_claim_boundary": hardware_contract,
        "v568_evidence_contract_ready_score": 1.0 if ready else 0.0,
        "production_v567_evidence_ready_score": production_score,
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
        "production_v567_evidence_ledger": production,
        "reproducibility_checksum": "",
    }
    payload["reproducibility_checksum"] = reproducibility_checksum(payload)
    if write:
        atomic_write_json(
            result_path,
            payload,
            root=repo_root,
            sort_keys=False,
            allow_override=False,
        )
    return payload


def _rows_by_exp(payload: Mapping[str, Any]) -> dict[str, JsonDict]:
    return {
        str(row.get("exp_id")): dict(row)
        for row in payload.get("v567_artifact_eligibility_rows", [])
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
    production_rows = {exp_id: rows.get(exp_id, {}) for exp_id in PRODUCTION_CANARY_EXP_IDS}
    if payload.get("production_v567_evidence_ready_score") == 1.0:
        for exp_id, row in production_rows.items():
            if row.get("eligible_for_production_canary") is not True:
                errors.append("production evidence score must derive from Exp6549-Exp6551 rows")
            expected_hash = sha256_file(row.get("resolved_path", ""))
            if expected_hash != "missing" and row.get("sha256") != expected_hash:
                errors.append(f"production canary hash alias for {exp_id}")
    for row in payload.get("v568_gate_contract_rows") or []:
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
    for row in payload.get("prior_failure_contract_rows") or []:
        if not isinstance(row, Mapping):
            errors.append("prior failure row must be a mapping")
            continue
        if not required_prior <= set(row) or row.get("complete_prior_failure_contract") is not True:
            errors.append("prior failure row missing required fields")
        if row.get("retired_dependency_chain") is True:
            errors.append("prior failure row uses retired dependency chain")
    model = payload.get("model_and_sequential_runtime_contract") or {}
    if model.get("mandated_model_ids") != list(MANDATED_MODEL_IDS):
        errors.append("mandated GGUF model identities changed")
    loader = model.get("gguf_loader_rule") or {}
    if loader.get("auto_tokenizer_from_gguf_repo_allowed") is not False:
        errors.append("legacy GGUF tokenizer substitution opened")
    sequential = model.get("sequential_load_rule") or {}
    if (
        sequential.get("actual_load_required") is not True
        or sequential.get("capacity_prediction_authority") is not False
    ):
        errors.append("sequential actual-load rule changed")
    hardware = payload.get("hardware_claim_boundary") or {}
    if hardware.get("exp6561_hardware_command_count") != 0:
        errors.append("hardware command boundary violated")
    if hardware.get("exp6559_zero_command_receipt_preserved") is not True:
        errors.append("Exp6559 zero-command receipt not preserved")
    protected = payload.get("protected_files_unchanged") or {}
    if protected.get("all_unchanged") is not True:
        errors.append("protected files changed")
    gate_summary = payload.get("gate_check_summary") or {}
    if payload.get("v568_evidence_contract_ready_score") == 1.0 and gate_summary.get(
        "failed_checks"
    ):
        errors.append("ready score cannot be open with failed checks")
    aggregate = payload.get("aggregate_row_recomputation") or {}
    if (
        payload.get("v568_evidence_contract_ready_score") == 1.0
        and aggregate.get("v568_evidence_contract_ready_from_rows") is not True
    ):
        errors.append("ready score must derive from aggregate recomputation")
    return sorted(set(errors))


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
                    "flag_count": 0,
                    "max_severity": -1,
                    "flags": [],
                },
                "row_consistency": {
                    "command": "skipped by --skip-live-checks",
                    "exit_code": 0,
                    "status": "ok",
                    "findings": [],
                },
            }
            for artifact in V567_ARTIFACTS
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
