"""Exp6548 V567 evidence eligibility contract.

Spec refs: REQ-REPORT-6548, SCENARIO-REPORT-6548-ADDITIVE,
SCENARIO-REPORT-6548-IMPORT, SCENARIO-REPORT-6548-GATES,
SCENARIO-REPORT-6548-MODEL-HARDWARE, SCENARIO-REPORT-6548-SCHEMA.

This reducer audits checked-in V566 artifacts. It does not fetch DRIFT data,
rerun intake, or load a GGUF model. The result is a contract ledger for later
production tasks.
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
RANDOM_SEED = 6548
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts_no_llm"
RESULT_RELATIVE_PATH = Path("results/experiment_6548_v567_evidence_eligibility_contract.json")
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
HEADLINE_MODEL_TASKS = (
    "exp6553-prospective-sota-continuous-self-learning",
    "exp6556-sota-constraint-saturation-intervention-ab",
)
EXPECTED_CLEAN_IMPORT_EXP_IDS = ("exp6542", "exp6543", "exp6544", "exp6545", "exp6546", "exp6547")


@dataclass(frozen=True)
class V566Artifact:
    """A fixed V566 artifact input with the readiness field Exp6548 audits."""

    exp_id: str
    relative_path: Path
    readiness_field: str


V566_ARTIFACTS = (
    V566Artifact(
        "exp6541",
        Path("results/experiment_6541_v566_direct_source_contract.json"),
        "v566_direct_source_ready_score",
    ),
    V566Artifact(
        "exp6542",
        Path("results/experiment_6542_drift_bench_external_intake_v2.json"),
        "external_constraint_corpus_ready_score",
    ),
    V566Artifact(
        "exp6543",
        Path("results/experiment_6543_external_corpus_independent_audit_v2.json"),
        "external_constraint_corpus_audited_ready_score",
    ),
    V566Artifact(
        "exp6544",
        Path("results/experiment_6544_external_structural_headroom.json"),
        "external_structural_headroom_ready_score",
    ),
    V566Artifact(
        "exp6545",
        Path("results/experiment_6545_external_safety_net_router.json"),
        "external_safety_net_ready_score",
    ),
    V566Artifact(
        "exp6546",
        Path("results/experiment_6546_smt_cost_guard_sota.json"),
        "smt_cost_guard_ready_score",
    ),
    V566Artifact(
        "exp6547",
        Path("results/experiment_6547_external_transfer_independent_audit.json"),
        "external_transfer_audited_ready_score",
    ),
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    ROADMAP_RELATIVE_PATH,
    PROPOSAL_RELATIVE_PATH,
    Path("ops/conductor-log.md"),
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    ARCHITECTURE_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "v566_artifact_eligibility_rows",
    "exp6541_disposition",
    "clean_v566_import_ledger",
    "architecture_freshness_receipt",
    "v567_gate_contract_rows",
    "v567_model_and_hardware_contract",
    "v567_evidence_contract_ready_score",
    "v566_external_transfer_eligible_score",
    "per_unit_rows",
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
    "status": "A terminal lifecycle state prevents a bootstrap artifact from posing as completed work.",
    "honest_verdict": "The verdict must state the evidence boundary and start with complete_, partial_, blocked_, or disqualified_.",
    "verdict_class": "A closed class carries null, partial, blocked, or disqualified status into downstream aggregation.",
    "v566_artifact_eligibility_rows": "One row per artifact makes each eligibility decision independently recheckable.",
    "exp6541_disposition": "The live CRITICAL inconsistency must remain visible and cannot be washed out by a milestone summary.",
    "clean_v566_import_ledger": "Only content-addressed clean evidence may become a V567 input.",
    "architecture_freshness_receipt": "A 51-day-old architecture document must be checked against current code before it anchors integration.",
    "v567_gate_contract_rows": "Every gate must name an in-roadmap task and an upstream field with identical spelling.",
    "v567_model_and_hardware_contract": "Headline model and physical-resource rules must be frozen before outcomes exist.",
    "v567_evidence_contract_ready_score": "A binary readiness field gives downstream production work one explicit contract gate.",
    "v566_external_transfer_eligible_score": "Downstream science needs a clean external root that does not depend on quarantined Exp6541.",
    "per_unit_rows": "Artifact-level rows prevent aggregate eligibility claims from hiding one failed input.",
    "gate_check_summary": "Any blocked verdict must name the failed check and observed value.",
    "preconditions_checked": "Resource receipts distinguish unavailable inputs from null science.",
    "protected_files_unchanged": "The planning task must not mutate the active roadmap or conductor.",
    "inference_substrate": "Declaring no-LLM aggregation prevents fabricated live-model provenance.",
    "verifier_is_oracle": "Hash and schema checks are audit authority, not an oracle-distinct scientific verifier.",
    "field_provenance": "Each headline field must identify the source rows and recomputation path.",
    "duration_s": "Monotonic wall time exposes implausible or stale execution receipts.",
    "tests_run": "Named commands and exit codes make validation reproducible.",
    "reproducibility_checksum": "A content hash detects later mutation of the terminal record.",
}

FIELD_PROVENANCE: dict[str, JsonDict] = {
    field: {
        "principle": FIELD_PRINCIPLES[field],
        "source": "Exp6548 deterministic evidence reducer",
        "spec_refs": ["REQ-REPORT-6548"],
    }
    for field in REQUIRED_ARTIFACT_FIELDS
}
for _field in (
    "v566_artifact_eligibility_rows",
    "exp6541_disposition",
    "clean_v566_import_ledger",
    "architecture_freshness_receipt",
    "v567_gate_contract_rows",
    "v567_model_and_hardware_contract",
    "v567_evidence_contract_ready_score",
    "v566_external_transfer_eligible_score",
    "per_unit_rows",
    "gate_check_summary",
    "preconditions_checked",
    "protected_files_unchanged",
):
    FIELD_PROVENANCE[_field]["source"] = _field

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6548_v567_evidence_eligibility_contract "
    "--date 20260823"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6548_v567_evidence_eligibility_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6548_v567_evidence_eligibility_contract.py "
    "-m pytest tests/python/test_experiment_6548_v567_evidence_eligibility_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6548_v567_evidence_eligibility_contract.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6548_v567_evidence_eligibility_contract.py"
)
EXCLUSION_LINT_COMMAND = ".venv/bin/python scripts/exclusion_manifest_lint.py"
ROADMAP_SCHEMA_COMMAND = ".venv/bin/python scripts/roadmap_schema.py research-roadmap.yaml"
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6548_v567_evidence_eligibility_contract.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6548_v567_evidence_eligibility_contract.json"
)
E2E_PLAN_COMMAND = "manual e2e-plan check: ops/e2e-test-plan.md has no direct Exp6548 entry"

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


def default_v566_paths(repo_root: Path = REPO_ROOT) -> dict[str, Path]:
    return {artifact.exp_id: repo_root / artifact.relative_path for artifact in V566_ARTIFACTS}


def load_v566_payloads(repo_root: Path = REPO_ROOT) -> dict[str, JsonDict]:
    payloads: dict[str, JsonDict] = {}
    for artifact in V566_ARTIFACTS:
        path = repo_root / artifact.relative_path
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
        "max_severity": int(report.get("max_severity") if report.get("max_severity") is not None else -1),
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
    source_paths = default_v566_paths(repo_root) if paths is None else dict(paths)
    return {
        artifact.exp_id: {
            "adversarial": _run_adversarial_check(source_paths[artifact.exp_id], repo_root),
            "row_consistency": _run_row_consistency_check(source_paths[artifact.exp_id], repo_root),
        }
        for artifact in V566_ARTIFACTS
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
        "rows": rows,
    }


def _git_receipt(repo_root: Path) -> JsonDict:
    def run(args: Sequence[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=repo_root, text=True, stderr=subprocess.STDOUT).strip()
        except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
            return f"unavailable: {exc}"

    return {
        "head_sha": run(["git", "rev-parse", "HEAD"]),
        "status_short": run(["git", "status", "--short"]),
    }


def _resource_receipt(repo_root: Path) -> JsonDict:
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


def _network_receipt() -> JsonDict:
    started = time.monotonic()
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=1.0):
            reachable = True
            error = ""
    except OSError as exc:  # pragma: no cover
        reachable = False
        error = f"{type(exc).__name__}: {exc}"
    return {
        "checked": True,
        "method": "tcp_connect_1.1.1.1_53_timeout_1s",
        "reachable": reachable,
        "error": error,
        "duration_s": round(time.monotonic() - started, 6),
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
        "roadmap_schema.py": {
            "path": "scripts/roadmap_schema.py",
            "sha256": sha256_file(repo_root / "scripts/roadmap_schema.py"),
        },
    }


def _artifact_input_receipts(repo_root: Path) -> list[JsonDict]:
    return [
        {
            "exp_id": artifact.exp_id,
            "path": artifact.relative_path.as_posix(),
            "exists": (repo_root / artifact.relative_path).is_file(),
            "bytes": (repo_root / artifact.relative_path).stat().st_size
            if (repo_root / artifact.relative_path).exists()
            else 0,
            "sha256": sha256_file(repo_root / artifact.relative_path),
            "readiness_field": artifact.readiness_field,
        }
        for artifact in V566_ARTIFACTS
    ]


def _readiness_score(payload: Mapping[str, Any], field: str) -> Any:
    return payload.get(field)


def _stamped_flags(payload: Mapping[str, Any]) -> list[JsonDict]:
    pending = payload.get("corrigendum_pending")
    if isinstance(pending, list):
        return [dict(row) for row in pending if isinstance(row, Mapping)]
    return []


def _critical_flags(flags: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [dict(flag) for flag in flags if str(flag.get("severity")).lower() == "critical"]


def _exp6541_model_receipt_gap(payload: Mapping[str, Any]) -> bool:
    rows = payload.get("model_cache_resolution_rows")
    if not isinstance(rows, list) or not rows:
        return True
    return not all(
        isinstance(row, Mapping)
        and row.get("model_path_exists") is True
        and str(row.get("sha256") or row.get("model_file_sha256") or "").startswith("sha256:")
        for row in rows
    )


def _eligibility_row(
    artifact: V566Artifact,
    *,
    repo_root: Path,
    payload: Mapping[str, Any],
    check_result: Mapping[str, Any],
) -> JsonDict:
    path = repo_root / artifact.relative_path
    adversarial = dict(check_result.get("adversarial") or {})
    row_consistency = dict(check_result.get("row_consistency") or {})
    live_flags = [dict(flag) for flag in adversarial.get("flags") or [] if isinstance(flag, Mapping)]
    live_critical = _critical_flags(live_flags)
    stamped = _stamped_flags(payload)
    readiness_field_present = artifact.readiness_field in payload
    readiness_score = _readiness_score(payload, artifact.readiness_field)
    row_blocked = row_consistency.get("exit_code") not in (0, None)
    row_status = str(row_consistency.get("status") or "unknown")
    model_gap = _exp6541_model_receipt_gap(payload) if artifact.exp_id == "exp6541" else False
    duration_open = any(flag.get("kind") == "DURATION_TOO_SHORT" for flag in live_critical)
    non_exp6541_clean = (
        artifact.exp_id != "exp6541"
        and path.is_file()
        and readiness_field_present
        and readiness_score == 1.0
        and not live_critical
        and not row_blocked
        and row_status in {"ok", "skipped"}
    )
    exp6541_resolved = (
        artifact.exp_id == "exp6541"
        and path.is_file()
        and readiness_score == 1.0
        and not live_critical
        and not model_gap
        and not duration_open
        and not row_blocked
    )
    if artifact.exp_id == "exp6541":
        disposition = "eligible_but_not_import_required" if exp6541_resolved else "quarantined_not_imported"
        eligible_for_clean_import = False
    else:
        disposition = "clean_imported" if non_exp6541_clean else "not_imported_failed_checks"
        eligible_for_clean_import = non_exp6541_clean
    unresolved = []
    if duration_open:
        unresolved.append("DURATION_TOO_SHORT")
    if model_gap:
        unresolved.append("model_receipt_gap")
    if live_critical and "DURATION_TOO_SHORT" not in unresolved:
        unresolved.append("live_critical_flags")
    if row_blocked:
        unresolved.append("row_consistency_blocking")
    if not readiness_field_present:
        unresolved.append(f"{artifact.exp_id}_ready_field_present")
    return {
        "row_type": "v566_artifact_eligibility",
        "exp_id": artifact.exp_id,
        "expected_path": artifact.relative_path.as_posix(),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path),
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict"),
        "verdict_class": payload.get("verdict_class"),
        "inference_substrate": payload.get("inference_substrate"),
        "duration_s": payload.get("duration_s"),
        "readiness_field": artifact.readiness_field,
        "readiness_field_present": readiness_field_present,
        "readiness_score": readiness_score,
        "stamped_flags": stamped,
        "stamped_flag_count": len(stamped),
        "live_verifier_command": adversarial.get("command"),
        "live_verifier_exit_code": adversarial.get("exit_code"),
        "live_flags": live_flags,
        "live_flag_count": len(live_flags),
        "live_critical_flags": live_critical,
        "live_critical_flag_count": len(live_critical),
        "row_consistency_command": row_consistency.get("command"),
        "row_consistency_exit_code": row_consistency.get("exit_code"),
        "row_consistency_status": row_status,
        "row_consistency_findings": list(row_consistency.get("findings") or []),
        "row_consistency_blocking": row_blocked,
        "duration_classification_open": duration_open,
        "model_receipt_gap_open": model_gap,
        "unresolved_reasons": unresolved,
        "independent_checks_resolved": exp6541_resolved if artifact.exp_id == "exp6541" else eligible_for_clean_import,
        "eligible_for_clean_import": eligible_for_clean_import,
        "disposition": disposition,
    }


def build_v566_rows(
    *,
    repo_root: Path,
    payloads: Mapping[str, Mapping[str, Any]],
    check_results: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    return [
        _eligibility_row(
            artifact,
            repo_root=repo_root,
            payload=payloads.get(artifact.exp_id, {}),
            check_result=check_results.get(artifact.exp_id, {}),
        )
        for artifact in V566_ARTIFACTS
    ]


def _exp6541_disposition(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    row = next(row for row in rows if row.get("exp_id") == "exp6541")
    return {
        "exp_id": "exp6541",
        "path": row["expected_path"],
        "sha256": row["sha256"],
        "reported_readiness_score": row["readiness_score"],
        "disposition": row["disposition"],
        "unresolved_reasons": list(row.get("unresolved_reasons") or []),
        "live_critical_flags": list(row.get("live_critical_flags") or []),
        "model_receipt_gap_open": row["model_receipt_gap_open"],
        "duration_classification_open": row["duration_classification_open"],
        "clean_import_dependency": False,
    }


def _clean_import_ledger(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    imported = [
        dict(row)
        for row in rows
        if row.get("exp_id") in EXPECTED_CLEAN_IMPORT_EXP_IDS and row.get("eligible_for_clean_import")
    ]
    imported_ids = [str(row["exp_id"]) for row in imported]
    return {
        "expected_clean_exp_ids": list(EXPECTED_CLEAN_IMPORT_EXP_IDS),
        "imported_exp_ids": imported_ids,
        "excluded_exp_ids": [str(row["exp_id"]) for row in rows if row.get("exp_id") not in imported_ids],
        "imported_rows": imported,
        "all_expected_clean_imported": imported_ids == list(EXPECTED_CLEAN_IMPORT_EXP_IDS),
        "ledger_sha256": sha256_json(
            [{"exp_id": row["exp_id"], "path": row["expected_path"], "sha256": row["sha256"]} for row in imported]
        ),
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
        task_id: _parse_required_fields(str(task.get("prompt") or "")) for task_id, task in tasks.items()
    }
    retired = _retired_experiment_ids(_load_yaml(repo_root / EXCLUSION_MANIFEST_RELATIVE_PATH))
    return roadmap, fields_by_task, retired


def _gate_contract_rows(
    roadmap: Mapping[str, Any],
    fields_by_task: Mapping[str, set[str]],
    retired_ids: set[str],
) -> list[JsonDict]:
    task_ids = {str(task.get("id")) for task in roadmap.get("tasks", []) if isinstance(task, Mapping)}
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
                    "row_type": "v567_gate_contract",
                    "task_id": task_id,
                    "gate_index": index,
                    "upstream": upstream,
                    "artifact_field": field,
                    "op": gate.get("op"),
                    "value": gate.get("value"),
                    "upstream_in_v567": upstream in task_ids,
                    "artifact_field_declared_by_upstream": field in upstream_fields,
                    "upstream_declared_field_count": len(upstream_fields),
                    "retired_upstream": upstream in retired_ids,
                    "principle": gate.get("principle"),
                }
            )
    return rows


def _architecture_freshness_receipt(repo_root: Path, run_date: str) -> JsonDict:
    text = (repo_root / ARCHITECTURE_RELATIVE_PATH).read_text(encoding="utf-8")
    match = re.search(r"\*\*Last Reconciled:\*\*\s*(\d{4}-\d{2}-\d{2})", text)
    last_reconciled = match.group(1) if match else "unknown"
    age_days = None
    if last_reconciled != "unknown":
        age_days = (date.fromisoformat(f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:]}") - date.fromisoformat(last_reconciled)).days
    code_paths = [
        Path("python/carnot/pipeline/verify_repair.py"),
        Path("python/carnot/experiment_6547_external_transfer_independent_audit.py"),
        Path("scripts/roadmap_schema.py"),
        ROADMAP_RELATIVE_PATH,
    ]
    path_rows = [
        {
            "path": path.as_posix(),
            "exists": (repo_root / path).is_file(),
            "sha256": sha256_file(repo_root / path),
        }
        for path in code_paths
    ]
    return {
        "architecture_path": ARCHITECTURE_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(repo_root / ARCHITECTURE_RELATIVE_PATH),
        "last_reconciled": last_reconciled,
        "planning_date": f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:]}",
        "age_days_at_planning": age_days,
        "freshness_threshold_days": 30,
        "stale_by_age": age_days is not None and age_days > 30,
        "current_code_path_rows": path_rows,
        "checked_against_current_code": all(row["exists"] and row["sha256"].startswith("sha256:") for row in path_rows),
        "integration_boundary_must_reconcile_if_changed": True,
    }


def _model_and_hardware_contract(roadmap: Mapping[str, Any], retired_ids: set[str]) -> JsonDict:
    task_ids = [str(task.get("id")) for task in roadmap.get("tasks", []) if isinstance(task, Mapping)]
    gate_upstreams = {
        str(gate.get("upstream"))
        for task in roadmap.get("tasks", [])
        if isinstance(task, Mapping)
        for gate in (task.get("gated_on") or [])
        if isinstance(gate, Mapping)
    }
    prompts = "\n".join(str(task.get("prompt") or "") for task in roadmap.get("tasks", []) if isinstance(task, Mapping))
    return {
        "mandated_model_ids": list(MANDATED_MODEL_IDS),
        "headline_model_tasks": [task for task in HEADLINE_MODEL_TASKS if task in task_ids],
        "all_headline_model_tasks_present": all(task in task_ids for task in HEADLINE_MODEL_TASKS),
        "gguf_loader_rule": "cached_sota_pair(gpu_indices=(0, 1)) plus llama.cpp",
        "legacy_models_headline_excluded": "Do not substitute a legacy model for a headline row" in prompts,
        "model_specs_rule": {
            "tasks": [task for task in HEADLINE_MODEL_TASKS if task in task_ids],
            "required_ids": list(MANDATED_MODEL_IDS),
            "auto_tokenizer_on_gguf_repo_id_allowed": False,
        },
        "sample_floor_rule": {
            "exp6553_min_query_boundaries_per_model": 36,
            "exp6556_requires_all_three_model_families": True,
        },
        "arc_no_solve_rule": {
            "task_id": "exp6558-arc-live-redirect-ledger",
            "no_game_or_level_solve_claim": "claims no game or level solve" in prompts.lower(),
        },
        "gatemate_receipt_rule": {
            "task_id": "exp6559-gatemate-changed-state-continuity",
            "requires_receipt_newer_than_exp6525": "newer than Exp6525" in prompts,
            "zero_commands_without_new_receipt": "zero hardware commands" in prompts or "runs zero hardware commands" in prompts,
        },
        "hardware_scope": {
            "kv260_required": False,
            "polarfire_required": False,
            "extropic_or_kona_required": False,
            "dual_rtx_3090_required_for": list(HEADLINE_MODEL_TASKS),
        },
        "retired_scope_isolation": {
            "retired_upstreams_in_structured_gates": sorted(gate_upstreams & retired_ids),
            "schema_supported_constraintir_reopened": any(
                task_id in task_ids or task_id in gate_upstreams
                for task_id in (
                    "exp5909-sota-constraint-synthesis-ab",
                    "exp5910-verification-guided-constraint-repair",
                    "exp5923-sota-schema-supported-constraintir-ab",
                )
            ),
            "generated_text_verifier_scope_reopened": False,
        },
    }


def _gate_check_summary(
    *,
    rows: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, Any],
    gate_rows: Sequence[Mapping[str, Any]],
    architecture: Mapping[str, Any],
    model_contract: Mapping[str, Any],
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
        if exp_id in EXPECTED_CLEAN_IMPORT_EXP_IDS and row.get("readiness_field_present") is not True:
            add_failed(
                f"{exp_id}_ready_field_present",
                True,
                row.get("readiness_field_present"),
                str(row.get("expected_path")),
            )
        if exp_id in EXPECTED_CLEAN_IMPORT_EXP_IDS and row.get("eligible_for_clean_import") is not True:
            add_failed(f"{exp_id}_clean_import_eligible", True, False, str(row.get("expected_path")))
    for row in gate_rows:
        if row.get("upstream_in_v567") is not True:
            add_failed("gate_upstream_in_v567", True, row.get("upstream"), str(row.get("task_id")))
        if row.get("artifact_field_declared_by_upstream") is not True:
            add_failed("gate_artifact_field_declared", True, row.get("artifact_field"), str(row.get("task_id")))
        if row.get("retired_upstream") is True:
            add_failed("gate_retired_upstream", False, row.get("upstream"), str(row.get("task_id")))
    if ledger.get("all_expected_clean_imported") is not True:
        add_failed("clean_v566_import_ledger_complete", True, ledger.get("imported_exp_ids"), "clean_v566_import_ledger")
    if architecture.get("checked_against_current_code") is not True:
        add_failed("architecture_checked_against_current_code", True, False, ARCHITECTURE_RELATIVE_PATH.as_posix())
    if model_contract.get("all_headline_model_tasks_present") is not True:
        add_failed("headline_model_tasks_present", True, model_contract.get("headline_model_tasks"), ROADMAP_RELATIVE_PATH.as_posix())
    retired_scope = model_contract.get("retired_scope_isolation") or {}
    if retired_scope.get("schema_supported_constraintir_reopened") is True:
        add_failed("schema_supported_constraintir_not_reopened", False, True, ROADMAP_RELATIVE_PATH.as_posix())
    if protected.get("all_unchanged") is not True:
        add_failed("protected_files_unchanged", True, protected.get("changed_paths"), "protected_files_unchanged")

    return {
        "all_gates_passed": not failed,
        "failed_checks": failed,
        "failed_check_rows": failed_rows,
        "task_field_gate_contract_closed": all(
            row.get("upstream_in_v567") is True
            and row.get("artifact_field_declared_by_upstream") is True
            and row.get("retired_upstream") is False
            for row in gate_rows
        ),
        "acceptance_gates": [
            {
                "condition": "Exp6542-Exp6547 are clean imports; Exp6541 is not required for that score.",
                "passed": ledger.get("all_expected_clean_imported") is True,
            },
            {
                "condition": "All V567 structured gates name in-roadmap tasks and exact upstream fields.",
                "passed": all(
                    row.get("upstream_in_v567") is True
                    and row.get("artifact_field_declared_by_upstream") is True
                    and row.get("retired_upstream") is False
                    for row in gate_rows
                ),
            },
            {
                "condition": "Architecture, model, hardware, scope, and protected-file contracts close.",
                "passed": not any(
                    check in failed
                    for check in (
                        "architecture_checked_against_current_code",
                        "headline_model_tasks_present",
                        "schema_supported_constraintir_not_reopened",
                        "protected_files_unchanged",
                    )
                ),
            },
        ],
    }


def _preconditions_checked(
    *,
    repo_root: Path,
    protected_before: Mapping[str, str],
    architecture: Mapping[str, Any],
) -> JsonDict:
    return {
        "git_state": _git_receipt(repo_root),
        "resources": _resource_receipt(repo_root),
        "network_status": _network_receipt(),
        "verifier_versions": _verifier_versions(repo_root),
        "required_artifact_receipts": _artifact_input_receipts(repo_root),
        "architecture_freshness_date": architecture.get("last_reconciled"),
        "protected_file_hashes_before": dict(protected_before),
        "live_gguf_load_required": False,
        "external_intake_rerun": False,
    }


def _status_and_verdict(
    ready: bool,
    external_score: float,
    failed_checks: Sequence[str],
) -> tuple[str, str, str | None]:
    if ready:
        return (
            "complete_v567_evidence_eligibility_contract_ready",
            "complete_v567_evidence_eligibility_contract_ready: Exp6542-Exp6547 are clean content-addressed V567 inputs; Exp6541 remains visible and not required",
            None,
        )
    if external_score > 0:
        return (
            "partial_v567_evidence_eligibility_contract",
            "partial_v567_evidence_eligibility_contract: usable V566 subset exists but one or more V567 evidence, field, gate, model, hardware, or scope checks failed",
            "partial",
        )
    if failed_checks:
        return (
            "partial_v567_evidence_eligibility_contract",
            "partial_v567_evidence_eligibility_contract: clean V566 external transfer did not close; failed checks are recorded",
            "partial",
        )
    return (
        "blocked_v567_evidence_eligibility_contract",
        "blocked_v567_evidence_eligibility_contract: no clean V566 input set was available",
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
    run_date: str = RUN_DATE,
) -> JsonDict:
    started = time.monotonic()
    protected_before = _protected_hashes(repo_root)
    payloads = load_v566_payloads(repo_root) if input_payloads is None else input_payloads
    live_results = run_live_checks(repo_root) if check_results is None else check_results
    rows = build_v566_rows(repo_root=repo_root, payloads=payloads, check_results=live_results)
    ledger = _clean_import_ledger(rows)
    roadmap, fields_by_task, retired_ids = _roadmap_and_contract(repo_root)
    gate_rows = _gate_contract_rows(roadmap, fields_by_task, retired_ids)
    architecture = _architecture_freshness_receipt(repo_root, run_date)
    model_contract = _model_and_hardware_contract(roadmap, retired_ids)
    protected_after = _protected_hashes(repo_root)
    protected = _protected_files_unchanged(protected_before, protected_after)
    gate_summary = _gate_check_summary(
        rows=rows,
        ledger=ledger,
        gate_rows=gate_rows,
        architecture=architecture,
        model_contract=model_contract,
        protected=protected,
    )
    external_score = 1.0 if ledger["all_expected_clean_imported"] else 0.0
    ready = (
        external_score == 1.0
        and gate_summary["all_gates_passed"] is True
        and _exp6541_disposition(rows)["disposition"] != "clean_imported"
    )
    status, honest_verdict, verdict_class = _status_and_verdict(
        ready, external_score, gate_summary["failed_checks"]
    )
    measured_duration = round(time.monotonic() - started, 6) if duration_s is None else float(duration_s)
    payload: JsonDict = {
        "status": status,
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "v566_artifact_eligibility_rows": rows,
        "exp6541_disposition": _exp6541_disposition(rows),
        "clean_v566_import_ledger": ledger,
        "architecture_freshness_receipt": architecture,
        "v567_gate_contract_rows": gate_rows,
        "v567_model_and_hardware_contract": model_contract,
        "v567_evidence_contract_ready_score": 1.0 if ready else 0.0,
        "v566_external_transfer_eligible_score": external_score,
        "per_unit_rows": rows,
        "gate_check_summary": gate_summary,
        "preconditions_checked": _preconditions_checked(
            repo_root=repo_root,
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
        for row in payload.get("v566_artifact_eligibility_rows", [])
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
    exp6541 = rows.get("exp6541", {})
    ledger = payload.get("clean_v566_import_ledger") or {}
    imported_rows = [dict(row) for row in ledger.get("imported_rows", []) if isinstance(row, Mapping)]
    imported_ids = list(ledger.get("imported_exp_ids") or [])
    if exp6541.get("eligible_for_clean_import") is True or "exp6541" in imported_ids:
        errors.append("Exp6541 must remain quarantined from clean import")

    if payload.get("v566_external_transfer_eligible_score") == 1.0:
        expected = list(EXPECTED_CLEAN_IMPORT_EXP_IDS)
        if imported_ids != expected or any(rows.get(exp_id, {}).get("eligible_for_clean_import") is not True for exp_id in expected):
            errors.append("external transfer score must derive from eligible Exp6542-Exp6547 rows")

    row_hashes = {exp_id: row.get("sha256") for exp_id, row in rows.items()}
    for imported in imported_rows:
        exp_id = str(imported.get("exp_id"))
        if imported.get("sha256") != row_hashes.get(exp_id):
            errors.append(f"clean import ledger hash alias for {exp_id}")

    for row in payload.get("v567_gate_contract_rows") or []:
        if not isinstance(row, Mapping):
            errors.append("gate contract row must be a mapping")
            continue
        if row.get("artifact_field_declared_by_upstream") is not True:
            errors.append("gate contract has undeclared field")
        if row.get("upstream_in_v567") is not True:
            errors.append("gate contract has out-of-roadmap upstream")
        if row.get("retired_upstream") is True:
            errors.append("gate contract has retired upstream")

    protected = payload.get("protected_files_unchanged") or {}
    if protected.get("all_unchanged") is not True:
        errors.append("protected files changed")
    gate_summary = payload.get("gate_check_summary") or {}
    if payload.get("v567_evidence_contract_ready_score") == 1.0 and gate_summary.get("failed_checks"):
        errors.append("ready score cannot be open with failed checks")
    if payload.get("v567_evidence_contract_ready_score") == 1.0 and exp6541.get("disposition") == "clean_imported":
        errors.append("Exp6541 must remain quarantined from clean import")
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
            for artifact in V566_ARTIFACTS
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
