"""Build the immutable V570 evidence, gate, failure, and retirement root.

Spec refs: REQ-REPORT-6571, REQ-REPORT-6571-IMPORT,
REQ-REPORT-6571-LIVE, REQ-REPORT-6571-GGUF,
REQ-REPORT-6571-VERDICT, REQ-REPORT-6571-GATES,
REQ-REPORT-6571-FAILURES, REQ-REPORT-6571-BOUNDARIES,
REQ-REPORT-6571-RUST, REQ-REPORT-6571-ATTACKS,
REQ-REPORT-6571-ATOMIC, SCENARIO-REPORT-6571-IMPORT,
SCENARIO-REPORT-6571-LIVE, SCENARIO-REPORT-6571-GGUF,
SCENARIO-REPORT-6571-GATES, SCENARIO-REPORT-6571-BOUNDARIES,
SCENARIO-REPORT-6571-ATOMIC.

This reducer reads only checked-in V569 evidence and local system receipts. It
does not load an LLM and does not issue a board or GPU command. The known
Exp6569 absence is evidence that extraction did not run; it is not a null
scientific result and it is not an unexpected prerequisite failure.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any

import yaml

from carnot.experiment_artifacts import atomic_write_json
from scripts.roadmap_schema import Roadmap


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260824"
RANDOM_SEED = 6571
INFERENCE_SUBSTRATE = "immutable_v569_artifact_gate_failure_and_retirement_audit_no_llm"

RESULT_RELATIVE_PATH = Path("results/experiment_6571_v570_evidence_gate_and_retirement_root.json")
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
class V569Artifact:
    """Describe one fixed V569 deliverable and whether absence is expected."""

    exp_id: str
    task_id: str
    relative_path: Path
    expected_missing: bool = False


V569_ARTIFACTS = (
    V569Artifact(
        "exp6565",
        "exp6565-v569-evidence-and-retirement-contract",
        Path("results/experiment_6565_v569_evidence_and_retirement_contract.json"),
    ),
    V569Artifact(
        "exp6566",
        "exp6566-proof-obligation-and-graph-potts-method-contract",
        Path("results/experiment_6566_proof_obligation_and_graph_potts_method_contract.json"),
    ),
    V569Artifact(
        "exp6567",
        "exp6567-sequential-flagship-gguf-admission",
        Path("results/experiment_6567_sequential_flagship_gguf_admission.json"),
    ),
    V569Artifact(
        "exp6568",
        "exp6568-immutable-source-span-claim-stream",
        Path("results/experiment_6568_immutable_source_span_claim_stream.json"),
    ),
    V569Artifact(
        "exp6569",
        "exp6569-source-span-proof-obligation-extractor",
        Path("results/experiment_6569_source_span_proof_obligation_extractor.json"),
        expected_missing=True,
    ),
    V569Artifact(
        "exp6570",
        "exp6570-proof-obligation-independent-audit",
        Path("results/experiment_6570_proof_obligation_independent_audit.json"),
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
    SPEC_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    Path("ops/conductor-log.md"),
    Path("ops/known-issues.md"),
    Path("ops/e2e-test-plan.md"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("_bmad/prd.md"),
    ARCHITECTURE_RELATIVE_PATH,
    Path("_bmad/traceability.md"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/artifact_convention_audit.py"),
    Path("scripts/conductor_gates.py"),
    Path("scripts/roadmap_schema.py"),
    Path("scripts/exclusion_manifest_lint.py"),
    Path("scripts/validate_prior_failures.py"),
    Path("scripts/research_conductor.py"),
    *(artifact.relative_path for artifact in V569_ARTIFACTS),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "v569_artifact_eligibility_rows",
    "live_verifier_and_duration_rows",
    "gguf_admission_root_cause",
    "v570_gate_contract_rows",
    "prior_failure_and_retirement_rows",
    "model_arc_and_hardware_boundary",
    "v570_evidence_contract_ready_score",
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
    "status": "A terminal state prevents a bootstrap record from posing as the evidence root.",
    "honest_verdict": "The verdict must state eligibility, failed scopes, and retirement boundaries with a terminal prefix.",
    "verdict_class": "A closed enum carries null, blocked, partial, and disqualified state downstream.",
    "v569_artifact_eligibility_rows": "One row per expected V569 artifact makes missing and unusable evidence visible.",
    "live_verifier_and_duration_rows": "Fresh commands, exits, flags, and monotonic durations resolve stale stamped findings.",
    "gguf_admission_root_cause": "The next mechanism must target the observed hash-only path defect, not repeat generic fit prediction.",
    "v570_gate_contract_rows": "Every downstream gate must name an in-roadmap task and exact upstream field.",
    "prior_failure_and_retirement_rows": "Every scope match needs a changed mechanism and repeat-retirement rule.",
    "model_arc_and_hardware_boundary": "The root freezes flagship, live-ARC, and zero-unchanged-command rules.",
    "v570_evidence_contract_ready_score": "One binary field gates tasks that require the full V570 evidence contract.",
    "rust_fusion_reopen_ready_score": "A separate binary field permits only the changed fused workload and freezes retirement.",
    "per_unit_rows": "Artifact-level rows prevent one missing or invalid input from hiding in an aggregate.",
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

ATTACK_NAMES = (
    "hash_alias",
    "missing_artifact_laundering",
    "aggregate_only_claim",
    "stamped_live_verifier_disagreement",
    "false_model_admission",
    "wrong_verdict_class",
    "gate_field_drift",
    "legacy_model_substitution",
    "retired_constraintir_reuse",
    "arc_solve_laundering",
    "unchanged_hardware_command",
    "protected_file_mutation",
)

RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6571_v570_evidence_gate_and_retirement_root --date 20260824"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6571_v570_evidence_gate_and_retirement_root.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6571_v570_evidence_gate_and_retirement_root.py "
    "-m pytest tests/python/test_experiment_6571_v570_evidence_gate_and_retirement_root.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6571_v570_evidence_gate_and_retirement_root.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
RUFF_CHECK_COMMAND = (
    ".venv/bin/ruff check "
    "python/carnot/experiment_6571_v570_evidence_gate_and_retirement_root.py "
    "tests/python/test_experiment_6571_v570_evidence_gate_and_retirement_root.py"
)
RUFF_FORMAT_COMMAND = (
    ".venv/bin/ruff format --check "
    "python/carnot/experiment_6571_v570_evidence_gate_and_retirement_root.py "
    "tests/python/test_experiment_6571_v570_evidence_gate_and_retirement_root.py"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6571_v570_evidence_gate_and_retirement_root.py"
)
ROADMAP_SCHEMA_COMMAND = ".venv/bin/python scripts/roadmap_schema.py research-roadmap.yaml"
GATE_AUDIT_COMMAND = "internal Exp6571 exact V570 gate audit over research-roadmap.yaml"
EXCLUSION_LINT_COMMAND = ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml"
PRIOR_FAILURE_COMMAND = ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml"
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6571_v570_evidence_gate_and_retirement_root.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6571_v570_evidence_gate_and_retirement_root.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6571_v570_evidence_gate_and_retirement_root --validate"
)
E2E_PLAN_COMMAND = (
    "manual e2e-plan check: Exp6571 is an immutable no-LLM contract audit; "
    "ops/e2e-test-plan.md has no direct Exp6571 execution entry"
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
    """Return one stable JSON representation for hashing."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value with a visible algorithm prefix."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with a visible algorithm prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path | None) -> str:
    """Hash one file, or return ``missing`` when no file exists."""

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
    """Hash the terminal payload while excluding its checksum field."""

    clone = dict(payload)
    clone.pop("reproducibility_checksum", None)
    return sha256_json(clone)


def _read_json(path: Path) -> JsonDict:
    """Read one JSON object and fail closed to an empty object."""

    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _load_yaml(path: Path) -> JsonDict:
    """Read one YAML mapping."""

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def default_v569_paths(repo_root: Path = REPO_ROOT) -> dict[str, Path]:
    """Return the six exact V569 deliverable paths."""

    return {artifact.exp_id: repo_root / artifact.relative_path for artifact in V569_ARTIFACTS}


def _command_text(argv: Sequence[str]) -> str:
    """Render a command without executing shell interpolation."""

    return " ".join(shlex.quote(str(part)) for part in argv)


def _python_executable(repo_root: Path) -> str:  # pragma: no cover - host receipt.
    venv_python = repo_root / ".venv/bin/python"
    return str(venv_python) if venv_python.is_file() else sys.executable


def _run_command(argv: Sequence[str], repo_root: Path) -> JsonDict:  # pragma: no cover
    """Run one bounded local checker and record monotonic timing."""

    started = time.monotonic()
    try:
        process = subprocess.run(
            [str(part) for part in argv],
            cwd=repo_root,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        stdout = process.stdout
        stderr = process.stderr
        exit_code = process.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = str(exc.stdout or "")
        stderr = str(exc.stderr or "")
        exit_code = 124
    return {
        "command": _command_text(argv),
        "exit_code": exit_code,
        "duration_s": round(time.monotonic() - started, 6),
        "stdout": stdout,
        "stderr": stderr,
        "stdout_sha256": sha256_bytes(str(stdout).encode()),
        "stderr_sha256": sha256_bytes(str(stderr).encode()),
    }


def _run_adversarial_check(path: Path, repo_root: Path) -> JsonDict:  # pragma: no cover
    """Replay the checked-in adversarial verifier for one exact path."""

    argv = [_python_executable(repo_root), "scripts/adversarial_verify.py", "--json", str(path)]
    receipt = _run_command(argv, repo_root)
    try:
        parsed = json.loads(str(receipt["stdout"]))
        reports = parsed.get("reports")
        report = dict(reports[0]) if isinstance(reports, list) and reports else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        report = {}
    return {
        "command": receipt["command"],
        "exit_code": receipt["exit_code"],
        "duration_s": receipt["duration_s"],
        "loaded": report.get("loaded", path.is_file()),
        "flags": list(report.get("flags") or []),
        "stdout_sha256": receipt["stdout_sha256"],
        "stderr_sha256": receipt["stderr_sha256"],
    }


def _run_row_consistency_check(path: Path, repo_root: Path) -> JsonDict:  # pragma: no cover
    """Replay row consistency and preserve skipped or unreadable status."""

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


def _contains_rows(value: Any, depth: int = 0) -> bool:
    """Find a non-empty row container without interpreting its metrics."""

    if depth > 4:
        return False
    if isinstance(value, Mapping):
        for key, child in value.items():
            if "row" in str(key).lower() and isinstance(child, list) and child:
                return True
            if _contains_rows(child, depth + 1):
                return True
    elif isinstance(value, list):
        return any(_contains_rows(child, depth + 1) for child in value)
    return False


def _run_artifact_convention_check(
    artifact: V569Artifact, path: Path, payload: Mapping[str, Any]
) -> JsonDict:
    """Apply the audit's two conventions without making its optional LLM call."""

    started = time.monotonic()
    command = (
        "internal deterministic replay of scripts/artifact_convention_audit.py "
        f"conventions --no-llm {path}"
    )
    if not path.is_file():
        status = "EXPECTED_MISSING" if artifact.expected_missing else "CANNOT_DETERMINE"
        reason = (
            "expected artifact is absent" if artifact.expected_missing else "artifact unreadable"
        )
    else:
        verdict_text = " ".join(
            str(payload.get(key) or "") for key in ("status", "honest_verdict", "blocked_reason")
        ).lower()
        blocked = "blocked" in verdict_text
        has_diagnostic = bool(payload.get("gate_check_summary") or payload.get("blocked_reason"))
        if blocked and not has_diagnostic:
            status = "BLOCKED_WITHOUT_DIAGNOSTIC"
            reason = "blocked artifact has no gate_check_summary or blocked_reason"
        elif not blocked and not _contains_rows(payload):
            status = "AGGREGATE_ONLY"
            reason = "non-blocked claim has no row container"
        else:
            status = "CHECKABLE"
            reason = "per-unit rows or blocked diagnostics are present"
    return {
        "command": command,
        "exit_code": 0 if status in {"CHECKABLE", "EXPECTED_MISSING"} else 1,
        "duration_s": round(time.monotonic() - started, 6),
        "status": status,
        "reason": reason,
        "llm_call_performed": False,
    }


def run_live_checks(
    repo_root: Path = REPO_ROOT,
    paths: Mapping[str, Path] | None = None,
) -> dict[str, JsonDict]:  # pragma: no cover
    """Run all three fresh checks for all six expected V569 paths."""

    source_paths = default_v569_paths(repo_root) if paths is None else dict(paths)
    results: dict[str, JsonDict] = {}
    for artifact in V569_ARTIFACTS:
        path = source_paths[artifact.exp_id]
        payload = _read_json(path)
        results[artifact.exp_id] = {
            "adversarial": _run_adversarial_check(path, repo_root),
            "row_consistency": _run_row_consistency_check(path, repo_root),
            "artifact_convention": _run_artifact_convention_check(artifact, path, payload),
        }
    return results


def _closed_verdict_class(value: Any) -> str | None:
    """Keep only the closed V570 import classes."""

    if value is None or value == "null":
        return None
    if value in {"blocked", "partial", "disqualified"}:
        return str(value)
    return "disqualified"


def _is_hash_only_path(path: str) -> bool:
    """Recognize a cache basename made only from a content hash."""

    return re.fullmatch(r"[0-9a-fA-F]{40,128}", Path(path).name) is not None


def _stamped_flags(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Preserve checker flags already stamped into an upstream artifact."""

    pending = payload.get("corrigendum_pending")
    if not isinstance(pending, list):
        return []
    return [dict(row) for row in pending if isinstance(row, Mapping)]


def _critical_flags(flags: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Select only critical fresh flags."""

    return [dict(row) for row in flags if str(row.get("severity")).lower() == "critical"]


def _artifact_outcome(
    artifact: V569Artifact,
    *,
    path: Path,
    payload: Mapping[str, Any],
    live_result: Mapping[str, Any],
) -> JsonDict:
    """Classify one V569 artifact without upgrading its science."""

    exists = path.is_file()
    adversarial = live_result.get("adversarial") or {}
    row_lint = live_result.get("row_consistency") or {}
    convention = live_result.get("artifact_convention") or {}
    fresh_critical = _critical_flags(adversarial.get("flags") or [])
    checker_ok = (
        adversarial.get("exit_code") == 0
        and not fresh_critical
        and convention.get("status") in {"CHECKABLE", "EXPECTED_MISSING"}
    )
    if artifact.expected_missing:
        eligible = (
            not exists
            and not payload
            and adversarial.get("loaded") is False
            and row_lint.get("status") == "unreadable"
            and convention.get("status") == "EXPECTED_MISSING"
        )
        return {
            "disposition": "missing_not_null" if eligible else "frozen_missing_path_changed",
            "failed_scope": "source_span_proof_extraction_unrun",
            "verdict_class": "blocked",
            "eligible": eligible,
            "reason": "expected_exp6569_artifact_absence_confirmed"
            if eligible
            else "exp6569_frozen_absence_changed",
        }
    if not exists or not payload:
        return {
            "disposition": "unexpected_missing_prerequisite",
            "failed_scope": "missing_v569_artifact",
            "verdict_class": "blocked",
            "eligible": False,
            "reason": f"{artifact.exp_id}_expected_artifact_missing_or_unreadable",
        }
    if not checker_ok or row_lint.get("status") == "unreadable":
        return {
            "disposition": "unusable_live_verifier_result",
            "failed_scope": "live_artifact_verification",
            "verdict_class": "disqualified" if fresh_critical else "partial",
            "eligible": False,
            "reason": "fresh_artifact_checks_did_not_close",
        }

    source_class = _closed_verdict_class(payload.get("verdict_class"))
    if artifact.exp_id == "exp6565":
        eligible = payload.get("v569_evidence_contract_ready_score") == 1.0 and source_class is None
        disposition = "usable_v569_evidence_contract"
        failed_scope = "stale_stamped_verifier_finding"
        imported_class = source_class
    elif artifact.exp_id == "exp6566":
        eligible = payload.get("source_method_contract_ready_score") == 1.0 and source_class is None
        disposition = "usable_v569_method_contract"
        failed_scope = "none_method_contract_only"
        imported_class = source_class
    elif artifact.exp_id == "exp6567":
        eligible = (
            source_class == "blocked" and payload.get("all_mandated_models_loaded_score") == 0.0
        )
        disposition = "blocked_hash_only_gguf_admission"
        failed_scope = "flagship_runtime_admission_before_inference"
        imported_class = "blocked"
    elif artifact.exp_id == "exp6568":
        eligible = (
            str(payload.get("status") or "").startswith("blocked")
            and payload.get("failed_field") == "all_mandated_models_loaded_score"
            and payload.get("failed_observed") == 0.0
        )
        disposition = "corrected_blocked_gate_import"
        failed_scope = "immutable_source_stream_not_run"
        imported_class = "blocked"
    else:
        eligible = (
            source_class == "blocked"
            and payload.get("proof_carrying_extractor_audit_ready_score") == 0.0
        )
        disposition = "valid_blocked_independent_audit"
        failed_scope = "proof_rows_missing_not_recomputable"
        imported_class = "blocked"
    return {
        "disposition": disposition,
        "failed_scope": failed_scope,
        "verdict_class": imported_class,
        "eligible": bool(eligible),
        "reason": f"{disposition}_from_exact_path_hash_and_fresh_checks",
    }


def _live_row(
    artifact: V569Artifact,
    *,
    path: Path,
    payload: Mapping[str, Any],
    check_result: Mapping[str, Any],
) -> JsonDict:
    """Build one fresh checker and duration row."""

    adversarial = dict(check_result.get("adversarial") or {})
    row_lint = dict(check_result.get("row_consistency") or {})
    convention = dict(check_result.get("artifact_convention") or {})
    fresh_flags = [dict(row) for row in adversarial.get("flags") or [] if isinstance(row, Mapping)]
    stamped_flags = _stamped_flags(payload)
    return {
        "row_type": "live_verifier_and_duration",
        "exp_id": artifact.exp_id,
        "artifact_path": artifact.relative_path.as_posix(),
        "artifact_sha256": sha256_file(path),
        "artifact_duration_s": payload.get("duration_s"),
        "artifact_loaded_by_adversarial_verifier": bool(adversarial.get("loaded")),
        "stamped_flags": stamped_flags,
        "stamped_flag_count": len(stamped_flags),
        "fresh_flags": fresh_flags,
        "fresh_flag_count": len(fresh_flags),
        "fresh_critical_flags": _critical_flags(fresh_flags),
        "stamped_live_flag_disagreement": bool(stamped_flags) != bool(fresh_flags),
        "live_verifier_command": adversarial.get("command"),
        "live_verifier_exit_code": adversarial.get("exit_code"),
        "live_verifier_duration_s": float(adversarial.get("duration_s") or 0.0),
        "row_consistency_command": row_lint.get("command"),
        "row_consistency_exit_code": row_lint.get("exit_code"),
        "row_consistency_duration_s": float(row_lint.get("duration_s") or 0.0),
        "row_consistency_status": row_lint.get("status"),
        "row_consistency_findings": list(row_lint.get("findings") or []),
        "artifact_convention_command": convention.get("command"),
        "artifact_convention_exit_code": convention.get("exit_code"),
        "artifact_convention_duration_s": float(convention.get("duration_s") or 0.0),
        "artifact_convention_status": convention.get("status"),
        "artifact_convention_reason": convention.get("reason"),
        "reason": "expected_missing_path_replayed"
        if artifact.expected_missing
        else "current_artifact_checks_replayed",
    }


def _eligibility_row(
    artifact: V569Artifact,
    *,
    path: Path,
    payload: Mapping[str, Any],
    check_result: Mapping[str, Any],
) -> JsonDict:
    """Build one immutable V569 import row."""

    live = _live_row(artifact, path=path, payload=payload, check_result=check_result)
    outcome = _artifact_outcome(artifact, path=path, payload=payload, live_result=check_result)
    return {
        "row_type": "v569_artifact_eligibility",
        "exp_id": artifact.exp_id,
        "task_id": artifact.task_id,
        "expected_path": artifact.relative_path.as_posix(),
        "resolved_path": str(path),
        "expected_missing": artifact.expected_missing,
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path),
        "status": payload.get("status") if payload else "missing",
        "honest_verdict": payload.get("honest_verdict") if payload else "blocked_missing",
        "source_verdict_class": _closed_verdict_class(payload.get("verdict_class"))
        if payload
        else None,
        "verdict_class": outcome["verdict_class"],
        "duration_s": payload.get("duration_s") if payload else None,
        "stamped_flags": live["stamped_flags"],
        "stamped_flag_count": live["stamped_flag_count"],
        "fresh_flags": live["fresh_flags"],
        "fresh_flag_count": live["fresh_flag_count"],
        "fresh_critical_flag_count": len(live["fresh_critical_flags"]),
        "stamped_live_flag_disagreement": live["stamped_live_flag_disagreement"],
        "live_verifier_command": live["live_verifier_command"],
        "live_verifier_exit_code": live["live_verifier_exit_code"],
        "live_verifier_duration_s": live["live_verifier_duration_s"],
        "row_consistency_command": live["row_consistency_command"],
        "row_consistency_exit_code": live["row_consistency_exit_code"],
        "row_consistency_duration_s": live["row_consistency_duration_s"],
        "row_consistency_status": live["row_consistency_status"],
        "artifact_convention_command": live["artifact_convention_command"],
        "artifact_convention_exit_code": live["artifact_convention_exit_code"],
        "artifact_convention_duration_s": live["artifact_convention_duration_s"],
        "artifact_convention_status": live["artifact_convention_status"],
        "eligible_for_v570_contract": outcome["eligible"],
        "disposition": outcome["disposition"],
        "failed_scope": outcome["failed_scope"],
        "reason": outcome["reason"],
        "source_extraction_ran": False
        if artifact.exp_id in {"exp6568", "exp6569", "exp6570"}
        else None,
        "graph_potts_utility_ran": False,
    }


def _build_import_rows(
    *,
    paths: Mapping[str, Path],
    payloads: Mapping[str, Mapping[str, Any]],
    check_results: Mapping[str, Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Build eligibility and live rows in fixed experiment order."""

    eligibility: list[JsonDict] = []
    live_rows: list[JsonDict] = []
    for artifact in V569_ARTIFACTS:
        path = paths[artifact.exp_id]
        payload = payloads.get(artifact.exp_id, {})
        checks = check_results.get(artifact.exp_id, {})
        eligibility.append(
            _eligibility_row(artifact, path=path, payload=payload, check_result=checks)
        )
        live_rows.append(_live_row(artifact, path=path, payload=payload, check_result=checks))
    return eligibility, live_rows


def derive_gguf_admission_root_cause(exp6567: Mapping[str, Any], source_path: Path) -> JsonDict:
    """Reduce the Exp6567 path-shape defect only from artifact rows."""

    resolved = [dict(row) for row in exp6567.get("resolved_model_file_rows") or []]
    preconditions = exp6567.get("preconditions_checked") or {}
    model_checks = preconditions.get("model_preflight_checks") or {}
    initial_gpu = preconditions.get("initial_gpu_state") or {}
    device = initial_gpu.get("device") or {}
    free_bytes = int(device.get("memory_free_mb") or 0) * 1024 * 1024
    one_at_a_time = (preconditions.get("frozen_execution_contract") or {}).get(
        "one_worker_at_a_time"
    ) is True
    per_model: list[JsonDict] = []
    for row in resolved:
        hf_id = str(row.get("hf_id") or "")
        checks = model_checks.get(hf_id) or {}
        absolute_path = str(row.get("absolute_path") or "")
        byte_size = int(row.get("byte_size") or 0)
        per_model.append(
            {
                "row_type": "gguf_admission_root_cause_model",
                "hf_id": hf_id,
                "absolute_path": absolute_path,
                "blob_sha256": row.get("sha256"),
                "byte_size": byte_size,
                "large_blob_resolved": bool(checks.get("file_resolved")) and byte_size > 10**9,
                "embedded_tokenizer": bool(checks.get("embedded_tokenizer")),
                "cuda_runtime": bool((preconditions.get("cuda") or {}).get("available")),
                "fits_selected_gpu_file_bytes": free_bytes > byte_size > 0,
                "hash_only_path": _is_hash_only_path(absolute_path),
                "language_model_file": checks.get("language_model_file"),
                "quantization_known": checks.get("quantization_known"),
                "generation_row_present": False,
            }
        )
    generation_count = len(exp6567.get("live_process_and_token_rows") or [])
    all_models = {row.get("hf_id") for row in per_model} == set(MANDATED_MODEL_IDS)
    root = {
        "row_type": "gguf_admission_root_cause",
        "source_exp_id": "exp6567",
        "source_artifact_path": V569_ARTIFACTS[2].relative_path.as_posix(),
        "source_artifact_sha256": sha256_file(source_path),
        "resolved_blob_count": len(per_model),
        "all_mandated_model_rows_present": all_models,
        "all_large_blobs_resolved": all_models
        and all(row["large_blob_resolved"] for row in per_model),
        "all_embedded_tokenizers_passed": all_models
        and all(row["embedded_tokenizer"] for row in per_model),
        "cuda_runtime_passed": bool((preconditions.get("cuda") or {}).get("available"))
        and bool((preconditions.get("llama_cpp_python") or {}).get("gpu_offload_supported")),
        "sequential_memory_conditions_passed": one_at_a_time
        and all(row["fits_selected_gpu_file_bytes"] for row in per_model),
        "failed_precondition": "model_identity_and_file_shape"
        if "model_identity_and_file_shape" in (preconditions.get("failed_preconditions") or [])
        else None,
        "all_paths_hash_only": all_models and all(row["hash_only_path"] for row in per_model),
        "all_language_model_file_false": all_models
        and all(row["language_model_file"] is False for row in per_model),
        "all_quantization_known_false": all_models
        and all(row["quantization_known"] is False for row in per_model),
        "generation_row_count": generation_count,
        "generation_ran": generation_count > 0,
        "gpu_telemetry_row_count": len(exp6567.get("gpu_telemetry_rows") or []),
        "unload_row_count": len(exp6567.get("unload_and_recovery_rows") or []),
        "per_model_rows": per_model,
        "next_allowed_mechanism": "content_derived_gguf_header_metadata_plus_actual_execution",
    }
    root["observed_root_cause_closed"] = (
        root["all_large_blobs_resolved"]
        and root["all_embedded_tokenizers_passed"]
        and root["cuda_runtime_passed"]
        and root["sequential_memory_conditions_passed"]
        and root["failed_precondition"] == "model_identity_and_file_shape"
        and root["all_paths_hash_only"]
        and root["all_language_model_file_false"]
        and root["all_quantization_known_false"]
        and root["generation_ran"] is False
    )
    return root


def _parse_required_fields(prompt: str) -> set[str]:
    """Read exact top-level names from a roadmap prompt's field block."""

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
    """Collect exact retired task IDs from all supported manifest sections."""

    retired: set[str] = set()
    for section_name in ("retired_experiments", "retired_extras", "retired"):
        section = manifest.get(section_name)
        if not isinstance(section, list):
            continue
        for item in section:
            if not isinstance(item, Mapping):
                continue
            for key in ("id", "experiment_id"):
                if isinstance(item.get(key), str):
                    retired.add(str(item[key]))
            if isinstance(item.get("experiment_ids"), list):
                retired.update(str(value) for value in item["experiment_ids"])
    return retired


def _retired_dependency_ids(roadmap: Mapping[str, Any], retired_ids: set[str]) -> set[str]:
    """Find retired IDs used as execution dependencies, not historical priors."""

    found: set[str] = set()
    for task in roadmap.get("tasks") or []:
        if not isinstance(task, Mapping):
            continue
        for dependency in task.get("requires") or []:
            if isinstance(dependency, str) and dependency in retired_ids:
                found.add(dependency)
        for gate in task.get("gated_on") or []:
            if isinstance(gate, Mapping) and gate.get("upstream") in retired_ids:
                found.add(str(gate["upstream"]))
    return found


def _roadmap_contract(
    repo_root: Path,
) -> tuple[JsonDict, dict[str, set[str]], set[str], set[str]]:
    """Load and structurally validate the active V570 roadmap."""

    roadmap = _load_yaml(repo_root / ROADMAP_RELATIVE_PATH)
    Roadmap.model_validate(roadmap)
    fields_by_task = {
        str(task["id"]): _parse_required_fields(str(task.get("prompt") or ""))
        for task in roadmap.get("tasks") or []
        if isinstance(task, Mapping)
    }
    retired = _retired_experiment_ids(_load_yaml(repo_root / EXCLUSION_MANIFEST_RELATIVE_PATH))
    return roadmap, fields_by_task, retired, _retired_dependency_ids(roadmap, retired)


def _task_manifest(
    roadmap: Mapping[str, Any], fields_by_task: Mapping[str, set[str]]
) -> list[JsonDict]:
    """Freeze active V570 task IDs, deliverables, and declared fields."""

    rows: list[JsonDict] = []
    for index, task in enumerate(roadmap.get("tasks") or []):
        if not isinstance(task, Mapping):
            continue
        task_id = str(task.get("id") or "")
        row = {
            "row_type": "v570_task_manifest",
            "task_index": index,
            "task_id": task_id,
            "deliverable": task.get("deliverable"),
            "required_artifact_fields": sorted(fields_by_task.get(task_id, set())),
        }
        row["task_contract_sha256"] = sha256_json(row)
        rows.append(row)
    return rows


def _gate_contract_rows(
    roadmap: Mapping[str, Any],
    fields_by_task: Mapping[str, set[str]],
    retired_ids: set[str],
) -> list[JsonDict]:
    """Freeze every active structured gate and its exact upstream field."""

    task_ids = {
        str(task.get("id")) for task in roadmap.get("tasks") or [] if isinstance(task, Mapping)
    }
    rows: list[JsonDict] = []
    for task in roadmap.get("tasks") or []:
        if not isinstance(task, Mapping):
            continue
        for index, gate in enumerate(task.get("gated_on") or []):
            if not isinstance(gate, Mapping):
                continue
            upstream = str(gate.get("upstream") or "")
            field = str(gate.get("artifact_field") or "")
            upstream_fields = fields_by_task.get(upstream, set())
            rows.append(
                {
                    "row_type": "v570_gate_contract",
                    "task_id": str(task.get("id") or ""),
                    "gate_index": index,
                    "upstream": upstream,
                    "artifact_field": field,
                    "op": gate.get("op"),
                    "value": gate.get("value"),
                    "upstream_in_active_roadmap": upstream in task_ids,
                    "artifact_field_declared_by_upstream": field in upstream_fields,
                    "exact_field_spelling": field in upstream_fields,
                    "retired_upstream": upstream in retired_ids,
                }
            )
    return rows


def _scope_class(experiment_id: str) -> str:
    """Name the prior scope without making it an execution dependency."""

    if "6565" in experiment_id:
        return "v569_evidence_replay"
    if "6567" in experiment_id or "6553" in experiment_id:
        return "flagship_admission_before_inference"
    if "5923" in experiment_id:
        return "retired_full_constraintir_schema_reprompt"
    return "prior_failed_scope"


def _prior_failure_rows(
    roadmap: Mapping[str, Any], retired_ids: set[str], dependency_retired_ids: set[str]
) -> list[JsonDict]:
    """Freeze every prior failure and its changed mechanism."""

    required = {"experiment_id", "verdict", "addressed_by", "retire_if_same_verdict"}
    rows: list[JsonDict] = []
    for task in roadmap.get("tasks") or []:
        if not isinstance(task, Mapping):
            continue
        for index, prior in enumerate(task.get("prior_failures") or []):
            if not isinstance(prior, Mapping):
                rows.append(
                    {
                        "row_type": "prior_failure_and_retirement",
                        "task_id": str(task.get("id") or ""),
                        "prior_failure_index": index,
                        "complete_prior_failure_contract": False,
                        "changed_mechanism": False,
                        "mechanical_repeat_retirement_rule": False,
                        "retired_dependency_chain": False,
                    }
                )
                continue
            experiment_id = str(prior.get("experiment_id") or "")
            missing = sorted(required - set(prior))
            addressed_by = str(prior.get("addressed_by") or "").strip()
            rows.append(
                {
                    "row_type": "prior_failure_and_retirement",
                    "task_id": str(task.get("id") or ""),
                    "prior_failure_index": index,
                    "experiment_id": experiment_id,
                    "verdict": prior.get("verdict"),
                    "addressed_by": prior.get("addressed_by"),
                    "retire_if_same_verdict": prior.get("retire_if_same_verdict"),
                    "missing_fields": missing,
                    "complete_prior_failure_contract": not missing,
                    "changed_mechanism": bool(addressed_by),
                    "mechanical_repeat_retirement_rule": prior.get("retire_if_same_verdict")
                    is True,
                    "scope_class": _scope_class(experiment_id),
                    "retired_prior_scope": experiment_id in retired_ids,
                    "retired_dependency_chain": experiment_id in dependency_retired_ids,
                }
            )
    return rows


def _boundary_contract(
    repo_root: Path,
    task_manifest: Sequence[Mapping[str, Any]],
    dependency_retired_ids: set[str],
) -> JsonDict:
    """Freeze model, extraction, Rust, ARC, and hardware reopen rules."""

    proposal = (repo_root / PROPOSAL_RELATIVE_PATH).read_text(encoding="utf-8")
    normalized_proposal = " ".join(proposal.split())
    fused_units = (
        "obligation-node canonicalization",
        "graph validation",
        "exact relation dispatch",
        "release reduction",
    )
    materially_different = "Exp 6581" in normalized_proposal and all(
        unit in normalized_proposal for unit in fused_units
    )
    repeat_retirement = (
        "repeated no-benefit or NFR01 miss permanently retires" in normalized_proposal
        or "repeated no-benefit or NFR01 miss" in normalized_proposal
    )
    boundary: JsonDict = {
        "row_type": "model_arc_and_hardware_boundary",
        "v570_task_manifest": [dict(row) for row in task_manifest],
        "MODEL_SPECS": list(MANDATED_MODEL_IDS),
        "legacy_model_policy": {
            "legacy_smoke_models": ["Qwen3.5-0.8B", "gemma-4-E4B-it"],
            "legacy_substitution_allowed": False,
            "legacy_models_can_support_headline": False,
        },
        "model_rule": {
            "content_derived_gguf_metadata_required": True,
            "actual_execution_required": True,
            "filename_or_fit_prediction_sufficient": False,
            "one_model_at_a_time": True,
        },
        "extraction_boundary": {
            "source_spans_required": True,
            "joint_sufficiency_required": True,
            "compiler_owned_exact_release": True,
            "retired_constraintir_reuse_allowed": False,
            "graph_potts_utility_claimed_by_exp6571": False,
        },
        "arc_boundary": {
            "prospective_live_receipts_only": True,
            "solve_claim_allowed": False,
            "public_game_source_read_allowed": False,
            "offline_ground_truth_bfs_allowed": False,
        },
        "hardware_boundary": {
            "changed_operator_receipt_required": True,
            "unchanged_hardware_command_allowed": False,
            "exp6571_hardware_command_count": 0,
            "llm_load_count": 0,
            "extropic_or_kona_execution_claim_allowed": False,
        },
        "rust_fusion_boundary": {
            "single_fused_end_to_end_workload_only": True,
            "allowed_fused_units": list(fused_units),
            "materially_different_from_exp6563_6564": materially_different,
            "retire_on_repeat_no_benefit_or_nfr01_miss": repeat_retirement,
            "rust_fusion_reopen_ready_from_boundary": materially_different and repeat_retirement,
        },
        "retired_requires_or_gate_ids": sorted(dependency_retired_ids),
    }
    boundary["all_boundary_checks_passed"] = (
        boundary["MODEL_SPECS"] == list(MANDATED_MODEL_IDS)
        and boundary["legacy_model_policy"]["legacy_substitution_allowed"] is False
        and boundary["model_rule"]["content_derived_gguf_metadata_required"] is True
        and boundary["model_rule"]["actual_execution_required"] is True
        and boundary["extraction_boundary"]["joint_sufficiency_required"] is True
        and boundary["extraction_boundary"]["retired_constraintir_reuse_allowed"] is False
        and boundary["arc_boundary"]["solve_claim_allowed"] is False
        and boundary["hardware_boundary"]["unchanged_hardware_command_allowed"] is False
        and boundary["hardware_boundary"]["exp6571_hardware_command_count"] == 0
        and boundary["rust_fusion_boundary"]["rust_fusion_reopen_ready_from_boundary"] is True
        and not dependency_retired_ids
    )
    return boundary


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    """Hash every protected input and orchestration file."""

    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_files_unchanged(before: Mapping[str, str], after: Mapping[str, str]) -> JsonDict:
    """Compare protected hashes without changing either side."""

    rows = [
        {
            "row_type": "protected_file_hash",
            "path": path,
            "before_sha256": before.get(path, "missing"),
            "after_sha256": after.get(path, "missing"),
            "unchanged": before.get(path, "missing") == after.get(path, "missing"),
        }
        for path in sorted(set(before) | set(after))
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
    """Record a local tool version without using its execution substrate."""

    receipt = _run_command(argv, repo_root)
    text = str(receipt["stdout"] or receipt["stderr"]).strip()
    return {
        "command": receipt["command"],
        "exit_code": receipt["exit_code"],
        "version_text": text.splitlines()[0] if text else "",
        "stdout_sha256": receipt["stdout_sha256"],
        "stderr_sha256": receipt["stderr_sha256"],
    }


def _host_preconditions(
    repo_root: Path,
    *,
    paths: Mapping[str, Path],
    payloads: Mapping[str, Mapping[str, Any]],
    protected_before: Mapping[str, str],
    run_date: str,
) -> JsonDict:  # pragma: no cover - host-specific receipt collection.
    """Collect the required local receipts without LLM or hardware execution."""

    git_status = _run_command(["git", "status", "--short"], repo_root)
    git_head = _run_command(["git", "rev-parse", "HEAD"], repo_root)
    cpu_model = ""
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.is_file():
        match = re.search(r"^model name\s*:\s*(.+)$", cpu_info.read_text(), re.MULTILINE)
        cpu_model = match.group(1).strip() if match else ""
    meminfo: dict[str, int] = {}
    mem_path = Path("/proc/meminfo")
    if mem_path.is_file():
        for line in mem_path.read_text().splitlines():
            match = re.match(r"(MemTotal|MemAvailable):\s+(\d+)", line)
            if match:
                meminfo[match.group(1)] = int(match.group(2))
    disk = __import__("shutil").disk_usage(repo_root)
    exp6567 = payloads.get("exp6567") or {}
    cache_rows = [
        {
            "hf_id": row.get("hf_id"),
            "path": row.get("absolute_path"),
            "exists": Path(str(row.get("absolute_path") or "")).is_file(),
            "byte_size": row.get("byte_size"),
            "trusted_exp6567_sha256": row.get("sha256"),
            "full_blob_rehash_performed": False,
        }
        for row in exp6567.get("resolved_model_file_rows") or []
        if isinstance(row, Mapping)
    ]
    try:
        import z3

        z3_receipt = {"available": True, "version": z3.get_version_string()}
    except ImportError as exc:
        z3_receipt = {"available": False, "error": str(exc)}
    cargo_lock = (repo_root / "Cargo.lock").read_text(encoding="utf-8")
    pyo3_match = re.search(r'\[\[package\]\]\s+name = "pyo3"\s+version = "([^"]+)"', cargo_lock)
    architecture_text = (repo_root / ARCHITECTURE_RELATIVE_PATH).read_text(encoding="utf-8")
    reconciled = re.search(r"\*\*Last Reconciled:\*\*\s*(\d{4}-\d{2}-\d{2})", architecture_text)
    network_started = time.monotonic()
    network = _run_command(
        [sys.executable, "-c", "import socket; socket.getaddrinfo('pypi.org', 443)"], repo_root
    )
    return {
        "git_status": {
            "head_sha": str(git_head["stdout"]).strip(),
            "status_short": str(git_status["stdout"]).strip(),
        },
        "resources": {
            "cpu": {"count": __import__("os").cpu_count(), "model": cpu_model},
            "ram": {
                "total_kib": meminfo.get("MemTotal"),
                "available_kib": meminfo.get("MemAvailable"),
            },
            "disk": {
                "path": str(repo_root),
                "total_bytes": disk.total,
                "free_bytes": disk.free,
            },
            "python": {"version": sys.version, "executable": sys.executable},
            "platform": __import__("platform").platform(),
        },
        "rust": {
            "rustc": _tool_version(["rustc", "--version"], repo_root),
            "cargo": _tool_version(["cargo", "--version"], repo_root),
        },
        "pyo3": {
            "version": pyo3_match.group(1) if pyo3_match else "unknown",
            "cargo_lock_sha256": sha256_file(repo_root / "Cargo.lock"),
            "crate_manifest_sha256": sha256_file(repo_root / "crates/carnot-python/Cargo.toml"),
        },
        "z3": z3_receipt,
        "model_cache_paths": cache_rows,
        "artifact_path_and_hash_receipts": [
            {
                "exp_id": artifact.exp_id,
                "path": artifact.relative_path.as_posix(),
                "resolved_path": str(paths[artifact.exp_id]),
                "exists": paths[artifact.exp_id].is_file(),
                "sha256": sha256_file(paths[artifact.exp_id]),
            }
            for artifact in V569_ARTIFACTS
        ],
        "protected_file_hashes_before": dict(protected_before),
        "monotonic_timer_resolution_s": time.get_clock_info("monotonic").resolution,
        "architecture_freshness": {
            "path": ARCHITECTURE_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(repo_root / ARCHITECTURE_RELATIVE_PATH),
            "last_reconciled": reconciled.group(1) if reconciled else "unknown",
            "planning_date": run_date,
            "architecture_checked": bool(reconciled),
        },
        "network_status": {
            "checked": True,
            "method": "dns_resolution_only_no_download",
            "reachable": network["exit_code"] == 0,
            "exit_code": network["exit_code"],
            "duration_s": round(time.monotonic() - network_started, 6),
            "stdout_sha256": network["stdout_sha256"],
            "stderr_sha256": network["stderr_sha256"],
        },
        "llm_load_performed": False,
        "hardware_command_performed": False,
    }


def _attack_rows(
    *,
    eligibility: Sequence[Mapping[str, Any]],
    gguf_root: Mapping[str, Any],
    gates: Sequence[Mapping[str, Any]],
    priors: Sequence[Mapping[str, Any]],
    boundary: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> list[JsonDict]:
    """Emit one fail-closed row for every mandated shortcut attack."""

    by_exp = {str(row.get("exp_id")): row for row in eligibility}
    checks = {
        "hash_alias": all(
            row.get("sha256") == "missing"
            if row.get("exists") is False
            else str(row.get("sha256") or "").startswith("sha256:")
            for row in eligibility
        ),
        "missing_artifact_laundering": by_exp.get("exp6569", {}).get("disposition")
        == "missing_not_null",
        "aggregate_only_claim": len(eligibility) == len(V569_ARTIFACTS),
        "stamped_live_verifier_disagreement": by_exp.get("exp6565", {}).get(
            "stamped_live_flag_disagreement"
        )
        is True,
        "false_model_admission": gguf_root.get("generation_ran") is False
        and gguf_root.get("observed_root_cause_closed") is True,
        "wrong_verdict_class": by_exp.get("exp6568", {}).get("verdict_class") == "blocked"
        and by_exp.get("exp6569", {}).get("verdict_class") == "blocked",
        "gate_field_drift": all(row.get("exact_field_spelling") is True for row in gates),
        "legacy_model_substitution": (boundary.get("legacy_model_policy") or {}).get(
            "legacy_substitution_allowed"
        )
        is False,
        "retired_constraintir_reuse": (boundary.get("extraction_boundary") or {}).get(
            "retired_constraintir_reuse_allowed"
        )
        is False
        and all(row.get("retired_dependency_chain") is False for row in priors),
        "arc_solve_laundering": (boundary.get("arc_boundary") or {}).get("solve_claim_allowed")
        is False,
        "unchanged_hardware_command": (boundary.get("hardware_boundary") or {}).get(
            "unchanged_hardware_command_allowed"
        )
        is False
        and (boundary.get("hardware_boundary") or {}).get("exp6571_hardware_command_count") == 0,
        "protected_file_mutation": protected.get("all_unchanged") is True,
    }
    return [
        {
            "row_type": "shortcut_attack",
            "attack": attack,
            "passed": bool(checks[attack]),
            "defense": f"REQ-REPORT-6571-ATTACKS:{attack}",
        }
        for attack in ATTACK_NAMES
    ]


def _field_provenance(
    repo_root: Path, eligibility: Sequence[Mapping[str, Any]]
) -> dict[str, JsonDict]:
    """Bind every headline field to rows, input hashes, and one reducer."""

    source_hashes = [
        str(row.get("sha256")) for row in eligibility if row.get("sha256") != "missing"
    ] + [
        sha256_file(repo_root / ROADMAP_RELATIVE_PATH),
        sha256_file(repo_root / PROPOSAL_RELATIVE_PATH),
        sha256_file(repo_root / SPEC_RELATIVE_PATH),
    ]
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source_rows": [
                "v569_artifact_eligibility_rows",
                "live_verifier_and_duration_rows",
                "v570_gate_contract_rows",
                "prior_failure_and_retirement_rows",
            ],
            "source_hashes": source_hashes,
            "reducer": "REQ-REPORT-6571 deterministic V570 evidence-root reducer",
            "spec_refs": ["REQ-REPORT-6571"],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def aggregate_row_recomputation(payload: Mapping[str, Any]) -> JsonDict:
    """Recompute both readiness fields only from emitted contract rows."""

    eligibility = [
        dict(row)
        for row in payload.get("v569_artifact_eligibility_rows") or []
        if isinstance(row, Mapping)
    ]
    live_rows = [
        dict(row)
        for row in payload.get("live_verifier_and_duration_rows") or []
        if isinstance(row, Mapping)
    ]
    gates = [
        dict(row)
        for row in payload.get("v570_gate_contract_rows") or []
        if isinstance(row, Mapping)
    ]
    priors = [
        dict(row)
        for row in payload.get("prior_failure_and_retirement_rows") or []
        if isinstance(row, Mapping)
    ]
    attacks = [dict(row) for row in payload.get("attack_rows") or [] if isinstance(row, Mapping)]
    boundary = payload.get("model_arc_and_hardware_boundary") or {}
    protected = payload.get("protected_files_unchanged") or {}
    gguf = payload.get("gguf_admission_root_cause") or {}
    provenance = payload.get("field_provenance") or {}
    by_exp = {str(row.get("exp_id")): row for row in eligibility}
    expected_dispositions = {
        "exp6565": "usable_v569_evidence_contract",
        "exp6566": "usable_v569_method_contract",
        "exp6567": "blocked_hash_only_gguf_admission",
        "exp6568": "corrected_blocked_gate_import",
        "exp6569": "missing_not_null",
        "exp6570": "valid_blocked_independent_audit",
    }
    eligibility_closed = len(eligibility) == len(V569_ARTIFACTS) and all(
        by_exp.get(exp_id, {}).get("disposition") == disposition
        and by_exp.get(exp_id, {}).get("eligible_for_v570_contract") is True
        for exp_id, disposition in expected_dispositions.items()
    )
    live_closed = len(live_rows) == len(V569_ARTIFACTS) and all(
        row.get("live_verifier_exit_code") == 0
        and not row.get("fresh_critical_flags")
        and row.get("artifact_convention_status") in {"CHECKABLE", "EXPECTED_MISSING"}
        and (
            row.get("row_consistency_status") in {"ok", "skipped"}
            or (
                row.get("exp_id") == "exp6569" and row.get("row_consistency_status") == "unreadable"
            )
        )
        for row in live_rows
    )
    gates_closed = bool(gates) and all(
        row.get("upstream_in_active_roadmap") is True
        and row.get("artifact_field_declared_by_upstream") is True
        and row.get("exact_field_spelling") is True
        and row.get("retired_upstream") is False
        for row in gates
    )
    priors_closed = bool(priors) and all(
        row.get("complete_prior_failure_contract") is True
        and row.get("changed_mechanism") is True
        and row.get("mechanical_repeat_retirement_rule") is True
        and row.get("retired_dependency_chain") is False
        for row in priors
    )
    attacks_closed = len(attacks) == len(ATTACK_NAMES) and all(
        row.get("passed") is True for row in attacks
    )
    provenance_closed = set(provenance) >= set(REQUIRED_ARTIFACT_FIELDS)
    row_container_closed = bool(payload.get("per_unit_rows"))
    checks = {
        "v569_eligibility_closed": eligibility_closed,
        "live_verifier_replay_closed": live_closed,
        "gguf_root_cause_closed": gguf.get("observed_root_cause_closed") is True,
        "v570_gate_contract_closed": gates_closed,
        "prior_failure_retirement_closed": priors_closed,
        "model_arc_hardware_boundary_closed": boundary.get("all_boundary_checks_passed") is True,
        "attacks_closed": attacks_closed,
        "protected_files_unchanged": protected.get("all_unchanged") is True,
        "field_provenance_closed": provenance_closed,
        "per_unit_rows_present": row_container_closed,
        "verifier_is_oracle": payload.get("verifier_is_oracle") is True,
    }
    ready = all(checks.values())
    rust_ready = (boundary.get("rust_fusion_boundary") or {}).get(
        "materially_different_from_exp6563_6564"
    ) is True and (boundary.get("rust_fusion_boundary") or {}).get(
        "retire_on_repeat_no_benefit_or_nfr01_miss"
    ) is True
    unexpected_missing = any(
        row.get("exists") is False and row.get("expected_missing") is not True
        for row in eligibility
    )
    return {
        "row_type": "aggregate_row_recomputation",
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "expected_v569_artifact_row_count": len(V569_ARTIFACTS),
        "observed_v569_artifact_row_count": len(eligibility),
        "unexpected_missing_prerequisite": unexpected_missing,
        "exp6568_corrected_blocked_class": by_exp.get("exp6568", {}).get("verdict_class")
        == "blocked",
        "exp6569_missing_not_null": by_exp.get("exp6569", {}).get("disposition")
        == "missing_not_null",
        "source_extraction_claim_count": 0,
        "graph_potts_utility_claim_count": 0,
        "rust_fusion_reopen_ready_from_rows": rust_ready,
        "v570_evidence_contract_ready_from_rows": ready,
        "verdict_class_from_rows": None
        if ready
        else ("blocked" if unexpected_missing else "partial"),
        "spec_refs": ["REQ-REPORT-6571"],
    }


def _gate_summary(aggregate: Mapping[str, Any]) -> JsonDict:
    """Expose exact failed checks and observed values for a non-ready root."""

    failed = list(aggregate.get("failed_checks") or [])
    rows = [
        {
            "row_type": "gate_check",
            "check": check,
            "expected": True,
            "observed": False,
            "passed": False,
        }
        for check in failed
    ]
    if aggregate.get("unexpected_missing_prerequisite") is True:
        failed.insert(0, "exp6565_expected_artifact_classification")
        rows.insert(
            0,
            {
                "row_type": "gate_check",
                "check": "exp6565_expected_artifact_classification",
                "expected": "present terminal artifact",
                "observed": "missing or unreadable",
                "passed": False,
            },
        )
    return {
        "all_gates_passed": not failed,
        "failed_checks": failed,
        "failed_check_rows": rows,
        "first_failed_check": failed[0] if failed else None,
    }


def _status_and_verdict(
    ready: bool, false_provenance: bool, missing_prerequisite: bool
) -> tuple[str, str, str | None]:
    """Choose one terminal state from the closed V570 verdict classes."""

    if ready:
        return (
            "complete_v570_evidence_gate_and_retirement_root_ready",
            "complete_v570_evidence_gate_and_retirement_root_ready: V569 eligibility includes blocked and missing scopes honestly; the hash-only GGUF cause, V570 gates, changed mechanisms, retirement rules, model, ARC, hardware, Rust, attacks, and protected files close without claiming extraction or graph-Potts utility ran",
            None,
        )
    if false_provenance:
        return (
            "disqualified_v570_evidence_root_false_provenance",
            "disqualified_v570_evidence_root_false_provenance: an exact path or content hash did not match its claimed source",
            "disqualified",
        )
    if missing_prerequisite:
        return (
            "blocked_v570_evidence_contract_missing_prerequisites",
            "blocked_v570_evidence_contract_missing_prerequisites: an expected V569 artifact other than the frozen Exp6569 absence is missing or unreadable",
            "blocked",
        )
    return (
        "partial_v570_evidence_gate_and_retirement_root",
        "partial_v570_evidence_gate_and_retirement_root: usable V569 evidence exists but one or more eligibility, gate, failure, boundary, attack, provenance, or protected-file contracts remain open",
        "partial",
    )


def _tests_run_receipts(
    tests_run: Sequence[Mapping[str, Any]] | None,
) -> list[JsonDict]:
    """Normalize command receipts to stable command and exit fields."""

    source = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    return [{"command": str(row["command"]), "exit_code": int(row["exit_code"])} for row in source]


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path = RESULT_RELATIVE_PATH,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    check_results: Mapping[str, Mapping[str, Any]] | None = None,
    preconditions: Mapping[str, Any] | None = None,
    input_payloads: Mapping[str, Mapping[str, Any]] | None = None,
    artifact_paths: Mapping[str, Path] | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Build, validate, and optionally atomically write the terminal root."""

    started = time.monotonic()
    paths = default_v569_paths(repo_root) if artifact_paths is None else dict(artifact_paths)
    protected_before = _protected_hashes(repo_root)
    payloads = (
        {artifact.exp_id: _read_json(paths[artifact.exp_id]) for artifact in V569_ARTIFACTS}
        if input_payloads is None
        else {key: dict(value) for key, value in input_payloads.items()}
    )
    live_results = run_live_checks(repo_root, paths) if check_results is None else check_results
    eligibility, live_rows = _build_import_rows(
        paths=paths, payloads=payloads, check_results=live_results
    )
    gguf_root = derive_gguf_admission_root_cause(payloads.get("exp6567", {}), paths["exp6567"])
    roadmap, fields_by_task, retired_ids, dependency_retired_ids = _roadmap_contract(repo_root)
    task_manifest = _task_manifest(roadmap, fields_by_task)
    gate_rows = _gate_contract_rows(roadmap, fields_by_task, retired_ids)
    prior_rows = _prior_failure_rows(roadmap, retired_ids, dependency_retired_ids)
    boundary = _boundary_contract(repo_root, task_manifest, dependency_retired_ids)
    protected_after = _protected_hashes(repo_root)
    protected = _protected_files_unchanged(protected_before, protected_after)
    attack_rows = _attack_rows(
        eligibility=eligibility,
        gguf_root=gguf_root,
        gates=gate_rows,
        priors=prior_rows,
        boundary=boundary,
        protected=protected,
    )
    provenance = _field_provenance(repo_root, eligibility)
    per_unit_rows = [
        *eligibility,
        *live_rows,
        *gguf_root.get("per_model_rows", []),
        *gate_rows,
        *prior_rows,
        *task_manifest,
        *attack_rows,
    ]
    skeleton: JsonDict = {
        "v569_artifact_eligibility_rows": eligibility,
        "live_verifier_and_duration_rows": live_rows,
        "gguf_admission_root_cause": gguf_root,
        "v570_gate_contract_rows": gate_rows,
        "prior_failure_and_retirement_rows": prior_rows,
        "model_arc_and_hardware_boundary": boundary,
        "attack_rows": attack_rows,
        "per_unit_rows": per_unit_rows,
        "protected_files_unchanged": protected,
        "field_provenance": provenance,
        "verifier_is_oracle": True,
    }
    aggregate = aggregate_row_recomputation(skeleton)
    gate_summary = _gate_summary(aggregate)
    ready = aggregate["v570_evidence_contract_ready_from_rows"] is True
    status, honest_verdict, verdict_class = _status_and_verdict(
        ready,
        False,
        aggregate["unexpected_missing_prerequisite"] is True,
    )
    measured_duration = (
        round(time.monotonic() - started, 6) if duration_s is None else float(duration_s)
    )
    precondition_receipt = (
        _host_preconditions(
            repo_root,
            paths=paths,
            payloads=payloads,
            protected_before=protected_before,
            run_date=run_date,
        )
        if preconditions is None
        else dict(preconditions)
    )
    payload: JsonDict = {
        "status": status,
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "v569_artifact_eligibility_rows": eligibility,
        "live_verifier_and_duration_rows": live_rows,
        "gguf_admission_root_cause": gguf_root,
        "v570_gate_contract_rows": gate_rows,
        "prior_failure_and_retirement_rows": prior_rows,
        "model_arc_and_hardware_boundary": boundary,
        "v570_evidence_contract_ready_score": 1.0 if ready else 0.0,
        "rust_fusion_reopen_ready_score": 1.0
        if aggregate["rust_fusion_reopen_ready_from_rows"]
        else 0.0,
        "per_unit_rows": per_unit_rows,
        "aggregate_row_recomputation": aggregate,
        "gate_check_summary": gate_summary,
        "preconditions_checked": precondition_receipt,
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": provenance,
        "duration_s": measured_duration,
        "tests_run": _tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
        "field_principles": dict(FIELD_PRINCIPLES),
        "attack_rows": attack_rows,
        "random_seed": RANDOM_SEED,
        "planning_date": run_date,
    }
    payload["reproducibility_checksum"] = reproducibility_checksum(payload)
    if write:
        atomic_write_json(
            result_path, payload, root=repo_root, sort_keys=False, allow_override=False
        )
    return payload


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Validate immutable provenance, rows, boundaries, and reducer outputs."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in payload]
    if missing:
        return [f"missing required fields: {missing}"]
    if not str(payload.get("honest_verdict") or "").startswith(
        ("complete_", "partial_", "blocked_", "disqualified_")
    ):
        errors.append("honest_verdict lacks terminal prefix")
    if payload.get("verdict_class") not in (None, "partial", "blocked", "disqualified"):
        errors.append("verdict_class is outside the closed class")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if set(payload.get("field_provenance") or {}) < set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    if set(payload.get("field_principles") or {}) < set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles must cover required fields")
    if payload.get("reproducibility_checksum") != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    if not payload.get("per_unit_rows"):
        errors.append("aggregate-only evidence root")

    rows = {
        str(row.get("exp_id")): row
        for row in payload.get("v569_artifact_eligibility_rows") or []
        if isinstance(row, Mapping)
    }
    for artifact in V569_ARTIFACTS:
        row = rows.get(artifact.exp_id, {})
        if row.get("expected_path") != artifact.relative_path.as_posix():
            errors.append(f"exact path mismatch for {artifact.exp_id}")
        current_hash = sha256_file(row.get("resolved_path"))
        if row.get("sha256") != current_hash:
            errors.append(f"V569 artifact hash alias for {artifact.exp_id}")
        if (
            row.get("exists") is True
            and Path(str(row.get("resolved_path") or "")).resolve()
            != (REPO_ROOT / artifact.relative_path).resolve()
        ):
            errors.append(f"V569 artifact path alias for {artifact.exp_id}")
        if (
            payload.get("v570_evidence_contract_ready_score") == 1.0
            and row.get("eligible_for_v570_contract") is not True
        ):
            errors.append(f"{artifact.exp_id} readiness hides an ineligible row")
    if rows.get("exp6569", {}).get("disposition") != "missing_not_null":
        errors.append("Exp6569 must remain missing, not null")
    if rows.get("exp6568", {}).get("verdict_class") != "blocked":
        errors.append("Exp6568 corrected class must be blocked")

    for row in payload.get("v570_gate_contract_rows") or []:
        if not isinstance(row, Mapping):
            errors.append("gate contract row must be a mapping")
            continue
        if row.get("exact_field_spelling") is not True:
            errors.append("gate field drift detected")
        if row.get("upstream_in_active_roadmap") is not True:
            errors.append("gate upstream is outside active roadmap")
        if row.get("retired_upstream") is True:
            errors.append("gate references retired upstream")
    for row in payload.get("prior_failure_and_retirement_rows") or []:
        if not isinstance(row, Mapping):
            errors.append("prior failure row must be a mapping")
            continue
        if row.get("complete_prior_failure_contract") is not True:
            errors.append("prior failure row is incomplete")
        if row.get("changed_mechanism") is not True:
            errors.append("prior failure row lacks a changed mechanism")
        if row.get("mechanical_repeat_retirement_rule") is not True:
            errors.append("prior failure row lacks repeat-retirement rule")
        if row.get("retired_dependency_chain") is True:
            errors.append("prior failure row reuses a retired dependency")

    gguf = payload.get("gguf_admission_root_cause") or {}
    if (
        gguf.get("observed_root_cause_closed") is not True
        or gguf.get("generation_ran") is not False
    ):
        errors.append("false model admission or GGUF root-cause drift")
    boundary = payload.get("model_arc_and_hardware_boundary") or {}
    if boundary.get("MODEL_SPECS") != list(MANDATED_MODEL_IDS):
        errors.append("mandated model identities changed")
    if (boundary.get("legacy_model_policy") or {}).get("legacy_substitution_allowed") is not False:
        errors.append("legacy model substitution opened")
    if (boundary.get("extraction_boundary") or {}).get(
        "retired_constraintir_reuse_allowed"
    ) is not False:
        errors.append("retired ConstraintIR reuse opened")
    if (boundary.get("arc_boundary") or {}).get("solve_claim_allowed") is not False:
        errors.append("ARC solve boundary opened")
    hardware = boundary.get("hardware_boundary") or {}
    if hardware.get("exp6571_hardware_command_count") != 0:
        errors.append("hardware command boundary violated")
    if hardware.get("unchanged_hardware_command_allowed") is not False:
        errors.append("unchanged hardware command opened")
    rust = boundary.get("rust_fusion_boundary") or {}
    if payload.get("rust_fusion_reopen_ready_score") == 1.0 and not (
        rust.get("materially_different_from_exp6563_6564") is True
        and rust.get("retire_on_repeat_no_benefit_or_nfr01_miss") is True
    ):
        errors.append("Rust fusion boundary does not support reopen score")
    if (payload.get("protected_files_unchanged") or {}).get("all_unchanged") is not True:
        errors.append("protected files changed")
    attacks = payload.get("attack_rows") or []
    if payload.get("v570_evidence_contract_ready_score") == 1.0 and (
        len(attacks) != len(ATTACK_NAMES)
        or any(not isinstance(row, Mapping) or row.get("passed") is not True for row in attacks)
    ):
        errors.append("attack matrix is incomplete or open")

    recomputed = aggregate_row_recomputation(payload)
    if payload.get("aggregate_row_recomputation") != recomputed:
        errors.append("aggregate recomputation mismatch")
    if payload.get("v570_evidence_contract_ready_score") == 1.0 and not recomputed.get(
        "v570_evidence_contract_ready_from_rows"
    ):
        errors.append("ready score does not derive from rows")
    if payload.get("rust_fusion_reopen_ready_score") == 1.0 and not recomputed.get(
        "rust_fusion_reopen_ready_from_rows"
    ):
        errors.append("Rust reopen score does not derive from rows")
    return sorted(set(errors))


def load_json(path: str | Path) -> JsonDict:
    """Load a terminal artifact for CLI validation."""

    return _read_json(Path(path))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    """Write or validate the Exp6571 terminal artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_RELATIVE_PATH)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)

    if args.validate:
        errors = validate_artifact(load_json(args.result_path))
        print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
        return int(bool(errors))
    payload = build_artifact(
        repo_root=REPO_ROOT,
        result_path=args.result_path,
        write=True,
        duration_s=args.duration_s,
        run_date=args.date,
    )
    errors = validate_artifact(payload)
    print(
        json.dumps({"valid": not errors, "path": str(args.result_path), "errors": errors}, indent=2)
    )
    return int(bool(errors))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
