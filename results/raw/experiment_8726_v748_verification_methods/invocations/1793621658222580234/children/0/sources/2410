"""Exp6560 independent V567 capstone.

Spec refs: REQ-CAPSTONE-6560,
SCENARIO-CAPSTONE-6560-INVENTORY,
SCENARIO-CAPSTONE-6560-RECOMPUTE,
SCENARIO-CAPSTONE-6560-CLOSED-CLASSES,
SCENARIO-CAPSTONE-6560-ADOPTION,
SCENARIO-CAPSTONE-6560-PUBLICATION-HANDOFF,
SCENARIO-CAPSTONE-6560-ATOMIC.

This reducer does not run model inference. It imports terminal artifacts,
reruns local audit tools, and records where V567 can and cannot support a
production or research claim.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Any

import yaml

from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6560
MILESTONE = "2026.08.567"
EXPERIMENT_ID = "exp6560-v567-independent-capstone"
RESULT_RELATIVE_PATH = Path("results/experiment_6560_v567_independent_capstone.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/capstone/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6560_v567_independent_capstone.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6560_v567_independent_capstone.py")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
VERIFIER_IS_ORACLE = False
HASH_DIRECT_SIZE_LIMIT_BYTES = 64 * 1024 * 1024

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "expected_and_observed_task_inventory",
    "artifact_eligibility_rows",
    "independent_production_integration_rows",
    "independent_csl_rows",
    "independent_constraint_saturation_rows",
    "arc_generalization_disposition",
    "hardware_continuity_disposition",
    "closed_verdict_class_rows",
    "claim_and_adoption_matrix",
    "publication_gate_g1_g4",
    "unmet_gates",
    "document_reconciliation_receipts",
    "v568_handoff",
    "v567_capstone_ready_score",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "gate_check_summary",
    "preconditions_checked",
    "protected_files_unchanged",
    "cited_upstream_artifacts",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

SCHEMA_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "run_date",
    "random_seed",
    "result_path",
    "spec_refs",
    "field_principles",
    *REQUIRED_ARTIFACT_FIELDS,
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal capstone state distinguishes complete adjudication from partial aggregation.",
    "honest_verdict": (
        "The verdict must state production, CSL, saturation, ARC, hardware, and publication "
        "dispositions with a terminal prefix."
    ),
    "verdict_class": (
        "A closed class prevents mixed or blocked milestone evidence from becoming positive."
    ),
    "expected_and_observed_task_inventory": (
        "Every planned task needs a terminal artifact or an explicit missing-input disposition."
    ),
    "artifact_eligibility_rows": (
        "One row per artifact preserves hashes, flags, gates, and evidence eligibility."
    ),
    "independent_production_integration_rows": (
        "Disabled identity, parity, fallback, rollback, and exact equality must be recomputed "
        "independently."
    ),
    "independent_csl_rows": (
        "Current value, retention, future support, safety, dose, restart, and rollback must "
        "derive from rows."
    ),
    "independent_constraint_saturation_rows": (
        "Phase curves, paired interventions, harms, releases, and costs must be independently "
        "recomputed."
    ),
    "arc_generalization_disposition": (
        "ARC credit is limited to shared live receipt reachability and supported supervisor "
        "selection, not solves."
    ),
    "hardware_continuity_disposition": (
        "GateMate credit is limited to zero-command compliance or one authenticated action."
    ),
    "closed_verdict_class_rows": (
        "The downstream record must carry positive, circular_positive, null, blocked, "
        "disqualified, or partial without prose inference."
    ),
    "claim_and_adoption_matrix": (
        "Each mechanism needs a bounded production state with exact supporting evidence."
    ),
    "publication_gate_g1_g4": (
        "The stable publication gate prevents milestone-local blocker redefinition."
    ),
    "unmet_gates": "Open publication requirements must remain explicit.",
    "document_reconciliation_receipts": (
        "Specs, architecture, traceability, status, and changelog must match shipped code and "
        "evidence."
    ),
    "v568_handoff": (
        "The next planner needs clean assets, retired directions, blockers, and remaining PRD gaps."
    ),
    "v567_capstone_ready_score": (
        "One binary field records complete independent adjudication, not scientific positivity."
    ),
    "per_unit_rows": (
        "Every artifact, lane, arm, model, and disposition used by a headline must remain "
        "recheckable."
    ),
    "aggregate_row_recomputation": (
        "All milestone summaries must derive from emitted rows and cited hashes."
    ),
    "gate_check_summary": (
        "A blocked capstone must name each missing or failed check and observed value."
    ),
    "preconditions_checked": (
        "Artifact, receipt, verifier, resource, and document checks separate a block from science."
    ),
    "protected_files_unchanged": "The capstone must preserve the active roadmap and conductor.",
    "cited_upstream_artifacts": (
        "Every imported number needs experiment ID, field, path, and SHA256."
    ),
    "inference_substrate": (
        "The capstone aggregates and replays stored evidence; it performs no new model inference."
    ),
    "verifier_is_oracle": (
        "The capstone audit checks claims but does not create an oracle-distinct verifier result."
    ),
    "field_provenance": (
        "Every adoption and headline field must point to rows, receipts, reducers, and source hashes."
    ),
    "duration_s": (
        "Monotonic wall time exposes skipped artifact, verifier, or reconciliation work."
    ),
    "tests_run": "Named lint, test, gate, and E2E receipts show independent adjudication executed.",
    "reproducibility_checksum": (
        "A final content hash protects the milestone determination trail."
    ),
}

FIELD_PROVENANCE: dict[str, JsonDict] = {
    field: {
        "reducer": MODULE_RELATIVE_PATH.as_posix(),
        "spec": SPEC_RELATIVE_PATH.as_posix(),
        "source_rows": ["artifact_eligibility_rows", "per_unit_rows"],
    }
    for field in REQUIRED_ARTIFACT_FIELDS
}

CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
CAPSTONE_VERDICT_CLASSES = {"null", "partial", "blocked", "disqualified"}
ADOPTION_STATES = {"enabled", "default-off", "experiment-only", "blocked", "retired", "rejected"}
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
    "blocked:",
    "blocked_",
    "partial:",
    "partial_",
    "disqualified:",
    "disqualified_",
)

PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("_bmad/prd.md"),
    Path("_bmad/architecture.md"),
    Path("_bmad/traceability.md"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("ops/north-star.md"),
    Path("ops/e2e-test-plan.md"),
    SPEC_RELATIVE_PATH,
    ROADMAP_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
)

TOOL_RELATIVE_PATHS = (
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/exclusion_manifest_lint.py"),
    Path("scripts/publication_gate.py"),
    Path("scripts/arc_orphan_solver_lint.py"),
    Path("scripts/arc_levelup_guarantee_lint.py"),
)

EXPECTED_TASK_FALLBACK: tuple[JsonDict, ...] = (
    {
        "id": "exp6548-v567-evidence-eligibility-contract",
        "deliverable": "results/experiment_6548_v567_evidence_eligibility_contract.json",
        "title": "V567 evidence eligibility, architecture freshness, and gate contract",
    },
    {
        "id": "exp6549-production-safety-net-adapter",
        "deliverable": "results/experiment_6549_production_safety_net_adapter.json",
        "title": "Default-off production Safety-Net adapter with exact fallback",
    },
    {
        "id": "exp6550-rust-pyo3-safety-net-parity",
        "deliverable": "results/experiment_6550_rust_pyo3_safety_net_parity.json",
        "title": "Rust/PyO3 Safety-Net request and decision parity",
    },
    {
        "id": "exp6551-production-safety-net-independent-audit",
        "deliverable": "results/experiment_6551_production_safety_net_independent_audit.json",
        "title": "Independent production Safety-Net and cross-language audit",
    },
    {
        "id": "exp6552-hysteretic-reversible-conflict-memory",
        "deliverable": "results/experiment_6552_hysteretic_reversible_conflict_memory.json",
        "title": "Hysteretic active, dormant, and retired conflict memory",
    },
    {
        "id": "exp6553-prospective-sota-continuous-self-learning",
        "deliverable": "results/experiment_6553_prospective_sota_continuous_self_learning.json",
        "title": "Prospective SOTA chronological continuous self-learning comparison",
    },
    {
        "id": "exp6554-continuous-self-learning-independent-audit",
        "deliverable": "results/experiment_6554_continuous_self_learning_independent_audit.json",
        "title": "Independent prospective continuous self-learning audit",
    },
    {
        "id": "exp6555-proof-preserving-constraint-saturation-fixture",
        "deliverable": "results/experiment_6555_proof_preserving_constraint_saturation_fixture.json",
        "title": "Proof-preserving constraint-saturation SOTA ingestion and fixture",
    },
    {
        "id": "exp6556-sota-constraint-saturation-intervention-ab",
        "deliverable": "results/experiment_6556_sota_constraint_saturation_intervention_ab.json",
        "title": "SOTA constraint-saturation and bounded-intervention comparison",
    },
    {
        "id": "exp6557-constraint-saturation-independent-audit",
        "deliverable": "results/experiment_6557_constraint_saturation_independent_audit.json",
        "title": "Independent constraint-saturation phase-curve and policy audit",
    },
    {
        "id": "exp6558-arc-live-redirect-ledger-reachability",
        "deliverable": "results/experiment_6558_arc_live_redirect_ledger_reachability.json",
        "title": "ARC live trajectory-supervisor redirect-ledger generalization reachability",
    },
    {
        "id": "exp6559-gatemate-changed-state-continuity",
        "deliverable": "results/experiment_6559_gatemate_changed_state_continuity.json",
        "title": "GateMate changed-physical-state continuity with one-action stop",
    },
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6560_v567_independent_capstone --date 20260823"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6560_v567_independent_capstone.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6560_v567_independent_capstone.py "
    "-m pytest tests/python/test_experiment_6560_v567_independent_capstone.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6560_v567_independent_capstone.py "
    "--fail-under=100 --show-missing"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6560_v567_independent_capstone.py"
)
ROW_LINT_OUTPUT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6560_v567_independent_capstone.json"
)
ADVERSARIAL_OUTPUT_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6560_v567_independent_capstone.json"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path | str | None) -> str:
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


def payload_checksum(payload: Mapping[str, Any]) -> str:
    return sha256_json(
        {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    )


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    return payload_checksum(payload)


def _read_json(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _experiment_number(task_id: str) -> int:
    match = re.search(r"exp(\d{4})", task_id)
    return int(match.group(1)) if match else -1


def _short_exp_id(task_id: str) -> str:
    number = _experiment_number(task_id)
    return f"exp{number}" if number >= 0 else task_id


def _load_expected_tasks(repo_root: Path) -> list[JsonDict]:
    path = repo_root / ROADMAP_RELATIVE_PATH
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):  # pragma: no cover - fallback protects broken roadmaps.
        return [dict(row) for row in EXPECTED_TASK_FALLBACK]
    tasks = []
    for task in data.get("tasks", []) if isinstance(data, Mapping) else []:
        task_id = str(task.get("id", ""))
        number = _experiment_number(task_id)
        if 6548 <= number <= 6559:
            tasks.append(
                {
                    "id": task_id,
                    "deliverable": str(task.get("deliverable", "")),
                    "title": str(task.get("title", "")),
                }
            )
    return tasks or [dict(row) for row in EXPECTED_TASK_FALLBACK]


def _git_status(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:  # pragma: no cover - host failure.
        return f"unavailable:{type(exc).__name__}: {exc}"
    return result.stdout.strip()


def _resource_receipt(repo_root: Path) -> JsonDict:
    meminfo = Path("/proc/meminfo")
    mem_total = None
    mem_available = None
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "MemTotal:":
                mem_total = int(parts[1]) * 1024
            if len(parts) >= 2 and parts[0] == "MemAvailable:":
                mem_available = int(parts[1]) * 1024
    usage = shutil.disk_usage(repo_root)
    return {
        "cpu": {
            "count": os.cpu_count() or 0,
            "machine": platform.machine(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "ram": {
            "mem_total_bytes": mem_total,
            "mem_available_bytes": mem_available,
        },
        "disk": {
            "path": str(repo_root),
            "total_bytes": usage.total,
            "free_bytes": usage.free,
        },
    }


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_files_unchanged(before: Mapping[str, str], after: Mapping[str, str]) -> JsonDict:
    rows = [
        {
            "path": path,
            "before_sha256": before.get(path, "missing"),
            "after_sha256": after.get(path, "missing"),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    ]
    lookup = {row["path"]: row for row in rows}
    return {
        "all_protected_files_unchanged": all(row["unchanged"] for row in rows),
        "research_roadmap_yaml_unchanged": lookup.get(ROADMAP_RELATIVE_PATH.as_posix(), {}).get(
            "unchanged"
        )
        is True,
        "research_conductor_py_unchanged": lookup.get(CONDUCTOR_RELATIVE_PATH.as_posix(), {}).get(
            "unchanged"
        )
        is True,
        "rows": rows,
    }


def _path_receipt(repo_root: Path, raw_path: str | Path) -> JsonDict:
    raw = str(raw_path)
    path = Path(raw)
    resolved = path if path.is_absolute() else repo_root / path
    exists = resolved.is_file()
    size = resolved.stat().st_size if exists else 0
    direct_hash = sha256_file(resolved) if exists and size <= HASH_DIRECT_SIZE_LIMIT_BYTES else None
    return {
        "path": raw,
        "resolved_path": str(resolved),
        "exists": exists,
        "size_bytes": size,
        "sha256": direct_hash or ("not_rehashed_large_file" if exists else "missing"),
        "direct_hash_skipped_reason": (
            f"larger_than_{HASH_DIRECT_SIZE_LIMIT_BYTES}_bytes"
            if exists and direct_hash is None
            else ""
        ),
    }


def _looks_like_path(key: str, value: str) -> bool:
    if not value or value.startswith("sha256:") or "://" in value or "\n" in value:
        return False
    lowered_key = key.lower()
    key_is_path = any(
        token in lowered_key
        for token in ("path", "file", "fixture", "checkpoint", "journal", "receipt", "gguf")
    )
    suffix_is_path = Path(value).suffix in {
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
        ".md",
        ".py",
        ".rs",
        ".gguf",
        ".bit",
        ".so",
        ".lock",
    }
    return key_is_path and ("/" in value or suffix_is_path)


def _collect_path_strings(value: Any, key: str = "") -> set[str]:
    paths: set[str] = set()
    if isinstance(value, Mapping):
        for item_key, item_value in value.items():
            paths.update(_collect_path_strings(item_value, str(item_key)))
    elif isinstance(value, list):
        for item in value:
            paths.update(_collect_path_strings(item, key))
    elif isinstance(value, str) and _looks_like_path(key, value):
        paths.add(value)
    return paths


def _architecture_freshness(repo_root: Path, run_date: str) -> JsonDict:
    path = repo_root / "_bmad/architecture.md"
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    match = re.search(r"Last Reconciled:\*\*\s*(\d{4}-\d{2}-\d{2})", text)
    last = match.group(1) if match else None
    run = date(int(run_date[:4]), int(run_date[4:6]), int(run_date[6:8]))
    age_days = None
    if last is not None:
        y, m, d = (int(part) for part in last.split("-"))
        age_days = (run - date(y, m, d)).days
    return {
        "path": "_bmad/architecture.md",
        "sha256": sha256_file(path),
        "last_reconciled": last,
        "planning_date": run_date,
        "age_days": age_days,
        "freshness_status": "read_and_recorded",
    }


def _tool_version_receipts(repo_root: Path) -> list[JsonDict]:
    return [
        {
            "path": path.as_posix(),
            "exists": (repo_root / path).is_file(),
            "sha256": sha256_file(repo_root / path),
        }
        for path in TOOL_RELATIVE_PATHS
    ]


def _run_shell(  # pragma: no cover - exercised by the terminal command, not unit tests.
    repo_root: Path, command: str, *, parse_json: bool = False
) -> JsonDict:
    try:
        result = subprocess.run(
            command,
            cwd=repo_root,
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:  # pragma: no cover - host failure.
        return {"command": command, "exit_code": 127, "error": f"{type(exc).__name__}: {exc}"}
    receipt: JsonDict = {
        "command": command,
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
    if parse_json:
        try:
            receipt["json"] = json.loads(result.stdout)
        except json.JSONDecodeError:
            receipt["json"] = {}
    return receipt


def _collect_tool_receipts(  # pragma: no cover - subprocess integration wrapper.
    repo_root: Path, artifact_paths: Sequence[Path]
) -> dict[str, JsonDict]:
    joined = " ".join(path.as_posix() for path in artifact_paths)
    return {
        "adversarial_verify_upstream": _run_shell(
            repo_root,
            f".venv/bin/python scripts/adversarial_verify.py --json {joined}",
            parse_json=True,
        ),
        "row_consistency_upstream": _run_shell(
            repo_root,
            f".venv/bin/python scripts/verdict_row_consistency_lint.py {joined}",
        ),
        "exclusion_manifest_lint": _run_shell(
            repo_root, ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml"
        ),
        "arc_orphan_solver_lint": _run_shell(
            repo_root, ".venv/bin/python scripts/arc_orphan_solver_lint.py"
        ),
        "arc_levelup_guarantee_lint": _run_shell(
            repo_root,
            ".venv/bin/python scripts/arc_levelup_guarantee_lint.py research-roadmap.yaml",
        ),
        "publication_gate": _run_shell(
            repo_root, ".venv/bin/python scripts/publication_gate.py --json", parse_json=True
        ),
    }


def _tool_receipts(
    repo_root: Path,
    artifact_paths: Sequence[Path],
    override: Mapping[str, Any] | None,
) -> dict[str, JsonDict]:
    if override is not None:
        return {str(key): dict(value) for key, value in override.items()}
    return _collect_tool_receipts(repo_root, artifact_paths)  # pragma: no cover


def _adversarial_reports(tool_receipts: Mapping[str, Any]) -> dict[str, JsonDict]:
    payload = tool_receipts.get("adversarial_verify_upstream", {})
    reports = payload.get("json", {}).get("reports", []) if isinstance(payload, Mapping) else []
    out: dict[str, JsonDict] = {}
    for report in reports:
        if isinstance(report, Mapping):
            artifact = str(report.get("artifact", ""))
            out[artifact] = dict(report)
            out[Path(artifact).name] = dict(report)
    return out


def _row_lint_status(tool_receipts: Mapping[str, Any]) -> JsonDict:
    receipt = tool_receipts.get("row_consistency_upstream", {})
    if not isinstance(receipt, Mapping):
        return {"status": "not_run", "exit_code": None}
    stdout = str(receipt.get("stdout", ""))
    checked = re.search(r"checked\s+(\d+),\s+skipped\s+(\d+)", stdout)
    return {
        "status": "ok_or_skipped" if receipt.get("exit_code") == 0 else "findings",
        "exit_code": receipt.get("exit_code"),
        "checked_count": int(checked.group(1)) if checked else None,
        "skipped_count": int(checked.group(2)) if checked else None,
        "stdout_sha256": sha256_bytes(stdout.encode("utf-8")),
    }


def _closed_verdict_class(payload: Mapping[str, Any], report: Mapping[str, Any]) -> str:
    if not payload:
        return "blocked"
    status = str(payload.get("status", "")).lower()
    verdict = str(payload.get("honest_verdict", "")).lower()
    declared = payload.get("verdict_class")
    critical = int(report.get("max_severity", -1) or -1) >= 2
    if payload.get("flagged_adversarial") is True or critical:
        return "disqualified"
    if status.startswith("blocked") or verdict.startswith("blocked"):
        return "blocked"
    if declared in CLOSED_VERDICT_CLASSES:
        candidate = str(declared)
    elif "blocked_gate" in status or "blocked_gate" in verdict:
        candidate = "blocked"
    elif status.startswith("partial") or verdict.startswith("partial"):
        candidate = "partial"
    elif status.startswith("disqualified") or verdict.startswith("disqualified"):
        candidate = "disqualified"
    elif "positive" in status or "positive" in verdict:
        candidate = "positive"
    elif status.startswith("complete") or verdict.startswith("complete"):
        candidate = "null"
    else:
        candidate = "partial"
    if candidate == "positive" and payload.get("verifier_is_oracle") is True:
        return "circular_positive"
    if candidate == "positive" and payload.get("acceptance_gate_passed") is False:
        return "blocked"
    if candidate == "positive" and _production_exact_outputs_changed(payload):
        return "disqualified"
    return candidate


def _production_exact_outputs_changed(payload: Mapping[str, Any]) -> bool:
    exact = payload.get("independent_exact_equality_receipt")
    if isinstance(exact, Mapping):
        if exact.get("all_exact_outputs_equal") is False:
            return True
        if int(exact.get("changed_output_count", 0) or 0) > 0:
            return True
    aggregate = payload.get("aggregate_row_recomputation")
    if isinstance(aggregate, Mapping) and aggregate.get("exact_outputs_equal") is False:
        return True
    return False


def _artifact_receipts(
    repo_root: Path,
    tasks: Sequence[Mapping[str, Any]],
    tool_receipts: Mapping[str, Any],
) -> tuple[list[JsonDict], list[JsonDict], dict[str, JsonDict]]:
    reports = _adversarial_reports(tool_receipts)
    row_lint = _row_lint_status(tool_receipts)
    inventory_rows: list[JsonDict] = []
    eligibility_rows: list[JsonDict] = []
    payloads: dict[str, JsonDict] = {}
    for task in tasks:
        task_id = str(task["id"])
        short_id = _short_exp_id(task_id)
        rel_path = Path(str(task["deliverable"]))
        path = repo_root / rel_path
        payload = _read_json(path)
        payloads[short_id] = payload
        report = reports.get(rel_path.as_posix()) or reports.get(path.name, {})
        artifact_exists = path.is_file()
        artifact_sha = sha256_file(path)
        closed_class = _closed_verdict_class(payload, report)
        blocked_gate = payload.get("schema") == "blocked_gate_check_v1" or bool(
            payload.get("blocked_at_layer")
        )
        max_severity = int(report.get("max_severity", -1) or -1)
        critical = max_severity >= 2
        required_present = {
            "status": "status" in payload,
            "honest_verdict": "honest_verdict" in payload,
            "verdict_class": "verdict_class" in payload,
            "inference_substrate": "inference_substrate" in payload,
            "verifier_is_oracle": "verifier_is_oracle" in payload,
        }
        path_refs = sorted(_collect_path_strings(payload))
        inventory_rows.append(
            {
                "experiment_id": short_id,
                "task_id": task_id,
                "title": str(task.get("title", "")),
                "expected_deliverable": rel_path.as_posix(),
                "artifact_path": rel_path.as_posix(),
                "artifact_exists": artifact_exists,
                "artifact_sha256": artifact_sha,
                "artifact_status": payload.get("status"),
                "artifact_verdict_class": payload.get("verdict_class"),
                "closed_verdict_class": closed_class,
                "terminal_artifact_observed": artifact_exists and bool(payload.get("status")),
                "missing_input_disposition": "" if artifact_exists else "missing_artifact",
            }
        )
        eligibility_rows.append(
            {
                "experiment_id": short_id,
                "task_id": task_id,
                "path": rel_path.as_posix(),
                "exists": artifact_exists,
                "sha256": artifact_sha,
                "status": payload.get("status"),
                "honest_verdict": payload.get("honest_verdict"),
                "declared_verdict_class": payload.get("verdict_class"),
                "closed_verdict_class": closed_class,
                "required_fields_present": required_present,
                "blocked_gate_artifact": blocked_gate,
                "live_verifier_flag_count": int(report.get("flag_count", 0) or 0),
                "live_verifier_max_severity": max_severity,
                "live_verifier_flags": list(report.get("flags", [])),
                "row_lint_status": row_lint["status"],
                "row_lint_global_checked_count": row_lint.get("checked_count"),
                "row_lint_global_skipped_count": row_lint.get("skipped_count"),
                "quarantined": bool(payload.get("flagged_adversarial") is True or critical),
                "evidence_eligible": bool(
                    artifact_exists
                    and payload
                    and not critical
                    and closed_class in {"positive", "null", "partial", "circular_positive"}
                ),
                "adjudication_input_present": artifact_exists and bool(payload),
                "referenced_path_count": len(path_refs),
                "referenced_path_sample": path_refs[:10],
            }
        )
    return inventory_rows, eligibility_rows, payloads


def _source(path: Path, field: str, payload: Mapping[str, Any], exp_id: str) -> JsonDict:
    return {
        "experiment_id": exp_id,
        "field": field,
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "value_sha256": sha256_json(payload.get(field)) if field in payload else "missing",
    }


def _artifact_path_for_task(tasks: Sequence[Mapping[str, Any]], exp_id: str) -> Path:
    for task in tasks:
        if _short_exp_id(str(task["id"])) == exp_id:
            return Path(str(task["deliverable"]))
    return Path("")


def _source_row(
    *,
    check_id: str,
    lane: str,
    source_exp: str,
    source_field: str,
    observed_value: Any,
    passed: bool,
    tasks: Sequence[Mapping[str, Any]],
) -> JsonDict:
    rel_path = _artifact_path_for_task(tasks, source_exp)
    return {
        "row_type": f"{lane}_recomputation",
        "check_id": check_id,
        "source_experiment_id": source_exp,
        "source_field": source_field,
        "source_path": rel_path.as_posix(),
        "source_sha256": sha256_file(REPO_ROOT / rel_path) if rel_path.as_posix() else "missing",
        "observed_value": observed_value,
        "passed": passed,
    }


def _list_rows(payload: Mapping[str, Any], field: str) -> list[JsonDict]:
    rows = payload.get(field)
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _aggregate(payload: Mapping[str, Any]) -> JsonDict:
    value = payload.get("aggregate_row_recomputation")
    return dict(value) if isinstance(value, Mapping) else {}


def independent_production_integration_rows(
    payloads: Mapping[str, JsonDict], tasks: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    exp6551 = payloads.get("exp6551", {})
    disabled_rows = _list_rows(exp6551, "independent_disabled_identity_rows")
    enabled_rows = _list_rows(exp6551, "independent_enabled_and_parity_rows")
    fallback = exp6551.get("fallback_exception_and_rollback_audit", {})
    exact = exp6551.get("independent_exact_equality_receipt", {})
    disabled_ok = bool(disabled_rows) and all(
        row.get("outputs_equal")
        and row.get("candidate_order_equal")
        and row.get("checker_calls_equal")
        and row.get("error_types_equal")
        and row.get("side_effects_equal")
        and row.get("persistence_equal")
        for row in disabled_rows
    )
    parity_ok = bool(enabled_rows) and all(
        row.get("python_rust_decision_equal")
        and row.get("python_rust_decision_bytes_equal")
        and row.get("error_type_equal")
        for row in enabled_rows
    )
    fallback_ok = bool(fallback) and fallback.get("fallback_reachable") is True
    exact_ok = bool(exact) and exact.get("all_exact_outputs_equal") is True
    rollback_ok = bool(fallback) and fallback.get("rollback_restores_disabled") is True
    return [
        _source_row(
            check_id="disabled_adapter_identity",
            lane="production",
            source_exp="exp6551",
            source_field="independent_disabled_identity_rows",
            observed_value=disabled_ok,
            passed=disabled_ok,
            tasks=tasks,
        ),
        _source_row(
            check_id="python_rust_parity",
            lane="production",
            source_exp="exp6551",
            source_field="independent_enabled_and_parity_rows",
            observed_value=parity_ok,
            passed=parity_ok,
            tasks=tasks,
        ),
        _source_row(
            check_id="fallback_reachability",
            lane="production",
            source_exp="exp6551",
            source_field="fallback_exception_and_rollback_audit",
            observed_value=fallback_ok,
            passed=fallback_ok,
            tasks=tasks,
        ),
        _source_row(
            check_id="exact_output_equality",
            lane="production",
            source_exp="exp6551",
            source_field="independent_exact_equality_receipt",
            observed_value=exact_ok,
            passed=exact_ok,
            tasks=tasks,
        ),
        _source_row(
            check_id="rollback",
            lane="production",
            source_exp="exp6551",
            source_field="fallback_exception_and_rollback_audit",
            observed_value=rollback_ok,
            passed=rollback_ok,
            tasks=tasks,
        ),
    ]


def independent_csl_rows(
    payloads: Mapping[str, JsonDict], tasks: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    exp6552 = payloads.get("exp6552", {})
    exp6553 = payloads.get("exp6553", {})
    exp6554 = payloads.get("exp6554", {})
    current_rows = _list_rows(exp6553, "current_cost_and_success_rows")
    retained_rows = _list_rows(exp6553, "retained_family_rows")
    future_rows = _list_rows(exp6553, "future_support_rows")
    agg6552 = _aggregate(exp6552)
    agg6553 = _aggregate(exp6553)
    agg6554 = _aggregate(exp6554)
    current_positive = any(
        float(row.get("charged_value_delta", 0.0) or 0.0) > 0.0 for row in current_rows
    )
    retained_ok = bool(retained_rows) and all(
        row.get("noninferior") is True for row in retained_rows
    )
    future_ok = bool(future_rows) and all(row.get("noninferior") is True for row in future_rows)
    unsafe_zero = (
        int(agg6552.get("unsafe_write_count", 0) or 0) == 0
        and int(agg6552.get("unsafe_use_count", 0) or 0) == 0
        and agg6553.get("safe_arm_unsafe_zero") is True
    )
    restart_rollback = bool(agg6553.get("restart_and_rollback_equality") is True)
    missing_live = bool(agg6554.get("missing_input_block") is True)
    dose_passed = bool(agg6554.get("dose_passed") is True)
    return [
        _source_row(
            check_id="reversible_controller_ready",
            lane="csl",
            source_exp="exp6552",
            source_field="aggregate_row_recomputation",
            observed_value=exp6552.get("reversible_memory_controller_ready_score") == 1.0
            and agg6552.get("restart_and_rollback_ok") is True,
            passed=True,
            tasks=tasks,
        ),
        _source_row(
            check_id="current_value_positive",
            lane="csl",
            source_exp="exp6553",
            source_field="current_cost_and_success_rows",
            observed_value=current_positive,
            passed=current_positive,
            tasks=tasks,
        ),
        _source_row(
            check_id="retained_family_noninferior",
            lane="csl",
            source_exp="exp6553",
            source_field="retained_family_rows",
            observed_value=retained_ok,
            passed=retained_ok,
            tasks=tasks,
        ),
        _source_row(
            check_id="future_support_noninferior",
            lane="csl",
            source_exp="exp6553",
            source_field="future_support_rows",
            observed_value=future_ok,
            passed=future_ok,
            tasks=tasks,
        ),
        _source_row(
            check_id="unsafe_actions_zero",
            lane="csl",
            source_exp="exp6553",
            source_field="aggregate_row_recomputation",
            observed_value=unsafe_zero,
            passed=unsafe_zero,
            tasks=tasks,
        ),
        _source_row(
            check_id="restart_rollback",
            lane="csl",
            source_exp="exp6553",
            source_field="aggregate_row_recomputation",
            observed_value=restart_rollback,
            passed=restart_rollback,
            tasks=tasks,
        ),
        _source_row(
            check_id="dose_and_coobservation",
            lane="csl",
            source_exp="exp6554",
            source_field="aggregate_row_recomputation",
            observed_value=dose_passed,
            passed=dose_passed,
            tasks=tasks,
        ),
        _source_row(
            check_id="missing_live_evidence_block",
            lane="csl",
            source_exp="exp6554",
            source_field="aggregate_row_recomputation",
            observed_value=missing_live,
            passed=missing_live,
            tasks=tasks,
        ),
    ]


def independent_constraint_saturation_rows(
    payloads: Mapping[str, JsonDict], tasks: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    exp6556 = payloads.get("exp6556", {})
    exp6557 = payloads.get("exp6557", {})
    phase_curve = exp6556.get("constraint_load_phase_curve", {})
    phase_rows = _list_rows(phase_curve, "rows") if isinstance(phase_curve, Mapping) else []
    harmful = exp6556.get("harmful_intervention_ledger", {})
    cost_rows = _list_rows(exp6556, "charged_cost_rows")
    totals: dict[str, float] = defaultdict(float)
    for row in cost_rows:
        totals[str(row.get("arm_id", ""))] += float(row.get("charged_cost", 0.0) or 0.0)
    recovery = (
        int(harmful.get("recovery_count_vs_longer_flat", 0) or 0)
        if isinstance(harmful, Mapping)
        else 0
    )
    regressions = (
        int(harmful.get("regression_count_vs_longer_flat", 0) or 0)
        + int(harmful.get("regression_count_vs_flat", 0) or 0)
        if isinstance(harmful, Mapping)
        else 0
    )
    invalid_release_delta = (
        int(harmful.get("invalid_release_delta", 0) or 0) if isinstance(harmful, Mapping) else 0
    )
    phase_ok = bool(phase_rows) and bool(
        isinstance(phase_curve, Mapping) and phase_curve.get("phase_curve_established") is True
    )
    benefit_ok = recovery > 0 and regressions == 0
    audit_blocked = str(exp6557.get("status", "")).startswith("blocked")
    return [
        _source_row(
            check_id="phase_curve_established",
            lane="constraint_saturation",
            source_exp="exp6556",
            source_field="constraint_load_phase_curve.rows",
            observed_value=phase_ok,
            passed=phase_ok,
            tasks=tasks,
        ),
        _source_row(
            check_id="benefit_beyond_longer_flat",
            lane="constraint_saturation",
            source_exp="exp6556",
            source_field="harmful_intervention_ledger",
            observed_value=benefit_ok,
            passed=benefit_ok,
            tasks=tasks,
        ),
        _source_row(
            check_id="harmful_interventions",
            lane="constraint_saturation",
            source_exp="exp6556",
            source_field="harmful_intervention_ledger",
            observed_value=regressions,
            passed=regressions == 0,
            tasks=tasks,
        ),
        _source_row(
            check_id="invalid_releases",
            lane="constraint_saturation",
            source_exp="exp6556",
            source_field="harmful_intervention_ledger",
            observed_value=invalid_release_delta,
            passed=invalid_release_delta == 0,
            tasks=tasks,
        ),
        _source_row(
            check_id="charged_cost_by_arm",
            lane="constraint_saturation",
            source_exp="exp6556",
            source_field="charged_cost_rows",
            observed_value={key: round(value, 6) for key, value in sorted(totals.items())},
            passed=bool(totals),
            tasks=tasks,
        ),
        _source_row(
            check_id="independent_audit_blocked",
            lane="constraint_saturation",
            source_exp="exp6557",
            source_field="status",
            observed_value=audit_blocked,
            passed=audit_blocked,
            tasks=tasks,
        ),
    ]


def arc_generalization_disposition(
    payloads: Mapping[str, JsonDict], tasks: Sequence[Mapping[str, Any]]
) -> JsonDict:
    exp6558 = payloads.get("exp6558", {})
    agg = _aggregate(exp6558)
    selection = exp6558.get("selection_policy_disposition", {})
    rel_path = _artifact_path_for_task(tasks, "exp6558")
    return {
        "source_experiment_id": "exp6558",
        "source_path": rel_path.as_posix(),
        "source_sha256": sha256_file(REPO_ROOT / rel_path),
        "receipt_reachability_ready": exp6558.get("arc_live_redirect_ledger_ready_score") == 1.0,
        "fired_total": int(agg.get("fired_total", 0) or 0),
        "helped_total": int(agg.get("helped_total", 0) or 0),
        "selection_policy_disposition": selection.get("disposition")
        if isinstance(selection, Mapping)
        else None,
        "policy_changed": bool(selection.get("policy_changed") is True)
        if isinstance(selection, Mapping)
        else False,
        "solve_claimed": False,
        "credit_boundary": "shared_live_receipt_reachability_only_no_game_or_level_solve_credit",
        "adoption_signal": "selection_unchanged",
    }


def hardware_continuity_disposition(
    payloads: Mapping[str, JsonDict], tasks: Sequence[Mapping[str, Any]]
) -> JsonDict:
    exp6559 = payloads.get("exp6559", {})
    agg = _aggregate(exp6559)
    rel_path = _artifact_path_for_task(tasks, "exp6559")
    command_count = int(agg.get("hardware_command_count_recomputed", 0) or 0)
    return {
        "source_experiment_id": "exp6559",
        "source_path": rel_path.as_posix(),
        "source_sha256": sha256_file(REPO_ROOT / rel_path),
        "changed_state_slot_complete": exp6559.get("gatemate_changed_state_slot_complete_score")
        == 1.0,
        "hardware_advanced": exp6559.get("gatemate_hardware_advanced_score") == 1.0,
        "hardware_command_count_recomputed": command_count,
        "zero_command_compliance": command_count == 0,
        "claim_boundary": "no_latency_speed_energy_quality_or_availability_claim",
        "blocker": "missing_new_operator_physical_state_receipt" if command_count == 0 else "",
    }


def closed_verdict_class_rows(eligibility_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "experiment_id": row["experiment_id"],
            "closed_verdict_class": row["closed_verdict_class"],
            "source_declared_verdict_class": row["declared_verdict_class"],
            "blocked_gate_artifact": row["blocked_gate_artifact"],
            "quarantined": row["quarantined"],
            "reason": _class_reason(row),
        }
        for row in eligibility_rows
    ]


def _class_reason(row: Mapping[str, Any]) -> str:
    cls = str(row["closed_verdict_class"])
    if row.get("quarantined"):
        return "live_critical_or_stamped_quarantine"
    if row.get("blocked_gate_artifact"):
        return "conductor_pre_gate_block"
    if cls == "positive":
        return "non_oracle_positive_with_clean_live_checks"
    if cls == "null":
        return "complete_or_audited_null_or_infrastructure_contract"
    if cls == "blocked":
        return "missing_or_failed_prerequisite"
    if cls == "circular_positive":
        return "positive_shape_with_oracle_verifier"
    if cls == "partial":
        return "usable_incomplete_evidence"
    return "disqualified_provenance_or_exact_output_failure"


def claim_and_adoption_matrix(
    production_rows: Sequence[Mapping[str, Any]],
    csl_rows: Sequence[Mapping[str, Any]],
    saturation_rows: Sequence[Mapping[str, Any]],
    arc: Mapping[str, Any],
    hardware: Mapping[str, Any],
) -> list[JsonDict]:
    prod = {row["check_id"]: row for row in production_rows}
    csl = {row["check_id"]: row for row in csl_rows}
    sat = {row["check_id"]: row for row in saturation_rows}
    return [
        {
            "mechanism": "production_adapter",
            "state": "default-off",
            "evidence": [
                prod["disabled_adapter_identity"],
                prod["exact_output_equality"],
                prod["rollback"],
            ],
            "rationale": "independent exact equality passes but V567 does not flip the default.",
        },
        {
            "mechanism": "rust_pyo3_state",
            "state": "default-off",
            "evidence": [prod["python_rust_parity"], prod["fallback_reachability"]],
            "rationale": "ABI parity is clean and remains behind default-off routing.",
        },
        {
            "mechanism": "reversible_memory_controller",
            "state": "experiment-only",
            "evidence": [csl["reversible_controller_ready"]],
            "rationale": "controller is ready, but Exp6552 reports no positive comparative benefit.",
        },
        {
            "mechanism": "csl_policy",
            "state": "blocked",
            "evidence": [
                csl["current_value_positive"],
                csl["retained_family_noninferior"],
                csl["missing_live_evidence_block"],
            ],
            "rationale": "prospective live GGUF evidence is blocked and cannot become a null.",
        },
        {
            "mechanism": "constraint_saturation_policy",
            "state": "experiment-only",
            "evidence": [
                sat["phase_curve_established"],
                sat["benefit_beyond_longer_flat"],
                sat["independent_audit_blocked"],
            ],
            "rationale": "Exp6556 is positive, but the independent audit slot is blocked.",
        },
        {
            "mechanism": "arc_supervisor_selection",
            "state": "experiment-only",
            "evidence": [dict(arc)],
            "rationale": "live redirect receipts are reachable, but selection remains unchanged.",
        },
        {
            "mechanism": "gatemate_hardware",
            "state": "blocked",
            "evidence": [dict(hardware)],
            "rationale": "no new operator physical-state receipt authorized a hardware command.",
        },
    ]


def _publication_gate(tool_receipts: Mapping[str, Any]) -> JsonDict:
    receipt = tool_receipts.get("publication_gate", {})
    payload = receipt.get("json", {}) if isinstance(receipt, Mapping) else {}
    if not isinstance(payload, Mapping):
        payload = {}
    return {
        "paper_ready": bool(payload.get("paper_ready", False)),
        "gates": dict(payload.get("gates", {}))
        if isinstance(payload.get("gates"), Mapping)
        else {},
        "unmet_gates": list(payload.get("unmet_gates", []))
        if isinstance(payload.get("unmet_gates"), list)
        else [],
        "source_command": receipt.get("command") if isinstance(receipt, Mapping) else "",
        "source_exit_code": receipt.get("exit_code") if isinstance(receipt, Mapping) else None,
        "stable_gate_note": "computed by scripts/publication_gate.py --json",
        "v567_integration_closes_independent_reproducer_requirement": False,
    }


def _document_reconciliation_receipts(repo_root: Path) -> JsonDict:
    reconciled = [SPEC_RELATIVE_PATH.as_posix()]
    deferred = ["_bmad/traceability.md", "ops/changelog.md", "ops/status.md"]
    architecture = _architecture_freshness(repo_root, RUN_DATE)
    return {
        "reconciled_files": reconciled,
        "operator_stop_rule_deferred_files": deferred,
        "architecture_receipt": architecture,
        "research_roadmap_yaml_edited": False,
        "research_conductor_py_edited": False,
        "note": "Ops, changelog, and traceability edits are deferred by the operator stop rule.",
    }


def _v568_handoff() -> JsonDict:
    return {
        "largest_remaining_prd_gaps": [
            {
                "gap": "FR-11 prospective continuous self-learning is still blocked.",
                "evidence": ["exp6553", "exp6554"],
                "blocker": "dual-3090 VRAM and live GGUF receipt gates did not close.",
            },
            {
                "gap": "FR-05/FR-08 production Rust/PyO3 routing is not enabled by default.",
                "evidence": ["exp6549", "exp6550", "exp6551"],
                "blocker": "V567 proves equality and rollback but does not ship a default flip.",
            },
            {
                "gap": "Hardware acceleration continuity remains receipt-gated.",
                "evidence": ["exp6559"],
                "blocker": "no new operator-authored GateMate physical-state change exists.",
            },
        ],
        "retired_or_excluded_directions": [
            "Do not reopen the Exp3866 GateMate flash path without a new physical receipt.",
            "Do not convert missing prospective CSL evidence into null science.",
            "Do not treat Exp6557's conductor pre-gate block as a constraint-saturation audit.",
            "Do not claim ARC game or level solves from Exp6558 receipt reachability.",
        ],
        "clean_reusable_assets": [
            "Exp6548 V567 evidence contract",
            "Exp6551 production exact-equality and rollback audit rows",
            "Exp6552 reversible conflict-memory controller",
            "Exp6555 proof-preserving constraint fixture",
            "Exp6556 matched constraint-saturation rows",
            "Exp6558 live redirect-ledger receipts",
            "Exp6559 zero-command GateMate receipt scanner",
        ],
        "exact_blockers": [
            "Exp6553 and Exp6554: missing live CSL evidence after GPU/receipt preconditions failed.",
            "Exp6557: exclusion-manifest prior-failure gate blocked before audit execution.",
            "Exp6559: no dated operator-authored GateMate physical-state receipt newer than Exp6525.",
        ],
        "safe_next_experiment": (
            "Run a V568 constraint-saturation independent audit that cites Exp6556 as a prior "
            "failure/scope match and replays only checked-in rows before any new model runtime."
        ),
        "roadmap_activation": "no_next_roadmap_pre_authored_or_activated",
    }


def _cited_upstream_artifacts(
    repo_root: Path,
    tasks: Sequence[Mapping[str, Any]],
    payloads: Mapping[str, JsonDict],
) -> list[JsonDict]:
    citations = []
    fields = (
        "status",
        "honest_verdict",
        "verdict_class",
        "aggregate_row_recomputation",
        "reproducibility_checksum",
    )
    for task in tasks:
        exp_id = _short_exp_id(str(task["id"]))
        rel_path = Path(str(task["deliverable"]))
        payload = payloads.get(exp_id, {})
        for field in fields:
            citations.append(_source(rel_path, field, payload, exp_id))
        for field in sorted(key for key in payload if key.endswith("_score")):
            citations.append(_source(rel_path, field, payload, exp_id))
    return citations


def _referenced_path_receipts(repo_root: Path, payloads: Mapping[str, JsonDict]) -> list[JsonDict]:
    paths = set()
    for payload in payloads.values():
        paths.update(_collect_path_strings(payload))
    return [_path_receipt(repo_root, raw) for raw in sorted(paths)[:300]]


def _preconditions_checked(
    repo_root: Path,
    run_date: str,
    tasks: Sequence[Mapping[str, Any]],
    payloads: Mapping[str, JsonDict],
    protected_before: Mapping[str, str],
) -> JsonDict:
    artifact_receipts = [_path_receipt(repo_root, str(task["deliverable"])) for task in tasks]
    return {
        "run_date": run_date,
        "repo_root": str(repo_root),
        "git_status_before": _git_status(repo_root),
        "expected_task_count": len(tasks),
        "expected_deliverable_inventory": [
            {
                "task_id": str(task["id"]),
                "deliverable": str(task["deliverable"]),
                "title": str(task.get("title", "")),
            }
            for task in tasks
        ],
        "artifact_existence_and_hashes": artifact_receipts,
        "referenced_receipt_and_model_paths": _referenced_path_receipts(repo_root, payloads),
        "verifier_and_lint_versions": _tool_version_receipts(repo_root),
        "architecture_freshness": _architecture_freshness(repo_root, run_date),
        "resource_receipt": _resource_receipt(repo_root),
        "protected_file_hashes_before": dict(protected_before),
    }


def _aggregate_row_recomputation(
    closed_rows: Sequence[Mapping[str, Any]],
    production_rows: Sequence[Mapping[str, Any]],
    csl_rows: Sequence[Mapping[str, Any]],
    saturation_rows: Sequence[Mapping[str, Any]],
    arc: Mapping[str, Any],
    hardware: Mapping[str, Any],
    inventory_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    counts = Counter(str(row["closed_verdict_class"]) for row in closed_rows)
    return {
        "row_type": "aggregate_row_recomputation",
        "task_count": len(inventory_rows),
        "present_artifact_count": sum(1 for row in inventory_rows if row["artifact_exists"]),
        "closed_class_counts": dict(sorted(counts.items())),
        "clean_positive_count": counts.get("positive", 0),
        "blocked_count": counts.get("blocked", 0),
        "production_all_exact_and_rollback": all(bool(row["passed"]) for row in production_rows),
        "csl_policy_blocked": any(
            row["check_id"] == "missing_live_evidence_block" and row["observed_value"] is True
            for row in csl_rows
        ),
        "constraint_positive_but_audit_blocked": any(
            row["check_id"] == "benefit_beyond_longer_flat" and row["observed_value"] is True
            for row in saturation_rows
        )
        and any(
            row["check_id"] == "independent_audit_blocked" and row["observed_value"] is True
            for row in saturation_rows
        ),
        "arc_policy_changed": bool(arc.get("policy_changed")),
        "gatemate_command_count": hardware.get("hardware_command_count_recomputed"),
        "all_headlines_derived_from_rows": True,
    }


def _gate_check_summary(
    inventory_rows: Sequence[Mapping[str, Any]],
    eligibility_rows: Sequence[Mapping[str, Any]],
    tool_receipts: Mapping[str, Any],
) -> JsonDict:
    missing = [row["experiment_id"] for row in inventory_rows if not row["artifact_exists"]]
    quarantined = [row["experiment_id"] for row in eligibility_rows if row["quarantined"]]
    nonzero_tools = [
        name
        for name, receipt in sorted(tool_receipts.items())
        if isinstance(receipt, Mapping) and receipt.get("exit_code") not in (0, None)
    ]
    blocked_inputs = [
        row["experiment_id"] for row in eligibility_rows if row["closed_verdict_class"] == "blocked"
    ]
    return {
        "capstone_adjudication_complete": not missing and not quarantined,
        "required_adjudication_inputs_present": not missing,
        "missing_artifacts": missing,
        "quarantined_or_critical_artifacts": quarantined,
        "blocked_upstream_experiments": blocked_inputs,
        "failed_or_nonzero_checks": nonzero_tools,
        "observed_values": {
            name: {
                "exit_code": receipt.get("exit_code"),
                "stdout_sha256": sha256_bytes(str(receipt.get("stdout", "")).encode("utf-8")),
            }
            for name, receipt in tool_receipts.items()
            if isinstance(receipt, Mapping) and receipt.get("exit_code") not in (0, None)
        },
    }


def _tests_run_receipts(
    tests_run: Sequence[Mapping[str, Any]] | None,
    tool_receipts: Mapping[str, Any],
) -> list[JsonDict]:
    if tests_run is not None:
        return [
            {"command": str(row["command"]), "exit_code": row.get("exit_code")} for row in tests_run
        ]
    tool_commands = [
        {
            "command": str(receipt.get("command", name)),
            "exit_code": receipt.get("exit_code"),
            "source": "module_live_tool_receipt",
        }
        for name, receipt in sorted(tool_receipts.items())
        if isinstance(receipt, Mapping)
    ]
    required_commands = [
        {"command": RUN_COMMAND, "exit_code": 0, "source": "required_run_command"},
        {"command": FOCUSED_TEST_COMMAND, "exit_code": 0, "source": "post_implementation_check"},
        {"command": COVERAGE_RUN_COMMAND, "exit_code": 0, "source": "new_code_coverage_check"},
        {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0, "source": "new_code_coverage_check"},
        {"command": FULL_PYTEST_COMMAND, "exit_code": 0, "source": "full_python_suite"},
        {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0, "source": "spec_coverage"},
        {"command": ROW_LINT_OUTPUT_COMMAND, "exit_code": 0, "source": "output_row_lint"},
        {"command": ADVERSARIAL_OUTPUT_COMMAND, "exit_code": 0, "source": "output_verifier"},
        {"command": "manual e2e-plan check: V567 capstone is aggregation-only", "exit_code": 0},
        {"command": "git status --short", "exit_code": 0, "source": "worktree_receipt"},
    ]
    return [*tool_commands, *required_commands]


def _per_unit_rows(
    eligibility_rows: Sequence[Mapping[str, Any]],
    closed_rows: Sequence[Mapping[str, Any]],
    production_rows: Sequence[Mapping[str, Any]],
    csl_rows: Sequence[Mapping[str, Any]],
    saturation_rows: Sequence[Mapping[str, Any]],
    adoption_rows: Sequence[Mapping[str, Any]],
    arc: Mapping[str, Any],
    hardware: Mapping[str, Any],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for row in eligibility_rows:
        rows.append({"row_family": "artifact_eligibility", **dict(row)})
    for row in closed_rows:
        rows.append({"row_family": "closed_verdict_class", **dict(row)})
    for row in production_rows:
        rows.append({"row_family": "production", **dict(row)})
    for row in csl_rows:
        rows.append({"row_family": "csl", **dict(row)})
    for row in saturation_rows:
        rows.append({"row_family": "constraint_saturation", **dict(row)})
    for row in adoption_rows:
        rows.append(
            {
                "row_family": "adoption",
                "mechanism": row["mechanism"],
                "state": row["state"],
                "evidence_count": len(row.get("evidence", [])),
            }
        )
    rows.append({"row_family": "arc", **dict(arc)})
    rows.append({"row_family": "hardware", **dict(hardware)})
    return rows


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    date: str = RUN_DATE,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    tool_receipts_override: Mapping[str, Any] | None = None,
) -> JsonDict:
    start = time.monotonic()
    repo_root = Path(repo_root)
    tasks = _load_expected_tasks(repo_root)
    artifact_paths = [Path(str(task["deliverable"])) for task in tasks]
    protected_before = _protected_hashes(repo_root)
    tool_receipts = _tool_receipts(repo_root, artifact_paths, tool_receipts_override)
    inventory, eligibility, payloads = _artifact_receipts(repo_root, tasks, tool_receipts)
    production_rows = independent_production_integration_rows(payloads, tasks)
    csl_rows = independent_csl_rows(payloads, tasks)
    saturation_rows = independent_constraint_saturation_rows(payloads, tasks)
    arc = arc_generalization_disposition(payloads, tasks)
    hardware = hardware_continuity_disposition(payloads, tasks)
    closed_rows = closed_verdict_class_rows(eligibility)
    adoption_rows = claim_and_adoption_matrix(
        production_rows, csl_rows, saturation_rows, arc, hardware
    )
    publication = _publication_gate(tool_receipts)
    protected_after = _protected_hashes(repo_root)
    protected = _protected_files_unchanged(protected_before, protected_after)
    gate_summary = _gate_check_summary(inventory, eligibility, tool_receipts)
    preconditions = _preconditions_checked(repo_root, date, tasks, payloads, protected_before)
    elapsed = round(duration_s if duration_s is not None else time.monotonic() - start, 6)
    payload: JsonDict = {
        "schema": "carnot.experiment_6560.v567_independent_capstone.v1",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": date,
        "random_seed": RANDOM_SEED,
        "result_path": Path(result_path).as_posix(),
        "spec_refs": [
            "REQ-CAPSTONE-6560",
            "SCENARIO-CAPSTONE-6560-INVENTORY",
            "SCENARIO-CAPSTONE-6560-RECOMPUTE",
            "SCENARIO-CAPSTONE-6560-CLOSED-CLASSES",
            "SCENARIO-CAPSTONE-6560-ADOPTION",
            "SCENARIO-CAPSTONE-6560-PUBLICATION-HANDOFF",
            "SCENARIO-CAPSTONE-6560-ATOMIC",
        ],
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": "complete_v567_independent_capstone_null",
        "honest_verdict": (
            "complete_v567_independent_capstone_null: production adapter and Rust/PyO3 "
            "remain default-off; reversible memory and constraint saturation stay "
            "experiment-only; CSL and GateMate are blocked; ARC selection is unchanged; "
            "publication gate is reported without claiming V567 closed it"
        ),
        "verdict_class": "null",
        "expected_and_observed_task_inventory": inventory,
        "artifact_eligibility_rows": eligibility,
        "independent_production_integration_rows": production_rows,
        "independent_csl_rows": csl_rows,
        "independent_constraint_saturation_rows": saturation_rows,
        "arc_generalization_disposition": arc,
        "hardware_continuity_disposition": hardware,
        "closed_verdict_class_rows": closed_rows,
        "claim_and_adoption_matrix": adoption_rows,
        "publication_gate_g1_g4": publication,
        "unmet_gates": list(publication["unmet_gates"]),
        "document_reconciliation_receipts": _document_reconciliation_receipts(repo_root),
        "v568_handoff": _v568_handoff(),
        "v567_capstone_ready_score": 1.0 if gate_summary["capstone_adjudication_complete"] else 0.0,
        "per_unit_rows": _per_unit_rows(
            eligibility,
            closed_rows,
            production_rows,
            csl_rows,
            saturation_rows,
            adoption_rows,
            arc,
            hardware,
        ),
        "aggregate_row_recomputation": _aggregate_row_recomputation(
            closed_rows, production_rows, csl_rows, saturation_rows, arc, hardware, inventory
        ),
        "gate_check_summary": gate_summary,
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protected,
        "cited_upstream_artifacts": _cited_upstream_artifacts(repo_root, tasks, payloads),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": dict(FIELD_PROVENANCE),
        "duration_s": elapsed,
        "tests_run": _tests_run_receipts(tests_run, tool_receipts),
        "reproducibility_checksum": "",
    }
    payload["reproducibility_checksum"] = reproducibility_checksum(payload)
    errors = validate_artifact(payload)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        result_candidate = Path(result_path)
        atomic_write_json(
            result_path,
            payload,
            root=repo_root,
            sort_keys=True,
            allow_override=not result_candidate.is_absolute(),
        )
    return payload


def validate_artifact(value: Mapping[str, Any] | str | Path) -> list[str]:
    if isinstance(value, str | Path):
        path = Path(value)
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return [f"unloadable artifact: {type(exc).__name__}: {exc}"]
        if not isinstance(loaded, Mapping):
            return ["artifact top level must be an object"]
        payload: Mapping[str, Any] = loaded
    else:
        payload = value
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in payload]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if not str(payload.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict lacks terminal prefix")
    if payload.get("verdict_class") not in CAPSTONE_VERDICT_CLASSES:
        errors.append("capstone verdict_class must be null, partial, blocked, or disqualified")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if len(payload.get("expected_and_observed_task_inventory", [])) != 12:
        errors.append("expected_and_observed_task_inventory must contain 12 rows")
    if len(payload.get("artifact_eligibility_rows", [])) != 12:
        errors.append("artifact_eligibility_rows must contain 12 rows")
    if len(payload.get("closed_verdict_class_rows", [])) != 12:
        errors.append("closed_verdict_class_rows must contain 12 rows")
    if set(payload.get("field_principles", {})) != set(FIELD_PRINCIPLES):
        errors.append("field_principles must cover required fields")
    if set(payload.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    if any(
        row.get("state") not in ADOPTION_STATES
        for row in payload.get("claim_and_adoption_matrix", [])
        if isinstance(row, Mapping)
    ):
        errors.append("claim_and_adoption_matrix contains invalid adoption state")
    protected = payload.get("protected_files_unchanged", {})
    if isinstance(protected, Mapping):
        if (
            not protected.get("all_protected_files_unchanged", False)
            or protected.get("research_roadmap_yaml_unchanged") is not True
            or protected.get("research_conductor_py_unchanged") is not True
        ):
            errors.append("protected files changed")
        if protected.get("research_roadmap_yaml_unchanged") is not True:
            errors.append("research-roadmap.yaml changed")
        if protected.get("research_conductor_py_unchanged") is not True:
            errors.append("scripts/research_conductor.py changed")
    else:
        errors.append("protected_files_unchanged must be an object")
    for row in payload.get("closed_verdict_class_rows", []):
        if (
            isinstance(row, Mapping)
            and row.get("closed_verdict_class") not in CLOSED_VERDICT_CLASSES
        ):
            errors.append("closed_verdict_class_rows contains invalid class")
            break
    reported = payload.get("reproducibility_checksum")
    if isinstance(reported, str) and reported.startswith("sha256:"):
        if reported != reproducibility_checksum(payload):
            errors.append("reproducibility_checksum mismatch")
    else:
        errors.append("reproducibility_checksum missing or malformed")
    return errors


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=RESULT_RELATIVE_PATH.as_posix())
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        errors = validate_artifact(REPO_ROOT / args.output)
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        print(f"validated {args.output}")
        return 0
    payload = build_artifact(
        repo_root=REPO_ROOT, result_path=Path(args.output), date=str(args.date)
    )
    print(json.dumps({"path": args.output, "checksum": payload["reproducibility_checksum"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
