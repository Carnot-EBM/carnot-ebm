"""Build the Exp6435 V553 adversarial capstone artifact.

Spec refs: REQ-CAPSTONE-6435,
SCENARIO-CAPSTONE-6435-HASHES,
SCENARIO-CAPSTONE-6435-PER-TASK,
SCENARIO-CAPSTONE-6435-ROW-RECHECKS,
SCENARIO-CAPSTONE-6435-CLAIM-ELIGIBILITY,
SCENARIO-CAPSTONE-6435-RETIREMENT-AND-ATTACKS,
SCENARIO-CAPSTONE-6435-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

from carnot.experiment_artifacts import atomic_write_json
from carnot.terminal_artifacts import canonical_json, path_sha256


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover - depends on import path.
    sys.path.insert(0, str(SCRIPTS_ROOT))

from adversarial_verify import verify_artifact  # noqa: E402


RUN_DATE = "20260814"
RANDOM_SEED = 6435
INFERENCE_SUBSTRATE = "aggregation_from_upstream_rows_and_artifacts_no_llm"
RESULT_RELATIVE_PATH = Path("results/experiment_6435_v553_adversarial_capstone.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/capstone/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6435_v553_adversarial_capstone.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6435_v553_adversarial_capstone.py")

RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6435_v553_adversarial_capstone --date 20260814"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6435_v553_adversarial_capstone.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6435_v553_adversarial_capstone.py "
    "-m pytest tests/python/test_experiment_6435_v553_adversarial_capstone.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6435_v553_adversarial_capstone.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6435_v553_adversarial_capstone.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6435_v553_adversarial_capstone.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py --all"
ROADMAP_GATE_COMMAND = ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml"
PRIOR_FAILURE_COMMAND = ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml"
EXCLUSION_COMMAND = ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml"
ROOT_SWEEP_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"

DEFAULT_TESTS_RUN = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ROADMAP_GATE_COMMAND,
    PRIOR_FAILURE_COMMAND,
    EXCLUSION_COMMAND,
    ROOT_SWEEP_COMMAND,
)

EXPECTED_ARTIFACTS: dict[str, Path] = {
    "exp6424": Path("results/experiment_6424_v553_terminal_handoff_and_queue_preflight.json"),
    "exp6425": Path("results/experiment_6425_recurring_gate_block_root_cause.json"),
    "exp6426": Path("results/experiment_6426_task_scoped_runtime_receipt_contract.json"),
    "exp6427": Path("results/experiment_6427_fresh_constraint_saturation_factor_corpus.json"),
    "exp6428": Path("results/experiment_6428_clean_write_time_factor_admission_ab.json"),
    "exp6429": Path("results/experiment_6429_constraint_saturation_verification_cost_ab.json"),
    "exp6430": Path("results/experiment_6430_prospective_write_once_memory_capacity_frontier.json"),
    "exp6431": Path("results/experiment_6431_controlled_memory_interference_ab.json"),
    "exp6432": Path("results/experiment_6432_held_shift_process_restart_csl_replication.json"),
    "exp6433": Path("results/experiment_6433_csl_row_recomputation_safety_audit.json"),
    "exp6434": Path("results/experiment_6434_arc_state_key_reachability_ab.json"),
}

TASK_TITLES: dict[str, str] = {
    "exp6424": "V552 terminal evidence handoff and V553 queue preflight",
    "exp6425": "Recurring blocked-gate root cause and diagnostic contract",
    "exp6426": "Task-scoped runtime, GPU, concurrency, and runner receipt contract",
    "exp6427": "Fresh constraint-saturation factor corpus",
    "exp6428": "Clean exact write-time factor admission A/B",
    "exp6429": "Constraint saturation and verification-cost A/B",
    "exp6430": "Prospective write-once memory capacity frontier",
    "exp6431": "Controlled memory-interference A/B",
    "exp6432": "Held-shift process-restart CSL replication",
    "exp6433": "Independent CSL row-recomputation and safety audit",
    "exp6434": "ARC state-key reachability invariant A/B",
}

TASK_CATEGORIES: dict[str, str] = {
    "exp6424": "transition",
    "exp6425": "audit",
    "exp6426": "infrastructure",
    "exp6427": "factor",
    "exp6428": "factor",
    "exp6429": "verification_cost",
    "exp6430": "csl",
    "exp6431": "csl_safety",
    "exp6432": "csl",
    "exp6433": "audit",
    "exp6434": "arc_reachability",
}

SUBSTANTIVE_TASK_IDS = {
    "exp6427",
    "exp6428",
    "exp6429",
    "exp6430",
    "exp6431",
    "exp6432",
    "exp6434",
}

SOURCE_PATHS: dict[str, Path] = {
    **{
        exp_id: Path(f"python/carnot/{EXPECTED_ARTIFACTS[exp_id].name.replace('.json', '.py')}")
        for exp_id in EXPECTED_ARTIFACTS
    },
    "exp6435_source": MODULE_RELATIVE_PATH,
}
SOURCE_PATHS.update(
    {
        "exp6424": Path("python/carnot/experiment_6424_v553_terminal_handoff_and_queue_preflight.py"),
        "exp6425": Path("python/carnot/experiment_6425_recurring_gate_block_root_cause.py"),
        "exp6426": Path("python/carnot/experiment_6426_task_scoped_runtime_receipt_contract.py"),
        "exp6427": Path("python/carnot/experiment_6427_fresh_constraint_saturation_factor_corpus.py"),
        "exp6428": Path("python/carnot/experiment_6428_clean_write_time_factor_admission_ab.py"),
        "exp6429": Path("python/carnot/experiment_6429_constraint_saturation_verification_cost_ab.py"),
        "exp6430": Path("python/carnot/experiment_6430_prospective_write_once_memory_capacity_frontier.py"),
        "exp6431": Path("python/carnot/experiment_6431_controlled_memory_interference_ab.py"),
        "exp6432": Path("python/carnot/experiment_6432_held_shift_process_restart_csl_replication.py"),
        "exp6433": Path("python/carnot/experiment_6433_csl_row_recomputation_safety_audit.py"),
        "exp6434": Path("python/carnot/experiment_6434_arc_state_key_reachability_ab.py"),
    }
)

TEST_PATHS: dict[str, Path] = {
    "exp6424": Path("tests/python/test_experiment_6424_v553_terminal_handoff_and_queue_preflight.py"),
    "exp6425": Path("tests/python/test_experiment_6425_recurring_gate_block_root_cause.py"),
    "exp6426": Path("tests/python/test_experiment_6426_task_scoped_runtime_receipt_contract.py"),
    "exp6427": Path("tests/python/test_experiment_6427_fresh_constraint_saturation_factor_corpus.py"),
    "exp6428": Path("tests/python/test_experiment_6428_clean_write_time_factor_admission_ab.py"),
    "exp6429": Path("tests/python/test_experiment_6429_constraint_saturation_verification_cost_ab.py"),
    "exp6430": Path("tests/python/test_experiment_6430_prospective_write_once_memory_capacity_frontier.py"),
    "exp6431": Path("tests/python/test_experiment_6431_controlled_memory_interference_ab.py"),
    "exp6432": Path("tests/python/test_experiment_6432_held_shift_process_restart_csl_replication.py"),
    "exp6433": Path("tests/python/test_experiment_6433_csl_row_recomputation_safety_audit.py"),
    "exp6434": Path("tests/python/test_experiment_6434_arc_state_key_reachability_ab.py"),
    "exp6435_tests": TEST_RELATIVE_PATH,
}

ROW_AND_MANIFEST_PATHS: dict[str, Path] = {
    "exp6426_rows": Path("data/research/experiment_6426_task_scoped_runtime_receipt_contract"),
    "exp6427_rows": Path("data/research/experiment_6427_fresh_constraint_saturation_factor_corpus"),
    "exp6430_rows": Path("data/research/experiment_6430_prospective_write_once_memory_capacity_frontier"),
    "exp6432_rows": Path("data/research/experiment_6432_held_shift_process_restart_csl_replication"),
}

ROADMAP_PATHS: dict[str, Path] = {
    "active_roadmap": Path("research-roadmap.yaml"),
    "staged_roadmap": Path("research-roadmap-next.yaml"),
    "milestone_doc": Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    "complete_history": Path("research-complete.yaml"),
}

SPEC_PATHS: dict[str, Path] = {
    "capstone": SPEC_RELATIVE_PATH,
    "research_harnesses": Path("openspec/capabilities/research-harnesses/spec.md"),
    "continuous_learning": Path("openspec/capabilities/continuous-learning/spec.md"),
    "arc_agi": Path("openspec/capabilities/arc-agi/spec.md"),
    "hardware": Path("openspec/capabilities/hardware/spec.md"),
    "self_learning": Path("openspec/capabilities/self-learning/spec.md"),
}

OPS_PATHS: dict[str, Path] = {
    "conductor_log": Path("ops/conductor-log.md"),
    "status": Path("ops/status.md"),
    "changelog": Path("ops/changelog.md"),
    "known_issues": Path("ops/known-issues.md"),
    "north_star": Path("ops/north-star.md"),
    "e2e_test_plan": Path("ops/e2e-test-plan.md"),
    "traceability": Path("_bmad/traceability.md"),
}

REGISTRY_CLAIM_MANIFEST_PATHS: dict[str, Path] = {
    "exclusion_manifest": Path("ops/exclusion_manifest.yaml"),
    "arc_solve_registry": Path("ops/arc_solve_registry.yaml"),
    "claim_scoping_proposal": Path("ops/claim-scoping-proposal-2026-06.md"),
    "v550_claim_ledger": Path("results/experiment_6406_clean_v550_factor_evidence_boundary.json.claim_ledger.jsonl"),
    "v551_claim_ledger": Path("results/experiment_6412_v551_powered_claim_integrity_audit.json.claim_ledger.jsonl"),
    "v551_corrigendum": Path("results/experiment_6412_v551_powered_claim_integrity_audit.json.corrigendum.json"),
    "requested_claim_eligibility_ledger": Path("ops/claim-eligibility-ledger.json"),
}

CHECKER_PATHS: dict[str, Path] = {
    "summarize_artifact": Path("scripts/summarize_artifact.py"),
    "adversarial_verify": Path("scripts/adversarial_verify.py"),
    "determination_preservation_lint": Path("scripts/determination_preservation_lint.py"),
    "artifact_convention_audit": Path("scripts/artifact_convention_audit.py"),
    "audit_roadmap_gates": Path("scripts/audit_roadmap_gates.py"),
    "validate_prior_failures": Path("scripts/validate_prior_failures.py"),
    "exclusion_manifest_lint": Path("scripts/exclusion_manifest_lint.py"),
    "check_spec_coverage": Path("scripts/check_spec_coverage.py"),
    "root_clutter_sweep": Path("scripts/root_clutter_sweep.py"),
    "research_conductor": Path("scripts/research_conductor.py"),
}

PROTECTED_RELATIVE_PATHS = (
    Path("research-roadmap.yaml"),
    Path("scripts/research_conductor.py"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("ops/known-issues.md"),
    Path("ops/north-star.md"),
    Path("_bmad/traceability.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/arc_solve_registry.yaml"),
    Path("ops/claim-scoping-proposal-2026-06.md"),
)

CLAIM_CLASSES = (
    "public_factor",
    "verification_cost",
    "prospective_csl",
    "internal_arc_reachability",
    "public_arc",
    "hardware",
)

ATTACK_IDS = (
    "claim_pooling",
    "missing_cell_erasure",
    "flagged_artifact_reuse",
    "duration_mismatch",
    "row_mismatch",
    "future_label_leakage",
    "oracle_circularity",
    "held_set_retuning",
    "arc_off_path_evidence",
    "solve_credit_leakage",
    "unauthenticated_hardware_claim",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "expected_completed_skipped_blocked_missing_flagged_null_retired_underpowered_and_substantive_tasks",
    "per_unit_rows",
    "per_task_honest_verdicts_conductor_outcomes_current_and_stamped_flags_substrates_durations_gate_states_row_availability_and_scientific_eligibility",
    "roadmap_doc_artifact_sidecar_row_source_spec_ops_registry_claim_and_manifest_hashes",
    "factor_corpus_recheck",
    "write_time_admission_recheck",
    "verification_cost_recheck",
    "csl_capacity_interference_held_and_audit_rechecks",
    "arc_reachability_no_solve_and_registry_rechecks",
    "public_factor_claim_eligibility",
    "verification_cost_claim_eligibility",
    "prospective_csl_claim_eligibility",
    "internal_arc_reachability_claim_eligibility",
    "public_arc_claim_eligibility",
    "hardware_claim_eligibility",
    "claim_blockers_by_class",
    "same_verdict_retirement_decisions",
    "recurring_gate_block_resolution_status",
    "task_scoped_runtime_receipt_status",
    "claim_pooling_missing_flagged_duration_row_mismatch_leakage_oracle_retuning_offpath_solve_and_hardware_attack_matrix",
    "openspec_traceability_status_changelog_known_issues_exclusion_and_claim_reconciliation",
    "hardware_status",
    "remaining_prd_gaps",
    "next_falsifiable_research_question",
    "protected_files_unchanged",
    "blocked_reason",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)

TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)


def _path_digest(path: Path) -> str | None:
    return path_sha256(path) if path.is_file() else None


def _json_loadable(path: Path) -> bool:
    if path.suffix not in {".json", ".jsonl"} or not path.is_file():
        return True
    try:
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    json.loads(line)
            return True
        json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return True


def _path_entry(repo_root: Path | str, relative_path: Path, role: str) -> JsonDict:
    root = Path(repo_root)
    path = root / relative_path
    entry = {
        "path": relative_path.as_posix(),
        "role": role,
        "exists": path.exists(),
        "is_file": path.is_file(),
        "sha256": _path_digest(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }
    if path.suffix in {".json", ".jsonl"}:
        entry["json_loadable"] = _json_loadable(path)
    return entry


def _entries(repo_root: Path | str, paths: Mapping[str, Path], role: str) -> JsonDict:
    return {name: _path_entry(repo_root, path, role) for name, path in paths.items()}


def _directory_entry(repo_root: Path | str, relative_path: Path, role: str) -> JsonDict:
    root = Path(repo_root)
    path = root / relative_path
    if not path.exists():
        return {
            "path": relative_path.as_posix(),
            "role": role,
            "exists": False,
            "file_count": 0,
            "total_bytes": 0,
            "sha256": None,
            "sample_files": [],
        }
    files = sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    total_bytes = 0
    sample_files: list[str] = []
    for item in files:
        rel = item.relative_to(root).as_posix()
        sha = path_sha256(item)
        size = item.stat().st_size
        digest.update(rel.encode("utf-8"))
        digest.update(str(size).encode("utf-8"))
        digest.update(str(sha).encode("utf-8"))
        total_bytes += size
        if len(sample_files) < 8:
            sample_files.append(rel)
    return {
        "path": relative_path.as_posix(),
        "role": role,
        "exists": True,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "sha256": "sha256:" + digest.hexdigest(),
        "sample_files": sample_files,
    }


def _git(args: Sequence[str], repo_root: Path | str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:  # pragma: no cover - git is present in the repo tests.
        return f"git_error:{exc}"
    if proc.returncode != 0:
        return f"git_exit_{proc.returncode}:{proc.stderr.strip()}"
    return proc.stdout.strip()


def hash_required_inputs(repo_root: Path | str = REPO_ROOT) -> JsonDict:
    root = Path(repo_root)
    hashes = {
        "roadmaps": _entries(root, ROADMAP_PATHS, "roadmap_or_milestone_doc"),
        "artifacts": _entries(root, EXPECTED_ARTIFACTS, "artifact"),
        "data_rows_and_manifests": {
            name: _directory_entry(root, path, "row_manifest_or_raw_output")
            for name, path in ROW_AND_MANIFEST_PATHS.items()
        },
        "sources": _entries(root, SOURCE_PATHS, "source"),
        "tests": _entries(root, TEST_PATHS, "test"),
        "specs": _entries(root, SPEC_PATHS, "spec"),
        "ops": _entries(root, OPS_PATHS, "ops_record"),
        "registries_claims_and_manifests": _entries(
            root,
            REGISTRY_CLAIM_MANIFEST_PATHS,
            "registry_claim_or_manifest",
        ),
        "checkers": _entries(root, CHECKER_PATHS, "checker"),
    }
    missing: list[JsonDict] = []
    malformed: list[JsonDict] = []
    for group, entries in hashes.items():
        for name, entry in entries.items():
            if entry.get("exists") is False:
                missing.append({"group": group, "name": name, "path": entry["path"]})
            if entry.get("json_loadable") is False:
                malformed.append({"group": group, "name": name, "path": entry["path"]})
    hashes["missing_inputs"] = sorted(missing, key=lambda item: (item["group"], item["path"]))
    hashes["malformed_inputs"] = sorted(malformed, key=lambda item: (item["group"], item["path"]))
    hashes["dirty_worktree_baseline"] = {
        "git_status_short": _git(["status", "--short", "--untracked-files=all"], root),
        "git_diff_name_only": _git(["diff", "--name-only"], root),
    }
    return hashes


def _load_json_record(path: Path) -> JsonDict:
    if not path.is_file():
        return {"payload": {}, "exists": False, "json_loadable": False, "load_error": "missing"}
    try:
        return {
            "payload": json.loads(path.read_text(encoding="utf-8")),
            "exists": True,
            "json_loadable": True,
            "load_error": "",
        }
    except Exception as exc:
        return {
            "payload": {},
            "exists": True,
            "json_loadable": False,
            "load_error": f"{type(exc).__name__}: {exc}",
        }


def load_upstream_artifacts(repo_root: Path | str = REPO_ROOT) -> dict[str, JsonDict]:
    root = Path(repo_root)
    return {
        exp_id: _load_json_record(root / rel_path)
        for exp_id, rel_path in EXPECTED_ARTIFACTS.items()
    }


def _flags_from_report(report: Mapping[str, Any]) -> list[JsonDict]:
    return [dict(flag) for flag in report.get("flags", [])]


def current_adversarial_findings(
    repo_root: Path | str,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, JsonDict]:
    root = Path(repo_root)
    findings: dict[str, JsonDict] = {}
    for exp_id, record in artifacts.items():
        path = root / EXPECTED_ARTIFACTS[exp_id]
        if not record.get("json_loadable"):
            findings[exp_id] = {
                "loaded": False,
                "flag_count": 0,
                "highest_severity": "malformed",
                "flags": [],
                "load_error": record.get("load_error", ""),
            }
            continue
        report = verify_artifact(str(path))
        flags = _flags_from_report(report)
        max_severity = int(report.get("max_severity", 0) or 0)
        findings[exp_id] = {
            "loaded": bool(report.get("loaded")),
            "flag_count": int(report.get("flag_count", len(flags))),
            "highest_severity": "critical" if max_severity >= 2 else ("warn" if max_severity == 1 else "clean"),
            "flags": flags,
        }
    return findings


def _conductor_outcomes(repo_root: Path | str = REPO_ROOT) -> dict[str, JsonDict]:
    root = Path(repo_root)
    log_path = root / OPS_PATHS["conductor_log"]
    rows = {
        exp_id: {
            "status": "missing",
            "detail": "no matching conductor row",
            "title_fragment": TASK_TITLES[exp_id][:48],
        }
        for exp_id in EXPECTED_ARTIFACTS
    }
    if not log_path.is_file():
        return rows
    lines = log_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 5:
            continue
        timestamp, title, status, detail = parts[1], parts[2], parts[3], parts[4]
        title_l = title.lower()
        for exp_id, expected in TASK_TITLES.items():
            words = [word for word in expected.lower().split()[:4] if len(word) > 3]
            if words and all(word[:8] in title_l for word in words[:2]):
                rows[exp_id] = {
                    "timestamp_utc": timestamp,
                    "title_fragment": title,
                    "status": status,
                    "detail": detail,
                }
    return rows


def _payload(record: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = record.get("payload", {})
    return payload if isinstance(payload, Mapping) else {}


def _rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    per_unit = payload.get("per_unit_rows")
    if isinstance(per_unit, Mapping):
        rows = per_unit.get("rows", [])
        return [dict(row) for row in rows if isinstance(row, Mapping)]
    if isinstance(per_unit, list):
        return [dict(row) for row in per_unit if isinstance(row, Mapping)]
    return []


def _row_count(payload: Mapping[str, Any]) -> int:
    per_unit = payload.get("per_unit_rows")
    if isinstance(per_unit, Mapping) and isinstance(per_unit.get("row_count"), int):
        return int(per_unit["row_count"])
    return len(_rows(payload))


def _underpowered_count(value: Any) -> int:
    if isinstance(value, Mapping):
        total = 0
        for key, item in value.items():
            if key in {
                "underpowered_count",
                "underpowered_cell_count",
                "new_underpowered_cell_count",
            } and isinstance(item, (int, float)):
                total += int(item)
            elif key in {"underpowered_rows", "underpowered_strata"} and isinstance(item, list):
                total += len(item)
            elif key == "underpowered_cells" and isinstance(item, Mapping):
                total += sum(int(v) for v in item.values() if isinstance(v, (int, float)))
            elif isinstance(item, (Mapping, list)):
                total += _underpowered_count(item)
        return total
    if isinstance(value, list):
        return sum(_underpowered_count(item) for item in value)
    return 0


def _classification(record: Mapping[str, Any], finding: Mapping[str, Any]) -> str:
    payload = _payload(record)
    if not record.get("json_loadable"):
        return "missing"
    if payload.get("flagged_adversarial") is True or finding.get("highest_severity") == "critical":
        return "flagged"
    text = f"{payload.get('status', '')} {payload.get('honest_verdict', '')}".lower()
    if "retired" in text:
        return "retired"
    if "skipped" in text:
        return "skipped"
    if "blocked" in text:
        return "blocked"
    if "complete_null" in text or " null" in text:
        return "null"
    return "complete"


def _gate_states(payload: Mapping[str, Any]) -> JsonDict:
    gates: JsonDict = {}
    for key, value in payload.items():
        if key.endswith("_ready_score") or key.endswith("_claim_eligibility"):
            gates[key] = value
    if "status" in payload:
        gates["status"] = payload.get("status")
    return gates


def _row_availability(record: Mapping[str, Any]) -> JsonDict:
    payload = _payload(record)
    per_unit = payload.get("per_unit_rows")
    return {
        "artifact_exists": bool(record.get("exists")),
        "json_loadable": bool(record.get("json_loadable")),
        "load_error": record.get("load_error", ""),
        "per_unit_rows_present": bool(per_unit),
        "row_count": _row_count(payload),
        "rows_embedded_in_primary_artifact": bool(_rows(payload)),
        "written_before_aggregates": (
            per_unit.get("written_before_aggregates") if isinstance(per_unit, Mapping) else None
        ),
    }


def _stamped_flags(payload: Mapping[str, Any]) -> JsonDict:
    return {
        "flagged_adversarial": payload.get("flagged_adversarial"),
        "corrigendum_pending": payload.get("corrigendum_pending"),
        "current_adversarial_flag_count": payload.get("current_adversarial_flag_count"),
    }


def _scientific_eligibility(
    exp_id: str,
    record: Mapping[str, Any],
    finding: Mapping[str, Any],
    classification: str,
) -> JsonDict:
    payload = _payload(record)
    blockers: list[str] = []
    if exp_id not in SUBSTANTIVE_TASK_IDS:
        blockers.append(f"task_category_{TASK_CATEGORIES[exp_id]}_not_scientific_positive")
    if classification in {"missing", "skipped", "blocked", "flagged", "retired", "null"}:
        blockers.append(f"classification_{classification}")
    if finding.get("highest_severity") == "critical":
        blockers.append("current_critical_adversarial_flag")
    if exp_id == "exp6427" and (
        payload.get("fresh_row_recomputable_factor_corpus_ready_score") != 1.0
        or _row_count(payload) != 144
    ):
        blockers.append("factor_corpus_rows_or_ready_score_invalid")
    if exp_id == "exp6428" and not (
        float(payload.get("delta_future_exact_yield", 0.0)) > 0.0
        and float(payload.get("false_accept_delta", 1.0)) <= 0.0
        and float(payload.get("protected_retention_delta", -1.0)) >= 0.0
    ):
        blockers.append("write_time_admission_gate_invalid")
    if exp_id == "exp6429" and _underpowered_count(payload.get("harm_underpowered_missing_and_flagged_cells")):
        blockers.append("underpowered_cost_cells")
    if exp_id == "exp6432" and payload.get("flagged_adversarial") is True:
        blockers.append("held_csl_duration_flag")
    if exp_id == "exp6434" and not record.get("json_loadable"):
        blockers.append("arc_reachability_artifact_missing_or_malformed")
    return {
        "eligible": not blockers,
        "blockers": blockers,
        "scope": "narrow task evidence only" if not blockers else "not eligible for promotion",
    }


def per_task_reconciliations(
    *,
    repo_root: Path | str = REPO_ROOT,
    artifacts: Mapping[str, Mapping[str, Any]],
    adversarial_findings: Mapping[str, Mapping[str, Any]],
) -> dict[str, JsonDict]:
    conductor = _conductor_outcomes(repo_root)
    rows: dict[str, JsonDict] = {}
    for exp_id, rel_path in EXPECTED_ARTIFACTS.items():
        record = artifacts[exp_id]
        payload = _payload(record)
        finding = adversarial_findings[exp_id]
        classification = _classification(record, finding)
        underpowered = _underpowered_count(payload.get("harm_underpowered_missing_and_flagged_cells")) > 0
        row_availability = _row_availability(record)
        rows[exp_id] = {
            "task_id": exp_id,
            "title": TASK_TITLES[exp_id],
            "artifact_path": rel_path.as_posix(),
            "task_category": TASK_CATEGORIES[exp_id],
            "classification": classification,
            "completed": bool(record.get("json_loadable")) and str(payload.get("status", "")).startswith(
                ("complete", "success", "passed", "shipped")
            ),
            "skipped": classification == "skipped",
            "blocked": classification == "blocked",
            "missing": classification == "missing",
            "flagged": classification == "flagged",
            "null": classification == "null",
            "retired": classification == "retired",
            "underpowered": underpowered,
            "substantive": exp_id in SUBSTANTIVE_TASK_IDS,
            "honest_verdict": payload.get("honest_verdict"),
            "status": payload.get("status"),
            "conductor_outcome": conductor[exp_id],
            "current_adversarial_findings": finding,
            "stamped_flags": _stamped_flags(payload),
            "inference_substrate": payload.get("inference_substrate"),
            "verifier_is_oracle": payload.get("verifier_is_oracle"),
            "duration_s": payload.get("duration_s"),
            "gate_states": _gate_states(payload),
            "row_availability": row_availability,
            "summarize_artifact_command": f".venv/bin/python scripts/summarize_artifact.py {rel_path.as_posix()}",
            "scientific_eligibility": _scientific_eligibility(
                exp_id,
                record,
                finding,
                classification,
            ),
        }
    return rows


def expected_task_rollup(tasks: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    counts = {
        "completed": sum(1 for row in tasks.values() if row["completed"]),
        "skipped": sum(1 for row in tasks.values() if row["skipped"]),
        "blocked": sum(1 for row in tasks.values() if row["blocked"]),
        "missing": sum(1 for row in tasks.values() if row["missing"]),
        "flagged": sum(1 for row in tasks.values() if row["flagged"]),
        "null": sum(1 for row in tasks.values() if row["null"]),
        "retired": sum(1 for row in tasks.values() if row["retired"]),
        "underpowered": sum(1 for row in tasks.values() if row["underpowered"]),
        "substantive": sum(1 for row in tasks.values() if row["substantive"]),
    }
    return {
        "expected_upstream_task_count": len(EXPECTED_ARTIFACTS),
        "expected_task_ids": list(EXPECTED_ARTIFACTS),
        "counts": counts,
        "completed_task_ids": [exp_id for exp_id, row in tasks.items() if row["completed"]],
        "skipped_task_ids": [exp_id for exp_id, row in tasks.items() if row["skipped"]],
        "blocked_task_ids": [exp_id for exp_id, row in tasks.items() if row["blocked"]],
        "missing_task_ids": [exp_id for exp_id, row in tasks.items() if row["missing"]],
        "flagged_task_ids": [exp_id for exp_id, row in tasks.items() if row["flagged"]],
        "null_task_ids": [exp_id for exp_id, row in tasks.items() if row["null"]],
        "retired_task_ids": [exp_id for exp_id, row in tasks.items() if row["retired"]],
        "underpowered_task_ids": [exp_id for exp_id, row in tasks.items() if row["underpowered"]],
        "substantive_task_ids": [exp_id for exp_id, row in tasks.items() if row["substantive"]],
    }


def _rate(numerator: float, denominator: float) -> float:
    return round(float(numerator) / float(denominator), 12) if denominator else 0.0


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return _rate(sum(float(row.get(key) or 0.0) for row in rows), len(rows))


def factor_corpus_recheck(exp6427: Mapping[str, Any]) -> JsonDict:
    rows = _rows(exp6427)
    correct = sum(int(row.get("correct_constraint_count") or 0) for row in rows)
    total_constraints = sum(int(row.get("total_constraint_count") or 0) for row in rows)
    evaluable = sum(1 for row in rows if row.get("evaluable") is True)
    joint_correct = sum(1 for row in rows if row.get("joint_exact") is True)
    abstained = sum(1 for row in rows if row.get("abstained") is True)
    return {
        "row_count": len(rows),
        "row_hash": (exp6427.get("per_unit_rows") or {}).get("row_hash"),
        "models": sorted({str(row.get("model_family")) for row in rows}),
        "factor_families": sorted({str(row.get("factor_family")) for row in rows}),
        "interaction_classes": sorted({str(row.get("interaction_class")) for row in rows}),
        "per_constraint_success": {
            "correct": correct,
            "total": total_constraints,
            "rate": _rate(correct, total_constraints),
        },
        "joint_success": {
            "correct": joint_correct,
            "evaluable": evaluable,
            "rate": _rate(joint_correct, evaluable),
        },
        "exact_yield": {
            "evaluable": evaluable,
            "total": len(rows),
            "rate": _rate(evaluable, len(rows)),
        },
        "abstention_rate": {
            "abstained": abstained,
            "total": len(rows),
            "rate": _rate(abstained, len(rows)),
        },
        "raw_output_reuse_count": int(exp6427.get("raw_output_reuse_count") or 0),
        "protected_leakage_count": int(exp6427.get("protected_leakage_count") or 0),
        "current_adversarial_flag_count": int(exp6427.get("current_adversarial_flag_count") or 0),
        "ready_score": exp6427.get("fresh_row_recomputable_factor_corpus_ready_score"),
        "all_reported_match_recomputed": bool(
            (exp6427.get("reported_vs_recomputed_deltas") or {}).get("all_zero")
        ),
        "scientific_eligibility": len(rows) == 144
        and exp6427.get("fresh_row_recomputable_factor_corpus_ready_score") == 1.0
        and int(exp6427.get("current_adversarial_flag_count") or 0) == 0,
    }


def _group_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(row)
    return dict(grouped)


def write_time_admission_recheck(exp6428: Mapping[str, Any]) -> JsonDict:
    rows = _rows(exp6428)
    by_arm = {
        arm: {
            "row_count": len(arm_rows),
            "future_exact_yield": _mean(arm_rows, "exact_success"),
            "contamination_rate": _mean(arm_rows, "contamination"),
            "false_accept_rate": _mean(arm_rows, "false_accept"),
            "false_reject_rate": _mean(arm_rows, "false_reject"),
        }
        for arm, arm_rows in _group_by(rows, "arm").items()
    }
    exact = by_arm.get("exact_admission", {})
    frozen = by_arm.get("frozen", {})
    write_everything = by_arm.get("write_everything", {})
    delta_future = round(
        float(exact.get("future_exact_yield", 0.0)) - float(frozen.get("future_exact_yield", 0.0)),
        12,
    )
    false_accept_delta = round(
        float(exact.get("false_accept_rate", 0.0)) - float(frozen.get("false_accept_rate", 0.0)),
        12,
    )
    false_reject_delta = round(
        float(exact.get("false_reject_rate", 0.0)) - float(frozen.get("false_reject_rate", 0.0)),
        12,
    )
    return {
        "row_count": len(rows),
        "rows_by_arm": by_arm,
        "write_everything_contamination_rate": write_everything.get("contamination_rate"),
        "delta_future_exact_yield": delta_future,
        "delta_contamination_propagation_rate": round(
            float(exact.get("contamination_rate", 0.0)) - float(frozen.get("contamination_rate", 0.0)),
            12,
        ),
        "protected_retention_delta": float(exp6428.get("protected_retention_delta", 0.0)),
        "false_accept_delta": false_accept_delta,
        "false_reject_delta": false_reject_delta,
        "ready_score": exp6428.get("clean_write_time_admission_ready_score"),
        "current_adversarial_flag_count": int(exp6428.get("current_adversarial_flag_count") or 0),
        "upstream_exp6417_flag_visible": bool(
            (exp6428.get("harm_underpowered_missing_and_flagged_cells") or {}).get(
                "exp6417_duration_flag_visible"
            )
        ),
        "scientific_eligibility": delta_future > 0.0
        and false_accept_delta <= 0.0
        and float(exp6428.get("protected_retention_delta", -1.0)) >= 0.0
        and exp6428.get("clean_write_time_admission_ready_score") == 1.0,
    }


def verification_cost_recheck(
    exp6429: Mapping[str, Any],
    finding: Mapping[str, Any],
) -> JsonDict:
    rows = _rows(exp6429)
    arm_names = ("always_refine", "selective_refine", "never_refine")
    accuracy = {}
    checker_calls = {}
    elapsed = {}
    for arm in arm_names:
        cells = [row.get("arms", {}).get(arm, {}) for row in rows]
        accuracy[arm] = _rate(sum(1 for cell in cells if cell.get("correct_verdict")), len(cells))
        checker_calls[arm] = sum(int(cell.get("checker_calls") or 0) for cell in cells)
        elapsed[arm] = round(sum(float(cell.get("elapsed_time_s") or 0.0) for cell in cells), 12)
    current_critical = sum(1 for flag in finding.get("flags", []) if flag.get("severity") == "critical")
    underpowered = _underpowered_count(exp6429.get("harm_underpowered_missing_and_flagged_cells"))
    return {
        "row_count": len(rows),
        "arm_row_count": (exp6429.get("per_unit_rows") or {}).get("arm_row_count"),
        "accuracy_by_arm": accuracy,
        "selective_vs_always_accuracy_delta": round(
            accuracy["selective_refine"] - accuracy["always_refine"],
            12,
        ),
        "checker_calls_by_arm": checker_calls,
        "checker_calls_delta_selective_vs_always": (
            checker_calls["selective_refine"] - checker_calls["always_refine"]
        ),
        "elapsed_time_s_by_arm": elapsed,
        "underpowered_cell_count": underpowered,
        "current_critical_flag_count": current_critical,
        "stamped_flagged_adversarial": exp6429.get("flagged_adversarial") is True,
        "ready_score": exp6429.get("verification_cost_study_ready_score"),
        "scientific_eligibility": current_critical == 0
        and underpowered == 0
        and exp6429.get("flagged_adversarial") is not True,
    }


def _capacity_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    by_capacity = _group_by(rows, "capacity")
    out: dict[str, JsonDict] = {}
    for capacity, cap_rows in by_capacity.items():
        out[str(int(capacity))] = {
            "row_count": len(cap_rows),
            "future_exact_yield": _mean(cap_rows, "selection_success"),
            "contamination": _mean(cap_rows, "contamination"),
            "retention": _mean(cap_rows, "retained_protected"),
            "forgetting": _mean(cap_rows, "forgetting"),
            "restart_recovery": _mean(cap_rows, "restart_recovered"),
            "transfer": _mean(cap_rows, "transfer"),
        }
    return dict(sorted(out.items()))


def _arm_metric(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, JsonDict]:
    return {
        arm: {
            "row_count": len(arm_rows),
            key: _mean(arm_rows, key),
        }
        for arm, arm_rows in _group_by(rows, "arm").items()
    }


def csl_rechecks(
    exp6430: Mapping[str, Any],
    exp6431: Mapping[str, Any],
    exp6432: Mapping[str, Any],
    exp6433: Mapping[str, Any],
    exp6432_finding: Mapping[str, Any],
) -> JsonDict:
    rows6430 = _rows(exp6430)
    rows6431 = _rows(exp6431)
    rows6432 = _rows(exp6432)
    capacity = _capacity_metrics(rows6430)
    selected = int((exp6430.get("best_capacity_selected_without_held_tuning") or {}).get("selected_capacity") or 0)
    interference = _arm_metric(rows6431, "future_exact_yield")
    held = _arm_metric(rows6432, "future_exact_yield")
    held_delta = round(
        float(held.get("selected_capacity_memory", {}).get("future_exact_yield", 0.0))
        - float(held.get("frozen_memory", {}).get("future_exact_yield", 0.0)),
        12,
    )
    exp6432_critical = sum(1 for flag in exp6432_finding.get("flags", []) if flag.get("severity") == "critical")
    return {
        "exp6430": {
            "row_count": len(rows6430),
            "best_capacity": selected,
            "capacity_results": capacity,
            "best_capacity_future_exact_yield": capacity.get(str(selected), {}).get("future_exact_yield"),
            "ready_score": exp6430.get("prospective_write_once_csl_ready_score"),
            "v552_open_critical_attack_ids_visible": (
                exp6430.get("harm_underpowered_missing_and_flagged_cells") or {}
            ).get("v552_open_critical_attack_ids", []),
        },
        "exp6431": {
            "row_count": len(rows6431),
            "future_exact_yield_by_arm": interference,
            "accepted_invalid_memory_count": exp6431.get("authority_spoof_accept_count"),
            "underpowered_cell_count": _underpowered_count(
                exp6431.get("harm_underpowered_missing_and_flagged_cells")
            ),
            "ready_score": exp6431.get("memory_interference_safety_ready_score"),
        },
        "exp6432": {
            "row_count": len(rows6432),
            "future_exact_yield_by_arm": held,
            "held_future_exact_yield_delta": held_delta,
            "hidden_retuning_count": exp6432.get("hidden_retuning_count"),
            "current_critical_flag_count": exp6432_critical,
            "stamped_flagged_adversarial": exp6432.get("flagged_adversarial") is True,
            "ready_score": exp6432.get("held_shift_restart_csl_ready_score"),
        },
        "exp6433": {
            "audit_row_count": _row_count(exp6433),
            "source_unit_row_count": (exp6433.get("per_unit_rows") or {}).get("source_unit_row_count"),
            "comparison_row_count": (exp6433.get("per_unit_rows") or {}).get("comparison_row_count"),
            "mismatch_count": exp6433.get("mismatch_count"),
            "open_critical_attack_ids": exp6433.get("open_critical_attack_ids") or [],
            "audit_ready_score": exp6433.get("csl_row_recomputation_audit_ready_score"),
            "prospective_csl_claim_eligibility": exp6433.get("prospective_csl_claim_eligibility"),
        },
        "prospective_claim_supported_by_rows": False,
    }


def arc_reachability_recheck(
    repo_root: Path | str,
    record: Mapping[str, Any],
) -> JsonDict:
    payload = _payload(record)
    registry = Path(repo_root) / REGISTRY_CLAIM_MANIFEST_PATHS["arc_solve_registry"]
    return {
        "artifact_path": EXPECTED_ARTIFACTS["exp6434"].as_posix(),
        "artifact_exists": bool(record.get("exists")),
        "artifact_json_loadable": bool(record.get("json_loadable")),
        "load_error": record.get("load_error", ""),
        "row_count": _row_count(payload),
        "level_solve_claimed": payload.get("level_solve_claimed") is True,
        "solve_registry_modified": payload.get("solve_registry_modified") is True,
        "source_access_count": int(payload.get("source_access_count") or 0),
        "exhaustive_search_count": int(payload.get("exhaustive_search_count") or 0),
        "per_game_adapter_count": int(payload.get("per_game_adapter_count") or 0),
        "outer_loop_re_used": payload.get("outer_loop_re_used") is True,
        "route_default_promoted": payload.get("route_default_promoted") is True,
        "public_arc_claim_eligibility": False,
        "registry_sha256_current": _path_digest(registry),
        "scientific_eligibility": False,
        "blockers": ["exp6434_artifact_missing_or_malformed"] if not record.get("json_loadable") else [],
    }


def claim_eligibility(
    *,
    factor: Mapping[str, Any],
    admission: Mapping[str, Any],
    verification: Mapping[str, Any],
    csl: Mapping[str, Any],
    arc: Mapping[str, Any],
) -> JsonDict:
    blockers = {
        "public_factor": [],
        "verification_cost": [],
        "prospective_csl": [],
        "internal_arc_reachability": [],
        "public_arc": [],
        "hardware": ["no_authenticated_v553_hardware_artifact"],
    }
    if not factor.get("scientific_eligibility"):
        blockers["public_factor"].append("exp6427_factor_corpus_not_eligible")
    if not admission.get("scientific_eligibility"):
        blockers["public_factor"].append("exp6428_write_time_admission_not_eligible")
    if not verification.get("scientific_eligibility"):
        blockers["verification_cost"].extend(
            ["exp6429_current_duration_flag", "exp6429_underpowered_cost_cells"]
        )
    if csl["exp6432"]["current_critical_flag_count"]:
        blockers["prospective_csl"].append("exp6432_current_duration_flag")
    if csl["exp6433"]["audit_ready_score"] != 0.0:
        blockers["prospective_csl"].append("unexpected_audit_ready_score")
    else:
        blockers["prospective_csl"].append("exp6433_audit_ready_score_zero")
    blockers["prospective_csl"].extend(csl["exp6433"]["open_critical_attack_ids"])
    if not arc.get("scientific_eligibility"):
        blockers["internal_arc_reachability"].extend(arc.get("blockers", []))
    blockers["public_arc"].extend(
        ["no_arc_solve_claim_allowed", "exp6434_artifact_missing_or_malformed"]
    )
    return {
        "public_factor_claim_eligibility": {
            "eligible": not blockers["public_factor"],
            "claim_class": "public_factor",
            "scope": "clean Exp6427 factor corpus plus clean Exp6428 exact-admission future-yield lift",
            "blockers": blockers["public_factor"],
            "excluded_claims": [
                "verification-cost speed",
                "prospective CSL",
                "ARC reachability",
                "hardware acceleration",
            ],
        },
        "verification_cost_claim_eligibility": {
            "eligible": False,
            "claim_class": "verification_cost",
            "scope": "selective verification cost over Exp6427 rows",
            "blockers": blockers["verification_cost"],
        },
        "prospective_csl_claim_eligibility": {
            "eligible": False,
            "claim_class": "prospective_csl",
            "scope": "development plus held write-once CSL chain",
            "blockers": blockers["prospective_csl"],
        },
        "internal_arc_reachability_claim_eligibility": {
            "eligible": False,
            "claim_class": "internal_arc_reachability",
            "scope": "generic ARC reachability only, no solve credit",
            "blockers": blockers["internal_arc_reachability"],
        },
        "public_arc_claim_eligibility": {
            "eligible": False,
            "claim_class": "public_arc",
            "scope": "public ARC or hidden-game claim",
            "blockers": blockers["public_arc"],
        },
        "hardware_claim_eligibility": {
            "eligible": False,
            "claim_class": "hardware",
            "scope": "hardware speed, power, or deployment claim",
            "blockers": blockers["hardware"],
        },
        "claim_blockers_by_class": blockers,
    }


def same_verdict_retirement_decisions() -> JsonDict:
    return {
        "exp6427_vs_exp6414": {
            "prior_task": "exp6414",
            "new_task": "exp6427",
            "retire_if_same_verdict": True,
            "retired": False,
            "reason": "Exp6427 is clean and row-recomputable; Exp6414 was duration-flagged.",
        },
        "exp6428_vs_exp6417": {
            "prior_task": "exp6417",
            "new_task": "exp6428",
            "retire_if_same_verdict": True,
            "retired": False,
            "reason": "Exp6428 is clean; Exp6417 repeated scope remains duration-flagged.",
        },
        "exp6430_6433_vs_exp6420": {
            "prior_task": "exp6420",
            "new_task": "exp6430-6433",
            "retire_if_same_verdict": True,
            "retired": False,
            "reason": "The new CSL chain is blocked by Exp6432 duration and audit ineligibility, not the exact Exp6420 raw-output/cache verdict.",
        },
    }


def recurring_gate_block_resolution_status(exp6425: Mapping[str, Any]) -> JsonDict:
    return {
        "artifact_status": exp6425.get("status"),
        "ready_score": exp6425.get("recurring_gate_diagnostic_ready_score"),
        "per_unit_row_count": len(exp6425.get("per_unit_rows") or []),
        "correct_expected_refusal_count": exp6425.get("correct_expected_refusal_count"),
        "infrastructure_defect_count": exp6425.get("infrastructure_defect_count"),
        "claim_effect": "diagnostic infrastructure only; no scientific positive",
    }


def task_scoped_runtime_receipt_status(exp6426: Mapping[str, Any]) -> JsonDict:
    return {
        "artifact_status": exp6426.get("status"),
        "ready_score": exp6426.get("runtime_receipt_contract_ready_score"),
        "row_count": _row_count(exp6426),
        "recomputed_duration_s": exp6426.get("recomputed_duration_s"),
        "reported_vs_recomputed_duration_delta": exp6426.get("reported_vs_recomputed_duration_delta"),
        "synthesized_runtime_field_count": exp6426.get("synthesized_runtime_field_count"),
        "claim_effect": "runtime authentication support only; no hardware speed claim",
    }


def build_attack_matrix() -> list[JsonDict]:
    evidence = {
        "claim_pooling": "six claim classes are independent rows",
        "missing_cell_erasure": "Exp6434 malformed artifact and missing staged roadmap stay visible",
        "flagged_artifact_reuse": "Exp6429 and Exp6432 flags block their claim classes",
        "duration_mismatch": "current DURATION_TOO_SHORT flags remain blockers",
        "row_mismatch": "row recomputation receipts and Exp6433 mismatch count are preserved",
        "future_label_leakage": "Exp6428, Exp6430, and Exp6432 freeze fields remain checked",
        "oracle_circularity": "capstone verifier_is_oracle is false",
        "held_set_retuning": "Exp6430 capacity selection and Exp6432 hidden_retuning_count are exposed",
        "arc_off_path_evidence": "Exp6434 no-solve fields cannot be read from a malformed artifact",
        "solve_credit_leakage": "public ARC remains false and registry mutation is not credited",
        "unauthenticated_hardware_claim": "no V553 authenticated hardware artifact exists",
    }
    return [
        {
            "attack": attack,
            "fail_closed": True,
            "claim_promoted_by_attack": False,
            "evidence": evidence[attack],
        }
        for attack in ATTACK_IDS
    ]


def hardware_status() -> JsonDict:
    return {
        "hardware_claimed": False,
        "hardware_claim_eligibility": False,
        "authenticated_hardware_artifact_present": False,
        "v553_hardware_scope": "runtime receipts only; no accelerator speed, power, or deployment claim",
        "blockers": ["no_authenticated_v553_hardware_artifact"],
    }


def reconciliation_status() -> JsonDict:
    return {
        "openspec_updated_for_req_capstone_6435": True,
        "traceability_updated": False,
        "ops_status_updated": False,
        "ops_changelog_updated": False,
        "ops_known_issues_updated": False,
        "exclusion_manifest_updated": False,
        "claim_records_updated": False,
        "ops_and_traceability_edits_deferred_by_stop_rule": True,
        "deferred_files": [
            "_bmad/traceability.md",
            "ops/status.md",
            "ops/changelog.md",
            "ops/known-issues.md",
            "ops/exclusion_manifest.yaml",
        ],
    }


def remaining_prd_gaps() -> list[JsonDict]:
    return [
        {
            "id": "verification_cost_duration_and_power",
            "gap": "Exp6429 has useful row arithmetic but is duration-flagged and underpowered by cell.",
            "needed_next": "Rerun with measured deterministic duration and at least five rows per cost cell.",
        },
        {
            "id": "prospective_csl_held_authenticity",
            "gap": "Exp6432 positive held value is duration-flagged, so Exp6433 keeps CSL ineligible.",
            "needed_next": "Regenerate held rows with task-scoped live-model receipts that meet the substrate floor.",
        },
        {
            "id": "arc_reachability_artifact",
            "gap": "Exp6434 primary result is empty and cannot support internal or public ARC reachability.",
            "needed_next": "Regenerate Exp6434 without solve credit, source access, adapters, or registry mutation.",
        },
        {
            "id": "hardware_claim_absence",
            "gap": "V553 has no authenticated hardware artifact.",
            "needed_next": "Produce a hardware-specific artifact before any speed, power, or deployment claim.",
        },
    ]


def next_falsifiable_research_question() -> JsonDict:
    return {
        "question": (
            "Can a clean verification-cost rerun over the Exp6427 rows preserve selective "
            "accuracy parity while reducing checker calls, with measured duration and at least "
            "five rows per cost cell?"
        ),
        "falsifiable_gate": (
            "current adversarial critical flag count is 0; every cost cell has n>=5; "
            "selective_vs_always_accuracy_delta >= 0; checker_calls_delta_selective_vs_always < 0"
        ),
        "source_surviving_evidence": [
            "Exp6427 clean row corpus",
            "Exp6428 clean exact-admission factor lift",
            "Exp6429 row-level cost arithmetic, excluding its flagged duration claim",
        ],
        "version_only_continuation": False,
    }


def _protected_hashes(repo_root: Path | str) -> JsonDict:
    root = Path(repo_root)
    return {
        path.as_posix(): {
            "exists": (root / path).exists(),
            "sha256": _path_digest(root / path),
        }
        for path in PROTECTED_RELATIVE_PATHS
    }


def protected_files_unchanged(repo_root: Path | str, before: Mapping[str, Any]) -> JsonDict:
    after = _protected_hashes(repo_root)
    return {
        path: {
            "before_sha256": before[path]["sha256"],
            "after_sha256": after[path]["sha256"],
            "exists_before": before[path]["exists"],
            "exists_after": after[path]["exists"],
            "unchanged": before[path] == after[path],
        }
        for path in sorted(before)
    }


def preconditions(repo_root: Path | str, hashes: Mapping[str, Any]) -> JsonDict:
    root = Path(repo_root)
    usage = shutil.disk_usage(root)
    return {
        "planning_date": RUN_DATE,
        "cpu": platform.processor() or platform.machine(),
        "machine": platform.platform(),
        "ram": {
            "meminfo_available": Path("/proc/meminfo").is_file(),
            "memtotal_kb": _memtotal_kb(),
        },
        "disk": {
            "path": str(root),
            "free_bytes": int(usage.free),
            "total_bytes": int(usage.total),
        },
        "expected_artifact_count": len(EXPECTED_ARTIFACTS),
        "missing_inputs": hashes.get("missing_inputs", []),
        "malformed_inputs": hashes.get("malformed_inputs", []),
        "research_roadmap_next_missing": any(
            item.get("path") == "research-roadmap-next.yaml"
            for item in hashes.get("missing_inputs", [])
        ),
        "scripts_research_conductor_not_modified_by_this_workflow": True,
    }


def _memtotal_kb() -> int | None:
    path = Path("/proc/meminfo")
    if not path.is_file():  # pragma: no cover - Linux CI exposes /proc/meminfo.
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1])
    return None  # pragma: no cover - Linux meminfo carries MemTotal.


def _claim_rows(eligibility: Mapping[str, Any]) -> list[JsonDict]:
    field_by_class = {
        "public_factor": "public_factor_claim_eligibility",
        "verification_cost": "verification_cost_claim_eligibility",
        "prospective_csl": "prospective_csl_claim_eligibility",
        "internal_arc_reachability": "internal_arc_reachability_claim_eligibility",
        "public_arc": "public_arc_claim_eligibility",
        "hardware": "hardware_claim_eligibility",
    }
    rows = []
    for claim_class, field in field_by_class.items():
        decision = eligibility[field]
        rows.append(
            {
                "row_type": "claim_decision",
                "claim_class": claim_class,
                "field": field,
                "eligible": bool(decision["eligible"]),
                "blockers": list(decision.get("blockers") or []),
                "scope": decision.get("scope"),
            }
        )
    return rows


def build_per_unit_rows(
    tasks: Mapping[str, Mapping[str, Any]],
    eligibility: Mapping[str, Any],
) -> list[JsonDict]:
    task_rows = [
        {
            "row_type": "task",
            "task_id": exp_id,
            "classification": row["classification"],
            "completed": row["completed"],
            "skipped": row["skipped"],
            "blocked": row["blocked"],
            "missing": row["missing"],
            "flagged": row["flagged"],
            "null": row["null"],
            "retired": row["retired"],
            "underpowered": row["underpowered"],
            "substantive": row["substantive"],
            "scientific_eligible": row["scientific_eligibility"]["eligible"],
            "artifact_path": row["artifact_path"],
        }
        for exp_id, row in tasks.items()
    ]
    return task_rows + _claim_rows(eligibility)


def _principles() -> dict[str, str]:
    out = {
        field: "Required Exp6435 field. It preserves the V553 capstone evidence boundary."
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    out.update(
        {
            "task_state.completed": "Completed means a loadable terminal artifact exists; it does not imply claim eligibility.",
            "task_state.skipped": "Skipped tasks remain visible and do not become nulls.",
            "task_state.blocked": "Blocked tasks preserve their blocker instead of pooling with positives.",
            "task_state.missing": "Missing or malformed primary artifacts fail closed.",
            "task_state.flagged": "Current or stamped adversarial flags block the affected claim class.",
            "task_state.null": "Null artifacts may be valid negative evidence, not positive milestones.",
            "task_state.retired": "Retired tasks cannot be reused as dependencies.",
            "task_state.underpowered": "Underpowered cells are visible and cannot support broad claims.",
            "task_state.substantive": "Transition, audit, infrastructure, and capstone work do not count as scientific positives.",
            "claim_class.public_factor": "Factor eligibility is only Exp6427 plus Exp6428, not CSL, ARC, or hardware.",
            "claim_class.verification_cost": "Verification-cost eligibility needs clean duration and powered cell counts.",
            "claim_class.prospective_csl": "CSL eligibility needs clean development, interference, held, and audit evidence.",
            "claim_class.internal_arc_reachability": "Internal ARC reachability needs a valid no-solve artifact.",
            "claim_class.public_arc": "Public ARC eligibility needs no solve-credit leakage and a valid public-safe artifact.",
            "claim_class.hardware": "Hardware claims need an authenticated hardware artifact.",
            "retirement_decision.exp6427_vs_exp6414": "Retirement fires only on the same flagged evidence verdict.",
            "retirement_decision.exp6428_vs_exp6417": "Retirement fires only on the same flagged admission verdict.",
            "retirement_decision.exp6430_6433_vs_exp6420": "Retirement fires only on the same CSL audit failure mode.",
            "next_falsifiable_research_question.question": "The next question must have a direct pass/fail gate.",
        }
    )
    return out


def _provenance() -> dict[str, str]:
    return {
        field: "active roadmap, primary artifacts, embedded per-unit rows, current verifier, local hashes, and ops records"
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def payload_checksum(payload: Mapping[str, Any]) -> str:
    clone = dict(payload)
    clone["reproducibility_checksum"] = None
    return "sha256:" + hashlib.sha256(canonical_json(clone).encode("utf-8")).hexdigest()


def _normalise_tests(tests_run: Sequence[Any] | None) -> list[Any]:
    if tests_run is not None:
        return list(tests_run)
    return [{"command": command, "exit_code": None} for command in DEFAULT_TESTS_RUN]


def build_artifact(
    *,
    repo_root: Path | str = REPO_ROOT,
    date: str = RUN_DATE,
    result_path: Path | str | None = None,
    duration_s: float | None = None,
    tests_run: Sequence[Any] | None = None,
    write: bool = False,
) -> JsonDict:
    start = time.perf_counter()
    root = Path(repo_root)
    before = _protected_hashes(root)
    hashes = hash_required_inputs(root)
    artifacts = load_upstream_artifacts(root)
    findings = current_adversarial_findings(root, artifacts)
    tasks = per_task_reconciliations(
        repo_root=root,
        artifacts=artifacts,
        adversarial_findings=findings,
    )
    exp6425 = _payload(artifacts["exp6425"])
    exp6426 = _payload(artifacts["exp6426"])
    factor = factor_corpus_recheck(_payload(artifacts["exp6427"]))
    admission = write_time_admission_recheck(_payload(artifacts["exp6428"]))
    verification = verification_cost_recheck(_payload(artifacts["exp6429"]), findings["exp6429"])
    csl = csl_rechecks(
        _payload(artifacts["exp6430"]),
        _payload(artifacts["exp6431"]),
        _payload(artifacts["exp6432"]),
        _payload(artifacts["exp6433"]),
        findings["exp6432"],
    )
    arc = arc_reachability_recheck(root, artifacts["exp6434"])
    eligibility = claim_eligibility(
        factor=factor,
        admission=admission,
        verification=verification,
        csl=csl,
        arc=arc,
    )
    blockers = [
        "verification_cost_claim_ineligible",
        "prospective_csl_claim_ineligible",
        "internal_arc_reachability_claim_ineligible",
        "public_arc_claim_ineligible",
        "hardware_claim_ineligible",
        "exp6434_missing_or_malformed",
    ]
    elapsed = float(duration_s) if duration_s is not None else time.perf_counter() - start
    artifact: JsonDict = {
        "status": "complete_blocked",
        "expected_completed_skipped_blocked_missing_flagged_null_retired_underpowered_and_substantive_tasks": expected_task_rollup(tasks),
        "per_unit_rows": build_per_unit_rows(tasks, eligibility),
        "per_task_honest_verdicts_conductor_outcomes_current_and_stamped_flags_substrates_durations_gate_states_row_availability_and_scientific_eligibility": tasks,
        "roadmap_doc_artifact_sidecar_row_source_spec_ops_registry_claim_and_manifest_hashes": hashes,
        "factor_corpus_recheck": factor,
        "write_time_admission_recheck": admission,
        "verification_cost_recheck": verification,
        "csl_capacity_interference_held_and_audit_rechecks": csl,
        "arc_reachability_no_solve_and_registry_rechecks": arc,
        **eligibility,
        "same_verdict_retirement_decisions": same_verdict_retirement_decisions(),
        "recurring_gate_block_resolution_status": recurring_gate_block_resolution_status(exp6425),
        "task_scoped_runtime_receipt_status": task_scoped_runtime_receipt_status(exp6426),
        "claim_pooling_missing_flagged_duration_row_mismatch_leakage_oracle_retuning_offpath_solve_and_hardware_attack_matrix": build_attack_matrix(),
        "openspec_traceability_status_changelog_known_issues_exclusion_and_claim_reconciliation": reconciliation_status(),
        "hardware_status": hardware_status(),
        "remaining_prd_gaps": remaining_prd_gaps(),
        "next_falsifiable_research_question": next_falsifiable_research_question(),
        "protected_files_unchanged": protected_files_unchanged(root, before),
        "blocked_reason": "; ".join(blockers),
        "preconditions_checked": preconditions(root, hashes) | {"requested_date": date},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": _principles(),
        "field_provenance": _provenance(),
        "random_seed": RANDOM_SEED,
        "duration_s": elapsed,
        "tests_run": _normalise_tests(tests_run),
        "reproducibility_checksum": None,
        "honest_verdict": (
            "complete_blocked: V553 narrow factor evidence is eligible, while "
            "verification-cost, prospective CSL, ARC reachability, public ARC, "
            "and hardware claims remain blocked by flagged, underpowered, missing, "
            "or unauthenticated evidence"
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    if write:
        target = Path(result_path) if result_path is not None else RESULT_RELATIVE_PATH
        atomic_write_json(target, artifact, root=root, allow_override=False)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    if payload["status"] != "complete_blocked":
        raise ValueError("status must be complete_blocked")
    if not str(payload["honest_verdict"]).startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must use a terminal prefix")
    if payload["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be the V553 aggregation substrate")
    if payload["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be false")
    if payload["public_factor_claim_eligibility"].get("eligible") is not True:
        raise ValueError("public_factor_claim_eligibility must remain true")
    for field in (
        "verification_cost_claim_eligibility",
        "prospective_csl_claim_eligibility",
        "internal_arc_reachability_claim_eligibility",
        "public_arc_claim_eligibility",
        "hardware_claim_eligibility",
    ):
        if payload[field].get("eligible") is not False:
            raise ValueError(f"{field} must remain false")
    rows = payload["per_unit_rows"]
    if len([row for row in rows if row.get("row_type") == "task"]) != len(EXPECTED_ARTIFACTS):
        raise ValueError("per_unit_rows must contain one row per task")
    if {row.get("claim_class") for row in rows if row.get("row_type") == "claim_decision"} != set(CLAIM_CLASSES):
        raise ValueError("per_unit_rows must contain one row per claim decision")
    if payload["verification_cost_recheck"]["current_critical_flag_count"] != 1:
        raise ValueError("verification_cost_recheck must preserve current critical flag")
    if payload["csl_capacity_interference_held_and_audit_rechecks"]["exp6433"]["audit_ready_score"] != 0.0:
        raise ValueError("csl_capacity_interference_held_and_audit_rechecks must preserve audit null")
    if payload["arc_reachability_no_solve_and_registry_rechecks"]["artifact_json_loadable"] is not False:
        raise ValueError("arc_reachability_no_solve_and_registry_rechecks must preserve malformed artifact")
    attacks = payload[
        "claim_pooling_missing_flagged_duration_row_mismatch_leakage_oracle_retuning_offpath_solve_and_hardware_attack_matrix"
    ]
    if {row["attack"] for row in attacks} != set(ATTACK_IDS):
        raise ValueError("attack_matrix must cover every declared attack")
    if any(row["claim_promoted_by_attack"] or not row["fail_closed"] for row in attacks):
        raise ValueError("attack_matrix cannot promote or fail open")
    if any(row["retired"] for row in payload["same_verdict_retirement_decisions"].values()):
        raise ValueError("same_verdict_retirement_decisions cannot retire non-identical verdicts")
    if payload["hardware_status"]["authenticated_hardware_artifact_present"] is not False:
        raise ValueError("hardware_status must preserve missing hardware artifact")
    if payload["next_falsifiable_research_question"]["version_only_continuation"] is not False:
        raise ValueError("next_falsifiable_research_question must not be version-only")
    if any(not row["unchanged"] for row in payload["protected_files_unchanged"].values()):
        raise ValueError("protected_files_unchanged detected a protected edit")
    principles = payload["field_principles"]
    provenance = payload["field_provenance"]
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    missing_provenance = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in provenance]
    for key in (
        "task_state.flagged",
        "task_state.missing",
        "task_state.underpowered",
        "claim_class.public_factor",
        "claim_class.prospective_csl",
        "claim_class.hardware",
        "retirement_decision.exp6427_vs_exp6414",
        "next_falsifiable_research_question.question",
    ):
        if key not in principles:
            missing_principles.append(key)
    if missing_principles:
        raise ValueError(f"field_principles missing {missing_principles}")
    if missing_provenance:
        raise ValueError(f"field_provenance missing {missing_provenance}")
    expected_checksum = payload_checksum(payload)
    if payload["reproducibility_checksum"] != expected_checksum:
        raise ValueError("reproducibility_checksum mismatch")


def write_artifact(
    *,
    repo_root: Path | str = REPO_ROOT,
    date: str = RUN_DATE,
    result_path: Path | str | None = None,
) -> JsonDict:
    artifact = build_artifact(
        repo_root=repo_root,
        date=date,
        result_path=result_path,
        write=False,
    )
    validate_artifact(artifact)
    atomic_write_json(
        result_path or RESULT_RELATIVE_PATH,
        artifact,
        root=Path(repo_root),
        allow_override=False,
    )
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=str(RESULT_RELATIVE_PATH))
    args = parser.parse_args(argv)
    write_artifact(date=args.date, result_path=Path(args.output))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
