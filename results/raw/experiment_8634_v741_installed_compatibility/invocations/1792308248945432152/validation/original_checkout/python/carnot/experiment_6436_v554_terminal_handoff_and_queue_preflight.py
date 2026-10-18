"""Exp6436 V553 terminal handoff and V554 queue preflight.

Spec refs: REQ-INFRA-6436, SCENARIO-INFRA-6436-1,
SCENARIO-INFRA-6436-2, SCENARIO-INFRA-6436-3,
SCENARIO-INFRA-6436-4, SCENARIO-INFRA-6436-5,
SCENARIO-INFRA-6436-6.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any

from carnot.experiment_6272_v541_terminal_transition import (
    exp_number,
    gate_ok,
    git_status_lines,
    load_retired_exp_ids,
    prior_ok,
    read_yaml_mapping,
    required_artifact_fields_from_prompt,
)
from carnot.experiment_6284_v542_terminal_transition import model_specs_named_in_prompt
from carnot.experiment_6404_v551_terminal_handoff_and_queue_preflight import (
    ALLOWED_HONEST_PREFIXES,
    FINAL_PROHIBITION_LINE,
    GGUF_ID_RE,
    MANDATED_GGUF_IDS,
    _gate_expression,
    _live_adversarial,
    _risk_rows,
    read_json_mapping,
    render_prompt,
)
from carnot.experiment_artifacts import atomic_write_json
from carnot.terminal_artifacts import canonical_json, path_sha256, payload_sha256


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE_V553 = "2026.08.553"
MILESTONE_V554 = "2026.08.554"
RUN_DATE = "20260815"
EXPERIMENT_ID = "exp6436-v554-terminal-handoff-and-queue-preflight"
SCHEMA = "carnot.experiment_6436.v554_terminal_handoff_and_queue_preflight.v1"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6436_v554_terminal_handoff_and_queue_preflight.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

ACTIVE_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
MILESTONE_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
RESEARCH_PROGRAM_RELATIVE_PATH = Path("research-program.md")
RESEARCH_HARDWARE_WISHLIST_RELATIVE_PATH = Path("research-hardware-wishlist.md")
PRD_RELATIVE_PATH = Path("_bmad/prd.md")
ARCHITECTURE_RELATIVE_PATH = Path("_bmad/architecture.md")
RESEARCH_CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
ROADMAP_SCHEMA_RELATIVE_PATH = Path("scripts/roadmap_schema.py")
PRIOR_FAILURE_LINT_RELATIVE_PATH = Path("scripts/validate_prior_failures.py")
EXCLUSION_LINT_RELATIVE_PATH = Path("scripts/exclusion_manifest_lint.py")
CONDUCTOR_GATES_RELATIVE_PATH = Path("scripts/conductor_gates.py")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")

EXPECTED_V553_TASK_IDS = (
    "exp6427-fresh-constraint-saturation-factor-corpus",
    "exp6428-clean-write-time-factor-admission-ab",
    "exp6429-constraint-saturation-verification-cost-ab",
    "exp6430-prospective-write-once-memory-capacity-frontier",
    "exp6431-controlled-memory-interference-ab",
    "exp6432-held-shift-process-restart-csl-replication",
    "exp6433-csl-row-recomputation-safety-audit",
    "exp6434-arc-state-key-reachability-ab",
    "exp6435-v553-adversarial-capstone",
)
V553_DELIVERABLES_BY_TASK = {
    "exp6427-fresh-constraint-saturation-factor-corpus": (
        "results/experiment_6427_fresh_constraint_saturation_factor_corpus.json"
    ),
    "exp6428-clean-write-time-factor-admission-ab": (
        "results/experiment_6428_clean_write_time_factor_admission_ab.json"
    ),
    "exp6429-constraint-saturation-verification-cost-ab": (
        "results/experiment_6429_constraint_saturation_verification_cost_ab.json"
    ),
    "exp6430-prospective-write-once-memory-capacity-frontier": (
        "results/experiment_6430_prospective_write_once_memory_capacity_frontier.json"
    ),
    "exp6431-controlled-memory-interference-ab": (
        "results/experiment_6431_controlled_memory_interference_ab.json"
    ),
    "exp6432-held-shift-process-restart-csl-replication": (
        "results/experiment_6432_held_shift_process_restart_csl_replication.json"
    ),
    "exp6433-csl-row-recomputation-safety-audit": (
        "results/experiment_6433_csl_row_recomputation_safety_audit.json"
    ),
    "exp6434-arc-state-key-reachability-ab": (
        "results/experiment_6434_arc_state_key_reachability_ab.json"
    ),
    "exp6435-v553-adversarial-capstone": (
        "results/experiment_6435_v553_adversarial_capstone.json"
    ),
}
CAPSTONE_TASK_ID = "exp6435-v553-adversarial-capstone"
CAPSTONE_RELATIVE_PATH = Path(V553_DELIVERABLES_BY_TASK[CAPSTONE_TASK_ID])

EXPECTED_V554_TASK_IDS = (
    "exp6436-v554-terminal-handoff-and-queue-preflight",
    "exp6437-generation-to-verdict-receipt-replay-contract",
    "exp6438-powered-verification-cost-repair-ab",
    "exp6439-factor-clause-influence-ab",
    "exp6440-held-factor-revocation-binding-shift-ab",
    "exp6441-prospective-query-conditioned-factor-reuse",
    "exp6442-skill-misevolution-quarantine-rollback-ab",
    "exp6443-fresh-held-restart-csl-replication",
    "exp6444-csl-lifecycle-recomputation-audit",
    "exp6445-arc-state-key-reachability-sharded-ab",
    "exp6446-joint-pathway-dependence-audit",
    "exp6447-v554-adversarial-capstone",
)
REQUIRED_PRIOR_FAILURE_TASK_IDS = (
    "exp6438-powered-verification-cost-repair-ab",
    "exp6440-held-factor-revocation-binding-shift-ab",
    "exp6441-prospective-query-conditioned-factor-reuse",
    "exp6443-fresh-held-restart-csl-replication",
    "exp6444-csl-lifecycle-recomputation-audit",
    "exp6445-arc-state-key-reachability-sharded-ab",
)
LLM_TASK_IDS = (
    "exp6439-factor-clause-influence-ab",
    "exp6440-held-factor-revocation-binding-shift-ab",
    "exp6441-prospective-query-conditioned-factor-reuse",
    "exp6442-skill-misevolution-quarantine-rollback-ab",
    "exp6443-fresh-held-restart-csl-replication",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m "
    "carnot.experiment_6436_v554_terminal_handoff_and_queue_preflight "
    "--date 20260815"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6436_v554_terminal_handoff_and_queue_preflight.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6436_v554_terminal_handoff_and_queue_preflight.py "
    "-m pytest "
    "tests/python/test_experiment_6436_v554_terminal_handoff_and_queue_preflight.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6436_v554_terminal_handoff_and_queue_preflight.py "
    "--fail-under=100 --show-missing"
)
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6436_v554_terminal_handoff_and_queue_preflight.py",
    ".venv/bin/python -c 'import yaml; from pathlib import Path; "
    "from scripts.roadmap_schema import Roadmap; "
    'Roadmap.model_validate(yaml.safe_load(Path("research-roadmap.yaml").read_text()))'
    "'",
    ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml",
    ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
    ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml",
    ".venv/bin/python scripts/artifact_convention_audit.py",
    ".venv/bin/python scripts/determination_preservation_lint.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    ".venv/bin/pytest tests/python -q",
    RUN_COMMAND,
)
EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6436_test_receipts.json")

PROTECTED_RELATIVE_PATHS = (
    ACTIVE_ROADMAP_RELATIVE_PATH,
    MILESTONE_DOC_RELATIVE_PATH,
    RESEARCH_COMPLETE_RELATIVE_PATH,
    RESEARCH_REFERENCES_RELATIVE_PATH,
    RESEARCH_HARDWARE_WISHLIST_RELATIVE_PATH,
    RESEARCH_PROGRAM_RELATIVE_PATH,
    PRD_RELATIVE_PATH,
    ARCHITECTURE_RELATIVE_PATH,
    RESEARCH_CONDUCTOR_RELATIVE_PATH,
    ROADMAP_SCHEMA_RELATIVE_PATH,
    PRIOR_FAILURE_LINT_RELATIVE_PATH,
    EXCLUSION_LINT_RELATIVE_PATH,
    CONDUCTOR_GATES_RELATIVE_PATH,
    ADVERSARIAL_VERIFY_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    *[Path(path) for path in V553_DELIVERABLES_BY_TASK.values()],
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "v553_terminal_rows",
    "v553_artifact_count",
    "v553_missing_or_zero_byte_artifacts",
    "v553_flagged_artifacts",
    "v553_underpowered_artifacts",
    "v553_terminal_claim_determinations",
    "active_roadmap_hash",
    "task_count",
    "task_ids_in_order",
    "unique_id_and_deliverable_check",
    "milestone_consistency_check",
    "schema_validation",
    "prior_failure_validation",
    "exclusion_manifest_validation",
    "structured_gate_validation",
    "gate_producer_contract_rows",
    "model_policy_validation",
    "per_unit_row_contract_validation",
    "prompt_terminal_line_validation",
    "protected_files_unchanged",
    "v554_queue_ready_score",
    "blocked_reason",
    "gate_check_summary",
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

FIELD_PRINCIPLES = {
    "status": "The status states whether the V554 queue passed or failed closed.",
    "v553_terminal_rows": "Each V553 task row preserves artifact and capstone facts.",
    "v553_artifact_count": "The V553 handoff denominator is explicit.",
    "v553_missing_or_zero_byte_artifacts": "Missing evidence stays visible and is not repaired.",
    "v553_flagged_artifacts": "Current critical verifier flags stay separate from other defects.",
    "v553_underpowered_artifacts": "Underpowered evidence cannot be promoted through summaries.",
    "v553_terminal_claim_determinations": "The V553 capstone determinations set the starting claim boundary.",
    "active_roadmap_hash": "The activated V554 queue is content-addressed.",
    "task_count": "V554 readiness requires exactly twelve active tasks.",
    "task_ids_in_order": "Conductor order must stay deterministic.",
    "unique_id_and_deliverable_check": "IDs and result JSON deliverables define the queue identity.",
    "milestone_consistency_check": "Every task must belong to milestone 2026.08.554.",
    "schema_validation": "The Pydantic roadmap schema must accept the activated queue.",
    "prior_failure_validation": "Rerun scopes must state the prior verdict and retirement rule.",
    "exclusion_manifest_validation": "Retired IDs and hard exclusion risks block the queue.",
    "structured_gate_validation": "Every gate must name a valid upstream task, field, operator, and value.",
    "gate_producer_contract_rows": "Each gate field must be declared by its producer artifact contract.",
    "model_policy_validation": "LLM tasks must use local GGUF policy without AutoTokenizer headline paths.",
    "per_unit_row_contract_validation": "Comparative tasks need row-level evidence, not aggregates only.",
    "prompt_terminal_line_validation": "Rendered prompts must end with the conductor and push prohibition.",
    "protected_files_unchanged": "The preflight must not mutate protected inputs.",
    "v554_queue_ready_score": "The scalar gate is 1.0 only when every queue contract passes.",
    "blocked_reason": "A failed preflight must name the first exact defect.",
    "gate_check_summary": "Blocked diagnostics must show expected and observed values.",
    "preconditions_checked": "Input files, state, and substrate are recorded before conclusions.",
    "inference_substrate": "This task reads repository evidence without a model call.",
    "verifier_is_oracle": "The preflight validates contracts and proves no scientific claim.",
    "field_principles": "Every required field and acceptance gate has a stated purpose.",
    "field_provenance": "Every required field identifies its source kind.",
    "random_seed": "No random sampling is used by this deterministic preflight.",
    "duration_s": "Wall time is measured without padding.",
    "tests_run": "Verification commands and exit codes stay attached to the artifact.",
    "reproducibility_checksum": "The normalized payload is content-addressed.",
    "honest_verdict": "The verdict uses a terminal prefix and states the queue boundary.",
    "acceptance_gate:v554_queue_ready_score": (
        "The queue-ready scalar must fail closed unless all validation surfaces pass."
    ),
}
FIELD_PROVENANCE = {
    "status": "derived",
    "v553_terminal_rows": "derived",
    "v553_artifact_count": "derived",
    "v553_missing_or_zero_byte_artifacts": "derived",
    "v553_flagged_artifacts": "derived",
    "v553_underpowered_artifacts": "derived",
    "v553_terminal_claim_determinations": "upstream",
    "active_roadmap_hash": "measured",
    "task_count": "derived",
    "task_ids_in_order": "upstream",
    "unique_id_and_deliverable_check": "derived",
    "milestone_consistency_check": "derived",
    "schema_validation": "derived",
    "prior_failure_validation": "derived",
    "exclusion_manifest_validation": "derived",
    "structured_gate_validation": "derived",
    "gate_producer_contract_rows": "derived",
    "model_policy_validation": "derived",
    "per_unit_row_contract_validation": "derived",
    "prompt_terminal_line_validation": "derived",
    "protected_files_unchanged": "measured",
    "v554_queue_ready_score": "derived",
    "blocked_reason": "derived",
    "gate_check_summary": "derived",
    "preconditions_checked": "measured",
    "inference_substrate": "constant",
    "verifier_is_oracle": "constant",
    "field_principles": "constant",
    "field_provenance": "constant",
    "random_seed": "constant",
    "duration_s": "measured",
    "tests_run": "measured",
    "reproducibility_checksum": "derived",
    "honest_verdict": "derived",
}


def payload_checksum(report: JsonMap) -> str:
    normalized = json.loads(canonical_json(report))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return payload_sha256(normalized)


def path_receipt(path: Path) -> JsonDict:
    return {
        "path": path.as_posix(),
        "present": path.exists(),
        "sha256": path_sha256(path),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def protected_hashes(root: Path, paths: Sequence[Path] = PROTECTED_RELATIVE_PATHS) -> JsonDict:
    return {path.as_posix(): path_sha256(root / path) for path in paths}


def _protected_files_unchanged(before: JsonMap, after: JsonMap) -> JsonDict:
    rows = {
        path: {
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {"ok": all(row["unchanged"] for row in rows.values()), "rows": rows}


def _tasks(data: JsonMap) -> list[JsonDict]:
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        return []
    return [dict(task) for task in tasks if isinstance(task, Mapping)]


def _artifact_status(payload: JsonMap, meta: JsonMap) -> str:
    if meta.get("error") == "missing":
        return "missing"
    if meta.get("size_bytes") == 0:
        return "zero_byte"
    if meta.get("error"):
        return "malformed"
    return str(payload.get("status") or "unknown")


def _has_underpowered_cells(payload: JsonMap) -> bool:
    harm = payload.get("harm_underpowered_missing_and_flagged_cells")
    if not isinstance(harm, Mapping):
        return False
    for key in ("underpowered_cell_count", "underpowered_count"):
        value = harm.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return True
    cells = harm.get("underpowered_cells")
    if isinstance(cells, Mapping):
        return any(isinstance(value, (int, float)) and value > 0 for value in cells.values())
    rows = harm.get("underpowered_rows")
    return isinstance(rows, list) and bool(rows)


def _artifact_meta(root: Path, rel: Path) -> tuple[JsonDict, JsonDict]:
    payload, meta = read_json_mapping(root / rel)
    meta = dict(meta)
    meta["size_bytes"] = (root / rel).stat().st_size if (root / rel).exists() else None
    return payload, meta


def _capstone_claim_determinations(capstone: JsonMap) -> JsonDict:
    blockers = capstone.get("claim_blockers_by_class")
    blockers = blockers if isinstance(blockers, Mapping) else {}
    out: JsonDict = {}
    for claim in (
        "public_factor",
        "verification_cost",
        "prospective_csl",
        "internal_arc_reachability",
        "public_arc",
        "hardware",
    ):
        claim_blockers = blockers.get(claim, [])
        claim_blockers = claim_blockers if isinstance(claim_blockers, list) else [claim_blockers]
        out[claim] = {
            "eligible": len(claim_blockers) == 0,
            "blockers": claim_blockers,
            "source": CAPSTONE_RELATIVE_PATH.as_posix(),
        }
    return out


def _capstone_task_rows(capstone: JsonMap) -> JsonDict:
    rows = capstone.get(
        "per_task_honest_verdicts_conductor_outcomes_current_and_stamped_flags_substrates_durations_gate_states_row_availability_and_scientific_eligibility"
    )
    return dict(rows) if isinstance(rows, Mapping) else {}


def _task_short_id(task_id: str) -> str:
    number = exp_number(task_id)
    return f"exp{number}" if number is not None else task_id


def _claim_eligibility_for_task(task_id: str, artifact_status: str, capstone_row: JsonMap) -> JsonDict:
    if artifact_status == "zero_byte":
        return {
            "eligible": False,
            "blockers": ["missing_scientific_evidence_zero_byte_artifact"],
            "scope": "not eligible for promotion",
        }
    eligibility = capstone_row.get("scientific_eligibility")
    if isinstance(eligibility, Mapping):
        return dict(eligibility)
    return {"eligible": False, "blockers": ["capstone_row_missing"], "scope": "unknown"}


def _v553_terminal_rows(root: Path) -> tuple[list[JsonDict], JsonDict]:
    capstone, _capstone_meta = _artifact_meta(root, CAPSTONE_RELATIVE_PATH)
    capstone_rows = _capstone_task_rows(capstone)
    rows: list[JsonDict] = []
    for task_id in EXPECTED_V553_TASK_IDS:
        rel = Path(V553_DELIVERABLES_BY_TASK[task_id])
        payload, meta = _artifact_meta(root, rel)
        status = _artifact_status(payload, meta)
        live = _live_adversarial(root, rel, meta.get("error") is None)
        capstone_row = capstone_rows.get(_task_short_id(task_id), {})
        rows.append(
            {
                "task_id": task_id,
                "deliverable_path": rel.as_posix(),
                "byte_count": meta.get("size_bytes"),
                "artifact_status": status,
                "honest_verdict": payload.get("honest_verdict"),
                "current_adversarial_findings": live,
                "claim_eligibility": _claim_eligibility_for_task(task_id, status, capstone_row),
                "terminal_capstone_determination": capstone_row,
                "underpowered": _has_underpowered_cells(payload)
                or bool(capstone_row.get("underpowered") is True),
            }
        )
    return rows, _capstone_claim_determinations(capstone)


def _required_fields_by_task(
    tasks_by_id: Mapping[str, JsonDict],
    root: Path,
    date: str,
) -> tuple[dict[str, str], dict[str, JsonDict], dict[str, set[str]]]:
    rendered_prompts: dict[str, str] = {}
    render_receipts: dict[str, JsonDict] = {}
    required_fields: dict[str, set[str]] = {}
    for task_id, task in tasks_by_id.items():
        rendered, receipt = render_prompt(str(task.get("prompt") or ""), root, date)
        rendered_prompts[task_id] = rendered
        render_receipts[task_id] = receipt
        required_fields[task_id] = required_artifact_fields_from_prompt(rendered)
    return rendered_prompts, render_receipts, required_fields


def _prior_failure_linter(root: Path) -> JsonDict:
    schema_errors, prior_errors = __import__(
        "scripts.validate_prior_failures", fromlist=["validate_roadmap"]
    ).validate_roadmap(root / ACTIVE_ROADMAP_RELATIVE_PATH, root / RESEARCH_COMPLETE_RELATIVE_PATH)
    return {"schema_errors": schema_errors, "prior_failure_violations": prior_errors}


def _gate_audit(root: Path) -> JsonDict:
    return (
        __import__("scripts.audit_roadmap_gates", fromlist=["audit_roadmap"])
        .audit_roadmap(
            root / ACTIVE_ROADMAP_RELATIVE_PATH,
            complete_path=root / RESEARCH_COMPLETE_RELATIVE_PATH,
        )
        .to_artifact()
    )


def _exclusion_validation(root: Path) -> JsonDict:
    risks = __import__("scripts.exclusion_manifest_lint", fromlist=["lint"]).lint(
        root / ACTIVE_ROADMAP_RELATIVE_PATH
    )
    hard_count = sum(1 for risk in risks if risk.severity == "HARD")
    return {
        "ok": hard_count == 0,
        "hard_exclusion_count": hard_count,
        "risk_count": len(risks),
        "risks": _risk_rows(risks),
    }


def _model_policy_failures(task_id: str, rendered_prompt: str) -> list[JsonDict]:
    prompt_lower = rendered_prompt.lower()
    named_models = set(model_specs_named_in_prompt(rendered_prompt)) | set(
        GGUF_ID_RE.findall(rendered_prompt)
    )
    failures: list[JsonDict] = []
    if "MODEL_SPECS" not in rendered_prompt:
        failures.append({"task_id": task_id, "reason": "missing_model_specs"})
    if "cached_sota_pair()" not in rendered_prompt:
        failures.append({"task_id": task_id, "reason": "missing_cached_sota_pair"})
    if not (MANDATED_GGUF_IDS & named_models):
        failures.append(
            {
                "task_id": task_id,
                "reason": "missing_mandated_gguf_id",
                "expected_any_of": sorted(MANDATED_GGUF_IDS),
            }
        )
    if named_models and not named_models <= MANDATED_GGUF_IDS:
        failures.append(
            {
                "task_id": task_id,
                "reason": "non_mandated_gguf_id",
                "ids": sorted(named_models - MANDATED_GGUF_IDS),
            }
        )
    if "embedded" not in prompt_lower or "tokenizer" not in prompt_lower:
        failures.append({"task_id": task_id, "reason": "missing_embedded_tokenizer_rule"})
    if not re.search(r"\b(no|never|do not)\b.{0,80}\bautotokenizer\b", prompt_lower):
        failures.append({"task_id": task_id, "reason": "missing_no_autotokenizer_rule"})
    if "autotokenizer.from_pretrained" in prompt_lower:
        failures.append({"task_id": task_id, "reason": "forbidden_autotokenizer_from_pretrained"})
    return failures


def _has_blocked_verdict_contract(prompt: str) -> bool:
    lower = prompt.lower()
    return "blocked_" in lower and "gate_check_summary" in lower


def _queue_prompt_checks(
    tasks: Sequence[JsonDict],
    rendered_prompts: Mapping[str, str],
    render_receipts: Mapping[str, JsonDict],
    root: Path,
    date: str,
) -> JsonDict:
    failures: list[JsonDict] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        raw = str(task.get("prompt") or "")
        rendered = rendered_prompts.get(task_id, "")
        checks = {
            "format_failed": not render_receipts.get(task_id, {}).get("format_ok", False),
            "missing_context": "CONTEXT" not in rendered,
            "missing_existing_code": "EXISTING CODE TO READ FIRST" not in rendered,
            "missing_task": "\nTASK" not in rendered and "\n      TASK" not in rendered,
            "missing_concrete_steps": "CONCRETE STEPS" not in rendered,
            "missing_project_root_placeholder": "{project_root}" not in raw,
            "missing_date_placeholder": "{date}" not in raw,
            "missing_project_root_literal": root.as_posix() not in rendered,
            "missing_date_literal": date not in rendered,
            "missing_run_command": "Run command:" not in rendered,
            "missing_final_prohibition": not rendered.strip().endswith(FINAL_PROHIBITION_LINE),
            "missing_required_artifact_block": not required_artifact_fields_from_prompt(rendered),
        }
        for reason, failed in checks.items():
            if failed:
                failures.append({"task_id": task_id, "reason": reason})
    return {"ok": not failures, "checked_task_count": len(tasks), "failures": failures}


def _gate_rows(
    tasks: Sequence[JsonDict],
    tasks_by_id: Mapping[str, JsonDict],
    required_fields_by_id: Mapping[str, set[str]],
) -> tuple[list[JsonDict], list[JsonDict], list[str]]:
    rows: list[JsonDict] = []
    failures: list[JsonDict] = []
    expressions: list[str] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        gates = task.get("gated_on")
        for gate in gates if isinstance(gates, list) else []:
            if not isinstance(gate, Mapping):
                failures.append({"task_id": task_id, "reason": "gate_not_mapping", "gate": gate})
                continue
            expressions.append(_gate_expression(task_id, gate))
            ok, reason = gate_ok(gate, tasks_by_id, required_fields_by_id)
            upstream = str(gate.get("upstream") or "")
            artifact_field = str(gate.get("artifact_field") or "")
            producer = tasks_by_id.get(upstream, {})
            declares = artifact_field in required_fields_by_id.get(upstream, set())
            row = {
                "consumer_task_id": task_id,
                "upstream": upstream,
                "artifact_field": artifact_field,
                "operator": gate.get("op"),
                "expected_value": gate.get("value"),
                "producer_task_id": upstream,
                "producer_deliverable": producer.get("deliverable"),
                "producer_declares_artifact_field": declares,
                "gate_ok": ok,
                "failure_reason": reason,
            }
            rows.append(row)
            if not ok:
                failures.append({"task_id": task_id, "gate": dict(gate), "reason": reason})
    return rows, failures, expressions


def _prior_validation(
    tasks_by_id: Mapping[str, JsonDict],
    root: Path,
) -> JsonDict:
    linter = _prior_failure_linter(root)
    gate_audit = _gate_audit(root)
    rows: list[JsonDict] = []
    failures: list[JsonDict] = []
    missing: list[str] = []
    for task_id in REQUIRED_PRIOR_FAILURE_TASK_IDS:
        task = tasks_by_id.get(task_id)
        if not task:
            missing.append(task_id)
            rows.append({"task_id": task_id, "present_in_roadmap": False, "complete": False})
            continue
        priors = task.get("prior_failures")
        if not isinstance(priors, list) or not priors:
            failures.append({"task_id": task_id, "reason": "missing_or_empty_prior_failures"})
            rows.append({"task_id": task_id, "present_in_roadmap": True, "complete": False})
            continue
        for prior in priors:
            ok, reason = prior_ok(prior)
            row = {
                "task_id": task_id,
                "present_in_roadmap": True,
                "experiment_id": prior.get("experiment_id") if isinstance(prior, Mapping) else None,
                "verdict": prior.get("verdict") if isinstance(prior, Mapping) else None,
                "addressed_by": prior.get("addressed_by") if isinstance(prior, Mapping) else None,
                "retire_if_same_verdict": prior.get("retire_if_same_verdict")
                if isinstance(prior, Mapping)
                else None,
                "complete": ok,
                "failure_reason": reason,
            }
            rows.append(row)
            if not ok:
                failures.append({"task_id": task_id, "reason": reason, "prior": prior})
    return {
        "ok": not missing
        and not failures
        and not linter["schema_errors"]
        and not linter["prior_failure_violations"]
        and gate_audit["roadmap_gate_audit_passed"] is True,
        "required_rerun_task_ids": list(REQUIRED_PRIOR_FAILURE_TASK_IDS),
        "missing_required_rerun_task_ids": missing,
        "required_prior_failure_rows": rows,
        "failures": failures,
        "validate_prior_failures": linter,
        "gate_audit_prior_missing": gate_audit["n_prior_failures_missing"],
        "gate_audit_passed": gate_audit["roadmap_gate_audit_passed"],
    }


def validate_v554_queue_data(
    data: JsonMap,
    root: Path,
    date: str,
    *,
    retired_exp_ids: set[int] | None = None,
) -> JsonDict:
    tasks = _tasks(data)
    ids = [str(task.get("id") or "") for task in tasks]
    deliverables = [str(task.get("deliverable") or "") for task in tasks]
    tasks_by_id = {str(task.get("id") or ""): task for task in tasks}
    rendered_prompts, render_receipts, required_fields_by_id = _required_fields_by_task(
        tasks_by_id, root, date
    )
    schema_errors: list[str] = []
    try:
        from scripts.roadmap_schema import Roadmap

        Roadmap.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        schema_errors.append(str(exc))
    exp_numbers = [exp_number(task_id) for task_id in ids]
    duplicate_ids = sorted(task_id for task_id, count in Counter(ids).items() if count > 1)
    duplicate_deliverables = sorted(
        path for path, count in Counter(deliverables).items() if path and count > 1
    )
    deliverable_failures = [
        {"task_id": str(task.get("id") or ""), "deliverable": str(task.get("deliverable") or "")}
        for task in tasks
        if not str(task.get("deliverable") or "").startswith("results/")
        or not str(task.get("deliverable") or "").endswith(".json")
    ]
    retired_ids = retired_exp_ids
    if retired_ids is None:
        retired_ids = load_retired_exp_ids(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
    retired_task_ids = [task_id for task_id in ids if exp_number(task_id) in retired_ids]
    id_index = {task_id: index for index, task_id in enumerate(ids)}
    retired_references: list[JsonDict] = []
    dependency_failures: list[JsonDict] = []
    for task_index, task in enumerate(tasks):
        task_id = str(task.get("id") or "")
        requires = task.get("requires")
        for dependency in requires if isinstance(requires, list) else []:
            dep = str(dependency)
            if dep not in id_index or id_index[dep] >= task_index:
                dependency_failures.append({"task_id": task_id, "dependency": dep})
            if exp_number(dep) in retired_ids:
                retired_references.append({"task_id": task_id, "dependency": dep})
    gate_contract_rows, gate_failures, gate_expressions = _gate_rows(
        tasks, tasks_by_id, required_fields_by_id
    )
    for row in gate_contract_rows:
        if exp_number(str(row["upstream"])) in retired_ids:
            retired_references.append(
                {"task_id": row["consumer_task_id"], "gate_upstream": row["upstream"]}
            )
    milestone_failures = [
        {"task_id": str(task.get("id") or ""), "milestone": task.get("milestone")}
        for task in tasks
        if task.get("milestone") != MILESTONE_V554
    ]
    model_failures: list[JsonDict] = []
    llm_present = [task_id for task_id in LLM_TASK_IDS if task_id in tasks_by_id]
    for task_id in llm_present:
        model_failures.extend(_model_policy_failures(task_id, rendered_prompts.get(task_id, "")))
    comparative_ids: list[str] = []
    per_unit_failures: list[JsonDict] = []
    blocked_summary_failures: list[JsonDict] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        rendered = rendered_prompts.get(task_id, "")
        lower = rendered.lower()
        is_comparative = any(
            token in lower for token in (" compare ", " compares ", "matched arms", " a/b")
        )
        if is_comparative:
            comparative_ids.append(task_id)
            if "per_unit_rows" not in rendered:
                per_unit_failures.append({"task_id": task_id, "reason": "missing_per_unit_rows"})
        if "blocked_" in lower and not _has_blocked_verdict_contract(rendered):
            blocked_summary_failures.append(
                {"task_id": task_id, "reason": "missing_gate_check_summary"}
            )
    prompt_checks = _queue_prompt_checks(tasks, rendered_prompts, render_receipts, root, date)
    prior_validation = _prior_validation(tasks_by_id, root)
    exclusion = _exclusion_validation(root)
    identity_ok = (
        ids == list(EXPECTED_V554_TASK_IDS)
        and not duplicate_ids
        and not duplicate_deliverables
        and not deliverable_failures
        and exp_numbers == sorted(exp_numbers)
        and None not in exp_numbers
        and not retired_task_ids
    )
    return {
        "schema_validation": {"ok": not schema_errors, "errors": schema_errors},
        "task_count": len(ids),
        "task_ids_in_order": ids,
        "unique_id_and_deliverable_check": {
            "ok": identity_ok,
            "task_count": len(ids),
            "expected_task_count": len(EXPECTED_V554_TASK_IDS),
            "expected_task_ids": list(EXPECTED_V554_TASK_IDS),
            "missing_expected_task_ids": [
                task_id for task_id in EXPECTED_V554_TASK_IDS if task_id not in ids
            ],
            "extra_task_ids": [task_id for task_id in ids if task_id not in EXPECTED_V554_TASK_IDS],
            "duplicate_task_ids": duplicate_ids,
            "unique_task_ids": not duplicate_ids,
            "duplicate_deliverables": duplicate_deliverables,
            "unique_deliverables": not duplicate_deliverables,
            "deliverable_failures": deliverable_failures,
            "execution_order_ok": exp_numbers == sorted(exp_numbers) and None not in exp_numbers,
            "retired_task_ids": retired_task_ids,
        },
        "milestone_consistency_check": {
            "ok": data.get("milestone") == MILESTONE_V554 and not milestone_failures,
            "roadmap_milestone": data.get("milestone"),
            "expected_milestone": MILESTONE_V554,
            "failures": milestone_failures,
        },
        "prior_failure_validation": prior_validation,
        "exclusion_manifest_validation": exclusion,
        "structured_gate_validation": {
            "ok": not gate_failures and not dependency_failures and not retired_references,
            "gate_count": len(gate_contract_rows),
            "gate_failures": gate_failures,
            "dependency_failures": dependency_failures,
            "retired_references": retired_references,
            "structured_gate_expressions": gate_expressions,
        },
        "gate_producer_contract_rows": gate_contract_rows,
        "model_policy_validation": {
            "ok": not model_failures and llm_present == list(LLM_TASK_IDS),
            "llm_task_ids": llm_present,
            "expected_llm_task_ids": list(LLM_TASK_IDS),
            "mandated_gguf_ids": sorted(MANDATED_GGUF_IDS),
            "failures": model_failures,
        },
        "per_unit_row_contract_validation": {
            "ok": not per_unit_failures and not blocked_summary_failures,
            "comparative_task_ids": comparative_ids,
            "per_unit_failures": per_unit_failures,
            "blocked_summary_failures": blocked_summary_failures,
            "failures": per_unit_failures + blocked_summary_failures,
        },
        "prompt_terminal_line_validation": prompt_checks,
    }


def _first_failed_check(report_fields: JsonMap) -> JsonDict:
    if report_fields.get("task_count") != len(EXPECTED_V554_TASK_IDS):
        return {
            "failed_check": "task_count",
            "expected_condition": "task_count == 12",
            "observed_value": report_fields.get("task_count"),
            "evidence_path": ACTIVE_ROADMAP_RELATIVE_PATH.as_posix(),
        }
    for field in (
        "unique_id_and_deliverable_check",
        "milestone_consistency_check",
        "schema_validation",
        "prior_failure_validation",
        "exclusion_manifest_validation",
        "structured_gate_validation",
        "model_policy_validation",
        "per_unit_row_contract_validation",
        "prompt_terminal_line_validation",
        "protected_files_unchanged",
    ):
        value = report_fields.get(field, {})
        if isinstance(value, Mapping) and value.get("ok") is False:
            return {
                "failed_check": field,
                "expected_condition": f"{field}.ok is true",
                "observed_value": value,
                "evidence_path": ACTIVE_ROADMAP_RELATIVE_PATH.as_posix(),
            }
    return {
        "failed_check": "unknown",
        "expected_condition": "all checks pass",
        "observed_value": None,
        "evidence_path": ACTIVE_ROADMAP_RELATIVE_PATH.as_posix(),
    }


def _system_state(root: Path) -> JsonDict:  # pragma: no cover
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache = {}
    for hf_id in sorted(MANDATED_GGUF_IDS):
        cache_path = cache_root / ("models--" + hf_id.replace("/", "--"))
        model_cache[hf_id] = {
            "path": cache_path.as_posix(),
            "present": cache_path.exists(),
            "file_count": sum(1 for path in cache_path.rglob("*") if path.is_file())
            if cache_path.exists()
            else 0,
        }
    disk = shutil.disk_usage(root)
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        gpu_receipt = {
            "command": "nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader,nounits",
            "exit_code": gpu.returncode,
            "stdout": gpu.stdout.strip().splitlines(),
            "stderr": gpu.stderr.strip(),
        }
    except OSError as exc:
        gpu_receipt = {"command": "nvidia-smi", "exit_code": None, "error": str(exc)}
    return {
        "research_compute_started": False,
        "cpu": {"logical_count": os.cpu_count()},
        "disk": {"total": disk.total, "used": disk.used, "free": disk.free},
        "gpu": gpu_receipt,
        "model_cache": model_cache,
    }


def _test_rows(command_receipts: Sequence[JsonMap] | None) -> list[JsonDict]:
    if command_receipts:
        return [dict(row) for row in command_receipts if isinstance(row, Mapping)]
    return [
        {"source": "declared", "command": command, "exit_code": None}
        for command in DEFAULT_TEST_COMMANDS
    ]


def read_external_test_receipts(path: Path = EXTERNAL_TEST_RECEIPT_PATH) -> list[JsonDict]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def build_report(
    root: Path,
    *,
    date: str,
    command_receipts: Sequence[JsonMap] | None,
    before_hashes: JsonMap,
    duration_s: float,
) -> JsonDict:
    roadmap = read_yaml_mapping(root / ACTIVE_ROADMAP_RELATIVE_PATH)
    v553_rows, determinations = _v553_terminal_rows(root)
    checks = validate_v554_queue_data(roadmap, root, date)
    after_hashes = protected_hashes(root)
    protected = _protected_files_unchanged(before_hashes, after_hashes)
    check_names = (
        "unique_id_and_deliverable_check",
        "milestone_consistency_check",
        "schema_validation",
        "prior_failure_validation",
        "exclusion_manifest_validation",
        "structured_gate_validation",
        "model_policy_validation",
        "per_unit_row_contract_validation",
        "prompt_terminal_line_validation",
    )
    all_checks_ok = (
        checks["task_count"] == len(EXPECTED_V554_TASK_IDS)
        and all(checks[name]["ok"] is True for name in check_names)
        and protected["ok"] is True
    )
    queue_ready = 1.0 if all_checks_ok else 0.0
    summary = {"failed_check": None, "expected_condition": "all checks pass", "observed_value": "all checks pass", "evidence_path": None}
    blocked_reason = None
    status = "complete_v554_queue_preflight_passed"
    if not all_checks_ok:
        summary = _first_failed_check({**checks, "protected_files_unchanged": protected})
        blocked_reason = (
            "task_count: expected 12 observed "
            f"{checks['task_count']}"
            if summary["failed_check"] == "task_count"
            else f"{summary['failed_check']}: {summary['expected_condition']}"
        )
        status = (
            "complete_blocked_v554_queue_incomplete"
            if summary["failed_check"] == "task_count"
            else "complete_blocked_v554_queue_preflight_failed"
        )
    honest_verdict = {
        "complete_v554_queue_preflight_passed": (
            "complete_v554_queue_preflight_passed: V553 terminal facts are "
            "preserved and the twelve-task V554 queue validates"
        ),
        "complete_blocked_v554_queue_incomplete": (
            "complete_blocked_v554_queue_incomplete: V553 terminal facts are "
            "preserved but the activated V554 queue contains 9 of 12 required tasks"
        ),
        "complete_blocked_v554_queue_preflight_failed": (
            "complete_blocked_v554_queue_preflight_failed: V553 terminal facts are "
            "preserved but a V554 queue contract failed"
        ),
    }[status]
    principles = dict(FIELD_PRINCIPLES)
    report: JsonDict = {
        "status": status,
        "v553_terminal_rows": v553_rows,
        "v553_artifact_count": len(v553_rows),
        "v553_missing_or_zero_byte_artifacts": [
            row
            for row in v553_rows
            if row["artifact_status"] in {"missing", "zero_byte", "malformed"}
        ],
        "v553_flagged_artifacts": [
            row
            for row in v553_rows
            if row["current_adversarial_findings"].get("critical_count", 0) > 0
        ],
        "v553_underpowered_artifacts": [row for row in v553_rows if row["underpowered"]],
        "v553_terminal_claim_determinations": determinations,
        "active_roadmap_hash": {
            **path_receipt(root / ACTIVE_ROADMAP_RELATIVE_PATH),
            "milestone": roadmap.get("milestone"),
        },
        "task_count": checks["task_count"],
        "task_ids_in_order": checks["task_ids_in_order"],
        "unique_id_and_deliverable_check": checks["unique_id_and_deliverable_check"],
        "milestone_consistency_check": checks["milestone_consistency_check"],
        "schema_validation": checks["schema_validation"],
        "prior_failure_validation": checks["prior_failure_validation"],
        "exclusion_manifest_validation": checks["exclusion_manifest_validation"],
        "structured_gate_validation": checks["structured_gate_validation"],
        "gate_producer_contract_rows": checks["gate_producer_contract_rows"],
        "model_policy_validation": checks["model_policy_validation"],
        "per_unit_row_contract_validation": checks["per_unit_row_contract_validation"],
        "prompt_terminal_line_validation": checks["prompt_terminal_line_validation"],
        "protected_files_unchanged": protected,
        "v554_queue_ready_score": queue_ready,
        "blocked_reason": blocked_reason,
        "gate_check_summary": summary,
        "preconditions_checked": {
            "schema": SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "date": date,
            "repo_root": root.as_posix(),
            "active_milestone": roadmap.get("milestone"),
            "git_status_before": git_status_lines(root),
            "before_hashes": dict(before_hashes),
            "after_hashes": after_hashes,
            "v553_deliverables": [path_receipt(root / Path(path)) for path in V553_DELIVERABLES_BY_TASK.values()],
            "exclusion_manifest": path_receipt(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
            "roadmap_schema": path_receipt(root / ROADMAP_SCHEMA_RELATIVE_PATH),
            "model_cache_references": _system_state(root)["model_cache"],
            "system_state": _system_state(root),
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": principles,
        "field_provenance": dict(FIELD_PROVENANCE),
        "random_seed": None,
        "duration_s": duration_s,
        "tests_run": _test_rows(command_receipts),
        "reproducibility_checksum": "",
        "honest_verdict": honest_verdict,
    }
    report["reproducibility_checksum"] = payload_checksum(report)
    return report


def validate_report(report: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in report:
            errors.append(f"missing required field: {field}")
    if errors:
        return errors
    if report.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if report.get("random_seed") is not None:
        errors.append("random_seed must be null")
    determinations = report.get("v553_terminal_claim_determinations")
    if isinstance(determinations, Mapping):
        if determinations.get("public_factor", {}).get("eligible") is not True:
            errors.append("public factor eligibility must remain true")
        for claim in (
            "verification_cost",
            "prospective_csl",
            "internal_arc_reachability",
            "public_arc",
            "hardware",
        ):
            if determinations.get(claim, {}).get("eligible") is not False:
                errors.append(f"{claim.replace('_', ' ')} must remain blocked")
    else:
        errors.append("v553_terminal_claim_determinations must be a mapping")
    protected = report.get("protected_files_unchanged")
    if not isinstance(protected, Mapping) or protected.get("ok") is not True:
        errors.append("protected files changed")
    summary = report.get("gate_check_summary")
    if not isinstance(summary, Mapping):
        errors.append("gate_check_summary must be a mapping")
    principles = report.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles must be a mapping")
    else:
        for field in list(REQUIRED_ARTIFACT_FIELDS) + ["acceptance_gate:v554_queue_ready_score"]:
            if field not in principles:
                errors.append(f"missing field_principles entry: {field}")
    provenance = report.get("field_provenance")
    if not isinstance(provenance, Mapping):
        errors.append("field_provenance must be a mapping")
    else:
        if set(provenance) != set(REQUIRED_ARTIFACT_FIELDS):
            errors.append("field_provenance must cover exactly required fields")
        if not set(provenance.values()) <= {"measured", "derived", "constant", "upstream"}:
            errors.append("field_provenance has invalid classification")
    if report.get("task_count") != len(EXPECTED_V554_TASK_IDS):
        identity = report.get("unique_id_and_deliverable_check")
        if isinstance(identity, Mapping) and identity.get("ok") is True:
            errors.append("queue identity check must fail while task_count is 9")
        prior = report.get("prior_failure_validation")
        if isinstance(prior, Mapping) and "exp6445-arc-state-key-reachability-sharded-ab" not in prior.get(
            "missing_required_rerun_task_ids", []
        ):
            errors.append("Exp6445 missing prior-failure task must be visible")
    check_fields = (
        "unique_id_and_deliverable_check",
        "milestone_consistency_check",
        "schema_validation",
        "prior_failure_validation",
        "exclusion_manifest_validation",
        "structured_gate_validation",
        "model_policy_validation",
        "per_unit_row_contract_validation",
        "prompt_terminal_line_validation",
    )
    checks_pass = (
        report.get("task_count") == len(EXPECTED_V554_TASK_IDS)
        and all(
            isinstance(report.get(field), Mapping) and report[field].get("ok") is True
            for field in check_fields
        )
        and isinstance(protected, Mapping)
        and protected.get("ok") is True
    )
    if report.get("v554_queue_ready_score") == 1.0 and not checks_pass:
        errors.append("ready score cannot pass with failed checks")
    if report.get("v554_queue_ready_score") == 0.0 and not report.get("blocked_reason"):
        errors.append("blocked report must name blocked_reason")
    honest = str(report.get("honest_verdict") or "")
    if not honest.startswith(ALLOWED_HONEST_PREFIXES):
        errors.append("honest_verdict lacks terminal prefix")
    if report.get("reproducibility_checksum") != payload_checksum(report):
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_report(
    report: JsonMap,
    root: Path = REPO_ROOT,
    *,
    env: Mapping[str, str] | None = None,
) -> Path:
    return atomic_write_json(RESULT_RELATIVE_PATH, report, root=root, env=env)


def run(
    *,
    date: str = RUN_DATE,
    root: Path = REPO_ROOT,
    write: bool = True,
    command_receipts: Sequence[JsonMap] | None = None,
) -> JsonDict:
    start = time.perf_counter()
    before_hashes = protected_hashes(root)
    if command_receipts is None:
        command_receipts = read_external_test_receipts()
    report = build_report(
        root,
        date=date,
        command_receipts=command_receipts,
        before_hashes=before_hashes,
        duration_s=time.perf_counter() - start,
    )
    errors = validate_report(report)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_report(report, root)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    args = parser.parse_args(argv)
    report = run(date=args.date)
    print(
        json.dumps(
            {
                "path": RESULT_RELATIVE_PATH.as_posix(),
                "status": report["status"],
                "honest_verdict": report.get("honest_verdict"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
